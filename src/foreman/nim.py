"""Thin client for NVIDIA NIM endpoints on integrate.api.nvidia.com.

Everything in Foreman runs against NVIDIA-hosted NIM microservices, so the whole
pipeline is reproducible on a laptop with nothing but an API key. The same model
names map onto self-hosted NIM containers if you later move to your own GPUs, which
is the intended production path.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://integrate.api.nvidia.com/v1"

# Perception. Nemotron Nano VL reads a window of sampled frames in one turn and
# returns structured observations. It is the smallest model that reliably holds the
# JSON contract, which matters because every chunk of every video goes through it.
VLM_PRIMARY = "nvidia/nemotron-nano-12b-v2-vl"

# Second opinion from a smaller sibling. The dual-VLM ablation uses this to measure
# how much of a false positive is one model's hallucination versus a real ambiguity
# in the footage that any model would trip on.
VLM_SECONDARY = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"

# NOTE ON COSMOS: nvidia/cosmos-reason2-8b is the natural perception model here, since
# Cosmos Reason is post-trained for physical-world spatial and temporal reasoning
# rather than generic captioning. It is not provisioned on the free build.nvidia.com
# tier (404s on account-scoped function lookup as of 2026-08), so this pipeline runs
# on Nemotron VL. The client is model-agnostic: pointing VLM_PRIMARY at Cosmos Reason
# on a self-hosted NIM is a one-line change, and the eval harness will then report the
# comparison honestly rather than by assertion.

# Reasoning model that adjudicates candidate hazards on the perception text alone.
# Kept because the eval needs it: it is the arm that shows why text-only review fails.
VERIFIER_LLM = "nvidia/nemotron-3-super-120b-a12b"

# Reasoning VLM that re-opens the frames to adjudicate. Larger and slower than the
# perception model and from a different family, so a confirmation is a genuine second
# look rather than the proposing model agreeing with itself.
VERIFIER_VLM = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

# Vision-language embeddings, so natural-language search runs over the same
# representation space the perception pass produced.
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"


class NIMError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        env = Path(__file__).resolve().parents[2] / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("NVIDIA_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise NIMError(
            "No NVIDIA_API_KEY. Get a free one at https://build.nvidia.com and put it "
            "in .env (see .env.example)."
        )
    return key


@dataclass
class Usage:
    """Token and wall-clock accounting, aggregated so evals can report real cost."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, resp: dict[str, Any], elapsed: float) -> None:
        u = resp.get("usage") or {}
        self.calls += 1
        self.prompt_tokens += u.get("prompt_tokens", 0)
        self.completion_tokens += u.get("completion_tokens", 0)
        self.seconds += elapsed
        self.by_model[model] = self.by_model.get(model, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "seconds": round(self.seconds, 2),
            "by_model": self.by_model,
        }


USAGE = Usage()


def _b64_image(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()


def chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    retries: int = 4,
) -> str:
    """POST to the NIM chat completions endpoint with backoff on 429/5xx."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    last = None
    for attempt in range(retries):
        start = time.time()
        try:
            r = requests.post(
                f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=180
            )
        except requests.RequestException as exc:
            last = exc
            time.sleep(2**attempt)
            continue
        elapsed = time.time() - start
        if r.status_code == 200:
            body = r.json()
            USAGE.add(model, body, elapsed)
            return body["choices"][0]["message"]["content"]
        last = f"{r.status_code}: {r.text[:400]}"
        # Rate limits and transient upstream failures are worth retrying; a 400 is not.
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2**attempt + 1)
            continue
        break
    raise NIMError(f"{model} failed after {retries} attempts -> {last}")


def vlm_frames(
    model: str,
    prompt: str,
    frame_paths: list[str | Path],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Send a prompt plus one or more frames to a vision model.

    NVIDIA's VLM NIMs accept OpenAI-style content parts, so a chunk of video becomes
    an ordered list of sampled frames in a single turn. Keeping the frames in one
    turn is what lets the model reason about motion across them rather than
    describing each still in isolation.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in frame_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_b64_image(p)}"},
            }
        )
    return chat(
        model,
        [{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
    )


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> Any:
    """Recover JSON from model output that may be fenced or prose-wrapped.

    Reasoning models in particular like to narrate before answering, so the raw
    string is rarely valid JSON on the first try. This is deliberately forgiving:
    a dropped chunk is a silent hole in the timeline, which is worse than a
    slightly messy parse.
    """
    text = (text or "").strip()
    # Reasoning models emit a <think> block before the answer.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"could not parse JSON from: {text[:300]}")


def embed(texts: list[str], *, input_type: str = "passage") -> list[list[float]]:
    """Embed text with the vision-language retriever used for video search."""
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    out: list[list[float]] = []
    for i in range(0, len(texts), 32):
        batch = texts[i : i + 32]
        r = requests.post(
            f"{BASE_URL}/embeddings",
            headers=headers,
            json={
                "model": EMBED_MODEL,
                "input": batch,
                "input_type": input_type,
                "encoding_format": "float",
            },
            timeout=120,
        )
        if r.status_code != 200:
            raise NIMError(f"embed failed {r.status_code}: {r.text[:300]}")
        out.extend(d["embedding"] for d in r.json()["data"])
    return out
