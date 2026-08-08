"""Perception pass: sampled frames in, structured hazard candidates out.

This stage is deliberately tuned for recall. It proposes; it does not decide. A
candidate it never raises can never be recovered downstream, whereas a candidate it
raises wrongly gets one more chance to be caught by the verifier. Every prompt choice
below follows from that asymmetry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from . import nim
from .hazards import KEYS, taxonomy_prompt_block
from .video import Chunk

PERCEPTION_PROMPT = """You are reviewing footage from a fixed warehouse safety camera.

The {n} images are sequential frames from a single {window:.0f} second window, in time
order. Read them as motion, not as {n} separate photographs: what changes between
frames is usually the safety-relevant part.

Report any of these hazard classes you observe:
{taxonomy}

Before anything else, decide what you are actually looking at. Safety footage gets
edited into training material, so a window may contain a title card, a slide, an
animation, a presenter against a plain backdrop, or a caption burned over the frame.
Words printed on screen describe the video; they are not events happening in a
warehouse. Never treat on-screen text as something you observed.

Return ONLY a JSON object, no prose before or after:
{{
  "is_camera_footage": <true if these frames show a real work area viewed by a camera;
                        false for title cards, slides, animations, diagrams, or a
                        presenter filmed against a plain background>,
  "frame_content": "one of: footage | slide | animation | presenter | mixed",
  "scene": "one sentence on the setting and camera view",
  "activity": "one sentence on what changes across the frames",
  "people_count": <integer>,
  "truck_present": <true|false>,
  "truck_moving": <true|false|null>,
  "candidates": [
    {{
      "type": "<one of: {keys}>",
      "description": "what specifically is happening, naming the people and equipment",
      "evidence": "cite frame numbers and describe WHERE things are relative to each other: who is on foot, where they stand relative to the truck's front and direction of travel, roughly how many truck-lengths apart, and what changes between the frames",
      "confidence": <0.0 to 1.0>
    }}
  ]
}}

Rules:
- If is_camera_footage is false, return "candidates": []. There is no hazard in a
  slide, however alarming its wording.
- Report what is visible. Do not infer a hazard from the fact that this is a safety
  camera, and do not assume footage must contain a violation.
- If nothing meets a class definition, return "candidates": []. An empty list is a
  valid and common answer.
- confidence is your read of whether the frames actually show it, not how bad it
  would be if true.
"""


@dataclass
class Candidate:
    type: str
    description: str
    evidence: str
    confidence: float


@dataclass
class Perception:
    chunk_id: str
    video_id: str
    start_s: float
    end_s: float
    scene: str = ""
    activity: str = ""
    # Scene gate. Training and marketing footage is full of title cards and slides, and
    # a VLM will happily read the words printed on one and report them as an observed
    # event. Gating on this is the cheapest precision win in the pipeline: it costs no
    # extra call, because the perception pass answers it in the same turn.
    is_camera_footage: bool = True
    frame_content: str = "footage"
    people_count: int = 0
    truck_present: bool = False
    truck_moving: bool | None = None
    candidates: list[Candidate] = field(default_factory=list)
    # Carried through so the verifier can re-open the same frames this pass judged.
    frame_paths: list[str] = field(default_factory=list)
    model: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def caption(self) -> str:
        """The text that gets embedded for natural-language search."""
        parts = [self.scene, self.activity]
        parts += [f"{c.type}: {c.description}" for c in self.candidates]
        return " ".join(p for p in parts if p)


def _coerce(raw: dict[str, Any], chunk: Chunk, model: str) -> Perception:
    """Normalise model output into the dataclass, dropping anything off-taxonomy.

    A VLM asked for an enum will occasionally invent a sixth hazard class. Silently
    keeping it would corrupt the eval, since ground truth has no such class to match
    it against, so unknown types are discarded here rather than at scoring time.
    """
    cands: list[Candidate] = []
    for c in raw.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        ctype = str(c.get("type", "")).strip()
        if ctype not in KEYS:
            continue
        try:
            conf = float(c.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        cands.append(
            Candidate(
                type=ctype,
                description=str(c.get("description", ""))[:600],
                evidence=str(c.get("evidence", ""))[:400],
                confidence=max(0.0, min(1.0, conf)),
            )
        )

    def _b(v: Any) -> bool | None:
        return v if isinstance(v, bool) else None

    try:
        people = int(raw.get("people_count") or 0)
    except (TypeError, ValueError):
        people = 0

    return Perception(
        chunk_id=chunk.chunk_id,
        video_id=chunk.video_id,
        start_s=chunk.start_s,
        end_s=chunk.end_s,
        scene=str(raw.get("scene", ""))[:400],
        activity=str(raw.get("activity", ""))[:400],
        # Absent field defaults to True so a model that ignores the gate degrades to
        # the ungated behaviour rather than silently suppressing every detection.
        is_camera_footage=bool(raw.get("is_camera_footage", True)),
        frame_content=str(raw.get("frame_content", "footage"))[:32],
        people_count=people,
        truck_present=bool(raw.get("truck_present")),
        truck_moving=_b(raw.get("truck_moving")),
        candidates=cands,
        frame_paths=list(chunk.frame_paths),
        model=model,
    )


def perceive_chunk(chunk: Chunk, *, model: str = "") -> Perception:
    """Run one chunk through the VLM and return structured candidates."""
    model = model or nim.VLM_PRIMARY
    prompt = PERCEPTION_PROMPT.format(
        n=len(chunk.frame_paths),
        window=chunk.end_s - chunk.start_s,
        taxonomy=taxonomy_prompt_block(),
        keys=", ".join(KEYS),
    )
    try:
        text = nim.vlm_frames(model, prompt, chunk.frame_paths, max_tokens=900)
        raw = nim.parse_json(text)
        if not isinstance(raw, dict):
            raise ValueError("expected a JSON object")
        return _coerce(raw, chunk, model)
    except Exception as exc:  # noqa: BLE001 - a bad chunk must not kill the run
        # A dropped chunk is a hole in the timeline, so it is recorded as an explicit
        # error rather than as an absence of hazards. The eval counts these separately
        # so a parse failure can never be mistaken for a clean window.
        return Perception(
            chunk_id=chunk.chunk_id,
            video_id=chunk.video_id,
            start_s=chunk.start_s,
            end_s=chunk.end_s,
            frame_paths=list(chunk.frame_paths),
            model=model,
            error=str(exc)[:300],
        )


def perceive_all(
    chunks: list[Chunk], *, model: str = "", workers: int = 6, progress=None
) -> list[Perception]:
    """Perceive every chunk, in parallel, preserving input order.

    Six workers is where throughput stopped improving against the free NIM tier's
    rate limit. Above that the retries cost more than the concurrency buys.
    """
    from concurrent.futures import ThreadPoolExecutor

    results: list[Perception | None] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(perceive_chunk, c, model=model): i for i, c in enumerate(chunks)
        }
        done = 0
        for fut in __import__("concurrent.futures", fromlist=["as_completed"]).as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if progress:
                progress(done, len(chunks))
    return [r for r in results if r is not None]


def save(perceptions: list[Perception], path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([p.as_dict() for p in perceptions], indent=2))


def load(path) -> list[Perception]:
    from pathlib import Path

    raw = json.loads(Path(path).read_text())
    out = []
    for r in raw:
        cands = [Candidate(**c) for c in r.pop("candidates", [])]
        out.append(Perception(**r, candidates=cands))
    return out
