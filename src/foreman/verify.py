"""Verifier agent: turns hazard candidates into alerts a human will actually trust.

The perception pass is tuned for recall, which means it over-reports. That is not a
prompt bug to be fixed upstream. A VLM shown a safety camera and asked about hazards
is being handed a strong prior that hazards are present, and suppressing that prior in
the perception prompt costs real detections along with the phantom ones.

So the fix is structural rather than lexical: a second model that only ever removes.
It sees three things the perception pass did not - the evidence standard for the
class, the failure mode that class attracts, and what the adjacent windows saw - and
its single job is to decide whether the frames support the claim. Every arm of this
is measured in `evals/run_eval.py`; none of it is asserted.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from . import nim
from .hazards import BY_KEY, verifier_prompt_block
from .perception import Candidate, Perception

VERIFIER_SYSTEM = """You are a safety review analyst auditing automated hazard \
detections from warehouse camera footage.

A vision model has proposed a hazard. Your job is to decide whether the described \
evidence actually meets the standard for that hazard class. You are the check on a \
system deliberately tuned to over-report, so rejecting a weak detection is a correct \
and expected outcome, not a failure.

You never propose new hazards. You only confirm or reject the one put in front of you.
"""

VERIFIER_PROMPT = """Hazard classes and their evidence standards:
{taxonomy}

CANDIDATE UNDER REVIEW
  class:       {ctype}
  description: {description}
  evidence:    {evidence}
  vision model confidence: {confidence}

WHAT THE VISION MODEL SAW IN THIS WINDOW ({start:.0f}s - {end:.0f}s)
  scene:    {scene}
  activity: {activity}
  people visible: {people}
  powered truck present: {truck_present}
  truck moving: {truck_moving}
{temporal}
DECIDE
Confirm only if the described evidence meets the "evidence required" standard for \
this class. Reject if the evidence is generic, if it restates the class definition \
without describing what is visible, if it matches the known false positive for the \
class, or if a required element of the scene is missing.

Return ONLY this JSON:
{{
  "verdict": "confirmed" | "rejected",
  "severity": "low" | "medium" | "high",
  "rationale": "one or two sentences a supervisor could act on, citing the specific evidence",
  "missing_evidence": "if rejected, the specific thing that would have been needed; else empty string"
}}

severity reflects consequence if the hazard is real: high means a struck-by or \
crushing injury is plausible in this window, low means a policy deviation with no \
immediate exposure.
"""

VLM_VERIFIER_PROMPT = """You are auditing an automated hazard detection against the \
footage it came from.

The {n} images are the exact frames the detection was made on, in time order, covering \
{start:.0f}s to {end:.0f}s. Look at them yourself. Do not take the detection's word for \
what is in them.

Hazard classes and their evidence standards:
{taxonomy}

DETECTION UNDER REVIEW
  class:       {ctype}
  claim:       {description}
  cited as:    {evidence}
  proposing model confidence: {confidence}
{temporal}
DECIDE
Confirm only if you can see, in these frames, the evidence the class requires. Reject \
if what you see does not meet that bar, or if it matches the known false positive for \
the class.

The proposing model is deliberately tuned to over-report, so rejecting is a normal \
outcome. But do not reject a hazard you can plainly see just because the written claim \
was vaguely worded. Judge the frames, not the prose.

Return ONLY this JSON:
{{
  "verdict": "confirmed" | "rejected",
  "severity": "low" | "medium" | "high",
  "rationale": "one or two sentences citing what you actually see in the frames",
  "missing_evidence": "if rejected, what you looked for and did not find; else empty string"
}}

severity reflects consequence if the hazard is real: high means a struck-by or \
crushing injury is plausible in this window, low means a policy deviation with no \
immediate exposure.
"""

TEMPORAL_BLOCK = """
WHAT THE ADJACENT WINDOWS SAW
{lines}
Use this only to judge whether the candidate is consistent with the surrounding \
footage. A truck that is stationary in every neighbouring window is unlikely to be \
bearing down on someone in this one.
"""


@dataclass
class Alert:
    alert_id: str
    chunk_id: str
    video_id: str
    start_s: float
    end_s: float
    type: str
    label: str
    description: str
    evidence: str
    vlm_confidence: float
    verdict: str
    severity: str
    rationale: str
    missing_evidence: str
    standard: str
    verifier_model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _temporal_context(
    perceptions: list[Perception], index: int, span: int = 1
) -> str:
    """Summarise neighbouring windows so the verifier can sanity-check continuity."""
    lines = []
    for j in range(max(0, index - span), min(len(perceptions), index + span + 1)):
        if j == index:
            continue
        p = perceptions[j]
        rel = "before" if j < index else "after"
        cand = ", ".join(c.type for c in p.candidates) or "none"
        lines.append(
            f"  [{rel}, {p.start_s:.0f}s-{p.end_s:.0f}s] {p.activity or p.scene} "
            f"(truck_present={p.truck_present}, moving={p.truck_moving}, "
            f"candidates={cand})"
        )
    if not lines:
        return ""
    return TEMPORAL_BLOCK.format(lines="\n".join(lines))


def verify_candidate(
    cand: Candidate,
    perception: Perception,
    *,
    temporal: str = "",
    model: str = "",
    mode: str = "vlm",
) -> Alert:
    """Adjudicate one candidate.

    mode="vlm" re-opens the frames with a reasoning VLM. mode="llm" reviews only the
    perception text. The eval runs both because the gap between them is the whole
    argument for doing verification this way.
    """
    hz = BY_KEY[cand.type]
    alert_id = f"{perception.chunk_id}:{cand.type}"
    use_vlm = mode == "vlm" and bool(perception.frame_paths)
    model = model or (nim.VERIFIER_VLM if use_vlm else nim.VERIFIER_LLM)

    if use_vlm:
        prompt = VLM_VERIFIER_PROMPT.format(
            n=len(perception.frame_paths),
            start=perception.start_s,
            end=perception.end_s,
            taxonomy=verifier_prompt_block(),
            ctype=cand.type,
            description=cand.description,
            evidence=cand.evidence,
            confidence=f"{cand.confidence:.2f}",
            temporal=temporal,
        )
    else:
        prompt = VERIFIER_PROMPT.format(
            taxonomy=verifier_prompt_block(),
            ctype=cand.type,
            description=cand.description,
            evidence=cand.evidence,
            confidence=f"{cand.confidence:.2f}",
            start=perception.start_s,
            end=perception.end_s,
            scene=perception.scene,
            activity=perception.activity,
            people=perception.people_count,
            truck_present=perception.truck_present,
            truck_moving=perception.truck_moving,
            temporal=temporal,
        )

    try:
        if use_vlm:
            text = nim.vlm_frames(
                model, prompt, perception.frame_paths, max_tokens=2400
            )
        else:
            text = nim.chat(
                model,
                [
                    {"role": "system", "content": VERIFIER_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2400,
            )
        raw = nim.parse_json(text)
        verdict = str(raw.get("verdict", "")).lower().strip()
        if verdict not in ("confirmed", "rejected"):
            verdict = "rejected"
        severity = str(raw.get("severity", "medium")).lower().strip()
        if severity not in ("low", "medium", "high"):
            severity = "medium"
        rationale = str(raw.get("rationale", ""))[:800]
        missing = str(raw.get("missing_evidence", ""))[:400]
    except Exception as exc:  # noqa: BLE001
        # If the verifier itself fails, keep the detection and mark it. Dropping a
        # real hazard because the adjudicator errored is the worse failure, and the
        # eval reports these separately so they cannot inflate precision.
        verdict, severity = "confirmed", "medium"
        rationale = f"verifier unavailable, detection passed through: {str(exc)[:160]}"
        missing = ""

    return Alert(
        alert_id=alert_id,
        chunk_id=perception.chunk_id,
        video_id=perception.video_id,
        start_s=perception.start_s,
        end_s=perception.end_s,
        type=cand.type,
        label=hz.label,
        description=cand.description,
        evidence=cand.evidence,
        vlm_confidence=cand.confidence,
        verdict=verdict,
        severity=severity,
        rationale=rationale,
        missing_evidence=missing,
        standard=hz.standard,
        verifier_model=model,
    )


def verify_all(
    perceptions: list[Perception],
    *,
    use_temporal: bool = True,
    model: str = "",
    mode: str = "vlm",
    workers: int = 6,
    progress=None,
    include=None,
) -> list[Alert]:
    """Adjudicate every candidate across every window.

    `include` is an optional predicate on a Perception deciding whether its candidates
    are adjudicated at all. Windows it excludes still contribute temporal context to
    their neighbours, so switching a filter on and off changes exactly one thing. The
    eval depends on that: an earlier version filtered the list before building context,
    which quietly moved two variables per arm and made the comparison meaningless.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = []
    for i, p in enumerate(perceptions):
        if include is not None and not include(p):
            continue
        ctx = _temporal_context(perceptions, i) if use_temporal else ""
        for c in p.candidates:
            jobs.append((c, p, ctx))

    alerts: list[Alert | None] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                verify_candidate, c, p, temporal=ctx, model=model, mode=mode
            ): i
            for i, (c, p, ctx) in enumerate(jobs)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            alerts[i] = fut.result()
            done += 1
            if progress:
                progress(done, len(jobs))
    return [a for a in alerts if a is not None]


def save(alerts: list[Alert], path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([a.as_dict() for a in alerts], indent=2))


def load(path) -> list[Alert]:
    from pathlib import Path

    return [Alert(**a) for a in json.loads(Path(path).read_text())]
