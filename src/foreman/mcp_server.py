"""MCP server exposing the video timeline as agent tools.

The pipeline produces a searchable, adjudicated record of a shift. Putting that behind
MCP means the consumer does not have to be this repo's UI: Claude Code, a Metropolis
agent, or an on-call assistant can all ask the same questions over the same index.

The tool boundary is drawn where a supervisor's questions actually fall, not where the
code happens to split. `search_timeline` answers "did anyone ever...", `list_alerts`
answers "what needs attention", `explain_alert` answers "why did you flag this", and
`shift_summary` answers "how was the shift". Tools that mirror internal call graphs
instead of user questions are how agent surfaces become unusable.

Run:
    python -m foreman.mcp_server            # stdio
Register with Claude Code:
    claude mcp add foreman -- /abs/path/.venv/bin/python -m foreman.mcp_server
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import verify
from .hazards import BY_KEY, HAZARDS
from .index import TimelineIndex
from .perception import load as load_perceptions

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "runs"

mcp = MCPServer(
    "foreman",
    instructions=(
        "Query analysed warehouse safety footage. Call hazard_taxonomy first if you "
        "need to interpret an alert: each class has a specific evidence bar, and the "
        "alerts only mean something against it. list_alerts returns confirmed hazards; "
        "list_rejected_detections shows what was filtered and why."
    ),
)


def _runs() -> list[str]:
    if not RUNS.exists():
        return []
    return sorted(p.name for p in RUNS.iterdir() if (p / "summary.json").exists())


def _resolve(video_id: str | None) -> Path:
    available = _runs()
    if not available:
        raise ValueError(
            "No processed video runs found. Run: python -m foreman.pipeline <video.mp4>"
        )
    if video_id is None:
        if len(available) > 1:
            raise ValueError(
                f"Multiple runs available, pass video_id. Options: {', '.join(available)}"
            )
        return RUNS / available[0]
    if video_id not in available:
        raise ValueError(f"Unknown video_id {video_id!r}. Options: {', '.join(available)}")
    return RUNS / video_id


@mcp.tool()
def list_processed_videos() -> list[dict[str, Any]]:
    """List videos that have been processed and are available to query."""
    out = []
    for name in _runs():
        s = json.loads((RUNS / name / "summary.json").read_text())
        out.append({
            "video_id": name,
            "windows": s["windows"],
            "confirmed_alerts": s["confirmed"],
            "by_class": s.get("by_class", {}),
        })
    return out


@mcp.tool()
def list_alerts(
    video_id: str | None = None,
    hazard_type: str | None = None,
    min_severity: str = "low",
) -> list[dict[str, Any]]:
    """List verified safety alerts for a processed video.

    Args:
        video_id: which processed video; omit if only one has been processed.
        hazard_type: optional filter, one of the taxonomy keys.
        min_severity: low, medium or high. Defaults to low, meaning everything.

    Returns confirmed alerts only. Rejected detections are available via
    `list_rejected_detections` if you want to audit what the verifier removed.
    """
    run = _resolve(video_id)
    alerts = verify.load(run / "alerts.json")
    rank = {"low": 0, "medium": 1, "high": 2}
    floor = rank.get(min_severity.lower(), 0)
    out = []
    for a in alerts:
        if a.verdict != "confirmed":
            continue
        if hazard_type and a.type != hazard_type:
            continue
        if rank.get(a.severity, 1) < floor:
            continue
        out.append({
            "alert_id": a.alert_id, "type": a.type, "label": a.label,
            "severity": a.severity, "start_s": a.start_s, "end_s": a.end_s,
            "timestamp": f"{int(a.start_s // 60)}:{int(a.start_s % 60):02d}",
            "rationale": a.rationale, "standard": a.standard,
        })
    out.sort(key=lambda x: (-rank.get(x["severity"], 1), x["start_s"]))
    return out


@mcp.tool()
def list_rejected_detections(video_id: str | None = None) -> list[dict[str, Any]]:
    """List detections the verifier rejected, with the evidence it found missing.

    Useful for auditing whether the system is suppressing real hazards. A rejection
    that looks wrong here is the fastest signal that the taxonomy's evidence bar for
    that class is set incorrectly.
    """
    run = _resolve(video_id)
    return [
        {
            "alert_id": a.alert_id, "type": a.type, "start_s": a.start_s,
            "claim": a.description, "missing_evidence": a.missing_evidence,
            "rationale": a.rationale, "vlm_confidence": a.vlm_confidence,
        }
        for a in verify.load(run / "alerts.json")
        if a.verdict == "rejected"
    ]


@mcp.tool()
def search_timeline(query: str, video_id: str | None = None, k: int = 8) -> list[dict[str, Any]]:
    """Search the footage in natural language.

    Searches every analysed window, not just the ones that produced an alert, so it
    answers questions the hazard taxonomy does not cover: "someone carrying a long
    load past the racking", "the aisle by the loading door", "anyone on a phone".

    Args:
        query: what to look for, in plain language.
        video_id: which processed video; omit if only one has been processed.
        k: how many windows to return.
    """
    run = _resolve(video_id)
    idx = TimelineIndex.load(run / "index")
    return [
        {
            "chunk_id": h.chunk_id,
            "timestamp": f"{int(h.start_s // 60)}:{int(h.start_s % 60):02d}",
            "start_s": h.start_s, "end_s": h.end_s,
            "relevance": round(h.score, 3),
            "what_was_seen": h.caption,
            "hazard_candidates": h.hazards,
        }
        for h in idx.search(query, k=k)
    ]


@mcp.tool()
def explain_alert(alert_id: str, video_id: str | None = None) -> dict[str, Any]:
    """Return the full evidence chain behind one alert.

    Gives the proposing model's claim, its confidence, the verifier's reasoning and the
    standard cited, so a supervisor can judge the alert rather than take it on trust.
    """
    run = _resolve(video_id)
    for a in verify.load(run / "alerts.json"):
        if a.alert_id == alert_id:
            hz = BY_KEY.get(a.type)
            return {
                "alert_id": a.alert_id, "type": a.type, "label": a.label,
                "verdict": a.verdict, "severity": a.severity,
                "window": f"{a.start_s:.0f}s - {a.end_s:.0f}s",
                "proposed_by": "perception VLM",
                "claim": a.description,
                "cited_evidence": a.evidence,
                "vlm_confidence": a.vlm_confidence,
                "verifier_model": a.verifier_model,
                "verifier_rationale": a.rationale,
                "missing_evidence": a.missing_evidence,
                "standard": a.standard,
                "evidence_required_for_class": hz.requires if hz else "",
                "known_false_positive_for_class": hz.common_false_positive if hz else "",
            }
    raise ValueError(f"no alert {alert_id!r}")


@mcp.tool()
def shift_summary(video_id: str | None = None) -> dict[str, Any]:
    """Summarise a processed shift: volumes, alert mix, and what was filtered out."""
    run = _resolve(video_id)
    s = json.loads((run / "summary.json").read_text())
    per = load_perceptions(run / "perceptions.json")
    covered = sum(p.end_s - p.start_s for p in per if p.is_camera_footage)
    return {
        "video_id": s["video_id"],
        "windows_analysed": s["windows"],
        "footage_seconds_reviewed": round(covered, 1),
        "candidates_raised": s["candidates"],
        "windows_gated_as_non_footage": s["gated_non_footage"],
        "detections_reviewed": s["reviewed"],
        "alerts_confirmed": s["confirmed"],
        "by_class": s.get("by_class", {}),
        "by_severity": s.get("by_severity", {}),
        "processing_seconds": s["wall_seconds"],
        "models": s.get("models", {}),
    }


@mcp.tool()
def hazard_taxonomy() -> list[dict[str, str]]:
    """The hazard classes this system detects, with evidence bars and known failure modes.

    Read this before interpreting alerts: each class has a specific evidence standard,
    and knowing it is the difference between reading an alert and trusting it.
    """
    return [
        {
            "key": h.key, "label": h.label, "definition": h.definition,
            "evidence_required": h.requires, "standard": h.standard,
            "common_false_positive": h.common_false_positive,
        }
        for h in HAZARDS
    ]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
