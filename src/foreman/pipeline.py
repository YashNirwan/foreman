"""End-to-end run: video in, verified alerts and a searchable timeline out.

    python -m foreman.pipeline data/raw_yt/E0wdFL4WsGU.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import nim, perception, verify
from .index import TimelineIndex
from .video import chunk_video, extract_clip

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "data" / "runs"


def run(
    video: Path,
    *,
    max_chunks: int | None = None,
    clips: bool = True,
    quiet: bool = False,
) -> dict:
    def say(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    video = Path(video)
    out = OUT_ROOT / video.stem
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    say(f"[1/5] chunking {video.name}")
    chunks = chunk_video(video, ROOT / "data" / "chunks", max_chunks=max_chunks)
    say(f"      {len(chunks)} windows")

    say(f"[2/5] perception via {nim.VLM_PRIMARY}")
    per = perception.perceive_all(
        chunks,
        progress=None if quiet else (lambda d, n: print(f"      {d}/{n}", end="\r", flush=True)),
    )
    perception.save(per, out / "perceptions.json")
    n_cand = sum(len(p.candidates) for p in per)
    n_gated = sum(1 for p in per if not p.is_camera_footage)
    say(f"\n      {n_cand} candidates, {n_gated} windows gated as non-footage")

    say(f"[3/5] verification via {nim.VERIFIER_VLM}")
    alerts = verify.verify_all(
        per,
        include=lambda p: p.is_camera_footage,
        progress=None if quiet else (lambda d, n: print(f"      {d}/{n}", end="\r", flush=True)),
    )
    confirmed = [a for a in alerts if a.verdict == "confirmed"]
    verify.save(alerts, out / "alerts.json")
    say(f"\n      {len(confirmed)} confirmed of {len(alerts)} reviewed")

    say("[4/5] building search index")
    idx = TimelineIndex.build(per)
    idx.save(out / "index")

    if clips and confirmed:
        say(f"[5/5] cutting {len(confirmed)} evidence clips")
        for a in confirmed:
            extract_clip(video, a.start_s, a.end_s, out / "clips" / f"{a.alert_id.replace(':', '_')}.mp4")
    else:
        say("[5/5] no confirmed alerts, skipping clips")

    summary = {
        "video": str(video),
        "video_id": video.stem,
        "windows": len(chunks),
        "candidates": n_cand,
        "gated_non_footage": n_gated,
        "reviewed": len(alerts),
        "confirmed": len(confirmed),
        "by_class": {
            t: sum(1 for a in confirmed if a.type == t) for t in {a.type for a in confirmed}
        },
        "by_severity": {
            s: sum(1 for a in confirmed if a.severity == s) for s in {a.severity for a in confirmed}
        },
        "wall_seconds": round(time.time() - t0, 1),
        "usage": nim.USAGE.as_dict(),
        "models": {
            "perception": nim.VLM_PRIMARY,
            "verifier": nim.VERIFIER_VLM,
            "embeddings": nim.EMBED_MODEL,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    say(f"\ndone in {summary['wall_seconds']}s -> {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Foreman pipeline over a video.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--no-clips", action="store_true")
    args = ap.parse_args()
    if not args.video.exists():
        sys.exit(f"no such file: {args.video}")
    run(args.video, max_chunks=args.max_chunks, clips=not args.no_clips)


if __name__ == "__main__":
    main()
