"""Rebuild the local footage the eval set refers to.

This repo does not redistribute video. The eval ships as labels and derived model
output keyed on chunk ids; this script rebuilds the footage those ids point at so the
numbers can be reproduced. Sources are public safety training videos, fetched to your
machine and used for analysis only.

    python scripts/fetch_data.py

Requires yt-dlp and ffmpeg on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw_yt"

# The eval set draws from these. h5cMg8SdE0s was pulled after review: it is
# AI-generated CGI rather than camera footage, and scoring a perception model on
# synthetic renders would not say anything about how it reads a real warehouse.
VIDEOS = {
    "E0wdFL4WsGU": "Warehouse Pedestrian Safety - Crossing in Front of a Forklift",
    "N4OQFi4UrR4": "Forklift Accidents Caused by Pedestrians Standing Too Close",
    "pLjVBBW_g7c": "Forklift near miss, pedestrian not watching",
    "nK6UGe1FwCI": "View from a Forklift Driver's Perspective",
    "MqvOjo62BHQ": "Forklift safety | Pedestrians",
    "fz9Q7-7l-40": "Forklift and Pedestrian Safety: Avoiding Accidents in the Warehouse",
    "4p2vg_IbmKY": "Fields of Vision: Pedestrian Safety around Forklifts",
}


def main() -> None:
    for tool in ("yt-dlp", "ffmpeg"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH. brew install {tool}")

    DEST.mkdir(parents=True, exist_ok=True)
    failed = []
    for vid, title in VIDEOS.items():
        out = DEST / f"{vid}.mp4"
        if out.exists():
            print(f"  have  {vid}  {title}")
            continue
        print(f"  fetch {vid}  {title}")
        r = subprocess.run(
            [
                "yt-dlp",
                "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]",
                "-o", str(DEST / "%(id)s.%(ext)s"),
                f"https://www.youtube.com/watch?v={vid}",
            ],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not out.exists():
            failed.append(vid)
            print(f"        failed: {r.stderr.strip().splitlines()[-1:] or 'unknown error'}")

    if failed:
        print(
            f"\n{len(failed)} video(s) unavailable: {', '.join(failed)}.\n"
            "Sources can be taken down or region-locked over time. The eval will still "
            "run over whatever resolves; ground_truth.json records the chunk ids so any "
            "missing video simply drops those windows from the score."
        )
    have = sorted(p.stem for p in DEST.glob("*.mp4"))
    print(f"\n{len(have)}/{len(VIDEOS)} videos present in {DEST}")
    gt = json.loads((ROOT / "evals" / "ground_truth.json").read_text())
    covered = sum(1 for g in gt if g["video_id"] in have)
    print(f"{covered}/{len(gt)} labelled windows reproducible")


if __name__ == "__main__":
    main()
