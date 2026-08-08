"""Video ingest: split source footage into fixed windows and sample frames.

Chunking is the first real design decision in a video understanding pipeline and it
sets the ceiling on everything downstream. Too short and the model cannot see motion,
so a pedestrian standing still and a pedestrian walking into a travel path look
identical. Too long and a single alert cannot be localised to a moment an
investigator can actually review. Eight seconds at four sampled frames is the
setting this pipeline defaults to, and `evals/sweep_chunking.py` is what justified it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_WINDOW_S = 8.0
DEFAULT_FRAMES_PER_CHUNK = 4
DEFAULT_WIDTH = 768


@dataclass
class Chunk:
    chunk_id: str
    video_id: str
    start_s: float
    end_s: float
    frame_paths: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(video),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def chunk_video(
    video: Path,
    out_root: Path,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    frames_per_chunk: int = DEFAULT_FRAMES_PER_CHUNK,
    width: int = DEFAULT_WIDTH,
    max_chunks: int | None = None,
    skip_head_s: float = 0.0,
    force: bool = False,
) -> list[Chunk]:
    """Cut `video` into windows and extract evenly spaced JPEG frames from each.

    Frames are pulled with one ffmpeg call per chunk using an input seek (`-ss` before
    `-i`), which keyframe-seeks and is dramatically faster than decoding from zero for
    every window. Extraction is idempotent so re-running the pipeline after a model
    change does not redo the ffmpeg work.
    """
    video_id = video.stem
    out_dir = out_root / video_id
    manifest = out_dir / "chunks.json"

    if manifest.exists() and not force:
        return [Chunk(**c) for c in json.loads(manifest.read_text())]

    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(video)
    fps = frames_per_chunk / window_s

    chunks: list[Chunk] = []
    idx = 0
    t = skip_head_s
    while t < duration:
        if max_chunks is not None and idx >= max_chunks:
            break
        # A trailing partial window shorter than half the stride has too little motion
        # context to reason over, so it is dropped rather than scored on thin evidence.
        if duration - t < window_s / 2:
            break

        chunk_id = f"{video_id}_{idx:04d}"
        frame_dir = out_dir / chunk_id
        frame_dir.mkdir(exist_ok=True)

        subprocess.run(
            [
                "ffmpeg", "-v", "error",
                "-ss", f"{t:.2f}",
                "-i", str(video),
                "-t", f"{window_s:.2f}",
                "-vf", f"fps={fps},scale={width}:-2",
                "-frames:v", str(frames_per_chunk),
                "-q:v", "3",
                str(frame_dir / "f%02d.jpg"),
                "-y",
            ],
            capture_output=True, check=False,
        )

        frames = sorted(str(p) for p in frame_dir.glob("*.jpg"))
        if frames:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    video_id=video_id,
                    start_s=round(t, 2),
                    end_s=round(min(t + window_s, duration), 2),
                    frame_paths=frames,
                )
            )
            idx += 1
        t += window_s

    manifest.write_text(json.dumps([c.as_dict() for c in chunks], indent=2))
    return chunks


def extract_clip(video: Path, start_s: float, end_s: float, out_path: Path) -> Path:
    """Cut the source clip behind an alert, so a reviewer can watch the evidence.

    An alert a human cannot verify in one click is an alert they learn to ignore, so
    this is load-bearing for the product rather than a convenience.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-ss", f"{max(0.0, start_s - 1.0):.2f}",
            "-i", str(video),
            "-t", f"{(end_s - start_s) + 2.0:.2f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-an", str(out_path), "-y",
        ],
        capture_output=True, check=False,
    )
    return out_path
