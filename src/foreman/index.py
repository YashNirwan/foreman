"""Semantic search over the perception timeline.

Once every window has a description, natural-language search over the footage is just
retrieval over those descriptions. The embeddings come from NeMo Retriever's
vision-language model, so query and window text land in the same space the perception
pass was describing.

A flat numpy matrix is the right call at this scale. A 10-hour shift at 8-second
windows is 4,500 vectors, and an exact dot product over that is sub-millisecond. The
moment this needs to span a site's worth of cameras and weeks of retention, the
interface below is the one a real vector store slots behind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import nim
from .perception import Perception


@dataclass
class SearchHit:
    chunk_id: str
    video_id: str
    start_s: float
    end_s: float
    score: float
    caption: str
    hazards: list[str]


class TimelineIndex:
    def __init__(self, records: list[dict], matrix: np.ndarray):
        self.records = records
        self.matrix = matrix

    @classmethod
    def build(cls, perceptions: list[Perception]) -> "TimelineIndex":
        usable = [p for p in perceptions if p.caption.strip()]
        vecs = nim.embed([p.caption for p in usable], input_type="passage")
        m = np.asarray(vecs, dtype=np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        records = [
            {
                "chunk_id": p.chunk_id,
                "video_id": p.video_id,
                "start_s": p.start_s,
                "end_s": p.end_s,
                "caption": p.caption,
                "hazards": [c.type for c in p.candidates],
                "is_camera_footage": p.is_camera_footage,
            }
            for p in usable
        ]
        return cls(records, m)

    def search(self, query: str, k: int = 8, footage_only: bool = True) -> list[SearchHit]:
        q = np.asarray(nim.embed([query], input_type="query")[0], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9
        scores = self.matrix @ q
        order = np.argsort(-scores)
        hits: list[SearchHit] = []
        for i in order:
            r = self.records[int(i)]
            if footage_only and not r.get("is_camera_footage", True):
                continue
            hits.append(
                SearchHit(
                    chunk_id=r["chunk_id"], video_id=r["video_id"],
                    start_s=r["start_s"], end_s=r["end_s"],
                    score=float(scores[int(i)]), caption=r["caption"],
                    hazards=r["hazards"],
                )
            )
            if len(hits) >= k:
                break
        return hits

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.matrix)
        path.with_suffix(".json").write_text(json.dumps(self.records, indent=2))

    @classmethod
    def load(cls, path: Path) -> "TimelineIndex":
        path = Path(path)
        return cls(
            json.loads(path.with_suffix(".json").read_text()),
            np.load(path.with_suffix(".npy")),
        )
