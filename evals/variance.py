"""Repeat the headline arms to size run-to-run variance.

With ten positive events, a single run's precision is not a number worth quoting on
its own. Both models are called at temperature 0, but the reasoning verifier is not
deterministic in practice, so the honest reporting unit is a spread across repeats
rather than one figure. This script produces that spread.

    python evals/variance.py --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_eval import PERCEPT_CACHE, arm_threshold, arm_verifier, load_gt, score  # noqa: E402

from foreman import perception  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    gt = load_gt()
    per = perception.load(PERCEPT_CACHE)
    by_id = {p.chunk_id: p for p in per}
    per = [by_id[g["chunk_id"]] for g in gt if g["chunk_id"] in by_id]

    runs: dict[str, list[dict]] = {"vlm_only": [], "vlm_verifier": []}

    # Perception is cached, so vlm_only is deterministic by construction. It is scored
    # once and repeated in the table as the fixed baseline the verifier is measured
    # against.
    base = score(arm_threshold(per, 0.0, False), gt)
    runs["vlm_only"] = [base] * args.repeats

    for i in range(args.repeats):
        print(f"--- repeat {i + 1}/{args.repeats} ---")
        t = time.time()
        pred, _ = arm_verifier(per, "vlm", True)
        s = score(pred, gt)
        s["wall_seconds"] = round(time.time() - t, 1)
        runs["vlm_verifier"].append(s)
        print(f"  P={s['precision']:.2f} R={s['recall']:.2f} F1={s['f1']:.2f} "
              f"({s['wall_seconds']:.0f}s)")

    summary = {}
    for arm, rs in runs.items():
        summary[arm] = {}
        for metric in ("precision", "recall", "f1"):
            vals = [r[metric] for r in rs]
            summary[arm][metric] = {
                "mean": round(statistics.mean(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "runs": vals,
            }

    out = {"repeats": args.repeats, "summary": summary}
    (ROOT / "evals" / "variance.json").write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
