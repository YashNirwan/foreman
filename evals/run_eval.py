"""Score the pipeline against hand-labelled ground truth, one arm at a time.

The point of this file is to make every claim in the README falsifiable. Each arm is a
different answer to "what should become an alert", run over identical perception
output so the comparison isolates the decision rule rather than sampling noise.

  vlm_only          every candidate the perception pass raised
  vlm_conf_60       candidates at confidence >= 0.60
  vlm_conf_80       candidates at confidence >= 0.80
  scene_gate        candidates from windows the model judged to be real footage
  llm_verifier      scene gate, then a reasoning LLM reviews the perception text
  vlm_verifier      scene gate, then a reasoning VLM re-opens the frames
  vlm_verifier_nogate  the VLM verifier without the scene gate, to price the gate

Scoring is per (chunk, hazard class). A window where the model flags the right class
counts once; flagging two classes in a window that has one is one hit and one false
positive. Localisation beyond the window is out of scope, since the window is the unit
a reviewer acts on.

Usage:
    python evals/run_eval.py --perceive     # run perception, cache it
    python evals/run_eval.py                # score all arms from cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foreman import nim, perception, verify  # noqa: E402
from foreman.video import Chunk  # noqa: E402

GT_PATH = ROOT / "evals" / "ground_truth.json"
PERCEPT_CACHE = ROOT / "evals" / "cache" / "perceptions.json"
RESULTS_PATH = ROOT / "evals" / "results.json"
REPORT_PATH = ROOT / "evals" / "RESULTS.md"


def load_gt() -> list[dict]:
    return json.loads(GT_PATH.read_text())


def run_perception(gt: list[dict]) -> list[perception.Perception]:
    """Run the perception pass once. Every arm scores the same output."""
    eval_set = {c["chunk_id"]: c for c in json.loads((ROOT / "evals" / "eval_set.json").read_text())}
    chunks = [Chunk(**eval_set[g["chunk_id"]]) for g in gt]
    print(f"perceiving {len(chunks)} chunks with {nim.VLM_PRIMARY} ...")
    t = time.time()
    per = perception.perceive_all(
        chunks,
        progress=lambda d, n: print(f"  {d}/{n}", end="\r", flush=True),
    )
    print(f"\nperception done in {time.time() - t:.0f}s")
    PERCEPT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    perception.save(per, PERCEPT_CACHE)
    return per


# --------------------------------------------------------------------------- arms


def arm_threshold(per, thresh: float, gate: bool):
    out = {}
    for p in per:
        if gate and not p.is_camera_footage:
            continue
        for c in p.candidates:
            if c.confidence >= thresh:
                out.setdefault(p.chunk_id, set()).add(c.type)
    return out


def arm_verifier(per, mode: str, gate: bool):
    # The full list is always passed so temporal context is identical across arms; the
    # gate only decides which windows get adjudicated.
    alerts = verify.verify_all(
        per, mode=mode,
        include=(lambda p: p.is_camera_footage) if gate else None,
        progress=lambda d, n: print(f"  verify {d}/{n}", end="\r", flush=True),
    )
    print()
    out = {}
    for a in alerts:
        if a.verdict == "confirmed":
            out.setdefault(a.chunk_id, set()).add(a.type)
    return out, alerts


# ------------------------------------------------------------------------- score


def score(pred: dict[str, set[str]], gt: list[dict]) -> dict:
    """Micro-averaged precision/recall over (chunk, class) pairs."""
    tp = fp = fn = 0
    fp_by_scene: Counter = Counter()
    fn_detail, fp_detail = [], []

    for g in gt:
        truth = set(g["hazards"])
        got = pred.get(g["chunk_id"], set())
        for cls in got - truth:
            fp += 1
            fp_by_scene[g["scene_type"]] += 1
            fp_detail.append({"chunk_id": g["chunk_id"], "class": cls,
                              "scene_type": g["scene_type"]})
        for cls in truth - got:
            fn += 1
            fn_detail.append({"chunk_id": g["chunk_id"], "class": cls})
        tp += len(truth & got)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # A safety queue is only worked if the noise is bearable, so report the rate a
    # reviewer actually feels: how many windows raise an alert that should not have.
    clean = [g for g in gt if not g["hazards"]]
    false_alarm_windows = sum(1 for g in clean if pred.get(g["chunk_id"]))
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "false_alarm_window_rate": round(false_alarm_windows / len(clean), 3) if clean else 0.0,
        "false_alarm_windows": false_alarm_windows,
        "clean_windows": len(clean),
        "fp_by_scene_type": dict(fp_by_scene),
        "false_negatives": fn_detail,
        "false_positives": fp_detail[:40],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perceive", action="store_true", help="re-run the perception pass")
    args = ap.parse_args()

    gt = load_gt()
    if args.perceive or not PERCEPT_CACHE.exists():
        per = run_perception(gt)
    else:
        per = perception.load(PERCEPT_CACHE)
        print(f"loaded cached perception for {len(per)} chunks")

    by_id = {p.chunk_id: p for p in per}
    per = [by_id[g["chunk_id"]] for g in gt if g["chunk_id"] in by_id]

    errs = [p for p in per if p.error]
    cands = sum(len(p.candidates) for p in per)
    gated_out = [p for p in per if not p.is_camera_footage]
    print(f"candidates raised: {cands}   parse errors: {len(errs)}   "
          f"windows gated as non-footage: {len(gated_out)}")

    results: dict[str, dict] = {}
    usage_before = nim.USAGE.as_dict()

    for name, thresh, gate in [
        ("vlm_only", 0.0, False),
        ("vlm_conf_60", 0.60, False),
        ("vlm_conf_80", 0.80, False),
        ("scene_gate", 0.0, True),
    ]:
        results[name] = score(arm_threshold(per, thresh, gate), gt)
        print(f"{name:22s} P={results[name]['precision']:.2f} "
              f"R={results[name]['recall']:.2f} F1={results[name]['f1']:.2f}")

    all_alerts = {}
    for name, mode, gate in [
        ("llm_verifier", "llm", True),
        ("vlm_verifier", "vlm", True),
        ("vlm_verifier_nogate", "vlm", False),
    ]:
        print(f"running {name} ...")
        t = time.time()
        pred, alerts = arm_verifier(per, mode, gate)
        results[name] = score(pred, gt)
        results[name]["wall_seconds"] = round(time.time() - t, 1)
        all_alerts[name] = [a.as_dict() for a in alerts]
        print(f"{name:22s} P={results[name]['precision']:.2f} "
              f"R={results[name]['recall']:.2f} F1={results[name]['f1']:.2f}")

    payload = {
        "meta": {
            "chunks": len(gt),
            "positive_chunks": sum(1 for g in gt if g["hazards"]),
            "positive_events": sum(len(g["hazards"]) for g in gt),
            "candidates_raised": cands,
            "perception_parse_errors": len(errs),
            "windows_gated_non_footage": len(gated_out),
            "vlm_primary": nim.VLM_PRIMARY,
            "verifier_vlm": nim.VERIFIER_VLM,
            "verifier_llm": nim.VERIFIER_LLM,
            "usage_before_arms": usage_before,
            "usage_total": nim.USAGE.as_dict(),
        },
        "arms": results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    (ROOT / "evals" / "cache" / "alerts_by_arm.json").write_text(json.dumps(all_alerts, indent=2))
    write_report(payload)
    print(f"\nwrote {RESULTS_PATH} and {REPORT_PATH}")


def write_report(payload: dict) -> None:
    m, arms = payload["meta"], payload["arms"]
    order = ["vlm_only", "vlm_conf_60", "vlm_conf_80", "scene_gate",
             "llm_verifier", "vlm_verifier", "vlm_verifier_nogate"]
    lines = [
        "# Eval results",
        "",
        f"{m['chunks']} hand-labelled windows, {m['positive_chunks']} containing a "
        f"hazard ({m['positive_events']} labelled events). Perception raised "
        f"{m['candidates_raised']} candidates.",
        "",
        f"Perception `{m['vlm_primary']}` | verifier VLM `{m['verifier_vlm']}` | "
        f"verifier LLM `{m['verifier_llm']}`",
        "",
        "| arm | precision | recall | F1 | false-alarm windows |",
        "|---|---|---|---|---|",
    ]
    for k in order:
        if k not in arms:
            continue
        a = arms[k]
        lines.append(
            f"| `{k}` | {a['precision']:.2f} | {a['recall']:.2f} | {a['f1']:.2f} | "
            f"{a['false_alarm_windows']}/{a['clean_windows']} "
            f"({a['false_alarm_window_rate']:.0%}) |"
        )
    lines += ["", "## False positives by frame content", "",
              "| arm | " + " | ".join(["footage", "mixed", "slide", "graphic", "talking_head"]) + " |",
              "|---|" + "---|" * 5]
    for k in order:
        if k not in arms:
            continue
        d = arms[k]["fp_by_scene_type"]
        lines.append(f"| `{k}` | " + " | ".join(
            str(d.get(s, 0)) for s in ["footage", "mixed", "slide", "graphic", "talking_head"]
        ) + " |")
    REPORT_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
