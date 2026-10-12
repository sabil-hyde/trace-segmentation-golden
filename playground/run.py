"""Run one classifier on one task configuration and save everything about the run.

    python -m playground.run --task seg --classifier similarity --fit-threshold
    python -m playground.run --task seg --classifier jev --context last2
    python -m playground.run --task tax --classifier llm --level industry --view first_turn --sample 300

Writes runs/<utc time>_<task>_<classifier>_<variant>/:
  predictions.jsonl  one row per item: gold, label, probabilities, latency, tokens, cost, error, slices, input preview
  manifest.json      dataset (path, sha256, review status, split, sample), task config + the exact spec sent,
                     classifier config, concurrency, threshold and how it was chosen, hardware, timings
  metrics.json       quality, speed, cost, reliability (see metrics.py)
  report.md          the same as readable tables, plus every mistake
No answers are cached, so every latency and cost is measured.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from golden.vendor.llm import parallel_map
from playground import DEFAULT_EXPORT, RUNS_DIR, metrics
from playground.classifiers import NAMES, make
from playground.data import load
from playground.tasks import SegTask, make_task

PREVIEW_CHARS = 1500


def preview(task, row: Dict) -> str:
    if task.name == "seg":
        return "\n".join(row["conversation_so_far"][-3:] + ["--- NEW USER TURN ---", row["new_user_request"]])[-PREVIEW_CHARS:]
    return row["text"][:PREVIEW_CHARS]


def predict_all(classifier, task, rows: List[Dict], concurrency: int) -> List:
    done = [0]

    def one(row):
        p = classifier.predict(task, row)
        done[0] += 1
        if done[0] % max(1, len(rows) // 10) == 0:
            print(f"  {done[0]}/{len(rows)}", flush=True)
        return p
    return parallel_map(one, rows, concurrency)


def fit_threshold(classifier, task: SegTask, args) -> Dict:
    """Pick the boundary threshold with the best F1 on the TUNE split, never on eval."""
    rows, info = load(args.export, "seg", "tune", args.fit_sample, args.seed)
    print(f"fitting the threshold on {len(rows)} tune rows")
    preds = predict_all(classifier, task, rows, args.concurrency)
    scored = [(p.probs.get("true", 0.0), r["gold"] == "true") for p, r in zip(preds, rows) if p.error is None]
    best = max((metrics.f1_at(scored, t / 100), t / 100) for t in range(1, 100))
    return {"threshold": best[1], "tune_f1": round(best[0], 4), "fit_on": info,
            "fit_cost_usd": round(sum(p.cost_usd or 0.0 for p in preds), 6)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["seg", "tax"])
    ap.add_argument("--classifier", required=True, choices=NAMES)
    ap.add_argument("--model", default=None, help="chat model for --classifier llm")
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT, help="folder with seg.jsonl / tax.jsonl")
    ap.add_argument("--split", default="eval", choices=["eval", "tune", "all"])
    ap.add_argument("--sample", type=int, default=None, help="stratified sample size (keeps the label mix)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context", default="full", choices=["full", "last2"], help="seg: how much history to show")
    ap.add_argument("--threshold", type=float, default=0.5, help="seg: p(new task) needed to call a boundary")
    ap.add_argument("--fit-threshold", action="store_true", help="seg: choose the threshold on the tune split")
    ap.add_argument("--fit-sample", type=int, default=1500, help="tune rows used by --fit-threshold")
    ap.add_argument("--level", default="task", choices=["task", "industry"], help="tax: 64 tasks or 8 industries")
    ap.add_argument("--view", default="segment", choices=["segment", "first_turn"], help="tax: what the model reads")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--tag", default="", help="extra words for the run folder name")
    args = ap.parse_args()

    task = make_task(args.task, args.context, args.threshold, args.level, args.view)
    classifier = make(args.classifier, args.model)
    rows, data_info = load(args.export, args.task, args.split, args.sample, args.seed)
    fit = fit_threshold(classifier, task, args) if args.fit_threshold and args.task == "seg" else None
    if fit:
        task.threshold = fit["threshold"]

    variant = args.context if args.task == "seg" else f"{args.level}-{args.view}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / "_".join(x for x in (stamp, args.task, args.classifier, variant, args.tag) if x)
    run_dir.mkdir(parents=True)
    print(f"{len(rows)} {args.split} rows, {args.task}/{variant}, classifier {args.classifier} -> {run_dir.name}")

    started = time.monotonic()
    preds = predict_all(classifier, task, rows, args.concurrency)
    wall = time.monotonic() - started

    with open(run_dir / "predictions.jsonl", "w") as f:
        for row, p in zip(rows, preds):
            f.write(json.dumps({"id": row["id"], "gold": task.gold(row), **p.to_json(), "slices": task.slices(row),
                                "input_preview": preview(task, row)}, ensure_ascii=False) + "\n")
    manifest = {
        "run": run_dir.name, "started_utc": stamp, "wall_seconds": round(wall, 3),
        "dataset": data_info, "task": {**task.config(), "labels": task.labels, "spec": task.spec()},
        "threshold_fit": fit, "classifier": classifier.config(),
        "batch_size": 1, "concurrency": args.concurrency,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "cpus": os.cpu_count(),
                     "python": sys.version.split()[0],
                     "note": "API classifiers run remotely; local hardware only sends requests"},
        "command": " ".join(sys.argv),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    m = metrics.compute(run_dir)
    print(metrics.headline(m) + f"\n  report: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
