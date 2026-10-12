"""Metrics for one run: quality, slices, speed, cost, reliability; and report.md with every mistake.

    python -m playground.metrics runs/<run>          # recompute after editing this file

Segmentation  boundary precision / recall / F1, false splits (FP), missed splits (FN); recall per boundary type and
              transition style; false-split rate per follow-up kind (decoys) and per user-turn length.
Taxonomy      accuracy, macro-F1, per-class P/R/F1, top confusions; industry accuracy derived from task labels.
Speed         p50 / p95 latency, total runtime, throughput at the run's concurrency.
Cost          total and per 1,000 items with its basis; items with unknown cost counted, never treated as free.
Reliability   failures, timeouts, over-budget inputs, invalid outputs.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def f1_at(scored: List[Tuple[float, bool]], threshold: float) -> float:
    tp = sum(p >= threshold and g for p, g in scored)
    fp = sum(p >= threshold and not g for p, g in scored)
    fn = sum(p < threshold and g for p, g in scored)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))], 4)


def seg_quality(rows: List[Dict]) -> Dict:
    pos = [r for r in rows if r["gold"] == "true"]
    neg = [r for r in rows if r["gold"] == "false"]
    tp = sum(r["label"] == "true" for r in pos)
    fn = len(pos) - tp
    fp = sum(r["label"] == "true" for r in neg)
    out = {"boundary": _prf(tp, fp, fn), "false_splits": fp, "missed_splits": fn,
           "gold_boundaries": len(pos), "gaps": len(rows), "base_rate": round(len(pos) / max(len(rows), 1), 4)}

    def rate(group: List[Dict], key: str, hit: str) -> Dict:
        by = defaultdict(list)
        for r in group:
            by[str(r["slices"].get(key))].append(r["label"] == hit)
        return {k: {"rate": round(sum(v) / len(v), 4), "n": len(v)} for k, v in sorted(by.items())}

    out["slices"] = {
        "recall_by_switch": rate(pos, "switch_in", "true"),
        "recall_by_transition_style": rate(pos, "transition_style", "true"),
        "recall_by_user_len": rate(pos, "user_len_bucket", "true"),
        "false_split_rate_by_followup_kind": rate(neg, "followup_kind", "true"),
        "false_split_rate_by_user_len": rate(neg, "user_len_bucket", "true"),
    }
    scored = [(r["probs"].get("true", 0.0), r["gold"] == "true") for r in rows]
    out["threshold_sweep"] = [{"threshold": t / 10, "f1": round(f1_at(scored, t / 10), 4)} for t in range(1, 10)]
    return out


def tax_quality(rows: List[Dict], industry_of: Dict[str, str]) -> Dict:
    per = {}
    for c in sorted({r["gold"] for r in rows}):
        tp = sum(r["gold"] == c and r["label"] == c for r in rows)
        fp = sum(r["gold"] != c and r["label"] == c for r in rows)
        fn = sum(r["gold"] == c and r["label"] != c for r in rows)
        per[c] = {**_prf(tp, fp, fn), "support": tp + fn}
    confusions = Counter((r["gold"], r["label"]) for r in rows if r["gold"] != r["label"])
    out = {"accuracy": round(sum(r["gold"] == r["label"] for r in rows) / max(len(rows), 1), 4),
           "macro_f1": round(sum(v["f1"] for v in per.values()) / max(len(per), 1), 4),
           "per_class": per, "top_confusions": [[g, p, n] for (g, p), n in confusions.most_common(15)]}
    if industry_of:      # task-level run: how often is at least the industry right?
        out["derived_industry_accuracy"] = round(
            sum(industry_of.get(r["gold"]) == industry_of.get(r["label"]) for r in rows) / max(len(rows), 1), 4)
    return out


def compute(run_dir: Path) -> Dict:
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    preds = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines() if line.strip()]
    answered = [p for p in preds if p["error"] is None]
    errors = [p["error"] for p in preds if p["error"]]
    task = man["task"]
    if task["task"] == "seg":
        quality = seg_quality(answered)
    else:
        industry_of = {}
        if task["level"] == "task":
            from golden.taxonomy import load_tasks
            industry_of = {t.id: t.industry for t in load_tasks().values()}
        quality = tax_quality(answered, industry_of)
    latencies = [p["latency_s"] for p in preds if p["latency_s"] is not None]
    known = [p["cost_usd"] for p in preds if p["cost_usd"] is not None]
    bases = sorted({p["cost_basis"] for p in preds if p["error"] is None})
    wall = man["wall_seconds"]
    in_tokens = [p["input_meta"].get("est_input_tokens") for p in preds if p["input_meta"].get("est_input_tokens")]
    trimmed = sum(bool(p["input_meta"].get("trimmed_turns")) or
                  p["input_meta"].get("text_chars_used", 0) < p["input_meta"].get("text_chars", 0) for p in preds)
    m = {
        "run": man["run"], "task": task, "classifier": man["classifier"]["classifier"],
        "model": man["classifier"].get("model"), "dataset": man["dataset"],
        "threshold": task.get("threshold"), "threshold_fit": bool(man.get("threshold_fit")),
        "counts": {"items": len(preds), "answered": len(answered)},
        "quality": quality,
        "speed": {"p50_latency_s": _pct(latencies, 0.5), "p95_latency_s": _pct(latencies, 0.95),
                  "total_runtime_s": wall, "throughput_per_s": round(len(preds) / wall, 3) if wall else None,
                  "concurrency": man["concurrency"]},
        "cost": {"total_usd": round(sum(known), 6), "per_1k_usd": round(1000 * sum(known) / len(known), 4) if known else None,
                 "basis": bases, "items_with_unknown_cost": len(preds) - len(known),
                 "threshold_fit_usd": (man.get("threshold_fit") or {}).get("fit_cost_usd")},
        "reliability": {"failures": len(errors), "timeouts": sum("Timeout" in e for e in errors),
                        "invalid_outputs": sum(e.startswith("invalid output") for e in errors),
                        "error_examples": sorted(set(e[:140] for e in errors))[:5]},
        "input": {"median_est_tokens": _pct(in_tokens, 0.5), "p95_est_tokens": _pct(in_tokens, 0.95),
                  "trimmed_share": round(trimmed / max(len(preds), 1), 4)},
    }
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=1))
    (run_dir / "report.md").write_text(report(m, preds))
    return m


def headline(m: Dict) -> str:
    q = m["quality"]
    quality = (f"F1 {q['boundary']['f1']} (P {q['boundary']['precision']}, R {q['boundary']['recall']}), "
               f"false splits {q['false_splits']}, missed {q['missed_splits']}" if m["task"]["task"] == "seg"
               else f"accuracy {q['accuracy']}, macro-F1 {q['macro_f1']}"
               + (f", industry (derived) {q['derived_industry_accuracy']}" if "derived_industry_accuracy" in q else ""))
    s, c, r = m["speed"], m["cost"], m["reliability"]
    return (f"  {quality}\n  p50 {s['p50_latency_s']}s p95 {s['p95_latency_s']}s, {s['throughput_per_s']}/s; "
            f"cost ${c['total_usd']} (${c['per_1k_usd']}/1k, {','.join(c['basis'])}); failures {r['failures']}")


def report(m: Dict, preds: List[Dict]) -> str:
    q, t = m["quality"], m["task"]
    L = [f"# {m['run']}", "",
         f"- task: **{t['task']}** ({', '.join(f'{k}={v}' for k, v in t.items() if k in ('context', 'level', 'view', 'threshold', 'n_classes'))})",
         f"- classifier: **{m['classifier']}**" + (f" ({m['model']})" if m.get("model") else ""),
         f"- data: `{m['dataset']['path']}` sha256 {m['dataset']['sha256'][:12]}, split {m['dataset']['split']}, "
         f"n={m['dataset']['n']}, review status {m['dataset']['review_status']}", "", "## Headline", "", "```",
         headline(m), "```", ""]
    if t["task"] == "seg":
        L += ["## Slices", ""]
        for name, table in q["slices"].items():
            L += [f"**{name}**: " + ", ".join(f"{k} {v['rate']:.0%} (n={v['n']})" for k, v in table.items()), ""]
        L += ["**threshold sweep (diagnostic)**: " + ", ".join(f"{s['threshold']}: {s['f1']}" for s in q["threshold_sweep"]), ""]
    else:
        L += ["## Top confusions (gold -> predicted)", ""] + [f"- {g} -> {p}: {n}" for g, p, n in q["top_confusions"]]
        worst = sorted(q["per_class"].items(), key=lambda kv: kv[1]["f1"])[:10]
        L += ["", "## Weakest classes", "", "| class | P | R | F1 | n |", "|---|---|---|---|---|"]
        L += [f"| {c} | {v['precision']} | {v['recall']} | {v['f1']} | {v['support']} |" for c, v in worst]
    mistakes = [p for p in preds if p["error"] is None and p["label"] != p["gold"]]
    L += ["", f"## Mistakes ({len(mistakes)})", ""]
    for p in mistakes[:300]:
        top = dict(sorted(p["probs"].items(), key=lambda kv: -kv[1])[:3])
        L += [f"### `{p['id']}` gold **{p['gold']}** -> **{p['label']}**",
              f"probs {json.dumps(top)} · slices {json.dumps({k: v for k, v in p['slices'].items() if v is not None})}",
              "", "```", p["input_preview"], "```", ""]
    if len(mistakes) > 300:
        L.append(f"... {len(mistakes) - 300} more in predictions.jsonl")
    return "\n".join(L)


if __name__ == "__main__":
    print(headline(compute(Path(sys.argv[1]))))
