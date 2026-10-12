"""Offline tests for the playground (no API calls).

    .venv/bin/python -m pytest playground/tests -q
"""

import json

import pytest

from playground import metrics
from playground.classifiers.base import Answer, Classifier
from playground.classifiers.lexical import Similarity, TfIdf
from playground.data import stratified_sample
from playground.tasks import OTHER, SegTask, TaxTask


def seg_row(k=3, gold="false", **slices):
    return {"id": f"s#{k}", "split": "eval", "gold": gold, "new_user_request": f"[{k}] USER: check the pump seal again",
            "conversation_so_far": [f"[{i}] USER: pump seal leaking at station {i} | ASSISTANT: noted" for i in range(k)],
            **slices}


def tax_row(text="[0] USER: summarize this visit note\n[1] USER: shorter please"):
    return {"id": "s-g0", "split": "eval", "text": text, "n_turns": 2,
            "gold": {"task": "healthcare.task.encounter_summary", "industry": "healthcare"}}


def test_seg_context_and_budget_trimming():
    row = seg_row(k=12)
    full, meta = SegTask("full").prepare(row)
    assert len(full["conversation_so_far"]) == 12 and meta["trimmed_turns"] == 0
    last2, _ = SegTask("last2").prepare(row)
    assert last2["conversation_so_far"] == row["conversation_so_far"][-2:]
    tight, meta = SegTask("full").prepare(row, budget_tokens=400)
    assert 1 <= len(tight["conversation_so_far"]) < 12 and meta["trimmed_turns"] > 0
    assert tight["conversation_so_far"][-1] == row["conversation_so_far"][-1]          # newest turns are kept


def test_tax_label_sets_and_views():
    task, industry = TaxTask("task"), TaxTask("industry")
    assert len(task.labels) == 65 and len(industry.labels) == 9 and OTHER in task.labels
    assert set(task.spec()["label"]["criteria"]) == set(task.labels)
    first, _ = TaxTask("task", "first_turn").prepare(tax_row())
    assert first["segment"] == "[0] USER: summarize this visit note"
    long_row = tax_row("x" * 200_000)
    cut, meta = TaxTask("task").prepare(long_row, budget_tokens=20_000)
    assert meta["text_chars_used"] < meta["text_chars"] and "[... middle omitted ...]" in cut["segment"]
    assert task.industry("healthcare.task.encounter_summary") == "healthcare"


def test_lexical_baselines_are_deterministic_and_sane():
    seg = SegTask()
    same = Similarity().predict(seg, seg_row())
    other = Similarity().predict(seg, {**seg_row(), "new_user_request": "[3] USER: draft a wedding toast for my sister"})
    assert same.probs["true"] < other.probs["true"] and same.cost_basis == "free_local"
    tax = TaxTask("task")
    first, second = TfIdf().predict(tax, tax_row()), TfIdf().predict(tax, tax_row())
    assert first.label == second.label and first.label in tax.labels


class Broken(Classifier):
    name = "broken"

    def __init__(self, mode):
        self.mode = mode

    def answer(self, spec, state):
        if self.mode == "raise":
            raise TimeoutError("read timed out")
        return Answer({"maybe": 1.0})


def test_errors_and_invalid_outputs_are_recorded_not_raised():
    task = SegTask()
    failed = Broken("raise").predict(task, seg_row())
    assert failed.label is None and failed.error.startswith("TimeoutError")
    invalid = Broken("invalid").predict(task, seg_row())
    assert invalid.label is None and "invalid output" in invalid.error


def test_stratified_sample_keeps_the_base_rate():
    rows = [{"id": str(i), "gold": "true" if i % 10 == 0 else "false"} for i in range(1000)]
    picked = stratified_sample(rows, 200, 0, lambda r: r["gold"])
    assert len(picked) == 200 and sum(r["gold"] == "true" for r in picked) == 20


def _write_run(tmp_path, task_config, preds):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "run": "test", "wall_seconds": 2.0, "concurrency": 4, "dataset": {"path": "x", "sha256": "0" * 64, "split": "eval",
        "n": len(preds), "review_status": ["unreviewed"]}, "task": task_config, "classifier": {"classifier": "stub"}}))
    with open(tmp_path / "predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps({"latency_s": 0.5, "cost_usd": 0.001, "cost_basis": "api_reported", "error": None,
                                "input_meta": {"est_input_tokens": 100}, "input_preview": "...", **p}) + "\n")
    return metrics.compute(tmp_path)


def test_seg_metrics_and_slices(tmp_path):
    def p(gold, label, kind=None, switch=None):
        return {"id": f"{gold}{label}{kind}{switch}", "gold": gold, "label": label, "probs": {"true": 0.9 if label == "true" else 0.1},
                "slices": {"followup_kind": kind, "switch_in": switch}}
    preds = [p("true", "true", switch="cross_industry"), p("true", "false", switch="same_task_new_object"),
             p("false", "true", kind="decoy"), p("false", "false", kind="decoy"), p("false", "false", kind="clarify")]
    m = _write_run(tmp_path, {"task": "seg", "context": "full", "threshold": 0.5, "n_classes": 2}, preds)
    q = m["quality"]
    assert q["boundary"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert (q["false_splits"], q["missed_splits"]) == (1, 1)
    assert q["slices"]["false_split_rate_by_followup_kind"]["decoy"] == {"rate": 0.5, "n": 2}
    assert q["slices"]["recall_by_switch"]["cross_industry"]["rate"] == 1.0
    assert m["cost"]["per_1k_usd"] == 1.0 and m["speed"]["p50_latency_s"] == 0.5
    assert (tmp_path / "report.md").read_text().count("### `") == 2                 # both mistakes listed


def test_tax_metrics_with_derived_industry(tmp_path):
    a, b = "healthcare.task.encounter_summary", "healthcare.task.record_fields"
    c = "insurance.task.loss_details"
    preds = [{"id": "1", "gold": a, "label": a}, {"id": "2", "gold": a, "label": b}, {"id": "3", "gold": c, "label": c}]
    for p in preds:
        p.update(probs={p["label"]: 1.0}, slices={})
    m = _write_run(tmp_path, {"task": "tax", "level": "task", "view": "segment", "n_classes": 65}, preds)
    q = m["quality"]
    assert q["accuracy"] == pytest.approx(0.6667, abs=1e-4)
    assert q["derived_industry_accuracy"] == 1.0                                       # the miss stays in healthcare
    assert q["top_confusions"] == [[a, b, 1]]
