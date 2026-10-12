"""Routine quality checks over everything generated so far. They warn; they never stop a run.

    python -m golden.checks            # any time, e.g. while generate runs

generate.py runs these after every batch, prints the table and appends one row to
<data dir>/quality.jsonl, so a bad trend shows up early and you can Ctrl-C (generation resumes).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from golden.vendor.minhash import MinHasher

from golden import DEFAULT_DATA_DIR
from golden.sessions import Session, load_sessions, read_jsonl_by_id

PILOT_COST_PER_SEGMENT = 2.823 / 180          # measured on golden-v1-pilot
NEAR_DUP_JACCARD = 0.8
SHORTCUT_F1_LIMIT = 0.5      # a single surface feature must not find boundaries better than this
MIN_BOUNDARIES = 20          # below this the shortcut fit is too noisy to report
META_PHRASES = ("as an ai", "i cannot", "i can't help", "i'm unable", "i am unable", "i don't have access",
                "language model")


@dataclass
class Check:
    name: str
    value: float
    limit: str          # human-readable bound, e.g. "<= 2%"
    ok: bool
    detail: str = ""


def _share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def generation_checks(generations: List[Dict]) -> List[Check]:
    segments = [g for r in generations for g in r["segments"]]
    n = len(segments)
    ok = [g for g in segments if g["status"] == "ok"]
    failed = n - len(ok)
    retried = sum(g["attempts"] > 1 for g in segments)
    # one count per segment and reason ("turn 3: ..." and "turn 5: ..." are the same reason)
    reasons = Counter(r for g in segments for r in {re.sub(r"^turn \d+: ", "", p)[:60] for p in g["problems"]})
    top_reason, top_count = reasons.most_common(1)[0] if reasons else ("-", 0)
    cost = sum(g["cost_usd"] for g in segments) / n if n else 0.0

    openers = [(g["segment_id"], g["turns"][0]["user"]) for g in ok]
    near_dups = MinHasher().dedup_ids(openers, NEAR_DUP_JACCARD) if openers else set()
    replies = [t["assistant"].lower() for g in ok for t in g["turns"]]
    meta = sum(any(p in r for p in META_PHRASES) for r in replies)

    return [
        Check("failed segments", _share(failed, n), "<= 2%", _share(failed, n) <= 0.02, f"{failed}/{n}"),
        Check("segments needing a retry", _share(retried, n), "<= 20%", _share(retried, n) <= 0.20, f"{retried}/{n}"),
        Check("top rejection reason", _share(top_count, n), "<= 10%", _share(top_count, n) <= 0.10,
              f"{top_reason!r} x{top_count}"),
        Check("cost per segment (USD)", round(cost, 4), f"<= {2 * PILOT_COST_PER_SEGMENT:.3f}",
              cost <= 2 * PILOT_COST_PER_SEGMENT, f"pilot {PILOT_COST_PER_SEGMENT:.4f}"),
        Check("near-duplicate openers", _share(len(near_dups), len(openers)), "<= 2%",
              _share(len(near_dups), len(openers)) <= 0.02, ", ".join(sorted(near_dups)[:5])),
        Check("assistant meta/refusal replies", _share(meta, len(replies)), "<= 1%",
              _share(meta, len(replies)) <= 0.01, f"{meta}/{len(replies)}"),
    ]


def verification_checks(verification: List[Dict]) -> List[Check]:
    segments = [g for v in verification for g in v["segments"]]
    # Recall is the health signal: a gold boundary the verifier cannot see means a switch is invisible.
    # Extra verifier splits are the intended difficulty (decoys), so precision is reported, not judged.
    gold = [set(v["boundaries_gold"]) - {0} for v in verification]
    pred = [set(v["boundaries_pred"]) - {0} for v in verification]
    found = sum(len(g & p) for g, p in zip(gold, pred))
    recall, precision = _share(found, sum(map(len, gold))), _share(found, sum(map(len, pred)))
    task = _share(sum(g["task_match"] for g in segments), len(segments))
    operation = _share(sum(g["operation_match"] for g in segments), len(segments))
    return [
        Check("verifier: boundary recall", recall, ">= 85%", recall >= 0.85,
              f"precision {precision:.0%} (low = decoys work), {len(verification)} sessions"),
        Check("verifier: task agreement", task, ">= 90%", task >= 0.90, f"{len(segments)} segments"),
        Check("verifier: operation agreement", operation, ">= 80%", operation >= 0.80, f"{len(segments)} segments"),
    ]


def _first_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z']+", text.lower())[:2])


def _f1(predicted: List[bool], gold: List[bool]) -> float:
    tp = sum(p and g for p, g in zip(predicted, gold))
    fp = sum(p and not g for p, g in zip(predicted, gold))
    fn = sum(g and not p for p, g in zip(predicted, gold))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _fit_then_score(fit: List[tuple], test: List[tuple]) -> float:
    """Best 'value >= threshold' rule on the fit half, scored as boundary F1 on the test half."""
    values = sorted({v for v, _ in fit})
    candidates = values[:: max(1, len(values) // 200)]
    best = max(candidates, key=lambda t: _f1([v >= t for v, _ in fit], [g for _, g in fit]))
    return _f1([v >= best for v, _ in test], [g for _, g in test])


def shortcut_checks(sessions: List[Session]) -> List[Check]:
    """Can one surface feature find the boundaries? Each rule is fit on half the sessions and scored on
    the other half. If any reaches SHORTCUT_F1_LIMIT, segmentation is being solved without understanding."""
    rows = []                                   # (half, is_boundary, user_text, previous_turn)
    for i, session in enumerate(sessions):
        turns, starts = session.turns(), set(session.gold_starts())
        rows += [(i % 2, k in starts, turns[k]["user"], turns[k - 1]) for k in range(1, len(turns))]
    if sum(gold for _, gold, _, _ in rows) < MIN_BOUNDARIES:
        return []

    # first-two-words lexicon: log-odds of starting a segment, learned on the fit half only
    pos = Counter(_first_words(u) for half, gold, u, _ in rows if half == 0 and gold)
    neg = Counter(_first_words(u) for half, gold, u, _ in rows if half == 0 and not gold)
    n_pos, n_neg, vocab = sum(pos.values()), sum(neg.values()), len(pos | neg) + 1

    def lexicon(text: str) -> float:
        w = _first_words(text)
        return math.log((pos[w] + 1) / (n_pos + vocab)) - math.log((neg[w] + 1) / (n_neg + vocab))

    features = {
        "user turn length": lambda u, prev: len(u),
        "line breaks": lambda u, prev: u.count("\n"),
        "length vs previous user turn": lambda u, prev: len(u) / max(len(prev["user"]), 1),
        "previous reply length": lambda u, prev: len(prev["assistant"]),
        "first two words": lambda u, prev: lexicon(u),
    }
    checks = []
    for name, feature in features.items():
        split = {0: [], 1: []}
        for half, gold, user, prev in rows:
            split[half].append((feature(user, prev), gold))
        f1 = _fit_then_score(split[0], split[1])
        checks.append(Check(f"shortcut: {name}", round(f1, 3), f"F1 < {SHORTCUT_F1_LIMIT}", f1 < SHORTCUT_F1_LIMIT))

    def median_len(texts: List[str]) -> float:
        return sorted(map(len, texts))[len(texts) // 2]
    first = [u for _, gold, u, _ in rows if gold]
    later = [u for _, gold, u, _ in rows if not gold]
    ratio = median_len(first) / max(median_len(later), 1)
    checks.append(Check("first/later user length (median ratio)", round(ratio, 2), "0.8-1.25", 0.8 <= ratio <= 1.25,
                        f"{median_len(first)} vs {median_len(later)} chars"))
    return checks


def run_checks(data_dir: Path) -> List[Check]:
    generations = list(read_jsonl_by_id(Path(data_dir) / "generations.jsonl").values())
    verification = list(read_jsonl_by_id(Path(data_dir) / "verification.jsonl").values())
    checks = generation_checks(generations) if generations else []
    checks += verification_checks(verification) if verification else []
    return checks + shortcut_checks(load_sessions(Path(data_dir)))


def format_table(checks: List[Check]) -> str:
    lines = []
    for c in checks:
        value = f"{c.value:.1%}" if c.limit.endswith("%") else f"{c.value}"
        lines.append(f"  {'ok' if c.ok else '⚠ '}  {c.name:<32} {value:>8}  (limit {c.limit})  {c.detail}")
    return "\n".join(lines)


def record(data_dir: Path, checks: List[Check], sessions_done: int) -> None:
    """Append one row to quality.jsonl so trends across batches can be read later."""
    row = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sessions": sessions_done,
           "warnings": [c.name for c in checks if not c.ok], "checks": [asdict(c) for c in checks]}
    with open(Path(data_dir) / "quality.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()
    print(format_table(run_checks(args.data_dir)))


if __name__ == "__main__":
    main()
