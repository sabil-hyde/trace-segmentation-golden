"""Export the human-accepted sessions as the golden dataset.

    python -m golden.export

Writes into <data dir>:
  sessions.jsonl   canonical: gold slots + turns per segment
  seg.jsonl        one row per gap before user turn k >= 1; gold "true" where a segment starts
  tax.jsonl        one row per segment; gold industry / use_case / workflow / task / operation
  manifest.json    counts, base rate, coverage, filter drops, models, file hashes, cost
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from golden.vendor.harness import Turn, turn_line

from golden import DEFAULT_DATA_DIR
from golden.generate import GENERATOR_MODEL
from golden.seeds import read_seeds
from golden.sessions import Session, load_sessions, read_jsonl_by_id
from golden.taxonomy import COST_LOOKUP, OPERATIONS_FILE, USAGE_LOOKUP, file_sha256, lookup_path
from golden.verify import VERIFIER_MODEL

TUNE_SHARE = 0.3
FULL = 10 ** 9               # turn_line's per-turn caps, lifted: classifiers see every turn in full


def split_of(session_ids: List[str], seed: int) -> Dict[str, str]:
    """Whole sessions go to one split: a seeded shuffle, first TUNE_SHARE are tune."""
    ids = sorted(session_ids)
    random.Random(seed).shuffle(ids)
    n_tune = round(TUNE_SHARE * len(ids))
    return {sid: ("tune" if i < n_tune else "eval") for i, sid in enumerate(ids)}


def seg_rows(session: Session, split: str) -> List[Dict]:
    """One row per gap k >= 1. The context is the FULL prefix from turn 0: cutting it at a gold
    boundary would hand the previous answer to the classifier."""
    turns = session.harness_turns()
    lines = [turn_line(i, t, request_chars=FULL, reply_chars=FULL) for i, t in enumerate(turns)]
    starts = set(session.gold_starts())
    per_turn = {}                    # turn index -> (its segment, position inside the segment)
    for start, segment in zip(session.gold_starts(), session.seed.segments):
        for pos in range(segment.n_turns):
            per_turn[start + pos] = (segment, pos)
    rows = []
    for k in range(1, len(turns)):
        rows.append({
            "id": f"{session.seed.session_id}#{k}", "session_id": session.seed.session_id, "k": k, "split": split,
            "provenance": "synthetic",
            "conversation_so_far": lines[:k],
            "new_user_request": f"[{k}] USER: {' '.join(turns[k].request.split())}",
            "gold": "true" if k in starts else "false",
            **slicing_fields(per_turn[k], per_turn[k - 1][0]),
        })
    return rows


def slicing_fields(here: tuple, previous_segment) -> Dict:
    """Seed slots at this gap, for slicing errors: boundary type and style, or the follow-up kind."""
    segment, pos = here
    starts_here = pos == 0
    return {"switch_in": segment.switch_in if starts_here else None,
            "transition_style": segment.transition_style if starts_here else None,
            "followup_kind": None if starts_here else segment.followup_kinds[pos - 1],
            "user_len_bucket": segment.user_lengths[pos] if segment.user_lengths else None,
            "segment_len_before": previous_segment.n_turns, "segment_len_after": segment.n_turns}


def tax_rows(session: Session, split: str) -> List[Dict]:
    rows = []
    for segment, turns in session.segment_turns():
        rows.append({
            "id": segment.segment_id, "session_id": session.seed.session_id, "split": split, "provenance": "synthetic",
            "text": "\n".join(turn_line(i, Turn(i, t["user"], 0, replies=[t["assistant"]]), request_chars=FULL, reply_chars=FULL)
                     for i, t in enumerate(turns)),
            "n_turns": segment.n_turns,
            "gold": {"industry": segment.industry, "use_case": segment.use_case, "workflow": segment.workflow,
                     "task": segment.task, "operation": segment.operation},
        })
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()
    d = args.data_dir

    seeded = read_seeds(d / "seeds.jsonl")
    generated = load_sessions(d, only_ok=False)
    usable = [s for s in generated if s.ok]
    reviews = read_jsonl_by_id(d / "review.jsonl")
    verification = read_jsonl_by_id(d / "verification.jsonl")
    accepted = [s for s in usable if reviews.get(s.seed.session_id, {}).get("decision") == "accept"]
    if not accepted:
        raise SystemExit("no accepted sessions yet: run golden.review first")

    splits = split_of([s.seed.session_id for s in accepted], args.split_seed)
    seg = [r for s in accepted for r in seg_rows(s, splits[s.seed.session_id])]
    tax = [r for s in accepted for r in tax_rows(s, splits[s.seed.session_id])]
    write_jsonl(d / "sessions.jsonl", [{"session_id": s.seed.session_id, "split": splits[s.seed.session_id],
                                        "persona": s.seed.persona,
                                        "segments": [{**asdict(g), "turns": t} for g, t in s.segment_turns()]}
                                       for s in accepted])
    write_jsonl(d / "seg.jsonl", seg)
    write_jsonl(d / "tax.jsonl", tax)

    segments = [g for s in accepted for g in s.seed.segments]
    checked = [verification[s.seed.session_id] for s in accepted if s.seed.session_id in verification]
    seg_checks = [g for v in checked for g in v["segments"]]
    manifest = {
        "dataset": d.name,
        "counts": {"seeded_sessions": len(seeded), "generated": len(generated), "generation_ok": len(usable),
                   "reviewed": sum(s.seed.session_id in reviews for s in usable), "accepted": len(accepted),
                   "rejected": sum(r["decision"] == "reject" for r in reviews.values()),
                   "segments": len(tax), "gaps": len(seg), "positive_gaps": sum(r["gold"] == "true" for r in seg)},
        "base_rate_new_task": round(sum(r["gold"] == "true" for r in seg) / max(len(seg), 1), 4),
        "splits": dict(Counter(splits.values())),
        "coverage": {
            "length": dict(sorted(Counter(g.n_turns for g in segments).items())),
            "industry": dict(Counter(g.industry for g in segments)),
            "operation": dict(Counter(g.operation for g in segments)),
            "switch_in": dict(Counter(g.switch_in for g in segments if g.switch_in)),
            "tasks_covered": len({g.task for g in segments}),
        },
        "verifier_agreement_on_accepted": {
            "sessions_checked": len(checked),
            "boundaries_exact": round(sum(v["boundaries_match"] for v in checked) / max(len(checked), 1), 4),
            "task": round(sum(g["task_match"] for g in seg_checks) / max(len(seg_checks), 1), 4),
            "operation": round(sum(g["operation_match"] for g in seg_checks) / max(len(seg_checks), 1), 4),
        },
        "models": {"generator": GENERATOR_MODEL, "verifier": VERIFIER_MODEL,
                   "do_not_rank_on_this_data": ["google/gemini-*", "openai/gpt-*"]},
        "files_sha256": {USAGE_LOOKUP: file_sha256(lookup_path(USAGE_LOOKUP)), COST_LOOKUP: file_sha256(lookup_path(COST_LOOKUP)),
                         OPERATIONS_FILE.name: file_sha256(OPERATIONS_FILE), "seeds.jsonl": file_sha256(d / "seeds.jsonl")},
        "cost_usd": {"generation": round(sum(g["cost_usd"] for s in generated for g in s.generation["segments"]), 4),
                     "verification": round(sum(v["cost_usd"] for v in verification.values()), 4)},
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(accepted)} sessions -> {len(seg)} seg rows ({manifest['counts']['positive_gaps']} boundaries), "
          f"{len(tax)} tax rows in {d}")


if __name__ == "__main__":
    main()
