"""Blind re-labelling by a model from another family. It never sees the gold; its disagreements
are shown to the human reviewer, and its agreement rates go into the manifest.

    python -m golden.verify

Two checks per session:
  boundaries  the whole session, labelled with the repo's own reference prompt
              (vendor/segmentation.py LABEL_SYSTEM), must give exactly the gold task starts;
  taxonomy    each segment, classified blind into one of the 64 tasks (+ other) and one of the
              10 operations, must match its gold task and operation.
Writes <data dir>/verification.jsonl (resumable).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from golden.vendor.segmentation import LABEL_SYSTEM, label_text
from golden.settings import client as make_client
from golden.vendor.llm import LLMClient, parallel_map

from golden import DEFAULT_DATA_DIR
from golden.sessions import Session, load_sessions, read_jsonl_by_id
from golden.taxonomy import load_operations, load_tasks

VERIFIER_MODEL = "openai/gpt-6-luna"

TAXONOMY_PROMPT = """Classify this chat segment (all of it is about one request).

TASK TYPES (pick exactly one id, or "other" if none fits):
{tasks}

OPERATIONS (judge the user's FIRST message only: pick the operation that opening request needs most):
{operations}

SEGMENT:
{segment}

Reply with JSON only: {{"task": "<task id>", "operation": "<operation id>"}}"""


def _ask(client: LLMClient, system: Optional[str], prompt: str) -> tuple:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    res = client.chat(VERIFIER_MODEL, messages, temperature=0.0, json_mode=True)
    cost = float((res.usage or {}).get("cost") or 0.0)
    if res.error:
        return None, cost, res.error[:200]
    try:
        return json.loads(res.content), cost, None
    except ValueError:
        return None, cost, f"unparseable reply: {res.content[:120]!r}"


def _starts(answer: Optional[Dict]) -> List[int]:
    """Task starts from a LABEL_SYSTEM answer; malformed entries are dropped (and then disagree with gold)."""
    out = []
    for task in (answer or {}).get("tasks") or []:
        try:
            out.append(int(task["start"]))
        except (TypeError, ValueError, KeyError):
            pass
    return out


def verify_session(client: LLMClient, session: Session, task_menu: str, operation_menu: str) -> Dict:
    disagreements: List[str] = []
    cost = 0.0

    turns = session.harness_turns()
    answer, c, error = _ask(client, LABEL_SYSTEM, label_text(turns, None, session.seed.session_id))
    cost += c
    gold = session.gold_starts()
    predicted = sorted({0} | {s for s in _starts(answer) if 0 < s < len(turns)})
    if error:
        disagreements.append(f"boundary check failed: {error}")
    elif predicted != gold:
        disagreements.append(f"boundaries: gold starts {gold}, verifier starts {predicted}")

    segment_rows = []
    for segment, seg_turns in session.segment_turns():
        text = "\n".join(f"USER: {t['user']}\nASSISTANT: {t['assistant']}" for t in seg_turns)
        answer, c, error = _ask(client, None, TAXONOMY_PROMPT.format(tasks=task_menu, operations=operation_menu, segment=text))
        cost += c
        task_pred, op_pred = (answer or {}).get("task"), (answer or {}).get("operation")
        segment_rows.append({"segment_id": segment.segment_id, "task_pred": task_pred, "operation_pred": op_pred,
                             "task_match": task_pred == segment.task, "operation_match": op_pred == segment.operation})
        if error:
            disagreements.append(f"{segment.segment_id}: taxonomy check failed: {error}")
        else:
            if task_pred != segment.task:
                disagreements.append(f"{segment.segment_id}: gold task {segment.task}, verifier {task_pred}")
            if op_pred != segment.operation:
                disagreements.append(f"{segment.segment_id}: gold operation {segment.operation}, verifier {op_pred}")

    return {"session_id": session.seed.session_id, "model": VERIFIER_MODEL,
            "boundaries_gold": gold, "boundaries_pred": predicted, "boundaries_match": predicted == gold,
            "segments": segment_rows, "disagreements": disagreements, "cost_usd": round(cost, 6)}


def menus() -> tuple:
    """The task and operation lists the verifier chooses from."""
    task_menu = "\n".join(f"- {t.id}: {t.name}. {t.card()}" for t in sorted(load_tasks().values(), key=lambda t: t.id))
    operation_menu = "\n".join(f"- {op}: {definition}" for op, definition in load_operations().items())
    return task_menu, operation_menu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    client = make_client()
    task_menu, operation_menu = menus()

    out_path = args.data_dir / "verification.jsonl"
    done = read_jsonl_by_id(out_path)
    todo = [s for s in load_sessions(args.data_dir) if s.seed.session_id not in done]
    print(f"{len(done)} sessions already verified, {len(todo)} to go (model {VERIFIER_MODEL})")
    results = parallel_map(lambda s: verify_session(client, s, task_menu, operation_menu), todo, args.concurrency)
    with open(out_path, "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    flagged = sum(bool(r["disagreements"]) for r in results)
    print(f"{len(results)} verified, {flagged} with disagreements, ${sum(r['cost_usd'] for r in results):.4f} -> {out_path}")


if __name__ == "__main__":
    main()
