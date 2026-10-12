"""Human review of every generated session. Only accepted sessions are exported.

    python -m golden.review

For each session: the gold slots per segment, every turn with the boundaries marked, and the
verifier's disagreements. Keys: a = accept, r = reject (asks why), n = add a note, q = quit.
Writes <data dir>/review.jsonl; resumable (reviewed sessions are skipped).
"""

from __future__ import annotations

import argparse
import getpass
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from golden import DEFAULT_DATA_DIR
from golden.sessions import Session, load_sessions, read_jsonl_by_id
from golden.taxonomy import load_tasks

WRAP = 110


def show(session: Session, verification: dict, task_names: dict) -> None:
    k = 0
    print("=" * WRAP)
    print(f"{session.seed.session_id}   persona: {session.seed.persona}")
    for segment, turns in session.segment_turns():
        print("-" * WRAP)
        print(f">>> SEGMENT {segment.segment_id}  [{segment.switch_in or 'session start'}]  "
              f"{segment.n_turns} turn(s)\n    task: {segment.task}  ({task_names[segment.task]})\n"
              f"    operation: {segment.operation}   item: {segment.object_ref}   follow-ups: {segment.followup_kinds}")
        for t in turns:
            print(textwrap.fill(f"[{k}] USER: {t['user']}", WRAP, subsequent_indent="      "))
            print(textwrap.fill(f"    ASSISTANT: {t['assistant']}", WRAP, subsequent_indent="      "))
            k += 1
    print("-" * WRAP)
    problems = verification.get("disagreements") if verification else ["not verified yet"]
    print("VERIFIER: " + ("agrees on everything" if not problems else "\n  - " + "\n  - ".join(problems)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    out_path = args.data_dir / "review.jsonl"
    reviewed = read_jsonl_by_id(out_path)
    verification = read_jsonl_by_id(args.data_dir / "verification.jsonl")
    task_names = {t.id: t.name for t in load_tasks().values()}
    todo = [s for s in load_sessions(args.data_dir) if s.seed.session_id not in reviewed]
    print(f"{len(reviewed)} reviewed, {len(todo)} to go")

    with open(out_path, "a") as f:
        for session in todo:
            show(session, verification.get(session.seed.session_id), task_names)
            notes = []
            while True:
                key = input("[a]ccept / [r]eject / [n]ote / [q]uit > ").strip().lower()
                if key == "q":
                    return
                if key == "n":
                    notes.append(input("note > ").strip())
                    continue
                if key in ("a", "r"):
                    reason = input("why rejected > ").strip() if key == "r" else ""
                    f.write(json.dumps({
                        "session_id": session.seed.session_id, "decision": "accept" if key == "a" else "reject",
                        "reason": reason, "notes": notes, "reviewer": getpass.getuser(),
                        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }, ensure_ascii=False) + "\n")
                    f.flush()
                    break


if __name__ == "__main__":
    main()
