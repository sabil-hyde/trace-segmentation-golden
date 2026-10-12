"""Write the text for every seeded segment. The model only writes text; every label comes from
seeds.jsonl and is never changed here.

    python -m golden.generate --limit 3      # try 3 sessions first
    python -m golden.generate                # the rest (resumes)

Writes <data dir>/generations.jsonl, one session per line. Each segment is checked in code
(exact turn count, no empty turn, no taxonomy names, no meta phrases, each user turn's length
bucket and shape) and
regenerated up to MAX_ATTEMPTS times with the reasons; then it is marked generation_failed.

After every batch the new sessions are verified blind (verification.jsonl, skip with --no-verify)
and the quality checks in checks.py are printed and appended to quality.jsonl. Checks only warn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from golden.settings import client as make_client
from golden.vendor.llm import LLMClient, parallel_map

from golden import DEFAULT_DATA_DIR, PACKAGE_DIR, checks, verify
from golden.seeds import SegmentSeed, SessionSeed, read_seeds
from golden.sessions import Session
from golden.taxonomy import Task, category_names, load_compatibility, load_operations, load_tasks

GENERATOR_MODEL = "google/gemini-3.8-flash"
TEMPERATURE = 0.8
MAX_ATTEMPTS = 4                 # calls per segment: full generations and cheaper repairs together
BANNED_USER_PHRASES = ("new task", "new topic", "change of topic", "switching gears", "different topic")
INTERNAL_REF = re.compile(r"REF-\d{5}")    # the seed's object_ref; printing it would mark every boundary
# Accepted user-turn length per target bucket. The upper end is tight (~1.35x) because the trial showed openers
# drifting to the top of wide ranges, which made them longer than follow-ups of the same bucket.
LENGTH_RANGE = {50: (25, 70), 100: (70, 135), 200: (140, 270), 400: (280, 540), 800: (560, 1080)}
STYLE_FILE = PACKAGE_DIR / "taxonomy" / "user_style-v1.json"

SYSTEM = ("You write realistic, fully fictional chat transcripts between a user and an AI assistant, used as "
          "benchmark data. Follow the turn plan exactly, including every length and shape. Reply with JSON only.")

# What a task's request must contain. The starter-pack definitions alone let the generator drift for these tasks
# (golden-v2 review: a catalog search became a one-item eligibility check, an option comparison a one-vendor check,
# a code review a change-ticket gate).
TASK_HINTS = {
    "retail_and_ecommerce.task.product_search": "The user supplies several catalog items (a list or table) and asks which "
                                                "of them satisfy the customer's constraints.",
    "professional_services.task.option_analysis": "The user supplies two or more options and asks how they compare "
                                                  "against the stated objectives or criteria.",
    "software_and_it.task.code_review": "The user supplies a code change (a diff or code excerpt) and asks for defects "
                                        "or requirement gaps in it.",
}

FOLLOWUP_MEANING = {
    "clarify": "asks the assistant to explain or clarify part of its previous reply",
    "provide_info": "adds a detail about the same item that changes or confirms the answer",
    "correct": "points out an error or a wrong assumption in the previous reply and asks for it to be fixed",
    "refine": "asks to adjust the format, length, order or emphasis of the same output",
    "confirm": "accepts the result and asks one brief closing check about it",
    "ask_detail": "asks a follow-up question about one specific detail of the same item",
    "paste_more": "pastes further material about the same item (another part of the record, log, table or rules)",
    "decoy": ("brings up something that SOUNDS new - another document, a person's name, a new number or date, or "
              "words like 'also', 'another', 'one more thing', 'actually' - but it is about the SAME item and the "
              "SAME goal, so the work continues"),
}

SWITCH_MEANING = {
    "same_task_new_object": "the same kind of work, on a different, unrelated item",
    "same_industry_other_task": "a different kind of work, in the same industry, on a different item",
    "cross_industry": "a different client in a different industry",
}

# The opening words themselves come from the seed's lead-in slot (seeds.LEAD_IN_POOLS), never from examples here:
# the trial showed the model copying example phrases verbatim, which made the first words mark every boundary.
TRANSITION_MEANING = {
    "explicit": "After the given opening words, go straight to the new matter.",
    "no_marker": "Just start talking about the new item. No transition words, no thanks, no reference to before.",
    "blended": ("The given opening words react to the assistant's last reply; then, in the SAME message, move on to "
                "the new item without announcing it."),
    "misleading_cue": "The given opening words sound like a continuation, but the message is about the new item.",
}

PROMPT = """Write ONE segment of a longer chat: exactly {n_turns} user turn(s), each followed by the assistant's reply.

THE USER: {persona}.

THE ONE GOAL OF THIS SEGMENT
- Kind of work: {task_card}
- What the request needs the assistant to do: {operation} - {operation_definition}
- The item: a new, specific {industry} item, different from anything earlier in the chat. Invent realistic details
  (names, figures, dates, rules) and, if the item needs an identifier, one in the style that industry really uses.
  Every person and company is fictional.

TURN PLAN (lengths are characters of the USER message: aim at the target number, the first turn included - do not
drift to the top of the range; "pasted" = several lines such as a list, table, log or excerpt, written with \\n line
breaks inside the JSON string; "prose" = a normal message with at most one line break)
{turn_plan}

RULES
- Turn 1 introduces this item and goal. The material the assistant needs may come in turn 1 or in later turns.
- This segment is not a step or variant of anything earlier in the chat, and never reuses an earlier item. Never
  use batch phrasing like "same for", "next one", "now this one" or "do it again".
- Every later user turn stays on this goal and this item and needs the same kind of answer. Follow-ups may be
  long or short and may paste material; they never start other work.
- Write the way real users type: uneven, sometimes terse, sometimes chatty, not polished. Real examples of the
  register (tone and formatting only; different domain, never copy their content):
{style}
- The assistant has no tools. It only analyses or drafts from what the user wrote, and never claims to have
  sent, booked, changed or executed anything. Its replies can be as long as the answer needs.
- Unless the request needs code, neither the user nor the assistant asks for or writes executable code,
  commands or configuration.{single_turn_rule}
- Never talk about the conversation itself (no "task", "topic", "new request", "switching").
{context}
Reply with JSON only: {{"turns": [{{"user": "...", "assistant": "..."}}]}} with exactly {n_turns} item(s)."""


def _style_examples(segment: SegmentSeed) -> str:
    """Three to four real user turns (fixed per segment) as register references."""
    examples = json.loads(STYLE_FILE.read_text())["examples"]
    rng = random.Random(segment.segment_id)
    kinds = ["short_followup", "pasted_followup", "followup_with_switch_words"]
    if segment.transition_style in ("blended", "misleading_cue"):
        kinds.append("pivot_inside_message")
    return "\n".join("  > " + rng.choice(examples[k]).replace("\n", " / ")[:240] for k in kinds)


def build_prompt(session: SessionSeed, segment: SegmentSeed, task: Task, operations: Dict[str, str],
                 previous_turns: List[Dict[str, str]]) -> str:
    def size(i: int) -> str:
        low, high = LENGTH_RANGE[segment.user_lengths[i]]
        return f"USER ~{segment.user_lengths[i]} chars ({low}-{high}), {segment.turn_shapes[i]}"

    def lead(i: int) -> str:
        words = segment.lead_ins[i] if segment.lead_ins else ""
        return f' Begin the user message with exactly "{words}".' if words else ""

    opening = "introduces this item and goal"
    if segment.transition_style:
        opening += " - " + TRANSITION_MEANING[segment.transition_style]
    plan = [f"- Turn 1 [{size(0)}]: {opening}.{lead(0)}"]
    plan += [f"- Turn {i + 2} [{size(i + 1)}]: the user {FOLLOWUP_MEANING[kind]}.{lead(i + 1)}"
             for i, kind in enumerate(segment.followup_kinds)]
    context = ""
    if previous_turns:
        lines = "\n".join(f"USER: {t['user']}\nASSISTANT: {t['assistant']}" for t in previous_turns)
        context = (f"\nEARLIER IN THIS CHAT (context only; do not continue it):\n{lines}\n"
                   f"In this segment the user moves on to {SWITCH_MEANING[segment.switch_in]}.\n")
    return PROMPT.format(
        n_turns=segment.n_turns, persona=session.persona, operation=segment.operation,
        task_card=task.card() + (" " + TASK_HINTS[task.id] if task.id in TASK_HINTS else ""),
        operation_definition=operations[segment.operation], industry=segment.industry.replace("_", " "),
        turn_plan="\n".join(plan), context=context, style=_style_examples(segment),
        single_turn_rule=("\n- There is only one turn, so the assistant's reply fully completes the request."
                          if segment.n_turns == 1 else ""))


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def starts_with(text: str, lead_in: str) -> bool:
    """Does the message begin with the lead-in's words (case and punctuation ignored)?"""
    want = _words(lead_in)
    return _words(text)[: len(want)] == want


def check_turns(raw: str, n_turns: int, names: List[str], user_lengths: Optional[List[int]] = None,
                turn_shapes: Optional[List[str]] = None,
                lead_ins: Optional[List[str]] = None) -> Tuple[Optional[List[Dict[str, str]]], List[str]]:
    """Parse the model's JSON and return (turns, problems). Turns are None when unusable.
    With user_lengths / turn_shapes / lead_ins, every user turn must also hit its length range, shape and
    opening words."""
    try:
        turns = json.loads(raw, strict=False)["turns"]       # pasted blocks may carry raw line breaks
    except (ValueError, KeyError, TypeError):
        return None, ["reply is not JSON with a 'turns' list"]
    if not isinstance(turns, list) or len(turns) != n_turns:
        return None, [f"expected exactly {n_turns} turns, got {len(turns) if isinstance(turns, list) else 'none'}"]
    problems: List[str] = []
    clean: List[Dict[str, str]] = []
    for i, t in enumerate(turns, 1):
        t = t if isinstance(t, dict) else {}
        user, reply = str(t.get("user") or "").strip(), str(t.get("assistant") or "").strip()
        clean.append({"user": user, "assistant": reply})
        if not user or not reply:
            problems.append(f"turn {i} has an empty user or assistant message")
            continue
        low_user, low_all = user.lower(), (user + " " + reply).lower()
        problems += [f"turn {i}: user says {p!r}" for p in BANNED_USER_PHRASES if p in low_user]
        problems += [f"turn {i}: mentions the category name {n!r}" for n in names if n.lower() in low_all]
        if INTERNAL_REF.search(user + reply):
            problems.append(f"turn {i}: prints the internal seed id (REF-#####)")
        if user_lengths:
            low, high = LENGTH_RANGE[user_lengths[i - 1]]
            if not low <= len(user) <= high:
                problems.append(f"turn {i}: user message has {len(user)} chars, needs {low}-{high}")
        if turn_shapes:
            breaks = user.count("\n")
            # loose on purpose: one off-by-one line break should not throw away a 20-turn segment;
            # the shortcut checks watch whether line breaks still predict boundaries
            if turn_shapes[i - 1] == "pasted" and breaks < 1:
                problems.append(f"turn {i}: user message must be a pasted block (line breaks)")
            if turn_shapes[i - 1] == "prose" and breaks > 2:
                problems.append(f"turn {i}: user message must be prose (at most 2 line breaks)")
        if lead_ins and lead_ins[i - 1] and not starts_with(user, lead_ins[i - 1]):
            problems.append(f"turn {i}: user message must begin with {lead_ins[i - 1]!r}")
    return clean, problems


REPAIR = """Below is a chat segment as JSON. Rewrite ONLY the user messages of the turns listed, so each meets its
requirement. Keep each message's meaning and its role in the conversation, so the assistant replies around it still fit.
{fixes}

SEGMENT:
{segment}

Reply with JSON only: {{"user_turns": {{"<turn number>": "<new user message>"}}}}"""

REPAIRABLE = re.compile(r"^turn (\d+): user message (has \d+ chars|must be|must begin)")


def repair_request(turns: List[Dict[str, str]], problems: List[str], segment: SegmentSeed) -> Optional[str]:
    """A prompt that fixes only the user turns with length/shape problems, or None when anything else is wrong
    (then the whole segment is regenerated)."""
    matches = [REPAIRABLE.match(p) for p in problems]
    if not all(matches):
        return None
    fixes = []
    for n in sorted({int(m.group(1)) for m in matches}):
        low, high = LENGTH_RANGE[segment.user_lengths[n - 1]]
        shape = ("a pasted block: several lines (list, table, log or excerpt) with line breaks"
                 if segment.turn_shapes[n - 1] == "pasted" else "prose, at most one line break")
        lead = segment.lead_ins[n - 1] if segment.lead_ins else ""
        fixes.append(f"- turn {n}: {low}-{high} characters (now {len(turns[n - 1]['user'])}), {shape}"
                     + (f', beginning with exactly "{lead}"' if lead else ""))
    return REPAIR.format(fixes="\n".join(fixes), segment=json.dumps({"turns": turns}, ensure_ascii=False, indent=1))


def generate_segment(client: LLMClient, prompt: str, segment: SegmentSeed, names: List[str]) -> Dict:
    """Generate, check, and fix. When only some user turns miss their length or shape, just those turns are
    rewritten; anything else regenerates the segment with the reasons. `problems` keeps the reasons of every
    rejected attempt, also when a later attempt succeeds, so the quality checks see which rule keeps breaking."""
    attempts, rejected, usage_total, cost = 0, [], {"prompt_tokens": 0, "completion_tokens": 0}, 0.0
    message, turns, repair = prompt, None, None
    for attempts in range(1, MAX_ATTEMPTS + 1):
        res = client.chat(GENERATOR_MODEL, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": repair or message}],
                          temperature=TEMPERATURE, json_mode=True)
        for k in usage_total:
            usage_total[k] += int((res.usage or {}).get(k) or 0)
        cost += float((res.usage or {}).get("cost") or 0.0)
        if res.error:
            rejected.append(f"API error: {res.error[:200]}")
            continue
        raw = res.content
        if repair:                                   # merge the rewritten user turns into the kept segment
            try:
                fixed = json.loads(raw, strict=False)["user_turns"]
                merged = [dict(t, user=str(fixed.get(str(i), t["user"]))) for i, t in enumerate(turns, 1)]
                raw = json.dumps({"turns": merged})
            except (ValueError, KeyError, TypeError, AttributeError):
                rejected.append("repair reply is not JSON with 'user_turns'")
                continue
        checked, problems = check_turns(raw, segment.n_turns, names, segment.user_lengths, segment.turn_shapes,
                                        segment.lead_ins)
        if checked is not None and not problems:
            return {"status": "ok", "turns": checked, "attempts": attempts, "problems": rejected,
                    "usage": usage_total, "cost_usd": round(cost, 6)}
        rejected += problems
        repair = repair_request(checked, problems, segment) if checked is not None else None
        turns = checked if repair else None
        message = prompt + "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED FOR: " + "; ".join(problems[:8]) + ". Fix all of these."
    return {"status": "generation_failed", "turns": None, "attempts": attempts, "problems": rejected,
            "usage": usage_total, "cost_usd": round(cost, 6)}


def network_failure(result: Dict) -> bool:
    """True when a segment failed only because calls could not reach the model (no content was rejected)."""
    return any(g["status"] != "ok" and g["problems"] and all(p.startswith("API error") for p in g["problems"])
               for g in result["segments"])


def generate_session(client: LLMClient, session: SessionSeed, tasks: Dict[str, Task], operations: Dict[str, str],
                     names: List[str]) -> Dict:
    segments, previous = [], []
    for segment in session.segments:
        prompt = build_prompt(session, segment, tasks[segment.task], operations, previous[-2:])
        result = generate_segment(client, prompt, segment, names)
        segments.append({"segment_id": segment.segment_id, "model": GENERATOR_MODEL, "temperature": TEMPERATURE,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), **result})
        previous = result["turns"] or previous
    return {"session_id": session.session_id, "segments": segments}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--limit", type=int, default=0, help="generate at most this many new sessions")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--unreviewed-ok", action="store_true",
                    help="trial runs only (use a separate --data-dir): allow a not-yet-reviewed operations table")
    ap.add_argument("--no-verify", action="store_true", help="skip the blind verifier after each batch")
    args = ap.parse_args()

    load_compatibility(require_reviewed=not args.unreviewed_ok)   # gold must come from a reviewed table
    client = make_client()
    tasks, operations, names = load_tasks(), load_operations(), category_names()

    out_path = args.data_dir / "generations.jsonl"
    done = {json.loads(l)["session_id"] for l in out_path.read_text().splitlines()} if out_path.exists() else set()
    todo = [s for s in read_seeds(args.data_dir / "seeds.jsonl") if s.session_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} sessions already generated, {len(todo)} to go (model {GENERATOR_MODEL})")

    menus = verify.menus()
    total_cost, failed = 0.0, 0
    for start in range(0, len(todo), args.concurrency):        # append per batch, so a crash loses little
        batch = todo[start: start + args.concurrency]
        results = parallel_map(lambda s: generate_session(client, s, tasks, operations, names), batch, args.concurrency)
        with open(out_path, "a") as f:
            for r in results:
                total_cost += sum(g["cost_usd"] for g in r["segments"])
                failed += sum(g["status"] != "ok" for g in r["segments"])
                if not network_failure(r):         # sessions lost to an outage are retried on the next run
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n{len(done) + start + len(batch)} sessions generated (+{len(batch)}), ${total_cost:.4f} this run, "
              f"{failed} failed segments", flush=True)
        if all(g["status"] != "ok" for r in results for g in r["segments"]):
            raise SystemExit("every segment in this batch failed: check the API key / gateway before spending more")

        if not args.no_verify:
            usable = [s for s in (Session(seed, r) for seed, r in zip(batch, results)) if s.ok]
            checked = parallel_map(lambda s: verify.verify_session(client, s, *menus), usable, args.concurrency)
            with open(args.data_dir / "verification.jsonl", "a") as f:
                for v in checked:
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
        report = checks.run_checks(args.data_dir)                  # warn only; Ctrl-C to stop, re-run resumes
        checks.record(args.data_dir, report, len(done) + start + len(batch))
        print(checks.format_table(report), flush=True)
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
