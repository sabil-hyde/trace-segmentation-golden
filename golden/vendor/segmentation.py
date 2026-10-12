"""Segmentation definition, the yes/no question asked of Jev, and the reference-labeller prompt, from
trace_mining/blocks/segment.py (unchanged)."""

from __future__ import annotations

from typing import Optional, Sequence

from golden.vendor import harness
from golden.vendor.harness import Turn
from golden.vendor.pseudo import Pseudonymizer

DEFINITION = (
    "A task is one goal the user is working toward: a feature, a bug fix, a review, an answer to a question, a "
    "document. Follow-ups, fixes, clarifications, variants and sub-steps toward that goal belong to it. A new task "
    "starts where the user turns to a different goal, even inside the same project or document: a different "
    "feature, bug, issue, review or release step, or an unrelated question. Returning to an earlier goal after "
    "another one also starts a new segment. Greetings, thanks and small talk belong to the task around them. When "
    "in doubt, do not split.")
QUESTION = {"new_task": {
    "type": "noul",
    "instructions": ("Does new_user_request begin a NEW task, or continue the task in progress at the end of "
                     "conversation_so_far? " + DEFINITION),
    "criteria": {"true": "new_user_request turns to a different goal, so a new task begins here.",
                 "false": "new_user_request continues the task in progress."},
}}
LABEL_SYSTEM = f"""You label where one task ends and the next begins in a session between a user and an AI \
assistant. Your labels are the gold standard that segmentation methods will be scored against, so be careful \
and consistent.

{DEFINITION}

Guidance:
- Judge by what the user is trying to get done, not by wording. Short replies ("yes", "try again", "and the \
tests?") continue the current task.
- "[context reset]" marks a technical reset of the assistant's context. It is not by itself a boundary.
- The excerpt may start or end in the middle of a task. Turn 0 always starts the first task.

Reply with JSON only: {{"tasks": [{{"start": <index of its first turn>, "name": "<3-8 word name>"}}, ...]}}, \
listing every task in order. The first one starts at 0."""


def label_text(turns: Sequence[Turn], pseudo: Optional[Pseudonymizer], scope: str) -> str:
    lines = []
    for i, t in enumerate(turns):
        p = " ".join((t.request or ("(first request not recorded)" if t.kind == "missing" else "")).split())[:600]
        reply = " ".join(" ".join(t.replies).split())
        a = reply[-500:] if len(reply) > 500 else reply
        tools = harness.rle([harness.short_tool(x) for x in t.tools])[:200]
        line = f"[{i}] USER: {p}" + (f"\n     ASSISTANT: {a}" if a else "") + (f"\n     TOOLS: {tools}" if tools else "")
        lines.append(pseudo.apply(line, scope) if pseudo else line)
    return "Turns of one session, in order:\n\n" + "\n".join(lines) + "\n\nList every task in this session as JSON."
