"""The two tasks. A task turns a dataset row into what a classifier reads (state) and asks (spec), and
turns the classifier's probabilities back into a label.

  seg  SegTask(context="full" | "last2")             does the new user turn start a new task?   (yes/no)
  tax  TaxTask(level="task" | "industry",            which task / industry is this segment?     (choice)
               view="segment" | "first_turn")

Specs use the Decisions API format every classifier understands:
  {qid: {"type": "noul" | "choice", "instructions": str, "criteria": {label: description}}}
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from golden.taxonomy import load_tasks
from golden.vendor.segmentation import QUESTION

CHARS_PER_TOKEN = 3.25          # same pre-send estimate trace-mining uses for Jev (errs high)
OTHER = "other"


def estimate_tokens(state: Dict, spec: Dict) -> int:
    return int(len(json.dumps(state, ensure_ascii=False) + json.dumps(spec)) / CHARS_PER_TOKEN)


def spec_sha(spec: Dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]


class SegTask:
    name = "seg"
    labels = ["true", "false"]

    def __init__(self, context: str = "full", threshold: float = 0.5):
        if context not in ("full", "last2"):
            raise ValueError("context must be full or last2")
        self.context, self.threshold = context, threshold

    def spec(self) -> Dict:
        return QUESTION

    def gold(self, row: Dict) -> str:
        return row["gold"]

    def prepare(self, row: Dict, budget_tokens: Optional[int] = None) -> Tuple[Dict, Dict]:
        """The state, trimmed from the oldest turn until it fits the classifier's budget."""
        turns = list(row["conversation_so_far"])
        if self.context == "last2":
            turns = turns[-2:]
        state = {"conversation_so_far": turns, "new_user_request": row["new_user_request"]}
        trimmed = 0
        while budget_tokens and estimate_tokens(state, self.spec()) > budget_tokens and len(state["conversation_so_far"]) > 1:
            state["conversation_so_far"] = state["conversation_so_far"][1:]
            trimmed += 1
        return state, {"context_turns_used": len(state["conversation_so_far"]), "trimmed_turns": trimmed,
                       "est_input_tokens": estimate_tokens(state, self.spec())}

    def decode(self, probs: Dict[str, float]) -> str:
        return "true" if probs.get("true", 0.0) >= self.threshold else "false"

    def slices(self, row: Dict) -> Dict:
        keys = ("switch_in", "transition_style", "followup_kind", "user_len_bucket", "segment_len_before", "segment_len_after")
        return {k: row.get(k) for k in keys}

    def config(self) -> Dict:
        return {"task": self.name, "context": self.context, "threshold": self.threshold, "n_classes": 2,
                "spec_sha": spec_sha(self.spec())}


class TaxTask:
    name = "tax"

    def __init__(self, level: str = "task", view: str = "segment"):
        if level not in ("task", "industry") or view not in ("segment", "first_turn"):
            raise ValueError("level must be task|industry and view segment|first_turn")
        self.level, self.view = level, view
        self.tasks = load_tasks()
        self.industry_of = {t.id: t.industry for t in self.tasks.values()}
        self.cards = self._cards()
        self.labels: List[str] = list(self.cards) + [OTHER]

    def _cards(self) -> Dict[str, str]:
        tasks = sorted(self.tasks.values(), key=lambda t: t.id)
        if self.level == "task":
            return OrderedDict((t.id, f"{t.name}. {t.card()}") for t in tasks)
        cards: Dict[str, List[str]] = OrderedDict()
        for t in tasks:
            cards.setdefault(t.industry, []).append(t.name)
        return OrderedDict((ind, f"Work in {ind.replace('_', ' ')}, e.g. " + "; ".join(names) + ".")
                           for ind, names in cards.items())

    def spec(self) -> Dict:
        what = "task type" if self.level == "task" else "industry"
        options = "\n".join(f"- {label}: {card}" for label, card in self.cards.items())
        criteria = dict(self.cards, **{OTHER: f"None of the listed {what}s fits this segment."})
        return {"label": {"type": "choice", "criteria": criteria,
                          "instructions": (f"This chat segment is about one request. Which {what} is it? Options:\n"
                                           f"{options}\n- {OTHER}: none of these fits.")}}

    def gold(self, row: Dict) -> str:
        return row["gold"][self.level]

    def prepare(self, row: Dict, budget_tokens: Optional[int] = None) -> Tuple[Dict, Dict]:
        """The segment text (or its first user turn), cut in the middle until it fits the budget."""
        text = row["text"] if self.view == "segment" else row["text"].split("\n", 1)[0]
        full = len(text)
        state = {"segment": text}
        while budget_tokens and estimate_tokens(state, self.spec()) > budget_tokens and len(state["segment"]) > 400:
            keep = int(len(state["segment"]) * 0.8) // 2
            state["segment"] = state["segment"][:keep] + "\n[... middle omitted ...]\n" + state["segment"][-keep:]
        return state, {"text_chars": full, "text_chars_used": len(state["segment"]),
                       "est_input_tokens": estimate_tokens(state, self.spec())}

    def decode(self, probs: Dict[str, float]) -> str:
        return max(probs, key=probs.get) if probs else ""

    def industry(self, label: str) -> str:
        """Industry of a task-level label (an industry label is returned as is)."""
        return self.industry_of.get(label, label)

    def slices(self, row: Dict) -> Dict:
        return {"industry": row["gold"]["industry"], "task": row["gold"]["task"], "n_turns": row.get("n_turns")}

    def config(self) -> Dict:
        return {"task": self.name, "level": self.level, "view": self.view, "n_classes": len(self.labels),
                "spec_sha": spec_sha(self.spec())}


def make_task(name: str, context: str = "full", threshold: float = 0.5, level: str = "task", view: str = "segment"):
    return SegTask(context, threshold) if name == "seg" else TaxTask(level, view)
