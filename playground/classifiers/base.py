"""What every classifier looks like.

A classifier implements one method, `answer(spec, state) -> Answer`: given the question (spec) and what to read
(state), return a probability for every label plus what the call cost. The base class does the rest the same way
for every classifier: fitting the input to the budget, timing, decoding the label, and turning any exception into
a recorded per-item error, so one bad item never stops a run.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass
class Answer:
    probs: Dict[str, float]
    cost_usd: Optional[float] = None          # None = unknown, never 0 by default
    cost_basis: str = "unknown"               # api_reported | estimated | free_local | unknown
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class Prediction:
    label: Optional[str]
    probs: Dict[str, float] = field(default_factory=dict)
    latency_s: Optional[float] = None
    cost_usd: Optional[float] = None
    cost_basis: str = "unknown"
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    input_meta: Dict = field(default_factory=dict)

    def to_json(self) -> Dict:
        return asdict(self)


class Classifier:
    name = "base"
    budget_tokens: Optional[int] = None       # inputs are trimmed to fit this (None = no limit)

    def config(self) -> Dict:
        return {"classifier": self.name, "budget_tokens": self.budget_tokens}

    def answer(self, spec: Dict, state: Dict) -> Answer:
        raise NotImplementedError

    def predict(self, task, row: Dict) -> Prediction:
        state, meta = task.prepare(row, self.budget_tokens)
        start = time.monotonic()
        try:
            ans = self.answer(task.spec(), state)
        except Exception as exc:  # noqa: BLE001 -- recorded per item, the run goes on
            return Prediction(None, latency_s=time.monotonic() - start, error=f"{type(exc).__name__}: {str(exc)[:300]}",
                              input_meta=meta)
        label = task.decode(ans.probs) if set(ans.probs) & set(task.labels) else None
        error = None if label in task.labels else f"invalid output: no probability on an allowed label ({sorted(ans.probs)[:3]})"
        return Prediction(label if error is None else None, {k: round(v, 5) for k, v in ans.probs.items()},
                          time.monotonic() - start, ans.cost_usd, ans.cost_basis, ans.input_tokens, ans.output_tokens,
                          error, meta)


def normalize(probs: Dict, labels) -> Dict[str, float]:
    """Probabilities over exactly `labels`, summing to 1 (missing labels get 0)."""
    clean = {lab: max(float(probs.get(lab, 0.0) or 0.0), 0.0) for lab in labels}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("no probability on any allowed label")
    return {k: v / total for k, v in clean.items()}
