"""Jev (typesafe/jev-1.13) through OpenRouter's Decisions API.

Request and response follow trace-mining's JevDecider: POST /alpha/decisions with {model, state, questions};
a yes/no question answers {"noul": p}, a choice answers {"probabilities": {...}}. Cost is what the API reports.
"""

from __future__ import annotations

from typing import Dict

from golden.settings import api_key
from golden.vendor.llm import LLMClient
from playground.classifiers.base import Answer, Classifier, normalize

DECISIONS_URL = "https://openrouter.ai/api"


class Jev(Classifier):
    name = "jev"
    model = "typesafe/jev-1.13"
    budget_tokens = 24_000            # Jev reads 32k tokens; the estimate errs high, so this leaves headroom

    def __init__(self) -> None:
        self.client = LLMClient(DECISIONS_URL, api_key(), timeout_s=120)   # Decisions lives under /api, not /api/v1

    def config(self) -> Dict:
        return {**super().config(), "model": self.model, "endpoint": DECISIONS_URL + "/alpha/decisions",
                "retries": "4 attempts on 429/5xx"}

    def answer(self, spec: Dict, state: Dict) -> Answer:
        data = self.client._post("/alpha/decisions", {"model": self.model, "state": state, "questions": spec})
        ((qid, question),) = spec.items()
        raw = (data.get("answers") or {}).get(qid)
        if raw is None:
            raise ValueError(f"no answer for {qid}")
        if question["type"] == "noul":
            p = float(raw["noul"])
            probs = {"true": p, "false": 1.0 - p}
        else:
            probs = normalize(raw.get("probabilities") or {raw["choice"]: 1.0}, question["criteria"])
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        return Answer(probs, float(cost) if cost is not None else None, "api_reported" if cost is not None else "unknown",
                      usage.get("input_tokens"), usage.get("output_tokens"))
