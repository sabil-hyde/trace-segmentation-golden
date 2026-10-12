"""Any OpenRouter chat model answering the same spec as Jev, as JSON probabilities.

Default anthropic/claude-haiku-5.5: a cheap reference that is neither Gemini (which wrote the data) nor GPT
(which verified it). Change it with `--model`.
"""

from __future__ import annotations

import json
from typing import Dict

from golden.settings import client as make_client
from playground.classifiers.base import Answer, Classifier, normalize

SYSTEM = """You answer one structured question about a STATE object.
Judge only from the evidence in STATE. Give a probability for every label (they must sum to 1) that reflects
how confident you are. Reply with JSON only: {"probabilities": {"<label>": <float>, ...}}"""


class ChatLLM(Classifier):
    name = "llm"
    budget_tokens = 150_000

    def __init__(self, model: str = "anthropic/claude-haiku-5.5") -> None:
        self.model = model
        self.client = make_client()

    def config(self) -> Dict:
        return {**super().config(), "model": self.model, "temperature": 0, "system_prompt": SYSTEM}

    def answer(self, spec: Dict, state: Dict) -> Answer:
        ((_, question),) = spec.items()
        labels = question["criteria"]
        if question["type"] == "noul":
            labels = {"true": labels["true"], "false": labels["false"]}
        prompt = json.dumps({"question": question["instructions"], "labels": labels, "STATE": state},
                            ensure_ascii=False, indent=1)
        res = self.client.chat(self.model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                               temperature=0.0, json_mode=True)
        if res.error:
            raise RuntimeError(res.error)
        text = res.content.strip()
        start, end = text.find("{"), text.rfind("}")
        probs = json.loads(text[start:end + 1])["probabilities"]
        usage = res.usage or {}
        cost = usage.get("cost")
        return Answer(normalize(probs, labels), float(cost) if cost is not None else None,
                      "api_reported" if cost is not None else "unknown",
                      usage.get("prompt_tokens"), usage.get("completion_tokens"))
