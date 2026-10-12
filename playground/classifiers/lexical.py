"""No-API baselines: word-overlap rules, deterministic and free. They show what surface overlap alone achieves.

  similarity (seg)  p(new task) = 1 - cosine(new user turn, the last two turns before it)
  tfidf      (tax)  the label whose description shares the most weighted words with the segment
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

from playground.classifiers.base import Answer, Classifier, normalize

STOPWORDS = set("""a an the and or but if of to in on for with at by from as is are was were be been it this that these
those i you we they he she my our your their me us them can could would should will shall do does did have has had not
no yes so just also any all some what which who how when where why there here than then into about over please thanks
ok okay""".split())


def words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9']+", text.lower()) if w not in STOPWORDS]


def cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[w] * b[w] for w in a.keys() & b.keys())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


class Similarity(Classifier):
    name = "similarity"

    def config(self) -> Dict:
        return {**super().config(), "rule": "p(new) = 1 - cosine(new user turn, previous two turns)", "cost": "free"}

    def answer(self, spec: Dict, state: Dict) -> Answer:
        if "new_user_request" not in state:
            raise ValueError("similarity only answers the segmentation task")
        recent = " ".join(state["conversation_so_far"][-2:])
        sim = cosine(Counter(words(state["new_user_request"])), Counter(words(recent)))
        return Answer({"true": 1.0 - sim, "false": sim}, 0.0, "free_local")


class TfIdf(Classifier):
    name = "tfidf"

    def __init__(self) -> None:
        self._index: Dict[str, tuple] = {}       # label set -> (label vectors, idf), built once per spec

    def config(self) -> Dict:
        return {**super().config(), "rule": "argmax TF-IDF cosine(segment, label description)", "cost": "free"}

    def _vectors(self, criteria: Dict[str, str]) -> tuple:
        docs = {label: Counter(words(text)) for label, text in criteria.items()}
        df = Counter(w for d in docs.values() for w in d)
        idf = {w: math.log(len(docs) / n) + 1.0 for w, n in df.items()}
        return {label: Counter({w: c * idf[w] for w, c in d.items()}) for label, d in docs.items()}, idf

    def answer(self, spec: Dict, state: Dict) -> Answer:
        if "segment" not in state:
            raise ValueError("tfidf only answers the taxonomy task")
        (question,) = spec.values()
        key = "|".join(question["criteria"])
        if key not in self._index:
            self._index[key] = self._vectors(question["criteria"])
        cards, idf = self._index[key]
        seg = Counter({w: c * idf.get(w, 0.0) for w, c in Counter(words(state["segment"])).items()})
        scores = {label: cosine(seg, vec) for label, vec in cards.items()}
        if max(scores.values()) <= 0:
            scores = {label: (1.0 if label == "other" else 0.0) for label in scores}
        return Answer(normalize(scores, question["criteria"]), 0.0, "free_local")
