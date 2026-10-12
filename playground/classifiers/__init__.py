"""Classifier registry. To add one: subclass `Classifier`, implement `answer(spec, state)`, add it below."""

from __future__ import annotations

from typing import Optional

from playground.classifiers.base import Answer, Classifier, Prediction


def make(name: str, model: Optional[str] = None) -> Classifier:
    # imported lazily so the free baselines run without an API key
    if name == "similarity":
        from playground.classifiers.lexical import Similarity
        return Similarity()
    if name == "tfidf":
        from playground.classifiers.lexical import TfIdf
        return TfIdf()
    if name == "jev":
        from playground.classifiers.jev import Jev
        return Jev()
    if name == "llm":
        from playground.classifiers.llm import ChatLLM
        return ChatLLM(model) if model else ChatLLM()
    raise SystemExit(f"unknown classifier {name!r}: similarity, tfidf, jev, llm")


NAMES = ("similarity", "tfidf", "jev", "llm")
__all__ = ["Answer", "Classifier", "Prediction", "make", "NAMES"]
