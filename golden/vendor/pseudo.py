"""Stable pseudonyms for identifiers before any text leaves the process.

Emails, JWT-style tokens, phone numbers, password values and tau2-style user ids become
per-session HMAC tokens such as <EMAIL_3f9a1c>. The same value within a session always maps
to the same token, so "the same order or a different one" stays visible to Jev and the
frontier, and the fixed key keeps the decision cache hitting across replays. Order,
reservation and product ids are not personal data and are kept: tasks are about them.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Callable, List, Tuple

DEFAULT_KEY = "trace-mining-blocks-v1"

_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("TOKEN", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?<![\w-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w-])")),
    ("USER", re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b")),
]
_PASSWORD = re.compile(r"""(["']?password["']?\s*[:=]\s*)(["'])(.*?)\2""", re.I)


class Pseudonymizer:
    def __init__(self, key: str = ""):
        self.key = (key or os.environ.get("TRACE_MINING_PSEUDO_KEY") or DEFAULT_KEY).encode()

    def token(self, kind: str, value: str, scope: str) -> str:
        digest = hmac.new(self.key, f"{scope}|{kind}|{value}".encode(), hashlib.sha256).hexdigest()[:6]
        return f"<{kind}_{digest}>"

    def apply(self, text: str, scope: str = "") -> str:
        if not text:
            return text

        def sub(kind: str) -> Callable[["re.Match[str]"], str]:
            return lambda m: self.token(kind, m.group(0), scope)

        out = _PASSWORD.sub(lambda m: m.group(1) + m.group(2) + self.token("PASSWORD", m.group(3), scope) + m.group(2), text)
        for kind, pat in _PATTERNS:
            out = pat.sub(sub(kind), out)
        return out
