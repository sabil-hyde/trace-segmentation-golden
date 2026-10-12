"""Where model calls go. Everything goes through one OpenAI-compatible endpoint (OpenRouter by default)."""

from __future__ import annotations

import os

from golden import REPO_DIR
from golden.vendor.llm import LLMClient

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY_ENV = "OPENROUTER_API_KEY"
TIMEOUT_S = 120


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from the repo's .env without overriding the real environment."""
    path = REPO_DIR / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def client() -> LLMClient:
    _load_dotenv()
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise SystemExit(f"{API_KEY_ENV} is not set: put it in the environment or in {REPO_DIR / '.env'}")
    return LLMClient(BASE_URL, key, timeout_s=TIMEOUT_S)
