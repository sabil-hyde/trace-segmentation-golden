"""One-off: mine how users in the open traces actually write, into a frozen style file.

    python -m golden.style_mining            # -> taxonomy/user_style-candidates.json

Reads the tau2 customer-service sessions (airline, retail, telecom) of the 100k open-trace sample,
rebuilds their user turns with trace-mining's own turn parser (vendor/harness.py), and writes:
  stats      real first-turn vs later-turn length quantiles and line-break shares
  examples   pseudonymized candidate turns per style kind (short follow-ups, pasted results,
             switch-like words, possible mid-message pivots)
The candidates are curated by hand into taxonomy/user_style-v1.json, which the generator reads.
tau2 customers are LLM-simulated; these are open traces, not human traffic.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List

import duckdb

from golden.vendor import harness
from golden.vendor.pseudo import Pseudonymizer

from golden import PACKAGE_DIR

PARQUET = Path(os.environ.get("TRACES_PARQUET", "isolated_agent_traces_100k/sample_100k.parquet"))  # the open-trace sample
OUT = PACKAGE_DIR / "taxonomy" / "user_style-candidates.json"
TAU2 = ("airline_customer_service", "retail_customer_service", "telecom_technical_support")
SWITCH_WORDS = re.compile(r"\b(also|another|one more|separately|while i have you|before we go|different|actually)\b", re.I)
PIVOT = re.compile(r"(also|one more thing|before we go further|while i have you|separately)[—–,:-]?\s+(i|i'm|i'd|can|could|my)\b", re.I)


def _openai(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """OTel GenAI parts -> OpenAI-style messages (tool results become their own messages)."""
    texts, calls, results = [], [], []
    for part in message.get("parts") or []:
        if part.get("type") == "text":
            texts.append(part.get("content") or "")
        elif part.get("type") == "tool_call":
            calls.append({"id": part.get("id"), "function": {"name": part.get("name"), "arguments": part.get("arguments")}})
        elif part.get("type") == "tool_call_response":
            result = part.get("result")
            results.append({"role": "tool", "tool_call_id": part.get("id"),
                            "content": result if isinstance(result, str) else json.dumps(result)})
    if message.get("role") == "tool":
        return results
    head = [{"role": message.get("role"), "content": "\n".join(texts), **({"tool_calls": calls} if calls else {})}]
    return (head if texts or calls else []) + results


def user_turns(con, n_sessions: int, seed: int) -> List[List[str]]:
    """Customer requests of up to n_sessions tau2 sessions (each from its longest recorded call)."""
    rows = con.execute(f"""
        select session_id, harness, input_messages_json, output_messages_json, system_instructions from (
          select *, row_number() over (partition by session_id order by length(input_messages_json) desc) rn
          from '{PARQUET}' where workstream in {TAU2}) where rn = 1
        order by md5(session_id || '{seed}') limit {n_sessions}""").fetchall()
    sessions = []
    for _, harness_name, inp, out, system in rows:
        messages = [{"role": "system", "content": system}] if system else []
        for m in json.loads(inp or "[]") + json.loads(out or "[]"):
            messages += _openai(m)
        turns, _ = harness.parse([harness.trim_message(m) for m in messages], "tau2", harness_name)
        requests = [t.request.strip() for t in turns if t.request and t.request.strip()]
        if len(requests) >= 2:
            sessions.append(requests)
    return sessions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, default=1500)
    ap.add_argument("--per-kind", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    sessions = user_turns(duckdb.connect(), args.sessions, args.seed)
    first = [s[0] for s in sessions]
    later = [t for s in sessions for t in s[1:]]
    pseudo = Pseudonymizer()

    def quantiles(texts: List[str]) -> Dict[str, int]:
        q = statistics.quantiles([len(t) for t in texts], n=10)
        return {"p10": int(q[0]), "p50": int(q[4]), "p90": int(q[8])}

    def sample(texts: List[str]) -> List[str]:
        picked = random.Random(args.seed).sample(texts, min(args.per_kind, len(texts)))
        return [pseudo.apply(t[:300], "style") for t in picked]

    candidates = {
        "short_followup": [t for t in later if len(t) < 120],
        "pasted_followup": [t for t in later if t.count("\n") >= 2 or re.search(r"(^|\n)\s*(-|\d\.)\s", t)],
        "switch_words_followup": [t for t in later if SWITCH_WORDS.search(t)],
        "possible_pivot": [t for t in later if PIVOT.search(t)],
    }
    out = {
        "source": f"{PARQUET.name}: tau2 airline/retail/telecom customer turns (LLM-simulated customers)",
        "sessions_used": len(sessions),
        "stats": {"first_turn_chars": quantiles(first), "later_turn_chars": quantiles(later),
                  "first_turn_multiline_share": round(sum("\n" in t for t in first) / len(first), 3),
                  "later_turn_multiline_share": round(sum("\n" in t for t in later) / len(later), 3)},
        "examples": {kind: sample(texts) for kind, texts in candidates.items()},
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{len(sessions)} sessions -> {OUT}\n{json.dumps(out['stats'], indent=1)}")
    print({k: len(v) for k, v in candidates.items()})


if __name__ == "__main__":
    main()
