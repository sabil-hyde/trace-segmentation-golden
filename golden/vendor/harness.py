"""Harness adapters: an agent session's LLM-call messages -> its user turns.

Every call resends the conversation so far, so a session's messages are the input of its
longest call (plus that call's output). Harnesses encode the user differently:

  tau2 customer service (a simulated customer, several requests per session)
    claude_code      request 0: "First Observation: sender='user' message='...'" in msg 0;
                     later: role=tool results of the message tool, JSON {"sender": "user", ...}
    openai_solo      request 0: the same marker in the system message (a JSON string of parts);
                     later: tool results, JSON or Python repr. gpt-5.2 runs carry no request 0.
    smolagents_code  request 0: the marker in the task message; later: user "Observation: ...
                     sender='user' message='...'"; the agent speaks through message(...) in code.
    tool_calling     msg 0 is the policy (as a user message); every later user text is the customer.
  single-instruction benchmarks (AppWorld, BrowseComp, SWE-bench): one request, found after
    "Task from supervisor:", "Question:" or "## Issue to resolve:" in the opening messages.

`trim_message` produces the compact, content-addressed form the spool stores; `parse`
is a pure function of a message list, so the turns (and every Jev state built from them)
are the same whatever order the calls arrive in.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TAU2 = {"airline_customer_service", "retail_customer_service", "telecom_technical_support"}
SINGLE = {"software_engineering", "deep_research", "personal_assistant"}

# Per-role caps for the spool. Head + tail are kept (U-shaped) so markers at either end
# survive: the tau2 opening request sits after an ~8k-char policy.
TEXT_CAP = {"system": 20_000, "user": 20_000, "developer": 2_000, "assistant": 6_000, "tool": 3_000}
OBSERVATION_CAP = 3_000       # smolagents tool observations arrive as user messages
ARGS_CAP = 4_000

MESSAGE_TOOLS = ("message", "send_message", "respond")
FINISH_TOOLS = ("submit", "finish", "complete_task", "final_answer", "done")
PSEUDO_USER = ("<system-reminder>", "Observation:", "Call id:", "Error: Sending a message is not allowed",
               "Your response was cut off", "This session is being continued")
SIDE_CALL_PREFIXES = ("Perform a web search for the query:", "<policy_spec>", "Command:",
                      "This session is being continued")
COMPACTION_PREFIX = "This session is being continued"
INSTRUCTION_MARKERS = ("First Observation: sender=", "## Issue to resolve:", "Task from supervisor:",
                       "Question:", "<policy>", "'policy':")
_BUILTINS = {"print", "len", "str", "int", "float", "range", "list", "dict", "set", "sorted", "sum", "min",
             "max", "enumerate", "zip", "json", "round", "abs", "any", "all", "isinstance", "type", "tuple",
             "format", "append", "get", "items", "keys", "values", "split", "join", "strip", "loads", "dumps",
             "lower", "upper", "replace", "startswith", "endswith", "findall", "search", "match", "sub",
             "extend", "pop", "update", "count", "index", "bool", "map", "filter", "reversed", "next", "iter"}


# ------------------------------------------------------------------ trimming

def _u_trim(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    half = cap // 2
    return text[:half] + f"\n[... {len(text) - cap} chars elided ...]\n" + text[-half:]


def _join_parts(parts: Iterable[Any]) -> str:
    out = []
    for p in parts:
        if isinstance(p, dict):
            if p.get("type") in ("thinking", "reasoning", "redacted_thinking"):
                continue
            t = p.get("text") if p.get("text") is not None else p.get("content")
            if t is not None:
                out.append(t if isinstance(t, str) else json.dumps(t, ensure_ascii=False))
        elif isinstance(p, str):
            out.append(p)
    return "\n".join(out)


def _strip_thinking(text: str) -> str:
    """The OTLP converter serializes thinking parts as one-line JSON objects: drop them."""
    if '"thinking"' not in text:
        return text
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("{") and '"type"' in s and "thinking" in s[:40]:
            try:
                if json.loads(s).get("type") in ("thinking", "reasoning", "redacted_thinking"):
                    continue
            except ValueError:
                pass
        kept.append(line)
    return "\n".join(kept)


def content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        s = content.strip()
        if s.startswith("[{"):
            try:
                parts = json.loads(s)
            except ValueError:
                return content
            if isinstance(parts, list):
                return _join_parts(parts)
        return content
    if isinstance(content, list):
        return _join_parts(content)
    if isinstance(content, dict):
        return _join_parts([content])
    return str(content)


def trim_message(m: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI-style message -> {r, t, tc?, id?}; stable, so equal history gives equal hashes."""
    role = str(m.get("role") or "user")
    text = content_text(m.get("content"))
    if role == "assistant":
        text = _strip_thinking(text)
    cap = OBSERVATION_CAP if role == "user" and text.startswith("Observation:") else TEXT_CAP.get(role, 6_000)
    out: Dict[str, Any] = {"r": role, "t": _u_trim(text, cap)}
    calls = []
    for tc in m.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, sort_keys=True)
        calls.append([str(tc.get("id") or ""), str(fn.get("name") or ""), _u_trim(args, ARGS_CAP)])
    if calls:
        out["tc"] = calls
    if m.get("tool_call_id"):
        out["id"] = str(m["tool_call_id"])
    return out


def message_hash(msg: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(msg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def trim_output(output: Any) -> Optional[Dict[str, Any]]:
    """A call's output ({"message": {...}, "content": ...}) -> one trimmed assistant message."""
    if not isinstance(output, dict):
        return None
    msg = output.get("message") if isinstance(output.get("message"), dict) else None
    if msg is None:
        return {"r": "assistant", "t": _u_trim(content_text(output.get("content")), TEXT_CAP["assistant"])}
    return trim_message({**msg, "role": "assistant"})


def tool_names(model_parameters: Any) -> List[str]:
    tools = (model_parameters or {}).get("tools") if isinstance(model_parameters, dict) else None
    names = []
    for t in tools or []:
        if isinstance(t, dict):
            n = t.get("name") or (t.get("function") or {}).get("name")
            if n:
                names.append(str(n))
    return names


# ------------------------------------------------------------------ small parsers

def _quoted_literal(text: str, start: int) -> Optional[str]:
    """The Python string literal starting at text[start] (a quote), decoded."""
    if start >= len(text) or text[start] not in "'\"":
        return None
    q = text[start]
    i = start + 1
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == q:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return ast.literal_eval(text[start:i + 1])
            except (ValueError, SyntaxError):
                return text[start + 1:i]
        i += 1
    return text[start + 1:]           # unterminated (trimmed): keep what there is


_SENDER_RE = re.compile(r"sender=['\"]user['\"]\s*,?\s*message=")


def sender_user_message(text: str) -> Optional[str]:
    """A simulated-customer message in JSON or Python repr form, else None."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("{") and '"sender"' in s[:80]:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and obj.get("sender") == "user" and "message" in obj:
                return str(obj.get("message") or "")
        except ValueError:
            m = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)', s, re.S)
            if m and re.search(r'"sender"\s*:\s*"user"', s):
                try:
                    return json.loads('"' + m.group(1) + '"')
                except ValueError:
                    return m.group(1)
    m = _SENDER_RE.search(text)
    if m:
        return _quoted_literal(text, m.end())
    return None


def first_observation(text: str) -> Optional[str]:
    i = text.find("First Observation:")
    if i < 0:
        return None
    return sender_user_message(text[i:])


def _loads_args(args: str) -> Any:
    try:
        return json.loads(args)
    except (ValueError, TypeError):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ast.literal_eval(args)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None


def _message_content(args: str) -> str:
    obj = _loads_args(args)
    if isinstance(obj, dict):
        for k in ("content", "message", "text"):
            if isinstance(obj.get(k), str):
                return obj[k]
    m = re.search(r"['\"]content['\"]\s*:\s*['\"](.*)", args or "", re.S)
    return m.group(1).rstrip("\"'} \n") if m else ""


def short_tool(name: str) -> str:
    n = name.split("functions.")[-1]
    n = n.split(":")[0]
    for p in ("mcp__environment__", "environment__"):
        if n.startswith(p):
            n = n[len(p):]
    return n


def _is_message_tool(name: str) -> bool:
    return short_tool(name) in MESSAGE_TOOLS


def _is_finish_tool(name: str) -> bool:
    n = short_tool(name).lower()
    return any(n == f or n.endswith("__" + f) or n.endswith("_" + f) for f in FINISH_TOOLS)


def smol_code(msg: Dict[str, Any]) -> str:
    """The code a smolagents step ran: the "code" field of its JSON blob, else the interpreter call."""
    t = msg.get("t") or ""
    s = t.lstrip()
    if s.startswith("{"):
        try:
            obj, _ = json.JSONDecoder().raw_decode(s)
            if isinstance(obj, dict) and isinstance(obj.get("code"), str):
                return obj["code"]
        except ValueError:
            m = re.search(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', s, re.S)
            if m:
                try:
                    return json.loads('"' + m.group(1) + '"')
                except ValueError:
                    return m.group(1)
    for _id, name, args in msg.get("tc") or []:
        if "python" in name or "interpreter" in name:
            obj = _loads_args(args)
            if isinstance(obj, dict) and isinstance(obj.get("code"), str):
                return obj["code"]
            return args if isinstance(args, str) else ""
    return ""


def code_calls(code: str) -> Tuple[List[str], List[str]]:
    """(customer messages sent with message(...), other function names called) in a code blob."""
    replies: List[str] = []
    names: List[str] = []
    if not code:
        return replies, names
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        for m in re.finditer(r"\bmessage\(\s*(?:content\s*=\s*)?(['\"])", code):
            lit = _quoted_literal(code, m.end() - 1)
            if lit:
                replies.append(lit)
        names = [n for n in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\(", code) if n not in _BUILTINS and n != "message"]
        return replies, names
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
        if not name:
            continue
        if name == "message":
            arg = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg in ("content", "message")), None)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                replies.append(arg.value)
            elif isinstance(arg, ast.JoinedStr):
                replies.append("".join(v.value for v in arg.values if isinstance(v, ast.Constant) and isinstance(v.value, str)))
        elif name not in _BUILTINS:
            names.append(name)
    return replies, names


def rle(names: Sequence[str], cap: int = 40) -> str:
    out: List[Tuple[str, int]] = []
    for n in names:
        if out and out[-1][0] == n:
            out[-1] = (n, out[-1][1] + 1)
        else:
            out.append((n, 1))
    parts = [f"{n}x{c}" if c > 1 else n for n, c in out]
    if len(parts) > cap:
        parts = parts[:cap] + [f"... +{len(parts) - cap} more"]
    return ", ".join(parts)


def is_pseudo_user(text: str) -> bool:
    s = (text or "").strip()
    return s in ("", "None", "null") or s.startswith(PSEUDO_USER)


def has_instruction_marker(text: str) -> bool:
    return any(k in (text or "") for k in INSTRUCTION_MARKERS)


def _after(text: str, marker: str, stops: Sequence[str], cap: int) -> Optional[str]:
    i = text.find(marker)
    if i < 0:
        return None
    rest = text[i + len(marker):]
    end = len(rest)
    for s in stops:
        j = rest.find(s)
        if 0 <= j < end:
            end = j
    out = rest[:end].strip()
    return out[:cap] if out else None


def extract_instruction(text: str) -> Optional[str]:
    """The task a single-instruction benchmark session was given."""
    issue = _after(text, "## Issue to resolve:", ("\n\nYour performance", "\n# Important"), 4_000)
    if issue:
        return "Resolve this issue in the repository: " + issue
    task = _after(text, "Task from supervisor:", ("\n\nContext:", "\nContext: {", "\n\n<policy>", "<policy>",
                                                 "\n\nComplete this task"), 2_000)
    if task:
        return task
    q = _after(text, "Question:", ("\nNote:", "\n\nYour performance", "\n\nImportant:"), 3_000)
    if q:
        return q
    return None


# ------------------------------------------------------------------ turns

@dataclass
class Turn:
    idx: int
    request: str
    pos: int                                  # message index of the request (0 for the opening message)
    kind: str = "customer"                    # customer | instruction | missing | root
    replies: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    final: str = ""                           # final answer / last agent text (single-instruction, OCEL)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Turn":
        return cls(**d)


def family_of(workstream: str, source_kind: str = "") -> str:
    if workstream in TAU2:
        return "tau2"
    if workstream in SINGLE:
        return "single"
    return "ocel"


def _agent_activity(m: Dict[str, Any], harness: str) -> Tuple[List[str], List[str], List[str]]:
    """(customer-facing replies, tool names, finish-tool answers) of one assistant message."""
    replies: List[str] = []
    tools: List[str] = []
    finals: List[str] = []
    if harness == "smolagents_code":
        r, names = code_calls(smol_code(m))
        replies += r
        tools += [n for n in names if n not in ("final_answer",)]
        if "final_answer" in names:
            finals.append(smol_code(m)[:1_000])
        return replies, tools, finals
    for _id, name, args in m.get("tc") or []:
        if _is_message_tool(name):
            c = _message_content(args)
            if c:
                replies.append(c)
        elif _is_finish_tool(name):
            finals.append(args[:1_000])
            tools.append(short_tool(name))
        else:
            tools.append(short_tool(name))
    text = (m.get("t") or "").strip()
    if harness in ("tool_calling", "tool_calling_with_shortlisting") and text and text not in ("None", "(no content)"):
        replies.append(text)
    return replies, tools, finals


def parse(msgs: Sequence[Dict[str, Any]], family: str, harness: str,
          root_request: Optional[str] = None) -> Tuple[List[Turn], List[str]]:
    """The user turns of one session's message list, and flags."""
    flags: List[str] = []
    if family == "ocel":
        return _parse_ocel(msgs, root_request), flags
    first_asst = next((i for i, m in enumerate(msgs) if m.get("r") == "assistant"), len(msgs))
    pre = msgs[:first_asst]
    turns: List[Turn] = []
    if family == "tau2":
        req0 = None
        for m in pre:
            req0 = first_observation(m.get("t") or "")
            if req0 is not None:
                break
        if req0 is not None:
            turns.append(Turn(0, req0, 0, kind="customer"))
        elif harness != "tool_calling":
            turns.append(Turn(0, "", 0, kind="missing"))
            flags.append("first_request_missing")
        seen_policy = False
        for i, m in enumerate(msgs):
            role, text = m.get("r"), m.get("t") or ""
            cust: Optional[str] = None
            if harness == "tool_calling":
                if role == "user":
                    if not seen_policy:
                        seen_policy = True          # msg 0: the policy, sent as a user message
                        continue
                    if not is_pseudo_user(text):
                        cust = text
            elif i >= first_asst and role in ("tool", "user"):
                cust = sender_user_message(text)
            if cust is not None:
                turns.append(Turn(len(turns), cust, i, kind="customer"))
                continue
            if role == "assistant" and turns:
                r, t, f = _agent_activity(m, harness)
                turns[-1].replies += r
                turns[-1].tools += t
                if f:
                    turns[-1].final = f[-1]
        if harness == "smolagents_code" and len(turns) <= 1 and turns and len(turns[0].replies) >= 2:
            flags.append("replies_unobservable")     # the agent messaged, no customer reply was recorded
        return turns, flags
    # single-instruction benchmarks
    opening = "\n\n".join(m.get("t") or "" for m in pre)
    instr = extract_instruction(opening)
    if instr is None:
        flags.append("instruction_marker_missing")
        firsts = [m.get("t") or "" for m in pre if m.get("r") == "user" and not is_pseudo_user(m.get("t") or "")]
        instr = (firsts[0] if firsts else opening)[-2_000:]
    turn = Turn(0, instr, 0, kind="instruction")
    last_text = ""
    for m in msgs[first_asst:]:
        if m.get("r") != "assistant":
            continue
        r, t, f = _agent_activity(m, harness)
        turn.tools += t
        if f:
            turn.final = f[-1]
        text = (m.get("t") or "").strip()
        if text and text not in ("None", "(no content)"):
            last_text = text
        if r:
            turn.replies = r[-2:]
    if not turn.final:
        turn.final = last_text[:1_000]
    return [turn], flags


def _parse_ocel(msgs: Sequence[Dict[str, Any]], root_request: Optional[str]) -> List[Turn]:
    """OCEL pipeline runs: standalone step prompts; the run's user_query is the request."""
    steps = [m for m in msgs if m.get("r") == "user"]
    outs = [m for m in msgs if m.get("r") == "assistant"]
    req = root_request or (steps[0].get("t") if steps else "")
    turn = Turn(0, req or "", 0, kind="root")
    turn.tools = [" ".join((m.get("t") or "").split())[:160] for m in steps]
    turn.final = (outs[-1].get("t") or "")[:1_000] if outs else ""
    return [turn]


def turn_line(i: int, t: Turn, request_chars: int = 400, reply_chars: int = 800, tool_chars: int = 200) -> str:
    """One turn as the segmentation state renders it: request, reply and tool calls (seg_final.jev_line)."""
    p = " ".join((t.request or ("(first request not recorded)" if t.kind == "missing" else "")).split())[:request_chars]
    reply = " ".join(" ".join(t.replies).split())
    if len(reply) > reply_chars:
        reply = reply[-reply_chars:]
    tools = rle([short_tool(x) for x in t.tools])[:tool_chars]
    return f"[{i}] USER: {p}" + (f" | ASSISTANT: {reply}" if reply else "") + (f" | TOOLS: {tools}" if tools else "")
