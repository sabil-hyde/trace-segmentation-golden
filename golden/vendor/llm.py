"""Thin OpenAI-compatible chat + embeddings client used by every stage
that talks to a model (cluster labeling, gold references, judging, the
model sweep). One retrying httpx client, optional thread-pool fan-out.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

logger = logging.getLogger("trace_mining.llm")

_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=180.0)

# Replayed agent-harness prompts make Claude answer with Anthropic tool_use
# blocks. Without a `tools` parameter the gateway drops those blocks and
# returns an empty message (~9 completion tokens) while still charging for
# the prompt. Passing this permissive catch-all makes the same answer come
# back as regular OpenAI tool_calls, which we serialize into text.
SALVAGE_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tool",
            "description": (
                "Invoke any tool available in this environment. If the tool "
                "you intend to call has a different name, use that name."
            ),
            "parameters": {"type": "object", "additionalProperties": True},
        },
    }
]


def tool_calls_text(tool_calls: Optional[List[Dict[str, Any]]]) -> str:
    """Serialize an OpenAI tool_calls array into reference-answer text."""
    calls = []
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                pass
        calls.append({"name": fn.get("name"), "arguments": args})
    return json.dumps({"tool_calls": calls}, ensure_ascii=False) if calls else ""


class ChatResult:
    def __init__(
        self,
        content: str,
        usage: Dict[str, int],
        latency_ms: float,
        raw: Dict[str, Any],
        error: Optional[str] = None,
        trace_id: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ):
        self.content = content
        self.usage = usage
        self.latency_ms = latency_ms
        self.raw = raw
        self.error = error
        self.trace_id = trace_id
        self.tool_calls = tool_calls


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_s: float = 120.0,
        stream: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        # With stream=True the read timeout applies BETWEEN chunks (an idle
        # timeout), not to the whole response -- long generations on big agent
        # transcripts no longer hit ReadTimeout while the model is still
        # actively producing tokens.
        self.stream = stream
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=timeout_s),
            headers={"Authorization": f"Bearer {api_key}"},
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
        )

    def _post(self, path: str, payload: Dict[str, Any], max_attempts: int = 4) -> Dict[str, Any]:
        delay = 2.0
        last_err: Optional[str] = None
        model = payload.get("model", "?")
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self._client.post(self.base_url + path, json=payload)
                if resp.status_code in (429,) or resp.status_code >= 500:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    retry_after = resp.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else delay
                    logger.warning(
                        "%s [%s] attempt %d/%d failed (%s) -- retrying in %.0fs",
                        path, model, attempt, max_attempts, last_err, wait,
                    )
                    time.sleep(wait)
                    delay *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                call_id = resp.headers.get("x-litellm-call-id", "")
                if call_id:
                    data["_gateway_trace_id"] = call_id
                return data
            except httpx.HTTPStatusError as exc:  # 4xx -- don't retry
                logger.warning(
                    "%s [%s] rejected: HTTP %d: %s",
                    path, model, exc.response.status_code, exc.response.text[:200],
                )
                raise RuntimeError(
                    f"{path} failed: HTTP {exc.response.status_code}: "
                    f"{exc.response.text[:300]}"
                ) from exc
            except httpx.HTTPError as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "%s [%s] attempt %d/%d failed (%s) -- retrying in %.0fs",
                    path, model, attempt, max_attempts, last_err, delay,
                )
                time.sleep(delay)
                delay *= 2
        logger.error("%s [%s] gave up after %d attempts: %s", path, model, max_attempts, last_err)
        raise RuntimeError(f"{path} failed after {max_attempts} attempts: {last_err}")

    def _chat_stream(
        self, payload: Dict[str, Any], max_attempts: int = 4
    ) -> Dict[str, Any]:
        """POST /chat/completions with SSE streaming, reassembled into the
        non-streaming response shape so chat() handles both paths the same.

        Falls back to the blocking call when the gateway rejects streaming.
        """
        path = "/chat/completions"
        model = payload.get("model", "?")
        body = dict(payload)
        body["stream"] = True
        # Ask for token usage in the final chunk (OpenAI extension). Dropped
        # transparently below if the gateway rejects the field.
        body["stream_options"] = {"include_usage": True}
        delay = 2.0
        last_err: Optional[str] = None
        for attempt in range(1, max_attempts + 1):
            try:
                with self._client.stream(
                    "POST", self.base_url + path, json=body
                ) as resp:
                    if resp.status_code == 429 or resp.status_code >= 500:
                        resp.read()
                        last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        retry_after = resp.headers.get("retry-after")
                        wait = float(retry_after) if retry_after else delay
                        logger.warning(
                            "%s [%s] attempt %d/%d failed (%s) -- retrying in %.0fs",
                            path, model, attempt, max_attempts, last_err, wait,
                        )
                        time.sleep(wait)
                        delay *= 2
                        continue
                    if resp.status_code >= 400:
                        resp.read()
                        text = resp.text
                        if "stream_options" in text and "stream_options" in body:
                            body.pop("stream_options", None)  # unknown field here
                            continue
                        if "stream" in text.lower():
                            logger.warning(
                                "%s [%s] streaming rejected by gateway -- "
                                "falling back to blocking call", path, model,
                            )
                            return self._post(path, payload)
                        logger.warning(
                            "%s [%s] rejected: HTTP %d: %s",
                            path, model, resp.status_code, text[:200],
                        )
                        raise RuntimeError(
                            f"{path} failed: HTTP {resp.status_code}: {text[:300]}"
                        )

                    parts: List[str] = []
                    tool_calls: Dict[int, Dict[str, Any]] = {}
                    usage: Dict[str, Any] = {}
                    finish: Optional[str] = None
                    provider: Optional[str] = None
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except ValueError:
                            continue
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        provider = chunk.get("provider") or provider
                        for choice in chunk.get("choices") or []:
                            if choice.get("finish_reason"):
                                finish = choice["finish_reason"]
                            delta = choice.get("delta") or {}
                            if delta.get("content"):
                                parts.append(delta["content"])
                            for tc in delta.get("tool_calls") or []:
                                idx = tc.get("index") or 0
                                slot = tool_calls.setdefault(
                                    idx,
                                    {"id": None, "type": "function",
                                     "function": {"name": "", "arguments": ""}},
                                )
                                if tc.get("id"):
                                    slot["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    slot["function"]["name"] = fn["name"]
                                if fn.get("arguments"):
                                    slot["function"]["arguments"] += fn["arguments"]
                    return {
                        "choices": [
                            {
                                "finish_reason": finish,
                                "message": {
                                    "content": "".join(parts),
                                    "tool_calls": [
                                        tool_calls[i] for i in sorted(tool_calls)
                                    ] or None,
                                },
                            }
                        ],
                        "usage": usage,
                        "provider": provider,
                        "streamed": True,
                    }
            except httpx.HTTPError as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "%s [%s] attempt %d/%d failed (%s) -- retrying in %.0fs",
                    path, model, attempt, max_attempts, last_err, delay,
                )
                time.sleep(delay)
                delay *= 2
        logger.error(
            "%s [%s] gave up after %d attempts: %s", path, model, max_attempts, last_err
        )
        raise RuntimeError(f"{path} failed after {max_attempts} attempts: {last_err}")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        warn_empty: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        # temperature=None omits the field: extended-thinking requests reject
        # anything but the provider default. `extra` carries provider fields
        # such as OpenRouter's {"reasoning": {"effort": "high"}}.
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
        if extra:
            payload.update(extra)
        t0 = time.monotonic()
        try:
            if self.stream:
                data = self._chat_stream(payload)
            else:
                data = self._post("/chat/completions", payload)
        except RuntimeError as exc:
            return ChatResult("", {}, (time.monotonic() - t0) * 1000, {}, error=str(exc))
        latency_ms = (time.monotonic() - t0) * 1000
        content = ""
        finish = None
        tool_calls = None
        try:
            choice = data["choices"][0]
            finish = choice.get("finish_reason")
            content = choice["message"].get("content") or ""
            tool_calls = choice["message"].get("tool_calls") or None
        except (KeyError, IndexError):
            pass
        usage = data.get("usage") or {}
        if warn_empty and not content.strip() and not tool_calls:
            # A 200 with no text is still a failed call for our purposes
            # (e.g. unreplayable image prompts, refusals, truncated payloads);
            # say so instead of silently storing an empty reference. Callers
            # that follow up with the tools-salvage retry pass warn_empty=False
            # on the first attempt so only unrecoverable emptiness is logged.
            logger.warning(
                "chat [%s] returned empty content (finish_reason=%s, usage=%s, %.0fms)",
                model, finish, usage or "{}", latency_ms,
            )
        return ChatResult(
            content,
            usage,
            latency_ms,
            data,
            trace_id=str(data.get("_gateway_trace_id") or ""),
            tool_calls=tool_calls,
        )

    def embed(self, model: str, texts: Sequence[str], batch: int = 128) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), batch):
            chunk = list(texts[i : i + batch])
            data = self._post("/embeddings", {"model": model, "input": chunk})
            rows = sorted(data["data"], key=lambda d: d["index"])
            out.extend(r["embedding"] for r in rows)
        return out


def parallel_map(
    fn: Callable[[Any], Any], items: Sequence[Any], concurrency: int
) -> List[Any]:
    """Ordered thread-pool map; item order in == result order out."""
    if concurrency <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(fn, items))


def truncate(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"
