"""Streaming client for llama-server's OpenAI-compatible chat endpoint.

Cancelling the consuming task closes the HTTP stream; llama-server then stops
generating for that request (verified in Phase 1 tests via its log).
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

import httpx
from loguru import logger

from app.core.config import LLMCfg

# llama-server's /completion says why it stopped in its own words; the chat endpoint uses OpenAI's
_STOP_TYPES = {"eos": "stop", "word": "stop", "limit": "length"}


class LLMError(RuntimeError):
    """llama-server reported an error inside a 200 response (an SSE ``error`` event), or sent no reply at all."""


def _error_event(line: str) -> str | None:
    """The error in an SSE line (``error: {...}``, or a ``data:`` chunk with an ``error`` object), else None.
    Only ``data:`` lines were read: such an error became a silent empty reply."""
    if line.startswith("error:"):
        return line[6:].strip() or "unknown error"
    if line.startswith("data:") and '"error"' in line:
        try:
            chunk = json.loads(line[5:])
        except ValueError:
            return None
        if isinstance(chunk, dict) and chunk.get("error"):
            err = chunk["error"]
            return str(err.get("message", err) if isinstance(err, dict) else err)
    return None


@dataclass
class LLMStats:
    prompt_tokens: int | None = None
    prompt_ms: float | None = None
    completion_tokens: int | None = None
    tokens_per_s: float | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = None
    extra: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, cfg: LLMCfg, wait_ready: Callable[[float], Awaitable[bool]] | None = None) -> None:
        self.cfg = cfg
        self.wait_ready = wait_ready     # supervisor hook: wait for a crashed llama-server to come back
        self._http = httpx.AsyncClient(base_url=cfg.base_url, timeout=httpx.Timeout(cfg.request_timeout_s, connect=5.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def stream_chat(self, messages: list[dict], stats: LLMStats | None = None,
                          max_tokens: int | None = None, pause_after: Callable[[str], bool] | None = None,
                          resume: Callable[[], Awaitable[None]] | None = None,
                          temperature: float | None = None) -> AsyncIterator[str]:
        """Stream the reply. With ``pause_after``/``resume``: whenever ``pause_after(text so far)`` is true
        (e.g. a new speakable clause exists) generation stops until ``resume()`` returns, then continues the
        same reply from the text so far (llama-server reuses its cache: only the new tokens are processed).

        Why: the LLM writes far faster than speech, but while it generates the rest of the reply it shares
        the GPU with the first clause's synthesis and the avatar's first lip-sync frames, and those are
        what the user waits for (measured: Kokoro 78 -> ~170 ms, avatar first frames ~170 -> ~330 ms)."""
        if pause_after is None or resume is None:
            async for delta in self._stream_chat(messages, stats, max_tokens, temperature):
                yield delta
            return
        text = ""
        stream = self._stream_chat(messages, stats, max_tokens)
        while True:                                     # it may pause again (e.g. after every clause)
            paused = False
            try:
                async for delta in stream:
                    text += delta
                    yield delta
                    if pause_after(text):
                        paused = True
                        break
            finally:
                await stream.aclose()                   # closing the HTTP stream stops llama-server now
            if not paused:
                return
            await resume()
            stream = self._continue(messages, text, stats, max_tokens)

    async def _continue(self, messages: list[dict], text: str, stats: LLMStats | None,
                        max_tokens: int | None) -> AsyncIterator[str]:
        """Continue an assistant reply that already starts with ``text`` (raw prompt: the chat template
        of ``messages`` with the generation prompt, then the text)."""
        resp = await self._http.post("/apply-template", json={"messages": messages})
        resp.raise_for_status()
        body = {"prompt": resp.json()["prompt"] + text, "stream": True, "cache_prompt": True,
                "temperature": self.cfg.temperature, "top_p": self.cfg.top_p, "top_k": 20, "min_p": 0.0,
                "n_predict": max(16, (max_tokens or self.cfg.max_tokens) - len(text) // 3)}
        async with self._http.stream("POST", "/completion", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if (error := _error_event(line)) is not None:
                    raise LLMError(f"llama-server: {error}")
                if not line.startswith("data:"):
                    continue
                chunk = json.loads(line[5:])
                if chunk.get("content"):
                    yield chunk["content"]
                if chunk.get("stop"):
                    if stats is not None:
                        t = chunk.get("timings") or {}
                        stop = chunk.get("stop_type")
                        stats.finish_reason = _STOP_TYPES.get(stop, stop)
                        stats.completion_tokens = (stats.completion_tokens or 0) + (t.get("predicted_n") or 0)
                        stats.extra["continued"] = {"prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n")}
                    break

    async def _stream_chat(self, messages: list[dict], stats: LLMStats | None = None,
                           max_tokens: int | None = None, temperature: float | None = None) -> AsyncIterator[str]:
        body = {
            "model": "local",
            "messages": messages,
            "stream": True,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "top_p": self.cfg.top_p,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "cache_prompt": True,
            "timings_per_token": False,
        }
        for attempt in (1, 2):
            streamed = False
            try:
                async for delta in self._stream(body, stats):
                    streamed = True
                    yield delta
                return
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError, httpx.HTTPStatusError) as exc:
                # llama-server crashed (or its restart is still loading the model: 503): if nothing was
                # streamed yet, wait for the watchdog's restart and retry once
                unavailable = not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code == 503
                if streamed or attempt == 2 or self.wait_ready is None or not unavailable:
                    raise
                logger.warning("LLM server unreachable; waiting for its restart")
                if not await self.wait_ready(90.0):
                    raise

    async def _stream(self, body: dict, stats: LLMStats | None) -> AsyncIterator[str]:
        async with self._http.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            other: list[str] = []          # what came instead of SSE data (for the error)
            data = False
            async for line in resp.aiter_lines():
                if (error := _error_event(line)) is not None:
                    raise LLMError(f"llama-server: {error}")
                if not line.startswith("data:"):
                    if line.strip() and len(other) < 3:
                        other.append(line.strip()[:200])
                    continue
                data = True
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
                    if choices[0].get("finish_reason") and stats is not None:
                        stats.finish_reason = choices[0]["finish_reason"]
                if stats is not None and "timings" in chunk:
                    t = chunk["timings"]
                    stats.prompt_tokens = t.get("prompt_n")
                    stats.prompt_ms = t.get("prompt_ms")
                    stats.completion_tokens = t.get("predicted_n")
                    stats.tokens_per_s = t.get("predicted_per_second")
                    stats.cached_tokens = t.get("cache_n")
            if not data:
                raise LLMError(f"llama-server sent no reply (HTTP {resp.status_code}): {' | '.join(other) or 'empty'}")

    async def complete(self, messages: list[dict], max_tokens: int = 16, temperature: float | None = None) -> str:
        """Non-streaming helper (warm-up, summaries; ``temperature`` 0 for a choice that must not vary)."""
        return "".join([piece async for piece in self.stream_chat(messages, max_tokens=max_tokens,
                                                                  temperature=temperature)])
