"""Streaming client for llama-server's OpenAI-compatible chat endpoint.

Cancelling the consuming task closes the HTTP stream; llama-server then stops
generating for that request (verified in Phase 1 tests via its log).
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from app.core.config import LLMCfg


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
    def __init__(self, cfg: LLMCfg) -> None:
        self.cfg = cfg
        self._http = httpx.AsyncClient(base_url=cfg.base_url, timeout=httpx.Timeout(cfg.request_timeout_s, connect=5.0))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def stream_chat(self, messages: list[dict], stats: LLMStats | None = None,
                          max_tokens: int | None = None) -> AsyncIterator[str]:
        body = {
            "model": "local",
            "messages": messages,
            "stream": True,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "cache_prompt": True,
            "timings_per_token": False,
        }
        async with self._http.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
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

    async def complete(self, messages: list[dict], max_tokens: int = 16) -> str:
        """Non-streaming helper (warm-up, summaries)."""
        return "".join([piece async for piece in self.stream_chat(messages, max_tokens=max_tokens)])
