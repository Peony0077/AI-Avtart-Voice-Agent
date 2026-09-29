"""Answer prefetch: start RAG + LLM at the VAD *pause*, adopt the result at the *endpoint*.

The VAD reports a pause after 120 ms of silence but ends the turn only after 550 ms, and the
speculative transcript is ready inside that window, so the answer can start there too. Nothing
reaches the user before the endpoint: the LLM deltas are buffered. At the endpoint the prefetch is
adopted only if the user stayed silent and the final user text is exactly the one it answered (same
rule as the speculative STT); otherwise it is cancelled, which also stops llama-server generating.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from loguru import logger

from app.llm.client import LLMClient, LLMStats

# (user text) -> (knowledge context, citations, retrieval stats)
Retrieve = Callable[[str], Awaitable[tuple[str | None, list[dict], dict | None]]]


class AnswerPrefetch:
    def __init__(self, speech_end_t: float, stt: asyncio.Task, prepare: Callable[[str], str | None],
                 retrieve: Retrieve, build_messages: Callable[[str, str | None], list[dict]],
                 llm: LLMClient) -> None:
        self.speech_end_t = speech_end_t
        self.text: str | None = None          # the user text being answered (None until STT is done)
        self.knowledge: str | None = None
        self.citations: list[dict] = []
        self.rag_stats: dict | None = None
        self.stats = LLMStats()
        self.t_rag: float | None = None
        self.t_request: float | None = None
        self.t_first_token: float | None = None
        self.failed = False
        self.decided = asyncio.Event()        # set once it is known what (or whether) it answers
        self._queue: asyncio.Queue = asyncio.Queue()
        self.task = asyncio.create_task(self._run(stt, prepare, retrieve, build_messages, llm), name="answer-prefetch")

    async def _run(self, stt, prepare, retrieve, build_messages, llm) -> None:
        try:
            text, _t = await asyncio.shield(stt)      # cancelling the prefetch must not cancel the STT
            text = prepare(text)
            if not text:                               # empty / echo: nothing to answer
                self.failed = True
                return
            self.text = text
            self.decided.set()
            self.knowledge, self.citations, self.rag_stats = await retrieve(text)
            self.t_rag = time.time()
            messages = build_messages(text, self.knowledge)
            self.t_request = time.time()
            async for delta in llm.stream_chat(messages, self.stats):
                if self.t_first_token is None:
                    self.t_first_token = time.time()
                self._queue.put_nowait(delta)
            self._queue.put_nowait(None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the adopting turn, or dropped
            logger.warning(f"answer prefetch failed: {exc!r}")
            self.failed = True
            self._queue.put_nowait(exc)
        finally:
            self.decided.set()

    async def matches(self, speech_end_t: float | None, text: str) -> bool:
        """Usable for this endpoint: no speech after the pause, same text, still healthy."""
        if speech_end_t != self.speech_end_t or self.task.cancelled():
            return False
        try:
            await asyncio.wait_for(self.decided.wait(), 1.0)
        except TimeoutError:
            return False
        return self.text == text and not self.failed

    async def deltas(self) -> AsyncIterator[str]:
        while (item := await self._queue.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    def cancel(self) -> None:
        if not self.task.done():
            self.task.cancel()
