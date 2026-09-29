"""Turn bookkeeping: every assistant response belongs to exactly one turn.

All asynchronous outputs (LLM deltas, TTS audio, avatar frames, events) carry
``turn_id``. Consumers compare it with ``TurnRegistry.current_id`` and drop
anything stale, so an interrupted turn can never leak packets.
"""
from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field


@dataclass
class TurnContext:
    turn_id: int
    source: str  # "voice" | "text"
    user_text: str = ""
    started_at: float = field(default_factory=time.time)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    timings: dict[str, float] = field(default_factory=dict)  # name -> epoch seconds
    tasks: list[asyncio.Task] = field(default_factory=list)
    header: str = ""        # raw behaviour tag of the reply (kept in history)
    spoken_text: str = ""   # reply text produced so far
    speech: object | None = None                  # SpeechOutput for voice replies
    played_ms: float = 0.0                        # browser playback position on this turn's audio
    playback_ended: asyncio.Event = field(default_factory=asyncio.Event)
    utterance_ms: int | None = None
    cancel_ms: float | None = None
    rag_stats: dict | None = None
    stt_speculative: bool | None = None
    stt_timing: dict | None = None
    avatar_first_ms: float | None = None
    prefetched: bool = False          # answer was started at the VAD pause (session/prefetch.py)

    def mark(self, name: str, at: float | None = None) -> float:
        """Record a timestamp once (first occurrence wins)."""
        ts = at if at is not None else time.time()
        self.timings.setdefault(name, ts)
        return self.timings[name]

    @property
    def is_cancelled(self) -> bool:
        return self.cancelled.is_set()

    def cancel(self) -> None:
        self.cancelled.set()
        for task in self.tasks:
            if not task.done():
                task.cancel()

    def elapsed_ms(self, start: str, end: str) -> float | None:
        if start in self.timings and end in self.timings:
            return round((self.timings[end] - self.timings[start]) * 1000, 1)
        return None


class TurnRegistry:
    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self.current: TurnContext | None = None

    @property
    def current_id(self) -> int:
        return self.current.turn_id if self.current else 0

    def new_turn(self, source: str, user_text: str = "") -> TurnContext:
        if self.current is not None:
            self.current.cancel()
        self.current = TurnContext(turn_id=next(self._ids), source=source, user_text=user_text)
        return self.current

    def is_current(self, turn_id: int) -> bool:
        return self.current is not None and self.current.turn_id == turn_id and not self.current.is_cancelled
