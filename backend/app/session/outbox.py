"""Single-writer outbound queue for one WebSocket, with priority lanes.

Starlette WebSockets must not be written from several tasks at once, and stale
data from an interrupted turn must never reach the browser. Every item carries
the turn it belongs to; data items of a turn that is no longer current are
dropped right before sending. Control messages (``force=True``) always go out.

Lanes (2026-10-05, owner: "delay and non-smooth" through the tunnel): everything used to wait in one
queue, so on a link slower than the video the voice queued behind video frames and a reply started
seconds late. Now the "hi" lane (control messages and the voice) always goes first, then the reply's
lip-synced video ("speech"), then the idle video ("idle").

Window (2026-10-05, owner: "sometimes stuck while she is talking"): the lanes only ordered what was
still in this queue. The socket (and the tunnel behind it) took megabytes at once, so on a slow link
the video was already on its way and the voice waited behind it there: at 800 kbit/s 8.9 s of gaps in
one reply, at 500 kbit/s the voice 5 s late. Now the page acknowledges the bytes it has received
(``ack``), and video goes out only while less than ``window`` bytes are unacknowledged; control and
the voice are never held. Video that cannot go out waits here, where it can be dropped (session.py).
"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import WebSocket
from loguru import logger

LANES = ("hi", "speech", "idle")
VIDEO_WINDOW = 64 * 1024   # unacknowledged bytes before video waits: ~0.5 s at 1 Mbit/s


@dataclass
class _Item:
    payload: dict | bytes
    turn_id: int
    force: bool


class Outbox:
    def __init__(self, ws: WebSocket, is_current: Callable[[int], bool]) -> None:
        self._ws = ws
        self._is_current = is_current
        self._lanes: dict[str, deque[_Item]] = {lane: deque() for lane in LANES}
        self._wake = asyncio.Event()
        self._stopping = False
        self._task: asyncio.Task | None = None
        self.dropped = 0
        self.sent_bytes = 0            # binary bytes sent (the page counts the same)
        self.acked_bytes = 0
        self.window: int | None = None  # None: the page does not acknowledge (older page) - no window
        self.window_waits = 0          # times video waited for the page

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="outbox")

    async def stop(self) -> None:
        """Send what is queued, then stop."""
        self._stopping = True
        self._wake.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=2)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    def json(self, msg: dict, turn_id: int = 0, force: bool = False) -> None:
        self._put("hi", _Item(msg, turn_id, force))

    def binary(self, data: bytes, turn_id: int = 0, force: bool = False, lane: str = "hi") -> None:
        self._put(lane, _Item(data, turn_id, force))

    def ack(self, received: int) -> None:
        """The page has received this many binary bytes."""
        if received > self.acked_bytes:
            self.acked_bytes = min(received, self.sent_bytes)
            self._wake.set()

    @property
    def in_flight(self) -> int:
        return self.sent_bytes - self.acked_bytes

    def _put(self, lane: str, item: _Item) -> None:
        self._lanes[lane].append(item)
        self._wake.set()

    @property
    def backlog(self) -> int:
        return sum(len(q) for q in self._lanes.values())

    def lane_backlog(self, lane: str) -> int:
        return len(self._lanes[lane])

    def clear_lane(self, lane: str) -> int:
        n = len(self._lanes[lane])
        self._lanes[lane].clear()
        self.dropped += n
        return n

    def _next(self) -> _Item | None:
        if self._lanes["hi"]:
            return self._lanes["hi"].popleft()
        if self.window is not None and self.in_flight >= self.window and (self._lanes["speech"] or self._lanes["idle"]):
            self.window_waits += 1
            return None                           # the page is behind: video waits (an ack wakes us)
        for lane in LANES[1:]:
            if self._lanes[lane]:
                return self._lanes[lane].popleft()
        return None

    async def _run(self) -> None:
        while True:
            item = self._next()
            if item is None:
                if self._stopping:
                    return
                self._wake.clear()
                await self._wake.wait()
                continue
            if item.turn_id and not item.force and not self._is_current(item.turn_id):
                self.dropped += 1
                continue
            try:
                if isinstance(item.payload, bytes):
                    await self._ws.send_bytes(item.payload)
                    self.sent_bytes += len(item.payload)
                else:
                    await self._ws.send_json(item.payload)
            except Exception as exc:  # noqa: BLE001 - socket closed; stop quietly
                logger.debug(f"outbox stopped: {exc!r}")
                return
