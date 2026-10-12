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
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import WebSocket
from loguru import logger

LANES = ("hi", "speech", "idle")
VIDEO_WINDOW = 64 * 1024   # unacknowledged bytes before video waits: ~0.5 s at 1 Mbit/s
# Most video frames a lane holds. The session drops H.264 frames long before this (session.on_avatar_frame);
# the reply's JPEG frames had no limit, and a page that stopped acknowledging (window full) let them pile
# up without bound. Beyond this the oldest go first (a JPEG frame stands alone; frames that late are
# useless anyway: ~5 s at 20 fps).
MAX_VIDEO_FRAMES = 100


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
        self._acks: deque[tuple[float, int]] = deque(maxlen=128)   # (time, acked bytes): the delivery rate
        self.window: int | None = None  # None: the page does not acknowledge (older page) - no window
        self.window_waits = 0          # times video waited for the page

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="outbox")

    async def stop(self) -> None:
        """Send what is queued (for at most 2 s), then stop. A cancelled caller stays cancelled: catching
        CancelledError here hid the cancellation of the session's close()."""
        self._stopping = True
        self._wake.set()
        if self._task:
            try:
                await asyncio.wait({self._task}, timeout=2)
            finally:
                if not self._task.done():
                    self._task.cancel()

    def json(self, msg: dict, turn_id: int = 0, force: bool = False) -> None:
        self._put("hi", _Item(msg, turn_id, force))

    def binary(self, data: bytes, turn_id: int = 0, force: bool = False, lane: str = "hi") -> None:
        self._put(lane, _Item(data, turn_id, force))

    def ack(self, received: int) -> None:
        """The page has received this many binary bytes."""
        if received > self.acked_bytes:
            self.acked_bytes = min(received, self.sent_bytes)
            self._acks.append((time.monotonic(), self.acked_bytes))
            self._wake.set()

    def delivery_kbps(self, span_s: float = 3.0) -> float | None:
        """kbit/s the page received over the last ``span_s`` (its acknowledgements); None when unknown. While
        video waits for the window (a slow link) this is what the link carries."""
        now = time.monotonic()
        recent = [(t, b) for t, b in self._acks if now - t <= span_s]
        if len(recent) < 2 or recent[-1][0] - recent[0][0] < 0.5:
            return None
        return (recent[-1][1] - recent[0][1]) * 8 / 1000 / (recent[-1][0] - recent[0][0])

    @property
    def in_flight(self) -> int:
        return self.sent_bytes - self.acked_bytes

    def _put(self, lane: str, item: _Item) -> None:
        queue = self._lanes[lane]
        queue.append(item)
        if lane != "hi" and len(queue) > MAX_VIDEO_FRAMES:
            self._purge(lane)
            while len(queue) > MAX_VIDEO_FRAMES:
                queue.popleft()
                self.dropped += 1
        self._wake.set()

    def _purge(self, lane: str) -> None:
        """Drop the lane's frames of turns that are no longer current (they would be dropped when sent)."""
        queue = self._lanes[lane]
        keep = [i for i in queue if i.force or not i.turn_id or self._is_current(i.turn_id)]
        if len(keep) < len(queue):
            self.dropped += len(queue) - len(keep)
            queue.clear()
            queue.extend(keep)

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
            self._purge("speech")                 # an interrupted reply's frames need not wait for the page
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
