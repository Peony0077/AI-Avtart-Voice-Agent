"""The outbox's video lanes stay bounded when the page stops acknowledging, and stop() does not swallow
its caller's cancellation (app/session/outbox.py)."""
import asyncio

import pytest

from app.session.outbox import MAX_VIDEO_FRAMES, VIDEO_WINDOW, Outbox


class FakeWS:
    def __init__(self, block: asyncio.Event | None = None):
        self.sent, self.block = [], block

    async def send_bytes(self, b):
        if self.block is not None:
            await self.block.wait()
        self.sent.append(b)

    async def send_json(self, m):
        self.sent.append(m)


async def test_jpeg_reply_frames_stay_bounded_when_the_page_stops_acknowledging():
    current = {"turn": 1}
    out = Outbox(FakeWS(), lambda tid: tid == current["turn"])
    out.window = VIDEO_WINDOW
    out.start()
    out.binary(b"x" * VIDEO_WINDOW, 1, lane="speech")      # fills the window; the page never acks
    await asyncio.sleep(0.01)
    for _ in range(MAX_VIDEO_FRAMES * 5):
        out.binary(b"j" * 1000, 1, lane="speech")           # the reply's JPEG frames keep coming
    await asyncio.sleep(0.01)
    assert out.lane_backlog("speech") == MAX_VIDEO_FRAMES and out.dropped >= MAX_VIDEO_FRAMES * 4
    current["turn"] = 2                                      # interrupted: its frames are stale
    out.json({"type": "state"})                              # (wakes the sender)
    await asyncio.sleep(0.01)
    assert out.lane_backlog("speech") == 0                   # purged while the window is still full
    await out.stop()


async def test_stop_keeps_the_callers_cancellation():
    gate = asyncio.Event()
    out = Outbox(FakeWS(block=gate), lambda tid: True)
    out.start()
    out.binary(b"x" * 10)                                    # the send never finishes (a dead link)
    await asyncio.sleep(0.01)
    stopper = asyncio.create_task(out.stop())
    await asyncio.sleep(0.01)
    stopper.cancel()                                         # e.g. the session's close() is cancelled
    with pytest.raises(asyncio.CancelledError):
        await stopper
    await asyncio.sleep(0)
    assert out._task.done()                                  # and the sender does not linger
