"""H.264 video for the browser (avatar/musetalk/h264.py) and the outbox lanes (app/session/outbox.py):
owner, 2026-10-05: "delay and non-smooth" through the tunnel - JPEG frames needed 6.7 Mbit/s and the
voice waited behind them."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "avatar" / "musetalk"))
from app.session.outbox import Outbox  # noqa: E402
from h264 import START_CODE, FlvReader, is_keyframe  # noqa: E402

SPS, PPS = b"\x67sps-bytes", b"\x68pps"
IDR, P1, P2 = b"\x65" + b"I" * 300, b"\x41" + b"P" * 40, b"\x41" + b"Q" * 30     # slices (NAL types 5, 1, 1)


def _tag(kind: int, data: bytes) -> bytes:
    return bytes([kind]) + len(data).to_bytes(3, "big") + bytes(7) + data + (11 + len(data)).to_bytes(4, "big")


def _flv() -> bytes:
    avcc = bytes([1, 0x42, 0xC0, 0x1E, 0xFF, 0xE1]) + len(SPS).to_bytes(2, "big") + SPS + bytes([1]) \
        + len(PPS).to_bytes(2, "big") + PPS
    nal = lambda *ns: b"".join(len(n).to_bytes(4, "big") + n for n in ns)      # noqa: E731
    return (b"FLV\x01\x01" + (9).to_bytes(4, "big") + bytes(4)
            + _tag(18, b"\x02\x00\x0aonMetaData")                                 # script data: skipped
            + _tag(9, bytes([0x17, 0, 0, 0, 0]) + avcc)                             # sequence header
            + _tag(9, bytes([0x17, 1, 0, 0, 0]) + nal(IDR))                         # keyframe
            + _tag(9, bytes([0x27, 1, 0, 0, 0]) + nal(P1, P2)))                     # inter frame, two slices


def test_flv_gives_whole_frames_however_the_bytes_arrive():
    # the raw stream was split by timing ("no bytes for 4 ms = complete"): under load a frame arrived in
    # pieces further apart, was cut in two and broke the browser's decoder until the next keyframe
    data = _flv()
    expected = [(True, START_CODE + SPS + START_CODE + PPS + START_CODE + IDR), (False, START_CODE + P1 + START_CODE + P2)]
    for step in (len(data), 1, 7, 64):
        reader, frames = FlvReader(), []
        for i in range(0, len(data), step):
            frames += reader.feed(data[i:i + step])
        assert frames == expected, step
    assert is_keyframe(expected[0][1]) and not is_keyframe(expected[1][1])
    reader = FlvReader()
    assert reader.feed(data[:-1]) == expected[:1]          # the last frame is not complete yet ...
    assert reader.feed(data[-1:]) == expected[1:]          # ... until its last byte


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_bytes(self, b):
        self.sent.append(b)
        await asyncio.sleep(0)

    async def send_json(self, m):
        self.sent.append(m)
        await asyncio.sleep(0)


async def test_the_voice_never_waits_behind_video():
    ws = FakeWS()
    out = Outbox(ws, lambda _turn: True)
    for k in range(5):
        out.binary(b"idle%d" % k, lane="idle")
    for k in range(5):
        out.binary(b"video%d" % k, 1, lane="speech")
    out.binary(b"audio", 1)                               # queued last ...
    out.json({"type": "state"})
    out.start()
    await out.stop()
    assert ws.sent[0] == b"audio" and ws.sent[1] == {"type": "state"}   # ... sent first
    assert ws.sent[2:7] == [b"video%d" % k for k in range(5)]            # then the reply's video
    assert ws.sent[7:] == [b"idle%d" % k for k in range(5)]              # the idle video last


async def test_idle_video_can_be_dropped_as_a_whole():
    ws = FakeWS()
    out = Outbox(ws, lambda _turn: True)
    for _ in range(25):
        out.binary(b"idle", lane="idle")
    assert out.lane_backlog("idle") == 25 and out.clear_lane("idle") == 25 and out.backlog == 0
    out.start()
    await out.stop()
    assert ws.sent == []
