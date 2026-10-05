"""A lighter, steadier stream (owner, 2026-10-05: "sometimes stuck while she is talking ... make it light and
smooth"): the voice as Opus (app/tts/opus.py) and video only while the page keeps up (outbox window)."""
import asyncio
import io
import struct

import numpy as np
import soundfile as sf

from app.session.outbox import Outbox
from app.session.session import VIDEO_SCALE_MIN, next_video_scale
from app.tts.opus import opus_payload


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_bytes(self, b):
        self.sent.append(b)
        await asyncio.sleep(0)

    async def send_json(self, m):
        self.sent.append(m)
        await asyncio.sleep(0)


async def test_video_waits_for_the_page_but_the_voice_never_does():
    ws = FakeWS()
    out = Outbox(ws, lambda _turn: True)
    out.window = 50                                   # bytes the page may not have acknowledged yet
    out.binary(b"v" * 80, 1, lane="speech")
    out.binary(b"w" * 80, 1, lane="speech")
    out.start()
    await asyncio.sleep(0.01)
    assert ws.sent == [b"v" * 80]                     # 80 unacknowledged: the next frame waits ...
    out.binary(b"voice", 1)
    await asyncio.sleep(0.01)
    assert ws.sent[-1] == b"voice"                    # ... the voice does not
    out.ack(85)                                       # the page has it all
    await asyncio.sleep(0.01)
    assert ws.sent[-1] == b"w" * 80 and out.in_flight == 80
    await out.stop()


async def test_without_acknowledgements_nothing_waits():
    ws = FakeWS()
    out = Outbox(ws, lambda _turn: True)              # an older page: window None
    for k in range(5):
        out.binary(b"x" * 100_000 + bytes([k]), 1, lane="speech")
    out.start()
    await out.stop()
    assert len(ws.sent) == 5


def test_opus_voice_is_small_and_exactly_as_long():
    t = np.arange(2 * 24000) / 24000
    pcm = (np.sin(2 * np.pi * 220 * t) * (0.3 + 0.2 * np.sin(2 * np.pi * 3 * t)) * 32767).astype(np.int16)   # 2 s
    payload = opus_payload(pcm)
    assert struct.unpack("<I", payload[:4])[0] == len(pcm)
    decoded, rate = sf.read(io.BytesIO(payload[4:]), dtype="int16")
    assert rate == 24000 and len(decoded) == len(pcm)            # the lip-sync clock stays exact
    assert len(payload) < len(pcm.tobytes()) / 5                 # ~56 instead of 384 kbit/s


def test_video_bitrate_settles_below_what_the_link_carries():
    # a link that carries about half the full video (frames drop above 0.5), a reply every 10 s
    scale, ceiling, last_drop, drops, scales = 1.0, 1.0, 0.0, [], []
    for k in range(30):
        congested = scale > 0.5
        drops.append(congested)
        scales.append(scale)
        scale, ceiling = next_video_scale(scale, ceiling, congested, calm_s=10 * k - last_drop)
        if congested:
            last_drop = 10 * k
    assert drops[:2] == [True, True]                    # 700 and 420 kbit/s are too much: down at once
    assert sum(drops[2:]) <= 5                          # then rarely (every other reply before)
    assert all(VIDEO_SCALE_MIN <= s <= 0.6 for s in scales[2:])
    # the link is fast again: back to full quality within a few replies
    scale, ceiling = 0.4, 0.45
    for k in range(8):
        scale, ceiling = next_video_scale(scale, ceiling, False, calm_s=70 + 10 * k)
    assert scale == 1.0
