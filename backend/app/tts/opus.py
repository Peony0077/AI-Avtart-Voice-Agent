"""The voice as Opus for the browser (owner, 2026-10-05: "make the avatar light and smooth").

PCM16 at 24 kHz is 384 kbit/s: more than half of what a slow connection carries, and a reply's first
sentence (~58 KB) took 0.6 s to arrive at 800 kbit/s. Ogg Opus from libsndfile, in-process, is ~56
kbit/s (a 2 s sentence: 11 KB instead of 94 KB, encoded in ~8 ms). The page decodes it with
decodeAudioData and cuts it to the exact sample count sent along, so the lip-sync clock (the samples
played) stays exact.
"""
from __future__ import annotations

import io
import struct

import numpy as np
import soundfile as sf

from app.transport.protocol import TTS_SAMPLE_RATE

OPUS_LEVEL = 0.8   # libsndfile's compression level for Opus: 0.8 = ~56 kbit/s (0.75 = 68, 0.85 = 41)


def opus_payload(pcm16: np.ndarray, rate: int = TTS_SAMPLE_RATE) -> bytes:
    """u32 sample count, then an Ogg Opus file of the clause."""
    buf = io.BytesIO()
    sf.write(buf, pcm16, rate, format="OGG", subtype="OPUS", compression_level=OPUS_LEVEL)
    return struct.pack("<I", len(pcm16)) + buf.getvalue()
