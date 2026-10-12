"""Wire protocol shared by backend, avatar worker and browser.

JSON messages carry ``type`` (+ ``turn_id`` where relevant).
Binary server→client packets have a 16-byte little-endian header:

    offset 0  u8   kind      1 = audio PCM16 mono 24 kHz, 2 = video JPEG, 3 = idle video JPEG,
                             4 = video H.264 access unit, 5 = idle video H.264 access unit,
                             6 = audio Ogg Opus (payload: u32 sample count at 24 kHz, then the file)
    offset 1  u8   flags     video: 1 = closing tail of a reply, 2 = H.264 keyframe
    offset 2  u8[2] reserved
    offset 4  u32  turn_id   (0 for idle frames)
    offset 8  u32  seq       packet sequence within the turn
    offset 12 u32  pts_ms    position on the turn's audio timeline (ms)

``HEADER`` below reads kind, turn_id, seq and pts_ms and skips bytes 1-3: the flags are read as
``packet[1]`` (the avatar worker writes them; ``pack`` here sends 0).

Binary client→server packets are raw PCM16 mono 16 kHz microphone audio.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

HEADER = struct.Struct("<B3xIII")
KIND_AUDIO, KIND_VIDEO, KIND_IDLE_VIDEO = 1, 2, 3
KIND_VIDEO_H264, KIND_IDLE_H264 = 4, 5
KIND_AUDIO_OPUS = 6
FLAG_TAIL, FLAG_KEY = 1, 2
MIC_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000


@dataclass(frozen=True)
class PacketHeader:
    kind: int
    turn_id: int
    seq: int
    pts_ms: int


def pack(kind: int, turn_id: int, seq: int, pts_ms: int, payload: bytes) -> bytes:
    return HEADER.pack(kind, turn_id, seq, max(0, int(pts_ms))) + payload


def unpack(data: bytes) -> tuple[PacketHeader, bytes]:
    kind, turn_id, seq, pts = HEADER.unpack_from(data)
    return PacketHeader(kind, turn_id, seq, pts), data[HEADER.size:]
