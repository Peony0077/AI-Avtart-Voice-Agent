"""Which audio each lip-sync frame is made from (pure functions, unit-tested in backend/tests).

MuseTalk gives each video frame a window of 10 whisper features (20 ms each = 200 ms of audio). In
training the window is centred on the middle of the frame (musetalk/utils/utils.py
process_audio_features at 25 fps: features 2F-4 .. 2F+6 for frame F). Its inference windowing
(audio_processor.get_whisper_chunk) only matches that at 25 fps: at our 20 fps it centred each window
45 ms before the frame's middle, so every frame showed a sound that had already been heard.

Measured on rendered frames (lipsync_check.py; > 0 = the mouth is late): mouth movement against the
loudness of 25-35 s of continuous speech in Kokoro voices the agent can use:
                                   af_heart   am_michael   bf_emma
    MuseTalk's windows at 20 fps     +44         +75         +85    ms
    centred on the frame (here)       -2         +22         +35    ms
Isolated words (af_heart): +15..+23 ms before, -6..-23 ms now. Centring 50 ms later than training made
the mouth move before the sound (-40 ms), which people notice sooner than lateness (ITU-R BT.1359).
"""
from __future__ import annotations

import math

FEATURE_MS = 20          # whisper encoder: 50 features a second
WINDOW = 10              # features per frame (MuseTalk: 2 * (2 + 2 + 1))
SHIFT_MS = 0.0           # training alignment; the measurement above found no reason to move from it
PAD = 20                 # zero features before and after the clip (400 ms each)


def frame_count(n_samples_16k: int, fps: float) -> int:
    return math.floor(n_samples_16k / 16000 * fps)


def window_starts(n_frames: int, fps: float, shift_ms: float = SHIFT_MS, pad: int = PAD) -> list[int]:
    """Index of each frame's first feature in the features padded with ``pad`` zeros at both ends: the
    10-feature window is centred on the frame's middle (+ ``shift_ms``)."""
    starts = []
    for k in range(n_frames):
        centre_ms = (k + 0.5) * 1000 / fps + shift_ms
        starts.append(int(round(centre_ms / FEATURE_MS)) - WINDOW // 2 + pad)
    return starts
