"""Breathing that follows the speech (avatar/musetalk/speech_body.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "avatar" / "musetalk"))
from speech_body import BREATH_PX, breath_levels, frame_rms, settle, to_lift  # noqa: E402


def test_shoulders_rise_in_a_pause_and_settle_while_speaking():
    fps, spf = 20, 1200
    speech = (np.sin(np.arange(spf * 20) / 5) * 8000).astype(np.int16)      # 1 s of "speech"
    pause = np.zeros(spf * 12, np.int16)                                     # 0.6 s pause
    pcm = np.concatenate([speech, pause, speech])
    rms = frame_rms(pcm, spf, len(pcm) // spf)
    levels = breath_levels(rms, fps)
    assert levels[:20].max() == 0                                            # speaking: no breath taken
    assert levels[20:32].max() >= 0.99                                       # the pause: a full breath in 0.45 s
    assert 0.4 < levels[-1] < levels[31]                                     # speaking again: let out slowly
    assert np.abs(np.diff(levels)).max() <= 1 / (0.45 * fps) + 1e-6          # no jumps
    lift = to_lift(levels)
    assert lift.min() >= -BREATH_PX - 1e-6 and lift.max() <= 0               # up only, at most BREATH_PX


def test_the_level_carries_over_and_the_tail_brings_it_back_to_zero():
    quiet = np.zeros(10, np.float32)
    a = breath_levels(quiet, 20, start=0.5)
    assert a[0] > 0.5                                                        # continues from the last sentence
    tail = settle(0.7, 12)
    assert tail[0] == np.float32(0.7) and tail[-1] == 0 and np.all(np.diff(tail) <= 0)
    assert to_lift(np.array([0.0]))[0] == 0                                  # the idle loop joins without a jump


def test_the_fast_runtime_lift_matches_the_build_time_body_motion():
    import pytest
    pytest.importorskip("cv2")                     # OpenCV is in the avatar venv
    from body_motion import move_body
    from speech_body import lift_body
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    expected = move_body(frame.copy(), 380, 0.0, -1.5)
    assert np.array_equal(lift_body(frame.copy(), 380, -1.5), expected)
    same = frame.copy()
    assert lift_body(same, 380, 0.01) is same and np.array_equal(same, frame)   # no lift: untouched
