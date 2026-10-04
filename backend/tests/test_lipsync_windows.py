"""Lip-sync timing (avatar/musetalk/lipsync.py): which 200 ms of audio each video frame is made from."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "avatar" / "musetalk"))
from lipsync import FEATURE_MS, PAD, SHIFT_MS, WINDOW, frame_count, window_starts  # noqa: E402


def centre_ms(start: int) -> float:
    """Audio time at the middle of a window that starts at feature ``start`` (padded index)."""
    return (start - PAD + WINDOW / 2) * FEATURE_MS


def test_each_window_is_centred_on_its_frame_plus_the_measured_shift():
    for fps in (20, 25):
        for k, s in enumerate(window_starts(40, fps)):
            want = (k + 0.5) * 1000 / fps + SHIFT_MS
            assert abs(centre_ms(s) - want) <= FEATURE_MS / 2        # nearest 20 ms feature


def test_windows_advance_with_the_frames_and_stay_inside_the_padded_features():
    fps = 20
    n_samples = 16000 * 3                                             # 3 s at 16 kHz
    n = frame_count(n_samples, fps)
    assert n == 60
    starts = window_starts(n, fps)
    steps = {b - a for a, b in zip(starts, starts[1:], strict=False)}
    assert steps <= {2, 3}                                            # 50 ms per frame = 2.5 features
    n_features = n_samples * 50 // 16000
    assert starts[0] >= 0 and starts[-1] + WINDOW <= n_features + 2 * PAD


def test_at_25_fps_the_windows_are_the_ones_musetalk_was_trained_with():
    """Training (MuseTalk utils.process_audio_features): frame F gets features 2F-4 .. 2F+6."""
    assert SHIFT_MS == 0
    for f, s in enumerate(window_starts(50, 25)):
        assert s - PAD == 2 * f - 4


def test_at_20_fps_the_windows_are_not_early_like_musetalks_inference_layout():
    """get_whisper_chunk at 20 fps: frame k starts at feature floor(2.5 k) - 6, centred 45 ms before
    the frame's middle (the mouth showed sounds already heard)."""
    for k, s in enumerate(window_starts(40, 20)):
        musetalk_centre = (int(2.5 * k) - 6 + WINDOW / 2) * FEATURE_MS
        assert centre_ms(s) - musetalk_centre >= 30


def test_short_audio_has_no_frames():
    assert frame_count(799, 20) == 0                                  # under one 50 ms frame
    assert window_starts(0, 20) == []
