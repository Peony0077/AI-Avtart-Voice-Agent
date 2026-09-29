"""Avatar catch-up rule (avatar/musetalk/catchup.py, pure Python): skip only frames that would be late."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "avatar" / "musetalk"))
from catchup import STALL_MS, audible_ms, first_on_time  # noqa: E402

FPS, LATE, GPU, COMPOSE = 20, 100.0, 45.0, 12.0


def test_audible_extrapolates_from_the_last_report():
    assert audible_ms(played_ms=1000, played_at=10.0, now=10.08) == pytest.approx(1080)


def test_stalled_playback_is_not_extrapolated_but_the_lookahead_still_counts():
    # no report for 2 s: only STALL_MS of it counts ...
    assert audible_ms(1000, 10.0, now=12.0) == pytest.approx(1000 + STALL_MS)
    # ... but asking about a moment 0.4 s ahead adds the full 400 ms (this was a real bug: capping the
    # look-ahead made the worker believe it was on time and it never caught up)
    assert audible_ms(1000, 10.0, now=10.05, at=10.45) == pytest.approx(1450)


def test_nothing_skipped_when_rendering_is_ahead_of_the_audio():
    # batch of frames 8..16 of a segment starting at pts 2000; audio is only at 1500 ms
    assert first_on_time(8, 16, 2000, FPS, LATE, audible_now=1500, gpu_ms_per_frame=GPU, compose_ms=COMPOSE) == 8


def test_skips_from_the_front_until_the_first_frame_is_on_time():
    # frames 0..8 at pts 0..350; audio already at 300 ms; the first frame would take 8*45+12 ms to render
    k = first_on_time(0, 8, 0, FPS, LATE, audible_now=300, gpu_ms_per_frame=GPU, compose_ms=COMPOSE)
    assert 0 < k < 8
    ready_first = 300 + (8 - k) * GPU + COMPOSE          # audio position when frame k goes out
    assert k * 50 + LATE >= ready_first                  # frame k is at most LATE ms late
    prev_ready = 300 + (8 - (k - 1)) * GPU + COMPOSE
    assert (k - 1) * 50 + LATE < prev_ready              # and the frame before it would have been later


def test_whole_batch_skipped_when_all_of_it_is_already_past():
    assert first_on_time(0, 8, 0, FPS, LATE, audible_now=5000, gpu_ms_per_frame=GPU, compose_ms=COMPOSE) == 8
