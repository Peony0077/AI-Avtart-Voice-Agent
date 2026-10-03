"""Avatar catch-up rule (avatar/musetalk/catchup.py, pure Python): skip only frames that would be late."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "avatar" / "musetalk"))
from catchup import MAX_STRIDE, STALL_MS, audible_ms, batch_frames, first_on_time, stride_for  # noqa: E402

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
    # batch of 8 starting at frame 8 of a 16-frame segment at pts 2000; audio is only at 1500 ms
    assert first_on_time(8, 8, 16, 2000, FPS, LATE, audible_now=1500, gpu_ms_per_frame=GPU, compose_ms=COMPOSE) == 8


def test_batch_slides_forward_and_keeps_its_size():
    # 40-frame segment at pts 0; audio already at 300 ms; a batch of 4 takes 4*45+12 ms to its first frame
    k = first_on_time(0, 4, 40, 0, FPS, LATE, audible_now=300, gpu_ms_per_frame=GPU, compose_ms=COMPOSE)
    ready_first = 300 + 4 * GPU + COMPOSE                # audio position when frame k goes out (full batch)
    assert k * 50 + LATE >= ready_first                  # frame k is at most LATE ms late
    assert (k - 1) * 50 + LATE < ready_first             # and the frame before it would have been later
    assert k + 4 <= 40                                   # the batch after the skip is still 4 frames


def test_at_the_end_of_a_segment_the_batch_shrinks_to_what_is_left():
    # frames 0..8 at pts 0..350, batch of 8; audio at 300 ms: only frames near the end are worth it
    k = first_on_time(0, 8, 8, 0, FPS, LATE, audible_now=300, gpu_ms_per_frame=GPU, compose_ms=COMPOSE)
    assert 0 < k < 8
    assert k * 50 + LATE >= 300 + (8 - k) * GPU + COMPOSE
    assert (k - 1) * 50 + LATE < 300 + (8 - (k - 1)) * GPU + COMPOSE


def test_whole_segment_skipped_when_all_of_it_is_already_past():
    assert first_on_time(0, 8, 8, 0, FPS, LATE, audible_now=5000, gpu_ms_per_frame=GPU, compose_ms=COMPOSE) == 8


@pytest.mark.parametrize("gpu_ms, stride", [(30, 1), (45, 1), (60, 1), (73, 2), (85, 2), (90, 3), (100, 3), (400, MAX_STRIDE)])
def test_every_frame_while_the_gpu_keeps_up_else_every_2nd_or_3rd(gpu_ms, stride):
    # 20 fps = 50 ms per frame; up to 25 % slower keeps every frame (the occasional skip handles that)
    assert stride_for(gpu_ms, FPS) == stride


def test_strided_batch_is_evenly_spaced_and_stays_in_the_segment():
    assert batch_frames(10, 4, 40, 2) == [10, 12, 14, 16]
    assert batch_frames(10, 4, 15, 3) == [10, 13]
    assert batch_frames(0, 8, 40) == list(range(8))


def test_strided_batch_is_on_time_with_fewer_frames_skipped():
    # audio at 300 ms, GPU at 100 ms per frame (LLM still writing): a contiguous batch of 4 must start
    # later in the segment than a strided one, which covers more time with the same GPU work
    contiguous = first_on_time(0, 4, 80, 0, FPS, LATE, audible_now=300, gpu_ms_per_frame=100, compose_ms=COMPOSE)
    strided = first_on_time(0, 4, 80, 0, FPS, LATE, audible_now=300, gpu_ms_per_frame=100, compose_ms=COMPOSE,
                            stride=2)
    assert strided == contiguous                     # same first frame: the GPU time to it is the same
    assert batch_frames(strided, 4, 80, 2)[-1] - strided == 6   # but the batch covers 7 frames, not 4
