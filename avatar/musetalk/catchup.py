"""Keeping lip-sync frames on time when the GPU cannot quite keep up (pure functions, unit-tested).

The browser shows, for every screen refresh, the newest speech frame whose pts is not in the
future of the audio. A frame that arrives after its audio has played is useless, and rendering it
only makes the next ones later still. So when the renderer is behind, the worker skips frames that
could only arrive more than ``late_ms`` late; the lips then update a little less often but stay in
sync. Measured on a busy PC: 12-20 % late frames without this, 3.9 % with it (docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import math

STALL_MS = 250.0   # the browser reports playback every ~100 ms; no report for longer = playback stalled


def audible_ms(played_ms: float, played_at: float, now: float, at: float | None = None) -> float:
    """Playback position (ms on the turn's timeline) at time ``at`` (default ``now``), from the last
    browser report ``played_ms`` received at ``played_at``. Only the time since that report is capped
    (stalled playback must not be extrapolated); the look-ahead ``at - now`` always counts in full."""
    at = now if at is None else at
    return played_ms + min((now - played_at) * 1000, STALL_MS) + (at - now) * 1000


MAX_STRIDE = 3    # render at least every 3rd frame (8 lip updates per second at 25 fps)


def batch_frames(s: int, size: int, n: int, stride: int = 1) -> list[int]:
    """The frames of a batch: up to ``size`` frames from s, every ``stride``-th, below n."""
    return list(range(s, n, stride))[:size]


def stride_for(gpu_ms_per_frame: float, fps: float) -> int:
    """Render every frame while the GPU keeps up with real time; when it is slower (another model
    on the GPU, e.g. the LLM still writing the reply), every 2nd or 3rd frame. The lips then move
    evenly at a lower rate instead of in bursts between skipped stretches (measured before this:
    160 ms of movement, then a 240 ms freeze, at the start of long replies). A GPU up to 25 % slower
    than real time keeps every frame (the occasional skip handles that); beyond, the stride leaves
    ~15 % headroom, since a stride that only just keeps up still falls behind at every batch start."""
    ratio = gpu_ms_per_frame * fps / 1000          # GPU time per frame / time per frame
    return 1 if ratio <= 1.25 else min(MAX_STRIDE, math.ceil(ratio * 1.15))


def first_on_time(s: int, size: int, n: int, seg_pts: float, fps: float, late_ms: float, audible_now: float,
                  gpu_ms_per_frame: float, compose_ms: float, stride: int = 1) -> int:
    """First frame index in [s, n) worth rendering as the start of a batch of ``size`` (``n`` if none).

    Frames go out one by one after the GPU stage of their batch: the first frame of the batch
    starting at k is ready after its frames' GPU time plus its own compositing. The batch slides
    forward over the segment's n frames until that first frame would arrive at most ``late_ms`` after
    its audio. It keeps its size: shrinking it instead (the first version) rendered single frames
    under load, the least efficient batch on the GPU (115-150 ms per frame while the LLM was also
    generating), so the worker fell further behind and showed only 1 frame in 4.
    """
    k = s
    while k < n and (seg_pts + k * 1000 / fps + late_ms
                     < audible_now + len(batch_frames(k, size, n, stride)) * gpu_ms_per_frame + compose_ms):
        k += 1
    return k
