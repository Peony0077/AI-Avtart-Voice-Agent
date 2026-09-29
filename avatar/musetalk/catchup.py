"""Keeping lip-sync frames on time when the GPU cannot quite keep up (pure functions, unit-tested).

The browser shows, for every screen refresh, the newest speech frame whose pts is not in the
future of the audio. A frame that arrives after its audio has played is useless, and rendering it
only makes the next ones later still. So when the renderer is behind, the worker skips frames that
could only arrive more than ``late_ms`` late; the lips then update a little less often but stay in
sync. Measured on a busy PC: 12-20 % late frames without this, 3.9 % with it (docs/ARCHITECTURE.md).
"""
from __future__ import annotations

STALL_MS = 250.0   # the browser reports playback every ~100 ms; no report for longer = playback stalled


def audible_ms(played_ms: float, played_at: float, now: float, at: float | None = None) -> float:
    """Playback position (ms on the turn's timeline) at time ``at`` (default ``now``), from the last
    browser report ``played_ms`` received at ``played_at``. Only the time since that report is capped
    (stalled playback must not be extrapolated); the look-ahead ``at - now`` always counts in full."""
    at = now if at is None else at
    return played_ms + min((now - played_at) * 1000, STALL_MS) + (at - now) * 1000


def first_on_time(s: int, e: int, seg_pts: float, fps: float, late_ms: float, audible_now: float,
                  gpu_ms_per_frame: float, compose_ms: float) -> int:
    """First frame index in [s, e) that is worth rendering (``e`` if none).

    Frames go out one by one after the GPU stage of their batch: the first frame of a batch that
    starts at k is ready after (e - k) GPU frames plus its own compositing. Frames are dropped from the
    front of the batch until that first frame would arrive at most ``late_ms`` after its audio.
    """
    k = s
    while k < e and seg_pts + k * 1000 / fps + late_ms < audible_now + (e - k) * gpu_ms_per_frame + compose_ms:
        k += 1
    return k
