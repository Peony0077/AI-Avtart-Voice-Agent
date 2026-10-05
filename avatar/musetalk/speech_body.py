"""Breathing that follows the speech (pure functions; the worker applies them to lip-synced frames).

The loops breathe on a fixed rhythm (body_motion.py: 7 breaths per 30 s). People who talk breathe in
the pauses between their sentences instead: the shoulders rise in a pause and settle slowly while they
speak. Per frame of a reply, from the loudness of its audio: in a pause the breath level rises to 1
over INHALE_S, while speaking it falls to 0 over EXHALE_S; the shoulders are lifted by up to
BREATH_PX. The level carries over from sentence to sentence, and the reply's closing tail brings it
back to 0, where the idle loop continues without a jump.
"""
from __future__ import annotations

import numpy as np

INHALE_S = 0.45          # a quick breath in a pause ...
EXHALE_S = 2.2           # ... let out slowly while speaking
BREATH_PX = 2.2          # shoulders lifted at a full breath (512 px frame)
SILENT_RMS = 0.012       # audio below this (float, -1..1) is a pause


def frame_rms(pcm16: np.ndarray, samples_per_frame: int, n_frames: int) -> np.ndarray:
    """Loudness of each video frame's audio."""
    x = pcm16[: n_frames * samples_per_frame].astype(np.float32) / 32768.0
    x = np.pad(x, (0, max(0, n_frames * samples_per_frame - len(x))))
    return np.sqrt(np.mean(x.reshape(n_frames, samples_per_frame) ** 2, axis=1))


def breath_levels(rms: np.ndarray, fps: float, start: float = 0.0) -> np.ndarray:
    """Breath level 0..1 per frame, starting from ``start`` (the previous sentence's last level)."""
    up, down = 1.0 / (INHALE_S * fps), 1.0 / (EXHALE_S * fps)
    out = np.empty(len(rms), np.float32)
    level = float(start)
    for k, r in enumerate(rms):
        level = min(1.0, level + up) if r < SILENT_RMS else max(0.0, level - down)
        out[k] = level
    return out


def to_lift(levels: np.ndarray) -> np.ndarray:
    """Body shift in px (negative = up), eased so the shoulders start and stop softly."""
    x = np.clip(levels, 0.0, 1.0)
    return -BREATH_PX * x * x * (3 - 2 * x)


def settle(start: float, n: int) -> np.ndarray:
    """Levels over the closing tail: from ``start`` down to exactly 0 at the last frame."""
    if n <= 1:
        return np.zeros(max(n, 0), np.float32)
    return np.linspace(start, 0.0, n).astype(np.float32)


GAP_PX = 2               # as body_motion: the motion starts just below the face box ...
RAMP_PX = 45             # ... and eases in over this many rows
_identity: dict[tuple[int, int], np.ndarray] = {}


def lift_body(frame: np.ndarray, face_bottom: int, dy: float) -> np.ndarray:
    """The image below the face moved up by -dy px (dy < 0), easing in from 0 at the face box - the same
    shape as body_motion.move_body, for this vertical-only case at run time: only the rows below the
    face are warped and the x map is reused (3.5 ms -> ~1 ms a frame; the compose stage sets how many
    lip-synced frames arrive in time)."""
    import cv2

    h, w = frame.shape[:2]
    y0 = min(h - 1, max(0, face_bottom + GAP_PX))
    n = h - y0
    if n <= 1 or abs(dy) < 0.05:
        return frame
    rows = np.arange(y0, h, dtype=np.float32)
    weight = np.clip((rows - y0) / RAMP_PX, 0, 1)
    weight = weight * weight * (3 - 2 * weight)
    map_y = np.ascontiguousarray(np.broadcast_to((rows - dy * weight)[:, None], (n, w)))
    map_x = _identity.get((n, w))
    if map_x is None:
        map_x = _identity[(n, w)] = np.ascontiguousarray(np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (n, w)))
    # in place: the callers pass a freshly composed frame (remap returns a new array before the copy-in)
    frame[y0:] = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return frame
