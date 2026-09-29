"""Procedural idle motion for the face rig: blinks, subtle head drift, micro-saccades.

All emotion loops of an avatar share the same seed → identical head/blink/gaze
timing; they differ only in expression offsets. The worker can therefore switch
between loops at the same frame index without a visible jump in head position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RigParams:
    pitch: float = 0.0      # degrees, relative to the photo
    yaw: float = 0.0
    roll: float = 0.0
    eye_open: float = 1.0   # 1 = as in the photo, 0 = closed
    gaze_x: float = 0.0     # LivePortrait eyeball units (≈ -15..15)
    gaze_y: float = 0.0
    smile: float = 0.0      # ≈ -0.3..1.2
    eyebrow: float = 0.0    # ≈ -15..15


# name -> expression offsets (tuned on the dev portrait; LivePortrait retargeting units)
EXPRESSIONS: dict[str, dict] = {
    "neutral": {"smile": 0.0, "eyebrow": 0.0},
    "friendly": {"smile": 0.35, "eyebrow": 2.0},
    "happy": {"smile": 0.85, "eyebrow": 4.0},
    "concerned": {"smile": -0.15, "eyebrow": -8.0},
    "thinking": {"smile": 0.0, "eyebrow": 3.0, "gaze": (-7.0, 5.0), "yaw": -1.5, "pitch": -1.0},
}

# BehaviorManager expression names (backend) -> loop names
EXPRESSION_TO_LOOP = {
    "neutral": "neutral", "neutral_firm": "neutral", "soft_smile": "friendly", "smile": "happy",
    "soft_concern": "concerned", "concern": "concerned", "thinking": "thinking",
}

_BLINK = np.array([1.0, 0.55, 0.12, 0.0, 0.08, 0.35, 0.7, 0.92, 1.0])  # eye openness at 25 fps


def _smooth_noise(n: int, fps: float, rng: np.random.Generator, freqs=(0.07, 0.13, 0.23)) -> np.ndarray:
    t = np.arange(n) / fps
    sig = sum(rng.uniform(0.6, 1.0) * np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi)) for f in freqs)
    return sig / (np.abs(sig).max() + 1e-6)


def idle_curves(n: int, fps: float, seed: int = 7, head_amp=(1.8, 2.6, 1.0), blinks_per_min: float = 16) -> dict:
    """Shared motion curves (length n)."""
    rng = np.random.default_rng(seed)
    curves = {
        "pitch": head_amp[0] * _smooth_noise(n, fps, rng),
        "yaw": head_amp[1] * _smooth_noise(n, fps, rng),
        "roll": head_amp[2] * _smooth_noise(n, fps, rng),
        "eye_open": np.ones(n),
        "gaze_x": 1.2 * _smooth_noise(n, fps, rng, freqs=(0.3, 0.55)),
        "gaze_y": 0.8 * _smooth_noise(n, fps, rng, freqs=(0.25, 0.5)),
    }
    # blinks: jittered spacing (2.0-3.6 s) → ~17-26 blinks/min; none at the loop ends (ping-pong playback)
    blink = np.interp(np.linspace(0, len(_BLINK) - 1, max(4, round(len(_BLINK) * fps / 25))),
                      np.arange(len(_BLINK)), _BLINK)
    t = rng.uniform(0.6, 1.2)
    while (start := int(t * fps)) + len(blink) < n - int(0.4 * fps):
        curves["eye_open"][start:start + len(blink)] = np.minimum(curves["eye_open"][start:start + len(blink)], blink)
        t += rng.uniform(2.0, 3.6) * 16.0 / blinks_per_min
    return curves


def loop_params(curves: dict, expression: str) -> list[RigParams]:
    e = EXPRESSIONS[expression]
    gx, gy = e.get("gaze", (0.0, 0.0))
    n = len(curves["pitch"])
    # ease into an averted gaze for "thinking" so it doesn't snap
    return [
        RigParams(
            pitch=float(curves["pitch"][i] + e.get("pitch", 0.0)),
            yaw=float(curves["yaw"][i] + e.get("yaw", 0.0)),
            roll=float(curves["roll"][i]),
            eye_open=float(curves["eye_open"][i]),
            gaze_x=float(curves["gaze_x"][i] + gx),
            gaze_y=float(curves["gaze_y"][i] + gy),
            smile=e["smile"],
            eyebrow=e["eyebrow"],
        )
        for i in range(n)
    ]
