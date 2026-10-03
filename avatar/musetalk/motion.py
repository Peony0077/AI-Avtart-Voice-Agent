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
# "eye" scales eye openness (blinks stay complete): slightly narrowed eyes read as a real, warm smile.
# friendly is the avatar's default face (idle, listening, ordinary replies), so it must look warm:
# the earlier 0.35 smile was not visible next to neutral (logs/expressions_now.jpg vs smile_trials.jpg).
EXPRESSIONS: dict[str, dict] = {
    "neutral": {"smile": 0.0, "eyebrow": 0.0},
    "friendly": {"smile": 0.6, "eyebrow": 3.0, "eye": 0.92},
    "happy": {"smile": 1.15, "eyebrow": 5.0, "eye": 0.85},
    "concerned": {"smile": -0.15, "eyebrow": -8.0},
    "thinking": {"smile": 0.0, "eyebrow": 3.0, "gaze": (-7.0, 5.0), "yaw": -1.5, "pitch": -1.0},
}

# BehaviorManager expression names (backend) -> loop names
EXPRESSION_TO_LOOP = {
    "neutral": "neutral", "neutral_firm": "neutral", "soft_smile": "friendly", "smile": "happy",
    "soft_concern": "concerned", "concern": "concerned", "thinking": "thinking",
}

# Head gestures (degrees; + pitch = head down), as the BehaviorManager names them. They are rendered
# offline into *gesture variants* of the expression loops: identical frames outside short gesture
# windows, the gesture inside them. At run time the worker plays the variant's frames for the next
# window when a gesture is requested (gestures.py). Kept small on purpose ("subtle and natural").
GESTURES: dict[str, dict] = {
    "small_nod": {"seconds": 0.6, "pitch": 4.0, "cycles": 1},
    "nod": {"seconds": 1.1, "pitch": 6.0, "cycles": 2},
    "head_tilt": {"seconds": 1.6, "roll": 5.0, "hold": 0.5},
    "shake": {"seconds": 1.0, "yaw": 5.5, "cycles": 1.5},
}
# which expression loops get which gesture (natural pairings; each variant costs build time and RAM)
GESTURE_VARIANTS: dict[str, tuple[str, ...]] = {
    "small_nod": ("neutral", "friendly", "happy", "concerned"),
    "nod": ("neutral", "friendly", "happy", "concerned"),            # concerned: the empathetic nod
    "head_tilt": ("neutral", "friendly", "concerned", "thinking"),   # neutral: "serious" replies tilt too
    "shake": ("neutral", "concerned"),
}
GESTURE_GAP_FRAMES = 8        # frames between two windows of the same gesture
GESTURE_MARGIN_FRAMES = 4     # keep windows away from the loop ends (ping-pong turning points)


def gesture_length(name: str, fps: float) -> int:
    return round(GESTURES[name]["seconds"] * fps)


def gesture_delta(name: str, fps: float) -> dict[str, np.ndarray]:
    """Head-pose offsets over the gesture, per frame. They start and end at 0, so a gesture window
    joins the unchanged loop frames around it without a jump."""
    g = GESTURES[name]
    n = gesture_length(name, fps)
    t = np.linspace(0.0, 1.0, n)
    out = {axis: np.zeros(n) for axis in ("pitch", "yaw", "roll")}
    if "pitch" in g:            # nod: down and back up; a second nod is smaller
        phase = t * g["cycles"]
        amp = g["pitch"] * 0.6 ** np.minimum(np.floor(phase), g["cycles"] - 1)
        out["pitch"] = amp * 0.5 * (1 - np.cos(2 * math.pi * phase))
    if "roll" in g:             # tilt: ease in, hold, ease out
        ramp = (1 - g["hold"]) / 2
        env = np.clip(np.minimum(t / ramp, (1 - t) / ramp), 0.0, 1.0)
        out["roll"] = g["roll"] * 0.5 * (1 - np.cos(math.pi * env))
    if "yaw" in g:              # shake: side to side inside a smooth envelope
        out["yaw"] = g["yaw"] * np.sin(math.pi * t) * np.sin(2 * math.pi * g["cycles"] * t)
    return out


def gesture_windows(n_frames: int, fps: float, name: str) -> list[tuple[int, int]]:
    """[start, end) frame windows of a gesture inside a loop's forward half (n_frames frames)."""
    length = gesture_length(name, fps)
    step = length + GESTURE_GAP_FRAMES
    last = n_frames - GESTURE_MARGIN_FRAMES - length
    return [(s, s + length) for s in range(GESTURE_MARGIN_FRAMES, last + 1, step)]


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


def gesture_params(curves: dict, expression: str, gesture: str, fps: float) -> dict[int, RigParams]:
    """Rig parameters of the frames inside the gesture windows (the rest equals the expression loop)."""
    base = loop_params(curves, expression)
    delta = gesture_delta(gesture, fps)
    out: dict[int, RigParams] = {}
    for start, end in gesture_windows(len(base), fps, gesture):
        for k, f in enumerate(range(start, end)):
            p = base[f]
            out[f] = RigParams(pitch=p.pitch + float(delta["pitch"][k]), yaw=p.yaw + float(delta["yaw"][k]),
                               roll=p.roll + float(delta["roll"][k]), eye_open=p.eye_open, gaze_x=p.gaze_x,
                               gaze_y=p.gaze_y, smile=p.smile, eyebrow=p.eyebrow)
    return out


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
            eye_open=float(curves["eye_open"][i] * e.get("eye", 1.0)),
            gaze_x=float(curves["gaze_x"][i] + gx),
            gaze_y=float(curves["gaze_y"][i] + gy),
            smile=e["smile"],
            eyebrow=e["eyebrow"],
        )
        for i in range(n)
    ]
