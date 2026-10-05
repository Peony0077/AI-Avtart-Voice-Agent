"""Procedural idle motion for the face rig: blinks, head drift, eye movements.

All emotion loops of an avatar share the same seed → identical head/blink/gaze
timing; they differ only in expression offsets. The worker can therefore switch
between loops at the same frame index without a visible jump in head position.

Two kinds of loops:
  * ping-pong (first avatars): 7 s played forward then backward - every 7 s the head, eyes and
    blinks visibly rewound, and the same 14 s repeated (owner: "not fully realistic").
  * seamless (periodic=True, default for new builds): 30 s played forward and repeated. Every
    motion repeats exactly once per loop (head drift frequencies are whole cycles per loop), so the
    end joins the start without a seam. The eyes hold a point and jump to the next one like real
    eyes (fixations and saccades) instead of wobbling, blinks come at irregular intervals (sometimes
    two in a row), and a slight breathing rhythm moves the head.
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
# v3 (owner, 2026-10-05: "only small & short changes in face emotion"): stronger and whole-face -
# the happy smile reaches the eyes and lifts the chin a little, concern draws the brows and tilts the
# head, and "interested" (raised brows, open eyes, a slight tilt) is the face of active listening.
# pitch/yaw/roll: head pose offsets (degrees) held for the whole expression.
EXPRESSIONS: dict[str, dict] = {
    "neutral": {"smile": 0.0, "eyebrow": 0.0},
    "friendly": {"smile": 0.7, "eyebrow": 4.0, "eye": 0.9},
    # eye 0.99 (was 0.8): the big smile narrows the eyes by itself (~0.89); with 0.8 on top her eyes were
    # 19 % smaller than the idle face's whenever she spoke happily (owner, 2026-10-05: "whenever she
    # talks, her eye height changes very small"). Not 1.0: that switches eye retargeting off between
    # blinks, and each blink would start with the eyes jumping wider.
    "happy": {"smile": 1.3, "eyebrow": 8.0, "eye": 0.99, "pitch": -1.0, "roll": 1.5},
    "concerned": {"smile": 0.05, "eyebrow": -8.0, "eye": 0.95, "pitch": 1.0, "roll": -2.0},   # v3.1: -12 brows read as pitiful over a whole reply
    "thinking": {"smile": 0.0, "eyebrow": 5.0, "gaze": (-8.0, 6.0), "yaw": -2.5, "pitch": -1.5},
    "interested": {"smile": 0.45, "eyebrow": 11.0, "eye": 1.06, "roll": 2.0},
}

# BehaviorManager expression names (backend) -> loop names
EXPRESSION_TO_LOOP = {
    "neutral": "neutral", "neutral_firm": "neutral", "soft_smile": "friendly", "smile": "happy",
    "soft_concern": "concerned", "concern": "concerned", "thinking": "thinking", "interested": "interested",
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
    # Talking: people move their head while they speak. Windows of small, varied head movement (each
    # window its own random pattern, 0.5-1.6 Hz like speech-related motion) that the worker chains
    # while the avatar speaks; between them the head is back on the idle motion for a moment.
    # v3: larger (2.0/2.2/1.0 deg looked still) with 1-3 quick "beat" nods per window - the head dips
    # on stressed words while people talk.
    "talk": {"seconds": 2.4, "talk": {"pitch": 3.0, "yaw": 3.4, "roll": 1.6}, "beats": (1, 3), "beat_pitch": 1.8,
             "gap": 4},
}
# which expression loops get which gesture (natural pairings; each variant costs build time and RAM)
GESTURE_VARIANTS: dict[str, tuple[str, ...]] = {
    "small_nod": ("neutral", "friendly", "happy", "concerned", "interested"),   # interested: listening nods
    "nod": ("neutral", "friendly", "happy", "concerned", "interested"),         # concerned: the empathetic nod
    "head_tilt": ("neutral", "friendly", "concerned", "thinking"),   # neutral: "serious" replies tilt too
    "shake": ("neutral", "concerned"),
    "talk": ("neutral", "friendly", "happy", "concerned"),         # the loops replies are spoken with
}
GESTURE_GAP_FRAMES = 8        # frames between two windows of the same gesture (ping-pong loops)
GESTURE_GAP_FRAMES_LONG = 12  # seamless loops are 4x longer: a little sparser, to keep the RAM in check
GESTURE_MARGIN_FRAMES = 4     # keep windows away from the loop ends (turning point / seam)


def gesture_length(name: str, fps: float) -> int:
    return round(GESTURES[name]["seconds"] * fps)


def talk_delta(n: int, fps: float, seed: int) -> dict[str, np.ndarray]:
    """One window of talking head motion: smooth random movement on every axis inside an envelope
    that is 0 (and flat) at both ends. ``seed`` (the window's start frame) makes every window
    different, so chained windows do not repeat a pattern."""
    rng = np.random.default_rng(1000 + seed)
    t = np.arange(n) / fps
    env = np.sin(math.pi * np.arange(n) / max(1, n - 1)) ** 2
    out = {}
    g = GESTURES["talk"]
    for axis, amp in g["talk"].items():
        sig = sum(rng.uniform(0.5, 1.0) * np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi))
                  for f in rng.uniform(0.5, 1.6, size=2))
        out[axis] = amp * env * sig / (np.abs(sig).max() + 1e-6)
    if "beats" in g:                                # quick dips of the head (+pitch = down), 0.3 s each
        beat = 0.5 * (1 - np.cos(2 * math.pi * np.arange(max(2, round(0.3 * fps))) / max(1, round(0.3 * fps))))
        for _ in range(int(rng.integers(g["beats"][0], g["beats"][1] + 1))):
            s0 = int(rng.integers(int(0.15 * n), max(int(0.15 * n) + 1, int(0.85 * n) - len(beat))))
            out["pitch"][s0:s0 + len(beat)] += g["beat_pitch"] * rng.uniform(0.6, 1.0) * beat[: n - s0]
    return out


def gesture_delta(name: str, fps: float, seed: int = 0) -> dict[str, np.ndarray]:
    """Head-pose offsets over the gesture, per frame. They start and end at 0, so a gesture window
    joins the unchanged loop frames around it without a jump."""
    g = GESTURES[name]
    n = gesture_length(name, fps)
    if "talk" in g:
        return talk_delta(n, fps, seed)
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


def gesture_windows(n_frames: int, fps: float, name: str, gap: int = GESTURE_GAP_FRAMES) -> list[tuple[int, int]]:
    """[start, end) frame windows of a gesture inside a loop's forward part (n_frames frames)."""
    length = gesture_length(name, fps)
    step = length + GESTURES[name].get("gap", gap)
    last = n_frames - GESTURE_MARGIN_FRAMES - length
    return [(s, s + length) for s in range(GESTURE_MARGIN_FRAMES, last + 1, step)]


_BLINK = np.array([1.0, 0.55, 0.12, 0.0, 0.08, 0.35, 0.7, 0.92, 1.0])  # eye openness at 25 fps


def _smooth_noise(n: int, fps: float, rng: np.random.Generator, freqs=(0.07, 0.13, 0.23)) -> np.ndarray:
    t = np.arange(n) / fps
    sig = sum(rng.uniform(0.6, 1.0) * np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi)) for f in freqs)
    return sig / (np.abs(sig).max() + 1e-6)


def _periodic_noise(n: int, fps: float, rng: np.random.Generator, freqs=(0.07, 0.13, 0.23)) -> np.ndarray:
    """Smooth noise that repeats exactly every n frames: each frequency is rounded to a whole number
    of cycles per loop, so the last frame joins the first without a seam."""
    span = n / fps
    t = np.arange(n) / fps
    sig = np.zeros(n)
    for f in freqs:
        k = max(1, round(f * span))
        sig += rng.uniform(0.6, 1.0) * np.sin(2 * math.pi * k * t / span + rng.uniform(0, 2 * math.pi))
    return sig / (np.abs(sig).max() + 1e-6)


def _circular_smooth(x: np.ndarray, frames: float) -> np.ndarray:
    """Gaussian smoothing that wraps around the loop end (keeps a periodic curve periodic)."""
    radius = max(1, int(3 * frames))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / frames) ** 2)
    k /= k.sum()
    padded = np.concatenate([x[-radius:], x, x[:radius]])
    return np.convolve(padded, k, mode="valid")


def eye_movements(n: int, fps: float, rng: np.random.Generator, amp=(2.0, 1.2)) -> tuple[np.ndarray, np.ndarray]:
    """Gaze as real eyes move: fixations of 0.5-2.5 s, mostly near the viewer, now and then a glance
    a little aside, joined by quick saccades (~80 ms), plus a tiny drift. Periodic over n frames
    (the last fixation returns to the first one's point)."""
    gx, gy = np.zeros(n), np.zeros(n)
    t, points = 0, []
    while t < n:
        dur = int(rng.uniform(0.5, 2.5) * fps)
        if rng.random() < 0.7:          # near the centre: looking at the viewer
            x, y = rng.normal(0, 0.3 * amp[0]), rng.normal(0, 0.3 * amp[1])
        else:                           # a glance aside, more often sideways than up/down
            x, y = rng.uniform(-amp[0], amp[0]), rng.uniform(-0.6 * amp[1], 0.6 * amp[1])
        points.append((t, min(n, t + dur), x, y))
        t += dur
    for k, (s, e, x, y) in enumerate(points):
        if k == len(points) - 1 and len(points) > 1:
            x, y = points[0][2], points[0][3]      # the seam: back to where the loop starts
        gx[s:e], gy[s:e] = x, y
    drift_x = 0.15 * amp[0] * _periodic_noise(n, fps, rng, freqs=(0.3, 0.55))
    drift_y = 0.15 * amp[1] * _periodic_noise(n, fps, rng, freqs=(0.25, 0.5))
    sacc = 0.03 * fps                               # ~80 ms transitions at any frame rate
    return _circular_smooth(gx, sacc) + drift_x, _circular_smooth(gy, sacc) + drift_y


def blink_curve(n: int, fps: float, rng: np.random.Generator, blinks_per_min: float = 15) -> np.ndarray:
    """Eye openness with blinks at irregular intervals (1.6-5.6 s at 15/min), each a little different
    in speed, and now and then two in a row; none across the loop seam."""
    eye = np.ones(n)
    scale = 15.0 / blinks_per_min
    t = rng.uniform(0.4, 1.5)
    while True:
        length = rng.uniform(0.85, 1.25)            # slower or quicker blink
        blink = np.interp(np.linspace(0, len(_BLINK) - 1, max(4, round(len(_BLINK) * length * fps / 25))),
                          np.arange(len(_BLINK)), _BLINK)
        start = int(t * fps)
        if start + len(blink) >= n - int(0.6 * fps):
            break
        eye[start:start + len(blink)] = np.minimum(eye[start:start + len(blink)], blink)
        if rng.random() < 0.12:                     # a double blink
            second = start + len(blink) + int(rng.uniform(0.12, 0.25) * fps)
            if second + len(blink) < n - int(0.6 * fps):
                eye[second:second + len(blink)] = np.minimum(eye[second:second + len(blink)], blink)
                t = second / fps
        t += rng.uniform(1.6, 5.6) * scale
    return eye


def _min_jerk(x: np.ndarray) -> np.ndarray:
    """0 -> 1 with zero speed and acceleration at both ends (how people move their head)."""
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (10 - 15 * x + 6 * x * x)


def head_shifts(n: int, fps: float, rng: np.random.Generator, amp=(1.6, 2.6, 1.2)) -> dict[str, np.ndarray]:
    """Short natural head movements (owner: "there is no short natural movement of the head"): every
    2-6 s the head moves to a slightly different position in 0.35-0.7 s and stays there, as people
    resettle their head while they listen. Periodic over n frames: the last move returns to the start."""
    out = {axis: np.zeros(n) for axis in ("pitch", "yaw", "roll")}
    t = int(rng.uniform(1.0, 3.0) * fps)
    moves = []
    while t < n - int(1.0 * fps):
        target = np.array([rng.uniform(-1, 1) * amp[0], rng.uniform(-1, 1) * amp[1], rng.uniform(-1, 1) * amp[2]])
        moves.append((t, int(rng.uniform(0.35, 0.7) * fps), target))
        t += int(rng.uniform(2.0, 6.0) * fps)
    if moves:                                       # the seam: the last move goes back to the start pose
        s, d, _ = moves[-1]
        moves[-1] = (s, d, np.zeros(3))
    cur = np.zeros(3)
    for k, (start, dur, target) in enumerate(moves):
        end = moves[k + 1][0] if k + 1 < len(moves) else n
        ramp = _min_jerk(np.arange(end - start) / max(1, dur))[:, None]
        seg = cur + (target - cur) * ramp
        for i, axis in enumerate(("pitch", "yaw", "roll")):
            out[axis][start:end] = seg[:, i]
        cur = target
    return out


def head_follows_eyes(gaze_x: np.ndarray, gaze_y: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Yaw and pitch (degrees) of the head following the eyes a little: a glance aside turns the head
    slightly, ~120 ms after the eyes and more slowly (eye-head coordination)."""
    lag = int(round(0.12 * fps))
    yaw = _circular_smooth(np.roll(gaze_x, lag), 0.12 * fps)
    pitch = _circular_smooth(np.roll(gaze_y, lag), 0.12 * fps)
    return 0.7 * yaw, -0.4 * pitch


def idle_curves(n: int, fps: float, seed: int = 7, head_amp=(1.8, 2.6, 1.0), blinks_per_min: float = 16,
                periodic: bool = False, lively: bool = True) -> dict:
    """Shared motion curves (length n). periodic: a seamless loop (see the module docstring).
    lively (seamless loops): besides the slow drift, short head moves (head_shifts), a fine fast
    micro-motion and the head following the eyes - the slow drift alone (0.07-0.23 Hz) looked like a
    still head (owner, 2026-10-05)."""
    rng = np.random.default_rng(seed)
    if periodic:
        span = n / fps
        breath = 0.25 * np.sin(2 * math.pi * max(1, round(0.24 * span)) * np.arange(n) / n)   # ~0.24 Hz
        gaze_x, gaze_y = eye_movements(n, fps, rng)
        curves = {
            "pitch": head_amp[0] * _periodic_noise(n, fps, rng) + breath,
            "yaw": head_amp[1] * _periodic_noise(n, fps, rng),
            "roll": head_amp[2] * _periodic_noise(n, fps, rng),
            "eye_open": blink_curve(n, fps, rng),
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
        }
        if lively:
            lrng = np.random.default_rng(seed + 100)       # the drift, blinks and gaze stay as they were
            drift = 0.7                                     # the slow drift a little smaller: the moves add up
            shifts = head_shifts(n, fps, lrng)
            fy, fp = head_follows_eyes(gaze_x, gaze_y, fps)
            micro = {a: amp * _periodic_noise(n, fps, lrng, freqs=(0.45, 0.8, 1.25))
                     for a, amp in (("pitch", 0.35), ("yaw", 0.4), ("roll", 0.25))}
            curves["pitch"] = drift * (curves["pitch"] - breath) + breath + shifts["pitch"] + micro["pitch"] + fp
            curves["yaw"] = drift * curves["yaw"] + shifts["yaw"] + micro["yaw"] + fy
            curves["roll"] = drift * curves["roll"] + shifts["roll"] + micro["roll"]
        return curves
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


def gesture_params(curves: dict, expression: str, gesture: str, fps: float,
                   gap: int = GESTURE_GAP_FRAMES) -> dict[int, RigParams]:
    """Rig parameters of the frames inside the gesture windows (the rest equals the expression loop)."""
    base = loop_params(curves, expression)
    out: dict[int, RigParams] = {}
    for start, end in gesture_windows(len(base), fps, gesture, gap):
        delta = gesture_delta(gesture, fps, seed=start)
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
            roll=float(curves["roll"][i] + e.get("roll", 0.0)),
            eye_open=float(curves["eye_open"][i] * e.get("eye", 1.0)),
            gaze_x=float(curves["gaze_x"][i] + gx),
            gaze_y=float(curves["gaze_y"][i] + gy),
            smile=e["smile"],
            eyebrow=e["eyebrow"],
        )
        for i in range(n)
    ]
