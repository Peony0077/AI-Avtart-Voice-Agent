"""Head-gesture scheduling for the avatar worker (pure functions, unit-tested in backend/tests).

Gestures are pre-rendered into *variants* of the expression loops that differ only inside short
windows (motion.gesture_windows). The worker plays a gesture by showing the variant's frames for the
next window; outside windows the variant and the base loop are identical, so there is no jump.
Loops play ping-pong (forward, then backward), so every forward window also exists mirrored.
"""
from __future__ import annotations

MIN_INTENSITY = 0.2            # quieter requests are ignored ("keep motions subtle")
FULL_NOD_INTENSITY = 0.45      # below this a "nod" is played as a small nod


def cycle_windows(forward: list[tuple[int, int]], n_forward: int) -> list[tuple[int, int]]:
    """Windows in loop-index space over one ping-pong cycle of 2 * n_forward frames."""
    n2 = 2 * n_forward
    return sorted([(s, e) for s, e in forward] + [(n2 - e, n2 - s) for s, e in forward])


def next_window(after: int, windows: list[tuple[int, int]], cycle: int) -> tuple[int, int]:
    """First window (absolute frame indices) that starts at or after frame ``after``."""
    if not windows:
        raise ValueError("no gesture windows")
    base = (after // cycle) * cycle
    for offset in (base, base + cycle):
        for s, e in windows:
            if offset + s >= after:
                return offset + s, offset + e
    raise AssertionError("unreachable: the next cycle always has a window")


def earliest_start(current: tuple[str, int, int] | None, now: int, after: int, state_default: bool,
                   min_gap: int) -> int | None:
    """Earliest frame a newly requested gesture may start at, or None to drop the request.

    ``current`` is the last scheduled gesture (name, start, end). A gesture that is playing is never cut
    short; a reply's own gesture (state_default=False) replaces one that is only scheduled and follows
    one that is playing; state defaults (e.g. the listening nod) never replace anything and keep
    ``min_gap`` frames after the previous gesture."""
    if current is None:
        return after
    _, start, end = current
    if start <= now < end:                       # playing
        return None if state_default else max(after, end)
    if start > now:                              # scheduled, not started yet
        return None if state_default else after
    if state_default and after < end + min_gap:  # the previous one has just finished
        return None
    return after


def choose_gesture(name: str | None, intensity: float, available: set[str]) -> str | None:
    """BehaviorManager gesture + intensity -> the rendered gesture to play (None = none)."""
    if not name or name in ("none", "smile") or intensity < MIN_INTENSITY:   # smile is an expression
        return None
    if name == "nod" and (intensity < FULL_NOD_INTENSITY or "nod" not in available):
        name = "small_nod"
    return name if name in available else None
