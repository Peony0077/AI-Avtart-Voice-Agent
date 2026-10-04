"""Avatar head gestures: curves (avatar/musetalk/motion.py) and scheduling (gestures.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "avatar" / "musetalk"))
from gestures import choose_gesture, cycle_windows, earliest_start, next_window  # noqa: E402
from motion import GESTURE_VARIANTS, GESTURES, gesture_delta, gesture_windows, idle_curves, loop_params  # noqa: E402


@pytest.mark.parametrize("name", list(GESTURES))
def test_gesture_curves_start_and_end_at_rest_and_stay_subtle(name):
    d = gesture_delta(name, 20)
    for axis, values in d.items():
        assert abs(values[0]) < 1e-9 and abs(values[-1]) < 1e-9, (name, axis)   # joins the loop seamlessly
        assert np.abs(values).max() <= 6.0 + 1e-9                                 # degrees: subtle
    assert any(np.abs(v).max() > 2.0 for v in d.values())                         # but visible


@pytest.mark.parametrize("name", list(GESTURES))
def test_windows_fit_inside_the_loop_and_do_not_overlap(name):
    w = gesture_windows(140, 20, name)
    assert w and w[0][0] >= 1 and w[-1][1] <= 139
    assert all(a[1] < b[0] for a, b in zip(w, w[1:], strict=False))


def test_gesture_variants_only_use_existing_expressions():
    from motion import EXPRESSIONS
    assert all(set(ex) <= set(EXPRESSIONS) for ex in GESTURE_VARIANTS.values())


def test_gesture_params_only_change_head_pose_inside_windows():
    from motion import gesture_params
    curves = idle_curves(140, 20)
    base = loop_params(curves, "friendly")
    changed = gesture_params(curves, "friendly", "nod", 20)
    inside = {f for s, e in gesture_windows(140, 20, "nod") for f in range(s, e)}
    assert set(changed) == inside
    for f, p in changed.items():
        assert (p.smile, p.eyebrow, p.eye_open) == (base[f].smile, base[f].eyebrow, base[f].eye_open)


def test_cycle_windows_mirror_the_forward_half():
    assert cycle_windows([(4, 16)], 140) == [(4, 16), (264, 276)]


def test_next_window_wraps_into_the_next_cycle():
    wins = cycle_windows([(4, 16)], 140)
    assert next_window(0, wins, 280) == (4, 16)
    assert next_window(5, wins, 280) == (264, 276)
    assert next_window(270, wins, 280) == (284, 296)          # next cycle
    assert next_window(1000, wins, 280) == (1104, 1116)       # 1000 is in cycle 3 (840..1119): mirrored window


def test_choose_gesture():
    avail = {"small_nod", "nod", "head_tilt"}
    assert choose_gesture("nod", 0.8, avail) == "nod"
    assert choose_gesture("nod", 0.3, avail) == "small_nod"            # gentle request -> small nod
    assert choose_gesture("nod", 0.8, {"small_nod"}) == "small_nod"    # fallback when not rendered
    assert choose_gesture("shake", 0.9, avail) is None                 # not rendered for this expression
    assert choose_gesture("smile", 0.9, avail) is None                 # an expression, not a head motion
    assert choose_gesture("head_tilt", 0.1, avail) is None             # too faint
    assert choose_gesture("none", 1.0, avail) is None


def test_earliest_start_rules():
    playing, scheduled, done = ("small_nod", 100, 112), ("small_nod", 130, 142), ("small_nod", 40, 52)
    now = 105
    # nothing scheduled: go
    assert earliest_start(None, now, 110, state_default=True, min_gap=100) == 110
    # a gesture is playing: never cut short; a reply's gesture follows it, a state default is dropped
    assert earliest_start(playing, now, 106, state_default=False, min_gap=0) == 112
    assert earliest_start(playing, now, 106, state_default=True, min_gap=100) is None
    # only scheduled: the reply's own gesture replaces the default one, a state default does not
    assert earliest_start(scheduled, now, 108, state_default=False, min_gap=0) == 108
    assert earliest_start(scheduled, now, 108, state_default=True, min_gap=100) is None
    # finished: state defaults keep their distance, replies do not wait
    assert earliest_start(done, now, 110, state_default=True, min_gap=100) is None
    assert earliest_start(done, now, 160, state_default=True, min_gap=100) == 160
    assert earliest_start(done, now, 110, state_default=False, min_gap=0) == 110


# Seamless loops (new builds): 30 s played forward, no ping-pong rewind (owner: "not fully realistic").
def test_seamless_curves_join_without_a_seam():
    c = idle_curves(600, 20, periodic=True)
    for axis, v in c.items():
        steps = np.abs(np.diff(v))
        # the step from the last frame back to the first is no bigger than an ordinary step
        assert abs(v[0] - v[-1]) <= max(steps.max(), 1e-6) + 1e-9, axis


def test_seamless_blinks_are_irregular_and_never_cross_the_seam():
    e = idle_curves(600, 20, periodic=True)["eye_open"]
    starts = np.flatnonzero((e[1:] < 0.5) & (e[:-1] >= 0.5))
    assert 6 <= len(starts) <= 14                                   # 12-28 a minute
    assert np.diff(starts).std() > 4                                # not a metronome
    assert e[0] == 1.0 and e[-12:].min() == 1.0                     # open across the seam


def test_seamless_gaze_holds_and_jumps_like_real_eyes():
    gx = idle_curves(600, 20, periodic=True)["gaze_x"]
    steps = np.abs(np.diff(gx))
    assert (steps < 0.1).mean() > 0.8                               # mostly fixations (small drift) ...
    assert steps.max() > 0.5                                        # ... and quick saccades


def test_seamless_windows_play_forward_only():
    w = gesture_windows(600, 20, "small_nod", gap=12)
    assert cycle_windows(w, 600, pingpong=False) == w
    start, end = next_window(595, w, 600)                           # wraps into the next loop cycle
    assert start >= 600 and end - start == w[0][1] - w[0][0]


def test_talking_windows_each_move_differently_and_join_the_idle_motion():
    a, b = gesture_delta("talk", 20, seed=4), gesture_delta("talk", 20, seed=56)
    assert not np.allclose(a["yaw"], b["yaw"])                      # chained windows do not repeat
    for d in (a, b):
        for v in d.values():
            assert abs(v[0]) < 1e-9 and abs(v[-1]) < 1e-9           # starts and ends on the idle motion
            assert np.abs(v).max() <= 2.2 + 1e-9                    # small: talking, not gesturing
    w = gesture_windows(600, 20, "talk")
    assert all(b[0] - a[1] <= 4 for a, b in zip(w, w[1:], strict=False))   # nearly back to back
