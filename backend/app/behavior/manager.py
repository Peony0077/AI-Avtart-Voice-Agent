"""BehaviorManager: turns LLM metadata + conversation state into avatar behaviour.

The LLM only suggests an emotion/gesture/intensity. This module validates it and
maps it to renderer-agnostic controls (expression, gaze, blink rate, head motion,
gesture intensity). Renderers use whatever subset they support.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace

EMOTIONS = ("neutral", "friendly", "happy", "empathetic", "concerned", "serious", "thinking", "excited")
GESTURES = ("none", "small_nod", "nod", "head_tilt", "shake", "smile")

# emotion -> (expression, gaze, blink_rate_per_min, head_motion 0..1)
_PROFILE = {
    "neutral": ("neutral", "user", 16, 0.30),
    "friendly": ("soft_smile", "user", 16, 0.40),
    "happy": ("smile", "user", 18, 0.50),
    "empathetic": ("soft_concern", "user_soft", 14, 0.25),
    "concerned": ("concern", "user", 14, 0.25),
    "serious": ("neutral_firm", "user", 12, 0.20),
    "thinking": ("thinking", "away_up", 10, 0.20),
    "excited": ("smile", "user", 20, 0.60),
}
# Conversation-state defaults while no reply is being spoken.
_STATE_DEFAULT = {
    "LISTENING": ("friendly", "small_nod", 0.3),
    "THINKING": ("thinking", "none", 0.3),
    "IDLE": ("neutral", "none", 0.2),
}


@dataclass
class Behavior:
    emotion: str = "neutral"
    gesture: str = "none"
    intensity: float = 0.4
    expression: str = "neutral"
    gaze: str = "user"
    blink_rate: int = 16
    head_motion: float = 0.3
    source: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp(value, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def from_meta(meta: dict | None) -> Behavior:
    meta = meta or {}
    emotion = str(meta.get("emotion", "neutral")).lower().strip()
    if emotion not in EMOTIONS:
        emotion = "neutral"
    gesture = str(meta.get("gesture", "none")).lower().strip()
    if gesture not in GESTURES:
        gesture = "none"
    intensity = _clamp(meta.get("intensity", 0.4), 0.0, 1.0, 0.4)
    # Keep motion subtle: scale head motion by intensity, never above the profile maximum.
    expression, gaze, blink, head = _PROFILE[emotion]
    return Behavior(emotion=emotion, gesture=gesture, intensity=round(intensity, 2), expression=expression,
                    gaze=gaze, blink_rate=blink, head_motion=round(head * (0.5 + 0.5 * intensity), 2),
                    source="llm" if meta else "default")


def for_state(state: str) -> Behavior:
    emotion, gesture, intensity = _STATE_DEFAULT.get(state, _STATE_DEFAULT["IDLE"])
    behavior = from_meta({"emotion": emotion, "gesture": gesture, "intensity": intensity})
    behavior.source = f"state:{state}"
    return behavior


# The reply's first words show its stance more reliably than the tag the LLM writes before them: a
# 4B model picks the tag before it "knows" whether its answer is yes or no (scripts/bench_gestures.py:
# it never chose nod or shake for 28 yes/no prompts; asked to, it nodded at "No, ..." answers).
_TEXT_GESTURES = (
    ("shake", re.compile(r"^(no|nope|nah)\b(?! (problem|worries|doubt))|^not (quite|really|exactly|at all|true)\b"
                         r"|^i'?m afraid not\b|^unfortunately,? (no|not)\b")),
    ("nod", re.compile(r"^(yes|yeah|yep|yup|absolutely|exactly|definitely|of course|certainly)\b"
                       r"|^(sure|right|correct|indeed|true)[,.!]|^that'?s (right|correct|true)\b")),
    ("small_nod", re.compile(r"^(i'?m|i am) (so |really |very |truly )?sorry\b")),
)
_FILLER = re.compile(r"^(oh|well|hmm+|ah|um+|uh)\b[,.!]?\s*")


def gesture_from_text(first_clause: str) -> str | None:
    """Head gesture implied by how a reply starts ("Yes, ..." -> nod, "No, ..." -> shake,
    "I'm sorry ..." -> small nod, an opening question -> head tilt), or None."""
    text = first_clause.strip().replace("’", "'")
    t = _FILLER.sub("", text.lower().lstrip("\"'([ "), count=1)
    for gesture, pattern in _TEXT_GESTURES:
        if pattern.search(t):
            return gesture
    if text.endswith("?"):
        return "head_tilt"
    return None


def refine_gesture(behavior: Behavior, first_clause: str) -> Behavior | None:
    """The behaviour with the gesture implied by the first clause, or None if nothing changes."""
    gesture = gesture_from_text(first_clause)
    if gesture is None or gesture == behavior.gesture:
        return None
    intensity = max(behavior.intensity, 0.5) if gesture in ("nod", "shake") else behavior.intensity
    return replace(behavior, gesture=gesture, intensity=round(intensity, 2), source="llm+text")
