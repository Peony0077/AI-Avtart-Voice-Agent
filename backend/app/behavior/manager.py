"""BehaviorManager: turns LLM metadata + conversation state into avatar behaviour.

The LLM only suggests an emotion/gesture/intensity. This module validates it and
maps it to renderer-agnostic controls (expression, gaze, blink rate, head motion,
gesture intensity). Renderers use whatever subset they support.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

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
