"""BehaviorManager: turns LLM metadata + conversation state into avatar behaviour.

The LLM only suggests an emotion/gesture/intensity. This module validates it and
maps it to renderer-agnostic controls (expression, gaze, blink rate, head motion,
gesture intensity). Renderers use whatever subset they support.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace

EMOTIONS = ("neutral", "friendly", "happy", "empathetic", "concerned", "serious", "thinking", "excited", "interested")
GESTURES = ("none", "small_nod", "nod", "head_tilt", "shake", "smile")

# emotion -> (expression, gaze, blink_rate_per_min, head_motion 0..1)
# Warm persona: a "neutral" reply keeps the soft smile; the plain face is for serious content only.
_PROFILE = {
    "neutral": ("soft_smile", "user", 16, 0.30),
    "friendly": ("soft_smile", "user", 16, 0.40),
    "happy": ("smile", "user", 18, 0.50),
    "empathetic": ("soft_concern", "user_soft", 14, 0.25),
    "concerned": ("concern", "user", 14, 0.25),
    "serious": ("neutral_firm", "user", 12, 0.20),
    "thinking": ("thinking", "away_up", 10, 0.20),
    "excited": ("smile", "user", 20, 0.60),
    "interested": ("interested", "user", 16, 0.35),     # listening: raised brows, open eyes, slight tilt
}
# Conversation-state defaults while no reply is being spoken.
_STATE_DEFAULT = {
    "LISTENING": ("friendly", "small_nod", 0.3),
    "THINKING": ("thinking", "none", 0.3),
    "IDLE": ("friendly", "none", 0.2),
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
    if gesture == "smile" and expression in ("neutral", "soft_smile"):
        expression = "smile"                 # the LLM asked for a smile: show it (it has no head motion)
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


# Warmth (owner: "make the avatar warm and sometimes smile according to the case"). The default face is
# the soft smile; a real smile when the moment is happy - read from the user's words (while the avatar
# thinks) and from the reply's first words - and never next to sad or worrying content.
_HAPPY_REPLY = re.compile(
    r"\b(congratulations|congrats|that'?s (so |really |just )?(great|wonderful|amazing|awesome|fantastic|lovely|"
    r"exciting|brilliant|great news|good news)|how (wonderful|exciting|lovely|fun|nice)|what a (great|lovely|"
    r"wonderful|fun|nice)|sounds (like )?(a |so |really )?(great|fun|wonderful|amazing|lovely|exciting)|haha|hehe|"
    r"ha ha|yay|hooray|"
    r"i'?m (so |really |very )?(glad|happy|delighted)|you'?re (very |so |most )?welcome|my pleasure|thank you|"
    r"thanks|nice to (meet|see|hear)|good (morning|afternoon|evening))\b|^(hello|hi|hey)\b")
_HAPPY_USER = re.compile(
    r"\b(thank(s| you)|hello|hi|hey|good (morning|afternoon|evening)|haha|hehe|lol|great news|good news|"
    r"i (just |finally )?got (the|a|my|an)|i (just |finally )?(passed|won|did it)|"
    r"i'?m (so |really )?(happy|excited|glad|thrilled)|(it'?s|today is) my birthday|congratulat\w*|"
    r"you'?re (great|awesome|amazing|the best|funny|sweet|kind)|nice to meet|jokes?|funny)\b")
# (whole words: "pain\w*" matched "painting", "cry\w*" "crypto", "scar\w*" "scarf")
_SAD = re.compile(
    r"\b(sad|sorry|die[ds]?|dead|death|passed away|sick|ill|hurt(s|ing|ful)?|pain(s|ful|fully)?|lost|los(e|ing)|"
    r"fail\w*|worr\w*|anxi\w*|scar(e|ed|es|y|ier|iest|ing)|afraid|angry|upset|terribl\w*|awful|bad news|"
    r"cr(y|ies|ied|ying)|depress\w*|lonely|accident|hospital|stress\w*|unfortunately|broke up|fired|nervous|"
    r"frighten\w*|difficult|tough|hard time)\b")
_NO_SMILE = ("empathetic", "concerned", "serious")


def is_happy_text(text: str, reply: bool = True) -> bool:
    """Happy moment (greeting, thanks, good news, congratulations, laughter) and nothing sad in it."""
    t = text.lower().replace("’", "'")
    return bool((_HAPPY_REPLY if reply else _HAPPY_USER).search(t)) and not _SAD.search(t)


def for_user_speaking() -> Behavior:
    """Active listening (as Anam/HeyGen avatars do): while the user talks, an attentive face."""
    behavior = from_meta({"emotion": "interested", "gesture": "none", "intensity": 0.4})
    behavior.source = "state:USER_SPEAKING"
    return behavior


def for_user_pause() -> Behavior:
    """The user paused mid-utterance: a small nod, like "mm-hm" (the worker allows one every ~5 s)."""
    behavior = from_meta({"emotion": "interested", "gesture": "small_nod", "intensity": 0.4})
    behavior.source = "state:USER_PAUSE"
    return behavior


def clause_behavior(current: Behavior, clause: str) -> Behavior | None:
    """The face for a later sentence of a reply: concern for sad or worrying words, a smile for happy
    ones, or None to keep the face it has (an emotion stays over the next sentences instead of flashing
    for one). Owner: "only small & short changes in face emotion" - the face was chosen once per reply."""
    t = clause.lower().replace("’", "'")
    if _SAD.search(t):
        if current.expression in ("soft_concern", "concern"):
            return None
        return replace(current, emotion="empathetic", expression="soft_concern", gesture="none", source="clause")
    # no smile inside a sad or worried reply ("Thank you for sharing that with me" after a loss)
    if is_happy_text(clause) and current.expression != "smile" and current.emotion not in _NO_SMILE:
        return replace(current, emotion="happy", expression="smile", gesture="none", source="clause")
    return None


def smile_for_user_text(user_text: str) -> Behavior | None:
    """While the avatar thinks: smile at a greeting, thanks or good news instead of the thinking look."""
    if not is_happy_text(user_text, reply=False):
        return None
    behavior = from_meta({"emotion": "happy", "gesture": "none", "intensity": 0.4})
    behavior.source = "state:THINKING"
    return behavior


def smile_for_happy_user(behavior: Behavior, user_text: str) -> Behavior:
    """Keep smiling while answering a happy message (the avatar already smiled while thinking), unless
    the reply itself is serious, worried or sad."""
    if behavior.expression != "smile" and behavior.emotion not in _NO_SMILE and is_happy_text(user_text, reply=False):
        return replace(behavior, expression="smile")
    return behavior


def behavior_from_text(user_text: str, first_clause: str) -> Behavior:
    """The reply's behaviour from words alone, when the LLM writes no tag (llm.behavior_tag false):
    concern for sad or worrying content, a smile for happy moments, otherwise the warm default; the
    gesture from how the reply starts. Writing the tag cost ~180 ms before every reply's first word
    (bench_voice: first token 409 ms, first word 583 ms), and the 4B model's gestures were unreliable
    anyway (see _TEXT_GESTURES)."""
    user = user_text.lower().replace("’", "'")
    reply = first_clause.lower().replace("’", "'")
    # the reply's own first words lead: "I passed my test, but I'm worried about the highway" ->
    # "Congratulations!" is a smile (a worried word anywhere in the message made the whole reply look
    # worried); the user's words decide only when the reply starts neutrally. Later sentences can still
    # change the face (clause_behavior).
    if _SAD.search(reply):
        emotion = "empathetic"
    elif is_happy_text(first_clause):
        emotion = "happy"
    elif _SAD.search(user):
        emotion = "empathetic"
    elif is_happy_text(user_text, reply=False):
        emotion = "happy"
    else:
        emotion = "friendly"
    gesture = gesture_from_text(first_clause) or "none"
    behavior = from_meta({"emotion": emotion, "gesture": gesture, "intensity": 0.5 if gesture in ("nod", "shake") else 0.4})
    behavior.source = "text"
    return behavior


def refine_behavior(behavior: Behavior, first_clause: str) -> Behavior | None:
    """The behaviour adjusted to the reply's first clause - the gesture its stance implies, a smile for a
    happy start ("Congratulations!", "You're welcome!") - or None if nothing changes."""
    changes: dict = {}
    gesture = gesture_from_text(first_clause)
    if gesture is not None and gesture != behavior.gesture:
        changes["gesture"] = gesture
        if gesture in ("nod", "shake"):
            changes["intensity"] = round(max(behavior.intensity, 0.5), 2)
    if behavior.expression != "smile" and behavior.emotion not in _NO_SMILE and is_happy_text(first_clause):
        changes["expression"] = "smile"
    if not changes:
        return None
    return replace(behavior, **changes, source="llm+text")
