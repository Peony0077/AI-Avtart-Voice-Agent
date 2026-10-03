"""BehaviorManager: the reply's first words refine the head gesture the LLM's tag suggested."""
import pytest

from app.behavior.manager import from_meta, gesture_from_text, refine_gesture


@pytest.mark.parametrize("text, gesture", [
    ("Yes, absolutely!", "nod"),
    ("Exactly, that's how it works.", "nod"),
    ("Of course you can.", "nod"),
    ("Sure!", "nod"),
    ("That's right.", "nod"),
    ("No, people cannot breathe underwater.", "shake"),
    ("Not quite, actually the Earth rotates.", "shake"),
    ("Well, no.", "shake"),
    ("Hmm, not really.", "shake"),
    ("I'm afraid not.", "shake"),
    ("I'm really sorry you're going through this.", "small_nod"),
    ("I’m so sorry.", "small_nod"),
    ("Oh really?", "head_tilt"),
    ("What happened?", "head_tilt"),
])
def test_stance_from_first_words(text, gesture):
    assert gesture_from_text(text) == gesture


@pytest.mark.parametrize("text", [
    "No problem!", "No worries, take your time.", "Nobody knows for sure.", "Right now it's late.",
    "The capital of Japan is Tokyo.", "Actually, from our perspective it looks that way.", "Oh wow, that's great!",
    "Notably, it rained.",
])
def test_no_gesture_without_a_clear_cue(text):
    assert gesture_from_text(text) is None


def test_refine_keeps_the_tag_unless_the_words_say_otherwise():
    llm = from_meta({"emotion": "serious", "gesture": "none", "intensity": 0.3})
    refined = refine_gesture(llm, "No, it's not safe to look at the sun.")
    assert refined.gesture == "shake" and refined.intensity == 0.5 and refined.source == "llm+text"
    assert (refined.emotion, refined.expression) == (llm.emotion, llm.expression)     # only the gesture changes
    assert refine_gesture(llm, "The capital of Japan is Tokyo.") is None               # no cue: keep the tag
    nodding = from_meta({"emotion": "friendly", "gesture": "nod", "intensity": 0.6})
    assert refine_gesture(nodding, "Yes, definitely.") is None                         # already right
