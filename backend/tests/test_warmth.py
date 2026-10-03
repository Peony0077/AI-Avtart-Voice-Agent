"""Warm persona: soft smile by default, a real smile in happy moments, never next to sad content."""
import pytest

from app.behavior.manager import for_state, from_meta, is_happy_text, refine_behavior, smile_for_user_text


def test_default_faces_are_warm():
    assert from_meta({"emotion": "neutral"}).expression == "soft_smile"     # ordinary replies
    assert for_state("IDLE").expression == "soft_smile"
    assert for_state("LISTENING").expression == "soft_smile"
    assert from_meta({"emotion": "serious"}).expression == "neutral_firm"   # serious keeps the plain face
    assert from_meta({"emotion": "empathetic"}).expression == "soft_concern"


def test_llm_smile_gesture_shows_a_smile():
    assert from_meta({"emotion": "friendly", "gesture": "smile"}).expression == "smile"
    assert from_meta({"emotion": "concerned", "gesture": "smile"}).expression == "concern"   # not at worries


@pytest.mark.parametrize("text", [
    "Congratulations, that's wonderful news!", "You're welcome!", "Thank you so much!", "Hello! Nice to meet you.",
    "Haha, that's a good one.", "That sounds like fun!", "I'm so glad to hear that!", "Good morning!",
])
def test_happy_reply_starts(text):
    assert is_happy_text(text)


@pytest.mark.parametrize("text", [
    "I'm sorry to hear that.", "I'm glad you're okay after the accident.", "The capital of Japan is Tokyo.",
    "No, it's not safe to look at the sun.", "That must be really stressful.", "Thanks for telling me, that sounds hard and sad.",
])
def test_no_smile_for_sad_or_plain_replies(text):
    assert not is_happy_text(text)


@pytest.mark.parametrize("text, smile", [
    ("Hi there!", True), ("Thank you, that helped a lot.", True), ("I got the job!", True),
    ("Today is my birthday", True), ("Tell me a joke", True), ("Good morning", True),
    ("My dog died yesterday.", False), ("Thanks, but I'm still worried about my exam.", False),
    ("What is the capital of France?", False), ("I failed my exam.", False), ("I passed away", False),
])
def test_smile_while_thinking_only_for_happy_user_words(text, smile):
    b = smile_for_user_text(text)
    assert (b is not None) == smile
    if b is not None:
        assert b.expression == "smile" and b.source == "state:THINKING"


def test_reply_refinement_smiles_but_never_over_empathy():
    friendly = from_meta({"emotion": "friendly", "gesture": "none", "intensity": 0.4})
    r = refine_behavior(friendly, "Congratulations, that's amazing!")
    assert r.expression == "smile" and r.emotion == "friendly"
    empathetic = from_meta({"emotion": "empathetic", "gesture": "small_nod", "intensity": 0.5})
    assert refine_behavior(empathetic, "I'm so glad you told me.") is None       # sad context: no smile
    yes = refine_behavior(friendly, "Yes, thank you for asking!")
    assert (yes.gesture, yes.expression) == ("nod", "smile")                      # both refinements at once


def test_smile_carries_into_the_reply_to_a_happy_message():
    from app.behavior.manager import smile_for_happy_user
    friendly = from_meta({"emotion": "friendly", "gesture": "none", "intensity": 0.4})
    assert smile_for_happy_user(friendly, "Tell me a short joke").expression == "smile"
    assert smile_for_happy_user(friendly, "I just got the job!").expression == "smile"
    assert smile_for_happy_user(friendly, "What is the capital of Japan?").expression == "soft_smile"
    serious = from_meta({"emotion": "serious", "gesture": "none", "intensity": 0.4})
    assert smile_for_happy_user(serious, "Thanks! Is it safe to drive tired?").expression == "neutral_firm"
