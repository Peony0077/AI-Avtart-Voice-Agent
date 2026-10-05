"""Emotions that change within a reply and active listening (owner, 2026-10-05: "only small & short
changes in face emotion"; app/behavior/manager.py)."""
from app.behavior.manager import behavior_from_text, clause_behavior, for_user_pause, for_user_speaking


def test_a_later_sentence_can_change_the_face_and_the_change_stays():
    b = behavior_from_text("Tell me about your plans.", "Sure, here is how it works.")
    assert b.expression == "soft_smile"
    happy = clause_behavior(b, "That's great news, congratulations!")
    assert happy is not None and happy.expression == "smile" and happy.source == "clause"
    assert clause_behavior(happy, "We start on Monday.") is None          # a neutral sentence keeps the smile
    sad = clause_behavior(happy, "Unfortunately the office is closed that week.")
    assert sad is not None and sad.expression == "soft_concern"
    assert clause_behavior(sad, "I'm sorry about that.") is None           # already concerned
    assert clause_behavior(b, "Here are the three steps.") is None          # nothing to change


def test_listening_gets_an_attentive_face_and_nods_at_pauses():
    talking, pause = for_user_speaking(), for_user_pause()
    assert talking.expression == "interested" and talking.gesture == "none"
    assert pause.expression == "interested" and pause.gesture == "small_nod"
    assert talking.source.startswith("state:") and pause.source.startswith("state:")   # not a reply's face


def test_no_smile_inside_a_sad_reply():
    b = behavior_from_text("My grandmother passed away last week.", "I'm so sorry for your loss.")
    assert b.expression == "soft_concern"
    assert clause_behavior(b, "Thank you for sharing that with me.") is None


def test_the_sales_rules_never_ask_for_placeholders():
    from app.llm.prompt import lead_rules
    assert "Never write a placeholder" in lead_rules()
