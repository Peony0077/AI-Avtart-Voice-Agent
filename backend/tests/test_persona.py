"""Persona settings and the memory of the user across sessions (app/persona.py, llm/prompt.py)."""
import json

from app.llm.prompt import ConversationMemory, system_prompt, time_hint
from app.persona import MAX_MEMORY_WORDS, MAX_PERSONALITY, MemoryStore, Persona, PersonaStore, memory_request


def test_persona_settings_are_validated_and_stored(tmp_path):
    store = PersonaStore(tmp_path / "persona.json", Persona(name="Ava", voice="af_heart"))
    assert store.load().name == "Ava"                                      # defaults without a file
    p = PersonaStore.merge(store.load(), {"name": "  Mia\x07 ", "personality": "x" * 5000, "voice": "bf_emma",
                                          "memory": False, "unknown": 1}, voices=["af_heart", "bf_emma"])
    assert (p.name, len(p.personality), p.voice, p.memory) == ("Mia", MAX_PERSONALITY, "bf_emma", False)
    store.save(p)
    assert store.load() == p
    q = PersonaStore.merge(p, {"name": "", "voice": "not_a_voice"}, voices=["af_heart"])
    assert (q.name, q.voice) == ("Mia", "bf_emma")                        # empty name / unknown voice ignored
    assert Persona(name="Ava").greeting_text().startswith("Hi, I'm Ava")


def test_memory_is_kept_short_and_can_be_forgotten(tmp_path):
    mem = MemoryStore(tmp_path / "memory.json")
    assert mem.load() is None
    mem.save("The user is Sam. " * 100)
    assert len(mem.load().split()) <= MAX_MEMORY_WORDS
    assert "updated" in json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))
    mem.clear()
    assert mem.load() is None and not (tmp_path / "memory.json").exists()
    req = memory_request("The user is Sam.", "- I moved to Osaka.")
    assert "The user is Sam." in req[1]["content"] and "Osaka" in req[1]["content"]
    assert "never guess their gender" in req[0]["content"]


def test_personality_and_memory_reach_the_llm():
    prompt = system_prompt("Mia", personality="Cheerful, loves cooking.", long_term="The user is Sam from Osaka.")
    assert "You are Mia" in prompt and "loves cooking" in prompt and "Sam from Osaka" in prompt
    mem = ConversationMemory(persona="Mia", personality="Calm.", long_term="Sam likes tea.")
    assert "Sam likes tea." in mem.build("Hi")[0]["content"]
    mem.add("(The user started the conversation.)", "Hi!")
    mem.add("I'm Sam.", "Nice to meet you, Sam. I love cooking.")
    assert mem.user_turns == 2 and "I'm Sam." in mem.transcript()
    assert mem.user_said() == "- I'm Sam."                              # only the user's own words


def test_the_time_is_given_only_when_the_question_is_about_time():
    assert "local time" in time_hint("What time is it?") and "local time" in time_hint("What day is it today?")
    assert time_hint("What is the capital of France?") == ""
    hint = time_hint("What's the date?", now=1791100000)                 # no zero-padded day ("04")
    assert ", 0" not in hint
