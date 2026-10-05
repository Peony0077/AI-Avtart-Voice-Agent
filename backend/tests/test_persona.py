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
    assert Persona(name="Ava").greeting_text().startswith("Hi there! I'm Ava")
    assert "from Acme Solar" in Persona(name="Ava", company="Acme Solar").greeting_text()


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


def test_forget_me_wins_over_an_update_that_was_already_running(tmp_path):
    """A session that ended just before "Forget me" merges its words in the background; saving that
    afterwards brought the erased memory back (seen in the e2e run)."""
    mem = MemoryStore(tmp_path / "memory.json")
    mem.save("The user is Sam.")
    began = mem.generation                                   # the update starts ...
    mem.clear()                                              # ... the user says "Forget me" ...
    assert mem.save("The user is Sam and likes hiking.", began) is False     # ... its result is dropped
    assert mem.load() is None
    assert mem.save("The user is Kim.", mem.generation) is True
    assert mem.load() == "The user is Kim."


def background_services(llm, store, monkeypatch):
    """A Services with only what the background LLM jobs use (no models)."""
    import app.services as services_mod
    from app.services import Services
    monkeypatch.setattr(services_mod, "BACKGROUND_QUIET_S", 0.05)
    svc = Services.__new__(Services)
    svc.llm, svc.memory_store, svc._background = llm, store, set()
    svc._llm_jobs, svc._llm_runner, svc.conversation_busy = [], None, lambda: False
    return svc


async def test_remember_drops_the_words_of_a_session_from_before_forget_me(tmp_path, monkeypatch):
    import asyncio

    class SlowLLM:
        async def complete(self, messages, max_tokens=0):
            await asyncio.sleep(0.05)
            return "The user is Sam."

    svc = background_services(SlowLLM(), MemoryStore(tmp_path / "memory.json"), monkeypatch)
    began = svc.memory_store.generation                      # a tab opened before the erase
    svc.memory_store.clear()
    svc.remember("- I'm Sam.", began)
    svc.remember("- I'm Sam.")                               # a session that ends now ...
    svc.memory_store.clear()                                 # ... and "Forget me" while it is merged
    await asyncio.gather(*svc._background)
    assert svc.memory_store.load() is None
    svc.remember("- I'm Kim.", svc.memory_store.generation)
    await asyncio.gather(*svc._background)
    assert svc.memory_store.load() == "The user is Sam."     # (the fake LLM's answer) saved this time


def test_the_whole_session_is_remembered_but_not_what_came_before_forget_me():
    mem = ConversationMemory(persona="Ava", max_turns=4)
    mem.add("(The user started the conversation.)", "Hi!")
    mem.add("I'm Sam.", "Hello Sam!")
    for i in range(6):                                       # older turns leave the LLM's window ...
        mem.add(f"Question {i}?", "Answer.")
    mem.apply_summary("Sam asked questions.")                # ... and are summarised
    assert "- I'm Sam." in mem.user_said() and "- Question 5?" in mem.user_said()
    mem.forget_user()
    assert mem.user_said() == ""
    mem.add("I live in Osaka.", "Nice!")
    assert mem.user_said() == "- I live in Osaka."


def test_business_settings_are_validated_and_reach_the_prompt():
    from app.llm.prompt import lead_rules
    p = PersonaStore.merge(Persona(name="Ava"), {"company": " Acme Solar ", "offer": "Home solar panels. " * 100,
                                                 "next_step": "book a free roof check", "lead_gen": False}, voices=None)
    assert p.company == "Acme Solar" and len(p.offer) == 800 and p.lead_gen is False
    prompt = system_prompt("Ava", company="Acme Solar", lead=lead_rules("Acme Solar", "Home solar panels.", "book a free roof check"))
    assert "assistant for Acme Solar" in prompt and "book a free roof check" in prompt and "Home solar panels." in prompt
    assert "United States or Canada" in prompt and "American spelling" in prompt
    assert "a short call with our team" in lead_rules()                  # a next step even when none is set
    assert "Your goal" not in system_prompt("Ava")                        # lead generation off: no sales rules


def test_each_customer_has_their_own_memory(tmp_path):
    from app.persona import VisitorMemories
    v = VisitorMemories(tmp_path / "visitors")
    a, b = v.store("visitor-aaaa-1111"), v.store("visitor-bbbb-2222")
    a.save("The user is Sam.")
    assert b.load() is None and a.load() == "The user is Sam."
    assert v.store("visitor-aaaa-1111") is a                                 # the same store (its Forget-me counter)
    assert v.store("../../etc/passwd") is None and v.store("") is None      # only plain ids


def test_time_questions_get_the_customers_own_time_the_american_way():
    from app.llm.prompt import time_hint
    now = 1791158400.0                                        # 2026-10-05 00:00 UTC (a Monday)
    hint = time_hint("What time is it?", now=now, utc_offset_min=-4 * 60)   # New York (EDT)
    assert "Sunday, October 4, 2026, 8:00 PM" in hint and "where the customer is" in hint
    assert time_hint("Tell me a joke.", now=now, utc_offset_min=0) == ""


async def test_background_llm_jobs_give_way_to_a_conversation(tmp_path, monkeypatch):
    """Memory and lead extraction wait while a conversation uses the LLM, stop when one starts, and
    run again later (the next reply must never wait behind them)."""
    import asyncio

    calls = {"started": 0, "finished": 0}

    class LLM:
        async def complete(self, messages, max_tokens=0):
            calls["started"] += 1
            await asyncio.sleep(0.3)
            calls["finished"] += 1
            return "The user is Sam."

    store = MemoryStore(tmp_path / "memory.json")
    svc = background_services(LLM(), store, monkeypatch)
    busy = {"now": True}
    svc.conversation_busy = lambda: busy["now"]
    svc.remember("- I'm Sam.")
    await asyncio.sleep(0.3)
    assert calls["started"] == 0                              # a conversation is on: it waits
    busy["now"] = False
    await asyncio.sleep(0.2)
    assert calls["started"] == 1                              # quiet: it starts ...
    busy["now"] = True
    await asyncio.sleep(0.2)
    assert calls["finished"] == 0                             # ... and stops when a conversation needs the LLM
    busy["now"] = False
    await asyncio.gather(*svc._background)
    assert calls["started"] == 2 and calls["finished"] == 1 and store.load() == "The user is Sam."
