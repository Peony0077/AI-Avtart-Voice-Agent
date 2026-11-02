"""The avatar's side of HPN mode: what she says (app/hpn/voice.py), the re-matching of unclear questions
(app/hpn/match.py), the avatar-only wording (knowledge/hpn/avatar_overrides.json) and a whole turn through
the session: HPN's answer word for word, no LLM text, no memory, no leads, private details out of the log."""
import asyncio
import re
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.behavior.speech_text import normalize_for_speech
from app.core.config import HPNCfg, Settings
from app.hpn import match
from app.hpn.engine import Engine
from app.hpn.knowledge import HOUSTON
from app.hpn.voice import card_sentence, spoken
from app.persona import Persona
from app.session.session import Session
from app.session.state_machine import Event
from tests.hpn_fixture import THURSDAY_8PM, HelperTestCase, real_knowledge

TUESDAY_730PM = datetime(2026, 11, 3, 19, 30, tzinfo=HOUSTON)


# --- what she says ----------------------------------------------------------------------------------

class SpokenTests(HelperTestCase):
    def test_prices_are_said_where_the_website_shows_its_cards(self):
        reply = self.ask("How much is a wedding?")
        said = spoken(reply)
        for card in reply["cards"]:
            name = re.sub(r"\s*\(.*\)$", "", card["title"])
            self.assertIn(name, said)
            self.assertIn(card["subtitle"][1:], said)            # "tarting at $1,495": the price as written
        self.assertIn("Signature Wedding, most popular: starting at $1,495.", said)
        # the cards come after the first paragraph, the rest of the text after them
        self.assertLess(said.index("Here are our three wedding packages"), said.index("Reception Essentials:"))
        self.assertLess(said.index("Reception Essentials:"), said.index("Craig plans each wedding"))
        self.assertNotIn("**", said)
        self.assertNotIn("http", said)

    def test_tonight_is_said_with_the_venue_kind_time_and_town(self):
        said = spoken(self.ask("what's on tonight?"))
        self.assertIn("Molly's Pub Tomball: Karaoke at 9 PM, Tomball, tonight.", said)

    def test_a_date_on_a_card_is_said_in_full(self):
        card = {"title": "Mongo's", "subtitle": "Jukebox Bingo · 7 PM", "detail": "Montgomery · Wed 4 Nov",
                "path": "/night-out/"}
        self.assertEqual(card_sentence(card), "Mongo's: Jukebox Bingo at 7 PM, Montgomery, Wednesday, November 4.")

    def test_a_package_by_name_is_not_named_twice(self):
        said = spoken(self.ask("What does Signature Wedding include?"))
        self.assertEqual(said.count("Signature Wedding"), 1, said)
        self.assertIn("Starting at $1,495.", said)

    def test_an_hourly_package_says_what_each_extra_hour_costs(self):
        said = spoken(self.ask("party prices"))
        self.assertIn("Party Essentials: starting at $549 for 3 hours, then $175 for each extra hour.", said)

    def test_a_long_list_says_the_rest_is_on_the_screen(self):
        said = spoken(self.ask("what extras do you have?"))
        self.assertIn("The rest are on the screen.", said)

    def test_every_answer_has_something_to_say(self):
        for question in ("hi", "who are you", "deposit?", "llamas", "", "what's on tonight", "prices"):
            self.assertTrue(spoken(self.ask(question)).strip(), question)


def test_a_phone_number_is_said_digit_by_digit():
    assert normalize_for_speech("Call (832) 295-0852 now.") == "Call 8 3 2, 2 9 5, 0 8 5 2 now."
    assert normalize_for_speech("or 832-295-0852") == "or 8 3 2, 2 9 5, 0 8 5 2"
    # prices, ranges and dates are not phone numbers
    assert normalize_for_speech("Starting at $1,495 for 2-3 hours") == "Starting at $1,495 for 2 to 3 hours"
    assert normalize_for_speech("on 6/14/2027") == "on 6/14/2027"


# --- the avatar-only wording ------------------------------------------------------------------------

class FollowUpTests(HelperTestCase):
    def test_every_suggested_question_after_a_price_answer_answers_itself(self):
        # HPN's helper (and so h-townparty.com) answered the wedding prices' own suggestion "What does Signature
        # Wedding include?" with the wedding prices again; the avatar answers the package
        reply = self.ask("What does Signature Wedding include?", last="prices.wedding")
        self.assertEqual(reply["intent"], "prices.package")
        self.assertIn("Everything in Reception Essentials, plus:", reply["text"])
        for question in ("How much is a wedding?", "party prices", "foam party price"):
            first = self.ask(question)
            for chip in first["chips"]:
                again = self.ask(chip, last=first["intent"])
                self.assertFalse(again["intent"] == first["intent"] and again["text"] == first["text"],
                                 f"{question} -> {chip}: the same answer again")
        # the short follow-ups still work
        self.assertEqual(self.intent("what about foam?", last="prices.party"), "prices.foam")
        self.assertEqual(self.intent("and weddings?", last="prices.foam"), "prices.wedding")


class OverrideTests(HelperTestCase):
    def test_the_avatar_never_claims_to_use_no_ai(self):
        for question in ("What happens to what I type here?", "Do you use AI?", "What happened to the AI chat?",
                         "who are you", "are you a bot"):
            text = self.ask(question)["text"]
            self.assertNotRegex(text, r"(?i)uses no AI|not AI\b|and not AI", question)

    def test_the_rewritten_topics_keep_hpns_facts(self):
        ai = self.ask("Do you use AI?")["text"]
        self.assertIn("**AI check of your plan**", ai)
        self.assertIn("Claude, made by Anthropic", ai)
        concierge = self.ask("What happened to the AI chat?")["text"]
        self.assertIn("Until 29 October 2026", concierge)
        self.assertIn("(832) 295-0852", concierge)
        privacy = self.ask("Is this chat private?")["text"]
        self.assertIn("card numbers, phone numbers and email addresses are taken out", privacy)

    def test_only_the_listed_topics_are_rewritten(self):
        rewritten = {t.slug for t in self.kb.topics if t.overridden}
        self.assertEqual(rewritten, {"helper-privacy", "concierge-replaced", "ai-check"})


# --- unclear questions --------------------------------------------------------------------------------

def test_every_candidate_question_reaches_its_own_answer_with_the_live_knowledge():
    engine = Engine(real_knowledge(), "Emma")
    titles = {t.title: t.slug for t in engine.kb.topics}
    for c in match.candidates(engine):
        reply = engine.ask(c.question, now=TUESDAY_730PM)
        assert reply.confident and reply.kind in ("answer", "live", "smalltalk"), c
        if c.label in titles:
            assert reply.intent == f"topic:{titles[c.label]}", c


def test_the_llm_reply_is_read_as_a_number_in_range_or_none():
    assert match.parse("12", 50) == 12
    assert match.parse(" 7.", 50) == 7
    assert match.parse("Number: 3", 50) == 3
    assert match.parse("0", 50) == 0
    assert match.parse("99", 50) == 0
    assert match.parse("none", 50) == 0


def test_the_prompt_lists_the_questions_and_the_customer_words():
    engine = Engine(real_knowledge(), "Emma")
    cands = match.candidates(engine)
    msgs = match.prompt(cands, "how much for a phone party", "Foam parties")
    assert msgs[0]["role"] == "system" and f"{len(cands)}. " in msgs[0]["content"]
    assert "how much for a phone party" in msgs[1]["content"] and "Foam parties" in msgs[1]["content"]


# --- a whole turn through the session -----------------------------------------------------------------

class FakeLLM:
    def __init__(self, answer="0"):
        self.answer, self.calls = answer, []

    async def complete(self, messages, max_tokens=16, temperature=None):
        self.calls.append((messages, temperature))
        return self.answer


class Recorder:
    def __init__(self):
        self.sent = []

    def json(self, msg, *args, **kwargs):
        self.sent.append(msg)

    def of(self, kind):
        return [m for m in self.sent if m.get("type") == kind]


def hpn_session(tmp_path, llm=None) -> tuple[Session, SimpleNamespace]:
    from tests.hpn_fixture import fixture_knowledge

    settings = Settings(hpn=HPNCfg(enabled=True), log_dir=tmp_path)
    jobs = []
    services = SimpleNamespace(
        settings=settings, persona=Persona(name="Emma"), hpn_kb=fixture_knowledge(), llm=llm or FakeLLM(),
        memory_store=None, visitor_memories=SimpleNamespace(store=lambda v: None), stt=None, avatar_ready=False,
        voice_ready=False, avatar=None, avatar_reset=lambda *a, **k: None,
        remember=lambda *a: jobs.append("memory"), capture_lead=lambda *a: jobs.append("lead"))
    s = Session(SimpleNamespace(), services, role="customer", visitor_id="v1")
    s.outbox = Recorder()
    s.outbox.start = lambda: None

    async def stop():
        pass
    s.outbox.stop = stop
    s._emit_metrics = lambda *a, **k: None
    s.sm.fire(Event.START)
    return s, SimpleNamespace(jobs=jobs, services=services)


async def run_turn(s: Session, text: str) -> dict:
    turn = await s.start_turn("text", text)
    await asyncio.gather(*turn.tasks)
    done = s.outbox.of("assistant_done")[-1]
    return {"helper": s.outbox.of("helper_reply")[-1], "said": done["text"]}


async def test_a_question_gets_hpns_answer_word_for_word_and_no_llm_text(tmp_path):
    llm = FakeLLM("1")
    s, _ = hpn_session(tmp_path, llm)
    got = await run_turn(s, "Do you do weddings?")
    reply = got["helper"]["reply"]
    assert reply["intent"] == "topic:weddings" and got["helper"]["matched"] is None
    topic = s.hpn.kb.topic("weddings")
    assert reply["text"] == s.hpn.kb.fill(topic.answer_template)
    assert got["said"] == spoken(reply).replace("\n\n", " ") or got["said"].startswith("Yes, and weddings")
    assert "Craig plans each wedding personally with the couple" in got["said"]
    assert llm.calls == []                          # a sure answer: the LLM is not asked anything
    assert [link["url"] for link in reply["links"]][0].startswith("https://h-townparty.com/")


async def test_the_session_tells_the_page_it_is_hpn(tmp_path):
    s, _ = hpn_session(tmp_path)
    info = s._hpn_session()
    assert info["site"] == "https://h-townparty.com" and info["phone_display"] == "(832) 295-0852"
    assert "Plan an event" in info["chips"]


async def test_an_unclear_question_is_re_matched_and_says_which_question_it_answers(tmp_path):
    s, _ = hpn_session(tmp_path)
    cands = match.candidates(s.hpn)
    foam = next(i for i, c in enumerate(cands, 1) if c.label == "Foam party prices")
    s.svc.llm = FakeLLM(str(foam))
    got = await run_turn(s, "what would the soap suds shindig set me back")   # the helper: not sure
    assert got["helper"]["matched"] == "Foam party prices"
    assert got["helper"]["reply"]["intent"] == "prices.foam"
    assert s.svc.llm.calls[0][1] == 0.0             # temperature 0: the same pick every time


async def test_when_the_llm_finds_nothing_the_helpers_own_not_sure_answer_stands(tmp_path):
    s, _ = hpn_session(tmp_path, FakeLLM("0"))
    got = await run_turn(s, "do you rent llamas for petting zoos")
    assert got["helper"]["reply"]["intent"] == "fallback" and got["helper"]["matched"] is None
    assert "I'm not sure about that one yet" in got["said"]


async def test_a_broken_llm_never_breaks_the_answer(tmp_path):
    class Broken(FakeLLM):
        async def complete(self, *a, **k):
            raise RuntimeError("llama-server down")
    s, _ = hpn_session(tmp_path, Broken())
    got = await run_turn(s, "do you rent llamas for petting zoos")
    assert got["helper"]["reply"]["intent"] == "fallback"


async def test_follow_ups_keep_the_subject(tmp_path):
    s, _ = hpn_session(tmp_path)
    first = await run_turn(s, "what's on tonight?")
    assert first["helper"]["reply"]["intent"].startswith("nights.")
    follow = await run_turn(s, "and tomorrow?")
    assert follow["helper"]["reply"]["intent"] == "nights.tomorrow"


async def test_greeting_is_the_helpers_opening_with_her_name(tmp_path):
    s, _ = hpn_session(tmp_path)
    s.voice_out = True                               # greet() only greets a spoken conversation
    started = []

    async def start_turn(source, text, **kw):
        started.append((source, kw))
    s.start_turn = start_turn
    await s.greet()
    source, kw = started[0]
    assert source == "greeting"
    assert "I'm Emma, Houston Party Network's AI Agent" in kw["script"]
    assert kw["helper"]["reply"]["intent"] == "greeting"


async def test_no_memory_no_leads_and_private_details_kept_out_of_the_log(tmp_path):
    s, env = hpn_session(tmp_path)
    await run_turn(s, "my email is jo@example.com and my number is (832) 555-0101, how much is a wedding")
    assert s._loggable("call me on (832) 555-0101 or jo@example.com") == \
        "call me on [phone removed] or [email removed]"
    assert s.memory.long_term is None
    await s.close("test")
    assert env.jobs == []                            # no memory update, no lead capture in HPN mode


@pytest.mark.parametrize("text", ["my card is 4111 1111 1111 1111", "4111-1111-1111-1111 please charge it"])
async def test_a_card_number_is_never_said_back(tmp_path, text):
    s, _ = hpn_session(tmp_path)
    got = await run_turn(s, text)
    assert got["helper"]["reply"]["intent"] == "smalltalk.card"
    assert "4111" not in got["said"]


def test_fixture_time_is_a_thursday_evening():
    assert THURSDAY_8PM.strftime("%A %H:%M") == "Thursday 20:00"


# --- contact questions (3 Nov 2026) -------------------------------------------------------------------

class ContactTests(HelperTestCase):
    """Contact details are where a wrong answer costs a customer: each case gives exactly the right ones."""

    def links(self, reply):
        return [link["path"] for link in reply["links"]]

    def test_every_way_of_asking_for_hpns_contact_gets_the_phone_email_and_form(self):
        for q in ("What's your phone number?", "How can I contact you?", "What is your email address?", "phone",
                  "email", "contact", "Can I book by phone?", "Is 832-295-0852 your number?"):
            reply = self.ask(q)
            self.assertEqual(reply["intent"], "contact", q)
            self.assertIn("(832) 295-0852", reply["text"], q)
            self.assertIn("Info@H-TownParty.com", reply["text"], q)
            self.assertEqual(self.links(reply), ["tel:+18322950852", "mailto:Info@H-TownParty.com", "/contact-us/"], q)

    def test_someones_own_number_is_never_given_only_the_business_line(self):
        for q in ("What's Craig's number?", "Can I have the owner's cell phone?", "give me craig's personal email",
                  "what is the manager's phone number", "can I text the DJ"):
            reply = self.ask(q)
            self.assertEqual(reply["intent"], "smalltalk.person_contact", q)
            self.assertIn("I can only give out Houston Party Network's own contact details", reply["text"])
            numbers = re.findall(r"\(\d{3}\) \d{3}-\d{4}", reply["text"])
            self.assertEqual(numbers, ["(832) 295-0852"], q)
        self.assertEqual(self.intent("Can I talk to Craig?"), "smalltalk.human")
        self.assertEqual(self.intent("Who is Craig?"), self.intent("Who started Houston Party Network?"))

    def test_a_venues_phone_and_address_are_the_venues_own(self):
        from app.hpn.knowledge import Venue
        venues = [Venue(name=v.name, city=v.city, street="15945 Kuykendahl Rd", postcode="77068", region="TX",
                        phone="+12815806167") if v.name == "Tony's Tavern" else v for v in self.kb.venues]
        slots = [s for s in self.kb.slots]
        for s in slots:
            s.venue = next(v for v in venues if v.name == s.venue.name)
        self.rebuild(venues=venues, slots=slots)
        for q in ("What's the phone number for Tony's Tavern?", "Where is Tony's Tavern?", "address of tonys tavern"):
            reply = self.ask(q)
            self.assertEqual(reply["intent"], "venue.contact", q)
            self.assertIn("15945 Kuykendahl Rd, Houston, TX 77068", reply["text"])
            self.assertIn("Phone (281) 580-6167", reply["text"])
            self.assertIn("That is the venue's own number. For Houston Party Network, call (832) 295-0852", reply["text"])
            self.assertEqual(self.links(reply)[0], "tel:+12815806167")
        said = spoken(self.ask("Where is Tony's Tavern?"))
        self.assertIn("15945 Kuykendahl Road, Houston.", said)
        self.assertNotIn("TX", said)
        self.assertNotIn("77068", said)
        # nights at a venue are still the nights, not its phone
        self.assertTrue(self.intent("Is there karaoke at Tony's Tavern this weekend?").startswith("nights."))

    def test_a_customers_own_details_are_never_kept_or_repeated_and_the_form_is_offered(self):
        for q in ("My email is jo@example.com", "my number is 832 555 0101", "call me on (713) 555-0199"):
            reply = self.ask(q)
            self.assertEqual(reply["intent"], "smalltalk.callback", q)
            self.assertNotIn("jo@example.com", reply["text"])
            self.assertNotIn("555", reply["text"])
            self.assertIn("a number or email left with me won't reach the team", reply["text"])
            self.assertEqual(self.links(reply)[0], "/contact-us/")
        reply = self.ask("How much is a wedding? my number is 832 555 0101")
        self.assertEqual(reply["intent"], "prices.wedding")          # the question is still answered
        self.assertTrue(reply["text"].startswith("Please don't leave your phone number or email with me"))
        self.assertNotIn("555", reply["text"])
        self.assertEqual(self.links(reply)[0], "/contact-us/")

    def test_when_the_team_answers_is_not_made_up_nor_the_nights_out(self):
        for q in ("Do you answer on weekends?", "What are your business hours?", "How fast do you reply?",
                  "When are you open?"):
            reply = self.ask(q)
            self.assertFalse(reply["intent"].startswith("nights."), q)
            self.assertIn("(832) 295-0852", reply["text"], q)
