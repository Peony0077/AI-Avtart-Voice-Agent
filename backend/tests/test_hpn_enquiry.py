"""A customer's details sent to HPN as an enquiry (app/hpn/enquiry.py): taken down, read back, sent only on a
yes, through HPN's own endpoint. HPN's endpoint is replaced by a fake here: no test ever creates a real lead."""
import asyncio

import pytest

from app.hpn import enquiry as E
from tests.test_hpn_avatar import hpn_session

# --- reading the details ----------------------------------------------------------------------------

def test_a_spoken_email_is_put_back_together():
    assert E.find_email("my email is jo at example dot com") == "jo@example.com"
    assert E.find_email("it's Jo.Smith at gmail.com please") == "jo.smith@gmail.com"
    assert E.find_email("JO@Example.COM.") == "jo@example.com"
    assert E.find_email("I'm at the venue") is None


def test_a_phone_number_is_taken_but_never_a_card_number():
    assert E.find_phone("call me on (713) 555-0199") == "(713) 555-0199"
    assert E.find_phone("832.555.0101") == "(832) 555-0101"
    assert E.find_phone("my card is 4111 1111 1111 1111") is None


@pytest.mark.parametrize("text, name", [
    ("my name is Jo Smith", "Jo Smith"), ("This is Maria and my email is m@x.com", "Maria"),
    ("I'm planning a wedding", None), ("I am looking for a DJ", None), ("Hi, I'm Dave", "Dave"),
])
def test_names(text, name):
    assert E.find_name(text) == name


def test_after_the_question_a_short_answer_is_the_name():
    assert E.find_name("Jo Smith", asked=True) == "Jo Smith"
    assert E.find_name("ok", asked=True) is None
    assert E.find_name("what packages do you have for a big wedding", asked=True) is None


def test_the_event_is_the_customers_own_words_first():
    assert E.event_type_from(["prices.party"], ["we are getting married in June"]) == "wedding"
    assert E.event_type_from(["prices.foam"], ["how much?"]) == "foam"
    assert E.event_type_from(["topic:birthdays"], []) == "party"
    assert E.event_type_from([], ["hello"]) == ""


# --- the whole conversation -------------------------------------------------------------------------

class FakeHPN:
    def __init__(self, status=200, body=None):
        self.status, self.body, self.posts = status, body if body is not None else {"ok": True, "lead_id": 42}, []

    async def __call__(self, payload, session_key, timeout=15.0):
        self.posts.append((payload, session_key))
        return self.status, self.body


async def say(s, text):
    turn = await s.start_turn("text", text)
    await asyncio.gather(*turn.tasks)
    helper = s.outbox.of("helper_reply")[-1]
    return helper["reply"]


@pytest.fixture
def live(tmp_path, monkeypatch):
    hpn = FakeHPN()
    monkeypatch.setattr(E, "post", hpn)
    s, _ = hpn_session(tmp_path)
    s.enquiry.mode = "live"
    return s, hpn


async def test_details_are_read_back_and_sent_only_after_yes(live):
    s, hpn = live
    await say(s, "How much is a wedding?")
    r = await say(s, "Please have someone call me, my email is jo at example dot com")
    assert r["intent"] == "enquiry.ask_name" and "what name" in r["text"].lower()
    r = await say(s, "Jo Smith")
    assert r["intent"] == "enquiry.confirm"
    assert "Name: **Jo Smith**" in r["text"] and "Email: **jo@example.com**" in r["text"]
    assert "About: **a wedding**" in r["text"]
    assert hpn.posts == []                                      # nothing sent before the yes
    r = await say(s, "Yes, send it")
    assert r["intent"] == "enquiry.sent" and "with the Houston Party Network team" in r["text"]
    (payload, key), = hpn.posts
    assert payload["name"] == "Jo Smith" and payload["email"] == "jo@example.com"
    assert payload["event_type"] == "wedding" and payload["source"] == "AI avatar"
    assert "How much is a wedding?" in payload["message"] and key == s.id
    # sent once: asking again does not send it twice
    r = await say(s, "call me back please")
    assert r["intent"] == "enquiry.already_sent" and len(hpn.posts) == 1


async def test_the_phone_number_goes_too_and_details_can_be_changed(live):
    s, hpn = live
    r = await say(s, "My name is Maria, call me on (713) 555-0199")
    assert r["intent"] == "enquiry.ask_email"
    r = await say(s, "maria@example.com")
    assert "Phone: **(713) 555-0199**" in r["text"]
    r = await say(s, "No, change it")
    assert r["intent"] == "enquiry.change"
    r = await say(s, "the email is maria.g@example.com")
    assert r["intent"] == "enquiry.confirm" and "maria.g@example.com" in r["text"]
    await say(s, "yes")
    assert hpn.posts[0][0]["email"] == "maria.g@example.com" and hpn.posts[0][0]["phone"] == "(713) 555-0199"


async def test_cancel_and_another_question_send_nothing(live):
    s, hpn = live
    await say(s, "my email is jo@example.com")
    r = await say(s, "cancel")
    assert r["intent"] == "enquiry.cancelled" and "haven't sent anything" in r["text"]
    await say(s, "my email is jo@example.com")
    r = await say(s, "What's on tonight?")
    assert r["intent"].startswith("nights.") and r["text"].startswith("I haven't sent your details.")
    assert hpn.posts == []


async def test_hpn_refusing_the_email_asks_for_it_again(tmp_path, monkeypatch):
    hpn = FakeHPN(400, {"error": "We need a name and an email address to reply to."})
    monkeypatch.setattr(E, "post", hpn)
    s, _ = hpn_session(tmp_path)
    s.enquiry.mode = "live"
    await say(s, "My name is Jo, my email is jo@example.com")
    r = await say(s, "yes")
    assert r["intent"] == "enquiry.ask_email" and "type it" in r["text"]


@pytest.mark.parametrize("status", [None, 429, 500])
async def test_when_it_cannot_be_sent_the_customer_is_told_and_given_the_other_ways(tmp_path, monkeypatch, status):
    monkeypatch.setattr(E, "post", FakeHPN(status, {}))
    s, _ = hpn_session(tmp_path)
    s.enquiry.mode = "live"
    await say(s, "My name is Jo, my email is jo@example.com")
    r = await say(s, "yes")
    assert r["intent"] == "enquiry.failed" and "weren't sent" in r["text"]
    assert [link["path"] for link in r["links"]][0] == "/contact-us/"


async def test_test_mode_goes_through_the_conversation_and_sends_nothing(tmp_path, monkeypatch):
    hpn = FakeHPN()
    monkeypatch.setattr(E, "post", hpn)
    s, _ = hpn_session(tmp_path)                                # the profile's default: test
    assert s.enquiry.mode == "test"
    await say(s, "My name is Jo, my email is jo@example.com")
    r = await say(s, "yes")
    assert r["intent"] == "enquiry.sent_test" and "Nothing was sent" in r["text"] and hpn.posts == []


async def test_a_card_number_never_starts_an_enquiry(live):
    s, hpn = live
    r = await say(s, "my card is 4111 1111 1111 1111")
    assert r["intent"] == "smalltalk.card" and hpn.posts == []


async def test_what_she_says_reads_each_detail_as_a_sentence(live):
    from app.hpn.voice import spoken
    s, _ = live
    await say(s, "My name is Jo, my email is jo@example.com")
    said = spoken(s.outbox.of("helper_reply")[-1]["reply"])
    assert "Name: Jo.\nEmail: jo@example.com." in said and "team:\n" in said or "team:" in said
