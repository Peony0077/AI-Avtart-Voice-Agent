"""Leads from customer conversations (app/leads.py)."""
from app.leads import LeadStore, is_lead, lead_request, parse_lead
from app.llm.prompt import ConversationMemory


def test_the_llms_answer_becomes_a_clean_lead():
    text = ('Sure! {"name": "Sam Rivera", "email": "Sam.Rivera @Gmail.com", "phone": "(415) 555-0134", '
            '"company": "", "need": "solar panels for a 3-bedroom house", "timeline": "this summer", '
            '"budget": null, "interest": "High", "next_step": "free roof check on Tuesday", "notes": "N/A", "x": 1}')
    lead = parse_lead(text)
    assert lead["email"] == "sam.rivera@gmail.com" and lead["phone"] == "4155550134"
    assert lead["interest"] == "high" and lead["budget"] == "" and lead["notes"] == "" and "x" not in lead
    assert is_lead(lead)


def test_no_lead_without_a_way_to_reach_them_or_real_interest():
    assert parse_lead("I could not find anything.") is None
    assert parse_lead("{not json}") is None
    empty = parse_lead('{"name": "", "email": "", "phone": "", "need": "just browsing", "interest": "low"}')
    assert not is_lead(empty)
    keen = parse_lead('{"need": "a quote for 20 laptops", "interest": "medium"}')
    assert is_lead(keen)
    bad_email = parse_lead('{"email": "sam at gmail", "name": "Sam"}')
    assert bad_email["email"] == "" and "sam at gmail" in bad_email["notes"]   # kept for the owner, not as an email


def test_the_request_uses_only_what_the_customer_said():
    mem = ConversationMemory(persona="Ava")
    mem.add("(The user started the conversation.)", "Hi there! I'm Ava.")
    mem.add("I'm Sam, I need a website for my bakery.", "Great! What's the best email to reach you?")
    t = mem.lead_transcript()
    assert t.startswith("Assistant: Hi there!") and "Customer: I'm Sam" in t and "(The user" not in t
    req = lead_request(t, "Acme Web")
    assert "ONLY what the customer said" in req[0]["content"] and "for Acme Web" in req[0]["content"]
    assert "Customer: I'm Sam" in req[1]["content"]


def test_leads_are_stored_once_per_conversation_and_exported_as_csv(tmp_path):
    store = LeadStore(tmp_path / "leads.jsonl")
    assert store.load() == []
    lead = parse_lead('{"name": "Sam", "email": "sam@example.com", "need": "a website"}')
    store.save(lead, "s1", "visitor-1")
    store.save({**lead, "phone": "5550134000"}, "s1", "visitor-1")       # the same conversation again: updated
    store.save(parse_lead('{"name": "Kim"}'), "s2")
    rows = store.load()
    assert [r["session"] for r in rows] == ["s1", "s2"] and rows[0]["phone"] == "5550134000"
    csv = LeadStore.to_csv(rows)
    assert csv.splitlines()[0].startswith("time,name,email,phone") and "sam@example.com" in csv


def test_only_conversations_that_can_hold_a_lead_are_read():
    from app.leads import worth_reading
    assert not worth_reading(["(The user started the conversation.)", "What is the capital of France?"])
    assert worth_reading(["Hi, I'm Sam and my email is sam@example.com"])
    assert worth_reading(["Call me at 512 555 0134"])
    assert worth_reading(["Do you install solar panels?", "How much does it cost?"])
    # spoken contact details (speech recognition writes "at" and "dot") and introductions
    for said in ("You can reach me at sam at gmail dot com", "My name is Sam Rivera", "Here's my phone",
                 "I'm Sam, I need a quote"):
        assert worth_reading([said]), said
    assert not worth_reading(["Can you name three planets?"])
