from app.leads import lead_request
from app.llm.prompt import fit_transcript


def test_a_short_transcript_is_unchanged():
    assert fit_transcript("Customer: hi", 100) == "Customer: hi"
    assert fit_transcript("x" * 10_000, None) == "x" * 10_000


def test_a_long_transcript_keeps_its_start_and_end():
    text = "START " + "middle " * 5000 + "my email is sam@example.com"
    out = fit_transcript(text, 2000)
    assert len(out) <= 2000
    assert out.startswith("START ")
    assert out.endswith("my email is sam@example.com")
    assert "left out" in out


def test_the_lead_request_is_cut_to_its_room():
    text = "Customer: hello\n" * 5000
    request = lead_request(text, max_chars=3000)
    assert len(request[1]["content"]) < 3200
