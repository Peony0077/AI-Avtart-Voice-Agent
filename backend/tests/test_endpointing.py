"""Mid-sentence pauses: does the transcript look like the user will go on?"""
import pytest

from app.session.endpointing import looks_unfinished


@pytest.mark.parametrize("text", [
    "I would like to know", "I would like to know.", "I want to go to the", "Tell me about the history of",
    "My question is about", "I was wondering if", "Um,", "So I think...", "What is the difference between cats and",
    "Can you tell me how",
])
def test_unfinished(text):
    assert looks_unfinished(text)


@pytest.mark.parametrize("text", [
    "What is the capital of Japan?", "Thank you.", "Okay.", "No.", "How many hearts does an octopus have?",
    "I would like to know what the weather will be like tomorrow.", "Tell me a joke.", "Do you know?", "Why?", "",
    # a question mark after a verb counts as finished; if the user goes on anyway, the session merges
    # the next utterance into this question (see Session.interrupt, "continued")
    "Could you explain?",
])
def test_finished(text):
    assert not looks_unfinished(text)
