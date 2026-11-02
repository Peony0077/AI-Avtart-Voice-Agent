"""When the Event helper is not sure: let the local LLM pick which of HPN's own questions was meant.

Spoken questions reach the helper through speech recognition, so they are noisier than typed ones ("phone
party" for "foam party") and phrased more loosely. When the helper's matching falls back ("I'm not sure
about that one yet"), the LLM is shown the list of questions the helper CAN answer - every topic's title
and the live answers' own questions - and asked for the number of the one the customer meant, or 0.

The LLM never writes the answer: the chosen question goes back through the same Event helper, and its
answer is used only if the helper is sure of it. The page then says which question was answered
("Answering: ..."), so a wrong pick is visible instead of passing as an answer to what was asked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# The live answers (prices, nights out...) are not topics: the questions that reach them.
LIVE_QUESTIONS = [
    "Prices and packages", "Wedding prices", "Party prices", "Foam party prices", "Add-ons", "Extra hours",
    "What's on tonight?", "What's on tomorrow?", "What's on this weekend?", "Karaoke nights",
    "Jukebox Bingo nights", "Contact us", "Is my date available?", "Which areas do you serve?",
]


@dataclass(frozen=True)
class Candidate:
    label: str        # what the LLM and the page see
    question: str     # what is asked of the helper


def candidates(engine) -> list[Candidate]:
    """Every question the helper answers with confidence, each once (the tests check each round-trips)."""
    out, seen = [], set()
    for question in LIVE_QUESTIONS:
        out.append(Candidate(question, question))
        seen.add(question.lower())
    for topic in engine.kb.topics:
        if topic.title.lower() in seen:
            continue
        seen.add(topic.title.lower())
        out.append(Candidate(topic.title, topic.title))
    return out


SYSTEM = ("You route customer questions for Houston Party Network (a Houston DJ, karaoke, Jukebox Bingo, foam party "
          "and event entertainment company) to the right answer. The customer spoke, and speech recognition may "
          "have misheard words (for example 'phone party' for 'foam party'). Reply with ONLY the number of the "
          "question below that the customer is asking, or 0 if none of them fits. Never guess a loosely related "
          "one: 0 is the right answer when unsure.")


def prompt(cands: list[Candidate], said: str, previous: str | None = None) -> list[dict]:
    listing = "\n".join(f"{i}. {c.label}" for i, c in enumerate(cands, 1))
    context = f"\nThe assistant's previous answer was about: {previous}" if previous else ""
    return [{"role": "system", "content": f"{SYSTEM}\n\nQuestions:\n{listing}"},
            {"role": "user", "content": f"Customer said: \"{said.strip()[:400]}\"{context}\n\nNumber:"}]


def parse(text: str, count: int) -> int:
    """The number the LLM chose (0 when none, or when the reply is not a number in range)."""
    m = re.search(r"\d+", text or "")
    if not m:
        return 0
    n = int(m.group())
    return n if 1 <= n <= count else 0
