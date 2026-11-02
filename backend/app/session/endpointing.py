"""Did the user stop talking, or only pause in the middle of a sentence?

The VAD ends an utterance after 550 ms of silence. People pause longer than that while they think,
especially in a second language; each half then became its own turn and the avatar answered the
first half (seen in user testing; scripts/bench_stt.py: 60 of 60 sentences with a 0.9 s pause were
split). Whisper punctuates the halves anyway ("I would like to know.", "Could you explain?"), so the
check looks at the last word: a sentence that ends in "to", "and", "the", "know", "explain", ...
probably goes on, and the session waits a little longer before answering.
"""
from __future__ import annotations

import re

# a sentence never ends with these: conjunctions, prepositions, articles, possessives, fillers
_ALWAYS_CONTINUES = {
    "and", "but", "or", "because", "if", "when", "while", "than", "whether", "the", "a", "an", "my", "your", "our", "their", "his", "her", "its",
    "these", "those", "um", "uh", "er", "erm", "i", "i'm",
}
# ...these usually take an object or a clause ("I would like to know | what ..."), but also end real
# questions ("Do you know?", "How many hearts does it have?"): unfinished only without a question mark
_USUALLY_CONTINUES = {
    "to", "so", "that", "which", "who", "like", "as", "this", "some", "any", "is", "are", "was", "were", "be",
    "know", "want", "need", "think", "mean", "explain", "tell", "ask", "wonder", "wondering", "show", "find",
    "get", "have", "has", "make", "say", "see", "learn", "understand", "can", "could", "would", "should",
    "will", "do", "does", "did", "how", "what", "why", "where", "well", "just", "also", "then", "maybe",
    "really", "very",
    # prepositions end real questions too ("Where are you from?", "What are you looking for?"): they were
    # always "unfinished", and every such question waited continue_ms (1 s) longer for its answer
    "of", "in", "on", "at", "for", "with", "about", "from", "into", "by",
}
_WORD = re.compile(r"[a-z']+")


def looks_unfinished(text: str) -> bool:
    """True if the transcript probably stops in the middle of a sentence."""
    t = text.strip().lower()
    if not t:
        return False
    if t.endswith(("...", "…", ",", "-", ";", ":")):
        return True
    words = _WORD.findall(t)
    if not words:
        return False
    return words[-1] in _ALWAYS_CONTINUES or (words[-1] in _USUALLY_CONTINUES and not t.endswith("?"))
