"""Make LLM text safe to speak: strip emojis, markdown and URLs; normalise punctuation.

Runs before the text is displayed and before it reaches TTS - what the user sees is what is spoken.
The LLM streams tokens, and a URL, an email or "~20" arrives in pieces: SpeechTextStream cleans the
stream a whole word at a time (clean_delta per token read URLs aloud, and turned "john_doe@" into
"johndoe@").
"""
from __future__ import annotations

import re

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF️‍]"
)
# markdown marks; "_" only as emphasis (not inside a word: "john_doe"), "~" not before a number (the
# speech rules say "~20" as "about 20")
_MARKDOWN = re.compile(r"[*`#>]|(?<!\w)_+|_+(?!\w)|~(?!\s?\d)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+?(?=[.,;:!?)\]\"']*(?:\s|$))")   # its sentence's full stop stays
_DASH = re.compile(r"\s*—\s*")   # em dash = a pause; an en dash is a range ("2–3") -> hyphen
_REPLACE = {"…": "...", "“": '"', "”": '"', "‘": "'", "’": "'", " ": " ", "–": "-"}

_LAST_SPACES = re.compile(r"\s+(?=\S*\Z)")
MAX_HELD_CHARS = 64     # a "word" longer than this (no spaces at all) is let through as it is


def clean_delta(text: str) -> str:
    """Clean a piece of text that ends at a word boundary (or a whole text)."""
    text = _URL.sub("", text)
    text = _EMOJI.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _DASH.sub(", ", text)
    for src, dst in _REPLACE.items():
        text = text.replace(src, dst)
    return text


class SpeechTextStream:
    """clean_delta for streamed LLM text: each delta is released up to its last whitespace, the word still
    being written waits for the next delta (or flush), so the rules see whole words. A space after a word
    waits too (an em dash may follow: "normal —" is "normal,"), but not one after punctuation: "Sure! "
    must reach the chunker as a finished sentence at once (the LLM pauses there, session.clause_pauses).
    What is released depends only on the text so far, not on how it was split into deltas."""

    def __init__(self) -> None:
        self._held = ""
        self._last = ""          # the last character released

    def feed(self, delta: str) -> str:
        text = _DASH.sub(", ", self._held + delta)
        spaces = _LAST_SPACES.search(text)
        cut = spaces.end() if spaces else 0
        if spaces and cut == len(text):
            before = text[spaces.start() - 1] if spaces.start() else self._last
            if before.isalnum():
                cut = spaces.start()
        if len(text) - cut > MAX_HELD_CHARS:
            cut = len(text)
        self._held = text[cut:]
        return self._release(text[:cut])

    def flush(self) -> str:
        text, self._held = self._held, ""
        return self._release(text)

    def _release(self, text: str) -> str:
        if self._last == " ":
            text = text.lstrip(" ")
        if text:
            self._last = text[-1]
        return clean_delta(text)


def collapse_spaces(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# Spoken forms Kokoro gets wrong (TTS -> Whisper round trip, docs/ARCHITECTURE.md "Speech output"):
# "100°C" -> "degrees C", "9:00" -> "nine zero zero", "~20" -> "tilde 20", "2-3" -> "two three".
# Applied to each clause right before synthesis; the chat text stays as the LLM wrote it.
_UNITS = {"km": "kilometers", "kg": "kilograms", "cm": "centimeters", "mm": "millimeters", "ml": "milliliters",
          "mph": "miles per hour", "km/h": "kilometers per hour", "kph": "kilometers per hour", "lbs": "pounds",
          "lb": "pounds", "hrs": "hours", "mins": "minutes", "sec": "seconds", "secs": "seconds"}
_RANGE_UNITS = (r"(?:seconds?|minutes?|mins|hours?|hrs|days?|weeks?|months?|years?|times|people|persons|percent|"
                r"dollars|cents|miles|kilometers|km|pounds|lbs|kilograms|kg|inches|feet|meters|items|pieces|steps|"
                r"business days|calls|sessions)")
_STREET = {"St": "Street", "Ave": "Avenue", "Rd": "Road", "Blvd": "Boulevard", "Ln": "Lane"}


def _street(m: re.Match) -> str:
    """"Main St." -> "Main Street", "St. Louis" -> "Saint Louis"; keeps the full stop when the
    abbreviation also ended the sentence ("... on Park Rd. Then ..." / at the end)."""
    name, abbr, before_capital = m[1] or "", m[2], m[3]
    if abbr == "St" and not name and before_capital:
        return "Saint"
    ends = before_capital is not None or m.end() == len(m.string.rstrip())
    return f"{name}{_STREET[abbr]}{'.' if ends else ''}"


_SPEECH_RULES: list[tuple[re.Pattern, object]] = [
    (re.compile(r"(\d)\s?°\s?C\b"), r"\1 degrees Celsius"),
    (re.compile(r"(\d)\s?°\s?F\b"), r"\1 degrees Fahrenheit"),
    (re.compile(r"(\d)\s?°"), r"\1 degrees"),
    (re.compile(r"(?:\babout\s+)?~\s?(?=\d)", re.I), "about "),
    (re.compile(r"~"), ""),
    (re.compile(r"(?<=\w)_(?=\w)"), " underscore "),          # an email or a user name read back
    (re.compile(r"\be\.g\.,?", re.I), "for example"),
    (re.compile(r"\bi\.e\.,?", re.I), "that is"),
    (re.compile(r"\betc\.(?=\s*$)", re.I), "et cetera."),
    (re.compile(r"\betc\.", re.I), "et cetera"),
    (re.compile(r"\bapprox\.", re.I), "approximately"),
    # street and place abbreviations (scripts/bench_tts.py: "Main St." was spoken "Maincent")
    (re.compile(r"(\b[A-Z][a-z]+ )?\b(St|Ave|Rd|Blvd|Ln)\.(?=(\s+[A-Z])|\s*$)?"), _street),
    (re.compile(r"\bMt\.(?=\s+[A-Z])"), "Mount"),
    # numeric range "2-3", "10-15" (short numbers only: not phone numbers or dates), and "10 - 15 minutes";
    # not arithmetic ("x - 5 - 3", "5-3=2", "7 - 2 is 5"): no operator next to it, spaced only before a unit
    (re.compile(r"(?<![\d/.\-+*=×÷])(?<![\-+*=×÷] )\b(\d{1,3})(?:-|\s-\s(?=\d{1,3}\s+" + _RANGE_UNITS + r"\b))"
                r"(\d{1,3})\b(?![-/:.]\d)(?!\s?[\-+*/=×÷]\s?\d)"), r"\1 to \2"),
    # clock times: "9:00" -> "9 o'clock", "13:00" -> "13 hundred", "9:05" -> "9 oh 5", "9:30" -> "9 30"
    (re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b"),
     lambda m: (f"{int(m[1])} o'clock" if int(m[1]) <= 12 else f"{int(m[1])} hundred") if m[2] == "00"
     else f"{int(m[1])} oh {int(m[2])}" if m[2][0] == "0" else f"{int(m[1])} {m[2]}"),
    (re.compile(r"(\d)\s?(" + "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True)) + r")\b"),
     lambda m: f"{m[1]} {_UNITS[m[2]]}"),
]

# Numbered lists (the model writes them despite the prompt): the chunker cut "2." and "3." into
# clauses of their own, spoken "Two." / "Three." with pauses around them, and a clause ending in a
# line break and "1." was spoken "... One." with a long pause (890 ms of silence in the e2e reply).
# Spoken, the items just follow each other; the chat still shows the list.
_LONE_MARK = re.compile(r"^\s*\d{1,2}[.)]\s*$")
_TRAIL_MARK = re.compile(r"\s*\n\s*\d{1,2}[.)]\s*$")
_LIST_MARK = re.compile(r"(^|\n)\s*(?:\d{1,2}[.)]|[-•])\s+")


def normalize_for_speech(text: str) -> str:
    """What TTS gets for a clause ('' = nothing to say)."""
    if _LONE_MARK.match(text):
        return ""
    text = _TRAIL_MARK.sub("", text)
    text = _LIST_MARK.sub(r"\1", text).replace("\n", " ")
    for pattern, repl in _SPEECH_RULES:
        text = pattern.sub(repl, text)
    return collapse_spaces(text)


