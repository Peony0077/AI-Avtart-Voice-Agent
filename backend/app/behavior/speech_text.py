"""Make LLM text safe to speak: strip emojis, markdown and URLs; normalise punctuation.

Works on streamed deltas (character-level rules only), so it can run before the
text is displayed and before it reaches TTS — what the user sees is what is spoken.
"""
from __future__ import annotations

import re

_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF️‍]"
)
_MARKDOWN = re.compile(r"[*_`#~>]")
_URL = re.compile(r"https?://\S+")
_DASH = re.compile(r"\s*—\s*")   # em dash = a pause; an en dash is a range ("2–3") -> hyphen
_REPLACE = {"…": "...", "“": '"', "”": '"', "‘": "'", "’": "'", " ": " ", "–": "-"}


def clean_delta(text: str) -> str:
    text = _URL.sub("", text)
    text = _EMOJI.sub("", text)
    text = _MARKDOWN.sub("", text)
    text = _DASH.sub(", ", text)
    for src, dst in _REPLACE.items():
        text = text.replace(src, dst)
    return text


def collapse_spaces(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", text).strip()


# Spoken forms Kokoro gets wrong (TTS -> Whisper round trip, docs/ARCHITECTURE.md "Speech output"):
# "100°C" -> "degrees C", "9:00" -> "nine zero zero", "~20" -> "tilde 20", "2-3" -> "two three".
# Applied to each clause right before synthesis; the chat text stays as the LLM wrote it.
_UNITS = {"km": "kilometers", "kg": "kilograms", "cm": "centimeters", "mm": "millimeters", "ml": "milliliters",
          "mph": "miles per hour", "km/h": "kilometers per hour", "kph": "kilometers per hour", "lbs": "pounds",
          "lb": "pounds", "hrs": "hours", "mins": "minutes", "sec": "seconds", "secs": "seconds"}
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
    (re.compile(r"\be\.g\.,?", re.I), "for example"),
    (re.compile(r"\bi\.e\.,?", re.I), "that is"),
    (re.compile(r"\betc\.(?=\s*$)", re.I), "et cetera."),
    (re.compile(r"\betc\.", re.I), "et cetera"),
    (re.compile(r"\bapprox\.", re.I), "approximately"),
    # street and place abbreviations (scripts/bench_tts.py: "Main St." was spoken "Maincent")
    (re.compile(r"(\b[A-Z][a-z]+ )?\b(St|Ave|Rd|Blvd|Ln)\.(?=(\s+[A-Z])|\s*$)?"), _street),
    (re.compile(r"\bMt\.(?=\s+[A-Z])"), "Mount"),
    # numeric range "2-3", "10-15" (short numbers only: not phone numbers or dates)
    (re.compile(r"(?<![\d/.-])\b(\d{1,3})\s?-\s?(\d{1,3})\b(?![-/:.]\d)"), r"\1 to \2"),
    # clock times: "9:00" -> "9 o'clock", "13:00" -> "13 hundred", "9:05" -> "9 oh 5", "9:30" -> "9 30"
    (re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b"),
     lambda m: (f"{int(m[1])} o'clock" if int(m[1]) <= 12 else f"{int(m[1])} hundred") if m[2] == "00"
     else f"{int(m[1])} oh {int(m[2])}" if m[2][0] == "0" else f"{int(m[1])} {m[2]}"),
    (re.compile(r"(\d)\s?(" + "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True)) + r")\b"),
     lambda m: f"{m[1]} {_UNITS[m[2]]}"),
]


def normalize_for_speech(text: str) -> str:
    for pattern, repl in _SPEECH_RULES:
        text = pattern.sub(repl, text)
    return collapse_spaces(text)
