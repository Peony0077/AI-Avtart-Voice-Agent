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
_DASH = re.compile(r"\s*[—–]\s*")
_REPLACE = {"…": "...", "“": '"', "”": '"', "‘": "'", "’": "'", " ": " "}


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
