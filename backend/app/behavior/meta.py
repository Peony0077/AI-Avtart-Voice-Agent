"""Split the LLM's behaviour metadata header from the spoken text, while streaming.

The model is asked to start every reply with a compact header
    [friendly|small_nod|0.4]
(emotion | gesture | intensity) followed by the words to speak. The compact form
costs ~12 tokens instead of ~26 for JSON (measured: JSON header delayed the first
spoken word by ~230 ms). The legacy form <meta>{json}</meta> is still accepted.
If the model omits the header, everything is speech and default behaviour is used.
"""
from __future__ import annotations

import json
import re

MAX_HEADER_CHARS = 120
_OPENERS = {"[": "]", "<meta>": "</meta>"}


class MetaStreamParser:
    """Feed text deltas; get back (metadata | None, speech_text) increments."""

    def __init__(self) -> None:
        self._buf = ""
        self._header_done = False
        self._spoke = False
        self.meta: dict | None = None
        self.raw_header: str | None = None

    def feed(self, delta: str) -> tuple[dict | None, str]:
        if self._header_done:
            return None, self._speech(delta)
        self._buf += delta
        stripped = self._buf.lstrip()
        if not stripped:
            return None, ""
        opener = next((o for o in _OPENERS if stripped.startswith(o)), None)
        if opener is None:
            if any(o.startswith(stripped) for o in _OPENERS):  # could still become "<meta>"
                return None, ""
            return self._finish(None, None, stripped)
        closer = _OPENERS[opener]
        end = stripped.find(closer, len(opener))
        if end == -1:
            if len(stripped) > MAX_HEADER_CHARS:  # malformed → give up, speak the rest
                return self._finish(None, None, stripped[len(opener):])
            return None, ""
        body = stripped[len(opener):end]
        raw = stripped[: end + len(closer)]
        meta = _parse_json(body) if opener == "<meta>" else _parse_compact(body)
        if opener == "[" and meta is None:  # an ordinary bracket in speech, not a header
            return self._finish(None, None, stripped)
        return self._finish(meta, raw, stripped[end + len(closer):])

    def flush(self) -> str:
        """End of stream: return any text still buffered as speech."""
        if self._header_done:
            return ""
        self._header_done = True
        rest = self._buf.lstrip()
        return "" if rest.startswith("<meta>") else self._speech(rest)

    def _finish(self, meta: dict | None, raw: str | None, speech: str) -> tuple[dict | None, str]:
        self._header_done = True
        self.meta, self.raw_header, self._buf = meta, raw, ""
        return meta, self._speech(speech)

    def _speech(self, text: str) -> str:
        text = re.sub(r"</?meta>", "", text)
        if not self._spoke:
            text = text.lstrip()
            self._spoke = bool(text)
        return text


def _parse_compact(body: str) -> dict | None:
    parts = [p.strip() for p in body.split("|")]
    if len(parts) != 3 or not parts[0].replace("_", "").isalpha():
        return None
    try:
        intensity = float(parts[2])
    except ValueError:
        return None
    return {"emotion": parts[0], "gesture": parts[1], "intensity": intensity}


def _parse_json(text: str) -> dict | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None
