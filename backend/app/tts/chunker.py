"""Split streamed LLM text into speakable clauses.

The first clause of a turn is emitted as early as possible (at a comma once it
is long enough) so audio can start while the LLM is still generating; later
clauses are whole sentences for better prosody.
"""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(\s+|$)")
_SOFT_BREAK = re.compile(r"[,;:]\s+")


class ClauseChunker:
    def __init__(self, first_min_chars: int = 14, max_chars: int = 220) -> None:
        self.first_min_chars = first_min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        while True:
            clause = self._take()
            if clause is None:
                return out
            out.append(clause)

    def flush(self) -> list[str]:
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._emitted += 1
            return [rest]
        return []

    def _take(self) -> str | None:
        buf = self._buf
        match = _SENTENCE_END.search(buf)
        # require at least one more char after the punctuation unless the buffer ends there,
        # so "3.5" or "Dr." mid-stream is less likely to split (the regex needs whitespace/end)
        if match and match.end() < len(buf) or (match and buf.endswith(("\n",))):
            return self._cut(match.end())
        if self._emitted == 0:
            soft = None
            for m in _SOFT_BREAK.finditer(buf):
                if m.start() >= self.first_min_chars:
                    soft = m
                    break
            if soft:
                return self._cut(soft.end())
        if len(buf) > self.max_chars:
            cut = buf.rfind(" ", 0, self.max_chars)
            return self._cut(cut if cut > 0 else self.max_chars)
        return None

    def _cut(self, pos: int) -> str | None:
        clause, self._buf = self._buf[:pos].strip(), self._buf[pos:]
        if not clause:
            return None if not self._buf else self._take()
        self._emitted += 1
        return clause
