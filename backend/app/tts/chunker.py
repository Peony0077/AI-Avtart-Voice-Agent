"""Split streamed LLM text into speakable clauses.

The first clause of a turn is emitted as early as possible (at a comma once it
is long enough) so audio can start while the LLM is still generating; later
clauses are whole sentences for better prosody.
"""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(\s+|$)")
_SOFT_BREAK = re.compile(r"[,;:]\s+")
_LINE_BREAK = re.compile(r"\n")
# a period after these is not a sentence end ("Dr. Lee", "e.g. apples"): splitting there made a
# one-word clause with its own pauses (measured: "Dr." alone was a 1.45 s clip)
_ABBREVIATION = re.compile(r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|mt|vs|etc|approx|dept|inc|ltd|co|fig|vol)"
                           r"|\b(?:e\.g|i\.e|a\.m|p\.m|u\.s|u\.k)|(?:^|\s)(?-i:[A-HJ-Z]))\.$", re.I)   # initials, not "I."


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
        # a line break ends a clause too (lists, paragraphs): "Here are four short facts about octopuses:
        # \n\nOctopuses have three hearts, ..." was one 8.7 s clause, slow to synthesise
        line_end = next((m.start() for m in _LINE_BREAK.finditer(buf) if any(ch.isalnum() for ch in buf[:m.start()])),
                        None)
        # require at least one more char after the punctuation unless the buffer ends there,
        # so "3.5" mid-stream is less likely to split (the regex needs whitespace/end)
        for match in _SENTENCE_END.finditer(buf):
            if line_end is not None and match.start() > line_end:
                break
            if buf[match.start()] == "." and _ABBREVIATION.search(buf[:match.start() + 1]):
                continue
            if match.end() < len(buf) or buf.endswith("\n"):
                return self._cut(match.end())
            break
        if line_end is not None:
            return self._cut(line_end)
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
