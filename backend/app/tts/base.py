from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class TTSResult:
    audio: np.ndarray                      # float32 mono at sample_rate
    sample_rate: int
    words: list[tuple[str, float, float]] = field(default_factory=list)  # (word, start_s, end_s)
    synth_ms: float = 0.0

    @property
    def duration_ms(self) -> float:
        return len(self.audio) / self.sample_rate * 1000


def pause_after(clause: str) -> float:
    """Natural pause (ms) after a clause: longer after a sentence than after a comma."""
    end = clause.rstrip().rstrip("\"')]")[-1:]
    if end and end in ".!?":            # ("" is in every string: an empty clause got a sentence's pause)
        return 260.0
    if end and end in ",;:":
        return 130.0
    return 60.0


def trim_silence(result: TTSResult, pause_ms: float, lead_ms: float = 20.0, release_ms: float = 40.0,
                 threshold: float = 0.01) -> TTSResult:
    """Replace the engine's padding with a natural pause. Kokoro wraps every clause in ~0.3 s of silence
    before and ~0.5 s after (measured), which made ~0.8 s gaps between sentences and delayed the first
    audible word of a reply by ~0.3 s. Word timestamps are shifted to the trimmed audio."""
    audio, sr = result.audio, result.sample_rate
    if not len(audio):
        return result
    peak = float(np.abs(audio).max())
    voiced = np.flatnonzero(np.abs(audio) > max(1e-4, threshold * peak))
    if not len(voiced):
        return result
    start = max(0, voiced[0] - int(lead_ms * sr / 1000))
    end = min(len(audio), voiced[-1] + 1 + int(release_ms * sr / 1000))
    pause = np.zeros(int(pause_ms * sr / 1000), dtype=audio.dtype)
    shift = start / sr
    words = [(w, max(0.0, s - shift), max(0.0, e - shift)) for w, s, e in result.words]
    body, words = cap_pauses(audio[start:end], sr, words, threshold=max(1e-4, threshold * peak))
    return TTSResult(audio=np.concatenate([body, pause]), sample_rate=sr, words=words, synth_ms=result.synth_ms)


def cap_pauses(audio: np.ndarray, sr: int, words: list, max_ms: float = 320.0, threshold: float = 0.01,
               win_ms: float = 10.0) -> tuple[np.ndarray, list]:
    """Shorten silences inside a clause to ``max_ms``. Kokoro pauses up to ~520 ms at a comma (at
    speed 0.95: "...their surroundings, thanks to special cells ..."), which sounds halting; people
    pause ~200-400 ms there. The excess is cut from the middle of the silence (near-silent samples, so
    the cut is inaudible); word timestamps after it move up."""
    win = max(1, int(win_ms * sr / 1000))
    n = len(audio) // win
    if n < 3:
        return audio, words
    quiet = np.abs(audio[: n * win]).reshape(n, win).max(axis=1) <= threshold
    keep = int(max_ms / win_ms)
    cuts = []                                      # (start sample, length) to remove
    k = 0
    while k < n:
        if not quiet[k]:
            k += 1
            continue
        j = k
        while j < n and quiet[j]:
            j += 1
        if 0 < k and j < n and j - k > keep:       # inside the clause, not at its ends
            excess = j - k - keep
            cuts.append(((k + keep // 2) * win, excess * win))
        k = j
    if not cuts:
        return audio, words
    pieces, pos = [], 0
    for start, length in cuts:
        pieces.append(audio[pos:start])
        pos = start + length
    pieces.append(audio[pos:])

    def moved(t: float) -> float:
        removed = sum(length for start, length in cuts if start < t * sr)
        return max(0.0, t - removed / sr)

    return np.concatenate(pieces), [(w, moved(s), moved(e)) for w, s, e in words]


class TTSEngine(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str) -> TTSResult: ...

    def warmup(self) -> float: ...
