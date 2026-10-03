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
    end = clause.rstrip().rstrip("\"')]")[-1:] if clause.strip() else ""
    if end in ".!?":
        return 260.0
    if end in ",;:":
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
    return TTSResult(audio=np.concatenate([audio[start:end], pause]), sample_rate=sr, words=words,
                     synth_ms=result.synth_ms)


class TTSEngine(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str) -> TTSResult: ...

    def warmup(self) -> float: ...
