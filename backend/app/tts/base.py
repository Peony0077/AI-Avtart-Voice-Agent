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


class TTSEngine(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str) -> TTSResult: ...

    def warmup(self) -> float: ...
