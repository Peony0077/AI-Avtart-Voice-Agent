"""What arrived from the user's microphone in a session, for the log (owner: "it doesn't hear my words").

The owner's sessions logged nothing about the microphone: not whether audio arrived, how loud it was,
or whether speech was ever detected. One line per session now says, and a warning comes as soon as
10 s of audio were only silence (a muted or wrong input device)."""
from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 16000
SILENT_DBFS = -60.0     # below this the microphone is muted, disabled or the wrong device
SILENT_AFTER_S = 10.0


def dbfs(peak: int) -> float:
    return 20 * math.log10(peak / 32768) if peak > 0 else -math.inf


class MicStats:
    def __init__(self) -> None:
        self.samples = 0
        self.peak = 0
        self.speech = 0              # speech starts detected (VAD)
        self.ignored = 0             # audio received while voice was off on the server
        self._warned = False

    def add(self, pcm16: bytes, voice_on: bool = True) -> str | None:
        """Count a frame of PCM16 16 kHz; returns a warning once if the microphone stays silent."""
        n = len(pcm16) // 2
        if not voice_on:
            self.ignored += n
            return None
        self.samples += n
        if n:
            self.peak = max(self.peak, int(np.abs(np.frombuffer(pcm16[: 2 * n], np.int16).astype(np.int32)).max()))
        if (not self._warned and self.speech == 0 and self.seconds >= SILENT_AFTER_S
                and dbfs(self.peak) < SILENT_DBFS):
            self._warned = True
            return (f"microphone sent {self.seconds:.0f} s of silence (peak {dbfs(self.peak):.0f} dBFS): "
                    "muted, disabled or the wrong input device")
        return None

    @property
    def seconds(self) -> float:
        return self.samples / SAMPLE_RATE

    def summary(self) -> str | None:
        if not self.samples and not self.ignored:
            return None
        parts = [f"{self.seconds:.1f} s received", f"peak {dbfs(self.peak):.0f} dBFS", f"speech detected {self.speech}x"]
        if self.ignored:
            parts.append(f"{self.ignored / SAMPLE_RATE:.1f} s ignored (voice off)")
        return ", ".join(parts)
