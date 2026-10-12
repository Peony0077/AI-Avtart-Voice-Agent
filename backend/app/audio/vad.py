"""Streaming voice-activity detection (Silero VAD, ONNX on CPU) with utterance segmentation.

Feed raw PCM16 16 kHz microphone bytes as they arrive. The detector emits:
  * ``start``   speech began (``bargein=True`` if it happened while the assistant was talking)
  * ``pause``   early_stt_ms of silence inside an utterance (lets the caller start STT
                speculatively while the endpoint timer is still running); carries the audio
  * ``resume``  speech continued after a ``pause`` (speculative work for that pause is stale)
  * ``end``     utterance finished (trailing silence >= endpoint_ms); carries the audio
The threshold is stricter while the assistant is speaking ("playback-aware"), so
residual echo is less likely to trigger an interruption.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from app.core.config import VADCfg

# silero_vad's import calls torch.set_num_threads(1) for the whole process (its README's advice for the
# TorchScript model), and StreamingVAD did again: the RAG embedder (CPU) then ran on one thread. The ONNX
# model used here has its own session with one thread (silero_vad.utils_vad.OnnxWrapper).
_THREADS = torch.get_num_threads()
from silero_vad import load_silero_vad  # noqa: E402

torch.set_num_threads(_THREADS)

SR = 16000
CHUNK = 512                     # Silero needs 512-sample windows at 16 kHz
CHUNK_MS = CHUNK * 1000 / SR    # 32 ms


@dataclass
class VadEvent:
    kind: str                   # "start" | "pause" | "resume" | "end"
    t: float                    # wall-clock time of the relevant audio (arrival time)
    bargein: bool = False
    audio: np.ndarray | None = None       # float32 mono 16 kHz (for "end")
    speech_end_t: float | None = None     # time the last voiced chunk arrived (for "end")
    stats: dict = field(default_factory=dict)


class StreamingVAD:
    def __init__(self, cfg: VADCfg) -> None:
        self.cfg = cfg
        try:
            self.model = load_silero_vad(onnx=True)
        except ImportError:
            # onnxruntime is not in requirements.txt (a Windows install had it from elsewhere): on a fresh
            # Linux venv the microphone failed to start ("No module named 'onnxruntime'"). The same model
            # as TorchScript, bundled with silero_vad.
            self.model = load_silero_vad(onnx=False)
        self._pending = np.zeros(0, dtype=np.int16)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, int(cfg.preroll_ms / CHUNK_MS)))
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self._pending = np.zeros(0, dtype=np.int16)
        self._preroll.clear()
        self.in_speech = False
        self._voiced_ms = 0.0
        self._silence_ms = 0.0
        self._utterance: list[np.ndarray] = []
        self._last_voice_t = 0.0
        self._start_bargein = False
        self._pause_sent = False
        self.last_prob = 0.0
        self.max_prob = 0.0

    def process(self, pcm16: bytes, assistant_active: bool, now: float | None = None) -> list[VadEvent]:
        now = now or time.time()
        if len(pcm16) % 2:              # (the session refuses such frames; np.frombuffer would raise)
            pcm16 = pcm16[:-1]
        self._pending = np.concatenate([self._pending, np.frombuffer(pcm16, dtype=np.int16)])
        events: list[VadEvent] = []
        n = len(self._pending) // CHUNK
        for i in range(n):
            chunk = self._pending[i * CHUNK:(i + 1) * CHUNK]
            # arrival time of this chunk's last sample (chunks at the end of the buffer are the newest)
            t_chunk = now - (n - 1 - i) * CHUNK_MS / 1000
            ev = self._step(chunk, assistant_active, t_chunk)
            if ev:
                events.append(ev)
        self._pending = self._pending[n * CHUNK:]
        return events

    @property
    def silence_ms(self) -> float:
        """Trailing silence inside the current utterance (0 while the user is talking)."""
        return self._silence_ms

    def _step(self, chunk: np.ndarray, assistant_active: bool, t: float) -> VadEvent | None:
        audio = chunk.astype(np.float32) / 32768.0
        prob = float(self.model(torch.from_numpy(audio), SR).item())
        self.last_prob = prob
        cfg = self.cfg
        if not self.in_speech:
            threshold = cfg.bargein_threshold if assistant_active else cfg.threshold
            need_ms = cfg.bargein_min_ms if assistant_active else cfg.min_speech_ms
            self._preroll.append(audio)
            if prob >= threshold and _dbfs(audio) > cfg.min_dbfs:
                self._voiced_ms += CHUNK_MS
                self._last_voice_t = t
            else:
                self._voiced_ms = max(0.0, self._voiced_ms - CHUNK_MS)  # tolerate single dropouts
            if self._voiced_ms >= need_ms:
                self.in_speech = True
                self._start_bargein = assistant_active
                self._utterance = list(self._preroll)
                self._silence_ms = 0.0
                self.max_prob = prob
                return VadEvent("start", t, bargein=assistant_active)
            return None

        self._utterance.append(audio)
        self.max_prob = max(self.max_prob, prob)
        if prob >= cfg.threshold - 0.15:
            self._silence_ms = 0.0
            self._last_voice_t = t
            if self._pause_sent:          # speech continued after a "pause": speculative work is stale
                self._pause_sent = False
                return VadEvent("resume", t)
        else:
            self._silence_ms += CHUNK_MS
        duration_s = len(self._utterance) * CHUNK_MS / 1000
        if self._silence_ms >= cfg.endpoint_ms or duration_s >= cfg.max_utterance_s:
            utterance = self._trimmed_audio()
            ev = VadEvent("end", t, bargein=self._start_bargein, audio=utterance, speech_end_t=self._last_voice_t,
                          stats={"duration_ms": round(len(utterance) / SR * 1000), "max_prob": round(self.max_prob, 3)})
            self.in_speech = False
            self._voiced_ms = 0.0
            self._pause_sent = False
            self._utterance = []
            self._preroll.clear()
            self.model.reset_states()
            return ev
        if cfg.early_stt_ms and not self._pause_sent and self._silence_ms >= cfg.early_stt_ms:
            self._pause_sent = True
            return VadEvent("pause", t, audio=self._trimmed_audio(), speech_end_t=self._last_voice_t)
        return None

    def _trimmed_audio(self) -> np.ndarray:
        """Utterance audio without the trailing silence (keeps ~120 ms)."""
        keep = max(0, len(self._utterance) - int(max(0.0, self._silence_ms - 120) / CHUNK_MS))
        return np.concatenate(self._utterance[:keep]) if keep else np.zeros(0, np.float32)


def _dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio * audio)) + 1e-9)
    return 20 * np.log10(rms)
