"""Stock-voice TTS: Kokoro-82M (Apache-2.0), 24 kHz, word timestamps from its duration predictor."""
from __future__ import annotations

import time

import numpy as np
import torch
from loguru import logger

from app.core.config import TTSCfg
from app.tts.base import TTSResult

REPO = "hexgrad/Kokoro-82M"


class KokoroTTS:
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, cfg: TTSCfg) -> None:
        from kokoro import KModel, KPipeline

        t0 = time.perf_counter()
        self.cfg = cfg
        self.model = KModel(repo_id=REPO).to(cfg.device).eval()
        # voice prefix a*/b* = American/British English; the G2P must match (accent is in the phonemes)
        self.voice = cfg.voice
        self.lang = cfg.voice[0] if cfg.voice[:1] in ("a", "b") else "a"
        self.pipeline = KPipeline(lang_code=self.lang, repo_id=REPO, model=self.model)
        self.pipeline.load_voice(cfg.voice)
        self.load_s = round(time.perf_counter() - t0, 2)
        logger.info(f"Kokoro loaded in {self.load_s}s (voice {cfg.voice}, {cfg.device})")

    @torch.inference_mode()
    def synthesize(self, text: str) -> TTSResult:
        t0 = time.perf_counter()
        pieces: list[np.ndarray] = []
        words: list[tuple[str, float, float]] = []
        offset = 0.0
        for result in self.pipeline(text, voice=self.voice, speed=self.cfg.speed):
            if result.audio is None:
                continue
            audio = result.audio.detach().float().cpu().numpy()
            for tok in result.tokens or []:
                if tok.start_ts is not None and tok.end_ts is not None and tok.text.strip():
                    words.append((tok.text, offset + tok.start_ts, offset + tok.end_ts))
            pieces.append(audio)
            offset += len(audio) / self.sample_rate
        audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        return TTSResult(audio=audio, sample_rate=self.sample_rate, words=words,
                         synth_ms=round((time.perf_counter() - t0) * 1000, 1))

    def set_voice(self, voice: str) -> None:
        """Another stock voice (persona setting). British voices (b*) need the British phonemizer."""
        from kokoro import KPipeline

        lang = voice[0] if voice[:1] in ("a", "b") else "a"
        if lang != self.lang:
            self.pipeline = KPipeline(lang_code=lang, repo_id=REPO, model=self.model)
            self.lang = lang
        self.pipeline.load_voice(voice)
        self.voice = voice
        logger.info(f"Kokoro voice {voice}")

    @staticmethod
    def voices() -> list[str]:
        """English voices available offline (in the Hugging Face cache)."""
        import os
        from pathlib import Path

        hub = Path(os.environ.get("HF_HOME", "")) / "hub" / "models--hexgrad--Kokoro-82M" / "snapshots"
        names = {p.stem for p in hub.glob("*/voices/*.pt")}
        return sorted(n for n in names if n[:1] in ("a", "b"))

    def warmup(self) -> float:
        t0 = time.perf_counter()
        self.synthesize("Hello! I'm ready.")
        return round(time.perf_counter() - t0, 2)
