"""Speech-to-text with Whisper through transformers (uses the venv's torch 2.9.1+cu130).

Weights are loaded straight to the GPU in fp16 (no full CPU copy — RAM is scarce).
Segment-level transcription: one call per VAD utterance.
"""
from __future__ import annotations

import re
import time

import numpy as np
import torch
from loguru import logger

from app.core.config import STTCfg

# Whisper sometimes "hears" these in noise/silence.
_HALLUCINATIONS = {
    "you", "thank you", "thank you.", "thanks for watching", "thanks for watching!", "bye", "bye.",
    "thank you for watching.", "you.", ".", "", "so", "okay.", "hmm.", "uh", "um",
}


class WhisperSTT:
    def __init__(self, cfg: STTCfg) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()

        t0 = time.perf_counter()
        self.cfg = cfg
        self.device = cfg.device
        self.processor = WhisperProcessor.from_pretrained(cfg.model)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            cfg.model, dtype=torch.float16, attn_implementation="sdpa", device_map=cfg.device, low_cpu_mem_usage=True,
        ).eval()
        self.load_s = round(time.perf_counter() - t0, 2)
        self.last_timing: dict = {}
        logger.info(f"Whisper {cfg.model} loaded in {self.load_s}s")

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if len(audio) < sample_rate * self.cfg.min_utterance_ms / 1000:
            return ""
        t0 = time.perf_counter()
        feats = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=torch.float16)
        t1 = time.perf_counter()
        ids = self.model.generate(feats, language=self.cfg.language, task="transcribe", max_new_tokens=160)
        t2 = time.perf_counter()
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        self.last_timing = {"features_ms": round((t1 - t0) * 1000, 1), "generate_ms": round((t2 - t1) * 1000, 1),
                            "tokens": int(ids.shape[-1])}
        text = re.sub(r"\s+", " ", text)
        if text.lower() in _HALLUCINATIONS:
            return ""
        return text

    def warmup(self) -> float:
        t0 = time.perf_counter()
        rng = np.random.default_rng(0)
        self.transcribe((rng.standard_normal(16000) * 0.01).astype(np.float32))
        return round(time.perf_counter() - t0, 2)
