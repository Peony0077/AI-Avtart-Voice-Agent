"""Speech-to-text with Whisper through transformers (uses the venv's torch 2.9.1+cu130).

Weights are loaded straight to the GPU in fp16 (no full CPU copy — RAM is scarce).
Segment-level transcription: one call per VAD utterance.

Decoding uses a minimal greedy loop instead of ``generate()``: same prompt, same token suppression,
identical transcripts on all 15 test fixtures (both models), ~18 ms less per utterance (196 -> 178 ms
warm, large-v3-turbo). The encoder dominates: 141 ms with the GPU clocked up, ~330 ms after 2 s idle.
"""
from __future__ import annotations

import re
import time

import numpy as np
import torch
from loguru import logger

from app.core.config import STTCfg

# What Whisper "hears" in breath, clicks, hum or room noise (measured: "you" or "."), never a real turn.
_JUNK = {"", ".", "you", "you.", "so", "so.", "uh", "uh.", "um", "um.", "hmm", "hmm.", "mm.", "ah."}
# Subtitle phrases from its training data: never said to an avatar.
_SUBTITLE = {"thanks for watching", "thanks for watching!", "thank you for watching", "thank you for watching.",
             "thank you for watching!", "please subscribe.", "subscribe to my channel."}
# Real short answers that are also classic hallucinations: kept when Whisper is confident. Dropping them
# always (the earlier rule) threw away 55 of 90 spoken "Thank you." / "Okay." in scripts/bench_stt.py;
# real ones scored avg log-prob -0.02 .. -0.63 (one noisy -1.14), the threshold is OpenAI's default.
_SHORT_ANSWERS = {"thank you", "thank you.", "thank you!", "thanks.", "thanks", "thanks!", "okay", "okay.",
                  "ok.", "bye", "bye.", "bye-bye.", "bye!"}
_MIN_LOGPROB_SHORT = -1.0


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
        # English-only checkpoints (*.en) reject language/task arguments
        self.english_only = not getattr(self.model.generation_config, "is_multilingual", True)
        g = self.model.generation_config
        prompt = [g.decoder_start_token_id]
        if not self.english_only:
            prompt += [g.lang_to_id[f"<|{cfg.language}|>"], g.task_to_id["transcribe"]]
        self._prompt = torch.tensor([prompt + [g.no_timestamps_token_id]], device=self.device)
        self._suppress = torch.tensor(g.suppress_tokens or [], dtype=torch.long, device=self.device)
        self._begin_suppress = torch.tensor(g.begin_suppress_tokens or [], dtype=torch.long, device=self.device)
        self._eos = g.eos_token_id
        lang_to_id = getattr(g, "lang_to_id", None) or {}
        self._lang_ids = torch.tensor(list(lang_to_id.values()), device=self.device) if lang_to_id else None
        self._lang_index = (list(lang_to_id).index(f"<|{cfg.language}|>")
                            if f"<|{cfg.language}|>" in lang_to_id else None)
        self.last_info: dict = {}
        self.fast_decode = True
        self._warm_feats = torch.zeros(1, self.model.config.num_mel_bins, 3000, dtype=torch.float16,
                                       device=self.device)
        logger.info(f"Whisper {cfg.model} loaded in {self.load_s}s")

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if len(audio) < sample_rate * self.cfg.min_utterance_ms / 1000:
            return ""
        t0 = time.perf_counter()
        feats = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
        feats = feats.to(self.device, dtype=torch.float16)
        t1 = time.perf_counter()
        ids = self._greedy(feats) if self.fast_decode else self._generate(feats)
        t2 = time.perf_counter()
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        self.last_timing = {"features_ms": round((t1 - t0) * 1000, 1), "generate_ms": round((t2 - t1) * 1000, 1),
                            "tokens": int(ids.shape[-1])}
        text = re.sub(r"\s+", " ", text)
        key = text.lower()
        if key in _JUNK or key in _SUBTITLE:
            return ""
        if key in _SHORT_ANSWERS and self.last_info.get("avg_logprob", 0.0) < _MIN_LOGPROB_SHORT:
            return ""
        return text

    def _generate(self, feats: torch.Tensor) -> torch.Tensor:
        lang = {} if self.english_only else {"language": self.cfg.language, "task": "transcribe"}
        return self.model.generate(feats, max_new_tokens=160, **lang)

    def _greedy(self, feats: torch.Tensor, max_new_tokens: int = 160) -> torch.Tensor:
        """Greedy decoding equivalent to generate() for short-form transcription without timestamps."""
        m = self.model
        enc = m.model.encoder(feats)
        step, past, out, logprob = self._prompt, None, [], 0.0
        info: dict = {}
        for i in range(max_new_tokens):
            res = m(encoder_outputs=enc, decoder_input_ids=step, past_key_values=past, use_cache=True)
            past = res.past_key_values
            if i == 0 and self._lang_ids is not None and self._lang_index is not None:
                # for the log only: how sure Whisper is the speech is English (low for strong accents too,
                # measured 0.02-0.36 for accented English, so it is not used to reject anything).
                # (large-v3-turbo gives <|nospeech|> ~0 at this position even for silence: not usable.)
                langs = torch.softmax(res.logits[0, 0].float(), dim=-1)[self._lang_ids]
                info["language_prob"] = round(float(langs[self._lang_index] / langs.sum()), 3)
            logits = res.logits[0, -1]
            if len(self._suppress):
                logits[self._suppress] = -float("inf")
            if i == 0 and len(self._begin_suppress):
                logits[self._begin_suppress] = -float("inf")
            token = int(logits.argmax())
            logprob += float(torch.log_softmax(logits.float(), dim=-1)[token])
            if token == self._eos:
                break
            out.append(token)
            step = torch.tensor([[token]], device=self.device)
        info["avg_logprob"] = round(logprob / (len(out) + 1), 3)
        self.last_info = info
        return torch.cat([self._prompt[0], torch.tensor(out, dtype=torch.long, device=self.device)])[None]

    @torch.inference_mode()
    def keep_warm(self) -> None:
        """One encoder pass on silence: keeps the GPU clocked up while the user is talking.

        Measured: after 2 s of GPU idle the encoder takes 150-300 ms instead of 141 ms (the clocks
        drop within ~2 s); with an encoder pass every 250-500 ms it stays at ~133 ms. Small matmul
        or memcopy keep-alives, a CPU spin or a higher process priority did not help.
        """
        self.model.model.encoder(self._warm_feats)
        torch.cuda.synchronize()

    def warmup(self) -> float:
        t0 = time.perf_counter()
        rng = np.random.default_rng(0)
        self.transcribe((rng.standard_normal(16000) * 0.01).astype(np.float32))
        return round(time.perf_counter() - t0, 2)
