"""Light cloned voice: Kokoro speech -> OpenVoice V2 tone-colour converter (Apache-2.0 / MIT, local).

Why not XTTS-v2 with the avatar: XTTS's autoregressive GPT and the 20 fps MuseTalk renderer
time-slice the RTX 3060 and together need ~1.4x its capacity (measured with both running:
avatar 24 -> 10.4 fps, XTTS real-time factor 0.57 -> 0.75), so the lips fall behind the voice.
The converter is a small VITS-style posterior encoder + flow + HiFi-GAN decoder (0.13 GB). It keeps
Kokoro's timing, so word timestamps stay valid, and only changes the timbre to that of the
reference recording. XTTS stays the higher-fidelity choice for the voice-only ``clone`` profile.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from loguru import logger
from scipy.signal import resample_poly

from app.core.config import Settings
from app.tts.base import TTSResult
from app.tts.kokoro_tts import KokoroTTS
from app.tts.openvoice.models import SynthesizerTrn

REPO = "myshell-ai/OpenVoiceV2"
# spoken once at start-up to measure the tone colour of the Kokoro base voice
SOURCE_TEXT = [
    "Hello, it's lovely to meet you. How has your day been so far?",
    "The weather report says it will be sunny tomorrow, with a light breeze in the afternoon.",
    "Let me think about that for a second. I believe the answer is in the second chapter.",
]


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    g = math.gcd(sr_in, sr_out)
    return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


class ToneColorConverter:
    def __init__(self, device: str = "cuda") -> None:
        folder = Path(snapshot_download(REPO, allow_patterns=["converter/*"])) / "converter"
        cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        data = cfg["data"]
        self.sr = data["sampling_rate"]
        self.n_fft, self.hop, self.win = data["filter_length"], data["hop_length"], data["win_length"]
        self.device = device
        self.model = SynthesizerTrn(0, self.n_fft // 2 + 1, n_speakers=data["n_speakers"], **cfg["model"])
        state = torch.load(folder / "checkpoint.pth", map_location="cpu", weights_only=True)["model"]
        missing, _unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"OpenVoice converter checkpoint is missing weights: {missing[:5]}")
        self.model.to(device).eval()
        self.window = torch.hann_window(self.win, device=device)

    def _spec(self, audio: np.ndarray) -> torch.Tensor:
        """Linear magnitude spectrogram exactly as OpenVoice's spectrogram_torch (center=False)."""
        y = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(self.device)[None, None]
        pad = (self.n_fft - self.hop) // 2
        y = torch.nn.functional.pad(y, (pad, pad), mode="reflect")[:, 0]
        spec = torch.stft(y, self.n_fft, hop_length=self.hop, win_length=self.win, window=self.window,
                          center=False, return_complex=True)
        return torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-6)

    @torch.inference_mode()
    def embed(self, clips: list[tuple[np.ndarray, int]]) -> torch.Tensor:
        """Tone-colour embedding, averaged over clips (as OpenVoice's extract_se)."""
        gs = [self.model.ref_enc(self._spec(resample(a, sr, self.sr)).transpose(1, 2)).unsqueeze(-1)
              for a, sr in clips if len(a) > sr // 2]
        return torch.stack(gs).mean(0)

    @torch.inference_mode()
    def convert(self, audio: np.ndarray, sr: int, src: torch.Tensor, tgt: torch.Tensor, tau: float) -> np.ndarray:
        spec = self._spec(resample(audio, sr, self.sr))
        lengths = torch.tensor([spec.shape[-1]], device=self.device)
        out = self.model.voice_conversion(spec, lengths, sid_src=src, sid_tgt=tgt, tau=tau)[0][0, 0]
        out = resample(out.float().cpu().numpy(), self.sr, sr)
        n = len(audio)                      # keep the input length: word timestamps stay exact
        return np.pad(out, (0, max(0, n - len(out))))[:n]


def reference_clips(path: Path, seconds: float = 4.0) -> list[tuple[np.ndarray, int]]:
    """The reference recording in ~4 s pieces (OpenVoice also averages over segments)."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(1)
    step = int(seconds * sr)
    return [(wav[i:i + step], sr) for i in range(0, len(wav), step)]


class ConvertedTTS(KokoroTTS):
    name = "clone_light"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.tts)
        ref = settings.tts.clone_reference_wav
        if not ref:
            raise ValueError("tts.clone_reference_wav is required for engine 'clone_light'")
        t0 = time.perf_counter()
        self.tau = settings.tts.clone_tau
        self.converter = ToneColorConverter(settings.tts.device)
        self.src_se = self.converter.embed([(KokoroTTS.synthesize(self, t).audio, self.sample_rate)
                                            for t in SOURCE_TEXT])
        self.tgt_se = self.converter.embed(reference_clips(settings.repo_path(ref)))
        self.load_s = round(self.load_s + time.perf_counter() - t0, 2)
        logger.info(f"clone_light ready: Kokoro '{settings.tts.voice}' -> timbre of {ref} (tau {self.tau})")

    def synthesize(self, text: str) -> TTSResult:
        t0 = time.perf_counter()
        result = super().synthesize(text)
        if len(result.audio):
            result.audio = self.converter.convert(result.audio, self.sample_rate, self.src_se, self.tgt_se, self.tau)
        result.synth_ms = round((time.perf_counter() - t0) * 1000, 1)
        return result
