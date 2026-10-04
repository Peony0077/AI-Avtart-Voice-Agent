"""Measure lip-sync timing on rendered frames: does the mouth move when the sound is heard?

    .venv\\Scripts\\python lipsync_check.py [speech.wav ...] [--avatar ID] [--compare]

Renders the speech with the real engine (no browser, no network) and compares, per frame, how much the
mouth differs from the loop's own closed-mouth frame with the loudness of the sound:
  * continuous speech: cross-correlation lag (sub-frame, parabolic peak)
  * isolated words (sound with >= 0.4 s of silence around it): mouth movement centroid - sound centroid
> 0 = the mouth is late. People notice lips about 45 ms early or 125 ms late (ITU-R BT.1359).
Frame k pictures the middle of its 50 ms (the browser blends between frame middles, AvatarStage.jsx).
--compare also renders with MuseTalk's own windowing (get_whisper_chunk) for reference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from common import AVATARS
from engine import Loop, MuseTalkEngine

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "e2e" / "fixtures"
DEFAULT_WAVS = [FIXTURES / f"bench_{n}.wav" for n in ("long", "q1", "q2", "q3")]
MOUTH = (slice(300, 380), slice(215, 325))      # rows, cols of the 512 px frame
PIECE_S = 20                                     # whisper sees at most 30 s at a time


def load(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    wav = wav if wav.ndim == 1 else wav.mean(1)
    return resample_poly(wav, 24000, sr).astype(np.float32) if sr != 24000 else wav


def mouth_change(eng: MuseTalkEngine, loop: Loop, chunks: torch.Tensor, base: int) -> np.ndarray:
    out = []
    for s in range(0, chunks.shape[0], eng.batch_size):
        idx, faces = eng.faces(loop, chunks[s:s + eng.batch_size], base + s)
        for i, f in zip(idx, faces, strict=True):
            a = cv2.cvtColor(eng.compose(loop, i, f)[MOUTH], cv2.COLOR_BGR2GRAY).astype(np.float32)
            b = cv2.cvtColor(loop.frame(i)[MOUTH], cv2.COLOR_BGR2GRAY).astype(np.float32)
            out.append(float(np.abs(a - b).mean()))
    return np.array(out)


def render(eng: MuseTalkEngine, loop: Loop, features, audio: np.ndarray, base: int = 100) -> np.ndarray:
    step = 24000 * PIECE_S
    chunks = torch.cat([features((audio[i:i + step] * 32767).astype(np.int16)) for i in range(0, len(audio), step)])
    return mouth_change(eng, loop, chunks, base)


def xcorr_lag(m: np.ndarray, a: np.ndarray, frame_ms: float, max_frames: int = 6) -> tuple[float, float]:
    lags = list(range(-max_frames, max_frames + 1))
    rs = [float(np.corrcoef(*((m[x:], a[:len(a) - x]) if x >= 0 else (m[:x], a[-x:])))[0, 1]) for x in lags]
    i = int(np.argmax(rs))
    frac = 0.0
    if 0 < i < len(rs) - 1 and (rs[i - 1] - 2 * rs[i] + rs[i + 1]):
        frac = 0.5 * (rs[i - 1] - rs[i + 1]) / (rs[i - 1] - 2 * rs[i] + rs[i + 1])
    return (lags[i] + frac) * frame_ms, rs[i]


def islands(audio: np.ndarray, min_gap_s: float = 0.4) -> list[tuple[int, int]]:
    """(start, end) samples of single words: 0.15-1.2 s of sound with min_gap_s of silence around it."""
    e = np.sqrt(np.convolve(audio ** 2, np.ones(480) / 480, mode="same"))
    loud = e > 0.1 * np.percentile(e, 99)
    edges = np.flatnonzero(np.diff(loud.astype(np.int8)))
    runs = list(zip(edges[::2] + 1, edges[1::2] + 1, strict=False)) if not loud[0] else []
    merged: list[list[int]] = []
    for s, t in runs:                                     # join pieces of one word (gaps < 150 ms)
        if merged and s - merged[-1][1] < 0.15 * 24000:
            merged[-1][1] = t
        else:
            merged.append([s, t])
    gap = int(min_gap_s * 24000)
    return [(s, t) for j, (s, t) in enumerate(merged)
            if (j == 0 or s - merged[j - 1][1] >= gap) and (j == len(merged) - 1 or merged[j + 1][0] - t >= gap)
            and s >= gap and len(audio) - t >= gap and 0.15 * 24000 <= t - s <= 1.2 * 24000]


def word_lags(eng: MuseTalkEngine, loop: Loop, features, audio: np.ndarray) -> list[float]:
    spf = 24000 // eng.fps
    out = []
    for s, t in islands(audio):
        s0 = (s - int(0.4 * 24000)) // spf * spf                       # on the frame grid
        seg = audio[s0: t + int(0.4 * 24000)]
        m = render(eng, loop, features, seg, 100 + s0 // spf)
        m = np.clip(m - np.median(np.concatenate([m[:4], m[-4:]])), 0, None)
        e = np.convolve(seg ** 2, np.ones(480) / 480, mode="same")[::24]   # 1 ms steps
        if m.sum() <= 0:
            continue
        tm = (np.arange(len(m)) + 0.5) * 1000 / eng.fps
        out.append(float((tm * m).sum() / m.sum() - (np.arange(len(e)) * e).sum() / e.sum()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wavs", nargs="*", type=Path, default=DEFAULT_WAVS)
    ap.add_argument("--avatar", default="dev_girl_v2")
    ap.add_argument("--loop", default="friendly")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--decoder", default="taesd")
    ap.add_argument("--compare", action="store_true", help="also MuseTalk's own windowing")
    args = ap.parse_args()

    eng = MuseTalkEngine(fps=args.fps, batch_size=8, decoder=args.decoder)
    loop = Loop.load(AVATARS / args.avatar / "loops" / args.loop)

    @torch.no_grad()
    def musetalk_features(pcm16: np.ndarray) -> torch.Tensor:
        a = resample_poly(pcm16.astype(np.float32) / 32768.0, 2, 3).astype(np.float32)
        feats = eng.audio.feature_extractor(a, return_tensors="pt", sampling_rate=16000).input_features
        return eng.audio.get_whisper_chunk([feats.to(eng.dtype)], eng.device, eng.dtype, eng.whisper, len(a),
                                           fps=args.fps, audio_padding_length_left=2, audio_padding_length_right=2)

    variants = [("engine", eng.audio_features)] + ([("musetalk", musetalk_features)] if args.compare else [])
    spf = 24000 // args.fps
    for path in args.wavs:
        audio = load(path)
        for name, features in variants:
            m = render(eng, loop, features, audio)
            rms = np.array([np.sqrt(np.mean(audio[k * spf:(k + 1) * spf] ** 2)) for k in range(len(m))])
            lag, r = xcorr_lag(m, rms, 1000 / args.fps)
            words = word_lags(eng, loop, features, audio)
            w = (f"isolated words {np.median(words):+4.0f} ms (n {len(words)}, IQR {np.percentile(words, 25):+.0f}.."
                 f"{np.percentile(words, 75):+.0f})") if words else "no isolated words"
            print(f"{path.name:24s} {name:8s} continuous {lag:+4.0f} ms (r {r:.2f}) | {w}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
