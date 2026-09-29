"""Benchmark the light clone voice (Kokoro -> OpenVoice V2 converter): cost per clause, and samples
for the objective checks (intelligibility with Whisper here; speaker similarity with
voice\\xtts\\bench_clone.py --score, the same metric used for XTTS).

    .venv\\Scripts\\python scripts\\bench_clone_light.py --ref data\\voices\\dev_emma.wav --json docs\\benchmarks\\x.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("HF_HOME", "E:/AI-models/hf")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.tts.converted_tts import ConvertedTTS  # noqa: E402
from app.tts.kokoro_tts import KokoroTTS  # noqa: E402

CLAUSES = ["Sure!", "Octopuses have three hearts,", "and two of them pump blood to the gills.",
           "That's a great question.", "The capital of France is Paris, a city famous for art, food and history."]


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--voice", default="af_heart", help="Kokoro base voice")
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=ROOT / "logs")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    settings = Settings()
    settings.tts.voice, settings.tts.clone_reference_wav, settings.tts.clone_tau = args.voice, str(args.ref), args.tau
    torch.cuda.reset_peak_memory_stats()
    tts = ConvertedTTS(settings)
    tts.warmup()
    rows = []
    for text in CLAUSES * 3:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        base = KokoroTTS.synthesize(tts, text)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        tts.converter.convert(base.audio, tts.sample_rate, tts.src_se, tts.tgt_se, tts.tau)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        rows.append({"text": text, "kokoro_ms": round((t1 - t0) * 1000, 1), "convert_ms": round((t2 - t1) * 1000, 1),
                     "audio_ms": round(base.duration_ms)})
    rows = rows[len(CLAUSES):]
    long_text = " ".join(CLAUSES)
    args.out.mkdir(parents=True, exist_ok=True)
    sample, base_sample = args.out / "clone_light_sample.wav", args.out / "clone_light_base.wav"
    sf.write(sample, tts.synthesize(long_text).audio, 24000)
    sf.write(base_sample, KokoroTTS.synthesize(tts, long_text).audio, 24000)

    from app.stt.whisper import WhisperSTT
    stt = WhisperSTT(settings.stt)
    audio, _sr = sf.read(sample, dtype="float32")
    heard = stt.transcribe(resample_poly(audio, 2, 3).astype(np.float32))      # 24 -> 16 kHz
    ref_w, got_w = words(long_text), words(heard)
    missing = sum(1 for w in ref_w if w not in got_w)
    out = {
        "base_voice": args.voice, "tau": args.tau, "load_s": tts.load_s,
        "kokoro_ms_p50": statistics.median(r["kokoro_ms"] for r in rows),
        "convert_ms_p50": statistics.median(r["convert_ms"] for r in rows),
        "convert_rtf": round(sum(r["convert_ms"] for r in rows) / sum(r["audio_ms"] for r in rows), 4),
        "total_rtf": round(sum(r["kokoro_ms"] + r["convert_ms"] for r in rows) / sum(r["audio_ms"] for r in rows), 4),
        "vram_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "whisper_transcript": heard, "approx_word_error": round(missing / len(ref_w), 3),
        "sample": str(sample), "base_sample": str(base_sample), "rows": rows[:len(CLAUSES)],
    }
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
