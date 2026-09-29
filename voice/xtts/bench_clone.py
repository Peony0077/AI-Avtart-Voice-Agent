"""Benchmark the clone voice (XTTS-v2) on this GPU: first-chunk latency, real-time factor, VRAM,
and an objective similarity check using XTTS's own speaker encoder:
cosine(clone output, reference) should be clearly higher than cosine(other voice, reference).

    voice\\xtts\\.venv\\Scripts\\python voice\\xtts\\bench_clone.py --ref data\\voices\\dev_emma.wav --other <wav>
    voice\\xtts\\.venv\\Scripts\\python voice\\xtts\\bench_clone.py --ref <ref.wav> --score a.wav b.wav   # similarity only
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from worker import XTTS

CLAUSES = ["Sure!", "Octopuses have three hearts,", "and two of them pump blood to the gills.",
           "That's a great question.", "The capital of France is Paris, a city famous for art, food and history."]


def speaker_vec(tts: XTTS, wav_path: Path) -> torch.Tensor:
    _latent, spk = tts.model.get_conditioning_latents(audio_path=[str(wav_path)])
    return torch.nn.functional.normalize(spk.flatten().float(), dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--other", type=Path, help="a different voice, for the similarity baseline")
    ap.add_argument("--score", type=Path, nargs="+", help="only print the similarity of these WAVs to --ref")
    ap.add_argument("--out", type=Path, default=Path("logs/clone_sample.wav"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    torch.cuda.reset_peak_memory_stats()
    tts = XTTS(args.ref)
    if args.score:
        ref = speaker_vec(tts, args.ref)
        print(json.dumps({str(w): round(float(speaker_vec(tts, w) @ ref), 3) for w in args.score}, indent=2))
        return
    if args.other is None:
        ap.error("--other is required (unless --score)")
    list(tts.stream("Warming up."))
    rows = []
    for text in CLAUSES * 2:
        t0 = time.perf_counter()
        first, parts = None, []
        for chunk in tts.stream(text):
            first = first or (time.perf_counter() - t0) * 1000
            parts.append(chunk)
        total = (time.perf_counter() - t0) * 1000
        audio = np.concatenate(parts)
        rows.append({"text": text, "first_chunk_ms": round(first, 1), "total_ms": round(total, 1),
                     "audio_ms": round(len(audio) / 24), "rtf": round(total / (len(audio) / 24), 3)})
    long_text = " ".join(CLAUSES)
    sample = np.concatenate(list(tts.stream(long_text)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, sample, 24000)
    ref, clone, other = speaker_vec(tts, args.ref), speaker_vec(tts, args.out), speaker_vec(tts, args.other)
    out = {
        "load_s": tts.load_s,
        "first_chunk_ms_p50": statistics.median(r["first_chunk_ms"] for r in rows),
        "rtf_p50": statistics.median(r["rtf"] for r in rows),
        "similarity_clone_vs_reference": round(float(clone @ ref), 3),
        "similarity_other_voice_vs_reference": round(float(other @ ref), 3),
        "vram_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "sample": str(args.out), "sample_text": long_text, "rows": rows[:len(CLAUSES)],
    }
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    for r in out["rows"]:
        print("  ", r)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
