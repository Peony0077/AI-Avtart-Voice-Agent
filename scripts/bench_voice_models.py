"""Benchmark the voice models in isolation: load time, VRAM, STT latency/accuracy, TTS latency.

Usage: .venv\\Scripts\\python scripts/bench_voice_models.py [--json docs/benchmarks/phase2_models.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("HF_HOME", "E:/AI-models/hf")

from app.core.config import STTCfg, TTSCfg  # noqa: E402
from app.core.gpu import gpu_snapshot  # noqa: E402
from app.stt.whisper import WhisperSTT  # noqa: E402
from app.tts.chunker import ClauseChunker  # noqa: E402
from app.tts.kokoro_tts import KokoroTTS  # noqa: E402

CLAUSES = [
    "Sure!", "Octopuses have three hearts,", "and two of them pump blood to the gills.",
    "That's a great question.", "The capital of France is Paris, a city famous for art, food and history.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--stt-model", default="openai/whisper-large-v3-turbo")
    args = parser.parse_args()

    out: dict = {"vram_before_mib": gpu_snapshot()["vram_used_mib"]}
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    stt = WhisperSTT(STTCfg(model=args.stt_model))
    out["stt_load_s"] = round(time.perf_counter() - t0, 2)
    out["stt_warmup_s"] = stt.warmup()
    out["vram_after_stt_mib"] = gpu_snapshot()["vram_used_mib"]

    t0 = time.perf_counter()
    tts = KokoroTTS(TTSCfg())
    out["tts_load_s"] = round(time.perf_counter() - t0, 2)
    out["tts_warmup_s"] = tts.warmup()
    out["vram_after_tts_mib"] = gpu_snapshot()["vram_used_mib"]

    # TTS: latency per clause and real-time factor
    tts_rows = []
    for text in CLAUSES * 2:
        res = tts.synthesize(text)
        tts_rows.append({"text": text, "synth_ms": res.synth_ms, "audio_ms": round(res.duration_ms),
                         "rtf": round(res.synth_ms / max(1, res.duration_ms), 3), "words": len(res.words)})
    out["tts"] = {"synth_ms_p50": statistics.median(r["synth_ms"] for r in tts_rows),
                  "synth_ms_short_clause": tts_rows[0]["synth_ms"],
                  "rtf_p50": statistics.median(r["rtf"] for r in tts_rows), "rows": tts_rows[:len(CLAUSES)]}

    # STT: latency + accuracy on the Kokoro-generated fixtures
    manifest = json.loads((ROOT / "tests/e2e/fixtures/manifest.json").read_text())
    stt_rows = []
    for name, meta in manifest.items():
        wav, sr = sf.read(ROOT / f"tests/e2e/fixtures/{name}.wav", dtype="float32")
        for ev in meta["speech"]:
            seg = wav[int(ev["start_s"] * sr): int(ev["end_s"] * sr) + int(0.2 * sr)]
            t0 = time.perf_counter()
            text = stt.transcribe(seg)
            ms = (time.perf_counter() - t0) * 1000
            stt_rows.append({"expected": ev["text"], "got": text, "ms": round(ms, 1),
                             "audio_s": round(len(seg) / sr, 2)})
    out["stt"] = {"ms_p50": statistics.median(r["ms"] for r in stt_rows), "ms_max": max(r["ms"] for r in stt_rows),
                  "rows": stt_rows}
    out["torch_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    out["vram_end_mib"] = gpu_snapshot()["vram_used_mib"]
    out["backend_rss_mib"] = gpu_snapshot()["backend_rss_mib"]

    chunker = ClauseChunker()
    stream = "Sure, octopuses have three hearts. Two pump blood to the gills, and one to the body! Neat, right?"
    out["chunker_demo"] = [c for i in range(0, len(stream), 4) for c in chunker.feed(stream[i:i + 4])] + chunker.flush()

    print(json.dumps({k: v for k, v in out.items() if k not in ("tts", "stt")}, indent=2))
    print("TTS", json.dumps({k: v for k, v in out["tts"].items() if k != "rows"}))
    for r in out["tts"]["rows"]:
        print("  ", r)
    print("STT", json.dumps({k: v for k, v in out["stt"].items() if k != "rows"}))
    for r in stt_rows:
        print("  ", r)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
