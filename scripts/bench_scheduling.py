"""Model scheduling benchmark (spec: "benchmark A. permanently resident models and B. controlled /
sequential GPU scheduling"). For the backend's GPU models, per turn:

  A  resident          model stays on the GPU; cost = the inference itself
  B1 priority gating   (what the backend does: resident weights + one GPU lock) — same cost as A
  B2 swap              weights parked in CPU RAM between turns, moved to the GPU for the turn and back
  B3 load on demand    weights loaded from disk for every turn and freed afterwards

Reports the added latency per turn, the VRAM freed while parked, and the extra system RAM needed.

    .venv\\Scripts\\python scripts\\bench_scheduling.py --json docs\\benchmarks\\phase6_scheduling.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from model_home import hf_home  # noqa: E402  (scripts/model_home.py: the env, .env, else the platform default)

os.environ.setdefault("HF_HOME", hf_home())
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from app.core.config import STTCfg, TTSCfg  # noqa: E402

AUDIO, SR = sf.read(ROOT / "tests/e2e/fixtures/bench_q3.wav", dtype="float32")
AUDIO = AUDIO[int(1.0 * SR): int(3.6 * SR)].astype(np.float32)
TEXT = "Octopuses have three hearts, and two of them pump blood to the gills."


def gpu_mib() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 2**20


def timed(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def med(values: list[float]) -> float:
    return round(statistics.median(values), 1)


def bench(name: str, load, infer, module_of, rounds: int = 5) -> dict:
    out: dict = {"model": name}
    t0 = time.perf_counter()
    obj = load()
    torch.cuda.synchronize()
    out["load_from_disk_ms"] = round((time.perf_counter() - t0) * 1000)
    for _ in range(3):
        infer(obj)
    out["resident_infer_ms"] = med([timed(lambda: infer(obj)) for _ in range(rounds)])
    on_gpu = gpu_mib()
    module = module_of(obj)
    rss0 = psutil.Process().memory_info().rss
    module.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    out["vram_freed_when_parked_mib"] = round(on_gpu - gpu_mib())
    out["extra_ram_when_parked_mib"] = round((psutil.Process().memory_info().rss - rss0) / 2**20)
    swap_in, swap_out, total = [], [], []
    for _ in range(rounds):
        a = timed(lambda: module.to("cuda"))
        b = timed(lambda: infer(obj))
        c = timed(lambda: module.to("cpu"))
        swap_in.append(a)
        swap_out.append(c)
        total.append(a + b)          # what the user waits for (swap-out can happen after the turn)
    out["swap_in_ms"] = med(swap_in)
    out["swap_out_ms"] = med(swap_out)
    out["swap_turn_ms"] = med(total)
    out["swap_added_latency_ms"] = round(out["swap_turn_ms"] - out["resident_infer_ms"], 1)
    module.to("cuda")
    obj = module = None          # free the resident copy before measuring load-on-demand
    gc.collect()
    torch.cuda.empty_cache()
    reload = []
    for _ in range(2):
        t0 = time.perf_counter()
        fresh = load()
        infer(fresh)
        torch.cuda.synchronize()
        reload.append((time.perf_counter() - t0) * 1000)
        fresh = None
        gc.collect()
        torch.cuda.empty_cache()
    out["load_on_demand_turn_ms"] = med(reload)
    out["load_on_demand_added_latency_ms"] = round(out["load_on_demand_turn_ms"] - out["resident_infer_ms"])
    print(json.dumps(out), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    from app.stt.whisper import WhisperSTT
    from app.tts.kokoro_tts import KokoroTTS

    rows = [
        bench("whisper-large-v3-turbo (STT)", lambda: WhisperSTT(STTCfg()), lambda m: m.transcribe(AUDIO),
              lambda m: m.model),
        bench("kokoro-82M (TTS)", lambda: KokoroTTS(TTSCfg()), lambda m: m.synthesize(TEXT), lambda m: m.model),
    ]
    out = {"gpu": torch.cuda.get_device_name(0), "h2d_note": "pageable copies (no pinned buffers)", "rows": rows,
           "ram_available_mib": psutil.virtual_memory().available // 2**20}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
