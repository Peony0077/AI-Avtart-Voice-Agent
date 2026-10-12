"""Does GPU idle time slow down the next inference? (P-state / clock ramp-up on Windows)

Transcribes the same utterance back-to-back vs. after an idle gap, optionally with a
background "keep-warm" load during the gap, and samples SM clocks via NVML.
"""
from __future__ import annotations

import os
import statistics
import sys
import threading
import time
from pathlib import Path

import pynvml
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from model_home import hf_home  # noqa: E402  (scripts/model_home.py: the env, .env, else the platform default)

os.environ.setdefault("HF_HOME", hf_home())
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from app.core.config import STTCfg  # noqa: E402
from app.stt.whisper import WhisperSTT  # noqa: E402

pynvml.nvmlInit()
H = pynvml.nvmlDeviceGetHandleByIndex(0)


def clock() -> int:
    return pynvml.nvmlDeviceGetClockInfo(H, pynvml.NVML_CLOCK_SM)


def keep_warm(stop: threading.Event) -> None:
    x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    while not stop.is_set():
        for _ in range(4):
            x = torch.tanh(x @ x)
        torch.cuda.synchronize()
        time.sleep(0.015)


def main() -> None:
    stt = WhisperSTT(STTCfg())
    wav, _ = sf.read(ROOT / "tests/e2e/fixtures/bench_q1.wav", dtype="float32")
    stt.warmup()
    results = {}
    for mode in ("back_to_back", "idle_2s", "idle_2s_keepwarm"):
        times, clocks = [], []
        for _ in range(6):
            if mode != "back_to_back":
                stop = threading.Event()
                warm = threading.Thread(target=keep_warm, args=(stop,)) if mode.endswith("keepwarm") else None
                if warm:
                    warm.start()
                time.sleep(2.0)
                if warm:
                    stop.set()
                    warm.join()
            clocks.append(clock())
            t0 = time.perf_counter()
            stt.transcribe(wav)
            times.append((time.perf_counter() - t0) * 1000)
        results[mode] = {"ms_p50": round(statistics.median(times)), "ms_all": [round(t) for t in times],
                         "sm_clock_mhz_before": clocks}
        print(mode, results[mode])


if __name__ == "__main__":
    main()
