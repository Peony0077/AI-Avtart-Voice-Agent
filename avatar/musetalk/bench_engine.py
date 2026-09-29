"""Benchmark MuseTalk lip-sync throughput on this GPU (avatar venv).

Measures frames/s for several batch sizes on a real TTS clip, VRAM, CPU-side blend/JPEG
cost, and — with --llm-url — the slowdown while llama-server is generating concurrently.

    avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\bench_engine.py --avatar dev_girl --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from common import AVATARS, REPO
from engine import Loop, MuseTalkEngine


def llm_load(url: str, stop: threading.Event) -> None:
    body = json.dumps({"messages": [{"role": "user", "content": "Write a long story about the sea."}],
                       "max_tokens": 400, "stream": False}).encode()
    while not stop.is_set():
        try:
            req = urllib.request.Request(f"{url}/v1/chat/completions", body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=60).read()
        except Exception:  # noqa: BLE001
            time.sleep(0.5)


def run(engine: MuseTalkEngine, loop: Loop, pcm: np.ndarray, batch: int) -> dict:
    """Same two-stage pipeline as worker.py: GPU faces(k+1) overlaps CPU compose+JPEG(k)."""
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()
    chunks = engine.audio_features(pcm)
    torch.cuda.synchronize()
    feat_ms = (time.perf_counter() - t0) * 1000
    n = chunks.shape[0]
    starts = [0] + list(range(2, n, batch))
    spans = [(s, starts[i + 1] if i + 1 < len(starts) else n) for i, s in enumerate(starts)]
    first_ms, gpu_ms = None, []

    def compose(idx, faces):
        return [cv2.imencode(".jpg", engine.compose(loop, i, f), [cv2.IMWRITE_JPEG_QUALITY, 80])[1] for i, f in zip(idx, faces, strict=True)]

    t1 = time.perf_counter()
    with ThreadPoolExecutor(1) as cpu:
        pending = None
        for s, e in spans:
            tg = time.perf_counter()
            idx, faces = engine.faces(loop, chunks[s:e], s)
            gpu_ms.append((time.perf_counter() - tg) * 1000 / (e - s))
            if pending is not None:
                pending.result()
            pending = cpu.submit(compose, idx, faces)
            if first_ms is None:
                pending.result()
                first_ms = (time.perf_counter() - t0) * 1000
        pending.result()
    total = time.perf_counter() - t1
    return {"batch": batch, "frames": n, "audio_s": round(len(pcm) / 24000, 2), "features_ms": round(feat_ms, 1),
            "first_frames_ready_ms": round(first_ms, 1), "fps": round(n / total, 1),
            "gpu_ms_per_frame": round(statistics.median(gpu_ms), 1), "realtime_factor": round((n / total) / engine.fps, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", default="dev_girl")
    ap.add_argument("--loop", default="friendly")
    ap.add_argument("--batches", default="2,4,8")
    ap.add_argument("--llm-url", default=None, help="e.g. http://127.0.0.1:8081 to add concurrent LLM load")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    meta = json.loads((AVATARS / args.avatar / "avatar.json").read_text())
    torch.cuda.reset_peak_memory_stats()
    engine = MuseTalkEngine(fps=meta["fps"])
    loop = Loop.load(AVATARS / args.avatar / "loops" / args.loop)
    engine.warmup(loop)
    wav, sr = sf.read(REPO / "tests/e2e/fixtures/mic_story_bargein.wav", dtype="float32")
    speech = wav[int(1.5 * sr): int(4.8 * sr)]
    pcm = (resample_poly(np.tile(speech, 3), 3, 2) * 32767).astype(np.int16)   # ~10 s at 24 kHz
    out = {"engine_load_s": engine.load_s, "fps_target": meta["fps"], "runs": []}
    for b in [int(x) for x in args.batches.split(",")]:
        out["runs"].append(run(engine, loop, pcm, b))
        print(out["runs"][-1], flush=True)
    if args.llm_url:
        stop = threading.Event()
        th = threading.Thread(target=llm_load, args=(args.llm_url, stop), daemon=True)
        th.start()
        time.sleep(1.0)
        best = max(out["runs"], key=lambda r: r["fps"])["batch"]
        r = run(engine, loop, pcm, best)
        r["with_llm_generating"] = True
        out["runs"].append(r)
        print(r, flush=True)
        stop.set()
    out["vram_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    out["vram_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}))
    if args.json:
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
