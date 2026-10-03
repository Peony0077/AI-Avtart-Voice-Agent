"""LiteAvatar (HumanAIGC, MIT) on this PC - third approach of the avatar renderer comparison.

CPU pipeline: 16 kHz audio -> Paraformer features -> audio2mouth (ONNX) -> 2D mouth generator
(TorchScript) -> pasted into the avatar's pre-recorded background frames. Benchmark only.

    avatar\\liteavatar\\.venv\\Scripts\\python avatar\\liteavatar\\bench_liteavatar.py --json docs\\benchmarks\\phase6_liteavatar.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import soundfile as sf
import torch
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VENDOR = HERE / "vendor" / "lite-avatar"


def speech_16k_bytes(path: Path, seconds: float | None = None) -> bytes:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr)
    if seconds:
        audio = audio[: int(seconds * 16000)]
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


def frames_ms(av, params: list, n: int) -> list[float]:
    out = []
    for i in range(n):
        bg = i % av.bg_video_frame_count
        t = time.perf_counter()
        mouth = av.param2img(params[i % len(params)], bg)
        av.merge_mouth_to_bg(mouth, bg)
        out.append((time.perf_counter() - t) * 1000)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=REPO / "logs" / "clone_light_base.wav",
                    help="English speech sample (Kokoro); any 16-24 kHz WAV")
    ap.add_argument("--workers", type=int, default=4, help="LiteAvatar's own parallel face-generation threads")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    audio_path = args.audio.resolve()
    json_path = args.json.resolve() if args.json else None
    sys.path.insert(0, str(VENDOR))
    os.chdir(VENDOR)                                  # the vendor code opens ./weights/...
    from lite_avatar import liteAvatar

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    av = liteAvatar(data_dir="data/preload", num_threads=args.workers, generate_offline=True, fps=30, use_gpu=False)
    load_s = time.perf_counter() - t0

    speech = speech_16k_bytes(audio_path)
    audio_s = len(speech) / 32000
    av.audio2param(speech)                            # warm-up
    t = time.perf_counter()
    params = av.audio2param(speech)
    a2p_ms = (time.perf_counter() - t) * 1000

    clause = speech_16k_bytes(audio_path, seconds=2.0)
    t = time.perf_counter()
    clause_params = av.audio2param(clause)
    frames_ms(av, clause_params, 1)
    first_frame_ms = (time.perf_counter() - t) * 1000

    frames_ms(av, params, 10)                         # warm-up
    n = min(len(params), 300)
    all_threads = frames_ms(av, params, n)            # one frame at a time, torch uses all cores

    # LiteAvatar's own pipeline: `workers` threads generate frames in parallel (how upstream reaches 30 fps)
    torch.set_num_threads(max(1, (os.cpu_count() or 12) // args.workers))
    t = time.perf_counter()
    for i in range(n):
        av.input_queue.put((params[i], i % av.bg_video_frame_count, i))
    av.input_queue.put(None)
    done = 0
    while av.output_queue.get() is not None:
        done += 1
    parallel_fps = done / (time.perf_counter() - t)

    torch.set_num_threads(1)
    one_thread = frames_ms(av, params, 90)
    bg = 0
    frame, _ = av.merge_mouth_to_bg(av.param2img(params[len(params) // 2], bg), bg)
    (REPO / "logs").mkdir(exist_ok=True)
    cv2.imwrite(str(REPO / "logs" / "liteavatar_frame.jpg"), frame)

    p50_all, p50_one = statistics.median(all_threads), statistics.median(one_thread)
    out = {
        "engine": "LiteAvatar (HumanAIGC/lite-avatar @5b7ec85, sample gallery avatar)",
        "device": "CPU (i5-12400, 6 cores / 12 threads)", "torch": torch.__version__,
        "load_s": round(load_s, 1), "audio_s": round(audio_s, 2),
        "audio_to_params_ms": round(a2p_ms), "audio_to_params_rtf": round(a2p_ms / 1000 / audio_s, 3),
        "frames": len(params), "fps_native": 30,
        "frame_ms_p50_all_threads": round(p50_all, 1), "fps_all_threads": round(1000 / p50_all, 1),
        "frame_ms_p50_one_thread": round(p50_one, 1), "fps_one_thread": round(1000 / p50_one, 1),
        "parallel_workers": args.workers, "fps_parallel_workers": round(parallel_fps, 1),
        "first_frame_after_2s_clause_ms": round(first_frame_ms),
        "output_resolution": [int(frame.shape[1]), int(frame.shape[0])],
        "rss_mib": round(proc.memory_info().rss / 2**20), "rss_added_mib": round((proc.memory_info().rss - rss0) / 2**20),
        "vram_mib": 0,
        "sample_frame": "logs/liteavatar_frame.jpg",
    }
    print(json.dumps(out, indent=2))
    if json_path:
        json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    sys.stdout.flush()
    os._exit(0)                                       # vendor face-generation threads block on their queue


if __name__ == "__main__":
    main()
