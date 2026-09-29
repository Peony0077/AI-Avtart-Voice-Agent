"""GPU baseline benchmark (Phase 0).

Measures, on this machine:
  * CUDA context cost (VRAM used by the process after init, via NVML)
  * fp16 / bf16 matmul throughput
  * host->device / device->host bandwidth, pinned vs pageable (decides whether
    swapping model weights between turns is viable)
  * a SAFE allocation ceiling: allocates in 256 MiB steps and stops while at least
    --reserve-gib stays free (never provokes WDDM system-memory fallback)

Usage:  .venv\\Scripts\\python scripts/bench_gpu.py [--json logs/bench_gpu.json]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pynvml
import torch


def nvml_proc_used_mib(handle) -> int | None:
    pid = os.getpid()
    for getter in (pynvml.nvmlDeviceGetComputeRunningProcesses, pynvml.nvmlDeviceGetGraphicsRunningProcesses):
        try:
            for proc in getter(handle):
                if proc.pid == pid and proc.usedGpuMemory:
                    return proc.usedGpuMemory // 2**20
        except pynvml.NVMLError:
            pass
    return None  # WDDM often hides per-process usage


def matmul_tflops(dtype: torch.dtype, n: int = 4096, iters: int = 50) -> float:
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return round(2 * n**3 * iters / (time.perf_counter() - t0) / 1e12, 2)


def bandwidth_gbs(pinned: bool, direction: str, mib: int = 512, iters: int = 5) -> float:
    host = torch.empty(mib * 2**20, dtype=torch.uint8, pin_memory=pinned)
    dev = torch.empty(mib * 2**20, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        if direction == "h2d":
            dev.copy_(host, non_blocking=pinned)
        else:
            host.copy_(dev, non_blocking=pinned)
    torch.cuda.synchronize()
    return round(mib * iters / 1024 / (time.perf_counter() - t0), 2)


def safe_ceiling_gib(reserve_gib: float) -> float:
    blocks = []
    step = 256 * 2**20
    try:
        while torch.cuda.mem_get_info()[0] - step > reserve_gib * 2**30:
            blocks.append(torch.empty(step, dtype=torch.uint8, device="cuda"))
    except torch.OutOfMemoryError:
        pass
    got = len(blocks) * step / 2**30
    del blocks
    torch.cuda.empty_cache()
    return round(got, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--reserve-gib", type=float, default=0.5)
    args = parser.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    used_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used // 2**20
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    used_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used // 2**20

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "vram_used_before_ctx_mib": used_before,
        "vram_used_after_ctx_mib": used_after,
        "cuda_context_mib_device_delta": used_after - used_before,
        "cuda_context_mib_nvml_process": nvml_proc_used_mib(handle),
        "fp16_tflops": matmul_tflops(torch.float16),
        "bf16_tflops": matmul_tflops(torch.bfloat16),
        "h2d_pinned_gbs": bandwidth_gbs(True, "h2d"),
        "h2d_pageable_gbs": bandwidth_gbs(False, "h2d"),
        "d2h_pinned_gbs": bandwidth_gbs(True, "d2h"),
        "safe_alloc_ceiling_gib": safe_ceiling_gib(args.reserve_gib),
        "reserve_gib": args.reserve_gib,
    }
    for key, value in result.items():
        print(f"{key:32} {value}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
