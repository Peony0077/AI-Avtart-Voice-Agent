"""GPU / memory health helpers (NVML via nvidia-ml-py, RAM via psutil), and GPU work under the lock."""
from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import Any

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # noqa: BLE001 - NVML unavailable → degrade gracefully
    pynvml = None
    _HANDLE = None


def gpu_snapshot() -> dict:
    snap: dict = {"available": _HANDLE is not None}
    if _HANDLE is not None:
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(_HANDLE)
            util = pynvml.nvmlDeviceGetUtilizationRates(_HANDLE)
            snap.update(
                name=pynvml.nvmlDeviceGetName(_HANDLE),
                vram_total_mib=mem.total // 2**20,
                vram_used_mib=mem.used // 2**20,
                vram_free_mib=mem.free // 2**20,
                gpu_util_pct=util.gpu,
                temperature_c=pynvml.nvmlDeviceGetTemperature(_HANDLE, pynvml.NVML_TEMPERATURE_GPU),
            )
        except pynvml.NVMLError as exc:   # e.g. the driver was updated or reset: /health and start-up go on
            snap.update(available=False, error=f"NVML: {exc}")
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    snap.update(
        ram_total_mib=vm.total // 2**20,
        ram_available_mib=vm.available // 2**20,
        backend_rss_mib=proc.memory_info().rss // 2**20,
    )
    return snap


def vram_free_gib() -> float | None:
    if _HANDLE is None:
        return None
    try:
        return pynvml.nvmlDeviceGetMemoryInfo(_HANDLE).free / 2**30
    except pynvml.NVMLError:
        return None                         # unknown: the VRAM check lets the load go ahead


async def locked_thread(lock: asyncio.Lock, fn: Callable[..., Any], *args: Any) -> Any:
    """``fn(*args)`` in a worker thread while holding ``lock``. A cancelled caller keeps the lock until the
    thread has finished: the GPU work goes on regardless (a thread cannot be stopped), and with the lock
    released at once a barge-in let the next transcription or synthesis overlap it."""
    async with lock:
        work = asyncio.ensure_future(asyncio.to_thread(fn, *args))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            await asyncio.wait({work})
            raise
