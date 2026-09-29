"""GPU / memory health helpers (NVML via nvidia-ml-py, RAM via psutil)."""
from __future__ import annotations

import os

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
    return pynvml.nvmlDeviceGetMemoryInfo(_HANDLE).free / 2**30
