"""Environment check for the backend venv.

Prints (and optionally saves as JSON) the facts that must stay true:
Python version, torch build, CUDA availability, GPU name, VRAM, RAM, tools.

Exit code 1 if the CUDA PyTorch build is missing or broken, so the script can
gate installs (see scripts/safe_install.ps1).

Usage:
    .venv\\Scripts\\python scripts/check_env.py [--json logs/env.json] [--expect-torch 2.9.1+cu130]
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect() -> dict:
    info: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            info["gpu"] = props.name
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["vram_total_gib"] = round(total / 2**30, 2)
            info["vram_free_gib"] = round(free / 2**30, 2)
            info["cudnn"] = torch.backends.cudnn.version()
    except Exception as exc:  # noqa: BLE001 - report any import/runtime failure
        info["torch_error"] = repr(exc)
        info["cuda_available"] = False

    smi = _run([
        "nvidia-smi",
        "--query-gpu=driver_version,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if smi:
        driver, used, total = [s.strip() for s in smi.splitlines()[0].split(",")]
        info["driver"] = driver
        info["nvsmi_vram_used_mib"] = int(used)
        info["nvsmi_vram_total_mib"] = int(total)

    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gib"] = round(vm.total / 2**30, 2)
        info["ram_available_gib"] = round(vm.available / 2**30, 2)
    except ImportError:
        info["ram_total_gib"] = None

    for tool in ("ffmpeg", "node", "npm", "git", "uv"):
        info[f"tool_{tool}"] = shutil.which(tool) is not None
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write the report to this JSON file")
    parser.add_argument("--expect-torch", default="2.9.1+cu130", help="required torch version ('' to skip)")
    args = parser.parse_args()

    info = collect()
    for key, value in info.items():
        print(f"{key:24} {value}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(info, indent=2), encoding="utf-8")

    problems = []
    if not info.get("cuda_available"):
        problems.append("CUDA is not available to PyTorch")
    if args.expect_torch and info.get("torch") != args.expect_torch:
        problems.append(f"torch is {info.get('torch')}, expected {args.expect_torch}")
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
