"""Where the model weights live, for the helper scripts (the backend's own rule, app/core/config.py).

AVATAR_MODEL_HOME and HF_HOME come from the environment, else from the repo's .env (the scripts are
usually started from a plain shell, where nothing loaded .env), else the platform default:
E:/AI-models on Windows, ~/AI-models on Linux. Use it before anything imports a Hugging Face library:

    from model_home import hf_home
    os.environ.setdefault("HF_HOME", hf_home())
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT = Path("E:/AI-models") if os.name == "nt" else Path.home() / "AI-models"


def dotenv(path: Path = REPO / ".env") -> dict[str, str]:
    """KEY=VALUE lines of .env (comments and blank lines skipped, quotes stripped)."""
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip("'\"")
    return values


def model_home() -> Path:
    return Path(os.environ.get("AVATAR_MODEL_HOME") or dotenv().get("AVATAR_MODEL_HOME") or DEFAULT)


def hf_home() -> str:
    return os.environ.get("HF_HOME") or dotenv().get("HF_HOME") or str(model_home() / "hf")
