"""Shared paths/setup for the avatar worker (runs in avatar/musetalk/.venv).

Weights layout (downloaded by download_weights.py), MODEL_HOME = E:/AI-models (Windows) or ~/AI-models:
    MODEL_HOME/avatar/models/{musetalkV15, sd-vae, whisper, dwpose, face-parse-bisent, taesd, realesrgan}
    MODEL_HOME/avatar/liveportrait/{liveportrait, insightface}                            LivePortrait
    MODEL_HOME/avatar/torch_home/hub/checkpoints/s3fd-*.pth                                face detector
Built avatars: MODEL_HOME/avatars/<avatar_id>/
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VENDOR = HERE / "vendor"


def _model_home() -> Path:
    """AVATAR_MODEL_HOME from the environment, else from the repo's .env (a worker started by hand does not
    have the backend's environment), else the platform default (as backend/app/core/config.py)."""
    value = os.environ.get("AVATAR_MODEL_HOME")
    env_file = REPO / ".env"
    if not value and env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            key, sep, val = line.partition("=")
            if sep and key.strip() == "AVATAR_MODEL_HOME":
                value = val.strip().strip('"').strip("'")
    if value:
        return Path(value)
    return Path("E:/AI-models") if os.name == "nt" else Path.home() / "AI-models"


MODEL_HOME = _model_home()
WEIGHTS = MODEL_HOME / "avatar"
AVATARS = MODEL_HOME / "avatars"

os.environ.setdefault("TORCH_HOME", str(WEIGHTS / "torch_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

for sub in ("MuseTalk", "LivePortrait"):
    path = str(VENDOR / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import hashlib  # noqa: E402
from functools import lru_cache  # noqa: E402

import torch  # noqa: E402

# torch>=2.6 loads checkpoints with weights_only=True (no code can run while loading). Two files that the
# vendored MuseTalk code loads are in PyTorch's legacy format, which only the full unpickler reads: these
# exact files - checked by SHA-256 (their published names start with it) - and nothing else. (All
# checkpoints used to be loaded with weights_only=False.)
LEGACY_CHECKPOINTS = {
    "resnet18-5c106cde.pth": "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8",   # face parsing
    "s3fd-619a316812.pth": "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543",     # face detector
}
_torch_load = torch.load


@lru_cache
def _sha256(path: str, mtime: float) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(f, *args, **kwargs):
    if "weights_only" not in kwargs and isinstance(f, str | os.PathLike):
        expected = LEGACY_CHECKPOINTS.get(Path(f).name)
        if expected is not None and _sha256(str(f), os.path.getmtime(f)) == expected:
            kwargs["weights_only"] = False
    return _torch_load(f, *args, **kwargs)


torch.load = _load


@contextlib.contextmanager
def chdir(path: Path):
    """MuseTalk resolves some weights via './models/...' relative paths."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)
