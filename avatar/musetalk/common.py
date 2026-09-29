"""Shared paths/setup for the avatar worker (runs in avatar/musetalk/.venv).

Weights layout (downloaded by download_weights.py):
    E:/AI-models/avatar/models/{musetalkV15, sd-vae, whisper, dwpose, face-parse-bisent}   MuseTalk's "./models"
    E:/AI-models/avatar/liveportrait/{liveportrait, insightface}                            LivePortrait
    E:/AI-models/avatar/torch_home/hub/checkpoints/s3fd-*.pth                                face detector
Built avatars: E:/AI-models/avatars/<avatar_id>/
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VENDOR = HERE / "vendor"
MODEL_HOME = Path(os.environ.get("AVATAR_MODEL_HOME", "E:/AI-models"))
WEIGHTS = MODEL_HOME / "avatar"
AVATARS = MODEL_HOME / "avatars"

os.environ.setdefault("TORCH_HOME", str(WEIGHTS / "torch_home"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

for sub in ("MuseTalk", "LivePortrait"):
    path = str(VENDOR / sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402

# Legacy .pth checkpoints (MuseTalk, LivePortrait, S3FD) contain pickled objects;
# torch>=2.6 defaults to weights_only=True. These files come from pinned, known sources.
_torch_load = torch.load


def _load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


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
