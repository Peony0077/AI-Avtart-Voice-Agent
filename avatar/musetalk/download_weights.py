"""Download avatar weights into MODEL_HOME/avatar.

MODEL_HOME is AVATAR_MODEL_HOME from the environment, else from the repo's .env, else E:\\AI-models on
Windows and ~/AI-models on Linux. MuseTalk v1.5 (MIT) + its dependencies, TAESD (the fast latent
decoder), Real-ESRGAN (sharpening) and LivePortrait (MIT code). The InsightFace detection models that
LivePortrait's cropper uses (buffalo_l) are licensed for NON-COMMERCIAL research only - see
docs/THIRD_PARTY_NOTICES.md before any commercial use. Every Hugging Face repo is pinned to a commit and
every plain URL download is checked against its SHA-256, so a changed upload cannot slip in.
Run with any Python that has huggingface_hub (e.g. the avatar venv):
    avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\download_weights.py
    avatar/musetalk/.venv/bin/python avatar/musetalk/download_weights.py
"""
from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[2]


def _dotenv_value(key: str) -> str | None:
    """KEY=VALUE from the repo's .env (this script is often started from a plain shell)."""
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() == key and not name.startswith("#"):
                return value.strip().strip("'\"")
    return None


_DEFAULT_HOME = Path("E:/AI-models") if os.name == "nt" else Path.home() / "AI-models"
ROOT = Path(os.environ.get("AVATAR_MODEL_HOME") or _dotenv_value("AVATAR_MODEL_HOME") or _DEFAULT_HOME) / "avatar"

# commits the weights were downloaded and tested at (huggingface.co/api/models/<repo> "sha")
REVISIONS = {
    "TMElyralab/MuseTalk": "2bcb936e2fddb4d86db4c62fd45b387d0c061571",
    "stabilityai/sd-vae-ft-mse": "31f26fdeee1355a5c34592e401dd41e45d25a493",
    "openai/whisper-tiny": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "yzd-v/DWPose": "1a7144101628d69ee7a3768d1ee3a094070dc388",
    "ManyOtherFunctions/face-parse-bisent": "0073b233a5a3c4b1377d4dbf49245017938a72b5",
    "KlingTeam/LivePortrait": "82a4fa6735ca58432b6ce39301b4b9ee066dea47",
    "madebyollin/taesd": "614f76814bbe30edbe2e627ace1c2234c81a2c0e",
}

HF_FILES = [
    # (repo, filename, target subdir)            license
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "models"),            # MIT
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "models"),
    ("stabilityai/sd-vae-ft-mse", "config.json", "models/sd-vae"),        # MIT
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", "models/sd-vae"),
    # the tiny SD autoencoder: MuseTalk's latent decoder by default (avatar.decoder "taesd", engine.py)
    ("madebyollin/taesd", "config.json", "models/taesd"),                 # MIT
    ("madebyollin/taesd", "diffusion_pytorch_model.safetensors", "models/taesd"),
    ("openai/whisper-tiny", "config.json", "models/whisper"),             # Apache-2.0 (weights)
    ("openai/whisper-tiny", "pytorch_model.bin", "models/whisper"),
    ("openai/whisper-tiny", "preprocessor_config.json", "models/whisper"),
    ("yzd-v/DWPose", "dw-ll_ucoco_384.onnx", "models/dwpose"),            # Apache-2.0
    ("ManyOtherFunctions/face-parse-bisent", "79999_iter.pth", "models/face-parse-bisent"),   # WTFPL
    ("ManyOtherFunctions/face-parse-bisent", "resnet18-5c106cde.pth", "models/face-parse-bisent"),
    ("KlingTeam/LivePortrait", "liveportrait/base_models/appearance_feature_extractor.pth", "liveportrait"),  # MIT
    ("KlingTeam/LivePortrait", "liveportrait/base_models/motion_extractor.pth", "liveportrait"),
    ("KlingTeam/LivePortrait", "liveportrait/base_models/spade_generator.pth", "liveportrait"),
    ("KlingTeam/LivePortrait", "liveportrait/base_models/warping_module.pth", "liveportrait"),
    ("KlingTeam/LivePortrait", "liveportrait/retargeting_models/stitching_retargeting_module.pth", "liveportrait"),
    ("KlingTeam/LivePortrait", "liveportrait/landmark.onnx", "liveportrait"),
    ("KlingTeam/LivePortrait", "insightface/models/buffalo_l/2d106det.onnx", "liveportrait"),   # NON-COMMERCIAL
    ("KlingTeam/LivePortrait", "insightface/models/buffalo_l/det_10g.onnx", "liveportrait"),    # NON-COMMERCIAL
]
# (url, target relative to ROOT, SHA-256)
URLS = [
    # S3FD face detector used by MuseTalk's preprocessing (face-alignment project, BSD-3-Clause); the
    # file name carries the start of this digest (torch.hub convention)
    ("https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
     "torch_home/hub/checkpoints/s3fd-619a316812.pth",
     "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543"),
    # Real-ESRGAN (xinntao, BSD-3-Clause): avatar sharpening
    ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
     "models/realesrgan/RealESRGAN_x2plus.pth",
     "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"),
    ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
     "models/realesrgan/realesr-general-x4v3.pth",
     "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, target: Path, digest: str) -> None:
    """Download to a temporary name, check the SHA-256, then rename: an interrupted or altered download
    never ends up under the real name (which the next run would otherwise take as done)."""
    if target.exists():
        if sha256(target) == digest:
            return
        print(f"  {target.name}: wrong SHA-256, downloading again")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.unlink(missing_ok=True)
    urllib.request.urlretrieve(url, part)
    got = sha256(part)
    if got != digest:
        part.unlink()
        raise SystemExit(f"{url}: SHA-256 {got} does not match the expected {digest}")
    part.replace(target)


def main() -> None:
    t0 = time.time()
    for repo, filename, sub in HF_FILES:
        path = hf_hub_download(repo, filename, revision=REVISIONS[repo], local_dir=ROOT / sub)
        print(f"{Path(path).stat().st_size / 2**20:8.1f} MiB  {repo}/{filename}", flush=True)
    for url, rel, digest in URLS:
        target = ROOT / rel
        fetch(url, target, digest)
        print(f"{target.stat().st_size / 2**20:8.1f} MiB  {url}", flush=True)
    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file() and ".cache" not in f.parts)
    print(f"done in {time.time() - t0:.0f}s, {total / 2**30:.2f} GiB under {ROOT}")


if __name__ == "__main__":
    main()
