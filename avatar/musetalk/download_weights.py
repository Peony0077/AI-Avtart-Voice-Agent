"""Download avatar weights into MODEL_HOME/avatar (default E:\\AI-models\\avatar).

MuseTalk v1.5 (MIT) + its dependencies, and LivePortrait (MIT code; InsightFace
detection models are non-commercial - fine for this personal project).
Run with any Python that has huggingface_hub (e.g. the avatar venv):
    avatar\\musetalk\\.venv\\Scripts\\python avatar\\musetalk\\download_weights.py
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(os.environ.get("AVATAR_MODEL_HOME", "E:/AI-models")) / "avatar"

HF_FILES = [
    # (repo, filename, target subdir)            license
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "models"),            # MIT
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "models"),
    ("stabilityai/sd-vae-ft-mse", "config.json", "models/sd-vae"),        # MIT
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.bin", "models/sd-vae"),
    ("openai/whisper-tiny", "config.json", "models/whisper"),             # Apache-2.0 (weights)
    ("openai/whisper-tiny", "pytorch_model.bin", "models/whisper"),
    ("openai/whisper-tiny", "preprocessor_config.json", "models/whisper"),
    ("yzd-v/DWPose", "dw-ll_ucoco_384.onnx", "models/dwpose"),            # Apache-2.0
    ("ManyOtherFunctions/face-parse-bisent", "79999_iter.pth", "models/face-parse-bisent"),   # WTFPL
    ("ManyOtherFunctions/face-parse-bisent", "resnet18-5c106cde.pth", "models/face-parse-bisent"),
    ("KlingTeam/LivePortrait", "liveportrait/base_models/appearance_feature_extractor.pth", "liveportrait"),
    ("KlingTeam/LivePortrait", "liveportrait/base_models/motion_extractor.pth", "liveportrait"),
    ("KlingTeam/LivePortrait", "liveportrait/base_models/spade_generator.pth", "liveportrait"),
    ("KlingTeam/LivePortrait", "liveportrait/base_models/warping_module.pth", "liveportrait"),
    ("KlingTeam/LivePortrait", "liveportrait/retargeting_models/stitching_retargeting_module.pth", "liveportrait"),
    ("KlingTeam/LivePortrait", "liveportrait/landmark.onnx", "liveportrait"),
    ("KlingTeam/LivePortrait", "insightface/models/buffalo_l/2d106det.onnx", "liveportrait"),   # non-commercial
    ("KlingTeam/LivePortrait", "insightface/models/buffalo_l/det_10g.onnx", "liveportrait"),    # non-commercial
]
# S3FD face detector used by MuseTalk's preprocessing (face-alignment project, BSD-3)
URLS = [("https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth", "torch_home/hub/checkpoints/s3fd-619a316812.pth")]


def main() -> None:
    t0 = time.time()
    for repo, filename, sub in HF_FILES:
        path = hf_hub_download(repo, filename, local_dir=ROOT / sub)
        print(f"{Path(path).stat().st_size / 2**20:8.1f} MiB  {repo}/{filename}", flush=True)
    for url, rel in URLS:
        target = ROOT / rel
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, target)
        print(f"{target.stat().st_size / 2**20:8.1f} MiB  {url}")
    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file() and ".cache" not in f.parts)
    print(f"done in {time.time() - t0:.0f}s, {total / 2**30:.2f} GiB under {ROOT}")


if __name__ == "__main__":
    main()
