"""Download model weights into MODEL_HOME (default E:\\AI-models) — never into the repo.

Hugging Face repos go to the HF cache at %HF_HOME% (E:\\AI-models\\hf) so that
``from_pretrained("<repo>")`` finds them offline. Groups can be selected:

    .venv\\Scripts\\python scripts/download_models.py voice        # Whisper + Kokoro + spaCy
    .venv\\Scripts\\python scripts/download_models.py rag
    .venv\\Scripts\\python scripts/download_models.py --list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_HOME = Path(os.environ.get("AVATAR_MODEL_HOME", "E:/AI-models"))
os.environ.setdefault("HF_HOME", str(MODEL_HOME / "hf"))

from huggingface_hub import snapshot_download  # noqa: E402  (after HF_HOME)

# group -> list of (repo_id, allow_patterns or None, note)
MANIFEST: dict[str, list[tuple[str, list[str] | None, str]]] = {
    "voice": [
        ("openai/whisper-large-v3-turbo", ["*.json", "model.safetensors", "*.txt"], "STT, MIT, ~1.6 GB"),
        ("hexgrad/Kokoro-82M", ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt", "voices/af_bella.pt",
                                "voices/am_michael.pt", "voices/bf_emma.pt", "voices/am_adam.pt"], "TTS, Apache-2.0, ~0.35 GB"),
    ],
    "stt_small": [
        ("openai/whisper-small.en", ["*.json", "model.safetensors", "*.txt"], "STT fallback, MIT, ~0.5 GB"),
    ],
    "rag": [
        ("BAAI/bge-small-en-v1.5", ["*.json", "model.safetensors", "*.txt", "1_Pooling/*"], "embeddings, MIT, 0.13 GB"),
    ],
}
SPACY_WHEEL = ("https://github.com/explosion/spacy-models/releases/download/"
               "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")


def download(group: str) -> None:
    for repo, patterns, note in MANIFEST[group]:
        t0 = time.time()
        print(f"-> {repo} ({note})", flush=True)
        path = snapshot_download(repo, allow_patterns=patterns)
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        print(f"   done in {time.time() - t0:.0f}s, {size / 2**30:.2f} GiB at {path}", flush=True)
    if group == "voice":
        try:
            import en_core_web_sm  # noqa: F401
            print("-> spaCy en_core_web_sm already installed")
        except ImportError:
            print("-> spaCy en_core_web_sm (needed by Kokoro's English G2P)")
            root = Path(__file__).resolve().parents[1]
            subprocess.check_call(["uv", "pip", "install", "--python", sys.executable,
                                   "-c", str(root / "constraints" / "backend.txt"), SPACY_WHEEL])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="*")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list or not args.groups:
        for group, items in MANIFEST.items():
            print(group)
            for repo, _, note in items:
                print(f"   {repo:40} {note}")
        return
    for group in args.groups:
        download(group)


if __name__ == "__main__":
    main()
