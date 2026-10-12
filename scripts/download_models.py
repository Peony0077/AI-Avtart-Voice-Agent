"""Download model weights into MODEL_HOME — never into the repo.

MODEL_HOME is AVATAR_MODEL_HOME from the environment, else from .env, else E:\\AI-models on Windows and
~/AI-models on Linux. Hugging Face repos go to the HF cache at HF_HOME (default MODEL_HOME/hf) so that
``from_pretrained("<repo>")`` finds them offline. Every repo is pinned to the commit this project was
tested with, so a later upload to the repo cannot change what gets installed. Groups can be selected:

    .venv\\Scripts\\python scripts/download_models.py llm          # Qwen3-4B GGUF -> MODEL_HOME/llm
    .venv\\Scripts\\python scripts/download_models.py voice        # Whisper + Kokoro + spaCy
    .venv\\Scripts\\python scripts/download_models.py rag voice_convert
    .venv/bin/python scripts/download_models.py stt_small          # Linux; whisper-small.en for the lite profile
    .venv\\Scripts\\python scripts/download_models.py --list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from model_home import hf_home, model_home

MODEL_HOME = model_home()
os.environ.setdefault("HF_HOME", hf_home())

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402  (after HF_HOME)

# group -> list of (repo_id, commit, allow_patterns or None, note)
MANIFEST: dict[str, list[tuple[str, str, list[str] | None, str]]] = {
    "llm": [
        ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "a06e946bb6b655725eafa393f4a9745d460374c9",
         ["Qwen3-4B-Instruct-2507-Q4_K_M.gguf"], "LLM, Apache-2.0, 2.3 GiB"),
    ],
    "voice": [
        ("openai/whisper-large-v3-turbo", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
         ["*.json", "model.safetensors", "*.txt"], "STT, MIT, ~1.6 GB"),
        ("hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987",
         ["config.json", "kokoro-v1_0.pth", "voices/a[fm]_*.pt", "voices/b[fm]_*.pt"],
         "TTS + the 28 English voices, Apache-2.0, ~0.35 GB"),
    ],
    "stt_small": [   # the lite profile's STT (and the STT fallback)
        ("openai/whisper-small.en", "e8727524f962ee844a7319d92be39ac1bd25655a",
         ["*.json", "model.safetensors", "*.txt"], "STT fallback, Apache-2.0, ~0.5 GB"),
    ],
    "rag": [
        ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
         ["*.json", "model.safetensors", "*.txt", "1_Pooling/*"], "embeddings, MIT, 0.13 GB"),
    ],
    "voice_convert": [
        ("myshell-ai/OpenVoiceV2", "f36e7edfe1684461a8343844af60babc2efbb727",
         ["converter/*"], "tone-colour converter for the light clone voice, MIT, 0.13 GB"),
    ],
    "voice_accents": [   # only scripts/bench_stt.py: non-English Kokoro voices speaking English (accents)
        ("hexgrad/Kokoro-82M", "f3ff3571791e39611d31c381e3a41a3af07b4987",
         ["voices/hf_alpha.pt", "voices/jf_alpha.pt", "voices/ef_dora.pt", "voices/zf_xiaobei.pt"],
         "4 accented voices for the STT benchmark, Apache-2.0, ~2 MB"),
    ],
}
SPACY_WHEEL = ("https://github.com/explosion/spacy-models/releases/download/"
               "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")


def pin_main(snapshot: Path, commit: str) -> None:
    """Point the cache's refs/main at the pinned commit. The app loads repos offline (HF_HUB_OFFLINE=1) by
    their default branch, which the cache resolves through refs/main - and a download by commit hash does
    not write that file, so on a new machine the app would not find the pinned snapshot."""
    ref = snapshot.parents[1] / "refs" / "main"
    if not ref.exists() or ref.read_text().strip() != commit:
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(commit)


def download(group: str) -> None:
    for repo, commit, patterns, note in MANIFEST[group]:
        t0 = time.time()
        print(f"-> {repo}@{commit[:7]} ({note})", flush=True)
        if group == "llm":   # llama-server loads a plain file (llm.model_path is relative to MODEL_HOME)
            path = Path(hf_hub_download(repo, patterns[0], revision=commit, local_dir=MODEL_HOME / "llm")).parent
        else:
            path = Path(snapshot_download(repo, revision=commit, allow_patterns=patterns))
            pin_main(path, commit)
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
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
        print(f"MODEL_HOME {MODEL_HOME}, HF_HOME {os.environ['HF_HOME']}")
        for group, items in MANIFEST.items():
            print(group)
            for repo, commit, _, note in items:
                print(f"   {repo:40} {commit[:7]}  {note}")
        return
    unknown = [g for g in args.groups if g not in MANIFEST]
    if unknown:
        parser.error(f"unknown group(s) {', '.join(unknown)}; groups: {', '.join(MANIFEST)}")
    for group in args.groups:
        download(group)


if __name__ == "__main__":
    main()
