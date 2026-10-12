#!/usr/bin/env bash
# First-time setup on Linux (tested on Ubuntu 24.04, NVIDIA driver for CUDA 13). The Linux counterpart of
# the Windows steps in README.md. Idempotent: run it again after a failure or to check an install - done
# steps are verified and skipped. Models go to AVATAR_MODEL_HOME (.env; default ~/AI-models), never the repo.
#
#   scripts/setup_linux.sh             # backend venv, llama-server, models, frontend
#   scripts/setup_linux.sh --avatar    # + the avatar worker (its venv, MuseTalk/LivePortrait, weights)
#
# Needs uv, git, curl and Node.js 20+ (npm); ffmpeg for the avatar's H.264 video (sudo apt install ffmpeg).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
AVATAR=0
while [ $# -gt 0 ]; do
  case "$1" in
    --avatar) AVATAR=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

need() { command -v "$1" >/dev/null || { echo "missing: $1 - $2" >&2; exit 1; }; }
need uv "https://docs.astral.sh/uv/getting-started/installation/"
need git "sudo apt install git"
need curl "sudo apt install curl"
need npm "Node.js 20 or newer"
command -v ffmpeg >/dev/null || echo "note: no ffmpeg - the avatar's video falls back to JPEG frames (sudo apt install ffmpeg)"
command -v nvidia-smi >/dev/null || echo "note: no nvidia-smi - the backend needs an NVIDIA GPU and driver"

# .env: where this machine keeps the models (written once; edit it to move them)
if [ ! -f .env ]; then
  printf 'AVATAR_PROFILE=default\nAVATAR_MODEL_HOME=%s\nHF_HOME=%s\n' "$HOME/AI-models" "$HOME/AI-models/hf" > .env
  echo "wrote .env (models in $HOME/AI-models)"
fi
set -a; . ./.env; set +a
export AVATAR_MODEL_HOME="${AVATAR_MODEL_HOME:-$HOME/AI-models}"
export HF_HOME="${HF_HOME:-$AVATAR_MODEL_HOME/hf}"
export HF_HUB_OFFLINE=0          # downloads below (the app itself runs offline)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Download to a temporary name, check the SHA-256, then rename: a partial or altered file is never reused.
fetch() {   # url file sha256
  local url="$1" file="$2" sum="$3"
  if [ -f "$file" ] && echo "$sum  $file" | sha256sum -c --quiet - 2>/dev/null; then return 0; fi
  rm -f "$file" "$file.part"
  echo "downloading $(basename "$file")"
  curl -fL --retry 3 -o "$file.part" "$url"
  echo "$sum  $file.part" | sha256sum -c --quiet - || { rm -f "$file.part"; echo "checksum mismatch: $url" >&2; exit 1; }
  mv "$file.part" "$file"
}

echo "== backend venv (.venv: Python 3.11, torch 2.9.1+cu130)"
PY=.venv/bin/python
[ -x "$PY" ] || uv venv --python 3.11 .venv
if ! "$PY" -c 'import sys, torch; sys.exit(torch.__version__ != "2.9.1+cu130")' 2>/dev/null; then
  # CUDA PyTorch first, from PyTorch's own index; constraints/backend.txt keeps any later install from
  # replacing it (e.g. with a CPU wheel pulled in as a dependency)
  uv pip install --python "$PY" -c constraints/backend.txt --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.9.1+cu130 torchaudio==2.9.1+cu130 torchvision==0.24.1+cu130
fi
# constraints/backend.lock.txt is the Windows venv's freeze: drop its UTF-8 BOM (uv would read the first
# line as a package named "﻿accelerate") and the pins that are Windows-only or a URL (spaCy, below)
sed '1s/^\xEF\xBB\xBF//' constraints/backend.lock.txt \
  | grep -v -E '^(colorama|pywin32|pywin32-ctypes|en-core-web-sm)\b' > "$TMP/backend.lock.txt"
uv pip install --python "$PY" -c constraints/backend.txt -c "$TMP/backend.lock.txt" -r backend/requirements.txt
# spaCy's English model (Kokoro's G2P), a wheel from spaCy's releases
"$PY" -c 'import en_core_web_sm' 2>/dev/null || uv pip install --python "$PY" -c constraints/backend.txt \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
"$PY" scripts/check_env.py --expect-torch 2.9.1+cu130 || { echo "the backend venv has no working CUDA PyTorch" >&2; exit 1; }

echo "== llama-server (llama.cpp b11242, CUDA 12.8 runtime bundled; MIT)"
LLAMA_BUILD=b11242
LLAMA_DIR="tools/llama.cpp/$LLAMA_BUILD-cuda12.8"
LLAMA_TGZ="llama-$LLAMA_BUILD-bin-ubuntu-cuda-12.8-x64.tar.gz"
CUDART_TGZ="cudart-llama-$LLAMA_BUILD-bin-ubuntu-cuda-12.8-x64.tar.gz"
if [ ! -x "$LLAMA_DIR/llama-$LLAMA_BUILD/llama-server" ] || [ ! -f "$LLAMA_DIR/llama-$LLAMA_BUILD/libcudart.so.12" ]; then
  mkdir -p tools/llama.cpp/dl "$LLAMA_DIR"
  base="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD"
  fetch "$base/$LLAMA_TGZ" "tools/llama.cpp/dl/$LLAMA_TGZ" 6ab3e1d60a6cab34806ae23c90956ba668047c3bc056eab3be2bfa73eca24fac
  fetch "$base/$CUDART_TGZ" "tools/llama.cpp/dl/$CUDART_TGZ" e67423c1553f8f4cd22d6412c74604a6d50b9b3cccb48546758b55e912407603
  tar -xzf "tools/llama.cpp/dl/$LLAMA_TGZ" -C "$LLAMA_DIR"      # -> llama-b11242/
  tar -xzf "tools/llama.cpp/dl/$CUDART_TGZ" -C "$LLAMA_DIR"     # -> cudart-llama-b11242-bin-ubuntu-cuda-12.8-x64/
  # the CUDA 12.8 runtime next to llama-server (its RUNPATH is $ORIGIN): no system CUDA toolkit needed,
  # and PyTorch's CUDA 13 libraries stay separate
  cp -f "$LLAMA_DIR/${CUDART_TGZ%.tar.gz}"/lib* "$LLAMA_DIR/llama-$LLAMA_BUILD/"
fi
# config llm.server_exe points to tools/llama.cpp/current/llama-server
ln -sfn "$LLAMA_BUILD-cuda12.8/llama-$LLAMA_BUILD" tools/llama.cpp/current
tools/llama.cpp/current/llama-server --version 2>&1 | grep -i version || { echo "llama-server does not start" >&2; exit 1; }

echo "== models -> $AVATAR_MODEL_HOME"
"$PY" scripts/download_models.py llm voice rag voice_convert

echo "== frontend"
(
  cd frontend
  if [ ! -f node_modules/.package-lock.json ] || [ package-lock.json -nt node_modules/.package-lock.json ]; then
    npm ci --no-audit --no-fund
  fi
  npm run build
)

if [ "$AVATAR" = 1 ]; then
  echo "== avatar worker (avatar/musetalk/.venv, MuseTalk + LivePortrait, weights)"
  bash avatar/musetalk/setup.sh
  avatar/musetalk/.venv/bin/python avatar/musetalk/download_weights.py
fi

echo
echo "done. Start:   scripts/run.sh                      (http://127.0.0.1:8000)"
if [ "$AVATAR" = 1 ]; then
  echo "       scripts/run.sh --profile avatar    (build an avatar first: README.md, \"Make it yours\")"
fi
echo "Internet:      scripts/tunnel.sh --profile avatar"
