#!/usr/bin/env bash
# Linux counterpart of avatar/musetalk/setup.ps1: the isolated avatar-worker environment and the pinned
# third-party code. Never touches the backend .venv. Idempotent (run it again after a failure).
#
#   avatar/musetalk/.venv                  Python 3.11 + torch 2.9.1+cu130 + MuseTalk/LivePortrait deps
#   avatar/musetalk/vendor/MuseTalk        TMElyralab/MuseTalk @ 0a89dec (MIT)
#   avatar/musetalk/vendor/LivePortrait    KlingAIResearch/LivePortrait @ 9b294b3 (MIT; the InsightFace
#                                          models its cropper uses are NON-COMMERCIAL, docs/THIRD_PARTY_NOTICES.md)
#
# Weights afterwards: avatar/musetalk/.venv/bin/python avatar/musetalk/download_weights.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$HERE/vendor"
mkdir -p "$VENDOR"
# the repos keep sample media and weights in Git LFS: not needed (weights come from download_weights.py)
export GIT_LFS_SKIP_SMUDGE=1

get_repo() {   # url dir commit
  local url="$1" dir="$2" commit="$3"
  if [ ! -d "$dir/.git" ]; then
    git clone --filter=blob:none -q "$url" "$dir"
  fi
  # an existing clone may predate the pinned commit: fetch only when it is missing
  git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q "$commit"
  echo "$dir @ $(git -C "$dir" rev-parse --short HEAD)"
}
get_repo https://github.com/TMElyralab/MuseTalk.git "$VENDOR/MuseTalk" 0a89dec45a0192b824e3cf4daf96c239440c5ed8
get_repo https://github.com/KlingAIResearch/LivePortrait.git "$VENDOR/LivePortrait" 9b294b3d0536135442ea73cb01e6cb3ca7029dd3

cd "$HERE"
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
# unsafe-best-match: torch comes from PyTorch's cu130 index (requirements.txt), everything else from PyPI
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
.venv/bin/python -c "import torch, numpy, diffusers, transformers, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
