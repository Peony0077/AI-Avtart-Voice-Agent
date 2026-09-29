<#
.SYNOPSIS
  Create the isolated avatar-worker environment and fetch pinned third-party code.
  Never touches the backend .venv.

  - avatar/musetalk/.venv          Python 3.11 + torch 2.9.1+cu130 + MuseTalk/LivePortrait deps
  - avatar/musetalk/vendor/MuseTalk      TMElyralab/MuseTalk @ 0a89dec (MIT)
  - avatar/musetalk/vendor/LivePortrait  KlingAIResearch/LivePortrait @ 9b294b3 (MIT; InsightFace models non-commercial)
#>
$ErrorActionPreference = 'Continue'  # git/uv write progress to stderr
$here = $PSScriptRoot
$venv = Join-Path $here '.venv'
$vendor = Join-Path $here 'vendor'
New-Item -ItemType Directory -Force $vendor | Out-Null

function Get-Repo($url, $dir, $commit) {
  if (-not (Test-Path (Join-Path $dir '.git'))) {
    $env:GIT_LFS_SKIP_SMUDGE = '1'
    git clone --filter=blob:none -q $url $dir
  }
  git -C $dir checkout -q $commit
  Write-Host "$dir @ $(git -C $dir rev-parse --short HEAD)"
}
Get-Repo 'https://github.com/TMElyralab/MuseTalk.git' (Join-Path $vendor 'MuseTalk') '0a89dec45a0192b824e3cf4daf96c239440c5ed8'
Get-Repo 'https://github.com/KlingAIResearch/LivePortrait.git' (Join-Path $vendor 'LivePortrait') '9b294b3d0536135442ea73cb01e6cb3ca7029dd3'

if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
  # Use the exact interpreter behind the backend venv (uv's 'cpython-3.11' junction can be
  # blocked by Windows as an "untrusted mount point").
  $home311 = ((Get-Content (Join-Path $here '..\..\.venv\pyvenv.cfg') | Where-Object { $_ -match '^home' }) -split '=\s*')[1].Trim()
  # 'python -m venv' (not 'uv venv'): uv's launcher resolves through the blocked junction.
  & (Join-Path $home311 'python.exe') -m venv $venv
}
$py = Join-Path $venv 'Scripts\python.exe'
uv pip install --python $py --index-strategy unsafe-best-match -r (Join-Path $here 'requirements.txt')
& $py -c "import torch, numpy, diffusers, transformers, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__, 'diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
