<#
.SYNOPSIS
  LiteAvatar (HumanAIGC, MIT) - third avatar approach for the renderer comparison (docs/ARCHITECTURE.md).
  CPU audio-to-2D-avatar. Benchmark only: not part of the live stack.

  - avatar/liteavatar/vendor/lite-avatar    HumanAIGC/lite-avatar @ 5b7ec85 (git-ignored)
  - avatar/liteavatar/.venv                 Python 3.11 + CPU torch (never touches the backend .venv)
  - E:\AI-models\liteavatar\weights         Paraformer + audio2mouth weights from ModelScope (1.3 GB)
#>
$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$vendor = Join-Path $here 'vendor\lite-avatar'
if (-not (Test-Path $vendor)) {
  git clone -q https://github.com/HumanAIGC/lite-avatar.git $vendor
  git -C $vendor checkout -q 5b7ec850945e03d56fb290b05fb68440c359fa86
}
$venv = Join-Path $here '.venv'
if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
  $home311 = ((Get-Content (Join-Path $here '..\..\.venv\pyvenv.cfg') | Where-Object { $_ -match '^home' }) -split '=\s*')[1].Trim()
  & (Join-Path $home311 'python.exe') -m venv $venv
}
$py = Join-Path $venv 'Scripts\python.exe'
uv pip install --python $py --index-strategy unsafe-best-match -r (Join-Path $here 'requirements.txt')

# weights live outside the repo; the vendor code expects ./weights -> directory junction
$models = if ($env:AVATAR_MODEL_HOME) { $env:AVATAR_MODEL_HOME } else { 'E:\AI-models' }
$weights = Join-Path $models 'liteavatar\weights'
$para = 'speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
if (-not (Test-Path $weights)) {
  New-Item -ItemType Directory -Force $weights | Out-Null
  Copy-Item -Recurse (Join-Path $vendor "weights\$para") $weights   # small config files from the repo
}
$base = 'https://www.modelscope.cn/models/HumanAIGC-Engineering/LiteAvatarGallery/resolve/master/lite_avatar_weights'
$files = @{ 'model_1.onnx' = 'model_1.onnx'; 'model.pb' = "$para\model.pb"; 'lm.pb' = "$para\lm\lm.pb" }
foreach ($name in $files.Keys) {
  $dest = Join-Path $weights $files[$name]
  if (-not (Test-Path $dest)) {
    New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
    Write-Host "downloading $name"
    # -f: an HTTP error fails instead of saving the error page as the weights; a temporary name, so an
    # interrupted download is not taken as done by the next run
    curl.exe -fsSL -o "$dest.part" "$base/$name"
    if ($LASTEXITCODE -ne 0) { Remove-Item "$dest.part" -ErrorAction SilentlyContinue; throw "download failed: $base/$name" }
    Move-Item -Force "$dest.part" $dest
  }
}
$link = Join-Path $vendor 'weights'
if (-not ((Get-Item $link -ErrorAction SilentlyContinue).LinkType -eq 'Junction')) {
  Remove-Item -Recurse -Force $link -ErrorAction SilentlyContinue
  New-Item -ItemType Junction -Path $link -Target $weights | Out-Null
}
$sample = Join-Path $vendor 'data\preload'          # sample_data.zip unpacks to data\preload
if (-not (Test-Path $sample)) { Expand-Archive (Join-Path $vendor 'data\sample_data.zip') (Join-Path $vendor 'data') }
& $py -c "import torch, onnxruntime; print('torch', torch.__version__, 'onnxruntime', onnxruntime.__version__)"
