<#
.SYNOPSIS
  Create the isolated clone-voice TTS environment (voice/xtts/.venv) and download XTTS-v2
  weights to MODEL_HOME\tts\XTTS-v2 (AVATAR_MODEL_HOME from the environment or .env, default
  E:\AI-models). Never touches the backend .venv.
  XTTS-v2 is licensed under the Coqui Public Model License: NON-COMMERCIAL use only
  (docs/THIRD_PARTY_NOTICES.md).
#>
$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot
$venv = Join-Path $here '.venv'
if (-not (Test-Path (Join-Path $venv 'Scripts\python.exe'))) {
  $home311 = ((Get-Content (Join-Path $here '..\..\.venv\pyvenv.cfg') | Where-Object { $_ -match '^home' }) -split '=\s*')[1].Trim()
  & (Join-Path $home311 'python.exe') -m venv $venv
}
$py = Join-Path $venv 'Scripts\python.exe'
uv pip install --python $py --index-strategy unsafe-best-match -r (Join-Path $here 'requirements.txt')
# the same model folder as the backend and the worker (voice/xtts/worker.py reads AVATAR_MODEL_HOME)
$models = $env:AVATAR_MODEL_HOME
$dotenv = Join-Path $here '..\..\.env'
if (-not $models -and (Test-Path $dotenv)) {
  $m = Select-String -Path $dotenv -Pattern '^\s*AVATAR_MODEL_HOME\s*=\s*(.+?)\s*$' | Select-Object -First 1
  if ($m) { $models = $m.Matches[0].Groups[1].Value.Trim('"').Trim("'") }
}
if (-not $models) { $models = 'E:\AI-models' }
$dest = Join-Path $models 'tts\XTTS-v2'
$offline = $env:HF_HUB_OFFLINE
$env:HF_HUB_OFFLINE = '0'
# pinned to the commit this project was tested with; the folder is passed as an argument (no quoting issues)
& $py -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download('coqui/XTTS-v2', revision='6c2b0d75eae4b7047358e3b6bd9325f857d43f77', local_dir=sys.argv[1], allow_patterns=['config.json','model.pth','vocab.json','speakers_xtts.pth','dvae.pth','mel_stats.pth']))" $dest
$env:HF_HUB_OFFLINE = $offline
& $py -c "import torch, TTS, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'coqui-tts', TTS.__version__, 'transformers', transformers.__version__)"
