<#
.SYNOPSIS
  Create the isolated clone-voice TTS environment (voice/xtts/.venv) and download XTTS-v2
  weights to E:\AI-models\tts\XTTS-v2. Never touches the backend .venv.
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
$env:HF_HUB_OFFLINE = '0'
& $py -c "from huggingface_hub import snapshot_download; p=snapshot_download('coqui/XTTS-v2', local_dir='E:/AI-models/tts/XTTS-v2', allow_patterns=['config.json','model.pth','vocab.json','speakers_xtts.pth','dvae.pth','mel_stats.pth']); print(p)"
& $py -c "import torch, TTS, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'coqui-tts', TTS.__version__, 'transformers', transformers.__version__)"
