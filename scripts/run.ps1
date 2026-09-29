<#
.SYNOPSIS
  Start / stop the backend (which starts llama-server and, if enabled, the avatar worker).

.EXAMPLE
  scripts\run.ps1                    # start with profile from .env (default)
  scripts\run.ps1 -Profile avatar    # start with config/profiles/avatar.yaml
  scripts\run.ps1 -Build             # rebuild the frontend first
  scripts\run.ps1 -Stop              # stop everything
#>
param(
  [string]$Profile = '',
  [int]$Port = 8000,
  [string]$BindHost = '127.0.0.1',
  [switch]$Build,
  [switch]$Stop,
  [switch]$Foreground
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force logs | Out-Null
$pidFile = Join-Path $root 'logs\backend.pid'

function Stop-All {
  if (Test-Path $pidFile) {
    $procId = Get-Content $pidFile
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    Remove-Item $pidFile -ErrorAction SilentlyContinue
  }
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'avatar\\musetalk\\worker\.py|voice\\xtts\\worker\.py|uvicorn app\.main' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

if ($Stop) { Stop-All; Write-Host 'stopped'; exit 0 }
Stop-All
if ($Profile) { $env:AVATAR_PROFILE = $Profile }
$env:HF_HOME = 'E:\AI-models\hf'
$env:CUDA_MODULE_LOADING = 'LAZY'
$env:HF_HUB_OFFLINE = '1'

if ($Build -or -not (Test-Path frontend\dist\index.html)) {
  Push-Location frontend; npm.cmd run build | Out-Host; Pop-Location
}

$uvArgs = @('-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', $BindHost, '--port', "$Port", '--ws-max-size', '16777216')
if ($Foreground) { & .\.venv\Scripts\python.exe @uvArgs; exit $LASTEXITCODE }

$p = Start-Process -FilePath .\.venv\Scripts\python.exe -ArgumentList $uvArgs -PassThru -WindowStyle Hidden `
  -RedirectStandardOutput logs\uvicorn.out -RedirectStandardError logs\uvicorn.err
$p.Id | Out-File $pidFile
Write-Host "backend pid $($p.Id); waiting for health..."
for ($i = 0; $i -lt 300; $i++) {
  Start-Sleep 1
  if ($p.HasExited) { Get-Content logs\uvicorn.err -Tail 40; throw 'backend exited during startup' }
  try { $h = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2; break } catch {}
}
if (-not $h) { throw 'backend did not become healthy in 300 s' }
$h | ConvertTo-Json -Compress
Write-Host "open http://127.0.0.1:$Port/"
