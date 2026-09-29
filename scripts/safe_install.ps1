<#
.SYNOPSIS
  Install packages into the backend venv without ever replacing CUDA PyTorch.

.DESCRIPTION
  1. Verifies the environment (torch 2.9.1+cu130, CUDA available).
  2. Installs the given packages with constraints/backend.txt.
  3. Verifies again; on failure prints the recovery command.
  4. Refreshes constraints/backend.lock.txt (pip freeze snapshot).

.EXAMPLE
  scripts\safe_install.ps1 fastapi "uvicorn[standard]"
  scripts\safe_install.ps1 -r backend\requirements.txt
#>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Packages)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'
$constraints = Join-Path $root 'constraints\backend.txt'
$check = Join-Path $root 'scripts\check_env.py'

if (-not $Packages) { throw 'No packages given.' }

Write-Host '== Pre-install check ==' -ForegroundColor Cyan
& $py $check
if ($LASTEXITCODE -ne 0) { throw 'Environment already broken before install - aborting.' }

Write-Host "== Installing: $Packages ==" -ForegroundColor Cyan
& uv pip install --python $py -c $constraints @Packages
if ($LASTEXITCODE -ne 0) { throw 'uv pip install failed.' }

Write-Host '== Post-install check ==' -ForegroundColor Cyan
& $py $check
if ($LASTEXITCODE -ne 0) {
  Write-Host 'CUDA PyTorch changed! Recover with:' -ForegroundColor Red
  Write-Host "uv pip install --python $py torch==2.9.1+cu130 torchaudio==2.9.1+cu130 torchvision==0.24.1+cu130 --index-url https://download.pytorch.org/whl/cu130"
  exit 1
}

& uv pip freeze --python $py | Out-File -Encoding utf8 (Join-Path $root 'constraints\backend.lock.txt')
Write-Host 'OK - lock snapshot refreshed: constraints\backend.lock.txt' -ForegroundColor Green
