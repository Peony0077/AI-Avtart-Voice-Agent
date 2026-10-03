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
  [switch]$Https,     # LAN access: self-signed TLS, listen on all interfaces, random access token
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
  # our quick tunnel (scripts\tunnel.ps1): its link and token would point at the stopped backend
  $cf = Join-Path $root 'tools\cloudflared\cloudflared.exe'
  Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" | Where-Object { $_.ExecutablePath -eq $cf } |
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

$scheme = 'http'
$uvArgs = @('-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', $BindHost, '--port', "$Port", '--ws-max-size', '16777216')
if ($Https) {
  if (-not (Test-Path certsvatar.pem)) { & .\.venv\Scripts\python.exe scripts\make_cert.py | Out-Host }
  if (-not $env:AVATAR_ACCESS_TOKEN) { $env:AVATAR_ACCESS_TOKEN = -join ((48..57) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ }) }
  $uvArgs = @('-m', 'uvicorn', 'app.main:app', '--app-dir', 'backend', '--host', '0.0.0.0', '--port', "$Port", '--ws-max-size', '16777216',
              '--ssl-certfile', 'certsvatar.pem', '--ssl-keyfile', 'certsvatar-key.pem')
  $scheme = 'https'
}
if ($Foreground) { & .\.venv\Scripts\python.exe @uvArgs; exit $LASTEXITCODE }

$p = Start-Process -FilePath .\.venv\Scripts\python.exe -ArgumentList $uvArgs -PassThru -WindowStyle Hidden `
  -RedirectStandardOutput logs\uvicorn.out -RedirectStandardError logs\uvicorn.err
$p.Id | Out-File $pidFile
Write-Host "backend pid $($p.Id); waiting for health..."
for ($i = 0; $i -lt 300; $i++) {
  Start-Sleep 1
  if ($p.HasExited) { Get-Content logs\uvicorn.err -Tail 40; throw 'backend exited during startup' }
  try { $h = Invoke-RestMethod "${scheme}://127.0.0.1:$Port/health" -TimeoutSec 2; break } catch {
    try { if ($Https) { $h = (& .\.venv\Scripts\python.exe -c "import httpx,sys; print(httpx.get('https://127.0.0.1:$Port/health', verify=False).text)") | ConvertFrom-Json; if ($h) { break } } } catch {} }
}
if (-not $h) { throw 'backend did not become healthy in 300 s' }
$h | ConvertTo-Json -Compress
if ($Https) {
  Write-Host "LAN access (accept the certificate warning once):" -ForegroundColor Green
  Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127|169\.254)\.' } | ForEach-Object { Write-Host "  https://$($_.IPAddress):$Port/?token=$env:AVATAR_ACCESS_TOKEN" }
  Write-Host "If other PCs cannot connect, allow the port in Windows Firewall (admin):"
  Write-Host "  New-NetFirewallRule -DisplayName 'AI Avatar' -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow"
} else { Write-Host "open http://127.0.0.1:$Port/" }
