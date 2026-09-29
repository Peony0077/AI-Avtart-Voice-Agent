<#
.SYNOPSIS
  Reach the avatar from another PC over the internet (HTTPS is required for the microphone).

  Starts the backend with a random access token and opens a free Cloudflare "quick tunnel"
  (no account needed) to it. Prints the https://....trycloudflare.com link to open on your PC.
  Traffic passes through Cloudflare. Quick tunnels are meant for testing; the link changes
  every time. Stop with Ctrl+C (then scripts\run.ps1 -Stop).

.EXAMPLE
  scripts\tunnel.ps1 -Profile avatar
#>
param([string]$Profile = 'avatar', [int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'tools\cloudflared\cloudflared.exe'
if (-not (Test-Path $exe)) {
  New-Item -ItemType Directory -Force (Split-Path $exe) | Out-Null
  Write-Host 'downloading cloudflared (Apache-2.0)...'
  Invoke-WebRequest 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile $exe -UseBasicParsing
}
$token = -join ((48..57) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ })
$env:AVATAR_ACCESS_TOKEN = $token
& (Join-Path $root 'scripts\run.ps1') -Profile $Profile -Port $Port | Out-Host
$log = Join-Path $root 'logs\cloudflared.log'
$p = Start-Process $exe -ArgumentList @('tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$Port") -PassThru -WindowStyle Hidden -RedirectStandardError $log
$url = $null
for ($i = 0; $i -lt 60 -and -not $url; $i++) {
  Start-Sleep 1
  $m = Select-String -Path $log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($m) { $url = $m.Matches[0].Value }
}
if (-not $url) { Stop-Process -Id $p.Id -Force; throw "tunnel did not start; see $log" }
"$url/?token=$token" | Out-File (Join-Path $root 'logs\tunnel_url.txt') -Encoding ascii
Write-Host ''
Write-Host "Open on your PC:  $url/?token=$token" -ForegroundColor Green
Write-Host '(use headphones; allow the microphone when the browser asks)'
Write-Host "tunnel pid $($p.Id) - press Ctrl+C to stop"
try { Wait-Process -Id $p.Id } finally { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
