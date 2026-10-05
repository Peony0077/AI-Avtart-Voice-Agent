<#
.SYNOPSIS
  Reach the avatar from another PC over the internet (HTTPS is required for the microphone).

  Starts the backend with a random access token and opens a free Cloudflare "quick tunnel"
  (no account needed) to it. Prints the https://....trycloudflare.com link to open on your PC.
  Traffic passes through Cloudflare. Quick tunnels are meant for testing; the link changes
  every time. Stop with Ctrl+C, or with -Detach it keeps running in the background until
  scripts\run.ps1 -Stop (which also closes the tunnel). The last link is in logs\tunnel_url.txt.

.EXAMPLE
  scripts\tunnel.ps1 -Profile avatar
  scripts\tunnel.ps1 -Profile avatar -Detach
#>
param([string]$Profile = 'avatar', [int]$Port = 8000, [switch]$Detach)
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
# the customer link: only the conversation (no settings or leads), each customer with their own memory
$customer = -join ((48..57) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ })
$env:AVATAR_CUSTOMER_TOKEN = $customer
& (Join-Path $root 'scripts\run.ps1') -Profile $Profile -Port $Port | Out-Host
$log = Join-Path $root 'logs\cloudflared.log'
$url = $null
for ($attempt = 1; $attempt -le 3 -and -not $url; $attempt++) {
  $p = Start-Process $exe -ArgumentList @('tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$Port") -PassThru -WindowStyle Hidden -RedirectStandardError $log
  for ($i = 0; $i -lt 60 -and -not $url -and -not $p.HasExited; $i++) {
    Start-Sleep 1
    # not api.trycloudflare.com: that one is in the error when Cloudflare does not answer (2026-10-05 it
    # was handed out as the link)
    $m = Select-String -Path $log -Pattern 'https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($m) { $url = $m.Matches[0].Value }
  }
  if (-not $url) {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Write-Host "quick tunnel attempt $attempt failed (Cloudflare did not answer), see $log"
    Start-Sleep 5
  }
}
if (-not $url) { throw "tunnel did not start after 3 attempts; see $log" }
"$url/?token=$token" | Out-File (Join-Path $root 'logs\tunnel_url.txt') -Encoding ascii
"$url/?token=$customer" | Out-File (Join-Path $root 'logs\customer_url.txt') -Encoding ascii
Write-Host ''
Write-Host "Owner link (settings, leads):       $url/?token=$token" -ForegroundColor Green
Write-Host "Customer link (conversation only):  $url/?token=$customer" -ForegroundColor Green
Write-Host '(use headphones; allow the microphone when the browser asks)'
if ($Detach) { Write-Host "tunnel pid $($p.Id) runs in the background; stop everything with scripts\run.ps1 -Stop"; exit 0 }
Write-Host "tunnel pid $($p.Id) - press Ctrl+C to stop"
try { Wait-Process -Id $p.Id } finally { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
