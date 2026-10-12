<#
.SYNOPSIS
  Reach the avatar from another PC over the internet (HTTPS is required for the microphone).

  Starts the backend with random owner and customer tokens and opens a free Cloudflare "quick tunnel"
  (no account needed) to it. Prints the https://....trycloudflare.com links to open on the other PC.
  Traffic passes through Cloudflare. Quick tunnels are meant for testing; the link changes
  every time. Stop with Ctrl+C, or with -Detach it keeps running in the background until
  scripts\run.ps1 -Stop (which also closes the tunnel). The last links are in logs\tunnel_url.txt
  (owner) and logs\customer_url.txt (customer).

.EXAMPLE
  scripts\tunnel.ps1 -Profile avatar
  scripts\tunnel.ps1 -Profile avatar -Detach
#>
param(
  # $ProfileName, not $Profile (that would hide PowerShell's automatic $PROFILE); -Profile still works
  [Alias('Profile')][string]$ProfileName = 'avatar',
  [int]$Port = 8000,
  [switch]$Detach
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

# cloudflared (Apache-2.0): a pinned release, checked against the SHA-256 GitHub publishes for the file
# (api.github.com/repos/cloudflare/cloudflared/releases/tags/<version>, asset "digest")
$cfVersion = '2026.10.0'
$cfSha256 = '86aee4017b26625cee8484c113558f48effa4cd47f7aa05fcf425604e5d2b23c'
$exe = Join-Path $root "tools\cloudflared\cloudflared-$cfVersion.exe"
if (-not (Test-Path $exe)) {
  New-Item -ItemType Directory -Force (Split-Path $exe) | Out-Null
  $part = "$exe.part"
  Remove-Item $part -Force -ErrorAction SilentlyContinue      # never resume or reuse a partial download
  Write-Host "downloading cloudflared $cfVersion (Apache-2.0)..."
  Invoke-WebRequest "https://github.com/cloudflare/cloudflared/releases/download/$cfVersion/cloudflared-windows-amd64.exe" `
    -OutFile $part -UseBasicParsing
  $got = (Get-FileHash $part -Algorithm SHA256).Hash
  if ($got -ne $cfSha256) { Remove-Item $part -Force; throw "cloudflared download: SHA-256 $got, expected $cfSha256" }
  Move-Item $part $exe
}

function New-Token {
  # 18 bytes from the OS's cryptographic generator (Get-Random is not meant for secrets), URL-safe base64
  $bytes = New-Object byte[] 18
  $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
  try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
  return [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_')
}

$token = New-Token
# the customer link: only the conversation (no settings or leads), each customer with their own memory
$customer = New-Token
# run.ps1 hands the environment to the backend; the caller's session gets its own values back afterwards
$savedTokens = @{ AVATAR_ACCESS_TOKEN = $env:AVATAR_ACCESS_TOKEN; AVATAR_CUSTOMER_TOKEN = $env:AVATAR_CUSTOMER_TOKEN }
$env:AVATAR_ACCESS_TOKEN = $token
$env:AVATAR_CUSTOMER_TOKEN = $customer
try {
  & (Join-Path $root 'scripts\run.ps1') -Profile $ProfileName -Port $Port | Out-Host
} finally {
  foreach ($name in $savedTokens.Keys) { [Environment]::SetEnvironmentVariable($name, $savedTokens[$name], 'Process') }
}

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
# written after run.ps1 wrote logs\backend.pid: the e2e tests take these links only while they are newer
# than the running backend (tests\e2e\helpers.js)
"$url/?token=$token" | Out-File (Join-Path $root 'logs\tunnel_url.txt') -Encoding ascii
"$url/?token=$customer" | Out-File (Join-Path $root 'logs\customer_url.txt') -Encoding ascii
Write-Host ''
Write-Host "Owner link (settings, leads):       $url/?token=$token" -ForegroundColor Green
Write-Host "Customer link (conversation only):  $url/?token=$customer" -ForegroundColor Green
Write-Host '(use headphones; allow the microphone when the browser asks)'
if ($Detach) { Write-Host "tunnel pid $($p.Id) runs in the background; stop everything with scripts\run.ps1 -Stop"; exit 0 }
Write-Host "tunnel pid $($p.Id) - press Ctrl+C to stop"
try { Wait-Process -Id $p.Id } finally { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
