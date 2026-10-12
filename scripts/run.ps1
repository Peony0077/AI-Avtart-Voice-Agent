<#
.SYNOPSIS
  Start / stop the backend (which starts llama-server and, if enabled, the avatar worker).

.EXAMPLE
  scripts\run.ps1                    # start with the profile from .env (AVATAR_PROFILE), else default
  scripts\run.ps1 -Profile avatar    # start with config/profiles/avatar.yaml
  scripts\run.ps1 -Build             # rebuild the frontend first
  scripts\run.ps1 -Https             # LAN: self-signed TLS on all interfaces, owner + customer tokens
  scripts\run.ps1 -Stop              # stop everything of this repo (backend, helpers, quick tunnel)
#>
param(
  # $ProfileName, not $Profile: a parameter named Profile would hide PowerShell's automatic $PROFILE
  # inside the script. -Profile still works (alias).
  [Alias('Profile')][string]$ProfileName = '',
  [int]$Port = 8000,
  [string]$BindHost = '127.0.0.1',
  [switch]$Build,
  [switch]$Https,     # LAN access: self-signed TLS, listen on all interfaces, random access tokens
  [switch]$Stop,
  [switch]$Foreground
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root 'logs\backend.pid'
$py = Join-Path $root '.venv\Scripts\python.exe'

function Test-Contains([string]$text, [string]$part) {
  # a plain substring test: -like would read [ ] in a folder name as a wildcard pattern
  if (-not $text) { return $false }
  return $text.IndexOf($part, [StringComparison]::OrdinalIgnoreCase) -ge 0
}

function New-Token {
  # 18 bytes from the OS's cryptographic generator (Get-Random is not meant for secrets), URL-safe base64
  $bytes = New-Object byte[] 18
  $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
  try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
  return [Convert]::ToBase64String($bytes).Replace('+', '-').Replace('/', '_')
}

function Stop-All {
  # Only processes of THIS repo: other llama-server / uvicorn / cloudflared processes on the PC are not ours.
  $all = @(Get-CimInstance Win32_Process)
  $kill = New-Object System.Collections.Generic.List[int]
  if (Test-Path $pidFile) {
    $backend = 0
    [void][int]::TryParse("$(Get-Content $pidFile -TotalCount 1)".Trim(), [ref]$backend)
    # the recorded pid only if it still is our backend (Windows reuses pids)
    $p = $all | Where-Object { $_.ProcessId -eq $backend -and (Test-Contains $_.CommandLine 'uvicorn app.main') }
    if ($p) {
      # with its whole process tree: the venv's python.exe is a launcher, the real interpreter its child
      $todo = @($backend)
      while ($todo.Count) {
        $kill.AddRange([int[]]$todo)
        $todo = @($all | Where-Object { ($todo -contains $_.ParentProcessId) -and -not $kill.Contains([int]$_.ProcessId) } |
                  ForEach-Object { [int]$_.ProcessId })
      }
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
  }
  # helpers left behind by a backend that was killed, matched by this repo's paths
  $llama = Join-Path $root 'tools\llama.cpp\'
  $workers = @((Join-Path $root 'avatar\musetalk\worker.py'), (Join-Path $root 'voice\xtts\worker.py'))
  $backendDir = Join-Path $root 'backend'
  $cfDir = Join-Path $root 'tools\cloudflared\'
  foreach ($p in $all) {
    $ours = $false
    if ($p.Name -eq 'llama-server.exe') {
      $ours = (Test-Contains $p.ExecutablePath $llama) -or (Test-Contains $p.CommandLine $llama)
    } elseif ($p.Name -eq 'python.exe') {
      $ours = (Test-Contains $p.CommandLine $workers[0]) -or (Test-Contains $p.CommandLine $workers[1]) -or
              ((Test-Contains $p.CommandLine 'uvicorn app.main') -and (Test-Contains $p.CommandLine $backendDir))
    } elseif ($p.Name -like 'cloudflared*.exe') {
      # our quick tunnel (scripts\tunnel.ps1): its link and token would point at the stopped backend
      $ours = (Test-Contains $p.ExecutablePath $cfDir) -or (Test-Contains $p.CommandLine $cfDir)
    }
    if ($ours -and -not $kill.Contains([int]$p.ProcessId)) { $kill.Add([int]$p.ProcessId) }
  }
  # (llama-server may exit on its own once the backend is gone: no error for that)
  foreach ($id in $kill) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
}

if ($Stop) { Stop-All; Write-Host 'stopped'; exit 0 }

# Environment for the backend process only. Start-Process hands the current environment to the child;
# everything set here is put back afterwards, so the caller's PowerShell session keeps its own values
# (a leftover AVATAR_PROFILE or HF_HUB_OFFLINE there silently changed later runs and other tools).
$saved = @{}
function Set-ChildEnv([string]$name, [string]$value) {
  if (-not $saved.ContainsKey($name)) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process') }
  [Environment]::SetEnvironmentVariable($name, $value, 'Process')
}

function Start-Backend {
  New-Item -ItemType Directory -Force logs | Out-Null
  Stop-All
  # .env (the file the backend reads too): its values unless the environment already has them, so that
  # HF_HOME / AVATAR_MODEL_HOME reach the backend's helpers and workers as well
  if (Test-Path .env) {
    foreach ($line in Get-Content .env) {
      if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
        $name = $Matches[1]; $value = $Matches[2].Trim('"').Trim("'")
        if (-not [Environment]::GetEnvironmentVariable($name, 'Process')) { Set-ChildEnv $name $value }
      }
    }
  }
  if ($ProfileName) { Set-ChildEnv 'AVATAR_PROFILE' $ProfileName }
  if (-not $env:AVATAR_MODEL_HOME) { Set-ChildEnv 'AVATAR_MODEL_HOME' 'E:\AI-models' }
  if (-not $env:HF_HOME) { Set-ChildEnv 'HF_HOME' (Join-Path $env:AVATAR_MODEL_HOME 'hf') }
  Set-ChildEnv 'CUDA_MODULE_LOADING' 'LAZY'
  Set-ChildEnv 'HF_HUB_OFFLINE' '1'

  if ($Build -or -not (Test-Path frontend\dist\index.html)) {
    Push-Location frontend
    try {
      if (-not (Test-Path node_modules)) {
        npm.cmd ci --no-audit --no-fund | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed (frontend)' }
      }
      npm.cmd run build | Out-Host
      # never start with a stale or missing UI after a failed build
      if ($LASTEXITCODE -ne 0) { throw 'frontend build failed (npm run build)' }
    } finally { Pop-Location }
  }

  $loopback = $BindHost -in @('127.0.0.1', 'localhost', '::1')
  $listen = if ($Https) { '0.0.0.0' } else { $BindHost }
  if ($Https -or -not $loopback) {
    # reachable from other machines: never without tokens (with no access token everyone is the owner)
    if (-not $env:AVATAR_ACCESS_TOKEN) { Set-ChildEnv 'AVATAR_ACCESS_TOKEN' (New-Token) }
    if (-not $env:AVATAR_CUSTOMER_TOKEN) { Set-ChildEnv 'AVATAR_CUSTOMER_TOKEN' (New-Token) }
  }
  $owner = $env:AVATAR_ACCESS_TOKEN
  $customer = $env:AVATAR_CUSTOMER_TOKEN

  # absolute --app-dir: Stop-All recognises this repo's backend by it. --no-access-log: the access log
  # would write every ?token=... into logs\uvicorn.err.
  $uvArgs = @('-m', 'uvicorn', 'app.main:app', '--app-dir', (Join-Path $root 'backend'), '--host', $listen,
              '--port', "$Port", '--ws-max-size', '16777216', '--no-access-log')
  $scheme = 'http'
  if ($Https) {
    if (-not (Test-Path certs\avatar.pem)) {
      & $py scripts\make_cert.py --port $Port | Out-Host
      if ($LASTEXITCODE -ne 0) { throw 'make_cert.py failed' }
    }
    $uvArgs += @('--ssl-certfile', 'certs\avatar.pem', '--ssl-keyfile', 'certs\avatar-key.pem')
    $scheme = 'https'
  }
  if ($Foreground) { & $py @uvArgs | Out-Host; $script:exitCode = $LASTEXITCODE; return }

  # Start-Process joins its arguments with spaces: quote the ones that contain spaces (the repo path may)
  $argLine = ($uvArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
  $p = Start-Process -FilePath $py -ArgumentList $argLine -WorkingDirectory $root -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput logs\uvicorn.out -RedirectStandardError logs\uvicorn.err
  $p.Id | Out-File $pidFile -Encoding ascii
  Write-Host "backend pid $($p.Id); waiting for health..."
  # the health check goes to the address the backend listens on (127.0.0.1 when it listens on all of them)
  $probe = if ($listen -in @('0.0.0.0', '::')) { '127.0.0.1' } elseif ($listen -eq '::1') { '[::1]' } else { $listen }
  $clock = [Diagnostics.Stopwatch]::StartNew()     # seconds, not loop rounds: one round can take 3 s
  $h = $null
  while (-not $h -and $clock.Elapsed.TotalSeconds -lt 300) {
    Start-Sleep 1
    if ($p.HasExited) { Get-Content logs\uvicorn.err -Tail 40 | Out-Host; throw 'backend exited during startup' }
    try {
      if ($Https) {
        # Windows PowerShell 5.1 cannot skip the check of the self-signed certificate: ask with Python
        $text = & $py -c "import httpx; r = httpx.get('https://${probe}:$Port/health', verify=False, timeout=2); print(r.text if r.status_code == 200 else '')" 2>$null
        if ($text) { $h = $text | ConvertFrom-Json }
      } else {
        $h = Invoke-RestMethod "http://${probe}:$Port/health" -TimeoutSec 2
      }
    } catch { $h = $null }
  }
  if (-not $h) { throw 'backend did not become healthy in 300 s (see logs\uvicorn.err)' }
  $h | ConvertTo-Json -Compress | Out-Host

  if ($owner) {
    $hosts = if ($listen -in @('0.0.0.0', '::')) {
      @(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127|169\.254)\.' } | ForEach-Object { $_.IPAddress })
    } else { @($listen) }
    if ($Https) { Write-Host 'LAN access (accept the certificate warning once):' -ForegroundColor Green }
    foreach ($ip in $hosts) {
      Write-Host "  owner:    ${scheme}://${ip}:$Port/?token=$owner"
      Write-Host "  customer: ${scheme}://${ip}:$Port/?token=$customer"
    }
    if (-not $loopback -or $Https) {
      if (-not $Https) { Write-Host '(plain http: other devices get text only - browsers allow the microphone on https; use -Https)' }
      Write-Host 'If other PCs cannot connect, allow the port in Windows Firewall for private networks (admin):'
      Write-Host "  New-NetFirewallRule -DisplayName 'AI Avatar' -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Private"
    }
  } else { Write-Host "open http://127.0.0.1:$Port/" }
}

$script:exitCode = 0
Push-Location $root       # (Set-Location would leave the caller's session in the repo folder)
try {
  Start-Backend
} finally {
  foreach ($name in $saved.Keys) { [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process') }
  Pop-Location
}
exit $script:exitCode
