<#
.SYNOPSIS
  Fetch the prebuilt llama.cpp server for Windows + CUDA (MIT) into tools\llama.cpp (git-ignored).

  Uses the release build tested in this project (b11242, CUDA 12.4 runtime bundled: it does not
  touch the system CUDA install or PyTorch). tools\llama.cpp\current is a junction to the build,
  which is what config llm.server_exe points to. Every download is checked against the SHA-256 that
  GitHub publishes for the release file. (Linux: scripts/setup_linux.sh.)

.EXAMPLE
  scripts\get_llama_cpp.ps1
#>
param([string]$Build = 'b11242', [string]$Cuda = '12.4')
$ErrorActionPreference = 'Stop'
$root = Join-Path (Split-Path -Parent $PSScriptRoot) 'tools\llama.cpp'
$target = Join-Path $root "$Build-cuda$Cuda"
$dl = Join-Path $root 'dl'
New-Item -ItemType Directory -Force $dl | Out-Null

# SHA-256 of the release files (api.github.com/repos/ggml-org/llama.cpp/releases/tags/<build>, asset
# "digest"). Another build or CUDA version: add its two files here first.
$sha256 = @{
  'b11242/llama-b11242-bin-win-cuda-12.4-x64.zip' = '3580a58446efa1c49ec0efb128399a923e87d2db7a3cfb17eb36d3ae18f1b1f0'
  'b11242/cudart-llama-bin-win-cuda-12.4-x64.zip' = '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'
}

if (-not (Test-Path (Join-Path $target 'llama-server.exe'))) {
  $base = "https://github.com/ggml-org/llama.cpp/releases/download/$Build"
  foreach ($zip in @("llama-$Build-bin-win-cuda-$Cuda-x64.zip", "cudart-llama-bin-win-cuda-$Cuda-x64.zip")) {
    $want = $sha256["$Build/$zip"]
    if (-not $want) { throw "no pinned SHA-256 for ${Build}/${zip}: add it to `$sha256 in this script" }
    $file = Join-Path $dl $zip
    # a zip from an earlier, interrupted run (or a different file under the same name) is never reused
    if ((Test-Path $file) -and (Get-FileHash $file -Algorithm SHA256).Hash -ne $want) { Remove-Item $file -Force }
    if (-not (Test-Path $file)) {
      $part = "$file.part"
      Remove-Item $part -Force -ErrorAction SilentlyContinue
      Write-Host "downloading $zip"
      Invoke-WebRequest "$base/$zip" -OutFile $part -UseBasicParsing
      $got = (Get-FileHash $part -Algorithm SHA256).Hash
      if ($got -ne $want) { Remove-Item $part -Force; throw "${zip}: SHA-256 $got, expected $want" }
      Move-Item $part $file
    }
    Expand-Archive $file -DestinationPath $target -Force
  }
}

$current = Join-Path $root 'current'
if (Test-Path $current) {
  cmd /c rmdir "$current" | Out-Null     # removes only the junction, never the build it points to
}
New-Item -ItemType Junction -Path $current -Target $target | Out-Null
$ErrorActionPreference = 'Continue'     # llama-server prints its version on stderr
& (Join-Path $current 'llama-server.exe') --version 2>&1 | ForEach-Object { "$_" } | Select-String 'version'
Write-Host "llama-server ready: $current"
