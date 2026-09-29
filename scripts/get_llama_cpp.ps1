<#
.SYNOPSIS
  Fetch the prebuilt llama.cpp server for Windows + CUDA (MIT) into tools\llama.cpp (git-ignored).

  Uses the release build tested in this project (b11242, CUDA 12.4 runtime bundled: it does not
  touch the system CUDA install or PyTorch). tools\llama.cpp\current is a junction to the build,
  which is what config llm.server_exe points to.

.EXAMPLE
  scripts\get_llama_cpp.ps1
#>
param([string]$Build = 'b11242', [string]$Cuda = '12.4')
$ErrorActionPreference = 'Stop'
$root = Join-Path (Split-Path -Parent $PSScriptRoot) 'tools\llama.cpp'
$target = Join-Path $root "$Build-cuda$Cuda"
$dl = Join-Path $root 'dl'
New-Item -ItemType Directory -Force $dl | Out-Null

if (-not (Test-Path (Join-Path $target 'llama-server.exe'))) {
  $base = "https://github.com/ggml-org/llama.cpp/releases/download/$Build"
  foreach ($zip in @("llama-$Build-bin-win-cuda-$Cuda-x64.zip", "cudart-llama-bin-win-cuda-$Cuda-x64.zip")) {
    $file = Join-Path $dl $zip
    if (-not (Test-Path $file)) {
      Write-Host "downloading $zip"
      Invoke-WebRequest "$base/$zip" -OutFile $file -UseBasicParsing
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
