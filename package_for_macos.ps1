param(
  [string]$OutDir = "artifacts"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$stamp = Get-Date -Format "yyyy-MM-dd"
$outDirPath = Join-Path $root $OutDir
New-Item -ItemType Directory -Force -Path $outDirPath | Out-Null

$zipName = "miriam_macos_transfer_$stamp.zip"
$zipPath = Join-Path $outDirPath $zipName

if (Test-Path $zipPath) {
  Remove-Item -Force $zipPath
}

$stage = Join-Path $outDirPath "macos_stage"
if (Test-Path $stage) {
  Remove-Item -Recurse -Force $stage
}
New-Item -ItemType Directory -Force -Path $stage | Out-Null

# Copy only what the Mac needs to build.
$include = @(
  "panacea_desktop",
  "run_app.py",
  "panacea_desktop.spec",
  "requirements.txt",
  "requirements-lite.txt",
  "README.md",
  "MACOS_SETUP.md",
  "build_macos_lite.sh",
  "build_macos_full.sh"
)

foreach ($item in $include) {
  $src = Join-Path $root $item
  if (!(Test-Path $src)) {
    throw "Missing required item: $item"
  }
  Copy-Item -Recurse -Force -Path $src -Destination (Join-Path $stage $item)
}

# Defensive cleanup (if any caches exist inside panacea_desktop).
Get-ChildItem -Recurse -Force -Path $stage |
  Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".mypy_cache") } |
  ForEach-Object { Remove-Item -Recurse -Force $_.FullName }

Get-ChildItem -Recurse -Force -File -Path $stage -Include *.pyc,*.pyo |
  ForEach-Object { Remove-Item -Force $_.FullName }

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -Force
Remove-Item -Recurse -Force $stage

Write-Host "Created: $zipPath"

