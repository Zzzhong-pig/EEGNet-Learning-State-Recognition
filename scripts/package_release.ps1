param(
    [string]$Output = "dist/EEG_Project_release.zip",
    [switch]$IncludeData
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$staging = Join-Path $root "dist/_package_staging/EEG_Project"
$excludeDirs = @(".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist", ".vscode")
$excludeFiles = @("*.pyc", "*.pyo", "*.log", ".env")

if (Test-Path $staging) {
    Remove-Item $staging -Recurse -Force
}
New-Item -ItemType Directory -Path $staging -Force | Out-Null

Write-Host "==> Staging project files"
robocopy $root $staging /E /NFL /NDL /NJH /NJS /NC /NS /NP `
    /XD $excludeDirs `
    /XF $excludeFiles | Out-Null
if ($LASTEXITCODE -ge 8) {
    throw "robocopy failed with exit code $LASTEXITCODE"
}

if (-not $IncludeData) {
    Write-Host "==> Removing training data (use -IncludeData to keep)"
    Get-ChildItem (Join-Path $staging "data") -Filter "*.npy" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force
}

$outputPath = Join-Path $root $Output
$outputDir = Split-Path $outputPath -Parent
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}
if (Test-Path $outputPath) {
    Remove-Item $outputPath -Force
}

Write-Host "==> Creating archive: $outputPath"
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $outputPath -Force
Remove-Item (Split-Path $staging -Parent) -Recurse -Force

$sizeMb = [math]::Round((Get-Item $outputPath).Length / 1MB, 1)
Write-Host "Done. Package size: ${sizeMb} MB"
