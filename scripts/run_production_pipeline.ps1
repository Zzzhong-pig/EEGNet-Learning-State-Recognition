param(
    [string]$EEGNetConfig = "configs/eegnet.yaml",
    [string]$EEGNetOutput = "artifacts/production/eegnet",
    [string]$EEGNetEnsembleOutput = "artifacts/production/eegnet_ensemble",
    [string]$FBCSPConfig = "configs/fbcsp_auxiliary.yaml",
    [string]$FBCSPOutput = "artifacts/production/fbcsp",
    [string]$HybridOutput = "artifacts/production",
    [int]$Estimators = 600
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$env:PYTHONPATH = $projectRoot
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    $pythonCommand = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $pythonCommand) {
    throw "Python executable not found. Install Python 3.11+ and add python or py to PATH."
}
$pythonExe = $pythonCommand.Source

Write-Host "==> Preprocess EEG for the EEGNet training path"
& $pythonExe preprocess.py --config $EEGNetConfig

Write-Host "==> Train and validate EEGNet"
& $pythonExe train.py --config $EEGNetConfig --output $EEGNetOutput

Write-Host "==> Build the leakage-audited EEGNet ensemble manifest"
& $pythonExe scripts/build_mixed_ensemble.py --config $EEGNetConfig --artifact-dirs $EEGNetOutput --output $EEGNetEnsembleOutput --method mean --target accuracy

Write-Host "==> Train and validate the FBCSP auxiliary model"
& $pythonExe scripts/train_fbcsp.py --config $FBCSPConfig --output $FBCSPOutput --estimators $Estimators

Write-Host "==> Fit and package the EEGNet + FBCSP fusion policy"
& $pythonExe scripts/build_eegnet_fbcsp_hybrid.py --eegnet-artifact $EEGNetOutput --eegnet-manifest "$EEGNetEnsembleOutput/manifest.json" --fbcsp-artifact $FBCSPOutput --fbcsp-manifest "$FBCSPOutput/manifest.json" --output $HybridOutput

Write-Host "==> Done. Deploy with $HybridOutput/manifest.json"
