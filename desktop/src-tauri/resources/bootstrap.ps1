# OpenHydra first-run bootstrap (Windows)
# Creates a Python venv and installs openhydra + platform-specific deps.

$ErrorActionPreference = "Stop"

$SupportDir = "$env:APPDATA\OpenHydra"
$VenvDir = "$SupportDir\venv"

Write-Output "checking_python"
try {
    $pythonVersion = python --version 2>&1
    Write-Output "Found $pythonVersion"
} catch {
    Write-Error "Python 3 not found. Install from https://python.org"
    exit 1
}

Write-Output "creating_venv"
New-Item -ItemType Directory -Force -Path $SupportDir | Out-Null
python -m venv $VenvDir
& "$VenvDir\Scripts\Activate.ps1"

Write-Output "installing_deps"
pip install --upgrade pip --quiet

# Check for NVIDIA GPU
$hasNvidia = $false
try {
    nvidia-smi | Out-Null
    $hasNvidia = $true
} catch {}

if ($hasNvidia) {
    Write-Output "installing_cuda"
    pip install openhydra-network torch --quiet
} else {
    Write-Output "installing_cpu"
    pip install openhydra-network torch --quiet
}

Write-Output "bootstrap_complete"
Write-Output "Ready!"
