#!/bin/bash
# OpenHydra first-run bootstrap
# Creates a Python venv and installs openhydra + platform-specific deps.

set -euo pipefail

SUPPORT_DIR="${HOME}/Library/Application Support/OpenHydra"
VENV_DIR="${SUPPORT_DIR}/venv"

echo "checking_python"
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 not found. Install from https://python.org" >&2
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python ${PYTHON_VERSION}"

echo "creating_venv"
mkdir -p "${SUPPORT_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "installing_deps"
pip install --upgrade pip --quiet

# Detect platform and install appropriate backend
ARCH=$(uname -m)
OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    echo "installing_mlx"
    pip install openhydra mlx mlx-lm --quiet
elif [ "$OS" = "Linux" ]; then
    # Check for NVIDIA vs AMD
    if command -v nvidia-smi &>/dev/null; then
        echo "installing_cuda"
        pip install openhydra torch --quiet
    elif [ -d "/opt/rocm" ]; then
        echo "installing_rocm"
        pip install openhydra torch --index-url https://download.pytorch.org/whl/rocm6.2 --quiet
    else
        echo "installing_cpu"
        pip install openhydra torch --quiet
    fi
fi

echo "bootstrap_complete"
echo "Ready!"
