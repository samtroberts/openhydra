#!/usr/bin/env bash
# Setup a Linode Nanode (1GB) as an OpenHydra peer node.
# Usage: ssh root@<ip> 'bash -s' < setup_nanode_peer.sh
set -euo pipefail

echo "=== OpenHydra Peer Node Setup ==="

# System packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential python3-dev python3-venv python3-pip git libssl-dev

# Create openhydra user
id -u openhydra &>/dev/null || useradd -r -m -d /opt/openhydra -s /bin/bash openhydra

# Setup Python venv
VENV=/opt/openhydra/.venv
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi

# Install Python deps (CPU-only PyTorch to save RAM)
"$VENV/bin/pip" install --upgrade pip -q
"$VENV/bin/pip" install -q \
    torch --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install -q \
    transformers safetensors sentencepiece \
    grpcio grpcio-tools protobuf \
    requests cryptography

# Download Qwen2.5-0.5B model
echo "=== Downloading Qwen2.5-0.5B ==="
"$VENV/bin/python3" -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B', local_dir='/opt/openhydra/.cache/models/Qwen2.5-0.5B')
print('Model downloaded.')
"

# Set memory limits
echo "MALLOC_ARENA_MAX=2" >> /etc/environment

# Open gRPC port
ufw allow 50051/tcp 2>/dev/null || true

# Set file descriptor limits
cat >> /etc/security/limits.conf <<'EOF'
openhydra soft nofile 65535
openhydra hard nofile 65535
EOF

echo "=== Setup complete ==="
echo "Next: rsync the OpenHydra codebase and create systemd service"
