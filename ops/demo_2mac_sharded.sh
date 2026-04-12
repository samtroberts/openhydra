#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# OpenHydra 2-Mac Sharded Inference Demo
#
# Runs Qwen3.5-2B (24 layers) sharded across two MacBooks on the same WiFi.
# Each Mac loads 12 layers on its Metal GPU (MPS) via PyTorch.
#
# Usage:
#   Mac A:  ./ops/demo_2mac_sharded.sh mac-a <FRIENDS_IP>
#   Mac B:  ./ops/demo_2mac_sharded.sh mac-b <YOUR_IP>
#
# Each Mac auto-detects its own LAN IP.
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROLE="${1:-}"
OTHER_IP="${2:-}"

MODEL_ID="openhydra-qwen3.5-2b"
HF_MODEL="Qwen/Qwen3.5-2B"
TOTAL_LAYERS=24
TOTAL_SHARDS=2
GRPC_PORT=50051
API_PORT=8080

if [[ -z "$ROLE" || -z "$OTHER_IP" ]]; then
    echo "Usage: $0 <mac-a|mac-b> <OTHER_MAC_IP>"
    echo ""
    echo "  mac-a  →  layers 0-11  (shard 0)"
    echo "  mac-b  →  layers 12-23 (shard 1)"
    echo ""
    echo "Your IP is auto-detected. Just pass the OTHER Mac's IP."
    echo "  Example: $0 mac-a 192.168.1.20"
    exit 1
fi

if [[ "$ROLE" == "mac-a" ]]; then
    SHARD_INDEX=0
    LAYER_START=0
    LAYER_END=12
elif [[ "$ROLE" == "mac-b" ]]; then
    SHARD_INDEX=1
    LAYER_START=12
    LAYER_END=24
else
    echo "Unknown role: $ROLE (must be 'mac-a' or 'mac-b')"
    exit 1
fi

# Auto-detect LAN IP
MY_IP=$(python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(('8.8.8.8', 80))
    print(s.getsockname()[0])
finally:
    s.close()
" 2>/dev/null || echo "127.0.0.1")

# Assign IPs based on role
if [[ "$ROLE" == "mac-a" ]]; then
    MAC_A_IP="$MY_IP"
    MAC_B_IP="$OTHER_IP"
else
    MAC_A_IP="$OTHER_IP"
    MAC_B_IP="$MY_IP"
fi

echo "================================================================"
echo "  OpenHydra 2-Mac Sharded Demo"
echo "  Model:  $HF_MODEL ($TOTAL_LAYERS layers)"
echo "  Role:   $ROLE (shard $SHARD_INDEX, layers $LAYER_START-$((LAYER_END - 1)))"
echo "  My IP:  $MY_IP (auto-detected)"
echo "  Other:  $OTHER_IP"
echo "================================================================"

# Create peers config with both Macs
PEERS_FILE="/tmp/openhydra_2mac_peers.json"
cat > "$PEERS_FILE" <<PEERSEOF
[
  {
    "peer_id": "mac-a-peer",
    "host": "$MAC_A_IP",
    "port": $GRPC_PORT,
    "model_id": "$MODEL_ID",
    "operator_id": "demo-a",
    "runtime_backend": "pytorch_auto",
    "runtime_model_id": "$HF_MODEL",
    "layer_start": 0,
    "layer_end": 12,
    "total_layers": $TOTAL_LAYERS
  },
  {
    "peer_id": "mac-b-peer",
    "host": "$MAC_B_IP",
    "port": $GRPC_PORT,
    "model_id": "$MODEL_ID",
    "operator_id": "demo-b",
    "runtime_backend": "pytorch_auto",
    "runtime_model_id": "$HF_MODEL",
    "layer_start": 12,
    "layer_end": $TOTAL_LAYERS,
    "total_layers": $TOTAL_LAYERS
  }
]
PEERSEOF

echo ""
echo "Starting peer..."
echo ""

python3 -m coordinator.node \
    --peer-id "$ROLE-peer" \
    --model-id "$MODEL_ID" \
    --runtime-model-id "$HF_MODEL" \
    --runtime-backend pytorch_auto \
    --layer-start "$LAYER_START" \
    --layer-end "$LAYER_END" \
    --shard-index "$SHARD_INDEX" \
    --total-shards "$TOTAL_SHARDS" \
    --grpc-port "$GRPC_PORT" \
    --api-port "$API_PORT" \
    --api-host 0.0.0.0 \
    --advertise-host "$MY_IP" \
    --peers-config "$PEERS_FILE" \
    --log-level INFO
