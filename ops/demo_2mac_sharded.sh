#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# OpenHydra 2-Mac Sharded Inference Demo
#
# Runs Qwen3.5-2B (24 layers) sharded across two MacBooks.
# Each Mac loads 12 layers on its Metal GPU (MPS) via PyTorch.
# Peers discover each other via libp2p (mDNS on LAN, Kademlia DHT cross-ISP).
#
# Prerequisites:
#   pip install openhydra-network   (or: cd network && maturin build --release && pip install target/wheels/*.whl)
#
# Usage:
#   Mac A:  ./ops/demo_2mac_sharded.sh mac-a
#   Mac B:  ./ops/demo_2mac_sharded.sh mac-b
#
# Wait ~60s for model loading + DHT announcements, then chat from Mac A:
#   curl -s http://127.0.0.1:8080/v1/chat/completions \
#     -H 'Content-Type: application/json' \
#     -d '{"model":"openhydra-qwen3.5-2b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":32}'
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROLE="${1:-}"

MODEL_ID="openhydra-qwen3.5-2b"
HF_MODEL="Qwen/Qwen3.5-2B"
TOTAL_LAYERS=24
TOTAL_SHARDS=2
GRPC_PORT=50051
API_PORT=8080

if [[ -z "$ROLE" ]]; then
    echo "Usage: $0 <mac-a|mac-b>"
    echo ""
    echo "  mac-a  →  layers 0-11  (shard 0)"
    echo "  mac-b  →  layers 12-23 (shard 1)"
    echo ""
    echo "Both Macs auto-discover each other via the global DHT."
    echo "No IP addresses needed."
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

echo "================================================================"
echo "  OpenHydra 2-Mac Sharded Demo (libp2p)"
echo "  Model:  $HF_MODEL ($TOTAL_LAYERS layers)"
echo "  Role:   $ROLE (shard $SHARD_INDEX, layers $LAYER_START-$((LAYER_END - 1)))"
echo "  P2P:    mDNS (LAN) + Kademlia DHT (cross-ISP)"
echo "================================================================"
echo ""

# Check openhydra_network is installed.
if ! python3 -c "import openhydra_network" 2>/dev/null; then
    echo "ERROR: openhydra_network not installed."
    echo "Run: cd network && maturin build --release && pip install target/wheels/*.whl"
    exit 1
fi

echo "Starting peer with libp2p P2P networking..."
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
    --p2p-enabled \
    --log-level INFO
