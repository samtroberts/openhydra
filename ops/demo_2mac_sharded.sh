#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# OpenHydra 2-Mac Sharded Inference Demo
#
# Runs Qwen3.5-2B (24 layers) sharded across two MacBooks.
# Each Mac loads 12 layers on its Metal GPU (MPS) via PyTorch.
# Peers discover each other through the global DHT — no IPs needed.
#
# Usage:
#   Mac A:  ./ops/demo_2mac_sharded.sh mac-a
#   Mac B:  ./ops/demo_2mac_sharded.sh mac-b
#
# Wait ~60s for DHT announcements, then chat:
#   curl -s http://127.0.0.1:8080/v1/chat/completions \
#     -H 'Content-Type: application/json' \
#     -d '{"model":"openhydra-qwen3.5-2b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
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
echo "  OpenHydra 2-Mac Sharded Demo"
echo "  Model:  $HF_MODEL ($TOTAL_LAYERS layers)"
echo "  Role:   $ROLE (shard $SHARD_INDEX, layers $LAYER_START-$((LAYER_END - 1)))"
echo "  DHT:    auto-discovery (STUN detects reachable IP)"
echo "================================================================"
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
    --log-level INFO
