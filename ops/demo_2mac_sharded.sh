#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# OpenHydra 2-Mac Sharded Inference Demo
#
# Runs Qwen3.5-2B (24 layers) sharded across two MacBooks on the same WiFi.
# Each Mac loads 12 layers on its Metal GPU (MPS) via PyTorch.
#
# Usage:
#   Mac A (your Mac):   ./ops/demo_2mac_sharded.sh mac-a <YOUR_IP> <FRIENDS_IP>
#   Mac B (friend's):   ./ops/demo_2mac_sharded.sh mac-b <YOUR_IP> <FRIENDS_IP>
#
# Example:
#   Mac A:  ./ops/demo_2mac_sharded.sh mac-a 192.168.1.10 192.168.1.20
#   Mac B:  ./ops/demo_2mac_sharded.sh mac-b 192.168.1.10 192.168.1.20
#
# Mac A runs both a peer (layers 0-11) AND the coordinator that knows
# about both peers. Mac B runs just a peer (layers 12-23).
#
# Once both are running, chat:
#   curl -s http://<MAC_A_IP>:8080/v1/chat/completions \
#     -H 'Content-Type: application/json' \
#     -d '{"model":"openhydra-qwen3.5-2b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

ROLE="${1:-}"
MAC_A_IP="${2:-}"
MAC_B_IP="${3:-}"

MODEL_ID="openhydra-qwen3.5-2b"
HF_MODEL="Qwen/Qwen3.5-2B"
TOTAL_LAYERS=24
LAYERS_PER_SHARD=12
GRPC_PORT=50051
API_PORT=8080

if [[ -z "$ROLE" || -z "$MAC_A_IP" || -z "$MAC_B_IP" ]]; then
    echo "Usage: $0 <mac-a|mac-b> <MAC_A_IP> <MAC_B_IP>"
    echo "Example: $0 mac-a 192.168.1.10 192.168.1.20"
    exit 1
fi

echo "================================================================"
echo "  OpenHydra 2-Mac Sharded Demo"
echo "  Model:  $HF_MODEL ($TOTAL_LAYERS layers)"
echo "  Role:   $ROLE"
echo "  Mac A:  $MAC_A_IP (layers 0-$((LAYERS_PER_SHARD - 1)), coordinator)"
echo "  Mac B:  $MAC_B_IP (layers $LAYERS_PER_SHARD-$((TOTAL_LAYERS - 1)))"
echo "================================================================"

# Create a peers config that tells the coordinator about BOTH peers
# and their layer ranges. Without this, the coordinator only knows
# about the local peer and can't assemble a sharded pipeline.
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
    "layer_end": $LAYERS_PER_SHARD,
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
    "layer_start": $LAYERS_PER_SHARD,
    "layer_end": $TOTAL_LAYERS,
    "total_layers": $TOTAL_LAYERS
  }
]
PEERSEOF
echo "Peers config: $PEERS_FILE"

if [[ "$ROLE" == "mac-a" ]]; then
    echo ""
    echo "[Mac A] Starting peer (layers 0-$((LAYERS_PER_SHARD - 1))) + coordinator..."
    echo ""

    # Mac A passes --peers-config so the coordinator knows BOTH peers
    # and can assemble the sharded pipeline.
    python3 -m coordinator.node \
        --peer-id mac-a-peer \
        --model-id "$MODEL_ID" \
        --runtime-model-id "$HF_MODEL" \
        --runtime-backend pytorch_auto \
        --layer-start 0 \
        --layer-end "$LAYERS_PER_SHARD" \
        --shard-index 0 \
        --total-shards 2 \
        --grpc-port "$GRPC_PORT" \
        --api-port "$API_PORT" \
        --api-host 0.0.0.0 \
        --peers-config "$PEERS_FILE" \
        --dht-url http://172.104.164.98:8468 \
        --log-level INFO

elif [[ "$ROLE" == "mac-b" ]]; then
    echo ""
    echo "[Mac B] Starting peer (layers $LAYERS_PER_SHARD-$((TOTAL_LAYERS - 1)))..."
    echo ""

    python3 -m coordinator.node \
        --peer-id mac-b-peer \
        --model-id "$MODEL_ID" \
        --runtime-model-id "$HF_MODEL" \
        --runtime-backend pytorch_auto \
        --layer-start "$LAYERS_PER_SHARD" \
        --layer-end "$TOTAL_LAYERS" \
        --shard-index 1 \
        --total-shards 2 \
        --grpc-port "$GRPC_PORT" \
        --api-port "$API_PORT" \
        --api-host 0.0.0.0 \
        --peers-config "$PEERS_FILE" \
        --dht-url http://172.104.164.98:8468 \
        --log-level INFO

else
    echo "Unknown role: $ROLE (must be 'mac-a' or 'mac-b')"
    exit 1
fi
