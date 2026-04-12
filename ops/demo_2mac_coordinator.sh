#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# OpenHydra 2-Mac Demo — Coordinator + Benchmark
#
# Run this on Mac A AFTER both peers are up. It creates a static peers file,
# starts the coordinator, fires test requests, and benchmarks TPS.
#
# Usage:
#   ./ops/demo_2mac_coordinator.sh <MAC_A_IP> <MAC_B_IP> [max_tokens]
#
# Example:
#   ./ops/demo_2mac_coordinator.sh 192.168.1.10 192.168.1.20 16
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

MAC_A_IP="${1:-}"
MAC_B_IP="${2:-}"
MAX_TOKENS="${3:-16}"

MODEL_ID="openhydra-qwen3.5-2b"
HF_MODEL="Qwen/Qwen3.5-2B"
GRPC_PORT=50051
API_PORT=8080

if [[ -z "$MAC_A_IP" || -z "$MAC_B_IP" ]]; then
    echo "Usage: $0 <MAC_A_IP> <MAC_B_IP> [max_tokens]"
    exit 1
fi

PEERS_FILE="/tmp/openhydra_2mac_peers.json"

echo "================================================================"
echo "  OpenHydra 2-Mac Coordinator + Benchmark"
echo "  Model:      $HF_MODEL (24 layers, sharded 12/12)"
echo "  Mac A peer: $MAC_A_IP:$GRPC_PORT (layers 0-11)"
echo "  Mac B peer: $MAC_B_IP:$GRPC_PORT (layers 12-23)"
echo "  Max tokens: $MAX_TOKENS"
echo "================================================================"

# Create static peers file
cat > "$PEERS_FILE" <<PEERSEOF
[
  {
    "peer_id": "mac-a-peer",
    "host": "$MAC_A_IP",
    "port": $GRPC_PORT,
    "model_id": "$MODEL_ID",
    "operator_id": "demo",
    "runtime_backend": "pytorch_mps",
    "runtime_model_id": "$HF_MODEL",
    "layer_start": 0,
    "layer_end": 12,
    "total_layers": 24
  },
  {
    "peer_id": "mac-b-peer",
    "host": "$MAC_B_IP",
    "port": $GRPC_PORT,
    "model_id": "$MODEL_ID",
    "operator_id": "demo",
    "runtime_backend": "pytorch_mps",
    "runtime_model_id": "$HF_MODEL",
    "layer_start": 12,
    "layer_end": 24,
    "total_layers": 24
  }
]
PEERSEOF

echo ""
echo "Peers file: $PEERS_FILE"
echo ""

# Check connectivity to both peers
echo "Checking peer connectivity..."
for ip in "$MAC_A_IP" "$MAC_B_IP"; do
    if nc -z -w 3 "$ip" "$GRPC_PORT" 2>/dev/null; then
        echo "  $ip:$GRPC_PORT ✓"
    else
        echo "  $ip:$GRPC_PORT ✗ — is the peer running? Check firewall."
        echo ""
        echo "  Tip: on each Mac, allow incoming connections:"
        echo "    System Settings > Network > Firewall > turn off (for demo)"
        echo "    Or: sudo pfctl -d"
        exit 1
    fi
done
echo ""

# Start coordinator in background
echo "Starting coordinator on http://127.0.0.1:$API_PORT ..."
python3 -m coordinator.api_server \
    --host 0.0.0.0 --port "$API_PORT" \
    --peers "$PEERS_FILE" \
    --model-catalog-path models.catalog.json \
    --dht-url http://127.0.0.1:1 --dht-lookup-timeout 0.1 \
    --required-replicas 1 --pipeline-width 2 \
    --timeout-ms 120000 --max-latency-ms 300000 \
    --audit-rate 0.0 --redundant-exec-rate 0.0 --auditor-rate 0.0 \
    --hydra-ledger-bridge-mock-mode &
COORD_PID=$!
sleep 4

echo ""
echo "================================================================"
echo "  Coordinator running (pid $COORD_PID)"
echo "  API: http://127.0.0.1:$API_PORT"
echo "================================================================"

# Smoke test
echo ""
echo "=== Smoke Test ==="
RESPONSE=$(curl -sS -m 120 "http://127.0.0.1:$API_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in three words.\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":0.0}")

echo "$RESPONSE" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
if 'choices' in d:
    print('Output:', repr(d['choices'][0].get('message', {}).get('content', '')))
if 'error' in d:
    print('ERROR:', d['error'][:300])
if 'openhydra' in d:
    oh = d['openhydra']
    print(f'Latency: {oh.get(\"latency_ms\", 0):.0f} ms')
    print(f'Pipeline: {oh.get(\"pipeline_mode\", \"?\")}')
    tokens = len(oh.get('response', '').split())
    latency_s = oh.get('latency_ms', 1) / 1000
    print(f'~TPS: {tokens / latency_s:.2f}')
"

echo ""
echo "=== Benchmark ($MAX_TOKENS tokens) ==="
BENCH_PROMPT="Explain peer-to-peer inference in one paragraph."
echo "Prompt: \"$BENCH_PROMPT\""
echo ""

START=$(python3 -c "import time; print(time.time())")
BENCH=$(curl -sS -m 600 "http://127.0.0.1:$API_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"$BENCH_PROMPT\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0.0}")
END=$(python3 -c "import time; print(time.time())")

echo "$BENCH" | python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
if 'choices' in d:
    print('Text:', repr(d['choices'][0].get('text', '')))
if 'error' in d:
    print('ERROR:', d['error'][:300])
if 'openhydra' in d:
    oh = d['openhydra']
    latency = oh.get('latency_ms', 0)
    print(f'Coordinator latency: {latency:.0f} ms')
    print(f'Pipeline mode: {oh.get(\"pipeline_mode\", \"?\")}')
    # Count actual generated tokens from activation
    gen_tokens = len(oh.get('response', '').split()) if oh.get('response') else 0
    if latency > 0:
        print(f'Approx TPS: {gen_tokens / (latency/1000):.2f}')
"

echo ""
echo "================================================================"
echo "  Demo complete! Coordinator still running at pid $COORD_PID"
echo "  To stop: kill $COORD_PID"
echo "  To chat: curl http://127.0.0.1:$API_PORT/v1/chat/completions ..."
echo "================================================================"
