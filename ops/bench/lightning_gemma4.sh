#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# Lightning T4×2: Gemma 4 E2B-it Sharded Benchmark
#
# Handles: SSH keepalive, process cleanup, model download, peer launch,
#          API tunnel, and benchmark execution.
#
# Usage:
#   ./ops/bench/lightning_gemma4.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU1="s_01knk4fr41phzt8w0ad8q49cjy@ssh.lightning.ai"
GPU2="s_01knk4ftdeeng3zctxh2zq0w3m@ssh.lightning.ai"
SSH_OPTS="-o ConnectTimeout=15 -o ServerAliveInterval=5 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

MODEL_ID="openhydra-gemma4-e2b"
HF_MODEL="google/gemma-4-E2B-it"
TOTAL_LAYERS=35
SPLIT=17  # GPU1: 0-16, GPU2: 17-34

trap 'echo "Cleaning up..."; kill $(jobs -p) 2>/dev/null; exit 0' EXIT INT TERM

# ── Helper: run SSH command with retry ───────────────────────────────
run_ssh() {
    local host="$1"; shift
    local attempts=3
    for i in $(seq 1 $attempts); do
        if ssh $SSH_OPTS "$host" "$@" 2>&1; then
            return 0
        fi
        echo "  [retry $i/$attempts] SSH to $host failed, waiting 5s..."
        sleep 5
    done
    echo "  FAILED after $attempts attempts"
    return 1
}

# ── Step 1: Verify connectivity ──────────────────────────────────────
echo "=== Step 1: Testing SSH connectivity ==="
run_ssh "$GPU1" "echo GPU1 connected"
run_ssh "$GPU2" "echo GPU2 connected"

# ── Step 2: Kill ALL old processes ───────────────────────────────────
echo ""
echo "=== Step 2: Killing old OpenHydra processes ==="
# Kill only screen sessions we created — NEVER killall python3 (kills studio services)
run_ssh "$GPU1" 'for s in gemma gemma2 gemma4 g4 q9b q9b2 q9b3 p2p p2p9b vpc; do screen -S $s -X quit 2>/dev/null; done; sleep 3; pgrep -f coordinator.node && echo "WARNING: still running" || echo "GPU1 clear"'
run_ssh "$GPU2" 'for s in gemma gemma2 gemma4 g4 q9b q9b2 q9b3 p2p p2p9b vpc; do screen -S $s -X quit 2>/dev/null; done; sleep 3; pgrep -f coordinator.node && echo "WARNING: still running" || echo "GPU2 clear"'

echo "Waiting 5s for ports to release..."
sleep 5

run_ssh "$GPU1" 'ss -tlnp | grep -E "50051|8080|4001" && echo "WARNING: ports still held — try rebooting studio" || echo "GPU1 ports free"'
run_ssh "$GPU2" 'ss -tlnp | grep -E "50051|8080|4001" && echo "WARNING: ports still held — try rebooting studio" || echo "GPU2 ports free"'

# ── Step 3: Ensure deps installed ────────────────────────────────────
echo ""
echo "=== Step 3: Checking dependencies ==="
run_ssh "$GPU1" 'python3 -c "import openhydra_network, torch, transformers, sentencepiece; print(\"GPU1 deps OK\")"'
run_ssh "$GPU2" 'python3 -c "import openhydra_network, torch, transformers, sentencepiece; print(\"GPU2 deps OK\")"'

# ── Step 4: Ensure model is downloaded ───────────────────────────────
echo ""
echo "=== Step 4: Checking model cache ==="
run_ssh "$GPU1" "python3 -c \"
from huggingface_hub import try_to_load_from_cache
r = try_to_load_from_cache('$HF_MODEL', 'config.json')
print('GPU1 model cached:', r is not None and isinstance(r, str))
if r is None or not isinstance(r, str):
    print('Downloading...')
    from huggingface_hub import snapshot_download
    snapshot_download('$HF_MODEL')
    print('Downloaded')
\""

run_ssh "$GPU2" "python3 -c \"
from huggingface_hub import try_to_load_from_cache
r = try_to_load_from_cache('$HF_MODEL', 'config.json')
print('GPU2 model cached:', r is not None and isinstance(r, str))
if r is None or not isinstance(r, str):
    print('Downloading...')
    from huggingface_hub import snapshot_download
    snapshot_download('$HF_MODEL')
    print('Downloaded')
\""

# ── Step 5: Launch peers ─────────────────────────────────────────────
echo ""
echo "=== Step 5: Launching peers ==="

run_ssh "$GPU1" "screen -wipe 2>/dev/null; screen -dmS gemma4 bash -c 'cd ~/openhydra && python3 -m coordinator.node \
    --peer-id gpu1-gemma \
    --model-id $MODEL_ID \
    --runtime-model-id $HF_MODEL \
    --runtime-backend pytorch_auto \
    --layer-start 0 --layer-end $SPLIT \
    --shard-index 0 --total-shards 2 \
    --grpc-port 50051 --api-port 8080 --api-host 0.0.0.0 \
    --p2p-enabled \
    --log-level INFO > /tmp/gemma4_gpu1.log 2>&1' && echo GPU1 launched"

run_ssh "$GPU2" "screen -wipe 2>/dev/null; screen -dmS gemma4 bash -c 'cd ~/openhydra && python3 -m coordinator.node \
    --peer-id gpu2-gemma \
    --model-id $MODEL_ID \
    --runtime-model-id $HF_MODEL \
    --runtime-backend pytorch_auto \
    --layer-start $SPLIT --layer-end $TOTAL_LAYERS \
    --shard-index 1 --total-shards 2 \
    --grpc-port 50051 --api-port 8080 --api-host 0.0.0.0 \
    --p2p-enabled \
    --log-level INFO > /tmp/gemma4_gpu2.log 2>&1' && echo GPU2 launched"

# ── Step 6: Wait for model loading ───────────────────────────────────
echo ""
echo "=== Step 6: Waiting for model load (up to 3 min) ==="
for i in $(seq 1 18); do
    sleep 10
    GPU1_READY=$(ssh $SSH_OPTS "$GPU1" "grep -c 'proxy_handler_loop started' /tmp/gemma4_gpu1.log 2>/dev/null || echo 0" | tr -d '[:space:]')
    GPU2_READY=$(ssh $SSH_OPTS "$GPU2" "grep -c 'proxy_handler_loop started' /tmp/gemma4_gpu2.log 2>/dev/null || echo 0" | tr -d '[:space:]')
    GPU1_READY=${GPU1_READY:-0}
    GPU2_READY=${GPU2_READY:-0}
    echo "  [${i}0s] GPU1=$([ "$GPU1_READY" -gt 0 ] 2>/dev/null && echo READY || echo loading) GPU2=$([ "$GPU2_READY" -gt 0 ] 2>/dev/null && echo READY || echo loading)"
    if [ "$GPU1_READY" -gt 0 ] 2>/dev/null && [ "$GPU2_READY" -gt 0 ] 2>/dev/null; then
        echo "Both peers ready!"
        break
    fi
done

# Check for errors
echo ""
echo "=== GPU1 status ==="
run_ssh "$GPU1" "grep -E 'listening|announced.*Kademlia|Error|OOM' /tmp/gemma4_gpu1.log | grep -vi 'DHT announce trans' | tail -3"
echo ""
echo "=== GPU2 status ==="
run_ssh "$GPU2" "grep -E 'listening|announced.*Kademlia|Error|OOM' /tmp/gemma4_gpu2.log | grep -vi 'DHT announce trans' | tail -3"

# ── Step 7: Start persistent API tunnel ──────────────────────────────
echo ""
echo "=== Step 7: Starting API tunnel ==="
# Kill any old tunnels on port 8080
pkill -f "ssh.*8080.*lightning" 2>/dev/null || true
pkill -f "ssh.*-L 8080" 2>/dev/null || true
sleep 2
ssh $SSH_OPTS -N -L 8080:127.0.0.1:8080 "$GPU1" &
TUNNEL_PID=$!
sleep 5
# Verify tunnel works
if curl -s --max-time 5 http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
    echo "API tunnel OK (PID=$TUNNEL_PID)"
else
    echo "WARNING: API tunnel may not be working"
fi

# ── Step 8: Benchmark ────────────────────────────────────────────────
echo ""
echo "=== Step 8: Running benchmark ==="
echo ""

cd "$(dirname "$0")/../.." || exit 1

.venv/bin/python3 -c "
import requests, json, time

API = 'http://127.0.0.1:8080/v1/chat/completions'
MODEL = '$MODEL_ID'

prompts = [
    ('50-word letter', 'Write a 50-word letter to my friend', 80),
    ('P2P in 3 sentences', 'Explain how peer-to-peer inference works in 3 sentences.', 128),
    ('P2P in 10 sentences', 'Explain how peer-to-peer inference works in 10 sentences.', 256),
    ('Pulp Fiction review', 'Write a Pulp Fiction movie review in 20 sentences.', 512),
]

print('Warming up...')
for _ in range(2):
    try:
        r = requests.post(API, json={'model':MODEL,'messages':[{'role':'user','content':'Hi'}],'max_tokens':4,'temperature':0.0,'grounding':False}, timeout=120)
    except: pass
    time.sleep(2)

print(f'\n{\"Prompt\":<25} {\"Latency\":>10} {\"Mode\":>12}')
print('-' * 50)

for name, prompt, max_tok in prompts:
    try:
        r = requests.post(API, json={
            'model': MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tok,
            'temperature': 0.0,
            'grounding': False,
        }, timeout=600)
        d = r.json()
        if 'error' in d:
            print(f'{name:<25} ERROR: {d[\"error\"][:80]}')
        else:
            m = d['openhydra']
            print(f'{name:<25} {round(m[\"latency_ms\"]/1000,1):>8}s {m[\"pipeline_mode\"]:>12}')
    except Exception as e:
        print(f'{name:<25} EXCEPTION: {str(e)[:60]}')
    time.sleep(0.5)

print()
"

# ── Step 9: Get exact token counts ───────────────────────────────────
echo "=== Token counts ==="
run_ssh "$GPU1" "grep sharded_done /tmp/gemma4_gpu1.log | tail -6"

echo ""
echo "=== Done ==="
