# OpenHydra Nanode Peer Setup Guide

Complete guide to recreating the 4-node Linode nanode cluster for sharded
pipeline inference. Written from the experience of the April 2026 deployment.

## Overview

**Purpose**: 5-stage WAN sharded pipeline for Qwen2.5-0.5B (24 layers)
**Topology**: Mac (Bangalore, layers 0-7) + 4 Linode nanodes (layers 8-23)
**Cost**: 4 × $5/month = $20/month total

| Node            | IP              | Region    | Linode DC     | Layers   | Shard |
|-----------------|-----------------|-----------|---------------|----------|-------|
| mac-bangalore   | 127.0.0.1:50061 | Bangalore | (local Mac)   | [0, 8)   | 0     |
| nano-chennai    | 172.232.97.56   | Chennai   | in-maa        | [8, 12)  | 1     |
| nano-mumbai     | 172.236.187.59  | Mumbai    | in-bom (was in-maa) | [12, 16) | 2     |
| nano-singapore1 | 139.162.16.181  | Singapore | ap-south      | [16, 20) | 3     |
| nano-singapore2 | 139.162.4.85    | Singapore | ap-south      | [20, 24) | 4     |

## Step 1: Create Nanodes on Linode

Create 4 Linode Nanodes (1GB RAM, $5/month each):
- **Plan**: Nanode 1GB
- **OS**: Ubuntu 24.04 LTS
- **Regions**: Choose based on proximity to coordinator
  - Chennai (in-maa) or Mumbai (in-bom) for India-based coordinator
  - Singapore (ap-south) for Asia-Pacific
- **SSH key**: Add your public key during creation

```bash
# Note the IPs assigned to each nanode after creation
```

## Step 2: Base System Setup

SSH into each nanode and run:

```bash
# System packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential python3-dev python3-venv python3-pip git libssl-dev

# Create work directory
mkdir -p /opt/openhydra/.cache/models
```

## Step 3: Add Swap (CRITICAL for 1GB nodes)

The 1GB nanodes need extra swap for PyTorch model loading. The default Linode
swap partition (~500MB) is NOT enough. Add a 1GB swapfile:

```bash
# Create 1GB swap file
fallocate -l 1G /swapfile2
chmod 600 /swapfile2
mkswap /swapfile2
swapon /swapfile2

# Make persistent across reboots
echo '/swapfile2 none swap sw 0 0' >> /etc/fstab

# Verify: should show ~1.5GB total swap
swapon --show
free -h
```

**Why**: PyTorch loads the FULL model into RAM before layer selection. Even with
fp16, Qwen2.5-0.5B needs ~1.1GB peak during load. With OS overhead, 1GB RAM +
500MB swap is not enough. The extra 1GB swap gets you through the load spike.

## Step 4: Python Virtual Environment

```bash
VENV=/opt/openhydra/.venv
python3 -m venv "$VENV"

# Install CPU-only PyTorch (saves ~800MB vs CUDA version)
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu

# Install ML + gRPC dependencies
"$VENV/bin/pip" install \
    transformers safetensors sentencepiece huggingface_hub \
    grpcio grpcio-tools protobuf \
    requests cryptography \
    accelerate>=1.13.0
```

**Critical**: Use `--index-url https://download.pytorch.org/whl/cpu` for PyTorch.
The GPU version pulls CUDA libs that won't fit on 1GB RAM.

### Key package versions (tested working):
- torch==2.11.0+cpu
- transformers==5.5.0
- accelerate==1.13.0
- grpcio==1.80.0
- safetensors==0.7.0

Full frozen requirements saved in each node's `venv-requirements.txt`.

## Step 5: Deploy OpenHydra Code

```bash
# From your development machine:
rsync -az --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    /path/to/openhydra/ root@<nanode-ip>:/opt/openhydra/src/
```

## Step 6: Download Model Weights

```bash
/opt/openhydra/.venv/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B',
                  local_dir='/opt/openhydra/.cache/models/Qwen2.5-0.5B')
print('Model downloaded.')
"
```

This downloads ~1GB of model weights. On nanodes with slow connections, this
can take 5-10 minutes.

## Step 7: Create systemd Service

Create `/etc/systemd/system/openhydra-peer.service`:

```ini
[Unit]
Description=OpenHydra Peer (<peer-id>, layers <start>-<end>)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/openhydra/src
Environment=MALLOC_ARENA_MAX=2
Environment=PYTHONPATH=/opt/openhydra/src
ExecStart=/opt/openhydra/.venv/bin/python3 -m coordinator.node \
    --peer-id <PEER_ID> \
    --model-id openhydra-qwen2.5-0.5b \
    --runtime-model-id /opt/openhydra/.cache/models/Qwen2.5-0.5B \
    --runtime-backend pytorch_auto \
    --layer-start <LAYER_START> \
    --layer-end <LAYER_END> \
    --shard-index <SHARD_INDEX> \
    --total-shards 5 \
    --grpc-port 50051 \
    --api-port 8080 \
    --api-host 0.0.0.0 \
    --dht-url http://172.104.164.98:8468 \
    --advertise-host <PUBLIC_IP>
Restart=on-failure
RestartSec=10
MemoryMax=950M
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### Per-node values:

| Node            | PEER_ID         | LAYER_START | LAYER_END | SHARD_INDEX | PUBLIC_IP        |
|-----------------|-----------------|-------------|-----------|-------------|------------------|
| nano-chennai    | nano-chennai    | 8           | 12        | 1           | 172.232.97.56    |
| nano-mumbai     | nano-mumbai     | 12          | 16        | 2           | 172.236.187.59   |
| nano-singapore1 | nano-singapore1 | 16          | 20        | 3           | 139.162.16.181   |
| nano-singapore2 | nano-singapore2 | 20          | 24        | 4           | 139.162.4.85     |

```bash
systemctl daemon-reload
systemctl enable openhydra-peer
systemctl start openhydra-peer

# Verify
systemctl status openhydra-peer
journalctl -u openhydra-peer -f
```

**Expected startup log sequence**:
1. `OpenHydra API listening on http://0.0.0.0:8080`
2. Model loading progress bar
3. `pytorch_layer_cleanup: freed 20/24 unused layers`
4. `peer <name> listening on 0.0.0.0:50051`
5. `peer <name> announced to 1 DHT endpoints`

## Step 8: Coordinator Peers Config

On the Mac (coordinator), create a peers config JSON:

```json
[
  {"peer_id":"mac-bangalore","host":"127.0.0.1","port":50061,
   "model_id":"openhydra-qwen2.5-0.5b","runtime_backend":"pytorch_auto",
   "runtime_model_id":"Qwen/Qwen2.5-0.5B",
   "layer_start":0,"layer_end":8,"total_layers":24},
  {"peer_id":"nano-chennai","host":"172.232.97.56","port":50051,
   "model_id":"openhydra-qwen2.5-0.5b","runtime_backend":"pytorch_auto",
   "runtime_model_id":"Qwen/Qwen2.5-0.5B",
   "layer_start":8,"layer_end":12,"total_layers":24},
  {"peer_id":"nano-mumbai","host":"172.236.187.59","port":50051,
   "model_id":"openhydra-qwen2.5-0.5b","runtime_backend":"pytorch_auto",
   "runtime_model_id":"Qwen/Qwen2.5-0.5B",
   "layer_start":12,"layer_end":16,"total_layers":24},
  {"peer_id":"nano-singapore1","host":"139.162.16.181","port":50051,
   "model_id":"openhydra-qwen2.5-0.5b","runtime_backend":"pytorch_auto",
   "runtime_model_id":"Qwen/Qwen2.5-0.5B",
   "layer_start":16,"layer_end":20,"total_layers":24},
  {"peer_id":"nano-singapore2","host":"139.162.4.85","port":50051,
   "model_id":"openhydra-qwen2.5-0.5b","runtime_backend":"pytorch_auto",
   "runtime_model_id":"Qwen/Qwen2.5-0.5B",
   "layer_start":20,"layer_end":24,"total_layers":24}
]
```

**CRITICAL**: The `layer_start`, `layer_end`, `total_layers` fields MUST be
present. Without them, the coordinator cannot build a sharded pipeline and falls
back to full_model mode (which sends all layers to one nanode — guaranteed OOM).

## Step 9: Start Coordinator

```bash
python3 -m coordinator.node \
  --peer-id mac-bangalore \
  --model-id openhydra-qwen2.5-0.5b \
  --runtime-model-id Qwen/Qwen2.5-0.5B \
  --runtime-backend pytorch_auto \
  --layer-start 0 --layer-end 8 \
  --shard-index 0 --total-shards 5 \
  --grpc-port 50061 \
  --api-port 8080 \
  --api-host 127.0.0.1 \
  --dht-url http://172.104.164.98:8468 \
  --peers-config /path/to/peers.json \
  --specpipe \
  --chunked-prefill
```

---

## Memory Optimizations Applied

These are ESSENTIAL for 1GB nanodes:

1. **MALLOC_ARENA_MAX=2**: Limits glibc memory arenas. Default is 8× cores,
   each arena reserves virtual memory. On 1GB nodes, 2 is the safe maximum.

2. **MemoryMax=950M**: systemd cgroup limit prevents OOM-killer from being
   unpredictable. Leaves ~70MB for OS.

3. **fp16 + device_map={"":"cpu"}**: PyTorchRuntime auto-detects sharded mode
   and loads in float16 with explicit CPU placement. Cuts model memory by 50%.

4. **Layer cleanup**: After model loads, `_run_layers` only keeps the assigned
   layers. The other 20/24 layers are freed. This is logged as
   `pytorch_layer_cleanup: freed 20/24 unused layers`.

5. **1GB swap file**: Added on top of Linode's default 500MB swap partition.
   Without this, the model loading peak causes OOM.

6. **CPU-only PyTorch**: `torch+cpu` avoids pulling CUDA libraries.

7. **LimitNOFILE=65536**: Prevents "too many open files" under gRPC load.

---

## Errors Encountered & Fixes

### Error 1: OOM during model load
**Symptom**: Service crashes immediately after starting model download.
**Cause**: PyTorch loads the FULL model before layer selection. 1GB RAM + 500MB
default swap is insufficient for Qwen2.5-0.5B (~1.1GB in fp16).
**Fix**: Add 1GB swap file (`/swapfile2`). See Step 3.

### Error 2: `FATAL: no GPU accelerator detected`
**Symptom**: Peer refuses to start on CPU-only nanode.
**Cause**: Default startup guard requires GPU detection.
**Fix**: Code fix in `peer/server.py` — allows CPU when `_is_sharded=True`.
Already in the codebase.

### Error 3: `device_map requires accelerate`
**Symptom**: `ValueError: Using a device_map, tp_plan... requires accelerate`
**Cause**: `transformers>=5.3` requires `accelerate` when models have
`base_model_tp_plan` (Qwen2.5 does). Old `accelerate 0.32.1` was incompatible.
**Fix**: Upgrade to `accelerate>=1.13.0`. Added to `requirements.txt`.

### Error 4: Garbled CJK output from sharded pipeline
**Symptom**: Pipeline produces Chinese/Japanese characters instead of English.
**Cause**: `_run_layers()` had `try/except Exception: pass` around
`rotary_emb()` call. Error was silently swallowed, leaving
`position_embeddings=None`. Transformer blocks then produced garbage.
**Fix**: Removed the bare `except` in `peer/model_shard.py`. Position embeddings
are now always computed.

### Error 5: Sharded pipeline not selected (falls back to full_model)
**Symptom**: Coordinator sends request to single nanode in full_model mode.
**Cause**: `_dedupe_peer_entries()` used "last wins" — DHT peers (without layer
info) overwrote static config peers (with layer info), erasing layer_start/end.
**Fix**: Dedup now preserves layer metadata from the more-complete entry.
Fixed in `coordinator/discovery_service.py`.

### Error 6: Gemma-3-270m architecture incompatibility
**Symptom**: `cannot unpack non-iterable NoneType object` during forward pass.
**Cause**: `Gemma3ForCausalLM` blocks return a different tuple format than
LlamaForCausalLM. Also, model is gated (401 Unauthorized without HF token).
**Fix**: Switched to SmolLM2-360M, then to Qwen2.5-0.5B. Gemma-3 deferred.

### Error 7: WARN vs WARNING in JSON logs
**Symptom**: `test_json_formatter_promotes_extra_fields` fails in full suite.
**Cause**: `absl-py` (imported via gRPC) remaps `WARNING` → `WARN` in Python's
logging module. Only manifests when gRPC-using tests run before logging tests.
**Fix**: `_JsonFormatter` now normalizes `WARN` → `WARNING`.

### Error 8: Model catalog routing mismatch
**Symptom**: `openhydra-qwen2.5-0.5b` not found on HuggingFace.
**Cause**: `model_id` (catalog alias) used as HF weight path.
**Fix**: Separate `--model-id` (routing) from `--runtime-model-id` (HF weights).

### Error 9: Multiple models cached, wasting disk
**Note**: Each nanode has 3 models cached (Qwen2.5-0.5B, SmolLM2-360M,
gemma-3-270m) from switching during development. Only Qwen2.5-0.5B is needed.
Clean up with: `rm -rf /opt/openhydra/.cache/models/{SmolLM2-360M,gemma-3-270m}`

---

## Changing the Model

To switch from Qwen2.5-0.5B to a different model:

1. Download new model weights on all nanodes
2. Update `--runtime-model-id` in each service file
3. Adjust `--layer-start` / `--layer-end` for new model's layer count
4. Update `--total-shards` and `--shard-index` if changing node count
5. Update the coordinator's peers config JSON
6. Restart all services: `systemctl restart openhydra-peer`

### Layer assignment formula (non-equal sharding)
Mac gets more layers (8) because it has more RAM. Nanodes get 4 each.
For a model with N layers and 5 nodes:
- Mac: [0, 8) — 8 layers (33% of model)
- Nanode 1: [8, 12) — 4 layers
- Nanode 2: [12, 16) — 4 layers
- Nanode 3: [16, 20) — 4 layers
- Nanode 4: [20, N) — remaining layers

---

## Deploying Code Updates

```bash
# From development machine:
for host in 172.232.97.56 172.236.187.59 139.162.16.181 139.162.4.85; do
    rsync -az peer/model_shard.py root@$host:/opt/openhydra/src/peer/model_shard.py
done

# Restart all services:
for host in 172.232.97.56 172.236.187.59 139.162.16.181 139.162.4.85; do
    ssh root@$host "systemctl restart openhydra-peer"
done
```

---

## Snapshot Contents

Each node's snapshot directory (`ops/nanode-snapshots/<name>/`) contains:
- `openhydra-peer.service` — systemd unit file (copy to `/etc/systemd/system/`)
- `venv-requirements.txt` — pip freeze of the venv (pip install -r to recreate)
- `system-info.txt` — OS version, RAM, swap, disk usage
- `sysctl.conf` — kernel parameters
- `fstab.txt` — filesystem mounts (includes swap)
- `recent-logs.txt` — last 50 journal entries
- `cached-models.txt` — model cache directory listing
- `enabled-services.txt` — systemd enabled services
- `directory-listing.txt` — /opt/openhydra/ layout
- `venv-config.txt` — Python venv configuration

## Quick Recreate Checklist

1. [ ] Create 4 Linode Nanodes (Ubuntu 24.04, 1GB, regions: in-maa, in-bom, ap-south ×2)
2. [ ] SSH in, install system packages (`build-essential python3-dev python3-venv`)
3. [ ] Create 1GB swap file
4. [ ] Create Python venv with CPU-only PyTorch
5. [ ] Install ML deps (transformers, accelerate>=1.13.0, grpcio, etc.)
6. [ ] Download Qwen2.5-0.5B weights
7. [ ] rsync OpenHydra codebase
8. [ ] Create systemd service with correct layer range per node
9. [ ] Enable and start service
10. [ ] Verify: `journalctl -u openhydra-peer -f` shows "announced to DHT"
11. [ ] Update coordinator peers config JSON with new IPs
