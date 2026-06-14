<p align="center">
  <h1 align="center">OpenHydra</h1>
  <p align="center"><strong>Run AI in a herd, not a data center.</strong></p>
  <p align="center"><em>A peer-to-peer protocol for free, distributed AI inference.</em></p>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
</p>

---

OpenHydra is a peer-to-peer **protocol** for free, distributed AI inference. Plug in whatever you already run — Ollama, vLLM, LM Studio, llama.cpp, Exo — and join a global network that routes inference to whoever has the model. Requests run on a single node at native local speed; models too big for any one machine fall back to sharding across peers. No central server. No API keys. No subscription. Think BitTorrent, for AI inference.

> **Evolving:** OpenHydra started as a sharded inference network (a "faster Petals"). It is becoming a general, engine-agnostic inference protocol — see the [Protocol design](docs/protocol.md). Sharded inference is retained as the fallback for very large models.

**Why OpenHydra?**

- **Plug in any engine.** Already running Ollama, vLLM, LM Studio, llama.cpp, or Exo? Join with a thin adapter and keep your stack. Any Mac, NVIDIA, or AMD GPU — any platform.
- **No VRAM ceiling.** Run any Hugging Face model. Ones too big for your machine shard across peers — a 70B that needs 140 GB runs across 8 peers, ~18 GB each.
- **Free, by reciprocity.** Give-to-get: serve inference to earn priority when you consume. No tokens, no crypto, no $20/month.
- **No central server.** Peers discover each other over a Kademlia DHT + mDNS with NAT hole-punching and relay fallback. The network is the computer.
- **Private options.** Per-hop AES-256-GCM encryption, trusted-peer pinning, and LAN-only or sharded modes where no single peer sees your full prompt.

---

## Quick Start

### Prerequisites (all platforms)

- **Python 3.11+** (3.12 recommended; 3.13 also works)
- **Rust toolchain** — the P2P wheel (`openhydra-network`) is not on PyPI yet, so it must be built from source with `maturin`. `maturin` needs `cargo` on `PATH`.
- **C compiler** — `grpcio` and `cryptography` compile C extensions if no wheel is available for your Python / arch.
- **Protobuf compiler (`protoc`)** — the Rust `prost-build` step compiles `peer.proto` at wheel-build time. Fresh macOS usually has it via Xcode CLT; on Linux install via `apt install protobuf-compiler`.
- **`numpy<2`** — some Linux managed-Python environments (e.g. Lightning AI, Modal) ship numpy 2.x but with scipy/pandas/scikit-learn wheels pinned against numpy 1.x, causing `ImportError: numpy.core.multiarray failed to import` cascades inside `transformers`. We pin to `numpy<2` in `requirements.txt`; if you ever manually upgrade inside a studio, pin it back. See Troubleshooting.
- **No `torchvision`** — OpenHydra does not use it. Some PyTorch distributions (especially on Linux CUDA studios) pre-install `torchvision` bundled to an older torch, which then fails to register C++ ops at import time (`operator torchvision::nms does not exist`) and takes `transformers` down with it. Uninstall it if present: `pip uninstall -y torchvision`.

The commands below install all of these plus OpenHydra itself.

### Install — macOS (Apple Silicon, M1/M2/M3/M4)

```bash
# One-time system prerequisites
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12 rust protobuf
xcode-select --install     # C compiler — no-op if already installed

# Clone + Python deps
git clone https://github.com/samtroberts/openhydra.git
cd openhydra
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-mlx.txt   # Apple Silicon MLX backend

# Build + install the P2P networking wheel (Rust + PyO3)
pip install maturin
cd network && maturin develop --release && cd ..
```

> **Note:** You must use Python 3.11 or 3.12. Python 3.13 works but 3.14 does not (PyTorch doesn't support it yet). If `python3 --version` shows 3.14, use `python3.12` explicitly. See Troubleshooting.

### Install — Linux (Ubuntu 22.04+ or Debian 12+, CPU or NVIDIA)

```bash
# One-time system prerequisites
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip \
                    build-essential libssl-dev python3-dev pkg-config \
                    protobuf-compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Clone + Python deps
git clone https://github.com/samtroberts/openhydra.git
cd openhydra
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# NVIDIA GPU users: install CUDA-enabled PyTorch wheels (optional but recommended)
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# Build + install the P2P networking wheel (Rust + PyO3)
pip install maturin
cd network && maturin develop --release && cd ..
```

### Install — Windows (WSL2 recommended)

OpenHydra's P2P stack and MLX/PyTorch runtimes are validated on **WSL2 Ubuntu 22.04 and 24.04**. Native Windows is not yet supported — the Rust `libp2p` build and several Python deps don't have Windows wheels for all arches.

```powershell
# In PowerShell (run as Administrator)
wsl --install -d Ubuntu
# Reboot if prompted, then open the new "Ubuntu" app and follow the Linux
# install steps above inside the WSL shell.
```

Optional desktop app (no Python needed): install the pre-built Tauri bundle from the [releases page](https://github.com/samtroberts/openhydra/releases). The bundled app runs Local Mode only; to join the swarm on Windows natively, use WSL2.

### First run

```bash
python3 -m coordinator.node --peer-id my-node --p2p-enabled
```

Your node announces to the global Kademlia DHT. From another terminal (or another machine), query the OpenAI-compatible API at `http://127.0.0.1:8080/v1/chat/completions`.

---

## Run Sharded Inference on 2 Macs

Qwen 3.5 2B has 24 transformer layers. Split them across two Macs — each loads 12 layers on its Metal GPU. Peers discover each other automatically via the global DHT.

**Mac A (layers 0-11):**

```bash
python3 -m coordinator.node \
    --peer-id mac-a-peer \
    --model-id openhydra-qwen3.5-2b \
    --runtime-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
    --runtime-backend mlx \
    --layer-start 0 --layer-end 12 \
    --shard-index 0 --total-shards 2 \
    --api-port 8080 --api-host 0.0.0.0 \
    --p2p-enabled --push-mode --log-level INFO
```

**Mac B (layers 12-23):**

```bash
python3 -m coordinator.node \
    --peer-id mac-b-peer \
    --model-id openhydra-qwen3.5-2b \
    --runtime-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
    --runtime-backend mlx \
    --layer-start 12 --layer-end 24 \
    --shard-index 1 --total-shards 2 \
    --api-port 8080 --api-host 0.0.0.0 \
    --p2p-enabled --push-mode --log-level INFO
```

`--push-mode` enables server-to-server forwarding (peer-to-peer, skipping the coordinator round-trip). On LAN this is 2× faster than the default coordinator-round-trip mode; on cross-ISP it is essential. Every example in this README uses it.

Wait for both to show `announced to Kademlia DHT (libp2p)`, then test:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openhydra-qwen3.5-2b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32
  }' | python3 -m json.tool
```

---

## Run Sharded Inference on 2 Cloud GPUs (same VPC)

On cloud providers where peers can reach each other over a private subnet (AWS VPC, GCP VPC, DigitalOcean VPC), point each peer at its internal IP with `--advertise-host` and give the second peer an explicit `--p2p-bootstrap` multiaddr for the first. This forces direct libp2p routing over the internal subnet and avoids tunneling through the public Circuit Relay — direct P2P is **30–75% faster** than relay (see the Benchmarks section).

**Peer 1 (coordinator + shard 0):**

```bash
python3 -m coordinator.node \
    --peer-id gpu1-peer \
    --runtime-model-id Qwen/Qwen3.5-2B \
    --shard-index 0 --total-shards 2 \
    --layer-start 0 --layer-end 12 \
    --p2p-enabled --push-mode \
    --advertise-host 10.0.0.1 \
    --api-host 0.0.0.0
```

Capture peer 1's libp2p peer id from its startup log (line `p2p_node_started libp2p_peer_id=12D3Koo...`).

**Peer 2 (shard 1):**

```bash
GPU1_LIBP2P_ID=12D3KooW...   # from peer 1's log

python3 -m coordinator.node \
    --peer-id gpu2-peer \
    --runtime-model-id Qwen/Qwen3.5-2B \
    --shard-index 1 --total-shards 2 \
    --layer-start 12 --layer-end 24 \
    --p2p-enabled --push-mode \
    --advertise-host 10.0.0.2 \
    --p2p-bootstrap /ip4/10.0.0.1/tcp/4001/p2p/$GPU1_LIBP2P_ID
```

Replace `10.0.0.1` / `10.0.0.2` with your own internal IPs. For genuinely public reachable peers (home LAN with port-forwarding, dedicated servers with public IPs), the same pattern works with public IPs. If peers can't reach each other directly, omit these flags and libp2p will fall back to Circuit Relay automatically.

---

## Run Sharded Inference — Cross-ISP Mac + GPU (heterogeneous)

Split Qwen 3.5 2B across a Mac (MLX, Apple Silicon) at home and a cloud GPU (PyTorch / CUDA) anywhere on the public internet. No VPN, no port forwarding, no WireGuard. libp2p Circuit Relay handles the NAT traversal via the three Linode bootstrap nodes.

**Mac (stage 0, layers 0-11, MLX on Metal):**

```bash
python3 -m coordinator.node \
    --peer-id mac-peer \
    --model-id openhydra-qwen3.5-2b \
    --runtime-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
    --runtime-backend mlx \
    --layer-start 0 --layer-end 12 \
    --shard-index 0 --total-shards 2 \
    --api-port 8080 --api-host 0.0.0.0 \
    --p2p-enabled --push-mode --log-level INFO
```

**Cloud GPU (stage 1, layers 12-23, PyTorch on CUDA):**

```bash
python3 -m coordinator.node \
    --peer-id gpu-peer \
    --model-id openhydra-qwen3.5-2b \
    --runtime-model-id Qwen/Qwen3.5-2B \
    --runtime-backend pytorch \
    --layer-start 12 --layer-end 24 \
    --shard-index 1 --total-shards 2 \
    --api-port 8080 --api-host 0.0.0.0 \
    --p2p-enabled --push-mode --log-level INFO
```

No `--advertise-host` / `--p2p-bootstrap` on either side — the peers can't reach each other directly (different ISPs), so libp2p falls back to Circuit Relay via the Linode bootstraps automatically. DCUtR will try to hole-punch a direct connection; if the NAT pair is symmetric it stays on the relay.

**Tokenizer alignment is automatic.** The Mac's MLX runtime detects that `mlx-community/Qwen3.5-2B-MLX-8bit` derives from `Qwen/Qwen3.5-2B` and transparently overrides the MLX tokenizer with the HF tokenizer so integer token ids match across the MLX↔PyTorch boundary (log line: `mlx_runtime: tokenizer overridden mlx=... -> hf=Qwen/Qwen3.5-2B (vocab=248044)`). No flags needed.

Wait for both to log `announced to Kademlia DHT (libp2p)`, then test from the Mac:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openhydra-qwen3.5-2b",
    "messages": [{"role":"user","content":"Write a three-sentence haiku-style poem about a mountain at dawn."}],
    "max_tokens": 64,
    "temperature": 0.0,
    "seed": 42
  }' | python3 -m json.tool
```

Expected output (greedy / seed=42): a short haiku about a mountain, terminated on EOS. Benchmark (verified 2026-04-24): **`tps=0.93`** on 20 ring tokens via Circuit Relay. Relay RTT dominates — for the 3.76 TPS all-LAN topology, see "Run 3-Node True Petals" below.

---

## Run Sharded Inference on 4 Cloud GPUs (Qwen3.5-9B)

Qwen 3.5 9B has 32 transformer layers. Split them across 4 GPUs (8 layers each) on the same VPC. One GPU acts as both coordinator and peer; the other three are peer-only nodes. Uses static peer discovery via `peers.local.json` for deterministic routing.

**GPU1 (coordinator + peer, layers 0-8):**

```bash
python3 -m coordinator.node \
    --peer-id gpu1-coord \
    --model-id openhydra-qwen3.5-9b \
    --runtime-model-id Qwen/Qwen3.5-9B \
    --hf-model-id Qwen/Qwen3.5-9B \
    --layer-start 0 --layer-end 8 --total-shards 4 \
    --grpc-port 50051 --runtime-backend pytorch \
    --autoregressive-sharded --pipeline-depth 1 \
    --draft-location off \
    --dht-url http://127.0.0.1:1 \
    --api-port 8080 --api-host 0.0.0.0 \
    --peers-config peers.local.json
```

**GPU2 (layers 8-16), GPU3 (layers 16-24), GPU4 (layers 24-32):**

```bash
python3 -m coordinator.node \
    --peer-id gpu2-peer \
    --model openhydra-qwen3.5-9b \
    --runtime-model-id Qwen/Qwen3.5-9B \
    --hf-model-id Qwen/Qwen3.5-9B \
    --layer-start 8 --layer-end 16 \
    --dht-url http://127.0.0.1:1
```

Repeat for GPU3 (`--layer-start 16 --layer-end 24`) and GPU4 (`--layer-start 24 --layer-end 32`), each with a unique `--peer-id`.

**`peers.local.json`** on GPU1:

```json
[
  {"peer_id":"gpu1-coord","host":"127.0.0.1","port":50051,"model_id":"openhydra-qwen3.5-9b",
   "layer_start":0,"layer_end":8,"total_layers":32,"runtime_backend":"pytorch",
   "runtime_model_id":"Qwen/Qwen3.5-9B","operator_id":"local"},
  {"peer_id":"gpu2-peer","host":"10.0.0.2","port":50051,"model_id":"openhydra-qwen3.5-9b",
   "layer_start":8,"layer_end":16,"total_layers":32,"runtime_backend":"pytorch",
   "runtime_model_id":"Qwen/Qwen3.5-9B"},
  {"peer_id":"gpu3-peer","host":"10.0.0.3","port":50051,"model_id":"openhydra-qwen3.5-9b",
   "layer_start":16,"layer_end":24,"total_layers":32,"runtime_backend":"pytorch",
   "runtime_model_id":"Qwen/Qwen3.5-9B"},
  {"peer_id":"gpu4-peer","host":"10.0.0.4","port":50051,"model_id":"openhydra-qwen3.5-9b",
   "layer_start":24,"layer_end":32,"total_layers":32,"runtime_backend":"pytorch",
   "runtime_model_id":"Qwen/Qwen3.5-9B"}
]
```

Replace `10.0.0.2-4` with your actual LAN IPs. The `runtime_backend` and `runtime_model_id` fields are **required** for all peers — without them, the autoregressive sharded loop is silently skipped and you get garbage single-token output.

Wait for all 4 GPUs to load models (~20-30s on T4), then query from GPU1:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openhydra-qwen3.5-9b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32
  }' | python3 -m json.tool
```

Verified benchmark: **~7.0 TPS sustained** on 4× Lightning AI T4 GPUs with Qwen3.5-9B. See the Benchmarks section for full results.

---

## Run 3-Node True Petals (pure-coordinator topology)

OpenHydra now supports the **true Petals client-terminated topology**: a dedicated coordinator node holds only the lm_head + embedding weights, while remote peers run the transformer layers. The coordinator samples tokens locally and re-injects them into the ring — no single peer needs to own the full model.

This is the best-performing topology when all three nodes share a LAN (VPC, home network, datacenter rack) — on three Lightning AI studios in the same `10.192.0.0/16` subnet, the ring hit **3.76 TPS on Qwen 3.5 2B**, vs 0.97 TPS on the 2-node cross-ISP legacy ring (a 3.87× gain).

**Coordinator node (no transformer layers, PyTorch backend on CPU):**

```bash
python3 -m coordinator.node \
    --peer-id coord-pure \
    --p2p-enabled --push-mode \
    --no-local-peer --sample-on-coordinator \
    --standalone-head-backend pytorch \
    --standalone-head-device cpu \
    --standalone-head-dtype bfloat16 \
    --runtime-model-id Qwen/Qwen3.5-2B \
    --hf-model-id Qwen/Qwen3.5-2B \
    --total-shards 2 \
    --api-port 7050 \
    --peers-config /path/to/peers.json
```

On Apple Silicon, use `--standalone-head-backend mlx` with an MLX-quantised model id (e.g. `mlx-community/Qwen3.5-2B-MLX-8bit`) instead. The backend dispatches automatically when `--standalone-head-backend auto` (default).

**Peer 1 and Peer 2** (stage 0 + stage 1, one per peer): use the two-GPU launch commands above, each with its `--layer-start` / `--layer-end` slice.

**`peers.json`** example for the coordinator (mirror each peer's libp2p_peer_id and LAN host):

```json
[
  {"peer_id": "gpu1", "host": "10.192.11.221",
   "layer_start": 0, "layer_end": 12, "total_layers": 24,
   "libp2p_peer_id": "12D3KooW...", "requires_relay": false},
  {"peer_id": "gpu2", "host": "10.192.15.173",
   "layer_start": 12, "layer_end": 24, "total_layers": 24,
   "libp2p_peer_id": "12D3KooW...", "requires_relay": false}
]
```

**Required guardrails** (enforced by argparse):

- `--no-local-peer` requires `--sample-on-coordinator` (otherwise the coord has no work to do).
- `--no-local-peer` requires `--runtime-model-id` with an HF repo id (slash-bearing: `Qwen/Qwen3.5-2B`, `mlx-community/Qwen3.5-2B-MLX-8bit`).
- The backend's host must have the matching runtime deps (MLX on Apple Silicon; torch + transformers on Linux).

**LAN-first routing** — when peers share a LAN subnet, libp2p mDNS discovers them automatically and establishes direct QUIC connections (no relay, no DCUtR needed). This is what makes the 3-node LAN benchmark ~4× faster than the 2-node cross-ISP ring.

**Qwen3.5 `<think>` preamble** — disabled by default. The HF chat template's `enable_thinking=False` is now wired into `EngineConfig.chat_template_default_kwargs`, saving ~40% of the user-visible token budget on Qwen3.5. Override via `EngineConfig(chat_template_default_kwargs={"enable_thinking": True})` if you want chain-of-thought.

---

## Benchmarks

Measured on real hardware from a clean `git clone` + Quick Start install. Push ring topology, KV-aware caching, deterministic seed (`seed=42`, `temperature=0.7`) — outputs are reproducible.

**Latest (OHV2 binary ring mode, 2026-06):** Qwen 3.5 2B hits **17 tok/s** on two T4 GPUs — **17 tok/s** across two T4 GPUs on a LAN, **12–13 tok/s** over a cross-ISP relay, and **8 tok/s** on two M1 MacBook Airs.

### Headline numbers

| Model | Hardware | Transport | 64 tok | 128 tok | 256 tok |
|-------|----------|-----------|--------|---------|---------|
| Qwen 3.5 2B | 2 × MacBook Air M1 8GB (MLX 8-bit) | LAN push mode | — | 6.9 TPS | — |
| Qwen 3.5 2B | 2 × NVIDIA T4 (CUDA) | **Direct P2P** (same VPC) | **9.64** | **9.57** | **9.70** |
| Qwen 3.5 9B | 2 × NVIDIA T4 (CUDA) | **Direct P2P** (same VPC) | **6.94** | **6.93** | **6.94** |
| Qwen 3.5 2B | MacBook Air M1 (MLX) ↔ T4 (CUDA) | **Cross-ISP via Circuit Relay** | 0.93 | 1.09 | — |
| Qwen 3.5 2B | **3-node True Petals** (CPU coord + 2 × T4, same VPC, Path A) | Direct LAN | — | **3.76** (`32 tok`) | — |
| Qwen 3.5 9B | **4 × NVIDIA T4** (CUDA, sharded 32 layers) | Static peers (same VPC) | **6.78** | **6.92** | **7.03** |

### 3-Node True Petals (Path A, 2026-04-24)

Three Lightning AI studios on the same `10.192.0.0/16` VPC, one as a pure coordinator (no transformer layers, just lm_head + embeddings on CPU bfloat16), two as stage 0 / stage 1 peers (PyTorch T4). The coordinator samples every token locally and re-injects into the ring — no peer ever sees the full model. Output text is a correctly-formed haiku ("Silent peak rises from the mist…"), EOS hit on the real `<|im_end|>` token.

| Topology | TPS | vs 2-node cross-ISP baseline |
|---|---|---|
| 2-node cross-ISP (Mac-MLX stage-0 + T4 stage-1), legacy ring | 0.97 | 1.00× |
| 2-node cross-ISP (Mac pure-coord + 2 × T4), Path A | 0.95 | 0.98× |
| **3-node all-LAN (CPU pure-coord + 2 × T4)**, Path A | **3.76** | **3.87×** |

Path A's theoretical 2× compounded with LAN-first wire savings + homogeneous PyTorch compute (no MLX↔PyTorch dtype casts) for the 3.87× headline. See `BENCHMARK_PATH_A.md` for the full log.

### 4-GPU Sharded Qwen3.5-9B (4 × T4 Lightning.ai, 2026-05-06)

Four Lightning AI studios, each with 1× NVIDIA T4 (15 GB VRAM), sharding Qwen3.5-9B's 32 transformer layers (8 layers per GPU). Coordinator-mediated mode (no push), KV-aware caching, `temperature=0.0`.

| Prompt | Tokens | Latency | TPS |
|--------|--------|---------|-----|
| 50-word letter | 80 | 11.8s | 6.77 |
| P2P 3 sentences | 117 | 16.9s | 6.92 |
| P2P 10 sentences | 234 | 33.3s | 7.03 |
| Pulp Fiction review | 512 | 73.1s | 7.01 |
| History of computing | 1024 | 144.6s | 7.08 |

**Sustained average: ~7.0 TPS.** Comparable to the 2-GPU baseline (7.1-7.3 TPS) because the bottleneck is the sequential coordinator-mediated round-trip per token per stage — 4 stages × ~35ms ≈ 140ms per token.

### Unified libp2p Transport (2 × T4 Lightning.ai, 2026-05-07)

After removing the gRPC server and routing all tensor traffic through the native libp2p request/response protocol (`/openhydra/tensor/1.0.0`), performance is within 4% of the gRPC baseline on direct connections:

| Prompt | Tokens | Latency | TPS (libp2p) | TPS (gRPC baseline) |
|--------|--------|---------|-------------|---------------------|
| 50-word letter | 56 | 6.3 s | **8.9** | 9.3 |
| P2P 3 sentences | 81 | 9.1 s | **8.9** | 9.3 |
| P2P 10 sentences | 226 | 24.3 s | **9.3** | 9.0 |
| Pulp Fiction review | 512 | 53.7 s | **9.5** | 9.8 |

**Qwen 3.5 9B** on the same hardware:

| Prompt | Tokens | Latency | TPS (libp2p) | TPS (gRPC baseline) |
|--------|--------|---------|-------------|---------------------|
| 50-word letter | 72 | 11.3 s | **6.4** | 7.2 |
| P2P 3 sentences | 80 | 12.3 s | **6.5** | 7.1 |
| P2P 10 sentences | 208 | 31.1 s | **6.7** | 7.3 |
| Pulp Fiction review | 512 | 75.8 s | **6.8** | 7.3 |

GPU2 discovery latency: 1.0–1.5 ms (direct QUIC on same VPC).

**After SharedProxyQueue optimization** (condvar-based shared queue bypassing the event loop, VecDeque O(1) pop, bytearray pre-allocation):

| Prompt | Tokens | Latency | TPS (optimized) | TPS (pre-opt libp2p) | TPS (gRPC baseline) |
|--------|--------|---------|-----------------|---------------------|---------------------|
| 50-word letter | 80 | 6.4 s | **12.6** | 8.9 | 9.3 |
| P2P 3 sentences | 120 | 9.5 s | **12.6** | 8.9 | 9.3 |
| P2P 10 sentences | 350 | 27.5 s | **12.7** | 9.3 | 9.0 |
| Pulp Fiction review | 600 | 45.9 s | **13.1** | 9.5 | 9.8 |

**Qwen 3.5 9B** (optimized):

| Prompt | Tokens | Latency | TPS (optimized) | TPS (pre-opt libp2p) | TPS (gRPC baseline) |
|--------|--------|---------|-----------------|---------------------|---------------------|
| 50-word letter | 68 | 7.9 s | **8.6** | 6.4 | 7.2 |
| P2P 3 sentences | 93 | 10.6 s | **8.7** | 6.5 | 7.1 |
| P2P 10 sentences | 350 | 39.0 s | **9.0** | 6.7 | 7.3 |
| Pulp Fiction review | 600 | 61.6 s | **9.7** | 6.8 | 7.3 |

Per-token latency dropped ~29% for both models. The main driver is the SharedProxyQueue: `poll_proxy_request()` now pops directly from a shared `Arc<Mutex<VecDeque>>` with condvar notification instead of round-tripping through the tokio event loop command channel.

### Direct P2P vs Circuit Relay (2 × T4 Lightning.ai, 2026-04-20)

When peers can reach each other directly (same VPC / LAN / reachable public IP), the ring uses a direct libp2p connection. When peers are behind NAT (or public reachability is blocked), traffic tunnels through a Linode Circuit Relay bootstrap node. Running the exact same prompts with the same seed, routing is the only variable:

| Model | Tokens | Direct P2P | Circuit Relay | Relay overhead |
|-------|--------|-----------|---------------|----------------|
| Qwen 3.5 2B | 64 | 9.64 TPS | 5.54 TPS | +74% |
| Qwen 3.5 2B | 128 | 9.57 TPS | 6.48 TPS | +48% |
| Qwen 3.5 2B | 256 | 9.70 TPS | 6.58 TPS | +47% |
| Qwen 3.5 9B | 64 | 6.94 TPS | 4.88 TPS | +42% |
| Qwen 3.5 9B | 128 | 6.93 TPS | 5.07 TPS | +37% |
| Qwen 3.5 9B | 256 | 6.94 TPS | 5.35 TPS | +30% |

Direct TPS is essentially flat across token counts — per-token ring cycle cost is consistent. Relay TPS scales upward with token count as the higher per-token overhead amortizes over more tokens. Relay is not catastrophic — the gap is 30–74%, not orders of magnitude.

### Cross-ISP ring topology (2026-04-17)

Genuine cross-ISP sharding — Mac on home broadband (NAT) and Lightning.ai T4 GPU on AWS — forces Circuit Relay because neither end is directly reachable:

| Tokens | Latency | TPS | Output |
|--------|---------|-----|--------|
| 64 | 68.6 s | 0.93 | Lighthouse-keeper short story |
| 128 | 118.0 s | 1.09 | Pulp Fiction review with markdown headers |

Each token cycle: Mac runs layers 0-11 (MLX Metal) → relay → GPU runs layers 12-23 (PyTorch CUDA) → relay → back to Mac. No port forwarding, no VPN, no SSH tunnel. TPS trends upward with longer outputs as fixed startup cost amortizes.

Uses the **push ring** topology — after the coordinator kicks off the first forward, tokens circulate peer-to-peer (fire-and-forget protocol, 0x03 proxy method) until EOS/max_tokens. Each hop ACKs instantly so the relay circuit is released in ~500 ms rather than held open for the full inference duration.

### Time to announce

From process launch to `announced to Kademlia DHT (libp2p)` on a clean install:

| Node | Model | HF cache | Time to announce | Dominant cost |
|------|-------|----------|------------------|---------------|
| MacBook Air M1 | Qwen 3.5 2B | warm | **~10–13 s** | MLX model load (~6 s) |
| NVIDIA T4 | Qwen 3.5 2B | warm | **~10–14 s** | PyTorch CUDA init + shard select |
| NVIDIA T4 | Qwen 3.5 2B | cold | ~71 s | 4.3 GB HF download |
| NVIDIA T4 | Qwen 3.5 9B | warm | **~31 s** | 10 GB weights load into VRAM |

The libp2p layer itself (3 relay reservations accepted across US/EU/AP bootstraps) completes in under **1 second** every run. First-request latency is model-load time, not P2P overhead.

---

## Supported Platforms

| Platform | Backend | Notes |
|----------|---------|-------|
| Apple Silicon (M1-M4) | MLX (Metal) | Native Apple Silicon. 8-bit quantized models via mlx-community. |
| NVIDIA GPU (CUDA) | PyTorch | Any CUDA-capable GPU. NF4 quantization for large models. |
| AMD GPU (ROCm) | PyTorch | ROCm 6.2+. Same PyTorch backend. |

---

## Architecture

```
  Client (curl / SDK / Open WebUI)
      |  POST /v1/chat/completions
      v
  Coordinator (HTTP :8080)
      |  Kademlia DHT + mDNS peer discovery
      |  Pipeline assembly (sharded or full-model)
      |  Push mode: peer-to-peer activation forwarding
      v
  Peer Pipeline (libp2p QUIC + TCP)
      peer-A (layers 0-11)  -->  peer-B (layers 12-23, emits tokens)
      |                          |
      |  KV-aware caching       |  Token sampling
      |  Binary-packed wire     |  INT8 activation compression
      v                          v
  libp2p Bootstrap (Kademlia DHT + Circuit Relay v2)
      3 geo-distributed nodes (US / EU / AP, dual-stack IPv4+IPv6)
```

Each node bundles two components in a single process:

- **Peer** — a libp2p inference service that loads one model shard (or a full model) and announces itself to the Kademlia DHT. All tensor traffic (forward requests, push results, pings) flows over the native libp2p request/response protocol (`/openhydra/tensor/1.0.0`) via QUIC or TCP.
- **Coordinator** — an OpenAI-compatible HTTP API that discovers peers, routes requests to providers, assembles inference pipelines, and manages verification.

### Peer Discovery

OpenHydra uses **libp2p** (Rust, via PyO3) for peer discovery and NAT traversal:

- **Kademlia DHT** — decentralized peer discovery via 3 bootstrap nodes (US/EU/AP)
- **mDNS** — zero-config LAN discovery (peers on the same WiFi find each other instantly)
- **Circuit Relay v2** — NAT traversal for peers behind firewalls (traffic relayed through bootstrap nodes)
- **AutoNAT** — automatic NAT type detection
- **DCUtR** — direct connection upgrade through relay (hole punching)

### Layer Sharding

A model's transformer layers are split across N peers. The coordinator's `LayerCoverageMap` detects which layers are available and assembles complete pipelines using a greedy algorithm.

```bash
# Example: 4 peers x 8 layers each = 32-layer model
openhydra-node --peer-id shard-0 --total-shards 4 --shard-index 0 \
    --runtime-model-id Qwen/Qwen3.5-9B --runtime-backend pytorch_auto --p2p-enabled
```

### Push Mode

By default, peers forward activations directly to each other (peer-to-peer push) instead of routing through the coordinator. This eliminates one network round-trip per token, improving throughput by ~2x on LAN and ~10x on VPC.

### Push Ring (Cross-ISP)

When peers are on different ISPs behind NAT, push-mode activations are routed through libp2p Circuit Relay. The ring topology keeps tokens circulating peer-to-peer after the initial kick-off — no coordinator round-trip per token. The fire-and-forget proxy protocol (`0x03`) releases the relay circuit within ~500 ms of the ACK, so relays never have to hold a circuit open for the full inference duration.

---

## Can more than 2 Macs connect?

Yes. There is no hardcoded peer limit. `--total-shards N` accepts any integer. The `LayerCoverageMap` assembles pipelines from any number of shards. The model catalog includes models requiring 4 peers (Qwen 3.5 27B: 4 x 16 GB).

---

## REST API

### OpenAI-compatible

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat inference (streaming + non-streaming) |
| `POST` | `/v1/completions` | Text completion |

### Ollama-compatible

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Ollama generate |
| `POST` | `/api/chat` | Ollama chat |
| `GET` | `/api/tags` | List models |

### Health & Metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/metrics` | Prometheus metrics |

---

## Model Catalog

| Model | HuggingFace ID | VRAM | Peers | Status |
|-------|----------------|------|-------|--------|
| Qwen 3.5 27B | `Qwen/Qwen3.5-27B` | 16 GB x 4 | 4 | Available |
| Qwen 3.5 9B | `Qwen/Qwen3.5-9B` | 18 GB x 2 | 2 | Available |
| Qwen 3.5 4B | `Qwen/Qwen3.5-4B` | 9 GB | 1 | Available |
| Qwen 3.5 2B | `Qwen/Qwen3.5-2B` | 5 GB | 1 | Available |
| Gemma 4 E4B-it | `google/gemma-4-E4B-it` | 8 GB | 1 | Available |
| Gemma 4 E2B-it | `google/gemma-4-E2B-it` | 4 GB | 1 | Available |

Full catalog: [`models.catalog.json`](models.catalog.json)

---

## Project Structure

```
peer/              Inference engine: libp2p service, model shard, MLX/PyTorch runtimes,
                   KV compaction, request coalescing, P2P model cache, batching
coordinator/       HTTP API, provider routing, pipeline failover, speculative decode,
                   auto-scaler, verification
network/           Rust libp2p networking layer (Kademlia, Circuit Relay, mDNS, PyO3)
dht/               HTTP DHT bootstrap server
verification/      Redundant execution, auditor spot-checks, reputation
ops/               Terraform, Docker Compose, Prometheus/Grafana, deploy scripts
tests/             1794 tests (unit + integration + API emulation)
```

---

## Troubleshooting

<details>
<summary><strong>Python 3.14: "ModuleNotFoundError: No module named 'compression._common'"</strong></summary>

Python 3.14 restructured the `gzip` stdlib module. PyTorch, transformers, and most ML libraries don't support 3.14 yet. The error cascades through `torch.profiler` → `gzip` → `compression._common` and kills the peer thread at startup.

Fix: use Python 3.11 or 3.12 (recommended):

```bash
# macOS
brew install python@3.12
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>"externally-managed-environment" when running pip install</strong></summary>

Homebrew-installed Python on macOS (3.12+) blocks system-wide pip installs via PEP 668. Use a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All subsequent `pip install` and `python3` commands will use the venv automatically. Remember to `source .venv/bin/activate` each time you open a new terminal.
</details>

<details>
<summary><strong>"zsh: command not found: maturin"</strong></summary>

`maturin` was installed by pip but isn't on your shell's PATH. Use:

```bash
python3 -m maturin develop --release
```

Or, if you're in a venv, deactivate and reactivate to refresh PATH:

```bash
deactivate && source .venv/bin/activate
maturin develop --release
```
</details>

<details>
<summary><strong>"Could not find `protoc`" during maturin build</strong></summary>

The Rust networking extension compiles `peer.proto` at build time using `protoc` (the Protocol Buffers compiler). Install it:

```bash
# macOS
brew install protobuf

# Ubuntu/Debian
sudo apt install -y protobuf-compiler

# Verify
protoc --version
```

Then retry: `cd network && maturin develop --release`
</details>

<details>
<summary><strong>"Couldn't find a virtualenv or conda environment" during maturin develop</strong></summary>

On managed-Python environments (Lightning AI studios, Modal, bare-metal servers using system Python), `maturin develop` requires a virtualenv. If you can't create one, use `maturin build` instead:

```bash
cd network
pip install maturin
maturin build --release
pip install --force-reinstall target/wheels/openhydra_network-*.whl
cd ..
```

On PEP 668 systems, add `--break-system-packages` to the `pip install` commands.
</details>

<details>
<summary><strong>"rustc, the rust compiler, is not installed"</strong></summary>

The P2P networking extension is written in Rust. Install the Rust toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
```

Then retry: `cd network && maturin develop --release`
</details>

<details>
<summary><strong>"error: command 'gcc' failed"</strong> during pip install</summary>

grpcio and cryptography require a C compiler:

```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt-get install build-essential python3-dev libssl-dev
```
</details>

<details>
<summary><strong>"Address already in use" (port 8080 or 4001)</strong></summary>

A previous OpenHydra process is still running. Kill it:

```bash
# Kill all stale OpenHydra processes
lsof -ti:8080 -ti:4001 | xargs kill -9

# Then retry your command
```

Or use a different API port: `--api-port 8081`
</details>

<details>
<summary><strong>"No viable model found" / 503</strong></summary>

The coordinator can't find peers. Ensure `--p2p-enabled` is set and both peers show `announced to Kademlia DHT`.
</details>

<details>
<summary><strong>"numpy.core.multiarray failed to import" or "numpy.dtype size changed"</strong></summary>

Managed-Python envs (Lightning AI, Modal, some Docker images) often ship numpy 2.x but with scipy / pandas / scikit-learn wheels pinned against numpy 1.x. The transitive import chain through `transformers.generation.candidate_generator → sklearn → scipy.sparse` then fails at import time.

Fix:

```bash
pip install "numpy<2" --force-reinstall
```

This is also why we pin `numpy<2` in `requirements.txt`. Don't upgrade inside a studio without verifying the rest of the stack supports it.
</details>

<details>
<summary><strong>"operator torchvision::nms does not exist" / "Could not import module 'Qwen3_5ForCausalLM'"</strong></summary>

Some PyTorch distributions (especially Lightning AI's default CUDA studios) pre-install `torchvision` built against an older torch. When torch is upgraded without torchvision being re-built, torchvision's C++ ops fail to register — and since `transformers.generation.candidate_generator` imports `torchvision` transitively, every HF model import fails with a misleading `ModuleNotFoundError: Could not import module 'Qwen3_5ForCausalLM'`.

OpenHydra does not use torchvision. Uninstall it:

```bash
pip uninstall -y torchvision
```

Verify:

```bash
python3 -c "from transformers.models.qwen3_5 import Qwen3_5ForCausalLM; print('OK')"
```
</details>

<details>
<summary><strong>Standalone head: "apply_final_head: this shard does not own the last layer"</strong></summary>

Only applies to `--sample-on-coordinator` mode. The coordinator didn't successfully register a `HeadSampler` source. Two common causes:

1. `--no-local-peer` set without `--runtime-model-id Qwen/... ` (or `mlx-community/...-MLX-...`) — the HeadSampler has no model to load. Argparse enforces this guardrail at startup; double-check the error message.
2. `--standalone-head-backend pytorch` on a host without `torch + transformers`, or `mlx` on non-Apple Silicon. Use `--standalone-head-backend auto` to let the loader pick.
</details>

---

## Security & Privacy

| Layer | Mechanism |
|-------|-----------|
| **Identity** | Ed25519 keys at `~/.openhydra/identity.key` (mode 0600) |
| **Transport** | X25519 ECDH + AES-256-GCM per hop |
| **Privacy modes** | Trusted-peer pinning; LAN-only and sharded routing so no single peer sees your full prompt |
| **NAT Traversal** | libp2p Circuit Relay v2 + DCUtR hole punching |

---

## License

Licensed under [Apache 2.0](LICENSE). The BitTorrent of AI should be free for everyone.

---

## Acknowledgements

- **[Petals](https://github.com/bigscience-workshop/petals)** — proved pipeline-parallel LLM inference over the internet is viable
- **[Exo](https://github.com/exo-explore/exo)** — MLX tensor parallelism for local clusters
- **[Apple MLX](https://github.com/ml-explore/mlx)** — native Apple Silicon inference
- **[libp2p](https://libp2p.io/)** — peer-to-peer networking (Kademlia, Circuit Relay, NAT traversal)
- **KV Cache Compaction** ([arXiv:2602.16284](https://arxiv.org/abs/2602.16284))
