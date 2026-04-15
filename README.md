<p align="center">
  <h1 align="center">OpenHydra</h1>
  <p align="center"><strong>Turn on. Tune in. Drop in.</strong></p>
  <p align="center"><em>Your laptop is a supercomputer waiting to happen.</em></p>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License"></a>
</p>

---

OpenHydra is a peer-to-peer inference network that turns idle hardware into a global AI swarm. Any Mac, NVIDIA GPU, or AMD GPU can join. No central server. No API keys. No $20/month subscription. Just start a node and you're contributing compute.

**Why OpenHydra?**

- **No VRAM ceiling.** A 70B model that needs 140 GB runs across 8 peers, each contributing 18 GB.
- **No central server.** Every node is both client and server. The network is the computer.
- **Auto-discovery.** Peers find each other via Kademlia DHT + mDNS. No IPs to configure.
- **Privacy by default.** Onion routing + AES-256-GCM encryption + differential privacy. No peer sees your full query.
- **Earn while you idle.** HYDRA tokens and barter credits for every request your node serves.

---

## Quick Start

### Install (macOS Apple Silicon)

```bash
# Prerequisites (one-time)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12
xcode-select --install

# Clone and install
git clone https://github.com/samtroberts/openhydra.git
cd openhydra
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-mlx.txt

# Build the P2P networking module (Rust + PyO3)
pip install maturin
cd network && maturin build --release && pip install target/wheels/*.whl && cd ..
```

### Install (Linux / NVIDIA GPU)

```bash
git clone https://github.com/samtroberts/openhydra.git
cd openhydra
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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
    --grpc-port 50051 --api-port 8080 --api-host 0.0.0.0 \
    --p2p-enabled --log-level INFO
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
    --grpc-port 50051 --api-port 8080 --api-host 0.0.0.0 \
    --p2p-enabled --log-level INFO
```

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

## Benchmarks

Measured on real hardware with push mode (peer-to-peer forwarding) and KV-aware caching:

| Model | Hardware | Transport | Short prompt | Long prompt |
|-------|----------|-----------|-------------|------------|
| Qwen 3.5 2B | 2 x MacBook Air M1 8GB (MLX 8-bit) | LAN push mode | **6.9 TPS** | **6.9 TPS** |
| Qwen 3.5 2B | 2 x NVIDIA T4 GPU (CUDA) | P2P auto-discovered | **9.3 TPS** | **9.8 TPS** |
| Qwen 3.5 9B | 2 x NVIDIA T4 GPU (CUDA) | P2P auto-discovered | **7.2 TPS** | **7.3 TPS** |

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
  Peer Pipeline (gRPC :50051)
      peer-A (layers 0-11)  -->  peer-B (layers 12-23, emits tokens)
      |                          |
      |  KV-aware caching       |  Token sampling
      |  Binary-packed wire     |  INT8 activation compression
      v                          v
  libp2p Bootstrap (Kademlia DHT + Circuit Relay v2)
      3 geo-distributed nodes (US / EU / AP) — auto-connected via --p2p-enabled
```

Each node bundles two components in a single process:

- **Peer** — a gRPC inference server that loads one model shard (or a full model) and announces itself to the Kademlia DHT.
- **Coordinator** — an OpenAI-compatible HTTP API that discovers peers, assembles inference pipelines, and manages verification + token economy.

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
peer/              Inference engine: gRPC server, model shard, MLX/PyTorch runtimes,
                   KV compaction, request coalescing, P2P model cache, batching
coordinator/       HTTP API, pipeline routing, chain failover, speculative decode,
                   auto-scaler, verification, economy
network/           Rust libp2p networking layer (Kademlia, Circuit Relay, mDNS, PyO3)
dht/               HTTP DHT bootstrap server
economy/           Barter credits + HYDRA token + state channels (SQLite & Postgres)
verification/      Mystery Shopper, redundant execution, auditor spot-checks
ops/               Terraform, Docker Compose, Prometheus/Grafana, deploy scripts
tests/             1100+ tests (unit + integration + API emulation)
```

---

## Troubleshooting

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
<summary><strong>"Address already in use" (port 8080, 50051, or 4001)</strong></summary>

A previous OpenHydra process is still running. Kill it:

```bash
# Kill all stale OpenHydra processes
lsof -ti:8080 -ti:50051 -ti:4001 | xargs kill -9

# Then retry your command
```

Or use a different port: `--api-port 8081`
</details>

<details>
<summary><strong>"No viable model found" / 503</strong></summary>

The coordinator can't find peers. Ensure `--p2p-enabled` is set and both peers show `announced to Kademlia DHT`.
</details>

---

## Security & Privacy

| Layer | Mechanism |
|-------|-----------|
| **Identity** | Ed25519 keys at `~/.openhydra/identity.key` (mode 0600) |
| **Transport** | X25519 ECDH + AES-256-GCM per activation |
| **Routing** | Onion routing — layered encryption through pipeline stages |
| **Privacy** | Differential privacy noise injection with verifiable audit tags |
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
