<p align="center">
  <h1 align="center">OpenHydra</h1>
  <p align="center"><strong>BitTorrent for LLMs &mdash; run frontier models across volunteer laptops.</strong></p>
</p>

<p align="center">
  <a href="https://github.com/openhydra-ai/openhydra/actions"><img src="https://github.com/openhydra-ai/openhydra/actions/workflows/python-app.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0%20%2F%20AGPL%203.0-green.svg" alt="License"></a>
</p>

---

OpenHydra is a fully decentralised peer-to-peer inference network that pools idle hardware &mdash; your MacBook, an NVIDIA gaming PC, an AMD workstation, a cloud VM &mdash; into a single virtual GPU cluster capable of running models far larger than any single device could hold. There is no central server, no single company that sees your queries, and no single point of failure.

Every participant runs their own node. Contributing compute earns **HYDRA** tokens and barter credits. Spending compute gives you access to frontier-class models across the swarm.

**Why OpenHydra?**

- **No VRAM ceiling.** A 70B model that needs 140 GB of VRAM runs across 8 peers, each contributing 18 GB.
- **No central server.** Every node is both client and server. The network is the computer.
- **Privacy by default.** Concentric onion routing + AES-256-GCM activation encryption + differential privacy noise. No peer sees your full query.
- **Works today.** 867 tests passing. MLX on Apple Silicon (~252 tok/s local, ~10 tok/s net over network). PyTorch + NF4 quantization on NVIDIA CUDA and AMD ROCm. Production DHT live on 3 continents.

---

## Quick Start

### Install

```bash
git clone https://github.com/samtroberts/openhydra.git
cd openhydra
pip install -e .
```

### Run your node

```bash
openhydra-node --peer-id my-node --model-id Qwen/Qwen3.5-0.8B
```

That's it. OpenHydra auto-detects your hardware (Apple Silicon → MLX, NVIDIA GPU → PyTorch CUDA, AMD GPU → PyTorch ROCm), joins the global DHT via three bootstrap signpost nodes (EU/US/AP), registers itself as a local peer, and starts an OpenAI-compatible API at `http://127.0.0.1:8080`.

**Supported platforms:**

| Platform | Backend | Install |
|----------|---------|---------|
| 🍎 Apple Silicon (M1–M4) | MLX (Metal) | `pip install mlx mlx-lm` |
| 🟢 NVIDIA GPU (CUDA) | PyTorch | `pip install torch` |
| 🔴 AMD GPU (ROCm) | PyTorch | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` |

You can also set the backend explicitly: `--runtime-backend mlx` (Mac), `--runtime-backend pytorch_auto` (NVIDIA/AMD).

### Chat with your node

```bash
# Chat completion
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openhydra-qwen3.5-0.8b",
    "messages": [{"role": "user", "content": "Explain P2P inference in one sentence."}]
  }' | python3 -m json.tool

# Streaming
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "openhydra-qwen3.5-0.8b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Ollama-compatible API

Already using Open WebUI or Continue.dev? OpenHydra speaks Ollama natively:

```bash
# Ollama-style generate
curl http://127.0.0.1:8080/api/generate \
  -d '{"model": "openhydra-qwen3.5-0.8b", "prompt": "Why is the sky blue?"}'

# Ollama-style chat
curl http://127.0.0.1:8080/api/chat \
  -d '{"model": "openhydra-qwen3.5-0.8b", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Local development (private DHT, two terminals)

```bash
# Terminal 1 - DHT bootstrap
openhydra-dht --host 127.0.0.1 --port 8468

# Terminal 2 - node
openhydra-node --peer-id dev-node \
    --dht-url http://127.0.0.1:8468
```

### Docker (full stack: node + Prometheus + Grafana)

```bash
docker compose up
# API:        http://localhost:8080
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (admin / openhydra)
```

---

## Architecture

```
  Client (curl / SDK / Open WebUI / Continue.dev)
      |  POST /v1/chat/completions  OR  /api/chat (Ollama)
      v
  Coordinator (HTTP :8080)
      |  Dual-stack DHT lookup (HTTP + Hivemind Kademlia)
      |  Pipeline assembly (sharded or full-model)
      |  Onion route construction + activation encryption
      v
  Peer Pipeline (gRPC :50051)
      peer-A (layers 0-7)  -->  peer-B (layers 8-15)  -->  peer-C (layers 16-31, emits tokens)
      |                          |                          |
      |  KV compaction          |  NF4 quantization       |  Speculative decode
      |  Radix prefix cache     |  Batched inference       |  Token streaming
      v                          v                          v
  DHT Bootstrap (HTTP :8468) + Hivemind Signposts (libp2p :38751)
      EU: 172.105.69.49  |  US: 45.79.190.172  |  AP: 172.104.164.98
```

Each node bundles two components in a single process:

- **Peer** &mdash; a gRPC inference server that loads one model shard (or a full model on capable hardware) and announces itself to the DHT every 60 seconds.
- **Coordinator** &mdash; an OpenAI-compatible HTTP API that discovers peers, assembles inference pipelines, enforces verification, and manages the token economy.

Because every participant runs their own coordinator, there is no central authority. Accessing the network means running a node &mdash; the same model as BitTorrent.

### Dual-Stack Peer Discovery

OpenHydra runs two DHT protocols simultaneously for maximum resilience:

1. **HTTP DHT** (port 8468) &mdash; lightweight announce/lookup REST API behind nginx. Sub-millisecond lookups. Used for peer discovery today.
2. **Hivemind Kademlia DHT** (port 38751) &mdash; production libp2p Kademlia network with persistent peer IDs. Three signpost nodes (EU/US/AP) bootstrap the swarm. Peers auto-join via hardcoded multiaddrs.

### Layer Sharding

A 32-layer model can be split across 4 peers, each running 8 layers. The coordinator's `LayerCoverageMap` detects which layers are available across the swarm and assembles complete pipelines using a greedy O(n*s) algorithm. If a sharded pipeline can't cover all layers, it falls back to a peer running the full model.

```bash
# Run only layers 0-7 of a 32-layer model
openhydra-node --peer-id shard-0 \
  --shard-index 0 --total-shards 4 \
  --runtime-backend pytorch_auto \
  --runtime-model-id meta-llama/Llama-3.1-8B
```

---

## The AppChain Economy

### Barter Credits (Tier 1)

Every inference request is settled peer-to-peer in barter credits (1,000 tokens served = 1 credit). Credits decay at 5%/day to prevent hoarding. SQLite WAL-mode ledger, zero external dependencies.

### HYDRA Token (Tier 2)

HYDRA is a capped-supply token (69M max) with Burn-and-Mint Equilibrium:

| Mechanism | Purpose |
|-----------|---------|
| **Mint on serve** | Peers earn HYDRA for inference work |
| **Burn on use** | Clients burn HYDRA for priority access |
| **Stake** | Staked peers get priority routing |
| **Slash** | Failed audits reduce stake |
| **State channels** | Off-chain micro-payments (15-min TTL, 8 channels/peer) |

### Three-Tier Verification

1. **Mystery Shopper** (Tier 1) &mdash; probabilistic re-execution (10% default sample rate), output comparison
2. **Redundant Execution** (Tier 2) &mdash; N-peer majority vote
3. **Auditor Spot-check** (Tier 3) &mdash; independent Bernoulli sampling when primary matches secondary

Verification outcomes feed into a weighted reputation score (40% verification, 25% uptime, 20% latency consistency, 15% stake factor) that determines routing priority.

---

## Security & Privacy

| Layer | Mechanism |
|-------|-----------|
| **Identity** | Ed25519 keys at `~/.openhydra/identity.key` (mode 0600) |
| **Transport** | X25519 ECDH key agreement + AES-256-GCM per activation |
| **Routing** | Concentric onion routing &mdash; layered encryption through pipeline stages. Each peer peels one layer and forwards the rest. |
| **Privacy** | Differential privacy noise injection with verifiable audit tags (HMAC-SHA256) |
| **Sybil resistance** | Geo-challenge: SHA-256 proof-of-work bound to claimed region |
| **Systemd hardening** | Non-root user, `ProtectSystem=strict`, `PrivateTmp`, `NoNewPrivileges`, iptables rate limiting |

Encryption overhead is ~0.15ms per activation (~0.02% of total inference latency). We keep it on by default.

---

## KV Cache Compaction

OpenHydra implements Q-Tensor KV Compaction via Attention Matching, enabling unbounded effective context length on fixed-memory hardware.

**Four phases, incrementally composable:**

| Phase | What it does | Output |
|-------|-------------|--------|
| **Phase 1** | HAK (Highest Attention Keys) or OMP (greedy residual pursuit) token selection | Standard HF DynamicCache |
| **Phase 2** | Beta bias correction (NNLS) + Cv value refit (least-squares) | CompactedKVCache with per-head beta |
| **Phase 3** | Per-layer/per-head token budgets from JSON | Non-uniform compression |
| **Phase 4** | Online mid-trajectory compaction when seq_len > threshold | Unbounded context on fixed memory |

```bash
openhydra-node --kv-compaction-mode auto \
  --kv-compaction-ratio 0.5 \
  --kv-compaction-phase 4
```

Reference: [arXiv:2602.16284](https://arxiv.org/abs/2602.16284)

---

## Model Catalog

Graceful degradation built in &mdash; if the requested model lacks peers, the coordinator serves the nearest smaller model and reports it via `X-OpenHydra-Degradation-Reason`.

| Tier | Model | HuggingFace ID | VRAM | Peers | Quant | Status |
|------|-------|----------------|------|-------|-------|--------|
| Frontier | Qwen 3.5 27B | `Qwen/Qwen3.5-27B` | 16 GB × 4 | 4 | int4 | ✅ Available |
| Advanced | Qwen 3.5 9B | `Qwen/Qwen3.5-9B` | 18 GB × 2 | 2 | int8 | ✅ Available |
| Standard | Qwen 3.5 4B | `Qwen/Qwen3.5-4B` | 9 GB | 1 | int4 | ✅ Available |
| Basic | Qwen 3.5 2B | `Qwen/Qwen3.5-2B` | 5 GB | 1 | fp32 | ✅ Available |
| Basic | Qwen 3.5 0.8B | `Qwen/Qwen3.5-0.8B` | 2 GB | 1 | fp32 | ✅ Available |

Full catalog: [`models.catalog.json`](models.catalog.json)

---

## REST API

### Public (no auth)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/readyz` | Readiness probe |
| `GET` | `/metrics` | Prometheus metrics |

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

### Economy & Network

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/network/status` | Peer inventory, reputation, economy |
| `GET` | `/v1/account/balance` | Barter + HYDRA balance |
| `POST` | `/v1/hydra/stake` | Stake HYDRA |
| `POST` | `/v1/hydra/channels/open` | Open state channel |

### Rate Limiting

Every response includes standard rate-limit headers:

```
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1741474823
Retry-After: 30          # only on 429
```

Default: 120 requests per 60-second sliding window per IP.

---

## Project Structure

```
peer/              Inference engine: gRPC server, model shard, MLX/PyTorch runtimes,
                   KV compaction, request coalescing, P2P model cache, batching
coordinator/       HTTP API, pipeline routing, chain failover, speculative decode,
                   auto-scaler, verification, economy, interactive CLI
dht/               HTTP DHT bootstrap + Hivemind Kademlia signpost daemon
economy/           Barter credits + HYDRA token + state channels (SQLite & Postgres)
verification/      Mystery Shopper, redundant execution, auditor spot-checks, reputation
compression/       LZ4 codec + learned tensor autoencoder
grounding/         DuckDuckGo RAG with local cache fallback
sdk/               Python and TypeScript SDK clients
desktop/           Tauri v2 desktop app (Rust + vanilla JS)
ops/               Terraform, Docker Compose, Prometheus/Grafana, TLS, deploy scripts
scripts/           SLO chaos test, KV benchmark, head-budget optimizer, canary rollout
tests/             867 tests (858 unit + 9 real-model integration)
```

---

## License

OpenHydra uses a **dual open-core license**:

| Component | Directories | License | Rationale |
|-----------|-------------|---------|-----------|
| **Inference engine** | `peer/`, `dht/` | [Apache 2.0](Apache-2.0.txt) | Maximum adoption. Use in proprietary products. Patent grant included. |
| **Network services** | `coordinator/`, `economy/`, `verification/`, `desktop/` | [AGPL v3](AGPL-3.0.txt) | Network copyleft. Running a modified coordinator as a hosted API requires publishing source. |

This structure ensures cloud providers cannot close-source the orchestration layer while keeping the peer engine maximally permissive for hardware vendors, embedded systems, and proprietary integrations.

Full details: [LICENSE](LICENSE)

Commercial licensing: `sam@openhydra.co`

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and commit conventions.

Good first issues: [`good first issue`](../../issues?q=label%3A%22good+first+issue%22)

---

## Security

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.

**Do not open public issues for security vulnerabilities.** Email `sam@openhydra.co` or use [GitHub Security Advisories](../../security/advisories/new).

---

## Acknowledgements & Prior Art

OpenHydra stands on the shoulders of remarkable projects and research:

- **[Petals](https://github.com/bigscience-workshop/petals)** (BigScience) &mdash; proved that pipeline-parallel LLM inference over the public internet is viable. Their work on collaborative serving of BLOOM-176B demonstrated that volunteer hardware can collectively run frontier models. OpenHydra's layer sharding pipeline is philosophically descended from Petals.
- **[Exo](https://github.com/exo-explore/exo)** &mdash; demonstrated MLX tensor parallelism for local network clusters and pioneered shard-aware model downloads. Their focus on Apple Silicon performance informed our MLX runtime design.
- **[Apple MLX](https://github.com/ml-explore/mlx)** &mdash; the MLX framework and its unified memory architecture make zero-copy inference on Apple Silicon practical. Our MLX runtime achieves ~252 tok/s thanks to their work.
- **[Hivemind](https://github.com/learning-at-home/hivemind)** &mdash; production-grade decentralised training and DHT infrastructure. OpenHydra's Kademlia signpost layer uses hivemind's libp2p DHT implementation.
- **Kademlia DHT** (Maymounkov & Mazieres, 2002) &mdash; the distributed hash table protocol that underpins peer discovery in BitTorrent, IPFS, and now OpenHydra.
- **KV Cache Compaction** ([arXiv:2602.16284](https://arxiv.org/abs/2602.16284)) &mdash; the attention matching framework that inspired our Q-Tensor compaction pipeline.
- **Speculative Decoding** (Leviathan et al., 2023; Chen et al., 2023) &mdash; the draft-then-verify paradigm that OpenHydra extends to distributed multi-peer pipelines.

---

## Roadmap

| Status | Milestone |
|--------|-----------|
| Done | Core inference, DHT, TLS, three-tier verification, barter credits, HYDRA token |
| Done | Ed25519 identity, PostgreSQL economy, Terraform IaC, Grafana dashboards |
| Done | KV cache compaction (4 phases + Option A query capture) |
| Done | MLX backend, NF4 quantization, request coalescing, P2P model cache |
| Done | Layer sharding, auto-scaler, Hivemind Kademlia DHT, Ollama API |
| Done | Open-core licensing, rate-limit headers, CI, documentation |
| Next | On-chain DAO (Solidity state-channel contract on Arbitrum/Base) |
| Next | SDK v1 (Python + TypeScript, streaming, retry, type-safe) |
| Next | P2P agentic swarms (agent sessions, tool execution, MCP) |
| Next | Full documentation site (MkDocs Material + API reference) |
