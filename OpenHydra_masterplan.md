# OpenHydra Masterplan

> The canonical architecture document and strategic roadmap for OpenHydra v1.0+.

---

## Vision

**Build the BitTorrent of AI inference** &mdash; a permissionless, decentralised network where anyone with idle hardware can contribute compute and earn tokens, while anyone needing inference can access models larger than their own device through the swarm.

No central server. No API keys from a megacorp. No single point of failure. The network is the computer.

---

## Architectural Evolution

OpenHydra's architecture evolved through four distinct tiers, each building on the last. This progression was not accidental &mdash; each tier addresses a specific scale boundary.

### Tier 1: Local PyTorch Routing (Foundation)

**Problem:** Prove that pipeline-parallel inference works across multiple Python processes.

**Solution:**
- `ModelShard` (`peer/model_shard.py`) wraps any HuggingFace `AutoModelForCausalLM` and exposes a `forward()` that processes a slice of transformer layers (`_selected_layers`)
- `PyTorchRuntime` manages model loading, layer decomposition, and token decoding
- gRPC `ForwardRequest`/`ForwardResponse` protocol (`peer.proto`) enables peer-to-peer activation streaming on port 50051
- Coordinator (`coordinator/engine.py`) discovers peers, assembles pipelines, and chains forward calls through `coordinator/chain.py`
- `ToyRuntime` provides a deterministic CPU-only runtime for testing (no GPU required)

**Key design decisions:**
- One process = one model shard. No in-process model switching.
- Activations are serialised as flat float arrays in protobuf. No custom tensor format.
- The coordinator is stateless &mdash; any node can be a coordinator. This is what makes the system truly decentralised.

**What this proved:** Pipeline parallelism works in pure Python with standard HuggingFace models. No custom CUDA kernels required.

### Tier 2: MLX Zero-Copy on Apple Silicon

**Problem:** PyTorch on Mac is slow (MPS backend: ~1.3 tok/s). Apple Silicon's unified memory should enable much faster inference.

**Solution:**
- `MLXRuntime` (`peer/mlx_runtime.py`) loads models via `mlx_lm` and runs inference on Metal
- Unified memory eliminates CPU-GPU copies entirely &mdash; the model weights, KV cache, and activations all live in the same address space
- Quantisation detection: `_is_model_quantized()` checks for `"scales"` keys (QuantizedLinear indicator); `_detect_mlx_quantization()` infers bit-width from weight dtype
- `_apply_mlx_quantization(bits)` calls `mlx.nn.quantize(model, bits=bits)` at runtime; pre-quantised checkpoints (e.g., mlx-community 4-bit) load transparently without re-quantisation
- True tensor batching via `mx.concatenate(arrays, axis=0)` with per-request EOS tracking

**Performance impact:** 1.3 tok/s (PyTorch MPS) -> ~252 tok/s (MLX). A 194x improvement on the same hardware.

**Key design decisions:**
- MLX peers run full models only (no layer sharding). MLX's eager execution model doesn't support clean layer-level decomposition.
- Warmup call on startup eliminates 34-second cold TTFT (Metal shader compilation).
- `estimated_tokens_per_sec` is computed from actual benchmark runs during model load and announced in the DHT.

### Tier 3: Global Hivemind Kademlia DHT

**Problem:** The HTTP DHT works but doesn't scale beyond ~1,000 peers. Need O(log n) lookup for global deployment.

**Solution:**
- Dual-stack peer discovery: existing HTTP DHT (port 8468) + Hivemind Kademlia (port 38751)
- `dht/signpost.py`: Signpost daemon runs `hivemind.DHT` with no initial peers (it IS the bootstrap). Three production signposts with persistent identity keys (`identity_path` parameter) on Linode Nanodes across EU, US, and AP.
- `dht/hivemind_bridge.py`: Bridge that announces peers to both HTTP and Hivemind DHTs simultaneously
- `_DEFAULT_HIVEMIND_SIGNPOSTS` in `peer/server.py`: Hardcoded multiaddrs so peers auto-join without CLI configuration
- Systemd hardened: non-root user, `ProtectSystem=strict`, `PrivateTmp`, `NoNewPrivileges`, iptables rate limiting (20 conn/min per source IP)
- Persistent libp2p identity keys survive restarts and redeployments (excluded from rsync via `--exclude='.hivemind_identity.key'`)

**Production signpost nodes:**
| Region | IP | Port | Peer ID |
|--------|-----|------|---------|
| EU | 172.105.69.49 | 38751 | QmaEBYaG3gm8W1neRMvyyuUmeYgM8cKVe54Sy4wPE4XhBY |
| US | 45.79.190.172 | 38751 | QmPMFTzpJ5NE1FsSCjdMPgRYMhe488sXmrGfGV7tJ1ykc2 |
| AP | 172.104.164.98 | 38751 | QmPWABM7D1j41UzCtyk3r3X2yH3HPZf1UZPDJ9qTyQYYqG |

**What this proved:** A volunteer network needs robust, geographically distributed peer discovery with persistent identity. HTTP is fine for bootstrapping; Kademlia provides the scaling path.

### Tier 4: L2 AppChain Token Settlement (Future)

**Problem:** The in-memory HYDRA token economy has no real economic consequence. Slashing a peer's stake means nothing if they can restart with a fresh identity.

**Planned solution:**
- Deploy Solidity state-channel contract on Arbitrum or Base (low gas, high throughput)
- `OpenHydraLedgerBridge` (`coordinator/ledger_bridge.py`) already has the interface: `external_stake_resolver` and `external_stake_slasher` callables accept pluggable EVM RPC implementations
- On-chain stake makes slashing irrevocable &mdash; cheating has a real economic cost
- Burn-and-Mint Equilibrium: supply cap of 69M tokens, daily mint rate of 250K (governable by stakers)
- State channels for micro-payments: escrowed deposits, monotonic nonce reconciliation, 15-minute auto-expiry

**Why L2, not L1?** Gas costs on Ethereum mainnet would exceed the value of individual inference settlements. L2 rollups provide finality guarantees at ~0.01x the cost.

---

## Core Subsystems

### Inference Pipeline

```
Client Request
    |
    v
Coordinator (api_server.py)
    |-- Rate limiter (120 req/60s/IP, X-RateLimit-* headers)
    |-- DegradationPolicy (fallback to smaller model if insufficient peers)
    |-- _select_pipeline_sharded() (greedy LayerCoverageMap assembly)
    |-- Onion route construction (build_onion_route_envelope)
    |-- Speculative decode (DraftTokenModel proposes, pipeline verifies)
    v
Chain (chain.py)
    |-- ForwardRequest hop 0 -> peer A (layers 0-7)
    |-- ForwardRequest hop 1 -> peer B (layers 8-15)
    |-- ForwardRequest hop 2 -> peer C (layers 16-31, token emission)
    v
Response (SSE stream or JSON)
```

### KV Cache Compaction

```
Phase 1: HAK or OMP key selection -> standard DynamicCache
Phase 2: + beta bias (NNLS) + Cv refit (lstsq) -> CompactedKVCache
Phase 3: + per-layer/per-head JSON budgets -> non-uniform compression
Phase 4: + online mode (compact when seq_len > threshold) -> unbounded context
Option A: + real W_q(hidden) queries via AttentionQueryCapture -> quality uplift
```

Target: 10% retention ratio (keep 50 tokens from 500). Minimum 4 tokens per head.

### Security Stack

```
Identity:    Ed25519 key at ~/.openhydra/identity.key (mode 0600)
Key agree:   X25519 ECDH per pipeline hop
Encryption:  AES-256-GCM with HKDF-SHA256 derived keys
Routing:     Concentric onion layers (1/2/3 levels)
Privacy:     DP noise injection + HMAC-SHA256 audit tags
Sybil:       Geo-challenge (SHA-256 proof-of-work)
Transport:   Optional mTLS on gRPC (required in prod profile)
```

Encryption overhead: ~0.15ms per activation (0.02% of inference latency). Always on.

### Token Economy

```
Tier 1 (Barter):  1000 tokens = 1 credit, 5%/day decay, SQLite WAL
Tier 2 (HYDRA):   69M supply cap, mint/burn/stake/slash, state channels
Tier 3 (L1):      EVM bridge (mock mode), Arbitrum/Base target
```

### Verification

```
Tier 1 (Mystery Shopper):  10% re-execution rate, output comparison
Tier 2 (Redundant Exec):   N-peer majority vote
Tier 3 (Auditor):          Bernoulli spot-check when P==S match
```

Reputation score: 40% verification + 25% uptime + 20% latency + 15% stake. Range [0, 100].

---

## Strategic Roadmap

### Completed (v1.0)

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Core inference pipeline (PyTorch, gRPC, DHT) | Done |
| 2 | Ed25519 identity, TLS/mTLS, onion routing | Done |
| 3 | Three-tier verification (Mystery Shopper, redundant, auditor) | Done |
| 4 | Barter credits + HYDRA token economy + state channels | Done |
| 5 | KV cache compaction (4 phases + Option A query capture) | Done |
| 6 | MLX backend (~252 tok/s Apple Silicon) | Done |
| 7 | NF4 quantization (4x VRAM reduction) | Done |
| 8 | Request coalescing (BatchingQueue, true tensor batching) | Done |
| 9 | Layer sharding (pipeline assembly, greedy coverage) | Done |
| 10 | P2P model cache (SHA-256 verified, Range request resume) | Done |
| 11 | Auto-scaler (promote/demote models based on fleet capacity) | Done |
| 12 | Hivemind Kademlia DHT (3 signpost nodes, persistent identity) | Done |
| 13 | Ollama API compatibility (/api/generate, /api/chat) | Done |
| 14 | Open-core dual license (Apache 2.0 / AGPL v3) | Done |
| 15 | Rate limiting headers (X-RateLimit-*, Retry-After) | Done |
| 16 | CI/CD, .gitignore, CONTRIBUTING.md, SECURITY.md | Done |
| 17 | Desktop app (Tauri v2, macOS/Linux/Windows) | Done |
| 18 | SDKs (Python + TypeScript) | Done |
| 19 | Documentation (whitepaper, operator guide, API reference) | Done |
| -- | **867 tests passing, 9 skipped** | Done |

### v1.1 (Post-Launch)

| Priority | Feature | Rationale |
|----------|---------|-----------|
| P0 | SDK v1 release (Python + TypeScript, streaming, retry) | Developer adoption |
| P0 | Full MkDocs Material documentation site | SEO, onboarding |
| P1 | Adaptive speculation depth | Measure draft accuracy, auto-tune K |
| P1 | KV affinity routing (session stickiness) | Reduce cold-start TTFT |
| P1 | Connection multiplexing (HTTP/2 gRPC) | Reduce per-hop overhead |
| P2 | WebSocket transport option | Browser-native streaming |
| P2 | Model download progress UI in desktop app | UX polish |

### v2.0 (AppChain)

| Priority | Feature | Rationale |
|----------|---------|-----------|
| P0 | Solidity state-channel contract (Arbitrum/Base) | Real economic consequences for cheating |
| P0 | On-chain stake + irrevocable slashing | Sybil deterrent |
| P1 | DAO governance (mint rate, model catalog, protocol params) | Decentralised control |
| P1 | HYDRA stipend for early contributors | Bootstrap supply-side |
| P2 | Cross-chain bridge (Base <-> Arbitrum) | Liquidity |

### v3.0 (Agentic Network)

| Priority | Feature | Rationale |
|----------|---------|-----------|
| P0 | Agent sessions (multi-turn stateful inference) | LLM agents need persistent context |
| P0 | Tool execution network (MCP-compatible) | Agents need to call tools |
| P1 | Agent-to-agent delegation | Hierarchical task decomposition |
| P1 | Encrypted agent memory (per-session KV store) | Privacy for agent state |
| P2 | Agent marketplace (publish/discover/compose) | Network effects |

---

## Open-Core Licensing Strategy

The dual license structure is the economic foundation of the project:

**Apache 2.0** (`peer/`, `dht/`): The inference engine is maximally permissive. Hardware vendors can embed it. Startups can build proprietary products on it. This drives adoption and ensures the peer network grows.

**AGPL v3** (`coordinator/`, `economy/`, `verification/`, `desktop/`): The orchestration layer has network copyleft. If a cloud provider runs a modified OpenHydra coordinator as a hosted API, they must publish their source. This is the ASP loophole defense &mdash; it prevents Amazon from taking the coordinator, adding proprietary optimisations, and offering "Managed OpenHydra" without contributing back.

**Why this works:** The value of the network is in the network effect (number of peers), not in any single component. Making the peer engine Apache 2.0 maximises the number of peers. Making the coordinator AGPL ensures the orchestration logic remains open.

Commercial licensing is available for companies that need a proprietary coordinator without AGPL obligations.

---

## Competitive Positioning

| Dimension | OpenHydra | Petals | Exo |
|-----------|-----------|--------|-----|
| **Network** | Global WAN, untrusted peers | Global WAN, semi-trusted | Local LAN, trusted |
| **Discovery** | Dual-stack HTTP + Hivemind Kademlia | Hivemind DHT | mDNS / manual |
| **Parallelism** | Pipeline (layer sharding) | Pipeline (layer sharding) | Tensor (RDMA/Thunderbolt) |
| **Backends** | PyTorch + MLX + NF4 | PyTorch + CUDA | MLX |
| **Trust** | 3-tier verification + token economy | Reputation only | Implicit (trusted LAN) |
| **Privacy** | Onion routing + DP noise | None | None |
| **Economy** | Barter + HYDRA + state channels | None | None |
| **KV Compaction** | 4-phase attention matching | None | None |
| **Desktop** | Tauri v2 (macOS/Linux/Win) | None | Swift (macOS only) |
| **License** | Apache 2.0 / AGPL v3 | MIT | GPL v3 |

**Our moat:** Trust infrastructure (verification + economy + privacy) is the hardest thing to replicate. Petals has the peer network but no economic incentive. Exo has the performance but only works on trusted LANs. OpenHydra is the only system that combines all three: performance, trust, and privacy for a global untrusted network.

---

## Technical Constants

```python
# DHT
DHT_CACHE_TTL_S = 120              # coordinator cache
DHT_ANNOUNCE_INTERVAL_S = 60       # peer re-announce
DHT_ENTRY_EXPIRY_S = 300           # entry expiry
DHT_LOOKUP_TIMEOUT_S = 3.0         # per-lookup timeout

# Pipeline
PIPELINE_WIDTH = 3                  # peers per pipeline (max 16)
GRPC_TIMEOUT_MS = 500               # per-hop (60000 for real models)
MAX_LATENCY_MS = 5000               # overall deadline
KV_AFFINITY_TTL_S = 1800            # 30 min session stickiness

# Economy
BARTER_TOKENS_PER_CREDIT = 1000
BARTER_DECAY_PER_DAY = 0.05
HYDRA_SUPPLY_CAP = 69_000_000
HYDRA_DAILY_MINT = 250_000
HYDRA_CHANNEL_TTL_S = 900
HYDRA_MAX_CHANNELS_PER_PAYER = 8
HYDRA_MIN_SLASH_PENALTY = 0.10

# Rate Limiting
RATE_LIMIT_REQUESTS = 120
RATE_LIMIT_WINDOW_S = 60

# Batching
BATCH_WINDOW_MS = 50
MAX_BATCH_SIZE = 8

# Security
ENCRYPTION_LEVELS = {"standard": 1, "enhanced": 2, "maximum": 3}
KEEPIDLE_S = 10
KEEPINTVL_S = 5
KEEPCNT = 3
ULIMIT_NOFILE = 65536
```

---

## Guiding Principles

1. **No central server.** Every participant runs their own coordinator. The network is the computer.
2. **Privacy by default.** Encryption is always on. The overhead is negligible. No peer sees your full query.
3. **Earn before you burn.** Contribute compute to earn credits. Spend credits to access larger models. The economy is self-sustaining.
4. **Degrade gracefully.** If the requested model isn't available, serve the best available alternative. Never return an error when a degraded response is possible.
5. **Test everything.** 867 tests. No exceptions. Every feature ships with tests. The CI gate is non-negotiable.
6. **Open core, not open bait.** The peer engine is Apache 2.0 because adoption matters. The coordinator is AGPL because cloud providers must contribute back.
