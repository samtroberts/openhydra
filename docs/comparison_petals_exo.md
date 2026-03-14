# OpenHydra vs Petals vs Exo — Deep Architecture Comparison

*Last updated: 2026-03-11 (pre-HN launch)*

## Architecture at a Glance

| | **OpenHydra** | **Petals** | **Exo** |
|---|---|---|---|
| **Parallelism** | Pipeline (layer sharding) + full-model replication | Pipeline (layer-span) | Tensor parallelism (matrix splitting) + pipeline fallback |
| **Network scope** | WAN (internet-scale, untrusted peers) | WAN (internet, but trusted peers) | LAN only (mDNS, Thunderbolt/RDMA) |
| **DHT** | Dual-stack: HTTP bootstrap + Hivemind Kademlia | Hivemind (Kademlia, full P2P) | libp2p gossipsub + mDNS |
| **Trust model** | 3-tier verification + reputation + economy | None (trust all peers) | None (trust all peers) |
| **Privacy** | E2E encryption, onion routing, DP noise | Plaintext activations | Plaintext activations |
| **Request batching** | True tensor batching (torch.cat / mx.concatenate) | None (sequential per-request) | Batch generation support |
| **Apple Silicon** | MLX backend (~120 tok/s advertised) | No MLX support | MLX native (primary engine) |
| **KV compaction** | 4-phase HAK/OMP (arXiv:2602.16284) | None | None |
| **Quantization** | NF4/INT8 (CUDA + MLX) | NF4/INT8 (CUDA only, bitsandbytes) | MLX native quantization |
| **API compat** | OpenAI + Ollama | Custom client library | OpenAI + Claude + Ollama + Responses + Images |
| **Economy** | Barter credits + HYDRA tokens | None | None |
| **Routing** | Dijkstra cost-optimal pipeline + KV affinity | Dijkstra min-latency + RTT graph | Ring topology |

---

## What OpenHydra Borrowed / Was Inspired By

### From Petals

| Feature | Status | Notes |
|---------|--------|-------|
| Layer sharding architecture | Implemented | `peer/model_shard.py` decomposes into `_selected_layers`; `coordinator/layer_coverage.py` has Dijkstra pipeline assembly |
| Request coalescing concept | Implemented | `peer/batching.py` — OpenHydra went further with true tensor batching (Petals does NOT batch across clients) |
| NF4 quantization | Implemented | Both CUDA (bitsandbytes) and MLX paths |
| Model warmup call | Implemented | Phase 0, brought TTFT from 34s to 1-3s |
| Dijkstra path routing | Implemented | Cost-optimal pipeline selection using RTT + inference time + reputation |

### From Exo

| Feature | Status | Notes |
|---------|--------|-------|
| MLX inference backend | Implemented | `peer/mlx_runtime.py` — 100x TPS improvement on Apple Silicon |
| Multi-API compatibility | Partial | OpenAI + Ollama done; Claude Messages API planned for v1.1 |
| Shard-aware model downloads | Implemented | `peer/p2p_model_cache.py` with HTTP range requests + SHA-256 verification |
| eval_with_timeout watchdog | Implemented | `_MlxWatchdog` in `peer/mlx_runtime.py` prevents GPU hangs |

### From DLPack (dmlc/dlpack)

| Feature | Status | Notes |
|---------|--------|-------|
| Zero-copy tensor bridge | Used strategically | `peer/mlx_runtime.py` uses `torch.from_dlpack()` and `mx.array()` for PyTorch-MLX tensor handoff. DLPack is built into both frameworks — no library import needed |
| Full DLPack library adoption | Skipped | Assessment: solves same-machine interchange, not network serialization |

### From Hivemind (learning-at-home/hivemind)

| Feature | Status | Notes |
|---------|--------|-------|
| Kademlia DHT | Implemented | Dual-stack: HTTP bootstrap (legacy) + hivemind Kademlia (new). Peers join global DHT via signpost nodes |
| NAT traversal | Implemented | Via libp2p relay in hivemind — free with the Go daemon |

### From adamzweiger/compaction (arXiv:2602.16284)

| Feature | Status | Notes |
|---------|--------|-------|
| Attention Matching algorithm | Fully implemented | `peer/kv_compaction/` — all 4 phases. Independent reimplementation with real-Q threading, radix prefix sharing, and entropy-based head budgets beyond the paper |

---

## Where OpenHydra Is Uniquely Ahead

Neither Petals nor Exo have any of these:

| Feature | OpenHydra | Petals | Exo |
|---------|-----------|--------|-----|
| 3-tier verification (mystery shopper, redundant exec, auditor) | Yes | No | No |
| Economic incentives (barter credits, HYDRA tokens, staking/slashing) | Yes | No | No |
| KV compaction (4-phase HAK/OMP, 50x reduction) | Yes | No | No |
| Onion routing + E2E encryption | Yes | No | No |
| Differential privacy noise injection | Yes | No | No |
| True request coalescing (cross-client tensor batching) | Yes | No (sequential) | Partial |
| Graceful degradation (auto-fallback to smaller models) | Yes | No | No |
| Dynamic layer rebalancing | Yes | No | No |

---

## Detailed Subsystem Comparisons

### Inference Pipeline

**Petals**: Activations flow sequentially through layer-sharded peers. Client chains
multiple `_ServerInferenceSession` objects. `TransformerConnectionHandler._push_outputs()`
serializes and pushes to the next server. No tensor batching across clients.

**Exo**: Tensor parallelism splits weight matrices across devices. Uses `mx.distributed`
for zero-copy Metal-to-Metal transfer on local networks. Ring topology for coordination.

**OpenHydra**: Pipeline parallelism with Dijkstra-optimal path selection. True tensor
batching via `torch.cat`/`mx.concatenate` across concurrent client requests. KV affinity
for session stickiness. Graceful degradation to smaller models when target is unavailable.

### Peer Discovery

**Petals**: Full hivemind Kademlia DHT. Peers call `declare_active_modules()` to advertise
hosted blocks. Key = `{model_prefix}.{block_index}`, subkey = base58 PeerID,
value = `ServerInfo` tuple. Reachability protocol checks NAT status before joining.

**Exo**: mDNS for local device discovery. Not designed for WAN deployment.

**OpenHydra**: Dual-stack architecture. Three Linode signpost nodes provide both HTTP
bootstrap (port 8468, backward compat) and hivemind Kademlia entry points (port 31337).
Peers join the global Kademlia network for fully decentralized discovery. HTTP DHT remains
for legacy peers.

### Trust and Verification

**Petals**: No trust mechanism. Any peer can return garbage activations with no detection.
Only viable in private/trusted swarms.

**Exo**: No trust mechanism. Assumes all devices are owned by the same user.

**OpenHydra**: Three-tier verification:
1. Mystery shopper — probabilistic re-execution of random requests
2. Redundant execution — N-peer majority vote on results
3. Auditor — Bernoulli spot-check via `AuditSampler`

Outcomes feed into `reputation_score` (0-100), which affects routing priority and
economic rewards. Ed25519 identity + geo-challenge SHA-256 proof-of-work for Sybil
resistance.

### KV Cache Optimization

**Petals**: Standard HuggingFace DynamicCache. No compaction or prefix sharing.

**Exo**: Relies on MLX's internal cache management.

**OpenHydra**: Four-phase compaction pipeline (peer/kv_compaction/):
1. HAK or OMP key selection
2. Beta bias correction + Cv refit
3. Per-layer/per-head token budgets from JSON
4. Online mode — compact when seq_len exceeds threshold

Plus RadixKVCache for prefix sharing across requests with identical system prompts.

---

## Migration Roadmap

### Current (v0.9 Beta)
- HTTP DHT with 3 fixed bootstrap nodes
- Hivemind Kademlia as parallel discovery layer
- Dual-stack announce (HTTP + Kademlia)

### v1.0 (Post-Launch)
- Deprecate HTTP-only peers (require hivemind capability)
- Full mx.distributed for local Apple Silicon clusters
- Claude Messages API compatibility
- Peer blacklisting with exponential backoff

### v1.1
- Remove HTTP DHT dependency (hivemind-only)
- Dynamic model promotion based on swarm demand
- Cross-model KV sharing for related model families
