# OpenHydra Beta Launch Strategy
## Three-Way Comparison + Borrowed Technologies + Design Decisions

> Based on full codebase analysis of OpenHydra, Petals (bigscience-workshop/petals), and Exo (exo-explore/exo).
> DLPack and hivemind evaluated for integration value.
> Target: Qwen 3.5 family. Success criteria: faster inference than Petals and Exo.

---

## Table of Contents

1. [Three-Way Comparison: OpenHydra vs Petals vs Exo](#1-three-way-comparison)
2. [DLPack and Hivemind Assessment](#2-dlpack-and-hivemind-assessment)
3. [Features to Borrow (Without Reinventing the Wheel)](#3-features-to-borrow-without-reinventing-the-wheel)
4. [Auto-Scaling and Model Promotion Design](#4-auto-scaling-and-model-promotion)
5. [Model Selection: Automatic vs User Choice](#5-model-selection-automatic-vs-user-choice)
6. [Torrent-Based Model Distribution](#6-torrent-based-model-distribution)
7. [Encryption Overhead Analysis](#7-encryption-overhead-analysis)
8. [Beta Launch Roadmap](#8-beta-launch-roadmap)

---

## 1. Three-Way Comparison

### Architecture Overview

| Dimension | OpenHydra | Petals | Exo |
|-----------|-----------|--------|-----|
| **Primary goal** | Trusted, economically sustainable P2P inference | Run models too large for any single machine | Run frontier AI on local device clusters |
| **Sharding model** | Full-model replication (but layer infra exists!) | Layer-level pipeline parallelism | Tensor parallelism (MLX distributed) |
| **Target network** | Internet-wide, untrusted strangers | Internet-wide, semi-trusted volunteers | LAN/Thunderbolt, trusted local devices |
| **DHT** | Custom HTTP (3 fixed bootstrap nodes) | Hivemind (Kademlia, fully P2P) | libp2p auto-discovery (mDNS/LAN) |
| **Inference backends** | PyTorch (eager, MPS/CUDA/CPU) | PyTorch + bitsandbytes + FlashAttention | MLX (primary), tinygrad, PyTorch |
| **Quantization** | Basic int4/int8 | NF4 via bitsandbytes | MLX native quantization (4-bit/8-bit) |
| **Request coalescing** | No | Yes (RuntimeWithDeduplicatedPools) | No (single-user focus) |
| **Output verification** | Yes (3-way consensus) | No | No |
| **Economy** | Barter credits + HYDRA tokens | None (altruistic) | None (personal cluster) |
| **KV cache management** | HAK/OMP compaction (research-grade) | MemoryCache (bounded, eviction) | MLX native cache |
| **API compatibility** | OpenAI `/v1/chat/completions` | Custom Python library | OpenAI + Claude + Ollama |
| **Graceful degradation** | Yes (falls back to lighter model) | No (MissingBlocksError) | No (requires all nodes) |
| **Privacy** | Onion routing + DP noise | None | None (trusted LAN) |
| **Model download** | HuggingFace (+ torrent scaffold) | HuggingFace | HuggingFace (with shard-aware download) |

### Performance Comparison

| Metric | OpenHydra (current) | Petals (published) | Exo (published) |
|--------|--------------------|--------------------|------------------|
| **TTFT (cold)** | ~34 s (MPS, no warmup) | ~5 s (CUDA) | ~1-2 s (MLX) |
| **TTFT (warm)** | ~2-5 s (MPS, pytorch_auto) | ~200 ms (CUDA) | ~100 ms (MLX) |
| **TPS (0.8B)** | 1.3 tok/s (MPS) | N/A (targets 70B+) | ~100-200 tok/s (MLX) |
| **TPS (70B)** | Cannot run (no layer sharding) | ~6 tok/s (4x A100) | ~20-40 tok/s (4x M3 Ultra) |
| **Max model size** | Single GPU VRAM | Unlimited (distributed) | Unlimited (tensor parallel) |
| **Concurrent scaling** | Linear degradation | Near-linear (coalescing) | Single-user design |

### What Each Project Does Best

**Petals excels at:**
- Running 70B-405B models across internet-connected consumer GPUs
- Dijkstra-based path routing considering inter-server latency
- Request coalescing for multi-user throughput scaling
- Dynamic layer rebalancing when servers join/leave
- NF4 quantization with bitsandbytes

**Exo excels at:**
- Raw inference speed on Apple Silicon (MLX native Metal kernels)
- Zero-config device discovery (mDNS/libp2p)
- Tensor parallelism (true matrix splitting, not just pipeline)
- RDMA over Thunderbolt 5 (99% latency reduction between devices)
- Multi-API compatibility (OpenAI + Claude + Ollama in one server)
- Shard-aware model downloads (each node downloads only its share)

**OpenHydra excels at:**
- Trust infrastructure (verification, economy, privacy)
- KV cache compaction (research-grade HAK/OMP)
- Graceful degradation under peer churn
- Production operations (monitoring, rate limiting, TLS)
- Existing layer-range infrastructure in PyTorchRuntime (discovered!)

---

## 2. DLPack and Hivemind Assessment

### DLPack — Verdict: Low priority, not needed for beta

**What it does:** DLPack provides a standard C-level tensor structure for zero-copy interchange between frameworks (PyTorch, TensorFlow, JAX, NumPy, CuPy). It avoids serialising tensors when passing between libraries on the same machine.

**Value to OpenHydra:** Minimal for the beta. OpenHydra's activation tensors travel over gRPC between machines. The bottleneck is network serialisation (`struct.pack` → gRPC → `struct.unpack`), not in-process tensor format conversion. DLPack solves same-machine, same-process interchange — not the distributed case.

**When it matters:** If OpenHydra later supports multiple backends on the same peer (e.g., MLX for prefill, PyTorch for decode), DLPack would avoid copying the KV cache between frameworks. But this is a post-launch optimisation.

**Recommendation:** Skip for beta. Revisit when multi-backend peers are implemented.

### Hivemind — Verdict: High value, but defer to post-beta

**What it does:** Hivemind is a production-grade Kademlia DHT with:
- True P2P peer discovery (no fixed bootstrap servers needed)
- NAT traversal via libp2p relay nodes
- Built-in decentralised averaging for gradient aggregation
- Automatic peer churn handling (Kademlia republishing)
- Sybil resistance through proof-of-work
- Battle-tested in Petals' production network

**Value to OpenHydra:** Replaces the 3 fixed Linode bootstrap nodes with a fully decentralised DHT. Eliminates the single-point-of-failure. Also brings NAT traversal for free (many home peers are behind NAT).

**Why defer:** Hivemind is a heavy dependency (~50 MB, requires Go for the P2P daemon subprocess). The current HTTP DHT works for a beta with <1000 peers. The migration is a 2-3 month project.

**Recommendation:** Keep HTTP DHT for beta. Plan hivemind migration for v1.0. In the meantime, add 2-3 more bootstrap nodes in different cloud providers for redundancy.

---

## 3. Features to Borrow (Without Reinventing the Wheel)

### From Exo: MLX Inference Backend

**What to borrow:** Exo's MLX inference engine delivers 80-200 tok/s on Apple Silicon. OpenHydra's PyTorchRuntime on MPS delivers 1.3 tok/s. This is the single biggest TPS improvement available.

**Implementation:** The `mlx-lm` library provides a drop-in inference API:

```python
# New file: peer/mlx_runtime.py
from mlx_lm import load, generate

class MLXRuntime:
    def __init__(self, model_id: str, **kwargs):
        self._model, self._tokenizer = load(model_id)
        # Warmup (critical!)
        generate(self._model, self._tokenizer, "warmup", max_tokens=1)

    def forward(self, prompt: str, max_tokens: int, temperature: float = 0.0):
        return generate(self._model, self._tokenizer, prompt,
                       max_tokens=max_tokens, temp=temperature)
```

Add `"mlx"` to `--runtime-backend` choices in `peer/model_shard.py`. Add `mlx` and `mlx-lm` to optional dependencies.

**Effort:** 1 week. **Impact:** 50-150x TPS improvement on Apple Silicon.

### From Exo: Shard-Aware Model Downloads

**What to borrow:** Exo downloads only the model weight files needed for a peer's assigned layer range, not the entire model. For a 70B model split across 4 peers, each peer downloads only 25% of the weights.

**Implementation:** HuggingFace models store weights in multiple safetensors files. Each file contains specific layers. Map `layer_indices` → required safetensors files → download only those.

**Effort:** 2 weeks. **Impact:** 4x faster peer bootstrap for layer-sharded models.

### From Exo: Multi-API Compatibility

**What to borrow:** Exo serves OpenAI, Claude Messages, and Ollama APIs from the same server. OpenHydra only serves OpenAI-compatible. Adding Ollama compatibility would let users use OpenHydra as a drop-in Ollama replacement.

**Effort:** 3 days. **Impact:** Broader adoption.

### From Petals: Request Coalescing

**What to borrow:** Petals' `RuntimeWithDeduplicatedPools` batches requests from multiple concurrent users into a single GPU forward pass.

**Implementation:** In `peer/server.py`, replace per-request threading with a batching queue:

```python
class BatchingQueue:
    def __init__(self, model, max_batch=32, flush_interval_ms=20):
        self._queue = queue.Queue()
        threading.Thread(target=self._batch_loop, daemon=True).start()

    def _batch_loop(self):
        while True:
            batch = self._collect_batch(timeout_ms=20, max_size=32)
            if batch:
                results = self._model.forward_batch([r.input for r in batch])
                for request, result in zip(batch, results):
                    request.future.set_result(result)
```

**Effort:** 2 weeks. **Impact:** Near-linear throughput scaling with concurrent users.

### From Petals: NF4 Quantization

**What to borrow:** `bitsandbytes` NF4 quantization, already used by Petals. Reduces a 7B model from 14 GB to ~4 GB VRAM.

**Implementation:**

```python
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

OpenHydra's `model_shard.py` already has the `quantization_config` plumbing (lines 395-415). It just needs NF4 as a first-class option.

**Effort:** 1 day. **Impact:** 7B model runs on 8 GB consumer GPU.

### From Petals: Dijkstra Path Routing

**What to borrow:** Petals' `SequenceManager` considers inter-server latency when assembling pipelines, not just individual peer scores.

**Implementation:** After DHT lookup returns candidate peers, build a directed graph with measured RTT as edge weights. Run Dijkstra to find the minimum-latency path through the full layer sequence.

**Effort:** 2 weeks. **Impact:** Lower TTFT for geo-distributed pipelines.

---

## 4. Auto-Scaling and Model Promotion

> **Detailed sub-plan:** [`plans/auto-scaling-policy.md`](https://github.com/openhydra-ai/openhydra/blob/main/plans/auto-scaling-policy.md) — full design with code, test scenarios, and implementation plan.

### Current State

**No auto-scaling exists.** The `DegradationPolicy` in `coordinator/degradation.py` only scales **down** — falling back to lighter models when the requested model has insufficient peers. There is no logic to scale **up** to larger models when capacity increases.

The `ReplicationMonitor` (`coordinator/replication_monitor.py`) checks if a model has enough healthy peers (`required_replicas`), but it only reports status — it doesn't trigger model promotion.

### Why Naive Redundancy Ratios Fail

A simple rule like "promote when 3x peers are available" breaks under real-world conditions:

| Failure Mode | What Happens |
|--------------|-------------|
| **Weak node promotion** | 40 low-VRAM nodes join; naive ratio says "promote to 8B!"; only 8 original peers can actually serve it; service degrades for everyone |
| **Gravitational collapse** | Only powerful nodes get useful work and earn HYDRA; weak nodes earn nothing, leave; swarm shrinks to a handful of elite nodes |
| **Oscillation** | Borderline redundancy triggers promote → demote → promote cycles; unstable model selection |

### Capability-Aware Design

The fix: **effective redundancy** counts only peers that can actually serve a model.

```
effective_redundancy(model) = capable_peers / shards_needed
    where capable_peers = peers with available_vram >= model.shard_vram
```

#### Qwen 3.5 Redundancy Requirements

| Model | Params | VRAM/shard (FP16) | VRAM/shard (NF4) | Shards | Min peers for 3x |
|-------|--------|-------------------|-------------------|--------|-------------------|
| Qwen3.5-0.6B | 0.6B | ~2 GB | ~1 GB | 1 | 3 |
| Qwen3.5-1.7B | 1.7B | ~4 GB | ~2 GB | 1 | 3 |
| Qwen3.5-4B | 4B | ~8 GB | ~3 GB | 1 | 3 |
| Qwen3.5-8B | 8B | ~16 GB | ~5 GB | 1-2 | 3-6 |
| Qwen3.5-14B | 14B | ~28 GB | ~8 GB | 2-4 | 6-12 |
| Qwen3.5-32B | 32B | ~64 GB | ~18 GB | 4-8 | 12-24 |
| Qwen3.5-72B | 72B | ~144 GB | ~40 GB | 8-16 | 24-48 |

#### Promotion Constraints (All Must Pass)

```python
def should_promote(candidate, current_models, peers, request_log) -> bool:
    # 1. Candidate has ≥ 3x effective redundancy (capable peers only)
    if effective_redundancy(candidate, peers) < 3.0:
        return False
    # 2. No existing model drops below 2x floor after reassignment
    for model in current_models:
        if effective_redundancy_after_reassignment(model, candidate, peers) < 2.0:
            return False
    # 3. Demand exists for this quality tier (prevents promoting models nobody wants)
    if request_log.demand_weight(candidate.quality_tier) < 0.3:
        return False
    # 4. Not in cooldown from recent promotion/demotion
    if candidate.model_id in recently_changed(window=15_minutes):
        return False
    return True
```

#### Hysteresis (Anti-Oscillation)

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Promote | 3.0x | High bar to enter a new model |
| Demote | 1.5x | Low bar to exit — gap prevents flapping |
| Floor | 2.0x | No served model may drop below this |
| Cooldown | 15 min | No changes after a promotion/demotion event |
| Re-evaluate | 5 min | Scaler tick interval (not per-request) |

#### Weak Node Roles (Preventing the Small Swarm Problem)

Nodes too weak for inference must still earn credits, or they leave and the network collapses:

| Role | VRAM Needed | Earnings | Work |
|------|-------------|----------|------|
| Embedding server | < 1 GB | 0.3x base | Serve small embedding models |
| Verification auditor | Varies | 0.5x base | Re-execute random inference samples |
| KV compaction offload | < 1 GB | 0.2x base | CPU-bound HAK/OMP for busy peers |
| Model cache seed | Disk only | 0.2x base | Serve weight chunks to new peers |
| Activation relay | < 512 MB | 0.1x base | Forward activations in sharded pipelines |

### Implementation

**Estimated effort:** ~7 days. See [`plans/auto-scaling-policy.md`](https://github.com/openhydra-ai/openhydra/blob/main/plans/auto-scaling-policy.md) for:
- Full `AutoScaler` class design with effective redundancy, demand weighting, and reassignment simulation
- `RoleAssigner` for weak node support roles
- `RequestLog` sliding-window demand tracker
- 8 key test scenarios (weak nodes, floor violations, hysteresis, gradual growth)
- File-by-file implementation plan

---

## 5. Model Selection: Automatic vs User Choice

### Current State

Users **must** specify a `model` field in the request body. If missing, the coordinator uses `default_model` from `EngineConfig` (which is a fixed config value, not dynamic).

### Recommended Design: Automatic Best-Model with User Override

```
User sends:
  POST /v1/chat/completions
  {"messages": [...]}               ← no model field

Coordinator:
  1. Compute best_available = AutoPromoter.best_available_model(dht_peer_counts)
  2. Route request to best_available
  3. Response includes: {"model": "openhydra-qwen3.5-7b", "note": "auto-selected"}

User sends:
  POST /v1/chat/completions
  {"model": "openhydra-qwen3.5-32b", "messages": [...]}  ← explicit model

Coordinator:
  1. Check if qwen3.5-32b has sufficient peers
  2. If yes → route to it
  3. If no → DegradationPolicy fallback
  4. Response includes: {"model": "openhydra-qwen3.5-7b", "degraded": true}
```

**Default behaviour for beta:** Auto-select the best model. Users who care about consistency can pin a model explicitly.

This is not currently implemented. The `default_model` config is static, not dynamically computed from network capacity.

---

## 6. Torrent-Based Model Distribution

### Current State: Scaffolding Only

The `torrent/` module has three files:
- **`genesis.py`** — Generates a manifest + piece digests (SHA-256 per 1 MB chunk) for model weights. This is the torrent metadata, not actual P2P transfer. It downloads from HuggingFace or a URL, then produces a `genesis.torrent.json` file.
- **`seeder.py`** — `BandwidthArbitrator` throttles upload bandwidth during inference (10% of upload budget for seeding, 100% when idle). This is the policy layer, but there's no actual BitTorrent protocol.
- **`session.py`** — Bootstrap config for torrent sessions. Placeholder.

**No actual peer-to-peer weight transfer is implemented.** Peers currently download from HuggingFace directly via `AutoModelForCausalLM.from_pretrained()`.

### Is This the Best Approach?

**For the beta: No.** BitTorrent is complex to implement correctly and has diminishing returns when the network is small (<100 peers). At beta scale:
- Most peers download the same model from HuggingFace
- HuggingFace CDN is fast and reliable
- Adding a BitTorrent protocol is 2-3 months of work

**For scale (1000+ peers): Yes, but not BitTorrent.**

Better alternatives:
1. **HTTP chunk transfer between peers** — Much simpler than BitTorrent. When Peer A has Qwen3.5-7B cached, it can serve weight chunks to Peer B over HTTP. The coordinator tracks which peers have which model weights cached. This is the "poor man's CDN" approach and takes ~1 week to build.

2. **IPFS** — Content-addressed storage. Peers announce CIDs for model weight chunks. Other peers fetch from the nearest source. The `ipfshttpclient` Python library handles the protocol. However, IPFS has reliability issues.

3. **Keep HuggingFace + add a local cache registry** — Simplest option. Each peer announces to DHT which models it has cached locally. New peers check if any nearby peer has the model before falling back to HuggingFace. Transfer via HTTP range requests.

### Recommendation for Beta

**Option 3** (local cache registry + HTTP fallback):
1. Peers announce their cached models to DHT: `POST /announce {model_id, cached: true}`
2. New peer wanting Qwen3.5-7B queries DHT for peers that have it cached
3. Downloads from nearest peer via HTTP (safetensors files, range requests)
4. Falls back to HuggingFace if no peer has it or transfer is too slow
5. All integrity verified via SHA-256 checksums (already implemented in genesis.py)

**Effort:** 1-2 weeks. **Impact:** Faster peer bootstrap, reduced HuggingFace bandwidth.

---

## 7. Encryption Overhead Analysis

### How Encryption Currently Works

From `peer/crypto.py`, three encryption levels exist:

| Level | Onion layers | Padding | Operations per activation |
|-------|-------------|---------|---------------------------|
| `standard` | 1 | 0 bytes | 1x ECDH + 1x AES-GCM |
| `enhanced` | 2 | 32 bytes | 2x ECDH + 2x AES-GCM |
| `maximum` | 3 | 96 bytes | 3x ECDH + 3x AES-GCM |

**What gets encrypted per forward pass:**
- The activation tensor: `ACTIVATION_SIZE = 64` floats = 256 bytes (in toy mode)
- In PyTorch mode with `hidden_size=896` (Qwen3.5-0.8B): 896 floats = 3,584 bytes
- Plus padding (0/32/96 bytes depending on level)
- Plus 5-byte header + 12-byte nonce + 16-byte GCM tag per layer

**Per-request crypto operations:**
1. `_encode_activation`: `struct.pack` 896 floats → ~3.6 KB payload
2. X25519 ECDH key exchange: 1 scalar multiplication → ~0.1 ms
3. HKDF key derivation: SHA-256 based → ~0.01 ms
4. AES-256-GCM encrypt: ~3.6 KB payload → ~0.01 ms (hardware-accelerated)
5. Base64 encode for gRPC transport: ~0.01 ms

**Total crypto overhead per activation: ~0.15 ms at `standard` level.**

### Overhead vs. Inference Latency

| Component | Time | % of total |
|-----------|------|------------|
| Model forward pass (MPS, 0.8B) | ~770 ms | 99.7% |
| gRPC serialisation + network | ~1-3 ms | 0.3% |
| **Encryption (standard)** | **~0.15 ms** | **0.02%** |
| Encryption (enhanced) | ~0.3 ms | 0.04% |
| Encryption (maximum) | ~0.5 ms | 0.06% |

**Encryption adds 0.02-0.06% to total latency.** It is completely negligible. The bottleneck is the model forward pass (99.7% of time), not crypto.

### Should Encryption Be Removed for Beta?

**No.** The overhead is negligible (0.15 ms out of 770 ms). Removing it would:
- Save almost zero latency
- Remove a key differentiator vs Petals and Exo (neither has activation encryption)
- Create security debt that's hard to add back later

**Recommendation:**
- Keep `standard` level as default for beta (1 onion layer, 0 padding)
- `enhanced` and `maximum` are also fine since they add < 0.5 ms
- The `standard` level already provides ECDH key exchange + AES-GCM — that's strong encryption for ~0.15 ms overhead

### How to Make Inference Faster

The encryption is not the problem. The real performance bottlenecks are:

| Bottleneck | Current cost | Fix | Improvement |
|------------|-------------|-----|-------------|
| No model warmup | +34 s TTFT | Add warmup in `__init__` | 34 s → 1-3 s |
| PyTorch on MPS (eager mode) | 1.3 tok/s | Switch to MLX backend | 1.3 → 100-200 tok/s |
| No request coalescing | Linear degradation | Batching queue in peer | Near-linear scaling |
| No layer sharding | Can't run 14B+ | Activate existing layer code | Run 72B models |
| gRPC overhead (protobuf) | ~1-3 ms | Keep; negligible | N/A |
| Encryption | ~0.15 ms | Keep; negligible | N/A |

---

## 8. Beta Launch Roadmap

### Success Criteria

1. **Faster inference than Petals** (>6 tok/s on comparable model) ✓ achievable with MLX
2. **Faster inference than Exo** on heterogeneous internet hardware ✓ different topology
3. **Robust P2P network** with automatic model selection
4. **Initial Qwen 3.5 family support** (0.8B → 7B, with path to 72B)
5. **Democratic** — anyone can contribute and earn credits

### Implementation Phases

#### Phase 0: Quick Wins (Week 1)
- [ ] Add warmup call in `PyTorchRuntime.__init__` (TTFT: 34s → 1-3s)
- [ ] Set `default_model` to auto-select best available
- [ ] Raise `timeout_ms` default from 500 → 5000
- [ ] Add `--warmup-on-start` CLI flag to peer server

#### Phase 1: MLX Backend (Weeks 2-3)
- [ ] Implement `MLXRuntime` class in `peer/mlx_runtime.py`
- [ ] Add `"mlx"` to `--runtime-backend` choices
- [ ] Add `mlx` and `mlx-lm` to optional dependencies
- [ ] Warmup call in MLX runtime
- [ ] Test with Qwen3.5-0.8B, 1.5B, 4B, 7B

**After Phase 1: TPS should be 100-200 tok/s on Apple Silicon, beating both Petals and Exo for single-node.**

#### Phase 2: Auto-Scaling + Model Selection (Week 3-4)
- [ ] Implement `AutoPromoter` in `coordinator/auto_promoter.py`
- [ ] Wire into `CoordinatorEngine` — `default_model` becomes dynamic
- [ ] Redundancy ratio = 3x (configurable via `--redundancy-ratio`)
- [ ] Add Qwen 3.5 family to `models.catalog.json` (0.8B through 72B)
- [ ] `list_models()` response includes `best_available_model` field

#### Phase 3: Layer Sharding Activation (Weeks 4-6)
**Key discovery:** `PyTorchRuntime` already has layer-range infrastructure!
```python
# model_shard.py already does:
self.layer_indices = self._resolve_layer_indices(
    total_layers=self.total_layers,
    shard_index=max(0, int(config.shard_index)),
    total_shards=max(1, int(config.total_shards)),
)
self._selected_layers = [self._blocks[idx] for idx in self.layer_indices]
```
This means `--shard-index 0 --total-shards 4` on a 32-layer model already selects layers 0-7. What's missing:
- [ ] DHT announces `layer_start` and `layer_end` per peer
- [ ] Coordinator assembles pipelines that cover the full layer range
- [ ] `InferenceChain` validates layer coverage before routing
- [ ] Hidden state (not just activation) passed between stages
- [ ] Shard-aware model downloads (download only needed weight files)

#### Phase 4: NF4 Quantization + Request Coalescing (Weeks 5-7)
- [ ] Add NF4 quantization via bitsandbytes (`--quantization nf4`)
- [ ] Implement `BatchingQueue` in `peer/server.py`
- [ ] Measure throughput scaling with concurrent users

#### Phase 5: Peer-to-Peer Model Distribution (Weeks 6-8)
- [ ] Peers announce cached models to DHT
- [ ] New peers query DHT for nearby cache sources
- [ ] HTTP range-request download from peer (fallback: HuggingFace)
- [ ] SHA-256 integrity verification (reuse genesis.py piece digests)

#### Phase 6: Polish + Launch (Weeks 8-10)
- [ ] Add Ollama API compatibility layer
- [ ] Dijkstra path routing in coordinator
- [ ] Add 2-3 additional bootstrap nodes (different cloud providers)
- [ ] Load testing: 100 concurrent users
- [ ] Documentation for peer operators
- [ ] Public beta announcement

### Post-Beta Roadmap
- Hivemind DHT migration (v1.0)
- Tensor parallelism (Exo-style, for same-LAN clusters)
- RDMA over Thunderbolt 5
- On-chain HYDRA token contract (Arbitrum/Base)
- DLPack integration for multi-backend peers

---

## Summary of Key Decisions

| Decision | Answer | Rationale |
|----------|--------|-----------|
| **Redundancy ratio for promotion** | 3x effective (capability-filtered) | Only counts peers with sufficient VRAM; hysteresis band 1.5x-3.0x |
| **User model selection** | Auto by default, manual override | Maximises quality; power users can pin |
| **Torrent for model distribution** | HTTP peer-cache for beta; defer BitTorrent | Simpler, sufficient at beta scale |
| **Remove encryption for beta?** | **No** — keep `standard` level | 0.15 ms overhead = 0.02% of latency; negligible |
| **DLPack integration** | Defer to post-launch | Solves same-machine problem; beta bottleneck is network |
| **Hivemind DHT** | Defer to v1.0 | Current HTTP DHT works for <1000 peers |
| **Primary inference backend** | MLX (Apple Silicon), llama.cpp (Linux) | 50-150x faster than pytorch_auto |
| **Layer sharding** | Activate existing code in Phase 3 | Infrastructure already exists in model_shard.py! |
| **Best tech to borrow from Exo** | MLX backend + shard-aware downloads | Biggest TPS gain + faster peer bootstrap |
| **Best tech to borrow from Petals** | Request coalescing + NF4 quantization | Multi-user scaling + lower VRAM requirement |

---

## Expected Performance After Beta Optimisations

| Metric | Current | After Phase 1 (MLX) | After Phase 3 (sharding) |
|--------|---------|---------------------|--------------------------|
| **TTFT** | 34 s | ~100-200 ms | ~200-500 ms (multi-hop) |
| **TPS (0.8B, single node)** | 1.3 | 100-200 | 100-200 |
| **TPS (7B, single node, NF4)** | N/A | 30-80 | 30-80 |
| **TPS (72B, 8-node sharded)** | Impossible | Impossible | 10-30 (estimated) |
| **Max model size** | ~7B | ~7B | Unlimited |
| **Concurrent users (10)** | 10x slower | ~2x slower (coalescing) | ~2x slower |

These numbers would make OpenHydra **faster than Petals** (6 tok/s on 70B) and **competitive with Exo** on LAN while also working across the internet — something Exo cannot do.
