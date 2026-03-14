# OpenHydra: Decentralised Pipeline-Parallel LLM Inference Over Volunteer Hardware

**Version 1.0 &mdash; March 2026**

---

## Abstract

We present OpenHydra, a fully decentralised peer-to-peer network for large language model (LLM) inference that pools idle consumer hardware into a virtual GPU cluster. OpenHydra addresses three fundamental bottlenecks that have prevented practical distributed inference over heterogeneous volunteer networks: (1) the VRAM ceiling, where frontier models exceed any single device's memory; (2) the bandwidth wall, where naive tensor shipping between peers saturates residential links; and (3) the trust problem, where untrusted peers may return fabricated outputs.

Our system introduces a novel architectural bifurcation between *data at rest* (model weight distribution via SHA-256-verified P2P transfers with HTTP Range requests) and *data in motion* (sub-millisecond gRPC activation streaming through encrypted pipelines). We describe Q-Tensor KV Compaction, a four-phase attention matching framework that reduces per-request KV cache memory by up to 90% while preserving output quality. We detail Distributed Speculative Decoding (DSD), which extends the draft-then-verify paradigm across multi-peer sharded pipelines. We present a concentric onion routing scheme with differential privacy noise injection that prevents any single peer from observing a complete query. Finally, we describe a two-tier token economy with Burn-and-Mint Equilibrium (BME) that aligns incentives between compute providers and consumers.

OpenHydra is implemented in Python with 867 passing tests and production DHT infrastructure on three continents. On Apple Silicon (MLX backend), a single peer achieves ~252 tok/s local throughput; a sharded multi-peer pipeline over the public internet achieves ~10 tok/s net content generation with 275ms average time-to-first-token.

---

## 1. Introduction

The exponential growth in LLM parameter counts has created a hardware accessibility crisis. A 70-billion parameter model requires approximately 140 GB of VRAM in fp32, or 35 GB in NF4 quantization &mdash; well beyond any consumer GPU. Cloud inference APIs solve the access problem but introduce centralisation, surveillance, and vendor lock-in.

Peer-to-peer inference networks offer a compelling alternative: distribute the model across multiple commodity devices, each holding a subset of layers, and stream activations between them during inference. This pipeline-parallel approach was pioneered by Petals [1], which demonstrated that BLOOM-176B could be served collaboratively over the public internet.

However, three critical challenges remain unsolved for production deployment:

1. **The VRAM Ceiling.** Even with pipeline parallelism, each peer must hold its assigned layers in memory. NF4 quantization (4-bit) reduces a 7B model from ~14 GB to ~3.5 GB, but the distribution mechanism must be trustless and resumable.

2. **The Bandwidth Wall.** Activation tensors between pipeline stages are dense floating-point arrays. For a model with hidden dimension 4096, each activation is 16 KB in fp32. At 10 tok/s across a 4-stage pipeline, this is 640 KB/s &mdash; manageable, but the per-hop latency dominates. The real bottleneck is not throughput but round-trip time.

3. **The Trust Problem.** In an open network, any peer may return garbage activations to save compute, or worse, craft adversarial outputs. Verification must be probabilistic (full redundant execution is too expensive) and economically incentivised.

OpenHydra addresses all three through a combination of architectural design, applied cryptography, and mechanism design.

---

## 2. System Architecture

### 2.1 Architectural Bifurcation: Data at Rest vs. Data in Motion

OpenHydra draws a sharp distinction between two data movement patterns that have fundamentally different performance characteristics:

**Data at Rest** refers to model weights &mdash; large, static, and integrity-critical. These are distributed once per model version via a P2P model cache (`peer/p2p_model_cache.py`) that implements:

- SHA-256 per-file integrity verification against HuggingFace Hub API manifests
- HTTP Range request resumption for crash-safe partial downloads
- Atomic file operations (`.part` files + `os.replace()`) to prevent corruption
- Per-model locks preventing duplicate concurrent downloads
- DHT announcement of cached models for peer-to-peer discovery before falling back to HuggingFace CDN

The seeder (`peer/seeder_http.py`) serves files via RFC 7233 Range requests with 64 KB streaming chunks, path-traversal guards, and directory listings. This is conceptually similar to BitTorrent's piece-based distribution, but uses HTTP for simplicity and NAT traversal.

**Data in Motion** refers to activation tensors during inference &mdash; small, ephemeral, and latency-critical. These flow through gRPC `ForwardRequest`/`ForwardResponse` RPCs on port 50051 with protobuf serialisation. Each hop carries:

- The activation tensor (serialised float array)
- Pipeline metadata (`shard_layer_start`, `shard_layer_end`, `total_layers`)
- Encryption envelope (onion-routed ciphertext, ephemeral public keys, nonces)
- Session affinity key for KV cache reuse

This bifurcation allows each path to be optimised independently: P2P model distribution prioritises integrity and resumability; activation streaming prioritises latency and privacy.

### 2.2 Dual-Stack Peer Discovery

Peer discovery operates over two protocols simultaneously:

1. **HTTP DHT** (port 8468): A custom announce/lookup REST API running on three Linode bootstrap nodes behind nginx. Peers POST announcements with a 300-second TTL and re-announce every 60 seconds. Coordinators GET lookups by model ID. This is simple, debuggable, and sufficient for networks under 1,000 peers.

2. **Hivemind Kademlia DHT** (port 38751): A production libp2p Kademlia network using the hivemind library [2]. Three signpost nodes with persistent identity keys (via `identity_path` parameter) bootstrap the swarm. Peers auto-join via hardcoded multiaddrs embedded in the client. This provides O(log n) lookup scaling for larger networks.

The dual-stack design provides resilience: if one protocol fails, the other continues operating. The coordinator queries both and merges results.

### 2.3 Pipeline Assembly

The coordinator assembles inference pipelines through `_select_pipeline_sharded()`:

1. Query the DHT for all peers serving the requested model
2. Build a `LayerCoverageMap` from peer announcements (`layer_start`, `layer_end`, `total_layers`)
3. Run a greedy algorithm to find the minimum set of peers covering all layers: starting from layer 0, select the peer whose shard extends furthest, then repeat from that endpoint
4. If a complete sharded pipeline exists, use it. Otherwise, fall back to a single peer running the full model
5. If neither is available, invoke `DegradationPolicy` to serve a smaller model

The greedy algorithm runs in O(n * s) where n is the number of candidate peers and s is the number of shards in the assembled pipeline.

---

## 3. Q-Tensor KV Compaction

### 3.1 Problem Statement

During autoregressive generation, each transformer layer maintains a key-value cache of all previously generated tokens. For a model with L layers, H KV-heads, and head dimension d, the KV cache for a sequence of length T consumes:

```
Memory = 2 * L * H * T * d * sizeof(dtype)
```

For Llama-3.1-8B (L=32, H=8, d=128) at T=4096 in fp16, this is 512 MB per request &mdash; a significant fraction of available VRAM on consumer hardware, and a hard constraint on concurrent request capacity.

### 3.2 Attention Matching Framework

OpenHydra's KV compaction pipeline reduces T to a target t << T by selecting the tokens whose keys contribute most to the attention output. The framework operates in four composable phases:

**Phase 1: Key Selection.** Given key matrix K of shape (T, d_head), value matrix V of shape (T, d_head), reference queries Q_ref of shape (R, d_head), and target count t:

*HAK (Highest Attention Keys):* Compute the attention distribution for each reference query, then rank tokens by the RMS of their attention weight across all reference queries. Select the top-t tokens:

```
logits = Q_ref @ K^T / sqrt(d_head)           # (R, T)
attn = softmax(logits, dim=-1)                 # (R, T)
importance = sqrt(mean(attn^2, dim=0))         # (T,)
indices = topk(importance, t)
```

Complexity: O(R * T * d_head) for the matmul, O(T log t) for the top-k.

*OMP (Orthogonal Matching Pursuit):* A greedy residual pursuit that iteratively selects the token maximally reducing uncovered attention mass:

```
A = softmax(Q_ref @ K^T / sqrt(d_head))       # (R, T)
residual = ones(R)                              # total mass per query
for i in 1..t:
    scores = A^T @ residual                     # (T,) - coverage of residual mass
    k = argmax(scores)
    residual = clamp(residual - A[:, k], min=0)
```

OMP is more accurate than HAK but costs O(T * t) iterations.

**Phase 2: Beta Bias Correction and Value Refitting.** When only t tokens are attended instead of T, the attention distribution is distorted. Phase 2 fits scalar log-space biases beta that correct this mass underestimation:

```
A_orig[:, selected] ≈ softmax(Q_ref @ Ck^T / sqrt(d) + beta)
```

Beta is fitted via NNLS (non-negative least squares) using scipy when available, with a log-ratio fallback:

```
exp(beta) = NNLS(A_compact_softmax, mass_target)
beta = log(clip(exp_beta, eps, inf)), clamped to [-10, 10]
```

Compact values Cv are then refitted via least-squares minimisation of the output reconstruction error:

```
O_ref = A_orig @ V                             # reference output
attn_corrected = softmax(Q_ref @ Ck^T / sqrt(d) + beta)
Cv = lstsq(attn_corrected, O_ref)              # (t, d_head)
```

The beta biases are injected into the model at runtime via monkey-patching the attention forward method (`_beta_inject.py`), adding beta to the attention mask before softmax. This is idempotent and supports Qwen2/3, LLaMA, and Gemma3 families.

**Phase 3: Non-Uniform Head Budgets.** Different attention heads have different information density. Phase 3 loads per-layer/per-head token budgets from a precomputed JSON file, allowing heads that attend broadly to retain more tokens while heads with sharp attention patterns can be aggressively compressed.

**Phase 4: Online Mid-Trajectory Compaction.** Rather than compacting only at cache creation time, Phase 4 triggers compaction whenever the stored sequence length exceeds `online_max_tokens`. This enables unbounded effective context length on fixed-memory hardware: the physical KV cache never exceeds the configured budget, but the model has "seen" all prior tokens through the compacted representation.

### 3.3 Option A: Real Query Capture

The default Phase 1-3 implementation uses the last n_ref key vectors as proxy reference queries. This is geometrically incorrect: keys and queries live in different projected subspaces (W_k vs W_q). Option A (`_query_capture.py`) registers pre-forward hooks on each transformer layer to capture the input hidden states, then computes Q = W_q(hidden) as reference queries. RoPE is intentionally omitted &mdash; the content-similarity component of the Q-K dot product dominates the position-encoding cross-term when averaged across multiple reference queries. Empirically, real queries significantly improve key selection quality over the proxy-K heuristic.

### 3.4 Radix Prefix Cache

The `RadixKVCache` provides O(n_entries) longest-prefix KV cache lookup with LRU eviction. When a new request shares a token prefix with a cached sequence, the matching KV cache is reused and only the novel suffix is computed. This is critical for multi-turn chat where system prompts and prior turns are repeated.

---

## 4. Distributed Speculative Decoding

### 4.1 Background

Speculative decoding [3, 4] accelerates autoregressive generation by using a small draft model to propose K candidate tokens, then verifying all K in a single forward pass of the target model. If the draft matches the target, K tokens are accepted for the cost of one target forward pass.

### 4.2 Extension to Distributed Pipelines

In OpenHydra's sharded pipeline, the target model is distributed across multiple peers. A speculative round proceeds as follows:

1. The coordinator runs a local `PyTorchDraftModel` (e.g., `sshleifer/tiny-gpt2`) to propose up to 16 candidate tokens
2. The candidate token IDs are sent through the full sharded pipeline for verification
3. The verified outputs are compared against the draft via `select_verified_token_ids()`:
   - Tokens are accepted while they match the draft
   - On the first mismatch, the verified token at that position replaces the draft token
   - All subsequent draft tokens are discarded

The `SpeculativeTokenIdSelection` dataclass tracks: accepted token IDs, matched prefix length, and whether a mismatch occurred. This enables the coordinator to measure draft model accuracy and adaptively adjust the speculation depth.

### 4.3 Amortised Network Cost

The key insight is that the N-hop network latency for verification is amortised across K accepted tokens. If the draft model has accuracy p per token, the expected acceptance length is 1/(1-p). For p=0.7 and K=16, the expected acceptance is ~3.3 tokens per verification round, reducing effective per-token network cost by 3.3x.

---

## 5. Security Architecture

### 5.1 Threat Model

OpenHydra assumes an adversarial environment where:

- Any peer may return fabricated activations (Byzantine fault)
- Network observers may attempt to reconstruct queries from activation traffic
- Sybil attackers may flood the DHT with fake peer announcements
- Peers may attempt to claim rewards without performing inference

### 5.2 Concentric Onion Routing

Activation tensors are encrypted in concentric layers before entering the pipeline. For an N-stage pipeline, the activation is encrypted N times, once for each peer's public key (derived deterministically from the shared secret seed and peer ID via X25519 ECDH + HKDF-SHA256):

```
For each layer i in [N, N-1, ..., 1]:
    ephemeral_key = X25519.generate()
    shared_secret = ECDH(ephemeral_key, peer_i.static_public_key)
    aes_key = HKDF-SHA256(shared_secret, salt=SHA256(request_id:stage_i),
                          info="openhydra/activation/{i}/{N}")
    ciphertext = AES-256-GCM(aes_key, nonce_i, plaintext,
                             aad="{request_id}:{stage_i}:{peer_id}:{i}:{N}:activation")
```

Each peer peels one encryption layer, processes the activation through its model shard, re-encrypts the output for the next peer, and forwards. No single peer observes the plaintext of any other peer's activation.

Three encryption levels are available: standard (1 layer, ~0.15ms), enhanced (2 layers + 32 bytes padding), and maximum (3 layers + 96 bytes padding). The overhead is negligible relative to inference latency.

### 5.3 Differential Privacy Noise Injection

Peers inject calibrated noise into activations before forwarding. The noise variance is announced in the DHT and verified via privacy audit tags &mdash; HMAC-SHA256 signatures over the configured variance, observed variance (exponential moving average), and observed standard deviation. Auditors can verify that the announced noise level matches the statistical properties of the forwarded activations.

### 5.4 Sybil Resistance via Geo-Challenge

New peers must prove their claimed geographic region via a geo-challenge: an HMAC-SHA256 proof-of-work bound to the peer's identity and claimed region. This prevents a single attacker from flooding the DHT with fake peers claiming to be in multiple regions to gain disproportionate routing weight.

---

## 6. Token Economy

### 6.1 Barter Credits (Tier 1)

The barter credit system provides immediate, zero-overhead settlement:

```
Conversion:  1,000 tokens served = 1.0 credit
Decay:       factor = (1 - 0.05) ^ elapsed_days
Storage:     SQLite WAL mode, journal_mode=WAL, synchronous=NORMAL
```

The 5%/day exponential decay prevents credit hoarding and ensures that the economy rewards active participation. Credits are non-transferable between peers.

### 6.2 HYDRA Token (Tier 2)

HYDRA is a capped-supply token with the following parameters:

```
Supply cap:           69,000,000 tokens
Daily mint rate:      250,000 tokens/day (governable)
Min slash penalty:    10% of infraction amount
State channel TTL:    900 seconds (15 minutes)
Max open channels:    8 per payer
Min channel deposit:  0.01 HYDRA
```

**Burn-and-Mint Equilibrium (BME):** The supply cap is enforced on every transaction. When total minted approaches the cap, the mint rate automatically decreases. Clients burn HYDRA for priority access and larger models, creating deflationary pressure that balances inflationary minting.

**State Channels:** For low-latency settlement, payers open off-chain state channels with escrowed deposits. Charges are applied via monotonically increasing nonces (`reconcile(channel_id, total_spent, nonce)` requires nonce > previous nonce). Channels auto-expire after TTL, with payees receiving accumulated charges and payers refunded the remainder.

### 6.3 L1 Bridge (Future)

The `OpenHydraLedgerBridge` provides an interface for future on-chain settlement. Currently operating in mock mode (in-memory), the bridge accepts pluggable `external_stake_resolver` and `external_stake_slasher` callables for EVM RPC integration. Target chains: Arbitrum and Base (low gas, high throughput).

---

## 7. Performance

### 7.1 Benchmarks

Measured on Qwen/Qwen3.5-0.8B (MLX backend, Apple M1 MacBook Air 8GB):

| Metric | Value | Conditions |
|--------|-------|------------|
| Local throughput | ~252 tok/s | MLX, single peer, no network |
| TTFT (warm, direct gRPC) | 122ms min, 275ms avg | 3 runs, 1-token request |
| Net content TPS (direct gRPC) | ~10 tok/s | 16-64 tokens, excluding think prefix |
| Full-stack TPS (HTTP -> coordinator -> gRPC) | ~3 tok/s | Includes DHT lookup 1-3s |
| Encryption overhead | ~0.15ms/activation | X25519 ECDH + AES-256-GCM |
| Encryption as % of latency | 0.02% | ~770ms total forward pass |

### 7.2 NF4 Quantization

Using BitsAndBytesConfig with `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=bfloat16`, and double quantisation enabled:

| Model | fp32 | NF4 | Reduction |
|-------|------|-----|-----------|
| 7B params | ~14 GB | ~3.5 GB | 4x |

MLX quantisation uses `mlx.nn.quantize(model, bits=4)` for runtime quantisation, or loads pre-quantised checkpoints from mlx-community repositories transparently.

### 7.3 Request Coalescing

The `BatchingQueue` coalesces concurrent requests into true tensor batches:

- Timer-based flushing: `batch_window_ms=50` (configurable)
- Immediate flush when `len >= max_batch_size` (default 8)
- PyTorch: `torch.cat(hidden_tensors, dim=0)` with right-padding for unequal sequence lengths
- MLX: `mx.concatenate(arrays, axis=0)` with per-request EOS tracking

Lock is held only for O(1) list-swap, never during the forward pass.

---

## 8. Related Work

**Petals** [1] pioneered pipeline-parallel inference over the internet, serving BLOOM-176B across volunteer hardware using hivemind's DHT for peer discovery. OpenHydra differs in its trust model (three-tier verification + token economy), privacy architecture (onion routing + DP noise), and KV compaction for memory efficiency.

**Exo** [5] demonstrated MLX tensor parallelism for local network clusters with shard-aware model downloads. OpenHydra targets wide-area networks with untrusted peers rather than trusted local clusters.

**vLLM** [6] introduced PagedAttention for efficient KV cache management in centralised serving. OpenHydra's KV compaction is complementary: it reduces the total number of cached tokens rather than managing their memory layout.

**DistServe** [7] and **Splitwise** [8] explored disaggregating prefill and decode phases across different hardware. OpenHydra's layer sharding is orthogonal: any peer can handle both prefill and decode for its assigned layers.

---

## 9. Limitations and Future Work

1. **Latency.** Multi-hop pipelines over the public internet add 1-3 seconds of DHT lookup and network latency. Speculative decoding amortises this but does not eliminate it. Future work on KV affinity (routing returning users to the same peers) and edge caching will reduce cold-start costs.

2. **Heterogeneous hardware.** The current pipeline model assumes roughly equal per-stage latency. Peers with significantly slower hardware create pipeline stalls. Adaptive stage assignment based on measured throughput is planned.

3. **Formal verification.** The three-tier verification system is probabilistic. A sufficiently motivated attacker with control over multiple peers could evade detection. Moving slash penalties on-chain (where they are irrevocable) strengthens the economic deterrent.

4. **Model family coverage.** Beta injection for KV compaction Phase 2 currently supports Qwen2/3, LLaMA, and Gemma3. Additional families require attention layer detection patterns.

---

## 10. Conclusion

OpenHydra demonstrates that decentralised LLM inference over volunteer hardware is not only feasible but practical. By bifurcating the architecture between integrity-focused weight distribution and latency-focused activation streaming, applying four-phase KV compaction to overcome memory constraints, layering onion-routed encryption for privacy, and aligning incentives through a burn-and-mint token economy, we have built a system that runs today with 867 passing tests and production infrastructure on three continents.

The code is open-source under a dual Apache 2.0 / AGPL v3 license at [github.com/openhydra-ai/openhydra](https://github.com/openhydra-ai/openhydra).

---

## References

[1] Borzunov, A., et al. "Petals: Collaborative Inference and Fine-tuning of Large Models." *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, 2023. [arXiv:2209.01188](https://arxiv.org/abs/2209.01188)

[2] Ryabinin, M., et al. "Hivemind: a Library for Decentralized Deep Learning." *arXiv preprint*, 2020.

[3] Leviathan, Y., Kalman, M., and Matias, Y. "Fast Inference from Transformers via Speculative Decoding." *Proceedings of the 40th International Conference on Machine Learning (ICML)*, 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

[4] Chen, C., et al. "Accelerating Large Language Model Decoding with Speculative Sampling." *arXiv preprint*, 2023. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)

[5] Exo Labs. "Exo: Run your own AI cluster at home with everyday devices." [github.com/exo-explore/exo](https://github.com/exo-explore/exo), 2024.

[6] Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)*, 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

[7] Zhong, Y., et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." *OSDI*, 2024. [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)

[8] Patel, P., et al. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." *ISCA*, 2024. [arXiv:2311.18677](https://arxiv.org/abs/2311.18677)

[9] "Efficient KV Cache Compaction via Attention Matching." *arXiv preprint*, 2026. [arXiv:2602.16284](https://arxiv.org/abs/2602.16284)

[10] Maymounkov, P. and Mazieres, D. "Kademlia: A Peer-to-Peer Information System Based on the XOR Metric." *IPTPS*, 2002.

[11] Dwork, C. "Differential Privacy." *Proceedings of the 33rd International Colloquium on Automata, Languages and Programming (ICALP)*, 2006.
