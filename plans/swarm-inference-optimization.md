# Swarm Inference Optimization — Research & Implementation Plan

**Date**: 2026-04-04 (implemented 2026-04-04 → 2026-04-06)
**Status**: **COMPLETE** — all P0/P1/P2 techniques implemented and tested on 5-node WAN pipeline
**Goal**: Minimize latency and maximize TPS in Swarm Mode
**Current baseline**: 20 TPS single-peer, ~3 TPS multi-peer (pipeline latency dominated)
**Post-optimization**: 0.43 TPS on 5-stage WAN (limited by sequential round-trips); local: 75 TPS

---

## The Core Problem

In Swarm Mode with N pipeline stages, each token requires a full sequential round-trip:

```
Token T generation:
  Stage 1 (peer A) → network → Stage 2 (peer B) → network → ... → Stage N (peer N)
  Total: N × compute_time + (N-1) × network_latency

For 4 peers at 50ms network + 30ms compute per stage:
  Per token: 4×30 + 3×50 = 270ms → 3.7 tok/s
```

The pipeline is idle 60% of the time (network stalls). Every technique below attacks this idle time.

---

## Priority 0 — Implement First (highest impact, proven feasibility)

### P0-A: Decentralized Speculative Decoding (DSD)

**Paper**: [arxiv.org/abs/2511.11733](https://arxiv.org/abs/2511.11733) (Nov 2025)
**Expected impact**: 2-3x TPS

**Key insight**: During the network stall between pipeline stages, run a small local draft model to generate K candidate tokens. Then verify all K tokens in a single pipeline pass instead of K separate passes.

**How it works in OpenHydra**:
```
Without DSD (current):
  Token 1: peer1 → peer2 → peer3 → peer4  (270ms)
  Token 2: peer1 → peer2 → peer3 → peer4  (270ms)
  Token 3: peer1 → peer2 → peer3 → peer4  (270ms)
  Total for 3 tokens: 810ms

With DSD:
  Draft model (local, 0.3B): generates tokens 1,2,3 in ~15ms
  Verify all 3: peer1 → peer2 → peer3 → peer4 (single pass, ~280ms)
  Accept 2 of 3 tokens on average
  Total for 2 tokens: 295ms → 6.8 tok/s (vs 3.7)
```

**Implementation plan**:
1. Add `--draft-model` CLI flag to `coordinator/node.py` (default: none)
2. Load a small MLX model (Qwen3.5-0.3B or 0.8B-4bit) as the draft model in `LocalInferenceEngine`
3. New `coordinator/speculative_swarm.py`:
   - `draft(prompt, k=5)` → generate K candidate tokens locally
   - `verify(candidates, pipeline)` → single pipeline pass, batch-verify
   - `accept(draft_tokens, verified_tokens)` → determine accepted prefix
4. Modify `InferenceService.infer()` to use spec decode loop when draft model available
5. Adaptive K: increase K when acceptance rate > 80%, decrease when < 50%

**Risk**: Draft model quality varies. Mitigation: DSI paper proves speculative decoding is provably never slower than non-speculative, even with weak drafters.

---

### P0-B: INT8 Activation Compression on Wire

**Paper**: Petals (2022), PALU (ICLR 2025, [arxiv.org/abs/2407.21118](https://arxiv.org/abs/2407.21118))
**Expected impact**: 2x bandwidth savings → 1.3-2x TPS on bandwidth-constrained links

**Key insight**: Hidden state activations transferred between pipeline stages can be quantized to INT8 with <0.1% perplexity loss. This halves the bytes per gRPC transfer.

**How it works in OpenHydra**:
```
Current: 4096-dim hidden state × fp32 = 16,384 bytes per activation transfer
With INT8: 4096-dim × int8 + 256 bytes (scale factors) = 4,352 bytes
Savings: 73% bandwidth reduction per hop
```

**Implementation plan**:
1. New `peer/activation_codec.py`:
   - `quantize_activation(tensor: list[float]) -> bytes` — per-channel INT8 with scale factors
   - `dequantize_activation(data: bytes) -> list[float]` — reconstruct fp32
2. Add `compression_codec = "int8_activation"` to `ForwardRequest` protobuf
3. Peer server: quantize output activation before gRPC response
4. Chain: dequantize received activation before passing to next stage
5. Backwards compatible: peers advertise codec support in DHT announcement

**Risk**: Minimal — Petals has shipped this since 2022. Well-proven technique.

---

## Priority 1 — Implement Second (high impact, moderate effort)

### P1-A: SpecPipe — Fill Pipeline Bubbles with Speculation

**Paper**: [arxiv.org/abs/2504.04104](https://arxiv.org/abs/2504.04104) (Apr 2025)
**Expected impact**: 3-5x TPS

**Key insight**: While stage N processes token T, stages 1..(N-1) are idle. Fill them with speculative future tokens from a dynamic token tree. One verified token emerges per pipeline step at steady state.

```
Without SpecPipe:          With SpecPipe:
Stage 1: [T1][ ][ ][ ]    Stage 1: [T1][T2'][T3'][T4']
Stage 2: [ ][T1][ ][ ]    Stage 2: [ ][T1][T2'][T3']
Stage 3: [ ][ ][T1][ ]    Stage 3: [ ][ ][T1][T2']
Stage 4: [ ][ ][ ][T1]    Stage 4: [ ][ ][ ][T1]
         ↑ 4 cycles for 1 token     ↑ 4 cycles for 1 verified + 3 speculative
```

**Implementation plan**:
1. Extend DSD (P0-A) with pipeline-aware scheduling
2. New `coordinator/pipeline_scheduler.py`:
   - Token tree management (expand/prune based on acceptance rate)
   - Stage assignment: map speculative tokens to idle stages
   - Verification: compare speculative output with verified output at each stage
3. Modify `chain.py` to support concurrent multi-token pipeline execution
4. Requires streaming gRPC (already have `ForwardStream` RPC defined in proto)

**Dependency**: P0-A (DSD) must be implemented first. SpecPipe extends it.

---

### P1-B: Chunked Prefill (Sarathi-Serve)

**Paper**: [arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310) (OSDI 2024)
**Expected impact**: 2-3x throughput, significant TTFT reduction

**Key insight**: Split prefill into chunks and interleave with ongoing decode steps. Prevents prefill from blocking the pipeline for other requests.

**Implementation plan**:
1. Modify `MLXRuntime.forward()` to accept `prefill_chunk_size` parameter
2. Coordinator splits long prompts into chunks of `chunk_size` tokens
3. Each chunk is processed as a pipeline micro-batch
4. Decode requests from other sessions interleave between chunks
5. Stall-free: no request ever waits for another's prefill to complete

---

### P1-C: Adaptive Peer Selection with RTT Tracking

**Expected impact**: 1.5-2x tail latency reduction

**Implementation plan**:
1. Add `_peer_rtt_history: dict[str, deque[float]]` to `DiscoveryService`
2. After each gRPC Forward(), record `(peer_id, latency_ms)` in the history
3. Compute exponentially-weighted moving average RTT per peer
4. In `PipelineService._select_pipeline()`, weight peer selection by inverse RTT
5. Peers with consistently high RTT (>2x median) are deprioritized
6. Already have `_next_hop_rtts` dict in `peer/server.py` — wire it into pipeline assembly

---

## Priority 2 — Implement Third (significant impact, higher effort)

### P2-A: PALU Low-Rank KV Cache Compression

**Paper**: [arxiv.org/abs/2407.21118](https://arxiv.org/abs/2407.21118) (ICLR 2025)
**Expected impact**: 5-10x wire compression

Decompose projection matrices into low-rank factors. Cache compressed intermediate states. Reconstruct on the fly. Combined with INT8, achieves 11.4x compression.

### P2-B: TOPLOC Hash Verification

**Source**: [primeintellect.ai/blog/toploc](https://www.primeintellect.ai/blog/toploc)
**Expected impact**: Eliminate verification overhead entirely

Replace expensive redundant execution with locality-sensitive hash signatures. Each peer includes an LSH hash of its intermediate activations. The coordinator verifies hashes match expected distributions without re-running the computation.

### P2-C: Coordinator-Level Continuous Batching

**Expected impact**: Nx throughput for N concurrent clients

Extend `BatchingQueue` to the coordinator level. Multiple client requests share a single pipeline pass, amortizing the N-hop cost across all requests in the batch.

---

## Priority 3 — Research Phase (high effort, transformative potential)

### P3-A: Sequence Parallelism for Prefill

**Paper**: [arxiv.org/abs/2411.01783](https://arxiv.org/abs/2411.01783) (Meta, Nov 2024)

Split the sequence dimension across peers using ring attention. 2-4x TTFT reduction for long prompts. Requires reasonably fast interconnect for the ring communication pattern.

### P3-B: Parallax-Style Two-Phase Scheduling

**Paper**: [arxiv.org/abs/2509.26182](https://arxiv.org/abs/2509.26182) (Gradient Network, Oct 2025)

Optimal layer placement across heterogeneous peers using a two-phase scheduler. Profile each peer's compute capability, then solve for the layer-to-peer assignment that minimizes end-to-end latency.

---

## Projected TPS Gains (Cumulative)

```
Current Swarm Mode (single peer, 8GB M1):    20 TPS
Current Swarm Mode (4-peer pipeline, est):   ~3 TPS

After P0 (DSD + INT8 compression):           ~8-12 TPS
After P1 (SpecPipe + chunked prefill):       ~20-30 TPS
After P2 (PALU + TOPLOC + batching):         ~40-60 TPS

Goal for production multi-peer:              >25 TPS sustained
```

---

## Key Papers Reference

| Paper | Year | Venue | Link |
|-------|------|-------|------|
| DSD: Decentralized Speculative Decoding | 2025 | arXiv | [2511.11733](https://arxiv.org/abs/2511.11733) |
| SpecPipe | 2025 | arXiv | [2504.04104](https://arxiv.org/abs/2504.04104) |
| DSI: Distributed Speculative Inference | 2025 | ICLR | [2405.14105](https://arxiv.org/abs/2405.14105) |
| PipeSpec | 2025 | ACL Findings | [2505.01572](https://arxiv.org/abs/2505.01572) |
| FlowSpec | 2025 | arXiv | [2507.02620](https://arxiv.org/abs/2507.02620) |
| Sarathi-Serve | 2024 | OSDI | [2403.02310](https://arxiv.org/abs/2403.02310) |
| PALU: Low-Rank KV Compression | 2025 | ICLR | [2407.21118](https://arxiv.org/abs/2407.21118) |
| KVQuant | 2024 | NeurIPS | [link](https://www.stat.berkeley.edu/~mmahoney/pubs/neurips-2024-kvquant.pdf) |
| Parallax (Gradient Network) | 2025 | arXiv | [2509.26182](https://arxiv.org/abs/2509.26182) |
| Petals | 2022/2024 | NeurIPS | [2209.01188](https://arxiv.org/abs/2209.01188) |
| Context Parallelism (Meta) | 2024 | arXiv | [2411.01783](https://arxiv.org/abs/2411.01783) |
| LoongServe | 2024 | SOSP | [2404.09526](https://arxiv.org/abs/2404.09526) |
| TOPLOC | 2025 | Prime Intellect | [blog](https://www.primeintellect.ai/blog/toploc) |
| Parallel Token Prediction | 2026 | ICLR | [2512.21323](https://arxiv.org/abs/2512.21323) |
| DynamoLLM | 2025 | HPCA | [2408.00741](https://arxiv.org/abs/2408.00741) |
| BucketServe | 2025 | arXiv | [2507.17120](https://arxiv.org/abs/2507.17120) |
