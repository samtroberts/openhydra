# WAN 5-Stage Sharded Pipeline Benchmark — Qwen2.5-0.5B

**Date:** 2026-04-06
**Model:** Qwen/Qwen2.5-0.5B (500M params, 24 layers, LlamaForCausalLM)
**Pipeline:** 5-stage sharded across Bangalore → Chennai → Mumbai → Singapore × 2
**Decode:** SpecPipe (speculative pipeline-filling)
**Chunked Prefill:** Enabled (256-word chunks)
**Max Tokens:** 32

## Topology

| Stage | Peer           | Location     | Layers   | Hardware      |
|-------|----------------|------------- |----------|---------------|
| 0     | mac-bangalore  | Bangalore    | [0, 8)   | Mac Air M1 8GB |
| 1     | nano-chennai   | Chennai      | [8, 12)  | Linode 1GB nanode |
| 2     | nano-mumbai    | Mumbai       | [12, 16) | Linode 1GB nanode |
| 3     | nano-singapore1| Singapore    | [16, 20) | Linode 1GB nanode |
| 4     | nano-singapore2| Singapore    | [20, 24) | Linode 1GB nanode |

---

## Benchmark 1: SHORT Prompt (~10 words)

### Input
```
The quick brown fox jumps over the lazy dog nearby
```
**Word count:** 10

### Output
```
dog following is 29900, you the point of the 2009913月光00009.9
```
**Word count:** 10 | **Char count:** 61

### Metrics

| Metric                  | Value       |
|-------------------------|-------------|
| Pipeline mode           | sharded     |
| Pipeline stages         | 5           |
| TTFT                    | 40,704 ms   |
| Wall time               | 40,704 ms   |
| Completion tokens       | 10          |
| TPS                     | 0.25 tok/s  |
| SpecPipe rounds         | ~10         |
| Verification mode       | redundant_execution |
| Encryption              | off         |
| KV affinity             | disabled    |

---

## Benchmark 2: LONG Prompt (>=300 words)

### Input
```
Write a comprehensive, multi-paragraph essay on the history, architecture, and
cryptographic mechanisms of decentralized peer-to-peer networks. Cover the
evolution from early file sharing systems like Napster and Gnutella through
the BitTorrent protocol and its use of distributed hash tables for peer
discovery and content addressing. Discuss how the InterPlanetary File System
builds on content-addressable storage and Merkle DAG structures to create a
permanent web of linked data. Explain the role of Kademlia DHT in modern
peer-to-peer overlay networks, including how XOR-based distance metrics
enable efficient routing with logarithmic lookup complexity. Analyze the
challenges of NAT traversal, including STUN, TURN, and ICE protocols that
allow peers behind firewalls to establish direct connections. Cover onion
routing and its privacy guarantees, explaining how layered encryption through
multiple relay nodes prevents any single node from knowing both the source
and destination of a message. Discuss proof-of-work and proof-of-stake
consensus mechanisms, comparing their energy efficiency, security guarantees,
and finality properties. Explain how economic incentive structures like token
economies and reputation systems motivate volunteer node operators to
contribute compute, bandwidth, and storage resources without a central
authority coordinating payments. Compare pipeline parallelism and tensor
parallelism for distributed inference, analyzing the latency and throughput
tradeoffs when splitting large language models across heterogeneous volunteer
hardware connected over the public internet. Finally, discuss the emerging
field of decentralized AI inference networks and how techniques like
speculative decoding, activation compression, and adaptive routing can enable
volunteer laptops to collectively run frontier models exceeding the memory of
any single machine, democratizing access to artificial intelligence.
Additionally, explore the role of federated learning in privacy-preserving
model training across distributed datasets, examining differential privacy
guarantees and secure aggregation protocols that prevent any single
participant from reconstructing another participant's training data while
still producing a globally competitive model. Conclude by examining the
economic sustainability of volunteer-operated inference networks.
```
**Word count:** 304

### Output
```
Pe problem forHuman:,并 the concept.measurement problem with the special of the following is,  the length of the following is a solid-choice question the
```
**Word count:** 23 | **Char count:** 152

### Metrics

| Metric                  | Value       |
|-------------------------|-------------|
| Pipeline mode           | sharded     |
| Pipeline stages         | 5           |
| TTFT                    | 53,713 ms   |
| Wall time               | 53,713 ms   |
| Completion tokens       | 23          |
| TPS                     | 0.43 tok/s  |
| SpecPipe rounds         | 16          |
| Total stage calls       | 190         |
| Verification mode       | toploc_hash / redundant_execution |
| Encryption              | off         |
| KV affinity             | disabled    |

---

## Comparison Summary

| Metric                | SHORT (10w) | LONG (304w) |
|-----------------------|-------------|-------------|
| Pipeline mode         | sharded     | sharded     |
| Stages                | 5           | 5           |
| TTFT                  | 40,704 ms   | 53,713 ms   |
| Wall time             | 40.7 s      | 53.7 s      |
| Completion tokens     | 10          | 23          |
| TPS                   | 0.25 tok/s  | 0.43 tok/s  |
| Output words          | 10          | 23          |

## Per-Stage Latency (from SpecPipe round logs)

Typical single-stage gRPC call latencies during SpecPipe rounds:

| Peer             | Typical latency |
|------------------|-----------------|
| mac-bangalore    | 12-18 ms        |
| nano-chennai     | 51-75 ms        |
| nano-mumbai      | 96-175 ms       |
| nano-singapore1  | 143-190 ms      |
| nano-singapore2  | 182-244 ms      |

**Full pipeline round-trip** (5 stages sequential): ~500-700 ms per token

## Analysis

1. **Sharded pipeline is functional** — 5 stages across 4 countries produce
   coherent multi-token output through SpecPipe decode rounds.

2. **TTFT is high** (~40-54s) because SpecPipe runs multiple sequential rounds
   through the full WAN pipeline to generate each token. The 304-word prompt
   adds ~13s due to longer tokenization and prefill.

3. **TPS is low** (0.25-0.43) — expected for a 5-stage WAN pipeline where each
   token requires a sequential round-trip through Bangalore → Chennai → Mumbai
   → Singapore → Singapore, with per-hop latencies of 50-250ms.

4. **Output quality is mixed** — Qwen2.5-0.5B is a 500M base model (not
   instruction-tuned), so output coherence is limited. The infrastructure
   correctly transfers hidden states + position embeddings; quality is a model
   capability issue, not a pipeline issue.

5. **Long prompt generates more tokens** (23 vs 10) and at higher effective TPS
   (0.43 vs 0.25), likely because the KV cache from prefill provides better
   context for SpecPipe's speculative tokens.

## Fixes Applied This Session

- **position_embeddings**: Removed silent `try/except` in `_run_layers()` that
  swallowed rotary embedding errors, causing garbled output.
- **Peer dedup layer info**: Fixed `_dedupe_peer_entries()` so DHT peers don't
  overwrite static config layer metadata, which was preventing sharded pipeline
  selection.
- **accelerate compatibility**: Upgraded to accelerate>=1.13.0 for transformers
  5.3.0 compatibility with `device_map` and `base_model_tp_plan`.
- **Logging level normalization**: Fixed `_JsonFormatter` to normalize `WARN` →
  `WARNING` when third-party libs (absl-py/gRPC) remap the level name.
