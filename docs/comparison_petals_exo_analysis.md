# OpenHydra vs Petals vs Exo — Comparative Code Analysis

## Overview

This analysis was conducted in March 2026 comparing the three leading open-source peer-to-peer inference frameworks: OpenHydra, Petals, and Exo. The comparison examines architectural decisions, code quality, test coverage, and feature completeness across each project's codebase. All observations are based on publicly available source code at the time of writing.

---

## 1. DHT / Peer Discovery

**OpenHydra**: Custom HTTP-based DHT with 3 geographically distributed bootstrap nodes (EU, US, AP). Uses `ThreadPoolExecutor` fan-out for parallel queries, TCP keepalive via `_TcpKeepAliveAdapter`, and a `requests.Session` connection pool. Cache TTL is 120s at the coordinator; peers re-announce every 60s with a 300s expiry. Simple and effective for the current scale target (<1000 peers).

**Petals**: Built on Hivemind's Kademlia DHT with libp2p transport. Uses pydantic-based `ServerInfo` tuple serialization and a `ModuleAnnouncerThread` that includes RTT ping measurements alongside peer metadata. The DHT is a mature, battle-tested component that has operated at scale in production.

**Exo**: Uses Rust libp2p with mDNS discovery, bridged to Python via PyO3. Network isolation is achieved through pre-shared keys. This architecture is limited to LAN discovery only — there is no WAN peer discovery mechanism.

**Winner**: Petals. Its Kademlia DHT is production-proven and supports WAN natively. OpenHydra's HTTP DHT is functional but simpler. Exo's LAN-only constraint is a fundamental limitation for distributed inference across the internet.

---

## 2. Routing / Pipeline Assembly

**OpenHydra**: Implements both greedy (`find_complete_pipeline`) and optimal (`find_optimal_pipeline` via Dijkstra) pipeline assembly in `coordinator/layer_coverage.py`. The `LayerCoverageMap` data structure tracks per-layer peer availability with O(n*s) greedy assembly. This module has 114 tests and is among the best-documented files across all three codebases.

**Petals**: Uses a 5-factor Dijkstra routing algorithm that considers RTT, KV cache cost, compute latency, server ping freshness, and allocation delay. The routing logic is more sophisticated than OpenHydra's but contains inline magic constants without documentation explaining their derivation.

**Exo**: Employs a `rustworkx` topology graph with largest-remainder layer allocation. The placement logic is pure functional, making it easy to reason about and test in isolation.

**Winner**: Tie. Petals has the most sophisticated routing heuristics. OpenHydra has significantly better test coverage and documentation for its routing logic. Exo's functional approach is clean but less feature-rich.

---

## 3. Inference Chain

**OpenHydra**: Per-request gRPC unary calls with deadline propagation through the pipeline. The main orchestration lives in `coordinator/engine.py`, which at 3244 lines qualifies as a god class with zero docstrings. Functional but difficult to maintain.

**Petals**: Persistent bidirectional gRPC streams with server-to-server push. Supports history replay on failover, allowing clients to recover from mid-pipeline server failures without restarting from scratch. This is the most network-efficient approach.

**Exo**: Event-sourced pub/sub architecture with a pure `apply()` state machine and explicit runner lifecycle management. The cleanest architectural separation of the three projects.

**Winner**: Exo for architectural cleanliness. Petals for stream efficiency and failover resilience. OpenHydra's god class is the weakest point in the entire codebase.

---

## 4. Verification / Trust

**OpenHydra**: Three-tier verification system:
- Tier 1: Mystery shopper (probabilistic re-execution via `AuditSampler`)
- Tier 2: Redundant execution with N-peer majority vote
- Tier 3: Bernoulli spot-check auditor with HMAC audit tags

Verification outcomes feed into a `reputation_score` used for peer ranking in pipeline assembly. This is the only project that addresses the fundamental trust problem in decentralized inference.

**Petals**: No verification system. Relies on trust and self-reported throughput metrics. A malicious peer can return arbitrary activations.

**Exo**: No verification system. Same trust assumptions as Petals.

**Winner**: OpenHydra, by a wide margin. Verification and trust are the defining competitive advantage of the project. Without verification, decentralized inference networks cannot safely serve production workloads from untrusted peers.

---

## 5. Batching / Coalescing

**OpenHydra**: `BatchingQueue` in `peer/batching.py` (197 lines) implements cross-request tensor batching with a timer-based flush and max-batch-size immediate flush. Uses `torch.cat` (PyTorch) and `mx.concatenate` (MLX) for true tensor batching across concurrent requests. Lock is held only for O(1) list-swap, never during the forward pass. Clean thread-safety contract with `concurrent.futures.Future` per request item.

**Petals**: `PrioritizedTaskPool` handles request scheduling but does not perform cross-request batching. This is an acknowledged non-goal in the Petals architecture.

**Exo**: `BatchGenerator` with multi-rank `mx_all_gather_tasks` synchronization. Designed for tensor-parallel batching across ranks rather than cross-request coalescing.

**Winner**: OpenHydra for cross-request batching. The `BatchingQueue` is compact, well-tested (11 tests), and solves a real throughput problem for peers serving multiple concurrent clients.

---

## 6. Model Loading / Quantization

**OpenHydra**: Supports NF4 via bitsandbytes (`BitsAndBytesConfig` with double quantization and bfloat16 compute) and MLX int4/int8 via `mlx.nn.quantize`. Includes `MlxWatchdog` for GPU hang protection. Graceful fallback to fp32 on non-CUDA hardware or missing bitsandbytes. Pre-quantized MLX checkpoints (e.g., mlx-community 4-bit models) load transparently without re-quantization.

**Petals**: NF4 + INT8 + tensor parallelism with a clean pipeline: freeze, tensor-parallelize, quantize, move to device. Deferred bitsandbytes import avoids load-time failures on machines without CUDA. The loading pipeline is the most organized of the three.

**Exo**: MLX int4/int8 with 15+ model-specific sharding strategies. The `auto_parallel.py` module (1397 lines) uses monkey-patching to inject parallelism into model classes, which is effective but fragile.

**Winner**: Petals for the cleanest loading pipeline. OpenHydra's watchdog and fallback handling are practical additions. Exo's monkey-patching approach is a maintenance risk.

---

## 7. Economy / Privacy (OpenHydra Unique)

These features are unique to OpenHydra and have no counterpart in Petals or Exo:

- **HYDRA token economy**: 69M supply cap, mint/burn/stake/slash mechanics, barter credits (1000 tokens = 1 credit, 5%/day decay), SQLite WAL storage with `threading.Lock` concurrency control. State channels for micro-payments.
- **Encryption**: X25519 ECDH key exchange + AES-GCM activation encryption. Measured overhead: ~0.15ms per activation (~0.02% of total inference latency).
- **Onion routing**: Layered encryption through pipeline stages prevents intermediate peers from reading activations in cleartext.
- **Differential privacy**: DP noise injection with HMAC audit tags for verifiable noise addition.
- **Identity**: Ed25519 key at `~/.openhydra/identity.key` (mode 0600) for peer authentication and message signing.

---

## 8. Test Coverage

| Project | Test Count | Test Files | Notes |
|---|---|---|---|
| OpenHydra | 876 | 70+ | Unit-heavy, all runnable without GPU or network |
| Exo | 259 | ~20 | Good coverage for core modules |
| Petals | 29 | ~5 | Integration-heavy, many require a live swarm |

OpenHydra's test suite is the most comprehensive. Key test modules include:
- `test_layer_coverage.py`: 85 tests (coverage map, greedy pipeline assembly, gap detection)
- `test_quantization_config.py`: 53 tests (alias normalization, NF4, MLX detection)
- `test_p2p_model_distribution.py`: 28 tests (seeder HTTP, Range requests, path traversal guards, SHA-256 verification)
- `test_sharded_grpc_pipeline.py`: 19 tests (mock gRPC handoff, end-to-end validation)
- `test_request_coalescing.py`: 11 tests (batching, overflow, exception propagation)

---

## Summary Verdict Table

| Area | Winner | Notes |
|---|---|---|
| DHT / Peer Discovery | Petals | Battle-tested Kademlia via Hivemind |
| Routing / Pipeline | Tie | Petals: more sophisticated heuristics; OpenHydra: better tested and documented |
| Inference Chain | Exo | Event-sourcing architecture is cleanest; Petals best for stream efficiency |
| Verification / Trust | OpenHydra | Only project with any trust mechanism |
| Batching / Coalescing | OpenHydra | True cross-request tensor batching |
| Model Loading | Petals | Cleanest loading pipeline with tensor parallelism |
| MLX Support | Exo (capability) / OpenHydra (quality) | Exo has more model-specific strategies; OpenHydra has watchdog + fallback |
| Economy + Privacy | OpenHydra (unique) | No equivalent in Petals or Exo |
| Test Coverage | OpenHydra (876) | 3x more tests than Exo, 30x more than Petals |
| Type Safety | Exo | Uses basedpyright strict mode |

---

## Action Items (v0.1.1)

1. **Fixed**: `_dedupe_peer_entries` field drop bug — sharding fields (`layer_start`, `layer_end`, `total_layers`) were silently zeroed during deduplication
2. **Fixed**: `PeerEndpoint.from_dict()` class method — eliminates 5x constructor duplication across coordinator modules
3. **Fixed**: `EconomyService` extracted from `engine.py` — first step in decomposing the god class
4. **Planned (v0.2.0)**: Full `engine.py` decomposition into 9 focused service classes (routing, pipeline, verification, economy, rate limiting, model catalog, health, metrics, configuration)
