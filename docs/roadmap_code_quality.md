# OpenHydra Code Quality Roadmap (v0.2.0+)

## Context

Based on comparative analysis against Petals and Exo (see docs/comparison_petals_exo_analysis.md), this roadmap addresses areas where OpenHydra can improve.

## 1. 5-Factor Dijkstra Routing (match Petals)

- **Current**: 3-factor greedy + Dijkstra in `coordinator/layer_coverage.py`
- **Target**: Incorporate RTT measurement, KV cache cost, compute latency (1/inference_rps), server-to-server measured pings, and reputation score into edge costs.
- **Key file**: `coordinator/layer_coverage.py`
- **Effort**: 2-3 days

## 2. Tensor Parallelism for Multi-GPU (match Petals)

- **Current**: Single GPU per peer
- **Target**: `torch.distributed` / NCCL support in `peer/model_shard.py` so a single peer splits one layer across multiple GPUs
- **Key files**: `peer/model_shard.py`, `peer/server.py`
- **Effort**: 1-2 weeks

## 3. Strict Typing (match Exo)

- **Current**: No type checker in CI
- **Target**: basedpyright in CI, Pydantic v2 models for API request/response types with `extra="forbid"`
- **Key files**: `.github/workflows/python-app.yml`, `coordinator/api_server.py`
- **Effort**: 3-5 days initial pass

## 4. Event-Sourcing Runner State Machine (match Exo)

- **Current**: Imperative `_run_chain` in `coordinator/chain.py`
- **Target**: Pure `apply()` state transitions with audit log, typed Event/Task objects
- **Key files**: `coordinator/chain.py`, `coordinator/engine.py`
- **Effort**: 1 week

## 5. Overlapped Pipeline Prefill for MLX (match Exo)

- **Current**: Synchronous MLX inference
- **Target**: `mx.async_eval` overlap for pipeline stages to hide network latency
- **Key file**: `peer/mlx_runtime.py`
- **Effort**: 3-5 days

## 6. KV Prefix Cache with RAM-Threshold Eviction (match Exo)

- **Current**: Session-peer affinity only (no KV tensor caching)
- **Target**: LRU cache with configurable RAM threshold (128GB->85%, 64GB->80%, 32GB->75%)
- **Key files**: `peer/mlx_runtime.py`, `peer/model_shard.py`
- **Effort**: 1 week

## 7. Docstrings for All Public Methods

- **Current**: `engine.py` has 71 methods and 0 docstrings
- **Target**: Google-style docstrings on all public methods
- **Key file**: `coordinator/engine.py` (and all extracted services)
- **Effort**: 2 days

## 8. engine.py Full Decomposition

- **Current**: 3,031 lines after v0.1.1 economy extraction
- **Target**: 9 focused services behind `CoordinatorEngine` facade

Extraction order (risk-sorted):

1. **EconomyService** (DONE in v0.1.1)
2. **PeerDiscoveryService** -- `_discover_for_model`, `_scan_network`, `_load_candidate_peers`, DHT cache
3. **PipelineSelectionService** -- `_select_pipeline_sharded`, `_select_pipeline`, bandwidth, MoE
4. **KVAffinityService** -- 7 KV affinity methods
5. **HealthScoringService** -- health recording, verification feedback
6. **TokenizationService** -- tokenizer/model loading, runtime resolution
7. **StatusService** -- `list_models`, `network_status`, `metrics_snapshot`
8. **InferenceExecutionService** -- `infer()`, `infer_stream()`, `_run_chain()`

- **Effort**: 1 week incremental

## Priority Order

1. **Docstrings** -- immediate, zero risk, high documentation value
2. **Strict typing** -- CI improvement, catches bugs early
3. **engine.py decomposition** -- ongoing, incremental
4. **5-factor routing** -- matches Petals, improves swarm efficiency
5. **KV prefix cache** -- performance win for repeat queries
6. **Overlapped prefill** -- MLX performance
7. **Event-sourcing** -- architectural, big refactor
8. **Tensor parallelism** -- new capability, largest effort
