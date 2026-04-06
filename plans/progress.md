# OpenHydra Progress Tracker

> Updated: 2026-04-06 (v1.2 Swarm Optimization complete — DSD, SpecPipe, 5-node WAN pipeline, 1103 tests)

---

## Legend

| Symbol | Meaning |
|--------|---------|
| :white_check_mark: | Complete |
| :construction: | In progress |
| :black_square_button: | Not started |
| :no_entry_sign: | Blocked / deferred |

---

## Beta Launch Roadmap (from master plan)

### Phase 0: Quick Wins (Week 1)

| Task | Status | Notes |
|------|--------|-------|
| Add warmup call in `PyTorchRuntime.__init__` | :white_check_mark: | `_warmup()` in model_shard.py; errors caught, never raised |
| Set `default_model` to auto-select best available | :black_square_button: | Depends on Phase 2 AutoScaler |
| Raise `timeout_ms` default 500 → 5000 | :white_check_mark: | coordinator/engine.py line 69 |
| Add `--warmup-on-start` CLI flag | :white_check_mark: | Wired through argparse → serve() → PeerService → ToyShardConfig |
| Tests for Phase 0 | :white_check_mark: | 12 tests in tests/test_phase0.py |

### Phase 1: MLX Backend (Weeks 2-3)

| Task | Status | Notes |
|------|--------|-------|
| Implement `MLXRuntime` in `peer/mlx_runtime.py` | :white_check_mark: | DLPack bridges, stream_generate, make_sampler |
| Add `"mlx"` to `--runtime-backend` choices | :white_check_mark: | peer/server.py argparse + ModelShard routing |
| Add `mlx`, `mlx-lm` to optional deps | :white_check_mark: | pyproject.toml `[mlx]` group |
| Warmup call in MLX runtime | :white_check_mark: | `_warmup()` via stream_generate, 0.61 s on 0.8B |
| Benchmark Qwen3.5-0.8B | :white_check_mark: | TTFT 0.37 s warm, **252 tok/s avg** (194× pytorch_auto) |
| 31 tests in `tests/test_mlx_runtime.py` | :white_check_mark: | Skip-guarded; 454 passed, 9 skipped total |

### Phase 2: Auto-Scaling + Model Selection (Weeks 3-4)

| Task | Status | Notes |
|------|--------|-------|
| Design capability-aware auto-scaling policy | :white_check_mark: | `plans/auto-scaling-policy.md` |
| `coordinator/auto_scaler.py` — AutoScaler + ModelSpec/PeerView | :white_check_mark: | Promote@3x, demote@1.5x, floor@2x, cooldown 15 min |
| `coordinator/request_log.py` — sliding-window demand tracker | :white_check_mark: | Records by tier; demand_weight; window_seconds configurable |
| `coordinator/role_assigner.py` — support roles for weak peers | :white_check_mark: | inference → embedding → auditor → cache → relay |
| `coordinator/degradation.py` — ModelAvailability extended | :white_check_mark: | Added shard_vram_gb, shards_needed, quality_tier |
| `models.catalog.json` — shard metadata added | :white_check_mark: | All 18 models have shard_vram_gb, shards_needed, quality_tier |
| `peer/dht_announce.py` — available_vram_mb field | :white_check_mark: | Phase 2 field; 0 = unknown |
| `coordinator/path_finder.py` — available_vram_mb in PeerEndpoint | :white_check_mark: | Parsed from DHT |
| `coordinator/engine.py` — AutoScaler wired; demand tracking | :white_check_mark: | _auto_scaler + _request_log in CoordinatorEngine |
| `peer/server.py` — _probe_available_vram_mb() | :white_check_mark: | CUDA → MLX → 0 fallback |
| 65 tests in `tests/test_auto_scaler.py` | :white_check_mark: | All 8 scenarios + unit tests; 519 passed, 9 skipped |

### Phase 3: Layer Sharding Activation (Weeks 4-6)

| Task | Status | Notes |
|------|--------|-------|
| `RuntimeProfile.total_layers` field | :white_check_mark: | Wired from `PyTorchRuntime`; `layer_start` + `layer_end` already existed |
| DHT announces `layer_start` / `layer_end` / `total_layers` | :white_check_mark: | Added to `Announcement` in `peer/dht_announce.py`; wired in `peer/server.py` |
| `PeerEndpoint` carries layer range | :white_check_mark: | Parsed from DHT payload in `coordinator/path_finder.py` |
| `coordinator/layer_coverage.py` — coverage validation + pipeline assembly | :white_check_mark: | `LayerRange`, `coverage_gaps`, `find_complete_pipeline`, `LayerCoverageMap` |
| `coordinator/engine.py` — exposes sharded pipeline info | :white_check_mark: | `"layer_coverage"` key in `network_status()`; `sharding_incomplete` alert |
| `tests/test_layer_coverage.py` | :white_check_mark: | 85 tests; covers all edge cases |
| **Tensor Flow Wiring ("Map vs Territory")** | | |
| `peer/peer.proto` — shard routing hint fields | :white_check_mark: | Fields 28/29/30: `shard_layer_start`, `shard_layer_end`, `shard_total_layers`; pb2 regenerated |
| `coordinator/chain.py` — shard fields in ForwardRequest | :white_check_mark: | `_request_stage()` passes `layer_start/end/total_layers` from `PeerEndpoint` per hop |
| `coordinator/engine.py` — `_select_pipeline_sharded()` | :white_check_mark: | Tries sharded pipeline first; falls back to full-model if incomplete or non-sharded peer selected |
| `coordinator/engine.py` — `pipeline_mode` in `InferencePreparation` | :white_check_mark: | `"sharded"` or `"full_model"`; surfaced in `infer()` response dict |
| `peer/server.py` — shard layer-range validation in `Forward()` | :white_check_mark: | Reads from `self.runtime_profile`; raises `shard_layer_mismatch` on coordinator/peer mismatch |
| `tests/test_sharded_grpc_pipeline.py` | :white_check_mark: | 19 tests: mock gRPC handoff, `_select_pipeline_sharded` unit tests, real gRPC end-to-end, integration |

### Phase 4: NF4 Quantization + Request Coalescing (Weeks 5-7)

| Task | Status | Notes |
|------|--------|-------|
| NF4 quantization via bitsandbytes | :white_check_mark: | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=True)`; graceful CPU fallback |
| MLXRuntime: reads quantization_mode from config | :white_check_mark: | Detects pre-quantized checkpoints; applies `mlx.nn.quantize()` for runtime quant; fixes profile keys |
| RuntimeProfile + DHT broadcast quantization level | :white_check_mark: | Already existed; `estimated_tps` → `estimated_tokens_per_sec` bug fixed; `runtime_model_id` + `total_layers` added to MLX profile |
| `tests/test_quantization_config.py` — 53 tests | :white_check_mark: | Alias normalisation, ToyRuntime propagation, memory/TPS scaling, NF4 source inspection, DHT fields, ModelShard facade |
| `BatchingQueue` in `peer/batching.py` + server wiring | :white_check_mark: | `_BatchItem` + `BatchingQueue` (timer + max_batch flush); `--batch-window-ms` / `--max-batch-size` CLI flags; non-PyTorch Forward() path coalesced |
| True tensor batching in `PyTorchRuntime.forward_batch()` | :white_check_mark: | `torch.cat(tensors, dim=0)` → single `_run_layers()` / `_model()` kernel; 3 dispatch cases (full model, first shard, intermediate/last) |
| True tensor batching in `MLXRuntime.forward_batch()` | :white_check_mark: | `mx.concatenate(arrays, axis=0)` → `model(current_ids)` per decode step; per-request EOS tracking |
| `ToyRuntime.forward_batch()` + `ModelShard.forward_batch()` | :white_check_mark: | Sequential loop for ToyRuntime (test spy); facade delegates to runtime |
| `tests/test_request_coalescing.py` — 11 tests | :white_check_mark: | 8 unit tests (coalescing, max-batch, overflow, fallback, exception propagation) + 3 gRPC integration tests |
| Throughput benchmarking | :black_square_button: | |

### Phase 5: Peer-to-Peer Model Distribution (Weeks 6-8)

| Task | Status | Notes |
|------|--------|-------|
| Peers announce cached models to DHT | :white_check_mark: | `seeder_http_port` + `cached_model_ids` in `Announcement` + `PeerEndpoint` |
| HTTP range-request download from peer | :white_check_mark: | `peer/seeder_http.py` — `ModelSeedServer`; RFC 7233 Range + path-traversal guard |
| SHA-256 integrity verification | :white_check_mark: | `peer/p2p_model_cache.py` — verified against HF Hub API manifest; `.part` + atomic rename |

### Phase 6: Polish + Launch (Weeks 8-10)

| Task | Status | Notes |
|------|--------|-------|
| MkDocs documentation site | :white_check_mark: | 4 Mermaid diagrams, CI job |
| Tauri desktop: 5 UX features | :white_check_mark: | Theme, earnings, palette, history, peer map |
| Ollama API compatibility | :white_check_mark: | `/api/generate`, `/api/chat`, `/api/show`, `/api/ps`, `/api/tags` |
| Dijkstra path routing | :white_check_mark: | 6-factor cost + Dijkstra in `layer_coverage.py` |
| Additional bootstrap nodes | :white_check_mark: | 3 production nodes (EU/US/AP) |
| Load testing: 100 concurrent users | :black_square_button: | |

### v1.1: Hybrid Local/Swarm Mode (2026-04-01 → 2026-04-04)

| Task | Status | Notes |
|------|--------|-------|
| LocalInferenceEngine | :white_check_mark: | Direct ModelShard.forward(), zero gRPC/DHT |
| API emulation (local/swarm routing) | :white_check_mark: | Transparent engine routing in api_server.py |
| Mode toggle (503 drain gate) | :white_check_mark: | Thread-safe, localhost-only, Tauri wired |
| VRAM reallocation | :white_check_mark: | gc.collect() + mx.metal.clear_cache() |
| 75 TPS Local / 20 TPS Swarm | :white_check_mark: | 50× Swarm speedup from 0.4 TPS |

### v1.2: Swarm Inference Optimization (2026-04-04 → 2026-04-06)

| Task | Status | Notes |
|------|--------|-------|
| Decentralized Speculative Decoding (DSD) | :white_check_mark: | coordinator/speculative_swarm.py |
| SpecPipe (pipeline-filling speculation) | :white_check_mark: | coordinator/specpipe_scheduler.py, KV-cached continuation |
| INT8 Activation Compression | :white_check_mark: | peer/activation_codec.py |
| TOPLOC Hash Verification | :white_check_mark: | verification/toploc.py |
| PALU Low-Rank SVD Compression | :white_check_mark: | peer/palu_codec.py |
| Chunked Prefill | :white_check_mark: | coordinator/chunked_prefill.py |
| RTT-Aware Peer Selection | :white_check_mark: | coordinator/peer_selector.py, 5-factor Dijkstra |
| Coordinator-Level Batching | :white_check_mark: | coordinator/request_batcher.py |
| ToyRuntime → real tinyllama-15M | :white_check_mark: | nickypro/tinyllama-15M, module-level caching |
| Model catalog expansion (5→9) | :white_check_mark: | +Qwen2.5-0.5B, SmolLM2-360M, Gemma-3-270m, TinyLLaMA-15M |
| 5-node WAN sharded pipeline | :white_check_mark: | Bangalore→Chennai→Mumbai→Singapore×2 |
| position_embeddings fix | :white_check_mark: | Removed silent try/except in _run_layers() |
| Peer dedup layer info fix | :white_check_mark: | Preserve layer metadata during dedup |
| accelerate>=1.13.0 compatibility | :white_check_mark: | For transformers 5.3.0 + device_map |
| WARN→WARNING log normalization | :white_check_mark: | _JsonFormatter normalizes absl-py remapping |
| Nanode snapshots + setup guide | :white_check_mark: | ops/nanode-snapshots/SETUP_GUIDE.md |
| Nanodes deleted (4 nodes) | :white_check_mark: | Chennai, Mumbai, Singapore ×2 |

---

## Infrastructure / DevOps

| Task | Status | Notes |
|------|--------|-------|
| DHT lookup timeout 0.5 -> 3.0 | :white_check_mark: | coordinator/engine.py |
| Lazy `list_models()` (no DHT scan) | :white_check_mark: | |
| TCP connection pooling (`_DHT_SESSION`) | :white_check_mark: | path_finder.py |
| `ulimit -n 65536` on all Linodes | :white_check_mark: | |
| nginx `access_log off` on all Linodes | :white_check_mark: | |
| app.js infinite recursion bug fixed | :white_check_mark: | Monkey-patching hoisting issue |
| mkdocs build --strict passes | :white_check_mark: | 0 warnings |
| **1103 tests pass, 9 skipped** | :white_check_mark: | Up from 715; includes v1.1 + v1.2 tests |
| 1GB nanode memory optimization | :white_check_mark: | fp16 + layer cleanup + 1GB swap + MALLOC_ARENA_MAX=2 |
| accelerate>=1.13.0 in requirements.txt | :white_check_mark: | Required for transformers 5.3.0 |

---

## Sub-Plans

| # | File | Topic | Status |
|---|------|-------|--------|
| 1 | `auto-scaling-policy.md` | Capability-aware auto-scaling | :white_check_mark: Design + implementation complete |
