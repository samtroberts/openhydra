# OpenHydra Progress Tracker

> Updated: 2026-04-12 (Phases 1+2+3+4+6+7 landed: streaming fixed, Gemma 4 KV reuse, T4 cuDNN workaround; both models coherent through coordinator; 1129 tests)

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
| transformers>=5.5.0 in requirements.txt | :white_check_mark: | Required for Gemma 4 + Qwen 3.5 |
| Selective weight loading (disk offload) | :white_check_mark: | 14GB model on 8GB Mac |
| Accelerate hook removal (5x speedup) | :white_check_mark: | remove_hook_from_module() after selective load |

### Petals Parity (2026-04-06 → 2026-04-10)

| Task | Status | Notes |
|------|--------|-------|
| Phase A: Server-to-server push | :white_check_mark: | +509 lines; 2.66x TPS verified |
| Phase A: INT8 compression enabled | :white_check_mark: | Default on |
| Phase B: Streaming sessions | :white_check_mark: | StreamPool + InferenceSession wired |
| Phase B: History replay failover | :white_check_mark: | Session records req/resp for replay |
| Phase C: STUN client (RFC 5389) | :white_check_mark: | Real UDP probe, replaces stub |
| Phase C: gRPC relay server | :white_check_mark: | coordinator/relay.py |
| Phase C: NAT fields in announcements | :white_check_mark: | nat_type, requires_relay, relay_address |
| Phase C: 20% relay routing penalty | :white_check_mark: | Dijkstra cost function |
| Phase D: Throughput self-benchmark | :white_check_mark: | peer/throughput_bench.py |
| Phase D: Benchmark cache (24h TTL) | :white_check_mark: | ~/.openhydra/throughput_cache.json |
| Push mode CLI flags | :white_check_mark: | --push-mode, --push-callback-address |

### New Model Families (2026-04-10)

| Task | Status | Notes |
|------|--------|-------|
| Gemma 4 architecture detection | :white_check_mark: | model.model.language_model.layers unwrap |
| Qwen 3.5 support | :white_check_mark: | Works out of the box (qwen_llama family) |
| Instruct models (Qwen2.5-7B-Instruct) | :white_check_mark: | Catalog entry, no code changes |
| Model catalog expanded (21 entries) | :white_check_mark: | +10 Gemma 4 + Qwen 3.5 entries |

### Sharded Inference Fixes — Phase 7 Streaming + Gemma 4 KV Reuse (2026-04-12)

| Phase | Task | Status | Notes |
|-------|------|--------|-------|
| 7A | Streaming endpoint rewrite (`infer_stream` pytorch_autoregressive) | :white_check_mark: | Rewrote to use Phase 6 prefill+decode loop instead of legacy verify/commit dance. Legacy path sent `context_token_ids[-1]` (the last PROMPT token, e.g. `<\|im_start\|>assistant`) as verify input → model predicted EOS → 0 SSE chunks. New path correctly captures the prefill's sampled token and continues. Streaming Qwen3.5-9B: 24 chunks in 45.3s = **0.53 TPS** |
| 7B | Gemma 4 KV reuse — persist `DynamicCache` across calls | :white_check_mark: | `_run_layers_gemma4` now accepts an existing `past_key_values` and returns it for storage. Previously created a fresh `DynamicCache` every call → effectively stateless re-prefill. With reuse: Gemma 4 E4B-it 20 tok = **0.60 TPS** (vs 0.50 stateless), 40 tok = **0.71 TPS** |
| 7 | T4 cuDNN workaround (`_patch_depthwise_conv1d_t4_fallback`) | :white_check_mark: | (Shipped in Phase 6 session) cuDNN on T4 has no kernel for `Conv1d(C, C, k=4, groups=C)` in bf16 with `seq_len<4`; the exact shape Qwen 3.5 linear_attn hits on every KV-aware decode step. Wrapper retries inside `torch.backends.cudnn.flags(enabled=False)`. Logs `conv1d_t4_fallback_patched: N modules` at startup |
| 7 | `_run_layers` materialises `DynamicCache` for prefill | :white_check_mark: | (Shipped in Phase 6 session) Native `Qwen3_5TextModel.forward` creates `DynamicCache(config=...)` when `use_cache=True, past_key_values=None`; our `_run_layers` now does the same. Prevents "tuple of Nones" cache → garbage single-token decode |
| 7 | Live validation | :white_check_mark: | Both models coherent through coordinator on Lightning T4×2. See summary below |

**Final TPS summary (Lightning T4×2 via Mac coordinator, SSH-tunnelled WAN):**

| Model | Endpoint | Tokens | Phase 1 (stateless) | Phase 6+7 (KV-aware) | Speedup |
|---|---|---|---|---|---|
| Qwen3.5-9B | `/v1/completions` | 16 | 74.8s / 0.21 TPS | 25.5s / **0.63 TPS** | **3.0×** |
| Qwen3.5-9B | `/v1/completions` | 32 | — | 49.9s / **0.64 TPS** | — |
| Qwen3.5-9B | `/v1/chat/completions` stream | 24 | empty | 45.3s / **0.53 TPS** | **∞** (was broken) |
| Gemma4 E4B-it | `/v1/chat/completions` | 20 | 40.0s / 0.50 TPS | 33.3s / **0.60 TPS** | **1.2×** |
| Gemma4 E4B-it | `/v1/chat/completions` | 40 | — | 56.6s / **0.71 TPS** | — |

### Sharded Inference Fixes — Phase 6 KV-aware Decode (2026-04-12)

| Phase | Task | Status | Notes |
|-------|------|--------|-------|
| 6A | KV-aware decode loop in `infer()` (prefill once + decode N) | :white_check_mark: | `coordinator/inference_service.py` — drops O(N²) re-prefill for sharded PyTorch pipelines |
| 6A | Stateless fallback on `RuntimeError` | :white_check_mark: | First failure switches mode for the rest of the request; visible in `autoregressive_sharded_done mode=...` log |
| 6B | `_run_layers` materialises `DynamicCache` when `use_cache=True` | :white_check_mark: | `peer/model_shard.py` — fixes the "tuple of Nones" cache that made every Phase 6 decode step a fresh single-token re-prefill (silent garbage logits) |
| 6C | T4 cuDNN workaround for depthwise `Conv1d` | :white_check_mark: | `_patch_depthwise_conv1d_t4_fallback()` — wraps every depthwise `nn.Conv1d` with a `cudnn.flags(enabled=False)` retry path. cuDNN on T4 has no kernel for `Conv1d(C, C, k=4, groups=C, padding=3)` in bf16 with `seq_len<4`; the exact shape Qwen 3.5 `linear_attn` hits on every decode step. Logs `conv1d_t4_fallback_patched: N depthwise Conv1d modules wrapped` |
| 6D | Regression tests — `tests/test_autoregressive_sharded.py` | :white_check_mark: | 5 tests: KV prefill→decode call sequence, stateless fallback on KV failure, stateless full-context per step, EOS early-exit, prefill happens exactly once |
| 6E | Full test suite | :white_check_mark: | **1129 passed, 9 skipped** (up from 1124; +5 new) |
| 6F | Live validation on Lightning T4×2 | :white_check_mark: | `/v1/completions` 16 tok in **25.5s = 0.63 TPS**, 32 tok in **49.9s = 0.64 TPS**; `/v1/chat/completions` 24 tok in **35.8s = 0.67 TPS**. **~3× faster** than Phase 1 stateless on the same 16-token generation (74.8s → 25.5s). Coherent reasoning output (`"[Reasoning]\n- Step 1: Multiply 17 by"`). Decode latency now constant ~1.35s/token regardless of context length |

### Sharded Inference Fixes — Phase 4 Gemma 4 Adapter (2026-04-12)

| Phase | Task | Status | Notes |
|-------|------|--------|-------|
| 4A | `_DecoderArchitecture` extras: `layer_types`, `per_layer_embed/proj/norm`, `hidden_size_per_layer`, `text_model` | :white_check_mark: | Defaults to empty/None so non-Gemma-4 paths are untouched |
| 4B | `ForwardRequest.prompt_token_ids` proto field (#42) | :white_check_mark: | Repeated int64; `peer/peer_pb2{,_grpc}.py` regenerated |
| 4C | `_detect_decoder_architecture` populates Gemma 4 extras + sets `family="gemma4"` | :white_check_mark: | Previously set Gemma 4 to `family="llama"` and produced garbage |
| 4D | `_run_layers_gemma4` branch with layer-type rotary + full/sliding masks + per-layer-input slicing + local `DynamicCache` | :white_check_mark: | `peer/model_shard.py` — 130 lines |
| 4E | `_forward_impl` threads `prompt_token_ids` through and computes per-layer-input sidecar before `_run_layers` | :white_check_mark: | Stage 0 falls back to `activation` when field is empty for backward compat |
| 4F | `PyTorchRuntime` / `ModelShard` forward + forward_async signatures carry `prompt_token_ids` | :white_check_mark: | Isolated to `isinstance(runtime, PyTorchRuntime)` so ToyRuntime / MLXRuntime stay untouched |
| 4G | `chain.run()` auto-derives `prompt_token_ids` from `initial_activation` when it's all integer-valued floats | :white_check_mark: | Phase 1 and Phase 4 compose automatically; no inference_service changes |
| 4H | Peer server reads `request.prompt_token_ids` + passes to `shard.forward_async` | :white_check_mark: | `peer/server.py` |
| 4I | Hidden-size probe uses `text_config.hidden_size` for multimodal wrappers | :white_check_mark: | Fixes `invalid_hidden_payload:hidden_size` from 768 default |
| 4J | Gemma 4 KV-sharing: create `DynamicCache` + pass to every layer | :white_check_mark: | `num_kv_shared_layers=18` — layers 24-41 read from `cache.shared_layers` populated by layers 22-23. Without this, produces garbage tokens (reproduced + fixed via `/tmp/diag_gemma4_sharded.py`) |
| 4K | Shard-split safety check raises `gemma4_shard_split_breaks_kv_sharing` when a shard has shared layers but no storing layers | :white_check_mark: | Clear error with recovery suggestion |
| 4L | Regression tests — `tests/test_gemma4_sharded.py` | :white_check_mark: | 10 tests: `_DecoderArchitecture` defaults + Gemma 4 population, proto roundtrip + field-number stability, chain.run auto-derive cases, `_compute_gemma4_per_layer_inputs` safe defaults |
| 4M | Full test suite | :white_check_mark: | **1124 passed, 9 skipped** (up from 1114; +10 new Gemma 4 tests) |
| 4N | Live validation on Lightning T4×2 | :white_check_mark: | `openhydra-gemma4-e4b-it` sharded 21/21 across Lightning Studios 1+2 via Mac coordinator. Coherent output: `"Write one sentence about the ocean."` → `"The vast, mysterious ocean covers most of our planet, teeming with incredible life beneath its surface."` (20 tok, 40.0s ≈ 0.50 TPS). Both peers log `multimodal_strip: kept=21 replaced=21` at startup. |

### Sharded Inference Fixes — Phases 1+2+3 (2026-04-12)

See `plans/sharded-inference-fixes.md` for the full plan.

| Phase | Task | Status | Notes |
|-------|------|--------|-------|
| 1A | Non-streaming outer decode loop in `infer()` | :white_check_mark: | `coordinator/inference_service.py` — stateless re-prefill per token over sharded PyTorch pipelines; shared `_collect_eos_token_ids` helper also used by streaming path |
| 1B | `--autoregressive-sharded` kill-switch | :white_check_mark: | Added to both `api_server.py` and `coordinator/node.py`; wires to `EngineConfig.autoregressive_sharded_enabled` (default True) |
| 1C | Drop `decode_text` `.strip()` | :white_check_mark: | `peer/model_shard.py`; preserves whitespace-only first tokens (Qwen 3.5 thinking preamble) |
| 2A | Investigate meta-tensor error | :white_check_mark: | Root cause: `self._blocks[idx] = None` only clears local ref — the real `nn.ModuleList` still holds meta-device modules that DynamicCache / `_update_causal_mask` iterate over |
| 2B | Identity-replace offloaded layers in ModuleList | :white_check_mark: | New helpers `_find_decoder_layer_list` + `_replace_offloaded_layers_with_identity`; called from `PyTorchRuntime.__init__`; `_strip_multimodal_components` refactored to reuse |
| 2C | Regression test suite | :white_check_mark: | `tests/test_sharded_kv_cache.py` — 8 tests covering meta-tensor elimination, indexing preservation, idempotence, no-op full-model case, bare-model handling, parameter iteration |
| 3A | SpecPipe stage-worker sentinel fix | :white_check_mark: | `_stage_worker` normalizes error tuples per downstream stage shape; adds `activation is None` sentinel check at top of loop |
| 3B | SpecPipe regression tests | :white_check_mark: | `tests/test_specpipe.py::TestSpecPipePipelinedErrorHandling` — 2 tests exercising stage 0 failures in 2- and 3-stage pipelines |
| — | Decode-text whitespace regression test | :white_check_mark: | `tests/test_model_shard.py::test_decode_text_preserves_whitespace_only_token` |
| — | Full test suite | :white_check_mark: | **1114 passed, 9 skipped** (up from 1103; +11 new tests) |
| — | Live T4×2 validation | :white_check_mark: | `/v1/completions` + `/v1/chat/completions` both generate coherent multi-token output for Qwen3.5-9B sharded across Lightning Studios 1+2 via the Mac coordinator. 16 tok/79.7s = ~0.20 TPS (WAN + stateless re-prefill). Both peers log `pytorch_layer_identity_swap: 16 offloaded layers replaced with nn.Identity` at startup. |

### Qwen 3.5 9B Sharded T4×2 (2026-04-11)

| Task | Status | Notes |
|------|--------|-------|
| Qwen3.5-9B downloaded to both studios | :white_check_mark: | 19 GB each at `/teamspace/studios/this_studio/openhydra/models/Qwen3.5-9B` |
| Native-dtype loader probe | :white_check_mark: | `peer/model_shard.py` — reads `config.torch_dtype` / `config.text_config.torch_dtype`; Qwen 3.5 forces bf16 because `linear_attn` state-space buffers cannot be down-cast |
| `_detect_layer_prefix` / `_build_selective_device_map` kept simple | :white_check_mark: | `AutoModelForCausalLM` picks `Qwen3_5ForCausalLM` (text-only, `model.layers.N`); the multimodal checkpoint keys are auto-remapped |
| Sharded peers loaded | :white_check_mark: | gpu1 layers 0-15, gpu2 layers 16-31; **10.6 GB VRAM per T4**, ~23 s load each |
| Sharded forward pipeline runs end-to-end | :white_check_mark: | Via `ops/bench/qwen9b_sharded_bench.py` — direct gRPC to both peers, no coordinator |
| Coherent output verified | :white_check_mark: | Qwen thinking-chain output ("Thinking Process:\n\n1.  **Analyze the Request:** ...") on math-reasoning prompts |
| Direct-bench TPS (no KV cache, O(N²) re-prefill) | :white_check_mark: | "Hi"→4 tok = **0.67 TPS** · "Say hello in five words."→8 tok = **0.32 TPS** · "What is 17 times 23?"→24 tok = **0.44 TPS** |
| `api_server.py --specpipe` flag | :white_check_mark: | Wires `specpipe_enabled` + `specpipe_max_depth` into `EngineConfig` (was hard-coded off) |
| Coordinator non-streaming sharded multi-token generation | :no_entry_sign: | **Blocked** by three separate coordinator bugs — documented in `plans/memory.md` ("Coordinator Bugs Uncovered"). Direct-bench works, coordinator path does not. |

### Gemma 4 Loader + T4 Bench (2026-04-11)

| Task | Status | Notes |
|------|--------|-------|
| Multimodal CPU-first loader path | :white_check_mark: | `_strip_multimodal_components()` in `peer/model_shard.py`: loads full `Gemma4ForConditionalGeneration` into CPU RAM, drops `vision_tower` / `audio_tower` / `embed_vision` / `embed_audio` / `multi_modal_projector`, Identity-replaces out-of-shard text layers, moves surviving decoder + `lm_head` to target device |
| `_is_multimodal_model_type()` detector | :white_check_mark: | Config-only probe for `gemma4` / `qwen3_5` |
| Single-T4 direct bench (model.generate) | :white_check_mark: | `ops/bench/gemma4_direct_bench.py` — Studio 1: **10.71 TPS**, Studio 2: **8.88 TPS** (warm, greedy, 95 tokens, fp16, ~14.5 GB VRAM) |
| Full regression suite after refactor | :white_check_mark: | 1103 passed, 9 skipped |
| Sharded Gemma 4 through `_run_layers` | :no_entry_sign: | **Blocked** by Gemma 4's layer-type-aware rotary + per-layer-input multiplication. Needs (1) `family="gemma4"` branch in `_run_layers`, (2) per-layer-input sidecar in the activation pipeline payload, (3) full + sliding causal masks. See `plans/memory.md` "Known Limitation" section. Single-peer / direct-generate works. |

### Autonomous Rebalancing (2026-04-10)

| Task | Status | Notes |
|------|--------|-------|
| should_rebalance() algorithm | :white_check_mark: | Per-layer throughput analysis |
| AutonomousRebalancer class | :white_check_mark: | Cooldown, history, jitter |
| Wired into announce loop | :white_check_mark: | peer/server.py |
| CLI flags | :white_check_mark: | --rebalance-enabled, --rebalance-interval, etc. |

---

## Sub-Plans

| # | File | Topic | Status |
|---|------|-------|--------|
| 1 | `auto-scaling-policy.md` | Capability-aware auto-scaling | :white_check_mark: Design + implementation complete |
| 2 | `swarm-inference-optimization.md` | Swarm inference optimization | :white_check_mark: All P0/P1/P2 implemented |
| 3 | `autonomous-rebalancing.md` | Autonomous dynamic rebalancing | :white_check_mark: Designed + implemented |
