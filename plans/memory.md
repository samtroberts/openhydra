# OpenHydra Session Memory

> Updated: 2026-04-12 (Phase 6 KV-aware sharded decode live on T4×2 — 3× TPS over Phase 1 stateless)
> Purpose: Gives Claude the context needed to resume work without re-reading everything.

---

## Where We Stand

### Completed Passes

| Pass | Scope | Status |
|------|-------|--------|
| 1-6 | Core infra, DHT, TCP, tests, MkDocs, Tauri desktop, Ollama API | Done |
| Equalization | Engine decomp, 6-factor routing, gRPC streaming, MLX parallelism | Done |
| Pass 8 QA | RAM fix, CI codesign, Golden Path docs, 8GB benchmark | Done |
| v1.1 | Hybrid Local/Swarm Mode (4 pillars), 75 TPS Local, 20 TPS Swarm | Done |
| v1.2 | Swarm Optimization: DSD, SpecPipe, INT8, TOPLOC, Chunked Prefill | Done |
| **Petals Parity** | 4 phases: push mode, streaming sessions, NAT relay, throughput bench | **Done** |
| **GPU Benchmarks** | Lightning.ai T4: 18.8 TPS localhost, 0.54 WAN, push 2.66x | **Done** |
| **New Models** | Gemma 4 + Qwen 3.5 (transformers 5.5.3, multimodal arch detection) | **Done** |
| **Rebalancing** | Autonomous peer-driven layer assignment | **Done** |

### Current State

- **1103 tests pass, 0 failures** (transformers 5.5.3)
- **Model catalog: 21 entries** — Qwen 2.5/3/3.5, Gemma 3/4, SmolLM2, TinyLLaMA (base + instruct)
- **Petals parity achieved** — push mode, streaming, NAT relay, throughput bench (+1,327 lines)
- **Selective weight loading** — 14GB model on 8GB Mac (peak 1.5GB)
- **Push mode verified** — 2.66x TPS (4.95→13.16 on localhost)
- **Autonomous rebalancing** — peers self-optimize layer positions

### Key Files (recent additions)

| File | What it is |
|------|------------|
| `coordinator/push_receiver.py` | PushResult callback registry for push mode |
| `coordinator/relay.py` | gRPC relay for NATted peers |
| `coordinator/stun_client.py` | Real RFC 5389 STUN (was stub) |
| `peer/throughput_bench.py` | Actual forward-pass TPS measurement + cache |
| `peer/autonomous_rebalancer.py` | Peer-autonomous layer rebalancing algorithm |
| `coordinator/specpipe_scheduler.py` | SpecPipe + run_pipelined() |
| `peer/activation_codec.py` | INT8 activation compression (now default on) |
| `ops/bench/gemma4_direct_bench.py` | Standalone Gemma 4 benchmark (bypasses OpenHydra's `_run_layers`) |
| `ops/bench/qwen9b_sharded_bench.py` | Direct 2-peer gRPC sharded benchmark (bypasses coordinator's infer loop) |

### Gemma 4 Loader Status (2026-04-11)

- `peer/model_shard.py::_strip_multimodal_components()` — drops `vision_tower` / `audio_tower` / `embed_vision` / `embed_audio` / `multi_modal_projector` after CPU load, replaces out-of-shard text layers with `nn.Identity()`, moves surviving text decoder + `lm_head` to the target device.
- `_is_multimodal_model_type()` drives the loader into a CPU-first path (`device_map={"": "cpu"}`, `low_cpu_mem_usage=True`, native dtype) only for Gemma 4 now; Qwen 3.5 VL checkpoints are handled by the standard selective-device-map path because `AutoModelForCausalLM` resolves to the text-only `Qwen3_5ForCausalLM` class which auto-unwraps the multimodal checkpoint keys.
- Direct bench (model.generate()) on Lightning T4: **~10.5 TPS warm** (Studio 1), **~8.9 TPS warm** (Studio 2); ~14.5 GB VRAM on a 15 GB T4; cold load ~190s (CPU read + GPU move).

### Qwen 3.5 9B Sharded Status (2026-04-11)

- **Loader**: `AutoModelForCausalLM` → `Qwen3_5ForCausalLM` (text-only class picked automatically for `model_type="qwen3_5"`). Selective device_map `model.layers.N` → target / disk works unchanged. 16/32 layers per peer on T4 = **~10.6 GB VRAM per peer**, ~23 s load with free flash-linear-attention fallback.
- **Critical dtype fix** in `peer/model_shard.py` — the loader now probes `config.torch_dtype` / `config.text_config.torch_dtype` and uses it for the sharded load path (`_native_dtype`, falls back to fp16). Qwen 3.5's `linear_attn` state-space buffers are bf16 and cannot be down-cast to fp16 without triggering `mat1/mat2 dtype mismatch` errors partway through the forward pass.
- **Forward pipeline works end-to-end**: prompt → gpu1 (layers 0-15, qwen_llama family, linear+full attention mix) → hidden → gpu2 (layers 16-31 + lm_head) → next token. Verified via direct gRPC benchmark with coherent Qwen-style thinking output.
- Sample results (direct bench, no KV cache, re-prefill per decode step over WAN SSH tunnels):
  - `"Hi"` → 4 tokens in 5.95s (**0.67 TPS**)
  - `"Say hello in five words."` → 8 tokens in 25.4s (**0.32 TPS**)
  - `"What is 17 times 23? Answer briefly."` → 24 tokens in 54.9s (**0.44 TPS**)
  All outputs are coherent and begin with Qwen 3.5's `<think>...` thinking chain. TPS is dominated by full-context re-prefill over the SSH-tunneled WAN gRPC; KV-cache-aware decoding will remove the O(N²) factor.

### Sharded Inference Fixes — 2026-04-12 (FIXED)

Every coordinator bug uncovered while benching Qwen 3.5 9B is now fixed; the plan lives at `plans/sharded-inference-fixes.md`. Test suite: **1114 passed, 9 skipped**.

1. **P0 Non-streaming outer decode loop** — `coordinator/inference_service.py::infer()` now has an autoregressive loop branch for `pytorch_autoregressive and len(pipeline) > 1`. It tokenizes on the coordinator, calls `_run_chain(max_tokens=1, initial_activation=...)` in a loop (no KV session), decodes per step, stops on EOS. Controlled by `EngineConfig.autoregressive_sharded_enabled` (default True) — CLI flag `--autoregressive-sharded` / `--no-autoregressive-sharded` in both `api_server.py` and `coordinator/node.py`.
2. **P0 SpecPipe tuple sentinel** — `coordinator/specpipe_scheduler.py::_stage_worker` now normalizes its error sentinel to the downstream stage's expected tuple shape (4-tuple for intermediate, 2-tuple for last-stage collector) and checks for `activation is None` at the top of the loop to forward the sentinel without crashing. 2 new regression tests in `tests/test_specpipe.py::TestSpecPipePipelinedErrorHandling`.
3. **P0 Meta-tensor leak from disk-offloaded layers** — New helpers in `peer/model_shard.py`: `_find_decoder_layer_list(model)` and `_replace_offloaded_layers_with_identity(model, kept)`. `PyTorchRuntime.__init__` now calls the latter after `pytorch_layer_cleanup`, replacing every out-of-shard slot in the real `nn.ModuleList` with `nn.Identity()`. This stops `DynamicCache` / `_update_causal_mask` / `model.parameters()` iterations from dispatching aten ops on meta tensors (previously crashed with `GET was unable to find an engine to execute this computation`). `_strip_multimodal_components` (Gemma 4 path) now reuses the same helper. 8 new regression tests in `tests/test_sharded_kv_cache.py`.
4. **P1 `decode_text` whitespace collapse** — `ModelShard.decode_text` no longer `.strip()`s its tokenizer-decoded output. A single-`"\n\n"`-token Qwen 3.5 "thinking preamble" is now preserved. Regression test: `tests/test_model_shard.py::test_decode_text_preserves_whitespace_only_token`.
5. **Live validation (2026-04-12)**: `/v1/completions` and `/v1/chat/completions` both generate coherent multi-token output for Qwen3.5-9B sharded across the two Lightning T4s **through the coordinator** (no direct gRPC bench, no SpecPipe, no KV session):
    - `"What is 17 times 23?"` → `"[Reasoning]\n- Calculation of 17 * 23"` (16 tokens, 79.7s = ~0.20 TPS).
    - `"Name three colors."` → `"[Reasoning]\n1.  **Analyze the Request:** The user is asking for"` (20 tokens, 91.7s = ~0.22 TPS).
    - Both peers logged `pytorch_layer_identity_swap: 16 offloaded layers replaced with nn.Identity` at startup.
    - `--no-autoregressive-sharded` kill-switch verified: falls back to legacy 1-token output (`"\n"`, 10s).
6. **TPS note**: 0.2 TPS is dominated by WAN SSH-tunnel RTT and full re-prefill per decode step (no KV cache reuse yet). Adding KV-cache-aware decoding on top of the new Identity-swap foundation is a Phase 5 TPS optimization — Phase 2B has already unblocked the meta-tensor issue that was preventing it.

### Phase 4 — Gemma 4 Sharded Adapter (2026-04-12, DONE)

- **`_DecoderArchitecture` extended** with `layer_types`, `per_layer_embed`, `per_layer_proj`, `per_layer_norm`, `hidden_size_per_layer`, `text_model`. Gemma 4 detection now sets `family="gemma4"` (was "llama") and captures the `Gemma4TextModel` reference so the adapter can call `get_per_layer_inputs` / `project_per_layer_inputs` directly.
- **New proto field `ForwardRequest.prompt_token_ids` (#42)** — ships the original prompt token IDs on every sharded hop so downstream peers can recompute the per-layer input tensor locally (via `embed_tokens_per_layer` + `per_layer_model_projection` + `per_layer_projection_norm`). Non-Gemma-4 families ignore the field; old peers stay backward-compatible.
- **`_run_layers_gemma4`** — new branch in `_run_layers` that:
  1. Computes layer-type-aware rotary per unique type (`self._rotary_emb(hidden, position_ids, layer_type)`).
  2. Builds full + sliding causal masks via `create_causal_mask` / `create_sliding_window_causal_mask`.
  3. Creates a **local `DynamicCache`** so the KV-sharing mechanism works. Gemma 4 E4B-it has `num_kv_shared_layers=18` — layers 24-41 read their K/V from `past_key_values.shared_layers` populated by layers 22, 23 (the last non-shared layer of each attention type). Without a cache object those shared-layer reads return uninitialised state and produce garbage tokens (reproduced and fixed via `/tmp/diag_gemma4_sharded.py`: native "The" vs manual "糙" → after fix both "The").
  4. Loops over `self._selected_layers`, passing each layer its slice `per_layer_inputs[:, :, layer_idx, :]`, the right mask, the right rotary, and the shared cache.
- **`chain.run()` auto-derives `prompt_token_ids`** from `initial_activation` when the Phase 1 non-streaming loop passes integer-valued floats — the Phase 1 and Phase 4 paths compose automatically without inference_service changes.
- **Hidden-size probe fix**: multimodal configs have `config.hidden_size=None` and the real dim in `config.text_config.hidden_size`. Previously defaulted to 768 which broke the `invalid_hidden_payload:hidden_size` check on stage 1; now probes text_config.
- **Per-peer shard safety check** in `_run_layers_gemma4`: if a shard contains KV-shared layers (>= `first_kv_shared_layer_idx`) but no layers that store the full-length K/V (< `first_kv_shared_layer_idx`), raises `gemma4_shard_split_breaks_kv_sharing` with a clear recovery suggestion. Cross-peer KV sharing (serialising `cache.shared_layers` on the wire) is a Phase 5 follow-up.

**Live validation on Lightning T4×2 (2026-04-12):** `/v1/chat/completions` on `openhydra-gemma4-e4b-it` sharded across 21/21 layers on two T4s produces coherent output via the Mac coordinator:

- `"Name three colors."` → `"Here are three colors:\n\n1. **Red**"` (12 tok, 23.3s ≈ 0.52 TPS)
- `"Write one sentence about the ocean."` → `"The vast, mysterious ocean covers most of our planet, teeming with incredible life beneath its surface."` (20 tok, 40.0s ≈ 0.50 TPS)

Pipeline trace shows 4 gRPC round-trips per decode step (gpu1-gemma → gpu2-gemma → gpu1-gemma → gpu2-gemma) because the Phase 1 outer decode loop does a full re-prefill per token over SSH-tunnelled WAN. KV-aware decoding on top of Phase 4 (next step) would collapse each decode to a single token per stage.

**10 new regression tests** in `tests/test_gemma4_sharded.py` covering `_DecoderArchitecture` extras defaults, Gemma 4 field population, `ForwardRequest.prompt_token_ids` wire roundtrip + field number stability, `chain.run()` auto-derive from `initial_activation` / explicit override / hidden-state-not-token-ids skip, and `_compute_gemma4_per_layer_inputs` safe defaults for non-Gemma-4 / missing-text-model / empty-ids cases. **Total test count: 1124 passed, 9 skipped** (up from 1114 after Phase 3, +10 new).

### Phase 6 — KV-aware Sharded Decode (2026-04-12, DONE)

Turned Phase 1's stateless O(N²) loop into an O(1)-per-token KV-cached loop with a transparent fallback.

**What shipped** (`coordinator/inference_service.py::infer()` + `peer/model_shard.py::_run_layers`):
1. **KV-aware decode loop** in `infer()`: `autoregressive_sharded_enabled` branch now runs a prefill call (`kv_store_activation=True`, full context) followed by N decode calls (`initial_activation=[prev_token]`, `kv_use_cached_activation=True`). Falls back transparently to stateless re-prefill on any `RuntimeError` — the first KV failure logs a warning, the rest of the request completes in stateless mode. Mode appears in the `autoregressive_sharded_done` log line (`kv_aware` vs `stateless`).
2. **`_run_layers` now materialises a `DynamicCache`** when the caller requests `use_cache=True` without providing one. Previously returned a tuple of `None`s — the saved "cache" was useless and every decode step silently re-prefilled with a single token (wrong position encodings → garbage logits). Fix mirrors what `Qwen3_5TextModel.forward` / `LlamaModel.forward` do natively.
3. **T4 cuDNN workaround for depthwise Conv1d**: `_patch_depthwise_conv1d_t4_fallback()` wraps every `nn.Conv1d` whose `in_channels == groups` (depthwise pattern used by Qwen 3.5 `linear_attn`) with a fallback that retries inside `torch.backends.cudnn.flags(enabled=False)` if the dispatcher raises `GET was unable to find an engine to execute this computation`. cuDNN on T4 has no kernel for `Conv1d(C=3200, kernel=4, groups=3200, padding=3)` in `bfloat16` with `seq_len<4` — the exact shape every Qwen 3.5 decode step hits. Prefill (`seq_len>=4`) keeps the fast cuDNN path; only the failing decode calls pay the software-fallback cost. Runs transparently — `conv1d_t4_fallback_patched: N depthwise Conv1d modules wrapped` at startup.

**Live validation on Lightning T4×2 (2026-04-12)** — Qwen 3.5 9B sharded via Mac coordinator:

| Request | Tokens | Latency | TPS | Output |
|---|---|---|---|---|
| `/v1/completions` — "What is 17 times 23?" | 16 | 25.5s | **0.63 TPS** | `"[Reasoning]\n- Step 1: Multiply 17 by"` |
| `/v1/completions` — "Explain sharding in one paragraph." | 32 | 49.9s | **0.64 TPS** | `"[Task]\n- Write a one-paragraph explanation of sharding.\n\n[Constraints]\n..."` |
| `/v1/chat/completions` — "Name three programming languages." | 24 | 35.8s | **0.67 TPS** | `"languages\n\n[Reasoning]\n1.  Identify the request: ..."` |

**Speedup**: Phase 6 KV-aware is **~3× faster** than Phase 1 stateless on the same 16-token generation (25.5s vs 74.8s). Pipeline trace confirms the decode loop is now constant-cost per step (~1.35s per token over the WAN SSH tunnel, gpu1+gpu2 round-trip each) instead of growing linearly with context length. The prefill single-shot cost is ~7.2s (unchanged from Phase 1) — the savings come entirely from the decode phase.

**Test count: 1129 passed, 9 skipped** (up from 1124; +5 new tests in `tests/test_autoregressive_sharded.py` covering:
- KV-aware prefill-then-decode call sequence (first call ships full context + store=True + use=False; every later call ships a single token + store=True + use=True)
- Stateless fallback when KV prefill raises (switches mode, completes request, never retries KV)
- Stateless mode sends full running context on every step (Gemma 4 relies on this)
- EOS stops immediately (regression for Phase 1 early-exit)
- Prefill happens exactly once per request (`_prefill_done` flag invariant)

**Gemma 4 still works**: the `_run_layers_gemma4` branch is unaffected by the `_run_layers` DynamicCache change — Gemma 4 has its own local cache creation for KV sharing. Qwen 3.5 + Phase 6 + Gemma 4 + Phase 4 coexist cleanly. No regressions in the 1129-test suite.

### Still To Do

- **Phase 6 Gemma 4 KV reuse**: extend `_run_layers_gemma4` to persist its local `DynamicCache` across `_forward_impl` calls so Gemma 4 decode steps also benefit from KV reuse. Currently `_run_layers_gemma4` creates a fresh cache per call, so Gemma 4's KV-aware path is effectively stateless re-prefill.
- **Cross-peer Gemma 4 KV sharing** (wire-format effort): serialise `DynamicCache.shared_layers` onto the wire between peers so shards can split the model at ANY layer boundary (not just boundaries where every shard contains its own non-shared anchor layers). Currently the 0-20/21-41 split works because peer 2 has layers 22 and 23 locally; a 0-30/31-41 split would need this.
- **Install `causal-conv1d`** on Lightning studios (CUDA-ext build, takes ~15-30 min from source): gives Qwen 3.5's `linear_attn` its fast path and lets us drop the T4 cuDNN fallback shim eventually.
- **Streaming endpoint KV reuse**: `coordinator/inference_service.py::infer_chat_stream` has its own KV-aware loop that pre-dates Phase 1 + Phase 6. It should be audited to make sure it composes with the Phase 2 Identity swap + Phase 6 `_run_layers` DynamicCache fix. Current live tests only exercise the non-streaming path.

### Known Limitation — Gemma 4 sharded via OpenHydra `_run_layers`

Gemma 4 decoder layers can NOT be driven by the current `_run_layers()` loop. Each layer requires:

1. **Layer-type-aware rotary** — `Gemma4TextRotaryEmbedding(hidden, position_ids, layer_type)`; buffers named `{layer_type}_inv_freq` per `config.layer_types[i]` (`full_attention` vs `sliding_attention`).
2. **Layer-type-aware causal mask** — separate `create_causal_mask` vs `create_sliding_window_causal_mask`.
3. **`per_layer_input` multiplication** — `hidden_states = hidden_states * per_layer_input` inside each block. `per_layer_input` comes from `embed_tokens_per_layer(input_ids)` + `project_per_layer_inputs`, so the **first-stage input_ids must travel alongside hidden activations** to every later shard (or per_layer_inputs itself must be carried as a sidecar tensor in the activation payload).

Until `_run_layers()` gets a `family="gemma4"` branch AND the activation payload carries the per-layer input tensor (or input_ids), multi-peer sharding of Gemma 4 will fail with `'Gemma4TextRotaryEmbedding' object has no attribute 'None_inv_freq'` or silently produce zero output. Single-peer unsharded Gemma 4 works via the direct bench.

### Test Suite

- **1103 passed, 9 skipped** (pytest)

### Production Bootstrap Nodes

- EU: 172.105.69.49:8468
- US: 45.79.190.172:8468
- AP: 172.104.164.98:8468
- Peer nanodes: **DELETED** (snapshots at `ops/nanode-snapshots/`, gitignored)

### Critical Architecture Facts

- Entry point: `openhydra-node` → `coordinator/node.py`
- gRPC: peer ↔ coordinator on port 50051
- **Push mode**: peers forward activations directly to next peer (no coordinator round-trip)
- **Streaming sessions**: ForwardStream RPC with StreamPool + InferenceSession history replay
- **NAT relay**: STUN probe → relay registration → proxied Forward calls
- **Selective loading**: `_build_selective_device_map()` maps unused layers to "disk"
- **Accelerate hooks**: removed after selective load for native speed
- **Throughput bench**: real model.generate() measurement, cached 24h
- **Autonomous rebalancing**: `should_rebalance()` runs every 6 announce cycles
- Layer sharding: FULLY ACTIVATED — DHT announces layer ranges, Dijkstra routing
- position_embeddings: computed per-layer via `rotary_emb()` — MUST NOT be wrapped in try/except
- `accelerate>=1.13.0` + `transformers>=5.5.0` required

---

## What to Do Next

1. **Coordinator decode loop for non-streaming sharded inference** — either fix the meta-tensor leak in the streaming/KV-cache path, fix the SpecPipe tuple-unpack bug in `coordinator/specpipe_scheduler.py:357`, OR add a simple outer-loop fallback in `inference_service.infer()` that calls `chain.run()` in a loop when `pytorch_autoregressive and len(pipeline) > 1`. Right now `api_server.py` cannot drive multi-token sharded generation for PyTorch peers.
2. **Gemma 4 sharded adapter** — `_run_layers()` branch with layer-type rotary + sidecar `per_layer_input` tensor in the activation payload (see "Known Limitation — Gemma 4 sharded" above)
3. **On-chain HYDRA integration** — replace mock_mode=True with real Solidity contracts
4. **Multi-GPU demo** — 7B instruct model across 4+ GPU peers with push mode (>5 TPS coherent output). Use Qwen 2.5-7B-Instruct or Qwen 3-8B (both in catalog, both llama-family, fully sharded-compatible with current runtime).
5. **Agent GUI** — point Open WebUI at OpenHydra's OpenAI-compatible API (zero code changes)
6. **Combine SpecPipe + push mode** — speculative tokens through the push pipeline
7. **Launch prep** — HN post, demo video, public endpoint

Always update this file and `progress.md` at the end of each session.
