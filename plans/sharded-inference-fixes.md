# Plan: Fix Sharded Inference Blockers

> Created: 2026-04-11
> Scope: Fix every defect uncovered while benching Qwen3.5-9B sharded across 2× Lightning T4 GPUs. The direct-gRPC bench works end-to-end; the OpenHydra coordinator path does not. Unblocks every sharded model >1 peer.

---

## Why this matters

Right now `coordinator.api_server` cannot drive multi-token generation for any sharded PyTorch pipeline. Every model we care about (Qwen 3.5 9B/27B, Qwen 2.5-7B-Instruct, Qwen 3-8B, Gemma 4 E4B-it) is blocked on this. The direct-bench workaround (`ops/bench/qwen9b_sharded_bench.py`) proves the peers and the forward pipeline are correct — the bugs are all in the coordinator's decode orchestration. Fixing these unlocks: the Petals-parity launch demo, the 4-peer multi-GPU demo, and any real user workflow that talks to the OpenAI-compatible API.

---

## Issue inventory (root-cause summary)

| # | Severity | Where | What's broken | Root cause |
|---|----------|-------|----------------|------------|
| **1** | P0 | `coordinator/inference_service.py:1290` (`infer()`) | Non-streaming calls `chain.run()` **once**; sharded last-stage peer returns **1 token per call**; user sees empty output. | No outer decode loop. The peer can't autoregress on its own because it doesn't hold all the layers. Only the streaming and SpecPipe paths have a loop — neither is wired into non-streaming `/v1/completions` / `/v1/chat/completions`. |
| **2** | P0 | `coordinator/inference_service.py:1844` (`infer_chat_stream()`) | Streaming + KV cache hits `GET was unable to find an engine to execute this computation`. | Prefill with `kv_store_activation=True` builds a `DynamicCache` that references **all 32 layer slots**, including layers 16–31 which accelerate put on the `meta` device (disk offload). Touching the meta-device cache entries through a torch dispatcher op blows up. |
| **3** | P0 | `coordinator/specpipe_scheduler.py:357` | Stage worker raises `ValueError: not enough values to unpack (expected 4, got 2)` after any stage failure. | Error path at line 387 puts `(None, tok_idx)` (2-tuple) into `out_q`; downstream stage at 357 always unpacks a 4-tuple — a poison-pill tuple type is not distinguished from a real activation tuple. |
| **4** | P1 | `peer/model_shard.py::decode_text` (line ~2620) | Output is `.strip()`ed; a first-token generation of `"\n\n"` (Qwen 3.5 thinking preamble) is silently dropped to `""`. Caused hours of debugging. | `.strip()` on a pre-decoded string is too aggressive when the tokenizer decode already returns a clean whitespace token. |
| **5** | P1 | `peer/model_shard.py::_forward_impl` (~line 1974) | Sharded last stage hard-codes `output_count = 1` unless `cache_requested and len(activation) > 1`. Even with KV cache, the outer loop still has to traverse the whole pipeline for every token. | Current design assumes the coordinator orchestrates the loop; the peer is stateless per call. That's fine once issue #1 is fixed, but the one-shot behavior is not documented. |
| **6** | P1 | `peer/model_shard.py::_run_layers` + Gemma 4 | No `family="gemma4"` branch. Gemma 4 needs (a) layer-type-aware rotary (`self._rotary_emb(hidden, pos, layer_type)`), (b) a `per_layer_input` sidecar tensor carried in the activation payload, (c) `full_attention` vs `sliding_attention` causal masks per layer. | Architectural gap. Qwen 3.5 sidesteps all three because its decoder is simpler: rotary is global, no per-layer input, masks are implicit. |
| **7** | P2 | Lightning studios | `flash-linear-attention` + `causal-conv1d` not installed — Qwen 3.5's `linear_attn` layers run in slow torch fallback. | Infra. Bench TPS is masked by this. |
| **8** | P2 | `ops/bench/qwen9b_sharded_bench.py` | Re-prefills the full context through both peers for every decode step (O(N²)). | Intentional: proves the forward path works without depending on KV cache. Once issue #2 is fixed we can add a KV-aware variant and get realistic TPS. |

---

## Fix plan

### Phase 1 — P0 coordinator decode loop (unblocks everything)

**Goal**: Make non-streaming `/v1/completions` and `/v1/chat/completions` produce multi-token output through a sharded pipeline **without** depending on the KV-cache machinery (which is broken) or SpecPipe (which is also broken).

#### 1A. Stateless outer decode loop for sharded PyTorch (fixes #1)

**File**: `coordinator/inference_service.py::infer()` (around line 1290)

**Change**: When `_pipeline_uses_pytorch_runtime(prep.primary_pipeline)` is true **and** the pipeline has `>1` stage, replace the single `chain.run()` call with a stateless autoregressive loop:

```python
# Tokenize the effective prompt once on the coordinator
tok = self._engine._load_generation_tokenizer(runtime_model_id)
context_token_ids = list(tok.encode(prep.effective_prompt, add_special_tokens=True))
eos_ids = _collect_eos_token_ids(tok)                       # reuse the same
special_ids = set(map(int, tok.all_special_ids or []))      # logic used by infer_chat_stream

generated_token_ids: list[int] = []
chain_latency_ms = 0.0
for _step in range(max_tokens):
    step_result = self._engine._run_chain(
        "",                                                  # empty prompt — context is in activation
        prep.candidates,
        prep.primary_pipeline,
        max_tokens=1,
        request_id=request_id,
        deadline=deadline,
        initial_activation=[float(t) for t in context_token_ids + generated_token_ids],
        **decode_controls,                                   # NO kv_session_id — pure stateless
    )
    chain_latency_ms += step_result.latency_ms
    if not step_result.activation:
        break
    next_id = int(round(float(step_result.activation[0])))
    if next_id in eos_ids:
        break
    generated_token_ids.append(next_id)

output_text = tok.decode(generated_token_ids, skip_special_tokens=True)
primary = ChainResult(
    request_id=request_id,
    text=output_text,
    activation=[float(t) for t in generated_token_ids],
    traces=_merge_traces_from_steps(...),
    latency_ms=chain_latency_ms,
)
```

Key points:
- **No `kv_session_id`** — avoids issue #2 completely. Each step re-prefills through both peers (O(N²) per request), but correctness comes first and it matches what the direct bench does today.
- **One new helper `_collect_eos_token_ids(tok)`** — same logic as lines 1714–1732 in the streaming path. Extract it into a module-level function so both paths use one copy.
- **Pure addition of a branch** — the existing `chain.run()` path stays for non-PyTorch pipelines and single-peer full-model cases (`len(pipeline) == 1`).
- **Special-token handling**: skip `special_ids` but do **not** strip whitespace from the decoded text (see 1C).

Cost: ~80 lines of code in `infer()`. No other files touched.

**Expected behavior after fix**: `/v1/completions` on Qwen3.5-9B sharded across 2 peers produces coherent multi-token output at roughly the same TPS as the direct bench (0.3–0.7 tok/s in the current WAN/no-KV-cache setup).

#### 1B. Expose `--autoregressive-sharded` kill-switch

**File**: `coordinator/api_server.py`

Add one CLI flag defaulting to `True`: `--autoregressive-sharded / --no-autoregressive-sharded`. Wire into a new `EngineConfig.autoregressive_sharded_enabled` field (default True). The loop in 1A checks this flag; operators can opt out to restore old behavior while debugging.

#### 1C. Stop stripping decoded tokens (fixes #4)

**File**: `peer/model_shard.py::decode_text`

Replace `return "".join(words).strip()` with `return "".join(words)`. The tokenizer already emits clean pieces; the `.strip()` is a leftover from the ToyRuntime fallback path. If callers genuinely need the text trimmed they can do it themselves.

Verify `tests/test_model_shard.py` doesn't depend on the strip behavior; add a regression test for a single-`"\n\n"`-token output round-trip.

### Phase 2 — P0 meta-tensor leak in the KV-cache path (#2)

**Goal**: Make the streaming path and any KV-cache-aware path actually work with disk-offloaded sharded models. This is the high-leverage fix — once done, sharded TPS jumps by the context-length factor vs. the Phase 1 stateless loop.

#### 2A. Investigate (1–2 hours)

Reproduce with `--log-level DEBUG` on the Lightning peers and capture the full Python traceback for the `GET was unable to find an engine` error. Three hypotheses, in priority order:

1. `transformers` `DynamicCache` constructs placeholder tensors for all `config.num_hidden_layers` slots at prefill time. When `use_cache=True` is passed, PyTorch inspects the whole cache, touching meta-device entries from our disk-offloaded layers (layers 16–31 on peer 1, 0–15 on peer 2).
2. `accelerate` pre-hooks are re-attached to meta-mapped layers even after our `remove_hook_from_module()` pass. The hook tries to materialize a meta parameter on first call.
3. `_hidden_to_next_token_payload` calls `self._lm_head(hidden)` and `lm_head` has leaked to meta (unlikely — device_map explicitly pins it).

Instrumentation to add inside `peer/model_shard.py::_forward_impl`:
```python
logging.info(
    "forward_impl_meta_check stage=%d/%d is_first=%s is_last=%s "
    "layers_meta=%d lm_head_device=%s embed_device=%s",
    stage, stages, is_first, is_last,
    sum(1 for b in self._blocks if b is None or _any_meta_param(b)),
    next(self._lm_head.parameters()).device if self._lm_head else None,
    next(self._embed_tokens.parameters()).device if self._embed_tokens else None,
)
```

#### 2B. Fix — stub out offloaded layers (most likely approach)

After `_build_selective_device_map(...)` + the existing `accelerate_hooks_removed` pass, replace every layer at a `"disk"` index with a cheap `nn.Identity()` **before** we hand the model to any code path that iterates `self._model.model.layers[:]`. This is what the Gemma 4 multimodal strip path already does (`_strip_multimodal_components`). Pulling the same pattern into the regular sharded path means the `DynamicCache` ends up with Identity layers at the offloaded positions — no meta tensors, no dispatcher errors.

Key refactor:
```python
# Extract from _strip_multimodal_components and re-use for regular sharded path
def _replace_offloaded_layers_with_identity(
    model: Any,
    kept_layer_indices: tuple[int, ...],
) -> int:
    """Replace every decoder layer not in kept_layer_indices with nn.Identity().

    This prevents meta-device tensors from leaking into DynamicCache /
    accelerate hooks when the coordinator enables use_cache=True for KV-aware
    decoding on a sharded peer.
    """
```

Call it from `PyTorchRuntime.__init__` right after `pytorch_layer_cleanup`. Verify: no behavior change for non-sharded single-peer runs; sharded peers show zero meta-device params in the `forward_impl_meta_check` log.

#### 2C. Regression test

Add `tests/test_sharded_kv_cache.py`:
- Fake a 4-layer `ToyRuntime`-backed model with `shard_index=0, total_shards=2` (runs layers 0–1).
- Run a forward pass with `kv_session_id="test", kv_store_activation=True` → no exception.
- Inspect the stored cache via `peer._kv_cache_get("test")` → verify only the 2 kept layers have non-empty K/V.

### Phase 3 — P0 SpecPipe tuple-unpack + poison pill (#3)

**Goal**: Make SpecPipe's pipelined execution survive a stage failure without crashing the whole generation.

#### 3A. Fix the tuple shape at line 357

**File**: `coordinator/specpipe_scheduler.py`

Change the error-path put at line 387 to match the 4-tuple shape:
```python
# before
out_q.put((None, tok_idx))
# after
out_q.put((None, "", False, tok_idx))
```

Then at line 357 check for the sentinel:
```python
activation, tok_prompt, is_continuation, tok_idx = item
if activation is None:
    # Upstream stage failed — propagate sentinel downstream and exit this token.
    if stage_idx < n_stages - 1:
        out_q.put((None, "", False, tok_idx))
    else:
        out_q.put((None, tok_idx))
    continue
```

#### 3B. Unit test

Add `tests/test_specpipe_scheduler.py::test_stage_failure_propagates`:
- Stage 0 fn raises `RuntimeError("boom")` on `tok_idx == 2`.
- `run_pipelined` returns the tokens it successfully generated before the failure (rather than crashing).

### Phase 4 — P1 Gemma 4 sharded adapter (#6)

**Goal**: Put Gemma 4 on equal footing with Qwen 3.5 for sharded inference. Currently single-peer unsharded works (benchmarked at 10.7 TPS on a T4) but a 2-peer shard crashes inside the layer forward.

#### 4A. Extend `_DecoderArchitecture` dataclass

**File**: `peer/model_shard.py`

Add two optional fields:
```python
@dataclass(frozen=True)
class _DecoderArchitecture:
    family: str
    layers: tuple[Any, ...]
    embed_tokens: Any
    position_embeddings: Any | None = None
    final_norm: Any | None = None
    rotary_emb: Any | None = None
    # --- Gemma 4 extras ---
    layer_types: tuple[str, ...] = ()      # config.layer_types (full_attention/sliding_attention)
    per_layer_embed: Any | None = None     # model.language_model.embed_tokens_per_layer
    per_layer_proj: Any | None = None      # model.language_model.per_layer_model_projection
    per_layer_norm: Any | None = None      # model.language_model.per_layer_projection_norm
```

Update `_detect_decoder_architecture` to populate the extras for the Gemma 4 branch (already exists via the multimodal-wrapper unwrap).

#### 4B. Extend the activation payload format

**File**: `peer/peer.proto`

Add two new repeated-float fields on `ForwardRequest`:
```proto
// Gemma 4 per-layer input sidecar. Shape [B, S, num_layers, hidden_per_layer],
// flattened row-major. Computed once by the first stage from input_ids via
// embed_tokens_per_layer + per_layer_model_projection, then carried to every
// later stage so linear/full attention layers can multiply by their slice.
repeated float per_layer_input = 40;
uint32 per_layer_input_hidden = 41;  // hidden_per_layer, for reshape on receipt
```

Regenerate `peer_pb2.py` / `peer_pb2_grpc.py`.

#### 4C. `family="gemma4"` branch in `_run_layers`

**File**: `peer/model_shard.py::_run_layers`

New branch triggered by `self._decoder_family == "gemma4"`:

```python
if self._decoder_family == "gemma4":
    # 1. Build position_embeddings per unique layer type
    unique_layer_types = {self._layer_types[i] for i in self.layer_indices}
    pe_per_type = {
        lt: self._rotary_emb(output, position_ids, lt) for lt in unique_layer_types
    }

    # 2. Build causal masks per unique layer type
    mask_per_type = _build_gemma4_masks(seq_len, past_len, unique_layer_types, device=self._device)

    # 3. Reshape the per_layer_input sidecar back to [B, S, L, hidden_per_layer]
    pli = _per_layer_input_from_payload(self._pending_per_layer_input, ...)

    # 4. Loop over selected layers, pass each its slice
    for layer_idx, block in zip(self.layer_indices, self._selected_layers):
        lt = self._layer_types[layer_idx]
        output = block(
            output,
            per_layer_input=pli[:, :, layer_idx, :],
            position_embeddings=pe_per_type[lt],
            attention_mask=mask_per_type[lt],
            position_ids=position_ids,
            past_key_values=layer_past,
        )
    return output, ...
```

First stage computes `pli` via `self._per_layer_embed(input_ids)` → `self._per_layer_proj(...)` → `self._per_layer_norm(...)`, serializes into the new protobuf field. Later stages deserialize the field from the activation payload.

#### 4D. Test

Integration test in `tests/test_gemma4_sharded.py`:
- Mock a tiny Gemma 4 config (2 layers, 1 full + 1 sliding).
- Run a 2-stage shard forward end-to-end.
- Compare last hidden state against the same model unsharded — token-level equality (up to fp precision).

### Phase 5 — P2 infrastructure / DX polish

5A. `ops/bench/` README enumerating `gemma4_direct_bench.py`, `qwen9b_sharded_bench.py`, usage, expected numbers.

5B. Install `flash-linear-attention` + `causal-conv1d` on both Lightning studios (add to `ops/nanode-snapshots/SETUP_GUIDE.md` or equivalent Lightning setup doc).

5C. Add a KV-aware variant of `ops/bench/qwen9b_sharded_bench.py` that passes `kv_session_id` + `kv_use_cached_activation=True` from step 1 onwards — only useful after Phase 2 lands, so gate it.

5D. `CHANGELOG` entry calling out the decoder-loop rework and the meta-tensor fix.

---

## Execution order + time estimate

```
Day 1  Phase 1A + 1B + 1C + tests        (~6 hrs)  ← unblocks all benchmarks immediately
Day 1  Phase 3A + 3B                     (~2 hrs)  ← SpecPipe is unblocking but optional
Day 2  Phase 2A (investigate)            (~2 hrs)
Day 2  Phase 2B + 2C                     (~4 hrs)  ← real TPS jump via KV cache
Day 3  Phase 4A–D (Gemma 4 sharded)      (~6 hrs)  ← independent of 1/2/3
Day 3  Phase 5A–D polish                 (~2 hrs)
```

Total: ~22 hours of focused work across three days. Phase 1 alone (6 hours) gives us a working sharded API; everything after that is performance and parity.

---

## Validation checklist

After all phases land:

- [ ] **1103 existing tests still pass** (`pytest tests/`).
- [ ] **New tests pass**: `test_sharded_kv_cache.py`, `test_specpipe_scheduler.py::test_stage_failure_propagates`, `test_gemma4_sharded.py`, `decode_text` regression test.
- [ ] **Qwen3.5-9B end-to-end via `/v1/completions`** (non-streaming, no `--specpipe`, no custom flags) generates coherent multi-token output on 2× T4 Lightning studios. Capture TPS for a 32-token response.
- [ ] **Qwen3.5-9B end-to-end via `/v1/chat/completions` streaming** generates the same output with KV-cache-aware TPS ≥ 5× the stateless loop.
- [ ] **Gemma 4 E4B-it sharded 2× T4** produces coherent output through the same API (tests Phase 4).
- [ ] **`ops/bench/qwen9b_sharded_bench.py` (stateless) still works** as the control reference.
- [ ] **Direct gRPC bench + KV-cache variant** hit ≥ 2 TPS on 2× T4 for Qwen3.5-9B (indicates Phase 2 fix is real, not just the wire format).

---

## Risk register

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Phase 2 root cause turns out to be deeper than "meta tensors in DynamicCache" (e.g. transformers internal path we can't patch) | Medium | Ship Phase 1 first — it works without KV cache. Phase 2 is a TPS optimization, not a correctness fix. |
| Phase 4 Gemma 4 proto change breaks wire compat for existing deployed peers | Low | New fields are optional and default-empty; old peers ignore them; `shard_total_layers > 0 and per_layer_input_hidden == 0` keeps the non-Gemma-4 path unchanged. |
| Phase 1 stateless loop is too slow over WAN to be useful | Low | It's already usable at 0.3–0.7 TPS in the bench; Phase 2 takes it higher. LAN/co-located peers will be much faster. |
| SpecPipe fix (#3) exposes additional latent bugs in the scheduler | Medium | Keep SpecPipe opt-in (`--specpipe`). Phase 1 doesn't depend on SpecPipe. |
| `_strip_multimodal_components`-style Identity replacement breaks ToyRuntime test determinism | Low | Only applies when layers are on the `disk` device in the device_map; ToyRuntime never hits that path. |

---

## Out of scope (explicit non-goals)

- On-chain HYDRA integration (separate plan).
- Autonomous rebalancing polish (already shipped).
- Agent GUI / Open WebUI wiring (separate plan; blocked only on the API working, which this plan delivers).
- Launch prep / demo video (post-implementation).
