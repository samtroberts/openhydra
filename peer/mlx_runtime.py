# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLX inference runtime for OpenHydra peers.

Uses Apple MLX for high-throughput inference on Apple Silicon (M1/M2/M3/M4).

DLPack is the designated bridge between the PyTorch network/tokenizer layer
and the MLX compute layer.  Every tensor handoff uses the DLPack protocol
(``mx.array(torch_tensor.detach())`` and ``torch.from_dlpack(mx_array)``)
to avoid NumPy as an intermediate copy.  A fallback to ``.numpy()`` is kept
for environments where a device-level DLPack path is unavailable (e.g.
pre-unification system Metal allocators), but the primary path is zero-copy.

Phase 1 scope (single-peer, full model):
  - ``stream_generate`` for the end-to-end generation hot path
  - ``encode_prompt`` for pre-tokenisation (DLPack torch→mlx)
  - ``_warmup`` to JIT-compile Metal kernels before the first real request
  - Full ``runtime_profile()`` compatible with the rest of the system

Phase 3 scope (multi-peer, layer sharding):
  - ``forward()`` with ``total_stages > 1``: NOT_YET_IMPLEMENTED
    (the hidden-state embedding/extraction helpers are stubbed out with
    ``NotImplementedError`` so the compiler path is clear when Phase 3 lands)
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    import mlx.core as _mx
    import mlx.nn as _nn
    import torch as _torch

__all__ = ["MLXRuntime"]

_T = TypeVar("_T")


class _MlxWatchdog:
    """Timeout wrapper for MLX Metal GPU operations.

    MLX ``mx.eval()`` calls are synchronous and can hang indefinitely on
    memory pressure or Metal shader stalls.  This watchdog runs MLX compute
    in a dedicated single-thread executor and enforces a wall-clock deadline
    via ``future.result(timeout=...)``.

    Design constraints:
        - ``signal.alarm`` cannot be used in gRPC threadpool workers (not main thread).
        - Metal kernels cannot be forcibly cancelled — a timed-out operation
          becomes a "zombie" that holds the GPU until it naturally completes.
        - While a zombie is in-flight, the busy-guard rejects new submissions
          to prevent queue buildup.

    Args:
        default_timeout_s: Maximum seconds per computation (default 30.0).
    """

    def __init__(self, default_timeout_s: float = 30.0) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mlx-watchdog",
        )
        self._default_timeout_s = max(1.0, float(default_timeout_s))
        self._busy = threading.Event()

    @property
    def is_busy(self) -> bool:
        """True if a computation is currently in-flight (including zombies)."""
        return self._busy.is_set()

    def run(
        self,
        fn: Callable[..., _T],
        *args: Any,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> _T:
        """Submit *fn* to the MLX executor and wait up to *timeout_s* seconds.

        Raises:
            RuntimeError: If a previous computation is still in-flight (zombie).
            TimeoutError: If *fn* does not complete within the deadline.
        """
        if self._busy.is_set():
            raise RuntimeError(
                "mlx_watchdog_busy: previous computation still running — "
                "GPU may be hung.  Wait for completion or restart the peer."
            )
        self._busy.set()
        effective_timeout = timeout_s if timeout_s is not None else self._default_timeout_s
        future = self._executor.submit(fn, *args, **kwargs)
        try:
            result = future.result(timeout=effective_timeout)
        except FuturesTimeoutError:
            # Do NOT clear _busy — the zombie Metal kernel is still running.
            logging.error(
                "mlx_watchdog_timeout: computation exceeded %.1f s deadline — "
                "Metal GPU may be hung.  Marking runtime as unhealthy.",
                effective_timeout,
            )
            raise TimeoutError(
                f"mlx_watchdog_timeout: {effective_timeout:.1f}s exceeded"
            )
        except Exception:
            # Non-timeout error — computation finished (with error), GPU is free.
            self._busy.clear()
            raise
        else:
            # Success — GPU is free.
            self._busy.clear()
            return result

    def shutdown(self) -> None:
        """Best-effort shutdown of the executor thread."""
        self._executor.shutdown(wait=False)

# Maps canonical quantization_mode names → bit-width.
# "fp32" / "fp16" are floating-point modes (bits used only for memory estimates).
_MLX_QUANT_BITS: dict[str, int] = {
    "fp32": 0,
    "fp16": 16,
    "int8": 8,
    "int4": 4,
}


class MLXRuntime:
    """Apple MLX inference backend.

    Typical performance on Apple Silicon (single peer, full model):
        - Qwen3.5-0.8B: ~100-200 tok/s  (vs ~1.3 tok/s with pytorch_auto/MPS)
        - TTFT warm:     ~50-150 ms
        - TTFT cold:     ~1-3 s after warmup (vs ~34 s without)

    DLPack contract
    ~~~~~~~~~~~~~~~
    Every boundary between PyTorch (tokenizer / gRPC layer) and MLX (compute)
    MUST go through ``_torch_to_mx`` / ``_mx_to_torch``.  Do NOT use NumPy
    as an intermediate — it doubles memory and kills the zero-copy guarantee.

    Args:
        config: ``ToyShardConfig`` dataclass.  Relevant fields:
            - ``runtime_model_id``: HuggingFace model ID
            - ``runtime_warmup_on_start``: run warmup pass on init
            - ``shard_index`` / ``total_shards``: layer-range for Phase 3
            - ``runtime_max_context_tokens``: hard cap on sequence length
    """

    def __init__(self, config: Any) -> None:
        import mlx.core as mx
        import mlx.nn as nn  # noqa: F401  (needed for eval)
        from mlx_lm import load as mlx_load

        self._mx = mx
        self.model_name = str(getattr(config, "runtime_model_id", "Qwen/Qwen3.5-0.8B"))
        self.shard_index = int(getattr(config, "shard_index", 0))
        self.total_shards = int(getattr(config, "total_shards", 1))
        self.max_context_tokens = int(getattr(config, "runtime_max_context_tokens", 2048))
        self._kv_cache: dict[str, Any] = {}

        # ── Watchdog: timeout wrapper for mx.eval() calls ────────────────────
        _timeout_s = float(getattr(config, "runtime_mlx_eval_timeout_s", 30.0))
        self._watchdog = _MlxWatchdog(default_timeout_s=_timeout_s)

        # Requested quantization mode (already normalised by ToyShardConfig caller).
        _req_mode = str(getattr(config, "quantization_mode", "fp32"))
        _req_bits = _MLX_QUANT_BITS.get(_req_mode, 0)

        logging.info(
            "mlx_runtime: loading model=%s requested_quant=%s",
            self.model_name, _req_mode,
        )
        _t0 = time.perf_counter()
        self._model, self._tokenizer = mlx_load(self.model_name)
        _load_s = time.perf_counter() - _t0
        logging.info("mlx_runtime: loaded in %.1f s", _load_s)

        # Force Metal allocation of all parameters now, not lazily on first use.
        self._watchdog.run(mx.eval, self._model.parameters())

        # Apply runtime quantization if requested and the checkpoint is not
        # already quantized (pre-quantized MLX models, e.g. from mlx-community,
        # load as QuantizedLinear layers and must NOT be re-quantized).
        if _req_bits in (4, 8) and not self._is_model_quantized():
            self._apply_mlx_quantization(bits=_req_bits)

        # Detect effective quantization from the loaded (and possibly just
        # quantized) model rather than trusting the config alone — a
        # pre-quantized 4-bit checkpoint overrides a "fp32" request.
        self.quantization_mode, self.quantization_bits = (
            self._detect_mlx_quantization(
                fallback_mode=_req_mode,
                fallback_bits=_req_bits,
            )
        )

        # Count parameters for runtime_profile.
        # model.parameters() returns a nested dict in MLX (not (name, tensor) pairs).
        # tree_flatten collapses it to [(key_path, mx.array), ...].
        from mlx.utils import tree_flatten as _tree_flatten
        n_params = sum(v.size for _, v in _tree_flatten(self._model.parameters()))
        self._param_count = int(n_params)

        # Discover total transformer layer count (needed for profile + Phase 3).
        _inner = getattr(self._model, "model", self._model)
        _layers = getattr(_inner, "layers", None) or []
        self._total_layers = int(len(_layers))

        # ── Phase 4A: Pipeline parallelism ────────────────────────────────
        self._mlx_world_size = int(getattr(config, "mlx_world_size", 1))
        self._mlx_rank = int(getattr(config, "mlx_rank", 0))
        self._parallel: Any = None
        if self._mlx_world_size > 1:
            from peer.mlx_parallel import PipelineParallelMLX
            self._parallel = PipelineParallelMLX(
                model=self._model,
                world_size=self._mlx_world_size,
                rank=self._mlx_rank,
                async_eval=True,
            )
            logging.info(
                "mlx_parallel_enabled: rank=%d/%d layers=[%d,%d)",
                self._parallel.rank, self._parallel.world_size,
                self._parallel.layer_start, self._parallel.layer_end,
            )

        # Memory estimate: bit width → bytes per parameter.
        # Use 16 as default when bits == 0 (floating-point modes without explicit bits).
        _bits_pp = self.quantization_bits if self.quantization_bits > 0 else 16
        _est_mem_mb = int(self._param_count * (_bits_pp / 8) / (1024 * 1024))

        self._runtime_profile: dict[str, Any] = {
            "backend":                  "mlx",
            "target":                   "metal",
            "runtime_model_id":         self.model_name,
            "quantization_mode":        self.quantization_mode,
            "quantization_bits":        self.quantization_bits,
            "gpu_available":            True,
            "param_count":              self._param_count,
            "estimated_memory_mb":      _est_mem_mb,
            "estimated_tokens_per_sec": 120.0,
            "layer_start":              0,
            "layer_end":                self._total_layers,
            "total_layers":             self._total_layers,
        }

        # ── KV prefix cache (RadixKVCache) ───────────────────────────────────
        self._radix_cache = None
        self._prompt_cache_supported = False
        if bool(getattr(config, "runtime_kv_radix_cache_enabled", False)):
            try:
                from peer.kv_compaction._radix_cache import RadixKVCache
                self._radix_cache = RadixKVCache(
                    max_entries=int(getattr(config, "runtime_kv_radix_cache_max_entries", 128)),
                    min_prefix_len=int(getattr(config, "runtime_kv_radix_cache_min_prefix_len", 16)),
                )
                # Verify that the installed mlx_lm supports prompt_cache kwarg
                # in generate_step (available since mlx_lm ~0.20).
                import inspect
                from mlx_lm.generate import generate_step as _gs
                _gs_params = inspect.signature(_gs).parameters
                self._prompt_cache_supported = "prompt_cache" in _gs_params
                if not self._prompt_cache_supported:
                    logging.warning(
                        "mlx_runtime: mlx_lm does not support prompt_cache — "
                        "RadixKVCache disabled (upgrade mlx_lm >= 0.20)"
                    )
                    self._radix_cache = None
                else:
                    logging.info(
                        "mlx_runtime: RadixKVCache enabled (max_entries=%d, min_prefix=%d)",
                        self._radix_cache._max_entries,
                        self._radix_cache._min_prefix_len,
                    )
            except ImportError as exc:
                logging.warning("mlx_runtime: RadixKVCache import failed: %s", exc)
                self._radix_cache = None

        # Phase W: warmup.
        if bool(getattr(config, "runtime_warmup_on_start", False)):
            self._warmup()

    # ── Quantization helpers ───────────────────────────────────────────────────

    def _is_model_quantized(self) -> bool:
        """Return True if the loaded model already has quantized (QuantizedLinear) layers.

        MLX stores quantized weights alongside ``scales`` and ``biases`` tensors.
        Checking for ``"scales"`` anywhere in the flat parameter tree is a
        reliable signal that ``mlx.nn.quantize()`` (or a pre-quantized
        checkpoint) has already been applied.
        """
        try:
            from mlx.utils import tree_flatten as _tree_flatten
            params = dict(_tree_flatten(self._model.parameters()))
            return any("scales" in k for k in params)
        except Exception:
            return False

    def _detect_mlx_quantization(
        self,
        fallback_mode: str = "fp32",
        fallback_bits: int = 0,
    ) -> tuple[str, int]:
        """Infer (quantization_mode, quantization_bits) from the loaded model.

        Checks for quantized layers first (``scales`` presence), then inspects
        floating-point dtypes.  Falls back to the caller-supplied defaults if
        detection fails.
        """
        try:
            from mlx.utils import tree_flatten as _tree_flatten
            params = dict(_tree_flatten(self._model.parameters()))

            # Quantized checkpoint: look for weight tensor dtype to distinguish
            # 4-bit (uint32 packed) from 8-bit (uint8).
            if any("scales" in k for k in params):
                for k, v in params.items():
                    if k.endswith(".weight") and hasattr(v, "dtype"):
                        dtype_str = str(v.dtype)
                        if "uint32" in dtype_str:
                            return "int4", 4
                        if "uint8" in dtype_str:
                            return "int8", 8
                # Has scales but couldn't pin the dtype — assume 4-bit.
                return "int4", 4

            # Floating-point checkpoint: report fp16 (MLX default for LLMs).
            for _, v in params.items():
                if hasattr(v, "dtype"):
                    dtype_str = str(v.dtype)
                    if "bfloat16" in dtype_str or "float16" in dtype_str:
                        return "fp16", 16

        except Exception as exc:
            logging.warning("mlx_runtime: quantization detection failed: %s", exc)

        return fallback_mode, fallback_bits

    def _apply_mlx_quantization(self, bits: int) -> None:
        """Apply in-place MLX quantization to the loaded model.

        Uses ``mlx.nn.quantize()`` with the requested bit-width.  Errors are
        caught and logged — the model continues in fp16 rather than crashing.
        """
        try:
            import mlx.core as mx
            import mlx.nn as nn
            nn.quantize(self._model, bits=bits)
            self._watchdog.run(mx.eval, self._model.parameters())
            logging.info("mlx_runtime: applied %d-bit quantization to model", bits)
        except Exception as exc:
            logging.warning(
                "mlx_runtime: %d-bit quantization failed (%s); continuing in fp16",
                bits, exc,
            )

    # ── Warmup ────────────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """JIT-compile Metal shaders with a tiny dummy generation.

        Moves Metal shader compilation (~30 s cold) from the first real request
        to peer startup.  Errors are caught and logged — never propagated.
        """
        _t0 = time.perf_counter()
        try:
            logging.info(
                "mlx_runtime_warmup: starting model=%s device=%s",
                self.model_name, str(self._mx.default_device()),
            )
            from mlx_lm import stream_generate
            for _resp in stream_generate(
                self._model,
                self._tokenizer,
                prompt="hi",
                max_tokens=2,
            ):
                pass
        except Exception as exc:
            logging.warning(
                "mlx_runtime_warmup_failed: %s — first inference will be slow", exc,
            )
        else:
            logging.info(
                "mlx_runtime_warmup: complete in %.1f s",
                time.perf_counter() - _t0,
            )

    # ── DLPack bridges ────────────────────────────────────────────────────────

    def _torch_to_mx(self, t: Any) -> Any:
        """Zero-copy PyTorch tensor → MLX array via DLPack.

        This is the REQUIRED path for any PyTorch tensor entering the MLX
        compute layer (e.g. tokenizer output, activation tensors from a
        PyTorch-backend peer upstream in the pipeline).

        Falls back to an explicit ``numpy()`` copy only if the DLPack path
        raises (e.g. device-level incompatibility), and logs a warning.
        """
        try:
            # Primary path: zero-copy via __dlpack__ protocol.
            return self._mx.array(t.detach())
        except (TypeError, RuntimeError, AttributeError):
            logging.warning(
                "mlx_runtime: DLPack torch→mlx unavailable, falling back to copy"
            )
            return self._mx.array(t.detach().numpy())

    def _mx_to_torch(self, a: Any) -> Any:
        """Zero-copy MLX array → PyTorch tensor via DLPack.

        Required for any MLX activation tensor that must be handed back to
        a downstream PyTorch-backend peer or the gRPC serialisation layer.
        """
        import torch
        return torch.from_dlpack(a)

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def encode_prompt(self, prompt: str, max_tokens: int) -> list[float]:
        """Tokenise *prompt* and return token IDs as floats.

        mlx-lm's ``TokenizerWrapper`` is not directly callable but exposes an
        ``encode(text) -> list[int]`` method.  We use this native path rather
        than ``return_tensors="pt"`` + DLPack; the DLPack bridges are reserved
        for activation tensors arriving from an upstream PyTorch-backend peer.

        Returns:
            List of token ID floats, or ``[]`` on failure.
        """
        try:
            max_len = max(4, min(self.max_context_tokens, max(1, int(max_tokens)) * 4))
            ids: list[int] = self._tokenizer.encode(prompt or "")
            if not ids:
                return []
            # Truncate to context window.
            ids = ids[:max_len]
            return [float(t) for t in ids]
        except Exception as exc:
            logging.warning("mlx_encode_prompt_failed: %s", exc)
            return []

    # ── Core forward pass ─────────────────────────────────────────────────────

    def forward(
        self,
        prompt: str,
        activation: list[float],
        max_tokens: int,
        stage_index: int = 0,
        total_stages: int = 1,
        kv_session_id: str | None = None,
        kv_store_activation: bool = False,
        kv_use_cached_activation: bool = False,
        request_id: str | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
    ) -> list[float]:
        """Run inference and return token IDs (or hidden state) as floats.

        Phase 1 (``total_stages == 1``):
            Runs the full model via ``stream_generate`` and returns the
            generated token IDs as ``[t1_f, t2_f, ...]``.

        Phase 3 (``total_stages > 1``):
            Multi-peer layer sharding — raises ``NotImplementedError``.
            Will be wired when Phase 3 DHT layer-range announcements land.

        DLPack note:
            All tensor transfers cross the PyTorch↔MLX boundary via
            ``_torch_to_mx`` / ``_mx_to_torch`` (zero-copy DLPack).
        """
        if total_stages > 1:
            raise NotImplementedError(
                "mlx_runtime: multi-stage layer sharding (total_stages > 1) "
                "requires Phase 3.  Use --total-shards 1 with --runtime-backend mlx."
            )

        # ── Build sampler via make_sampler() ─────────────────────────────────
        # mlx-lm ≥0.21 passes sampling params through a sampler callable, not
        # as bare kwargs to stream_generate/generate_step.
        from mlx_lm.sample_utils import make_sampler

        # decode_do_sample=False → greedy (temp=0.0); True → sampling.
        if decode_do_sample is False:
            _temp = 0.0
        elif decode_temperature is not None:
            _temp = max(0.0, float(decode_temperature))
        else:
            _temp = 0.0   # default: greedy

        _top_p = float(decode_top_p) if decode_top_p is not None else 0.0
        _top_k = int(decode_top_k) if (decode_top_k is not None and int(decode_top_k) > 0) else 0

        sampler = make_sampler(temp=_temp, top_p=_top_p, top_k=_top_k)

        # ── RadixKVCache: prefix lookup ──────────────────────────────────────
        _cached_prompt_cache = None
        _prompt_token_ids: tuple[int, ...] = ()
        if self._radix_cache is not None:
            _prompt_token_ids = tuple(self._tokenizer.encode(str(prompt or "")))
            _cached_prompt_cache, _prefix_len = self._radix_cache.lookup(_prompt_token_ids)
            if _prefix_len > 0 and _cached_prompt_cache is not None:
                # Deep-copy so generation's in-place updates don't corrupt the
                # cached entry.  KVCache objects are small (metadata + mx.array
                # references) — the underlying Metal buffers are NOT copied.
                _cached_prompt_cache = copy.deepcopy(_cached_prompt_cache)
                logging.debug(
                    "mlx_radix_cache_hit: prefix_len=%d prompt_len=%d",
                    _prefix_len, len(_prompt_token_ids),
                )
            else:
                _cached_prompt_cache = None

        # ── Generate (watchdog-guarded) ───────────────────────────────────────
        from mlx_lm import stream_generate

        # We capture the prompt_cache produced by generation for later caching.
        _generation_prompt_cache: list[Any] = []

        def _generate_impl() -> list[int]:
            tokens: list[int] = []
            tps_last: float = 0.0

            gen_kwargs: dict[str, Any] = dict(
                max_tokens=max(1, int(max_tokens)),
                sampler=sampler,
            )
            if _cached_prompt_cache is not None:
                gen_kwargs["prompt_cache"] = _cached_prompt_cache

            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt=str(prompt or ""),
                **gen_kwargs,
            ):
                tokens.append(int(response.token))
                tps_last = float(response.generation_tps or 0.0)
                if response.finish_reason is not None:
                    break

            # Capture the prompt_cache for post-generation caching.
            if _cached_prompt_cache is not None:
                _generation_prompt_cache.extend(_cached_prompt_cache)

            return tokens

        _t0 = time.perf_counter()
        tokens = self._watchdog.run(_generate_impl)
        _elapsed = time.perf_counter() - _t0
        logging.debug(
            "mlx_runtime: generated %d tokens in %.3f s",
            len(tokens), _elapsed,
        )

        # ── RadixKVCache: insert after generation ────────────────────────────
        if self._radix_cache is not None and _prompt_token_ids:
            # Store the full prompt tokens (not generated tokens) as the cache key.
            # The prompt_cache was updated in-place during generation.
            if _cached_prompt_cache is not None and _generation_prompt_cache:
                # Cache was used and updated — store the updated version.
                self._radix_cache.insert(
                    _prompt_token_ids,
                    copy.deepcopy(_generation_prompt_cache),
                )
            elif _cached_prompt_cache is None:
                # No cache was used — we need to generate a fresh prompt_cache
                # for this prefix.  Create one by doing a cache-priming pass.
                # For efficiency, we only cache if we have prompt_cache support
                # and the prompt is long enough to benefit.
                try:
                    from mlx_lm.models.cache import make_prompt_cache as _make_pc
                    fresh_cache = _make_pc(self._model)
                    # Prime the cache with the prompt tokens.
                    import mlx.core as mx
                    prompt_arr = mx.array(list(_prompt_token_ids))
                    self._model(prompt_arr[None], cache=fresh_cache)
                    mx.eval([c.state for c in fresh_cache])
                    self._radix_cache.insert(
                        _prompt_token_ids,
                        fresh_cache,
                    )
                except Exception as exc:
                    logging.debug("mlx_radix_cache_insert_failed: %s", exc)

        return [float(t) for t in tokens]

    # ── Profile ───────────────────────────────────────────────────────────────

    def runtime_profile(self) -> dict[str, Any]:
        return dict(self._runtime_profile)

    # ── Batched forward pass ───────────────────────────────────────────────────

    def forward_batch(self, items: list[Any]) -> list[list[float]]:
        """True batched autoregressive generation via mx.concatenate.

        Tokenises all prompts, right-pads to uniform length, stacks along the
        batch dimension as [B, prompt_len], then calls model(current_ids) once
        per decode step.  The resulting [B, seq, vocab] logits are argmax'd at
        the last position to pick the next token for each item.

        Each item stops when it reaches its own max_tokens or generates an EOS.

        MLX only supports single-stage (full model) mode.  Raises
        NotImplementedError for total_stages > 1 (same as forward()).

        Note: RadixKVCache is intentionally SKIPPED for batch>1.  The batch
        path uses a single concatenated [B, seq] tensor, which is incompatible
        with per-request prompt_cache objects.  The single-request path
        (forward()) handles caching instead.
        """
        import mlx.core as mx

        if not items:
            return []

        _stages = [int(item.total_stages) for item in items]
        if any(s > 1 for s in _stages):
            raise NotImplementedError(
                f"mlx_runtime: multi-stage layer sharding not supported (got total_stages={_stages}); "
                "use total_shards=1 with --runtime-backend mlx."
            )

        # Tokenise all prompts → list of int lists.
        all_tokens: list[list[int]] = []
        for item in items:
            max_len = max(4, min(self.max_context_tokens, max(1, int(item.max_tokens)) * 4))
            ids = list(self._tokenizer.encode(item.prompt or "")) or [0]
            ids = ids[:max_len]
            all_tokens.append(ids)

        # Right-pad to uniform length for rectangular [B, prompt_len] array.
        max_prompt_len = max(len(t) for t in all_tokens)
        pad_id = int(getattr(self._tokenizer, "pad_token_id", None) or 0)
        padded = [t + [pad_id] * (max_prompt_len - len(t)) for t in all_tokens]

        B = len(items)
        # Stack: [B, prompt_len]
        current_ids = mx.concatenate(
            [mx.array([p], dtype=mx.uint32) for p in padded],
            axis=0,
        )

        max_new = max(item.max_tokens for item in items)
        eos_id = int(getattr(self._tokenizer, "eos_token_id", None) or 0)

        def _batch_decode_impl() -> list[list[int]]:
            nonlocal current_ids
            generated: list[list[int]] = [[] for _ in range(B)]
            active = [True] * B

            for _ in range(max(1, int(max_new))):
                # ONE GPU call for all B requests.
                logits = self._model(current_ids)  # [B, seq_len, vocab_size]
                mx.eval(logits)

                # Greedy: argmax over vocab at the last sequence position.
                next_tok_mx = mx.argmax(logits[:, -1, :], axis=-1)  # [B]
                mx.eval(next_tok_mx)
                next_tok = next_tok_mx.tolist()
                if isinstance(next_tok, int):
                    next_tok = [next_tok]

                all_done = True
                for i, tok in enumerate(next_tok):
                    if active[i]:
                        generated[i].append(int(tok))
                        if int(tok) == eos_id or len(generated[i]) >= items[i].max_tokens:
                            active[i] = False
                        else:
                            all_done = False

                if all_done:
                    break

                # Append new column of tokens and continue.
                next_col = mx.array([[int(t)] for t in next_tok], dtype=mx.uint32)
                current_ids = mx.concatenate([current_ids, next_col], axis=1)

            return generated

        generated = self._watchdog.run(_batch_decode_impl)
        return [[float(t) for t in tokens] for tokens in generated]

    # ── Pass 6: VRAM monitoring ─────────────────────────────────────────────

    def _vram_usage_pct(self) -> float:
        """Return current Metal GPU memory utilisation as a fraction [0.0, 1.0].

        Uses ``mx.metal.get_active_memory()`` / recommended working set size.
        Returns ``0.0`` if Metal memory info is unavailable.
        """
        try:
            mx = self._mx
            if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
                active = mx.metal.get_active_memory()
                info = mx.metal.device_info()
                total = info.get("recommendedMaxWorkingSetSize", 0)
                if total > 0:
                    return active / total
        except Exception:
            pass
        return 0.0

    # ── Phase 2B: Reshard (logical bounds update) ───────────────────────────

    def reshard(self, new_layer_start: int, new_layer_end: int, total_layers: int) -> bool:
        """Update logical layer bounds without physical weight reload.

        MLXRuntime loads the full model into Apple Silicon Unified Memory,
        so resharding only needs to update the metadata — no weight slicing
        or reloading is required.

        Args:
            new_layer_start: New inclusive start layer.
            new_layer_end: New exclusive end layer.
            total_layers: Full model depth.

        Returns:
            ``True`` — always succeeds for MLX (logical update only).
        """
        logger.info(
            "mlx_reshard_logical: new_range=[%d, %d) total=%d (no physical reload needed)",
            new_layer_start, new_layer_end, total_layers,
        )
        return True

    # ── Phase 3 stubs (hidden-state exchange, DLPack path) ───────────────────

    def _activation_to_hidden(self, activation: list[float]) -> Any:
        """Deserialise a gRPC float-list into an MLX hidden-state tensor.

        Incoming format (same as PyTorchRuntime):
            [seq_len_f, hidden_size_f, v0, v1, …]

        The float data is converted directly to ``mx.array`` — no PyTorch
        intermediate is needed here because the payload was already serialised
        as a plain float list by the previous peer.  If the previous peer sent
        a PyTorch tensor directly (future optimisation), use ``_torch_to_mx``.

        Raises:
            NotImplementedError: until Phase 3 is wired.
        """
        raise NotImplementedError(
            "mlx_runtime: hidden-state activation exchange requires Phase 3"
        )

    def _hidden_to_payload(self, hidden: Any) -> list[float]:
        """Serialise an MLX hidden-state tensor to a gRPC float list.

        Output format (same as PyTorchRuntime):
            [seq_len_f, hidden_size_f, v0, v1, …]

        Uses ``_mx_to_torch`` (DLPack) then ``.tolist()`` for the final
        CPU serialisation, avoiding a redundant ``mx.eval`` → numpy step.

        Raises:
            NotImplementedError: until Phase 3 is wired.
        """
        raise NotImplementedError(
            "mlx_runtime: hidden-state payload serialisation requires Phase 3"
        )
