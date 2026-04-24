# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Standalone coordinator-side final head + sampler (Phase 6).

Path A Phase 5 let a co-located peer's runtime serve as the
``HeadSampler``'s borrow source — but only when the coordinator process
also runs a peer that owns the relevant weights. That works for the
2-node case (Mac = coord = stage 0) but precludes the **true Petals
topology**: client / coordinator on its own host, peers in a swarm,
coordinator orchestrates + samples but runs no transformer layers
itself.

Phase 6 fills the gap by building a minimal "head-only" module loaded
directly on the coordinator. We retain ``embed_tokens`` and ``norm``
(and ``lm_head`` for untied models) from a freshly-loaded model, then
*discard* the transformer layer modules so the coordinator's memory
footprint is just ``vocab × hidden + hidden`` (plus tied lm_head for
free on Qwen3.5).

Dual backend
------------
The head loader supports **two backends**:

* **MLX** (default on Apple Silicon) — loads via ``mlx_lm`` and uses
  ``mx.eval`` + ``embed_tokens.as_linear`` for the matmul. Natural fit
  for Mac coordinators.
* **PyTorch** (for Linux / any non-Mac coordinator) — loads via HF
  ``AutoModelForCausalLM``, runs on CPU (or CUDA if available), uses
  ``torch.matmul`` for the head. Required for the all-LAN True Petals
  benchmark where the coordinator is a cheap Linux studio rather than
  an Apple Silicon box.

``load_standalone_head(hf_model_id, backend='auto' | 'mlx' | 'pytorch')``
picks the backend. ``'auto'`` prefers MLX when ``mlx_lm`` is importable
(Apple Silicon), otherwise falls back to PyTorch.

Both backends expose the same :class:`StandaloneHead` surface:

* ``apply_final_head(hidden_state, *, packed_bytes, decode_*) -> int``
* ``_has_final_head: bool`` (always True)
* ``_is_last_shard: bool`` (always True — for legacy gate compatibility)

So ``HeadSampler.sample`` is oblivious to which backend owns the
weights — identical to borrowing from a peer runtime.

Memory accounting (Qwen3.5-2B, tied embeds)
-------------------------------------------

| Backend | embed_tokens | final_norm | lm_head | layers (pruned) | net |
|---|---|---|---|---|---|
| MLX-8bit | ~500 MB | ~8 KB | free (tied) | ~1.5 GB freed | ~500 MB |
| PyTorch fp32 | ~2 GB | ~8 KB | free (tied) | ~6 GB freed | ~2 GB |
| PyTorch bf16 | ~1 GB | ~4 KB | free (tied) | ~3 GB freed | ~1 GB |

Design notes
------------
* The PyTorch backend keeps the embed/norm weights on CPU by default
  (``device_map='cpu'``) — a Linux coordinator usually doesn't have a
  GPU, and the head matmul on CPU for one hidden-state vector is ~2 ms
  for 2 K × 248 K, negligible against the relay RTT.
* Both backends handle ``packed_bytes`` via ``openhydra_network``
  DLPack decode. Falls back to the list-float wire format when the
  Rust wheel isn't available.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Backend selection helpers ─────────────────────────────────────────


def _mlx_available() -> bool:
    """True iff the MLX stack is importable (Apple Silicon-only)."""
    try:
        import mlx_lm  # noqa: F401
        return True
    except ImportError:
        return False


def _pytorch_available() -> bool:
    """True iff torch + transformers are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_backend(backend: str) -> str:
    """Resolve a backend name, applying 'auto' preference rules.

    Rules:
      * 'auto' → prefer MLX if importable, else PyTorch.
      * 'mlx'  → explicit MLX; raises if unavailable.
      * 'pytorch' → explicit PyTorch; raises if unavailable.
    """
    b = str(backend or "auto").strip().lower()
    if b == "auto":
        if _mlx_available():
            return "mlx"
        if _pytorch_available():
            return "pytorch"
        raise RuntimeError(
            "standalone_head: neither mlx_lm nor torch+transformers "
            "is importable — install one of the two backends"
        )
    if b == "mlx":
        if not _mlx_available():
            raise RuntimeError(
                "standalone_head: backend='mlx' requested but mlx_lm is "
                "not installed. Install with `pip install mlx-lm`."
            )
        return "mlx"
    if b == "pytorch":
        if not _pytorch_available():
            raise RuntimeError(
                "standalone_head: backend='pytorch' requested but "
                "torch+transformers are not installed."
            )
        return "pytorch"
    raise RuntimeError(
        f"standalone_head: unknown backend={b!r} "
        "(expected 'auto' | 'mlx' | 'pytorch')"
    )


# ── StandaloneHead — dual-backend container ───────────────────────────


class StandaloneHead:
    """Coordinator-owned final head module, backed by MLX or PyTorch.

    Implements the duck-typed runtime interface that
    ``coordinator.head_sampler.HeadSampler`` expects:

    * ``apply_final_head(hidden_state, *, packed_bytes, decode_*) -> int``
    * ``_has_final_head`` (always ``True`` here)
    * ``_is_last_shard`` (always ``True`` — legacy gate compatibility)
    """

    def __init__(
        self,
        *,
        backend: str,
        hf_model_id: str,
        norm_module: Any,
        embed_tokens_module: Any,
        lm_head_module: Optional[Any],
        tie_word_embeddings: bool,
        hidden_size: int,
        vocab_size: int,
        # PyTorch backend only — retained so apply_final_head can move
        # the incoming hidden state to the right device/dtype.
        torch_device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
    ) -> None:
        self._backend = str(backend)
        self._hf_model_id = str(hf_model_id)
        self._shard_norm = norm_module
        self._shard_embed_tokens = embed_tokens_module
        self._shard_lm_head = lm_head_module
        self._tie_word_embeddings = bool(tie_word_embeddings)
        self._hidden_size = int(hidden_size)
        self._vocab_size = int(vocab_size)
        self._torch_device = torch_device
        self._torch_dtype = torch_dtype
        # Legacy duck-type signals.
        self._has_final_head = True
        self._is_last_shard = True

    # ── Public surface — HeadSampler calls this ─────────────────────────

    def apply_final_head(
        self,
        hidden_state: Any,
        *,
        packed_bytes: bytes | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
    ) -> int:
        """Apply ``final_norm`` + ``lm_head`` + sampler to a hidden state.

        Same contract as :meth:`peer.mlx_runtime.MLXRuntime.apply_final_head`
        and :meth:`peer.model_shard.PyTorchRuntime.apply_final_head` —
        accepts either a list-format hidden state or zero-copy
        ``packed_bytes`` from the Rust encoder; returns a single
        sampled token id.
        """
        if self._backend == "mlx":
            return self._apply_mlx(
                hidden_state, packed_bytes=packed_bytes,
                decode_do_sample=decode_do_sample,
                decode_temperature=decode_temperature,
                decode_top_p=decode_top_p,
                decode_top_k=decode_top_k,
                decode_seed=decode_seed,
            )
        if self._backend == "pytorch":
            return self._apply_pytorch(
                hidden_state, packed_bytes=packed_bytes,
                decode_do_sample=decode_do_sample,
                decode_temperature=decode_temperature,
                decode_top_p=decode_top_p,
                decode_top_k=decode_top_k,
                decode_seed=decode_seed,
            )
        raise RuntimeError(
            f"standalone_head: unknown backend={self._backend!r}"
        )

    # ── MLX backend ────────────────────────────────────────────────────

    def _apply_mlx(
        self,
        hidden_state: Any,
        *,
        packed_bytes: bytes | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
    ) -> int:
        import mlx.core as mx

        h = self._mlx_activation_to_hidden(hidden_state, packed_bytes=packed_bytes)
        mx.eval(h)
        h = self._shard_norm(h)
        if self._tie_word_embeddings:
            logits = self._shard_embed_tokens.as_linear(h)
        else:
            if self._shard_lm_head is None:
                raise RuntimeError(
                    "standalone_head: untied model but no lm_head loaded"
                )
            logits = self._shard_lm_head(h)
        mx.eval(logits)
        # Sample: match MLXRuntime._sample_from_logits semantics.
        last_logits = logits[:, -1, :]
        if decode_do_sample is False or not decode_temperature:
            return int(mx.argmax(last_logits, axis=-1).item())
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(
            temp=max(1e-6, float(decode_temperature or 1.0)),
            top_p=float(decode_top_p or 0.0),
        )
        token_arr = sampler(last_logits)
        mx.eval(token_arr)
        return int(token_arr.item())

    def _mlx_activation_to_hidden(
        self,
        activation: list[float],
        packed_bytes: bytes | None = None,
    ) -> Any:
        import mlx.core as mx

        if packed_bytes is not None and len(packed_bytes) >= 8:
            try:
                import openhydra_network  # type: ignore
                rust_tensor = openhydra_network.decode_activation(packed_bytes)
                import torch  # type: ignore
                torch_tensor = torch.from_dlpack(rust_tensor)
                import numpy as np  # type: ignore
                arr = np.asarray(
                    torch_tensor.detach().cpu().contiguous(), dtype=np.float32,
                )
                return mx.array(arr)
            except Exception as exc:
                logger.debug("mlx_dlpack_fallback: %s — using list path", exc)

        if len(activation) < 3:
            raise RuntimeError("invalid_hidden_payload: too short")
        seq_len = int(round(activation[0]))
        hidden_size = int(round(activation[1]))
        if seq_len <= 0 or hidden_size <= 0:
            raise RuntimeError(
                f"invalid_hidden_payload: seq={seq_len} hidden={hidden_size}"
            )
        if hidden_size != int(self._hidden_size):
            raise RuntimeError(
                f"invalid_hidden_payload: hidden_size mismatch "
                f"(wire={hidden_size}, expected={self._hidden_size})"
            )
        payload = activation[2:]
        expected = seq_len * hidden_size
        if len(payload) != expected:
            raise RuntimeError(
                f"invalid_hidden_payload: expected {expected} values, "
                f"got {len(payload)}"
            )
        return mx.array(payload, dtype=mx.float32).reshape(
            1, seq_len, hidden_size
        )

    # ── PyTorch backend ────────────────────────────────────────────────

    def _apply_pytorch(
        self,
        hidden_state: Any,
        *,
        packed_bytes: bytes | None = None,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
    ) -> int:
        import torch

        h = self._pytorch_activation_to_hidden(
            hidden_state, packed_bytes=packed_bytes,
        )
        # ``norm`` / ``lm_head`` / tied ``embed_tokens`` live on
        # ``self._torch_device``; make sure the hidden state matches.
        if self._torch_device is not None:
            h = h.to(self._torch_device)
        if self._torch_dtype is not None and h.dtype != self._torch_dtype:
            h = h.to(dtype=self._torch_dtype)

        with torch.inference_mode():
            normed = self._shard_norm(h)
            if self._tie_word_embeddings:
                # Tied embeddings: logits = normed @ embed_tokens.weight.T
                # HF nn.Embedding doesn't expose an as_linear; use the
                # underlying weight matrix directly.
                weight = self._shard_embed_tokens.weight
                if weight.dtype != normed.dtype:
                    weight = weight.to(dtype=normed.dtype)
                logits = torch.matmul(normed, weight.t())
            else:
                if self._shard_lm_head is None:
                    raise RuntimeError(
                        "standalone_head: untied model but no lm_head loaded"
                    )
                logits = self._shard_lm_head(normed)

        # Sample: mirror PyTorchRuntime._logits_to_next_token_payload logic.
        return self._pytorch_sample(
            logits,
            decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature,
            decode_top_p=decode_top_p,
            decode_top_k=decode_top_k,
            decode_seed=decode_seed,
        )

    def _pytorch_activation_to_hidden(
        self,
        activation: list[float],
        packed_bytes: bytes | None = None,
    ) -> Any:
        import torch

        if packed_bytes is not None and len(packed_bytes) >= 8:
            try:
                import openhydra_network  # type: ignore
                rust_tensor = openhydra_network.decode_activation(packed_bytes)
                decoded = torch.from_dlpack(rust_tensor)
                seq_len = int(rust_tensor.shape[1])
                hidden_size = int(rust_tensor.shape[2])
                if hidden_size != int(self._hidden_size):
                    raise RuntimeError(
                        f"invalid_hidden_payload: hidden_size mismatch "
                        f"(wire={hidden_size}, expected={self._hidden_size})"
                    )
                return decoded.reshape(1, seq_len, hidden_size)
            except Exception as exc:
                logger.debug("pytorch_dlpack_fallback: %s — using list path", exc)

        if len(activation) < 3:
            raise RuntimeError("invalid_hidden_payload: too short")
        seq_len = int(round(activation[0]))
        hidden_size = int(round(activation[1]))
        if seq_len <= 0 or hidden_size <= 0:
            raise RuntimeError(
                f"invalid_hidden_payload: seq={seq_len} hidden={hidden_size}"
            )
        if hidden_size != int(self._hidden_size):
            raise RuntimeError(
                f"invalid_hidden_payload: hidden_size mismatch "
                f"(wire={hidden_size}, expected={self._hidden_size})"
            )
        payload = activation[2:]
        expected = seq_len * hidden_size
        if len(payload) != expected:
            raise RuntimeError(
                f"invalid_hidden_payload: expected {expected} values, "
                f"got {len(payload)}"
            )
        return torch.tensor(payload, dtype=torch.float32).reshape(
            1, seq_len, hidden_size
        )

    @staticmethod
    def _pytorch_sample(
        logits: Any,
        *,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,
        decode_seed: int | None = None,
    ) -> int:
        import torch

        last_logits = logits[:, -1, :]
        _do_sample = bool(decode_do_sample) and bool(decode_temperature)
        if not _do_sample:
            return int(torch.argmax(last_logits, dim=-1).squeeze(0).item())

        work = last_logits / max(1e-5, float(decode_temperature or 1.0))
        _top_k = max(0, int(decode_top_k or 0))
        if _top_k > 0:
            k = min(_top_k, int(work.shape[-1]))
            topk_vals, topk_idx = torch.topk(work, k, dim=-1)
            filtered = torch.full_like(work, float("-inf"))
            filtered.scatter_(1, topk_idx, topk_vals)
            work = filtered
        _top_p = max(0.0, min(1.0, float(decode_top_p or 1.0)))
        if 0.0 < _top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(work, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > _top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            unsorted = torch.full_like(work, float("-inf"))
            unsorted.scatter_(1, sorted_indices, sorted_logits)
            work = unsorted
        probs = torch.softmax(work, dim=-1)
        generator = None
        if decode_seed is not None and int(decode_seed) > 0:
            generator = torch.Generator(device=work.device)
            generator.manual_seed(max(1, int(decode_seed)))
        sampled = torch.multinomial(probs, num_samples=1, generator=generator)
        return int(sampled.squeeze(0).item())

    # ── Diagnostics ─────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def hf_model_id(self) -> str:
        return self._hf_model_id

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def tie_word_embeddings(self) -> bool:
        return self._tie_word_embeddings


# ── Loaders ────────────────────────────────────────────────────────────


def _load_mlx_standalone_head(
    hf_model_id: str, *, free_layers: bool,
) -> StandaloneHead:
    """MLX backend loader — loads via ``mlx_lm.load`` and prunes layers."""
    try:
        import mlx_lm  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "standalone_head[mlx]: mlx_lm is not installed — "
            "install with `pip install mlx-lm`"
        ) from exc

    logger.info("standalone_head_loading[mlx]: model=%s", hf_model_id)
    model, _tokenizer = mlx_lm.load(str(hf_model_id))

    _lm = getattr(model, "language_model", model)
    _text_model = getattr(_lm, "model", _lm)

    norm_module = getattr(_text_model, "norm", None)
    embed_tokens_module = getattr(_text_model, "embed_tokens", None)
    lm_head_module = getattr(_lm, "lm_head", None)
    _lm_args = getattr(_lm, "args", None)
    tie = bool(getattr(_lm_args, "tie_word_embeddings", False))
    hidden_size = int(getattr(_lm_args, "hidden_size", 0))
    vocab_size = int(getattr(_lm_args, "vocab_size", 0))

    if norm_module is None:
        raise RuntimeError(
            "standalone_head[mlx]: no `norm` module — unsupported architecture"
        )
    if embed_tokens_module is None:
        raise RuntimeError(
            "standalone_head[mlx]: no `embed_tokens` — unsupported architecture"
        )
    if not tie and lm_head_module is None:
        raise RuntimeError(
            "standalone_head[mlx]: untied embeddings but no `lm_head`"
        )
    if hidden_size <= 0 or vocab_size <= 0:
        raise RuntimeError(
            f"standalone_head[mlx]: bogus args (hidden={hidden_size}, "
            f"vocab={vocab_size})"
        )

    if free_layers:
        try:
            _layers = list(getattr(_text_model, "layers", []))
            n_freed = 0
            for i in range(len(_layers)):
                _layers[i] = None
                n_freed += 1
            _text_model.layers = _layers
            logger.info(
                "standalone_head[mlx]_layers_pruned: freed=%d", n_freed,
            )
        except Exception as exc:
            logger.warning(
                "standalone_head[mlx]_prune_failed: %s", exc,
            )

    return StandaloneHead(
        backend="mlx",
        hf_model_id=str(hf_model_id),
        norm_module=norm_module,
        embed_tokens_module=embed_tokens_module,
        lm_head_module=(None if tie else lm_head_module),
        tie_word_embeddings=tie,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )


def _load_pytorch_standalone_head(
    hf_model_id: str,
    *,
    free_layers: bool,
    device: str,
    dtype: str,
) -> StandaloneHead:
    """PyTorch backend loader — loads via HF AutoModelForCausalLM, prunes layers.

    Default ``device='cpu'`` + ``dtype='float32'`` suits a cheap Linux
    coordinator studio (no GPU). For an Apple Silicon coordinator
    running PyTorch-MPS, or a GPU-equipped Linux coordinator, override
    via the CLI.
    """
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "standalone_head[pytorch]: install torch + transformers"
        ) from exc

    _dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    _torch_dtype = _dtype_map.get(str(dtype).lower(), torch.float32)

    logger.info(
        "standalone_head_loading[pytorch]: model=%s device=%s dtype=%s",
        hf_model_id, device, dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(hf_model_id),
        torch_dtype=_torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    # Place on the requested device. For CPU this is a no-op on a
    # fresh model; for CUDA/MPS it moves all weights.
    model = model.to(device)

    # Module structure walk — same as MLXRuntime / PyTorchRuntime.
    # Qwen3.5 + most causal LMs: model.model.norm, model.model.embed_tokens,
    # model.lm_head (maybe tied to embed_tokens.weight).
    root = getattr(model, "model", None) or model
    # Nested inner Qwen3.5 TextModel on some architectures.
    inner = getattr(root, "model", None)
    text_model = inner if inner is not None else root

    norm_module = (
        getattr(text_model, "norm", None)
        or getattr(text_model, "final_layer_norm", None)
        or getattr(root, "norm", None)
    )
    embed_tokens_module = (
        getattr(text_model, "embed_tokens", None)
        or getattr(root, "embed_tokens", None)
    )
    lm_head_module = getattr(model, "lm_head", None)

    _config = getattr(model, "config", None)
    tie = bool(getattr(_config, "tie_word_embeddings", False))
    hidden_size = int(getattr(_config, "hidden_size", 0))
    vocab_size = int(getattr(_config, "vocab_size", 0))

    if norm_module is None:
        raise RuntimeError(
            "standalone_head[pytorch]: no `norm` found — unsupported architecture"
        )
    if embed_tokens_module is None:
        raise RuntimeError(
            "standalone_head[pytorch]: no `embed_tokens` — unsupported architecture"
        )
    if not tie and lm_head_module is None:
        raise RuntimeError(
            "standalone_head[pytorch]: untied embeddings but no `lm_head`"
        )
    if hidden_size <= 0 or vocab_size <= 0:
        raise RuntimeError(
            f"standalone_head[pytorch]: bogus config "
            f"(hidden={hidden_size}, vocab={vocab_size})"
        )

    # Prune transformer layers to free memory. The layers are under
    # ``text_model.layers`` (nn.ModuleList). Setting entries to None or
    # replacing them with nn.Identity frees the parameters on the next
    # garbage collection. Wrapped in try/except because some
    # architectures (sparse MoE, hybrid Mamba) may store layers
    # differently.
    if free_layers:
        try:
            layers = getattr(text_model, "layers", None)
            if layers is not None:
                import torch.nn as _nn
                n_freed = 0
                for i in range(len(layers)):
                    layers[i] = _nn.Identity()
                    n_freed += 1
                logger.info(
                    "standalone_head[pytorch]_layers_pruned: freed=%d", n_freed,
                )
                # Drop optional auxiliary modules that add bulk and
                # aren't needed for the head path.
                for _aux in ("rotary_emb",):
                    if hasattr(text_model, _aux):
                        try:
                            setattr(text_model, _aux, None)
                        except Exception:
                            pass
                # Trigger a GC pass to actually release the tensors.
                import gc
                gc.collect()
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning(
                "standalone_head[pytorch]_prune_failed: %s", exc,
            )

    return StandaloneHead(
        backend="pytorch",
        hf_model_id=str(hf_model_id),
        norm_module=norm_module,
        embed_tokens_module=embed_tokens_module,
        lm_head_module=(None if tie else lm_head_module),
        tie_word_embeddings=tie,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        torch_device=str(device),
        torch_dtype=_torch_dtype,
    )


def load_standalone_head(
    hf_model_id: str,
    *,
    backend: str = "auto",
    free_layers: bool = True,
    pytorch_device: str = "cpu",
    pytorch_dtype: str = "float32",
) -> StandaloneHead:
    """Load a StandaloneHead for the given HF model id.

    Args:
        hf_model_id: HF repo id or local path, e.g.
            ``mlx-community/Qwen3.5-2B-MLX-8bit`` for MLX,
            ``Qwen/Qwen3.5-2B`` for PyTorch.
        backend: ``'auto'`` (prefer MLX on Apple Silicon, else PyTorch),
            ``'mlx'``, or ``'pytorch'``.
        free_layers: When True (default), prune the transformer layer
            modules after load to free memory. Keep the head only.
        pytorch_device: PyTorch backend only — device to place the head
            weights on. Default ``'cpu'`` (cheap Linux coord studios).
        pytorch_dtype: PyTorch backend only — weight dtype.
            One of ``'float32'`` / ``'bfloat16'`` / ``'float16'``.
            Default ``'float32'`` (safest for CPU inference).

    Returns:
        A ready ``StandaloneHead`` that ``HeadSampler`` can borrow from.
    """
    resolved = _resolve_backend(backend)
    if resolved == "mlx":
        return _load_mlx_standalone_head(hf_model_id, free_layers=free_layers)
    return _load_pytorch_standalone_head(
        hf_model_id,
        free_layers=free_layers,
        device=pytorch_device,
        dtype=pytorch_dtype,
    )
