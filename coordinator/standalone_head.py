# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Standalone coordinator-side final head + sampler (Phase 6).

Path A Phase 5 let a co-located peer's runtime serve as the
``HeadSampler``'s borrow source — but only when the coordinator process
also runs a peer that owns the relevant weights. That works for the
2-node case (Mac = coord = stage 0) but precludes the **true Petals
topology**: client / coordinator on a laptop, peers in a swarm,
coordinator orchestrates + samples but runs no transformer layers
itself.

Phase 6 fills the gap by building a minimal "head-only" MLX module
loaded directly on the coordinator. We borrow ``embed_tokens`` and
``norm`` (and ``lm_head`` for untied models) from a freshly-loaded
``mlx_lm`` model, then *discard* the transformer layer modules so the
coordinator's memory footprint is just ``vocab × hidden + hidden``
(plus tied lm_head for free on Qwen3.5).

Public surface
--------------
* :class:`StandaloneHead` — duck-typed runtime exposing
  ``apply_final_head(hidden_state, **decode_kwargs) -> int`` so it slots
  straight into ``HeadSampler``'s existing borrow protocol.
* :func:`load_standalone_head` — convenience constructor that loads
  the MLX model, prunes the layers, and returns a ready
  ``StandaloneHead``.

Memory accounting (Qwen3.5-2B MLX-8bit, tied embeds)
----------------------------------------------------
* embed_tokens: ``248320 × 2048 × 1 byte`` ≈ 500 MB (8-bit)
* final_norm:    ``2048 × 4 bytes`` ≈ 8 KB
* lm_head:       free (tied with embed_tokens — same tensor reference)
* transformer layers: discarded → ~1.5 GB freed
* **Net coordinator footprint: ~500 MB** (vs ~2 GB if we kept layers)

Design notes
------------
* Reuses the exact same ``apply_final_head`` contract as
  :class:`peer.mlx_runtime.MLXRuntime`. ``HeadSampler.sample`` is
  oblivious to whether it's borrowing from a peer-runtime or from a
  StandaloneHead — both expose the same callable shape.
* The "discard layers" trick mirrors ``MLXRuntime``'s
  ``_all_layers[i] = None`` eviction at init — same MLX module
  surgery, applied unconditionally instead of "non-shard layers only".
* No PyTorch backend in this PR. The Mac coordinator path is the one
  that needs solving today; a PyTorch standalone head is an obvious
  follow-up if/when the coordinator ever runs on a non-Mac host.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StandaloneHead:
    """Coordinator-owned final head module backed by MLX.

    Holds references to ``norm``, ``embed_tokens`` (which doubles as
    ``lm_head`` for tied-embedding models like Qwen3.5), and optionally
    a separate ``lm_head`` matrix for untied models. The transformer
    layers are pruned at construction time to free memory.

    Implements the duck-typed runtime interface that
    ``coordinator.head_sampler.HeadSampler`` expects:

    * ``apply_final_head(hidden_state, *, packed_bytes, decode_*) -> int``
    * ``_has_final_head`` (always ``True`` here)
    * ``_is_last_shard`` (always ``True`` — for legacy gate compatibility)
    """

    def __init__(
        self,
        *,
        hf_model_id: str,
        norm_module: Any,
        embed_tokens_module: Any,
        lm_head_module: Optional[Any],
        tie_word_embeddings: bool,
        hidden_size: int,
        vocab_size: int,
    ) -> None:
        # Make the module references explicit and immutable from outside.
        self._hf_model_id = str(hf_model_id)
        self._shard_norm = norm_module
        self._shard_embed_tokens = embed_tokens_module
        self._shard_lm_head = lm_head_module
        self._tie_word_embeddings = bool(tie_word_embeddings)
        self._hidden_size = int(hidden_size)
        self._vocab_size = int(vocab_size)
        # Legacy duck-type signals — let the existing
        # _maybe_register_head_source heuristic accept us.
        self._has_final_head = True
        self._is_last_shard = True

    # ── Hidden-state decode (mirrors MLXRuntime._activation_to_hidden) ──

    def _activation_to_hidden(
        self,
        activation: list[float],
        packed_bytes: bytes | None = None,
    ) -> Any:
        import mlx.core as mx

        # Zero-copy DLPack path first.
        if packed_bytes is not None and len(packed_bytes) >= 8:
            try:
                import openhydra_network  # type: ignore
                rust_tensor = openhydra_network.decode_activation(packed_bytes)
                import torch  # type: ignore
                torch_tensor = torch.from_dlpack(rust_tensor)
                return _torch_to_mx(
                    torch_tensor.reshape(
                        1, rust_tensor.shape[1], rust_tensor.shape[2],
                    ),
                )
            except Exception as exc:  # pragma: no cover — exercised live
                logger.debug("dlpack_fallback: %s — using list path", exc)

        # Fallback: ``[seq_len_f, hidden_size_f, v0, v1, ...]`` list.
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

    # ── Sampling (matches MLXRuntime._sample_from_logits semantics) ────

    @staticmethod
    def _sample_from_logits(
        logits: Any,
        *,
        decode_do_sample: bool | None = None,
        decode_temperature: float | None = None,
        decode_top_p: float | None = None,
        decode_top_k: int | None = None,  # noqa: ARG004 — for parity
        decode_seed: int | None = None,   # noqa: ARG004 — make_sampler doesn't expose seed
    ) -> int:
        import mlx.core as mx
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
        — accepts either a list-format hidden state or zero-copy
        ``packed_bytes`` from the Rust encoder, returns a single sampled
        token id.
        """
        import mlx.core as mx

        h = self._activation_to_hidden(hidden_state, packed_bytes=packed_bytes)
        mx.eval(h)
        h = self._shard_norm(h)
        if self._tie_word_embeddings:
            logits = self._shard_embed_tokens.as_linear(h)
        else:
            if self._shard_lm_head is None:
                raise RuntimeError(
                    "standalone_head: untied model but no lm_head loaded — "
                    "construction error"
                )
            logits = self._shard_lm_head(h)
        mx.eval(logits)
        return self._sample_from_logits(
            logits,
            decode_do_sample=decode_do_sample,
            decode_temperature=decode_temperature,
            decode_top_p=decode_top_p,
            decode_top_k=decode_top_k,
            decode_seed=decode_seed,
        )

    # ── Diagnostics ─────────────────────────────────────────────────────

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


# ── Loader ──────────────────────────────────────────────────────────────


def _torch_to_mx(t: Any) -> Any:
    """Bridge a PyTorch tensor to MLX via numpy (DLPack-free fallback).

    MLX doesn't expose ``from_dlpack``; numpy is the safe interchange.
    """
    import mlx.core as mx
    import numpy as np  # type: ignore
    arr = np.asarray(t.detach().cpu().contiguous(), dtype=np.float32)
    return mx.array(arr)


def load_standalone_head(
    hf_model_id: str,
    *,
    free_layers: bool = True,
) -> StandaloneHead:
    """Load an MLX model and extract a head-only ``StandaloneHead``.

    Args:
        hf_model_id: MLX-quantised HF repo id, e.g.
            ``mlx-community/Qwen3.5-2B-MLX-8bit``. Same string as the
            ``--runtime-model-id`` flag the local peer would use; the
            coordinator process loads its own copy of the head weights
            independent of any peer.
        free_layers: When ``True`` (default), set every transformer
            layer to ``None`` after init so MLX can release the
            compute-heavy weights. Disable only for diagnostics —
            keeping the layers loaded would defeat the whole point of
            running the coordinator as "head-only".

    Returns:
        Ready-to-use ``StandaloneHead``.

    Raises:
        RuntimeError: if MLX or mlx_lm are not installed, or the model
            structure is unfamiliar (the loader walks the same
            ``model.language_model.model`` path as MLXRuntime).
    """
    try:
        import mlx_lm  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "standalone_head: mlx_lm is not installed — Phase 6 requires "
            "the MLX backend. Install with `pip install mlx-lm`."
        ) from exc

    logger.info("standalone_head_loading: model=%s", hf_model_id)
    model, _tokenizer = mlx_lm.load(str(hf_model_id))

    # Walk the same module structure MLXRuntime uses.
    _lm = getattr(model, "language_model", model)
    _text_model = getattr(_lm, "model", _lm)

    norm_module = getattr(_text_model, "norm", None)
    embed_tokens_module = getattr(_text_model, "embed_tokens", None)
    lm_head_module = getattr(_lm, "lm_head", None)
    _lm_args = getattr(_lm, "args", None)
    tie = bool(getattr(_lm_args, "tie_word_embeddings", False))
    hidden_size = int(getattr(_lm_args, "hidden_size", 0))
    vocab_size = int(getattr(_lm_args, "vocab_size", 0))

    # Sanity checks before we discard the layers.
    if norm_module is None:
        raise RuntimeError(
            "standalone_head: model has no `norm` module under "
            "language_model.model — unsupported architecture"
        )
    if embed_tokens_module is None:
        raise RuntimeError(
            "standalone_head: model has no `embed_tokens` module under "
            "language_model.model — unsupported architecture"
        )
    if not tie and lm_head_module is None:
        raise RuntimeError(
            "standalone_head: untied embeddings but no `lm_head` module — "
            "unsupported architecture"
        )
    if hidden_size <= 0 or vocab_size <= 0:
        raise RuntimeError(
            f"standalone_head: bogus model args (hidden={hidden_size}, "
            f"vocab={vocab_size}) — couldn't read from language_model.args"
        )

    # Discard transformer layer modules. Same eviction pattern
    # MLXRuntime uses — set the entry to ``None`` so MLX can drop the
    # weight references on the next garbage-collection pass. The
    # ``layers`` container is usually a list-like; in-place mutation
    # is what the codebase already does.
    if free_layers:
        try:
            _layers = list(getattr(_text_model, "layers", []))
            n_freed = 0
            for i in range(len(_layers)):
                _layers[i] = None
                n_freed += 1
            # Re-bind so MLX sees the pruned list.
            _text_model.layers = _layers
            logger.info(
                "standalone_head_layers_pruned: freed=%d "
                "(only norm + embed_tokens%s retained)",
                n_freed, "" if tie else " + lm_head",
            )
        except Exception as exc:
            # Non-fatal — head still works, just uses more memory than
            # strictly necessary.
            logger.warning(
                "standalone_head_prune_failed: %s — keeping layers loaded "
                "(coordinator memory will be ~2 GB rather than ~500 MB)",
                exc,
            )

    head = StandaloneHead(
        hf_model_id=str(hf_model_id),
        norm_module=norm_module,
        embed_tokens_module=embed_tokens_module,
        lm_head_module=(None if tie else lm_head_module),
        tie_word_embeddings=tie,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
    )
    logger.info(
        "standalone_head_ready: model=%s hidden=%d vocab=%d tie=%s",
        hf_model_id, hidden_size, vocab_size, tie,
    )
    return head
