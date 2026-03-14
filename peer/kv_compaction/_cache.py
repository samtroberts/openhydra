"""CompactedKVCache — HF Cache wrapper that carries per-layer β biases.

Used in Phase 2 only.  The standard DynamicCache stores (Ck, Cv) while this
wrapper adds the β tensor list.  Patched attention layers read β via the
``beta_for_layer`` method and add it to the attention logits before softmax.

For Phase 1 (beta_enabled=False) the compacted cache is a plain DynamicCache
(or tuple) and this class is never instantiated.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


class CompactedKVCache:
    """Wraps a standard HF DynamicCache and attaches per-layer β tensors.

    Attributes:
        _cache:         The underlying DynamicCache (or tuple fallback).
        _beta:          List of (n_kv_heads, t) tensors; None entries mean
                        "no correction for this layer".
        rope_base:      RoPE offset = original_seq_len − max_compact_prefix.
                        Set to 0 if not applicable.
        prefix_length:  Physical token count of the compact prefix.
    """

    def __init__(
        self,
        cache: Any,
        beta_per_layer: list["Tensor | None"],
        prefix_length: int,
        rope_base: int = 0,
    ) -> None:
        self._cache = cache
        self._beta: list["Tensor | None"] = beta_per_layer
        self.prefix_length = int(prefix_length)
        self.rope_base = int(rope_base)

    # ── HF Cache protocol (delegate everything to _cache) ────────────────────

    def get_seq_length(self, layer_idx: int = 0) -> int:
        fn = getattr(self._cache, "get_seq_length", None)
        if callable(fn):
            return int(fn(layer_idx))
        # Legacy tuple format fallback
        try:
            return int(self._cache[0][0].shape[-2])
        except Exception:
            return self.prefix_length

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        fn = getattr(self._cache, "get_usable_length", None)
        if callable(fn):
            return int(fn(new_seq_length, layer_idx))
        return self.get_seq_length(layer_idx)

    @property
    def key_cache(self) -> list["Tensor"]:
        return self._cache.key_cache  # type: ignore[return-value]

    @property
    def value_cache(self) -> list["Tensor"]:
        return self._cache.value_cache  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:
        # Transparent delegation for any other Cache method
        return getattr(self._cache, name)

    # ── β access ─────────────────────────────────────────────────────────────

    def beta_for_layer(self, layer_idx: int) -> "Tensor | None":
        """Return the β tensor for *layer_idx*, or None if not available."""
        if layer_idx < len(self._beta):
            return self._beta[layer_idx]
        return None

    def has_beta(self) -> bool:
        """True if any layer has a non-None β tensor."""
        return any(b is not None for b in self._beta)

    # ── Conversion helpers ────────────────────────────────────────────────────

    def to_standard_cache(self) -> Any:
        """Return the underlying DynamicCache without β (for use in unpatched models)."""
        return self._cache

    def __repr__(self) -> str:  # pragma: no cover
        n_beta = sum(1 for b in self._beta if b is not None)
        return (
            f"CompactedKVCache(prefix_length={self.prefix_length}, "
            f"layers={len(self._beta)}, beta_layers={n_beta})"
        )
