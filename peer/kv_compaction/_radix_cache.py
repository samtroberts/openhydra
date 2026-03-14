"""
peer.kv_compaction._radix_cache — flat-dict radix (longest-prefix) KV cache.

RadixKVCache stores ``(token_sequence → kv_cache)`` pairs and supports
O(n_entries) longest-prefix lookup.  For ≤256 cached sequences this is
fast enough in CPython; a full trie can replace it later without changing
the public API.

Typical use
-----------
    from peer.kv_compaction import RadixKVCache, _slice_kv_prefix

    cache = RadixKVCache(max_entries=128, min_prefix_len=16)
    # After a full forward pass:
    cache.insert(full_token_tuple, past_key_values)
    # At the start of the next forward pass:
    prefix_kv, prefix_len = cache.lookup(new_full_token_tuple)
    if prefix_len > 0:
        kv_prefix = _slice_kv_prefix(prefix_kv, prefix_len)
        # Feed kv_prefix as past_key_values and trim input_ids accordingly
"""
from __future__ import annotations

import threading
import time
from typing import Any


class RadixKVCache:
    """Flat-dict longest-prefix KV cache with LRU eviction.

    Parameters
    ----------
    max_entries:
        Maximum number of token sequences to store simultaneously.
        When full, the least-recently-accessed entry is evicted.
    min_prefix_len:
        Sequences shorter than this are neither stored nor matched.
        Guards against uselessly short cache entries.
    """

    def __init__(self, max_entries: int = 128, min_prefix_len: int = 16) -> None:
        self._max_entries = max(1, int(max_entries))
        self._min_prefix_len = max(1, int(min_prefix_len))
        # Maps token tuple → (kv_cache, last_access_monotonic)
        self._store: dict[tuple[int, ...], tuple[Any, float]] = {}
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def insert(self, tokens: tuple[int, ...], kv_cache: Any) -> None:
        """Store ``kv_cache`` under the key ``tokens``.

        Silently ignores sequences shorter than ``min_prefix_len``.
        Evicts the LRU entry when the cache is full.
        """
        if len(tokens) < self._min_prefix_len:
            return
        with self._lock:
            self._store[tokens] = (kv_cache, time.monotonic())
            # LRU eviction
            while len(self._store) > self._max_entries:
                oldest = min(self._store, key=lambda k: self._store[k][1])
                del self._store[oldest]

    def lookup(self, tokens: tuple[int, ...]) -> tuple[Any, int]:
        """Return the kv_cache and length of the longest matching prefix.

        Iterates all stored entries and finds the one whose token key is both:
          - a prefix of ``tokens`` (i.e. ``tokens[:slen] == stored_tokens``), and
          - the longest such prefix found.

        The matched entry's last-access time is updated (LRU bump).

        Returns
        -------
        (kv_cache, prefix_len)
            ``prefix_len == 0`` and ``kv_cache is None`` on a complete miss.
        """
        best_kv: Any = None
        best_len: int = 0
        now = time.monotonic()
        query_len = len(tokens)

        with self._lock:
            for stored_tokens, (kv, _ts) in list(self._store.items()):
                slen = len(stored_tokens)
                # Must beat current best, must fit inside query, must be long enough
                if slen <= best_len or slen > query_len or slen < self._min_prefix_len:
                    continue
                if tokens[:slen] == stored_tokens:
                    best_kv = kv
                    best_len = slen
                    # LRU bump: update access time in-place
                    self._store[stored_tokens] = (kv, now)

        return best_kv, best_len

    def stats(self) -> dict[str, int]:
        """Return a snapshot of current cache occupancy."""
        with self._lock:
            return {
                "radix_entries": len(self._store),
                "radix_max_entries": self._max_entries,
            }

    def clear(self) -> None:
        """Evict all cached entries."""
        with self._lock:
            self._store.clear()


# ─── KV cache slicing helper ──────────────────────────────────────────────────

def _slice_kv_prefix(past_key_values: Any, prefix_len: int) -> Any:
    """Return a copy of ``past_key_values`` truncated to ``prefix_len`` tokens.

    Supports the two KV cache formats used by ``transformers``:

    * **DynamicCache** — object with ``.key_cache`` / ``.value_cache`` lists of
      tensors shaped ``(batch, n_heads, seq, head_dim)``.
    * **Tuple-of-tuples** — ``past_key_values[layer] = (K, V)`` where ``K`` and
      ``V`` are shaped ``(batch, n_heads, seq, head_dim)``.

    Returns ``None`` if the format is unrecognised or on any error.

    The returned object is a *copy* (all tensors are cloned) so that the
    original cached value is not mutated.
    """
    if past_key_values is None or prefix_len <= 0:
        return None

    # ── DynamicCache (or any duck-typed equivalent with key_cache/value_cache) ─
    key_cache = getattr(past_key_values, "key_cache", None)
    val_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and val_cache is not None:
        try:
            # Build a lightweight output that mirrors the DynamicCache interface.
            # We deliberately avoid importing transformers.DynamicCache so that
            # _slice_kv_prefix works even when transformers is not installed.
            class _SlicedCache:
                def __init__(self) -> None:
                    self.key_cache: list = []
                    self.value_cache: list = []

                def get_seq_length(self, layer_idx: int = 0) -> int:
                    return self.key_cache[0].shape[-2] if self.key_cache else 0

            sliced = _SlicedCache()
            for k, v in zip(key_cache, val_cache):
                sliced.key_cache.append(k[:, :, :prefix_len, :].clone())
                sliced.value_cache.append(v[:, :, :prefix_len, :].clone())
            return sliced
        except Exception:
            return None

    # ── Tuple-of-tuples ───────────────────────────────────────────────────────
    if isinstance(past_key_values, (tuple, list)):
        try:
            sliced_tuple = tuple(
                (layer[0][:, :, :prefix_len, :].clone(),
                 layer[1][:, :, :prefix_len, :].clone())
                for layer in past_key_values
            )
            return sliced_tuple
        except Exception:
            return None

    return None
