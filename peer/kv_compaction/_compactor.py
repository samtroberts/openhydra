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

"""compact_past_key_values — the main KV cache compaction entry point.

Implements all four phases end-to-end:

  Phase 1 (no-beta):
      Select t compact key/value pairs per head using HAK or OMP.
      Returns a standard HF DynamicCache (or tuple) that works with any model.

  Phase 2 (beta):
      Fit scalar log-space bias corrections β and re-fitted compact values Cv.
      Returns a CompactedKVCache that requires the model to be patched.

  Phase 3 (nonuniform budgets):
      Loads per-layer / per-kv-head token budgets from a JSON file.
      Overrides ``config.target_ratio`` with per-head targets.

  Phase 4 (online):
      Called from ``compact_past_key_values`` whenever the stored sequence
      length exceeds ``config.online_max_tokens``.  Compacts back to that
      limit at every store point, enabling unbounded effective context length.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from ._config import CompactionConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DynamicCache extraction / reconstruction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_kv_tensors(
    past_key_values: Any,
) -> "tuple[list[Tensor], list[Tensor], str]":
    """Return (key_tensors, value_tensors, cache_type).

    cache_type is one of ``"dynamic"``, ``"tuple"``, or ``"unknown"``.
    """
    if past_key_values is None:
        return [], [], "none"

    # CompactedKVCache — unwrap first
    if hasattr(past_key_values, "_cache"):
        past_key_values = past_key_values._cache

    # HF DynamicCache (transformers >= 4.35)
    if (
        hasattr(past_key_values, "key_cache")
        and hasattr(past_key_values, "value_cache")
        and isinstance(past_key_values.key_cache, list)
    ):
        return past_key_values.key_cache, past_key_values.value_cache, "dynamic"

    # Legacy tuple-of-(K, V) tuples
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        first = past_key_values[0]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            key_tensors = [layer[0] for layer in past_key_values]
            value_tensors = [layer[1] for layer in past_key_values]
            return key_tensors, value_tensors, "tuple"

    return [], [], "unknown"


def _pack_kv_tensors(
    key_tensors: "list[Tensor]",
    value_tensors: "list[Tensor]",
    cache_type: str,
    original: Any,
) -> Any:
    """Pack compacted tensors back into the original cache format."""
    if cache_type == "dynamic":
        try:
            from transformers import DynamicCache
            new_cache = DynamicCache()
            new_cache.key_cache = key_tensors
            new_cache.value_cache = value_tensors
            if key_tensors:
                new_cache._seen_tokens = key_tensors[0].shape[-2]
            return new_cache
        except Exception as exc:
            logger.debug("DynamicCache_pack_failed: %s — falling back to tuple", exc)
            cache_type = "tuple"

    if cache_type == "tuple":
        return tuple(zip(key_tensors, value_tensors))

    return original  # unknown format — return unchanged


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: head budget loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_head_budgets(path: str) -> "dict[str, Any] | None":
    """Load a head-budget JSON file.  Returns None on any error."""
    try:
        data = json.loads(Path(path).read_text())
        return data
    except Exception as exc:
        logger.warning("head_budget_load_failed path=%s err=%s", path, exc)
        return None


def _get_budget_for_layer_head(
    budgets_data: "dict[str, Any] | None",
    layer_idx: int,
    head_idx: int,
    seq_len: int,
    default_ratio: float,
) -> int:
    """Return the target token count for (layer_idx, head_idx).

    The JSON format is::

        {
          "model": "...",
          "layer_budgets": [
            [0.05, 0.08, ...],   // layer 0 — one ratio per kv-head
            ...
          ]
        }

    Values are fractions (0–1) of the *current* sequence length.
    """
    t_default = max(4, int(seq_len * default_ratio))
    if budgets_data is None:
        return t_default

    layer_budgets = budgets_data.get("layer_budgets")
    if not layer_budgets or layer_idx >= len(layer_budgets):
        return t_default

    head_budgets = layer_budgets[layer_idx]
    if not head_budgets or head_idx >= len(head_budgets):
        return t_default

    ratio = float(head_budgets[head_idx])
    return max(4, int(seq_len * ratio))


# ─────────────────────────────────────────────────────────────────────────────
# Per-head compaction
# ─────────────────────────────────────────────────────────────────────────────

def _compact_single_head(
    K: "Tensor",   # (T, d_head)
    V: "Tensor",   # (T, d_head)
    t: int,
    method: str,
    n_ref: int,
    fit_beta: bool,
    Q_ref_actual: "Tensor | None" = None,  # Option A: real Q from W_q(hidden)
) -> "tuple[Tensor, Tensor, Tensor | None, Tensor]":
    """Compact one (K, V) slice to *t* tokens.

    Option A:  when ``Q_ref_actual`` is provided (shape (R, d_head)) it is used
    directly as reference queries.  These come from ``AttentionQueryCapture``
    and are computed as W_q(hidden) — the correct query subspace, without RoPE.

    Fallback:  when ``Q_ref_actual`` is None, the last *n_ref* key vectors are
    used as proxy queries.  This is worse (wrong subspace) but requires no
    external inputs.

    Returns:
        (Ck, Cv, beta, indices) where beta is None when fit_beta=False.
    """
    from ._algorithms import select_hak, select_omp, fit_beta_and_cv

    T = K.shape[0]
    t = min(t, T)

    # ── Reference queries ────────────────────────────────────────────────────
    if Q_ref_actual is not None and Q_ref_actual.shape[0] > 0:
        # Option A: real W_q-projected queries (correct subspace).
        Q_ref = Q_ref_actual.to(K.device)
    else:
        # Proxy fallback: last n_ref key vectors (Phase 1-3 legacy path).
        Q_ref = K[-min(n_ref, T):]              # (R, d_head)

    if method == "omp":
        indices = select_omp(K, Q_ref, t)
    else:
        indices = select_hak(K, Q_ref, t)

    Ck = K[indices]     # (t, d_head)
    Cv = V[indices]     # (t, d_head) — default; overwritten in Phase 2

    beta = None
    if fit_beta and t < T:
        try:
            beta, Cv = fit_beta_and_cv(K, V, Ck, Q_ref, indices)
        except Exception as exc:
            logger.debug("fit_beta_failed layer/head: %s", exc)

    return Ck, Cv, beta, indices


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compact_past_key_values(
    past_key_values: Any,
    config: "CompactionConfig",
    budgets_data: "dict[str, Any] | None" = None,
    q_ref_per_layer: "list | None" = None,
) -> Any:
    """Compact *past_key_values* according to *config*.

    Phase 1 (config.beta_enabled=False):
        Returns a standard DynamicCache / tuple with compacted (Ck, Cv).
        Compatible with any unmodified HF model.

    Phase 2 (config.beta_enabled=True):
        Returns a CompactedKVCache with compacted (Ck, Cv) AND β per layer.
        Requires the model to be patched via _beta_inject.

    Phase 3 (budgets_data not None):
        Uses per-layer / per-head token budgets from a preloaded JSON.

    Phase 4 (config.online_enabled=True):
        Compacts when seq_len > online_max_tokens; this function is called at
        every cache-write point, making it suitable for online compaction.

    Option A (q_ref_per_layer not None):
        Uses real W_q-projected query vectors (captured by AttentionQueryCapture)
        as reference queries instead of the proxy-key fallback.
        q_ref_per_layer is a list of length n_layers; each element is either
        Tensor(n_kv_heads, n_ref, d_head) or None (falls back to proxy for
        that layer).  Improves key selection and β fitting quality.

    Args:
        past_key_values:  HF DynamicCache, tuple-of-tuples, or CompactedKVCache.
        config:           CompactionConfig instance.
        budgets_data:     Pre-loaded JSON dict from _load_head_budgets, or None.
        q_ref_per_layer:  Optional list of per-layer Q tensors from
                          AttentionQueryCapture.compute_q_ref().

    Returns:
        Compacted cache in the same format as the input (or CompactedKVCache for
        Phase 2).
    """
    import torch

    key_tensors, value_tensors, cache_type = _extract_kv_tensors(past_key_values)

    if cache_type in ("none", "unknown") or not key_tensors:
        return past_key_values

    n_layers = len(key_tensors)
    # K shape: (batch, n_kv_heads, T, d_head)
    K0 = key_tensors[0]
    seq_len = K0.shape[-2]

    # ── Guard: skip when not worth compacting ────────────────────────────────
    if seq_len < config.min_source_tokens:
        return past_key_values

    # Auto mode (6.1): skip compaction when sequence is short
    if getattr(config, "mode", "on") == "auto" and seq_len <= getattr(config, "auto_threshold", 512):
        return past_key_values

    # Phase 4: online threshold check
    if config.online_enabled:
        if seq_len <= config.online_max_tokens:
            return past_key_values
        # Target is online_max_tokens
        target_override: int | None = config.online_max_tokens
    else:
        target_override = None

    n_kv_heads = K0.shape[1]
    batch_size = K0.shape[0]
    d_head = K0.shape[-1]

    new_key_tensors: list["Tensor"] = []
    new_value_tensors: list["Tensor"] = []
    beta_per_layer: list["Tensor | None"] = []

    for layer_idx in range(n_layers):
        K_layer = key_tensors[layer_idx]    # (B, n_kv, T, d)
        V_layer = value_tensors[layer_idx]  # (B, n_kv, T, d)

        compacted_keys: list["Tensor"] = []
        compacted_vals: list["Tensor"] = []
        layer_betas: list["Tensor | None"] = []

        # Option A: extract Q_ref for this layer (n_kv_heads, n_ref, d_head)
        q_layer: "Any" = None
        if q_ref_per_layer is not None and layer_idx < len(q_ref_per_layer):
            q_layer = q_ref_per_layer[layer_idx]

        for head_idx in range(n_kv_heads):
            K_h = K_layer[0, head_idx]      # (T, d_head)
            V_h = V_layer[0, head_idx]      # (T, d_head)

            if target_override is not None:
                t = max(config.min_kept_tokens, target_override)
            else:
                t = _get_budget_for_layer_head(
                    budgets_data, layer_idx, head_idx, seq_len, config.target_ratio
                )
            t = max(config.min_kept_tokens, min(t, seq_len))

            # Option A: per-kv-head Q_ref slice — (n_ref, d_head)
            Q_ref_actual = None
            if q_layer is not None and head_idx < q_layer.shape[0]:
                Q_ref_actual = q_layer[head_idx]  # (n_ref, d_head)

            Ck, Cv, beta, _ = _compact_single_head(
                K_h, V_h, t, config.method, config.n_ref_queries,
                config.beta_enabled, Q_ref_actual=Q_ref_actual,
            )

            compacted_keys.append(Ck)   # (t, d_head) — may vary per head for Phase 3
            compacted_vals.append(Cv)
            layer_betas.append(beta)

        # All heads may have different t in Phase 3 — pad to max t
        max_t = max(k.shape[0] for k in compacted_keys)

        new_K = torch.zeros(batch_size, n_kv_heads, max_t, d_head,
                            dtype=K_layer.dtype, device=K_layer.device)
        new_V = torch.zeros(batch_size, n_kv_heads, max_t, d_head,
                            dtype=V_layer.dtype, device=V_layer.device)

        for h, (Ck, Cv) in enumerate(zip(compacted_keys, compacted_vals)):
            t_h = Ck.shape[0]
            new_K[0, h, :t_h] = Ck
            new_V[0, h, :t_h] = Cv

        new_key_tensors.append(new_K)
        new_value_tensors.append(new_V)

        # Per-layer β: stack into (n_kv_heads, max_t) tensor or None
        if config.beta_enabled and any(b is not None for b in layer_betas):
            import torch as _torch
            beta_tensors = []
            for b in layer_betas:
                if b is None:
                    beta_tensors.append(_torch.zeros(max_t, device=K_layer.device))
                else:
                    # Pad shorter β vectors
                    if b.shape[0] < max_t:
                        pad = _torch.zeros(max_t - b.shape[0], device=b.device)
                        b = _torch.cat([b, pad])
                    beta_tensors.append(b[:max_t])
            beta_per_layer.append(_torch.stack(beta_tensors, dim=0))  # (n_kv, max_t)
        else:
            beta_per_layer.append(None)

    # ── Pack result ───────────────────────────────────────────────────────────
    t_final = new_key_tensors[0].shape[-2] if new_key_tensors else seq_len
    standard_cache = _pack_kv_tensors(new_key_tensors, new_value_tensors, cache_type, past_key_values)

    if config.beta_enabled and any(b is not None for b in beta_per_layer):
        from ._cache import CompactedKVCache
        return CompactedKVCache(
            cache=standard_cache,
            beta_per_layer=beta_per_layer,
            prefix_length=t_final,
        )

    return standard_cache
