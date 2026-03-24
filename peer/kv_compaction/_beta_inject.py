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

"""Phase 2: monkey-patch loaded HF models to support CompactedKVCache β injection.

When ``beta_enabled=True`` the attention layers need to add per-token scalar
biases (β) to the attention logits before softmax.  Rather than replacing full
model files this module wraps the ``forward`` method of each attention layer at
runtime.

Supported model families detected by transformers class name:
  • Qwen2 / Qwen3  — Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention
  • LLaMA          — LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention
  • Gemma 3        — Gemma3Attention (best-effort)

The patch is **idempotent**: calling it twice on the same model is safe.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_openhydra_beta_patched"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def patch_model_for_beta_injection(model: Any, model_family: str) -> bool:
    """Patch *model* so that CompactedKVCache β biases are honoured.

    Args:
        model:        A loaded HuggingFace CausalLM model.
        model_family: One of ``"qwen2"``, ``"qwen3"``, ``"llama"``, ``"gemma3"``.
                      ``"qwen3"`` is treated identically to ``"qwen2"``.

    Returns:
        True if at least one attention layer was patched, False otherwise.
    """
    if getattr(model, _PATCHED_ATTR, False):
        logger.debug("model already patched for beta injection")
        return True

    family = str(model_family or "").strip().lower()
    if family in {"qwen2", "qwen3"}:
        patched = _patch_layers(model, _QWEN2_ATTN_CLASSES, _wrap_attention_forward)
    elif family == "llama":
        patched = _patch_layers(model, _LLAMA_ATTN_CLASSES, _wrap_attention_forward)
    elif family == "gemma3":
        patched = _patch_layers(model, _GEMMA3_ATTN_CLASSES, _wrap_attention_forward)
    else:
        logger.warning("beta_inject: unknown model family %r — skipping patch", family)
        return False

    if patched:
        setattr(model, _PATCHED_ATTR, True)
        logger.info("beta_inject: patched %d attention layers for family=%r", patched, family)
    else:
        logger.warning("beta_inject: no attention layers found for family=%r", family)
    return patched > 0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Lazy class-name sets — we do string matching so we don't need to import the
# transformers model modules at module load time.
_QWEN2_ATTN_CLASSES: set[str] = {
    "Qwen2Attention",
    "Qwen2FlashAttention2",
    "Qwen2SdpaAttention",
}
_LLAMA_ATTN_CLASSES: set[str] = {
    "LlamaAttention",
    "LlamaFlashAttention2",
    "LlamaSdpaAttention",
}
_GEMMA3_ATTN_CLASSES: set[str] = {
    "Gemma3Attention",
    "Gemma3SdpaAttention",
}


def _patch_layers(
    model: Any,
    class_names: set[str],
    wrap_fn: Any,
) -> int:
    """Walk model.model.layers, patch matching attention layers.  Returns count."""
    decoder = getattr(model, "model", None)
    if decoder is None:
        return 0
    layers = getattr(decoder, "layers", None)
    if not layers:
        return 0

    count = 0
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        if type(attn).__name__ in class_names:
            if not getattr(attn, _PATCHED_ATTR, False):
                wrap_fn(attn, layer_idx)
                setattr(attn, _PATCHED_ATTR, True)
                count += 1
    return count


def _wrap_attention_forward(attn_module: Any, layer_idx: int) -> None:
    """Replace ``attn_module.forward`` with a β-aware wrapper."""
    from peer.kv_compaction._cache import CompactedKVCache

    original_forward = attn_module.forward
    n_heads_attr = "num_heads"
    n_kv_attr = "num_key_value_heads"

    @functools.wraps(original_forward)
    def patched_forward(
        hidden_states: "Tensor",
        attention_mask: "Any | None" = None,
        position_ids: "Any | None" = None,
        past_key_value: "Any | None" = None,
        **kwargs: Any,
    ) -> Any:
        # ── Detect CompactedKVCache ───────────────────────────────────────────
        beta: "Tensor | None" = None
        prefix_len: int = 0

        if isinstance(past_key_value, CompactedKVCache):
            beta = past_key_value.beta_for_layer(layer_idx)
            prefix_len = past_key_value.prefix_length
            # Expose the underlying standard cache to the model internals
            past_key_value = past_key_value.to_standard_cache()

        # ── Inject β into attention_mask ─────────────────────────────────────
        if beta is not None and attention_mask is not None:
            try:
                n_kv_heads: int = getattr(attn_module, n_kv_attr, 1)
                n_query_heads: int = getattr(attn_module, n_heads_attr, n_kv_heads)
                n_groups = max(1, n_query_heads // max(1, n_kv_heads))

                # beta: (n_kv_heads, t) → (n_query_heads, t)
                beta_q = beta.repeat_interleave(n_groups, dim=0)   # (n_q, t)
                # → (1, n_query_heads, 1, t) broadcastable with attn_weights
                beta_4d = beta_q.unsqueeze(0).unsqueeze(2)

                t = min(prefix_len, attention_mask.shape[-1])
                mod_mask = attention_mask.clone()
                mod_mask[:, :, :, :t] = mod_mask[:, :, :, :t] + beta_4d[:, :, :, :t]
                attention_mask = mod_mask
            except Exception as exc:
                logger.debug("beta_inject_layer_%d_failed: %s", layer_idx, exc)

        return original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

    attn_module.forward = patched_forward


# ─────────────────────────────────────────────────────────────────────────────
# Model-family detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_model_family(model_name: str) -> str:
    """Return the model family string for a given HuggingFace model ID."""
    name = str(model_name or "").lower()
    if "qwen3" in name or "qwen2" in name or "qwen" in name:
        return "qwen2"
    if "llama" in name or "meta-llama" in name:
        return "llama"
    if "gemma" in name:
        return "gemma3"
    return "unknown"
