"""Qwen3 / Qwen2 model-family constants for KV cache compaction.

GQA configuration (Qwen3-4B):
    n_query_heads  = 32
    n_kv_heads     = 8
    n_groups (GQA) = 4

The β expansion in _beta_inject.py repeats each kv-head β tensor 4 times to
align with the 32 query heads (``repeat_interleave(n_groups, dim=0)``).
"""

QWEN3_4B_N_LAYERS: int = 36
QWEN3_4B_N_KV_HEADS: int = 8
QWEN3_4B_N_QUERY_HEADS: int = 32
QWEN3_4B_HEAD_DIM: int = 128
