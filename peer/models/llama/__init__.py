"""Llama model-family constants for KV cache compaction.

GQA configuration (Llama-3.1-8B-Instruct):
    n_query_heads  = 32
    n_kv_heads     = 8
    n_groups (GQA) = 4

The β expansion in _beta_inject.py repeats each kv-head β tensor 4 times to
align with the 32 query heads (``repeat_interleave(n_groups, dim=0)``).
"""

LLAMA3_8B_N_LAYERS: int = 32
LLAMA3_8B_N_KV_HEADS: int = 8
LLAMA3_8B_N_QUERY_HEADS: int = 32
LLAMA3_8B_HEAD_DIM: int = 128
