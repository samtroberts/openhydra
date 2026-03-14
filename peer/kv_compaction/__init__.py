"""peer.kv_compaction — KV cache compaction via Attention Matching.

Implements the four phases from arXiv:2602.16284 (Zweiger et al., 2026):

  Phase 1 — no-beta compaction:
      Token selection via Highest Attention Keys (HAK) or Orthogonal Matching
      Pursuit (OMP).  Works with any standard HF model.

  Phase 2 — β + Cv fitting:
      Adds scalar log-space bias corrections (β) and refits compact values
      (Cv) via least-squares.  Requires model patching.

  Phase 3 — nonuniform per-head budgets:
      Loads precomputed JSON budgets that assign different target token counts
      to each attention head, reflecting head-level compressibility.

  Phase 4 — online mid-trajectory compaction:
      Compacts the KV cache at every store point when the sequence exceeds a
      configured token limit, enabling unbounded effective context length.

Public API
----------
    from peer.kv_compaction import CompactionConfig, compact_past_key_values

    config = CompactionConfig(enabled=True, method="hak", target_ratio=0.1)
    compacted_pkv = compact_past_key_values(past_key_values, config)
"""

from ._config import CompactionConfig
from ._compactor import compact_past_key_values, _load_head_budgets
from ._query_capture import AttentionQueryCapture
from ._radix_cache import RadixKVCache, _slice_kv_prefix

__all__ = [
    "CompactionConfig",
    "compact_past_key_values",
    "AttentionQueryCapture",
    "_load_head_budgets",
    "RadixKVCache",
    "_slice_kv_prefix",
]
