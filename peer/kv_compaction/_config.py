"""CompactionConfig — configuration for KV cache compaction."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompactionConfig:
    """Configuration for KV cache compaction via Attention Matching.

    Phase 1 — no-beta compaction:
        enabled=True, beta_enabled=False
        Selects the top-``target_ratio`` tokens per head using HAK or OMP key
        selection.  The compacted past_key_values is a standard HF cache that
        works with any unmodified model.

    Phase 2 — beta injection:
        enabled=True, beta_enabled=True
        Adds scalar log-space bias corrections (β) that preserve attention mass
        across the compacted prefix, plus refits compact values (Cv) via least
        squares.  Requires the model to be patched via
        ``peer.kv_compaction._beta_inject.patch_model_for_beta_injection``.

    Phase 3 — nonuniform head budgets:
        head_budget_path=<json file>
        Loads per-layer / per-kv-head target token counts from a precomputed
        JSON file instead of applying a uniform ``target_ratio`` everywhere.

    Phase 4 — online mid-trajectory compaction:
        online_enabled=True, online_max_tokens=<N>
        After every forward that would store a KV cache entry, if the cache
        exceeds ``online_max_tokens``, compact it back to that limit.  This
        caps physical KV memory to a fixed budget while allowing unlimited
        effective context length.
    """

    # ── Core controls ────────────────────────────────────────────────────────
    enabled: bool = False

    # Key-selection algorithm: "hak" (Highest Attention Keys, fast) or
    # "omp" (Orthogonal Matching Pursuit, more accurate but slower).
    method: str = "hak"

    # Uniform target: keep this fraction of tokens.  Ignored when
    # head_budget_path is set (Phase 3).
    target_ratio: float = 0.10

    # Only compact when the stored sequence has at least this many tokens.
    min_source_tokens: int = 32

    # Never keep fewer than this many tokens per head (floor).
    min_kept_tokens: int = 4

    # Number of reference queries to use for key-selection scoring.
    # Uses the last ``n_ref_queries`` key vectors as proxy queries.
    n_ref_queries: int = 8

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    beta_enabled: bool = False  # fit β biases and Cv via least-squares

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    head_budget_path: str | None = None  # JSON file with per-head budgets

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    online_enabled: bool = False
    # After compaction the cache will be kept at most this many tokens.
    online_max_tokens: int = 512

    # ── Auto mode (6.1) ──────────────────────────────────────────────────────
    # "on"   — always compact (existing behaviour when CompactionConfig is created)
    # "auto" — compact only when stored seq_len > auto_threshold
    # "off"  — never compact (compaction skipped entirely)
    # Default is "on" so existing callers get the old behaviour unchanged.
    mode: str = "on"
    auto_threshold: int = 512   # tokens; only used when mode == "auto"
