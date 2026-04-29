# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b — DFlash drafter scaffolding tests.

Exercises the catalog wiring and the abstract drafter interface
without pulling either upstream DFlash package. The MockDFlashDrafter
is the workhorse for downstream Phase 2b tests (verify_block,
dual-topology, failover) that need deterministic draft output without
an optional dep.

What this file does NOT cover (deferred to later commits):
    * Actual DFlash inference (lands with the Topology A driver in
      Commit 9, requires the pip extras to run).
    * Verify-block prefix matching (Commit 5 — HeadSampler.verify_block).
    * KV rollback strategies (Commit 6).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ── Catalog resolution: pinned-model paths from Phase 2b launch matrix ──

def test_catalog_resolves_dflash_draft_for_qwen35_4b():
    """Phase 2b launch default: ``z-lab/Qwen3.5-4B-DFlash`` is the
    DFlash draft for ``openhydra-qwen3.5-4b``. Locked here so a
    catalog edit can't silently break the launch matrix."""
    from peer.model_catalog import resolve_dflash_draft_model_id

    catalog = Path(__file__).resolve().parents[1] / "models.catalog.json"
    assert resolve_dflash_draft_model_id(
        "openhydra-qwen3.5-4b", catalog_path=catalog,
    ) == "z-lab/Qwen3.5-4B-DFlash"


def test_catalog_resolves_mlx_target_for_qwen35_4b():
    """MLX peers running the 4B target use the 4-bit
    ``mlx-community/Qwen3.5-4B-MLX-4bit`` checkpoint. The 4-bit
    choice is deliberate — fp16 doesn't leave KV headroom for the
    16-position verify on 8 GB Macs until Phase 4 KV compression
    lands."""
    from peer.model_catalog import resolve_mlx_model_id

    catalog = Path(__file__).resolve().parents[1] / "models.catalog.json"
    assert resolve_mlx_model_id(
        "openhydra-qwen3.5-4b", catalog_path=catalog,
    ) == "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_catalog_resolves_cuda_target_for_qwen35_4b():
    """CUDA peers running the 4B target use the canonical
    ``Qwen/Qwen3.5-4B`` HF weights (fp16). The existing resolver
    already handled this; the test pins the launch-matrix value."""
    from peer.model_catalog import resolve_hf_model_id

    catalog = Path(__file__).resolve().parents[1] / "models.catalog.json"
    assert resolve_hf_model_id(
        "openhydra-qwen3.5-4b", catalog_path=catalog,
    ) == "Qwen/Qwen3.5-4B"


def test_catalog_dflash_lookup_returns_empty_for_unsupported_target():
    """Models without a published DFlash draft return ``""``. Caller
    must treat that as 'pass --draft-model explicitly or use
    --draft-location off' rather than silently using the wrong draft."""
    from peer.model_catalog import resolve_dflash_draft_model_id

    catalog = Path(__file__).resolve().parents[1] / "models.catalog.json"
    # 0.8B has no DFlash draft (per the plan's "out of scope" note —
    # waits on z-lab releasing the training recipe).
    assert resolve_dflash_draft_model_id(
        "openhydra-qwen3.5-0.8b", catalog_path=catalog,
    ) == ""


# ── DFlashConfig contract ──────────────────────────────────────────────

def test_dflash_config_defaults_match_phase_2b_launch_matrix():
    """``DFlashConfig()`` defaults to the Phase 2b launch matrix:
    ``Qwen/Qwen3.5-4B`` target, ``z-lab/Qwen3.5-4B-DFlash`` draft,
    block_size=16, MLX backend (because the Phase 2b launch hardware
    is the user's Apple Silicon coord)."""
    from coordinator.dflash_draft import DFlashConfig

    cfg = DFlashConfig()
    assert cfg.target_model_path == "Qwen/Qwen3.5-4B"
    assert cfg.draft_model_path == "z-lab/Qwen3.5-4B-DFlash"
    assert cfg.block_size == 16
    assert cfg.backend == "mlx"


def test_dflash_config_rejects_invalid_backend():
    """Backend must be one of the three supported values; anything else
    is a typo that should fail loud."""
    from coordinator.dflash_draft import DFlashConfig

    with pytest.raises(ValueError, match="backend must be"):
        DFlashConfig(backend="ollama")


def test_dflash_config_clamps_block_size_range():
    """block_size in [1, 32]. 0 and 33 both rejected."""
    from coordinator.dflash_draft import DFlashConfig

    with pytest.raises(ValueError, match="block_size"):
        DFlashConfig(block_size=0)
    with pytest.raises(ValueError, match="block_size"):
        DFlashConfig(block_size=33)


def test_dflash_config_is_frozen():
    """Immutable config — failover swaps the whole object via
    ``DFlashDrafter.reload(new_cfg)``, never mutates fields in place."""
    from coordinator.dflash_draft import DFlashConfig

    cfg = DFlashConfig()
    with pytest.raises(Exception):  # FrozenInstanceError or similar
        cfg.block_size = 8  # type: ignore[misc]


# ── Mock drafter — deterministic surface for downstream Phase 2b tests ──

def test_mock_drafter_is_deterministic_under_same_prefix():
    """Same prefix → same draft. Required for byte-equivalence
    regression guards under temp=0.0 in later phase tests."""
    from coordinator.dflash_draft import (
        DFlashConfig, MockDFlashDrafter,
    )

    cfg = DFlashConfig(backend="mock", block_size=16)
    drafter = MockDFlashDrafter(cfg)
    prefix = [1, 2, 3, 4, 5]
    assert drafter.draft(prefix) == drafter.draft(prefix)


def test_mock_drafter_returns_block_size_tokens():
    """Block-diffusion contract: every draft call emits exactly
    block_size tokens, regardless of acceptance pattern."""
    from coordinator.dflash_draft import (
        DFlashConfig, MockDFlashDrafter,
    )

    for bs in (1, 4, 16, 32):
        cfg = DFlashConfig(backend="mock", block_size=bs)
        drafter = MockDFlashDrafter(cfg)
        out = drafter.draft([42, 7])
        assert len(out) == bs, f"expected {bs} tokens, got {len(out)}"
        # All ids in valid uint32 range — survives proto serialise.
        assert all(0 <= t < 2**16 for t in out)


def test_mock_drafter_different_prefixes_diverge():
    """Different prefixes produce different drafts — sanity check
    that the seed actually depends on prefix content."""
    from coordinator.dflash_draft import (
        DFlashConfig, MockDFlashDrafter,
    )

    drafter = MockDFlashDrafter(DFlashConfig(backend="mock"))
    a = drafter.draft([1, 2, 3])
    b = drafter.draft([100, 200, 300])
    assert a != b


# ── Loader factory: backend selection + lazy load ──────────────────────

def test_load_dflash_drafter_picks_mock_backend():
    """Factory dispatches by ``cfg.backend``."""
    from coordinator.dflash_draft import (
        DFlashConfig, MockDFlashDrafter, load_dflash_drafter,
    )

    drafter = load_dflash_drafter(DFlashConfig(backend="mock"))
    assert isinstance(drafter, MockDFlashDrafter)


def test_load_dflash_drafter_does_not_load_eagerly():
    """``load_dflash_drafter`` must NOT touch the upstream package
    at construction time. Eager loads pay the dflash import cost
    even when ``--draft-location off`` was set."""
    from coordinator.dflash_draft import (
        DFlashConfig, load_dflash_drafter,
    )

    # PyTorch backend without dflash installed: should construct
    # successfully (lazy), only blow up if/when ensure_loaded is called.
    cfg = DFlashConfig(backend="pytorch")
    drafter = load_dflash_drafter(cfg)   # must not raise
    assert drafter.is_loaded is False    # not loaded yet


def test_load_dflash_drafter_pytorch_raises_clear_error_when_missing():
    """Without ``dflash`` on the path, ``ensure_loaded`` must raise
    ``DFlashNotAvailableError`` carrying the install hint."""
    import sys
    from coordinator.dflash_draft import (
        DFlashConfig, DFlashNotAvailableError, load_dflash_drafter,
    )

    if "dflash" in sys.modules:
        pytest.skip("dflash is installed on this runner; can't test missing-dep path")

    drafter = load_dflash_drafter(DFlashConfig(backend="pytorch"))
    with pytest.raises(DFlashNotAvailableError) as exc:
        drafter.ensure_loaded()
    assert exc.value.backend == "pytorch"
    assert "speculative-pytorch" in exc.value.hint


def test_load_dflash_drafter_unknown_backend_raises():
    """Defence in depth — DFlashConfig validates first, but if a
    subclass somehow sneaks through with a bad backend the factory
    must also fail loud."""
    from coordinator.dflash_draft import DFlashConfig, load_dflash_drafter

    # We have to bypass the dataclass validator to exercise this.
    cfg = DFlashConfig(backend="mock")
    object.__setattr__(cfg, "backend", "tensorrt")
    with pytest.raises(ValueError, match="unknown DFlash backend"):
        load_dflash_drafter(cfg)
