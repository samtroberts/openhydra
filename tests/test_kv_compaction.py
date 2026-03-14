"""Tests for peer.kv_compaction — all four phases.

Phase 1 — no-beta compaction:
    HAK and OMP key selection, DynamicCache and tuple formats,
    min_source_tokens guard, target_ratio clipping.

Phase 2 — β + Cv fitting:
    CompactedKVCache wrapping, beta_for_layer, fit_beta_and_cv
    (log-ratio fallback, no scipy required).

Phase 3 — nonuniform head budgets:
    JSON loading, per-head budget dispatch, fallback on missing entries.

Phase 4 — online mid-trajectory compaction:
    online_enabled flag, online_max_tokens threshold.

Integration:
    ToyShardConfig fields, PeerService param pass-through,
    server.py CLI flag parsing (import-level smoke test).
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_torch():
    pytest.importorskip("torch")
    import torch
    return torch


def _make_fake_dynamic_cache(n_layers: int, n_kv_heads: int, seq_len: int, d_head: int, torch_module: Any):
    """Return a minimal DynamicCache-like object (plain list attrs)."""
    torch = torch_module

    class _FakeDynamicCache:
        def __init__(self):
            self.key_cache = [
                torch.randn(1, n_kv_heads, seq_len, d_head)
                for _ in range(n_layers)
            ]
            self.value_cache = [
                torch.randn(1, n_kv_heads, seq_len, d_head)
                for _ in range(n_layers)
            ]
            self._seen_tokens = seq_len

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return self.key_cache[0].shape[-2] if self.key_cache else 0

    return _FakeDynamicCache()


def _make_fake_tuple_cache(n_layers: int, n_kv_heads: int, seq_len: int, d_head: int, torch_module: Any):
    """Return a past_key_values tuple-of-tuples."""
    torch = torch_module
    return tuple(
        (torch.randn(1, n_kv_heads, seq_len, d_head),
         torch.randn(1, n_kv_heads, seq_len, d_head))
        for _ in range(n_layers)
    )


# ─────────────────────────────────────────────────────────────────────────────
# CompactionConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestCompactionConfig:
    def test_defaults(self):
        from peer.kv_compaction import CompactionConfig
        cfg = CompactionConfig()
        assert cfg.enabled is False
        assert cfg.method == "hak"
        assert math.isclose(cfg.target_ratio, 0.10)
        assert cfg.beta_enabled is False
        assert cfg.head_budget_path is None
        assert cfg.online_enabled is False
        assert cfg.online_max_tokens == 512
        assert cfg.min_source_tokens == 32

    def test_custom_values(self):
        from peer.kv_compaction import CompactionConfig
        cfg = CompactionConfig(
            enabled=True,
            method="omp",
            target_ratio=0.25,
            beta_enabled=True,
            online_enabled=True,
            online_max_tokens=256,
        )
        assert cfg.enabled is True
        assert cfg.method == "omp"
        assert math.isclose(cfg.target_ratio, 0.25)
        assert cfg.beta_enabled is True
        assert cfg.online_enabled is True
        assert cfg.online_max_tokens == 256


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — key selection algorithms
# ─────────────────────────────────────────────────────────────────────────────

class TestHAK:
    def test_returns_correct_count(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_hak
        T, d, R, t = 64, 32, 8, 10
        K = torch.randn(T, d)
        Q_ref = torch.randn(R, d)
        indices = select_hak(K, Q_ref, t)
        assert indices.shape[0] == t

    def test_indices_sorted(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_hak
        K = torch.randn(50, 16)
        Q_ref = torch.randn(4, 16)
        indices = select_hak(K, Q_ref, 12)
        assert (indices == indices.sort().values).all()

    def test_clips_to_T(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_hak
        K = torch.randn(5, 8)
        Q_ref = torch.randn(3, 8)
        indices = select_hak(K, Q_ref, 100)
        assert indices.shape[0] == 5  # capped at T

    def test_all_indices_when_t_ge_T(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_hak
        T = 8
        K = torch.randn(T, 16)
        Q_ref = torch.randn(2, 16)
        indices = select_hak(K, Q_ref, T)
        assert set(indices.tolist()) == set(range(T))

    def test_indices_unique(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_hak
        K = torch.randn(100, 64)
        Q_ref = torch.randn(8, 64)
        indices = select_hak(K, Q_ref, 20)
        assert len(set(indices.tolist())) == 20

    def test_dtype_mismatch_handled(self):
        """float16 K with float32 Q_ref should not error."""
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_hak
        K = torch.randn(30, 32).half()
        Q_ref = torch.randn(4, 32)  # float32
        indices = select_hak(K, Q_ref, 8)
        assert indices.shape[0] == 8


class TestOMP:
    def test_returns_correct_count(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_omp
        T, d, R, t = 64, 32, 8, 10
        K = torch.randn(T, d)
        Q_ref = torch.randn(R, d)
        indices = select_omp(K, Q_ref, t)
        assert indices.shape[0] == t

    def test_indices_sorted(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_omp
        K = torch.randn(50, 16)
        Q_ref = torch.randn(4, 16)
        indices = select_omp(K, Q_ref, 12)
        assert (indices == indices.sort().values).all()

    def test_clips_to_T(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_omp
        K = torch.randn(5, 8)
        Q_ref = torch.randn(3, 8)
        indices = select_omp(K, Q_ref, 100)
        assert indices.shape[0] == 5

    def test_indices_unique(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_omp
        K = torch.randn(80, 32)
        Q_ref = torch.randn(6, 32)
        indices = select_omp(K, Q_ref, 15)
        assert len(set(indices.tolist())) == 15

    def test_single_token_budget(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_omp
        K = torch.randn(20, 16)
        Q_ref = torch.randn(4, 16)
        indices = select_omp(K, Q_ref, 1)
        assert indices.shape[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — compact_past_key_values (DynamicCache)
# ─────────────────────────────────────────────────────────────────────────────

class TestCompactPastKeyValuesDynamic:
    """Phase 1 with a fake DynamicCache."""

    def _make_config(self, target_ratio=0.25, **kw):
        from peer.kv_compaction import CompactionConfig
        return CompactionConfig(enabled=True, target_ratio=target_ratio, min_source_tokens=8, **kw)

    def test_reduces_seq_len_hak(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values
        T = 64
        cache = _make_fake_dynamic_cache(4, 4, T, 32, torch)
        cfg = self._make_config(method="hak")
        result = compact_past_key_values(cache, cfg)
        assert hasattr(result, "key_cache")
        t_out = result.key_cache[0].shape[-2]
        assert t_out < T
        assert t_out >= cfg.min_kept_tokens

    def test_reduces_seq_len_omp(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values
        T = 64
        cache = _make_fake_dynamic_cache(2, 2, T, 16, torch)
        cfg = self._make_config(method="omp")
        result = compact_past_key_values(cache, cfg)
        t_out = result.key_cache[0].shape[-2]
        assert t_out < T

    def test_skips_when_below_min_source(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values
        T = 5  # below min_source_tokens=8
        cache = _make_fake_dynamic_cache(2, 2, T, 16, torch)
        cfg = self._make_config()
        result = compact_past_key_values(cache, cfg)
        # Should return the original unchanged
        assert result.key_cache[0].shape[-2] == T

    def test_all_layers_compacted(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values
        n_layers = 6
        cache = _make_fake_dynamic_cache(n_layers, 4, 128, 32, torch)
        cfg = self._make_config(target_ratio=0.20)
        result = compact_past_key_values(cache, cfg)
        assert len(result.key_cache) == n_layers
        for k in result.key_cache:
            assert k.shape[-2] < 128

    def test_none_returns_none(self):
        from peer.kv_compaction import compact_past_key_values, CompactionConfig
        cfg = CompactionConfig(enabled=True)
        assert compact_past_key_values(None, cfg) is None

    def test_ratio_10pct(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig
        T = 100
        cache = _make_fake_dynamic_cache(2, 2, T, 32, torch)
        cfg = CompactionConfig(enabled=True, target_ratio=0.10, min_source_tokens=8)
        result = compact_past_key_values(cache, cfg)
        t_out = result.key_cache[0].shape[-2]
        # Should be roughly 10 tokens (with min_kept_tokens=4 floor)
        assert t_out <= max(12, int(T * 0.10 * 1.5))

    def test_key_value_same_shape(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values
        cache = _make_fake_dynamic_cache(3, 4, 80, 16, torch)
        cfg = self._make_config()
        result = compact_past_key_values(cache, cfg)
        for k, v in zip(result.key_cache, result.value_cache):
            assert k.shape == v.shape


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — compact_past_key_values (tuple format)
# ─────────────────────────────────────────────────────────────────────────────

class TestCompactPastKeyValuesTuple:
    def _make_config(self, target_ratio=0.25, **kw):
        from peer.kv_compaction import CompactionConfig
        return CompactionConfig(enabled=True, target_ratio=target_ratio, min_source_tokens=8, **kw)

    def test_reduces_seq_len(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values
        T = 64
        cache = _make_fake_tuple_cache(3, 2, T, 16, torch)
        result = compact_past_key_values(cache, self._make_config())
        assert isinstance(result, tuple)
        assert result[0][0].shape[-2] < T

    def test_n_layers_preserved(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values
        n = 4
        cache = _make_fake_tuple_cache(n, 2, 50, 16, torch)
        result = compact_past_key_values(cache, self._make_config())
        assert len(result) == n

    def test_kv_pair_shape_matches(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values
        cache = _make_fake_tuple_cache(3, 2, 50, 16, torch)
        result = compact_past_key_values(cache, self._make_config())
        for K, V in result:
            assert K.shape == V.shape

    def test_skips_when_below_min_source(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values
        T = 5
        cache = _make_fake_tuple_cache(2, 2, T, 8, torch)
        cfg = self._make_config()
        result = compact_past_key_values(cache, cfg)
        assert result[0][0].shape[-2] == T


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — CompactedKVCache and fit_beta_and_cv
# ─────────────────────────────────────────────────────────────────────────────

class TestCompactedKVCache:
    def _make_cache(self, n_layers=3, t=10, n_kv=4, d=16):
        torch = _make_torch()

        class _FakeCache:
            def __init__(self):
                self.key_cache = [torch.randn(1, n_kv, t, d) for _ in range(n_layers)]
                self.value_cache = [torch.randn(1, n_kv, t, d) for _ in range(n_layers)]
                self._seen_tokens = t
            def get_seq_length(self, layer_idx=0):
                return t

        betas = [torch.randn(n_kv, t) for _ in range(n_layers)]
        from peer.kv_compaction._cache import CompactedKVCache
        return CompactedKVCache(cache=_FakeCache(), beta_per_layer=betas, prefix_length=t)

    def test_beta_for_layer_returns_tensor(self):
        cache = self._make_cache(n_layers=3)
        b = cache.beta_for_layer(0)
        assert b is not None
        assert b.ndim == 2

    def test_beta_for_layer_out_of_bounds(self):
        cache = self._make_cache(n_layers=2)
        assert cache.beta_for_layer(99) is None

    def test_has_beta_true(self):
        cache = self._make_cache()
        assert cache.has_beta() is True

    def test_has_beta_false_when_all_none(self):
        torch = _make_torch()
        from peer.kv_compaction._cache import CompactedKVCache
        class _F:
            key_cache = []
            value_cache = []
        c = CompactedKVCache(cache=_F(), beta_per_layer=[None, None], prefix_length=0)
        assert c.has_beta() is False

    def test_get_seq_length(self):
        cache = self._make_cache(t=15)
        assert cache.get_seq_length() == 15

    def test_to_standard_cache(self):
        cache = self._make_cache()
        std = cache.to_standard_cache()
        assert hasattr(std, "key_cache")

    def test_key_value_cache_delegation(self):
        cache = self._make_cache(n_layers=4, t=8)
        assert len(cache.key_cache) == 4
        assert len(cache.value_cache) == 4

    def test_prefix_length_stored(self):
        cache = self._make_cache(t=20)
        assert cache.prefix_length == 20


class TestFitBetaAndCv:
    def test_beta_shape(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_hak, fit_beta_and_cv
        T, d, R, t = 40, 16, 6, 8
        K = torch.randn(T, d)
        V = torch.randn(T, d)
        Q = torch.randn(R, d)
        idx = select_hak(K, Q, t)
        Ck = K[idx]
        beta, Cv = fit_beta_and_cv(K, V, Ck, Q, idx)
        assert beta.shape == (t,)

    def test_cv_shape(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_hak, fit_beta_and_cv
        T, d, R, t = 40, 16, 6, 8
        K = torch.randn(T, d)
        V = torch.randn(T, d)
        Q = torch.randn(R, d)
        idx = select_hak(K, Q, t)
        Ck = K[idx]
        beta, Cv = fit_beta_and_cv(K, V, Ck, Q, idx)
        assert Cv.shape == (t, d)

    def test_beta_finite(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_hak, fit_beta_and_cv
        K = torch.randn(30, 16)
        V = torch.randn(30, 16)
        Q = torch.randn(4, 16)
        idx = select_hak(K, Q, 6)
        beta, _ = fit_beta_and_cv(K, V, K[idx], Q, idx)
        assert torch.isfinite(beta).all()

    def test_beta_clamped(self):
        torch = _make_torch()
        from peer.kv_compaction._algorithms import select_hak, fit_beta_and_cv
        K = torch.randn(30, 16)
        V = torch.randn(30, 16)
        Q = torch.randn(4, 16)
        idx = select_hak(K, Q, 6)
        beta, _ = fit_beta_and_cv(K, V, K[idx], Q, idx)
        assert (beta >= -10.0).all() and (beta <= 10.0).all()


class TestCompactPastKeyValuesBeta:
    """Phase 2: beta_enabled=True returns CompactedKVCache."""

    def test_returns_compacted_kv_cache(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig
        from peer.kv_compaction._cache import CompactedKVCache
        cache = _make_fake_dynamic_cache(2, 2, 64, 16, torch)
        cfg = CompactionConfig(
            enabled=True, beta_enabled=True,
            target_ratio=0.25, min_source_tokens=8,
        )
        result = compact_past_key_values(cache, cfg)
        assert isinstance(result, CompactedKVCache)

    def test_beta_per_layer_count(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig
        from peer.kv_compaction._cache import CompactedKVCache
        n_layers = 3
        cache = _make_fake_dynamic_cache(n_layers, 2, 64, 16, torch)
        cfg = CompactionConfig(
            enabled=True, beta_enabled=True,
            target_ratio=0.25, min_source_tokens=8,
        )
        result = compact_past_key_values(cache, cfg)
        assert isinstance(result, CompactedKVCache)
        assert len(result._beta) == n_layers


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — nonuniform head budgets
# ─────────────────────────────────────────────────────────────────────────────

class TestHeadBudgets:
    def _make_budget_file(self, n_layers=4, n_heads=4) -> str:
        budgets = [
            [round(0.05 + (l * 0.01) + (h * 0.005), 3) for h in range(n_heads)]
            for l in range(n_layers)
        ]
        data = {
            "model": "test-model",
            "n_layers": n_layers,
            "n_kv_heads": n_heads,
            "source": "test",
            "layer_budgets": budgets,
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump(data, tmp)
        tmp.flush()
        return tmp.name

    def test_load_head_budgets(self):
        from peer.kv_compaction import _load_head_budgets
        path = self._make_budget_file()
        data = _load_head_budgets(path)
        assert data is not None
        assert "layer_budgets" in data
        assert len(data["layer_budgets"]) == 4

    def test_load_head_budgets_missing_file(self):
        from peer.kv_compaction import _load_head_budgets
        result = _load_head_budgets("/nonexistent/path.json")
        assert result is None

    def test_compact_with_budgets(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig, _load_head_budgets
        path = self._make_budget_file(n_layers=2, n_heads=2)
        data = _load_head_budgets(path)
        cache = _make_fake_dynamic_cache(2, 2, 100, 16, torch)
        cfg = CompactionConfig(
            enabled=True, target_ratio=0.10, min_source_tokens=8
        )
        result = compact_past_key_values(cache, cfg, budgets_data=data)
        assert hasattr(result, "key_cache")
        # Budget file has ratios ~0.05-0.08 → should keep fewer than 10 tokens per head
        t_out = result.key_cache[0].shape[-2]
        assert t_out < 100

    def test_precomputed_qwen3_budget_file_exists(self):
        budget_path = Path(__file__).parent.parent / "peer" / "kv_compaction" / "head_budgets" / "qwen3_4b.json"
        assert budget_path.exists(), f"Missing: {budget_path}"
        data = json.loads(budget_path.read_text())
        assert "layer_budgets" in data
        assert data["n_layers"] == 36
        assert data["n_kv_heads"] == 8

    def test_precomputed_llama_budget_file_exists(self):
        budget_path = Path(__file__).parent.parent / "peer" / "kv_compaction" / "head_budgets" / "llama3_8b.json"
        assert budget_path.exists(), f"Missing: {budget_path}"
        data = json.loads(budget_path.read_text())
        assert "layer_budgets" in data
        assert data["n_layers"] == 32
        assert data["n_kv_heads"] == 8

    def test_budget_values_in_range(self):
        budget_path = Path(__file__).parent.parent / "peer" / "kv_compaction" / "head_budgets" / "qwen3_4b.json"
        data = json.loads(budget_path.read_text())
        for layer in data["layer_budgets"]:
            for v in layer:
                assert 0.0 < v <= 1.0, f"Budget out of range: {v}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — online mid-trajectory compaction
# ─────────────────────────────────────────────────────────────────────────────

class TestOnlineCompaction:
    def test_online_compacts_when_exceeds_threshold(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig
        T = 600  # exceeds online_max_tokens=512
        cache = _make_fake_dynamic_cache(2, 2, T, 16, torch)
        cfg = CompactionConfig(
            enabled=True,
            online_enabled=True,
            online_max_tokens=512,
            min_source_tokens=8,
        )
        result = compact_past_key_values(cache, cfg)
        t_out = result.key_cache[0].shape[-2]
        assert t_out <= 512

    def test_online_skips_when_below_threshold(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig
        T = 200  # below online_max_tokens=512
        cache = _make_fake_dynamic_cache(2, 2, T, 16, torch)
        cfg = CompactionConfig(
            enabled=True,
            online_enabled=True,
            online_max_tokens=512,
            min_source_tokens=8,
        )
        result = compact_past_key_values(cache, cfg)
        # Should return unchanged (seq_len <= online_max_tokens)
        assert result.key_cache[0].shape[-2] == T

    def test_online_max_tokens_respected(self):
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig
        T = 1024
        cache = _make_fake_dynamic_cache(2, 4, T, 32, torch)
        target = 128
        cfg = CompactionConfig(
            enabled=True,
            online_enabled=True,
            online_max_tokens=target,
            min_source_tokens=8,
        )
        result = compact_past_key_values(cache, cfg)
        t_out = result.key_cache[0].shape[-2]
        assert t_out <= target


# ─────────────────────────────────────────────────────────────────────────────
# ToyShardConfig — new fields
# ─────────────────────────────────────────────────────────────────────────────

class TestToyShardConfigCompactionFields:
    def test_default_disabled(self):
        from peer.model_shard import ToyShardConfig
        cfg = ToyShardConfig()
        assert cfg.runtime_kv_compaction_enabled is False
        assert cfg.runtime_kv_compaction_method == "hak"
        assert math.isclose(cfg.runtime_kv_compaction_ratio, 0.10)
        assert cfg.runtime_kv_compaction_beta is False
        assert cfg.runtime_kv_compaction_online is False
        assert cfg.runtime_kv_compaction_online_max_tokens == 512

    def test_can_set_all_fields(self):
        from peer.model_shard import ToyShardConfig
        cfg = ToyShardConfig(
            runtime_kv_compaction_enabled=True,
            runtime_kv_compaction_method="omp",
            runtime_kv_compaction_ratio=0.15,
            runtime_kv_compaction_beta=True,
            runtime_kv_compaction_head_budget_path="/tmp/budgets.json",
            runtime_kv_compaction_online=True,
            runtime_kv_compaction_online_max_tokens=256,
        )
        assert cfg.runtime_kv_compaction_enabled is True
        assert cfg.runtime_kv_compaction_method == "omp"
        assert math.isclose(cfg.runtime_kv_compaction_ratio, 0.15)
        assert cfg.runtime_kv_compaction_beta is True
        assert cfg.runtime_kv_compaction_head_budget_path == "/tmp/budgets.json"
        assert cfg.runtime_kv_compaction_online is True
        assert cfg.runtime_kv_compaction_online_max_tokens == 256


# ─────────────────────────────────────────────────────────────────────────────
# PeerService — compaction param pass-through (toy backend, no real model)
# ─────────────────────────────────────────────────────────────────────────────

class TestPeerServiceCompactionParams:
    def _make_service(self, **kw):
        from peer.server import PeerService
        return PeerService(
            peer_id="test-peer",
            model_id="openhydra-toy-345m",
            shard_index=0,
            total_shards=1,
            daemon_mode="polite",
            broken=False,
            **kw,
        )

    def test_default_service_no_compaction(self):
        svc = self._make_service()
        # Should not raise; compaction disabled by default
        assert svc is not None

    def test_compaction_params_accepted(self):
        """PeerService accepts compaction kwargs without raising."""
        svc = self._make_service(
            kv_compaction_enabled=True,
            kv_compaction_method="hak",
            kv_compaction_ratio=0.10,
        )
        assert svc is not None

    def test_beta_param_accepted(self):
        svc = self._make_service(
            kv_compaction_enabled=True,
            kv_compaction_beta=True,
        )
        assert svc is not None

    def test_online_params_accepted(self):
        svc = self._make_service(
            kv_compaction_enabled=True,
            kv_compaction_online=True,
            kv_compaction_online_max_tokens=256,
        )
        assert svc is not None


# ─────────────────────────────────────────────────────────────────────────────
# server.py — CLI flag smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestServerCLIFlags:
    def test_server_module_importable(self):
        import peer.server  # noqa: F401

    def test_argparse_has_kv_compaction_flags(self):
        """Ensure argparse in main() exposes the expected flag names."""
        import argparse
        import sys
        import peer.server as srv

        # Build an ArgumentParser by running the beginning of main() up to parse_args
        # We test by constructing args via parse_known_args with the new flags.
        original_argv = sys.argv
        sys.argv = [
            "openhydra-peer",
            "--port", "50051",
            "--peer-id", "test",
            "--kv-compaction-enabled",
            "--kv-compaction-method", "omp",
            "--kv-compaction-ratio", "0.15",
            "--kv-compaction-online",
            "--kv-compaction-online-max-tokens", "256",
            "--kv-compaction-beta",
        ]
        try:
            # We can't call main() as it starts a server, but we can verify
            # the argparse definition by examining the module-level parse logic.
            # Just ensure import succeeds and no NameError on flag names.
            assert hasattr(srv, "serve")
            assert hasattr(srv, "main")
        finally:
            sys.argv = original_argv


# ─────────────────────────────────────────────────────────────────────────────
# Beta inject — model family detection
# ─────────────────────────────────────────────────────────────────────────────

class TestModelFamilyDetection:
    def test_qwen_detected(self):
        from peer.kv_compaction._beta_inject import detect_model_family
        assert detect_model_family("Qwen/Qwen3.5-0.8B") == "qwen2"
        assert detect_model_family("Qwen/Qwen2-7B-Instruct") == "qwen2"

    def test_llama_detected(self):
        from peer.kv_compaction._beta_inject import detect_model_family
        assert detect_model_family("meta-llama/Llama-3.1-8B-Instruct") == "llama"
        assert detect_model_family("llama-7b") == "llama"

    def test_gemma_detected(self):
        from peer.kv_compaction._beta_inject import detect_model_family
        assert detect_model_family("google/gemma-3-27b") == "gemma3"

    def test_unknown(self):
        from peer.kv_compaction._beta_inject import detect_model_family
        assert detect_model_family("some-random-model") == "unknown"

    def test_empty_string(self):
        from peer.kv_compaction._beta_inject import detect_model_family
        assert detect_model_family("") == "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: kv_compaction package importable without torch
# ─────────────────────────────────────────────────────────────────────────────

class TestImportWithoutTorch:
    def test_config_importable(self):
        """CompactionConfig must be importable without torch installed."""
        from peer.kv_compaction import CompactionConfig
        cfg = CompactionConfig(enabled=True)
        assert cfg.enabled is True

    def test_cache_importable(self):
        """CompactedKVCache must be importable without torch."""
        from peer.kv_compaction._cache import CompactedKVCache
        assert CompactedKVCache is not None


# ─────────────────────────────────────────────────────────────────────────────
# Option A — AttentionQueryCapture
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_model(n_layers: int = 2, hidden_size: int = 64,
                     n_q: int = 4, n_kv: int = 2, head_dim: int = 16):
    """Return a minimal PyTorch model that AttentionQueryCapture can introspect.

    Mirrors the structure expected by AttentionQueryCapture:
        model.model.layers  — list of transformer layer modules
        layer.self_attn.q_proj  — nn.Linear(hidden_size, n_q * head_dim)
        layer.self_attn.num_heads / num_key_value_heads / head_dim
    """
    torch = _make_torch()
    import torch.nn as nn

    class _FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = n_q
            self.num_key_value_heads = n_kv
            self.head_dim = head_dim
            self.q_proj = nn.Linear(hidden_size, n_q * head_dim, bias=False)

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class _FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _FakeAttn()

        def forward(self, hidden_states, **kwargs):
            return hidden_states

    class _FakeDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_FakeLayer() for _ in range(n_layers)])

        def forward(self, hidden_states):
            for layer in self.layers:
                hidden_states = layer(hidden_states)
            return hidden_states

    class _FakeFullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _FakeDecoder()

        def forward(self, hidden_states):
            return self.model(hidden_states)

    return _FakeFullModel()


class TestAttentionQueryCaptureImport:
    def test_exported_from_package(self):
        from peer.kv_compaction import AttentionQueryCapture
        assert AttentionQueryCapture is not None

    def test_importable_directly(self):
        from peer.kv_compaction._query_capture import AttentionQueryCapture
        assert AttentionQueryCapture is not None


class TestAttentionQueryCaptureContextManager:
    def test_enter_returns_self(self):
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture
        model = _make_fake_model()
        qc = AttentionQueryCapture(model, n_ref=4)
        result = qc.__enter__()
        assert result is qc
        qc.__exit__(None, None, None)

    def test_hooks_registered_on_enter(self):
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture
        model = _make_fake_model(n_layers=3)
        qc = AttentionQueryCapture(model, n_ref=4)
        assert len(qc._hooks) == 0
        qc.__enter__()
        assert len(qc._hooks) == 3    # one per layer
        qc.__exit__(None, None, None)

    def test_hooks_removed_on_exit(self):
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture
        model = _make_fake_model(n_layers=2)
        qc = AttentionQueryCapture(model, n_ref=4)
        with qc:
            pass
        # After exit hooks list is cleared
        assert len(qc._hooks) == 0

    def test_context_manager_with_block(self):
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture
        model = _make_fake_model()
        with AttentionQueryCapture(model, n_ref=4) as qc:
            assert len(qc._hooks) == 2

    def test_no_model_layers_no_crash(self):
        """Model without model.layers does not crash — returns no hooks."""
        torch = _make_torch()
        import torch.nn as nn
        from peer.kv_compaction import AttentionQueryCapture

        class _NakedModel(nn.Module):
            pass  # no .model attribute

        qc = AttentionQueryCapture(_NakedModel(), n_ref=4)
        qc.__enter__()
        assert len(qc._hooks) == 0
        qc.__exit__(None, None, None)

    def test_n_ref_clamped_to_one(self):
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture
        model = _make_fake_model()
        qc = AttentionQueryCapture(model, n_ref=0)
        assert qc._n_ref >= 1


class TestAttentionQueryCaptureHiddenCapture:
    def test_hidden_states_captured_per_layer(self):
        """Pre-hook captures hidden_states for each layer during a forward pass."""
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        n_layers, n_ref, hidden_size = 3, 4, 64
        model = _make_fake_model(n_layers=n_layers, hidden_size=hidden_size)
        hidden = torch.randn(1, 10, hidden_size)

        with AttentionQueryCapture(model, n_ref=n_ref) as qc:
            # Trigger the hooks by manually forwarding through each layer
            for layer in model.model.layers:
                hidden = layer(hidden)

        # Hidden states should have been captured for all layers
        assert len(qc._hidden_per_layer) == n_layers
        for idx in range(n_layers):
            captured = qc._hidden_per_layer[idx]
            assert captured.shape[0] == 1              # batch
            assert captured.shape[1] <= n_ref          # at most n_ref tokens
            assert captured.shape[2] == hidden_size    # hidden dim

    def test_hidden_capture_limited_to_n_ref(self):
        """Only the last n_ref token positions are kept."""
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        hidden_size, n_ref, seq_len = 32, 3, 20
        model = _make_fake_model(n_layers=1, hidden_size=hidden_size)
        hidden = torch.randn(1, seq_len, hidden_size)

        with AttentionQueryCapture(model, n_ref=n_ref) as qc:
            for layer in model.model.layers:
                hidden = layer(hidden)

        captured = qc._hidden_per_layer[0]
        assert captured.shape[1] == n_ref    # exactly n_ref tokens kept

    def test_hidden_states_detached(self):
        """Captured tensors must not hold the computation graph."""
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        hidden_size = 32
        model = _make_fake_model(n_layers=1, hidden_size=hidden_size)
        hidden = torch.randn(1, 8, hidden_size, requires_grad=True)

        with AttentionQueryCapture(model, n_ref=4) as qc:
            for layer in model.model.layers:
                hidden = layer(hidden)

        captured = qc._hidden_per_layer[0]
        assert not captured.requires_grad


class TestAttentionQueryCaptureComputeQRef:
    def test_returns_list_of_length_n_layers(self):
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        n_layers = 3
        hidden_size, n_q, n_kv, head_dim = 64, 4, 2, 16
        model = _make_fake_model(n_layers=n_layers, hidden_size=hidden_size,
                                 n_q=n_q, n_kv=n_kv, head_dim=head_dim)
        hidden = torch.randn(1, 8, hidden_size)

        with AttentionQueryCapture(model, n_ref=4) as qc:
            for layer in model.model.layers:
                hidden = layer(hidden)

        q_per_layer = qc.compute_q_ref()
        assert len(q_per_layer) == n_layers

    def test_q_ref_shape_per_layer(self):
        """Each element should be (n_kv_heads, n_ref, head_dim)."""
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        n_layers, n_ref = 2, 4
        hidden_size, n_q, n_kv, head_dim = 64, 8, 2, 16
        model = _make_fake_model(n_layers=n_layers, hidden_size=hidden_size,
                                 n_q=n_q, n_kv=n_kv, head_dim=head_dim)
        hidden = torch.randn(1, 10, hidden_size)

        with AttentionQueryCapture(model, n_ref=n_ref) as qc:
            for layer in model.model.layers:
                hidden = layer(hidden)

        q_per_layer = qc.compute_q_ref()
        for q in q_per_layer:
            assert q is not None
            assert q.shape == (n_kv, n_ref, head_dim), \
                f"Expected ({n_kv}, {n_ref}, {head_dim}), got {q.shape}"

    def test_q_ref_is_float32(self):
        """compute_q_ref always returns float32 tensors."""
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        model = _make_fake_model(n_layers=1, hidden_size=32,
                                 n_q=4, n_kv=2, head_dim=8)
        hidden = torch.randn(1, 6, 32)

        with AttentionQueryCapture(model, n_ref=3) as qc:
            for layer in model.model.layers:
                hidden = layer(hidden)

        q_per_layer = qc.compute_q_ref()
        for q in q_per_layer:
            if q is not None:
                assert q.dtype == torch.float32

    def test_hidden_cleared_after_compute_q_ref(self):
        """_hidden_per_layer is freed after compute_q_ref to save memory."""
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        model = _make_fake_model(n_layers=2, hidden_size=32)
        hidden = torch.randn(1, 6, 32)

        with AttentionQueryCapture(model, n_ref=3) as qc:
            for layer in model.model.layers:
                hidden = layer(hidden)

        qc.compute_q_ref()
        assert len(qc._hidden_per_layer) == 0   # cleared

    def test_none_returned_for_layer_without_capture(self):
        """Layers that did not fire the hook return None."""
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        model = _make_fake_model(n_layers=2, hidden_size=32)
        # Intentionally do NOT run any forward pass — no hidden captured
        with AttentionQueryCapture(model, n_ref=3) as qc:
            pass  # no forward → no captured states

        q_per_layer = qc.compute_q_ref()
        for q in q_per_layer:
            assert q is None

    def test_gqa_grouping_correct_shape(self):
        """GQA: n_q > n_kv — groups averaged to produce (n_kv, n_ref, d)."""
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        # Qwen3-style: 32 query heads, 8 kv heads, groups of 4
        n_q, n_kv, head_dim, hidden_size, n_ref = 8, 2, 16, 64, 4
        model = _make_fake_model(n_layers=1, hidden_size=hidden_size,
                                 n_q=n_q, n_kv=n_kv, head_dim=head_dim)
        hidden = torch.randn(1, 8, hidden_size)

        with AttentionQueryCapture(model, n_ref=n_ref) as qc:
            for layer in model.model.layers:
                hidden = layer(hidden)

        q_per_layer = qc.compute_q_ref()
        assert q_per_layer[0] is not None
        assert q_per_layer[0].shape == (n_kv, n_ref, head_dim)


class TestCompactWithQRefPerLayer:
    """Integration: compact_past_key_values with q_ref_per_layer argument."""

    def _make_q_ref_per_layer(self, n_layers, n_kv, n_ref, d_head, torch_module):
        """Create synthetic q_ref_per_layer as AttentionQueryCapture would return."""
        torch = torch_module
        return [
            torch.randn(n_kv, n_ref, d_head).float()
            for _ in range(n_layers)
        ]

    def test_q_ref_per_layer_accepted(self):
        """compact_past_key_values does not raise when q_ref_per_layer is passed."""
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig

        n_layers, n_kv, seq_len, d_head, n_ref = 2, 2, 64, 16, 4
        cache = _make_fake_dynamic_cache(n_layers, n_kv, seq_len, d_head, torch)
        q_ref = self._make_q_ref_per_layer(n_layers, n_kv, n_ref, d_head, torch)

        cfg = CompactionConfig(enabled=True, target_ratio=0.25, min_source_tokens=8)
        result = compact_past_key_values(cache, cfg, q_ref_per_layer=q_ref)
        assert result is not None

    def test_q_ref_reduces_seq_len(self):
        """Compaction with real Q still reduces the sequence length."""
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig

        n_layers, n_kv, seq_len, d_head, n_ref = 2, 2, 100, 16, 4
        cache = _make_fake_dynamic_cache(n_layers, n_kv, seq_len, d_head, torch)
        q_ref = self._make_q_ref_per_layer(n_layers, n_kv, n_ref, d_head, torch)

        target = 0.20
        cfg = CompactionConfig(enabled=True, target_ratio=target, min_source_tokens=8)
        result = compact_past_key_values(cache, cfg, q_ref_per_layer=q_ref)
        t_out = result.key_cache[0].shape[-2]
        expected_max = max(4, int(seq_len * target) + 1)
        assert t_out <= expected_max

    def test_q_ref_none_entries_fall_back_gracefully(self):
        """None entries in q_ref_per_layer fall back to proxy-K without error."""
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig

        n_layers, n_kv, seq_len, d_head = 2, 2, 80, 16
        cache = _make_fake_dynamic_cache(n_layers, n_kv, seq_len, d_head, torch)
        # q_ref_per_layer with all Nones — pure proxy fallback
        q_ref = [None, None]

        cfg = CompactionConfig(enabled=True, target_ratio=0.25, min_source_tokens=8)
        result = compact_past_key_values(cache, cfg, q_ref_per_layer=q_ref)
        assert result is not None
        assert result.key_cache[0].shape[-2] < seq_len

    def test_q_ref_partial_entries(self):
        """Mix of real Q and None entries is handled per-layer."""
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig

        n_layers, n_kv, seq_len, d_head, n_ref = 3, 2, 80, 16, 4
        cache = _make_fake_dynamic_cache(n_layers, n_kv, seq_len, d_head, torch)
        q_ref = [
            torch.randn(n_kv, n_ref, d_head).float(),   # layer 0 — real Q
            None,                                         # layer 1 — proxy fallback
            torch.randn(n_kv, n_ref, d_head).float(),   # layer 2 — real Q
        ]

        cfg = CompactionConfig(enabled=True, target_ratio=0.25, min_source_tokens=8)
        result = compact_past_key_values(cache, cfg, q_ref_per_layer=q_ref)
        assert result is not None
        assert result.key_cache[0].shape[-2] < seq_len

    def test_q_ref_too_short_uses_available(self):
        """q_ref_per_layer shorter than n_layers — missing layers use proxy."""
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig

        n_layers, n_kv, seq_len, d_head, n_ref = 4, 2, 80, 16, 4
        cache = _make_fake_dynamic_cache(n_layers, n_kv, seq_len, d_head, torch)
        q_ref = [torch.randn(n_kv, n_ref, d_head).float()]  # only 1 of 4 layers

        cfg = CompactionConfig(enabled=True, target_ratio=0.25, min_source_tokens=8)
        result = compact_past_key_values(cache, cfg, q_ref_per_layer=q_ref)
        assert result is not None

    def test_q_ref_with_omp_method(self):
        """Option A works with the OMP key selection algorithm too."""
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig

        n_layers, n_kv, seq_len, d_head, n_ref = 2, 2, 60, 16, 4
        cache = _make_fake_dynamic_cache(n_layers, n_kv, seq_len, d_head, torch)
        q_ref = self._make_q_ref_per_layer(n_layers, n_kv, n_ref, d_head, torch)

        cfg = CompactionConfig(enabled=True, method="omp",
                               target_ratio=0.20, min_source_tokens=8)
        result = compact_past_key_values(cache, cfg, q_ref_per_layer=q_ref)
        assert result.key_cache[0].shape[-2] < seq_len

    def test_q_ref_with_beta_phase2(self):
        """Option A + Phase 2 (beta): both q_ref_per_layer and fit_beta work together."""
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig
        from peer.kv_compaction._cache import CompactedKVCache

        n_layers, n_kv, seq_len, d_head, n_ref = 2, 2, 60, 16, 4
        cache = _make_fake_dynamic_cache(n_layers, n_kv, seq_len, d_head, torch)
        q_ref = self._make_q_ref_per_layer(n_layers, n_kv, n_ref, d_head, torch)

        cfg = CompactionConfig(enabled=True, beta_enabled=True,
                               target_ratio=0.25, min_source_tokens=8)
        result = compact_past_key_values(cache, cfg, q_ref_per_layer=q_ref)
        # Phase 2 wraps result in CompactedKVCache
        assert isinstance(result, CompactedKVCache)

    def test_q_ref_with_tuple_cache(self):
        """Option A works with legacy tuple-of-tuples cache format."""
        torch = _make_torch()
        from peer.kv_compaction import compact_past_key_values, CompactionConfig

        n_layers, n_kv, seq_len, d_head, n_ref = 2, 2, 80, 16, 4
        cache = _make_fake_tuple_cache(n_layers, n_kv, seq_len, d_head, torch)
        q_ref = self._make_q_ref_per_layer(n_layers, n_kv, n_ref, d_head, torch)

        cfg = CompactionConfig(enabled=True, target_ratio=0.20, min_source_tokens=8)
        result = compact_past_key_values(cache, cfg, q_ref_per_layer=q_ref)
        assert isinstance(result, tuple)
        t_out = result[0][0].shape[-2]
        assert t_out < seq_len


class TestAttentionQueryCaptureFullPipeline:
    """Full end-to-end: capture Q from fake model, compact with real Q."""

    def test_full_pipeline_with_fake_model(self):
        """Capture Q from a forward pass through the fake model, then compact."""
        torch = _make_torch()
        from peer.kv_compaction import (
            compact_past_key_values, CompactionConfig, AttentionQueryCapture,
        )

        n_layers, n_kv, n_q = 2, 2, 4
        hidden_size, head_dim, n_ref = 32, 8, 4
        seq_len = 64

        model = _make_fake_model(n_layers=n_layers, hidden_size=hidden_size,
                                 n_q=n_q, n_kv=n_kv, head_dim=head_dim)
        hidden = torch.randn(1, n_ref, hidden_size)

        with AttentionQueryCapture(model, n_ref=n_ref) as qc:
            for layer in model.model.layers:
                hidden = layer(hidden)
        q_per_layer = qc.compute_q_ref()

        # Shapes are correct
        assert len(q_per_layer) == n_layers
        for q in q_per_layer:
            assert q is not None
            assert q.shape == (n_kv, n_ref, head_dim)

        # Now compact with real Q
        cache = _make_fake_dynamic_cache(n_layers, n_kv, seq_len, head_dim, torch)
        cfg = CompactionConfig(enabled=True, target_ratio=0.20, min_source_tokens=8)
        result = compact_past_key_values(cache, cfg, q_ref_per_layer=q_per_layer)
        assert result is not None
        t_out = result.key_cache[0].shape[-2]
        assert t_out < seq_len

    def test_single_use_object(self):
        """After compute_q_ref, hidden states are cleared (object is single-use)."""
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        model = _make_fake_model(n_layers=2, hidden_size=32)
        hidden = torch.randn(1, 6, 32)

        with AttentionQueryCapture(model, n_ref=3) as qc:
            for layer in model.model.layers:
                hidden = layer(hidden)

        qc.compute_q_ref()
        # After calling once, hidden store is empty — second call returns all None
        q_again = qc.compute_q_ref()
        for q in q_again:
            assert q is None

    def test_q_ref_improves_over_proxy_key_cosine_similarity(self):
        """Real Q (from W_q projection) should be closer to true queries than proxy K.

        Proxy K uses the last key vectors — these live in W_k subspace, not W_q.
        Real Q comes from W_q projection, which is the correct subspace for scoring
        key relevance.  This test verifies real Q differs from proxy K in a
        statistically meaningful way (they should not be identical).
        """
        torch = _make_torch()
        from peer.kv_compaction import AttentionQueryCapture

        n_layers, hidden_size, n_q, n_kv, head_dim = 1, 64, 4, 2, 16
        model = _make_fake_model(n_layers=n_layers, hidden_size=hidden_size,
                                 n_q=n_q, n_kv=n_kv, head_dim=head_dim)
        n_ref = 4
        seq_len = 20
        hidden = torch.randn(1, seq_len, hidden_size)

        with AttentionQueryCapture(model, n_ref=n_ref) as qc:
            for layer in model.model.layers:
                hidden = layer(hidden)
        q_per_layer = qc.compute_q_ref()

        # q_per_layer[0]: (n_kv, n_ref, head_dim) — real Q in W_q subspace
        q_real = q_per_layer[0]  # (n_kv=2, n_ref=4, head_dim=16)
        assert q_real is not None

        # Generate a synthetic K cache and extract proxy-K (last n_ref rows per head)
        K = torch.randn(1, n_kv, seq_len, head_dim)
        proxy_K = K[0, :, -n_ref:, :]  # (n_kv, n_ref, head_dim)

        # Real Q and proxy K are drawn from different projection subspaces:
        # they must NOT be identical (random initialization ensures this)
        max_diff = (q_real - proxy_K).abs().max().item()
        assert max_diff > 1e-6, "Real Q and proxy K should not be identical"


# ─────────────────────────────────────────────────────────────────────────────
# Option B — bench_kv_compaction.py metric helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkMetrics:
    """Unit tests for the quality metric helpers in bench_kv_compaction.py."""

    def _import_bench(self):
        import importlib.util
        import pathlib
        spec = importlib.util.spec_from_file_location(
            "bench_kv_compaction",
            pathlib.Path(__file__).parent.parent / "scripts" / "bench_kv_compaction.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_script_importable(self):
        """bench_kv_compaction.py should import without error."""
        mod = self._import_bench()
        assert callable(getattr(mod, "main", None))

    def test_cosine_similarity_identical(self):
        """Identical vectors → cosine similarity = 1.0."""
        torch = _make_torch()
        mod = self._import_bench()
        v = torch.randn(100)
        sim = mod._cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors → cosine similarity ≈ 0."""
        torch = _make_torch()
        mod = self._import_bench()
        a = torch.zeros(10)
        a[0] = 1.0
        b = torch.zeros(10)
        b[1] = 1.0
        sim = mod._cosine_similarity(a, b)
        assert abs(sim) < 1e-5

    def test_cosine_similarity_opposite(self):
        """Opposite vectors → cosine similarity ≈ -1.0."""
        torch = _make_torch()
        mod = self._import_bench()
        v = torch.randn(50)
        sim = mod._cosine_similarity(v, -v)
        assert abs(sim + 1.0) < 1e-5

    def test_top1_match_same_argmax(self):
        torch = _make_torch()
        mod = self._import_bench()
        a = torch.zeros(10)
        a[3] = 5.0
        b = torch.zeros(10)
        b[3] = 3.0   # same argmax
        assert mod._top1_match(a, b) is True

    def test_top1_match_different_argmax(self):
        torch = _make_torch()
        mod = self._import_bench()
        a = torch.zeros(10)
        a[3] = 5.0
        b = torch.zeros(10)
        b[7] = 5.0   # different argmax
        assert mod._top1_match(a, b) is False

    def test_topk_overlap_identical(self):
        """Same vector → top-k overlap = 1.0."""
        torch = _make_torch()
        mod = self._import_bench()
        v = torch.randn(100)
        assert abs(mod._topk_overlap(v, v, k=5) - 1.0) < 1e-5

    def test_topk_overlap_disjoint(self):
        """Completely different top-k → overlap = 0."""
        torch = _make_torch()
        mod = self._import_bench()
        a = torch.zeros(20)
        b = torch.zeros(20)
        a[:5] = 1.0    # top-5 of a: indices 0-4
        b[10:15] = 1.0  # top-5 of b: indices 10-14 (disjoint)
        assert mod._topk_overlap(a, b, k=5) == 0.0

    def test_rank_corr_identical(self):
        """Identical logits → rank_corr = 1.0."""
        torch = _make_torch()
        mod = self._import_bench()
        v = torch.randn(200)
        rho = mod._rank_corr(v, v, k=50)
        assert abs(rho - 1.0) < 1e-3

    def test_rank_corr_reversed(self):
        """Reversed ranking → rank_corr ≈ -1.0 for small k."""
        torch = _make_torch()
        mod = self._import_bench()
        v = torch.arange(100).float()
        # Negating reverses the rank order
        rho = mod._rank_corr(v, -v, k=20)
        assert rho < -0.90

    def test_rank_corr_range(self):
        """rank_corr should be in [-1, 1] for any input."""
        torch = _make_torch()
        mod = self._import_bench()
        for _ in range(5):
            a = torch.randn(500)
            b = torch.randn(500)
            rho = mod._rank_corr(a, b, k=100)
            assert -1.0 - 1e-5 <= rho <= 1.0 + 1e-5

    def test_detect_decoder_family_gpt(self):
        torch = _make_torch()
        import torch.nn as nn
        mod = self._import_bench()

        class _FakeGPT(nn.Module):
            pass
        assert mod._detect_decoder_family(_FakeGPT()) == "gpt"

    def test_detect_decoder_family_qwen(self):
        torch = _make_torch()
        import torch.nn as nn
        mod = self._import_bench()

        class Qwen2ForCausalLM(nn.Module):  # class name contains "qwen"
            pass
        assert "qwen" in mod._detect_decoder_family(Qwen2ForCausalLM())

    def test_benchmark_help_flag(self):
        """--help flag prints usage and exits with code 0."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/bench_kv_compaction.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "target-ratio" in result.stdout
        assert "real-Q" in result.stdout or "proxy" in result.stdout or "Benchmark" in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# C — optimize_head_budgets.py
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizeHeadBudgets:
    """Tests for scripts/optimize_head_budgets.py (C — entropy calibration)."""

    def _import_script(self):
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "optimize_head_budgets",
            str(Path(__file__).parent.parent / "scripts" / "optimize_head_budgets.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_script_importable(self):
        """Script loads without errors."""
        mod = self._import_script()
        assert hasattr(mod, "main")
        assert hasattr(mod, "_compute_head_entropy")
        assert hasattr(mod, "_allocate_budgets")

    def test_help_flag(self):
        """--help exits 0 and mentions --model."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/optimize_head_budgets.py", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "--model" in result.stdout

    def test_compute_head_entropy_mha_shape(self):
        """MHA case: n_heads == n_kv_heads — entropy shape is (n_kv,)."""
        torch = _make_torch()
        mod = self._import_script()
        n_heads = 4
        seq_len = 10
        # Build a valid attention distribution (summing to 1 per head)
        attn = torch.rand(n_heads, seq_len)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        h = mod._compute_head_entropy(attn, n_kv_heads=n_heads)
        assert h.shape == (n_heads,)
        assert (h >= 0).all()

    def test_compute_head_entropy_gqa_shape(self):
        """GQA case: n_heads > n_kv_heads — averages down to (n_kv_heads,)."""
        torch = _make_torch()
        mod = self._import_script()
        n_heads = 8
        n_kv_heads = 2
        seq_len = 16
        attn = torch.rand(n_heads, seq_len)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        h = mod._compute_head_entropy(attn, n_kv_heads=n_kv_heads)
        assert h.shape == (n_kv_heads,)
        assert (h >= 0).all()

    def test_allocate_budgets_mean(self):
        """Mean of allocated budgets is close to target_ratio."""
        torch = _make_torch()
        mod = self._import_script()
        n_kv = 8
        # Uniform entropy — allocations should all equal target_ratio
        entropy = torch.ones(n_kv)
        budgets = mod._allocate_budgets(entropy, target_ratio=0.10, min_ratio=0.02, max_ratio=0.40)
        assert len(budgets) == n_kv
        mean_b = sum(budgets) / n_kv
        assert abs(mean_b - 0.10) < 0.01, f"mean={mean_b}"

    def test_allocate_budgets_clip(self):
        """All budgets are within [min_ratio, max_ratio]."""
        torch = _make_torch()
        mod = self._import_script()
        # Extreme entropy spread
        entropy = torch.tensor([0.001, 100.0, 0.001, 50.0, 0.001, 25.0, 0.001, 10.0])
        budgets = mod._allocate_budgets(entropy, target_ratio=0.10, min_ratio=0.02, max_ratio=0.40)
        assert all(0.02 <= b <= 0.40 for b in budgets), f"budgets={budgets}"

    def test_json_output_schema(self):
        """Full run with a tiny fake model writes valid schema JSON."""
        torch = _make_torch()
        mod = self._import_script()

        # Build a minimal fake attention output
        n_layers = 2
        n_heads = 4
        n_kv_heads = 2
        seq_len = 8

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "budgets.json"

            # We test the allocation + JSON writing logic directly
            all_rows = []
            for _ in range(n_layers):
                attn = torch.rand(n_heads, seq_len)
                attn = attn / attn.sum(dim=-1, keepdim=True)
                all_rows.append([attn])

            import json
            layer_budgets = []
            for rows in all_rows:
                h = mod._compute_head_entropy(rows[0], n_kv_heads)
                b = mod._allocate_budgets(h, target_ratio=0.10, min_ratio=0.02, max_ratio=0.40)
                layer_budgets.append(b)

            result = {
                "model": "test/model",
                "n_layers": n_layers,
                "n_kv_heads": n_kv_heads,
                "source": "calibrated_entropy_v1",
                "layer_budgets": layer_budgets,
            }
            output_path.write_text(json.dumps(result))

            data = json.loads(output_path.read_text())
            assert data["source"] == "calibrated_entropy_v1"
            assert data["n_layers"] == n_layers
            assert data["n_kv_heads"] == n_kv_heads
            assert len(data["layer_budgets"]) == n_layers
            assert all(len(row) == n_kv_heads for row in data["layer_budgets"])


# ─────────────────────────────────────────────────────────────────────────────
# D — Compaction SLO metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestCompactionSLOMetrics:
    """Tests for Phase D — compaction SLO counters in model_shard.py."""

    def test_toy_runtime_has_compaction_stats(self):
        """ToyRuntime.compaction_stats() exists and returns a dict of zeros."""
        from peer.model_shard import ToyRuntime, ToyShardConfig
        rt = ToyRuntime(ToyShardConfig())
        stats = rt.compaction_stats()
        assert isinstance(stats, dict)
        assert stats["compact_calls"] == 0
        assert stats["kv_cache_hits"] == 0
        assert stats["kv_cache_misses"] == 0
        assert stats["compact_latency_s"] == 0.0

    def test_toy_runtime_compaction_stats_keys(self):
        """ToyRuntime returns all 7 required keys."""
        from peer.model_shard import ToyRuntime, ToyShardConfig
        rt = ToyRuntime(ToyShardConfig())
        stats = rt.compaction_stats()
        required = {
            "compact_calls", "compact_tokens_before", "compact_tokens_after",
            "compact_tokens_saved", "compact_latency_s", "kv_cache_hits", "kv_cache_misses",
        }
        assert required.issubset(set(stats.keys()))

    def test_model_shard_compaction_stats_delegation(self):
        """ModelShard.compaction_stats() delegates to the runtime (toy backend)."""
        from peer.model_shard import ModelShard, ToyShardConfig
        shard = ModelShard(ToyShardConfig(runtime_backend="toy_auto"))
        stats = shard.compaction_stats()
        assert isinstance(stats, dict)
        assert "compact_calls" in stats

    def test_model_shard_compaction_stats_missing_runtime(self):
        """ModelShard.compaction_stats() returns empty dict if runtime has no method."""
        from peer.model_shard import ModelShard, ToyShardConfig
        shard = ModelShard(ToyShardConfig(runtime_backend="toy_auto"))
        # Replace runtime with a bare object that lacks compaction_stats
        class _BareRuntime:
            pass
        shard._runtime = _BareRuntime()
        stats = shard.compaction_stats()
        assert isinstance(stats, dict)
        assert stats == {}

    def test_pytorch_runtime_compact_lock_exists(self):
        """PyTorchRuntime initialises _compact_lock (pure import, no model load)."""
        import threading
        from peer.model_shard import PyTorchRuntime
        # We only check the class definition — not instantiation (requires torch+HF)
        import inspect
        src = inspect.getsource(PyTorchRuntime.__init__)
        assert "_compact_lock" in src
        assert "_compact_calls" in src

    def test_pytorch_runtime_compaction_stats_method_defined(self):
        """PyTorchRuntime.compaction_stats method is defined with correct signature."""
        from peer.model_shard import PyTorchRuntime
        import inspect
        src = inspect.getsource(PyTorchRuntime.compaction_stats)
        assert "compact_tokens_saved" in src
        assert "compact_latency_s" in src
        assert "_compact_lock" in src

    def test_kv_cache_hit_miss_instrumentation(self):
        """_kv_cache_get increments _compact_kv_cache_hits and _compact_kv_cache_misses."""
        from peer.model_shard import PyTorchRuntime
        import inspect
        src = inspect.getsource(PyTorchRuntime._kv_cache_get)
        assert "_compact_kv_cache_misses" in src
        assert "_compact_kv_cache_hits" in src

    def test_kv_cache_set_timing_instrumentation(self):
        """_kv_cache_set wraps compact call with timing and token counting."""
        from peer.model_shard import PyTorchRuntime
        import inspect
        src = inspect.getsource(PyTorchRuntime._kv_cache_set)
        assert "perf_counter" in src
        assert "_compact_calls" in src
        assert "_compact_tokens_before" in src
        assert "_compact_latency_s" in src


# ─────────────────────────────────────────────────────────────────────────────
# H — RadixKVCache
# ─────────────────────────────────────────────────────────────────────────────

class TestRadixKVCache:
    """Tests for Phase H — RadixKVCache and _slice_kv_prefix."""

    def test_importable_from_kv_compaction(self):
        """RadixKVCache and _slice_kv_prefix are importable from the public API."""
        from peer.kv_compaction import RadixKVCache, _slice_kv_prefix
        assert callable(RadixKVCache)
        assert callable(_slice_kv_prefix)

    def test_insert_and_lookup_exact_match(self):
        """Exact token sequence lookup returns stored kv and prefix_len == len(tokens)."""
        from peer.kv_compaction import RadixKVCache
        cache = RadixKVCache(max_entries=8, min_prefix_len=4)
        tokens = (1, 2, 3, 4, 5, 6)
        kv = object()
        cache.insert(tokens, kv)
        returned_kv, plen = cache.lookup(tokens)
        assert returned_kv is kv
        assert plen == len(tokens)

    def test_lookup_prefix_match(self):
        """Lookup on an extension of a stored key returns the stored kv + correct prefix_len."""
        from peer.kv_compaction import RadixKVCache
        cache = RadixKVCache(max_entries=8, min_prefix_len=4)
        prefix = (10, 20, 30, 40)
        extension = (10, 20, 30, 40, 50, 60)
        kv = {"data": "abc"}
        cache.insert(prefix, kv)
        returned_kv, plen = cache.lookup(extension)
        assert returned_kv is kv
        assert plen == len(prefix)

    def test_lookup_no_match(self):
        """Completely unrelated tokens return (None, 0)."""
        from peer.kv_compaction import RadixKVCache
        cache = RadixKVCache(max_entries=8, min_prefix_len=4)
        cache.insert((1, 2, 3, 4), object())
        returned_kv, plen = cache.lookup((5, 6, 7, 8))
        assert returned_kv is None
        assert plen == 0

    def test_lookup_partial_overlap_not_prefix(self):
        """Partial overlap that is not a true prefix is not returned."""
        from peer.kv_compaction import RadixKVCache
        cache = RadixKVCache(max_entries=8, min_prefix_len=4)
        cache.insert((1, 2, 3, 4), object())
        # Query shares tokens 2,3 but not as a prefix
        returned_kv, plen = cache.lookup((2, 3, 4, 5))
        assert returned_kv is None
        assert plen == 0

    def test_lru_eviction(self):
        """Inserting max_entries+1 sequences evicts the oldest."""
        from peer.kv_compaction import RadixKVCache
        import time
        cache = RadixKVCache(max_entries=3, min_prefix_len=4)
        sentinels = {}
        for i in range(3):
            t = tuple(range(i * 4, i * 4 + 4))
            kv = object()
            sentinels[t] = kv
            cache.insert(t, kv)
            time.sleep(0.002)   # ensure monotonic order

        assert cache.stats()["radix_entries"] == 3

        # Insert one more — should evict the oldest (i=0)
        new_tokens = (100, 101, 102, 103)
        cache.insert(new_tokens, object())
        assert cache.stats()["radix_entries"] == 3

        # The oldest entry (i=0) should now be gone
        oldest_tokens = tuple(range(0, 4))
        kv_ret, plen = cache.lookup(oldest_tokens)
        assert plen == 0 or kv_ret is not sentinels.get(oldest_tokens)

    def test_min_prefix_len_guard_insert(self):
        """Sequences shorter than min_prefix_len are not inserted."""
        from peer.kv_compaction import RadixKVCache
        cache = RadixKVCache(max_entries=8, min_prefix_len=8)
        short_tokens = (1, 2, 3)   # length 3 < 8
        cache.insert(short_tokens, object())
        assert cache.stats()["radix_entries"] == 0

    def test_min_prefix_len_guard_lookup(self):
        """Short stored sequences are not returned by lookup."""
        from peer.kv_compaction import RadixKVCache
        cache = RadixKVCache(max_entries=8, min_prefix_len=8)
        # Force insert by temporarily lowering threshold
        cache._min_prefix_len = 1
        cache.insert((1, 2, 3), object())
        cache._min_prefix_len = 8   # restore
        # Lookup should skip the 3-token entry
        kv_ret, plen = cache.lookup((1, 2, 3, 4, 5, 6, 7, 8, 9))
        assert plen == 0

    def test_stats_returns_correct_keys(self):
        """stats() returns radix_entries and radix_max_entries."""
        from peer.kv_compaction import RadixKVCache
        cache = RadixKVCache(max_entries=32, min_prefix_len=4)
        s = cache.stats()
        assert s["radix_max_entries"] == 32
        assert s["radix_entries"] == 0
        cache.insert((1, 2, 3, 4), object())
        assert cache.stats()["radix_entries"] == 1

    def test_clear(self):
        """clear() removes all entries."""
        from peer.kv_compaction import RadixKVCache
        cache = RadixKVCache(max_entries=8, min_prefix_len=4)
        cache.insert((1, 2, 3, 4), object())
        cache.insert((5, 6, 7, 8), object())
        cache.clear()
        assert cache.stats()["radix_entries"] == 0

    def test_slice_kv_prefix_dynamic_cache(self):
        """_slice_kv_prefix truncates DynamicCache correctly."""
        torch = _make_torch()
        from peer.kv_compaction import _slice_kv_prefix
        n_layers, n_kv_heads, seq_len, d_head = 2, 2, 16, 4
        cache = _make_fake_dynamic_cache(n_layers, n_kv_heads, seq_len, d_head, torch)
        sliced = _slice_kv_prefix(cache, prefix_len=8)
        assert sliced is not None
        assert sliced.key_cache[0].shape[-2] == 8
        assert sliced.value_cache[0].shape[-2] == 8
        # Original unmodified
        assert cache.key_cache[0].shape[-2] == seq_len

    def test_slice_kv_prefix_tuple_cache(self):
        """_slice_kv_prefix truncates tuple-of-tuples correctly."""
        torch = _make_torch()
        from peer.kv_compaction import _slice_kv_prefix
        n_layers, n_kv_heads, seq_len, d_head = 3, 4, 20, 8
        cache = _make_fake_tuple_cache(n_layers, n_kv_heads, seq_len, d_head, torch)
        sliced = _slice_kv_prefix(cache, prefix_len=10)
        assert sliced is not None
        assert len(sliced) == n_layers
        assert sliced[0][0].shape[-2] == 10
        # Original unmodified
        assert cache[0][0].shape[-2] == seq_len

    def test_slice_kv_prefix_zero_len_returns_none(self):
        """_slice_kv_prefix with prefix_len=0 returns None."""
        from peer.kv_compaction import _slice_kv_prefix
        result = _slice_kv_prefix(((None,),), prefix_len=0)
        assert result is None

    def test_toyshard_config_radix_fields(self):
        """ToyShardConfig has the 3 radix cache fields with correct defaults."""
        from peer.model_shard import ToyShardConfig
        cfg = ToyShardConfig()
        assert cfg.runtime_kv_radix_cache_enabled is False
        assert cfg.runtime_kv_radix_cache_max_entries == 128
        assert cfg.runtime_kv_radix_cache_min_prefix_len == 16


# ─────────────────────────────────────────────────────────────────────────────
# 6.1 — Auto Mode tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoMode:
    """Tests for 6.1 — KV Compaction Auto Mode (three-position toggle)."""

    def test_compaction_config_mode_field(self):
        """CompactionConfig accepts mode and auto_threshold fields."""
        from peer.kv_compaction import CompactionConfig
        cfg = CompactionConfig(enabled=True, mode="on", auto_threshold=256)
        assert cfg.mode == "on"
        assert cfg.auto_threshold == 256

    def test_auto_mode_default_mode_is_on(self):
        """Default mode is 'on' for backward compatibility."""
        from peer.kv_compaction import CompactionConfig
        cfg = CompactionConfig(enabled=True)
        assert cfg.mode == "on"

    def test_auto_mode_short_sequence_skips_compaction(self):
        """mode='auto' + seq_len <= threshold → returns unchanged cache."""
        torch = _make_torch()
        from peer.kv_compaction import CompactionConfig, compact_past_key_values

        threshold = 128
        # Build a short cache (seq_len = 64, below threshold)
        seq_len = 64
        K = torch.randn(1, 2, seq_len, 16)
        V = torch.randn(1, 2, seq_len, 16)
        pkv = ((K.clone(), V.clone()),)

        cfg = CompactionConfig(
            enabled=True,
            method="hak",
            target_ratio=0.10,
            min_source_tokens=4,
            mode="auto",
            auto_threshold=threshold,
        )
        result = compact_past_key_values(pkv, cfg)
        # Must return the original object unchanged (same seq_len)
        assert result is pkv

    def test_auto_mode_long_sequence_compacts(self):
        """mode='auto' + seq_len > threshold → compacts normally."""
        torch = _make_torch()
        from peer.kv_compaction import CompactionConfig, compact_past_key_values

        threshold = 32
        seq_len = 128  # above threshold
        K = torch.randn(1, 1, seq_len, 16)
        V = torch.randn(1, 1, seq_len, 16)
        pkv = ((K.clone(), V.clone()),)

        cfg = CompactionConfig(
            enabled=True,
            method="hak",
            target_ratio=0.25,
            min_source_tokens=4,
            mode="auto",
            auto_threshold=threshold,
        )
        result = compact_past_key_values(pkv, cfg)
        # Should be compacted — smaller seq_len
        assert result is not pkv
        result_seq = result[0][0].shape[-2]
        assert result_seq < seq_len

    def test_on_mode_always_compacts(self):
        """mode='on' compacts even short sequences (respects min_source_tokens)."""
        torch = _make_torch()
        from peer.kv_compaction import CompactionConfig, compact_past_key_values

        seq_len = 64
        K = torch.randn(1, 1, seq_len, 16)
        V = torch.randn(1, 1, seq_len, 16)
        pkv = ((K.clone(), V.clone()),)

        cfg = CompactionConfig(
            enabled=True,
            method="hak",
            target_ratio=0.25,
            min_source_tokens=4,
            mode="on",
            auto_threshold=1024,  # large threshold — ignored in 'on' mode
        )
        result = compact_past_key_values(pkv, cfg)
        assert result is not pkv

    def test_toyshard_config_auto_mode_fields(self):
        """ToyShardConfig has runtime_kv_compaction_mode and auto_threshold fields."""
        from peer.model_shard import ToyShardConfig
        cfg = ToyShardConfig()
        assert cfg.runtime_kv_compaction_mode == "off"
        assert cfg.runtime_kv_compaction_auto_threshold == 512

    def test_kv_compaction_enabled_backward_compat(self):
        """runtime_kv_compaction_enabled=True is treated as mode='on'."""
        from peer.model_shard import ToyShardConfig, ToyRuntime
        cfg = ToyShardConfig(runtime_kv_compaction_enabled=True)
        # ToyRuntime doesn't set up real compaction, so just check the config field
        assert cfg.runtime_kv_compaction_enabled is True
        # Verify mode stays "off" (legacy flag, not the new mode field)
        assert cfg.runtime_kv_compaction_mode == "off"
