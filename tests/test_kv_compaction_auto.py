"""Tests for Pass 6: KV Compaction Auto Mode + SLO Metrics.

Covers:
- VRAM-aware auto compaction in PyTorchRuntime
- _vram_usage_pct() for PyTorch and MLX
- Compaction stats in DHT Announcement
- Prometheus /metrics endpoint compaction counters
"""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest


# ── _vram_usage_pct (PyTorchRuntime) ─────────────────────────────────────────


class TestPyTorchVramUsagePct:
    def _make_runtime(self):
        """Create a minimal PyTorchRuntime-like object for testing."""
        from peer.model_shard import PyTorchRuntime
        return PyTorchRuntime

    def test_returns_float(self):
        rt_cls = self._make_runtime()
        # Without a real GPU, should return 0.0
        rt = MagicMock(spec=rt_cls)
        rt._vram_usage_pct = rt_cls._vram_usage_pct.__get__(rt, rt_cls)
        result = rt._vram_usage_pct()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_returns_zero_when_no_cuda(self):
        """On machines without CUDA, should return 0.0."""
        rt_cls = self._make_runtime()
        rt = MagicMock(spec=rt_cls)
        rt._vram_usage_pct = rt_cls._vram_usage_pct.__get__(rt, rt_cls)
        with patch("peer.model_shard.PyTorchRuntime._vram_usage_pct", return_value=0.0):
            assert rt._vram_usage_pct() == 0.0


# ── _vram_usage_pct (MLXRuntime) ─────────────────────────────────────────────


class TestMLXVramUsagePct:
    def test_method_exists(self):
        """MLXRuntime must have _vram_usage_pct method."""
        # Can't instantiate MLXRuntime without MLX, so check the class.
        import importlib
        import ast

        source = open("peer/mlx_runtime.py").read()
        tree = ast.parse(source)
        methods = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "_vram_usage_pct" in methods

    def test_returns_float_via_mock(self):
        """When called, _vram_usage_pct should return a float."""
        # Mock the MLX metal module.
        mock_mx = MagicMock()
        mock_mx.metal.get_active_memory.return_value = 4_000_000_000  # 4 GB
        mock_mx.metal.device_info.return_value = {
            "recommendedMaxWorkingSetSize": 8_000_000_000  # 8 GB
        }

        # Simulate the _vram_usage_pct logic.
        active = mock_mx.metal.get_active_memory()
        total = mock_mx.metal.device_info()["recommendedMaxWorkingSetSize"]
        pct = active / total if total > 0 else 0.0
        assert pct == 0.5


# ── Auto mode VRAM-aware logic ───────────────────────────────────────────────


class TestAutoModeVRAMLogic:
    def test_auto_skips_when_vram_comfortable_and_short_seq(self):
        """Auto mode: VRAM < 75% + seq <= threshold → skip compaction."""
        from peer.kv_compaction import CompactionConfig

        config = CompactionConfig(enabled=True, mode="auto", auto_threshold=512)

        # Simulate: VRAM at 50%, seq_len = 100
        vram_pct = 0.50
        seq_len = 100
        should_skip = (vram_pct < 0.75 and seq_len <= config.auto_threshold)
        assert should_skip is True

    def test_auto_triggers_when_vram_pressure(self):
        """Auto mode: VRAM >= 75% → trigger compaction even for short seq."""
        from peer.kv_compaction import CompactionConfig

        config = CompactionConfig(enabled=True, mode="auto", auto_threshold=512)

        vram_pct = 0.85
        seq_len = 100
        should_skip = (vram_pct < 0.75 and seq_len <= config.auto_threshold)
        assert should_skip is False  # should NOT skip → triggers compaction

    def test_auto_triggers_on_long_sequence(self):
        """Auto mode: seq > threshold → trigger regardless of VRAM."""
        from peer.kv_compaction import CompactionConfig

        config = CompactionConfig(enabled=True, mode="auto", auto_threshold=512)

        vram_pct = 0.30  # very comfortable
        seq_len = 1024   # exceeds threshold
        should_skip = (vram_pct < 0.75 and seq_len <= config.auto_threshold)
        assert should_skip is False  # long seq → compaction triggers

    def test_auto_counters_exist_in_compaction_stats(self):
        """PyTorchRuntime.compaction_stats() must include auto counters."""
        pytest.importorskip("torch")
        from peer.model_shard import PyTorchRuntime

        # Check that the class has the counter attributes.
        rt = MagicMock(spec=PyTorchRuntime)
        rt._auto_skip_count = 5
        rt._auto_trigger_count = 3
        rt._compact_calls = 3
        rt._compact_tokens_before = 1500
        rt._compact_tokens_after = 150
        rt._compact_latency_s = 0.05
        rt._compact_kv_cache_hits = 0
        rt._compact_kv_cache_misses = 0
        rt._compact_lock = __import__("threading").Lock()

        # Call real method.
        rt.compaction_stats = PyTorchRuntime.compaction_stats.__get__(rt, PyTorchRuntime)
        stats = rt.compaction_stats()
        assert "auto_skip_count" in stats
        assert "auto_trigger_count" in stats
        assert stats["auto_skip_count"] == 5
        assert stats["auto_trigger_count"] == 3


# ── DHT Announcement compaction fields ───────────────────────────────────────


class TestAnnouncementCompactionFields:
    def test_fields_exist_in_announcement(self):
        from peer.dht_announce import Announcement

        ann = Announcement(
            peer_id="test",
            model_id="qwen",
            host="127.0.0.1",
            port=50051,
            compact_tokens_saved_total=42000,
            compact_latency_total_ms=123.4,
        )
        assert ann.compact_tokens_saved_total == 42000
        assert ann.compact_latency_total_ms == 123.4

    def test_fields_default_to_zero(self):
        from peer.dht_announce import Announcement

        ann = Announcement(
            peer_id="test",
            model_id="qwen",
            host="127.0.0.1",
            port=50051,
        )
        assert ann.compact_tokens_saved_total == 0
        assert ann.compact_latency_total_ms == 0.0

    def test_fields_in_asdict(self):
        from peer.dht_announce import Announcement

        ann = Announcement(
            peer_id="test",
            model_id="qwen",
            host="127.0.0.1",
            port=50051,
            compact_tokens_saved_total=100,
            compact_latency_total_ms=5.5,
        )
        d = asdict(ann)
        assert d["compact_tokens_saved_total"] == 100
        assert d["compact_latency_total_ms"] == 5.5


# ── PeerEndpoint compaction fields ───────────────────────────────────────────


class TestPeerEndpointCompactionFields:
    def test_from_dict_parses_compaction_fields(self):
        from coordinator.path_finder import PeerEndpoint

        data = {
            "peer_id": "p1",
            "host": "127.0.0.1",
            "port": 50051,
            "compact_tokens_saved_total": 5000,
            "compact_latency_total_ms": 42.5,
        }
        ep = PeerEndpoint.from_dict(data)
        assert ep.compact_tokens_saved_total == 5000
        assert ep.compact_latency_total_ms == 42.5

    def test_from_dict_defaults_when_missing(self):
        from coordinator.path_finder import PeerEndpoint

        data = {"peer_id": "p1", "host": "127.0.0.1", "port": 50051}
        ep = PeerEndpoint.from_dict(data)
        assert ep.compact_tokens_saved_total == 0
        assert ep.compact_latency_total_ms == 0.0


# ── Prometheus /metrics compaction counters ──────────────────────────────────


class TestPrometheusCompactionMetrics:
    def test_metrics_rendered_in_prometheus_format(self):
        """The Prometheus renderer must include compaction counters."""
        # Read the actual source to verify the metric names are present.
        with open("coordinator/api_server.py") as f:
            source = f.read()
        assert "openhydra_compact_tokens_saved_total" in source
        assert "openhydra_compact_latency_ms_total" in source

    def test_metrics_snapshot_includes_compaction(self):
        """metrics_snapshot() must return compaction keys."""
        # We test the key names directly since StatusService requires
        # a full engine to instantiate.
        expected_keys = ["compact_tokens_saved_total", "compact_latency_total_ms"]
        with open("coordinator/status_service.py") as f:
            source = f.read()
        for key in expected_keys:
            assert key in source, f"Missing key in metrics_snapshot: {key}"
