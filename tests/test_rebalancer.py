"""Tests for coordinator.rebalancer — dynamic layer rebalancing.

Groups:
    A — RebalanceDirective (5 tests): serialization, expiry, from_dict.
    B — LayerRebalancer.compute_directives (7 tests): gap detection, candidate
        selection, adjacency preference, VRAM/load filtering, MLX exclusion.
    C — DHT endpoint integration (4 tests): POST/GET /rebalance round-trip,
        expiry pruning, missing fields.
    D — ModelShard.reshard (3 tests): ToyRuntime reshard, profile update,
        invalid range.
"""
from __future__ import annotations

import json
import time
import threading
import unittest
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from typing import Any
from unittest.mock import patch

from coordinator.layer_coverage import LayerCoverageMap, LayerRange
from coordinator.rebalancer import (
    LayerRebalancer,
    RebalanceDirective,
    poll_directives_from_dht,
    publish_directives_to_dht,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class _FakePeer:
    peer_id: str
    layer_start: int = 0
    layer_end: int = 0
    total_layers: int = 32
    available_vram_mb: int = 4096
    runtime_backend: str = "pytorch_cpu"
    host: str = "127.0.0.1"
    port: int = 50051


@dataclass
class _FakeHealth:
    peer: _FakePeer
    load_pct: float = 10.0
    healthy: bool = True
    latency_ms: float = 5.0


# ═════════════════════════════════════════════════════════════════════════════
# Group A — RebalanceDirective
# ═════════════════════════════════════════════════════════════════════════════


class TestRebalanceDirective(unittest.TestCase):
    """Group A: RebalanceDirective data class."""

    def test_to_dict_roundtrip(self):
        now = int(time.time() * 1000)
        d = RebalanceDirective(
            target_peer_id="peer-01",
            new_layer_start=0,
            new_layer_end=16,
            total_layers=32,
            reason="gap_fill:[8,16)",
            issued_unix_ms=now,
            expires_unix_ms=now + 120_000,
        )
        serialized = d.to_dict()
        restored = RebalanceDirective.from_dict(serialized)
        self.assertEqual(restored.target_peer_id, "peer-01")
        self.assertEqual(restored.new_layer_start, 0)
        self.assertEqual(restored.new_layer_end, 16)
        self.assertEqual(restored.total_layers, 32)
        self.assertEqual(restored.reason, "gap_fill:[8,16)")

    def test_is_expired_false(self):
        future = int(time.time() * 1000) + 120_000
        d = RebalanceDirective(
            target_peer_id="p", new_layer_start=0, new_layer_end=16,
            total_layers=32, expires_unix_ms=future,
        )
        self.assertFalse(d.is_expired)

    def test_is_expired_true(self):
        past = int(time.time() * 1000) - 1000
        d = RebalanceDirective(
            target_peer_id="p", new_layer_start=0, new_layer_end=16,
            total_layers=32, expires_unix_ms=past,
        )
        self.assertTrue(d.is_expired)

    def test_from_dict_defaults(self):
        d = RebalanceDirective.from_dict({})
        self.assertEqual(d.target_peer_id, "")
        self.assertEqual(d.new_layer_start, 0)
        self.assertEqual(d.reason, "gap_fill")

    def test_from_dict_with_json_roundtrip(self):
        original = RebalanceDirective(
            target_peer_id="peer-42",
            new_layer_start=16,
            new_layer_end=32,
            total_layers=32,
        )
        json_str = json.dumps(original.to_dict())
        restored = RebalanceDirective.from_dict(json.loads(json_str))
        self.assertEqual(restored.target_peer_id, "peer-42")
        self.assertEqual(restored.new_layer_start, 16)
        self.assertEqual(restored.new_layer_end, 32)


# ═════════════════════════════════════════════════════════════════════════════
# Group B — LayerRebalancer.compute_directives
# ═════════════════════════════════════════════════════════════════════════════


class TestLayerRebalancerComputeDirectives(unittest.TestCase):
    """Group B: gap detection and directive generation."""

    def test_no_gaps_returns_empty(self):
        """Complete coverage → no directives."""
        ranges = [
            LayerRange("a", 0, 16, 32),
            LayerRange("b", 16, 32, 32),
        ]
        cmap = LayerCoverageMap(ranges, 32)
        rebalancer = LayerRebalancer()
        directives = rebalancer.compute_directives(cmap)
        self.assertEqual(directives, [])

    def test_single_gap_generates_directive(self):
        """Gap in [16, 24) with adjacent peer 'a' ending at 16."""
        ranges = [
            LayerRange("a", 0, 16, 32),
            LayerRange("b", 24, 32, 32),
        ]
        cmap = LayerCoverageMap(ranges, 32)
        health = [
            _FakeHealth(_FakePeer("a", 0, 16, 32)),
            _FakeHealth(_FakePeer("b", 24, 32, 32)),
        ]
        rebalancer = LayerRebalancer()
        directives = rebalancer.compute_directives(cmap, health)
        self.assertEqual(len(directives), 1)
        d = directives[0]
        # Peer 'a' is adjacent (layer_end=16 == gap_start=16), so it should expand.
        self.assertEqual(d.target_peer_id, "a")
        self.assertEqual(d.new_layer_start, 0)
        self.assertEqual(d.new_layer_end, 24)
        self.assertEqual(d.total_layers, 32)

    def test_prefers_adjacent_peer(self):
        """When two candidates exist, prefer the one adjacent to the gap."""
        ranges = [
            LayerRange("a", 0, 8, 32),
            LayerRange("b", 8, 16, 32),
            LayerRange("c", 24, 32, 32),
        ]
        cmap = LayerCoverageMap(ranges, 32)
        health = [
            _FakeHealth(_FakePeer("a", 0, 8, 32)),
            _FakeHealth(_FakePeer("b", 8, 16, 32)),
            _FakeHealth(_FakePeer("c", 24, 32, 32)),
        ]
        rebalancer = LayerRebalancer()
        directives = rebalancer.compute_directives(cmap, health)
        self.assertEqual(len(directives), 1)
        # 'b' ends at 16, 'c' starts at 24 — gap is [16,24).
        # 'b' is directly adjacent (layer_end=16 == gap_start=16).
        self.assertEqual(directives[0].target_peer_id, "b")

    def test_skips_high_load_peers(self):
        """Peers with load > max_load_pct are excluded."""
        ranges = [
            LayerRange("a", 0, 16, 32),
            LayerRange("b", 24, 32, 32),
        ]
        cmap = LayerCoverageMap(ranges, 32)
        health = [
            _FakeHealth(_FakePeer("a", 0, 16, 32), load_pct=80.0),
            _FakeHealth(_FakePeer("b", 24, 32, 32), load_pct=10.0),
        ]
        rebalancer = LayerRebalancer(max_load_pct=50.0)
        directives = rebalancer.compute_directives(cmap, health)
        # 'a' is too loaded, 'b' is adjacent on the right.
        self.assertEqual(len(directives), 1)
        self.assertEqual(directives[0].target_peer_id, "b")

    def test_skips_mlx_peers(self):
        """MLX peers cannot reshard — they run the full model."""
        ranges = [
            LayerRange("a", 0, 16, 32),
            LayerRange("b", 24, 32, 32),
        ]
        cmap = LayerCoverageMap(ranges, 32)
        health = [
            _FakeHealth(_FakePeer("a", 0, 16, 32, runtime_backend="mlx")),
            _FakeHealth(_FakePeer("b", 24, 32, 32, runtime_backend="pytorch_cpu")),
        ]
        rebalancer = LayerRebalancer()
        directives = rebalancer.compute_directives(cmap, health)
        self.assertEqual(len(directives), 1)
        self.assertEqual(directives[0].target_peer_id, "b")

    def test_vram_constraint(self):
        """Peers with insufficient VRAM are excluded when min_vram_mb > 0."""
        ranges = [
            LayerRange("a", 0, 16, 32),
            LayerRange("b", 24, 32, 32),
        ]
        cmap = LayerCoverageMap(ranges, 32)
        health = [
            _FakeHealth(_FakePeer("a", 0, 16, 32, available_vram_mb=128)),
            _FakeHealth(_FakePeer("b", 24, 32, 32, available_vram_mb=4096)),
        ]
        rebalancer = LayerRebalancer(min_vram_mb=256)
        directives = rebalancer.compute_directives(cmap, health)
        self.assertEqual(len(directives), 1)
        self.assertEqual(directives[0].target_peer_id, "b")

    def test_no_candidates_returns_empty(self):
        """All peers are too loaded → no directives generated."""
        ranges = [
            LayerRange("a", 0, 16, 32),
            # Gap at [16, 32)
        ]
        cmap = LayerCoverageMap(ranges, 32)
        health = [
            _FakeHealth(_FakePeer("a", 0, 16, 32), load_pct=90.0),
        ]
        rebalancer = LayerRebalancer(max_load_pct=50.0)
        directives = rebalancer.compute_directives(cmap, health)
        self.assertEqual(directives, [])


# ═════════════════════════════════════════════════════════════════════════════
# Group C — DHT endpoint integration (POST/GET /rebalance)
# ═════════════════════════════════════════════════════════════════════════════


class TestDhtRebalanceEndpoints(unittest.TestCase):
    """Group C: POST/GET /rebalance on DhtBootstrapHandler."""

    @classmethod
    def setUpClass(cls):
        from dht.bootstrap import DhtBootstrapHandler
        from dht.node import InMemoryDhtNode
        # Start a test bootstrap server.
        cls._dht_node = InMemoryDhtNode(ttl_seconds=300)
        DhtBootstrapHandler.dht = cls._dht_node
        DhtBootstrapHandler._layer_rebalance_directives = {}
        cls._server = ThreadingHTTPServer(("127.0.0.1", 0), DhtBootstrapHandler)
        cls._port = cls._server.server_address[1]
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()
        cls._base_url = f"http://127.0.0.1:{cls._port}"

    @classmethod
    def tearDownClass(cls):
        cls._server.shutdown()
        cls._server.server_close()

    def test_post_and_get_roundtrip(self):
        """POST a directive, then GET it back by peer_id."""
        now_ms = int(time.time() * 1000)
        directive = RebalanceDirective(
            target_peer_id="peer-roundtrip",
            new_layer_start=0,
            new_layer_end=24,
            total_layers=32,
            reason="test_gap_fill",
            issued_unix_ms=now_ms,
            expires_unix_ms=now_ms + 60_000,
        )
        ok, fail = publish_directives_to_dht(
            [directive],
            [self._base_url],
            timeout_s=5.0,
        )
        self.assertEqual(ok, 1)
        self.assertEqual(fail, 0)

        # Poll for the directive.
        result = poll_directives_from_dht(
            "peer-roundtrip",
            [self._base_url],
            timeout_s=5.0,
        )
        self.assertGreaterEqual(len(result), 1)
        found = [d for d in result if d.target_peer_id == "peer-roundtrip"]
        self.assertTrue(len(found) >= 1)
        self.assertEqual(found[0].new_layer_start, 0)
        self.assertEqual(found[0].new_layer_end, 24)

    def test_get_empty_for_unknown_peer(self):
        """GET /rebalance for a peer with no directives returns empty list."""
        result = poll_directives_from_dht(
            "peer-nonexistent",
            [self._base_url],
            timeout_s=5.0,
        )
        self.assertEqual(result, [])

    def test_expired_directives_pruned(self):
        """Expired directives are not returned by GET."""
        now_ms = int(time.time() * 1000)
        expired_directive = RebalanceDirective(
            target_peer_id="peer-expired",
            new_layer_start=0,
            new_layer_end=16,
            total_layers=32,
            issued_unix_ms=now_ms - 200_000,
            expires_unix_ms=now_ms - 1000,  # Already expired.
        )
        # Manually insert the expired directive.
        from dht.bootstrap import DhtBootstrapHandler
        with DhtBootstrapHandler._layer_rebalance_lock:
            DhtBootstrapHandler._layer_rebalance_directives["peer-expired"] = [
                expired_directive.to_dict()
            ]

        result = poll_directives_from_dht(
            "peer-expired",
            [self._base_url],
            timeout_s=5.0,
        )
        self.assertEqual(result, [])

    def test_post_missing_peer_id_returns_error(self):
        """POST /rebalance without target_peer_id returns 400."""
        import urllib.request
        payload = json.dumps({"new_layer_start": 0, "new_layer_end": 16}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/rebalance",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                self.fail("Expected HTTP error")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)


# ═════════════════════════════════════════════════════════════════════════════
# Group D — ModelShard.reshard
# ═════════════════════════════════════════════════════════════════════════════


class TestModelShardReshard(unittest.TestCase):
    """Group D: reshard via ToyRuntime."""

    def _make_shard(self, shard_index=0, total_shards=4) -> Any:
        from peer.model_shard import ModelShard, ToyShardConfig
        config = ToyShardConfig(
            model_id="test-reshard",
            shard_index=shard_index,
            total_shards=total_shards,
            runtime_backend="toy_cpu",
        )
        return ModelShard(config)

    def test_toy_reshard_success(self):
        shard = self._make_shard(shard_index=0, total_shards=4)
        profile_before = shard.runtime_profile()
        ok = shard.reshard(0, 16, 32)
        self.assertTrue(ok)
        profile_after = shard.runtime_profile()
        self.assertEqual(profile_after["layer_start"], 0)
        self.assertEqual(profile_after["layer_end"], 16)
        self.assertEqual(profile_after["total_layers"], 32)

    def test_toy_reshard_updates_profile(self):
        shard = self._make_shard(shard_index=1, total_shards=4)
        ok = shard.reshard(8, 24, 32)
        self.assertTrue(ok)
        profile = shard.runtime_profile()
        self.assertEqual(profile["layer_start"], 8)
        self.assertEqual(profile["layer_end"], 24)

    def test_reshard_method_exists_on_model_shard(self):
        shard = self._make_shard()
        self.assertTrue(hasattr(shard, "reshard"))
        self.assertTrue(callable(shard.reshard))


# ═════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    unittest.main()
