"""Tests for dht.hivemind_bridge — Hivemind DHT dual-stack adapter.

Groups:
    A — HivemindDHTAdapter stub mode (4 tests): when hivemind not installed.
    B — HivemindDHTAdapter with mocked hivemind (5 tests): announce, lookup,
        subkey serialization, shutdown.
    C — merge_peer_lists (4 tests): dedup by peer_id, newest-wins, empty
        inputs, mixed sources.
    D — Coordinator integration (3 tests): load_peers_from_dht with
        hivemind_adapter.
    E — Signpost module (2 tests): module exists, serve signature.
"""
from __future__ import annotations

import json
import time
import unittest
from dataclasses import asdict
from typing import Any
from unittest.mock import MagicMock, patch

from dht.hivemind_bridge import (
    HivemindDHTAdapter,
    hivemind_available,
    merge_peer_lists,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_peer_dict(
    peer_id: str = "peer-1",
    model_id: str = "openhydra-test",
    host: str = "192.168.1.10",
    port: int = 50051,
    updated_unix_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "peer_id": peer_id,
        "model_id": model_id,
        "host": host,
        "port": port,
        "updated_unix_ms": updated_unix_ms or int(time.time() * 1000),
        "load_pct": 10.0,
        "runtime_backend": "toy_cpu",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Group A — Stub mode (hivemind not installed)
# ═════════════════════════════════════════════════════════════════════════════


class TestHivemindStubMode(unittest.TestCase):
    """Group A: adapter degrades gracefully when hivemind is missing."""

    def test_hivemind_available_returns_bool(self):
        result = hivemind_available()
        self.assertIsInstance(result, bool)

    @patch("dht.hivemind_bridge._hivemind_available", False)
    def test_adapter_start_returns_false_without_hivemind(self):
        adapter = HivemindDHTAdapter(start=False)
        result = adapter.start()
        self.assertFalse(result)
        self.assertFalse(adapter.is_alive)

    @patch("dht.hivemind_bridge._hivemind_available", False)
    def test_announce_noop_without_hivemind(self):
        adapter = HivemindDHTAdapter(start=False)
        result = adapter.announce({"peer_id": "p1", "model_id": "m1"})
        self.assertFalse(result)

    @patch("dht.hivemind_bridge._hivemind_available", False)
    def test_lookup_returns_empty_without_hivemind(self):
        adapter = HivemindDHTAdapter(start=False)
        result = adapter.lookup("some-model")
        self.assertEqual(result, [])


# ═════════════════════════════════════════════════════════════════════════════
# Group B — Mocked hivemind
# ═════════════════════════════════════════════════════════════════════════════


class TestHivemindMocked(unittest.TestCase):
    """Group B: adapter with mocked hivemind.DHT."""

    def _make_adapter_with_mock(self) -> tuple[HivemindDHTAdapter, MagicMock]:
        """Create an adapter with a mocked hivemind.DHT instance."""
        mock_dht = MagicMock()
        mock_dht.peer_id = "QmMockPeerId"
        adapter = HivemindDHTAdapter(start=False)
        adapter._dht = mock_dht
        adapter._started = True
        return adapter, mock_dht

    def test_announce_calls_store(self):
        adapter, mock_dht = self._make_adapter_with_mock()
        mock_dht.store.return_value = True

        with patch("dht.hivemind_bridge.hivemind") as mock_hm:
            mock_hm.get_dht_time.return_value = 1000.0
            result = adapter.announce(
                {"peer_id": "p1", "model_id": "m1", "host": "1.2.3.4"},
                ttl_seconds=300,
            )
        self.assertTrue(result)
        mock_dht.store.assert_called_once()
        call_kwargs = mock_dht.store.call_args
        self.assertEqual(call_kwargs.kwargs.get("key") or call_kwargs[1].get("key", call_kwargs[0][0] if call_kwargs[0] else None), "m1")

    def test_announce_missing_model_id_returns_false(self):
        adapter, mock_dht = self._make_adapter_with_mock()
        result = adapter.announce({"peer_id": "p1"})  # No model_id.
        self.assertFalse(result)
        mock_dht.store.assert_not_called()

    def test_lookup_parses_subkeys(self):
        adapter, mock_dht = self._make_adapter_with_mock()
        peer_json = json.dumps(_make_peer_dict(peer_id="p1"))
        mock_dht.get.return_value = {
            "p1": (peer_json, time.time() + 300),
        }
        result = adapter.lookup("openhydra-test")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["peer_id"], "p1")

    def test_lookup_returns_empty_on_none(self):
        adapter, mock_dht = self._make_adapter_with_mock()
        mock_dht.get.return_value = None
        result = adapter.lookup("no-such-model")
        self.assertEqual(result, [])

    def test_shutdown(self):
        adapter, mock_dht = self._make_adapter_with_mock()
        adapter.shutdown()
        self.assertFalse(adapter.is_alive)
        mock_dht.shutdown.assert_called_once()

    def test_peer_id_property(self):
        adapter, mock_dht = self._make_adapter_with_mock()
        self.assertEqual(adapter.peer_id, "QmMockPeerId")


# ═════════════════════════════════════════════════════════════════════════════
# Group C — merge_peer_lists
# ═════════════════════════════════════════════════════════════════════════════


class TestMergePeerLists(unittest.TestCase):
    """Group C: deduplication and newest-wins merge."""

    def test_no_overlap(self):
        http = [_make_peer_dict("a", updated_unix_ms=1000)]
        hm = [_make_peer_dict("b", updated_unix_ms=2000)]
        merged = merge_peer_lists(http, hm)
        self.assertEqual(len(merged), 2)
        ids = {d["peer_id"] for d in merged}
        self.assertEqual(ids, {"a", "b"})

    def test_newest_wins(self):
        old = _make_peer_dict("p1", host="1.1.1.1", updated_unix_ms=1000)
        new = _make_peer_dict("p1", host="2.2.2.2", updated_unix_ms=2000)
        merged = merge_peer_lists([old], [new])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["host"], "2.2.2.2")

    def test_http_wins_when_newer(self):
        http = [_make_peer_dict("p1", host="http-host", updated_unix_ms=5000)]
        hm = [_make_peer_dict("p1", host="hm-host", updated_unix_ms=3000)]
        merged = merge_peer_lists(http, hm)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["host"], "http-host")

    def test_empty_inputs(self):
        self.assertEqual(merge_peer_lists([], []), [])

    def test_sorted_by_timestamp_descending(self):
        peers = [
            _make_peer_dict("a", updated_unix_ms=100),
            _make_peer_dict("b", updated_unix_ms=300),
            _make_peer_dict("c", updated_unix_ms=200),
        ]
        merged = merge_peer_lists(peers, [])
        self.assertEqual(merged[0]["peer_id"], "b")
        self.assertEqual(merged[1]["peer_id"], "c")
        self.assertEqual(merged[2]["peer_id"], "a")


# ═════════════════════════════════════════════════════════════════════════════
# Group D — Coordinator integration
# ═════════════════════════════════════════════════════════════════════════════


class TestCoordinatorHivemindIntegration(unittest.TestCase):
    """Group D: load_peers_from_dht with hivemind_adapter."""

    def test_hivemind_only_mode(self):
        """When HTTP lookup returns empty but hivemind has peers, they appear."""
        from coordinator.path_finder import load_peers_from_dht

        mock_adapter = MagicMock()
        mock_adapter.lookup.return_value = [
            _make_peer_dict("hm-peer-1", host="10.0.0.1", port=50051),
        ]

        with patch(
            "coordinator.path_finder._lookup_peers_payload",
            return_value=[],
        ):
            peers = load_peers_from_dht(
                dht_urls=["http://fake-dht:8468"],
                model_id="test-model",
                timeout_s=1.0,
                hivemind_adapter=mock_adapter,
            )

        mock_adapter.lookup.assert_called_once()
        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0].peer_id, "hm-peer-1")

    def test_merge_http_and_hivemind(self):
        """Simulate both sources returning peers; verify dedup."""
        from coordinator.path_finder import load_peers_from_dht, _lookup_peers_payload

        http_peer = _make_peer_dict("shared-peer", host="http-host", updated_unix_ms=1000)
        hm_peer = _make_peer_dict("shared-peer", host="hm-host", updated_unix_ms=2000)

        mock_adapter = MagicMock()
        mock_adapter.lookup.return_value = [hm_peer]

        with patch(
            "coordinator.path_finder._lookup_peers_payload",
            return_value=[http_peer],
        ):
            peers = load_peers_from_dht(
                dht_urls=["http://fake-dht:8468"],
                model_id="test-model",
                timeout_s=1.0,
                hivemind_adapter=mock_adapter,
            )

        # Should have 1 peer (deduped), with hm_host winning (newer).
        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0].host, "hm-host")

    def test_hivemind_adapter_none_fallback(self):
        """When hivemind_adapter is None, only HTTP is used."""
        from coordinator.path_finder import load_peers_from_dht

        http_peer = _make_peer_dict("http-only", host="1.2.3.4")

        with patch(
            "coordinator.path_finder._lookup_peers_payload",
            return_value=[http_peer],
        ):
            peers = load_peers_from_dht(
                dht_urls=["http://fake-dht:8468"],
                model_id="test-model",
                timeout_s=1.0,
                hivemind_adapter=None,
            )

        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0].peer_id, "http-only")


# ═════════════════════════════════════════════════════════════════════════════
# Group E — Signpost module
# ═════════════════════════════════════════════════════════════════════════════


class TestSignpostModule(unittest.TestCase):
    """Group E: dht.signpost module structure."""

    def test_module_importable(self):
        import dht.signpost
        self.assertTrue(hasattr(dht.signpost, "serve"))
        self.assertTrue(hasattr(dht.signpost, "main"))

    def test_serve_signature(self):
        import inspect
        from dht.signpost import serve
        sig = inspect.signature(serve)
        params = list(sig.parameters.keys())
        self.assertIn("host", params)
        self.assertIn("port", params)
        self.assertIn("identity_path", params)


# ═════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    unittest.main()
