# Copyright 2026 OpenHydra contributors — Apache 2.0

"""PR-2 (A3) — DCUtR stats exposure tests.

Covers the Rust → Python → HTTP path for libp2p Direct-Connection-Upgrade-
through-Relay hole-punch counters:

* The ``_collect_network_block`` helper handles the full failure matrix
  (p2p disabled / binding missing / runtime error / happy path) without
  raising so ``/v1/internal/capacity`` remains available even when the
  Rust wheel is unbuilt.
* The ``/v1/internal/capacity`` HTTP handler emits a ``network.dcutr``
  block with the correct shape and populated counters when a stub P2P
  node is wired in.
* The ``openhydra_network.P2PNode.get_dcutr_stats()`` PyO3 method is
  reachable and returns a well-formed dict on a freshly-started node
  (all counters zero, no peers yet). Skipped when the Rust wheel is not
  installed.

Run: ``pytest tests/test_dcutr_stats.py -v``
"""

from __future__ import annotations

import json
from typing import Any

import pytest


# -----------------------------------------------------------------------------
# _collect_network_block — failure-mode matrix
# -----------------------------------------------------------------------------

class TestCollectNetworkBlock:
    def _block(self, p2p_node: Any) -> dict:
        from coordinator.api_server import _collect_network_block
        return _collect_network_block(p2p_node)

    def test_none_returns_disabled_marker(self):
        block = self._block(None)
        assert "dcutr" in block
        assert block["dcutr"]["available"] is False
        assert block["dcutr"]["reason"] == "p2p_disabled"
        # Counters must always be present (0) so downstream dashboards don't
        # need to treat missing keys as a separate case.
        for key in ("successes", "failures", "direct_peers_count"):
            assert block["dcutr"][key] == 0

    def test_missing_binding_marker(self):
        class StubOldNode:
            pass  # no get_dcutr_stats method

        block = self._block(StubOldNode())
        assert block["dcutr"]["available"] is False
        assert block["dcutr"]["reason"] == "binding_missing"

    def test_runtime_error_is_caught(self):
        class StubRaisingNode:
            def get_dcutr_stats(self):
                raise RuntimeError("swarm not running")

        block = self._block(StubRaisingNode())
        assert block["dcutr"]["available"] is False
        assert block["dcutr"]["reason"] == "runtime_error"
        assert "swarm not running" in block["dcutr"]["error"]

    def test_happy_path_populated_counters(self):
        class StubGoodNode:
            def get_dcutr_stats(self):
                return {"successes": 7, "failures": 2, "direct_peers_count": 3}

        block = self._block(StubGoodNode())
        assert block["dcutr"]["available"] is True
        assert block["dcutr"]["successes"] == 7
        assert block["dcutr"]["failures"] == 2
        assert block["dcutr"]["direct_peers_count"] == 3

    def test_non_int_values_coerced_to_zero(self):
        """Defensive: if the Rust binding ever returns garbage, degrade
        cleanly rather than 500 the whole capacity endpoint."""
        class StubBadTypesNode:
            def get_dcutr_stats(self):
                return {"successes": "garbage", "failures": None, "direct_peers_count": 5}

        block = self._block(StubBadTypesNode())
        assert block["dcutr"]["available"] is True
        assert block["dcutr"]["successes"] == 0
        assert block["dcutr"]["failures"] == 0
        assert block["dcutr"]["direct_peers_count"] == 5


# -----------------------------------------------------------------------------
# End-to-end /v1/internal/capacity HTTP integration
# -----------------------------------------------------------------------------

class _StubP2PNode:
    """Minimal stand-in for openhydra_network.P2PNode.

    The HTTP handler accesses only ``libp2p_peer_id`` (for ``_node_meta``
    bootstrap) and ``get_dcutr_stats()`` (for the network block).
    """

    def __init__(self, *, successes: int = 0, failures: int = 0, direct: int = 0):
        self.libp2p_peer_id = "12D3KooWStubStubStubStubStubStubStubStubStubStubStub"
        self._successes = int(successes)
        self._failures = int(failures)
        self._direct = int(direct)

    def get_dcutr_stats(self) -> dict:
        return {
            "successes": self._successes,
            "failures": self._failures,
            "direct_peers_count": self._direct,
        }


class _StubEngine:
    """Minimal engine stand-in for the capacity handler."""

    model_catalog: list = []

    def __init__(self):
        self._discovery_svc = None


def _boot_handler(
    *,
    p2p_node: Any | None,
    node_meta: dict | None = None,
) -> None:
    """Wire the OpenHydraHandler class attributes directly without
    spinning up a ThreadingHTTPServer. The capacity endpoint is pure
    request→response so we only need the class state."""
    from coordinator.api_server import OpenHydraHandler
    OpenHydraHandler.engine = _StubEngine()
    OpenHydraHandler._node_meta = node_meta or {
        "peer_id": "oh-stub01",
        "libp2p_peer_id": getattr(p2p_node, "libp2p_peer_id", "") if p2p_node else "",
        "ports": {"api": 11434, "grpc": 50051, "p2p": 4001},
        "advertise_host": "127.0.0.1",
    }
    OpenHydraHandler._p2p_node = p2p_node


def _call_capacity_handler() -> dict:
    """Invoke _handle_capacity_report against a fake request harness and
    return the JSON payload it emitted."""
    from coordinator.api_server import OpenHydraHandler

    captured: dict[str, Any] = {}

    # Build a handler instance without invoking BaseHTTPRequestHandler.__init__
    # (which would attempt real socket I/O). We only need the bound methods.
    handler = object.__new__(OpenHydraHandler)
    handler.path = "/v1/internal/capacity"

    def fake_send_json(payload, *, status=200, headers=None):
        captured["payload"] = payload
        captured["status"] = status

    handler._send_json = fake_send_json  # type: ignore[method-assign]
    handler._handle_capacity_report(rid_headers=None)
    return captured.get("payload") or {}


class TestCapacityEndpointDcutrBlock:
    def test_network_dcutr_block_present_with_p2p(self):
        _boot_handler(p2p_node=_StubP2PNode(successes=12, failures=3, direct=4))
        payload = _call_capacity_handler()
        assert "network" in payload
        assert "dcutr" in payload["network"]
        dcutr = payload["network"]["dcutr"]
        assert dcutr["available"] is True
        assert dcutr["successes"] == 12
        assert dcutr["failures"] == 3
        assert dcutr["direct_peers_count"] == 4

    def test_network_dcutr_block_disabled_without_p2p(self):
        _boot_handler(p2p_node=None)
        payload = _call_capacity_handler()
        assert payload["network"]["dcutr"]["available"] is False
        assert payload["network"]["dcutr"]["reason"] == "p2p_disabled"

    def test_payload_remains_schema_v2(self):
        """Regression: the network block is additive — the Phase 1 / 1.5
        CapacityReport shape must still be v2."""
        _boot_handler(p2p_node=_StubP2PNode())
        payload = _call_capacity_handler()
        assert payload["schema_version"] == 2
        # node_persona still emitted at root level.
        assert "node_persona" in payload

    def test_payload_is_json_serialisable(self):
        """End-to-end shape sanity — the network block must survive a
        ``json.dumps`` round-trip without a custom encoder."""
        _boot_handler(p2p_node=_StubP2PNode(successes=1))
        payload = _call_capacity_handler()
        text = json.dumps(payload)
        restored = json.loads(text)
        assert restored["network"]["dcutr"]["successes"] == 1


# -----------------------------------------------------------------------------
# PyO3 end-to-end — actual openhydra_network wheel (skipped when absent)
# -----------------------------------------------------------------------------

try:
    import openhydra_network  # type: ignore
    _HAS_NETWORK = True
except Exception:
    _HAS_NETWORK = False


@pytest.mark.skipif(not _HAS_NETWORK, reason="openhydra_network wheel not installed")
class TestRustBinding:
    def test_get_dcutr_stats_returns_dict(self, tmp_path):
        import openhydra_network as ohn  # type: ignore
        node = ohn.P2PNode(
            identity_key_path=str(tmp_path / "id.key"),
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
            bootstrap_peers=[],
        )
        node.start()
        try:
            stats = node.get_dcutr_stats()
        finally:
            node.stop()
        assert isinstance(stats, dict)
        for key in ("successes", "failures", "direct_peers_count"):
            assert key in stats
            assert isinstance(stats[key], int)
        # A freshly-started node has no DCUtR activity yet.
        assert stats["successes"] == 0
        assert stats["failures"] == 0
        assert stats["direct_peers_count"] == 0

    def test_method_available_before_start_raises_cleanly(self, tmp_path):
        """Calling the method before ``start()`` must raise a PyRuntimeError
        (not segfault or return garbage)."""
        import openhydra_network as ohn  # type: ignore
        node = ohn.P2PNode(
            identity_key_path=str(tmp_path / "id2.key"),
            listen_addrs=["/ip4/127.0.0.1/tcp/0"],
            bootstrap_peers=[],
        )
        with pytest.raises(Exception):  # noqa: PT011 — any runtime error is acceptable
            node.get_dcutr_stats()
