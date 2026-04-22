# Copyright 2026 OpenHydra contributors — Apache 2.0

"""PR-3 (B1) — gossip client tests.

Covers :class:`peer.gossip_client.GossipClient` in isolation from the Rust
layer via a stub ``FakeP2PNode`` that mimics
``publish_event(bytes)``/``poll_event() -> tuple[str, bytes] | None``.

The tests verify:

* JSON envelope format (type / data / observed_by / unix_ms).
* Subscribe by event type, wildcard subscriber, registration-order dispatch.
* Inbound queue is fully drained in a single ``tick_once()``.
* ``PEER_DEAD`` debounce (1 s default) caps per-peer amplification.
* Counters (published / received / dispatched / errors) are accurate.
* Publish errors from the Rust layer do not raise.
* Decode errors on malformed payloads increment ``decode_errors`` and
  don't break the dispatcher for subsequent messages.
* Subscriber exceptions don't block other subscribers.
* Thread lifecycle: ``start()`` then ``stop()`` joins cleanly.

Run:  ``pytest tests/test_gossip_client.py -v``
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest

from peer.gossip_client import (
    EVENT_PEER_DEAD,
    EVENT_REQUEST_HOLE_PUNCH,
    WILDCARD,
    GossipClient,
    GossipMessage,
    publish_peer_dead,
    publish_request_hole_punch,
)


# --- fake P2PNode ------------------------------------------------------------

class FakeP2PNode:
    """Drop-in for ``openhydra_network.P2PNode`` with thread-safe in-memory
    queues. ``publish_fail`` lets tests simulate ``InsufficientPeers``."""

    def __init__(self, *, self_peer_id: str = "12D3KooWSelf", publish_fail: bool = False):
        self.libp2p_peer_id = self_peer_id
        self._published: list[bytes] = []
        self._inbound: list[tuple[str, bytes]] = []
        self._lock = threading.Lock()
        self.publish_fail = bool(publish_fail)

    def publish_event(self, payload: bytes) -> None:
        if self.publish_fail:
            raise RuntimeError("gossipsub publish: InsufficientPeers")
        with self._lock:
            self._published.append(bytes(payload))

    def poll_event(self) -> tuple[str, bytes] | None:
        with self._lock:
            if not self._inbound:
                return None
            return self._inbound.pop(0)

    # --- test helpers ---

    def enqueue_from(self, sender_peer_id: str, payload: bytes) -> None:
        with self._lock:
            self._inbound.append((sender_peer_id, bytes(payload)))

    def published(self) -> list[dict]:
        with self._lock:
            return [json.loads(b.decode("utf-8")) for b in self._published]


def _make_client(
    node: FakeP2PNode,
    *,
    self_id: str = "12D3KooWSelf",
    debounce_s: float = 1.0,
    clock_fn=None,
) -> GossipClient:
    kwargs: dict[str, Any] = {
        "p2p_node": node,
        "self_libp2p_peer_id": self_id,
        "poll_interval_s": 0.02,
        "peer_dead_debounce_s": debounce_s,
    }
    if clock_fn is not None:
        kwargs["clock_fn"] = clock_fn
    return GossipClient(**kwargs)


# --- publish envelope --------------------------------------------------------

class TestPublishEnvelope:
    def test_envelope_has_canonical_keys(self):
        node = FakeP2PNode()
        client = _make_client(node)
        assert client.publish(EVENT_PEER_DEAD, {"libp2p_peer_id": "12D3KooWDead"}) is True
        sent = node.published()
        assert len(sent) == 1
        env = sent[0]
        assert env["type"] == EVENT_PEER_DEAD
        assert env["data"] == {"libp2p_peer_id": "12D3KooWDead"}
        assert env["observed_by"] == "12D3KooWSelf"
        assert isinstance(env["unix_ms"], int)
        assert env["unix_ms"] > 0

    def test_publish_without_data_uses_empty_dict(self):
        node = FakeP2PNode()
        client = _make_client(node)
        assert client.publish(EVENT_REQUEST_HOLE_PUNCH) is True
        assert node.published()[0]["data"] == {}

    def test_publish_empty_type_rejected(self):
        node = FakeP2PNode()
        client = _make_client(node)
        assert client.publish("", {"x": 1}) is False
        assert node.published() == []

    def test_publish_failure_does_not_raise(self):
        node = FakeP2PNode(publish_fail=True)
        client = _make_client(node)
        # Must not raise; must return False and bump error counter.
        assert client.publish(EVENT_PEER_DEAD, {"libp2p_peer_id": "x"}) is False
        assert client.stats()["publish_errors"] == 1

    def test_helper_publish_peer_dead(self):
        node = FakeP2PNode()
        client = _make_client(node)
        assert publish_peer_dead(client, libp2p_peer_id="12D3KooWX", reason="timeout")
        env = node.published()[0]
        assert env["type"] == EVENT_PEER_DEAD
        assert env["data"]["libp2p_peer_id"] == "12D3KooWX"
        assert env["data"]["reason"] == "timeout"

    def test_helper_publish_request_hole_punch(self):
        node = FakeP2PNode()
        client = _make_client(node)
        assert publish_request_hole_punch(
            client, from_peer_id="A", to_peer_id="B"
        )
        env = node.published()[0]
        assert env["type"] == EVENT_REQUEST_HOLE_PUNCH
        assert env["data"] == {"from_peer_id": "A", "to_peer_id": "B"}


# --- debounce ----------------------------------------------------------------

class TestPeerDeadDebounce:
    def test_same_peer_suppressed_within_window(self):
        node = FakeP2PNode()
        # Fake clock; first call at t=0, second at t=0.5 < 1.0 debounce.
        t = [0.0]
        client = _make_client(node, clock_fn=lambda: t[0], debounce_s=1.0)
        assert publish_peer_dead(client, libp2p_peer_id="P")
        t[0] = 0.5
        assert publish_peer_dead(client, libp2p_peer_id="P") is False
        # After the window, re-publishing is allowed again.
        t[0] = 1.5
        assert publish_peer_dead(client, libp2p_peer_id="P")
        assert len(node.published()) == 2

    def test_different_peers_not_suppressed(self):
        node = FakeP2PNode()
        t = [0.0]
        client = _make_client(node, clock_fn=lambda: t[0], debounce_s=1.0)
        assert publish_peer_dead(client, libp2p_peer_id="A")
        assert publish_peer_dead(client, libp2p_peer_id="B")
        assert len(node.published()) == 2

    def test_debounce_zero_disables(self):
        node = FakeP2PNode()
        t = [0.0]
        client = _make_client(node, clock_fn=lambda: t[0], debounce_s=0.0)
        for _ in range(5):
            assert publish_peer_dead(client, libp2p_peer_id="P")
        assert len(node.published()) == 5


# --- subscribe + dispatch ----------------------------------------------------

class TestSubscribe:
    def _enqueue_message(
        self,
        node: FakeP2PNode,
        *,
        event_type: str,
        data: dict[str, Any],
        observed_by: str = "12D3KooWOther",
        sender: str = "12D3KooWHop",
    ) -> None:
        envelope = {
            "type": event_type,
            "data": data,
            "observed_by": observed_by,
            "unix_ms": int(time.time() * 1000),
        }
        node.enqueue_from(sender, json.dumps(envelope).encode("utf-8"))

    def test_specific_subscriber_receives_matching(self):
        node = FakeP2PNode()
        client = _make_client(node)
        received: list[GossipMessage] = []
        client.on(EVENT_PEER_DEAD, received.append)
        self._enqueue_message(node, event_type=EVENT_PEER_DEAD, data={"libp2p_peer_id": "P"})
        n = client.tick_once()
        assert n == 1
        assert len(received) == 1
        assert received[0].type == EVENT_PEER_DEAD
        assert received[0].data["libp2p_peer_id"] == "P"
        assert received[0].propagation_source == "12D3KooWHop"
        assert received[0].observed_by == "12D3KooWOther"

    def test_other_types_not_delivered(self):
        node = FakeP2PNode()
        client = _make_client(node)
        received: list[GossipMessage] = []
        client.on(EVENT_PEER_DEAD, received.append)
        self._enqueue_message(node, event_type=EVENT_REQUEST_HOLE_PUNCH, data={})
        client.tick_once()
        assert received == []

    def test_wildcard_gets_everything(self):
        node = FakeP2PNode()
        client = _make_client(node)
        seen: list[str] = []
        client.on(WILDCARD, lambda m: seen.append(m.type))
        self._enqueue_message(node, event_type=EVENT_PEER_DEAD, data={})
        self._enqueue_message(node, event_type=EVENT_REQUEST_HOLE_PUNCH, data={})
        client.tick_once()
        assert seen == [EVENT_PEER_DEAD, EVENT_REQUEST_HOLE_PUNCH]

    def test_specific_and_wildcard_both_fire(self):
        node = FakeP2PNode()
        client = _make_client(node)
        specific: list[GossipMessage] = []
        wild: list[GossipMessage] = []
        client.on(EVENT_PEER_DEAD, specific.append)
        client.on(WILDCARD, wild.append)
        self._enqueue_message(node, event_type=EVENT_PEER_DEAD, data={})
        client.tick_once()
        assert len(specific) == 1
        assert len(wild) == 1

    def test_off_unsubscribes(self):
        node = FakeP2PNode()
        client = _make_client(node)
        received: list[GossipMessage] = []
        cb = received.append
        client.on(EVENT_PEER_DEAD, cb)
        assert client.off(EVENT_PEER_DEAD, cb) is True
        self._enqueue_message(node, event_type=EVENT_PEER_DEAD, data={})
        client.tick_once()
        assert received == []

    def test_off_returns_false_when_missing(self):
        node = FakeP2PNode()
        client = _make_client(node)
        assert client.off(EVENT_PEER_DEAD, lambda m: None) is False

    def test_raising_subscriber_does_not_block_others(self):
        node = FakeP2PNode()
        client = _make_client(node)
        seen = []

        def boom(msg):
            raise RuntimeError("intentional")

        client.on(EVENT_PEER_DEAD, boom)
        client.on(EVENT_PEER_DEAD, seen.append)
        self._enqueue_message(node, event_type=EVENT_PEER_DEAD, data={})
        client.tick_once()
        assert len(seen) == 1

    def test_tick_drains_all_queued(self):
        node = FakeP2PNode()
        client = _make_client(node)
        seen = []
        client.on(EVENT_PEER_DEAD, seen.append)
        for i in range(10):
            self._enqueue_message(
                node, event_type=EVENT_PEER_DEAD, data={"i": i}
            )
        n = client.tick_once()
        assert n == 10
        assert [m.data["i"] for m in seen] == list(range(10))

    def test_decode_error_does_not_break_dispatcher(self):
        node = FakeP2PNode()
        client = _make_client(node)
        seen = []
        client.on(EVENT_PEER_DEAD, seen.append)
        # Malformed bytes, then a real message — the real message must
        # still be delivered.
        node.enqueue_from("hopA", b"not-json")
        self._enqueue_message(node, event_type=EVENT_PEER_DEAD, data={"ok": True})
        client.tick_once()
        assert client.stats()["decode_errors"] == 1
        assert len(seen) == 1

    def test_as_peer_dead_helper(self):
        node = FakeP2PNode()
        client = _make_client(node)
        captured = []
        client.on(EVENT_PEER_DEAD, captured.append)
        self._enqueue_message(
            node,
            event_type=EVENT_PEER_DEAD,
            data={"libp2p_peer_id": "12D3KooWDead", "reason": "timeout"},
        )
        client.tick_once()
        typed = captured[0].as_peer_dead()
        assert typed is not None
        assert typed.libp2p_peer_id == "12D3KooWDead"
        assert typed.reason == "timeout"

    def test_as_request_hole_punch_helper(self):
        node = FakeP2PNode()
        client = _make_client(node)
        captured = []
        client.on(EVENT_REQUEST_HOLE_PUNCH, captured.append)
        self._enqueue_message(
            node,
            event_type=EVENT_REQUEST_HOLE_PUNCH,
            data={"from_peer_id": "A", "to_peer_id": "B"},
        )
        client.tick_once()
        typed = captured[0].as_request_hole_punch()
        assert typed is not None
        assert typed.from_peer_id == "A"
        assert typed.to_peer_id == "B"


# --- thread lifecycle --------------------------------------------------------

class TestLifecycle:
    def test_start_stop_clean_join(self):
        node = FakeP2PNode()
        client = _make_client(node)
        seen = []
        client.on(EVENT_PEER_DEAD, seen.append)
        t = client.start()
        try:
            # Enqueue after start → background thread should dispatch.
            envelope = {
                "type": EVENT_PEER_DEAD,
                "data": {"libp2p_peer_id": "P"},
                "observed_by": "other",
                "unix_ms": 0,
            }
            node.enqueue_from("hop", json.dumps(envelope).encode("utf-8"))
            # Wait up to 1s for dispatch.
            deadline = time.monotonic() + 1.0
            while not seen and time.monotonic() < deadline:
                time.sleep(0.02)
            assert len(seen) == 1
        finally:
            client.stop(join_timeout_s=1.0)
        assert not t.is_alive()

    def test_double_start_is_idempotent(self):
        node = FakeP2PNode()
        client = _make_client(node)
        t1 = client.start()
        t2 = client.start()
        assert t1 is t2
        client.stop(join_timeout_s=1.0)


# --- counters ----------------------------------------------------------------

class TestCounters:
    def test_stats_after_mixed_activity(self):
        node = FakeP2PNode()
        client = _make_client(node)
        client.on(EVENT_PEER_DEAD, lambda m: None)
        # 2 publishes, one suppressed by debounce → 1 real publish.
        t = [0.0]
        client = _make_client(node, clock_fn=lambda: t[0], debounce_s=1.0)
        client.on(EVENT_PEER_DEAD, lambda m: None)
        client.publish(EVENT_PEER_DEAD, {"libp2p_peer_id": "P"})
        client.publish(EVENT_PEER_DEAD, {"libp2p_peer_id": "P"})  # suppressed
        # 3 inbound: 1 malformed, 2 good.
        node.enqueue_from("hop", b"bad")
        for i in range(2):
            env = {"type": EVENT_PEER_DEAD, "data": {}, "observed_by": "o", "unix_ms": i}
            node.enqueue_from("hop", json.dumps(env).encode("utf-8"))
        client.tick_once()
        s = client.stats()
        assert s["published"] == 1
        assert s["received"] == 3
        assert s["dispatched"] == 2
        assert s["decode_errors"] == 1
        assert s["publish_errors"] == 0
