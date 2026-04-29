# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b live-bench — LibP2PSwarmEventBus adapter tests.

The libp2p adapter wraps the existing GossipClient (peer/gossip_client.py)
to plug into the SwarmEventBus ABC. Tests use a fake P2PNode (the
same shape the real Rust binding exposes) so we exercise the
adapter end-to-end without an actual libp2p network.

Coverage:
* Publish round-trip: typed payload → GossipClient envelope →
  fake P2PNode.publish_event(bytes) → poll_event() → adapter
  decoder → typed SwarmEvent delivered to handler.
* Schema-version handling: events with mismatched ``v`` decode
  cleanly (forward-compat) or are rejected with a structured
  log when v > SCHEMA_VERSION.
* Subscription unsubscribe propagates through GossipClient.off().
* Handler errors are isolated.
* Wrong payload type rejected at publish time.
"""

from __future__ import annotations

from collections import deque
import json
from typing import Any

import pytest

from coordinator.swarm_events import (
    EVENT_TYPE_PROMOTE_DRAFTER,
    EVENT_TYPE_REGISTER_DRAFT_MODEL,
    EVENT_TYPE_VERIFY_RESULT,
    LibP2PSwarmEventBus,
    PromoteDrafter,
    RegisterDraftModel,
    SCHEMA_VERSION,
    SwarmEvent,
    VerifyResult,
)


# ── Fake P2P node + GossipClient harness ───────────────────────────────


class _FakeP2PNode:
    """Minimal stand-in for the openhydra_network.P2PNode binding.

    Implements just enough surface for GossipClient: ``publish_event(bytes)``
    appends to an outbound queue; ``poll_event()`` pops from an
    inbound queue. Tests inject inbound events via ``deliver()`` and
    inspect outbound publishes via ``outbound``.
    """

    def __init__(self):
        self.outbound: list[bytes] = []
        self._inbound: deque = deque()

    def publish_event(self, payload_bytes: bytes) -> None:
        self.outbound.append(bytes(payload_bytes))

    def poll_event(self):
        if not self._inbound:
            return None
        return self._inbound.popleft()

    def deliver(self, propagation_source: str, payload_bytes: bytes) -> None:
        """Tests use this to simulate an inbound event."""
        self._inbound.append((str(propagation_source), bytes(payload_bytes)))


@pytest.fixture
def harness():
    """Yields (gossip_client, p2p, libp2p_bus). Tick the gossip
    client manually via ``tick_once`` for deterministic tests."""
    from peer.gossip_client import GossipClient

    p2p = _FakeP2PNode()
    gc = GossipClient(
        p2p_node=p2p,
        self_libp2p_peer_id="local-peer",
        poll_interval_s=0.01,
    )
    bus = LibP2PSwarmEventBus(gc, local_peer_id="local-peer")
    yield gc, p2p, bus
    bus.close()


# ── Construction ───────────────────────────────────────────────────────


def test_libp2p_bus_requires_gossip_client():
    with pytest.raises(ValueError, match="gossip_client"):
        LibP2PSwarmEventBus(None, local_peer_id="x")


def test_libp2p_bus_requires_local_peer_id():
    from peer.gossip_client import GossipClient
    p2p = _FakeP2PNode()
    gc = GossipClient(p2p_node=p2p, self_libp2p_peer_id="x")
    with pytest.raises(ValueError, match="local_peer_id"):
        LibP2PSwarmEventBus(gc, local_peer_id="")


# ── Publish round-trip ─────────────────────────────────────────────────


def test_publish_verify_result_serialises_via_gossip_client(harness):
    gc, p2p, bus = harness
    payload = VerifyResult(
        session_id="s1", accepted_len=8, bonus_token=42,
        kv_rollback_to=128, block_index=2,
    )
    bus.publish(payload, from_peer="coord")

    assert len(p2p.outbound) == 1
    envelope = json.loads(p2p.outbound[0].decode("utf-8"))
    assert envelope["type"] == EVENT_TYPE_VERIFY_RESULT
    # Schema version + payload fields live inside `data`.
    assert envelope["data"]["v"] == SCHEMA_VERSION
    assert envelope["data"]["session_id"] == "s1"
    assert envelope["data"]["accepted_len"] == 8


def test_publish_register_draft_model_round_trip(harness):
    gc, p2p, bus = harness
    payload = RegisterDraftModel(
        target_path="Qwen/Qwen3.5-4B",
        draft_path="z-lab/Qwen3.5-4B-DFlash",
        block_size=16, backend="mlx",
    )
    bus.publish(payload, from_peer="coord")

    envelope = json.loads(p2p.outbound[0].decode("utf-8"))
    assert envelope["type"] == EVENT_TYPE_REGISTER_DRAFT_MODEL
    assert envelope["data"]["target_path"] == "Qwen/Qwen3.5-4B"


def test_publish_promote_drafter_round_trip(harness):
    gc, p2p, bus = harness
    payload = PromoteDrafter(
        from_peer_id="coord", to_peer_id="stage0", unix_ms=1000,
    )
    bus.publish(payload, from_peer="stage0")

    envelope = json.loads(p2p.outbound[0].decode("utf-8"))
    assert envelope["type"] == EVENT_TYPE_PROMOTE_DRAFTER
    assert envelope["data"]["to_peer_id"] == "stage0"


def test_publish_validates_payload_before_serialise(harness):
    gc, p2p, bus = harness
    # block_size=99 fails validate(); LibP2PSwarmEventBus.publish
    # surfaces the error rather than silently sending invalid data.
    bad = RegisterDraftModel(
        target_path="x", draft_path="y", block_size=99, backend="mlx",
    )
    with pytest.raises(Exception):  # EventDecodeError or ValueError
        bus.publish(bad)
    assert p2p.outbound == []


def test_publish_unknown_payload_type_raises(harness):
    gc, p2p, bus = harness
    with pytest.raises(TypeError, match="unsupported payload"):
        bus.publish({"not": "a payload"})


# ── Subscribe round-trip ───────────────────────────────────────────────


def test_subscribe_decodes_inbound_event_to_typed_payload(harness):
    gc, p2p, bus = harness
    received: list[SwarmEvent] = []
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, received.append)

    # Simulate an inbound event from another peer.
    inbound = json.dumps({
        "type": EVENT_TYPE_VERIFY_RESULT,
        "data": {
            "v": SCHEMA_VERSION,
            "session_id": "remote-session",
            "accepted_len": 12,
            "bonus_token": 7,
            "kv_rollback_to": 200,
            "block_index": 5,
        },
        "observed_by": "remote-peer",
        "unix_ms": 1700000000000,
    }).encode("utf-8")
    p2p.deliver("remote-peer", inbound)

    # Drain the gossip client (would otherwise be done by the
    # background poll thread).
    gc.tick_once()

    assert len(received) == 1
    event = received[0]
    assert event.type == EVENT_TYPE_VERIFY_RESULT
    assert event.payload.session_id == "remote-session"
    assert event.payload.accepted_len == 12
    assert event.from_peer == "remote-peer"


def test_subscribe_multiple_handlers_all_receive(harness):
    gc, p2p, bus = harness
    a, b = [], []
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, a.append)
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, b.append)

    inbound = json.dumps({
        "type": EVENT_TYPE_VERIFY_RESULT,
        "data": {
            "v": SCHEMA_VERSION,
            "session_id": "s", "accepted_len": 0, "bonus_token": 0,
            "kv_rollback_to": 0, "block_index": 0,
        },
        "observed_by": "remote",
        "unix_ms": 1,
    }).encode("utf-8")
    p2p.deliver("remote", inbound)
    gc.tick_once()

    assert len(a) == 1
    assert len(b) == 1


def test_subscribe_unsubscribe_propagates_to_gossip_client(harness):
    gc, p2p, bus = harness
    received: list[SwarmEvent] = []
    unsub = bus.subscribe(EVENT_TYPE_VERIFY_RESULT, received.append)
    unsub()
    # Remove via off() returned True — verified indirectly: a subsequent
    # inbound event is NOT delivered to the handler.
    inbound = json.dumps({
        "type": EVENT_TYPE_VERIFY_RESULT,
        "data": {
            "v": SCHEMA_VERSION,
            "session_id": "s", "accepted_len": 0, "bonus_token": 0,
            "kv_rollback_to": 0, "block_index": 0,
        },
        "observed_by": "remote",
        "unix_ms": 1,
    }).encode("utf-8")
    p2p.deliver("remote", inbound)
    gc.tick_once()
    assert received == []


def test_subscribe_unknown_event_type_rejected(harness):
    gc, p2p, bus = harness
    with pytest.raises(ValueError, match="unknown event_type"):
        bus.subscribe("not_a_real_type", lambda e: None)


# ── Decode failures ────────────────────────────────────────────────────


def test_subscribe_handles_malformed_data_gracefully(harness):
    """An inbound event with broken data — e.g. negative
    accepted_len — must NOT crash the gossip dispatcher. The
    handler simply doesn't fire."""
    gc, p2p, bus = harness
    received: list[SwarmEvent] = []
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, received.append)

    bad = json.dumps({
        "type": EVENT_TYPE_VERIFY_RESULT,
        "data": {
            "v": SCHEMA_VERSION,
            "session_id": "s",
            "accepted_len": -5,    # invalid
            "bonus_token": 0,
            "kv_rollback_to": 0,
            "block_index": 0,
        },
        "observed_by": "remote",
        "unix_ms": 1,
    }).encode("utf-8")
    p2p.deliver("remote", bad)
    # Must not raise.
    gc.tick_once()
    assert received == []


def test_subscribe_drops_future_schema_events(harness):
    """A peer running v=1 receives an event with v=999 — the
    decoder must drop it cleanly rather than mis-parsing."""
    gc, p2p, bus = harness
    received: list[SwarmEvent] = []
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, received.append)

    future = json.dumps({
        "type": EVENT_TYPE_VERIFY_RESULT,
        "data": {
            "v": 999,
            "session_id": "s", "accepted_len": 0, "bonus_token": 0,
            "kv_rollback_to": 0, "block_index": 0,
        },
        "observed_by": "remote",
        "unix_ms": 1,
    }).encode("utf-8")
    p2p.deliver("remote", future)
    gc.tick_once()
    assert received == []


def test_handler_exception_isolated_from_other_handlers(harness):
    gc, p2p, bus = harness
    boom_count = [0]
    other_count = [0]

    def boom(e):
        boom_count[0] += 1
        raise RuntimeError("synthetic")

    def other(e):
        other_count[0] += 1

    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, boom)
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, other)

    inbound = json.dumps({
        "type": EVENT_TYPE_VERIFY_RESULT,
        "data": {
            "v": SCHEMA_VERSION,
            "session_id": "s", "accepted_len": 0, "bonus_token": 0,
            "kv_rollback_to": 0, "block_index": 0,
        },
        "observed_by": "remote",
        "unix_ms": 1,
    }).encode("utf-8")
    p2p.deliver("remote", inbound)
    gc.tick_once()

    assert boom_count[0] == 1
    assert other_count[0] == 1


# ── close() detaches all subscriptions ────────────────────────────────


def test_close_unsubscribes_all_handlers(harness):
    gc, p2p, bus = harness
    a, b = [], []
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, a.append)
    bus.subscribe(EVENT_TYPE_PROMOTE_DRAFTER, b.append)

    bus.close()

    # Subsequent inbound events are not delivered.
    for ev_type in (EVENT_TYPE_VERIFY_RESULT, EVENT_TYPE_PROMOTE_DRAFTER):
        if ev_type == EVENT_TYPE_VERIFY_RESULT:
            payload_dict = {
                "session_id": "s", "accepted_len": 0, "bonus_token": 0,
                "kv_rollback_to": 0, "block_index": 0,
            }
        else:
            payload_dict = {
                "from_peer_id": "a", "to_peer_id": "b", "unix_ms": 1,
            }
        inbound = json.dumps({
            "type": ev_type,
            "data": {"v": SCHEMA_VERSION, **payload_dict},
            "observed_by": "remote",
            "unix_ms": 1,
        }).encode("utf-8")
        p2p.deliver("remote", inbound)

    gc.tick_once()
    assert a == []
    assert b == []
    # Idempotent.
    bus.close()
