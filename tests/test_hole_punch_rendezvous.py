# Copyright 2026 OpenHydra contributors — Apache 2.0

"""PR-3 B1 rendezvous follow-up — simultaneous-dial hole-punch tests.

Covers the two halves that together force a DCUtR hole punch on
symmetric NAT:

* **Active side** — :func:`coordinator.path_finder.maybe_request_hole_punch`
  publishes a ``REQUEST_HOLE_PUNCH`` gossip event **iff** routing to the
  candidate peer would traverse a libp2p circuit relay (via
  :func:`coordinator.path_finder.is_relay_bound`).

* **Passive side** — :func:`peer.gossip_client.attach_hole_punch_responder`
  subscribes to incoming ``REQUEST_HOLE_PUNCH`` events and, when the
  ``to_peer_id`` targets this node, calls ``p2p_node.dial_peer(from)``
  to issue the simultaneous dial.

Plus the per-pair 5 s debounce on the gossip client that keeps the
active side from asking the same peer twice in a burst.

The tests stay at the Python level — the Rust ``dial_peer`` is covered
by the wheel-rebuild smoke test in the PR description; here we inject a
stub :class:`FakeP2PNode` that records ``dial_peer`` calls.

Run:  ``pytest tests/test_hole_punch_rendezvous.py -v``
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from coordinator.path_finder import (
    PeerEndpoint,
    is_relay_bound,
    maybe_request_hole_punch,
    request_hole_punch,
)
from peer.gossip_client import (
    EVENT_REQUEST_HOLE_PUNCH,
    GossipClient,
    attach_hole_punch_responder,
    publish_request_hole_punch,
)


# --- shared fakes ------------------------------------------------------------

class FakeP2PNode:
    """Stub matching the openhydra_network.P2PNode surface we use.

    Records dial_peer calls and lets tests inject inbound gossip.
    """

    def __init__(self, *, publish_fail: bool = False, dial_fail: Exception | None = None):
        self._published: list[bytes] = []
        self._inbound: list[tuple[str, bytes]] = []
        self._dials: list[str] = []
        self.publish_fail = publish_fail
        self.dial_fail = dial_fail

    def publish_event(self, payload: bytes) -> None:
        if self.publish_fail:
            raise RuntimeError("gossipsub publish: InsufficientPeers")
        self._published.append(bytes(payload))

    def poll_event(self) -> tuple[str, bytes] | None:
        return self._inbound.pop(0) if self._inbound else None

    def dial_peer(self, peer_id: str) -> None:
        if self.dial_fail is not None:
            raise self.dial_fail
        self._dials.append(str(peer_id))

    # --- helpers ---
    def enqueue_from(self, sender: str, payload: bytes) -> None:
        self._inbound.append((sender, bytes(payload)))

    def published(self) -> list[dict]:
        return [json.loads(b.decode("utf-8")) for b in self._published]


def _peer(
    peer_id: str = "remote-1",
    libp2p: str = "12D3KooWRemote",
    *,
    requires_relay: bool = False,
    relay_address: str = "",
    host: str = "10.192.11.15",
    port: int = 50051,
) -> PeerEndpoint:
    return PeerEndpoint(
        peer_id=peer_id,
        host=host,
        port=port,
        model_id="openhydra-qwen3.5-2b",
        operator_id="test",
        runtime_backend="pytorch",
        libp2p_peer_id=libp2p,
        requires_relay=requires_relay,
        relay_address=relay_address,
    )


# --- is_relay_bound ----------------------------------------------------------

class TestIsRelayBound:
    def test_requires_relay_flag(self):
        assert is_relay_bound(_peer(requires_relay=True)) is True

    def test_relay_address_alone(self):
        assert is_relay_bound(
            _peer(relay_address="/ip4/45.79.190.172/tcp/4001/p2p-circuit")
        ) is True

    def test_no_relay_signals_returns_false(self):
        assert is_relay_bound(_peer()) is False

    def test_circuit_in_derived_address(self):
        # Fabricate a host string containing /p2p-circuit — PeerEndpoint's
        # ``address`` property composes ``host:port`` so we use the
        # relay-address fallback instead; the string-search fallback
        # kicks in only when someone hand-builds an address.
        class _Fake:
            requires_relay = False
            relay_address = ""
            address = "/ip4/1.1.1.1/tcp/4001/p2p/X/p2p-circuit/p2p/Y"

        assert is_relay_bound(_Fake()) is True  # type: ignore[arg-type]


# --- publish_request_hole_punch & request_hole_punch -------------------------

class TestRequestHolePunch:
    def _client(self, node: FakeP2PNode, *, debounce_s: float = 0.0) -> GossipClient:
        return GossipClient(
            p2p_node=node,
            self_libp2p_peer_id="12D3KooWSelf",
            poll_interval_s=0.02,
            peer_dead_debounce_s=0.0,
            hole_punch_debounce_s=debounce_s,
        )

    def test_request_publishes_envelope(self):
        node = FakeP2PNode()
        client = self._client(node)
        ok = request_hole_punch(
            client,
            self_libp2p_peer_id="12D3KooWSelf",
            peer=_peer(libp2p="12D3KooWTarget"),
        )
        assert ok is True
        published = node.published()
        assert len(published) == 1
        env = published[0]
        assert env["type"] == EVENT_REQUEST_HOLE_PUNCH
        assert env["data"]["from_peer_id"] == "12D3KooWSelf"
        assert env["data"]["to_peer_id"] == "12D3KooWTarget"

    def test_self_loop_guard(self):
        """Never ask ourselves to dial ourselves."""
        node = FakeP2PNode()
        client = self._client(node)
        ok = request_hole_punch(
            client,
            self_libp2p_peer_id="12D3KooWSelf",
            peer=_peer(libp2p="12D3KooWSelf"),
        )
        assert ok is False
        assert node.published() == []

    def test_empty_ids_suppressed(self):
        node = FakeP2PNode()
        client = self._client(node)
        assert request_hole_punch(
            client, self_libp2p_peer_id="", peer=_peer(libp2p="X")
        ) is False
        assert request_hole_punch(
            client, self_libp2p_peer_id="Y", peer=_peer(libp2p="")
        ) is False
        assert node.published() == []

    def test_gossip_none_returns_false(self):
        # Before the gossip client is wired, path_finder shouldn't crash.
        assert request_hole_punch(
            None, self_libp2p_peer_id="A", peer=_peer(libp2p="B")
        ) is False

    def test_publish_failure_is_non_raising(self):
        node = FakeP2PNode(publish_fail=True)
        client = self._client(node)
        # Must not raise; must return False.
        assert request_hole_punch(
            client, self_libp2p_peer_id="A", peer=_peer(libp2p="B")
        ) is False


class TestHolePunchDebounce:
    def test_pair_debounce_suppresses_burst(self):
        """Two calls within the window → only one reaches the wire."""
        node = FakeP2PNode()
        t = [0.0]
        client = GossipClient(
            p2p_node=node,
            self_libp2p_peer_id="A",
            hole_punch_debounce_s=5.0,
            clock_fn=lambda: t[0],
        )
        assert publish_request_hole_punch(client, from_peer_id="A", to_peer_id="B")
        t[0] = 1.0
        assert publish_request_hole_punch(client, from_peer_id="A", to_peer_id="B") is False
        t[0] = 4.9
        assert publish_request_hole_punch(client, from_peer_id="A", to_peer_id="B") is False
        # After the window expires, re-publishing is allowed.
        t[0] = 5.5
        assert publish_request_hole_punch(client, from_peer_id="A", to_peer_id="B")
        assert len(node.published()) == 2

    def test_different_pairs_independent(self):
        """Debounce on (A,B) must not block (A,C)."""
        node = FakeP2PNode()
        t = [0.0]
        client = GossipClient(
            p2p_node=node,
            self_libp2p_peer_id="A",
            hole_punch_debounce_s=5.0,
            clock_fn=lambda: t[0],
        )
        assert publish_request_hole_punch(client, from_peer_id="A", to_peer_id="B")
        assert publish_request_hole_punch(client, from_peer_id="A", to_peer_id="C")
        # (A,B) again in the window — suppressed.
        assert publish_request_hole_punch(
            client, from_peer_id="A", to_peer_id="B"
        ) is False
        assert len(node.published()) == 2

    def test_direction_matters(self):
        """(A→B) and (B→A) are distinct pairs (you could be both the
        initiator and the target in different chains)."""
        node = FakeP2PNode()
        t = [0.0]
        client = GossipClient(
            p2p_node=node,
            self_libp2p_peer_id="A",
            hole_punch_debounce_s=5.0,
            clock_fn=lambda: t[0],
        )
        assert publish_request_hole_punch(client, from_peer_id="A", to_peer_id="B")
        assert publish_request_hole_punch(client, from_peer_id="B", to_peer_id="A")
        assert len(node.published()) == 2


# --- maybe_request_hole_punch ------------------------------------------------

class TestMaybeRequestHolePunch:
    def _client(self, node: FakeP2PNode) -> GossipClient:
        return GossipClient(
            p2p_node=node,
            self_libp2p_peer_id="12D3KooWSelf",
            peer_dead_debounce_s=0.0,
            hole_punch_debounce_s=0.0,
        )

    def test_non_relay_peer_not_published(self):
        node = FakeP2PNode()
        client = self._client(node)
        assert maybe_request_hole_punch(
            client,
            self_libp2p_peer_id="12D3KooWSelf",
            peer=_peer(libp2p="T"),
        ) is False
        assert node.published() == []

    def test_requires_relay_peer_published(self):
        node = FakeP2PNode()
        client = self._client(node)
        assert maybe_request_hole_punch(
            client,
            self_libp2p_peer_id="12D3KooWSelf",
            peer=_peer(libp2p="T", requires_relay=True),
        ) is True
        assert len(node.published()) == 1

    def test_relay_address_only_published(self):
        node = FakeP2PNode()
        client = self._client(node)
        assert maybe_request_hole_punch(
            client,
            self_libp2p_peer_id="12D3KooWSelf",
            peer=_peer(
                libp2p="T",
                relay_address="/ip4/45.79.190.172/tcp/4001/p2p-circuit",
            ),
        ) is True


# --- attach_hole_punch_responder --------------------------------------------

class TestHolePunchResponder:
    def _client_with_responder(
        self,
        node: FakeP2PNode,
        *,
        self_id: str = "12D3KooWSelf",
    ) -> GossipClient:
        client = GossipClient(
            p2p_node=node,
            self_libp2p_peer_id=self_id,
            peer_dead_debounce_s=0.0,
            hole_punch_debounce_s=0.0,
        )
        attach_hole_punch_responder(
            client, p2p_node=node, self_libp2p_peer_id=self_id
        )
        return client

    def _enqueue_hole_punch(
        self,
        node: FakeP2PNode,
        *,
        from_peer: str,
        to_peer: str,
        sender_hop: str = "12D3KooWHop",
    ) -> None:
        env = {
            "type": EVENT_REQUEST_HOLE_PUNCH,
            "data": {"from_peer_id": from_peer, "to_peer_id": to_peer},
            "observed_by": from_peer,
            "unix_ms": int(time.time() * 1000),
        }
        node.enqueue_from(sender_hop, json.dumps(env).encode("utf-8"))

    def test_matching_request_triggers_dial(self):
        node = FakeP2PNode()
        client = self._client_with_responder(node, self_id="12D3KooWSelf")
        self._enqueue_hole_punch(node, from_peer="12D3KooWReq", to_peer="12D3KooWSelf")
        client.tick_once()
        assert node._dials == ["12D3KooWReq"]

    def test_non_matching_request_ignored(self):
        """If to_peer_id is someone else, we don't dial."""
        node = FakeP2PNode()
        client = self._client_with_responder(node, self_id="12D3KooWSelf")
        self._enqueue_hole_punch(
            node, from_peer="12D3KooWReq", to_peer="12D3KooWOther"
        )
        client.tick_once()
        assert node._dials == []

    def test_self_addressed_self_loop_ignored(self):
        """A malformed request where from == to == self is suppressed."""
        node = FakeP2PNode()
        client = self._client_with_responder(node, self_id="12D3KooWSelf")
        self._enqueue_hole_punch(
            node, from_peer="12D3KooWSelf", to_peer="12D3KooWSelf"
        )
        client.tick_once()
        assert node._dials == []

    def test_dial_failure_does_not_crash_dispatcher(self):
        node = FakeP2PNode(dial_fail=RuntimeError("no addresses for peer"))
        client = self._client_with_responder(node, self_id="12D3KooWSelf")
        self._enqueue_hole_punch(
            node, from_peer="12D3KooWReq", to_peer="12D3KooWSelf"
        )
        # Must not raise; dispatcher continues.
        client.tick_once()
        # Next (successful) request still dials.
        node.dial_fail = None
        self._enqueue_hole_punch(
            node, from_peer="12D3KooWReq2", to_peer="12D3KooWSelf"
        )
        client.tick_once()
        assert node._dials == ["12D3KooWReq2"]

    def test_from_peer_falls_back_to_propagation_source(self):
        """If the envelope is missing ``from_peer_id`` (older producer
        or the peer/server.py raw-publish path without the field), the
        responder falls back to the gossip propagation hop — which
        with flood_publish enabled is typically the original sender
        in a small mesh."""
        node = FakeP2PNode()
        client = self._client_with_responder(node, self_id="12D3KooWSelf")
        # Enqueue a REQUEST_HOLE_PUNCH with only ``to_peer_id``.
        env = {
            "type": EVENT_REQUEST_HOLE_PUNCH,
            "data": {"to_peer_id": "12D3KooWSelf"},  # no from_peer_id
            "observed_by": "12D3KooWRemote",
            "unix_ms": 0,
        }
        node.enqueue_from("12D3KooWRemote", json.dumps(env).encode("utf-8"))
        client.tick_once()
        assert node._dials == ["12D3KooWRemote"]

    def test_empty_self_id_logs_and_skips(self):
        """If the responder was wired before the P2P node finished
        starting (self_libp2p_peer_id empty), it must NOT crash — log
        and skip."""
        node = FakeP2PNode()
        client = self._client_with_responder(node, self_id="")  # empty
        self._enqueue_hole_punch(
            node, from_peer="12D3KooWReq", to_peer="12D3KooWSomeone"
        )
        # Must not raise.
        client.tick_once()
        assert node._dials == []

    def test_wrong_event_type_ignored(self):
        node = FakeP2PNode()
        client = self._client_with_responder(node, self_id="12D3KooWSelf")
        # Enqueue a PEER_DEAD — the responder must ignore it entirely.
        env = {
            "type": "PEER_DEAD",
            "data": {"libp2p_peer_id": "12D3KooWGhost"},
            "observed_by": "obs",
            "unix_ms": 0,
        }
        node.enqueue_from("hop", json.dumps(env).encode("utf-8"))
        client.tick_once()
        assert node._dials == []

    def test_end_to_end_two_node_rendezvous(self):
        """Simulate the full rendezvous: Mac publishes REQUEST_HOLE_PUNCH
        aimed at GPU1; GPU1's responder dials Mac. We fake the gossip
        mesh as a zero-latency passthrough between two FakeP2PNodes."""
        mac_node = FakeP2PNode()
        gpu_node = FakeP2PNode()
        mac_client = GossipClient(
            p2p_node=mac_node,
            self_libp2p_peer_id="12D3KooWMac",
            peer_dead_debounce_s=0.0,
            hole_punch_debounce_s=0.0,
        )
        gpu_client = GossipClient(
            p2p_node=gpu_node,
            self_libp2p_peer_id="12D3KooWGpu",
            peer_dead_debounce_s=0.0,
            hole_punch_debounce_s=0.0,
        )
        attach_hole_punch_responder(
            gpu_client, p2p_node=gpu_node, self_libp2p_peer_id="12D3KooWGpu"
        )

        # Mac publishes. Pretend the gossip mesh delivers it to GPU1 by
        # copying the bytes into gpu's inbound queue.
        assert maybe_request_hole_punch(
            mac_client,
            self_libp2p_peer_id="12D3KooWMac",
            peer=_peer(libp2p="12D3KooWGpu", requires_relay=True),
        ) is True
        bytes_on_wire = mac_node._published[-1]
        gpu_node.enqueue_from("12D3KooWMac", bytes_on_wire)

        # GPU processes the inbound event; its responder dials Mac.
        gpu_client.tick_once()
        assert gpu_node._dials == ["12D3KooWMac"]

        # Real production requires the reverse-direction dial too —
        # Mac also needs to dial GPU. In production this happens because
        # the InferenceChain invoked maybe_request_hole_punch just
        # *before* Forward() — so the Mac's gRPC client attempts a
        # dial at the same moment the GPU's dial_peer fires. We simulate
        # that co-ordination here by asserting both sides took action
        # within one mock gossip round-trip.
        # (Mac side dial is implicit in its gRPC Forward call; the
        # responder fires the complementary dial.)
