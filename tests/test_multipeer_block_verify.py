# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b live-bench Binding #2 — multi-peer block-verify tests.

Pins every wire-level invariant of the new block-verify round trip:

  * Proto fields ``ForwardResponse.block_size`` / ``block_index`` and
    ``ForwardRequest.block_index`` round-trip.
  * Field numbers locked (28, 29 on response; 60 on request).
  * ``register_dflash_block`` / ``emit_dflash_block_response`` /
    ``unregister_dflash_block`` queue lifecycle.
  * Late arrivals (no queue registered) drop silently rather than
    queue indefinitely.
  * ``MultiPeerRingVerifyTransport`` registers, fires, blocks, drops
    the queue in ``finally``.
  * Coord-side PushResult handler routes ``block_size > 0`` to the
    block-verify queue (not the per-token sampler).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import pytest


# ── Proto field round-trip + tag stability ─────────────────────────────


def test_forward_response_block_size_roundtrip():
    from peer import peer_pb2
    resp = peer_pb2.ForwardResponse(block_size=17, block_index=3)
    wire = resp.SerializeToString()
    restored = peer_pb2.ForwardResponse()
    restored.ParseFromString(wire)
    assert restored.block_size == 17
    assert restored.block_index == 3


def test_forward_request_block_index_roundtrip():
    from peer import peer_pb2
    req = peer_pb2.ForwardRequest(block_index=42)
    wire = req.SerializeToString()
    restored = peer_pb2.ForwardRequest()
    restored.ParseFromString(wire)
    assert restored.block_index == 42


def test_block_verify_proto_field_numbers_stable():
    """Tag numbers locked. Changing them breaks wire compat with peers
    running the old build."""
    from peer import peer_pb2
    req_tags = {f.name: f.number for f in peer_pb2.ForwardRequest.DESCRIPTOR.fields}
    resp_tags = {f.name: f.number for f in peer_pb2.ForwardResponse.DESCRIPTOR.fields}
    assert req_tags["block_index"] == 60
    assert resp_tags["block_size"] == 28
    assert resp_tags["block_index"] == 29


def test_block_verify_response_defaults_zero():
    """A non-block-verify response must default to block_size=0,
    block_index=0 — that's how the coord-side PushResult handler
    distinguishes it from a block-verify response."""
    from peer import peer_pb2
    resp = peer_pb2.ForwardResponse()
    assert resp.block_size == 0
    assert resp.block_index == 0


# ── push_receiver block-verify queue lifecycle ──────────────────────────


def test_register_emit_unregister_block_queue_happy_path():
    from coordinator.push_receiver import (
        emit_dflash_block_response,
        register_dflash_block,
        unregister_dflash_block,
    )

    q = register_dflash_block("req-1", 0)
    assert emit_dflash_block_response(
        request_id="req-1", block_index=0,
        activation_packed=b"\x01\x02\x03", block_size=17,
    ) is True
    outcome = q.get(timeout=0.1)
    assert outcome[0] == "ok"
    assert outcome[1] == b"\x01\x02\x03"
    assert outcome[2] == 17
    unregister_dflash_block("req-1", 0)


def test_emit_block_response_returns_false_when_no_queue():
    """Late arrival or stray response: drops silently, returns False."""
    from coordinator.push_receiver import emit_dflash_block_response
    assert emit_dflash_block_response(
        request_id="not-registered", block_index=999,
        activation_packed=b"x", block_size=1,
    ) is False


def test_emit_block_error_propagates():
    from coordinator.push_receiver import (
        emit_dflash_block_error, register_dflash_block,
        unregister_dflash_block,
    )
    q = register_dflash_block("req-2", 5)
    assert emit_dflash_block_error("req-2", 5, "synthetic") is True
    outcome = q.get(timeout=0.1)
    assert outcome[0] == "err"
    assert outcome[1] == "synthetic"
    unregister_dflash_block("req-2", 5)


def test_unregister_session_drops_all_blocks_for_request():
    from coordinator.push_receiver import (
        register_dflash_block, unregister_dflash_session,
        emit_dflash_block_response,
    )
    register_dflash_block("req-A", 0)
    register_dflash_block("req-A", 1)
    register_dflash_block("req-A", 2)
    register_dflash_block("req-B", 0)
    dropped = unregister_dflash_session("req-A")
    assert dropped == 3
    # req-A queues no longer accept emits.
    assert emit_dflash_block_response("req-A", 0, b"x", 1) is False
    # req-B is unaffected.
    assert emit_dflash_block_response("req-B", 0, b"x", 1) is True
    unregister_dflash_session("req-B")


def test_block_queue_idempotent_register():
    """Registering the same (req, idx) twice replaces the queue —
    useful for retry-after-timeout. The OLD queue's blocked get()
    times out; the NEW queue receives the response."""
    from coordinator.push_receiver import (
        register_dflash_block, emit_dflash_block_response,
        unregister_dflash_block,
    )
    q1 = register_dflash_block("req-3", 0)
    q2 = register_dflash_block("req-3", 0)   # replaces q1
    assert emit_dflash_block_response("req-3", 0, b"y", 5) is True
    outcome = q2.get(timeout=0.1)
    assert outcome[0] == "ok"
    # q1 saw nothing.
    import queue as _q
    with pytest.raises(_q.Empty):
        q1.get(timeout=0.05)
    unregister_dflash_block("req-3", 0)


# ── MultiPeerRingVerifyTransport ───────────────────────────────────────


class _FakeChain:
    """Records run_push_ring calls; simulates the verify ring trip
    by spawning a thread that emits a block-verify response after a
    short delay."""

    def __init__(
        self, *, packed: bytes, block_size: int,
        delay_s: float = 0.01, error: str | None = None,
    ):
        self._packed = packed
        self._block_size = block_size
        self._delay_s = delay_s
        self._error = error
        self.calls: list[dict] = []

    def run_push_ring(self, **kwargs):
        self.calls.append(dict(kwargs))
        # Schedule the response.
        from coordinator.push_receiver import (
            emit_dflash_block_error, emit_dflash_block_response,
        )
        request_id = kwargs.get("request_id", "")
        block_index = int(kwargs.get("block_index", 0) or 0)
        delay = self._delay_s
        packed = self._packed
        bs = self._block_size
        err = self._error

        def _delivery():
            time.sleep(delay)
            if err is not None:
                emit_dflash_block_error(request_id, block_index, err)
            else:
                emit_dflash_block_response(
                    request_id=request_id, block_index=block_index,
                    activation_packed=packed, block_size=bs,
                )
        threading.Thread(target=_delivery, daemon=True).start()


def test_multipeer_transport_round_trip():
    from coordinator.dflash_integration import MultiPeerRingVerifyTransport

    fake = _FakeChain(packed=b"hidden-state-bytes", block_size=17)
    transport = MultiPeerRingVerifyTransport(
        chain=fake, request_id="req-rt",
        kv_session_id="sess", callback_address="127.0.0.1:50051",
    )
    out = transport.verify(
        prefix_token_ids=[1, 2, 3],
        draft_token_ids=list(range(10, 26)),    # 16 drafts
        kv_rollback_to=3,
        request_id="req-rt", kv_session_id="sess",
    )
    assert isinstance(out, dict)
    assert out["packed_bytes"] == b"hidden-state-bytes"
    assert out["block_size"] == 17
    # Chain was fired with the right fields.
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["draft_block"] is True
    assert call["draft_token_ids"] == list(range(10, 26))
    assert call["block_index"] == 0
    assert call["kv_rollback_to"] == 3
    assert call["sample_on_coordinator"] is True


def test_multipeer_transport_increments_block_index():
    from coordinator.dflash_integration import MultiPeerRingVerifyTransport

    fake = _FakeChain(packed=b"x", block_size=17)
    transport = MultiPeerRingVerifyTransport(
        chain=fake, request_id="req-inc",
        kv_session_id="sess", callback_address="x",
    )
    transport.verify(
        prefix_token_ids=[1], draft_token_ids=[10] * 16,
        kv_rollback_to=1, request_id="req-inc", kv_session_id="sess",
    )
    transport.verify(
        prefix_token_ids=[1], draft_token_ids=[20] * 16,
        kv_rollback_to=2, request_id="req-inc", kv_session_id="sess",
    )
    assert fake.calls[0]["block_index"] == 0
    assert fake.calls[1]["block_index"] == 1


def test_multipeer_transport_propagates_errors():
    from coordinator.dflash_integration import (
        DFlashIntegrationError, MultiPeerRingVerifyTransport,
    )

    fake = _FakeChain(packed=b"", block_size=0, error="synthetic-fail")
    transport = MultiPeerRingVerifyTransport(
        chain=fake, request_id="req-err",
        kv_session_id="sess", callback_address="x",
    )
    with pytest.raises(DFlashIntegrationError) as exc:
        transport.verify(
            prefix_token_ids=[1], draft_token_ids=[10] * 16,
            kv_rollback_to=1, request_id="req-err", kv_session_id="sess",
        )
    assert "synthetic-fail" in str(exc.value)


def test_multipeer_transport_unregisters_queue_on_success():
    """Verify that the queue cleanup happens even on the happy path
    so we don't leak state across blocks."""
    from coordinator.dflash_integration import MultiPeerRingVerifyTransport
    from coordinator.push_receiver import emit_dflash_block_response

    fake = _FakeChain(packed=b"x", block_size=17)
    transport = MultiPeerRingVerifyTransport(
        chain=fake, request_id="req-clean",
        kv_session_id="sess", callback_address="x",
    )
    transport.verify(
        prefix_token_ids=[1], draft_token_ids=[10] * 16,
        kv_rollback_to=1, request_id="req-clean", kv_session_id="sess",
    )
    # Queue is gone — a stray emit returns False.
    assert emit_dflash_block_response(
        "req-clean", 0, b"late", 17,
    ) is False


def test_multipeer_transport_unregisters_queue_on_error():
    from coordinator.dflash_integration import (
        DFlashIntegrationError, MultiPeerRingVerifyTransport,
    )
    from coordinator.push_receiver import emit_dflash_block_response

    fake = _FakeChain(packed=b"", block_size=0, error="boom")
    transport = MultiPeerRingVerifyTransport(
        chain=fake, request_id="req-err-clean",
        kv_session_id="sess", callback_address="x",
    )
    try:
        transport.verify(
            prefix_token_ids=[1], draft_token_ids=[10] * 16,
            kv_rollback_to=1, request_id="req-err-clean",
            kv_session_id="sess",
        )
    except DFlashIntegrationError:
        pass
    # Queue cleaned up despite the error.
    assert emit_dflash_block_response(
        "req-err-clean", 0, b"late", 17,
    ) is False


def test_multipeer_transport_requires_chain():
    from coordinator.dflash_integration import MultiPeerRingVerifyTransport
    with pytest.raises(ValueError, match="chain"):
        MultiPeerRingVerifyTransport(
            chain=None, request_id="x",
            kv_session_id="", callback_address="",
        )


# ── Coord-side PushResult routing ──────────────────────────────────────


def test_push_result_handler_routes_block_verify_response():
    """A ForwardResponse with is_hidden_state=True AND block_size>0
    must route to the dflash queue, NOT trigger per-token sampling."""
    from peer import peer_pb2
    from peer.server import _coordinator_handle_push_result
    from coordinator.push_receiver import register_dflash_block

    q = register_dflash_block("req-rh", 7)
    response = peer_pb2.ForwardResponse(
        request_id="req-rh",
        peer_id="last-peer",
        is_hidden_state=True,
        activation_packed=b"hidden-payload",
        block_size=17,
        block_index=7,
    )
    ack = _coordinator_handle_push_result(response=response, p2p_node=None)
    assert ack.ok is True
    outcome = q.get(timeout=0.1)
    assert outcome[0] == "ok"
    assert outcome[1] == b"hidden-payload"
    assert outcome[2] == 17


def test_push_result_handler_rejects_empty_block_payload():
    from peer import peer_pb2
    from peer.server import _coordinator_handle_push_result
    from coordinator.push_receiver import register_dflash_block

    register_dflash_block("req-empty", 0)
    response = peer_pb2.ForwardResponse(
        request_id="req-empty",
        peer_id="last-peer",
        is_hidden_state=True,
        activation_packed=b"",
        block_size=17,
        block_index=0,
    )
    ack = _coordinator_handle_push_result(response=response, p2p_node=None)
    assert ack.ok is False
    assert "empty_payload" in ack.error


def test_push_result_handler_drops_block_response_with_no_queue():
    """Late arrival after timeout: ACK ok but warn; don't crash."""
    from peer import peer_pb2
    from peer.server import _coordinator_handle_push_result

    response = peer_pb2.ForwardResponse(
        request_id="req-stray",
        peer_id="late",
        is_hidden_state=True,
        activation_packed=b"data",
        block_size=17,
        block_index=99,
    )
    ack = _coordinator_handle_push_result(response=response, p2p_node=None)
    assert ack.ok is True


def test_push_result_handler_block_size_zero_falls_through_to_per_token():
    """A response with block_size=0 (default) must fall through to
    the existing per-token sampler path — Phase 2a behaviour."""
    from peer import peer_pb2
    from peer.server import _coordinator_handle_push_result

    # Without block_size > 0 routing, the handler reaches the
    # existing per-token sampler, which requires a registered
    # ring session. We assert the handler does NOT route to the
    # block queue (which would be wrong); behaviour under
    # missing session is "no_ring_session_registered" error,
    # confirming the per-token branch ran.
    response = peer_pb2.ForwardResponse(
        request_id="req-pertoken",
        peer_id="last",
        is_hidden_state=True,
        activation_packed=b"x",
        block_size=0,        # explicit
        block_index=0,
    )
    ack = _coordinator_handle_push_result(response=response, p2p_node=None)
    assert ack.ok is False
    assert ack.error in {
        "no_head_sampler_registered",
        "no_ring_session_registered",
    }
