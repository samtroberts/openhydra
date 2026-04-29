# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b — swarm event envelope tests.

Locks the wire format so a future schema change can't silently
break peers running mixed Phase 2b versions. Covers:

* Round-trip for all three event types.
* Schema-version forward-compat (peer drops events with v > 1
  rather than mis-parsing them).
* Every error code in EventDecodeError's reason taxonomy.
* InMemorySwarmEventBus pub/sub correctness and isolation across
  handler errors.
"""

from __future__ import annotations

import json

import pytest

from coordinator.swarm_events import (
    EVENT_TYPE_PROMOTE_DRAFTER,
    EVENT_TYPE_REGISTER_DRAFT_MODEL,
    EVENT_TYPE_VERIFY_RESULT,
    EventDecodeError,
    InMemorySwarmEventBus,
    PromoteDrafter,
    RegisterDraftModel,
    SCHEMA_VERSION,
    SwarmEvent,
    VerifyResult,
    decode_event,
    encode_event,
)


# ── VerifyResult round-trip ────────────────────────────────────────────

def test_verify_result_roundtrip():
    payload = VerifyResult(
        session_id="sess-42",
        accepted_len=8,
        bonus_token=12345,
        kv_rollback_to=128,
        block_index=3,
    )
    wire = encode_event(payload, from_peer="12D3KooWStage0")
    event = decode_event(wire)
    assert event.type == EVENT_TYPE_VERIFY_RESULT
    assert event.payload == payload
    assert event.from_peer == "12D3KooWStage0"
    assert event.unix_ms > 0


def test_verify_result_validates_negative_accepted_len():
    with pytest.raises(EventDecodeError) as exc:
        VerifyResult(
            session_id="s", accepted_len=-1, bonus_token=0,
            kv_rollback_to=0, block_index=0,
        ).validate()
    assert exc.value.reason == "data_invalid"


def test_verify_result_validates_empty_session_id():
    with pytest.raises(EventDecodeError) as exc:
        VerifyResult(
            session_id="", accepted_len=0, bonus_token=0,
            kv_rollback_to=0, block_index=0,
        ).validate()
    assert exc.value.reason == "data_invalid"


# ── RegisterDraftModel round-trip ──────────────────────────────────────

def test_register_draft_model_roundtrip():
    payload = RegisterDraftModel(
        target_path="Qwen/Qwen3.5-4B",
        draft_path="z-lab/Qwen3.5-4B-DFlash",
        block_size=16,
        backend="mlx",
    )
    wire = encode_event(payload, from_peer="coord-libp2p-id")
    event = decode_event(wire)
    assert event.type == EVENT_TYPE_REGISTER_DRAFT_MODEL
    assert event.payload == payload


def test_register_draft_model_validates_block_size_range():
    for bs in (0, 33, -1, 100):
        with pytest.raises(EventDecodeError, match="block_size"):
            RegisterDraftModel(
                target_path="x", draft_path="y", block_size=bs, backend="mlx",
            ).validate()


def test_register_draft_model_validates_backend():
    with pytest.raises(EventDecodeError, match="backend"):
        RegisterDraftModel(
            target_path="x", draft_path="y", block_size=16, backend="tensorrt",
        ).validate()


# ── PromoteDrafter round-trip ──────────────────────────────────────────

def test_promote_drafter_roundtrip():
    payload = PromoteDrafter(
        from_peer_id="12D3KooWCoord",
        to_peer_id="12D3KooWStage0",
        unix_ms=1745000000000,
    )
    wire = encode_event(payload, from_peer="12D3KooWStage0")
    event = decode_event(wire)
    assert event.type == EVENT_TYPE_PROMOTE_DRAFTER
    assert event.payload == payload


def test_promote_drafter_validates_unix_ms_positive():
    with pytest.raises(EventDecodeError, match="unix_ms"):
        PromoteDrafter(
            from_peer_id="a", to_peer_id="b", unix_ms=0,
        ).validate()


# ── Decode error taxonomy ──────────────────────────────────────────────

def test_decode_malformed_json():
    with pytest.raises(EventDecodeError) as exc:
        decode_event(b"\xff not json")
    assert exc.value.reason == "malformed_json"


def test_decode_missing_field():
    raw = json.dumps({"type": "verify_result"}).encode("utf-8")
    with pytest.raises(EventDecodeError) as exc:
        decode_event(raw)
    assert exc.value.reason == "missing_field"


def test_decode_unknown_type():
    raw = json.dumps({
        "type": "totally_made_up", "v": 1, "data": {},
    }).encode("utf-8")
    with pytest.raises(EventDecodeError) as exc:
        decode_event(raw)
    assert exc.value.reason == "unknown_type"


def test_decode_future_schema():
    """Forward-compat: peer running v=1 must drop v=999 cleanly,
    not mis-parse it as v=1 with extra fields."""
    raw = json.dumps({
        "type": EVENT_TYPE_VERIFY_RESULT, "v": 999,
        "data": {
            "session_id": "s", "accepted_len": 0, "bonus_token": 0,
            "kv_rollback_to": 0, "block_index": 0,
        },
    }).encode("utf-8")
    with pytest.raises(EventDecodeError) as exc:
        decode_event(raw)
    assert exc.value.reason == "future_schema"
    assert "999" in str(exc.value)


def test_decode_data_not_dict():
    raw = json.dumps({
        "type": EVENT_TYPE_VERIFY_RESULT, "v": 1, "data": "not a dict",
    }).encode("utf-8")
    with pytest.raises(EventDecodeError) as exc:
        decode_event(raw)
    assert exc.value.reason == "data_invalid"


def test_decode_extra_data_fields_rejected():
    """Extra fields in the data dict signal a wire-format mismatch
    we should fail on, not silently ignore."""
    raw = json.dumps({
        "type": EVENT_TYPE_VERIFY_RESULT, "v": 1, "data": {
            "session_id": "s", "accepted_len": 0, "bonus_token": 0,
            "kv_rollback_to": 0, "block_index": 0,
            "rogue_field_from_v2": "uh oh",
        },
    }).encode("utf-8")
    with pytest.raises(EventDecodeError) as exc:
        decode_event(raw)
    assert exc.value.reason == "data_invalid"


def test_decode_accepts_str_input():
    raw = encode_event(
        VerifyResult(
            session_id="s", accepted_len=1, bonus_token=2,
            kv_rollback_to=3, block_index=4,
        ),
        from_peer="p",
    )
    # bytes form...
    assert decode_event(raw).type == EVENT_TYPE_VERIFY_RESULT
    # ...str form...
    assert decode_event(raw.decode("utf-8")).type == EVENT_TYPE_VERIFY_RESULT
    # ...dict form (already-parsed).
    parsed = json.loads(raw.decode("utf-8"))
    assert decode_event(parsed).type == EVENT_TYPE_VERIFY_RESULT


# ── InMemorySwarmEventBus ──────────────────────────────────────────────

def test_bus_publish_subscribe_basic():
    bus = InMemorySwarmEventBus()
    received: list[SwarmEvent] = []
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, received.append)
    bus.publish(
        VerifyResult(
            session_id="s1", accepted_len=4, bonus_token=99,
            kv_rollback_to=128, block_index=0,
        ),
        from_peer="coord",
    )
    assert len(received) == 1
    assert received[0].payload.session_id == "s1"


def test_bus_routes_only_matching_event_type():
    bus = InMemorySwarmEventBus()
    verify_received: list[SwarmEvent] = []
    promote_received: list[SwarmEvent] = []
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, verify_received.append)
    bus.subscribe(EVENT_TYPE_PROMOTE_DRAFTER, promote_received.append)
    bus.publish(
        VerifyResult(
            session_id="s", accepted_len=0, bonus_token=0,
            kv_rollback_to=0, block_index=0,
        ),
        from_peer="p",
    )
    assert len(verify_received) == 1
    assert len(promote_received) == 0


def test_bus_unsubscribe_removes_handler():
    bus = InMemorySwarmEventBus()
    received: list[SwarmEvent] = []
    unsub = bus.subscribe(EVENT_TYPE_VERIFY_RESULT, received.append)
    bus.publish(
        VerifyResult(session_id="s", accepted_len=0, bonus_token=0,
                     kv_rollback_to=0, block_index=0),
        from_peer="p",
    )
    unsub()
    bus.publish(
        VerifyResult(session_id="s", accepted_len=0, bonus_token=0,
                     kv_rollback_to=0, block_index=0),
        from_peer="p",
    )
    assert len(received) == 1   # second publish reaches no handler


def test_bus_multiple_subscribers_all_receive():
    bus = InMemorySwarmEventBus()
    a: list = []
    b: list = []
    c: list = []
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, a.append)
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, b.append)
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, c.append)
    bus.publish(
        VerifyResult(session_id="s", accepted_len=0, bonus_token=0,
                     kv_rollback_to=0, block_index=0),
        from_peer="p",
    )
    assert len(a) == len(b) == len(c) == 1


def test_bus_handler_error_isolated_from_other_handlers():
    """One handler crashing must not stop other handlers from
    receiving the same event."""
    bus = InMemorySwarmEventBus()
    boom_received: list = []
    other_received: list = []

    def angry(event):
        boom_received.append(event)
        raise RuntimeError("synthetic")

    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, angry)
    bus.subscribe(EVENT_TYPE_VERIFY_RESULT, other_received.append)
    # Publish must not raise.
    bus.publish(
        VerifyResult(session_id="s", accepted_len=0, bonus_token=0,
                     kv_rollback_to=0, block_index=0),
        from_peer="p",
    )
    assert len(boom_received) == 1
    assert len(other_received) == 1


def test_bus_subscribe_unknown_type_rejected():
    bus = InMemorySwarmEventBus()
    with pytest.raises(ValueError, match="unknown event_type"):
        bus.subscribe("nope", lambda e: None)


def test_bus_unsubscribe_idempotent():
    bus = InMemorySwarmEventBus()
    unsub = bus.subscribe(EVENT_TYPE_VERIFY_RESULT, lambda e: None)
    unsub()
    unsub()   # second call must not raise
