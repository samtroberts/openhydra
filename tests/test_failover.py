# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b — DraftModelRegistry + FailoverManager tests.

The failover surface is the difference between "stage-0 promotes
seamlessly" and "the swarm hangs when coord crashes mid-generation."
These tests pin every concrete invariant.
"""

from __future__ import annotations

import pytest

from coordinator.failover import (
    DraftModelRegistry,
    FailoverError,
    FailoverManager,
)
from coordinator.swarm_events import (
    EVENT_TYPE_PROMOTE_DRAFTER,
    EVENT_TYPE_REGISTER_DRAFT_MODEL,
    InMemorySwarmEventBus,
    PromoteDrafter,
    RegisterDraftModel,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def bus():
    return InMemorySwarmEventBus()


@pytest.fixture
def spec():
    return RegisterDraftModel(
        target_path="Qwen/Qwen3.5-4B",
        draft_path="z-lab/Qwen3.5-4B-DFlash",
        block_size=16,
        backend="mlx",
    )


# ── DraftModelRegistry ─────────────────────────────────────────────────


def test_registry_starts_empty(bus):
    reg = DraftModelRegistry(bus)
    assert reg.get_active_spec() is None


def test_registry_records_announcement(bus, spec):
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord-id")
    out = reg.get_active_spec()
    assert out == spec
    snap = reg.get_active_announcement()
    assert snap is not None
    assert snap.from_peer == "coord-id"


def test_registry_later_unix_ms_wins(bus, spec):
    """Two peers register simultaneously: later timestamp wins.
    Pre-encoded events sent in reverse-chronological order — the
    registry must still end up with the newer one."""
    reg = DraftModelRegistry(bus)
    later = RegisterDraftModel(
        target_path="Qwen/Qwen3.5-4B",
        draft_path="z-lab/Qwen3.5-4B-DFlash-v2",
        block_size=16,
        backend="mlx",
    )
    # Publish "later" then "earlier" — the InMemoryBus stamps both
    # with current wall-clock time but we override via direct
    # encode + decode.
    bus.publish(later, from_peer="coord-A")
    # Synthesise an earlier-timestamp event by lying about unix_ms.
    from coordinator.swarm_events import encode_event, decode_event
    earlier_wire = encode_event(spec, from_peer="coord-B", unix_ms=1)
    earlier_event = decode_event(earlier_wire)
    # Manually invoke the registry's handler with the synthetic event.
    reg._on_event(earlier_event)
    assert reg.get_active_spec() == later


def test_registry_unknown_payload_type_ignored(bus, spec):
    """Defence in depth — if the bus erroneously routed a
    PromoteDrafter to the registry's handler, the registry skips it
    silently rather than crashing."""
    reg = DraftModelRegistry(bus)
    # Build a synthetic event with the wrong payload type but routed
    # to the registry's _on_event handler.
    from coordinator.swarm_events import SwarmEvent
    wrong = SwarmEvent(
        type=EVENT_TYPE_PROMOTE_DRAFTER,
        payload=PromoteDrafter("a", "b", 1),
        from_peer="x",
        unix_ms=1,
    )
    reg._on_event(wrong)   # must not raise
    assert reg.get_active_spec() is None


def test_registry_close_unsubscribes(bus, spec):
    reg = DraftModelRegistry(bus)
    reg.close()
    bus.publish(spec, from_peer="late")
    # After close, new events are not picked up.
    assert reg.get_active_spec() is None
    # close is idempotent.
    reg.close()


def test_registry_announce_validates_spec(bus):
    reg = DraftModelRegistry(bus)
    bad = RegisterDraftModel(
        target_path="x", draft_path="y",
        block_size=99,   # out of range
        backend="mlx",
    )
    with pytest.raises(Exception):
        reg.announce(bad, from_peer="me")


# ── FailoverManager — basic semantics ──────────────────────────────────


def test_failover_requires_local_peer_id(bus, spec):
    reg = DraftModelRegistry(bus)
    with pytest.raises(ValueError, match="local_peer_id"):
        FailoverManager(
            bus=bus, local_peer_id="", registry=reg,
        )


def test_failover_promote_without_spec_raises(bus):
    """No RegisterDraftModel event → cannot know what to load → fail loud."""
    reg = DraftModelRegistry(bus)
    fm = FailoverManager(bus=bus, local_peer_id="me", registry=reg)
    with pytest.raises(FailoverError) as exc:
        fm.promote()
    assert exc.value.code == "no_active_spec"


def test_failover_promote_emits_event(bus, spec):
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord")
    fm = FailoverManager(bus=bus, local_peer_id="stage0", registry=reg)
    assert fm.promote(now_ms=1_000_000) is True
    # Active drafter is now the local peer.
    assert fm.active_drafter_id == "stage0"
    assert fm.is_local_active is True


def test_failover_promote_event_carries_correct_fields(bus, spec):
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord-id")
    fm = FailoverManager(bus=bus, local_peer_id="stage0", registry=reg)

    received = []
    bus.subscribe(EVENT_TYPE_PROMOTE_DRAFTER, received.append)

    fm.promote(now_ms=42_000)
    assert len(received) == 1
    payload = received[0].payload
    assert payload.from_peer_id == "coord-id"   # the previous registrar
    assert payload.to_peer_id == "stage0"
    assert payload.unix_ms == 42_000


# ── FailoverManager — coord-absence detection ─────────────────────────


def test_failover_check_coord_alive_no_promotion(bus, spec):
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord")
    fm = FailoverManager(
        bus=bus, local_peer_id="stage0", registry=reg,
        coord_alive=lambda: True,
    )
    assert fm.check_coord(now_ms=1000) is False
    assert fm.active_drafter_id is None


def test_failover_check_coord_brief_absence_below_threshold(bus, spec):
    """A short blip does not trigger promotion — only sustained
    absence over the threshold does."""
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord")
    alive = [False]
    fm = FailoverManager(
        bus=bus, local_peer_id="stage0", registry=reg,
        coord_alive=lambda: alive[0],
        absence_threshold_ms=5_000,
    )
    # First sample at t=1000: starts the absence streak.
    assert fm.check_coord(now_ms=1000) is False
    # Coord recovers at t=1500.
    alive[0] = True
    assert fm.check_coord(now_ms=1500) is False
    # Coord absent again at t=2000: NEW streak starts; promotion
    # would only happen if it stays absent through t=7000.
    alive[0] = False
    assert fm.check_coord(now_ms=2000) is False
    assert fm.check_coord(now_ms=3000) is False
    # Total absence in the new streak: 1000ms < 5000ms threshold.
    assert fm.active_drafter_id is None


def test_failover_check_coord_sustained_absence_promotes(bus, spec):
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord")
    fm = FailoverManager(
        bus=bus, local_peer_id="stage0", registry=reg,
        coord_alive=lambda: False,
        absence_threshold_ms=5_000,
    )
    # First sample starts the streak.
    assert fm.check_coord(now_ms=1000) is False
    # Sample inside the threshold: still no promote.
    assert fm.check_coord(now_ms=4000) is False
    # Sample past the threshold: promote.
    assert fm.check_coord(now_ms=7000) is True
    assert fm.active_drafter_id == "stage0"


def test_failover_check_coord_resets_streak_on_recovery(bus, spec):
    """After promotion, if the coord is somehow seen alive again,
    the streak resets — a future absence requires the full
    threshold to re-promote."""
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord")
    alive = [False]
    fm = FailoverManager(
        bus=bus, local_peer_id="stage0", registry=reg,
        coord_alive=lambda: alive[0],
        absence_threshold_ms=1_000,
    )
    fm.check_coord(now_ms=0)
    fm.check_coord(now_ms=1500)   # promotes
    assert fm.is_local_active

    # Coord recovers; streak resets.
    alive[0] = True
    fm.check_coord(now_ms=2000)
    # Coord absent again briefly.
    alive[0] = False
    assert fm.check_coord(now_ms=2100) is False
    # Just under threshold from the new streak start (2100).
    assert fm.check_coord(now_ms=3050) is False
    # Past the new threshold: re-promote.
    assert fm.check_coord(now_ms=3200) is True


# ── FailoverManager — incoming PromoteDrafter handling ────────────────


def test_failover_records_incoming_promote(bus, spec):
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord")
    fm = FailoverManager(bus=bus, local_peer_id="me", registry=reg)
    bus.publish(
        PromoteDrafter(from_peer_id="coord", to_peer_id="other-stage0",
                       unix_ms=5_000),
        from_peer="other-stage0",
    )
    assert fm.active_drafter_id == "other-stage0"
    assert fm.is_local_active is False


def test_failover_later_unix_ms_wins_on_promote_race(bus, spec):
    """Two peers race to promote. The one with the LATER unix_ms
    wins — gives operators a deterministic tiebreaker."""
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord")
    fm = FailoverManager(bus=bus, local_peer_id="me", registry=reg)

    bus.publish(
        PromoteDrafter("coord", "peer-A", unix_ms=10_000),
        from_peer="peer-A",
    )
    assert fm.active_drafter_id == "peer-A"

    # Later promote from peer-B.
    bus.publish(
        PromoteDrafter("coord", "peer-B", unix_ms=20_000),
        from_peer="peer-B",
    )
    assert fm.active_drafter_id == "peer-B"


def test_failover_drops_stale_promote(bus, spec):
    """An incoming PromoteDrafter with unix_ms older than the
    current active drafter's timestamp is dropped — late-arriving
    duplicate would otherwise reset the active drafter to a
    no-longer-correct peer."""
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord")
    fm = FailoverManager(bus=bus, local_peer_id="me", registry=reg)

    bus.publish(
        PromoteDrafter("coord", "peer-B", unix_ms=20_000),
        from_peer="peer-B",
    )
    # Late-arriving older event from peer-A.
    bus.publish(
        PromoteDrafter("coord", "peer-A", unix_ms=10_000),
        from_peer="peer-A",
    )
    assert fm.active_drafter_id == "peer-B"   # unchanged


def test_failover_close_unsubscribes(bus, spec):
    reg = DraftModelRegistry(bus)
    reg.announce(spec, from_peer="coord")
    fm = FailoverManager(bus=bus, local_peer_id="me", registry=reg)
    fm.close()
    bus.publish(
        PromoteDrafter("coord", "other", unix_ms=1),
        from_peer="other",
    )
    # After close, no longer tracking.
    assert fm.active_drafter_id is None
    fm.close()   # idempotent


# ── End-to-end: registry + manager wired together ─────────────────────


def test_failover_end_to_end_topology_a_crash():
    """Topology A: coord registers spec + drafts. Coord crashes.
    Stage-0 detects absence, promotes, takes over. Other peers
    record stage-0 as new drafter."""
    bus = InMemorySwarmEventBus()

    # Coord side: announce the spec on startup.
    coord_reg = DraftModelRegistry(bus)
    coord_reg.announce(
        RegisterDraftModel(
            target_path="Qwen/Qwen3.5-4B",
            draft_path="z-lab/Qwen3.5-4B-DFlash",
            block_size=16, backend="mlx",
        ),
        from_peer="coord",
    )

    # Stage-0 side: subscribe to events, set up failover manager.
    stage0_reg = DraftModelRegistry(bus)
    # Stage-0 registry doesn't see coord's pre-subscription publish
    # (in-memory bus has no replay), so re-announce for stage-0's
    # benefit. Real libp2p deployments use the historical-event
    # replay; we model it by re-broadcasting the stored coord spec.
    stage0_reg.announce(coord_reg.get_active_spec(), from_peer="coord")

    # Other peers also track active drafter.
    other_fm = FailoverManager(
        bus=bus, local_peer_id="other", registry=stage0_reg,
        coord_alive=lambda: True,   # 'other' doesn't promote
    )

    # Stage-0's manager: coord is absent.
    stage0_fm = FailoverManager(
        bus=bus, local_peer_id="stage0", registry=stage0_reg,
        coord_alive=lambda: False,
        absence_threshold_ms=1_000,
    )

    # Simulate the absence-detect loop.
    stage0_fm.check_coord(now_ms=0)
    promoted = stage0_fm.check_coord(now_ms=1_500)
    assert promoted is True

    # Stage-0 sees itself as active drafter.
    assert stage0_fm.is_local_active

    # Other peers also see stage-0 as the active drafter.
    assert other_fm.active_drafter_id == "stage0"
