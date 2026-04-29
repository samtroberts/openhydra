# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Phase 2b §7 — manual layer-range override (--layers) validator tests.

Locks the two correctness invariants:
    1. All-or-nothing: any peer using --layers means all must.
    2. Union covers [0, total_layers) exactly once: no gap, no overlap.

The validator refuses to start on any violation rather than silently
producing wrong inference output. These tests pin every error code in
the ManualShardingError taxonomy so a regression in the dispatcher
can't silently downgrade an error path.
"""

from __future__ import annotations

import pytest

from coordinator.manual_sharding import (
    ManualShardingError,
    ParsedLayerRange,
    _PeerRange,
    parse_layers_arg,
    validate_manual_sharding,
)


# ── parse_layers_arg ───────────────────────────────────────────────────

def test_parse_empty_returns_none_opt_out():
    assert parse_layers_arg("") is None
    assert parse_layers_arg("   ") is None
    assert parse_layers_arg(None) is None


def test_parse_valid_range():
    r = parse_layers_arg("0-12")
    assert r == ParsedLayerRange(start=0, end=12)
    assert r.width == 12


def test_parse_valid_range_with_whitespace():
    r = parse_layers_arg("  8 - 24  ")
    assert r == ParsedLayerRange(start=8, end=24)


def test_parse_layer_membership():
    r = parse_layers_arg("4-8")
    assert 4 in r and 7 in r
    assert 3 not in r and 8 not in r   # exclusive end


@pytest.mark.parametrize("bad", ["abc", "1", "1-", "-5", "1-2-3", "twelve-twenty"])
def test_parse_malformed_raises_with_code(bad):
    with pytest.raises(ManualShardingError) as exc:
        parse_layers_arg(bad)
    assert exc.value.code in {"malformed", "out_of_range_negative", "empty_range"}


def test_parse_negative_start_rejected():
    with pytest.raises(ManualShardingError) as exc:
        parse_layers_arg("-2-5")
    assert exc.value.code in {"malformed", "out_of_range_negative"}


def test_parse_empty_range_rejected():
    """end == start is a zero-width interval — rejected."""
    with pytest.raises(ManualShardingError) as exc:
        parse_layers_arg("12-12")
    assert exc.value.code == "empty_range"


def test_parse_inverted_range_rejected():
    with pytest.raises(ManualShardingError) as exc:
        parse_layers_arg("20-10")
    assert exc.value.code == "empty_range"


# ── validate_manual_sharding — happy paths ─────────────────────────────

def test_validate_all_auto_passes():
    """Empty manual configuration = use the auto-assigner. No error."""
    peers = [
        _PeerRange("p1", None),
        _PeerRange("p2", None),
        _PeerRange("p3", None),
    ]
    validate_manual_sharding(peers, total_layers=32)   # must not raise


def test_validate_perfect_split_passes():
    """3 peers, contiguous, covers [0, 32) — happy path."""
    peers = [
        _PeerRange("p1", ParsedLayerRange(0, 12)),
        _PeerRange("p2", ParsedLayerRange(12, 24)),
        _PeerRange("p3", ParsedLayerRange(24, 32)),
    ]
    validate_manual_sharding(peers, total_layers=32)


def test_validate_asymmetric_split_passes():
    """Topology B's stage-0 takes fewer layers (also hosts the
    drafter). 0-8 / 8-24 / 24-32 = 8 / 16 / 8."""
    peers = [
        _PeerRange("stage0", ParsedLayerRange(0, 8)),
        _PeerRange("stage1", ParsedLayerRange(8, 24)),
        _PeerRange("stage2", ParsedLayerRange(24, 32)),
    ]
    validate_manual_sharding(peers, total_layers=32)


def test_validate_unsorted_input_passes():
    """Peers can announce in any order; the validator sorts internally."""
    peers = [
        _PeerRange("p3", ParsedLayerRange(24, 32)),
        _PeerRange("p1", ParsedLayerRange(0, 12)),
        _PeerRange("p2", ParsedLayerRange(12, 24)),
    ]
    validate_manual_sharding(peers, total_layers=32)


def test_validate_tuples_accepted():
    """Convenience: caller may pass plain (peer_id, range) tuples."""
    validate_manual_sharding(
        [
            ("p1", ParsedLayerRange(0, 16)),
            ("p2", ParsedLayerRange(16, 32)),
        ],
        total_layers=32,
    )


# ── validate_manual_sharding — failure modes ───────────────────────────

def test_validate_partial_adoption_rejected():
    """Mixing manual and auto = correctness bug. Refuse to start."""
    peers = [
        _PeerRange("p1", ParsedLayerRange(0, 12)),
        _PeerRange("p2", None),                       # auto
        _PeerRange("p3", ParsedLayerRange(24, 32)),
    ]
    with pytest.raises(ManualShardingError) as exc:
        validate_manual_sharding(peers, total_layers=32)
    assert exc.value.code == "partial_adoption"


def test_validate_gap_rejected():
    """Layers 12..15 unassigned — three peers don't cover [0, 32)."""
    peers = [
        _PeerRange("p1", ParsedLayerRange(0, 12)),
        _PeerRange("p2", ParsedLayerRange(16, 24)),   # gap: 12-16
        _PeerRange("p3", ParsedLayerRange(24, 32)),
    ]
    with pytest.raises(ManualShardingError) as exc:
        validate_manual_sharding(peers, total_layers=32)
    assert exc.value.code == "gap"
    assert "12" in str(exc.value)


def test_validate_overlap_rejected():
    """Two peers both claim layers 8..11 — would double-process."""
    peers = [
        _PeerRange("p1", ParsedLayerRange(0, 12)),
        _PeerRange("p2", ParsedLayerRange(8, 24)),    # overlap 8-12
        _PeerRange("p3", ParsedLayerRange(24, 32)),
    ]
    with pytest.raises(ManualShardingError) as exc:
        validate_manual_sharding(peers, total_layers=32)
    assert exc.value.code == "overlap"


def test_validate_short_rejected():
    """Union ends at 28 but model has 32 layers — last 4 unassigned."""
    peers = [
        _PeerRange("p1", ParsedLayerRange(0, 14)),
        _PeerRange("p2", ParsedLayerRange(14, 28)),   # short: missing 28-32
    ]
    with pytest.raises(ManualShardingError) as exc:
        validate_manual_sharding(peers, total_layers=32)
    assert exc.value.code == "short"


def test_validate_out_of_range_rejected():
    """Peer claims layer 40 but model has only 32 layers."""
    peers = [
        _PeerRange("p1", ParsedLayerRange(0, 16)),
        _PeerRange("p2", ParsedLayerRange(16, 40)),   # 40 > 32
    ]
    with pytest.raises(ManualShardingError) as exc:
        validate_manual_sharding(peers, total_layers=32)
    assert exc.value.code == "out_of_range"


def test_validate_total_layers_must_be_positive():
    """Defence in depth — total_layers=0 is meaningless."""
    with pytest.raises(ManualShardingError) as exc:
        validate_manual_sharding(
            [_PeerRange("p1", ParsedLayerRange(0, 1))],
            total_layers=0,
        )
    assert exc.value.code == "out_of_range"


def test_validate_empty_peer_list_no_op():
    """No peers announced yet = nothing to validate. Coord rejects
    inference for unrelated reasons; this validator stays out of it."""
    validate_manual_sharding([], total_layers=32)   # must not raise
