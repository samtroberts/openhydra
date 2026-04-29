# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Phase 2b — manual layer-range override (`--layers`) parser + validator.

Phase 2b §7. Operators can pin a peer's layer range with
``--layers START-END`` (inclusive-exclusive) for asymmetric-sharding
benchmarks (Topology B's stage-0 wants fewer layers because it also
hosts the draft model).

Two correctness invariants enforced here:

1. **All-or-nothing.** When ANY peer in the swarm uses ``--layers``,
   ALL peers must. Mixing manual and automatic sharding silently
   loses correctness because the auto-assigner doesn't know about
   the manual pins.

2. **Union covers ``[0, total_layers)`` exactly once.** Gaps drop
   layers; overlaps double-process them. Either silently produces
   wrong output. Coord validates the union on first announce and
   refuses to start otherwise (rather than discovering the bug at
   inference time as a numerical mismatch in the head sampler).

Both invariants are enforced by ``validate_manual_sharding``. The
parser ``parse_layers_arg`` is separated so that the CLI bind step
can fail fast on malformed strings (e.g. ``"twelve-twenty"``) before
any peer state is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "ManualShardingError",
    "ParsedLayerRange",
    "parse_layers_arg",
    "validate_manual_sharding",
]


class ManualShardingError(ValueError):
    """Raised on any manual-sharding rule violation.

    Carries a structured ``code`` field so the coord startup logger
    can emit a stable taxonomy:

      * ``"malformed"``   — input string didn't parse.
      * ``"empty_range"`` — start >= end (zero or negative width).
      * ``"out_of_range"``— start < 0 or end > total_layers.
      * ``"partial_adoption"`` — some peers manual, others auto.
      * ``"gap"``         — union doesn't cover layer N.
      * ``"overlap"``     — two peers claim layer N.
      * ``"short"``       — union ends before total_layers.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ParsedLayerRange:
    """A parsed ``--layers START-END`` value.

    Inclusive ``start``, exclusive ``end``. ``end > start`` is
    enforced by ``parse_layers_arg``.
    """

    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start

    def __contains__(self, layer_idx: int) -> bool:
        return self.start <= int(layer_idx) < self.end


def parse_layers_arg(value: str) -> ParsedLayerRange | None:
    """Parse a ``--layers`` CLI value.

    Empty / whitespace input returns ``None`` (operator opted out).
    Otherwise expects ``"START-END"`` with both integers parseable;
    raises ``ManualShardingError(code='malformed' | 'empty_range' |
    'out_of_range_negative')`` on invalid input.

    Note: this parser does NOT know ``total_layers`` (caller doesn't
    have it at flag-parse time). Range vs. total_layers is enforced
    later by ``validate_manual_sharding`` once the peer announce
    arrives at the coord.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if "-" not in s:
        raise ManualShardingError(
            code="malformed",
            message=(
                f"--layers expects 'START-END' (inclusive-exclusive); "
                f"got {s!r}"
            ),
        )
    parts = s.split("-", 1)
    try:
        start = int(parts[0].strip())
        end = int(parts[1].strip())
    except ValueError as exc:
        raise ManualShardingError(
            code="malformed",
            message=(
                f"--layers START-END requires integer endpoints; "
                f"got {s!r} ({exc})"
            ),
        ) from exc
    if start < 0:
        raise ManualShardingError(
            code="out_of_range_negative",
            message=f"--layers START must be >= 0; got {start}",
        )
    if end <= start:
        raise ManualShardingError(
            code="empty_range",
            message=(
                f"--layers requires END > START "
                f"(inclusive-exclusive); got {start}-{end}"
            ),
        )
    return ParsedLayerRange(start=start, end=end)


@dataclass(frozen=True)
class _PeerRange:
    """One peer's layer claim, for validation.

    Internal to this module; the coord constructs these from the
    DHT-announced peer set and feeds them to
    ``validate_manual_sharding`` along with ``total_layers``.
    """

    peer_id: str
    range_or_none: ParsedLayerRange | None  # None = automatic


def validate_manual_sharding(
    peers: Sequence[_PeerRange] | Iterable[tuple[str, ParsedLayerRange | None]],
    *,
    total_layers: int,
) -> None:
    """Validate a swarm-wide manual-sharding configuration.

    Args:
        peers: Sequence of ``_PeerRange`` (or tuples convertible to it)
            describing each announced peer's claimed layer range. A
            peer with ``range_or_none=None`` opted out of manual
            sharding.
        total_layers: Total transformer layers in the model. The
            union of manual ranges must equal exactly
            ``[0, total_layers)``.

    Raises:
        ManualShardingError: with a ``code`` attribute identifying
        which invariant was violated. See class docstring for the
        taxonomy. Never returns a "partial pass" — the swarm either
        starts clean or refuses outright.

    Returns:
        ``None`` on success; the coord proceeds with the validated
        layer assignment.
    """
    if total_layers < 1:
        raise ManualShardingError(
            code="out_of_range",
            message=f"total_layers must be >= 1; got {total_layers}",
        )

    # Normalise to a list of _PeerRange.
    normalised: list[_PeerRange] = []
    for item in peers:
        if isinstance(item, _PeerRange):
            normalised.append(item)
        else:
            pid, rng = item
            normalised.append(_PeerRange(peer_id=str(pid), range_or_none=rng))

    if not normalised:
        # Empty swarm — nothing to validate. The coord will refuse to
        # serve inference for unrelated reasons.
        return

    # Invariant 1: all-or-nothing.
    manual = [p for p in normalised if p.range_or_none is not None]
    auto = [p for p in normalised if p.range_or_none is None]
    if manual and auto:
        manual_ids = ", ".join(p.peer_id for p in manual)
        auto_ids = ", ".join(p.peer_id for p in auto)
        raise ManualShardingError(
            code="partial_adoption",
            message=(
                f"manual_sharding_partial: {len(manual)} peers used "
                f"--layers ({manual_ids}); {len(auto)} did not "
                f"({auto_ids}). All peers or none."
            ),
        )

    if not manual:
        # Nobody used --layers; auto-assigner takes over. Nothing to do.
        return

    # Invariant 2a: every range fits in [0, total_layers).
    for p in manual:
        rng = p.range_or_none
        assert rng is not None  # for type checker
        if rng.end > total_layers:
            raise ManualShardingError(
                code="out_of_range",
                message=(
                    f"manual_sharding_out_of_range: peer {p.peer_id} "
                    f"claims layers {rng.start}-{rng.end} but model has "
                    f"only {total_layers} layers"
                ),
            )

    # Invariant 2b: union covers [0, total_layers) exactly once.
    # Sort by start; walk and require contiguous, non-overlapping
    # coverage from 0 to total_layers.
    manual_sorted = sorted(manual, key=lambda p: p.range_or_none.start)
    covered = 0
    for p in manual_sorted:
        rng = p.range_or_none
        assert rng is not None
        if rng.start < covered:
            # Overlap: rng.start is inside an already-claimed interval.
            raise ManualShardingError(
                code="overlap",
                message=(
                    f"manual_sharding_overlap: peer {p.peer_id} "
                    f"claims layer {rng.start} which was already "
                    f"covered by a previous peer (covered up to "
                    f"layer {covered})"
                ),
            )
        if rng.start > covered:
            # Gap: layer ``covered`` is unassigned.
            raise ManualShardingError(
                code="gap",
                message=(
                    f"manual_sharding_gap: layer {covered} unassigned "
                    f"(next peer {p.peer_id} starts at layer "
                    f"{rng.start})"
                ),
            )
        covered = rng.end

    if covered != total_layers:
        raise ManualShardingError(
            code="short",
            message=(
                f"manual_sharding_short: union covered "
                f"{covered}/{total_layers} layers; layer {covered} "
                f"and beyond are unassigned"
            ),
        )

    # Success: every layer claimed exactly once.
    return
