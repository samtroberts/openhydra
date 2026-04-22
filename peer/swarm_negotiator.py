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
# See the License for the specific language governing permissions and
# limitations under the License.

"""SwarmNegotiator — Phase 3 decentralised shard self-assignment.

A booting peer decides *which model* to serve and *which layers* of it to
cover by scanning the DHT for existing peers and filling the biggest gap
it can with its available VRAM budget.  No coordinator approval needed —
the rest of the swarm sees the claim via the next DHT announce and
converges.

Design goals:

* **Persona-aware.**  ``atomic_worker`` nodes never shard — they claim
  layer 0→N for the first model they declare as hosted.  ``native_shard``
  nodes run :func:`pick_best_fit` against discovered coverage gaps.
* **Self-healing under race.**  Two nodes booting simultaneously may
  both target the same gap.  After the first DHT announce both see the
  overlap; whichever has lower priority (smaller VRAM, lexicographically
  larger libp2p peer id) concedes on its next negotiation tick and
  re-picks a different gap.
* **Offline-safe.**  DHT scan failures collapse to "no existing peers"
  so a first-boot-on-empty-swarm still assigns the whole model to itself.
* **Manual CLI wins.**  Callers consult :attr:`ShardAssignment` only
  when ``--layer-start`` / ``--layer-end`` / ``--shard-index`` were not
  passed explicitly.  See ``coordinator/node.py`` for the call site.

Inputs are injected — the negotiator itself does no I/O, so unit tests
feed synthetic :class:`PeerClaim` lists via a stub ``dht_scan`` callable.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

from peer.capacity import (
    CapacityReport,
    ModelCapacity,
    NODE_PERSONA_ATOMIC_WORKER,
    NODE_PERSONA_NATIVE_SHARD,
    STATUS_CAPABLE,
    STATUS_SHARDABLE,
)


logger = logging.getLogger(__name__)


# Sources for a :class:`ShardAssignment` — documents WHY the negotiator
# picked the range it did, useful for logs and for the future Phase 4
# dashboard.  Kept as plain strings rather than an enum so the values
# round-trip through JSON without a custom encoder.
SOURCE_ATOMIC_WORKER = "atomic_worker"          # layer 0→N from hosted_model_ids
SOURCE_PICK_BEST_FIT = "pick_best_fit"          # filled a coverage gap
SOURCE_FALLBACK_WHOLE = "fallback_whole_model"  # no gaps found, serve 0..num_layers
SOURCE_CONFLICT_SPLIT = "conflict_split"  # two+ peers both claim whole — force split


@dataclass(frozen=True)
class PeerClaim:
    """A single peer's declared layer range for a specific model.

    Produced by the DHT scan callable.  All fields come directly from
    :class:`peer.dht_announce.Announcement` — this class is a narrow view
    carrying only what the negotiator needs for gap computation and
    conflict resolution.
    """

    libp2p_peer_id: str
    model_id: str
    layer_start: int
    layer_end: int
    total_layers: int
    available_vram_mb: int = 0


@dataclass(frozen=True)
class ShardAssignment:
    """The negotiator's verdict.

    ``source`` is one of :data:`SOURCE_ATOMIC_WORKER`,
    :data:`SOURCE_PICK_BEST_FIT`, :data:`SOURCE_FALLBACK_WHOLE`.
    """

    model_id: str
    layer_start: int
    layer_end: int
    total_layers: int
    source: str


# DHT scan signature — model_id → list of peer claims for that model.
# Callers supply an implementation (HTTP DHT, libp2p Kademlia, or a test
# stub).  The function MUST NOT raise — return [] on any error.
DhtScanFn = Callable[[str], list[PeerClaim]]


# ─── Pure helpers (testable without any DHT) ─────────────────────────────────


def compute_gaps(
    claims: list[PeerClaim],
    total_layers: int,
) -> list[tuple[int, int]]:
    """Return the list of ``(start, end)`` uncovered layer ranges.

    Ignores claims that don't match ``total_layers`` (wrong model depth)
    and claims whose range is outside ``[0, total_layers)``.  Overlapping
    claims are merged before gap computation — if two peers both cover
    layers 4..8, that's one block, not two.
    """
    if total_layers <= 0:
        return []

    # Clamp + filter.
    segments: list[tuple[int, int]] = []
    for c in claims:
        if int(c.total_layers) != int(total_layers):
            continue
        s = max(0, int(c.layer_start))
        e = min(int(total_layers), int(c.layer_end))
        if e > s:
            segments.append((s, e))

    if not segments:
        return [(0, total_layers)]

    # Merge overlapping/adjacent segments.
    segments.sort()
    merged: list[tuple[int, int]] = [segments[0]]
    for s, e in segments[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))

    # Invert to gaps.
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = e
    if cursor < total_layers:
        gaps.append((cursor, total_layers))
    return gaps


def pick_best_fit(
    gaps: list[tuple[int, int]],
    max_layers_hostable: int,
) -> tuple[int, int] | None:
    """Choose which (start, end) range from ``gaps`` to claim.

    Strategy:
        1. Prefer a gap I can cover *exactly* — fill the whole hole.
           Among those, take the widest so the network gains the most
           coverage from a single new peer.
        2. If no gap fits entirely, bite off the widest ``max_layers_hostable``
           slice of the widest gap.  Better to partially plug a big hole
           than to leave it wide open.

    Returns ``None`` if ``gaps`` is empty or ``max_layers_hostable <= 0``.
    """
    if not gaps or max_layers_hostable <= 0:
        return None

    # Coverable in one peer (the whole gap fits in max_layers_hostable).
    coverable = [(s, e) for s, e in gaps if (e - s) <= max_layers_hostable]
    if coverable:
        # Widest first — ties are rare enough that any deterministic order works;
        # sort by -span, then by starting layer for stability.
        coverable.sort(key=lambda r: (-(r[1] - r[0]), r[0]))
        return coverable[0]

    # Nothing fits whole.  Take the biggest slice we can of the widest gap.
    widest_s, widest_e = max(gaps, key=lambda r: r[1] - r[0])
    return (widest_s, widest_s + int(max_layers_hostable))


def compute_conflict_split(
    *,
    peer_claims: list[PeerClaim],
    total_layers: int,
    my_libp2p_peer_id: str,
    max_layers_hostable: int,
) -> tuple[int, int] | None:
    """Deterministically carve ``[0, total_layers)`` into contiguous
    chunks when the swarm has deadlocked on every peer claiming the
    whole model.

    The ``fallback_whole_model`` deadlock looks like this: peers A and B
    both boot, neither sees the other, each self-assigns [0, N), and
    each re-announces [0, N). On the next negotiation tick A scans the
    DHT and sees B's [0, N) claim; B sees A's. Because each claim
    already covers the whole range, :func:`compute_gaps` returns ``[]``
    (fully covered) → ``_native_shard_assignment`` skips the model →
    no reshard ever fires → ring stays single-peer-served forever.

    The fix: when the scan surfaces one or more other peers who also
    claim ``[0, total_layers)``, treat the swarm as ``k+1`` peers (me +
    the k overlappers), sort the combined libp2p_peer_ids
    lexicographically, and assign myself the chunk at my position.

    This uses the **same lex-order tie-break** already used by
    :func:`should_concede`, so the two heuristics agree on who wins
    which range without a second round-trip.

    Returns ``None`` if no deadlock is detected (no full-range peer
    claims), if our ``max_layers_hostable`` can't hold even one chunk,
    or if inputs are degenerate. The caller should fall through to the
    existing ``pick_best_fit`` path on ``None``.

    Determinism
    -----------
    * Every peer with the same inputs computes the same (my_start,
      my_end). No RPCs, no randomness.
    * Ties on peer_id are impossible (libp2p PeerIds are cryptographic
      hashes).
    * On an N-layer model with k+1 peers, chunks are size
      ``ceil(N / (k+1))``; any remainder is absorbed by the last peer.
      E.g. 24 layers, 2 peers → 12 + 12. 24 layers, 3 peers → 8 + 8 + 8.
      24 layers, 5 peers → 5 + 5 + 5 + 5 + 4.
    """
    if total_layers <= 0 or max_layers_hostable <= 0:
        return None
    my_id = str(my_libp2p_peer_id or "").strip()
    if not my_id:
        return None

    # Detect "whole model" overlappers — every peer whose claim covers
    # the entire [0, total_layers) range for this model.
    full_range_ids: set[str] = set()
    for c in peer_claims:
        if int(c.total_layers) != int(total_layers):
            continue
        if int(c.layer_start) <= 0 and int(c.layer_end) >= int(total_layers):
            pid = str(c.libp2p_peer_id or "").strip()
            if pid and pid != my_id:
                full_range_ids.add(pid)

    if not full_range_ids:
        return None  # no deadlock — let pick_best_fit handle it normally

    # Build the ordered participant list: me + every full-range peer.
    participants = sorted({my_id, *full_range_ids})
    n = len(participants)
    my_position = participants.index(my_id)

    # Ceiling division — the last participant absorbs any remainder.
    chunk_size = (int(total_layers) + n - 1) // n
    my_start = my_position * chunk_size
    my_end = min(int(total_layers), my_start + chunk_size)

    if my_end <= my_start:
        return None  # degenerate — position beyond layer count

    # Respect the local peer's hostable-layer budget. If the
    # computed chunk is bigger than we can hold, shrink the end.
    if (my_end - my_start) > int(max_layers_hostable):
        my_end = my_start + int(max_layers_hostable)

    return (my_start, my_end)


def should_concede(
    candidate: tuple[int, int],
    *,
    peer_claims: list[PeerClaim],
    model_id: str,
    total_layers: int,
    my_vram_mb: int,
    my_libp2p_peer_id: str,
) -> bool:
    """Return True if a higher-priority peer has already staked an
    overlapping range for the same model.

    Priority rules (highest wins):
        1. Greater ``available_vram_mb`` — a bigger peer is a better
           servant for the full gap.
        2. Tiebreak on ``available_vram_mb`` → lexicographically **smaller**
           ``libp2p_peer_id``.  Smaller-id-wins is arbitrary but
           deterministic; both peers reach the same verdict independently.

    Self-claims (``claim.libp2p_peer_id == my_libp2p_peer_id``) are
    ignored — we never concede to ourselves.  Claims on a different
    model or different total_layers are ignored — no overlap possible.
    """
    cs, ce = candidate
    for claim in peer_claims:
        if claim.libp2p_peer_id == my_libp2p_peer_id:
            continue  # that's us
        if claim.model_id != model_id:
            continue  # different model — no conflict
        if int(claim.total_layers) != int(total_layers):
            continue  # different model depth — ignore
        # Overlap detection: [cs, ce) intersects [claim.layer_start, claim.layer_end)
        if claim.layer_start < ce and claim.layer_end > cs:
            # Overlap exists — apply priority.
            their_vram = int(claim.available_vram_mb or 0)
            if their_vram > my_vram_mb:
                return True
            if their_vram == my_vram_mb and claim.libp2p_peer_id < my_libp2p_peer_id:
                return True
    return False


# ─── The negotiator class ────────────────────────────────────────────────────


class SwarmNegotiator:
    """Boot-time shard self-assigner.

    The negotiator is pure synchronous logic — no threading, no sleeps,
    no network I/O of its own.  All I/O is performed by the injected
    ``dht_scan`` callable.  This keeps unit tests fast and deterministic.

    Typical usage in :mod:`coordinator.node`::

        negotiator = SwarmNegotiator(
            capacity_report=report,
            libp2p_peer_id=p2p_node.libp2p_peer_id,
            dht_scan=lambda mid: _scan_dht_for_claims(mid, p2p_node, dht_urls),
        )
        assignment = negotiator.negotiate()
        if assignment is not None:
            args.shard_index = 0          # ignored; we use explicit layer range
            args.total_shards = 1         # ditto
            # feed layer_start/layer_end into the peer thread.

    Callers should ONLY consult the returned :class:`ShardAssignment` if
    the user did not pass ``--layer-start`` / ``--layer-end`` / ``--shard-index``
    explicitly — manual CLI intent always wins.
    """

    def __init__(
        self,
        *,
        capacity_report: CapacityReport,
        libp2p_peer_id: str,
        dht_scan: DhtScanFn,
        preferred_model_order: tuple[str, ...] = (),
    ):
        self.report = capacity_report
        self.libp2p_peer_id = str(libp2p_peer_id or "")
        self.dht_scan = dht_scan
        self.preferred_model_order = tuple(preferred_model_order or ())

    # ── public entry ─────────────────────────────────────────────────

    def negotiate(self) -> ShardAssignment | None:
        """Return the shard this node should serve, or ``None`` if nothing is
        feasible.  Always returns an immutable :class:`ShardAssignment`
        instance on success, never raises."""
        if self.report.node_persona == NODE_PERSONA_ATOMIC_WORKER:
            return self._atomic_assignment()
        if self.report.node_persona == NODE_PERSONA_NATIVE_SHARD:
            return self._native_shard_assignment()
        logger.warning(
            "swarm_negotiate_unknown_persona: %r — returning None",
            self.report.node_persona,
        )
        return None

    # ── atomic worker path ───────────────────────────────────────────

    def _atomic_assignment(self) -> ShardAssignment | None:
        """V1 constraint (Phase 1.5): one model per peer.  Pick the first
        ``capable`` capacity entry — that's the model the user declared
        their upstream hosts.  layer_start=0, layer_end=num_layers_total.

        We DO NOT consult the DHT for atomic workers — they serve the
        whole model as an opaque text API and can freely replicate.
        Multiple atomic workers of the same model is desirable redundancy.
        """
        for entry in self.report.capacity:
            if entry.status == STATUS_CAPABLE:
                return ShardAssignment(
                    model_id=entry.model_id,
                    layer_start=0,
                    layer_end=int(entry.num_layers_total),
                    total_layers=int(entry.num_layers_total),
                    source=SOURCE_ATOMIC_WORKER,
                )
        logger.info(
            "swarm_negotiate_atomic_no_capable: persona=atomic_worker "
            "but no capacity entry has status=capable (check --hosted-model-ids)"
        )
        return None

    # ── native shard path ────────────────────────────────────────────

    def _native_shard_assignment(self) -> ShardAssignment | None:
        """Rank usable capacity entries, then for each one scan the DHT,
        compute gaps, apply ``pick_best_fit``, check for conflicts, and
        emit the first uncontested assignment.  Returns ``None`` iff
        every candidate model is fully covered or every candidate range
        is overridden by a higher-priority peer."""
        my_vram_mb = self._my_vram_mb()
        candidates = self._rank_candidates()

        for entry in candidates:
            claims = self._safe_scan(entry.model_id)
            total_layers = int(entry.num_layers_total)

            # Deadlock-breaker: if one or more other peers are already
            # claiming the whole ``[0, total_layers)`` range (because
            # they, like us, fell back to ``fallback_whole_model`` on
            # their first tick), force a deterministic split so every
            # peer advances to a distinct contiguous chunk on this
            # tick. Must run *before* ``compute_gaps`` — otherwise
            # overlapping full-range claims hide the gap.
            split_candidate = compute_conflict_split(
                peer_claims=claims,
                total_layers=total_layers,
                my_libp2p_peer_id=self.libp2p_peer_id,
                max_layers_hostable=int(entry.max_layers_hostable),
            )
            if split_candidate is not None:
                # Before committing to the split, run the concede check
                # on the proposed range. If a higher-priority peer
                # (more VRAM, or lex-smaller id on a tie) already
                # overlaps our slice, let them serve the model alone
                # — don't force an unnecessary split. Fall through to
                # the next candidate model in the ranked list.
                my_start, my_end = split_candidate
                if should_concede(
                    (my_start, my_end),
                    peer_claims=claims,
                    model_id=entry.model_id,
                    total_layers=total_layers,
                    my_vram_mb=my_vram_mb,
                    my_libp2p_peer_id=self.libp2p_peer_id,
                ):
                    logger.info(
                        "swarm_negotiate_conflict_split_conceded: model=%s — "
                        "higher-priority peer already covers overlapping range",
                        entry.model_id,
                    )
                    continue
                logger.info(
                    "swarm_negotiate_conflict_split: model=%s layers=[%d, %d) "
                    "total=%d (breaking whole-model deadlock with %d peer(s))",
                    entry.model_id, my_start, my_end, total_layers,
                    sum(1 for c in claims
                        if int(c.total_layers) == total_layers
                        and int(c.layer_start) <= 0
                        and int(c.layer_end) >= total_layers),
                )
                return ShardAssignment(
                    model_id=entry.model_id,
                    layer_start=my_start,
                    layer_end=my_end,
                    total_layers=total_layers,
                    source=SOURCE_CONFLICT_SPLIT,
                )

            gaps = compute_gaps(claims, total_layers)

            if not gaps:
                logger.debug(
                    "swarm_negotiate_fully_covered: model=%s — skipping",
                    entry.model_id,
                )
                continue

            candidate = pick_best_fit(gaps, int(entry.max_layers_hostable))
            if candidate is None:
                continue

            if should_concede(
                candidate,
                peer_claims=claims,
                model_id=entry.model_id,
                total_layers=total_layers,
                my_vram_mb=my_vram_mb,
                my_libp2p_peer_id=self.libp2p_peer_id,
            ):
                logger.info(
                    "swarm_negotiate_conceded: model=%s range=%s-%s "
                    "(higher-priority peer already claimed overlapping range)",
                    entry.model_id, candidate[0], candidate[1],
                )
                continue

            # Empty swarm for this model? Log a friendlier message.
            source = SOURCE_PICK_BEST_FIT
            if len(claims) == 0 and candidate == (0, total_layers):
                source = SOURCE_FALLBACK_WHOLE

            logger.info(
                "swarm_negotiate_assigned: model=%s layers=[%d, %d) total=%d source=%s",
                entry.model_id, candidate[0], candidate[1], total_layers, source,
            )
            return ShardAssignment(
                model_id=entry.model_id,
                layer_start=int(candidate[0]),
                layer_end=int(candidate[1]),
                total_layers=total_layers,
                source=source,
            )

        logger.info(
            "swarm_negotiate_no_assignment: no candidate model could be "
            "claimed (fully covered or all conceded)"
        )
        return None

    # ── internals ────────────────────────────────────────────────────

    def _rank_candidates(self) -> list[ModelCapacity]:
        """Return capacity entries we could plausibly serve, ordered by
        preference.

        Preference rules:
            1. Models in ``preferred_model_order`` first (honouring the
               explicit order).
            2. Among non-preferred, larger ``max_layers_hostable`` first
               — we'd rather contribute a big chunk than a tiny one.
            3. Status must be ``capable`` or ``shardable``; ``incapable``
               entries are skipped outright.
        """
        usable = [
            c for c in self.report.capacity
            if c.status in (STATUS_CAPABLE, STATUS_SHARDABLE)
        ]
        if not usable:
            return []

        pref_rank = {mid: i for i, mid in enumerate(self.preferred_model_order)}
        # Sentinel for non-preferred models so they sort after all preferred ones.
        non_pref_rank = len(self.preferred_model_order) + 1

        usable.sort(key=lambda c: (
            pref_rank.get(c.model_id, non_pref_rank),
            -int(c.max_layers_hostable),
            c.model_id,  # stable tiebreak for deterministic test output
        ))
        return usable

    def _my_vram_mb(self) -> int:
        """Best-effort read of this node's available VRAM in MB, used for
        conflict resolution priority.  Falls back to available RAM (for
        Apple Silicon unified memory) then 0 (unknown)."""
        hw = dict(self.report.hardware or {})
        for key in ("vram_available_mb", "vram_total_mb", "ram_available_mb", "ram_total_mb"):
            value = hw.get(key)
            if value is None:
                continue
            try:
                v = int(value)
            except (TypeError, ValueError):
                continue
            if v > 0:
                return v
        return 0

    def _safe_scan(self, model_id: str) -> list[PeerClaim]:
        """Call ``self.dht_scan`` but swallow any exception — first-boot
        peers on an empty swarm (or when the DHT bootstrap is unreachable)
        MUST still be able to self-assign."""
        try:
            result = self.dht_scan(model_id) or []
        except Exception as exc:
            logger.warning(
                "swarm_negotiate_dht_scan_failed: model=%s err=%s "
                "— treating as empty swarm",
                model_id, exc,
            )
            return []
        # Defensive: filter out any junk the scan returned.
        out: list[PeerClaim] = []
        for item in result:
            if isinstance(item, PeerClaim):
                out.append(item)
        return out
