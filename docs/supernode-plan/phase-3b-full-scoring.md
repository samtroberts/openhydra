# Phase 3b — Full Scoring Pipeline

> Part of [Phase 3](phase-3-smart-routing-verification.md) · [Masterplan](MASTERPLAN.md)
> **Status:** 🔲 Not started **Owner:** _unassigned_ **Plan ID:** P3.2 (3b)
> Architecture refs: §4.3 (`compute_supernode_score`), §4.3.1 (selection), §9.3.3 (reputation decay)

## 1. Goal
Merge **static DHT data** + **live probe/gossip data** into the full `compute_supernode_score()`, with queue-aware backpressure, a rolling measured-TPS estimate, and reputation decay — feeding the randomized selector from 1d.

## 2. Scope
**In:**
- `compute_supernode_score()` (§4.3): tps/latency/load/warm/trust/reputation/nat weights; hard filters (context fit, `min_trust_level`); queue penalty.
- `measured_tps` rolling EMA (last ~10 requests) per supernode, overriding the static estimate when present.
- Queue-aware backpressure: if all candidates have `queue_depth > 0`, return an estimated wait to the client via `PromptChunk(status)`.
- Reputation decay toward 0.5 — **traffic-proportional**, not pure wall-clock (§9.3.3); the 0.001/hr figure is a placeholder to calibrate.
- Registry-verified boost from [2d](phase-2d-model-hash-registry.md) folded into trust.
- Clamp/validate `integration_level` before the trust lookup (avoid KeyError, §4.3 note).

**Out (deferred):**
- Verification-derived reputation values → [3c](phase-3c-output-verification.md) (this phase consumes a reputation store; 3c populates it).
- Geo tiebreak → [4b](phase-4b-geographic-affinity.md).
- Stake weighting → [4a](phase-4a-stake-weighted-routing.md).

## 3. Dependencies
**Upstream:** [3a](phase-3a-realtime-load.md) (live load/warm/queue), [1d](phase-1d-prompt-routing-streaming.md) (selector + basic score), [2c](phase-2c-embedded-attested-runtimes.md)/[2a](phase-2a-managed-ollama-l2.md) (trust tiers), [2d](phase-2d-model-hash-registry.md) (registry boost).
**Downstream:** [3c](phase-3c-output-verification.md), [3d](phase-3d-failover.md), [4a](phase-4a-stake-weighted-routing.md), [4b](phase-4b-geographic-affinity.md).
**Code touchpoints:** `coordinator/peer_selector.py` (current `compute_routing_score(latency_ms, load_pct, reputation, bandwidth_mbps, tier, s2s_rtt_ms)` — effectively rewritten), `supernode/router.py`.

## 4. Design & Approach
- Treat `compute_supernode_score` as a **rewrite** of `compute_routing_score`, not a backward-compatible extension (signature differs).
- Keep selection randomized among near-equal (§4.3.1) — scoring ranks, the selector spreads.
- Backpressure surfaces `estimated_wait_ms` from probe/gossip data; client gets a `status` chunk rather than a silent stall.
- Reputation read from a store that 3c writes; until 3c lands, default reputation = 0.5 (neutral).

## 5. Tasks (sub-sub checklist)
- [ ] Implement `compute_supernode_score()` with all weights + hard filters + queue penalty.
- [ ] `measured_tps` EMA store + override logic.
- [ ] Queue backpressure → `PromptChunk(status, estimated_wait_s)`.
- [ ] Reputation decay (traffic-proportional) + neutral default.
- [ ] Registry boost + `integration_level` clamp.
- [ ] Tests: scoring ranks correctly across mixed tiers/loads; backpressure path; EMA convergence; decay curve; filter correctness.

## 6. Files
**Create:** `supernode/scoring.py` (or extend `peer_selector.py`), `supernode/reputation_store.py` (interface; populated by 3c), `tests/test_supernode_scoring.py`.
**Modify:** `coordinator/peer_selector.py`, `supernode/router.py`.

## 7. Risks & Open Questions
- Weight tuning is empirical — expose weights as config; calibrate with real traffic.
- Decay/sampling constants are placeholders (§9.3.3) — run a simulation before committing values.
- `load` + `queue_penalty` correlate (intentional double-count) — tune jointly.
- **Rust hot-path candidate:** At scale (100K+ supernodes), `compute_supernode_score()` runs per-request over a large candidate set. If profiling shows this is a bottleneck, move the scoring loop to a PyO3 Rust extension (same pattern as `network/`). Profile first — GPU inference latency likely dominates, but the scoring loop is the most likely Python-side bottleneck at scale.

## 8. Test & Verification Plan
- Synthetic candidate sets: best-throughput warm node wins; overloaded node penalized; under-trust filtered.
- Backpressure: all-busy candidates → client receives wait estimate, not a hang.

## 9. Exit Criteria (Definition of Done)
- [ ] Scoring merges static + live signals; selection stays randomized near-equal.
- [ ] Queue backpressure returns wait estimates.
- [ ] `measured_tps` EMA + reputation decay in place; constants flagged as tunable.
