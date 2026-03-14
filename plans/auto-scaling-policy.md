# Sub-Plan 1: Capability-Aware Auto-Scaling Policy

> **Parent:** [Section 4 of Beta Launch Strategy](../docs/beta-launch-strategy.md#4-auto-scaling-and-model-promotion)
> **Status:** Design complete, implementation not started
> **Created:** 2026-03-10

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Redundancy Ratios for Qwen 3.5 Family](#2-redundancy-ratios-for-qwen-35-family)
3. [The Weak Nodes Problem](#3-the-weak-nodes-problem)
4. [The Small Swarm Problem](#4-the-small-swarm-problem)
5. [Capability-Aware Promotion Logic](#5-capability-aware-promotion-logic)
6. [Demand Weighting](#6-demand-weighting)
7. [Fleet Minimums and Hysteresis](#7-fleet-minimums-and-hysteresis)
8. [Tiered Roles for Weak Nodes](#8-tiered-roles-for-weak-nodes)
9. [Full Policy Architecture](#9-full-policy-architecture)
10. [Implementation Plan](#10-implementation-plan)

---

## 1. Problem Statement

The master plan (section 4) proposed a simple auto-scaling rule:

> When a model achieves 3x redundancy, promote to the next larger model.

This is a good starting point but breaks down under three real-world conditions:

| Failure Mode | Cause | Result |
|--------------|-------|--------|
| **Weak node promotion** | Many nodes join but lack the VRAM for a bigger model | Scaler promotes; most nodes can't serve it; service degrades |
| **Gravitational collapse** | Only powerful nodes get useful work; weak nodes earn nothing and leave | Swarm shrinks to a handful of elite nodes |
| **Oscillation** | Borderline redundancy causes promote → demote → promote cycles | Unstable model selection, inconsistent user experience |

This sub-plan redesigns the auto-scaling policy to handle all three.

---

## 2. Redundancy Ratios for Qwen 3.5 Family

Redundancy ratio = `capable_peers / shards_needed(model)`.

The critical distinction: **capable_peers** is not the total peer count. It is the count of peers whose `available_vram >= shard_size(model)`.

### Per-Model Requirements

| Model | Params | VRAM per shard (FP16) | VRAM per shard (NF4) | Shards for full model | Min peers for 3x ratio |
|-------|--------|-----------------------|----------------------|-----------------------|------------------------|
| Qwen3.5-0.6B | 0.6B | ~2 GB | ~1 GB | 1 | 3 |
| Qwen3.5-1.7B | 1.7B | ~4 GB | ~2 GB | 1 | 3 |
| Qwen3.5-4B | 4B | ~8 GB | ~3 GB | 1 | 3 |
| Qwen3.5-8B | 8B | ~16 GB | ~5 GB | 1-2 | 3-6 |
| Qwen3.5-14B | 14B | ~28 GB | ~8 GB | 2-4 | 6-12 |
| Qwen3.5-32B | 32B | ~64 GB | ~18 GB | 4-8 | 12-24 |
| Qwen3.5-72B | 72B | ~144 GB | ~40 GB | 8-16 | 24-48 |

### Example: 50-Peer Network with Mixed Hardware

```
Peer composition:
  30x RPi/phone    (4 GB)
  12x laptop       (16 GB)
   5x gaming PC    (24 GB)
   3x workstation  (48+ GB)

Effective redundancy per model:
  Qwen3.5-0.6B:  50 capable / 1 shard  = 50.0x  (all peers)
  Qwen3.5-1.7B:  50 capable / 1 shard  = 50.0x  (all peers)
  Qwen3.5-4B:    20 capable / 1 shard  = 20.0x  (laptops + PCs + workstations)
  Qwen3.5-8B:    20 capable / 1 shard  = 20.0x  (laptops + PCs + workstations)
  Qwen3.5-14B:    8 capable / 2 shards =  4.0x  (PCs + workstations)
  Qwen3.5-32B:    3 capable / 4 shards =  0.75x (workstations only — NOT promotable)
  Qwen3.5-72B:    3 capable / 8 shards =  0.38x (impossible with current fleet)

Naive total-peer ratio would say:
  Qwen3.5-32B:  50 / 4 = 12.5x   ← WRONG, only 3 peers can actually serve it
```

This is why capability filtering is essential.

---

## 3. The Weak Nodes Problem

### Scenario

```
Before:   8 peers serving Qwen3.5-4B at 8x redundancy (stable)

Event:    40 Raspberry Pi-class nodes join (4 GB VRAM each)

Naive:    Total peers = 48
          Naive ratio for Qwen3.5-8B = 48/2 = 24x → "PROMOTE!"

Reality:  40 new nodes can't run 8B shards (need 16 GB, have 4 GB)
          Only 8 original peers can serve 8B
          Half of those 8 get reassigned away from 4B
          4B ratio drops from 8x to 4x
          8B ratio is actually 4x (not 24x)
          Both models are now fragile
```

### Root Cause

The naive ratio counts **all peers**, not **capable peers**. It conflates network size with network capability.

### Fix: Effective Redundancy

```python
def effective_redundancy(model: ModelSpec, peers: list[PeerInfo]) -> float:
    """Only count peers that can actually serve this model."""
    capable = [p for p in peers if p.available_vram_gb >= model.shard_vram_gb]
    if model.shards_needed == 0:
        return float('inf')
    return len(capable) / model.shards_needed
```

---

## 4. The Small Swarm Problem

Even with capability filtering, there's a subtler failure mode.

### Scenario: Gravitational Collapse

If auto-scaling only considers redundancy, the network naturally evolves toward:

```
Tier A:   5 workstations (64 GB) → serve Qwen3.5-32B → get all requests → earn all HYDRA
Tier B:  15 laptops (16 GB)      → serve Qwen3.5-4B  → some requests   → some HYDRA
Tier C:  80 phones/RPi (4 GB)    → serve Qwen3.5-0.6B → nobody wants it → earn nothing → leave
```

Over time, Tier C peers leave (no earnings). Tier B peers leave (low earnings relative to Tier A). The "swarm" collapses into 5 workstations.

### Root Cause

The auto-scaler optimises for **model quality** (bigger is better) without considering:
- Whether anyone is **requesting** smaller models
- Whether weak nodes have **any useful work** to earn credits
- Whether promoting kills diversity

### Fix: Three-Part Solution

1. **Demand weighting** — only promote to models people actually request (Section 6)
2. **Fleet minimums** — never let a served model drop below a safety floor (Section 7)
3. **Tiered roles** — give weak nodes non-inference work that earns credits (Section 8)

---

## 5. Capability-Aware Promotion Logic

The promotion decision must satisfy ALL of the following:

```python
def should_promote(
    candidate: ModelSpec,
    current_models: list[ModelSpec],
    peers: list[PeerInfo],
    request_log: RequestLog,
) -> bool:
    # 1. Candidate has enough capable peers
    candidate_ratio = effective_redundancy(candidate, peers)
    if candidate_ratio < PROMOTE_THRESHOLD:  # 3.0
        return False

    # 2. No currently served model drops below the safety floor
    # Simulate: what if capable peers get reassigned to candidate?
    for model in current_models:
        remaining_ratio = effective_redundancy_after_reassignment(
            model, candidate, peers
        )
        if remaining_ratio < FLOOR_RATIO:  # 2.0
            return False

    # 3. There's actual demand for the candidate's quality tier
    demand = request_log.demand_weight(candidate.quality_tier)
    if demand < MIN_DEMAND_WEIGHT:  # 0.3
        return False

    # 4. Hysteresis: don't promote if we recently demoted this model
    if candidate.model_id in recently_demoted(window=timedelta(minutes=15)):
        return False

    return True
```

### Reassignment Simulation

When evaluating a promotion, we must check: "If we start serving Model B, what happens to Model A?"

Not all capable-for-B peers were serving A — some might have been idle or serving a different model. The simulation:

```python
def effective_redundancy_after_reassignment(
    existing_model: ModelSpec,
    new_model: ModelSpec,
    peers: list[PeerInfo],
) -> float:
    """Worst case: all peers capable of new_model get pulled from existing_model."""
    capable_for_new = {p.id for p in peers if p.available_vram_gb >= new_model.shard_vram_gb}
    serving_existing = {p.id for p in peers if p.assigned_model == existing_model.model_id}

    # Peers that might leave existing for new
    at_risk = serving_existing & capable_for_new

    # Worst case: all at-risk peers leave
    remaining = len(serving_existing) - len(at_risk)
    return remaining / existing_model.shards_needed
```

In practice, the coordinator would reassign a subset (not all), so this is a conservative estimate.

---

## 6. Demand Weighting

Not all models are equally requested. Promoting to a model nobody wants wastes capacity.

### Demand Weight Calculation

```python
class RequestLog:
    """Sliding window of recent requests, bucketed by model quality tier."""

    def __init__(self, window: timedelta = timedelta(hours=1)):
        self._window = window
        self._buckets: dict[str, int] = defaultdict(int)

    def record(self, model_id: str):
        tier = quality_tier(model_id)
        self._buckets[tier] += 1

    def demand_weight(self, tier: str) -> float:
        """0.0 = nobody requested this tier; 1.0 = all requests were for this tier."""
        total = sum(self._buckets.values())
        if total == 0:
            return 0.5  # no data yet — neutral
        return self._buckets[tier] / total
```

### Quality Tiers

Models map to quality tiers based on capability:

| Tier | Models | Typical use case |
|------|--------|------------------|
| `basic` | 0.6B, 1.7B | Autocomplete, simple Q&A, classification |
| `standard` | 4B, 8B | General chat, summarisation, code assist |
| `advanced` | 14B, 32B | Complex reasoning, long-form generation |
| `frontier` | 72B+ | Research, multi-step reasoning |

### How It Interacts with Promotion

```python
promotion_score = effective_redundancy * demand_weight

# Example:
# Qwen3.5-0.6B:  50x redundancy * 0.05 demand = 2.5  (overkill, nobody wants it)
# Qwen3.5-4B:     8x redundancy * 0.40 demand = 3.2  (sweet spot)
# Qwen3.5-8B:     4x redundancy * 0.35 demand = 1.4  (not ready yet)
# Qwen3.5-14B:    2x redundancy * 0.15 demand = 0.3  (nowhere near ready)
```

This prevents over-allocating capacity to models with no demand, even if the raw redundancy looks good.

---

## 7. Fleet Minimums and Hysteresis

### Fleet Minimum (Safety Floor)

Every model currently being served must maintain at least `FLOOR_RATIO` (default: 2.0) effective redundancy.

```python
FLOOR_RATIO = 2.0  # minimum viable redundancy for any active model
```

This prevents the "rob Peter to pay Paul" problem where promoting a new model cannibalises an existing one.

### Hysteresis Band

Promotion and demotion use different thresholds to prevent oscillation:

```python
PROMOTE_THRESHOLD = 3.0   # need 3x effective redundancy to promote
DEMOTE_THRESHOLD  = 1.5   # don't demote until below 1.5x

# The gap (1.5x) is the hysteresis band.
# Prevents: promote → peers leave → demote → peers return → promote → ...
```

### Cooldown Period

After any model change (promotion or demotion), the scaler enters a 15-minute cooldown:

```python
COOLDOWN_AFTER_CHANGE = timedelta(minutes=15)
```

During cooldown:
- The scaler **monitors** but does not **act**
- This gives the network time to stabilise after a topology change
- Prevents cascading changes from a single event (e.g., 10 peers joining simultaneously)

### Re-Evaluation Cadence

The scaler runs every **5 minutes** (not per-request):

```python
RE_EVALUATE_INTERVAL = timedelta(minutes=5)
```

Per-request evaluation would be too reactive and cause flapping. 5 minutes is long enough to smooth out transient peer churn but short enough to respond to real capacity changes.

---

## 8. Tiered Roles for Weak Nodes

Weak nodes (low VRAM, slow compute) can't run large models. But if they have **no way to earn credits**, they leave — and the network shrinks. This is the death spiral.

### Solution: Non-Inference Work That Earns Credits

| Role | What It Does | VRAM Needed | Earnings Rate |
|------|-------------|-------------|---------------|
| **Embedding server** | Serve small embedding models (e.g., `all-MiniLM-L6-v2`, 80 MB) | < 1 GB | 0.3x base |
| **Verification auditor** | Re-execute random inference samples for verification (Tier 1/3) | Varies (uses their max model) | 0.5x base |
| **KV compaction offload** | Run CPU-bound HAK/OMP compaction on behalf of busy peers | < 1 GB (CPU-bound) | 0.2x base |
| **Activation relay** | Cache and forward activations in layer-sharded pipelines | < 512 MB | 0.1x base |
| **Model cache seed** | Store and serve model weight chunks to new peers (HTTP) | Disk only | 0.2x base |

### Role Assignment Logic

```python
def assign_role(peer: PeerInfo, model_roster: list[ModelSpec]) -> str:
    """Assign the most valuable role this peer can fill."""
    # Can it serve any model in the roster?
    for model in sorted(model_roster, key=lambda m: m.params, reverse=True):
        if peer.available_vram_gb >= model.shard_vram_gb:
            return f"inference:{model.model_id}"

    # Can't serve any rostered model — assign support role
    if peer.available_vram_gb >= 1.0:
        return "embedding_server"
    if peer.cpu_score >= MIN_CPU_SCORE:
        return "verification_auditor"
    if peer.disk_free_gb >= 10.0:
        return "model_cache_seed"
    return "activation_relay"  # absolute minimum: forward packets
```

### Economy Integration

All roles must earn credits to prevent the death spiral:

```python
EARNINGS_MULTIPLIER = {
    "inference":            1.0,   # base rate (1000 tokens = 1 credit)
    "embedding_server":     0.3,
    "verification_auditor": 0.5,
    "kv_compaction":        0.2,
    "activation_relay":     0.1,
    "model_cache_seed":     0.2,
}
```

Even at 0.1x, weak nodes earn *something*, which keeps them in the network. The economy's 5%/day credit decay ensures credits don't accumulate indefinitely for low-activity roles.

---

## 9. Full Policy Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO-SCALING POLICY ENGINE                    │
│                                                                 │
│  Inputs (collected every 5 minutes):                            │
│    Per-peer:   available_vram, compute_score, latency, uptime   │
│    Per-model:  shard_vram, shards_needed, quality_tier           │
│    Fleet:      current_assignments, request_log (1h window)     │
│                                                                 │
│  Step 1: Compute effective redundancy for each candidate model  │
│    effective_ratio[m] = capable_peers(m) / shards_needed(m)     │
│                                                                 │
│  Step 2: Compute demand-weighted promotion score                │
│    score[m] = effective_ratio[m] * demand_weight(m.tier)        │
│                                                                 │
│  Step 3: Check constraints                                      │
│    For each candidate where score >= PROMOTE_THRESHOLD:         │
│      a. Would any existing model drop below FLOOR_RATIO?        │
│      b. Is the candidate in the cooldown blacklist?             │
│      c. Is demand_weight >= MIN_DEMAND_WEIGHT?                  │
│                                                                 │
│  Step 4: Select best promotion (if any pass all checks)         │
│    best = argmax(score[m]) among passing candidates             │
│                                                                 │
│  Step 5: Check for demotions                                    │
│    For each active model where effective_ratio < DEMOTE_THRESH: │
│      Demote to next smaller model in the ladder                 │
│                                                                 │
│  Step 6: Assign roles to remaining peers                        │
│    Peers not assigned to inference → support roles              │
│                                                                 │
│  Output:                                                        │
│    model_roster:  list of (model_id, assigned_peers)            │
│    role_roster:   list of (peer_id, role)                       │
│    next_eval_at:  now + RE_EVALUATE_INTERVAL                    │
│                                                                 │
│  Constants:                                                     │
│    PROMOTE_THRESHOLD   = 3.0                                    │
│    DEMOTE_THRESHOLD    = 1.5                                    │
│    FLOOR_RATIO         = 2.0                                    │
│    MIN_DEMAND_WEIGHT   = 0.3                                    │
│    RE_EVALUATE_INTERVAL = 5 min                                 │
│    COOLDOWN_AFTER_CHANGE = 15 min                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Implementation Plan

### Files to Create / Modify

| File | Action | Description |
|------|--------|-------------|
| `coordinator/auto_scaler.py` | **Create** | `AutoScaler` class with the full policy engine |
| `coordinator/request_log.py` | **Create** | `RequestLog` — sliding window demand tracker |
| `coordinator/role_assigner.py` | **Create** | `RoleAssigner` — support role assignment for weak nodes |
| `coordinator/engine.py` | **Modify** | Wire `AutoScaler` into the coordinator's 5-minute tick |
| `coordinator/degradation.py` | **Modify** | `DegradationPolicy` now consults `AutoScaler` for the model roster |
| `models/catalog.py` | **Modify** | Add `shard_vram_gb`, `shards_needed`, `quality_tier` per model |
| `peer/server.py` | **Modify** | Peers report `available_vram` and `compute_score` in DHT announcements |
| `tests/test_auto_scaler.py` | **Create** | Unit tests for all promotion/demotion/role scenarios |

### Key Test Scenarios

| # | Scenario | Expected Outcome |
|---|----------|------------------|
| 1 | 10 capable peers, 3x ratio for 4B | Promote to 4B |
| 2 | 40 weak peers join (4 GB), only 10 can run 8B | Do NOT promote to 8B |
| 3 | Promotion would drop existing model below 2x floor | Block promotion |
| 4 | High redundancy but zero demand for that tier | Block promotion (demand < 0.3) |
| 5 | Ratio oscillates around 3.0x | Hysteresis prevents flapping |
| 6 | 80 weak peers, no inference role available | Assign support roles, all earn credits |
| 7 | Peer leaves mid-cooldown | No re-evaluation until cooldown expires |
| 8 | Network grows from 5 to 500 peers over 1 hour | Gradual promotion ladder, no jumps |

### Estimated Effort

| Component | Effort | Depends On |
|-----------|--------|------------|
| `AutoScaler` core logic | 2 days | `models/catalog.py` having shard metadata |
| `RequestLog` | 0.5 day | Nothing |
| `RoleAssigner` | 1 day | Economy module awareness |
| Coordinator wiring | 1 day | `AutoScaler` |
| Peer VRAM reporting | 0.5 day | Nothing |
| Tests | 2 days | All of the above |
| **Total** | **~7 days** | |

---

## Open Questions

1. **Should demand weight include latency preference?** Users might prefer a faster 4B over a slower 14B (layer-sharded over WAN). The demand signal could include a latency SLA.

2. **How should the scaler handle models that require layer sharding?** Assembling a 4-shard pipeline is fundamentally different from having 4 single-shard peers. The `shards_needed` metric is a simplification — in reality you need peers with complementary layer ranges.

3. **Should weak nodes be able to "graduate"?** If a phone peer upgrades to a laptop, its role should change. The 5-minute re-evaluation handles this automatically, but there may be edge cases.

4. **What's the minimum network size for the scaler to activate?** Below some threshold (e.g., 5 peers), the scaler should stay on the smallest model regardless of redundancy, to avoid instability.

---

## References

- Master plan: [docs/beta-launch-strategy.md, Section 4](../docs/beta-launch-strategy.md#4-auto-scaling-and-model-promotion)
- KV compaction: [docs/architecture.md, Section 8](../docs/architecture.md#8-kv-compaction)
- Economy: [docs/architecture.md, Section 9](../docs/architecture.md#9-economy)
- Verification tiers: [docs/architecture.md, Section 11](../docs/architecture.md#11-verification)
