# Plan: Autonomous Dynamic Rebalancing

## Context

OpenHydra has two rebalancing systems already built:
1. **Gap-fill** (`coordinator/rebalancer.py`) — detects missing layers and directs peers to expand
2. **Throughput** (`coordinator/swarm_rebalance.py`) — finds bottleneck layers and suggests migration

Both are **coordinator-driven**: the coordinator detects the problem, computes a directive, publishes it to DHT, and the peer polls + applies. This is reactive and centralized.

**Petals' approach**: Each peer autonomously runs `should_choose_other_blocks()` which:
1. Computes the swarm-wide per-block throughput
2. Hypothetically removes itself from its current position
3. Finds the position where it would maximize the minimum throughput
4. If the improvement exceeds a threshold, autonomously restarts with new blocks

This is **decentralized, self-healing, and continuous**.

## Current State

| Aspect | OpenHydra Today | Petals | Gap |
|---|---|---|---|
| **Who decides** | Coordinator computes directives | Each peer decides independently | Centralized vs decentralized |
| **Trigger** | Incomplete pipeline at request time | Periodic self-check (every ~60s) | Reactive vs proactive |
| **Data source** | Static TPS estimates (now fixed by Phase D) | Self-measured compute + network TPS | ✅ Fixed |
| **Application** | Peer polls DHT for directives | Peer restarts with new layer range | Similar mechanism |
| **Safety** | Drain + inflight check | Drain + migration | Similar |

## Design: Peer-Autonomous Rebalancing

### Core Idea

Each peer periodically evaluates: "Would the swarm be better off if I served different layers?" This runs entirely on the peer side — no coordinator involvement.

### Algorithm: `should_rebalance()`

```python
def should_rebalance(
    my_peer_id: str,
    my_layer_start: int,
    my_layer_end: int,
    my_tps: float,
    swarm_peers: list[PeerInfo],  # from DHT lookup
    total_layers: int,
    balance_quality: float = 1.15,  # 15% improvement threshold
) -> RebalanceDecision | None:
    """Determine if this peer should serve different layers.
    
    1. Compute per-layer throughput for the whole swarm
    2. Find the current minimum throughput (bottleneck)
    3. Hypothetically remove this peer from its current position
    4. Find the position that maximizes the new minimum throughput
    5. If improvement ≥ balance_quality, recommend migration
    """
    # Step 1: Compute current per-layer throughput
    per_layer = [0.0] * total_layers
    for peer in swarm_peers:
        for layer in range(peer.layer_start, peer.layer_end):
            per_layer[layer] += peer.tps
    
    current_min = min(per_layer) if per_layer else 0.0
    
    # Step 2: Remove self from current position
    for layer in range(my_layer_start, my_layer_end):
        per_layer[layer] -= my_tps
    
    # Step 3: Try every possible position for our span width
    span = my_layer_end - my_layer_start
    best_start = my_layer_start
    best_min = current_min
    
    for start in range(0, total_layers - span + 1):
        # Simulate placing self at [start, start+span)
        simulated = list(per_layer)  # copy without self
        for layer in range(start, start + span):
            simulated[layer] += my_tps
        new_min = min(simulated)
        if new_min > best_min:
            best_min = new_min
            best_start = start
    
    # Step 4: Check if improvement is significant
    if current_min > 0 and best_min / current_min >= balance_quality:
        if best_start != my_layer_start:
            return RebalanceDecision(
                new_layer_start=best_start,
                new_layer_end=best_start + span,
                current_min_tps=current_min,
                new_min_tps=best_min,
                improvement_ratio=best_min / current_min,
            )
    
    # Special case: current_min == 0 means there's a gap we could fill
    if current_min == 0 and best_min > 0:
        return RebalanceDecision(
            new_layer_start=best_start,
            new_layer_end=best_start + span,
            current_min_tps=0,
            new_min_tps=best_min,
            improvement_ratio=float('inf'),
        )
    
    return None  # No beneficial rebalance found
```

### Where It Runs

In the peer's **announce loop** (`peer/server.py` lines 1019-1130), after announcing to DHT:

```python
# Every N announce cycles (e.g., every 6th = ~60s), evaluate rebalancing
if announce_count % rebalance_check_interval == 0:
    swarm_peers = load_swarm_snapshot_from_dht(dht_urls, model_id)
    decision = should_rebalance(
        my_peer_id=service.peer_id,
        my_layer_start=current_layer_start,
        my_layer_end=current_layer_end,
        my_tps=measured_tps,  # From Phase D benchmark
        swarm_peers=swarm_peers,
        total_layers=total_layers,
    )
    if decision is not None:
        apply_rebalance(service, decision)
```

### Data Source: DHT Swarm Snapshot

The peer needs to know the entire swarm state to evaluate positions. It gets this from the DHT:

```python
def load_swarm_snapshot_from_dht(dht_urls, model_id) -> list[PeerInfo]:
    """Fetch all peers for this model from DHT bootstrap nodes."""
    from coordinator.path_finder import load_peers_from_dht
    peers = load_peers_from_dht(dht_urls[0], model_id=model_id, timeout_s=3.0)
    return [
        PeerInfo(
            peer_id=p.peer_id,
            layer_start=p.layer_start,
            layer_end=p.layer_end,
            tps=p.runtime_estimated_tokens_per_sec,
        )
        for p in peers
        if p.layer_start < p.layer_end and p.total_layers > 0
    ]
```

This uses the already-measured TPS from Phase D announcements.

### Safety Guards

1. **Cooldown**: After a rebalance, wait at least 5 minutes before checking again
2. **Inflight drain**: Only rebalance when `inflight_count() == 0`
3. **Load signal**: Set load to 100% during resharding (prevents routing to this peer)
4. **Min improvement**: Only rebalance if improvement ≥ 15% (configurable)
5. **MLX guard**: Skip for MLX peers (can't reshard)
6. **Concurrent guard**: If multiple peers want to rebalance simultaneously, DHT announcement ordering provides natural serialization — each peer sees the updated state before deciding
7. **Oscillation prevention**: Track last 3 positions; don't return to a recent position

### Stability: Preventing Oscillation

Petals uses an iterative simulation: before committing, simulate other peers also repositioning. If the stable state still shows improvement, proceed. OpenHydra can simplify this:

- **Hysteresis**: Require 15% improvement to move, but only 5% to stay. This creates a "dead zone" that prevents small oscillations.
- **Position history**: Keep a ring buffer of last 3 (start, end) positions. Don't move to a position in the history.
- **Jitter**: Add random delay (0-30s) before applying rebalance to prevent herding.

## Files to Modify/Create

### New Files

| File | Purpose | Lines (est) |
|---|---|---|
| `peer/autonomous_rebalancer.py` | `should_rebalance()` algorithm + `RebalanceDecision` dataclass | ~200 |
| `tests/test_autonomous_rebalancer.py` | Unit tests for the algorithm | ~150 |

### Modified Files

| File | Change |
|---|---|
| `peer/server.py` | Add rebalance check in announce loop (~20 lines) |
| `peer/dht_announce.py` | Add `last_rebalance_unix_ms` field to prevent herding |
| `coordinator/node.py` | Add `--rebalance-enabled` / `--rebalance-interval` CLI flags |

## Implementation Phases

### Phase 1: Algorithm + Unit Tests
- Create `peer/autonomous_rebalancer.py` with `should_rebalance()`
- Thorough unit tests covering: gap filling, bottleneck migration, no-op when balanced, cooldown, oscillation prevention

### Phase 2: Wire into Announce Loop
- Add periodic `should_rebalance()` call in `_announce_loop()`
- Integrate with existing `shard.reshard()` mechanism
- Safety guards (inflight drain, load signal, cooldown)

### Phase 3: CLI + Config
- `--rebalance-enabled` flag (default True for PyTorch, False for MLX)
- `--rebalance-interval` (default 6 announce cycles = ~60s)
- `--rebalance-min-improvement` (default 1.15)
- `--rebalance-cooldown-s` (default 300)

## Expected Impact

- **Self-healing gaps**: When a peer goes down, remaining peers detect the gap and autonomously expand to fill it (within their memory capacity)
- **Bottleneck relief**: When one layer range is slower than others (heterogeneous hardware), faster peers migrate to cover it
- **No coordinator involvement**: Works even when the coordinator is down or unreachable
- **Data-driven**: Uses Phase D measured TPS, not static estimates

## Estimated Effort

| Component | Hours |
|---|---|
| `autonomous_rebalancer.py` | 6 |
| Unit tests | 5 |
| Announce loop integration | 4 |
| CLI + config | 2 |
| Integration testing | 4 |
| **Total** | **~21 hours** |
