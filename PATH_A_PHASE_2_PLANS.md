# Path A — Phase 2: Throughput Compounding Plans

Two parallel plans for the next TPS-compounding step on top of the
2026-04-24 baseline of **3.76 TPS / 3.87× cross-ISP baseline**.

Both build on the Path A infrastructure already shipped:
- Pure-coordinator mode with `StandaloneHead` (MLX + PyTorch backends)
- LAN-first routing (`peer/lan_routing.py`)
- Rust `P2PNode.add_address` + auto-populate Kademlia
- `coordinator_proxy_handler_loop` + coord-side gRPC `PushResult` server
- Ring session state (`coordinator/head_sampler.py::RingSession`)

Where the two plans differ is **what they overlap with what**.

---

## Timing budget (measured, not estimated)

From the 3-node True Petals run on 2 × T4 + 1 × CPU coord:

```
autoregressive_ring_done: tokens=28 latency_ms=7442.8 tps=3.76
→ ~266 ms per token cycle
```

Per-cycle breakdown (instrumented via `PROFILE _request_stage`,
`push_forwarded_via_lan`, `coord_push_result_received`,
`coord_ring_sampled` log timestamps):

| Phase | Time per cycle | Notes |
|---|---|---|
| Coord → peer-0 proxy_forward + deserialize | ~35 ms | libp2p + gRPC ser/deser |
| Peer-0 layers 0-11 (PyTorch T4, bf16) | ~40-55 ms | One forward on 1-token input with KV cache |
| Peer-0 → peer-1 `push_forwarded_via_lan` | ~10-15 ms | Direct LAN gRPC, ~1500-element fp32 activation |
| Peer-1 layers 12-23 (PyTorch T4, bf16) | ~40-55 ms | Same compute as stage 0 |
| Peer-1 → coord `PushResult` (libp2p) | ~20-35 ms | The return hop is libp2p (coord is NAT'd on VPC) |
| Coord `final_norm` + `lm_head` + sample on CPU bf16 | ~10 ms | Matmul dominates |
| Coord fire-and-forget re-inject via proxy | ~15-25 ms | libp2p proxy |
| **Total per token** | **~170-220 ms** | Remainder (~50 ms) is Python GIL + gRPC overhead + unaccounted |

Two observations:

1. Peers are **idle ~60% of the time per cycle**. When peer-1 is computing,
   peer-0 sits waiting. When the coord is sampling, both peers wait. When
   tokens are in flight on the wire, everyone waits.
2. **No single phase dominates.** The gains from further compute
   acceleration on the peers are bounded because wire + sampling eat
   ~100-120 ms/cycle regardless.

---

## Plan (a) — Pipeline Parallelism (speculative / inter-token pipelining)

**Goal:** while peer-1 runs layers 12-23 for token N, peer-0 runs
layers 0-11 for token N+1 **speculatively**. Both peers are busy
~100% of the cycle instead of ~40%.

**Expected TPS multiplier:** ~1.8-2.2× (bounded by speculation accept
rate and the serial coord-side sampling step).

**Scope:** the bigger of the two. ~2-3 days of focused work + bench.

### The core idea

Today the ring is strictly serial:

```
t=0:   coord → peer-0 → peer-1 → coord(sample) → coord → peer-0 → ...
       [blocked]      [blocked] [idle coord]    [idle]  [peer-0]
```

With pipelining:

```
t=0:   coord → peer-0 → peer-1 → coord(sample N)   → peer-0(draft N+1 layers 0-11)
                                → peer-0(verify N+1 output) → peer-1(layers 12-23)
       [coord always has the next draft in flight, peers never idle]
```

Two variants — we pick whichever fits our existing ring better:

#### Variant A1 — Draft-on-peer-0, verify via full ring

Approach:
1. Coord's ring request to peer-0 carries both the current token AND a
   "speculative draft budget" (`k` draft tokens).
2. Peer-0 runs its layers once on the committed token, then locally runs
   a **tiny draft model** (e.g. Qwen3.5-0.8B already in our catalog)
   for `k` additional steps to produce a draft token sequence.
3. Peer-0 sends the hidden state for the committed token → peer-1 as
   today, AND pushes `k` draft tokens to peer-1 as batched inputs.
4. Peer-1 runs its layers over the `k+1`-token batch in one forward pass
   (`batched verify`), returns `k+1` hidden states to coord.
5. Coord samples all `k+1` positions, compares against drafts, accepts
   the longest prefix where sampled == drafted.

Accept rate on small-draft-model / Qwen3.5-2B pairs is typically 60-75%
greedy or 40-55% sampling. Expected effective TPS:
`3.76 × (1 + accept_rate × k / (k + verify_overhead))`. For `k=3`, accept=0.6:
`3.76 × (1 + 1.8 / 3.3) ≈ 5.8 TPS`.

#### Variant A2 — Inter-token pipelining, no draft model

Approach:
1. Coord maintains **two rings in flight**: ring for token N + ring for
   token N+1, offset by one stage.
2. When peer-1 finishes computing token N (and sends PushResult), it
   immediately accepts peer-0's already-queued input for token N+1 and
   starts computing that. Meanwhile coord samples N, fires re-inject
   for token N+2.
3. Peer-0 is continuously busy: finishing N+1's embed+layers-0-11 while
   N was still on peer-1.

No draft model needed, no verification. Pure pipelining.

Expected TPS: up to `2×` on perfectly-balanced peers, bounded by
`max(peer_0_compute, peer_1_compute)` + wire. For balanced 2-node: ~7 TPS.

### Risk / complexity

- **Ring-state machine becomes nontrivial.** Today's ring is a linear
  state (`remaining`, `generated`). With pipelining we need per-token
  slots (`slot[N].state ∈ {in_flight_p0, in_flight_p1, sampled}`) and
  out-of-order arrival handling.
- **KV cache consistency.** Variant A2 has peer-0 running token N+1's
  forward *before* coord has sampled token N. If sampling picks a
  different token than the ring seed, peer-0's KV cache entry for
  N+1 is poisoned and must be rolled back. Verify-and-rewind logic is
  the hard part.
- **Variant A1 (speculative) avoids the rewind** because we verify
  every draft — but adds the draft model inference on peer-0.

### Files touched (estimate)

| File | Changes | Why |
|---|---|---|
| `peer/peer.proto` | +4 fields on `ForwardRequest` / `ForwardResponse` for draft tokens + slot id | Wire protocol for pipelining |
| `coordinator/head_sampler.py` | `RingSession` → `PipelinedRingSession` with per-slot state dict | Out-of-order arrival |
| `coordinator/chain.py::run_push_ring` | Fire `n_pipeline` concurrent rings, wait on per-slot reply | Multi-ring driver |
| `peer/server.py` last-peer branch | Handle batched-verify response; carry slot_id on PushResult | Peer-side slot routing |
| `peer/model_shard.py` | Optional draft-model loading for Variant A1; KV cache snapshot+rollback for Variant A2 | Speculation mechanics |
| `tests/` | +new test file `test_pipelined_ring.py` | Correctness under rewind / accept |

Estimated ~1,500-2,000 LoC across Python + proto, ~50 LoC Rust (proto field propagation only).

### Recommended target

**Variant A2 (inter-token pipelining, no draft model)** is the closer
fit. We already have the ring primitives and the existing
`SpecPipe`/`specpipe_enabled` scaffolding was built for roughly this
shape (even though it was never wired end-to-end).

### Verification

- Unit: `test_pipelined_ring.py` — multi-slot session, simulate out-of-
  order PushResult arrival, verify the final token stream is
  identical to the serial-ring baseline.
- Integration: run the existing 3-node True Petals benchmark with
  `--pipeline-depth 2` and confirm tokens match the serial run
  byte-for-byte (deterministic seed) AND TPS is ≥1.8×.
- Rollback: force-inject a token mismatch at sampler time, verify the
  ring recovers and produces the correct sequence.

---

## Plan (b) — Compute/I-O Overlap within a Peer (CUDA streams + pinned buffers)

**Goal:** while peer-1 is sending the hidden state for token N over
the wire to coord, peer-1's next-token receive + layers 12-23 forward
for token N+2 (arriving from peer-0 on the re-inject cycle) can *already
be in flight on a separate CUDA stream*.

**Expected TPS multiplier:** ~1.2-1.5× — bounded by the fraction of
per-cycle time that is wire-send vs compute. Today's split is roughly
~30% wire / ~60% compute / ~10% sampling, so max overlap is ~30%.

**Scope:** the smaller of the two. ~0.5-1 day of focused work + bench.

### The core idea

Within a single peer process, two operations happen sequentially today:

```
PyTorch forward on CUDA:  |████████████████████|                      |████...
gRPC send hidden state:                        |██████████████|
```

With CUDA streams + async serialization:

```
PyTorch forward on CUDA:  |████████████████████|████████████████████|████...
gRPC send hidden state:                        |██████████████|
                                                  ^-- overlapped with the next forward
```

The serialize+send step runs on a CPU thread, reading from pinned
memory; the CUDA forward uses a separate stream so the host's
copy-to-pinned for the *current* hidden state doesn't block the
*next* forward from starting.

### Mechanics

Current `_forward_impl` in `peer/model_shard.py::PyTorchRuntime`:

```python
hidden, next_past = self._run_layers(hidden, ...)  # CUDA compute on default stream
output = self._hidden_to_packed_bytes(hidden)        # d2h copy + serialize
# (server.py then sends output over gRPC)
```

Target:

```python
hidden, next_past = self._run_layers(hidden, ...)              # stream A
self._async_d2h_copy(hidden, buffer=self._pinned_out_buffer)    # stream B (copy-only)
# return a "future" that resolves when stream B completes
# control returns to server.py which fires gRPC send when the future resolves,
# while stream A is already running the NEXT forward on the re-inject input
```

### Key techniques

1. **CUDA streams.** Dedicate one stream to compute (`torch.cuda.Stream()`)
   and one to host-copy. `torch.cuda.current_stream().wait_stream(...)`
   for explicit dependencies.
2. **Pinned output buffer.** Pre-allocate
   `torch.empty(max_seq, hidden_size, dtype=..., pin_memory=True)` at
   peer startup. All hidden-state d2h copies reuse this buffer instead
   of allocating every cycle.
3. **Async serialize.** The serialize path in `_hidden_to_packed_bytes`
   → `openhydra_network.encode_activation` is already Rust + zero-copy
   on the Python side, but the d2h step is synchronous. Split it:
   async `copy_(..., non_blocking=True)` + explicit
   `stream.synchronize()` before the encode call.
4. **Concurrent forward.** The receive side (`_activation_to_hidden`)
   does h2d via `tensor.to(device)`. Make this `non_blocking=True`
   and issue on a separate stream so the previous cycle's send can
   still be on the wire.

### Risk / complexity

- **Smaller than (a) but subtle.** CUDA streams are easy to get wrong —
  a missing synchronize turns into nondeterministic output.
- **Only helps if wire-send is a real fraction of the per-cycle time.**
  On the 3.76 TPS baseline, wire per hop is ~15-35 ms out of ~266 ms.
  Overlap theoretical ceiling is ~30% → ~4.9 TPS (estimated).
- **MLX equivalent is different.** For the Mac coord case, MLX has
  `mx.eval` lazy execution — we'd overlap by not calling `mx.eval`
  until the last moment. Similar idea, different mechanism.

### Files touched (estimate)

| File | Changes | Why |
|---|---|---|
| `peer/model_shard.py::PyTorchRuntime` | Add `_forward_stream`, `_copy_stream`, `_pinned_out_buffer`; split `_hidden_to_packed_bytes` into async d2h + sync encode | Core overlap |
| `peer/server.py::Forward` handler | Returns a future-like handle; proxy_forward fires when d2h is ready | Caller coordination |
| `peer/mlx_runtime.py::MLXRuntime` | Mirror with `mx.eval` deferral — don't eval the hidden state until just before serialize | MLX overlap |
| `tests/test_async_forward.py` | NEW — synthetic forward with injected delays, verify overlap windows | Correctness |

Estimated ~400-600 LoC across Python, no Rust or proto changes.

### Verification

- Unit: `test_async_forward.py` — mock torch.cuda streams; assert that
  `_async_d2h_copy` runs concurrently with the next forward's setup;
  regression-assert output bit-identity vs the sync path.
- Benchmark: re-run the 3-node True Petals benchmark with the new
  async path; capture `autoregressive_ring_done.tps` vs the 3.76 TPS
  baseline. Expected 4.5-5.0 TPS.

---

## Decision matrix

| Criterion | Plan (a) Pipeline parallelism | Plan (b) Compute/I-O overlap |
|---|---|---|
| TPS upside | ~1.8-2.2× (6.8-8.3 TPS) | ~1.2-1.5× (4.5-5.6 TPS) |
| Scope (LoC) | ~1,500-2,000 | ~400-600 |
| Proto change | Yes (+4 fields) | No |
| Rust change | Minor (proto propagation) | No |
| Correctness risk | High (out-of-order + rewind) | Medium (CUDA streams subtle) |
| Wall-clock effort | 2-3 focused days | 0.5-1 focused day |
| MLX/PyTorch symmetry | Requires both backends to pipeline | Different mechanism each side |
| Depends on other infra | — | — |
| Natural incremental | Can ship Variant A2 only, defer A1 | Can ship PyTorch-only, defer MLX |

## Recommendation

**Ship (b) first.** It's one focused session, gets us to ~5 TPS at low
risk, and lets us instrument the wire + compute fractions precisely —
which then tells us exactly how much (a) can give us on top.

Then **ship (a) — Variant A2** as a dedicated multi-session effort. The
expected stacked result is `3.76 × 1.3 × 2.0 ≈ 9.8 TPS` — crossing the
single-machine Qwen 3.5 2B on M1 MLX benchmark (6.9 TPS) on a ring
running across three distinct hosts.

If the user wants just one, the ROI-per-effort winner is **(b)**. If
the user wants the biggest number, the winner is stacking (a) + (b)
over two sessions.
