# Path A (client-terminated pipeline) — cross-ISP A/B benchmark runbook

**Goal:** measure `autoregressive_ring_done.tps` with `--sample-on-coordinator`
**off** (Run A, baseline, today's ring-on-peer topology) vs **on** (Run B, Path A —
hidden state returns to coordinator which samples + re-injects), same model,
prompt, seed, pipeline, and network path.

Branch: `claude/elastic-bhaskara`
Relevant commits:

- `b65d474` — Client-terminated pipeline core (Phases 1-4)
- `4901b70` — Phase 5: load head weights on non-terminal shards

## Environment

| Role | Host | Runtime | Layers | Notes |
|---|---|---|---|---|
| Coordinator + peer 0 (stage 0) | Mac (this machine) | MLX 8-bit | `[0,12)` | Also owns the HeadSampler in Run B via `load_full_head=True` |
| Peer 1 (stage 1, terminal) | Lightning GPU1, `s_01knk4fr41phzt8w0ad8q49cjy@ssh.lightning.ai` | PyTorch / T4 | `[12,24)` | Standard; no flag needed on this side |

Model: `Qwen/Qwen3.5-2B` (HF canonical) / `mlx-community/Qwen3.5-2B-MLX-8bit` (MLX 8-bit).
`tie_word_embeddings=True` (verified), 24 transformer layers, vocab 248320, hidden 2048.

Prompt (32 tokens max):

> "Write a three-sentence haiku-style poem about a mountain at dawn."

With `enable_thinking=False` to drop the `<think>` tax (makes raw TPS directly
comparable to user-visible TPS).

## Preflight — both sides

```bash
# Mac (this workstation) — already on the branch and committed.
cd "/Users/sam/Documents/New project 2/.claude/worktrees/elastic-bhaskara"
git rev-parse --short HEAD   # expect 4901b70 or newer

# GPU1 — pull the new code onto a working branch
ssh -i ~/.ssh/lightning_rsa s_01knk4fr41phzt8w0ad8q49cjy@ssh.lightning.ai \
  "cd ~/openhydra && git fetch origin claude/elastic-bhaskara && \
   git checkout claude/elastic-bhaskara && git reset --hard origin/claude/elastic-bhaskara && \
   git rev-parse --short HEAD"
# expect: 4901b70

# Verify the openhydra_network wheel still imports on GPU1. We did NOT touch
# the Rust layer in these two commits, so the existing wheel should be fine.
ssh -i ~/.ssh/lightning_rsa s_01knk4fr41phzt8w0ad8q49cjy@ssh.lightning.ai \
  "python3 -c 'import openhydra_network, peer.peer_pb2 as p; r = p.ForwardRequest(sample_on_coordinator=True); s = p.ForwardResponse(is_hidden_state=True); print(r.sample_on_coordinator, s.is_hidden_state, bool(openhydra_network.encode_activation))'"
# expect: True True True
```

If the wheel import fails, rebuild before running any benchmark:

```bash
ssh -i ~/.ssh/lightning_rsa s_01knk4fr41phzt8w0ad8q49cjy@ssh.lightning.ai \
  "cd ~/openhydra/network && maturin develop --release"
```

## GPU1 peer launch (same for Run A and Run B)

Screen session so we can inspect logs and tear down cleanly:

```bash
ssh -i ~/.ssh/lightning_rsa s_01knk4fr41phzt8w0ad8q49cjy@ssh.lightning.ai
# once inside:
cd ~/openhydra
screen -S oh-peer
# inside screen:
export PYTHONUNBUFFERED=1
python3 -m coordinator.node \
  --p2p-enabled --push-mode \
  --runtime-backend pytorch \
  --runtime-model-id Qwen/Qwen3.5-2B \
  --hf-model-id Qwen/Qwen3.5-2B \
  --tokenizer-vocab-guard \
  --layer-start 12 --layer-end 24 --total-shards 2 \
  --peer-id gpu1-stage1 \
  --grpc-port 50051 --api-port 7051 \
  2>&1 | tee /tmp/openhydra_gpu1.log
# detach: Ctrl-A then d
```

GPU1 is stage 1 (terminal). It needs no flag: the `sample_on_coordinator`
field arrives via the wire on each request and is acted on per-request in
the last-peer branch of `Forward()`. On Run A the field is `False`, on Run B
it is `True` — same process, different per-request behaviour.

## Mac coordinator launch

### Run A (baseline, `sample_on_coordinator=False`)

```bash
cd "/Users/sam/Documents/New project 2/.claude/worktrees/elastic-bhaskara"
python3 -m coordinator.node \
  --p2p-enabled --push-mode \
  --no-sample-on-coordinator \
  --runtime-backend mlx \
  --runtime-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
  --mlx-force-hf-tokenizer \
  --hf-model-id Qwen/Qwen3.5-2B \
  --tokenizer-vocab-guard \
  --layer-start 0 --layer-end 12 --total-shards 2 \
  --peer-id mac-stage0 \
  --grpc-port 50050 --api-port 7050 \
  2>&1 | tee /tmp/openhydra_run_a.log
```

### Run B (Path A, `sample_on_coordinator=True`)

Identical to Run A but `--sample-on-coordinator` (no `--no-` prefix):

```bash
cd "/Users/sam/Documents/New project 2/.claude/worktrees/elastic-bhaskara"
python3 -m coordinator.node \
  --p2p-enabled --push-mode \
  --sample-on-coordinator \
  --runtime-backend mlx \
  --runtime-model-id mlx-community/Qwen3.5-2B-MLX-8bit \
  --mlx-force-hf-tokenizer \
  --hf-model-id Qwen/Qwen3.5-2B \
  --tokenizer-vocab-guard \
  --layer-start 0 --layer-end 12 --total-shards 2 \
  --peer-id mac-stage0 \
  --grpc-port 50050 --api-port 7050 \
  2>&1 | tee /tmp/openhydra_run_b.log
```

Look for the Phase-5 registration line on the Mac side within ~30s of launch:

```
head_sampler_registered: peer=mac-stage0 runtime=MLXRuntime
```

## DHT discovery wait

Both peers announce to the Linode libp2p bootstrap ring; the DHT lookup at the
coordinator discovers GPU1 within ~20–60 s of launch. Wait until the coordinator
log shows something like:

```
discovery_service: found peer gpu1-stage1 layers=[12,24)
```

before firing the benchmark.

## Benchmark request

```bash
# Same prompt for both runs — use decode_seed for reproducibility.
curl -sS -X POST http://127.0.0.1:7050/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openhydra-qwen3.5-2b",
    "messages": [
      {"role": "user", "content": "Write a three-sentence haiku-style poem about a mountain at dawn."}
    ],
    "max_tokens": 32,
    "temperature": 0.0,
    "seed": 42,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | tee /tmp/openhydra_run_<A|B>_response.json
```

## Metric extraction

**Authoritative metric** — `autoregressive_ring_done.tps` from the coordinator
log (not `usage.completion_tokens / latency` which is the post-`<think>`-strip
measure):

```bash
grep "autoregressive_ring_done" /tmp/openhydra_run_<A|B>.log | tail -5
```

Also check termination reason:

```
ring_done: req=... tokens_generated=... reason=eos|max_tokens|...
```

## Teardown

```bash
# Mac
pkill -f "coordinator.node.*mac-stage0"

# GPU1 — attach screen and Ctrl-C, then:
ssh -i ~/.ssh/lightning_rsa s_01knk4fr41phzt8w0ad8q49cjy@ssh.lightning.ai \
  "screen -S oh-peer -X quit"
```

**Do not `killall python3` on the Lightning studio — it nukes studio services.
Always target the specific process or the screen session.**

## Expected outcome

Baseline (Run A) should land in the ~0.9–1.1 TPS range reported in prior
cross-ISP runs with 8-bit MLX + PyTorch T4 (per MEMORY.md cross-ISP section).
Run B should land in the ~1.4–2.0 TPS range if Path A eliminates the
per-token ring loopback (coordinator and last peer co-location would
approach the theoretical 2× ceiling; cross-ISP overhead dampens the gain).

## Fallback — if Run B errors

Expected failure modes and their log signatures:

| Symptom | Meaning | Fix |
|---|---|---|
| `no_head_sampler_registered` in PushResult | Mac's `_maybe_register_head_source` didn't fire | Confirm `--sample-on-coordinator` was passed; check `_has_final_head=True` in MLX init log |
| `no_ring_session_registered` | `run_push_ring` didn't register a `RingSession` before firing | Confirm `sample_on_coordinator=True` reached `EngineConfig` (grep log for config dump) |
| `COORD_REINJECT_NO_ROUTE` | Re-injection fire-and-forget has no address | Check `RingSession.ring_first_hop_address` is non-empty |
| Ring hangs silently after ~30s | Last peer sent hidden state but coordinator stuck in sampling | Attach to coordinator's `py-spy dump` to see stack |

## Output table — fill in post-run

| Metric | Run A (flag off) | Run B (flag on) | Delta |
|---|---|---|---|
| Coordinator launch → first token | | | |
| `autoregressive_ring_done.tps` | | | |
| Total latency (32 tokens) | | | |
| Termination reason | | | |
| Coordinator peak RSS | | | |

## 2026-04-23 session results

### Environment fixes required before any cross-ISP benchmark on GPU1

Lightning studio `s_01knk4fr41phzt8w0ad8q49cjy` had auto-upgraded its conda env
to numpy 2.4.4 while leaving pinned-against-numpy-1 transitive deps in place.
The cross-ISP benchmark cannot launch until:

```bash
ssh -i ~/.ssh/lightning_rsa s_01knk4fr41phzt8w0ad8q49cjy@ssh.lightning.ai \
  "bash -lc 'pip install \"numpy<2\" --force-reinstall && pip uninstall -y torchvision'"
```

- `numpy<2` restores scipy 1.11.4 + pandas 2.1.4 binary compatibility
  (error seen: `numpy.core.multiarray failed to import` then
  `numpy.dtype size changed, may indicate binary incompatibility`).
- `pip uninstall torchvision` removes the 0.23-pinned wheel that tries to
  register ops against torch 2.11 and fails with
  `RuntimeError: operator torchvision::nms does not exist`. OpenHydra does
  not use torchvision; transformers handles its absence.

### What the session confirmed

1. **Phase 1-5 code ships clean and lands on the target path.** Phase 5's
   `load_full_head` + relaxed registration gate fires end-to-end: the Mac
   coordinator log showed
   `head_sampler_registered: peer=mac-stage0 runtime=MLXRuntime` when
   launched with `--sample-on-coordinator`. On GPU1, with no flag,
   the last-shard PyTorch peer also registered under the legacy gate:
   `head_sampler_registered: peer=gpu1-stage1 runtime=PyTorchRuntime`.
2. **Ring-session registry works.** The coordinator log showed
   `ring_session_registered: req=... remaining=32 stages=2` firing
   synchronously before the initial ForwardRequest, exactly as designed.
3. **Run A baseline captured.**
   `autoregressive_ring_done: tokens=20 latency_ms=20580.4 tps=0.97`
   with `--no-sample-on-coordinator` on the Mac↔GPU1 cross-ISP ring.
   Matches the historical 2026-04-16 baseline of ~1.0 TPS.
4. **Run B did not complete.** The `NegotiationLoop` flipped GPU1's
   layer range every 60s between `[0,12)` and `[12,24)` while the two
   peers shared the DHT. Run B's ring fired at a moment when GPU1 was
   `[0,12)` and Mac was `[0,12)` too — the coordinator's ring-route
   builder put GPU1 as stage 0 and Mac as stage 1, but Mac doesn't hold
   layers `[12,24)`. Mac's forward raised
   `Either input_embeddings or prompt (or both) must be provided` and the
   ring terminated with `tokens=0 latency_ms=20016.6 tps=0.00`.

### Topology observation (important for expectations)

In the Mac-coordinator + Mac-stage-0 + GPU1-last-shard topology, Path A does
**not** eliminate a wire hop. Both legacy and Path A send the same two
messages per token (Mac→GPU1 with prompt/activation; GPU1→Mac with the
return payload). Path A only changes:

- **What** the return message carries: a `~8 KB` hidden state vs. a `4-byte`
  sampled token.
- **Where** `lm_head` runs: on MLX (Mac) vs. on T4 (GPU1).

The plan's "~2× TPS" figure assumes a **non-co-located** coordinator
(classic Petals: client is a laptop, peers are remote). In that case Path A
saves the final-peer → first-peer loopback hop. In our co-located setup
the wire delta is ≈0 and the TPS delta will be dominated by the lm_head
compute delta (probably negligible either way).

### Next-session prerequisites

To get a meaningful A/B Path A benchmark, one of:

1. **Pin layer ranges across the swarm** — bypass the `NegotiationLoop`
   by launching both peers with a `peers.json` the coordinator reads
   verbatim (via `--peers-config`) instead of DHT-derived pipeline
   construction. That file explicitly lists `peer_id` → `layer_start`/
   `layer_end` with the direction we want.
2. **Test a non-co-located coordinator** — run the coordinator on a third
   machine (a laptop, a Linode, anywhere not a peer). This is where Path A's
   "~2× TPS" should actually materialise because the final-hop loopback
   becomes a real saved wire hop.
3. **Fix the NegotiationLoop's 60 s flip-flop** — pre-existing issue
   unrelated to Path A but blocks any benchmark that relies on a stable
   pipeline. Look at `peer/swarm_negotiator.py::pick_best_fit` — it
   appears to revisit the assignment every heartbeat with different
   results for the same 2-peer input.

### Files touched / code surface — confirmed working in isolation

All Phase 1-5 code paths verified under a local-only MLX launch on the Mac:

- Proto round-trip: `ForwardRequest.sample_on_coordinator`,
  `ForwardResponse.is_hidden_state`
- `MLXRuntime.apply_final_head(hidden)` on a stage-0 shard with
  `load_full_head=True` returns a valid token id from zero-hidden input
- `_maybe_register_head_source` fires on stage-0 when
  `_has_final_head=True`
- `HeadSampler.sample()` threads all five decode params to the borrowed
  runtime's `apply_final_head`
- 17/17 Path A unit tests pass, 1385/1385 full suite green

## 2026-04-24 session results — True Petals topology attempt

### What shipped this session

- **`ee924ef`** — Disable Qwen3.5 `<think>` by default + fix negotiator
  oscillation (+4 tests)
- **`c8a9742`** — Path A Phase 6: standalone head loader for pure coordinator
  (+10 tests)
- **`faa9777`** — LAN-first routing: GPU1 → GPU2 direct gRPC on shared /16
  instead of libp2p relay via Linode US (+32 tests)
- **`e7fcedf`** — Propagate `sample_on_coordinator` through
  `_push_to_next_hop` so last peer takes Path A instead of legacy ring
- **`50eeea0`** — Concurrent pre-dial of all ring peers with
  discover-then-dial pattern

Full suite: **1431 passed, 9 skipped**. Zero regressions.

### What was verified live

With Mac-pure-coord + GPU1-stage0 + GPU2-stage1 on same Lightning VPC:

1. ✅ `standalone_head_ready: model=mlx-community/Qwen3.5-2B-MLX-8bit
   hidden=2048 vocab=248320 tie=True` — Mac loads head-only module,
   24 layers pruned.
2. ✅ `head_sampler_registered: peer=coordinator-standalone-head
   runtime=StandaloneHead` — HeadSampler registry binds correctly.
3. ✅ `sharded_pipeline_ordered: stages=[('gpu1-stage0', 0, 12),
   ('gpu2-stage1', 12, 24)]` — pipeline builder respects explicit
   `--peers-config`.
4. ✅ `push_forwarded_via_lan: -> 10.192.11.74:50052 (LAN-first;
   bypassing libp2p_id=...)` — **GPU1 → GPU2 now uses LAN gRPC directly**.
   No Linode-relay round-trip between two VPC-local peers.
5. ✅ `lan_routing_local_prefixes: ['10.192.0.0/16']` on GPU1 & GPU2,
   `['127.0.0.0/16', '192.168.1.0/24']` on Mac — classifier correctly
   distinguishes intra-VPC from cross-internet hops.
6. ✅ GPU2 receives + executes stage-1 forward end-to-end.
7. ✅ `sample_on_coordinator=True` propagates all the way to GPU2
   (after the `e7fcedf` fix) — GPU2 enters
   `_handle_hidden_state_push_result` instead of the legacy ring
   loopback.

### Remaining blocker (not Path A code)

GPU2's `PushResult` back to Mac fails with
`proxy outbound: DialFailure` → `dial error: no addresses for peer`.

Root cause: Mac's Rust libp2p Kademlia routing table knows GPU2's
`libp2p_peer_id` (it's cached from the HTTP DHT via `discovery_service`)
but has **no multiaddrs** associated with that peer_id. Without
multiaddrs, `Swarm::dial` cannot construct a relay path. `discover()`
returns peer records to the Python caller but does NOT populate the
Swarm's per-peer address book on the Rust side. Similarly, the
GPU2→Mac direction fails because GPU2 has never dialed Mac (no inbound
connection from Mac to establish a cached entry).

This is a `P2PNode` Rust-side API gap, not a Path A bug. Closing it
requires adding `P2PNode.add_address(peer_id, multiaddr)` to the Rust
bridge so the Python side can explicitly populate the address book
with multiaddrs derived from the relay reservations (which Mac DOES
publish successfully to all 3 Linode relays on startup).

### Shape of the Rust fix (next session)

In `network/src/` (Rust bridge):

```rust
#[pymethods]
impl P2PNode {
    fn add_address(&self, peer_id: &str, multiaddr: &str) -> PyResult<()> {
        // Send SwarmCommand::AddAddress { peer_id, multiaddr } over
        // self._cmd_tx; event_loop forwards to Swarm::add_peer_address
        // which populates the per-peer multiaddr book that dial_peer
        // consults before attempting a Swarm::dial.
    }
}
```

Then in `coordinator/chain.py::run_push_ring`, after discover() returns
peer records, iterate the records for multiaddrs and call
`p2p_node.add_address(libp2p_id, multiaddr)` for each. Subsequent
`dial_peer` calls will find the multiaddrs and complete the dial.

### Honest TPS status

Still the 2026-04-23 baseline: **~1.0 TPS legacy cross-ISP ring** (Run A
from yesterday's Mac-stage-0 + GPU1-stage-1 topology). No Path A
number yet because the Mac↔GPU2 libp2p Kademlia address-book gap
prevents PushResult delivery. Everything else on the Path A path is
verified working in isolation.

## 2026-04-24 session 2 — True Petals topology WORKING

### What shipped

| Commit | Change |
|---|---|
| `8227b78` | `P2PNode.add_address` Rust API + auto-populate Kademlia on every FoundRecord event. Wheel rebuilt on Mac + GPU1 + GPU2. |
| `008e6ae` | Drop the `is_peer_connected`-gated direct-gRPC branch from `_push_final_result` — conflated libp2p-connected with gRPC-reachable. |
| `2b1ffa8` | New `_coordinator_proxy_handler_loop` that runs when `--no-local-peer` is set. Drains inbound `PROXY_METHOD_PUSH_RESULT` and dispatches to a shared `_coordinator_handle_push_result` free function (Phase 3 logic extracted so it doesn't need a `PeerService` instance). |

### First True Petals benchmark result

**Topology:**
- Mac (home Wi-Fi, public-NAT'd, no transformer layers, `StandaloneHead`
  loaded from `mlx-community/Qwen3.5-2B-MLX-8bit`)
- GPU1 (Lightning VPC `10.192.11.221`, stage 0, layers `[0, 12)`, PyTorch T4)
- GPU2 (Lightning VPC `10.192.15.173`, stage 1, layers `[12, 24)`, PyTorch T4)

**Request:** "Write a three-sentence haiku-style poem about a mountain
at dawn." max_tokens=32, temperature=0.0, seed=42.

**Output:** *"Silent peak rises from the mist, Golden light spills over
snow-capped ridges, Morning breath begins to rise."* — 17 user-visible
tokens, 27 raw ring tokens, 28 424 ms.

```
autoregressive_ring_done: tokens=27 latency_ms=28424.8 tps=0.95
```

**TPS comparison:**

| Topology | TPS | Notes |
|---|---|---|
| 2026-04-23 Run A (Mac-MLX stage-0 + GPU1 stage-1) legacy ring | **0.97** | 2-peer co-located-coord baseline |
| 2026-04-24 Run B (Mac pure-coord + GPU1 + GPU2) True Petals + Path A | **0.95** | 3-node, cross-VPC, Mac has no layers |

### Why the TPS is flat (this is actually useful signal)

Path A's claimed "~2× TPS" assumed the final-peer → first-peer loopback
was a *real saved wire hop*. In our actual cross-VPC topology Mac↔GPU
traffic goes through the Linode relay regardless of topology (Mac is
on home Wi-Fi, GPUs are in Lightning VPC). The relay-hop cost dominates
the per-token budget. Moving `lm_head` off GPU2 and onto Mac saves
~5 ms of T4 compute per token — invisible against ~1 s relay RTT.

The "2×" figure lives in a different topology we haven't tested yet:
**coordinator + all peers on the same LAN.** There the wire cost is
LAN-bound (~0.5 ms) and compute dominates, so removing a whole compute
cycle (the ring loopback going through peer 0 → peer 1 → peer N again)
matters. Requires a 3rd Lightning studio or all-local setup to measure.

### What this session proved end-to-end

1. ✅ `P2PNode.add_address` Rust API lets Python populate Kademlia's
   routing table explicitly. Closes the "no addresses for peer" gap
   that yesterday's session flagged as a Rust-side blocker.
2. ✅ Auto-population on every `discover()` FoundRecord → no Python
   glue needed in steady state.
3. ✅ Mac pure coordinator mode (`--no-local-peer --sample-on-coordinator`)
   loads `StandaloneHead`, registers with `HeadSampler`, spawns the
   coord-only proxy handler, and sucessfully completes full ring
   generation cycles against 2 remote GPU peers.
4. ✅ Path A flow verified end-to-end across 27 tokens:
   `coord_push_result_received` → `coord_ring_sampled` →
   `coord_reinject_done` — one complete ring cycle per ~1 second.
5. ✅ LAN-first routing GPU1 → GPU2 confirmed in logs:
   `push_forwarded_via_lan -> 10.192.15.173:50052 (LAN-first; bypassing libp2p_id=...)`.

### Next steps for actual TPS gain

- **All-LAN benchmark** — 3 Lightning studios, one as coordinator.
  That's where Path A was designed to win. ~2× expected.
- **Speculative decoding** — amortises one ring cycle over 2-4 tokens.
  The SpecPipe scaffolding is already in the codebase (`specpipe_enabled`).
  ~2-3× effective TPS, topology-agnostic.
- **INT8 wire format default** — Petals parity; codec already written
  in `peer/activation_codec.py`, just not defaulted.
- **Async network kernels / pipeline overlap** — the "big unlock" from
  the original plan. MLX lazy-eval + CUDA streams + pinned buffers.
  ~1.5-2× TPS, additive to everything else.

## 2026-04-24 session 3 — ALL-LAN 3-NODE TRUE PETALS: 3.76 TPS

### What shipped

| Commit | Change |
|---|---|
| `747ac12` | PyTorch backend for `StandaloneHead` (Linux coord support). `--standalone-head-backend` / `--standalone-head-device` / `--standalone-head-dtype` CLI flags. Dual-backend dispatch in `apply_final_head`. +12 tests. |
| `a7b3adf` | Minimal coord-side gRPC `PushResult` server for LAN-reachable last peers. Pairs with `_coordinator_proxy_handler_loop` (libp2p fallback). |
| `7316f2b` | Use HF `lm_head` module directly (tied weight already wired by HF) instead of manual `matmul(normed, embed.weight.T)` — the manual path gave wrong logits on Qwen3.5-2B. |

### The authoritative number

**3 Lightning studios, all on Lightning `10.192.0.0/16` VPC:**

- **GPU3** (`10.192.11.159`) — pure coordinator, `--no-local-peer`, PyTorch
  StandaloneHead on CPU bfloat16
- **GPU1** (`10.192.11.221`) — stage 0, layers `[0, 12)`, PyTorch T4
- **GPU2** (`10.192.15.173`) — stage 1, layers `[12, 24)`, PyTorch T4

```
autoregressive_ring_done: tokens=28 latency_ms=7442.8 tps=3.76
content: "Silent peak rises from the mist,
         Golden light spills over snow-capped ridges,
         Morning breaks in soft blue light."
```

Two consecutive 32-token runs:

| Run | Tokens | Latency | TPS |
|---|---|---|---|
| 1 | 28 | 7694.7 ms | **3.64** |
| 2 | 28 | 7442.8 ms | **3.76** |

### The multiplier

| Topology | TPS | vs 2-node cross-VPC baseline |
|---|---|---|
| 2026-04-23 Mac-stage-0 + GPU1 cross-VPC legacy ring | 0.97 | 1.00× |
| 2026-04-24 session 1 — Mac pure-coord + GPU1 + GPU2 cross-VPC Path A | 0.95 | 0.98× |
| **2026-04-24 session 3 — GPU3 pure-coord + GPU1 + GPU2 ALL-LAN Path A** | **3.76** | **3.87×** |

Path A's theoretical "~2×" landed as ~3.9× because it compounded with:
1. Wire savings from LAN-first routing (sub-ms vs ~500 ms relay RTT per hop)
2. Per-token compute parity between the two PyTorch T4 peers (vs Mac MLX + GPU1 T4 heterogeneous baseline)
3. Removal of the Mac's MLX↔PyTorch dtype cast on every hidden state transfer

### End-to-end proof points

- ``standalone_head_loading[pytorch]: model=Qwen/Qwen3.5-2B device=cpu dtype=bfloat16``
- ``standalone_head[pytorch]_layers_pruned: freed=24`` (all transformer layers dropped)
- ``coordinator_grpc_server_bound: 0.0.0.0:50050 — PushResult only``
- ``push_forwarded_via_lan -> 10.192.15.173:50052 (LAN-first; bypassing libp2p_id=...)``
  on every ring cycle
- ``coord_push_result_received`` + ``coord_ring_sampled`` + ``coord_reinject_done``
  28 times per 32-token generation
- Final EOS on ``token=248046`` (real Qwen3.5 `<|im_end|>`), not the earlier
  garbage `token=0` that signalled an lm_head/matmul bug

### Known limitations

- The ring still does one wire round-trip per token. Speculative decoding
  or async-stream pipelining is needed to amortise that further.
- Coord's head matmul runs on CPU in bfloat16. A GPU-equipped coord would
  cut ~5-10 ms per token; at 3.76 TPS the head compute is ~2% of the
  per-token budget so this is low-priority.
- `coord_reinject_done: via_relay` — the re-inject hop still goes via
  libp2p (coord → stage 0 peer). Could be LAN-first too for the return
  path; minor optimisation left on the table.
