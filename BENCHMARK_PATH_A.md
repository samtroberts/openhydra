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
