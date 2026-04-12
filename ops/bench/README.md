# OpenHydra Benchmarks

Benchmarks for validating sharded-inference code paths end-to-end against real
models on Lightning T4 GPUs. These scripts bypass parts of the coordinator on
purpose so they stay useful as a **control reference** when the coordinator
path itself is being debugged — every fix in
`plans/sharded-inference-fixes.md` landed in a session where one of these
benches was the only thing producing coherent output.

---

## Prerequisites

- Two Lightning Studios with a T4 GPU each (`s_01knk4fr41phzt8w0ad8q49cjy` and
  `s_01knk4ftdeeng3zctxh2zq0w3m` in the session history — substitute your own
  studio IDs).
- SSH tunnels from the Mac to each studio forwarding the peer gRPC port:
  ```bash
  ssh -o ServerAliveInterval=10 -N -L 50099:127.0.0.1:50051 <studio-1>@ssh.lightning.ai &
  ssh -o ServerAliveInterval=10 -N -L 50098:127.0.0.1:50051 <studio-2>@ssh.lightning.ai &
  ```
- `transformers>=5.5.0` locally so the Mac can load the Qwen 3.5 / Gemma 4
  tokenizer for decoding.

---

## gemma4_direct_bench.py

**Scope**: Single-GPU unsharded Gemma 4 benchmark using `model.generate()`
directly. Runs entirely on the remote studio — no coordinator, no OpenHydra
pipeline, no gRPC. Exists as the control reference for "Gemma 4 E4B-it works
with the transformers-native forward path".

**When to use**: you're debugging Gemma 4 and want a known-good baseline TPS
number that isolates the model from every OpenHydra layer. Also useful when
`_run_layers_gemma4` is changed and you need to verify the single-peer path
still works before testing multi-peer shards.

**Usage** (from a Lightning studio):
```bash
cd /teamspace/studios/this_studio/openhydra
source /teamspace/studios/this_studio/openhydra-venv/bin/activate
python3 ops/bench/gemma4_direct_bench.py \
    --max-new-tokens 64 \
    --runs 3 \
    --prompt "Write a detailed 100-word paragraph about the history of the Pacific Ocean"
```

**Expected numbers** (Lightning T4, fp16, greedy, 95-token output, ~14.5 GB
VRAM, one-time cold load ~190 s):
- Studio 1: **~10.7 TPS** warm
- Studio 2: **~8.9 TPS** warm

If TPS drops below 5 the issue is likely the `flash-linear-attention` /
`causal-conv1d` fallbacks — install them on the studio venv for the fast path.

---

## qwen9b_sharded_bench.py

**Scope**: 2-peer sharded Qwen 3.5 9B benchmark. Drives both peers manually
over gRPC from the Mac, bypassing the OpenHydra coordinator's decode loop
entirely. Each decode step tokenises on the Mac, sends the full context as
`activation` (stage 0) + `prompt_token_ids` (sidecar) to peer 1, receives the
hidden state back, ships it to peer 2 as the next stage's activation, receives
one sampled token, appends it, and repeats — a stateless re-prefill per token
with no KV cache reuse.

**When to use**:
1. You want to validate that `_run_layers` is correct on the peer side
   without being affected by the coordinator's decode loop bugs.
2. The coordinator is mis-routing / mis-sampling / dropping tokens and you
   need to isolate whether the peers' forward math is correct.
3. You want real-but-conservative TPS numbers for the current stateless
   re-prefill design (before KV-aware decoding lands).

**Usage** (from the Mac, with both SSH tunnels up):
```bash
cd "/Users/sam/Documents/New project 2"
python3 ops/bench/qwen9b_sharded_bench.py \
    --peer1 127.0.0.1:50099 \
    --peer2 127.0.0.1:50098 \
    --tokenizer Qwen/Qwen3.5-9B \
    --prompt "What is 17 times 23? Answer briefly." \
    --max-new-tokens 24 \
    --runs 1
```

The bench can also be pointed at the Gemma 4 peers — just change
`--tokenizer google/gemma-4-E4B-it` and make sure the Gemma 4 peers are the
ones listening on 50099/50098.

**Expected numbers** (Lightning T4×2, SSH-tunnelled WAN, no KV cache reuse):

| Prompt | Model | Tokens | Elapsed | TPS |
|---|---|---|---|---|
| "Hi" | Qwen3.5-9B | 4 | 5.95 s | 0.67 |
| "Say hello in five words." | Qwen3.5-9B | 8 | 25.4 s | 0.32 |
| "What is 17 times 23? Answer briefly." | Qwen3.5-9B | 24 | 54.9 s | 0.44 |
| "Write one sentence about the ocean." | Gemma4-E4B-it | 20 | 40.0 s | 0.50 |

The O(N²) cost is expected — this script deliberately runs stateless re-prefill
to be useful as a control reference. The **coordinator's**
``/v1/completions`` and ``/v1/chat/completions`` endpoints now use the Phase 6
KV-aware decode loop and run **~3× faster** for the same generation length —
see "Coordinator KV-aware path" below for measured numbers.

---

## Coordinator KV-aware path (Phase 6)

After Phases 1+2+3+6 landed, the recommended way to drive sharded inference is
through the OpenHydra coordinator's standard endpoints. The coordinator runs
prefill once (full context) and then ships **a single token per decode step**
through the pipeline, reading peer-side KV cache state on every hop.

**Expected numbers** (Lightning T4×2, SSH-tunnelled WAN, KV cache reuse):

| Prompt | Endpoint | Model | Tokens | Elapsed | TPS |
|---|---|---|---|---|---|
| "What is 17 times 23?" | `/v1/completions` | Qwen3.5-9B | 16 | 25.5 s | **0.63** |
| "Explain sharding in one paragraph." | `/v1/completions` | Qwen3.5-9B | 32 | 49.9 s | **0.64** |
| "Name three programming languages." | `/v1/chat/completions` | Qwen3.5-9B | 24 | 35.8 s | **0.67** |

Pipeline trace shows decode steps cost a constant ~1.35 s each regardless of
context length (one round-trip per peer over the WAN tunnel) instead of growing
linearly with the prompt size. The **prefill** is still the single dominant
cost (~7 s for a 37-token prompt) — co-locating the coordinator with the peers
on the same VPC removes the SSH-tunnel RTT entirely and pushes both prefill
and decode below 200 ms each.

If a peer fails to handle the KV-aware request (e.g. an old build without
``conv1d_t4_fallback``, or a ``DynamicCache`` schema mismatch), the coordinator
transparently falls back to stateless re-prefill for the rest of the request
and logs a one-line warning:

```
WARNING coordinator.inference_service autoregressive_sharded_kv_fallback: step=1 err=...
INFO    coordinator.inference_service autoregressive_sharded_done: tokens=N latency_ms=... mode=stateless
```

You can opt out of the KV-aware path entirely with
``--no-autoregressive-sharded`` on ``coordinator.api_server`` — that restores
Phase 1 behavior (O(N²) stateless re-prefill).

---

## Launching peers

The setup used for the numbers above launches each peer as a full
`coordinator.node` process (which starts both a peer gRPC server on port 50051
AND a local coordinator HTTP server on 8081 — the local coordinator is
ignored; only the peer matters for the bench):

**Qwen 3.5 9B** — 2-peer shards `0-15` and `16-31`:
```bash
# studio 1
python3 -m coordinator.node \
    --peer-id gpu1-qwen9b \
    --model-id openhydra-qwen3.5-9b \
    --runtime-model-id /teamspace/studios/this_studio/openhydra/models/Qwen3.5-9B \
    --runtime-backend pytorch_auto \
    --layer-start 0 --layer-end 16 --shard-index 0 --total-shards 2 \
    --grpc-port 50051 --api-port 8081 --api-host 0.0.0.0 \
    --dht-url http://172.104.164.98:8468

# studio 2 — same flags but --layer-start 16 --layer-end 32 --shard-index 1
```

**Gemma 4 E4B-it** — 2-peer shards `0-20` and `21-41`. Gemma 4 is the
**only** family where the shard boundary matters: because of KV sharing
(`num_kv_shared_layers=18`, shared layers 24-41), every shard that contains a
KV-shared layer MUST also contain layers 22 and 23 (the "full-length KV
storing" anchors). The 21-layer 0-20/21-41 split satisfies this — peer 2 has
layers 22+23 AND the shared layers 24-41. A 0-30/31-41 split would fail with
`gemma4_shard_split_breaks_kv_sharing` (until Phase 5 adds cross-peer cache
serialisation).

```bash
# studio 1
python3 -m coordinator.node \
    --peer-id gpu1-gemma --model-id openhydra-gemma4-e4b-it \
    --runtime-model-id /teamspace/studios/this_studio/openhydra/models/gemma-4-E4B-it \
    --runtime-backend pytorch_auto \
    --layer-start 0 --layer-end 21 --shard-index 0 --total-shards 2 \
    --grpc-port 50051 --api-port 8081 --api-host 0.0.0.0 \
    --dht-url http://172.104.164.98:8468

# studio 2 — same flags but --layer-start 21 --layer-end 42 --shard-index 1
```

Both Gemma 4 peers log `multimodal_strip: kept=21 replaced=21 target=0` at
startup, confirming the vision/audio towers were dropped and the out-of-shard
text layers were replaced with `nn.Identity` (Phase 2B fix to avoid meta-tensor
dispatcher errors).

---

## Known gotchas

- **`from peer import peer_pb2_grpc` fails on Lightning** — `grpc_tools.protoc`
  generates `import peer_pb2 as peer__pb2` (unqualified) instead of the
  package-relative form. If you regenerate the proto, manually fix the import
  line in `peer/peer_pb2_grpc.py` to `from peer import peer_pb2 as peer__pb2`
  (documented in `plans/memory.md`).

- **Qwen 3.5 linear_attn fallback** — `fla-org/flash-linear-attention` +
  `Dao-AILab/causal-conv1d` are not installed by default on Lightning
  studios. Qwen 3.5 runs in the torch fallback for its `Qwen3_5GatedDeltaNet`
  layers, which is ~3-5× slower than the fused kernel. Install both for
  realistic sharded TPS numbers.

- **Tokenizer caching on the Mac** — the coordinator's auto-regressive decode
  loop calls `AutoTokenizer.from_pretrained(model_id, local_files_only=True)`
  first and only falls back to HF Hub on `OSError`. The first request for a
  new model fails with "Couldn't instantiate the backend tokenizer" (not an
  `OSError` — the exception propagates up and the coordinator takes the
  legacy fallback path). Workaround: call
  `AutoTokenizer.from_pretrained("<model_id>")` once manually to populate the
  local cache, then restart the coordinator.
