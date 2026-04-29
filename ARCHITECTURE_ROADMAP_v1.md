# OpenHydra Architecture Roadmap — v1

**Status:** Immutable engineering blueprint. Supersede via `ARCHITECTURE_ROADMAP_v2.md`, do not mutate this file.
**Last updated:** 2026-04-30
**Branch baseline:** `claude/elastic-bhaskara` @ `dbc15d9`
**Suite baseline:** 1457 passed, 9 skipped, 1 deselected

This document formalises the architectural future of OpenHydra from the
completed Phase 2a (async network/compute decoupling) through Phase 4.1
(hidden-state compression on cross-ISP hops). It supersedes the per-phase
plan files in `/Users/sam/.claude/plans/` for the purpose of canonical
reference; those plan files remain the implementation playbooks for
their respective phases.

---

## Executive summary

OpenHydra is a peer-to-peer, layer-sharded LLM inference network. Each
peer holds a slice of transformer layers; the coordinator orchestrates a
ring that walks a hidden state through the slices, samples the next
token, and either re-injects (token-by-token, today) or moves to a
verify pass (Phase 2b onward).

The bottleneck stack reshapes as we descend the roadmap:

| Phase | Status | Bottleneck addressed | Expected TPS (all-LAN 3-node, Qwen3.5-9B) |
|---|---|---|---|
| 1 + Path A 1-6 | Done | Coordinator round-trip per token | 3.76 |
| **2a** | **Done** | Per-cycle peer idle time (compute waits on transmit) | ~5.0 (1.3×) |
| 2b — DFlash | Planned | Per-token network RTT (16 tokens / 1 RTT) | ~12-13 (3.2×) |
| 2b.1 — DDTree | Optional | Per-domain acceptance rate vs DFlash | ~13-15 (varies) |
| 2c — ForwardStream | Planned | Per-RTT gRPC handshake cost | +5-10% over 2b |
| 3 — Auto-negotiator | Planned | Operator burden of choosing topology | No TPS change; ops simplification |
| 4 — KV compression | Planned | KV cache VRAM caps context window | Enables longer contexts on 8GB Macs / 15GB T4s |
| 4.1 — Wire compression | Planned | Cross-ISP libp2p relay payload size | 2-4× cross-ISP TPS for relay-bound case |

The cumulative target is a working **Qwen3.5-9B at ≥10 TPS on the all-LAN
3-node ring** and **≥3 TPS on the cross-ISP Mac+GPU heterogeneous ring**,
with context windows ≥16k tokens supportable on 8GB peers — all on
commodity hardware, all without leaving the existing libp2p mesh.

---

## Phase 2a — Async Network/Compute Decoupling (DONE)

Shipped 2026-04-29 across nine commits on `claude/elastic-bhaskara`:

| Hash | Subject |
|---|---|
| `1a606a5` | proto: `slot_id=55` + `pipeline_depth=56` on `ForwardRequest`, `slot_id=27` on `ForwardResponse` |
| `3815a65` | `SlotState` dataclass + `RingSession.lock` (per-session mutex) + state constants |
| `5718f03` | Locked compound-op in coord-side PushResult handler + worker pool dispatch |
| `108ed4a` | `--pipeline-depth` CLI flag wired end-to-end (engine, chain, infer) |
| `e74a965` | Concurrency stress tests for `RingSession.lock` (8 tests, 32 workers, 64 tokens) |
| `c49c39d` | `PyTorchRuntime` executor `max_workers=N` plumbing + `ToyShardConfig.runtime_pipeline_depth` |
| `c44efc0` | MLX: drop mid-shard `mx.eval` fence (lazy graph through serialise boundary) |
| `3de5d79` | Fire-and-forget `_push_to_next_hop` + `_push_final_result` under depth ≥ 2 |
| `dbc15d9` | `slot_id` proto round-trip tests + Phase 2a regression guards |

### What 2a achieved

- Per-peer PyTorch executor accepts up to `--pipeline-depth N` concurrent
  forward passes on the same shard. MLX peers carry the lazy graph
  through to the serialise boundary, removing the mid-shard `mx.eval`
  fence.
- Coord-side `_coordinator_proxy_handler_loop` runs a
  `ThreadPoolExecutor(max_workers=max(2, pipeline_depth))` so multiple
  PushResults reconcile concurrently. The compound op (slot state
  transition + token append + EOS check + in-flight count + next-slot
  reservation) runs under `RingSession.lock` to prevent over-fire under
  contention.
- Direct-gRPC peer-to-peer and peer-to-coord sends become fire-and-forget
  via daemon threads under depth ≥ 2; libp2p relay paths were already
  fire-and-forget via `PROXY_METHOD_FIRE_FORGET`.
- Default `--pipeline-depth 1` preserves byte-identical pre-2a behaviour
  (validated by the regression guards in `dbc15d9`).

### What 2a deliberately did not do

- No drafting. No speculation. No lossless multi-token-per-RTT win.
- No KV cache changes.
- No proto field renumbering (additive only: tags 55-56-27).
- No Rust changes; the `openhydra_network` extension is unchanged.

2a is the prerequisite scaffolding for 2b. Without 2a, Phase 2b's
"draft block N+1 while verify N is in flight" overlap could not happen.

---

## Phase 2b — DFlash Block-Diffusion Speculative Decoding

**Plan file:** `/Users/sam/.claude/plans/dflash-block-diffusion-phase-2b.md`

Replaces single-token autoregressive decoding with block-diffusion
speculative decoding via DFlash (Chen et al., 2026 — `arxiv:2602.06036`).
A small draft model generates 16 candidate tokens in one parallel
forward; the full target model verifies all 16 in a single layer-sharded
ring trip; the coordinator runs `lm_head`, computes greedy argmax per
position, and accepts the longest matching prefix plus a bonus token.

Lossless under `temperature=0.0`. Acceptance rate ~86-91% on Qwen3.5
at 1024-4096 tokens (per published DFlash benchmarks).

### Two topologies

Both topologies live behind one CLI flag: `--draft-location {local,stage-0,off}`.

**Topology A — coordinator-side drafting (`--draft-location local`):**
coordinator hosts the draft model, packs 16 candidate token ids into
`ForwardRequest.prompt_token_ids` with `draft_block=True`. Stage-0
unpacks and runs its layer shard over all 16 positions in parallel;
ring continues to last peer; last peer pushes 16-position hidden states
back to coord; coord runs `verify_block` and re-injects.

**Topology B — stage-0 drafting (`--draft-location stage-0`):**
coordinator sends only the prefix; stage-0 hosts the draft model,
generates the 16-token block locally, and feeds straight into its own
layer shard. After verify, coord broadcasts a `VerifyResult`
SwarmCommand carrying `(accepted_len, bonus_token, kv_rollback_to)` so
stage-0 can start drafting block N+1 immediately. All other peers
receive `kv_rollback_to` inline on the next `ForwardRequest`.

### Seven sharpenings locked in design review

1. `VerifyResult` SwarmCommand for Topology B's return path.
2. `kv_rollback_to: uint32 = 58` on `ForwardRequest` — race-free
   piggybacked rollback, applied INLINE on each peer before the forward
   starts.
3. Per-layer-type rollback strategy (snapshot-restore for attention,
   tape-replay for GatedDeltaNet/Mamba — not per-backend, because
   Qwen3.5 is hybrid and a single shard owns mixed layer types).
4. `--layers START-END` manual override with startup union-coverage
   validation. Refuses to start on gap/overlap/partial-adoption rather
   than silently producing wrong output.
5. `--draft-block-size N` configurable, default 16, max 32.
6. Failover: draft model is a swarm-level resource. Stage-0 always
   preloads draft weights regardless of topology so a coord crash under
   Topology A can be recovered by stage-0 promotion. Spec lives in the
   `RegisterDraftModel` SwarmCommand registry.
7. Telemetry surface for Phase 3: `draft.inflight_p50_ms`,
   `draft.ram_mb`, `target.verify_block_p50_ms`,
   `ring.acceptance_rate_ema`, `peer.gpu_free_ram_mb`,
   `peer.target_layers_owned`.

### Default model rotation

Default target moves from `openhydra-qwen3.5-2b` to **Qwen3.5-4B**
because 2B has no published DFlash draft. 4B/9B/27B-4bit/35B-A3B-4bit
all have `z-lab/Qwen3.5-{N}B-DFlash` published on HuggingFace. 2B
default returns in Phase 4 once the z-lab training recipe ships.

### Composition with Phase 2a

`--pipeline-depth 2` plus DFlash: coord drafts block N+1 while block N
is verifying through the ring. Both speedups multiply.

---

## Phase 2b.1 — DDTree as Alternative Drafter

**Status:** Planned post-2b validation.

`humanrouter/ddtree-mlx` and `liranringel/ddtree` implement
tree-based speculative decoding — multi-branch draft exploration per
step, claimed ~10-15% faster than DFlash on code generation, ~1.5×
over autoregressive in general.

Phase 2b's `DFlashDrafter` interface (`draft(prefix) → list[int]`) is
deliberately algorithm-agnostic. Phase 2b.1 adds a parallel
`DDTreeDrafter` implementation behind the same interface, exposed via
`--draft-algorithm {dflash,ddtree}`. No architectural change.

Decision criterion: per-domain benchmark. Code-heavy prompts may
prefer DDTree; conversational prompts may prefer DFlash. Phase 3's
auto-negotiator can flip the algorithm per session if the telemetry
shows divergent acceptance.

---

## Phase 2c — `ForwardStream` Bidirectional gRPC

**Status:** Planned. Independent of 2b — can land in any order post-2a.

`peer/peer.proto` line 10 already declares
`rpc ForwardStream(stream ForwardRequest) returns (stream ForwardResponse)`,
but the Rust handler does not yet exist; today's traffic uses unary
`Forward` RPCs.

Phase 2c implements the Rust-side bidirectional stream handler.
Persistent per-session streams between adjacent peers cut the gRPC
handshake cost per token (~2-5 ms per hop on T4, ~1-3 ms on M-series),
enable continuous batching across distinct sessions in the future, and
reduce socket churn on long generations.

Expected TPS lift: +5-10% on top of Phase 2b. Smaller in absolute terms
than 2b's amortisation factor, but proportionally meaningful when
combined with 2b's 16-tokens-per-RTT factor.

---

## Phase 3 — Auto-Negotiator

**Status:** Planned. Pure telemetry consumer; no peer-side or coord-side
code changes if Phase 2b's telemetry surface is correct.

Reads the six metrics emitted by Phase 2b and emits two SwarmCommands:

- `SetDraftLocation { local | stage-0 }` — flips topology mid-session
  when, e.g., the coord's draft inflight exceeds the link RTT.
- `SetDraftBlockSize { N }` — tunes block size per session. Low
  acceptance → smaller blocks (less wasted verify); high acceptance →
  larger blocks up to 32.

Decision logic lives in `coordinator/auto_negotiator.py` (new). Hand-
coded thresholds first; ML-based policy is a non-goal for v1.

---

## Phase 4 — KV Cache Compression

**Status:** Planned. Independent of speculative decoding chain
(2b/2b.1/2c). Can land in parallel.

Adopts **TurboQuant** (`RecursiveIntell/turbo-quant`, Rust impl of
TurboQuant + PolarQuant + QJL, ICLR 2026). Published claims: 73-99%
KV cache VRAM savings at zero accuracy loss.

### Why this matters for OpenHydra specifically

Each peer holds per-session KV caches that grow O(context_length).
On 8GB Macs and 15GB T4s, KV is what caps the context window we can
serve. Even a 4× compression factor changes which models we can run
with which context lengths on which peer hardware — a category change,
not a percentage improvement.

### Integration shape

- New module `peer/kv_quant.py` wraps the chosen scheme behind an ABC
  with `quantize(kv) → compressed`, `dequantize(compressed) → kv`,
  `truncate(compressed, n)`, `replay(compressed, accepted)`.
- Routes through the existing `openhydra_network` Rust extension —
  TurboQuant is already Rust-native, binding is mechanical.
- New flag `--kv-cache-quant {none,polarquant,turboquant}`,
  `--kv-cache-quant-bits {2,4,8}`.
- Composes with Phase 2b's per-layer-type rollback: quantized KV still
  needs rollback, the strategies (truncate, tape-replay) operate on the
  quantized representation directly.

### Default policy

Default `none` until Phase 4 lands and ships green; then `turboquant`
at 4-bit becomes the default for all 8GB Mac and 15GB T4 peers (auto-
detected at peer startup based on available VRAM).

---

## Phase 4.1 — Hidden-State Compression on Cross-ISP Hops

**Status:** Planned. Companion to Phase 4 but distinct.

The cross-ISP libp2p relay path is dominated by payload size, not RTT:
relayed bandwidth caps at ~10-30 Mbps in current Linode bootstrap
deployments. A 4 KB hidden state at fp16 is ~16 KB at fp32; an int8
compression is 4× smaller; an int4 compression is 8× smaller.

### Scope

- LAN hops keep fp16 — payload is not the bottleneck on LAN, accuracy
  cost is not worth it.
- Cross-ISP hops (detected via `peer/lan_routing.py::is_reachable_lan`
  returning False) compress hidden state to int8 (default) or int4
  (opt-in via `--cross-isp-quant-bits`).
- Compression happens at the serialise boundary in
  `_hidden_to_packed_bytes` (MLX) and the equivalent PyTorch path; the
  Rust encoder gains a `quant_bits: u8` parameter.
- Decompression at receive boundary in `_activation_to_hidden` mirrors.

### Expected impact

Relay-bound cross-ISP TPS scales roughly with payload size when network
is the bottleneck. 4× compression → 2-4× TPS on the
cross-ISP heterogeneous Mac+GPU benchmark (currently ~0.93 TPS, target
~3 TPS post-2b, ~6-12 TPS post-4.1 depending on link).

### Composition

- Phase 2b ships first; verify pass payload becomes 16× the per-token
  payload, so 4.1's compression matters proportionally more under 2b.
- Phase 4 (KV compression) is orthogonal: KV lives on peers, hidden
  state lives on the wire. Both can be in flight simultaneously.

---

## Cross-cutting concerns

### Testing posture

- Every phase ships byte-equivalence regression guards under
  `temperature=0.0, seed=42` against the previous phase's baseline.
- Phase 2a's `tests/test_ring_session_concurrency.py` (8 tests, up to
  16 concurrent workers stressing 64-slot reservation) is the template
  for concurrency stress tests in 2b/2b.1/3.
- Live benchmarks A through I (defined in the per-phase plan files)
  cover regression, all-LAN, cross-ISP, asymmetric sharding, and
  failover — required before phase merge.
- Suite baseline grows monotonically. Phase 2a was 1457; Phase 2b adds
  ~9 tests; Phase 4 adds ~12. Any phase that drops the suite count
  below the previous phase's baseline is rejected.

### Telemetry surface

Established in Phase 2b §9. The auto-negotiator (Phase 3) is a pure
consumer. Phase 4 adds three more metrics
(`peer.kv_quant_ratio`, `peer.kv_decompress_p50_us`,
`session.context_tokens_compressed`). Phase 4.1 adds two
(`hop.wire_compress_ratio`, `hop.wire_compress_p50_us`).

### Rollback gating discipline

Every new feature is gated behind a CLI flag whose default is
"off / 1 / no-op." Defaults preserve byte-identical behaviour against
the previous phase. The proto fields are additive (tag 55, 56, 27 in
2a; 58, 59 in 2b; future tags reserved 60-69 for 4 and 4.1). No proto
field is renumbered, ever.

### Wire format strategy — explicit non-adoptions

We have evaluated and rejected the following alternatives. This
decision is documented here so it is not relitigated:

| Alternative | Why we are not adopting it |
|---|---|
| Cap'n Proto / FlatBuffers | Saves ~0.3-0.8 ms per hop on marshal cost. Our hop latency is 5-50 ms LAN / 80-300 ms cross-ISP relay. Marshaling is ≤5% of cost. Migration cost is total. |
| Aeron | Sub-µs LAN messaging. Cannot replace libp2p (no NAT traversal). On LAN we already have direct gRPC fire-and-forget; gain is in microseconds while we are losing milliseconds to physics. |
| ZeroMQ | Same NAT problem. No type system. Solves no problem we have. |
| Native vLLM / SGLang as peer runtime | Massive refactor. Their value is in single-device throughput; OpenHydra's value is in distributed orchestration over libp2p. Different problem domains. |
| Megakernel compilation (Mirage et al.) | Per-shard compute is not yet the bottleneck post-2b. Revisit in Phase 6+ if it becomes one. |

Where the real wire-format wins live: zero-copy receive of the hidden
state (already shipped via DLPack in the Rust extension) and payload
compression on bandwidth-bound hops (Phase 4.1).

---

## Explicitly out of scope through Phase 4.1

Documented here so future contributors do not relitigate:

- vLLM, SGLang, vllm-metal as peer runtimes (see wire format strategy).
- Single-Mac competing products (`omlx`, `mlxstudio`, `vmlx`,
  `SwiftLM`, `lucebox-hub`, `jangq`) — different problem domain,
  different deployment model.
- Megakernel compilation (Mirage, MegaQwen, qwen_megakernel) — deferred
  to Phase 6+.
- Tree speculation beyond the DDTree adoption in 2b.1 — research
  direction not yet justified.
- Continuous batching across distinct sessions on the same peer — only
  meaningful with multi-tenant deployments which are not on the
  near-term roadmap.
- Custom DFlash draft training for `openhydra-qwen3.5-2b` — waits on
  z-lab releasing the public training recipe.
- Cap'n Proto, FlatBuffers, Aeron, ZeroMQ migrations.

---

## Glossary

- **Path A** — Client-terminated pipeline. Last peer returns hidden
  state to coord; coord runs `lm_head` and samples. Shipped in six
  phases ending at commit `f078c57`.
- **Topology A / B** — Phase 2b drafting locations. A = coord drafts;
  B = stage-0 drafts.
- **slot_id** — Per-ring-session identifier for a single in-flight
  token (Phase 2a) or block (Phase 2b). Echoed by last peer on
  PushResult so coord can match the response back to its
  `RingSession.slots[slot_id]` SlotState entry under
  `pipeline_depth ≥ 2`.
- **pipeline_depth** — Maximum in-flight tokens (or blocks) per ring
  session. Default 1 = serial. 2+ = async pipeline.
- **RingSession** — Per-(request_id) state on the coord. Fields:
  generated tokens, slots dict, next_slot_id, lock, plus 2b additions
  for pending block.
- **kv_rollback_to** — `ForwardRequest` field 58 (Phase 2b). Absolute
  sequence position the receiver should truncate the per-session KV
  cache to BEFORE running the forward. Race-free.
- **VerifyResult** — SwarmCommand body added in Phase 2b carrying
  `(session_id, accepted_len, bonus_token, kv_rollback_to)` from coord
  to stage-0 under Topology B so stage-0 can start drafting block N+1.
- **HeadSampler** — Coord-side singleton holding `lm_head` weights.
  Borrows from a co-located peer when coord shares a process; Phase 2b
  extends with `verify_block`.

---

## Document governance

- This file is `v1` and is **immutable**. Phase decisions can be
  amended by appending an ADR-style "Amendment N" section at the
  bottom; structural reorganisation requires a fresh
  `ARCHITECTURE_ROADMAP_v2.md` that explicitly cites and supersedes
  this one.
- The per-phase plan files in `/Users/sam/.claude/plans/` remain the
  implementation playbooks. This roadmap is the canonical reference
  for "why" and "in what order."
- Commit hashes referenced in this document are pinned to the
  `claude/elastic-bhaskara` branch as of `dbc15d9`. Future rebases
  must preserve these hashes or update this document in lock-step.
- Phase status updates (Planned → In Progress → Done) happen via
  amendment, not in-place edit.
