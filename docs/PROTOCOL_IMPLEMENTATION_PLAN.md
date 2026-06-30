# OpenHydra Protocol — Implementation Plan

> Companion to [`docs/protocol.md`](./protocol.md). This is the *how* to that
> document's *what*. It maps the spec onto the code that exists today, fixes the
> implementation language, and breaks the work into ordered, verifiable milestones.

**Status:** v4 — **pivot shipped** (snapshot below) · **Branch:** `main` (worked in the
`openhydra-netfix` worktree) · **Original planning pass:** 2026-06-14 · **Status refresh:** 2026-06-16

---

## ⚑ Direction change (2026-06-14): pure protocol, no inference engine

**Decision:** OpenHydra is a **pure, engine-agnostic distributed-inference protocol in
Rust** — it routes, verifies, and settles credit for inference served by *external*
engines (Ollama, vLLM, LM Studio, llama.cpp). **It does not run models itself.**

The earlier hybrid framing — keep the Python MLX/PyTorch **sharded** path as a
subordinate "fallback tier" — is **superseded.** The sharded engine is being **removed,
not kept.** Concretely, this pivot supersedes anything below that implies a retained
sharded tier (notably the "Stays Python" list in §1.3, the sharded island in the §2
diagram, the §2.1 "sharded provider needs Python" row, and **Phase 4 — sharded fallback**,
which is deleted).

**Why:** you can't out-engine vLLM/llama.cpp and don't need to; the moat is the
trust/credit/discovery layer (already Rust). Dropping the engine collapses the
Python↔Rust straddle into one static Rust binary, makes one-click distribution trivial,
and is the honest endpoint of the BitTorrent-for-AI thesis (BitTorrent ships no codec).
**Cost accepted:** no cross-node sharding of models too big for one provider (external
engines serve only what fits on one box); verification softens to redundant-exec +
reputation + engine logprobs (no activations ⇒ TOPLOC `activation_hash` becomes
vestigial). The product is *"a trust/credit market over the engines people already run."*

**Migration discipline (additive-first, deletion-last — `main` stays working):** build
the Rust path (engine adapter → libp2p stream → Rust HTTP/SSE gateway → Rust
verification) and prove it end-to-end *before* deleting the Python engine + coordinator.

---

## ✅ Current status (2026-06-16): the pivot is shipped

The pure-protocol pivot is **complete and live on `main`.** OpenHydra is now a
single-language Rust project — the Python coordinator/engine tree and the PyO3
bridge are **deleted**, and the Rust `openhydra-agent` is the sole host. **The
detailed sections below are the original plan; the PyO3-transition framing in
§0–§2 and §5 is now *historical* — that bridge is gone.** This banner is the
source of truth for current state.

**Done:** Phase 1 (M1.1–M1.3) · M2.1 receipts · M2.2 verification *primitives*
(`protocol::verify`) · M2.3 redb ledger (`protocol::store`) · M3.1 Ollama adapter ·
M3.2 adapters (vLLM / LM Studio / llama.cpp / OpenAI-compatible) · **R1** HTTP/SSE
gateway (`agent/src/gateway.rs` — non-streaming + streaming, `GET /v1/models`,
optional API-key auth) · **R2** cutover (Python tree deleted, single static binary) ·
**R2b-1** (pyo3/FFI removed → `network` is a pure rlib) · all of R-DHT-1…11
(DHT + NAT-traversal hardening). The §5 near-term goal — *"an Ollama-friendly swarm
serving verified inference end-to-end with no Python in the path"* — is **achieved.**

**Remaining:**

| Area | Item | Notes |
|---|---|---|
| Incentive layer | **M3.3 — scarcity pricing on** | the `scarcity_multiplier` switch is **not yet wired** (no pricing code in `protocol/`) |
| Incentive layer | **M2.4 — cold-start correctness** | priority-not-access under an empty ledger |
| Incentive layer | Verification *depth* | redundant-exec / sampling *enforcement* in the router, beyond the built hash + reputation primitives |
| Incentive layer | Priority-not-access *enforcement* | wire the ledger ratio → rate-cap into routing |
| Reach | **R3 — packaging** | Homebrew / signed single-binary install |
| Reach | **Desktop D1–D3** | the Tauri one-click app — now unblocked by R1 |
| Adapters | Apple Foundation Models | optional macOS Swift-FFI adapter (own crate) |
| Connectivity | **Connection reversal (Tier 2)** | NAT'd provider dials a reachable consumer — escapes relay on symmetric CGNAT (see [PEER_CONNECTIVITY.md](PEER_CONNECTIVITY.md)) |
| Cleanup | R2b-2 → R2b-3 | excise the legacy IPC/sharding Rust modules woven into `event_loop.rs`, then drop the `proto` mod once `dispatcher` is gone |

**Critical-path read:** the original plan's near-term goal is met. What remains splits
into (a) **finishing the incentive layer** (M2.4 + M3.3 + enforcement — so give-to-get
actually bites) and (b) **reach** (R3 + Desktop — adoption). Connection reversal is
orthogonal and arguably the highest-leverage item for real-world cross-NAT usability.

---

## 0. Where we actually are (grounding)

> **Historical (2026-06-14).** This table described the pre-pivot codebase. It is kept
> for context; see the status banner above for current state. Statuses below are
> annotated `→ now:` where they changed.

The spec's roadmap reads as four greenfield phases. The codebase says otherwise —
**the protocol substrate is largely built, and the hardest part of it is already Rust.**
An honest plan starts from that.

| Spec capability | Today | Where |
|---|---|---|
| libp2p transport (QUIC+TCP), Kademlia DHT, Relay v2, DCUtR, AutoNAT, mDNS, gossipsub | ✅ Built (**Rust**) | `network/src/{dht,relay,nat,mdns,swarm,behaviour}.rs` |
| DHT model records `/openhydra/model/{model_id}/{peer_id}` (CBOR, Ed25519-signed) | ✅ Built (**Rust**) | `network/src/{types,dht}.rs` |
| OHV2 binary wire format (CBOR header + raw activation), zero-copy DLPack | ✅ Built (**Rust**) | `network/src/{forward_msg,ipc_codec,dlpack}.rs` |
| Rate-limit / relay leech control | ✅ Built (**Rust**) | `network/src/bootstrap_bin.rs` |
| Whole-model single-peer routing | ⚠️ Partial (Python) | `coordinator/chain.py::run_push`, `peer_selector.py`, `path_finder.py` |
| ~~Sharded fallback (layer-parallel, ring mode)~~ — **being removed** (pivot) | Built (Python/MLX), to be deleted at cutover | `coordinator/chain.py::run_push_ring`, `peer/mlx_runtime.py` |
| OpenAI + Ollama HTTP API w/ SSE streaming | ✅ Built (Python) | `coordinator/api_server.py` |
| Peer ranking (RTT/load/reputation) | ✅ Built (Python) | `coordinator/peer_selector.py` |
| TOPLOC proof-of-inference (activation hash) | ⚠️ Basic (Python) | `verification/toploc.py` |
| Canonical model-id `{family}/{params}/{quant}/{template_hash}` | ❌ Absent | catalog is static (`models.catalog.json`) |
| Verification *policy* (sampling, redundant exec, reputation feedback) | ⚠️ Stubs (Python) | `verification/{redundant,auditor,reputation}.py` |
| Co-signed receipts, credit ledger, pricing | ❌ Absent | `economy/` removed in `7316de0` |
| Engine adapters (Ollama/vLLM/…) | ❌ Absent | only native MLX/PyTorch shards today |
| Rust agent daemon (HTTP host) | ❌ Absent | all HTTP is Python |

**Reading:** the I/O-bound substrate is done and Rust. The *product* layers — the
agent as a frictionless gateway, the trust+credit economy, multi-engine reach — are
the real work, and per the decision below they are **written in Rust**.

---

## 1. Decision (made): Rust-first for all new protocol code

Spec §3 argues for a **single statically-linked Rust daemon**, because frictionless
provider onboarding — `brew install`, drop in a binary, flip a switch — is *the*
determinant of supply. **That decision is now made and binding for this plan:**

> **All new protocol code is written in Rust. The only standing exception is code
> where Rust genuinely doesn't make sense — see the language boundary in §1.1.**

This supersedes the earlier "prototype in Python, port later" sequencing. We do not
build the credit/verification/model-id logic in Python and re-implement it; we write
it in Rust once.

**This is feasible to start *now*, without first finishing the host rewrite.** The
existing `openhydra-network` crate already builds two ways — a pure-Rust `rlib` *and*
a PyO3 `cdylib` (pyo3 is an optional feature in `network/Cargo.toml`). So new Rust
protocol modules can be:

1. **Consumed as Rust** by the eventual `openhydra-agent` daemon (the end state), and
2. **Exposed via PyO3** to the *current* Python coordinator during the transition,

from the same source. The Python host calls into the new Rust logic immediately; when
the host itself inverts to Rust (R-track), the PyO3 boundary simply flips direction —
no protocol logic is rewritten. Rust-first therefore *de-risks* the inversion: by the
time we invert the host, every protocol rule is already Rust and test-covered.

**Honest tradeoff.** Rust iterates slower than Python on rules that are still being
designed (credit pricing, verification sampling). We accept that, and mitigate it
**not** by prototyping in Python but by **specifying first**: each rule gets a
language-neutral spec + golden **test vectors** (§6) *before* the Rust implementation,
so Rust targets a settled contract rather than a moving target. The narrow exception
list (§1.1) is the only place this rule bends.

### 1.1 The language boundary — what's Rust, what isn't

**Rust (all new protocol code):** HTTP/SSE API server · provider agent + engine
adapters · canonical model-id + template hash · router (resolve/rank/route) ·
co-signed receipts · credit ledger + scarcity pricing + priority gating · verification
policy (sampling, redundant-exec orchestration, reputation, TOPLOC hashing) · per-hop
encryption · private-swarm scoping · anti-abuse (whitewash throttle, PoW-on-join,
counterparty-diversity). Most of these are *additions to a stack whose foundation —
libp2p, DHT, relay, wire format, signing — is already Rust.*

**Not Rust (where it genuinely doesn't make sense):**

| Stays non-Rust | Why |
|---|---|
| **The inference engine itself** — *external*, not ours: Ollama / vLLM / LM Studio / llama.cpp | We don't run models; the provider's agent proxies to whatever engine that operator already runs, over its HTTP API. The engine is a *separate process* in someone else's language — never in our binary. (Pure-protocol pivot: the old in-repo MLX/PyTorch **sharded** engine is **removed**, not "kept as a permanent island" — see the pivot banner.) |
| **Transitional Python glue** — thin wiring that lets the current Python coordinator call the new Rust modules via PyO3 | Temporary by construction. Shrinks to zero when the host inverts (R-track) and the Python engine is deleted. Not new *protocol* logic — just adapter code over the boundary that's moving anyway. |

The existing Python is **rewritten in Rust where it benefits, and otherwise deleted** —
the protocol-layer code (front door, crypto, routing, protocol logic) is ported to Rust;
the inference-orchestration layer (sharded scheduling, KV/cache, batching) is **removed
with the sharded engine**, not retained. §1.3 lists the concrete targets.

### 1.2 Where the Rust code lives — crate structure

Today there is one crate, `openhydra-network` (no workspace). Grow it into a **Cargo
workspace** so the daemon, the protocol logic, and the libp2p core are separable and
independently testable:

```
Cargo.toml            # [workspace] — add at repo root; keeps network/ in place
network/              # openhydra-network  — existing libp2p core (unchanged location)
protocol/             # openhydra-protocol — NEW: model-id, receipts, ledger, pricing,
                      #   verification policy. Pure logic, no libp2p/HTTP deps, 100%
                      #   unit-tested against golden vectors. The portable spec, in Rust.
agent/                # openhydra-agent    — NEW: the daemon binary. HTTP/SSE server,
                      #   router, engine adapters. Depends on -network + -protocol.
                      #   This is the single static binary (`brew install openhydra`).
pyext/                # openhydra-pyext    — PyO3 cdylib re-exporting -protocol (+ the
                      #   existing -network bindings) to the transitional Python host.
desktop/src-tauri/    # openhydra-desktop  — Tauri backend; depends on -agent/-protocol/
                      #   -network. The one-click GUI app (WS-DIST); existing scaffold.
```

**Incremental path:** start M1.1 by adding a `protocol` module *inside* the existing
`network` crate (zero workspace churn, fastest to first green test), and promote it to
the standalone `protocol/` crate when the `agent/` binary work begins (R-track). New
crate deps as needed: `sha2` (hashing), `ciborium`/existing CBOR (records), the
existing `ed25519-dalek` (receipts), `redb` (ledger store — pure-Rust ACID; or `rusqlite`), `axum`+`hyper`+`tower`
(HTTP/SSE), optional `minijinja` (chat-template rendering).

**Engine adapters live in `agent`, one crate, not one-crate-each.** Keep all adapters
in the `agent` crate as **one module per adapter** (`agent::adapters::{ollama, vllm, …}`)
behind the stable `EngineAdapter` trait, each gated by its own Cargo **feature**
(`features = ["ollama", "vllm", "llamacpp", "apple-fm", …]`, `default = ["ollama"]`) so a
build can ship a subset — e.g. a Linux server build drops `apple-fm` entirely. They share
the same deps today (`reqwest` + `serde_json`; the OpenAI-compatible engines are
near-identical thin shims), so separate crates would buy only churn. **Promote a single
adapter to its own crate only when a dependency forces it** — i.e. it pulls a heavy or
**platform-specific** dep that would otherwise burden every build. Apple Foundation
Models (objc/Swift FFI, macOS-only) is the likely first — and possibly only — such split.

### 1.3 Existing Python — concrete rewrite targets

A rewrite earns its keep when a component is on the whole-model/daemon hot path,
network-facing (untrusted input → memory safety), protocol-defining (should be the
canonical reference), concurrency/IO-bound (GIL relief), or required in the static
binary. It does **not** when coupled to the Python tensor stack, rarely run, or pure
glue that disappears at inversion anyway. The targets:

**High benefit — rewrite to Rust:**

| Existing (Python) | → Rust | Milestone | Why |
|---|---|---|---|
| `coordinator/api_server.py` (HTTP/SSE, OpenAI/Ollama) | `agent` HTTP server (`axum`/`hyper`) | R1 | Front door; concurrent SSE strangled by `ThreadingHTTPServer`+GIL; must be in the static binary; network-facing |
| `peer/crypto.py` (Ed25519 / X25519 / AES-256-GCM) | transport/`protocol` crypto (`ed25519-dalek`, `aes-gcm`) | R1 / transport | Security-critical, processes stranger bytes; audited crates; Ed25519 identity already part-Rust |
| `peer_selector.py` + `discovery_service.py` + `path_finder.py` (ranking & resolution) | `protocol::router` | M1.3 | Per-request routing; removes Python↔Rust hops around the Rust DHT; canonical decision logic |
| `peer/model_catalog.py` / model-id logic | `protocol::model_id` | M1.1 | Protocol-defining; should be the Rust reference others implement against |

**Medium benefit — port when the owning milestone lands:**

| Existing (Python) | → Rust | Milestone | Why |
|---|---|---|---|
| `verification/toploc.py` (activation hashing) | `protocol::verify` | M2.2 | Activations already byte-level in Rust (OHV2) → avoids a copy in the verifier path; modest where data is a Python tensor (shard path) |
| `peer/dht_announce.py` (record construction) | `network` announce path | M1.2 | The record is already a Rust `PeerRecord`; fold construction in to drop the glue |

**To be REMOVED** (the pure-protocol pivot — these were the sharded MLX/PyTorch engine,
which we no longer run): `mlx_runtime.py`, `model_shard.py`, `chain.py::run_push_ring` +
the sharded ring/per-token coordinator loop, `pipeline_service.py`, `speculative.py`,
`kv_affinity_service.py`, `request_batcher.py`, `reshard_executor.py`,
`swarm_negotiator.py`, `autonomous_rebalancer.py`, `p2p_model_cache.py`, plus
`coordinator/inference_service.py`'s tensor orchestration and `verification/toploc.py`
(no activations without the shard path). **Deletion is the final migration step** (after
the Rust adapter + gateway serve a request end-to-end), not a precondition — until then
the Python stack remains the live implementation.

**Cleanup — finish removing legacy gRPC (tracked, do later).** gRPC was dropped as the
peer *tensor transport* (commit `bbc65ac` → libp2p), but a gRPC footprint remains and
should be retired as part of the lean-down (it is **not** fully vestigial — a first pass
already removed the genuinely-dead paths in `stream_pool.py` / `relay.py` / `server.py`,
tests green):

- `relay/relay_service.py` — a **standalone legacy gRPC relay server**, already superseded
  by the Rust Circuit Relay v2 (the deploy script only `pkill`s it). Decommission/delete it.
- Live `import grpc` users to migrate off gRPC: `coordinator/transport.py` (channels),
  `peer/tls.py` (server creds), `coordinator/chain.py` (`grpc.RpcError` in the failover
  except-clause), `coordinator/path_finder.py`.
- Only after those land can `grpcio`/`grpcio-tools` and `peer/peer_pb2_grpc.py` be dropped.
  Keep `peer/peer_pb2.py` message types + the `protobuf` dep (still used as in-memory
  containers via OHV2). This is a real migration, not a delete — scope it as its own task.

---

## 2. Target architecture

```
                    ┌─────────────────────────────────────────────┐
                    │     openhydra-agent  (single Rust binary)    │
   consumer apps    │                                              │
  (Open WebUI, ───▶ │  HTTP/SSE server  ── OpenAI/Ollama compat    │   ← Rust
   Continue,        │        │                                     │
   OpenAI SDK)      │        ▼                                     │
                    │   Router  ── resolve · rank · route · verify │   ← Rust
                    │        │            │                        │
                    │        ▼            ▼                        │
                    │  openhydra-     openhydra-protocol           │   ← Rust
                    │  network        (model-id · receipts ·       │
                    │  (DHT/relay/    ledger · pricing ·           │
                    │   dcutr/gossip) verification policy)         │
                    │        │                                     │
                    │   ┌────┴───────────┐                         │
                    │   │ engine adapter │  ── BYO-engine, over HTTP│   ← Rust (thin)
                    │   └────┬───────────┘                         │
                    └────────┼─────────────────────────────────────┘
                             ▼
                    local inference engine (separate process)
                  Ollama / vLLM / LM Studio / llama.cpp  — NOT ours
```

- **100% Rust, and it never runs a model.** A provider's agent proxies the inference to
  whatever engine that operator already runs locally (over the engine's HTTP API) and
  streams tokens back over libp2p; the consumer's agent serves them via SSE. OpenHydra's
  job is discover → route → stream → verify → settle → credit.
- **There is no sharded path.** Each provider serves only models that fit on its own
  hardware (whatever its engine can run). Cross-node sharding of oversized models is out
  of scope (see the pivot banner's accepted cost).
- **Verification** without activations = sampled **redundant execution** (run on ≥2
  providers, compare) + **reputation** + engine **logprobs** where exposed. TOPLOC
  activation-hashing no longer applies.
- **Transition state:** until `openhydra-agent` serves end-to-end, the *current Python
  coordinator* remains the live host and calls the same Rust `protocol`/`network` crates
  via `openhydra-pyext`. Same Rust source; the Python engine is deleted at cutover.

### 2.1 Distribution & the single-binary boundary

The end-state artifact is a **single static Rust binary** — `brew install openhydra`,
drop-in daemon, no Python env, no wheel build (spec §3; plan R3). That is the whole
distribution thesis: frictionless onboarding is what grows supply. The binary holds
the *entire protocol agent* — discovery, routing, trust/verification, credit/receipts,
the HTTP/SSE API, and the engine adapters.

Exactly one thing is deliberately **outside** the binary — and it's the whole point, not
a packaging shortfall: **the local inference engine** (Ollama, vLLM, LM Studio, …) is
always a *separate process* the agent reaches over HTTP. That is the BYO-engine design —
the agent is a gateway *in front of* whatever you already run. With the sharded tier
gone, **no participant needs Python or an ML runtime from us.**

What each participant actually installs:

| Node role | What it needs | Python? |
|---|---|---|
| Consumer / router | The binary | No |
| Provider | The binary + the engine it already runs (separate process) | No |

So the single-binary story now holds for **everyone** — there is no node role that
requires a Python env. The R2 exit test simplifies to: *"`openhydra` runs as a single
static binary with no Python; a provider serves by proxying to its local engine over
HTTP."*

---

## 3. Workstreams

Nine parallelizable streams; each milestone tags its stream(s).

- **WS-ID** Canonical model identity & equivalence (spec §4) — *Rust*
- **WS-CAP** Capability records & routing quality (spec §3, §5) — *Rust*
- **WS-RCPT** Co-signed receipts & ledger plumbing (spec §6) — *Rust*
- **WS-CREDIT** Credit accounting, priority-not-access, scarcity pricing (spec §6, §10) — *Rust*
- **WS-VERIFY** Verification policy: sampling, redundant exec, reputation (spec §7) — *Rust*
- **WS-ENGINE** Engine adapters (Ollama → vLLM/LM Studio/llama.cpp/Exo/AFM) (spec §3) — *Rust*
- **WS-AGENT** The Rust daemon: HTTP server, host inversion, packaging (spec §3) — *Rust*
- **WS-SWARM** Private swarms, bootstrapping, anti-abuse hardening (spec §8–§10) — *Rust*
- **WS-DIST** Desktop distribution: Tauri app over the shared crates, per-OS one-click installers, code-signing/notarization, tray/autostart/auto-update (spec §3) — *Rust + web UI*

---

## 4. Milestones

Ordered for value-first delivery. Each has an **exit test** — the concrete check that
it's done. Estimates are rough engineering-weeks for one focused engineer (Rust, so
slightly higher than a Python equivalent); streams parallelize. Unless noted, "tests"
means `cargo test` in the owning crate, plus golden vectors (§6).

### Phase 1 — Protocol core hardened (whole-model routing, end to end)

**M1.1 — Canonical model id** · WS-ID · Rust · ~2w · **✅ DONE** (`protocol::model_id`)
- `model_id` module — **landed inside the existing `network` crate** (`network/src/model_id.rs`,
  M1.1 core: 20 tests green, clippy/fmt clean). Starting in-crate keeps the workspace
  refactor *iterative* — extract to a `protocol` crate later, not up front. `CanonicalModelId`
  (`{family}/{params}/{quant}/{template_hash}`), `chat_template_hash` (`sha2`), an
  `hf_model_id` parser (explicit catalog fields preferred — see follow-up #1 — heuristic
  fallback), quant normalization, and `is_compatible` with wildcard request ids.
- The runtime quant (what the provider *loaded*) is the canonical quant — not the
  catalog's `recommended_quantization`. Template hash comes from the engine's live
  chat template (Ollama `/api/show`, or the loaded tokenizer in the shard path).
- Expose via PyO3 — **done**: surfaced through the existing `openhydra_network` extension
  (`canonical_id_from_hf` / `is_compatible` / `chat_template_hash` / `parse_hf_model_name`),
  rebuilt + installed + verified from Python.
- Wiring — **done**: resolved at the load site (`peer/canonical_id.py`, called in
  `PeerService.__init__` — `dht_announce` stays a dumb carrier), carried on
  `Announcement.canonical_model_id`, parsed into `PeerEndpoint`, and refused on by
  `discovery_service` via `filter_compatible_peers` (Rust `is_compatible`). Integration
  test `tests/test_canonical_model_id_wiring.py` (8) proves load→announce→parse→refuse;
  71 affected tests green, no new regressions.
- **Carries into M1.2:** the field travels the *Python* announce record today; getting it
  onto the *libp2p DHT* `PeerRecord` (Rust) is M1.2's capability-records work.
- **Exit test:** Rust unit tests over all 16 catalog entries (parser) + the
  equivalence contract (same weights+template → same id; different template/quant →
  incompatible; router refuses incompatible id). A 2-node integration test via the
  PyO3 path.
- **Deferred follow-ups** (from the M1.1 code review — handle before M1.1 is "done",
  none are blockers):
  1. **Catalog override reader.** The explicit-`family`/`params` override path exists at
     the API (`canonical_model_id(...)`) but no function reads `models.catalog.json` yet;
     add the catalog reader (the Python `canonical_id_for_catalog_model` equivalent) so
     curated fields can override the heuristic. The module doc already implies this.
  2. **Heuristic fragility.** `parse_hf_model_name` is tested only against today's 16
     names; novel conventions mis-parse to `"unknown"`. Confirm "heuristic + explicit
     override" as the governance stance; grow the golden vectors as the catalog grows.
  3. **`generation_params` in the template hash.** Dropped for simplicity; fold back if a
     forced system prompt / default sampling must affect equivalence.
  4. **Conservative quant folding.** Only `q4→int4` / `q8→int8` fold; `q5/q6/q3/q2` pass
     through un-normalized. Decide whether to fold more GGUF spellings.
  5. **Provider-id hash validation.** `CanonicalModelId::parse` accepts a non-hex / wrong-
     length `template_hash`; add hex + length validation for concrete (non-wildcard)
     provider ids (harmless for matching today, but a bogus provider hash isn't rejected).

**M1.2 — Capability records on the wire** · WS-CAP · Rust · ~1w · **✅ DONE**
- Extend the DHT `PeerRecord` (already Rust, `network/src/types.rs`) with the spec §4
  set used by ranking: `context_length`, `max_output_tokens`, `throughput_tok_s`,
  `queue_depth`, `backend`, `hardware_class`, `region`, `requires_relay`, `reputation`,
  plus the M1.1 `canonical_model_id`. Wire live `queue_depth`/`throughput`.
- **Exit test:** records round-trip through the DHT with the new fields; a seeded
  multi-peer test proves ranking orders by live health/RTT/throughput/queue.

**M1.3 — Router: resolve → rank → route (in Rust)** · WS-CAP · Rust · ~2w · **✅ DONE** (`protocol::router`)
- Port ranking + resolution (`coordinator/peer_selector.py`, `discovery_service.py`,
  `path_finder.py`) into `protocol::router` — pure scoring (liveness, RTT, throughput,
  queue, reputation, contribution ratio). Drive
  routing over the existing Rust libp2p `proxy_forward`/`open_tunnel`. Graceful
  degradation to the nearest smaller same-family model (spec §5).
- **Exit test:** end-to-end serve through a single remote provider over (a) direct
  LAN, (b) relay; lowest-RTT healthy provider wins in a 3-provider test; graceful
  degradation fires when a model has no live providers.

### Phase 2 — Trust & credit (Rust; the economic spine)

Spec center of gravity, biggest greenfield. Each rule: **vectors first, then Rust.**

**M2.1 — Co-signed receipts** · WS-RCPT · Rust · ~2w · **✅ DONE** (`protocol::receipts`)
- `protocol::receipts`: `sign_provider(sign_consumer(provider, consumer, model_id,
  tokens, nonce, ts))` with the existing `ed25519-dalek` keys; nonce store
  (double-count), monotonic per-peer counters (rollback). Plumb into the lifecycle
  "settle" step.
- **Exit test:** a completed request yields a valid co-signed receipt; tampered token
  count / replayed nonce rejected; property tests over signing; golden vectors fix the
  byte layout.

**M2.2 — Verification policy** · WS-VERIFY · Rust · ~2.5w · **⚠️ PARTIAL** (`protocol::verify` — hash + reputation primitives built; sampled redundant-exec *enforcement* in the router pending)
- Port TOPLOC hashing into `protocol::verify`; build the *policy*: sampled
  proof-of-inference (rate tuned by reputation), redundant execution (run a sampled
  fraction on ≥2 providers, compare), reputation feedback that downranks repeat
  failures into the router (M1.3).
- **Exit test:** a deliberately-bad provider is caught by sampled verification and
  downranked out of routing within N requests; sample rate scales with reputation.
- **⚑ Post-pivot reframing (2026-06-27).** TOPLOC activation-hashing was designed for the
  *sharded* path — it hashes hidden activations. The pivot **removed sharding**; the
  BYO-engine path proxies black-box engines that return **text, not activations**, so
  **activation-TOPLOC is inapplicable to the current architecture.** The verification
  ladder for the text world:
  1. **Reputation-from-delivery — M2.2(a), the core (fully applicable today).** Wire
     `verify::ReputationTracker` into the live consumer path + the router's
     `PeerScoreInput.reputation`: a provider that errors / refuses to co-sign / disappears
     is downranked out of routing. This is the closed loop the exit test really needs.
  2. **Redundant-execution with deterministic decode — M2.2(b), the robust output check.**
     Re-run a sampled fraction at `temperature=0` on ≥2 providers of the same canonical
     `model_id` and compare via `agrees()`; on disagreement escalate to a 3rd for
     majority, then feed `Failed`. *Caveat: temp=0 is not bit-identical across
     hardware/engine builds → needs a near-match tolerance.*
     - **inc1 ✅ (pure primitive, `protocol::verify`):** `agrees()` is now real — a
       near-match check (`common_prefix_ratio ≥ AGREEMENT_THRESHOLD`, tolerant of benign
       late cross-HW divergence, fails a freeloader's canned/empty/wrong-model output) —
       plus `redundant_verdict(outputs) → Agree | Outliers(ix) | Inconclusive` (majority
       by pairwise agreement; a 1-vs-1 or 2-vs-2 tie is Inconclusive → escalate, not
       punish) and `RedundantVerdict::outcome_for(i)` mapping majority→`Honored` /
       outlier→`Failed`. 13 unit tests.
     - **inc2 ⏳ (agent orchestration):** sampled deterministic dual-dispatch — issue an
       unpredictable temp=0 challenge to ≥2 providers of the same `model_id`, compare,
       record outcomes; escalate to a 3rd on Inconclusive. *Live validation gated on the
       parked ≥2-provider cross-NAT harness* (same dependency as M2.3 inc3 enforcement).
  3. **Logprob-fingerprint — optional, weaker.** Where the engine exposes `logprobs`, a
     cheaper-but-weaker fingerprint is possible: a middle ground, not the primitive.
  - **`verify::activation_hash` (TOPLOC) is retained but DORMANT** — it revives if an
    in-process Candle/Burn engine is added (activations exist again; see
    `docs/PQC_IMPLEMENTATION_PLAN.md` and the enclave/Candle direction).
  - **Revised exit test:** a provider that refuses/errors is downranked out of routing
    within N requests (M2.2(a)); a provider returning *wrong output* is caught by sampled
    redundant-exec and downranked (M2.2(b)).

**M2.3 — Credit ledger & priority-not-access** · WS-CREDIT · Rust · ~2.5w · **⚠️ PARTIAL** (redb ledger `protocol::store` ✅; priority-not-access *enforcement* + scarcity pricing pending — see M3.3)
- `protocol::ledger` over **`redb`** (pure-Rust, ACID, crash-safe — chosen over `sled`,
  which is unmaintained/crash-risky; `rusqlite`/SQLite is the battle-tested alt, already
  used in-repo for `credits.db`): single fungible balance; `price = compute_weight ×
  scarcity_multiplier` (multiplier clamped 0.5–2×, off until Phase 3); time-decayed
  `ratio` → `rate_cap` with a non-zero floor (throttle, never block); optimistic-unchoke
  reserve + one-time `starter_grant`; half-life decay; counterparty-diversity weighting.
  Ledger = aggregate of co-signed receipts replicated over gossip (`publish_event`) — so
  the local store is a rebuildable materialized view of the signed receipts, which bounds
  any DB-corruption blast-radius.
- **Exit test:** simulation harness — a leecher is throttled to the floor under
  contention while a contributor stays full-speed; a collusion ring minting mutual
  receipts gains ~no usable credit; decay reduces a stale balance on schedule.
- **Increment status:**
  - **inc1 ✅** credit core `protocol::credit::CreditAccount` (decayed give/take balance,
    `rate_cap` floor, per-counterparty anti-collusion cap, bytes codec; 10 sim tests).
  - **inc2 ✅** persistence + **take-side** accrual: `store::PEER_CREDIT` + the provider
    records `record_consumed` per accepted receipt, keyed by consumer libp2p id.
  - **inc3 ✅** **give-side** accrual: on a successfully co-signed receipt the consumer
    records the provider's `record_served` (`consumer.rs::record_contribution`, keyed by
    provider libp2p id, counterparty = self for the anti-collusion cap), sharing the
    `PEER_CREDIT` table. This closes the give/take loop — balances rise, not only fall.
    *Cross-process note:* provide/serve are separate roles; one ledger materializes when
    both use the same `--db` and merges on rehydrate. Live single-process unification lands
    with inc4.
  - **inc4 ✅ ENFORCEMENT**: unblocked by the **concurrent serve loop**
    (`workpool::WorkerPool` + `run_inbound(Arc<Self>, …, max_concurrency)` — the poll thread
    hands each request to a bounded pool, so a delay no longer head-of-line-blocks). The
    worker calls `provider.maybe_throttle`: pure `credit::throttle_multiplier(rate_cap)`
    (`0` for contributors, up to `MAX_THROTTLE_MULT=9×` a `BASE_THROTTLE=200ms` base at the
    floor) scales a delay applied off the poll thread, **budget-capped** to leave ≥1 worker
    free so a leecher flood can't stall the pool (priority, not access — slowed, never
    blocked). `provide --max-concurrency` (default 8). Live-validation under real contention
    pending the harness; scarcity pricing stays M3.3.
  - The same concurrent serve loop is the host for the **M2.2(b) audit-sampler trigger**
    (consumer-side follow-up: fire `audit_model` against `audit_rate_for` off the response
    path).

**M2.4 — Cold-start correctness** · WS-CREDIT/WS-SWARM · Rust · ~0.5w
- **Exit test:** simulated "launch" load → everyone served, credit inert; ramp demand
  → throttle engages exactly when contention appears.

### Phase 3 — The inference path: engine adapters + Rust gateway (Rust)

> **Promoted to the critical path by the pure-protocol pivot.** With no sharded engine,
> the engine adapter + the Rust HTTP/SSE gateway (**R1**, below) *are* how inference
> happens — they're the immediate next work after the protocol core, not a late
> "markets & reach" phase. **M3.1 is the first step.**

**M3.1 — Ollama engine adapter** · WS-ENGINE · Rust · ~1.5w · **✅ DONE (exit test passed live 2026-06-15)**
- `agent::adapters::ollama`: detect local Ollama models (quant + ctx via `/api/tags`,
  `/api/show`), advertise canonical ids, proxy inbound swarm requests to Ollama's
  OpenAI shim. The "anyone running Ollama flips a switch" flagship. Built end-to-end:
  Ollama adapter → provider swarm wiring (announce + serve loop) → consumer gateway
  (axum HTTP/SSE) → co-signed receipt at EOS → redb ledger. Runnable `openhydra-agent`
  binary (`provide` / `serve` roles).
- **Exit test:** a machine with Ollama and zero OpenHydra model files joins the swarm
  and serves a verified request from its Ollama models. **PASSED** — GPU1 (Lightning,
  Ollama `qwen2.5:0.5b`, behind NAT) provider ↔ Mac gateway (behind NAT), cross-NAT via
  the Linode Circuit Relay: gateway discovered the provider, routed over the relay, Ollama
  streamed the completion back as OpenAI SSE, and the co-signed receipt settled + ledgered
  on the provider (15 tokens). Two bugs surfaced + fixed by the live run: (1) provider
  one-shot announce → **periodic re-announce** within the relays' 300s provider-record TTL;
  (2) `get_providers` discover returned only the provider PeerId and dropped it (the full
  signed `PeerRecord` lives on the relays, not locally) → **chain a `get_record`** to pull
  it, and keep partial results on query timeout.

**M3.2 — Additional adapters** · WS-ENGINE · Rust · ~2w · **✅ DONE** (vLLM / LM Studio / llama.cpp / OpenAI-compatible in `agent/src/adapters/`; Apple Foundation Models still pending)
- vLLM, LM Studio, Exo (OpenAI-compatible → thin), then llama.cpp server and Apple
  Foundation Models (bespoke shims) behind a stable `EngineAdapter` trait.
- **Structure:** the M3.1 Ollama adapter ships as a single `agent/src/ollama.rs`. The
  *second* engine is the moment to introduce `agent/src/adapters/mod.rs` and move
  `ollama.rs` under it (`adapters::ollama`) — a cheap refactor that keeps the namespace
  honest as the set grows. Each new adapter is a module under `adapters/`, feature-gated
  (see §1.2). Apple Foundation Models, if its macOS-only FFI deps prove heavy, is the
  candidate to split into its own crate rather than a feature.
- **Exit test:** each adapter passes one conformance suite (detect → advertise → serve
  → receipt).

**M3.3 — Scarcity pricing on** · WS-CREDIT · Rust · ~1w · **❌ NOT STARTED** (no pricing code wired yet)
- Enable the damped multiplier with real rolling supply/demand per model class.
- **Exit test:** under-supplied class earns/costs more within the clamp; smooth across
  epochs (no oscillation).

**M3.4 — Private-swarm tooling** · WS-SWARM · Rust · ~1.5w · **❌ NOT STARTED**
- Allow-list / shared-secret scoping of discovery, routing, verification, ledger.
- **Exit test:** a 3-machine private swarm self-bootstraps with no public peers; an
  outsider can't discover or route to it.

### Phase 4 — Sharded fallback behind the router  ❌ REMOVED (pure-protocol pivot)

**Deleted.** There is no sharded tier and no two-tier routing — a provider serves only
what its own engine can run. The Python layer-parallel shard chain (`run_push_ring`, the
MLX/PyTorch stack) is **removed**, not kept behind the router (see §1.3 + the pivot
banner). The final "delete the Python engine + coordinator" step lives in the **R-track**
(R2/cutover), gated on the Rust inference path serving end-to-end.

### Rust daemon track (WS-AGENT) — the host inversion

Runs alongside Phases 1–3; consumes the crates they produce. Gated only on the HTTP
front door, not on the credit/verification rules (those are already Rust).

**R1 — Rust HTTP/SSE gateway** · ~3w · **(critical path — the consumer front door)** · **✅ DONE** (`agent/src/gateway.rs`: streaming + non-streaming, `GET /v1/models`, API-key auth)
- `openhydra-agent`: OpenAI/Ollama-compatible HTTP+SSE server (`axum`/`hyper`)
  fronting the libp2p core via `proxy_forward` + a streaming variant. The new host's
  front door. With no sharded tier, R1 + the M3.1 engine adapter together *are* the
  end-to-end inference path.
- **Exit test:** an OpenAI SDK client hits the Rust daemon and gets a streamed
  completion routed over libp2p to a provider's local engine — no Python anywhere.

**R2 — Cutover: delete the Python engine + coordinator** · ~2w (protocol logic already Rust) · **✅ DONE (2026-06-16)** — Python tree deleted; `network` is now a pure rlib (R2b-1) with the legacy IPC/sharding modules slated for R2b-2
- Once R1 + the engine adapter serve a request end-to-end, `openhydra-agent` becomes the
  sole host: **delete** the Python coordinator/inference engine (the §1.3 removal list)
  and retire the transitional `openhydra-pyext` Python-host glue. There is no Python
  worker to manage — a provider serves only via its external engine over HTTP.
- **Exit test:** `openhydra` runs as a single static binary with **no Python env at
  all**; a provider serves by proxying to its local engine; the Python tree is gone.

**R3 — Packaging & distribution** · ~1.5w · **❌ NOT STARTED** (now unblocked — R1/R2 done)
- Static binary, Homebrew formula, signed releases, one-line install, `--join` toggle.
- **Exit test:** clean machine, `brew install openhydra && openhydra --join`, serving
  within minutes, no toolchain.

### Desktop distribution track (WS-DIST) — the BitTorrent-client experience

The headless binary (R-track) serves servers and power users; this track delivers the
**one-click, GUI install** for everyone else — a qBittorrent/Transmission-style app on
macOS, Windows, and Linux. It reuses the existing `desktop/` scaffold (already **Tauri 2
+ React/Vite/Tailwind**) and the same workspace crates — the Tauri Rust backend embeds
the agent; the React frontend is the control panel. Engine bundling is **out of scope
for now** (the app assumes the user supplies/points at a local engine). Gated on the
`openhydra-agent` crate existing (≈ R1).

**D1 — Tauri shell over the shared crates** · WS-DIST · ~2.5w · **❌ NOT STARTED** (gated on R1 ✅ → now unblocked)
- Wire `desktop/src-tauri` to depend on `openhydra-agent`/`openhydra-protocol`/
  `openhydra-network`; run the agent in-process (or as a managed sidecar). Build the
  React control panel: join/leave swarm, pick local models to seed, live peers,
  credit balance + earnings, reputation, throughput.
- **Exit test:** launching the app starts the agent and shows live swarm state; toggling
  "join" advertises the node and serves a request end-to-end from the GUI.

**D2 — Per-OS one-click installers + signing** · WS-DIST · ~2w · **❌ NOT STARTED**
- Tauri bundler targets: `.dmg`/`.app` (macOS), `.msi`/NSIS `.exe` (Windows),
  `.deb`/`.rpm`/`.AppImage` (Linux). Apple Developer ID signing + **notarization**;
  Windows **Authenticode** signing. Tauri **auto-updater** channel + signed update
  artifacts. *(The hard part is certs/notarization, not the build — secure these early.)*
- **Networking & firewall UX.** Bind the control UI to **loopback only** (no firewall
  prompt). For P2P, rely on the existing **DCUtR hole-punching + Circuit Relay v2** so a
  node works behind NAT/firewall **without opening inbound ports** — the torrent-client
  "no port-forwarding needed" path. Handle the one-time OS inbound-connection prompt
  gracefully; optionally offer best-effort **UPnP/NAT-PMP** mapping. Do **not** request
  elevated privileges to silently punch firewall holes — scary, fragile, and unnecessary
  given the relay/hole-punch stack.
- **Exit test:** on each of the three OSes, a non-technical user double-clicks the
  installer and reaches a running app with **no terminal and no security warnings**;
  a shipped update auto-applies.

**D3 — Tray, autostart & background seeding** · WS-DIST · ~1w · **❌ NOT STARTED**
- System-tray presence, run-in-background (close-to-tray), autostart-on-login — the
  "always seeding" behavior of a torrent client. Surface seed/serve status in the tray.
- **Exit test:** the app keeps serving from the tray after the window is closed and
  across a reboot; the user can pause/resume seeding from the tray.

---

## 5. Sequencing & critical path

```
Phase 1 (M1.1→M1.3 ✅) ─► Phase 2 (M2.1 ✅ · M2.2/M2.3 ⚠️ · M2.4 ❌) ─► Phase 3 (M3.1 ✅ · M3.2 ✅ · M3.3/M3.4 ❌)
        └──────────────────────────► R1 gateway ✅ ─► R2 delete-Python ✅ ─► R2b-1 de-pyo3 ✅ ─► R3 package ❌
                                          Phase 4 (sharded) ❌ removed   ·   Desktop D1–D3 ❌ (unblocked)
```

> The PyO3-during-transition note that used to sit under this diagram is gone:
> R2b-1 removed PyO3 entirely; the network crate is a pure rlib.

- **Critical path to a usable network:** M1.1 → M1.2 → M1.3 → M2.1 → (M2.2) → **M3.1
  engine adapter → R1 gateway** — a verifiable, Ollama-friendly swarm that serves real
  inference end-to-end with **no Python in the path**. The engine adapter + R1 are now
  the near-term focus (the pivot promoted them from "Phase 3 markets" to the inference
  path itself).
- **No Python prototype tier.** Each rule is specified + vectored, then implemented in
  Rust once. The Python host calls the Rust crates via PyO3 until R2 deletes it.
- **R-track is the path, not a side-track.** With no sharded engine, R1 (front door) +
  M3.1 (engine adapter) *are* how inference happens; they can start as soon as the
  protocol core is stable (now).
- **Phase 4 is deleted** — there is no sharded fallback to route to.
- **Desktop track (D1→D3)** is gated on the `openhydra-agent` crate (≈ R1) and then
  runs in parallel — it consumes the same crates, so it adds packaging/UI, not protocol
  work. Secure code-signing certs + Apple notarization early; they're the long pole.

---

## 6. Cross-cutting concerns

- **Spec + golden vectors as the contract.** Every protocol rule (canonical id, receipt
  bytes, ledger transitions, pricing) gets a language-neutral spec note + golden test
  vectors *before* implementation. In a Rust-first world these double as the iteration
  discipline (replacing the discarded Python prototype) and as the basis for
  independent clients (spec §3 "a real protocol").
- **Simulation harness (Rust).** Credit/verification/anti-abuse need a multi-agent
  simulator (leechers, colluders, whitewashers, honest providers); it's the exit test
  for M2.2–M2.4 and M3.3. Build it as a Rust integration-test harness over the
  `protocol` crate.
- **Security review at boundaries.** Each network-facing addition (receipts, ledger
  gossip, engine proxy accepting stranger prompts) gets a `/security-review` pass.
  Rust's memory safety is a stated motivation (spec §3) — keep `unsafe` audited,
  especially the DLPack/bytemuck zero-copy paths.
- **Backwards compatibility / migration safety.** The existing Python HTTP API + engine
  keep serving throughout (additive-first); the Rust inference path (adapter → gateway)
  is proven end-to-end *before* R2 deletes the Python tree. No mid-air rebuild.

---

## 7. Risks & open questions

| Risk / question | Mitigation / owner action |
|---|---|
| **Rust iterates slower on still-unsettled rules** (credit pricing, verification sampling) | Specify each rule + golden vectors *before* the Rust implementation (§6), so Rust targets a frozen contract. No Python prototype tier — that's the deliberate tradeoff of the Rust-first decision. |
| Proof-of-inference cost at scale (sampling, overhead) | Tunable in M2.2; measure overhead in the Rust simulator before fixing a default. |
| Proof-of-inference has no activations in the whole-model path | The external engine returns text, not activations — TOPLOC applies only in the shard path. Whole-model verification leans on redundant-exec + reputation (+ engine logprobs where exposed). Designed into M2.2. |
| Canonical-model-id governance (who curates equivalence classes) | Ship a signed, versioned catalog; explicit `family`/`params` fields override the heuristic parser. Defer decentralised governance. |
| Decay half-life, scarcity clamp, unchoke reserve — need empirical tuning | Config constants with documented defaults; tune against the simulator, not guesses. |
| Content/safety defaults for stranger prompts | Per-provider opt-out + category filters in WS-ENGINE; conservative default-on local policy. |
| Single crate → workspace churn | Start M1.1 in-crate (`network`), promote to the `protocol`/`agent` workspace at the R-track. Keep `network/` in place; add only a root `[workspace]` + sibling crates. |

---

## 8. Immediate next actions (2026-06-16)

The protocol core + Rust inference path + cutover are done (see the status banner).
The open work, roughly in priority order:

1. **Connection reversal (Tier-2 NAT traversal).** Highest-leverage for real-world
   cross-NAT usability: let a NAT'd provider dial a reachable consumer so inference
   escapes the relay on symmetric CGNAT (where DCUtR can't help). Scope + design in
   [PEER_CONNECTIVITY.md](PEER_CONNECTIVITY.md) (provider-side proactive direct-upgrade,
   feature-flagged, live cross-NAT validation required).
2. **Finish the incentive layer so give-to-get bites:**
   - **M3.3 scarcity pricing** — wire the `scarcity_multiplier` (clamped 0.5–2×) over a
     rolling supply/demand signal per model class. *(No pricing code exists yet.)*
   - **Priority-not-access enforcement** — feed the ledger's time-decayed ratio →
     `rate_cap` into the router so a leecher is throttled (never blocked) under contention.
   - **M2.4 cold-start correctness** + **verification depth** (sampled redundant-exec
     enforcement, beyond the built hash/reputation primitives).
   - Build the **Rust simulation harness** (§6) — it's the exit test for all of the above.
3. **Reach:** **R3** packaging (Homebrew / signed single-binary) → **Desktop D1–D3**
   (the Tauri one-click app, now unblocked by R1).
4. **Cleanup:** **R2b-2** (excise the legacy IPC/sharding Rust modules woven into
   `event_loop.rs` — see [PEER_CONNECTIVITY.md] note + `r2-cutover-plan` memory) →
   **R2b-3** (drop the `proto` mod + build.rs prost step once `dispatcher` is gone).
5. **Optional:** Apple Foundation Models adapter (macOS Swift-FFI, its own crate).
