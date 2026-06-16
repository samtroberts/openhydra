# OpenHydra Protocol

> A peer-to-peer protocol for free, distributed AI inference. Plug in whatever you
> already run — Ollama, vLLM, LM Studio, llama.cpp, Exo, Apple Foundation Models —
> and join a global network that serves and consumes inference. BitTorrent, for AI.

This document describes the OpenHydra protocol: its architecture, the request
lifecycle, model identity, the give-to-get incentive layer, trust and
verification, privacy, anti-abuse, and how the network bootstraps from empty.

---

## 1. What OpenHydra is

OpenHydra is a **discovery, routing, and trust protocol** for AI inference — not
an inference engine. Nodes run their own local LLM stack and "seed" spare
capacity to the network; clients route requests to whoever can serve the model
they want.

The analogy is BitTorrent. BitTorrent never splits the *computation* of a file —
it routes you to peers who **have** the content. OpenHydra routes you to peers who
**have a model loaded** and have capacity to run it. The protocol's job is:

1. **Discovery** — who has model *X* with spare capacity, right now?
2. **Routing** — reach them through NAT, pick the best one.
3. **Trust** — verify the answer actually came from that model.
4. **Fairness** — reward those who serve, so supply exists at all.

Two execution modes sit under one protocol:

- **Whole-model routing (primary).** A request goes to a single node that has the
  whole model loaded and runs it locally at native speed. The network pays **one
  round-trip per request**.
- **Sharded inference (fallback).** For models too large for any single node
  (e.g. 70B+ on consumer hardware), OpenHydra falls back to its layer-parallel
  pipeline, splitting the model across peers. This is the original OpenHydra
  capability, retained for the cases nothing else can serve.

### Design goals

- **Engine-agnostic** — any local inference engine can join via a thin adapter.
- **Platform-agnostic** — macOS, Linux, Windows; Apple Silicon, NVIDIA, AMD, CPU.
- **No central server** — discovery and routing are peer-to-peer.
- **Free, by reciprocity** — give-to-get, no money, no crypto, no chain.
- **Verifiable** — consumers can trust the inference actually ran.
- **Decentralised but bootstrappable** — works from a cold, empty network.

---

## 2. Why a protocol (and not "a faster Petals")

OpenHydra began as a sharded inference network — a "faster Petals." Sharded
inference across a wide-area network is **latency-bound by design**: layer-parallel
execution makes *every token cross the network N times* (N = pipeline depth).
Measured throughput bears this out — strong on a LAN, materially slower over a
relay, and worst across consumer ISPs. No wire format or ring optimisation beats
the speed of light; per-token round-trips dominate.

Whole-model routing inverts the cost model. If one node holds the model and runs
it locally, the network cost is **one round-trip per request**, and the model runs
at the node's native local speed (tens of tokens/second on modern hardware). The
value shifts from *raw speed we cannot deliver over a WAN* to *access and
aggregate capacity*, which a network delivers naturally.

So the sharding stays — as the fallback for models nothing else can hold — and the
**protocol** becomes the product.

---

## 3. Architecture

### Roles

| Role | Responsibility |
|---|---|
| **Provider** | Runs a local inference engine + the OpenHydra agent. Advertises its models and capacity to the network, serves inbound requests. |
| **Consumer** | Issues inference requests (OpenAI/Ollama-compatible) and receives results. Every node can be both. |
| **Router** | Resolves providers for a requested model, ranks them, forwards the request, verifies the result. Runs in-process on the consumer or as a shared coordinator. |
| **Bootstrap** | Long-lived nodes that seed the DHT and relay traffic for peers behind NAT. |

### The provider agent

The agent is a thin sidecar that makes any local engine speak OpenHydra:

1. **Detects** locally available models (and their quantisation, context length).
2. **Advertises** them to the DHT, with a capability record.
3. **Accepts** inbound swarm requests and **proxies** them to the local engine
   over its native API.
4. **Streams** results back and exchanges signed receipts.

Most engines already expose an OpenAI-compatible HTTP API (vLLM, LM Studio, Exo,
and Ollama via its OpenAI shim), so the adapter is small. Engines with bespoke
APIs (llama.cpp server, Apple Foundation Models) get a dedicated shim.

### Transport & discovery

The networking layer is a Rust **libp2p** stack:

- **Kademlia DHT** for peer and model discovery.
- **QUIC + TCP** transports.
- **DCUtR hole-punching** and **Circuit Relay v2** fallback for NAT traversal.
- **AutoNAT** for reachability detection.
- **gossipsub** for swarm-wide events.

How these combine into a connection-establishment *ladder* (IPv6-direct →
connection reversal → DCUtR → relay), how to keep connections steady and the
DHT full of reachable providers, and — because relay is a per-token latency tax
on inference — how to keep traffic **off the relay**, are covered in
[PEER_CONNECTIVITY.md](PEER_CONNECTIVITY.md).

Model records live in the DHT under a stable key
(`/openhydra/model/{model_id}/{peer_id}`), so resolving "who has model *X*?" is a
DHT lookup.

### API contract

Consumers speak the APIs they already use:

- **OpenAI-compatible** — `POST /v1/chat/completions` with SSE streaming.
- **Ollama-compatible** — `POST /api/chat`, `/api/generate`.

This means existing clients (Open WebUI, Continue.dev, the OpenAI SDKs) point at
OpenHydra by changing one base URL.

### Implementation: a Rust-first agent

The protocol layer is built as a **single, statically-linked Rust daemon** —
not a Python application. The distinction that drives this: in the whole-model
path the protocol **never runs the model**; it is I/O-bound systems glue (HTTP in
→ route over libp2p → proxy to the local engine's HTTP API → stream out →
exchange receipts). Discovery, routing, NAT traversal, the credit ledger,
receipts, gossip, rate-limiting, and proof checks are all distributed-systems
work, for which Rust is the natural fit — and the hard foundation (libp2p,
Kademlia, Circuit Relay, the OHV2 wire format, zero-copy tensor decode) is
already Rust today.

**Why Rust for the agent:**

- **Distribution is the deciding factor.** The network's success depends on
  frictionless provider onboarding — *anyone running Ollama flips a switch*. A
  single static binary (`brew install`, a dropped-in daemon, no Python env, no
  wheel build, no dependency pinning) is what makes that true. A Python app with
  a compiled extension is the opposite of frictionless and would throttle supply.
- **Right tool for the job.** No GIL, real concurrency, predictable latency, and
  a tiny background-daemon footprint that doesn't contend with the user's actual
  inference process.
- **Safety.** A network-facing daemon accepting requests from strangers benefits
  directly from Rust's memory safety.
- **A real protocol.** A clean spec plus a reference Rust daemon makes OpenHydra
  a protocol others can implement — the way BitTorrent succeeded through many
  independent clients, not one app.
- **Builds on strength.** It extends the existing Rust networking core and
  removes the per-request Python↔Rust (PyO3) boundary from the hot path.

**The boundary — what stays Python.** The protocol is Rust; the *inference* is
not. Engine adapters are just HTTP (trivial in Rust), but the **sharded-inference
fallback** (§11) executes real tensors in MLX / PyTorch, which cannot be rewritten
in Rust. So the Rust router invokes the existing Python/MLX sharding subsystem as
a subordinate fallback tier, only when no single provider can host a model. This
inverts today's arrangement (Python host calling Rust) rather than adding new
infrastructure: Rust becomes the host and calls Python for the fallback math. It
also means the existing Python coordinator is **not** rewritten wholesale — most
of it is sharding orchestration, which becomes that fallback.

**Sequencing.** The networking/agent core (HTTP proxy + libp2p routing +
capability advertisement) is stable enough to build in Rust immediately. The
credit-ledger and verification rules are still being designed, so those are
specified first and hardened into Rust once settled — avoiding building a moving
target fast in a slow-to-change language.

---

## 4. Model identity & equivalence

The same "model" is not the same everywhere. `Llama-3-8B` quantised Q4_K_M in
Ollama produces different outputs than fp16 in vLLM or Q5 in LM Studio. Routing a
request to an incompatible provider silently degrades quality.

OpenHydra therefore defines a **canonical model id** — the protocol's equivalent
of a BitTorrent infohash:

```
model_id = {family}/{params}/{quantization}/{chat_template_hash}
# e.g.  llama-3/8b/Q4_K_M/9f2c…
```

- A request names a canonical id (or a family + constraints, and the router picks
  a compatible quant).
- Providers advertise the exact canonical ids they host.
- The router only sends a request to providers whose canonical id is compatible.

A provider's **capability record** carries everything needed to route well:

```
{
  model_id, context_length, max_output_tokens,
  throughput_tok_s, queue_depth, backend ("ollama"|"vllm"|…),
  hardware_class, region, requires_relay, reputation
}
```

---

## 5. Request lifecycle

```
Consumer                Router                 DHT            Provider
   │  chat request        │                     │                │
   ├─────────────────────▶│  resolve(model_id)  │                │
   │                      ├────────────────────▶│                │
   │                      │◀── provider list ───┤                │
   │                      │  rank by health /    │                │
   │                      │  RTT / throughput /  │                │
   │                      │  ratio               │                │
   │                      ├─ forward request ───────────────────▶│
   │                      │                      │   run locally  │
   │◀──── stream tokens ──┼◀─────────────────────────────────────┤
   │                      │  sample-verify       │                │
   │  signed receipt ─────┼─────────────────────────────────────▶│
```

1. **Resolve.** Router looks up providers for the canonical model id in the DHT.
2. **Rank.** Candidates are scored by liveness, RTT, advertised throughput, queue
   depth, reputation, and the consumer's contribution ratio (see §6).
3. **Route.** The request is forwarded to the chosen provider (direct, or via a
   relay circuit if the provider is behind NAT).
4. **Execute.** The provider runs the request on its local engine and streams the
   result back.
5. **Verify.** A sampled fraction of requests are checked (see §7).
6. **Settle.** Consumer and provider exchange a co-signed receipt that updates the
   contribution ledger.

**Two-tier routing.** If no single provider can host the requested model (too
large for any one node), the router falls back to the **sharded pipeline**:
assemble a layer-parallel chain of peers and stream activations between them. The
consumer sees the same API; only the path differs.

**Graceful degradation.** If a model has no live providers, the router can fall
back to the nearest available smaller model in the same family (opt-in), rather
than failing.

---

## 6. Incentive layer — give-to-get, without crypto

Serving inference costs real GPU time, power, and responsiveness. Without a reason
to provide, a free network collects consumers and no providers. OpenHydra's
incentive layer is a **non-monetary, non-transferable contribution ledger** — the
BitTorrent idea, minus the coin, chain, and speculation.

### A single fungible credit

Each peer has one credit balance. Credit is earned by serving verified inference
and spent by consuming it. The **rate** of earning and spending depends on the
model:

```
price(model) = compute_weight(model) × scarcity_multiplier(model_class)
```

- **`compute_weight`** tracks the rough cost of a token for that model (a function
  of active parameters / memory). Serving a 70B earns faster; consuming one spends
  faster.
- **`scarcity_multiplier`** is a *damped, slow-moving* factor per model class,
  recomputed each epoch from rolling supply/demand and clamped to a tight band
  (e.g. 0.5×–2×). When a model class is under-supplied, serving it earns more and
  consuming it costs more — pulling supply toward where it's needed and rationing
  scarce capacity to those who contributed most.

A single fungible credit (rather than separate per-tier currencies) means a
modest node can serve plenty of cheap, abundant small-model work and still earn
its way to occasional big-model access. Pricing on **scarcity** (not raw compute
alone) is what prevents the "seed cheap, leech expensive" arbitrage: if scarce
capacity were under-priced, it would be drained by people who never provided it,
and providers would stop providing it. The damping (slow epochs, tight clamp)
keeps the rate from becoming a volatile, gameable market.

### Priority, not access

Contribution ratio sets **priority**, never hard access:

```
ratio   = (served + starter_grant) / (consumed + ε)        # time-decayed
rate_cap(model) = base_rate × clamp(ratio, floor, ceil)
```

- `ratio ≥ 1` → full-speed service.
- `ratio → 0` → throttled to a small non-zero floor, never zero.

Nobody is locked out; heavy leechers simply queue behind contributors. This
mirrors BitTorrent, which slows leechers rather than banning them.

**Optimistic unchoke.** Every provider reserves a slice of capacity (round-robin)
for zero-history peers, plus a one-time `starter_grant`. Newcomers can always
start; the reserve is also the primary defence against whitewashing (see §9).

**Decay.** Balances decay with a half-life (≈ weeks), so credit reflects *recent*
contribution and cannot be banked indefinitely then drained.

### Receipts & the ledger

Credit accrues only against a **co-signed receipt**:

```
receipt = sign_provider( sign_consumer( provider, consumer, model_id, tokens, nonce, ts ) )
```

- The consumer signs that they received the tokens; the provider submits the
  receipt to claim credit. Neither side can unilaterally inflate.
- The `nonce` prevents double-counting.
- The ledger is the aggregate of co-signed receipts, replicated over the DHT /
  gossip, with monotonic per-peer counters to resist rollback.

Credit is **weighted by counterparty diversity**: work served to many distinct,
reputable consumers is worth more than the same volume served to a small clique —
which neutralises collusion rings that "serve" each other to mint credit.

---

## 7. Trust & verification

When one node runs an entire request, the consumer must trust that the output
genuinely came from the named model and wasn't cached, truncated, or tampered.
**Verification is the foundation the entire incentive layer rests on** — credit is
only as trustworthy as the proof beneath it.

Three complementary mechanisms:

1. **Proof-of-inference (TOPLOC-style).** A cheap, locality-sensitive proof over
   the model's activations lets a verifier confirm a result is consistent with the
   claimed model, checked on a *sampled* fraction of requests.
2. **Redundant execution.** A fraction of requests are run on two or more
   independent providers and compared; divergence flags a bad actor.
3. **Reputation.** Providers accrue a reputation score from verification outcomes;
   repeated failures downrank them until the network stops routing to them.

Sample rates are tuned by stakes and reputation: trusted, long-lived providers are
checked rarely; new or suspect ones, often. This balances verification cost
against coverage — verifying every token would tax the whole network, while
verifying none would make credit forgeable.

---

## 8. Privacy

Whole-model routing means the serving provider sees the prompt — the same trust
relationship you have with any inference provider. OpenHydra mitigates rather than
eliminates this:

- **Transport encryption.** Ed25519 identities, X25519 ECDH key agreement, and
  AES-256-GCM per hop. Traffic is encrypted in transit end-to-end between peers.
- **Trusted-peer pinning.** Consumers can prefer or restrict to providers they
  trust (by reputation threshold or explicit allow-list).
- **LAN-only / private mode.** Run entirely within your own machines or a private
  swarm, where no external peer participates (see §10).
- **Sharded mode for sensitive queries.** Routing through the layer-parallel
  fallback means no single peer sees the full prompt — a stronger privacy posture
  at the cost of throughput.

Privacy posture is a per-request choice, not a global setting.

---

## 9. Anti-abuse

- **Rate limiting / leech control.** Per-peer request and byte budgets with
  jittered lockout on abuse — already implemented at the bootstrap/relay layer.
- **Whitewashing (cheap identities).** The deepest P2P attack: dump a bad-ratio
  identity, rejoin fresh. Defended structurally — newcomers are *throttled, not
  banned*, so a fresh identity is strictly worse than an established good one, and
  resetting buys nothing. A mild identity cost (proof-of-work on join, or
  attestation) raises the bar further.
- **Sybil / collusion minting.** Verified-only accrual plus counterparty-diversity
  weighting (see §6) make manufactured credit economically worthless.
- **Content & safety.** Providers execute prompts from strangers; the agent
  supports local content policy, per-provider opt-outs, and category filters.
  Routing respects provider-declared policies.

---

## 10. Bootstrapping a cold network

A new network has no credits and few providers. These are two distinct problems,
and the protocol's design dissolves the first.

### Credit cold-start dissolves under priority-not-access

Credit is a **rationing** mechanism, and a new network has nothing to ration. At
launch, utilisation is low and providers have idle capacity, so the optimistic
floor serves everyone regardless of balance. Credit accrues but does not *gate*
anything until contention appears.

The system "turns on" gradually:

1. **Launch (low utilisation).** Spare capacity everywhere → everyone served, free
   and fast. Credit accumulates but is inert.
2. **Growth (demand approaches supply).** Contention appears on popular models →
   ratio begins biasing priority. By now, anyone who served has banked credit;
   pure leechers start to feel the throttle. The incentive to provide switches on
   exactly when scarcity makes it matter.
3. **Mature.** Full give-to-get under contention.

A small, identity-gated `starter_grant` smooths the first request, but it is *not*
what makes cold-start work — the priority floor is.

### Supply cold-start: make the node useful before the network exists

The real bootstrapping challenge is getting providers in the door. The protocol is
designed so that:

- **The agent is useful standalone.** It is a clean, OpenAI/Ollama-compatible
  gateway in front of whatever engine you already run — model management, one
  endpoint, local-first. People install it for the *local* utility; joining the
  swarm is an opt-in toggle on an already-installed base. The network is upside,
  not a precondition.
- **Private swarms come first.** Closed, semi-trusted clusters — a lab, a team, a
  group of friends pooling their own machines — have co-located supply and demand
  and internal incentive (it's your own hardware; nobody needs persuading). Each
  private swarm self-bootstraps. The public network is the union of many such
  swarms over time, so cold-start is many small self-closing graphs rather than
  one large empty one.
- **Early providers are rewarded extra.** A time-boxed earn-rate bonus and a
  durable "founding seeder" reputation recruit the scarce side first.

### Private swarms

A private swarm is the same protocol scoped to an allow-list of peers (or a shared
secret). Discovery, routing, verification, and the credit ledger all operate
within the group. Private swarms double as the privacy-maximal deployment (§8) and
the cold-start unit (above).

---

## 11. Backwards compatibility: sharded inference

The original layer-parallel sharded pipeline is retained as the **fallback tier**
for models too large for any single node. It uses the same transport, discovery,
and verification machinery; only the execution path differs (activations stream
between peers instead of a single node running the whole model). Routing tries
whole-model providers first and falls back to assembling a shard chain only when
necessary.

---

## 12. Roadmap

| Phase | Scope |
|---|---|
| **1 — Protocol core** | Provider agent (Ollama first), canonical model-id scheme, DHT capability records, router with health/RTT ranking. Whole-model routing end-to-end. |
| **2 — Trust & credit** | Sampled proof-of-inference + redundant-execution audit, reputation, co-signed receipts, fungible credit ledger, priority-not-access gating. |
| **3 — Markets & reach** | Damped scarcity pricing, more engine adapters (vLLM, LM Studio, llama.cpp, Exo, Apple Foundation Models), private-swarm tooling. |
| **4 — Sharded fallback** | Integrate the layer-parallel pipeline as the large-model fallback tier behind the same router. |

---

## 13. Non-goals & open questions

**Non-goals.** A token or coin; an on-chain settlement layer; guaranteeing a
specific latency or throughput SLA across heterogeneous volunteer hardware; being
an inference engine.

**Open questions.**

- The exact form and cost of proof-of-inference at scale (sampling strategy,
  overhead budget).
- Canonical-model-id governance: who curates equivalence classes across engines
  and quantisations.
- Decay half-life, scarcity-multiplier clamp band, and optimistic-unchoke reserve
  — all need empirical tuning against real demand.
- Content/safety policy defaults for providers serving arbitrary prompts.
