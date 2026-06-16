# Peer Connectivity & Relay Minimization

How two OpenHydra peers establish a connection, how to keep that connection
steady, how to keep the DHT full of *reachable* providers, and — the point of
this document — how to keep traffic **off the circuit relay**.

## Why relay-avoidance is a performance goal, not just robustness

A BitTorrent client behind the same carrier-grade NAT (CGNAT) that OpenHydra
struggles with will happily seed and leech at tens of Mbit/s over a relayed or
hole-punched path. It gets away with this because bulk file transfer is
**throughput-bound and latency-tolerant**: a 1 GB file is thousands of
independent pieces pulled from many peers in parallel, so the per-hop round-trip
is amortised into invisibility.

Autoregressive inference is the opposite — **sequential, stateful, and
latency-bound**. Token *N+1* cannot start until token *N* returns from the one
provider holding the KV-cache state. Every token pays one full path round-trip,
serially, and nothing can be parallelised away. So the same relay hop that is
free for BitTorrent turns ~12 tokens/s into ~3 tokens/s (measured cross-ISP:
direct IPv6 TCP ≈ 4.6–7.6 TPS vs. relay ≈ 3 TPS).

**Conclusion:** for OpenHydra the relay is a *latency tax*. Minimising it is the
single biggest cross-network performance lever, not a nice-to-have.

## The core principle (borrowed from BitTorrent)

> You do not need to be **connectable**. You need **one reachable end plus a
> willingness to dial out.** A connection is bidirectional regardless of who
> opened it.

A symmetric-CGNAT BitTorrent seed uploads fine because it *dials out* to
connectable peers and data flows back over that same socket. Crucially this
works **even on symmetric NAT, where hole-punching fails** — outbound flows
always traverse CGNAT, and the return path rides the mapping the outbound dial
created. The trick is to make sure the NAT'd side is the one dialing whenever the
other end is reachable.

## The connection ladder

For any two peers, climb this ladder; relay is only the floor. Each rung escapes
NAT for a strictly larger set of peers than the one below it.

| Tier | Path | Works when | Beats |
|---|---|---|---|
| 1 | **Direct IPv6** | either peer has public IPv6 (CGNAT is v4-only) | everything — no NAT at all |
| 2 | **NAT'd side dials a reachable v4 peer** (connection reversal) | one end is public / UPnP-mapped / promoted DHT server | **symmetric CGNAT**, where Tier 3 fails |
| 3 | **DCUtR hole-punch** (relay coordinates, then upgrade to direct) | both NAT'd, at least one *cone* NAT | endpoint-independent NATs |
| 4 | **Circuit Relay v2** (relay carries the data) | both symmetric-CGNAT, no IPv6 | last resort — eats the latency tax |

The design target is to push as many sessions as possible into Tiers 1–3, so
only the genuine both-symmetric-CGNAT-no-v6 residue lands on Tier 4.

## What OpenHydra does today

Grounded in the current `network/` + `agent/` code (function names, not line
numbers, to stay durable):

- **Flow is strictly consumer → provider.** The gateway/consumer selects a
  provider and calls `NetworkHandle::proxy_forward`; the event loop's
  `ProxyForward` handler dials the provider if not already connected. The
  provider (`agent/src/provider.rs`) is **100 % passive** — a `poll_inbound` →
  `dispatch` → `respond` loop that **never dials out**. (See the gap below.)
- **Direct before relay.** `ProxyForward` reuses an existing connection
  (`swarm.is_connected`) and only falls back to dialing relay circuits
  (`relay_circuit_addrs`) when there is none. Relay is fallback-only.
- **IPv6-first is explicit.** A startup probe (`probe_ipv6_capable`, "F-9")
  detects working outbound v6; `relay_circuit_addrs` filters out `/ip6/` relay
  addresses on v6-incapable hosts; an auto-dial path prefers cached **QUIC IPv6**
  addresses from Identify and skips public relays (hole-punch there is pointless).
- **DCUtR upgrade** is coordinated by a gossip signal (`REQUEST_HOLE_PUNCH` →
  `DialPeer` with `PeerCondition::Always`), so a relayed connection can be
  promoted to direct once both sides dial simultaneously. Success/failure are
  counted (`dcutr_successes`/`dcutr_failures`, surfaced via `GetDcutrStats`).
- **Keepalive** is an explicit 15 s libp2p `ping` interval (`swarm.rs`), tuned to
  keep mobile-hotspot NAT mappings alive through the 1–3 s silence between tokens
  (without it, the mapping is evicted and each re-dial costs 2–4 s — the dominant
  factor in the worst cross-ISP benchmark). libp2p's idle-connection timeout is
  the default 300 s.
- **Connected-first ranking** (R-DHT-8): the consumer floats providers it already
  has a live connection to to the top of the ranking, so a warm connection is
  reused and failover to a cold provider is the exception.
- **No explicit pre-warming or pooling.** Connection reuse is libp2p's implicit
  per-peer pool; the consumer dials only at request time.

Mapping to the ladder: Tier 1 (IPv6) and Tier 3 (DCUtR) are implemented; Tier 4
(relay) is the wired fallback. **Tier 2 — connection reversal — is the gap.**

## The gap: let the NAT'd provider be the dialer

Today, when a consumer wants a NAT'd provider, the **consumer dials the
provider**. Because the provider is unreachable, that resolves to a Tier-4 relay
circuit. But the provider *is the NAT'd side* — the very side that, per the core
principle, should be doing the dialing.

This matters most for **symmetric CGNAT** (e.g. the ACT/CABLELITE connection in
the cross-ISP tests): a NAT test there shows inbound TCP/UDP closed and
endpoint-dependent mapping, so **DCUtR (Tier 3) will usually fail** — simultaneous
open can't predict the mapped port. Connection *reversal* needs no timing or port
prediction: if the **consumer** is reachable (public, UPnP-mapped, or a promoted
DHT server), the provider simply dials *out* to it, and the established
connection carries the inference directly — no third-party relay in the data
path. This is exactly how a CGNAT BitTorrent seed serves.

**Proposed mechanism (connection reversal):**

1. Consumer discovers a NAT'd provider (`requires_relay` / no reachable direct
   addr) but is itself reachable.
2. Instead of opening a relay circuit for the whole session, the consumer signals
   the provider (via the existing relay/gossip rendezvous) with its own reachable
   multiaddr(s).
3. The provider **dials the consumer directly** and the request/response protocol
   runs over that provider-initiated connection.
4. Fall back to Tier 3 (DCUtR) then Tier 4 (relay) only if the consumer is *also*
   unreachable.

Requirements this introduces: the provider needs explicit dial logic (absent
today), and the consumer needs to accept an inbound serve connection (i.e. the
request/response roles decouple from who-dialed). Both are tractable on top of
the existing swarm, and the payoff is converting a class of relay sessions —
precisely the symmetric-CGNAT class DCUtR cannot rescue — into direct ones.

## Keeping connections steady

- **Keepalive on direct links too.** The 15 s ping already protects relay
  circuits; apply the same discipline to hard-won Tier 1/2 direct connections so a
  CGNAT mapping (evicted after ~40–70 s of silence) survives the gaps between
  requests instead of decaying back to relay.
- **Pre-warm + reuse.** Pay the ~1.2 s connection setup once, ahead of the first
  token, and keep the connection hot for the session (and ideally across
  back-to-back requests). This is the parked "route pre-warming" idea; combined
  with connected-first ranking it turns steady-state inference into zero-dial.
- **Hedged dispatch for the tail.** Racing a second provider and keeping
  whichever answers first is BitTorrent's "pull from whoever's fastest" applied at
  the request level — it covers dead/slow providers without trying to make a
  single token stream parallel (which is impossible).

## Keeping the DHT full of *reachable* peers

A healthy peer list means *reachable providers discoverable for a model*, not
merely a big table. OpenHydra has the primitives (R-DHT-1/6/7/8); the posture to
adopt from a mature DHT (BiglyBT's mlDHT, ~128k nodes):

- **PEX-first.** Learn peers *from* peers via gossipsub PEX (R-DHT-1) as the
  primary fill path; treat DHT lookups as the backstop. PEX is how a fat, fresh
  peer list materialises in seconds.
- **Aggressive maintenance + fast eviction.** Continuous bucket refresh and quick
  liveness-based eviction of dead providers (R-DHT-8) keep the list *fresh*.
  Pruning unreachable records matters more than retaining many.
- **Re-announce with margin.** Providers re-announce every 120 s against a 300 s
  record TTL (≈2.5× headroom) so they never silently fall out of discovery.
- **Maximise connectable anchors.** Every promoted DHT server / public-IPv6 /
  UPnP-mapped peer (R-DHT-2, R-DHT-4) is a direct dial target that pulls NAT'd
  peers *off* relay. More anchors ⇒ a smaller both-NAT'd residue ⇒ less relay.
- **Redundant entry points.** Multiple bootstraps + mDNS + DHT + PEX; resilience
  comes from never depending on a single discovery source.

## The relay-minimization KPI

The relay should be a **stepping stone** (coordinate a hole-punch / reversal),
not the data path. If sessions *linger* on relay, that is exactly where the
latency tax — the 3 TPS — lives.

**Instrument and watch the relay→direct upgrade rate** (extend the existing
DCUtR counters): for each cross-NAT session, did it upgrade to Tier 1/2/3, or
stay on Tier 4, and how long until upgrade? That ratio is the concrete success
metric for everything in this document.

## Prioritised recommendations

1. **Lean on IPv6 (Tier 1) hardest.** It sidesteps NAT entirely and is your best
   measured TPS. Prefer v6 addresses when dialing, advertise v6 prominently, and
   ensure two v6-capable peers never relay or hole-punch.
2. **Add connection reversal (Tier 2).** Let a NAT'd provider dial a reachable
   consumer. This is the missing rung and the only one that rescues symmetric
   CGNAT, where DCUtR cannot.
3. **Pre-warm + keepalive direct connections** so a Tier 1/2 link, once won, is
   not lost to NAT-mapping eviction between tokens or requests.
4. **Maximise connectable anchors** (server promotion, UPnP, IPv6) to shrink the
   relay residue.
5. **Measure the relay→direct upgrade rate** and treat it as the headline
   connectivity KPI.

## Appendix — Connection reversal (Tier 2): implementation scope

The single highest-leverage gap from the ladder above. Today the flow is strictly
consumer-dials-provider, so a NAT'd provider resolves to a relay. Connection
reversal makes the **provider** (the NAT'd side) dial a **reachable consumer**, so
inference runs direct. It is the only escape on **symmetric CGNAT**, where DCUtR
(simultaneous-open) structurally fails.

**Key enabler:** libp2p `request_response` is **bidirectional once connected** — the
provider only reverses the *connection direction*, not the request/response roles.
Once the provider dials in, the consumer's existing `proxy_forward → send_request`
runs unchanged over that connection. The provider's outbound dial succeeds even on
symmetric CGNAT (outbound flows always traverse the NAT).

### Design — two options

**Option A (recommended): provider-side proactive direct-upgrade.** On accepting a
**relayed inbound** connection, the provider checks whether the remote peer
(consumer) has **globally-routable** addresses libp2p already learned via
Identify/observed-addr, and if so **dials them directly**; libp2p migrates traffic
off the relay. No new wire protocol, no consumer-side changes (Identify already
advertises listen addrs over the circuit). It is the v4 + relayed-trigger
generalization of the existing auto-QUIC-IPv6 upgrade in `event_loop.rs`.

**Option B (fallback/general): explicit reverse-connect signal.** A control frame the
consumer sends the provider **over the relay circuit it already opens** (not gossip),
carrying its reachable multiaddrs. More general, but needs the anti-abuse gates built
explicitly. Prefer A; keep B for the case where Identify-over-relay is insufficient.

### Change list (Option A)

| # | Change | File |
|---|---|---|
| 1 | `SwarmCommand::GetExternalAddrs` + `NetworkHandle::external_addrs()` (surface `state.external_addrs` + UPnP addrs) | `event_loop.rs`, `handle.rs` |
| 2 | On `ConnectionEstablished` over a `/p2p-circuit` endpoint, look up the peer's Identify-learned addrs | `event_loop.rs` |
| 3 | If any pass `is_globally_reachable_addr`, dial them: `DialOpts::peer_id(p).addresses(addrs).condition(Always)` | `event_loop.rs` |
| 4 | Prefer the resulting direct connection; tear down the relay circuit only after direct is confirmed carrying traffic | `event_loop.rs` |
| 5 | Reversal-upgrade counters + extend `GetDcutrStats` (feeds the relay→direct KPI above) | `event_loop.rs` |
| 6 | Feature flag `enable_connection_reversal` (default **off**) | `node.rs` `NodeConfig`, `main.rs` |

Option A needs **no** `provider.rs`/`consumer.rs` agent-logic changes — it is a
network-layer behavior. (Option B adds consumer "I'm reachable → request reversal"
logic + a proxy control-frame.)

### Anti-abuse (amplification / DoS — the part to get right)

Option A is safe by construction: it (1) only ever dials a peer it is **already** in
an authenticated relayed session with, (2) only dials **libp2p-observed/Identify**
addresses, never self-reported ones, (3) **validates** every candidate with
`is_globally_reachable_addr` (drops CGNAT/private/loopback/ULA/circuit), and (4)
**rate-limits** reversal dials per-peer and globally (reuse the relay leech-table
pattern). Optionally reputation-gate the upgrade.

### Edge cases
- **Both peers NAT'd** → no globally-routable addrs → fall through to DCUtR then relay
  (reversal is purely additive).
- **Consumer also symmetric-CGNAT** → its addrs fail the routable filter → no reversal
  attempted (correct).
- **Stale Identify addrs** → dial fails fast, session stays on relay, failure counted.
- **Migration race** → never drop the working relay path before the direct one is
  observed carrying a response.
- **Both have IPv6** → Tier 1 wins before reversal is needed.

### Validation
- **Unit:** address-validation + gating (routable filter, rate-limit, reputation floor).
- **Integration:** 2-node with a simulated relay; assert the relayed session upgrades
  to direct and `reversal_successes` increments.
- **Live cross-NAT (mandatory before default-on):** NAT'd provider on ACT/CGNAT
  (GPU1 or Mac) ↔ reachable consumer (public VPS / public-IPv6 Mac). Measure the
  **relay→direct upgrade rate** and **TPS before/after** — target: symmetric-CGNAT
  sessions climb from the ~3 TPS relay floor toward the ~7.6 TPS direct number.

### Risk & rollout
Moderate-risk (touches the live event loop the bootstraps share): feature-flag off →
unit/integration green → live cross-NAT validation → default on. Compiler + the
network test suite are guardrails; the live test is the real gate. **Effort:** ~1–1.5
weeks for Option A incl. live validation; +~0.5w for Option B if needed.

---

See also: [protocol.md](protocol.md) §3 (Transport & discovery),
[PROTOCOL_IMPLEMENTATION_PLAN.md](PROTOCOL_IMPLEMENTATION_PLAN.md) §8 (next actions),
`docs/DHT_ROBUSTNESS_REMEDIATION.md` (the R-DHT-1…11 work this builds on).
