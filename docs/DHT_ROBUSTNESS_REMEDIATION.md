# OpenHydra DHT Robustness — Remediation Plan

> Companion to [`PROTOCOL_IMPLEMENTATION_PLAN.md`](./PROTOCOL_IMPLEMENTATION_PLAN.md).
> Authored 2026-06-15 after a live cross-NAT bring-up (Mac ⟷ GPU1 over the Linode
> relay) in which discovery **degraded to zero** after repeated peer churn.

## 1. Problem statement

OpenHydra's discovery DHT is **not actually peer-to-peer**. The peer Kademlia config
(`network/src/swarm.rs:114`) never calls `set_mode(Server)`, so NAT'd providers and
consumers — reachable only via relay circuits — stay in libp2p Kademlia **client mode**.
A client node **does not store records and does not answer queries**. Therefore the
*entire* DHT (every record, every lookup) is served by the **3 bootstrap relays** — the
only nodes in `Server` mode (`network/src/bootstrap_bin.rs`).

Consequences observed live:
- Storage redundancy = 3 (or **1**, when a single `--bootstrap` is used) instead of the
  millions BitTorrent enjoys.
- Stale provider records (300 s TTL) **dominate** a tiny result set instead of being
  statistically diluted.
- A single relay hiccup (dropped reservation → the provider's re-announces stop landing)
  collapses discovery to **zero**.
- Under *real-world* churn this degrades **faster**, not slower.

**Root cause:** the DHT has the fragility of a centralized index with none of its
consistency. The fix is to make reachable peers first-class DHT participants — BitTorrent's
actual property — while keeping libp2p's cryptographic-identity security model.

## 2. What "reachable" means (promotion gate)

A peer may be promoted to Kademlia **server** (store + route + answer) **only** when it has
a confirmed-reachable address. "Reachable" =

1. **Public IPv4** (rare for residential, common for servers/VPS), or
2. **Public IPv6** (globally routable, usually un-NATed — high modern coverage; this is the
   underrated near-term win), or
3. Behind **full-cone / restricted-cone NAT** (the UDP "outbound opens inbound" property —
   others can reach the observed `IP:port`), or
4. **DCUtR-direct** connectivity established (a hole-punched direct path), or
5. A **UPnP/NAT-PMP-mapped** port on a cooperating router.

**CGNAT and symmetric-NAT peers stay clients.** BitTorrent does the same — it does not make
*every* node a server; it makes every *reachable* node a server and absorbs the rest via
scale. Promotion must be **conditional**: never `set_mode(Server)` unconditionally —
advertising as a server while unreachable causes **black-hole routing** (queries routed to
you time out, degrading the DHT for everyone). Instead, feed confirmed-reachable addresses
into the swarm's external-address set (`add_external_address`, already wired at
`event_loop.rs:1591` / the `ExternalAddrConfirmed` handler) and let auto-mode promote.

## 3. BitTorrent contrast (why theirs doesn't degrade)

| | BitTorrent Mainline DHT | OpenHydra today |
|---|---|---|
| DHT nodes | millions; every reachable client is a node | 3 relays (peers are clients) |
| Wire | KRPC: bencode RPC over **raw UDP**, connectionless | libp2p Kad: protobuf over Noise-encrypted streams |
| Identity | self-assigned (sybil-prone; BEP 42 mitigations) | **cryptographic ed25519 peer-ids** (needed for the trust layer) |
| NAT | UDP hole property; UPnP; symmetric/CGNAT are query-only | QUIC/UDP present but peers never serve |
| Bootstrap | ~3–6 well-known nodes (join only) | 3 relays (join **and** the only storage) |
| Extra discovery | **PEX** (BEP 11) gossips peers over swarm conns | gossipsub present but unused for providers |
| Routing table | persisted to disk, refreshed, dead-node eviction | re-bootstrapped from 3 IPs each start |
| Lookup | `get_peers` → one query returns IP:port | `get_providers` **then** `get_record` (two queries) |

**Non-goal:** adopting KRPC. Its connectionless UDP design is why BitTorrent scales and
traverses NAT, but it is cleartext / unauthenticated. OpenHydra's signed receipts +
ed25519 identity depend on libp2p's authenticated transport, so the path is **hardening
libp2p Kad**, not replacing it.

## 4. Remediation items (priority order)

### R-DHT-1 · Gossipsub provider PEX — *DHT-independent discovery* 🔥 (highest leverage)
Use the **already-wired gossipsub** (`behaviour.rs:43`; `swarm.rs:206` has a TODO for
per-model topics) as a BitTorrent-PEX equivalent: providers publish "I serve model X" on a
per-model (or sharded) topic; consumers subscribe and learn providers **without the DHT**.
Survives total DHT failure; once you reach one peer you learn the rest. Lowest blast radius
(no relay/bootstrap changes), infrastructure exists.
- *Exit test:* with the DHT deliberately broken, a consumer still discovers a live provider
  via gossip and serves a request.

### R-DHT-2 · Server-mode promotion for reachable peers 🔥 (the core architectural fix)
Conditionally promote peers to Kademlia `Server` when reachability is confirmed (§2 gate).
Grows the DHT organically with the network — BitTorrent's actual property. Keep auto-mode;
feed confirmed addresses rather than forcing.
- *Exit test:* a public-IPv6 (or DCUtR-direct) peer joins, AutoNAT/identify confirms an
  external address, it promotes to server, and a third node's lookup is answered by it
  (not just the relays).

### R-DHT-3 · IPv6-first reachability
Advertise and **use** public IPv6 for the DHT (don't skip it — cf. the `F-9 skipping IPv6`
path). IPv6 is usually un-NATed → instant server eligibility with no hole-punching. (This
session's Mac has a public global IPv6 and is a prime server candidate that currently
serves nothing.)

### R-DHT-4 · UPnP/NAT-PMP port mapping
Add `libp2p-upnp` so home-router-NAT nodes can map their listen port and become reachable.
Helps the full/restricted-cone home-router case; **does not** help CGNAT. One tool in the
reachability kit.

### R-DHT-5 · Reliable DCUtR (UDP hole-punch)
Fix the failing DCUtR path (observed: `Handshake timed out` / `Address already in use
(os error 48)` — a client-side port-reuse conflict). DCUtR-direct connectivity is a
reachability source (§2.4) → more servers, and removes the relay hop for data too. Likely
client-side (no bootstrap redeploy).

### R-DHT-6 · Persistent routing table + active maintenance
Cache good nodes to disk and reload on restart (BitTorrent does this); run periodic bucket
refresh / `bootstrap()`; ping + evict dead nodes. Survives churn instead of amplifying it.

### R-DHT-7 · One-query discovery
Store the provider's full connection info (addresses + ed25519 pubkey) **as the provider
record value**, so a single lookup returns everything — drop the `get_providers` →
`get_record` chain. Halves the failure surface and latency; also removes the early-reply /
stale-candidate fragility seen in the consumer.

### R-DHT-8 · Liveness-aware, churn-tolerant records
Evict dead providers fast (disconnect-eviction + 300 s TTL exist; the issue is a *tiny*
node set where stale records dominate — R-DHT-2 dilutes them). Pair with consumer-side
hardening: don't return a single unverified candidate; prefer connected/recently-verified
providers.

### R-DHT-9 · More relays as interim backbone (not the long-term fix)
3 *bootstraps* is correct and BitTorrent-like; the defect is that they're also the only
*servers*. Until R-DHT-2 lands, a handful more well-distributed relays improves redundancy.
Never bootstrap to a single relay.

## 5. Sequencing

```
R-DHT-1 (gossip PEX) ─┬─► R-DHT-2 (server promotion) ─► R-DHT-6/7/8 (harden)
                      │        ▲
R-DHT-3 (IPv6) ───────┘        │  reachability sources
R-DHT-4 (UPnP) ────────────────┤
R-DHT-5 (DCUtR) ───────────────┘
R-DHT-9 (more relays) = interim, parallel
```

R-DHT-1 and R-DHT-2 together would have prevented the exact degradation observed, and
neither touches the relays. R-DHT-3/4/5 are the reachability feeders that make R-DHT-2 reach
more of the network.

## 6. Bootstrap-node impact

Most items are **agent/peer-side** (no relay redeploy): R-DHT-1, -2, -3, -4, -5, -6, -7, -8.
Only items that change the relays' DHT record TTL, relay limits, or DCUtR *coordination*
would need rebuilding + redeploying `openhydra-bootstrap` to the 3 Linodes — none of the
above require that as currently scoped (the DCUtR fix is client-side port-reuse).
