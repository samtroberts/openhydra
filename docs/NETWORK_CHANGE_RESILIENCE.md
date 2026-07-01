# Network-Change Resilience (design)

**Status:** IMPLEMENTED 2026-07-02 (#42, commit `6f0e422`) — and **LIVE-VALIDATED
2026-07-02** on a real double roam (Asus provider, GTX 1050 + Ollama/tinyllama:
"Don Quixote" ACT WiFi → "Pixel 2" Airtel hotspot → back), observed from a netcup
consumer. Both roams fired the exact designed sequence with **no restart and an
unchanged pinned peer id** (same PID throughout):
`rebootstrap: rebuilding connectivity reason="reactive: interface change"`
(count=1 out, count=2 back) → relay re-reservations (all 3 accepted ~400ms on
return) → `network change (generation N) — re-announcing` (gen 1 and 2). The
bootstrap journal showed the same identity reconnecting from the hotspot within
~90s of the roam. Post-return end-to-end inference recovered immediately
(completion in 2.2s, discover 1ms). Caveat: *sustained serving from the hotspot
itself* was degraded by Airtel CGNAT connection churn (~30s evictions,
reservations wouldn't stick) — a documented network condition (see the CGNAT
notes in memory/benchmarks), not a heal failure; the heal loop kept
re-establishing throughout. The consumer-side zombie-connection masking
(dispatching onto dead pre-roam connections) is a separate, now-observed issue —
candidate follow-up: liveness-gate `proxy_forward` connection selection. Filed after a live roam (Airtel → ACT + laptop sleep) on
2026-07-01 left a long-running `serve` node stranded: stale `::1` relay dials,
flapping listeners, dead reservations, and `no provider` on discovery until a
manual restart.

**What shipped vs this design:** `rebootstrap()` (re-seed Kad + re-dial ALL
bootstrap peers incl. relays + re-request only missing reservations via a
`reserved_circuits` set + clear hole-punch/reversal back-off + bump
`net_generation`); a 5 s `heal_ticker` driving the connectivity watchdog
(sustained 0 connected peers past a 20 s grace) and the debounced (3 s) reactive
trigger with a 30 s cooldown; stale-state expiry on real (non-circuit) listener
close/expiry (`expire_direct_external_addrs` → demote to Kad client); and the
`net_generation` counter surfaced through `NetworkHandle::network_generation()`
so the provider re-announces immediately on change. Deferred from the design
below: OS wake/path signals (`NWPathMonitor`/netlink) and the Prometheus metric
surface (`rebootstrap_count` is tracked in-state + logged, not yet exported).

## Problem

A long-lived node that **roams, sleeps/wakes, or changes interface** does not
self-heal. The low-level pieces are present but one-shot or reactive-per-listener;
there is **no holistic "the network changed → rebuild connectivity *and* refresh
my DHT address" trigger.**

What already works:
- **Wildcard listen** (`/ip4/0.0.0.0`, `/ip6/::`) → libp2p's internal `if-watch`
  auto-binds new interfaces; the *listening* side recovers.
- **Reactive relay-retry** (`ExpiredListenAddr`/`ListenerClosed` → re-request
  reservations, F-5 backoff).
- Kademlia auto-bootstraps on its own slow timer.

What's missing: the **explicit bootstrap sequence** — dial bootstrap peers +
`kademlia.bootstrap()` + acquire relay reservations + trigger AutoNAT — runs
**once, at startup.** After a wholesale change the routing table, relay
reservations, and AutoNAT verdict are all stale and nothing proactively rebuilds
them.

## Principle

**Identity is stable; address is ephemeral.** The Ed25519 keypair / PeerId
(pinned via `--identity`) never changes across a roam — so the heal only ever
refreshes *addresses* under the *same* key. (Running without a pinned identity
defeats this: peer-id churn invalidates every cached route — the "Unexpected peer
ID" thrash.) A network-change event triggers **two coordinated reactions**:

- **(A) re-establish my connectivity** (consumer + provider), and
- **(B) refresh my address in the DHT** (provider only).

## Design

### 1. Trigger detection ("the network changed")
- **Primary (fast):** libp2p `NewListenAddr` for a **non-loopback, non-circuit**
  address → a real interface came up.
- **Secondary (catch-all):** a **connectivity watchdog** every ~15–30 s:
  `degraded = 0 live bootstrap/relay connections OR 0 confirmed relay
  external-addrs OR empty Kademlia routing table`, sustained > N s. Covers
  **wake-from-sleep** (post-wake all connections are dead).
- **Debounce** a burst of `NewListenAddr`/`ListenerClosed` (~2–3 s) into one heal
  so a roam doesn't thrash.
- *(Optional later: OS path/wake signals — macOS `NWPathMonitor`, Linux
  netlink/`logind PrepareForSleep` — for lowest latency. Not needed for v1.)*

### 2. (A) `rebootstrap()` — reusable, called at startup **and** on trigger
Re-dial all configured bootstrap peers · `kademlia.bootstrap()` · re-request
relay reservations · **re-trigger AutoNAT** (reachability can differ per
network). Idempotent; backed-off (reuse the F-5 style) so repeated failures don't
tight-loop.

### 3. (B) `reannounce_now()` — providers only (the address-update half)
Immediately re-run `announce_models()` so the `model_id → peer_id` record's TTL
refreshes and **Identify carries the new relay addresses** to the peers it just
reconnected to. The record **key and peer_id are unchanged** — only the dialable
address re-propagates. **Gate on regained connectivity** (fire after the first
relay reservation is re-accepted); announcing into a dead network is pointless.

### 4. Stale-state expiry
On a **real** (non-circuit) `ListenerClosed`/`ExpiredListenAddr`: drop the
confirmed-external + relay-circuit addresses tied to the old network and stop
advertising them; clear AutoNAT's confirmed-address cache so it re-probes. This
kills the `::1`/stale-relay artifacts.

### Heal sequence
```
trigger → debounce → expire stale addrs → rebootstrap()
       → (on first relay reservation / AutoNAT re-confirm) → reannounce_now()  [providers]
```

## Architecture note

- **(A) lives entirely in `network/src/event_loop.rs`** (it owns
  connections/bootstrap/relay/AutoNAT).
- **(B) needs the agent** — `announce_models()` is in `agent/src/provider.rs`
  (the agent owns engine model-detection). So add a small **network → agent
  "network-changed" notification** over the existing command/event channel; the
  provider run-loop responds by re-announcing. The consumer ignores it.

## Roles
- **Consumer:** (A) only (nothing to announce).
- **Provider:** (A) + (B).
- Both-roles host = two processes, each heals itself.

## Exit criteria / test
- Repeat the roam (or bring an interface down/up): node recovers connectivity
  within **X s**, and — for a provider — its record reflects the new relay within
  **Y s**, with **no manual restart** and an **unchanged peer id**.
- Sleep/wake test.
- Metrics into the #33 Prometheus surface: `rebootstrap_count`,
  `reannounce_count`, `time_to_recover`.

## Operational stopgap (until implemented)
Run under `launchd`/`systemd` with `Restart=on-failure` + a liveness probe, or
restart on a known network change. A crutch — the watchdog is the real fix.
