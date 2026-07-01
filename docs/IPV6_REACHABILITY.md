# IPv6 Reachability — PCP v6 pinholing + v6-biased DCUtR (design)

**Status:** IMPLEMENTED 2026-07-02 — compile + unit tests only; the v6 network
paths are NOT live-validated (no PCP-capable CPE was reachable; the firewalled-v6
DCUtR pair needs a physical v6↔v6 test).
- **W1** (v6-biased hole-punch) — commit `6f0e422`: auto QUIC-v6 punch attempt
  cap raised 3→8 (a firewall punch is reliable once timed, so a firewalled-v6
  peer isn't shelved after 3 uncoordinated misses); back-off cleared on
  `rebootstrap()`; a `quic_v6_holepunch_dials` counter. The *coordinated* punch
  remains libp2p DCUtR (already present); its v6 effectiveness is still untested.
- **W2** (PCP RFC-6887 v6 firewall pinhole) — commit `1626f4c`: new
  `network/src/pcp.rs` (pure wire codec, unit-tested; best-effort async client;
  opt-in maintainer) wired behind `--pcp-gateway <IP>` (off by default).
  Confirmed external v6 addrs feed AutoNAT → promotion. **Live-exercised
  2026-07-02 against a real CPE (TP-Link Archer C5 v6.8):** the maintainer
  started, derived `[(TCP,4001),(UDP,4001)]` from the listen addrs, sent real
  MAP requests, and handled failure cleanly — the router **refused UDP :5351**
  (no PCP server; definitive, not a timeout). So: client request/failure paths
  live-validated; the **success path (granted pinhole → promotion) still needs
  a PCP-capable CPE.** Product datum: users behind such routers stay
  relay-bound for inbound v6 — the "bounded fraction" measured on real
  hardware. (Manual-pinhole side experiment, same day: a hand-added router
  IPv6-firewall rule for the node's stable v6 :4001 was reachable from the
  public internet — netcup TCP-connected through it — but promotion did NOT
  follow passively: AutoNAT only probes *observed* candidates, and peers
  observe the node's rotating privacy v6, never the stable pinholed one. A
  `--advertise-addr`-style operator flag was rejected as product-contrary —
  real users won't manually pinhole. Refinement candidates: shorter PCP retry
  on transient failures (currently ~1h after a refusal), and advertising the
  stable (non-temporary) v6 as a candidate when a pinhole mechanism confirms.)

## The finding (2026-07-01, live-verified)

A globally-routable **public IPv6 address ≠ inbound-reachable.** Consumer-ISP CPEs
run a **default-deny stateful IPv6 firewall** that drops *unsolicited* inbound,
so AutoNAT v2's v6 dial-back fails and the node stays Private / relay-bound —
**even with a public v6.**

- Home Mac (ACT, `2406:7400:56:7e7::e4c6`): `netcup → :4001` **BLOCKED** (macOS
  firewall OFF; node listening). `consumer.log`: 0 direct addrs, 3 relay
  reservations, no promotion.
- Friend's Mac (Airtel, `2401:4900:…`): same — v4 **and** v6 inbound blocked.
- So task #18 ("promote on public v6") **failed in practice**; tasks #20/#21
  (AutoNAT dial-back failures / require-Public-to-promote) followed for this
  reason. Neither ACT nor Airtel allows unsolicited inbound v6 by default.

Conclusion: lots of globally-routable v6 sits behind firewalls. Two distinct ways
to use it, for two distinct goals.

## Two goals (different mechanisms)

### Goal A — direct connection between two specific peers (escape the relay)
A stateful firewall allows inbound that matches an **established outbound flow**.
Two firewalled-v6 peers, coordinated over a relay, fire **simultaneous outbound
QUIC** at each other's `v6:4001`; each opens its own firewall's state and the
packets cross → direct connection. This is **DCUtR**, and it does **not** require
promotion (AutoNAT may stay Private).

**Why v6 is the easy case:** no port translation. v4 CGNAT defeats hole-punching
because symmetric NAT remaps the source port per destination, so the port the
peer was told isn't the one used. On v6 the firewall **doesn't remap** — the
`(addr, port)` learned via Identify *is* reachable once firewall state opens. So
v6 firewall-punch over QUIC is reliable where v4 CGNAT isn't.

### Goal B — promote the node to a public relay / DHT server
A relay/server must accept inbound from **arbitrary** peers with **no prior
outbound flow** — firewall-punching (pairwise, coordinated) cannot do this. You
must actually **open the CPE firewall**:
- **PCP — Port Control Protocol (RFC 6887):** the IPv6-era successor to
  NAT-PMP/UPnP that can request an **inbound v6 firewall pinhole** (not just a NAT
  mapping). Where the CPE runs a PCP server, the agent opens `:4001` inbound on
  its v6 → AutoNAT confirms → promote. Automatic.
- Fallbacks: UPnP IGDv2 `WANIPv6FirewallControl`; manual per-device pinhole.

## Current code state (investigated 2026-07-01)

**v6 *is* preferred and attempted today** — but only as an *uncoordinated* eager
dial, not the *timing-synchronized* DCUtR punch, and with an aggressive backoff:

- `connection_tier()` (`event_loop.rs:53`) prefers
  `direct_quic_v6 > direct_quic_v4 > direct_tcp_* > relay`.
- **Fix 2/4** (`event_loop.rs:~1937`): on a peer's Identify, caches its
  `/quic + /ip6/` (non-circuit) listen addrs and, if no QUIC-direct connection
  exists, **auto-dials them** (`auto_quic_holepunch_dial`). Gated on `ipv6_capable`
  (F-9 startup probe); skips bootstrap relays (F7).
- **Limitations for the both-firewalled case:**
  1. The auto-dial is **independent per peer**, not synchronized over the relay —
     a stateful-firewall punch needs both sides to open state ~simultaneously, so
     this relies on coincidental timing.
  2. It **backs off after 3 attempts** (`MAX_QUIC_HOLEPUNCH_ATTEMPTS`), so it gives
     up before a properly-timed punch can land.
  3. The *coordinated* punch is libp2p's **DCUtR behaviour** (present, eager
     config) — but its **v6 effectiveness for two firewalled peers is unverified**
     (never live-tested; today's relay fallback was v4-only because GPU3 is
     v4-only — no v6 path existed to punch).
- **No PCP / v6 firewall pinholing at all** — R-DHT-4 (UPnP/NAT-PMP) is **v4-only**.

## Proposed work

### W1 — Coordinated, v6-biased DCUtR + live validation (Goal A)
- Ensure libp2p **DCUtR fires on v6 paths** (relay-coordinated, synchronized), not
  just the eager auto-dial; surface v6 vs v4 punch outcomes in the DCUtR counters.
- **Relax the backoff for v6:** a firewall-punch is reliable once correctly timed,
  so don't permanently shelve a v6 peer after 3 uncoordinated misses — prefer a
  coordinated retry.
- **Live test:** two firewalled-v6 peers (e.g. Mac + Asus on home v6, GPU3 can't —
  v4-only) coordinated by a relay → expect `path="direct" … direct_quic_v6`.

### W2 — PCP v6 firewall pinholing (Goal B) — the v6 sibling of R-DHT-4
- Add a PCP (RFC 6887) client: on startup/`rebootstrap`, request an inbound v6
  pinhole for the listen port from the CPE (gateway addr); renew before lease
  expiry (mirror the UPnP re-assert ticker).
- On success → advertise the direct v6 addr + let AutoNAT confirm → promotion.
- **Pin a stable v6** (disable macOS temporary/privacy addresses, or set static),
  else the pinhole and advertised address drift apart.
- Fallback to UPnP IGDv2 `WANIPv6FirewallControl` where PCP is absent.

## Framing / priority
- **W1 first** — highest ROI, fully automatic, no router config; reclaims direct
  inference for v6↔v6 pairs without needing promotion.
- **W2 second** — turns firewalled-v6 consumer nodes into relays/servers where the
  CPE supports PCP; a capacity multiplier at scale.
- The genuinely-stuck node (v4-CGNAT **and** v6-firewalled **and** no PCP/router
  access) remains relay-only — a real but bounded fraction.

See also: [`NETWORK_CHANGE_RESILIENCE.md`](NETWORK_CHANGE_RESILIENCE.md) (the
`rebootstrap` hook is the natural place to (re)assert the PCP pinhole and re-probe
AutoNAT after a roam).
