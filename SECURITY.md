# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| `main` branch | ✅ Active development |
| Tagged releases | ✅ Critical fixes backported |
| Older branches | ❌ Not supported |

---

## Reporting a Vulnerability

**Please do not open a public GitHub Issue for security vulnerabilities.**

If you discover a security issue in OpenHydra, report it privately via one of these channels:

| Channel | Address |
|---------|---------|
| Email | sam@openhydra.co |
| GitHub Security Advisories | [Report here](../../security/advisories/new) |

Include in your report:
- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept code if applicable)
- Affected components (`agent/`, `network/`, `protocol/`)
- Any suggested mitigations

We aim to acknowledge reports within **48 hours** and provide an initial assessment within **7 days**.

---

## Disclosure Policy

OpenHydra follows a **coordinated disclosure** model:

1. Reporter submits vulnerability privately
2. OpenHydra team reproduces and assesses severity
3. Patch is developed and reviewed in a private branch
4. Fix is released and a GitHub Security Advisory is published
5. Reporter is credited (unless they prefer anonymity)

We ask that reporters keep the vulnerability confidential until a patch is released (typically within **30 days** for critical issues, **90 days** for lower severity).

---

## Security Model

OpenHydra is a pure-Rust protocol: a single `openhydra-agent` binary that routes,
verifies, and credits inference served by external engines over libp2p. Every byte
received from a peer is attacker-controlled. Understanding this trust model helps
scope what we consider in-scope vulnerabilities.

### In-scope

- **Consumer gateway** (`openhydra-agent serve`) — authentication bypass on `/v1/*`, request smuggling, injection, information disclosure
- **Provider serve path** (`openhydra-agent provide`) — malformed-request handling, resource exhaustion, engine-proxy abuse
- **libp2p swarm / transport** — Noise/TLS downgrade, relay-circuit abuse, DCUtR/AutoNAT manipulation, message-framing attacks on the `/openhydra/*` protocols
- **DHT** — Sybil attacks, routing-table / provider-record poisoning, geo-challenge bypass
- **Receipts & cryptography** — Ed25519 signature bypass or forgery, receipt replay, ledger tampering, identity-key disclosure
- **Dependency vulnerabilities** — critical CVEs in core crates (`libp2p`, `ed25519-dalek`, `rustls`, `axum`, `reqwest`, `redb`)

### Out-of-scope

- Volumetric DDoS (handled at the network edge)
- Issues only reproducible with physical access to the server
- Social engineering of maintainers
- Vulnerabilities in a third-party inference engine itself (report those upstream); the operator chooses what their engine exposes
- Issues requiring a malicious local user on the same machine

---

## Cryptographic Primitives

OpenHydra uses the following cryptographic constructions, all implemented in Rust:

| Use | Primitive | Implementation |
|-----|-----------|----------------|
| Peer identity & receipt signatures | Ed25519 | `libp2p-identity` / `ed25519-dalek` |
| Transport encryption | Noise (TCP) / TLS 1.3 (QUIC) | `libp2p` |
| DHT geo-challenge | Ed25519 signature over a nonce | `ed25519-dalek` |

The node's Ed25519 identity key **never leaves the Rust daemon** — receipts are
co-signed in-process and only the resulting signatures and public keys cross the
wire. Any vulnerability in these constructions or their usage should be reported
privately.

---

## Known Limitations

The following are **intentional design decisions**, not vulnerabilities:

- The CLI gateway (`openhydra-agent serve`) binds loopback (`127.0.0.1:16527`) by
  default and leaves `/v1/*` open unless a key is set. For any non-loopback or
  internet-facing deployment, require a key with `--api-key` (or the
  `OPENHYDRA_API_KEY` env var) and terminate TLS at a reverse proxy. The desktop
  app's gateway (since v0.3.21) always requires its own key, stored in
  `~/.openhydra/gateway.key`.
- Providers read the prompts they serve in plain text; nothing seals a prompt to
  the engine. The CLI gateway's audit replay (on by default, `--audit-replay off`
  to disable) can resend a temperature-0 request for a public model to up to three
  other providers.
- Provider operators are responsible for what their local engine exposes; OpenHydra
  routes by model id and does not sandbox the engine.

---

## Hall of Fame

We gratefully acknowledge responsible reporters:

| Researcher | Issue | Date |
|------------|-------|------|
| _(none yet — be the first!)_ | — | — |
