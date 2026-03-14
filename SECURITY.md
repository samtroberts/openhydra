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
- Affected components (`peer/`, `coordinator/`, `dht/`, etc.)
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

Understanding OpenHydra's trust model helps scope what we consider in-scope vulnerabilities:

### In-scope

- **Coordinator API** — authentication bypass, rate-limit evasion, injection attacks, information disclosure via response headers
- **Peer gRPC service** — unauthenticated remote code execution, TLS bypass, KV cache poisoning, model-output manipulation
- **DHT bootstrap** — Sybil attacks, peer table poisoning, geo-challenge bypass
- **Cryptography** — Ed25519 signature bypass, AES-GCM nonce reuse, key material disclosure
- **HYDRA token economy** — double-spend, unauthorized minting/burning, channel hijacking
- **Verification system** — Mystery Shopper bypass, collusion between malicious peers to evade auditing
- **Dependency vulnerabilities** — critical CVEs in `grpcio`, `cryptography`, `torch`, `transformers`

### Out-of-scope

- Volumetric DDoS (handled at the Cloudflare / Linode network edge)
- Issues only reproducible with physical access to the server
- Social engineering of maintainers
- Security issues in the toy/deterministic model backend (not intended for production inference)
- Self-XSS in the desktop application
- Issues requiring a malicious local user on the same machine

---

## Cryptographic Primitives

OpenHydra uses the following cryptographic constructions:

| Use | Primitive | Implementation |
|-----|-----------|----------------|
| Peer identity | Ed25519 | `cryptography` (PyCA) |
| Transport encryption | TLS 1.3 / mTLS | gRPC built-in |
| Per-hop activation encryption | X25519 + AES-256-GCM | `cryptography` (PyCA) |
| DHT geo-challenge | Ed25519 signature over nonce | `cryptography` (PyCA) |
| HYDRA token integrity | HMAC-SHA256 (mock mode) | `hmac` (stdlib) |

Any vulnerability in these constructions or their usage should be reported privately.

---

## Known Limitations

The following are **intentional design decisions**, not vulnerabilities:

- The HYDRA ledger bridge runs in **mock mode** by default. No real on-chain settlement occurs until the Solidity contract is deployed and wired. See `coordinator/ledger_bridge.py`.
- In `dev` deployment profile, API key authentication is **disabled by default**. Enable it with `--api-key` for any internet-facing deployment.
- The speculative decoding draft model is a **toy model**. Predictions are deterministic and do not reflect real model capability.

---

## Hall of Fame

We gratefully acknowledge responsible reporters:

| Researcher | Issue | Date |
|------------|-------|------|
| _(none yet — be the first!)_ | — | — |
