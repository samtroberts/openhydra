# OpenHydra — Post-Quantum Cryptography (PQC) Implementation Plan

> Status: DRAFT · Created 2026-06-27 · Owner: Sam Roberts
> Companion: `docs/PROTOCOL_IMPLEMENTATION_PLAN.md` (mirrors its milestone/exit-test style + crate boundaries), `docs/LAUNCH_PLAN.md` §8 (crypto posture this plan expands).
> Scope: a standalone, code-grounded plan for migrating OpenHydra's cryptography to a post-quantum-secure, **crypto-agile** posture. Decisions were captured in LAUNCH_PLAN §8; this consolidates them into trackable milestones against the actual code.

---

## ⚑ Threat model & direction (why, and how urgent)

Two quantum threats with **very different urgency** — this drives the entire sequencing:

1. **Harvest-now-decrypt-later (confidentiality)** — an adversary records ciphertext today, decrypts it once a CRQC exists. Hits **key agreement** (X25519). **Retroactive → the only time-sensitive item.**
2. **Signature forgery (authentication)** — identity, receipts, DHT records. A forged signature is only useful *at attack time*, not retroactively. **Not retroactive → migrate later**, EXCEPT long-lived signing keys (release/attestation roots) which must be PQC *from inception* because they can't be cheaply rotated (Google's 2029-migration logic).
3. **Symmetric / hashes** — ChaCha20-Poly1305 (Noise), AES (QUIC TLS1.3), SHA-256 (`sha2`), TOPLOC hash. Grover only halves strength → 256-bit is fine. **No action.**

**Governing principle: crypto-agility *first*.** The highest-value near-term work is making algorithms swappable, so every later migration is a version bump, not a hard fork. **Hybrid** (classical + PQC) everywhere, never pure-PQC (defends against PQC-scheme breaks + keeps interop).

---

## 0. Current crypto inventory (grounded in code)

| Surface | Algorithm today | Where (file) | Quantum-vulnerable? | Agility hook present? |
|---|---|---|---|---|
| **Peer identity key** | Ed25519 (`ed25519-dalek` 2) | `network/src/identity.rs:66-87` | sig (auth) | libp2p PeerId is Ed25519-derived — PQC identity blocked on upstream libp2p key-type support |
| **QUIC transport KEX** | X25519 via rustls TLS 1.3 | `network/patched-deps/libp2p-quic/src/config.rs:94-98` (`libp2p_tls::make_{client,server}_config`) | **KEX (HNDL) — retroactive** | patched crate exists = natural patch site |
| **TCP transport KEX** | X25519 (Noise XX, `snow` 0.9.6) | `network/src/transport.rs:14`, `swarm.rs:88` | **KEX (HNDL)** | `snow` has no PQ-Noise → gap |
| **DHT record signature** | Ed25519, CBOR-canonical | `network/src/dht.rs:97-164` (`public_key` hex + `signature` base64 + `ed_pk.verify`) | sig (auth) | ✅ struct has `public_key`/`signature` string fields → add `alg` |
| **Co-signed receipts** | Ed25519 (`ed25519-dalek` 2) | `protocol/src/receipts.rs:33-35` (`RECEIPT_DOMAIN="openhydra/receipt/v1"`) | sig (auth) | ✅ domain string is `/v1` — version anchor; BUT layout hardcodes 64-byte sigs (`RECEIPT_BLOB_FIXED`, `:130`) |
| **Generic sign() (settle path)** | Ed25519 via libp2p identity | `network/src/handle.rs:69-70` | sig (auth) | inherits identity key |
| **Symmetric / AEAD / hash** | ChaCha20-Poly1305, AES-GCM, SHA-256 (`sha2` 0.10) | transport + `protocol` | NO (Grover-safe at 256-bit) | n/a |

**Resolved dependency versions (Cargo.lock):** `rustls 0.23.40` (✅ ≥0.23.27 — *supports* `X25519MLKEM768`), `ring 0.17.14` (**the active rustls provider — NO ML-KEM**), **no `aws-lc-rs`**, `libp2p-tls 0.5.0`, `libp2p-quic 0.11.1` (patched, vendored), `quinn 0.11.9`, `snow 0.9.6`, `x25519-dalek 2.0.1`. No PQC crates present. No `zeroize`/`secrecy`.

**Key finding (resolves the LAUNCH_PLAN §4.5 spike):** PQ key exchange on QUIC is **NOT a config flag** — the rustls provider in use is `ring` (no ML-KEM), and `libp2p-tls` fixes the kx-group list for the libp2p TLS spec. It requires (a) switching rustls to the **`aws-lc-rs`** provider and (b) adding `X25519MLKEM768` to the kx groups in the vendored `libp2p-quic`/`libp2p-tls` path. The existing patched crate is the natural home.

---

## 1. Decisions (made)

1. **Crypto-agility before scale** — algorithm discriminants on every signed/exchanged artifact, landed *before* the wire format is public (alpha gate).
2. **Hybrid, never pure** — X25519+ML-KEM for KEX; Ed25519+ML-DSA for the signatures that migrate.
3. **KEX now** (HNDL is the only retroactive risk); **ephemeral signatures later** (~2029 window; ML-DSA's ~2.4 KB sigs would bloat the gossip/receipt path).
4. **Long-lived signing keys are PQC from inception** — the R3 release/update-signing key and any attestation root are hybrid from day one.
5. **Symmetric/hash unchanged** (256-bit Grover-safe).
6. **PQC libraries:** ML-KEM via the `aws-lc-rs` rustls provider (KEX); ML-DSA via RustCrypto `ml-dsa` or `pqcrypto-dilithium` (signatures). All additive, feature-gated.

---

## 2. Workstreams

- **WS-PQC-AGILITY** — algorithm registry + versioned wire formats + key hygiene. (`protocol` + `network`)
- **WS-PQC-KEX** — hybrid key exchange on the transports. (`network`, vendored `libp2p-quic`/`libp2p-tls`)
- **WS-PQC-SIG** — hybrid signatures for long-lived keys (now) + ephemeral keys (deferred). (`protocol` + `agent`/R3)

Testing discipline mirrors the protocol plan: **golden vectors first, then Rust**; every milestone has an interop/negotiation exit test against a non-PQC peer (hybrid must degrade gracefully).

---

## 3. Milestones

### Phase 0 — Crypto-agility (foundation · **alpha gate**)

**PQC0.1 — Algorithm registry + versioned wire formats** · WS-PQC-AGILITY · Rust · ~1.5w · ✅ DONE (2026-06-27) — `protocol::crypto_agility` registry (`SigAlg`/`KexAlg`/`AlgError`, stable discriminants, unknown rejected); receipts **v2** (alg bound into preimage + length-prefixed sigs + `UnsupportedAlg`); DHT `PeerRecord` **v3** (`sig_alg` field bound into `canonical_bytes`, sign sets it, verify rejects unknown/unimplemented) — capability records are folded into `PeerRecord`, so covered. Protocol 80 / network 196 / agent 57 tests green. **Remaining sub-item:** the agent's over-the-wire receipt-*request* framing (`agent/src/receipt.rs` `encode_request`/`REQ_FIXED`) still uses the fixed Ed25519 layout (defaults via `payload_from_bytes`); fold the alg byte in for full self-description.
- Define `enum SigAlg { Ed25519, MlDsa65, HybridEd25519MlDsa65 }` and `enum KexAlg { X25519, X25519MlKem768 }` in `protocol` with a stable u8/varint wire discriminant.
- **Receipts (`protocol/src/receipts.rs`):** introduce `RECEIPT_DOMAIN="openhydra/receipt/v2"` carrying a 1-byte `SigAlg` + **length-prefixed** signatures (today `RECEIPT_BLOB_FIXED` hardcodes 64+64 — Ed25519-only). Keep a v1 reader for back-compat.
- **DHT records (`network/src/dht.rs`):** add an `alg` field to the signed struct alongside `public_key`/`signature`; verifier dispatches on it.
- **Capability records (M1.2) + identity-record paths:** same discriminant.
- **Exit:** golden vectors for v1 *and* v2 round-trip; a record with an unknown `alg` is rejected cleanly (not panicked); a v1 receipt still verifies under the v2 reader; property tests over the discriminant.

**PQC0.2 — Signing-key hygiene** · WS-PQC-AGILITY · Rust · ~0.5w · ✅ DONE (2026-06-27) — `zeroize` scrubs the transient secret-byte/hex copies in `network/src/identity.rs` (load + generate; the live `SigningKey`/`SecretKey` already zeroize on drop). Agent `harden_process()` (`agent/src/hardening.rs`, called first in `main`) disables core dumps (`RLIMIT_CORE=0`) + best-effort `mlockall(MCL_CURRENT|MCL_FUTURE)` so no secret reaches disk via core file / swap — unix-gated, never fails startup. Workspace green. **Note:** `secrecy`-wrapping was *not* added — the keys are already opaque libp2p/dalek types (no accidental Debug/serialize of the secret); `secrecy` becomes relevant for the future BYOK key vault. Residual transient copies (the `serde_json::Value` hex in generate) are covered by the process-level core-dump/swap hardening, not individually scrubbed.
- Wrap Ed25519 secret material (`network/src/identity.rs`, receipt signing) in `zeroize`/`secrecy`; `mlock` the pages (`region`/`memsec`); `MADV_DONTDUMP`; never log. (Shared with LAUNCH_PLAN §4.5 + the future BYOK vault.)
- **Exit:** secret bytes are zeroed on drop; not present in a core dump; not swappable.

### Phase 1 — Hybrid KEX / HNDL (confidentiality · **the retroactive-risk item · v1 gate**)

**PQC1.1 — Hybrid `X25519MLKEM768` on QUIC** · WS-PQC-KEX · Rust · ~2w · ❌ NOT STARTED
- In the vendored `network/patched-deps/libp2p-quic` (+ a vendored/patched `libp2p-tls` if needed): switch the rustls `CryptoProvider` to **`aws-lc-rs`** and add `X25519MLKEM768` to the negotiated kx groups in `make_client_config`/`make_server_config` (`config.rs:94-98`). Add `aws-lc-rs` to the dep graph (drops/duplicates `ring`).
- **Exit:** two agents complete a QUIC handshake negotiating `X25519MLKEM768` (assert via rustls' negotiated-group API / key-log / capture); a PQ agent ↔ a classical agent still connect (hybrid negotiates down to X25519); cross-NAT relay path unaffected; perf delta measured (expect small).

**PQC1.2 — TCP+Noise gap: bound & prefer-QUIC** · WS-PQC-KEX · Rust · ~0.5w · ❌ NOT STARTED
- `snow 0.9.6` has no PQ-Noise → document TCP as classical-only KEX. Add a routing/transport preference that favors QUIC for confidentiality-sensitive traffic (ties to the connection-selection work). PQ-Noise itself is deferred (PQC3.2).
- **Exit:** a documented prefer-QUIC policy/flag; the gap is recorded; sensitive-tier routing prefers QUIC when available.

### Phase 2 — Long-lived signing keys hybrid-from-inception (**ships WITH R3**)

**PQC2.1 — Hybrid release/update-signing key** · WS-PQC-SIG · Rust · ~1w · ❌ NOT STARTED (gated on `PROTOCOL_IMPLEMENTATION_PLAN R3`)
- Generate the R3 release/update-signing key as **hybrid Ed25519+ML-DSA-65** from day one; sign release + auto-update artifacts with both; verifier requires both valid. Do NOT ship a classical-only release key (can't retrofit the installed base).
- **Exit:** a release artifact carries both signatures; the updater rejects it if either fails or if a classical-only artifact is presented.

**PQC2.2 — Attestation-root hybrid (conditional)** · WS-PQC-SIG · Rust · ~0.5w · ❌ NOT STARTED (gated on the attestation/enclave tier)
- If/when the attestation tier is built (see `memory: enclave-attestation-candle`), its root signing key is hybrid from inception.
- **Exit:** attestation chain verifies under hybrid; no classical-only root is ever issued.

### Phase 3 — Ephemeral signature migration (**deferred · ~2029 window**)

**PQC3.1 — Hybrid ML-DSA receipts & DHT records** · WS-PQC-SIG · Rust · ~2w · ❌ DEFERRED
- Using the PQC0.1 v2 layout, add an ML-DSA-65 co-signature alongside Ed25519 on receipts (`protocol`) and DHT records (`network`). Add `ml-dsa`/`pqcrypto`. Measure the size/throughput cost (≈2.4 KB/sig × per-request receipts + gossiped records — the reason this is deferred).
- **Exit:** receipts/records carry hybrid sigs; ledger + DHT verify both; size & gossip-bandwidth impact measured and acceptable.

**PQC3.2 — PQ-Noise on TCP** · WS-PQC-KEX · Rust · ~? · ❌ DEFERRED (blocked on upstream)
- Requires a PQ-capable Noise (post-`snow`) or migrating the TCP path. Blocked on the ecosystem; tracked, not scheduled.

**PQC3.3 — PQC peer identity** · WS-PQC-SIG · Rust · ~? · ❌ DEFERRED (blocked on upstream libp2p)
- A PQC PeerId key type needs libp2p support + a spec direction. Tracked, not scheduled.

---

## 4. Sequencing & critical path

```
PQC0.1 agility ─┬─► PQC1.1 hybrid-KEX-QUIC (v1) ───────────► PQC3.1 hybrid-sigs (2029)
   (alpha)      │                                            PQC3.2 PQ-Noise (blocked)
PQC0.2 hygiene ─┘                                            PQC3.3 PQC identity (blocked)
                 └─► PQC1.2 prefer-QUIC
   R3 packaging ───► PQC2.1 hybrid release key (with R3)  ·  PQC2.2 attestation root (with enclave tier)
```
- **Alpha:** PQC0.1 + PQC0.2 (agility + hygiene — must precede a public wire format).
- **v1:** PQC1.1 (HNDL mitigation) + PQC1.2; PQC2.1 ships *with* R3.
- **Post-v1 / ~2029:** PQC3.x.

---

## 5. Risks & open questions

| Risk / question | Note |
|---|---|
| `aws-lc-rs` provider swap breaks the libp2p-tls handshake | libp2p-tls fixes cipher suites/kx for the spec; verify aws-lc-rs satisfies them. **Spike: prototype PQC1.1 in the vendored crate first.** |
| Duplicate `ring`/`aws-lc-rs` in the graph | acceptable transitionally; aim to consolidate the provider. |
| ML-DSA size bloat on gossip/receipts | the explicit reason PQC3.1 is deferred; measure before committing. |
| PQC1.1 only covers QUIC | TCP+Noise stays classical (PQC1.2 bounds it; PQC3.2 blocked). Prefer-QUIC for sensitive traffic. |
| Upstream gating | PQ-Noise (snow) and PQC identity (libp2p) are not in our control — track upstream. |
| `X25519MLKEM768` is pre-final | treat as experimental; the agility layer (PQC0.1) lets us swap to the standardized id cheaply. |

---

## 6. Immediate next actions
1. **Spike PQC1.1** — in `network/patched-deps/libp2p-quic`, prototype the `aws-lc-rs` provider + `X25519MLKEM768` kx group; confirm two agents negotiate it and still interop with a classical peer. (Converts the resolved "it's a patch not a flag" finding into a working branch.)
2. **PQC0.1** — land the algorithm discriminant + receipt `v2`/DHT `alg` field **before** the wire format is public.
3. **PQC0.2** — key hygiene (shared with LAUNCH_PLAN §4.5).
4. Cross-reference: add a PQC milestone row to `PROTOCOL_IMPLEMENTATION_PLAN.md` so the two plans stop drifting.
