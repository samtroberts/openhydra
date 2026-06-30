# OpenHydra — Alpha & v1 Launch Plan

> Status: DRAFT · Last updated: 2026-06-27 · Owner: Sam Roberts
> Companion docs: `docs/PROTOCOL_IMPLEMENTATION_PLAN.md` (engineering milestones), `docs/ENGINE_COMPATIBILITY.md`, `docs/PEER_CONNECTIVITY.md`.

This plan covers what must be in place to ship a private **Alpha** and a public **v1** of OpenHydra — the pure-Rust, BYO-engine, decentralized inference protocol. It is a productization/operationalization/de-risking plan, **not** a protocol plan: the core protocol is already built and live-validated.

---

## 0. The one-paragraph reality

The hard part is done. M1–M3 (canonical model-id + equivalence, capability records, resolve→rank→route router, co-signed receipts, reputation/TOPLOC verification, redb ledger, the engine adapters covering Ollama/vLLM/LM-Studio/Exo/llama.cpp, and a feature-complete OpenAI-SDK gateway) are implemented and validated over **both direct and cross-NAT-relay** paths on a real multi-ISP fleet. What remains for launch is mostly **not** core-protocol work: it is **packaging & onboarding, reliability-under-churn polish, the safety/legal floor, security hygiene, operations, and bootstrapping a two-sided network.**

---

## 1. Scope & the foundational decision

### 1.1 Launch is FREE; billing is deferred
**Decision:** Alpha and v1 ship **free**, with **optional BYOK** (users supply their own Anthropic/OpenAI/Gemini/DeepSeek/GLM/etc. keys) as the only monetization surface. **Custodial billing and provider payouts are post-v1.**

**Why:** the monetization deep-research (2026-06-26) established that a platform holding user balances and forwarding funds to providers *is* a money transmitter — the "we just facilitate payments" exemption was adversarially **refuted**. The compliant path (Stripe Connect / Tipalti + a Merchant-of-Record like Paddle/Lemon Squeezy + escrow + DAC7) is a real integration **and** legal lift. BYOK monetizes with **zero fund custody** (the user's key pays the upstream directly), entirely sidestepping the licensing question, and is also the substrate for the future `openhydra/auto` orchestration tier. So: free + BYOK now, custodial billing later. See `memory: monetization-architecture`.

### 1.2 Definitions
- **Alpha** = private / invite-gated. "It works for friendly users on common paths; rough edges expected." Goal: prove the loop end-to-end with real external operators, gather failure data.
- **v1** = public / supported. "We stand behind it." Stable, documented, monitored, legally reviewed, with a liquidity base.

### 1.3 Explicitly OUT of scope for v1 (deferred / premium)
- Custodial billing, provider payouts, escrow (post-v1; see §11).
- Attestation / confidential-enclave provider tier (the paid **privacy SKU** — SIP / SEV-SNP / TDX / VBS / H100 CC). v1 ships the **verify-and-reputation** trust baseline only. See `memory: enclave-attestation-candle`.
- `openhydra/auto` Fugu-style capability-orchestration. See `memory: sakana-fugu-orchestration`.
- In-process Candle/Burn engine.
- Full PQC signatures (KEX-on-QUIC + crypto-agility tags only; see §8).
- Training/RL-rollout workloads.

---

## 2. Current state (what's already done ✅)

| Area | Status |
|---|---|
| Canonical model-id + equivalence resolver (M1.1) | ✅ |
| Capability records on the wire (M1.2) | ✅ |
| Router resolve→rank→route (M1.3) | ✅ |
| Co-signed receipts (M2.1) | ✅ |
| Reputation + TOPLOC-hash verification (M2.2) | ✅ |
| redb accepted-receipt ledger (M2.3) | ✅ |
| Engine adapters: ollama / openai (=vLLM·LM-Studio·Exo·LocalAI) / llama.cpp (M3.1–3.2) | ✅ |
| OpenAI-SDK gateway: streaming + non-streaming, errors, `/v1/models`, API-key auth, `/health` (R1) | ✅ |
| Pure-Rust cutover, no Python in any role (R2) | ✅ |
| libp2p transport: QUIC+TCP, Kademlia DHT, Relay v2, DCUtR, AutoNAT v2, mDNS, gossipsub PEX | ✅ |
| 4 bootstraps live (US/EU/AP + DE-netcup), dual-stack v4/v6 | ✅ |
| E2E inference validated direct + cross-NAT-relay on multi-ISP fleet | ✅ |

---

## 3. Launch principles

1. **The protocol stays neutral.** Billing, identity, and policy live in an application/control plane *on top*, never baked into the protocol.
2. **Trust is earned and verified, not assumed.** "Trust but verify" is the baseline posture for **every** provider; attestation only raises the floor (post-v1).
3. **No install, no alpha.** Onboarding friction is the primary adoption gate; treat it as a first-class feature.
4. **Free doesn't mean liability-free.** An AUP + abuse path + content controls are launch gates, not niceties.
5. **Build agility before scale.** Crypto-agility and version negotiation go in *before* the wire format and release channel are public.

---

## 4. Readiness checklist by workstream

Tiers: **A** = Alpha gate · **V** = v1 gate · **P** = post-v1. Status: ✅ done · 🔨 partial · ❌ missing.

### 4.1 Core path reliability
- [ ] **A** 🔨 Run the queued cross-NAT permutation test ({Mac, Asus, GPU3, netcup}, every consumer→provider combo) and confirm **provider-can-also-consume** (dual role, likely two processes). *(see `memory: connsel-permutation-test-plan`)*
- [ ] **A** 🔨 Graceful failover: provider dies mid-request → clean error + automatic retry to another provider.
- [ ] **A** 🔨 Relay-circuit-death + CGNAT-eviction: mitigate or **document-and-bound** (known issue) with keepalive + reconnect.
- [ ] **V** ❌ Hedged/speculative dispatch (tail latency + instant dead-provider failover). *(parked enhancement)*
- [ ] **V** ❌ Soak test: sustained load over 24–72h across the fleet; measure error rate, p50/p95/p99 latency, circuit survival.

### 4.2 Onboarding / install (highest-leverage)
- [ ] **A** ❌ Docker **sidecar** image (agent-only, CPU-only, engine-agnostic) — Linux + Mac.
- [ ] **A** ❌ Native binary install, one command, provider + consumer — Mac + Linux.
- [ ] **A** ❌ `docker compose` bundles (agent + Ollama / + llama.cpp) for non-devs with no engine.
- [ ] **A** ❌ Quickstart docs: run a provider, run a consumer, "point your OpenAI client here."
- [ ] **V** ❌ Windows support (native agent; document bridge-NAT caveat for containers).
- [ ] **V** ❌ Signed releases (**hybrid Ed25519+ML-DSA signing key from inception**) + auto-update + protocol-version negotiation.

### 4.3 Safety, legal & content
- [ ] **A** ❌ Acceptable Use Policy + provider **content opt-out** controls + abuse-report contact.
- [ ] **A** ❌ Uncensored/abliterated-model **policy decision** (your sharpest differentiator *and* liability — providers opt in to what they host; jurisdiction-aware).
- [ ] **A** ❌ ToS + privacy policy (you process prompts) + provider agreement — drafted.
- [ ] **V** ❌ Same, **counsel-reviewed** (incl. data-processing, export, CSAM/DMCA handling).

### 4.4 Trust / anti-abuse
- [x] **A** ✅ Verify-and-reputation baseline (receipts / TOPLOC / reputation).
- [ ] **A** 🔨 Rate limiting + basic sybil resistance + abuse handling on the gateway and provider registration.
- [ ] **P** ❌ Attestation/enclave provider tiers (integrity-attested → confidential-enclave). Trust model: two axes (correctness vs confidentiality), routed per workload class. *(see `memory: enclave-attestation-candle`)*

### 4.5 Security & crypto
- [x] **A** ✅ Transport encryption (libp2p Noise/TLS).
- [ ] **A** ❌ Key hygiene: `zeroize` + `secrecy` + `mlock` on signing keys (and future BYOK vault); `MADV_DONTDUMP`; no key logging.
- [ ] **A** ❌ **Crypto-agility version tags** on receipts, DHT records, capability records, handshake selection (cheap; do before the wire format is public). *(= PQC0.1 in `docs/PQC_IMPLEMENTATION_PLAN.md`)*
- [ ] **V** 🔨 PQ key exchange on the QUIC path (rustls ≥0.23.27 + aws-lc-rs provider → X25519MLKEM768); TCP+Noise documented as the classical-only gap.
- [ ] **V** ❌ Security review (`/security-review`) + external crypto audit before public launch.
- [ ] **A** ❌ Pin `--identity` everywhere; document peer-id-churn footgun.

### 4.6 BYOK (the v1 money surface)
- [ ] **V** ❌ Encrypted-at-rest, per-user envelope-encrypted key vault; never logged.
- [ ] **V** ❌ Per-provider translation adapters where APIs differ (Anthropic Messages, Gemini); reuse the `openai` adapter where they don't.
- [ ] **V** ❌ Routing fee/metering on BYOK traffic (no fund custody).

### 4.7 Operations & observability
- [x] **A** ✅ 4 bootstraps live.
- [ ] **A** 🔨 Bootstrap uptime monitoring + alerting.
- [ ] **A** 🔨 Network observability: who's online, error rates, request volume, route mix (direct/relay) via the truthful `proxy_forward dispatch` log + telemetry.
- [ ] **V** ❌ Dashboards + incident runbook + rollback procedure for a bad agent version.
- [ ] **V** ❌ Status page.

### 4.8 Cold-start / liquidity (non-engineering, easy to under-weight)
- [ ] **A** 🔨 3–5 reliable **seed providers** across regions (own fleet + recruited) so consumers actually get served.
- [ ] **A** ❌ Early-provider incentive (reputation now; payout-credit later).
- [ ] **A** ❌ A handful of friendly **seed consumers** (real workloads → real failure data).
- [ ] **V** 🔨 Landing page + model catalog (`openhydra-landing` exists; `/v1/models` populated dynamically).
- [ ] **V** ❌ Provider-recruitment funnel (incl. courting idle-Apple-Silicon operators — the Darkbloom base — with the open, engine-agnostic alternative).

---

## 5. Critical path & sequencing

### Phase A → Alpha (private, invite-gated)
1. **Prove reliability** — permutation test + dual-role confirmation; add failover+retry; bound the relay/CGNAT issues.
2. **R3 packaging** — Docker sidecar + native binaries (Mac/Linux), consumer + provider; quickstart docs.
3. **Safety floor** — AUP, content opt-out, abuse contact, uncensored-model policy.
4. **Security floor** — key hygiene + crypto-agility tags (before the wire format is public).
5. **Liquidity** — 3–5 seed providers across regions + friendly consumers; bootstrap monitoring + basic telemetry.

**Alpha exit = the Go/No-Go gate in §9.1.**

### Phase B → v1 (public, supported)
6. **Reliability hardening** — hedged dispatch, relay/CGNAT mitigations, soak test; Windows support.
7. **Releases** — signed (hybrid-PQC key) + auto-update + version negotiation.
8. **Docs & ops** — full docs site/landing, dashboards, incident runbooks, status page.
9. **Security & legal** — security review + external crypto audit; counsel-reviewed legal.
10. **Trust/anti-sybil hardening** at public scale.
11. **BYOK** — ship the key-vault + per-provider adapters + routing fee as the v1 money surface.

**v1 exit = the Go/No-Go gate in §9.2.**

---

## 6. Trust & verification posture (v1)

Every provider is scored on **two orthogonal axes**, and trust is a **match to the workload**, not a global label:
- **Correctness/integrity** — earned via behavioral verification (TOPLOC, receipts, redundancy) + reputation. Applies to all providers including pure BYOE.
- **Confidentiality** — provided *only* by hardware memory secrecy (attestation tier). Cannot be earned by reputation. Pure BYOE is structurally excluded (external-engine IPC boundary).

v1 routes only **non-private** workload classes (public + free), so the verify-and-reputation baseline is sufficient. The confidential-enclave tier (and the private SKU it unlocks) is post-v1.

---

## 7. Safety, legal & content posture

- **Neutral protocol, opt-in providers.** Providers choose which models/content they host; the network does not filter, but providers can opt out.
- **Uncensored models** are supported technically (just weights on standard engines) and are a differentiator vs closed aggregators — but concentrate content liability. v1 requires: provider opt-in, an AUP, an abuse-report path, and jurisdiction awareness.
- **Prompt data** transits third-party providers in the free tier (no confidentiality guarantee) — this MUST be disclosed in the ToS/privacy policy. Confidentiality is a post-v1 paid tier, not a v1 promise.

---

## 8. Security & crypto posture

- **Now (pre-public):** crypto-agility version tags everywhere; key hygiene (`zeroize`/`secrecy`/`mlock`); pinned identities.
- **v1:** PQ key exchange on QUIC (rustls/aws-lc-rs X25519MLKEM768 — store-now-decrypt-later mitigation); security review + crypto audit.
- **Long-lived keys are PQC from inception** — the **release/update-signing key** (R3) and any future attestation root must be **hybrid Ed25519+ML-DSA** from day one (per Google's 2029 migration logic: long-lived trust anchors can't be retrofitted cheaply). Ephemeral receipt/identity signatures migrate later within the window.
- **TCP+Noise** has no PQ path yet — documented as the classical-only transport; prefer QUIC for sensitive traffic.

---

## 9. Go / No-Go criteria

### 9.1 Alpha gate (all must be true)
- [ ] E2E inference succeeds across every {consumer→provider} permutation in the seed fleet, incl. cross-NAT, with a measured success rate ≥ target (e.g. ≥95% on stable links).
- [ ] Provider-dies-mid-request yields a clean consumer error + retry (no hangs).
- [ ] One-command install works for a non-dev on **Mac and Linux**, both roles.
- [ ] AUP + content opt-out + abuse contact are live.
- [ ] Key hygiene + crypto-agility tags merged.
- [ ] ≥3 seed providers across ≥2 regions online with monitoring.

### 9.2 v1 gate (all must be true, in addition to Alpha)
- [ ] 24–72h soak passes with bounded error rate and documented p50/p95/p99.
- [ ] Windows supported; signed + auto-updating releases with version negotiation.
- [ ] Security review + external crypto audit closed with no criticals.
- [ ] ToS/AUP/privacy/provider agreement counsel-reviewed.
- [ ] Observability dashboards + incident runbook + status page live.
- [ ] BYOK shipped (vault + adapters + routing fee) **or** explicitly deferred with a dated follow-up.
- [ ] Liquidity: provider count + regional coverage sufficient that a typical consumer request is served within target latency.

---

## 10. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Install friction → no adoption | Kills alpha | R3 Docker sidecar + native one-liner; treat onboarding as a feature |
| Cross-NAT reliability under churn (CGNAT, relay deaths) | Bad first impression | Keepalive + reconnect; failover/retry; document bounds; QUIC-preferred |
| Cold-start (no providers ↔ no consumers) | Empty network | Seed own fleet + recruit; early-provider reputation incentive |
| Content/abuse liability (free, uncensored, strangers' GPUs) | Legal exposure | AUP + provider opt-out + abuse path; jurisdiction awareness; counsel before public |
| Wire-format lock-in pre-agility | Expensive migration later | Crypto-agility + version tags **before** public |
| Treating BYOK key vault casually | Catastrophic breach | Envelope encryption + key hygiene; never log; TEE for the strong version later |
| Scope creep into billing/attestation pre-v1 | Slips launch by a year | Hard-defer per §1.3 |

---

## 11. Deferred roadmap (post-v1, sequenced)

1. **Custodial billing** — Stripe Connect/Tipalti payouts (no own MTL) + Paddle/Lemon-Squeezy MoR for VAT + escrow gated on receipts/TOPLOC + DAC7 provider reporting. Two-class providers (anon-free / KYC'd-paid).
2. **Attestation / confidential-enclave tier** — the paid **privacy SKU**: in-process Candle/Burn (Layer A) + per-platform attestation gate (Apple SE+SIP / Windows VBS / Linux SEV-SNP/TDX) (Layer B), "attest-don't-prevent," routed to private workloads only.
3. **`openhydra/auto` orchestration** — Fugu-style capability router over the decentralized open-model pool; top-tier-rate billing, no fee stacking.
4. **Full PQC** — PQ-Noise on TCP; hybrid ML-DSA receipt/identity signatures within the ~2029 window.
5. **Training workloads** — synthetic-data generation, distillation, RL rollouts (inference-bound; receipts = verifiable provenance).

---

## 11.1 Target users & enterprise angle (data-grounded 2026-06-27)

Market data (Menlo Ventures 2025; a16z 2025 CIO survey): ~**80–87% of enterprise inference spend is managed/frontier APIs**, ~13–20% self-hosted on GPUs; open-source workloads **flat/declining at ~13%** (security/compliance/support favor closed). Implications for positioning:

- **Primary open-pool SUPPLY users** = cost-sensitive / high-volume / privacy-sensitive / startup / individual operators — NOT mainstream enterprises (who prefer closed models via managed APIs).
- **Enterprise wedge = the router/FinOps CONSUMER layer, not open-model supply.** OpenHydra + BYOK + `openhydra/auto` lets a company stop over-provisioning (the documented "buy $50k of credits, strand $30k" waste lives in the dominant API bucket) by paying-as-you-go across a pool and downshifting the high-volume tail to cheap open models. This targets the ~80% spend bucket + the 37% of enterprises already running hybrid. **Post-v1**, but it shapes messaging now.
- **Enterprise as PROVIDER** (recover stranded reserved GPU capacity, open models only — reselling frontier-API capacity is ToS-blocked) = a **narrower secondary niche** (self-hosters skew high-volume = less idle). Curated/KYC'd higher-trust provider tier.

See `memory: monetization-architecture` (Enterprise angles) for detail and sources.

## 12. Open decisions

- Alpha access model: fully open vs invite-gated (recommend invite-gated).
- Uncensored-model policy: allow-by-default-with-opt-out vs allowlist (recommend opt-out + jurisdiction notes).
- BYOK in v1 vs first post-v1 increment.
- Default seed-provider regions and target served-latency SLO.
