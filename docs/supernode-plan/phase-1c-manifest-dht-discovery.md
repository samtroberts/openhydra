# Phase 1c — SupernodeManifest + DHT Advertisement + Discovery

> Part of [Phase 1: Adoption MVP](phase-1-adoption-mvp.md) · [Masterplan](MASTERPLAN.md)
> **Status:** 🟡 In progress **Owner:** _unassigned_ **Plan ID:** P1.3 (1c)
> Architecture refs: §3.1–3.3 (manifest, DHT keys, refresh), §4.1 step 2 (candidate discovery)

## 1. Goal
Let supernodes advertise **static capabilities** to the Kademlia DHT and let any router discover "who CAN run model X" — without putting load data in the DHT (§3.4). This is the discovery substrate routing depends on.

## 2. Scope
**In:**
- `SupernodeManifest` record (CBOR, Ed25519-signed) — identity, backend_type, models, hardware, network, trust fields (`integration_level`, `trust_tier`; `binary_hash`/`weights_hash` empty at L1).
- DHT keys: `/openhydra/supernode/{libp2p_peer_id}` (full manifest) + Kademlia provider records per normalized model id (`START_PROVIDING`).
- Publish on startup, on model change, every 5 min TTL refresh; `STOP_PROVIDING` on graceful shutdown.
- Router-side: `discover_supernodes(model_id)` → `GET_PROVIDERS` + local manifest cache (TTL 120s).

**Out (deferred):**
- Load data / real-time probe → [3a](phase-3a-realtime-load.md) (explicitly NOT in DHT).
- Full model-id normalization/alias registry → [2e](phase-2e-autodetect-trust-normalization-sticky.md) (basic lowercase/strip here).
- Region/geo fields used for routing → [4b](phase-4b-geographic-affinity.md) (field carried now, unused).

## 3. Dependencies
**Upstream:** [1a](phase-1a-adapter-and-ollama-bridge.md) (`list_models` feeds the manifest).
**Downstream:** [1d](phase-1d-prompt-routing-streaming.md) (consumes candidate list), [2d](phase-2d-model-hash-registry.md), [2e](phase-2e-autodetect-trust-normalization-sticky.md), [2f](phase-2f-web-dashboard-cli-surfaces.md).
**Code touchpoints:** `network/src/dht.rs` (new `supernode_record_key()` + manifest encoding), `peer/dht_announce.py` (new sibling record, **not** an edit to `Announcement`), `coordinator/discovery_service.py` (reuse `_discover_for_model()`, `_cached_dht_peers()`, ranking/degradation).

## 4. Design & Approach
- `SupernodeManifest` is a **new** record type alongside the existing `PeerRecord` (`network/src/types.rs`), not a field add (per §8.2 reality note).
- Canonical CBOR encoding for deterministic signing; verify signature + freshness (timestamp within TTL) before trusting a discovered manifest.
- Provider-record cardinality: popular models return a bounded Kademlia subset (~20), not all providers — document this; routing must treat the candidate set as a sample, not the universe.
- Reuse the existing `discovery_service.py` machinery rather than building a parallel path.

## 5. Tasks (sub-sub checklist)
- [x] Define `SupernodeManifest` + `ModelCapability` + `HardwareInfo` (Python with CBOR; Rust mirror deferred to deploy).
- [x] `supernode_record_key(peer_id)` + provider-key normalization.
- [x] Sign/verify (Ed25519 over canonical CBOR) + freshness check.
- [x] Publish loop: startup, on-change, 5-min refresh; `START_PROVIDING` per model.
- [x] Graceful shutdown: `STOP_PROVIDING` + tombstone manifest (§7.3).
- [x] `discover_supernodes(model_id)` in `supernode/discovery.py` + 120s manifest cache.
- [x] Tests: round-trip encode/sign/verify; discovery returns providers; stale manifest rejected.

## 6. Files
**Create:** `supernode/manifest.py`, `tests/test_supernode_manifest.py`, `tests/test_supernode_discovery.py`.
**Modify:** `network/src/dht.rs`, `network/src/types.rs`, `peer/dht_announce.py`, `coordinator/discovery_service.py`.

## 7. Risks & Open Questions
- Rust↔Python manifest schema must stay in lockstep (codegen or a shared spec doc).
- DHT PUT latency (1–5s) means a just-started node isn't instantly discoverable — acceptable; mDNS/gossip covers LAN.
- Provider-set truncation could starve a model with many providers of good candidates — mitigated by randomized selection (1d) over whatever subset returns.

## 8. Test & Verification Plan
- Two-node local: node B advertises, node A discovers B for its model.
- Stale/forged manifest rejected (bad signature, expired timestamp).
- Rust unit tests for key derivation + encoding.

## 9. Exit Criteria (Definition of Done)
- [ ] Node B's Ollama models are discoverable from node A by model id via DHT.
- [ ] Manifest signature + freshness verified before use.
- [ ] Graceful shutdown removes provider records.
- [ ] No load/dynamic data present in any DHT record.
