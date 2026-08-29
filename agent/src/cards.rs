//! Agent-side `.openhydra` card glue (M2): assemble a [`Card`] from a locally-detected model + this
//! node's identity, gate it on the share policy, and sign it. The signed [`Card`] type + crypto live
//! in [`openhydra_network::card`]; this module is the thin bridge to the agent's detection, identity,
//! and share-policy so the desktop (which shells out to `openhydra-agent card export`) gets a ready
//! artifact. Import (verify + registry injection) is separate — verify is pure in the network crate.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use openhydra_network::card::{sign_card, verify_card, Capability, Card, PricingMode};
use openhydra_network::types::DiscoveredPeer;

use crate::adapter::{normalize_engine_ref, DetectedModel};

/// The export envelope printed by `openhydra-agent card export`: the signed card plus its compact
/// copy-paste magnet string. JSON on stdout; the desktop parses it back into this.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CardExport {
    pub card: Card,
    pub magnet: String,
}

/// Build an UNSIGNED card from a detected model + this node's OpenHydra id. Pure + unit-testable.
/// Defensively runs `model.engine_ref` through [`normalize_engine_ref`] so a raw llama.cpp `-m` path
/// can never reach `model_id` (belt-and-suspenders with the privacy gate in `sign_card`). Capability
/// context/output are left 0 (not known at detection — hint fields).
pub fn build_card(
    model: &DetectedModel,
    openhydra_peer_id: &str,
    signed_at_ms: u64,
    ttl_secs: u64,
    pricing: PricingMode,
    region: Option<String>,
    swarm_public_key: Option<String>,
) -> Card {
    let expires_at = signed_at_ms.saturating_add(ttl_secs.saturating_mul(1000));
    let mut card = Card::new_unsigned(
        openhydra_peer_id,
        normalize_engine_ref(&model.engine_ref),
        signed_at_ms,
        expires_at,
    );
    card.canonical_id = model.canonical_id.clone();
    card.capability = Capability {
        params: model.params.clone(),
        context_length: 0,
        max_output_tokens: 0,
        modalities: vec!["text".to_string()],
    };
    card.pricing_mode = pricing;
    card.region = region;
    // M4: a swarm binding makes this a private card (the model is served only to a member of this
    // swarm). `sign_card` promotes it to schema 2.
    if let Some(pk) = swarm_public_key.filter(|s| !s.is_empty()) {
        card.swarm_public_key = pk;
    }
    card
}

/// Run the `card export` subcommand: load the identity, detect `model` locally, run the reach gate,
/// then build + sign the card. Two modes: a PUBLIC card (no `--swarm`) requires the model to be
/// Global + consented; a PRIVATE card (`--swarm <pk>`, M4) requires the model to be shared with reach
/// Private and the node to OWN that swarm (`--swarms-dir`), so the card names the swarm its provider
/// gates on. `now_ms` is injected (no hidden clock).
#[allow(clippy::too_many_arguments)]
pub fn run_export(
    model: &str,
    share_policy_file: Option<PathBuf>,
    ttl_secs: u64,
    pricing: PricingMode,
    region: Option<String>,
    identity_path: &Path,
    now_ms: u64,
    swarm_public_key: Option<String>,
    swarms_dir: Option<PathBuf>,
) -> Result<CardExport, String> {
    // Export gate, part 1 — fail-closed without a policy file (checked FIRST so it fails fast and is
    // testable without a live engine). Review fix (MODERATE): previously `None` fell back to
    // `share_all()`, so `card export --model X` with the flag omitted would export a model the user
    // had marked private. A card carries reach, so refuse unless the policy confirms the model's reach.
    let policy_file = share_policy_file.ok_or_else(|| {
        format!(
            "refusing to export {model:?} without --share-policy-file — pass the share-policy file to \
             run the reach/consent gate"
        )
    })?;
    let policy = crate::share_policy::SharePolicy::load(&policy_file)
        .map_err(|e| format!("read share policy {}: {e}", policy_file.display()))?;

    let id = openhydra_network::identity::Identity::load_or_create(identity_path)
        .map_err(|e| format!("load identity: {e}"))?;

    // Detect the model on a local engine (fast localhost probes; no swarm).
    let engines = crate::detect::detect_engines();
    let detected = engines
        .iter()
        .flat_map(|e| &e.models)
        .find(|m| m.engine_ref == model)
        .ok_or_else(|| {
            format!("model {model:?} is not detected on any local engine — start the engine that serves it, then retry")
        })?;

    // Export gate, part 2 — reach-dependent:
    let swarm_binding = match swarm_public_key.filter(|s| !s.is_empty()) {
        // M4 PRIVATE card: bind to a swarm. The model must be shared + Private (not a public model),
        // and we must OWN that swarm — the provider's serve gate only authorizes credentials for
        // swarms it owns, so a card naming a swarm we don't own would be unusable. Verified against
        // the swarms dir.
        Some(pk) => {
            if !policy.is_shared(&detected.engine_ref) {
                return Err(format!("model {model:?} is not shared, so it can't be exported"));
            }
            if policy.scope_of(&detected.engine_ref) != crate::share_policy::Scope::Private {
                return Err(format!(
                    "model {model:?} is not Private — a swarm-bound card is for a private model; set \
                     its reach to Private first (or export a public card without --swarm)"
                ));
            }
            let dir = swarms_dir.ok_or_else(|| {
                "a swarm-bound card needs --swarms-dir to confirm you own the swarm".to_string()
            })?;
            match crate::swarms::read_swarm(&dir, &pk)? {
                Some(rec) if rec.role == crate::swarms::SwarmRole::Owner => {}
                _ => {
                    return Err(format!(
                        "you don't own swarm {pk} — only a swarm's owner can issue a card for it"
                    ))
                }
            }
            Some(pk)
        }
        // PUBLIC card: only a globally-shared, consented model may become one.
        None => {
            if !policy.announce_globally(&detected.engine_ref) {
                return Err(format!(
                    "model {model:?} is not shared to the global network (or its global-publish consent \
                     is missing) — set the model's reach to Global, or pass --swarm to export a private card"
                ));
            }
            None
        }
    };

    let card = build_card(detected, &id.openhydra_peer_id, now_ms, ttl_secs, pricing, region, swarm_binding);
    let signed = sign_card(card, &id.keypair).map_err(|e| e.to_string())?;
    let magnet = signed.to_magnet().map_err(|e| e.to_string())?;
    Ok(CardExport { card: signed, magnet })
}

/// Parse a card from either a magnet string (`openhydra:card:...`) or JSON, then fully verify it
/// (signature, key↔peer-id binding, privacy, schema, expiry). The shared parse+verify entry point for
/// `card verify` and, later, import. `now_ms` is injected.
pub fn parse_and_verify(
    input: &str,
    now_ms: u64,
) -> Result<openhydra_network::card::VerifiedCard, String> {
    let trimmed = input.trim();
    let card = if trimmed.starts_with("openhydra:card:") {
        Card::from_magnet(trimmed).map_err(|e| e.to_string())?
    } else {
        Card::from_json(trimmed).map_err(|e| e.to_string())?
    };
    verify_card(&card, now_ms).map_err(|e| e.to_string())
}

// ── import store: verified cards routed by peer id without live discovery ──

/// Map a verified card to the consumer's discovery record. Host/port are left empty on purpose — the
/// dial resolves the current address by `libp2p_peer_id` (the card's whole reason to exist), so the
/// only fields that matter are the ids, the model, and the capability hints used for ranking.
pub fn card_to_discovered(card: &Card) -> DiscoveredPeer {
    DiscoveredPeer {
        peer_id: card.openhydra_peer_id.clone(),
        libp2p_peer_id: card.libp2p_peer_id.clone(),
        model_id: card.model_id.clone(),
        canonical_model_id: card.canonical_id.clone(),
        context_length: card.capability.context_length,
        max_output_tokens: card.capability.max_output_tokens,
        // M4: a private card carries its swarm binding through to routing, so the consumer presents
        // the matching credential (and only on this private route).
        swarm_public_key: card.swarm_public_key.clone(),
        ..Default::default()
    }
}

/// Read a cards file (a JSON array of [`Card`]). Missing/empty → `[]`; a parse error is surfaced.
pub fn read_cards_file(path: &Path) -> Result<Vec<Card>, String> {
    match std::fs::read_to_string(path) {
        Ok(s) if s.trim().is_empty() => Ok(Vec::new()),
        Ok(s) => serde_json::from_str(&s).map_err(|e| format!("parse cards file: {e}")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("read cards file: {e}")),
    }
}

/// Atomically write the cards array (temp + rename, same dir).
fn write_cards_file(path: &Path, cards: &[Card]) -> Result<(), String> {
    let json = serde_json::to_string_pretty(cards).map_err(|e| format!("encode cards: {e}"))?;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename into {}: {e}", path.display()))
}

/// Add a verified card to the file, replacing any existing entry for the same `(libp2p_peer_id,
/// model_id)` (a re-import refreshes rather than duplicates). Atomic.
pub fn add_card(path: &Path, card: &Card) -> Result<(), String> {
    let mut cards = read_cards_file(path)?;
    cards.retain(|c| !(c.libp2p_peer_id == card.libp2p_peer_id && c.model_id == card.model_id));
    cards.push(card.clone());
    write_cards_file(path, &cards)
}

/// Remove every card matching `(libp2p_peer_id, model_id)`. Returns how many were removed.
pub fn remove_card(path: &Path, libp2p_peer_id: &str, model_id: &str) -> Result<usize, String> {
    let mut cards = read_cards_file(path)?;
    let before = cards.len();
    cards.retain(|c| !(c.libp2p_peer_id == libp2p_peer_id && c.model_id == model_id));
    let removed = before - cards.len();
    if removed > 0 {
        write_cards_file(path, &cards)?;
    }
    Ok(removed)
}

/// A hot-reloaded set of imported cards the consumer merges into discovery. Re-reads + re-verifies
/// the file only when its (mtime, size) changes, so a `providers_for` call on the serve path is a
/// cheap `stat` in steady state. Verification happens on load (signature/privacy/schema; a bad or
/// already-expired card is dropped); expiry is re-checked per query so a card that lapses while
/// cached stops being offered.
pub struct CardStore {
    path: PathBuf,
    cache: Mutex<CardCache>,
}

#[derive(Default)]
struct CardCache {
    /// The raw file content last parsed (`Some("")` for absent/empty; `None` until first load).
    /// Content-based (not `(mtime,size)`), so an equal-size content swap or a metadata() error can't
    /// leave stale cards — a small local file re-read per query is cheap; the expensive per-card
    /// verify still runs only when the content actually changed.
    raw: Option<String>,
    /// Cards that verified at load time (expiry re-checked per query).
    cards: Vec<Card>,
}

impl CardStore {
    pub fn new(path: PathBuf) -> Arc<Self> {
        Arc::new(Self { path, cache: Mutex::new(CardCache::default()) })
    }

    /// Re-read + re-verify iff the file's CONTENT changed. Best-effort: a missing/unreadable file →
    /// empty content → no cards (never errors the serve path). Poison-tolerant lock.
    fn refresh(&self, now_ms: u64) {
        let raw = std::fs::read_to_string(&self.path).unwrap_or_default();
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if c.raw.as_deref() == Some(raw.as_str()) {
            return;
        }
        let parsed: Vec<Card> = if raw.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(&raw).unwrap_or_default()
        };
        c.cards = parsed.into_iter().filter(|card| verify_card(card, now_ms).is_ok()).collect();
        c.raw = Some(raw);
    }

    /// Verified, unexpired card providers for `model_id`, as discovery records to dial by peer id.
    pub fn providers_for(&self, model_id: &str, now_ms: u64) -> Vec<DiscoveredPeer> {
        self.refresh(now_ms);
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.cards
            .iter()
            .filter(|card| card.model_id == model_id && now_ms < card.expires_at)
            .map(card_to_discovered)
            .collect()
    }

    /// All currently-valid (verified, unexpired) cards — for the desktop "imported" list.
    pub fn valid_cards(&self, now_ms: u64) -> Vec<Card> {
        self.refresh(now_ms);
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.cards.iter().filter(|card| now_ms < card.expires_at).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detected(engine_ref: &str, canonical: &str, params: &str) -> DetectedModel {
        DetectedModel {
            engine_ref: engine_ref.to_string(),
            canonical_id: canonical.to_string(),
            family: "qwen3".to_string(),
            params: params.to_string(),
            quant: "int4".to_string(),
            size_bytes: 0,
        }
    }

    #[test]
    fn build_card_maps_fields_and_computes_expiry() {
        let m = detected("qwen3:1.7b", "qwen3/1.7b/int4/deadbeef", "1.7b");
        let c = build_card(&m, "oh_abc", 1_000, 3600, PricingMode::Reciprocal, Some("in".into()), None);
        assert_eq!(c.openhydra_peer_id, "oh_abc");
        assert_eq!(c.model_id, "qwen3:1.7b");
        assert_eq!(c.canonical_id, "qwen3/1.7b/int4/deadbeef");
        assert_eq!(c.capability.params, "1.7b");
        assert_eq!(c.region.as_deref(), Some("in"));
        assert_eq!(c.expires_at, 1_000 + 3600 * 1000);
    }

    #[test]
    fn build_card_with_a_swarm_binding_is_private() {
        let m = detected("qwen3:1.7b", "qwen3/1.7b/int4/deadbeef", "1.7b");
        let pk = "a".repeat(64);
        let c = build_card(&m, "oh_abc", 0, 60, PricingMode::Reciprocal, None, Some(pk.clone()));
        assert_eq!(c.swarm_public_key, pk);
        assert!(c.is_private());
        // No binding → public.
        let pubc = build_card(&m, "oh_abc", 0, 60, PricingMode::Reciprocal, None, None);
        assert!(!pubc.is_private());
    }

    #[test]
    fn build_card_normalizes_a_leaky_llama_cpp_path_ref() {
        // A llama.cpp `-m /home/alice/...gguf` engine_ref must become a clean, path-free handle.
        let m = detected(
            "/home/alice/models/Qwen3-1.7B-Q4_K_M.gguf",
            "qwen3/1.7b/int4/deadbeef",
            "1.7b",
        );
        let c = build_card(&m, "oh_abc", 0, 60, PricingMode::Reciprocal, None, None);
        assert_eq!(c.model_id, "Qwen3-1.7B-Q4_K_M");
        assert!(!c.model_id.contains('/') && !c.model_id.to_ascii_lowercase().contains(".gguf"));
        // The card would pass the sign-side privacy gate.
        assert!(openhydra_network::card::card_is_privacy_safe(&c).is_ok());
    }

    // ── import store ──

    fn an_identity() -> openhydra_network::identity::Identity {
        // A throwaway ed25519 identity (the temp key file is gone after this call; the keypair lives
        // in the returned value).
        let dir = tempfile::tempdir().unwrap();
        openhydra_network::identity::Identity::load_or_create(&dir.path().join("id.key")).unwrap()
    }

    fn signed_card(id: &openhydra_network::identity::Identity, model_id: &str, expires_at: u64) -> Card {
        let c = Card::new_unsigned(id.openhydra_peer_id.clone(), model_id, 1_000, expires_at);
        sign_card(c, &id.keypair).unwrap()
    }

    #[test]
    fn card_to_discovered_maps_ids_and_model() {
        let id = an_identity();
        let c = signed_card(&id, "qwen3:1.7b", 9_999_999_999_999);
        let d = card_to_discovered(&c);
        assert_eq!(d.libp2p_peer_id, c.libp2p_peer_id, "the dial target");
        assert_eq!(d.peer_id, c.openhydra_peer_id, "the reputation/receipt key");
        assert_eq!(d.model_id, "qwen3:1.7b");
    }

    #[test]
    fn add_card_dedups_by_peer_and_model_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let id = an_identity();
        let c1 = signed_card(&id, "qwen3:1.7b", 9_999_999_999_999);
        add_card(&path, &c1).unwrap();
        add_card(&path, &c1).unwrap(); // same (peer, model) → replace, not duplicate
        let back = read_cards_file(&path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0], c1);
        // a different model from the same provider is a distinct entry
        add_card(&path, &signed_card(&id, "qwen2.5:0.5b", 9_999_999_999_999)).unwrap();
        assert_eq!(read_cards_file(&path).unwrap().len(), 2);
    }

    #[test]
    fn store_providers_for_filters_by_model_and_expiry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let id = an_identity();
        add_card(&path, &signed_card(&id, "qwen3:1.7b", 5_000)).unwrap();
        let store = CardStore::new(path);
        let p = store.providers_for("qwen3:1.7b", 4_000);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].libp2p_peer_id, id.libp2p_peer_id.to_string());
        assert!(store.providers_for("other:model", 4_000).is_empty(), "wrong model → none");
        assert!(store.providers_for("qwen3:1.7b", 5_000).is_empty(), "at expiry → none");
    }

    #[test]
    fn store_hot_reloads_on_add_and_remove() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let id = an_identity();
        let store = CardStore::new(path.clone());
        assert!(store.providers_for("qwen3:1.7b", 1_500).is_empty(), "empty at first");
        add_card(&path, &signed_card(&id, "qwen3:1.7b", 9_999_999_999_999)).unwrap();
        assert_eq!(store.providers_for("qwen3:1.7b", 1_500).len(), 1, "picks up the import");
        assert_eq!(remove_card(&path, &id.libp2p_peer_id.to_string(), "qwen3:1.7b").unwrap(), 1);
        assert!(store.providers_for("qwen3:1.7b", 1_500).is_empty(), "picks up the removal");
    }

    #[test]
    fn store_drops_an_unverifiable_card_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let id = an_identity();
        let mut tampered = signed_card(&id, "qwen3:1.7b", 9_999_999_999_999);
        tampered.model_id = "qwen3:8b".into(); // breaks the signature
        std::fs::write(&path, serde_json::to_string(&vec![tampered]).unwrap()).unwrap();
        let store = CardStore::new(path);
        assert!(
            store.providers_for("qwen3:8b", 1_500).is_empty(),
            "a card whose signature doesn't verify must never be served"
        );
    }

    #[test]
    fn export_refuses_without_a_share_policy_file() {
        // Review fix (MODERATE): fail-closed — no --share-policy-file must NOT fall back to share-all
        // (which would export a private model). Checked before detection, so no live engine needed.
        let dir = tempfile::tempdir().unwrap();
        let err = run_export("qwen3:1.7b", None, 60, PricingMode::Reciprocal, None, &dir.path().join("id.key"), 1_000, None, None)
            .unwrap_err();
        assert!(err.contains("--share-policy-file"), "got: {err}");
    }

    #[test]
    fn store_reloads_on_an_equal_size_content_swap() {
        // Review fix (LOW): content-based invalidation must catch a change that keeps the file's
        // (mtime, size) identical — e.g. swapping one card for another of the same byte length.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let id = an_identity();
        // Two models with the SAME-length engine handle → same-length serialized files.
        let a = signed_card(&id, "modelaaaa", 9_999_999_999_999);
        let b = signed_card(&id, "modelbbbb", 9_999_999_999_999);
        std::fs::write(&path, serde_json::to_string(&vec![a]).unwrap()).unwrap();
        let store = CardStore::new(path.clone());
        assert_eq!(store.providers_for("modelaaaa", 1_500).len(), 1);
        // Overwrite with the equal-length b (same file size); content-based refresh must notice.
        std::fs::write(&path, serde_json::to_string(&vec![b]).unwrap()).unwrap();
        assert!(store.providers_for("modelaaaa", 1_500).is_empty(), "old card must be gone");
        assert_eq!(store.providers_for("modelbbbb", 1_500).len(), 1, "new card must be served");
    }
}
