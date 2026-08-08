//! C7: bootstrap-side provider registry.
//!
//! A bootstrap node already sees every `PROVIDER_ANNOUNCE` gossip message from the providers
//! connected to it (it subscribes to the swarm topic to forward hole-punch / PEX traffic), but
//! it only *forwards* them over the D-sized gossipsub mesh — which does not reliably reach a
//! specific consumer, especially across NATs. That is why passive discovery is partial.
//!
//! This registry has the bootstrap **retain** the verified records it sees, keyed by model, so
//! it can later answer "who serves model X?" authoritatively (the C7 query protocol) instead of
//! hoping a gossip forward lands. The store conveys *self-signed* records — it never vouches for
//! them; the caller must verify authenticity (`dht::pex_record_is_authentic`) before `insert`.
//!
//! Bounded on every axis (models, providers-per-model) and TTL-swept, per the codebase's
//! "bound every map" discipline — a bootstrap must not let untrusted gossip grow its memory.

use crate::types::PeerRecord;
use std::collections::HashMap;

/// Max distinct model ids retained. A provider can announce arbitrary model strings, so this
/// caps memory against a flood of junk model names (each still costs a verified announce).
pub const MAX_MODELS: usize = 8_192;
/// Max providers retained per model.
pub const MAX_PROVIDERS_PER_MODEL: usize = 512;

/// Verified provider records the bootstrap has seen, grouped by model, each stamped with the
/// last time we saw an announce for it (for TTL expiry). Keyed inner by OpenHydra `peer_id`.
#[derive(Debug, Default)]
pub struct ProviderRegistry {
    by_model: HashMap<String, HashMap<String, Entry>>,
    ttl_ms: u64,
}

#[derive(Debug, Clone)]
struct Entry {
    record: PeerRecord,
    last_seen_ms: u64,
}

impl ProviderRegistry {
    /// New registry expiring entries not re-announced within `ttl_ms` (match the DHT record TTL,
    /// ~300 s, so a provider that stops re-announcing ages out).
    pub fn new(ttl_ms: u64) -> Self {
        Self { by_model: HashMap::new(), ttl_ms }
    }

    /// Ingest an **already-authenticity-verified** record (caller ran
    /// `dht::pex_record_is_authentic`). Refreshes `last_seen`; inserts if new. Bounds are
    /// enforced by evicting the stalest entry when a bucket is full, so a burst can't grow the
    /// store past its caps. Ignores records with an empty `model_id` or `peer_id`.
    pub fn insert(&mut self, record: PeerRecord, now_ms: u64) {
        if record.model_id.trim().is_empty() || record.peer_id.trim().is_empty() {
            return;
        }
        // New model bucket: enforce the model cap first (evict the model whose freshest entry is
        // oldest), so an attacker announcing many distinct models can't grow us unbounded.
        if !self.by_model.contains_key(&record.model_id) && self.by_model.len() >= MAX_MODELS {
            if let Some(stalest) = self.stalest_model() {
                self.by_model.remove(&stalest);
            }
        }
        let bucket = self.by_model.entry(record.model_id.clone()).or_default();
        if !bucket.contains_key(&record.peer_id) && bucket.len() >= MAX_PROVIDERS_PER_MODEL {
            if let Some(stalest) = stalest_peer(bucket) {
                bucket.remove(&stalest);
            }
        }
        bucket.insert(record.peer_id.clone(), Entry { record, last_seen_ms: now_ms });
    }

    /// Fresh providers for `model_id` (last seen within `ttl_ms`), newest-announce first. Stale
    /// entries are skipped here even before the reaper runs, so a query never returns an aged
    /// record.
    pub fn providers_for(&self, model_id: &str, now_ms: u64) -> Vec<PeerRecord> {
        let mut out: Vec<&Entry> = match self.by_model.get(model_id) {
            Some(b) => b
                .values()
                .filter(|e| now_ms.saturating_sub(e.last_seen_ms) < self.ttl_ms)
                .collect(),
            None => return Vec::new(),
        };
        out.sort_by(|a, b| b.last_seen_ms.cmp(&a.last_seen_ms));
        out.into_iter().map(|e| e.record.clone()).collect()
    }

    /// Drop entries older than `ttl_ms` (and now-empty model buckets). Returns the number of
    /// provider records removed. Call on a periodic tick.
    pub fn reap(&mut self, now_ms: u64) -> usize {
        let mut removed = 0;
        self.by_model.retain(|_model, bucket| {
            let before = bucket.len();
            bucket.retain(|_pid, e| now_ms.saturating_sub(e.last_seen_ms) < self.ttl_ms);
            removed += before - bucket.len();
            !bucket.is_empty()
        });
        removed
    }

    /// Total provider records across all models (for observability / bounds logging).
    pub fn len(&self) -> usize {
        self.by_model.values().map(HashMap::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.by_model.values().all(HashMap::is_empty)
    }

    /// Distinct model count (for logging).
    pub fn model_count(&self) -> usize {
        self.by_model.len()
    }

    /// The model whose freshest entry is oldest — the eviction victim when at the model cap.
    fn stalest_model(&self) -> Option<String> {
        self.by_model
            .iter()
            .filter_map(|(m, b)| b.values().map(|e| e.last_seen_ms).max().map(|freshest| (m, freshest)))
            .min_by_key(|(_, freshest)| *freshest)
            .map(|(m, _)| m.clone())
    }
}

/// The peer whose entry is oldest in a bucket — the eviction victim at the per-model cap.
fn stalest_peer(bucket: &HashMap<String, Entry>) -> Option<String> {
    bucket
        .iter()
        .min_by_key(|(_, e)| e.last_seen_ms)
        .map(|(pid, _)| pid.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(model: &str, peer: &str) -> PeerRecord {
        PeerRecord {
            peer_id: peer.into(),
            model_id: model.into(),
            libp2p_peer_id: format!("libp2p-{peer}"),
            ..Default::default()
        }
    }

    #[test]
    fn insert_query_and_refresh() {
        let mut r = ProviderRegistry::new(1000);
        r.insert(rec("m1", "p1"), 100);
        r.insert(rec("m1", "p2"), 200);
        r.insert(rec("m2", "p3"), 150);
        // newest-announce first within a model
        let m1: Vec<String> = r.providers_for("m1", 250).into_iter().map(|x| x.peer_id).collect();
        assert_eq!(m1, vec!["p2".to_string(), "p1".to_string()]);
        assert_eq!(r.providers_for("m2", 250).len(), 1);
        assert_eq!(r.providers_for("absent", 250).len(), 0);
        assert_eq!(r.len(), 3);
        // re-announce refreshes last_seen (moves p1 to front)
        r.insert(rec("m1", "p1"), 300);
        let m1b: Vec<String> = r.providers_for("m1", 350).into_iter().map(|x| x.peer_id).collect();
        assert_eq!(m1b, vec!["p1".to_string(), "p2".to_string()]);
        assert_eq!(r.len(), 3); // refresh, not a new row
    }

    #[test]
    fn ttl_expiry_on_query_and_reap() {
        let mut r = ProviderRegistry::new(1000);
        r.insert(rec("m1", "p1"), 100);
        r.insert(rec("m1", "p2"), 900);
        // at now=1100, p1 is 1000ms old (>= ttl) → excluded from query; p2 still fresh.
        let live: Vec<String> = r.providers_for("m1", 1100).into_iter().map(|x| x.peer_id).collect();
        assert_eq!(live, vec!["p2".to_string()]);
        // reap physically drops the stale one.
        assert_eq!(r.reap(1100), 1);
        assert_eq!(r.len(), 1);
        // reaping everything drops the empty model bucket.
        assert_eq!(r.reap(3000), 1);
        assert!(r.is_empty());
        assert_eq!(r.model_count(), 0);
    }

    #[test]
    fn rejects_empty_ids() {
        let mut r = ProviderRegistry::new(1000);
        r.insert(rec("", "p1"), 100);
        r.insert(rec("m1", ""), 100);
        assert!(r.is_empty());
    }

    #[test]
    fn per_model_cap_evicts_stalest() {
        let mut r = ProviderRegistry::new(u64::MAX); // isolate the cap logic from TTL
        for i in 0..MAX_PROVIDERS_PER_MODEL {
            r.insert(rec("m", &format!("p{i}")), 1000 + i as u64);
        }
        assert_eq!(r.providers_for("m", 2_000_000).len(), MAX_PROVIDERS_PER_MODEL);
        // one more (fresher) → evicts the stalest (p0 @ ts 1000), stays at the cap.
        r.insert(rec("m", "pNEW"), 9_000_000);
        assert_eq!(r.providers_for("m", 10_000_000).len(), MAX_PROVIDERS_PER_MODEL);
        let ids: Vec<String> = r.providers_for("m", 10_000_000).into_iter().map(|x| x.peer_id).collect();
        assert!(ids.contains(&"pNEW".to_string()));
        assert!(!ids.contains(&"p0".to_string()));
    }

    #[test]
    fn model_cap_evicts_stalest_model() {
        let mut r = ProviderRegistry::new(u64::MAX); // isolate the cap logic from TTL
        for i in 0..MAX_MODELS {
            r.insert(rec(&format!("m{i}"), "p"), 1000 + i as u64);
        }
        assert_eq!(r.model_count(), MAX_MODELS);
        // a new model → evicts the stalest existing model (m0), stays at the cap.
        r.insert(rec("mNEW", "p"), 9_000_000);
        assert_eq!(r.model_count(), MAX_MODELS);
        assert_eq!(r.providers_for("m0", 10_000_000).len(), 0);
        assert_eq!(r.providers_for("mNEW", 10_000_000).len(), 1);
    }
}
