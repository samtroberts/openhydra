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
pub const MAX_MODELS: usize = 2_048;
/// Max providers retained per model.
pub const MAX_PROVIDERS_PER_MODEL: usize = 256;
/// Max models one provider (OpenHydra id, bound to its key) holds here: above any real machine's
/// shared library, so one identity can't fill the model table.
pub const MAX_MODELS_PER_PEER: usize = 128;
/// Max entries in all: with records of at most [`MAX_STORED_RECORD_BYTES`], some 64 MB at worst,
/// whatever the other caps would allow (review 2026-10-29: 8,192 models × 512 providers allowed
/// ~20 GB on a 1 GB bootstrap).
pub const MAX_ENTRIES: usize = 8_192;
/// Max entries announced from one network (an IPv4 address or IPv6 /48, see
/// [`ProviderRegistry::insert_from`]): identities are free, the networks they come from aren't.
pub const MAX_ENTRIES_PER_ORIGIN: usize = 128;
/// Max providers of one model announced from one network: well below [`MAX_PROVIDERS_PER_MODEL`],
/// so no handful of networks can fill a model's list.
pub const MAX_PER_ORIGIN_PER_MODEL: usize = 8;
/// The largest record kept, as serialized. The signed fields are bounded (`dht::MAX_RECORD_BYTES`)
/// but not every field is signed: padded ones made 64 records outgrow the 1 MB a consumer reads, so
/// every lookup for that model failed (review 2026-10-29). 64 × 8 KB stays well under it.
pub const MAX_STORED_RECORD_BYTES: usize = 8 * 1024;
/// Max records in one query reply. A record is at most 4 KB (`dht::MAX_RECORD_BYTES`), so a reply
/// stays far below the 1 MB a consumer accepts: a model announced by many (padded) identities can't
/// make every lookup for it fail.
pub const MAX_REPLY_RECORDS: usize = 64;

/// Verified provider records the bootstrap has seen, grouped by model, each stamped with the
/// last time we saw an announce for it (for TTL expiry). Keyed inner by OpenHydra `peer_id`.
///
/// When a cap is reached, a new entry is refused (once expired ones are reaped) rather than one
/// already there evicted: a flood of fresh identities or model names can then only fill free room,
/// never push out the providers already listed.
#[derive(Debug, Default)]
pub struct ProviderRegistry {
    by_model: HashMap<String, HashMap<String, Entry>>,
    ttl_ms: u64,
    /// Entries per provider and per origin, so the caps are checked without a scan.
    per_peer: HashMap<String, usize>,
    per_origin: HashMap<String, usize>,
    per_origin_model: HashMap<(String, String), usize>,
    entries: usize,
    /// When a full registry was last reaped (so a flood can't make every insert a full scan).
    last_full_reap_ms: u64,
}

#[derive(Debug, Clone)]
struct Entry {
    record: PeerRecord,
    last_seen_ms: u64,
    origin: String,
}

impl ProviderRegistry {
    /// New registry expiring entries not re-announced within `ttl_ms` (match the DHT record TTL,
    /// ~300 s, so a provider that stops re-announcing ages out).
    pub fn new(ttl_ms: u64) -> Self {
        Self { ttl_ms, ..Default::default() }
    }

    /// [`insert_from`](Self::insert_from) with no origin (none is counted against its cap).
    pub fn insert(&mut self, record: PeerRecord, now_ms: u64) -> bool {
        self.insert_from(record, now_ms, "")
    }

    /// Ingest an **already-authenticity-verified** record (caller ran
    /// `dht::pex_record_is_authentic`), announced from network `origin` (the IPv4 address or IPv6
    /// /64 of the author's own connection; "" for none). Refreshes `last_seen`; inserts if new and
    /// every cap has room. Returns whether the record is held now. Ignores records with an empty
    /// `model_id` or `peer_id`.
    pub fn insert_from(&mut self, record: PeerRecord, now_ms: u64, origin: &str) -> bool {
        if record.model_id.trim().is_empty() || record.peer_id.trim().is_empty() {
            return false;
        }
        // An hour-old record, replayed, would keep a gone provider or a withdrawn model listed; a
        // padded one would make replies too big to read.
        if crate::dht::check_record_age(&record, now_ms).is_err()
            || serde_json::to_vec(&record).map_or(true, |v| v.len() > MAX_STORED_RECORD_BYTES)
        {
            return false;
        }
        if let Some(e) = self.by_model.get_mut(&record.model_id).and_then(|b| b.get_mut(&record.peer_id)) {
            // A refresh only by a record at least as new as the one held.
            if record.updated_unix_ms < e.record.updated_unix_ms {
                return false;
            }
            e.record = record;
            e.last_seen_ms = now_ms;
            return true;
        }
        if !self.room_for(&record, origin) {
            // Room may come from expired entries: reap, at most once a second while full.
            if now_ms.saturating_sub(self.last_full_reap_ms) < 1_000 {
                return false;
            }
            self.last_full_reap_ms = now_ms;
            self.reap(now_ms);
            if !self.room_for(&record, origin) {
                return false;
            }
        }
        *self.per_peer.entry(record.peer_id.clone()).or_default() += 1;
        if !origin.is_empty() {
            *self.per_origin.entry(origin.to_string()).or_default() += 1;
            *self.per_origin_model.entry((origin.to_string(), record.model_id.clone())).or_default() += 1;
        }
        self.entries += 1;
        let bucket = self.by_model.entry(record.model_id.clone()).or_default();
        bucket.insert(record.peer_id.clone(), Entry { record, last_seen_ms: now_ms, origin: origin.to_string() });
        true
    }

    /// Whether a new entry for `record` from `origin` fits every cap.
    fn room_for(&self, record: &PeerRecord, origin: &str) -> bool {
        self.entries < MAX_ENTRIES
            && self.per_peer.get(&record.peer_id).copied().unwrap_or(0) < MAX_MODELS_PER_PEER
            && (origin.is_empty()
                || (self.per_origin.get(origin).copied().unwrap_or(0) < MAX_ENTRIES_PER_ORIGIN
                    && self.per_origin_model.get(&(origin.to_string(), record.model_id.clone())).copied().unwrap_or(0)
                        < MAX_PER_ORIGIN_PER_MODEL))
            && match self.by_model.get(&record.model_id) {
                Some(b) => b.len() < MAX_PROVIDERS_PER_MODEL,
                None => self.by_model.len() < MAX_MODELS,
            }
    }

    /// Fresh providers for `model_id` (last seen within `ttl_ms`), at most [`MAX_REPLY_RECORDS`].
    /// When more are fresh, a random sample: re-announcing faster can't keep a set of identities at
    /// the top of every reply, and repeated queries see different providers. Stale entries are
    /// skipped here even before the reaper runs, so a query never returns an aged record.
    pub fn providers_for(&self, model_id: &str, now_ms: u64) -> Vec<PeerRecord> {
        use rand::seq::SliceRandom;
        let mut out: Vec<&Entry> = match self.by_model.get(model_id) {
            Some(b) => b
                .values()
                .filter(|e| now_ms.saturating_sub(e.last_seen_ms) < self.ttl_ms)
                .collect(),
            None => return Vec::new(),
        };
        if out.len() > MAX_REPLY_RECORDS {
            out.shuffle(&mut rand::thread_rng());
            out.truncate(MAX_REPLY_RECORDS);
        }
        out.sort_by(|a, b| b.last_seen_ms.cmp(&a.last_seen_ms));
        out.into_iter().map(|e| e.record.clone()).collect()
    }

    /// Drop entries older than `ttl_ms` (and now-empty model buckets). Returns the number of
    /// provider records removed. Call on a periodic tick.
    pub fn reap(&mut self, now_ms: u64) -> usize {
        let ttl = self.ttl_ms;
        self.remove_where(|_, e| now_ms.saturating_sub(e.last_seen_ms) >= ttl).0
    }

    /// Whether any entry is from the node with this libp2p id.
    pub fn serves(&self, libp2p_peer_id: &str) -> bool {
        self.by_model.values().any(|b| b.values().any(|e| e.record.libp2p_peer_id == libp2p_peer_id))
    }

    /// Remove every entry of the node with this libp2p id (it left: no connection to us for a
    /// while). Returns the newest record time among them, or `None` if there were none.
    pub fn remove_peer(&mut self, libp2p_peer_id: &str) -> Option<u64> {
        let (n, newest) = self.remove_where(|_, e| e.record.libp2p_peer_id == libp2p_peer_id);
        (n > 0).then_some(newest)
    }

    /// Remove one model of the node with this libp2p id (it withdrew it). Returns whether it was held.
    pub fn remove_model(&mut self, libp2p_peer_id: &str, model_id: &str) -> bool {
        self.remove_where(|m, e| m == model_id && e.record.libp2p_peer_id == libp2p_peer_id).0 > 0
    }

    /// Remove the entries `gone(model, entry)` picks; returns how many and the newest record time.
    fn remove_where(&mut self, mut gone_if: impl FnMut(&str, &Entry) -> bool) -> (usize, u64) {
        let mut gone: Vec<(String, String, String)> = Vec::new();
        let mut newest = 0;
        self.by_model.retain(|model, bucket| {
            bucket.retain(|pid, e| {
                let keep = !gone_if(model, e);
                if !keep {
                    newest = newest.max(e.record.updated_unix_ms);
                    gone.push((pid.clone(), e.origin.clone(), model.clone()));
                }
                keep
            });
            !bucket.is_empty()
        });
        for (pid, origin, model) in gone.iter().cloned() {
            dec(&mut self.per_peer, &pid);
            if !origin.is_empty() {
                dec(&mut self.per_origin, &origin);
                let k = (origin, model);
                if let Some(n) = self.per_origin_model.get_mut(&k) {
                    *n = n.saturating_sub(1);
                    if *n == 0 {
                        self.per_origin_model.remove(&k);
                    }
                }
            }
        }
        self.entries -= gone.len();
        (gone.len(), newest)
    }

    /// Total provider records across all models (for observability / bounds logging).
    pub fn len(&self) -> usize {
        self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries == 0
    }

    /// Distinct model count (for logging).
    pub fn model_count(&self) -> usize {
        self.by_model.len()
    }
}

fn dec(counts: &mut HashMap<String, usize>, key: &str) {
    if let Some(n) = counts.get_mut(key) {
        *n = n.saturating_sub(1);
        if *n == 0 {
            counts.remove(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reply_stays_small_however_many_identities_announce() {
        let mut r = ProviderRegistry::new(1000);
        for i in 0..MAX_PROVIDERS_PER_MODEL {
            r.insert(rec("m", &format!("p{i}")), 1);
        }
        let a = r.providers_for("m", 2);
        assert_eq!(a.len(), MAX_REPLY_RECORDS);
        // A sample, not always the same set.
        let b = r.providers_for("m", 2);
        let c = r.providers_for("m", 2);
        assert!(a.iter().map(|x| &x.peer_id).collect::<Vec<_>>() != b.iter().map(|x| &x.peer_id).collect::<Vec<_>>()
            || b.iter().map(|x| &x.peer_id).collect::<Vec<_>>() != c.iter().map(|x| &x.peer_id).collect::<Vec<_>>());
        // Fewer than the cap: everyone.
        let mut small = ProviderRegistry::new(1000);
        for i in 0..5 {
            small.insert(rec("m", &format!("p{i}")), 1);
        }
        assert_eq!(small.providers_for("m", 2).len(), 5);
    }

    #[test]
    fn one_peer_cannot_fill_the_model_table() {
        let mut r = ProviderRegistry::new(1000);
        for i in 0..MAX_MODELS_PER_PEER + 50 {
            r.insert(rec(&format!("m{i}"), "flood"), 1);
        }
        assert_eq!(r.model_count(), MAX_MODELS_PER_PEER);
        // Its existing entries still refresh; other peers are unaffected.
        r.insert(rec("m0", "flood"), 5);
        assert_eq!(r.providers_for("m0", 6).len(), 1);
        r.insert(rec("m9999", "honest"), 5);
        assert_eq!(r.providers_for("m9999", 6).len(), 1);
    }

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
    fn a_full_model_refuses_new_providers_and_keeps_the_ones_it_has() {
        let mut r = ProviderRegistry::new(u64::MAX); // isolate the cap logic from TTL
        for i in 0..MAX_PROVIDERS_PER_MODEL {
            assert!(r.insert_from(rec("m", &format!("p{i}")), 1000 + i as u64, &format!("net{i}")));
        }
        assert!(!r.insert_from(rec("m", "pNEW"), 2_000, "netX"), "full: refused");
        assert!(r.by_model["m"].contains_key("p0"), "nobody pushed out");
        // A refresh of one already there still lands.
        assert!(r.insert_from(rec("m", "p0"), 2_001, "net0"));
    }

    #[test]
    fn a_full_model_table_refuses_new_names_and_frees_room_as_entries_expire() {
        let mut r = ProviderRegistry::new(1_000);
        for i in 0..MAX_MODELS {
            assert!(r.insert(rec(&format!("m{i}"), &format!("p{}", i / MAX_MODELS_PER_PEER)), 100));
        }
        assert_eq!(r.model_count(), MAX_MODELS);
        assert!(!r.insert(rec("mNEW", "pNEW"), 200), "no room for a new model name");
        assert_eq!(r.providers_for("m0", 300).len(), 1, "the existing ones stay");
        // Once they've expired, room comes back without waiting for the reaper.
        assert!(r.insert(rec("mNEW", "pNEW"), 5_000));
        assert_eq!(r.model_count(), 1);
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn one_network_cannot_fill_the_registry_whatever_identities_it_uses() {
        let mut r = ProviderRegistry::new(u64::MAX);
        let mut held = 0;
        for i in 0..2_000 {
            if r.insert_from(rec(&format!("m{}", i % 100), &format!("sybil{i}")), 1, "2001:db8:1::/48") {
                held += 1;
            }
        }
        assert_eq!(held, MAX_ENTRIES_PER_ORIGIN);
        // Nor fill one model's list: at most a handful of its providers per network.
        let mut one = ProviderRegistry::new(u64::MAX);
        let n = (0..100).filter(|i| one.insert_from(rec("popular", &format!("s{i}")), 1, "203.0.113.9")).count();
        assert_eq!(n, MAX_PER_ORIGIN_PER_MODEL);
        // Another network still gets in.
        assert!(r.insert_from(rec("m0", "honest"), 2, "49.207.0.0"));
    }

    #[test]
    fn a_padded_or_replayed_advert_is_refused() {
        let mut r = ProviderRegistry::new(u64::MAX);
        let mut padded = rec("m", "p");
        padded.operator_id = Some("x".repeat(MAX_STORED_RECORD_BYTES));
        assert!(!r.insert_from(padded, 1, "n"), "padded past the record cap");
        let mut fresh = rec("m", "p");
        fresh.updated_unix_ms = 5_000_000;
        assert!(r.insert_from(fresh.clone(), 5_000_000, "n"));
        let mut older = fresh.clone();
        older.updated_unix_ms = 4_000_000;
        assert!(!r.insert_from(older, 5_000_100, "n"), "an older record doesn't refresh a newer one");
        let mut stale = rec("m", "q");
        stale.updated_unix_ms = 1;
        assert!(!r.insert_from(stale, 2 * crate::dht::MAX_RECORD_AGE_MS, "n"), "over an hour old");
    }

    #[test]
    fn the_registry_never_holds_more_than_its_entry_cap() {
        let mut r = ProviderRegistry::new(u64::MAX);
        for i in 0..MAX_ENTRIES + 500 {
            r.insert_from(rec(&format!("m{}", i % MAX_MODELS), &format!("p{i}")), 1, &format!("net{}", i / 100));
        }
        assert_eq!(r.len(), MAX_ENTRIES);
        assert_eq!(r.len(), r.by_model.values().map(HashMap::len).sum::<usize>());
        // Counters stay in step through a reap.
        let mut r = ProviderRegistry::new(10);
        r.insert_from(rec("a", "p1"), 1, "n1");
        r.insert_from(rec("b", "p1"), 1, "n1");
        r.insert_from(rec("a", "p2"), 50, "n2");
        assert_eq!(r.reap(30), 2);
        assert_eq!((r.len(), r.per_peer.get("p1"), r.per_origin.get("n1"), r.per_origin["n2"]), (1, None, None, 1));
    }

    /// GPU test 2026-10-30: a dead provider stayed in every registry until its TTL; now a bootstrap
    /// removes it when it has gone (and a single model when withdrawn), counters and all.
    #[test]
    fn remove_peer_and_remove_model() {
        let mut r = ProviderRegistry::new(300_000);
        let rec = |pid: &str, lp: &str, m: &str, t: u64| PeerRecord {
            peer_id: pid.into(), libp2p_peer_id: lp.into(), model_id: m.into(), updated_unix_ms: t, ..Default::default()
        };
        assert!(r.insert_from(rec("a", "LA", "m1", 5), 10, "1.2.3.4"));
        assert!(r.insert_from(rec("a", "LA", "m2", 7), 10, "1.2.3.4"));
        assert!(r.insert_from(rec("b", "LB", "m1", 9), 10, "5.6.7.8"));
        assert!(r.serves("LA") && !r.serves("LX"));
        assert!(r.remove_model("LA", "m2"));
        assert!(!r.remove_model("LA", "m2"));
        assert_eq!(r.remove_peer("LA"), Some(5));
        assert_eq!(r.remove_peer("LA"), None);
        assert!(!r.serves("LA"));
        assert_eq!(r.len(), 1);
        assert_eq!(r.providers_for("m1", 10).len(), 1);
        assert_eq!(r.per_peer.get("a"), None, "counters follow");
        assert_eq!(r.per_origin.get("1.2.3.4"), None);
    }

}
