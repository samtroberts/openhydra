// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Persistent ledger over `redb` (protocol.md §6) — M2.3 scaffold.
//!
//! A crash-safe, pure-Rust embedded database backing three tables:
//!
//! - **`RECEIPTS`** — co-signed receipts, keyed by their 16-byte nonce (unique per
//!   receipt). The value is an opaque serialized blob; the on-disk receipt encoding is
//!   defined by [`crate::receipts`] and folded in when the ledger goes live.
//! - **`PEER_REPUTATION`** — a durable snapshot of each peer's reputation, keyed by
//!   `peer_id`. The live in-memory [`crate::verify::ReputationTracker`] is rehydrated
//!   from / checkpointed to this table so a restart doesn't reset trust.
//! - **`NONCES`** — the replay-protection set: every spent receipt nonce, so a
//!   restarted node still rejects a re-submitted receipt.
//!
//! This is the *schema + persistence* scaffold only: it initializes the tables, does
//! typed put/get, and survives a reopen. It is **not** wired into the Python runtime or
//! the live node state yet — the value encodings, gossip replication, and the
//! materialized-view rebuild from signed receipts land on top in the rest of M2.3.

use std::collections::HashMap;
use std::path::Path;

use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};

use crate::receipts::{CoSignedReceipt, NonceTracker};
use crate::verify::ReputationTracker;

/// Receipts, keyed by 16-byte nonce → opaque serialized receipt blob.
const RECEIPTS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("receipts");
/// Durable per-peer reputation snapshot, keyed by `peer_id` → opaque blob.
const PEER_REPUTATION: TableDefinition<&str, &[u8]> = TableDefinition::new("peer_reputation");
/// Spent receipt nonces (replay protection). Presence is the only signal → unit value.
const NONCES: TableDefinition<&[u8], ()> = TableDefinition::new("nonces");

/// An error from the persistent store. Wraps the underlying `redb` failure as a string
/// so callers (and the future FFI surface) get one error type without leaking redb's.
#[derive(Debug)]
pub struct StoreError(String);

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "store error: {}", self.0)
    }
}

impl std::error::Error for StoreError {}

impl StoreError {
    fn from<E: std::fmt::Display>(e: E) -> Self {
        StoreError(e.to_string())
    }
}

/// The persistent ledger handle. Cheap to clone-free share by reference; a single
/// `Store` owns one `redb::Database` (which internally permits concurrent read txns and
/// serializes write txns).
pub struct Store {
    db: Database,
}

impl Store {
    /// Open (or create) the ledger at `path`, ensuring all tables exist.
    ///
    /// `redb`'s `create` opens an existing file or creates a new one; we then open every
    /// table once inside a write transaction so the schema is materialized up front
    /// (idempotent — opening an existing table is a no-op) and a subsequent read
    /// transaction never trips over a missing table on a brand-new database.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, StoreError> {
        let db = Database::create(path).map_err(StoreError::from)?;
        let store = Self { db };
        store.init_tables()?;
        Ok(store)
    }

    /// Open an **ephemeral** in-memory ledger (no file). Used as the node's default when
    /// no `db_path` is configured: every API works identically, but nothing survives a
    /// restart. Handy for tests and stateless/throwaway nodes.
    pub fn open_in_memory() -> Result<Self, StoreError> {
        let db = Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())
            .map_err(StoreError::from)?;
        let store = Self { db };
        store.init_tables()?;
        Ok(store)
    }

    /// Materialize all table definitions (idempotent).
    fn init_tables(&self) -> Result<(), StoreError> {
        let wtx = self.db.begin_write().map_err(StoreError::from)?;
        {
            wtx.open_table(RECEIPTS).map_err(StoreError::from)?;
            wtx.open_table(PEER_REPUTATION).map_err(StoreError::from)?;
            wtx.open_table(NONCES).map_err(StoreError::from)?;
        }
        wtx.commit().map_err(StoreError::from)?;
        Ok(())
    }

    // ── RECEIPTS ──

    /// Persist a co-signed receipt blob under its nonce. Overwrites an existing entry
    /// for the same nonce (callers gate replays via [`mark_nonce`](Self::mark_nonce)).
    pub fn put_receipt(&self, nonce: &[u8], blob: &[u8]) -> Result<(), StoreError> {
        let wtx = self.db.begin_write().map_err(StoreError::from)?;
        {
            let mut t = wtx.open_table(RECEIPTS).map_err(StoreError::from)?;
            t.insert(nonce, blob).map_err(StoreError::from)?;
        }
        wtx.commit().map_err(StoreError::from)?;
        Ok(())
    }

    /// Fetch a receipt blob by nonce, or `None` if absent.
    pub fn get_receipt(&self, nonce: &[u8]) -> Result<Option<Vec<u8>>, StoreError> {
        let rtx = self.db.begin_read().map_err(StoreError::from)?;
        let t = rtx.open_table(RECEIPTS).map_err(StoreError::from)?;
        let got = t.get(nonce).map_err(StoreError::from)?;
        Ok(got.map(|g| g.value().to_vec()))
    }

    /// Total number of receipts held.
    pub fn receipt_count(&self) -> Result<u64, StoreError> {
        let rtx = self.db.begin_read().map_err(StoreError::from)?;
        let t = rtx.open_table(RECEIPTS).map_err(StoreError::from)?;
        t.len().map_err(StoreError::from)
    }

    // ── PEER_REPUTATION ──

    /// Checkpoint a peer's reputation snapshot blob.
    pub fn put_reputation(&self, peer_id: &str, blob: &[u8]) -> Result<(), StoreError> {
        let wtx = self.db.begin_write().map_err(StoreError::from)?;
        {
            let mut t = wtx.open_table(PEER_REPUTATION).map_err(StoreError::from)?;
            t.insert(peer_id, blob).map_err(StoreError::from)?;
        }
        wtx.commit().map_err(StoreError::from)?;
        Ok(())
    }

    /// Read a peer's reputation snapshot blob, or `None` if we hold none.
    pub fn get_reputation(&self, peer_id: &str) -> Result<Option<Vec<u8>>, StoreError> {
        let rtx = self.db.begin_read().map_err(StoreError::from)?;
        let t = rtx.open_table(PEER_REPUTATION).map_err(StoreError::from)?;
        let got = t.get(peer_id).map_err(StoreError::from)?;
        Ok(got.map(|g| g.value().to_vec()))
    }

    // ── NONCES (replay protection) ──

    /// Mark a receipt nonce as spent.
    pub fn mark_nonce(&self, nonce: &[u8]) -> Result<(), StoreError> {
        let wtx = self.db.begin_write().map_err(StoreError::from)?;
        {
            let mut t = wtx.open_table(NONCES).map_err(StoreError::from)?;
            t.insert(nonce, ()).map_err(StoreError::from)?;
        }
        wtx.commit().map_err(StoreError::from)?;
        Ok(())
    }

    /// Whether a nonce has already been spent (a restart-durable replay check).
    pub fn has_nonce(&self, nonce: &[u8]) -> Result<bool, StoreError> {
        let rtx = self.db.begin_read().map_err(StoreError::from)?;
        let t = rtx.open_table(NONCES).map_err(StoreError::from)?;
        Ok(t.get(nonce).map_err(StoreError::from)?.is_some())
    }

    // ── Atomic transaction flush + boot rehydration ──

    /// Atomically persist one completed transaction: the co-signed `receipt`, its
    /// now-spent nonce, and `peer_id`'s updated `reputation` snapshot — **all in a single
    /// `redb` write transaction**. The all-or-nothing commit means a crash mid-flush can
    /// never leave a receipt without its replay-guard nonce (which would let it be
    /// double-counted), nor a reputation bump without the receipt that justified it.
    ///
    /// The nonce is taken from `receipt.payload.nonce`. `peer_id` is whichever peer the
    /// reputation update is *about* (e.g. the serving provider, for a consumer-side
    /// honored receipt) — the caller passes the post-update tracker.
    pub fn flush_receipt_and_reputation(
        &self,
        receipt: &CoSignedReceipt,
        peer_id: &str,
        reputation: &ReputationTracker,
    ) -> Result<(), StoreError> {
        let nonce = receipt.payload.nonce;
        let receipt_blob = receipt.to_bytes();
        let rep_blob = reputation.to_bytes();

        let wtx = self.db.begin_write().map_err(StoreError::from)?;
        {
            let mut t = wtx.open_table(RECEIPTS).map_err(StoreError::from)?;
            t.insert(nonce.as_slice(), receipt_blob.as_slice()).map_err(StoreError::from)?;
        }
        {
            let mut t = wtx.open_table(NONCES).map_err(StoreError::from)?;
            t.insert(nonce.as_slice(), ()).map_err(StoreError::from)?;
        }
        {
            let mut t = wtx.open_table(PEER_REPUTATION).map_err(StoreError::from)?;
            t.insert(peer_id, rep_blob.as_slice()).map_err(StoreError::from)?;
        }
        wtx.commit().map_err(StoreError::from)?; // one atomic commit for all three writes
        Ok(())
    }

    /// Rehydrate in-memory state from the store on boot: repopulate the replay guard with
    /// every spent nonce, and the reputation map with every persisted peer snapshot — so a
    /// restart resumes with the same trust scores and replay protection it shut down with.
    ///
    /// Operates on the protocol crate's own pure types (`NonceTracker`, `HashMap<String,
    /// ReputationTracker>`); the live node wraps the map in its `Arc<RwLock<…>>` and calls
    /// this under the write lock at startup. Existing in-memory entries are preserved;
    /// persisted reputation snapshots overwrite same-peer entries (the store is the source
    /// of truth on boot).
    pub fn load_state_into_memory(
        &self,
        nonce_tracker: &mut NonceTracker,
        reputation: &mut HashMap<String, ReputationTracker>,
    ) -> Result<(), StoreError> {
        let rtx = self.db.begin_read().map_err(StoreError::from)?;

        let nonces = rtx.open_table(NONCES).map_err(StoreError::from)?;
        for entry in nonces.iter().map_err(StoreError::from)? {
            let (key, _) = entry.map_err(StoreError::from)?;
            let bytes = key.value();
            let nonce: [u8; 16] = bytes
                .try_into()
                .map_err(|_| StoreError(format!("corrupt nonce key: {} bytes", bytes.len())))?;
            nonce_tracker.mark_seen(nonce);
        }

        let reps = rtx.open_table(PEER_REPUTATION).map_err(StoreError::from)?;
        for entry in reps.iter().map_err(StoreError::from)? {
            let (key, val) = entry.map_err(StoreError::from)?;
            let tracker = ReputationTracker::from_bytes(val.value()).map_err(StoreError)?;
            reputation.insert(key.value().to_string(), tracker);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A temp DB path inside a fresh tempdir. The tempdir is returned so the caller
    /// keeps it alive (dropping it deletes the directory).
    fn temp_db() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ledger.redb");
        (dir, path)
    }

    #[test]
    fn opens_clean_and_is_initially_empty() {
        let (_dir, path) = temp_db();
        let store = Store::open(&path).unwrap();
        assert_eq!(store.receipt_count().unwrap(), 0);
        assert_eq!(store.get_receipt(b"nope-nonce-16byt").unwrap(), None);
        assert_eq!(store.get_reputation("unknown").unwrap(), None);
        assert!(!store.has_nonce(b"unseen-nonce-16b").unwrap());
    }

    #[test]
    fn writes_and_reads_back_each_table() {
        let (_dir, path) = temp_db();
        let store = Store::open(&path).unwrap();

        let nonce = [7u8; 16];
        store.put_receipt(&nonce, b"receipt-blob").unwrap();
        store.put_reputation("peerA", b"rep-snapshot").unwrap();
        store.mark_nonce(&nonce).unwrap();

        assert_eq!(store.get_receipt(&nonce).unwrap().as_deref(), Some(&b"receipt-blob"[..]));
        assert_eq!(store.get_reputation("peerA").unwrap().as_deref(), Some(&b"rep-snapshot"[..]));
        assert!(store.has_nonce(&nonce).unwrap());
        assert_eq!(store.receipt_count().unwrap(), 1);
    }

    #[test]
    fn persists_across_a_simulated_restart() {
        // Write, drop the Store (closing the DB file), then reopen the SAME path and
        // assert every table's data survived — proving on-disk durability, not just
        // in-memory state.
        let (_dir, path) = temp_db();
        let nonce = [42u8; 16];
        {
            let store = Store::open(&path).unwrap();
            store.put_receipt(&nonce, b"durable-receipt").unwrap();
            store.put_reputation("peerB", b"durable-rep").unwrap();
            store.mark_nonce(&nonce).unwrap();
            // `store` dropped here → database handle closed.
        }

        let reopened = Store::open(&path).unwrap();
        assert_eq!(
            reopened.get_receipt(&nonce).unwrap().as_deref(),
            Some(&b"durable-receipt"[..]),
            "receipt must survive a restart"
        );
        assert_eq!(
            reopened.get_reputation("peerB").unwrap().as_deref(),
            Some(&b"durable-rep"[..]),
            "reputation snapshot must survive a restart"
        );
        assert!(reopened.has_nonce(&nonce).unwrap(), "spent nonce must survive a restart");
        assert_eq!(reopened.receipt_count().unwrap(), 1);
    }

    #[test]
    fn overwrites_receipt_for_same_nonce() {
        let (_dir, path) = temp_db();
        let store = Store::open(&path).unwrap();
        let nonce = [1u8; 16];
        store.put_receipt(&nonce, b"first").unwrap();
        store.put_receipt(&nonce, b"second").unwrap();
        assert_eq!(store.get_receipt(&nonce).unwrap().as_deref(), Some(&b"second"[..]));
        assert_eq!(store.receipt_count().unwrap(), 1);
    }

    #[test]
    fn flush_then_reopen_rehydrates_reputation_and_nonce() {
        // The full M2.3 loop: a simulated Honored transaction is flushed atomically,
        // then a fresh Store against the same file rehydrates the burnt nonce + updated
        // reputation back into in-memory state — exactly what a node does on restart.
        use crate::receipts::{build_receipt, verify_receipt, ReceiptPayload};
        use crate::verify::{ReputationTracker, VerificationOutcome};
        use ed25519_dalek::SigningKey;

        let (_dir, path) = temp_db();
        let consumer = SigningKey::from_bytes(&[7u8; 32]);
        let provider = SigningKey::from_bytes(&[9u8; 32]);
        let peer_id = "12D3KooWProvider";
        let nonce = [42u8; 16];

        let payload = ReceiptPayload {
            provider: provider.verifying_key(),
            consumer: consumer.verifying_key(),
            model_id: "qwen3.5/2b/fp16/5632a1b48425a5ae".to_string(),
            tokens: 512,
            nonce,
            ts_unix_ms: 1_700_000_000_000,
        };
        let receipt = build_receipt(payload, &consumer, &provider);

        // Post-transaction reputation: an honored receipt lifts the provider above neutral.
        let now = 1_700_000_000_000u64;
        let mut tracker = ReputationTracker::new(now);
        let honored_score = tracker.record(VerificationOutcome::Honored, now);
        assert!(honored_score > 50.0);

        // Flush the whole transaction atomically, then close the DB.
        {
            let store = Store::open(&path).unwrap();
            store.flush_receipt_and_reputation(&receipt, peer_id, &tracker).unwrap();
            assert_eq!(store.receipt_count().unwrap(), 1);
        }

        // Reopen (simulated restart) and rehydrate into fresh in-memory state.
        let reopened = Store::open(&path).unwrap();
        let mut nonces = NonceTracker::new();
        let mut reputation: HashMap<String, ReputationTracker> = HashMap::new();
        reopened.load_state_into_memory(&mut nonces, &mut reputation).unwrap();

        // The spent nonce is back in the replay guard.
        assert!(nonces.contains(&nonce), "burnt nonce must rehydrate into the replay guard");
        assert_eq!(nonces.len(), 1);

        // The provider's updated reputation is back in the map, decaying identically.
        let rt = reputation.get(peer_id).expect("provider reputation must rehydrate");
        assert_eq!(rt.score_at(now), honored_score, "rehydrated score matches the flushed one");
        assert!(rt.score_at(now) > 50.0);

        // And the stored receipt blob still decodes and verifies.
        let blob = reopened.get_receipt(&nonce).unwrap().expect("receipt persisted");
        let decoded = CoSignedReceipt::from_bytes(&blob).unwrap();
        assert_eq!(decoded.payload.tokens, 512);
        assert_eq!(verify_receipt(&decoded), Ok(()), "signatures survived the flush");
    }
}
