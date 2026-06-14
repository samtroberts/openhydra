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

use std::path::Path;

use redb::{Database, ReadableTableMetadata, TableDefinition};

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
}
