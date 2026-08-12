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

use crate::credit::CreditAccount;
use crate::receipts::{CoSignedReceipt, NonceTracker};
use crate::verify::ReputationTracker;

/// Receipts, keyed by 16-byte nonce → opaque serialized receipt blob.
const RECEIPTS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("receipts");
/// Durable per-peer reputation snapshot, keyed by `peer_id` → opaque blob.
const PEER_REPUTATION: TableDefinition<&str, &[u8]> = TableDefinition::new("peer_reputation");
/// Durable per-peer give/take credit snapshot (M2.3), keyed by `peer_id` → opaque blob.
const PEER_CREDIT: TableDefinition<&str, &[u8]> = TableDefinition::new("peer_credit");
/// Spent receipt nonces (replay protection). Presence is the only signal → unit value.
const NONCES: TableDefinition<&[u8], ()> = TableDefinition::new("nonces");
/// Durable ledger rows for the desktop Ledger view, keyed by a monotonic sequence → a compact
/// [`LedgerEntry`] blob. Distinct from [`RECEIPTS`]: it records BOTH the provider's `served` side
/// and the consumer's `used` side (the latter are not receipts at all), so the view and its
/// lifetime totals survive a restart. Append-only + pruned to [`LEDGER_ROWS_CAP`].
const LEDGER_ROWS: TableDefinition<u64, &[u8]> = TableDefinition::new("ledger_rows");
/// Single-row counters for the ledger (next sequence). Keyed by a static tag.
const LEDGER_META: TableDefinition<&str, u64> = TableDefinition::new("ledger_meta");
/// Meta key holding the next `LEDGER_ROWS` sequence number.
const LEDGER_SEQ_KEY: &str = "seq";
/// Meta keys holding **monotonic lifetime aggregates** — incremented on every append and NEVER
/// pruned, so totals stay true even after old rows age out of `LEDGER_ROWS`. (Summing the rows
/// table would undercount once pruning kicks in, and since the counters are overwritten from these
/// on each restart, the displayed lifetime figures would shrink — so keep them here, not derived.)
const LEDGER_SERVED_TOKENS_KEY: &str = "served_tokens";
const LEDGER_USED_TOKENS_KEY: &str = "used_tokens";
const LEDGER_SERVED_COUNT_KEY: &str = "served_count";
/// Max durable ledger rows retained; the oldest are pruned past this. Bounds growth on a
/// long-lived node while comfortably covering the desktop's 250-row view + lifetime totals.
const LEDGER_ROWS_CAP: u64 = 5000;

/// One durable ledger transaction row for the desktop Ledger view. A compact, self-describing
/// blob — NOT a cryptographic receipt — capturing a `served` (provider) or `used` (consumer)
/// transfer so the Ledger view and lifetime token totals persist across restarts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerEntry {
    /// Unix-ms when recorded.
    pub ts_ms: u64,
    /// `"served"` (this node served a peer) or `"used"` (this node consumed from a peer).
    pub kind: String,
    /// Canonical model id transacted.
    pub model: String,
    /// Counterparty short peer id.
    pub counterparty: String,
    /// Tokens transferred.
    pub tokens: u64,
}

impl LedgerEntry {
    /// Compact little-endian wire form: `ts_ms(8) ‖ tokens(8) ‖ kind_len(1) ‖ kind ‖
    /// model_len(2) ‖ model ‖ cp_len(2) ‖ counterparty`. Strings are clamped to their
    /// length-prefix's max (a 255-byte kind / 64 KiB model or peer id is never reached in
    /// practice).
    pub fn to_bytes(&self) -> Vec<u8> {
        fn push_str(b: &mut Vec<u8>, s: &str, max: usize, len_bytes: usize) {
            let bytes = s.as_bytes();
            let n = bytes.len().min(max);
            match len_bytes {
                1 => b.push(n as u8),
                _ => b.extend_from_slice(&(n as u16).to_le_bytes()),
            }
            b.extend_from_slice(&bytes[..n]);
        }
        let mut b = Vec::with_capacity(24 + self.kind.len() + self.model.len() + self.counterparty.len());
        b.extend_from_slice(&self.ts_ms.to_le_bytes());
        b.extend_from_slice(&self.tokens.to_le_bytes());
        push_str(&mut b, &self.kind, u8::MAX as usize, 1);
        push_str(&mut b, &self.model, u16::MAX as usize, 2);
        push_str(&mut b, &self.counterparty, u16::MAX as usize, 2);
        b
    }

    /// Parse [`to_bytes`]; `None` on any truncation/format error so one corrupt row is skipped
    /// rather than poisoning the whole view.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        let mut p = 0usize;
        fn take<'a>(data: &'a [u8], p: &mut usize, n: usize) -> Option<&'a [u8]> {
            let s = data.get(*p..*p + n)?;
            *p += n;
            Some(s)
        }
        let ts_ms = u64::from_le_bytes(take(data, &mut p, 8)?.try_into().ok()?);
        let tokens = u64::from_le_bytes(take(data, &mut p, 8)?.try_into().ok()?);
        let kl = *take(data, &mut p, 1)?.first()? as usize;
        let kind = std::str::from_utf8(take(data, &mut p, kl)?).ok()?.to_string();
        let ml = u16::from_le_bytes(take(data, &mut p, 2)?.try_into().ok()?) as usize;
        let model = std::str::from_utf8(take(data, &mut p, ml)?).ok()?.to_string();
        let cl = u16::from_le_bytes(take(data, &mut p, 2)?.try_into().ok()?) as usize;
        let counterparty = std::str::from_utf8(take(data, &mut p, cl)?).ok()?.to_string();
        Some(Self { ts_ms, kind, model, counterparty, tokens })
    }
}

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
            wtx.open_table(PEER_CREDIT).map_err(StoreError::from)?;
            wtx.open_table(NONCES).map_err(StoreError::from)?;
            wtx.open_table(LEDGER_ROWS).map_err(StoreError::from)?;
            wtx.open_table(LEDGER_META).map_err(StoreError::from)?;
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

    // ── LEDGER_ROWS (desktop Ledger view — durable transaction rows) ──

    /// Append one durable ledger row (a `served`/`used` transaction) and prune the oldest rows
    /// past [`LEDGER_ROWS_CAP`]. Best-effort ordering via a monotonic sequence; the caller
    /// treats a failure as non-fatal (the swarm keeps working, the node just keeps a thinner
    /// local history).
    pub fn append_ledger_row(&self, entry: &LedgerEntry) -> Result<(), StoreError> {
        let blob = entry.to_bytes();
        let wtx = self.db.begin_write().map_err(StoreError::from)?;
        {
            let mut meta = wtx.open_table(LEDGER_META).map_err(StoreError::from)?;
            let seq = meta
                .get(LEDGER_SEQ_KEY)
                .map_err(StoreError::from)?
                .map(|v| v.value())
                .unwrap_or(0);
            {
                let mut rows = wtx.open_table(LEDGER_ROWS).map_err(StoreError::from)?;
                rows.insert(seq, blob.as_slice()).map_err(StoreError::from)?;
                let len = rows.len().map_err(StoreError::from)?;
                if len > LEDGER_ROWS_CAP {
                    // Prune the oldest (smallest-key) rows back down to the cap.
                    let excess = (len - LEDGER_ROWS_CAP) as usize;
                    let mut oldest: Vec<u64> = Vec::with_capacity(excess);
                    for item in rows.iter().map_err(StoreError::from)?.take(excess) {
                        let (k, _) = item.map_err(StoreError::from)?;
                        oldest.push(k.value());
                    }
                    for k in oldest {
                        rows.remove(k).map_err(StoreError::from)?;
                    }
                }
            }
            meta.insert(LEDGER_SEQ_KEY, seq.wrapping_add(1)).map_err(StoreError::from)?;
            // Bump the monotonic lifetime aggregates (never pruned) so totals stay true even after
            // old rows age out of LEDGER_ROWS. The `.get(...).map(value)` guard drops before the
            // `.insert`, mirroring the seq read above.
            let (tok_key, count_key) = if entry.kind == "served" {
                (LEDGER_SERVED_TOKENS_KEY, Some(LEDGER_SERVED_COUNT_KEY))
            } else {
                (LEDGER_USED_TOKENS_KEY, None)
            };
            let cur = meta.get(tok_key).map_err(StoreError::from)?.map(|v| v.value()).unwrap_or(0);
            meta.insert(tok_key, cur.saturating_add(entry.tokens)).map_err(StoreError::from)?;
            if let Some(ck) = count_key {
                let c = meta.get(ck).map_err(StoreError::from)?.map(|v| v.value()).unwrap_or(0);
                meta.insert(ck, c.saturating_add(1)).map_err(StoreError::from)?;
            }
        }
        wtx.commit().map_err(StoreError::from)?;
        Ok(())
    }

    /// The most recent `limit` ledger rows, **newest-first** (matches the desktop's ring order).
    /// Corrupt rows are skipped, not fatal.
    pub fn recent_ledger_rows(&self, limit: usize) -> Result<Vec<LedgerEntry>, StoreError> {
        let rtx = self.db.begin_read().map_err(StoreError::from)?;
        let rows = rtx.open_table(LEDGER_ROWS).map_err(StoreError::from)?;
        let mut out = Vec::new();
        for item in rows.iter().map_err(StoreError::from)?.rev() {
            if out.len() >= limit {
                break;
            }
            let (_, v) = item.map_err(StoreError::from)?;
            if let Some(e) = LedgerEntry::from_bytes(v.value()) {
                out.push(e);
            }
        }
        Ok(out)
    }

    /// Lifetime aggregates: `(served_tokens, used_tokens, served_count)`. Read from the monotonic
    /// `LEDGER_META` counters — NOT by summing `LEDGER_ROWS`, which is pruned to a rolling window —
    /// so these stay true lifetime figures across restarts and past the prune cap. The served
    /// count is the co-signed-receipt tally the Ledger view shows.
    pub fn ledger_totals(&self) -> Result<(u64, u64, u64), StoreError> {
        let rtx = self.db.begin_read().map_err(StoreError::from)?;
        let meta = match rtx.open_table(LEDGER_META) {
            Ok(t) => t,
            Err(_) => return Ok((0, 0, 0)), // table absent on a brand-new/empty ledger
        };
        let get = |k: &str| -> Result<u64, StoreError> {
            Ok(meta.get(k).map_err(StoreError::from)?.map(|v| v.value()).unwrap_or(0))
        };
        Ok((
            get(LEDGER_SERVED_TOKENS_KEY)?,
            get(LEDGER_USED_TOKENS_KEY)?,
            get(LEDGER_SERVED_COUNT_KEY)?,
        ))
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

    // ── PEER_CREDIT (M2.3 give/take) ──

    /// Checkpoint a peer's give/take credit snapshot blob.
    pub fn put_credit(&self, peer_id: &str, blob: &[u8]) -> Result<(), StoreError> {
        let wtx = self.db.begin_write().map_err(StoreError::from)?;
        {
            let mut t = wtx.open_table(PEER_CREDIT).map_err(StoreError::from)?;
            t.insert(peer_id, blob).map_err(StoreError::from)?;
        }
        wtx.commit().map_err(StoreError::from)?;
        Ok(())
    }

    /// Read a peer's credit snapshot blob, or `None` if we hold none.
    pub fn get_credit(&self, peer_id: &str) -> Result<Option<Vec<u8>>, StoreError> {
        let rtx = self.db.begin_read().map_err(StoreError::from)?;
        let t = rtx.open_table(PEER_CREDIT).map_err(StoreError::from)?;
        let got = t.get(peer_id).map_err(StoreError::from)?;
        Ok(got.map(|g| g.value().to_vec()))
    }

    /// Rehydrate the in-memory give/take credit map from the store on boot (M2.3),
    /// keyed by peer id. Mirrors the reputation rehydration in
    /// [`load_state_into_memory`](Self::load_state_into_memory); kept separate so callers
    /// that don't track credit aren't forced to thread an extra map.
    pub fn load_credit_into_memory(
        &self,
        credit: &mut HashMap<String, CreditAccount>,
    ) -> Result<(), StoreError> {
        let rtx = self.db.begin_read().map_err(StoreError::from)?;
        let t = rtx.open_table(PEER_CREDIT).map_err(StoreError::from)?;
        for entry in t.iter().map_err(StoreError::from)? {
            let (key, val) = entry.map_err(StoreError::from)?;
            let account = CreditAccount::from_bytes(val.value()).map_err(StoreError)?;
            credit.insert(key.value().to_string(), account);
        }
        Ok(())
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

    /// Atomically ledger one **accepted** co-signed receipt: persist the receipt blob under
    /// its nonce *and* burn that nonce in the replay guard, in a single write transaction.
    ///
    /// Returns `Ok(true)` if newly recorded, `Ok(false)` if the nonce was already spent (a
    /// replay — nothing is written or overwritten). Unlike
    /// [`flush_receipt_and_reputation`](Self::flush_receipt_and_reputation) this touches no
    /// reputation: it's the **provider-side** "I served this, here is the co-signed proof"
    /// entry, with no trust-score change implied. The all-or-nothing commit means a crash
    /// can never persist the receipt without burning its nonce (which would let the same
    /// receipt be re-ledgered).
    pub fn record_receipt(&self, receipt: &CoSignedReceipt) -> Result<bool, StoreError> {
        let nonce = receipt.payload.nonce;
        let wtx = self.db.begin_write().map_err(StoreError::from)?;
        let newly = {
            let mut nonces = wtx.open_table(NONCES).map_err(StoreError::from)?;
            if nonces.get(nonce.as_slice()).map_err(StoreError::from)?.is_some() {
                false // replay: nonce already spent
            } else {
                nonces.insert(nonce.as_slice(), ()).map_err(StoreError::from)?;
                true
            }
        };
        if newly {
            let blob = receipt.to_bytes();
            let mut receipts = wtx.open_table(RECEIPTS).map_err(StoreError::from)?;
            receipts.insert(nonce.as_slice(), blob.as_slice()).map_err(StoreError::from)?;
        }
        wtx.commit().map_err(StoreError::from)?; // atomic for both writes (or the no-op replay)
        Ok(newly)
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

    fn entry(ts: u64, kind: &str, model: &str, cp: &str, tokens: u64) -> LedgerEntry {
        LedgerEntry { ts_ms: ts, kind: kind.into(), model: model.into(), counterparty: cp.into(), tokens }
    }

    #[test]
    fn ledger_entry_bytes_roundtrip() {
        let e = entry(1_700_000_000_123, "served", "qwen3-vl:30b", "12D3KooWabc", 256);
        assert_eq!(LedgerEntry::from_bytes(&e.to_bytes()), Some(e));
        // A truncated blob decodes to None instead of panicking.
        assert_eq!(LedgerEntry::from_bytes(&[0u8; 4]), None);
        // Unicode model id survives.
        let u = entry(1, "used", "modèle/7b", "peer", 1);
        assert_eq!(LedgerEntry::from_bytes(&u.to_bytes()), Some(u));
    }

    #[test]
    fn ledger_rows_persist_across_reopen_newest_first() {
        let (_dir, path) = temp_db();
        {
            let store = Store::open(&path).unwrap();
            store.append_ledger_row(&entry(10, "used", "m1", "peerA", 5)).unwrap();
            store.append_ledger_row(&entry(20, "served", "m2", "peerB", 30)).unwrap();
            store.append_ledger_row(&entry(30, "served", "m2", "peerC", 7)).unwrap();
        }
        // Reopen the SAME file — the whole point: rows survive a restart.
        let store = Store::open(&path).unwrap();
        let recent = store.recent_ledger_rows(10).unwrap();
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].counterparty, "peerC"); // newest first
        assert_eq!(recent[2].counterparty, "peerA"); // oldest last
        // Totals: served 30+7=37 over 2 rows; used 5.
        assert_eq!(store.ledger_totals().unwrap(), (37, 5, 2));
    }

    #[test]
    fn recent_ledger_rows_respects_limit() {
        let store = Store::open_in_memory().unwrap();
        for i in 0..10 {
            store.append_ledger_row(&entry(i, "served", "m", "p", 1)).unwrap();
        }
        assert_eq!(store.recent_ledger_rows(3).unwrap().len(), 3);
        assert_eq!(store.recent_ledger_rows(100).unwrap().len(), 10);
        // Totals count every row, not just the windowed view.
        assert_eq!(store.ledger_totals().unwrap(), (10, 0, 10));
    }

    #[test]
    fn empty_ledger_totals_are_zero() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(store.ledger_totals().unwrap(), (0, 0, 0));
        assert!(store.recent_ledger_rows(5).unwrap().is_empty());
    }

    #[test]
    fn lifetime_totals_survive_pruning() {
        // Regression guard: totals must NOT shrink when old rows are pruned past LEDGER_ROWS_CAP.
        // The rows table is a rolling window; the lifetime aggregates live in LEDGER_META and count
        // every append. (Earlier, ledger_totals summed the prunable rows and undercounted.)
        let store = Store::open_in_memory().unwrap();
        let n = LEDGER_ROWS_CAP + 5;
        for i in 0..n {
            store.append_ledger_row(&entry(i, "served", "m", "p", 2)).unwrap();
        }
        // Rows are a capped rolling window…
        assert_eq!(store.recent_ledger_rows(usize::MAX).unwrap().len() as u64, LEDGER_ROWS_CAP);
        // …but lifetime totals reflect ALL appends, not just the retained window.
        assert_eq!(store.ledger_totals().unwrap(), (n * 2, 0, n));
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
    fn credit_snapshot_roundtrips_and_rehydrates() {
        // M2.3: a give/take credit account persists and rehydrates with its full state.
        use crate::credit::CreditAccount;
        let (_dir, path) = temp_db();
        let store = Store::open(&path).unwrap();
        assert_eq!(store.get_credit("unknown").unwrap(), None);

        let mut acct = CreditAccount::new(1_000);
        acct.record_consumed(60_000, 1_000); // a leecher
        store.put_credit("peerC", &acct.to_bytes()).unwrap();

        let mut credit = std::collections::HashMap::new();
        store.load_credit_into_memory(&mut credit).unwrap();
        let back = credit.get("peerC").expect("credit must rehydrate");
        // Same rate_cap as the original → the decay/give/take state round-tripped.
        assert_eq!(back.rate_cap(1_000).to_bits(), acct.rate_cap(1_000).to_bits());
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
    fn record_receipt_is_atomic_and_replay_safe() {
        // The provider-side ledger entry: an accepted co-signed receipt persists with its
        // nonce burnt in one shot; re-submitting the same receipt is a no-op replay.
        use crate::crypto_agility::SigAlg;
        use crate::receipts::{build_receipt, ReceiptPayload};
        use ed25519_dalek::SigningKey;

        let (_dir, path) = temp_db();
        let consumer = SigningKey::from_bytes(&[3u8; 32]);
        let provider = SigningKey::from_bytes(&[5u8; 32]);
        let nonce = [11u8; 16];
        let payload = ReceiptPayload {
            sig_alg: SigAlg::Ed25519,
            provider: provider.verifying_key(),
            consumer: consumer.verifying_key(),
            model_id: "qwen2.5/7b/q4_k_m/abcd0123abcd0123".to_string(),
            tokens: 128,
            nonce,
            ts_unix_ms: 1_700_000_000_000,
        };
        let receipt = build_receipt(payload, &consumer, &provider);

        let store = Store::open(&path).unwrap();

        // First record: newly ledgered.
        assert_eq!(store.record_receipt(&receipt).unwrap(), true);
        assert_eq!(store.receipt_count().unwrap(), 1);
        assert!(store.has_nonce(&nonce).unwrap(), "nonce burnt in the same txn");

        // Second record of the same receipt: replay → false, nothing changes.
        assert_eq!(store.record_receipt(&receipt).unwrap(), false);
        assert_eq!(store.receipt_count().unwrap(), 1, "replay must not add a row");

        // And the stored blob still decodes back to the same receipt.
        let blob = store.get_receipt(&nonce).unwrap().expect("receipt persisted");
        let decoded = CoSignedReceipt::from_bytes(&blob).unwrap();
        assert_eq!(decoded.payload.tokens, 128);
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
            sig_alg: crate::crypto_agility::SigAlg::Ed25519,
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
