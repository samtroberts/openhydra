// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Reference registry core: signed round packs and the pool they're issued from.
//!
//! A registry holds batches of reference texts (each batch = one `refbuild assemble` output: texts
//! with bands, a calibrated cut-off, a model identity) and hands verifying gateways **round packs** —
//! enough never-issued texts from one batch for one verification round, plus that batch's cut-off.
//! Every text is issued **once, network-wide**, so a text a provider (or its colluders) has seen is
//! never used against anyone again; only un-issued texts need to stay secret.
//!
//! Packs are signed (Ed25519, domain-separated) over a canonical binary encoding; a gateway pins the
//! registry's public key and rejects anything else. Issuing is one database write transaction, so two
//! gateways can never receive the same text.
//!
//! Pure protocol logic + persistence (redb), no networking — the HTTP service and gateway client live
//! in the agent.

use std::path::Path;

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use redb::{Database, ReadableTable, TableDefinition};

use crate::logprob::{Checkpoint, ModelIdentity, ReferenceSet, ReferenceText};

/// Signature domain for round packs.
const PACK_DOMAIN: &[u8] = b"openhydra-refpack-v1\0";
const PACK_FORMAT_V1: u8 = 0x01;
/// v2 adds early-stop checkpoints.
const PACK_FORMAT_V2: u8 = 0x02;
const MAX_CHECKPOINTS: usize = 64;
/// Bounds enforced when decoding (a pack is attacker-reachable input if the signature check is ever
/// skipped or the key leaks; decode never allocates without a bound).
const MAX_PACK_TEXTS: usize = 512;
const MAX_TEXT_TOKENS: usize = 4096;
const MAX_STRING: usize = 1024;
const MAX_HANDLES: usize = 64;

// ── binary codec ──

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
    fn text(&mut self, t: &ReferenceText) {
        self.str(&t.id);
        self.f64(t.band);
        self.u32(t.token_ids.len() as u32);
        for id in &t.token_ids {
            self.u32(*id);
        }
        self.u32(t.logprobs.len() as u32);
        for lp in &t.logprobs {
            self.u32(lp.to_bits());
        }
    }
    fn checkpoints(&mut self, cps: &[Checkpoint]) {
        self.u32(cps.len() as u32);
        for c in cps {
            self.u32(c.after as u32);
            match c.pass_below {
                Some(p) => {
                    self.u8(1);
                    self.f64(p);
                }
                None => self.u8(0),
            }
            self.f64(c.fail_above);
        }
    }
    fn identity(&mut self, id: &Option<ModelIdentity>) {
        match id {
            None => self.u8(0),
            Some(id) => {
                self.u8(1);
                self.str(&id.family);
                self.str(&id.params);
                self.str(&id.quant);
            }
        }
    }
}

struct R<'a>(&'a [u8]);

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.0.len() < n {
            return None;
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Some(head)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn f64(&mut self) -> Option<f64> {
        Some(f64::from_bits(self.u64()?))
    }
    fn arr16(&mut self) -> Option<[u8; 16]> {
        self.take(16)?.try_into().ok()
    }
    fn len(&mut self, max: usize) -> Option<usize> {
        let n = self.u32()? as usize;
        (n <= max).then_some(n)
    }
    fn str(&mut self) -> Option<String> {
        let n = self.len(MAX_STRING)?;
        String::from_utf8(self.take(n)?.to_vec()).ok()
    }
    fn text(&mut self) -> Option<ReferenceText> {
        let id = self.str()?;
        let band = self.f64()?;
        let n = self.len(MAX_TEXT_TOKENS)?;
        let token_ids = (0..n).map(|_| self.u32()).collect::<Option<Vec<_>>>()?;
        let m = self.len(MAX_TEXT_TOKENS)?;
        let logprobs = (0..m).map(|_| self.u32().map(f32::from_bits)).collect::<Option<Vec<_>>>()?;
        Some(ReferenceText { id, token_ids, logprobs, band })
    }
    fn checkpoints(&mut self) -> Option<Vec<Checkpoint>> {
        let n = self.len(MAX_CHECKPOINTS)?;
        (0..n)
            .map(|_| {
                let after = self.u32()? as usize;
                let pass_below = match self.u8()? {
                    0 => None,
                    1 => Some(self.f64()?),
                    _ => return None,
                };
                Some(Checkpoint { after, pass_below, fail_above: self.f64()? })
            })
            .collect()
    }
    fn identity(&mut self) -> Option<Option<ModelIdentity>> {
        match self.u8()? {
            0 => Some(None),
            1 => {
                let (f, p, q) = (self.str()?, self.str()?, self.str()?);
                Some(Some(ModelIdentity::new(&f, &p, &q)?))
            }
            _ => None,
        }
    }
}

// ── round packs ──

/// Enough never-issued texts from one batch for one verification round, with the batch's decision
/// parameters. Converts to a [`ReferenceSet`] the gateway's scorer uses as-is.
#[derive(Debug, Clone, PartialEq)]
pub struct RefPack {
    pub pack_id: [u8; 16],
    pub batch_id: [u8; 16],
    pub issued_ms: u64,
    /// Name of the gateway key it was issued to (for accountability if a pack leaks).
    pub issued_to: String,
    /// The batch label, e.g. `qwen3.5-9b@q4_k_m` — stable across batches of one model, so a
    /// provider's pass streak carries across packs.
    pub label: String,
    pub identity: Option<ModelIdentity>,
    pub handles: Vec<String>,
    pub cutoff: f64,
    pub min_texts: u32,
    pub early_stop: Vec<Checkpoint>,
    pub texts: Vec<ReferenceText>,
}

/// Why a pack couldn't be decoded or verified.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    /// The signature doesn't verify under the pinned registry key (or the key bytes are invalid).
    BadSignature,
    /// Signed by a different key than the one pinned.
    WrongKey,
    /// Malformed, truncated, over-long, unknown version, or not a valid reference set.
    Malformed,
}

impl RefPack {
    /// Canonical encoding — what is signed.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(PACK_FORMAT_V2);
        w.bytes(&self.pack_id);
        w.bytes(&self.batch_id);
        w.u64(self.issued_ms);
        w.str(&self.issued_to);
        w.str(&self.label);
        w.identity(&self.identity);
        w.u32(self.handles.len() as u32);
        for h in &self.handles {
            w.str(h);
        }
        w.f64(self.cutoff);
        w.u32(self.min_texts);
        w.checkpoints(&self.early_stop);
        w.u32(self.texts.len() as u32);
        for t in &self.texts {
            w.text(t);
        }
        w.0
    }

    /// Strict inverse of [`encode`](Self::encode): bounded, rejects trailing bytes, and only returns
    /// a pack whose [`to_reference_set`](Self::to_reference_set) validates.
    pub fn decode(bytes: &[u8]) -> Result<Self, PackError> {
        let mut r = R(bytes);
        let pack = (|| {
            let version = r.u8()?;
            if version != PACK_FORMAT_V1 && version != PACK_FORMAT_V2 {
                return None;
            }
            let pack_id = r.arr16()?;
            let batch_id = r.arr16()?;
            let issued_ms = r.u64()?;
            let issued_to = r.str()?;
            let label = r.str()?;
            let identity = r.identity()?;
            let nh = r.len(MAX_HANDLES)?;
            let handles = (0..nh).map(|_| r.str()).collect::<Option<Vec<_>>>()?;
            let cutoff = r.f64()?;
            let min_texts = r.u32()?;
            let early_stop = if version >= PACK_FORMAT_V2 { r.checkpoints()? } else { Vec::new() };
            let nt = r.len(MAX_PACK_TEXTS)?;
            let texts = (0..nt).map(|_| r.text()).collect::<Option<Vec<_>>>()?;
            Some(RefPack { pack_id, batch_id, issued_ms, issued_to, label, identity, handles, cutoff, min_texts, early_stop, texts })
        })()
        .ok_or(PackError::Malformed)?;
        if !r.0.is_empty() {
            return Err(PackError::Malformed);
        }
        // A pack may hold fewer texts than a full round (a first stage, topped up on request).
        pack.to_reference_set().validate_texts().map_err(|_| PackError::Malformed)?;
        Ok(pack)
    }

    /// The pack as a one-round reference set.
    pub fn to_reference_set(&self) -> ReferenceSet {
        ReferenceSet {
            model_key: self.label.clone(),
            identity: self.identity.clone(),
            texts: self.texts.clone(),
            cutoff: self.cutoff,
            min_texts: self.min_texts as usize,
            early_stop: self.early_stop.clone(),
        }
    }
}

/// Sign an encoded pack with the registry key.
pub fn sign_pack(pack: &RefPack, key: &SigningKey) -> (Vec<u8>, [u8; 64]) {
    let bytes = pack.encode();
    let mut msg = PACK_DOMAIN.to_vec();
    msg.extend_from_slice(&bytes);
    (bytes, key.sign(&msg).to_bytes())
}

/// Verify a signed pack against the **pinned** registry public key, then decode it. `signer` is the
/// key the pack claims to be signed with; it must equal `pinned`.
pub fn verify_pack(bytes: &[u8], signature: &[u8; 64], signer: &[u8; 32], pinned: &[u8; 32]) -> Result<RefPack, PackError> {
    if signer != pinned {
        return Err(PackError::WrongKey);
    }
    let key = VerifyingKey::from_bytes(pinned).map_err(|_| PackError::BadSignature)?;
    let mut msg = PACK_DOMAIN.to_vec();
    msg.extend_from_slice(bytes);
    key.verify_strict(&msg, &Signature::from_bytes(signature)).map_err(|_| PackError::BadSignature)?;
    RefPack::decode(bytes)
}

// ── the pool ──

/// Batch id (hex) → batch metadata.
const BATCHES: TableDefinition<&str, &[u8]> = TableDefinition::new("ref_batches");
/// (batch id, text index) → encoded reference text.
const TEXTS: TableDefinition<(&str, u32), &[u8]> = TableDefinition::new("ref_texts");
/// Batch id → index of the next never-issued text (texts are issued in order).
const NEXT: TableDefinition<&str, u32> = TableDefinition::new("ref_next");
/// Pack id (hex) → issuance record (who, which batch range, when).
const PACKS: TableDefinition<&str, &[u8]> = TableDefinition::new("ref_packs");
/// SHA-256 (hex) of a gateway API key → key record.
const KEYS: TableDefinition<&str, &[u8]> = TableDefinition::new("ref_keys");

/// What a registry knows about one imported batch.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchInfo {
    pub batch_id: [u8; 16],
    pub label: String,
    pub identity: Option<ModelIdentity>,
    pub handles: Vec<String>,
    pub cutoff: f64,
    pub min_texts: u32,
    pub early_stop: Vec<Checkpoint>,
    pub total: u32,
    pub imported_ms: u64,
}

impl BatchInfo {
    /// Texts one round takes from this batch with `spare` extra per pack: `(decided at the first
    /// early-stop checkpoint, run to the end)`. Without checkpoints both are a full round.
    pub fn texts_per_round(&self, spare: u32) -> (u32, u32) {
        let full = self.min_texts + spare;
        let first = self.early_stop.first().map_or(full, |c| (c.after as u32 + spare).min(full));
        (first, full)
    }

    fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(2);
        w.bytes(&self.batch_id);
        w.str(&self.label);
        w.identity(&self.identity);
        w.u32(self.handles.len() as u32);
        for h in &self.handles {
            w.str(h);
        }
        w.f64(self.cutoff);
        w.u32(self.min_texts);
        w.u32(self.total);
        w.u64(self.imported_ms);
        w.checkpoints(&self.early_stop);
        w.0
    }

    fn decode(b: &[u8]) -> Option<Self> {
        let mut r = R(b);
        let version = r.u8()?;
        if version != 1 && version != 2 {
            return None;
        }
        let batch_id = r.arr16()?;
        let label = r.str()?;
        let identity = r.identity()?;
        let nh = r.len(MAX_HANDLES)?;
        let handles = (0..nh).map(|_| r.str()).collect::<Option<Vec<_>>>()?;
        let (cutoff, min_texts, total, imported_ms) = (r.f64()?, r.u32()?, r.u32()?, r.u64()?);
        let early_stop = if version >= 2 { r.checkpoints()? } else { Vec::new() };
        Some(Self { batch_id, label, identity, handles, cutoff, min_texts, early_stop, total, imported_ms })
    }

    fn matches_identity(&self, identity: Option<&ModelIdentity>) -> bool {
        identity.is_some() && self.identity.as_ref() == identity
    }

    fn matches_handle(&self, handle: Option<&str>) -> bool {
        handle.is_some_and(|h| self.handles.iter().any(|x| x == h))
    }
}

/// A batch with how many texts are still un-issued.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchStatus {
    pub info: BatchInfo,
    pub remaining: u32,
}

/// A gateway API key's record (the key itself is never stored — only its SHA-256).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyRecord {
    pub name: String,
    pub created_ms: u64,
    pub revoked: bool,
}

impl KeyRecord {
    fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(1);
        w.str(&self.name);
        w.u64(self.created_ms);
        w.u8(self.revoked as u8);
        w.0
    }
    fn decode(b: &[u8]) -> Option<Self> {
        let mut r = R(b);
        if r.u8()? != 1 {
            return None;
        }
        Some(Self { name: r.str()?, created_ms: r.u64()?, revoked: r.u8()? == 1 })
    }
}

/// Why a pool operation failed.
#[derive(Debug, Clone, PartialEq)]
pub enum PoolError {
    /// No batch serves this model.
    NoReference,
    /// Batches exist for this model but none has a full round of un-issued texts left.
    Exhausted,
    /// The batch being imported is invalid or already imported.
    BadBatch(String),
    /// A key name is taken, or unknown.
    Key(String),
    Db(String),
}

fn db<E: std::fmt::Display>(e: E) -> PoolError {
    PoolError::Db(e.to_string())
}

/// The registry's pool of reference texts.
pub struct RefPool {
    db: Database,
}

impl RefPool {
    /// Open (or create) the pool at `path`.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, PoolError> {
        Self::init(Database::create(path).map_err(db)?)
    }

    /// An in-memory pool (tests).
    pub fn open_in_memory() -> Result<Self, PoolError> {
        Self::init(Database::builder().create_with_backend(redb::backends::InMemoryBackend::new()).map_err(db)?)
    }

    fn init(db_: Database) -> Result<Self, PoolError> {
        let wtx = db_.begin_write().map_err(db)?;
        {
            wtx.open_table(BATCHES).map_err(db)?;
            wtx.open_table(TEXTS).map_err(db)?;
            wtx.open_table(NEXT).map_err(db)?;
            wtx.open_table(PACKS).map_err(db)?;
            wtx.open_table(KEYS).map_err(db)?;
        }
        wtx.commit().map_err(db)?;
        Ok(Self { db: db_ })
    }

    /// Add a built reference set as a new batch. The set must validate; `batch_id` must be new.
    /// Texts already in the pool (same token ids, in any batch) are refused, so a text can't be
    /// imported twice and issued twice.
    pub fn import(&self, set: &ReferenceSet, handles: &[String], batch_id: [u8; 16], now_ms: u64) -> Result<BatchInfo, PoolError> {
        set.validate().map_err(|e| PoolError::BadBatch(format!("{e:?}")))?;
        if set.identity.is_none() && handles.is_empty() {
            return Err(PoolError::BadBatch("needs a model identity or a handle".into()));
        }
        let bid = hex::encode(batch_id);
        let info = BatchInfo {
            batch_id,
            label: set.model_key.clone(),
            identity: set.identity.clone(),
            handles: handles.to_vec(),
            cutoff: set.cutoff,
            min_texts: set.min_texts as u32,
            early_stop: set.early_stop.clone(),
            total: set.texts.len() as u32,
            imported_ms: now_ms,
        };
        let digests: std::collections::HashSet<[u8; 32]> = set.texts.iter().map(|t| t.digest()).collect();
        let wtx = self.db.begin_write().map_err(db)?;
        {
            let batches = wtx.open_table(BATCHES).map_err(db)?;
            if batches.get(bid.as_str()).map_err(db)?.is_some() {
                return Err(PoolError::BadBatch(format!("batch {bid} already imported")));
            }
            let texts = wtx.open_table(TEXTS).map_err(db)?;
            for row in texts.iter().map_err(db)? {
                let (_, v) = row.map_err(db)?;
                if let Some(t) = R(v.value()).text() {
                    if digests.contains(&t.digest()) {
                        return Err(PoolError::BadBatch(format!("text {} is already in the pool", t.id)));
                    }
                }
            }
        }
        {
            let mut batches = wtx.open_table(BATCHES).map_err(db)?;
            batches.insert(bid.as_str(), info.encode().as_slice()).map_err(db)?;
            let mut texts = wtx.open_table(TEXTS).map_err(db)?;
            for (i, t) in set.texts.iter().enumerate() {
                let mut w = W(Vec::new());
                w.text(t);
                texts.insert((bid.as_str(), i as u32), w.0.as_slice()).map_err(db)?;
            }
            wtx.open_table(NEXT).map_err(db)?.insert(bid.as_str(), 0u32).map_err(db)?;
        }
        wtx.commit().map_err(db)?;
        Ok(info)
    }

    /// Every batch with its remaining count, oldest first.
    pub fn status(&self) -> Result<Vec<BatchStatus>, PoolError> {
        let rtx = self.db.begin_read().map_err(db)?;
        let batches = rtx.open_table(BATCHES).map_err(db)?;
        let next = rtx.open_table(NEXT).map_err(db)?;
        let mut out = Vec::new();
        for row in batches.iter().map_err(db)? {
            let (k, v) = row.map_err(db)?;
            let info = BatchInfo::decode(v.value()).ok_or_else(|| PoolError::Db("corrupt batch row".into()))?;
            let used = next.get(k.value()).map_err(db)?.map(|g| g.value()).unwrap_or(info.total);
            out.push(BatchStatus { remaining: info.total.saturating_sub(used), info });
        }
        out.sort_by_key(|b| b.info.imported_ms);
        Ok(out)
    }

    /// Issue a round pack from the oldest batch serving `identity` / `handle` that still has a full
    /// round (`min_texts + spare`) left. With early-stop checkpoints the pack is a **first stage** —
    /// enough texts for the first checkpoint plus `spare` — and the rest of the round is issued only
    /// if the gateway asks ([`continue_pack`](Self::continue_pack)), so rounds that stop early don't
    /// burn a full round of texts. Atomic: texts are marked issued in the same transaction.
    pub fn issue_pack(
        &self,
        identity: Option<&ModelIdentity>,
        handle: Option<&str>,
        spare: u32,
        issued_to: &str,
        pack_id: [u8; 16],
        now_ms: u64,
    ) -> Result<RefPack, PoolError> {
        let wtx = self.db.begin_write().map_err(db)?;
        let pack = {
            let batches = wtx.open_table(BATCHES).map_err(db)?;
            let next = wtx.open_table(NEXT).map_err(db)?;
            // Batches made from the exact declared identity win; name matches are only the fallback.
            let (mut by_identity, mut by_handle) = (Vec::new(), Vec::new());
            for row in batches.iter().map_err(db)? {
                let (k, v) = row.map_err(db)?;
                let info = BatchInfo::decode(v.value()).ok_or_else(|| PoolError::Db("corrupt batch row".into()))?;
                if info.matches_identity(identity) {
                    by_identity.push((k.value().to_string(), info));
                } else if info.matches_handle(handle) {
                    by_handle.push((k.value().to_string(), info));
                }
            }
            let mut candidates = if by_identity.is_empty() { by_handle } else { by_identity };
            if candidates.is_empty() {
                return Err(PoolError::NoReference);
            }
            candidates.sort_by_key(|(_, i)| i.imported_ms);
            let mut chosen = None;
            for (bid, info) in candidates {
                let used = next.get(bid.as_str()).map_err(db)?.map(|g| g.value()).unwrap_or(info.total);
                let left = info.total.saturating_sub(used);
                if left >= info.min_texts {
                    let (first, full) = info.texts_per_round(spare);
                    chosen = Some((bid, info, first.min(full.min(left))));
                    break;
                }
            }
            drop(next);
            let (bid, info, count) = chosen.ok_or(PoolError::Exhausted)?;
            let picked = take_texts(&wtx, &bid, count)?;
            record_pack(&wtx, pack_id, issued_to, &bid, count, false, now_ms)?;
            RefPack {
                pack_id,
                batch_id: info.batch_id,
                issued_ms: now_ms,
                issued_to: issued_to.to_string(),
                label: info.label,
                identity: info.identity,
                handles: info.handles,
                cutoff: info.cutoff,
                min_texts: info.min_texts,
                early_stop: info.early_stop,
                texts: picked,
            }
        };
        wtx.commit().map_err(db)?;
        Ok(pack)
    }

    /// Issue the rest of a round whose first stage (`first_pack_id`) didn't decide it: up to
    /// `min_texts + spare` texts in all, from the **same batch** (same cut-off and calibration).
    /// Only the key the first stage went to may continue it, and only once.
    pub fn continue_pack(
        &self,
        first_pack_id: [u8; 16],
        spare: u32,
        issued_to: &str,
        pack_id: [u8; 16],
        now_ms: u64,
    ) -> Result<RefPack, PoolError> {
        let wtx = self.db.begin_write().map_err(db)?;
        let pack = {
            let first_hex = hex::encode(first_pack_id);
            let rec = {
                let packs = wtx.open_table(PACKS).map_err(db)?;
                let row = packs.get(first_hex.as_str()).map_err(db)?.ok_or_else(|| PoolError::Key("unknown pack".into()))?;
                PackRecord::decode(row.value()).ok_or_else(|| PoolError::Db("corrupt pack record".into()))?
            };
            if rec.issued_to != issued_to || rec.continuation || rec.continued {
                return Err(PoolError::Key("pack not continuable by this key".into()));
            }
            let info = {
                let batches = wtx.open_table(BATCHES).map_err(db)?;
                let row = batches.get(rec.batch.as_str()).map_err(db)?.ok_or_else(|| PoolError::Db("missing batch".into()))?;
                BatchInfo::decode(row.value()).ok_or_else(|| PoolError::Db("corrupt batch row".into()))?
            };
            let want = (info.min_texts + spare).saturating_sub(rec.count);
            let used = wtx.open_table(NEXT).map_err(db)?.get(rec.batch.as_str()).map_err(db)?.map(|g| g.value()).unwrap_or(info.total);
            let left = info.total.saturating_sub(used);
            // Enough to finish the round (the spares are best-effort).
            if want == 0 || left < info.min_texts.saturating_sub(rec.count) {
                return Err(PoolError::Exhausted);
            }
            let count = want.min(left);
            let picked = take_texts(&wtx, &rec.batch, count)?;
            let mut done = rec.clone();
            done.continued = true;
            wtx.open_table(PACKS).map_err(db)?.insert(first_hex.as_str(), done.encode().as_slice()).map_err(db)?;
            record_pack(&wtx, pack_id, issued_to, &rec.batch, count, true, now_ms)?;
            RefPack {
                pack_id,
                batch_id: info.batch_id,
                issued_ms: now_ms,
                issued_to: issued_to.to_string(),
                label: info.label,
                identity: info.identity,
                handles: info.handles,
                cutoff: info.cutoff,
                min_texts: info.min_texts,
                early_stop: info.early_stop,
                texts: picked,
            }
        };
        wtx.commit().map_err(db)?;
        Ok(pack)
    }

    /// Register a gateway key by its SHA-256 (hex). Names are unique.
    pub fn add_key(&self, name: &str, key_sha256_hex: &str, now_ms: u64) -> Result<(), PoolError> {
        let wtx = self.db.begin_write().map_err(db)?;
        {
            let mut keys = wtx.open_table(KEYS).map_err(db)?;
            for row in keys.iter().map_err(db)? {
                let (_, v) = row.map_err(db)?;
                if KeyRecord::decode(v.value()).is_some_and(|k| k.name == name) {
                    return Err(PoolError::Key(format!("a key named {name} exists")));
                }
            }
            let rec = KeyRecord { name: name.to_string(), created_ms: now_ms, revoked: false };
            keys.insert(key_sha256_hex, rec.encode().as_slice()).map_err(db)?;
        }
        wtx.commit().map_err(db)
    }

    /// Revoke the key named `name`.
    pub fn revoke_key(&self, name: &str) -> Result<(), PoolError> {
        let wtx = self.db.begin_write().map_err(db)?;
        {
            let mut keys = wtx.open_table(KEYS).map_err(db)?;
            let mut hit = None;
            for row in keys.iter().map_err(db)? {
                let (k, v) = row.map_err(db)?;
                if let Some(rec) = KeyRecord::decode(v.value()).filter(|r| r.name == name) {
                    hit = Some((k.value().to_string(), rec));
                }
            }
            let (k, mut rec) = hit.ok_or_else(|| PoolError::Key(format!("no key named {name}")))?;
            rec.revoked = true;
            keys.insert(k.as_str(), rec.encode().as_slice()).map_err(db)?;
        }
        wtx.commit().map_err(db)
    }

    /// The live (non-revoked) key with this SHA-256, if any.
    pub fn key(&self, key_sha256_hex: &str) -> Result<Option<KeyRecord>, PoolError> {
        let rtx = self.db.begin_read().map_err(db)?;
        let keys = rtx.open_table(KEYS).map_err(db)?;
        Ok(keys.get(key_sha256_hex).map_err(db)?.and_then(|v| KeyRecord::decode(v.value())).filter(|r| !r.revoked))
    }

    /// Every key record (never the keys themselves).
    pub fn keys(&self) -> Result<Vec<KeyRecord>, PoolError> {
        let rtx = self.db.begin_read().map_err(db)?;
        let keys = rtx.open_table(KEYS).map_err(db)?;
        let mut out = Vec::new();
        for row in keys.iter().map_err(db)? {
            let (_, v) = row.map_err(db)?;
            out.extend(KeyRecord::decode(v.value()));
        }
        out.sort_by_key(|k| k.created_ms);
        Ok(out)
    }
}

/// Take the next `count` never-issued texts of batch `bid` and advance its cursor.
fn take_texts(wtx: &redb::WriteTransaction, bid: &str, count: u32) -> Result<Vec<ReferenceText>, PoolError> {
    let mut next = wtx.open_table(NEXT).map_err(db)?;
    let texts = wtx.open_table(TEXTS).map_err(db)?;
    let start = next.get(bid).map_err(db)?.map(|g| g.value()).ok_or_else(|| PoolError::Db("missing cursor".into()))?;
    let mut picked = Vec::with_capacity(count as usize);
    for i in start..start + count {
        let row = texts.get((bid, i)).map_err(db)?.ok_or_else(|| PoolError::Db("missing text".into()))?;
        picked.push(R(row.value()).text().ok_or_else(|| PoolError::Db("corrupt text".into()))?);
    }
    next.insert(bid, start + count).map_err(db)?;
    Ok(picked)
}

/// Who a pack went to, from which batch, how many texts; whether it is a continuation, and whether it
/// has been continued.
#[derive(Debug, Clone, PartialEq)]
struct PackRecord {
    issued_to: String,
    batch: String,
    count: u32,
    ms: u64,
    continuation: bool,
    continued: bool,
}

impl PackRecord {
    fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.u8(2);
        w.str(&self.issued_to);
        w.str(&self.batch);
        w.u32(self.count);
        w.u64(self.ms);
        w.u8(self.continuation as u8);
        w.u8(self.continued as u8);
        w.0
    }
    fn decode(b: &[u8]) -> Option<Self> {
        let mut r = R(b);
        if r.u8()? != 2 {
            return None;
        }
        Some(Self {
            issued_to: r.str()?,
            batch: r.str()?,
            count: r.u32()?,
            ms: r.u64()?,
            continuation: r.u8()? == 1,
            continued: r.u8()? == 1,
        })
    }
}

fn record_pack(
    wtx: &redb::WriteTransaction,
    pack_id: [u8; 16],
    issued_to: &str,
    batch: &str,
    count: u32,
    continuation: bool,
    now_ms: u64,
) -> Result<(), PoolError> {
    let rec = PackRecord { issued_to: issued_to.into(), batch: batch.into(), count, ms: now_ms, continuation, continued: false };
    wtx.open_table(PACKS).map_err(db)?.insert(hex::encode(pack_id).as_str(), rec.encode().as_slice()).map_err(db)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_round_takes_the_first_checkpoint_plus_spare_when_decided_early() {
        let mut info = BatchInfo {
            batch_id: [0; 16],
            label: "x".into(),
            identity: None,
            handles: vec![],
            cutoff: 2.0,
            min_texts: 24,
            early_stop: vec![],
            total: 240,
            imported_ms: 0,
        };
        assert_eq!(info.texts_per_round(4), (28, 28), "no checkpoints: a full round either way");
        info.early_stop = vec![Checkpoint { after: 8, pass_below: Some(2.0), fail_above: 2.0 }];
        assert_eq!(info.texts_per_round(4), (12, 28));
        info.early_stop[0].after = 26;
        assert_eq!(info.texts_per_round(4).0, 28, "never more than a full round");
    }
    use crate::logprob::{ChallengeRound, DEFAULT_CUTOFF};

    fn set(label: &str, n: usize, offset: u32, identity: Option<ModelIdentity>) -> ReferenceSet {
        ReferenceSet {
            model_key: label.into(),
            identity,
            texts: (0..n)
                .map(|i| {
                    let ids: Vec<u32> = (0..20).map(|k| offset + (i * 100 + k) as u32).collect();
                    ReferenceText {
                        id: format!("{label}#{i}"),
                        logprobs: ids[1..].iter().map(|&v| -((v % 7) as f32) * 0.25 - 0.01).collect(),
                        token_ids: ids,
                        band: 0.03,
                    }
                })
                .collect(),
            cutoff: DEFAULT_CUTOFF,
            min_texts: 4,
            early_stop: vec![],
        }
    }

    fn q4km() -> Option<ModelIdentity> {
        ModelIdentity::new("qwen3.5", "9b", "q4_k_m")
    }

    #[test]
    fn packs_round_trip_and_verify_only_under_the_pinned_key() {
        let pool = RefPool::open_in_memory().unwrap();
        pool.import(&set("q", 10, 0, q4km()), &[], [1; 16], 5).unwrap();
        let pack = pool.issue_pack(q4km().as_ref(), None, 2, "gw1", [9; 16], 7).unwrap();
        assert_eq!(pack.texts.len(), 6);
        assert_eq!((pack.issued_to.as_str(), pack.issued_ms, pack.identity.clone()), ("gw1", 7, q4km()));
        let key = SigningKey::from_bytes(&[3; 32]);
        let pubkey = key.verifying_key().to_bytes();
        let (bytes, sig) = sign_pack(&pack, &key);
        assert_eq!(verify_pack(&bytes, &sig, &pubkey, &pubkey).unwrap(), pack);
        // Another key, a tampered byte, or a mismatched signer are all rejected.
        let other = SigningKey::from_bytes(&[4; 32]).verifying_key().to_bytes();
        assert_eq!(verify_pack(&bytes, &sig, &pubkey, &other), Err(PackError::WrongKey));
        assert_eq!(verify_pack(&bytes, &sig, &other, &other), Err(PackError::BadSignature));
        let mut tampered = bytes.clone();
        let at = tampered.len() - 3;
        tampered[at] ^= 1;
        assert_eq!(verify_pack(&tampered, &sig, &pubkey, &pubkey), Err(PackError::BadSignature));
        // The pack is directly usable as a one-round reference set.
        let rs = pack.to_reference_set();
        assert_eq!((rs.min_texts, rs.texts.len()), (4, 6));
        assert!(ChallengeRound::new().can_complete_round(&rs));
    }

    #[test]
    fn decoding_is_strict() {
        let pool = RefPool::open_in_memory().unwrap();
        pool.import(&set("q", 6, 0, q4km()), &["Qwen3.5-9B".into()], [1; 16], 0).unwrap();
        let bytes = pool.issue_pack(None, Some("Qwen3.5-9B"), 0, "gw", [2; 16], 0).unwrap().encode();
        for cut in 0..bytes.len() {
            assert_eq!(RefPack::decode(&bytes[..cut]), Err(PackError::Malformed), "cut {cut}");
        }
        let mut long = bytes.clone();
        long.push(0);
        assert_eq!(RefPack::decode(&long), Err(PackError::Malformed));
        let mut unknown = bytes;
        unknown[0] = 3;
        assert_eq!(RefPack::decode(&unknown), Err(PackError::Malformed));
    }

    #[test]
    fn packs_carry_checkpoints_and_may_hold_less_than_a_round() {
        let pool = RefPool::open_in_memory().unwrap();
        let mut s = set("q", 12, 0, q4km());
        s.min_texts = 10;
        s.early_stop = vec![Checkpoint { after: 4, pass_below: Some(1.2), fail_above: 1.6 }, Checkpoint { after: 6, pass_below: None, fail_above: 1.5 }];
        pool.import(&s, &[], [1; 16], 0).unwrap();
        assert_eq!(pool.status().unwrap()[0].info.early_stop, s.early_stop);
        let mut p = pool.issue_pack(q4km().as_ref(), None, 0, "g", [2; 16], 0).unwrap();
        assert_eq!(p.early_stop, s.early_stop);
        // A first-stage pack: fewer texts than a full round still decodes.
        p.texts.truncate(5);
        assert_eq!(RefPack::decode(&p.encode()).unwrap(), p);
        // A version-1 pack (no checkpoint list) still decodes, with no early stopping.
        let mut v1 = p.clone();
        v1.early_stop.clear();
        let mut bytes = v1.encode();
        let at = 1 + 16 + 16 + 8 + (4 + v1.issued_to.len()) + (4 + v1.label.len());
        let id = v1.identity.as_ref().unwrap();
        let at = at + 1 + (4 + id.family.len()) + (4 + id.params.len()) + (4 + id.quant.len()) + 4 + 8 + 4;
        assert_eq!(&bytes[at..at + 4], &[0, 0, 0, 0], "the empty checkpoint list");
        bytes.drain(at..at + 4);
        bytes[0] = 1;
        assert_eq!(RefPack::decode(&bytes).unwrap(), v1);
    }

    #[test]
    fn with_checkpoints_a_round_is_issued_in_two_stages() {
        let pool = RefPool::open_in_memory().unwrap();
        let mut s = set("q", 30, 0, q4km());
        s.min_texts = 12;
        s.early_stop = vec![Checkpoint { after: 4, pass_below: Some(1.2), fail_above: 1.6 }];
        pool.import(&s, &[], [1; 16], 0).unwrap();
        let first = pool.issue_pack(q4km().as_ref(), None, 2, "gw1", [10; 16], 0).unwrap();
        assert_eq!(first.texts.len(), 6, "first checkpoint + spare");
        assert!(matches!(pool.continue_pack([10; 16], 2, "gw2", [11; 16], 0), Err(PoolError::Key(_))), "another key can't");
        let rest = pool.continue_pack([10; 16], 2, "gw1", [11; 16], 0).unwrap();
        assert_eq!(rest.texts.len(), 8, "the rest of min_texts + spare");
        assert_eq!((rest.batch_id, rest.cutoff), (first.batch_id, first.cutoff));
        assert!(rest.texts.iter().all(|t| first.texts.iter().all(|u| u.digest() != t.digest())));
        assert!(matches!(pool.continue_pack([10; 16], 2, "gw1", [12; 16], 0), Err(PoolError::Key(_))), "only once");
        assert!(matches!(pool.continue_pack([11; 16], 2, "gw1", [13; 16], 0), Err(PoolError::Key(_))), "not a continuation");
        assert!(matches!(pool.continue_pack([99; 16], 2, "gw1", [14; 16], 0), Err(PoolError::Key(_))), "unknown pack");
        // 30 − 14 = 16 left: another full round can start (6), but its continuation then needs 6 of
        // the remaining 10 — fine; a third first stage finds only 4 < 12 left.
        let second = pool.issue_pack(q4km().as_ref(), None, 2, "gw1", [20; 16], 0).unwrap();
        assert_eq!(second.texts.len(), 6);
        assert_eq!(pool.issue_pack(q4km().as_ref(), None, 2, "gw1", [21; 16], 0).unwrap_err(), PoolError::Exhausted);
    }

    #[test]
    fn every_text_is_issued_once_across_gateways_and_batches() {
        let pool = RefPool::open_in_memory().unwrap();
        pool.import(&set("a", 9, 0, q4km()), &[], [1; 16], 1).unwrap();
        pool.import(&set("b", 5, 10_000, q4km()), &[], [2; 16], 2).unwrap();
        let mut seen = std::collections::HashSet::new();
        let mut packs = 0;
        loop {
            match pool.issue_pack(q4km().as_ref(), None, 1, if packs % 2 == 0 { "gw1" } else { "gw2" }, [packs as u8; 16], 3) {
                Ok(p) => {
                    assert!(p.texts.len() >= 4);
                    for t in &p.texts {
                        assert!(seen.insert(t.digest()), "issued twice: {}", t.id);
                    }
                    packs += 1;
                }
                Err(e) => {
                    assert_eq!(e, PoolError::Exhausted);
                    break;
                }
            }
        }
        // Batch a (9 texts): 5 + 4 (fewer spares when that's all that's left); batch b (5): 5.
        assert_eq!(packs, 3);
        assert_eq!(seen.len(), 14);
        assert!(pool.status().unwrap().iter().all(|b| b.remaining == 0));
    }

    #[test]
    fn requests_match_identity_first_then_handle() {
        let pool = RefPool::open_in_memory().unwrap();
        pool.import(&set("named", 4, 0, None), &["Qwen/Qwen3.5-9B".into()], [1; 16], 1).unwrap();
        pool.import(&set("q4", 4, 5000, q4km()), &[], [2; 16], 2).unwrap();
        assert_eq!(pool.issue_pack(q4km().as_ref(), Some("Qwen/Qwen3.5-9B"), 0, "g", [1; 16], 0).unwrap().label, "q4");
        assert_eq!(pool.issue_pack(None, Some("Qwen/Qwen3.5-9B"), 0, "g", [2; 16], 0).unwrap().label, "named");
        let q40 = ModelIdentity::new("qwen3.5", "9b", "q4_0");
        assert_eq!(pool.issue_pack(q40.as_ref(), Some("other"), 0, "g", [3; 16], 0), Err(PoolError::NoReference));
        assert_eq!(pool.issue_pack(q4km().as_ref(), None, 0, "g", [4; 16], 0), Err(PoolError::Exhausted));
    }

    #[test]
    fn a_text_cannot_be_imported_twice() {
        let pool = RefPool::open_in_memory().unwrap();
        pool.import(&set("a", 4, 0, q4km()), &[], [1; 16], 0).unwrap();
        assert!(matches!(pool.import(&set("a", 4, 0, q4km()), &[], [2; 16], 0), Err(PoolError::BadBatch(m)) if m.contains("already in the pool")));
        assert!(matches!(pool.import(&set("c", 4, 9000, q4km()), &[], [1; 16], 0), Err(PoolError::BadBatch(m)) if m.contains("already imported")));
        assert!(matches!(pool.import(&set("d", 4, 7000, None), &[], [3; 16], 0), Err(PoolError::BadBatch(_))));
    }

    #[test]
    fn keys_are_stored_by_hash_and_can_be_revoked() {
        let pool = RefPool::open_in_memory().unwrap();
        pool.add_key("gw1", "aa", 1).unwrap();
        pool.add_key("gw2", "bb", 2).unwrap();
        assert!(matches!(pool.add_key("gw1", "cc", 3), Err(PoolError::Key(_))));
        assert_eq!(pool.key("aa").unwrap().unwrap().name, "gw1");
        assert_eq!(pool.key("zz").unwrap(), None);
        pool.revoke_key("gw1").unwrap();
        assert_eq!(pool.key("aa").unwrap(), None);
        assert_eq!(pool.keys().unwrap().iter().map(|k| (k.name.as_str(), k.revoked)).collect::<Vec<_>>(), vec![("gw1", true), ("gw2", false)]);
        assert!(matches!(pool.revoke_key("nope"), Err(PoolError::Key(_))));
    }

    #[test]
    fn the_pool_persists_issuance_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pool.redb");
        {
            let pool = RefPool::open(&path).unwrap();
            pool.import(&set("a", 8, 0, q4km()), &[], [1; 16], 0).unwrap();
            pool.issue_pack(q4km().as_ref(), None, 0, "g", [1; 16], 0).unwrap();
        }
        let pool = RefPool::open(&path).unwrap();
        assert_eq!(pool.status().unwrap()[0].remaining, 4);
        let p = pool.issue_pack(q4km().as_ref(), None, 0, "g", [2; 16], 0).unwrap();
        assert_eq!(p.texts[0].id, "a#4");
    }
}
