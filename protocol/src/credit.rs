// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Credit / give-to-get accounting (protocol.md §6) — M2.3.
//!
//! **"Priority, not access."** This never *blocks* a peer — it only *throttles* one that
//! consumes far more than it contributes. A node tracks, per peer, a decayed give/take
//! balance derived from the co-signed receipts (M2.1):
//!
//! ```text
//! balance = STARTER_GRANT + earned(served tokens) − spent(consumed tokens)   [decayed]
//! ```
//!
//! and maps it to a serve-rate cap in `[RATE_FLOOR, 1.0]`: a contributor serves at full
//! speed, a leecher is throttled *toward* the floor under contention but is **never cut
//! off** (the floor is non-zero). A one-time [`STARTER_GRANT`] lets a brand-new peer
//! bootstrap before it has served anything.
//!
//! **Anti-collusion.** Credit *earned* from any single counterparty is capped
//! ([`PER_COUNTERPARTY_CAP`]), so a ring minting mutual receipts can't manufacture
//! unbounded credit — only genuinely *diverse* contribution earns full credit.
//!
//! Tokenomics were removed (`7316de0`); this is a "no-crypto" give-to-get reputation of
//! contribution, not money. Pure & clock-injected (every method takes `now_ms`), exactly
//! like [`crate::verify::ReputationTracker`] — the live store/gossip wiring lands on top.
//!
//! NOTE: [`to_bytes`](CreditAccount::to_bytes) is a storage codec, **not** a signed or
//! canonical preimage (map order is unspecified) — credit blobs are never signed.

use std::collections::HashMap;

/// One-time credit (in tokens) extended to a never-seen peer so it can bootstrap before
/// it has served anything. A pure leecher gets roughly this much at full speed, then
/// throttles. Does not decay (a peer can always eventually re-bootstrap).
pub const STARTER_GRANT: f64 = 5_000.0;

/// Maximum credit a peer may *earn* from a single counterparty. Blunts collusion rings:
/// two peers minting mutual receipts can each manufacture at most this much.
pub const PER_COUNTERPARTY_CAP: f64 = 20_000.0;

/// The throttle floor — the minimum serve rate (fraction of full speed). Non-zero, so a
/// leecher is slowed, never blocked ("priority, not access").
pub const RATE_FLOOR: f64 = 0.1;

/// Token deficit at which the serve rate reaches [`RATE_FLOOR`] (linear from a balance of
/// 0 at full speed down to the floor).
pub const DEFICIT_SPAN: f64 = 50_000.0;

/// Default give/take half-life (~30 days) — slower than the reputation half-life (~7d), so
/// the balance tracks medium-term contribution rather than recent behaviour.
pub const DEFAULT_HALF_LIFE_MS: u64 = 30 * 24 * 60 * 60 * 1000;

/// One peer's give/take credit account, from the local node's view. Earned/spent tallies
/// decay toward zero so neither old contribution nor old leeching persists forever.
#[derive(Debug, Clone)]
pub struct CreditAccount {
    /// Decayed credit earned (tokens this peer served), **after** per-counterparty capping.
    earned: f64,
    /// Decayed credit spent (tokens this peer consumed).
    spent: f64,
    /// Per-counterparty served tally (decayed) — drives the per-counterparty earn cap so
    /// collusion can't manufacture credit.
    by_counterparty: HashMap<String, f64>,
    /// Unix-ms the tallies are current as of.
    last_update_ms: u64,
    /// Half-life (ms) of the exponential decay toward zero. 0 = no decay.
    half_life_ms: u64,
}

impl CreditAccount {
    /// A fresh account as of `now_ms`, with the default half-life.
    pub fn new(now_ms: u64) -> Self {
        Self::with_half_life(now_ms, DEFAULT_HALF_LIFE_MS)
    }

    /// As [`new`](Self::new) but with an explicit decay half-life (`0` = no decay).
    pub fn with_half_life(now_ms: u64, half_life_ms: u64) -> Self {
        Self {
            earned: 0.0,
            spent: 0.0,
            by_counterparty: HashMap::new(),
            last_update_ms: now_ms,
            half_life_ms,
        }
    }

    /// Decay multiplier `0.5^(elapsed / half_life)` for reading a tally as of `now_ms`
    /// without mutating. Clock skew (earlier `now_ms`) yields no decay.
    fn decay_factor(&self, now_ms: u64) -> f64 {
        if self.half_life_ms == 0 {
            return 1.0;
        }
        let elapsed = now_ms.saturating_sub(self.last_update_ms) as f64;
        if elapsed <= 0.0 {
            return 1.0;
        }
        0.5_f64.powf(elapsed / self.half_life_ms as f64)
    }

    /// Roll every tally forward to `now_ms` in place; prune counterparties that decay to
    /// negligible. Called before each mutating record so accrual acts on current values.
    fn decay_to(&mut self, now_ms: u64) {
        let f = self.decay_factor(now_ms);
        self.last_update_ms = self.last_update_ms.max(now_ms);
        if f >= 1.0 {
            return;
        }
        self.earned *= f;
        self.spent *= f;
        self.by_counterparty.retain(|_, v| {
            *v *= f;
            *v > 1.0 // drop sub-token residue to bound the map
        });
    }

    /// Record that this peer **served** `tokens` to `counterparty` (it earns credit). The
    /// contribution to `earned` is capped per counterparty, so a collusion partner can
    /// only ever add up to [`PER_COUNTERPARTY_CAP`].
    pub fn record_served(&mut self, counterparty: &str, tokens: u64, now_ms: u64) {
        self.decay_to(now_ms);
        let entry = self.by_counterparty.entry(counterparty.to_string()).or_insert(0.0);
        let before = entry.min(PER_COUNTERPARTY_CAP);
        *entry += tokens as f64;
        let after = entry.min(PER_COUNTERPARTY_CAP);
        self.earned += after - before; // only the un-capped marginal counts toward credit
    }

    /// Record that this peer **consumed** `tokens` (it spends credit).
    pub fn record_consumed(&mut self, tokens: u64, now_ms: u64) {
        self.decay_to(now_ms);
        self.spent += tokens as f64;
    }

    /// The decayed give/take balance as of `now_ms`: `STARTER_GRANT + earned − spent`.
    /// Read-only (does not mutate). Positive ⇒ a contributor; deeply negative ⇒ a leecher.
    pub fn balance(&self, now_ms: u64) -> f64 {
        let f = self.decay_factor(now_ms);
        STARTER_GRANT + self.earned * f - self.spent * f
    }

    /// The serve-rate cap in `[RATE_FLOOR, 1.0]` as of `now_ms`. A non-negative balance ⇒
    /// full speed; a deficit scales linearly to the floor over [`DEFICIT_SPAN`]; never
    /// below the floor (throttle, never block).
    pub fn rate_cap(&self, now_ms: u64) -> f64 {
        let bal = self.balance(now_ms);
        if bal >= 0.0 {
            return 1.0;
        }
        let t = (-bal / DEFICIT_SPAN).clamp(0.0, 1.0);
        (1.0 - t * (1.0 - RATE_FLOOR)).max(RATE_FLOOR)
    }

    /// The timestamp the raw tallies are current as of.
    pub fn last_update_ms(&self) -> u64 {
        self.last_update_ms
    }

    /// Serialize for the persistent store (M2.3 wiring): `earned:f64 spent:f64
    /// last_update_ms:u64 half_life_ms:u64 n:u32 [cp_len:u16 cp[..] amount:f64]*`, all
    /// little-endian. A storage codec, not a signed preimage (map order is unspecified).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(8 + 8 + 8 + 8 + 4 + self.by_counterparty.len() * 16);
        b.extend_from_slice(&self.earned.to_le_bytes());
        b.extend_from_slice(&self.spent.to_le_bytes());
        b.extend_from_slice(&self.last_update_ms.to_le_bytes());
        b.extend_from_slice(&self.half_life_ms.to_le_bytes());
        b.extend_from_slice(&(self.by_counterparty.len() as u32).to_le_bytes());
        for (cp, amt) in &self.by_counterparty {
            let cb = cp.as_bytes();
            b.extend_from_slice(&(cb.len() as u16).to_le_bytes());
            b.extend_from_slice(cb);
            b.extend_from_slice(&amt.to_le_bytes());
        }
        b
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes). Bounds-checked; `Err` on a malformed
    /// or truncated blob.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut c = Cur { d: data, p: 0 };
        let earned = f64::from_le_bytes(c.arr::<8>()?);
        let spent = f64::from_le_bytes(c.arr::<8>()?);
        let last_update_ms = u64::from_le_bytes(c.arr::<8>()?);
        let half_life_ms = u64::from_le_bytes(c.arr::<8>()?);
        let n = u32::from_le_bytes(c.arr::<4>()?) as usize;
        let mut by_counterparty = HashMap::with_capacity(n);
        for _ in 0..n {
            let len = u16::from_le_bytes(c.arr::<2>()?) as usize;
            let cp = std::str::from_utf8(c.take(len)?)
                .map_err(|e| format!("bad counterparty utf8: {e}"))?
                .to_string();
            let amt = f64::from_le_bytes(c.arr::<8>()?);
            by_counterparty.insert(cp, amt);
        }
        if c.remaining() != 0 {
            return Err(format!("credit blob has {} trailing bytes", c.remaining()));
        }
        Ok(Self { earned, spent, by_counterparty, last_update_ms, half_life_ms })
    }
}

/// Minimal bounds-checked reader for [`CreditAccount::from_bytes`].
struct Cur<'a> {
    d: &'a [u8],
    p: usize,
}
impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.p.checked_add(n).ok_or("length overflow")?;
        if end > self.d.len() {
            return Err(format!("credit blob truncated at offset {}", self.p));
        }
        let s = &self.d[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N], String> {
        Ok(self.take(N)?.try_into().expect("take returns exactly N"))
    }
    fn remaining(&self) -> usize {
        self.d.len() - self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_MS: u64 = 24 * 60 * 60 * 1000;

    // ── the M2.3 sim harness (exit test) ──

    #[test]
    fn contributor_full_speed_leecher_throttled_to_floor() {
        let now = 0;
        // Contributor: serves 60k across 3 distinct peers, consumes 40k → positive balance.
        let mut good = CreditAccount::new(now);
        good.record_served("a", 20_000, now);
        good.record_served("b", 20_000, now);
        good.record_served("c", 20_000, now);
        good.record_consumed(40_000, now);
        assert!(good.balance(now) > 0.0);
        assert_eq!(good.rate_cap(now), 1.0, "a contributor serves at full speed");

        // Leecher: serves nothing, consumes 60k → deep deficit.
        let mut leech = CreditAccount::new(now);
        leech.record_consumed(60_000, now);
        let cap = leech.rate_cap(now);
        assert!((cap - RATE_FLOOR).abs() < 1e-9, "deep deficit → floor, got {cap}");
        assert!(cap >= RATE_FLOOR, "never below the floor (throttle, never block)");
    }

    #[test]
    fn starter_grant_lets_a_fresh_peer_bootstrap() {
        let now = 0;
        assert_eq!(CreditAccount::new(now).rate_cap(now), 1.0, "fresh peer starts at full speed");
        // Consuming exactly the grant → balance 0 → still full; one token over → throttling begins.
        let mut p = CreditAccount::new(now);
        p.record_consumed(STARTER_GRANT as u64, now);
        assert_eq!(p.rate_cap(now), 1.0);
        p.record_consumed(1, now);
        assert!(p.rate_cap(now) < 1.0, "consuming past the grant without serving begins to throttle");
    }

    #[test]
    fn collusion_ring_earns_only_capped_credit() {
        let now = 0;
        // A mints 1,000,000 tokens of "service" to its single ring partner B, while
        // consuming 200k from the real network.
        let mut a = CreditAccount::new(now);
        a.record_served("B", 1_000_000, now);
        a.record_consumed(200_000, now);
        // earned is capped at PER_COUNTERPARTY_CAP, so the balance stays deeply negative.
        assert!(a.balance(now) < 0.0, "a 2-peer collusion ring can't manufacture usable credit");
        assert!((a.rate_cap(now) - RATE_FLOOR).abs() < 1e-9);
    }

    #[test]
    fn diverse_contribution_beats_a_ring_for_the_same_total() {
        let now = 0;
        // Same 1,000,000 tokens served, but spread across 50 distinct peers → full credit.
        let mut diverse = CreditAccount::new(now);
        for i in 0..50 {
            diverse.record_served(&format!("p{i}"), 20_000, now);
        }
        diverse.record_consumed(200_000, now);
        assert!(diverse.balance(now) > 0.0);
        assert_eq!(diverse.rate_cap(now), 1.0, "diverse contribution earns full credit");
    }

    #[test]
    fn decay_forgives_a_stale_deficit() {
        let mut leech = CreditAccount::with_half_life(0, 30 * DAY_MS);
        leech.record_consumed(60_000, 0);
        assert!((leech.rate_cap(0) - RATE_FLOOR).abs() < 1e-9, "throttled while the deficit is fresh");
        // Idle for ~10 half-lives: the deficit decays away → balance → grant → un-throttled.
        let far = 30 * DAY_MS * 10;
        assert_eq!(leech.rate_cap(far), 1.0, "a stale deficit decays away (old leeching is forgiven)");
    }

    #[test]
    fn decay_does_not_let_a_contributor_coast_forever() {
        let mut good = CreditAccount::with_half_life(0, 30 * DAY_MS);
        for i in 0..50 {
            good.record_served(&format!("p{i}"), 20_000, 0);
        }
        assert!(good.balance(0) > 500_000.0);
        // Far in the future with no new activity, earned decays → balance → the grant.
        let far = 30 * DAY_MS * 20;
        assert!((good.balance(far) - STARTER_GRANT).abs() < 1.0, "earned credit can't be coasted on forever");
    }

    #[test]
    fn record_acts_on_the_decayed_value_not_a_stale_one() {
        // Earn a lot, let it decay a half-life, then a fresh consume must spend against the
        // decayed (smaller) balance, not the stale peak.
        let mut a = CreditAccount::with_half_life(0, 30 * DAY_MS);
        a.record_served("x", 20_000, 0);
        let decayed_earned_balance = a.balance(30 * DAY_MS); // grant + 20000*0.5
        a.record_consumed(0, 30 * DAY_MS); // forces decay_to without changing spent
        assert!((a.balance(30 * DAY_MS) - decayed_earned_balance).abs() < 1.0);
    }

    #[test]
    fn clock_skew_never_decays_backward() {
        let mut a = CreditAccount::with_half_life(100, 30 * DAY_MS);
        a.record_served("x", 10_000, 100);
        let b = a.balance(100);
        assert_eq!(a.balance(50), b, "a past timestamp must not grow the balance");
    }

    // ── serialization ──

    #[test]
    fn bytes_roundtrip_preserves_state() {
        let mut a = CreditAccount::with_half_life(1_000, 10 * DAY_MS);
        a.record_served("alice", 12_345, 1_000);
        a.record_served("bob", 67_890, 1_000);
        a.record_consumed(5_000, 1_000);
        let back = CreditAccount::from_bytes(&a.to_bytes()).unwrap();
        // Same balance now and after a day → the full decay state round-tripped.
        assert_eq!(back.balance(1_000).to_bits(), a.balance(1_000).to_bits());
        assert_eq!(back.balance(1_000 + DAY_MS).to_bits(), a.balance(1_000 + DAY_MS).to_bits());
        assert_eq!(back.last_update_ms(), a.last_update_ms());
    }

    #[test]
    fn from_bytes_rejects_malformed() {
        assert!(CreditAccount::from_bytes(b"short").is_err());
        let mut blob = CreditAccount::new(0).to_bytes();
        blob.push(0xFF); // trailing byte
        assert!(CreditAccount::from_bytes(&blob).is_err());
    }
}
