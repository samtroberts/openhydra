// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Gateway ingress DoS rate-limit — per-identity token bucket + concurrency cap.
//!
//! Distinct from the M2.3 credit throttle (per-*peer*, give/take-weighted, on the provider)
//! and the AUP floor (size/content): this guards the consumer's **own gateway** against an
//! abusive caller. Two levers, both default-off (a mechanism, opt-in):
//! - **concurrency** (`max_inflight`) — the primary lever, since each completion ties up a
//!   generation for *seconds*;
//! - **rps + burst** — a secondary token-bucket guard against cheap-request floods.
//!
//! Identity keying is the caller's concern (operator API key → socket IP → never a
//! client-spoofable header); this module just rate-limits whatever opaque identity string it
//! is handed. Clock is injected (`now_ms`) so the policy is pure and unit-testable.
//!
//! **Bounded memory.** A full token bucket is indistinguishable from a fresh one, so idle
//! identities (full bucket, no in-flight) are evicted when the map reaches `max_tracked` —
//! the limiter can't be turned into its own memory-DoS by cycling identities. Memory is
//! `O(currently-throttled identities)`, capped by `max_tracked`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Rate-limit configuration. `rps`/`max_inflight` of 0 mean "that lever off"; the limiter is
/// active if either is set.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Sustained requests/sec per identity (0 = off).
    pub rps: f64,
    /// Token-bucket capacity — the allowed burst above the sustained rate (0 → defaults to
    /// `rps`, clamped to ≥1, so `rps` alone yields a sane bucket).
    pub burst: f64,
    /// Max concurrent in-flight requests per identity (0 = off — the primary lever).
    pub max_inflight: u32,
    /// Hard cap on distinct tracked identities (memory bound). Clamped to ≥1 when active.
    pub max_tracked: usize,
}

impl RateLimitConfig {
    /// A fully-disabled config (the default — opt in via flags).
    pub fn disabled() -> Self {
        Self { rps: 0.0, burst: 0.0, max_inflight: 0, max_tracked: 0 }
    }

    /// Whether any lever is configured (callers skip the limiter entirely when `false`).
    pub fn is_active(&self) -> bool {
        self.rps > 0.0 || self.max_inflight > 0
    }

    fn effective_burst(&self) -> f64 {
        if self.burst > 0.0 {
            self.burst
        } else {
            self.rps.max(1.0)
        }
    }

    fn effective_max_tracked(&self) -> usize {
        self.max_tracked.max(1)
    }
}

/// A lazily-refilled token bucket. Pure; the clock is passed in.
struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_per_sec: f64,
    last_ms: u64,
}

impl TokenBucket {
    fn new(rps: f64, burst: f64, now_ms: u64) -> Self {
        Self { tokens: burst, capacity: burst, refill_per_sec: rps, last_ms: now_ms }
    }

    fn refill(&mut self, now_ms: u64) {
        if now_ms > self.last_ms {
            let secs = (now_ms - self.last_ms) as f64 / 1000.0;
            self.tokens = (self.tokens + secs * self.refill_per_sec).min(self.capacity);
            self.last_ms = now_ms;
        }
    }

    /// Take one token if available (refilling first). `true` = admitted.
    fn try_take(&mut self, now_ms: u64) -> bool {
        self.refill(now_ms);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Would the bucket be full as of `now_ms`? A full bucket carries no state worth keeping,
    /// so a full + idle entry is safe to evict.
    fn would_be_full(&self, now_ms: u64) -> bool {
        let secs = now_ms.saturating_sub(self.last_ms) as f64 / 1000.0;
        self.tokens + secs * self.refill_per_sec >= self.capacity
    }
}

struct Entry {
    bucket: TokenBucket,
    in_flight: u32,
    last_ms: u64,
}

/// Per-identity ingress limiter.
pub struct RateLimiter {
    cfg: RateLimitConfig,
    entries: Mutex<HashMap<String, Entry>>,
}

/// Returned when a request is shed; the caller answers `429`.
#[derive(Debug)]
pub struct RateLimitReject;

/// Held for the lifetime of an admitted request; decrements the identity's in-flight count on
/// drop — including on cancellation, so a dropped request future never leaks a slot.
pub struct InFlightGuard {
    limiter: Arc<RateLimiter>,
    identity: String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if let Ok(mut map) = self.limiter.entries.lock() {
            if let Some(entry) = map.get_mut(&self.identity) {
                entry.in_flight = entry.in_flight.saturating_sub(1);
            }
        }
    }
}

impl RateLimiter {
    pub fn new(cfg: RateLimitConfig) -> Self {
        Self { cfg, entries: Mutex::new(HashMap::new()) }
    }

    pub fn is_active(&self) -> bool {
        self.cfg.is_active()
    }

    /// Admit or shed a request from `identity`. On admit, returns a guard that releases the
    /// in-flight slot when dropped. Enforces the concurrency cap first (no state change), then
    /// the token bucket (consumes a token only on success).
    pub fn try_acquire(
        self: &Arc<Self>,
        identity: &str,
        now_ms: u64,
    ) -> Result<InFlightGuard, RateLimitReject> {
        let mut map = self.entries.lock().map_err(|_| RateLimitReject)?;

        if !map.contains_key(identity) {
            let cap = self.cfg.effective_max_tracked();
            if map.len() >= cap {
                // Reclaim idle identities (full bucket, nothing in flight) — lossless, since a
                // full bucket == a fresh one.
                map.retain(|_, e| e.in_flight > 0 || !e.bucket.would_be_full(now_ms));
            }
            if map.len() >= cap {
                // Still saturated with active/throttled identities → shed the newcomer rather
                // than evict an entry that's mid-request or being throttled.
                return Err(RateLimitReject);
            }
            map.insert(
                identity.to_string(),
                Entry {
                    bucket: TokenBucket::new(self.cfg.rps, self.cfg.effective_burst(), now_ms),
                    in_flight: 0,
                    last_ms: now_ms,
                },
            );
        }

        let entry = map.get_mut(identity).expect("inserted above");
        entry.last_ms = now_ms;

        if self.cfg.max_inflight > 0 && entry.in_flight >= self.cfg.max_inflight {
            return Err(RateLimitReject);
        }
        if self.cfg.rps > 0.0 && !entry.bucket.try_take(now_ms) {
            return Err(RateLimitReject);
        }
        entry.in_flight += 1;

        Ok(InFlightGuard { limiter: Arc::clone(self), identity: identity.to_string() })
    }

    #[cfg(test)]
    fn tracked(&self) -> usize {
        self.entries.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limiter(cfg: RateLimitConfig) -> Arc<RateLimiter> {
        Arc::new(RateLimiter::new(cfg))
    }

    #[test]
    fn disabled_is_inactive() {
        assert!(!RateLimitConfig::disabled().is_active());
    }

    #[test]
    fn rps_burst_then_throttle_then_refill() {
        // 2 rps, burst 3: 3 immediate admits, 4th shed; after ~0.5s one token refills.
        let rl = limiter(RateLimitConfig { rps: 2.0, burst: 3.0, max_inflight: 0, max_tracked: 100 });
        let t = 1_000_000;
        for _ in 0..3 {
            assert!(rl.try_acquire("a", t).is_ok());
        }
        assert!(rl.try_acquire("a", t).is_err(), "burst exhausted → shed");
        // 500ms later → +1 token (2/sec).
        assert!(rl.try_acquire("a", t + 500).is_ok());
        assert!(rl.try_acquire("a", t + 500).is_err());
    }

    #[test]
    fn distinct_identities_have_independent_buckets() {
        let rl = limiter(RateLimitConfig { rps: 1.0, burst: 1.0, max_inflight: 0, max_tracked: 100 });
        assert!(rl.try_acquire("a", 0).is_ok());
        assert!(rl.try_acquire("a", 0).is_err());
        assert!(rl.try_acquire("b", 0).is_ok(), "b is independent of a");
    }

    #[test]
    fn concurrency_cap_holds_and_releases_on_drop() {
        let rl = limiter(RateLimitConfig { rps: 0.0, burst: 0.0, max_inflight: 2, max_tracked: 100 });
        let g1 = rl.try_acquire("a", 0).unwrap();
        let _g2 = rl.try_acquire("a", 0).unwrap();
        assert!(rl.try_acquire("a", 0).is_err(), "3rd concurrent over cap of 2");
        drop(g1); // a slot frees
        assert!(rl.try_acquire("a", 0).is_ok());
    }

    #[test]
    fn idle_full_buckets_are_evicted_at_cap() {
        // Cap of 2 distinct identities. Two idle ones fill it; a third triggers reclaim of the
        // (full, no-in-flight) idle entries, so it is admitted and the map stays bounded.
        let rl = limiter(RateLimitConfig { rps: 100.0, burst: 100.0, max_inflight: 0, max_tracked: 2 });
        assert!(rl.try_acquire("a", 0).is_ok());
        assert!(rl.try_acquire("b", 0).is_ok());
        assert_eq!(rl.tracked(), 2);
        // Later, both a and b have refilled to full (idle) → evicted to make room for c.
        assert!(rl.try_acquire("c", 10_000).is_ok());
        assert!(rl.tracked() <= 2, "map stayed bounded at the cap");
    }

    #[test]
    fn newcomer_shed_when_cap_full_of_active_identities() {
        // Cap of 1, and the single slot is held by an in-flight request → a different identity
        // is shed rather than evicting the active one (no memory blow-up, no slot leak).
        let rl = limiter(RateLimitConfig { rps: 0.0, burst: 0.0, max_inflight: 5, max_tracked: 1 });
        let _g = rl.try_acquire("a", 0).unwrap(); // a is in-flight
        assert!(rl.try_acquire("b", 0).is_err(), "cap full of active identity → shed newcomer");
        assert_eq!(rl.tracked(), 1);
    }
}
