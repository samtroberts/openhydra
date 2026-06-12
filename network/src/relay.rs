//! Circuit Relay v2 helpers.
//!
//! libp2p's Circuit Relay v2 is fundamentally different from OpenHydra's
//! broken Python relay: the NATted peer opens an *outbound* connection to
//! the relay and holds it open (a "reservation"). Other peers reach the
//! NATted peer by connecting *through* the relay on that same connection.
//!
//! This module provides helpers for building relay multiaddrs and managing
//! relay reservations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use libp2p::{Multiaddr, PeerId};

/// Build a relay circuit multiaddr for reaching a peer through a relay.
///
/// Format: `<relay_addr>/p2p/<relay_peer_id>/p2p-circuit/p2p/<target_peer_id>`
///
/// This is the address a coordinator would use to connect to a NATted peer.
pub fn relay_circuit_addr(
    relay_addr: &Multiaddr,
    relay_peer_id: &PeerId,
    target_peer_id: &PeerId,
) -> Multiaddr {
    relay_addr.clone()
        .with(libp2p::multiaddr::Protocol::P2p(*relay_peer_id))
        .with(libp2p::multiaddr::Protocol::P2pCircuit)
        .with(libp2p::multiaddr::Protocol::P2p(*target_peer_id))
}

/// Build a relay reservation address (what the NATted peer listens on).
///
/// Format: `<relay_addr>/p2p/<relay_peer_id>/p2p-circuit`
pub fn relay_reservation_addr(
    relay_addr: &Multiaddr,
    relay_peer_id: &PeerId,
) -> Multiaddr {
    relay_addr.clone()
        .with(libp2p::multiaddr::Protocol::P2p(*relay_peer_id))
        .with(libp2p::multiaddr::Protocol::P2pCircuit)
}

/// Known bootstrap relay servers (production Linode nodes).
pub const BOOTSTRAP_RELAYS: &[&str] = &[
    // US (Dallas) — IPv4 (keep for legacy nodes)
    "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb",
    // EU (London) — IPv4 (keep for legacy nodes)
    "/ip4/172.105.69.49/tcp/4001/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT",
    // AP (Singapore) — IPv4 (keep for legacy nodes)
    "/ip4/172.104.164.98/tcp/4001/p2p/12D3KooWPgqZBgLZ1f94AQ7sbeyEz5UJ4jiT4d3zuQp2t61VLPZo",
    // US (Dallas) — IPv6
    "/ip6/2600:3c03::2000:68ff:fe81:55b0/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb",
    // EU (London) — IPv6
    "/ip6/2a01:7e01::2000:84ff:fec5:4520/tcp/4001/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT",
    // AP (Singapore) — IPv6
    "/ip6/2400:8901::2000:86ff:fe58:6d39/tcp/4001/p2p/12D3KooWPgqZBgLZ1f94AQ7sbeyEz5UJ4jiT4d3zuQp2t61VLPZo",
];

/// IP addresses of the production bootstrap relay servers.
/// Used by the event loop to detect when a ConnectionEstablished endpoint
/// routes through a relay's IP — such connections should NOT be classified
/// as "direct" even if the multiaddr doesn't contain `/p2p-circuit/`.
pub const BOOTSTRAP_RELAY_IPS: &[&str] = &[
    "45.79.190.172",   // US (Dallas) IPv4
    "172.105.69.49",   // EU (London) IPv4
    "172.104.164.98",  // AP (Singapore) IPv4
    "2600:3c03::2000:68ff:fe81:55b0",   // US (Dallas) IPv6
    "2a01:7e01::2000:84ff:fec5:4520",   // EU (London) IPv6
    "2400:8901::2000:86ff:fe58:6d39",   // AP (Singapore) IPv6
];

/// Check if an IP string matches a known bootstrap relay server.
pub fn is_bootstrap_relay_ip(ip: &str) -> bool {
    BOOTSTRAP_RELAY_IPS.contains(&ip)
}

// ─── WS-F F-6: relay cap + circuit migration + leech policy ──────────────────
//
// These are the SHARED policy primitives used by both the Linode bootstrap
// relays (`bootstrap_bin.rs`) and opt-in peer-relays (F-4) so abuse limits are
// identical everywhere. They are pure functions/constants — the wiring (byte
// accounting, the reservation handler, the client-side migration trigger) lives
// in the event loops; keeping the *policy* here makes it unit-testable and
// guarantees one source of truth.

/// F-6: per-circuit token budget. ~25k tokens at ~8 KB/token — generous for any
/// real session, bites only sustained abuse. Long sessions span multiple
/// circuits via seamless migration (see migration thresholds below).
pub const PER_CIRCUIT_BUDGET_BYTES: u64 = 200 * 1024 * 1024;

/// F-6 seamless migration: at this %% of the per-circuit budget, the NATted peer
/// pre-establishes a FRESH circuit in the background (no traffic moved yet).
pub const MIGRATION_PREESTABLISH_PCT: u8 = 85;
/// F-6 seamless migration: at this %% the in-flight inference is moved onto the
/// pre-established circuit, then the old circuit is closed + destroyed.
pub const MIGRATION_CUTOVER_PCT: u8 = 95;

/// F-6 leech lockout window (random within [MIN,MAX] — jitter avoids a
/// synchronized retry storm after a mass cap-out).
pub const LEECH_LOCKOUT_MIN_SECS: u64 = 15 * 60;
pub const LEECH_LOCKOUT_MAX_SECS: u64 = 30 * 60;

/// F-6: a relay at or above this many concurrent circuits is "congested" — a
/// peer that has served its lockout is then queued (admitted only as capacity
/// frees) instead of immediately re-admitted.
pub const RELAY_CONGESTION_CIRCUITS: usize = 25;

/// F-6: map a jitter fraction in `[0.0, 1.0]` to a lockout duration in
/// `[LEECH_LOCKOUT_MIN_SECS, LEECH_LOCKOUT_MAX_SECS]`. The caller supplies the
/// randomness (so this stays pure/testable); the relay binary feeds a real RNG.
pub fn leech_lockout_secs(jitter_frac: f64) -> u64 {
    let span = LEECH_LOCKOUT_MAX_SECS - LEECH_LOCKOUT_MIN_SECS;
    LEECH_LOCKOUT_MIN_SECS + (span as f64 * jitter_frac.clamp(0.0, 1.0)) as u64
}

/// F-6 admission verdict for a peer requesting a reservation/circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeechVerdict {
    /// Admit the request now.
    Admit,
    /// Still serving a lockout; reject and retry after this many seconds.
    Locked { retry_after_secs: u64 },
    /// Lockout served but the relay is congested — hold in the queue.
    Queued,
}

/// F-6 "longest-lockout-wins" admission. A capped-out peer is ALWAYS at minimum
/// timed out (even when the relay is not congested); once the timeout is served,
/// a still-congested relay queues it. `lockout_until_secs == 0` means the peer
/// has no active lockout.
pub fn leech_admit(now_secs: u64, lockout_until_secs: u64, current_circuits: usize) -> LeechVerdict {
    if now_secs < lockout_until_secs {
        return LeechVerdict::Locked { retry_after_secs: lockout_until_secs - now_secs };
    }
    if current_circuits >= RELAY_CONGESTION_CIRCUITS {
        return LeechVerdict::Queued;
    }
    LeechVerdict::Admit
}

/// F-6 circuit-migration action for a NATted peer, given bytes pushed through
/// the current circuit vs the per-circuit budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationAction {
    /// Under 85% — keep using the current circuit.
    Continue,
    /// 85–95% — pre-establish a fresh circuit in the background.
    PreEstablish,
    /// ≥95% — cut the in-flight inference over to the fresh circuit, close old.
    Cutover,
}

/// F-6: decide the migration action from current circuit usage. `budget` is
/// normally [`PER_CIRCUIT_BUDGET_BYTES`]; passed in so peer-relays with smaller
/// budgets reuse the same thresholds.
pub fn circuit_migration_action(bytes_used: u64, budget: u64) -> MigrationAction {
    if budget == 0 {
        return MigrationAction::Continue;
    }
    let pct = (bytes_used.saturating_mul(100) / budget) as u8;
    if pct >= MIGRATION_CUTOVER_PCT {
        MigrationAction::Cutover
    } else if pct >= MIGRATION_PREESTABLISH_PCT {
        MigrationAction::PreEstablish
    } else {
        MigrationAction::Continue
    }
}

/// F-4: whether this node should offer itself as a temporary peer-relay for
/// NATted peers. Only publicly-reachable nodes (static IPv4 or AutoNAT-confirmed
/// global IPv6) with spare capacity qualify; bootstrap nodes never "opt in"
/// (they are always-on relays). Circuit Relay v2 is transport-only — Noise keeps
/// payloads end-to-end encrypted, so a peer-relay only ever sees metadata.
pub fn should_offer_peer_relay(
    is_publicly_reachable: bool,
    has_spare_capacity: bool,
    is_bootstrap: bool,
) -> bool {
    !is_bootstrap && is_publicly_reachable && has_spare_capacity
}

/// F-6 wall-clock unix seconds. Used as the leech clock so the relay event loop
/// (which sets lockouts) and the [`LeechRateLimiter`] (which reads them) share
/// one comparable timebase, decoupled from the `web_time::Instant` the libp2p
/// `RateLimiter` trait passes in.
pub fn unix_secs_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// F-6: a 0.0–1.0 jitter fraction from the wall clock's sub-second nanos, for
/// [`leech_lockout_secs`]. Spreads lockout expiries so a mass cap-out doesn't
/// produce a synchronized retry storm — without pulling in an RNG dep on the
/// relay binary. (`io::Error` close + cap-out are rare, so clock entropy is
/// plenty here.)
pub fn wallclock_jitter_frac() -> f64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    (nanos % 1_000_000) as f64 / 1_000_000.0
}

/// Substring of the `io::Error` libp2p's relay raises when a circuit blows its
/// `max_circuit_bytes` budget (copy_future.rs). Used by the relay event loop to
/// distinguish a cap-out (= leech) from a normal circuit close.
pub const MAX_CIRCUIT_BYTES_ERROR: &str = "Max circuit bytes reached";

/// F-6 shared leech-lockout state: `peer -> lockout-expiry` (unix secs). The
/// relay event loop calls [`LeechTable::record_cap_out`] on a byte-cap
/// `CircuitClosed` (a peer that blew its per-circuit budget = sustained abuse);
/// [`LeechRateLimiter`] reads it to deny that peer's reservations/circuits until
/// the lockout expires. Wrap in `Arc<Mutex<_>>` to share both sides.
#[derive(Default)]
pub struct LeechTable {
    locked_until: HashMap<PeerId, u64>,
}

impl LeechTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `peer` capped out at `now_secs`; lock it for a jittered
    /// 15-30 min window. **Longest-lockout-wins** — never shortens an existing
    /// lockout. Returns the (possibly unchanged) lockout expiry.
    pub fn record_cap_out(&mut self, peer: PeerId, now_secs: u64, jitter_frac: f64) -> u64 {
        let until = now_secs.saturating_add(leech_lockout_secs(jitter_frac));
        let e = self.locked_until.entry(peer).or_insert(0);
        if until > *e {
            *e = until;
        }
        *e
    }

    /// True while `peer` is locked out (reservation/circuit should be denied).
    pub fn is_locked(&self, peer: &PeerId, now_secs: u64) -> bool {
        self.locked_until.get(peer).map_or(false, |&u| now_secs < u)
    }

    /// Seconds until `peer`'s lockout expires (0 if not locked).
    pub fn retry_after(&self, peer: &PeerId, now_secs: u64) -> u64 {
        self.locked_until
            .get(peer)
            .map_or(0, |&u| u.saturating_sub(now_secs))
    }

    /// Drop expired entries (call periodically to bound memory). Returns the
    /// number removed.
    pub fn prune(&mut self, now_secs: u64) -> usize {
        let before = self.locked_until.len();
        self.locked_until.retain(|_, &mut u| now_secs < u);
        before - self.locked_until.len()
    }

    pub fn locked_count(&self) -> usize {
        self.locked_until.len()
    }
}

/// F-6 libp2p relay [`RateLimiter`](libp2p::relay::RateLimiter) that denies
/// reservations/circuits from leech-locked peers. Plug into
/// `relay::Config.reservation_rate_limiters` AND `circuit_src_rate_limiters`
/// (two instances sharing one `Arc<Mutex<LeechTable>>`) on the Linode bootstraps
/// and opt-in peer-relays (F-4). The trait's `now: web_time::Instant` is ignored
/// — we read wall-clock unix secs so the table is updatable from the event loop.
/// Total/congestion caps stay with the relay's built-in
/// `max_circuits`/`max_reservations`; this limiter only adds the time-based
/// leech lockout on top.
pub struct LeechRateLimiter {
    table: Arc<Mutex<LeechTable>>,
}

impl LeechRateLimiter {
    pub fn new(table: Arc<Mutex<LeechTable>>) -> Self {
        Self { table }
    }
}

impl libp2p::relay::RateLimiter for LeechRateLimiter {
    fn try_next(&mut self, peer: PeerId, _addr: &Multiaddr, _now: web_time::Instant) -> bool {
        let now_secs = unix_secs_now();
        match self.table.lock() {
            Ok(t) => !t.is_locked(&peer, now_secs),
            // Poisoned lock: fail OPEN — never block legitimate peers because a
            // panic poisoned the mutex; abuse is still bounded by the built-in caps.
            Err(_) => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relay_circuit_addr_format() {
        let relay_addr: Multiaddr = "/ip4/45.79.190.172/tcp/4001".parse().unwrap();
        let relay_peer = PeerId::random();
        let target_peer = PeerId::random();

        let circuit = relay_circuit_addr(&relay_addr, &relay_peer, &target_peer);
        let s = circuit.to_string();

        assert!(s.contains("/p2p-circuit/p2p/"));
        assert!(s.starts_with("/ip4/45.79.190.172/tcp/4001/p2p/"));
    }

    #[test]
    fn test_relay_reservation_addr_format() {
        let relay_addr: Multiaddr = "/ip4/172.105.69.49/tcp/4001".parse().unwrap();
        let relay_peer = PeerId::random();

        let reservation = relay_reservation_addr(&relay_addr, &relay_peer);
        let s = reservation.to_string();

        assert!(s.ends_with("/p2p-circuit"));
        assert!(s.starts_with("/ip4/172.105.69.49/tcp/4001/p2p/"));
    }

    // ── WS-F F-6 / F-4 policy ────────────────────────────────────────────

    #[test]
    fn test_leech_lockout_secs_jitter_range() {
        // Endpoints + midpoint, and out-of-range fractions clamp.
        assert_eq!(leech_lockout_secs(0.0), LEECH_LOCKOUT_MIN_SECS);
        assert_eq!(leech_lockout_secs(1.0), LEECH_LOCKOUT_MAX_SECS);
        let mid = leech_lockout_secs(0.5);
        assert!(mid > LEECH_LOCKOUT_MIN_SECS && mid < LEECH_LOCKOUT_MAX_SECS);
        assert_eq!(leech_lockout_secs(-1.0), LEECH_LOCKOUT_MIN_SECS);
        assert_eq!(leech_lockout_secs(2.0), LEECH_LOCKOUT_MAX_SECS);
        // Always within the 15–30 min window.
        for i in 0..=10 {
            let s = leech_lockout_secs(i as f64 / 10.0);
            assert!((LEECH_LOCKOUT_MIN_SECS..=LEECH_LOCKOUT_MAX_SECS).contains(&s));
        }
    }

    #[test]
    fn test_leech_admit_longest_lockout_wins() {
        // Still locked out → Locked with remaining time, regardless of congestion.
        assert_eq!(
            leech_admit(100, 250, 0),
            LeechVerdict::Locked { retry_after_secs: 150 },
        );
        assert_eq!(
            leech_admit(100, 250, 999),
            LeechVerdict::Locked { retry_after_secs: 150 },
        );
        // Lockout served, relay NOT congested → admit.
        assert_eq!(leech_admit(300, 250, 10), LeechVerdict::Admit);
        // Lockout served, relay congested (≥25) → queued (longer lockout wins).
        assert_eq!(leech_admit(300, 250, RELAY_CONGESTION_CIRCUITS), LeechVerdict::Queued);
        assert_eq!(leech_admit(300, 250, 100), LeechVerdict::Queued);
        // No prior lockout (until=0), uncongested → admit immediately.
        assert_eq!(leech_admit(300, 0, 0), LeechVerdict::Admit);
    }

    #[test]
    fn test_circuit_migration_thresholds() {
        let b = PER_CIRCUIT_BUDGET_BYTES;
        assert_eq!(circuit_migration_action(0, b), MigrationAction::Continue);
        assert_eq!(circuit_migration_action(b / 2, b), MigrationAction::Continue);
        // 85% → pre-establish; 95% → cutover.
        assert_eq!(circuit_migration_action(b * 85 / 100, b), MigrationAction::PreEstablish);
        assert_eq!(circuit_migration_action(b * 90 / 100, b), MigrationAction::PreEstablish);
        assert_eq!(circuit_migration_action(b * 95 / 100, b), MigrationAction::Cutover);
        assert_eq!(circuit_migration_action(b, b), MigrationAction::Cutover);
        assert_eq!(circuit_migration_action(b * 2, b), MigrationAction::Cutover);
        // Degenerate budget never triggers migration.
        assert_eq!(circuit_migration_action(1_000, 0), MigrationAction::Continue);
    }

    #[test]
    fn test_leech_table_lockout_lifecycle() {
        let peer = PeerId::random();
        let mut t = LeechTable::new();
        assert!(!t.is_locked(&peer, 1000));
        // Cap out at t=1000 with min jitter → locked for 15 min (900s).
        let until = t.record_cap_out(peer, 1000, 0.0);
        assert_eq!(until, 1000 + LEECH_LOCKOUT_MIN_SECS);
        assert!(t.is_locked(&peer, 1000));
        assert!(t.is_locked(&peer, until - 1));
        assert!(!t.is_locked(&peer, until)); // expired exactly at `until`
        assert_eq!(t.retry_after(&peer, 1000), LEECH_LOCKOUT_MIN_SECS);
        assert_eq!(t.retry_after(&peer, until + 50), 0);
    }

    #[test]
    fn test_leech_table_longest_lockout_wins() {
        let peer = PeerId::random();
        let mut t = LeechTable::new();
        let long = t.record_cap_out(peer, 1000, 1.0);  // +30 min
        // A later cap-out with a SHORTER window must not shorten the lockout.
        let after = t.record_cap_out(peer, 1100, 0.0);  // would be 1100+900 < long
        assert_eq!(after, long, "shorter re-lock must not win");
        assert!(after > 1100 + LEECH_LOCKOUT_MIN_SECS);
    }

    #[test]
    fn test_leech_table_prune() {
        let (p1, p2) = (PeerId::random(), PeerId::random());
        let mut t = LeechTable::new();
        t.record_cap_out(p1, 1000, 0.0);  // expires 1900
        t.record_cap_out(p2, 1000, 1.0);  // expires 2800
        assert_eq!(t.locked_count(), 2);
        assert_eq!(t.prune(2000), 1);     // p1 expired, p2 still locked
        assert_eq!(t.locked_count(), 1);
        assert!(t.is_locked(&p2, 2000) && !t.is_locked(&p1, 2000));
    }

    #[test]
    fn test_should_offer_peer_relay() {
        // Publicly reachable + spare capacity + not bootstrap → offer.
        assert!(should_offer_peer_relay(true, true, false));
        // Any disqualifier blocks it.
        assert!(!should_offer_peer_relay(false, true, false));  // NAT'd
        assert!(!should_offer_peer_relay(true, false, false));  // no capacity
        assert!(!should_offer_peer_relay(true, true, true));    // bootstrap (always-on, not opt-in)
    }
}
