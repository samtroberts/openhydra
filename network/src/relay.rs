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

/// The relay peer ids embedded in `BOOTSTRAP_RELAYS` — the always-trusted relay
/// hops, regardless of runtime `--bootstrap` config (D-S5).
pub fn bootstrap_relay_peer_ids() -> Vec<PeerId> {
    use libp2p::multiaddr::Protocol;
    BOOTSTRAP_RELAYS
        .iter()
        .filter_map(|s| s.parse::<Multiaddr>().ok())
        .filter_map(|ma| {
            ma.iter().find_map(|p| match p {
                Protocol::P2p(pid) => Some(pid),
                _ => None,
            })
        })
        .collect()
}

/// D-S5: decide whether a record's self-declared `relay_address` is safe to
/// inject into the Kademlia routing table for `expected_pid`.
///
/// A signed record proves *authorship*, not that the declared address is honest.
/// If we blindly `add_address(pid, declared)` then we — and, via replication,
/// every peer that caches the record — would dial whatever host the record
/// names, turning the DHT into a reflection/amplification vector aimed at an
/// arbitrary victim `ip:port`. We therefore accept ONLY a relay-**circuit**
/// address whose relay hop is a relay this node **actually uses** — identified by
/// its peer id against `trusted_relay_pids` (the runtime `--bootstrap` set ∪ the
/// hardcoded `BOOTSTRAP_RELAYS`) — and whose circuit target (when named) matches
/// the record's peer id. Keying on the relay's *peer id* rather than a static IP
/// list is what lets a runtime-configured relay like netcup work (it's in
/// `--bootstrap` but not the hardcoded IPs) while still rejecting a circuit whose
/// hop is an unknown/attacker relay. A legitimate NAT'd provider always
/// advertises exactly this (it reserved on a relay we also use).
pub fn safe_injectable_circuit_addr(
    relay_address: &str,
    expected_pid: &PeerId,
    trusted_relay_pids: &std::collections::HashSet<PeerId>,
) -> Option<Multiaddr> {
    use libp2p::multiaddr::Protocol;
    let ma: Multiaddr = relay_address.parse().ok()?;

    let mut seen_circuit = false;
    let mut relay_hop_pid: Option<PeerId> = None;
    let mut target_after_circuit: Option<PeerId> = None;
    for p in ma.iter() {
        match p {
            // The relay hop peer id is the /p2p/ *before* /p2p-circuit.
            Protocol::P2p(pid) if !seen_circuit => relay_hop_pid = Some(pid),
            Protocol::P2pCircuit => seen_circuit = true,
            // The target peer id is the /p2p/ *after* /p2p-circuit.
            Protocol::P2p(pid) if seen_circuit => target_after_circuit = Some(pid),
            _ => {}
        }
    }

    // Must be circuit-scoped through a relay whose peer id we trust at runtime.
    if !seen_circuit {
        return None;
    }
    match relay_hop_pid {
        Some(pid) if trusted_relay_pids.contains(&pid) => {}
        _ => return None,
    }
    // If the circuit names a target, it must be this peer — no seeding an
    // address that dials on behalf of a different identity.
    if let Some(t) = target_after_circuit {
        if &t != expected_pid {
            return None;
        }
    }
    Some(ma)
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

/// F-6 client-side circuit-migration monitor (the DECISION core; the hot-path
/// byte-attribution + the transparent `tensor_stream` cutover are wired in the
/// event loop). A NATted peer keeps one monitor per active relay circuit,
/// feeds it the bytes it pushes through that circuit, and acts on the returned
/// [`MigrationAction`]: **pre-establish** a fresh circuit at 85 % of the budget
/// (once), then **cut over** the in-flight inference onto it at 95 % and close
/// the old one. The ring session (KV + ring state) is logically independent of
/// the circuit, so it survives the transport swap once the stream re-opens.
#[derive(Debug, Clone)]
pub struct CircuitMonitor {
    bytes_used: u64,
    budget: u64,
    /// Latched so `PreEstablish` is emitted exactly once per circuit (the
    /// caller should reserve the fresh circuit a single time, not every chunk).
    preestablished: bool,
}

impl CircuitMonitor {
    pub fn new(budget: u64) -> Self {
        Self { bytes_used: 0, budget: budget.max(1), preestablished: false }
    }

    /// A monitor sized to the standard [`PER_CIRCUIT_BUDGET_BYTES`].
    pub fn with_default_budget() -> Self {
        Self::new(PER_CIRCUIT_BUDGET_BYTES)
    }

    /// Account `n` bytes pushed through the circuit and return the action to
    /// take. `PreEstablish` is returned at most once (latched); `Cutover` fires
    /// on every call once ≥95 % so a missed cutover retries on the next chunk.
    pub fn record_bytes(&mut self, n: u64) -> MigrationAction {
        self.bytes_used = self.bytes_used.saturating_add(n);
        match circuit_migration_action(self.bytes_used, self.budget) {
            MigrationAction::Cutover => MigrationAction::Cutover,
            MigrationAction::PreEstablish if !self.preestablished => {
                self.preestablished = true;
                MigrationAction::PreEstablish
            }
            _ => MigrationAction::Continue,
        }
    }

    pub fn bytes_used(&self) -> u64 {
        self.bytes_used
    }

    /// Reset after a successful cutover onto a fresh circuit (re-arms the
    /// pre-establish latch for the next migration).
    pub fn reset(&mut self) {
        self.bytes_used = 0;
        self.preestablished = false;
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
    fn test_circuit_monitor_preestablish_then_cutover() {
        let budget = PER_CIRCUIT_BUDGET_BYTES;
        let mut m = CircuitMonitor::new(budget);
        // Push to ~50% — keep going.
        assert_eq!(m.record_bytes(budget / 2), MigrationAction::Continue);
        // Cross 85% — pre-establish ONCE.
        assert_eq!(m.record_bytes(budget * 36 / 100), MigrationAction::PreEstablish);
        // Still in the 85–95% band — latched, so Continue (don't re-reserve).
        assert_eq!(m.record_bytes(1), MigrationAction::Continue);
        // Cross 95% — cut over (fires every call while ≥95% so a missed cutover retries).
        assert_eq!(m.record_bytes(budget * 10 / 100), MigrationAction::Cutover);
        assert_eq!(m.record_bytes(0), MigrationAction::Cutover);
        assert!(m.bytes_used() >= budget * 95 / 100);
    }

    #[test]
    fn test_circuit_monitor_reset_rearms() {
        let budget = PER_CIRCUIT_BUDGET_BYTES;
        let mut m = CircuitMonitor::new(budget);
        assert_eq!(m.record_bytes(budget * 90 / 100), MigrationAction::PreEstablish);
        // After cutover onto a fresh circuit, reset re-arms the latch + counter.
        m.reset();
        assert_eq!(m.bytes_used(), 0);
        assert_eq!(m.record_bytes(budget / 2), MigrationAction::Continue);
        assert_eq!(m.record_bytes(budget * 40 / 100), MigrationAction::PreEstablish);
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

    // ── WS-F F-6 leech-lockout END-TO-END through the libp2p RateLimiter ──────
    // The unit tests above cover the LeechTable state machine in isolation.
    // These drive the *production integration point* — `LeechRateLimiter`'s impl
    // of `libp2p::relay::RateLimiter`, which the relay calls to admit/deny every
    // reservation and circuit — so the wiring the live hardware soak couldn't
    // exercise (the relay never carried 200MB cross-NAT) is validated here
    // deterministically.

    #[test]
    fn test_leech_rate_limiter_denies_capped_peer_end_to_end() {
        use libp2p::relay::RateLimiter;
        // The bootstrap relay shares ONE LeechTable across two LeechRateLimiters
        // (reservation_rate_limiters + circuit_src_rate_limiters), so a peer whose
        // circuit blew the byte budget is denied BOTH new reservations AND new
        // circuits for the lockout window. Mirror that exact two-limiter wiring.
        let table = Arc::new(Mutex::new(LeechTable::new()));
        let mut reservation_limiter = LeechRateLimiter::new(table.clone());
        let mut circuit_limiter = LeechRateLimiter::new(table.clone());

        let leech = PeerId::random();
        let innocent = PeerId::random();
        let addr = Multiaddr::empty();
        let now = web_time::Instant::now(); // `try_next` ignores it (uses wall clock)

        // 1. Fresh peers are admitted by both limiters (fail-open default).
        assert!(reservation_limiter.try_next(leech, &addr, now));
        assert!(circuit_limiter.try_next(leech, &addr, now));

        // 2. The leech's circuit caps out — the bootstrap CircuitClosed handler
        //    records it against the shared table (jitter 0.0 → min 15-min lockout).
        let until = table
            .lock()
            .unwrap()
            .record_cap_out(leech, unix_secs_now(), 0.0);
        assert!(until > unix_secs_now(), "lockout extends into the future");

        // 3. The capped peer is now DENIED on BOTH limiters.
        assert!(!reservation_limiter.try_next(leech, &addr, now), "denied new reservation");
        assert!(!circuit_limiter.try_next(leech, &addr, now), "denied new circuit");

        // 4. An innocent peer is unaffected — only the abuser is locked out.
        assert!(reservation_limiter.try_next(innocent, &addr, now));
        assert!(circuit_limiter.try_next(innocent, &addr, now));
    }

    #[test]
    fn test_leech_lockout_admits_again_after_expiry() {
        // `try_next` reads wall-clock `unix_secs_now()`, so expiry can't be
        // fast-forwarded through it; assert the `is_locked` predicate it
        // delegates to flips back to admit once the lockout window elapses.
        let mut t = LeechTable::new();
        let leech = PeerId::random();
        let until = t.record_cap_out(leech, 1_000, 0.0);
        assert!(t.is_locked(&leech, 1_000), "locked at cap-out");
        assert!(t.is_locked(&leech, until - 1), "still locked just before expiry");
        assert!(!t.is_locked(&leech, until + 1), "admitted once lockout expires");
    }

    #[test]
    fn test_cap_out_predicate_matches_only_byte_cap() {
        // The bootstrap handler records a lockout ONLY for the byte-budget
        // breach — benign closes (EOF, duration, disconnect) must NOT penalize,
        // else normal long sessions would get peers locked out.
        let byte_cap = format!("io error: {MAX_CIRCUIT_BYTES_ERROR}");
        assert!(byte_cap.contains(MAX_CIRCUIT_BYTES_ERROR), "byte-cap close penalizes");
        for benign in [
            "connection reset by peer",
            "Reservation(Io(Kind(UnexpectedEof)))",
            "reservation expired",
            "max circuit duration exceeded",
        ] {
            assert!(
                !benign.contains(MAX_CIRCUIT_BYTES_ERROR),
                "benign close `{benign}` must not trigger lockout",
            );
        }
    }

    // ── D-S5: only inject circuit addresses through a trusted relay ────────

    #[test]
    fn ds5_hardcoded_bootstrap_relay_pids_parse() {
        // The BOOTSTRAP_RELAYS list must yield one relay peer id per entry.
        assert_eq!(bootstrap_relay_peer_ids().len(), BOOTSTRAP_RELAYS.len());
    }

    #[test]
    fn ds5_accepts_circuit_through_a_trusted_relay_for_matching_target() {
        let relay = PeerId::random();
        let target = PeerId::random();
        let trusted: std::collections::HashSet<PeerId> = [relay].into_iter().collect();
        let good = format!("/ip4/45.79.190.172/tcp/4001/p2p/{relay}/p2p-circuit/p2p/{target}");
        assert!(safe_injectable_circuit_addr(&good, &target, &trusted).is_some());
        // Reservation-form (no explicit target) through a trusted relay is fine too.
        let reservation = format!("/ip4/45.79.190.172/tcp/4001/p2p/{relay}/p2p-circuit");
        assert!(safe_injectable_circuit_addr(&reservation, &target, &trusted).is_some());
    }

    #[test]
    fn ds5_accepts_runtime_configured_relay_not_in_hardcoded_ip_list() {
        // Regression test for the netcup gap: a relay we use at runtime (in the
        // trusted set via --bootstrap) is accepted even though its IP is nowhere
        // in BOOTSTRAP_RELAYS — the whole point of keying on peer id.
        let netcup = PeerId::random();
        let target = PeerId::random();
        let trusted: std::collections::HashSet<PeerId> = [netcup].into_iter().collect();
        let addr = format!("/ip4/85.209.48.209/tcp/4001/p2p/{netcup}/p2p-circuit/p2p/{target}");
        assert!(safe_injectable_circuit_addr(&addr, &target, &trusted).is_some());
    }

    #[test]
    fn ds5_rejects_circuit_through_untrusted_relay() {
        // The reflection vector: a circuit whose relay hop is a relay we don't use.
        let unknown_relay = PeerId::random();
        let target = PeerId::random();
        let trusted: std::collections::HashSet<PeerId> = [PeerId::random()].into_iter().collect();
        let addr =
            format!("/ip4/45.79.190.172/tcp/4001/p2p/{unknown_relay}/p2p-circuit/p2p/{target}");
        assert!(safe_injectable_circuit_addr(&addr, &target, &trusted).is_none());
    }

    #[test]
    fn ds5_rejects_circuit_naming_a_different_target() {
        let relay = PeerId::random();
        let target = PeerId::random();
        let other = PeerId::random();
        let trusted: std::collections::HashSet<PeerId> = [relay].into_iter().collect();
        let addr = format!("/ip4/45.79.190.172/tcp/4001/p2p/{relay}/p2p-circuit/p2p/{target}");
        // A record for `other` must not seed an address that dials `target`.
        assert!(safe_injectable_circuit_addr(&addr, &other, &trusted).is_none());
    }

    #[test]
    fn ds5_rejects_direct_address() {
        // A signed record advertising a direct (non-circuit) victim address is
        // dropped — this is exactly the DHT-amplification vector D-S5 closes.
        let relay = PeerId::random();
        let target = PeerId::random();
        let trusted: std::collections::HashSet<PeerId> = [relay].into_iter().collect();
        let direct = format!("/ip4/9.9.9.9/udp/443/quic-v1/p2p/{target}");
        assert!(safe_injectable_circuit_addr(&direct, &target, &trusted).is_none());
        assert!(safe_injectable_circuit_addr("not-a-multiaddr", &target, &trusted).is_none());
    }
}
