//! mDNS LAN peer discovery helpers.
//!
//! mDNS is configured inside the swarm behaviour (see behaviour.rs).
//! This module provides helpers for filtering and processing mDNS events.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use libp2p::swarm::ConnectionId;
use libp2p::{Multiaddr, PeerId};

/// A peer discovered via mDNS on the local network.
#[derive(Debug, Clone)]
pub struct LanPeer {
    pub peer_id: PeerId,
    pub addrs: Vec<Multiaddr>,
}

/// Filter mDNS-discovered addresses to keep only LAN-routable ones.
///
/// mDNS can return loopback (127.x), link-local (169.254.x), and actual
/// LAN addresses. We keep only private-range addresses that are useful
/// for direct gRPC connections.
pub fn filter_lan_addrs(addrs: &[Multiaddr]) -> Vec<Multiaddr> {
    addrs
        .iter()
        .filter(|addr| {
            let s = addr.to_string();
            // Keep private ranges, exclude loopback and link-local.
            !s.starts_with("/ip4/127.") && !s.starts_with("/ip4/169.254.")
        })
        .cloned()
        .collect()
}

/// Dials to peers mDNS found, at most this many at once.
pub const MAX_LAN_DIALS_IN_FLIGHT: usize = 32;
/// Connections we made because of mDNS (being dialled or established), at most this many at once:
/// a host on the LAN announcing many made-up peers can't use up the node's connection limit.
pub const MAX_LAN_CONNECTIONS: usize = 64;
/// Of those, at most this many to one IP (one machine can't take every slot with made-up peers).
pub const MAX_LAN_DIALS_PER_IP: usize = 4;
/// A LAN connection that closes sooner than this after it was established counts as a failed
/// dial (backed off): otherwise one that keeps closing early (the remote's limits, a duplicate
/// connection closed on purpose) would be redialled at every heal tick.
pub const LAN_CONN_MIN_LIFE: Duration = Duration::from_secs(30);
/// Dial rounds driven by mDNS packets, at most one this often (the heal tick drives them too).
pub const LAN_DIAL_ROUND_EVERY: Duration = Duration::from_secs(1);
/// Waits before dialling a LAN peer again after a failed dial: 30 s, then 2, 5 and 10 minutes.
pub const LAN_DIAL_BACKOFF: [Duration; 4] =
    [Duration::from_secs(30), Duration::from_secs(120), Duration::from_secs(300), Duration::from_secs(600)];
/// How long a LAN peer that turned out not to be an OpenHydra node isn't dialled again.
pub const FOREIGN_PEER_HOLD: Duration = Duration::from_secs(3600);
/// Peers remembered for back-off or as foreign, at most (past it the table starts over).
const LAN_PEERS_TRACKED: usize = 4096;

/// The bookkeeping for dialling peers that mDNS discovers (review 2026-10-29: discoveries were only
/// logged, so two nodes on a LAN with no bootstrap never connected). It decides which discovered
/// peers to dial, bounds the dials, backs off after failures, and remembers LAN peers that aren't
/// OpenHydra nodes (mDNS is libp2p's shared service, so IPFS and other libp2p nodes answer too).
/// The event loop does the dialling; this holds no swarm, so it's tested on its own.
#[derive(Debug, Default)]
pub struct LanDialer {
    /// The LAN addresses mDNS currently lists for each peer (mDNS reports a (peer, address) only
    /// when it's new, so a peer skipped or backed off is retried from here: [`Self::due`]).
    known: HashMap<PeerId, Vec<Multiaddr>>,
    /// Connections we dialled because of mDNS (in flight or established), by connection id: the
    /// peer, the IP dialled, and when it was established.
    dialled: HashMap<ConnectionId, (PeerId, Option<std::net::IpAddr>, Option<Instant>)>,
    /// How many of `dialled` each peer has (so a lookup is O(1)).
    dialled_peers: HashMap<PeerId, usize>,
    /// Of `dialled`, the ones still being dialled, with the IP dialled.
    in_flight: HashMap<ConnectionId, Option<std::net::IpAddr>>,
    /// When the last mDNS-driven dial round ran ([`LAN_DIAL_ROUND_EVERY`]).
    last_round: Option<Instant>,
    /// Failed dials per peer, and when it may be dialled again.
    backoff: HashMap<PeerId, (usize, Instant)>,
    /// LAN peers that aren't OpenHydra nodes, and until when they're left alone.
    foreign: HashMap<PeerId, Instant>,
}

impl LanDialer {
    pub fn new() -> Self {
        Self::default()
    }

    /// mDNS lists `addr` for `peer`.
    pub fn discovered(&mut self, peer: PeerId, addr: Multiaddr) {
        if !self.known.contains_key(&peer) && self.known.len() >= LAN_PEERS_TRACKED {
            return;
        }
        let addrs = self.known.entry(peer).or_default();
        if !addrs.contains(&addr) && addrs.len() < 16 {
            addrs.push(addr);
        }
    }

    /// mDNS no longer lists `addr` for `peer`.
    pub fn expired(&mut self, peer: &PeerId, addr: &Multiaddr) {
        if let Some(addrs) = self.known.get_mut(peer) {
            addrs.retain(|a| a != addr);
            if addrs.is_empty() {
                self.known.remove(peer);
            }
        }
    }

    /// The LAN peers to dial now, with their addresses. `has_lan_connection` says whether a peer
    /// already has a connection to us over the LAN (nothing to gain then).
    pub fn due(&self, has_lan_connection: impl Fn(&PeerId) -> bool, now: Instant) -> Vec<(PeerId, Vec<Multiaddr>)> {
        let mut out = Vec::new();
        let mut ips: HashMap<std::net::IpAddr, usize> = HashMap::new();
        for (_, ip, _) in self.dialled.values() {
            if let Some(ip) = ip {
                *ips.entry(*ip).or_default() += 1;
            }
        }
        let mut slots = MAX_LAN_DIALS_IN_FLIGHT
            .saturating_sub(self.in_flight.len())
            .min(MAX_LAN_CONNECTIONS.saturating_sub(self.dialled.len()));
        for (peer, addrs) in &self.known {
            if slots == 0 {
                break;
            }
            if !self.should_dial(peer, has_lan_connection(peer), now) {
                continue;
            }
            let ip = first_ip(addrs);
            if let Some(ip) = ip {
                let n = ips.entry(ip).or_default();
                if *n >= MAX_LAN_DIALS_PER_IP {
                    continue;
                }
                *n += 1;
            }
            out.push((*peer, addrs.clone()));
            slots -= 1;
        }
        out
    }

    /// Whether to dial `peer` now. `has_lan_connection`: it already has a connection to us over the
    /// LAN, so there's nothing to gain.
    pub fn should_dial(&self, peer: &PeerId, has_lan_connection: bool, now: Instant) -> bool {
        !has_lan_connection
            && self.in_flight.len() < MAX_LAN_DIALS_IN_FLIGHT
            && self.dialled.len() < MAX_LAN_CONNECTIONS
            && !self.dialled_peers.contains_key(peer)
            && self.foreign.get(peer).map_or(true, |until| now >= *until)
            && self.backoff.get(peer).map_or(true, |(_, until)| now >= *until)
    }

    /// A dial to `peer` at `addrs` was started as connection `id`.
    pub fn dialled(&mut self, id: ConnectionId, peer: PeerId, addrs: &[Multiaddr]) {
        let ip = first_ip(addrs);
        if self.dialled.insert(id, (peer, ip, None)).is_none() {
            *self.dialled_peers.entry(peer).or_default() += 1;
        }
        self.in_flight.insert(id, ip);
    }

    /// Whether an mDNS-driven dial round may run at `now` (and if so, starts the wait): at most one
    /// per [`LAN_DIAL_ROUND_EVERY`], however many mDNS packets arrive.
    pub fn round_due(&mut self, now: Instant) -> bool {
        if self.last_round.is_some_and(|t| now.duration_since(t) < LAN_DIAL_ROUND_EVERY) {
            return false;
        }
        self.last_round = Some(now);
        true
    }

    /// Stop tracking connection `id`; the peer it was to, if it was ours.
    fn forget(&mut self, id: &ConnectionId) -> Option<(PeerId, Option<Instant>)> {
        self.in_flight.remove(id);
        let (peer, _, since) = self.dialled.remove(id)?;
        if let Some(n) = self.dialled_peers.get_mut(&peer) {
            *n -= 1;
            if *n == 0 {
                self.dialled_peers.remove(&peer);
            }
        }
        Some((peer, since))
    }

    fn back_off(&mut self, peer: PeerId, now: Instant) {
        self.bound();
        let n = self.backoff.get(&peer).map_or(0, |(n, _)| *n);
        let wait = LAN_DIAL_BACKOFF[n.min(LAN_DIAL_BACKOFF.len() - 1)];
        self.backoff.insert(peer, (n + 1, now + wait));
    }

    /// Whether connection `id` is one we dialled because of mDNS.
    pub fn is_lan_dial(&self, id: &ConnectionId) -> bool {
        self.dialled.contains_key(id)
    }

    /// Connection `id` was established. Its peer's back-off isn't reset yet: that waits for the
    /// peer to turn out to be one of ours ([`Self::accepted`]) and the connection to last.
    pub fn established(&mut self, id: &ConnectionId, now: Instant) {
        if self.in_flight.remove(id).is_some() {
            if let Some((_, _, since)) = self.dialled.get_mut(id) {
                *since = Some(now);
            }
        }
    }

    /// Identify on our LAN dial `id` showed an OpenHydra node of our network: its back-off is reset.
    pub fn accepted(&mut self, id: &ConnectionId) {
        if let Some((peer, _, _)) = self.dialled.get(id) {
            self.backoff.remove(peer);
        }
    }

    /// The dial `id` failed: back its peer off.
    pub fn failed(&mut self, id: &ConnectionId, now: Instant) {
        if let Some((peer, _)) = self.forget(id) {
            self.back_off(peer, now);
        }
    }

    /// Connection `id` closed. One of ours that lasted less than [`LAN_CONN_MIN_LIFE`] counts as a
    /// failed dial.
    pub fn closed(&mut self, id: &ConnectionId, now: Instant) {
        if let Some((peer, since)) = self.forget(id) {
            if since.map_or(true, |t| now.duration_since(t) < LAN_CONN_MIN_LIFE) {
                self.back_off(peer, now);
            }
        }
    }

    /// The peer behind our LAN dial `id` doesn't speak OpenHydra's protocol, or is on another
    /// OpenHydra network (a private deployment and the public one on the same LAN): leave it alone
    /// for [`FOREIGN_PEER_HOLD`]. Returns whether `id` was one of our LAN dials (only those are dropped:
    /// an inbound connection or one made another way, a private deployment's bootstrap say, is
    /// never touched here).
    pub fn foreign(&mut self, id: &ConnectionId, now: Instant) -> bool {
        match self.dialled.get(id) {
            Some((peer, _, _)) => {
                let peer = *peer;
                self.bound();
                self.foreign.insert(peer, now + FOREIGN_PEER_HOLD);
                true
            }
            None => false,
        }
    }

    fn bound(&mut self) {
        if self.known.len() >= LAN_PEERS_TRACKED {
            self.known.clear();
        }
        if self.backoff.len() >= LAN_PEERS_TRACKED {
            self.backoff.clear();
        }
        if self.foreign.len() >= LAN_PEERS_TRACKED {
            self.foreign.clear();
        }
    }
}

/// The IP of the first IP address in `addrs`.
fn first_ip(addrs: &[Multiaddr]) -> Option<std::net::IpAddr> {
    use libp2p::multiaddr::Protocol;
    addrs.iter().find_map(|a| match a.iter().next() {
        Some(Protocol::Ip4(ip)) => Some(ip.into()),
        Some(Protocol::Ip6(ip)) => Some(ip.into()),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lan(n: u8) -> Vec<Multiaddr> {
        vec![format!("/ip4/192.168.0.{n}/tcp/4001").parse().unwrap()]
    }

    #[test]
    fn a_skipped_or_backed_off_peer_is_retried_from_what_mdns_lists() {
        let mut d = LanDialer::new();
        let p = PeerId::random();
        let t0 = Instant::now();
        d.discovered(p, lan(5)[0].clone());
        assert_eq!(d.due(|_| false, t0).len(), 1);
        d.dialled(id(1), p, &lan(5));
        d.failed(&id(1), t0);
        assert!(d.due(|_| false, t0).is_empty(), "backed off");
        assert_eq!(d.due(|_| false, t0 + LAN_DIAL_BACKOFF[0]).len(), 1, "due again, with no new mDNS event");
        d.expired(&p, &lan(5)[0]);
        assert!(d.due(|_| false, t0 + LAN_DIAL_BACKOFF[0]).is_empty(), "gone from mDNS: not dialled");
    }

    #[test]
    fn one_ip_cant_take_every_dial_slot() {
        let mut d = LanDialer::new();
        let now = Instant::now();
        // An attacker announcing many made-up peers at its one address...
        for _ in 0..50 {
            d.discovered(PeerId::random(), lan(66)[0].clone());
        }
        // ...and one real peer elsewhere.
        let real = PeerId::random();
        d.discovered(real, lan(7)[0].clone());
        let due = d.due(|_| false, now);
        assert_eq!(due.iter().filter(|(_, a)| a == &lan(66)).count(), MAX_LAN_DIALS_PER_IP);
        assert!(due.iter().any(|(p, _)| *p == real), "the real peer gets a slot");
    }

    fn id(n: usize) -> ConnectionId {
        ConnectionId::new_unchecked(n)
    }

    #[test]
    fn a_discovered_peer_is_dialled_once_and_not_while_a_dial_is_open() {
        let mut d = LanDialer::new();
        let p = PeerId::random();
        let now = Instant::now();
        assert!(d.should_dial(&p, false, now));
        assert!(!d.should_dial(&p, true, now), "already on the LAN");
        d.dialled(id(1), p, &[]);
        assert!(!d.should_dial(&p, false, now), "a dial is open");
        d.established(&id(1), now);
        d.accepted(&id(1));
        assert!(!d.should_dial(&p, false, now), "connected through our dial");
        let later = now + LAN_CONN_MIN_LIFE;
        d.closed(&id(1), later);
        assert!(d.should_dial(&p, false, later), "a connection that lasted: dialled again when it's gone");
    }

    #[test]
    fn failures_back_off_30s_2m_5m_10m_and_a_success_resets_it() {
        let mut d = LanDialer::new();
        let p = PeerId::random();
        let t0 = Instant::now();
        let mut t = t0;
        for (n, wait) in LAN_DIAL_BACKOFF.iter().chain(std::iter::once(&LAN_DIAL_BACKOFF[3])).enumerate() {
            d.dialled(id(n), p, &[]);
            d.failed(&id(n), t);
            assert!(!d.should_dial(&p, false, t + *wait - Duration::from_secs(1)));
            assert!(d.should_dial(&p, false, t + *wait));
            t += *wait;
        }
        d.dialled(id(99), p, &[]);
        d.established(&id(99), t);
        d.accepted(&id(99));
        let t = t + LAN_CONN_MIN_LIFE;
        d.closed(&id(99), t);
        assert!(d.should_dial(&p, false, t), "back-off reset after a success");
    }

    #[test]
    fn a_connection_that_closes_early_counts_as_a_failure() {
        let mut d = LanDialer::new();
        let p = PeerId::random();
        let t0 = Instant::now();
        // Established, even accepted, then closed at once (a duplicate closed on purpose, say).
        d.dialled(id(1), p, &[]);
        d.established(&id(1), t0);
        d.accepted(&id(1));
        d.closed(&id(1), t0 + Duration::from_secs(1));
        assert!(!d.should_dial(&p, false, t0 + Duration::from_secs(6)), "not again at the next heal tick");
        assert!(d.should_dial(&p, false, t0 + Duration::from_secs(1) + LAN_DIAL_BACKOFF[0]));
        // Each early close lengthens the wait; establishing alone doesn't reset it.
        let t1 = t0 + Duration::from_secs(1) + LAN_DIAL_BACKOFF[0];
        d.dialled(id(2), p, &[]);
        d.established(&id(2), t1);
        d.closed(&id(2), t1 + Duration::from_secs(2));
        assert!(!d.should_dial(&p, false, t1 + Duration::from_secs(2) + LAN_DIAL_BACKOFF[0]));
        assert!(d.should_dial(&p, false, t1 + Duration::from_secs(2) + LAN_DIAL_BACKOFF[1]));
    }

    #[test]
    fn established_lan_connections_count_toward_the_caps() {
        let mut d = LanDialer::new();
        let now = Instant::now();
        // One host holding established connections for 4 made-up peers can't get a 5th.
        for n in 0..MAX_LAN_DIALS_PER_IP {
            d.dialled(id(n), PeerId::random(), &lan(66));
            d.established(&id(n), now);
        }
        d.discovered(PeerId::random(), lan(66)[0].clone());
        assert!(d.due(|_| false, now).is_empty(), "the per-IP cap counts established connections");
        // And established connections across many hosts stop at the total cap.
        let mut d = LanDialer::new();
        for n in 0..MAX_LAN_CONNECTIONS {
            d.dialled(id(n), PeerId::random(), &lan((n % 200) as u8 + 1));
            d.established(&id(n), now);
        }
        assert!(!d.should_dial(&PeerId::random(), false, now));
        d.discovered(PeerId::random(), lan(250)[0].clone());
        assert!(d.due(|_| false, now).is_empty(), "the total cap counts established connections");
    }

    #[test]
    fn mdns_packets_drive_at_most_one_dial_round_a_second() {
        let mut d = LanDialer::new();
        let t0 = Instant::now();
        assert!(d.round_due(t0));
        assert!(!d.round_due(t0 + Duration::from_millis(500)));
        assert!(d.round_due(t0 + LAN_DIAL_ROUND_EVERY));
    }

    #[test]
    fn dials_in_flight_are_capped() {
        let mut d = LanDialer::new();
        let now = Instant::now();
        for n in 0..MAX_LAN_DIALS_IN_FLIGHT {
            d.dialled(id(n), PeerId::random(), &[]);
        }
        assert!(!d.should_dial(&PeerId::random(), false, now));
        d.established(&id(0), now);
        assert!(d.should_dial(&PeerId::random(), false, now), "established frees a dial slot (not a connection slot)");
    }

    #[test]
    fn only_our_lan_dials_are_dropped_as_foreign_and_they_are_held_for_an_hour() {
        let mut d = LanDialer::new();
        let now = Instant::now();
        assert!(!d.foreign(&id(7), now), "not a LAN dial of ours: never dropped");
        let p = PeerId::random();
        d.dialled(id(1), p, &[]);
        d.established(&id(1), now);
        assert!(d.foreign(&id(1), now));
        d.closed(&id(1), now);
        assert!(!d.should_dial(&p, false, now + FOREIGN_PEER_HOLD - Duration::from_secs(1)));
        assert!(d.should_dial(&p, false, now + FOREIGN_PEER_HOLD));
    }

    #[test]
    fn test_filter_lan_addrs() {
        let addrs: Vec<Multiaddr> = vec![
            "/ip4/127.0.0.1/tcp/4001".parse().unwrap(),
            "/ip4/192.168.1.10/tcp/4001".parse().unwrap(),
            "/ip4/169.254.0.1/tcp/4001".parse().unwrap(),
            "/ip4/10.0.0.5/tcp/4001".parse().unwrap(),
        ];
        let filtered = filter_lan_addrs(&addrs);
        assert_eq!(filtered.len(), 2);
        assert!(filtered[0].to_string().contains("192.168.1.10"));
        assert!(filtered[1].to_string().contains("10.0.0.5"));
    }
}
