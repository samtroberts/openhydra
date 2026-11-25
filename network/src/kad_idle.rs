// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Closing the connections only the DHT used (21 Nov 2026; un-parks #16, narrowed).
//!
//! Every Kademlia contact left a connection open, and gossip's long-lived streams kept every
//! connection alive, so a public node's connections grew without end: in a 1,500-node simulation,
//! 378 per node after a minute and 866 after five, near the 1,024 inbound limit, with the
//! handshakes and upkeep driving the hosts' load up and DHT announcements into timeouts.
//!
//! The node that dialled a connection for Kademlia closes it once it is idle, under a narrow rule
//! ([`should_close`]): nothing else of ours uses the peer, and the peer runs a version that knows
//! such a close isn't a loss ([`AGENT_TOKEN`]). The closer never treats its own close as losing the
//! peer, and the other side doesn't either for a peer it hasn't used ([`AppPeers::wanted`]).
//! The six review findings that parked #16 are answered in `docs/IDLE_CLOSE_PLAN_2026-11-21.md`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use libp2p::{Multiaddr, PeerId};

/// How long a Kademlia connection must have gone without Kademlia traffic (and have been open)
/// before it is closed.
pub const IDLE: Duration = Duration::from_secs(60);
/// How long after a peer's last real use (a request, a reply, a push stream) its connections are
/// kept, idle or not.
pub const APP_KEEP: Duration = Duration::from_secs(10 * 60);
/// How often the event loop looks for connections to close.
pub const SWEEP_EVERY: Duration = Duration::from_secs(15);
/// Added to identify's agent version by a node that closes idle Kademlia connections and doesn't
/// take such a close as losing the peer. Only peers that send it are closed on: an older node
/// (v0.3.24) evicts a peer from its routing table and drops its records when the last connection
/// closes, so closing on it would make it forget live providers.
pub const AGENT_TOKEN: &str = "kad-idle=1";
/// Peers remembered as used at most; past it, those not used within [`APP_KEEP`] are forgotten,
/// then the least recently used.
const MAX_REMEMBERED: usize = 16384;

/// Whether a peer's identify agent version says it knows idle Kademlia closes ([`AGENT_TOKEN`]).
pub fn peer_supports(agent_version: &str) -> bool {
    agent_version.split_whitespace().any(|w| w == AGENT_TOKEN)
}

/// The QUIC application error code an idle close goes out with (25 Nov 2026). Any other close
/// (code 0: a quit, a close for another reason) isn't one.
pub const IDLE_CLOSE_CODE: u32 = 0x4f48_1d1e;
/// Added to identify's agent version, after [`AGENT_TOKEN`], by a node that closes an idle QUIC
/// connection with [`IDLE_CLOSE_CODE`]. From such a peer only that code is an idle close over QUIC,
/// so a quit or a crash counts as losing it; from a peer with [`AGENT_TOKEN`] alone (the bootstraps
/// deployed 24 Nov), any close it meant still counts as idle. TCP has no close code (yamux), so
/// there a close the peer meant counts as idle either way.
pub const CODE_TOKEN: &str = "kad-idle-code=1";

/// Whether a peer's identify agent version says it closes idle QUIC connections with
/// [`IDLE_CLOSE_CODE`] ([`CODE_TOKEN`]).
pub fn peer_sends_idle_code(agent_version: &str) -> bool {
    agent_version.split_whitespace().any(|w| w == CODE_TOKEN)
}

/// The code each of this node's QUIC connections will be closed with, by peer, so the idle sweep
/// can close one with [`IDLE_CLOSE_CODE`]: the swarm closes a connection by its id and never shows
/// the QUIC connection behind it, so the transport registers each one here as it comes up
/// (`transport::build_quic_transport`) and the sweep finds it by the peer and the address.
#[derive(Clone, Default)]
pub struct QuicCloseCodes(Arc<Mutex<HashMap<PeerId, Vec<(SocketAddr, libp2p::quic::CloseCode)>>>>);

impl QuicCloseCodes {
    /// A QUIC connection with `peer` came up.
    pub fn register(&self, peer: PeerId, connection: &libp2p::quic::Connection) {
        let mut by_peer = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let codes = by_peer.entry(peer).or_default();
        codes.retain(|(_, c)| c.in_use());
        codes.push((connection.remote_address(), connection.close_code()));
    }

    /// The connection with `peer` at `remote` is about to be closed as idle: it will close with
    /// [`IDLE_CLOSE_CODE`]. False when `remote` isn't a direct QUIC address or no live connection
    /// matches it (the close then goes out with code 0, which the peer takes as a loss).
    pub fn close_as_idle(&self, peer: &PeerId, remote: &Multiaddr) -> bool {
        let Some(at) = quic_socket_addr(remote) else { return false };
        let mut by_peer = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let Some(codes) = by_peer.get_mut(peer) else { return false };
        codes.retain(|(_, c)| c.in_use());
        let mut set = false;
        for (addr, code) in codes.iter() {
            if canonical(*addr) == canonical(at) {
                code.set(IDLE_CLOSE_CODE);
                set = true;
            }
        }
        // The peer moved (QUIC follows a NAT rebinding, the swarm's address doesn't): its only
        // connection is the one.
        if !set && codes.len() == 1 {
            codes[0].1.set(IDLE_CLOSE_CODE);
            set = true;
        }
        set
    }

    /// Forget closed connections, and peers with none left.
    pub fn prune(&self) {
        let mut by_peer = self.0.lock().unwrap_or_else(|e| e.into_inner());
        by_peer.retain(|_, codes| {
            codes.retain(|(_, c)| c.in_use());
            !codes.is_empty()
        });
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.0.lock().unwrap().values().map(Vec::len).sum()
    }
}

/// The UDP socket address of a direct QUIC multiaddr (`/ip4|ip6/…/udp/…/quic-v1[/p2p/…]`); `None`
/// for anything else, a relay circuit among them.
fn quic_socket_addr(addr: &Multiaddr) -> Option<SocketAddr> {
    use libp2p::multiaddr::Protocol;
    let mut iter = addr.iter();
    let ip = match iter.next()? {
        Protocol::Ip4(ip) => std::net::IpAddr::V4(ip),
        Protocol::Ip6(ip) => std::net::IpAddr::V6(ip),
        _ => return None,
    };
    let Protocol::Udp(port) = iter.next()? else { return None };
    let Protocol::QuicV1 = iter.next()? else { return None };
    match iter.next() {
        None | Some(Protocol::P2p(_)) if iter.next().is_none() => Some(SocketAddr::new(ip, port)),
        _ => None,
    }
}

/// An IPv4-mapped IPv6 address as IPv4, so a dual-stack socket's view matches the multiaddr's.
fn canonical(addr: SocketAddr) -> SocketAddr {
    SocketAddr::new(addr.ip().to_canonical(), addr.port())
}

/// The peers this node has really used lately (sent or served a request, exchanged a push stream),
/// and those with a push stream open now, shared between the event loop and the push streams.
#[derive(Clone, Default)]
pub struct AppPeers(Arc<Mutex<Used>>);

#[derive(Default)]
struct Used {
    /// Each peer's last use.
    at: HashMap<PeerId, Instant>,
    /// Push streams open per peer (review 22 Nov: a stream longer than [`APP_KEEP`] was not
    /// counted as use after its open, so its connection could be closed under it).
    open: HashMap<PeerId, usize>,
}

/// A push stream open with a peer: the peer counts as used until it is dropped, and from then for
/// [`APP_KEEP`].
pub struct Held {
    peers: AppPeers,
    peer: PeerId,
}

impl Drop for Held {
    fn drop(&mut self) {
        let mut used = self.peers.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = used.open.get_mut(&self.peer) {
            *n -= 1;
            if *n == 0 {
                used.open.remove(&self.peer);
            }
        }
        drop(used);
        self.peers.note(self.peer);
    }
}

impl AppPeers {
    /// `peer` was really used just now.
    pub fn note(&self, peer: PeerId) {
        self.note_at(peer, Instant::now());
    }

    /// A push stream with `peer` is open until the returned guard is dropped.
    pub fn hold(&self, peer: PeerId) -> Held {
        self.note(peer);
        *self.0.lock().unwrap_or_else(|e| e.into_inner()).open.entry(peer).or_default() += 1;
        Held { peers: self.clone(), peer }
    }

    fn note_at(&self, peer: PeerId, now: Instant) {
        let mut used = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let at = &mut used.at;
        if at.len() >= MAX_REMEMBERED && !at.contains_key(&peer) {
            at.retain(|_, t| now.saturating_duration_since(*t) < APP_KEEP);
            // Still full of peers used lately (a busy gateway): forget the least recent tenth in one
            // pass (review 22 Nov: one full scan per new peer) rather than not recording this one,
            // which would leave its connections closable.
            if at.len() >= MAX_REMEMBERED {
                forget_oldest_tenth(at);
            }
        }
        at.insert(peer, now);
    }

    /// Whether `peer` has a push stream open or was used within [`APP_KEEP`].
    pub fn wanted(&self, peer: &PeerId) -> bool {
        self.wanted_at(peer, Instant::now())
    }

    fn wanted_at(&self, peer: &PeerId, now: Instant) -> bool {
        let used = self.0.lock().unwrap_or_else(|e| e.into_inner());
        used.open.contains_key(peer) || used.at.get(peer).is_some_and(|t| now.saturating_duration_since(*t) < APP_KEEP)
    }
}

/// Remove the least recently used tenth of `at` (at least one entry), in one pass.
fn forget_oldest_tenth(at: &mut HashMap<PeerId, Instant>) {
    let n = (at.len() / 10).max(1);
    let mut times: Vec<Instant> = at.values().copied().collect();
    let (_, cutoff, _) = times.select_nth_unstable(n - 1);
    let cutoff = *cutoff;
    let mut removed = 0;
    at.retain(|_, t| {
        if removed < n && *t <= cutoff {
            removed += 1;
            false
        } else {
            true
        }
    });
}

/// What the event loop knows about one direct connection when it sweeps.
#[derive(Debug, Clone, Copy)]
pub struct ConnFacts {
    /// Kademlia dialled it (not our own code, not another behaviour).
    pub kad_dialled: bool,
    /// How long it has been open.
    pub age: Duration,
    /// How long since Kademlia last exchanged anything with the peer (its age if never).
    pub since_kad_use: Duration,
    /// The peer's identify carries [`AGENT_TOKEN`].
    pub peer_supports: bool,
    /// The peer is one this node always keeps: a bootstrap or relay, pinned, on the LAN (mDNS), or
    /// a gossip mesh peer.
    pub kept: bool,
    /// The peer was really used within [`APP_KEEP`].
    pub app_used: bool,
    /// Something is under way with the peer: a request of ours or of its, a response being
    /// written, a queued forward (the event loop's `peer_busy`).
    pub in_flight: bool,
}

/// Whether to close this connection now.
pub fn should_close(f: &ConnFacts) -> bool {
    f.kad_dialled
        && f.peer_supports
        && f.age >= IDLE
        && f.since_kad_use >= IDLE
        && !f.kept
        && !f.app_used
        && !f.in_flight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_idle_code_token_is_read_as_its_own_word() {
        assert!(peer_sends_idle_code("openhydra-agent/0.3.24 net=public kad-idle=1 kad-idle-code=1"));
        assert!(peer_supports("openhydra-agent/0.3.24 net=public kad-idle=1 kad-idle-code=1"));
        // The bootstraps deployed 24 Nov: idle closes, but no code.
        assert!(!peer_sends_idle_code("openhydra-agent/0.3.24 net=public kad-idle=1"));
        assert!(!peer_sends_idle_code("openhydra-agent/0.3.24 net=public kad-idle-code=10"));
    }

    #[test]
    fn only_a_direct_quic_address_names_a_quic_connection() {
        let at = |s: &str| quic_socket_addr(&s.parse().unwrap());
        let peer = PeerId::random();
        assert_eq!(at("/ip4/192.0.2.1/udp/4001/quic-v1"), Some("192.0.2.1:4001".parse().unwrap()));
        assert_eq!(at(&format!("/ip6/2001:db8::1/udp/4001/quic-v1/p2p/{peer}")), Some("[2001:db8::1]:4001".parse().unwrap()));
        assert_eq!(at("/ip4/192.0.2.1/tcp/4001"), None);
        assert_eq!(at("/ip4/192.0.2.1/udp/4001/quic"), None);
        assert_eq!(at(&format!("/ip4/192.0.2.1/udp/4001/quic-v1/p2p/{peer}/p2p-circuit/p2p/{peer}")), None);
        assert_eq!(at("/dns4/example.com/udp/4001/quic-v1"), None);
        let mapped: SocketAddr = "[::ffff:192.0.2.1]:4001".parse().unwrap();
        assert_eq!(canonical(mapped), "192.0.2.1:4001".parse().unwrap());
        // Nothing registered: nothing to set, nothing kept.
        let codes = QuicCloseCodes::default();
        assert!(!codes.close_as_idle(&peer, &"/ip4/192.0.2.1/udp/4001/quic-v1".parse().unwrap()));
        codes.prune();
        assert_eq!(codes.len(), 0);
    }

    fn idle_kad_conn() -> ConnFacts {
        ConnFacts {
            kad_dialled: true,
            age: Duration::from_secs(120),
            since_kad_use: Duration::from_secs(90),
            peer_supports: true,
            kept: false,
            app_used: false,
            in_flight: false,
        }
    }

    #[test]
    fn an_idle_connection_kademlia_dialled_closes() {
        assert!(should_close(&idle_kad_conn()));
    }

    #[test]
    fn every_other_case_stays_open() {
        let cases: [(&str, fn(&mut ConnFacts)); 7] = [
            ("dialled by something else, or by the peer", |f| f.kad_dialled = false),
            ("an older peer that would take the close as a loss", |f| f.peer_supports = false),
            ("too young", |f| f.age = Duration::from_secs(30)),
            ("Kademlia used it lately", |f| f.since_kad_use = Duration::from_secs(10)),
            ("a peer we always keep", |f| f.kept = true),
            ("a peer we used for a request lately", |f| f.app_used = true),
            ("a request in flight", |f| f.in_flight = true),
        ];
        for (what, change) in cases {
            let mut f = idle_kad_conn();
            change(&mut f);
            assert!(!should_close(&f), "{what}");
        }
    }

    #[test]
    fn the_token_is_a_whole_word_of_the_agent_version() {
        assert!(peer_supports("openhydra-agent/0.3.3 net=public kad-idle=1"));
        assert!(!peer_supports("openhydra-agent/0.3.3 net=public"));
        assert!(!peer_supports("openhydra-agent/0.3.3 net=public kad-idle=10"));
    }

    #[test]
    fn a_peer_is_wanted_for_app_keep_after_its_last_use() {
        let peers = AppPeers::default();
        let p = PeerId::random();
        let t0 = Instant::now();
        assert!(!peers.wanted_at(&p, t0));
        peers.note_at(p, t0);
        assert!(peers.wanted_at(&p, t0 + APP_KEEP - Duration::from_secs(1)));
        assert!(!peers.wanted_at(&p, t0 + APP_KEEP));
    }

    #[test]
    fn a_peer_with_a_push_stream_open_is_wanted_however_long_it_runs() {
        let peers = AppPeers::default();
        let p = PeerId::random();
        let t0 = Instant::now();
        let held = peers.hold(p);
        let _second = peers.hold(p);
        assert!(peers.wanted_at(&p, t0 + 3 * APP_KEEP), "open");
        drop(held);
        assert!(peers.wanted_at(&p, t0 + 3 * APP_KEEP), "the other stream is still open");
        drop(_second);
        assert!(peers.wanted_at(&p, Instant::now() + APP_KEEP - Duration::from_secs(1)), "kept after the end");
        assert!(!peers.wanted_at(&p, Instant::now() + APP_KEEP + Duration::from_secs(1)));
    }

    #[test]
    fn a_full_memory_forgets_its_least_recent_tenth_at_once() {
        let peers = AppPeers::default();
        let t0 = Instant::now();
        let old: Vec<PeerId> = (0..MAX_REMEMBERED).map(|_| PeerId::random()).collect();
        for (i, p) in old.iter().enumerate() {
            peers.note_at(*p, t0 + Duration::from_millis(i as u64));
        }
        let fresh = PeerId::random();
        peers.note_at(fresh, t0 + Duration::from_secs(60));
        let used = peers.0.lock().unwrap();
        assert_eq!(used.at.len(), MAX_REMEMBERED - MAX_REMEMBERED / 10 + 1, "a tenth forgotten, the new one kept");
        assert!(!used.at.contains_key(&old[0]) && used.at.contains_key(&old[MAX_REMEMBERED - 1]) && used.at.contains_key(&fresh));
    }

    #[test]
    fn a_full_memory_of_recent_peers_still_records_a_new_one() {
        let peers = AppPeers::default();
        let now = Instant::now();
        for _ in 0..MAX_REMEMBERED {
            peers.note_at(PeerId::random(), now);
        }
        let p = PeerId::random();
        peers.note_at(p, now + Duration::from_secs(1));
        assert!(peers.wanted_at(&p, now + Duration::from_secs(2)));
    }
}
