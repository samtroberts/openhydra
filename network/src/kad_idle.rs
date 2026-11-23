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
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use libp2p::PeerId;

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
            // Still full of peers used lately (a busy gateway): forget the least recent rather
            // than not recording this one, which would leave its connections closable.
            while at.len() >= MAX_REMEMBERED {
                let Some(oldest) = at.iter().min_by_key(|(_, t)| **t).map(|(p, _)| *p) else { break };
                at.remove(&oldest);
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
