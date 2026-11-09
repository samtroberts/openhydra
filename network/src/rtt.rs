// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The latest ping round trip to each peer, shared between the event loop (which records each ping
//! result) and the [`NetworkHandle`](crate::handle::NetworkHandle) (which a consumer asks when it
//! chooses between providers).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A round trip older than this says little about the path now, and is not reported.
const RTT_FRESH: Duration = Duration::from_secs(300);
/// At most this many peers are remembered; the stalest is dropped first.
const MAX_PEERS: usize = 8_192;

/// Peer id (base58) → the latest ping round trip and when it was measured.
#[derive(Debug, Default)]
pub struct PeerRtts {
    rtts: Mutex<HashMap<String, (Duration, Instant)>>,
}

impl PeerRtts {
    /// Record a ping round trip to `peer`.
    pub fn record(&self, peer: &str, rtt: Duration) {
        self.record_at(peer, rtt, Instant::now());
    }

    fn record_at(&self, peer: &str, rtt: Duration, at: Instant) {
        let mut m = self.rtts.lock().unwrap_or_else(|e| e.into_inner());
        if m.len() >= MAX_PEERS && !m.contains_key(peer) {
            if let Some(stalest) = m.iter().min_by_key(|(_, (_, t))| *t).map(|(k, _)| k.clone()) {
                m.remove(&stalest);
            }
        }
        m.insert(peer.to_string(), (rtt, at));
    }

    /// Drop `peer`'s round trip: its last connection closed, so there is no path to measure.
    pub fn forget(&self, peer: &str) {
        self.rtts.lock().unwrap_or_else(|e| e.into_inner()).remove(peer);
    }

    /// The latest round trip to `peer`, if one was measured in the last five minutes and a
    /// connection to it has stayed open since (pings only flow over a live connection, and the
    /// event loop forgets a peer when its last connection closes).
    pub fn get(&self, peer: &str) -> Option<Duration> {
        self.get_at(peer, Instant::now())
    }

    fn get_at(&self, peer: &str, now: Instant) -> Option<Duration> {
        let m = self.rtts.lock().unwrap_or_else(|e| e.into_inner());
        m.get(peer).filter(|(_, at)| now.saturating_duration_since(*at) <= RTT_FRESH).map(|(rtt, _)| *rtt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_latest_fresh_round_trip_is_reported() {
        let r = PeerRtts::default();
        let t0 = Instant::now();
        assert_eq!(r.get_at("p", t0), None);
        r.record_at("p", Duration::from_millis(80), t0);
        r.record_at("p", Duration::from_millis(40), t0);
        assert_eq!(r.get_at("p", t0), Some(Duration::from_millis(40)), "the latest wins");
        assert_eq!(r.get_at("p", t0 + RTT_FRESH + Duration::from_secs(1)), None, "a stale one isn't reported");
        r.forget("p");
        assert_eq!(r.get_at("p", t0), None, "a disconnected peer has none");
    }
}
