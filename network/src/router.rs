// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Router scoring & ranking (protocol.md §5) — M1.3 scaffold.
//!
//! Ports the ranking logic from `coordinator/peer_selector.py` into Rust. This is
//! the *rank* stage of the router's resolve → rank → route pipeline; the resolve
//! (DHT lookup) and route (libp2p forward) stages, plus graceful degradation, land
//! on top of this in the rest of M1.3.
//!
//! Candidates are scored (higher is better) by ping latency, load headroom,
//! reputation, bandwidth, server-to-server RTT, advertised throughput (§4), and
//! live queue depth (§4). The scoring mirrors the Python implementation so the two
//! agree during the transition.

/// The scoring inputs for one candidate provider.
#[derive(Debug, Clone)]
pub struct PeerScoreInput {
    pub peer_id: String,
    /// Ping latency to the peer (ms); lower is better.
    pub latency_ms: f64,
    /// Reported load percentage (0–100); lower is better.
    pub load_pct: f64,
    /// Reputation from verification history (0–100); higher is better.
    pub reputation: f64,
    /// Advertised bandwidth (Mbps); higher is better.
    pub bandwidth_mbps: f64,
    /// Mean measured server-to-server RTT to downstream peers (ms); 0 = unknown.
    pub s2s_rtt_ms: f64,
    /// Advertised live decode throughput (tokens/s, §4); higher is better.
    pub throughput_tok_s: f64,
    /// Live queued/in-flight request count (§4); lower is better.
    pub queue_depth: u32,
}

impl Default for PeerScoreInput {
    fn default() -> Self {
        Self {
            peer_id: String::new(),
            latency_ms: 1.0,
            load_pct: 0.0,
            reputation: 50.0,
            bandwidth_mbps: 0.0,
            s2s_rtt_ms: 0.0,
            throughput_tok_s: 0.0,
            queue_depth: 0,
        }
    }
}

/// A scored candidate, produced by [`rank_peers`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredPeer {
    pub peer_id: String,
    pub score: f64,
}

/// Compute a routing score for one candidate (higher is better).
///
/// `tier` selects the weighting profile, mirroring the Python
/// `compute_routing_score`: `tier <= 2` is latency-focused, `tier > 2` is balanced.
/// Both incorporate the §4 throughput and queue-depth signals.
pub fn compute_routing_score(input: &PeerScoreInput, tier: u8) -> f64 {
    let latency_ms = input.latency_ms.max(1.0);
    let headroom = (100.0 - input.load_pct).max(1.0);
    let rep_norm = input.reputation.clamp(0.0, 100.0) / 100.0;
    let bw_norm = input.bandwidth_mbps.max(0.0) / 1000.0;
    // S2S RTT: 0 means no measurement available (neutral). Higher = worse.
    let s2s_penalty = if input.s2s_rtt_ms > 0.0 {
        1.0 / input.s2s_rtt_ms.max(1.0)
    } else {
        0.5
    };
    // Throughput (§4): normalise against a 50 tok/s reference; higher is better.
    let tput_norm = (input.throughput_tok_s.max(0.0) / 50.0).min(1.0);
    // Queue depth (§4): fewer queued/in-flight requests is better.
    let queue_term = 1.0 / (1.0 + f64::from(input.queue_depth));

    // ping, load, reputation, bandwidth, S2S, throughput, queue
    let (w1, w2, w3, w4, w5, w6, w7) = if tier <= 2 {
        (0.30, 0.18, 0.15, 0.05, 0.07, 0.15, 0.10)
    } else {
        (0.20, 0.15, 0.20, 0.10, 0.10, 0.15, 0.10)
    };

    w1 * (1.0 / latency_ms)
        + w2 * (headroom / 100.0)
        + w3 * rep_norm
        + w4 * bw_norm
        + w5 * s2s_penalty
        + w6 * tput_norm
        + w7 * queue_term
}

/// Rank candidates best-first.
///
/// `tier == 1` is the pure latency + S2S-RTT profile (no §4 signals), matching the
/// Python `rank_peers` tier-1 branch; `tier >= 2` uses [`compute_routing_score`].
/// The sort is stable, so equal-scoring peers keep their input order.
pub fn rank_peers(peers: &[PeerScoreInput], tier: u8) -> Vec<ScoredPeer> {
    let mut scored: Vec<ScoredPeer> = peers
        .iter()
        .map(|p| {
            let score = if tier == 1 {
                let base = 1.0 / p.latency_ms.max(1.0);
                let s2s_penalty = if p.s2s_rtt_ms > 0.0 {
                    1.0 / p.s2s_rtt_ms.max(1.0)
                } else {
                    0.5
                };
                0.85 * base + 0.15 * s2s_penalty
            } else {
                compute_routing_score(p, tier)
            };
            ScoredPeer {
                peer_id: p.peer_id.clone(),
                score,
            }
        })
        .collect();
    // Stable sort by score descending (NaN-safe: treat as equal).
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an input varying one dimension; all else held at neutral defaults.
    fn input(peer_id: &str, f: impl FnOnce(&mut PeerScoreInput)) -> PeerScoreInput {
        let mut p = PeerScoreInput {
            peer_id: peer_id.to_string(),
            latency_ms: 10.0,
            ..Default::default()
        };
        f(&mut p);
        p
    }

    fn order(peers: &[PeerScoreInput], tier: u8) -> Vec<String> {
        rank_peers(peers, tier).into_iter().map(|s| s.peer_id).collect()
    }

    #[test]
    fn ranks_by_rtt() {
        let peers = [
            input("far", |p| p.latency_ms = 80.0),
            input("near", |p| p.latency_ms = 5.0),
        ];
        assert_eq!(order(&peers, 2)[0], "near");
    }

    #[test]
    fn ranks_by_health_load() {
        let peers = [
            input("loaded", |p| p.load_pct = 90.0),
            input("free", |p| p.load_pct = 5.0),
        ];
        assert_eq!(order(&peers, 2)[0], "free");
    }

    #[test]
    fn ranks_by_throughput() {
        let peers = [
            input("slow", |p| p.throughput_tok_s = 5.0),
            input("fast", |p| p.throughput_tok_s = 45.0),
        ];
        assert_eq!(order(&peers, 2), vec!["fast", "slow"]);
    }

    #[test]
    fn ranks_by_queue_depth() {
        let peers = [
            input("busy", |p| p.queue_depth = 8),
            input("idle", |p| p.queue_depth = 0),
        ];
        assert_eq!(order(&peers, 2), vec!["idle", "busy"]);
    }

    #[test]
    fn seeded_multi_peer_overall_order() {
        let peers = [
            input("mid", |p| {
                p.latency_ms = 20.0;
                p.load_pct = 40.0;
                p.throughput_tok_s = 20.0;
                p.queue_depth = 2;
            }),
            input("best", |p| {
                p.latency_ms = 5.0;
                p.load_pct = 10.0;
                p.throughput_tok_s = 48.0;
                p.queue_depth = 0;
            }),
            input("worst", |p| {
                p.latency_ms = 90.0;
                p.load_pct = 85.0;
                p.throughput_tok_s = 4.0;
                p.queue_depth = 9;
            }),
        ];
        let o = order(&peers, 2);
        assert_eq!(o.first().unwrap(), "best");
        assert_eq!(o.last().unwrap(), "worst");
    }

    #[test]
    fn tier1_ignores_throughput_and_queue() {
        // Tier 1 is pure latency + S2S; equal latency ties regardless of §4 signals.
        let peers = [
            input("a", |p| {
                p.throughput_tok_s = 5.0;
                p.queue_depth = 9;
            }),
            input("b", |p| {
                p.throughput_tok_s = 50.0;
                p.queue_depth = 0;
            }),
        ];
        let scored = rank_peers(&peers, 1);
        assert_eq!(scored[0].score, scored[1].score);
    }
}
