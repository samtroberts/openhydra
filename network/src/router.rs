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

/// A candidate provider seen by the router: its advertised canonical model id plus
/// the scoring signals (built from a `DiscoveredPeer` + its §4 capability fields).
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The provider's advertised canonical model id (`""` if it didn't advertise one).
    pub canonical_model_id: String,
    /// Scoring inputs; `score.peer_id` is the peer to route to.
    pub score: PeerScoreInput,
}

/// The result of a successful route.
#[derive(Debug, Clone, PartialEq)]
pub struct RouteOutcome {
    /// The model id actually served (a fallback when `degraded`).
    pub model_id: String,
    pub peer_id: String,
    pub response: Vec<u8>,
    /// True when served by a fallback (nearest-smaller same-family) model.
    pub degraded: bool,
}

/// Why a request could not be routed.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteError {
    /// No compatible, reachable provider across all candidate models.
    NoProvider,
}

/// Resolve → (graceful) degrade → rank → route, with the network I/O **injected**.
///
/// `model_candidates` is an ordered list of `(model_id, request_canonical_id)`: the
/// exact request first, then nearest-smaller same-family fallbacks (the caller
/// supplies the order from the catalog — that governance lives with the catalog).
/// For each candidate model:
///
/// 1. `discover(model_id)` yields its providers;
/// 2. providers whose advertised canonical id is *incompatible* with the request are
///    dropped (a provider that advertised none is kept — backward-compatible);
/// 3. the rest are ranked by [`rank_peers`];
/// 4. `route(peer_id, request)` is tried best-first, failing over to the next-ranked
///    peer on error.
///
/// The first model that yields a successful route wins; `degraded` is set for any
/// non-first candidate. Returns [`RouteError::NoProvider`] if nothing routes.
///
/// I/O is injected so the orchestration is unit-testable without a live swarm; the
/// PyO3 method supplies real `discover`/`route` closures backed by the libp2p
/// `Discover`/`ProxyForward` commands.
pub fn resolve_and_route<D, R>(
    model_candidates: &[(String, String)],
    request: &[u8],
    tier: u8,
    mut discover: D,
    mut route: R,
) -> Result<RouteOutcome, RouteError>
where
    D: FnMut(&str) -> Vec<Candidate>,
    R: FnMut(&str, &[u8]) -> Result<Vec<u8>, String>,
{
    for (idx, (model_id, req_canonical)) in model_candidates.iter().enumerate() {
        let candidates = discover(model_id);
        let compatible: Vec<&Candidate> = candidates
            .iter()
            .filter(|c| {
                req_canonical.is_empty()
                    || c.canonical_model_id.is_empty()
                    || crate::model_id::is_compatible(req_canonical, &c.canonical_model_id)
            })
            .collect();
        if compatible.is_empty() {
            continue; // graceful degradation → next (nearest-smaller) candidate
        }
        let inputs: Vec<PeerScoreInput> = compatible.iter().map(|c| c.score.clone()).collect();
        for sp in rank_peers(&inputs, tier) {
            if let Ok(response) = route(&sp.peer_id, request) {
                return Ok(RouteOutcome {
                    model_id: model_id.clone(),
                    peer_id: sp.peer_id,
                    response,
                    degraded: idx > 0,
                });
            }
            // this peer failed to route → fail over to the next-ranked peer
        }
    }
    Err(RouteError::NoProvider)
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

    // --- resolve_and_route orchestration (I/O injected) ---

    fn candidate(peer_id: &str, canonical: &str, f: impl FnOnce(&mut PeerScoreInput)) -> Candidate {
        let mut score = PeerScoreInput {
            peer_id: peer_id.to_string(),
            latency_ms: 10.0,
            ..Default::default()
        };
        f(&mut score);
        Candidate {
            canonical_model_id: canonical.to_string(),
            score,
        }
    }

    const TPL_A: &str = "qwen3.5/2b/fp16/aaaaaaaaaaaaaaaa";
    const TPL_B: &str = "qwen3.5/2b/fp16/bbbbbbbbbbbbbbbb";

    #[test]
    fn routes_to_highest_ranked_compatible_peer() {
        let cands = vec![
            candidate("slow", TPL_A, |p| p.throughput_tok_s = 5.0),
            candidate("fast", TPL_A, |p| p.throughput_tok_s = 45.0),
        ];
        let out = resolve_and_route(
            &[("openhydra-qwen3.5-2b".into(), "qwen3.5/2b/*/*".into())],
            b"req",
            2,
            |_| cands.clone(),
            |peer, _| Ok(format!("served-by-{peer}").into_bytes()),
        )
        .unwrap();
        assert_eq!(out.peer_id, "fast");
        assert_eq!(out.response, b"served-by-fast");
        assert!(!out.degraded);
    }

    #[test]
    fn filters_incompatible_canonical_id() {
        // The incompatible (different template) peer is dropped despite higher throughput.
        let cands = vec![
            candidate("wrong-tpl", TPL_B, |p| p.throughput_tok_s = 99.0),
            candidate("right", TPL_A, |p| p.throughput_tok_s = 1.0),
        ];
        let out = resolve_and_route(
            &[("m".into(), TPL_A.into())],
            b"req",
            2,
            |_| cands.clone(),
            |peer, _| Ok(peer.as_bytes().to_vec()),
        )
        .unwrap();
        assert_eq!(out.peer_id, "right");
    }

    #[test]
    fn graceful_degradation_to_nearest_smaller() {
        // The 9b request has no live providers → fall back to the smaller 2b model.
        let out = resolve_and_route(
            &[
                ("openhydra-qwen3.5-9b".into(), "qwen3.5/9b/*/*".into()),
                ("openhydra-qwen3.5-2b".into(), "qwen3.5/2b/*/*".into()),
            ],
            b"req",
            2,
            |model_id| {
                if model_id == "openhydra-qwen3.5-2b" {
                    vec![candidate("small", TPL_A, |_| {})]
                } else {
                    vec![] // 9b: no live providers
                }
            },
            |peer, _| Ok(peer.as_bytes().to_vec()),
        )
        .unwrap();
        assert_eq!(out.model_id, "openhydra-qwen3.5-2b");
        assert_eq!(out.peer_id, "small");
        assert!(out.degraded);
    }

    #[test]
    fn route_fails_over_to_next_ranked_peer() {
        let cands = vec![
            candidate("best-but-dead", TPL_A, |p| p.throughput_tok_s = 45.0),
            candidate("backup", TPL_A, |p| p.throughput_tok_s = 5.0),
        ];
        let out = resolve_and_route(
            &[("m".into(), "qwen3.5/2b/*/*".into())],
            b"req",
            2,
            |_| cands.clone(),
            |peer, _| {
                if peer == "best-but-dead" {
                    Err("connection refused".into())
                } else {
                    Ok(peer.as_bytes().to_vec())
                }
            },
        )
        .unwrap();
        assert_eq!(out.peer_id, "backup"); // failed over from the higher-ranked dead peer
    }

    #[test]
    fn no_provider_when_nothing_compatible_anywhere() {
        let err = resolve_and_route(
            &[("m".into(), TPL_A.into())],
            b"req",
            2,
            |_| vec![candidate("x", "gemma-4/e4b/fp16/cccccccccccccccc", |_| {})], // wrong family
            |peer, _| Ok(peer.as_bytes().to_vec()),
        )
        .unwrap_err();
        assert_eq!(err, RouteError::NoProvider);
    }

    #[test]
    fn provider_without_canonical_id_is_kept() {
        // Backward-compat: a legacy provider that advertised no canonical id is eligible.
        let out = resolve_and_route(
            &[("m".into(), TPL_A.into())],
            b"req",
            2,
            |_| vec![candidate("legacy", "", |_| {})],
            |peer, _| Ok(peer.as_bytes().to_vec()),
        )
        .unwrap();
        assert_eq!(out.peer_id, "legacy");
    }
}
