// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Consumer-side serve client — the core the HTTP/SSE gateway calls.
//!
//! [`request_completion`] sends a [`ServeRequest`] to a chosen provider over an injected
//! transport (live: a `proxy_forward` to the provider's libp2p id), parses the buffered
//! framed response, and pushes each text delta to a callback (→ SSE). Transport is
//! injected so the request→serve→response→parse loop is unit-tested in-process against
//! the provider handler — no swarm, no engine.
//!
//! Provider *selection* (discover → filter → rank → pick) and the HTTP front door land on
//! top of this; this is just "talk to one provider".

use openhydra_network::handle::NetworkHandle;
use openhydra_network::types::DiscoveredPeer;
use openhydra_protocol::model_id::is_compatible;
use openhydra_protocol::router::{rank_peers, PeerScoreInput};

use crate::adapter::{AdapterError, ChatMessage};
use crate::provider::SERVE_REQUEST;
use crate::receipt::request_receipt;
use crate::serve::{parse_response, ServeChunk, ServeRequest, ServeSummary};

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The provider the consumer chose to serve a request.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedProvider {
    /// libp2p id — the `proxy_forward` dial target.
    pub libp2p_peer_id: String,
    /// OpenHydra id — the reputation / receipt key.
    pub peer_id: String,
    /// Provider's ed25519 public key (hex) — for the co-signed receipt at EOS.
    pub public_key: String,
    /// The model id the provider serves.
    pub model_id: String,
}

/// Per-provider attempt budget for the serve round-trip. Generous enough for a real
/// generation (including a cold model load on the provider), but bounded so a dead /
/// stale-but-advertised provider frees its slot for failover instead of hanging the
/// request on libp2p's ~15s (or unbounded) request-response wait.
const ATTEMPT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(45);

/// Pick the best provider from `peers` for a request whose canonical id is
/// `request_canonical` (use `""` to match any provider of the discovered model_id).
///
/// Filters by canonical-id compatibility (a provider that advertised none is kept —
/// backward-compatible), ranks the rest with the shared router scoring
/// ([`rank_peers`], reputation-aware), and returns the top. `None` if nothing compatible.
pub fn select_provider(
    peers: &[DiscoveredPeer],
    request_canonical: &str,
    tier: u8,
) -> Option<SelectedProvider> {
    rank_providers(peers, request_canonical, tier).into_iter().next()
}

/// All compatible providers for the request, **in preference order** (best first).
///
/// Same filter + [`rank_peers`] scoring as [`select_provider`], but returns the whole
/// ranked list so the caller can fail over to the next provider when one is dead or
/// unreachable (discovery can surface stale-but-still-advertised providers).
pub fn rank_providers(
    peers: &[DiscoveredPeer],
    request_canonical: &str,
    tier: u8,
) -> Vec<SelectedProvider> {
    let compatible: Vec<&DiscoveredPeer> = peers
        .iter()
        .filter(|p| {
            request_canonical.is_empty()
                || p.canonical_model_id.is_empty()
                || is_compatible(request_canonical, &p.canonical_model_id)
        })
        .collect();
    if compatible.is_empty() {
        return Vec::new();
    }
    let inputs: Vec<PeerScoreInput> = compatible
        .iter()
        .map(|p| PeerScoreInput {
            peer_id: p.peer_id.clone(),
            latency_ms: 1.0, // RTT survey deferred (matches the M1.3 router default)
            load_pct: p.load_pct,
            reputation: if p.reputation_score > 0.0 { p.reputation_score } else { 50.0 },
            bandwidth_mbps: 0.0,
            s2s_rtt_ms: 0.0,
            throughput_tok_s: p.throughput_tok_s,
            queue_depth: p.queue_depth,
        })
        .collect();
    rank_peers(&inputs, tier)
        .into_iter()
        .filter_map(|ranked| {
            compatible
                .iter()
                .find(|p| p.peer_id == ranked.peer_id)
                .map(|p| SelectedProvider {
                    libp2p_peer_id: p.libp2p_peer_id.clone(),
                    peer_id: p.peer_id.clone(),
                    public_key: p.public_key.clone(),
                    model_id: p.model_id.clone(),
                })
        })
        .collect()
}

/// Send `request` over `transport`, stream each delta to `on_delta`, and return the
/// [`ServeSummary`] (token count for the receipt) on a clean `Done`.
///
/// `transport(framed_request) -> response_bytes` is the network round-trip — live, a
/// `proxy_forward` to the chosen provider. Errors if the transport fails, the provider
/// returns an `Error` frame, or the stream ends without a `Done`.
pub fn request_completion(
    transport: &mut dyn FnMut(&[u8]) -> Result<Vec<u8>, AdapterError>,
    request: &ServeRequest,
    on_delta: &mut dyn FnMut(&str),
) -> Result<ServeSummary, AdapterError> {
    let mut framed = Vec::with_capacity(1 + request.messages.len() * 32);
    framed.push(SERVE_REQUEST);
    framed.extend_from_slice(&request.encode());

    let response = transport(&framed)?;

    for chunk in parse_response(&response)? {
        match chunk {
            ServeChunk::Delta(text) => on_delta(&text),
            ServeChunk::Done { tokens, metrics } => {
                return Ok(ServeSummary {
                    tokens,
                    ok: true,
                    metrics,
                    discover_ns: 0,
                    proxy_roundtrip_ns: 0,
                })
            }
            ServeChunk::Error(msg) => return Err(AdapterError::Http(msg)),
        }
    }
    Err(AdapterError::Parse(
        "serve response ended without a Done/Error frame".into(),
    ))
}

/// A consumer node: discovers providers and serves completions over the swarm — the
/// **synchronous core** the HTTP/SSE gateway wraps (`complete` blocks; the gateway calls
/// it from a `spawn_blocking` task and streams the deltas).
pub struct ConsumerNode {
    net: NetworkHandle,
    tier: u8,
}

impl ConsumerNode {
    pub fn new(net: NetworkHandle) -> Self {
        Self { net, tier: 2 }
    }

    /// Discover a provider for `model`, pick the best, and stream the completion's text
    /// deltas to `on_delta`. Returns the [`ServeSummary`] (token count → receipt) on
    /// success. `model` is the engine handle / DHT key (e.g. `"qwen2.5:7b"`).
    pub fn complete(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        max_tokens: Option<u32>,
        temperature: Option<f64>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeSummary, AdapterError> {
        let t_discover = std::time::Instant::now();
        let peers = self
            .net
            .discover(model)
            .map_err(|e| AdapterError::Http(format!("discover: {e}")))?;
        // "" canonical → any provider of this model_id (template-hash filtering is later).
        let candidates = rank_providers(&peers, "", self.tier);
        let discover_ns = t_discover.elapsed().as_nanos() as u64;
        tracing::debug!(elapsed = ?t_discover.elapsed(), candidates = candidates.len(), "discover");
        if candidates.is_empty() {
            return Err(AdapterError::Http(format!("no provider for model '{model}'")));
        }
        let total = candidates.len();

        // Try providers in preference order. A dead/stale-but-advertised provider can't
        // hang the request (bounded proxy_forward) and we fail over to the next — but only
        // while nothing has been streamed yet, since partial output can't be un-sent.
        let mut delivered = false;
        let mut last_err: Option<AdapterError> = None;
        for (i, provider) in candidates.into_iter().enumerate() {
            let request = ServeRequest {
                reply_to: self.net.libp2p_peer_id().to_string(),
                model_ref: provider.model_id.clone(),
                messages: messages.clone(),
                max_tokens,
                temperature,
            };
            let provider_libp2p = provider.libp2p_peer_id.clone();
            let t_serve = std::time::Instant::now();
            let mut proxy_roundtrip_ns = 0u64;
            let result = {
                let rt = &mut proxy_roundtrip_ns;
                let mut transport = |framed: &[u8]| -> Result<Vec<u8>, AdapterError> {
                    let t = std::time::Instant::now();
                    let r = self
                        .net
                        .proxy_forward_timeout(
                            provider_libp2p.clone(),
                            framed.to_vec(),
                            ATTEMPT_TIMEOUT,
                        )
                        .map_err(|e| AdapterError::Http(format!("proxy_forward: {e}")));
                    *rt = t.elapsed().as_nanos() as u64;
                    r
                };
                let mut guarded = |d: &str| {
                    delivered = true;
                    on_delta(d);
                };
                request_completion(&mut transport, &request, &mut guarded)
            };
            match result {
                Ok(mut summary) => {
                    summary.discover_ns = discover_ns;
                    summary.proxy_roundtrip_ns = proxy_roundtrip_ns;
                    tracing::debug!(elapsed = ?t_serve.elapsed(), attempt = i + 1, "serve ok");
                    // Settle the co-signed receipt at EOS (best-effort — tokens already
                    // delivered; a failed/slow settlement must not fail the completion).
                    if summary.ok && summary.tokens > 0 {
                        self.settle_receipt(&provider, summary.tokens);
                    }
                    return Ok(summary);
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %provider.libp2p_peer_id, attempt = i + 1, total,
                        error = %e, "provider attempt failed"
                    );
                    if delivered {
                        // Already streamed part of a completion to the client — failing over
                        // would duplicate output. Surface the error instead.
                        return Err(e);
                    }
                    last_err = Some(e);
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| AdapterError::Http(format!("all providers failed for '{model}'"))))
    }

    /// Fire the co-signed receipt for a completed request. Skips a provider that
    /// advertised no usable public key; swallows all errors (trust settlement is
    /// auxiliary to delivering the completion).
    fn settle_receipt(&self, provider: &SelectedProvider, tokens: u64) {
        let provider_pub = match hex::decode(&provider.public_key) {
            Ok(b) if b.len() == 32 => b,
            _ => return, // legacy / unkeyed provider — nothing to settle against
        };
        let consumer_pub = match self.net.public_key_bytes() {
            Ok(b) => b,
            Err(_) => return,
        };
        let sign = |msg: &[u8]| self.net.sign(msg).unwrap_or_default();
        let provider_libp2p = provider.libp2p_peer_id.clone();
        let mut transport = |framed: &[u8]| -> Result<Vec<u8>, AdapterError> {
            self.net
                .proxy_forward(provider_libp2p.clone(), framed.to_vec())
                .map_err(AdapterError::Http)
        };
        let _ = request_receipt(
            &sign,
            &mut transport,
            &provider_pub,
            &consumer_pub,
            &provider.model_id,
            tokens,
            rand::random::<[u8; 16]>(),
            now_unix_ms(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{DetectedModel, EngineAdapter, InferenceRequest, ServeOutcome};
    use crate::provider::handle_serve_inbound;

    /// A canned engine: emits fixed deltas (or fails).
    struct StubAdapter {
        deltas: Vec<&'static str>,
        tokens: u64,
        fail: Option<&'static str>,
    }
    impl EngineAdapter for StubAdapter {
        fn engine_name(&self) -> &'static str {
            "stub"
        }
        fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
            Ok(vec![])
        }
        fn serve_stream(
            &self,
            _req: &InferenceRequest,
            on_delta: &mut dyn FnMut(&str),
        ) -> Result<ServeOutcome, AdapterError> {
            if let Some(e) = self.fail {
                return Err(AdapterError::Http(e.into()));
            }
            for d in &self.deltas {
                on_delta(d);
            }
            Ok(ServeOutcome { tokens: self.tokens, done: true, engine: Default::default() })
        }
    }

    fn request() -> ServeRequest {
        ServeRequest {
            reply_to: "12D3KooWConsumer".into(),
            model_ref: "qwen2.5:7b".into(),
            messages: vec![ChatMessage { role: "user".into(), content: "hi".into() }],
            max_tokens: Some(64),
            temperature: None,
        }
    }

    #[test]
    fn round_trips_request_through_the_provider_handler() {
        // The mock transport IS the provider: consumer encodes → provider serves → consumer
        // parses. Proves the full in-process loop without a swarm.
        let adapter = StubAdapter { deltas: vec!["Hello", " world"], tokens: 5, fail: None };
        let mut transport = |req: &[u8]| -> Result<Vec<u8>, AdapterError> {
            Ok(handle_serve_inbound(req, &adapter))
        };
        let mut out = String::new();
        let summary =
            request_completion(&mut transport, &request(), &mut |d| out.push_str(d)).unwrap();
        assert_eq!(out, "Hello world");
        assert_eq!(summary.tokens, 5);
        assert!(summary.ok);
    }

    #[test]
    fn provider_error_frame_becomes_an_error() {
        let adapter = StubAdapter { deltas: vec![], tokens: 0, fail: Some("engine down") };
        let mut transport = |req: &[u8]| Ok(handle_serve_inbound(req, &adapter));
        let err = request_completion(&mut transport, &request(), &mut |_| {}).unwrap_err();
        assert!(matches!(err, AdapterError::Http(m) if m.contains("engine down")));
    }

    #[test]
    fn transport_failure_propagates() {
        let mut transport =
            |_: &[u8]| -> Result<Vec<u8>, AdapterError> { Err(AdapterError::Http("no route".into())) };
        let err = request_completion(&mut transport, &request(), &mut |_| {}).unwrap_err();
        assert!(matches!(err, AdapterError::Http(_)));
    }

    #[test]
    fn missing_done_frame_is_an_error() {
        // A response with only a delta (no Done) → the consumer flags an incomplete stream.
        let mut transport = |_: &[u8]| {
            Ok(crate::serve::frame_response(&[ServeChunk::Delta("x".into()).encode()]))
        };
        let err = request_completion(&mut transport, &request(), &mut |_| {}).unwrap_err();
        assert!(matches!(err, AdapterError::Parse(_)));
    }

    // ── provider selection ──

    const TPL_A: &str = "qwen3.5/2b/fp16/aaaaaaaaaaaaaaaa";
    const TPL_B: &str = "qwen3.5/2b/fp16/bbbbbbbbbbbbbbbb";

    fn discovered(peer_id: &str, canonical: &str, throughput: f64) -> DiscoveredPeer {
        DiscoveredPeer {
            peer_id: peer_id.into(),
            libp2p_peer_id: format!("{peer_id}-libp2p"),
            public_key: format!("{peer_id}-pk"),
            model_id: "m".into(),
            canonical_model_id: canonical.into(),
            throughput_tok_s: throughput,
            ..Default::default()
        }
    }

    #[test]
    fn selects_highest_ranked_compatible_provider() {
        let peers = vec![discovered("slow", TPL_A, 5.0), discovered("fast", TPL_A, 45.0)];
        let sel = select_provider(&peers, "qwen3.5/2b/*/*", 2).unwrap();
        assert_eq!(sel.peer_id, "fast");
        assert_eq!(sel.libp2p_peer_id, "fast-libp2p"); // dial target
        assert_eq!(sel.public_key, "fast-pk"); // for the receipt
    }

    #[test]
    fn filters_incompatible_canonical_id() {
        let peers = vec![discovered("wrong", TPL_B, 99.0), discovered("right", TPL_A, 1.0)];
        let sel = select_provider(&peers, TPL_A, 2).unwrap();
        assert_eq!(sel.peer_id, "right"); // incompatible dropped despite higher throughput
    }

    #[test]
    fn keeps_legacy_provider_without_canonical_id() {
        let sel = select_provider(&[discovered("legacy", "", 10.0)], "qwen3.5/2b/*/*", 2).unwrap();
        assert_eq!(sel.peer_id, "legacy");
    }

    #[test]
    fn none_when_nothing_compatible_or_empty() {
        let peers = vec![discovered("x", "gemma-4/e4b/fp16/cccccccccccccccc", 10.0)];
        assert!(select_provider(&peers, TPL_A, 2).is_none());
        assert!(select_provider(&[], TPL_A, 2).is_none());
    }

    #[test]
    fn rank_providers_returns_all_compatible_best_first_for_failover() {
        // The failover list: every compatible provider, ranked — so a dead top pick can
        // hand off to the next. Incompatible ones are still dropped.
        let peers = vec![
            discovered("slow", TPL_A, 5.0),
            discovered("incompatible", TPL_B, 99.0),
            discovered("fast", TPL_A, 45.0),
        ];
        let ranked = rank_providers(&peers, TPL_A, 2);
        let ids: Vec<&str> = ranked.iter().map(|p| p.peer_id.as_str()).collect();
        assert_eq!(ids, vec!["fast", "slow"]); // best first, incompatible excluded
        assert!(rank_providers(&[], TPL_A, 2).is_empty());
    }
}
