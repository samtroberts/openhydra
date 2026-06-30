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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use openhydra_network::handle::NetworkHandle;
use openhydra_network::types::DiscoveredPeer;
use openhydra_protocol::credit::CreditAccount;
use openhydra_protocol::model_id::is_compatible;
use openhydra_protocol::receipts::NonceTracker;
use openhydra_protocol::router::{rank_peers, PeerScoreInput};
use openhydra_protocol::store::Store;
use openhydra_protocol::verify::{
    resolve_audit, sample_rate_for_reputation, AuditReport, RedundantVerdict, ReputationTracker,
    VerificationOutcome, NEUTRAL_REPUTATION,
};

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

/// Redundant-execution audit sampling bounds (M2.2(b)), fed to
/// [`sample_rate_for_reputation`]. A fully-trusted provider is audited rarely
/// ([`AUDIT_MIN_RATE`]); a fresh / low-reputation one often ([`AUDIT_MAX_RATE`]), since the
/// audit re-runs the work on a second provider and so costs real compute. Tunable.
const AUDIT_MIN_RATE: f64 = 0.02;
const AUDIT_MAX_RATE: f64 = 0.5;

/// Token budget for an audit challenge completion — kept short to bound the audit's compute
/// cost while leaving enough output to discriminate an honest run from a freeloader.
const AUDIT_MAX_TOKENS: u32 = 64;

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
///
/// Uses each provider's *self-reported* DHT reputation (always neutral post-v2, since the
/// signed record excludes `reputation_score`). For **earned** reputation from the
/// consumer's local [`ReputationTracker`], use [`rank_providers_with_reputation`].
pub fn rank_providers(
    peers: &[DiscoveredPeer],
    request_canonical: &str,
    tier: u8,
) -> Vec<SelectedProvider> {
    rank_providers_with_reputation(peers, request_canonical, tier, &|_| None)
}

/// As [`rank_providers`], but `earned_reputation(peer_id)` may override a provider's
/// reputation with the consumer's locally-earned score (M2.2(a)). `Some(score)` wins;
/// `None` falls back to the self-reported DHT value (neutral when absent). This is the
/// closed loop that downranks a provider that has misbehaved with *this* consumer.
pub fn rank_providers_with_reputation(
    peers: &[DiscoveredPeer],
    request_canonical: &str,
    tier: u8,
    earned_reputation: &dyn Fn(&str) -> Option<f64>,
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
            // Earned reputation (local tracker) wins; else the self-reported DHT value;
            // else neutral. A provider that failed *us* ranks below an untried one.
            reputation: earned_reputation(&p.peer_id)
                .unwrap_or(if p.reputation_score > 0.0 { p.reputation_score } else { 50.0 }),
            bandwidth_mbps: 0.0,
            s2s_rtt_ms: 0.0,
            throughput_tok_s: p.throughput_tok_s,
            queue_depth: p.queue_depth,
        })
        .collect();
    let mut ranked: Vec<SelectedProvider> = rank_peers(&inputs, tier)
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
        .collect();

    // R-DHT-8: prefer providers we already have a live connection to. The
    // capability/reputation score (above) decides order *within* the connected
    // and disconnected groups; this stable partition just floats live providers
    // to the front so a stale-but-still-advertised record is only tried after
    // every connected provider — failover as the exception, not the rule.
    let connected: std::collections::HashSet<&str> = peers
        .iter()
        .filter(|p| p.connected)
        .map(|p| p.peer_id.as_str())
        .collect();
    ranked.sort_by_key(|sp| !connected.contains(sp.peer_id.as_str()));
    ranked
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
    /// Per-provider earned reputation (M2.2(a)), keyed by OpenHydra peer_id — the same key
    /// the receipts use. Mutated on `&self` (the gateway calls `complete` from a blocking
    /// task), so it lives behind a `Mutex`.
    reputation: Mutex<HashMap<String, ReputationTracker>>,
    /// Optional durable backing for `reputation` (M2.2(a)) and `credit` (M2.3). When
    /// present, both maps are rehydrated from it at startup and flushed back on each
    /// update, so trust and give/take survive a restart. `None` = ephemeral.
    store: Option<Store>,
    /// M2.3 give-side credit, keyed by **provider libp2p PeerId** — the dial target we
    /// settled a receipt with. When a provider co-signs our receipt it *served* us, so we
    /// record its contribution (`record_served`) here, the symmetric half of the provider's
    /// take-side `record_consumption`. Shares the store's `PEER_CREDIT` table with the
    /// provider role: a node that both serves and provides against the **same `--db`** builds
    /// one give/take ledger per counterparty. (Within a single process only this role
    /// mutates it; cross-process unification merges on rehydrate — full live unification
    /// lands with the combined-role / enforcement work.)
    credit: Mutex<HashMap<String, CreditAccount>>,
}

impl ConsumerNode {
    /// An in-memory consumer (reputation and credit are not persisted across restarts).
    pub fn new(net: NetworkHandle) -> Self {
        Self {
            net,
            tier: 2,
            reputation: Mutex::new(HashMap::new()),
            store: None,
            credit: Mutex::new(HashMap::new()),
        }
    }

    /// A consumer whose earned reputation (M2.2(a)) and give-side credit (M2.3) are
    /// **persisted** to `store`: both maps are rehydrated now and flushed back durably.
    pub fn with_store(net: NetworkHandle, store: Store) -> Self {
        let mut reputation = HashMap::new();
        // The consumer doesn't replay-guard nonces (that's the provider's job); this
        // throwaway tracker just satisfies the shared rehydration API.
        let mut nonces = NonceTracker::new();
        if let Err(e) = store.load_state_into_memory(&mut nonces, &mut reputation) {
            tracing::warn!(error = %e, "could not rehydrate reputation from store; starting fresh");
        }
        let mut credit = HashMap::new();
        if let Err(e) = store.load_credit_into_memory(&mut credit) {
            tracing::warn!(error = %e, "could not rehydrate credit from store; starting fresh");
        }
        Self {
            net,
            tier: 2,
            reputation: Mutex::new(reputation),
            store: Some(store),
            credit: Mutex::new(credit),
        }
    }

    /// The consumer's locally-earned reputation for `peer_id`, decayed to `now_ms`, or
    /// `None` if this consumer has no history with it (→ neutral in ranking).
    fn earned_reputation(&self, peer_id: &str, now_ms: u64) -> Option<f64> {
        self.reputation.lock().ok()?.get(peer_id).map(|t| t.score_at(now_ms))
    }

    /// Record a verification outcome for `peer_id` (M2.2(a)): a served + clean completion
    /// is `Honored`; a failed/refused serve attempt is `Rejected`. Updates the in-memory
    /// tracker, then best-effort persists the snapshot (reputation is advisory — a failed
    /// write must never break the request path). Feeds the next ranking so a provider that
    /// misbehaves with this consumer is downranked out of routing.
    fn record_outcome(&self, peer_id: &str, outcome: VerificationOutcome, now_ms: u64) {
        let snapshot = {
            let mut map = match self.reputation.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            let tracker = map
                .entry(peer_id.to_string())
                .or_insert_with(|| ReputationTracker::new(now_ms));
            tracker.record(outcome, now_ms);
            tracker.to_bytes()
        };
        // Persist outside the lock (don't hold the Mutex across the redb write).
        if let Some(store) = &self.store {
            if let Err(e) = store.put_reputation(peer_id, &snapshot) {
                tracing::debug!(error = %e, peer_id, "reputation persist failed");
            }
        }
    }

    /// The distinct model ids this node currently knows about (PEX-learned / discovered).
    /// Backs the gateway's `GET /v1/models`; empty until discovery has populated the cache.
    pub fn known_models(&self) -> Result<Vec<String>, String> {
        self.net.known_models()
    }

    /// Discover a provider for `model`, pick the best, and stream the completion's text
    /// deltas to `on_delta`. Returns the [`ServeSummary`] (token count → receipt) on
    /// success. `model` is the engine handle / DHT key (e.g. `"qwen2.5:7b"`).
    pub fn complete(
        self: &Arc<Self>,
        model: &str,
        messages: Vec<ChatMessage>,
        max_tokens: Option<u32>,
        temperature: Option<f64>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeSummary, AdapterError> {
        let now = now_unix_ms();
        let t_discover = std::time::Instant::now();
        let peers = self
            .net
            .discover(model)
            .map_err(|e| AdapterError::Http(format!("discover: {e}")))?;
        // "" canonical → any provider of this model_id (template-hash filtering is later).
        // M2.2(a): earned local reputation overrides the (neutral) self-reported score, so
        // a provider that has failed this consumer ranks below an untried one.
        let candidates = rank_providers_with_reputation(&peers, "", self.tier, &|pid| {
            self.earned_reputation(pid, now)
        });
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
                    // M2.2(a): a clean served completion earns the provider reputation.
                    self.record_outcome(&provider.peer_id, VerificationOutcome::Honored, now);
                    // M2.2(b): sampled background redundant-exec audit of this provider (off
                    // the response path — the caller already has its completion).
                    self.maybe_audit(model, &provider.peer_id);
                    return Ok(summary);
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %provider.libp2p_peer_id, attempt = i + 1, total,
                        error = %e, "provider attempt failed"
                    );
                    // M2.2(a): a failed/refused serve attempt costs the provider
                    // reputation, so a dead/erroring one is downranked on the next discover.
                    self.record_outcome(&provider.peer_id, VerificationOutcome::Rejected, now);
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
        // On a *successfully co-signed* receipt the provider has served us `tokens` — credit
        // its contribution (M2.3 give-side), the mirror of the provider's take-side accrual.
        if request_receipt(
            &sign,
            &mut transport,
            &provider_pub,
            &consumer_pub,
            &provider.model_id,
            tokens,
            rand::random::<[u8; 16]>(),
            now_unix_ms(),
        )
        .is_ok()
        {
            self.record_contribution(&provider.libp2p_peer_id, tokens);
        }
    }

    /// Record that `provider_libp2p_id` **served** us `tokens` (M2.3 give-side): the provider
    /// earns give/take credit in our ledger, keyed by its libp2p id. The counterparty is our
    /// own node, so the per-counterparty cap bounds how much any one provider can earn from
    /// serving *us* alone (anti-collusion). Best-effort persisted; a failed write never
    /// affects the completion (credit is advisory).
    fn record_contribution(&self, provider_libp2p_id: &str, tokens: u64) {
        let now = now_unix_ms();
        let me = self.net.libp2p_peer_id().to_string();
        let snapshot = {
            let mut map = match self.credit.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            let acct = map
                .entry(provider_libp2p_id.to_string())
                .or_insert_with(|| CreditAccount::new(now));
            acct.record_served(&me, tokens, now);
            acct.to_bytes()
        };
        if let Some(store) = &self.store {
            if let Err(e) = store.put_credit(provider_libp2p_id, &snapshot) {
                tracing::debug!(error = %e, provider_libp2p_id, "credit persist failed");
            }
        }
    }

    /// The give/take **balance** we hold for a provider (by libp2p id), decayed to now —
    /// `STARTER_GRANT` when we've no record. Positive ⇒ a net contributor to us. Exposed for
    /// observability and tests; the provider role consults its own map's `rate_cap` to
    /// throttle (enforcement is the concurrency-gated step).
    pub fn provider_balance(&self, provider_libp2p_id: &str) -> f64 {
        let now = now_unix_ms();
        self.credit
            .lock()
            .ok()
            .and_then(|m| m.get(provider_libp2p_id).map(|a| a.balance(now)))
            .unwrap_or(openhydra_protocol::credit::STARTER_GRANT)
    }

    /// The redundant-execution audit sampling rate for `peer_id` (M2.2(b)): low for a
    /// trusted provider, high for a fresh / low-reputation one. A background sampler draws
    /// against this to decide whether to [`audit_model`](Self::audit_model) after serving.
    /// Pure read of the earned-reputation map — exposed so the trigger policy is testable
    /// and tunable separately from the dispatch.
    pub fn audit_rate_for(&self, peer_id: &str, now_ms: u64) -> f64 {
        let rep = self.earned_reputation(peer_id, now_ms).unwrap_or(NEUTRAL_REPUTATION);
        sample_rate_for_reputation(rep, AUDIT_MIN_RATE, AUDIT_MAX_RATE)
    }

    /// M2.2(b) trigger: after serving, *sometimes* audit the provider that served. Draws
    /// against [`audit_rate_for`](Self::audit_rate_for) (rare for a trusted provider, up to
    /// [`AUDIT_MAX_RATE`] for a fresh/suspect one); on a hit it spawns a **detached
    /// background thread** that runs [`audit_model`](Self::audit_model) with a fresh
    /// unpredictable [`default_challenge`], so the redundant-exec cross-check never touches
    /// the caller's latency path. The audit records its own outcomes (a confirmed outlier →
    /// `Failed`); here we only log the verdict. A `< 2 providers` error just means the model
    /// can't be cross-checked right now — logged at debug, not an error.
    fn maybe_audit(self: &Arc<Self>, model: &str, provider_peer_id: &str) {
        let rate = self.audit_rate_for(provider_peer_id, now_unix_ms());
        if rand::random::<f64>() >= rate {
            return; // not sampled this time
        }
        let node = Arc::clone(self);
        let model = model.to_string();
        std::thread::spawn(move || match node.audit_model(&model, &default_challenge()) {
            Ok(report) => {
                tracing::info!(model, verdict = ?report.verdict, "redundant-exec audit complete")
            }
            Err(e) => tracing::debug!(model, error = %e, "redundant-exec audit skipped"),
        });
    }

    /// Dispatch `challenge` to one specific `provider` deterministically (`temperature = 0`)
    /// and return its **full** completion text (the audit compares whole outputs, not a
    /// streamed view). Bounded by [`ATTEMPT_TIMEOUT`] so a dead provider can't stall the
    /// audit.
    fn dispatch_full(
        &self,
        provider: &SelectedProvider,
        challenge: &[ChatMessage],
    ) -> Result<String, AdapterError> {
        let request = ServeRequest {
            reply_to: self.net.libp2p_peer_id().to_string(),
            model_ref: provider.model_id.clone(),
            messages: challenge.to_vec(),
            max_tokens: Some(AUDIT_MAX_TOKENS),
            // Redundant-execution comparison only holds for greedy decoding — a sampled
            // (temperature > 0) answer would legitimately differ between honest providers.
            temperature: Some(0.0),
        };
        let provider_libp2p = provider.libp2p_peer_id.clone();
        let mut transport = |framed: &[u8]| -> Result<Vec<u8>, AdapterError> {
            self.net
                .proxy_forward_timeout(provider_libp2p.clone(), framed.to_vec(), ATTEMPT_TIMEOUT)
                .map_err(|e| AdapterError::Http(format!("proxy_forward: {e}")))
        };
        let mut output = String::new();
        let mut sink = |delta: &str| output.push_str(delta);
        request_completion(&mut transport, &request, &mut sink)?;
        Ok(output)
    }

    /// Run one **redundant-execution audit** (M2.2(b)) for `model`: send the same
    /// deterministic `challenge` to up to two distinct providers of the model, compare
    /// their full outputs via [`resolve_audit`], and record the resulting reputation
    /// outcomes (a confirmed outlier → `Failed`; agreeing providers → `Honored`; a
    /// non-responder → `Rejected`). On a 1-vs-1 `Inconclusive` split, escalates to a third
    /// provider (if discovery surfaced one) for a majority before recording.
    ///
    /// This is the *audit* path — separate from [`complete`](Self::complete) and off the
    /// user's latency path (the user already has their answer; a background sampler invokes
    /// this against a sampled fraction, rate from [`audit_rate_for`](Self::audit_rate_for)).
    /// Use an unpredictable `challenge` (see [`default_challenge`]) so a provider can't
    /// pre-cache the answer. Errors only if fewer than two providers are discoverable.
    ///
    /// *Caveat:* providers of the same canonical `model_id` may run different quantizations
    /// / engine builds whose greedy output differs; [`agrees`](openhydra_protocol::verify::agrees)
    /// absorbs benign late divergence and the `Inconclusive`-on-tie rule avoids punishing on
    /// ambiguous evidence, but heterogeneous fleets weaken the signal (reputation still
    /// carries the long-run trust).
    pub fn audit_model(
        &self,
        model: &str,
        challenge: &[ChatMessage],
    ) -> Result<AuditReport, AdapterError> {
        let now = now_unix_ms();
        let peers = self
            .net
            .discover(model)
            .map_err(|e| AdapterError::Http(format!("discover: {e}")))?;
        let candidates =
            rank_providers_with_reputation(&peers, "", self.tier, &|pid| self.earned_reputation(pid, now));
        if candidates.len() < 2 {
            return Err(AdapterError::Http(format!(
                "redundant-exec audit needs ≥2 providers for '{model}', found {}",
                candidates.len()
            )));
        }

        // Dispatch to the top two, then escalate to a third only if they can't decide.
        let mut results: Vec<(String, Result<String, String>)> = Vec::with_capacity(3);
        for provider in candidates.iter().take(2) {
            let out = self.dispatch_full(provider, challenge).map_err(|e| e.to_string());
            results.push((provider.peer_id.clone(), out));
        }
        let mut report = resolve_audit(&results);
        if report.verdict == RedundantVerdict::Inconclusive {
            if let Some(third) = candidates.get(2) {
                let out = self.dispatch_full(third, challenge).map_err(|e| e.to_string());
                results.push((third.peer_id.clone(), out));
                report = resolve_audit(&results);
            }
        }

        for (peer_id, outcome) in &report.outcomes {
            self.record_outcome(peer_id, *outcome, now);
        }
        Ok(report)
    }
}

/// An unpredictable, deterministic-to-answer audit challenge (M2.2(b)). The embedded random
/// nonce defeats answer-caching (a provider can't have precomputed it), and the
/// echo-then-restate instruction yields a long shared prefix that two honest runs of the
/// same model reproduce while a freeloader cannot. Prompt design is heuristic and tunable.
pub fn default_challenge() -> Vec<ChatMessage> {
    let nonce: u128 = rand::random();
    vec![ChatMessage {
        role: "user".to_string(),
        content: format!(
            "Verification probe {nonce:032x}. Respond with exactly, and nothing else: \
             first the token {nonce:032x} on its own line, then on the next line the single \
             sentence \"This is a deterministic audit response.\""
        ),
    }]
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

    #[test]
    fn rank_providers_floats_connected_above_higher_scored_disconnected() {
        // R-DHT-8: a live (connected) provider is preferred even when a
        // disconnected one scores higher on throughput — a stale-but-advertised
        // record shouldn't be dialed first and eat a failover round-trip.
        let mut fast_dead = discovered("fast_dead", TPL_A, 45.0);
        fast_dead.connected = false;
        let mut slow_live = discovered("slow_live", TPL_A, 5.0);
        slow_live.connected = true;
        let ranked = rank_providers(&[fast_dead, slow_live], TPL_A, 2);
        let ids: Vec<&str> = ranked.iter().map(|p| p.peer_id.as_str()).collect();
        // Connected first; the disconnected (higher-scored) one is still present
        // as a failover fallback, just after.
        assert_eq!(ids, vec!["slow_live", "fast_dead"]);
    }

    #[test]
    fn rank_providers_score_order_within_connected_group() {
        // Within the connected group, capability score still decides order.
        let mut fast = discovered("fast", TPL_A, 45.0);
        fast.connected = true;
        let mut slow = discovered("slow", TPL_A, 5.0);
        slow.connected = true;
        let ranked = rank_providers(&[slow, fast], TPL_A, 2);
        let ids: Vec<&str> = ranked.iter().map(|p| p.peer_id.as_str()).collect();
        assert_eq!(ids, vec!["fast", "slow"]);
    }

    // ── M2.2(a): earned-reputation-aware ranking ──

    #[test]
    fn earned_reputation_downranks_a_provider_that_failed_us() {
        // The closed loop: two equal-throughput providers, both self-reporting neutral;
        // the one with a low EARNED reputation ranks last.
        let peers = vec![discovered("bad", TPL_A, 20.0), discovered("good", TPL_A, 20.0)];
        let earned = |pid: &str| if pid == "bad" { Some(5.0) } else { None };
        let ranked = rank_providers_with_reputation(&peers, TPL_A, 2, &earned);
        let ids: Vec<&str> = ranked.iter().map(|p| p.peer_id.as_str()).collect();
        assert_eq!(ids, vec!["good", "bad"], "low earned reputation must rank a provider last");
    }

    #[test]
    fn default_challenge_is_unpredictable_and_well_formed() {
        // A single user-role message; the random nonce makes successive challenges differ
        // (so a provider can't pre-cache the answer) and appear twice (echo + restate).
        let c1 = default_challenge();
        let c2 = default_challenge();
        assert_eq!(c1.len(), 1);
        assert_eq!(c1[0].role, "user");
        assert_ne!(c1[0].content, c2[0].content, "challenge nonce must vary");
        assert!(c1[0].content.contains("deterministic audit response"));
    }

    #[test]
    fn no_earned_history_falls_back_to_the_plain_ranking() {
        // Earned reputation only ever *overrides* — with no history (lookup → None for
        // all), the result must equal the plain reputation-agnostic ranking.
        let peers = vec![discovered("a", TPL_A, 5.0), discovered("b", TPL_A, 45.0)];
        let with_none = rank_providers_with_reputation(&peers, TPL_A, 2, &|_| None);
        let plain = rank_providers(&peers, TPL_A, 2);
        assert_eq!(
            with_none.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>(),
            plain.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>(),
        );
    }
}
