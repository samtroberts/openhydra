// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Provider-side swarm wiring: advertise an engine's models to the DHT and serve inbound
//! requests by proxying to the engine.
//!
//! [`Provider`] joins an [`EngineAdapter`] to a [`NetworkHandle`]:
//! * [`announce_models`](Provider::announce_models) — detect the engine's models and
//!   publish a [`PeerRecord`] (canonical id + this node's libp2p id + ed25519 key) for
//!   each, so consumers/routers can discover them.
//! * [`run_inbound`](Provider::run_inbound) — the blocking serve loop: poll inbound proxy
//!   requests, dispatch the serve method byte to [`handle_serve_inbound`], and reply with
//!   the (buffered) completion.
//!
//! The pure pieces — record construction and the method-byte dispatch — are unit-tested;
//! the loop itself is thin glue over the live swarm.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openhydra_network::handle::NetworkHandle;
use openhydra_network::types::PeerRecord;
use openhydra_protocol::credit::{throttle_multiplier, CreditAccount};
use openhydra_protocol::receipts::CoSignedReceipt;
use openhydra_protocol::store::Store;

use crate::adapter::{AdapterError, DetectedModel, EngineAdapter};
use crate::aup::{AupDecision, AupPolicy};
use crate::receipt::{handle_receipt_inbound, RECEIPT_REQUEST};
use crate::serve::{frame_response, handle_serve_request, ServeChunk, ServeRequest};
use crate::status::TransferStats;
use crate::workpool::WorkerPool;

/// Base serve delay scaled by [`throttle_multiplier`] for a throttled (leecher) consumer
/// (M2.3 enforcement). Modest — even the deepest leecher's wait is `BASE_THROTTLE *
/// MAX_THROTTLE_MULT` (≈1.8s at the defaults), a slowdown, never a block. Tunable.
const BASE_THROTTLE: Duration = Duration::from_millis(200);

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// libp2p proxy method byte for an inference serve request (consumer → provider). Sits
/// alongside the existing peer method bytes (0x01 Forward … 0x07 Receipt).
pub const SERVE_REQUEST: u8 = 0x10;

/// Build the DHT record advertising one detected model.
///
/// `model_id` (the DHT key consumers look up) is the **engine handle** (e.g. Ollama's
/// `"qwen2.5:7b"`) — the familiar name a consumer requests; it's also what the provider's
/// adapter serves (`ServeRequest.model_ref`). `canonical_model_id` carries the precise
/// `family/params/quant/template_hash` for the router's `is_compatible` filter.
/// `host`/`port` are advisory — routing is by `libp2p_peer_id`.
///
/// MVP-limitation: keying on the engine handle means the same model served via two
/// different engines (Ollama vs vLLM) advertises under different ids; a normalized
/// cross-engine discovery key is the model-id-governance refinement (deferred).
pub fn build_peer_record(
    model: &DetectedModel,
    openhydra_peer_id: &str,
    libp2p_peer_id: &str,
    public_key_hex: &str,
    host: &str,
    port: u16,
) -> PeerRecord {
    PeerRecord {
        peer_id: openhydra_peer_id.to_string(),
        model_id: model.engine_ref.clone(),
        host: host.to_string(),
        port,
        canonical_model_id: model.canonical_id.clone(),
        libp2p_peer_id: libp2p_peer_id.to_string(),
        public_key: public_key_hex.to_string(),
        ..Default::default()
    }
}

/// Provider-side dispatch for one inbound proxy request → the buffered serve response.
///
/// On the [`SERVE_REQUEST`] method byte, run the request through `adapter` and return the
/// length-framed [`ServeChunk`](crate::serve::ServeChunk) sequence. Any other method byte
/// yields a single framed `Error` chunk (this loop only speaks the serve protocol).
pub fn handle_serve_inbound(data: &[u8], adapter: &dyn EngineAdapter) -> Vec<u8> {
    if data.first() != Some(&SERVE_REQUEST) {
        return frame_response(&[
            crate::serve::ServeChunk::Error("unsupported method".into()).encode(),
        ]);
    }
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    handle_serve_request(&data[1..], adapter, &mut |c| chunks.push(c.to_vec()));
    frame_response(&chunks)
}

/// Persist an accepted co-signed receipt to the ledger, if a store is configured.
///
/// Best-effort and side-effecting only: a missing store or a ledger error never fails the
/// request (the consumer already holds the same co-signed receipt as its own proof). Pure
/// enough to unit-test without a live node — the signing/co-signing happens upstream in
/// [`handle_receipt_inbound`]; this is just the persistence decision.
fn ledger_receipt(store: Option<&Store>, accepted: Option<&CoSignedReceipt>) {
    if let (Some(store), Some(receipt)) = (store, accepted) {
        match store.record_receipt(receipt) {
            Ok(true) => {
                eprintln!("openhydra-agent: ledgered receipt ({} tokens)", receipt.payload.tokens)
            }
            Ok(false) => eprintln!("openhydra-agent: receipt replay ignored (nonce already spent)"),
            Err(e) => eprintln!("openhydra-agent: ledger error (receipt not persisted): {e}"),
        }
    }
}

/// An engine joined to the swarm: advertises its models and serves inbound requests.
pub struct Provider<A: EngineAdapter> {
    adapter: A,
    net: NetworkHandle,
    host: String,
    port: u16,
    /// Ledger for accepted co-signed receipts. `None` → receipts are co-signed and returned
    /// but not persisted (the swarm still works; this node just keeps no local record).
    store: Option<Store>,
    /// M2.3 give/take credit, keyed by **consumer libp2p PeerId** — the same id a consumer
    /// announces as its `reply_to`, derived at receipt time from the consumer's ed25519
    /// pubkey. Rehydrated from `store` when one is attached, flushed on each accrual.
    credit: Mutex<HashMap<String, CreditAccount>>,
    /// Count of worker threads currently sleeping in an M2.3 throttle delay. Read to reserve
    /// at least one non-throttling worker (see [`maybe_throttle`](Self::maybe_throttle)), so a
    /// flood of leechers can never put every worker to sleep at once.
    throttling: AtomicUsize,
    /// Acceptable-use policy floor: an inbound serve request the operator's policy refuses is
    /// answered with an `Error` frame instead of being run through the engine. Default
    /// permissive (allow everything) until limits are configured.
    aup: AupPolicy,
    /// P0 introspection: shared transfer counters read by the `--status-bind` server.
    /// `None` → no counting (status endpoint not enabled).
    stats: Option<Arc<TransferStats>>,
}

impl<A: EngineAdapter> Provider<A> {
    pub fn new(adapter: A, net: NetworkHandle) -> Self {
        Self {
            adapter,
            net,
            host: String::new(),
            port: 0,
            store: None,
            credit: Mutex::new(HashMap::new()),
            throttling: AtomicUsize::new(0),
            aup: AupPolicy::permissive(),
            stats: None,
        }
    }

    /// Attach shared transfer counters (P0 introspection — the `--status-bind` server
    /// reads them; the serve path updates them once per request).
    pub fn with_stats(mut self, stats: Arc<TransferStats>) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Set the advisory host/port advertised in records (routing is via libp2p regardless).
    pub fn with_address(mut self, host: impl Into<String>, port: u16) -> Self {
        self.host = host.into();
        self.port = port;
        self
    }

    /// Attach an acceptable-use policy (AUP floor) applied to inbound serve requests. Default
    /// is permissive (allow everything).
    pub fn with_aup(mut self, aup: AupPolicy) -> Self {
        self.aup = aup;
        self
    }

    /// AUP floor: if `data` is a serve request the policy refuses, return the framed `Error`
    /// to send back **instead of** serving it; otherwise `None` (serve normally). Receipts and
    /// unparseable frames are not policy-checked here — they fall through to `dispatch`.
    fn aup_refusal(&self, data: &[u8]) -> Option<Vec<u8>> {
        if !self.aup.is_active() || data.first() != Some(&SERVE_REQUEST) {
            return None;
        }
        let req = ServeRequest::decode(&data[1..]).ok()?;
        match self.aup.evaluate(&req.messages, req.max_tokens) {
            AupDecision::Deny(reason) => Some(frame_response(&[ServeChunk::Error(format!(
                "rejected by acceptable-use policy: {reason}"
            ))
            .encode()])),
            AupDecision::Allow => None,
        }
    }

    /// Attach a ledger so accepted co-signed receipts are persisted (M2.3), and rehydrate
    /// the give/take credit map from it so throttling survives a restart.
    pub fn with_store(self, store: Store) -> Self {
        if let Ok(mut map) = self.credit.lock() {
            if let Err(e) = store.load_credit_into_memory(&mut map) {
                eprintln!("openhydra-agent: could not rehydrate credit from store: {e}");
            }
        }
        Self { store: Some(store), ..self }
    }

    /// Accrue a consumer's give/take after it co-signs a receipt: it **consumed**
    /// `tokens` from us (M2.3). Keyed by the consumer's libp2p id (derived from the
    /// receipt's ed25519 pubkey), so it matches the `reply_to` seen at serve time. The
    /// snapshot is flushed durably. Best-effort — never affects the reply.
    fn record_consumption(&self, accepted: &CoSignedReceipt) {
        let now = now_unix_ms();
        let consumer_id = match self
            .net
            .peer_id_from_ed25519_pubkey(accepted.payload.consumer.as_bytes())
        {
            Ok(id) => id,
            Err(_) => return,
        };
        let snapshot = {
            let mut map = match self.credit.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            let acct = map.entry(consumer_id.clone()).or_insert_with(|| CreditAccount::new(now));
            acct.record_consumed(accepted.payload.tokens, now);
            acct.to_bytes()
        };
        if let Some(store) = &self.store {
            if let Err(e) = store.put_credit(&consumer_id, &snapshot) {
                eprintln!("openhydra-agent: credit persist failed: {e}");
            }
        }
    }

    /// The current serve-rate cap in `[credit::RATE_FLOOR, 1.0]` for a consumer (by its
    /// libp2p id), from its give/take balance (M2.3). `1.0` when we hold no record (the
    /// starter grant). The throttle that *consults* this lands with the provider
    /// concurrency model (a per-request delay in today's serial loop would head-of-line
    /// block other consumers — see the M2.3 increment-2 note).
    pub fn consumer_rate_cap(&self, consumer_libp2p_id: &str) -> f64 {
        let now = now_unix_ms();
        self.credit
            .lock()
            .ok()
            .and_then(|m| m.get(consumer_libp2p_id).map(|a| a.rate_cap(now)))
            .unwrap_or(1.0)
    }

    /// M2.3 **enforcement**: before serving, slow a leecher proportionally to its give/take
    /// deficit — *priority, not access* (throttle, never block). Only `SERVE_REQUEST`s are
    /// throttled; a contributor (`rate_cap >= 1.0` → multiplier `0`) and every non-serve
    /// method pass straight through.
    ///
    /// Runs on a worker thread (post-pool, so the delay never blocks the poll loop), and is
    /// **budget-capped**: it only sleeps while fewer than `max_concurrency - 1` workers are
    /// already throttling, so at least one worker always stays free to serve immediately.
    /// A flood of leechers therefore can't put the whole pool to sleep and stall an arriving
    /// contributor — the soft throttle simply stops biting once the budget is spent.
    fn maybe_throttle(&self, data: &[u8], max_concurrency: usize) {
        if data.first() != Some(&SERVE_REQUEST) {
            return; // receipts and unknown methods are never throttled
        }
        let reply_to = match ServeRequest::decode(&data[1..]) {
            Ok(req) => req.reply_to,
            Err(_) => return, // unparseable → let dispatch emit the error frame
        };
        let mult = throttle_multiplier(self.consumer_rate_cap(&reply_to));
        if mult <= 0.0 {
            return; // contributor — full speed
        }
        // Reserve ≥1 non-throttling worker: only sleep if we are not the one that would
        // exhaust the pool's throttle budget.
        let already = self.throttling.fetch_add(1, Ordering::SeqCst);
        if already + 1 < max_concurrency {
            std::thread::sleep(BASE_THROTTLE.mul_f64(mult));
        }
        self.throttling.fetch_sub(1, Ordering::SeqCst);
    }

    /// Detect the engine's models and announce a record for each. Returns how many were
    /// announced.
    pub fn announce_models(&self) -> Result<usize, AdapterError> {
        let models = self.adapter.detect_models()?;
        for model in &models {
            let record = build_peer_record(
                model,
                self.net.openhydra_peer_id(),
                self.net.libp2p_peer_id(),
                self.net.public_key_hex(),
                &self.host,
                self.port,
            );
            self.net
                .announce(record)
                .map_err(|e| AdapterError::Http(format!("announce: {e}")))?;
        }
        Ok(models.len())
    }

    /// Blocking serve loop: poll inbound requests, serve them, reply, and **periodically
    /// re-announce** our models. Runs until the process exits (call on a dedicated thread).
    ///
    /// Polls in `poll_timeout` slices so it stays responsive; a failed reply is
    /// logged-by-return and the loop continues. Every `reannounce_every` it re-publishes
    /// the DHT records: the bootstrap relays expire provider records on a short TTL
    /// (300s today), and the very first announce at startup can race an empty routing
    /// table — so a one-shot announce silently vanishes. Re-announcing on an interval
    /// shorter than the TTL keeps the node discoverable.
    /// `max_concurrency` bounds in-flight serves: the poll loop hands each request to a
    /// [`WorkerPool`] of that many threads instead of serving inline, so a long generation
    /// (or an M2.3 throttle delay) on one request no longer blocks the others. Re-announce
    /// still runs on the poll thread between polls.
    pub fn run_inbound(
        self: Arc<Self>,
        poll_timeout: std::time::Duration,
        reannounce_every: std::time::Duration,
        max_concurrency: usize,
    ) -> !
    where
        A: Send + Sync + 'static,
    {
        let pool = WorkerPool::new(max_concurrency);
        let mut last_announce = std::time::Instant::now();
        // #42: track the network generation. The event loop bumps it whenever it
        // rebuilds connectivity after a network change (roam / wake / interface
        // up-down); when it advances we re-announce immediately so the DHT record
        // carries the new relay addresses under the same pinned PeerId, instead
        // of waiting out the periodic `reannounce_every` interval.
        let mut last_generation = self.net.network_generation();
        loop {
            if let Some((request_id, data)) = self.net.poll_inbound(poll_timeout) {
                // Serve off the poll thread so the loop can keep accepting requests and a
                // slow one doesn't head-of-line-block the rest.
                let provider = Arc::clone(&self);
                let shed_id = request_id.clone();
                let accepted = pool.submit(move || {
                    // M2.3: throttle a leecher first (off the poll thread, budget-capped so
                    // it never stalls the pool); contributors pass straight through.
                    provider.maybe_throttle(&data, max_concurrency);
                    // AUP floor: refuse a policy-violating serve request without running it.
                    let response = match provider.aup_refusal(&data) {
                        Some(refusal) => {
                            if let Some(stats) = &provider.stats {
                                stats.aup_refusals.fetch_add(1, Ordering::Relaxed);
                            }
                            refusal
                        }
                        None => provider.dispatch(&data),
                    };
                    // Best-effort reply; if the swarm is gone the next poll will error too.
                    let _ = provider.net.respond(request_id, response);
                });
                if !accepted {
                    // A3: the bounded serve queue is full — shed with a retryable error so
                    // the consumer fails fast instead of waiting out its request timeout,
                    // and provider memory stays bounded under a burst.
                    // See docs/CODEBASE_HARDENING_PLAN.md (A3).
                    let busy = frame_response(&[
                        ServeChunk::Error("provider overloaded, retry".into()).encode(),
                    ]);
                    let _ = self.net.respond(shed_id, busy);
                }
            }
            // #42: a network change → re-announce now (not on the slow interval).
            let generation = self.net.network_generation();
            let network_changed = generation != last_generation;
            if network_changed {
                last_generation = generation;
                eprintln!(
                    "openhydra-agent: network change (generation {generation}) — re-announcing"
                );
            }
            if network_changed || last_announce.elapsed() >= reannounce_every {
                match self.announce_models() {
                    Ok(n) => eprintln!("openhydra-agent: re-announced {n} model(s)"),
                    Err(e) => eprintln!("openhydra-agent: re-announce failed: {e}"),
                }
                last_announce = std::time::Instant::now();
            }
        }
    }

    /// Route one inbound request by its method byte: serve completions, settle receipts.
    fn dispatch(&self, data: &[u8]) -> Vec<u8> {
        match data.first() {
            Some(&RECEIPT_REQUEST) => {
                let sign = |msg: &[u8]| self.net.sign(msg).unwrap_or_default();
                let provider_pub = self.net.public_key_bytes().unwrap_or_default();
                let (response, accepted) = handle_receipt_inbound(data, &sign, &provider_pub);
                // Persist the accepted co-signed receipt (best-effort; never fails the reply).
                ledger_receipt(self.store.as_ref(), accepted.as_ref());
                // M2.3: accrue the consumer's give/take (it consumed `tokens` from us).
                if let Some(receipt) = &accepted {
                    self.record_consumption(receipt);
                    if let Some(stats) = &self.stats {
                        stats.receipts_ledgered.fetch_add(1, Ordering::Relaxed);
                    }
                }
                response
            }
            // A serve request: run it, and — when the status endpoint is on — fold the
            // outcome into the shared transfer counters (per-model tokens + native TPS).
            Some(&SERVE_REQUEST) => {
                let model_ref = ServeRequest::decode(&data[1..]).ok().map(|r| r.model_ref);
                let mut chunks: Vec<Vec<u8>> = Vec::new();
                let summary =
                    handle_serve_request(&data[1..], &self.adapter, &mut |c| chunks.push(c.to_vec()));
                if let (Some(stats), Some(model)) = (&self.stats, model_ref) {
                    let e = &summary.metrics.engine;
                    let native_tps = if e.eval_duration_ns > 0 {
                        e.eval_count as f64 / (e.eval_duration_ns as f64 / 1e9)
                    } else {
                        0.0
                    };
                    stats.record_serve(&model, summary.tokens, native_tps, summary.ok);
                }
                frame_response(&chunks)
            }
            // Unknown method byte → a framed Error from the serve handler.
            _ => handle_serve_inbound(data, &self.adapter),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{InferenceRequest, ServeOutcome};
    use crate::serve::{parse_response, ServeChunk, ServeRequest};

    fn detected(canonical: &str) -> DetectedModel {
        DetectedModel {
            engine_ref: "qwen2.5:7b".into(),
            canonical_id: canonical.into(),
            family: "qwen2".into(),
            params: "7.6b".into(),
            quant: "q4_k_m".into(),
            size_bytes: 42,
        }
    }

    #[test]
    fn peer_record_keys_on_engine_handle_and_carries_canonical_id() {
        let r = build_peer_record(
            &detected("qwen2/7.6b/q4_k_m/abcd0123abcd0123"),
            "oh-peer",
            "12D3KooWlibp2p",
            "deadbeef",
            "10.0.0.5",
            4001,
        );
        assert_eq!(r.peer_id, "oh-peer");
        assert_eq!(r.model_id, "qwen2.5:7b"); // engine handle is the DHT key consumers request
        assert_eq!(r.canonical_model_id, "qwen2/7.6b/q4_k_m/abcd0123abcd0123"); // precise, for filtering
        assert_eq!(r.libp2p_peer_id, "12D3KooWlibp2p");
        assert_eq!(r.public_key, "deadbeef");
        assert_eq!(r.port, 4001);
    }

    #[test]
    fn peer_record_keys_on_engine_handle_even_without_canonical_id() {
        let r = build_peer_record(&detected(""), "oh", "lib", "pk", "", 0);
        assert_eq!(r.model_id, "qwen2.5:7b"); // always the engine handle
        assert_eq!(r.canonical_model_id, "");
    }

    struct StubAdapter;
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
            on_delta("Hello");
            on_delta(" world");
            Ok(ServeOutcome { tokens: 5, done: true, engine: Default::default() })
        }
    }

    fn serve_request_bytes() -> Vec<u8> {
        let req = ServeRequest {
            reply_to: "12D3KooWConsumer".into(),
            model_ref: "qwen2.5:7b".into(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
        };
        let mut data = vec![SERVE_REQUEST];
        data.extend_from_slice(&req.encode());
        data
    }

    #[test]
    fn serves_an_inbound_request_into_a_framed_completion() {
        let response = handle_serve_inbound(&serve_request_bytes(), &StubAdapter);
        let chunks = parse_response(&response).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], ServeChunk::Delta("Hello".into()));
        assert_eq!(chunks[1], ServeChunk::Delta(" world".into()));
        assert!(matches!(chunks[2], ServeChunk::Done { tokens: 5, .. }));
    }

    #[test]
    fn rejects_unknown_method_byte() {
        let response = handle_serve_inbound(&[0xEE, 1, 2, 3], &StubAdapter);
        let chunks = parse_response(&response).unwrap();
        assert!(matches!(chunks.as_slice(), [ServeChunk::Error(_)]));
    }

    fn test_receipt(nonce: [u8; 16], tokens: u64) -> CoSignedReceipt {
        use openhydra_protocol::crypto_agility::SigAlg;
        use openhydra_protocol::receipts::{build_receipt, ReceiptPayload};
        use ed25519_dalek::SigningKey;
        let consumer = SigningKey::from_bytes(&[3u8; 32]);
        let provider = SigningKey::from_bytes(&[5u8; 32]);
        let payload = ReceiptPayload {
            sig_alg: SigAlg::Ed25519,
            provider: provider.verifying_key(),
            consumer: consumer.verifying_key(),
            model_id: "qwen2.5/7b/q4_k_m/abcd0123abcd0123".to_string(),
            tokens,
            nonce,
            ts_unix_ms: 1_700_000_000_000,
        };
        build_receipt(payload, &consumer, &provider)
    }

    #[test]
    fn ledger_receipt_persists_when_store_present_and_is_replay_safe() {
        let store = Store::open_in_memory().unwrap();
        let receipt = test_receipt([11u8; 16], 128);

        ledger_receipt(Some(&store), Some(&receipt));
        assert_eq!(store.receipt_count().unwrap(), 1);

        // Replaying the same receipt must not add a second row.
        ledger_receipt(Some(&store), Some(&receipt));
        assert_eq!(store.receipt_count().unwrap(), 1);
    }

    #[test]
    fn ledger_receipt_is_a_noop_without_a_store_or_receipt() {
        let store = Store::open_in_memory().unwrap();
        // No receipt to ledger.
        ledger_receipt(Some(&store), None);
        assert_eq!(store.receipt_count().unwrap(), 0);
        // No store configured → nothing persisted, no panic.
        let receipt = test_receipt([22u8; 16], 64);
        ledger_receipt(None, Some(&receipt));
    }
}
