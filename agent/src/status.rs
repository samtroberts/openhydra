// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The agent's introspection endpoint (`--status-bind`, P0 of the network-suite plan).
//!
//! A tiny, GET-only, loopback-oriented HTTP server exposing read-only JSON about the
//! running node: the network snapshot (peers / DHT / reservations / counters — assembled
//! by the event loop's `Status` command) plus agent-side transfer counters (requests,
//! tokens, per-model TPS aggregates). This is what the desktop app's Peers/DHT/Swarm
//! views read; it is also `curl`-able for debugging.
//!
//! Deliberately **not** axum: the provider role is tokio-free by design, responses are
//! tiny one-shot JSON documents, and a `std::net::TcpListener` + thread-per-connection
//! loop keeps the dependency surface at zero. HTTP/1.1, `Connection: close`, GET only.
//!
//! Security: bind loopback (the default; the flag accepts other binds but warns), plus
//! an optional bearer token — if `OPENHYDRA_STATUS_TOKEN` is set at launch, every
//! request must carry `Authorization: Bearer <token>` (the desktop app sets a random
//! one per launch so other local users/processes can't read peer data).

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

/// How many recent ledger rows each role keeps in memory for the desktop's Ledger view.
/// A small bounded ring: the view shows "recent" activity, not the durable receipt archive
/// (which the provider persists to redb only when launched with `--db`).
const LEDGER_RING_CAP: usize = 250;

use openhydra_network::handle::StatusClient;

/// Per-model rolling serve aggregates.
#[derive(Debug, Default, Clone, Serialize)]
pub struct ModelStats {
    pub requests: u64,
    pub tokens: u64,
    /// Sum of per-request native TPS (divide by `requests` for the mean).
    #[serde(skip)]
    tps_sum: f64,
    /// Mean engine-native generation TPS across serves (0 when unknown).
    pub avg_native_tps: f64,
}

/// One recent transaction for the desktop's Ledger view (#5). A `served` row is minted
/// when the provider co-signs a settlement receipt; a `used` row when the consumer finishes
/// a completion. Bounded ring (newest-first) — not the durable archive.
#[derive(Debug, Clone, Serialize)]
pub struct LedgerRow {
    /// Unix ms when the row was recorded.
    pub ts_ms: u64,
    /// `"served"` (this node served a peer) or `"used"` (this node consumed from a peer).
    pub kind: &'static str,
    pub model: String,
    /// The counterparty's short peer id (the consumer we served / the provider we used).
    pub counterparty: String,
    pub tokens: u64,
}

/// Agent-side transfer counters, shared between the serving role (writer) and the
/// status server (reader). Cheap atomics for the totals; one small mutex for the
/// per-model map (touched once per request, never per token).
///
/// Both roles share one instance: the **provider** writes the served side
/// (`record_serve` + `served` ledger rows), the **consumer/gateway** the consumed side
/// (`record_consume` + `used` ledger rows). The desktop merges the two processes' views.
#[derive(Debug, Default)]
pub struct TransferStats {
    pub requests_served: AtomicU64,
    pub tokens_served: AtomicU64,
    pub serve_errors: AtomicU64,
    pub aup_refusals: AtomicU64,
    pub receipts_ledgered: AtomicU64,
    /// Consumer-side totals (mirror of the served ones), for the give-to-get "used" figures.
    pub requests_consumed: AtomicU64,
    pub tokens_consumed: AtomicU64,
    per_model: Mutex<HashMap<String, ModelStats>>,
    /// Per-model consumed aggregates (consumer/gateway side). TPS is not meaningful here.
    consumed_per_model: Mutex<HashMap<String, ModelStats>>,
    /// Newest-first ring of recent ledger rows (both `served` and `used`), for the Ledger view.
    recent: Mutex<VecDeque<LedgerRow>>,
    /// The provider's live share view (policy mode + intended list + last-announced set), written
    /// by the provider on each announce and read by the `/status/share` endpoint. Empty for the
    /// consumer/gateway role, which announces nothing.
    share: Mutex<crate::share_policy::ShareStatusView>,
}

impl TransferStats {
    /// Record one completed serve (called by the provider after the engine stream ends).
    pub fn record_serve(&self, model: &str, tokens: u64, native_tps: f64, ok: bool) {
        self.requests_served.fetch_add(1, Ordering::Relaxed);
        if !ok {
            self.serve_errors.fetch_add(1, Ordering::Relaxed);
        }
        self.tokens_served.fetch_add(tokens, Ordering::Relaxed);
        if let Ok(mut map) = self.per_model.lock() {
            let entry = map.entry(model.to_string()).or_default();
            entry.requests += 1;
            entry.tokens += tokens;
            if native_tps > 0.0 {
                entry.tps_sum += native_tps;
            }
            let counted = entry.requests.max(1) as f64;
            entry.avg_native_tps = (entry.tps_sum / counted * 10.0).round() / 10.0;
        }
    }

    /// Record one completed **consumption** (called by the consumer after a completion). The
    /// symmetric half of `record_serve`: net-new per-model consumed tracking so the desktop's
    /// give-to-get "used" figures and the served-vs-used timeline are correct — including
    /// tokens consumed by external OpenAI clients (e.g. a coding agent) pointed at the gateway,
    /// which the desktop's own chat counter never sees.
    pub fn record_consume(&self, model: &str, tokens: u64) {
        self.requests_consumed.fetch_add(1, Ordering::Relaxed);
        self.tokens_consumed.fetch_add(tokens, Ordering::Relaxed);
        if let Ok(mut map) = self.consumed_per_model.lock() {
            let entry = map.entry(model.to_string()).or_default();
            entry.requests += 1;
            entry.tokens += tokens;
        }
    }

    /// Push a recent ledger row (newest-first, bounded). `now_ms` is passed in so the caller
    /// owns the clock (keeps this pure/testable).
    pub fn record_ledger(&self, now_ms: u64, kind: &'static str, model: &str, counterparty: &str, tokens: u64) {
        if let Ok(mut ring) = self.recent.lock() {
            ring.push_front(LedgerRow {
                ts_ms: now_ms,
                kind,
                model: model.to_string(),
                counterparty: counterparty.to_string(),
                tokens,
            });
            while ring.len() > LEDGER_RING_CAP {
                ring.pop_back();
            }
        }
    }

    /// Publish the provider's current share view (called by the provider after each announce). The
    /// status server's `/status/share` endpoint reads it so the desktop can render each model's
    /// real state (announced / pending / off) rather than guessing from detection + settings.
    pub fn publish_share(&self, view: crate::share_policy::ShareStatusView) {
        if let Ok(mut g) = self.share.lock() {
            *g = view;
        }
    }

    /// The most recently published share view (default/empty until the provider announces, and for
    /// the consumer/gateway role which never announces).
    pub fn share_snapshot(&self) -> crate::share_policy::ShareStatusView {
        self.share.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Rehydrate the recent ring + lifetime totals from the durable ledger on boot, so the
    /// desktop Ledger view and its counters survive a restart (the in-memory ring is otherwise
    /// zeroed each launch). `rows` are newest-first (as [`Store::recent_ledger_rows`] returns);
    /// the totals are the all-time aggregates from [`Store::ledger_totals`]. Called once at
    /// startup, before serving begins.
    ///
    /// [`Store::recent_ledger_rows`]: openhydra_protocol::store::Store::recent_ledger_rows
    /// [`Store::ledger_totals`]: openhydra_protocol::store::Store::ledger_totals
    pub fn rehydrate_ledger(
        &self,
        rows: &[openhydra_protocol::store::LedgerEntry],
        served_tokens: u64,
        used_tokens: u64,
        served_count: u64,
    ) {
        if let Ok(mut ring) = self.recent.lock() {
            ring.clear();
            for e in rows.iter().take(LEDGER_RING_CAP) {
                ring.push_back(LedgerRow {
                    ts_ms: e.ts_ms,
                    kind: if e.kind == "served" { "served" } else { "used" },
                    model: e.model.clone(),
                    counterparty: e.counterparty.clone(),
                    tokens: e.tokens,
                });
            }
        }
        self.tokens_served.store(served_tokens, Ordering::Relaxed);
        self.tokens_consumed.store(used_tokens, Ordering::Relaxed);
        self.receipts_ledgered.store(served_count, Ordering::Relaxed);
    }

    fn snapshot(&self) -> TransfersView {
        TransfersView {
            requests_served: self.requests_served.load(Ordering::Relaxed),
            tokens_served: self.tokens_served.load(Ordering::Relaxed),
            serve_errors: self.serve_errors.load(Ordering::Relaxed),
            aup_refusals: self.aup_refusals.load(Ordering::Relaxed),
            receipts_ledgered: self.receipts_ledgered.load(Ordering::Relaxed),
            requests_consumed: self.requests_consumed.load(Ordering::Relaxed),
            tokens_consumed: self.tokens_consumed.load(Ordering::Relaxed),
            per_model: self.per_model.lock().map(|m| m.clone()).unwrap_or_default(),
            consumed_per_model: self.consumed_per_model.lock().map(|m| m.clone()).unwrap_or_default(),
            recent: self.recent.lock().map(|r| r.iter().cloned().collect()).unwrap_or_default(),
        }
    }
}

#[derive(Debug, Default, Serialize)]
struct TransfersView {
    requests_served: u64,
    tokens_served: u64,
    serve_errors: u64,
    aup_refusals: u64,
    receipts_ledgered: u64,
    requests_consumed: u64,
    tokens_consumed: u64,
    per_model: HashMap<String, ModelStats>,
    consumed_per_model: HashMap<String, ModelStats>,
    recent: Vec<LedgerRow>,
}

/// One counterparty's earned reputation (consumer-side, keyed by **OpenHydra** peer id —
/// the receipt/reputation key).
#[derive(Debug, Clone, Serialize)]
pub struct RepEntry {
    pub openhydra_peer_id: String,
    /// Earned reputation in `[0, 100]`, decayed to now (50 = neutral). See `verify.rs`.
    pub score: f64,
}

/// One counterparty's give/take credit standing (keyed by **libp2p** peer id — the dial
/// target). `rate_cap` is set only on the provider side (the serve-rate throttle we apply
/// to that consumer); on the consumer side it's `None`.
#[derive(Debug, Clone, Serialize)]
pub struct CreditEntry {
    pub libp2p_peer_id: String,
    /// Give/take balance decayed to now (`credit::STARTER_GRANT` baseline). See `credit.rs`.
    pub balance: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_cap: Option<f64>,
}

/// The node's local view of the give-to-get economy (M2.2 reputation + M2.3 credit). This is
/// **relational, not a wallet**: reputation is what *this* node has earned-assigned to the
/// providers it used; credit is the pairwise give/take standing per counterparty. Published
/// by the running role into [`EconomyStats`] and read by the status server.
#[derive(Debug, Default, Clone, Serialize)]
pub struct EconomyView {
    /// "consumer" (gateway earned-rep + give-side credit) or "provider" (take-side credit).
    pub role: String,
    pub reputation: Vec<RepEntry>,
    pub credit: Vec<CreditEntry>,
    /// Mean earned reputation across known counterparties, or `None` if none yet.
    pub avg_reputation: Option<f64>,
    /// Sum of credit balances (a rough standing number; not money).
    pub total_credit: f64,
}

impl EconomyView {
    /// Assemble a view from the raw per-peer lists, computing the summary aggregates.
    pub fn new(role: &str, reputation: Vec<RepEntry>, credit: Vec<CreditEntry>) -> Self {
        let avg_reputation = if reputation.is_empty() {
            None
        } else {
            Some((reputation.iter().map(|r| r.score).sum::<f64>() / reputation.len() as f64 * 10.0).round() / 10.0)
        };
        let total_credit = (credit.iter().map(|c| c.balance).sum::<f64>() * 10.0).round() / 10.0;
        Self { role: role.to_string(), reputation, credit, avg_reputation, total_credit }
    }
}

/// Shared, publish-once-poll-many economy handle: the running role publishes a fresh
/// [`EconomyView`] on a short interval; the status server reads the latest under `/status`.
/// A small mutex around a cheap clone — touched a few times a second at most.
#[derive(Debug, Default)]
pub struct EconomyStats {
    inner: Mutex<EconomyView>,
}

impl EconomyStats {
    pub fn publish(&self, view: EconomyView) {
        if let Ok(mut g) = self.inner.lock() {
            *g = view;
        }
    }
    fn snapshot(&self) -> EconomyView {
        self.inner.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

/// Everything the status server needs, bundled for the serving thread.
pub struct StatusServer {
    pub role: &'static str,
    pub agent_version: &'static str,
    pub libp2p_peer_id: String,
    pub openhydra_peer_id: String,
    pub net: StatusClient,
    pub stats: Arc<TransferStats>,
    /// Live give-to-get economy view, published by the running role (empty until the role
    /// has interacted with a counterparty).
    pub economy: Arc<EconomyStats>,
    /// Process start (unix ms) — the status server derives `uptime_secs` from it.
    pub started_at_ms: u64,
    /// Required bearer token, if any (`OPENHYDRA_STATUS_TOKEN`).
    pub token: Option<String>,
}

impl StatusServer {
    /// Bind `addr` and serve forever on a background thread. Returns the actual bound
    /// address (useful with port 0).
    pub fn spawn(self, addr: &str) -> Result<std::net::SocketAddr, String> {
        let listener = TcpListener::bind(addr).map_err(|e| format!("status bind {addr}: {e}"))?;
        let local = listener.local_addr().map_err(|e| e.to_string())?;
        if !local.ip().is_loopback() {
            eprintln!(
                "openhydra-agent: WARNING — status endpoint bound to non-loopback {local}; \
                 anyone who can reach it can read peer/DHT state"
            );
        }
        let server = Arc::new(self);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let server = Arc::clone(&server);
                // Tiny one-shot responses; a thread per connection is plenty here.
                std::thread::spawn(move || {
                    let _ = server.handle(stream);
                });
            }
        });
        Ok(local)
    }

    fn handle(&self, mut stream: TcpStream) -> std::io::Result<()> {
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut request_line = String::new();
        reader.read_line(&mut request_line)?;
        let mut parts = request_line.split_whitespace();
        let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or("/"));

        // Drain headers, capturing only Authorization.
        let mut authorized = self.token.is_none();
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line)? == 0 || line.trim().is_empty() {
                break;
            }
            if let Some(expected) = &self.token {
                // The header name and the "Bearer" scheme are case-insensitive, but the token
                // value is NOT — downcasing it (as before) silently weakened mixed-case secrets.
                // Compare the raw token in constant time so a match can't be recovered by timing
                // the response prefix-by-prefix.
                let line_t = line.trim_end();
                if let Some(rest) = strip_ci_prefix(line_t, "authorization:") {
                    if let Some(tok) = strip_ci_prefix(rest.trim_start(), "bearer ") {
                        if constant_time_eq(tok.trim().as_bytes(), expected.as_bytes()) {
                            authorized = true;
                        }
                    }
                }
            }
        }

        if method != "GET" {
            return respond(&mut stream, 405, r#"{"error":"GET only"}"#);
        }
        if !authorized {
            return respond(&mut stream, 401, r#"{"error":"missing/invalid bearer token"}"#);
        }

        // Route: /healthz is tokenless-cheap; everything else serializes a view.
        let path = path.split('?').next().unwrap_or(path);
        let body = match path {
            "/healthz" => r#"{"ok":true}"#.to_string(),
            "/status" | "/status/" => match self.net.status() {
                Ok(net) => json(&FullStatus {
                    role: self.role,
                    agent_version: self.agent_version,
                    libp2p_peer_id: &self.libp2p_peer_id,
                    openhydra_peer_id: &self.openhydra_peer_id,
                    uptime_secs: uptime_secs(self.started_at_ms),
                    network: net,
                    transfers: self.stats.snapshot(),
                    economy: self.economy.snapshot(),
                }),
                Err(e) => return respond(&mut stream, 500, &json(&ErrBody { error: e })),
            },
            "/status/economy" => json(&self.economy.snapshot()),
            "/status/peers" => match self.net.status() {
                Ok(net) => json(&net.peers),
                Err(e) => return respond(&mut stream, 500, &json(&ErrBody { error: e })),
            },
            "/status/dht" => match self.net.status() {
                Ok(mut net) => {
                    // The DHT view is the snapshot minus the (possibly large) peer/provider
                    // tables — trim so a dashboard polling /status/dht stays light.
                    net.peers.clear();
                    net.known_providers.clear();
                    json(&net)
                }
                Err(e) => return respond(&mut stream, 500, &json(&ErrBody { error: e })),
            },
            "/status/swarm" => match self.net.status() {
                Ok(net) => json(&SwarmView {
                    known_models: net.known_models,
                    known_providers: net.known_providers,
                }),
                Err(e) => return respond(&mut stream, 500, &json(&ErrBody { error: e })),
            },
            "/status/transfers" => json(&self.stats.snapshot()),
            // Provider share view: policy mode + intended list + the real last-announced set. Empty
            // for the gateway role. The desktop polls this to render each model's true state.
            "/status/share" => json(&self.stats.share_snapshot()),
            _ => return respond(&mut stream, 404, r#"{"error":"unknown path"}"#),
        };
        respond(&mut stream, 200, &body)
    }
}

#[derive(Serialize)]
struct FullStatus<'a> {
    role: &'a str,
    agent_version: &'a str,
    libp2p_peer_id: &'a str,
    openhydra_peer_id: &'a str,
    /// Seconds since this agent process started.
    uptime_secs: u64,
    network: openhydra_network::types::StatusSnapshot,
    transfers: TransfersView,
    economy: EconomyView,
}

/// Wall-clock seconds since `started_at_ms` (saturating; 0 if the clock went backwards).
fn uptime_secs(started_at_ms: u64) -> u64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(started_at_ms);
    now.saturating_sub(started_at_ms) / 1000
}

#[derive(Serialize)]
struct SwarmView {
    known_models: Vec<String>,
    known_providers: Vec<openhydra_network::types::KnownProvider>,
}

#[derive(Serialize)]
struct ErrBody {
    error: String,
}

fn json<T: Serialize>(v: &T) -> String {
    serde_json::to_string(v).unwrap_or_else(|e| format!(r#"{{"error":"serialize: {e}"}}"#))
}

/// Case-insensitively strip an ASCII `prefix` (an HTTP header name or auth scheme), returning
/// the remainder with its original case preserved. `None` if `s` doesn't start with `prefix`.
fn strip_ci_prefix<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let n = prefix.len();
    if s.len() >= n && s.as_bytes()[..n].eq_ignore_ascii_case(prefix.as_bytes()) {
        Some(&s[n..]) // `n` is a char boundary: the matched bytes are all ASCII
    } else {
        None
    }
}

/// Constant-time byte equality. Length is allowed to leak (it's not the secret); the token
/// value is compared without an early-exit so a match can't be timed out byte-by-byte.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod auth_tests {
    use super::{constant_time_eq, strip_ci_prefix};

    #[test]
    fn ci_prefix_preserves_token_case() {
        assert_eq!(strip_ci_prefix("Authorization: Bearer AbC123", "authorization:"), Some(" Bearer AbC123"));
        assert_eq!(strip_ci_prefix("AUTHORIZATION:x", "authorization:"), Some("x"));
        assert_eq!(strip_ci_prefix("x-other: y", "authorization:"), None);
    }

    #[test]
    fn token_compare_is_case_sensitive() {
        assert!(constant_time_eq(b"AbC123", b"AbC123"));
        assert!(!constant_time_eq(b"abc123", b"AbC123")); // case matters now
        assert!(!constant_time_eq(b"AbC12", b"AbC123")); // length differs
    }
}

fn respond(stream: &mut TcpStream, code: u16, body: &str) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len(),
    )?;
    stream.flush()
}

/// Read a full response from a status-server TCP stream (test helper shape, but usable
/// by any std-only caller).
#[cfg(test)]
fn read_response(stream: &mut TcpStream) -> String {
    use std::io::Read;
    let mut buf = String::new();
    let _ = stream.read_to_string(&mut buf);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_stats_aggregate_per_model() {
        let stats = TransferStats::default();
        stats.record_serve("llama3.2:1b", 10, 90.0, true);
        stats.record_serve("llama3.2:1b", 20, 110.0, true);
        stats.record_serve("tinyllama", 5, 0.0, false); // engine reported no TPS + an error
        let view = stats.snapshot();
        assert_eq!(view.requests_served, 3);
        assert_eq!(view.tokens_served, 35);
        assert_eq!(view.serve_errors, 1);
        let llama = &view.per_model["llama3.2:1b"];
        assert_eq!(llama.requests, 2);
        assert_eq!(llama.tokens, 30);
        assert_eq!(llama.avg_native_tps, 100.0);
        assert_eq!(view.per_model["tinyllama"].avg_native_tps, 0.0);
    }

    #[test]
    fn records_consumption_and_bounded_ledger_ring() {
        let s = TransferStats::default();
        s.record_consume("qwen2.5:7b", 100);
        s.record_consume("qwen2.5:7b", 50);
        s.record_ledger(1000, "used", "qwen2.5:7b", "12D3KooWpeer", 100);
        s.record_ledger(2000, "served", "tinyllama", "12D3KooWother", 40);
        let v = s.snapshot();
        assert_eq!(v.tokens_consumed, 150);
        assert_eq!(v.requests_consumed, 2);
        assert_eq!(v.consumed_per_model["qwen2.5:7b"].tokens, 150);
        assert_eq!(v.consumed_per_model["qwen2.5:7b"].requests, 2);
        // Newest-first ordering.
        assert_eq!(v.recent.len(), 2);
        assert_eq!(v.recent[0].kind, "served");
        assert_eq!(v.recent[1].kind, "used");
        // The ring is bounded at LEDGER_RING_CAP.
        for i in 0..(LEDGER_RING_CAP + 25) {
            s.record_ledger(i as u64, "used", "m", "p", 1);
        }
        assert_eq!(s.snapshot().recent.len(), LEDGER_RING_CAP);
    }

    #[test]
    fn rehydrate_ledger_restores_ring_newest_first_and_totals() {
        use openhydra_protocol::store::LedgerEntry;
        let s = TransferStats::default();
        // Store returns rows NEWEST-FIRST; rehydrate must preserve that in the ring and set the
        // lifetime totals — the whole point of surviving a restart.
        let rows = vec![
            LedgerEntry { ts_ms: 30, kind: "served".into(), model: "m2".into(), counterparty: "peerC".into(), tokens: 7 },
            LedgerEntry { ts_ms: 20, kind: "served".into(), model: "m2".into(), counterparty: "peerB".into(), tokens: 30 },
            LedgerEntry { ts_ms: 10, kind: "used".into(), model: "m1".into(), counterparty: "peerA".into(), tokens: 5 },
        ];
        s.rehydrate_ledger(&rows, 37, 5, 2);
        let v = s.snapshot();
        assert_eq!(v.recent.len(), 3);
        assert_eq!(v.recent[0].counterparty, "peerC"); // newest first
        assert_eq!(v.recent[2].counterparty, "peerA"); // oldest last
        assert_eq!(v.tokens_served, 37);
        assert_eq!(v.tokens_consumed, 5);
        assert_eq!(v.receipts_ledgered, 2);
        // A subsequent live row still lands at the front (newest), atop the rehydrated history.
        s.record_ledger(40, "served", "m3", "peerD", 3);
        assert_eq!(s.snapshot().recent[0].counterparty, "peerD");
    }

    #[test]
    fn share_view_publishes_and_snapshots() {
        use crate::share_policy::{ShareMode, ShareStatusView};
        let s = TransferStats::default();
        // Default (never published) reads honestly as "nothing shared".
        let empty = s.share_snapshot();
        assert_eq!(empty.share_mode, ShareMode::List);
        assert!(empty.shared_models.is_empty() && empty.announced_models.is_empty());
        // After the provider announces, the real advertised set is visible.
        s.publish_share(ShareStatusView {
            share_mode: ShareMode::List,
            shared_models: vec!["qwen3-coder:30b-a3b-q8_0".into(), "qwen3.8:27b-q8_0".into()],
            announced_models: vec!["qwen3-coder:30b-a3b-q8_0".into()],
        });
        let v = s.share_snapshot();
        assert_eq!(v.shared_models.len(), 2);
        assert_eq!(v.announced_models, vec!["qwen3-coder:30b-a3b-q8_0"]);
    }

    /// Full loop: live loopback swarm node → StatusClient → HTTP server → parsed JSON,
    /// including the bearer-token gate.
    #[test]
    fn serves_status_over_http_with_token_gate() {
        let dir = tempfile::tempdir().unwrap();
        let config = openhydra_network::node::NodeConfig {
            identity_path: dir.path().join("id.key"),
            listen_addrs: vec![
                "/ip4/127.0.0.1/tcp/0".into(),
                "/ip4/127.0.0.1/udp/0/quic-v1".into(),
            ],
            bootstrap_peers: vec![],
            enable_peer_relay: false,
            enable_connection_reversal: false,
            pcp_gateway: None,
        };
        let net = openhydra_network::handle::NetworkHandle::start(config).unwrap();
        let stats = Arc::new(TransferStats::default());
        stats.record_serve("m1", 7, 50.0, true);
        stats.publish_share(crate::share_policy::ShareStatusView {
            share_mode: crate::share_policy::ShareMode::List,
            shared_models: vec!["m1".into(), "m2".into()],
            announced_models: vec!["m1".into()],
        });
        let addr = StatusServer {
            role: "provider",
            agent_version: "test",
            libp2p_peer_id: net.libp2p_peer_id().to_string(),
            openhydra_peer_id: net.openhydra_peer_id().to_string(),
            net: net.status_client(),
            stats: Arc::clone(&stats),
            economy: Arc::new(EconomyStats::default()),
            started_at_ms: 0,
            token: Some("s3cret".into()),
        }
        .spawn("127.0.0.1:0")
        .unwrap();

        let get = |path: &str, auth: Option<&str>| -> String {
            let mut s = TcpStream::connect(addr).unwrap();
            let auth_line = auth.map(|t| format!("Authorization: Bearer {t}\r\n")).unwrap_or_default();
            write!(s, "GET {path} HTTP/1.1\r\nHost: x\r\n{auth_line}\r\n").unwrap();
            read_response(&mut s)
        };

        // No token → 401; wrong path → 404; happy path → parsed JSON with our data.
        assert!(get("/status", None).starts_with("HTTP/1.1 401"));
        assert!(get("/nope", Some("s3cret")).starts_with("HTTP/1.1 404"));
        let ok = get("/status", Some("s3cret"));
        assert!(ok.starts_with("HTTP/1.1 200"), "got: {}", &ok[..40.min(ok.len())]);
        let body = ok.split("\r\n\r\n").nth(1).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(parsed["role"], "provider");
        assert_eq!(parsed["transfers"]["tokens_served"], 7);
        assert!(parsed["network"]["listen_addrs"].as_array().unwrap().len() > 0);
        // Sub-view: /status/transfers returns just the counters.
        let t = get("/status/transfers", Some("s3cret"));
        let tbody: serde_json::Value =
            serde_json::from_str(t.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(tbody["per_model"]["m1"]["tokens"], 7);
        // Sub-view: /status/share returns the provider's real share view (mode + intended list +
        // the actually-announced set) — the source of truth the desktop renders from.
        let sh = get("/status/share", Some("s3cret"));
        let sbody: serde_json::Value =
            serde_json::from_str(sh.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(sbody["share_mode"], "list");
        assert_eq!(sbody["shared_models"], serde_json::json!(["m1", "m2"]));
        assert_eq!(sbody["announced_models"], serde_json::json!(["m1"]));
    }
}
