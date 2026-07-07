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

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::Serialize;

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

/// Agent-side transfer counters, shared between the serving role (writer) and the
/// status server (reader). Cheap atomics for the totals; one small mutex for the
/// per-model map (touched once per request, never per token).
#[derive(Debug, Default)]
pub struct TransferStats {
    pub requests_served: AtomicU64,
    pub tokens_served: AtomicU64,
    pub serve_errors: AtomicU64,
    pub aup_refusals: AtomicU64,
    pub receipts_ledgered: AtomicU64,
    per_model: Mutex<HashMap<String, ModelStats>>,
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

    fn snapshot(&self) -> TransfersView {
        TransfersView {
            requests_served: self.requests_served.load(Ordering::Relaxed),
            tokens_served: self.tokens_served.load(Ordering::Relaxed),
            serve_errors: self.serve_errors.load(Ordering::Relaxed),
            aup_refusals: self.aup_refusals.load(Ordering::Relaxed),
            receipts_ledgered: self.receipts_ledgered.load(Ordering::Relaxed),
            per_model: self.per_model.lock().map(|m| m.clone()).unwrap_or_default(),
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
    per_model: HashMap<String, ModelStats>,
}

/// Everything the status server needs, bundled for the serving thread.
pub struct StatusServer {
    pub role: &'static str,
    pub agent_version: &'static str,
    pub libp2p_peer_id: String,
    pub openhydra_peer_id: String,
    pub net: StatusClient,
    pub stats: Arc<TransferStats>,
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
                let lower = line.to_ascii_lowercase();
                if let Some(rest) = lower.strip_prefix("authorization:") {
                    if rest.trim() == format!("bearer {}", expected.to_ascii_lowercase()) {
                        authorized = true;
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
                    network: net,
                    transfers: self.stats.snapshot(),
                }),
                Err(e) => return respond(&mut stream, 500, &json(&ErrBody { error: e })),
            },
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
    network: openhydra_network::types::StatusSnapshot,
    transfers: TransfersView,
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
        let addr = StatusServer {
            role: "provider",
            agent_version: "test",
            libp2p_peer_id: net.libp2p_peer_id().to_string(),
            openhydra_peer_id: net.openhydra_peer_id().to_string(),
            net: net.status_client(),
            stats: Arc::clone(&stats),
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
    }
}
