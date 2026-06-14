//! P2PNode — the main PyO3 class exposed to Python.
//!
//! Architecture:
//! ```text
//! Python thread ──[mpsc::Sender<SwarmCommand>]──▶ tokio background thread
//!                                                       │
//!               ◀──[oneshot::Sender<Result>]───── swarm event loop
//! ```
//!
//! All network I/O runs on a dedicated tokio runtime in a background thread.
//! Python methods send commands via an mpsc channel and block on a oneshot
//! receiver for the result, with the GIL released during the wait.

use std::path::PathBuf;
use std::sync::Arc;

use libp2p::{Multiaddr, PeerId};
use tokio::sync::{mpsc, oneshot};
use tracing::info;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "pyo3")]
use pyo3::IntoPyObjectExt;

use crate::event_loop::{self, SharedProxyQueue, SwarmCommand};
use crate::forward_msg;
use crate::identity::Identity;
use crate::ipc_codec::{IpcForwardHeader, IpcResponseHeader, ActivationDtype, IpcStatus};
use crate::swarm::{self, SwarmOptions};
#[cfg(feature = "pyo3")]
use crate::types::PeerRecord;

/// Internal state shared between the Python-facing struct and the background thread.
#[cfg(feature = "pyo3")]
struct NodeInner {
    cmd_tx: mpsc::Sender<SwarmCommand>,
    proxy_queue: Arc<SharedProxyQueue>,
    /// Handle to the background thread running tokio + swarm.
    _thread: std::thread::JoinHandle<()>,
}

/// Configuration parsed from Python __init__ kwargs.
pub struct NodeConfig {
    pub identity_path: PathBuf,
    pub listen_addrs: Vec<String>,
    pub bootstrap_peers: Vec<String>,
    /// WS-F F-4: opt into being a temporary peer-relay (off by default).
    pub enable_peer_relay: bool,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            identity_path: dirs_default_identity(),
            listen_addrs: vec![
                "/ip4/0.0.0.0/tcp/4001".into(),
                "/ip6/::/tcp/4001".into(),
                // QUIC (UDP) — critical for DCUtR hole punching.
                // UDP hole punching has ~70-80% success rate vs TCP's
                // ~5-10% against symmetric NAT. Without these, DCUtR
                // can only attempt TCP simultaneous-open which fails
                // against most residential and cloud NATs.
                "/ip4/0.0.0.0/udp/4001/quic-v1".into(),
                "/ip6/::/udp/4001/quic-v1".into(),
            ],
            bootstrap_peers: Vec::new(),
            enable_peer_relay: false,
        }
    }
}

fn dirs_default_identity() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".openhydra").join("identity.key")
}

/// Start the P2P node: load identity, build swarm, spawn event loop.
///
/// Returns a command sender and thread handle. The swarm runs until
/// a Shutdown command is received or the sender is dropped.
pub fn start_node(
    config: &NodeConfig,
) -> Result<(mpsc::Sender<SwarmCommand>, Arc<SharedProxyQueue>, std::thread::JoinHandle<()>), String> {
    // Load or generate identity.
    let identity = Identity::load_or_create(&config.identity_path)
        .map_err(|e| format!("identity: {e}"))?;

    info!(
        libp2p_peer_id = %identity.libp2p_peer_id,
        openhydra_peer_id = %identity.openhydra_peer_id,
        "P2P node starting"
    );

    // Parse listen addresses.
    let listen_addrs: Vec<Multiaddr> = config
        .listen_addrs
        .iter()
        .map(|s| s.parse().map_err(|e| format!("bad listen addr '{s}': {e}")))
        .collect::<Result<Vec<_>, _>>()?;

    // Parse bootstrap peers: "/ip4/.../tcp/.../p2p/12D3KooW..."
    let bootstrap_peers = parse_bootstrap_peers(&config.bootstrap_peers)?;
    let enable_peer_relay = config.enable_peer_relay; // WS-F F-4 (captured into the thread)

    // Create the command channel.
    let (cmd_tx, cmd_rx) = mpsc::channel::<SwarmCommand>(256);

    // Shared proxy queue: event loop pushes, Python poll_proxy_request pops.
    let proxy_queue = Arc::new(SharedProxyQueue::new());
    let proxy_queue_clone = Arc::clone(&proxy_queue);

    // Use a oneshot to communicate any startup error from the background thread.
    let (startup_tx, startup_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    // Spawn a background thread with its own tokio runtime.
    // The swarm MUST be built inside the tokio context because listen_on()
    // needs an active reactor for TCP binding.
    let thread = std::thread::Builder::new()
        .name("openhydra-p2p".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");
            rt.block_on(async move {
                let bootstrap_peers_for_dial = bootstrap_peers.clone();
                let opts = SwarmOptions {
                    listen_addrs,
                    bootstrap_peers,
                    protocol_version: "openhydra/0.1.0".into(),
                    enable_peer_relay,
                };

                let keypair_for_loop = identity.keypair.clone();
                match swarm::build_swarm(&identity, opts) {
                    Ok((swarm, stream_control, peer_relay_leech)) => {
                        let _ = startup_tx.send(Ok(()));
                        event_loop::run_event_loop(swarm, cmd_rx, proxy_queue_clone, bootstrap_peers_for_dial, stream_control, keypair_for_loop, peer_relay_leech).await;
                    }
                    Err(e) => {
                        let _ = startup_tx.send(Err(format!("build_swarm: {e}")));
                    }
                }
            });
            info!("P2P event loop exited");
        })
        .map_err(|e| format!("thread spawn: {e}"))?;

    // Wait for the background thread to report startup success/failure.
    startup_rx
        .recv()
        .map_err(|_| "background thread died during startup".to_string())?
        .map_err(|e| format!("startup failed: {e}"))?;

    Ok((cmd_tx, proxy_queue, thread))
}

/// Parse bootstrap peer multiaddrs, extracting PeerId from the /p2p/ component.
fn parse_bootstrap_peers(addrs: &[String]) -> Result<Vec<(PeerId, Multiaddr)>, String> {
    let mut result = Vec::new();
    for s in addrs {
        let addr: Multiaddr = s.parse().map_err(|e| format!("bad bootstrap addr '{s}': {e}"))?;
        // Extract PeerId from the last /p2p/... component.
        let peer_id = addr
            .iter()
            .find_map(|p| match p {
                libp2p::multiaddr::Protocol::P2p(id) => Some(id),
                _ => None,
            })
            .ok_or_else(|| format!("bootstrap addr missing /p2p/ component: {s}"))?;
        // Strip /p2p/ from the address for Kademlia (it wants addr without peer id).
        let base_addr: Multiaddr = addr
            .iter()
            .filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
            .collect();
        result.push((peer_id, base_addr));
    }
    Ok(result)
}

// ── Blocking helpers (used by PyO3 methods with GIL released) ──

/// Send a command and wait for the reply, blocking the current thread.
#[cfg(feature = "pyo3")]
fn send_and_wait<T>(
    cmd_tx: &mpsc::Sender<SwarmCommand>,
    make_cmd: impl FnOnce(oneshot::Sender<T>) -> SwarmCommand,
) -> Result<T, String> {
    let (reply_tx, reply_rx) = oneshot::channel();
    cmd_tx
        .blocking_send(make_cmd(reply_tx))
        .map_err(|_| "swarm event loop not running".to_string())?;
    reply_rx
        .blocking_recv()
        .map_err(|_| "swarm dropped reply channel".to_string())
}

// ── M2.2: reputation store helpers ──

type RepStore = std::sync::RwLock<std::collections::HashMap<String, crate::verify::ReputationTracker>>;

/// Current Unix time in milliseconds — the live clock fed to the (otherwise
/// clock-injected) reputation math. Falls back to 0 if the system clock predates the
/// epoch (decay then simply treats no time as elapsed).
#[cfg(feature = "pyo3")]
fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The reputation the router should use for a peer: a locally-observed, time-decayed
/// score overrides the static DHT value when we've interacted with the peer; otherwise
/// fall back to the DHT's `reputation_score` (or the neutral baseline if it advertised
/// none). Shared by `resolve_and_route` (production) and `rank_discovered` (test seam)
/// so both apply the identical override.
#[cfg(feature = "pyo3")]
fn effective_reputation(store: &RepStore, peer_id: &str, dht_reputation: f64, now_ms: u64) -> f64 {
    if let Ok(guard) = store.read() {
        if let Some(tracker) = guard.get(peer_id) {
            return tracker.score_at(now_ms); // local, time-decayed score wins
        }
    }
    if dht_reputation > 0.0 {
        dht_reputation
    } else {
        crate::verify::NEUTRAL_REPUTATION
    }
}

/// Parse the FFI outcome string into a `VerificationOutcome`.
#[cfg(feature = "pyo3")]
fn parse_outcome(s: &str) -> PyResult<crate::verify::VerificationOutcome> {
    use crate::verify::VerificationOutcome::{Failed, Honored, Rejected};
    match s.to_ascii_lowercase().as_str() {
        "honored" => Ok(Honored),
        "rejected" => Ok(Rejected),
        "failed" => Ok(Failed),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown reputation outcome '{other}' (expected honored|rejected|failed)"
        ))),
    }
}

/// Read an optional f64 from a peer dict (missing/None → `default`).
#[cfg(feature = "pyo3")]
fn dict_get_f64(d: &Bound<'_, pyo3::types::PyDict>, key: &str, default: f64) -> PyResult<f64> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(v.extract::<f64>()?),
        _ => Ok(default),
    }
}

/// Read an optional u32 from a peer dict (missing/None → `default`).
#[cfg(feature = "pyo3")]
fn dict_get_u32(d: &Bound<'_, pyo3::types::PyDict>, key: &str, default: u32) -> PyResult<u32> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => Ok(v.extract::<u32>()?),
        _ => Ok(default),
    }
}

// ── PyDict ↔ serde_json conversion helpers (OHV2 wire format) ──
//
// These convert between PyDict and serde_json::Value for the
// encode_forward_msg / decode_forward_msg static methods.
// Using serde_json as the intermediate format reuses all the
// skip_serializing_if / default annotations on IpcForwardHeader.

#[cfg(feature = "pyo3")]
fn pydict_to_json_value(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<serde_json::Value> {
    pyany_to_json_value(dict.as_any())
}

#[cfg(feature = "pyo3")]
fn pyany_to_json_value(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::json!(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(bytes_obj) = obj.extract::<Vec<u8>>() {
        // Encode bytes as JSON array of u8 values (matches Vec<u8> serde).
        let arr: Vec<serde_json::Value> = bytes_obj.iter().map(|b| serde_json::json!(*b)).collect();
        Ok(serde_json::Value::Array(arr))
    } else if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
        let arr: Vec<serde_json::Value> = list
            .iter()
            .map(|item| pyany_to_json_value(&item))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(serde_json::Value::Array(arr))
    } else if let Ok(dict) = obj.downcast::<pyo3::types::PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, pyany_to_json_value(&v)?);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        // Fallback: try string extraction.
        let s: String = obj.str()?.extract()?;
        Ok(serde_json::Value::String(s))
    }
}

#[cfg(feature = "pyo3")]
fn json_value_to_pyobject(py: Python<'_>, val: &serde_json::Value) -> PyResult<PyObject> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_py_any(py)?),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_py_any(py)?)
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_py_any(py)?)
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_py_any(py)?)
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_py_any(py)?),
        serde_json::Value::Array(arr) => {
            let list = pyo3::types::PyList::empty(py);
            for item in arr {
                list.append(json_value_to_pyobject(py, item)?)?;
            }
            Ok(list.into_py_any(py)?)
        }
        serde_json::Value::Object(map) => {
            let dict = pyo3::types::PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_value_to_pyobject(py, v)?)?;
            }
            Ok(dict.into_py_any(py)?)
        }
    }
}

// ── PyO3 class ──

#[cfg(feature = "pyo3")]
#[pyclass(name = "P2PNode")]
pub struct PyP2PNode {
    inner: Option<NodeInner>,
    config: NodeConfig,
    /// Cached identity info (set after start).
    libp2p_peer_id: String,
    openhydra_peer_id: String,
    /// Ed25519 keypair for signing (retained for 6.0 identity methods).
    keypair: libp2p::identity::Keypair,
    /// M2.2 — live per-peer reputation, keyed by `peer_id`. Updated via
    /// `record_reputation_outcome` (the FFI feedback loop) and consulted by the router
    /// to override the static DHT `reputation_score` with a locally-observed,
    /// time-decayed score. `Arc` so it can be shared into the `resolve_and_route`
    /// discover closure that runs with the GIL released.
    reputation: Arc<std::sync::RwLock<std::collections::HashMap<String, crate::verify::ReputationTracker>>>,
    /// M2.3 — the persistent ledger (redb). File-backed when a `db_path` was given to the
    /// constructor, else an ephemeral in-memory store. Honored receipts are flushed here
    /// atomically; on boot the constructor rehydrates `reputation` + `nonces` from it.
    store: Arc<crate::store::Store>,
    /// M2.3 — in-memory replay guard, rehydrated from the ledger's NONCES table on boot.
    nonces: Arc<std::sync::RwLock<crate::receipts::NonceTracker>>,
}

// Phase 4.3: Drop implementation for crash safety.
// If Python doesn't call stop() (SIGKILL, crash, GC), attempt a graceful
// shutdown so PEER_DEPARTED gossip is published and DHT records are cleaned.
#[cfg(feature = "pyo3")]
impl Drop for PyP2PNode {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let (tx, rx) = oneshot::channel();
            if inner.cmd_tx.blocking_send(SwarmCommand::Shutdown { reply: tx }).is_ok() {
                // F14: bound the wait so interpreter shutdown can't hang if the
                // event loop is wedged. Hand the oneshot to a helper thread and
                // wait at most 500ms; if it doesn't complete, Drop proceeds
                // (the process is exiting anyway).
                let (done_tx, done_rx) = std::sync::mpsc::channel();
                std::thread::spawn(move || {
                    let _ = rx.blocking_recv();
                    let _ = done_tx.send(());
                });
                let _ = done_rx.recv_timeout(std::time::Duration::from_millis(500));
            }
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyP2PNode {
    /// Create a new P2P node.
    ///
    /// Args:
    ///     identity_key_path: Path to Ed25519 key file (default: ~/.openhydra/identity.key)
    ///     listen_addrs: Multiaddrs to listen on (default: ["/ip4/0.0.0.0/tcp/4001"])
    ///     bootstrap_peers: Bootstrap peer multiaddrs with /p2p/ suffix
    ///     enable_peer_relay: WS-F F-4 — opt into being a temporary peer-relay
    ///         for NATted peers (default False; only for publicly-reachable nodes)
    ///     db_path: M2.3 — path to the persistent ledger (redb). When None, an ephemeral
    ///         in-memory ledger is used (nothing survives a restart).
    #[new]
    #[pyo3(signature = (identity_key_path=None, listen_addrs=None, bootstrap_peers=None, enable_peer_relay=false, db_path=None))]
    fn new(
        identity_key_path: Option<String>,
        listen_addrs: Option<Vec<String>>,
        bootstrap_peers: Option<Vec<String>>,
        enable_peer_relay: bool,
        db_path: Option<String>,
    ) -> PyResult<Self> {
        let config = NodeConfig {
            identity_path: identity_key_path
                .map(PathBuf::from)
                .unwrap_or_else(dirs_default_identity),
            listen_addrs: listen_addrs
                .unwrap_or_else(|| vec![
                    "/ip4/0.0.0.0/tcp/4001".into(),
                    "/ip6/::/tcp/4001".into(),
                    "/ip4/0.0.0.0/udp/4001/quic-v1".into(),
                    "/ip6/::/udp/4001/quic-v1".into(),
                ]),
            bootstrap_peers: bootstrap_peers.unwrap_or_default(),
            enable_peer_relay,
        };

        // Pre-load identity to get peer IDs for properties.
        let identity = Identity::load_or_create(&config.identity_path)
            .map_err(|e| PyRuntimeError::new_err(format!("identity: {e}")))?;

        // M2.3: open the persistent ledger (file-backed if db_path given, else ephemeral
        // in-memory), then rehydrate reputation + the replay guard from it so a restart
        // resumes with the trust scores and spent nonces it shut down with.
        let store = match db_path {
            Some(ref p) => crate::store::Store::open(p),
            None => crate::store::Store::open_in_memory(),
        }
        .map_err(|e| PyRuntimeError::new_err(format!("ledger open: {e}")))?;

        let mut reputation_map = std::collections::HashMap::new();
        let mut nonce_tracker = crate::receipts::NonceTracker::new();
        store
            .load_state_into_memory(&mut nonce_tracker, &mut reputation_map)
            .map_err(|e| PyRuntimeError::new_err(format!("ledger rehydrate: {e}")))?;

        Ok(Self {
            inner: None,
            config,
            libp2p_peer_id: identity.libp2p_peer_id.to_string(),
            openhydra_peer_id: identity.openhydra_peer_id.clone(),
            keypair: identity.keypair,
            reputation: Arc::new(std::sync::RwLock::new(reputation_map)),
            store: Arc::new(store),
            nonces: Arc::new(std::sync::RwLock::new(nonce_tracker)),
        })
    }

    /// Start the P2P node (spawns background tokio thread).
    fn start(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.inner.is_some() {
            return Err(PyRuntimeError::new_err("node already started"));
        }
        // Release GIL while starting (involves I/O for identity + socket binding).
        let config = NodeConfig {
            identity_path: self.config.identity_path.clone(),
            listen_addrs: self.config.listen_addrs.clone(),
            bootstrap_peers: self.config.bootstrap_peers.clone(),
            enable_peer_relay: self.config.enable_peer_relay,
        };
        let result = py.allow_threads(|| start_node(&config));
        let (cmd_tx, proxy_queue, thread) = result
            .map_err(|e| PyRuntimeError::new_err(format!("start failed: {e}")))?;
        self.inner = Some(NodeInner {
            cmd_tx,
            proxy_queue,
            _thread: thread,
        });
        Ok(())
    }

    /// Stop the P2P node.
    fn stop(&mut self, py: Python<'_>) -> PyResult<()> {
        if let Some(inner) = self.inner.take() {
            py.allow_threads(|| {
                let _ = send_and_wait(&inner.cmd_tx, |reply| SwarmCommand::Shutdown { reply });
            });
        }
        Ok(())
    }

    /// Announce a peer record to the Kademlia DHT.
    ///
    /// Args:
    ///     record: dict with peer record fields (peer_id, model_id, host, port, ...)
    fn announce(&self, py: Python<'_>, record: &Bound<'_, PyAny>) -> PyResult<()> {
        let inner = self.require_started()?;
        // Convert Python dict → JSON → PeerRecord.
        let json_str: String = py
            .import("json")?
            .call_method1("dumps", (record,))?
            .extract()?;
        let peer_record: PeerRecord = serde_json::from_str(&json_str)
            .map_err(|e| PyRuntimeError::new_err(format!("bad record: {e}")))?;

        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            send_and_wait(&cmd_tx, |reply| SwarmCommand::Announce {
                record: peer_record,
                reply,
            })
        })
        .map_err(|e| PyRuntimeError::new_err(e))?
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Discover peers serving a model.
    ///
    /// Returns:
    ///     list[dict]: discovered peers with reachable_address field
    fn discover(&self, py: Python<'_>, model_id: String) -> PyResult<PyObject> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        let peers = py
            .allow_threads(move || {
                send_and_wait(&cmd_tx, |reply| SwarmCommand::Discover { model_id, reply })
            })
            .map_err(|e| PyRuntimeError::new_err(e))?
            .map_err(|e| PyRuntimeError::new_err(e))?;

        // Convert Vec<DiscoveredPeer> → list[dict] via JSON.
        let json_str = serde_json::to_string(&peers)
            .map_err(|e| PyRuntimeError::new_err(format!("json: {e}")))?;
        let json_mod = py.import("json")?;
        let result = json_mod.call_method1("loads", (json_str,))?;
        Ok(result.into_py_any(py)?)
    }

    /// Resolve → (graceful) degrade → rank → route a request, end to end (M1.3).
    ///
    /// `model_candidates` is an ordered list of `(model_id, request_canonical_id)`:
    /// the exact request first, then nearest-smaller same-family fallbacks (the caller
    /// computes that order from the catalog). For each model this discovers providers
    /// over the DHT, drops canonical-id-incompatible ones, ranks the rest
    /// (`protocol::router`), and proxies the request to the best reachable peer,
    /// failing over to the next-ranked peer / next candidate model as needed.
    ///
    /// Returns a dict ``{model_id, peer_id, response: bytes, degraded: bool}`` on
    /// success, or raises ``RuntimeError("no_provider: …")`` when nothing routes.
    fn resolve_and_route(
        &self,
        py: Python<'_>,
        model_candidates: Vec<(String, String)>,
        request: Vec<u8>,
        tier: u8,
    ) -> PyResult<PyObject> {
        let inner = self.require_started()?;
        let cmd_tx_d = inner.cmd_tx.clone();
        let cmd_tx_r = inner.cmd_tx.clone();
        // M2.2: share the live reputation store into the (GIL-released) discover closure,
        // and stamp one `now` for the whole call so decay is consistent across ranking.
        let rep_store = Arc::clone(&self.reputation);
        let now_ms = now_unix_ms();

        let outcome = py.allow_threads(move || {
            // discover: DHT lookup → router Candidates (RTT via ping is deferred).
            let discover = |model_id: &str| -> Vec<crate::router::Candidate> {
                let peers = match send_and_wait(&cmd_tx_d, |reply| SwarmCommand::Discover {
                    model_id: model_id.to_string(),
                    reply,
                }) {
                    Ok(Ok(peers)) => peers,
                    _ => Vec::new(), // discover error → treat as no providers (degrade)
                };
                peers
                    .iter()
                    .map(|dp| crate::router::Candidate {
                        canonical_model_id: dp.canonical_model_id.clone(),
                        // M2.1: carry the provider's ed25519 key (hex) so a winning route
                        // can surface it for the consumer's co-signed-receipt handshake.
                        public_key: dp.public_key.clone(),
                        score: crate::router::PeerScoreInput {
                            peer_id: dp.peer_id.clone(),
                            // TODO(M1.3): measure RTT via a ping survey — deferred (a
                            // concurrent ping fan-out is its own perf pass). Neutral 1.0
                            // here so latency does not bias ranking yet.
                            latency_ms: 1.0,
                            load_pct: dp.load_pct,
                            // M2.2: a locally-observed, time-decayed score overrides the
                            // static DHT reputation; unknown peers fall back to neutral.
                            reputation: effective_reputation(
                                &rep_store,
                                &dp.peer_id,
                                dp.reputation_score,
                                now_ms,
                            ),
                            bandwidth_mbps: 0.0,
                            s2s_rtt_ms: 0.0,
                            throughput_tok_s: dp.throughput_tok_s,
                            queue_depth: dp.queue_depth,
                        },
                    })
                    .collect()
            };
            // route: proxy the request to a chosen peer over libp2p.
            let route = |peer_id: &str, data: &[u8]| -> Result<Vec<u8>, String> {
                match send_and_wait(&cmd_tx_r, |reply| SwarmCommand::ProxyForward {
                    peer_id: peer_id.to_string(),
                    data: data.to_vec(),
                    reply,
                }) {
                    Ok(Ok(resp)) => Ok(resp),
                    Ok(Err(e)) | Err(e) => Err(e),
                }
            };
            crate::router::resolve_and_route(&model_candidates, &request, tier, discover, route)
        });

        match outcome {
            Ok(o) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("model_id", o.model_id)?;
                dict.set_item("peer_id", o.peer_id)?;
                dict.set_item("response", pyo3::types::PyBytes::new(py, &o.response))?;
                dict.set_item("degraded", o.degraded)?;
                // M2.1: surface the serving provider's raw 32-byte ed25519 public key so
                // the consumer can immediately fire a co-signed receipt at its identity.
                // The record carries it as hex; a malformed/absent key yields empty bytes
                // (a legacy provider that advertised none simply can't be receipted).
                let provider_pub = hex::decode(&o.public_key).unwrap_or_default();
                dict.set_item("provider_pub", pyo3::types::PyBytes::new(py, &provider_pub))?;
                Ok(dict.into_py_any(py)?)
            }
            Err(crate::router::RouteError::NoProvider) => Err(PyRuntimeError::new_err(
                "no_provider: no compatible live provider for any candidate model",
            )),
        }
    }

    /// Get current NAT status.
    ///
    /// Returns:
    ///     dict: {"nat_type": str, "external_ip": str, "external_port": int, "is_public": bool}
    fn nat_status(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        let info = py
            .allow_threads(move || {
                send_and_wait(&cmd_tx, |reply| SwarmCommand::NatStatus { reply })
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let json_str = serde_json::to_string(&info)
            .map_err(|e| PyRuntimeError::new_err(format!("json: {e}")))?;
        let result = py.import("json")?.call_method1("loads", (json_str,))?;
        Ok(result.into_py_any(py)?)
    }

    /// Snapshot of DCUtR hole-punch counters (PR-2 — DCUtR Verification).
    ///
    /// Returns a dict with:
    ///
    /// * ``successes`` (int) — cumulative DCUtR hole-punch successes since
    ///   node start.
    /// * ``failures``  (int) — cumulative DCUtR hole-punch failures. A peer
    ///   that fails DCUtR stays on the relay path.
    /// * ``direct_peers_count`` (int) — number of peers currently in the
    ///   direct-connection set (updated by DCUtR success events and
    ///   non-relay ConnectionEstablished events).
    ///
    /// Surfaces on the HTTP capacity endpoint as ``network.dcutr``; drives
    /// the A3 field-audit benchmark (cross-ISP direct-vs-relay ratio).
    fn get_dcutr_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        let snapshot = py
            .allow_threads(move || {
                send_and_wait(&cmd_tx, |reply| SwarmCommand::GetDcutrStats { reply })
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let (successes, failures, direct_peers_count) = snapshot;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("successes", successes)?;
        dict.set_item("failures", failures)?;
        dict.set_item("direct_peers_count", direct_peers_count)?;
        Ok(dict.into_py_any(py)?)
    }

    /// F-3: per-tier connection-success metrics + DCUtR outcome counters.
    ///
    /// Returns a dict ``{"tiers": {tier_name: count, ...}, "dcutr_successes":
    /// int, "dcutr_failures": int}``. ``tiers`` always contains every ladder
    /// rung (``direct_tcp_v6``, ``direct_quic_v6``, ``direct_tcp_v4``,
    /// ``direct_quic_v4``, ``relay``), 0 if unused — so callers can compute a
    /// success *distribution* across the connection ladder without guessing
    /// which keys exist.
    fn get_tier_metrics(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        let (tiers, dcutr_successes, dcutr_failures) = py
            .allow_threads(move || {
                send_and_wait(&cmd_tx, |reply| SwarmCommand::GetTierMetrics { reply })
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let dict = pyo3::types::PyDict::new(py);
        let tiers_dict = pyo3::types::PyDict::new(py);
        for (name, count) in tiers {
            tiers_dict.set_item(name, count)?;
        }
        dict.set_item("tiers", tiers_dict)?;
        dict.set_item("dcutr_successes", dcutr_successes)?;
        dict.set_item("dcutr_failures", dcutr_failures)?;
        Ok(dict.into_py_any(py)?)
    }

    /// Publish a raw bytes payload on the Gossipsub topic
    /// ``openhydra/swarm/v1/events`` (PR-3 / B1).
    ///
    /// The Python caller (``peer/gossip_client.py``) is responsible for the
    /// JSON codec — this method is intentionally format-agnostic so future
    /// event types can evolve without a wheel rebuild.
    ///
    /// Returns ``None`` on success; raises ``PyRuntimeError`` if publishing
    /// fails (no subscribed peers yet, payload too large, signing error).
    /// A common benign failure is ``InsufficientPeers`` right after boot,
    /// before the mesh has formed — callers should treat it as retryable.
    fn publish_event(&self, py: Python<'_>, payload: Vec<u8>) -> PyResult<()> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            send_and_wait(&cmd_tx, |reply| SwarmCommand::PublishEvent {
                payload,
                reply,
            })
        })
        .map_err(|e| PyRuntimeError::new_err(e))?
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Drain the oldest queued inbound gossip message (PR-3 / B1).
    ///
    /// Returns a tuple ``(sender_libp2p_peer_id, payload_bytes)`` or
    /// ``None`` when the queue is empty. The sender id is the immediate
    /// propagation hop — callers that need the original author must
    /// extract it from the JSON payload itself.
    fn poll_event(&self, py: Python<'_>) -> PyResult<Option<(String, Vec<u8>)>> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        let result = py
            .allow_threads(move || {
                send_and_wait(&cmd_tx, |reply| SwarmCommand::PollEvent { reply })
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;
        Ok(result)
    }

    /// Issue an active dial to a libp2p peer id (PR-3 / B1 rendezvous).
    ///
    /// Intended to be called from a ``REQUEST_HOLE_PUNCH`` gossip
    /// subscriber when the inbound event targets this node: the remote
    /// peer has just asked us to simultaneously dial them, so we do.
    /// Both sides dialling at the same time under a coordinated ~100 ms
    /// gossip-delivery window is what gives DCUtR a real chance of
    /// hole-punching through symmetric NAT.
    ///
    /// Fire-and-forget in the sense that success / failure of the dial
    /// itself is not returned here — it's surfaced asynchronously via
    /// ``ConnectionEstablished`` / ``DialFailure`` events that already
    /// drive the direct-peers set and the DCUtR counters. Returns
    /// ``None`` once the dial has been enqueued; raises
    /// ``PyRuntimeError`` only on validation failure (bad peer id).
    fn dial_peer(&self, py: Python<'_>, peer_id: String) -> PyResult<()> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            send_and_wait(&cmd_tx, |reply| SwarmCommand::DialPeer { peer_id, reply })
        })
        .map_err(|e| PyRuntimeError::new_err(e))?
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Explicitly populate Kademlia's routing table with a known
    /// ``(peer_id, multiaddr)`` pair so the next ``dial_peer`` /
    /// ``proxy_forward`` can find a route.
    ///
    /// The ``discover()`` Kademlia walk returns ``DiscoveredPeer`` records
    /// with ``relay_address`` strings up to Python, but the Swarm's
    /// per-peer address book is only auto-populated for peers that have
    /// already been dialed. For peers we learn about second-hand
    /// (e.g. via ``--peers-config`` or via a relay record pushed through
    /// the HTTP DHT), Python must feed the addresses back into Kademlia
    /// explicitly.
    ///
    /// Args:
    ///     peer_id: the libp2p peer id as a base-58 string.
    ///     multiaddr: a full multiaddr string, e.g.
    ///         ``"/ip4/45.79.190.172/tcp/4001/p2p/12D3KooW.../p2p-circuit/p2p/12D3KooW..."``
    ///         for a relayed address, or ``"/ip4/10.192.11.74/tcp/4001"``
    ///         for a LAN-direct address.
    ///
    /// Raises ``PyRuntimeError`` on parse failure. Returns ``None`` on
    /// success — Kademlia's ``add_address`` is internally idempotent so
    /// repeat calls are harmless.
    fn add_address(&self, py: Python<'_>, peer_id: String, multiaddr: String) -> PyResult<()> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            send_and_wait(&cmd_tx, |reply| SwarmCommand::AddAddress {
                peer_id,
                multiaddr,
                reply,
            })
        })
        .map_err(|e| PyRuntimeError::new_err(e))?
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Resolve a reachable address for a peer (direct host:port or relay multiaddr).
    fn resolve_address(&self, py: Python<'_>, peer_id: String) -> PyResult<String> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            send_and_wait(&cmd_tx, |reply| SwarmCommand::ResolveAddress { peer_id, reply })
        })
        .map_err(|e| PyRuntimeError::new_err(e))?
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Poll for the next inbound proxy request from a remote peer.
    /// Returns (request_id, data_bytes) or None if queue is empty.
    /// CRITICAL: releases GIL during the blocking recv to avoid deadlocking
    /// other Python threads (gRPC server, coordinator, announce loop).
    #[pyo3(signature = (timeout_ms=500))]
    fn poll_proxy_request(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<Option<(String, Vec<u8>)>> {
        let inner = self.require_started()?;
        let queue = Arc::clone(&inner.proxy_queue);
        let timeout = std::time::Duration::from_millis(timeout_ms);
        Ok(py.allow_threads(move || queue.pop(timeout)))
    }

    /// Send a response to an inbound proxy request (identified by request_id).
    fn respond_proxy(&self, py: Python<'_>, request_id: String, data: Vec<u8>) -> PyResult<()> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            cmd_tx
                .blocking_send(SwarmCommand::RespondProxy { request_id, data })
                .map_err(|_| "swarm not running".to_string())
        })
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Forward raw bytes to a peer via libp2p (through Circuit Relay if needed).
    /// Returns response bytes. Used for cross-ISP gRPC tunneling.
    fn proxy_forward(&self, py: Python<'_>, target_peer_id: String, data: Vec<u8>) -> PyResult<Vec<u8>> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            send_and_wait(&cmd_tx, |reply| SwarmCommand::ProxyForward {
                peer_id: target_peer_id,
                data,
                reply,
            })
        })
        .map_err(|e| PyRuntimeError::new_err(e))?
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Fire-and-forget variant of proxy_forward — sends raw bytes to a
    /// peer via libp2p but returns immediately without waiting for an
    /// ACK/response. Eliminates the ~200ms synchronous wait per token
    /// in cross-ISP push mode.
    fn proxy_forward_no_wait(&self, py: Python<'_>, target_peer_id: String, data: Vec<u8>) -> PyResult<()> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            cmd_tx
                .blocking_send(SwarmCommand::ProxyForwardNoWait {
                    peer_id: target_peer_id,
                    data,
                })
                .map_err(|_| "swarm not running".to_string())
        })
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Open a proxy connection to a remote peer. Dials the peer via libp2p
    /// (through Circuit Relay if needed) and sets the local gRPC port for
    /// inbound proxy requests.
    fn open_proxy(&self, py: Python<'_>, target_peer_id: String, local_grpc_port: u16) -> PyResult<String> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            send_and_wait(&cmd_tx, |reply| SwarmCommand::OpenProxy {
                target_libp2p_peer_id: target_peer_id,
                local_grpc_port,
                reply,
            })
        })
        .map_err(|e| PyRuntimeError::new_err(e))?
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Check if a peer is currently connected (direct or relayed).
    fn is_peer_connected(&self, py: Python<'_>, peer_id: String) -> PyResult<bool> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            send_and_wait(&cmd_tx, |reply| SwarmCommand::IsConnected {
                peer_id,
                reply,
            })
        })
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Get detailed connection info for a peer (Fix 4 observability).
    ///
    /// Returns a dict with has_quic, has_tcp_direct, has_relay,
    /// preferred_transport, or None if the peer is unknown.
    fn get_connection_info(&self, py: Python<'_>, peer_id: String) -> PyResult<Option<PyObject>> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        let snapshot = py
            .allow_threads(move || {
                send_and_wait(&cmd_tx, |reply| SwarmCommand::GetConnectionInfo {
                    peer_id,
                    reply,
                })
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;

        match snapshot {
            Some(info) => {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("has_quic", info.has_quic)?;
                dict.set_item("has_tcp_direct", info.has_tcp_direct)?;
                dict.set_item("has_relay", info.has_relay)?;
                dict.set_item("preferred_transport", info.preferred_transport)?;
                Ok(Some(dict.into_py_any(py)?))
            }
            None => Ok(None),
        }
    }

    /// Open a TCP-to-libp2p tunnel to a remote peer.
    fn open_tunnel(&self, py: Python<'_>, peer_id: String) -> PyResult<String> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            send_and_wait(&cmd_tx, |reply| SwarmCommand::OpenTunnel { peer_id, reply })
        })
        .map_err(|e| PyRuntimeError::new_err(e))?
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Close an active tunnel to a remote peer.
    fn close_tunnel(&self, py: Python<'_>, peer_id: String) -> PyResult<()> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            cmd_tx
                .blocking_send(SwarmCommand::CloseTunnel { peer_id })
                .map_err(|_| "swarm not running".to_string())
        })
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Poll for DCUtR success events.
    fn poll_dcutr_event(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        let result = py
            .allow_threads(move || {
                send_and_wait(&cmd_tx, |reply| SwarmCommand::PollDcutrEvent { reply })
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;
        Ok(result)
    }

    /// Poll for tunnel close events.
    fn poll_tunnel_close_event(&self, py: Python<'_>) -> PyResult<Option<String>> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        let result = py
            .allow_threads(move || {
                send_and_wait(&cmd_tx, |reply| SwarmCommand::PollTunnelCloseEvent { reply })
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;
        Ok(result)
    }

    /// Dial a raw multiaddr (for manual hole-punch testing).
    fn dial_address(&self, py: Python<'_>, multiaddr: String) -> PyResult<()> {
        let inner = self.require_started()?;
        let cmd_tx = inner.cmd_tx.clone();
        py.allow_threads(move || {
            send_and_wait(&cmd_tx, |reply| SwarmCommand::DialAddress { multiaddr, reply })
        })
        .map_err(|e| PyRuntimeError::new_err(e))?
        .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// The libp2p PeerId (base58 multihash).
    #[getter]
    fn libp2p_peer_id(&self) -> &str {
        &self.libp2p_peer_id
    }

    /// The OpenHydra peer_id (SHA256[:16] hex).
    #[getter]
    fn openhydra_peer_id(&self) -> &str {
        &self.openhydra_peer_id
    }

    // ── OHV2 wire format: PyO3 bindings for ForwardMsg encode/decode ──
    //
    // These static methods expose the Rust CBOR-based wire format to Python,
    // replacing protobuf serialization on the per-token hot path.

    /// Encode a forward/push message in OHV2 wire format.
    ///
    /// Args:
    ///     header_dict: Python dict with IpcForwardHeader fields (only non-default
    ///         fields need to be present — serde fills defaults).
    ///     activation: raw activation bytes (already binary-packed by Python).
    ///     msg_type: 0=Forward, 1=PushResult, 2=Ping.
    ///
    /// Returns:
    ///     bytes: the OHV2 wire-format message (12-byte preamble + CBOR header + activation).
    #[staticmethod]
    fn encode_forward_msg(py: Python<'_>, header_dict: &Bound<'_, pyo3::types::PyDict>, activation: &[u8], msg_type: u16) -> PyResult<PyObject> {
        // Convert PyDict → serde_json::Value → IpcForwardHeader.
        // This reuses all skip_serializing_if / default annotations on IpcForwardHeader.
        let json_val = pydict_to_json_value(header_dict)?;
        let header: IpcForwardHeader = serde_json::from_value(json_val)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid header fields: {e}")))?;

        let mt = forward_msg::MsgType::from_u16(msg_type)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let wire = forward_msg::encode(mt, &header, activation)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        Ok(pyo3::types::PyBytes::new(py, &wire).into())
    }

    /// Decode an OHV2 wire-format message.
    ///
    /// Args:
    ///     data: raw OHV2 bytes (must start with magic 0x4F485632).
    ///
    /// Returns:
    ///     tuple: (header_dict, activation_bytes, msg_type)
    ///         header_dict: Python dict with all IpcForwardHeader fields.
    ///         activation_bytes: raw activation payload (bytes).
    ///         msg_type: int (0=Forward, 1=PushResult, 2=Ping).
    #[staticmethod]
    fn decode_forward_msg(py: Python<'_>, data: &[u8]) -> PyResult<(PyObject, PyObject, u16)> {
        let decoded = forward_msg::decode(data)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        // Convert IpcForwardHeader → serde_json::Value → PyDict.
        let json_val = serde_json::to_value(&decoded.header)
            .map_err(|e| PyRuntimeError::new_err(format!("header to json: {e}")))?;
        let dict = json_value_to_pyobject(py, &json_val)?;

        let act_bytes = pyo3::types::PyBytes::new(py, decoded.activation);

        Ok((dict, act_bytes.into(), decoded.msg_type as u16))
    }

    /// Encode an OHV2 response message (PushResult).
    ///
    /// Args:
    ///     header_dict: Python dict with IpcResponseHeader fields.
    ///     activation: raw activation bytes.
    ///
    /// Returns:
    ///     bytes: OHV2 wire-format response.
    #[staticmethod]
    fn encode_response_msg(py: Python<'_>, header_dict: &Bound<'_, pyo3::types::PyDict>, activation: &[u8]) -> PyResult<PyObject> {
        let json_val = pydict_to_json_value(header_dict)?;
        let header: IpcResponseHeader = serde_json::from_value(json_val)
            .map_err(|e| PyRuntimeError::new_err(format!("invalid response header: {e}")))?;

        let wire = forward_msg::encode_response(&header, activation)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        Ok(pyo3::types::PyBytes::new(py, &wire).into())
    }

    /// Decode an OHV2 response message (PushResult).
    ///
    /// Args:
    ///     data: raw OHV2 bytes (must start with magic 0x4F485632).
    ///
    /// Returns:
    ///     tuple: (header_dict, activation_bytes)
    ///         header_dict: Python dict with IpcResponseHeader fields.
    ///         activation_bytes: raw activation payload (bytes).
    #[staticmethod]
    fn decode_response_msg(py: Python<'_>, data: &[u8]) -> PyResult<(PyObject, PyObject)> {
        let (header, activation) = forward_msg::decode_response(data)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let json_val = serde_json::to_value(&header)
            .map_err(|e| PyRuntimeError::new_err(format!("response header to json: {e}")))?;
        let dict = json_value_to_pyobject(py, &json_val)?;

        let act_bytes = pyo3::types::PyBytes::new(py, activation);

        Ok((dict, act_bytes.into()))
    }

    /// Check if raw bytes start with the OHV2 magic (0x4F485632).
    ///
    /// Use this for dual-format detection: if True, decode with
    /// decode_forward_msg; otherwise fall back to protobuf.
    #[staticmethod]
    fn is_ohv2_msg(data: &[u8]) -> bool {
        forward_msg::is_forward_msg(data)
    }

    // ── Task 6.0: Identity methods ──

    /// Return the libp2p PeerId as a base58 string.
    /// Alias for the libp2p_peer_id property — named for clarity in identity code.
    fn peer_id_base58(&self) -> &str {
        &self.libp2p_peer_id
    }

    /// Sign arbitrary data with the node's Ed25519 keypair.
    /// Returns the raw 64-byte Ed25519 signature.
    fn sign_record(&self, data: Vec<u8>) -> PyResult<Vec<u8>> {
        self.keypair
            .sign(&data)
            .map_err(|e| PyRuntimeError::new_err(format!("sign failed: {e}")))
    }

    /// Sign a receipt payload as the CONSUMER, using this node's **internal** ed25519
    /// identity key — the key never crosses the FFI. The node is the consumer;
    /// `provider_pub` names the peer that served the request. Returns the 64-byte
    /// consumer signature (protocol.md §6, M2.1).
    fn receipt_sign(
        &self,
        py: Python<'_>,
        provider_pub: Vec<u8>,
        model_id: &str,
        tokens: u64,
        nonce: Vec<u8>,
        ts_unix_ms: u64,
    ) -> PyResult<PyObject> {
        let consumer_pub = self.public_key_bytes()?;
        let payload = crate::receipts::payload_from_bytes(
            &provider_pub,
            &consumer_pub,
            model_id,
            tokens,
            &nonce,
            ts_unix_ms,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let sig = self
            .keypair
            .sign(&payload.canonical_bytes())
            .map_err(|e| PyRuntimeError::new_err(format!("sign failed: {e}")))?;
        Ok(pyo3::types::PyBytes::new(py, &sig).into_py_any(py)?)
    }

    /// Co-sign a receipt as the PROVIDER, using this node's **internal** ed25519
    /// identity key, after verifying the whole package. `provider_pub` must equal this
    /// node's identity (a peer cannot be made to co-sign a receipt naming a different
    /// provider). Returns the 64-byte provider signature; raises on a key mismatch or
    /// a bad consumer signature (protocol.md §6, M2.1).
    #[allow(clippy::too_many_arguments)]
    fn receipt_cosign(
        &self,
        py: Python<'_>,
        provider_pub: Vec<u8>,
        consumer_pub: Vec<u8>,
        model_id: &str,
        tokens: u64,
        nonce: Vec<u8>,
        ts_unix_ms: u64,
        consumer_sig: Vec<u8>,
    ) -> PyResult<PyObject> {
        if provider_pub != self.public_key_bytes()? {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "receipt: provider key does not match this node's identity",
            ));
        }
        let payload = crate::receipts::payload_from_bytes(
            &provider_pub,
            &consumer_pub,
            model_id,
            tokens,
            &nonce,
            ts_unix_ms,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let consumer_sig_arr: [u8; 64] = consumer_sig.as_slice().try_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("consumer signature must be 64 bytes")
        })?;
        let consumer_sig_obj = ed25519_dalek::Signature::from_bytes(&consumer_sig_arr);
        // Co-sign (payload ‖ consumer_sig) with the node's internal identity key.
        let cosign = crate::receipts::cosign_bytes(&payload, &consumer_sig_obj);
        let provider_sig = self
            .keypair
            .sign(&cosign)
            .map_err(|e| PyRuntimeError::new_err(format!("sign failed: {e}")))?;
        let provider_sig_arr: [u8; 64] = provider_sig
            .as_slice()
            .try_into()
            .map_err(|_| PyRuntimeError::new_err("produced a bad signature length"))?;
        // Verify the whole package before returning — rejects a bad consumer signature.
        let receipt = crate::receipts::CoSignedReceipt {
            payload,
            consumer_sig: consumer_sig_obj,
            provider_sig: ed25519_dalek::Signature::from_bytes(&provider_sig_arr),
        };
        crate::receipts::verify_receipt(&receipt).map_err(|e| {
            let which = match e {
                crate::receipts::ReceiptError::BadConsumerSig => "bad_consumer_sig",
                crate::receipts::ReceiptError::BadProviderSig => "bad_provider_sig",
                crate::receipts::ReceiptError::ReplayedNonce => "replayed_nonce",
            };
            pyo3::exceptions::PyValueError::new_err(format!("receipt rejected: {which}"))
        })?;
        Ok(pyo3::types::PyBytes::new(py, &provider_sig).into_py_any(py)?)
    }

    /// Export the raw 32-byte Ed25519 public key.
    fn public_key_bytes(&self) -> PyResult<Vec<u8>> {
        let ed25519_pk = self
            .keypair
            .public()
            .try_into_ed25519()
            .map_err(|e| PyRuntimeError::new_err(format!("not ed25519: {e}")))?;
        Ok(ed25519_pk.to_bytes().to_vec())
    }

    /// Export the hex-encoded 32-byte Ed25519 public key.
    fn public_key_hex(&self) -> PyResult<String> {
        let bytes = self.public_key_bytes()?;
        Ok(hex::encode(bytes))
    }

    // ── M2.2: reputation feedback loop ──

    /// Record a verification/receipt outcome for a peer into the live reputation store
    /// (protocol.md §7). `outcome` is one of ``"honored"``, ``"rejected"``, ``"failed"``.
    /// The Python receipt lifecycle calls this after a transaction so the router can
    /// downrank misbehaving providers. Returns the peer's new (post-update) score.
    /// Raises ``ValueError`` on an unknown outcome string.
    fn record_reputation_outcome(&self, peer_id: String, outcome: &str) -> PyResult<f64> {
        let outcome = parse_outcome(outcome)?;
        let now_ms = now_unix_ms();
        let mut guard = self
            .reputation
            .write()
            .map_err(|_| PyRuntimeError::new_err("reputation store poisoned"))?;
        let tracker = guard
            .entry(peer_id)
            .or_insert_with(|| crate::verify::ReputationTracker::new(now_ms));
        Ok(tracker.record(outcome, now_ms))
    }

    /// The peer's current locally-observed, time-decayed reputation score. Returns the
    /// neutral baseline (50.0) for a peer we've never recorded an outcome for — the same
    /// value the router uses for an unknown peer. Read-only (does not mutate the store).
    fn reputation_score(&self, peer_id: String) -> PyResult<f64> {
        let now_ms = now_unix_ms();
        let guard = self
            .reputation
            .read()
            .map_err(|_| PyRuntimeError::new_err("reputation store poisoned"))?;
        Ok(guard
            .get(&peer_id)
            .map(|t| t.score_at(now_ms))
            .unwrap_or(crate::verify::NEUTRAL_REPUTATION))
    }

    /// Commit an **honored** receipt to the persistent ledger and bump the provider's
    /// reputation, in one durable step (protocol.md §6/§7, M2.3).
    ///
    /// The receipt is passed by components (like `receipt_sign`/`receipt_cosign`) so the
    /// byte encoding lives only in Rust. This method: assembles the `CoSignedReceipt`,
    /// **re-verifies both signatures** (never bump trust on a bad receipt), computes the
    /// provider's `+Honored` reputation, **flushes receipt + burnt nonce + reputation to
    /// the redb ledger atomically**, and only then commits the bump to the in-memory map
    /// and marks the nonce in the in-memory replay guard. Durability-first: if the flush
    /// fails, in-memory state is left untouched.
    ///
    /// `peer_id` keys the reputation (the same key the router and `record_reputation_outcome`
    /// use — keeping honored/rejected/failed unified). Returns the provider's new score.
    #[allow(clippy::too_many_arguments)]
    fn commit_honored_receipt(
        &self,
        peer_id: String,
        provider_pub: Vec<u8>,
        consumer_pub: Vec<u8>,
        model_id: &str,
        tokens: u64,
        nonce: Vec<u8>,
        ts_unix_ms: u64,
        consumer_sig: Vec<u8>,
        provider_sig: Vec<u8>,
    ) -> PyResult<f64> {
        let payload = crate::receipts::payload_from_bytes(
            &provider_pub,
            &consumer_pub,
            model_id,
            tokens,
            &nonce,
            ts_unix_ms,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let consumer_sig_arr: [u8; 64] = consumer_sig.as_slice().try_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("consumer signature must be 64 bytes")
        })?;
        let provider_sig_arr: [u8; 64] = provider_sig.as_slice().try_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("provider signature must be 64 bytes")
        })?;
        let nonce_arr: [u8; 16] = nonce
            .as_slice()
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("nonce must be 16 bytes"))?;
        let receipt = crate::receipts::CoSignedReceipt {
            payload,
            consumer_sig: ed25519_dalek::Signature::from_bytes(&consumer_sig_arr),
            provider_sig: ed25519_dalek::Signature::from_bytes(&provider_sig_arr),
        };
        // Defense in depth — never persist or reward a receipt that doesn't verify.
        crate::receipts::verify_receipt(&receipt).map_err(|e| {
            let which = match e {
                crate::receipts::ReceiptError::BadConsumerSig => "bad_consumer_sig",
                crate::receipts::ReceiptError::BadProviderSig => "bad_provider_sig",
                crate::receipts::ReceiptError::ReplayedNonce => "replayed_nonce",
            };
            pyo3::exceptions::PyValueError::new_err(format!("receipt rejected: {which}"))
        })?;

        let now_ms = now_unix_ms();
        // Update reputation under the write lock; flush to disk BEFORE committing the bump
        // to memory, so a flush failure can't leave memory ahead of the durable ledger.
        let new_score = {
            let mut rep = self
                .reputation
                .write()
                .map_err(|_| PyRuntimeError::new_err("reputation store poisoned"))?;
            let mut updated = rep
                .get(&peer_id)
                .cloned()
                .unwrap_or_else(|| crate::verify::ReputationTracker::new(now_ms));
            let score = updated.record(crate::verify::VerificationOutcome::Honored, now_ms);
            self.store
                .flush_receipt_and_reputation(&receipt, &peer_id, &updated)
                .map_err(|e| PyRuntimeError::new_err(format!("ledger flush: {e}")))?;
            rep.insert(peer_id.clone(), updated); // commit to memory only after a durable flush
            score
        };
        // The nonce is already durable (flushed above); mirror it into the in-memory guard.
        self.nonces
            .write()
            .map_err(|_| PyRuntimeError::new_err("nonce guard poisoned"))?
            .mark_seen(nonce_arr);
        Ok(new_score)
    }

    /// Whether a receipt nonce has already been spent, per the in-memory replay guard
    /// (rehydrated from the ledger on boot). A restart-durable replay check at the FFI.
    fn has_seen_nonce(&self, nonce: Vec<u8>) -> PyResult<bool> {
        let nonce_arr: [u8; 16] = nonce
            .as_slice()
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("nonce must be 16 bytes"))?;
        let guard = self
            .nonces
            .read()
            .map_err(|_| PyRuntimeError::new_err("nonce guard poisoned"))?;
        Ok(guard.contains(&nonce_arr))
    }

    /// Rank discovered peers exactly as `resolve_and_route`'s internal rank stage does —
    /// applying the **same** local-reputation override (`effective_reputation`) — but
    /// over a caller-supplied candidate list instead of a live DHT discovery. This is the
    /// in-process seam for exercising the reputation→ranking path without a live swarm.
    ///
    /// `peers` is a list of dicts (as `discover()` returns); only the scoring keys are
    /// read: ``peer_id`` (required), and optional ``reputation_score``, ``throughput_tok_s``,
    /// ``queue_depth``, ``load_pct``, ``latency_ms``. Returns a list of
    /// ``{"peer_id", "score"}`` dicts, best-first.
    fn rank_discovered(
        &self,
        py: Python<'_>,
        peers: &Bound<'_, pyo3::types::PyList>,
        tier: u8,
    ) -> PyResult<PyObject> {
        let now_ms = now_unix_ms();
        let mut inputs: Vec<crate::router::PeerScoreInput> = Vec::with_capacity(peers.len());
        for item in peers.iter() {
            let d = item
                .downcast::<pyo3::types::PyDict>()
                .map_err(|_| PyRuntimeError::new_err("rank_discovered: each peer must be a dict"))?;
            let peer_id: String = match d.get_item("peer_id")? {
                Some(v) => v.extract()?,
                None => return Err(PyRuntimeError::new_err("rank_discovered: peer missing 'peer_id'")),
            };
            let dht_rep = dict_get_f64(d, "reputation_score", 0.0)?;
            inputs.push(crate::router::PeerScoreInput {
                latency_ms: dict_get_f64(d, "latency_ms", 1.0)?,
                load_pct: dict_get_f64(d, "load_pct", 0.0)?,
                // identical override to the live resolve_and_route discover closure.
                reputation: effective_reputation(&self.reputation, &peer_id, dht_rep, now_ms),
                bandwidth_mbps: 0.0,
                s2s_rtt_ms: 0.0,
                throughput_tok_s: dict_get_f64(d, "throughput_tok_s", 0.0)?,
                queue_depth: dict_get_u32(d, "queue_depth", 0)?,
                peer_id,
            });
        }
        let out = pyo3::types::PyList::empty(py);
        for sp in crate::router::rank_peers(&inputs, tier) {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("peer_id", sp.peer_id)?;
            dict.set_item("score", sp.score)?;
            out.append(dict)?;
        }
        Ok(out.into_py_any(py)?)
    }
}

#[cfg(feature = "pyo3")]
impl PyP2PNode {
    fn require_started(&self) -> PyResult<&NodeInner> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("node not started — call start() first"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bootstrap_peers() {
        let addrs = vec![
            "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN".into(),
        ];
        let result = parse_bootstrap_peers(&addrs).unwrap();
        assert_eq!(result.len(), 1);
        let (_peer_id, addr) = &result[0];
        assert!(!addr.to_string().contains("/p2p/"));
        assert!(addr.to_string().contains("/tcp/4001"));
    }

    #[test]
    fn test_parse_bootstrap_peers_missing_p2p() {
        let addrs = vec!["/ip4/45.79.190.172/tcp/4001".into()];
        assert!(parse_bootstrap_peers(&addrs).is_err());
    }

    #[test]
    fn test_default_config() {
        let config = NodeConfig::default();
        assert!(config.identity_path.to_string_lossy().contains("identity.key"));
        assert_eq!(config.listen_addrs.len(), 4); // TCP + QUIC, IPv4 + IPv6
    }
}
