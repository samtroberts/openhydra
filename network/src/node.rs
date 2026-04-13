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

use libp2p::{Multiaddr, PeerId};
use tokio::sync::mpsc;
use tracing::info;

#[cfg(feature = "pyo3")]
use tokio::sync::oneshot;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyRuntimeError;
#[cfg(feature = "pyo3")]
use pyo3::IntoPyObjectExt;

use crate::event_loop::{self, SwarmCommand};
use crate::identity::Identity;
use crate::swarm::{self, SwarmOptions};
#[cfg(feature = "pyo3")]
use crate::types::PeerRecord;

/// Internal state shared between the Python-facing struct and the background thread.
#[cfg(feature = "pyo3")]
struct NodeInner {
    cmd_tx: mpsc::Sender<SwarmCommand>,
    /// Handle to the background thread running tokio + swarm.
    _thread: std::thread::JoinHandle<()>,
}

/// Configuration parsed from Python __init__ kwargs.
pub struct NodeConfig {
    pub identity_path: PathBuf,
    pub listen_addrs: Vec<String>,
    pub bootstrap_peers: Vec<String>,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            identity_path: dirs_default_identity(),
            listen_addrs: vec!["/ip4/0.0.0.0/tcp/4001".into()],
            bootstrap_peers: Vec::new(),
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
) -> Result<(mpsc::Sender<SwarmCommand>, std::thread::JoinHandle<()>), String> {
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

    // Create the command channel.
    let (cmd_tx, cmd_rx) = mpsc::channel::<SwarmCommand>(256);

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
                let opts = SwarmOptions {
                    listen_addrs,
                    bootstrap_peers,
                    protocol_version: "openhydra/0.1.0".into(),
                };

                match swarm::build_swarm(&identity, opts) {
                    Ok(swarm) => {
                        let _ = startup_tx.send(Ok(()));
                        event_loop::run_event_loop(swarm, cmd_rx).await;
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

    Ok((cmd_tx, thread))
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

// ── PyO3 class ──

#[cfg(feature = "pyo3")]
#[pyclass(name = "P2PNode")]
pub struct PyP2PNode {
    inner: Option<NodeInner>,
    config: NodeConfig,
    /// Cached identity info (set after start).
    libp2p_peer_id: String,
    openhydra_peer_id: String,
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
    #[new]
    #[pyo3(signature = (identity_key_path=None, listen_addrs=None, bootstrap_peers=None))]
    fn new(
        identity_key_path: Option<String>,
        listen_addrs: Option<Vec<String>>,
        bootstrap_peers: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let config = NodeConfig {
            identity_path: identity_key_path
                .map(PathBuf::from)
                .unwrap_or_else(dirs_default_identity),
            listen_addrs: listen_addrs
                .unwrap_or_else(|| vec!["/ip4/0.0.0.0/tcp/4001".into()]),
            bootstrap_peers: bootstrap_peers.unwrap_or_default(),
        };

        // Pre-load identity to get peer IDs for properties.
        let identity = Identity::load_or_create(&config.identity_path)
            .map_err(|e| PyRuntimeError::new_err(format!("identity: {e}")))?;

        Ok(Self {
            inner: None,
            config,
            libp2p_peer_id: identity.libp2p_peer_id.to_string(),
            openhydra_peer_id: identity.openhydra_peer_id.clone(),
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
        };
        let result = py.allow_threads(|| start_node(&config));
        let (cmd_tx, thread) = result
            .map_err(|e| PyRuntimeError::new_err(format!("start failed: {e}")))?;
        self.inner = Some(NodeInner {
            cmd_tx,
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
        assert_eq!(config.listen_addrs.len(), 1);
    }
}
