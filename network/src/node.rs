//! Swarm node bring-up: `start_node` + the blocking `send_and_wait` helper that
//! [`crate::handle::NetworkHandle`] drives.
//!
//! Architecture:
//! ```text
//! caller thread ──[mpsc::Sender<SwarmCommand>]──▶ tokio background thread
//!                                                       │
//!               ◀──[oneshot::Sender<Result>]───── swarm event loop
//! ```
//!
//! All network I/O runs on a dedicated tokio runtime in a background thread.
//! Callers send commands via an mpsc channel and block on a oneshot receiver
//! for the result.

use std::path::PathBuf;
use std::sync::Arc;

use libp2p::{Multiaddr, PeerId};
use tokio::sync::{mpsc, oneshot};
use tracing::info;


use crate::event_loop::{self, SharedProxyQueue, SwarmCommand};
use crate::identity::Identity;
use crate::swarm::{self, SwarmOptions};


/// Swarm node configuration (identity path, listen addrs, bootstrap peers).
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

    // R-DHT-6: persist/reload the routing table beside the identity key. Derived
    // here (config is a borrow that can't move into the thread) and handed to the
    // event loop.
    let routing_cache_path = Some(crate::routing_cache::cache_path_for(&config.identity_path));

    // Create the command channel.
    let (cmd_tx, cmd_rx) = mpsc::channel::<SwarmCommand>(256);

    // Shared proxy queue: event loop pushes, the handle's poll_inbound pops.
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
                        event_loop::run_event_loop(swarm, cmd_rx, proxy_queue_clone, bootstrap_peers_for_dial, stream_control, keypair_for_loop, peer_relay_leech, routing_cache_path).await;
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

// ── Blocking helpers ──

/// Send a command and wait for the reply, blocking the current thread.
/// Used by [`crate::handle::NetworkHandle`].
pub(crate) fn send_and_wait<T>(
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
