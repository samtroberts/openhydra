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
use tracing::{info, warn};


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
    /// Tier-2 connection reversal (off by default): when this node holds only a
    /// relayed connection to a peer that advertises globally-routable direct
    /// addresses, dial those directly to escape the relay. The one NAT-escape that
    /// works on symmetric CGNAT (our outbound dial traverses our own NAT), where
    /// DCUtR's simultaneous-open cannot. See docs/PEER_CONNECTIVITY.md.
    pub enable_connection_reversal: bool,
    /// #43-W2: CPE gateway address for the PCP (RFC 6887) inbound-v6
    /// firewall-pinhole maintainer. `None` (default) disables PCP entirely. When
    /// set, the node periodically asks this gateway to open its listen ports
    /// inbound on its global v6 so AutoNAT can confirm reachability and promote
    /// it — the v6 sibling of R-DHT-4's v4 UPnP/NAT-PMP mapping. Opt-in because
    /// gateway discovery is not yet automatic and PCP is unvalidated live.
    pub pcp_gateway: Option<std::net::IpAddr>,
    /// Whether to relay through the production relays as well as the bootstraps
    /// ([`crate::relay::PublicRelays`]; `Auto` by default).
    pub public_relays: crate::relay::PublicRelays,
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
            enable_connection_reversal: false,
            pcp_gateway: None,
            public_relays: crate::relay::PublicRelays::Auto,
        }
    }
}

static MDNS_DISABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn LAN discovery (mDNS) off for every node this process starts from now on (`--no-mdns`): an
/// off switch if LAN dialling misbehaves somewhere, without a new release. A process-wide setting
/// rather than a [`NodeConfig`] field, so the many places that build a `NodeConfig` don't change.
pub fn disable_mdns() {
    MDNS_DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
}

fn mdns_enabled() -> bool {
    !MDNS_DISABLED.load(std::sync::atomic::Ordering::Relaxed)
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
) -> Result<
    (
        mpsc::Sender<SwarmCommand>,
        Arc<SharedProxyQueue>,
        std::thread::JoinHandle<()>,
        // #42: shared network-generation counter — bumped on every rebootstrap
        // so the handle (and the provider run-loop) can observe a network change
        // and re-announce.
        Arc<std::sync::atomic::AtomicU64>,
        // The latest ping round trip to each peer, recorded by the event loop.
        Arc<crate::rtt::PeerRtts>,
        // The peers dialled on pinned addresses (`crate::pinned`), shared with the event loop.
        crate::pinned::PinnedPeers,
        // Whether the node is on OpenHydra's public network (`relay::uses_public_network`).
        bool,
        // The node's push streams (#4, `crate::push`).
        Arc<crate::push::PushStreams>,
    ),
    String,
> {
    // Freeze the process's network settings (key, allowlist, QUIC) before reading any of them below:
    // `build_swarm` freezes them too, but only after this function has already decided relays,
    // bootstraps and the routing-cache scope from them, so a setter called in between would leave
    // this node half on the old settings and half on the new.
    crate::transport::freeze_network_settings();

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

    // A private network (a pre-shared key) can't reach OpenHydra's public bootstraps or relays, which
    // don't have its key: it runs as a private deployment, `--public-relays off`.
    let public_relays = effective_public_relays(config);
    if let Some(key) = crate::transport::private_network_key() {
        info!(
            fingerprint = %crate::transport::private_network_fingerprint(&key),
            quic = crate::transport::quic_enabled(),
            "private network: TCP and relay circuits keyed, QUIC only with a peer allowlist, no public bootstraps or relays"
        );
    }
    // Parse bootstrap peers: "/ip4/.../tcp/.../p2p/12D3KooW..."
    let mut bootstrap_peers = parse_bootstrap_peers(&config.bootstrap_peers)?;
    if bootstrap_peers.is_empty() && fenced_without_bootstraps(config) {
        // Not OpenHydra's public bootstraps: the allowlist would refuse every one of them, and the
        // node would sit there with no clear reason why.
        info!("peer allowlist set and no --bootstrap given: not using OpenHydra's public bootstraps (a fenced node only reaches the peers it is given; pass --bootstrap)");
    }
    // SL5: given none (a fresh CLI install), start from OpenHydra's own on the public network.
    if bootstrap_peers.is_empty() {
        let defaults = crate::relay::default_bootstraps(public_relays);
        if !defaults.is_empty() {
            info!(n = defaults.len(), "no --bootstrap given: starting from OpenHydra's public bootstraps");
            bootstrap_peers = parse_bootstrap_peers(&defaults)?;
        }
    }
    // A private deployment never contacts OpenHydra's own bootstraps, even ones it was given.
    if public_relays == crate::relay::PublicRelays::Off {
        let (kept, dropped) = crate::relay::without_public_bootstraps(bootstrap_peers);
        for (pid, addr) in &dropped {
            warn!(%pid, %addr, "--public-relays off: not using OpenHydra's public bootstrap");
        }
        bootstrap_peers = kept;
    }
    let public_network = crate::relay::uses_public_network(&bootstrap_peers, public_relays);
    let network_id = crate::relay::network_id(&bootstrap_peers, public_network);
    if public_relays == crate::relay::PublicRelays::Auto && !public_network {
        // Said once, plainly: a node given only its own bootstraps relays through them alone (L9).
        info!("none of the bootstraps is OpenHydra's: using them as relays, not OpenHydra's (pass --public-relays on to use OpenHydra's relays too)");
    }
    let enable_peer_relay = config.enable_peer_relay; // WS-F F-4 (captured into the thread)
    // A private deployment never connects to OpenHydra's own peers, however it hears of them.
    let fence_openhydra = public_relays == crate::relay::PublicRelays::Off;
    // The relays this node reserves on and dials others through: its bootstraps, plus the
    // production relays on the public network (see `PublicRelays`).
    let relays = crate::relay::relay_addrs(&bootstrap_peers, public_relays);
    info!(relays = relays.len(), mode = ?public_relays, "relay set");
    let enable_connection_reversal = config.enable_connection_reversal; // Tier-2 (captured into the thread)

    // #43-W2: resolve the opt-in PCP v6-pinhole wiring (gateway + the listen
    // ports to open inbound). None disables PCP. Derived here while we still hold
    // `listen_addrs`, then moved into the thread.
    let pcp_bind = config.pcp_gateway.map(|gateway| crate::pcp::PcpBind {
        gateway,
        ports: crate::pcp::ports_from_listen_addrs(&listen_addrs),
    });

    // R-DHT-6: persist/reload the routing table beside the identity key. Derived
    // here (config is a borrow that can't move into the thread) and handed to the
    // event loop.
    // Kept apart for a private deployment: a node that was once on the public network must not dial
    // the public peers it remembers (M1).
    // Each private deployment keeps its own too (by its network key, else its network id), so a node
    // moved from one to another doesn't dial the first one's peers.
    let private_scope = (!public_network).then(|| {
        let fingerprint = crate::transport::private_network_key().map(|k| crate::transport::private_network_fingerprint(&k));
        // An unkeyed node with no bootstraps and no network name has the network id every such node
        // has (a hash of no bootstraps); with an allowlist, its allowed peers tell it apart instead.
        // (A named network's id is `private-name-…`; a hash id is hex, so never starts that way.)
        let unnamed_without_bootstraps = bootstrap_peers.is_empty() && !network_id.starts_with("private-name-");
        match crate::fence::allowed_peers() {
            Some(allowed) if fingerprint.is_none() && unnamed_without_bootstraps => crate::routing_cache::allowlist_scope(allowed),
            _ => crate::routing_cache::private_scope(&network_id, fingerprint.as_deref()),
        }
    });
    let routing_cache_path = Some(crate::routing_cache::cache_path_for(&config.identity_path, private_scope.as_deref()));

    // Create the command channel.
    let (cmd_tx, cmd_rx) = mpsc::channel::<SwarmCommand>(256);

    // Shared proxy queue: event loop pushes, the handle's poll_inbound pops.
    let proxy_queue = Arc::new(SharedProxyQueue::new());
    let proxy_queue_clone = Arc::clone(&proxy_queue);

    // #42: shared network-generation counter — one clone into the event loop
    // (which bumps it on rebootstrap), one returned to the handle.
    let net_generation = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let net_generation_loop = Arc::clone(&net_generation);
    let peer_rtts = Arc::new(crate::rtt::PeerRtts::default());
    let peer_rtts_loop = Arc::clone(&peer_rtts);
    let pinned_peers = crate::pinned::PinnedPeers::default();
    let pinned = pinned_peers.clone();

    // Use a oneshot to communicate any startup error from the background thread.
    let (startup_tx, startup_rx) = std::sync::mpsc::channel::<Result<Arc<crate::push::PushStreams>, String>>();

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
                    fence_openhydra,
                    public_network,
                    enable_mdns: mdns_enabled(),
                    network_id,
                    pinned: pinned.clone(),
                };

                let keypair_for_loop = identity.keypair.clone();
                match swarm::build_swarm(&identity, opts) {
                    Ok((swarm, peer_relay_leech)) => {
                        // Push streams (#4): opened by this node as a consumer, accepted as a provider.
                        let push = crate::push::PushStreams::new(swarm.behaviour().push.new_control(), *swarm.local_peer_id());
                        if let Err(e) = push.accept(Arc::clone(&proxy_queue_clone)) {
                            let _ = startup_tx.send(Err(format!("push streams: {e}")));
                            return;
                        }
                        let _ = startup_tx.send(Ok(push));
                        event_loop::run_event_loop(swarm, cmd_rx, proxy_queue_clone, bootstrap_peers_for_dial, relays, keypair_for_loop, peer_relay_leech, routing_cache_path, enable_connection_reversal, net_generation_loop, peer_rtts_loop, pcp_bind, pinned).await;
                    }
                    Err(e) => {
                        let _ = startup_tx.send(Err(format!("build_swarm: {}", crate::transport::error_chain(e.as_ref()))));
                    }
                }
            });
            info!("P2P event loop exited");
        })
        .map_err(|e| format!("thread spawn: {e}"))?;

    // Wait for the background thread to report startup success/failure.
    let push = startup_rx
        .recv()
        .map_err(|_| "background thread died during startup".to_string())?
        .map_err(|e| format!("startup failed: {e}"))?;

    Ok((cmd_tx, proxy_queue, thread, net_generation, peer_rtts, pinned_peers, public_network, push))
}

/// Check one bootstrap multiaddr the way the node will parse it at startup (a valid multiaddr with a
/// `/p2p/<peer id>` component), so a settings form can refuse one the node would fail on.
/// Whether `addr` names one of OpenHydra's own public bootstraps (by its `/p2p/` peer id): a
/// private deployment doesn't use them.
pub fn is_openhydra_bootstrap(addr: &str) -> bool {
    let Ok(ma) = addr.trim().parse::<Multiaddr>() else { return false };
    let pid = ma.iter().find_map(|p| match p {
        libp2p::multiaddr::Protocol::P2p(pid) => Some(pid),
        _ => None,
    });
    pid.is_some_and(|pid| crate::relay::bootstrap_relay_peer_ids().contains(&pid))
}

pub fn check_bootstrap_addr(addr: &str) -> Result<(), String> {
    parse_bootstrap_peers(&[addr.to_string()]).map(|_| ())
}

/// Whether a node with `config` joins the public network ([`crate::relay::uses_public_network`]):
/// no bootstraps, or one of OpenHydra's, under `--public-relays auto`. A node that doesn't is a
/// private deployment, and must not contact OpenHydra's services either (the tokenizer index, the
/// seed list). An unparsable bootstrap fails the node's start anyway.
pub fn joins_public_network(config: &NodeConfig) -> bool {
    let bootstraps = parse_bootstrap_peers(&config.bootstrap_peers).unwrap_or_default();
    crate::relay::uses_public_network(&bootstraps, effective_public_relays(config))
}

/// `config.public_relays`, or `Off` on a private network (a pre-shared key is set), whose key
/// OpenHydra's public bootstraps and relays don't have; `Off` too for a node with a peer allowlist
/// and no bootstraps ([`fenced_without_bootstraps`]).
fn effective_public_relays(config: &NodeConfig) -> crate::relay::PublicRelays {
    if crate::transport::private_network_key().is_some() || fenced_without_bootstraps(config) {
        crate::relay::PublicRelays::Off
    } else {
        config.public_relays
    }
}

/// A peer allowlist is set ([`crate::fence::set_allowed_peers`]) and `config` names no bootstrap. The
/// node then reaches only the peers it is given: OpenHydra's public bootstraps and relays aren't
/// filled in (the allowlist would refuse them), and it doesn't count as on the public network.
fn fenced_without_bootstraps(config: &NodeConfig) -> bool {
    crate::fence::allowed_peers().is_some() && config.bootstrap_peers.is_empty()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openhydras_own_bootstraps_are_recognised_by_peer_id() {
        let us = crate::relay::bootstrap_relay_peer_ids()[0];
        assert!(is_openhydra_bootstrap(&format!("/dns4/bootstrap-us.openhydra.co/tcp/4001/p2p/{us}")));
        assert!(is_openhydra_bootstrap(&format!(" /ip4/1.2.3.4/tcp/4001/p2p/{us} ")), "any address, by peer id");
        let other = libp2p::PeerId::random();
        assert!(!is_openhydra_bootstrap(&format!("/ip4/10.0.0.5/tcp/4001/p2p/{other}")));
        assert!(!is_openhydra_bootstrap("/dns4/bootstrap-us.openhydra.co/tcp/4001"), "no peer id: not judged");
        assert!(!is_openhydra_bootstrap("not an address"));
    }
}
