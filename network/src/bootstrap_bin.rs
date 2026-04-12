//! Standalone bootstrap node binary for OpenHydra Linode servers.
//!
//! Runs:
//! - Kademlia DHT (bootstrap node mode)
//! - Circuit Relay v2 server (accepts relay reservations)
//! - AutoNAT server (responds to NAT probes)
//! - Identify (peer metadata exchange)
//!
//! Usage:
//! ```bash
//! openhydra-bootstrap \
//!     --identity /opt/openhydra/.libp2p_identity.key \
//!     --listen /ip4/0.0.0.0/tcp/4001
//! ```

use std::path::PathBuf;
use std::time::Duration;

use futures::StreamExt;
use libp2p::swarm::Config as SwarmConfig;
use libp2p::{autonat, identify, kad, relay, Multiaddr, Swarm};
use tracing::info;

// Re-use crate modules for identity and transport.
// Note: since this is a [[bin]], we import the library crate.
use openhydra_network::identity::Identity;

/// Bootstrap-specific behaviour — includes relay::Behaviour (server mode).
#[derive(libp2p::swarm::NetworkBehaviour)]
struct BootstrapBehaviour {
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
    relay_server: relay::Behaviour,
    autonat: autonat::Behaviour,
    identify: identify::Behaviour,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Init tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Parse CLI args (minimal, no clap dependency).
    let args: Vec<String> = std::env::args().collect();
    let identity_path = parse_flag(&args, "--identity")
        .unwrap_or_else(|| "/opt/openhydra/.libp2p_identity.key".to_string());
    let listen_addrs: Vec<Multiaddr> = parse_flag_multi(&args, "--listen")
        .unwrap_or_else(|| vec!["/ip4/0.0.0.0/tcp/4001".parse().unwrap()]);

    info!("loading identity from {identity_path}");
    let identity = Identity::load_or_create(&PathBuf::from(&identity_path))?;
    info!(
        peer_id = %identity.libp2p_peer_id,
        openhydra_id = %identity.openhydra_peer_id,
        "bootstrap node starting"
    );

    let peer_id = identity.libp2p_peer_id;
    let keypair = identity.keypair.clone();

    // Transport: TCP + Noise + Yamux
    let transport = openhydra_network::transport::build_transport(&keypair)?;

    // Kademlia in server mode (bootstrap node).
    let mut kad_config = kad::Config::new(
        libp2p::StreamProtocol::new("/openhydra/kad/1.0.0"),
    );
    kad_config.set_query_timeout(Duration::from_secs(30));
    kad_config.set_record_ttl(Some(Duration::from_secs(600)));
    kad_config.set_provider_record_ttl(Some(Duration::from_secs(600)));
    kad_config.set_publication_interval(Some(Duration::from_secs(240)));

    let store = kad::store::MemoryStore::new(peer_id);
    let mut kademlia = kad::Behaviour::with_config(peer_id, store, kad_config);

    // Add other bootstrap peers to the routing table.
    let bootstrap_peers: Vec<Multiaddr> = parse_flag_multi(&args, "--peer")
        .unwrap_or_default();
    for peer_addr in &bootstrap_peers {
        // Extract PeerId from /p2p/ component.
        let remote_peer = peer_addr
            .iter()
            .find_map(|p| match p {
                libp2p::multiaddr::Protocol::P2p(id) => Some(id),
                _ => None,
            });
        if let Some(remote_id) = remote_peer {
            let base_addr: Multiaddr = peer_addr
                .iter()
                .filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
                .collect();
            kademlia.add_address(&remote_id, base_addr);
            info!(%remote_id, "added bootstrap peer");
        }
    }

    // Relay server — accepts reservations from NATted peers.
    let relay_server = relay::Behaviour::new(
        peer_id,
        relay::Config {
            max_reservations: 256,
            max_circuits: 512,
            max_circuits_per_peer: 8,
            reservation_duration: Duration::from_secs(3600),
            ..Default::default()
        },
    );

    // AutoNAT server — responds to NAT probes from peers.
    let autonat = autonat::Behaviour::new(
        peer_id,
        autonat::Config {
            boot_delay: Duration::from_secs(1),
            ..Default::default()
        },
    );

    // Identify.
    let identify = identify::Behaviour::new(
        identify::Config::new("openhydra/0.1.0".to_string(), keypair.public())
            .with_push_listen_addr_updates(true),
    );

    let behaviour = BootstrapBehaviour {
        kademlia,
        relay_server,
        autonat,
        identify,
    };

    let swarm_config = SwarmConfig::with_tokio_executor()
        .with_idle_connection_timeout(Duration::from_secs(600));

    let mut swarm = Swarm::new(transport, behaviour, peer_id, swarm_config);

    for addr in &listen_addrs {
        swarm.listen_on(addr.clone())?;
        info!(%addr, "listening");
    }

    info!("bootstrap node running — press Ctrl+C to stop");

    // Event loop.
    loop {
        match swarm.select_next_some().await {
            libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } => {
                info!(%address, "new listen address");
            }
            libp2p::swarm::SwarmEvent::Behaviour(event) => {
                match event {
                    BootstrapBehaviourEvent::Kademlia(kad::Event::RoutingUpdated {
                        peer, ..
                    }) => {
                        info!(%peer, "kademlia routing table updated");
                    }
                    BootstrapBehaviourEvent::RelayServer(
                        relay::Event::ReservationReqAccepted { src_peer_id, .. },
                    ) => {
                        info!(%src_peer_id, "relay reservation accepted");
                    }
                    BootstrapBehaviourEvent::RelayServer(
                        relay::Event::CircuitClosed { src_peer_id, dst_peer_id, .. },
                    ) => {
                        info!(%src_peer_id, %dst_peer_id, "relay circuit closed");
                    }
                    _ => {}
                }
            }
            libp2p::swarm::SwarmEvent::ConnectionEstablished { peer_id: p, .. } => {
                info!(%p, "connection established");
            }
            libp2p::swarm::SwarmEvent::ConnectionClosed { peer_id: p, .. } => {
                info!(%p, "connection closed");
            }
            _ => {}
        }
    }
}

// Minimal CLI flag parsing (no clap dependency to keep binary small).

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn parse_flag_multi(args: &[String], name: &str) -> Option<Vec<Multiaddr>> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == name {
            if let Some(val) = args.get(i + 1) {
                if let Ok(addr) = val.parse() {
                    result.push(addr);
                }
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}
