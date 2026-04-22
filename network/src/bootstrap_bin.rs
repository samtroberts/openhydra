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
use libp2p::{autonat, gossipsub, identify, kad, relay, Multiaddr, Swarm, Transport};
use tracing::info;

// Re-use crate modules for identity and transport.
// Note: since this is a [[bin]], we import the library crate.
use openhydra_network::identity::Identity;

/// Bootstrap-specific behaviour — includes relay::Behaviour (server mode)
/// and DCUtR for advertising hole-punch support in Identify.
#[derive(libp2p::swarm::NetworkBehaviour)]
struct BootstrapBehaviour {
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
    relay_server: relay::Behaviour,
    autonat: autonat::Behaviour,
    identify: identify::Behaviour,
    dcutr: libp2p::dcutr::Behaviour,
    /// Gossipsub (B1 rendezvous support).
    ///
    /// Bootstrap nodes subscribe to the same ``openhydra/swarm/v1/events``
    /// topic as peers so they can **forward** ``REQUEST_HOLE_PUNCH`` /
    /// ``PEER_DEAD`` messages between peers that don't have a direct
    /// libp2p connection to each other — the common case for two
    /// NATted peers whose only shared connection point is a Linode
    /// relay. Without this, peer A's publish never reaches peer B
    /// because neither is connected to anyone who'll forward the
    /// topic message.
    gossipsub: gossipsub::Behaviour,
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

    // Transport: QUIC + TCP (bootstrap is public, no relay needed)
    let tcp_transport = openhydra_network::transport::build_tcp_transport(&keypair)?;
    let quic_transport = openhydra_network::transport::build_quic_transport(&keypair)?;
    let transport = libp2p::core::transport::OrTransport::new(quic_transport, tcp_transport)
        .map(|either, _| match either {
            futures::future::Either::Left((pid, mux)) => (pid, libp2p::core::muxing::StreamMuxerBox::new(mux)),
            futures::future::Either::Right((pid, mux)) => (pid, libp2p::core::muxing::StreamMuxerBox::new(mux)),
        })
        .boxed();

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
            // 10 MB per circuit — activation tensors can be several MB.
            max_circuit_bytes: 10 * 1024 * 1024,
            // 10 minutes per circuit — autoregressive decode can take a while.
            max_circuit_duration: Duration::from_secs(600),
            ..Default::default()
        },
    );

    // AutoNAT reporter — responds to NAT probes from peers.
    //
    // A3 DCUtR fix: configure this node to act as an authoritative
    // reporter for every peer that probes it, including peers whose
    // candidate external addrs fall in LAN / ULA space (``only_global_ips
    // = false``). Without this, a peer behind NAT that registered its LAN
    // IP as an external candidate (PR A3 event_loop.rs change) would
    // never get a Falsified verdict — the bootstrap would silently skip
    // the probe, leaving AutoNAT in ``Unknown`` forever and DCUtR
    // dormant.
    //
    // Throttle limits are relaxed above the libp2p defaults so a steady
    // swarm of a few dozen peers probing at the same time doesn't get
    // rate-limited out. Each probe is cheap (a single TCP dial) so the
    // bootstrap can comfortably serve ~500 req/min.
    let autonat = autonat::Behaviour::new(
        peer_id,
        autonat::Config {
            boot_delay: Duration::from_secs(1),
            only_global_ips: false,
            throttle_clients_global_max: 128,
            throttle_clients_peer_max: 8,
            throttle_clients_period: Duration::from_secs(1),
            ..Default::default()
        },
    );

    // Identify.
    let identify = identify::Behaviour::new(
        identify::Config::new("openhydra/0.1.0".to_string(), keypair.public())
            .with_push_listen_addr_updates(true),
    );

    // DCUtR — advertises hole-punch support in Identify protocol list.
    let dcutr = libp2p::dcutr::Behaviour::new(peer_id);

    // Gossipsub forwarder — bootstraps subscribe to the swarm-wide topic so
    // peers that can only reach each other through a bootstrap still see
    // each other's events. The message-authenticity signing guarantees the
    // bootstrap can't forge events; it just propagates signed messages.
    // Small-swarm tuning identical to the peer side — critical so the
    // bootstrap forwards every published message to every topic peer,
    // not just the D-sized mesh slice.
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(1))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .max_transmit_size(64 * 1024)
        .flood_publish(true)
        .mesh_outbound_min(1)
        .mesh_n_low(1)
        .mesh_n(3)
        .mesh_n_high(6)
        .build()
        .map_err(|e| format!("gossipsub config: {e}"))?;
    let mut gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(keypair.clone()),
        gossipsub_config,
    )
    .map_err(|e| format!("gossipsub behaviour: {e}"))?;
    let gossip_topic =
        gossipsub::IdentTopic::new(openhydra_network::swarm::GOSSIPSUB_TOPIC);
    gossipsub
        .subscribe(&gossip_topic)
        .map_err(|e| format!("gossipsub subscribe: {e:?}"))?;

    let behaviour = BootstrapBehaviour {
        kademlia,
        relay_server,
        autonat,
        identify,
        dcutr,
        gossipsub,
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
