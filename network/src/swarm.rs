//! Swarm creation and configuration.

use std::time::Duration;

use libp2p::core::muxing::StreamMuxerBox;
use libp2p::swarm::Config as SwarmConfig;
use libp2p::{autonat, dcutr, identify, kad, mdns, relay, Multiaddr, PeerId, Swarm, Transport};

use crate::behaviour::OpenHydraBehaviour;
use crate::identity::Identity;

/// Configuration for creating a new OpenHydra swarm.
pub struct SwarmOptions {
    /// Addresses to listen on (e.g. "/ip4/0.0.0.0/tcp/4001").
    pub listen_addrs: Vec<Multiaddr>,
    /// Bootstrap peer multiaddrs for Kademlia.
    pub bootstrap_peers: Vec<(PeerId, Multiaddr)>,
    /// Protocol version string for Identify.
    pub protocol_version: String,
}

impl Default for SwarmOptions {
    fn default() -> Self {
        Self {
            listen_addrs: vec![
                "/ip4/0.0.0.0/tcp/4001".parse().unwrap(),
            ],
            bootstrap_peers: Vec::new(),
            protocol_version: "openhydra/0.1.0".to_string(),
        }
    }
}

/// Build and configure a libp2p Swarm with the OpenHydra behaviour.
pub fn build_swarm(
    identity: &Identity,
    opts: SwarmOptions,
) -> Result<Swarm<OpenHydraBehaviour>, Box<dyn std::error::Error>> {
    let peer_id = identity.libp2p_peer_id;
    let keypair = identity.keypair.clone();

    // Transport: TCP + Noise + Yamux
    let tcp_transport = crate::transport::build_transport(&keypair)?;

    // Relay client — returns a (Transport, Behaviour) pair.
    // The Transport MUST be combined with the base transport and kept alive;
    // dropping it panics the relay client behaviour.
    let (relay_transport, relay_client) = relay::client::new(peer_id);

    // The relay transport outputs `relay::client::Connection` which is an
    // AsyncRead+AsyncWrite stream. Upgrade it with Noise+Yamux to get the
    // same `(PeerId, StreamMuxerBox)` output type as the TCP transport.
    let relay_upgraded = relay_transport
        .upgrade(libp2p::core::upgrade::Version::V1Lazy)
        .authenticate(
            libp2p::noise::Config::new(&keypair).expect("noise config for relay"),
        )
        .multiplex(libp2p::yamux::Config::default())
        .boxed();

    // Combine TCP with relay transport: try TCP first, fall back to relay.
    let combined_transport = libp2p::core::transport::OrTransport::new(
        tcp_transport,
        relay_upgraded,
    )
    .map(|either_output, _| match either_output {
        futures::future::Either::Left((peer_id, muxer)) => (peer_id, StreamMuxerBox::new(muxer)),
        futures::future::Either::Right((peer_id, muxer)) => (peer_id, StreamMuxerBox::new(muxer)),
    })
    .boxed();

    // Kademlia configuration
    let mut kad_config = kad::Config::new(
        libp2p::StreamProtocol::new("/openhydra/kad/1.0.0"),
    );
    kad_config.set_query_timeout(Duration::from_secs(10));
    kad_config.set_record_ttl(Some(Duration::from_secs(300)));
    kad_config.set_provider_record_ttl(Some(Duration::from_secs(300)));
    kad_config.set_publication_interval(Some(Duration::from_secs(120)));

    let store = kad::store::MemoryStore::new(peer_id);
    let mut kademlia = kad::Behaviour::with_config(peer_id, store, kad_config);

    // Add bootstrap peers to Kademlia routing table.
    for (peer, addr) in &opts.bootstrap_peers {
        kademlia.add_address(peer, addr.clone());
    }

    // DCUtR (hole punching).
    let dcutr = dcutr::Behaviour::new(peer_id);

    // AutoNAT (NAT type detection).
    let autonat = autonat::Behaviour::new(
        peer_id,
        autonat::Config {
            boot_delay: Duration::from_secs(5),
            refresh_interval: Duration::from_secs(60),
            retry_interval: Duration::from_secs(30),
            ..Default::default()
        },
    );

    // Identify (peer metadata exchange).
    let identify = identify::Behaviour::new(
        identify::Config::new(opts.protocol_version.clone(), keypair.public())
            .with_push_listen_addr_updates(true),
    );

    // mDNS (LAN discovery).
    let mdns = mdns::tokio::Behaviour::new(
        mdns::Config::default(),
        peer_id,
    )?;

    // gRPC proxy (cross-ISP tunneling through relay).
    let grpc_proxy = crate::proxy::proxy_behaviour();

    let behaviour = OpenHydraBehaviour {
        kademlia,
        relay_client,
        dcutr,
        autonat,
        identify,
        mdns,
        grpc_proxy,
    };

    let swarm_config = SwarmConfig::with_tokio_executor()
        .with_idle_connection_timeout(Duration::from_secs(300));

    let mut swarm = Swarm::new(combined_transport, behaviour, peer_id, swarm_config);

    // Listen on configured addresses.
    for addr in &opts.listen_addrs {
        swarm.listen_on(addr.clone())?;
    }

    Ok(swarm)
}
