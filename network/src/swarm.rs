//! Swarm creation and configuration.

use std::time::Duration;

use libp2p::core::muxing::StreamMuxerBox;
use libp2p::swarm::Config as SwarmConfig;
use libp2p::{
    autonat, dcutr, gossipsub, identify, kad, mdns, relay, Multiaddr, PeerId, Swarm, Transport,
};

use crate::behaviour::OpenHydraBehaviour;

/// PR-3: the single topic that carries all swarm-wide events. Keeping one
/// topic for v1 intentionally bounds the blast radius — future versions can
/// add per-model topics once the coordinator has logic to subscribe /
/// unsubscribe as models come and go.
pub const GOSSIPSUB_TOPIC: &str = "openhydra/swarm/v1/events";
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
                "/ip4/0.0.0.0/udp/4001/quic-v1".parse().unwrap(),
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
    let tcp_transport = crate::transport::build_tcp_transport(&keypair)?;

    // Transport: QUIC (built-in TLS 1.3 + multiplexing, UDP-based)
    let quic_transport = crate::transport::build_quic_transport(&keypair)?;

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

    // 3-way transport composition: relay → QUIC → TCP (fallback order).
    // Relay first: handles /p2p-circuit multiaddrs.
    // QUIC second: handles /udp/.../quic-v1 addresses (faster, UDP hole punch).
    // TCP last: fallback for /tcp/ addresses.
    let quic_tcp = libp2p::core::transport::OrTransport::new(quic_transport, tcp_transport)
        .map(|either, _| match either {
            futures::future::Either::Left((pid, mux)) => (pid, StreamMuxerBox::new(mux)),
            futures::future::Either::Right((pid, mux)) => (pid, StreamMuxerBox::new(mux)),
        });

    let combined_transport = libp2p::core::transport::OrTransport::new(
        relay_upgraded,
        quic_tcp,
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
    //
    // A3 DCUtR fix: tune the client config so probes fire eagerly enough
    // that the peer reaches a confident Private/Public verdict *before*
    // the first DCUtR hole-punch attempt. Previously ``refresh_interval``
    // was 60 s and ``use_connected`` defaulted to ``true`` but a few
    // fields mattered:
    //
    // * ``use_connected = true`` — reuse already-established bootstrap
    //   connections for probes instead of dialing fresh TCP sockets
    //   (cheaper, and more reliable against aggressive NATs).
    // * ``only_global_ips = false`` — allow AutoNAT to probe LAN / ULA
    //   candidates as well as globally-routable ones. We register a
    //   bunch of LAN candidates in the ``NewListenAddr`` handler (the
    //   direct-listen fix above), and we want AutoNAT to falsify them
    //   quickly rather than silently ignore them.
    // * ``confidence_max = 3`` — match libp2p's default; three agreeing
    //   probes needed before the verdict latches (stability vs. speed).
    let autonat = autonat::Behaviour::new(
        peer_id,
        autonat::Config {
            boot_delay: Duration::from_secs(5),
            refresh_interval: Duration::from_secs(60),
            retry_interval: Duration::from_secs(30),
            use_connected: true,
            only_global_ips: false,
            confidence_max: 3,
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

    // PR-3 (B1) — Gossipsub over a single topic, signed with our Ed25519
    // identity so recipients can verify the message came from a real swarm
    // member rather than a spoofed peer.
    // Small-swarm reliability tuning (B1 follow-up). With only a handful
    // of peers + 3 bootstraps, the default gossipsub mesh targets
    // (D_lo=4, D_hi=12) aren't reachable, so messages get dropped before
    // they cross the mesh. ``flood_publish(true)`` instructs libp2p to
    // send a published message to *every* known peer of the topic, not
    // just the D-sized mesh slice — trading bandwidth for reliability.
    // At v1 message volumes (control-plane events, not activations) this
    // is cheap.
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(1))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .max_transmit_size(64 * 1024)
        .flood_publish(true)
        // Ensure small meshes still exist; libp2p defaults assume large
        // swarms. The invariant the validator enforces is
        // ``mesh_outbound_min <= mesh_n_low <= mesh_n <= mesh_n_high``
        // — so we must lower ``mesh_outbound_min`` too (default is 2)
        // otherwise a 2-peer topology fails config validation.
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

    // Subscribe immediately so we start participating in the mesh as soon
    // as we have connected peers. Subscription is idempotent; publishing
    // works whether or not we're subscribed to the topic.
    let topic = gossipsub::IdentTopic::new(GOSSIPSUB_TOPIC);
    gossipsub
        .subscribe(&topic)
        .map_err(|e| format!("gossipsub subscribe: {e:?}"))?;

    let behaviour = OpenHydraBehaviour {
        kademlia,
        relay_client,
        dcutr,
        autonat,
        identify,
        mdns,
        grpc_proxy,
        gossipsub,
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
