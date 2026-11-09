// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! QUIC on a private network (`transport::allow_quic_on_private_network`, the agent's
//! `--private-network-quic`), end to end on loopback. QUIC can't carry the network key, so it is
//! guarded by the peer allowlist: over QUIC a listed peer gets in and an unlisted one is refused;
//! over TCP the key is still required, listed or not. The key, the allowlist and the QUIC switch are
//! process-wide, so this lives in its own test binary; the outsiders are bare libp2p swarms built
//! here, so each refusal has a matching control that connects.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use futures::StreamExt;
use libp2p::{identity::Keypair, noise, ping, swarm::SwarmEvent, tcp, yamux, Multiaddr, Transport};
use openhydra_network::handle::NetworkHandle;
use openhydra_network::identity::Identity;
use openhydra_network::node::NodeConfig;
use openhydra_network::transport;

fn free_port() -> u16 {
    loop {
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        if std::net::UdpSocket::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
}

fn node(dir: &std::path::Path, name: &str, listen: Vec<String>, bootstrap: Option<String>) -> NetworkHandle {
    NetworkHandle::start(NodeConfig {
        identity_path: dir.join(format!("{name}.key")),
        listen_addrs: listen,
        bootstrap_peers: bootstrap.into_iter().collect(),
        enable_peer_relay: false,
        enable_connection_reversal: false,
        pcp_gateway: None,
        public_relays: openhydra_network::relay::PublicRelays::Off,
    })
    .unwrap()
}

/// `t` with Noise and Yamux on top.
fn secure<T>(t: T, keypair: &Keypair) -> transport::BoxedTransport
where
    T: Transport + Send + Unpin + 'static,
    T::Output: futures::AsyncRead + futures::AsyncWrite + Send + Unpin + 'static,
    T::Error: std::error::Error + Send + Sync + 'static,
    T::Dial: Send + 'static,
    T::ListenerUpgrade: Send + 'static,
{
    t.map_err(|e| std::io::Error::other(e.to_string()))
        .upgrade(libp2p::core::upgrade::Version::V1Lazy)
        .authenticate(noise::Config::new(keypair).unwrap())
        .multiplex(yamux::Config::default())
        .map(|(p, m), _| (p, libp2p::core::muxing::StreamMuxerBox::new(m)))
        .boxed()
}

/// How a bare outsider dials.
enum Via {
    Quic,
    Tcp(Option<libp2p::pnet::PreSharedKey>),
}

/// Whether a bare libp2p node with identity `keypair` gets a working connection (a ping answered)
/// to `addr`.
fn outsider_connects(keypair: Keypair, addr: Multiaddr, via: Via) -> bool {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let peer_id = keypair.public().to_peer_id();
        let transport = match via {
            Via::Quic => libp2p::quic::tokio::Transport::new(libp2p::quic::Config::new(&keypair))
                .map(|(p, c), _| (p, libp2p::core::muxing::StreamMuxerBox::new(c)))
                .boxed(),
            Via::Tcp(Some(k)) => secure(
                tcp::tokio::Transport::new(tcp::Config::default()).and_then(move |s, _| libp2p::pnet::PnetConfig::new(k).handshake(s)),
                &keypair,
            ),
            Via::Tcp(None) => secure(tcp::tokio::Transport::new(tcp::Config::default()), &keypair),
        };
        let mut swarm = libp2p::Swarm::new(
            transport,
            ping::Behaviour::default(),
            peer_id,
            libp2p::swarm::Config::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(30)),
        );
        swarm.dial(addr).unwrap();
        let deadline = tokio::time::sleep(Duration::from_secs(20));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                _ = &mut deadline => return false,
                ev = swarm.select_next_some() => match ev {
                    SwarmEvent::Behaviour(ping::Event { result: Ok(_), .. }) => return true,
                    SwarmEvent::OutgoingConnectionError { .. } | SwarmEvent::ConnectionClosed { .. } => return false,
                    _ => {}
                },
            }
        }
    })
}

#[test]
fn quic_on_a_private_network_admits_only_listed_peers_and_tcp_still_needs_the_key() {
    openhydra_network::node::disable_mdns();
    let dir = tempfile::tempdir().unwrap();
    let id = |n: &str| Identity::load_or_create(&dir.path().join(format!("{n}.key"))).unwrap().libp2p_peer_id;
    let (a_id, b_id) = (id("a"), id("b"));
    // Bare outsiders: one listed, one not.
    let listed = Keypair::generate_ed25519();
    let unlisted = Keypair::generate_ed25519();

    let key = transport::parse_private_network_key(&transport::new_private_network_key_file()).unwrap();
    transport::set_private_network_key(key).unwrap();
    // QUIC on a private network is refused without an allowlist...
    assert!(transport::allow_quic_on_private_network().is_err(), "needs an allowlist");
    assert!(!transport::quic_enabled(), "a private network without the switch has no QUIC");
    openhydra_network::fence::set_allowed_peers(HashSet::from([a_id, b_id, listed.public().to_peer_id()])).unwrap();
    assert!(!transport::quic_enabled(), "an allowlist alone doesn't turn QUIC on");
    // ...and allowed with one.
    transport::allow_quic_on_private_network().unwrap();
    assert!(transport::quic_enabled());

    let a_port = free_port();
    let a = node(
        dir.path(),
        "a",
        vec![format!("/ip4/127.0.0.1/tcp/{a_port}"), format!("/ip4/127.0.0.1/udp/{a_port}/quic-v1")],
        None,
    );
    // B knows A only by its QUIC address and listens on QUIC only, so their connection is QUIC.
    let quic_addr: Multiaddr = format!("/ip4/127.0.0.1/udp/{a_port}/quic-v1").parse().unwrap();
    let b = node(
        dir.path(),
        "b",
        vec![format!("/ip4/127.0.0.1/udp/{}/quic-v1", free_port())],
        Some(format!("{quic_addr}/p2p/{a_id}")),
    );
    let tcp_addr: Multiaddr = format!("/ip4/127.0.0.1/tcp/{a_port}").parse().unwrap();

    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            let until = Instant::now() + Duration::from_secs(120);
            while Instant::now() < until && !done.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(item) = a.poll_inbound(Duration::from_millis(100)) {
                    let _ = a.respond(item.0, b"pong".to_vec());
                }
            }
        });
        // A listed node with the key, over QUIC: answered.
        let end = Instant::now() + Duration::from_secs(30);
        let mut reply = Err(String::new());
        while Instant::now() < end {
            reply = b.proxy_forward_timeout(a_id.to_base58(), b"ping".to_vec(), Duration::from_secs(5));
            if reply.is_ok() {
                break;
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        assert_eq!(reply.as_deref(), Ok(&b"pong"[..]), "a listed peer gets through over QUIC");
        let b_at_a = a.status().unwrap().peers.into_iter().find(|p| p.peer_id == b_id.to_base58()).expect("A sees B");
        assert!(b_at_a.quic_direct_v4 > 0 && b_at_a.tcp_direct == 0, "the connection is QUIC: {b_at_a:?}");

        // QUIC: the allowlist decides (no key is involved).
        assert!(outsider_connects(listed.clone(), quic_addr.clone(), Via::Quic), "a listed peer connects over QUIC");
        assert!(!outsider_connects(unlisted.clone(), quic_addr.clone(), Via::Quic), "an unlisted peer must not, over QUIC");
        // TCP: the key is still required, even of a listed peer; and the allowlist still applies.
        assert!(outsider_connects(listed.clone(), tcp_addr.clone(), Via::Tcp(Some(key))), "listed, with the key, over TCP");
        assert!(!outsider_connects(listed.clone(), tcp_addr.clone(), Via::Tcp(None)), "listed but keyless must not, over TCP");
        assert!(!outsider_connects(unlisted.clone(), tcp_addr.clone(), Via::Tcp(Some(key))), "keyed but unlisted must not");
        let other = transport::parse_private_network_key(&transport::new_private_network_key_file()).unwrap();
        assert!(!outsider_connects(listed, tcp_addr, Via::Tcp(Some(other))), "listed with another key must not");
        done.store(true, std::sync::atomic::Ordering::Relaxed);
    });
}
