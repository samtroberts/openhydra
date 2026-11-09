// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! A private network (`transport::set_private_network_key`, the agent's
//! `--private-network-key-file`) end to end on loopback: two nodes holding the key reach each other;
//! a plain libp2p node without the key, and one with a different key, can't connect to them at all.
//! The key is process-wide, so it lives in its own test binary, and the outsiders are bare libp2p
//! swarms built here.

use std::time::{Duration, Instant};

use futures::StreamExt;
use libp2p::{noise, ping, swarm::SwarmEvent, tcp, yamux, Multiaddr, Transport};
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

/// Whether a bare libp2p node (TCP, Noise, Yamux, ping) can connect to `addr`: with no network key,
/// or with `key`.
fn outsider_connects(addr: Multiaddr, key: Option<libp2p::pnet::PreSharedKey>) -> bool {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();
        let tcp = tcp::tokio::Transport::new(tcp::Config::default());
        let transport = match key {
            Some(k) => secure(tcp.and_then(move |s, _| libp2p::pnet::PnetConfig::new(k).handshake(s)), &keypair),
            None => secure(tcp, &keypair),
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
                    // Connected, and the far side answered a ping over it: a real connection.
                    SwarmEvent::Behaviour(ping::Event { result: Ok(_), .. }) => return true,
                    SwarmEvent::OutgoingConnectionError { .. } => return false,
                    SwarmEvent::ConnectionClosed { .. } => return false,
                    _ => {}
                },
            }
        }
    })
}

/// `t` with Noise and Yamux on top.
fn secure<T>(t: T, keypair: &libp2p::identity::Keypair) -> transport::BoxedTransport
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

#[test]
fn only_nodes_with_the_network_key_can_connect() {
    openhydra_network::node::disable_mdns();
    let file = transport::new_private_network_key_file();
    let key = transport::parse_private_network_key(&file).unwrap();
    transport::set_private_network_key(key).unwrap();
    assert!(transport::set_private_network_key(key).is_err(), "set once");
    assert!(!transport::private_network_fingerprint(&key).contains(file.lines().nth(2).unwrap()));

    let dir = tempfile::tempdir().unwrap();
    let a_id = Identity::load_or_create(&dir.path().join("a.key")).unwrap().libp2p_peer_id;
    let a_port = free_port();
    // Asked to listen on QUIC as well: a private node must not (QUIC can't carry the key).
    let a = node(
        dir.path(),
        "a",
        vec![format!("/ip4/127.0.0.1/tcp/{a_port}"), format!("/ip4/127.0.0.1/udp/{a_port}/quic-v1")],
        None,
    );
    let a_addr = format!("/ip4/127.0.0.1/tcp/{a_port}/p2p/{a_id}");
    // A node is running: the network settings can't change under it any more.
    assert!(openhydra_network::fence::set_allowed_peers(std::collections::HashSet::new()).is_err(), "too late for an allowlist");
    let b = node(dir.path(), "b", vec![format!("/ip4/127.0.0.1/tcp/{}", free_port())], Some(a_addr.clone()));
    let a_id = a_id.to_base58();

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
        // B holds the key: answered (retry while it connects).
        let end = Instant::now() + Duration::from_secs(30);
        let mut reply = Err(String::new());
        while Instant::now() < end {
            reply = b.proxy_forward_timeout(a_id.clone(), b"ping".to_vec(), Duration::from_secs(5));
            if reply.is_ok() {
                break;
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        assert_eq!(reply.as_deref(), Ok(&b"pong"[..]), "a node with the key gets through");

        let tcp_addr: Multiaddr = format!("/ip4/127.0.0.1/tcp/{a_port}").parse().unwrap();
        // The control: the outsider's own setup works with the right key.
        assert!(outsider_connects(tcp_addr.clone(), Some(key)), "a bare node with the key connects");
        assert!(!outsider_connects(tcp_addr.clone(), None), "a node without the key must not connect");
        let other = transport::parse_private_network_key(&transport::new_private_network_key_file()).unwrap();
        assert!(!outsider_connects(tcp_addr, Some(other)), "a node with another key must not connect");
        // Nothing listens on A's QUIC port: it is free to bind.
        assert!(std::net::UdpSocket::bind(("127.0.0.1", a_port)).is_ok(), "a private node must not listen on QUIC");
        done.store(true, std::sync::atomic::Ordering::Relaxed);
    });
}
