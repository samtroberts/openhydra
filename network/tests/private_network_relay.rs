// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! A private network's relay circuits, end to end on loopback. The relay is the real
//! `openhydra-bootstrap`, run with the network key (`--private-network-key-file`) and a peer allowlist
//! (`--allow-peers-file`). Two keyed nodes that can only reach each other through it (neither
//! listens on anything, so there is no direct path) talk over a circuit. A circuit is keyed end to
//! end as well as on each hop, so a bare libp2p node that reaches the relay with the key but doesn't
//! key the circuit itself can't use it to reach a keyed node, and one without the key can't reach
//! the relay at all; a matching control, keyed throughout, gets through. The relay also turns away a
//! keyed node that isn't on its allowlist.
//!
//! The key is process-wide (every node built here has it), so this lives in its own test binary,
//! and the keyless parties are bare libp2p swarms built here.
//!
//! Run:  cargo test -p openhydra-network --test private_network_relay

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use futures::StreamExt;
use libp2p::pnet::PreSharedKey;
use libp2p::{identity::Keypair, noise, ping, relay, swarm::SwarmEvent, tcp, yamux, Multiaddr, PeerId, Transport};
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

/// The relay (`openhydra-bootstrap`), killed when dropped.
struct Relay {
    child: Child,
    log: PathBuf,
}

impl Drop for Relay {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Relay {
    fn log(&self) -> String {
        std::fs::read_to_string(&self.log).unwrap_or_default()
    }
}

fn start_relay(dir: &Path, identity: &Path, key_file: &Path, allow_file: &Path, port: u16) -> Relay {
    let log = dir.join("relay.log");
    let out = std::fs::File::create(&log).unwrap();
    let tcp = format!("/ip4/127.0.0.1/tcp/{port}");
    let quic = format!("/ip4/127.0.0.1/udp/{port}/quic-v1");
    let child = Command::new(env!("CARGO_BIN_EXE_openhydra-bootstrap"))
        .args(["--identity", identity.to_str().unwrap(), "--listen", &tcp, "--listen", &quic, "--external", &tcp])
        .args(["--private-network-key-file", key_file.to_str().unwrap()])
        .args(["--allow-peers-file", allow_file.to_str().unwrap()])
        .env("RUST_LOG", "openhydra_bootstrap=debug,warn")
        .env("NO_COLOR", "1")
        .stdout(Stdio::from(out.try_clone().unwrap()))
        .stderr(Stdio::from(out))
        .spawn()
        .expect("start openhydra-bootstrap");
    Relay { child, log }
}

/// A keyed node with no listen address (so it can be reached only through the relay), whose only
/// bootstrap and relay is `relay`.
fn node(dir: &Path, name: &str, relay: &str) -> NetworkHandle {
    NetworkHandle::start(NodeConfig {
        identity_path: dir.join(format!("{name}.key")),
        listen_addrs: Vec::new(),
        bootstrap_peers: vec![relay.to_string()],
        enable_peer_relay: false,
        enable_connection_reversal: false,
        pcp_gateway: None,
        public_relays: openhydra_network::relay::PublicRelays::Off,
    })
    .unwrap()
}

fn wait_for(limit: Duration, mut done: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + limit;
    while Instant::now() < end {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    done()
}

/// `t`, keyed with `key` when given, then Noise and Yamux.
fn secure<T>(t: T, key: Option<PreSharedKey>, keypair: &Keypair) -> transport::BoxedTransport
where
    T: Transport + Send + Unpin + 'static,
    T::Output: futures::AsyncRead + futures::AsyncWrite + Send + Unpin + 'static,
    T::Error: std::error::Error + Send + Sync + 'static,
    T::Dial: Send + 'static,
    T::ListenerUpgrade: Send + 'static,
{
    t.and_then(move |s, _| async move {
        match key {
            Some(k) => libp2p::pnet::PnetConfig::new(k).handshake(s).await.map(futures::future::Either::Left),
            None => Ok(futures::future::Either::Right(s)),
        }
    })
    .map_err(|e| std::io::Error::other(e.to_string()))
    .upgrade(libp2p::core::upgrade::Version::V1Lazy)
    .authenticate(noise::Config::new(keypair).unwrap())
    .multiplex(yamux::Config::default())
    .map(|(p, m), _| (p, libp2p::core::muxing::StreamMuxerBox::new(m)))
    .boxed()
}

#[derive(libp2p::swarm::NetworkBehaviour)]
#[behaviour(prelude = "libp2p::swarm::derive_prelude")]
struct Outsider {
    relay: relay::client::Behaviour,
    ping: ping::Behaviour,
}

/// Whether a bare libp2p node with identity `keypair` gets a working connection (a ping answered)
/// to `target` through the circuit address `circuit`: its connection to the relay keyed with `hop`,
/// the circuit itself keyed with `end_to_end` (each `None`: not keyed).
fn outsider_reaches(keypair: Keypair, circuit: Multiaddr, target: PeerId, hop: Option<PreSharedKey>, end_to_end: Option<PreSharedKey>) -> bool {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let peer_id = keypair.public().to_peer_id();
        let (relay_transport, relay_client) = relay::client::new(peer_id);
        let circuits = secure(relay_transport, end_to_end, &keypair);
        let direct = secure(tcp::tokio::Transport::new(tcp::Config::default()), hop, &keypair);
        let transport = transport::either_transport(circuits, direct);
        let mut swarm = libp2p::Swarm::new(
            transport,
            Outsider { relay: relay_client, ping: ping::Behaviour::default() },
            peer_id,
            libp2p::swarm::Config::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(30)),
        );
        swarm.dial(circuit).unwrap();
        let deadline = tokio::time::sleep(Duration::from_secs(20));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                _ = &mut deadline => return false,
                ev = swarm.select_next_some() => match ev {
                    // The target answered a ping over the circuit: a real connection to it.
                    SwarmEvent::Behaviour(OutsiderEvent::Ping(ping::Event { peer, result: Ok(_), .. })) if peer == target => return true,
                    SwarmEvent::OutgoingConnectionError { peer_id: Some(p), .. } if p == target => return false,
                    _ => {}
                },
            }
        }
    })
}

/// Write `text` to `path`, readable by its owner only (the relay refuses a key file others can read).
fn write_private(path: &Path, text: &str) {
    std::fs::write(path, text).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
}

#[test]
fn relay_circuits_on_a_private_network_need_the_key_end_to_end() {
    openhydra_network::node::disable_mdns();
    let dir = tempfile::tempdir().unwrap();
    let key_text = transport::new_private_network_key_file();
    let key = transport::parse_private_network_key(&key_text).unwrap();
    transport::set_private_network_key(key).unwrap();
    let key_file = dir.path().join("swarm.key");
    write_private(&key_file, &key_text);

    let id = |n: &str| Identity::load_or_create(&dir.path().join(format!("{n}.key"))).unwrap().libp2p_peer_id;
    let (relay_id, a_id, b_id) = (id("relay"), id("a"), id("b"));
    // Bare outsiders: one on the relay's allowlist, one not.
    let listed = Keypair::generate_ed25519();
    let unlisted = Keypair::generate_ed25519();
    let allow_file = dir.path().join("peers.txt");
    std::fs::write(&allow_file, format!("{a_id}\n{b_id}\n{}  # the bare outsider\n", listed.public().to_peer_id())).unwrap();

    let port = free_port();
    let relay = start_relay(dir.path(), &dir.path().join("relay.key"), &key_file, &allow_file, port);
    let relay_addr = format!("/ip4/127.0.0.1/tcp/{port}/p2p/{relay_id}");

    let a = node(dir.path(), "a", &relay_addr);
    assert!(
        wait_for(Duration::from_secs(60), || a.status().map(|s| !s.relay_reservations.is_empty()).unwrap_or(false)),
        "A never reserved on the keyed relay (relay log:\n{})",
        relay.log()
    );
    let b = node(dir.path(), "b", &relay_addr);

    let done = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| {
            let until = Instant::now() + Duration::from_secs(180);
            while Instant::now() < until && !done.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(item) = a.poll_inbound(Duration::from_millis(100)) {
                    let _ = a.respond(item.0, b"pong".to_vec());
                }
            }
        });
        // B, keyed, reaches A through the relay (retry while it connects).
        let end = Instant::now() + Duration::from_secs(60);
        let mut reply = Err(String::new());
        while Instant::now() < end {
            reply = b.proxy_forward_timeout(a_id.to_base58(), b"ping".to_vec(), Duration::from_secs(5));
            if reply.is_ok() {
                break;
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        assert_eq!(reply.as_deref(), Ok(&b"pong"[..]), "a keyed node gets through the keyed relay (relay log:\n{})", relay.log());
        let b_at_a = a.status().unwrap().peers.into_iter().find(|p| p.peer_id == b_id.to_base58()).expect("A sees B");
        assert!(b_at_a.tcp_relay > 0 && b_at_a.tcp_direct == 0, "the connection is a circuit: {b_at_a:?}");

        let circuit: Multiaddr = format!("{relay_addr}/p2p-circuit/p2p/{a_id}").parse().unwrap();
        // The control: keyed on the hop and the circuit, the outsider's own setup gets through.
        assert!(outsider_reaches(listed.clone(), circuit.clone(), a_id, Some(key), Some(key)), "keyed throughout: reaches A");
        // Keyed to the relay, but the circuit not: the relay carries it, A refuses it.
        assert!(!outsider_reaches(listed.clone(), circuit.clone(), a_id, Some(key), None), "an unkeyed circuit must not reach A");
        // Keyed to the relay, the circuit with another key.
        let other = transport::parse_private_network_key(&transport::new_private_network_key_file()).unwrap();
        assert!(!outsider_reaches(listed.clone(), circuit.clone(), a_id, Some(key), Some(other)), "a circuit with another key must not");
        // No key at all: not even the relay.
        assert!(!outsider_reaches(listed, circuit.clone(), a_id, None, None), "a keyless node must not reach the relay");
        // Keyed throughout but off the relay's allowlist: the relay refuses it.
        assert!(!outsider_reaches(unlisted, circuit, a_id, Some(key), Some(key)), "the relay's allowlist applies");
        done.store(true, std::sync::atomic::Ordering::Relaxed);
    });

    // The relay ran keyed, said so by fingerprint only, and left QUIC off.
    let log = relay.log();
    assert!(log.contains(&transport::private_network_fingerprint(&key)), "the relay logs the key's fingerprint");
    assert!(!log.contains(key_text.lines().nth(2).unwrap()), "never the key itself");
    assert!(std::net::UdpSocket::bind(("127.0.0.1", port)).is_ok(), "a keyed relay must not listen on QUIC");
}
