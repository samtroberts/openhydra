// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The peer allowlist (`fence::set_allowed_peers`, the agent's `--allow-peers-file`) end to end: three
//! nodes on loopback, A and B on the list, C not. B reaches A; C, which knows A's address and id,
//! can't. The allowlist is process-wide, so it lives in its own test binary.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use openhydra_network::handle::NetworkHandle;
use openhydra_network::identity::Identity;
use openhydra_network::node::NodeConfig;

fn free_port() -> u16 {
    loop {
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        if std::net::UdpSocket::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
}

fn node(dir: &std::path::Path, name: &str, port: u16, bootstrap: Option<String>) -> NetworkHandle {
    NetworkHandle::start(NodeConfig {
        identity_path: dir.join(format!("{name}.key")),
        listen_addrs: vec![format!("/ip4/127.0.0.1/tcp/{port}")],
        bootstrap_peers: bootstrap.into_iter().collect(),
        enable_peer_relay: false,
        enable_connection_reversal: false,
        pcp_gateway: None,
        public_relays: openhydra_network::relay::PublicRelays::Off,
    })
    .unwrap()
}

/// Answer every inbound request on `net` with "pong" until `until`.
fn answer(net: &NetworkHandle, until: Instant) {
    while Instant::now() < until {
        if let Some(item) = net.poll_inbound(Duration::from_millis(100)) {
            let _ = net.respond(item.0, b"pong".to_vec());
        }
    }
}

#[test]
fn a_peer_off_the_allowlist_cannot_reach_a_listed_one() {
    openhydra_network::node::disable_mdns();
    let dir = tempfile::tempdir().unwrap();
    let id = |n: &str| Identity::load_or_create(&dir.path().join(format!("{n}.key"))).unwrap().libp2p_peer_id;
    let (a_id, b_id) = (id("a"), id("b"));
    let _c_id = id("c");
    openhydra_network::fence::set_allowed_peers(HashSet::from([a_id, b_id])).unwrap();
    // A fenced node given no bootstrap isn't sent to OpenHydra's public ones (the allowlist would
    // refuse them): it doesn't count as on the public network.
    let unbootstrapped = NodeConfig { bootstrap_peers: Vec::new(), ..NodeConfig::default() };
    assert!(!openhydra_network::node::joins_public_network(&unbootstrapped));

    let a_port = free_port();
    let a = node(dir.path(), "a", a_port, None);
    let a_addr = format!("/ip4/127.0.0.1/tcp/{a_port}/p2p/{a_id}");
    let a_id = a_id.to_base58();
    let b = node(dir.path(), "b", free_port(), Some(a_addr.clone()));
    let c = node(dir.path(), "c", free_port(), Some(a_addr));

    std::thread::scope(|s| {
        s.spawn(|| answer(&a, Instant::now() + Duration::from_secs(40)));
        // B, on the list: answered (retry while it connects).
        let end = Instant::now() + Duration::from_secs(30);
        let mut b_reply = Err(String::new());
        while Instant::now() < end {
            b_reply = b.proxy_forward_timeout(a_id.clone(), b"ping".to_vec(), Duration::from_secs(5));
            if b_reply.is_ok() {
                break;
            }
            std::thread::sleep(Duration::from_millis(300));
        }
        assert_eq!(b_reply.as_deref(), Ok(&b"pong"[..]), "a listed peer gets through");
        // C, off the list: refused every time, though it knows A's address and id.
        for _ in 0..3 {
            let r = c.proxy_forward_timeout(a_id.clone(), b"ping".to_vec(), Duration::from_secs(5));
            assert!(r.is_err(), "an unlisted peer must not reach A: {r:?}");
        }
    });
}
