// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! A gateway reaches its own provider on this machine by a pinned loopback address
//! (`NetworkHandle::pin_peer_addrs`, the agent's `serve --self-provider-addr`), with no LAN discovery,
//! no DHT entry and no relay between them: what the desktop app needs when LAN discovery is off.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use openhydra_network::handle::NetworkHandle;
use openhydra_network::identity::Identity;
use openhydra_network::node::NodeConfig;

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn node(dir: &std::path::Path, name: &str, port: u16, bootstrap: Option<String>) -> NetworkHandle {
    openhydra_network::node::disable_mdns();
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

/// Sets the flag when dropped, so a failed assertion still stops the responder thread (else the
/// scope would wait for it forever).
struct StopOnDrop<'a>(&'a AtomicBool);
impl Drop for StopOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

/// Answer every request to `p` with "pong" until `done`.
fn respond_until(p: &NetworkHandle, done: &AtomicBool) {
    while !done.load(Ordering::Relaxed) {
        if let Some(item) = p.poll_inbound(Duration::from_millis(50)) {
            let _ = p.respond(item.0, b"pong".to_vec());
        }
    }
}

fn ask(g: &NetworkHandle, p: &str) -> Result<Vec<u8>, String> {
    g.proxy_forward_timeout(p.to_string(), b"ping".to_vec(), Duration::from_secs(5))
}

/// Ask until answered or `within` runs out; returns how long it took.
fn ask_until_answered(g: &NetworkHandle, p: &str, within: Duration) -> Option<Duration> {
    let start = Instant::now();
    while start.elapsed() < within {
        if ask(g, p).as_deref() == Ok(&b"pong"[..]) {
            return Some(start.elapsed());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    None
}

#[test]
fn the_gateway_reaches_its_own_provider_by_the_pinned_address_alone() {
    let dir = tempfile::tempdir().unwrap();
    // The provider: no bootstrap, nobody knows its address.
    let p_port = free_port();
    let p = node(dir.path(), "p", p_port, None);
    let p_id = p.libp2p_peer_id().to_string();
    // The gateway's bootstrap is an unrelated node that has never heard of the provider.
    let b_port = free_port();
    let b = node(dir.path(), "b", b_port, None);
    let g = node(dir.path(), "g", free_port(), Some(format!("/ip4/127.0.0.1/tcp/{b_port}/p2p/{}", b.libp2p_peer_id())));

    let done = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| respond_until(&p, &done));
        let _stop = StopOnDrop(&done);
        // The control: without the pin there is no way to it (today's gap).
        assert!(ask(&g, &p_id).is_err(), "without a pinned address the provider must be unreachable here");

        // Refused before anything is sent: not loopback, a host name, another peer's id.
        let other = b.libp2p_peer_id();
        for bad in [
            format!("/ip4/192.168.1.5/tcp/{p_port}"),
            format!("/dns4/localhost/tcp/{p_port}"),
            format!("/ip4/127.0.0.1/tcp/{p_port}/p2p/{other}"),
        ] {
            assert!(g.pin_peer_addrs(&p_id, &[bad.clone()]).is_err(), "{bad} must be refused");
        }

        g.pin_peer_addrs(&p_id, &[format!("/ip4/127.0.0.1/tcp/{p_port}/p2p/{p_id}")]).unwrap();
        let took = ask_until_answered(&g, &p_id, Duration::from_secs(20)).expect("pinned: answered");
        assert!(took < Duration::from_secs(5), "the pinned dial should be immediate, took {took:?}");

        // Unpinned again: the gateway keeps the live connection, so this only checks the call works.
        g.pin_peer_addrs(&p_id, &[]).unwrap();
    });
}

#[test]
fn a_provider_that_starts_after_the_pin_is_reached_at_once_with_no_back_off() {
    let dir = tempfile::tempdir().unwrap();
    let p_port = free_port();
    let p_id = Identity::load_or_create(&dir.path().join("late.key")).unwrap().libp2p_peer_id.to_base58();
    let g = node(dir.path(), "g2", free_port(), None);
    g.pin_peer_addrs(&p_id, &[format!("/ip4/127.0.0.1/tcp/{p_port}")]).unwrap();

    // Not running yet: each request fails, and fails fast (a refused loopback dial). Spaced so that
    // each one would be a real dial under the relayed-dial back-off too, which would then have grown
    // to 2, 4, then 8 s: the provider starting now would stay unreachable for up to 8 s more.
    for wait in [0, 2100, 4100] {
        std::thread::sleep(Duration::from_millis(wait));
        let t = Instant::now();
        assert!(ask(&g, &p_id).is_err());
        assert!(t.elapsed() < Duration::from_secs(4), "a refused loopback dial fails at once, took {:?}", t.elapsed());
    }

    let p = node(dir.path(), "late", p_port, None);
    let done = AtomicBool::new(false);
    std::thread::scope(|s| {
        s.spawn(|| respond_until(&p, &done));
        let _stop = StopOnDrop(&done);
        let took = ask_until_answered(&g, &p_id, Duration::from_secs(20)).expect("answered once it runs");
        assert!(took < Duration::from_secs(3), "no back-off window to wait out, took {took:?}");
    });
}
