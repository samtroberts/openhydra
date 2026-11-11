// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The leak guard of the own-provider pin (`crate::pinned`): the gateway's loopback address for its
//! own provider must never be handed to another node in a DHT answer, on any network.
//!
//! The leak could only happen where no other filter applies (a private network) and when the gateway
//! answers DHT queries (Kademlia server mode). Full nodes in a test never reach server mode (it needs
//! a globally reachable address), so this drives Kademlia directly, over real TCP on loopback, with
//! every node in server mode and the gateway using the node's own filter
//! (`swarm::kad_address_filter`, private-network settings). Three nodes:
//!   P  the provider (pinned by G)          G  the gateway          Q  another node, asking G
//! P is in G's table by one non-loopback address (TEST-NET, never dialled), as a provider is by its
//! public or relay address; G then dials P on loopback and talks Kademlia to it (that is what makes
//! Kademlia record the dialled address). Q asks G for the peers closest to P.

use std::collections::HashSet;
use std::time::Duration;

use futures::StreamExt;
use libp2p::swarm::{Swarm, SwarmEvent};
use libp2p::{identity, kad, noise, tcp, yamux, Multiaddr, PeerId, Transport};
use openhydra_network::pinned::{self, PinnedPeers};

type Kad = kad::Behaviour<kad::store::MemoryStore>;

/// A Kademlia-only node on TCP, in server mode, with `filter` (or none).
fn node(filter: Option<kad::AddressFilter>) -> Swarm<Kad> {
    let key = identity::Keypair::generate_ed25519();
    let peer = PeerId::from(key.public());
    let transport = tcp::tokio::Transport::new(tcp::Config::default())
        .upgrade(libp2p::core::upgrade::Version::V1)
        .authenticate(noise::Config::new(&key).unwrap())
        .multiplex(yamux::Config::default())
        .boxed();
    let mut cfg = kad::Config::new(libp2p::StreamProtocol::new("/openhydra/kad/1.0.0"));
    if let Some(f) = filter {
        cfg.set_address_filter(f);
    }
    let mut behaviour = Kad::with_config(peer, kad::store::MemoryStore::new(peer), cfg);
    behaviour.set_mode(Some(kad::Mode::Server));
    let swarm_cfg = libp2p::swarm::Config::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(60));
    Swarm::new(transport, behaviour, peer, swarm_cfg)
}

/// The gateway's filter on a private network (no public-network rule), with these bootstraps.
fn gateway_filter(pinned: &PinnedPeers, bootstraps: HashSet<PeerId>) -> kad::AddressFilter {
    openhydra_network::swarm::kad_address_filter(false, bootstraps, pinned.clone())
}

fn is_loopback(a: &Multiaddr) -> bool {
    matches!(a.iter().next(), Some(libp2p::multiaddr::Protocol::Ip4(ip)) if ip.is_loopback())
        || matches!(a.iter().next(), Some(libp2p::multiaddr::Protocol::Ip6(ip)) if ip.is_loopback())
}

/// What a peer appears under in a node's Kademlia table.
fn addrs_in_table(swarm: &mut Swarm<Kad>, peer: &PeerId) -> Vec<Multiaddr> {
    let mut out = Vec::new();
    for bucket in swarm.behaviour_mut().kbuckets() {
        for entry in bucket.iter() {
            if entry.node.key.preimage() == peer {
                out.extend(entry.node.value.iter().cloned());
            }
        }
    }
    out
}

/// Drive all swarms until `done` says so, or panic after 15 s.
async fn run_until(swarms: &mut [&mut Swarm<Kad>], mut done: impl FnMut(usize, &SwarmEvent<kad::Event>) -> bool) {
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            let (i, e) = futures::future::select_all(
                swarms.iter_mut().enumerate().map(|(i, s)| Box::pin(async move { (i, s.select_next_some().await) })),
            )
            .await
            .0;
            if done(i, &e) {
                return;
            }
        }
    })
    .await
    .expect("timed out");
}

/// Listen on loopback TCP and return the address.
async fn listen(s: &mut Swarm<Kad>) -> Multiaddr {
    s.listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap()).unwrap();
    let mut addr = None;
    run_until(&mut [s], |_, e| {
        if let SwarmEvent::NewListenAddr { address, .. } = e {
            addr = Some(address.clone());
        }
        addr.is_some()
    })
    .await;
    addr.unwrap()
}

/// A provider's public address in the gateway's table: never dialled (TEST-NET-1).
fn public_addr() -> Multiaddr {
    "/ip4/192.0.2.7/tcp/4111".parse().unwrap()
}

/// The scene above. `pin` decides whether G pins P. Returns P's addresses in G's table after G
/// dialled P on loopback, and in G's answer to a lookup of P from Q (what G would send Q). With
/// `over_the_wire`, Q also really asks G, and the addresses Q received for P are returned too: P's
/// public address is never reachable, so Q's query only finishes when G handed out the loopback one,
/// which is why only the control asks over the wire.
async fn gateway_dials_its_provider_then_answers(pin: bool, over_the_wire: bool) -> (Vec<Multiaddr>, Vec<Multiaddr>, Vec<Multiaddr>) {
    let pinned = PinnedPeers::default();
    let mut p = node(None);
    let mut g = node(Some(gateway_filter(&pinned, HashSet::new())));
    let mut q = node(None);
    let (p_id, g_id, q_id) = (*p.local_peer_id(), *g.local_peer_id(), *q.local_peer_id());
    let p_lo = listen(&mut p).await;
    let g_addr = listen(&mut g).await;
    if pin {
        pinned.write().unwrap().insert(p_id, vec![p_lo.clone()]);
    }

    g.behaviour_mut().add_address(&p_id, public_addr());
    // G dials P on loopback and talks Kademlia to it.
    g.dial(p_lo.with(libp2p::multiaddr::Protocol::P2p(p_id))).unwrap();
    g.behaviour_mut().get_closest_peers(PeerId::random());
    let mut talked_to_p = false;
    run_until(&mut [&mut g, &mut p], |i, e| {
        if i == 0 && matches!(e, SwarmEvent::ConnectionEstablished { peer_id, .. } if *peer_id == p_id) {
            talked_to_p = true;
        }
        i == 0 && matches!(e, SwarmEvent::Behaviour(kad::Event::OutboundQueryProgressed { step, .. }) if step.last)
    })
    .await;
    assert!(talked_to_p, "G must have reached P on loopback");
    let in_table = without_p2p(addrs_in_table(&mut g, &p_id));
    assert!(!in_table.is_empty(), "P is still in G's table (by its public address at least)");
    let answer = g.behaviour_mut().closest_answer_for_test(&p_id, &q_id);
    let (_, hook) = answer.iter().find(|(peer, _)| *peer == p_id).cloned().expect("G's answer names P");
    let hook = without_p2p(hook);
    if !over_the_wire {
        return (in_table, hook, Vec::new());
    }

    // Q asks G for the peers closest to P.
    q.behaviour_mut().add_address(&g_id, g_addr);
    q.behaviour_mut().get_closest_peers(p_id);
    let mut wire = Vec::new();
    run_until(&mut [&mut g, &mut p, &mut q], |i, e| {
        if i == 2 {
            if let SwarmEvent::Behaviour(kad::Event::OutboundQueryProgressed {
                result: kad::QueryResult::GetClosestPeers(Ok(ok)),
                step,
                ..
            }) = e
            {
                for peer in ok.peers.iter().filter(|x| x.peer_id == p_id) {
                    wire.extend(peer.addrs.iter().cloned());
                }
                return step.last;
            }
        }
        false
    })
    .await;
    (in_table, hook, without_p2p(wire))
}

/// `addrs` without a trailing `/p2p/<id>` (the table keeps them with it).
fn without_p2p(addrs: Vec<Multiaddr>) -> Vec<Multiaddr> {
    addrs
        .into_iter()
        .map(|a| a.into_iter().filter(|c| !matches!(c, libp2p::multiaddr::Protocol::P2p(_))).collect())
        .collect()
}

/// The guard: G's own provider, pinned and dialled on loopback, is answered for with its public
/// address alone.
#[tokio::test]
async fn a_pinned_providers_loopback_address_never_reaches_a_dht_answer() {
    let (in_table, answered, _) = gateway_dials_its_provider_then_answers(true, false).await;
    assert!(in_table.iter().all(|a| !is_loopback(a)), "a loopback address entered G's table: {in_table:?}");
    assert!(answered.iter().all(|a| !is_loopback(a)), "G would hand out P's loopback address: {answered:?}");
    assert_eq!(answered, vec![public_addr()], "the public address is still answered");
}

/// The control: unpinned, the same scene does put the loopback address in the table and in the
/// answer, and Q really receives it over the wire, so the test above shows the guard at work, not a
/// path that never runs, and the answer it checks is what goes out.
#[tokio::test]
async fn control_unpinned_the_dialled_loopback_address_is_answered() {
    let (in_table, answered, wire) = gateway_dials_its_provider_then_answers(false, true).await;
    assert!(in_table.iter().any(is_loopback), "control: the dialled address should be in the table: {in_table:?}");
    assert!(answered.iter().any(is_loopback), "control: and in the answer: {answered:?}");
    assert!(wire.iter().any(is_loopback), "control: and Q received it: {wire:?}");
}

/// Addresses in the table before the pin (restored from the routing cache, added by hand): the pin
/// purges every loopback one of the pinned peer, any port or family, and leaves the rest.
#[test]
fn pinning_purges_loopback_addresses_already_in_the_table() {
    let pinned = PinnedPeers::default();
    let mut g = node(Some(gateway_filter(&pinned, HashSet::new())));
    let (p, other) = (PeerId::random(), PeerId::random());
    let stale: Vec<Multiaddr> = ["/ip4/127.0.0.1/tcp/4111", "/ip4/127.0.0.1/udp/4111/quic-v1", "/ip6/::1/tcp/4111"]
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
    for a in stale.iter().chain([&public_addr()]) {
        g.behaviour_mut().add_address(&p, a.clone());
    }
    g.behaviour_mut().add_address(&other, stale[0].clone());
    assert_eq!(addrs_in_table(&mut g, &p).len(), 4, "not pinned yet: all in");

    pinned.write().unwrap().insert(p, vec![stale[0].clone()]);
    assert_eq!(pinned::purge_refused(g.behaviour_mut(), &pinned, p), 3);
    assert_eq!(without_p2p(addrs_in_table(&mut g, &p)), vec![public_addr()]);
    assert_eq!(without_p2p(addrs_in_table(&mut g, &other)), vec![stale[0].clone()], "another peer's loopback address is untouched");
    // And none comes back afterwards.
    assert!(matches!(g.behaviour_mut().add_address(&p, stale[2].clone()), kad::RoutingUpdate::Failed));
}

/// A pinned peer that is also one of the node's bootstraps keeps its address (the user gave it,
/// and the DHT needs it).
#[test]
fn a_bootstrap_keeps_its_loopback_address_even_when_pinned() {
    let pinned = PinnedPeers::default();
    let p = PeerId::random();
    pinned.write().unwrap().insert(p, vec!["/ip4/127.0.0.1/tcp/4111".parse().unwrap()]);
    let mut g = node(Some(gateway_filter(&pinned, [p].into())));
    assert!(!matches!(g.behaviour_mut().add_address(&p, "/ip4/127.0.0.1/tcp/4111".parse().unwrap()), kad::RoutingUpdate::Failed));
    assert_eq!(addrs_in_table(&mut g, &p).len(), 1);
}

/// Provider answers carry addresses too (an ADD_PROVIDER's, as sent): a pinned peer's loopback
/// address is not handed out there either.
#[test]
fn a_provider_answer_never_carries_a_pinned_peers_loopback_address() {
    use libp2p::kad::store::RecordStore;
    let pinned = PinnedPeers::default();
    let p = PeerId::random();
    pinned.write().unwrap().insert(p, vec!["/ip4/127.0.0.1/tcp/4111".parse().unwrap()]);
    let mut g = node(Some(gateway_filter(&pinned, HashSet::new())));
    let key = kad::RecordKey::new(&"qwen2.5:0.5b");
    g.behaviour_mut()
        .store_mut()
        .add_provider(kad::ProviderRecord {
            key: key.clone(),
            provider: p,
            expires: None,
            addresses: vec!["/ip4/127.0.0.1/tcp/4111".parse().unwrap(), public_addr()],
        })
        .unwrap();
    let answer = g.behaviour_mut().provider_answer_for_test(&key, &PeerId::random());
    let (_, addrs) = answer.iter().find(|(peer, _)| *peer == p).expect("G answers with the stored provider");
    assert_eq!(addrs, &vec![public_addr()]);
}
