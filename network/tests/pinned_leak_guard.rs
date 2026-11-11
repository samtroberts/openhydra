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
    node_with(filter, None)
}

/// [`node`] with a query timeout (a lookup that can't reach a peer then ends with what it heard).
fn node_with(filter: Option<kad::AddressFilter>, query_timeout: Option<Duration>) -> Swarm<Kad> {
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
    if let Some(t) = query_timeout {
        cfg.set_query_timeout(t);
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

/// Whether a peer's entry in a node's table is marked connected: only Kademlia confirming its
/// protocol over a connection sets that here, the very step that offers the dialled address to the
/// filter (so a test that sees it knows the filter was asked).
fn connected_in_table(swarm: &mut Swarm<Kad>, peer: &PeerId) -> bool {
    swarm.behaviour_mut().kbuckets().any(|b| b.iter().any(|e| e.node.key.preimage() == peer && e.status == kad::NodeStatus::Connected))
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

/// A provider's public (non-loopback) address in the gateway's table. A DNS name: these nodes have
/// no DNS transport, so a dial to it fails at once and reports the address (Q's dials are how the
/// test reads what G handed out).
fn public_addr() -> Multiaddr {
    "/dns4/provider.invalid/tcp/4111".parse().unwrap()
}

/// The scene above. `pin` decides whether G pins P. Returns P's addresses in G's table after G
/// dialled P on loopback, in G's answer to a lookup of P from Q (what G would send Q), and in what
/// Q really received when it asked G over the wire: Q dials every address G gave it for P (a peer
/// it can't reach isn't in a lookup's result, so Q's dials are what's read). Q's lookups time out
/// after 3 s, in case one dial hangs.
async fn gateway_dials_its_provider_then_answers(pin: bool) -> (Vec<Multiaddr>, Vec<Multiaddr>, Vec<Multiaddr>) {
    let pinned = PinnedPeers::default();
    let mut p = node(None);
    let mut g = node(Some(gateway_filter(&pinned, HashSet::new())));
    let mut q = node_with(None, Some(Duration::from_secs(3)));
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
    assert!(connected_in_table(&mut g, &p_id), "Kademlia confirmed P over the loopback connection (the filter was asked)");
    let in_table = without_p2p(addrs_in_table(&mut g, &p_id));
    assert!(!in_table.is_empty(), "P is still in G's table (by its public address at least)");
    let answer = g.behaviour_mut().closest_answer_for_test(&p_id, &q_id);
    let (_, hook) = answer.iter().find(|(peer, _)| *peer == p_id).cloned().expect("G's answer names P");
    let hook = without_p2p(hook);

    // Q asks G for the peers closest to P, and then tries the addresses G gave it for P: every one
    // it dials (or fails to) is one G handed out.
    q.behaviour_mut().add_address(&g_id, g_addr);
    q.behaviour_mut().get_closest_peers(p_id);
    let mut wire = Vec::new();
    run_until(&mut [&mut g, &mut p, &mut q], |i, e| {
        if i == 2 {
            match e {
                SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } if *peer_id == p_id => {
                    wire.push(endpoint.get_remote_address().clone());
                }
                SwarmEvent::OutgoingConnectionError { peer_id: Some(peer), error: libp2p::swarm::DialError::Transport(errs), .. } if *peer == p_id => {
                    wire.extend(errs.iter().map(|(a, _)| a.clone()));
                }
                SwarmEvent::Behaviour(kad::Event::OutboundQueryProgressed { result: kad::QueryResult::GetClosestPeers(r), step, .. }) => {
                    let peers = match r {
                        Ok(ok) => &ok.peers,
                        Err(kad::GetClosestPeersError::Timeout { peers, .. }) => peers,
                    };
                    for peer in peers.iter().filter(|x| x.peer_id == p_id) {
                        wire.extend(peer.addrs.iter().cloned());
                    }
                    return step.last;
                }
                _ => {}
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
    let (in_table, answered, wire) = gateway_dials_its_provider_then_answers(true).await;
    assert!(in_table.iter().all(|a| !is_loopback(a)), "a loopback address entered G's table: {in_table:?}");
    assert!(answered.iter().all(|a| !is_loopback(a)), "G would hand out P's loopback address: {answered:?}");
    assert_eq!(answered, vec![public_addr()], "the public address is still answered");
    assert!(wire.iter().all(|a| !is_loopback(a)), "Q received P's loopback address over the wire: {wire:?}");
    assert!(wire.contains(&public_addr()), "Q did hear of P, by its public address: {wire:?}");
}

/// The control: unpinned, the same scene does put the loopback address in the table and in the
/// answer, and Q really receives it over the wire, so the test above shows the guard at work, not a
/// path that never runs, and the answer it checks is what goes out.
#[tokio::test]
async fn control_unpinned_the_dialled_loopback_address_is_answered() {
    let (in_table, answered, wire) = gateway_dials_its_provider_then_answers(false).await;
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
    assert_eq!(pinned::purge_refused(g.behaviour_mut(), &pinned, &[p].into(), p), 0, "a bootstrap keeps its addresses");
    assert_eq!(addrs_in_table(&mut g, &p).len(), 4);
    assert_eq!(pinned::purge_refused(g.behaviour_mut(), &pinned, &HashSet::new(), p), 3);
    assert_eq!(without_p2p(addrs_in_table(&mut g, &p)), vec![public_addr()]);
    assert_eq!(without_p2p(addrs_in_table(&mut g, &other)), vec![stale[0].clone()], "another peer's loopback address is untouched");
    // And none comes back afterwards.
    assert!(matches!(g.behaviour_mut().add_address(&p, stale[2].clone()), kad::RoutingUpdate::Failed));
}

/// A pinned provider G has never heard of by any other address: dialled on loopback, it gets no
/// entry at all (Kademlia has no address it may keep), so no answer can name it.
#[tokio::test]
async fn a_pinned_provider_known_only_by_loopback_gets_no_entry() {
    let pinned = PinnedPeers::default();
    let mut p = node(None);
    let mut g = node(Some(gateway_filter(&pinned, HashSet::new())));
    let p_id = *p.local_peer_id();
    let p_lo = listen(&mut p).await;
    let g_addr = listen(&mut g).await;
    pinned.write().unwrap().insert(p_id, vec![p_lo.clone()]);
    g.dial(p_lo.with(libp2p::multiaddr::Protocol::P2p(p_id))).unwrap();
    let mut connected = false;
    run_until(&mut [&mut g, &mut p], |i, e| {
        connected |= i == 0 && matches!(e, SwarmEvent::ConnectionEstablished { peer_id, .. } if *peer_id == p_id);
        connected
    })
    .await;
    // G's table is empty, so G has nobody to ask; P asks G instead, over the connection G dialled.
    // G then confirms P's Kademlia protocol on that connection, which offers G's table the
    // dialled loopback address: refused, and with no other address P is unroutable for G.
    let g_id = *g.local_peer_id();
    p.behaviour_mut().add_address(&g_id, g_addr);
    p.behaviour_mut().get_closest_peers(PeerId::random());
    let mut unroutable = false;
    run_until(&mut [&mut g, &mut p], |i, e| {
        unroutable |= i == 0 && matches!(e, SwarmEvent::Behaviour(kad::Event::UnroutablePeer { peer }) if *peer == p_id);
        unroutable
    })
    .await;
    assert!(unroutable, "Kademlia was offered P's loopback address and refused it (P unroutable)");
    assert!(addrs_in_table(&mut g, &p_id).is_empty(), "no entry for P");
    let q = PeerId::random();
    assert!(g.behaviour_mut().closest_answer_for_test(&p_id, &q).iter().all(|(peer, _)| *peer != p_id), "no answer names P");
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
