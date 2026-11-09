//! The patched libp2p-kad's address filter (network/patched-deps/libp2p-kad): an address the filter
//! refuses never reaches the routing table by any path, so it's never in a DHT answer. On the public
//! network the filter refuses private addresses (`swarm::routable_on_public_network`); here, over the
//! in-memory transport, it refuses one memory port, which exercises the same code paths.

use std::time::Duration;

use futures::StreamExt;
use libp2p::core::transport::{MemoryTransport, Transport};
use libp2p::core::upgrade::Version;
use libp2p::multiaddr::Protocol;
use libp2p::swarm::{Swarm, SwarmEvent};
use libp2p::{identity, kad, noise, yamux, Multiaddr, PeerId};

/// Memory ports the filter refuses. Memory transport ports are shared by the whole test process, so
/// each test listens on its own one in this range.
const REFUSED_PORT: u64 = 0xb00;
const REFUSED_PORTS: std::ops::Range<u64> = 0xb00..0xc00;

fn refused(addr: &Multiaddr) -> bool {
    addr.iter().any(|p| matches!(p, Protocol::Memory(port) if REFUSED_PORTS.contains(&port)))
}

fn node(filtered: bool) -> Swarm<kad::Behaviour<kad::store::MemoryStore>> {
    let key = identity::Keypair::generate_ed25519();
    let peer = PeerId::from(key.public());
    let transport = MemoryTransport::default()
        .upgrade(Version::V1)
        .authenticate(noise::Config::new(&key).unwrap())
        .multiplex(yamux::Config::default())
        .boxed();
    let mut cfg = kad::Config::new(libp2p::StreamProtocol::new("/openhydra/kad/1.0.0"));
    if filtered {
        cfg.set_address_filter(kad::AddressFilter::new(|_peer, addr| !refused(addr)));
    }
    let mut behaviour = kad::Behaviour::with_config(peer, kad::store::MemoryStore::new(peer), cfg);
    behaviour.set_mode(Some(kad::Mode::Server));
    let swarm_cfg = libp2p::swarm::Config::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(60));
    Swarm::new(transport, behaviour, peer, swarm_cfg)
}

fn addrs_in_table(swarm: &mut Swarm<kad::Behaviour<kad::store::MemoryStore>>, peer: &PeerId) -> Vec<Multiaddr> {
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

/// Drive all swarms until `done` says so, or panic after 10 s.
async fn run_until(
    swarms: &mut [&mut Swarm<kad::Behaviour<kad::store::MemoryStore>>],
    mut done: impl FnMut(usize, &SwarmEvent<kad::Event>) -> bool,
) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let (i, e) = futures::future::select_all(swarms.iter_mut().enumerate().map(|(i, s)| {
                Box::pin(async move { (i, s.select_next_some().await) })
            }))
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

#[test]
fn add_address_refuses_a_filtered_address() {
    let mut a = node(true);
    let b = PeerId::random();
    let bad: Multiaddr = format!("/memory/{REFUSED_PORT}").parse().unwrap();
    let good: Multiaddr = "/memory/7".parse().unwrap();
    assert!(matches!(a.behaviour_mut().add_address(&b, bad), kad::RoutingUpdate::Failed));
    assert!(addrs_in_table(&mut a, &b).is_empty());
    assert!(!matches!(a.behaviour_mut().add_address(&b, good), kad::RoutingUpdate::Failed));
    assert_eq!(addrs_in_table(&mut a, &b).len(), 1);
    // Control: without the filter the same address goes in.
    let mut u = node(false);
    let bad: Multiaddr = format!("/memory/{REFUSED_PORT}").parse().unwrap();
    assert!(!matches!(u.behaviour_mut().add_address(&b, bad), kad::RoutingUpdate::Failed));
    assert_eq!(addrs_in_table(&mut u, &b).len(), 1);
}

/// A peer already in A's table by an allowed address, then dialled on a refused one (an mDNS dial
/// to a LAN peer listed by its public address, in real life): the refused address doesn't join its
/// entry, and C, asking A for peers, never hears it.
#[tokio::test]
async fn a_dialled_refused_address_never_reaches_a_dht_answer() {
    let (in_table, answered) = dialled_refused_address(true, REFUSED_PORT).await;
    assert!(in_table.iter().all(|x| !refused(x)), "a refused address entered the table: {in_table:?}");
    assert!(answered.iter().all(|x| !refused(x)), "the refused address was handed out: {answered:?}");
}

/// The control: without the filter, the same dial does put the refused address in the table and
/// in the answer (so the test above shows the filter at work, not a path that never runs).
#[tokio::test]
async fn without_the_filter_a_dialled_address_does_reach_a_dht_answer() {
    let (in_table, answered) = dialled_refused_address(false, REFUSED_PORT + 5).await;
    assert!(in_table.iter().any(refused), "control: the dialled address should be in the table: {in_table:?}");
    assert!(answered.iter().any(refused), "control: and in the answer: {answered:?}");
}

/// A (filtered or not) has B in its table by an allowed address, then dials B at the refused
/// address `port`; C asks A for peers close to B. Returns B's addresses in A's table and in A's
/// answer to C.
async fn dialled_refused_address(filtered: bool, port: u64) -> (Vec<Multiaddr>, Vec<Multiaddr>) {
    let mut a = node(filtered);
    let mut b = node(false);
    let mut c = node(false);
    let (a_id, b_id) = (*a.local_peer_id(), *b.local_peer_id());

    let good: Multiaddr = "/memory/0".parse().unwrap();
    b.listen_on(good).unwrap();
    b.listen_on(format!("/memory/{port}").parse().unwrap()).unwrap();
    a.listen_on("/memory/0".parse().unwrap()).unwrap();
    let mut b_addrs = Vec::new();
    let mut a_addr = None;
    run_until(&mut [&mut a, &mut b], |i, e| {
        if let SwarmEvent::NewListenAddr { address, .. } = e {
            if i == 0 { a_addr = Some(address.clone()) } else { b_addrs.push(address.clone()) }
        }
        a_addr.is_some() && b_addrs.len() == 2
    })
    .await;
    let b_good = b_addrs.iter().find(|x| !refused(x)).unwrap().clone();
    let b_bad = b_addrs.iter().find(|x| refused(x)).unwrap().clone();

    // B enters A's table by its allowed address.
    a.behaviour_mut().add_address(&b_id, b_good.clone());
    // A then dials B on the refused address and talks Kademlia to it (that's what makes kad record
    // the dialled address).
    a.dial(b_bad.clone()).unwrap();
    a.behaviour_mut().get_closest_peers(PeerId::random());
    run_until(&mut [&mut a, &mut b], |i, e| {
        i == 0 && matches!(e, SwarmEvent::Behaviour(kad::Event::OutboundQueryProgressed { step, .. }) if step.last)
    })
    .await;
    let in_table = addrs_in_table(&mut a, &b_id);
    assert!(!in_table.is_empty(), "B is still in A's table");

    // C asks A for peers close to B: A's answer names B without the refused address.
    c.behaviour_mut().add_address(&a_id, a_addr.unwrap());
    c.behaviour_mut().get_closest_peers(b_id);
    let mut answered: Vec<Multiaddr> = Vec::new();
    let mut found_b = false;
    run_until(&mut [&mut a, &mut b, &mut c], |i, e| {
        if i == 2 {
            if let SwarmEvent::Behaviour(kad::Event::OutboundQueryProgressed {
                result: kad::QueryResult::GetClosestPeers(Ok(ok)),
                step,
                ..
            }) = e
            {
                for p in &ok.peers {
                    if p.peer_id == b_id {
                        found_b = true;
                        answered.extend(p.addrs.iter().cloned());
                    }
                }
                return step.last;
            }
        }
        false
    })
    .await;
    assert!(found_b, "A's answer should name B");
    (in_table, answered)
}

#[test]
fn the_public_network_rule_refuses_private_addresses_and_keeps_the_rest() {
    use openhydra_network::swarm::routable_on_public_network as routable;
    let ma = |s: &str| -> Multiaddr { s.parse().unwrap() };
    for refused in [
        "/ip4/192.168.0.4/tcp/4001",
        "/ip4/10.1.2.3/udp/4001/quic-v1",
        "/ip4/172.17.0.1/tcp/4001",
        "/ip4/127.0.0.1/tcp/4001",
        "/ip4/169.254.3.4/tcp/4001",
        // CGNAT and Tailscale: unreachable from elsewhere, decided and pinned (plan A1).
        "/ip4/100.64.0.1/tcp/4001",
        "/ip4/100.127.255.254/tcp/4001",
        "/ip6/fd00::1/tcp/4001",
        "/ip6/fe80::1/tcp/4001",
        "/ip6/::1/tcp/4001",
        "/ip6/::ffff:192.168.1.1/tcp/4001",
        // A circuit is judged by its relay hop's address too.
        "/ip4/10.0.0.5/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit",
        "/dns4/localhost/tcp/4001",
        "/dns4/printer.local/tcp/4001",
        "/ip6zone/en0/ip6/fe80::1/tcp/4001",
    ] {
        assert!(!routable(&ma(refused)), "{refused} should be refused");
    }
    for kept in [
        "/ip4/45.79.190.172/tcp/4001",
        "/ip4/100.128.0.1/tcp/4001",
        "/ip6/2a03:4000:41:ed1:384a:c3ff:fef1:2c58/udp/4001/quic-v1",
        "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit",
        "/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit",
        "/dns4/bootstrap-us.openhydra.co/tcp/4001",
    ] {
        assert!(routable(&ma(kept)), "{kept} should be kept");
    }
}

/// Provider answers carry addresses too: our own listen addresses for our own records, and the ones
/// an ADD_PROVIDER came with, as sent. A refused one must not be handed out there either.
#[tokio::test]
async fn a_provider_answer_never_carries_a_refused_address() {
    let mut a = node(true);
    let a_id = *a.local_peer_id();
    a.listen_on("/memory/0".parse().unwrap()).unwrap();
    a.listen_on(format!("/memory/{}", REFUSED_PORT + 1).parse().unwrap()).unwrap();
    let mut n = 0;
    run_until(&mut [&mut a], |_, e| {
        n += matches!(e, SwarmEvent::NewListenAddr { .. }) as usize;
        n == 2
    })
    .await;
    // Our own record: answered with our listen addresses, minus the refused one.
    let ours = kad::RecordKey::new(&"model-a");
    a.behaviour_mut().start_providing(ours.clone()).unwrap();
    let answer = a.behaviour_mut().provider_answer_for_test(&ours, &PeerId::random());
    let (_, addrs) = answer.iter().find(|(p, _)| *p == a_id).expect("A answers for its own record");
    assert!(!addrs.is_empty(), "the allowed listen address is still there");
    assert!(addrs.iter().all(|x| !refused(x)), "a refused own address was handed out: {addrs:?}");
    // Someone else's record, as an ADD_PROVIDER brought it.
    use libp2p::kad::store::RecordStore;
    let other = PeerId::random();
    let theirs = kad::RecordKey::new(&"model-b");
    a.behaviour_mut()
        .store_mut()
        .add_provider(kad::ProviderRecord {
            key: theirs.clone(),
            provider: other,
            expires: None,
            addresses: vec![format!("/memory/{}", REFUSED_PORT + 2).parse().unwrap(), "/memory/77".parse().unwrap()],
        })
        .unwrap();
    let answer = a.behaviour_mut().provider_answer_for_test(&theirs, &PeerId::random());
    let (_, addrs) = answer.iter().find(|(p, _)| *p == other).expect("A answers with the stored provider");
    assert_eq!(addrs, &vec!["/memory/77".parse::<Multiaddr>().unwrap()]);
}

/// Whether A, asking C for peers close to B, ends up connected to B. B listens only at a refused
/// address, and C (unfiltered) has it in its table there. `port` keeps runs apart.
async fn connects_through_a_query_answer(filtered: bool, port: u64) -> bool {
    let mut a = node(filtered);
    let mut b = node(false);
    let mut c = node(false);
    let (b_id, c_id) = (*b.local_peer_id(), *c.local_peer_id());
    b.listen_on(format!("/memory/{port}").parse().unwrap()).unwrap();
    c.listen_on("/memory/0".parse().unwrap()).unwrap();
    let (mut b_addr, mut c_addr) = (None, None);
    run_until(&mut [&mut b, &mut c], |i, e| {
        if let SwarmEvent::NewListenAddr { address, .. } = e {
            if i == 0 { b_addr = Some(address.clone()) } else { c_addr = Some(address.clone()) }
        }
        b_addr.is_some() && c_addr.is_some()
    })
    .await;
    c.behaviour_mut().add_address(&b_id, b_addr.unwrap());
    a.behaviour_mut().add_address(&c_id, c_addr.unwrap());
    a.behaviour_mut().get_closest_peers(b_id);
    let mut reached_b = false;
    run_until(&mut [&mut a, &mut b, &mut c], |i, e| {
        if i == 0 {
            if matches!(e, SwarmEvent::ConnectionEstablished { peer_id, .. } if *peer_id == b_id) {
                reached_b = true;
            }
            if let SwarmEvent::Behaviour(kad::Event::OutboundQueryProgressed { step, .. }) = e {
                return step.last;
            }
        }
        false
    })
    .await;
    reached_b
}

/// Other nodes' answers add addresses to a query's dials (`handle_pending_outbound_connection`): a
/// refused one must not be dialled. The unfiltered run is the control: there the same answer does
/// lead to a connection, so the filtered run's "no" means something.
#[tokio::test]
async fn an_address_from_a_query_answer_that_the_filter_refuses_is_never_dialled() {
    assert!(connects_through_a_query_answer(false, REFUSED_PORT + 3).await, "control: unfiltered, the answer is dialled");
    assert!(!connects_through_a_query_answer(true, REFUSED_PORT + 4).await, "filtered: never dialled");
}

/// The control for the provider-answer test: without the filter, our own refused listen address is
/// in the answer.
#[tokio::test]
async fn without_the_filter_a_provider_answer_carries_every_listen_address() {
    let mut a = node(false);
    let a_id = *a.local_peer_id();
    a.listen_on("/memory/0".parse().unwrap()).unwrap();
    a.listen_on(format!("/memory/{}", REFUSED_PORT + 6).parse().unwrap()).unwrap();
    let mut n = 0;
    run_until(&mut [&mut a], |_, e| {
        n += matches!(e, SwarmEvent::NewListenAddr { .. }) as usize;
        n == 2
    })
    .await;
    let ours = kad::RecordKey::new(&"model-a");
    a.behaviour_mut().start_providing(ours.clone()).unwrap();
    let answer = a.behaviour_mut().provider_answer_for_test(&ours, &PeerId::random());
    let (_, addrs) = answer.iter().find(|(p, _)| *p == a_id).expect("A answers for its own record");
    assert!(addrs.iter().any(refused), "control: the address is there without the filter: {addrs:?}");
}
