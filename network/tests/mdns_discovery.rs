//! LAN discovery (mDNS) with the patched libp2p-mdns (network/patched-deps/libp2p-mdns).
//!
//! A node whose listener comes up after mDNS has settled must be found within seconds, from its own
//! announcement. Once an interface has heard any answer, peers query it only every 5 minutes, and
//! they learn addresses only from answers, so without the patch's announcements the late node
//! stayed invisible until the next query (this test fails against the unpatched crate).
//!
//! The test reads the mDNS packets on the wire itself, so it can't pass for the wrong reason: any
//! query on the LAN after B started listening (ours or another node's) makes B answer, and an answer
//! looks like an announcement, so such a run is reported inconclusive rather than passed. Answers
//! from other mDNS nodes on the LAN (an OpenHydra app, say) don't matter. It also checks that every
//! packet A, B and C send fits an unfragmented IPv6 packet (1232 bytes).
//!
//! It runs on this machine's real network interfaces (mDNS ignores loopback), so it needs one that
//! carries multicast. Hence `#[ignore]`.
//!
//! Run:  cargo test -p openhydra-network --test mdns_discovery -- --ignored --nocapture

use futures::StreamExt;
use libp2p::{mdns, noise, swarm::SwarmEvent, tcp, yamux, PeerId, Swarm, SwarmBuilder};
use std::collections::HashSet;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn node() -> Swarm<mdns::tokio::Behaviour> {
    SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(tcp::Config::default(), noise::Config::new, yamux::Config::default)
        .unwrap()
        .with_behaviour(|k| mdns::tokio::Behaviour::new(mdns::Config::default(), k.public().to_peer_id()).unwrap())
        .unwrap()
        .build()
}

fn note_found(event: SwarmEvent<mdns::Event>, seen: &mut HashSet<PeerId>) {
    if let SwarmEvent::Behaviour(mdns::Event::Discovered(list)) = event {
        seen.extend(list.into_iter().map(|(peer, _)| peer));
    }
}

/// One mDNS packet seen on the wire.
#[derive(Debug, Clone)]
struct Packet {
    at: Instant,
    /// It asks something (a query); otherwise it's an answer or announcement.
    query: bool,
    /// The peer ids its records name (`dnsaddr=…/p2p/<id>`): who sent an answer.
    peers: Vec<PeerId>,
    len: usize,
}

/// Listen to IPv4 mDNS on port 5353 (shared with the responders) and record every packet.
fn wire_listener(packets: Arc<Mutex<Vec<Packet>>>) {
    use socket2::{Domain, Protocol, Socket, Type};
    let sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP)).unwrap();
    sock.set_reuse_address(true).unwrap();
    sock.set_reuse_port(true).unwrap();
    sock.bind(&SocketAddr::from((Ipv4Addr::UNSPECIFIED, 5353)).into()).unwrap();
    sock.join_multicast_v4(&Ipv4Addr::new(224, 0, 0, 251), &Ipv4Addr::UNSPECIFIED).unwrap();
    let sock: UdpSocket = sock.into();
    sock.set_read_timeout(Some(Duration::from_millis(200))).unwrap();
    std::thread::spawn(move || {
        let mut buf = vec![0u8; 9000];
        loop {
            let Ok((n, _from)) = sock.recv_from(&mut buf) else { continue };
            let at = Instant::now();
            let Ok(msg) = hickory_proto::op::Message::from_vec(&buf[..n]) else { continue };
            let mut peers = Vec::new();
            for record in msg.answers.iter().chain(&msg.additionals) {
                if let hickory_proto::rr::RData::TXT(txt) = &record.data {
                    for part in txt.txt_data.iter() {
                        let s = String::from_utf8_lossy(part);
                        if let Some(id) = s.rsplit("/p2p/").next().filter(|_| s.contains("/p2p/")) {
                            if let Ok(p) = id.parse::<PeerId>() {
                                peers.push(p);
                            }
                        }
                    }
                }
            }
            packets.lock().unwrap().push(Packet { at, query: !msg.queries.is_empty(), peers, len: n });
        }
    });
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs a real network interface with multicast"]
async fn a_late_listener_is_found_within_seconds() {
    let packets = Arc::new(Mutex::new(Vec::new()));
    wire_listener(packets.clone());

    let (mut a, mut b, mut c) = (node(), node(), node());
    let (pa, pb, pc) = (*a.local_peer_id(), *b.local_peer_id(), *c.local_peer_id());
    a.listen_on("/ip4/0.0.0.0/tcp/0".parse().unwrap()).unwrap();
    c.listen_on("/ip4/0.0.0.0/tcp/0".parse().unwrap()).unwrap();
    let (mut seen_a, mut seen_b, mut seen_c) = (HashSet::new(), HashSet::new(), HashSet::new());

    // A and C find each other, so all three hear answers and stop probing: from then on they
    // query only every 5 minutes. B has no listen address yet.
    let settled = Instant::now() + Duration::from_secs(8);
    while Instant::now() < settled {
        tokio::select! {
            e = a.select_next_some() => note_found(e, &mut seen_a),
            e = b.select_next_some() => note_found(e, &mut seen_b),
            e = c.select_next_some() => note_found(e, &mut seen_c),
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
    }
    assert!(seen_a.contains(&pc) && seen_c.contains(&pa), "A and C should have found each other");
    assert!(!seen_a.contains(&pb), "B has no address yet");

    // B starts listening: A must find it from B's announcement, not from an answer to a query.
    let listening = Instant::now();
    b.listen_on("/ip4/0.0.0.0/tcp/0".parse().unwrap()).unwrap();
    while !seen_a.contains(&pb) && listening.elapsed() < Duration::from_secs(30) {
        tokio::select! {
            e = a.select_next_some() => note_found(e, &mut seen_a),
            e = b.select_next_some() => note_found(e, &mut seen_b),
            e = c.select_next_some() => note_found(e, &mut seen_c),
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
    }
    let took = listening.elapsed();
    tokio::time::sleep(Duration::from_millis(300)).await; // the listener catches up
    let packets = packets.lock().unwrap().clone();
    println!("A found B {took:?} after B started listening; {} mDNS packets seen", packets.len());

    // Every packet of ours fits an unfragmented IPv6 packet.
    let ours = [pa, pb, pc];
    for p in packets.iter().filter(|p| p.peers.iter().any(|id| ours.contains(id))) {
        assert!(p.len <= 1232, "a {}-byte mDNS packet from one of ours", p.len);
    }

    assert!(seen_a.contains(&pb), "A didn't find B within 30 s");
    let first_from_b = packets
        .iter()
        .filter(|p| p.at >= listening && !p.query && p.peers.contains(&pb))
        .map(|p| p.at)
        .min()
        .expect("A found B, so B sent a packet");
    if let Some(q) = packets.iter().find(|p| p.query && p.at >= listening && p.at < first_from_b) {
        panic!(
            "inconclusive: a query went out {:?} after B started listening, before B's first packet, so B may only have \
             been answering it (rerun; another mDNS node on this LAN queried at that moment)",
            q.at - listening
        );
    }
    assert!(took < Duration::from_secs(5), "A found B only after {took:?}");
}
