//! The patched libp2p-gossipsub (network/patched-deps/libp2p-gossipsub) against hostile control
//! messages from a connected peer (23 Nov 2026 backports): a PRUNE whose backoff overflows the clock
//! (RUSTSEC-2026-0535, -0537) used to panic the node; GRAFTs for made-up topics were recorded as the
//! peer's subscriptions without bound (RUSTSEC-2026-0536). The vendored crate's own tests can't run
//! here (they need quickcheck), so a real node is sent the raw RPCs over a stream, as an attacker would.
//!
//! Run:  cargo test -p openhydra-network --test gossipsub_hostile_control

use futures::{AsyncWriteExt, StreamExt};
use libp2p::gossipsub::{self, IdentTopic, MessageAuthenticity};
use libp2p::swarm::SwarmEvent;
use libp2p::{Multiaddr, PeerId, StreamProtocol};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

fn varint(mut v: u64, out: &mut Vec<u8>) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

/// A length-delimited protobuf field.
fn bytes_field(tag: u64, body: &[u8], out: &mut Vec<u8>) {
    varint(tag << 3 | 2, out);
    varint(body.len() as u64, out);
    out.extend_from_slice(body);
}

/// One gossipsub frame (varint length prefix) holding an RPC with only `control`.
fn frame(control: &[u8]) -> Vec<u8> {
    let mut rpc = Vec::new();
    bytes_field(3, control, &mut rpc); // RPC.control
    let mut out = Vec::new();
    varint(rpc.len() as u64, &mut out);
    out.extend_from_slice(&rpc);
    out
}

fn prune(topic: &str, backoff: u64) -> Vec<u8> {
    let mut p = Vec::new();
    bytes_field(1, topic.as_bytes(), &mut p); // ControlPrune.topic_id
    varint(3 << 3, &mut p); // ControlPrune.backoff (varint)
    varint(backoff, &mut p);
    let mut control = Vec::new();
    bytes_field(4, &p, &mut control); // ControlMessage.prune
    frame(&control)
}

fn grafts(topics: impl Iterator<Item = String>) -> Vec<u8> {
    let mut control = Vec::new();
    for t in topics {
        let mut g = Vec::new();
        bytes_field(1, t.as_bytes(), &mut g); // ControlGraft.topic_id
        bytes_field(3, &g, &mut control); // ControlMessage.graft
    }
    frame(&control)
}

/// What the test asks the node running in its own task.
enum Ask {
    Addr(oneshot::Sender<Multiaddr>),
    TopicsOf(PeerId, oneshot::Sender<usize>),
}

#[tokio::test]
async fn hostile_prune_and_graft_neither_crash_the_node_nor_grow_its_state() {
    // The node under test: gossipsub on topic "t", a fast heartbeat so a deferred overflow would fire.
    let mut node = libp2p::SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(libp2p::tcp::Config::default(), libp2p::noise::Config::new, libp2p::yamux::Config::default)
        .unwrap()
        .with_behaviour(|key| {
            let config = gossipsub::ConfigBuilder::default()
                .heartbeat_interval(Duration::from_millis(50))
                .build()
                .unwrap();
            gossipsub::Behaviour::<gossipsub::IdentityTransform, gossipsub::AllowAllSubscriptionFilter>::new(
                MessageAuthenticity::Signed(key.clone()),
                config,
            )
            .unwrap()
        })
        .unwrap()
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();
    node.behaviour_mut().subscribe(&IdentTopic::new("t")).unwrap();
    node.listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap()).unwrap();
    let node_id = *node.local_peer_id();
    let (ask_tx, mut ask_rx) = mpsc::channel::<Ask>(8);
    let running = tokio::spawn(async move {
        let mut addr = None;
        let mut waiting = Vec::new();
        loop {
            tokio::select! {
                e = node.select_next_some() => {
                    if let SwarmEvent::NewListenAddr { address, .. } = e {
                        addr = Some(address.clone());
                        for w in waiting.drain(..) { let _: Result<(), _> = oneshot::Sender::send(w, address.clone()); }
                    }
                }
                Some(ask) = ask_rx.recv() => match ask {
                    Ask::Addr(tx) => match &addr { Some(a) => { let _ = tx.send(a.clone()); } None => waiting.push(tx) },
                    Ask::TopicsOf(p, tx) => {
                        let n = node.behaviour().all_peers().find(|(id, _)| **id == p).map_or(0, |(_, t)| t.len());
                        let _ = tx.send(n);
                    }
                },
            }
        }
    });

    // The attacker: a plain node that opens gossipsub's protocol and writes raw frames.
    let mut attacker = libp2p::SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(libp2p::tcp::Config::default(), libp2p::noise::Config::new, libp2p::yamux::Config::default)
        .unwrap()
        .with_behaviour(|_| libp2p_stream::Behaviour::new())
        .unwrap()
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();
    let attacker_id = *attacker.local_peer_id();
    let mut control = attacker.behaviour().new_control();
    let (tx, rx) = oneshot::channel();
    ask_tx.send(Ask::Addr(tx)).await.unwrap();
    let addr = rx.await.unwrap();
    attacker.dial(addr.with(libp2p::multiaddr::Protocol::P2p(node_id))).unwrap();
    tokio::spawn(async move {
        loop {
            attacker.select_next_some().await;
        }
    });

    let mut stream = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match control.open_stream(node_id, StreamProtocol::new("/meshsub/1.1.0")).await {
                Ok(s) => break s,
                Err(_) => tokio::time::sleep(Duration::from_millis(50)).await,
            }
        }
    })
    .await
    .expect("opened gossipsub's protocol on the node");

    // A PRUNE for the node's topic with the largest backoff, then one just short of overflowing
    // the clock at once (it would have overflowed a heartbeat later), then GRAFTs for 500 topics
    // the attacker never subscribed to.
    stream.write_all(&prune("t", u64::MAX)).await.unwrap();
    stream.write_all(&prune("t", u64::MAX / 1_000_000_000)).await.unwrap();
    stream.write_all(&grafts((0..500).map(|i| format!("made-up-{i}")))).await.unwrap();
    stream.flush().await.unwrap();

    // Several heartbeats later the node is still running and recorded none of the made-up topics.
    tokio::time::sleep(Duration::from_millis(600)).await;
    assert!(!running.is_finished(), "the node's task ended (a panic) after the hostile PRUNE");
    let (tx, rx) = oneshot::channel();
    ask_tx.send(Ask::TopicsOf(attacker_id, tx)).await.unwrap();
    let topics = tokio::time::timeout(Duration::from_secs(5), rx).await.expect("the node answers").unwrap();
    assert_eq!(topics, 0, "GRAFTs recorded {topics} topics the peer never subscribed to");
}
