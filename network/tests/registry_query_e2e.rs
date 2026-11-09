//! C7 end-to-end: the `/openhydra/registry/1.0.0` query protocol between two real libp2p
//! swarms over TCP + Noise + Yamux. Proves the wire path a NAT'd consumer relies on when passive
//! discovery (DHT get_providers + gossip PEX) comes up empty: it asks a bootstrap's provider
//! registry directly and gets back the retained records.
//!
//! One swarm plays the bootstrap (Inbound registry responder, backed by a populated
//! `ProviderRegistry`); the other plays the consumer (Outbound). The consumer dials, then queries
//! for an unknown model (expects an empty reply over the wire) and a known model (expects the
//! provider record). This is the deterministic complement to the unit tests — it exercises the
//! codec framing, protocol negotiation, and the `providers_for` lookup in one flow.
//!
//! Run:  cargo test --test registry_query_e2e

use std::time::Duration;

use futures::StreamExt;
use libp2p::request_response::{self, ProtocolSupport};
use libp2p::swarm::SwarmEvent;
use libp2p::Swarm;
use openhydra_network::registry::ProviderRegistry;
use openhydra_network::registry_proto::{
    registry_behaviour, RegistryCodec, RegistryQuery, RegistryReply,
};
use openhydra_network::types::PeerRecord;

type RegBehaviour = request_response::Behaviour<RegistryCodec>;

fn build_swarm(support: ProtocolSupport) -> Swarm<RegBehaviour> {
    libp2p::SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )
        .unwrap()
        .with_behaviour(move |_| registry_behaviour(support))
        .unwrap()
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(30)))
        .build()
}

fn record(model: &str, peer: &str) -> PeerRecord {
    PeerRecord {
        peer_id: peer.into(),
        model_id: model.into(),
        libp2p_peer_id: format!("12D3KooW-{peer}"),
        ..Default::default()
    }
}

/// Live probe against a DEPLOYED bootstrap — proves the real `bootstrap_bin.rs` answer path
/// (protocol negotiation + `providers_for` lookup + reply) over the network, which the in-process
/// test above can't (it uses an inline responder). Ignored by default (needs egress + a running
/// bootstrap); run explicitly:
///
///   OH_BOOTSTRAP='/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb' \
///   OH_MODELS='qwen2.5-coder:1.5b,tinyllama:latest,tinyllama,__absent__' \
///   cargo test -p openhydra-network --test registry_query_e2e -- --ignored --nocapture live_probe
///
/// Success = every query gets a Response (not OutboundFailure). A non-empty count additionally
/// proves the ingest→query chain end to end. Record counts per model are printed.
#[tokio::test]
#[ignore = "hits a live deployed bootstrap; run with --ignored"]
async fn live_probe_deployed_bootstrap_answers() {
    let addr: libp2p::Multiaddr = std::env::var("OH_BOOTSTRAP")
        .expect("set OH_BOOTSTRAP to the bootstrap multiaddr")
        .parse()
        .expect("valid multiaddr");
    let models: Vec<String> = std::env::var("OH_MODELS")
        .unwrap_or_else(|_| "qwen2.5-coder:1.5b,tinyllama:latest,tinyllama,__absent__".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let mut consumer = build_swarm(ProtocolSupport::Outbound);
    consumer.dial(addr).expect("dial bootstrap");

    use std::collections::HashMap;
    let mut pending: HashMap<request_response::OutboundRequestId, String> = HashMap::new();
    let mut answered = 0usize;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);

    loop {
        tokio::select! {
            ev = consumer.select_next_some() => match ev {
                SwarmEvent::ConnectionEstablished { peer_id, .. } if pending.is_empty() => {
                    for m in &models {
                        let id = consumer.behaviour_mut().send_request(&peer_id, RegistryQuery { model_id: m.clone() });
                        pending.insert(id, m.clone());
                    }
                }
                SwarmEvent::Behaviour(request_response::Event::Message {
                    message: request_response::Message::Response { request_id, response }, ..
                }) => {
                    let model = pending.remove(&request_id).unwrap_or_default();
                    println!("LIVE registry answer: model={model:?} records={}", response.records.len());
                    for r in &response.records {
                        println!("    provider peer_id={} libp2p={} model_id={}", r.peer_id, r.libp2p_peer_id, r.model_id);
                    }
                    answered += 1;
                    if answered == models.len() {
                        assert!(answered > 0, "deployed bootstrap answered no queries");
                        return; // every query got a Response → answer path proven
                    }
                }
                SwarmEvent::Behaviour(request_response::Event::OutboundFailure { error, request_id, .. }) => {
                    let model = pending.get(&request_id).cloned().unwrap_or_default();
                    panic!("deployed bootstrap FAILED to answer model={model:?}: {error} — answer path broken or protocol unsupported");
                }
                _ => {}
            },
            _ = tokio::time::sleep_until(deadline) => {
                panic!("timed out — {answered}/{} queries answered by the deployed bootstrap", models.len());
            }
        }
    }
}

#[tokio::test]
async fn consumer_queries_bootstrap_registry_over_the_wire() {
    let mut bootstrap = build_swarm(ProtocolSupport::Inbound);
    let mut consumer = build_swarm(ProtocolSupport::Outbound);

    // The bootstrap has retained one provider for "m1" (and nothing for "absent").
    let mut registry = ProviderRegistry::new(300_000);
    registry.insert(record("m1", "p1"), 1_000);

    bootstrap
        .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
        .unwrap();
    let addr = loop {
        if let SwarmEvent::NewListenAddr { address, .. } = bootstrap.select_next_some().await {
            break address;
        }
    };
    consumer.dial(addr).unwrap();

    // Phase machine: 0 = wait for connection → query "absent"; 1 = expect empty → query "m1";
    // 2 = expect the record → done. The bootstrap answers every inbound query from the registry.
    let mut phase = 0u8;
    let mut bootstrap_pid = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);

    loop {
        tokio::select! {
            ev = bootstrap.select_next_some() => {
                if let SwarmEvent::Behaviour(request_response::Event::Message {
                    message: request_response::Message::Request { request, channel, .. }, ..
                }) = ev {
                    let records = registry.providers_for(&request.model_id, 2_000);
                    bootstrap
                        .behaviour_mut()
                        .send_response(channel, RegistryReply { records })
                        .expect("send_response");
                }
            }
            ev = consumer.select_next_some() => {
                match ev {
                    SwarmEvent::ConnectionEstablished { peer_id, .. } if phase == 0 => {
                        bootstrap_pid = Some(peer_id);
                        consumer.behaviour_mut().send_request(&peer_id, RegistryQuery { model_id: "absent".into() });
                        phase = 1;
                    }
                    SwarmEvent::Behaviour(request_response::Event::Message {
                        message: request_response::Message::Response { response, .. }, ..
                    }) => {
                        match phase {
                            1 => {
                                assert!(response.records.is_empty(), "unknown model must return empty over the wire");
                                let pid = bootstrap_pid.expect("connected");
                                consumer.behaviour_mut().send_request(&pid, RegistryQuery { model_id: "m1".into() });
                                phase = 2;
                            }
                            2 => {
                                assert_eq!(response.records.len(), 1, "m1 has exactly one provider");
                                assert_eq!(response.records[0].peer_id, "p1");
                                assert_eq!(response.records[0].model_id, "m1");
                                return; // success
                            }
                            _ => unreachable!(),
                        }
                    }
                    SwarmEvent::Behaviour(request_response::Event::OutboundFailure { error, .. }) => {
                        panic!("registry query failed: {error}");
                    }
                    _ => {}
                }
            }
            _ = tokio::time::sleep_until(deadline) => panic!("timed out at phase {phase} waiting for registry response"),
        }
    }
}
