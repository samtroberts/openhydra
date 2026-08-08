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
