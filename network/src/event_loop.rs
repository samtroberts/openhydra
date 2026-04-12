//! Swarm event loop — runs on a background tokio task.
//!
//! Receives commands from the Python thread via mpsc and drives the
//! libp2p swarm. Results are sent back via oneshot channels.

use std::collections::HashMap;

use futures::StreamExt;
use libp2p::swarm::SwarmEvent;
use libp2p::{kad, Multiaddr};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

use crate::behaviour::{OpenHydraBehaviour, OpenHydraBehaviourEvent};
use crate::dht;
use crate::types::{DiscoveredPeer, NatInfo, PeerRecord};

/// Commands sent from the Python thread to the swarm event loop.
pub enum SwarmCommand {
    /// Publish a peer record to Kademlia DHT.
    Announce {
        record: PeerRecord,
        reply: oneshot::Sender<Result<(), String>>,
    },
    /// Discover peers serving a given model_id.
    Discover {
        model_id: String,
        reply: oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>,
    },
    /// Get current NAT status.
    NatStatus {
        reply: oneshot::Sender<NatInfo>,
    },
    /// Resolve a reachable address for a peer (direct or via relay).
    ResolveAddress {
        peer_id: String,
        reply: oneshot::Sender<Result<String, String>>,
    },
    /// Graceful shutdown.
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

/// State tracked by the event loop.
struct LoopState {
    /// Cached NAT info from AutoNAT probes.
    nat_info: NatInfo,
    /// Known peers from Kademlia queries, keyed by OpenHydra peer_id.
    known_peers: HashMap<String, PeerRecord>,
    /// Pending Kademlia GET queries: query_id → reply channel.
    pending_discovers: HashMap<kad::QueryId, PendingDiscover>,
    /// External addresses discovered by AutoNAT / Identify.
    external_addrs: Vec<Multiaddr>,
    /// Relay addresses we've reserved.
    #[allow(dead_code)]
    relay_addrs: Vec<Multiaddr>,
}

struct PendingDiscover {
    #[allow(dead_code)]
    model_id: String,
    records: Vec<PeerRecord>,
    reply: oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>,
}

impl Default for LoopState {
    fn default() -> Self {
        Self {
            nat_info: NatInfo {
                nat_type: "unknown".into(),
                external_ip: String::new(),
                external_port: 0,
                is_public: false,
            },
            known_peers: HashMap::new(),
            pending_discovers: HashMap::new(),
            external_addrs: Vec::new(),
            relay_addrs: Vec::new(),
        }
    }
}

/// Run the swarm event loop until shutdown.
pub async fn run_event_loop(
    mut swarm: libp2p::Swarm<OpenHydraBehaviour>,
    mut cmd_rx: mpsc::Receiver<SwarmCommand>,
) {
    let mut state = LoopState::default();

    // Kick off Kademlia bootstrap (populate routing table from bootstrap peers).
    if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
        warn!("kademlia bootstrap failed (no peers yet?): {e}");
    }

    loop {
        tokio::select! {
            // Process commands from Python.
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(SwarmCommand::Announce { record, reply }) => {
                        handle_announce(&mut swarm, &record, reply);
                    }
                    Some(SwarmCommand::Discover { model_id, reply }) => {
                        handle_discover(&mut swarm, &model_id, reply, &mut state);
                    }
                    Some(SwarmCommand::NatStatus { reply }) => {
                        let _ = reply.send(state.nat_info.clone());
                    }
                    Some(SwarmCommand::ResolveAddress { peer_id, reply }) => {
                        handle_resolve(&state, &peer_id, reply);
                    }
                    Some(SwarmCommand::Shutdown { reply }) => {
                        info!("swarm shutting down");
                        let _ = reply.send(());
                        return;
                    }
                    None => {
                        info!("command channel closed, shutting down swarm");
                        return;
                    }
                }
            }
            // Process swarm events.
            event = swarm.select_next_some() => {
                handle_swarm_event(event, &mut swarm, &mut state);
            }
        }
    }
}

/// Handle an announce command: PUT the peer record into Kademlia.
fn handle_announce(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    record: &PeerRecord,
    reply: oneshot::Sender<Result<(), String>>,
) {
    let key = dht::peer_record_key(&record.model_id, &record.peer_id);
    match dht::encode_record(record) {
        Ok(value) => {
            let kad_record = kad::Record {
                key,
                value,
                publisher: None,
                expires: None,
            };
            match swarm
                .behaviour_mut()
                .kademlia
                .put_record(kad_record, kad::Quorum::One)
            {
                Ok(_) => {
                    info!(
                        model_id = %record.model_id,
                        peer_id = %record.peer_id,
                        "announced to kademlia"
                    );
                    let _ = reply.send(Ok(()));
                }
                Err(e) => {
                    let _ = reply.send(Err(format!("kademlia put_record: {e}")));
                }
            }
        }
        Err(e) => {
            let _ = reply.send(Err(e));
        }
    }
}

/// Handle a discover command: GET records from Kademlia matching the model_id.
fn handle_discover(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    model_id: &str,
    reply: oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>,
    state: &mut LoopState,
) {
    // Use Kademlia get_record with the model provider key.
    let key = dht::model_provider_key(model_id);
    let query_id = swarm.behaviour_mut().kademlia.get_record(key);
    state.pending_discovers.insert(
        query_id,
        PendingDiscover {
            model_id: model_id.to_string(),
            records: Vec::new(),
            reply,
        },
    );
}

/// Handle a resolve_address command.
fn handle_resolve(
    state: &LoopState,
    peer_id: &str,
    reply: oneshot::Sender<Result<String, String>>,
) {
    // Look up the peer in our known_peers cache.
    if let Some(record) = state.known_peers.get(peer_id) {
        if record.requires_relay && !record.relay_address.is_empty() {
            // Peer needs relay — return the relay circuit address.
            let _ = reply.send(Ok(record.relay_address.clone()));
        } else {
            // Direct connection.
            let _ = reply.send(Ok(format!("{}:{}", record.host, record.port)));
        }
    } else {
        let _ = reply.send(Err(format!("peer {peer_id} not found in cache")));
    }
}

/// Process a swarm event.
fn handle_swarm_event(
    event: SwarmEvent<OpenHydraBehaviourEvent>,
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
) {
    match event {
        // ── Kademlia ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Kademlia(kad_event)) => {
            handle_kad_event(kad_event, state);
        }

        // ── AutoNAT ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Autonat(autonat_event)) => {
            handle_autonat_event(autonat_event, state);
        }

        // ── Identify ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Identify(identify_event)) => {
            if let libp2p::identify::Event::Received { peer_id, info, .. } = identify_event {
                debug!(%peer_id, protocol = %info.protocol_version, "identify received");
                // Add observed addresses to Kademlia.
                for addr in &info.listen_addrs {
                    swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr.clone());
                }
            }
        }

        // ── mDNS ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Mdns(mdns_event)) => {
            match mdns_event {
                libp2p::mdns::Event::Discovered(peers) => {
                    for (peer_id, addr) in peers {
                        info!(%peer_id, %addr, "mDNS discovered peer");
                        swarm
                            .behaviour_mut()
                            .kademlia
                            .add_address(&peer_id, addr);
                    }
                }
                libp2p::mdns::Event::Expired(peers) => {
                    for (peer_id, _) in peers {
                        debug!(%peer_id, "mDNS peer expired");
                    }
                }
            }
        }

        // ── Relay Client ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::RelayClient(relay_event)) => {
            match relay_event {
                libp2p::relay::client::Event::ReservationReqAccepted {
                    relay_peer_id, ..
                } => {
                    info!(%relay_peer_id, "relay reservation accepted");
                }
                libp2p::relay::client::Event::OutboundCircuitEstablished {
                    relay_peer_id, ..
                } => {
                    info!(%relay_peer_id, "outbound circuit established through relay");
                }
                libp2p::relay::client::Event::InboundCircuitEstablished {
                    src_peer_id, ..
                } => {
                    info!(%src_peer_id, "inbound circuit established from peer");
                }
            }
        }

        // ── DCUtR ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Dcutr(dcutr_event)) => {
            let peer = dcutr_event.remote_peer_id;
            match dcutr_event.result {
                Ok(conn_id) => {
                    info!(%peer, ?conn_id, "DCUtR: direct connection established (hole punch success)");
                }
                Err(ref e) => {
                    warn!(%peer, error = %e, "DCUtR: hole punch failed, staying on relay");
                }
            }
        }

        // ── Connection lifecycle ──
        SwarmEvent::NewListenAddr { address, .. } => {
            info!(%address, "listening on");
        }
        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
            debug!(%peer_id, "connection established");
        }
        SwarmEvent::ConnectionClosed { peer_id, .. } => {
            debug!(%peer_id, "connection closed");
        }
        SwarmEvent::ExternalAddrConfirmed { address } => {
            info!(%address, "external address confirmed");
            state.external_addrs.push(address);
            // Update NAT status — if we have a confirmed external addr, we're public.
            state.nat_info.is_public = true;
            state.nat_info.nat_type = "open".into();
            if let Some(ip) = extract_ip_from_multiaddr(&state.external_addrs.last().unwrap()) {
                state.nat_info.external_ip = ip;
            }
        }
        _ => {}
    }
}

/// Handle Kademlia events.
fn handle_kad_event(event: kad::Event, state: &mut LoopState) {
    match event {
        kad::Event::OutboundQueryProgressed {
            id,
            result: kad::QueryResult::GetRecord(result),
            ..
        } => {
            match result {
                Ok(kad::GetRecordOk::FoundRecord(kad::PeerRecord { record, .. })) => {
                    // Decode the record and add to pending discover results.
                    if let Some(pending) = state.pending_discovers.get_mut(&id) {
                        match dht::decode_record(&record.value) {
                            Ok(peer_record) => {
                                // Cache the peer.
                                state
                                    .known_peers
                                    .insert(peer_record.peer_id.clone(), peer_record.clone());
                                pending.records.push(peer_record);
                            }
                            Err(e) => {
                                warn!("failed to decode DHT record: {e}");
                            }
                        }
                    }
                }
                Ok(kad::GetRecordOk::FinishedWithNoAdditionalRecord { .. }) => {
                    // Query complete — send results back.
                    if let Some(pending) = state.pending_discovers.remove(&id) {
                        let peers = pending
                            .records
                            .into_iter()
                            .map(|r| record_to_discovered(&r))
                            .collect();
                        let _ = pending.reply.send(Ok(peers));
                    }
                }
                Err(e) => {
                    if let Some(pending) = state.pending_discovers.remove(&id) {
                        let _ = pending.reply.send(Err(format!("kademlia get_record: {e:?}")));
                    }
                }
            }
        }
        kad::Event::OutboundQueryProgressed {
            result: kad::QueryResult::PutRecord(result),
            ..
        } => {
            match result {
                Ok(kad::PutRecordOk { .. }) => {
                    debug!("kademlia put_record succeeded");
                }
                Err(e) => {
                    warn!("kademlia put_record failed: {e:?}");
                }
            }
        }
        kad::Event::RoutingUpdated { peer, .. } => {
            debug!(%peer, "kademlia routing updated");
        }
        _ => {}
    }
}

/// Handle AutoNAT events.
fn handle_autonat_event(event: libp2p::autonat::Event, state: &mut LoopState) {
    match event {
        libp2p::autonat::Event::StatusChanged { old, new } => {
            info!(?old, ?new, "AutoNAT status changed");
            match new {
                libp2p::autonat::NatStatus::Public(addr) => {
                    state.nat_info.nat_type = "open".into();
                    state.nat_info.is_public = true;
                    if let Some(ip) = extract_ip_from_multiaddr(&addr) {
                        state.nat_info.external_ip = ip;
                    }
                }
                libp2p::autonat::NatStatus::Private => {
                    state.nat_info.nat_type = "symmetric".into();
                    state.nat_info.is_public = false;
                }
                libp2p::autonat::NatStatus::Unknown => {
                    state.nat_info.nat_type = "unknown".into();
                    state.nat_info.is_public = false;
                }
            }
        }
        _ => {}
    }
}

/// Convert a PeerRecord into a DiscoveredPeer.
fn record_to_discovered(r: &PeerRecord) -> DiscoveredPeer {
    let reachable_address = if r.requires_relay && !r.relay_address.is_empty() {
        r.relay_address.clone()
    } else {
        format!("{}:{}", r.host, r.port)
    };
    DiscoveredPeer {
        peer_id: r.peer_id.clone(),
        libp2p_peer_id: r.libp2p_peer_id.clone(),
        host: r.host.clone(),
        port: r.port,
        model_id: r.model_id.clone(),
        layer_start: r.layer_start,
        layer_end: r.layer_end,
        total_layers: r.total_layers,
        nat_type: r.nat_type.clone(),
        requires_relay: r.requires_relay,
        relay_address: r.relay_address.clone(),
        runtime_backend: r.runtime_backend.clone(),
        runtime_model_id: r.runtime_model_id.clone(),
        reachable_address,
    }
}

/// Extract an IP string from a multiaddr like `/ip4/1.2.3.4/tcp/4001`.
fn extract_ip_from_multiaddr(addr: &Multiaddr) -> Option<String> {
    for proto in addr.iter() {
        match proto {
            libp2p::multiaddr::Protocol::Ip4(ip) => return Some(ip.to_string()),
            libp2p::multiaddr::Protocol::Ip6(ip) => return Some(ip.to_string()),
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_to_discovered_direct() {
        let r = PeerRecord {
            peer_id: "mac-a".into(),
            model_id: "qwen".into(),
            host: "192.168.1.10".into(),
            port: 50051,
            layer_start: 0,
            layer_end: 12,
            total_layers: 24,
            requires_relay: false,
            ..Default::default()
        };
        let d = record_to_discovered(&r);
        assert_eq!(d.reachable_address, "192.168.1.10:50051");
        assert_eq!(d.layer_start, 0);
        assert_eq!(d.layer_end, 12);
    }

    #[test]
    fn test_record_to_discovered_relay() {
        let r = PeerRecord {
            peer_id: "mac-b".into(),
            model_id: "qwen".into(),
            host: "10.0.0.5".into(),
            port: 50051,
            requires_relay: true,
            relay_address: "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooW.../p2p-circuit".into(),
            ..Default::default()
        };
        let d = record_to_discovered(&r);
        assert_eq!(d.reachable_address, r.relay_address);
        assert!(d.requires_relay);
    }

    #[test]
    fn test_extract_ip() {
        let addr: Multiaddr = "/ip4/1.2.3.4/tcp/4001".parse().unwrap();
        assert_eq!(extract_ip_from_multiaddr(&addr), Some("1.2.3.4".into()));
    }
}
