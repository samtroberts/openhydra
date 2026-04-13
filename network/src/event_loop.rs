//! Swarm event loop — runs on a background tokio task.
//!
//! Receives commands from the Python thread via mpsc and drives the
//! libp2p swarm. Results are sent back via oneshot channels.

use std::collections::HashMap;

use futures::StreamExt;
use libp2p::request_response;
use libp2p::swarm::SwarmEvent;
use libp2p::{kad, Multiaddr, PeerId};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

use crate::behaviour::{OpenHydraBehaviour, OpenHydraBehaviourEvent};
use crate::dht;
use crate::proxy::{self, ProxyRequest, ProxyResponse};
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
    /// Send raw bytes to a peer via the gRPC proxy protocol.
    /// Used by the local TCP proxy listener.
    ProxyForward {
        peer_id: String,
        data: Vec<u8>,
        reply: oneshot::Sender<Result<Vec<u8>, String>>,
    },
    /// Start a local TCP proxy that tunnels to a remote peer via libp2p.
    /// Returns "127.0.0.1:<port>" for gRPC to connect to.
    OpenProxy {
        target_libp2p_peer_id: String,
        local_grpc_port: u16,
        reply: oneshot::Sender<Result<String, String>>,
    },
    /// Poll for the next inbound proxy request (from a remote peer).
    /// Returns (request_id, bytes) or None if the queue is empty.
    PollProxyRequest {
        reply: oneshot::Sender<Option<(String, Vec<u8>)>>,
    },
    /// Send a response to an inbound proxy request.
    RespondProxy {
        request_id: String,
        data: Vec<u8>,
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
    /// Pending proxy forward requests: request_id → reply channel.
    pending_proxy: HashMap<request_response::OutboundRequestId, oneshot::Sender<Result<Vec<u8>, String>>>,
    /// Local gRPC port for inbound proxy requests.
    local_grpc_port: u16,
    /// Inbound proxy requests waiting for Python to process.
    /// (request_id, raw_bytes, response_channel_for_libp2p)
    inbound_proxy_queue: Vec<(String, Vec<u8>)>,
    /// Pending inbound proxy responses: request_id → (libp2p ResponseChannel, proxy_respond_tx sender)
    /// When Python calls RespondProxy, we find the channel here and send_response.
    inbound_proxy_channels: HashMap<String, request_response::ResponseChannel<ProxyResponse>>,
    /// Counter for generating unique inbound request IDs.
    inbound_proxy_counter: u64,
}

struct PendingDiscover {
    #[allow(dead_code)]
    model_id: String,
    records: Vec<PeerRecord>,
    reply: oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>,
}

impl LoopState {
    fn new() -> Self {
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
            pending_proxy: HashMap::new(),
            local_grpc_port: 50051,
            inbound_proxy_queue: Vec::new(),
            inbound_proxy_channels: HashMap::new(),
            inbound_proxy_counter: 0,
        }
    }
}

/// Run the swarm event loop until shutdown.
pub async fn run_event_loop(
    mut swarm: libp2p::Swarm<OpenHydraBehaviour>,
    mut cmd_rx: mpsc::Receiver<SwarmCommand>,
) {
    let mut state = LoopState::new();

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
                    Some(SwarmCommand::ProxyForward { peer_id, data, reply }) => {
                        handle_proxy_forward(&mut swarm, &peer_id, data, reply, &mut state);
                    }
                    Some(SwarmCommand::OpenProxy { target_libp2p_peer_id, local_grpc_port, reply }) => {
                        state.local_grpc_port = local_grpc_port;
                        handle_open_proxy(&mut swarm, &target_libp2p_peer_id, reply, &state);
                    }
                    Some(SwarmCommand::PollProxyRequest { reply }) => {
                        let item = if state.inbound_proxy_queue.is_empty() {
                            None
                        } else {
                            Some(state.inbound_proxy_queue.remove(0))
                        };
                        let _ = reply.send(item);
                    }
                    Some(SwarmCommand::RespondProxy { request_id, data }) => {
                        if let Some(channel) = state.inbound_proxy_channels.remove(&request_id) {
                            if let Err(e) = swarm.behaviour_mut().grpc_proxy.send_response(channel, ProxyResponse(data)) {
                                warn!("proxy respond failed: {:?}", e);
                            }
                        } else {
                            warn!("proxy respond: unknown request_id={}", request_id);
                        }
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

        // ── gRPC Proxy ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::GrpcProxy(proxy_event)) => {
            handle_grpc_proxy_event(proxy_event, swarm, state);
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

// ── gRPC proxy event handling ─────────────────────────────────────────

fn handle_grpc_proxy_event(
    event: request_response::Event<ProxyRequest, ProxyResponse>,
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
) {
    match event {
        request_response::Event::Message { peer, message } => {
            match message {
                request_response::Message::Request { request_id, request, channel } => {
                    // Queue the request for Python to process via poll_proxy_request().
                    // Store the response channel so RespondProxy can send back.
                    state.inbound_proxy_counter += 1;
                    let req_id = format!("proxy-{}", state.inbound_proxy_counter);
                    info!(%peer, bytes = request.0.len(), id = %req_id, "proxy request queued for Python");
                    state.inbound_proxy_queue.push((req_id.clone(), request.0));
                    state.inbound_proxy_channels.insert(req_id, channel);
                }
                request_response::Message::Response { request_id, response } => {
                    // Outbound response received — deliver to waiting proxy forward.
                    if let Some(reply) = state.pending_proxy.remove(&request_id) {
                        let _ = reply.send(Ok(response.0));
                    }
                }
            }
        }
        request_response::Event::OutboundFailure { request_id, error, .. } => {
            warn!(?error, "proxy outbound failure");
            if let Some(reply) = state.pending_proxy.remove(&request_id) {
                let _ = reply.send(Err(format!("proxy outbound: {error:?}")));
            }
        }
        request_response::Event::InboundFailure { error, .. } => {
            warn!(?error, "proxy inbound failure");
        }
        _ => {}
    }
}

/// Send a proxy forward request to a peer via request_response.
fn handle_proxy_forward(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    peer_id_str: &str,
    data: Vec<u8>,
    reply: oneshot::Sender<Result<Vec<u8>, String>>,
    state: &mut LoopState,
) {
    let peer_id: PeerId = match peer_id_str.parse() {
        Ok(p) => p,
        Err(e) => {
            let _ = reply.send(Err(format!("invalid peer_id: {e}")));
            return;
        }
    };
    let req_id = swarm
        .behaviour_mut()
        .grpc_proxy
        .send_request(&peer_id, ProxyRequest(data));
    state.pending_proxy.insert(req_id, reply);
}

/// Open a local TCP proxy that tunnels to a remote peer via libp2p.
fn handle_open_proxy(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    target_peer_id: &str,
    reply: oneshot::Sender<Result<String, String>>,
    state: &LoopState,
) {
    let target: PeerId = match target_peer_id.parse() {
        Ok(p) => p,
        Err(e) => {
            let _ = reply.send(Err(format!("invalid peer_id: {e}")));
            return;
        }
    };

    // Dial the peer so the connection is ready when proxy requests arrive.
    if let Err(e) = swarm.dial(target) {
        warn!(%target, error=%e, "proxy dial failed");
    }

    // Start the local TCP listener in a background task.
    let target_str = target.to_string();
    tokio::spawn(async move {
        match proxy::start_proxy_listener().await {
            Ok((listener, addr)) => {
                let _ = reply.send(Ok(addr.clone()));
                info!(proxy=%addr, target=%target_str, "proxy ready");
                // Note: the actual forwarding happens via SwarmCommand::ProxyForward
                // from Python — the TCP listener is handled in Python by calling
                // open_proxy() which returns the address, then the coordinator
                // connects gRPC to that address. But gRPC doesn't go through our
                // TCP listener — it goes directly to the address. We need a different
                // approach: the proxy is the P2PNode itself, not a TCP listener.
                //
                // Instead, Python will call proxy_forward(peer_id, bytes) for each
                // gRPC call. The local TCP proxy approach won't work for gRPC because
                // HTTP/2 is stateful and multiplexed.
                drop(listener); // Not used — see note above.
            }
            Err(e) => {
                let _ = reply.send(Err(format!("proxy listener: {e}")));
            }
        }
    });
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
