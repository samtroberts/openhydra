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
    /// Fire-and-forget variant of ProxyForward — sends raw bytes to a peer
    /// via libp2p but does NOT block for an ACK/response. The sender
    /// returns immediately after enqueuing; the response (if any) is
    /// silently discarded when it arrives. Used by cross-ISP push mode
    /// to eliminate the ~200ms synchronous ACK wait per token.
    ProxyForwardNoWait {
        peer_id: String,
        data: Vec<u8>,
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
    /// Check if a peer is currently connected.
    IsConnected {
        peer_id: String,
        reply: oneshot::Sender<bool>,
    },
    /// Snapshot of DCUtR hole-punch counters (PR-2).
    /// Returns `(successes, failures, direct_peers_count)`.
    GetDcutrStats {
        reply: oneshot::Sender<(u64, u64, u64)>,
    },
    /// Publish a raw bytes payload on the Gossipsub topic
    /// ``openhydra/swarm/v1/events`` (PR-3 / B1). The Python
    /// ``GossipClient`` is responsible for the JSON codec and
    /// event-type semantics.
    PublishEvent {
        payload: Vec<u8>,
        reply: oneshot::Sender<Result<(), String>>,
    },
    /// Issue an active ``Dial`` to the given libp2p peer id (B1 rendezvous).
    /// The dial is fire-and-forget at this layer — success / failure is
    /// surfaced via the usual ``ConnectionEstablished`` / ``DialFailure``
    /// swarm events that already drive our direct-peers set and DCUtR
    /// counters.
    ///
    /// Returns ``Ok(())`` when the dial was successfully enqueued and
    /// ``Err`` with a short reason when the peer id was malformed or the
    /// dial slot couldn't be acquired.
    DialPeer {
        peer_id: String,
        reply: oneshot::Sender<Result<(), String>>,
    },
    /// Explicitly populate Kademlia's routing table with ``(peer_id, multiaddr)``.
    ///
    /// Bridges a known gap in the libp2p Kademlia behaviour: ``discover()``
    /// returns ``DiscoveredPeer`` records with ``relay_address`` strings
    /// to the Python caller, but does NOT automatically add those addresses
    /// to the Swarm's per-peer address book. Without this, ``dial_peer``
    /// fails with "no addresses for peer" even when Kademlia has just
    /// returned a valid multiaddr for that same peer_id.
    ///
    /// This command lets Python explicitly call
    /// ``swarm.behaviour_mut().kademlia.add_address(&pid, ma)`` after each
    /// ``discover()`` call — closing the gap that blocked the 2026-04-24
    /// True Petals cross-VPC benchmark (Mac coordinator dialing GPU2
    /// through a Linode relay).
    ///
    /// Returns ``Ok(())`` on successful enqueue. Reports a short error
    /// on peer_id or multiaddr parse failure so the Python side can log
    /// it and continue with the remaining peers.
    AddAddress {
        peer_id: String,
        multiaddr: String,
        reply: oneshot::Sender<Result<(), String>>,
    },
    /// Drain the oldest queued inbound gossip message, if any (PR-3 / B1).
    /// Returns ``None`` when the queue is empty. The returned tuple is
    /// ``(sender_libp2p_peer_id, payload_bytes)`` — callers that need the
    /// sender identity for the ``PEER_DEAD`` 2-observer quorum can read
    /// it directly off the gossip hop rather than trusting an embedded
    /// claim inside the JSON.
    PollEvent {
        reply: oneshot::Sender<Option<(String, Vec<u8>)>>,
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
    /// Proxy forward requests waiting for a relay connection to be established.
    /// (target_peer_id, data, reply_channel)
    pending_relay_forwards: Vec<(PeerId, Vec<u8>, oneshot::Sender<Result<Vec<u8>, String>>)>,
    /// DCUtR hole punch counters.
    dcutr_successes: u64,
    dcutr_failures: u64,
    /// Peers with confirmed direct (non-relayed) connections.
    /// Maps peer → count of active non-circuit connections.  When a
    /// non-circuit ConnectionEstablished fires the count increments;
    /// when ConnectionClosed fires for a non-circuit endpoint the count
    /// decrements.  `is_direct(peer)` ↔ `count > 0`.  This prevents a
    /// race where a failed DCUtR attempt briefly opens a direct
    /// connection, adds the peer, then closes — but the relay
    /// connection keeps `swarm.is_connected()` true, leaving the peer
    /// permanently mis-classified as "direct."
    direct_peers: HashMap<PeerId, u32>,
    /// Reply channels for local proxy forwards (Ouroboros: target == self).
    /// When respond_proxy is called with a "proxy-local-*" req_id, the response
    /// is delivered here instead of through libp2p.
    local_proxy_replies: HashMap<String, oneshot::Sender<Result<Vec<u8>, String>>>,
    /// PR-3 (B1): inbound gossip messages awaiting Python poll. Each entry
    /// is ``(sender_libp2p_peer_id, payload_bytes)``. Bounded ring — the
    /// Rust side drops the oldest when the queue exceeds
    /// ``GOSSIP_INBOUND_QUEUE_MAX`` to prevent unbounded memory growth
    /// when Python is slow to poll.
    gossip_inbound_queue: std::collections::VecDeque<(String, Vec<u8>)>,
}

/// PR-3: upper bound on pending inbound gossip messages.
/// The swarm-wide event rate is tiny (one ``PEER_DEAD`` per real failure,
/// plus the occasional ``REQUEST_HOLE_PUNCH``) so a soft cap of 256 is
/// roughly an hour of breathing room before oldest-drop kicks in.
const GOSSIP_INBOUND_QUEUE_MAX: usize = 256;

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
            pending_relay_forwards: Vec::new(),
            dcutr_successes: 0,
            dcutr_failures: 0,
            direct_peers: HashMap::new(),
            local_proxy_replies: HashMap::new(),
            gossip_inbound_queue: std::collections::VecDeque::new(),
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

    // Relay reservations are requested after a short delay (see below)
    // to ensure Kademlia has connected to the bootstrap peers first.
    // The relay client behaviour sends the reservation request on an
    // existing connection — without one, it dials via TCP which doesn't
    // install the relay reservation handler.
    let mut relay_reservation_pending = true;
    let relay_reservation_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);

    loop {
        // Delayed relay reservation: wait for Kademlia to connect to
        // bootstrap peers, then request relay reservations via listen_on.
        if relay_reservation_pending && tokio::time::Instant::now() >= relay_reservation_deadline {
            relay_reservation_pending = false;
            for relay_str in crate::relay::BOOTSTRAP_RELAYS {
                if let Ok(relay_multiaddr) = relay_str.parse::<Multiaddr>() {
                    let listen_addr = relay_multiaddr
                        .with(libp2p::multiaddr::Protocol::P2pCircuit);
                    match swarm.listen_on(listen_addr.clone()) {
                        Ok(_) => {
                            info!(addr = %listen_addr, "listening via relay (reservation requested)");
                        }
                        Err(e) => {
                            warn!(addr = %listen_addr, error = %e, "relay listen failed");
                        }
                    }
                }
            }
        }

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
                    Some(SwarmCommand::ProxyForwardNoWait { peer_id, data }) => {
                        handle_proxy_forward_no_wait(&mut swarm, &peer_id, data, &mut state);
                    }
                    Some(SwarmCommand::IsConnected { peer_id, reply }) => {
                        // Returns true ONLY if we have a direct (non-relayed) connection.
                        // Used by Python push mode to decide direct gRPC vs relay proxy.
                        let has_direct = match peer_id.parse::<PeerId>() {
                            Ok(pid) => state.direct_peers.get(&pid).copied().unwrap_or(0) > 0,
                            Err(_) => false,
                        };
                        let _ = reply.send(has_direct);
                    }
                    Some(SwarmCommand::GetDcutrStats { reply }) => {
                        let snapshot = (
                            state.dcutr_successes,
                            state.dcutr_failures,
                            state.direct_peers.len() as u64,
                        );
                        let _ = reply.send(snapshot);
                    }
                    Some(SwarmCommand::PublishEvent { payload, reply }) => {
                        // PR-3: publish raw bytes on the Gossipsub topic.
                        // The Python side has already JSON-encoded the
                        // message and decided on event_type semantics.
                        let topic = libp2p::gossipsub::IdentTopic::new(
                            crate::swarm::GOSSIPSUB_TOPIC,
                        );
                        let res = swarm
                            .behaviour_mut()
                            .gossipsub
                            .publish(topic, payload)
                            .map(|_| ())
                            .map_err(|e| format!("gossipsub publish: {e}"));
                        let _ = reply.send(res);
                    }
                    Some(SwarmCommand::PollEvent { reply }) => {
                        let item = state.gossip_inbound_queue.pop_front();
                        let _ = reply.send(item);
                    }
                    Some(SwarmCommand::DialPeer { peer_id, reply }) => {
                        // B1 rendezvous: enqueue an active dial to the
                        // peer id carried on a REQUEST_HOLE_PUNCH gossip
                        // event. Because libp2p already knows the peer's
                        // candidate addresses (via Kademlia / Identify
                        // / direct registration), we don't need to pass
                        // multiaddrs — just the PeerId. The resulting
                        // simultaneous dial from *both* sides is what
                        // forces DCUtR hole-punch against symmetric NAT.
                        // libp2p's default ``DialOpts`` uses
                        // ``PeerCondition::Disconnected`` which *rejects* a
                        // new dial when the peer is already connected via
                        // relay — precisely the state we're in when a
                        // REQUEST_HOLE_PUNCH gossip event fires. Force
                        // ``PeerCondition::Always`` so the dial is enqueued
                        // anyway. The libp2p transport stack will attempt
                        // an upgrade through DCUtR once the simultaneous
                        // connect from both sides lands, promoting the
                        // relayed connection into a direct one.
                        use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
                        let res = match peer_id.parse::<PeerId>() {
                            Ok(pid) => {
                                let opts = DialOpts::peer_id(pid)
                                    .condition(PeerCondition::Always)
                                    .build();
                                match swarm.dial(opts) {
                                    Ok(()) => {
                                        info!(%pid, "b1_hole_punch_dial_issued");
                                        Ok(())
                                    }
                                    Err(e) => Err(format!("dial error: {e}")),
                                }
                            }
                            Err(e) => Err(format!("invalid peer_id: {e}")),
                        };
                        let _ = reply.send(res);
                    }
                    Some(SwarmCommand::AddAddress { peer_id, multiaddr, reply }) => {
                        // Feed the ``(peer_id, multiaddr)`` pair into Kademlia's
                        // routing table so a subsequent ``DialPeer`` /
                        // ``ProxyForward`` can find a dialable address for this
                        // peer. The libp2p Swarm consults Kademlia during
                        // dial-address resolution, so this one API call closes
                        // the "no addresses for peer" dial failure surfaced in
                        // the 2026-04-24 cross-VPC benchmark.
                        let res = match (peer_id.parse::<PeerId>(), multiaddr.parse::<Multiaddr>()) {
                            (Ok(pid), Ok(ma)) => {
                                let update = swarm.behaviour_mut().kademlia.add_address(&pid, ma.clone());
                                info!(%pid, %ma, ?update, "add_address_applied");
                                Ok(())
                            }
                            (Err(e), _) => Err(format!("invalid peer_id: {e}")),
                            (_, Err(e)) => Err(format!("invalid multiaddr: {e}")),
                        };
                        let _ = reply.send(res);
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
                        // Check local proxy replies first (Ouroboros: self-targeted forwards).
                        if let Some(reply) = state.local_proxy_replies.remove(&request_id) {
                            let _ = reply.send(Ok(data));
                        } else if let Some(channel) = state.inbound_proxy_channels.remove(&request_id) {
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
            handle_kad_event(kad_event, swarm, state);
        }

        // ── AutoNAT ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Autonat(autonat_event)) => {
            handle_autonat_event(autonat_event, state);
        }

        // ── Identify ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Identify(identify_event)) => {
            if let libp2p::identify::Event::Received { peer_id, info, .. } = identify_event {
                debug!(%peer_id, protocol = %info.protocol_version, "identify received");
                // Add the remote peer's listen addresses to Kademlia.
                for addr in &info.listen_addrs {
                    swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr.clone());
                }
                // Critical for DCUtR: register the observed address (our NAT
                // external mapping) as an external address. Without this, DCUtR
                // has no candidate addresses for hole punching.
                if !info.observed_addr.to_string().is_empty() {
                    debug!(addr = %info.observed_addr, "adding observed addr as external");
                    swarm.add_external_address(info.observed_addr);
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
                    state.dcutr_successes += 1;
                    // DCUtR success: mark as direct.  The actual
                    // ConnectionEstablished for the hole-punched
                    // connection will also bump the count; using
                    // max(1, current) here ensures the peer is always
                    // marked direct after a successful DCUtR even if
                    // the event ordering varies.
                    let cnt = state.direct_peers.entry(peer).or_insert(0);
                    *cnt = (*cnt).max(1);
                    info!(
                        %peer, ?conn_id,
                        successes = state.dcutr_successes,
                        failures = state.dcutr_failures,
                        "DCUtR: direct connection established (hole punch success)"
                    );
                }
                Err(ref e) => {
                    state.dcutr_failures += 1;
                    warn!(
                        %peer, error = %e,
                        successes = state.dcutr_successes,
                        failures = state.dcutr_failures,
                        "DCUtR: hole punch failed, staying on relay"
                    );
                }
            }
        }

        // ── gRPC Proxy ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::GrpcProxy(proxy_event)) => {
            handle_grpc_proxy_event(proxy_event, swarm, state);
        }

        // ── Gossipsub (PR-3 / B1) ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Gossipsub(gossip_event)) => {
            if let libp2p::gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            } = gossip_event
            {
                // Queue the payload for Python to poll. ``propagation_source``
                // is the immediate gossip hop (NOT necessarily the original
                // author) — we surface it so Python can build a 2-observer
                // quorum from distinct hop sources when needed.
                if state.gossip_inbound_queue.len() >= GOSSIP_INBOUND_QUEUE_MAX {
                    state.gossip_inbound_queue.pop_front();
                    warn!("gossipsub_queue_overflow: dropped oldest message");
                }
                state
                    .gossip_inbound_queue
                    .push_back((propagation_source.to_string(), message.data));
            }
        }

        // Ping keepalive — log only failures (success is silent to avoid
        // flooding logs every 15 s per connection).
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Ping(ping_event)) => {
            if let Err(ref e) = ping_event.result {
                debug!(peer = %ping_event.peer, error = %e, "ping failed");
            }
        }

        // ── Connection lifecycle ──
        SwarmEvent::NewListenAddr { address, .. } => {
            info!(%address, "listening on");
            // A3 DCUtR fix: register non-loopback, non-wildcard, non-circuit
            // listen addresses as external address candidates *before* the
            // relay reservations come in (relay reservations fire 5s after
            // startup — see relay_reservation_deadline at the top of the
            // event loop). This gives DCUtR a pool of real direct addresses
            // to offer during hole-punch negotiation instead of only the
            // ``/p2p-circuit/`` multiaddrs that Identify would otherwise
            // observe once the peer is relay-bound.
            //
            // Safety: AutoNAT will probe each candidate; truly unreachable
            // LAN addresses get falsified and only contribute to the
            // DCUtR candidate set for same-LAN peers (where they *are* the
            // right answer). Reachable public addresses get confirmed via
            // ``ExternalAddrConfirmed`` and light up the DCUtR hot path.
            if is_direct_listen_candidate(&address) {
                debug!(%address, "registering direct listen addr as external");
                swarm.add_external_address(address.clone());
            }
        }
        SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
            debug!(%peer_id, ?endpoint, "connection established");
            // Track direct (non-relay) connections for push mode routing.
            //
            // A connection is "direct" only if:
            //   1. The endpoint address does NOT contain /p2p-circuit/
            //   2. The endpoint IP is NOT a known bootstrap relay IP
            //
            // Guard #2 fixes a false-positive where relay-upgraded
            // connections report the relay server's raw IP without the
            // /p2p-circuit/ suffix, causing peers to be mis-classified
            // as "direct" when they're actually relay-bound.
            let addr_str = match &endpoint {
                libp2p::core::ConnectedPoint::Dialer { address, .. } => address.to_string(),
                libp2p::core::ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr.to_string(),
            };
            let has_circuit = addr_str.contains("p2p-circuit");
            let endpoint_ip = extract_ip_from_multiaddr_str(&addr_str);
            let is_relay_ip = endpoint_ip
                .as_ref()
                .map(|ip| crate::relay::is_bootstrap_relay_ip(ip))
                .unwrap_or(false);
            if !has_circuit && !is_relay_ip {
                *state.direct_peers.entry(peer_id).or_insert(0) += 1;
                info!(%peer_id, %addr_str, count = state.direct_peers[&peer_id], "direct_peer_added");
            } else {
                debug!(
                    %peer_id, %addr_str, has_circuit, is_relay_ip,
                    "connection_established (not marking direct)"
                );
            }
            // Send any queued proxy forwards that were waiting for this connection.
            let mut remaining = Vec::new();
            for (target, data, reply) in state.pending_relay_forwards.drain(..) {
                if target == peer_id {
                    info!(%peer_id, "sending queued proxy forward after relay connection");
                    let req_id = swarm
                        .behaviour_mut()
                        .grpc_proxy
                        .send_request(&peer_id, ProxyRequest(data));
                    state.pending_proxy.insert(req_id, reply);
                } else {
                    remaining.push((target, data, reply));
                }
            }
            state.pending_relay_forwards = remaining;
        }
        SwarmEvent::ConnectionClosed { peer_id, endpoint, .. } => {
            debug!(%peer_id, "connection closed");
            // Decrement direct connection count — mirror the same
            // classification logic used in ConnectionEstablished.
            let addr_str = match &endpoint {
                libp2p::core::ConnectedPoint::Dialer { address, .. } => address.to_string(),
                libp2p::core::ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr.to_string(),
            };
            let has_circuit = addr_str.contains("p2p-circuit");
            let endpoint_ip = extract_ip_from_multiaddr_str(&addr_str);
            let is_relay_ip = endpoint_ip
                .as_ref()
                .map(|ip| crate::relay::is_bootstrap_relay_ip(ip))
                .unwrap_or(false);
            if !has_circuit && !is_relay_ip {
                if let Some(cnt) = state.direct_peers.get_mut(&peer_id) {
                    *cnt = cnt.saturating_sub(1);
                    if *cnt == 0 {
                        state.direct_peers.remove(&peer_id);
                        info!(%peer_id, "direct_peer_removed (last direct conn closed)");
                    }
                }
            }
            // Also clean up if the peer is completely disconnected.
            if !swarm.is_connected(&peer_id) {
                state.direct_peers.remove(&peer_id);
            }
        }
        SwarmEvent::ExternalAddrConfirmed { address } => {
            info!(%address, "external address confirmed");
            let is_circuit = address.to_string().contains("/p2p-circuit");
            state.external_addrs.push(address.clone());
            // A3 DCUtR fix: only flip the peer to ``is_public`` when the
            // confirmed address is a *direct* multiaddr. A ``/p2p-circuit/``
            // confirmation means "a relay forwarded traffic for us" — not
            // "we are publicly reachable". Previously any confirmation
            // (including circuit) marked the peer public, which suppressed
            // AutoNAT's Private verdict and kept DCUtR dormant.
            if !is_circuit {
                state.nat_info.is_public = true;
                state.nat_info.nat_type = "open".into();
                if let Some(ip) = extract_ip_from_multiaddr(&address) {
                    state.nat_info.external_ip = ip;
                }
            } else {
                // Record the relay path for observability but leave NAT
                // status untouched so AutoNAT probes continue to drive the
                // public/private classification.
                debug!(
                    "circuit external address recorded but not marking public"
                );
            }
        }
        _ => {}
    }
}

/// Handle Kademlia events.
fn handle_kad_event(
    event: kad::Event,
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
) {
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
                                // Auto-populate Kademlia's routing table with
                                // the peer's advertised relay_address so a
                                // subsequent ``dial_peer`` / ``proxy_forward``
                                // can find a dialable multiaddr. Without this,
                                // ``discover()`` returns records to Python
                                // but the Swarm's address book stays empty
                                // (surfaced as "no addresses for peer" in the
                                // 2026-04-24 cross-VPC benchmark).
                                //
                                // The relay_address may be empty (peer is
                                // publicly reachable and didn't advertise a
                                // circuit), in which case we skip — the
                                // direct host:port dial will be attempted by
                                // the gRPC layer instead.
                                if !peer_record.relay_address.is_empty()
                                    && !peer_record.libp2p_peer_id.is_empty()
                                {
                                    match (
                                        peer_record.libp2p_peer_id.parse::<PeerId>(),
                                        peer_record.relay_address.parse::<Multiaddr>(),
                                    ) {
                                        (Ok(pid), Ok(ma)) => {
                                            let update = swarm
                                                .behaviour_mut()
                                                .kademlia
                                                .add_address(&pid, ma.clone());
                                            debug!(
                                                %pid, %ma, ?update,
                                                "discover_auto_added_address"
                                            );
                                        }
                                        (Err(e), _) => {
                                            warn!(
                                                "discover: invalid libp2p_peer_id in record: {e}"
                                            );
                                        }
                                        (_, Err(e)) => {
                                            warn!(
                                                "discover: invalid relay_address in record: {e}"
                                            );
                                        }
                                    }
                                }
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

/// A3 DCUtR fix: decide whether a newly-bound listen address is a sensible
/// DCUtR external candidate.
///
/// Returns ``false`` for:
/// * ``/p2p-circuit/`` multiaddrs (relay-originated, useless for hole punching)
/// * loopback IPs (``127.0.0.0/8``, ``::1``)
/// * unspecified / wildcard IPs (``0.0.0.0``, ``::``)
///
/// Everything else — LAN / ULA / public — is returned as a candidate.
/// AutoNAT then probes the candidates; unreachable ones get falsified and
/// only contribute within the LAN scope where they're valid.
fn is_direct_listen_candidate(addr: &Multiaddr) -> bool {
    if addr.to_string().contains("/p2p-circuit") {
        return false;
    }
    for proto in addr.iter() {
        match proto {
            libp2p::multiaddr::Protocol::Ip4(ip) => {
                if ip.is_loopback() || ip.is_unspecified() {
                    return false;
                }
            }
            libp2p::multiaddr::Protocol::Ip6(ip) => {
                if ip.is_loopback() || ip.is_unspecified() {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
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

/// Extract an IP string from a multiaddr string representation.
/// Parses `/ip4/1.2.3.4/...` or `/ip6/::1/...` without requiring
/// a full Multiaddr parse (which can fail on partial addresses).
fn extract_ip_from_multiaddr_str(addr_str: &str) -> Option<String> {
    for segment in addr_str.split('/') {
        // The IP follows "/ip4/" or "/ip6/" — grab the next segment.
        if segment == "ip4" || segment == "ip6" {
            continue;
        }
        // Check if this segment looks like an IPv4 address.
        if segment.contains('.') && segment.chars().all(|c| c.is_ascii_digit() || c == '.') {
            return Some(segment.to_string());
        }
        // Check if this looks like an IPv6 address (contains colons).
        if segment.contains(':') && !segment.contains("p2p") {
            return Some(segment.to_string());
        }
    }
    // Fallback: try parsing as full multiaddr.
    if let Ok(addr) = addr_str.parse::<Multiaddr>() {
        return extract_ip_from_multiaddr(&addr);
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
///
/// If the peer isn't directly connected, initiates a relay circuit dial
/// and queues the request to be sent after the connection is established.
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

    // Ouroboros guard: if the target is our own peer ID, route locally.
    // Queue as inbound proxy request and store the reply channel so
    // respond_proxy() can deliver the response back to the caller.
    if peer_id == *swarm.local_peer_id() {
        info!("proxy_forward: target is self — routing locally");
        state.inbound_proxy_counter += 1;
        let req_id = format!("proxy-local-{}", state.inbound_proxy_counter);
        state.inbound_proxy_queue.push((req_id.clone(), data));
        state.local_proxy_replies.insert(req_id, reply);
        return;
    }

    if swarm.is_connected(&peer_id) {
        // Already connected — send immediately.
        let req_id = swarm
            .behaviour_mut()
            .grpc_proxy
            .send_request(&peer_id, ProxyRequest(data));
        state.pending_proxy.insert(req_id, reply);
    } else {
        // Not connected — dial through relay and queue the request.
        // It will be sent when ConnectionEstablished fires.
        info!(%peer_id, "proxy_forward: peer not connected, dialing via relay");
        let mut dialed = false;
        for relay_str in crate::relay::BOOTSTRAP_RELAYS {
            if let Ok(relay_multiaddr) = relay_str.parse::<Multiaddr>() {
                let circuit_addr = relay_multiaddr
                    .with(libp2p::multiaddr::Protocol::P2pCircuit)
                    .with(libp2p::multiaddr::Protocol::P2p(peer_id));
                info!(%peer_id, addr = %circuit_addr, "dialing peer through relay");
                match swarm.dial(circuit_addr) {
                    Ok(_) => {
                        dialed = true;
                        break;
                    }
                    Err(e) => {
                        warn!(%peer_id, error=%e, "relay dial failed, trying next");
                    }
                }
            }
        }
        if dialed {
            // Queue the request to be sent after connection is established.
            state.pending_relay_forwards.push((peer_id, data, reply));
        } else {
            let _ = reply.send(Err("proxy_forward: no relay dial succeeded".into()));
        }
    }
}

/// Fire-and-forget variant of handle_proxy_forward.
///
/// Sends data to a peer via request_response but does NOT store a reply
/// channel in `pending_proxy`. When the response arrives, the
/// `pending_proxy.remove()` call in the response handler returns `None`
/// and the response is silently discarded — exactly the desired behaviour
/// for fire-and-forget cross-ISP push mode.
///
/// For Ouroboros (self-targeted) forwards: queues in `inbound_proxy_queue`
/// without a `local_proxy_replies` entry. The respond_proxy for the
/// "proxy-local-*" request will hit the `warn!("unknown request_id")`
/// branch — harmless; the Python caller doesn't expect a response.
///
/// For not-yet-connected peers: creates a dummy oneshot pair and pushes
/// into `pending_relay_forwards`. When the connection establishes and
/// the actual send happens, the reply goes to the dummy receiver which
/// has already been dropped — silently discarded.
fn handle_proxy_forward_no_wait(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    peer_id_str: &str,
    data: Vec<u8>,
    state: &mut LoopState,
) {
    let peer_id: PeerId = match peer_id_str.parse() {
        Ok(p) => p,
        Err(e) => {
            warn!("proxy_forward_no_wait: invalid peer_id: {e}");
            return;
        }
    };

    // Ouroboros guard: self-targeted forward — queue locally, no reply channel.
    if peer_id == *swarm.local_peer_id() {
        debug!("proxy_forward_no_wait: target is self — routing locally (no reply)");
        state.inbound_proxy_counter += 1;
        let req_id = format!("proxy-local-{}", state.inbound_proxy_counter);
        state.inbound_proxy_queue.push((req_id, data));
        // No local_proxy_replies entry — respond_proxy will warn and discard.
        return;
    }

    if swarm.is_connected(&peer_id) {
        // Already connected — send immediately, don't track reply.
        let _req_id = swarm
            .behaviour_mut()
            .grpc_proxy
            .send_request(&peer_id, ProxyRequest(data));
        // Deliberately NOT inserting into pending_proxy.
        // Response will be silently discarded when it arrives.
    } else {
        // Not connected — dial through relay and queue with a dummy reply channel.
        info!(%peer_id, "proxy_forward_no_wait: peer not connected, dialing via relay");
        let mut dialed = false;
        for relay_str in crate::relay::BOOTSTRAP_RELAYS {
            if let Ok(relay_multiaddr) = relay_str.parse::<Multiaddr>() {
                let circuit_addr = relay_multiaddr
                    .with(libp2p::multiaddr::Protocol::P2pCircuit)
                    .with(libp2p::multiaddr::Protocol::P2p(peer_id));
                match swarm.dial(circuit_addr) {
                    Ok(_) => {
                        dialed = true;
                        break;
                    }
                    Err(e) => {
                        warn!(%peer_id, error=%e, "no_wait relay dial failed, trying next");
                    }
                }
            }
        }
        if dialed {
            // Dummy oneshot — the receiver is dropped immediately so the
            // response will be silently discarded when it arrives.
            let (dummy_tx, _dummy_rx) = oneshot::channel();
            state.pending_relay_forwards.push((peer_id, data, dummy_tx));
        } else {
            warn!(%peer_id, "proxy_forward_no_wait: no relay dial succeeded — data dropped");
        }
    }
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

    #[test]
    fn test_extract_ip_from_multiaddr_str() {
        assert_eq!(
            extract_ip_from_multiaddr_str("/ip4/45.79.190.172/tcp/4001"),
            Some("45.79.190.172".into()),
        );
        assert_eq!(
            extract_ip_from_multiaddr_str("/ip4/192.168.1.11/tcp/4001"),
            Some("192.168.1.11".into()),
        );
        assert_eq!(
            extract_ip_from_multiaddr_str(
                "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL/p2p-circuit/p2p/12D3KooW9xM53"
            ),
            Some("45.79.190.172".into()),
        );
    }

    #[test]
    fn test_relay_ip_detection() {
        // Bootstrap relay IPs should be detected.
        assert!(crate::relay::is_bootstrap_relay_ip("45.79.190.172"));
        assert!(crate::relay::is_bootstrap_relay_ip("172.105.69.49"));
        assert!(crate::relay::is_bootstrap_relay_ip("172.104.164.98"));
        // Non-relay IPs should not.
        assert!(!crate::relay::is_bootstrap_relay_ip("192.168.1.11"));
        assert!(!crate::relay::is_bootstrap_relay_ip("10.192.11.51"));
    }
}
