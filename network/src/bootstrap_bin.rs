//! Standalone bootstrap node binary for OpenHydra Linode servers.
//!
//! Runs:
//! - Kademlia DHT (bootstrap node mode)
//! - Circuit Relay v2 server (accepts relay reservations)
//! - AutoNAT server (responds to NAT probes)
//! - Identify (peer metadata exchange)
//!
//! Usage:
//! ```bash
//! openhydra-bootstrap \
//!     --identity /opt/openhydra/.libp2p_identity.key \
//!     --listen /ip4/0.0.0.0/tcp/4001
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use libp2p::swarm::Config as SwarmConfig;
use libp2p::{autonat, gossipsub, identify, kad, ping, relay, request_response, Multiaddr, Swarm, Transport};
use tracing::{debug, info, warn};

use openhydra_network::registry_proto::{RegistryCodec, RegistryReply};

// Re-use crate modules for identity and transport.
// Note: since this is a [[bin]], we import the library crate.
use openhydra_network::identity::Identity;
// WS-F F-6: shared leech-lockout policy (per-circuit byte budget + lockout).
use openhydra_network::relay::{
    LeechRateLimiter, LeechTable, MAX_CIRCUIT_BYTES_ERROR, PER_CIRCUIT_BUDGET_BYTES,
    unix_secs_now, wallclock_jitter_frac,
};

/// Bootstrap-specific behaviour — includes relay::Behaviour (server mode)
/// and DCUtR for advertising hole-punch support in Identify.
#[derive(libp2p::swarm::NetworkBehaviour)]
struct BootstrapBehaviour {
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
    relay_server: relay::Behaviour,
    /// R-DHT-11: AutoNAT **v2 server** (the only AutoNAT — v1 retired). Answers a
    /// peer's request to dial-back a *specific* address and report whether it's
    /// reachable — the reliable, per-address signal that drives the peer-side
    /// R-DHT-2 promotion.
    autonat_v2_server: autonat::v2::server::Behaviour,
    identify: identify::Behaviour,
    dcutr: libp2p::dcutr::Behaviour,
    /// Gossipsub (B1 rendezvous support).
    ///
    /// Bootstrap nodes subscribe to the same ``openhydra/swarm/v1/events``
    /// topic as peers so they can **forward** ``REQUEST_HOLE_PUNCH`` /
    /// ``PEER_DEAD`` messages between peers that don't have a direct
    /// libp2p connection to each other — the common case for two
    /// NATted peers whose only shared connection point is a Linode
    /// relay. Without this, peer A's publish never reaches peer B
    /// because neither is connected to anyone who'll forward the
    /// topic message.
    gossipsub: gossipsub::Behaviour,
    /// Phase 5.6: Ping detects stale connections — without it, dead
    /// connections persist for the full idle_connection_timeout (600s).
    ping: ping::Behaviour,
    /// C7: `/openhydra/registry/1.0.0` responder. Answers a consumer's "who serves model X?"
    /// query from the `ProviderRegistry` this bootstrap has retained. Inbound-only — a bootstrap
    /// answers registry queries but never issues them.
    registry_query: request_response::Behaviour<RegistryCodec>,
}

/// C7: registry entry TTL — a provider that stops re-announcing ages out. Matches the DHT
/// record TTL (~300 s); healthy providers re-announce well within this.
const REGISTRY_TTL_MS: u64 = 300_000;
/// How often to sweep expired registry entries.
const REGISTRY_REAP_SECS: u64 = 60;

/// Current Unix time in milliseconds (0 if the clock is before the epoch — never in practice).
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Init tracing.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Parse CLI args (minimal, no clap dependency).
    let args: Vec<String> = std::env::args().collect();
    let identity_path = parse_flag(&args, "--identity")
        .unwrap_or_else(|| "/opt/openhydra/.libp2p_identity.key".to_string());
    let listen_addrs: Vec<Multiaddr> = parse_flag_multi(&args, "--listen")
        .unwrap_or_else(|| vec![
            "/ip4/0.0.0.0/tcp/4001".parse().unwrap(),
            // QUIC (UDP) — enables UDP-based AutoNAT probing and DCUtR
            // hole punching. Without this, peers can only attempt TCP
            // hole punching which has ~5-10% success rate.
            "/ip4/0.0.0.0/udp/4001/quic-v1".parse().unwrap(),
            // IPv6 dual-stack — required for cross-ISP direct connections
            // where IPv6 may be the only routable path (e.g. Jio CGNAT).
            "/ip6/::/tcp/4001".parse().unwrap(),
            "/ip6/::/udp/4001/quic-v1".parse().unwrap(),
        ]);

    info!("loading identity from {identity_path}");
    let identity = Identity::load_or_create(&PathBuf::from(&identity_path))?;
    info!(
        peer_id = %identity.libp2p_peer_id,
        openhydra_id = %identity.openhydra_peer_id,
        "bootstrap node starting"
    );

    let peer_id = identity.libp2p_peer_id;
    let keypair = identity.keypair.clone();

    // Transport: QUIC + TCP (bootstrap is public, no relay needed)
    let tcp_transport = openhydra_network::transport::build_tcp_transport(&keypair)?;
    let quic_transport = openhydra_network::transport::build_quic_transport(&keypair)?;
    let transport = libp2p::core::transport::OrTransport::new(quic_transport, tcp_transport)
        .map(|either, _| match either {
            futures::future::Either::Left((pid, mux)) => (pid, libp2p::core::muxing::StreamMuxerBox::new(mux)),
            futures::future::Either::Right((pid, mux)) => (pid, libp2p::core::muxing::StreamMuxerBox::new(mux)),
        })
        .boxed();

    // Kademlia in server mode (bootstrap node).
    let mut kad_config = kad::Config::new(
        libp2p::StreamProtocol::new("/openhydra/kad/1.0.0"),
    );
    kad_config.set_query_timeout(Duration::from_secs(30));
    // Phase 7.5: Reduced from 600s to 300s to match peer nodes. The old
    // 600s TTL doubled the ghost window compared to peers.
    kad_config.set_record_ttl(Some(Duration::from_secs(300)));
    kad_config.set_provider_record_ttl(Some(Duration::from_secs(300)));
    kad_config.set_publication_interval(Some(Duration::from_secs(240)));

    let store = kad::store::MemoryStore::new(peer_id);
    let mut kademlia = kad::Behaviour::with_config(peer_id, store, kad_config);
    // Bootstrap nodes MUST be in server mode to accept GET_RECORD/PUT_RECORD
    // queries from peers. Without this, they may stay in client mode and
    // silently refuse DHT operations.
    kademlia.set_mode(Some(kad::Mode::Server));

    // Add other bootstrap peers to the routing table.
    let bootstrap_peers: Vec<Multiaddr> = parse_flag_multi(&args, "--peer")
        .unwrap_or_default();
    for peer_addr in &bootstrap_peers {
        // Extract PeerId from /p2p/ component.
        let remote_peer = peer_addr
            .iter()
            .find_map(|p| match p {
                libp2p::multiaddr::Protocol::P2p(id) => Some(id),
                _ => None,
            });
        if let Some(remote_id) = remote_peer {
            let base_addr: Multiaddr = peer_addr
                .iter()
                .filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_)))
                .collect();
            kademlia.add_address(&remote_id, base_addr);
            info!(%remote_id, "added bootstrap peer");
        }
    }

    // WS-F F-6: shared leech-lockout table. Updated below on a byte-cap
    // CircuitClosed; read by the LeechRateLimiter plugged into the relay config
    // so a peer that blows its per-circuit budget is denied new
    // reservations/circuits for a jittered 15-30 min window.
    let leech_table = Arc::new(Mutex::new(LeechTable::new()));

    // Ops knob: allow overriding the per-circuit byte budget at runtime via
    // OPENHYDRA_PER_CIRCUIT_BUDGET_BYTES (decimal bytes) without a rebuild.
    // Defaults to the compiled PER_CIRCUIT_BUDGET_BYTES (200 MB). Lowering it
    // is how the F-6 leech-lockout is validated against a live relay; unset
    // (or set to 0) to restore the production cap. Values <= 0 / unparseable
    // fall back to the default.
    let per_circuit_budget_bytes: u64 = std::env::var("OPENHYDRA_PER_CIRCUIT_BUDGET_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(PER_CIRCUIT_BUDGET_BYTES);
    if per_circuit_budget_bytes != PER_CIRCUIT_BUDGET_BYTES {
        info!(
            budget_bytes = per_circuit_budget_bytes,
            default_bytes = PER_CIRCUIT_BUDGET_BYTES,
            "per-circuit byte budget OVERRIDDEN via OPENHYDRA_PER_CIRCUIT_BUDGET_BYTES",
        );
    }

    // Relay server — accepts reservations from NATted peers.
    let relay_server = relay::Behaviour::new(peer_id, {
        let mut cfg = relay::Config {
            max_reservations: 256,
            max_circuits: 512,
            // Audit 4.1: 8 → 4. A free Ed25519 identity can hold up to this
            // many circuit slots; halving it doubles the identities an
            // attacker needs to squat all 512 slots. The single-host vector
            // is closed by the per-IP firewall caps in network_limits.sh.
            max_circuits_per_peer: 4,
            // Durations stay at 1h until F-6 client-side circuit MIGRATION
            // lands (lowering them without migration would cut long relayed
            // inferences mid-session — F-5 renewal alone isn't enough; the
            // active circuit itself must migrate). See SESSION_STATE F-6 #2/#3.
            reservation_duration: Duration::from_secs(3600),
            // F-6: per-circuit token budget (WS-F decision, ~25k tokens at
            // 8 KB/token). Generous for any real session; long sessions span
            // circuits via migration. Abuse beyond this triggers the leech
            // lockout below, so the higher ceiling no longer amplifies DoS.
            max_circuit_bytes: per_circuit_budget_bytes,
            max_circuit_duration: Duration::from_secs(3600),
            ..Default::default()
        };
        // F-6: deny reservations AND circuits from leech-locked peers. Two
        // limiters share one table (built-in max_* still bound congestion).
        cfg.reservation_rate_limiters
            .push(Box::new(LeechRateLimiter::new(leech_table.clone())));
        cfg.circuit_src_rate_limiters
            .push(Box::new(LeechRateLimiter::new(leech_table.clone())));
        cfg
    });

    // AutoNAT reporter — responds to NAT probes from peers.
    //
    // Identify.
    let identify = identify::Behaviour::new(
        identify::Config::new("openhydra/0.1.0".to_string(), keypair.public())
            .with_push_listen_addr_updates(true),
    );

    // DCUtR — advertises hole-punch support in Identify protocol list.
    let dcutr = libp2p::dcutr::Behaviour::new(peer_id);

    // Gossipsub forwarder — bootstraps subscribe to the swarm-wide topic so
    // peers that can only reach each other through a bootstrap still see
    // each other's events. The message-authenticity signing guarantees the
    // bootstrap can't forge events; it just propagates signed messages.
    // Small-swarm tuning identical to the peer side — critical so the
    // bootstrap forwards every published message to every topic peer,
    // not just the D-sized mesh slice.
    //
    // KNOWN LIMITATION: Gossipsub validates message signatures (Strict mode)
    // but does not validate message payload schemas. Malformed payloads are
    // forwarded as-is. Add payload validation when the message format stabilizes.
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(1))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .max_transmit_size(64 * 1024)
        .flood_publish(true)
        .mesh_outbound_min(1)
        .mesh_n_low(1)
        .mesh_n(3)
        .mesh_n_high(6)
        .build()
        .map_err(|e| format!("gossipsub config: {e}"))?;
    let mut gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(keypair.clone()),
        gossipsub_config,
    )
    .map_err(|e| format!("gossipsub behaviour: {e}"))?;
    let gossip_topic =
        gossipsub::IdentTopic::new(openhydra_network::swarm::GOSSIPSUB_TOPIC);
    gossipsub
        .subscribe(&gossip_topic)
        .map_err(|e| format!("gossipsub subscribe: {e:?}"))?;

    // Phase 5.6: Ping with 15s interval detects stale connections faster
    // than the idle_connection_timeout alone.
    let ping = ping::Behaviour::new(
        ping::Config::new().with_interval(Duration::from_secs(15)),
    );

    // R-DHT-11: AutoNAT v2 server (the only AutoNAT — v1 retired) — dials back
    // specific addresses on request so peers get reliable per-address
    // reachability verdicts.
    let autonat_v2_server = autonat::v2::server::Behaviour::default();

    // C7: registry responder — Inbound-only (this bootstrap answers "who serves model X?"
    // queries from the registry it retains; it never issues such queries itself).
    let registry_query = openhydra_network::registry_proto::registry_behaviour(
        request_response::ProtocolSupport::Inbound,
    );

    let behaviour = BootstrapBehaviour {
        kademlia,
        relay_server,
        autonat_v2_server,
        identify,
        dcutr,
        gossipsub,
        ping,
        registry_query,
    };

    // Phase 5.6: Reduced from 600s to 300s to match peer nodes and
    // reclaim dead connections faster.
    let swarm_config = SwarmConfig::with_tokio_executor()
        .with_idle_connection_timeout(Duration::from_secs(300));

    let mut swarm = Swarm::new(transport, behaviour, peer_id, swarm_config);

    for addr in &listen_addrs {
        swarm.listen_on(addr.clone())?;
        info!(%addr, "listening");
    }

    info!("bootstrap node running — press Ctrl+C to stop");

    // Relay metrics counters — updated by event handlers, logged by ticker.
    let mut active_reservations: u64 = 0;
    let mut active_circuits: u64 = 0;
    let mut total_circuits: u64 = 0;
    let mut denied_circuits: u64 = 0;

    let mut metrics_ticker = tokio::time::interval(Duration::from_secs(300));
    metrics_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // C7: retain the verified PROVIDER_ANNOUNCE records we see (this bootstrap subscribes to the
    // swarm topic, so it receives every announce from its connected providers) so we can answer
    // "who serves model X?" authoritatively — instead of relying on the D-sized gossip mesh to
    // forward the advert to the right consumer, which it does not do reliably across NATs.
    let mut registry = openhydra_network::registry::ProviderRegistry::new(REGISTRY_TTL_MS);
    let mut registry_reaper = tokio::time::interval(Duration::from_secs(REGISTRY_REAP_SECS));
    registry_reaper.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Event loop — Phase 4.4: graceful shutdown via signal handling.
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("received SIGTERM/SIGINT, shutting down gracefully");
                break;
            }
            _ = metrics_ticker.tick() => {
                let peers = swarm.connected_peers().count();
                info!(
                    active_reservations,
                    active_circuits,
                    total_circuits,
                    denied_circuits,
                    connected_peers = peers,
                    registry_models = registry.model_count(),
                    registry_providers = registry.len(),
                    "relay_metrics"
                );
            }
            _ = registry_reaper.tick() => {
                let removed = registry.reap(now_ms());
                if removed > 0 {
                    debug!(removed, remaining = registry.len(), "C7 registry: reaped stale providers");
                }
            }
            event = swarm.select_next_some() => {
                match event {
                    libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } => {
                        info!(%address, "new listen address");
                        // A bootstrap is unambiguously public (its public IP sits directly
                        // on the interface), so confirm each globally-routable listen addr
                        // as an EXTERNAL address. Without this the relay server has no
                        // external addresses to put in reservations and hands clients an
                        // empty reservation (`NoAddressesInReservation`), breaking all
                        // cross-NAT circuit routing. This restores what AutoNAT v1 used to
                        // do here (it emitted ExternalAddrConfirmed on its Public verdict);
                        // retiring v1 left only the v2 *server*, which confirms other peers,
                        // never the bootstrap itself. Safe for a bootstrap (unlike a peer,
                        // which may be firewalled — see event_loop's R-DHT-2 reasoning).
                        if openhydra_network::event_loop::is_globally_reachable_addr(&address) {
                            info!(%address, "confirming public listen addr as external (relay reservations)");
                            swarm.add_external_address(address.clone());
                        }
                    }
                    libp2p::swarm::SwarmEvent::Behaviour(event) => {
                        match event {
                            BootstrapBehaviourEvent::Kademlia(kad::Event::RoutingUpdated {
                                peer, ..
                            }) => {
                                info!(%peer, "kademlia routing table updated");
                            }
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::ReservationReqAccepted { src_peer_id, .. },
                            ) => {
                                active_reservations += 1;
                                info!(%src_peer_id, active_reservations, "relay reservation accepted");
                            }
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::CircuitClosed { src_peer_id, dst_peer_id, error },
                            ) => {
                                active_circuits = active_circuits.saturating_sub(1);
                                info!(%src_peer_id, %dst_peer_id, ?error, active_circuits, "relay circuit closed");
                                // F-6 leech lockout: a circuit closed because it
                                // blew the per-circuit byte budget = sustained
                                // abuse → lock the source peer out (jittered
                                // 15-30 min) so the LeechRateLimiter denies its
                                // next reservations/circuits. Only this specific
                                // error counts; normal closes don't penalize.
                                if error
                                    .as_ref()
                                    .map(|e| e.to_string().contains(MAX_CIRCUIT_BYTES_ERROR))
                                    .unwrap_or(false)
                                {
                                    let now = unix_secs_now();
                                    if let Ok(mut t) = leech_table.lock() {
                                        let until = t.record_cap_out(src_peer_id, now, wallclock_jitter_frac());
                                        t.prune(now); // bound memory (cap-outs are rare)
                                        warn!(
                                            %src_peer_id, lockout_until = until,
                                            "leech: circuit exceeded byte budget — locked out (F-6)"
                                        );
                                    }
                                }
                            }
                            // Task 7.1: Relay server — circuit/reservation lifecycle
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::CircuitReqAccepted { src_peer_id, dst_peer_id },
                            ) => {
                                active_circuits += 1;
                                total_circuits += 1;
                                info!(%src_peer_id, %dst_peer_id, active_circuits, total_circuits, "relay circuit request accepted");
                            }
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::CircuitReqDenied { src_peer_id, dst_peer_id },
                            ) => {
                                denied_circuits += 1;
                                warn!(%src_peer_id, %dst_peer_id, denied_circuits, "relay circuit request denied (capacity limit)");
                            }
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::ReservationReqDenied { src_peer_id },
                            ) => {
                                warn!(%src_peer_id, "relay reservation request denied");
                            }
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::ReservationTimedOut { src_peer_id },
                            ) => {
                                active_reservations = active_reservations.saturating_sub(1);
                                info!(%src_peer_id, active_reservations, "relay reservation timed out");
                            }
                            #[allow(deprecated)]
                            BootstrapBehaviourEvent::RelayServer(other) => {
                                debug!(?other, "relay server event");
                            }

                            // Task 7.1: Kademlia — routing & query diagnostics
                            BootstrapBehaviourEvent::Kademlia(kad::Event::UnroutablePeer { peer }) => {
                                warn!(%peer, "kademlia: unroutable peer (no known listen address)");
                            }
                            BootstrapBehaviourEvent::Kademlia(kad::Event::OutboundQueryProgressed {
                                result, step, ..
                            }) => {
                                debug!(?result, last = step.last, "kademlia: outbound query progressed");
                            }
                            BootstrapBehaviourEvent::Kademlia(kad::Event::RoutablePeer { peer, address }) => {
                                debug!(%peer, %address, "kademlia: routable peer");
                            }
                            BootstrapBehaviourEvent::Kademlia(kad::Event::PendingRoutablePeer { peer, address }) => {
                                debug!(%peer, %address, "kademlia: pending routable peer");
                            }
                            BootstrapBehaviourEvent::Kademlia(kad::Event::InboundRequest { request }) => {
                                debug!(?request, "kademlia: inbound request");
                            }
                            BootstrapBehaviourEvent::Kademlia(kad::Event::ModeChanged { new_mode }) => {
                                info!(?new_mode, "kademlia: mode changed");
                            }

                            // R-DHT-11: AutoNAT v2 server — answered a peer's
                            // per-address dial-back request.
                            BootstrapBehaviourEvent::AutonatV2Server(ev) => {
                                debug!(?ev, "autonat v2 server: dial-back handled");
                            }

                            // Task 7.1: Gossipsub — message forwarding & topic membership
                            BootstrapBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                                propagation_source, message_id, message,
                            }) => {
                                debug!(
                                    %propagation_source,
                                    %message_id,
                                    topic = %message.topic,
                                    len = message.data.len(),
                                    "gossipsub: message received/forwarded"
                                );
                                // C7: retain verified PROVIDER_ANNOUNCE records into the registry.
                                // `message.source` is the strict-signed ORIGINAL author (not the
                                // relay hop `propagation_source`), so it is the right trust anchor
                                // for `pex_record_is_authentic` — which checks the embedded signed
                                // record AND that this author owns it. Forged/unsigned adverts are
                                // rejected; the store only ever conveys self-signed records.
                                if let (Some(author), Ok(parsed)) = (
                                    message.source,
                                    serde_json::from_slice::<serde_json::Value>(&message.data),
                                ) {
                                    if parsed.get("type").and_then(|v| v.as_str())
                                        == Some(openhydra_network::event_loop::PROVIDER_ANNOUNCE_TYPE)
                                    {
                                        if let Some(rec) = parsed
                                            .get("record")
                                            .and_then(|v| serde_json::from_value::<openhydra_network::types::PeerRecord>(v.clone()).ok())
                                        {
                                            let author_str = author.to_base58();
                                            match openhydra_network::dht::pex_record_is_authentic(&rec, &author_str) {
                                                Ok(()) => registry.insert(rec, now_ms()),
                                                Err(e) => debug!(%author_str, %e, "C7 registry: rejected unauthentic advert"),
                                            }
                                        }
                                    }
                                }
                            }
                            BootstrapBehaviourEvent::Gossipsub(gossipsub::Event::Subscribed {
                                peer_id, topic,
                            }) => {
                                info!(%peer_id, %topic, "gossipsub: peer subscribed");
                            }
                            BootstrapBehaviourEvent::Gossipsub(gossipsub::Event::Unsubscribed {
                                peer_id, topic,
                            }) => {
                                info!(%peer_id, %topic, "gossipsub: peer unsubscribed");
                            }
                            BootstrapBehaviourEvent::Gossipsub(gossipsub::Event::GossipsubNotSupported {
                                peer_id,
                            }) => {
                                debug!(%peer_id, "gossipsub: protocol not supported by peer");
                            }

                            // Task 7.1: Identify — peer metadata exchange
                            BootstrapBehaviourEvent::Identify(identify::Event::Received {
                                peer_id, info, ..
                            }) => {
                                debug!(
                                    %peer_id,
                                    protocol_version = %info.protocol_version,
                                    agent_version = %info.agent_version,
                                    listen_addrs = info.listen_addrs.len(),
                                    "identify: received peer info"
                                );
                            }
                            BootstrapBehaviourEvent::Identify(identify::Event::Sent { peer_id, .. }) => {
                                debug!(%peer_id, "identify: sent local info");
                            }
                            BootstrapBehaviourEvent::Identify(identify::Event::Pushed { peer_id, .. }) => {
                                debug!(%peer_id, "identify: pushed local info");
                            }
                            BootstrapBehaviourEvent::Identify(identify::Event::Error {
                                peer_id, error, ..
                            }) => {
                                warn!(%peer_id, %error, "identify: error");
                            }

                            // Task 7.1: DCUtR — hole-punch results
                            BootstrapBehaviourEvent::Dcutr(dcutr_event) => {
                                match &dcutr_event.result {
                                    Ok(connection_id) => {
                                        info!(
                                            peer_id = %dcutr_event.remote_peer_id,
                                            %connection_id,
                                            "dcutr: direct connection upgrade succeeded"
                                        );
                                    }
                                    Err(error) => {
                                        warn!(
                                            peer_id = %dcutr_event.remote_peer_id,
                                            %error,
                                            "dcutr: direct connection upgrade failed"
                                        );
                                    }
                                }
                            }

                            // Task 7.1: Ping — connection health
                            BootstrapBehaviourEvent::Ping(ping::Event { peer, result, .. }) => {
                                match result {
                                    Ok(rtt) => {
                                        debug!(%peer, ?rtt, "ping: success");
                                    }
                                    Err(error) => {
                                        warn!(%peer, %error, "ping: failure");
                                    }
                                }
                            }

                            // C7: answer a consumer's registry query from the retained store. The
                            // records are the providers' own self-signed PeerRecords (verified on
                            // ingest); the consumer re-verifies before dialing. Inbound-only, so
                            // only Request/InboundFailure ever occur — other variants are ignored.
                            BootstrapBehaviourEvent::RegistryQuery(request_response::Event::Message {
                                peer, message: request_response::Message::Request { request, channel, .. },
                            }) => {
                                let records = registry.providers_for(&request.model_id, now_ms());
                                debug!(%peer, model = %request.model_id, n = records.len(), "C7 registry: answering query");
                                if swarm
                                    .behaviour_mut()
                                    .registry_query
                                    .send_response(channel, RegistryReply { records })
                                    .is_err()
                                {
                                    debug!(%peer, "C7 registry: response channel closed before send");
                                }
                            }
                            BootstrapBehaviourEvent::RegistryQuery(other) => {
                                debug!(?other, "C7 registry: non-request event");
                            }
                        }
                    }
                    libp2p::swarm::SwarmEvent::ConnectionEstablished { peer_id: p, .. } => {
                        info!(%p, "connection established");
                    }
                    libp2p::swarm::SwarmEvent::ConnectionClosed { peer_id: p, cause, .. } => {
                        info!(%p, ?cause, "connection closed");
                    }
                    // Task 7.1: Swarm-level events — no more catch-all
                    libp2p::swarm::SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                        warn!(?peer_id, %error, "outgoing connection error");
                    }
                    libp2p::swarm::SwarmEvent::IncomingConnectionError { error, local_addr, send_back_addr, .. } => {
                        warn!(%error, %local_addr, %send_back_addr, "incoming connection error");
                    }
                    libp2p::swarm::SwarmEvent::ExternalAddrConfirmed { address } => {
                        info!(%address, "external address confirmed");
                    }
                    libp2p::swarm::SwarmEvent::ExternalAddrExpired { address } => {
                        info!(%address, "external address expired");
                    }
                    libp2p::swarm::SwarmEvent::NewExternalAddrCandidate { address } => {
                        debug!(%address, "new external address candidate");
                    }
                    libp2p::swarm::SwarmEvent::NewExternalAddrOfPeer { peer_id, address } => {
                        debug!(%peer_id, %address, "discovered new address of peer");
                    }
                    libp2p::swarm::SwarmEvent::ExpiredListenAddr { address, .. } => {
                        info!(%address, "listen address expired");
                    }
                    libp2p::swarm::SwarmEvent::ListenerClosed { addresses, reason, .. } => {
                        warn!(?addresses, ?reason, "listener closed");
                    }
                    libp2p::swarm::SwarmEvent::ListenerError { error, .. } => {
                        warn!(%error, "listener error");
                    }
                    libp2p::swarm::SwarmEvent::Dialing { peer_id, .. } => {
                        debug!(?peer_id, "dialing peer");
                    }
                    libp2p::swarm::SwarmEvent::IncomingConnection { local_addr, send_back_addr, .. } => {
                        debug!(%local_addr, %send_back_addr, "incoming connection");
                    }
                    // #[non_exhaustive] requires a catch-all for future SwarmEvent variants.
                    other => {
                        debug!(?other, "unhandled swarm event");
                    }
                }
            }
        }
    }

    Ok(())
}

// Minimal CLI flag parsing (no clap dependency to keep binary small).

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn parse_flag_multi(args: &[String], name: &str) -> Option<Vec<Multiaddr>> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == name {
            if let Some(val) = args.get(i + 1) {
                if let Ok(addr) = val.parse() {
                    result.push(addr);
                }
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}
