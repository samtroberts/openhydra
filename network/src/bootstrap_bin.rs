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
//!
//! A bootstrap of a private deployment (on a private network, e.g. `10.0.0.5`) declares the address
//! its peers reach it at with `--external /ip4/10.0.0.5/tcp/4001` (repeatable): its relay hands that
//! out in reservations. Globally routable listen addresses are confirmed on their own.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use libp2p::swarm::Config as SwarmConfig;
use libp2p::{autonat, gossipsub, identify, kad, ping, relay, request_response, Multiaddr, Swarm};
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
    /// Bounds on the incoming connections one network address can make the bootstrap hold.
    ip_limits: openhydra_network::ip_limits::Behaviour,
    /// Bounds on the connections other nodes can make the bootstrap hold.
    connection_limits: libp2p::connection_limits::Behaviour,
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
    gossipsub: openhydra_network::swarm::GossipBehaviour,
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

/// OpenHydra's Kademlia protocol (the peers' is the same, `swarm.rs`).
const KAD_PROTOCOL: &str = "/openhydra/kad/1.0.0";
/// How often to sweep expired registry entries.
const REGISTRY_REAP_SECS: u64 = 60;
/// How long a provider in the registry may have no connection to us before we report it gone
/// (`PROVIDER_GONE`): longer than a network change takes to reconnect, far shorter than the
/// registry TTL. GPU test 2026-10-30: a dead provider stayed listed for ~10 min on consumers that
/// had never connected to it.
const PROVIDER_GONE_GRACE: Duration = Duration::from_secs(30);
/// Disconnected providers watched at most, and `PROVIDER_GONE` notices published per check at most
/// (an outage of ours mustn't become a gossip flood; the rest age out on the TTL).
const MAX_GONE_WATCH: usize = 8192;
const MAX_GONE_PER_TICK: usize = 64;
/// The most peers a bootstrap keeps in its gossip mesh (as before; see the gossipsub config below).
const BOOTSTRAP_GOSSIP_MESH_MAX: usize = 6;

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
        // Colour only on a terminal: under systemd the escape codes ended up in the journal.
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .init();

    // Parse CLI args (minimal, no clap dependency).
    let args: Vec<String> = std::env::args().collect();
    let identity_path = parse_flag(&args, "--identity")
        .unwrap_or_else(|| "/opt/openhydra/.libp2p_identity.key".to_string());
    let listen_addrs: Vec<Multiaddr> = parse_flag_multi(&args, "--listen")?
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
    // The gossip mesh's upper bound (load tests A/B it; production keeps BOOTSTRAP_GOSSIP_MESH_MAX).
    let gossip_mesh_max: usize = match parse_flag(&args, "--gossip-mesh-max") {
        Some(v) => v.parse().map_err(|e| format!("--gossip-mesh-max: {e}"))?,
        None => BOOTSTRAP_GOSSIP_MESH_MAX,
    };
    let metrics_bind: Option<std::net::SocketAddr> = match parse_flag(&args, "--metrics-bind") {
        Some(v) => Some(v.parse().map_err(|e| format!("--metrics-bind: {e}"))?),
        None => None,
    };

    // Transport: QUIC + TCP (bootstrap is public, no relay needed)
    let tcp_transport = openhydra_network::transport::build_tcp_transport(&keypair)?;
    let quic_transport = openhydra_network::transport::build_quic_transport(&keypair)?;
    // Errors unwrapped, so the dial-failure check can read them (`transport::dial_means_unreachable`).
    let transport = openhydra_network::transport::either_transport(quic_transport, tcp_transport);
    // One of OpenHydra's own bootstraps never dials a private address (see `public_dials_only`); a
    // private deployment's does (its peers are private).
    let transport = if public_bootstrap(&keypair.public().to_peer_id()) {
        openhydra_network::transport::public_dials_only(transport)
    } else {
        transport
    };

    // Kademlia in server mode (bootstrap node).
    let mut kad_config = kad::Config::new(
        libp2p::StreamProtocol::new(KAD_PROTOCOL),
    );
    kad_config.set_query_timeout(Duration::from_secs(30));
    // Phase 7.5: Reduced from 600s to 300s to match peer nodes. The old
    // 600s TTL doubled the ghost window compared to peers.
    kad_config.set_record_ttl(Some(Duration::from_secs(300)));
    kad_config.set_provider_record_ttl(Some(Duration::from_secs(300)));
    kad_config.set_publication_interval(Some(Duration::from_secs(240)));
    // Every PUT_RECORD / ADD_PROVIDER comes to us to accept (below) instead of being stored as sent:
    // a record must verify and sit under its own key (review 2026-10-29: anyone could overwrite any
    // key, or fill the store with junk so honest PUTs failed).
    kad_config.set_record_filtering(kad::StoreInserts::FilterBoth);
    // One of OpenHydra's production bootstraps answers the whole public network: its routing table
    // takes no private address, so none goes into a DHT answer (review 2026-10-29). A bootstrap of a
    // private deployment, or a local test one, takes every address (their peers are private).
    if public_bootstrap(&peer_id) {
        kad_config.set_address_filter(kad::AddressFilter::new(|_peer, addr| {
            openhydra_network::swarm::routable_on_public_network(addr)
        }));
        info!("production bootstrap: private addresses are kept out of the routing table");
    }

    let store = kad::store::MemoryStore::new(peer_id);
    let mut kademlia = kad::Behaviour::with_config(peer_id, store, kad_config);
    // Bootstrap nodes MUST be in server mode to accept GET_RECORD/PUT_RECORD
    // queries from peers. Without this, they may stay in client mode and
    // silently refuse DHT operations.
    kademlia.set_mode(Some(kad::Mode::Server));

    // Add other bootstrap peers to the routing table.
    let bootstrap_peers: Vec<Multiaddr> = parse_flag_multi(&args, "--peer")?
        .unwrap_or_default();
    let mut configured_peers = std::collections::HashSet::new();
    let mut dial_failures: HashMap<libp2p::PeerId, (u32, std::time::Instant)> = HashMap::new();
    let mut dial_back = DialBack::default();
    let filter_addrs = public_bootstrap(&peer_id);
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
            configured_peers.insert(remote_id);
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

    // What each network holds on our relay right now (kept from `relay_ledger`; read by the caps).
    let net_holdings = std::sync::Arc::new(std::sync::Mutex::new(openhydra_network::relay::NetHoldings::default()));
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
        // Our limits replace libp2p's defaults, which `..Default::default()` would keep: one circuit
        // token per 2 minutes per peer and one a minute per IP left busy gateways (and everyone
        // behind one carrier-NAT address) with almost no circuits (GPU test 2026-10-29/30).
        cfg.reservation_rate_limiters.clear();
        cfg.circuit_src_rate_limiters.clear();
        // F-6: deny reservations AND circuits from leech-locked peers. Two
        // limiters share one table (built-in max_* still bound congestion).
        cfg.reservation_rate_limiters
            .push(Box::new(LeechRateLimiter::new(leech_table.clone())));
        cfg.circuit_src_rate_limiters
            .push(Box::new(LeechRateLimiter::new(leech_table.clone())));
        use openhydra_network::relay::{Held, HoldingsLimiter, PrefixRateLimiter as L};
        // Caps on what one network holds at once (they take no tokens), then the rates —
        // narrowest first: libp2p asks them in order and stops at the first "no", and every one
        // asked before has spent a token, so a peer over its own limit must not drain the bucket
        // its whole carrier-NAT address or /48 shares (review 2026-10-30).
        cfg.reservation_rate_limiters.push(Box::new(L::reservations_per_peer()));
        cfg.circuit_src_rate_limiters.push(Box::new(L::circuits_per_peer()));
        cfg.reservation_rate_limiters.insert(0, Box::new(HoldingsLimiter::new(
            net_holdings.clone(), Held::Reservations, openhydra_network::relay::MAX_RESERVATIONS_PER_NETWORK)));
        cfg.reservation_rate_limiters.push(Box::new(L::reservations_per_v4()));
        cfg.reservation_rate_limiters.push(Box::new(L::reservations_per_64()));
        cfg.reservation_rate_limiters.push(Box::new(L::reservations_per_48()));
        cfg.circuit_src_rate_limiters.insert(0, Box::new(HoldingsLimiter::new(
            net_holdings.clone(), Held::Circuits, openhydra_network::relay::MAX_CIRCUITS_PER_NETWORK)));
        cfg.circuit_src_rate_limiters.push(Box::new(L::circuits_per_v4()));
        cfg.circuit_src_rate_limiters.push(Box::new(L::circuits_per_64()));
        cfg.circuit_src_rate_limiters.push(Box::new(L::circuits_per_48()));
        cfg
    });

    // AutoNAT reporter — responds to NAT probes from peers.
    //
    // Identify.
    let identify = identify::Behaviour::new(
        identify::Config::new("openhydra/0.1.0".to_string(), keypair.public())
            .with_push_listen_addr_updates(true)
            // No address cache: dials use only what Kademlia admitted (see the peer node's).
            .with_cache_size(0),
    );

    // DCUtR — advertises hole-punch support in Identify protocol list.
    let dcutr = libp2p::dcutr::Behaviour::new(peer_id);

    // Gossipsub forwarder — bootstraps subscribe to the swarm-wide topic so
    // peers that can only reach each other through a bootstrap still see
    // each other's events. The message-authenticity signing guarantees the
    // bootstrap can't forge events; it just propagates signed messages.
    //
    // Mesh size (30 Oct): suspected of hiding announces from consumers outside the bootstraps' 6-peer
    // meshes; measured instead — 10 consumers received 50 of 50 announces with it — so it stays.
    // The real cause of the gaps was gossipsub not re-sending subscriptions to a peer that restarted
    // under the same identity (fixed in the vendored crate, see its CHANGELOG).
    //
    // KNOWN LIMITATION: Gossipsub validates message signatures (Strict mode)
    // but does not validate message payload schemas. Malformed payloads are
    // forwarded as-is. Add payload validation when the message format stabilizes.
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(1))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .max_transmit_size(openhydra_network::swarm::GOSSIP_MAX_TRANSMIT)
        // Every message received is kept this many heartbeats (1 s) for peers' IWANTs: what a
        // flood costs us is about this many seconds of its bytes (review 2026-10-29: 5 by default).
        .history_length(openhydra_network::swarm::GOSSIP_HISTORY)
        .history_gossip(1)
        .duplicate_cache_time(openhydra_network::swarm::GOSSIP_DUPLICATE_CACHE)
        .flood_publish(true)
        // Forward only what `check_gossip` accepts (see there); every message is reported on.
        .validate_messages()
        .mesh_outbound_min(1)
        .mesh_n_low(1)
        .mesh_n(3)
        .mesh_n_high(gossip_mesh_max)
        .build()
        .map_err(|e| format!("gossipsub config: {e}"))?;
    // Only our topic: a peer's other subscriptions aren't kept (review 2026-10-29).
    let mut gossipsub = openhydra_network::swarm::GossipBehaviour::new_with_subscription_filter(
        gossipsub::MessageAuthenticity::Signed(keypair.clone()),
        gossipsub_config,
        None,
        openhydra_network::swarm::gossip_topic_filter(),
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
        ip_limits: openhydra_network::ip_limits::Behaviour::new(openhydra_network::ip_limits::IpLimitsConfig::BOOTSTRAP),
        connection_limits: libp2p::connection_limits::Behaviour::new(openhydra_network::behaviour::bootstrap_connection_limits()),
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

    let transport = openhydra_network::transport::with_setup_timeouts(
        transport,
        openhydra_network::transport::OUTBOUND_SETUP_TIMEOUT,
        openhydra_network::transport::INBOUND_SETUP_TIMEOUT,
    );
    let mut swarm = Swarm::new(transport, behaviour, peer_id, swarm_config);
    // Addresses the operator says peers reach this bootstrap at (a private deployment's bootstrap is
    // on a private network, which isn't confirmed on its own): its relay puts these in reservations.
    for addr in parse_flag_multi(&args, "--external")?.unwrap_or_default() {
        info!(%addr, "external address (from --external)");
        swarm.add_external_address(addr);
    }

    for addr in &listen_addrs {
        swarm.listen_on(addr.clone())?;
        info!(%addr, "listening");
    }

    info!("bootstrap node running — press Ctrl+C to stop");

    // Relay metrics counters — updated by event handlers, logged by ticker.
    let mut relay_ledger = RelayLedger::default();
    let mut moved_conns = MovedConns::default();
    let mut put_quota = openhydra_network::dht::PutQuota::default();
    let mut provider_quota = openhydra_network::dht::PutQuota::default();
    let mut total_circuits: u64 = 0;
    let mut denied_circuits: u64 = 0;
    // Each open connection's kind (transport, direction), for the metrics endpoint.
    let mut conn_kinds: std::collections::HashMap<libp2p::swarm::ConnectionId, &'static str> = Default::default();

    // Load tests read this every second (`--metrics-bind`, loopback only; off by default).
    let metrics_json = std::sync::Arc::new(std::sync::Mutex::new(String::from("{}")));
    if let Some(addr) = metrics_bind {
        if !addr.ip().is_loopback() {
            return Err("--metrics-bind must be a loopback address".into());
        }
        let shared = metrics_json.clone();
        let listener = std::net::TcpListener::bind(addr)?;
        std::thread::spawn(move || {
            use std::io::{Read, Write};
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let body = shared.lock().map(|b| b.clone()).unwrap_or_default();
                let _ = write!(stream, "HTTP/1.0 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}", body.len(), body);
            }
        });
        info!(%addr, "metrics endpoint");
    }
    let mut metrics_tick = tokio::time::interval(Duration::from_secs(1));
    metrics_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut gossip_ticker = tokio::time::interval(Duration::from_secs(60));
    gossip_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut metrics_ticker = tokio::time::interval(Duration::from_secs(300));
    metrics_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // C7: retain the verified PROVIDER_ANNOUNCE records we see (this bootstrap subscribes to the
    // swarm topic, so it receives every announce from its connected providers) so we can answer
    // "who serves model X?" authoritatively — instead of relying on the D-sized gossip mesh to
    // forward the advert to the right consumer, which it does not do reliably across NATs.
    let mut registry = openhydra_network::registry::ProviderRegistry::new(REGISTRY_TTL_MS);
    // Peers whose adverts were refused for their content (an odd model name, an oversized record):
    // warned about once each, so an attempt to push such names through the registry is visible
    // without flooding the log. Bounded; past the cap only the running count is kept.
    let mut content_rejected: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut content_rejected_total: u64 = 0;
    let mut registry_reaper = tokio::time::interval(Duration::from_secs(REGISTRY_REAP_SECS));
    registry_reaper.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // Gossip validation (`check_gossip`): a per-author rate, and counts for the metrics line.
    let mut author_rate = AuthorRate::default();
    let mut gossip_rejected: u64 = 0;
    let mut gossip_ignored: u64 = 0;
    // Providers in the registry whose last connection to us closed, and when.
    let mut gone_watch: std::collections::HashMap<libp2p::PeerId, std::time::Instant> = Default::default();
    let mut gone_ticker = tokio::time::interval(Duration::from_secs(5));
    gone_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // Event loop — Phase 4.4: graceful shutdown via signal handling.
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("received SIGTERM/SIGINT, shutting down gracefully");
                break;
            }
            _ = metrics_tick.tick(), if metrics_bind.is_some() => {
                let topic = gossipsub::IdentTopic::new(openhydra_network::swarm::GOSSIPSUB_TOPIC).hash();
                let g = &swarm.behaviour().gossipsub;
                let counters = swarm.network_info().connection_counters().clone();
                let json = serde_json::json!({
                    "t_ms": now_ms(),
                    "connected_peers": swarm.connected_peers().count(),
                    "connections": counters.num_established(),
                    "connections_in": counters.num_established_incoming(),
                    "pending_in": counters.num_pending_incoming(),
                    "reservations": relay_ledger.reservations(),
                    "circuits": relay_ledger.circuits(),
                    "total_circuits": total_circuits,
                    "denied_circuits": denied_circuits,
                    "registry_models": registry.model_count(),
                    "registry_providers": registry.len(),
                    "gossip_topic_peers": g.all_peers().filter(|(_, t)| t.contains(&&topic)).count(),
                    "gossip_mesh_peers": g.mesh_peers(&topic).count(),
                    "gossip_rejected": gossip_rejected,
                    "gossip_ignored": gossip_ignored,
                    "conn_kinds": conn_kinds.values().fold(std::collections::BTreeMap::<&str, u64>::new(), |mut m, k| { *m.entry(k).or_default() += 1; m }),
                });
                if let Ok(mut b) = metrics_json.lock() {
                    *b = json.to_string();
                }
            }
            _ = gossip_ticker.tick() => {
                // Mesh vs topic peers: we forward only to mesh peers (30 Oct mesh verification).
                let topic = gossipsub::IdentTopic::new(openhydra_network::swarm::GOSSIPSUB_TOPIC).hash();
                let g = &swarm.behaviour().gossipsub;
                let topic_peers = g.all_peers().filter(|(_, t)| t.contains(&&topic)).count();
                let mesh: Vec<String> = g.mesh_peers(&topic).map(|p| p.to_base58()).collect();
                info!(topic_peers, mesh_peers = mesh.len(), "gossip_mesh");
                debug!(?mesh, "gossip_mesh members");
            }
            _ = metrics_ticker.tick() => {
                let peers = swarm.connected_peers().count();
                info!(
                    active_reservations = relay_ledger.reservations(),
                    active_circuits = relay_ledger.circuits(),
                    total_circuits,
                    denied_circuits,
                    connected_peers = peers,
                    registry_models = registry.model_count(),
                    registry_providers = registry.len(),
                    gossip_rejected,
                    gossip_ignored,
                    "relay_metrics"
                );
            }
            _ = gone_ticker.tick() => {
                let now = std::time::Instant::now();
                let due: Vec<libp2p::PeerId> = gone_watch
                    .iter()
                    .filter(|(p, at)| now.duration_since(**at) >= PROVIDER_GONE_GRACE && !swarm.is_connected(p))
                    .map(|(p, _)| *p)
                    .take(MAX_GONE_PER_TICK)
                    .collect();
                gone_watch.retain(|p, _| !swarm.is_connected(p));
                for p in due {
                    gone_watch.remove(&p);
                    let Some(last_record_ms) = registry.remove_peer(&p.to_base58()) else { continue };
                    let notice = serde_json::json!({
                        "type": openhydra_network::event_loop::PROVIDER_GONE_TYPE,
                        "libp2p_peer_id": p.to_base58(),
                        "last_record_ms": last_record_ms,
                        "timestamp": now_ms() / 1000,
                    });
                    let topic = gossipsub::IdentTopic::new(openhydra_network::swarm::GOSSIPSUB_TOPIC);
                    match swarm.behaviour_mut().gossipsub.publish(topic, notice.to_string().into_bytes()) {
                        Ok(_) => info!(%p, "provider gone: removed from the registry and reported"),
                        Err(e) => debug!(%p, %e, "provider gone: removed from the registry; report not sent"),
                    }
                }
            }
            _ = registry_reaper.tick() => {
                // Warn again (once each) for authors refused since the last reap.
                content_rejected.clear();
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
                                relay::Event::ReservationReqAccepted { src_peer_id, renewed },
                            ) => {
                                relay_ledger.reservation_accepted(src_peer_id, renewed); relay_ledger.sync(&net_holdings);
                                let active_reservations = relay_ledger.reservations();
                                if renewed {
                                    debug!(%src_peer_id, active_reservations, "relay reservation renewed");
                                } else {
                                    info!(%src_peer_id, active_reservations, "relay reservation accepted");
                                }
                            }
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::CircuitClosed { src_peer_id, dst_peer_id, error },
                            ) => {
                                relay_ledger.circuit_closed(src_peer_id, dst_peer_id); relay_ledger.sync(&net_holdings);
                                info!(%src_peer_id, %dst_peer_id, ?error, active_circuits = relay_ledger.circuits(), "relay circuit closed");
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
                                relay_ledger.circuit_accepted(src_peer_id, dst_peer_id); relay_ledger.sync(&net_holdings);
                                total_circuits += 1;
                                info!(%src_peer_id, %dst_peer_id, active_circuits = relay_ledger.circuits(), total_circuits, "relay circuit request accepted");
                            }
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::CircuitReqDenied { src_peer_id, dst_peer_id },
                            ) => {
                                denied_circuits += 1;
                                warn!(%src_peer_id, %dst_peer_id, denied_circuits, "relay circuit request denied (capacity limit)");
                            }
                            #[allow(deprecated)]
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::ReservationReqAcceptFailed { src_peer_id, error },
                            ) => {
                                relay_ledger.reservation_accept_failed(src_peer_id); relay_ledger.sync(&net_holdings);
                                warn!(%src_peer_id, %error, active_reservations = relay_ledger.reservations(), "relay reservation accept failed (its slot stays taken until the connection closes)");
                            }
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::ReservationReqDenied { src_peer_id },
                            ) => {
                                // (Debug: any peer can make these by the thousand.)
                                debug!(%src_peer_id, "relay reservation request denied");
                            }
                            BootstrapBehaviourEvent::RelayServer(
                                relay::Event::ReservationTimedOut { src_peer_id },
                            ) => {
                                relay_ledger.reservation_timed_out(src_peer_id); relay_ledger.sync(&net_holdings);
                                info!(%src_peer_id, active_reservations = relay_ledger.reservations(), "relay reservation timed out");
                            }
                            #[allow(deprecated)]
                            BootstrapBehaviourEvent::RelayServer(other) => {
                                debug!(?other, "relay server event");
                            }

                            // Task 7.1: Kademlia — routing & query diagnostics
                            BootstrapBehaviourEvent::Kademlia(kad::Event::UnroutablePeer { peer }) => {
                                debug!(%peer, "kademlia: unroutable peer (no known listen address)");
                                dial_back.unroutable(peer);
                                maybe_dial_back(&mut swarm, &mut dial_back, peer);
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
                                use libp2p::kad::store::RecordStore;
                                match request {
                                    kad::InboundRequest::PutRecord { source, record: Some(record), .. } => {
                                        // Verified, recent, under its own key, never over a newer one,
                                        // and within its publisher's and network's share of the store.
                                        match openhydra_network::dht::check_put_record(&record, now_ms()) {
                                            Ok(peer_record) => {
                                                let origin = moved_conns.origin_of(&source).unwrap_or_else(|| format!("relayed:{source}"));
                                                let newer = openhydra_network::dht::newer_than_stored(
                                                    swarm.behaviour_mut().kademlia.store_mut().get(&record.key).as_deref(),
                                                    &peer_record,
                                                );
                                                if !newer {
                                                    debug!(%source, "kademlia: refused a PUT_RECORD older than the one held");
                                                } else if !put_quota.admit(&record.key, &peer_record.libp2p_peer_id, &origin, std::time::Instant::now()) {
                                                    debug!(%source, %origin, "kademlia: refused a PUT_RECORD over its publisher's or network's share");
                                                } else if let Err(e) = swarm.behaviour_mut().kademlia.store_mut().put(record) {
                                                    debug!(%source, ?e, "kademlia: store full for a verified record");
                                                }
                                            }
                                            Err(e) => debug!(%source, %e, "kademlia: refused a PUT_RECORD"),
                                        }
                                    }
                                    // libp2p-kad only hands on a provider record from the provider itself;
                                    // each provider and network gets its share of the provider store.
                                    kad::InboundRequest::AddProvider { record: Some(provider) } => {
                                        let origin = moved_conns.origin_of(&provider.provider).unwrap_or_else(|| format!("relayed:{}", provider.provider));
                                        if !provider_quota.admit(&provider.key, &provider.provider.to_base58(), &origin, std::time::Instant::now()) {
                                            debug!(provider = %provider.provider, %origin, "kademlia: refused an ADD_PROVIDER over its share");
                                        } else if let Err(e) = swarm.behaviour_mut().kademlia.store_mut().add_provider(provider) {
                                            debug!(?e, "kademlia: provider store full");
                                        }
                                    }
                                    other => debug!(?other, "kademlia: inbound request"),
                                }
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
                                // Check it before it goes any further: gossipsub forwards it only on Accept.
                                let mut verdict = check_gossip(&message.data, message.source, &configured_peers, now_ms());
                                if verdict == GossipVerdict::Accept
                                    && !message.source.is_some_and(|a| author_rate.allow(a, std::time::Instant::now()))
                                {
                                    verdict = GossipVerdict::Ignore("author over its rate");
                                }
                                let acceptance = match verdict {
                                    GossipVerdict::Accept => gossipsub::MessageAcceptance::Accept,
                                    GossipVerdict::Reject(why) => {
                                        gossip_rejected += 1;
                                        debug!(%propagation_source, source = ?message.source, why, "gossip: rejected");
                                        gossipsub::MessageAcceptance::Reject
                                    }
                                    GossipVerdict::Ignore(why) => {
                                        gossip_ignored += 1;
                                        debug!(%propagation_source, source = ?message.source, why, "gossip: ignored");
                                        gossipsub::MessageAcceptance::Ignore
                                    }
                                };
                                let _ = swarm.behaviour_mut().gossipsub.report_message_validation_result(&message_id, &propagation_source, acceptance);
                                if verdict != GossipVerdict::Accept {
                                    continue;
                                }
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
                                    // A provider withdrawing one of its own models (author = subject,
                                    // fresh signed time, as consumers check): drop it here too, or the
                                    // registry would hand it out until its TTL.
                                    if parsed.get("type").and_then(|v| v.as_str())
                                        == Some(openhydra_network::event_loop::MODEL_WITHDRAWN_TYPE)
                                    {
                                        let subject = parsed.get("libp2p_peer_id").and_then(|v| v.as_str());
                                        let model = parsed.get("model_id").and_then(|v| v.as_str());
                                        let fresh = parsed.get("timestamp").and_then(|v| v.as_u64()).is_some_and(|ts| {
                                            (now_ms() / 1000).abs_diff(ts) <= openhydra_network::event_loop::MODEL_WITHDRAWN_MAX_AGE_SECS
                                        });
                                        if let (Some(subject), Some(model), true) = (subject, model, fresh) {
                                            if subject == author.to_base58() && registry.remove_model(subject, model) {
                                                debug!(%subject, %model, "C7 registry: model withdrawn");
                                            }
                                        }
                                    }
                                    if parsed.get("type").and_then(|v| v.as_str())
                                        == Some(openhydra_network::event_loop::PROVIDER_ANNOUNCE_TYPE)
                                    {
                                        if let Some(rec) = parsed
                                            .get("record")
                                            .and_then(|v| serde_json::from_value::<openhydra_network::types::PeerRecord>(v.clone()).ok())
                                        {
                                            let author_str = author.to_base58();
                                            match openhydra_network::dht::pex_record_is_authentic(&rec, &author_str) {
                                                // Only from an author connected to us directly, and
                                                // counted against the network it connects from: identities
                                                // are free, connections from many networks aren't (review
                                                // 2026-10-29). A provider connects to every bootstrap itself.
                                                // (Load tests: one generator's many peers share an address; count each peer as its own network.)
                                                Ok(()) => match moved_conns.origin_of(&author) {
                                                    Some(origin) => {
                                                        if let Err(e) = openhydra_network::dht::check_record_age(&rec, now_ms()) {
                                                            debug!(%author_str, %e, "C7 registry: refused advert (time)");
                                                        } else if !registry.insert_from(rec, now_ms(), &origin) {
                                                            debug!(%author_str, %origin, "C7 registry: full, or older than the advert held");
                                                        }
                                                    }
                                                    None => debug!(%author_str, "C7 registry: advert from a peer not connected directly; not kept"),
                                                },
                                                Err(e) if openhydra_network::dht::check_record_content(&rec).is_err() => {
                                                    content_rejected_total += 1;
                                                    if content_rejected.len() < 4096 && content_rejected.insert(author_str.clone()) {
                                                        warn!(%author_str, %e, total = content_rejected_total, "C7 registry: refused advert with invalid content");
                                                    }
                                                }
                                                Err(e) => debug!(%author_str, %e, "C7 registry: rejected unauthentic advert"),
                                            }
                                        }
                                    }
                                }
                            }
                            BootstrapBehaviourEvent::Gossipsub(gossipsub::Event::Subscribed {
                                peer_id, topic,
                            }) => {
                                debug!(%peer_id, %topic, "gossipsub: peer subscribed");
                            }
                            BootstrapBehaviourEvent::Gossipsub(gossipsub::Event::Unsubscribed {
                                peer_id, topic,
                            }) => {
                                debug!(%peer_id, %topic, "gossipsub: peer unsubscribed");
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
                                dial_back.identified(peer_id, &info.listen_addrs, |a| {
                                    !filter_addrs || openhydra_network::swarm::routable_on_public_network(a)
                                });
                                maybe_dial_back(&mut swarm, &mut dial_back, peer_id);
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
                                debug!(%peer_id, %error, "identify: error");
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
                            BootstrapBehaviourEvent::Ping(ping::Event { peer, connection, result }) => {
                                match result {
                                    Ok(rtt) => {
                                        debug!(%peer, ?rtt, "ping: success");
                                        moved_conns.ping_ok(connection, std::time::Instant::now());
                                    }
                                    Err(error) => {
                                        // (Debug: every connection of every peer can produce one every
                                        // 15–35 s, so a remote could fill the journal with them.)
                                        debug!(%peer, %error, "ping: failure");
                                        if !matches!(error, ping::Failure::Unsupported)
                                            && moved_conns.ping_failed(connection, std::time::Instant::now())
                                        {
                                            let (old, newer) = moved_conns.ips_for_close(connection);
                                            info!(%peer, ?connection, old = ?old, answering = ?newer, "closing a connection left behind on a network the peer moved off");
                                            swarm.close_connection(connection);
                                        }
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
                    libp2p::swarm::SwarmEvent::ConnectionEstablished { peer_id: p, connection_id, endpoint, .. } => {
                        // At info (with the address) only for the other bootstraps; anyone can open and
                        // close connections at will, and users' addresses don't belong in the journal.
                        if configured_peers.contains(&p) {
                            info!(%p, remote = %endpoint.get_remote_address(), "connection established");
                        } else {
                            debug!(%p, "connection established");
                        }
                        relay_ledger.connection_established(p, endpoint.is_relayed()); relay_ledger.sync(&net_holdings);
                        if metrics_bind.is_some() {
                            let quic = endpoint.get_remote_address().iter().any(|x| matches!(x, libp2p::multiaddr::Protocol::QuicV1));
                            conn_kinds.insert(connection_id, match (endpoint.is_relayed(), quic, endpoint.is_dialer()) {
                                (true, _, _) => "relayed",
                                (false, true, false) => "quic_in",
                                (false, true, true) => "quic_out",
                                (false, false, false) => "tcp_in",
                                (false, false, true) => "tcp_out",
                            });
                        }
                        gone_watch.remove(&p);
                        if !endpoint.is_relayed() {
                            if let Some(ip) = multiaddr_ip(endpoint.get_remote_address()) {
                                moved_conns.established(connection_id, p, ip, std::time::Instant::now());
                            }
                        }
                        dial_failures.remove(&p);
                        if dial_back.finished(&connection_id) {
                            info!(%p, "dial-back connected: the routing table can list it now");
                        }
                        if let libp2p::core::ConnectedPoint::Listener { send_back_addr, .. } = &endpoint {
                            dial_back.inbound(p, send_back_addr);
                        }
                    }
                    libp2p::swarm::SwarmEvent::ConnectionClosed { peer_id: p, connection_id, endpoint, cause, .. } => {
                        if configured_peers.contains(&p) {
                            info!(%p, remote = %endpoint.get_remote_address(), ?cause, "connection closed");
                        } else {
                            debug!(%p, ?cause, "connection closed");
                        }
                        moved_conns.closed(connection_id);
                        conn_kinds.remove(&connection_id);
                        relay_ledger.connection_closed(p, endpoint.is_relayed()); relay_ledger.sync(&net_holdings);
                        // Only a connection that broke (an I/O error: a reset, a QUIC idle timeout —
                        // what every killed provider's last connection showed, e2e test 30 Oct) can
                        // mean the provider died; one that went idle (keep-alive) or that we closed
                        // ourselves (no cause) is a live provider (review 2026-10-30).
                        let broke = matches!(cause, Some(libp2p::swarm::ConnectionError::IO(_)));
                        if broke && !swarm.is_connected(&p) && gone_watch.len() < MAX_GONE_WATCH && registry.serves(&p.to_base58()) {
                            // Mesh verification 30 Oct: how a provider's last connection ends (a
                            // crash vs an idle close vs a clean exit) decides whether "gone" is right.
                            info!(%p, ?cause, relayed = endpoint.is_relayed(), "provider's last connection closed: watching it");
                            gone_watch.entry(p).or_insert_with(std::time::Instant::now);
                        }
                    }
                    // Task 7.1: Swarm-level events — no more catch-all
                    libp2p::swarm::SwarmEvent::OutgoingConnectionError { peer_id, connection_id, error } => {
                        if dial_back.finished(&connection_id) {
                            // A peer behind NAT can't be dialled: expected, and tried once an hour.
                            info!(?peer_id, %error, "dial-back failed");
                            continue;
                        }
                        // A failed dial to one of the other bootstraps matters; others (AutoNAT dial-backs a
                        // peer asks for, Kademlia's own dials) are routine and a remote can cause them at will.
                        if peer_id.is_some_and(|p| configured_peers.contains(&p)) {
                            warn!(?peer_id, %error, "outgoing connection error");
                        } else {
                            debug!(?peer_id, %error, "outgoing connection error");
                        }
                        // A routing-table peer we've failed to reach for a while leaves the table, so we
                        // stop dialling it and stop handing it to every agent that asks us for peers
                        // (2026-10-27: two retired seeds were still dialled by all four bootstraps and
                        // every agent, and agents' DHT writes kept timing out on them). Only table
                        // members are counted (peers a query merely learned can't flood the counts),
                        // never the configured bootstrap peers, and the failures must span
                        // DIAL_FAILURE_SPAN so a peer restarting through a burst of dials stays. Nothing
                        // is counted while we reach none of the other bootstraps: our own outage would
                        // otherwise empty the table. Kademlia doesn't re-add a peer that later dials
                        // in; one of our own lookups must dial it.
                        if let Some(p) = peer_id {
                            if dial_failure_means_unreachable(&error)
                                && configured_peers.iter().any(|b| swarm.is_connected(b))
                                && !swarm.is_connected(&p)
                                && !configured_peers.contains(&p)
                                && swarm.behaviour_mut().kademlia.kbucket(p).is_some_and(|b| b.iter().any(|e| *e.node.key.preimage() == p))
                                && count_dial_failure(&mut dial_failures, p, std::time::Instant::now())
                                && swarm.behaviour_mut().kademlia.remove_peer(&p).is_some()
                            {
                                info!(%p, "dropped unreachable peer from the routing table");
                            }
                        }
                    }
                    // A connection a limit turned away is routine (and can come by the thousand).
                    libp2p::swarm::SwarmEvent::IncomingConnectionError {
                        error: error @ libp2p::swarm::ListenError::Denied { .. }, send_back_addr, ..
                    } => {
                        debug!(%error, %send_back_addr, "incoming connection denied");
                    }
                    libp2p::swarm::SwarmEvent::IncomingConnectionError { error, local_addr, send_back_addr, .. } => {
                        debug!(%error, %local_addr, %send_back_addr, "incoming connection error");
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

/// Every value of a repeatable multiaddr flag (`None` when it isn't given). A value that doesn't
/// parse is an error, not skipped: a typo in `--external` used to start the bootstrap without the
/// address its relay needs, silently (review 2026-10-14, L12).
fn parse_flag_multi(args: &[String], name: &str) -> Result<Option<Vec<Multiaddr>>, String> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == name {
            let val = args.get(i + 1).ok_or_else(|| format!("{name} needs a value"))?;
            result.push(val.parse().map_err(|e| format!("{name} {val}: not a multiaddr ({e})"))?);
            i += 2;
        } else {
            i += 1;
        }
    }
    Ok((!result.is_empty()).then_some(result))
}

/// Whether a failed dial says the peer can't be reached at what we know of it: no address answered,
/// or none is known. Not `WrongPeerId`: someone else answering at an address another peer reported
/// for it says nothing about the peer itself, and would let a liar evict an honest peer.
fn dial_failure_means_unreachable(error: &libp2p::swarm::DialError) -> bool {
    // Nothing answered at any address (review 2026-10-29: any transport error counted, so a peer
    // that closed the connection mid-handshake because it was full counted as dead).
    openhydra_network::transport::dial_means_unreachable(error)
}

/// Failed dials (with no connection in between) before a peer leaves the routing table, and the time
/// they must span from the first: one blip, or a restart caught by a burst of dials (the hourly
/// record re-replication), shouldn't drop a good peer.
const DIAL_FAILURES_BEFORE_DROP: u32 = 3;
const DIAL_FAILURE_SPAN: std::time::Duration = std::time::Duration::from_secs(600);
/// Bound on the failure-count map (only routing-table members are counted, so it stays small).
const DIAL_FAILURES_TRACKED: usize = 10_000;

/// A dial-back to a peer that connected to us but isn't in the routing table (review 2026-10-29: in
/// 24 h on the netcup bootstrap, 36 peers connected, 29 were "unroutable" and the table gained none).
/// Kademlia only lists a peer it dialled, so a public node that only ever dials out was never listed,
/// never handed out, and never held records. We dial it back once, at the listen addresses its
/// identify gives on the IP its connection really comes from (no third-party address can be put in
/// that way); if it answers, Kademlia lists it the normal way (through the routing-table filter).
#[derive(Debug, Default)]
struct DialBack {
    /// The IP each peer's direct inbound connection comes from.
    inbound_ip: HashMap<libp2p::PeerId, std::net::IpAddr>,
    /// The addresses to dial each peer back at.
    addrs: HashMap<libp2p::PeerId, Vec<Multiaddr>>,
    /// Peers Kademlia reported unroutable and we haven't dialled back yet.
    unroutable: std::collections::HashSet<libp2p::PeerId>,
    /// When each peer was last dialled back.
    last_try: HashMap<libp2p::PeerId, std::time::Instant>,
    /// Dial-backs still in flight.
    in_flight: std::collections::HashSet<libp2p::swarm::ConnectionId>,
    /// Dial-backs per IP in the current hour: one machine can present any number of peer ids, so
    /// it gets [`DIAL_BACKS_PER_IP`] of them listed an hour, not one per id (review 2026-10-29).
    per_ip: HashMap<std::net::IpAddr, (u32, std::time::Instant)>,
}

/// Dial-backs per IP per [`DIAL_BACK_EVERY`], at most.
const DIAL_BACKS_PER_IP: u32 = 2;

/// One dial-back per peer per this long, at most.
const DIAL_BACK_EVERY: std::time::Duration = std::time::Duration::from_secs(3600);
/// Dial-backs in flight, at most.
const DIAL_BACKS_IN_FLIGHT: usize = 16;
/// Listen addresses dialled back per peer, at most.
const DIAL_BACK_ADDRS: usize = 4;
/// Peers tracked, at most (past it the tables start over).
const DIAL_BACK_TRACKED: usize = 10_000;

impl DialBack {
    fn bound(&mut self) {
        if self.inbound_ip.len() >= DIAL_BACK_TRACKED {
            self.inbound_ip.clear();
            self.addrs.clear();
            self.unroutable.clear();
        }
        if self.last_try.len() >= DIAL_BACK_TRACKED {
            self.last_try.clear();
        }
        if self.per_ip.len() >= DIAL_BACK_TRACKED {
            self.per_ip.clear();
        }
    }

    /// A direct inbound connection from `peer` at `remote` (relayed ones say nothing about it).
    fn inbound(&mut self, peer: libp2p::PeerId, remote: &Multiaddr) {
        if remote.iter().any(|p| matches!(p, libp2p::multiaddr::Protocol::P2pCircuit)) {
            return;
        }
        if let Some(ip) = remote.iter().find_map(|p| match p {
            libp2p::multiaddr::Protocol::Ip4(ip) => Some(std::net::IpAddr::V4(ip)),
            libp2p::multiaddr::Protocol::Ip6(ip) => Some(std::net::IpAddr::V6(ip)),
            _ => None,
        }) {
            self.bound();
            self.inbound_ip.insert(peer, ip);
        }
    }

    /// `peer`'s identify: keep its listen addresses on its connection's IP that `allowed` passes.
    fn identified(&mut self, peer: libp2p::PeerId, listen: &[Multiaddr], allowed: impl Fn(&Multiaddr) -> bool) {
        let Some(ip) = self.inbound_ip.get(&peer) else { return };
        let ip_part = match ip {
            std::net::IpAddr::V4(v4) => libp2p::multiaddr::Protocol::Ip4(*v4),
            std::net::IpAddr::V6(v6) => libp2p::multiaddr::Protocol::Ip6(*v6),
        };
        let addrs: Vec<Multiaddr> = listen
            .iter()
            .filter(|a| a.iter().next() == Some(ip_part.clone()))
            .filter(|a| !a.iter().any(|p| matches!(p, libp2p::multiaddr::Protocol::P2pCircuit)))
            .filter(|a| allowed(a))
            .take(DIAL_BACK_ADDRS)
            .cloned()
            .collect();
        if !addrs.is_empty() {
            self.addrs.insert(peer, addrs);
        }
    }

    /// Kademlia reported `peer` unroutable.
    fn unroutable(&mut self, peer: libp2p::PeerId) {
        self.bound();
        self.unroutable.insert(peer);
    }

    /// If `peer` is due a dial-back now, the addresses to dial (and it's counted as tried).
    fn due(&mut self, peer: &libp2p::PeerId, now: std::time::Instant) -> Option<Vec<Multiaddr>> {
        if !self.unroutable.contains(peer) || self.in_flight.len() >= DIAL_BACKS_IN_FLIGHT {
            return None;
        }
        if self.last_try.get(peer).is_some_and(|t| now.duration_since(*t) < DIAL_BACK_EVERY) {
            return None;
        }
        let addrs = self.addrs.get(peer)?.clone();
        let ip = *self.inbound_ip.get(peer)?;
        let (n, since) = self.per_ip.entry(ip).or_insert((0, now));
        if now.duration_since(*since) >= DIAL_BACK_EVERY {
            (*n, *since) = (0, now);
        }
        if *n >= DIAL_BACKS_PER_IP {
            return None;
        }
        *n += 1;
        self.unroutable.remove(peer);
        self.last_try.insert(*peer, now);
        Some(addrs)
    }

    fn dialled(&mut self, id: libp2p::swarm::ConnectionId) {
        self.in_flight.insert(id);
    }

    fn finished(&mut self, id: &libp2p::swarm::ConnectionId) -> bool {
        self.in_flight.remove(id)
    }
}

/// Dial `peer` back if it's due ([`DialBack`]). `Always`: it's connected already, inbound. From a
/// new port, not our listening socket: a QUIC dial from :4001 is the exact reverse of the flow the peer
/// opened to us, so its own firewall lets it in, and "reachable" would then mean reachable by us alone
/// (review 2026-10-29). From a fresh port it reaches the peer only if anyone could.
fn maybe_dial_back(swarm: &mut Swarm<BootstrapBehaviour>, dial_back: &mut DialBack, peer: libp2p::PeerId) {
    let Some(addrs) = dial_back.due(&peer, std::time::Instant::now()) else { return };
    let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(peer)
        .condition(libp2p::swarm::dial_opts::PeerCondition::Always)
        .addresses(addrs)
        .allocate_new_port()
        .build();
    let id = opts.connection_id();
    match swarm.dial(opts) {
        Ok(()) => {
            dial_back.dialled(id);
            info!(%peer, "dialling back a peer that isn't in the routing table");
        }
        Err(e) => debug!(%peer, %e, "dial-back not started"),
    }
}

/// Failures older than this are forgotten (review 2026-10-29: the count only reset on a connection,
/// so three blips weeks apart dropped a healthy peer).
const DIAL_FAILURE_WINDOW: std::time::Duration = std::time::Duration::from_secs(3600);

/// Whether this bootstrap is one of OpenHydra's own (its peer id is a production relay's): only then
/// is it on the public network, whatever addresses it listens on.
fn public_bootstrap(peer_id: &libp2p::PeerId) -> bool {
    openhydra_network::relay::bootstrap_relay_peer_ids().contains(peer_id)
}

/// Count a failed dial at `now`; true when it's the one that should drop the peer (its count then
/// resets): at least DIAL_FAILURES_BEFORE_DROP failures within DIAL_FAILURE_WINDOW, spanning
/// DIAL_FAILURE_SPAN.
fn count_dial_failure(
    failures: &mut HashMap<libp2p::PeerId, (u32, std::time::Instant)>,
    peer: libp2p::PeerId,
    now: std::time::Instant,
) -> bool {
    if failures.len() >= DIAL_FAILURES_TRACKED && !failures.contains_key(&peer) {
        failures.clear();
    }
    let (n, first) = failures.entry(peer).or_insert((0, now));
    if now.duration_since(*first) > DIAL_FAILURE_WINDOW {
        (*n, *first) = (0, now);
    }
    *n += 1;
    if *n >= DIAL_FAILURES_BEFORE_DROP && now.duration_since(*first) >= DIAL_FAILURE_SPAN {
        failures.remove(&peer);
        true
    } else {
        false
    }
}

/// The relay's live reservations and circuits, for `relay_metrics`.
///
/// libp2p-relay 0.18 keeps the true counts private and its events don't carry what's needed to
/// mirror them: a reservation freed by its connection closing emits no event (only one that runs
/// out unrenewed does), and a circuit whose destination's connection closes first is reported
/// closed twice (`ConnectionAborted` from the close, then again from the source's handler).
/// Counting events alone, netcup's `active_reservations` passed 1,000 within three days with 4–8
/// peers connected, and 22 of its 165 circuit closes in that time were such repeats, each taking
/// one off whatever other circuits were open.
///
/// The events don't say which connection a reservation is on, so the count per peer is held to
/// what's possible: no more reservations than direct connections open (a reservation lives on one;
/// relayed connections can't hold one). That covers a reservation freed with its connection, the
/// usual case. A peer whose reserved connection closes while another direct one stays open is
/// counted one too many until that one closes too or it reserves again (which a client does).
/// Circuits are counted per source → destination pair, so a duplicate close can only undercount
/// that pair, and a pair's circuits end when either end's last direct connection does.
#[derive(Default)]
struct RelayLedger {
    /// Direct (not relayed) connections open, per peer.
    direct: HashMap<libp2p::PeerId, usize>,
    /// Reservations held, per peer.
    reservations: HashMap<libp2p::PeerId, usize>,
    /// Circuits open, per (source, destination).
    circuits: HashMap<(libp2p::PeerId, libp2p::PeerId), usize>,
}

impl RelayLedger {
    fn connection_established(&mut self, peer: libp2p::PeerId, relayed: bool) {
        if !relayed {
            *self.direct.entry(peer).or_default() += 1;
        }
    }

    fn connection_closed(&mut self, peer: libp2p::PeerId, relayed: bool) {
        if relayed {
            return;
        }
        let open = match self.direct.get_mut(&peer) {
            Some(n) => {
                *n = n.saturating_sub(1);
                *n
            }
            None => 0,
        };
        if open == 0 {
            self.direct.remove(&peer);
            self.circuits.retain(|(src, dst), _| *src != peer && *dst != peer);
        }
        self.hold_to_connections(peer);
    }

    fn reservation_accepted(&mut self, peer: libp2p::PeerId, renewed: bool) {
        let held = self.reservations.entry(peer).or_default();
        if !renewed || *held == 0 {
            *held += 1;
        }
        // An acceptance reported after its connection closed is already gone.
        self.hold_to_connections(peer);
    }

    /// The accept couldn't be sent. libp2p-relay took the slot when it decided to accept and
    /// keeps it until the connection closes (no timeout comes for it: the handler never started
    /// one), so it counts as held — once: a failed renewal of a held one takes nothing more.
    fn reservation_accept_failed(&mut self, peer: libp2p::PeerId) {
        let held = self.reservations.entry(peer).or_default();
        if *held == 0 {
            *held = 1;
        }
        self.hold_to_connections(peer);
    }

    fn reservation_timed_out(&mut self, peer: libp2p::PeerId) {
        if let Some(held) = self.reservations.get_mut(&peer) {
            *held = held.saturating_sub(1);
            if *held == 0 {
                self.reservations.remove(&peer);
            }
        }
    }

    fn circuit_accepted(&mut self, src: libp2p::PeerId, dst: libp2p::PeerId) {
        *self.circuits.entry((src, dst)).or_default() += 1;
    }

    fn circuit_closed(&mut self, src: libp2p::PeerId, dst: libp2p::PeerId) {
        if let Some(n) = self.circuits.get_mut(&(src, dst)) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                self.circuits.remove(&(src, dst));
            }
        }
    }

    fn hold_to_connections(&mut self, peer: libp2p::PeerId) {
        let open = self.direct.get(&peer).copied().unwrap_or(0);
        if let Some(held) = self.reservations.get_mut(&peer) {
            *held = (*held).min(open);
            if *held == 0 {
                self.reservations.remove(&peer);
            }
        }
    }

    /// Recount what each network holds (see `relay::NetHoldings`) after any change here.
    fn sync(&self, holdings: &std::sync::Mutex<openhydra_network::relay::NetHoldings>) {
        let mut by_src: HashMap<libp2p::PeerId, usize> = HashMap::new();
        for ((src, _), n) in &self.circuits {
            *by_src.entry(*src).or_default() += n;
        }
        if let Ok(mut h) = holdings.lock() {
            h.recompute(self.reservations.iter().map(|(p, n)| (*p, *n)), by_src.into_iter());
        }
    }

    fn reservations(&self) -> usize {
        self.reservations.values().sum()
    }

    fn circuits(&self) -> usize {
        self.circuits.values().sum()
    }
}

/// Failed pings, as libp2p reports them, after which a peer's connection can count as left behind
/// ([`MovedConns`]). libp2p keeps the first failure on a connection silent, so 2 reported means 3
/// in a row: with 15 s pings and the 20 s timeout, about a minute.
const MOVED_PING_FAILURES: u32 = 2;

/// How long a connection's pings must have been failing, from the first reported failure, before it
/// can count as left behind ([`MovedConns`]): about three reported failures. A host with two ways out
/// (a dual-ISP router, a cloud NAT with several addresses) can have connections from two /16s at
/// once; this asks its old connection to have answered nothing for over a minute and a half in all
/// (review 2026-10-29), which a connection that still works doesn't do.
const MOVED_MIN_FAILING: std::time::Duration = std::time::Duration::from_secs(60);

/// One direct connection of a peer's, for [`MovedConns`].
struct PeerConn {
    peer: libp2p::PeerId,
    ip: std::net::IpAddr,
    since: std::time::Instant,
    /// Reported ping failures in a row, and when the first of them came.
    failing: Option<(u32, std::time::Instant)>,
    last_ok: Option<std::time::Instant>,
}

/// Connections a peer left behind when it moved to another network. Nothing closes them: the
/// peer's side is gone without a word, TCP has no keepalive here and ping never closes a
/// connection, so they sat for ~15 min on all four bootstraps after a node changed Wi-Fi
/// (2026-10-29), and the relay could route circuits for the peer onto one of them.
///
/// A connection counts as left behind once its pings have failed [`MOVED_PING_FAILURES`] times in a
/// row and the same peer has a newer connection from another network — another IPv4 /16 or IPv6
/// /64 — that has answered a ping since those failures began. Failed pings alone never close one:
/// a connection carrying a big relayed stream can queue its pings behind the data (one TCP byte
/// stream), and the host's other connections come from the same addresses, so they never count.
#[derive(Default)]
struct MovedConns {
    conns: HashMap<libp2p::swarm::ConnectionId, PeerConn>,
    /// Each peer's tracked connections, so a check looks at its own (≤ the per-peer limit), not all.
    by_peer: HashMap<libp2p::PeerId, Vec<libp2p::swarm::ConnectionId>>,
}

/// Whether two of a peer's IPs are on different networks: another IPv4 /16 (a NAT pool hands out
/// addresses within one), or another IPv6 /64.
fn other_network(a: std::net::IpAddr, b: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match (a, b) {
        (IpAddr::V4(a), IpAddr::V4(b)) => a.octets()[..2] != b.octets()[..2],
        (IpAddr::V6(a), IpAddr::V6(b)) => a.segments()[..4] != b.segments()[..4],
        _ => false,
    }
}

impl MovedConns {
    fn established(&mut self, id: libp2p::swarm::ConnectionId, peer: libp2p::PeerId, ip: std::net::IpAddr, now: std::time::Instant) {
        self.conns.insert(id, PeerConn { peer, ip, since: now, failing: None, last_ok: None });
        self.by_peer.entry(peer).or_default().push(id);
    }

    fn closed(&mut self, id: libp2p::swarm::ConnectionId) {
        if let Some(c) = self.conns.remove(&id) {
            if let Some(ids) = self.by_peer.get_mut(&c.peer) {
                ids.retain(|i| *i != id);
                if ids.is_empty() {
                    self.by_peer.remove(&c.peer);
                }
            }
        }
    }

    fn ping_ok(&mut self, id: libp2p::swarm::ConnectionId, now: std::time::Instant) {
        if let Some(c) = self.conns.get_mut(&id) {
            c.failing = None;
            c.last_ok = Some(now);
        }
    }

    /// For the log of a close: `id`'s IP, and the IPs of its peer's newer connections that answered.
    fn ips_for_close(&self, id: libp2p::swarm::ConnectionId) -> (Option<std::net::IpAddr>, Vec<std::net::IpAddr>) {
        let Some(c) = self.conns.get(&id) else { return (None, Vec::new()) };
        let newer = self
            .by_peer
            .get(&c.peer)
            .into_iter()
            .flatten()
            .filter_map(|i| self.conns.get(i))
            .filter(|o| o.since > c.since && o.last_ok.is_some())
            .map(|o| o.ip)
            .collect();
        (Some(c.ip), newer)
    }

    /// The network (see `ip_limits::origin_key`) of `peer`'s newest direct connection, if it has one.
    fn origin_of(&self, peer: &libp2p::PeerId) -> Option<String> {
        self.by_peer
            .get(peer)?
            .iter()
            .filter_map(|i| self.conns.get(i))
            .max_by_key(|c| c.since)
            .map(|c| openhydra_network::ip_limits::origin_key(c.ip))
    }

    /// Count a failed ping on `id`; true when it should be closed (see [`MovedConns`]).
    fn ping_failed(&mut self, id: libp2p::swarm::ConnectionId, now: std::time::Instant) -> bool {
        let Some(c) = self.conns.get_mut(&id) else { return false };
        let (n, from) = c.failing.get_or_insert((0, now));
        *n += 1;
        if *n < MOVED_PING_FAILURES || now.duration_since(*from) < MOVED_MIN_FAILING {
            return false;
        }
        let (peer, ip, since, from) = (c.peer, c.ip, c.since, *from);
        self.by_peer.get(&peer).into_iter().flatten().filter_map(|i| self.conns.get(i)).any(|o| {
            o.since > since && other_network(o.ip, ip) && o.last_ok.is_some_and(|t| t >= from)
        })
    }
}

/// The IP in a multiaddr, if it has one.
fn multiaddr_ip(addr: &Multiaddr) -> Option<std::net::IpAddr> {
    addr.iter().find_map(|p| match p {
        libp2p::multiaddr::Protocol::Ip4(ip) => Some(ip.into()),
        libp2p::multiaddr::Protocol::Ip6(ip) => Some(ip.into()),
        _ => None,
    })
}

/// What a bootstrap does with a gossip message: forward it (Accept), drop it as bad (Reject: a
/// forgery, a malformed or oversized one), or drop it quietly (Ignore: stale, unknown, over rate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GossipVerdict {
    Accept,
    Reject(&'static str),
    Ignore(&'static str),
}

/// The most a provider announce may be (a signed record is under 8 KB), and any other message.
const MAX_ANNOUNCE_BYTES: usize = 16 * 1024;
const MAX_NOTICE_BYTES: usize = 1024;

/// Whether a bootstrap forwards this gossip message (it validates every one: `validate_messages`).
/// Before, it forwarded anything up to 64 KB to its whole mesh, unread (review 2026-10-30: a flood
/// amplifier). Only what OpenHydra nodes publish passes, each checked as consumers check it:
/// - `PROVIDER_ANNOUNCE`: a record signed by the message's author, with sane content, not over an
///   hour old or dated ahead;
/// - `MODEL_WITHDRAWN` / `PEER_DEPARTED`: about the author itself (`MODEL_WITHDRAWN` also fresh);
/// - `PROVIDER_GONE`: only from another bootstrap of ours, fresh.
/// Unknown types are ignored (not forwarded): a new type needs the bootstraps updated first. Pure.
fn check_gossip(
    data: &[u8],
    author: Option<libp2p::PeerId>,
    bootstraps: &std::collections::HashSet<libp2p::PeerId>,
    now_ms: u64,
) -> GossipVerdict {
    use openhydra_network::event_loop as el;
    let Some(author) = author else { return GossipVerdict::Reject("no author") };
    if data.len() > MAX_ANNOUNCE_BYTES {
        return GossipVerdict::Reject("too big");
    }
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(data) else { return GossipVerdict::Reject("not JSON") };
    let Some(kind) = v.get("type").and_then(|t| t.as_str()) else { return GossipVerdict::Reject("no type") };
    if kind != el::PROVIDER_ANNOUNCE_TYPE && data.len() > MAX_NOTICE_BYTES {
        return GossipVerdict::Reject("too big for its type");
    }
    let about_author = || v.get("libp2p_peer_id").and_then(|p| p.as_str()) == Some(author.to_base58().as_str());
    let fresh = |max_secs: u64| {
        v.get("timestamp").and_then(|t| t.as_u64()).is_some_and(|ts| (now_ms / 1000).abs_diff(ts) <= max_secs)
    };
    match kind {
        k if k == el::PROVIDER_ANNOUNCE_TYPE => {
            let Some(rec) = v.get("record").and_then(|r| serde_json::from_value::<openhydra_network::types::PeerRecord>(r.clone()).ok()) else {
                return GossipVerdict::Reject("announce without a record");
            };
            if let Err(_) = openhydra_network::dht::check_record_content(&rec) {
                return GossipVerdict::Reject("announce content");
            }
            if openhydra_network::dht::check_record_age(&rec, now_ms).is_err() {
                return GossipVerdict::Ignore("announce too old or dated ahead");
            }
            if openhydra_network::dht::pex_record_is_authentic(&rec, &author.to_base58()).is_err() {
                return GossipVerdict::Reject("announce not signed by its author");
            }
            GossipVerdict::Accept
        }
        k if k == el::MODEL_WITHDRAWN_TYPE => {
            if !about_author() {
                return GossipVerdict::Reject("withdrawal for another peer");
            }
            if v.get("model_id").and_then(|m| m.as_str()).is_none_or(|m| m.is_empty() || m.len() > 256) {
                return GossipVerdict::Reject("withdrawal without a model");
            }
            if !fresh(el::MODEL_WITHDRAWN_MAX_AGE_SECS) {
                return GossipVerdict::Ignore("stale withdrawal");
            }
            GossipVerdict::Accept
        }
        "PEER_DEPARTED" => {
            if about_author() { GossipVerdict::Accept } else { GossipVerdict::Reject("departure for another peer") }
        }
        k if k == el::PROVIDER_GONE_TYPE => {
            if !bootstraps.contains(&author) {
                return GossipVerdict::Reject("gone notice not from a bootstrap");
            }
            if !fresh(el::PROVIDER_GONE_MAX_AGE_SECS) {
                return GossipVerdict::Ignore("stale gone notice");
            }
            GossipVerdict::Accept
        }
        _ => GossipVerdict::Ignore("unknown type"),
    }
}

/// Gossip messages per author a bootstrap forwards: a burst (a provider re-announces each of its up
/// to 128 models at once, every 2 minutes) then a few a second. Per author, not per sender: a
/// message reaches us from many neighbours, but gossipsub hands each to us once.
#[derive(Default)]
struct AuthorRate {
    buckets: HashMap<libp2p::PeerId, (f64, std::time::Instant)>,
}

const AUTHOR_BURST: f64 = 300.0;
const AUTHOR_PER_SEC: f64 = 3.0;
const MAX_AUTHORS: usize = 16_384;

impl AuthorRate {
    fn allow(&mut self, author: libp2p::PeerId, now: std::time::Instant) -> bool {
        if self.buckets.len() >= MAX_AUTHORS && !self.buckets.contains_key(&author) {
            self.buckets.retain(|_, (t, at)| *t + now.saturating_duration_since(*at).as_secs_f64() * AUTHOR_PER_SEC < AUTHOR_BURST);
            if self.buckets.len() >= MAX_AUTHORS {
                if let Some(k) = self.buckets.iter().min_by_key(|(_, (_, at))| *at).map(|(k, _)| *k) {
                    self.buckets.remove(&k);
                }
            }
        }
        let (tokens, at) = self.buckets.entry(author).or_insert((AUTHOR_BURST, now));
        *tokens = (*tokens + now.saturating_duration_since(*at).as_secs_f64() * AUTHOR_PER_SEC).min(AUTHOR_BURST);
        *at = now;
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    /// Review 2026-10-30: bootstraps forwarded any payload up to 64 KB unread; now only what
    /// OpenHydra nodes publish, checked as consumers check it.
    #[test]
    fn gossip_validation_forwards_only_real_messages() {
        use openhydra_network::types::PeerRecord;
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let author = libp2p::PeerId::from(kp.public());
        let other = libp2p::PeerId::random();
        let boot = libp2p::PeerId::random();
        let boots: std::collections::HashSet<_> = [boot].into_iter().collect();
        let now = 2_000_000_000_000u64;
        let rec = |t: u64| openhydra_network::dht::sign_peer_record(
            &PeerRecord { model_id: "qwen2.5:0.5b".into(), host: "1.2.3.4".into(), port: 1, updated_unix_ms: t, ..Default::default() }, &kp).unwrap();
        let announce = |r: &PeerRecord| serde_json::to_vec(&serde_json::json!({ "type": "PROVIDER_ANNOUNCE", "record": r })).unwrap();
        let c = |d: &[u8], a: libp2p::PeerId| check_gossip(d, Some(a), &boots, now);
        // Announces: signed by the author, fresh (or undated, from older nodes).
        assert_eq!(c(&announce(&rec(now - 1_000)), author), GossipVerdict::Accept);
        assert_eq!(c(&announce(&rec(0)), author), GossipVerdict::Accept);
        assert!(matches!(c(&announce(&rec(now - 1_000)), other), GossipVerdict::Reject(_)), "someone else's record");
        let mut tampered = rec(now - 1_000); tampered.port = 2;
        assert!(matches!(c(&announce(&tampered), author), GossipVerdict::Reject(_)));
        assert!(matches!(c(&announce(&rec(now - 4_000_000)), author), GossipVerdict::Ignore(_)), "over an hour old");
        // Withdrawals and departures: about the author itself; withdrawals fresh.
        let notice = |v: serde_json::Value| serde_json::to_vec(&v).unwrap();
        let w = |p: &libp2p::PeerId, ts: u64| notice(serde_json::json!({ "type": "MODEL_WITHDRAWN", "libp2p_peer_id": p.to_base58(), "model_id": "m", "timestamp": ts }));
        assert_eq!(c(&w(&author, now / 1000), author), GossipVerdict::Accept);
        assert!(matches!(c(&w(&other, now / 1000), author), GossipVerdict::Reject(_)));
        assert!(matches!(c(&w(&author, now / 1000 - 1_000), author), GossipVerdict::Ignore(_)));
        let d = |p: &libp2p::PeerId| notice(serde_json::json!({ "type": "PEER_DEPARTED", "libp2p_peer_id": p.to_base58(), "timestamp": now / 1000 }));
        assert_eq!(c(&d(&author), author), GossipVerdict::Accept);
        assert!(matches!(c(&d(&other), author), GossipVerdict::Reject(_)));
        // Gone notices: from another bootstrap of ours only, fresh.
        let g = |ts: u64| notice(serde_json::json!({ "type": "PROVIDER_GONE", "libp2p_peer_id": other.to_base58(), "last_record_ms": 5, "timestamp": ts }));
        assert_eq!(c(&g(now / 1000), boot), GossipVerdict::Accept);
        assert!(matches!(c(&g(now / 1000), author), GossipVerdict::Reject(_)), "a node can't report others gone");
        assert!(matches!(c(&g(now / 1000 - 1_000), boot), GossipVerdict::Ignore(_)));
        // Anything else.
        assert!(matches!(c(br#"{"type":"SOMETHING_NEW"}"#, author), GossipVerdict::Ignore(_)));
        assert!(matches!(c(b"not json", author), GossipVerdict::Reject(_)));
        assert!(matches!(c(&vec![b' '; MAX_ANNOUNCE_BYTES + 1], author), GossipVerdict::Reject(_)), "too big");
        let padded = notice(serde_json::json!({ "type": "PEER_DEPARTED", "libp2p_peer_id": author.to_base58(), "pad": "x".repeat(MAX_NOTICE_BYTES) }));
        assert!(matches!(c(&padded, author), GossipVerdict::Reject(_)), "a notice padded past its size");
        assert!(matches!(check_gossip(&d(&author), None, &boots, now), GossipVerdict::Reject(_)), "unsigned");
    }

    #[test]
    fn an_author_gets_a_burst_then_a_few_a_second() {
        let mut r = AuthorRate::default();
        let (a, b) = (libp2p::PeerId::random(), libp2p::PeerId::random());
        let t0 = std::time::Instant::now();
        for _ in 0..AUTHOR_BURST as usize {
            assert!(r.allow(a, t0));
        }
        assert!(!r.allow(a, t0), "burst used up");
        assert!(r.allow(b, t0), "another author has its own");
        assert!(r.allow(a, t0 + std::time::Duration::from_millis(400)), "refills at a few a second");
    }

    use super::*;

    #[test]
    fn a_connection_is_closed_for_failed_pings_only_when_its_peer_answers_from_another_network() {
        let p = libp2p::PeerId::random();
        let conn = libp2p::swarm::ConnectionId::new_unchecked;
        let ip = |s: &str| -> std::net::IpAddr { s.parse().unwrap() };
        let t0 = std::time::Instant::now();
        let at = |s: u64| t0 + std::time::Duration::from_secs(s);
        let mut m = MovedConns::default();
        // On ACT: v4 and v6 connections; the v4 one carries a big stream and its pings fail.
        m.established(conn(1), p, ip("49.207.231.148"), at(0));
        m.established(conn(2), p, ip("2406:7400:56:9e97::4"), at(1));
        m.ping_ok(conn(2), at(30));
        for s in [40, 55, 70, 85] {
            assert!(!m.ping_failed(conn(1), at(s)), "no newer connection from elsewhere");
        }
        // Another connection from the same host (same /16, same /64): still never.
        m.established(conn(3), p, ip("49.207.9.9"), at(90));
        m.established(conn(4), p, ip("2406:7400:56:9e97::99"), at(90));
        m.ping_ok(conn(3), at(100));
        m.ping_ok(conn(4), at(100));
        assert!(!m.ping_failed(conn(1), at(100)));
        // It moves to Airtel: new connections from there. One that hasn't answered yet proves nothing.
        m.ping_ok(conn(1), at(110));
        m.established(conn(5), p, ip("106.192.224.140"), at(120));
        m.established(conn(6), p, ip("2401:4900:c907:8361::4"), at(120));
        assert!(!m.ping_failed(conn(1), at(130)));
        assert!(!m.ping_failed(conn(1), at(145)), "the new ones haven't answered a ping");
        // Once one of the same family has (after the failures began), the old one goes at its next
        // failure; the other family's answer says nothing (a busy v4 and the same host's idle v6).
        m.ping_ok(conn(6), at(150));
        assert!(!m.ping_failed(conn(1), at(160)), "v6 answering says nothing about v4");
        assert!(!m.ping_failed(conn(2), at(160)));
        assert!(!m.ping_failed(conn(2), at(175)), "its answer came before these failures began");
        // v4: the Airtel v4 connection answers; the old one has failed since 130 s, but not yet for
        // MOVED_MIN_FAILING (a two-line host's busy connection): kept, then closed.
        m.ping_ok(conn(5), at(178));
        assert!(!m.ping_failed(conn(1), at(179)), "failing 49 s: not yet");
        assert!(m.ping_failed(conn(1), at(195)), "v4 left behind: the peer answers from another /16");
        // v6 likewise, once the new v6 connection has answered since its failures began (160 s).
        m.ping_ok(conn(6), at(200));
        assert!(!m.ping_failed(conn(2), at(210)), "failing 50 s: not yet");
        assert!(m.ping_failed(conn(2), at(225)), "v6 left behind: the peer answers from another /64");
        // A success resets the count; the newest connections are never judged by older ones.
        m.ping_ok(conn(2), at(230));
        assert!(!m.ping_failed(conn(2), at(240)));
        for s in [250, 265, 280, 295, 310] {
            assert!(!m.ping_failed(conn(6), at(s)));
        }
        // Another peer's connections don't count, and closed ones are forgotten.
        let q = libp2p::PeerId::random();
        m.established(conn(7), q, ip("49.207.231.148"), at(0));
        assert!(!m.ping_failed(conn(7), at(300)) && !m.ping_failed(conn(7), at(315)) && !m.ping_failed(conn(7), at(400)));
        for c in 1..=7 {
            m.closed(conn(c));
        }
        assert!(m.conns.is_empty() && m.by_peer.is_empty());
        assert!(!m.ping_failed(conn(1), at(400)));
    }

    #[test]
    fn an_advert_counts_against_the_network_its_author_connects_from() {
        let p = libp2p::PeerId::random();
        let conn = libp2p::swarm::ConnectionId::new_unchecked;
        let t0 = std::time::Instant::now();
        let mut m = MovedConns::default();
        assert_eq!(m.origin_of(&p), None, "not connected directly: no origin");
        m.established(conn(1), p, "49.207.231.148".parse().unwrap(), t0);
        assert_eq!(m.origin_of(&p).as_deref(), Some("49.207.231.148"));
        m.established(conn(2), p, "2406:7400:56:9e97:c168:b2b7:3a1e:a913".parse().unwrap(), t0 + std::time::Duration::from_secs(1));
        assert_eq!(m.origin_of(&p).as_deref(), Some("2406:7400:56::/48"), "the newest connection's /48");
        m.closed(conn(1));
        m.closed(conn(2));
        assert_eq!(m.origin_of(&p), None);
    }

    #[test]
    fn a_reservation_whose_accept_failed_holds_its_slot_until_the_connection_closes() {
        let a = libp2p::PeerId::random();
        let mut l = RelayLedger::default();
        l.connection_established(a, false);
        l.reservation_accept_failed(a);
        assert_eq!(l.reservations(), 1, "libp2p-relay keeps the slot");
        l.reservation_accepted(a, true);
        l.reservation_accept_failed(a);
        assert_eq!(l.reservations(), 1, "a failed renewal takes nothing more");
        l.connection_closed(a, false);
        assert_eq!(l.reservations(), 0);
        // With no direct connection (it closed first), nothing is held.
        l.reservation_accept_failed(a);
        assert_eq!(l.reservations(), 0);
    }

    #[test]
    fn a_reservation_freed_with_its_connection_leaves_the_count() {
        let (a, b) = (libp2p::PeerId::random(), libp2p::PeerId::random());
        let mut l = RelayLedger::default();
        l.connection_established(a, false);
        l.connection_established(b, false);
        l.reservation_accepted(a, false);
        l.reservation_accepted(b, false);
        assert_eq!(l.reservations(), 2);
        l.reservation_accepted(a, true);
        l.reservation_accepted(a, true);
        assert_eq!(l.reservations(), 2, "renewals aren't new reservations");
        l.connection_closed(a, false);
        assert_eq!(l.reservations(), 1, "freed silently with its connection");
        l.reservation_timed_out(b);
        assert_eq!(l.reservations(), 0);
        l.connection_closed(b, false);
        assert!(l.direct.is_empty() && l.reservations.is_empty(), "nothing left behind");
    }

    #[test]
    fn a_peer_that_reconnects_holds_one_reservation() {
        let a = libp2p::PeerId::random();
        let mut l = RelayLedger::default();
        l.connection_established(a, false);
        l.reservation_accepted(a, false);
        // It reconnects and reserves on the new connection before the old one has closed.
        l.connection_established(a, false);
        l.reservation_accepted(a, false);
        assert_eq!(l.reservations(), 2, "two connections, two reservations, for now");
        l.connection_closed(a, false);
        assert_eq!(l.reservations(), 1);
        // Hours of reconnects don't add up.
        for _ in 0..100 {
            l.connection_established(a, false);
            l.reservation_accepted(a, false);
            l.connection_closed(a, false);
        }
        assert_eq!(l.reservations(), 1);
    }

    #[test]
    fn relayed_connections_hold_no_reservation() {
        let a = libp2p::PeerId::random();
        let mut l = RelayLedger::default();
        l.connection_established(a, false);
        l.connection_established(a, true);
        l.reservation_accepted(a, false);
        l.connection_closed(a, false);
        assert_eq!(l.reservations(), 0, "a relayed connection left open can't be holding it");
        l.connection_closed(a, true);
        assert!(l.direct.is_empty());
    }

    #[test]
    fn an_acceptance_reported_after_its_connection_closed_counts_nothing() {
        let a = libp2p::PeerId::random();
        let mut l = RelayLedger::default();
        l.connection_established(a, false);
        l.connection_closed(a, false);
        l.reservation_accepted(a, false);
        assert_eq!(l.reservations(), 0);
        l.reservation_timed_out(a);
        assert_eq!(l.reservations(), 0);
    }

    #[test]
    fn a_circuit_reported_closed_twice_counts_once() {
        let (a, b, c) = (libp2p::PeerId::random(), libp2p::PeerId::random(), libp2p::PeerId::random());
        let mut l = RelayLedger::default();
        for p in [a, b, c] {
            l.connection_established(p, false);
        }
        l.circuit_accepted(a, b);
        l.circuit_accepted(c, b);
        assert_eq!(l.circuits(), 2);
        // B's connection closes: the relay reports A → B closed, then A's handler does again.
        l.circuit_closed(a, b);
        l.circuit_closed(a, b);
        assert_eq!(l.circuits(), 1, "C → B is still open");
        // B's last connection closing ends whatever the relay didn't report.
        l.connection_closed(b, false);
        assert_eq!(l.circuits(), 0);
    }

    /// netcup, 28 Oct 21:49–21:50 UTC, as its log has it: K connected twice; G → K had one circuit
    /// left and L → K one; K's first connection closed and each was reported closed twice.
    #[test]
    fn netcups_repeated_closes_count_once() {
        let (g, l, k) = (libp2p::PeerId::random(), libp2p::PeerId::random(), libp2p::PeerId::random());
        let mut ledger = RelayLedger::default();
        for p in [g, l, k, k] {
            ledger.connection_established(p, false);
        }
        ledger.circuit_accepted(g, k);
        ledger.circuit_accepted(g, k);
        ledger.circuit_closed(g, k);
        for _ in 0..3 {
            ledger.circuit_accepted(l, k);
        }
        ledger.circuit_closed(l, k);
        ledger.circuit_closed(l, k);
        assert_eq!(ledger.circuits(), 2);
        ledger.connection_closed(k, false); // K's first connection
        ledger.circuit_closed(l, k); // ConnectionAborted
        ledger.circuit_closed(g, k); // ConnectionAborted
        ledger.circuit_closed(g, k); // again, from G's handler
        ledger.circuit_closed(l, k); // again, from L's handler
        assert_eq!(ledger.circuits(), 0);
        ledger.circuit_accepted(g, l);
        ledger.connection_closed(k, false);
        assert_eq!(ledger.circuits(), 1, "a circuit not through K is untouched");
    }

    #[test]
    fn a_circuit_outlives_one_of_two_connections() {
        let (a, b) = (libp2p::PeerId::random(), libp2p::PeerId::random());
        let mut l = RelayLedger::default();
        l.connection_established(a, false);
        l.connection_established(a, false);
        l.connection_established(b, false);
        l.circuit_accepted(a, b);
        l.connection_closed(a, false);
        assert_eq!(l.circuits(), 1, "A still has a connection; the relay reports the close if it was this one");
        l.connection_closed(a, false);
        assert_eq!(l.circuits(), 0);
    }

    #[test]
    fn only_a_production_bootstrap_filters_its_routing_table() {
        let production = openhydra_network::relay::bootstrap_relay_peer_ids();
        assert!(!production.is_empty());
        assert!(production.iter().all(public_bootstrap));
        assert!(!public_bootstrap(&libp2p::PeerId::random()), "a private or test bootstrap");
    }

    #[test]
    fn a_peer_is_dropped_only_after_three_failed_dials_over_ten_minutes() {
        let (a, b) = (libp2p::PeerId::random(), libp2p::PeerId::random());
        let t0 = std::time::Instant::now();
        let at = |s| t0 + std::time::Duration::from_secs(s);
        let mut failures = HashMap::new();
        assert!(!count_dial_failure(&mut failures, a, at(0)));
        assert!(!count_dial_failure(&mut failures, a, at(1)));
        assert!(!count_dial_failure(&mut failures, a, at(2)), "a burst of three within seconds keeps it");
        assert!(!count_dial_failure(&mut failures, b, at(3)), "counted per peer");
        assert!(count_dial_failure(&mut failures, a, at(600)), "still failing ten minutes on drops it");
        assert!(!failures.contains_key(&a), "and resets its count");
        failures.remove(&b); // a connection in between resets the count
        assert!(!count_dial_failure(&mut failures, b, at(601)));
    }

    #[test]
    fn a_peer_is_dialled_back_once_an_hour_at_its_own_ip_only() {
        let mut d = DialBack::default();
        let p = libp2p::PeerId::random();
        let now = std::time::Instant::now();
        let ma = |s: &str| -> Multiaddr { s.parse().unwrap() };
        d.inbound(p, &ma("/ip4/54.116.38.169/tcp/51868"));
        d.identified(
            p,
            &[
                ma("/ip4/54.116.38.169/tcp/4001"),
                ma("/ip4/54.116.38.169/udp/4001/quic-v1"),
                ma("/ip4/1.2.3.4/tcp/22"),     // someone else's machine
                ma("/ip4/10.0.0.5/tcp/4001"), // private, and not its connection's IP
            ],
            |_| true,
        );
        assert!(d.due(&p, now).is_none(), "not reported unroutable yet");
        d.unroutable(p);
        let addrs = d.due(&p, now).expect("due");
        assert_eq!(addrs, vec![ma("/ip4/54.116.38.169/tcp/4001"), ma("/ip4/54.116.38.169/udp/4001/quic-v1")]);
        d.unroutable(p);
        assert!(d.due(&p, now + std::time::Duration::from_secs(60)).is_none(), "once an hour");
        assert!(d.due(&p, now + DIAL_BACK_EVERY).is_some());
    }

    #[test]
    fn one_ip_gets_two_dial_backs_an_hour_however_many_peer_ids_it_shows() {
        let mut d = DialBack::default();
        let now = std::time::Instant::now();
        let ma = |s: &str| -> Multiaddr { s.parse().unwrap() };
        let mut dialled = 0;
        for i in 0..10u16 {
            let p = libp2p::PeerId::random();
            d.inbound(p, &ma(&format!("/ip4/203.0.113.9/tcp/{}", 50_000 + i)));
            d.identified(p, &[ma(&format!("/ip4/203.0.113.9/tcp/{}", 4001 + i))], |_| true);
            d.unroutable(p);
            dialled += d.due(&p, now).is_some() as u32;
        }
        assert_eq!(dialled, DIAL_BACKS_PER_IP);
        // The next hour, two more.
        let p = libp2p::PeerId::random();
        d.inbound(p, &ma("/ip4/203.0.113.9/tcp/60000"));
        d.identified(p, &[ma("/ip4/203.0.113.9/tcp/4500")], |_| true);
        d.unroutable(p);
        assert!(d.due(&p, now + DIAL_BACK_EVERY).is_some());
    }

    #[test]
    fn a_relayed_peer_or_one_with_no_usable_address_isnt_dialled_back() {
        let mut d = DialBack::default();
        let now = std::time::Instant::now();
        let ma = |s: &str| -> Multiaddr { s.parse().unwrap() };
        let relayed = libp2p::PeerId::random();
        d.inbound(relayed, &ma(&format!("/ip4/45.79.190.172/tcp/4001/p2p/{}/p2p-circuit", libp2p::PeerId::random())));
        d.identified(relayed, &[ma("/ip4/45.79.190.172/tcp/4001")], |_| true);
        d.unroutable(relayed);
        assert!(d.due(&relayed, now).is_none());
        let private = libp2p::PeerId::random();
        d.inbound(private, &ma("/ip4/10.0.0.5/tcp/50000"));
        d.identified(private, &[ma("/ip4/10.0.0.5/tcp/4001")], openhydra_network::swarm::routable_on_public_network);
        d.unroutable(private);
        assert!(d.due(&private, now).is_none(), "a production bootstrap doesn't dial private addresses");
    }

    #[test]
    fn failures_more_than_an_hour_apart_are_forgotten() {
        let a = libp2p::PeerId::random();
        let t0 = std::time::Instant::now();
        let at = |s| t0 + std::time::Duration::from_secs(s);
        let mut failures = HashMap::new();
        // One blip a day, for days: never three within the hour, so never dropped.
        for day in 0..10u64 {
            assert!(!count_dial_failure(&mut failures, a, at(day * 86_400)), "day {day}");
        }
        // Three within the hour, ten minutes apart from first to last: dropped.
        let t = 20 * 86_400;
        assert!(!count_dial_failure(&mut failures, a, at(t)));
        assert!(!count_dial_failure(&mut failures, a, at(t + 300)));
        assert!(count_dial_failure(&mut failures, a, at(t + 600)));
    }

    #[test]
    fn only_a_dial_that_reached_nobody_drops_the_peer() {
        use libp2p::swarm::{dial_opts::PeerCondition, DialError};
        assert!(!dial_failure_means_unreachable(&DialError::Transport(Vec::new())), "nothing tried says nothing");
        assert!(dial_failure_means_unreachable(&DialError::NoAddresses));
        assert!(!dial_failure_means_unreachable(&DialError::WrongPeerId {
            obtained: libp2p::PeerId::random(),
            endpoint: libp2p::core::ConnectedPoint::Dialer {
                address: "/ip4/1.2.3.4/tcp/4001".parse().unwrap(),
                role_override: libp2p::core::Endpoint::Dialer,
                port_use: libp2p::core::transport::PortUse::New,
            },
        }));
        assert!(!dial_failure_means_unreachable(&DialError::Aborted));
        assert!(!dial_failure_means_unreachable(&DialError::DialPeerConditionFalse(PeerCondition::Disconnected)));
    }

    #[test]
    fn a_multiaddr_flag_that_doesnt_parse_is_an_error() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_flag_multi(&args(&["x"]), "--external"), Ok(None));
        let two = parse_flag_multi(&args(&["--external", "/ip4/10.0.0.5/tcp/4001", "--external", "/ip4/10.0.0.6/tcp/4001"]), "--external");
        assert_eq!(two.unwrap().unwrap().len(), 2);
        let typo = parse_flag_multi(&args(&["--external", "/ip4/10.0.0.5/tpc/4001"]), "--external").unwrap_err();
        assert!(typo.contains("--external") && typo.contains("tpc"), "{typo}");
        assert!(parse_flag_multi(&args(&["--external"]), "--external").is_err(), "a missing value");
    }
}
