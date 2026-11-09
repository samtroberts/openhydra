//! Swarm creation and configuration.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use libp2p::core::muxing::StreamMuxerBox;
use libp2p::swarm::behaviour::toggle::Toggle;
use libp2p::swarm::Config as SwarmConfig;
use libp2p::{
    autonat, dcutr, gossipsub, identify, kad, mdns, ping, relay, Multiaddr, PeerId, Swarm,
    Transport,
};

use crate::behaviour::OpenHydraBehaviour;
use crate::relay::{LeechRateLimiter, LeechTable, PER_CIRCUIT_BUDGET_BYTES};

/// PR-3: the single topic that carries all swarm-wide events. Keeping one
/// topic for v1 intentionally bounds the blast radius — future versions can
/// add per-model topics once the coordinator has logic to subscribe /
/// unsubscribe as models come and go.
pub const GOSSIPSUB_TOPIC: &str = "openhydra/swarm/v1/events";

/// Gossipsub as OpenHydra runs it: only [`GOSSIPSUB_TOPIC`] can be subscribed to.
pub type GossipBehaviour = gossipsub::Behaviour<gossipsub::IdentityTransform, gossipsub::WhitelistSubscriptionFilter>;

/// The only topic a peer's subscriptions are kept for. Upstream keeps every topic a peer names, so
/// one connection could subscribe to 100,000 topics (305 MB on a bootstrap, review 2026-10-29).
pub fn gossip_topic_filter() -> gossipsub::WhitelistSubscriptionFilter {
    gossipsub::WhitelistSubscriptionFilter(
        [gossipsub::IdentTopic::new(GOSSIPSUB_TOPIC).hash()].into_iter().collect(),
    )
}

/// The largest gossip message sent or accepted: upstream's 64 KB, the size every deployed node
/// uses. Lowering it isn't safe while other nodes don't: one message between the two limits, which
/// older nodes accept and forward, breaks the stream it reaches a lower-limit node on, and gossip
/// from that peer stops without an error (review 2026-10-29, reproduced at 32 KB).
pub const GOSSIP_MAX_TRANSMIT: usize = 64 * 1024;

/// How long a gossip message id is remembered to drop duplicates. Upstream's 60 s let a flood of
/// distinct messages pile up ids for a minute; every honest copy of a message arrives within a few
/// heartbeats.
pub const GOSSIP_DUPLICATE_CACHE: std::time::Duration = std::time::Duration::from_secs(15);

/// Heartbeats (1 s each) a received gossip message is kept for peers' IWANT requests. Every message
/// is kept, so a flood costs about this many seconds of its bytes in memory; upstream's 5 made a
/// 36 MB/s flood hold ~180 MB (review 2026-10-29). Flood-publishing makes IWANT a fallback here.
pub const GOSSIP_HISTORY: usize = 2;
use crate::identity::Identity;

/// Configuration for creating a new OpenHydra swarm.
pub struct SwarmOptions {
    /// Addresses to listen on (e.g. "/ip4/0.0.0.0/tcp/4001").
    pub listen_addrs: Vec<Multiaddr>,
    /// Bootstrap peer multiaddrs for Kademlia.
    pub bootstrap_peers: Vec<(PeerId, Multiaddr)>,
    /// Protocol version string for Identify.
    pub protocol_version: String,
    /// WS-F F-4: opt into being a temporary peer-relay (Circuit Relay v2 SERVER)
    /// for NATted peers. Off by default — only publicly-reachable nodes that
    /// explicitly opt in should enable it.
    pub enable_peer_relay: bool,
    /// A private deployment (`--public-relays off`): never connect to OpenHydra's own bootstraps and
    /// relays, however the node hears of them ([`crate::fence`]).
    pub fence_openhydra: bool,
    /// The node is on OpenHydra's public network ([`crate::relay::uses_public_network`]): the routing
    /// table then takes no private address ([`routable_on_public_network`]), so none goes into a DHT
    /// answer. Off (a private deployment, a local test) takes every address.
    pub public_network: bool,
    /// LAN discovery (mDNS); off with `--no-mdns` ([`crate::node::disable_mdns`]), and on a private
    /// network whatever this says ([`build_swarm`]).
    pub enable_mdns: bool,
    /// Which network this node is on, told to peers in identify ([`crate::relay::network_id`]).
    pub network_id: String,
}

impl Default for SwarmOptions {
    fn default() -> Self {
        Self {
            // F-N5: these are compile-time-constant multiaddrs, so `expect` documents
            // the invariant (and points at the offending literal) instead of a bare unwrap.
            listen_addrs: vec![
                "/ip4/0.0.0.0/tcp/4001".parse().expect("const listen multiaddr: ip4 tcp"),
                "/ip4/0.0.0.0/udp/4001/quic-v1".parse().expect("const listen multiaddr: ip4 quic"),
                "/ip6/::/tcp/4001".parse().expect("const listen multiaddr: ip6 tcp"),
                "/ip6/::/udp/4001/quic-v1".parse().expect("const listen multiaddr: ip6 quic"),
            ],
            bootstrap_peers: Vec::new(),
            protocol_version: "openhydra/0.1.0".to_string(),
            enable_peer_relay: false,
            fence_openhydra: false,
            public_network: false,
            enable_mdns: true,
            network_id: "public".into(),
        }
    }
}

/// Whether `addr` may enter the routing table of a node on the public network: not a private,
/// loopback, link-local or CGNAT/Tailscale (100.64/10) address, which a querier elsewhere can't
/// reach, or which names a different machine in the querier's own network. A relay circuit
/// (`/p2p-circuit`) is judged by its relay hop elsewhere, and a DNS name or a non-IP transport
/// (tests) passes.
pub fn routable_on_public_network(addr: &Multiaddr) -> bool {
    use libp2p::multiaddr::Protocol;
    // The first component decides, for a circuit too: its relay hop's address must be one a querier
    // elsewhere can reach (which relay it names is checked where circuits come in: relay.rs).
    match addr.iter().next() {
        Some(Protocol::Ip4(ip)) => !crate::event_loop::is_non_global_ip(ip.into()),
        Some(Protocol::Ip6(ip)) => !crate::event_loop::is_non_global_ip(ip.into()),
        Some(Protocol::Dns(n) | Protocol::Dns4(n) | Protocol::Dns6(n) | Protocol::Dnsaddr(n)) => {
            let n = n.trim_end_matches('.').to_ascii_lowercase();
            n != "localhost" && !n.ends_with(".localhost") && !n.ends_with(".local")
        }
        // A zone-scoped (link-local) IPv6 address only means something on one machine's link.
        Some(Protocol::Ip6zone(_)) => false,
        // A relay circuit with no address in front is dialled through the relay's own addresses.
        Some(Protocol::P2p(_)) => addr.iter().any(|p| matches!(p, Protocol::P2pCircuit)),
        _ => true,
    }
}

/// Build and configure a libp2p Swarm with the OpenHydra behaviour.
pub fn build_swarm(
    identity: &Identity,
    opts: SwarmOptions,
) -> Result<
    (
        Swarm<OpenHydraBehaviour>,
        // WS-F F-4: the peer-relay's leech table (None unless opted in), so the
        // event loop can record byte-cap cap-outs into it.
        Option<Arc<Mutex<LeechTable>>>,
    ),
    Box<dyn std::error::Error>,
> {
    let peer_id = identity.libp2p_peer_id;
    let keypair = identity.keypair.clone();

    // The key, the allowlist and the QUIC switch are read below, once; from here on they can't change
    // (a later change would apply to some of this swarm's parts and not others).
    crate::transport::freeze_network_settings();

    // Transport: TCP + Noise + Yamux
    let tcp_transport = crate::transport::build_tcp_transport(&keypair)?;

    // Transport: QUIC (built-in TLS 1.3 + multiplexing, UDP-based)
    let quic_transport = crate::transport::build_quic_transport(&keypair)?;

    // Relay client — returns a (Transport, Behaviour) pair.
    // The Transport MUST be combined with the base transport and kept alive;
    // dropping it panics the relay client behaviour.
    let (relay_transport, relay_client) = relay::client::new(peer_id);

    // The relay transport outputs `relay::client::Connection` which is an
    // AsyncRead+AsyncWrite stream. Upgrade it with Noise+Yamux to get the
    // same `(PeerId, StreamMuxerBox)` output type as the TCP transport.
    let relay_upgraded = relay_transport
        // On a private network a circuit is keyed end to end too, so a relay can't be a way in.
        .and_then(|conn, _| crate::transport::private_network_stream(conn))
        .map_err(|e| e.either(std::io::Error::other, |e| e))
        .upgrade(libp2p::core::upgrade::Version::V1Lazy)
        .authenticate(
            libp2p::noise::Config::new(&keypair).expect("noise config for relay"),
        )
        .multiplex(libp2p::yamux::Config::default())
        .boxed();

    // 3-way transport composition: relay → QUIC → TCP (fallback order).
    // Relay first: handles /p2p-circuit multiaddrs.
    // QUIC second: handles /udp/.../quic-v1 addresses (faster, UDP hole punch).
    // TCP last: fallback for /tcp/ addresses.
    // Errors unwrapped, so a dial error is the transport's own (`transport::addr_failure`).
    // A private network has no QUIC, which can't carry the pre-shared key, unless a peer allowlist
    // guards it (`transport::allow_quic_on_private_network`).
    let no_quic = !crate::transport::quic_enabled();
    let quic_tcp = if no_quic {
        tcp_transport
    } else {
        crate::transport::either_transport(quic_transport, tcp_transport)
    };

    let relay_upgraded = relay_upgraded.map(|(peer_id, muxer), _| (peer_id, StreamMuxerBox::new(muxer))).boxed();
    let combined_transport = crate::transport::either_transport(relay_upgraded, quic_tcp);

    // Kademlia configuration.
    //
    // R-DHT-10: the discovery layer is libp2p Kad (not KRPC), so robustness is
    // ultimately *its* tuning. These knobs were previously on defaults; each is
    // now set explicitly so the behaviour is intentional and survives upstream
    // default changes.
    let mut kad_config = kad::Config::new(
        libp2p::StreamProtocol::new("/openhydra/kad/1.0.0"),
    );

    // Query timeout. The iterative-lookup ceiling. Left at 10 s: long enough for a
    // multi-hop cross-continent lookup to converge, short enough that a dead-DHT
    // lookup fails fast. The old "10 s tail dominates latency" problem is mitigated
    // upstream by the early-reply path (maybe_reply_discover) + R-DHT-1 PEX seeding,
    // which return the instant providers are in hand rather than awaiting convergence.
    kad_config.set_query_timeout(Duration::from_secs(10));

    // Replication & query breadth.
    // * replication_factor (K) — replicate each record to the K closest nodes so a
    //   handful of dead/lying nodes can't sink it. Explicit K=20 (libp2p K_VALUE);
    //   on a small network it simply caps at the nodes available.
    // * parallelism (α) + disjoint_query_paths — run α=3 *disjoint* lookup paths
    //   (S/Kademlia). An attacker must capture the key's neighbourhood on ALL three
    //   independent paths to eclipse a lookup, not just one — eclipse/sybil
    //   resistance on top of the cryptographic ed25519 node-ids.
    let k = std::num::NonZeroUsize::new(20).expect("20 > 0");
    let alpha = std::num::NonZeroUsize::new(3).expect("3 > 0");
    kad_config.set_replication_factor(k);
    kad_config.set_parallelism(alpha);
    kad_config.disjoint_query_paths(true);

    // Republication coherence — reconcile every lifetime knob against the 300 s TTL
    // so records never silently expire and the two republish mechanisms (Kad's own
    // vs the agent's provider re-announce loop) stay comfortably inside the window:
    //   record TTL / provider TTL        = 300 s   (hard expiry)
    //   record (re)publication interval  = 120 s   (publisher refreshes its records)
    //   provider publication interval    = 120 s   (publisher refreshes provider recs)
    //   record replication interval      =  60 s   (spread copies to K closest between
    //                                               publications — directly addresses
    //                                               "records aren't deliberately spread")
    // All refresh intervals are << 300 s, so a record is re-published ~2× and
    // re-replicated ~5× before it can lapse.
    kad_config.set_record_ttl(Some(Duration::from_secs(300)));
    kad_config.set_provider_record_ttl(Some(Duration::from_secs(300)));
    kad_config.set_publication_interval(Some(Duration::from_secs(120)));
    kad_config.set_provider_publication_interval(Some(Duration::from_secs(120)));
    kad_config.set_replication_interval(Some(Duration::from_secs(60)));

    // Learned-record caching — on a successful lookup, cache the record at the
    // closest nodes that didn't return it (up to max_peers), so a popular model's
    // record stops re-hitting the network on every repeat discover. Default is 1;
    // 3 widens the cache footprint for hot keys at negligible cost.
    kad_config.set_caching(kad::Caching::Enabled { max_peers: 3 });

    // Active maintenance — periodically re-run bootstrap() to refresh buckets and
    // evict dead nodes (pairs with R-DHT-6's persistent routing table). Explicit
    // 5 min interval.
    kad_config.set_periodic_bootstrap_interval(Some(Duration::from_secs(300)));

    // D-S3 (write-side record verification): reject poisoned records on the
    // *inbound-PUT* side, not just at read time. Read-side verification
    // (`dht::verify_peer_record` on every discover path) protects THIS node's
    // routing/credit logic, but with Kad's default (unfiltered) inserts we still
    // *store and re-replicate* whatever peers PUT to us — silently amplifying
    // poison across the network. `FilterBoth` stops the auto-store: every inbound
    // PutRecord / AddProvider now surfaces as a `kad::Event::InboundRequest` that
    // the event loop must explicitly accept (see `handle_inbound_kad_request`),
    // which verifies the signed record before calling `store_mut().put(...)`.
    // A forged record is dropped instead of replicated.
    kad_config.set_record_filtering(kad::StoreInserts::FilterBoth);

    // On the public network, no private address enters the routing table by any path (an mDNS dial
    // to a LAN peer included), so none is handed out in a DHT answer. The filter lives in the
    // vendored libp2p-kad, the one place every address passes (review 2026-10-29).
    if opts.public_network {
        kad_config.set_address_filter(kad::AddressFilter::new(|_peer, addr| routable_on_public_network(addr)));
    }

    let store = kad::store::MemoryStore::new(peer_id);
    let mut kademlia = kad::Behaviour::with_config(peer_id, store, kad_config);

    // Add bootstrap peers to Kademlia routing table.
    for (peer, addr) in &opts.bootstrap_peers {
        kademlia.add_address(peer, addr.clone());
    }

    // R-DHT-2 (revised after the 2026-06-15 live test): take EXPLICIT control of
    // Kademlia's mode instead of relying on its automatic mode. Auto-mode promotes
    // to server on *any* confirmed external address — and libp2p confirms the
    // relay `/p2p-circuit` reservation address as external, so a relay-only node
    // would auto-promote into a server reachable only via a relay (a black hole).
    // We instead start as a client and flip to server ONLY when AutoNAT positively
    // confirms a direct, globally-routable address (or a UPnP mapping does) — see
    // the AutoNAT/UPnP handlers in `event_loop`. This decouples "act as a DHT
    // server" (reachability-gated) from "advertise this address" (still done via
    // add_external_address, including the circuit addr, so peers can reach us for
    // *data* over the relay even while we remain a DHT client).
    kademlia.set_mode(Some(kad::Mode::Client));

    // DCUtR (hole punching).
    let dcutr = dcutr::Behaviour::new(peer_id);

    // AutoNAT (NAT type detection).
    //
    // A3 DCUtR fix: tune the client config so probes fire eagerly enough
    // that the peer reaches a confident Private/Public verdict *before*
    // the first DCUtR hole-punch attempt. Previously ``refresh_interval``
    // Identify (peer metadata exchange).
    let identify = identify::Behaviour::new(
        identify::Config::new(opts.protocol_version.clone(), keypair.public())
            // Which network we're on (`net=…`), so a node of another one on the same LAN isn't
            // joined to ours (review 2026-10-29).
            .with_agent_version(format!("openhydra-agent/{} net={}", env!("CARGO_PKG_VERSION"), opts.network_id))
            .with_push_listen_addr_updates(true)
            // No address cache: identify would offer every address a peer claims (its loopback, its
            // LAN) to our dials. The admissible ones reach Kademlia, which offers those (L10).
            .with_cache_size(0),
    );

    // mDNS (LAN discovery), unless turned off (`--no-mdns`). Always off on a private network: mDNS
    // announces our peer id and addresses to anyone on the LAN, and a private deployment's members
    // are told where each other are (its bootstraps), so the announcement helps only outsiders.
    let keyed = crate::transport::private_network_key().is_some();
    if keyed && opts.enable_mdns {
        tracing::info!("private network: mDNS (LAN discovery) is off, so the LAN doesn't learn this node's peer id and addresses");
    }
    let mdns = if opts.enable_mdns && !keyed {
        Toggle::from(Some(mdns::tokio::Behaviour::new(mdns::Config::default(), peer_id)?))
    } else {
        Toggle::from(None)
    };

    // R-DHT-11: AutoNAT v2 client (per-address reachability verification) — the
    // only AutoNAT now that v1 is retired. Default config (OsRng, default probe
    // interval). Tests the swarm's external-address candidates against v2 servers
    // and reports per-address verdicts that drive R-DHT-2 promotion, `nat_info`,
    // and UPnP-re-assert suppression. No-op until a v2 server (a bootstrap) is
    // reachable.
    let autonat_v2_client = autonat::v2::client::Behaviour::default();

    // gRPC proxy (cross-ISP tunneling through relay).
    let grpc_proxy = crate::proxy::proxy_behaviour();
    // C7: registry query — this node only ever *asks* a bootstrap "who serves model X?", so it
    // advertises Outbound support only (bootstraps run the Inbound responder).
    let registry_query = crate::registry_proto::registry_behaviour(
        libp2p::request_response::ProtocolSupport::Outbound,
    );

    // PR-3 (B1) — Gossipsub over a single topic, signed with our Ed25519
    // identity so recipients can verify the message came from a real swarm
    // member rather than a spoofed peer.
    // Small-swarm reliability tuning (B1 follow-up). With only a handful
    // of peers + 3 bootstraps, the default gossipsub mesh targets
    // (D_lo=4, D_hi=12) aren't reachable, so messages get dropped before
    // they cross the mesh. ``flood_publish(true)`` instructs libp2p to
    // send a published message to *every* known peer of the topic, not
    // just the D-sized mesh slice — trading bandwidth for reliability.
    // At v1 message volumes (control-plane events, not activations) this
    // is cheap.
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_secs(1))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .max_transmit_size(GOSSIP_MAX_TRANSMIT)
        .history_length(GOSSIP_HISTORY)
        .history_gossip(1)
        .duplicate_cache_time(GOSSIP_DUPLICATE_CACHE)
        .flood_publish(true)
        // Ensure small meshes still exist; libp2p defaults assume large
        // swarms. The invariant the validator enforces is
        // ``mesh_outbound_min <= mesh_n_low <= mesh_n <= mesh_n_high``
        // — so we must lower ``mesh_outbound_min`` too (default is 2)
        // otherwise a 2-peer topology fails config validation.
        .mesh_outbound_min(1)
        .mesh_n_low(1)
        .mesh_n(3)
        .mesh_n_high(6)
        .build()
        .map_err(|e| format!("gossipsub config: {e}"))?;
    let mut gossipsub = GossipBehaviour::new_with_subscription_filter(
        gossipsub::MessageAuthenticity::Signed(keypair.clone()),
        gossipsub_config,
        None,
        gossip_topic_filter(),
    )
    .map_err(|e| format!("gossipsub behaviour: {e}"))?;

    // Subscribe immediately so we start participating in the mesh as soon
    // as we have connected peers. Subscription is idempotent; publishing
    // works whether or not we're subscribed to the topic.
    let topic = gossipsub::IdentTopic::new(GOSSIPSUB_TOPIC);
    gossipsub
        .subscribe(&topic)
        .map_err(|e| format!("gossipsub subscribe: {e:?}"))?;

    // Ping keepalive — 15 s interval keeps relay circuit TCP mappings
    // alive through aggressive mobile-hotspot NAT. Without this, the
    // 1-3 s inference silence between tokens causes the hotspot to
    // evict the NAT mapping, killing the relay circuit. Each re-dial
    // costs 2-4 s → the dominant factor in the 0.047 TPS cross-ISP
    // benchmark.
    let ping = ping::Behaviour::new(
        ping::Config::new()
            .with_interval(Duration::from_secs(15)),
    );

    // R-DHT-4: UPnP/NAT-PMP. Default behaviour searches for an IGD gateway on
    // first poll and maps each listen port; on success it confirms the mapped
    // external address with the swarm (→ Kad server promotion via R-DHT-2). A
    // peer with no IGD gateway just gets a GatewayNotFound event and is otherwise
    // unaffected.
    let upnp = libp2p::upnp::tokio::Behaviour::default();

    // WS-F F-4: optional peer-relay SERVER. Off unless opted in. When enabled,
    // it enforces the SAME caps + F-6 leech lockout as the Linode bootstraps via
    // the shared LeechTable (returned so the event loop can record cap-outs).
    let (relay_server, peer_relay_leech): (Toggle<relay::Behaviour>, Option<Arc<Mutex<LeechTable>>>) =
        if opts.enable_peer_relay {
            let leech_table = Arc::new(Mutex::new(LeechTable::new()));
            let server = relay::Behaviour::new(peer_id, {
                let mut cfg = relay::Config {
                    // Smaller caps than a bootstrap — a peer-relay is a helpful
                    // bonus, not core infra; keep its resource exposure modest.
                    max_reservations: 64,
                    max_circuits: 128,
                    max_circuits_per_peer: 4,
                    reservation_duration: Duration::from_secs(3600),
                    max_circuit_bytes: PER_CIRCUIT_BUDGET_BYTES,
                    max_circuit_duration: Duration::from_secs(3600),
                    ..Default::default()
                };
                // The bootstrap's limits, not libp2p's defaults (see `relay::PrefixRateLimiter`).
                cfg.reservation_rate_limiters.clear();
                cfg.circuit_src_rate_limiters.clear();
                cfg.reservation_rate_limiters
                    .push(Box::new(LeechRateLimiter::new(leech_table.clone())));
                cfg.circuit_src_rate_limiters
                    .push(Box::new(LeechRateLimiter::new(leech_table.clone())));
                use crate::relay::PrefixRateLimiter as L;
                // Narrowest first (see the bootstrap's config). No holdings caps here: this relay's
                // own small totals (64 reservations, 128 circuits) and its owner's choice to run it bound it.
                for l in [L::reservations_per_peer(), L::reservations_per_v4(), L::reservations_per_64(), L::reservations_per_48()] {
                    cfg.reservation_rate_limiters.push(Box::new(l));
                }
                for l in [L::circuits_per_peer(), L::circuits_per_v4(), L::circuits_per_64(), L::circuits_per_48()] {
                    cfg.circuit_src_rate_limiters.push(Box::new(l));
                }
                cfg
            });
            (Toggle::from(Some(server)), Some(leech_table))
        } else {
            (Toggle::from(None), None)
        };

    let behaviour = OpenHydraBehaviour {
        ip_limits: crate::ip_limits::Behaviour::new(crate::ip_limits::IpLimitsConfig::PEER),
        fence: crate::fence::Behaviour::new(opts.fence_openhydra.then(crate::fence::Fenced::openhydra)),
        connection_limits: libp2p::connection_limits::Behaviour::new(crate::behaviour::peer_connection_limits()),
        address_book: crate::address_book::Behaviour::new(),
        kademlia,
        relay_client,
        relay_server,
        dcutr,
        autonat_v2_client,
        identify,
        mdns,
        grpc_proxy,
        registry_query,
        gossipsub,
        ping,
        upnp,
    };

    // F-N8 (note): a 300 s idle-connection timeout keeps an otherwise-silent
    // connection open for up to 5 min. That is intentional for long inference
    // sessions (a slow completion can idle the proxy stream between chunks), but
    // it sits in tension with prompt dead-peer reaping — ping-failure eviction and
    // the known_peers reaper are what actually detect a dead peer here, not this
    // timeout. Kept at 300 s deliberately; revisit only if idle-conn count grows.
    let swarm_config = SwarmConfig::with_tokio_executor()
        .with_idle_connection_timeout(Duration::from_secs(300));

    let combined_transport = if opts.fence_openhydra {
        crate::fence::FenceTransport::new(combined_transport, crate::fence::Fenced::openhydra()).boxed()
    } else {
        combined_transport
    };
    let combined_transport = crate::transport::with_setup_timeouts(
        combined_transport,
        crate::transport::OUTBOUND_SETUP_TIMEOUT,
        crate::transport::INBOUND_SETUP_TIMEOUT,
    );
    let mut swarm = Swarm::new(combined_transport, behaviour, peer_id, swarm_config);

    // Listen on configured addresses. F-9 cross-peer half: on a host with no
    // working IPv6, do NOT listen on /ip6/ addresses — otherwise we'd advertise
    // (via identify listen_addrs) v6 addresses we can't actually serve, leading
    // remote peers to waste dial timeouts on our unreachable v6. "Don't
    // advertise what you can't serve" — mirrors the F-9 dial-side gating.
    let ipv6_capable = crate::event_loop::probe_ipv6_capable();
    for addr in &opts.listen_addrs {
        if !ipv6_capable && crate::event_loop::is_ipv6_multiaddr(addr) {
            tracing::info!(%addr, "F-9: skipping IPv6 listen addr (no working outbound IPv6)");
            continue;
        }
        if no_quic && addr.iter().any(|p| matches!(p, libp2p::multiaddr::Protocol::QuicV1 | libp2p::multiaddr::Protocol::Quic)) {
            tracing::info!(%addr, "private network: not listening on QUIC (it can't carry the network key; see --private-network-quic)");
            continue;
        }
        swarm.listen_on(addr.clone())?;
    }

    Ok((swarm, peer_relay_leech))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_our_gossip_topic_can_be_subscribed_to() {
        use libp2p::gossipsub::TopicSubscriptionFilter;
        let mut f = gossip_topic_filter();
        assert!(f.can_subscribe(&gossipsub::IdentTopic::new(GOSSIPSUB_TOPIC).hash()));
        assert!(!f.can_subscribe(&gossipsub::IdentTopic::new("junk-1").hash()));
        assert!(!f.can_subscribe(&gossipsub::IdentTopic::new("openhydra/swarm/v1/other").hash()));
    }
}
