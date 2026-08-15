//! Swarm event loop — runs on a background tokio task.
//!
//! Receives commands from the Python thread via mpsc and drives the
//! libp2p swarm. Results are sent back via oneshot channels.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};

use futures::StreamExt;
use libp2p::kad::store::RecordStore;
use libp2p::request_response;
use libp2p::swarm::SwarmEvent;
use libp2p::{kad, Multiaddr, PeerId};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, trace, warn};

use crate::behaviour::{OpenHydraBehaviour, OpenHydraBehaviourEvent};
use crate::dht;
use crate::proxy::{self, ProxyRequest, ProxyResponse};
use crate::registry_proto::{RegistryQuery, RegistryReply};
use crate::routing_cache;
use crate::types::{
    DiscoveredPeer, KnownProvider, NatInfo, NetCounters, PeerRecord, PeerStatus, StatusSnapshot,
};

// ── Fix 2: Transport classification ────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransportType {
    QuicDirect,
    TcpDirect,
    TcpRelay,
}

fn classify_transport(addr_str: &str) -> TransportType {
    if addr_str.contains("p2p-circuit") {
        return TransportType::TcpRelay;
    }
    // A genuine *direct* connection always carries a transport layer (`/tcp/` or
    // `/udp/`). A bare `/p2p/<peer>` address (no transport, no circuit) is how a
    // *relayed* inbound connection is frequently represented on the LISTENER side
    // — libp2p's `send_back_addr` there is just the source peer id. Classifying
    // that as `TcpDirect` inflated a provider's `direct_conns`, so C-N1 could
    // close a real relay believing a direct path existed. Without a transport
    // component we cannot prove the connection is direct → treat it as relay (the
    // conservative, and in practice correct, verdict). Live-caught 2026-07-12.
    let has_transport = addr_str.contains("/tcp/") || addr_str.contains("/udp/");
    if !has_transport {
        return TransportType::TcpRelay;
    }
    if addr_str.contains("/quic") {
        return TransportType::QuicDirect;
    }
    TransportType::TcpDirect
}

/// F-3: name the connection tier of an established connection for per-tier
/// success metrics. Maps (transport, address-family) to one of the ladder
/// rungs from WS-F's adaptive connection ladder so we can see which rung
/// actually carries traffic (direct-v6 preferred → relay last resort).
fn connection_tier(transport: TransportType, is_ipv6: bool) -> &'static str {
    match transport {
        TransportType::TcpRelay => "relay",
        TransportType::QuicDirect => if is_ipv6 { "direct_quic_v6" } else { "direct_quic_v4" },
        TransportType::TcpDirect => if is_ipv6 { "direct_tcp_v6" } else { "direct_tcp_v4" },
    }
}

/// F-3: all tier names, in ladder-preference order. Used so the metrics dict
/// always reports every tier (0 if never used), making success *rates*
/// computable without guessing which keys exist.
const CONNECTION_TIERS: [&str; 5] = [
    "direct_tcp_v6", "direct_quic_v6", "direct_tcp_v4", "direct_quic_v4", "relay",
];

#[derive(Debug, Default)]
struct PeerConnectionInfo {
    quic_direct_v4: u32,
    quic_direct_v6: u32,
    tcp_direct: u32,
    tcp_relay: u32,
    /// C-N1: ConnectionIds of the *relayed* (circuit) connections to this peer.
    /// `request_response::send_request` dispatches via `NotifyHandler::Any`, so
    /// while a peer has both a direct and a relay connection the chosen path is
    /// arbitrary and throughput silently caps at relay speed. Tracking the relay
    /// ids lets a *stabilized* direct path close them precisely (see the
    /// `relay_close_deadline` grace window) so only the fast path remains.
    relay_conn_ids: Vec<libp2p::swarm::ConnectionId>,
    /// v6-pref (C-N1 follow-on): ConnectionIds of the *v4* QUIC direct connections
    /// to this peer. Same problem as the relay case one rung up — with both a v4
    /// and a v6 QUIC direct, `NotifyHandler::Any` keeps dispatching over the
    /// (churn-prone, NAT-port-translated) v4 path instead of the stabler v6.
    /// Tracking the v4 ids lets a *stabilized* v6 direct retire them (see
    /// `v4_close_deadline`) so traffic settles on v6.
    quic_v4_conn_ids: Vec<libp2p::swarm::ConnectionId>,
}

impl PeerConnectionInfo {
    fn has_direct(&self) -> bool {
        self.quic_direct_v4 > 0 || self.quic_direct_v6 > 0 || self.tcp_direct > 0
    }

    fn direct_count(&self) -> u32 {
        self.quic_direct_v4 + self.quic_direct_v6 + self.tcp_direct
    }
}

/// C-N1: whether a peer with this connection mix should arm the relay-close
/// grace window — it has both a direct path and at least one relay connection,
/// so once the direct path proves stable the relay is redundant overhead.
fn should_arm_relay_close(info: &PeerConnectionInfo) -> bool {
    info.has_direct() && !info.relay_conn_ids.is_empty()
}

/// v6-first upgrade gate (2026-07-12): whether to (still) pursue a v6 QUIC
/// hole-punch to this peer. The goal is a *v6* QUIC direct — its firewall punch
/// has no NAT port translation, so it is far more stable than a v4 CGNAT hole.
/// We are "done" only once we already hold a v6 QUIC direct; a v4 QUIC direct is
/// NOT sufficient. Previously the gate accepted ANY QUIC direct (incl. v4), so a
/// node that reached a peer over v4 first never attempted the v6 upgrade and rode
/// a churning v4 path (live-caught 2026-07-12, netcup↔Mac). The back-off at the
/// call site still bounds wasted dials when the peer's v6 can't be reached.
fn should_pursue_v6_upgrade(info: Option<&PeerConnectionInfo>) -> bool {
    info.map_or(true, |i| i.quic_direct_v6 == 0)
}

/// C-N1: relay ConnectionIds to close for a peer whose grace window elapsed.
/// Empty when the direct path has since dropped — in that case we still depend
/// on the relay and must keep it (belt-and-suspenders to the un-arm on close).
fn relays_to_close(info: &PeerConnectionInfo) -> Vec<libp2p::swarm::ConnectionId> {
    if info.has_direct() {
        info.relay_conn_ids.clone()
    } else {
        Vec::new()
    }
}

/// v6-pref: whether to arm the v4-close grace window — the peer holds BOTH a v6
/// QUIC direct and at least one v4 QUIC direct, so once the v6 proves stable the
/// v4 is redundant and traffic should settle on v6. Mirrors
/// [`should_arm_relay_close`] one rung up (relay < v4-direct < v6-direct).
fn should_arm_v4_close(info: &PeerConnectionInfo) -> bool {
    info.quic_direct_v6 > 0 && !info.quic_v4_conn_ids.is_empty()
}

/// v6-pref: v4 QUIC ConnectionIds to retire for a peer whose grace window
/// elapsed. Empty once the v6 direct has dropped — then we still want the v4
/// (belt-and-suspenders to the un-arm on close), same safety as [`relays_to_close`].
fn v4s_to_close(info: &PeerConnectionInfo) -> Vec<libp2p::swarm::ConnectionId> {
    if info.quic_direct_v6 > 0 {
        info.quic_v4_conn_ids.clone()
    } else {
        Vec::new()
    }
}

type PendingRelayForward = (PeerId, Vec<u8>, oneshot::Sender<Result<Vec<u8>, String>>, std::time::Instant);

/// C-N2: partition queued relay-forwards into kept vs expired. Entries older
/// than `ttl` get an `Err` reply (so a blocking caller unblocks instead of
/// hanging on a relay dial that never connected) and are dropped; fresh entries
/// are returned in order. `now` is injected for testability.
fn sweep_relay_forwards(
    forwards: Vec<PendingRelayForward>,
    ttl: std::time::Duration,
    now: std::time::Instant,
) -> Vec<PendingRelayForward> {
    let mut kept = Vec::with_capacity(forwards.len());
    for (pid, data, reply, enqueued_at) in forwards {
        if now.saturating_duration_since(enqueued_at) < ttl {
            kept.push((pid, data, reply, enqueued_at));
        } else {
            let _ = reply.send(Err("relay dial timed out".into()));
        }
    }
    kept
}

/// Snapshot returned by GetConnectionInfo command (Fix 4).
pub struct ConnectionInfoSnapshot {
    pub has_quic: bool,
    pub has_tcp_direct: bool,
    pub has_relay: bool,
    pub preferred_transport: String,
}

/// Debounce interval for TriggerRepunch (Fix 4).
const REPUNCH_DEBOUNCE: std::time::Duration = std::time::Duration::from_secs(15);

/// C-N1: how long a direct connection must survive before we close the peer's
/// relay connection(s). Direct paths are most likely to flap in the window right
/// after a hole-punch, which is exactly when we still want the warm relay as a
/// fallback — so we only give up the relay once direct has proven stable.
const RELAY_CLOSE_GRACE: std::time::Duration = std::time::Duration::from_secs(25);
/// C-N1: how often the grace-window sweep runs to close now-stable relays.
const RELAY_CLOSE_TICK: std::time::Duration = std::time::Duration::from_secs(5);

/// C-N2: max age of a queued relay-forward before the reaper fails it and drops
/// the reply channel + buffered bytes (bounds the leak when a relay dial never
/// connects). Tied to the reaper tick, so effective sweep is TTL..TTL+reaper.
const RELAY_FORWARD_TTL: std::time::Duration = std::time::Duration::from_secs(30);
/// C-N2: hard cap on queued relay-forwards so a peer we can never dial can't
/// grow the queue without bound.
const MAX_PENDING_RELAY_FORWARDS: usize = 512;

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
    /// C11: drop this model's discover-cache entry so the next `Discover` does a fresh
    /// `get_providers` lookup. The consumer sends this when every candidate from a
    /// (possibly cached) discovery failed, so a stale cache hit can't strand the request.
    InvalidateDiscover {
        model_id: String,
        reply: oneshot::Sender<()>,
    },
    /// C7: ask a connected bootstrap's provider registry "who serves this model?". A best-effort
    /// supplementary discovery source for the cross-NAT case where DHT `get_providers` and gossip
    /// PEX both come up empty. Records are re-verified before use; a `Vec` (possibly empty) is
    /// returned, never an error for "no connected bootstrap" — the caller treats it as a fallback.
    QueryRegistry {
        model_id: String,
        reply: oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>,
    },
    /// Get current NAT status.
    NatStatus {
        reply: oneshot::Sender<NatInfo>,
    },
    /// P0 introspection: one read-only snapshot of the node's live network state
    /// (NAT, Kademlia, peers, reservations, counters) for the agent's status endpoint.
    Status {
        reply: oneshot::Sender<StatusSnapshot>,
    },
    /// List the distinct model ids currently known (PEX-learned / discovered providers in
    /// `known_peers`). Powers the gateway's `GET /v1/models`.
    KnownModels {
        reply: oneshot::Sender<Vec<String>>,
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
    /// F-3: per-tier connection-success metrics. Returns
    /// `(Vec<(tier_name, count)>, dcutr_successes, dcutr_failures)` — one entry
    /// per ladder rung (always all rungs, 0 if unused) plus the DCUtR
    /// hole-punch outcome counters.
    GetTierMetrics {
        reply: oneshot::Sender<(Vec<(String, u64)>, u64, u64)>,
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
    /// Phase 2: Open a TCP-to-libp2p tunnel to a remote peer.
    OpenTunnel {
        peer_id: String,
        reply: oneshot::Sender<Result<String, String>>,
    },
    /// Phase 2: Close an active tunnel to a remote peer.
    CloseTunnel {
        peer_id: String,
    },
    /// Phase 2: Poll for DCUtR success events.
    PollDcutrEvent {
        reply: oneshot::Sender<Option<String>>,
    },
    /// Phase 2: Poll for tunnel close events.
    PollTunnelCloseEvent {
        reply: oneshot::Sender<Option<String>>,
    },
    /// Dial a raw multiaddr (for manual hole-punch testing).
    DialAddress {
        multiaddr: String,
        reply: oneshot::Sender<Result<(), String>>,
    },
    /// Get detailed connection info for a peer (Fix 4).
    GetConnectionInfo {
        peer_id: String,
        reply: oneshot::Sender<Option<ConnectionInfoSnapshot>>,
    },
    /// Trigger a QUIC re-punch for a degraded peer (Fix 4).
    TriggerRepunch {
        peer_id: PeerId,
    },
    /// Graceful shutdown.
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

/// Maximum number of queued inbound proxy requests (audit 2.4).
///
/// Any connected peer can stream framed messages faster than Python drains
/// `poll_proxy_request`. Without a bound the `VecDeque` grows without limit
/// (memory-exhaustion DoS). Drop-oldest on overflow, mirroring the gossip
/// inbound queue (`GOSSIP_INBOUND_QUEUE_MAX`). Sized generously so it only
/// trips under genuine flooding, never normal backpressure.
pub const PROXY_QUEUE_MAX: usize = 4096;

/// Phase 2.4: TTL for inbound proxy response channels. A legitimate proxy
/// request is answered in well under this; anything older is a leaked channel
/// (Python responder crashed/dropped) and is swept by the reaper. Comfortably
/// above any real per-token round-trip so it never trips on a live request.
const PROXY_CHANNEL_TTL: std::time::Duration = std::time::Duration::from_secs(120);

/// Phase 2.4: max concurrent unanswered proxy requests a single peer may hold.
/// Bounds per-peer fairness (PROXY_QUEUE_MAX bounds total). Sized generously so
/// it only trips under genuine per-peer flooding, never normal pipelining.
const MAX_INFLIGHT_PER_PEER: usize = 256;

/// Thread-safe queue for inbound proxy requests, shared between the
/// event loop (producer) and Python poll_proxy_request (consumer).
/// Bypasses the command channel to avoid event-loop round-trip latency.
/// An inbound proxy request handed to the agent: `(request_id, source_peer, data)`.
/// E-S8: `source_peer` is the libp2p-authenticated peer that sent the request (or
/// our own peer id for loopback), so the agent can rate-limit per real identity —
/// a trustworthy key, unlike anything inside the (unverified) request bytes.
pub type InboundProxyItem = (String, String, Vec<u8>);

pub struct SharedProxyQueue {
    queue: Mutex<VecDeque<InboundProxyItem>>,
    condvar: Condvar,
}

impl SharedProxyQueue {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            condvar: Condvar::new(),
        }
    }

    pub fn push(&self, item: InboundProxyItem) {
        let mut q = self.queue.lock().unwrap();
        // Audit 2.4: drop-oldest on overflow so a flooding peer cannot grow
        // this queue without bound.
        if q.len() >= PROXY_QUEUE_MAX {
            q.pop_front();
            warn!("shared_proxy_queue_overflow: dropped oldest inbound request");
        }
        q.push_back(item);
        self.condvar.notify_one();
    }

    pub fn pop(&self, timeout: std::time::Duration) -> Option<InboundProxyItem> {
        let mut q = self.queue.lock().unwrap();
        if let Some(item) = q.pop_front() {
            return Some(item);
        }
        let (mut q, _) = self.condvar.wait_timeout(q, timeout).unwrap();
        q.pop_front()
    }
}

/// State tracked by the event loop.
struct LoopState {
    /// Cached NAT info from AutoNAT probes.
    nat_info: NatInfo,
    /// F-9: whether outbound IPv6 actually works (one-shot startup probe).
    /// When false, we skip dialing IPv6 addresses/relay circuits so a
    /// v6-incapable or partial-v6 host doesn't burn dial timeouts on
    /// unreachable `/ip6/` addresses learned via identify/Kademlia.
    ipv6_capable: bool,
    /// Known peers from Kademlia queries, keyed by OpenHydra peer_id.
    known_peers: HashMap<String, PeerRecord>,
    /// C11: model_id → (unix_ms of our last *completed* discover, net_generation then). A
    /// fresh entry lets `handle_discover` answer from `known_peers` without a fresh
    /// `get_providers` round-trip. Invalidated by `DISCOVER_CACHE_TTL_MS` and by a
    /// net-generation change (roam/wake) — so we never serve a stale relay path after moving.
    discover_cache: HashMap<String, (u64, u64)>,
    /// Pending Kademlia GET queries: query_id → reply channel.
    pending_discovers: HashMap<kad::QueryId, PendingDiscover>,
    /// Chained `get_record` fetches spawned by a `get_providers` discover, mapping the
    /// record query_id → the originating discover's query_id. Lets the GetRecord handler
    /// route a fetched provider record back into the right pending discover.
    pending_record_fetches: HashMap<kad::QueryId, kad::QueryId>,
    /// External addresses discovered by AutoNAT / Identify.
    external_addrs: Vec<Multiaddr>,
    /// R-DHT-4: external addresses currently mapped by UPnP/NAT-PMP (the IGD
    /// behaviour confirms them once, on first map, but does NOT re-emit on
    /// renewal). Tracked here so the re-assert ticker can restore a UPnP address
    /// that the R-DHT-2 AutoNAT-`Private` demotion retracted, once AutoNAT is no
    /// longer asserting we're unreachable. Entries are added on `NewExternalAddr`
    /// and removed on `ExpiredExternalAddr` (the mapping genuinely lapsed).
    upnp_external_addrs: std::collections::HashSet<Multiaddr>,
    /// R-DHT-2/4: whether AutoNAT currently holds a confidence-latched `Private`
    /// verdict. While true, the UPnP re-assert is suppressed — a genuinely broken
    /// port map keeps AutoNAT at `Private`, so it must never be re-promoted into a
    /// black hole; the re-assert only fires once AutoNAT clears (Public/Unknown).
    autonat_private: bool,
    /// P0 introspection: mirrors the last `kademlia.set_mode(..)` call (libp2p exposes
    /// no getter). true = Server (reachable; lives in others' routing tables).
    kad_server_mode: bool,
    /// Relay addresses we've reserved.
    #[allow(dead_code)]
    relay_addrs: Vec<Multiaddr>,
    /// Pending proxy forward requests: request_id → reply channel.
    pending_proxy: HashMap<request_response::OutboundRequestId, oneshot::Sender<Result<Vec<u8>, String>>>,
    /// C7: pending registry queries: outbound request_id → reply channel. The reply is completed
    /// when the bootstrap's `RegistryReply` (or an outbound failure) arrives.
    pending_registry: HashMap<request_response::OutboundRequestId, oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>>,
    /// Local gRPC port for inbound proxy requests.
    local_grpc_port: u16,
    /// Pending inbound proxy responses: request_id → (libp2p ResponseChannel, proxy_respond_tx sender)
    /// When Python calls RespondProxy, we find the channel here and send_response.
    // Phase 2.4: value carries (Instant, source PeerId). The Instant lets the
    // reaper TTL-sweep channels whose Python responder never replied (bounding
    // the leak vs relying only on libp2p's timeout); the PeerId lets the request
    // handler enforce a per-peer inflight cap (fairness — one peer can't fill
    // the global PROXY_QUEUE_MAX slots) by counting this peer's live channels.
    inbound_proxy_channels: HashMap<String, (request_response::ResponseChannel<ProxyResponse>, std::time::Instant, PeerId)>,
    /// Counter for generating unique inbound request IDs.
    inbound_proxy_counter: u64,
    /// Proxy forward requests waiting for a relay connection to be established.
    /// (target_peer_id, data, reply_channel, enqueued_at)
    /// C-N2: the `Instant` lets the reaper TTL-sweep entries whose relay dial
    /// never connects (`OutgoingConnectionError` retries but a permanently
    /// undialable peer would otherwise leak the reply channel + bytes forever).
    pending_relay_forwards:
        Vec<(PeerId, Vec<u8>, oneshot::Sender<Result<Vec<u8>, String>>, std::time::Instant)>,
    /// C-N1: peers with both a direct and a relay connection, mapped to the
    /// instant at which the relay may be closed (direct-established-at + grace).
    /// Cleared if the direct path drops inside the window (so we keep the relay).
    relay_close_deadline: HashMap<PeerId, std::time::Instant>,
    /// v6-pref (C-N1 follow-on): peers with both a v6 and a v4 QUIC direct, mapped
    /// to the instant at which the v4 may be retired (v6-established-at + grace).
    /// Cleared if the v6 direct drops inside the window (so we keep the v4).
    v4_close_deadline: HashMap<PeerId, std::time::Instant>,
    /// DCUtR hole punch counters.
    dcutr_successes: u64,
    dcutr_failures: u64,
    /// Tier-2 connection reversal (off unless `NodeConfig.enable_connection_reversal`).
    enable_connection_reversal: bool,
    /// Per-peer reversal dial back-off (mirrors `quic_holepunch_attempts`).
    reversal_attempts: HashMap<PeerId, u32>,
    /// Peers we've reverse-dialed, awaiting a direct connection (the success signal).
    reversal_pending: std::collections::HashSet<PeerId>,
    /// KPI counters: reversal dials issued / relay→direct upgrades achieved.
    reversal_dials: u64,
    reversal_successes: u64,
    /// F-3: cumulative per-tier connection-success counts, keyed by the
    /// `connection_tier()` rung name. Surfaced via GetTierMetrics so operators
    /// can see which ladder rung (direct-v6 … relay) actually carries
    /// connections — the data that justifies (or refutes) ladder tuning.
    tier_connect_success: std::collections::HashMap<&'static str, u64>,
    /// Fix 2: per-peer transport-type-aware connection tracking.
    peer_connections: HashMap<PeerId, PeerConnectionInfo>,
    /// Reply channels for local proxy forwards (Ouroboros: target == self).
    local_proxy_replies: HashMap<String, oneshot::Sender<Result<Vec<u8>, String>>>,
    /// Fix 4: debounce tracking for TriggerRepunch.
    last_repunch: HashMap<PeerId, std::time::Instant>,
    /// Fix 4: cached QUIC IPv6 addresses per peer (learned from Identify).
    peer_quic_addrs: HashMap<PeerId, Vec<Multiaddr>>,
    /// F7: per-peer auto-QUIC-hole-punch attempt counter. Backs off after
    /// repeated failures so a UDP-hostile path doesn't churn the connection
    /// (the EU-relay case: 5s QUIC timeouts every ~90s). Reset on a successful
    /// QUIC-direct connection.
    quic_holepunch_attempts: HashMap<PeerId, u32>,
    /// Phase 2: DCUtR success event queue (peer_ids as strings).
    dcutr_event_queue: VecDeque<String>,
    /// Phase 2: tunnel close event queue.
    tunnel_close_queue: VecDeque<String>,
    /// PR-3 (B1): inbound gossip messages awaiting Python poll. Each entry
    /// is ``(sender_libp2p_peer_id, payload_bytes)``. Bounded ring — the
    /// Rust side drops the oldest when the queue exceeds
    /// ``GOSSIP_INBOUND_QUEUE_MAX`` to prevent unbounded memory growth
    /// when Python is slow to poll.
    gossip_inbound_queue: std::collections::VecDeque<(String, Vec<u8>)>,
    /// F-5: Per-relay-circuit RESERVATION retry state, keyed by the
    /// ``/p2p-circuit`` listen multiaddr string: (attempt_count,
    /// next_attempt_at). Retries OUR OWN reservation
    /// (``listen_on`` a circuit) so we stay reachable when a relay rejects or
    /// drops the reservation. Without backoff a flapping relay caused either a
    /// tight re-listen loop or (on a hard listen_on error) a permanent strand
    /// for the whole session (the EU-relay gap). Cleared on NewListenAddr /
    /// ReservationReqAccepted for that circuit.
    relay_reservation_retries: HashMap<String, (u32, tokio::time::Instant)>,
    // ── #42: network-change resilience ────────────────────────────────
    /// User-supplied bootstrap peers, retained so `rebootstrap()` can re-dial
    /// them on a network change (the startup list is otherwise consumed by the
    /// run-loop locals). Identity is stable across a roam; only addresses churn.
    bootstrap_peers: Vec<(PeerId, Multiaddr)>,
    /// Monotonically bumped every time connectivity is rebuilt (`rebootstrap()`).
    /// Read by the provider run-loop (via `NetworkHandle::network_generation`)
    /// to trigger an immediate model re-announce so the DHT record carries the
    /// new relay addresses under the *same* pinned PeerId. Shared with the handle.
    net_generation: Arc<std::sync::atomic::AtomicU64>,
    /// Circuit listen-addrs we currently hold a live reservation for (inserted
    /// on NewListenAddr, removed on Expired/Closed). Lets `rebootstrap()`
    /// re-request only the *missing* reservations instead of blindly
    /// `listen_on`-ing every relay again — which would leak duplicate circuit
    /// listeners (the F6 hazard).
    reserved_circuits: std::collections::HashSet<String>,
    /// Last time `rebootstrap()` ran — cooldown so a burst of triggers (a roam
    /// emits many NewListenAddr) collapses into one rebuild.
    last_rebootstrap: Option<tokio::time::Instant>,
    /// Debounced reactive-rebootstrap deadline. A real (non-loopback,
    /// non-circuit) listener change sets this to now+debounce; the heal ticker
    /// fires the rebuild once it's due, so a burst settles into one heal.
    pending_rebootstrap_at: Option<tokio::time::Instant>,
    /// When the connectivity watchdog first observed 0 connected peers (None =
    /// healthy). Sustained degradation past the grace window triggers a
    /// rebootstrap — the wake-from-sleep / roam catch-all.
    degraded_since: Option<tokio::time::Instant>,
    /// How many times connectivity was rebuilt (metric / log).
    rebootstrap_count: u64,
    /// #42 follow-up (zombie connections): consecutive proxy outbound *path*
    /// failures (timeout / closed / dial / io) per peer. During the live roam
    /// test the consumer kept dispatching onto dead pre-roam connections for
    /// 6+ minutes — `send_request` picks ANY existing connection, and zombie
    /// TCP/circuit conns linger long after the peer left that network. When a
    /// peer's streak reaches `ZOMBIE_FAILURE_THRESHOLD`, its connections are
    /// force-closed and a fresh relay redial is issued so the next dispatch
    /// rides a live path. Reset on any successful response.
    proxy_failure_streak: HashMap<PeerId, u32>,
    /// #43-W1: count of auto QUIC-v6 hole-punch dials issued (pairs with the
    /// `direct_quic_v6` tier-success metric to show punches that actually land).
    quic_v6_holepunch_dials: u64,
    /// WS-F F-4: peer-relay leech table (None unless peer-relay mode is on).
    /// The RelayServer event handler records byte-cap cap-outs here.
    peer_relay_leech: Option<std::sync::Arc<std::sync::Mutex<crate::relay::LeechTable>>>,
}

/// PR-3: upper bound on pending inbound gossip messages.
/// The swarm-wide event rate is tiny (one ``PEER_DEAD`` per real failure,
/// plus the occasional ``REQUEST_HOLE_PUNCH``) so a soft cap of 256 is
/// roughly an hour of breathing room before oldest-drop kicks in.
const GOSSIP_INBOUND_QUEUE_MAX: usize = 256;

/// R-DHT-1 (gossip provider PEX): the gossipsub envelope `type` a provider uses
/// to advertise its signed `PeerRecord` swarm-wide. Consumers fast-path this in
/// Rust (verify + author-check + cache into `known_peers`) so discovery keeps
/// working even when the Kademlia DHT is fully degraded — BitTorrent's PEX
/// property. Travels on the same single topic as the other control events.
pub const PROVIDER_ANNOUNCE_TYPE: &str = "PROVIDER_ANNOUNCE";

/// R-DHT-1 audit fix: how long a `known_peers` entry we are NOT connected to may
/// survive in the reaper. A PEX-learned provider is, by definition, one we have
/// no live connection to (we heard it over gossip via a relay) — reaping it
/// purely on connection status defeated gossip discovery. We instead keep it
/// while it's *fresh* (its signed `updated_unix_ms` is within this window),
/// matching the 300 s DHT record TTL; a provider that stops (re)announcing then
/// ages out, while a live one is kept refreshed by each gossip.
const KNOWN_PEER_TTL_MS: u64 = 300_000;

/// C11: how long a *completed* `discover` for a model stays fresh enough to answer from the
/// `known_peers` cache without a new `get_providers` round-trip. Coalesces a request burst (a
/// chat session / an agent loop) so only ~1 request per window pays the DHT lookup — which
/// cross-NAT is the ~1.2 s `discover` hop (gossip PEX can't populate the cache across separate
/// networks). Kept low so a moved/departed provider is re-resolved promptly; failover + the
/// reaper cover a stale hit, and a net-generation change invalidates it outright.
const DISCOVER_CACHE_TTL_MS: u64 = 30_000;

/// Current wall-clock time in Unix milliseconds (0 if the clock is before the
/// epoch, which never happens in practice).
fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

struct PendingDiscover {
    model_id: String,
    records: Vec<PeerRecord>,
    reply: oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>,
    /// Set once the `get_providers` query terminates (found-all or timed-out). Until then,
    /// more providers — and thus more chained record fetches — may still arrive.
    providers_done: bool,
    /// In-flight `get_record` fetches for providers whose full `PeerRecord` wasn't held
    /// locally (cross-relay, the record lives on the bootstrap nodes, not on us). The
    /// reply is sent only once `providers_done && outstanding == 0`.
    outstanding: usize,
}

impl LoopState {
    fn new() -> Self {
        Self {
            nat_info: NatInfo {
                nat_type: "unknown".into(),
                external_ip: String::new(),
                external_ipv4: String::new(),
                external_ipv6: String::new(),
                external_port: 0,
                is_public: false,
            },
            ipv6_capable: probe_ipv6_capable(),
            known_peers: HashMap::new(),
            discover_cache: HashMap::new(),
            pending_discovers: HashMap::new(),
            pending_record_fetches: HashMap::new(),
            external_addrs: Vec::new(),
            upnp_external_addrs: std::collections::HashSet::new(),
            autonat_private: false,
            kad_server_mode: false, // nodes start as Kad clients (R-DHT-2)
            relay_addrs: Vec::new(),
            pending_proxy: HashMap::new(),
            pending_registry: HashMap::new(),
            local_grpc_port: 50051,
            inbound_proxy_channels: HashMap::new(),
            inbound_proxy_counter: 0,
            pending_relay_forwards: Vec::new(),
            relay_close_deadline: HashMap::new(),
            v4_close_deadline: HashMap::new(),
            dcutr_successes: 0,
            dcutr_failures: 0,
            enable_connection_reversal: false,
            reversal_attempts: HashMap::new(),
            reversal_pending: std::collections::HashSet::new(),
            reversal_dials: 0,
            reversal_successes: 0,
            tier_connect_success: std::collections::HashMap::new(),
            peer_connections: HashMap::new(),
            local_proxy_replies: HashMap::new(),
            last_repunch: HashMap::new(),
            peer_quic_addrs: HashMap::new(),
            quic_holepunch_attempts: HashMap::new(),
            dcutr_event_queue: VecDeque::new(),
            tunnel_close_queue: VecDeque::new(),
            gossip_inbound_queue: std::collections::VecDeque::new(),
            relay_reservation_retries: HashMap::new(),
            // #42: set to the shared handle in run_event_loop; a real interface
            // change or watchdog rebuild bumps it.
            bootstrap_peers: Vec::new(),
            net_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            reserved_circuits: std::collections::HashSet::new(),
            last_rebootstrap: None,
            pending_rebootstrap_at: None,
            degraded_since: None,
            rebootstrap_count: 0,
            proxy_failure_streak: HashMap::new(),
            quic_v6_holepunch_dials: 0,
            peer_relay_leech: None,
        }
    }
}

/// Run the swarm event loop until shutdown.
pub async fn run_event_loop(
    mut swarm: libp2p::Swarm<OpenHydraBehaviour>,
    mut cmd_rx: mpsc::Receiver<SwarmCommand>,
    proxy_queue: Arc<SharedProxyQueue>,
    bootstrap_peers: Vec<(PeerId, Multiaddr)>,
    keypair: libp2p::identity::Keypair,
    // WS-F F-4: shared leech table for the peer-relay server (None unless this
    // node opted into peer-relay mode). The RelayServer event handler records
    // byte-cap cap-outs into it; the relay::Config's LeechRateLimiter reads it.
    peer_relay_leech: Option<std::sync::Arc<std::sync::Mutex<crate::relay::LeechTable>>>,
    // R-DHT-6: where to persist/reload the Kademlia routing table. `None`
    // disables persistence (e.g. tests). Set by `start_node` to a file beside
    // the identity key.
    routing_cache_path: Option<std::path::PathBuf>,
    // Tier-2 connection reversal flag (NodeConfig.enable_connection_reversal).
    enable_connection_reversal: bool,
    // #42: shared network-generation counter. Bumped on every `rebootstrap()`
    // so the provider run-loop can re-announce its DHT record (fresh relay
    // addresses, same pinned PeerId) the moment connectivity is rebuilt.
    net_generation: Arc<std::sync::atomic::AtomicU64>,
    // #43-W2: opt-in PCP v6 firewall-pinhole wiring (None = disabled).
    pcp_bind: Option<crate::pcp::PcpBind>,
) {
    let mut state = LoopState::new();
    state.enable_connection_reversal = enable_connection_reversal;
    state.peer_relay_leech = peer_relay_leech; // WS-F F-4
    // #42: retain the bootstrap list + share the generation counter so
    // `rebootstrap()` can re-dial and signal the provider on a network change.
    state.bootstrap_peers = bootstrap_peers.clone();
    state.net_generation = net_generation;
    // Cover the startup window with the rebootstrap cooldown: the inline startup
    // sequence below IS a bootstrap, so a reactive trigger from the initial
    // NewListenAddr burst must not immediately re-run it.
    state.last_rebootstrap = Some(tokio::time::Instant::now());

    // #43-W2: opt-in PCP v6 firewall-pinhole maintainer. When an operator
    // supplied the CPE gateway, spawn a task that periodically opens our listen
    // ports inbound on the global v6 and reports each confirmed external addr
    // over `pcp_candidate_rx`; the select arm below adds them as external
    // addresses so AutoNAT probes them (→ promotion), mirroring the UPnP path.
    let (pcp_candidate_tx, mut pcp_candidate_rx) = mpsc::unbounded_channel::<Multiaddr>();
    if let Some(bind) = pcp_bind {
        if bind.ports.is_empty() {
            warn!(gateway = %bind.gateway, "pcp: no concrete listen ports to pinhole; PCP disabled");
        } else {
            info!(gateway = %bind.gateway, ports = ?bind.ports,
                "pcp: starting v6 firewall-pinhole maintainer (#43-W2)");
            tokio::spawn(crate::pcp::run_maintainer(bind.gateway, bind.ports, pcp_candidate_tx));
        }
    }

    // Kick off Kademlia bootstrap (populate routing table from bootstrap peers).
    if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
        warn!("kademlia bootstrap failed (no peers yet?): {e}");
    }

    // Explicitly dial every bootstrap peer to force a direct connection.
    // kademlia.add_address() only populates the routing table — it does
    // NOT guarantee a connection.  Without an explicit dial the swarm
    // may never establish a direct QUIC link and will fall back to relay.
    for (peer_id, addr) in &bootstrap_peers {
        let dial_addr = addr.clone().with(libp2p::multiaddr::Protocol::P2p(*peer_id));
        info!(%peer_id, %dial_addr, "bootstrap_dial: explicitly dialing bootstrap peer");
        if let Err(e) = swarm.dial(dial_addr.clone()) {
            warn!(%peer_id, %dial_addr, %e, "bootstrap_dial: failed");
        }
    }

    // R-DHT-6: warm-start the routing table from the persisted cache so a fresh
    // process rejoins with known-good contacts instead of leaning entirely on the
    // bootstrap relays. Best-effort: a missing/corrupt cache just yields nothing.
    if let Some(ref cache_path) = routing_cache_path {
        let warm = routing_cache::load(cache_path);
        let n = warm.len();
        for (peer_id, addr) in warm {
            swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
        }
        if n > 0 {
            info!(contacts = n, ?cache_path, "R-DHT-6: warm-started routing table from cache");
        }
    }

    // Relay reservations are requested after a short delay (see below)
    // to ensure Kademlia has connected to the bootstrap peers first.
    // The relay client behaviour sends the reservation request on an
    // existing connection — without one, it dials via TCP which doesn't
    // install the relay reservation handler.
    let mut relay_reservation_pending = true;
    let relay_reservation_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);

    // Bootstrap peer retry: periodically re-dial user-supplied bootstrap
    // peers (non-relay) that haven't established a direct connection.
    // This enables simultaneous-open hole-punching: both peers keep
    // dialing each other until both outbound packets cross in-flight
    // and punch through both NATs.  Only user-supplied peers (those
    // whose address is NOT a known relay IP) are retried.
    let non_relay_bootstrap: Vec<(PeerId, Multiaddr)> = bootstrap_peers
        .iter()
        .filter(|(_, addr)| {
            let addr_str = addr.to_string();
            let ip = extract_ip_from_multiaddr_str(&addr_str);
            let is_relay = ip
                .as_ref()
                .map(|ip| crate::relay::is_bootstrap_relay_ip(ip))
                .unwrap_or(false);
            !is_relay
        })
        .cloned()
        .collect();
    let mut bootstrap_retry_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
    // Phase 7.2: Cap bootstrap retries at 20 to prevent infinite loops.
    let mut bootstrap_retry_count: u32 = 0;
    const MAX_BOOTSTRAP_RETRIES: u32 = 20;

    // Phase 1.4: Periodic known_peers reaper — safety net that catches
    // ghosts missed by individual eviction paths (1.1, 1.2, 1.3, 1.6, 1.7).
    let mut reaper_interval = tokio::time::interval(std::time::Duration::from_secs(60));
    reaper_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // C-N1: grace-window sweep — closes a peer's relay connection(s) once its
    // direct path has survived RELAY_CLOSE_GRACE, so request_response can no
    // longer arbitrarily pin traffic to the (slower) relay.
    let mut relay_close_ticker = tokio::time::interval(RELAY_CLOSE_TICK);
    relay_close_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // R-DHT-6: periodically snapshot the routing table to disk (BitTorrent-style)
    // so the next start warm-rejoins. 5 min cadence — frequent enough to capture
    // a useful contact set, infrequent enough to be negligible I/O. First tick
    // fires immediately; we skip persisting an empty table on that one.
    let mut routing_save_ticker = tokio::time::interval(std::time::Duration::from_secs(300));
    routing_save_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // R-DHT-4: periodically re-assert UPnP-mapped external addresses. The IGD
    // behaviour confirms a mapping only once (not on renewal), so if an
    // AutoNAT-`Private` demotion retracted the address, nothing would restore it
    // until a full remap. This ticker re-adds any tracked UPnP address that has
    // fallen out of the confirmed set — but ONLY while AutoNAT is not asserting
    // `Private` (a genuinely-broken map keeps AutoNAT at `Private`, so it is never
    // re-promoted into a black hole). 120 s ≫ AutoNAT's confidence-latch time, so
    // AutoNAT always wins the race against a truly-unreachable map.
    let mut upnp_reassert_ticker = tokio::time::interval(std::time::Duration::from_secs(120));
    upnp_reassert_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // F6: B5's periodic re-listen was removed. It called `listen_on(circuit)`
    // every 30 min unconditionally, creating a NEW listener each time without
    // closing the old one (~144 leaked listeners/day → eventual per-peer
    // reservation-cap denial). rust-libp2p's relay::client already auto-renews
    // reservations for active listeners, and the reactive ExpiredListenAddr /
    // ListenerClosed handlers below re-listen if a reservation actually drops —
    // so the periodic re-listen was redundant and leaky.

    // F-5: relay-RESERVATION retry driver — fires often enough to honour the
    // reservation-retry backoff. Each circuit's retry is gated on its own
    // next_attempt_at, so this ticker is cheap (it only acts on due entries).
    let mut relay_retry_ticker = tokio::time::interval(std::time::Duration::from_millis(250));
    relay_retry_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // #42: network-change heal ticker — drives the connectivity watchdog
    // (rebuild after sustained loss of all peers, i.e. wake/roam) and fires any
    // debounced reactive rebootstrap. 5 s is snappy enough to recover a roam
    // within the debounce+grace budget while staying negligible when healthy.
    let mut heal_ticker = tokio::time::interval(std::time::Duration::from_secs(5));
    heal_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        // Retry dialing non-relay bootstrap peers every 15s until connected or retry cap hit.
        if !non_relay_bootstrap.is_empty()
            && bootstrap_retry_count < MAX_BOOTSTRAP_RETRIES
            && tokio::time::Instant::now() >= bootstrap_retry_deadline
        {
            bootstrap_retry_count += 1;
            bootstrap_retry_deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
            let mut all_connected = true;
            for (peer_id, addr) in &non_relay_bootstrap {
                let already_direct = state.peer_connections.get(peer_id)
                    .map_or(false, |info| info.has_direct());
                if already_direct {
                    continue;
                }
                all_connected = false;
                let dial_addr = addr.clone().with(libp2p::multiaddr::Protocol::P2p(*peer_id));
                info!(
                    %peer_id, %dial_addr,
                    attempt = bootstrap_retry_count,
                    max = MAX_BOOTSTRAP_RETRIES,
                    "bootstrap_retry: re-dialing (not yet direct)"
                );
                if let Err(e) = swarm.dial(dial_addr.clone()) {
                    warn!(%peer_id, %e, "bootstrap_retry: dial failed");
                }
            }
            // Reset counter when all bootstrap peers are connected.
            if all_connected {
                bootstrap_retry_count = 0;
            }
            if bootstrap_retry_count >= MAX_BOOTSTRAP_RETRIES {
                warn!(
                    "bootstrap_retries_exhausted: gave up after {} attempts, some bootstrap peers unreachable",
                    MAX_BOOTSTRAP_RETRIES
                );
            }
        }
        // Delayed relay reservation: wait for Kademlia to connect to
        // bootstrap peers, then request relay reservations via listen_on.
        if relay_reservation_pending && tokio::time::Instant::now() >= relay_reservation_deadline {
            relay_reservation_pending = false;
            // Dedup reservation targets by relay PeerId. BOOTSTRAP_RELAYS lists
            // each relay twice (its v4 AND v6 address). Reserving on the SAME
            // relay peer via both stacks makes libp2p hold ONE reservation per
            // relay peer and EOF the duplicate listener (`listener_closed …
            // Reservation(Io(UnexpectedEof))`, addresses=[]). The F-5 handler
            // then retries that doomed duplicate forever, and each retry
            // disrupts the working reservation to the same relay → permanent
            // reservation churn → the NATted peer never holds a stable
            // reservation and is unreachable via relay. Reserve ONCE per relay
            // (first address wins; v4 is listed first and is universally
            // reachable — and the only option for v6-less peers like cloud T4s).
            let mut seen_relays = std::collections::HashSet::new();
            for relay_str in crate::relay::BOOTSTRAP_RELAYS {
                if let Ok(relay_multiaddr) = relay_str.parse::<Multiaddr>() {
                    let relay_peer = relay_multiaddr.iter().find_map(|p| match p {
                        libp2p::multiaddr::Protocol::P2p(id) => Some(id),
                        _ => None,
                    });
                    if let Some(pid) = relay_peer {
                        if !seen_relays.insert(pid) {
                            continue; // already reserving on this relay peer
                        }
                    }
                    let listen_addr = relay_multiaddr
                        .with(libp2p::multiaddr::Protocol::P2pCircuit);
                    match swarm.listen_on(listen_addr.clone()) {
                        Ok(_) => {
                            info!(addr = %listen_addr, "listening via relay (reservation requested)");
                        }
                        Err(e) => {
                            warn!(addr = %listen_addr, error = %e, "relay listen failed");
                            // F-5: don't strand a relay whose initial reservation
                            // listen_on errored — schedule a backed-off retry.
                            schedule_reservation_retry(&mut state, &listen_addr.to_string());
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
                        handle_announce(&mut swarm, &record, reply, &mut state, &keypair);
                    }
                    Some(SwarmCommand::Discover { model_id, reply }) => {
                        handle_discover(&mut swarm, &model_id, reply, &mut state);
                    }
                    Some(SwarmCommand::InvalidateDiscover { model_id, reply }) => {
                        // C11 bypass: forget this model's cached discovery so the next Discover
                        // re-queries get_providers instead of early-replying from a stale seed.
                        state.discover_cache.remove(&model_id);
                        let _ = reply.send(());
                    }
                    Some(SwarmCommand::QueryRegistry { model_id, reply }) => {
                        handle_query_registry(&mut swarm, &model_id, reply, &mut state);
                    }
                    Some(SwarmCommand::NatStatus { reply }) => {
                        let _ = reply.send(state.nat_info.clone());
                    }
                    Some(SwarmCommand::Status { reply }) => {
                        // P0 introspection: assemble the read-only snapshot from state the
                        // loop already tracks. No hot-path cost — runs only when asked.
                        let peers = state
                            .peer_connections
                            .iter()
                            .map(|(pid, info)| {
                                let direct = info.direct_count() > 0;
                                let relayed = info.tcp_relay > 0;
                                PeerStatus {
                                    peer_id: pid.to_string(),
                                    quic_direct_v4: info.quic_direct_v4,
                                    quic_direct_v6: info.quic_direct_v6,
                                    tcp_direct: info.tcp_direct,
                                    tcp_relay: info.tcp_relay,
                                    failure_streak: state
                                        .proxy_failure_streak
                                        .get(pid)
                                        .copied()
                                        .unwrap_or(0),
                                    path: match (direct, relayed) {
                                        (true, true) => "mixed",
                                        (true, false) => "direct",
                                        (false, true) => "relay",
                                        (false, false) => "none",
                                    }
                                    .to_string(),
                                }
                            })
                            .collect();
                        let mut known_models: Vec<String> = state
                            .known_peers
                            .values()
                            .map(|r| r.model_id.clone())
                            .filter(|m| !m.is_empty())
                            .collect();
                        known_models.sort();
                        known_models.dedup();
                        let known_providers = state
                            .known_peers
                            .values()
                            .map(|r| KnownProvider {
                                model_id: r.model_id.clone(),
                                openhydra_peer_id: r.peer_id.clone(),
                                libp2p_peer_id: r.libp2p_peer_id.clone(),
                            })
                            .collect();
                        let kad_routing_peers = swarm
                            .behaviour_mut()
                            .kademlia
                            .kbuckets()
                            .map(|b| b.num_entries())
                            .sum();
                        let snapshot = StatusSnapshot {
                            nat: state.nat_info.clone(),
                            autonat_private: state.autonat_private,
                            ipv6_capable: state.ipv6_capable,
                            kad_server_mode: state.kad_server_mode,
                            kad_routing_peers,
                            network_generation: state
                                .net_generation
                                .load(std::sync::atomic::Ordering::Relaxed),
                            listen_addrs: swarm.listeners().map(|a| a.to_string()).collect(),
                            external_addrs: state
                                .external_addrs
                                .iter()
                                .map(|a| a.to_string())
                                .collect(),
                            relay_reservations: state
                                .reserved_circuits
                                .iter()
                                .cloned()
                                .collect(),
                            peers,
                            known_models,
                            known_providers,
                            counters: NetCounters {
                                dcutr_successes: state.dcutr_successes,
                                dcutr_failures: state.dcutr_failures,
                                reversal_dials: state.reversal_dials,
                                reversal_successes: state.reversal_successes,
                                tier_connect_success: state
                                    .tier_connect_success
                                    .iter()
                                    .map(|(k, v)| (k.to_string(), *v))
                                    .collect(),
                            },
                        };
                        let _ = reply.send(snapshot);
                    }
                    Some(SwarmCommand::KnownModels { reply }) => {
                        let mut models: Vec<String> = state
                            .known_peers
                            .values()
                            .map(|r| r.model_id.clone())
                            .filter(|m| !m.is_empty())
                            .collect();
                        models.sort();
                        models.dedup();
                        let _ = reply.send(models);
                    }
                    Some(SwarmCommand::ResolveAddress { peer_id, reply }) => {
                        handle_resolve(&state, &peer_id, reply);
                    }
                    Some(SwarmCommand::ProxyForward { peer_id, data, reply }) => {
                        handle_proxy_forward(&mut swarm, &peer_id, data, reply, &mut state, &proxy_queue);
                    }
                    Some(SwarmCommand::ProxyForwardNoWait { peer_id, data }) => {
                        handle_proxy_forward_no_wait(&mut swarm, &peer_id, data, &mut state, &proxy_queue);
                    }
                    Some(SwarmCommand::IsConnected { peer_id, reply }) => {
                        let has_direct = match peer_id.parse::<PeerId>() {
                            Ok(pid) => state.peer_connections.get(&pid)
                                .map_or(false, |info| info.has_direct()),
                            Err(_) => false,
                        };
                        let _ = reply.send(has_direct);
                    }
                    Some(SwarmCommand::GetDcutrStats { reply }) => {
                        let direct_count = state.peer_connections.values()
                            .filter(|info| info.has_direct())
                            .count() as u64;
                        let snapshot = (
                            state.dcutr_successes,
                            state.dcutr_failures,
                            direct_count,
                        );
                        let _ = reply.send(snapshot);
                    }
                    Some(SwarmCommand::GetTierMetrics { reply }) => {
                        // Report every ladder rung (0 if unused) so success
                        // rates are computable Python-side without guessing keys.
                        let tiers: Vec<(String, u64)> = CONNECTION_TIERS
                            .iter()
                            .map(|name| (
                                (*name).to_string(),
                                state.tier_connect_success.get(name).copied().unwrap_or(0),
                            ))
                            .collect();
                        let _ = reply.send((
                            tiers,
                            state.dcutr_successes,
                            state.dcutr_failures,
                        ));
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
                    Some(SwarmCommand::RespondProxy { request_id, data }) => {
                        if let Some(reply) = state.local_proxy_replies.remove(&request_id) {
                            let _ = reply.send(Ok(data));
                        } else if let Some((channel, _, _)) = state.inbound_proxy_channels.remove(&request_id) {
                            if let Err(e) = swarm.behaviour_mut().grpc_proxy.send_response(channel, ProxyResponse(data)) {
                                warn!("proxy respond failed: {:?}", e);
                            }
                        } else {
                            warn!("proxy respond: unknown request_id={}", request_id);
                        }
                    }
                    Some(SwarmCommand::OpenTunnel { peer_id, reply }) => {
                        handle_open_tunnel(&peer_id, reply, &mut state);
                    }
                    Some(SwarmCommand::CloseTunnel { peer_id }) => {
                        handle_close_tunnel(&peer_id, &mut state);
                    }
                    Some(SwarmCommand::PollDcutrEvent { reply }) => {
                        let item = state.dcutr_event_queue.pop_front();
                        let _ = reply.send(item);
                    }
                    Some(SwarmCommand::PollTunnelCloseEvent { reply }) => {
                        let item = state.tunnel_close_queue.pop_front();
                        let _ = reply.send(item);
                    }
                    Some(SwarmCommand::DialAddress { multiaddr, reply }) => {
                        use libp2p::swarm::dial_opts::DialOpts;
                        let res = match multiaddr.parse::<Multiaddr>() {
                            Ok(ma) => {
                                let opts = DialOpts::unknown_peer_id()
                                    .address(ma.clone())
                                    .build();
                                match swarm.dial(opts) {
                                    Ok(()) => {
                                        info!(%ma, "manual_dial_address_issued");
                                        Ok(())
                                    }
                                    Err(e) => Err(format!("dial error: {e}")),
                                }
                            }
                            Err(e) => Err(format!("invalid multiaddr: {e}")),
                        };
                        let _ = reply.send(res);
                    }
                    Some(SwarmCommand::GetConnectionInfo { peer_id, reply }) => {
                        let snapshot = match peer_id.parse::<PeerId>() {
                            Ok(pid) => {
                                state.peer_connections.get(&pid).map(|info| {
                                    let has_quic = info.quic_direct_v4 > 0 || info.quic_direct_v6 > 0;
                                    let preferred = if has_quic {
                                        "quic_direct".to_string()
                                    } else if info.tcp_direct > 0 {
                                        "tcp_direct".to_string()
                                    } else if info.tcp_relay > 0 {
                                        "tcp_relay".to_string()
                                    } else {
                                        "none".to_string()
                                    };
                                    ConnectionInfoSnapshot {
                                        has_quic,
                                        has_tcp_direct: info.tcp_direct > 0,
                                        has_relay: info.tcp_relay > 0,
                                        preferred_transport: preferred,
                                    }
                                })
                            }
                            Err(_) => None,
                        };
                        let _ = reply.send(snapshot);
                    }
                    Some(SwarmCommand::TriggerRepunch { peer_id }) => {
                        handle_trigger_repunch(&mut swarm, peer_id, &mut state);
                    }
                    // Phase 4.1: Graceful shutdown — publish PEER_DEPARTED
                    // gossip and remove self from Kademlia before exiting.
                    Some(SwarmCommand::Shutdown { reply }) => {
                        info!("swarm shutting down — cleaning up DHT records");

                        // 1. Publish self-departure gossip so other peers
                        // can immediately evict us (Phase 4.2 receiver side).
                        let departure_payload = format!(
                            r#"{{"type":"PEER_DEPARTED","libp2p_peer_id":"{}","timestamp":{}}}"#,
                            swarm.local_peer_id(),
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                        );
                        let topic = libp2p::gossipsub::IdentTopic::new(
                            crate::swarm::GOSSIPSUB_TOPIC,
                        );
                        match swarm
                            .behaviour_mut()
                            .gossipsub
                            .publish(topic, departure_payload.into_bytes())
                        {
                            Ok(_) => info!("published PEER_DEPARTED gossip"),
                            Err(e) => warn!(%e, "failed to publish PEER_DEPARTED"),
                        }

                        // 2. Remove self from Kademlia routing table.
                        let local_peer_id = *swarm.local_peer_id();
                        swarm.behaviour_mut().kademlia.remove_peer(&local_peer_id);

                        let _ = reply.send(());
                        return;
                    }
                    None => {
                        info!("command channel closed, shutting down swarm");
                        return;
                    }
                }
            }
            // Phase 1.4: Periodic known_peers reaper — removes entries whose
            // libp2p_peer_id is no longer connected. Safety net for ghosts
            // that slip through individual eviction paths.
            _ = reaper_interval.tick() => {
                let before = state.known_peers.len();
                let now_ms = now_unix_ms();
                state.known_peers.retain(|_openhydra_id, record| {
                    if record.libp2p_peer_id.is_empty() {
                        return true; // keep records without libp2p binding
                    }
                    match record.libp2p_peer_id.parse::<PeerId>() {
                        // Keep if we have a live connection (the original ghost-peer
                        // guard) OR if the record is still fresh. The freshness arm
                        // is the R-DHT-1 audit fix: a PEX-learned provider we never
                        // connected to must survive on its TTL, or this reaper would
                        // evict it ~60 s after we learned it and break gossip
                        // discovery for any re-announce interval > 60 s.
                        Ok(pid) => {
                            swarm.is_connected(&pid)
                                || (record.updated_unix_ms != 0
                                    && now_ms.saturating_sub(record.updated_unix_ms)
                                        < KNOWN_PEER_TTL_MS)
                        }
                        Err(_) => false, // unparseable peer_id = stale
                    }
                });
                let removed = before - state.known_peers.len();
                if removed > 0 {
                    info!(removed, remaining = state.known_peers.len(), "known_peers reaper sweep");
                }
                // C11: prune the discover cache on the same tick so it stays bounded — drops
                // entries past DISCOVER_CACHE_TTL_MS or from a superseded net-generation.
                prune_discover_cache(
                    &mut state.discover_cache,
                    now_ms,
                    state.net_generation.load(std::sync::atomic::Ordering::Relaxed),
                    DISCOVER_CACHE_TTL_MS,
                );
                // Phase 2.4: TTL-sweep leaked inbound proxy response channels.
                // Normally a channel is removed when the Python responder replies
                // (~lines 814/1022). If the responder crashes/drops, the channel
                // and its buffers leak until libp2p's protocol timeout — this
                // bounds that to PROXY_CHANNEL_TTL. Dropping the ResponseChannel
                // closes it, so the remote peer fails fast instead of hanging.
                let pc_before = state.inbound_proxy_channels.len();
                state.inbound_proxy_channels.retain(|_, (_, t, _)| t.elapsed() < PROXY_CHANNEL_TTL);
                let pc_swept = pc_before - state.inbound_proxy_channels.len();
                if pc_swept > 0 {
                    warn!(swept = pc_swept, remaining = state.inbound_proxy_channels.len(),
                          "inbound_proxy_channels TTL sweep (leaked unanswered channels)");
                }
                // C-N2: TTL-sweep queued relay-forwards whose relay dial never
                // connected. Without this a permanently-undialable target leaks
                // its reply channel + buffered bytes forever (the drain only
                // happens on ConnectionEstablished). Failing the reply lets the
                // caller error out instead of hanging.
                let rf_before = state.pending_relay_forwards.len();
                let drained = std::mem::take(&mut state.pending_relay_forwards);
                state.pending_relay_forwards =
                    sweep_relay_forwards(drained, RELAY_FORWARD_TTL, std::time::Instant::now());
                let rf_swept = rf_before - state.pending_relay_forwards.len();
                if rf_swept > 0 {
                    warn!(swept = rf_swept, remaining = state.pending_relay_forwards.len(),
                          "pending_relay_forwards TTL sweep (relay dial never connected)");
                }
            }
            // C-N1: close now-stable peers' relay connections so only the fast
            // direct path remains (request_response's NotifyHandler::Any can no
            // longer pin traffic to the relay).
            _ = relay_close_ticker.tick() => {
                let now = std::time::Instant::now();
                let due: Vec<PeerId> = state
                    .relay_close_deadline
                    .iter()
                    .filter(|(_, &deadline)| now >= deadline)
                    .map(|(&pid, _)| pid)
                    .collect();
                for pid in due {
                    state.relay_close_deadline.remove(&pid);
                    // Re-check live state: only close relays if a direct path is
                    // still up (it may have dropped and been un-armed already —
                    // this is the belt to that suspenders) and relays still exist.
                    let relay_ids = state
                        .peer_connections
                        .get(&pid)
                        .map(relays_to_close)
                        .unwrap_or_default();
                    if relay_ids.is_empty() {
                        continue;
                    }
                    for cid in &relay_ids {
                        // ConnectionClosed will decrement tcp_relay and drop the
                        // id from relay_conn_ids — no bookkeeping needed here.
                        let _ = swarm.close_connection(*cid);
                    }
                    info!(%pid, closed = relay_ids.len(),
                          "C-N1: closed relay connection(s) — direct path stabilized");
                }

                // v6-pref: retire now-redundant v4 QUIC directs for peers whose v6
                // direct has stabilised, so traffic settles on the v6 rung.
                let v4_due: Vec<PeerId> = state
                    .v4_close_deadline
                    .iter()
                    .filter(|(_, &deadline)| now >= deadline)
                    .map(|(&pid, _)| pid)
                    .collect();
                for pid in v4_due {
                    state.v4_close_deadline.remove(&pid);
                    // Re-check: only retire v4 if the v6 direct is still up (else we
                    // just un-armed and depend on v4 again — belt to that suspenders).
                    let v4_ids = state
                        .peer_connections
                        .get(&pid)
                        .map(v4s_to_close)
                        .unwrap_or_default();
                    if v4_ids.is_empty() {
                        continue;
                    }
                    for cid in &v4_ids {
                        // ConnectionClosed decrements quic_direct_v4 + drops the id.
                        let _ = swarm.close_connection(*cid);
                    }
                    info!(%pid, closed = v4_ids.len(),
                          "v6-pref: retired v4 QUIC direct(s) — v6 path stabilized");
                }
            }
            // R-DHT-6: snapshot the routing table to disk for warm restarts.
            _ = routing_save_ticker.tick() => {
                if let Some(ref cache_path) = routing_cache_path {
                    let entries: Vec<(PeerId, Vec<Multiaddr>)> = swarm
                        .behaviour_mut()
                        .kademlia
                        .kbuckets()
                        .flat_map(|bucket| {
                            bucket
                                .iter()
                                .map(|e| {
                                    (
                                        *e.node.key.preimage(),
                                        e.node.value.iter().cloned().collect::<Vec<_>>(),
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    // Skip writing an empty table (e.g. the immediate first tick
                    // before any contacts are learned) — don't clobber a good cache.
                    if !entries.is_empty() {
                        match routing_cache::save(cache_path, &entries) {
                            Ok(()) => debug!(contacts = entries.len(), "R-DHT-6: persisted routing table"),
                            Err(e) => warn!(%e, "R-DHT-6: routing table persist failed"),
                        }
                    }
                }
            }
            // R-DHT-4: re-assert UPnP external addresses (recover from a transient
            // AutoNAT-Private demotion). Suppressed while AutoNAT holds Private.
            _ = upnp_reassert_ticker.tick() => {
                if !state.upnp_external_addrs.is_empty() {
                    let confirmed: std::collections::HashSet<Multiaddr> =
                        swarm.external_addresses().cloned().collect();
                    let missing = upnp_addrs_to_reassert(
                        &state.upnp_external_addrs,
                        &confirmed,
                        state.autonat_private,
                    );
                    let restored_global = missing.iter().any(is_globally_reachable_addr);
                    for addr in missing {
                        info!(%addr, "R-DHT-4: re-asserting UPnP external addr (AutoNAT not Private)");
                        swarm.add_external_address(addr);
                    }
                    // Audit fix: under R-DHT-2 explicit Kad mode, re-adding an
                    // external address no longer auto-promotes — so the re-assert
                    // must also restore server mode. A UPnP mapping is a positive
                    // reachability signal and we only get here when AutoNAT isn't
                    // asserting Private, so promoting is correct.
                    if restored_global {
                        swarm.behaviour_mut().kademlia.set_mode(Some(kad::Mode::Server));
                        state.kad_server_mode = true;
                    }
                }
            }
            // F6: B5 periodic relay-renewal removed (see comment at ticker
            // decl). Reservation renewal is handled by libp2p auto-renewal +
            // the reactive ExpiredListenAddr / ListenerClosed handlers.
            // F-5: drive any due relay-RESERVATION retries (re-listen on
            // circuits whose reservation was lost).
            _ = relay_retry_ticker.tick() => {
                drive_reservation_retries(&mut swarm, &mut state);
            }
            // #42: connectivity watchdog + debounced reactive rebootstrap.
            _ = heal_ticker.tick() => {
                drive_heal(&mut swarm, &mut state);
            }
            // #43-W2: a PCP pinhole confirmed an inbound v6 addr — advertise it
            // as an external candidate so AutoNAT probes it and, if reachable,
            // promotes us (the same path UPnP/AutoNAT use).
            Some(addr) = pcp_candidate_rx.recv() => {
                if !state.external_addrs.iter().any(|a| a == &addr) {
                    info!(%addr, "pcp: adding confirmed external v6 addr (AutoNAT will verify)");
                    swarm.add_external_address(addr.clone());
                    state.external_addrs.push(addr);
                }
            }
            // Process swarm events.
            event = swarm.select_next_some() => {
                handle_swarm_event(event, &mut swarm, &mut state, &proxy_queue);
            }
        }
    }
}

/// Handle an announce command: sign the peer record, PUT it into Kademlia,
/// and register as a provider for the model (Tasks 2.1, 6.2).
fn handle_announce(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    record: &PeerRecord,
    reply: oneshot::Sender<Result<(), String>>,
    state: &mut LoopState,
    keypair: &libp2p::identity::Keypair,
) {
    // Task 6.2: sign the record with our Ed25519 keypair.
    let signed_record = match dht::sign_peer_record(record, keypair) {
        Ok(r) => r,
        Err(e) => {
            warn!("announce sign failed: {e}");
            // Fall back to unsigned record.
            record.clone()
        }
    };
    let local_libp2p_id = swarm.local_peer_id().to_base58();
    let key = dht::peer_record_key(&signed_record.model_id, &local_libp2p_id);
    match dht::encode_record(&signed_record) {
        Ok(value) => {
            let kad_record = kad::Record {
                key,
                value,
                publisher: Some(*swarm.local_peer_id()),
                expires: Some(std::time::Instant::now() + std::time::Duration::from_secs(300)),
            };
            match swarm
                .behaviour_mut()
                .kademlia
                .put_record(kad_record, kad::Quorum::One)
            {
                Ok(_) => {
                    // Register as provider for this model (Option C: provider API)
                    let model_key = dht::model_provider_key(&signed_record.model_id);
                    if let Err(e) = swarm.behaviour_mut().kademlia.start_providing(model_key) {
                        warn!(model_id = %signed_record.model_id, "start_providing failed: {e:?}");
                    }
                    // Cache locally for fast resolve + discover cache hits
                    state.known_peers.insert(known_peer_key(&signed_record), signed_record.clone());
                    info!(
                        model_id = %signed_record.model_id,
                        peer_id = %signed_record.peer_id,
                        "announced to kademlia (provider + record)"
                    );
                    // R-DHT-1: also advertise this signed record over gossipsub so
                    // peers learn us without the DHT (PEX). Best-effort — a publish
                    // failure (e.g. no gossip mesh peers yet) must never fail the
                    // announce; the next periodic re-announce will retry.
                    publish_provider_pex(swarm, &signed_record);
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

/// R-DHT-1: publish a signed `PeerRecord` as a `PROVIDER_ANNOUNCE` gossipsub
/// message so the swarm learns this provider independently of the DHT (PEX).
/// Best-effort: a publish error is logged at debug and swallowed — callers must
/// not let it fail the announce.
fn publish_provider_pex(swarm: &mut libp2p::Swarm<OpenHydraBehaviour>, record: &PeerRecord) {
    let envelope = serde_json::json!({
        "type": PROVIDER_ANNOUNCE_TYPE,
        "record": record,
    });
    let payload = match serde_json::to_vec(&envelope) {
        Ok(p) => p,
        Err(e) => {
            warn!(%e, "provider_pex: failed to encode envelope");
            return;
        }
    };
    let topic = libp2p::gossipsub::IdentTopic::new(crate::swarm::GOSSIPSUB_TOPIC);
    match swarm.behaviour_mut().gossipsub.publish(topic, payload) {
        Ok(_) => debug!(
            model_id = %record.model_id,
            peer_id = %record.peer_id,
            "provider_pex: gossiped PROVIDER_ANNOUNCE"
        ),
        // InsufficientPeers is expected right after startup before the mesh forms;
        // the periodic re-announce will publish again once peers are connected.
        Err(e) => debug!(%e, "provider_pex: gossip publish skipped"),
    }
}

/// R-DHT-1: ingest an inbound `PROVIDER_ANNOUNCE` gossip message (PEX). Verifies
/// the embedded signed record AND that the gossip author equals the record's
/// `libp2p_peer_id` (see [`dht::pex_record_is_authentic`]) before caching it into
/// `known_peers`, where `handle_discover` can return it without the DHT.
fn handle_provider_pex(
    state: &mut LoopState,
    parsed: &serde_json::Value,
    source: Option<PeerId>,
) {
    // The verified gossip author. Strict signing guarantees this is present for
    // accepted messages, but guard anyway — no author means no trust anchor.
    let source_str = match source {
        Some(pid) => pid.to_base58(),
        None => {
            debug!("provider_pex: dropping advert with no signed author");
            return;
        }
    };
    let record: PeerRecord = match parsed.get("record") {
        Some(v) => match serde_json::from_value(v.clone()) {
            Ok(r) => r,
            Err(e) => {
                debug!(%e, "provider_pex: malformed record in advert");
                return;
            }
        },
        None => {
            debug!("provider_pex: advert missing record field");
            return;
        }
    };
    if let Err(e) = dht::pex_record_is_authentic(&record, &source_str) {
        warn!(%source_str, %e, "provider_pex: rejected advert (failed authenticity)");
        return;
    }
    let is_new = !state.known_peers.contains_key(&known_peer_key(&record));
    state.known_peers.insert(known_peer_key(&record), record.clone());
    if is_new {
        info!(
            model_id = %record.model_id,
            peer_id = %record.peer_id,
            libp2p = %record.libp2p_peer_id,
            "provider_pex: learned new provider via gossip"
        );
    } else {
        debug!(peer_id = %record.peer_id, "provider_pex: refreshed cached provider");
    }
}

/// C11: is our last completed discover for `model` still within `ttl_ms` and the same
/// net-generation? Pure, so the cache-freshness policy is unit-testable without a swarm.
fn discover_cache_fresh(
    cache: &HashMap<String, (u64, u64)>,
    model: &str,
    now_ms: u64,
    cur_gen: u64,
    ttl_ms: u64,
) -> bool {
    matches!(cache.get(model), Some(&(at_ms, at_gen)) if at_gen == cur_gen && now_ms.saturating_sub(at_ms) < ttl_ms)
}

/// C11: drop discover-cache entries past `ttl_ms` or from a superseded net-generation, so the
/// map stays bounded to models discovered in the last TTL under the current generation — it
/// can't grow unbounded across distinct (client-supplied) model_ids, nor accumulate dead
/// entries after each roam/wake bumps the generation. Keep-condition mirrors
/// [`discover_cache_fresh`]. Run on the periodic reaper tick.
fn prune_discover_cache(cache: &mut HashMap<String, (u64, u64)>, now_ms: u64, cur_gen: u64, ttl_ms: u64) {
    cache.retain(|_model, &mut (at_ms, at_gen)| at_gen == cur_gen && now_ms.saturating_sub(at_ms) < ttl_ms);
}

/// Build `DiscoveredPeer`s from records, stamping the R-DHT-8 liveness hint (is there a live
/// libp2p connection to each provider right now). Shared by the C11 cache-hit early-reply and
/// the query-completion reply path so both surface the same connected/failover signal.
fn discovered_from_records(state: &LoopState, records: &[PeerRecord]) -> Vec<DiscoveredPeer> {
    let mut peers: Vec<DiscoveredPeer> = records.iter().map(record_to_discovered).collect();
    for p in &mut peers {
        if let Ok(pid) = p.libp2p_peer_id.parse::<PeerId>() {
            p.connected = state.peer_connections.contains_key(&pid);
        }
    }
    peers
}

/// Handle a discover command: find providers for the model via Kademlia
/// provider API (Task 2.1: Option C).
fn handle_discover(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    model_id: &str,
    reply: oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>,
    state: &mut LoopState,
) {
    // R-DHT-1: pre-seed the result from the local cache (`known_peers`), which is
    // populated by DHT reads AND by gossip PEX (`PROVIDER_ANNOUNCE`). This makes
    // discovery DHT-independent: if `get_providers` returns nothing — or the DHT is
    // entirely unreachable — the early-reply / convergence path still returns any
    // provider we learned over gossip. Exclude our own record. The get_providers
    // FoundProviders handler dedups by `libp2p_peer_id`, so a provider present in
    // both the cache and the DHT is not double-counted.
    let local_id = swarm.local_peer_id().to_base58();
    let seed: Vec<PeerRecord> = state
        .known_peers
        .values()
        .filter(|r| r.model_id == model_id && r.libp2p_peer_id != local_id)
        .cloned()
        .collect();

    // C11: if we completed a discover for this model recently (and haven't changed networks
    // since), answer straight from the cache seed — skipping the ~1.2 s get_providers
    // round-trip. The cold path below still runs (and refreshes the cache) once the entry
    // ages past DISCOVER_CACHE_TTL_MS or a net-generation change invalidates it. A stale hit
    // is bounded by the TTL and covered by the consumer's failover + the known_peers reaper.
    let now_ms = now_unix_ms();
    let cur_gen = state.net_generation.load(std::sync::atomic::Ordering::Relaxed);
    if !seed.is_empty()
        && discover_cache_fresh(&state.discover_cache, model_id, now_ms, cur_gen, DISCOVER_CACHE_TTL_MS)
    {
        let peers = discovered_from_records(state, &seed);
        debug!(model_id, n = peers.len(), "discover: C11 cache hit — skipped get_providers");
        let _ = reply.send(Ok(peers));
        return;
    }
    if !seed.is_empty() {
        debug!(model_id, seeded = seed.len(), "discover: pre-seeded from PEX/DHT cache");
    }

    let key = dht::model_provider_key(model_id);
    let query_id = swarm.behaviour_mut().kademlia.get_providers(key);
    state.pending_discovers.insert(
        query_id,
        PendingDiscover {
            model_id: model_id.to_string(),
            records: seed,
            reply,
            providers_done: false,
            outstanding: 0,
        },
    );
}

/// Handle a resolve_address command.
fn handle_resolve(
    state: &LoopState,
    peer_id: &str,
    reply: oneshot::Sender<Result<String, String>>,
) {
    // Look up the peer in our known_peers cache. `known_peers` is now keyed by (peer_id,
    // model_id), and the reachability fields (relay_address/host/port) are node-level — identical
    // across a node's models — so match by the record's peer_id and take any of its entries.
    if let Some(record) = state.known_peers.values().find(|r| r.peer_id == peer_id) {
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
/// Evict a peer that failed an outgoing dial: drop it from the Kademlia routing
/// table and the known-peers cache so routing stops preferring a dead path. A
/// later serve re-discovers the peer on demand, so eviction is safe and cheap.
/// (No-op if the peer is in fact still connected.)
fn evict_unreachable_peer(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    pid: PeerId,
) {
    if swarm.is_connected(&pid) {
        return;
    }
    let pid_str = pid.to_string();
    swarm.behaviour_mut().kademlia.remove_peer(&pid);
    state.known_peers.retain(|_, r| r.libp2p_peer_id != pid_str);
    info!(%pid, "evicted unreachable peer after dial failure");
}

/// F-5: backoff schedule (in ms) for relay-RESERVATION retries.
///
/// ``attempt`` is 1-based (the attempt number we just made / are scheduling
/// the wait after). Fast initial retries then exponential backoff, capped at
/// 120 s: 1s, 2s, 4s, 8s, 16s, 32s, 64s, 120s, 120s, … So a flapping or
/// down relay neither tight-loops the event loop nor strands us permanently —
/// we keep probing roughly every 2 minutes once backed off, and recover within
/// a couple seconds of a brief blip. Pure + unit-tested.
fn reservation_retry_delay_ms(attempt: u32) -> u64 {
    // Cap the shift so ``1 << shift`` can't overflow and the value saturates
    // at the 120 s ceiling well before then.
    let shift = attempt.saturating_sub(1).min(7);
    let ms = 1000u64.saturating_mul(1u64 << shift);
    ms.min(120_000)
}

/// F-5: record that a relay reservation for ``circuit_addr`` is lost and
/// schedule the next retry with backoff. Idempotent-ish: advances the attempt
/// counter and pushes the next-attempt deadline out per the backoff schedule.
fn schedule_reservation_retry(state: &mut LoopState, circuit_addr: &str) {
    let now = tokio::time::Instant::now();
    let entry = state
        .relay_reservation_retries
        .entry(circuit_addr.to_string())
        .or_insert((0, now));
    entry.0 = entry.0.saturating_add(1);
    let delay = reservation_retry_delay_ms(entry.0);
    entry.1 = now + std::time::Duration::from_millis(delay);
    warn!(
        circuit = %circuit_addr, attempt = entry.0, backoff_ms = delay,
        "relay reservation lost — retry scheduled with backoff (F-5)",
    );
}

/// F-5: re-issue any due relay-reservation ``listen_on`` requests. Driven by
/// the existing `relay_retry_ticker` (250 ms). A circuit is retried only once
/// its scheduled deadline passes, so repeated failure events coalesce into a
/// single spaced attempt instead of a tight loop.
fn drive_reservation_retries(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
) {
    if state.relay_reservation_retries.is_empty() {
        return;
    }
    let now = tokio::time::Instant::now();
    // Collect due addrs first to avoid holding a borrow on `state` while we
    // mutate the swarm and re-schedule.
    let due: Vec<String> = state
        .relay_reservation_retries
        .iter()
        .filter(|(_, (_, next))| now >= *next)
        .map(|(addr, _)| addr.clone())
        .collect();
    for addr_str in due {
        if let Ok(addr) = addr_str.parse::<Multiaddr>() {
            match swarm.listen_on(addr.clone()) {
                Ok(_) => info!(circuit = %addr_str, "relay re-reservation requested (F-5 retry)"),
                Err(e) => warn!(circuit = %addr_str, %e, "relay re-reservation listen_on failed; will retry"),
            }
        } else {
            warn!(circuit = %addr_str, "relay re-reservation skipped: unparseable circuit addr");
        }
        // Push the next attempt out per the backoff schedule. Cleared entirely
        // on success (NewListenAddr / ReservationReqAccepted for this circuit).
        schedule_reservation_retry(state, &addr_str);
    }
}

/// F-5: a reservation succeeded for ``addr`` — clear any pending retry so we
/// stop probing. Matches on the circuit substring so a peer-id-suffixed listen
/// addr still clears the base circuit retry entry.
fn clear_reservation_retry(state: &mut LoopState, addr: &str) {
    if state.relay_reservation_retries.is_empty() {
        return;
    }
    state
        .relay_reservation_retries
        .retain(|circuit, _| !addr.starts_with(circuit.as_str()) && !circuit.starts_with(addr));
}

// ── #42: network-change resilience ────────────────────────────────────────
/// How long connectivity must stay degraded (0 connected peers) before the
/// watchdog rebuilds it. Long enough that a brief blip doesn't trigger a
/// needless rebuild, short enough to recover a roam/wake promptly.
const DEGRADED_GRACE: std::time::Duration = std::time::Duration::from_secs(20);
/// Debounce window that collapses a burst of listener changes (a roam emits
/// several NewListenAddr/Expired in quick succession) into one rebuild.
const REBOOTSTRAP_DEBOUNCE: std::time::Duration = std::time::Duration::from_secs(3);
/// Minimum spacing between rebuilds — a cooldown so triggers that keep firing
/// (a flapping interface) can't spin the bootstrap sequence.
const REBOOTSTRAP_COOLDOWN: std::time::Duration = std::time::Duration::from_secs(30);

/// Is `addr` a real (non-loopback, non-circuit) interface address — i.e. does
/// its appearance/disappearance signal an actual network change? Loopback
/// (127.0.0.1 / ::1) churns on nothing meaningful, and `/p2p-circuit` addrs are
/// relay reservations owned by the F-5 path, not interface events. Pure.
fn is_real_interface_addr(addr: &Multiaddr) -> bool {
    if addr.to_string().contains("/p2p-circuit") {
        return false;
    }
    !addr.iter().any(|p| match p {
        libp2p::multiaddr::Protocol::Ip4(ip) => ip.is_loopback(),
        libp2p::multiaddr::Protocol::Ip6(ip) => ip.is_loopback(),
        _ => false,
    })
}

/// Arm a debounced rebootstrap after a real interface change. Coalesces a burst
/// (roam) into a single rebuild fired `REBOOTSTRAP_DEBOUNCE` after the last
/// trigger; the heal ticker does the firing.
fn arm_reactive_rebootstrap(state: &mut LoopState, reason: &str) {
    state.pending_rebootstrap_at = Some(tokio::time::Instant::now() + REBOOTSTRAP_DEBOUNCE);
    debug!(reason, "network-change: reactive rebootstrap armed (debounced)");
}

/// An interface went away — drop the direct external addresses tied to it and
/// demote to Kad client so AutoNAT re-confirms on the *new* network rather than
/// advertising a dead address. Circuit/relay external addrs are left to the F-5
/// reservation path. Called on a real (non-circuit) listener close/expiry.
fn expire_direct_external_addrs(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
) {
    let stale: Vec<Multiaddr> = state
        .external_addrs
        .iter()
        .filter(|a| !a.to_string().contains("/p2p-circuit"))
        .cloned()
        .collect();
    if stale.is_empty() {
        return;
    }
    for a in &stale {
        swarm.remove_external_address(a);
    }
    state
        .external_addrs
        .retain(|a| a.to_string().contains("/p2p-circuit"));
    // No verified-reachable direct addr remains → back to client; AutoNAT will
    // re-probe fresh candidates on the new network and re-promote if reachable.
    swarm
        .behaviour_mut()
        .kademlia
        .set_mode(Some(kad::Mode::Client));
    state.kad_server_mode = false;
    state.autonat_private = false;
    state.nat_info.nat_type = "unknown".into();
    state.nat_info.is_public = false;
    info!(
        dropped = stale.len(),
        "network-change: expired stale direct external addrs, demoted to Kad client"
    );
}

/// Rebuild connectivity under the *same* pinned identity — the network changed
/// (roam / wake / interface up-down), so the routing table, bootstrap
/// connections and relay reservations are stale. Idempotent and cheap.
///
/// (A) re-establish connectivity: re-seed Kademlia, re-dial every bootstrap peer
/// **including the relays** (the `non_relay_bootstrap` startup retry deliberately
/// skips relays, so nothing else re-dials a relay whose connection died on a
/// roam), and re-request any relay reservation we no longer hold. Clears the
/// hole-punch / reversal back-off so a fresh network gets fresh punch attempts.
/// (B) signal the provider: bump `net_generation` so it re-announces its record
/// with the new addresses (same PeerId, new relay path).
fn rebootstrap(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    reason: &str,
) {
    state.rebootstrap_count += 1;
    state.last_rebootstrap = Some(tokio::time::Instant::now());
    info!(
        reason,
        count = state.rebootstrap_count,
        "rebootstrap: rebuilding connectivity (network change)"
    );

    // Re-seed Kademlia from the bootstrap peers.
    if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
        debug!(%e, "rebootstrap: kademlia.bootstrap deferred (no peers yet)");
    }

    // Re-dial every bootstrap peer (relays included) to re-establish the direct
    // links a roam/wake tore down. libp2p dedups if a connection already exists.
    for (peer_id, addr) in &state.bootstrap_peers {
        let dial_addr = addr.clone().with(libp2p::multiaddr::Protocol::P2p(*peer_id));
        if let Err(e) = swarm.dial(dial_addr.clone()) {
            debug!(%peer_id, %dial_addr, %e, "rebootstrap: bootstrap re-dial failed");
        }
    }

    // Re-request only the relay reservations we don't currently hold. Reserving
    // a circuit we already have would leak a duplicate listener (F6), so skip
    // any circuit still in `reserved_circuits`; the F-5 reactive path owns those
    // that genuinely dropped.
    let mut seen_relays = std::collections::HashSet::new();
    for relay_str in crate::relay::BOOTSTRAP_RELAYS {
        if let Ok(relay_multiaddr) = relay_str.parse::<Multiaddr>() {
            let relay_peer = relay_multiaddr.iter().find_map(|p| match p {
                libp2p::multiaddr::Protocol::P2p(id) => Some(id),
                _ => None,
            });
            if let Some(pid) = relay_peer {
                if !seen_relays.insert(pid) {
                    continue;
                }
            }
            let listen_addr = relay_multiaddr.with(libp2p::multiaddr::Protocol::P2pCircuit);
            let listen_str = listen_addr.to_string();
            if state.reserved_circuits.contains(&listen_str) {
                continue; // already reserved — don't leak a duplicate listener
            }
            match swarm.listen_on(listen_addr.clone()) {
                Ok(_) => info!(addr = %listen_addr, "rebootstrap: re-requesting relay reservation"),
                Err(e) => {
                    warn!(addr = %listen_addr, %e, "rebootstrap: relay re-reservation failed");
                    schedule_reservation_retry(state, &listen_str);
                }
            }
        }
    }

    // #43-W1: a new network is a fresh chance to hole-punch — clear the QUIC-v6
    // punch and reversal back-off so shelved peers get re-attempted.
    state.quic_holepunch_attempts.clear();
    state.reversal_attempts.clear();

    // H (F-9): re-probe IPv6 capability on every rebootstrap (roam/wake). It was previously
    // evaluated only once at startup, so a node that booted on a v4-only link and roamed onto
    // a v6-capable one would forgo the v6-first path (and the mirror case would keep dialing
    // unreachable v6 relays) for the rest of the process lifetime. The probe is traffic-free.
    let was_v6 = state.ipv6_capable;
    state.ipv6_capable = probe_ipv6_capable();
    if state.ipv6_capable != was_v6 {
        info!(
            ipv6_capable = state.ipv6_capable,
            "F-9: IPv6 capability changed on rebootstrap (re-probed)"
        );
    }

    // (B) signal the provider to re-announce (fresh relay addr, same PeerId).
    state
        .net_generation
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Run `rebootstrap` unless one ran within the cooldown. Central choke so the
/// watchdog and the reactive trigger can't double-fire a rebuild.
fn maybe_rebootstrap(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    reason: &str,
) {
    if let Some(last) = state.last_rebootstrap {
        if last.elapsed() < REBOOTSTRAP_COOLDOWN {
            debug!(reason, "rebootstrap suppressed (cooldown)");
            return;
        }
    }
    rebootstrap(swarm, state, reason);
}

/// The connectivity watchdog tick: rebuild if we've had zero connected peers for
/// longer than `DEGRADED_GRACE` (wake-from-sleep / roam catch-all), and fire any
/// due debounced reactive rebootstrap. Runs on the heal ticker.
fn drive_heal(swarm: &mut libp2p::Swarm<OpenHydraBehaviour>, state: &mut LoopState) {
    let now = tokio::time::Instant::now();
    // Fire a due debounced reactive rebootstrap first (a real interface change).
    if let Some(at) = state.pending_rebootstrap_at {
        if now >= at {
            state.pending_rebootstrap_at = None;
            maybe_rebootstrap(swarm, state, "reactive: interface change");
        }
    }
    // Connectivity watchdog: sustained 0 connected peers ⇒ rebuild.
    let connected = swarm.network_info().num_peers() > 0;
    if connected {
        state.degraded_since = None;
    } else {
        let since = *state.degraded_since.get_or_insert(now);
        if now.duration_since(since) >= DEGRADED_GRACE {
            maybe_rebootstrap(swarm, state, "watchdog: no connected peers");
            // Reset the timer so we wait a full grace window (plus the cooldown)
            // before the next watchdog-driven attempt.
            state.degraded_since = Some(now);
        }
    }
}

// ── #42 follow-up: zombie-connection liveness gating ───────────────────────
/// Consecutive proxy outbound *path* failures to one peer before its
/// connections are presumed zombies and force-closed. 2 balances the live
/// failure modes: a single timeout can be a transiently slow provider (a long
/// generation, a CGNAT blip), but two in a row on the same peer — while
/// `send_request` is free to pick any of its connections — means every path we
/// hold is suspect. The roam incident this fixes burned 6+ minutes of
/// consecutive timeouts against connections whose network the peer had left.
const ZOMBIE_FAILURE_THRESHOLD: u32 = 2;

/// Record a proxy outbound failure for `peer` and decide whether to evict its
/// connections. `dead_path` = the failure class implicates the transport path
/// (timeout / connection closed / dial failure / io) rather than the protocol
/// (an `UnsupportedProtocols` peer is alive and answering — never evict on it).
/// Returns `true` when the streak reaches [`ZOMBIE_FAILURE_THRESHOLD`]; the
/// streak resets so a re-established peer starts clean. Pure + unit-tested.
fn record_proxy_failure(
    streaks: &mut HashMap<PeerId, u32>,
    peer: PeerId,
    dead_path: bool,
) -> bool {
    if !dead_path {
        return false;
    }
    let streak = streaks.entry(peer).or_insert(0);
    *streak += 1;
    if *streak >= ZOMBIE_FAILURE_THRESHOLD {
        streaks.remove(&peer);
        true
    } else {
        false
    }
}

/// A proxy round-trip to `peer` succeeded — its current path is live, so any
/// accumulated failure streak is stale. Pure + unit-tested.
fn record_proxy_success(streaks: &mut HashMap<PeerId, u32>, peer: &PeerId) {
    streaks.remove(peer);
}


fn handle_swarm_event(
    event: SwarmEvent<OpenHydraBehaviourEvent>,
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    proxy_queue: &SharedProxyQueue,
) {
    match event {
        // ── Kademlia ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Kademlia(kad_event)) => {
            handle_kad_event(kad_event, swarm, state);
        }

        // ── AutoNAT v1 ──
        // ── AutoNAT v2 (R-DHT-11; the only AutoNAT — v1 retired) ──
        // Per-address verdict from a v2 server: `Ok` ⇒ this *specific* address is
        // reachable by an arbitrary peer (the reliable promotion signal v1 lacked;
        // also covers IPv6 explicitly). `Err` ⇒ that address is not reachable. This
        // handler also owns the two side-effects v1's `StatusChanged` used to own:
        // populating `nat_info` (surfaced via the NatStatus query) and toggling
        // `autonat_private` (which gates the R-DHT-4 UPnP re-assert).
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::AutonatV2Client(ev)) => {
            match &ev.result {
                Ok(()) if is_globally_reachable_addr(&ev.tested_addr) => {
                    info!(addr = %ev.tested_addr, server = %ev.server,
                        "R-DHT-11: AutoNAT v2 confirmed reachable → promoting to Kad server");
                    if !state.external_addrs.iter().any(|a| a == &ev.tested_addr) {
                        state.external_addrs.push(ev.tested_addr.clone());
                    }
                    state.autonat_private = false; // R-DHT-4: re-assert may resume
                    // Reachable on a direct global addr ⇒ "open". Record the
                    // verified external IP (classified by family) for the
                    // NatStatus query consumers (was v1's NatStatus::Public path).
                    state.nat_info.nat_type = "open".into();
                    state.nat_info.is_public = true;
                    if let Some(ip) = extract_ip_from_multiaddr(&ev.tested_addr) {
                        state.nat_info.external_ip = ip.clone();
                        if ip.contains(':') {
                            state.nat_info.external_ipv6 = ip;
                        } else {
                            state.nat_info.external_ipv4 = ip;
                        }
                    }
                    swarm.add_external_address(ev.tested_addr.clone());
                    swarm.behaviour_mut().kademlia.set_mode(Some(kad::Mode::Server));
                    state.kad_server_mode = true;
                }
                Ok(()) => {
                    debug!(addr = %ev.tested_addr, "R-DHT-11: AutoNAT v2 OK but non-global addr; not promoting");
                }
                Err(e) => {
                    // This address is not reachable. Retract it if we'd confirmed
                    // it, and demote to client if no direct external remains.
                    debug!(addr = %ev.tested_addr, server = %ev.server, ?e,
                        "R-DHT-11: AutoNAT v2 says address not reachable");
                    if state.external_addrs.iter().any(|a| a == &ev.tested_addr) {
                        swarm.remove_external_address(&ev.tested_addr);
                        state.external_addrs.retain(|a| a != &ev.tested_addr);
                        let has_direct = state
                            .external_addrs
                            .iter()
                            .any(|a| !a.to_string().contains("/p2p-circuit"));
                        if !has_direct {
                            // No verified-reachable direct address remains: we're
                            // effectively private. Demote to client, mark
                            // nat_info accordingly, and suppress the UPnP re-assert
                            // (this was v1's NatStatus::Private path — but driven
                            // by a concrete failed dial-back rather than a v1
                            // `Private` verdict that for unreachable nodes only
                            // ever timed out into `Unknown`).
                            swarm.behaviour_mut().kademlia.set_mode(Some(kad::Mode::Client));
                            state.kad_server_mode = false;
                            state.nat_info.nat_type = "symmetric".into();
                            state.nat_info.is_public = false;
                            state.autonat_private = true; // R-DHT-4: suppress re-assert
                        }
                    }
                }
            }
        }

        // ── Identify ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Identify(identify_event)) => {
            if let libp2p::identify::Event::Received { peer_id, info, .. } = identify_event {
                debug!(%peer_id, protocol = %info.protocol_version, "identify received");
                for addr in &info.listen_addrs {
                    // F-9: skip IPv6 peer addresses on a host with no working
                    // IPv6 — adding them to the routing table only leads libp2p
                    // to later dial unreachable `/ip6/` addresses and eat
                    // connection timeouts. (Relay v6 circuits are `/ip6/` too,
                    // and equally undialable without v6.)
                    if !state.ipv6_capable && is_ipv6_multiaddr(addr) {
                        continue;
                    }
                    swarm
                        .behaviour_mut()
                        .kademlia
                        .add_address(&peer_id, addr.clone());
                }
                // R-DHT-2 (revised after the 2026-06-15 live test): do NOT confirm
                // an Identify *observed* address as external. An observed addr is
                // only what one peer claims to see; for a CGNAT/symmetric or
                // firewalled node it is not reachable by an arbitrary querier, and
                // confirming it optimistically created a black-hole server that
                // AutoNAT could not reliably tear down — a `Private` verdict for an
                // unreachable node times out into `Unknown`, so the demotion never
                // fired. libp2p still surfaces observed addrs as
                // `NewExternalAddrCandidate`s; AutoNAT v2 probes those and only a
                // positive per-address verdict promotes us (see the
                // AutonatV2Client handler).
                debug!(addr = %info.observed_addr, "identify observed addr (AutoNAT candidate; not auto-confirmed)");

                // Fix 2/4: cache QUIC IPv6 addresses and auto-dial for hole-punch.
                // F-9: on a host with no working IPv6, never collect (and thus
                // never auto-dial) peer QUIC v6 addrs — those dials are the
                // costly ones (QUIC handshake timeouts on unreachable v6).
                let quic_v6_addrs: Vec<Multiaddr> = if !state.ipv6_capable {
                    Vec::new()
                } else {
                    info.listen_addrs.iter()
                        .filter(|a| {
                            let s = a.to_string();
                            s.contains("/quic") && s.contains("/ip6/") && !s.contains("p2p-circuit")
                        })
                        .cloned()
                        .collect()
                };

                if !quic_v6_addrs.is_empty() {
                    state.peer_quic_addrs.insert(peer_id, quic_v6_addrs.clone());

                    // Auto QUIC hole-punch: pursue a *v6* QUIC direct to this peer
                    // by dialing their QUIC IPv6 addresses. v6-first: a v4 QUIC
                    // direct does NOT count as done — we keep upgrading toward v6
                    // (bounded by the back-off below) because v6's firewall punch
                    // is far more stable than a v4 CGNAT hole.
                    if !should_pursue_v6_upgrade(state.peer_connections.get(&peer_id)) {
                        // Already hold the preferred v6 QUIC direct — reset back-off.
                        state.quic_holepunch_attempts.remove(&peer_id);
                    } else {
                        // F7: skip public bootstrap relays — they're TCP-reachable
                        // and hole-punch is meaningless for them; auto-QUIC-dialing
                        // them is what churned the connection on UDP-hostile paths.
                        let is_relay = quic_v6_addrs.iter().any(|a| {
                            a.iter().any(|p| match p {
                                libp2p::multiaddr::Protocol::Ip6(ip) =>
                                    crate::relay::is_bootstrap_relay_ip(&ip.to_string()),
                                libp2p::multiaddr::Protocol::Ip4(ip) =>
                                    crate::relay::is_bootstrap_relay_ip(&ip.to_string()),
                                _ => false,
                            })
                        });
                        // F7 / #43-W1: back off after MAX_QUIC_HOLEPUNCH_ATTEMPTS
                        // so a UDP-filtered path stops re-dialing (and re-logging)
                        // every identify cycle. The cap was 3; raised to 8 for the
                        // v6 case: a stateful-firewall v6 punch is *reliable once
                        // correctly timed* (no port translation, unlike v4 CGNAT),
                        // so a firewalled-v6 peer shouldn't be permanently shelved
                        // after only 3 uncoordinated misses — it deserves more
                        // eager retries alongside libp2p's coordinated DCUtR. The
                        // back-off is also cleared wholesale on `rebootstrap()`, so
                        // a network change re-enables punching from scratch.
                        const MAX_QUIC_HOLEPUNCH_ATTEMPTS: u32 = 8;
                        let attempts = state.quic_holepunch_attempts.entry(peer_id).or_insert(0);
                        if is_relay {
                            debug!(%peer_id, "auto_quic_holepunch_skip: public relay");
                        } else if *attempts >= MAX_QUIC_HOLEPUNCH_ATTEMPTS {
                            debug!(%peer_id, attempts = *attempts, "auto_quic_holepunch_skip: backed off");
                        } else {
                            *attempts += 1;
                            for addr in &quic_v6_addrs {
                                use libp2p::swarm::dial_opts::DialOpts;
                                let ma = addr.clone();
                                match swarm.dial(DialOpts::unknown_peer_id().address(addr.clone()).build()) {
                                    Ok(()) => {
                                        state.quic_v6_holepunch_dials += 1;
                                        debug!(%peer_id, %ma, "auto_quic_holepunch_dial");
                                    }
                                    Err(e) => debug!(%peer_id, %ma, %e, "auto_quic_holepunch_dial_failed"),
                                }
                            }
                        }
                    }
                }

                // ── Connection reversal (Tier 2) ──
                // If enabled and we hold no *direct* connection to this peer (only a
                // relay), dial the globally-routable direct addresses they advertised.
                // Our outbound dial traverses our own NAT even on symmetric CGNAT — the
                // one escape DCUtR can't provide. Safe by construction: we only dial a
                // peer we're already in an authenticated session with, only addrs libp2p
                // surfaced via Identify, only globally-routable ones, with per-peer
                // back-off. See docs/PEER_CONNECTIVITY.md.
                if state.enable_connection_reversal {
                    let has_direct = state
                        .peer_connections
                        .get(&peer_id)
                        .is_some_and(|c| c.has_direct());
                    if !has_direct {
                        let candidates = reversal_candidate_addrs(
                            &info.listen_addrs,
                            state.ipv6_capable,
                            &quic_v6_addrs,
                        );
                        if !candidates.is_empty() {
                            const MAX_REVERSAL_ATTEMPTS: u32 = 3;
                            let attempts = state.reversal_attempts.entry(peer_id).or_insert(0);
                            if *attempts < MAX_REVERSAL_ATTEMPTS {
                                *attempts += 1;
                                state.reversal_pending.insert(peer_id);
                                for addr in &candidates {
                                    use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
                                    let opts = DialOpts::peer_id(peer_id)
                                        .addresses(vec![addr.clone()])
                                        .condition(PeerCondition::Always)
                                        .build();
                                    match swarm.dial(opts) {
                                        Ok(()) => {
                                            state.reversal_dials += 1;
                                            info!(%peer_id, %addr, "connection_reversal_dial");
                                        }
                                        Err(e) => {
                                            debug!(%peer_id, %addr, %e, "connection_reversal_dial_failed")
                                        }
                                    }
                                }
                            }
                        }
                    }
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
                // Phase 1.7: mDNS expiry — evict peers that are no longer
                // reachable on LAN.
                libp2p::mdns::Event::Expired(peers) => {
                    for (peer_id, _) in peers {
                        info!(%peer_id, "mDNS peer expired");
                        if !swarm.is_connected(&peer_id) {
                            swarm.behaviour_mut().kademlia.remove_peer(&peer_id);
                            let libp2p_id_str = peer_id.to_string();
                            state.known_peers.retain(|_, record| {
                                record.libp2p_peer_id != libp2p_id_str
                            });
                            debug!(%peer_id, "evicted expired mDNS peer");
                        }
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

        // ── WS-F F-4: peer-relay SERVER events (only fire when peer-relay mode
        // is on; Toggle::None emits none). Mirrors the bootstrap relay's F-6
        // leech trigger: a circuit that blew its byte budget locks the source
        // peer out so the LeechRateLimiter denies its next reservations.
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::RelayServer(relay_srv_event)) => {
            if let libp2p::relay::Event::CircuitClosed { src_peer_id, error, .. } = &relay_srv_event {
                if error
                    .as_ref()
                    .map(|e| e.to_string().contains(crate::relay::MAX_CIRCUIT_BYTES_ERROR))
                    .unwrap_or(false)
                {
                    if let Some(table) = &state.peer_relay_leech {
                        let now = crate::relay::unix_secs_now();
                        if let Ok(mut t) = table.lock() {
                            let until = t.record_cap_out(*src_peer_id, now, crate::relay::wallclock_jitter_frac());
                            t.prune(now);
                            warn!(%src_peer_id, lockout_until = until, "peer-relay leech: circuit exceeded byte budget — locked out (F-4/F-6)");
                        }
                    }
                }
            }
            debug!(?relay_srv_event, "peer-relay server event");
        }

        // ── DCUtR ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Dcutr(dcutr_event)) => {
            let peer = dcutr_event.remote_peer_id;
            match dcutr_event.result {
                Ok(conn_id) => {
                    state.dcutr_successes += 1;
                    // G6: do NOT synthesize a direct-connection counter here. The upgraded
                    // connection (usually QUIC-v6, the preferred punch) fires its own
                    // `ConnectionEstablished`, which is the *sole* source of truth for
                    // `peer_connections` counts. A phantom `tcp_direct` with no `ConnectionId`
                    // binding was never balanced by the matching `ConnectionClosed` (that closed
                    // connection decrements `quic_direct_*`, not `tcp_direct`), so it stuck at 1
                    // forever — `has_direct()` then lied and the C-N1 relay-close sweep could
                    // tear down the peer's *only* working (relay) path. Removing the push to the
                    // never-drained `dcutr_event_queue` here also closes its unbounded growth.
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
            handle_grpc_proxy_event(proxy_event, swarm, state, proxy_queue);
        }

        // ── C7 Registry query (consumer side) ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::RegistryQuery(reg_event)) => {
            handle_registry_query_event(reg_event, swarm, state);
        }

        // ── Gossipsub (PR-3 / B1) ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Gossipsub(gossip_event)) => {
            if let libp2p::gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            } = gossip_event
            {
                // Phase 4.2: Fast-path eviction for PEER_DEPARTED messages.
                // Parse and evict before queuing for Python to minimize the
                // ghost peer window on clean shutdown.
                //
                // SECURITY (audit F2): only honour a departure announcement
                // when the *signed message author* is the very peer being
                // declared departed. Gossipsub runs with
                // ``MessageAuthenticity::Signed`` + ``ValidationMode::Strict``
                // (see bootstrap_bin.rs), so ``message.source`` is the
                // cryptographically-verified author. Without this check any
                // peer could broadcast forged departures for every honest peer
                // and continuously purge the whole network's routing tables
                // (discovery DoS). A peer may only announce *its own*
                // departure; genuine third-party death is handled reactively
                // by the ConnectionClosed → abort_sessions_for_peer path.
                // R-DHT-1: PROVIDER_ANNOUNCE (gossip PEX) is consumed entirely in
                // Rust — verified and cached into `known_peers` — and deliberately
                // NOT forwarded to Python. Providers re-announce on a timer, so
                // queuing every advert would crowd real PEER_DEAD /
                // REQUEST_HOLE_PUNCH events out of the bounded inbound queue. This
                // flag suppresses the queue push below for fully-handled messages.
                let mut handled_internally = false;
                if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&message.data) {
                    let msg_type = parsed.get("type").and_then(|v| v.as_str());
                    if msg_type == Some(PROVIDER_ANNOUNCE_TYPE) {
                        handled_internally = true;
                        handle_provider_pex(state, &parsed, message.source);
                    } else if msg_type == Some("PEER_DEPARTED") {
                        if let Some(departed_id_str) = parsed.get("libp2p_peer_id").and_then(|v| v.as_str()) {
                            if let Ok(departed_pid) = departed_id_str.parse::<PeerId>() {
                                if message.source == Some(departed_pid) {
                                    swarm.behaviour_mut().kademlia.remove_peer(&departed_pid);
                                    info!(%departed_pid, "evicted departed peer via PEER_DEPARTED gossip");
                                    state.known_peers.retain(|_, record| {
                                        record.libp2p_peer_id != departed_id_str
                                    });
                                } else {
                                    warn!(
                                        ?message.source, %departed_pid,
                                        "rejected PEER_DEPARTED: author does not match subject"
                                    );
                                }
                            }
                        }
                    }
                }

                // Queue the payload for Python to poll. ``propagation_source``
                // is the immediate gossip hop (NOT necessarily the original
                // author) — we surface it so Python can build a 2-observer
                // quorum from distinct hop sources when needed. R-DHT-1:
                // Rust-consumed messages (PROVIDER_ANNOUNCE) skip the queue.
                if !handled_internally {
                    if state.gossip_inbound_queue.len() >= GOSSIP_INBOUND_QUEUE_MAX {
                        state.gossip_inbound_queue.pop_front();
                        warn!("gossipsub_queue_overflow: dropped oldest message");
                    }
                    state
                        .gossip_inbound_queue
                        .push_back((propagation_source.to_string(), message.data));
                }
            }
        }

        // Ping keepalive — log failures and evict unreachable peers
        // (Phase 1.6). Success is silent to avoid flooding logs.
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Ping(ping_event)) => {
            if let Err(ref e) = ping_event.result {
                let peer = ping_event.peer;
                warn!(%peer, error = %e, "ping failed");
                if !swarm.is_connected(&peer) {
                    swarm.behaviour_mut().kademlia.remove_peer(&peer);
                    let libp2p_id_str = peer.to_string();
                    state.known_peers.retain(|_, record| {
                        record.libp2p_peer_id != libp2p_id_str
                    });
                    info!(%peer, "evicted peer after ping failure");
                }
            }
        }

        // ── UPnP (R-DHT-4) ──
        // The mapped-address *confirmation* is handled automatically: the upnp
        // behaviour emits `ToSwarm::ExternalAddrConfirmed`, which the swarm routes
        // to the `SwarmEvent::ExternalAddrConfirmed` handler below (→ R-DHT-2 Kad
        // server promotion). These generated events are informational/diagnostic.
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Upnp(upnp_event)) => {
            match upnp_event {
                libp2p::upnp::Event::NewExternalAddr(addr) => {
                    info!(%addr, "R-DHT-4: UPnP mapped a public address");
                    // A successful UPnP map on a publicly-routable gateway is a
                    // positive reachability signal → explicitly promote (auto-mode
                    // is off; see swarm.rs). The libp2p-upnp behaviour only confirms
                    // on a routable gateway (NonRoutableGateway otherwise), but
                    // re-check global routability as defence in depth.
                    if is_globally_reachable_addr(&addr) {
                        swarm.behaviour_mut().kademlia.set_mode(Some(kad::Mode::Server));
                        state.kad_server_mode = true;
                    }
                    // Remember it so the re-assert ticker can restore the advertised
                    // address if an AutoNAT-Private demotion later retracts it (the
                    // IGD behaviour only confirms once, not on renewal).
                    state.upnp_external_addrs.insert(addr);
                }
                libp2p::upnp::Event::ExpiredExternalAddr(addr) => {
                    info!(%addr, "R-DHT-4: UPnP mapping expired");
                    // The lease genuinely lapsed — stop re-asserting it.
                    state.upnp_external_addrs.remove(&addr);
                }
                libp2p::upnp::Event::GatewayNotFound => {
                    debug!("R-DHT-4: no UPnP/IGD gateway found (expected off home routers)");
                }
                libp2p::upnp::Event::NonRoutableGateway => {
                    debug!("R-DHT-4: UPnP gateway is itself NATed (CGNAT/double-NAT) — no public mapping");
                }
            }
        }

        // ── Connection lifecycle ──
        SwarmEvent::NewListenAddr { address, .. } => {
            info!(%address, "listening on");
            // R-DHT-2 (revised after the 2026-06-15 live test): a globally-routable
            // *listen* address is NOT proof of inbound reachability. The live test
            // showed a home node with a global IPv6 (2406:…) still sits behind a
            // default-deny router firewall — externally unreachable — yet the old
            // "confirm the global listen addr → promote" path made it a black-hole
            // server, and AutoNAT could not reliably demote it (the `Private` verdict
            // for an unreachable node never latched; it timed out into `Unknown`).
            // So we no longer confirm listen addresses as external here. Promotion
            // now requires a positive AutoNAT v2 per-address verdict — the only
            // signal that actually proves an arbitrary querier can reach us — or a
            // UPnP mapping (see the AutonatV2Client / UPnP handlers). "No verdict" → stay
            // client (safe), never a black hole. DCUtR is unaffected: it offers our
            // listen addrs during hole-punch regardless of external confirmation.
            //
            // F-5: a circuit listen addr coming up means the reservation
            // succeeded — clear any pending backoff retry for it.
            if address.to_string().contains("/p2p-circuit") {
                clear_reservation_retry(state, &address.to_string());
                // #42: track live reservations so rebootstrap re-requests only
                // the missing ones (avoids leaking duplicate circuit listeners).
                state.reserved_circuits.insert(address.to_string());
            } else if is_real_interface_addr(&address) {
                // #42: a real interface came up (roam / wake / new link) — arm a
                // debounced rebuild so the routing table, bootstrap connections
                // and relay reservations refresh for the new network.
                arm_reactive_rebootstrap(state, "new listen addr");
            }
        }
        SwarmEvent::ConnectionEstablished { peer_id, connection_id, endpoint, .. } => {
            let addr_str = match &endpoint {
                libp2p::core::ConnectedPoint::Dialer { address, .. } => address.to_string(),
                libp2p::core::ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr.to_string(),
            };
            let transport = classify_transport(&addr_str);
            let endpoint_ip = extract_ip_from_multiaddr_str(&addr_str);
            let is_relay_ip = endpoint_ip
                .as_ref()
                .map(|ip| crate::relay::is_bootstrap_relay_ip(ip))
                .unwrap_or(false);

            // Reclassify: if the IP is a bootstrap relay but no /p2p-circuit/,
            // treat as relay (fixes false direct classification).
            let transport = if is_relay_ip && transport != TransportType::TcpRelay {
                TransportType::TcpRelay
            } else {
                transport
            };

            // Connection reversal: a *direct* connection to a peer we reverse-dialed
            // is a relay→direct upgrade — count it and clear the back-off.
            if transport != TransportType::TcpRelay && state.reversal_pending.remove(&peer_id) {
                state.reversal_successes += 1;
                state.reversal_attempts.remove(&peer_id);
                info!(%peer_id, %addr_str, successes = state.reversal_successes,
                    "connection_reversal_success");
            }

            // F-3: count this established connection under its ladder tier.
            // Done before the peer_connections borrow below to keep the field
            // borrows disjoint. Every establishment counts (incl. AF-dup QUIC
            // that gets closed just after) — the metric measures which rung
            // reaches "established", which is the success signal we want.
            {
                let _tier_is_ipv6 = endpoint_ip.as_ref().map_or(false, |ip| ip.contains(':'));
                *state
                    .tier_connect_success
                    .entry(connection_tier(transport, _tier_is_ipv6))
                    .or_insert(0) += 1;
            }

            let info = state.peer_connections.entry(peer_id).or_default();

            // Fix 2: AF-aware QUIC dedup — if we already have a QUIC-direct
            // connection on the *same* address family, close the new one to
            // prevent 16+ parallel connections.
            // Must increment BEFORE closing: ConnectionClosed will decrement,
            // so the original connection's count stays >= 1.
            let is_ipv6 = endpoint_ip.as_ref().map_or(false, |ip| ip.contains(':'));
            if transport == TransportType::QuicDirect {
                let same_af_count = if is_ipv6 { info.quic_direct_v6 } else { info.quic_direct_v4 };
                if same_af_count >= 1 {
                    if is_ipv6 {
                        info.quic_direct_v6 += 1;
                    } else {
                        info.quic_direct_v4 += 1;
                    }
                    debug!(%peer_id, %addr_str, v4=info.quic_direct_v4, v6=info.quic_direct_v6,
                        "quic_dedup: closing duplicate QUIC connection (same AF)");
                    let _ = swarm.close_connection(connection_id);
                } else {
                    if is_ipv6 {
                        info.quic_direct_v6 += 1;
                        info!(%peer_id, %addr_str, v6=info.quic_direct_v6, "quic_direct_v6_connected");
                    } else {
                        info.quic_direct_v4 += 1;
                        // v6-pref: remember this v4 QUIC id so a later stable v6
                        // direct can retire it precisely (see v4_close_deadline).
                        info.quic_v4_conn_ids.push(connection_id);
                        info!(%peer_id, %addr_str, v4=info.quic_direct_v4, "quic_direct_v4_connected");
                    }
                }
            } else {
                match transport {
                    TransportType::QuicDirect => unreachable!(), // handled above
                    TransportType::TcpDirect => {
                        info.tcp_direct += 1;
                        info!(%peer_id, %addr_str, tcp=info.tcp_direct, "tcp_direct_connected");
                    }
                    TransportType::TcpRelay => {
                        info.tcp_relay += 1;
                        // C-N1: remember this relay connection so a later stable
                        // direct path can close it precisely.
                        info.relay_conn_ids.push(connection_id);
                        debug!(%peer_id, %addr_str, relay=info.tcp_relay, "tcp_relay_connected");
                    }
                }
            }

            // C-N1: if this peer now has BOTH a direct and a relay connection,
            // arm the grace window after which the relay is closed. Only arm once
            // (keep the earliest deadline) so a burst of duplicate connections
            // can't keep pushing the close out. `info` is still borrowed here.
            if should_arm_relay_close(info) {
                state
                    .relay_close_deadline
                    .entry(peer_id)
                    .or_insert_with(|| std::time::Instant::now() + RELAY_CLOSE_GRACE);
            }

            // v6-pref: if this peer now has BOTH a v6 and a v4 QUIC direct, arm the
            // grace window after which the redundant v4 is retired so traffic
            // settles on v6. Same earliest-wins arming as the relay case.
            if should_arm_v4_close(info) {
                state
                    .v4_close_deadline
                    .entry(peer_id)
                    .or_insert_with(|| std::time::Instant::now() + RELAY_CLOSE_GRACE);
            }

            // Send any queued proxy forwards that were waiting for this connection.
            let mut flushed_for_peer = false;
            let mut remaining = Vec::new();
            for (target, data, reply, enqueued_at) in state.pending_relay_forwards.drain(..) {
                if target == peer_id {
                    info!(%peer_id, "sending queued proxy forward after relay connection");
                    flushed_for_peer = true;
                    let req_id = swarm
                        .behaviour_mut()
                        .grpc_proxy
                        .send_request(&peer_id, ProxyRequest(data));
                    state.pending_proxy.insert(req_id, reply);
                } else {
                    remaining.push((target, data, reply, enqueued_at));
                }
            }
            state.pending_relay_forwards = remaining;

            // C-N6: a forward we just flushed rode this connection only because it
            // was the one available. If that was a *relay* connection and we still
            // have no direct path, kick off a DCUtR hole-punch now so the NEXT
            // forward to this peer can take the fast direct path (C-N1 then retires
            // the relay once direct proves stable). A one-shot request_response call
            // already dispatched can't be re-targeted, so this upgrades subsequent
            // traffic, not the in-flight request.
            if flushed_for_peer
                && transport == TransportType::TcpRelay
                && state.peer_connections.get(&peer_id).map_or(true, |i| !i.has_direct())
            {
                handle_trigger_repunch(swarm, peer_id, state);
            }
        }
        SwarmEvent::ConnectionClosed { peer_id, connection_id, endpoint, .. } => {
            let addr_str = match &endpoint {
                libp2p::core::ConnectedPoint::Dialer { address, .. } => address.to_string(),
                libp2p::core::ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr.to_string(),
            };
            let transport = classify_transport(&addr_str);
            let endpoint_ip = extract_ip_from_multiaddr_str(&addr_str);
            let is_relay_ip = endpoint_ip
                .as_ref()
                .map(|ip| crate::relay::is_bootstrap_relay_ip(ip))
                .unwrap_or(false);
            let transport = if is_relay_ip && transport != TransportType::TcpRelay {
                TransportType::TcpRelay
            } else {
                transport
            };

            if let Some(info) = state.peer_connections.get_mut(&peer_id) {
                match transport {
                    TransportType::QuicDirect => {
                        let is_ipv6 = endpoint_ip.as_ref().map_or(false, |ip| ip.contains(':'));
                        if is_ipv6 {
                            info.quic_direct_v6 = info.quic_direct_v6.saturating_sub(1);
                        } else {
                            info.quic_direct_v4 = info.quic_direct_v4.saturating_sub(1);
                            // v6-pref: forget this v4 id (closed on its own or retired
                            // by the grace sweep).
                            info.quic_v4_conn_ids.retain(|&c| c != connection_id);
                        }
                    }
                    TransportType::TcpDirect => info.tcp_direct = info.tcp_direct.saturating_sub(1),
                    TransportType::TcpRelay => {
                        info.tcp_relay = info.tcp_relay.saturating_sub(1);
                        // C-N1: forget this relay id (whether it closed on its own
                        // or because the grace sweep closed it).
                        info.relay_conn_ids.retain(|&c| c != connection_id);
                    }
                }
                // C-N1: if the peer no longer has a direct path, cancel any armed
                // relay-close — we now depend on the relay again.
                if !info.has_direct() {
                    state.relay_close_deadline.remove(&peer_id);
                }
                // v6-pref: if the v6 direct dropped, cancel any armed v4-close — we
                // depend on the v4 again.
                if info.quic_direct_v6 == 0 {
                    state.v4_close_deadline.remove(&peer_id);
                }
                if info.quic_direct_v4 == 0 && info.quic_direct_v6 == 0 && info.tcp_direct == 0 && info.tcp_relay == 0 {
                    state.peer_connections.remove(&peer_id);
                    state.relay_close_deadline.remove(&peer_id);
                    state.v4_close_deadline.remove(&peer_id);
                    debug!(%peer_id, "peer_fully_disconnected");
                }
            }
            // Clean up when fully disconnected: evict from Kademlia routing
            // table, known_peers cache, and tensor stream cache.
            // This is the PRIMARY ghost peer elimination path (Phase 1.1).
            if !swarm.is_connected(&peer_id) {
                state.peer_connections.remove(&peer_id);

                // Ghost-peer purge: Kademlia routing table
                swarm.behaviour_mut().kademlia.remove_peer(&peer_id);
                debug!(%peer_id, "evicted from kademlia routing table");

                // Ghost-peer purge: known_peers cache
                let libp2p_id_str = peer_id.to_string();
                let before = state.known_peers.len();
                state.known_peers.retain(|_openhydra_id, record| {
                    record.libp2p_peer_id != libp2p_id_str
                });
                if state.known_peers.len() < before {
                    info!(%peer_id, "evicted from known_peers cache on disconnect");
                }

                // E-N3: prune the remaining per-peer maps that keyed off this
                // peer. These grew unbounded because the disconnect cleanup
                // evicted peer_connections / known_peers / kademlia but left
                // last_repunch (debounce timestamps) and peer_quic_addrs (learned
                // QUIC addrs) behind — one entry leaked per peer ever seen.
                state.last_repunch.remove(&peer_id);
                state.peer_quic_addrs.remove(&peer_id);
                // H (E-N3 follow-up): the sibling back-off maps keyed per peer were missed by
                // the original E-N3 prune and leaked one entry per peer ever seen on a stable
                // node. Evict them here too so every per-peer map clears on full disconnect.
                state.quic_holepunch_attempts.remove(&peer_id);
                state.reversal_attempts.remove(&peer_id);
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
                    state.nat_info.external_ip = ip.clone();
                    // Classify by address family.
                    if ip.contains(':') {
                        state.nat_info.external_ipv6 = ip;
                    } else {
                        state.nat_info.external_ipv4 = ip;
                    }
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
        // A failed outgoing dial to a peer we hold no live connection to →
        // evict it so routing stops preferring the dead path (re-discovered
        // on the next serve).
        SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
            warn!(?peer_id, %error, "outgoing_connection_error");
            if let Some(pid) = peer_id {
                if !swarm.is_connected(&pid) {
                    evict_unreachable_peer(swarm, state, pid);
                }
            }
        }
        SwarmEvent::IncomingConnectionError { error, .. } => {
            warn!(%error, "incoming_connection_error");
        }
        // B1: Explicit handlers for relay-critical events (previously silent).
        SwarmEvent::ExpiredListenAddr { address, .. } => {
            warn!(%address, "listen_addr_expired");
            if address.to_string().contains("/p2p-circuit") {
                // #42: reservation is no longer live — stop treating it as held.
                state.reserved_circuits.remove(&address.to_string());
                // Normal renewal: try an immediate re-listen first (libp2p
                // usually auto-renews; this covers the case it didn't). If the
                // immediate attempt errors, fall into the F-5 backoff retry so
                // a persistently-failing relay doesn't strand us.
                info!(%address, "relay reservation expired, requesting renewal");
                match swarm.listen_on(address.clone()) {
                    Ok(_) => info!(%address, "relay re-reservation requested"),
                    Err(e) => {
                        warn!(%address, %e, "relay re-reservation failed; scheduling backoff retry");
                        schedule_reservation_retry(state, &address.to_string());
                    }
                }
            } else if is_real_interface_addr(&address) {
                // #42: a real interface address went away (roam / interface
                // down) — expire the stale direct external addrs tied to it and
                // arm a debounced rebuild for the new network.
                expire_direct_external_addrs(swarm, state);
                arm_reactive_rebootstrap(state, "expired listen addr");
            }
        }
        SwarmEvent::ListenerClosed { addresses, reason, .. } => {
            warn!(?addresses, ?reason, "listener_closed");
            let mut real_interface_closed = false;
            for addr in &addresses {
                if addr.to_string().contains("/p2p-circuit") {
                    // #42: reservation dropped — no longer held.
                    state.reserved_circuits.remove(&addr.to_string());
                    // F-5: a closed circuit listener means the reservation
                    // dropped/was rejected. Schedule a backed-off retry instead
                    // of an immediate re-listen — an immediate re-listen against
                    // a flapping or capped relay tight-loops the event loop and
                    // spams the relay. The relay_retry_ticker drives the re-listen
                    // once the backoff window elapses.
                    schedule_reservation_retry(state, &addr.to_string());
                } else if is_real_interface_addr(addr) {
                    real_interface_closed = true;
                }
            }
            if real_interface_closed {
                // #42: a real interface listener closed (roam / wake) — expire
                // stale direct external addrs and arm a debounced rebuild.
                expire_direct_external_addrs(swarm, state);
                arm_reactive_rebootstrap(state, "listener closed");
            }
        }
        SwarmEvent::ListenerError { error, .. } => {
            warn!(%error, "listener_error");
        }
        SwarmEvent::Dialing { peer_id, .. } => {
            debug!(?peer_id, "dialing");
        }
        _ => {
            trace!("unhandled_swarm_event");
        }
    }
}

/// Handle Kademlia events.
fn handle_kad_event(
    event: kad::Event,
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
) {
    match event {
        // Task 2.1: Provider-based discovery (Option C)
        kad::Event::OutboundQueryProgressed {
            id,
            result: kad::QueryResult::GetProviders(result),
            ..
        } => {
            match result {
                Ok(kad::GetProvidersOk::FoundProviders { providers, .. }) => {
                    if let Some(pending) = state.pending_discovers.get_mut(&id) {
                        for provider_pid in providers {
                            let pid_str = provider_pid.to_base58();
                            // 1. Check known_peers cache (lookup by libp2p_peer_id field)
                            let cached = state.known_peers.values()
                                .find(|r| r.libp2p_peer_id == pid_str)
                                .cloned();
                            if let Some(record) = cached {
                                // Deduplicate: don't add if already in results
                                if !pending.records.iter().any(|r| r.libp2p_peer_id == pid_str) {
                                    pending.records.push(record);
                                }
                                continue;
                            }
                            // 2. Check local Kademlia store (populated by put_record replication)
                            let per_peer_key = dht::peer_record_key(&pending.model_id, &pid_str);
                            let stored_value = swarm.behaviour_mut().kademlia.store_mut()
                                .get(&per_peer_key)
                                .map(|cow| cow.into_owned());
                            if let Some(kad_record) = stored_value {
                                if let Ok(record) = dht::decode_record(&kad_record.value) {
                                    // H1: reject unverified records (DHT poisoning).
                                    if let Err(e) = dht::verify_peer_record(&record) {
                                        warn!(%pid_str, %e, "dht_record_rejected: provider-store verify failed");
                                    } else if !pending.records.iter().any(|r| r.libp2p_peer_id == pid_str) {
                                        state.known_peers.insert(known_peer_key(&record), record.clone());
                                        pending.records.push(record);
                                    }
                                }
                            } else {
                                // 3. Cross-relay: the full record was put_record'd to the
                                //    bootstrap nodes, not replicated to us. Chain a
                                //    get_record to fetch it, and defer this discover's reply
                                //    until the fetch lands (GetRecord handler + finalize).
                                let rec_qid =
                                    swarm.behaviour_mut().kademlia.get_record(per_peer_key);
                                state.pending_record_fetches.insert(rec_qid, id);
                                pending.outstanding += 1;
                                debug!(%provider_pid, "provider record not local; fetching via get_record");
                            }
                        }
                    }
                    // Reply the instant the resolved providers are in hand (local hits with no
                    // outstanding fetches) — don't wait for the query's slow convergence tail.
                    maybe_reply_discover(state, id);
                }
                Ok(kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. }) => {
                    // Query converged: mark done so an *empty* result can return (a populated
                    // one already replied early via maybe_reply_discover above).
                    if let Some(pending) = state.pending_discovers.get_mut(&id) {
                        pending.providers_done = true;
                    }
                    maybe_reply_discover(state, id);
                }
                Err(_e) => {
                    // get_providers timed out / failed: mark done so a no-provider result can
                    // resolve. Any providers already resolved were returned early.
                    if let Some(pending) = state.pending_discovers.get_mut(&id) {
                        pending.providers_done = true;
                    }
                    maybe_reply_discover(state, id);
                }
            }
        }
        // Task 2.1: Log start_providing results
        kad::Event::OutboundQueryProgressed {
            result: kad::QueryResult::StartProviding(result),
            ..
        } => {
            match result {
                Ok(kad::AddProviderOk { key }) => {
                    debug!(?key, "start_providing succeeded");
                }
                Err(e) => {
                    warn!("start_providing failed: {e:?}");
                }
            }
        }
        // Backward compat: keep GetRecord handler for records stored before Task 2.1
        kad::Event::OutboundQueryProgressed {
            id,
            result: kad::QueryResult::GetRecord(result),
            ..
        } => {
            // Is this a record fetch chained from a get_providers discover (cross-relay)?
            if let Some(discover_id) = state.pending_record_fetches.get(&id).copied() {
                match result {
                    Ok(kad::GetRecordOk::FoundRecord(kad::PeerRecord { record, .. })) => {
                        ingest_discovered_record(swarm, state, discover_id, &record.value);
                        // Got the provider's record — this fetch is resolved.
                        if state.pending_record_fetches.remove(&id).is_some() {
                            finalize_discover_fetch(state, discover_id);
                        }
                    }
                    Ok(kad::GetRecordOk::FinishedWithNoAdditionalRecord { .. }) | Err(_) => {
                        // Terminal without a record — still resolve the fetch so the
                        // discover can complete (one fewer provider).
                        if state.pending_record_fetches.remove(&id).is_some() {
                            finalize_discover_fetch(state, discover_id);
                        }
                    }
                }
                return;
            }
            match result {
                Ok(kad::GetRecordOk::FoundRecord(kad::PeerRecord { record, .. })) => {
                    // D-S5: compute the trusted relay set before borrowing
                    // `pending_discovers` mutably below (disjoint-borrow dodge).
                    let trusted = trusted_relay_pids(state);
                    // Decode the record and add to pending discover results.
                    if let Some(pending) = state.pending_discovers.get_mut(&id) {
                        match dht::decode_record(&record.value) {
                            Ok(peer_record) => {
                                // H1: reject unverified records BEFORE trusting
                                // any field — otherwise an attacker's record
                                // injects an attacker multiaddr into the routing
                                // table (add_address below) and poisons discovery.
                                if let Err(e) = dht::verify_peer_record(&peer_record) {
                                    warn!(%e, "dht_record_rejected: GetRecord verify failed");
                                    return;
                                }
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
                                // D-S5: only inject a bootstrap-relay circuit
                                // address that names this peer — a signed record
                                // proves authorship, not that the declared address
                                // is honest, so a direct/foreign address here is a
                                // reflection-attack vector and is dropped.
                                if !peer_record.relay_address.is_empty()
                                    && !peer_record.libp2p_peer_id.is_empty()
                                {
                                    match peer_record.libp2p_peer_id.parse::<PeerId>() {
                                        Ok(pid) => {
                                            match crate::relay::safe_injectable_circuit_addr(
                                                &peer_record.relay_address,
                                                &pid,
                                                &trusted,
                                            ) {
                                                Some(ma) => {
                                                    let update = swarm
                                                        .behaviour_mut()
                                                        .kademlia
                                                        .add_address(&pid, ma.clone());
                                                    debug!(
                                                        %pid, %ma, ?update,
                                                        "discover_auto_added_address"
                                                    );
                                                }
                                                None => {
                                                    warn!(
                                                        %pid,
                                                        addr = %peer_record.relay_address,
                                                        "discover: rejected non-injectable relay_address (D-S5)"
                                                    );
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            warn!(
                                                "discover: invalid libp2p_peer_id in record: {e}"
                                            );
                                        }
                                    }
                                }
                                // Cache the peer.
                                state
                                    .known_peers
                                    .insert(known_peer_key(&peer_record), peer_record.clone());
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
                        let peers = discovered_from_records(state, &pending.records);
                        // C11: cache a non-empty result so repeat requests skip get_providers.
                        if !peers.is_empty() {
                            state.discover_cache.insert(
                                pending.model_id.clone(),
                                (now_unix_ms(), state.net_generation.load(std::sync::atomic::Ordering::Relaxed)),
                            );
                        }
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
        // D-S3: with StoreInserts::FilterBoth, inbound PUTs are NOT auto-stored —
        // we must explicitly accept them, which lets us verify signed records
        // before storing/replicating so this node never amplifies poison.
        kad::Event::InboundRequest { request } => {
            handle_inbound_kad_request(request, swarm);
        }
        kad::Event::RoutingUpdated { peer, .. } => {
            debug!(%peer, "kademlia routing updated");
        }
        // Phase 1.2: UnroutablePeer is the strongest signal that a peer
        // should be evicted — Kademlia itself declares it unreachable.
        kad::Event::UnroutablePeer { peer } => {
            warn!(%peer, "kademlia reports peer unroutable, evicting");
            swarm.behaviour_mut().kademlia.remove_peer(&peer);
            let libp2p_id_str = peer.to_string();
            state.known_peers.retain(|_, record| {
                record.libp2p_peer_id != libp2p_id_str
            });
        }
        _ => {
            debug!(?event, "unhandled kademlia event");
        }
    }
}

/// D-S5: the set of relay peer ids this node trusts as circuit hops — the
/// runtime `--bootstrap`/relay peers ∪ the hardcoded `BOOTSTRAP_RELAYS`. A
/// provider's advertised `relay_address` is only injected into the routing table
/// if its relay hop is in this set (so a runtime-configured relay like netcup is
/// accepted while an arbitrary/attacker relay is rejected).
fn trusted_relay_pids(state: &LoopState) -> std::collections::HashSet<PeerId> {
    let mut set: std::collections::HashSet<PeerId> =
        state.bootstrap_peers.iter().map(|(pid, _)| *pid).collect();
    set.extend(crate::relay::bootstrap_relay_peer_ids());
    set
}

/// D-S3: accept-or-drop an inbound Kademlia PUT under `StoreInserts::FilterBoth`.
///
/// With filtering on, libp2p does NOT auto-store inbound records; each arrives
/// here as an `InboundRequest` carrying the record, and we must explicitly write
/// it back to the store to keep it (and let Kad replicate it onward). We verify
/// the signed `PeerRecord` first — a forged/undecodable one is dropped, so this
/// node never stores or re-replicates poison. Provider records carry no
/// signature (the real signed record is fetched + verified on the read side), so
/// they are stored as-is to preserve PEX/provider discovery.
fn handle_inbound_kad_request(
    request: kad::InboundRequest,
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
) {
    match request {
        kad::InboundRequest::PutRecord { source, record: Some(record), .. } => {
            match dht::decode_record(&record.value) {
                Ok(peer_record) => {
                    if let Err(e) = dht::verify_peer_record(&peer_record) {
                        warn!(%source, %e,
                              "kad_put_rejected: inbound record failed verification (poison?)");
                        return;
                    }
                    // H (key-binding): a valid signature proves the record is authentic but NOT
                    // that it is stored under its canonical key. Without this, one valid identity
                    // could replay its own signed record under thousands of distinct keys to fill
                    // the bounded MemoryStore and block legitimate stores. Require the storage key
                    // to be the one derived from the record's own (model_id, libp2p_peer_id).
                    let expected_key =
                        dht::peer_record_key(&peer_record.model_id, &peer_record.libp2p_peer_id);
                    if record.key != expected_key {
                        warn!(%source,
                              "kad_put_rejected: record key not bound to its (model_id, libp2p_peer_id)");
                        return;
                    }
                    if let Err(e) = swarm.behaviour_mut().kademlia.store_mut().put(record) {
                        debug!(%source, ?e, "kad_put: store rejected verified record (capacity?)");
                    }
                }
                Err(e) => {
                    warn!(%source, %e, "kad_put_rejected: undecodable inbound record");
                }
            }
        }
        // Filtered but empty (shouldn't happen with FilterBoth) — nothing to store.
        kad::InboundRequest::PutRecord { record: None, .. } => {}
        kad::InboundRequest::AddProvider { record: Some(provider) } => {
            // No signature to verify here; the read side fetches + verifies the
            // provider's actual signed record before trusting it.
            if let Err(e) = swarm.behaviour_mut().kademlia.store_mut().add_provider(provider) {
                debug!(?e, "kad_add_provider: store rejected (capacity?)");
            }
        }
        kad::InboundRequest::AddProvider { record: None } => {}
        // Read-side inbound requests (FindNode/GetProvider/GetRecord) need no action.
        _ => {}
    }
}

/// Handle AutoNAT events.
/// Decode + verify a fetched `PeerRecord`, make it dialable (add its advertised
/// relay address to the routing table so a later `proxy_forward` can reach a NAT'd
/// provider), cache it, and append it to the named pending discover (deduped).
fn ingest_discovered_record(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    discover_id: kad::QueryId,
    value: &[u8],
) {
    let peer_record = match dht::decode_record(value) {
        Ok(r) => r,
        Err(e) => {
            warn!("failed to decode fetched DHT record: {e}");
            return;
        }
    };
    if !accept_and_install_record(swarm, state, &peer_record) {
        return;
    }
    let pid_str = peer_record.libp2p_peer_id.clone();
    if let Some(pending) = state.pending_discovers.get_mut(&discover_id) {
        if !pending.records.iter().any(|r| r.libp2p_peer_id == pid_str) {
            pending.records.push(peer_record);
        }
    }
}

/// Verify a freshly-learned provider record and make it usable, returning whether it was
/// accepted. Shared by the DHT record-fetch path ([`ingest_discovered_record`]) and the C7
/// registry-query path so every newly-learned record goes through the identical trust +
/// dialability steps — there is exactly one place that decides "is this record safe to use?":
///
/// * H1 — reject records that fail `verify_peer_record` before trusting any field (DHT/registry
///   poisoning: an unverified record's multiaddr must never reach the routing table).
/// * D-S5 — install the peer's relay-circuit address into Kademlia only if it is safely
///   injectable (names this peer, via a trusted relay), so a signed-but-dishonest address can't
///   seed the routing table with a victim host. Without an installed address `proxy_forward`
///   fails with "no addresses for peer".
/// * Cache the verified record in `known_peers` (feeds the C11 seed / passive discovery).
fn accept_and_install_record(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    peer_record: &PeerRecord,
) -> bool {
    if let Err(e) = dht::verify_peer_record(peer_record) {
        warn!(%e, "record_rejected: verify failed");
        return false;
    }
    if !peer_record.relay_address.is_empty() && !peer_record.libp2p_peer_id.is_empty() {
        if let Ok(pid) = peer_record.libp2p_peer_id.parse::<PeerId>() {
            let trusted = trusted_relay_pids(state);
            match crate::relay::safe_injectable_circuit_addr(&peer_record.relay_address, &pid, &trusted) {
                Some(ma) => {
                    let update = swarm.behaviour_mut().kademlia.add_address(&pid, ma.clone());
                    debug!(%pid, %ma, ?update, "discover_auto_added_address");
                }
                None => {
                    warn!(%pid, addr = %peer_record.relay_address,
                          "record: rejected non-injectable relay_address (D-S5)");
                }
            }
        }
    }
    state.known_peers.insert(known_peer_key(&peer_record), peer_record.clone());
    true
}

/// C7 command handler: ask a connected bootstrap "who serves `model_id`?". Picks the first
/// bootstrap we currently have a live connection to (bootstraps run the Inbound registry
/// responder and the query rides that existing connection), sends the request, and parks the
/// reply until the response/failure event lands. If no bootstrap is connected there is nothing
/// to ask — reply with an empty `Vec` (best-effort: the caller falls back to whatever passive
/// discovery already returned), not an error.
fn handle_query_registry(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    model_id: &str,
    reply: oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>,
    state: &mut LoopState,
) {
    let target = state
        .bootstrap_peers
        .iter()
        .map(|(pid, _)| *pid)
        .find(|pid| swarm.is_connected(pid));
    let Some(pid) = target else {
        debug!(model_id, "registry: no connected bootstrap to query — returning empty");
        let _ = reply.send(Ok(Vec::new()));
        return;
    };
    let req_id = swarm
        .behaviour_mut()
        .registry_query
        .send_request(&pid, RegistryQuery { model_id: model_id.to_string() });
    state.pending_registry.insert(req_id, reply);
    debug!(%pid, model_id, "registry: querying bootstrap");
}

/// C7 event handler (consumer side): route registry request/response events. Responses carry the
/// bootstrap's self-signed provider records — each is re-verified and made dialable via
/// [`accept_and_install_record`] (the bootstrap is not a trust anchor) before being surfaced.
fn handle_registry_query_event(
    event: request_response::Event<RegistryQuery, RegistryReply>,
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
) {
    match event {
        request_response::Event::Message { peer, message } => match message {
            // This node advertises Outbound support only, so it should never receive an inbound
            // query. If a peer sends one anyway, answer empty rather than leave the stream hung.
            request_response::Message::Request { request, channel, .. } => {
                debug!(%peer, model = %request.model_id, "registry: unexpected inbound query — answering empty");
                let _ = swarm
                    .behaviour_mut()
                    .registry_query
                    .send_response(channel, RegistryReply::default());
            }
            request_response::Message::Response { request_id, response } => {
                if let Some(reply) = state.pending_registry.remove(&request_id) {
                    let mut verified: Vec<PeerRecord> = Vec::new();
                    for rec in response.records {
                        if accept_and_install_record(swarm, state, &rec) {
                            verified.push(rec);
                        }
                    }
                    debug!(%peer, returned = verified.len(), "registry: query answered");
                    let peers = discovered_from_records(state, &verified);
                    let _ = reply.send(Ok(peers));
                }
            }
        },
        request_response::Event::OutboundFailure { peer, request_id, error, .. } => {
            if let Some(reply) = state.pending_registry.remove(&request_id) {
                let _ = reply.send(Err(format!("registry query to {peer} failed: {error}")));
            }
        }
        request_response::Event::InboundFailure { .. }
        | request_response::Event::ResponseSent { .. } => {}
    }
}

/// Reply to a pending discover the moment its result is settled, **without waiting for the
/// get_providers query to converge** (that tail is a fixed ~10s timeout that otherwise
/// dominates per-request latency). "Settled" = no record fetches still outstanding AND
/// either at least one provider record resolved (fast path) or the query has fully finished
/// (so an empty result can return). Ranking / empty handling is the consumer's job.
fn maybe_reply_discover(state: &mut LoopState, discover_id: kad::QueryId) {
    let ready = match state.pending_discovers.get(&discover_id) {
        Some(p) => p.outstanding == 0 && (!p.records.is_empty() || p.providers_done),
        None => return,
    };
    if ready {
        let pending = state.pending_discovers.remove(&discover_id).expect("present");
        // R-DHT-8: `discovered_from_records` stamps the liveness hint (live libp2p connection?)
        // so the consumer prefers connected providers and failover is the exception. All
        // records here are already H1/PEX-verified, so we never surface an unverified candidate.
        let peers = discovered_from_records(state, &pending.records);
        // C11: remember this successful discover so the next request for the same model can
        // answer from `known_peers` without a fresh get_providers round-trip. Only cache a
        // non-empty result — an empty one has nothing to serve and should re-query.
        if !peers.is_empty() {
            state.discover_cache.insert(
                pending.model_id.clone(),
                (now_unix_ms(), state.net_generation.load(std::sync::atomic::Ordering::Relaxed)),
            );
        }
        let _ = pending.reply.send(Ok(peers));
    }
}

/// One chained record fetch resolved: drop the discover's outstanding count and try to
/// reply (it will the instant the last fetch lands — see [`maybe_reply_discover`]).
fn finalize_discover_fetch(state: &mut LoopState, discover_id: kad::QueryId) {
    if let Some(pending) = state.pending_discovers.get_mut(&discover_id) {
        if pending.outstanding > 0 {
            pending.outstanding -= 1;
        }
    }
    maybe_reply_discover(state, discover_id);
}

/// C7-flicker fix: the `known_peers` cache key. Keyed by **`(peer_id, model_id)`**, not `peer_id`
/// alone — a single provider node announces one record PER model it serves (all sharing its node
/// `peer_id`), so keying on `peer_id` made each model clobber the previous one and a multi-model
/// provider could only ever surface ONE model at a time (it flickered between them). The unit
/// separator can't appear in a hex peer id or a model handle, so the composite is unambiguous.
fn known_peer_key(record: &PeerRecord) -> String {
    format!("{}\u{1f}{}", record.peer_id, record.model_id)
}

/// Convert a PeerRecord into a DiscoveredPeer.
fn record_to_discovered(r: &PeerRecord) -> DiscoveredPeer {
    // Prefer IPv4 host, fall back to IPv6 if empty.
    let effective_host = if r.host.is_empty() && !r.host_ipv6.is_empty() {
        r.host_ipv6.clone()
    } else {
        r.host.clone()
    };
    let reachable_address = if r.requires_relay && !r.relay_address.is_empty() {
        r.relay_address.clone()
    } else {
        format!("{}:{}", effective_host, r.port)
    };
    DiscoveredPeer {
        peer_id: r.peer_id.clone(),
        libp2p_peer_id: r.libp2p_peer_id.clone(),
        host: effective_host,
        host_ipv6: r.host_ipv6.clone(),
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
        // protocol.md §4 (M1.2) — carry the capability fields through discover().
        canonical_model_id: r.canonical_model_id.clone(),
        context_length: r.context_length,
        max_output_tokens: r.max_output_tokens,
        throughput_tok_s: r.throughput_tok_s,
        queue_depth: r.queue_depth,
        hardware_class: r.hardware_class.clone(),
        load_pct: r.load_pct,
        reputation_score: r.reputation_score,
        // M2.1 — carry the provider's ed25519 public key (hex) so the consumer can
        // address a co-signed receipt at its identity after a whole-model route.
        public_key: r.public_key.clone(),
        reachable_address,
        // R-DHT-8: liveness is stamped by `maybe_reply_discover` (which has the
        // live connection table); default false here.
        connected: false,
    }
}


/// R-DHT-2 (server-mode promotion gate): is this multiaddr a *globally reachable*
/// direct address — one we may safely confirm as external and thereby promote
/// ourselves into a Kademlia **server**?
///
/// libp2p-kad 0.46 runs in automatic mode: the instant the swarm has ANY
/// confirmed external address, Kademlia flips to `Mode::Server` (stores records,
/// answers queries). Confirming a non-routable address therefore advertises us as
/// a server reachable at an address no remote node can actually dial — queries
/// routed to us **black-hole**, degrading the DHT for everyone (the exact failure
/// §2 of the remediation plan forbids).
///
/// So we confirm ONLY addresses in a globally-routable IP range:
/// * IPv4 — reject private (RFC1918), CGNAT (100.64/10), loopback, link-local,
///   unspecified, broadcast, documentation. A NAT'd peer's LAN listen addr
///   (`192.168.x`, `10.x`) is rejected here — it must NOT promote us to server.
/// * IPv6 — reject loopback, unspecified, link-local (fe80::/10), unique-local
///   (fc00::/7 ULA), documentation (2001:db8::/32). A global IPv6 is accepted —
///   it is normally un-NATed and the underrated near-term reachability win (R-DHT-3).
/// * Any `/p2p-circuit` address — reject. A relay forwarding for us is not us
///   being reachable.
///
/// This gate decides *eligibility* only. The actual promotion is still driven by
/// confirmation events: a global listen/observed address plus an AutoNAT `Public`
/// verdict (full/restricted-cone NAT and public hosts reach `Public`; symmetric
/// NAT / CGNAT reach `Private` and stay clients — §2).
/// Whether `addr` is a directly globally-routable address (rejects circuit, private,
/// CGNAT, loopback, link-local, ULA, and documentation ranges). Shared with the bootstrap
/// binary, which uses it to confirm its public *listen* addresses as external (a bootstrap
/// is unambiguously public, so unlike a peer it may trust its listen addrs — see
/// `bootstrap_bin`'s `NewListenAddr` handler).
/// Tier-2 connection-reversal candidates: from a peer's advertised `listen_addrs`,
/// the globally-routable, locally-dialable direct addresses to reverse-dial — minus
/// any already handled elsewhere (`already_dialed`, e.g. the QUIC-v6 hole-punch set).
/// Drops `/p2p-circuit` explicitly, plus (via `is_globally_reachable_addr`) private,
/// CGNAT, loopback, link-local and ULA ranges; and IPv6 on a v6-incapable host.
fn reversal_candidate_addrs(
    listen_addrs: &[Multiaddr],
    ipv6_capable: bool,
    already_dialed: &[Multiaddr],
) -> Vec<Multiaddr> {
    listen_addrs
        .iter()
        .filter(|a| !already_dialed.contains(a))
        .filter(|a| !a.to_string().contains("p2p-circuit"))
        .filter(|a| ipv6_capable || !is_ipv6_multiaddr(a))
        .filter(|a| is_globally_reachable_addr(a))
        .cloned()
        .collect()
}

#[cfg(test)]
mod reversal_tests {
    use super::*;

    fn ma(s: &str) -> Multiaddr {
        s.parse().unwrap()
    }

    #[test]
    fn reversal_candidates_keep_only_routable_dialable_new_addrs() {
        let listen = vec![
            ma("/ip4/1.1.1.1/udp/4001/quic-v1"),            // public v4 → keep
            ma("/ip4/192.168.1.5/tcp/4001"),                // private → drop
            ma("/ip4/100.64.0.9/udp/4001/quic-v1"),         // CGNAT → drop
            ma("/ip6/2606:4700:4700::1111/udp/4001/quic-v1"), // public v6
            ma("/ip4/45.79.190.172/tcp/4001/p2p-circuit"),  // relay circuit → drop
        ];

        // v6-capable host, nothing pre-dialed → keep both public addrs.
        let got = reversal_candidate_addrs(&listen, true, &[]);
        assert!(got.contains(&ma("/ip4/1.1.1.1/udp/4001/quic-v1")));
        assert!(got.contains(&ma("/ip6/2606:4700:4700::1111/udp/4001/quic-v1")));
        assert_eq!(got.len(), 2);

        // v6-incapable host drops the v6 addr.
        let got4 = reversal_candidate_addrs(&listen, false, &[]);
        assert_eq!(got4, vec![ma("/ip4/1.1.1.1/udp/4001/quic-v1")]);

        // Already-dialed addrs are excluded (no double-dial vs the QUIC-v6 path).
        let pre = vec![ma("/ip6/2606:4700:4700::1111/udp/4001/quic-v1")];
        let got_excl = reversal_candidate_addrs(&listen, true, &pre);
        assert_eq!(got_excl, vec![ma("/ip4/1.1.1.1/udp/4001/quic-v1")]);
    }
}

pub fn is_globally_reachable_addr(addr: &Multiaddr) -> bool {
    if addr.to_string().contains("/p2p-circuit") {
        return false;
    }
    let mut saw_ip = false;
    for proto in addr.iter() {
        match proto {
            libp2p::multiaddr::Protocol::Ip4(ip) => {
                saw_ip = true;
                // 100.64.0.0/10 — carrier-grade NAT (RFC6598). Not in std's
                // is_private(), but just as un-routable from the public internet.
                let o = ip.octets();
                let is_cgnat = o[0] == 100 && (o[1] & 0xc0) == 64;
                if ip.is_private()
                    || ip.is_loopback()
                    || ip.is_link_local()
                    || ip.is_unspecified()
                    || ip.is_broadcast()
                    || ip.is_documentation()
                    || is_cgnat
                {
                    return false;
                }
            }
            libp2p::multiaddr::Protocol::Ip6(ip) => {
                saw_ip = true;
                let seg = ip.segments();
                let is_link_local = (seg[0] & 0xffc0) == 0xfe80; // fe80::/10
                let is_unique_local = (seg[0] & 0xfe00) == 0xfc00; // fc00::/7
                let is_documentation = seg[0] == 0x2001 && seg[1] == 0x0db8; // 2001:db8::/32
                if ip.is_loopback()
                    || ip.is_unspecified()
                    || is_link_local
                    || is_unique_local
                    || is_documentation
                {
                    return false;
                }
            }
            _ => {}
        }
    }
    saw_ip
}

/// R-DHT-4: decide which UPnP-mapped external addresses to re-assert this tick.
///
/// Returns the tracked UPnP addresses that have dropped out of the swarm's
/// `confirmed` external set (so re-adding them is meaningful, not a redundant
/// re-confirmation) — **but an empty list whenever AutoNAT currently holds a
/// `Private` verdict**. A genuinely-broken port map keeps AutoNAT at `Private`,
/// so suppressing the re-assert there is what stops a black hole from being
/// re-promoted; once AutoNAT clears (Public/Unknown) the address is restored.
fn upnp_addrs_to_reassert(
    upnp: &std::collections::HashSet<Multiaddr>,
    confirmed: &std::collections::HashSet<Multiaddr>,
    autonat_private: bool,
) -> Vec<Multiaddr> {
    if autonat_private {
        return Vec::new();
    }
    upnp.iter()
        .filter(|a| !confirmed.contains(*a))
        .cloned()
        .collect()
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
    proxy_queue: &SharedProxyQueue,
) {
    match event {
        request_response::Event::Message { peer, message } => {
            match message {
                request_response::Message::Request { request_id: _, request, channel } => {
                    // Phase 2.4: per-peer inflight cap. Count this peer's live
                    // (unanswered) channels; if it already holds the cap, shed the
                    // request by dropping `channel` (the peer's request times out).
                    // PROXY_QUEUE_MAX bounds total memory; this bounds per-peer
                    // fairness so one peer can't monopolise the queue. Count is
                    // derived from the channel map (no separate counter to leak).
                    // handle_swarm_event ends after this match, so the early
                    // return cleanly finishes handling this rejected event.
                    let peer_inflight = state
                        .inbound_proxy_channels
                        .values()
                        .filter(|(_, _, p)| *p == peer)
                        .count();
                    if peer_inflight >= MAX_INFLIGHT_PER_PEER {
                        warn!(%peer, inflight = peer_inflight, cap = MAX_INFLIGHT_PER_PEER,
                              "per-peer inflight cap exceeded — shedding proxy request");
                        drop(channel);
                        return;
                    }
                    // Every inbound proxy request is a serve-protocol message (SERVE_REQUEST /
                    // SERVE_STREAM / FETCH_* / RECEIPT_REQUEST) or another agent-layer method.
                    // Route it to the shared proxy queue -> InboundProxyItem -> the agent's
                    // poll_inbound, which decodes the 1-byte method and replies. The agent's serve
                    // decode is length-guarded / Err-not-panic, so raw untrusted bodies are handled
                    // safely at that layer (the pre-pivot dispatcher/forward_msg parsers are gone).
                    //
                    // F4 (audit): the removed dispatcher used to answer two method bytes inline —
                    // METHOD_PING (0x05) and METHOD_GET_STATUS (0x06). Those now fall through to
                    // the agent, whose dispatch returns a framed "unsupported method" Error. No
                    // in-tree peer sends them (libp2p's own ping handles liveness); if a future
                    // need arises, add the inline responder in the agent's dispatch `_` arm.
                    state.inbound_proxy_counter += 1;
                    let req_id = format!("proxy-{}", state.inbound_proxy_counter);
                    // F6b: trace (not info — this fires for EVERY inbound request) for debugging.
                    trace!(%peer, id = %req_id, bytes = request.0.len(), "proxy request queued for the agent");
                    proxy_queue.push((req_id.clone(), peer.to_string(), request.0));
                    state.inbound_proxy_channels.insert(req_id, (channel, std::time::Instant::now(), peer.clone()));
                }
                request_response::Message::Response { request_id, response } => {
                    if let Some(reply) = state.pending_proxy.remove(&request_id) {
                        // Outbound response received — deliver to waiting proxy forward.
                        // #zombie: a successful round-trip proves the current path is
                        // live — clear any accumulated failure streak for this peer.
                        record_proxy_success(&mut state.proxy_failure_streak, &peer);
                        let _ = reply.send(Ok(response.0));
                    }
                }
            }
        }
        request_response::Event::OutboundFailure { peer, request_id, error, .. } => {
            warn!(%peer, ?error, "proxy outbound failure");
            if let Some(reply) = state.pending_proxy.remove(&request_id) {
                let _ = reply.send(Err(format!("proxy outbound: {error:?}")));
            }
            // #zombie (#42 live-roam finding): `send_request` picks ANY existing
            // connection, so zombie pre-roam connections mask a roamed peer's
            // fresh path until they reap (observed: 6+ min of timeouts). Classify
            // path failures vs protocol failures; after ZOMBIE_FAILURE_THRESHOLD
            // consecutive path failures, force-close every connection to the
            // peer and redial fresh relay circuits so the NEXT dispatch rides a
            // live path (the roamed peer's new reservation is reachable — its
            // re-announced record was never the problem, our stale conns were).
            let dead_path = matches!(
                error,
                request_response::OutboundFailure::Timeout
                    | request_response::OutboundFailure::ConnectionClosed
                    | request_response::OutboundFailure::DialFailure
                    | request_response::OutboundFailure::Io(_)
            );
            if record_proxy_failure(&mut state.proxy_failure_streak, peer, dead_path) {
                warn!(%peer, threshold = ZOMBIE_FAILURE_THRESHOLD,
                    "zombie_evict: consecutive proxy path failures — closing presumed-dead connections and redialing via relay");
                let _ = swarm.disconnect_peer_id(peer);
                let circuit_addrs = relay_circuit_addrs(peer, state.ipv6_capable);
                if let Err(e) = swarm.dial(relay_dial_opts(peer, circuit_addrs)) {
                    debug!(%peer, %e, "zombie_evict: relay redial enqueue failed");
                }
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
    proxy_queue: &SharedProxyQueue,
) {
    let peer_id: PeerId = match peer_id_str.parse() {
        Ok(p) => p,
        Err(e) => {
            let _ = reply.send(Err(format!("invalid peer_id: {e}")));
            return;
        }
    };

    if peer_id == *swarm.local_peer_id() {
        info!("proxy_forward: target is self — routing locally");
        state.inbound_proxy_counter += 1;
        let req_id = format!("proxy-local-{}", state.inbound_proxy_counter);
        let self_id = swarm.local_peer_id().to_string();
        proxy_queue.push((req_id.clone(), self_id, data));
        state.local_proxy_replies.insert(req_id, reply);
        return;
    }

    // Blocking proxy_forward always uses request-response: it's proven
    // reliable over both direct and relay connections. Tensor stream is
    // only used in the fire-and-forget path (proxy_forward_no_wait / push mode).
    if swarm.is_connected(&peer_id) {
        // Truthful dispatch log. `send_request` dispatches via NotifyHandler::Any,
        // so when a peer has BOTH a direct and a relay connection the chosen path
        // is arbitrary — the old single "direct"/"relay" label (derived from
        // has_direct()) hid that ambiguity and could claim "direct" while the
        // request actually rode the relay. Log the real connection composition
        // instead; "ambiguous(direct+relay)" flags exactly the case where the
        // (still-unfixed) connection-selection bug bites.
        let (direct_conns, relay_conns) = state
            .peer_connections
            .get(&peer_id)
            .map(|i| (i.direct_count(), i.tcp_relay))
            .unwrap_or((0, 0));
        let path = if direct_conns > 0 && relay_conns > 0 {
            "ambiguous(direct+relay)"
        } else if direct_conns > 0 {
            "direct"
        } else if relay_conns > 0 {
            "relay"
        } else {
            "unknown"
        };
        // #zombie: surface the live failure streak so ops can see a suspect
        // path before the eviction threshold trips.
        let failure_streak = state.proxy_failure_streak.get(&peer_id).copied().unwrap_or(0);
        info!(%peer_id, path, direct_conns, relay_conns, failure_streak, bytes = data.len(), "proxy_forward dispatch");
        let req_id = swarm
            .behaviour_mut()
            .grpc_proxy
            .send_request(&peer_id, ProxyRequest(data));
        state.pending_proxy.insert(req_id, reply);
    } else {
        info!(%peer_id, "proxy_forward: peer not connected, dialing via relay");
        // Finding A: dial the peer ONCE via every bootstrap-relay circuit as a
        // multi-address DialOpts, instead of looping and breaking on the first
        // swarm.dial()==Ok. swarm.dial() returns Ok the instant the dial is
        // *enqueued*, not when it connects — so the old break-on-first always
        // picked BOOTSTRAP_RELAYS[0] (US) regardless of which relay the target
        // had actually reserved on, and never fell through to EU/AP. A single
        // multi-addr dial lets libp2p race all circuits and connect via
        // whichever one holds the target's reservation.
        // C-N2: shed if the queue is already at its cap — a peer we can never
        // dial must not grow it without bound.
        if state.pending_relay_forwards.len() >= MAX_PENDING_RELAY_FORWARDS {
            warn!(%peer_id, cap = MAX_PENDING_RELAY_FORWARDS,
                  "proxy_forward: pending_relay_forwards at cap — shedding");
            let _ = reply.send(Err("proxy_forward: relay queue full".into()));
            return;
        }
        let circuit_addrs = relay_circuit_addrs(peer_id, state.ipv6_capable);
        match swarm.dial(relay_dial_opts(peer_id, circuit_addrs)) {
            Ok(()) => {
                state
                    .pending_relay_forwards
                    .push((peer_id, data, reply, std::time::Instant::now()));
            }
            Err(e) => {
                warn!(%peer_id, error=%e, "proxy_forward: relay dial failed");
                let _ = reply.send(Err("proxy_forward: relay dial failed".into()));
            }
        }
    }
}

/// True if a multiaddr contains an `/ip6/` component (F-9 gating).
pub(crate) fn is_ipv6_multiaddr(addr: &Multiaddr) -> bool {
    addr.iter().any(|p| matches!(p, libp2p::multiaddr::Protocol::Ip6(_)))
}

/// F-9: one-shot probe for working outbound IPv6. Binds the v6 wildcard and
/// `connect`s a UDP socket to a global v6 address — this sets the default
/// destination WITHOUT sending any packet, and the kernel returns
/// `ENETUNREACH`/`EHOSTUNREACH` immediately when there is no usable v6 route
/// (no-v6 hosts AND partial-v6 hosts where the address is assigned but
/// unroutable). Traffic-free, dependency-free, ~instant.
pub(crate) fn probe_ipv6_capable() -> bool {
    use std::net::UdpSocket;
    // 2606:4700:4700::1111 = Cloudflare DNS (stable global v6). Port is
    // irrelevant for a connectionless UDP connect.
    let ok = UdpSocket::bind("[::]:0")
        .and_then(|s| s.connect("[2606:4700:4700::1111]:53"))
        .is_ok();
    info!(ipv6_capable = ok, "F-9: outbound IPv6 capability probe");
    ok
}

/// Finding A helper: build a `/<relay>/p2p-circuit/p2p/<target>` multiaddr for
/// every known bootstrap relay, so a single dial can race all of them. F-9:
/// when the host has no working IPv6, drop the `/ip6/` relay circuits — dialing
/// them would only burn timeouts on unreachable addresses.
fn relay_circuit_addrs(peer_id: PeerId, ipv6_capable: bool) -> Vec<Multiaddr> {
    crate::relay::BOOTSTRAP_RELAYS
        .iter()
        .filter_map(|s| s.parse::<Multiaddr>().ok())
        .filter(|relay| ipv6_capable || !is_ipv6_multiaddr(relay))
        .map(|relay| {
            relay
                .with(libp2p::multiaddr::Protocol::P2pCircuit)
                .with(libp2p::multiaddr::Protocol::P2p(peer_id))
        })
        .collect()
}

/// Finding A helper: a single multi-address dial to `peer_id` over the given
/// relay-circuit addresses. `PeerCondition::Disconnected` so we only dial when
/// there is no established connection (we're in the not-connected branch).
fn relay_dial_opts(
    peer_id: PeerId,
    addrs: Vec<Multiaddr>,
) -> libp2p::swarm::dial_opts::DialOpts {
    use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
    DialOpts::peer_id(peer_id)
        .condition(PeerCondition::Disconnected)
        .addresses(addrs)
        .build()
}

/// Fire-and-forget variant of handle_proxy_forward.
///
/// Sends data to a peer via request_response but does NOT store a reply
/// channel in `pending_proxy`. When the response arrives, the
/// `pending_proxy.remove()` call in the response handler returns `None`
/// and the response is silently discarded — exactly the desired behaviour
/// for fire-and-forget cross-ISP push mode.
///
/// For Ouroboros (self-targeted) forwards: queues in `proxy_queue`
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
    proxy_queue: &SharedProxyQueue,
) {
    let peer_id: PeerId = match peer_id_str.parse() {
        Ok(p) => p,
        Err(e) => {
            warn!("proxy_forward_no_wait: invalid peer_id: {e}");
            return;
        }
    };

    if peer_id == *swarm.local_peer_id() {
        debug!("proxy_forward_no_wait: target is self — routing locally (no reply)");
        state.inbound_proxy_counter += 1;
        let req_id = format!("proxy-local-{}", state.inbound_proxy_counter);
        let self_id = swarm.local_peer_id().to_string();
        proxy_queue.push((req_id, self_id, data));
        return;
    }

    // C3: Log transport type at dispatch time.
    let transport = state.peer_connections.get(&peer_id)
        .map(|info| if info.has_direct() { "direct" } else { "relay" })
        .unwrap_or("unknown");

    // Fire-and-forget over request_response.
    if swarm.is_connected(&peer_id) {
        debug!(%peer_id, %transport, bytes = data.len(), "proxy_forward_no_wait via request_response");
        let _req_id = swarm
            .behaviour_mut()
            .grpc_proxy
            .send_request(&peer_id, ProxyRequest(data));
    } else {
        info!(%peer_id, "proxy_forward_no_wait: peer not connected, dialing via relay");
        // Finding A: one multi-address dial across all relay circuits (see
        // handle_proxy_forward) instead of break-on-first-queued (always US).
        // C-N2: same cap guard as the blocking path (fire-and-forget, so just
        // drop rather than reply on overflow).
        if state.pending_relay_forwards.len() >= MAX_PENDING_RELAY_FORWARDS {
            warn!(%peer_id, cap = MAX_PENDING_RELAY_FORWARDS,
                  "proxy_forward_no_wait: pending_relay_forwards at cap — dropping");
            return;
        }
        match swarm.dial(relay_dial_opts(peer_id, relay_circuit_addrs(peer_id, state.ipv6_capable))) {
            Ok(()) => {
                let (dummy_tx, _dummy_rx) = oneshot::channel();
                state
                    .pending_relay_forwards
                    .push((peer_id, data, dummy_tx, std::time::Instant::now()));
            }
            Err(e) => {
                warn!(%peer_id, error=%e, "proxy_forward_no_wait: relay dial failed — data dropped");
            }
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

// ── Fix 4: TriggerRepunch with debounce ────────────────────────────────

fn handle_trigger_repunch(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    peer_id: PeerId,
    state: &mut LoopState,
) {
    if let Some(last) = state.last_repunch.get(&peer_id) {
        if last.elapsed() < REPUNCH_DEBOUNCE {
            debug!(%peer_id, "repunch_debounced: skipping (last was {}s ago)",
                last.elapsed().as_secs());
            return;
        }
    }
    state.last_repunch.insert(peer_id, std::time::Instant::now());

    let addrs = match state.peer_quic_addrs.get(&peer_id) {
        Some(addrs) if !addrs.is_empty() => addrs.clone(),
        _ => {
            debug!(%peer_id, "repunch: no QUIC IPv6 addresses cached for peer");
            return;
        }
    };

    info!(%peer_id, count = addrs.len(), "repunch: re-dialing QUIC IPv6 addresses");
    for addr in addrs {
        use libp2p::swarm::dial_opts::DialOpts;
        let ma = addr.clone();
        match swarm.dial(DialOpts::unknown_peer_id().address(addr).build()) {
            Ok(()) => info!(%ma, "repunch_dial_issued"),
            Err(e) => debug!(%ma, %e, "repunch_dial_failed"),
        }
    }
}

// ── CP-3: Ring event handling + re-injection ─────────────────────────




// ── CP-4: Batch dispatch ──────────────────────────────────────────────



// ── Phase 2: Tunnel stubs ─────────────────────────────────────────────

fn handle_open_tunnel(
    _peer_id: &str,
    reply: oneshot::Sender<Result<String, String>>,
    _state: &mut LoopState,
) {
    let _ = reply.send(Err("tunnel not yet implemented — use proxy_forward".into()));
}

fn handle_close_tunnel(
    _peer_id: &str,
    _state: &mut LoopState,
) {
    debug!("close_tunnel: not yet implemented");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c11_discover_cache_fresh_policy() {
        let ttl = DISCOVER_CACHE_TTL_MS;
        let mut cache: HashMap<String, (u64, u64)> = HashMap::new();
        // model "m" discovered at t=1_000_000 under net-generation 7.
        cache.insert("m".into(), (1_000_000, 7));

        // Within TTL, same generation → fresh (early-reply eligible).
        assert!(discover_cache_fresh(&cache, "m", 1_000_000 + ttl - 1, 7, ttl));
        assert!(discover_cache_fresh(&cache, "m", 1_000_000, 7, ttl));

        // Past TTL → stale (re-query).
        assert!(!discover_cache_fresh(&cache, "m", 1_000_000 + ttl, 7, ttl));
        assert!(!discover_cache_fresh(&cache, "m", 2_000_000, 7, ttl));

        // Net-generation changed (roam/wake) → invalidated even if within TTL.
        assert!(!discover_cache_fresh(&cache, "m", 1_000_000 + 1, 8, ttl));

        // Unknown model → not fresh.
        assert!(!discover_cache_fresh(&cache, "other", 1_000_000, 7, ttl));

        // Clock skew (now < stored) must not underflow into a false "fresh".
        assert!(discover_cache_fresh(&cache, "m", 999_999, 7, ttl)); // saturating_sub → 0 < ttl
    }

    #[test]
    fn c11_prune_discover_cache_bounds_the_map() {
        let ttl = DISCOVER_CACHE_TTL_MS;
        let now = 1_000_000u64;
        let mut cache: HashMap<String, (u64, u64)> = HashMap::new();
        cache.insert("fresh".into(), (now - 1, 5)); // within TTL, current gen → keep
        cache.insert("stale".into(), (now - ttl - 1, 5)); // past TTL → drop
        cache.insert("oldgen".into(), (now - 1, 4)); // superseded net-generation → drop
        prune_discover_cache(&mut cache, now, 5, ttl);
        let mut keys: Vec<String> = cache.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, vec!["fresh".to_string()]);
    }

    #[test]
    fn known_peer_key_keeps_every_model_of_a_multi_model_provider() {
        // The flicker bug: a provider node serving N models announces N records that all share
        // its node `peer_id`. Keyed by peer_id alone they clobbered each other (one model shown,
        // flickering). Keyed by (peer_id, model_id) all coexist.
        let rec = |model: &str| PeerRecord {
            peer_id: "node-A".into(),
            model_id: model.into(),
            libp2p_peer_id: "12D3KooWnodeA".into(),
            ..Default::default()
        };
        let mut known: HashMap<String, PeerRecord> = HashMap::new();
        known.insert(known_peer_key(&rec("qwen3.5-4b-mlx")), rec("qwen3.5-4b-mlx"));
        known.insert(known_peer_key(&rec("nomic-embed")), rec("nomic-embed"));
        // Both survive — no clobber.
        assert_eq!(known.len(), 2);
        let mut models: Vec<String> = known.values().map(|r| r.model_id.clone()).collect();
        models.sort();
        assert_eq!(models, vec!["nomic-embed".to_string(), "qwen3.5-4b-mlx".to_string()]);
        // Re-announcing the same (peer, model) refreshes in place, not a third row.
        known.insert(known_peer_key(&rec("nomic-embed")), rec("nomic-embed"));
        assert_eq!(known.len(), 2);
        // A value-based evict (how disconnect/reap work) drops ALL of the node's models at once.
        known.retain(|_, r| r.libp2p_peer_id != "12D3KooWnodeA");
        assert!(known.is_empty());
    }

    #[test]
    fn test_globally_reachable_addr_gate() {
        let g = |s: &str| is_globally_reachable_addr(&s.parse::<Multiaddr>().unwrap());

        // Globally routable — eligible to promote us to a Kad server.
        assert!(g("/ip4/45.79.190.172/tcp/4001"));
        assert!(g("/ip6/2a03:4000:41:ed1::1/tcp/4001"));
        assert!(g("/ip4/8.8.8.8/udp/4001/quic-v1"));

        // Private / LAN (RFC1918) — a NAT'd peer's listen addr must NOT promote.
        assert!(!g("/ip4/192.168.1.5/tcp/4001"));
        assert!(!g("/ip4/10.0.0.9/tcp/4001"));
        assert!(!g("/ip4/172.16.4.4/tcp/4001"));
        // CGNAT (100.64/10) and link-local / loopback / unspecified.
        assert!(!g("/ip4/100.64.1.1/tcp/4001"));
        assert!(!g("/ip4/169.254.1.1/tcp/4001"));
        assert!(!g("/ip4/127.0.0.1/tcp/4001"));
        assert!(!g("/ip4/0.0.0.0/tcp/4001"));

        // IPv6 non-global: ULA (fc00::/7), link-local (fe80::/10), loopback, doc.
        assert!(!g("/ip6/fd00::1/tcp/4001"));
        assert!(!g("/ip6/fe80::1/tcp/4001"));
        assert!(!g("/ip6/::1/tcp/4001"));
        assert!(!g("/ip6/2001:db8::1/tcp/4001"));

        // A relay-circuit address is never "us being reachable".
        assert!(!g("/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit"));
    }

    #[test]
    fn test_is_real_interface_addr() {
        let r = |s: &str| is_real_interface_addr(&s.parse::<Multiaddr>().unwrap());

        // Real interface addrs (LAN or global, v4 or v6) — their up/down IS a
        // network change and should arm a rebootstrap.
        assert!(r("/ip4/192.168.1.5/tcp/4001"));
        assert!(r("/ip4/45.79.190.172/udp/4001/quic-v1"));
        assert!(r("/ip6/2406:7400:56:7e7::e4c6/tcp/4001"));
        assert!(r("/ip4/10.0.0.9/tcp/4001"));

        // Loopback churns on nothing meaningful — must NOT trigger a rebuild.
        assert!(!r("/ip4/127.0.0.1/tcp/4001"));
        assert!(!r("/ip6/::1/udp/4001/quic-v1"));

        // Relay-circuit reservations are the F-5 path's job, not interface events.
        assert!(!r(
            "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit"
        ));
    }

    #[test]
    fn test_zombie_failure_gating() {
        let mut streaks: HashMap<PeerId, u32> = HashMap::new();
        let p = PeerId::random();

        // A protocol-level failure (peer alive, e.g. UnsupportedProtocols)
        // never counts toward eviction and leaves no streak behind.
        assert!(!record_proxy_failure(&mut streaks, p, false));
        assert!(streaks.is_empty());

        // First dead-path failure: suspect but tolerated (could be one slow
        // generation). Second consecutive: evict — and the streak resets so a
        // re-established peer starts clean.
        assert!(!record_proxy_failure(&mut streaks, p, true));
        assert_eq!(streaks.get(&p), Some(&1));
        assert!(record_proxy_failure(&mut streaks, p, true));
        assert!(streaks.is_empty());

        // A successful round-trip clears an in-progress streak: the next
        // failure starts over at 1 instead of tripping the threshold.
        assert!(!record_proxy_failure(&mut streaks, p, true));
        record_proxy_success(&mut streaks, &p);
        assert!(!record_proxy_failure(&mut streaks, p, true));
        assert_eq!(streaks.get(&p), Some(&1));

        // Streaks are per-peer: q's failures don't advance p's streak.
        let q = PeerId::random();
        assert!(!record_proxy_failure(&mut streaks, q, true));
        assert!(record_proxy_failure(&mut streaks, p, true)); // p reaches 2
        assert_eq!(streaks.get(&q), Some(&1)); // q untouched at 1

        // A protocol failure mid-streak doesn't advance OR reset the count.
        assert!(!record_proxy_failure(&mut streaks, q, false));
        assert_eq!(streaks.get(&q), Some(&1));
    }

    #[test]
    fn test_upnp_reassert_decision() {
        let a: Multiaddr = "/ip4/45.79.190.172/tcp/4001".parse().unwrap();
        let b: Multiaddr = "/ip6/2a03:4000:41:ed1::1/tcp/4001".parse().unwrap();
        let upnp: std::collections::HashSet<Multiaddr> = [a.clone(), b.clone()].into_iter().collect();

        // AutoNAT not Private, both addrs already confirmed → nothing to re-assert.
        let confirmed: std::collections::HashSet<Multiaddr> = [a.clone(), b.clone()].into_iter().collect();
        assert!(upnp_addrs_to_reassert(&upnp, &confirmed, false).is_empty());

        // AutoNAT not Private, `a` dropped from the confirmed set → re-assert just `a`.
        let confirmed: std::collections::HashSet<Multiaddr> = [b.clone()].into_iter().collect();
        assert_eq!(upnp_addrs_to_reassert(&upnp, &confirmed, false), vec![a.clone()]);

        // AutoNAT Private → suppress entirely, even though `a` is missing (a broken
        // map keeps AutoNAT Private; must not be re-promoted into a black hole).
        assert!(upnp_addrs_to_reassert(&upnp, &confirmed, true).is_empty());

        // No tracked UPnP addresses → nothing to do.
        assert!(upnp_addrs_to_reassert(&std::collections::HashSet::new(), &confirmed, false).is_empty());
    }

    #[test]
    fn test_is_ipv6_multiaddr() {
        // F-9: detect /ip6/ in plain and relay-circuit multiaddrs.
        let v4: Multiaddr = "/ip4/1.2.3.4/tcp/4001".parse().unwrap();
        let v6: Multiaddr = "/ip6/2001:db8::1/tcp/4001".parse().unwrap();
        let v4_circuit: Multiaddr =
            "/ip4/1.2.3.4/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit"
                .parse().unwrap();
        let v6_circuit: Multiaddr =
            "/ip6/2001:db8::1/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit"
                .parse().unwrap();
        assert!(!is_ipv6_multiaddr(&v4));
        assert!(is_ipv6_multiaddr(&v6));
        assert!(!is_ipv6_multiaddr(&v4_circuit));
        assert!(is_ipv6_multiaddr(&v6_circuit));
    }

    #[test]
    fn test_relay_circuit_addrs_v6_gating() {
        // F-9: BOOTSTRAP_RELAYS has both v4 and v6 entries per relay. With
        // ipv6_capable=false the result must contain ZERO /ip6/ circuits;
        // with =true it must contain at least one (proves we didn't drop them
        // unconditionally). Every entry must be a p2p-circuit to the target.
        let target: PeerId =
            "12D3KooWRL36gMob4EcmXGb1wWd1HnzihPd8KBpiNdxF59kDUhCN".parse().unwrap();

        let no_v6 = relay_circuit_addrs(target, false);
        assert!(!no_v6.is_empty(), "should still have v4 relay circuits");
        assert!(
            no_v6.iter().all(|a| !is_ipv6_multiaddr(a)),
            "ipv6_capable=false must drop all /ip6/ circuits",
        );
        assert!(
            no_v6.iter().all(|a| a.iter()
                .any(|p| matches!(p, libp2p::multiaddr::Protocol::P2pCircuit))),
            "every entry must be a p2p-circuit",
        );

        let with_v6 = relay_circuit_addrs(target, true);
        assert!(with_v6.len() > no_v6.len(), "v6-capable yields more circuits");
        assert!(
            with_v6.iter().any(|a| is_ipv6_multiaddr(a)),
            "ipv6_capable=true must keep /ip6/ circuits",
        );
    }

    #[test]
    fn test_connection_tier_classification() {
        // F-3: (transport, is_ipv6) → ladder rung name; every result must be a
        // known tier so the metrics dict keys are stable.
        assert_eq!(connection_tier(TransportType::QuicDirect, true), "direct_quic_v6");
        assert_eq!(connection_tier(TransportType::QuicDirect, false), "direct_quic_v4");
        assert_eq!(connection_tier(TransportType::TcpDirect, true), "direct_tcp_v6");
        assert_eq!(connection_tier(TransportType::TcpDirect, false), "direct_tcp_v4");
        // Relay is AF-agnostic (always reported as "relay").
        assert_eq!(connection_tier(TransportType::TcpRelay, true), "relay");
        assert_eq!(connection_tier(TransportType::TcpRelay, false), "relay");
        // All produced names are in the canonical CONNECTION_TIERS set.
        for t in [TransportType::QuicDirect, TransportType::TcpDirect, TransportType::TcpRelay] {
            for v6 in [true, false] {
                assert!(CONNECTION_TIERS.contains(&connection_tier(t, v6)),
                    "tier {:?}/{} not in CONNECTION_TIERS", t, v6);
            }
        }
    }

    #[test]
    fn test_reservation_retry_delay_backoff_schedule() {
        // F-5: fast initial retries then exponential backoff capped at 120s.
        assert_eq!(reservation_retry_delay_ms(1), 1_000);   // 1s
        assert_eq!(reservation_retry_delay_ms(2), 2_000);   // 2s
        assert_eq!(reservation_retry_delay_ms(3), 4_000);   // 4s
        assert_eq!(reservation_retry_delay_ms(4), 8_000);   // 8s
        assert_eq!(reservation_retry_delay_ms(5), 16_000);  // 16s
        assert_eq!(reservation_retry_delay_ms(6), 32_000);  // 32s
        assert_eq!(reservation_retry_delay_ms(7), 64_000);  // 64s
        // Capped at 120s from attempt 8 onward — and never overflows for
        // large/pathological attempt counts.
        assert_eq!(reservation_retry_delay_ms(8), 120_000);
        assert_eq!(reservation_retry_delay_ms(50), 120_000);
        assert_eq!(reservation_retry_delay_ms(u32::MAX), 120_000);
        // Monotonic non-decreasing.
        let mut prev = 0u64;
        for a in 1..=20u32 {
            let d = reservation_retry_delay_ms(a);
            assert!(d >= prev, "backoff must be non-decreasing at attempt {a}");
            prev = d;
        }
    }

    #[test]
    fn test_shared_proxy_queue_caps_at_max() {
        // Audit 2.4: pushing past PROXY_QUEUE_MAX drops oldest, never grows
        // unbounded.
        let q = SharedProxyQueue::new();
        for i in 0..(PROXY_QUEUE_MAX + 100) {
            q.push((format!("req-{i}"), "12D3KooWsrc".to_string(), vec![0u8; 4]));
        }
        let len = q.queue.lock().unwrap().len();
        assert_eq!(len, PROXY_QUEUE_MAX, "queue must be capped at PROXY_QUEUE_MAX");
        // Oldest dropped: the first surviving item is req-100, not req-0.
        let front = q.queue.lock().unwrap().front().unwrap().0.clone();
        assert_eq!(front, "req-100");
    }

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
        assert!(crate::relay::is_bootstrap_relay_ip("45.79.190.172"));
        assert!(crate::relay::is_bootstrap_relay_ip("172.105.69.49"));
        assert!(crate::relay::is_bootstrap_relay_ip("172.104.164.98"));
        assert!(!crate::relay::is_bootstrap_relay_ip("192.168.1.11"));
        assert!(!crate::relay::is_bootstrap_relay_ip("10.192.11.51"));
    }

    // ── Fix 2 tests ──

    #[test]
    fn test_classify_transport_quic_direct() {
        assert_eq!(
            classify_transport("/ip6/2409:40f4:1e:b425::/udp/4001/quic-v1"),
            TransportType::QuicDirect,
        );
    }

    #[test]
    fn test_classify_transport_quic_v4() {
        assert_eq!(
            classify_transport("/ip4/192.168.1.10/udp/4001/quic-v1"),
            TransportType::QuicDirect,
        );
    }

    #[test]
    fn test_classify_transport_tcp_direct() {
        assert_eq!(
            classify_transport("/ip4/192.168.1.10/tcp/4001"),
            TransportType::TcpDirect,
        );
    }

    #[test]
    fn test_classify_transport_tcp_relay_circuit() {
        assert_eq!(
            classify_transport("/ip4/45.79.190.172/tcp/4001/p2p/12D3KooW.../p2p-circuit/p2p/12D3KooW..."),
            TransportType::TcpRelay,
        );
    }

    #[test]
    fn test_classify_transport_relay_ip_without_circuit() {
        // Without /p2p-circuit/ in the string, even relay IP classifies as TcpDirect.
        // The reclassification in ConnectionEstablished handles this case.
        assert_eq!(
            classify_transport("/ip4/45.79.190.172/tcp/4001"),
            TransportType::TcpDirect,
        );
    }

    #[test]
    fn test_classify_transport_bare_p2p_is_not_direct() {
        // Live-caught 2026-07-12: a relayed inbound connection's listener-side
        // send_back_addr is a bare `/p2p/<peer>` (no transport, no circuit). It
        // must NOT be counted as direct (that inflated provider direct_conns and
        // could make C-N1 close a real relay). No `/tcp/` or `/udp/` → relay.
        assert_eq!(
            classify_transport("/p2p/12D3KooWSkRDF79TqQ476KaisamEZqgTkR1W6rKyDta7jCjwXZcZ"),
            TransportType::TcpRelay,
        );
        // A genuine QUIC direct (has /udp/ + /quic-v1) still classifies correctly.
        assert_eq!(
            classify_transport("/ip4/85.209.48.209/udp/4101/quic-v1/p2p/12D3KooWSkRD"),
            TransportType::QuicDirect,
        );
    }

    #[test]
    fn test_peer_connection_info() {
        let mut info = PeerConnectionInfo::default();
        assert!(!info.has_direct());
        info.quic_direct_v4 = 1;
        assert!(info.has_direct());
        info.quic_direct_v4 = 0;
        info.quic_direct_v6 = 1;
        assert!(info.has_direct());
        info.quic_direct_v6 = 0;
        info.tcp_direct = 1;
        assert!(info.has_direct());
        info.tcp_direct = 0;
        info.tcp_relay = 1;
        assert!(!info.has_direct());
    }

    // ── C-N1: grace-delayed relay close ────────────────────────────────────

    #[test]
    fn v6_upgrade_pursued_until_a_v6_quic_direct_exists() {
        // v6-first: keep pursuing v6 even when a v4 QUIC direct already exists.
        let mut info = PeerConnectionInfo::default();
        assert!(should_pursue_v6_upgrade(None), "no tracked conn → pursue");
        assert!(should_pursue_v6_upgrade(Some(&info)), "no direct yet → pursue");
        info.quic_direct_v4 = 1;
        assert!(
            should_pursue_v6_upgrade(Some(&info)),
            "a v4 QUIC direct must NOT stop the v6 upgrade (the fix)"
        );
        info.tcp_direct = 1;
        assert!(should_pursue_v6_upgrade(Some(&info)), "a tcp direct doesn't count either");
        info.quic_direct_v6 = 1;
        assert!(
            !should_pursue_v6_upgrade(Some(&info)),
            "once we hold a v6 QUIC direct we're done — don't churn"
        );
    }

    #[test]
    fn relay_close_arms_only_when_both_direct_and_relay_present() {
        use libp2p::swarm::ConnectionId;
        let mut info = PeerConnectionInfo::default();
        // Relay only → don't arm (we have no faster path to fall back from).
        info.tcp_relay = 1;
        info.relay_conn_ids.push(ConnectionId::new_unchecked(1));
        assert!(!should_arm_relay_close(&info));
        // Direct arrives → now redundant relay, arm the close.
        info.quic_direct_v4 = 1;
        assert!(should_arm_relay_close(&info));
        // Direct only (no relay tracked) → nothing to close.
        info.tcp_relay = 0;
        info.relay_conn_ids.clear();
        assert!(!should_arm_relay_close(&info));
    }

    #[test]
    fn relays_to_close_returns_ids_only_while_direct_survives() {
        use libp2p::swarm::ConnectionId;
        let mut info = PeerConnectionInfo::default();
        info.quic_direct_v4 = 1;
        info.tcp_relay = 2;
        info.relay_conn_ids = vec![ConnectionId::new_unchecked(7), ConnectionId::new_unchecked(9)];
        // Direct up → both relay ids are eligible for closing.
        assert_eq!(relays_to_close(&info), info.relay_conn_ids);
        // Direct dropped in the window → keep the relay, close nothing.
        info.quic_direct_v4 = 0;
        assert!(relays_to_close(&info).is_empty());
    }

    // ── v6-pref (C-N1 follow-on): retire v4 QUIC once v6 stabilizes ────────

    #[test]
    fn v4_close_arms_only_when_both_v6_and_v4_quic_present() {
        use libp2p::swarm::ConnectionId;
        let mut info = PeerConnectionInfo::default();
        // v4 only → don't arm (no v6 to prefer).
        info.quic_direct_v4 = 1;
        info.quic_v4_conn_ids.push(ConnectionId::new_unchecked(3));
        assert!(!should_arm_v4_close(&info));
        // v6 arrives alongside v4 → arm (v4 is now redundant).
        info.quic_direct_v6 = 1;
        assert!(should_arm_v4_close(&info));
        // v6 only (no v4 tracked) → nothing to retire.
        info.quic_direct_v4 = 0;
        info.quic_v4_conn_ids.clear();
        assert!(!should_arm_v4_close(&info));
    }

    #[test]
    fn v4s_to_close_returns_ids_only_while_v6_survives() {
        use libp2p::swarm::ConnectionId;
        let mut info = PeerConnectionInfo::default();
        info.quic_direct_v6 = 1;
        info.quic_direct_v4 = 2;
        info.quic_v4_conn_ids = vec![ConnectionId::new_unchecked(4), ConnectionId::new_unchecked(6)];
        // v6 up → both v4 ids eligible for retirement.
        assert_eq!(v4s_to_close(&info), info.quic_v4_conn_ids);
        // v6 dropped in the window → keep the v4, retire nothing.
        info.quic_direct_v6 = 0;
        assert!(v4s_to_close(&info).is_empty());
    }

    // ── C-N2: pending_relay_forwards TTL sweep ─────────────────────────────

    #[test]
    fn relay_forward_sweep_expires_old_and_fails_reply_keeps_fresh() {
        let now = std::time::Instant::now();
        let old_at = now - std::time::Duration::from_secs(60);
        let pid = PeerId::random();
        let (tx_old, mut rx_old) = oneshot::channel::<Result<Vec<u8>, String>>();
        let (tx_new, mut rx_new) = oneshot::channel::<Result<Vec<u8>, String>>();
        let forwards = vec![
            (pid, vec![1, 2, 3], tx_old, old_at),
            (pid, vec![4, 5, 6], tx_new, now),
        ];
        let kept = sweep_relay_forwards(forwards, std::time::Duration::from_secs(30), now);
        // Only the fresh entry survives.
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].1, vec![4, 5, 6]);
        // The expired entry's caller got an error (unblocked, not left hanging).
        assert!(matches!(rx_old.try_recv(), Ok(Err(_))));
        // The fresh entry's channel is still open (no reply sent yet).
        assert!(matches!(rx_new.try_recv(), Err(oneshot::error::TryRecvError::Empty)));
    }

    #[test]
    fn relay_forward_sweep_keeps_all_when_none_expired() {
        let now = std::time::Instant::now();
        let pid = PeerId::random();
        let (tx, _rx) = oneshot::channel::<Result<Vec<u8>, String>>();
        let forwards = vec![(pid, vec![9], tx, now)];
        let kept = sweep_relay_forwards(forwards, std::time::Duration::from_secs(30), now);
        assert_eq!(kept.len(), 1);
    }
}
