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
    /// The remote IP of each *direct* connection to this peer, as we observed it (not what the peer
    /// advertises): which network it really sits on, for choosing independent audit partners.
    direct_ips: HashMap<libp2p::swarm::ConnectionId, String>,
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

/// The error a queued forward fails with when its peer's last dial failed (callers match on it to
/// retry after a short wait: a provider that just joined becomes reachable within seconds).
pub const DIAL_FAILED_UNREACHABLE: &str = "dial failed (peer unreachable)";
/// A dial that failed only because the relays refused it for their limits
/// ([`crate::transport::dial_refused_by_relay_limits`]). Deliberately not [`DIAL_FAILED_UNREACHABLE`],
/// so the consumer doesn't retry it within seconds and spend more of the relays' budget.
pub const DIAL_FAILED_RELAY_LIMIT: &str =
    "dial failed (the relays are refusing this gateway's dials for now: too many recent attempts)";

/// How long relayed dials to a peer wait after one ended unreachable: doubling from 2 s to 16 s, and
/// at least 20 s after the relays refused one for their limits. GPU test 2026-10-29: a consumer
/// retried a dead provider ~10 times in 30 s, each time through all 4 relays, and so used up its
/// circuit allowance there (30 per 2 min per source peer, libp2p's default) — the provider was
/// unreachable for another minute after it came back. A forward to a peer in its back-off fails at
/// once with the same error, so the caller's retry schedule is unchanged but costs no relay budget.
fn next_relay_dial_backoff(prev: Option<std::time::Duration>, relay_limit: bool) -> std::time::Duration {
    use std::time::Duration;
    let next = prev.map_or(Duration::from_secs(2), |d| (d * 2).min(Duration::from_secs(16)));
    if relay_limit { next.max(Duration::from_secs(20)) } else { next }
}

/// A peer's relayed-dial back-off: no relayed dial before `until`; `step` is the last wait;
/// `set_unix_ms` is when it was set (a record the peer signed later means it's back); `why` is the
/// error forwards get meanwhile.
#[derive(Debug, Clone, Copy)]
struct RelayDialBackoff {
    until: std::time::Instant,
    step: std::time::Duration,
    set_unix_ms: u64,
    why: &'static str,
}

/// How often a bootstrap may be re-dialled from a new port after an address-in-use refusal.
const IN_USE_REDIAL_EVERY: std::time::Duration = std::time::Duration::from_secs(30);

/// Peers in relayed-dial back-off at most (a back-off is dropped when it expires, on the reaper tick).
const MAX_RELAY_DIAL_BACKOFFS: usize = 4096;

/// Dials in flight per peer: see [`LoopState::dials_in_flight`].
type DialsInFlight = HashMap<PeerId, std::collections::HashSet<libp2p::swarm::ConnectionId>>;

/// `Swarm::dial`, noting the dial as in flight for its peer. libp2p reports `SwarmEvent::Dialing`
/// only for dials a behaviour starts (Kademlia, relay, AutoNAT), never for ours, so without this
/// a dial of ours (a queued forward's own) would be invisible, and another dial to the same peer
/// failing would fail the forward while its own dial was still running (review 2026-10-14, M4).
trait DialTracked {
    fn dial_tracked(&mut self, dials: &mut DialsInFlight, opts: impl Into<libp2p::swarm::dial_opts::DialOpts>) -> Result<(), libp2p::swarm::DialError>;
}

impl<B: libp2p::swarm::NetworkBehaviour> DialTracked for libp2p::Swarm<B> {
    fn dial_tracked(&mut self, dials: &mut DialsInFlight, opts: impl Into<libp2p::swarm::dial_opts::DialOpts>) -> Result<(), libp2p::swarm::DialError> {
        let opts = opts.into();
        let (peer, conn) = (opts.get_peer_id(), opts.connection_id());
        // A dial refused here (`Err`) ends without a `SwarmEvent`: its caller handles it.
        self.dial(opts)?;
        if let Some(peer) = peer {
            dials.entry(peer).or_default().insert(conn);
        }
        Ok(())
    }
}

/// A dial to `peer` (connection `conn`) just ended without a connection: forget it, and say whether
/// that leaves the peer with no dial still in flight and no connection — then nothing can carry the
/// forwards queued for it, and they fail now rather than wait out [`RELAY_FORWARD_TTL`] (a dial
/// without a relay reservation or known address fails in milliseconds; the caller then retries with
/// a fresh lookup). Another dial still pending (a Kademlia query dialing the same peer, say) can
/// still connect, so then they wait.
fn dial_ended_unreachable(
    dials: &mut DialsInFlight,
    peer: PeerId,
    conn: libp2p::swarm::ConnectionId,
    connected: bool,
) -> bool {
    let none_left = match dials.get_mut(&peer) {
        Some(set) => {
            set.remove(&conn);
            set.is_empty()
        }
        None => true,
    };
    if none_left {
        dials.remove(&peer);
    }
    none_left && !connected
}

/// Fail the forwards queued for `peer` (its dial failed and nothing else can reach it); keep the rest.
fn fail_relay_forwards_for(forwards: Vec<PendingRelayForward>, peer: PeerId, why: &str) -> (Vec<PendingRelayForward>, usize) {
    let mut kept = Vec::with_capacity(forwards.len());
    let mut failed = 0;
    for (pid, data, reply, enqueued_at) in forwards {
        if pid == peer {
            let _ = reply.send(Err(format!("proxy_forward: {why}")));
            failed += 1;
        } else {
            kept.push((pid, data, reply, enqueued_at));
        }
    }
    (kept, failed)
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

/// How long an *inbound* duplicate QUIC connection lives before the dedup closes it. An AutoNAT v2
/// server tests our address by dialing it back on a fresh port — a second QUIC connection from a
/// peer we already hold one to. Closing it on arrival kills the test, so a node whose only
/// connection to the server is QUIC never gets an address confirmed: it stays a Kademlia client,
/// the bootstrap keeps it out of its routing table, and nobody learns a direct address for it
/// (gateways fall back to relay circuits). The dial-back takes a few round trips; this leaves room.
const QUIC_INBOUND_DUP_GRACE: std::time::Duration = std::time::Duration::from_secs(10);

/// What the QUIC dedup does with a new QUIC-direct connection to a peer that already has one on the
/// same address family: a duplicate we dialed closes at once; one the peer dialed (`is_listener`)
/// gets [`QUIC_INBOUND_DUP_GRACE`] first, since it may be an AutoNAT dial-back.
#[derive(Debug, PartialEq, Eq)]
enum QuicDupAction {
    CloseNow,
    CloseAfterGrace,
}

fn quic_dup_action(is_listener: bool) -> QuicDupAction {
    if is_listener {
        QuicDupAction::CloseAfterGrace
    } else {
        QuicDupAction::CloseNow
    }
}

/// The deferred QUIC duplicates whose grace ran out (`due`: connection, peer, IPv6?): which to close,
/// and which stay as their peer's QUIC connection on that family. One is still redundant while the
/// peer holds another QUIC connection on that address family; if the original closed in the meantime
/// the "duplicate" is the only one, and stays.
/// Counted together: two duplicates of one peer due in the same sweep, after its original closed,
/// would each still see two connections, and both would be closed, leaving none (review 2026-10-14,
/// L11): each one closed here counts as gone for the next.
///
/// When both ends dial each other at once (a hole punch from each side), each has its own outbound
/// connection and the other's inbound one, and each would close the inbound one after the grace,
/// so both go. Of such a pair the one the lower peer id dialled stays: the side with the higher id
/// (`local` above the peer) leaves its inbound duplicate alone when it's one of just two, and the
/// other side, closing its own inbound duplicate, closes ours (A4 review 2026-10-29).
fn quic_dups_due(
    due: &[(libp2p::swarm::ConnectionId, PeerId, bool)],
    peers: &HashMap<PeerId, PeerConnectionInfo>,
    local: &PeerId,
) -> (Vec<libp2p::swarm::ConnectionId>, Vec<(libp2p::swarm::ConnectionId, PeerId, bool)>) {
    let mut closing: HashMap<(PeerId, bool), u32> = HashMap::new();
    let (mut close, mut stay) = (Vec::new(), Vec::new());
    for &(cid, pid, v6) in due {
        let Some(info) = peers.get(&pid) else { continue };
        let open = if v6 { info.quic_direct_v6 } else { info.quic_direct_v4 };
        let already = closing.entry((pid, v6)).or_default();
        let left = open.saturating_sub(*already);
        if left >= 3 || (left == 2 && *local < pid) {
            *already += 1;
            close.push(cid);
        } else {
            stay.push((cid, pid, v6));
        }
    }
    (close, stay)
}

/// C-N2: max age of a queued relay-forward before the reaper fails it and drops
/// the reply channel + buffered bytes (bounds the leak when a relay dial never
/// connects). Tied to the reaper tick, so effective sweep is TTL..TTL+reaper.
const RELAY_FORWARD_TTL: std::time::Duration = std::time::Duration::from_secs(30);
/// C-N2: hard cap on queued relay-forwards so a peer we can never dial can't
/// grow the queue without bound.
const MAX_PENDING_RELAY_FORWARDS: usize = 512;

/// Called once with whether a response reached its peer (see [`SwarmCommand::RespondProxy`]). Runs
/// on the event loop, so it must be quick and must not wait on the swarm.
pub type OnSent = Box<dyn FnOnce(bool) + Send>;

/// Responses sent but not yet confirmed that a caller asked to be told about, at most, and per peer:
/// one peer that reads its responses slowly can't take every place (review round 4 X4-13).
const MAX_PENDING_SENT: usize = 4096;
const MAX_PENDING_SENT_PER_PEER: usize = 256;

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
    /// M6: publish a `MODEL_WITHDRAWN` tombstone for one model this node has stopped
    /// serving (the operator un-shared it) so consumers evict just that `(peer, model)`
    /// entry immediately rather than waiting out the discover-record TTL. Best-effort:
    /// a publish failure (no mesh peers yet) is non-fatal — the model simply expires on
    /// the old TTL path, exactly as before this message existed.
    WithdrawModel {
        model_id: String,
        reply: oneshot::Sender<Result<(), String>>,
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
    /// Models whose last known provider withdrew them (`MODEL_WITHDRAWN`) within
    /// [`WITHDRAWN_MODEL_MEMORY`] and that no known provider serves now: lists that smooth over
    /// churn (the gateway's sticky `/v1/models`) drop these at once instead of keeping them.
    RecentlyWithdrawn {
        reply: oneshot::Sender<Vec<String>>,
    },
    /// The model ids one provider (by libp2p id) is known to serve: from its own signed records.
    ModelsOf {
        libp2p_peer_id: String,
        reply: oneshot::Sender<Vec<String>>,
    },
    /// Every (model, provider) pair in `known_peers`, for the gateway's `openhydra/auto` choice.
    KnownProviders {
        reply: oneshot::Sender<Vec<KnownProvider>>,
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
        /// Told whether the response reached the peer (libp2p `ResponseSent`), or didn't (an
        /// inbound failure, an expired or unknown request). A local reply counts as reached.
        on_sent: Option<OnSent>,
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
    /// Dial `peer_id` on these addresses (and only these: no relay circuit) whenever it isn't
    /// connected, and never back off from it ([`crate::pinned`]). Replaces any earlier pin of the peer; an empty
    /// list unpins it. The addresses must already be checked ([`crate::pinned::self_provider_addr`]).
    PinPeerAddrs {
        peer_id: PeerId,
        addrs: Vec<Multiaddr>,
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
    /// The agent accepted a serve request from `peer_id` (every gate passed): dial its claimed IPv6
    /// addresses back (A4), which is what opens our own router's firewall to it.
    ServeAccepted {
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
    /// Models withdrawn by their last known provider, and when (see `RecentlyWithdrawn`).
    withdrawn_models: HashMap<String, std::time::Instant>,
    /// `PROVIDER_GONE` reports gathering per provider: the bootstraps that sent one, the newest
    /// record time they named, and when the first came (unix ms).
    provider_gone: HashMap<PeerId, (Vec<PeerId>, u64, u64)>,
    /// Providers a quorum of bootstraps reported gone, by libp2p id: (newest record time they
    /// held, first report unix ms). For [`GONE_MEMORY_MS`] no older-or-equal record of theirs is
    /// cached again (a stale copy from a DHT store or a registry would bring the ghost back).
    gone_providers: HashMap<String, (u64, u64)>,
    /// When each (model, provider peer) was last announced to us, for [`WITHDRAWN_MODEL_MEMORY`]:
    /// a provider's record can lapse here for a while; a withdrawal by another then isn't the model
    /// leaving the network (review 2026-10-14, L8).
    model_seen: HashMap<(String, String), std::time::Instant>,
    /// When each `known_peers` entry was last received here (local clock): what the cache cap
    /// evicts by, since a record's own signed time isn't set by every node (round 9 C9-1).
    known_received: HashMap<String, u64>,
    /// When we lost our last connection to each peer whose provider records we hold (local clock,
    /// ms). Its records stay for [`LOST_PEER_GRACE_MS`] in case the connection comes back, as a
    /// dropped one usually does at once (#30); cleared when it does.
    lost_at_ms: HashMap<PeerId, u64>,
    /// This node's libp2p id: its own records are never evicted from `known_peers`.
    local_libp2p: String,
    /// C11: model_id → (unix_ms of our last *completed* discover, net_generation then). A
    /// fresh entry lets `handle_discover` answer from `known_peers` without a fresh
    /// `get_providers` round-trip. Invalidated by `DISCOVER_CACHE_TTL_MS` and by a
    /// net-generation change (roam/wake) — so we never serve a stale relay path after moving.
    discover_cache: HashMap<String, (u64, u64)>,
    /// Models whose discovers wait for `get_providers` instead of answering from `known_peers`,
    /// with the `discover_seq` of the `InvalidateDiscover` that set it (sent when every known
    /// candidate just failed and the consumer retries to find one only the DHT knows). Cleared
    /// when a lookup started after it completes, not by the next discover, which might be someone
    /// else's (an audit, another request) and use it up.
    discover_wait_for_lookup: HashMap<String, u64>,
    /// Each open direct connection's peer and remote address (relayed ones are left out):
    /// identify's claims are only trusted for the IP a connection really comes from, and a
    /// rebootstrap skips bootstraps already connected on that transport ([`bootstrap_redials`]).
    conn_remote_addrs: HashMap<libp2p::swarm::ConnectionId, (PeerId, Multiaddr, std::time::Instant)>,
    /// When one of our interface addresses last went away (a roam, an interface down): connections
    /// made before it may run over a network we've left ([`bootstrap_redials`]).
    last_addr_loss: Option<std::time::Instant>,
    /// Counter shared by lookups and `InvalidateDiscover`s, ordering one against the other.
    discover_seq: u64,
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
    /// Proxy requests sent and not yet answered (or failed), per peer, and each one's peer: at most
    /// [`MAX_PROXY_INFLIGHT_PER_PEER`] go out at once (#31).
    proxy_inflight: HashMap<PeerId, usize>,
    proxy_request_peer: HashMap<request_response::OutboundRequestId, PeerId>,
    /// Proxy requests waiting for one of their peer's in-flight requests to finish, oldest first.
    proxy_waiting: HashMap<PeerId, std::collections::VecDeque<WaitingProxy>>,
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
    inbound_proxy_channels: HashMap<
        String,
        (request_response::ResponseChannel<ProxyResponse>, std::time::Instant, PeerId, request_response::InboundRequestId),
    >,
    /// Responses sent whose caller wants to know they reached the peer, by libp2p request id: told on
    /// `ResponseSent` or an inbound failure.
    /// With the peer it went to and when (swept after [`PROXY_CHANNEL_TTL`], in case libp2p reports
    /// neither outcome).
    pending_sent: HashMap<request_response::InboundRequestId, (OnSent, PeerId, std::time::Instant)>,
    /// Counter for generating unique inbound request IDs.
    inbound_proxy_counter: u64,
    /// Proxy forward requests waiting for a relay connection to be established.
    /// (target_peer_id, data, reply_channel, enqueued_at)
    /// C-N2: the `Instant` lets the reaper TTL-sweep entries whose relay dial
    /// never connects (`OutgoingConnectionError` retries but a permanently
    /// undialable peer would otherwise leak the reply channel + bytes forever).
    pending_relay_forwards:
        Vec<(PeerId, Vec<u8>, oneshot::Sender<Result<Vec<u8>, String>>, std::time::Instant)>,
    /// Outbound dials in flight, per peer (from `SwarmEvent::Dialing` until the connection is
    /// established or the dial fails): a failed dial fails the peer's queued forwards only when no
    /// other dial to it is still pending.
    dials_in_flight: DialsInFlight,
    /// Relayed-dial back-off per peer (see [`next_relay_dial_backoff`]).
    relay_dial_backoff: HashMap<PeerId, RelayDialBackoff>,
    /// Peers dialled directly on addresses this node was told ([`crate::pinned`]): the gateway's own
    /// provider on loopback. Never in Kademlia, the address book or any answer.
    pinned: crate::pinned::PinnedPeers,
    /// When each bootstrap was last re-dialled from a new port after an address-in-use refusal.
    in_use_redials: HashMap<PeerId, std::time::Instant>,
    /// C-N1: peers with both a direct and a relay connection, mapped to the
    /// instant at which the relay may be closed (direct-established-at + grace).
    /// Cleared if the direct path drops inside the window (so we keep the relay).
    relay_close_deadline: HashMap<PeerId, std::time::Instant>,
    /// v6-pref (C-N1 follow-on): peers with both a v6 and a v4 QUIC direct, mapped
    /// to the instant at which the v4 may be retired (v6-established-at + grace).
    /// Cleared if the v6 direct drops inside the window (so we keep the v4).
    v4_close_deadline: HashMap<PeerId, std::time::Instant>,
    /// Inbound duplicate QUIC connections waiting out [`QUIC_INBOUND_DUP_GRACE`] before the dedup
    /// closes them: connection -> (peer, is_ipv6, deadline).
    quic_dup_close_deadline: HashMap<libp2p::swarm::ConnectionId, (PeerId, bool, std::time::Instant)>,
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
    /// Peers with a connection to us over a loopback / private-network address (and how many): only
    /// such a peer's own non-global listen addresses are taken from identify (see
    /// [`identify_addr_admissible`]).
    local_link_peers: HashMap<PeerId, u32>,
    /// What each trusted relay's identify says its own addresses are ([`relay_self_addrs`]).
    relay_self_addrs: HashMap<PeerId, Vec<Multiaddr>>,
    /// The addresses we reached each trusted relay at directly.
    relay_dialled_addrs: HashMap<PeerId, Vec<Multiaddr>>,
    /// Dials to peers mDNS discovered: which to make, their bounds and back-off ([`crate::mdns::LanDialer`]).
    lan_dialer: crate::mdns::LanDialer,
    /// When to re-publish our provider records for a LAN peer that just joined ([`note_lan_subscriber`]).
    lan_announce: LanAnnounce,
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
    /// F-5: RESERVATION retry state per relay: (attempt_count, next_attempt_at, the circuit to
    /// reserve on). Retries OUR OWN reservation (``listen_on`` a circuit) so we stay reachable
    /// when a relay rejects or drops it. Without backoff a flapping relay caused either a tight
    /// re-listen loop or (on a hard listen_on error) a permanent strand for the whole session (the
    /// EU-relay gap). Keyed by relay, not by circuit address: a reservation reports one circuit
    /// per relay transport, and retrying each of them reserved on one relay several times at
    /// once, which libp2p resolves by closing all but one, each close scheduling more (the
    /// roaming churn, 2026-10-29). Cleared when a circuit through that relay comes up.
    relay_reservation_retries: HashMap<PeerId, (u32, tokio::time::Instant, Multiaddr)>,
    /// Our live circuit listeners and the relay each reserves on: at most one per relay. Every
    /// reservation request goes through [`request_reservation`], which checks this first.
    relay_listeners: HashMap<libp2p::core::transport::ListenerId, PeerId>,
    /// LAN peers we've re-announced for since they connected (see `note_lan_subscriber`); cleared on
    /// full disconnect.
    lan_announced: std::collections::HashSet<PeerId>,
    /// Consecutive failed pings per connection to one of our relays (see [`relay_connection_dead`]).
    relay_ping_failures: HashMap<libp2p::swarm::ConnectionId, u32>,
    /// Our direct connections to our relays: when each was established and where its relay sees
    /// us (see [`relay_connection_dead`]).
    relay_conns: HashMap<libp2p::swarm::ConnectionId, RelayConn>,
    /// P2: startup bootstrap dials waiting for a TCP listen address to dial from
    /// ([`startup_dials_ready`]), and until when they wait.
    startup_dials: Vec<(PeerId, Multiaddr)>,
    startup_dials_until: Option<tokio::time::Instant>,
    /// Whether we've announced, and whether the first relay reservation and the first gossip peer
    /// on our topic have come yet. A provider announces at once on startup, before it holds a
    /// reservation or has a gossip peer, so its first announce reached nobody (`put_record`
    /// timeout, gossip `InsufficientPeers`) and it was found only at the next re-announce, up to
    /// 2 min later (2026-10-29); each of those moments re-announces once.
    announced: bool,
    reservation_seen: bool,
    gossip_peer_seen: bool,
    /// What peers' PUT_RECORDs hold of our record store (see `dht::PutQuota`).
    put_quota: dht::PutQuota,
    /// The same for ADD_PROVIDER.
    provider_quota: dht::PutQuota,
    /// Our relayed connections, to whom (see [`relayed_connection_dead`]).
    relayed_conns: HashMap<libp2p::swarm::ConnectionId, PeerId>,
    /// Consecutive failed pings per relayed connection.
    relayed_ping_failures: HashMap<libp2p::swarm::ConnectionId, u32>,
    /// Responses being written, to whom, since when: from `send_response` to libp2p's
    /// `ResponseSent` or inbound failure.
    responses_writing: HashMap<request_response::InboundRequestId, (PeerId, std::time::Instant)>,
    /// P1: our direct IPv6 QUIC connections to our relays, and the relay each goes to.
    relay_v6_quic_conns: HashMap<libp2p::swarm::ConnectionId, PeerId>,
    /// P1: when we last dialled each relay over IPv6 QUIC (see [`relay_v6_quic_dial`]).
    relay_v6_quic_dialled: HashMap<PeerId, std::time::Instant>,
    /// Node-wide budget for IPv6 hole-punch rounds: (window start, rounds in it). See
    /// [`MAX_PUNCH_ROUNDS_PER_MINUTE`].
    punch_window: (std::time::Instant, u32),
    /// Relays a "couldn't reserve" hint was logged for (once each).
    relay_hints_given: std::collections::HashSet<String>,
    // ── #42: network-change resilience ────────────────────────────────
    /// User-supplied bootstrap peers, retained so `rebootstrap()` can re-dial
    /// them on a network change (the startup list is otherwise consumed by the
    /// run-loop locals). Identity is stable across a roam; only addresses churn.
    bootstrap_peers: Vec<(PeerId, Multiaddr)>,
    /// Closing idle Kademlia connections (`kad_idle`): the peers this node really uses, ...
    app_peers: crate::kad_idle::AppPeers,
    /// ... the peers whose identify says they know such closes ([`crate::kad_idle::AGENT_TOKEN`]),
    kad_idle_peers: std::collections::HashSet<PeerId>,
    /// ... the connections this node closed as idle (until their close arrives),
    idle_closed: std::collections::HashSet<libp2p::swarm::ConnectionId>,
    /// ... when it last looked for some, and how many it has closed.
    last_idle_sweep: std::time::Instant,
    /// Simulator only (`--kad-slow`): each model's last Kademlia announce (record fingerprint, when).
    kad_announced: HashMap<String, (u64, std::time::Instant)>,
    idle_closes: u64,
    /// The relays this node reserves on and dials others through (`relay::relay_addrs`): its
    /// bootstraps, plus the production relays on the public network.
    relays: Vec<Multiaddr>,
    /// Monotonically bumped every time connectivity is rebuilt (`rebootstrap()`).
    /// Read by the provider run-loop (via `NetworkHandle::network_generation`)
    /// to trigger an immediate model re-announce so the DHT record carries the
    /// new relay addresses under the *same* pinned PeerId. Shared with the handle.
    net_generation: Arc<std::sync::atomic::AtomicU64>,
    /// The latest ping round trip to each peer (shared with the handle, for consumers choosing
    /// between providers).
    peer_rtts: Arc<crate::rtt::PeerRtts>,
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

/// How old a gossip message may be and still be forwarded (#7): the gossip duplicate window
/// ([`crate::swarm::GOSSIP_DUPLICATE_CACHE`], 300 s) less a minute for clocks that disagree. Every
/// node that saw a message saw it no longer ago than its age, so a copy forwarded within this is
/// still in their duplicate caches and dropped there; one forwarded later (by a node waking from a
/// pause longer than the window) would be taken as new and flooded again. Older messages are still
/// used where they arrive (a record is accepted up to an hour old), just not passed on.
pub const GOSSIP_FORWARD_MAX_AGE_MS: u64 = 240_000;

/// Whether a gossip message (parsed) is fresh enough to forward ([`GOSSIP_FORWARD_MAX_AGE_MS`]): by
/// its record's signed time for an announce, by its `timestamp` (seconds) for a notice. One that
/// carries no time (an older node's) is forwarded, as before.
pub fn gossip_forwardable(v: &serde_json::Value, now_ms: u64) -> bool {
    let at_ms = match v.get("type").and_then(|t| t.as_str()) {
        Some(PROVIDER_ANNOUNCE_TYPE) => v.get("record").and_then(|r| r.get("updated_unix_ms")).and_then(|t| t.as_u64()),
        _ => v.get("timestamp").and_then(|t| t.as_u64()).map(|secs| secs.saturating_mul(1000)),
    };
    match at_ms {
        None | Some(0) => true,
        Some(at) => now_ms.saturating_sub(at) <= GOSSIP_FORWARD_MAX_AGE_MS,
    }
}

/// M6 (per-model tombstone): the gossipsub envelope `type` a provider publishes to
/// withdraw a *single* model it has stopped serving (e.g. the operator toggled it off
/// in the desktop) while the node itself stays up serving its other models. Consumers
/// fast-path this in Rust — author-check + evict just the `(peer, model)` entry from
/// `known_peers` — so an un-shared model disappears from browse lists immediately
/// instead of lingering out its ~300s discover-record TTL. Distinct from `PEER_DEPARTED`,
/// which evicts the *whole* peer on clean shutdown. Purely additive: a node that predates
/// this type ignores the message and keeps its TTL-expiry behaviour.
pub const MODEL_WITHDRAWN_TYPE: &str = "MODEL_WITHDRAWN";

/// A bootstrap's notice that a provider it held in its registry has had no connection to it for a
/// while (a crash, a pulled cable: nothing the provider could announce itself). Fields:
/// `libp2p_peer_id`, `last_record_ms` (the newest record time the bootstrap held for it; 0 for an
/// undated one) and `timestamp` (unix seconds). GPU test 2026-10-30: without it, a consumer that
/// never connected to a dead provider listed it for up to ~10 min (300 s record TTL + 300 s sticky
/// browse list).
pub const PROVIDER_GONE_TYPE: &str = "PROVIDER_GONE";
/// Distinct bootstraps that must report a provider gone before a consumer drops it (fewer if it
/// knows fewer): one bootstrap losing its own path to a provider is not the provider dying.
const PROVIDER_GONE_QUORUM: usize = 2;
/// How old a `PROVIDER_GONE` may be, and how long reports are kept while the quorum gathers.
pub const PROVIDER_GONE_MAX_AGE_SECS: u64 = 120;
/// Providers with reports gathering at most (the oldest go first).
const MAX_PROVIDER_GONE_PENDING: usize = 4096;
/// How long a provider reported gone stays remembered (see `LoopState::gone_providers`): past the
/// DHT record TTL and the registry TTL (300 s each), so no stale copy outlives it.
const GONE_MEMORY_MS: u64 = 600_000;

/// Whether `record` is from a provider reported gone and no newer than what was reported then.
/// Undated records (older nodes) can't be told apart from a fresh announce, so they're never
/// refused here — refusing them would hide such a provider for the whole memory after it came
/// back; they only get the receive-time eviction. Pure.
fn is_gone_record(gone: &HashMap<String, (u64, u64)>, record: &PeerRecord, now_ms: u64) -> bool {
    match gone.get(&record.libp2p_peer_id) {
        Some((last, first)) if now_ms.saturating_sub(*first) < GONE_MEMORY_MS => {
            record.updated_unix_ms != 0 && record.updated_unix_ms <= *last
        }
        _ => false,
    }
}

/// M6/F1 (replay hardening): how far a `MODEL_WITHDRAWN` message's `timestamp` may be from the
/// receiver's clock (either direction) and still be honoured. The timestamp lives inside the
/// signed gossip payload, so an attacker replaying a captured withdrawal cannot refresh it — once
/// the original is older than this window the replay is ignored, while a legitimate re-publish
/// (the provider re-issues over a few announce passes, each a fresh signed message) always carries
/// a current timestamp and passes. The window is two-sided to tolerate modest clock skew between
/// nodes; it is deliberately generous (a false-reject merely falls back to the ~300s TTL, and a
/// false-accept is a bounded browse-list flip) and matches the 120s re-announce cadence.
pub const MODEL_WITHDRAWN_MAX_AGE_SECS: u64 = 120;

/// R-DHT-1 audit fix: how long a `known_peers` entry we are NOT connected to may
/// survive in the reaper. A PEX-learned provider is, by definition, one we have
/// no live connection to (we heard it over gossip via a relay) — reaping it
/// purely on connection status defeated gossip discovery. We instead keep it
/// while it's *fresh* (its signed `updated_unix_ms` is within this window),
/// matching the 300 s DHT record TTL; a provider that stops (re)announcing then
/// ages out, while a live one is kept refreshed by each gossip.
const KNOWN_PEER_TTL_MS: u64 = 300_000;
/// How old a `PEER_DEPARTED` may be and still be acted on: the gossip forward window (a copy can
/// still be travelling that long), no longer.
const PEER_DEPARTED_MAX_AGE_SECS: u64 = GOSSIP_FORWARD_MAX_AGE_MS / 1000;

/// Evict the records of `libp2p_peer_id` its departure at `departed_at_ms` (its signed time) is
/// newer than, or that carry no time (an older node's); one it signed since (it came back) stays.
/// Returns how many went.
fn evict_departed(known_peers: &mut HashMap<String, PeerRecord>, libp2p_peer_id: &str, departed_at_ms: u64) -> usize {
    let before = known_peers.len();
    known_peers.retain(|_, r| r.libp2p_peer_id != libp2p_peer_id || (r.updated_unix_ms != 0 && r.updated_unix_ms > departed_at_ms));
    before - known_peers.len()
}

/// How long a stopping node keeps its swarm running after publishing its departure, so the notice
/// is sent before it goes (#19).
const DEPARTURE_FLUSH: std::time::Duration = std::time::Duration::from_millis(800);

/// How long a provider's records outlive our last connection to it (#30). A connection that drops
/// is usually back within seconds (the next request or a redial reopens it), and the provider with
/// it; deleting its records at once left a gateway answering "no provider" until the provider's
/// next announcement, up to two minutes later. A provider that really left goes after this (a
/// departure notice removes it at once), checked every [`REAPER_EVERY`].
const LOST_PEER_GRACE_MS: u64 = 30_000;
/// How often the reaper runs.
const REAPER_EVERY: std::time::Duration = std::time::Duration::from_secs(15);

/// Whether the reaper keeps a provider record: while we're connected to the provider; not once we
/// lost our connection to it [`LOST_PEER_GRACE_MS`] ago with no record from it since; otherwise
/// for [`KNOWN_PEER_TTL_MS`] after we last received the record.
fn keep_known_peer(connected: bool, received_at_ms: Option<u64>, lost_at_ms: Option<u64>, now_ms: u64) -> bool {
    if connected {
        return true;
    }
    if let Some(lost) = lost_at_ms {
        let heard_since = received_at_ms.is_some_and(|at| at > lost);
        if !heard_since && now_ms.saturating_sub(lost) >= LOST_PEER_GRACE_MS {
            return false;
        }
    }
    received_at_ms.is_some_and(|at| now_ms.saturating_sub(at) < KNOWN_PEER_TTL_MS)
}

/// How far in the future a provider record's signed time may be (clock skew) before it is refused,
/// so a far-future one can't stay "newest" (round 8 N8-2); the same limit as the DHT's
/// [`dht::MAX_RECORD_FUTURE_MS`]. The reaper ages records by when we received them.
const MAX_RECORD_SKEW_MS: u64 = dht::MAX_RECORD_FUTURE_MS;
/// Provider records kept per libp2p id (one per model it serves); past it, its least recently
/// received goes. Above any one machine's model list, so a big library doesn't churn (C9-3).
const MAX_KNOWN_RECORDS_PER_PEER: usize = 256;
/// Provider records kept overall; past it, the least recently received goes.
const MAX_KNOWN_RECORDS: usize = 16_384;

/// Cache a verified provider record in `state.known_peers`, stamped received now: refused (`None`)
/// when it is dated too far ahead, over an hour old, or older than the one held; else
/// `Some(whether it is new)`. Bounded per peer and overall by
/// receive time (round 9 C9-1: a record's own time is 0 from most nodes), never evicting this
/// node's own records.
fn install_known_peer(state: &mut LoopState, record: &PeerRecord, now_ms: u64) -> Option<bool> {
    if is_gone_record(&state.gone_providers, record, now_ms) {
        return None;
    }
    let installed = install_known_peer_in(&mut state.known_peers, &mut state.known_received, &state.local_libp2p, record, now_ms);
    // A duplicate says nothing new about the provider (it may be a replay): no side effects.
    if matches!(installed, Some(Install::New | Install::Renewed)) {
        // It signed this after a dial to it failed: it's (back) up, so let the next forward dial.
        if let Ok(pid) = record.libp2p_peer_id.parse::<PeerId>() {
            if state.relay_dial_backoff.get(&pid).is_some_and(|b| record.updated_unix_ms > b.set_unix_ms) {
                state.relay_dial_backoff.remove(&pid);
            }
        }
        note_model_seen(&mut state.model_seen, &record.model_id, &record.libp2p_peer_id, std::time::Instant::now());
        // A lookup for this model running now learns it too (gossip often brings a just-joined
        // provider's new address before the DHT does); its waiters are answered on the next
        // discover tick.
        if record.libp2p_peer_id != state.local_libp2p {
            for pending in state.pending_discovers.values_mut().filter(|p| p.model_id == record.model_id) {
                pending.merge(record.clone(), now_ms);
            }
        }
    }
    installed.map(|i| i == Install::New)
}

/// What [`install_known_peer_in`] did with a record it accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Install {
    /// Not cached before.
    New,
    /// Replaced a held record with a newer one (or renewed an undated one): the provider is alive.
    Renewed,
    /// The same dated record again (a re-forwarded or replayed copy): kept as it was, renews nothing.
    Duplicate,
}

fn install_known_peer_in(
    known: &mut HashMap<String, PeerRecord>,
    received: &mut HashMap<String, u64>,
    local: &str,
    record: &PeerRecord,
    now_ms: u64,
) -> Option<Install> {
    if record.updated_unix_ms > now_ms.saturating_add(MAX_RECORD_SKEW_MS) {
        return None;
    }
    // Callers verify first (dht::verify_peer_record includes this); kept here too so no path can
    // cache a record with an odd model name.
    if dht::check_record_content(record).is_err() || dht::check_record_age(record, now_ms).is_err() {
        return None;
    }
    let key = known_peer_key(record);
    // Never older over newer, and only a NEWER record renews the reaper's clock: a re-forwarded or
    // replayed copy of a provider's last announce (gossip, a DHT store, a registry reply) would
    // otherwise keep a dead provider listed forever, or put a roamed provider's old address back
    // (review 2026-10-30). Records from nodes that don't date them (time 0) renew on receipt, as
    // before, until those nodes are gone.
    if let Some(held) = known.get(&key) {
        if record.updated_unix_ms < held.updated_unix_ms {
            return None;
        }
        if record.updated_unix_ms == held.updated_unix_ms && record.updated_unix_ms != 0 {
            return Some(Install::Duplicate);
        }
    }
    let is_new = !known.contains_key(&key);
    if is_new {
        if received.len() > known.len() {
            received.retain(|k, _| known.contains_key(k)); // entries removed elsewhere
        }
        let oldest = |known: &HashMap<String, PeerRecord>, received: &HashMap<String, u64>, same_peer: bool| {
            known
                .iter()
                .filter(|(_, r)| r.libp2p_peer_id != local && (!same_peer || r.libp2p_peer_id == record.libp2p_peer_id))
                .min_by_key(|(k, _)| received.get(*k).copied().unwrap_or(0))
                .map(|(k, _)| k.clone())
        };
        if record.libp2p_peer_id != local
            && known.values().filter(|r| r.libp2p_peer_id == record.libp2p_peer_id).count() >= MAX_KNOWN_RECORDS_PER_PEER
        {
            if let Some(k) = oldest(known, received, true) {
                known.remove(&k);
                received.remove(&k);
            }
        }
        if known.len() >= MAX_KNOWN_RECORDS {
            if let Some(k) = oldest(known, received, false) {
                known.remove(&k);
                received.remove(&k);
            }
        }
    }
    received.insert(key.clone(), now_ms);
    known.insert(key, record.clone());
    Some(if is_new { Install::New } else { Install::Renewed })
}

/// C11: how long a *completed* `discover` for a model stays fresh enough to answer from the
/// `known_peers` cache without a new `get_providers` round-trip. Coalesces a request burst (a
/// chat session / an agent loop) so only ~1 request per window pays the DHT lookup — which
/// cross-NAT is the ~1.2 s `discover` hop (gossip PEX can't populate the cache across separate
/// networks). A miss no longer waits on the lookup when providers are already known (see
/// `handle_discover`), so this sets how often a model in use is re-resolved in the background:
/// kept short so a provider only the DHT knows (cross-NAT) shows up promptly. Failover + the
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

/// One `get_providers` lookup for a model, and the callers waiting on it. There is at most one per
/// model at a time: a caller that needs a lookup while one is running joins it (review 2026-10-29:
/// the consumer invalidates before every retry, so each retry used to start its own).
struct PendingDiscover {
    model_id: String,
    /// The model's providers: the known ones it started with, plus what the lookup finds, newest
    /// signed record per provider ([`merge_record`]).
    records: Vec<PeerRecord>,
    /// Set once the `get_providers` query terminates (found-all or timed-out). Until then,
    /// more providers — and thus more chained record fetches — may still arrive.
    providers_done: bool,
    /// In-flight `get_record` fetches for providers whose full `PeerRecord` wasn't held
    /// locally (cross-relay, the record lives on the bootstrap nodes, not on us).
    outstanding: usize,
    /// Bumped whenever `records` gains a provider or a provider's address changes: what a retry
    /// waits for ([`waiter_settled`]).
    info_gen: u64,
    /// The callers still waiting for an answer. None: the lookup runs on as a background refresh.
    waiters: Vec<DiscoverWaiter>,
    /// Its place in `LoopState::discover_seq`: only a lookup started after a model's
    /// `InvalidateDiscover` may clear that model's wait flag or mark it fresh.
    seq: u64,
}

impl PendingDiscover {
    /// Merge `record` into the lookup's providers, noting new reachability information.
    fn merge(&mut self, record: PeerRecord, now_ms: u64) {
        match merge_record(&mut self.records, record, now_ms) {
            Merge::Added | Merge::Replaced { address_changed: true } => self.info_gen += 1,
            Merge::Replaced { address_changed: false } | Merge::Kept => {}
        }
    }
}

/// A caller waiting on a [`PendingDiscover`].
struct DiscoverWaiter {
    reply: oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>,
    kind: DiscoverKind,
    /// The lookup's `info_gen` when this caller joined: a retry is answered once there's newer.
    info_at_join: u64,
    /// A retry is answered with what's in hand by then, whatever the DHT is doing.
    deadline: Option<std::time::Instant>,
}

/// What a caller of a discover lookup is waiting for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiscoverKind {
    /// Nothing was known: answer with the first providers found.
    Reply,
    /// The caller's known providers all just failed: answer once a provider is new or reachable
    /// at a different address, or at [`RETRY_SETTLE`].
    Retry,
}

/// How long a retry's lookup may hold it before it gets what's in hand (review 2026-10-29: it waited
/// for the whole lookup, up to its 10 s timeout, so the 30 s retry budget had room for 2–3 tries).
/// The lookup itself runs on and refreshes the cache.
const RETRY_SETTLE: std::time::Duration = std::time::Duration::from_millis(1500);

/// How [`merge_record`] treated a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Merge {
    /// A provider not in the list.
    Added,
    /// A newer record for one already there; `address_changed` says whether it's reachable
    /// somewhere else now (a republish that only renews the time isn't).
    Replaced { address_changed: bool },
    /// Older, as old, or dated too far ahead: the one there stays.
    Kept,
}

/// Merge a verified provider record into a lookup's list: one record per provider, the newest
/// signed one winning (review 2026-10-29: the first one seen won, so a retry kept answering with a
/// provider's stale address after a fresher record arrived). A record dated more than
/// [`MAX_RECORD_SKEW_MS`] ahead is never preferred, the same limit `known_peers` applies.
fn merge_record(records: &mut Vec<PeerRecord>, record: PeerRecord, now_ms: u64) -> Merge {
    if record.updated_unix_ms > now_ms.saturating_add(MAX_RECORD_SKEW_MS) {
        return Merge::Kept;
    }
    match records.iter_mut().find(|r| r.libp2p_peer_id == record.libp2p_peer_id) {
        None => {
            records.push(record);
            Merge::Added
        }
        Some(r) if record.updated_unix_ms > r.updated_unix_ms => {
            let address_changed = reachability(r) != reachability(&record);
            *r = record;
            Merge::Replaced { address_changed }
        }
        Some(_) => Merge::Kept,
    }
}

/// Where a provider record says it can be reached.
fn reachability(r: &PeerRecord) -> (&str, &str, u16, &str, bool) {
    (&r.relay_address, &r.host, r.port, &r.host_ipv6, r.requires_relay)
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
            withdrawn_models: HashMap::new(),
            provider_gone: HashMap::new(),
            gone_providers: HashMap::new(),
            model_seen: HashMap::new(),
            known_received: HashMap::new(),
            lost_at_ms: HashMap::new(),
            local_libp2p: String::new(),
            discover_cache: HashMap::new(),
            discover_wait_for_lookup: HashMap::new(),
            conn_remote_addrs: HashMap::new(),
            last_addr_loss: None,
            discover_seq: 0,
            pending_discovers: HashMap::new(),
            pending_record_fetches: HashMap::new(),
            external_addrs: Vec::new(),
            upnp_external_addrs: std::collections::HashSet::new(),
            autonat_private: false,
            kad_server_mode: false, // nodes start as Kad clients (R-DHT-2)
            relay_addrs: Vec::new(),
            pending_proxy: HashMap::new(),
            proxy_inflight: HashMap::new(),
            proxy_request_peer: HashMap::new(),
            proxy_waiting: HashMap::new(),
            pending_registry: HashMap::new(),
            local_grpc_port: 50051,
            inbound_proxy_channels: HashMap::new(),
            pending_sent: HashMap::new(),
            inbound_proxy_counter: 0,
            pending_relay_forwards: Vec::new(),
            dials_in_flight: HashMap::new(),
            relay_dial_backoff: HashMap::new(),
            pinned: Default::default(),
            in_use_redials: HashMap::new(),
            relay_close_deadline: HashMap::new(),
            v4_close_deadline: HashMap::new(),
            quic_dup_close_deadline: HashMap::new(),
            dcutr_successes: 0,
            dcutr_failures: 0,
            enable_connection_reversal: false,
            reversal_attempts: HashMap::new(),
            reversal_pending: std::collections::HashSet::new(),
            reversal_dials: 0,
            reversal_successes: 0,
            tier_connect_success: std::collections::HashMap::new(),
            peer_connections: HashMap::new(),
            local_link_peers: HashMap::new(),
            relay_self_addrs: HashMap::new(),
            relay_dialled_addrs: HashMap::new(),
            lan_dialer: crate::mdns::LanDialer::new(),
            lan_announce: LanAnnounce::default(),
            local_proxy_replies: HashMap::new(),
            last_repunch: HashMap::new(),
            peer_quic_addrs: HashMap::new(),
            quic_holepunch_attempts: HashMap::new(),
            dcutr_event_queue: VecDeque::new(),
            tunnel_close_queue: VecDeque::new(),
            gossip_inbound_queue: std::collections::VecDeque::new(),
            relay_reservation_retries: HashMap::new(),
            relay_listeners: HashMap::new(),
            relay_ping_failures: HashMap::new(),
            lan_announced: std::collections::HashSet::new(),
            relay_conns: HashMap::new(),
            announced: false,
            reservation_seen: false,
            gossip_peer_seen: false,
            put_quota: dht::PutQuota::default(),
            provider_quota: dht::PutQuota::default(),
            relayed_conns: HashMap::new(),
            relayed_ping_failures: HashMap::new(),
            responses_writing: HashMap::new(),
            startup_dials: Vec::new(),
            startup_dials_until: None,
            relay_v6_quic_conns: HashMap::new(),
            relay_v6_quic_dialled: HashMap::new(),
            punch_window: (std::time::Instant::now(), 0),
            relay_hints_given: std::collections::HashSet::new(),
            // #42: set to the shared handle in run_event_loop; a real interface
            // change or watchdog rebuild bumps it.
            bootstrap_peers: Vec::new(),
            app_peers: Default::default(),
            kad_idle_peers: Default::default(),
            idle_closed: Default::default(),
            last_idle_sweep: std::time::Instant::now(),
            kad_announced: HashMap::new(),
            idle_closes: 0,
            relays: Vec::new(),
            net_generation: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            peer_rtts: Arc::new(crate::rtt::PeerRtts::default()),
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
    // The relays to reserve on and dial through (`relay::relay_addrs`).
    relays: Vec<Multiaddr>,
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
    // The latest ping round trip to each peer, shared with the handle.
    peer_rtts: Arc<crate::rtt::PeerRtts>,
    // #43-W2: opt-in PCP v6 firewall-pinhole wiring (None = disabled).
    pcp_bind: Option<crate::pcp::PcpBind>,
    // The own provider's loopback addresses (`crate::pinned`), shared with the Kademlia filter.
    pinned: crate::pinned::PinnedPeers,
    // The peers this node really uses (requests, push streams), shared with the push streams.
    app_peers: crate::kad_idle::AppPeers,
) {
    let mut state = LoopState::new();
    state.pinned = pinned;
    state.app_peers = app_peers;
    state.local_libp2p = swarm.local_peer_id().to_base58();
    state.enable_connection_reversal = enable_connection_reversal;
    state.peer_relay_leech = peer_relay_leech; // WS-F F-4
    // #42: retain the bootstrap list + share the generation counter so
    // `rebootstrap()` can re-dial and signal the provider on a network change.
    state.bootstrap_peers = bootstrap_peers.clone();
    state.relays = relays;
    state.net_generation = net_generation;
    state.peer_rtts = peer_rtts;
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

    // The Kademlia bootstrap (populate the routing table from the bootstrap peers) starts once the
    // startup dials below are out (P2): it dials the bootstraps itself when they aren't connected or
    // being dialled yet, and its dials, made before the listen addresses come up, would leave from
    // ephemeral ports (2026-10-29 retest: 3 of 4 bootstraps saw such a second connection).

    // Explicitly dial every bootstrap peer to force a direct connection.
    // kademlia.add_address() only populates the routing table — it does
    // NOT guarantee a connection.  Without an explicit dial the swarm
    // may never establish a direct QUIC link and will fall back to relay.
    // P2: TCP dials wait for a TCP listen address to dial from (see `startup_dials_ready`).
    for (peer_id, addr) in &bootstrap_peers {
        // (Not a v6 one on a host without working IPv6: it fails at once anyway, and waiting for a v6
        // listen address that never comes would hold up the Kademlia bootstrap.)
        if matches!(dial_transport(addr), Some((v6, _, false)) if state.ipv6_capable || v6 != Some(true)) {
            state.startup_dials.push((*peer_id, addr.clone()));
            continue;
        }
        let dial_addr = addr.clone().with(libp2p::multiaddr::Protocol::P2p(*peer_id));
        info!(%peer_id, %dial_addr, "bootstrap_dial: explicitly dialing bootstrap peer");
        if let Err(e) = swarm.dial_tracked(&mut state.dials_in_flight, dial_addr.clone()) {
            warn!(%peer_id, %dial_addr, %e, "bootstrap_dial: failed");
        }
    }
    if state.startup_dials.is_empty() {
        start_kad_bootstrap(&mut swarm);
    } else {
        state.startup_dials_until = Some(tokio::time::Instant::now() + STARTUP_DIAL_WAIT);
    }

    // R-DHT-6: warm-start the routing table from the persisted cache so a fresh
    // process rejoins with known-good contacts instead of leaning entirely on the
    // bootstrap relays. Best-effort: a missing/corrupt cache just yields nothing.
    if let Some(ref cache_path) = routing_cache_path {
        let warm = routing_cache::load(cache_path);
        let n = warm.len();
        for (peer_id, addr) in warm {
            // A relayed entry (from a cache written before the 20 Nov identify rule) is a peer that
            // doesn't answer DHT requests: kept for our own dials only.
            if addr.iter().any(|p| matches!(p, libp2p::multiaddr::Protocol::P2pCircuit)) {
                swarm.behaviour_mut().address_book.insert(peer_id, addr);
            } else {
                swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
            }
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
    let mut reaper_interval = tokio::time::interval(REAPER_EVERY);
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

    // Discover waiters: retry deadlines (RETRY_SETTLE) and records gossip brought to a running
    // lookup. Does nothing when no caller is waiting.
    let mut discover_ticker = tokio::time::interval(std::time::Duration::from_millis(250));
    discover_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

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
                if let Err(e) = swarm.dial_tracked(&mut state.dials_in_flight, dial_addr.clone()) {
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
        // P2: startup bootstrap dials that found no TCP listen address to dial from in time go out
        // from a new port, which identify then reports with our listen port.
        if state.startup_dials_until.is_some_and(|t| tokio::time::Instant::now() >= t) {
            state.startup_dials_until = None;
            for (peer_id, addr) in std::mem::take(&mut state.startup_dials) {
                info!(%peer_id, %addr, "bootstrap_dial: no TCP listen address yet, dialing from a new port");
                let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(peer_id)
                    .addresses(vec![addr.clone()])
                    .condition(libp2p::swarm::dial_opts::PeerCondition::Always)
                    .allocate_new_port()
                    .build();
                if let Err(e) = swarm.dial_tracked(&mut state.dials_in_flight, opts) {
                    warn!(%peer_id, %addr, %e, "bootstrap_dial: failed");
                }
            }
            start_kad_bootstrap(&mut swarm);
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
            reserve_on_relays(&mut swarm, &mut state, "startup");
        }

        tokio::select! {
            // Process commands from Python.
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(SwarmCommand::Announce { record, reply }) => {
                        handle_announce(&mut swarm, &record, reply, &mut state, &keypair);
                    }
                    Some(SwarmCommand::WithdrawModel { model_id, reply }) => {
                        handle_withdraw_model(&mut swarm, &model_id, reply);
                    }
                    Some(SwarmCommand::Discover { model_id, reply }) => {
                        handle_discover(&mut swarm, &model_id, reply, &mut state);
                    }
                    Some(SwarmCommand::InvalidateDiscover { model_id, reply }) => {
                        // C11 bypass: forget this model's cached discovery, and make the next
                        // Discover wait for get_providers instead of answering from the providers
                        // already known (every one of them just failed for the caller).
                        state.discover_cache.remove(&model_id);
                        state.discover_seq += 1;
                        // Bounded against junk model ids: when full, a new model just isn't flagged
                        // (its retry answers from known providers) rather than evicting others' flags.
                        if state.discover_wait_for_lookup.len() < 4096 || state.discover_wait_for_lookup.contains_key(&model_id) {
                            state.discover_wait_for_lookup.insert(model_id, state.discover_seq);
                        }
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
                                    direct_ips: {
                                        let mut ips: Vec<String> = info.direct_ips.values().cloned().collect();
                                        ips.sort();
                                        ips.dedup();
                                        ips
                                    },
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
                            .filter(|m| openhydra_protocol::model_id::is_valid_model_id(m))
                            .collect();
                        known_models.sort();
                        known_models.dedup();
                        let known_providers = state
                            .known_peers
                            .values()
                            .filter(|r| openhydra_protocol::model_id::is_valid_model_id(&r.model_id))
                            .map(|r| KnownProvider {
                                model_id: r.model_id.clone(),
                                openhydra_peer_id: r.peer_id.clone(),
                                libp2p_peer_id: r.libp2p_peer_id.clone(),
                                public_key: r.public_key.clone(),
                            })
                            .collect();
                        let kad_routing_peers = swarm
                            .behaviour_mut()
                            .kademlia
                            .kbuckets()
                            .map(|b| b.num_entries())
                            .sum();
                        let topic = libp2p::gossipsub::IdentTopic::new(crate::swarm::GOSSIPSUB_TOPIC).hash();
                        let gossip = &swarm.behaviour().gossipsub;
                        let gossip_topic_peers: Vec<String> = gossip
                            .all_peers()
                            .filter(|(_, topics)| topics.contains(&&topic))
                            .map(|(p, _)| p.to_base58())
                            .collect();
                        let gossip_mesh_peers: Vec<String> = gossip.mesh_peers(&topic).map(|p| p.to_base58()).collect();
                        let recently_withdrawn = recently_withdrawn(&mut state.withdrawn_models, &state.known_peers, std::time::Instant::now());
                        let snapshot = StatusSnapshot {
                            gossip_topic_peers,
                            gossip_mesh_peers,
                            recently_withdrawn,
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
                                gossip_bytes_in: gossip_count(&libp2p::gossipsub::openhydra_counters::RPC_BYTES_IN),
                                gossip_copies_checked: gossip_count(&libp2p::gossipsub::openhydra_counters::COPIES_CHECKED),
                                gossip_unique: gossip_count(&libp2p::gossipsub::openhydra_counters::UNIQUE_MESSAGES),
                                gossip_duplicates: gossip_count(&libp2p::gossipsub::openhydra_counters::DUPLICATES),
                            },
                        };
                        let _ = reply.send(snapshot);
                    }
                    Some(SwarmCommand::RecentlyWithdrawn { reply }) => {
                        let _ = reply.send(recently_withdrawn(
                            &mut state.withdrawn_models,
                            &state.known_peers,
                            std::time::Instant::now(),
                        ));
                    }
                    Some(SwarmCommand::KnownModels { reply }) => {
                        // Records are checked on the way in (verify_peer_record); filtering
                        // again here keeps an odd name out of /v1/models if any path missed it.
                        let mut models: Vec<String> = state
                            .known_peers
                            .values()
                            .map(|r| r.model_id.clone())
                            .filter(|m| openhydra_protocol::model_id::is_valid_model_id(m))
                            .collect();
                        models.sort();
                        models.dedup();
                        let _ = reply.send(models);
                    }
                    Some(SwarmCommand::ModelsOf { libp2p_peer_id, reply }) => {
                        let mut models: Vec<String> = state
                            .known_peers
                            .values()
                            .filter(|r| r.libp2p_peer_id == libp2p_peer_id)
                            .map(|r| r.model_id.clone())
                            .filter(|m| openhydra_protocol::model_id::is_valid_model_id(m))
                            .collect();
                        models.sort();
                        models.dedup();
                        let _ = reply.send(models);
                    }
                    Some(SwarmCommand::KnownProviders { reply }) => {
                        let mut providers: Vec<KnownProvider> = state
                            .known_peers
                            .values()
                            .filter(|r| openhydra_protocol::model_id::is_valid_model_id(&r.model_id))
                            .map(|r| KnownProvider {
                                model_id: r.model_id.clone(),
                                // Verified records carry their key's own OpenHydra id (dht::verify_peer_record).
                                openhydra_peer_id: r.peer_id.clone(),
                                libp2p_peer_id: r.libp2p_peer_id.clone(),
                                public_key: r.public_key.clone(),
                            })
                            .collect();
                        providers.sort_by(|a, b| (&a.model_id, &a.libp2p_peer_id).cmp(&(&b.model_id, &b.libp2p_peer_id)));
                        providers.dedup_by(|a, b| a.model_id == b.model_id && a.libp2p_peer_id == b.libp2p_peer_id);
                        let _ = reply.send(providers);
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
                                match swarm.dial_tracked(&mut state.dials_in_flight, opts) {
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
                    Some(SwarmCommand::PinPeerAddrs { peer_id, addrs, reply }) => {
                        let _ = reply.send(pin_peer_addrs(&mut swarm, &mut state, peer_id, addrs));
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
                        handle_open_proxy(&mut swarm, &target_libp2p_peer_id, reply, &mut state);
                    }
                    Some(SwarmCommand::RespondProxy { request_id, data, on_sent }) => {
                        if let Some(reply) = state.local_proxy_replies.remove(&request_id) {
                            let sent = reply.send(Ok(data)).is_ok();
                            if let Some(f) = on_sent {
                                f(sent);
                            }
                        } else if let Some((channel, _, peer, rid)) = state.inbound_proxy_channels.remove(&request_id) {
                            match swarm.behaviour_mut().grpc_proxy.send_response(channel, ProxyResponse(data)) {
                                Ok(()) => {
                                    if state.responses_writing.len() < MAX_PENDING_SENT {
                                        state.responses_writing.insert(rid, (peer, std::time::Instant::now()));
                                    }
                                    if let Some(f) = on_sent {
                                        let mine = state.pending_sent.values().filter(|(_, p, _)| *p == peer).count();
                                        if state.pending_sent.len() < MAX_PENDING_SENT && mine < MAX_PENDING_SENT_PER_PEER {
                                            state.pending_sent.insert(rid, (f, peer, std::time::Instant::now()));
                                        } else {
                                            f(false); // can't track it: don't claim it arrived
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!("proxy respond failed: {:?}", e);
                                    if let Some(f) = on_sent {
                                        f(false);
                                    }
                                }
                            }
                        } else {
                            warn!("proxy respond: unknown request_id={}", request_id);
                            if let Some(f) = on_sent {
                                f(false);
                            }
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
                                match swarm.dial_tracked(&mut state.dials_in_flight, opts) {
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
                    Some(SwarmCommand::ServeAccepted { peer_id }) => {
                        pursue_v6_holepunch_debounced(&mut swarm, &mut state, peer_id, "serve");
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

                        // 3. Keep the swarm running a moment, so the departure actually goes out
                        // (#19: returning at once dropped it, queued but unsent).
                        let until = tokio::time::Instant::now() + DEPARTURE_FLUSH;
                        loop {
                            tokio::select! {
                                _ = futures::StreamExt::select_next_some(&mut swarm) => {}
                                _ = tokio::time::sleep_until(until) => break,
                            }
                        }

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
                let received = &state.known_received;
                let lost = &state.lost_at_ms;
                state.known_peers.retain(|key, record| {
                    if record.libp2p_peer_id.is_empty() {
                        return true; // keep records without libp2p binding
                    }
                    match record.libp2p_peer_id.parse::<PeerId>() {
                        // Keep if we have a live connection (the original ghost-peer
                        // guard) OR if the record is still fresh. The freshness arm
                        // is the R-DHT-1 audit fix: a PEX-learned provider we never
                        // connected to must survive on its TTL, or this reaper would
                        // evict it ~60 s after we learned it and break gossip
                        // discovery for any re-announce interval > 60 s. Fresh = received in the
                        // window (each re-announce renews it): the record's own time was 0 from
                        // every node before 2026-10-29, so this arm never kept anything and a
                        // consumer with no connection to a provider lost it a minute after
                        // hearing of it (GPU test 2026-10-29), and a clock that's off mustn't
                        // decide it either.
                        Ok(pid) => keep_known_peer(
                            swarm.is_connected(&pid),
                            received.get(key).copied(),
                            lost.get(&pid).copied(),
                            now_ms,
                        ),
                        Err(_) => false, // unparseable peer_id = stale
                    }
                });
                let now_instant = std::time::Instant::now();
                state.relay_dial_backoff.retain(|_, b| b.until > now_instant);
                // Entries whose record went elsewhere (disconnect purge, departures, withdrawals).
                let known = &state.known_peers;
                state.known_received.retain(|k, _| known.contains_key(k));
                // Lost peers past their grace have been dealt with above.
                state.lost_at_ms.retain(|_, at| now_ms.saturating_sub(*at) < LOST_PEER_GRACE_MS);
                // Proxy requests whose callers gave up while waiting for a slot (#31).
                state.proxy_waiting.retain(|_, q| {
                    q.retain(|(_, reply)| !reply.as_ref().is_some_and(|r| r.is_closed()));
                    !q.is_empty()
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
                state.inbound_proxy_channels.retain(|_, (_, t, _, _)| t.elapsed() < PROXY_CHANNEL_TTL);
                state.responses_writing.retain(|_, (_, t)| t.elapsed() < RESPONSE_WRITING_TTL);
                // Responses libp2p never reported on: told they didn't arrive (review round 4 S4-12).
                let stale: Vec<_> =
                    state.pending_sent.iter().filter(|(_, (_, _, t))| t.elapsed() >= PROXY_CHANNEL_TTL).map(|(k, _)| *k).collect();
                for k in stale {
                    if let Some((f, _, _)) = state.pending_sent.remove(&k) {
                        f(false);
                    }
                }
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
                // QUIC dedup: close inbound duplicates whose grace ran out, if still redundant.
                let dup_due: Vec<_> = state
                    .quic_dup_close_deadline
                    .iter()
                    .filter(|(_, (_, _, deadline))| now >= *deadline)
                    .map(|(&cid, &(pid, v6, _))| (cid, pid, v6))
                    .collect();
                for (cid, _, _) in &dup_due {
                    state.quic_dup_close_deadline.remove(cid);
                }
                let (close, stay) = quic_dups_due(&dup_due, &state.peer_connections, swarm.local_peer_id());
                for cid in close {
                    debug!(?cid, "quic_dedup: closing inbound duplicate QUIC connection after grace");
                    let _ = swarm.close_connection(cid);
                }
                // A duplicate that outlived the original is the peer's QUIC connection now: a v4 one
                // is tracked as such, so a stable v6 direct can still retire it (v6-pref).
                for (cid, pid, v6) in stay {
                    if let Some(info) = state.peer_connections.get_mut(&pid) {
                        if !v6 && !info.quic_v4_conn_ids.contains(&cid) {
                            info.quic_v4_conn_ids.push(cid);
                        }
                    }
                }
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
            _ = discover_ticker.tick() => {
                drive_discover_waiters(&mut state);
                drive_lan_announce(&mut swarm, &mut state);
            }
            // #42: connectivity watchdog + debounced reactive rebootstrap.
            _ = heal_ticker.tick() => {
                drive_heal(&mut swarm, &mut state);
                drive_lan_dials(&mut swarm, &mut state);
                if state.last_idle_sweep.elapsed() >= crate::kad_idle::SWEEP_EVERY {
                    state.last_idle_sweep = std::time::Instant::now();
                    drive_kad_idle_close(&mut swarm, &mut state);
                }
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

/// Whether a one-time readiness step (`seen`) happens now for the first time with an announce
/// already made before it; marks it seen either way.
fn first_time(seen: &mut bool, announced: bool) -> bool {
    !std::mem::replace(seen, true) && announced
}

/// A gossip load counter's value (see `openhydra_counters` in the patched gossip library).
fn gossip_count(c: &std::sync::atomic::AtomicU64) -> u64 {
    c.load(std::sync::atomic::Ordering::Relaxed)
}

/// Tell the provider to re-announce (it re-announces when the generation advances).
fn bump_net_generation(state: &LoopState) {
    state.net_generation.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    state.announced = true;
    // Signed with the time it was made, so a newer announce replaces an older one everywhere and a
    // replayed old one is refused (GPU test 2026-10-29: nothing set this, so every record was dated
    // 1970 and the hour-old limit refused them all).
    let mut record = record.clone();
    record.updated_unix_ms = now_unix_ms();
    let record = &record;
    // Task 6.2: sign the record with our Ed25519 keypair.
    let signed_record = match dht::sign_peer_record(record, keypair) {
        Ok(r) => r,
        Err(e) => {
            warn!("announce sign failed: {e}");
            // Fall back to unsigned record.
            record.clone()
        }
    };
    // Simulator only (`oh_sim --announce-gossip-only`): no Kademlia announcement, only gossip.
    if SIM_ANNOUNCE_GOSSIP_ONLY.load(std::sync::atomic::Ordering::Relaxed) {
        install_known_peer(state, &signed_record, now_unix_ms());
        info!(model_id = %signed_record.model_id, peer_id = %signed_record.peer_id, "announced by gossip only (simulator)");
        publish_provider_pex(swarm, &signed_record);
        let _ = reply.send(Ok(()));
        return;
    }
    // Simulator only (`oh_sim --kad-slow SECS`): Kademlia as a slow index. The same record (apart
    // from its time and load) announced to Kademlia within the interval goes by gossip only.
    if let Some(every) = crate::swarm::kad_slow_interval() {
        let fingerprint = {
            use std::hash::{Hash, Hasher};
            let mut what = record.clone();
            what.updated_unix_ms = 0;
            what.signature = String::new();
            what.load_pct = 0.0;
            what.queue_depth = 0;
            what.throughput_tok_s = 0.0;
            let mut h = std::collections::hash_map::DefaultHasher::new();
            serde_json::to_string(&what).unwrap_or_default().hash(&mut h);
            h.finish()
        };
        let fresh = state
            .kad_announced
            .get(&signed_record.model_id)
            .is_some_and(|(fp, at)| *fp == fingerprint && at.elapsed() < every);
        if fresh {
            install_known_peer(state, &signed_record, now_unix_ms());
            info!(model_id = %signed_record.model_id, peer_id = %signed_record.peer_id, "announced by gossip only (Kademlia record still current)");
            publish_provider_pex(swarm, &signed_record);
            let _ = reply.send(Ok(()));
            return;
        }
        state.kad_announced.insert(signed_record.model_id.clone(), (fingerprint, std::time::Instant::now()));
    }
    let local_libp2p_id = swarm.local_peer_id().to_base58();
    let key = dht::peer_record_key(&signed_record.model_id, &local_libp2p_id);
    match dht::encode_record(&signed_record) {
        Ok(value) => {
            let kad_record = kad::Record {
                key,
                value,
                publisher: Some(*swarm.local_peer_id()),
                expires: Some(std::time::Instant::now() + crate::swarm::kad_record_ttl()),
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
                    install_known_peer(state, &signed_record, now_unix_ms());
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

static SIM_ANNOUNCE_GOSSIP_ONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Simulator only (`oh_sim --announce-gossip-only`): providers announce over gossip only, with no
/// Kademlia record or provider record, to measure what Kademlia announcements cost. A process-wide
/// setting like [`sim_treat_172_16_as_public`]; not reachable from the agent's command line.
pub fn sim_announce_gossip_only() {
    SIM_ANNOUNCE_GOSSIP_ONLY.store(true, std::sync::atomic::Ordering::Relaxed);
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

/// M6: publish a `MODEL_WITHDRAWN` tombstone for `model_id` served by this node. The
/// message is authored by our (signed) gossip identity, so a receiver can trust that the
/// author == the `libp2p_peer_id` being withdrawn and evict only that `(peer, model)`
/// entry. Best-effort — an `InsufficientPeers` publish error before the mesh forms is
/// swallowed (the model still expires on the discover-record TTL), matching the
/// [`publish_provider_pex`] contract.
fn handle_withdraw_model(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    model_id: &str,
    reply: oneshot::Sender<Result<(), String>>,
) {
    let envelope = serde_json::json!({
        "type": MODEL_WITHDRAWN_TYPE,
        "libp2p_peer_id": swarm.local_peer_id().to_base58(),
        "model_id": model_id,
        "timestamp": now_unix_ms() / 1000,
    });
    let payload = match serde_json::to_vec(&envelope) {
        Ok(p) => p,
        Err(e) => {
            let _ = reply.send(Err(format!("withdraw encode: {e}")));
            return;
        }
    };
    let topic = libp2p::gossipsub::IdentTopic::new(crate::swarm::GOSSIPSUB_TOPIC);
    match swarm.behaviour_mut().gossipsub.publish(topic, payload) {
        Ok(_) => {
            info!(%model_id, "published MODEL_WITHDRAWN tombstone");
            let _ = reply.send(Ok(()));
        }
        // InsufficientPeers (no mesh yet) is not an error worth surfacing: the model
        // will fall off consumers' lists on the normal TTL. Report success so the
        // provider's re-announce loop doesn't treat a benign no-mesh as a failure.
        Err(e) => {
            debug!(%e, %model_id, "MODEL_WITHDRAWN publish skipped (no mesh peers)");
            let _ = reply.send(Ok(()));
        }
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
) -> bool {
    // The verified gossip author. Strict signing guarantees this is present for
    // accepted messages, but guard anyway — no author means no trust anchor.
    let source_str = match source {
        Some(pid) => pid.to_base58(),
        None => {
            debug!("provider_pex: dropping advert with no signed author");
            return false;
        }
    };
    let record: PeerRecord = match parsed.get("record") {
        Some(v) => match serde_json::from_value(v.clone()) {
            Ok(r) => r,
            Err(e) => {
                debug!(%e, "provider_pex: malformed record in advert");
                return false;
            }
        },
        None => {
            debug!("provider_pex: advert missing record field");
            return false;
        }
    };
    // The record we already hold, again (each provider's announce reaches us by several paths):
    // nothing to check or change (#9: re-checking its signature took 19% of a relay's CPU at
    // saturation). Still only from its own author, and only while within the age a record is
    // accepted at. Not an undated record (an older node's): its every receipt renews it (see
    // `install_known_peer_in`), which skipping would stop (review 19 Nov).
    let now = now_unix_ms();
    // For measuring how often providers change (run with RUST_LOG=openhydra_pex_observe=debug):
    // every announcement received, with a fingerprint of what it says apart from its time.
    if tracing::enabled!(target: "openhydra_pex_observe", tracing::Level::DEBUG) {
        let mut what = record.clone();
        what.updated_unix_ms = 0;
        what.signature = String::new();
        what.load_pct = 0.0;
        what.queue_depth = 0;
        what.throughput_tok_s = 0.0;
        let fingerprint = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            serde_json::to_string(&what).unwrap_or_default().hash(&mut h);
            h.finish()
        };
        tracing::debug!(target: "openhydra_pex_observe", peer = %record.libp2p_peer_id, model = %record.model_id,
            updated_ms = record.updated_unix_ms, fingerprint = format!("{fingerprint:016x}"), bytes = parsed.to_string().len(), "announce");
    }
    if record.updated_unix_ms != 0
        && record.libp2p_peer_id == source_str
        && dht::check_record_age(&record, now).is_ok()
        && state.known_peers.get(&known_peer_key(&record)).is_some_and(|held| *held == record)
    {
        return true;
    }
    if let Err(e) = dht::pex_record_is_authentic(&record, &source_str) {
        warn!(%source_str, %e, "provider_pex: rejected advert (failed authenticity)");
        return false;
    }
    let Some(is_new) = install_known_peer(state, &record, now_unix_ms()) else {
        debug!(%source_str, "provider_pex: rejected advert (dated ahead, over an hour old, or older than the one held)");
        return false;
    };
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
    true
}

/// M6/F1: is a `MODEL_WITHDRAWN` timestamp (unix seconds) close enough to `now_secs` to honour?
/// Two-sided so clock skew in either direction is tolerated; `abs_diff` can't underflow. Pure, so
/// the replay-window policy is unit-testable without a swarm.
fn withdrawal_is_fresh(timestamp_secs: u64, now_secs: u64, max_age_secs: u64) -> bool {
    now_secs.abs_diff(timestamp_secs) <= max_age_secs
}

/// M6: evict every `known_peers` entry for `(libp2p_peer_id, model_id)` — the data half of
/// handling a `MODEL_WITHDRAWN` tombstone, kept pure so the eviction is unit-testable without a
/// swarm (the author-authenticity check stays at the call site, where `message.source` lives).
/// Returns how many entries were removed (0 = we never knew this model from this peer). Matches
/// on the record fields, not the map key, so it's independent of the `known_peer_key` scheme.
fn evict_withdrawn_model(
    known_peers: &mut HashMap<String, PeerRecord>,
    libp2p_peer_id: &str,
    model_id: &str,
) -> usize {
    let before = known_peers.len();
    known_peers
        .retain(|_, record| !(record.libp2p_peer_id == libp2p_peer_id && record.model_id == model_id));
    before - known_peers.len()
}

/// How long a withdrawn model is remembered (as long as the gateway's sticky model list keeps one).
pub const WITHDRAWN_MODEL_MEMORY: std::time::Duration = std::time::Duration::from_secs(300);

/// Count one bootstrap's `PROVIDER_GONE` for `provider`, sent at `report_ms` (its signed time);
/// `Some((last_record_ms, first_report_ms))` once `quorum` distinct bootstraps have reported it (the
/// reports are then forgotten). The first report's own time, not when it arrived, is the cutoff for
/// undated records: a replayed report (any peer can re-send one within its freshness window) then
/// can't evict what the provider announced after it (review 2026-10-30). Pure.
fn note_provider_gone(
    pending: &mut HashMap<PeerId, (Vec<PeerId>, u64, u64)>,
    provider: PeerId,
    reporter: PeerId,
    last_record_ms: u64,
    quorum: usize,
    now_ms: u64,
    report_ms: u64,
) -> Option<(u64, u64)> {
    pending.retain(|_, (_, _, at)| now_ms.saturating_sub(*at) <= PROVIDER_GONE_MAX_AGE_SECS * 1000);
    if !pending.contains_key(&provider) && pending.len() >= MAX_PROVIDER_GONE_PENDING {
        if let Some(oldest) = pending.iter().min_by_key(|(_, (_, _, at))| *at).map(|(p, _)| *p) {
            pending.remove(&oldest);
        }
    }
    let report_ms = report_ms.min(now_ms);
    let entry = pending.entry(provider).or_insert_with(|| (Vec::new(), 0, report_ms));
    entry.2 = entry.2.min(report_ms);
    if !entry.0.contains(&reporter) {
        entry.0.push(reporter);
    }
    entry.1 = entry.1.max(last_record_ms);
    if entry.0.len() >= quorum.max(1) {
        let (_, last, first) = pending.remove(&provider)?;
        return Some((last, first));
    }
    None
}

/// Drop `provider`'s cached records that a quorum of bootstraps reported gone: those no newer than
/// the newest record the bootstraps held (`last_record_ms`), and undated ones not received since
/// the first report (`first_at_ms`) — a provider that is in fact alive has announced since, and its
/// newer record stays. Returns the models dropped. Pure.
fn evict_gone_provider(
    known: &mut HashMap<String, PeerRecord>,
    received: &HashMap<String, u64>,
    provider: &str,
    last_record_ms: u64,
    first_at_ms: u64,
) -> Vec<String> {
    let mut models = Vec::new();
    known.retain(|key, r| {
        let gone = r.libp2p_peer_id == provider
            && if r.updated_unix_ms == 0 {
                received.get(key).map_or(true, |at| *at <= first_at_ms)
            } else {
                r.updated_unix_ms <= last_record_ms
            };
        if gone {
            models.push(r.model_id.clone());
        }
        !gone
    });
    models.sort();
    models.dedup();
    models
}

/// At most this many models remembered as withdrawn, and (model, provider) announcements remembered
/// for it (review 2026-10-14, L7: the map only shrank when a gateway listed its models).
const MAX_WITHDRAWN_MODELS: usize = 4096;
const MAX_MODEL_SEEN: usize = 32_768;

/// Note that `peer` announced `model` (now): see `LoopState::model_seen`.
fn note_model_seen(seen: &mut HashMap<(String, String), std::time::Instant>, model: &str, peer: &str, now: std::time::Instant) {
    let key = (model.to_string(), peer.to_string());
    if !seen.contains_key(&key) && seen.len() >= MAX_MODEL_SEEN {
        seen.retain(|_, at| now.duration_since(*at) <= WITHDRAWN_MODEL_MEMORY);
        if seen.len() >= MAX_MODEL_SEEN {
            return; // full of fresh announcements: this one isn't remembered
        }
    }
    seen.insert(key, now);
}

/// After a withdrawal by `peer` evicted `model`: remember it if no other provider serves it
/// (then it has really left the network, rather than lost one of its providers) - none known now,
/// and none announced it within [`WITHDRAWN_MODEL_MEMORY`] (its record may have lapsed here). The
/// withdrawing peer's own announcement is forgotten: it no longer serves the model, so when the last
/// of several providers withdraws, the model goes at once (second review 2026-10-19).
fn note_withdrawn_model(
    withdrawn: &mut HashMap<String, std::time::Instant>,
    known_peers: &HashMap<String, PeerRecord>,
    seen: &mut HashMap<(String, String), std::time::Instant>,
    model: &str,
    peer: &str,
    now: std::time::Instant,
) {
    seen.remove(&(model.to_string(), peer.to_string()));
    if known_peers.values().any(|r| r.model_id == model) {
        return;
    }
    let others_lately = seen
        .iter()
        .any(|((m, p), at)| m == model && p != peer && now.duration_since(*at) <= WITHDRAWN_MODEL_MEMORY);
    if others_lately {
        return;
    }
    withdrawn.retain(|_, at| now.duration_since(*at) <= WITHDRAWN_MODEL_MEMORY);
    if withdrawn.len() >= MAX_WITHDRAWN_MODELS {
        if let Some(oldest) = withdrawn.iter().min_by_key(|(_, at)| **at).map(|(m, _)| m.clone()) {
            withdrawn.remove(&oldest);
        }
    }
    withdrawn.insert(model.to_string(), now);
}

/// The remembered withdrawals still within [`WITHDRAWN_MODEL_MEMORY`] whose model no known provider
/// serves now (a re-share brings it back); older ones are forgotten.
fn recently_withdrawn(
    withdrawn: &mut HashMap<String, std::time::Instant>,
    known_peers: &HashMap<String, PeerRecord>,
    now: std::time::Instant,
) -> Vec<String> {
    withdrawn.retain(|_, at| now.duration_since(*at) <= WITHDRAWN_MODEL_MEMORY);
    let mut out: Vec<String> =
        withdrawn.keys().filter(|m| !known_peers.values().any(|r| &r.model_id == *m)).cloned().collect();
    out.sort();
    out
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

/// How [`handle_discover`] answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiscoverMode {
    /// A recent lookup is cached: answer from the known providers, no new lookup.
    CacheHit,
    /// Answer from the known providers now and run a lookup in the background.
    AnswerThenRefresh,
    /// Nothing known, or the caller just found every known provider failing: wait for the lookup.
    WaitForLookup,
}

fn discover_mode(providers_known: bool, cache_fresh: bool, wait_for_lookup: bool) -> DiscoverMode {
    match (providers_known, wait_for_lookup, cache_fresh) {
        (false, _, _) | (true, true, _) => DiscoverMode::WaitForLookup,
        (true, false, true) => DiscoverMode::CacheHit,
        (true, false, false) => DiscoverMode::AnswerThenRefresh,
    }
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
    let mode = discover_mode(
        !seed.is_empty(),
        discover_cache_fresh(&state.discover_cache, model_id, now_ms, cur_gen, DISCOVER_CACHE_TTL_MS),
        state.discover_wait_for_lookup.contains_key(model_id),
    );
    if mode == DiscoverMode::CacheHit {
        let peers = discovered_from_records(state, &seed);
        debug!(model_id, n = peers.len(), "discover: C11 cache hit — skipped get_providers");
        let _ = reply.send(Ok(peers));
        return;
    }
    // Providers already known (gossip, an earlier lookup): answer with them now and refresh in the
    // background. Waiting for get_providers here held the request until the query's first result,
    // up to its 10 s timeout when the DHT held no provider record, though the answer was in hand
    // (2026-10-27). A stale entry is covered by the consumer's failover, as for a cache hit.
    let running = state.pending_discovers.iter().find(|(_, p)| p.model_id == model_id).map(|(id, _)| *id);
    let waiter = if mode == DiscoverMode::AnswerThenRefresh {
        let peers = discovered_from_records(state, &seed);
        debug!(model_id, n = peers.len(), "discover: answered from known providers, refreshing in the background");
        let _ = reply.send(Ok(peers));
        if running.is_some() {
            return; // a lookup for this model is already running
        }
        None
    } else {
        let kind = if seed.is_empty() { DiscoverKind::Reply } else { DiscoverKind::Retry };
        let deadline = (kind == DiscoverKind::Retry).then(|| std::time::Instant::now() + RETRY_SETTLE);
        Some(DiscoverWaiter { reply, kind, info_at_join: 0, deadline })
    };

    // One lookup per model: join the one running (review 2026-10-29).
    let waiter = match (running, waiter) {
        (Some(id), Some(mut waiter)) => {
            let pending = state.pending_discovers.get_mut(&id).expect("present");
            for record in seed {
                pending.merge(record, now_ms);
            }
            waiter.info_at_join = pending.info_gen;
            debug!(model_id, "discover: joined the lookup already running");
            pending.waiters.push(waiter);
            maybe_reply_discover(state, id);
            return;
        }
        (_, waiter) => waiter,
    };

    let key = dht::model_provider_key(model_id);
    let query_id = swarm.behaviour_mut().kademlia.get_providers(key);
    state.discover_seq += 1;
    state.pending_discovers.insert(
        query_id,
        PendingDiscover {
            model_id: model_id.to_string(),
            records: seed,
            providers_done: false,
            outstanding: 0,
            info_gen: 0,
            waiters: waiter.into_iter().collect(),
            seq: state.discover_seq,
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

/// A peer we hold no live connection to failed an outgoing dial: drop it from the Kademlia
/// routing table so routing stops preferring that dead path (a later serve re-discovers it on
/// demand). (No-op if the peer is in fact still connected.)
///
/// It deliberately does **NOT** touch `known_peers`. An outgoing dial to a NAT'd / relayed
/// provider fails routinely (DCUtR attempts, relay churn, transient direct-dial failures), and
/// wiping the cache here dropped the provider's *entire* model set on every such hiccup — so a
/// multi-model provider collapsed to ~1 model in the browse list / desktop Models view (regression
/// vs v0.3.13, which only evicted after a 5-dial retry ladder). `known_peers` has its own liveness
/// reaper (retain-if-connected-or-fresh, `KNOWN_PEER_TTL_MS`) kept warm by PEX, so a genuinely
/// departed peer still ages out within the TTL while a transiently-unreachable one survives.
///
/// The address book (addresses kept for our own reconnects) loses only `failed`, the addresses a dial
/// just tried and couldn't reach: a ping failure, Kademlia finding no route, or a single-address dial
/// (a hole-punch, a reversal) failing says nothing about the peer's other addresses, which are what a
/// reconnect after a drop needs (review 2026-10-29: they were all wiped).
fn evict_unreachable_peer(swarm: &mut libp2p::Swarm<OpenHydraBehaviour>, pid: PeerId, failed: &[Multiaddr]) {
    if swarm.is_connected(&pid) {
        return;
    }
    swarm.behaviour_mut().kademlia.remove_peer(&pid);
    for addr in failed {
        swarm.behaviour_mut().address_book.remove(&pid, addr);
    }
    debug!(%pid, "removed unreachable peer from the routing table (known_peers left to the TTL reaper)");
}

/// The addresses a failed dial tried, as libp2p reports them (only transport failures name them).
fn dial_error_addrs(error: &libp2p::swarm::DialError) -> Vec<Multiaddr> {
    match error {
        libp2p::swarm::DialError::Transport(errs) => errs.iter().map(|(a, _)| a.clone()).collect(),
        _ => Vec::new(),
    }
}

/// F-5: backoff schedule (in ms) for relay-RESERVATION retries.
///
/// ``attempt`` is 1-based (the attempt number we just made / are scheduling
/// the wait after). Fast initial retries then exponential backoff, capped at
/// 120 s: 1s, 2s, 4s, 8s, 16s, 32s, 64s, 120s × 5, then every 10 minutes from the 13th
/// attempt. So a flapping or down relay neither tight-loops the event loop nor strands us
/// permanently — we keep probing, every 2 minutes for a while, then every 10, and recover
/// within a couple seconds of a brief blip. Pure + unit-tested.
fn reservation_retry_delay_ms(attempt: u32) -> u64 {
    // A relay that has refused us this often (about 13 minutes of retries) may never take us — a
    // private bootstrap that doesn't run a relay, or one at its limit: try it every 10 minutes from
    // then on, which still finds it once it's back.
    if attempt > RESERVATION_RETRIES_BEFORE_SLOW {
        return 600_000;
    }
    // Cap the shift so ``1 << shift`` can't overflow and the value saturates
    // at the 120 s ceiling well before then.
    let shift = attempt.saturating_sub(1).min(7);
    let ms = 1000u64.saturating_mul(1u64 << shift);
    ms.min(120_000)
}

/// Reservation retries on one relay after which they slow to every 10 minutes and are logged once
/// an hour (see [`reservation_retry_delay_ms`]).
const RESERVATION_RETRIES_BEFORE_SLOW: u32 = 12;

/// Whether retry `attempt` is worth a warning: the first [`RESERVATION_RETRIES_BEFORE_SLOW`], then
/// one an hour.
fn reservation_retry_loud(attempt: u32) -> bool {
    attempt <= RESERVATION_RETRIES_BEFORE_SLOW || attempt % 6 == 0
}

/// F-5: record that a relay reservation for ``circuit_addr`` is lost and
/// schedule the next retry with backoff. Idempotent-ish: advances the attempt
/// counter and pushes the next-attempt deadline out per the backoff schedule.
/// For a node whose relays are only its own bootstraps (not on OpenHydra's public network): when a
/// reservation on one fails, what to tell its operator (review 2026-10-14, L9). A bootstrap that
/// doesn't run a relay leaves a node behind NAT unreachable, and nothing said so. `None` on the
/// public network (OpenHydra's relays are there too).
fn private_relay_hint(relays: &[Multiaddr], circuit: &Multiaddr, reason: &str) -> Option<String> {
    let production = crate::relay::bootstrap_relay_peer_ids();
    if crate::relay::relay_peer_ids(relays).iter().any(|p| production.contains(p)) {
        return None;
    }
    let relay: Multiaddr = circuit.iter().take_while(|p| !matches!(p, libp2p::multiaddr::Protocol::P2pCircuit)).collect();
    Some(format!(
        "couldn't reserve a relay slot on {relay} ({reason}). This node relays only through its own bootstraps (none is OpenHydra's): \
         if they don't run a relay, peers behind NAT can't reach it. Run them as openhydra-bootstrap (which relays; \
         on a private network give it --external with the address peers reach it at), or pass --public-relays on to use \
         OpenHydra's relays too."
    ))
}

/// The relay a circuit address goes through: the peer id just before `/p2p-circuit`.
fn circuit_relay(addr: &Multiaddr) -> Option<PeerId> {
    let mut last = None;
    for p in addr.iter() {
        match p {
            libp2p::multiaddr::Protocol::P2p(id) => last = Some(id),
            libp2p::multiaddr::Protocol::P2pCircuit => return last,
            _ => {}
        }
    }
    None
}

/// The circuit to reserve on `relay` at: its first address in `relays` (the one startup uses;
/// v4 is listed first and is universally reachable), else `seen` cut at `/p2p-circuit`.
fn reservation_circuit(relays: &[Multiaddr], relay: PeerId, seen: Option<&Multiaddr>) -> Option<Multiaddr> {
    let circuit = libp2p::multiaddr::Protocol::P2pCircuit;
    relays
        .iter()
        .find(|a| a.iter().any(|p| p == libp2p::multiaddr::Protocol::P2p(relay)))
        .map(|a| a.clone().with(circuit.clone()))
        .or_else(|| {
            let seen = seen?;
            let mut out = Multiaddr::empty();
            for p in seen.iter() {
                let done = p == circuit;
                out.push(p);
                if done {
                    return Some(out);
                }
            }
            None
        })
}

fn relay_has_listener(state: &LoopState, relay: PeerId) -> bool {
    state.relay_listeners.values().any(|r| *r == relay)
}

/// Ask `relay` for a reservation at `circuit`, unless one of our listeners already holds or is
/// requesting one there: a second would make libp2p close one of them (the F6 hazard). Returns
/// whether a request went out.
fn request_reservation(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    relay: PeerId,
    circuit: Multiaddr,
    why: &'static str,
) -> bool {
    if relay_has_listener(state, relay) {
        return false;
    }
    match swarm.listen_on(circuit.clone()) {
        Ok(id) => {
            state.relay_listeners.insert(id, relay);
            info!(addr = %circuit, why, "listening via relay (reservation requested)");
            true
        }
        Err(e) => {
            warn!(addr = %circuit, error = %e, why, "relay listen failed");
            schedule_reservation_retry(state, relay, circuit);
            false
        }
    }
}

/// The relays to reserve on now and the circuit for each: every relay in `state.relays` with no
/// listener of ours on it, once, at its first listed address.
fn relays_to_reserve(state: &LoopState) -> Vec<(PeerId, Multiaddr)> {
    let mut out: Vec<(PeerId, Multiaddr)> = Vec::new();
    for relay_multiaddr in &state.relays {
        let Some(relay) = relay_multiaddr.iter().find_map(|p| match p {
            libp2p::multiaddr::Protocol::P2p(id) => Some(id),
            _ => None,
        }) else {
            continue;
        };
        if relay_has_listener(state, relay) || out.iter().any(|(r, _)| *r == relay) {
            continue;
        }
        out.push((relay, relay_multiaddr.clone().with(libp2p::multiaddr::Protocol::P2pCircuit)));
    }
    out
}

/// Reserve once on every relay in `state.relays` that we don't hold or request a reservation on.
fn reserve_on_relays(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    why: &'static str,
) {
    for (relay, circuit) in relays_to_reserve(state) {
        request_reservation(swarm, state, relay, circuit, why);
    }
}

fn schedule_reservation_retry(state: &mut LoopState, relay: PeerId, circuit: Multiaddr) {
    let now = tokio::time::Instant::now();
    let entry = state
        .relay_reservation_retries
        .entry(relay)
        .or_insert((0, now, circuit));
    entry.0 = entry.0.saturating_add(1);
    let delay = reservation_retry_delay_ms(entry.0);
    entry.1 = now + std::time::Duration::from_millis(delay);
    if reservation_retry_loud(entry.0) {
        warn!(
            circuit = %entry.2, attempt = entry.0, backoff_ms = delay,
            "relay reservation lost — retry scheduled with backoff (F-5)",
        );
    } else {
        debug!(circuit = %entry.2, attempt = entry.0, backoff_ms = delay, "relay reservation retry scheduled (F-5)");
    }
}

/// A circuit listener closed: which relays now need a reservation (the ones it was on, if no other
/// listener of ours is on them), with the circuit to reserve at. A listener that closes before
/// reporting an address (a refused or duplicate request) is known by its id.
fn relays_to_re_reserve(
    state: &mut LoopState,
    listener: libp2p::core::transport::ListenerId,
    addresses: &[Multiaddr],
) -> Vec<(PeerId, Multiaddr)> {
    let mut relays: Vec<(PeerId, Option<Multiaddr>)> = Vec::new();
    if let Some(relay) = state.relay_listeners.remove(&listener) {
        relays.push((relay, None));
    }
    for addr in addresses {
        if let Some(relay) = circuit_relay(addr) {
            match relays.iter_mut().find(|(r, _)| *r == relay) {
                Some((_, seen)) => {
                    seen.get_or_insert_with(|| addr.clone());
                }
                None => relays.push((relay, Some(addr.clone()))),
            }
        }
    }
    relays
        .into_iter()
        .filter(|(relay, _)| !relay_has_listener(state, *relay))
        .filter_map(|(relay, seen)| Some((relay, reservation_circuit(&state.relays, relay, seen.as_ref())?)))
        .collect()
}

/// F-5: re-issue any due relay-reservation ``listen_on`` requests. Driven by
/// the existing `relay_retry_ticker` (250 ms). A relay is retried only once
/// its scheduled deadline passes, so repeated failure events coalesce into a
/// single spaced attempt instead of a tight loop; and not while a listener of
/// ours is still on it.
fn drive_reservation_retries(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
) {
    if state.relay_reservation_retries.is_empty() {
        return;
    }
    let now = tokio::time::Instant::now();
    for (relay, circuit) in due_reservation_retries(state, now) {
        if request_reservation(swarm, state, relay, circuit, "F-5 retry") {
            // Wait for the outcome: success clears the entry, a failure closes the listener,
            // which schedules the next (longer) wait. Until then nothing is due.
            if let Some(entry) = state.relay_reservation_retries.get_mut(&relay) {
                entry.1 = now + std::time::Duration::from_millis(reservation_retry_delay_ms(entry.0));
            }
        }
    }
}

/// Failed pings, as libp2p reports them, after which a connection to one of our relays counts as
/// dead. libp2p keeps the first failure on a connection silent (libp2p-ping handler.rs), so 2
/// reported means 3 failed pings in a row: with 15 s pings and a 20 s timeout, about a minute.
const RELAY_PING_FAILURES_BEFORE_CLOSE: u32 = 2;

/// One of our direct connections to a relay.
#[derive(Debug, Clone)]
struct RelayConn {
    since: std::time::Instant,
    /// The IP the relay sees us at on it (identify's observed address), and whether that was one of
    /// our own interface addresses at the time (no NAT in between).
    seen_as: Option<(std::net::IpAddr, bool)>,
}

/// The IPs of our listen addresses (our interface addresses).
fn our_listen_ips(swarm: &libp2p::Swarm<OpenHydraBehaviour>) -> std::collections::HashSet<std::net::IpAddr> {
    swarm.listeners().filter_map(multiaddr_ip).collect()
}

/// Whether relay connection `connection` runs over a network this node has left: the relay saw us
/// at one of our own addresses and it isn't ours any more (no NAT: IPv6, or a public IPv4 on the
/// interface), or a newer relay connection sees us from another IPv4 /16 (behind NAT, the public
/// address changed). A container bridge going down, or an IPv6 temporary address the connection
/// doesn't use expiring, changes neither, so it never counts (review 2026-10-29).
fn relay_conn_on_old_network(
    conns: &HashMap<libp2p::swarm::ConnectionId, RelayConn>,
    connection: libp2p::swarm::ConnectionId,
    ours: &std::collections::HashSet<std::net::IpAddr>,
) -> bool {
    let Some(RelayConn { since, seen_as: Some((seen, was_ours)) }) = conns.get(&connection) else {
        return false;
    };
    if *was_ours && !ours.contains(seen) {
        return true;
    }
    let std::net::IpAddr::V4(seen) = seen else { return false };
    let slash16 = |ip: &std::net::Ipv4Addr| (ip.octets()[0], ip.octets()[1]);
    conns.values().any(|c| {
        c.since > *since
            && matches!(c.seen_as, Some((std::net::IpAddr::V4(other), _)) if slash16(&other) != slash16(seen))
    })
}

/// Count a failed ping on `connection`; true when it should be closed: a direct connection to one of
/// our relays that has failed [`RELAY_PING_FAILURES_BEFORE_CLOSE`] reported pings in a row (a success
/// resets the count) and runs over a network we've left ([`relay_conn_on_old_network`]). After a
/// network change the relay connections made on the old network can sit for many minutes with every
/// ping failing, holding the reservations' listeners open, so no new reservation is made and nothing
/// reaches us through the relay. Failed pings alone don't close a connection: a slow uplink carrying a
/// big relayed result can make pings fail on a connection that still works, and closing it would kill
/// every circuit through the relay. A peer that doesn't speak ping says nothing.
fn relay_connection_dead(
    state: &mut LoopState,
    peer: PeerId,
    connection: libp2p::swarm::ConnectionId,
    failure: &libp2p::ping::Failure,
    ours: &std::collections::HashSet<std::net::IpAddr>,
) -> bool {
    if matches!(failure, libp2p::ping::Failure::Unsupported)
        || !crate::relay::relay_peer_ids(&state.relays).contains(&peer)
    {
        return false;
    }
    let n = state.relay_ping_failures.entry(connection).or_insert(0);
    *n += 1;
    if *n < RELAY_PING_FAILURES_BEFORE_CLOSE {
        return false;
    }
    let old = relay_conn_on_old_network(&state.relay_conns, connection, ours);
    if old {
        state.relay_ping_failures.remove(&connection);
    }
    old
}

/// How long a response being written counts as under way when libp2p never reports on it: longer
/// than any write (a big result over a slow uplink), so a busy connection isn't taken for idle.
const RESPONSE_WRITING_TTL: std::time::Duration = std::time::Duration::from_secs(1800);

/// Whether anything is under way with `peer`: a request of ours waiting for its answer
/// (`pending_outbound` asks libp2p about one of our request ids), a request of its waiting for ours,
/// a response still being written to it, or a forward queued for it.
fn peer_busy(state: &LoopState, peer: PeerId, pending_outbound: impl Fn(&request_response::OutboundRequestId) -> bool) -> bool {
    state.inbound_proxy_channels.values().any(|(_, _, p, _)| *p == peer)
        || state.responses_writing.values().any(|(p, _)| *p == peer)
        || state.pending_relay_forwards.iter().any(|(p, ..)| *p == peer)
        || state.pending_proxy.keys().any(|id| pending_outbound(id))
        || state.proxy_inflight.contains_key(&peer)
        || state.proxy_waiting.contains_key(&peer)
}

/// Count a failed ping on `connection`; true when it should be closed: a relayed connection that
/// has failed [`RELAY_PING_FAILURES_BEFORE_CLOSE`] reported pings in a row while nothing is under
/// way with its peer (`busy`, see [`peer_busy`]). When a provider moves to another network, the
/// circuit a consumer had to it ends on the relay's connection to the provider's old network,
/// which nothing closes for minutes; the consumer's next request rode it and timed out, and it was
/// replaced only ~2 min later (2026-10-29). A connection carrying a transfer can fail pings and
/// still work (pings queue behind the data), so a busy one is left to the request's own timeout;
/// closing an idle one costs nothing, the next request dials a fresh circuit. Connections to our
/// relays have their own rule ([`relay_connection_dead`]); a peer without ping says nothing.
fn relayed_connection_dead(
    state: &mut LoopState,
    peer: PeerId,
    connection: libp2p::swarm::ConnectionId,
    failure: &libp2p::ping::Failure,
    busy: bool,
) -> bool {
    if matches!(failure, libp2p::ping::Failure::Unsupported) || state.relayed_conns.get(&connection) != Some(&peer) {
        return false;
    }
    let n = state.relayed_ping_failures.entry(connection).or_insert(0);
    *n += 1;
    if *n < RELAY_PING_FAILURES_BEFORE_CLOSE || busy {
        return false;
    }
    state.relayed_ping_failures.remove(&connection);
    true
}

/// The relays whose retry is due at `now` and has no listener of ours on it yet.
fn due_reservation_retries(state: &LoopState, now: tokio::time::Instant) -> Vec<(PeerId, Multiaddr)> {
    state
        .relay_reservation_retries
        .iter()
        .filter(|(relay, (_, next, _))| now >= *next && !relay_has_listener(state, **relay))
        .map(|(relay, (_, _, circuit))| (*relay, circuit.clone()))
        .collect()
}

/// F-5: a circuit through a relay came up — the reservation on that relay succeeded, so stop
/// retrying it.
fn clear_reservation_retry(state: &mut LoopState, addr: &Multiaddr) {
    if let Some(relay) = circuit_relay(addr) {
        state.relay_reservation_retries.remove(&relay);
    }
}

/// P1: how long after dialling a relay over IPv6 QUIC we wait before trying again (a network that
/// blocks UDP fails every time; a new reservation after a network change is the usual next try).
const RELAY_V6_QUIC_REDIAL: std::time::Duration = std::time::Duration::from_secs(300);

/// The relay's own IPv6 QUIC address in one of our circuit addresses:
/// `/ip6/<global>/udp/<port>/quic-v1/p2p/<relay>/p2p-circuit…` gives `<relay>` and the part before
/// `/p2p-circuit`. A reservation reports a circuit for each address the relay gives for itself.
fn circuit_relay_v6_quic(addr: &Multiaddr) -> Option<(PeerId, Multiaddr)> {
    use libp2p::multiaddr::Protocol;
    let mut it = addr.iter();
    let Protocol::Ip6(ip) = it.next()? else { return None };
    if is_non_global_ip(std::net::IpAddr::V6(ip)) {
        return None;
    }
    let Protocol::Udp(port) = it.next()? else { return None };
    let Protocol::QuicV1 = it.next()? else { return None };
    let Protocol::P2p(relay) = it.next()? else { return None };
    let Protocol::P2pCircuit = it.next()? else { return None };
    let out = Multiaddr::empty()
        .with(Protocol::Ip6(ip))
        .with(Protocol::Udp(port))
        .with(Protocol::QuicV1)
        .with(Protocol::P2p(relay));
    Some((relay, out))
}

/// P1: the relay to dial over IPv6 QUIC, and where, now that our listener `listener` reported
/// circuit address `circuit`. DCUtR offers only addresses peers have seen us at (identify), and a
/// node's bootstrap and relay connections are TCP, so without an IPv6 QUIC connection to some peer
/// its IPv6 QUIC address is never offered and hole punching over IPv6 — the way through CGNAT —
/// never starts (diagnosis 2026-10-29: only nodes that happened to reach an inbound-reachable IPv6
/// peer had one). Dialled after the reservation is held, so the reservation stays on the
/// connection it was made on. Nothing when IPv6 doesn't work here, the listener isn't one of our
/// reservations, we already have an IPv6 QUIC connection to that relay, or we tried recently.
fn relay_v6_quic_dial(
    state: &LoopState,
    listener: libp2p::core::transport::ListenerId,
    circuit: &Multiaddr,
    now: std::time::Instant,
) -> Option<(PeerId, Multiaddr)> {
    if !state.ipv6_capable {
        return None;
    }
    let (relay, addr) = circuit_relay_v6_quic(circuit)?;
    if state.relay_listeners.get(&listener) != Some(&relay)
        || state.relay_v6_quic_conns.values().any(|r| *r == relay)
        || state.relay_v6_quic_dialled.get(&relay).is_some_and(|t| now.duration_since(*t) < RELAY_V6_QUIC_REDIAL)
    {
        return None;
    }
    Some((relay, addr))
}

/// Whether a direct connection's remote address is IPv6 QUIC.
fn is_v6_quic(addr: &Multiaddr) -> bool {
    use libp2p::multiaddr::Protocol;
    let mut it = addr.iter();
    matches!(it.next(), Some(Protocol::Ip6(_)))
        && matches!(it.next(), Some(Protocol::Udp(_)))
        && matches!(it.next(), Some(Protocol::QuicV1))
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

/// An address's transport as far as port reuse goes: its IP family (`None` for a DNS name that
/// can resolve to either), whether that IP is loopback, and whether it's QUIC (else TCP).
fn dial_transport(addr: &Multiaddr) -> Option<(Option<bool>, bool, bool)> {
    use libp2p::multiaddr::Protocol;
    let mut it = addr.iter();
    let (v6, loopback) = match it.next()? {
        Protocol::Ip4(ip) => (Some(false), ip.is_loopback()),
        Protocol::Ip6(ip) => (Some(true), ip.is_loopback()),
        Protocol::Dns4(_) => (Some(false), false),
        Protocol::Dns6(_) => (Some(true), false),
        Protocol::Dns(_) => (None, false),
        _ => return None,
    };
    let quic = match it.next()? {
        Protocol::Tcp(_) => false,
        Protocol::Udp(_) => matches!(it.next(), Some(Protocol::QuicV1)),
        _ => return None,
    };
    Some((v6, loopback, quic))
}

/// IPv4 ranges virtual interfaces use — Docker bridges (172.16.0.0/12), Tailscale and carrier NAT
/// (100.64.0.0/10), libvirt (192.168.122.0/24), macOS virtualization (192.168.64.0/24) — and
/// link-local: such an address coming or going says nothing about the network we're on.
pub(crate) fn is_virtual_v4(ip: std::net::Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_link_local()
        || (o[0] == 172 && (16..32).contains(&o[1]))
        || (o[0] == 100 && (64..128).contains(&o[1]))
        || (o[0] == 192 && o[1] == 168 && (o[2] == 122 || o[2] == 64))
}

/// Whether losing listen address `gone` means we left the network it was on, given the listen
/// addresses still up: an IPv4 address outside the virtual ranges when no other such IPv4 address
/// is left, or a global IPv6 address when no other one in its /64 is left. An IPv6 temporary
/// address rotating (its /64 stays) or a container bridge going down doesn't count (review
/// 2026-10-29: both made a rebootstrap redial bootstraps still connected, from ephemeral ports).
fn addr_loss_leaves_network(gone: &Multiaddr, remaining: &[Multiaddr]) -> bool {
    use std::net::IpAddr;
    let others = remaining.iter().filter_map(multiaddr_ip);
    match multiaddr_ip(gone) {
        Some(IpAddr::V4(ip)) if !ip.is_loopback() && !is_virtual_v4(ip) => {
            !others.into_iter().any(|r| matches!(r, IpAddr::V4(r) if r != ip && !r.is_loopback() && !is_virtual_v4(r)))
        }
        Some(IpAddr::V6(ip)) if !is_non_global_ip(IpAddr::V6(ip)) => {
            let net = |a: std::net::Ipv6Addr| a.segments()[..4].to_vec();
            !others.into_iter().any(|r| matches!(r, IpAddr::V6(r) if r != ip && net(r) == net(ip)))
        }
        _ => false,
    }
}

/// P2: the bootstraps a rebootstrap dials: those with no direct connection of ours on the same
/// transport and IP family made since we last lost an interface address (an older one may run over
/// a network we've left and be dead without knowing it yet). libp2p doesn't skip them (a dial from a bare address carries no peer
/// condition), and a second TCP connection to the same bootstrap from our listen port can't share
/// the first one's addresses and ports, so libp2p-tcp makes it from an ephemeral port instead
/// (lib.rs, AddrNotAvailable), whose observed address DCUtR then offers to peers (diagnosis
/// 2026-10-29).
fn bootstrap_redials(state: &LoopState) -> Vec<(PeerId, Multiaddr)> {
    state
        .bootstrap_peers
        .iter()
        .filter(|(peer, addr)| {
            let Some((v6, _, quic)) = dial_transport(addr) else { return true };
            !state.conn_remote_addrs.values().any(|(p, remote, since)| {
                p == peer
                    && state.last_addr_loss.map_or(true, |lost| *since > lost)
                    && dial_transport(remote).is_some_and(|(rv6, _, rquic)| rquic == quic && (v6.is_none() || rv6 == v6))
            })
        })
        .cloned()
        .collect()
}

fn start_kad_bootstrap(swarm: &mut libp2p::Swarm<OpenHydraBehaviour>) {
    if let Err(e) = swarm.behaviour_mut().kademlia.bootstrap() {
        warn!("kademlia bootstrap failed (no peers yet?): {e}");
    }
}

/// P2: how long startup waits for a TCP listen address to dial the bootstraps from before dialling
/// from a new port instead. The listen addresses usually come up within milliseconds.
const STARTUP_DIAL_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

/// P2: the startup bootstrap dials that can go now that TCP listen address `listen` is up: TCP
/// dials to the same IP family (any family for a DNS name) and the same loopback-ness, which
/// libp2p-tcp makes from `listen`'s port. libp2p-tcp starts reusing a wildcard listener's port
/// only once it reports the address, so a TCP dial made before that leaves from an ephemeral port
/// and the bootstraps see us there: DCUtR then offers peers that address, which nothing listens
/// on (diagnosis 2026-10-29). QUIC dials bind the listen socket at once and never wait.
/// Returns the dials that can go; `pending` keeps the rest.
fn startup_dials_ready(pending: &mut Vec<(PeerId, Multiaddr)>, listen: &Multiaddr) -> Vec<(PeerId, Multiaddr)> {
    let Some((Some(lv6), lloop, false)) = dial_transport(listen) else { return Vec::new() };
    let (ready, rest): (Vec<_>, Vec<_>) = pending.drain(..).partition(|(_, addr)| {
        matches!(dial_transport(addr), Some((v6, loopback, false)) if v6.map_or(true, |v6| v6 == lv6) && loopback == lloop)
    });
    *pending = rest;
    ready
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

    // Re-dial the bootstrap peers (relays included) to re-establish the direct links a roam/wake
    // tore down, skipping those still connected on that transport (P2, see `bootstrap_redials`).
    for (peer_id, addr) in bootstrap_redials(state) {
        let dial_addr = addr.clone().with(libp2p::multiaddr::Protocol::P2p(peer_id));
        if let Err(e) = swarm.dial_tracked(&mut state.dials_in_flight, dial_addr.clone()) {
            debug!(%peer_id, %dial_addr, %e, "rebootstrap: bootstrap re-dial failed");
        }
    }

    // Re-request only the relay reservations we don't hold or request already: reserving on a
    // relay we have a listener on would leak a duplicate (F6). (This used to compare against the
    // circuits we hold, which end in our own peer id, so it never matched and re-requested every
    // relay.)
    reserve_on_relays(swarm, state, "rebootstrap");

    // #43-W1: a new network is a fresh chance to hole-punch — clear the QUIC-v6
    // punch and reversal back-off so shelved peers get re-attempted.
    state.quic_holepunch_attempts.clear();
    state.reversal_attempts.clear();
    // P1: and to have our new IPv6 QUIC address seen by a relay.
    state.relay_v6_quic_dialled.clear();

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
) -> bool {
    if let Some(last) = state.last_rebootstrap {
        if last.elapsed() < REBOOTSTRAP_COOLDOWN {
            debug!(reason, "rebootstrap suppressed (cooldown)");
            return false;
        }
    }
    rebootstrap(swarm, state, reason);
    true
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
            if !maybe_rebootstrap(swarm, state, "reactive: interface change") {
                // A second change inside the cooldown (A → B → C within 30 s) still needs its
                // rebuild: run it once the cooldown ends rather than dropping it.
                if let Some(last) = state.last_rebootstrap {
                    state.pending_rebootstrap_at = Some(last + REBOOTSTRAP_COOLDOWN);
                }
            }
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
            if let libp2p::identify::Event::Received { peer_id, info, connection_id } = identify_event {
                debug!(%peer_id, protocol = %info.protocol_version, "identify received");
                // A peer we dialled because mDNS found it, that doesn't speak OpenHydra's protocol, is
                // another libp2p node on the LAN (mDNS is libp2p's shared service): drop that
                // connection and leave the peer alone for a while. Only our LAN dials: an inbound
                // connection, or a bootstrap (which doesn't serve that protocol), is never touched.
                let openhydra = info.protocols.iter().any(|p| *p == crate::proxy::PROXY_PROTOCOL);
                let same_network = crate::relay::remote_network_id(&info.agent_version) == our_network_id(state);
                if state.lan_dialer.is_lan_dial(&connection_id)
                    && !(openhydra && same_network)
                    && state.lan_dialer.foreign(&connection_id, std::time::Instant::now())
                {
                    if openhydra {
                        info!(%peer_id, net = crate::relay::remote_network_id(&info.agent_version), "LAN peer is on another OpenHydra network; dropping the connection mDNS made");
                    } else {
                        info!(%peer_id, "LAN peer isn't an OpenHydra node; dropping the connection mDNS made");
                    }
                    let _ = swarm.close_connection(connection_id);
                    return;
                }
                // A LAN peer of our network: tell it what we serve shortly, rather than at the next
                // re-announce (up to 2 minutes). Without a reachable bootstrap there's no other way
                // for it to learn (review 2026-10-29: a consumer that joined a LAN after the provider
                // waited 58 s for its first answer). Over the internet, peers find providers through
                // the DHT and the bootstraps' registry, so it's for LAN peers only.
                if openhydra && same_network {
                    state.lan_dialer.accepted(&connection_id);
                }
                if state.relay_conns.contains_key(&connection_id) {
                    let ours = our_listen_ips(swarm);
                    if let Some(c) = state.relay_conns.get_mut(&connection_id) {
                        c.seen_as = multiaddr_ip(&info.observed_addr).map(|ip| (ip, ours.contains(&ip)));
                    }
                }
                // A LAN peer is one mDNS lists *and* that reaches us over a private address. A private
                // address alone isn't enough: inside one cloud network, a company network or a
                // carrier's NAT every peer has one, and then every new connection re-published our
                // records to the whole network, at most every 10 s each (20 Nov: ~4 times the
                // announce traffic in a 1,500-node test, which with connection churn overloaded the
                // public nodes). Over the internet and in such networks, peers find providers through
                // the DHT, the bootstraps' registry and the periodic announce.
                let on_lan = state.local_link_peers.contains_key(&peer_id) && state.lan_dialer.knows(&peer_id);
                debug!(%peer_id, openhydra, same_network, on_lan, due = ?state.lan_announce.due, "identify: LAN re-announce check");
                // Once per peer while it stays connected: identify also runs every 5 minutes and on
                // every push, and each re-announce floods the whole network (review 2026-10-29).
                if openhydra && same_network && on_lan && state.lan_announced.insert(peer_id) {
                    note_lan_subscriber(&mut state.lan_announce, std::time::Instant::now());
                }
                let local_link = state.local_link_peers.contains_key(&peer_id);
                // What a peer claims about itself is not trusted (2026-10-27 review: any peer could
                // put up to 32 addresses of its choosing, other people's machines included, into our
                // routing table and so into every DHT answer we give). The routing table only gets
                // its circuit addresses through a relay we use (the D-S5 rule signed records follow),
                // which bounds them by the relays we trust; Kademlia lists a directly reachable peer
                // itself once it has dialled it.
                // A trusted relay's own addresses: places it's known at, for the circuit check.
                if trusted_relay_pids(state).contains_peer(&peer_id) {
                    let own = relay_self_addrs(&info.listen_addrs, on_public_network(state));
                    state.relay_self_addrs.insert(peer_id, own);
                }
                let trusted = trusted_relay_pids(state);
                // Only a peer that answers DHT requests belongs in the routing table. A client-mode
                // peer (a node behind NAT reachable only through its relay) doesn't list our Kad
                // protocol; putting its circuit address in anyway made every DHT query dial it
                // through the relay and hole-punch, for an answer that never came (20 Nov: ~250 such
                // peers filled public providers' tables within minutes of a 1,500-node start, and
                // the dials were part of a connection storm that ran them out of memory).
                let serves_dht = info.protocols.iter().any(|p| *p == crate::swarm::KAD_PROTOCOL);
                if crate::kad_idle::peer_supports(&info.agent_version) {
                    state.kad_idle_peers.insert(peer_id);
                } else {
                    state.kad_idle_peers.remove(&peer_id);
                }
                // Its relay circuits are kept for our own dials only (the address book), for every
                // peer: a DHT server is directly reachable, and its circuits in the routing table made
                // lookups dial it by relay and hand the circuits on in DHT answers (21 Nov: 23.6% of
                // Kademlia's dials had only relay addresses; the routing table refuses circuits now,
                // `swarm::kad_address_filter`). The direct-first dialler tries them only if the
                // direct addresses fail.
                for addr in identify_routing_addrs(&info.listen_addrs, &peer_id, &trusted, state.ipv6_capable) {
                    swarm.behaviour_mut().address_book.insert(peer_id, addr);
                }
                // Its direct addresses on the IP this connection really comes from are kept for our
                // own dials to this peer (a reconnect after a drop): the address book, which is never
                // put in DHT answers. (`Swarm::add_peer_address` stores nothing: 2026-10-29.) For a
                // DHT server, the globally reachable ones also go into the routing table, in place
                // of its circuits.
                let conn_addr = state.conn_remote_addrs.get(&connection_id).map(|(_, a, _)| a);
                let direct = identify_dial_addrs(&info.listen_addrs, conn_addr, local_link, state.ipv6_capable);
                for addr in kad_addrs_from_identify(serves_dht, on_public_network(state), &direct) {
                    swarm.behaviour_mut().kademlia.add_address(&peer_id, addr);
                }
                for addr in direct {
                    swarm.behaviour_mut().address_book.insert(peer_id, addr);
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
                    holepunch_v6_addrs(&info.listen_addrs)
                };
                // Stored for every peer, dialled only with a reason (they're the peer's own claims,
                // so dialling them for every peer that connects let anyone point us at other
                // machines, 2026-10-27 review): here, that it's a provider we know. A provider
                // dials back when a peer sends it a request (see `pursue_v6_holepunch`).
                let known_provider = {
                    let pid = peer_id.to_base58();
                    state.known_peers.values().any(|r| r.libp2p_peer_id == pid)
                };
                if store_claimed_v6_addrs(state, peer_id, &quic_v6_addrs) {
                    if known_provider {
                        pursue_v6_holepunch(swarm, state, peer_id, "identify");
                    }
                }
                // What the punch dialled, for connection reversal not to dial again.
                let quic_v6_addrs = if known_provider { quic_v6_addrs } else { Vec::new() };

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
                                    match swarm.dial_tracked(&mut state.dials_in_flight, opts) {
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
                    // Dial what mDNS finds, so peers on a LAN connect with no bootstrap reachable
                    // (review 2026-10-29: these were only logged). Not into the routing table by hand:
                    // on the public network the routing table refuses private addresses anyway (the
                    // patched libp2p-kad's filter), so none goes into a DHT answer.
                    for (peer_id, addr) in peers {
                        debug!(%peer_id, %addr, "mDNS discovered peer");
                        state.lan_dialer.discovered(peer_id, addr);
                    }
                    if state.lan_dialer.round_due(std::time::Instant::now()) {
                        drive_lan_dials(swarm, state);
                    }
                }
                // Phase 1.7: mDNS expiry — evict peers that are no longer
                // reachable on LAN.
                libp2p::mdns::Event::Expired(peers) => {
                    for (peer_id, addr) in peers {
                        info!(%peer_id, %addr, "mDNS address expired");
                        // Drop that one LAN route, not the whole peer: its other addresses (another
                        // interface, its public address, a relay) still work, and evicting the peer
                        // here dropped them too (review 2026-10-29). `known_peers` stays with the TTL
                        // reaper. Not from the routing table while connected: Kademlia drops a peer
                        // with its last address, and a missed mDNS refresh (common on WiFi) would
                        // drop a live one.
                        state.lan_dialer.expired(&peer_id, &addr);
                        swarm.behaviour_mut().address_book.remove(&peer_id, &addr);
                        if !swarm.is_connected(&peer_id) {
                            swarm.behaviour_mut().kademlia.remove_address(&peer_id, &addr);
                        }
                    }
                }
            }
        }

        // ── Relay Client ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::RelayClient(relay_event)) => {
            match relay_event {
                libp2p::relay::client::Event::ReservationReqAccepted {
                    relay_peer_id, renewal, ..
                } => {
                    info!(%relay_peer_id, "relay reservation accepted");
                    if !renewal && first_time(&mut state.reservation_seen, state.announced) {
                        info!(%relay_peer_id, "first relay reservation: re-announcing (the first announce went out before peers could reach us)");
                        bump_net_generation(state);
                    }
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
                message_id,
                message,
            } = gossip_event
            {
                // #7: passed on (gossipsub validates through us, `validate_messages`) only while
                // fresh enough that every node it reaches still remembers it; used here either way.
                let parsed = serde_json::from_slice::<serde_json::Value>(&message.data).ok();
                let forward = parsed.as_ref().map_or(true, |v| gossip_forwardable(v, now_unix_ms()));
                // Diagnostic (target oh_gossip_types): every delivered gossip message, by type, author,
                // model and the record's own time, with its sequence number.
                if tracing::enabled!(target: "oh_gossip_types", tracing::Level::DEBUG) {
                    let p = parsed.as_ref();
                    let ty = p.and_then(|v| v.get("type")).and_then(|v| v.as_str()).unwrap_or("?");
                    let rec = p.and_then(|v| v.get("record"));
                    let model = rec.and_then(|r| r.get("model_id")).and_then(|v| v.as_str()).unwrap_or("");
                    let upd = rec.and_then(|r| r.get("updated_unix_ms")).and_then(|v| v.as_u64()).unwrap_or(0);
                    debug!(target: "oh_gossip_types", ty, source = ?message.source, model, updated_ms = upd, seqno = ?message.sequence_number, via = %propagation_source, bytes = message.data.len(), "gossip in");
                }
                let acceptance = if forward {
                    libp2p::gossipsub::MessageAcceptance::Accept
                } else {
                    libp2p::gossipsub::MessageAcceptance::Ignore
                };
                let _ = swarm.behaviour_mut().gossipsub.report_message_validation_result(&message_id, &propagation_source, acceptance);
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
                if let Some(parsed) = parsed {
                    let msg_type = parsed.get("type").and_then(|v| v.as_str());
                    if msg_type == Some(PROVIDER_ANNOUNCE_TYPE) {
                        handled_internally = true;
                        if handle_provider_pex(state, &parsed, message.source) {
                            if let Some(source) = message.source {
                                pursue_v6_holepunch_debounced(swarm, state, source, "record");
                            }
                        }
                    } else if msg_type == Some("PEER_DEPARTED") {
                        if let Some(departed_id_str) = parsed.get("libp2p_peer_id").and_then(|v| v.as_str()) {
                            if let Ok(departed_pid) = departed_id_str.parse::<PeerId>() {
                                // #19 (review 19 Nov): taken only while fresh (a recorded one replayed
                                // later is ignored), and only the records it is newer than go: a
                                // provider that restarted under the same identity keeps the records
                                // it announced since.
                                let at_ms = parsed.get("timestamp").and_then(|v| v.as_u64()).map(|s| s.saturating_mul(1000));
                                let fresh = at_ms.is_some_and(|at| withdrawal_is_fresh(at / 1000, now_unix_ms() / 1000, PEER_DEPARTED_MAX_AGE_SECS));
                                if message.source == Some(departed_pid) && fresh {
                                    let at = at_ms.unwrap_or(0);
                                    let evicted = evict_departed(&mut state.known_peers, departed_id_str, at);
                                    if !state.known_peers.values().any(|r| r.libp2p_peer_id == departed_id_str) {
                                        swarm.behaviour_mut().kademlia.remove_peer(&departed_pid);
                                    }
                                    info!(%departed_pid, evicted, "evicted departed peer via PEER_DEPARTED gossip");
                                } else if message.source == Some(departed_pid) {
                                    debug!(%departed_pid, "ignored a stale or undated PEER_DEPARTED (replay?)");
                                } else {
                                    warn!(
                                        ?message.source, %departed_pid,
                                        "rejected PEER_DEPARTED: author does not match subject"
                                    );
                                }
                            }
                        }
                    } else if msg_type == Some(PROVIDER_GONE_TYPE) {
                        // SECURITY: only from a bootstrap we were configured with (gossip is signed
                        // and strict, so `message.source` is the verified author), only fresh, and
                        // only once PROVIDER_GONE_QUORUM of them agree; never for a peer we're
                        // connected to (we know better); and only records no newer than the
                        // bootstraps saw, so a live provider's next announce wins.
                        handled_internally = true;
                        let bootstraps: Vec<PeerId> = state
                            .relays
                            .iter()
                            .filter_map(|a| a.iter().find_map(|p| match p {
                                libp2p::multiaddr::Protocol::P2p(id) => Some(id),
                                _ => None,
                            }))
                            .collect::<std::collections::BTreeSet<_>>() // one per bootstrap, not per address
                            .into_iter()
                            .collect();
                        let subject = parsed.get("libp2p_peer_id").and_then(|v| v.as_str()).and_then(|s| s.parse::<PeerId>().ok());
                        let ts = parsed.get("timestamp").and_then(|v| v.as_u64());
                        let last_record_ms = parsed.get("last_record_ms").and_then(|v| v.as_u64()).unwrap_or(0);
                        match (message.source, subject, ts) {
                            (Some(reporter), Some(gone), Some(ts))
                                if bootstraps.contains(&reporter)
                                    && withdrawal_is_fresh(ts, now_unix_ms() / 1000, PROVIDER_GONE_MAX_AGE_SECS)
                                    && gone != *swarm.local_peer_id() =>
                            {
                                let mut quorum = PROVIDER_GONE_QUORUM.min(bootstraps.len());
                                quorum = quorum.max(1);
                                let now_ms = now_unix_ms();
                                if let Some((last, first)) = note_provider_gone(&mut state.provider_gone, gone, reporter, last_record_ms, quorum, now_ms, ts.saturating_mul(1000)) {
                                    if swarm.is_connected(&gone) {
                                        debug!(%gone, "PROVIDER_GONE ignored: we're connected to it");
                                    } else {
                                        let id = gone.to_base58();
                                        state.gone_providers.retain(|_, (_, at)| now_ms.saturating_sub(*at) < GONE_MEMORY_MS);
                                        if state.gone_providers.len() < MAX_PROVIDER_GONE_PENDING {
                                            state.gone_providers.insert(id.clone(), (last, first));
                                        }
                                        let mut models = evict_gone_provider(&mut state.known_peers, &state.known_received, &id, last, first);
                                        // And what it served that's no longer cached here (a disconnect
                                        // already dropped its records): the browse list's sticky cache
                                        // keeps those for minutes unless they're marked gone too (e2e
                                        // test 30 Oct: the consumer next to a killed provider listed it 180 s+).
                                        models.extend(state.model_seen.keys().filter(|(_, p)| *p == id).map(|(m, _)| m.clone()));
                                        models.sort();
                                        models.dedup();
                                        swarm.behaviour_mut().kademlia.remove_peer(&gone);
                                        if !models.is_empty() {
                                            info!(%gone, ?models, "evicted a provider the bootstraps report gone");
                                        }
                                        for m in &models {
                                            note_withdrawn_model(&mut state.withdrawn_models, &state.known_peers, &mut state.model_seen, m, &id, std::time::Instant::now());
                                        }
                                    }
                                }
                            }
                            (source, _, _) => debug!(?source, "PROVIDER_GONE ignored (not from a bootstrap of ours, stale or malformed)"),
                        }
                    } else if msg_type == Some(MODEL_WITHDRAWN_TYPE) {
                        // M6: a provider withdrawing a SINGLE model (it un-shared one but
                        // stays up serving the rest). Handle entirely in Rust — evict only
                        // the matching `(peer, model)` entry from `known_peers` so the model
                        // drops out of browse lists at once instead of on the TTL.
                        //
                        // SECURITY: identical trust model to PEER_DEPARTED above — signed +
                        // strict gossip makes `message.source` the verified author, and we
                        // require it to equal the `libp2p_peer_id` being withdrawn. A peer may
                        // only withdraw its OWN model; a forged withdrawal for another peer is
                        // rejected. Unlike PEER_DEPARTED we deliberately do NOT touch Kademlia:
                        // the peer is still alive and still serves its other models.
                        handled_internally = true;
                        let withdrawn_id = parsed.get("libp2p_peer_id").and_then(|v| v.as_str());
                        let withdrawn_model = parsed.get("model_id").and_then(|v| v.as_str());
                        // F1: the timestamp is required (our publisher always sets it, and it is
                        // signed) — a withdrawal without one is malformed/forged, so drop it.
                        let ts = parsed.get("timestamp").and_then(|v| v.as_u64());
                        if let (Some(id_str), Some(model), Some(ts)) =
                            (withdrawn_id, withdrawn_model, ts)
                        {
                            match id_str.parse::<PeerId>() {
                                Ok(pid) if message.source == Some(pid) => {
                                    // F1: reject a stale (replayed) withdrawal before evicting. The
                                    // signed timestamp can't be refreshed by a replayer; a genuine
                                    // re-publish (F2) carries a current one.
                                    if !withdrawal_is_fresh(ts, now_unix_ms() / 1000, MODEL_WITHDRAWN_MAX_AGE_SECS) {
                                        debug!(%pid, %model, ts, "ignored stale MODEL_WITHDRAWN (replay?)");
                                    } else if evict_withdrawn_model(&mut state.known_peers, id_str, model) > 0 {
                                        info!(%pid, %model, "evicted model via MODEL_WITHDRAWN gossip");
                                        note_withdrawn_model(&mut state.withdrawn_models, &state.known_peers, &mut state.model_seen, model, id_str, std::time::Instant::now());
                                    }
                                }
                                Ok(pid) => warn!(
                                    ?message.source, %pid, %model,
                                    "rejected MODEL_WITHDRAWN: author does not match subject"
                                ),
                                Err(e) => debug!(%e, "MODEL_WITHDRAWN: unparseable libp2p_peer_id"),
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
            } else if let libp2p::gossipsub::Event::Subscribed { peer_id, topic } = gossip_event {
                if topic.as_str() == crate::swarm::GOSSIPSUB_TOPIC
                    && first_time(&mut state.gossip_peer_seen, state.announced)
                {
                    info!(%peer_id, "first gossip peer: re-announcing (the first announce had nobody to tell)");
                    bump_net_generation(state);
                }
            }
        }

        // Ping keepalive — log failures and evict unreachable peers
        // (Phase 1.6). Success is silent to avoid flooding logs.
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Ping(ping_event)) => {
            if let Ok(rtt) = ping_event.result {
                state.peer_rtts.record(&ping_event.peer.to_base58(), rtt);
            }
            if ping_event.result.is_ok() {
                state.relay_ping_failures.remove(&ping_event.connection);
                state.relayed_ping_failures.remove(&ping_event.connection);
            }
            if let Err(ref e) = ping_event.result {
                let peer = ping_event.peer;
                warn!(%peer, error = %e, "ping failed");
                if relay_connection_dead(state, peer, ping_event.connection, e, &our_listen_ips(swarm)) {
                    // Our reservation rides this connection; closing it closes the reservation's
                    // listener, and that schedules a fresh reservation on the relay.
                    warn!(%peer, connection = ?ping_event.connection, "closing a relay connection that stopped answering pings");
                    swarm.close_connection(ping_event.connection);
                }
                let busy = peer_busy(state, peer, |id| swarm.behaviour().grpc_proxy.is_pending_outbound(&peer, id));
                if relayed_connection_dead(state, peer, ping_event.connection, e, busy) {
                    warn!(%peer, connection = ?ping_event.connection, "closing an idle relayed connection that stopped answering pings");
                    swarm.close_connection(ping_event.connection);
                }
                // Drop the dead route but keep `known_peers`: a ping failure to a relayed/NAT'd
                // peer is transient, and wiping the cache here collapsed multi-model providers.
                // The TTL reaper (fresh-or-connected) ages out a genuinely gone peer.
                evict_unreachable_peer(swarm, peer, &[]);
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
        SwarmEvent::NewListenAddr { listener_id, address } => {
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
            // client (safe), never a black hole. DCUtR is unaffected: it offers the
            // addresses peers observed us at (identify), regardless of confirmation.
            //
            // F-5: a circuit listen addr coming up means the reservation
            // succeeded — clear any pending backoff retry for it.
            if address.to_string().contains("/p2p-circuit") {
                clear_reservation_retry(state, &address);
                // #42: track live reservations so rebootstrap re-requests only
                // the missing ones (avoids leaking duplicate circuit listeners).
                state.reserved_circuits.insert(address.to_string());
                let now = std::time::Instant::now();
                let v6_quic_listener = swarm.listeners().any(|a| is_v6_quic(a) && multiaddr_ip(a).is_some_and(|ip| !is_non_global_ip(ip)));
                if let Some((relay, addr)) = relay_v6_quic_dial(state, listener_id, &address, now).filter(|_| v6_quic_listener) {
                    state.relay_v6_quic_dialled.insert(relay, now);
                    // From the listen socket (QUIC's default), so the relay sees the address peers
                    // can punch to.
                    let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(relay)
                        .addresses(vec![addr.clone()])
                        .condition(libp2p::swarm::dial_opts::PeerCondition::Always)
                        .build();
                    match swarm.dial_tracked(&mut state.dials_in_flight, opts) {
                        Ok(()) => info!(%relay, %addr, "P1: dialling our relay over IPv6 QUIC so peers learn our IPv6 QUIC address"),
                        Err(e) => debug!(%relay, %addr, error = %e, "P1: IPv6 QUIC dial to relay not started"),
                    }
                }
            } else {
                for (peer_id, addr) in startup_dials_ready(&mut state.startup_dials, &address) {
                    let dial_addr = addr.clone().with(libp2p::multiaddr::Protocol::P2p(peer_id));
                    info!(%peer_id, %dial_addr, from = %address, "bootstrap_dial: explicitly dialing bootstrap peer");
                    if let Err(e) = swarm.dial_tracked(&mut state.dials_in_flight, dial_addr.clone()) {
                        warn!(%peer_id, %dial_addr, %e, "bootstrap_dial: failed");
                    }
                }
                if state.startup_dials.is_empty() && state.startup_dials_until.take().is_some() {
                    start_kad_bootstrap(swarm);
                }
                if is_real_interface_addr(&address) {
                    // #42: a real interface came up (roam / wake / new link) — arm a
                    // debounced rebuild so the routing table, bootstrap connections
                    // and relay reservations refresh for the new network.
                    arm_reactive_rebootstrap(state, "new listen addr");
                }
            }
        }
        SwarmEvent::ConnectionEstablished { peer_id, connection_id, endpoint, .. } => {
            state.relay_dial_backoff.remove(&peer_id);
            // Back within the grace: its provider records stay (#30).
            state.lost_at_ms.remove(&peer_id);
            state.lan_dialer.established(&connection_id, std::time::Instant::now());
            if endpoint.is_relayed() {
                state.relayed_conns.insert(connection_id, peer_id);
            }
            if !endpoint.is_relayed() && crate::relay::relay_peer_ids(&state.relays).contains(&peer_id) {
                state.relay_conns.insert(connection_id, RelayConn { since: std::time::Instant::now(), seen_as: None });
                if is_v6_quic(endpoint.get_remote_address()) {
                    state.relay_v6_quic_conns.insert(connection_id, peer_id);
                }
            }
            // An address we reached a relay of ours at directly is one it's known at (the circuit
            // check, relay.rs), whatever form (DNS name, IPv6) it was configured in.
            if let libp2p::core::ConnectedPoint::Dialer { address, .. } = &endpoint {
                if !endpoint.is_relayed() && trusted_relay_pids(state).contains_peer(&peer_id) {
                    let reached = state.relay_dialled_addrs.entry(peer_id).or_default();
                    if !reached.contains(address) && reached.len() < MAX_RELAY_SELF_ADDRS {
                        reached.push(address.clone());
                    }
                }
            }
            if let Some(set) = state.dials_in_flight.get_mut(&peer_id) {
                set.remove(&connection_id);
                if set.is_empty() {
                    state.dials_in_flight.remove(&peer_id);
                }
            }
            // The address this connection really runs over (relayed ones have no IP of the peer's):
            // what identify's claims are checked against.
            if !endpoint.is_relayed() {
                state.conn_remote_addrs.insert(connection_id, (peer_id, endpoint.get_remote_address().clone(), std::time::Instant::now()));
            }
            let addr_str = match &endpoint {
                libp2p::core::ConnectedPoint::Dialer { address, .. } => address.to_string(),
                libp2p::core::ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr.to_string(),
            };
            let transport = classify_transport(&addr_str);
            let endpoint_ip = extract_ip_from_multiaddr_str(&addr_str);
            if is_local_link(&addr_str, endpoint_ip.as_deref()) {
                *state.local_link_peers.entry(peer_id).or_insert(0) += 1;
            }
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

            let is_our_relay = crate::relay::relay_peer_ids(&state.relays).contains(&peer_id);
            let info = state.peer_connections.entry(peer_id).or_default();
            if transport != TransportType::TcpRelay {
                if let Some(ip) = &endpoint_ip {
                    info.direct_ips.insert(connection_id, ip.clone());
                }
            }

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
                    let is_listener = matches!(endpoint, libp2p::core::ConnectedPoint::Listener { .. });
                    match quic_dup_action(is_listener) {
                        QuicDupAction::CloseNow => {
                            debug!(%peer_id, %addr_str, v4=info.quic_direct_v4, v6=info.quic_direct_v6,
                                "quic_dedup: closing duplicate QUIC connection (same AF)");
                            let _ = swarm.close_connection(connection_id);
                        }
                        QuicDupAction::CloseAfterGrace => {
                            debug!(%peer_id, %addr_str, v4=info.quic_direct_v4, v6=info.quic_direct_v6,
                                grace_secs = QUIC_INBOUND_DUP_GRACE.as_secs(),
                                "quic_dedup: inbound duplicate QUIC connection (same AF) — closing after grace (may be an AutoNAT dial-back)");
                            state.quic_dup_close_deadline.insert(
                                connection_id,
                                (peer_id, is_ipv6, std::time::Instant::now() + QUIC_INBOUND_DUP_GRACE),
                            );
                        }
                    }
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
            // (Never for one of our relays: its v4 connection may hold our reservation, and P1 adds
            // a v6 QUIC connection to it on purpose.)
            if should_arm_v4_close(info) && !is_our_relay {
                state
                    .v4_close_deadline
                    .entry(peer_id)
                    .or_insert_with(|| std::time::Instant::now() + RELAY_CLOSE_GRACE);
            }

            // Send any queued proxy forwards that were waiting for this connection.
            let mut flushed_for_peer = false;
            let mut remaining = Vec::new();
            for (target, data, reply, enqueued_at) in std::mem::take(&mut state.pending_relay_forwards) {
                if target == peer_id {
                    info!(%peer_id, "sending queued proxy forward after relay connection");
                    flushed_for_peer = true;
                    send_proxy_request(swarm, state, peer_id, data, Some(reply));
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
        SwarmEvent::ConnectionClosed { peer_id, connection_id, endpoint, cause, .. } => {
            // A connection closed as idle (`kad_idle`), by us or by a peer that does that, says
            // nothing about the peer: below, its routing entry and records stay.
            // A close by the peer counts only when the peer closed on purpose (`closed_by_peer`):
            // a crash, a reset or a timeout is a loss whatever the peer runs (review 22 Nov).
            let closed_as_idle = state.idle_closed.remove(&connection_id)
                || (state.kad_idle_peers.contains(&peer_id)
                    && !state.app_peers.wanted(&peer_id)
                    && closed_by_peer(cause.as_ref()));
            state.relay_ping_failures.remove(&connection_id);
            state.relay_conns.remove(&connection_id);
            state.relay_v6_quic_conns.remove(&connection_id);
            state.relayed_conns.remove(&connection_id);
            state.relayed_ping_failures.remove(&connection_id);
            state.conn_remote_addrs.remove(&connection_id);
            state.lan_dialer.closed(&connection_id, std::time::Instant::now());
            let addr_str = match &endpoint {
                libp2p::core::ConnectedPoint::Dialer { address, .. } => address.to_string(),
                libp2p::core::ConnectedPoint::Listener { send_back_addr, .. } => send_back_addr.to_string(),
            };
            let transport = classify_transport(&addr_str);
            let endpoint_ip = extract_ip_from_multiaddr_str(&addr_str);
            if is_local_link(&addr_str, endpoint_ip.as_deref()) {
                if let Some(n) = state.local_link_peers.get_mut(&peer_id) {
                    *n = n.saturating_sub(1);
                    if *n == 0 {
                        state.local_link_peers.remove(&peer_id);
                    }
                }
            }
            let is_relay_ip = endpoint_ip
                .as_ref()
                .map(|ip| crate::relay::is_bootstrap_relay_ip(ip))
                .unwrap_or(false);
            let transport = if is_relay_ip && transport != TransportType::TcpRelay {
                TransportType::TcpRelay
            } else {
                transport
            };

            state.quic_dup_close_deadline.remove(&connection_id);
            if let Some(info) = state.peer_connections.get_mut(&peer_id) {
                info.direct_ips.remove(&connection_id);
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
                // A round trip stands for a live path; with no connection left there is none.
                state.peer_rtts.forget(&peer_id.to_base58());

                if closed_as_idle {
                    // Closed because nothing used it (`kad_idle`): not a lost peer. A peer that really
                    // left goes with its departure notice, the bootstraps' gone notice, its records'
                    // age, or Kademlia's own failed dials.
                    debug!(%peer_id, "an idle connection closed; the peer's routing entry and records stay");
                } else {
                    // Ghost-peer purge: Kademlia routing table
                    swarm.behaviour_mut().kademlia.remove_peer(&peer_id);
                    debug!(%peer_id, "evicted from kademlia routing table");

                    // Its provider records outlive the connection for LOST_PEER_GRACE_MS (#30): a
                    // dropped connection is usually back within seconds, and deleting them at once
                    // left us answering "no provider" until the provider's next announcement. The
                    // reaper deletes them once the grace runs out with no connection and no record
                    // since.
                    let libp2p_id_str = peer_id.to_string();
                    if state.known_peers.values().any(|record| record.libp2p_peer_id == libp2p_id_str) {
                        state.lost_at_ms.insert(peer_id, now_unix_ms());
                        debug!(%peer_id, "lost our last connection to a provider; its records stay for the grace period");
                    }
                }
                state.kad_idle_peers.remove(&peer_id);

                // E-N3: prune the remaining per-peer maps that keyed off this
                // peer. These grew unbounded because the disconnect cleanup
                // evicted peer_connections / known_peers / kademlia but left
                // last_repunch (debounce timestamps) and peer_quic_addrs (learned
                // QUIC addrs) behind — one entry leaked per peer ever seen.
                state.last_repunch.remove(&peer_id);
                state.lan_announced.remove(&peer_id);
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
            if !state.external_addrs.contains(&address) {
                state.external_addrs.push(address.clone());
            }
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
        // An address that no longer holds (a relay circuit whose relay connection closed) leaves
        // the list `/status` reports.
        SwarmEvent::ExternalAddrExpired { address } => {
            info!(%address, "external address expired");
            state.external_addrs.retain(|a| a != &address);
            // The last address we were reachable at going (a UPnP lease lapsing, say): a server
            // nobody can reach any more would hand out its own address in DHT answers and take
            // queries it can't answer, so back to client. AutoNAT or a new mapping promotes again.
            if should_demote_to_client(state) {
                info!(%address, "no reachable direct address left: Kademlia client mode");
                swarm.behaviour_mut().kademlia.set_mode(Some(kad::Mode::Client));
                state.kad_server_mode = false;
                state.nat_info.is_public = false;
                state.nat_info.nat_type = "unknown".into();
            }
        }
        // A failed outgoing dial to a peer we hold no live connection to → drop it from the
        // routing table (dead path). Do NOT wipe `known_peers` here — that collapsed multi-model
        // providers on transient dial failures; the TTL reaper owns cache lifecycle. See
        // [`evict_unreachable_peer`].
        SwarmEvent::OutgoingConnectionError { peer_id, connection_id, error } => {
            if state.lan_dialer.is_lan_dial(&connection_id) {
                // A LAN dial that failed says nothing about the peer's other routes (a firewall that
                // refuses LAN connections, or someone on the LAN announcing that peer at their own
                // address): back it off, and touch nothing else (review 2026-10-29).
                debug!(?peer_id, %error, "LAN dial failed");
                state.lan_dialer.failed(&connection_id, std::time::Instant::now());
                return;
            }
            if peer_id.is_some_and(|p| !pinned_addrs(state, &p).is_empty()) {
                // The own provider isn't running (sharing off): routine, and asked every few seconds.
                debug!(?peer_id, %error, "outgoing_connection_error (pinned peer)");
            } else {
                warn!(?peer_id, %error, "outgoing_connection_error");
            }
            // A bootstrap dial refused because our port's previous connection to it is still in
            // TIME_WAIT (just restarted): dial those addresses again at once from a new port, at most
            // once per bootstrap per IN_USE_REDIAL_EVERY.
            if let Some(pid) = peer_id.filter(|p| state.bootstrap_peers.iter().any(|(b, _)| b == p)) {
                let addrs: Vec<Multiaddr> = crate::transport::addrs_refused_as_in_use(&error)
                    .into_iter()
                    .map(|a: Multiaddr| a.into_iter().filter(|p| !matches!(p, libp2p::multiaddr::Protocol::P2p(_))).collect())
                    .collect();
                let due = state.in_use_redials.get(&pid).map_or(true, |t| t.elapsed() >= IN_USE_REDIAL_EVERY);
                if !addrs.is_empty() && due {
                    state.in_use_redials.insert(pid, std::time::Instant::now());
                    info!(%pid, ?addrs, "bootstrap_dial: local address in use (restart), dialing from a new port");
                    let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(pid)
                        .addresses(addrs)
                        .condition(libp2p::swarm::dial_opts::PeerCondition::Always)
                        .allocate_new_port()
                        .build();
                    if let Err(e) = swarm.dial_tracked(&mut state.dials_in_flight, opts) {
                        warn!(%pid, %e, "bootstrap_dial: failed");
                    }
                }
            }
            if let Some(pid) = peer_id {
                let connected = swarm.is_connected(&pid);
                if !connected {
                    evict_unreachable_peer(swarm, pid, &dial_error_addrs(&error));
                }
                // Nothing else can reach it now: fail its queued forwards at once (the caller
                // retries with a fresh lookup) instead of after RELAY_FORWARD_TTL.
                if dial_ended_unreachable(&mut state.dials_in_flight, pid, connection_id, connected) {
                    let relay_limit = crate::transport::dial_refused_by_relay_limits(&error);
                    let why = if relay_limit { DIAL_FAILED_RELAY_LIMIT } else { DIAL_FAILED_UNREACHABLE };
                    // Never for a pinned peer: a loopback dial is cheap, and the own provider that was
                    // down a moment ago must be reachable the moment it's up.
                    let pinned = !pinned_addrs(state, &pid).is_empty();
                    if !pinned && (state.relay_dial_backoff.len() < MAX_RELAY_DIAL_BACKOFFS || state.relay_dial_backoff.contains_key(&pid)) {
                        let step = next_relay_dial_backoff(state.relay_dial_backoff.get(&pid).map(|b| b.step), relay_limit);
                        state.relay_dial_backoff.insert(pid, RelayDialBackoff {
                            until: std::time::Instant::now() + step,
                            step,
                            set_unix_ms: now_unix_ms(),
                            why,
                        });
                        debug!(%pid, ?step, "relayed dials to this peer backed off");
                    }
                    let drained = std::mem::take(&mut state.pending_relay_forwards);
                    let (kept, failed) = fail_relay_forwards_for(drained, pid, why);
                    state.pending_relay_forwards = kept;
                    if failed > 0 {
                        info!(%pid, failed, "proxy_forward: dial failed, no other dial in flight — failing queued forwards now");
                    }
                }
            }
        }
        // A connection a limit turned away is routine (and can come by the thousand): not a warning.
        SwarmEvent::IncomingConnectionError { error: error @ libp2p::swarm::ListenError::Denied { .. }, .. } => {
            debug!(%error, "incoming_connection_denied");
        }
        // A handshake that fails is the other side's problem or a port scan, and hole punching makes
        // dozens a minute (GPU test 2026-10-29: the providers' logs were full of them): not a warning.
        SwarmEvent::IncomingConnectionError { error: error @ libp2p::swarm::ListenError::Transport(_), .. } => {
            debug!(%error, "incoming_connection_error");
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
                // (Defensive: libp2p-relay 0.18's listener doesn't expire circuit addresses on its own;
                // they go with ListenerClosed.) The listener that held it renews it, or closes (and
                // ListenerClosed schedules the retry). Only a relay none of our listeners is on needs one from here; listening
                // again on a relay we still have a listener on is what started the churn.
                if let Some(relay) = circuit_relay(&address) {
                    if relay_has_listener(state, relay) {
                        info!(%address, "relay reservation expired; its listener renews it");
                    } else if let Some(circuit) = reservation_circuit(&state.relays, relay, Some(&address)) {
                        schedule_reservation_retry(state, relay, circuit);
                    }
                }
            } else if is_real_interface_addr(&address) {
                // #42: a real interface address went away (roam / interface
                // down) — expire the stale direct external addrs tied to it and
                // arm a debounced rebuild for the new network.
                let remaining: Vec<Multiaddr> = swarm.listeners().cloned().collect();
                if addr_loss_leaves_network(&address, &remaining) {
                    state.last_addr_loss = Some(std::time::Instant::now());
                }
                expire_direct_external_addrs(swarm, state);
                arm_reactive_rebootstrap(state, "expired listen addr");
            }
        }
        SwarmEvent::ListenerClosed { listener_id, addresses, reason } => {
            warn!(?addresses, ?reason, "listener_closed");
            let mut real_interface_closed = false;
            for addr in &addresses {
                if addr.to_string().contains("/p2p-circuit") {
                    // #42: reservation dropped — no longer held.
                    state.reserved_circuits.remove(&addr.to_string());
                    if let Err(e) = &reason {
                        if let Some(hint) = private_relay_hint(&state.relays, addr, &e.to_string()) {
                            if state.relay_hints_given.insert(addr.to_string()) {
                                warn!("{hint}");
                            }
                        }
                    }
                } else if is_real_interface_addr(addr) {
                    real_interface_closed = true;
                }
            }
            let remaining: Vec<Multiaddr> = swarm.listeners().cloned().collect();
            let left_network = addresses.iter().any(|a| addr_loss_leaves_network(a, &remaining));
            // F-5: a closed circuit listener means the reservation dropped or was refused. Schedule
            // one backed-off retry per relay it was on (not one per address it reported), and none
            // for a relay another listener of ours is still on. An immediate re-listen against a
            // flapping or capped relay tight-loops the event loop and spams the relay; the
            // relay_retry_ticker re-listens once the backoff window elapses.
            for (relay, circuit) in relays_to_re_reserve(state, listener_id, &addresses) {
                schedule_reservation_retry(state, relay, circuit);
            }
            if real_interface_closed {
                // #42: a real interface listener closed (roam / wake) — expire
                // stale direct external addrs and arm a debounced rebuild.
                if left_network {
                    state.last_addr_loss = Some(std::time::Instant::now());
                }
                expire_direct_external_addrs(swarm, state);
                arm_reactive_rebootstrap(state, "listener closed");
            }
        }
        SwarmEvent::ListenerError { error, .. } => {
            warn!(%error, "listener_error");
        }
        SwarmEvent::Dialing { peer_id, connection_id } => {
            debug!(?peer_id, "dialing");
            if let Some(pid) = peer_id {
                state.dials_in_flight.entry(pid).or_default().insert(connection_id);
            }
        }
        _ => {
            trace!("unhandled_swarm_event");
        }
    }
}

/// Whether we're a Kademlia server with no direct external address left (circuits don't count:
/// reaching us through a relay doesn't make us reachable).
fn should_demote_to_client(state: &LoopState) -> bool {
    state.kad_server_mode && !state.external_addrs.iter().any(|a| !a.to_string().contains("/p2p-circuit"))
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
                            // 1. Check known_peers cache: this provider's record FOR THIS MODEL (a
                            //    provider of several models has one record each; another model's
                            //    would answer this discover with the wrong model).
                            let cached = state.known_peers.values()
                                .find(|r| r.libp2p_peer_id == pid_str && r.model_id == pending.model_id)
                                .cloned();
                            if let Some(record) = cached {
                                pending.merge(record, now_unix_ms());
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
                                    } else if !is_gone_record(&state.gone_providers, &record, now_unix_ms())
                                        && install_known_peer_in(
                                        &mut state.known_peers,
                                        &mut state.known_received,
                                        &state.local_libp2p,
                                        &record,
                                        now_unix_ms(),
                                    )
                                    .is_some()
                                    {
                                        pending.merge(record, now_unix_ms());
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
                                // Cache the peer, under the same rules as every other path (review
                                // 2026-10-30: this one skipped the age, newer-only and receive-time checks).
                                if !is_gone_record(&state.gone_providers, &peer_record, now_unix_ms())
                                    && install_known_peer_in(
                                    &mut state.known_peers,
                                    &mut state.known_received,
                                    &state.local_libp2p,
                                    &peer_record,
                                    now_unix_ms(),
                                )
                                .is_some()
                                {
                                    pending.merge(peer_record, now_unix_ms());
                                }
                            }
                            Err(e) => {
                                warn!("failed to decode DHT record: {e}");
                            }
                        }
                    }
                }
                Ok(kad::GetRecordOk::FinishedWithNoAdditionalRecord { .. }) => {
                    // Query complete — send results back.
                    if let Some(pending) = state.pending_discovers.get_mut(&id) {
                        pending.providers_done = true;
                        pending.outstanding = 0;
                    }
                    maybe_reply_discover(state, id);
                }
                Err(e) => {
                    if let Some(pending) = state.pending_discovers.remove(&id) {
                        settle_discover(state, &pending);
                        for w in pending.waiters {
                            let _ = w.reply.send(Err(format!("kademlia get_record: {e:?}")));
                        }
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
            handle_inbound_kad_request(request, swarm, state);
        }
        kad::Event::RoutingUpdated { peer, addresses, is_new_peer, .. } => {
            debug!(%peer, "kademlia routing updated");
            // Diagnostic (target oh_kad_diag): who enters the routing table, and by which address.
            debug!(target: "oh_kad_diag", %peer, is_new_peer, addresses = ?addresses.iter().collect::<Vec<_>>(), "routing table add");
        }
        // Phase 1.2: UnroutablePeer is the strongest signal that a peer
        // should be evicted — Kademlia itself declares it unreachable.
        kad::Event::UnroutablePeer { peer } => {
            if pinned_addrs(state, &peer).is_empty() {
                warn!(%peer, "kademlia reports peer unroutable — dropping route, keeping cache");
            } else {
                // Expected: its only address here is the loopback one Kademlia is not allowed to keep.
                debug!(%peer, "kademlia reports pinned peer unroutable (its loopback address is kept out)");
            }
            // Routing-only: keep `known_peers` (TTL reaper owns lifecycle). Wiping here collapsed
            // multi-model providers whenever Kademlia transiently declared a relayed peer unroutable.
            evict_unreachable_peer(swarm, peer, &[]);
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
fn trusted_relay_pids(state: &LoopState) -> crate::relay::TrustedRelays {
    let mut trusted = crate::relay::TrustedRelays::default();
    for (pid, addr) in &state.bootstrap_peers {
        trusted.add_for(*pid, addr.clone());
    }
    // The relays this node uses (the production ones only on the public network: a private
    // deployment never dials a circuit through them).
    for addr in &state.relays {
        trusted.add(addr);
    }
    // And the addresses those relays report for themselves (review 2026-10-29: peers reserve
    // through whichever relay address they dialled, which the configured list may not have).
    for (pid, addrs) in state.relay_self_addrs.iter().chain(&state.relay_dialled_addrs) {
        if trusted.contains_peer(pid) {
            for addr in addrs {
                trusted.add_for(*pid, addr.clone());
            }
        }
    }
    trusted
}

/// Which network this node is on ([`crate::relay::network_id`]), as its identify tells peers.
fn our_network_id(state: &LoopState) -> String {
    crate::relay::network_id(&state.bootstrap_peers, on_public_network(state))
}

/// Close the connections only the DHT used, once idle (`kad_idle`): Kademlia dialled them, nothing
/// else of ours uses the peer, and the peer knows such a close isn't a loss. Every
/// [`crate::kad_idle::SWEEP_EVERY`], from the heal tick.
fn drive_kad_idle_close(swarm: &mut libp2p::Swarm<OpenHydraBehaviour>, state: &mut LoopState) {
    let now = std::time::Instant::now();
    let mesh: std::collections::HashSet<PeerId> = swarm.behaviour().gossipsub.all_mesh_peers().copied().collect();
    let relays = trusted_relay_pids(state);
    let pinned: std::collections::HashSet<PeerId> =
        state.pinned.read().unwrap_or_else(|e| e.into_inner()).keys().copied().collect();
    let mut close = Vec::new();
    for (cid, (peer, _, opened)) in &state.conn_remote_addrs {
        let kad = &swarm.behaviour().kademlia;
        let age = now.saturating_duration_since(*opened);
        let facts = crate::kad_idle::ConnFacts {
            kad_dialled: kad.kad_dialled(cid),
            age,
            since_kad_use: kad.last_activity(peer).map_or(age, |t| now.saturating_duration_since(t)),
            peer_supports: state.kad_idle_peers.contains(peer),
            kept: state.bootstrap_peers.iter().any(|(b, _)| b == peer)
                || relays.contains_peer(peer)
                || pinned.contains(peer)
                || state.lan_dialer.knows(peer)
                || mesh.contains(peer),
            app_used: state.app_peers.wanted(peer),
            in_flight: false,
        };
        // Only for one that would close: whether anything is under way with the peer (ours or its
        // requests, a response being written, a queued forward; review 22 Nov: counting only our
        // own requests let a connection serving a long job be closed under it).
        if crate::kad_idle::should_close(&facts) {
            let busy = peer_busy(state, *peer, |id| swarm.behaviour().grpc_proxy.is_pending_outbound(peer, id));
            if !busy {
                close.push((*cid, *peer));
            }
        }
    }
    for (cid, peer) in close {
        if swarm.close_connection(cid) {
            state.idle_closed.insert(cid);
            state.idle_closes += 1;
            debug!(target: "oh_kad_idle", me = %swarm.local_peer_id(), %peer, connection_id = ?cid, "closed an idle Kademlia connection");
        }
    }
}

/// Dial the LAN peers mDNS lists that are due a dial ([`crate::mdns::LanDialer::due`]). On every
/// discovery and on the heal tick, so a peer skipped or backed off is tried again.
fn drive_lan_dials(swarm: &mut libp2p::Swarm<OpenHydraBehaviour>, state: &mut LoopState) {
    // Nothing to gain dialling a peer we already reach over the LAN, or directly by any path.
    let due = state.lan_dialer.due(
        |p| state.local_link_peers.contains_key(p) || state.peer_connections.get(p).is_some_and(|c| c.has_direct()),
        std::time::Instant::now(),
    );
    for (peer_id, addrs) in due {
        // `NotDialing`, not `Disconnected…`: a peer we reach only through a relay is worth a direct
        // LAN connection too.
        let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(peer_id)
            .condition(libp2p::swarm::dial_opts::PeerCondition::NotDialing)
            .addresses(addrs.clone())
            .build();
        let id = opts.connection_id();
        match swarm.dial(opts) {
            Ok(()) => {
                info!(%peer_id, "dialling a peer mDNS found on the LAN");
                state.lan_dialer.dialled(id, peer_id, &addrs);
            }
            Err(e) => debug!(%peer_id, %e, "mDNS peer not dialled"),
        }
    }
}

/// Whether this node is on OpenHydra's public network: its relays include the production ones,
/// which `relay::relay_addrs` adds exactly when `relay::uses_public_network` says so.
fn on_public_network(state: &LoopState) -> bool {
    let production = crate::relay::bootstrap_relay_peer_ids();
    crate::relay::relay_peer_ids(&state.relays).iter().any(|p| production.contains(p))
}

/// What a trusted relay's identify says about its own addresses that we take as places it's known
/// at: direct ones only, at most [`MAX_RELAY_SELF_ADDRS`]; on the public network global ones only
/// (identify also sends loopback and private listen addresses, and a trusted relay's id in front of
/// one would point queriers at their own machine or LAN).
fn relay_self_addrs(listen_addrs: &[Multiaddr], public_network: bool) -> Vec<Multiaddr> {
    listen_addrs
        .iter()
        .filter(|a| !a.iter().any(|p| matches!(p, libp2p::multiaddr::Protocol::P2pCircuit)))
        .filter(|a| !public_network || crate::swarm::routable_on_public_network(a))
        .filter(|a| matches!(a.iter().next(), Some(libp2p::multiaddr::Protocol::Ip4(_) | libp2p::multiaddr::Protocol::Ip6(_))))
        .take(MAX_RELAY_SELF_ADDRS)
        .cloned()
        .collect()
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
    state: &mut LoopState,
) {
    match request {
        kad::InboundRequest::PutRecord { source, record: Some(record), .. } => {
            use libp2p::kad::store::RecordStore;
            // Verified, recent, stored only under its canonical key (H key-binding) and never over
            // a newer one, within its publisher's and network's share (see `dht`).
            let peer_record = match dht::check_put_record(&record, now_unix_ms()) {
                Ok(r) => r,
                Err(e) => {
                    debug!(%source, %e, "kad_put_rejected");
                    return;
                }
            };
            if !dht::newer_than_stored(swarm.behaviour_mut().kademlia.store_mut().get(&record.key).as_deref(), &peer_record) {
                debug!(%source, "kad_put_rejected: older than the record we hold");
                return;
            }
            let origin = state
                .conn_remote_addrs
                .values()
                .find(|(p, _, _)| *p == source)
                .and_then(|(_, a, _)| multiaddr_ip(a))
                .map(crate::ip_limits::origin_key)
                .unwrap_or_else(|| format!("relayed:{source}"));
            if !state.put_quota.admit(&record.key, &peer_record.libp2p_peer_id, &origin, std::time::Instant::now()) {
                debug!(%source, %origin, "kad_put_rejected: publisher or network over its share of the store");
                return;
            }
            if let Err(e) = swarm.behaviour_mut().kademlia.store_mut().put(record) {
                debug!(%source, ?e, "kad_put: store rejected verified record (capacity?)");
            }
        }
        // Filtered but empty (shouldn't happen with FilterBoth) — nothing to store.
        kad::InboundRequest::PutRecord { record: None, .. } => {}
        kad::InboundRequest::AddProvider { record: Some(provider) } => {
            // No signature to verify here; the read side fetches + verifies the
            // provider's actual signed record before trusting it. Each provider and network gets its
            // share of the provider store (model keys are free).
            let origin = state
                .conn_remote_addrs
                .values()
                .find(|(p, _, _)| *p == provider.provider)
                .and_then(|(_, a, _)| multiaddr_ip(a))
                .map(crate::ip_limits::origin_key)
                .unwrap_or_else(|| format!("relayed:{}", provider.provider));
            if !state.provider_quota.admit(&provider.key, &provider.provider.to_base58(), &origin, std::time::Instant::now()) {
                debug!(provider = %provider.provider, %origin, "kad_add_provider: over its share of the store");
                return;
            }
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
    if let Some(pending) = state.pending_discovers.get_mut(&discover_id) {
        pending.merge(peer_record, now_unix_ms());
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
    // A stale copy of a provider the bootstraps reported gone: not even its address goes back into
    // the routing table (review 2026-10-30).
    if is_gone_record(&state.gone_providers, peer_record, now_unix_ms()) {
        return false;
    }
    if !peer_record.relay_address.is_empty() && !peer_record.libp2p_peer_id.is_empty() {
        if let Ok(pid) = peer_record.libp2p_peer_id.parse::<PeerId>() {
            let trusted = trusted_relay_pids(state);
            match crate::relay::safe_injectable_circuit_addr(&peer_record.relay_address, &pid, &trusted) {
                // Kept for our own dials to it only (the address book), not put in the routing
                // table: a provider reachable only through a relay doesn't answer DHT requests, and
                // in the table every DHT query nearby would dial it through the relay for nothing
                // and hand it to other peers in FIND_NODE answers (20 Nov; see the identify rule).
                Some(ma) => {
                    let new = swarm.behaviour_mut().address_book.insert(pid, ma.clone());
                    debug!(%pid, %ma, new, "discover_kept_relay_address");
                }
                None => {
                    warn!(%pid, addr = %peer_record.relay_address,
                          "record: rejected non-injectable relay_address (D-S5)");
                }
            }
        }
    }
    let installed = install_known_peer(state, peer_record, now_unix_ms()).is_some();
    if installed {
        if let Ok(pid) = peer_record.libp2p_peer_id.parse::<PeerId>() {
            pursue_v6_holepunch_debounced(swarm, state, pid, "record");
        }
    }
    installed
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

/// Whether a lookup has finished: the query is done and no record fetch is outstanding.
fn lookup_done(p: &PendingDiscover) -> bool {
    p.providers_done && p.outstanding == 0
}

/// Whether a waiting caller can be answered now: the lookup finished; or, with no record fetch
/// outstanding, a caller with nothing known has providers, or a retry has new information (a
/// provider new, or reachable at a different address, since it joined); or a retry's deadline has
/// passed, whatever is outstanding.
fn waiter_settled(
    kind: DiscoverKind,
    done: bool,
    outstanding: usize,
    records: usize,
    info_gen: u64,
    info_at_join: u64,
    deadline: Option<std::time::Instant>,
    now: std::time::Instant,
) -> bool {
    done || deadline.is_some_and(|d| now >= d)
        || (outstanding == 0
            && match kind {
                DiscoverKind::Reply => records > 0,
                DiscoverKind::Retry => info_gen > info_at_join,
            })
}

/// Did a lookup numbered `seq` start after the invalidate numbered `set_at` (or was there none)?
fn started_after(set_at: Option<u64>, seq: u64) -> bool {
    set_at.map_or(true, |set_at| seq > set_at)
}

/// A finished lookup: true if it started after the model's last `InvalidateDiscover` (or there was
/// none), in which case it clears the model's wait flag and may mark the model fresh in the C11
/// cache. One started before predates the failure that set the flag, so it does neither.
fn settle_discover(state: &mut LoopState, pending: &PendingDiscover) -> bool {
    let fresh = started_after(state.discover_wait_for_lookup.get(&pending.model_id).copied(), pending.seq);
    if fresh {
        state.discover_wait_for_lookup.remove(&pending.model_id);
    }
    fresh
}

/// Answer a lookup's waiting callers whose answer is settled ([`waiter_settled`]), **without
/// waiting for the get_providers query to converge** (that tail is a fixed ~10 s timeout that
/// otherwise dominates per-request latency); once the lookup has finished, answer the rest and
/// retire it. Ranking / empty handling is the consumer's job.
fn maybe_reply_discover(state: &mut LoopState, discover_id: kad::QueryId) {
    let now = std::time::Instant::now();
    let Some(pending) = state.pending_discovers.get(&discover_id) else { return };
    let done = lookup_done(pending);
    let any_settled = pending.waiters.iter().any(|w| {
        waiter_settled(w.kind, done, pending.outstanding, pending.records.len(), pending.info_gen, w.info_at_join, w.deadline, now)
    });
    if !done && !any_settled {
        return;
    }
    // Answer from the current state: a provider's newer record may have arrived by gossip or PEX
    // while the lookup ran (`known_peers`), not only from the DHT.
    let now_ms = now_unix_ms();
    let current: Vec<PeerRecord> = state
        .known_peers
        .values()
        .filter(|r| r.model_id == pending.model_id && r.libp2p_peer_id != state.local_libp2p)
        .cloned()
        .collect();
    let pending = state.pending_discovers.get_mut(&discover_id).expect("present");
    for record in current {
        merge_record(&mut pending.records, record, now_ms);
    }
    // R-DHT-8: `discovered_from_records` stamps the liveness hint (live libp2p connection?)
    // so the consumer prefers connected providers and failover is the exception. All
    // records here are already H1/PEX-verified, so we never surface an unverified candidate.
    let records = pending.records.clone();
    let peers = discovered_from_records(state, &records);
    let pending = state.pending_discovers.get_mut(&discover_id).expect("present");
    let (settled, waiting): (Vec<_>, Vec<_>) = std::mem::take(&mut pending.waiters).into_iter().partition(|w| {
        waiter_settled(w.kind, done, pending.outstanding, pending.records.len(), pending.info_gen, w.info_at_join, w.deadline, now)
    });
    pending.waiters = waiting;
    for w in settled {
        let _ = w.reply.send(Ok(peers.clone()));
    }
    if done {
        let pending = state.pending_discovers.remove(&discover_id).expect("present");
        let fresh = settle_discover(state, &pending);
        // C11: remember this successful discover so the next request for the same model can
        // answer from `known_peers` without a fresh get_providers round-trip. Only cache a
        // non-empty result — an empty one has nothing to serve and should re-query.
        if fresh && !peers.is_empty() {
            state.discover_cache.insert(
                pending.model_id.clone(),
                (now_unix_ms(), state.net_generation.load(std::sync::atomic::Ordering::Relaxed)),
            );
        }
    }
}

/// A re-publish of our provider records for LAN peers that just subscribed: due a moment after the
/// first one (so its gossip subscription is in place), at most once per [`LAN_ANNOUNCE_EVERY`].
#[derive(Debug, Default)]
struct LanAnnounce {
    due: Option<std::time::Instant>,
    last: Option<std::time::Instant>,
}

/// How long after a LAN peer subscribes our records go out.
const LAN_ANNOUNCE_DELAY: std::time::Duration = std::time::Duration::from_millis(1500);
/// At most one LAN re-publish per this long (a burst of joins is one re-publish).
const LAN_ANNOUNCE_EVERY: std::time::Duration = std::time::Duration::from_secs(10);

fn note_lan_subscriber(a: &mut LanAnnounce, now: std::time::Instant) {
    if a.due.is_some() {
        return;
    }
    let earliest = a.last.map_or(now, |l| l + LAN_ANNOUNCE_EVERY);
    a.due = Some(earliest.max(now + LAN_ANNOUNCE_DELAY));
}

/// Whether the LAN re-publish is due at `now` (it then counts as done).
fn lan_announce_due(a: &mut LanAnnounce, now: std::time::Instant) -> bool {
    match a.due {
        Some(d) if now >= d => {
            a.due = None;
            a.last = Some(now);
            true
        }
        _ => false,
    }
}

/// Re-publish this node's own provider records on the gossip topic (a LAN peer just joined).
fn drive_lan_announce(swarm: &mut libp2p::Swarm<OpenHydraBehaviour>, state: &mut LoopState) {
    if !lan_announce_due(&mut state.lan_announce, std::time::Instant::now()) {
        return;
    }
    let own: Vec<PeerRecord> =
        state.known_peers.values().filter(|r| r.libp2p_peer_id == state.local_libp2p).cloned().collect();
    if !own.is_empty() {
        debug!(n = own.len(), "provider_pex: re-publishing for a LAN peer that just joined");
    }
    for record in &own {
        publish_provider_pex(swarm, record);
    }
}

/// The discover tick: answer waiting callers whose retry deadline has passed or whose lookup
/// learned something by gossip since ([`install_known_peer`] merges into running lookups).
fn drive_discover_waiters(state: &mut LoopState) {
    let ids: Vec<kad::QueryId> =
        state.pending_discovers.iter().filter(|(_, p)| !p.waiters.is_empty()).map(|(id, _)| *id).collect();
    for id in ids {
        maybe_reply_discover(state, id);
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

/// C7-flicker fix: the `known_peers` cache key. Keyed by **`(node, model_id)`**, not the node
/// alone — a single provider node announces one record PER model it serves, so keying on the node
/// made each model clobber the previous one and a multi-model provider could only ever surface ONE
/// model at a time (it flickered between them). The node is its **libp2p** peer id: verified
/// against the record's key and full-length, where the 64-bit OpenHydra id can be shared by two
/// keys found to collide, which would let one provider's entry shadow the other's. (The OpenHydra
/// id only if a record somehow has no libp2p id.) The unit separator can't appear in either id or a
/// model handle, so the composite is unambiguous.
fn known_peer_key(record: &PeerRecord) -> String {
    let node = if record.libp2p_peer_id.is_empty() { &record.peer_id } else { &record.libp2p_peer_id };
    format!("{}\u{1f}{}", node, record.model_id)
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
        // A public (DHT) discovery is never a private-swarm route.
        swarm_public_key: String::new(),
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
        .take(MAX_IDENTIFY_DIAL_ADDRS)
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
    reachable_addr_check(addr, SIM_172_16_IS_PUBLIC.load(std::sync::atomic::Ordering::Relaxed))
}

static SIM_172_16_IS_PUBLIC: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Simulator only (`oh_sim --vpc-is-public`): count 172.16.0.0/12 as globally reachable for every
/// node this process starts. The AWS simulation puts its "public" nodes on VPC addresses
/// (172.31.x), which [`is_globally_reachable_addr`] treats as private, so no simulated agent was ever promoted to a
/// DHT server and the simulated DHT was only the bootstraps (20 Nov). 10/8 and 192.168/16 stay
/// private: the simulation's NAT homes (10.x) and home LANs keep behaving as they do for real. Not
/// reachable from the agent's command line; a process-wide setting like [`crate::node::disable_mdns`].
pub fn sim_treat_172_16_as_public() {
    SIM_172_16_IS_PUBLIC.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// [`is_globally_reachable_addr`], with 172.16.0.0/12 counted as public when `vpc_public`.
fn reachable_addr_check(addr: &Multiaddr, vpc_public: bool) -> bool {
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
                let sim_public = vpc_public && o[0] == 172 && (o[1] & 0xf0) == 16;
                if (ip.is_private() && !sim_public)
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

/// An address on this machine or a private network: loopback, unspecified, RFC 1918, CGNAT
/// (100.64/10), link-local, IPv6 unique-local.
pub(crate) fn is_non_global_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || (v4.octets()[0] == 100 && (v4.octets()[1] & 0xc0) == 64)
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                || v6.to_ipv4_mapped().is_some_and(|v4| is_non_global_ip(v4.into()))
        }
    }
}

/// Whether a connection (by its remote address) is over loopback or a private network, not a relay.
fn is_local_link(addr_str: &str, ip: Option<&str>) -> bool {
    !addr_str.contains("p2p-circuit") && ip.and_then(|i| i.parse::<std::net::IpAddr>().ok()).is_some_and(is_non_global_ip)
}

/// Circuit addresses a peer's identify may add to the routing table (per peer, at most).
const MAX_IDENTIFY_CIRCUIT_ADDRS: usize = 8;
/// A trusted relay's own addresses taken from its identify (per relay, at most).
const MAX_RELAY_SELF_ADDRS: usize = 16;
/// Direct addresses kept from a peer's identify for our own dials to it (per peer, at most).
const MAX_IDENTIFY_DIAL_ADDRS: usize = 4;
/// Claimed IPv6 QUIC addresses dialled per hole-punch attempt (per peer, at most).
const MAX_HOLEPUNCH_V6_ADDRS: usize = 2;

/// A peer's circuit addresses from its identify, through a relay we use (the D-S5 check signed
/// records go through), so a peer behind a home router stays reachable. Kept for our own dials only
/// (the address book); the routing table takes no circuit (`swarm::kad_address_filter`, 21 Nov).
fn identify_routing_addrs(
    listen_addrs: &[Multiaddr],
    peer: &PeerId,
    trusted_relays: &crate::relay::TrustedRelays,
    ipv6_capable: bool,
) -> Vec<Multiaddr> {
    let mut out: Vec<Multiaddr> = Vec::new();
    for a in listen_addrs.iter().filter(|a| ipv6_capable || !is_ipv6_multiaddr(a)) {
        if let Some(ma) = crate::relay::safe_injectable_circuit_addr(&a.to_string(), peer, trusted_relays) {
            if !out.contains(&ma) {
                out.push(ma);
            }
        }
        if out.len() >= MAX_IDENTIFY_CIRCUIT_ADDRS {
            break;
        }
    }
    out
}

/// The direct addresses from a peer's identify kept for our own dials to it: only on the IP the
/// connection really comes from (`conn_addr`; none for a relayed one), so a peer can only point at
/// itself, and only admissible ones (a LAN peer's LAN address is fine for a LAN dial).
/// The routing-table addresses a peer's identify gives: for a DHT server, its direct addresses on its
/// connection's IP ([`identify_dial_addrs`]) that are globally reachable (others will dial them from
/// elsewhere); nothing for a peer that doesn't serve the DHT.
/// On a private network (`public_network` false) any of them, as Kademlia's own address filter
/// takes (review 22 Nov: a 10.x deployment otherwise got no routing entries this way).
fn kad_addrs_from_identify(serves_dht: bool, public_network: bool, direct: &[Multiaddr]) -> Vec<Multiaddr> {
    if !serves_dht {
        return Vec::new();
    }
    direct.iter().filter(|a| !public_network || is_globally_reachable_addr(a)).cloned().collect()
}

/// Whether a connection's close `cause` says the peer closed it: over QUIC, an application close
/// (a crashed process sends none: its connections time out, which is a loss); over TCP, the
/// peer's end of the socket closing (yamux's orderly end, a broken pipe, a reset). TCP cannot tell
/// a close the peer meant from its process dying (the kernel closes the socket either way): such a
/// crash counts as a close here, and the peer goes with the bootstraps' gone notice, its records'
/// age or Kademlia's failed dials. A timeout (the host vanished or moved) or our own close is not.
fn closed_by_peer(cause: Option<&libp2p::swarm::ConnectionError>) -> bool {
    let Some(libp2p::swarm::ConnectionError::IO(e)) = cause else { return false };
    let Some(inner) = e.get_ref() else { return false };
    if let Some(q) = inner.downcast_ref::<libp2p::quic::Error>() {
        return matches!(q, libp2p::quic::Error::Connection(c) if c.closed_by_peer());
    }
    // yamux: its orderly end has only its text to go by; a socket error is in its source chain.
    if inner.to_string() == "connection is closed" {
        return true;
    }
    let mut source = inner.source();
    while let Some(err) = source {
        if let Some(io) = err.downcast_ref::<std::io::Error>() {
            return matches!(
                io.kind(),
                std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset | std::io::ErrorKind::UnexpectedEof
            );
        }
        source = err.source();
    }
    false
}

fn identify_dial_addrs(
    listen_addrs: &[Multiaddr],
    conn_addr: Option<&Multiaddr>,
    local_link: bool,
    ipv6_capable: bool,
) -> Vec<Multiaddr> {
    let Some(conn_ip) = conn_addr.and_then(multiaddr_ip) else { return Vec::new() };
    listen_addrs
        .iter()
        .filter(|a| !a.iter().any(|p| matches!(p, libp2p::multiaddr::Protocol::P2pCircuit)))
        .filter(|a| ipv6_capable || !is_ipv6_multiaddr(a))
        .filter(|a| identify_addr_admissible(a, local_link))
        .filter(|a| multiaddr_ip(a) == Some(conn_ip))
        .take(MAX_IDENTIFY_DIAL_ADDRS)
        .cloned()
        .collect()
}

/// A peer's claimed IPv6 QUIC addresses worth a hole-punch dial: global unicast only (no loopback,
/// unique-local, link-local, multicast, documentation or IPv4-mapped), never a circuit, at most
/// [`MAX_HOLEPUNCH_V6_ADDRS`].
fn holepunch_v6_addrs(listen_addrs: &[Multiaddr]) -> Vec<Multiaddr> {
    use libp2p::multiaddr::Protocol;
    listen_addrs
        .iter()
        .filter(|a| {
            let mut v6 = None;
            let mut quic = false;
            for p in a.iter() {
                match p {
                    Protocol::Ip6(ip) => v6 = Some(ip),
                    Protocol::QuicV1 => quic = true,
                    Protocol::P2pCircuit | Protocol::Ip4(_) => return false,
                    _ => {}
                }
            }
            quic && v6.is_some_and(|ip| {
                let seg = ip.segments();
                !(ip.is_loopback()
                    || ip.is_unspecified()
                    || ip.is_multicast()
                    || ip.to_ipv4_mapped().is_some()
                    || (seg[0] & 0xfe00) == 0xfc00
                    || (seg[0] & 0xffc0) == 0xfe80
                    || (seg[0] == 0x2001 && seg[1] == 0x0db8))
            })
        })
        .take(MAX_HOLEPUNCH_V6_ADDRS)
        .cloned()
        .collect()
}

/// A hole-punch dial to a claimed address that only counts if this very peer answers.
fn holepunch_dial_opts(peer: PeerId, addr: Multiaddr) -> libp2p::swarm::dial_opts::DialOpts {
    use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
    DialOpts::peer_id(peer).addresses(vec![addr]).condition(PeerCondition::Always).build()
}

/// The IP in a multiaddr, if it has one.
fn multiaddr_ip(addr: &Multiaddr) -> Option<std::net::IpAddr> {
    addr.iter().find_map(|p| match p {
        libp2p::multiaddr::Protocol::Ip4(ip) => Some(ip.into()),
        libp2p::multiaddr::Protocol::Ip6(ip) => Some(ip.into()),
        _ => None,
    })
}

/// Whether a peer's identify listen address goes into the routing table, where other nodes learn it
/// and dial it. A local-only transport (a Unix socket, memory) never does; an unspecified address
/// never does; a loopback or private-network address only from a peer we reach over loopback or the
/// same private network (`local_link`) — from anyone else it would point other nodes at their own
/// machine or LAN.
fn identify_addr_admissible(addr: &Multiaddr, local_link: bool) -> bool {
    use libp2p::multiaddr::Protocol;
    let mut ip = None;
    for p in addr.iter() {
        match p {
            Protocol::Unix(_) | Protocol::Memory(_) => return false,
            Protocol::Ip4(v4) if ip.is_none() => ip = Some(std::net::IpAddr::V4(v4)),
            Protocol::Ip6(v6) if ip.is_none() => ip = Some(std::net::IpAddr::V6(v6)),
            _ => {}
        }
    }
    match ip {
        Some(ip) if ip.is_unspecified() => false,
        Some(ip) if is_non_global_ip(ip) => local_link,
        _ => true,
    }
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

/// Most proxy requests out to one peer at once (#31). A connection carries at most
/// [`crate::proxy::MAX_CONCURRENT_STREAMS`] requests both ways together, and past it libp2p fails
/// the request ("max sub-streams reached"); held stream polls (long-poll) and queued opens each
/// keep one open for seconds. Below that, leaving room for the peer's requests to us; more wait
/// here for one to finish.
const MAX_PROXY_INFLIGHT_PER_PEER: usize = crate::proxy::MAX_CONCURRENT_STREAMS * 3 / 4;
/// Most proxy requests waiting for one peer; past it a request fails at once.
const MAX_PROXY_WAITING_PER_PEER: usize = 1024;

/// A proxy request waiting for a free slot to its peer: the bytes, and where its reply goes
/// (`None`: fire-and-forget).
type WaitingProxy = (Vec<u8>, Option<oneshot::Sender<Result<Vec<u8>, String>>>);

/// libp2p's error for a request refused because its connection already carries as many as it may.
fn is_local_stream_limit(e: &std::io::Error) -> bool {
    e.to_string().contains("max sub-streams reached")
}

/// multistream-select's error for a peer that refused the protocol: with lazy negotiation (the
/// patched swarm, for OpenHydra's request protocols) it arrives as an I/O error once the reply is
/// read, not as `UnsupportedProtocols`.
fn is_negotiation_refusal(e: &std::io::Error) -> bool {
    e.to_string().contains("Protocol negotiation failed")
}

/// Marks a proxy failure whose request never reached the peer's application, so the caller knows
/// nothing was started there (the consumer then neither resumes nor closes a stream).
pub const NEVER_SENT: &str = "(request never sent)";

/// The error text a failed proxy request is answered with: its failure, marked [`NEVER_SENT`] when
/// the request can't have reached the peer's application.
fn proxy_failure_text(error: &request_response::OutboundFailure) -> String {
    let never_sent = match error {
        request_response::OutboundFailure::DialFailure | request_response::OutboundFailure::UnsupportedProtocols => true,
        request_response::OutboundFailure::Io(e) => is_local_stream_limit(e) || is_negotiation_refusal(e),
        _ => false,
    };
    if never_sent {
        format!("proxy outbound: {error:?} {NEVER_SENT}")
    } else {
        format!("proxy outbound: {error:?}")
    }
}

/// Send a proxy request to `peer` now, or queue it until one of the peer's in-flight requests
/// finishes when [`MAX_PROXY_INFLIGHT_PER_PEER`] are out (#31).
fn send_proxy_request(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    peer: PeerId,
    data: Vec<u8>,
    reply: Option<oneshot::Sender<Result<Vec<u8>, String>>>,
) {
    state.app_peers.note(peer);
    if state.proxy_inflight.get(&peer).copied().unwrap_or(0) >= proxy_slots(state, peer) {
        let waiting = state.proxy_waiting.entry(peer).or_default();
        if waiting.len() >= MAX_PROXY_WAITING_PER_PEER {
            if let Some(reply) = reply {
                let _ = reply.send(Err("proxy_forward: too many requests waiting for this peer".into()));
            }
            return;
        }
        debug!(%peer, waiting = waiting.len() + 1, "proxy request waits for a free slot to its peer");
        waiting.push_back((data, reply));
        return;
    }
    let req_id = swarm.behaviour_mut().grpc_proxy.send_request(&peer, ProxyRequest(data));
    *state.proxy_inflight.entry(peer).or_default() += 1;
    state.proxy_request_peer.insert(req_id, peer);
    if let Some(reply) = reply {
        state.pending_proxy.insert(req_id, reply);
    }
}

/// How many proxy requests may be out to `peer` at once: [`MAX_PROXY_INFLIGHT_PER_PEER`], less any
/// room the peer's own requests to us are taking on the shared per-connection limit.
fn proxy_slots(state: &LoopState, peer: PeerId) -> usize {
    let inbound = state.inbound_proxy_channels.values().filter(|(_, _, p, _)| *p == peer).count();
    MAX_PROXY_INFLIGHT_PER_PEER.min(crate::proxy::MAX_CONCURRENT_STREAMS.saturating_sub(inbound + 1)).max(1)
}

/// A proxy request was answered or failed: its peer has a slot free, which the oldest request
/// still wanted takes (one whose caller has given up is dropped, not sent). Only when `usable`
/// (the peer is connected and that connection isn't being closed): otherwise every request still
/// waiting for the peer takes the reconnect path instead (one dial, queued for the connection it
/// opens), rather than each going out on a connection that is gone and failing at once (#31
/// review: a dropped connection drained the whole queue as dial failures).
fn proxy_request_finished(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    proxy_queue: &SharedProxyQueue,
    request_id: request_response::OutboundRequestId,
    usable: bool,
) {
    let Some(peer) = state.proxy_request_peer.remove(&request_id) else { return };
    if let Some(n) = state.proxy_inflight.get_mut(&peer) {
        *n = n.saturating_sub(1);
        if *n == 0 {
            state.proxy_inflight.remove(&peer);
        }
    }
    if !usable {
        let Some(waiting) = state.proxy_waiting.remove(&peer) else { return };
        // A connection still up is being closed (zombie eviction, which also dials the peer again),
        // and a dial already under way will bring a connection: either way queue for that one.
        let mut dialing = swarm.is_connected(&peer)
            || state.dials_in_flight.get(&peer).is_some_and(|d| !d.is_empty());
        for (data, reply) in waiting {
            let Some(reply) = reply.filter(|r| !r.is_closed()) else { continue };
            if !dialing {
                // The first one dials (or fails as a dial would, in back-off) and queues itself.
                handle_proxy_forward(swarm, &peer.to_string(), data, reply, state, proxy_queue);
                dialing = state.pending_relay_forwards.iter().any(|(p, ..)| *p == peer);
            } else if state.pending_relay_forwards.len() >= MAX_PENDING_RELAY_FORWARDS {
                let _ = reply.send(Err("proxy_forward: relay queue full".into()));
            } else {
                state.pending_relay_forwards.push((peer, data, reply, std::time::Instant::now()));
            }
        }
        return;
    }
    while state.proxy_inflight.get(&peer).copied().unwrap_or(0) < proxy_slots(state, peer) {
        let Some(waiting) = state.proxy_waiting.get_mut(&peer) else { break };
        let Some((data, reply)) = waiting.pop_front() else {
            state.proxy_waiting.remove(&peer);
            break;
        };
        if waiting.is_empty() {
            state.proxy_waiting.remove(&peer);
        }
        if reply.as_ref().is_some_and(|r| r.is_closed()) {
            continue;
        }
        send_proxy_request(swarm, state, peer, data, reply);
    }
}

fn handle_grpc_proxy_event(
    event: request_response::Event<ProxyRequest, ProxyResponse>,
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    proxy_queue: &SharedProxyQueue,
) {
    match event {
        request_response::Event::Message { peer, message } => {
            // A peer we exchange inference requests with: its connections are never closed as idle.
            state.app_peers.note(peer);
            match message {
                request_response::Message::Request { request_id: rid, request, channel } => {
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
                        .filter(|(_, _, p, _)| *p == peer)
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
                    state.inbound_proxy_channels.insert(req_id, (channel, std::time::Instant::now(), peer.clone(), rid));
                }
                request_response::Message::Response { request_id, response } => {
                    let usable = swarm.is_connected(&peer);
                    proxy_request_finished(swarm, state, proxy_queue, request_id, usable);
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
                let _ = reply.send(Err(proxy_failure_text(&error)));
            }
            // #zombie (#42 live-roam finding): `send_request` picks ANY existing
            // connection, so zombie pre-roam connections mask a roamed peer's
            // fresh path until they reap (observed: 6+ min of timeouts). Classify
            // path failures vs protocol failures; after ZOMBIE_FAILURE_THRESHOLD
            // consecutive path failures, force-close every connection to the
            // peer and redial fresh relay circuits so the NEXT dispatch rides a
            // live path (the roamed peer's new reservation is reachable — its
            // re-announced record was never the problem, our stale conns were).
            let dead_path = match &error {
                request_response::OutboundFailure::Timeout
                | request_response::OutboundFailure::ConnectionClosed
                | request_response::OutboundFailure::DialFailure => true,
                // The connection's own limit on requests at once says nothing about the path
                // (#31): counting it closed a busy gateway's connections to a healthy provider.
                // Nor does a peer that doesn't speak the protocol (lazy negotiation reports it as an
                // I/O error, not as UnsupportedProtocols): it is alive and answering.
                request_response::OutboundFailure::Io(e) => !is_local_stream_limit(e) && !is_negotiation_refusal(e),
                _ => false,
            };
            let evicted = record_proxy_failure(&mut state.proxy_failure_streak, peer, dead_path);
            // Its slot goes to the next waiting request only over a connection still up and not
            // being closed below; otherwise the waiting ones take the reconnect path (#31 review).
            let usable = !evicted && swarm.is_connected(&peer);
            proxy_request_finished(swarm, state, proxy_queue, request_id, usable);
            if evicted {
                warn!(%peer, threshold = ZOMBIE_FAILURE_THRESHOLD,
                    "zombie_evict: consecutive proxy path failures — closing presumed-dead connections and redialing via relay");
                let _ = swarm.disconnect_peer_id(peer);
                // The own provider is redialled on its loopback address, never through a relay.
                let pinned = pinned_addrs(state, &peer);
                let opts = if pinned.is_empty() {
                    relay_dial_opts(peer, relay_circuit_addrs(&state.relays, peer, state.ipv6_capable))
                } else {
                    pinned_dial_opts(peer, pinned, libp2p::swarm::dial_opts::PeerCondition::Always)
                };
                if let Err(e) = swarm.dial_tracked(&mut state.dials_in_flight, opts) {
                    debug!(%peer, %e, "zombie_evict: relay redial enqueue failed");
                }
            }
        }
        request_response::Event::InboundFailure { error, request_id, .. } => {
            warn!(?error, "proxy inbound failure");
            state.responses_writing.remove(&request_id);
            if let Some((f, _, _)) = state.pending_sent.remove(&request_id) {
                f(false);
            }
        }
        request_response::Event::ResponseSent { peer, request_id, .. } => {
            // Noted again when the reply goes out: a long request must not leave the peer looking
            // unused (review of #16, finding 2).
            state.app_peers.note(peer);
            state.responses_writing.remove(&request_id);
            if let Some((f, _, _)) = state.pending_sent.remove(&request_id) {
                f(true);
            }
        }
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
        send_proxy_request(swarm, state, peer_id, data, Some(reply));
        // A4: the provider dials us back over IPv6 once it accepts this request; punching toward
        // it now too means both ends' firewalls see outbound packets at the same time, which is
        // what a hole punch through two stateful firewalls needs (its rounds were one-sided).
        if direct_conns == 0 {
            pursue_v6_holepunch_debounced(swarm, state, peer_id, "request");
            // The own provider reached only through a relay (a dial elsewhere found its relay
            // address): this request rides it, the next one the loopback connection this opens.
            dial_pinned_if_only_relayed(swarm, state, peer_id);
        }
    } else {
        info!(%peer_id, "proxy_forward: peer not connected, dialing it");
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
        let pinned = pinned_addrs(state, &peer_id);
        if let Some(b) = state
            .relay_dial_backoff
            .get(&peer_id)
            .filter(|b| pinned.is_empty() && b.until > std::time::Instant::now())
        {
            // A dial to it just failed: answer as that dial did, without spending relay budget.
            if state.dials_in_flight.get(&peer_id).map_or(true, |d| d.is_empty()) {
                debug!(%peer_id, "proxy_forward: peer in relayed-dial back-off — failing at once");
                let _ = reply.send(Err(format!("proxy_forward: {}", b.why)));
                return;
            }
        }
        // The own provider is dialled on its loopback addresses alone: a relay circuit can't reach it
        // when they can't (it's on this machine), and waiting for circuits to time out made every
        // request fail 5 s late while it was down.
        let opts = if pinned.is_empty() {
            relay_dial_opts(peer_id, relay_circuit_addrs(&state.relays, peer_id, state.ipv6_capable))
        } else {
            pinned_dial_opts(peer_id, pinned, libp2p::swarm::dial_opts::PeerCondition::Disconnected)
        };
        match swarm.dial_tracked(&mut state.dials_in_flight, opts) {
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
    // A global address in the IPv6 documentation range (2001:db8::/32, RFC 3849):
    // the kernel routes it by the default route like any global address, so the
    // answer is the same as for a real one, and since no packet is sent, nobody
    // is contacted - nor seems to be in an audit of what this node connects to
    // (it was Cloudflare's DNS, review 2026-10-14). Port irrelevant.
    let ok = UdpSocket::bind("[::]:0")
        .and_then(|s| s.connect("[2001:db8::1]:9"))
        .is_ok();
    info!(ipv6_capable = ok, "F-9: outbound IPv6 capability probe");
    ok
}

/// Finding A helper: build a `/<relay>/p2p-circuit/p2p/<target>` multiaddr for
/// every known bootstrap relay, so a single dial can race all of them. F-9:
/// when the host has no working IPv6, drop the `/ip6/` relay circuits — dialing
/// them would only burn timeouts on unreachable addresses.
fn relay_circuit_addrs(relays: &[Multiaddr], peer_id: PeerId, ipv6_capable: bool) -> Vec<Multiaddr> {
    relays
        .iter()
        .filter(|relay| ipv6_capable || !is_ipv6_multiaddr(relay))
        .map(|relay| {
            relay
                .clone()
                .with(libp2p::multiaddr::Protocol::P2pCircuit)
                .with(libp2p::multiaddr::Protocol::P2p(peer_id))
        })
        .collect()
}

/// The addresses `peer` is pinned to that this node can dial now ([`crate::pinned::dialable`]).
fn pinned_addrs(state: &LoopState, peer: &PeerId) -> Vec<Multiaddr> {
    let quic = crate::transport::quic_enabled();
    let m = state.pinned.read().unwrap_or_else(|e| e.into_inner());
    m.get(peer).map(|a| crate::pinned::dialable(a, quic)).unwrap_or_default()
}

/// A dial to a pinned peer on exactly `addrs`: no relay circuit, and none of the addresses the
/// behaviours know for it.
fn pinned_dial_opts(
    peer: PeerId,
    addrs: Vec<Multiaddr>,
    condition: libp2p::swarm::dial_opts::PeerCondition,
) -> libp2p::swarm::dial_opts::DialOpts {
    libp2p::swarm::dial_opts::DialOpts::peer_id(peer).condition(condition).addresses(addrs).build()
}

/// Pin `peer` to `addrs` ([`SwarmCommand::PinPeerAddrs`]) and dial them at once if it isn't connected
/// directly, so the own provider is connected before the first request needs it. Refused when none
/// of the addresses can be dialled here (a QUIC-only pin with QUIC off): the peer would silently go
/// back to relay dials.
fn pin_peer_addrs(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    peer: PeerId,
    addrs: Vec<Multiaddr>,
) -> Result<(), String> {
    if peer == *swarm.local_peer_id() {
        return Err("can't pin this node's own peer id".into());
    }
    if addrs.is_empty() {
        state.pinned.write().unwrap_or_else(|e| e.into_inner()).remove(&peer);
        info!(%peer, "pinned addresses cleared");
        return Ok(());
    }
    // Checked again here (the handle checks them too): only loopback ever gets pinned.
    let addrs = addrs
        .iter()
        .map(|a| crate::pinned::self_provider_addr(a, &peer))
        .collect::<Result<Vec<_>, String>>()?;
    let dial = crate::pinned::dialable(&addrs, crate::transport::quic_enabled());
    if dial.is_empty() {
        return Err("none of these addresses can be dialled on this network (QUIC is off here: give a TCP address)".into());
    }
    state.pinned.write().unwrap_or_else(|e| e.into_inner()).insert(peer, addrs);
    state.relay_dial_backoff.remove(&peer);
    purge_refused_kad_addrs(swarm, state, peer);
    info!(%peer, addrs = ?dial, "pinned addresses set");
    if !swarm.is_connected(&peer) {
        let opts = pinned_dial_opts(peer, dial, libp2p::swarm::dial_opts::PeerCondition::DisconnectedAndNotDialing);
        if let Err(e) = swarm.dial_tracked(&mut state.dials_in_flight, opts) {
            // Not an error for the caller: the next request dials again.
            debug!(%peer, error = %e, "pinned dial not started");
        }
    } else {
        dial_pinned_if_only_relayed(swarm, state, peer);
    }
    Ok(())
}

/// Every address Kademlia holds for `peer` that its filter now refuses ([`crate::pinned::kad_allows`])
/// goes: one restored from the routing cache or added before the pin (any loopback port or family,
/// not just the pinned ones). A configured bootstrap is exempt, as in the filter.
fn purge_refused_kad_addrs(swarm: &mut libp2p::Swarm<OpenHydraBehaviour>, state: &LoopState, peer: PeerId) {
    let bootstraps = state.bootstrap_peers.iter().map(|(b, _)| *b).collect();
    let n = crate::pinned::purge_refused(&mut swarm.behaviour_mut().kademlia, &state.pinned, &bootstraps, peer);
    if n > 0 {
        info!(%peer, n, "removed the pinned peer's loopback addresses from the routing table");
    }
}

/// When `peer` is pinned and connected only through relays, dial its pinned addresses (unless a dial
/// to it is already under way), so its traffic moves to the direct loopback connection.
fn dial_pinned_if_only_relayed(swarm: &mut libp2p::Swarm<OpenHydraBehaviour>, state: &mut LoopState, peer: PeerId) {
    let pinned = pinned_addrs(state, &peer);
    let direct = state.peer_connections.get(&peer).map_or(0, |i| i.direct_count());
    if pinned.is_empty() || direct > 0 || state.dials_in_flight.get(&peer).is_some_and(|d| !d.is_empty()) {
        return;
    }
    info!(%peer, "pinned peer connected only through a relay: dialling its pinned address");
    let opts = pinned_dial_opts(peer, pinned, libp2p::swarm::dial_opts::PeerCondition::NotDialing);
    if let Err(e) = swarm.dial_tracked(&mut state.dials_in_flight, opts) {
        debug!(%peer, error = %e, "pinned dial not started");
    }
}

/// Finding A helper: a single multi-address dial to `peer_id` over the given
/// relay-circuit addresses. `PeerCondition::Disconnected` so we only dial when
/// there is no established connection (we're in the not-connected branch).
/// The addresses the behaviours know for the peer (its Kademlia routing-table
/// entry, i.e. direct listen addresses) are raced alongside the circuits —
/// without `extend_addresses_through_behaviour` libp2p discards them and a
/// directly reachable peer is only ever tried through a relay.
fn relay_dial_opts(
    peer_id: PeerId,
    addrs: Vec<Multiaddr>,
) -> libp2p::swarm::dial_opts::DialOpts {
    use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
    DialOpts::peer_id(peer_id)
        .condition(PeerCondition::Disconnected)
        .addresses(addrs)
        .extend_addresses_through_behaviour()
        // The circuits come first, and on a dual-stack host there are eight (four relays, both
        // families): at libp2p's default of eight dials at once, the direct addresses only got a turn
        // after a circuit failed. Room for them to race too (review 2026-10-14, L10).
        .override_dial_concurrency_factor(RELAY_DIAL_CONCURRENCY)
        .build()
}

/// How many of a relay dial's addresses are tried at once (see [`relay_dial_opts`]).
const RELAY_DIAL_CONCURRENCY: std::num::NonZeroU8 = match std::num::NonZeroU8::new(16) {
    Some(n) => n,
    None => unreachable!(),
};

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
        send_proxy_request(swarm, state, peer_id, data, None);
    } else {
        info!(%peer_id, "proxy_forward_no_wait: peer not connected, dialing it");
        // Finding A: one multi-address dial across all relay circuits (see
        // handle_proxy_forward) instead of break-on-first-queued (always US).
        // C-N2: same cap guard as the blocking path (fire-and-forget, so just
        // drop rather than reply on overflow).
        if state.pending_relay_forwards.len() >= MAX_PENDING_RELAY_FORWARDS {
            warn!(%peer_id, cap = MAX_PENDING_RELAY_FORWARDS,
                  "proxy_forward_no_wait: pending_relay_forwards at cap — dropping");
            return;
        }
        let pinned = pinned_addrs(state, &peer_id);
        if pinned.is_empty()
            && state.relay_dial_backoff.get(&peer_id).is_some_and(|b| b.until > std::time::Instant::now())
            && state.dials_in_flight.get(&peer_id).map_or(true, |d| d.is_empty())
        {
            debug!(%peer_id, "proxy_forward_no_wait: peer in relayed-dial back-off — dropped");
            return;
        }
        let opts = if pinned.is_empty() {
            relay_dial_opts(peer_id, relay_circuit_addrs(&state.relays, peer_id, state.ipv6_capable))
        } else {
            pinned_dial_opts(peer_id, pinned, libp2p::swarm::dial_opts::PeerCondition::Disconnected)
        };
        match swarm.dial_tracked(&mut state.dials_in_flight, opts) {
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
    state: &mut LoopState,
) {
    let target: PeerId = match target_peer_id.parse() {
        Ok(p) => p,
        Err(e) => {
            let _ = reply.send(Err(format!("invalid peer_id: {e}")));
            return;
        }
    };

    // Dial the peer so the connection is ready when proxy requests arrive.
    if let Err(e) = swarm.dial_tracked(&mut state.dials_in_flight, target) {
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

/// Dial a peer's claimed IPv6 QUIC addresses (stored from its identify) to get a v6 direct
/// connection, unless we already hold one. Callers decide whether there's a reason to:
///
/// * `identify`: the peer is a provider we know (consumer side);
/// * `serve`: the agent accepted a serve request from the peer, every gate passed (provider side;
///   [`SwarmCommand::ServeAccepted`]). Home routers commonly drop unsolicited inbound v6 (ACT does),
///   so a consumer's dials alone never reach a provider behind one; the provider's own dial is what
///   opens its firewall to the consumer. Measured 29 Oct, ACT Asus: only the other side sending, 0
///   of 14 packets arrive; both, they pass within 0.33 s. Only an accepted serve counts — not any
///   bytes a connected peer sends (review 2026-10-29);
/// * `record`: a provider's record arrived after its identify, so the identify trigger missed it.
///
/// The dial names the peer, so only that peer can count as answering (the packets go out regardless);
/// public relays are skipped; nothing without working IPv6; at most `MAX_QUIC_HOLEPUNCH_ATTEMPTS`
/// rounds per peer until it disconnects, a v6 direct forms, or the network changes (`rebootstrap`
/// clears the counts), and at most [`MAX_PUNCH_ROUNDS_PER_MINUTE`] rounds node-wide.
fn pursue_v6_holepunch(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    peer_id: PeerId,
    trigger: &'static str,
) {
    let targets = v6_holepunch_targets(state, peer_id);
    dial_v6_targets(swarm, state, peer_id, targets, trigger);
}

fn dial_v6_targets(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    peer_id: PeerId,
    targets: Vec<Multiaddr>,
    trigger: &'static str,
) {
    for addr in targets {
        let ma = addr.clone();
        // The peer's id is required: whoever answers at a claimed address must be this peer. Not
        // counted as a dial in flight: a punch failing says nothing about whether the peer can be
        // reached, so it mustn't keep a queued forward waiting for its outcome (A4 review).
        match swarm.dial(holepunch_dial_opts(peer_id, addr)) {
            Ok(()) => {
                state.quic_v6_holepunch_dials += 1;
                debug!(%peer_id, %ma, trigger, "auto_quic_holepunch_dial");
            }
            Err(e) => debug!(%peer_id, %ma, %e, trigger, "auto_quic_holepunch_dial_failed"),
        }
    }
}

/// F7 / #43-W1: punch rounds per peer. The cap was 3; raised to 8 for the v6 case: a
/// stateful-firewall v6 punch is reliable once correctly timed (no port translation, unlike v4
/// CGNAT), so a firewalled-v6 peer shouldn't be shelved after 3 uncoordinated misses.
const MAX_QUIC_HOLEPUNCH_ATTEMPTS: u32 = 8;

/// Node-wide cap on IPv6 hole-punch rounds a minute (each dials at most
/// [`MAX_HOLEPUNCH_V6_ADDRS`] addresses), whatever the trigger: these dial addresses peers claim, so
/// many peers (or many peer ids) can't turn us into a packet source at their command.
const MAX_PUNCH_ROUNDS_PER_MINUTE: u32 = 30;

/// Store `peer`'s claimed IPv6 QUIC addresses for hole-punching, leaving out any another connected
/// peer already claims (one host can't honestly be two peers at one address and port; dialling such
/// an address for each of many peer ids would multiply the packets sent to it). Returns whether any
/// were stored. The trade-off: the first connected claimant keeps an address, so a peer claiming a
/// host's address first blocks punches toward that host while it stays connected, and a node that
/// restarts with a new peer id while its old connection lingers gets none until it re-identifies.
fn store_claimed_v6_addrs(state: &mut LoopState, peer: PeerId, addrs: &[Multiaddr]) -> bool {
    let fresh: Vec<Multiaddr> = addrs
        .iter()
        .filter(|a| !state.peer_quic_addrs.iter().any(|(p, held)| *p != peer && held.contains(a)))
        .cloned()
        .collect();
    if fresh.is_empty() {
        return false;
    }
    state.peer_quic_addrs.insert(peer, fresh);
    true
}

/// The addresses [`pursue_v6_holepunch`] dials now (counting the round), or none.
fn v6_holepunch_targets(state: &mut LoopState, peer_id: PeerId) -> Vec<Multiaddr> {
    // F-9: no working IPv6 here (re-probed on every rebootstrap), so nothing to dial.
    if !state.ipv6_capable {
        return Vec::new();
    }
    let Some(addrs) = state.peer_quic_addrs.get(&peer_id).filter(|a| !a.is_empty()).cloned() else {
        return Vec::new();
    };
    // v6-first: a v4 QUIC direct doesn't count as done.
    if !should_pursue_v6_upgrade(state.peer_connections.get(&peer_id)) {
        state.quic_holepunch_attempts.remove(&peer_id);
        return Vec::new();
    }
    // F7: public bootstrap relays are reachable as they are; punching them churned connections
    // on UDP-hostile paths.
    let is_relay = addrs.iter().any(|a| {
        a.iter().any(|p| match p {
            libp2p::multiaddr::Protocol::Ip6(ip) => crate::relay::is_bootstrap_relay_ip(&ip.to_string()),
            libp2p::multiaddr::Protocol::Ip4(ip) => crate::relay::is_bootstrap_relay_ip(&ip.to_string()),
            _ => false,
        })
    });
    if is_relay {
        debug!(%peer_id, "auto_quic_holepunch_skip: public relay");
        return Vec::new();
    }
    let attempts = state.quic_holepunch_attempts.entry(peer_id).or_insert(0);
    if *attempts >= MAX_QUIC_HOLEPUNCH_ATTEMPTS {
        debug!(%peer_id, attempts = *attempts, "auto_quic_holepunch_skip: backed off");
        return Vec::new();
    }
    let now = std::time::Instant::now();
    if now.duration_since(state.punch_window.0) >= std::time::Duration::from_secs(60) {
        state.punch_window = (now, 0);
    }
    if state.punch_window.1 >= MAX_PUNCH_ROUNDS_PER_MINUTE {
        debug!(%peer_id, "auto_quic_holepunch_skip: node-wide budget spent");
        return Vec::new();
    }
    state.punch_window.1 += 1;
    *attempts += 1;
    addrs
}

/// [`pursue_v6_holepunch`] for the triggers that can fire in bursts (every request; every record
/// refresh): at most once per [`REPUNCH_DEBOUNCE`] per peer, shared with `TriggerRepunch`.
fn pursue_v6_holepunch_debounced(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    peer_id: PeerId,
    trigger: &'static str,
) {
    let targets = debounced_v6_targets(state, peer_id, std::time::Instant::now());
    dial_v6_targets(swarm, state, peer_id, targets, trigger);
}

/// [`v6_holepunch_targets`] for a debounced trigger at `now`: none while the peer's wait runs; the
/// wait starts only when a punch actually goes out (a refused one — budget, cap, no IPv6 — doesn't
/// use it up).
fn debounced_v6_targets(state: &mut LoopState, peer_id: PeerId, now: std::time::Instant) -> Vec<Multiaddr> {
    if !state.peer_quic_addrs.contains_key(&peer_id)
        || !should_pursue_v6_upgrade(state.peer_connections.get(&peer_id))
        || state.last_repunch.get(&peer_id).is_some_and(|t| now.duration_since(*t) < REPUNCH_DEBOUNCE)
    {
        return Vec::new();
    }
    let targets = v6_holepunch_targets(state, peer_id);
    if !targets.is_empty() {
        state.last_repunch.insert(peer_id, now);
    }
    targets
}

// ── Fix 4: TriggerRepunch with debounce ────────────────────────────────

/// A re-punch asked for (the agent's `TriggerRepunch`, or a forward that just rode a new relay
/// connection): the same guards as every other punch (working IPv6, no direct path yet, the
/// per-peer cap, the node-wide budget, the debounce). It used to dial the peer's claimed addresses
/// past all of them (A4 review 2026-10-29).
fn handle_trigger_repunch(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    peer_id: PeerId,
    state: &mut LoopState,
) {
    pursue_v6_holepunch_debounced(swarm, state, peer_id, "repunch");
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

    /// The relays' circuits as a reservation on EU reports them after a roam (28/29 Oct logs): one
    /// per transport, all through the same relay.
    fn eu_circuits(me: &PeerId) -> Vec<Multiaddr> {
        let eu = "12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT";
        ["/ip4/172.105.69.49/tcp/4001", "/ip4/172.105.69.49/udp/4001/quic-v1", "/ip6/2a01:7e01::2000:84ff:fec5:4520/tcp/4001", "/ip6/2a01:7e01::2000:84ff:fec5:4520/udp/4001/quic-v1"]
            .iter()
            .map(|a| format!("{a}/p2p/{eu}/p2p-circuit/p2p/{me}").parse().unwrap())
            .collect()
    }

    fn eu() -> PeerId {
        "12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT".parse().unwrap()
    }

    #[test]
    fn startup_tcp_dials_wait_for_a_listen_address_of_their_family() {
        let a = PeerId::random();
        let b = PeerId::random();
        let v4: Multiaddr = "/ip4/45.79.190.172/tcp/4001".parse().unwrap();
        let v6: Multiaddr = "/ip6/2600:3c03::1/tcp/4001".parse().unwrap();
        let dns: Multiaddr = "/dns/bootstrap-us.openhydra.co/tcp/4001".parse().unwrap();
        let lo: Multiaddr = "/ip4/127.0.0.1/tcp/4001".parse().unwrap();
        let mut pending = vec![(a, v4.clone()), (a, v6.clone()), (b, dns.clone()), (b, lo.clone())];
        // Loopback listen addresses serve only loopback dials; QUIC and circuit ones none.
        assert_eq!(startup_dials_ready(&mut pending, &"/ip6/::1/tcp/4987".parse().unwrap()), vec![]);
        assert_eq!(startup_dials_ready(&mut pending, &"/ip4/192.168.0.3/udp/4987/quic-v1".parse().unwrap()), vec![]);
        assert_eq!(startup_dials_ready(&mut pending, &"/ip4/127.0.0.1/tcp/4987".parse().unwrap()), vec![(b, lo)]);
        // The v4 interface address: v4 and the DNS name go, v6 waits for its own.
        assert_eq!(startup_dials_ready(&mut pending, &"/ip4/192.168.0.3/tcp/4987".parse().unwrap()), vec![(a, v4), (b, dns)]);
        assert_eq!(pending, vec![(a, v6.clone())]);
        assert_eq!(startup_dials_ready(&mut pending, &"/ip6/2406:7400:56::3/tcp/4987".parse().unwrap()), vec![(a, v6)]);
        assert!(pending.is_empty());
    }

    #[test]
    fn a_rebootstrap_skips_bootstraps_already_connected_on_that_transport_since_the_last_move() {
        let mut state = LoopState::new();
        let a = PeerId::random();
        let b = PeerId::random();
        let a4: Multiaddr = "/ip4/45.79.190.172/tcp/4001".parse().unwrap();
        let a6: Multiaddr = "/ip6/2600:3c03::1/tcp/4001".parse().unwrap();
        let b4: Multiaddr = "/dns4/bootstrap-eu.openhydra.co/tcp/4001".parse().unwrap();
        state.bootstrap_peers = vec![(a, a4.clone()), (a, a6.clone()), (b, b4.clone())];
        let conn = libp2p::swarm::ConnectionId::new_unchecked;
        let t0 = std::time::Instant::now();
        // a over v4 TCP; b only over v4 QUIC (another transport); a over v6 QUIC (another transport).
        state.conn_remote_addrs.insert(conn(1), (a, "/ip4/45.79.190.172/tcp/4001".parse().unwrap(), t0));
        state.conn_remote_addrs.insert(conn(2), (b, "/ip4/172.105.69.49/udp/4001/quic-v1".parse().unwrap(), t0));
        state.conn_remote_addrs.insert(conn(3), (a, "/ip6/2600:3c03::1/udp/4001/quic-v1".parse().unwrap(), t0));
        assert_eq!(bootstrap_redials(&state), vec![(a, a6.clone()), (b, b4.clone())]);
        // A /dns4 address is connected once a v4 TCP connection to that peer is.
        state.conn_remote_addrs.insert(conn(4), (b, "/ip4/172.105.69.49/tcp/4001".parse().unwrap(), t0));
        assert_eq!(bootstrap_redials(&state), vec![(a, a6.clone())]);
        // After we lost an interface address, older connections don't count: dial everything.
        state.last_addr_loss = Some(t0 + std::time::Duration::from_secs(1));
        assert_eq!(bootstrap_redials(&state).len(), 3);
        // A connection made since then does.
        state.conn_remote_addrs.insert(conn(5), (a, "/ip4/45.79.190.172/tcp/4001".parse().unwrap(), t0 + std::time::Duration::from_secs(2)));
        assert_eq!(bootstrap_redials(&state), vec![(a, a6), (b, b4)]);
    }

    #[test]
    fn a_departure_evicts_only_the_records_it_is_newer_than() {
        let rec = |peer: &str, model: &str, ms: u64| PeerRecord { libp2p_peer_id: peer.into(), model_id: model.into(), updated_unix_ms: ms, ..Default::default() };
        let mut known = HashMap::new();
        for r in [rec("P", "a", 1_000), rec("P", "b", 0), rec("P", "c", 5_000), rec("Q", "a", 1_000)] {
            known.insert(known_peer_key(&r), r);
        }
        // P departed at 3 s, then came back and announced "c" at 5 s.
        assert_eq!(evict_departed(&mut known, "P", 3_000), 2, "the older one and the undated one");
        let mut left: Vec<_> = known.values().map(|r| (r.libp2p_peer_id.clone(), r.model_id.clone())).collect();
        left.sort();
        assert_eq!(left, vec![("P".to_string(), "c".to_string()), ("Q".to_string(), "a".to_string())]);
    }

    #[test]
    fn an_undated_record_is_checked_and_renewed_every_time_it_comes() {
        let key = libp2p::identity::Keypair::generate_ed25519();
        let me = key.public().to_peer_id();
        let unsigned = PeerRecord { peer_id: "node".into(), model_id: "m".into(), host: "1.2.3.4".into(), port: 4001, libp2p_peer_id: me.to_base58(), updated_unix_ms: 0, ..Default::default() };
        let record = dht::sign_peer_record(&unsigned, &key).unwrap();
        let advert = serde_json::json!({"type": PROVIDER_ANNOUNCE_TYPE, "record": record});
        let mut state = LoopState::new();
        assert!(handle_provider_pex(&mut state, &advert, Some(me)));
        let k = known_peer_key(&record);
        state.known_received.insert(k.clone(), 1);
        assert!(handle_provider_pex(&mut state, &advert, Some(me)));
        assert!(state.known_received[&k] > 1, "renewed on receipt, as before #9");
    }

    #[test]
    fn gossip_is_forwarded_only_while_every_node_still_remembers_it() {
        let now = 10_000_000_000u64;
        let announce = |ms: u64| serde_json::json!({"type": PROVIDER_ANNOUNCE_TYPE, "record": {"updated_unix_ms": ms}});
        assert!(gossip_forwardable(&announce(now), now));
        assert!(gossip_forwardable(&announce(now - GOSSIP_FORWARD_MAX_AGE_MS), now));
        assert!(!gossip_forwardable(&announce(now - GOSSIP_FORWARD_MAX_AGE_MS - 1), now), "older than the window allows");
        assert!(gossip_forwardable(&announce(0), now), "an undated record (an older node's) as before");
        let notice = |secs: u64| serde_json::json!({"type": "PEER_DEPARTED", "timestamp": secs});
        assert!(gossip_forwardable(&notice(now / 1000 - 60), now));
        assert!(!gossip_forwardable(&notice(now / 1000 - 600), now));
        assert!(gossip_forwardable(&serde_json::json!({"type": "PEER_DEPARTED"}), now), "no time: as before");
        // The window fits inside the duplicate cache's, with room for clocks that disagree.
        assert!(GOSSIP_FORWARD_MAX_AGE_MS + 60_000 <= crate::swarm::GOSSIP_DUPLICATE_CACHE.as_millis() as u64);
    }

    #[test]
    fn the_record_already_held_is_accepted_again_without_a_second_check_and_only_from_its_author() {
        let key = libp2p::identity::Keypair::generate_ed25519();
        let me = key.public().to_peer_id();
        let unsigned = PeerRecord {
            peer_id: "node".into(),
            model_id: "qwen2.5-7b".into(),
            host: "1.2.3.4".into(),
            port: 4001,
            libp2p_peer_id: me.to_base58(),
            updated_unix_ms: now_unix_ms(),
            ..Default::default()
        };
        let record = dht::sign_peer_record(&unsigned, &key).unwrap();
        let advert = serde_json::json!({"type": PROVIDER_ANNOUNCE_TYPE, "record": record});
        let mut state = LoopState::new();
        assert!(handle_provider_pex(&mut state, &advert, Some(me)), "checked and installed");
        assert_eq!(state.known_peers.get(&known_peer_key(&record)), Some(&record));
        // The same record again: accepted (and so forwarded) as before.
        assert!(handle_provider_pex(&mut state, &advert, Some(me)));
        // From anyone else it is still refused.
        assert!(!handle_provider_pex(&mut state, &advert, Some(PeerId::random())));
        // A changed copy is checked, and fails (its signature no longer matches).
        let mut forged = record.clone();
        forged.host = "6.6.6.6".into();
        let forged = serde_json::json!({"type": PROVIDER_ANNOUNCE_TYPE, "record": forged});
        assert!(!handle_provider_pex(&mut state, &forged, Some(me)));
        assert_eq!(state.known_peers.get(&known_peer_key(&record)), Some(&record), "the held one stays");
    }

    #[test]
    fn the_first_reservation_and_gossip_peer_re_announce_once_and_only_after_an_announce() {
        let mut seen = false;
        assert!(!first_time(&mut seen, false), "nothing announced yet: nothing to repeat");
        assert!(!first_time(&mut seen, true), "only the first time counts");
        let mut seen = false;
        assert!(first_time(&mut seen, true));
        assert!(!first_time(&mut seen, true));
        // The bump is what makes the provider re-announce.
        let state = LoopState::new();
        let before = state.net_generation.load(std::sync::atomic::Ordering::Relaxed);
        bump_net_generation(&state);
        assert_eq!(state.net_generation.load(std::sync::atomic::Ordering::Relaxed), before + 1);
    }

    #[test]
    fn a_server_whose_last_direct_address_expires_goes_back_to_client() {
        let mut state = LoopState::new();
        let upnp: Multiaddr = "/ip4/49.207.231.148/tcp/4001".parse().unwrap();
        let circuit: Multiaddr = "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit".parse().unwrap();
        state.kad_server_mode = true;
        state.external_addrs = vec![upnp.clone(), circuit.clone()];
        assert!(!should_demote_to_client(&state), "still reachable at the mapping");
        state.external_addrs.retain(|a| a != &upnp); // the lease lapsed
        assert!(should_demote_to_client(&state), "only a circuit left");
        state.kad_server_mode = false;
        assert!(!should_demote_to_client(&state), "a client stays one");
    }

    #[test]
    fn only_losing_the_network_counts_as_an_address_loss() {
        let ma = |s: &str| -> Multiaddr { s.parse().unwrap() };
        let wifi4 = ma("/ip4/192.168.0.4/tcp/4993");
        let tmp6 = ma("/ip6/2406:7400:56:9e97:f1ce:9282:52ec:c9f0/udp/4993/quic-v1");
        let other6 = ma("/ip6/2406:7400:56:9e97:3abd:5d47:eb85:335f/tcp/4993");
        let docker = ma("/ip4/172.17.0.1/tcp/4993");
        let virbr = ma("/ip4/192.168.122.1/tcp/4993");
        let lo = ma("/ip4/127.0.0.1/tcp/4993");
        // Leaving the Wi-Fi: only virtual and loopback addresses are left.
        assert!(addr_loss_leaves_network(&wifi4, &[docker.clone(), virbr.clone(), lo.clone(), other6.clone()]));
        // Ethernet still up next to it: not a move.
        assert!(!addr_loss_leaves_network(&wifi4, &[ma("/ip4/10.0.0.7/tcp/4993")]));
        // The Wi-Fi's own other listen addresses (its QUIC one) don't keep it.
        assert!(addr_loss_leaves_network(&wifi4, &[ma("/ip4/192.168.0.4/udp/4993/quic-v1")]));
        // A container bridge or a VM network going: never.
        assert!(!addr_loss_leaves_network(&docker, &[]));
        assert!(!addr_loss_leaves_network(&virbr, &[]));
        assert!(!addr_loss_leaves_network(&ma("/ip4/100.101.1.2/tcp/1"), &[]));
        // An IPv6 temporary address rotating (another one in its /64 stays): not a move; the last
        // one in the /64 going: a move. Link-local and loopback never count.
        assert!(!addr_loss_leaves_network(&tmp6, &[other6.clone(), wifi4.clone()]));
        assert!(addr_loss_leaves_network(&tmp6, &[ma("/ip6/2401:4900:c907:8361::4/tcp/4993")]));
        assert!(!addr_loss_leaves_network(&ma("/ip6/fe80::1/tcp/4993"), &[]));
        assert!(!addr_loss_leaves_network(&lo, &[]));
    }

    #[test]
    fn a_held_reservation_makes_one_ipv6_quic_dial_to_its_relay() {
        let me = PeerId::random();
        let circuits = eu_circuits(&me);
        let mut state = LoopState::new();
        state.ipv6_capable = true;
        let listener = libp2p::core::transport::ListenerId::next();
        state.relay_listeners.insert(listener, eu());
        let now = std::time::Instant::now();
        let want: Multiaddr = "/ip6/2a01:7e01::2000:84ff:fec5:4520/udp/4001/quic-v1/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT".parse().unwrap();
        // Only the relay's IPv6 QUIC circuit names an address to dial; v4 and TCP ones don't.
        for c in &circuits[..3] {
            assert_eq!(relay_v6_quic_dial(&state, listener, c, now), None, "{c}");
        }
        assert_eq!(relay_v6_quic_dial(&state, listener, &circuits[3], now), Some((eu(), want.clone())));
        // A non-global IPv6 relay address is never dialled.
        let ula: Multiaddr = format!("/ip6/fd00::1/udp/4001/quic-v1/p2p/{}/p2p-circuit/p2p/{me}", eu()).parse().unwrap();
        assert_eq!(relay_v6_quic_dial(&state, listener, &ula, now), None);
        // Not one of our reservation listeners (or a listener on another relay): nothing.
        assert_eq!(relay_v6_quic_dial(&state, libp2p::core::transport::ListenerId::next(), &circuits[3], now), None);
        state.relay_listeners.insert(listener, PeerId::random());
        assert_eq!(relay_v6_quic_dial(&state, listener, &circuits[3], now), None);
        state.relay_listeners.insert(listener, eu());
        // No working IPv6 here: nothing.
        state.ipv6_capable = false;
        assert_eq!(relay_v6_quic_dial(&state, listener, &circuits[3], now), None);
        state.ipv6_capable = true;
        // Dialled recently: not again until RELAY_V6_QUIC_REDIAL has passed.
        state.relay_v6_quic_dialled.insert(eu(), now);
        assert_eq!(relay_v6_quic_dial(&state, listener, &circuits[3], now + std::time::Duration::from_secs(60)), None);
        assert!(relay_v6_quic_dial(&state, listener, &circuits[3], now + RELAY_V6_QUIC_REDIAL).is_some());
        // Already connected to it over IPv6 QUIC: nothing.
        state.relay_v6_quic_conns.insert(libp2p::swarm::ConnectionId::new_unchecked(7), eu());
        assert_eq!(relay_v6_quic_dial(&state, listener, &circuits[3], now + RELAY_V6_QUIC_REDIAL), None);
        assert!(is_v6_quic(&"/ip6/2a01:7e01::1/udp/4001/quic-v1".parse().unwrap()));
        assert!(!is_v6_quic(&"/ip6/2a01:7e01::1/tcp/4001".parse().unwrap()));
        assert!(!is_v6_quic(&"/ip4/1.2.3.4/udp/4001/quic-v1".parse().unwrap()));
    }

    #[test]
    fn a_circuit_names_its_relay_and_the_circuit_to_reserve_at() {
        let me = PeerId::random();
        let circuits = eu_circuits(&me);
        assert!(circuits.iter().all(|c| circuit_relay(c) == Some(eu())));
        assert_eq!(circuit_relay(&"/ip4/1.2.3.4/tcp/4001/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT".parse().unwrap()), None);
        let relays = crate::relay::relay_addrs(&[], crate::relay::PublicRelays::Auto);
        // The configured address (v4 TCP, listed first) wins over whichever circuit closed.
        let want: Multiaddr = "/ip4/172.105.69.49/tcp/4001/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT/p2p-circuit".parse().unwrap();
        assert_eq!(reservation_circuit(&relays, eu(), Some(&circuits[3])), Some(want));
        // A relay we weren't configured with: the circuit as seen, without our own id.
        let want: Multiaddr = "/ip6/2a01:7e01::2000:84ff:fec5:4520/udp/4001/quic-v1/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT/p2p-circuit".parse().unwrap();
        assert_eq!(reservation_circuit(&[], eu(), Some(&circuits[3])), Some(want));
    }

    #[test]
    fn a_lost_reservation_is_retried_once_per_relay_not_once_per_circuit() {
        let me = PeerId::random();
        let mut state = LoopState::new();
        state.relays = crate::relay::relay_addrs(&[], crate::relay::PublicRelays::Auto);
        let l = libp2p::core::transport::ListenerId::next();
        state.relay_listeners.insert(l, eu());
        // The reservation's listener closes reporting all four circuits (what a roam does).
        let again = relays_to_re_reserve(&mut state, l, &eu_circuits(&me));
        assert_eq!(again.len(), 1, "{again:?}");
        assert_eq!(again[0].0, eu());
        assert!(state.relay_listeners.is_empty());
        for (relay, circuit) in again {
            schedule_reservation_retry(&mut state, relay, circuit.clone());
            schedule_reservation_retry(&mut state, relay, circuit);
        }
        assert_eq!(state.relay_reservation_retries.len(), 1, "one entry per relay");
        assert_eq!(state.relay_reservation_retries[&eu()].0, 2, "a second failure lengthens the wait");
        // A circuit through the relay coming up ends the retries.
        clear_reservation_retry(&mut state, &eu_circuits(&me)[1]);
        assert!(state.relay_reservation_retries.is_empty());
    }

    #[test]
    fn a_relay_connection_is_closed_for_failed_pings_only_when_it_runs_over_a_network_we_left() {
        let mut state = LoopState::new();
        state.relays = crate::relay::relay_addrs(&[], crate::relay::PublicRelays::Auto);
        let t0 = std::time::Instant::now();
        let at = |s: u64| t0 + std::time::Duration::from_secs(s);
        let timeout = libp2p::ping::Failure::Timeout;
        let conn = libp2p::swarm::ConnectionId::new_unchecked;
        let ip = |s: &str| -> std::net::IpAddr { s.parse().unwrap() };
        let ours: std::collections::HashSet<std::net::IpAddr> = [ip("192.168.0.4"), ip("2406:7400:56:9e97::4")].into();
        // Behind NAT on ACT, a busy connection failing pings, nothing changed: never closed.
        state.relay_conns.insert(conn(1), RelayConn { since: at(0), seen_as: Some((ip("49.207.231.148"), false)) });
        for _ in 0..6 {
            assert!(!relay_connection_dead(&mut state, eu(), conn(1), &timeout, &ours));
        }
        // A newer relay connection from the same /16 (a CGNAT pool handing out another address): still not.
        state.relay_conns.insert(conn(2), RelayConn { since: at(10), seen_as: Some((ip("49.207.9.9"), false)) });
        assert!(!relay_connection_dead(&mut state, eu(), conn(1), &timeout, &ours));
        // After a move to Airtel a newer connection sees us from another /16: the old one goes, at
        // the 2nd reported failure; the new one is never judged by the older one.
        state.relay_ping_failures.clear();
        state.relay_conns.insert(conn(3), RelayConn { since: at(100), seen_as: Some((ip("106.192.224.140"), false)) });
        assert!(!relay_connection_dead(&mut state, eu(), conn(1), &timeout, &ours));
        assert!(relay_connection_dead(&mut state, eu(), conn(1), &timeout, &ours));
        for _ in 0..3 {
            assert!(!relay_connection_dead(&mut state, eu(), conn(3), &timeout, &ours), "the newest isn't from an old network");
        }
        // No NAT: the relay saw one of our own addresses, and it's gone.
        state.relay_conns.insert(conn(4), RelayConn { since: at(0), seen_as: Some((ip("2406:7400:56:9e97::4"), true)) });
        assert!(!relay_connection_dead(&mut state, eu(), conn(4), &timeout, &ours));
        assert!(!relay_connection_dead(&mut state, eu(), conn(4), &timeout, &ours), "still ours: not closed");
        let left: std::collections::HashSet<std::net::IpAddr> = [ip("10.120.222.114"), ip("2401:4900:c907::4")].into();
        assert!(relay_connection_dead(&mut state, eu(), conn(4), &timeout, &left));
        // A connection whose relay hasn't said where it sees us, a peer without ping, a non-relay: never.
        state.relay_conns.insert(conn(5), RelayConn { since: at(0), seen_as: None });
        for _ in 0..3 {
            assert!(!relay_connection_dead(&mut state, eu(), conn(5), &timeout, &left));
            assert!(!relay_connection_dead(&mut state, eu(), conn(1), &libp2p::ping::Failure::Unsupported, &ours));
            assert!(!relay_connection_dead(&mut state, PeerId::random(), conn(1), &timeout, &ours));
        }
    }

    #[test]
    fn an_idle_relayed_connection_is_closed_for_failed_pings_a_busy_one_never() {
        let mut state = LoopState::new();
        let p = PeerId::random();
        let conn = libp2p::swarm::ConnectionId::new_unchecked;
        let timeout = libp2p::ping::Failure::Timeout;
        state.relayed_conns.insert(conn(1), p);
        // Busy (a transfer under way): never, however many pings fail.
        for _ in 0..5 {
            assert!(!relayed_connection_dead(&mut state, p, conn(1), &timeout, true));
        }
        // Idle once it's done: closed at the next failure, the count already past the limit.
        assert!(relayed_connection_dead(&mut state, p, conn(1), &timeout, false));
        // A fresh idle one: the 2nd reported failure closes it; a success in between resets.
        assert!(!relayed_connection_dead(&mut state, p, conn(1), &timeout, false));
        state.relayed_ping_failures.remove(&conn(1)); // what a successful ping does
        assert!(!relayed_connection_dead(&mut state, p, conn(1), &timeout, false));
        assert!(relayed_connection_dead(&mut state, p, conn(1), &timeout, false));
        // A direct connection, another peer's connection, a peer without ping: never.
        for _ in 0..3 {
            assert!(!relayed_connection_dead(&mut state, p, conn(2), &timeout, false));
            assert!(!relayed_connection_dead(&mut state, PeerId::random(), conn(1), &timeout, false));
            assert!(!relayed_connection_dead(&mut state, p, conn(1), &libp2p::ping::Failure::Unsupported, false));
        }
        // What counts as busy.
        let never = |_: &request_response::OutboundRequestId| false;
        assert!(!peer_busy(&state, p, never));
        // (libp2p request ids can't be made in a test; a response being written and an inbound
        // request are counted the same way as this.)
        let (tx, _rx) = oneshot::channel();
        state.pending_relay_forwards.push((p, vec![], tx, std::time::Instant::now()));
        assert!(peer_busy(&state, p, never), "a forward queued for it");
        assert!(!peer_busy(&state, PeerId::random(), never));
    }

    #[test]
    fn a_failed_dial_names_only_the_addresses_it_tried() {
        let a: Multiaddr = "/ip4/203.0.113.7/tcp/4001".parse().unwrap();
        let b: Multiaddr = "/ip6/2001:db8::7/udp/4001/quic-v1".parse().unwrap();
        let err = libp2p::swarm::DialError::Transport(vec![
            (a.clone(), libp2p::TransportError::MultiaddrNotSupported(a.clone())),
            (b.clone(), libp2p::TransportError::MultiaddrNotSupported(b.clone())),
        ]);
        assert_eq!(dial_error_addrs(&err), vec![a, b]);
        assert!(dial_error_addrs(&libp2p::swarm::DialError::NoAddresses).is_empty(), "no address tried: none forgotten");
        assert!(dial_error_addrs(&libp2p::swarm::DialError::Aborted).is_empty());
    }

    #[test]
    fn relays_are_reserved_on_once_each_at_their_first_address() {
        let mut state = LoopState::new();
        state.relays = crate::relay::relay_addrs(&[], crate::relay::PublicRelays::Auto);
        let want: std::collections::HashSet<PeerId> = crate::relay::relay_peer_ids(&state.relays).into_iter().collect();
        assert_eq!(want.len(), 4);
        let picks = relays_to_reserve(&state);
        assert_eq!(picks.len(), want.len(), "one per relay, though each is listed twice: {picks:?}");
        for (relay, circuit) in &picks {
            let first = state.relays.iter().find(|a| a.iter().any(|p| p == libp2p::multiaddr::Protocol::P2p(*relay))).unwrap();
            assert_eq!(circuit, &first.clone().with(libp2p::multiaddr::Protocol::P2pCircuit));
        }
        // A relay with a listener of ours (held or still being requested) isn't asked again.
        state.relay_listeners.insert(libp2p::core::transport::ListenerId::next(), eu());
        let picks = relays_to_reserve(&state);
        assert_eq!(picks.len(), want.len() - 1);
        assert!(picks.iter().all(|(r, _)| *r != eu()));
    }

    #[test]
    fn a_due_retry_waits_while_a_listener_is_on_the_relay() {
        let mut state = LoopState::new();
        state.relays = crate::relay::relay_addrs(&[], crate::relay::PublicRelays::Auto);
        let circuit = reservation_circuit(&state.relays, eu(), None).unwrap();
        schedule_reservation_retry(&mut state, eu(), circuit.clone());
        let later = tokio::time::Instant::now() + std::time::Duration::from_secs(600);
        assert!(due_reservation_retries(&state, tokio::time::Instant::now()).is_empty(), "not yet");
        assert_eq!(due_reservation_retries(&state, later), vec![(eu(), circuit)]);
        let l = libp2p::core::transport::ListenerId::next();
        state.relay_listeners.insert(l, eu());
        assert!(due_reservation_retries(&state, later).is_empty(), "a request is out");
        state.relay_listeners.remove(&l);
        assert_eq!(due_reservation_retries(&state, later).len(), 1, "and it failed: retry");
    }

    #[test]
    fn an_address_two_peers_claim_is_kept_for_neither_new_claimant() {
        let mut state = LoopState::new();
        let (a, b) = (PeerId::random(), PeerId::random());
        let shared: Multiaddr = "/ip6/2406:7400:56:9e97::1/udp/4001/quic-v1".parse().unwrap();
        let own: Multiaddr = "/ip6/2406:7400:56:9e97::2/udp/4001/quic-v1".parse().unwrap();
        assert!(store_claimed_v6_addrs(&mut state, a, &[shared.clone()]));
        assert!(!store_claimed_v6_addrs(&mut state, b, &[shared.clone()]), "already a's");
        assert!(store_claimed_v6_addrs(&mut state, b, &[shared.clone(), own.clone()]));
        assert_eq!(state.peer_quic_addrs[&b], vec![own]);
        assert!(store_claimed_v6_addrs(&mut state, a, &[shared.clone()]), "a re-identifying keeps its own");
    }

    #[test]
    fn hole_punch_rounds_are_capped_node_wide_and_need_ipv6() {
        let mut state = LoopState::new();
        state.ipv6_capable = true;
        let addr: Multiaddr = "/ip6/2406:7400:56:9e97::1/udp/4001/quic-v1".parse().unwrap();
        let mut rounds = 0;
        for i in 0..(MAX_PUNCH_ROUNDS_PER_MINUTE + 5) {
            let p = PeerId::random();
            let a: Multiaddr = format!("/ip6/2406:7400:56:9e97::{:x}/udp/4001/quic-v1", i + 10).parse().unwrap();
            state.peer_quic_addrs.insert(p, vec![a]);
            rounds += (!v6_holepunch_targets(&mut state, p).is_empty()) as u32;
        }
        assert_eq!(rounds, MAX_PUNCH_ROUNDS_PER_MINUTE);
        // Without working IPv6, nothing at all.
        let mut state = LoopState::new();
        state.ipv6_capable = false;
        let p = PeerId::random();
        state.peer_quic_addrs.insert(p, vec![addr]);
        assert!(v6_holepunch_targets(&mut state, p).is_empty());
    }

    #[test]
    fn a_duplicate_listener_closing_leaves_the_live_one_alone() {
        let me = PeerId::random();
        let mut state = LoopState::new();
        state.relays = crate::relay::relay_addrs(&[], crate::relay::PublicRelays::Auto);
        let (live, dup) = (libp2p::core::transport::ListenerId::next(), libp2p::core::transport::ListenerId::next());
        state.relay_listeners.insert(live, eu());
        state.relay_listeners.insert(dup, eu());
        // libp2p keeps one reservation per relay and ends the other request, before any address.
        assert!(relays_to_re_reserve(&mut state, dup, &[]).is_empty(), "the relay still has a listener");
        assert!(relay_has_listener(&state, eu()));
        // The live one closing later (its connection went) does need a retry.
        assert_eq!(relays_to_re_reserve(&mut state, live, &eu_circuits(&me)).len(), 1);
        // And a request refused before reporting an address is still retried, known by its id.
        let refused = libp2p::core::transport::ListenerId::next();
        state.relay_listeners.insert(refused, eu());
        assert_eq!(relays_to_re_reserve(&mut state, refused, &[]).len(), 1);
        // A non-relay listener closing is no business of this.
        assert!(relays_to_re_reserve(&mut state, libp2p::core::transport::ListenerId::next(), &["/ip4/0.0.0.0/tcp/4001".parse().unwrap()]).is_empty());
    }

    #[test]
    fn a_v6_punch_dials_stored_addresses_until_a_v6_direct_or_the_cap() {
        let mut state = LoopState::new();
        state.ipv6_capable = true;
        let peer = PeerId::random();
        assert!(v6_holepunch_targets(&mut state, peer).is_empty(), "no addresses stored");
        let addr: Multiaddr = "/ip6/2406:7400:56:9e97::1/udp/4001/quic-v1".parse().unwrap();
        state.peer_quic_addrs.insert(peer, vec![addr.clone()]);
        for _ in 0..MAX_QUIC_HOLEPUNCH_ATTEMPTS {
            assert_eq!(v6_holepunch_targets(&mut state, peer), vec![addr.clone()]);
        }
        assert!(v6_holepunch_targets(&mut state, peer).is_empty(), "backed off after the cap");
        // A v4 QUIC direct isn't enough: with the count cleared, it still dials.
        state.quic_holepunch_attempts.clear();
        state.peer_connections.insert(peer, PeerConnectionInfo { quic_direct_v4: 1, ..Default::default() });
        assert_eq!(v6_holepunch_targets(&mut state, peer), vec![addr.clone()], "a v4 direct doesn't stop it");
        // A v6 one does, and clears the count.
        state.peer_connections.insert(peer, PeerConnectionInfo { quic_direct_v6: 1, ..Default::default() });
        assert!(v6_holepunch_targets(&mut state, peer).is_empty());
        assert!(!state.quic_holepunch_attempts.contains_key(&peer), "count cleared");
    }

    #[test]
    fn a_v6_punch_never_targets_a_public_relay() {
        let mut state = LoopState::new();
        state.ipv6_capable = true;
        let peer = PeerId::random();
        let ip = crate::relay::BOOTSTRAP_RELAY_IPS.iter().find(|ip| ip.contains(':')).expect("a v6 relay");
        state.peer_quic_addrs.insert(peer, vec![format!("/ip6/{ip}/udp/4001/quic-v1").parse().unwrap()]);
        assert!(v6_holepunch_targets(&mut state, peer).is_empty());
    }

    #[test]
    fn a_burst_of_requests_punches_once_per_debounce() {
        let mut state = LoopState::new();
        state.ipv6_capable = true;
        let peer = PeerId::random();
        let t0 = std::time::Instant::now();
        assert!(debounced_v6_targets(&mut state, peer, t0).is_empty(), "nothing stored: nothing to punch");
        state.peer_quic_addrs.insert(peer, vec!["/ip6/2401:4900:c907::1/udp/4941/quic-v1".parse().unwrap()]);
        assert_eq!(debounced_v6_targets(&mut state, peer, t0).len(), 1);
        assert!(debounced_v6_targets(&mut state, peer, t0 + std::time::Duration::from_secs(1)).is_empty());
        assert!(debounced_v6_targets(&mut state, peer, t0 + REPUNCH_DEBOUNCE - std::time::Duration::from_millis(1)).is_empty());
        assert_eq!(debounced_v6_targets(&mut state, peer, t0 + REPUNCH_DEBOUNCE).len(), 1);
        state.peer_connections.insert(peer, PeerConnectionInfo { quic_direct_v6: 1, ..Default::default() });
        assert!(debounced_v6_targets(&mut state, peer, t0 + REPUNCH_DEBOUNCE * 3).is_empty(), "already have v6 direct");
    }

    #[test]
    fn a_refused_punch_does_not_use_up_the_wait() {
        let mut state = LoopState::new();
        state.ipv6_capable = false;
        let peer = PeerId::random();
        state.peer_quic_addrs.insert(peer, vec!["/ip6/2401:4900:c907::1/udp/4941/quic-v1".parse().unwrap()]);
        let t0 = std::time::Instant::now();
        assert!(debounced_v6_targets(&mut state, peer, t0).is_empty(), "no IPv6: refused");
        assert!(!state.last_repunch.contains_key(&peer));
        state.ipv6_capable = true;
        assert_eq!(debounced_v6_targets(&mut state, peer, t0 + std::time::Duration::from_secs(1)).len(), 1, "so a punch can go out at once");
    }

    #[test]
    fn identify_addresses_cannot_point_other_nodes_at_their_own_network() {
        let a = |s: &str| s.parse::<Multiaddr>().unwrap();
        // Public addresses always pass; a relay circuit too.
        assert!(identify_addr_admissible(&a("/ip4/203.0.113.7/tcp/4001"), false));
        assert!(identify_addr_admissible(&a("/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit"), false));
        assert!(identify_addr_admissible(&a("/dns4/example.org/tcp/4001"), false));
        // Loopback / private / CGNAT / link-local / ULA: only from a peer reached that way.
        for local in ["/ip4/127.0.0.1/tcp/4001", "/ip4/192.168.1.10/tcp/22", "/ip4/10.0.0.1/udp/4001/quic-v1",
                      "/ip4/100.64.1.1/tcp/1", "/ip4/169.254.1.1/tcp/1", "/ip6/::1/tcp/1", "/ip6/fd00::1/tcp/1", "/ip6/fe80::1/tcp/1"] {
            assert!(!identify_addr_admissible(&a(local), false), "{local} from a remote peer");
            assert!(identify_addr_admissible(&a(local), true), "{local} from a LAN peer");
        }
        // Never: unspecified, or a local-only transport.
        assert!(!identify_addr_admissible(&a("/ip4/0.0.0.0/tcp/4001"), true));
        assert!(!identify_addr_admissible(&a("/unix/tmp%2Fsock"), true));
        // Connections: a relayed one is never "local link", whatever the relay's address.
        assert!(is_local_link("/ip4/192.168.1.2/tcp/4001", Some("192.168.1.2")));
        assert!(!is_local_link("/ip4/192.168.1.2/tcp/4001/p2p/x/p2p-circuit", Some("192.168.1.2")));
        assert!(!is_local_link("/ip4/8.8.8.8/tcp/4001", Some("8.8.8.8")));
    }


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
    fn discover_answers_from_known_providers_unless_they_all_just_failed() {
        use DiscoverMode::*;
        assert_eq!(discover_mode(true, true, false), CacheHit);
        assert_eq!(discover_mode(true, false, false), AnswerThenRefresh, "no wait on the DHT when providers are known");
        assert_eq!(discover_mode(true, true, true), WaitForLookup, "the retry after every candidate failed looks in the DHT");
        assert_eq!(discover_mode(true, false, true), WaitForLookup);
        assert_eq!(discover_mode(false, false, false), WaitForLookup);
        assert_eq!(discover_mode(false, true, false), WaitForLookup);
    }

    #[test]
    fn a_retry_waits_for_new_information_or_its_deadline() {
        use DiscoverKind::*;
        let now = std::time::Instant::now();
        let soon = Some(now + RETRY_SETTLE);
        let late = Some(now - std::time::Duration::from_millis(1));
        assert!(waiter_settled(Reply, false, 0, 1, 0, 0, None, now), "a caller with nothing known takes the first providers");
        assert!(!waiter_settled(Reply, false, 0, 0, 0, 0, None, now));
        assert!(!waiter_settled(Retry, false, 0, 2, 3, 3, soon, now), "the providers that just failed aren't an answer");
        assert!(waiter_settled(Retry, false, 0, 2, 4, 3, soon, now), "a new provider, or a new address, is");
        assert!(!waiter_settled(Retry, false, 1, 2, 4, 3, soon, now), "not with a record fetch outstanding");
        assert!(waiter_settled(Retry, false, 1, 2, 3, 3, late, now), "the deadline answers whatever is outstanding");
        for kind in [Reply, Retry] {
            assert!(waiter_settled(kind, true, 0, 0, 0, 0, None, now), "a finished lookup answers everyone");
        }
    }

    #[test]
    fn a_lan_join_republishes_soon_and_a_burst_only_once() {
        let t0 = std::time::Instant::now();
        let mut a = LanAnnounce::default();
        note_lan_subscriber(&mut a, t0);
        assert!(!lan_announce_due(&mut a, t0), "not before the subscription settles");
        assert!(lan_announce_due(&mut a, t0 + LAN_ANNOUNCE_DELAY));
        // More joins right after: one more re-publish, not before LAN_ANNOUNCE_EVERY.
        for ms in [100, 200, 300] {
            note_lan_subscriber(&mut a, t0 + LAN_ANNOUNCE_DELAY + std::time::Duration::from_millis(ms));
        }
        assert!(!lan_announce_due(&mut a, t0 + LAN_ANNOUNCE_DELAY + std::time::Duration::from_secs(5)));
        assert!(lan_announce_due(&mut a, t0 + LAN_ANNOUNCE_DELAY + LAN_ANNOUNCE_EVERY));
        assert!(!lan_announce_due(&mut a, t0 + LAN_ANNOUNCE_DELAY + LAN_ANNOUNCE_EVERY * 2), "nothing new since");
    }

    fn rec_at(peer: &str, ms: u64, relay: &str) -> PeerRecord {
        PeerRecord { libp2p_peer_id: peer.into(), model_id: "m".into(), updated_unix_ms: ms, relay_address: relay.into(), ..Default::default() }
    }

    #[test]
    fn the_newest_signed_record_wins_and_only_an_address_change_is_news() {
        let now = 1_000_000;
        let mut list = vec![rec_at("A", now - 10_000, "")];
        assert_eq!(merge_record(&mut list, rec_at("A", now - 20_000, "/old"), now), Merge::Kept, "older");
        assert_eq!(merge_record(&mut list, rec_at("A", now - 10_000, "/same-time"), now), Merge::Kept, "as old");
        assert_eq!(
            merge_record(&mut list, rec_at("A", now - 5_000, ""), now),
            Merge::Replaced { address_changed: false },
            "a republish with the same address"
        );
        assert_eq!(
            merge_record(&mut list, rec_at("A", now - 1_000, "/ip4/1.2.3.4/tcp/4001/p2p/R/p2p-circuit"), now),
            Merge::Replaced { address_changed: true },
            "reachable through a relay now"
        );
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].relay_address, "/ip4/1.2.3.4/tcp/4001/p2p/R/p2p-circuit");
        assert_eq!(merge_record(&mut list, rec_at("A", now + MAX_RECORD_SKEW_MS + 1, "/future"), now), Merge::Kept, "dated too far ahead");
        assert_eq!(merge_record(&mut list, rec_at("B", now, ""), now), Merge::Added);
        assert_eq!(list.len(), 2);
    }

    /// GPU test 2026-10-29: records carried no time, and the reaper kept an unconnected provider
    /// only by that time, so a consumer lost every provider it had only heard of over gossip a
    /// minute later. It goes by when the record arrived now.
    #[test]
    fn reaper_keeps_a_gossip_learned_provider_for_its_ttl() {
        let now = 10_000_000;
        assert!(keep_known_peer(false, Some(now - 60_000), None, now), "heard a minute ago, never connected");
        assert!(keep_known_peer(false, Some(now - KNOWN_PEER_TTL_MS + 1), None, now));
        assert!(!keep_known_peer(false, Some(now - KNOWN_PEER_TTL_MS), None, now), "not heard of for the whole TTL");
        assert!(!keep_known_peer(false, None, None, now));
        assert!(keep_known_peer(true, None, None, now), "connected");
    }

    /// #30 (18 Nov): a cut connection to a provider deleted its records at once, so a gateway that
    /// reconnected within seconds still answered "no provider" until the next announcement.
    #[test]
    fn a_provider_outlives_a_lost_connection_for_the_grace_only() {
        let now = 10_000_000;
        let heard = Some(now - 50_000);
        // Lost moments ago: kept, connected again or not.
        assert!(keep_known_peer(false, heard, Some(now - 5_000), now));
        assert!(keep_known_peer(true, heard, Some(now - 5_000), now));
        // Lost for the whole grace with no record since: gone, though the record is still fresh.
        assert!(!keep_known_peer(false, heard, Some(now - LOST_PEER_GRACE_MS), now));
        // ...unless we're connected again, or heard from it after losing the connection.
        assert!(keep_known_peer(true, heard, Some(now - LOST_PEER_GRACE_MS), now));
        assert!(keep_known_peer(false, Some(now - 1_000), Some(now - LOST_PEER_GRACE_MS), now));
    }

    /// Review (18 Nov): failures that can't have reached the peer's application are labelled so,
    /// and a peer refusing a lazily negotiated protocol is told apart from a dead path.
    #[test]
    fn never_sent_failures_are_labelled_and_a_refused_protocol_is_not_a_dead_path() {
        let refused = std::io::Error::from(libp2p::core::upgrade::NegotiationError::Failed);
        assert!(is_negotiation_refusal(&refused));
        let limit = std::io::Error::new(std::io::ErrorKind::Other, "max sub-streams reached");
        for f in [
            request_response::OutboundFailure::DialFailure,
            request_response::OutboundFailure::UnsupportedProtocols,
            request_response::OutboundFailure::Io(refused),
            request_response::OutboundFailure::Io(limit),
        ] {
            assert!(proxy_failure_text(&f).ends_with(NEVER_SENT), "{f:?}");
        }
        for f in [request_response::OutboundFailure::Timeout, request_response::OutboundFailure::ConnectionClosed] {
            assert!(!proxy_failure_text(&f).contains(NEVER_SENT), "{f:?}");
        }
    }

    /// Review (18 Nov): the outbound cap leaves room for the peer's own requests on the shared
    /// per-connection limit.
    #[test]
    fn the_outbound_cap_leaves_room_for_the_peers_requests() {
        let state = LoopState::new();
        let peer = PeerId::random();
        assert_eq!(proxy_slots(&state, peer), MAX_PROXY_INFLIGHT_PER_PEER);
    }

    /// #31 (18 Nov): libp2p's "max sub-streams reached" is the connection's own limit, not a dead
    /// path; counting it closed a busy gateway's connections to a healthy provider.
    #[test]
    fn the_connections_own_request_limit_is_not_a_dead_path() {
        let limit = std::io::Error::new(std::io::ErrorKind::Other, "max sub-streams reached");
        assert!(is_local_stream_limit(&limit));
        assert!(!is_local_stream_limit(&std::io::Error::new(std::io::ErrorKind::BrokenPipe, "broken pipe")));
        assert!(MAX_PROXY_INFLIGHT_PER_PEER < crate::proxy::MAX_CONCURRENT_STREAMS);
    }

    /// Review 2026-10-30: a replayed or re-forwarded copy of a provider's last record must neither
    /// renew it (a dead provider would stay listed for ever) nor replace a newer one (a roamed
    /// provider's old address would come back).
    #[test]
    fn provider_gone_needs_a_quorum_of_distinct_bootstraps() {
        let mut pending = HashMap::new();
        let (p, b1, b2) = (PeerId::random(), PeerId::random(), PeerId::random());
        let now = 1_000_000_000;
        assert_eq!(note_provider_gone(&mut pending, p, b1, 50, 2, now, now), None);
        assert_eq!(note_provider_gone(&mut pending, p, b1, 60, 2, now + 1, now + 1), None, "the same bootstrap twice is one report");
        assert_eq!(note_provider_gone(&mut pending, p, b2, 40, 2, now + 2, now + 2), Some((60, now)), "newest record time named");
        assert!(pending.is_empty());
        // Reports older than the window don't count towards a later quorum.
        assert_eq!(note_provider_gone(&mut pending, p, b1, 1, 2, now, now), None);
        assert_eq!(note_provider_gone(&mut pending, p, b2, 1, 2, now + PROVIDER_GONE_MAX_AGE_SECS * 1000 + 1, now + PROVIDER_GONE_MAX_AGE_SECS * 1000 + 1), None);
        // With one bootstrap configured, its report is enough.
        assert_eq!(note_provider_gone(&mut HashMap::new(), p, b1, 7, 1, now, now), Some((7, now)));
        // The cutoff is the earliest report's own time, not when it arrived (a replay arrives late).
        let mut late = HashMap::new();
        assert_eq!(note_provider_gone(&mut late, p, b1, 3, 2, now + 50_000, now + 50_000), None);
        assert_eq!(note_provider_gone(&mut late, p, b2, 3, 2, now + 60_000, now), Some((3, now)));
    }

    #[test]
    fn a_gone_provider_is_not_cached_again_from_a_stale_copy() {
        let rec = |t: u64| PeerRecord { libp2p_peer_id: "P".into(), model_id: "m".into(), updated_unix_ms: t, ..Default::default() };
        let mut gone = HashMap::new();
        gone.insert("P".to_string(), (150u64, 1_000u64));
        assert!(is_gone_record(&gone, &rec(150), 2_000), "the record the bootstraps saw");
        assert!(!is_gone_record(&gone, &rec(0), 2_000), "undated: can't tell, never refused");
        assert!(!is_gone_record(&gone, &rec(151), 2_000), "announced since: alive");
        assert!(!is_gone_record(&gone, &rec(100), 1_000 + GONE_MEMORY_MS), "forgotten after the memory");
        assert!(!is_gone_record(&gone, &PeerRecord { libp2p_peer_id: "Q".into(), ..rec(1) }, 2_000));
    }

    #[test]
    fn a_gone_provider_loses_only_records_no_newer_than_the_bootstraps_saw() {
        let rec = |lp: &str, m: &str, t: u64| PeerRecord { libp2p_peer_id: lp.into(), model_id: m.into(), updated_unix_ms: t, ..Default::default() };
        let mut known = HashMap::new();
        let mut received = HashMap::new();
        for r in [rec("P", "a", 100), rec("P", "b", 200), rec("Q", "a", 100), rec("P", "legacy", 0), rec("P", "legacy2", 0)] {
            let k = known_peer_key(&r);
            received.insert(k.clone(), if r.model_id == "legacy2" { 5_000 } else { 1_000 });
            known.insert(k, r);
        }
        // Bootstraps held P's records up to time 150; reports began at 2_000 (ms).
        let gone = evict_gone_provider(&mut known, &received, "P", 150, 2_000);
        assert_eq!(gone, vec!["a".to_string(), "legacy".to_string()]);
        let left: Vec<_> = known.values().map(|r| (r.libp2p_peer_id.as_str(), r.model_id.as_str())).collect();
        assert!(left.contains(&("P", "b")), "announced after what the bootstraps saw: alive");
        assert!(left.contains(&("P", "legacy2")), "an undated record received after the reports began");
        assert!(left.contains(&("Q", "a")), "another provider of the same model");
    }

    #[test]
    fn relayed_dial_backoff_doubles_to_a_cap_and_respects_relay_limits() {
        use std::time::Duration;
        let s = |d: u64| Duration::from_secs(d);
        assert_eq!(next_relay_dial_backoff(None, false), s(2));
        assert_eq!(next_relay_dial_backoff(Some(s(2)), false), s(4));
        assert_eq!(next_relay_dial_backoff(Some(s(8)), false), s(16));
        assert_eq!(next_relay_dial_backoff(Some(s(16)), false), s(16), "capped");
        assert_eq!(next_relay_dial_backoff(None, true), s(20), "the relays said no: let their budget refill");
        assert_eq!(next_relay_dial_backoff(Some(s(16)), true), s(20));
    }

    #[test]
    fn a_pinned_peer_is_dialled_on_its_pinned_addresses_alone() {
        // Exactly its pinned addresses: no relay circuit, nothing the behaviours add.
        let p = PeerId::random();
        let lo: Multiaddr = "/ip4/127.0.0.1/tcp/4111".parse().unwrap();
        let opts = pinned_dial_opts(p, vec![lo.clone()], libp2p::swarm::dial_opts::PeerCondition::Disconnected);
        assert_eq!(opts.get_peer_id(), Some(p));
        // (DialOpts keeps its address list and the extend flag private: read them off its Debug.)
        let dbg = format!("{opts:?}");
        assert!(dbg.contains(&format!("addresses: [{lo}]")), "{dbg}");
        assert!(dbg.contains("extend_addresses_through_behaviour: false"), "{dbg}");
    }

    #[test]
    fn only_a_newer_record_renews_or_replaces() {
        let (mut known, mut received) = (HashMap::new(), HashMap::new());
        let now = 10_000_000_000;
        let rec = |t: u64, host: &str| PeerRecord {
            libp2p_peer_id: "p".into(), model_id: "m".into(), host: host.into(), updated_unix_ms: t, ..Default::default()
        };
        assert_eq!(install_known_peer_in(&mut known, &mut received, "me", &rec(now - 10_000, "new"), now - 10_000), Some(Install::New));
        let key = known_peer_key(&rec(0, ""));
        // The same record again later: kept, not renewed.
        assert_eq!(install_known_peer_in(&mut known, &mut received, "me", &rec(now - 10_000, "new"), now), Some(Install::Duplicate));
        assert_eq!(received[&key], now - 10_000);
        // An older one: refused, the newer address stays.
        assert_eq!(install_known_peer_in(&mut known, &mut received, "me", &rec(now - 20_000, "old"), now), None);
        assert_eq!(known[&key].host, "new");
        // A legacy (time 0) copy can't roll a dated record back either.
        assert_eq!(install_known_peer_in(&mut known, &mut received, "me", &rec(0, "legacy"), now), None);
        assert_eq!(known[&key].host, "new");
        // A newer one replaces and renews.
        assert_eq!(install_known_peer_in(&mut known, &mut received, "me", &rec(now - 1_000, "newer"), now), Some(Install::Renewed));
        assert_eq!((known[&key].host.as_str(), received[&key]), ("newer", now));
        // Over an hour old: refused outright.
        let (mut k2, mut r2) = (HashMap::new(), HashMap::new());
        assert_eq!(install_known_peer_in(&mut k2, &mut r2, "me", &rec(now - dht::MAX_RECORD_AGE_MS - 1, "x"), now), None);
        // Undated records from older nodes still renew on receipt.
        assert_eq!(install_known_peer_in(&mut k2, &mut r2, "me", &rec(0, "a"), now - 5), Some(Install::New));
        assert_eq!(install_known_peer_in(&mut k2, &mut r2, "me", &rec(0, "a"), now), Some(Install::Renewed));
        assert_eq!(r2[&key], now);
    }

    #[test]
    fn only_a_lookup_started_after_the_invalidate_clears_it() {
        assert!(started_after(None, 1));
        assert!(started_after(Some(5), 6));
        assert!(!started_after(Some(5), 4), "a lookup already running when every candidate failed");
    }

    fn ma(s: &str) -> Multiaddr {
        s.parse().unwrap()
    }

    #[test]
    fn identify_puts_only_circuits_through_our_relays_in_the_routing_table() {
        let (peer, relay, stranger) = (PeerId::random(), PeerId::random(), PeerId::random());
        let mut trusted = crate::relay::TrustedRelays::default();
        trusted.add(&ma(&format!("/ip4/45.79.190.172/tcp/4001/p2p/{relay}")));
        let ours = ma(&format!("/ip4/45.79.190.172/tcp/4001/p2p/{relay}/p2p-circuit/p2p/{peer}"));
        let theirs = ma(&format!("/ip4/9.9.9.9/tcp/4001/p2p/{stranger}/p2p-circuit/p2p/{peer}"));
        let claims = vec![
            ma("/ip4/8.8.8.8/tcp/4001"),        // a direct claim: never into the table
            ma("/ip6/2606:4700::1111/udp/4001/quic-v1"),
            theirs,                              // a circuit through a relay we don't use
            ours.clone(),
        ];
        let got = identify_routing_addrs(&claims, &peer, &trusted, true);
        assert_eq!(got.len(), 1);
        assert!(got[0].to_string().contains(&relay.to_base58()), "{got:?}");
        // The relay's real peer id in front of other machines' addresses (review 2026-10-29, which
        // this test used to accept): none goes into the table, however many it claims.
        let flood: Vec<Multiaddr> = (0..50)
            .map(|i| ma(&format!("/ip4/45.79.190.{i}/tcp/4001/p2p/{relay}/p2p-circuit/p2p/{peer}")))
            .filter(|a| !a.to_string().starts_with("/ip4/45.79.190.172/"))
            .collect();
        assert!(identify_routing_addrs(&flood, &peer, &trusted, true).is_empty());
        // Many genuine ones (a relay known at several addresses) are still capped.
        for i in 0..20 {
            trusted.add_for(relay, ma(&format!("/ip4/45.79.191.{i}/tcp/4001")));
        }
        let many: Vec<Multiaddr> = (0..20)
            .map(|i| ma(&format!("/ip4/45.79.191.{i}/tcp/4001/p2p/{relay}/p2p-circuit/p2p/{peer}")))
            .collect();
        assert_eq!(identify_routing_addrs(&many, &peer, &trusted, true).len(), MAX_IDENTIFY_CIRCUIT_ADDRS);
    }

    #[test]
    fn a_relays_own_addresses_are_taken_global_only_on_the_public_network() {
        let claims = vec![
            ma("/ip4/127.0.0.1/tcp/4001"),
            ma("/ip4/10.0.0.5/tcp/4001"),
            ma("/ip4/172.17.0.1/tcp/4001"),
            ma("/ip4/45.79.190.172/udp/4001/quic-v1"),
            ma("/ip6/2600:3c03::2000:68ff:fe81:55b0/tcp/4001"),
            ma(&format!("/ip4/1.2.3.4/tcp/4001/p2p/{}/p2p-circuit", PeerId::random())),
        ];
        let public = relay_self_addrs(&claims, true);
        assert_eq!(public, vec![claims[3].clone(), claims[4].clone()]);
        let private = relay_self_addrs(&claims, false);
        assert_eq!(private.len(), 5, "a private deployment's relays are its own: their LAN addresses count");
    }

    #[test]
    fn identify_dial_addresses_only_on_the_ip_the_connection_comes_from() {
        let conn = ma("/ip4/54.116.38.169/tcp/51868");
        let claims = vec![
            ma("/ip4/54.116.38.169/tcp/4001"),
            ma("/ip4/54.116.38.169/udp/4001/quic-v1"),
            ma("/ip4/1.2.3.4/tcp/22"),          // someone else's machine
            ma("/ip4/10.0.0.5/tcp/4001"),       // a private address, not a LAN peer
        ];
        let got = identify_dial_addrs(&claims, Some(&conn), false, true);
        assert_eq!(got, vec![claims[0].clone(), claims[1].clone()]);
        assert!(identify_dial_addrs(&claims, None, false, true).is_empty(), "a relayed connection gives none");
        let many: Vec<Multiaddr> = (4000..4020).map(|p| ma(&format!("/ip4/54.116.38.169/tcp/{p}"))).collect();
        assert_eq!(identify_dial_addrs(&many, Some(&conn), false, true).len(), MAX_IDENTIFY_DIAL_ADDRS);
        // A LAN peer's own LAN address is fine for a LAN dial.
        let lan = ma("/ip4/192.168.1.6/tcp/60000");
        assert_eq!(identify_dial_addrs(&[ma("/ip4/192.168.1.6/tcp/4111")], Some(&lan), true, true).len(), 1);
    }

    #[test]
    fn only_a_dht_servers_global_direct_addresses_go_into_the_routing_table() {
        let ma = |s: &str| s.parse::<Multiaddr>().unwrap();
        let direct = vec![ma("/ip4/45.79.190.172/udp/4001/quic-v1"), ma("/ip4/192.168.1.6/tcp/4111")];
        assert_eq!(kad_addrs_from_identify(true, true, &direct), vec![ma("/ip4/45.79.190.172/udp/4001/quic-v1")]);
        assert!(kad_addrs_from_identify(false, true, &direct).is_empty(), "a peer that doesn't serve the DHT gives none");
        assert_eq!(kad_addrs_from_identify(true, false, &direct), direct, "a private network takes its private addresses");
        assert!(kad_addrs_from_identify(false, false, &direct).is_empty());
    }

    #[test]
    fn holepunch_dials_only_a_few_global_ipv6_quic_addresses() {
        let claims = vec![
            ma("/ip6/::1/udp/4001/quic-v1"),
            ma("/ip6/fd00::1/udp/4001/quic-v1"),
            ma("/ip6/fe80::1/udp/4001/quic-v1"),
            ma("/ip6/::ffff:1.2.3.4/udp/4001/quic-v1"),
            ma("/ip6/ff02::1/udp/4001/quic-v1"),
            ma("/ip6/2001:db8::1/udp/4001/quic-v1"),
            ma("/ip6/2401:4900:1cb9:a4e7::1/tcp/4001"), // not QUIC
            ma("/ip6/2401:4900:1cb9:a4e7::1/udp/4001/quic-v1"),
            ma("/ip6/2401:4900:1cb9:a4e7::2/udp/4001/quic-v1"),
            ma("/ip6/2401:4900:1cb9:a4e7::3/udp/4001/quic-v1"),
        ];
        let got = holepunch_v6_addrs(&claims);
        assert_eq!(got, vec![claims[7].clone(), claims[8].clone()]);
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
    fn two_providers_sharing_an_openhydra_id_keep_separate_entries() {
        // Two keys whose 64-bit OpenHydra ids collide still have distinct libp2p ids.
        let rec = |libp2p: &str| PeerRecord {
            peer_id: "0123456789abcdef".into(),
            model_id: "qwen".into(),
            libp2p_peer_id: libp2p.into(),
            ..Default::default()
        };
        let mut known: HashMap<String, PeerRecord> = HashMap::new();
        known.insert(known_peer_key(&rec("12D3KooWa")), rec("12D3KooWa"));
        known.insert(known_peer_key(&rec("12D3KooWb")), rec("12D3KooWb"));
        assert_eq!(known.len(), 2, "neither shadows the other's routing entry");
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

    /// Round 8 N8-2 / round 9 C9-1: a record dated far ahead is refused; one peer's records, and
    /// all of them, are bounded by when they were received (honest records carry time 0), never
    /// evicting our own.
    #[test]
    fn known_provider_records_are_bounded_and_never_dated_ahead() {
        let now = 10_000_000;
        let rec = |peer: &str, model: &str, at: u64| PeerRecord {
            peer_id: format!("n-{peer}"),
            model_id: model.into(),
            libp2p_peer_id: peer.into(),
            updated_unix_ms: at,
            ..Default::default()
        };
        let mut known: HashMap<String, PeerRecord> = HashMap::new();
        let mut received: HashMap<String, u64> = HashMap::new();
        let mut put = |r: PeerRecord, t: u64| install_known_peer_in(&mut known, &mut received, "ME", &r, t);
        assert_eq!(put(rec("A", "m", now + MAX_RECORD_SKEW_MS + 1), now), None);
        assert_eq!(put(rec("A", "m", 0), now), Some(Install::New), "an honest record, dated 0");
        assert_eq!(put(rec("A", "m", 0), now + 1), Some(Install::Renewed), "a refresh");
        assert_eq!(put(rec("ME", "mine", 0), now), Some(Install::New));
        assert_eq!(put(rec("C", "x\" onclick=\"y", 0), now), None, "an odd model name is never cached");
        // A flood of one peer's records, dated "now" by its signer, received after A's: its own
        // least recently received go, nobody else's.
        for i in 0..MAX_KNOWN_RECORDS_PER_PEER as u64 + 5 {
            put(rec("B", &format!("fake-{i}"), now), now + 10 + i);
        }
        drop(put);
        assert_eq!(known.values().filter(|r| r.libp2p_peer_id == "B").count(), MAX_KNOWN_RECORDS_PER_PEER);
        assert!(!known.values().any(|r| r.model_id == "fake-0"), "B's earliest went");
        assert!(known.values().any(|r| r.libp2p_peer_id == "A"), "A's honest record stays");
        assert!(known.values().any(|r| r.libp2p_peer_id == "ME"), "our own stays");
        assert_eq!(received.len(), known.len());
    }

    // ── M6: per-model MODEL_WITHDRAWN tombstone eviction ─────────────────
    fn m6_rec(libp2p: &str, model: &str) -> PeerRecord {
        PeerRecord {
            peer_id: format!("node-for-{libp2p}"),
            model_id: model.into(),
            libp2p_peer_id: libp2p.into(),
            ..Default::default()
        }
    }

    #[test]
    fn m6_evict_withdrawn_model_drops_only_the_matching_peer_and_model() {
        let mut known: HashMap<String, PeerRecord> = HashMap::new();
        // Peer A serves two models; peer B serves one of the same model names.
        for r in [
            m6_rec("12D3KooWA", "qwen3-coder"),
            m6_rec("12D3KooWA", "llama3.1"),
            m6_rec("12D3KooWB", "qwen3-coder"),
        ] {
            known.insert(known_peer_key(&r), r);
        }
        assert_eq!(known.len(), 3);

        // A withdraws just "qwen3-coder": only A's qwen3-coder goes. A's llama3.1 stays
        // (A is still up serving it), and B's qwen3-coder is untouched (different peer).
        let evicted = evict_withdrawn_model(&mut known, "12D3KooWA", "qwen3-coder");
        assert_eq!(evicted, 1);
        assert_eq!(known.len(), 2);
        let mut left: Vec<(String, String)> = known
            .values()
            .map(|r| (r.libp2p_peer_id.clone(), r.model_id.clone()))
            .collect();
        left.sort();
        assert_eq!(
            left,
            vec![
                ("12D3KooWA".to_string(), "llama3.1".to_string()),
                ("12D3KooWB".to_string(), "qwen3-coder".to_string()),
            ]
        );
    }

    #[test]
    fn m6_evict_withdrawn_model_unknown_pair_is_a_noop() {
        let mut known: HashMap<String, PeerRecord> = HashMap::new();
        let r = m6_rec("12D3KooWA", "qwen3-coder");
        known.insert(known_peer_key(&r), r);
        // A model we never learned from that peer → nothing removed, map intact.
        assert_eq!(evict_withdrawn_model(&mut known, "12D3KooWA", "not-served"), 0);
        assert_eq!(evict_withdrawn_model(&mut known, "12D3KooWZ", "qwen3-coder"), 0);
        assert_eq!(known.len(), 1);
    }

    #[test]
    fn m6_desired_diff_yields_exactly_the_unshared_models() {
        // Mirrors announce_models_inner's tombstone diff: last pass advertised {a, b, c};
        // this pass's intent is {a, c}; so exactly {b} is withdrawn, and {a,c} becomes the new
        // baseline (no tombstone for a still-shared or a newly-shared model).
        use std::collections::BTreeSet;
        let last: BTreeSet<String> =
            ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let desired: BTreeSet<String> =
            ["a", "c", "d"].iter().map(|s| s.to_string()).collect();
        let withdrawn: Vec<String> = last.difference(&desired).cloned().collect();
        assert_eq!(withdrawn, vec!["b".to_string()]);
    }

    #[test]
    fn m6_f1_withdrawal_freshness_window() {
        let max = MODEL_WITHDRAWN_MAX_AGE_SECS;
        let now = 1_000_000u64;
        // Exactly-now and within the window (either direction) → fresh.
        assert!(withdrawal_is_fresh(now, now, max));
        assert!(withdrawal_is_fresh(now - max, now, max)); // oldest still-accepted
        assert!(withdrawal_is_fresh(now + max, now, max)); // future-skew boundary
        assert!(withdrawal_is_fresh(now - max + 1, now, max));
        // Older than the window → stale (a replay lands here once the original ages out).
        assert!(!withdrawal_is_fresh(now - max - 1, now, max));
        assert!(!withdrawal_is_fresh(now - 3600, now, max));
        // Absurd future timestamp → also rejected (can't outrun the age check).
        assert!(!withdrawal_is_fresh(now + max + 1, now, max));
        // Clock at 0 vs a real timestamp must not underflow into a false "fresh".
        assert!(!withdrawal_is_fresh(1_000_000, 0, max));
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
    fn only_the_simulator_switch_makes_172_16_public() {
        let check = |s: &str, sim: bool| reachable_addr_check(&s.parse::<Multiaddr>().unwrap(), sim);
        // Off (every real node): 172.16.0.0/12 is private, as before.
        assert!(!check("/ip4/172.31.72.84/udp/4001/quic-v1", false));
        // On: all of 172.16.0.0/12 and nothing past it.
        assert!(check("/ip4/172.31.72.84/udp/4001/quic-v1", true));
        assert!(check("/ip4/172.16.0.1/tcp/4001", true));
        // Just outside the range (public addresses anyway): the switch changes nothing.
        for edge in ["/ip4/172.32.0.1/tcp/4001", "/ip4/172.15.255.255/tcp/4001"] {
            assert_eq!(check(edge, true), check(edge, false), "{edge}");
        }
        // The NAT homes (10/8), home LANs (192.168/16), CGNAT and loopback stay private.
        assert!(!check("/ip4/10.101.0.2/tcp/4001", true));
        assert!(!check("/ip4/192.168.0.5/tcp/4001", true));
        assert!(!check("/ip4/100.64.1.1/tcp/4001", true));
        assert!(!check("/ip4/127.0.0.1/tcp/4001", true));
        // A relay circuit through a 172.31 relay is still not "us being reachable".
        assert!(!check("/ip4/172.31.72.89/udp/4001/quic-v1/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit", true));
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

        let relays = crate::relay::relay_addrs(&[], crate::relay::PublicRelays::Auto);
        let no_v6 = relay_circuit_addrs(&relays, target, false);
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

        let with_v6 = relay_circuit_addrs(&relays, target, true);
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
        assert_eq!(reservation_retry_delay_ms(12), 120_000);
        // A relay that keeps refusing: every 10 minutes, logged once an hour.
        assert_eq!(reservation_retry_delay_ms(13), 600_000);
        assert_eq!(reservation_retry_delay_ms(50), 600_000);
        assert_eq!(reservation_retry_delay_ms(u32::MAX), 600_000);
        assert!((1..=12).all(reservation_retry_loud));
        assert_eq!((13..=36).filter(|a| reservation_retry_loud(*a)).count(), 4, "one an hour");
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

    // ── withdrawn models (sticky model lists drop them at once) ───────────

    #[test]
    fn a_model_is_remembered_as_withdrawn_only_when_nobody_serves_it() {
        let rec = |peer: &str, model: &str| PeerRecord { libp2p_peer_id: peer.into(), model_id: model.into(), ..Default::default() };
        let mut known: HashMap<String, PeerRecord> = HashMap::new();
        known.insert("a/m".into(), rec("a", "m"));
        known.insert("b/m".into(), rec("b", "m"));
        known.insert("a/n".into(), rec("a", "n"));
        let mut withdrawn = HashMap::new();
        let mut seen = HashMap::new();
        let t0 = std::time::Instant::now();
        // a withdraws m, but b still serves it: m hasn't left the network.
        assert_eq!(evict_withdrawn_model(&mut known, "a", "m"), 1);
        note_withdrawn_model(&mut withdrawn, &known, &mut seen, "m", "a", t0);
        assert!(recently_withdrawn(&mut withdrawn, &known, t0).is_empty());
        // a withdraws n, its only provider: remembered.
        evict_withdrawn_model(&mut known, "a", "n");
        note_withdrawn_model(&mut withdrawn, &known, &mut seen, "n", "a", t0);
        assert_eq!(recently_withdrawn(&mut withdrawn, &known, t0), vec!["n".to_string()]);
        // Re-shared: no longer reported (the record is back).
        known.insert("a/n".into(), rec("a", "n"));
        assert!(recently_withdrawn(&mut withdrawn, &known, t0).is_empty());
        known.remove("a/n");
        // Forgotten after the memory window.
        let later = t0 + WITHDRAWN_MODEL_MEMORY + std::time::Duration::from_secs(1);
        assert!(recently_withdrawn(&mut withdrawn, &known, later).is_empty());
        assert!(withdrawn.is_empty());
    }

    #[test]
    fn a_withdrawal_doesnt_hide_a_model_another_provider_announced_lately() {
        // Review 2026-10-14 L8: b's record for m lapsed here, then a withdrew m: b still serves it.
        let known: HashMap<String, PeerRecord> = HashMap::new();
        let mut seen = HashMap::new();
        let mut withdrawn = HashMap::new();
        let t0 = std::time::Instant::now();
        note_model_seen(&mut seen, "m", "b", t0);
        note_model_seen(&mut seen, "m", "a", t0);
        note_withdrawn_model(&mut withdrawn, &known, &mut seen, "m", "a", t0 + std::time::Duration::from_secs(60));
        assert!(withdrawn.is_empty(), "b announced m a minute ago");
        // Only a ever served n: its withdrawal hides n at once.
        note_model_seen(&mut seen, "n", "a", t0);
        note_withdrawn_model(&mut withdrawn, &known, &mut seen, "n", "a", t0);
        assert!(withdrawn.contains_key("n"));
        // b's announcement long past: m's withdrawal counts.
        let later = t0 + WITHDRAWN_MODEL_MEMORY + std::time::Duration::from_secs(1);
        note_withdrawn_model(&mut withdrawn, &known, &mut seen, "m", "a", later);
        assert!(withdrawn.contains_key("m"));
        // L7: older withdrawals are pruned as new ones come, and the map is bounded.
        assert!(!withdrawn.contains_key("n"), "n's was past the memory window");
        for i in 0..MAX_WITHDRAWN_MODELS + 10 {
            note_withdrawn_model(&mut withdrawn, &known, &mut seen, &format!("x{i}"), "a", later);
        }
        assert!(withdrawn.len() <= MAX_WITHDRAWN_MODELS);
    }

    #[test]
    fn when_the_last_of_two_providers_withdraws_the_model_goes_at_once() {
        // Second review 2026-10-19: b withdraws m (a still serves it), then a does: nobody serves it.
        let known: HashMap<String, PeerRecord> = HashMap::new();
        let mut seen = HashMap::new();
        let mut withdrawn = HashMap::new();
        let t0 = std::time::Instant::now();
        note_model_seen(&mut seen, "m", "a", t0);
        note_model_seen(&mut seen, "m", "b", t0);
        note_withdrawn_model(&mut withdrawn, &known, &mut seen, "m", "b", t0 + std::time::Duration::from_secs(10));
        assert!(withdrawn.is_empty(), "a announced m lately");
        note_withdrawn_model(&mut withdrawn, &known, &mut seen, "m", "a", t0 + std::time::Duration::from_secs(20));
        assert!(withdrawn.contains_key("m"), "both have withdrawn it");
    }

    #[test]
    fn two_due_duplicates_never_close_a_peers_last_quic_connection() {
        use libp2p::swarm::ConnectionId;
        let pid = PeerId::random();
        let mut peers = HashMap::new();
        // The original v4 connection closed during the grace; two inbound duplicates remain, due together.
        peers.insert(pid, PeerConnectionInfo { quic_direct_v4: 2, ..Default::default() });
        let (d1, d2) = (ConnectionId::new_unchecked(11), ConnectionId::new_unchecked(12));
        // (Seen from a node whose id is below the peer's: it closes inbound duplicates.)
        let low = lower_than(&pid);
        let (close, stay) = quic_dups_due(&[(d1, pid, false), (d2, pid, false)], &peers, &low);
        assert_eq!(close, vec![d1], "one closes, one stays");
        assert_eq!(stay, vec![(d2, pid, false)]);
        // With the original still open (three in all), both duplicates close.
        peers.insert(pid, PeerConnectionInfo { quic_direct_v4: 3, ..Default::default() });
        let (close, stay) = quic_dups_due(&[(d1, pid, false), (d2, pid, false)], &peers, &low);
        assert_eq!(close, vec![d1, d2]);
        assert!(stay.is_empty());
        // Another family is counted apart.
        peers.insert(pid, PeerConnectionInfo { quic_direct_v4: 2, quic_direct_v6: 1, ..Default::default() });
        let (close, _) = quic_dups_due(&[(d1, pid, false), (d2, pid, true)], &peers, &low);
        assert_eq!(close, vec![d1]);
    }

    /// A peer id below `pid` (ids compare by their bytes).
    fn lower_than(pid: &PeerId) -> PeerId {
        loop {
            let p = PeerId::random();
            if p < *pid {
                return p;
            }
        }
    }

    #[test]
    fn when_both_ends_dial_at_once_the_connection_the_lower_id_dialled_stays() {
        use libp2p::swarm::ConnectionId;
        let a = PeerId::random();
        let b = lower_than(&a); // b < a
        let (x, y) = (ConnectionId::new_unchecked(21), ConnectionId::new_unchecked(22));
        // x: dialled by b; y: dialled by a. Each side has both, one v6 pair.
        let mut at_a = HashMap::new();
        at_a.insert(b, PeerConnectionInfo { quic_direct_v6: 2, ..Default::default() });
        let mut at_b = HashMap::new();
        at_b.insert(a, PeerConnectionInfo { quic_direct_v6: 2, ..Default::default() });
        // At a (the higher id), its inbound duplicate is x, b's: it stays.
        let (close, _) = quic_dups_due(&[(x, b, true)], &at_a, &a);
        assert!(close.is_empty(), "the higher id leaves the lower id's connection alone");
        // At b (the lower id), its inbound duplicate is y, a's: it closes. x survives on both ends.
        let (close, _) = quic_dups_due(&[(y, a, true)], &at_b, &b);
        assert_eq!(close, vec![y]);
        // More than a pair is closed down as before, whoever is higher.
        at_a.insert(b, PeerConnectionInfo { quic_direct_v6: 3, ..Default::default() });
        let (close, _) = quic_dups_due(&[(x, b, true)], &at_a, &a);
        assert_eq!(close, vec![x]);
    }

    #[test]
    fn a_private_nodes_failed_reservation_says_what_to_do() {
        let own: Multiaddr = format!("/ip4/10.0.0.5/tcp/4001/p2p/{}", PeerId::random()).parse().unwrap();
        let circuit = own.clone().with(libp2p::multiaddr::Protocol::P2pCircuit);
        let hint = private_relay_hint(&[own.clone()], &circuit, "no reservation").unwrap();
        assert!(hint.contains(&own.to_string()) && hint.contains("--public-relays on"), "{hint}");
        // On the public network OpenHydra's relays are there too: nothing to say.
        let us: Multiaddr = crate::relay::BOOTSTRAP_RELAYS[0].parse().unwrap();
        assert!(private_relay_hint(&[us, own], &circuit, "no reservation").is_none());
    }

    // ── QUIC dedup vs AutoNAT v2 dial-back ─────────────────────────────────

    #[test]
    fn quic_dedup_defers_inbound_duplicates_so_autonat_dial_backs_complete() {
        // The AutoNAT server's dial-back arrives as an inbound (listener-side) second QUIC
        // connection; closing it at once failed every test of a QUIC-only node's address.
        assert_eq!(quic_dup_action(true), QuicDupAction::CloseAfterGrace);
        // A duplicate we dialed ourselves still closes immediately.
        assert_eq!(quic_dup_action(false), QuicDupAction::CloseNow);
        // Long enough for a dial-back's few round trips, short next to the 300s idle timeout.
        assert!(QUIC_INBOUND_DUP_GRACE >= std::time::Duration::from_secs(5));
        assert!(QUIC_INBOUND_DUP_GRACE <= std::time::Duration::from_secs(60));
    }

    #[test]
    fn quic_dup_closes_after_grace_only_while_another_same_af_connection_remains() {
        let redundant = |info: PeerConnectionInfo, v6: bool| {
            let pid = PeerId::random();
            let peers = HashMap::from([(pid, info)]);
            !quic_dups_due(&[(libp2p::swarm::ConnectionId::new_unchecked(1), pid, v6)], &peers, &lower_than(&pid)).0.is_empty()
        };
        // original + deferred duplicate
        assert!(redundant(PeerConnectionInfo { quic_direct_v4: 2, ..Default::default() }, false));
        assert!(!redundant(PeerConnectionInfo { quic_direct_v4: 2, ..Default::default() }, true), "no v6 connections: nothing to dedup");
        // The original closed during the grace window: the duplicate is now the only one — keep it.
        assert!(!redundant(PeerConnectionInfo { quic_direct_v4: 1, ..Default::default() }, false));
        assert!(redundant(PeerConnectionInfo { quic_direct_v4: 1, quic_direct_v6: 2, ..Default::default() }, true));
    }

    // ── queued forwards fail when the last dial to their peer fails ────────

    #[test]
    fn a_failed_dial_fails_queued_forwards_only_when_nothing_else_can_reach_the_peer() {
        use libp2p::swarm::ConnectionId;
        let peer = PeerId::random();
        let (c1, c2) = (ConnectionId::new_unchecked(1), ConnectionId::new_unchecked(2));
        let mut dials: HashMap<PeerId, std::collections::HashSet<ConnectionId>> = HashMap::new();
        dials.entry(peer).or_default().extend([c1, c2]);
        // Our relay dial fails while a second dial (a Kademlia query, say) is still pending: wait.
        assert!(!dial_ended_unreachable(&mut dials, peer, c1, false));
        // The last one fails too, no connection: fail the queued forwards now.
        assert!(dial_ended_unreachable(&mut dials, peer, c2, false));
        assert!(dials.is_empty(), "nothing left tracked for the peer");
        // A dial fails but the peer is connected some other way: the forwards ride that connection.
        dials.entry(peer).or_default().insert(c1);
        assert!(!dial_ended_unreachable(&mut dials, peer, c1, true));
        // An error for a dial we never saw start (no Dialing event): judged on the connection alone.
        assert!(dial_ended_unreachable(&mut dials, PeerId::random(), c2, false));
    }

    #[tokio::test]
    async fn our_own_dials_are_tracked_so_another_dial_failing_doesnt_fail_the_forward() {
        use futures::StreamExt;
        use libp2p::core::transport::{MemoryTransport, Transport};
        use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
        use libp2p::swarm::{dummy, ConnectionId, SwarmEvent};
        let key = libp2p::identity::Keypair::generate_ed25519();
        let transport = MemoryTransport::default()
            .upgrade(libp2p::core::upgrade::Version::V1)
            .authenticate(libp2p::noise::Config::new(&key).unwrap())
            .multiplex(libp2p::yamux::Config::default())
            .boxed();
        let cfg = libp2p::swarm::Config::with_tokio_executor();
        let mut swarm = libp2p::Swarm::new(transport, dummy::Behaviour, PeerId::from(key.public()), cfg);
        let peer = PeerId::random();
        // Nothing listens there: the dial fails, but only after it has started.
        let nowhere: Multiaddr = format!("/memory/{}", rand::random::<u64>()).parse().unwrap();
        let opts = DialOpts::peer_id(peer).addresses(vec![nowhere]).condition(PeerCondition::Always).build();
        let ours = opts.connection_id();
        let mut dials: DialsInFlight = HashMap::new();
        swarm.dial_tracked(&mut dials, opts).unwrap();
        assert!(dials.get(&peer).is_some_and(|s| s.contains(&ours)), "our dial is tracked");
        // Another dial to the peer (a Kademlia query's) fails first: ours is still running, so wait.
        assert!(!dial_ended_unreachable(&mut dials, peer, ConnectionId::new_unchecked(999_999), false));
        // Ours fails: now nothing can reach the peer.
        let failed = loop {
            if let SwarmEvent::OutgoingConnectionError { connection_id, peer_id: Some(p), .. } = swarm.select_next_some().await {
                assert_eq!(p, peer);
                break connection_id;
            }
        };
        assert_eq!(failed, ours);
        assert!(dial_ended_unreachable(&mut dials, peer, failed, false));
    }

    #[tokio::test]
    async fn failing_a_peers_forwards_answers_them_and_keeps_the_others() {
        let (a, b) = (PeerId::random(), PeerId::random());
        let now = std::time::Instant::now();
        let (ta, ra) = oneshot::channel();
        let (ta2, ra2) = oneshot::channel();
        let (tb, mut rb) = oneshot::channel();
        let forwards = vec![(a, vec![1], ta, now), (b, vec![2], tb, now), (a, vec![3], ta2, now)];
        let (kept, failed) = fail_relay_forwards_for(forwards, a, "dial failed");
        assert_eq!(failed, 2);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].0, b);
        assert!(ra.await.unwrap().unwrap_err().contains("dial failed"));
        assert!(ra2.await.unwrap().is_err());
        assert!(rb.try_recv().is_err(), "b's forward is still waiting");
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

    /// The cause the other side sees when a peer closes a connection on purpose over `listen`
    /// (a QUIC or TCP address on loopback), from real swarms.
    async fn cause_seen_when_the_peer_closes(listen: &str) -> Option<libp2p::swarm::ConnectionError> {
        use futures::StreamExt;
        use libp2p::swarm::{dummy, SwarmEvent};
        let build = || {
            libp2p::SwarmBuilder::with_new_identity()
                .with_tokio()
                .with_tcp(libp2p::tcp::Config::default(), libp2p::noise::Config::new, libp2p::yamux::Config::default)
                .unwrap()
                .with_quic()
                .with_behaviour(|_| dummy::Behaviour)
                .unwrap()
                .with_swarm_config(|c| c.with_idle_connection_timeout(std::time::Duration::from_secs(60)))
                .build()
        };
        let (mut a, mut b) = (build(), build());
        b.listen_on(listen.parse().unwrap()).unwrap();
        let addr = loop {
            if let SwarmEvent::NewListenAddr { address, .. } = b.select_next_some().await {
                break address;
            }
        };
        a.dial(addr.with(libp2p::multiaddr::Protocol::P2p(*b.local_peer_id()))).unwrap();
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut a_conn = None;
        let mut b_up = false;
        loop {
            tokio::select! {
                e = a.select_next_some() => {
                    if let SwarmEvent::ConnectionEstablished { connection_id, .. } = e {
                        a_conn = Some(connection_id);
                    }
                }
                e = b.select_next_some() => match e {
                    SwarmEvent::ConnectionEstablished { .. } => b_up = true,
                    SwarmEvent::ConnectionClosed { cause, .. } => return cause,
                    _ => {}
                },
                _ = tokio::time::sleep_until(deadline) => panic!("no close seen over {listen}"),
            }
            if b_up {
                if let Some(c) = a_conn.take() {
                    assert!(a.close_connection(c));
                }
            }
        }
    }

    #[tokio::test]
    async fn a_close_the_peer_meant_is_told_from_a_crash() {
        for listen in ["/ip4/127.0.0.1/udp/0/quic-v1", "/ip4/127.0.0.1/tcp/0"] {
            let cause = cause_seen_when_the_peer_closes(listen).await;
            assert!(closed_by_peer(cause.as_ref()), "{listen}: {cause:?}");
        }
        use libp2p::swarm::ConnectionError;
        assert!(!closed_by_peer(None), "our own close");
        assert!(!closed_by_peer(Some(&ConnectionError::KeepAliveTimeout)));
        let bare = ConnectionError::IO(std::io::Error::from(std::io::ErrorKind::ConnectionReset));
        assert!(!closed_by_peer(Some(&bare)), "not from a muxer");
        let timed_out = ConnectionError::IO(std::io::Error::new(std::io::ErrorKind::Other, "connection timed out"));
        assert!(!closed_by_peer(Some(&timed_out)));
    }
}
