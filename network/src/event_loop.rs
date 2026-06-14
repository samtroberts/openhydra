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

use crate::batcher::{Batcher, BatchItem, BatchKey, DtypeTag, FlushedBatch};
use crate::behaviour::{OpenHydraBehaviour, OpenHydraBehaviourEvent};
use crate::dht;
use crate::dispatcher::{self, DispatchAction, DispatchMode, Dispatcher, PeerStatusCache};
use crate::forward_msg;
use crate::ipc::IpcBridge;
use crate::ipc_codec::IpcForwardHeader;
use crate::proxy::{self, ProxyRequest, ProxyResponse};
use crate::ring::{RingAction, RingConfig, RingHandle, RingManager};
use crate::sampler_bridge::{SamplerBridge, SampleRequest};
use crate::tensor_stream::{self, TensorStreamManager};
use crate::types::{DiscoveredPeer, NatInfo, PeerRecord};

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
    if addr_str.contains("/quic") || addr_str.contains("/quic-v1") {
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
}

impl PeerConnectionInfo {
    fn has_direct(&self) -> bool {
        self.quic_direct_v4 > 0 || self.quic_direct_v6 > 0 || self.tcp_direct > 0
    }

    fn direct_count(&self) -> u32 {
        self.quic_direct_v4 + self.quic_direct_v6 + self.tcp_direct
    }
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

/// CP-3: Events from spawned sampler tasks back to the event loop.
///
/// The sampler runs async (IPC to Python HeadSampler), so results flow
/// back via this enum on an mpsc channel. The event loop then calls
/// ring_manager methods and re-injects into the ring.
#[derive(Debug)]
enum RingEvent {
    /// HeadSampler returned a token + next-token embedding.
    TokenSampled {
        session_id: String,
        token_id: u32,
        token_text: String,
        is_eos: bool,
        /// Raw float32 bytes of the next-token embedding vector.
        embedding: Vec<u8>,
    },
    /// HeadSampler call failed.
    SampleFailed {
        session_id: String,
        reason: String,
    },
    /// Audit F11: a fire-and-forget ring re-injection send failed. The send
    /// runs in a spawned task, so it reports failure back here to abort the
    /// session immediately instead of waiting ~30s for the watchdog.
    ReinjectFailed {
        session_id: String,
        reason: String,
    },
}

/// CP-4: Metadata for a request waiting inside the Batcher.
///
/// When a ForwardMsg enters the Batcher, the full IPC header is stored here
/// (keyed by `proxy_req_id`) so it can be reconstructed when the batch flushes.
struct BatchPendingItem {
    header: IpcForwardHeader,
    /// True for blocking ForwardToWorker (needs response routing).
    /// False for ForwardToWorkerAsync (fire-and-forget, already ACK'd).
    needs_response: bool,
}

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
    // ── CP-2: Dispatcher wiring ──────────────────────────────────────
    /// Start the IPC bridge (binds a Unix socket for the Python worker).
    /// Must be called before the dispatcher can forward to a local worker.
    StartIpcBridge {
        socket_path: String,
        reply: oneshot::Sender<Result<(), String>>,
    },
    /// Update the cached peer status used by inline Ping/GetPeerStatus
    /// responses.  Called periodically from Python to keep the cache fresh.
    UpdateDispatcherStatus {
        status: PeerStatusCache,
    },
    /// Switch the dispatcher between Peer and Coordinator mode.
    SetDispatcherMode {
        coordinator: bool,
    },
    // ── CP-3: Ring Manager commands ─────────────────────────────────
    /// Start a ring inference session.
    ///
    /// The event loop:
    ///   1. Registers the session with the RingManager.
    ///   2. Pre-dials every peer in the route (critical for cross-ISP).
    ///   3. Returns a `RingHandle` with a token receiver channel.
    ///
    /// The ring doesn't start circulating until the caller injects the
    /// first embedding via a separate ProxyForward to stage 0.
    StartRing {
        config: RingConfig,
        reply: oneshot::Sender<Result<RingHandle, String>>,
    },
    /// Connect the HeadSampler bridge (binds to the Python process's
    /// Unix socket for token sampling during ring inference).
    StartSampler {
        socket_path: String,
        reply: oneshot::Sender<Result<(), String>>,
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
pub struct SharedProxyQueue {
    queue: Mutex<VecDeque<(String, Vec<u8>)>>,
    condvar: Condvar,
}

impl SharedProxyQueue {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            condvar: Condvar::new(),
        }
    }

    pub fn push(&self, item: (String, Vec<u8>)) {
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

    pub fn pop(&self, timeout: std::time::Duration) -> Option<(String, Vec<u8>)> {
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
    /// (target_peer_id, data, reply_channel)
    pending_relay_forwards: Vec<(PeerId, Vec<u8>, oneshot::Sender<Result<Vec<u8>, String>>)>,
    /// DCUtR hole punch counters.
    dcutr_successes: u64,
    dcutr_failures: u64,
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
    /// Fix 1: tensor stream manager reference.
    tensor_mgr: Option<Arc<TensorStreamManager>>,
    /// Fix 1: inbound stream response handles for RespondProxy.
    /// Audit F4: value is the split write-half, not the whole stream.
    inbound_stream_responses: Arc<tokio::sync::Mutex<crate::tensor_stream::InboundStreamMap>>,
    /// Phase 2: DCUtR success event queue (peer_ids as strings).
    dcutr_event_queue: VecDeque<String>,
    /// Phase 2: tunnel close event queue.
    tunnel_close_queue: VecDeque<String>,
    // ── CP-2: Dispatcher wiring ──────────────────────────────────────
    /// Inbound message dispatcher (always present; routes ForwardMsg vs legacy).
    dispatcher: Dispatcher,
    /// IPC bridge to the Python worker daemon (set via StartIpcBridge command).
    ipc_bridge: Option<IpcBridge>,
    /// Channel for spawned IPC tasks to return responses to the event loop.
    /// The event loop picks up (request_id, response_data) and sends via
    /// the stored `inbound_proxy_channels` response channel.
    ipc_response_tx: mpsc::UnboundedSender<(String, Vec<u8>)>,
    // ── CP-3: Ring Manager ──────────────────────────────────────────
    /// Manages active ring inference sessions. Routes PushResult messages
    /// to the correct session via request_id lookup, tracks shard map,
    /// and provides the pre-dial peer list for cross-ISP connectivity.
    ring_manager: RingManager,
    /// HeadSampler bridge for token sampling during ring inference.
    sampler_bridge: Option<SamplerBridge>,
    /// Channel for spawned sampler tasks to return results to the event loop.
    ring_event_tx: mpsc::UnboundedSender<RingEvent>,
    // ── CP-4: Continuous Batching ──────────────────────────────────────
    /// Heterogeneous-safe batch accumulator. Groups incoming ForwardMsg
    /// payloads by BatchKey before dispatching to the IPC bridge.
    batcher: Batcher,
    /// Maps proxy_req_id → (IpcForwardHeader, needs_response).
    /// Populated when items enter the batcher, drained on batch dispatch.
    batch_pending: HashMap<String, BatchPendingItem>,
    // ── CP-5: Prefill Pipeline ────────────────────────────────────────
    /// Maps libp2p outbound request_id → session_id for Stage 0 ACK routing.
    ///
    /// When a prefill chunk is injected into Stage 0 via `send_request`,
    /// the outbound request_id is stored here. When Stage 0 responds
    /// (= "I processed the chunk and forwarded to Stage 1"), the event
    /// loop looks up the session_id and injects the next chunk.
    prefill_stage0_acks: HashMap<request_response::OutboundRequestId, String>,
    /// PR-3 (B1): inbound gossip messages awaiting Python poll. Each entry
    /// is ``(sender_libp2p_peer_id, payload_bytes)``. Bounded ring — the
    /// Rust side drops the oldest when the queue exceeds
    /// ``GOSSIP_INBOUND_QUEUE_MAX`` to prevent unbounded memory growth
    /// when Python is slow to poll.
    gossip_inbound_queue: std::collections::VecDeque<(String, Vec<u8>)>,
    /// B2: Per-peer relay dial retry state: (attempt_count, last_attempt).
    relay_dial_retries: HashMap<PeerId, (u32, tokio::time::Instant)>,
    /// F-5: Per-relay-circuit RESERVATION retry state, keyed by the
    /// ``/p2p-circuit`` listen multiaddr string: (attempt_count,
    /// next_attempt_at). Distinct from ``relay_dial_retries`` (which retries
    /// *dialing a remote peer* via relay) — this retries OUR OWN reservation
    /// (``listen_on`` a circuit) so we stay reachable when a relay rejects or
    /// drops the reservation. Without backoff a flapping relay caused either a
    /// tight re-listen loop or (on a hard listen_on error) a permanent strand
    /// for the whole session (the EU-relay gap). Cleared on NewListenAddr /
    /// ReservationReqAccepted for that circuit.
    relay_reservation_retries: HashMap<String, (u32, tokio::time::Instant)>,
    /// WS-F F-4: peer-relay leech table (None unless peer-relay mode is on).
    /// The RelayServer event handler records byte-cap cap-outs here.
    peer_relay_leech: Option<std::sync::Arc<std::sync::Mutex<crate::relay::LeechTable>>>,
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
    fn new(
        ipc_response_tx: mpsc::UnboundedSender<(String, Vec<u8>)>,
        ring_event_tx: mpsc::UnboundedSender<RingEvent>,
    ) -> Self {
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
            pending_discovers: HashMap::new(),
            external_addrs: Vec::new(),
            relay_addrs: Vec::new(),
            pending_proxy: HashMap::new(),
            local_grpc_port: 50051,
            inbound_proxy_channels: HashMap::new(),
            inbound_proxy_counter: 0,
            pending_relay_forwards: Vec::new(),
            dcutr_successes: 0,
            dcutr_failures: 0,
            tier_connect_success: std::collections::HashMap::new(),
            peer_connections: HashMap::new(),
            local_proxy_replies: HashMap::new(),
            last_repunch: HashMap::new(),
            peer_quic_addrs: HashMap::new(),
            quic_holepunch_attempts: HashMap::new(),
            tensor_mgr: None,
            inbound_stream_responses: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            dcutr_event_queue: VecDeque::new(),
            tunnel_close_queue: VecDeque::new(),
            dispatcher: Dispatcher::new(DispatchMode::Peer),
            ipc_bridge: None,
            ipc_response_tx,
            ring_manager: RingManager::new(),
            sampler_bridge: None,
            ring_event_tx,
            batcher: Batcher::with_defaults(),
            batch_pending: HashMap::new(),
            prefill_stage0_acks: HashMap::new(),
            gossip_inbound_queue: std::collections::VecDeque::new(),
            relay_dial_retries: HashMap::new(),
            relay_reservation_retries: HashMap::new(),
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
    mut stream_control: libp2p_stream::Control,
    keypair: libp2p::identity::Keypair,
    // WS-F F-4: shared leech table for the peer-relay server (None unless this
    // node opted into peer-relay mode). The RelayServer event handler records
    // byte-cap cap-outs into it; the relay::Config's LeechRateLimiter reads it.
    peer_relay_leech: Option<std::sync::Arc<std::sync::Mutex<crate::relay::LeechTable>>>,
) {
    // CP-2: IPC response channel — spawned IPC tasks send (request_id, data)
    // back here so the event loop can forward via request_response.
    let (ipc_response_tx, mut ipc_response_rx) =
        mpsc::unbounded_channel::<(String, Vec<u8>)>();

    // CP-3: Ring event channel — spawned sampler tasks send token results
    // back here so the event loop can record tokens and re-inject.
    let (ring_event_tx, mut ring_event_rx) =
        mpsc::unbounded_channel::<RingEvent>();

    let mut state = LoopState::new(ipc_response_tx, ring_event_tx);
    state.peer_relay_leech = peer_relay_leech; // WS-F F-4

    // Fix 1: set up persistent tensor streams.
    let (repunch_tx, mut repunch_rx) = mpsc::unbounded_channel::<PeerId>();
    // WS-F F-6: circuit-migration signals from the tensor stream manager.
    let (migrate_tx, mut migrate_rx) =
        mpsc::unbounded_channel::<(PeerId, crate::relay::MigrationAction)>();
    let tensor_control = stream_control.clone();
    let tensor_mgr = Arc::new(TensorStreamManager::new(tensor_control, repunch_tx, migrate_tx));
    state.tensor_mgr = Some(Arc::clone(&tensor_mgr));

    // Accept inbound tensor streams.
    let tensor_incoming = stream_control.accept(tensor_stream::TENSOR_STREAM_PROTOCOL)
        .expect("tensor stream protocol already registered");
    tensor_stream::spawn_inbound_acceptor(
        tensor_incoming,
        Arc::clone(&proxy_queue),
        Arc::clone(&state.inbound_stream_responses),
    );

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

    // CP-4: Batch flush ticker — fires every 5ms to drain time-expired batches.
    // C2: Reduced from 5ms to 1ms — caps per-hop queue delay at ~1ms instead of ~5ms.
    let mut batch_ticker = tokio::time::interval(std::time::Duration::from_millis(1));
    batch_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // B3: Ring session timeout watchdog — checks every 5s for stale sessions.
    let mut ring_timeout_ticker = tokio::time::interval(std::time::Duration::from_secs(5));
    ring_timeout_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // F6: B5's periodic re-listen was removed. It called `listen_on(circuit)`
    // every 30 min unconditionally, creating a NEW listener each time without
    // closing the old one (~144 leaked listeners/day → eventual per-peer
    // reservation-cap denial). rust-libp2p's relay::client already auto-renews
    // reservations for active listeners, and the reactive ExpiredListenAddr /
    // ListenerClosed handlers below re-listen if a reservation actually drops —
    // so the periodic re-listen was redundant and leaky.

    // F3: Relay-dial retry driver — fires often enough to honour the
    // 500ms–8s exponential backoff scheduled in relay_dial_retries. The
    // dial itself is gated on each entry's next_attempt_at, so this ticker
    // is cheap (it only acts on peers with a due, scheduled retry).
    let mut relay_retry_ticker = tokio::time::interval(std::time::Duration::from_millis(250));
    relay_retry_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

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
                    Some(SwarmCommand::NatStatus { reply }) => {
                        let _ = reply.send(state.nat_info.clone());
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
                        } else if request_id.starts_with("ts-") {
                            // Fix 1: tensor-stream inbound response.
                            let streams = Arc::clone(&state.inbound_stream_responses);
                            tokio::spawn(async move {
                                if let Err(e) = tensor_stream::write_response(
                                    &streams, &request_id, &data,
                                ).await {
                                    warn!(%request_id, %e, "tensor_stream_respond_failed");
                                }
                            });
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
                    // ── CP-2: Dispatcher commands ────────────────────────
                    Some(SwarmCommand::StartIpcBridge { socket_path, reply }) => {
                        let handle = tokio::runtime::Handle::current();
                        match IpcBridge::start("auto", Some(&socket_path), handle).await {
                            Ok(bridge) => {
                                info!(%socket_path, "IPC bridge started");
                                state.ipc_bridge = Some(bridge);
                                let _ = reply.send(Ok(()));
                            }
                            Err(e) => {
                                warn!(%socket_path, %e, "IPC bridge start failed");
                                let _ = reply.send(Err(e));
                            }
                        }
                    }
                    Some(SwarmCommand::UpdateDispatcherStatus { status }) => {
                        state.dispatcher.update_status(status);
                    }
                    Some(SwarmCommand::SetDispatcherMode { coordinator }) => {
                        let mode = if coordinator {
                            DispatchMode::Coordinator
                        } else {
                            DispatchMode::Peer
                        };
                        state.dispatcher = Dispatcher::new(mode);
                        info!(?coordinator, "dispatcher mode updated");
                    }
                    // ── CP-3: Ring Manager commands ──────────────────────
                    Some(SwarmCommand::StartRing { config, reply }) => {
                        // 1. Pre-dial every peer in the route before starting.
                        let peers_to_dial = RingManager::peers_from_route(&config);
                        let session_id = config.session_id.clone();

                        for peer_id_str in &peers_to_dial {
                            use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
                            match peer_id_str.parse::<PeerId>() {
                                Ok(pid) => {
                                    // Use PeerCondition::Always so we dial even
                                    // if already connected via relay — we want to
                                    // ensure the best available connection.
                                    let opts = DialOpts::peer_id(pid)
                                        .condition(PeerCondition::Disconnected)
                                        .build();
                                    match swarm.dial(opts) {
                                        Ok(()) => {
                                            info!(
                                                %session_id, %pid,
                                                "ring_predial: dialing peer"
                                            );
                                        }
                                        Err(e) => {
                                            // Dial failure at enqueue time is
                                            // not fatal — peer may already be
                                            // connected.
                                            debug!(
                                                %session_id, %pid, %e,
                                                "ring_predial: dial enqueue failed \
                                                 (peer may already be connected)"
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!(
                                        %session_id, %peer_id_str, %e,
                                        "ring_predial: invalid peer_id in route"
                                    );
                                }
                            }
                        }

                        // 2. Register the session with the ring manager.
                        let handle = state.ring_manager.start_session(config);

                        info!(
                            %session_id,
                            peers_dialed = peers_to_dial.len(),
                            "ring session started with pre-dial"
                        );

                        let _ = reply.send(Ok(handle));
                    }
                    Some(SwarmCommand::StartSampler { socket_path, reply }) => {
                        match SamplerBridge::start(&socket_path).await {
                            Ok(bridge) => {
                                info!(%socket_path, "HeadSampler bridge started");
                                state.sampler_bridge = Some(bridge);
                                let _ = reply.send(Ok(()));
                            }
                            Err(e) => {
                                warn!(%socket_path, %e, "HeadSampler bridge failed");
                                let _ = reply.send(Err(e));
                            }
                        }
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
            // CP-2: IPC bridge responses — spawned tasks send completed
            // forward results back here for delivery via request_response.
            Some((req_id, data)) = ipc_response_rx.recv() => {
                if let Some((channel, _, _)) = state.inbound_proxy_channels.remove(&req_id) {
                    if let Err(e) = swarm.behaviour_mut().grpc_proxy
                        .send_response(channel, ProxyResponse(data))
                    {
                        warn!(%req_id, "ipc_response_send_failed: {:?}", e);
                    }
                } else {
                    warn!(%req_id, "ipc_response: no channel found (may have timed out)");
                }
            }
            // CP-3: Ring events — sampler results flowing back from async tasks.
            Some(ring_event) = ring_event_rx.recv() => {
                handle_ring_event(ring_event, &mut swarm, &mut state);
            }
            // Fix 4: process re-punch requests from TensorStreamManager.
            Some(peer_id) = repunch_rx.recv() => {
                handle_trigger_repunch(&mut swarm, peer_id, &mut state);
            }
            // WS-F F-6: circuit-migration signals from TensorStreamManager.
            Some((peer_id, action)) = migrate_rx.recv() => {
                drive_circuit_migration(&mut swarm, &mut state, peer_id, action);
            }
            // Phase 1.4: Periodic known_peers reaper — removes entries whose
            // libp2p_peer_id is no longer connected. Safety net for ghosts
            // that slip through individual eviction paths.
            _ = reaper_interval.tick() => {
                let before = state.known_peers.len();
                state.known_peers.retain(|_openhydra_id, record| {
                    if record.libp2p_peer_id.is_empty() {
                        return true; // keep records without libp2p binding
                    }
                    match record.libp2p_peer_id.parse::<PeerId>() {
                        Ok(pid) => swarm.is_connected(&pid),
                        Err(_) => false, // unparseable peer_id = stale
                    }
                });
                let removed = before - state.known_peers.len();
                if removed > 0 {
                    info!(removed, remaining = state.known_peers.len(), "known_peers reaper sweep");
                }
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
            }
            // CP-4: Batch flush ticker — drain time-expired batches.
            _ = batch_ticker.tick() => {
                if state.batcher.has_pending() {
                    let flushed = state.batcher.flush_expired();
                    for batch in flushed {
                        dispatch_flushed_batch(
                            batch,
                            &state.ipc_bridge,
                            &mut state.batch_pending,
                            &state.ipc_response_tx,
                        );
                    }
                }
            }
            // B3: Ring session timeout watchdog.
            _ = ring_timeout_ticker.tick() => {
                let timed_out = state.ring_manager.check_timeouts();
                for (session_id, reason) in timed_out {
                    warn!(%session_id, %reason, "ring: session timed out, aborting");
                    // Audit F7: notify the caller with an error token before
                    // tearing down, instead of silently closing the channel.
                    state.ring_manager.fail_session(&session_id, &reason);
                }
            }
            // F6: B5 periodic relay-renewal removed (see comment at ticker
            // decl). Reservation renewal is handled by libp2p auto-renewal +
            // the reactive ExpiredListenAddr / ListenerClosed handlers.
            // F3: Drive any due relay-dial retries (scheduled with backoff).
            _ = relay_retry_ticker.tick() => {
                let now = tokio::time::Instant::now();
                let due: Vec<PeerId> = state
                    .relay_dial_retries
                    .iter()
                    .filter(|(_, (_, next_at))| now >= *next_at)
                    .map(|(pid, _)| *pid)
                    .collect();
                for pid in due {
                    drive_relay_retry(&mut swarm, &mut state, pid);
                }
                // F-5: drive any due relay-RESERVATION retries on the same
                // ticker (re-listen on circuits whose reservation was lost).
                drive_reservation_retries(&mut swarm, &mut state);
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
                    state.known_peers.insert(signed_record.peer_id.clone(), signed_record.clone());
                    info!(
                        model_id = %signed_record.model_id,
                        peer_id = %signed_record.peer_id,
                        "announced to kademlia (provider + record)"
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

/// Handle a discover command: find providers for the model via Kademlia
/// provider API (Task 2.1: Option C).
fn handle_discover(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    model_id: &str,
    reply: oneshot::Sender<Result<Vec<DiscoveredPeer>, String>>,
    state: &mut LoopState,
) {
    let key = dht::model_provider_key(model_id);
    let query_id = swarm.behaviour_mut().kademlia.get_providers(key);
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
/// F3: advance the relay-dial retry state machine for one peer.
///
/// Called both from `OutgoingConnectionError` (on failure) and from the
/// `relay_retry_ticker` (to drive scheduled attempts). A retry is only
/// dialed when `now >= next_attempt_at`, so failures that arrive faster than
/// the backoff window are coalesced into a single spaced attempt instead of
/// burning all five retries in milliseconds. After 5 spaced attempts
/// (~500ms,1s,2s,4s,8s) the peer is evicted and its ring sessions aborted.
fn drive_relay_retry(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    pid: PeerId,
) {
    if swarm.is_connected(&pid) {
        state.relay_dial_retries.remove(&pid);
        return;
    }
    let pid_str = pid.to_string();

    // No active ring session → existing behaviour: evict immediately.
    if !state.ring_manager.peer_has_active_session(&pid_str) {
        state.relay_dial_retries.remove(&pid);
        swarm.behaviour_mut().kademlia.remove_peer(&pid);
        state.known_peers.retain(|_, r| r.libp2p_peer_id != pid_str);
        info!(%pid, "evicted unreachable peer after dial failure");
        return;
    }

    let now = tokio::time::Instant::now();
    let entry = state
        .relay_dial_retries
        .entry(pid)
        .or_insert((0, now));
    // Not yet due — a scheduled attempt is pending; ignore this trigger.
    if now < entry.1 {
        return;
    }
    entry.0 += 1;

    if entry.0 <= 5 {
        let backoff_ms = 500 * (1u64 << (entry.0 - 1).min(4));
        entry.1 = now + std::time::Duration::from_millis(backoff_ms);
        info!(%pid, attempt = entry.0, backoff_ms, "relay_reconnect: dialing via relay");
        for relay_str in crate::relay::BOOTSTRAP_RELAYS {
            if let Ok(relay_ma) = relay_str.parse::<Multiaddr>() {
                let circuit_addr = relay_ma
                    .with(libp2p::multiaddr::Protocol::P2pCircuit)
                    .with(libp2p::multiaddr::Protocol::P2p(pid));
                if swarm.dial(circuit_addr).is_ok() {
                    break;
                }
            }
        }
    } else {
        warn!(%pid, "relay_reconnect: max retries (5) exceeded, evicting");
        state.relay_dial_retries.remove(&pid);
        swarm.behaviour_mut().kademlia.remove_peer(&pid);
        state.known_peers.retain(|_, r| r.libp2p_peer_id != pid_str);
        // B4 + F7: abort ring sessions for this peer (emits a caller-visible
        // error token per session inside abort_sessions_for_peer).
        let aborted = state.ring_manager.abort_sessions_for_peer(&pid_str);
        for (sid, _) in &aborted {
            warn!(%sid, %pid, "ring: aborted session after relay retry exhaustion");
        }
        info!(%pid, "evicted unreachable peer after relay retries exhausted");
    }
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

/// WS-F F-6: act on a circuit-migration signal from the tensor stream manager.
///
/// Only relay-reached peers have a per-circuit byte cap to migrate around (a
/// direct link has no cap), so we gate on the peer currently being reached via
/// a relay with no direct connection. At **pre-establish** (85% of budget) we
/// dial the peer through the bootstrap relays so an alternate circuit is already
/// open; when the capped circuit closes, libp2p's connection fallback +
/// tensor_stream's open-on-demand route the next send over the warm alternate
/// (the cutover), and the ring retries the in-flight token. The ring SESSION
/// (KV + ring state) is independent of the transport, so it survives the swap.
fn drive_circuit_migration(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    peer: PeerId,
    action: crate::relay::MigrationAction,
) {
    let relay_reached = state
        .peer_connections
        .get(&peer)
        .map_or(false, |info| info.tcp_relay > 0 && !info.has_direct());
    if !relay_reached {
        return; // direct link → no byte cap → nothing to migrate around
    }
    match action {
        crate::relay::MigrationAction::PreEstablish => {
            let mut dialed = 0u32;
            for relay_str in crate::relay::BOOTSTRAP_RELAYS {
                if let Ok(relay_ma) = relay_str.parse::<Multiaddr>() {
                    let circuit = relay_ma
                        .with(libp2p::multiaddr::Protocol::P2pCircuit)
                        .with(libp2p::multiaddr::Protocol::P2p(peer));
                    if swarm.dial(circuit).is_ok() {
                        dialed += 1;
                    }
                }
            }
            info!(%peer, dialed, "F-6 migration: pre-established alternate relay circuits (circuit at 85% budget)");
        }
        crate::relay::MigrationAction::Cutover => {
            // No forced stream surgery: when the capped circuit closes, libp2p
            // falls back onto a pre-established circuit and tensor_stream
            // re-opens there; the ring retries the token that was in flight.
            debug!(%peer, "F-6 migration: circuit at cutover threshold — relying on fallback to a pre-established circuit");
        }
        crate::relay::MigrationAction::Continue => {}
    }
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

        // ── AutoNAT ──
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Autonat(autonat_event)) => {
            handle_autonat_event(autonat_event, state);
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
                if !info.observed_addr.to_string().is_empty()
                    && (state.ipv6_capable || !is_ipv6_multiaddr(&info.observed_addr))
                {
                    // F-9: don't advertise a v6 external addr we can't serve.
                    debug!(addr = %info.observed_addr, "adding observed addr as external");
                    swarm.add_external_address(info.observed_addr);
                }

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

                    // Auto QUIC hole-punch: if we don't have a QUIC-direct
                    // connection to this peer, dial their QUIC IPv6 addresses.
                    let has_quic = state.peer_connections.get(&peer_id)
                        .map_or(false, |info| info.quic_direct_v4 > 0 || info.quic_direct_v6 > 0);
                    if has_quic {
                        // Already QUIC-direct — reset the back-off counter.
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
                        // F7: back off after MAX_QUIC_HOLEPUNCH_ATTEMPTS failures so a
                        // UDP-filtered path stops re-dialing (and re-logging) every
                        // identify cycle.
                        const MAX_QUIC_HOLEPUNCH_ATTEMPTS: u32 = 3;
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
                                    Ok(()) => debug!(%peer_id, %ma, "auto_quic_holepunch_dial"),
                                    Err(e) => debug!(%peer_id, %ma, %e, "auto_quic_holepunch_dial_failed"),
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
                    let info = state.peer_connections.entry(peer).or_default();
                    info.tcp_direct = info.tcp_direct.max(1);
                    state.dcutr_event_queue.push_back(peer.to_string());
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
                if let Ok(parsed) = serde_json::from_slice::<serde_json::Value>(&message.data) {
                    if parsed.get("type").and_then(|v| v.as_str()) == Some("PEER_DEPARTED") {
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

        // Ping keepalive — log failures and evict unreachable peers
        // (Phase 1.6). Success is silent to avoid flooding logs.
        SwarmEvent::Behaviour(OpenHydraBehaviourEvent::Ping(ping_event)) => {
            // WS-H: feed the observed round-trip time to the tensor-stream
            // manager so its adaptive write timeout reflects real path latency
            // (previously `update_rtt` was never called → every peer used the
            // default ceiling, which silently dropped large cross-ISP writes).
            if let Ok(rtt) = ping_event.result {
                if let Some(mgr) = state.tensor_mgr.as_ref() {
                    mgr.update_rtt(&ping_event.peer, rtt);
                }
            }
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
            if is_direct_listen_candidate(&address)
                && (state.ipv6_capable || !is_ipv6_multiaddr(&address))
            {
                // F-9: don't advertise a v6 external addr we can't serve.
                debug!(%address, "registering direct listen addr as external");
                swarm.add_external_address(address.clone());
            }
            // F-5: a circuit listen addr coming up means the reservation
            // succeeded — clear any pending backoff retry for it.
            if address.to_string().contains("/p2p-circuit") {
                clear_reservation_retry(state, &address.to_string());
            }
        }
        SwarmEvent::ConnectionEstablished { peer_id, connection_id, endpoint, .. } => {
            // B2: Clear relay dial retry state on successful reconnection.
            state.relay_dial_retries.remove(&peer_id);

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
                        info!(%peer_id, %addr_str, v4=info.quic_direct_v4, "quic_direct_v4_connected");
                    }
                    // Fix 4: proactively warm tensor stream on first QUIC connection of *either* AF.
                    let total_quic = info.quic_direct_v4 + info.quic_direct_v6;
                    if total_quic == 1 {
                        if let Some(ref mgr) = state.tensor_mgr {
                            let mgr = Arc::clone(mgr);
                            let pid = peer_id;
                            tokio::spawn(async move { mgr.warm_stream(&pid).await });
                        }
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
                        debug!(%peer_id, %addr_str, relay=info.tcp_relay, "tcp_relay_connected");
                    }
                }
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
                        }
                    }
                    TransportType::TcpDirect => info.tcp_direct = info.tcp_direct.saturating_sub(1),
                    TransportType::TcpRelay => info.tcp_relay = info.tcp_relay.saturating_sub(1),
                }
                if info.quic_direct_v4 == 0 && info.quic_direct_v6 == 0 && info.tcp_direct == 0 && info.tcp_relay == 0 {
                    state.peer_connections.remove(&peer_id);
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

                // Tensor stream cleanup
                if let Some(ref mgr) = state.tensor_mgr {
                    let mgr = Arc::clone(mgr);
                    let pid = peer_id;
                    tokio::spawn(async move { mgr.remove_peer(&pid).await });
                }

                // B4: Abort ring sessions involving this peer.
                let peer_id_str = peer_id.to_string();
                let aborted = state.ring_manager.abort_sessions_for_peer(&peer_id_str);
                for (session_id, _) in &aborted {
                    warn!(%session_id, %peer_id, "ring: aborted session due to peer disconnect");
                }
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
        // Phase 1.3 + B2: Failed dials with relay-aware retry for active
        // ring sessions, immediate eviction otherwise.
        SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
            warn!(?peer_id, %error, "outgoing_connection_error");
            if let Some(pid) = peer_id {
                if !swarm.is_connected(&pid) {
                    // F3: route through the scheduled-retry helper. The first
                    // failure dials immediately; subsequent failures within
                    // the backoff window are no-ops (the relay_retry_ticker
                    // drives the next spaced attempt), so a fast-failing relay
                    // no longer burns all 5 attempts in milliseconds.
                    drive_relay_retry(swarm, state, pid);
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
            }
        }
        SwarmEvent::ListenerClosed { addresses, reason, .. } => {
            warn!(?addresses, ?reason, "listener_closed");
            for addr in &addresses {
                if addr.to_string().contains("/p2p-circuit") {
                    // F-5: a closed circuit listener means the reservation
                    // dropped/was rejected. Schedule a backed-off retry instead
                    // of an immediate re-listen — an immediate re-listen against
                    // a flapping or capped relay tight-loops the event loop and
                    // spams the relay. The relay_retry_ticker drives the re-listen
                    // once the backoff window elapses.
                    schedule_reservation_retry(state, &addr.to_string());
                }
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
                                        state.known_peers.insert(record.peer_id.clone(), record.clone());
                                        pending.records.push(record);
                                    }
                                }
                            } else {
                                debug!(%provider_pid, "provider found but no data in cache or local store");
                            }
                        }
                    }
                }
                Ok(kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. }) => {
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
                        let _ = pending.reply.send(Err(format!("kademlia get_providers: {e:?}")));
                    }
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
            match result {
                Ok(kad::GetRecordOk::FoundRecord(kad::PeerRecord { record, .. })) => {
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
                        state.nat_info.external_ip = ip.clone();
                        // Classify by address family.
                        if ip.contains(':') {
                            state.nat_info.external_ipv6 = ip;
                        } else {
                            state.nat_info.external_ipv4 = ip;
                        }
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
                    // CP-2: Dispatch via Rust dispatcher.
                    //
                    // The dispatcher inspects the 1-byte method prefix and the
                    // wire format (ForwardMsg OHV2 vs legacy protobuf) and
                    // returns a routing decision. For ForwardMsg messages with
                    // an active IPC bridge, Rust handles the full round-trip
                    // (parse → IPC → response) without Python's proxy handler.
                    // Legacy protobuf and PushResult messages fall through to
                    // SharedProxyQueue for Python handling (until CP-3).
                    let action = state.dispatcher.dispatch(&request.0);
                    match action {
                        DispatchAction::ForwardToWorker(parsed) => {
                            state.inbound_proxy_counter += 1;
                            let req_id = format!("proxy-{}", state.inbound_proxy_counter);

                            if state.ipc_bridge.is_some() {
                                // CP-4: Push into Batcher instead of sending directly to IPC.
                                info!(%peer, id=%req_id,
                                    request_id=%parsed.header.request_id,
                                    stage=%parsed.header.stage_index,
                                    "dispatch: ForwardToWorker → batcher");
                                state.inbound_proxy_channels.insert(req_id.clone(), (channel, std::time::Instant::now(), peer.clone()));

                                // Extract batch key fields before moving the header.
                                let batch_key = BatchKey {
                                    layer_start: parsed.header.shard_layer_start,
                                    activation_dtype: DtypeTag::from(parsed.header.activation_dtype),
                                    is_prefill: !parsed.header.prompt_token_ids.is_empty(),
                                    draft_block: parsed.header.draft_block,
                                };
                                let session_id = parsed.header.kv_session_id.clone();
                                let activation_shape = parsed.header.activation_shape.clone();

                                // Store the full header for batch dispatch.
                                state.batch_pending.insert(req_id.clone(), BatchPendingItem {
                                    header: parsed.header,
                                    needs_response: true,
                                });

                                let item = BatchItem {
                                    request_id: req_id.clone(),
                                    session_id,
                                    activation: parsed.activation,
                                    activation_shape,
                                    enqueued_at: std::time::Instant::now(),
                                };

                                // Size-bound flush: dispatch immediately if batch is full.
                                if let Some(flushed) = state.batcher.add(batch_key, item) {
                                    dispatch_flushed_batch(
                                        flushed,
                                        &state.ipc_bridge,
                                        &mut state.batch_pending,
                                        &state.ipc_response_tx,
                                    );
                                }
                            } else {
                                // No IPC bridge — fall through to SharedProxyQueue.
                                debug!(%peer, id=%req_id,
                                    "dispatch: ForwardMsg but no IPC bridge, fallthrough");
                                proxy_queue.push((req_id.clone(), request.0));
                                state.inbound_proxy_channels.insert(req_id, (channel, std::time::Instant::now(), peer.clone()));
                            }
                        }
                        DispatchAction::ForwardToWorkerAsync { ack, forward } => {
                            // Fire-and-forget: ACK immediately, then push to batcher.
                            if let Err(e) = swarm.behaviour_mut().grpc_proxy
                                .send_response(channel, ProxyResponse(ack))
                            {
                                warn!("dispatch: fire-forget ACK failed: {:?}", e);
                            }
                            if state.ipc_bridge.is_some() {
                                // CP-4: Push into Batcher (async — no response routing needed).
                                state.inbound_proxy_counter += 1;
                                let req_id = format!("proxy-async-{}", state.inbound_proxy_counter);

                                let batch_key = BatchKey {
                                    layer_start: forward.header.shard_layer_start,
                                    activation_dtype: DtypeTag::from(forward.header.activation_dtype),
                                    is_prefill: !forward.header.prompt_token_ids.is_empty(),
                                    draft_block: forward.header.draft_block,
                                };
                                let session_id = forward.header.kv_session_id.clone();
                                let activation_shape = forward.header.activation_shape.clone();

                                state.batch_pending.insert(req_id.clone(), BatchPendingItem {
                                    header: forward.header,
                                    needs_response: false,
                                });

                                let item = BatchItem {
                                    request_id: req_id.clone(),
                                    session_id,
                                    activation: forward.activation,
                                    activation_shape,
                                    enqueued_at: std::time::Instant::now(),
                                };

                                if let Some(flushed) = state.batcher.add(batch_key, item) {
                                    dispatch_flushed_batch(
                                        flushed,
                                        &state.ipc_bridge,
                                        &mut state.batch_pending,
                                        &state.ipc_response_tx,
                                    );
                                }
                            }
                        }
                        DispatchAction::PushResultBlocking(parsed_pr) => {
                            // CP-3: Check if this PushResult belongs to a ring session.
                            let _from_peer = peer.to_string();
                            let ring_action = state.ring_manager.route_push_result(
                                &parsed_pr.header.request_id,
                                &_from_peer,
                                &parsed_pr.header,
                                parsed_pr.activation.clone(),
                            );
                            match ring_action {
                                RingAction::NeedSample { session_id, request_id, activation } => {
                                    // ACK the PushResult immediately so the last peer unblocks.
                                    let _ = swarm.behaviour_mut().grpc_proxy
                                        .send_response(channel, ProxyResponse(Vec::new()));

                                    // Spawn async sampler task.
                                    if let Some(ref bridge) = state.sampler_bridge {
                                        let bridge = bridge.clone();
                                        let tx = state.ring_event_tx.clone();

                                        // Build SampleRequest from ring session config.
                                        let sample_req = build_sample_request(
                                            &state.ring_manager,
                                            &session_id,
                                            &request_id,
                                        );

                                        tokio::spawn(async move {
                                            match bridge.sample(sample_req, activation).await {
                                                Ok((resp, embedding)) => {
                                                    let _ = tx.send(RingEvent::TokenSampled {
                                                        session_id,
                                                        token_id: resp.token_id,
                                                        token_text: resp.token_text,
                                                        is_eos: resp.is_eos,
                                                        embedding,
                                                    });
                                                }
                                                Err(e) => {
                                                    let _ = tx.send(RingEvent::SampleFailed {
                                                        session_id,
                                                        reason: e,
                                                    });
                                                }
                                            }
                                        });
                                    } else {
                                        warn!(
                                            %session_id,
                                            "ring: NeedSample but no SamplerBridge configured"
                                        );
                                    }
                                }
                                RingAction::Complete { session_id, generated_ids } => {
                                    info!(
                                        %session_id,
                                        tokens = generated_ids.len(),
                                        "ring: session complete"
                                    );
                                    state.ring_manager.remove_session(&session_id);
                                    let _ = swarm.behaviour_mut().grpc_proxy
                                        .send_response(channel, ProxyResponse(Vec::new()));
                                }
                                RingAction::Error { session_id, reason } => {
                                    warn!(
                                        %session_id, %reason,
                                        "ring: PushResult error"
                                    );
                                    state.ring_manager.remove_session(&session_id);
                                    let _ = swarm.behaviour_mut().grpc_proxy
                                        .send_response(channel, ProxyResponse(Vec::new()));
                                }
                                RingAction::PrefillChunkReceived {
                                    session_id,
                                    chunk_index,
                                    chunks_received,
                                    chunks_total,
                                } => {
                                    // CP-5: Prefill chunk stored. ACK the sender and wait.
                                    info!(
                                        %session_id, %chunk_index,
                                        %chunks_received, %chunks_total,
                                        "ring: prefill chunk PushResult stored"
                                    );
                                    let _ = swarm.behaviour_mut().grpc_proxy
                                        .send_response(channel, ProxyResponse(Vec::new()));
                                }
                                RingAction::NotRingRequest => {
                                    // Not a ring request — fall through to Python.
                                    state.inbound_proxy_counter += 1;
                                    let req_id = format!("proxy-{}", state.inbound_proxy_counter);
                                    proxy_queue.push((req_id.clone(), request.0));
                                    state.inbound_proxy_channels.insert(req_id, (channel, std::time::Instant::now(), peer.clone()));
                                }
                            }
                        }
                        DispatchAction::PushResultAsync { ack, push_result } => {
                            // Fire-and-forget PushResult: ACK immediately.
                            if let Err(e) = swarm.behaviour_mut().grpc_proxy
                                .send_response(channel, ProxyResponse(ack))
                            {
                                warn!("dispatch: push-result async ACK failed: {:?}", e);
                            }

                            // CP-3: Check ring manager ownership.
                            let _from_peer = peer.to_string();
                            let ring_action = state.ring_manager.route_push_result(
                                &push_result.header.request_id,
                                &_from_peer,
                                &push_result.header,
                                push_result.activation.clone(),
                            );
                            match ring_action {
                                RingAction::NeedSample { session_id, request_id, activation } => {
                                    // Spawn async sampler task (same as blocking path).
                                    if let Some(ref bridge) = state.sampler_bridge {
                                        let bridge = bridge.clone();
                                        let tx = state.ring_event_tx.clone();
                                        let sample_req = build_sample_request(
                                            &state.ring_manager,
                                            &session_id,
                                            &request_id,
                                        );
                                        tokio::spawn(async move {
                                            match bridge.sample(sample_req, activation).await {
                                                Ok((resp, embedding)) => {
                                                    let _ = tx.send(RingEvent::TokenSampled {
                                                        session_id,
                                                        token_id: resp.token_id,
                                                        token_text: resp.token_text,
                                                        is_eos: resp.is_eos,
                                                        embedding,
                                                    });
                                                }
                                                Err(e) => {
                                                    let _ = tx.send(RingEvent::SampleFailed {
                                                        session_id,
                                                        reason: e,
                                                    });
                                                }
                                            }
                                        });
                                    } else {
                                        warn!(%session_id,
                                            "ring: async NeedSample but no SamplerBridge");
                                    }
                                }
                                RingAction::Complete { session_id, generated_ids } => {
                                    info!(%session_id, tokens = generated_ids.len(),
                                        "ring: session complete (async)");
                                    state.ring_manager.remove_session(&session_id);
                                }
                                RingAction::Error { session_id, reason } => {
                                    warn!(%session_id, %reason,
                                        "ring: async PushResult error");
                                    state.ring_manager.remove_session(&session_id);
                                }
                                RingAction::PrefillChunkReceived {
                                    session_id,
                                    chunk_index,
                                    chunks_received,
                                    chunks_total,
                                } => {
                                    // CP-5: Prefill chunk stored (async path). Nothing to do.
                                    info!(
                                        %session_id, %chunk_index,
                                        %chunks_received, %chunks_total,
                                        "ring: async prefill chunk PushResult stored"
                                    );
                                }
                                RingAction::NotRingRequest => {
                                    // Not a ring request — fall through to Python.
                                    state.inbound_proxy_counter += 1;
                                    let req_id = format!("proxy-{}", state.inbound_proxy_counter);
                                    proxy_queue.push((req_id, request.0));
                                    // Channel already consumed by ACK — no need to store.
                                }
                            }
                        }
                        DispatchAction::PingResponse(data) => {
                            // Inline response — no Python round-trip.
                            debug!(%peer, "dispatch: inline ping response");
                            let _ = swarm.behaviour_mut().grpc_proxy
                                .send_response(channel, ProxyResponse(data));
                        }
                        DispatchAction::StatusResponse(data) => {
                            debug!(%peer, "dispatch: inline status response");
                            let _ = swarm.behaviour_mut().grpc_proxy
                                .send_response(channel, ProxyResponse(data));
                        }
                        DispatchAction::LegacyFallthrough => {
                            // Legacy protobuf — original SharedProxyQueue path.
                            state.inbound_proxy_counter += 1;
                            let req_id = format!("proxy-{}", state.inbound_proxy_counter);
                            info!(%peer, bytes = request.0.len(), id = %req_id,
                                "proxy request queued for Python (legacy)");
                            proxy_queue.push((req_id.clone(), request.0));
                            state.inbound_proxy_channels.insert(req_id, (channel, std::time::Instant::now(), peer.clone()));
                        }
                        DispatchAction::UnsupportedMethod { response, reason } => {
                            warn!(%peer, %reason, "dispatch: unsupported method");
                            let _ = swarm.behaviour_mut().grpc_proxy
                                .send_response(channel, ProxyResponse(response));
                        }
                        DispatchAction::ParseError(reason) => {
                            warn!(%peer, %reason, "dispatch: parse error");
                            let _ = swarm.behaviour_mut().grpc_proxy
                                .send_response(channel, ProxyResponse(Vec::new()));
                        }
                    }
                }
                request_response::Message::Response { request_id, response } => {
                    // CP-5: Check for Stage 0 ACK on a prefill chunk injection.
                    if let Some(session_id) = state.prefill_stage0_acks.remove(&request_id) {
                        // Stage 0 has processed the chunk and forwarded to Stage 1.
                        // Check if there's another chunk to inject.
                        if let Some(chunk_info) = state.ring_manager.prefill_next_chunk(&session_id) {
                            inject_prefill_chunk(
                                swarm, state, &session_id, chunk_info,
                            );
                        } else {
                            info!(
                                %session_id,
                                "ring: all prefill chunks injected, awaiting PushResults"
                            );
                        }
                    } else if let Some(reply) = state.pending_proxy.remove(&request_id) {
                        // Outbound response received — deliver to waiting proxy forward.
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
            // Audit F10: if this was a prefill chunk send, its ack mapping
            // would otherwise leak and the prefill pipeline would stall until
            // the watchdog. Clean it up and abort the session now.
            if let Some(session_id) = state.prefill_stage0_acks.remove(&request_id) {
                warn!(%session_id, "ring: prefill chunk send failed, aborting session");
                state.ring_manager.fail_session(
                    &session_id,
                    "prefill chunk send failed",
                );
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
        proxy_queue.push((req_id.clone(), data));
        state.local_proxy_replies.insert(req_id, reply);
        return;
    }

    // Blocking proxy_forward always uses request-response: it's proven
    // reliable over both direct and relay connections. Tensor stream is
    // only used in the fire-and-forget path (proxy_forward_no_wait / push mode).
    if swarm.is_connected(&peer_id) {
        // C3: Log transport type at dispatch time.
        let transport = state.peer_connections.get(&peer_id)
            .map(|info| if info.has_direct() { "direct" } else { "relay" })
            .unwrap_or("unknown");
        debug!(%peer_id, %transport, bytes = data.len(), "proxy_forward");
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
        let circuit_addrs = relay_circuit_addrs(peer_id, state.ipv6_capable);
        match swarm.dial(relay_dial_opts(peer_id, circuit_addrs)) {
            Ok(()) => {
                state.pending_relay_forwards.push((peer_id, data, reply));
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
        proxy_queue.push((req_id, data));
        return;
    }

    // C3: Log transport type at dispatch time.
    let transport = state.peer_connections.get(&peer_id)
        .map(|info| if info.has_direct() { "direct" } else { "relay" })
        .unwrap_or("unknown");

    // Fix 1: prefer tensor stream (fire-and-forget) for connected peers.
    if let Some(ref mgr) = state.tensor_mgr {
        if swarm.is_connected(&peer_id) {
            debug!(%peer_id, %transport, bytes = data.len(), "proxy_forward_no_wait via tensor_stream");
            let mgr = Arc::clone(mgr);
            let pid = peer_id;
            tokio::spawn(async move {
                if let Err(e) = mgr.send_tensor(&pid, &data).await {
                    warn!(%pid, %e, "tensor_stream_no_wait_failed");
                }
            });
            return;
        }
    }

    // Fallback: request-response.
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
        match swarm.dial(relay_dial_opts(peer_id, relay_circuit_addrs(peer_id, state.ipv6_capable))) {
            Ok(()) => {
                let (dummy_tx, _dummy_rx) = oneshot::channel();
                state.pending_relay_forwards.push((peer_id, data, dummy_tx));
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

/// Build a `SampleRequest` from the ring session config.
fn build_sample_request(
    ring_manager: &RingManager,
    session_id: &str,
    request_id: &str,
) -> SampleRequest {
    // Access session config for decode params. If the session disappeared
    // (race with abort), fall back to greedy defaults.
    if let Some(config) = ring_manager.session_config(session_id) {
        SampleRequest {
            session_id: session_id.to_string(),
            request_id: request_id.to_string(),
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            seed: config.seed,
        }
    } else {
        SampleRequest {
            session_id: session_id.to_string(),
            request_id: request_id.to_string(),
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            seed: None,
        }
    }
}

/// CP-5: Inject a prefill chunk into Stage 0 and set up ACK tracking.
///
/// Constructs a ForwardMsg for the chunk, sends it to Stage 0 via blocking
/// `send_request`, and stores the outbound request_id for Stage 0 ACK routing.
/// Also registers the application-level request_id with the ring manager
/// for PushResult routing from the last stage.
fn inject_prefill_chunk(
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
    session_id: &str,
    chunk_info: crate::ring::PrefillInjectInfo,
) {
    let inject_info = match state.ring_manager.build_inject_info(session_id) {
        Some(info) => info,
        None => {
            warn!(%session_id, "ring: session vanished before prefill chunk inject");
            return;
        }
    };

    // Generate a unique request_id for this chunk's ring traversal.
    // Audit F1: append 64 random bits so the id is unguessable.
    state.inbound_proxy_counter += 1;
    let chunk_request_id = format!(
        "ring-{}-pf{}-c{}-{:016x}",
        session_id,
        state.inbound_proxy_counter,
        chunk_info.chunk_index,
        {
            use rand::Rng;
            rand::thread_rng().gen::<u64>()
        },
    );

    // Register for PushResult routing (last stage → ring manager).
    state.ring_manager.register_prefill_request(
        chunk_request_id.clone(),
        session_id.to_string(),
        chunk_info.chunk_index,
    );

    // Build the ForwardMsg header for this chunk.
    let header = crate::ipc_codec::IpcForwardHeader {
        request_id: chunk_request_id.clone(),
        stage_index: 0,
        total_stages: inject_info.total_stages,
        push_mode: true,
        next_hop_peer_id: inject_info.stage0_peer_id.clone(),
        shard_layer_start: inject_info.stage0_layer_start,
        shard_layer_end: inject_info.stage0_layer_end,
        shard_total_layers: inject_info.stage0_total_layers,
        kv_session_id: session_id.to_string(),
        kv_store_activation: true,
        activation_dtype: crate::ipc_codec::ActivationDtype::Fp32,
        activation_shape: chunk_info.shape,
        ring_mode: true,
        ring_tokens_remaining: inject_info.tokens_remaining,
        ring_eos_ids: inject_info.eos_ids.iter().map(|&id| id as i64).collect(),
        ring_generated_ids: inject_info.generated_ids.iter().map(|&id| id as i64).collect(),
        remaining_route: inject_info.remaining_route.clone(),
        final_callback_libp2p_peer_id: inject_info.callback_libp2p_peer_id.clone(),
        prompt_token_ids: chunk_info.prompt_token_ids,
        ..Default::default()
    };

    // Encode as ForwardMsg wire format.
    let wire = match crate::forward_msg::encode(
        crate::forward_msg::MsgType::Forward,
        &header,
        &chunk_info.activation,
    ) {
        Ok(w) => w,
        Err(e) => {
            warn!(
                %session_id, %e,
                chunk = chunk_info.chunk_index,
                "ring: prefill chunk encode failed"
            );
            state.ring_manager.remove_session(session_id);
            return;
        }
    };

    // Use METHOD_FORWARD (blocking) so we get a response = Stage 0 ACK.
    let mut data = vec![crate::dispatcher::METHOD_FORWARD];
    data.extend(wire);

    // Send to Stage 0 via blocking request-response.
    let stage0_peer = &inject_info.stage0_peer_id;
    match stage0_peer.parse::<PeerId>() {
        Ok(pid) => {
            if swarm.is_connected(&pid) {
                let outbound_id = swarm
                    .behaviour_mut()
                    .grpc_proxy
                    .send_request(&pid, ProxyRequest(data));

                // Track for Stage 0 ACK → next chunk injection.
                state
                    .prefill_stage0_acks
                    .insert(outbound_id, session_id.to_string());

                info!(
                    %session_id,
                    chunk = chunk_info.chunk_index,
                    total = chunk_info.total_chunks,
                    %pid,
                    "ring: injected prefill chunk to stage 0"
                );
            } else {
                warn!(
                    %session_id, %stage0_peer,
                    "ring: stage 0 disconnected during prefill, aborting"
                );
                state.ring_manager.remove_session(session_id);
            }
        }
        Err(e) => {
            warn!(
                %session_id, %stage0_peer, %e,
                "ring: invalid stage 0 peer_id during prefill"
            );
            state.ring_manager.remove_session(session_id);
        }
    }
}

/// Handle a RingEvent from an async sampler task.
///
/// On `TokenSampled`: records the token, emits it to the caller, and
/// re-injects the next-token embedding into the ring via ProxyForward
/// to stage 0.
///
/// On `SampleFailed`: aborts the session and logs the error.
fn handle_ring_event(
    event: RingEvent,
    swarm: &mut libp2p::Swarm<OpenHydraBehaviour>,
    state: &mut LoopState,
) {
    match event {
        RingEvent::TokenSampled {
            session_id,
            token_id,
            token_text,
            is_eos,
            embedding,
        } => {
            // 1. Record the token and check completion.
            let done = state.ring_manager.record_token(
                &session_id,
                token_id,
                token_text.clone(),
                is_eos,
            );

            if done {
                info!(
                    %session_id, %token_id, %is_eos,
                    "ring: session complete after token"
                );
                state.ring_manager.remove_session(&session_id);
                return;
            }

            // 2. Re-inject: build ForwardMsg with the embedding and send
            //    to stage 0 (first peer in the ring route).
            let inject_info = state.ring_manager.build_inject_info(&session_id);
            let inject_info = match inject_info {
                Some(info) => info,
                None => {
                    warn!(%session_id, "ring: session vanished before re-inject");
                    return;
                }
            };

            // Generate a unique request_id for this ring pass. Audit F1:
            // append 64 random bits so the id is unguessable — the
            // session/counter prefix is kept only for log correlation.
            state.inbound_proxy_counter += 1;
            let new_request_id = format!(
                "ring-{}-t{}-{:016x}",
                session_id,
                state.inbound_proxy_counter,
                {
                    use rand::Rng;
                    rand::thread_rng().gen::<u64>()
                },
            );

            // Register the new request_id so the returning PushResult
            // is routed back to this session.
            state.ring_manager.register_request(
                new_request_id.clone(),
                session_id.clone(),
            );

            // Build the ForwardMsg header for re-injection.
            let header = crate::ipc_codec::IpcForwardHeader {
                request_id: new_request_id.clone(),
                stage_index: 0,
                total_stages: inject_info.total_stages,
                push_mode: true,
                next_hop_peer_id: inject_info.stage0_peer_id.clone(),
                shard_layer_start: inject_info.stage0_layer_start,
                shard_layer_end: inject_info.stage0_layer_end,
                shard_total_layers: inject_info.stage0_total_layers,
                kv_session_id: session_id.clone(),
                kv_store_activation: true,
                activation_dtype: crate::ipc_codec::ActivationDtype::Fp32,
                activation_shape: vec![
                    1, 1, (embedding.len() / 4) as u32,
                ],
                ring_mode: true,
                ring_tokens_remaining: inject_info.tokens_remaining,
                ring_eos_ids: inject_info.eos_ids.iter().map(|&id| id as i64).collect(),
                ring_generated_ids: inject_info.generated_ids.iter().map(|&id| id as i64).collect(),
                remaining_route: inject_info.remaining_route.clone(),
                final_callback_libp2p_peer_id: inject_info.callback_libp2p_peer_id.clone(),
                ..Default::default()
            };

            // Encode as ForwardMsg wire format with method prefix.
            let wire = match crate::forward_msg::encode(
                crate::forward_msg::MsgType::Forward,
                &header,
                &embedding,
            ) {
                Ok(w) => w,
                Err(e) => {
                    warn!(%session_id, %e, "ring: re-inject encode failed");
                    state.ring_manager.remove_session(&session_id);
                    return;
                }
            };

            // Prepend the method prefix for fire-and-forget forward.
            let mut data = vec![crate::dispatcher::METHOD_FIRE_FORGET];
            data.extend(wire);

            // Send to stage 0 peer via ProxyForwardNoWait.
            let stage0_peer = &inject_info.stage0_peer_id;
            match stage0_peer.parse::<PeerId>() {
                Ok(pid) => {
                    if swarm.is_connected(&pid) {
                        // Hot-path: prefer tensor stream (fire-and-forget)
                        // over request-response (new substream per token).
                        if let Some(ref mgr) = state.tensor_mgr {
                            let mgr = Arc::clone(mgr);
                            let pid_owned = pid;
                            let sid = session_id.clone();
                            let _rid = new_request_id.clone();
                            // Audit F11: report send failure back to the event
                            // loop so the session is aborted immediately
                            // rather than stalling until the watchdog fires.
                            let fail_tx = state.ring_event_tx.clone();
                            tokio::spawn(async move {
                                if let Err(e) = mgr.send_tensor(&pid_owned, &data).await {
                                    warn!(%pid_owned, %e, "ring: tensor_stream re-inject failed");
                                    let _ = fail_tx.send(RingEvent::ReinjectFailed {
                                        session_id: sid,
                                        reason: format!("re-inject send failed: {e}"),
                                    });
                                }
                            });
                            info!(
                                %session_id, %new_request_id, %pid,
                                "ring: re-injected embedding via tensor_stream"
                            );
                        } else {
                            // Fallback to request-response
                            let _req_id = swarm
                                .behaviour_mut()
                                .grpc_proxy
                                .send_request(&pid, ProxyRequest(data));
                            info!(
                                %session_id, %new_request_id, %pid,
                                "ring: re-injected embedding via request_response (fallback)"
                            );
                        }
                    } else {
                        warn!(
                            %session_id, %stage0_peer,
                            "ring: stage 0 peer disconnected, aborting"
                        );
                        state.ring_manager.remove_session(&session_id);
                    }
                }
                Err(e) => {
                    warn!(
                        %session_id, %stage0_peer, %e,
                        "ring: invalid stage 0 peer_id"
                    );
                    state.ring_manager.remove_session(&session_id);
                }
            }
        }
        RingEvent::SampleFailed { session_id, reason } => {
            warn!(%session_id, %reason, "ring: HeadSampler failed, aborting session");
            state.ring_manager.fail_session(&session_id, &reason);
        }
        RingEvent::ReinjectFailed { session_id, reason } => {
            // Audit F11: abort immediately with an error token instead of
            // waiting for the watchdog.
            warn!(%session_id, %reason, "ring: re-inject failed, aborting session");
            state.ring_manager.fail_session(&session_id, &reason);
        }
    }
}

// ── CP-4: Batch dispatch ──────────────────────────────────────────────

/// Dispatch a flushed batch to the IPC bridge.
///
/// For single-item batches, delegates to the proven `bridge.forward()` path.
/// For multi-item batches, uses `bridge.forward_batch()` with the batch wire
/// format.  Response routing uses the `proxy_req_id` stored in each BatchItem
/// to map back to the waiting libp2p response channel.
fn dispatch_flushed_batch(
    batch: FlushedBatch,
    ipc_bridge: &Option<IpcBridge>,
    batch_pending: &mut HashMap<String, BatchPendingItem>,
    ipc_response_tx: &mpsc::UnboundedSender<(String, Vec<u8>)>,
) {
    let bridge = match ipc_bridge {
        Some(ref b) => b.clone(),
        None => return,
    };

    // Collect items with their stored headers.
    let mut dispatch_items: Vec<(String, IpcForwardHeader, Vec<u8>, bool)> = Vec::new();
    for item in batch.items {
        if let Some(pending) = batch_pending.remove(&item.request_id) {
            dispatch_items.push((
                item.request_id,
                pending.header,
                item.activation,
                pending.needs_response,
            ));
        } else {
            warn!(
                req_id = %item.request_id,
                "batch dispatch: no pending header found"
            );
        }
    }

    if dispatch_items.is_empty() {
        return;
    }

    let tx = ipc_response_tx.clone();
    let batch_size = dispatch_items.len();

    if batch_size == 1 {
        // Single item: use the proven single-request IPC path.
        let (req_id, header, activation, needs_response) =
            dispatch_items.into_iter().next().unwrap();
        tokio::spawn(async move {
            let data = match bridge.forward(header, activation).await {
                Ok(resp) => encode_ipc_response_wire(&req_id, &resp),
                Err(e) => {
                    warn!(%req_id, %e, "batch: IPC forward failed");
                    Vec::new()
                }
            };
            if needs_response {
                let _ = tx.send((req_id, data));
            }
        });
    } else {
        // Multi-item batch: use the batch wire format.
        info!(batch_size, reason = ?batch.reason, "dispatching batch to IPC");
        tokio::spawn(async move {
            let ipc_items: Vec<(IpcForwardHeader, Vec<u8>)> = dispatch_items
                .iter()
                .map(|(_, h, a, _)| (h.clone(), a.clone()))
                .collect();

            match bridge.forward_batch(ipc_items).await {
                Ok(responses) => {
                    let n_resp = responses.len();
                    for (i, resp) in responses.into_iter().enumerate() {
                        if i < dispatch_items.len() {
                            let (ref req_id, _, _, needs_response) = dispatch_items[i];
                            if needs_response {
                                let data = encode_ipc_response_wire(req_id, &resp);
                                let _ = tx.send((req_id.clone(), data));
                            }
                        }
                    }
                    // F15: if Python returned FEWER responses than items, the
                    // unmatched callers (index >= n_resp) would never get a
                    // reply — leaking their inbound_proxy_channel and hanging
                    // the remote peer until its request-response timeout. Send
                    // empty replies to the tail so they fail fast.
                    if n_resp < dispatch_items.len() {
                        warn!(
                            got = n_resp, expected = dispatch_items.len(),
                            "batch: fewer responses than items — empty-replying the remainder"
                        );
                        for (ref req_id, _, _, needs_response) in &dispatch_items[n_resp..] {
                            if *needs_response {
                                let _ = tx.send((req_id.clone(), Vec::new()));
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(%e, batch_size, "batch: IPC forward_batch failed");
                    // Send empty responses so callers don't hang.
                    for (ref req_id, _, _, needs_response) in &dispatch_items {
                        if *needs_response {
                            let _ = tx.send((req_id.clone(), Vec::new()));
                        }
                    }
                }
            }
        });
    }
}

/// Encode a single IPC response into the ForwardMsg wire format
/// with the method prefix byte.
fn encode_ipc_response_wire(
    req_id: &str,
    resp: &crate::ipc::IpcResponse,
) -> Vec<u8> {
    match forward_msg::encode_response(&resp.header, &resp.activation) {
        Ok(wire) => {
            let mut buf = vec![dispatcher::METHOD_FORWARD];
            buf.extend(wire);
            buf
        }
        Err(e) => {
            warn!(%req_id, %e, "batch: encode response failed");
            Vec::new()
        }
    }
}

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
            q.push((format!("req-{i}"), vec![0u8; 4]));
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
}
