//! CP-3: Rust Ring Manager — autoregressive token circulation.
//!
//! Moves the ring orchestration loop from Python (`coordinator/chain.py::run_push_ring`)
//! to Rust. The coordinator becomes a pure sampler: receive hidden state from the
//! last peer → sample next token → re-inject embedding into the ring.
//!
//! Architecture:
//! ```text
//! RingManager (tokio task)
//!   ├─ Sends ForwardMsg to stage 0 (via swarm request_response)
//!   ├─ Receives PushResult from last peer (via mpsc from dispatcher)
//!   ├─ Sends activation to HeadSampler (via SamplerBridge)
//!   ├─ Receives (token_id, embedding) from HeadSampler
//!   └─ Emits RingToken to Python (via mpsc token channel)
//! ```
//!
//! Shard-aware failure model: each peer holds layers `[start, end)` of a
//! sequential transformer. Skipping a peer produces garbage, not degraded
//! output. The only valid failure response is to abort the ring session.

use std::collections::{HashMap, HashSet};

use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::ipc_codec::IpcResponseHeader;

// ── Ring action (returned by handle_push_result) ─────────────────────

/// Routing decision from the ring manager after receiving a PushResult.
///
/// The event loop pattern-matches on this to decide what to do next:
/// sample the next token, complete the session, or fall through.
#[derive(Debug)]
pub enum RingAction {
    /// Activation needs head-sampling → token → re-inject or complete.
    ///
    /// The event loop should send `activation` to the HeadSampler via
    /// SamplerBridge, then call `record_token()` with the result.
    NeedSample {
        session_id: String,
        request_id: String,
        activation: Vec<u8>,
    },

    /// The ring session has completed (EOS or max_tokens reached).
    /// The event loop should clean up and notify the caller.
    Complete {
        session_id: String,
        generated_ids: Vec<u32>,
    },

    /// The request_id is not associated with any ring session.
    /// The event loop should fall through to SharedProxyQueue / Python.
    NotRingRequest,

    /// An error occurred during PushResult processing.
    Error {
        session_id: String,
        reason: String,
    },

    /// CP-5: A prefill chunk's PushResult was received and stored.
    /// Not all chunks are in yet — the event loop should continue waiting.
    PrefillChunkReceived {
        session_id: String,
        chunk_index: usize,
        chunks_received: usize,
        chunks_total: usize,
    },
}

// ── Ring configuration ────────────────────────────────────────────────

/// Configuration for a ring inference session.
#[derive(Debug, Clone)]
pub struct RingConfig {
    /// Unique session identifier.
    pub session_id: String,
    /// Request ID from the originating HTTP API request.
    pub request_id: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Pipeline slot index (for pipeline_depth > 1).
    pub slot_id: u32,
    /// Ordered list of peers in the ring, each with their layer range.
    pub route: Vec<RingHop>,
    /// EOS token IDs (generation stops on any of these).
    pub eos_ids: Vec<u32>,
    /// Per-hop timeout in milliseconds.
    pub hop_timeout_ms: u64,
    /// Decoding parameters.
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub seed: Option<u64>,
}

/// A single hop in the ring route.
#[derive(Debug, Clone)]
pub struct RingHop {
    /// libp2p peer ID string.
    pub peer_id: String,
    /// First layer index this peer holds (inclusive).
    pub layer_start: u32,
    /// Last layer index this peer holds (exclusive).
    pub layer_end: u32,
    /// Total layers in the model.
    pub total_layers: u32,
}

/// Layer range for shard tracking.
#[derive(Debug, Clone, Copy)]
pub struct LayerRange {
    pub start: u32,
    pub end: u32,
    pub total: u32,
}

// ── Re-injection info ────────────────────────────────────────────────

/// Information needed to re-inject an embedding into the ring.
///
/// Returned by `RingManager::build_inject_info()` so the event loop
/// can construct the next ForwardMsg without reaching into private
/// session state.
#[derive(Debug, Clone)]
pub struct InjectInfo {
    /// libp2p peer_id of stage 0 (first hop).
    pub stage0_peer_id: String,
    /// Layer range of stage 0.
    pub stage0_layer_start: u32,
    pub stage0_layer_end: u32,
    pub stage0_total_layers: u32,
    /// Total stages in the ring.
    pub total_stages: u32,
    /// Tokens remaining to generate.
    pub tokens_remaining: u32,
    /// Tokens generated so far (for ring_generated_ids header field).
    pub generated_ids: Vec<u32>,
    /// EOS token IDs (for ring_eos_ids header field).
    pub eos_ids: Vec<u32>,
    /// Serialized remaining route (CBOR or JSON, for the header).
    pub remaining_route: Vec<u8>,
    /// The coordinator's libp2p peer_id (for final_callback routing).
    pub callback_libp2p_peer_id: String,
}

// ── CP-5: Prefill Pipelining ──────────────────────────────────────────

/// Default chunk size for prefill pipelining (tokens per chunk).
pub const DEFAULT_PREFILL_CHUNK_SIZE: usize = 128;

/// Minimum sequence length to trigger chunked prefill.
pub const PREFILL_CHUNK_THRESHOLD: usize = 256;

/// Info for injecting a prefill chunk into the ring.
///
/// Returned by `init_prefill_pipeline()` (first chunk) and
/// `prefill_next_chunk()` (subsequent chunks). The event loop uses
/// this to construct the ForwardMsg for the chunk.
#[derive(Debug, Clone)]
pub struct PrefillInjectInfo {
    /// Chunk activation bytes (raw float32).
    pub activation: Vec<u8>,
    /// Chunk activation shape `[1, chunk_tokens, hidden_dim]`.
    pub shape: Vec<u32>,
    /// Prompt token IDs for this chunk (Stage 0 embedding lookup).
    pub prompt_token_ids: Vec<i64>,
    /// Chunk index (0-based).
    pub chunk_index: usize,
    /// Total number of chunks.
    pub total_chunks: usize,
}

/// Chunker for slicing prefill activations along the sequence dimension.
///
/// A prefill activation has shape `[1, seq_len, hidden_dim]`.  When
/// `seq_len > PREFILL_CHUNK_THRESHOLD`, the chunker slices it into
/// `ceil(seq_len / chunk_size)` chunks, each `[1, chunk_tokens, hidden_dim]`.
///
/// This enables pipeline overlap: Chunk 0 enters Stage 1 while Chunk 1
/// enters Stage 0, reducing TTFT for long prompts.
pub struct PrefillChunker;

impl PrefillChunker {
    /// Split a prefill activation `[1, seq_len, hidden_dim]` into chunks.
    ///
    /// Returns a vector of `(chunk_bytes, chunk_shape)` pairs.
    /// If `seq_len <= chunk_size`, returns a single chunk (the original).
    ///
    /// `bytes_per_element` defaults to 4 (fp32) if not specified.
    pub fn chunk(
        activation: &[u8],
        shape: &[u32],
        chunk_size: usize,
        bytes_per_element: usize,
    ) -> Result<Vec<(Vec<u8>, Vec<u32>)>, String> {
        if shape.len() != 3 {
            return Err(format!(
                "expected 3D shape [batch, seq_len, hidden_dim], got {:?}",
                shape
            ));
        }

        let batch = shape[0] as usize;
        let seq_len = shape[1] as usize;
        let hidden_dim = shape[2] as usize;

        if batch != 1 {
            return Err(format!("expected batch=1, got {batch}"));
        }

        let expected_len = batch * seq_len * hidden_dim * bytes_per_element;
        if activation.len() != expected_len {
            return Err(format!(
                "activation length mismatch: expected {expected_len}, got {}",
                activation.len()
            ));
        }

        if seq_len <= chunk_size {
            return Ok(vec![(activation.to_vec(), shape.to_vec())]);
        }

        let mut chunks = Vec::new();
        let stride = hidden_dim * bytes_per_element; // bytes per token
        let mut offset = 0;
        let mut remaining = seq_len;

        while remaining > 0 {
            let chunk_tokens = remaining.min(chunk_size);
            let chunk_bytes = chunk_tokens * stride;
            let chunk_data = activation[offset..offset + chunk_bytes].to_vec();
            let chunk_shape = vec![1, chunk_tokens as u32, hidden_dim as u32];
            chunks.push((chunk_data, chunk_shape));
            offset += chunk_bytes;
            remaining -= chunk_tokens;
        }

        Ok(chunks)
    }

    /// Split prompt token IDs into chunks matching the activation chunks.
    pub fn chunk_prompt_ids(
        prompt_token_ids: &[i64],
        chunk_size: usize,
    ) -> Vec<Vec<i64>> {
        prompt_token_ids
            .chunks(chunk_size)
            .map(|c| c.to_vec())
            .collect()
    }

    /// Check if an activation should be chunked based on seq_len.
    pub fn should_chunk(shape: &[u32], threshold: usize) -> bool {
        shape.len() == 3 && shape[0] == 1 && (shape[1] as usize) > threshold
    }
}

/// A single prefill chunk with its activation data and metadata.
#[derive(Debug, Clone)]
struct PrefillChunk {
    activation: Vec<u8>,
    shape: Vec<u32>,
    prompt_token_ids: Vec<i64>,
}

/// State for an active prefill pipeline session.
///
/// Tracks chunk injection progress (which chunks have been sent to
/// Stage 0) and received PushResult activations (from the last stage)
/// for eventual concatenation before head-sampling.
struct PrefillPipelineState {
    /// Pre-chunked activation data, indexed 0..total_chunks.
    chunks: Vec<PrefillChunk>,
    /// Index of the next chunk to inject into Stage 0.
    next_inject: usize,
    /// Received PushResult activations from the last stage, by chunk index.
    received: Vec<Option<Vec<u8>>>,
    /// Number of chunks fully processed (PushResult received from last stage).
    received_count: usize,
}

impl PrefillPipelineState {
    fn total_chunks(&self) -> usize {
        self.chunks.len()
    }

    fn all_injected(&self) -> bool {
        self.next_inject >= self.chunks.len()
    }

    fn all_received(&self) -> bool {
        self.received_count == self.received.len()
    }

    fn next_chunk(&self) -> Option<&PrefillChunk> {
        self.chunks.get(self.next_inject)
    }

    fn advance_inject(&mut self) {
        self.next_inject += 1;
    }

    /// Store a processed chunk activation and return whether all are received.
    fn store_result(&mut self, chunk_index: usize, activation: Vec<u8>) -> bool {
        if chunk_index < self.received.len() && self.received[chunk_index].is_none() {
            self.received[chunk_index] = Some(activation);
            self.received_count += 1;
        }
        self.all_received()
    }

    /// Concatenate all received chunk activations into a single buffer.
    ///
    /// Called when `all_received()` is true. The chunks are concatenated
    /// in order along the sequence dimension (dim=1), reconstructing the
    /// full `[1, seq_len, hidden_dim]` activation for head-sampling.
    fn concatenate(&self) -> Vec<u8> {
        let total_len: usize = self
            .received
            .iter()
            .filter_map(|r| r.as_ref())
            .map(|r| r.len())
            .sum();
        let mut buf = Vec::with_capacity(total_len);
        for chunk_result in &self.received {
            if let Some(data) = chunk_result {
                buf.extend_from_slice(data);
            }
        }
        buf
    }
}

// ── Ring tokens (output stream) ───────────────────────────────────────

/// A single generated token emitted by the ring manager.
#[derive(Debug, Clone)]
pub struct RingToken {
    /// The generated token ID.
    pub token_id: u32,
    /// Human-readable token text (if available from the sampler).
    pub token_text: String,
    /// Whether this is the end-of-sequence token.
    pub is_eos: bool,
    /// Per-token latency in milliseconds (full ring round-trip).
    pub latency_ms: f64,
    /// Session this token belongs to.
    pub session_id: String,
}

// ── Ring handle (returned to caller) ──────────────────────────────────

/// Handle to a running ring session.
///
/// The caller (Python via PyO3) receives tokens through the `token_rx`
/// channel. Dropping the handle cancels the ring session.
pub struct RingHandle {
    /// Receive generated tokens as they're produced.
    pub token_rx: mpsc::Receiver<RingToken>,
    /// Session ID for correlation.
    pub session_id: String,
}

// ── Ring manager ──────────────────────────────────────────────────────

/// Manages active ring sessions.
///
/// Runs as part of the tokio event loop. PushResult messages from the
/// dispatcher are routed here via `push_result_tx`.
///
/// Key CP-3 addition: `pending_requests` maps `request_id` → `session_id`
/// so the event loop can route inbound PushResult messages (which only
/// carry `request_id` in their IpcResponseHeader) to the correct ring
/// session without consulting Python.
pub struct RingManager {
    /// Active ring sessions, keyed by session_id.
    sessions: HashMap<String, RingSessionState>,
    /// Shard map: which peer holds which layers.
    shard_map: HashMap<String, LayerRange>,
    /// Maps request_id → pending-request record for PushResult routing.
    ///
    /// When the ring manager sends a ForwardMsg to stage 0, it registers
    /// the request_id here. When the last peer's PushResult arrives, the
    /// event loop calls `is_ring_request(request_id)` to check ownership
    /// before routing to the ring manager (vs SharedProxyQueue/Python).
    ///
    /// The record also pins the `expected_peer` — the libp2p peer_id of the
    /// final ring hop that is legitimately allowed to deliver this result —
    /// so `route_push_result` can reject forged callbacks (audit F1).
    pending_requests: HashMap<String, PendingRequest>,
    /// CP-5: Maps request_id → chunk_index for prefill PushResult routing.
    ///
    /// When a prefill chunk is injected, its request_id is registered here
    /// alongside the chunk_index. When the PushResult arrives, we look up
    /// the chunk_index to store the partial activation in the correct slot.
    prefill_request_chunks: HashMap<String, usize>,
}

/// A pending ring request awaiting its PushResult callback.
///
/// `expected_peer` is the libp2p peer_id (canonical string form) of the
/// final hop in the ring route — the only peer permitted to deliver the
/// callback for this request. It is empty only when the route could not be
/// resolved at registration time (defensive; treated as "cannot verify").
struct PendingRequest {
    session_id: String,
    expected_peer: String,
}

/// Internal state for a single ring session.
struct RingSessionState {
    config: RingConfig,
    /// Tokens generated so far.
    generated_ids: Vec<u32>,
    /// Tokens remaining to generate.
    tokens_remaining: u32,
    /// Channel to emit tokens to the caller.
    token_tx: mpsc::Sender<RingToken>,
    /// Timestamp of last ring round-trip start (for latency measurement).
    last_inject_time: std::time::Instant,
    /// CP-5: Active prefill pipeline state. `Some(...)` while chunked
    /// prefill is in progress; `None` during standard decode or if
    /// the prefill was too short to chunk.
    prefill_state: Option<PrefillPipelineState>,
}

impl RingManager {
    /// Create a new ring manager.
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            shard_map: HashMap::new(),
            pending_requests: HashMap::new(),
            prefill_request_chunks: HashMap::new(),
        }
    }

    /// Start a new ring session.
    ///
    /// Returns a `RingHandle` with a token receiver channel.
    /// The ring doesn't start circulating until `inject_first_token()` is called.
    pub fn start_session(&mut self, config: RingConfig) -> RingHandle {
        let (token_tx, token_rx) = mpsc::channel::<RingToken>(64);
        let session_id = config.session_id.clone();

        // Populate shard map from route.
        for hop in &config.route {
            self.shard_map.insert(
                hop.peer_id.clone(),
                LayerRange {
                    start: hop.layer_start,
                    end: hop.layer_end,
                    total: hop.total_layers,
                },
            );
        }

        let state = RingSessionState {
            config,
            generated_ids: Vec::new(),
            tokens_remaining: 0, // set by inject_first_token
            token_tx,
            last_inject_time: std::time::Instant::now(),
            prefill_state: None,
        };

        // tokens_remaining is initialized from config in start
        let remaining = state.config.max_tokens;
        let mut state = state;
        state.tokens_remaining = remaining;

        self.sessions.insert(session_id.clone(), state);

        info!(%session_id, "ring session started");

        RingHandle {
            token_rx,
            session_id,
        }
    }

    /// Handle a PushResult from the last peer in the ring (legacy API).
    ///
    /// **Deprecated** — use `route_push_result()` instead, which looks up
    /// the session via request_id (the only identifier available in the
    /// IpcResponseHeader from the event loop).
    ///
    /// Returns `Some(session_id)` if the session is now complete.
    #[allow(dead_code)]
    pub fn handle_push_result(
        &mut self,
        session_id: &str,
        header: &IpcResponseHeader,
        _activation: &[u8],
    ) -> Option<String> {
        let session = self.sessions.get_mut(session_id)?;
        let _ = header;
        info!(
            %session_id,
            tokens_remaining = session.tokens_remaining,
            generated = session.generated_ids.len(),
            "ring: received PushResult (legacy path)"
        );
        None
    }

    /// Record a generated token for a session.
    ///
    /// Called after HeadSampler returns the token. Emits the token to
    /// the caller and checks for completion.
    pub fn record_token(
        &mut self,
        session_id: &str,
        token_id: u32,
        token_text: String,
        is_eos: bool,
    ) -> bool {
        let session = match self.sessions.get_mut(session_id) {
            Some(s) => s,
            None => {
                warn!(%session_id, "ring: record_token for unknown session");
                return true; // treat as complete
            }
        };

        session.generated_ids.push(token_id);
        if session.tokens_remaining > 0 {
            session.tokens_remaining -= 1;
        }

        let latency_ms = session.last_inject_time.elapsed().as_secs_f64() * 1000.0;
        session.last_inject_time = std::time::Instant::now();

        let token = RingToken {
            token_id,
            token_text,
            is_eos,
            latency_ms,
            session_id: session_id.to_string(),
        };

        // Emit to caller (non-blocking — if the channel is full, the token
        // is buffered by the mpsc channel's capacity of 64).
        if session.token_tx.try_send(token).is_err() {
            warn!(%session_id, "ring: token channel full or closed");
        }

        let complete = is_eos || session.tokens_remaining == 0;
        if complete {
            info!(
                %session_id,
                total_tokens = session.generated_ids.len(),
                "ring session complete"
            );
        }
        complete
    }

    /// Remove a completed or aborted session.
    pub fn remove_session(&mut self, session_id: &str) -> Option<Vec<u32>> {
        self.cleanup_pending_requests(session_id);
        self.sessions
            .remove(session_id)
            .map(|s| s.generated_ids)
    }

    /// Abort a session due to a hop failure.
    ///
    /// Returns the peer_id that failed and the layers that are lost.
    /// Also cleans up any pending request_id → session_id mappings.
    pub fn abort_session(
        &mut self,
        session_id: &str,
        failed_peer: &str,
    ) -> Option<(LayerRange, Vec<u32>)> {
        let layer_range = self.shard_map.get(failed_peer).copied();
        // remove_session already calls cleanup_pending_requests
        let generated = self.remove_session(session_id);

        match (layer_range, generated) {
            (Some(lr), Some(ids)) => {
                warn!(
                    %session_id,
                    %failed_peer,
                    lost_layers_start = lr.start,
                    lost_layers_end = lr.end,
                    tokens_generated = ids.len(),
                    "ring session aborted: peer failure"
                );
                Some((lr, ids))
            }
            _ => None,
        }
    }

    /// Number of active sessions.
    pub fn active_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Check if a session exists.
    pub fn has_session(&self, session_id: &str) -> bool {
        self.sessions.contains_key(session_id)
    }

    /// Check if any active session involves the given peer.
    pub fn peer_has_active_session(&self, peer_id: &str) -> bool {
        self.sessions
            .values()
            .any(|s| s.config.route.iter().any(|hop| hop.peer_id == peer_id))
    }

    /// Abort all sessions involving the given peer.
    /// Returns `Vec<(session_id, generated_token_ids)>`.
    pub fn abort_sessions_for_peer(&mut self, peer_id: &str) -> Vec<(String, Vec<u32>)> {
        let to_abort: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, s)| s.config.route.iter().any(|hop| hop.peer_id == peer_id))
            .map(|(id, _)| id.clone())
            .collect();
        let mut aborted = Vec::new();
        for session_id in to_abort {
            if let Some((_lr, ids)) = self.abort_session(&session_id, peer_id) {
                aborted.push((session_id, ids));
            }
        }
        aborted
    }

    /// Returns `(session_id, reason)` pairs for sessions that exceeded
    /// their hop timeout with no activity.
    pub fn check_timeouts(&self) -> Vec<(String, String)> {
        let now = std::time::Instant::now();
        let mut timed_out = Vec::new();
        for (session_id, session) in &self.sessions {
            // Use hop_timeout_ms with a minimum floor of 30 s.
            let timeout =
                std::time::Duration::from_millis(session.config.hop_timeout_ms.max(30_000));
            if now.duration_since(session.last_inject_time) > timeout {
                timed_out.push((
                    session_id.clone(),
                    format!(
                        "no activity for {:?}",
                        now.duration_since(session.last_inject_time)
                    ),
                ));
            }
        }
        timed_out
    }

    /// Get a reference to a session's config (for building SampleRequest).
    pub fn session_config(&self, session_id: &str) -> Option<&RingConfig> {
        self.sessions.get(session_id).map(|s| &s.config)
    }

    /// Build the information needed to re-inject an embedding into the ring.
    ///
    /// Returns `None` if the session doesn't exist (race with abort).
    pub fn build_inject_info(&self, session_id: &str) -> Option<InjectInfo> {
        let session = self.sessions.get(session_id)?;
        let config = &session.config;

        let stage0 = config.route.first()?;

        // Serialize the full route as CBOR for the remaining_route header field.
        // Each hop is serialized as a list of [peer_id, layer_start, layer_end, total_layers].
        let route_data: Vec<(String, u32, u32, u32)> = config
            .route
            .iter()
            .map(|h| (h.peer_id.clone(), h.layer_start, h.layer_end, h.total_layers))
            .collect();
        let mut remaining_route = Vec::new();
        let _ = ciborium::into_writer(&route_data, &mut remaining_route);

        Some(InjectInfo {
            stage0_peer_id: stage0.peer_id.clone(),
            stage0_layer_start: stage0.layer_start,
            stage0_layer_end: stage0.layer_end,
            stage0_total_layers: stage0.total_layers,
            total_stages: config.route.len() as u32,
            tokens_remaining: session.tokens_remaining,
            generated_ids: session.generated_ids.clone(),
            eos_ids: config.eos_ids.clone(),
            remaining_route,
            callback_libp2p_peer_id: String::new(), // set by event loop (local peer)
        })
    }

    // ── Pre-dial: ensure connections before ring starts ──────────────

    /// Return every unique peer_id from a session's route that needs
    /// to be pre-dialed before the ring starts circulating.
    ///
    /// Cross-ISP topology requires bidirectional libp2p connections
    /// (relay circuits) to be established *before* the first ForwardMsg
    /// is sent. Without pre-dialing, the first hop may fail with
    /// "peer not connected" — especially through CGNAT relays.
    ///
    /// Returns `None` if the session doesn't exist.
    pub fn peers_to_dial(&self, session_id: &str) -> Option<Vec<String>> {
        let session = self.sessions.get(session_id)?;
        let mut seen = HashSet::new();
        let mut peers = Vec::new();
        for hop in &session.config.route {
            if seen.insert(&hop.peer_id) {
                peers.push(hop.peer_id.clone());
            }
        }
        Some(peers)
    }

    /// Return all unique peer_ids from a RingConfig's route (static helper).
    ///
    /// Use this when you need the dial list *before* starting the session
    /// (e.g., to pre-dial, then start the session only if all dials succeed).
    pub fn peers_from_route(config: &RingConfig) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut peers = Vec::new();
        for hop in &config.route {
            if seen.insert(&hop.peer_id) {
                peers.push(hop.peer_id.clone());
            }
        }
        peers
    }

    // ── Request registration: request_id → session_id mapping ───────

    /// Register a request_id → session_id mapping.
    ///
    /// Called by the event loop when it injects a ForwardMsg into the ring
    /// (either the initial embedding or a re-injection after sampling).
    /// The request_id is carried through every hop and returned in the
    /// PushResult from the last peer.
    pub fn register_request(&mut self, request_id: String, session_id: String) {
        // B3: Refresh activity timestamp for timeout watchdog.
        // Audit F1: pin the expected callback peer = final hop of the route.
        let expected_peer = if let Some(s) = self.sessions.get_mut(&session_id) {
            s.last_inject_time = std::time::Instant::now();
            s.config.route.last().map(|h| h.peer_id.clone()).unwrap_or_default()
        } else {
            String::new()
        };
        info!(%request_id, %session_id, %expected_peer, "ring: registered request");
        self.pending_requests.insert(
            request_id,
            PendingRequest { session_id, expected_peer },
        );
    }

    /// Check if a request_id belongs to a ring session.
    ///
    /// Called by the event loop's PushResult handler to decide whether
    /// to route the result to the ring manager or fall through to
    /// SharedProxyQueue for Python handling.
    pub fn is_ring_request(&self, request_id: &str) -> bool {
        self.pending_requests.contains_key(request_id)
    }

    /// Look up the session_id for a given request_id.
    pub fn lookup_session(&self, request_id: &str) -> Option<&str> {
        self.pending_requests
            .get(request_id)
            .map(|p| p.session_id.as_str())
    }

    /// Number of pending request_id → session_id mappings.
    pub fn pending_request_count(&self) -> usize {
        self.pending_requests.len()
    }

    // ── CP-5: Prefill pipeline management ──────────────────────────

    /// Initialize prefill pipelining for a session.
    ///
    /// Chunks the activation along the sequence dimension and stores the
    /// pipeline state in the session. Returns the first chunk to inject.
    ///
    /// If `seq_len <= chunk_size`, returns an error — the caller should
    /// use the standard single-injection path instead.
    pub fn init_prefill_pipeline(
        &mut self,
        session_id: &str,
        activation: Vec<u8>,
        shape: Vec<u32>,
        prompt_token_ids: Vec<i64>,
        chunk_size: usize,
    ) -> Result<PrefillInjectInfo, String> {
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| format!("session {session_id} not found"))?;

        if session.prefill_state.is_some() {
            return Err("prefill pipeline already initialized".into());
        }

        // Chunk the activation.
        let activation_chunks =
            PrefillChunker::chunk(&activation, &shape, chunk_size, 4)?;

        if activation_chunks.len() <= 1 {
            return Err("activation too short to chunk".into());
        }

        // Chunk the prompt token IDs to match.
        let prompt_chunks =
            PrefillChunker::chunk_prompt_ids(&prompt_token_ids, chunk_size);

        // Build internal chunk records.
        let total_chunks = activation_chunks.len();
        let chunks: Vec<PrefillChunk> = activation_chunks
            .into_iter()
            .enumerate()
            .map(|(i, (act, shp))| PrefillChunk {
                activation: act,
                shape: shp,
                prompt_token_ids: prompt_chunks
                    .get(i)
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect();

        // Extract first chunk info before moving into state.
        let first = &chunks[0];
        let first_info = PrefillInjectInfo {
            activation: first.activation.clone(),
            shape: first.shape.clone(),
            prompt_token_ids: first.prompt_token_ids.clone(),
            chunk_index: 0,
            total_chunks,
        };

        session.prefill_state = Some(PrefillPipelineState {
            chunks,
            next_inject: 1, // chunk 0 is returned, so next is 1
            received: (0..total_chunks).map(|_| None).collect(),
            received_count: 0,
        });

        info!(
            %session_id,
            total_chunks,
            chunk_size,
            "ring: prefill pipeline initialized"
        );

        Ok(first_info)
    }

    /// Advance the prefill pipeline and return the next chunk to inject.
    ///
    /// Called when Stage 0 ACKs a chunk (response to `send_request`).
    /// Returns `None` when all chunks have been injected.
    pub fn prefill_next_chunk(
        &mut self,
        session_id: &str,
    ) -> Option<PrefillInjectInfo> {
        let session = self.sessions.get_mut(session_id)?;
        let prefill = session.prefill_state.as_mut()?;

        if prefill.all_injected() {
            return None;
        }

        let chunk = prefill.next_chunk()?;
        let info = PrefillInjectInfo {
            activation: chunk.activation.clone(),
            shape: chunk.shape.clone(),
            prompt_token_ids: chunk.prompt_token_ids.clone(),
            chunk_index: prefill.next_inject,
            total_chunks: prefill.total_chunks(),
        };

        prefill.advance_inject();
        Some(info)
    }

    /// Register a prefill chunk's request_id → (session_id, chunk_index).
    ///
    /// Called by the event loop when it injects a prefill chunk into the
    /// ring. The request_id is carried through every hop and returned in
    /// the PushResult from the last peer.
    pub fn register_prefill_request(
        &mut self,
        request_id: String,
        session_id: String,
        chunk_index: usize,
    ) {
        // B3: Refresh activity timestamp for timeout watchdog.
        // Audit F1: pin the expected callback peer = final hop of the route.
        let expected_peer = if let Some(s) = self.sessions.get_mut(&session_id) {
            s.last_inject_time = std::time::Instant::now();
            s.config.route.last().map(|h| h.peer_id.clone()).unwrap_or_default()
        } else {
            String::new()
        };
        info!(
            %request_id, %session_id, %chunk_index,
            "ring: registered prefill chunk request"
        );
        self.pending_requests.insert(
            request_id.clone(),
            PendingRequest { session_id, expected_peer },
        );
        self.prefill_request_chunks
            .insert(request_id, chunk_index);
    }

    /// Check if a session is in prefill pipeline mode.
    pub fn is_prefill_session(&self, session_id: &str) -> bool {
        self.sessions
            .get(session_id)
            .and_then(|s| s.prefill_state.as_ref())
            .is_some()
    }

    /// Number of pending prefill chunk request mappings.
    pub fn prefill_request_count(&self) -> usize {
        self.prefill_request_chunks.len()
    }

    // ── PushResult handling with RingAction ──────────────────────────

    /// Handle a PushResult from the last peer in the ring.
    ///
    /// Looks up the session via `request_id` from the pending_requests map,
    /// consumes the mapping, and returns a `RingAction` telling the event
    /// loop what to do next.
    ///
    /// This replaces the old `handle_push_result()` that took session_id
    /// directly — the event loop only has `request_id` from the
    /// IpcResponseHeader.
    pub fn route_push_result(
        &mut self,
        request_id: &str,
        from_peer: &str,
        header: &IpcResponseHeader,
        activation: Vec<u8>,
    ) -> RingAction {
        // Audit F1: authenticate the callback BEFORE consuming the mapping.
        // The PushResult must arrive from the final ring hop we registered;
        // any other peer (a route member trying to short-circuit the ring,
        // or an outsider that guessed the request_id) is rejected and the
        // pending mapping is LEFT IN PLACE so the genuine callback can still
        // be honoured. We peek first, then remove only on success.
        match self.pending_requests.get(request_id) {
            None => return RingAction::NotRingRequest,
            Some(pending) => {
                if !pending.expected_peer.is_empty()
                    && pending.expected_peer != from_peer
                {
                    warn!(
                        %request_id, %from_peer,
                        expected = %pending.expected_peer,
                        "ring: rejected push_result from unexpected peer (forgery?)"
                    );
                    // Treat as not-ours: don't consume, don't error the
                    // session — the legitimate peer may still deliver.
                    return RingAction::NotRingRequest;
                }
            }
        }

        // Authenticated — now consume the pending request mapping.
        let session_id = match self.pending_requests.remove(request_id) {
            Some(p) => p.session_id,
            None => return RingAction::NotRingRequest,
        };

        // Check session existence (non-borrowing: just contains_key).
        if !self.sessions.contains_key(&session_id) {
            warn!(
                %session_id, %request_id,
                "ring: push_result for removed session"
            );
            return RingAction::Error {
                session_id,
                reason: "session no longer active".into(),
            };
        }

        // Check for error status in the response header.
        if header.status != crate::ipc_codec::IpcStatus::Ok {
            let reason = header.error_message.clone();
            warn!(
                %session_id, %request_id, ?header.status, %reason,
                "ring: push_result error from peer"
            );
            return RingAction::Error {
                session_id,
                reason: format!("peer returned error: {reason}"),
            };
        }

        // CP-5: Check if this PushResult belongs to a prefill chunk.
        if let Some(chunk_index) = self.prefill_request_chunks.remove(request_id) {
            let session = match self.sessions.get_mut(&session_id) {
                Some(s) => s,
                None => {
                    return RingAction::Error {
                        session_id,
                        reason: "session vanished during prefill".into(),
                    };
                }
            };

            if let Some(ref mut prefill) = session.prefill_state {
                let all_done = prefill.store_result(chunk_index, activation);

                if all_done {
                    // All chunks received — concatenate and transition to decode.
                    let concatenated = prefill.concatenate();
                    let total = prefill.total_chunks();
                    session.prefill_state = None;

                    info!(
                        %session_id,
                        chunks = total,
                        concat_len = concatenated.len(),
                        "ring: prefill complete, all chunks reassembled"
                    );

                    return RingAction::NeedSample {
                        session_id,
                        request_id: request_id.to_string(),
                        activation: concatenated,
                    };
                } else {
                    info!(
                        %session_id, %chunk_index,
                        received = prefill.received_count,
                        total = prefill.total_chunks(),
                        "ring: prefill chunk received, waiting for more"
                    );

                    return RingAction::PrefillChunkReceived {
                        session_id,
                        chunk_index,
                        chunks_received: prefill.received_count,
                        chunks_total: prefill.total_chunks(),
                    };
                }
            }
        }

        // Standard decode path. Take a mutable borrow to read session state
        // for logging (no immutable borrow is live at this point).
        let session = self.sessions.get(&session_id).unwrap();
        info!(
            %session_id, %request_id,
            tokens_remaining = session.tokens_remaining,
            generated = session.generated_ids.len(),
            "ring: received PushResult, need head-sample"
        );

        RingAction::NeedSample {
            session_id,
            request_id: request_id.to_string(),
            activation,
        }
    }

    /// Clean up all pending requests for a session (e.g., on abort).
    fn cleanup_pending_requests(&mut self, session_id: &str) {
        // Collect request_ids being removed so we can also clean
        // prefill_request_chunks (separate HashMap, can't retain both).
        let removing: Vec<String> = self
            .pending_requests
            .iter()
            .filter_map(|(rid, pending)| {
                if pending.session_id == session_id {
                    Some(rid.clone())
                } else {
                    None
                }
            })
            .collect();

        for rid in &removing {
            self.pending_requests.remove(rid);
            self.prefill_request_chunks.remove(rid);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> RingConfig {
        RingConfig {
            session_id: "ring-001".into(),
            request_id: "req-001".into(),
            max_tokens: 10,
            slot_id: 0,
            route: vec![
                RingHop {
                    peer_id: "peer-A".into(),
                    layer_start: 0,
                    layer_end: 16,
                    total_layers: 32,
                },
                RingHop {
                    peer_id: "peer-B".into(),
                    layer_start: 16,
                    layer_end: 32,
                    total_layers: 32,
                },
            ],
            eos_ids: vec![2], // EOS token
            hop_timeout_ms: 500,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            seed: None,
        }
    }

    #[test]
    fn test_start_session() {
        let mut mgr = RingManager::new();
        let handle = mgr.start_session(test_config());

        assert_eq!(handle.session_id, "ring-001");
        assert!(mgr.has_session("ring-001"));
        assert_eq!(mgr.active_sessions(), 1);

        // Shard map populated.
        assert_eq!(mgr.shard_map.get("peer-A").unwrap().start, 0);
        assert_eq!(mgr.shard_map.get("peer-A").unwrap().end, 16);
        assert_eq!(mgr.shard_map.get("peer-B").unwrap().start, 16);
        assert_eq!(mgr.shard_map.get("peer-B").unwrap().end, 32);
    }

    #[test]
    fn test_record_tokens_until_max() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        // Generate 9 tokens — not complete.
        for i in 0..9 {
            let done = mgr.record_token("ring-001", 100 + i, format!("tok{i}"), false);
            assert!(!done);
        }

        // 10th token — complete (max_tokens reached).
        let done = mgr.record_token("ring-001", 109, "tok9".into(), false);
        assert!(done);
    }

    #[test]
    fn test_record_tokens_eos() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        // Generate 2 tokens, then EOS.
        mgr.record_token("ring-001", 42, "hello".into(), false);
        let done = mgr.record_token("ring-001", 2, "</s>".into(), true);
        assert!(done);
    }

    #[test]
    fn test_abort_session() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());
        mgr.record_token("ring-001", 42, "hello".into(), false);

        let result = mgr.abort_session("ring-001", "peer-A");
        assert!(result.is_some());

        let (layer_range, generated) = result.unwrap();
        assert_eq!(layer_range.start, 0);
        assert_eq!(layer_range.end, 16);
        assert_eq!(generated, vec![42]);

        // Session removed.
        assert!(!mgr.has_session("ring-001"));
        assert_eq!(mgr.active_sessions(), 0);
    }

    #[test]
    fn test_remove_session() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());
        mgr.record_token("ring-001", 1, "a".into(), false);
        mgr.record_token("ring-001", 2, "b".into(), false);

        let ids = mgr.remove_session("ring-001");
        assert_eq!(ids, Some(vec![1, 2]));
        assert!(!mgr.has_session("ring-001"));
    }

    #[test]
    fn test_abort_unknown_peer() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        // Abort with unknown peer — session is removed but no layer range.
        let result = mgr.abort_session("ring-001", "peer-unknown");
        assert!(result.is_none());
        // Session was still removed since generated IDs exist but no layer range.
        assert!(!mgr.has_session("ring-001"));
    }

    #[test]
    fn test_four_peer_ring() {
        let config = RingConfig {
            session_id: "ring-4peer".into(),
            request_id: "req-4".into(),
            max_tokens: 5,
            slot_id: 0,
            route: vec![
                RingHop { peer_id: "A".into(), layer_start: 0,  layer_end: 8,  total_layers: 32 },
                RingHop { peer_id: "B".into(), layer_start: 8,  layer_end: 16, total_layers: 32 },
                RingHop { peer_id: "C".into(), layer_start: 16, layer_end: 24, total_layers: 32 },
                RingHop { peer_id: "D".into(), layer_start: 24, layer_end: 32, total_layers: 32 },
            ],
            eos_ids: vec![2],
            hop_timeout_ms: 5000,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
        };

        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(config);

        assert_eq!(mgr.active_sessions(), 1);
        assert_eq!(mgr.shard_map.len(), 4);

        // Abort on peer C — verify lost layer range.
        let result = mgr.abort_session("ring-4peer", "C");
        let (lr, _) = result.unwrap();
        assert_eq!(lr.start, 16);
        assert_eq!(lr.end, 24);
    }

    // ── Pre-dial tests ───────────────────────────────────────────────

    #[test]
    fn test_peers_to_dial_2_peer_ring() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        let peers = mgr.peers_to_dial("ring-001").unwrap();
        assert_eq!(peers.len(), 2);
        assert_eq!(peers[0], "peer-A");
        assert_eq!(peers[1], "peer-B");
    }

    #[test]
    fn test_peers_to_dial_4_peer_ring() {
        let config = RingConfig {
            session_id: "ring-4dial".into(),
            request_id: "req-4d".into(),
            max_tokens: 10,
            slot_id: 0,
            route: vec![
                RingHop { peer_id: "A".into(), layer_start: 0,  layer_end: 8,  total_layers: 32 },
                RingHop { peer_id: "B".into(), layer_start: 8,  layer_end: 16, total_layers: 32 },
                RingHop { peer_id: "C".into(), layer_start: 16, layer_end: 24, total_layers: 32 },
                RingHop { peer_id: "D".into(), layer_start: 24, layer_end: 32, total_layers: 32 },
            ],
            eos_ids: vec![2],
            hop_timeout_ms: 5000,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            seed: None,
        };

        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(config);

        let peers = mgr.peers_to_dial("ring-4dial").unwrap();
        assert_eq!(peers.len(), 4, "4-peer ring must pre-dial all 4 peers");
        assert_eq!(peers, vec!["A", "B", "C", "D"]);
    }

    #[test]
    fn test_peers_to_dial_deduplicates() {
        // Hypothetical: same peer appears twice in route (shouldn't happen
        // in practice, but the dedup must be correct).
        let config = RingConfig {
            session_id: "ring-dedup".into(),
            request_id: "req-dd".into(),
            max_tokens: 5,
            slot_id: 0,
            route: vec![
                RingHop { peer_id: "A".into(), layer_start: 0,  layer_end: 16, total_layers: 32 },
                RingHop { peer_id: "A".into(), layer_start: 16, layer_end: 32, total_layers: 32 },
            ],
            eos_ids: vec![2],
            hop_timeout_ms: 500,
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            seed: None,
        };

        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(config);

        let peers = mgr.peers_to_dial("ring-dedup").unwrap();
        assert_eq!(peers.len(), 1, "duplicate peer_id must be deduplicated");
        assert_eq!(peers[0], "A");
    }

    #[test]
    fn test_peers_to_dial_unknown_session() {
        let mgr = RingManager::new();
        assert!(mgr.peers_to_dial("nonexistent").is_none());
    }

    #[test]
    fn test_peers_from_route_static() {
        let config = test_config();
        let peers = RingManager::peers_from_route(&config);
        assert_eq!(peers, vec!["peer-A", "peer-B"]);
    }

    #[test]
    fn test_peers_from_route_4_peer() {
        let config = RingConfig {
            session_id: "x".into(),
            request_id: "r".into(),
            max_tokens: 1,
            slot_id: 0,
            route: vec![
                RingHop { peer_id: "W".into(), layer_start: 0,  layer_end: 8,  total_layers: 32 },
                RingHop { peer_id: "X".into(), layer_start: 8,  layer_end: 16, total_layers: 32 },
                RingHop { peer_id: "Y".into(), layer_start: 16, layer_end: 24, total_layers: 32 },
                RingHop { peer_id: "Z".into(), layer_start: 24, layer_end: 32, total_layers: 32 },
            ],
            eos_ids: vec![2],
            hop_timeout_ms: 500,
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            seed: None,
        };
        let peers = RingManager::peers_from_route(&config);
        assert_eq!(peers.len(), 4);
        assert_eq!(peers, vec!["W", "X", "Y", "Z"]);
    }

    // ── Request registration tests ───────────────────────────────────

    #[test]
    fn test_register_and_lookup_request() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        mgr.register_request("req-inject-001".into(), "ring-001".into());

        assert!(mgr.is_ring_request("req-inject-001"));
        assert!(!mgr.is_ring_request("req-unknown"));
        assert_eq!(mgr.lookup_session("req-inject-001"), Some("ring-001"));
        assert_eq!(mgr.lookup_session("req-unknown"), None);
        assert_eq!(mgr.pending_request_count(), 1);
    }

    #[test]
    fn test_multiple_requests_same_session() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        // Multiple tokens in flight (pipeline depth > 1).
        mgr.register_request("req-t0".into(), "ring-001".into());
        mgr.register_request("req-t1".into(), "ring-001".into());
        mgr.register_request("req-t2".into(), "ring-001".into());

        assert_eq!(mgr.pending_request_count(), 3);
        assert!(mgr.is_ring_request("req-t0"));
        assert!(mgr.is_ring_request("req-t1"));
        assert!(mgr.is_ring_request("req-t2"));
    }

    #[test]
    fn test_remove_session_cleans_pending_requests() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        mgr.register_request("req-a".into(), "ring-001".into());
        mgr.register_request("req-b".into(), "ring-001".into());
        assert_eq!(mgr.pending_request_count(), 2);

        mgr.remove_session("ring-001");

        assert_eq!(mgr.pending_request_count(), 0);
        assert!(!mgr.is_ring_request("req-a"));
        assert!(!mgr.is_ring_request("req-b"));
    }

    #[test]
    fn test_abort_session_cleans_pending_requests() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        mgr.register_request("req-x".into(), "ring-001".into());
        assert_eq!(mgr.pending_request_count(), 1);

        mgr.abort_session("ring-001", "peer-A");

        assert_eq!(mgr.pending_request_count(), 0);
        assert!(!mgr.is_ring_request("req-x"));
    }

    // ── route_push_result tests ──────────────────────────────────────

    #[test]
    fn test_route_push_result_need_sample() {
        use crate::ipc_codec::{IpcStatus, ActivationDtype};

        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());
        mgr.register_request("req-pr-001".into(), "ring-001".into());

        let header = IpcResponseHeader {
            status: IpcStatus::Ok,
            request_id: "req-pr-001".into(),
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 4],
            ..Default::default()
        };
        let activation = vec![0u8; 16]; // 4 floats

        let action = mgr.route_push_result("req-pr-001", "peer-B", &header, activation.clone());
        match action {
            RingAction::NeedSample { session_id, request_id, activation: act } => {
                assert_eq!(session_id, "ring-001");
                assert_eq!(request_id, "req-pr-001");
                assert_eq!(act, activation);
            }
            other => panic!("expected NeedSample, got {:?}", other),
        }

        // Pending request consumed — second call returns NotRingRequest.
        let action2 = mgr.route_push_result("req-pr-001", "peer-B", &header, vec![]);
        assert!(matches!(action2, RingAction::NotRingRequest));
    }

    #[test]
    fn test_route_push_result_rejects_forged_peer() {
        // Audit F1: a PushResult from a peer other than the registered final
        // hop must be rejected, and the pending mapping must survive so the
        // genuine peer can still deliver.
        use crate::ipc_codec::{IpcStatus, ActivationDtype, IpcResponseHeader};

        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config()); // route ends at peer-B
        mgr.register_request("req-forge".into(), "ring-001".into());

        let header = IpcResponseHeader {
            status: IpcStatus::Ok,
            request_id: "req-forge".into(),
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 4],
            ..Default::default()
        };

        // Forged: comes from peer-A (an intermediate hop), not peer-B.
        let forged = mgr.route_push_result("req-forge", "peer-A", &header, vec![0u8; 16]);
        assert!(
            matches!(forged, RingAction::NotRingRequest),
            "forged callback must be rejected, got {:?}", forged
        );
        // Mapping survived — genuine peer-B can still deliver.
        assert!(mgr.is_ring_request("req-forge"));
        let genuine = mgr.route_push_result("req-forge", "peer-B", &header, vec![0u8; 16]);
        assert!(
            matches!(genuine, RingAction::NeedSample { .. }),
            "genuine callback must succeed, got {:?}", genuine
        );
    }

    #[test]
    fn test_route_push_result_not_ring_request() {
        use crate::ipc_codec::IpcResponseHeader;

        let mut mgr = RingManager::new();
        let header = IpcResponseHeader::default();

        let action = mgr.route_push_result("unknown-req", "peer-B", &header, vec![]);
        assert!(matches!(action, RingAction::NotRingRequest));
    }

    #[test]
    fn test_route_push_result_error_status() {
        use crate::ipc_codec::{IpcStatus, IpcResponseHeader};

        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());
        mgr.register_request("req-err".into(), "ring-001".into());

        let header = IpcResponseHeader {
            status: IpcStatus::Error,
            request_id: "req-err".into(),
            error_message: "GPU OOM".into(),
            ..Default::default()
        };

        let action = mgr.route_push_result("req-err", "peer-B", &header, vec![]);
        match action {
            RingAction::Error { session_id, reason } => {
                assert_eq!(session_id, "ring-001");
                assert!(reason.contains("GPU OOM"));
            }
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_route_push_result_session_removed() {
        use crate::ipc_codec::IpcResponseHeader;

        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());
        mgr.register_request("req-orphan".into(), "ring-001".into());

        // Remove session but leave the pending request.
        mgr.sessions.remove("ring-001");

        let header = IpcResponseHeader::default();
        let action = mgr.route_push_result("req-orphan", "peer-B", &header, vec![]);
        match action {
            RingAction::Error { session_id, reason } => {
                assert_eq!(session_id, "ring-001");
                assert!(reason.contains("no longer active"));
            }
            other => panic!("expected Error, got {:?}", other),
        }
    }

    // ── CP-5: Prefill Pipelining tests ──────────────────────────────

    /// Helper: create fp32 activation bytes for [1, seq_len, hidden_dim].
    fn make_activation(seq_len: usize, hidden_dim: usize) -> Vec<u8> {
        let n_floats = seq_len * hidden_dim;
        (0..n_floats)
            .map(|i| (i as f32))
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    #[test]
    fn test_prefill_chunker_basic() {
        let activation = make_activation(512, 4);
        let shape = vec![1, 512, 4];
        let chunks = PrefillChunker::chunk(&activation, &shape, 128, 4).unwrap();

        assert_eq!(chunks.len(), 4, "512 / 128 = 4 chunks");

        for (i, (data, shp)) in chunks.iter().enumerate() {
            assert_eq!(shp, &vec![1, 128, 4], "chunk {i} shape");
            assert_eq!(data.len(), 128 * 4 * 4, "chunk {i} bytes = 128 * 4 * 4");
        }

        // Verify data continuity: first float of each chunk.
        let first_float = |data: &[u8]| -> f32 {
            f32::from_le_bytes(data[0..4].try_into().unwrap())
        };
        assert_eq!(first_float(&chunks[0].0), 0.0);
        assert_eq!(first_float(&chunks[1].0), (128 * 4) as f32);
        assert_eq!(first_float(&chunks[2].0), (256 * 4) as f32);
        assert_eq!(first_float(&chunks[3].0), (384 * 4) as f32);
    }

    #[test]
    fn test_prefill_chunker_uneven() {
        // 300 tokens, chunk_size=128 → 3 chunks: 128, 128, 44
        let activation = make_activation(300, 2);
        let shape = vec![1, 300, 2];
        let chunks = PrefillChunker::chunk(&activation, &shape, 128, 4).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].1, vec![1, 128, 2]);
        assert_eq!(chunks[1].1, vec![1, 128, 2]);
        assert_eq!(chunks[2].1, vec![1, 44, 2]);
        assert_eq!(chunks[2].0.len(), 44 * 2 * 4);
    }

    #[test]
    fn test_prefill_chunker_no_split() {
        // seq_len <= chunk_size → single chunk (original data).
        let activation = make_activation(100, 4);
        let shape = vec![1, 100, 4];
        let chunks = PrefillChunker::chunk(&activation, &shape, 128, 4).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0.len(), activation.len());
        assert_eq!(chunks[0].1, vec![1, 100, 4]);
    }

    #[test]
    fn test_prefill_chunker_exact_multiple() {
        // 256 tokens, chunk_size=128 → 2 chunks exactly.
        let activation = make_activation(256, 8);
        let shape = vec![1, 256, 8];
        let chunks = PrefillChunker::chunk(&activation, &shape, 128, 4).unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].1, vec![1, 128, 8]);
        assert_eq!(chunks[1].1, vec![1, 128, 8]);

        // Total bytes should match original.
        let total: usize = chunks.iter().map(|(d, _)| d.len()).sum();
        assert_eq!(total, activation.len());
    }

    #[test]
    fn test_prefill_chunker_errors() {
        // Wrong shape dimensions.
        assert!(PrefillChunker::chunk(&[], &[1, 10], 128, 4).is_err());

        // Batch != 1.
        assert!(PrefillChunker::chunk(&[0u8; 32], &[2, 2, 2], 128, 4).is_err());

        // Length mismatch.
        assert!(PrefillChunker::chunk(&[0u8; 10], &[1, 4, 4], 128, 4).is_err());
    }

    #[test]
    fn test_prefill_chunk_prompt_ids() {
        let ids: Vec<i64> = (0..500).collect();
        let chunks = PrefillChunker::chunk_prompt_ids(&ids, 128);

        assert_eq!(chunks.len(), 4); // 128, 128, 128, 116
        assert_eq!(chunks[0].len(), 128);
        assert_eq!(chunks[1].len(), 128);
        assert_eq!(chunks[2].len(), 128);
        assert_eq!(chunks[3].len(), 116);
        assert_eq!(chunks[0][0], 0);
        assert_eq!(chunks[1][0], 128);
        assert_eq!(chunks[3][0], 384);
    }

    #[test]
    fn test_should_chunk() {
        assert!(PrefillChunker::should_chunk(&[1, 512, 896], 256));
        assert!(!PrefillChunker::should_chunk(&[1, 200, 896], 256));
        assert!(!PrefillChunker::should_chunk(&[1, 256, 896], 256));
        assert!(PrefillChunker::should_chunk(&[1, 257, 896], 256));
        // Wrong shape dims.
        assert!(!PrefillChunker::should_chunk(&[512, 896], 256));
        // batch != 1.
        assert!(!PrefillChunker::should_chunk(&[2, 512, 896], 256));
    }

    #[test]
    fn test_init_prefill_pipeline() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        let activation = make_activation(512, 4);
        let shape = vec![1u32, 512, 4];
        let prompt_ids: Vec<i64> = (0..512).collect();

        let first = mgr
            .init_prefill_pipeline("ring-001", activation, shape, prompt_ids, 128)
            .unwrap();

        assert_eq!(first.chunk_index, 0);
        assert_eq!(first.total_chunks, 4);
        assert_eq!(first.shape, vec![1, 128, 4]);
        assert_eq!(first.activation.len(), 128 * 4 * 4);
        assert_eq!(first.prompt_token_ids.len(), 128);
        assert_eq!(first.prompt_token_ids[0], 0);

        assert!(mgr.is_prefill_session("ring-001"));
    }

    #[test]
    fn test_prefill_next_chunk() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        let activation = make_activation(384, 2);
        let shape = vec![1u32, 384, 2];
        let prompt_ids: Vec<i64> = (0..384).collect();

        let _first = mgr
            .init_prefill_pipeline("ring-001", activation, shape, prompt_ids, 128)
            .unwrap();

        // Next chunk = index 1.
        let c1 = mgr.prefill_next_chunk("ring-001").unwrap();
        assert_eq!(c1.chunk_index, 1);
        assert_eq!(c1.shape, vec![1, 128, 2]);
        assert_eq!(c1.prompt_token_ids[0], 128);

        // Next chunk = index 2.
        let c2 = mgr.prefill_next_chunk("ring-001").unwrap();
        assert_eq!(c2.chunk_index, 2);
        assert_eq!(c2.shape, vec![1, 128, 2]);
        assert_eq!(c2.prompt_token_ids[0], 256);

        // No more chunks.
        assert!(mgr.prefill_next_chunk("ring-001").is_none());
    }

    #[test]
    fn test_prefill_push_result_routing() {
        use crate::ipc_codec::{IpcResponseHeader, IpcStatus};

        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        let activation = make_activation(256, 2);
        let shape = vec![1u32, 256, 2];
        let prompt_ids: Vec<i64> = (0..256).collect();

        let _first = mgr
            .init_prefill_pipeline("ring-001", activation, shape, prompt_ids, 128)
            .unwrap();

        // Register chunk request_ids.
        mgr.register_prefill_request("req-c0".into(), "ring-001".into(), 0);
        mgr.register_prefill_request("req-c1".into(), "ring-001".into(), 1);

        let header = IpcResponseHeader {
            status: IpcStatus::Ok,
            request_id: "req-c0".into(),
            ..Default::default()
        };

        // Chunk 0 PushResult — not all received yet.
        let chunk0_result = vec![10u8; 128 * 2 * 4]; // processed chunk 0
        let action = mgr.route_push_result("req-c0", "peer-B", &header, chunk0_result.clone());
        match action {
            RingAction::PrefillChunkReceived {
                session_id,
                chunk_index,
                chunks_received,
                chunks_total,
            } => {
                assert_eq!(session_id, "ring-001");
                assert_eq!(chunk_index, 0);
                assert_eq!(chunks_received, 1);
                assert_eq!(chunks_total, 2);
            }
            other => panic!("expected PrefillChunkReceived, got {:?}", other),
        }

        // Chunk 1 PushResult — all received, returns NeedSample.
        let chunk1_result = vec![20u8; 128 * 2 * 4]; // processed chunk 1
        let header1 = IpcResponseHeader {
            status: IpcStatus::Ok,
            request_id: "req-c1".into(),
            ..Default::default()
        };
        let action = mgr.route_push_result("req-c1", "peer-B", &header1, chunk1_result.clone());
        match action {
            RingAction::NeedSample {
                session_id,
                activation,
                ..
            } => {
                assert_eq!(session_id, "ring-001");
                // Concatenated: chunk0_result + chunk1_result.
                assert_eq!(activation.len(), chunk0_result.len() + chunk1_result.len());
                assert_eq!(&activation[..10], &[10u8; 10]);
                assert_eq!(
                    &activation[chunk0_result.len()..chunk0_result.len() + 10],
                    &[20u8; 10]
                );
            }
            other => panic!("expected NeedSample (prefill complete), got {:?}", other),
        }

        // Prefill state cleared.
        assert!(!mgr.is_prefill_session("ring-001"));
    }

    #[test]
    fn test_prefill_out_of_order_push_results() {
        use crate::ipc_codec::{IpcResponseHeader, IpcStatus};

        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        let activation = make_activation(384, 2);
        let shape = vec![1u32, 384, 2];
        let prompt_ids: Vec<i64> = (0..384).collect();

        let _first = mgr
            .init_prefill_pipeline("ring-001", activation, shape, prompt_ids, 128)
            .unwrap();

        mgr.register_prefill_request("req-c0".into(), "ring-001".into(), 0);
        mgr.register_prefill_request("req-c1".into(), "ring-001".into(), 1);
        mgr.register_prefill_request("req-c2".into(), "ring-001".into(), 2);

        let ok_header = |rid: &str| IpcResponseHeader {
            status: IpcStatus::Ok,
            request_id: rid.into(),
            ..Default::default()
        };

        // Chunks arrive out of order: 2, 0, 1.
        let r2 = vec![2u8; 128 * 2 * 4];
        let action = mgr.route_push_result("req-c2", "peer-B", &ok_header("req-c2"), r2.clone());
        assert!(matches!(action, RingAction::PrefillChunkReceived { chunk_index: 2, .. }));

        let r0 = vec![0u8; 128 * 2 * 4];
        let action = mgr.route_push_result("req-c0", "peer-B", &ok_header("req-c0"), r0.clone());
        assert!(matches!(action, RingAction::PrefillChunkReceived { chunk_index: 0, .. }));

        let r1 = vec![1u8; 128 * 2 * 4];
        let action = mgr.route_push_result("req-c1", "peer-B", &ok_header("req-c1"), r1.clone());
        // All received — concatenated in ORDER (0, 1, 2), not arrival order.
        match action {
            RingAction::NeedSample { activation, .. } => {
                assert_eq!(activation.len(), r0.len() + r1.len() + r2.len());
                // First byte should be from chunk 0.
                assert_eq!(activation[0], 0u8);
                // After chunk 0, chunk 1.
                assert_eq!(activation[r0.len()], 1u8);
                // After chunk 1, chunk 2.
                assert_eq!(activation[r0.len() + r1.len()], 2u8);
            }
            other => panic!("expected NeedSample, got {:?}", other),
        }
    }

    #[test]
    fn test_prefill_cleanup_on_abort() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        let activation = make_activation(256, 2);
        let shape = vec![1u32, 256, 2];

        let _first = mgr
            .init_prefill_pipeline("ring-001", activation, shape, vec![], 128)
            .unwrap();

        mgr.register_prefill_request("req-c0".into(), "ring-001".into(), 0);
        mgr.register_prefill_request("req-c1".into(), "ring-001".into(), 1);

        assert_eq!(mgr.prefill_request_count(), 2);

        // Abort cleans up everything.
        mgr.abort_session("ring-001", "peer-A");

        assert_eq!(mgr.pending_request_count(), 0);
        assert_eq!(mgr.prefill_request_count(), 0);
        assert!(!mgr.has_session("ring-001"));
    }

    #[test]
    fn test_prefill_init_too_short() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        // 100 tokens with chunk_size=128 → can't chunk.
        let activation = make_activation(100, 4);
        let shape = vec![1u32, 100, 4];

        let result =
            mgr.init_prefill_pipeline("ring-001", activation, shape, vec![], 128);
        assert!(result.is_err());
        assert!(!mgr.is_prefill_session("ring-001"));
    }

    #[test]
    fn test_prefill_pipeline_state_concatenate() {
        let mut state = PrefillPipelineState {
            chunks: vec![], // not used for this test
            next_inject: 0,
            received: vec![None, None, None],
            received_count: 0,
        };

        state.store_result(1, vec![10, 20]);
        assert!(!state.all_received());

        state.store_result(0, vec![1, 2]);
        assert!(!state.all_received());

        state.store_result(2, vec![30, 40]);
        assert!(state.all_received());

        // Concatenation is in order: chunk 0, 1, 2.
        let concat = state.concatenate();
        assert_eq!(concat, vec![1, 2, 10, 20, 30, 40]);
    }

    #[test]
    fn test_prefill_register_request_cleanup() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());

        mgr.register_prefill_request("pf-0".into(), "ring-001".into(), 0);
        mgr.register_prefill_request("pf-1".into(), "ring-001".into(), 1);

        assert!(mgr.is_ring_request("pf-0"));
        assert!(mgr.is_ring_request("pf-1"));
        assert_eq!(mgr.prefill_request_count(), 2);

        // remove_session should clean both maps.
        mgr.remove_session("ring-001");
        assert!(!mgr.is_ring_request("pf-0"));
        assert_eq!(mgr.prefill_request_count(), 0);
    }

    // ── B3/B4 tests ─────────────────────────────────────────────────

    #[test]
    fn test_check_timeouts_no_timeout() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());
        // Immediately after creation, nothing should be timed out.
        assert!(mgr.check_timeouts().is_empty());
    }

    #[test]
    fn test_check_timeouts_expired() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());
        // Manually backdate the last_inject_time to 60s ago.
        mgr.sessions
            .get_mut("ring-001")
            .unwrap()
            .last_inject_time = std::time::Instant::now() - std::time::Duration::from_secs(60);
        let timed_out = mgr.check_timeouts();
        assert_eq!(timed_out.len(), 1);
        assert_eq!(timed_out[0].0, "ring-001");
    }

    #[test]
    fn test_peer_has_active_session() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());
        assert!(mgr.peer_has_active_session("peer-A"));
        assert!(mgr.peer_has_active_session("peer-B"));
        assert!(!mgr.peer_has_active_session("peer-C"));
    }

    #[test]
    fn test_abort_sessions_for_peer() {
        let mut mgr = RingManager::new();
        let _h1 = mgr.start_session(test_config());
        // Second session on different peers.
        let mut cfg2 = test_config();
        cfg2.session_id = "ring-002".into();
        cfg2.request_id = "req-002".into();
        cfg2.route = vec![
            RingHop {
                peer_id: "peer-C".into(),
                layer_start: 0,
                layer_end: 16,
                total_layers: 32,
            },
            RingHop {
                peer_id: "peer-D".into(),
                layer_start: 16,
                layer_end: 32,
                total_layers: 32,
            },
        ];
        let _h2 = mgr.start_session(cfg2);

        assert_eq!(mgr.active_sessions(), 2);

        // Abort sessions involving peer-A — only ring-001 should be aborted.
        let aborted = mgr.abort_sessions_for_peer("peer-A");
        assert_eq!(aborted.len(), 1);
        assert_eq!(aborted[0].0, "ring-001");
        assert_eq!(mgr.active_sessions(), 1);
        assert!(mgr.has_session("ring-002"));
    }

    #[test]
    fn test_abort_sessions_for_peer_cleans_pending() {
        let mut mgr = RingManager::new();
        let _handle = mgr.start_session(test_config());
        mgr.register_request("req-100".into(), "ring-001".into());
        mgr.register_request("req-101".into(), "ring-001".into());
        assert!(mgr.is_ring_request("req-100"));
        assert!(mgr.is_ring_request("req-101"));

        mgr.abort_sessions_for_peer("peer-B");
        // Pending requests should be cleaned up.
        assert!(!mgr.is_ring_request("req-100"));
        assert!(!mgr.is_ring_request("req-101"));
        assert_eq!(mgr.active_sessions(), 0);
    }
}
