// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Provider-side swarm wiring: advertise an engine's models to the DHT and serve inbound
//! requests by proxying to the engine.
//!
//! [`Provider`] joins an [`EngineAdapter`] to a [`NetworkHandle`]:
//! * [`announce_models`](Provider::announce_models) — detect the engine's models and
//!   publish a [`PeerRecord`] (canonical id + this node's libp2p id + ed25519 key) for
//!   each, so consumers/routers can discover them.
//! * [`run_inbound`](Provider::run_inbound) — the blocking serve loop: poll inbound proxy
//!   requests, dispatch the serve method byte to [`handle_serve_inbound`], and reply with
//!   the (buffered) completion.
//!
//! The pure pieces — record construction and the method-byte dispatch — are unit-tested;
//! the loop itself is thin glue over the live swarm.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::share_policy::{PolicyWatcher, SharePolicy};

use openhydra_network::handle::NetworkHandle;
use openhydra_network::types::PeerRecord;
use openhydra_protocol::credit::{throttle_multiplier, CreditAccount};
use openhydra_protocol::receipts::CoSignedReceipt;
use openhydra_protocol::store::Store;

use crate::adapter::{AdapterError, DetectedModel, EngineAdapter};
use crate::aup::{AupDecision, AupPolicy};
use crate::ratelimit::{RateLimitConfig, RateLimiter};
use crate::receipt::{handle_receipt_inbound, RECEIPT_REQUEST};
use crate::serve::{
    frame_response, handle_serve_request, handle_serve_request_parsed, FetchChunksResponse,
    FetchResponse, ServeChunk, ServeRequest, ServeSummary,
};
use crate::status::TransferStats;
use crate::workpool::WorkerPool;

/// Base serve delay scaled by [`throttle_multiplier`] for a throttled (leecher) consumer
/// (M2.3 enforcement). Modest — even the deepest leecher's wait is `BASE_THROTTLE *
/// MAX_THROTTLE_MULT` (≈1.8s at the defaults), a slowdown, never a block. Tunable.
const BASE_THROTTLE: Duration = Duration::from_millis(200);

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// libp2p proxy method byte for an inference serve request (consumer → provider). Sits
/// alongside the existing peer method bytes (0x01 Forward … 0x07 Receipt).
pub const SERVE_REQUEST: u8 = 0x10;

/// libp2p proxy method byte for a **reconnect-and-fetch** request (consumer → provider):
/// `0x11 ‖ nonce(16)`. A consumer whose serve connection dropped during a long generation
/// re-requests the buffered result by the same nonce it committed for settlement, on a fresh
/// circuit, instead of losing the completed work. See `RECONNECT_AND_FETCH_PLAN.md`.
pub const FETCH_RESULT: u8 = 0x11;

/// libp2p proxy method byte for a streaming **chunk poll** (consumer → provider):
/// `0x14 ‖ nonce(16) ‖ offset(4, big-endian)`. The consumer polls for the serve frames at or
/// after `offset`; the provider replies with a [`FetchChunksResponse`] carrying the incremental
/// slice, prunes the acknowledged prefix, and reports `done` when the terminal frame is included.
/// (`0x12` is [`RECEIPT_REQUEST`](crate::receipt::RECEIPT_REQUEST); `0x13` is [`SERVE_STREAM`].)
/// See `STREAMING_SERVE_P1_PLAN.md`.
pub const FETCH_CHUNKS: u8 = 0x14;

/// libp2p proxy method byte to **open a stream** (consumer → provider): `0x13 ‖ ServeRequest`.
/// Same request body as [`SERVE_REQUEST`], but instead of buffering the whole completion and
/// returning it in one blob, the provider submits the generation to the worker pool, replies
/// with an immediate [`FetchChunksResponse`] ack, and streams the frames incrementally to
/// [`FETCH_CHUNKS`] polls. A legacy provider that doesn't know this byte answers with a framed
/// `Error` blob, which the consumer detects (it fails to decode as a `FetchChunksResponse`) and
/// falls back to [`SERVE_REQUEST`].
pub const SERVE_STREAM: u8 = 0x13;

/// How long a completed (or in-flight) serve result stays fetchable. Must comfortably exceed
/// the NAT/relay re-establishment window (~10–70s) so a reconnecting consumer still finds it.
const RESULT_TTL_MS: u64 = 120_000;
/// Hard cap on buffered results (count) — evict oldest-first past this.
const MAX_RESULT_ENTRIES: usize = 256;
/// Hard cap on total buffered result bytes (a single 4K video base64 is MBs) — evict
/// oldest-first past this so the buffer can't become a memory DoS.
const MAX_RESULT_BYTES: usize = 64 * 1024 * 1024;
/// E-S8-style per-peer rate cap on fetch requests (cheap map lookups, but still bounded).
const FETCH_RATE_RPS: f64 = 50.0;
const FETCH_RATE_BURST: f64 = 100.0;
const FETCH_RATE_MAX_TRACKED: usize = 4096;

/// A buffered serve result, keyed by the consumer's committed nonce, so a reconnecting
/// consumer can fetch it after the serve connection drops (reconnect-and-fetch).
struct BufferedResult {
    /// The consumer libp2p peer id allowed to fetch — the `reply_to` from the serve request.
    /// A fetch from any other authenticated peer is `Forbidden`.
    reply_to: String,
    state: ResultState,
    /// Encoded-frame byte size, for the total-bytes cap.
    bytes: usize,
    recorded_ms: u64,
}

/// Lifecycle of a buffered result: generating (in-flight; a fetch says "retry") → ready
/// (the encoded [`ServeChunk`] frames, byte-identical to a fresh serve response).
enum ResultState {
    Generating,
    Ready(Vec<Vec<u8>>),
}

/// Build the DHT record advertising one detected model.
///
/// `model_id` (the DHT key consumers look up) is the **engine handle** (e.g. Ollama's
/// `"qwen2.5:7b"`) — the familiar name a consumer requests; it's also what the provider's
/// adapter serves (`ServeRequest.model_ref`). `canonical_model_id` carries the precise
/// `family/params/quant/template_hash` for the router's `is_compatible` filter.
/// `host`/`port` are advisory — routing is by `libp2p_peer_id`.
///
/// MVP-limitation: keying on the engine handle means the same model served via two
/// different engines (Ollama vs vLLM) advertises under different ids; a normalized
/// cross-engine discovery key is the model-id-governance refinement (deferred).
pub fn build_peer_record(
    model: &DetectedModel,
    openhydra_peer_id: &str,
    libp2p_peer_id: &str,
    public_key_hex: &str,
    host: &str,
    port: u16,
) -> PeerRecord {
    PeerRecord {
        peer_id: openhydra_peer_id.to_string(),
        model_id: model.engine_ref.clone(),
        host: host.to_string(),
        port,
        canonical_model_id: model.canonical_id.clone(),
        libp2p_peer_id: libp2p_peer_id.to_string(),
        public_key: public_key_hex.to_string(),
        ..Default::default()
    }
}

/// Provider-side dispatch for one inbound proxy request → the buffered serve response.
///
/// On the [`SERVE_REQUEST`] method byte, run the request through `adapter` and return the
/// length-framed [`ServeChunk`](crate::serve::ServeChunk) sequence. Any other method byte
/// yields a single framed `Error` chunk (this loop only speaks the serve protocol).
pub fn handle_serve_inbound(data: &[u8], adapter: &dyn EngineAdapter) -> Vec<u8> {
    if data.first() != Some(&SERVE_REQUEST) {
        return frame_response(&[
            crate::serve::ServeChunk::Error("unsupported method".into()).encode(),
        ]);
    }
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    handle_serve_request(&data[1..], adapter, &mut |c| chunks.push(c.to_vec()));
    frame_response(&chunks)
}

/// Persist an accepted co-signed receipt to the ledger, if a store is configured.
///
/// Best-effort and side-effecting only: a missing store or a ledger error never fails the
/// request (the consumer already holds the same co-signed receipt as its own proof). Pure
/// enough to unit-test without a live node — the signing/co-signing happens upstream in
/// [`handle_receipt_inbound`]; this is just the persistence decision.
fn ledger_receipt(store: Option<&Store>, accepted: Option<&CoSignedReceipt>) {
    if let (Some(store), Some(receipt)) = (store, accepted) {
        match store.record_receipt(receipt) {
            Ok(true) => {
                eprintln!("openhydra-agent: ledgered receipt ({} tokens)", receipt.payload.tokens)
            }
            Ok(false) => eprintln!("openhydra-agent: receipt replay ignored (nonce already spent)"),
            Err(e) => eprintln!("openhydra-agent: ledger error (receipt not persisted): {e}"),
        }
    }
}

/// What the provider recorded about a serve, keyed by the consumer-committed receipt nonce,
/// so a settlement receipt can be validated against real work done (B-S1): same model,
/// `tokens <= served`, single-use, and fresh. Bounded + TTL-pruned (B-S7).
struct ServeCommitment {
    model_id: String,
    tokens: u64,
    recorded_ms: u64,
}

/// How long a serve commitment stays settleable, and the receipt-freshness window (B-S1/S7).
const COMMITMENT_TTL_MS: u64 = 5 * 60 * 1000;
/// Allowance for consumer/provider clock skew on the receipt timestamp.
const CLOCK_SKEW_MS: u64 = 60 * 1000;
/// Hard cap on outstanding (un-settled) commitments — a backstop so a flood of serves whose
/// receipts never arrive can't grow the map without bound.
const MAX_COMMITMENTS: usize = 10_000;

/// E-S8: sustained ceiling on inbound *receipt* settlements processed per second, **per
/// sending peer**. Each receipt costs an Ed25519 verify + co-sign; unbounded, a peer can flood
/// them to burn CPU and monopolize the worker pool, starving real serve requests. Honest
/// volume is ~1 per completed serve, so this ceiling is far above legitimate need and only
/// bites a flood — and because it is keyed on the libp2p-authenticated sender (threaded in via
/// `InboundProxyItem.source_peer`), one abusive peer can't shed another peer's settlements.
const RECEIPT_RATE_RPS: f64 = 50.0;
/// E-S8: token-bucket burst above the sustained receipt rate (per peer).
const RECEIPT_RATE_BURST: f64 = 100.0;
/// E-S8: cap on distinct sender identities the receipt limiter tracks. The limiter reclaims
/// idle (full-bucket) entries first, so a peer churning identities can't turn it into a memory
/// DoS; memory is O(currently-throttled peers), bounded by this.
const RECEIPT_RATE_MAX_TRACKED: usize = 4096;

/// Pure B-S1/B-S7 validation, extracted so it is unit-testable without a live `Provider`.
/// Prunes expired commitments, checks the receipt fields against the serve recorded under
/// `nonce`, and consumes the commitment (single-use) on success. `Err(reason)` rejects the
/// receipt before it is co-signed.
fn validate_and_consume_commitment(
    map: &mut HashMap<[u8; 16], ServeCommitment>,
    nonce: &[u8; 16],
    model_id: &str,
    tokens: u64,
    ts_unix_ms: u64,
    now: u64,
) -> Result<(), String> {
    // Receipt-timestamp freshness (B-S7): reject stale or far-future receipts.
    if ts_unix_ms + COMMITMENT_TTL_MS < now || ts_unix_ms > now + CLOCK_SKEW_MS {
        return Err("receipt timestamp outside freshness window".into());
    }
    map.retain(|_, c| now.saturating_sub(c.recorded_ms) <= COMMITMENT_TTL_MS);
    let c = map.get(nonce).ok_or("no serve commitment for this receipt nonce")?;
    if model_id != c.model_id {
        return Err("receipt model does not match the served model".into());
    }
    if tokens > c.tokens {
        return Err("receipt tokens exceed tokens actually served".into());
    }
    map.remove(nonce); // single-use
    Ok(())
}

/// Pure reconnect-and-fetch lookup, extracted so it is unit-testable without a live
/// `Provider`. `source_peer` is the libp2p-authenticated fetcher; only the consumer that
/// committed the nonce (`reply_to`) may retrieve the buffered result.
fn fetch_from_buffer(
    map: &HashMap<[u8; 16], BufferedResult>,
    nonce: &[u8; 16],
    source_peer: &str,
) -> FetchResponse {
    match map.get(nonce) {
        None => FetchResponse::NotFound,
        // Ownership binding: a fetch from anyone but the committing consumer is refused, so a
        // peer that learns/guesses a nonce can't steal another consumer's result.
        Some(r) if r.reply_to != source_peer => FetchResponse::Forbidden,
        Some(r) => match &r.state {
            ResultState::Generating => FetchResponse::Generating,
            ResultState::Ready(frames) => FetchResponse::Ready(frame_response(frames)),
        },
    }
}

/// Evict oldest-first until the result buffer is within both the count and byte caps, so a
/// flood of buffered results (or a few huge ones) can't grow provider memory without bound.
/// O(n·evictions) but n ≤ [`MAX_RESULT_ENTRIES`], so trivial. Free fn (no `Self`) to keep it
/// unit-testable without a live `Provider`.
fn enforce_result_caps(map: &mut HashMap<[u8; 16], BufferedResult>) {
    loop {
        let total: usize = map.values().map(|r| r.bytes).sum();
        if map.len() <= MAX_RESULT_ENTRIES && total <= MAX_RESULT_BYTES {
            break;
        }
        match map.iter().min_by_key(|(_, r)| r.recorded_ms).map(|(k, _)| *k) {
            Some(oldest) => {
                map.remove(&oldest);
            }
            None => break,
        }
    }
}

// ── P1 streaming: trimmable-log stream buffer ──────────────────────────────────────────
//
// The reconnect-fetch buffer above ([`BufferedResult`], whole-result-until-TTL) does not
// survive thousands of concurrent streams: it holds every completed result whole until the
// TTL and evicts oldest-first, so active streams evict each other under load. The streaming
// path instead keys each nonce to a [`StreamBuffer`] — an append-only log with a *trimmable
// prefix*. A poll for absolute chunk `offset` proves the consumer holds `[0..offset)`, so the
// provider drops that prefix; per-stream memory is the un-fetched tail only (a few KB),
// independent of total generation size. See `STREAMING_SERVE_P1_PLAN.md`.
//
// This is a **separate map** from `results` (0x10/0x11 stays byte-for-byte unchanged); a given
// nonce is created by *either* a buffered serve *or* a streaming serve, never both, so the two
// buffers never alias.

/// How long a stream buffer stays fetchable after its last activity (append or poll). Matches
/// [`RESULT_TTL_MS`] so a consumer whose relay dropped mid-stream has the same reconnect window.
const STREAM_TTL_MS: u64 = RESULT_TTL_MS;
/// Global cap on live stream buffers. Streams are tiny (prune-on-ack keeps only the un-fetched
/// tail), so this is generous; the real bound on *generating* streams is the worker pool.
const MAX_STREAM_ENTRIES: usize = 512;
/// Per-peer fairness cap: one consumer can hold at most this many live streams. Admission
/// evicts **that peer's own** oldest stream first, so one peer opening many streams can never
/// evict another peer's active stream (the property the scale scenario demands).
const MAX_STREAMS_PER_PEER: usize = 128;
/// Global byte backstop across all live stream tails — a last-resort guard (deltas are small and
/// pruned, so this rarely bites), evicting the globally-oldest stream if somehow exceeded.
const MAX_STREAM_BYTES: usize = 64 * 1024 * 1024;

/// A per-nonce append-only log of encoded [`ServeChunk`] frames with a trimmable prefix (P1
/// streaming). See the module note above.
struct StreamBuffer {
    /// Peer-bound: only the consumer that committed the nonce (its `reply_to`) may poll.
    reply_to: String,
    /// Absolute chunk index of `chunks[0]`; advances as the acknowledged prefix is pruned.
    base_offset: u32,
    /// Encoded [`ServeChunk`] frames covering absolute indices `[base_offset, base_offset+len)`.
    chunks: Vec<Vec<u8>>,
    /// A terminal `Done`/`Error` frame has been appended (end of stream). Set atomically with
    /// the terminal append so a poll never sees `done` without the terminal chunk in the log.
    done: bool,
    /// Live (un-pruned) byte size, for the byte backstop.
    bytes: usize,
    /// TTL + last-activity stamp (append or poll refreshes it).
    recorded_ms: u64,
}

/// F1: whether a serve request's `reply_to` is allowed, i.e. equals the libp2p-authenticated
/// sender. A legitimate consumer always sets `reply_to = its own peer id`, which is the
/// authenticated `source_peer` on every path; a mismatch means a peer is trying to attribute a
/// stream to someone else (spoofing the fairness-cap bucket / delivery target). The env
/// kill-switch `OPENHYDRA_DISABLE_REPLYTO_BIND=1` reverts to accept-all at runtime (no rebuild)
/// as an ops escape hatch if enforcement ever misfires on a legitimate path.
fn reply_to_authorized(reply_to: &str, source_peer: &str) -> bool {
    if std::env::var("OPENHYDRA_DISABLE_REPLYTO_BIND").as_deref() == Ok("1") {
        return true;
    }
    reply_to_matches(reply_to, source_peer)
}

/// The pure binding rule (no env): a `reply_to` is authentic iff it is the authenticated sender.
/// Split out so the core rule is unit-tested without touching the process-global kill-switch env
/// var (which would race the parallel test runner).
fn reply_to_matches(reply_to: &str, source_peer: &str) -> bool {
    reply_to == source_peer
}

/// F5: run a generation, catching a panic in the adapter/engine and turning it into a terminal
/// `Error` chunk so a bug can't strand the consumer. Used by **both** serve transports:
/// - streaming (`SERVE_STREAM`) — the appended terminal flips the stream buffer's `done`, so it
///   never lingers `done=false` (which would stall the consumer for the full ~120s stall deadline);
/// - buffered (`SERVE_REQUEST`) — the terminal is pushed into the response frames and stored under
///   the nonce, so a panic yields an immediate framed error instead of an unsent reply that leaves
///   the nonce stuck `Generating` until the consumer's serve timeout + reconnect-fetch deadline.
///
/// On a normal `Ok`/`Err` outcome the underlying [`handle_serve_request_parsed`] already emits the
/// terminal `Done`/`Error`, so those paths are byte-for-byte unchanged; only the panic branch is
/// new. Free fn (no `Self`) so it is unit-testable with a panicking stub adapter. `AssertUnwindSafe`
/// is sound here — the injected `on_chunk` sink (a `Mutex`-guarded append for the stream path, a
/// local `Vec::push` for the buffered path) holds nothing locked across calls, so a panic between
/// chunks leaves no poisoned state the terminal append can't recover.
fn generate_guarded(
    req: ServeRequest,
    adapter: &dyn EngineAdapter,
    on_chunk: &mut dyn FnMut(&[u8]),
) -> ServeSummary {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle_serve_request_parsed(req, adapter, on_chunk)
    })) {
        Ok(summary) => summary,
        Err(_) => {
            on_chunk(&ServeChunk::Error("provider generation failed".into()).encode());
            ServeSummary { ok: false, ..Default::default() } // tokens 0 → no commitment recorded
        }
    }
}

/// Admit a new stream under `nonce`: prune expired buffers, enforce the per-peer fairness cap
/// (evicting only *this* peer's oldest — and since `reply_to` is bound to the authenticated
/// sender at `SERVE_STREAM` time (F1), that "peer" is the real sender, so one peer genuinely
/// cannot evict another's active stream), enforce the global byte/count backstops, then insert
/// an empty `done=false` buffer. Free fn (no `Self`) so the buffer mechanics are unit-testable
/// without a live `Provider`.
fn stream_begin_in(
    map: &mut HashMap<[u8; 16], StreamBuffer>,
    nonce: [u8; 16],
    reply_to: &str,
    now: u64,
) {
    map.retain(|_, b| now.saturating_sub(b.recorded_ms) <= STREAM_TTL_MS);
    // Per-peer fairness: while this peer already holds the cap, drop its own oldest. Only ever
    // touches `reply_to`'s streams, so it cannot evict another peer's active stream.
    loop {
        let mine = map.values().filter(|b| b.reply_to == reply_to).count();
        if mine < MAX_STREAMS_PER_PEER {
            break;
        }
        match map
            .iter()
            .filter(|(_, b)| b.reply_to == reply_to)
            .min_by_key(|(_, b)| b.recorded_ms)
            .map(|(k, _)| *k)
        {
            Some(oldest) => {
                map.remove(&oldest);
            }
            None => break,
        }
    }
    // Global backstops: evict the globally-oldest until within the count + byte caps.
    loop {
        let total: usize = map.values().map(|b| b.bytes).sum();
        if map.len() < MAX_STREAM_ENTRIES && total <= MAX_STREAM_BYTES {
            break;
        }
        match map.iter().min_by_key(|(_, b)| b.recorded_ms).map(|(k, _)| *k) {
            Some(oldest) => {
                map.remove(&oldest);
            }
            None => break,
        }
    }
    map.insert(
        nonce,
        StreamBuffer { reply_to: reply_to.to_string(), base_offset: 0, chunks: Vec::new(), done: false, bytes: 0, recorded_ms: now },
    );
}

/// Append one encoded frame to the stream under `nonce` (no-op if the buffer was already reaped).
/// A terminal (`Done`/`Error`) frame sets `done` **in the same lock op** as the append, so a
/// concurrent poll can never observe `done=true` without the terminal chunk being fetchable.
fn stream_append_in(
    map: &mut HashMap<[u8; 16], StreamBuffer>,
    nonce: &[u8; 16],
    encoded: &[u8],
    now: u64,
) {
    if let Some(buf) = map.get_mut(nonce) {
        buf.bytes += encoded.len();
        buf.chunks.push(encoded.to_vec());
        buf.recorded_ms = now;
        if ServeChunk::frame_is_terminal(encoded) {
            buf.done = true;
        }
    }
}

/// Serve a `FETCH_CHUNKS` poll from the buffer: peer-bind, return the incremental slice
/// `[offset..]`, prune the acknowledged prefix `[base_offset..offset)`, and report `done`. Pure
/// (operates on `&mut map`) so it is unit-testable without a live `Provider`; TTL pruning + the
/// per-peer rate-limit stay in the [`Provider::handle_fetch_chunks`] wrapper.
fn fetch_chunks_from(
    map: &mut HashMap<[u8; 16], StreamBuffer>,
    nonce: &[u8; 16],
    source_peer: &str,
    offset: u32,
    now: u64,
) -> FetchChunksResponse {
    let buf = match map.get_mut(nonce) {
        None => return FetchChunksResponse::NotFound,
        // Ownership binding: a poll from anyone but the committing consumer is refused.
        Some(b) if b.reply_to != source_peer => return FetchChunksResponse::Forbidden,
        Some(b) => b,
    };
    let have_hi = buf.base_offset + buf.chunks.len() as u32;
    // Clamp the request into the live window: a stale/duplicate poll below base (already pruned)
    // resumes at base; a poll past the end yields an empty slice.
    let start_abs = offset.clamp(buf.base_offset, have_hi);
    let li = (start_abs - buf.base_offset) as usize;
    // Copy the slice to return BEFORE pruning (frame_response copies, so the drain below is safe).
    let framed = frame_response(&buf.chunks[li..]);
    let done = buf.done;
    // Prune-on-ack: the poll for `offset` proves `[0..offset)` was received, so drop
    // `[base_offset..start_abs)`. We keep `[start_abs..]` (the data just returned) so a transport
    // drop before the consumer processes it re-polls the same offset and gets it again.
    if li > 0 {
        buf.chunks.drain(0..li);
        buf.base_offset = start_abs;
        buf.bytes = buf.chunks.iter().map(|c| c.len()).sum();
    }
    buf.recorded_ms = now;
    FetchChunksResponse::Chunks { framed, next_offset: have_hi, done }
}

/// An engine joined to the swarm: advertises its models and serves inbound requests.
pub struct Provider<A: EngineAdapter> {
    adapter: A,
    net: NetworkHandle,
    host: String,
    port: u16,
    /// Ledger for accepted co-signed receipts. `None` → receipts are co-signed and returned
    /// but not persisted (the swarm still works; this node just keeps no local record).
    store: Option<Store>,
    /// M2.3 give/take credit, keyed by **consumer libp2p PeerId** — the same id a consumer
    /// announces as its `reply_to`, derived at receipt time from the consumer's ed25519
    /// pubkey. Rehydrated from `store` when one is attached, flushed on each accrual.
    credit: Mutex<HashMap<String, CreditAccount>>,
    /// Count of worker threads currently sleeping in an M2.3 throttle delay. Read to reserve
    /// at least one non-throttling worker (see [`maybe_throttle`](Self::maybe_throttle)), so a
    /// flood of leechers can never put every worker to sleep at once.
    throttling: AtomicUsize,
    /// Acceptable-use policy floor: an inbound serve request the operator's policy refuses is
    /// answered with an `Error` frame instead of being run through the engine. Default
    /// permissive (allow everything) until limits are configured.
    aup: AupPolicy,
    /// P0 introspection: shared transfer counters read by the `--status-bind` server.
    /// `None` → no counting (status endpoint not enabled).
    stats: Option<Arc<TransferStats>>,
    /// B-S1/B-S7: receipt-nonce → what we served under it, so a settlement receipt is checked
    /// against real work (tokens ≤ served, model match, single-use, fresh). Bounded + pruned.
    serve_commitments: Mutex<HashMap<[u8; 16], ServeCommitment>>,
    /// E-S8: global token-bucket cap on inbound receipt settlements — sheds a receipt flood
    /// *before* the Ed25519 verify+sign, so receipt crypto can't monopolize the worker pool.
    receipt_rl: Arc<RateLimiter>,
    /// Reconnect-and-fetch: nonce → buffered serve result, so a consumer whose connection
    /// dropped during a long generation can fetch the completed work on a fresh circuit
    /// instead of losing it. Bound to the serve's `reply_to`, TTL-pruned + byte/count-capped.
    results: Mutex<HashMap<[u8; 16], BufferedResult>>,
    /// E-S8: per-peer rate cap on fetch requests (mirrors `receipt_rl`; a separate bucket so a
    /// fetch flood can't shed receipts and vice-versa). Shared by `FETCH_RESULT` + `FETCH_CHUNKS`.
    fetch_rl: Arc<RateLimiter>,
    /// P1 streaming: nonce → trimmable-log [`StreamBuffer`], populated by `SERVE_STREAM` and
    /// polled/pruned by `FETCH_CHUNKS`. Separate from `results` (0x10/0x11 buffered path) so the
    /// proven reconnect-fetch path is untouched; prune-on-ack keeps per-stream memory bounded to
    /// the un-fetched tail so thousands of concurrent streams don't evict one another.
    streams: Mutex<HashMap<[u8; 16], StreamBuffer>>,
    /// Which models this provider shares (announces + serves), keyed on the engine handle
    /// (`DetectedModel::engine_ref`, the same string a consumer sends as `ServeRequest.model_ref`).
    /// A [`PolicyWatcher`] so the policy can be **hot-reloaded at runtime** — the user can toggle
    /// sharing in the desktop without restarting the node. Both the announce filter and the serve
    /// gate consult it, so a de-selected model is genuinely off (not merely hidden).
    policy: PolicyWatcher,
}

impl<A: EngineAdapter> Provider<A> {
    pub fn new(adapter: A, net: NetworkHandle) -> Self {
        Self {
            adapter,
            net,
            host: String::new(),
            port: 0,
            store: None,
            credit: Mutex::new(HashMap::new()),
            throttling: AtomicUsize::new(0),
            aup: AupPolicy::permissive(),
            stats: None,
            serve_commitments: Mutex::new(HashMap::new()),
            receipt_rl: Arc::new(RateLimiter::new(RateLimitConfig {
                rps: RECEIPT_RATE_RPS,
                burst: RECEIPT_RATE_BURST,
                max_inflight: 0, // rate-only; concurrency is already bounded by the worker pool
                max_tracked: RECEIPT_RATE_MAX_TRACKED, // per-peer buckets, bounded + idle-evicted
            })),
            results: Mutex::new(HashMap::new()),
            fetch_rl: Arc::new(RateLimiter::new(RateLimitConfig {
                rps: FETCH_RATE_RPS,
                burst: FETCH_RATE_BURST,
                max_inflight: 0,
                max_tracked: FETCH_RATE_MAX_TRACKED,
            })),
            streams: Mutex::new(HashMap::new()),
            policy: PolicyWatcher::r#static(SharePolicy::share_all()),
        }
    }

    /// Restrict which models this provider shares (announces + serves) via a **static** legacy
    /// list (the CLI `--share-models` path). An empty list means "share everything"; a non-empty
    /// list shares only those models. This path does not hot-reload — use
    /// [`with_share_policy_file`](Self::with_share_policy_file) for the runtime-toggleable policy.
    pub fn with_shared_models<I, S>(mut self, models: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.policy = PolicyWatcher::from_legacy_list(models);
        self
    }

    /// Back this provider's share policy with a JSON file at `path`, hot-reloaded whenever the file
    /// changes (the desktop rewrites it when the user toggles a model). Loads the current contents
    /// now; a missing or malformed file falls back to share-everything with a warning, so the node
    /// never silently shares nothing on a first-run/parse hiccup — the desktop writes a valid file
    /// before launch in the normal flow.
    pub fn with_share_policy_file(mut self, path: PathBuf) -> Self {
        self.policy = PolicyWatcher::from_file(path);
        self
    }

    /// Whether `model_ref` (an engine handle) is shared under the current (possibly hot-reloaded)
    /// policy. Fail-closed (share-nothing) on a poisoned lock — matters for the serve gate.
    fn model_shared(&self, model_ref: &str) -> bool {
        self.policy.is_shared(model_ref)
    }

    /// Attach shared transfer counters (P0 introspection — the `--status-bind` server
    /// reads them; the serve path updates them once per request).
    pub fn with_stats(mut self, stats: Arc<TransferStats>) -> Self {
        self.stats = Some(stats);
        self
    }

    /// Set the advisory host/port advertised in records (routing is via libp2p regardless).
    pub fn with_address(mut self, host: impl Into<String>, port: u16) -> Self {
        self.host = host.into();
        self.port = port;
        self
    }

    /// Attach an acceptable-use policy (AUP floor) applied to inbound serve requests. Default
    /// is permissive (allow everything).
    pub fn with_aup(mut self, aup: AupPolicy) -> Self {
        self.aup = aup;
        self
    }

    /// AUP floor: if `data` is a serve request the policy refuses, return the framed `Error`
    /// to send back **instead of** serving it; otherwise `None` (serve normally). Receipts and
    /// unparseable frames are not policy-checked here — they fall through to `dispatch`.
    fn aup_refusal(&self, req: Option<&ServeRequest>) -> Option<Vec<u8>> {
        if !self.aup.is_active() {
            return None;
        }
        let req = req?; // non-serve / unparseable → nothing to police (dispatch handles it)
        match self.aup.evaluate(&req.messages, req.max_tokens) {
            AupDecision::Deny(reason) => Some(frame_response(&[ServeChunk::Error(format!(
                "rejected by acceptable-use policy: {reason}"
            ))
            .encode()])),
            AupDecision::Allow => None,
        }
    }

    /// B-S1: record what we just served under the consumer's committed nonce, so the
    /// settlement receipt can be validated against it. Prunes expired commitments and
    /// enforces a hard cap so the map stays bounded (B-S7).
    fn record_commitment(&self, nonce: [u8; 16], model_id: String, tokens: u64) {
        let now = now_unix_ms();
        let mut map = self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, c| now.saturating_sub(c.recorded_ms) <= COMMITMENT_TTL_MS);
        if map.len() >= MAX_COMMITMENTS {
            return; // backstop — refuse to grow past the cap (stale entries already pruned)
        }
        map.insert(nonce, ServeCommitment { model_id, tokens, recorded_ms: now });
    }

    /// Reconnect-and-fetch: mark a serve in-flight under `nonce` so a fetch that arrives
    /// while it's still generating gets `Generating` (retry) rather than `NotFound`
    /// (re-serve) — which would double-run the same request.
    fn mark_generating(&self, nonce: [u8; 16], reply_to: &str) {
        let now = now_unix_ms();
        let mut map = self.results.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, r| now.saturating_sub(r.recorded_ms) <= RESULT_TTL_MS);
        enforce_result_caps(&mut map);
        map.insert(
            nonce,
            BufferedResult {
                reply_to: reply_to.to_string(),
                state: ResultState::Generating,
                bytes: 0,
                recorded_ms: now,
            },
        );
    }

    /// Reconnect-and-fetch: buffer the completed serve frames under `nonce`, bound to the
    /// consumer's `reply_to`, so a reconnecting consumer can fetch them. Called right after
    /// the serve completes (ok or error — the consumer gets whatever was produced).
    fn store_result(&self, nonce: [u8; 16], reply_to: &str, frames: &[Vec<u8>]) {
        let now = now_unix_ms();
        let bytes: usize = frames.iter().map(|c| c.len()).sum();
        let mut map = self.results.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, r| now.saturating_sub(r.recorded_ms) <= RESULT_TTL_MS);
        map.insert(
            nonce,
            BufferedResult {
                reply_to: reply_to.to_string(),
                state: ResultState::Ready(frames.to_vec()),
                bytes,
                recorded_ms: now,
            },
        );
        enforce_result_caps(&mut map);
    }

    /// Handle a [`FETCH_RESULT`] request: return the buffered result for the nonce in
    /// `payload` (`0x11 ‖ nonce(16)` → this is `payload` = the 16 nonce bytes). `source_peer`
    /// is the libp2p-authenticated sender; only the consumer that committed the nonce
    /// (`reply_to`) may fetch it.
    fn handle_fetch(&self, source_peer: &str, payload: &[u8]) -> Vec<u8> {
        // E-S8: shed a fetch flood, keyed on the authenticated sender. Throttled → "retry"
        // (Generating), so an honest reconnecting consumer just backs off.
        if self.fetch_rl.try_acquire(source_peer, now_unix_ms()).is_err() {
            return FetchResponse::Generating.encode();
        }
        let nonce: [u8; 16] = match payload.try_into() {
            Ok(n) => n,
            Err(_) => return FetchResponse::NotFound.encode(),
        };
        let now = now_unix_ms();
        let mut map = self.results.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, r| now.saturating_sub(r.recorded_ms) <= RESULT_TTL_MS);
        fetch_from_buffer(&map, &nonce, source_peer).encode()
    }

    /// P1 streaming: admit a new stream under `nonce` (empty, `done=false`). Called at the start
    /// of an accepted `SERVE_STREAM` generation job, so the buffer exists before the ack is sent
    /// and never lingers without a producer (a shed submit never reaches this).
    fn stream_begin(&self, nonce: [u8; 16], reply_to: &str) {
        let now = now_unix_ms();
        let mut map = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        stream_begin_in(&mut map, nonce, reply_to, now);
    }

    /// P1 streaming: append one encoded frame to the stream (a delta, or the terminal frame which
    /// also flips `done`). No-op if the buffer was reaped meanwhile — the consumer re-serves on
    /// the resulting `NotFound`.
    fn stream_append(&self, nonce: [u8; 16], encoded: &[u8]) {
        let now = now_unix_ms();
        let mut map = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        stream_append_in(&mut map, &nonce, encoded, now);
    }

    /// Handle a [`FETCH_CHUNKS`] poll: `payload` = `nonce(16) ‖ offset(4, big-endian)`. Returns
    /// the incremental slice from `offset`, pruning the acknowledged prefix. `source_peer` is the
    /// libp2p-authenticated sender; only the committing consumer (`reply_to`) may poll. Fast and
    /// **non-blocking** — never long-polls; the consumer paces itself.
    fn handle_fetch_chunks(&self, source_peer: &str, payload: &[u8]) -> Vec<u8> {
        // Parse `nonce ‖ offset`. A malformed poll → NotFound (the consumer re-serves).
        if payload.len() != 20 {
            return FetchChunksResponse::NotFound.encode();
        }
        let nonce: [u8; 16] = payload[0..16].try_into().unwrap();
        let offset = u32::from_be_bytes(payload[16..20].try_into().unwrap());
        // E-S8: shed a poll flood, keyed on the authenticated sender. A shed poll returns
        // "no progress at your offset" so an honest consumer simply backs off and re-polls the
        // same offset — it never re-serves or fails over on a transient rate-limit.
        if self.fetch_rl.try_acquire(source_peer, now_unix_ms()).is_err() {
            return FetchChunksResponse::Chunks { framed: Vec::new(), next_offset: offset, done: false }
                .encode();
        }
        let now = now_unix_ms();
        let mut map = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, b| now.saturating_sub(b.recorded_ms) <= STREAM_TTL_MS);
        fetch_chunks_from(&mut map, &nonce, source_peer, offset, now).encode()
    }

    /// P1 streaming: the single pool job for a `SERVE_STREAM`. It gates (throttle/AUP/share)
    /// exactly like the buffered path, then **responds with the immediate ack early** and keeps
    /// running to generate — so the one inbound request is answered at once while the tokens
    /// stream into the buffer for the consumer's `FETCH_CHUNKS` polls. Held on one pool worker for
    /// the generation's duration, so concurrent generations stay bounded by the pool (engine
    /// capacity), same as the buffered path.
    fn run_stream_job(
        self: Arc<Self>,
        request_id: String,
        source_peer: String,
        data: Vec<u8>,
        max_concurrency: usize,
    ) {
        // A stream reply is always a `FetchChunksResponse`. A refusal (bad request / AUP / not
        // shared) is delivered as a *complete* stream: an ack carrying the framed `Error` with
        // `done=true`, so the consumer's poll loop sees the error and stops without a second poll.
        let respond_done = |framed: Vec<u8>, next_offset: u32| {
            let _ = self
                .net
                .respond(request_id.clone(), FetchChunksResponse::Chunks { framed, next_offset, done: true }.encode());
        };
        let req = match ServeRequest::decode(&data[1..]) {
            Ok(r) => r,
            Err(e) => {
                let framed = frame_response(&[ServeChunk::Error(format!("bad serve request: {e}")).encode()]);
                respond_done(framed, 1);
                return;
            }
        };
        // F1: `reply_to` is where the result is delivered and the key the per-peer stream-fairness
        // cap is bucketed on. It MUST be the libp2p-authenticated sender — otherwise a peer could
        // open streams with `reply_to = <victim>`, filling the victim's fair-share and evicting the
        // victim's real streams (a targeted griefing DoS). A legitimate consumer always sets
        // `reply_to = its own peer id`, which equals `source_peer` on every path (direct, relayed,
        // reversal, local self-serve). Reject a mismatch as a done-error stream (no buffer created).
        if !reply_to_authorized(&req.reply_to, &source_peer) {
            tracing::warn!(
                reply_to = %req.reply_to, source = %source_peer,
                "serve_stream: reply_to does not match the authenticated sender — rejecting"
            );
            let framed = frame_response(&[
                ServeChunk::Error("reply_to must equal the authenticated sender".into()).encode(),
            ]);
            respond_done(framed, 1);
            return;
        }
        // M2.3: throttle a leecher first (off the poll thread, budget-capped). Contributors pass.
        self.maybe_throttle(Some(&req), max_concurrency);
        // AUP floor: a policy-violating request is refused without touching the engine.
        if let Some(refusal) = self.aup_refusal(Some(&req)) {
            if let Some(stats) = &self.stats {
                stats.aup_refusals.fetch_add(1, Ordering::Relaxed);
            }
            respond_done(refusal, 1);
            return;
        }
        // Per-model share gate: the real enforcement point (a consumer could learn the id via
        // cache/PEX/guess), refused before the engine.
        if !self.model_shared(&req.model_ref) {
            let framed = frame_response(&[ServeChunk::Error(format!(
                "model '{}' is not shared by this provider",
                req.model_ref
            ))
            .encode()]);
            respond_done(framed, 1);
            return;
        }
        // Capture the small fields needed after `req` is moved into the serve (which consumes
        // `messages`), for the ack, buffer, commitment, and stats.
        let nonce = req.nonce;
        let model_ref = req.model_ref.clone();
        let reply_to = req.reply_to.clone();
        // Create the buffer, then ack immediately (before generating), so the consumer starts
        // polling right away and never races an absent buffer (begin happens before the ack).
        self.stream_begin(nonce, &reply_to);
        let _ = self.net.respond(
            request_id,
            FetchChunksResponse::Chunks { framed: Vec::new(), next_offset: 0, done: false }.encode(),
        );
        // Generate, appending each frame (deltas, then the terminal Done/Error which flips `done`).
        // F5: `generate_guarded` catches a panic in the adapter/engine and appends a synthetic
        // terminal Error, so a bug can't leave this buffer `done=false` forever (which would stall
        // the consumer ~120s before it re-serves).
        let summary =
            generate_guarded(req, &self.adapter, &mut |c| self.stream_append(nonce, c));
        // B-S1: record what we served under the committed nonce so its settlement receipt can be
        // validated (right tokens/model, once) — identical to the buffered path.
        if summary.ok && summary.tokens > 0 {
            self.record_commitment(nonce, model_ref.clone(), summary.tokens);
        }
        if let Some(stats) = &self.stats {
            let e = &summary.metrics.engine;
            let native_tps = if e.eval_duration_ns > 0 {
                e.eval_count as f64 / (e.eval_duration_ns as f64 / 1e9)
            } else {
                0.0
            };
            stats.record_serve(&model_ref, summary.tokens, native_tps, summary.ok);
        }
    }

    /// B-S1/B-S7: validate a settlement receipt against the serve we recorded under its
    /// nonce, and consume the commitment (single-use). `Err(reason)` rejects the receipt
    /// *before* it is co-signed — closing the token-inflation hole.
    fn check_and_consume_commitment(
        &self,
        p: &openhydra_protocol::receipts::ReceiptPayload,
    ) -> Result<(), String> {
        let mut map = self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner());
        validate_and_consume_commitment(
            &mut map,
            &p.nonce,
            &p.model_id,
            p.tokens,
            p.ts_unix_ms,
            now_unix_ms(),
        )
    }

    /// Attach a ledger so accepted co-signed receipts are persisted (M2.3), and rehydrate
    /// the give/take credit map from it so throttling survives a restart.
    pub fn with_store(self, store: Store) -> Self {
        if let Ok(mut map) = self.credit.lock() {
            if let Err(e) = store.load_credit_into_memory(&mut map) {
                eprintln!("openhydra-agent: could not rehydrate credit from store: {e}");
            }
        }
        Self { store: Some(store), ..self }
    }

    /// Accrue a consumer's give/take after it co-signs a receipt: it **consumed**
    /// `tokens` from us (M2.3). Keyed by the consumer's libp2p id (derived from the
    /// receipt's ed25519 pubkey), so it matches the `reply_to` seen at serve time. The
    /// snapshot is flushed durably. Best-effort — never affects the reply.
    fn record_consumption(&self, accepted: &CoSignedReceipt) {
        let now = now_unix_ms();
        let consumer_id = match self
            .net
            .peer_id_from_ed25519_pubkey(accepted.payload.consumer.as_bytes())
        {
            Ok(id) => id,
            Err(_) => return,
        };
        let snapshot = {
            let mut map = match self.credit.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            let acct = map.entry(consumer_id.clone()).or_insert_with(|| CreditAccount::new(now));
            acct.record_consumed(accepted.payload.tokens, now);
            acct.to_bytes()
        };
        if let Some(store) = &self.store {
            if let Err(e) = store.put_credit(&consumer_id, &snapshot) {
                eprintln!("openhydra-agent: credit persist failed: {e}");
            }
        }
    }

    /// The current serve-rate cap in `[credit::RATE_FLOOR, 1.0]` for a consumer (by its
    /// libp2p id), from its give/take balance (M2.3). `1.0` when we hold no record (the
    /// starter grant). The throttle that *consults* this lands with the provider
    /// concurrency model (a per-request delay in today's serial loop would head-of-line
    /// block other consumers — see the M2.3 increment-2 note).
    pub fn consumer_rate_cap(&self, consumer_libp2p_id: &str) -> f64 {
        let now = now_unix_ms();
        self.credit
            .lock()
            .ok()
            .and_then(|m| m.get(consumer_libp2p_id).map(|a| a.rate_cap(now)))
            .unwrap_or(1.0)
    }

    /// Snapshot the provider's take-side economy for the status endpoint: per-consumer
    /// (libp2p-keyed) give/take balance and the serve-rate cap we currently apply to each.
    /// The provider holds no reputation of its own (that's the consumers' view of it), so the
    /// reputation list is always empty here. Pure read of the credit map.
    pub fn economy_snapshot(&self) -> (Vec<crate::status::RepEntry>, Vec<crate::status::CreditEntry>) {
        let now = now_unix_ms();
        let credit = self
            .credit
            .lock()
            .map(|m| {
                m.iter()
                    .map(|(id, a)| crate::status::CreditEntry {
                        libp2p_peer_id: id.clone(),
                        balance: (a.balance(now) * 10.0).round() / 10.0,
                        rate_cap: Some((a.rate_cap(now) * 100.0).round() / 100.0),
                    })
                    .collect()
            })
            .unwrap_or_default();
        (Vec::new(), credit)
    }

    /// M2.3 **enforcement**: before serving, slow a leecher proportionally to its give/take
    /// deficit — *priority, not access* (throttle, never block). Only `SERVE_REQUEST`s are
    /// throttled; a contributor (`rate_cap >= 1.0` → multiplier `0`) and every non-serve
    /// method pass straight through.
    ///
    /// Runs on a worker thread (post-pool, so the delay never blocks the poll loop), and is
    /// **budget-capped**: it only sleeps while fewer than `max_concurrency - 1` workers are
    /// already throttling, so at least one worker always stays free to serve immediately.
    /// A flood of leechers therefore can't put the whole pool to sleep and stall an arriving
    /// contributor — the soft throttle simply stops biting once the budget is spent.
    fn maybe_throttle(&self, req: Option<&ServeRequest>, max_concurrency: usize) {
        // Receipts, unknown methods, and unparseable serves are never throttled (the last
        // falls through to dispatch's error frame). F-C5: `req` is decoded once upstream.
        let Some(req) = req else { return };
        let mult = throttle_multiplier(self.consumer_rate_cap(&req.reply_to));
        if mult <= 0.0 {
            return; // contributor — full speed
        }
        // Reserve ≥1 non-throttling worker: only sleep if we are not the one that would
        // exhaust the pool's throttle budget.
        let already = self.throttling.fetch_add(1, Ordering::SeqCst);
        if already + 1 < max_concurrency {
            std::thread::sleep(BASE_THROTTLE.mul_f64(mult));
        }
        self.throttling.fetch_sub(1, Ordering::SeqCst);
    }

    /// Detect the engine's models and announce a record for each. Returns how many were
    /// announced.
    pub fn announce_models(&self) -> Result<usize, AdapterError> {
        let models = self.adapter.detect_models()?;
        // Snapshot the policy ONCE so the loop filter and the published view see one consistent
        // policy (also the source for `shared_models` below).
        let policy = self.policy.snapshot();
        let mut announced: Vec<String> = Vec::new();
        let mut outcome: Result<(), AdapterError> = Ok(());
        for model in &models {
            // Per-model share policy: skip models the operator hasn't opted to share. The serve
            // path enforces the same gate, so this is discovery hygiene, not the security boundary.
            if !policy.is_shared(&model.engine_ref) {
                continue;
            }
            let record = build_peer_record(
                model,
                self.net.openhydra_peer_id(),
                self.net.libp2p_peer_id(),
                self.net.public_key_hex(),
                &self.host,
                self.port,
            );
            if let Err(e) = self.net.announce(record) {
                outcome = Err(AdapterError::Http(format!("announce: {e}")));
                break;
            }
            announced.push(model.engine_ref.clone());
        }
        // Publish the status view for the UI (L6): the `shared_models` = the loaded policy's *intent*
        // (independent of whether the network announce succeeded), and `announced_models` = the subset
        // actually announced this pass — published even on a mid-loop announce error, so the UI can
        // reconcile the toggle and honestly show an un-announceable model as "pending" rather than
        // bouncing. Returns the announce error (if any) after publishing.
        let count = announced.len();
        if let Some(stats) = &self.stats {
            stats.publish_share(crate::share_policy::ShareStatusView {
                share_mode: policy.mode,
                shared_models: policy.models.iter().cloned().collect(),
                announced_models: announced,
            });
        }
        outcome.map(|()| count)
    }

    /// Blocking serve loop: poll inbound requests, serve them, reply, and **periodically
    /// re-announce** our models. Runs until the process exits (call on a dedicated thread).
    ///
    /// Polls in `poll_timeout` slices so it stays responsive; a failed reply is
    /// logged-by-return and the loop continues. Every `reannounce_every` it re-publishes
    /// the DHT records: the bootstrap relays expire provider records on a short TTL
    /// (300s today), and the very first announce at startup can race an empty routing
    /// table — so a one-shot announce silently vanishes. Re-announcing on an interval
    /// shorter than the TTL keeps the node discoverable.
    /// `max_concurrency` bounds in-flight serves: the poll loop hands each request to a
    /// [`WorkerPool`] of that many threads instead of serving inline, so a long generation
    /// (or an M2.3 throttle delay) on one request no longer blocks the others. Re-announce
    /// still runs on the poll thread between polls.
    pub fn run_inbound(
        self: Arc<Self>,
        poll_timeout: std::time::Duration,
        reannounce_every: std::time::Duration,
        max_concurrency: usize,
    ) -> !
    where
        A: Send + Sync + 'static,
    {
        let pool = WorkerPool::new(max_concurrency);
        let mut last_announce = std::time::Instant::now();
        // #42: track the network generation. The event loop bumps it whenever it
        // rebuilds connectivity after a network change (roam / wake / interface
        // up-down); when it advances we re-announce immediately so the DHT record
        // carries the new relay addresses under the same pinned PeerId, instead
        // of waiting out the periodic `reannounce_every` interval.
        let mut last_generation = self.net.network_generation();
        loop {
            if let Some((request_id, source_peer, data)) = self.net.poll_inbound(poll_timeout) {
                // P1 streaming: a SERVE_STREAM job acks early then keeps running to generate, so
                // it needs its own body (it responds itself, mid-job, rather than returning one
                // buffered blob). Everything else (SERVE_REQUEST / FETCH_RESULT / FETCH_CHUNKS /
                // receipts) takes the generic dispatch path below.
                if data.first() == Some(&SERVE_STREAM) {
                    let provider = Arc::clone(&self);
                    let shed_id = request_id.clone();
                    // F1: `source_peer` is the libp2p-authenticated sender — bound against the
                    // request's `reply_to` inside the job so a peer can't attribute streams to
                    // (and evict) a victim via a spoofed `reply_to`.
                    let src = source_peer.clone();
                    let accepted = pool.submit(move || {
                        provider.run_stream_job(request_id, src, data, max_concurrency);
                    });
                    if !accepted {
                        // A3: pool full → shed at submit. No buffer was created (begin runs inside
                        // the job), so there's no orphan `Generating` stream. The consumer fails
                        // over to another provider on the explicit `Overloaded` ack.
                        let _ = self.net.respond(shed_id, FetchChunksResponse::Overloaded.encode());
                    }
                    continue;
                }
                // Serve off the poll thread so the loop can keep accepting requests and a
                // slow one doesn't head-of-line-block the rest.
                let provider = Arc::clone(&self);
                let shed_id = request_id.clone();
                let accepted = pool.submit(move || {
                    // F-C5: decode the serve request ONCE here and thread it through
                    // throttle / AUP / dispatch, instead of each re-decoding the (possibly
                    // large) prompt. `None` for non-serve or unparseable frames.
                    let parsed: Option<ServeRequest> = if data.first() == Some(&SERVE_REQUEST) {
                        ServeRequest::decode(&data[1..]).ok()
                    } else {
                        None
                    };
                    // M2.3: throttle a leecher first (off the poll thread, budget-capped so
                    // it never stalls the pool); contributors pass straight through.
                    provider.maybe_throttle(parsed.as_ref(), max_concurrency);
                    // AUP floor: refuse a policy-violating serve request without running it.
                    let response = match provider.aup_refusal(parsed.as_ref()) {
                        Some(refusal) => {
                            if let Some(stats) = &provider.stats {
                                stats.aup_refusals.fetch_add(1, Ordering::Relaxed);
                            }
                            refusal
                        }
                        // E-S8: the libp2p-authenticated sender keys the per-peer receipt
                        // rate-limit, so one abusive peer can't shed everyone's settlements.
                        None => provider.dispatch(&source_peer, &data, parsed),
                    };
                    // Best-effort reply; if the swarm is gone the next poll will error too.
                    let _ = provider.net.respond(request_id, response);
                });
                if !accepted {
                    // A3: the bounded serve queue is full — shed with a retryable error so
                    // the consumer fails fast instead of waiting out its request timeout,
                    // and provider memory stays bounded under a burst.
                    // See docs/CODEBASE_HARDENING_PLAN.md (A3).
                    let busy = frame_response(&[
                        ServeChunk::Error("provider overloaded, retry".into()).encode(),
                    ]);
                    let _ = self.net.respond(shed_id, busy);
                }
            }
            // Hot-reload the share policy if the desktop rewrote it → re-announce now so a toggled-on
            // model is advertised within one poll slice (<1s), and a toggled-off one stops being
            // refreshed (its record ages out within the provider-record TTL; the serve gate refuses
            // it immediately). Cheap: one `stat` per poll, parse+swap only on an actual change.
            let policy_changed = self.policy.reload_if_changed();
            if policy_changed {
                eprintln!("openhydra-agent: share policy changed — re-announcing");
            }
            // #42: a network change → re-announce now (not on the slow interval).
            let generation = self.net.network_generation();
            let network_changed = generation != last_generation;
            if network_changed {
                last_generation = generation;
                eprintln!(
                    "openhydra-agent: network change (generation {generation}) — re-announcing"
                );
            }
            if policy_changed || network_changed || last_announce.elapsed() >= reannounce_every {
                match self.announce_models() {
                    Ok(n) => eprintln!("openhydra-agent: re-announced {n} model(s)"),
                    Err(e) => eprintln!("openhydra-agent: re-announce failed: {e}"),
                }
                last_announce = std::time::Instant::now();
            }
        }
    }

    /// Route one inbound request by its method byte: serve completions, settle receipts.
    /// `source_peer` is the libp2p-authenticated sender, used to rate-limit per peer.
    /// `parsed` is the already-decoded serve request (F-C5), `None` for non-serve frames.
    fn dispatch(&self, source_peer: &str, data: &[u8], parsed: Option<ServeRequest>) -> Vec<u8> {
        match data.first() {
            Some(&RECEIPT_REQUEST) => {
                // E-S8: shed a receipt flood before spending any crypto, keyed on the
                // authenticated sender so one abusive peer can't starve others' receipts.
                // The guard drops immediately — we use the bucket purely as a rate cap
                // (concurrency is already bounded by the worker pool).
                if self.receipt_rl.try_acquire(source_peer, now_unix_ms()).is_err() {
                    return frame_response(&[
                        ServeChunk::Error("receipt rate limited, retry".into()).encode(),
                    ]);
                }
                let sign = |msg: &[u8]| self.net.sign(msg).unwrap_or_default();
                let provider_pub = self.net.public_key_bytes().unwrap_or_default();
                // B-S1: only co-sign a receipt that settles a real serve we recorded.
                let bind = |p: &openhydra_protocol::receipts::ReceiptPayload| {
                    self.check_and_consume_commitment(p)
                };
                let (response, accepted) =
                    handle_receipt_inbound(data, &sign, &provider_pub, &bind);
                // Persist the accepted co-signed receipt (best-effort; never fails the reply).
                ledger_receipt(self.store.as_ref(), accepted.as_ref());
                // M2.3: accrue the consumer's give/take (it consumed `tokens` from us).
                if let Some(receipt) = &accepted {
                    self.record_consumption(receipt);
                    if let Some(stats) = &self.stats {
                        stats.receipts_ledgered.fetch_add(1, Ordering::Relaxed);
                        // #5 Ledger: a co-signed receipt is an authoritative "served" transaction
                        // (right model/tokens, counterparty = the consumer we served).
                        stats.record_ledger(
                            now_unix_ms(),
                            "served",
                            &receipt.payload.model_id,
                            source_peer,
                            receipt.payload.tokens,
                        );
                    }
                    // Durable Ledger row so the view + lifetime totals survive a restart
                    // (rehydrated on boot). Best-effort — never fails the settlement reply.
                    if let Some(store) = &self.store {
                        let _ = store.append_ledger_row(&openhydra_protocol::store::LedgerEntry {
                            ts_ms: now_unix_ms(),
                            kind: "served".to_string(),
                            model: receipt.payload.model_id.clone(),
                            counterparty: source_peer.to_string(),
                            tokens: receipt.payload.tokens,
                        });
                    }
                }
                response
            }
            // A serve request: run it, and — when the status endpoint is on — fold the
            // outcome into the shared transfer counters (per-model tokens + native TPS).
            Some(&SERVE_REQUEST) => {
                let mut chunks: Vec<Vec<u8>> = Vec::new();
                let summary = match parsed {
                    Some(req) => {
                        // Per-model share gate: refuse a request for a model the operator hasn't
                        // shared, before touching the engine. This is the real enforcement point —
                        // discovery filtering alone wouldn't stop a consumer that learned the id
                        // some other way (cache, PEX, a direct guess).
                        if !self.model_shared(&req.model_ref) {
                            return frame_response(&[ServeChunk::Error(format!(
                                "model '{}' is not shared by this provider",
                                req.model_ref
                            ))
                            .encode()]);
                        }
                        // Capture the small fields we still need after `req` is moved into
                        // the serve (which consumes `messages`), for commitment + stats +
                        // reconnect-and-fetch buffering.
                        let nonce = req.nonce;
                        let model_ref = req.model_ref.clone();
                        let reply_to = req.reply_to.clone();
                        // Reconnect-and-fetch: mark in-flight so a fetch racing this serve gets
                        // Generating (retry), not NotFound (which would re-run the request).
                        self.mark_generating(nonce, &reply_to);
                        // F5 (buffered twin): guard the generation so a panic in the adapter/engine
                        // becomes a terminal Error frame instead of unwinding out of `dispatch`
                        // (which would skip `respond` + `store_result`, leaving no reply and the
                        // nonce stuck `Generating` until the consumer's serve timeout). The panic
                        // path pushes the terminal into `chunks`, which is stored + returned below.
                        let summary = generate_guarded(
                            req,
                            &self.adapter,
                            &mut |c| chunks.push(c.to_vec()),
                        );
                        // Reconnect-and-fetch: buffer the produced frames under the nonce so a
                        // consumer whose connection dropped can fetch them on a fresh circuit.
                        self.store_result(nonce, &reply_to, &chunks);
                        // B-S1: record what we served under the consumer's committed nonce so
                        // its settlement receipt can be validated (right tokens/model, once).
                        if summary.ok && summary.tokens > 0 {
                            self.record_commitment(nonce, model_ref.clone(), summary.tokens);
                        }
                        if let Some(stats) = &self.stats {
                            let e = &summary.metrics.engine;
                            let native_tps = if e.eval_duration_ns > 0 {
                                e.eval_count as f64 / (e.eval_duration_ns as f64 / 1e9)
                            } else {
                                0.0
                            };
                            stats.record_serve(&model_ref, summary.tokens, native_tps, summary.ok);
                        }
                        summary
                    }
                    // Undecodable serve request → emit the framed "bad serve request" error.
                    None => handle_serve_request(&data[1..], &self.adapter, &mut |c| {
                        chunks.push(c.to_vec())
                    }),
                };
                let _ = summary;
                frame_response(&chunks)
            }
            // Reconnect-and-fetch: return the buffered result for a nonce (drop recovery).
            Some(&FETCH_RESULT) => self.handle_fetch(source_peer, &data[1..]),
            // P1 streaming: an incremental chunk poll (short, non-blocking). `SERVE_STREAM` is
            // handled earlier in `run_inbound` (it acks early then keeps generating); only the
            // poll reaches `dispatch`.
            Some(&FETCH_CHUNKS) => self.handle_fetch_chunks(source_peer, &data[1..]),
            // Unknown method byte → a framed Error from the serve handler.
            _ => handle_serve_inbound(data, &self.adapter),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{InferenceRequest, ServeOutcome};
    use crate::serve::{parse_response, ServeChunk, ServeRequest};

    fn detected(canonical: &str) -> DetectedModel {
        DetectedModel {
            engine_ref: "qwen2.5:7b".into(),
            canonical_id: canonical.into(),
            family: "qwen2".into(),
            params: "7.6b".into(),
            quant: "q4_k_m".into(),
            size_bytes: 42,
        }
    }

    #[test]
    fn peer_record_keys_on_engine_handle_and_carries_canonical_id() {
        let r = build_peer_record(
            &detected("qwen2/7.6b/q4_k_m/abcd0123abcd0123"),
            "oh-peer",
            "12D3KooWlibp2p",
            "deadbeef",
            "10.0.0.5",
            4001,
        );
        assert_eq!(r.peer_id, "oh-peer");
        assert_eq!(r.model_id, "qwen2.5:7b"); // engine handle is the DHT key consumers request
        assert_eq!(r.canonical_model_id, "qwen2/7.6b/q4_k_m/abcd0123abcd0123"); // precise, for filtering
        assert_eq!(r.libp2p_peer_id, "12D3KooWlibp2p");
        assert_eq!(r.public_key, "deadbeef");
        assert_eq!(r.port, 4001);
    }

    #[test]
    fn peer_record_keys_on_engine_handle_even_without_canonical_id() {
        let r = build_peer_record(&detected(""), "oh", "lib", "pk", "", 0);
        assert_eq!(r.model_id, "qwen2.5:7b"); // always the engine handle
        assert_eq!(r.canonical_model_id, "");
    }

    // ── per-model share policy ──
    // The policy logic (modes, migration, load/save, hot-reload semantics) is covered in
    // `crate::share_policy`. This test pins the provider-specific invariant: the string the policy
    // gates on is the *same* string that gets announced as the record's `model_id` — so an
    // allowlisted model is announced+servable and a de-selected sibling on the same node is not.

    #[test]
    fn share_gate_key_equals_the_announced_model_id() {
        let policy = SharePolicy::share_list(["qwen2.5:7b"]);
        let served = build_peer_record(&detected("q/7b/x/y"), "oh", "lib", "pk", "", 0);
        assert!(policy.is_shared(&served.model_id)); // announced id == gate key → shared
        assert!(!policy.is_shared("qwen2.5:0.5b")); // a different handle on the same node → refused
    }

    struct StubAdapter;
    impl EngineAdapter for StubAdapter {
        fn engine_name(&self) -> &'static str {
            "stub"
        }
        fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
            Ok(vec![])
        }
        fn serve_stream(
            &self,
            _req: &InferenceRequest,
            on_delta: &mut dyn FnMut(&str),
        ) -> Result<ServeOutcome, AdapterError> {
            on_delta("Hello");
            on_delta(" world");
            Ok(ServeOutcome { tokens: 5, done: true, engine: Default::default(), tool_calls: Vec::new() })
        }
    }

    fn serve_request_bytes() -> Vec<u8> {
        let req = ServeRequest {
            reply_to: "12D3KooWConsumer".into(),
            model_ref: "qwen2.5:7b".into(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
            tools: Vec::new(),
            nonce: [0u8; 16],
        };
        let mut data = vec![SERVE_REQUEST];
        data.extend_from_slice(&req.encode());
        data
    }

    #[test]
    fn serves_an_inbound_request_into_a_framed_completion() {
        let response = handle_serve_inbound(&serve_request_bytes(), &StubAdapter);
        let chunks = parse_response(&response).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], ServeChunk::Delta("Hello".into()));
        assert_eq!(chunks[1], ServeChunk::Delta(" world".into()));
        assert!(matches!(chunks[2], ServeChunk::Done { tokens: 5, .. }));
    }

    #[test]
    fn rejects_unknown_method_byte() {
        let response = handle_serve_inbound(&[0xEE, 1, 2, 3], &StubAdapter);
        let chunks = parse_response(&response).unwrap();
        assert!(matches!(chunks.as_slice(), [ServeChunk::Error(_)]));
    }

    fn receipt_limiter() -> Arc<RateLimiter> {
        // The exact config wired onto Provider.receipt_rl, so a bad constant
        // (e.g. burst=0 → sheds every receipt) is caught here.
        Arc::new(RateLimiter::new(RateLimitConfig {
            rps: RECEIPT_RATE_RPS,
            burst: RECEIPT_RATE_BURST,
            max_inflight: 0,
            max_tracked: RECEIPT_RATE_MAX_TRACKED,
        }))
    }

    #[test]
    fn es8_receipt_limiter_admits_burst_then_sheds_then_refills() {
        let rl = receipt_limiter();
        let t = 1_000_000u64;
        for _ in 0..(RECEIPT_RATE_BURST as usize) {
            assert!(rl.try_acquire("12D3KooWPeerA", t).is_ok());
        }
        assert!(
            rl.try_acquire("12D3KooWPeerA", t).is_err(),
            "burst exhausted at the same instant → shed before any crypto"
        );
        // ~1s later the sustained rate has refilled tokens → admits again.
        assert!(rl.try_acquire("12D3KooWPeerA", t + 1000).is_ok());
    }

    #[test]
    fn es8_receipt_limiter_is_per_peer_not_global() {
        // A flooding peer exhausting its own bucket must not shed a well-behaved
        // peer's settlement — the whole point of keying on the authenticated sender.
        let rl = receipt_limiter();
        let t = 2_000_000u64;
        for _ in 0..(RECEIPT_RATE_BURST as usize) {
            assert!(rl.try_acquire("12D3KooWFlooder", t).is_ok());
        }
        assert!(rl.try_acquire("12D3KooWFlooder", t).is_err(), "flooder exhausted its bucket");
        assert!(
            rl.try_acquire("12D3KooWHonest", t).is_ok(),
            "an unrelated peer has its own independent bucket"
        );
    }

    fn test_receipt(nonce: [u8; 16], tokens: u64) -> CoSignedReceipt {
        use openhydra_protocol::crypto_agility::SigAlg;
        use openhydra_protocol::receipts::{build_receipt, ReceiptPayload};
        use ed25519_dalek::SigningKey;
        let consumer = SigningKey::from_bytes(&[3u8; 32]);
        let provider = SigningKey::from_bytes(&[5u8; 32]);
        let payload = ReceiptPayload {
            sig_alg: SigAlg::Ed25519,
            provider: provider.verifying_key(),
            consumer: consumer.verifying_key(),
            model_id: "qwen2.5/7b/q4_k_m/abcd0123abcd0123".to_string(),
            tokens,
            nonce,
            ts_unix_ms: 1_700_000_000_000,
        };
        build_receipt(payload, &consumer, &provider)
    }

    #[test]
    fn ledger_receipt_persists_when_store_present_and_is_replay_safe() {
        let store = Store::open_in_memory().unwrap();
        let receipt = test_receipt([11u8; 16], 128);

        ledger_receipt(Some(&store), Some(&receipt));
        assert_eq!(store.receipt_count().unwrap(), 1);

        // Replaying the same receipt must not add a second row.
        ledger_receipt(Some(&store), Some(&receipt));
        assert_eq!(store.receipt_count().unwrap(), 1);
    }

    #[test]
    fn ledger_receipt_is_a_noop_without_a_store_or_receipt() {
        let store = Store::open_in_memory().unwrap();
        // No receipt to ledger.
        ledger_receipt(Some(&store), None);
        assert_eq!(store.receipt_count().unwrap(), 0);
        // No store configured → nothing persisted, no panic.
        let receipt = test_receipt([22u8; 16], 64);
        ledger_receipt(None, Some(&receipt));
    }

    // ---- B-S1 / B-S7: serve-commitment binding ---------------------------------------

    fn commit(map: &mut HashMap<[u8; 16], ServeCommitment>, n: [u8; 16], model: &str, tokens: u64, now: u64) {
        map.insert(n, ServeCommitment { model_id: model.into(), tokens, recorded_ms: now });
    }

    #[test]
    fn commitment_accepts_match_once_then_rejects_replay() {
        let now = 1_000_000u64;
        let mut map = HashMap::new();
        commit(&mut map, [7u8; 16], "qwen2.5:7b", 100, now);
        assert!(validate_and_consume_commitment(&mut map, &[7u8; 16], "qwen2.5:7b", 100, now, now).is_ok());
        // Single-use: the commitment is consumed, so a replay of the same receipt is rejected.
        assert!(validate_and_consume_commitment(&mut map, &[7u8; 16], "qwen2.5:7b", 100, now, now)
            .unwrap_err()
            .contains("no serve commitment"));
    }

    #[test]
    fn commitment_allows_underclaim_rejects_overclaim() {
        let now = 1_000_000u64;
        let mut map = HashMap::new();
        commit(&mut map, [8u8; 16], "m", 100, now);
        // Claiming fewer tokens than served is harmless (only reduces the provider's credit).
        assert!(validate_and_consume_commitment(&mut map, &[8u8; 16], "m", 40, now, now).is_ok());
        // The B-S1 attack — claiming more than served — is rejected.
        commit(&mut map, [9u8; 16], "m", 100, now);
        assert!(validate_and_consume_commitment(&mut map, &[9u8; 16], "m", u64::MAX, now, now)
            .unwrap_err()
            .contains("exceed"));
    }

    #[test]
    fn commitment_rejects_unknown_nonce_and_model_mismatch() {
        let now = 1_000_000u64;
        let mut map = HashMap::new();
        assert!(validate_and_consume_commitment(&mut map, &[1u8; 16], "m", 1, now, now)
            .unwrap_err()
            .contains("no serve commitment"));
        // Right nonce, wrong model → reject (no cross-model misattribution).
        commit(&mut map, [2u8; 16], "cheap-model", 100, now);
        assert!(validate_and_consume_commitment(&mut map, &[2u8; 16], "expensive-model", 100, now, now)
            .unwrap_err()
            .contains("model"));
    }

    #[test]
    fn commitment_enforces_freshness_and_prunes_stale() {
        let now = 10 * COMMITMENT_TTL_MS;
        let mut map = HashMap::new();
        commit(&mut map, [3u8; 16], "m", 100, now);
        // Receipt timestamp older than the window, or far in the future → reject (B-S7).
        assert!(validate_and_consume_commitment(&mut map, &[3u8; 16], "m", 100, 0, now)
            .unwrap_err()
            .contains("freshness"));
        assert!(validate_and_consume_commitment(&mut map, &[3u8; 16], "m", 100, now + 10 * CLOCK_SKEW_MS, now)
            .unwrap_err()
            .contains("freshness"));
        // A commitment older than the TTL is pruned on the next validate, bounding the map.
        let mut map2 = HashMap::new();
        commit(&mut map2, [4u8; 16], "m", 100, 0); // recorded at ts 0, now is 10*TTL later
        assert!(validate_and_consume_commitment(&mut map2, &[4u8; 16], "m", 100, now, now)
            .unwrap_err()
            .contains("no serve commitment"));
        assert!(map2.is_empty(), "stale commitment should have been pruned");
    }

    // ── Reconnect-and-fetch ──────────────────────────────────────────────

    fn ready_entry(reply_to: &str, frames: Vec<Vec<u8>>) -> BufferedResult {
        let bytes = frames.iter().map(|c| c.len()).sum();
        BufferedResult { reply_to: reply_to.into(), state: ResultState::Ready(frames), bytes, recorded_ms: 0 }
    }

    #[test]
    fn fetch_returns_buffered_frames_to_the_owner_byte_identical() {
        let frames = vec![
            ServeChunk::Delta("![img](data:image/png;base64,AAAA)".into()).encode(),
            ServeChunk::Done { tokens: 20, metrics: Default::default() }.encode(),
        ];
        let mut map = HashMap::new();
        map.insert([7u8; 16], ready_entry("consumerA", frames.clone()));
        match fetch_from_buffer(&map, &[7u8; 16], "consumerA") {
            FetchResponse::Ready(framed) => {
                // Byte-identical to a fresh serve round-trip → parses to the same chunks.
                assert_eq!(framed, frame_response(&frames));
                let chunks = parse_response(&framed).unwrap();
                assert!(matches!(chunks[0], ServeChunk::Delta(_)));
                assert!(matches!(chunks[1], ServeChunk::Done { tokens: 20, .. }));
            }
            other => panic!("expected Ready, got {other:?}"),
        }
    }

    #[test]
    fn fetch_binds_to_reply_to_and_reports_unknown_nonce() {
        let mut map = HashMap::new();
        map.insert(
            [1u8; 16],
            ready_entry("consumerA", vec![ServeChunk::Done { tokens: 1, metrics: Default::default() }.encode()]),
        );
        // A different authenticated peer cannot fetch someone else's result (ownership binding).
        assert_eq!(fetch_from_buffer(&map, &[1u8; 16], "attackerB"), FetchResponse::Forbidden);
        // An unknown nonce → NotFound (the consumer re-serves).
        assert_eq!(fetch_from_buffer(&map, &[9u8; 16], "consumerA"), FetchResponse::NotFound);
    }

    #[test]
    fn fetch_reports_generating_while_in_flight() {
        let mut map = HashMap::new();
        map.insert(
            [2u8; 16],
            BufferedResult { reply_to: "c".into(), state: ResultState::Generating, bytes: 0, recorded_ms: 0 },
        );
        // A fetch racing an in-flight serve gets Generating (retry), never NotFound (re-run).
        assert_eq!(fetch_from_buffer(&map, &[2u8; 16], "c"), FetchResponse::Generating);
    }

    // ── P1 streaming: trimmable-log buffer ───────────────────────────────

    fn delta(s: &str) -> Vec<u8> {
        ServeChunk::Delta(s.into()).encode()
    }
    fn done(tokens: u64) -> Vec<u8> {
        ServeChunk::Done { tokens, metrics: Default::default() }.encode()
    }

    #[test]
    fn stream_poll_returns_incremental_slice_and_prunes_the_acked_prefix() {
        let mut map = HashMap::new();
        let n = [1u8; 16];
        stream_begin_in(&mut map, n, "consumerA", 0);
        for (i, s) in ["a", "b", "c"].iter().enumerate() {
            stream_append_in(&mut map, &n, &delta(s), 1 + i as u64);
        }
        // Poll from 0: gets all 3, next_offset=3, nothing pruned yet (offset 0 acks nothing).
        match fetch_chunks_from(&mut map, &n, "consumerA", 0, 10) {
            FetchChunksResponse::Chunks { framed, next_offset, done } => {
                assert_eq!((next_offset, done), (3, false));
                assert_eq!(parse_response(&framed).unwrap().len(), 3);
            }
            o => panic!("expected Chunks, got {o:?}"),
        }
        assert_eq!(map[&n].base_offset, 0);
        assert_eq!(map[&n].chunks.len(), 3);

        // Two more frames, then poll @3: gets only the new slice, prunes [0..3), base → 3.
        stream_append_in(&mut map, &n, &delta("d"), 11);
        stream_append_in(&mut map, &n, &delta("e"), 11);
        match fetch_chunks_from(&mut map, &n, "consumerA", 3, 12) {
            FetchChunksResponse::Chunks { framed, next_offset, done } => {
                assert_eq!((next_offset, done), (5, false));
                let chunks = parse_response(&framed).unwrap();
                assert_eq!(chunks, vec![ServeChunk::Delta("d".into()), ServeChunk::Delta("e".into())]);
            }
            o => panic!("expected Chunks, got {o:?}"),
        }
        // Prune-on-ack: absolute offsets survive — base advanced to 3, only the tail remains.
        assert_eq!(map[&n].base_offset, 3);
        assert_eq!(map[&n].chunks.len(), 2);

        // Drop-resume: re-poll the SAME offset returns the same tail; the just-returned data is
        // not pruned until acked by a higher poll, so a transport drop loses nothing.
        match fetch_chunks_from(&mut map, &n, "consumerA", 3, 13) {
            FetchChunksResponse::Chunks { framed, next_offset, .. } => {
                assert_eq!(next_offset, 5);
                assert_eq!(parse_response(&framed).unwrap().len(), 2);
            }
            o => panic!("expected Chunks, got {o:?}"),
        }
        assert_eq!(map[&n].base_offset, 3, "re-poll at the same offset must not advance base");
    }

    #[test]
    fn terminal_frame_flips_done_atomically_and_survives_until_acked() {
        let mut map = HashMap::new();
        let n = [2u8; 16];
        stream_begin_in(&mut map, n, "c", 0);
        stream_append_in(&mut map, &n, &delta("hi"), 1);
        assert!(!map[&n].done, "a delta does not end the stream");
        // The terminal append sets `done` in the same op — a poll never sees done without it.
        stream_append_in(&mut map, &n, &done(1), 1);
        assert!(map[&n].done);
        // Poll @0 delivers delta + Done together with done=true.
        match fetch_chunks_from(&mut map, &n, "c", 0, 2) {
            FetchChunksResponse::Chunks { framed, next_offset, done } => {
                assert_eq!((next_offset, done), (2, true));
                let chunks = parse_response(&framed).unwrap();
                assert!(matches!(chunks.last().unwrap(), ServeChunk::Done { .. }));
            }
            o => panic!("expected Chunks, got {o:?}"),
        }
        // A poll at offset 0 prunes nothing (li=0), so a drop-resume re-poll@0 re-delivers the
        // terminal — it is never dropped before the consumer acks past it.
        match fetch_chunks_from(&mut map, &n, "c", 0, 3) {
            FetchChunksResponse::Chunks { framed, done, .. } => {
                assert!(done);
                assert_eq!(parse_response(&framed).unwrap().len(), 2, "terminal re-delivered on resume");
            }
            o => panic!("expected Chunks, got {o:?}"),
        }
    }

    #[test]
    fn stream_fetch_binds_to_reply_to_and_reports_unknown_nonce() {
        let mut map = HashMap::new();
        let n = [3u8; 16];
        stream_begin_in(&mut map, n, "consumerA", 0);
        stream_append_in(&mut map, &n, &delta("x"), 1);
        // A different authenticated peer cannot poll someone else's stream.
        assert_eq!(fetch_chunks_from(&mut map, &n, "attackerB", 0, 2), FetchChunksResponse::Forbidden);
        // An unknown nonce → NotFound (the consumer re-serves).
        assert_eq!(fetch_chunks_from(&mut map, &[9u8; 16], "consumerA", 0, 2), FetchChunksResponse::NotFound);
    }

    #[test]
    fn stale_low_poll_resumes_at_base_without_re_pruning() {
        let mut map = HashMap::new();
        let n = [4u8; 16];
        stream_begin_in(&mut map, n, "c", 0);
        for s in ["a", "b", "c", "d"] {
            stream_append_in(&mut map, &n, &delta(s), 1);
        }
        // Advance: poll @2 prunes [0..2), base → 2.
        let _ = fetch_chunks_from(&mut map, &n, "c", 2, 2);
        assert_eq!(map[&n].base_offset, 2);
        // A stale/duplicate poll below base clamps to base and returns the live tail, no re-prune.
        match fetch_chunks_from(&mut map, &n, "c", 0, 3) {
            FetchChunksResponse::Chunks { next_offset, framed, .. } => {
                assert_eq!(next_offset, 4);
                assert_eq!(parse_response(&framed).unwrap().len(), 2); // [2..4)
            }
            o => panic!("expected Chunks, got {o:?}"),
        }
        assert_eq!(map[&n].base_offset, 2, "a below-base poll must not move base");
    }

    #[test]
    fn per_peer_cap_evicts_only_the_same_peers_streams() {
        let mut map = HashMap::new();
        // Peer B holds one active stream.
        let bkey = [200u8; 16];
        stream_begin_in(&mut map, bkey, "peerB", 0);
        // Peer A floods well past its per-peer cap.
        let akey = |i: usize| {
            let mut k = [0u8; 16];
            k[..8].copy_from_slice(&(i as u64 + 1).to_le_bytes());
            k
        };
        for i in 0..(MAX_STREAMS_PER_PEER + 20) {
            stream_begin_in(&mut map, akey(i), "peerA", i as u64);
        }
        // Peer A's admissions only ever evict peer A's own oldest — B's active stream survives.
        assert!(map.contains_key(&bkey), "peer A's flood must not evict peer B's active stream");
        let a_count = map.values().filter(|b| b.reply_to == "peerA").count();
        assert!(a_count <= MAX_STREAMS_PER_PEER, "peer A bounded to its fair share, got {a_count}");
    }

    #[test]
    fn scale_sim_thousands_of_streams_no_active_eviction_under_fair_share() {
        // Many honest peers each holding one actively-polled stream, plus one greedy peer opening
        // far more than its share. Assert: (a) every honest stream survives and still returns its
        // data (prune + fairness hold — the greedy peer only evicts its own), (b) the greedy peer
        // is bounded to its per-peer cap, (c) the total stays within the global backstop.
        let mut map = HashMap::new();
        const HONEST: usize = 300; // < MAX_STREAM_ENTRIES so the global backstop never fires here
        let hkey = |i: usize| {
            let mut k = [0u8; 16];
            k[0] = 0xAA;
            k[8..].copy_from_slice(&(i as u64 + 1).to_le_bytes());
            k
        };
        for i in 0..HONEST {
            stream_begin_in(&mut map, hkey(i), &format!("honest-{i}"), i as u64);
            stream_append_in(&mut map, &hkey(i), &delta("tok"), i as u64);
        }
        // Greedy peer opens 1000 streams (all under one reply_to → its per-peer cap bites).
        let gkey = |i: usize| {
            let mut k = [0u8; 16];
            k[0] = 0xBB;
            k[8..].copy_from_slice(&(i as u64 + 1).to_le_bytes());
            k
        };
        for i in 0..1000 {
            stream_begin_in(&mut map, gkey(i), "greedy", 1000 + i as u64);
        }
        // (a) every honest stream survives with its data intact.
        for i in 0..HONEST {
            match fetch_chunks_from(&mut map, &hkey(i), &format!("honest-{i}"), 0, 5000) {
                FetchChunksResponse::Chunks { framed, next_offset, .. } => {
                    assert_eq!(next_offset, 1);
                    assert_eq!(parse_response(&framed).unwrap(), vec![ServeChunk::Delta("tok".into())]);
                }
                o => panic!("honest stream {i} evicted or corrupted: {o:?}"),
            }
        }
        // (b) the greedy peer is capped to its fair share.
        let greedy = map.values().filter(|b| b.reply_to == "greedy").count();
        assert!(greedy <= MAX_STREAMS_PER_PEER, "greedy peer bounded, got {greedy}");
        // (c) total within the global backstop.
        assert!(map.len() <= MAX_STREAM_ENTRIES, "global entry cap held, got {}", map.len());
    }

    #[test]
    fn stream_wire_loop_provider_buffer_to_consumer_decode_matches_buffered() {
        // End-to-end wire check WITHOUT libp2p: drive the real provider buffer (append via the
        // same free fns run_stream_job uses) → encode each poll reply on the wire → decode +
        // reassemble exactly as the consumer does. Proves the provider's bytes and the consumer's
        // decode agree, and that incremental prune-on-ack reassembles the buffered result.
        let adapter = StubAdapter; // emits "Hello" + " world", tokens 5
        let nonce = [42u8; 16];
        let req = ServeRequest {
            reply_to: "consumerZ".into(),
            model_ref: "qwen2.5:7b".into(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
            tools: Vec::new(),
            nonce,
        };

        // Collect the frames the engine produces (the buffered result), then reveal them into the
        // stream buffer one-per-poll to simulate tokens arriving between polls (the real case).
        let mut buffered_frames: Vec<Vec<u8>> = Vec::new();
        crate::serve::handle_serve_request_parsed(req, &adapter, &mut |c| buffered_frames.push(c.to_vec()));
        let mut pending: std::collections::VecDeque<Vec<u8>> = buffered_frames.iter().cloned().collect();

        let mut map = HashMap::new();
        stream_begin_in(&mut map, nonce, "consumerZ", 0);

        // Consumer: poll loop over the wire — encode the provider reply, decode as the consumer
        // would, parse_response the slice, accumulate, advance offset, prune-on-ack, until done.
        let mut text = String::new();
        let mut offset: u32 = 0;
        let mut tokens = 0u64;
        let mut guard = 0;
        loop {
            guard += 1;
            assert!(guard < 100, "poll loop should terminate");
            // A token "arrives" between polls: reveal one more frame into the provider buffer.
            if let Some(frame) = pending.pop_front() {
                stream_append_in(&mut map, &nonce, &frame, guard as u64);
            }
            let reply_bytes = fetch_chunks_from(&mut map, &nonce, "consumerZ", offset, 2 + guard as u64).encode();
            match crate::serve::FetchChunksResponse::decode(&reply_bytes).unwrap() {
                crate::serve::FetchChunksResponse::Chunks { framed, next_offset, done } => {
                    for chunk in parse_response(&framed).unwrap() {
                        match chunk {
                            ServeChunk::Delta(t) => text.push_str(&t),
                            ServeChunk::Done { tokens: n, .. } => tokens = n,
                            _ => {}
                        }
                    }
                    offset = next_offset;
                    if done {
                        break;
                    }
                }
                other => panic!("unexpected reply: {other:?}"),
            }
        }

        // Reassembled streamed output equals the buffered result, byte-for-byte.
        let buffered_text: String = parse_response(&frame_response(&buffered_frames))
            .unwrap()
            .into_iter()
            .filter_map(|c| if let ServeChunk::Delta(t) = c { Some(t) } else { None })
            .collect();
        assert_eq!(text, buffered_text);
        assert_eq!(text, "Hello world");
        assert_eq!(tokens, 5);
        // Prune-on-ack held: after fully draining, the live buffer keeps only the un-acked tail.
        assert!(map[&nonce].base_offset > 0, "prefix was pruned as the consumer acked it");
    }

    #[test]
    fn reply_to_bind_accepts_the_authenticated_sender_and_rejects_a_spoof() {
        // F1 core rule (env-free, so it never races the kill-switch test): a legitimate consumer
        // sets reply_to = its own peer id = the authenticated sender.
        assert!(reply_to_matches("12D3KooWConsumer", "12D3KooWConsumer"));
        // A peer attributing a stream to a *victim* (reply_to != sender) is rejected → run_stream_job
        // returns before stream_begin, so no buffer is created in the victim's fairness bucket.
        assert!(!reply_to_matches("12D3KooWVictim", "12D3KooWAttacker"));
        assert!(!reply_to_matches("", "12D3KooWAttacker"));
    }

    #[test]
    fn reply_to_bind_kill_switch_reverts_to_accept_all() {
        // The ONLY test that touches the process-global kill-switch env var; no other test reads
        // it (the core rule is tested via `reply_to_matches`), so there is no cross-test race.
        std::env::set_var("OPENHYDRA_DISABLE_REPLYTO_BIND", "1");
        assert!(reply_to_authorized("a", "b"), "kill-switch must accept a mismatch");
        std::env::remove_var("OPENHYDRA_DISABLE_REPLYTO_BIND");
        assert!(!reply_to_authorized("a", "b"), "enforcement restored once the var is cleared");
    }

    #[test]
    fn generate_guarded_turns_a_panic_into_a_terminal_error() {
        // F5: a panicking engine must not propagate out of generation — for streaming it would
        // leave the buffer `done=false` (→ 120s consumer stall); for the buffered path it would
        // unwind out of `dispatch` and skip `respond`/`store_result` (→ no reply, nonce stuck
        // `Generating`). The guard catches it and emits a terminal Error via the sink so the
        // consumer ends fast on either transport. The `Vec` sink here is the buffered path's exact
        // usage (`chunks.push`); the stream path feeds the same guard a `stream_append` sink.
        struct PanicAdapter;
        impl EngineAdapter for PanicAdapter {
            fn engine_name(&self) -> &'static str { "panic" }
            fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> { Ok(vec![]) }
            fn serve_stream(
                &self,
                _req: &InferenceRequest,
                _on_delta: &mut dyn FnMut(&str),
            ) -> Result<ServeOutcome, AdapterError> {
                panic!("engine exploded");
            }
        }
        let req = ServeRequest {
            reply_to: "c".into(), model_ref: "m".into(), messages: vec![],
            max_tokens: None, temperature: None, tools: Vec::new(), nonce: [0u8; 16],
        };
        let mut frames: Vec<Vec<u8>> = Vec::new();
        // Must NOT propagate the panic; returns a failed summary.
        let summary = generate_guarded(req, &PanicAdapter, &mut |c| frames.push(c.to_vec()));
        assert!(!summary.ok);
        assert_eq!(summary.tokens, 0, "no tokens → no commitment recorded");
        let last = frames.last().expect("a terminal frame was appended");
        assert!(matches!(ServeChunk::decode(last).unwrap(), ServeChunk::Error(_)));
        assert!(ServeChunk::frame_is_terminal(last), "the appended frame ends the stream");
    }

    #[test]
    fn stream_begin_ttl_prunes_expired_buffers() {
        let mut map = HashMap::new();
        stream_begin_in(&mut map, [1u8; 16], "c", 0); // recorded at t=0
        // A begin far past the TTL prunes the stale buffer as a side effect.
        stream_begin_in(&mut map, [2u8; 16], "c", STREAM_TTL_MS + 1);
        assert!(!map.contains_key(&[1u8; 16]), "expired stream pruned on the next begin");
        assert!(map.contains_key(&[2u8; 16]));
    }

    #[test]
    fn result_caps_evict_oldest_first() {
        // Distinct 16-byte keys (a u8 key would wrap past 256 and collide).
        let key = |i: usize| {
            let mut k = [0u8; 16];
            k[..8].copy_from_slice(&(i as u64 + 1).to_le_bytes());
            k
        };
        let mut map = HashMap::new();
        for i in 0..(MAX_RESULT_ENTRIES + 5) {
            let mut e = ready_entry("c", vec![vec![0u8; 8]]);
            e.recorded_ms = i as u64; // ascending age: i=0 is the oldest
            map.insert(key(i), e);
        }
        enforce_result_caps(&mut map);
        assert_eq!(map.len(), MAX_RESULT_ENTRIES, "count cap enforced");
        // The 5 oldest (recorded_ms 0..4) were evicted; the newest survive.
        assert!(!map.contains_key(&key(0)));
        assert!(!map.contains_key(&key(4)));
        assert!(map.contains_key(&key(MAX_RESULT_ENTRIES + 4)));
    }
}
