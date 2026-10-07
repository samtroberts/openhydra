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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::share_policy::{PolicyWatcher, SharePolicy};

use openhydra_network::handle::NetworkHandle;
use openhydra_network::types::PeerRecord;
use openhydra_protocol::credit::{throttle_multiplier, CreditAccount};
use openhydra_protocol::payout::SettlementAsset;
use openhydra_protocol::receipts::CoSignedReceipt;
use openhydra_protocol::store::Store;

use crate::adapter::{AdapterError, DetectedModel, EngineAdapter};
use crate::attest::{attest_refusal, Attestor, ATTEST_CHALLENGE};
use crate::prompt_score::{
    handle_model_info_inbound, handle_prompt_score_inbound, quant_hint_from_name, score_refusal, InflightGate,
    ModelInfo, MODEL_INFO_REQUEST, PROMPT_SCORE_CHALLENGE,
};
use crate::aup::{AupDecision, AupPolicy};
use crate::ratelimit::{RateLimitConfig, RateLimiter};
use crate::receipt::{handle_receipt_inbound, BindRejection, RECEIPT_REQUEST};
use crate::serve::{
    failed_serve_billable, frame_response, handle_serve_request, FetchChunksResponse,
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

/// libp2p proxy method byte for an M5 **remote scope-set** control command (a control-capable swarm
/// member → an owner's rig): `0x15 ‖ json(RemoteScopeSet)`. The provider authenticates it
/// owner-authoritatively (credential [`CAP_CONTROL`] bound to the connecting peer + a member
/// signature + a replay window) and applies the scope to its share policy. Replies with a
/// `json(RemoteScopeAck)`. (0x14 is [`FETCH_CHUNKS`].)
pub const REMOTE_SCOPE_SET: u8 = 0x15;

/// libp2p proxy method byte to **close a stream** (consumer → provider): `0x19 ‖ nonce(16) ‖
/// offset(4, big-endian)`. The consumer is done with it (it broke, or its client left): the provider
/// replies with the frames from `offset` up to what its polls already returned, and nothing newer
/// (so the consumer holds exactly what it owes for), stops generating, and keeps the stream until
/// it is settled or its commitment expires (review S8). An older provider answers with an error
/// blob that doesn't decode as a [`FetchChunksResponse`].
pub const STREAM_CLOSE: u8 = 0x19;

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
/// Per-peer cap on attestation challenges: one a minute. Each answer is a Secure Enclave signature
/// (slower than Ed25519), and a consumer re-attests only every 5 minutes.
const ATTEST_RATE_RPS: f64 = 1.0 / 60.0;
const ATTEST_RATE_BURST: f64 = 2.0;
/// How long after a serve its consumer's challenges skip the provider-wide attestation cap: a flood
/// from peers we never served (free identities) can't crowd out the consumers we serve.
const SERVED_ATTEST_PRIORITY_MS: u64 = 60 * 60 * 1000;
/// Peers remembered for that (the oldest are forgotten first).
const MAX_RECENTLY_SERVED: usize = 4096;
/// How long after we answered a verifier's challenge it counts as mid-round (see
/// `Provider::scored_recently`): a round sends its next text within seconds.
const IN_ROUND_MS: u64 = 2 * 60 * 1000;
/// A peer may move its challenges to another model at most once within this long: each switch can
/// need a side-scorer (re)load, which a peer must not be able to force back to back (review round 4
/// L4-7; round 8 L8-4 counts switches, not models, so alternating two models doesn't get round it),
/// while a gateway checking two models of one provider in turn isn't held up (R5L-3). At most two an
/// hour per peer, within the side-scorer's six starts an hour (R6-6).
const SCORE_SWITCH_MIN_MS: u64 = 30 * 60 * 1000;
const ATTEST_RATE_MAX_TRACKED: usize = 4096;
/// Cap on attestation challenges across all peers: the per-peer cap alone lets many peers (or one
/// with many identities) keep the Secure Enclave signing nonstop.
const ATTEST_GLOBAL_RPS: f64 = 20.0;
const ATTEST_GLOBAL_BURST: f64 = 40.0;
/// The single key the global attestation bucket is kept under.
const ATTEST_GLOBAL_KEY: &str = "*";
/// A second provider-wide bucket, for consumers we served recently once the main one is spent: they
/// still get attested under a flood, but a flood of identities that each took a tiny serve can't use
/// it to sign without bound (review round 4 X4-12).
const ATTEST_SERVED_RPS: f64 = 5.0;
const ATTEST_SERVED_BURST: f64 = 20.0;
/// How long a peer that passed a private model's swarm gate may attest that model (see
/// `Provider::private_attestable`).
const PRIVATE_ATTEST_WINDOW_MS: u64 = 24 * 60 * 60 * 1000;
/// `(peer, model)` pairs remembered for that (the oldest are forgotten first).
const MAX_PRIVATE_ATTEST_PAIRS: usize = 4096;
/// Per-peer cap on prompt-scoring challenges. A verifier round is 24 challenges sent one at a time
/// (~1 s each on an L4), so this admits a round without queueing but stops one peer hogging the
/// engine; the provider-wide `InflightGate` is the hard bound.
const SCORE_RATE_RPS: f64 = 2.0;
const SCORE_RATE_BURST: f64 = 4.0;

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

/// M6/F2: decide which models' `MODEL_WITHDRAWN` tombstones to (re)publish this announce pass, and
/// advance the re-publish countdown in `pending`. Pure so the bookkeeping is unit-testable without a
/// swarm. Ordering matters:
///   1. every model dropped since last pass (`last \ desired`) (re)starts a fresh countdown;
///   2. every model intended again (`desired`) is cancelled — a re-shared model must never be
///      withdrawn, even if a countdown was still running;
///   3. every still-pending model is published this pass and its counter decremented, expiring at
///      zero (so a tombstone is issued exactly `republish_passes` times).
/// Returns the models to publish now (a fresh, freshness-check-passing message each time).
fn plan_withdrawals(
    last: &BTreeSet<String>,
    desired: &BTreeSet<String>,
    pending: &mut BTreeMap<String, u8>,
    republish_passes: u8,
) -> Vec<String> {
    for gone in last.difference(desired) {
        pending.insert(gone.clone(), republish_passes);
    }
    for intended in desired {
        pending.remove(intended);
    }
    let to_publish: Vec<String> = pending.keys().cloned().collect();
    pending.retain(|_, remaining| {
        *remaining = remaining.saturating_sub(1);
        *remaining > 0
    });
    to_publish
}

/// Withdrawal hysteresis (Finding 2) — compute the **effective** announce set that feeds
/// [`plan_withdrawals`], damping a transient detection blip so it can't publish a spurious
/// `MODEL_WITHDRAWN`. A model drops out of the announceable set (`desired`) for one of two reasons,
/// which this distinguishes:
///   * it is **still detected** but no longer announceable (the operator un-shared it, or flipped it
///     to Private) — an intentional withdrawal → evicted **now** (not added back to `effective`), and
///     its absence streak cleared;
///   * it **vanished from detection** — possibly a transient (`auto`/[`MultiAdapter`] mode folds one
///     engine's failed probe into a partial `Ok` list) → kept in `effective` (so no tombstone) until
///     it has been undetected for `grace` consecutive passes, then released to be withdrawn.
/// A model announceable again clears its streak. Pure + unit-testable; `absence` carries the streak
/// across passes. `grace <= 1` disables the hysteresis (immediate withdrawal, the pre-hysteresis
/// behaviour). The ~300s advert TTL backstops the (bounded) delay for a genuine engine-vanish.
fn apply_absence_grace(
    last: &BTreeSet<String>,
    desired: &BTreeSet<String>,
    detected: &BTreeSet<String>,
    absence: &mut BTreeMap<String, u8>,
    grace: u8,
) -> BTreeSet<String> {
    let mut effective = desired.clone();
    for gone in last.difference(desired) {
        if detected.contains(gone) {
            absence.remove(gone); // still present → intentional policy withdrawal, no grace
            continue;
        }
        let streak = absence.entry(gone.clone()).or_insert(0);
        *streak = streak.saturating_add(1);
        if *streak < grace {
            effective.insert(gone.clone()); // within grace → treat as still announced (no tombstone)
        } else {
            absence.remove(gone); // grace exhausted → release for withdrawal, stop tracking
        }
    }
    for present in desired {
        absence.remove(present); // announceable again → reset any streak
    }
    effective
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
    /// The libp2p-authenticated peer the serve came from: only it may settle the receipt.
    served_peer: String,
    /// Whether the consumer got the whole completion (a buffered reply, or its stream polled to the
    /// end): only then is an unsettled serve its fault, and counts against it (credit window).
    delivered: bool,
    /// On Base, the wallet the serve request said its claims would come from: the receipt's claim
    /// must be signed by it.
    wallet: Option<[u8; 20]>,
}

/// How long a serve commitment stays settleable, and the receipt-freshness window (B-S1/S7).
const COMMITMENT_TTL_MS: u64 = 5 * 60 * 1000;
/// Allowance for consumer/provider clock skew on the receipt timestamp.
const CLOCK_SKEW_MS: u64 = 60 * 1000;
/// Hard cap on outstanding (un-settled) commitments — a backstop so a flood of serves whose
/// receipts never arrive can't grow the map without bound.
const MAX_COMMITMENTS: usize = 10_000;
/// Outstanding commitments one peer may hold; past it its oldest goes, so one peer can't crowd
/// everyone else's out of the map.
const MAX_COMMITMENTS_PER_PEER: usize = 256;

/// Credit window (a provider that charges, see [`payment_window`]): how long after a serve its
/// receipt may stay unsettled before the consumer gets no more serves. A consumer settles right
/// after the completion; a few queued settlements to one provider take well under this.
const SETTLE_GRACE_MS: u64 = 2 * 60 * 1000;
/// …and how many served tokens it may have unsettled at once (a burst of parallel serves).
const MAX_UNSETTLED_TOKENS: u64 = 65_536;
/// Serves whose receipt never came, within [`DEFAULT_WINDOW_MS`], after which a consumer is refused.
const MAX_DEFAULTS: usize = 3;
const DEFAULT_WINDOW_MS: u64 = 24 * 60 * 60 * 1000;
/// Consumers whose defaults a provider remembers (the oldest are forgotten first).
const MAX_DEFAULT_PEERS: usize = 4096;
/// Paid serves one consumer may have generating at once (each reserves its `max_tokens` against the
/// unpaid-token cap while it runs).
const MAX_UNPAID_CONCURRENT: usize = 8;
/// What a paid serve without `max_tokens` reserves while it runs.
const DEFAULT_RESERVE_TOKENS: u64 = 8_192;
/// Unsettled serves one consumer may have at once.
const MAX_UNSETTLED_SERVES: usize = 64;
/// How much a paid serve's reservation grows at a time once it has generated what it reserved.
const RESERVE_STEP_TOKENS: u64 = 1_024;

/// What a paid serve reserves against the unpaid-token cap when it starts: its `max_tokens`, at most
/// [`DEFAULT_RESERVE_TOKENS`]. It grows as the serve generates past it, while the credit window
/// has room ([`Provider::grow_reservation`]); a serve that can't grow stops there. So no serve can
/// leave more unpaid than the window, and the engine is never sent a cap the client didn't ask for.
fn paid_reservation(requested: Option<u32>) -> u64 {
    requested.map_or(DEFAULT_RESERVE_TOKENS, |t| u64::from(t).min(DEFAULT_RESERVE_TOKENS))
}

/// A consumer's serve not yet paid for, as the credit window sees it.
#[derive(Debug, Clone, Copy)]
struct Unsettled {
    at: u64,
    tokens: u64,
    /// Whether the consumer got all of it; a serve it didn't still counts toward its caps (it had the
    /// output it read), but isn't held against it once the grace period runs out.
    delivered: bool,
}

/// The credit window: whether a consumer may be served now, given its unsettled serves
/// (`(served at, tokens)`) and how many of its serves went unpaid within [`DEFAULT_WINDOW_MS`].
/// Payment is otherwise voluntary (a consumer has its tokens before it signs anything), so this
/// bounds what one consumer identity can take without paying: about one serve, or what fits in the
/// grace period. Not a guarantee against a consumer that keeps making new identities; that needs a
/// check of its on-chain escrow before serving.
///
/// `unsettled` are its serves not yet paid for, delivered or not (a stream abandoned just short of
/// its end was still read), `generating` the reservations of its paid serves still running, and
/// `reserving` what this request would reserve.
fn payment_window(unsettled: &[Unsettled], generating: &[u64], reserving: u64, defaults: usize, now: u64) -> Result<(), String> {
    if defaults >= MAX_DEFAULTS {
        return Err(format!("{defaults} recent serves were never paid for; this provider won't serve you for now"));
    }
    if unsettled.iter().any(|u| u.delivered && now.saturating_sub(u.at) > SETTLE_GRACE_MS) {
        return Err("a serve from this provider is still unpaid; settle it before asking for more".into());
    }
    if unsettled.len() >= MAX_UNSETTLED_SERVES || generating.len() >= MAX_UNPAID_CONCURRENT {
        return Err("too many serves are still unpaid or running; settle them before asking for more".into());
    }
    let owed = unsettled.iter().map(|u| u.tokens).sum::<u64>() + generating.iter().sum::<u64>();
    if owed.saturating_add(reserving) > MAX_UNSETTLED_TOKENS {
        return Err(if owed == 0 {
            format!("max_tokens {reserving} is more than this provider serves on credit ({MAX_UNSETTLED_TOKENS})")
        } else {
            format!(
                "{owed} served tokens are still unpaid or running, and this request's max_tokens ({reserving}) \
                 doesn't fit beside them ({MAX_UNSETTLED_TOKENS} on credit); lower max_tokens, or retry once they settle"
            )
        });
    }
    Ok(())
}

/// Record that `peer` passed private `model`'s swarm gate at `now` (with what it passed with, `v`),
/// keeping at most [`MAX_PRIVATE_ATTEST_PAIRS`] pairs: a pair already there is refreshed in place;
/// otherwise, when full, expired ones go first, then the oldest.
fn note_private_member_in<V>(m: &mut HashMap<(String, String), (u64, V)>, peer: &str, model: &str, now: u64, v: V) {
    let key = (peer.to_string(), model.to_string());
    if !m.contains_key(&key) && m.len() >= MAX_PRIVATE_ATTEST_PAIRS {
        m.retain(|_, (at, _)| now.saturating_sub(*at) <= PRIVATE_ATTEST_WINDOW_MS);
        if m.len() >= MAX_PRIVATE_ATTEST_PAIRS {
            if let Some(oldest) = m.iter().min_by_key(|(_, (at, _))| *at).map(|(k, _)| k.clone()) {
                m.remove(&oldest);
            }
        }
    }
    m.insert(key, (now, v));
}

/// What `peer` passed private `model`'s swarm gate with, if it did within
/// [`PRIVATE_ATTEST_WINDOW_MS`]. Anyone else gets the same refusal as for an unshared model, so a
/// challenge still can't probe for private models.
fn private_member_in<'a, V>(m: &'a HashMap<(String, String), (u64, V)>, peer: &str, model: &str, now: u64) -> Option<&'a V> {
    m.get(&(peer.to_string(), model.to_string()))
        .filter(|(at, _)| now.saturating_sub(*at) <= PRIVATE_ATTEST_WINDOW_MS)
        .map(|(_, v)| v)
}

/// A paid serve's reservation against its consumer's unpaid-token cap while it generates; released
/// when dropped (the serve recorded, or failed).
struct PaymentReservation {
    generating: Arc<Mutex<HashMap<String, Vec<Running>>>>,
    peer: String,
    /// Tells this reservation apart from the consumer's others in `generating`.
    id: u64,
    /// What it reserved at the start ([`paid_reservation`]).
    initial: u64,
}

/// A paid serve running now, as its consumer's credit window counts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Running {
    id: u64,
    /// What it holds against the window: its reservation (grown as it generates).
    tokens: u64,
    /// Its receipt nonce: a stream's commitment grows under it while it runs, and is already
    /// counted by `tokens` (never less than what was sent).
    nonce: [u8; 16],
}

/// What `peer`'s unsettled commitments owe, leaving out those of serves still running (`running`),
/// which their reservations count. A finished stream is counted for all it generated (`generated`,
/// [`generated_of`]), fetched yet or not: its reservation ended with its generation, and its consumer
/// can still fetch the rest (review round 4 M4).
fn unsettled_of(
    map: &HashMap<[u8; 16], ServeCommitment>,
    peer: &str,
    running: &[Running],
    generated: &HashMap<[u8; 16], u64>,
) -> Vec<Unsettled> {
    let is_running = |n: &[u8; 16]| running.iter().any(|r| r.nonce == *n);
    let committed = map.iter().filter(|(n, c)| c.served_peer == peer && !is_running(n)).map(|(n, c)| Unsettled {
        at: c.recorded_ms,
        tokens: c.tokens.max(generated.get(n).copied().unwrap_or(0)),
        delivered: c.delivered,
    });
    let unread = generated
        .iter()
        .filter(|(n, _)| !map.contains_key(*n) && !is_running(n))
        .map(|(_, t)| Unsettled { at: 0, tokens: *t, delivered: false });
    committed.chain(unread).collect()
}

/// What each of `peer`'s finished, still-open streams generated, by nonce: the billable count of all
/// its frames ([`StreamBuffer::tally`]).
fn generated_of(streams: &HashMap<[u8; 16], StreamBuffer>, peer: &str) -> HashMap<[u8; 16], u64> {
    streams
        .iter()
        .filter(|(_, b)| b.reply_to == peer && b.done && !b.closed)
        .map(|(n, b)| (*n, b.tally.billable(b.billing.max_tokens)))
        .filter(|(_, t)| *t > 0)
        .collect()
}

impl Drop for PaymentReservation {
    fn drop(&mut self) {
        let mut map = self.generating.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(v) = map.get_mut(&self.peer) {
            if let Some(i) = v.iter().position(|r| r.id == self.id) {
                v.swap_remove(i);
            }
            if v.is_empty() {
                map.remove(&self.peer);
            }
        }
    }
}

/// A verified on-chain claim a provider keeps per `(consumer, model)` to settle: the claim is the
/// pair's (consumer from the key's receipt, this provider, the model) at `cumulative`, until
/// `expires_at`; `signer` is the Base wallet that signed it (pinned per pair).
#[derive(Debug, Clone, PartialEq, Eq)]
struct KeptClaim {
    cumulative: u64,
    expires_at: u64,
    signer: Option<[u8; 20]>,
    sig: Vec<u8>,
}

/// The first byte of a [`KeptClaim`]'s encoding: rows kept before it (an expiry and a signature) don't
/// decode as one.
const KEPT_CLAIM_TAG: u8 = 0xC6;

impl KeptClaim {
    fn to_bytes(&self) -> Vec<u8> {
        let mut b = vec![KEPT_CLAIM_TAG];
        b.extend_from_slice(&self.cumulative.to_le_bytes());
        b.extend_from_slice(&self.expires_at.to_le_bytes());
        match self.signer {
            Some(s) => {
                b.push(1);
                b.extend_from_slice(&s);
            }
            None => b.push(0),
        }
        b.extend_from_slice(&self.sig);
        b
    }

    fn from_bytes(b: &[u8]) -> Option<Self> {
        let b = b.strip_prefix(&[KEPT_CLAIM_TAG])?;
        let cumulative = u64::from_le_bytes(b.get(..8)?.try_into().ok()?);
        let expires_at = u64::from_le_bytes(b.get(8..16)?.try_into().ok()?);
        let (signer, rest) = match b.get(16)? {
            0 => (None, b.get(17..)?),
            1 => (Some(b.get(17..37)?.try_into().ok()?), b.get(37..)?),
            _ => return None,
        };
        Some(KeptClaim { cumulative, expires_at, signer, sig: rest.to_vec() })
    }
}

/// What a receipt request is remembered by for an identical resend: its sender and its bytes.
fn recent_cosign_key(source_peer: &str, request: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update((source_peer.len() as u64).to_le_bytes());
    h.update(source_peer.as_bytes());
    h.update(request);
    h.finalize().into()
}

/// A `0x`-prefixed (or bare) hex EVM address.
fn parse_evm_address(s: &str) -> Option<[u8; 20]> {
    hex::decode(s.strip_prefix("0x").unwrap_or(s)).ok()?.try_into().ok()
}

/// Add a commitment to `map`, within [`MAX_COMMITMENTS`] and [`MAX_COMMITMENTS_PER_PEER`] (past it,
/// the peer's oldest goes). At [`MAX_COMMITMENTS`] the oldest undelivered commitment makes room
/// (a reply that never reached its consumer can't be held against it, so it costs only a payment
/// that wasn't coming); with none, nothing is added.
fn insert_commitment(map: &mut HashMap<[u8; 16], ServeCommitment>, nonce: [u8; 16], c: ServeCommitment) {
    // Never over an existing one: a commitment is replaced only by the serve that made it, in place
    // (review round 5, R5S-1).
    if map.contains_key(&nonce) {
        return;
    }
    let peers: Vec<([u8; 16], u64)> =
        map.iter().filter(|(_, o)| o.served_peer == c.served_peer).map(|(n, o)| (*n, o.recorded_ms)).collect();
    if peers.len() >= MAX_COMMITMENTS_PER_PEER {
        if let Some((oldest, _)) = peers.iter().min_by_key(|(_, at)| *at) {
            map.remove(oldest);
        }
    }
    if map.len() >= MAX_COMMITMENTS {
        match oldest_undelivered(map) {
            Some(n) => {
                map.remove(&n);
            }
            None => return, // backstop — refuse to grow past the cap (stale entries already pruned)
        }
    }
    map.insert(nonce, c);
}

/// The oldest commitment whose serve hasn't reached its consumer, if any.
fn oldest_undelivered(map: &HashMap<[u8; 16], ServeCommitment>) -> Option<[u8; 16]> {
    map.iter().filter(|(_, c)| !c.delivered).min_by_key(|(_, c)| c.recorded_ms).map(|(n, _)| *n)
}

/// Drop expired commitments from `map`, returning the peers whose serves they were: receipts that
/// never came.
fn prune_expired_commitments(map: &mut HashMap<[u8; 16], ServeCommitment>, now: u64) -> Vec<String> {
    let mut unpaid = Vec::new();
    map.retain(|_, c| {
        let live = now.saturating_sub(c.recorded_ms) <= COMMITMENT_TTL_MS;
        // Only a serve its consumer received is its default; one it never got isn't its fault.
        if !live && c.delivered {
            unpaid.push(c.served_peer.clone());
        }
        live
    });
    unpaid
}

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

/// The weighted units a crypto-rail receipt must pay at least (the F2 floor): what we served by our
/// own count (`committed`, the serve commitment under the receipt's nonce), at the blended rate
/// the consumer's own step uses. Never the receipt's `tokens` or `completion_tokens`, which the
/// consumer fills in. Without an open commitment, the receipt's count (the commitment check
/// rejects that receipt anyway).
fn settlement_floor_units(committed: Option<u64>, receipt_tokens: u64) -> u64 {
    openhydra_protocol::weight::receipt_weighted_units(committed.unwrap_or(receipt_tokens), None)
}

/// Pure B-S1/B-S7 validation, extracted so it is unit-testable without a live `Provider`.
/// Prunes expired commitments, checks the receipt fields against the serve recorded under
/// `nonce`, and consumes the commitment (single-use) on success. `Err(reason)` rejects the
/// receipt before it is co-signed.
///
/// `consumer_peer` is the libp2p id the receipt's consumer key derives, `source_peer` the
/// authenticated sender of the receipt: both must be the peer that was served. Otherwise another
/// key (a throwaway one, whose per-pair high-water starts at zero) could settle someone else's serve.
#[allow(clippy::too_many_arguments)]
fn validate_and_consume_commitment(
    map: &mut HashMap<[u8; 16], ServeCommitment>,
    nonce: &[u8; 16],
    model_id: &str,
    tokens: u64,
    ts_unix_ms: u64,
    now: u64,
    consumer_peer: &str,
    source_peer: &str,
) -> Result<(), String> {
    // Receipt-timestamp freshness (B-S7): reject stale or far-future receipts.
    if ts_unix_ms + COMMITMENT_TTL_MS < now || ts_unix_ms > now + CLOCK_SKEW_MS {
        return Err("receipt timestamp outside freshness window".into());
    }
    map.retain(|_, c| now.saturating_sub(c.recorded_ms) <= COMMITMENT_TTL_MS);
    let c = map.get(nonce).ok_or("no serve commitment for this receipt nonce")?;
    if consumer_peer != c.served_peer || source_peer != c.served_peer {
        return Err("receipt is not from the consumer that was served".into());
    }
    if model_id != c.model_id {
        return Err("receipt model does not match the served model".into());
    }
    if tokens > c.tokens {
        return Err("receipt tokens exceed tokens actually served".into());
    }
    map.remove(nonce); // single-use
    Ok(())
}

/// One entry per key in a shared set while held (see `Provider::scoring_now`).
struct OneAtATime<'a> {
    set: &'a Mutex<std::collections::HashSet<String>>,
    key: String,
}

impl<'a> OneAtATime<'a> {
    /// Enter for `key`, or `None` while another holder for it is in.
    fn enter(set: &'a Mutex<std::collections::HashSet<String>>, key: &str) -> Option<Self> {
        set.lock().unwrap_or_else(|e| e.into_inner()).insert(key.to_string()).then(|| Self { set, key: key.to_string() })
    }
}

impl Drop for OneAtATime<'_> {
    fn drop(&mut self) {
        self.set.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.key);
    }
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
/// How long an admitted serve nonce is remembered once its serve ended: well past the commitment it
/// backs ([`COMMITMENT_TTL_MS`]), allowing for the consumer's clock. While its serve's commitment,
/// stream or result is still held, it is refused regardless.
const NONCE_MEMORY_MS: u64 = 2 * COMMITMENT_TTL_MS;
/// The most serve nonces remembered; past it the oldest is forgotten. Only admitted serves are
/// remembered (a refused one isn't), so reusing one that old needs this many paid-for serves between.
const MAX_SEEN_NONCES: usize = 65_536;

/// Note that `peer` challenges `model` at `now`, if it may (see [`SCORE_SWITCH_MIN_MS`]): the same
/// model as its last, or a switch at least that long after its previous one. Bounded by
/// [`MAX_RECENTLY_SERVED`] peers (past it, peers whose last switch is that old go first).
fn note_score_model(map: &mut HashMap<String, (String, u64)>, peer: &str, model: String, now: u64) -> bool {
    match map.get(peer) {
        Some((m, _)) if *m == model => return true,
        Some((_, since)) if now.saturating_sub(*since) < SCORE_SWITCH_MIN_MS => return false,
        _ => {}
    }
    if !map.contains_key(peer) && map.len() >= MAX_RECENTLY_SERVED {
        map.retain(|_, (_, since)| now.saturating_sub(*since) < SCORE_SWITCH_MIN_MS);
    }
    if map.contains_key(peer) || map.len() < MAX_RECENTLY_SERVED {
        map.insert(peer.to_string(), (model, now));
    }
    true
}

/// Workers answering stream polls, closes, result fetches and receipts, apart from the serve workers
/// (round 8 G8-1).
const CONTROL_WORKERS: usize = 2;
/// Whether an inbound message (by its first byte) is answered apart from generations (receipts on
/// a pool of their own, the rest on the control pool): quick work that must not wait behind them.
fn is_control_message(method: Option<u8>) -> bool {
    matches!(method, Some(FETCH_CHUNKS) | Some(STREAM_CLOSE) | Some(FETCH_RESULT) | Some(RECEIPT_REQUEST))
}

/// Ready fetches of one buffered result answered (round 9 B9-8): a consumer that lost a reply
/// fetches it once or twice; more is copying a large result over and over.
const MAX_RESULT_FETCHES: u8 = 4;

/// Whose turn it is on the prompt scorer (round 9 P9-2): the peer scoring now and its texts in a
/// row, and a peer turned away "busy" meanwhile (and when).
#[derive(Default)]
struct ScoreTurn {
    holder: Option<(String, u32)>,
    waiting: Option<(String, u64)>,
}

/// Texts in a row one verifier scores while another waits, before it yields its turn.
const SCORE_TURN_TEXTS: u32 = 4;
/// How long a verifier turned away keeps its claim on the next turn (it retries every few seconds).
const SCORE_TURN_WAIT_MS: u64 = 60_000;

impl ScoreTurn {
    /// Whether `peer` must yield: another verifier has been waiting, and `peer` has had its texts.
    fn must_yield(&self, peer: &str, now: u64) -> bool {
        matches!(&self.waiting, Some((w, at)) if w != peer && now.saturating_sub(*at) <= SCORE_TURN_WAIT_MS)
            && matches!(&self.holder, Some((h, n)) if h == peer && *n >= SCORE_TURN_TEXTS)
    }

    /// Note how `peer`'s challenge went: `scored`, or turned away `busy`.
    fn note(&mut self, peer: &str, scored: bool, busy: bool, now: u64) {
        if scored {
            match &mut self.holder {
                Some((h, n)) if h == peer => *n += 1,
                _ => self.holder = Some((peer.to_string(), 1)),
            }
            if matches!(&self.waiting, Some((w, _)) if w == peer) {
                self.waiting = None;
            }
        } else if busy && self.waiting.as_ref().is_none_or(|(w, at)| w == peer || now.saturating_sub(*at) > SCORE_TURN_WAIT_MS) {
            self.waiting = Some((peer.to_string(), now));
        }
    }
}

/// A queued stream open older than this is dropped: its consumer stopped waiting for the open ack
/// (its call timeout, 30 s).
const STREAM_OPEN_STALE: Duration = Duration::from_secs(30);

/// Serve nonces admitted recently, oldest first (see `Provider::seen_nonces`).
#[derive(Default)]
struct SeenNonces {
    at: HashMap<[u8; 16], u64>,
    order: std::collections::VecDeque<([u8; 16], u64)>,
}

impl SeenNonces {
    /// Whether `nonce` was admitted within [`NONCE_MEMORY_MS`] of `now`.
    fn seen(&self, nonce: &[u8; 16], now: u64) -> bool {
        self.at.get(nonce).is_some_and(|at| now.saturating_sub(*at) <= NONCE_MEMORY_MS)
    }

    /// Remember `nonce` as admitted at `now`, within [`MAX_SEEN_NONCES`] (expired ones go first,
    /// then the oldest).
    fn remember(&mut self, nonce: [u8; 16], now: u64) {
        while let Some(&(n, at)) = self.order.front() {
            let stale = self.at.get(&n) != Some(&at);
            let room = self.at.len() < MAX_SEEN_NONCES && self.order.len() < 2 * MAX_SEEN_NONCES;
            if !stale && now.saturating_sub(at) <= NONCE_MEMORY_MS && room {
                break;
            }
            self.order.pop_front();
            if !stale {
                self.at.remove(&n);
            }
        }
        self.at.insert(nonce, now);
        self.order.push_back((nonce, now));
    }
}

/// M6/F2 (delivery hardening): how many announce passes a per-model `MODEL_WITHDRAWN` tombstone is
/// re-published for. A single gossip publish is best-effort — a consumer that was momentarily
/// un-meshed (or hadn't joined yet) when a model was un-shared would otherwise keep the stale entry
/// until the ~300s TTL. Re-issuing on the next few passes (each a fresh, freshness-check-passing
/// message) covers a consumer that (re)connects within that window, while a re-share cancels the
/// countdown so a model that comes back is never spuriously withdrawn.
const WITHDRAW_REPUBLISH_PASSES: u8 = 3;

/// Withdrawal hysteresis (Finding 2): how many *consecutive* announce passes a previously-announced
/// model must be **absent from detection** before it is tombstoned. In `auto`/[`MultiAdapter`] mode a
/// transient engine failure is folded into a partial `Ok` list ([`crate::detect::detect_engines`]
/// drops a failed probe silently), so a model can vanish for a single pass and reappear the next —
/// without this grace that blip would publish a spurious `MODEL_WITHDRAWN` and flap consumers. `2`
/// tolerates a single missing pass (the common transient) while still evicting a genuinely-gone model
/// promptly (well inside the ~300s advert TTL). A *policy* withdrawal (model still detected, just no
/// longer announceable) bypasses this entirely and evicts at once — see [`apply_absence_grace`].
///
/// The grace window is `this × reannounce_secs`. At the default (120s → 240s) a single-pass blip is
/// fully absorbed: the advert is refreshed on recovery before its ~300s TTL lapses, so consumers see
/// no eviction at all. If an operator raises `reannounce_secs` past ~150s the grace still suppresses
/// the *spurious tombstone*, but the advert may TTL-lapse passively before the release decision — a
/// safe degradation (eviction via TTL instead of an explicit, faster tombstone), not a correctness
/// issue. Keep the two invariants in mind: `effective ⊇ desired` (a detected+shared model is never
/// tombstoned) and, in production, `desired ⊆ detected` (both derive from the one detected `models`).
const WITHDRAW_ABSENCE_GRACE_PASSES: u8 = 2;

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
    /// When the consumer last polled (or opened the stream): generation stops once it stops reading
    /// for [`STREAM_ABANDON_MS`].
    last_poll_ms: u64,
    /// The billable count of every frame appended so far ([`crate::serve::StreamTally`]).
    tally: crate::serve::StreamTally,
    /// The absolute chunk index a poll has returned frames up to: what the consumer is owed for.
    fetched_hi: u32,
    /// What the stream's serve commitment is recorded under.
    billing: StreamBilling,
    /// The consumer closed it ([`STREAM_CLOSE`]): generation stops, and no poll returns anything new.
    closed: bool,
    /// Its consumer has polled it at least once.
    polled: bool,
}

/// Whether a stream buffer has outlived its use: [`STREAM_TTL_MS`] after its last activity, or,
/// once a poll returned frames (the consumer owes for them), [`COMMITMENT_TTL_MS`] so a consumer
/// can still close it and settle while its commitment is open.
/// A stream still generating, unread, is kept past [`STREAM_ABANDON_MS`], so a consumer that comes
/// back within it (a long prefill, a dropped connection) finds it rather than `NotFound` (review
/// round 5, R5S-6).
fn stream_expired(b: &StreamBuffer, now: u64) -> bool {
    let ttl = if b.fetched_hi > 0 {
        COMMITMENT_TTL_MS.max(STREAM_TTL_MS)
    } else if !b.done {
        STREAM_TTL_MS.max(STREAM_ABANDON_MS + 30_000)
    } else {
        STREAM_TTL_MS
    };
    now.saturating_sub(b.recorded_ms) > ttl
}

/// How long a stream nobody has polled at all keeps generating: an honest consumer polls the moment
/// its open is acknowledged, so this only ends a stream opened and walked away from (R5S-6). Still
/// past the consumer's whole poll-retry window (6 polls of up to 30 s and their backoffs, ~190 s), so
/// lost first polls don't cut an honest stream short (R6-6, R7-7).
const NEVER_POLLED_ABANDON_MS: u64 = 200_000;

/// What a stream's serve commitment is recorded under, taken from its request.
#[derive(Debug, Clone, Default)]
struct StreamBilling {
    model_id: String,
    max_tokens: Option<u32>,
    wallet: Option<[u8; 20]>,
}

/// What a poll returned beyond earlier ones: the stream is now owed `tokens`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Fetched {
    tokens: u64,
    model_id: String,
    wallet: Option<[u8; 20]>,
}

/// How long a stream keeps generating with no poll from its consumer. An honest consumer polls
/// every poll interval while it reads, but rides out a dropped connection for up to about 160 s
/// (its call timeout, retries and stall clock) before giving up, so this stays above that: it only
/// ends a stream its consumer stopped reading (gone, or settled for what it had). A serve stopped
/// this way says so in its `Done` (`stopped`), so a consumer that comes back late isn't shown a
/// cut-off answer as complete (review round 4 H8).
const STREAM_ABANDON_MS: u64 = 180_000;

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

/// M4-base reach-gate decision as pure logic (no `self`, unit-testable). Decides *who* may reach an
/// already-shared model, from its `scope`:
/// - `Global`  → anyone (public marketplace).
/// - `Device`  → loopback only: the request must come from this node (`own_peer`) — a self-serve.
/// - `Private` / `Unknown` → self-serve is allowed; otherwise a valid swarm `credential` (for a swarm
///   the `authorizer` owns, bound to `source_peer`) is required. `Unknown` (a newer policy's scope
///   this build can't read) is treated as `Private` so an unrecognised reach never serves the public.
///
/// `Ok(())` ⇒ proceed. `Err((reason, is_swarm_refusal))` ⇒ refuse; `is_swarm_refusal` is true only for
/// a credential-gate failure (so the caller bumps the swarm counter, not for a device-only refusal).
fn scope_gate(
    scope: crate::share_policy::Scope,
    model_ref: &str,
    source_peer: &str,
    own_peer: &str,
    credential: Option<&openhydra_network::membership::MembershipCredential>,
    authorizer: Option<&crate::swarms::SwarmAuthorizer>,
    now_ms: u64,
) -> Result<(), (String, bool)> {
    use crate::share_policy::Scope;
    match scope {
        Scope::Global => Ok(()),
        Scope::Device => {
            if source_peer == own_peer {
                Ok(())
            } else {
                Err((
                    format!("model '{model_ref}' is device-only (not shared beyond this machine)"),
                    false,
                ))
            }
        }
        Scope::Private | Scope::Unknown => {
            if source_peer == own_peer {
                return Ok(()); // self-serve of one's own private model needs no credential
            }
            let Some(auth) = authorizer else {
                return Err((
                    format!("model '{model_ref}' is private and this provider has no swarm configured"),
                    false,
                ));
            };
            auth.authorize(credential, source_peer, now_ms)
                .map(|_| ())
                .map_err(|e| (e.to_string(), true))
        }
    }
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
    keep_going: &mut dyn FnMut(u64) -> bool,
) -> ServeSummary {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::serve::handle_serve_request_within(req, adapter, on_chunk, keep_going)
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
#[cfg(test)]
fn stream_begin_in(
    map: &mut HashMap<[u8; 16], StreamBuffer>,
    nonce: [u8; 16],
    reply_to: &str,
    now: u64,
) {
    stream_begin_billed_in(map, nonce, reply_to, now, StreamBilling::default())
}

/// [`stream_begin_in`] for a serve whose commitment is recorded under `billing`.
fn stream_begin_billed_in(
    map: &mut HashMap<[u8; 16], StreamBuffer>,
    nonce: [u8; 16],
    reply_to: &str,
    now: u64,
    billing: StreamBilling,
) {
    map.retain(|_, b| !stream_expired(b, now));
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
    // Global backstops: evict until within the count + byte caps, unread streams first (oldest
    // first): one its consumer read from is owed for what it read, and its consumer may still close
    // it to settle (S8); evicting it would leave that unprovable (review round 4 S4-11).
    loop {
        let total: usize = map.values().map(|b| b.bytes).sum();
        if map.len() < MAX_STREAM_ENTRIES && total <= MAX_STREAM_BYTES {
            break;
        }
        match map.iter().min_by_key(|(_, b)| (b.fetched_hi > 0, b.recorded_ms)).map(|(k, _)| *k) {
            Some(oldest) => {
                map.remove(&oldest);
            }
            None => break,
        }
    }
    map.insert(
        nonce,
        StreamBuffer {
            reply_to: reply_to.to_string(),
            base_offset: 0,
            chunks: Vec::new(),
            done: false,
            bytes: 0,
            recorded_ms: now,
            last_poll_ms: now,
            tally: Default::default(),
            fetched_hi: 0,
            billing,
            closed: false,
            polled: false,
        },
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
        buf.tally.push_encoded(encoded);
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
#[cfg(test)]
fn fetch_chunks_from(
    map: &mut HashMap<[u8; 16], StreamBuffer>,
    nonce: &[u8; 16],
    source_peer: &str,
    offset: u32,
    now: u64,
) -> FetchChunksResponse {
    fetch_chunks_billed(map, nonce, source_peer, offset, now).0
}

/// [`fetch_chunks_from`], also saying what the stream is owed when this poll returned frames no
/// earlier one did: every frame up to the end of the log, which the consumer now has (the
/// commitment is recorded from it, see [`crate::serve::StreamTally`]).
fn fetch_chunks_billed(
    map: &mut HashMap<[u8; 16], StreamBuffer>,
    nonce: &[u8; 16],
    source_peer: &str,
    offset: u32,
    now: u64,
) -> (FetchChunksResponse, Option<Fetched>) {
    let buf = match map.get_mut(nonce) {
        None => return (FetchChunksResponse::NotFound, None),
        // Ownership binding: a poll from anyone but the committing consumer is refused.
        Some(b) if b.reply_to != source_peer => return (FetchChunksResponse::Forbidden, None),
        Some(b) => b,
    };
    // A closed stream returns nothing past what polls already returned (review S8).
    let have_hi = if buf.closed { buf.fetched_hi.max(buf.base_offset) } else { buf.base_offset + buf.chunks.len() as u32 };
    // Clamp the request into the live window: a stale/duplicate poll below base (already pruned)
    // resumes at base; a poll past the end yields an empty slice.
    let start_abs = offset.clamp(buf.base_offset, have_hi);
    let li = (start_abs - buf.base_offset) as usize;
    let hi = (have_hi - buf.base_offset) as usize;
    // Copy the slice to return BEFORE pruning (frame_response copies, so the drain below is safe).
    let framed = frame_response(&buf.chunks[li..hi]);
    let done = buf.done || buf.closed;
    // Prune-on-ack: the poll for `offset` proves `[0..offset)` was received, so drop
    // `[base_offset..start_abs)`. We keep `[start_abs..]` (the data just returned) so a transport
    // drop before the consumer processes it re-polls the same offset and gets it again.
    if li > 0 {
        buf.chunks.drain(0..li);
        buf.base_offset = start_abs;
        buf.bytes = buf.chunks.iter().map(|c| c.len()).sum();
    }
    buf.recorded_ms = now;
    buf.last_poll_ms = now;
    buf.polled = true;
    let fetched = (have_hi > buf.fetched_hi).then(|| {
        buf.fetched_hi = have_hi;
        Fetched {
            tokens: buf.tally.billable(buf.billing.max_tokens),
            model_id: buf.billing.model_id.clone(),
            wallet: buf.billing.wallet,
        }
    });
    (FetchChunksResponse::Chunks { framed, next_offset: have_hi, done }, fetched)
}

/// Close the stream under `nonce` for its consumer `source_peer` ([`STREAM_CLOSE`]): the frames
/// from `offset` up to what polls already returned, nothing newer, and the stream marked closed
/// (generation stops). Pure, like [`fetch_chunks_billed`]. What it returns was already counted, so
/// the commitment doesn't change.
fn close_stream_in(
    map: &mut HashMap<[u8; 16], StreamBuffer>,
    nonce: &[u8; 16],
    source_peer: &str,
    offset: u32,
    now: u64,
) -> FetchChunksResponse {
    match map.get_mut(nonce) {
        None => return FetchChunksResponse::NotFound,
        Some(b) if b.reply_to != source_peer => return FetchChunksResponse::Forbidden,
        Some(b) => {
            b.closed = true;
            b.recorded_ms = now;
        }
    }
    fetch_chunks_billed(map, nonce, source_peer, offset, now).0
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
    /// Review F2 — provider-side crypto-rail voucher validation: the last co-signed
    /// `cumulative_amount` per `(consumer, model)`, so a new receipt's cumulative is checked
    /// monotonic (and, when a peg is configured, above the token floor) BEFORE we counter-sign it.
    /// Keyed by `reputation_key(hex(consumer_pubkey), model_id)`; persisted like the credit map.
    settled_cumulative: Mutex<HashMap<String, u64>>,
    /// The co-signed receipt that set each `settled_cumulative` entry (same key): sent back to a
    /// consumer whose cumulative is behind it ([`BindRejection::Behind`]). Persisted with it.
    settled_receipts: Mutex<HashMap<String, openhydra_protocol::receipts::CoSignedReceipt>>,
    /// Per consumer (libp2p id): when its serves went unpaid (commitments that expired unsettled),
    /// for the credit window. Only kept by a provider that charges.
    payment_defaults: Mutex<HashMap<String, Vec<u64>>>,
    /// Held across a receipt's validate → co-sign → record, so two receipts for one pair can't both
    /// validate against the same high-water and both be co-signed (each is cheap: a verify + a sign).
    receipt_lock: Mutex<()>,
    /// Replies to receipts co-signed within [`COMMITMENT_TTL_MS`], by sender and request: a consumer
    /// whose reply was lost sends the identical receipt again, and gets the same co-signature back
    /// (its serve commitment is spent by then; refusing would leave a paid serve looking unpaid).
    recent_cosigns: Mutex<HashMap<[u8; 32], (Vec<u8>, u64)>>,
    /// `(peer, model)` → when that peer last passed the swarm gate for a private model, and the
    /// credential it passed with: it may attest the model while still entitled to it (an attestation
    /// challenge carries no credential).
    private_members: Mutex<HashMap<(String, String), (u64, Option<openhydra_network::membership::MembershipCredential>)>>,
    /// Paid serves generating now, per consumer: each one's token reservation (credit window).
    paid_generating: Arc<Mutex<HashMap<String, Vec<Running>>>>,
    /// The next [`PaymentReservation::id`].
    next_reservation: std::sync::atomic::AtomicU64,
    /// This machine's own gateway, when it runs under another identity (the desktop's): exempt from
    /// the payment gate, like our own peer id (it never settles with us).
    self_consumer: Option<String>,
    /// The on-chain claim kept with each `settled_receipts` entry (same key), on a settlement chain.
    settled_claims: Mutex<HashMap<String, KeptClaim>>,
    /// The on-chain deployment paying receipts settle on: a paying receipt is co-signed only with a
    /// valid claim for it (see `with_settlement_chain`).
    settlement_chain: Option<openhydra_protocol::settlement::ChainDomain>,
    /// The provider's configured settlement peg (USDC micro-units per credit unit), for the F2 floor
    /// check. `None` = not opted into the crypto rail → monotonic-only validation.
    settlement_peg: Option<u64>,
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
    /// Device attestation (ATTESTED_ENGINE_TIER_PLAN_v1 Phase 0): answers `ATTEST_CHALLENGE` when the
    /// operator opted in (`provide --attest`). `None` → every challenge is refused.
    attestor: Option<Arc<Attestor>>,
    /// Per-peer rate cap on attestation challenges (a separate bucket from receipts/fetches).
    attest_rl: Arc<RateLimiter>,
    /// The cap on attestation challenges across all peers (one bucket).
    attest_global_rl: Arc<RateLimiter>,
    /// The bucket consumers we served draw on once `attest_global_rl` is spent.
    attest_served_rl: Arc<RateLimiter>,
    /// Peers we served recently, and when (see [`SERVED_ATTEST_PRIORITY_MS`]).
    recently_served: Mutex<HashMap<String, u64>>,
    /// Prompt-scoring challenges (Track A2-Logprob): answered only when the operator declared the
    /// engine's scoring API (`provide --prompt-scoring`).
    prompt_scoring: bool,
    /// Caps challenges scored at once, provider-wide (engines OOM on concurrent prompt log-probs).
    score_gate: InflightGate,
    /// Verifiers whose challenge we answered, and when: one answered in the last
    /// [`IN_ROUND_MS`] is mid-round and gets a spare scoring slot.
    scored_recently: Mutex<HashMap<String, u64>>,
    /// Peers with a prompt-scoring challenge being answered now (one at a time each), and per peer
    /// the model it last challenged and when (see [`SCORE_SWITCH_MIN_MS`]).
    scoring_now: Mutex<std::collections::HashSet<String>>,
    score_models: Mutex<HashMap<String, (String, u64)>>,
    /// Whose turn it is on the prompt scorer (round 9 P9-2).
    score_turn: Mutex<ScoreTurn>,
    /// Streams whose commitment a close already refreshed (once each, round 9 B9-5).
    close_refreshed: Mutex<std::collections::HashSet<[u8; 16]>>,
    /// Ready fetches answered per buffered result (round 9 B9-8).
    result_fetches: Mutex<HashMap<[u8; 16], u8>>,
    /// Per-peer rate cap on prompt-scoring challenges (and model-info requests).
    score_rl: Arc<RateLimiter>,
    /// What the engine last reported for each model, keyed by handle — the source of the identity
    /// declared in a model-info reply. Refreshed on every announce pass.
    detected: Mutex<HashMap<String, DetectedModel>>,
    /// Quantisation to declare for models whose engine doesn't report one (`provide --declare-quant`).
    declared_quant: Option<String>,
    /// P1 streaming: nonce → trimmable-log [`StreamBuffer`], populated by `SERVE_STREAM` and
    /// polled/pruned by `FETCH_CHUNKS`. Separate from `results` (0x10/0x11 buffered path) so the
    /// proven reconnect-fetch path is untouched; prune-on-ack keeps per-stream memory bounded to
    /// the un-fetched tail so thousands of concurrent streams don't evict one another.
    streams: Mutex<HashMap<[u8; 16], StreamBuffer>>,
    /// Serve nonces admitted recently (nonce → when), so none is served twice: a reused nonce would
    /// rewrite the commitment of the serve that first used it, erasing what that one owes (review
    /// round 4 C2). Kept for [`NONCE_MEMORY_MS`], at most [`MAX_SEEN_NONCES`].
    seen_nonces: Mutex<SeenNonces>,
    /// Nonces of buffered serves still generating: held for the whole serve, however long it runs
    /// (a serve can outlast [`NONCE_MEMORY_MS`]), then remembered from its end (round 8 S8-2).
    serving_nonces: Mutex<std::collections::HashSet<[u8; 16]>>,
    /// Which models this provider shares (announces + serves), keyed on the engine handle
    /// (`DetectedModel::engine_ref`, the same string a consumer sends as `ServeRequest.model_ref`).
    /// A [`PolicyWatcher`] so the policy can be **hot-reloaded at runtime** — the user can toggle
    /// sharing in the desktop without restarting the node. Both the announce filter and the serve
    /// gate consult it, so a de-selected model is genuinely off (not merely hidden).
    policy: PolicyWatcher,
    /// Serializes `announce_models` so the poll thread and the filesystem-watcher thread never
    /// interleave two announce passes (which could publish a stale status view or refresh a
    /// just-removed model's DHT record). The poll loop yields via `try_lock` so it never blocks the
    /// serve loop; the watcher/startup path takes it blocking.
    announce_lock: Mutex<()>,
    /// M6: the set of models (engine refs) we advertised on the *previous* announce pass, so the
    /// next pass can diff it against the current intent and publish a `MODEL_WITHDRAWN` tombstone
    /// for each model that dropped out (un-shared, or gone from the engine). Tracks *intent*
    /// (policy ∩ detected), not network-announce success, so a still-shared-but-momentarily-
    /// unannounceable model is never spuriously withdrawn. Guarded by `announce_lock` in practice
    /// (only touched inside `announce_models_inner`), but a `Mutex` keeps it `Sync` on its own.
    last_announced: Mutex<BTreeSet<String>>,
    /// M6/F2: models with an in-flight `MODEL_WITHDRAWN` re-publish countdown (engine_ref → passes
    /// remaining). A model un-shared this pass is inserted at [`WITHDRAW_REPUBLISH_PASSES`] and its
    /// tombstone is re-issued (with a fresh timestamp) on each subsequent announce pass until the
    /// count hits zero — covering a consumer that missed the first gossip. Re-sharing the model
    /// removes it here so it is never withdrawn while intended. Bounded (drains to empty), guarded
    /// by `announce_lock` in practice like [`last_announced`].
    pending_withdrawals: Mutex<BTreeMap<String, u8>>,
    /// Withdrawal hysteresis (Finding 2): per-model count of consecutive announce passes a
    /// previously-announced model has been **absent from detection**. `auto`/[`MultiAdapter`] mode
    /// swallows one engine's transient failure into a partial `Ok` list (a dead engine's models just
    /// vanish), so a single missing pass must not tombstone. A vanished model is only withdrawn once
    /// this reaches [`WITHDRAW_ABSENCE_GRACE_PASSES`]; a reappearance (or a policy-driven drop, which
    /// is still *detected*) resets/skips it. Bounded, guarded by `announce_lock` in practice.
    absence_streak: Mutex<BTreeMap<String, u8>>,
    /// M4-base: owner-side swarm authorizer, consulted for a `Private`-scope model's serve gate.
    /// `None` → no swarms configured, so a private model is refused for everyone (fail-closed). A
    /// global model never touches it.
    swarm_auth: Option<Arc<crate::swarms::SwarmAuthorizer>>,
    /// M5: highest `issued_at_ms` of a remote-control command already applied, per control member
    /// (keyed by `member_public_key`). A new command must be strictly newer, so the same signed
    /// command can't be re-applied within the replay window (defence-in-depth on top of the Noise
    /// channel + the ±skew bound). Bounded by the number of distinct control members (small).
    remote_cmd_last_applied: Mutex<HashMap<String, u64>>,
}

/// v5 denomination gate (pure): a crypto-rail receipt's declared settlement asset + reference-price
/// peg must match the provider's own before it co-signs. A non-crypto receipt (`cumulative_amount ==
/// None`) always passes. This is the payoff of binding the asset + peg into the signed receipt
/// (`RECEIPT_FORMAT_V5`): a consumer using a different peg or a different asset is caught before the
/// provider co-signs a claim it can't honour.
///
/// **Fail-CLOSED on `our_peg == None` (review A):** a receipt that carries a `cumulative_amount` is a
/// crypto-rail voucher; only a consumer whose own peg is set stamps one. If THIS provider has no peg
/// configured, it cannot validate the denomination, so it MUST refuse rather than co-sign an
/// unvalidated claim — otherwise a provider that intends to run the crypto rail but boots with the peg
/// env unset/zero (`with_settlement_peg(0)` → `None`) would silently co-sign vouchers at
/// consumer-dictated pegs (and `check_cumulative` skips the floor too). Refusing surfaces the
/// misconfiguration instead of producing a signed artifact the provider can't disclaim.
fn check_settlement_denomination(
    cumulative_amount: Option<u64>,
    settlement_asset: Option<SettlementAsset>,
    peg_micros_per_credit: Option<u64>,
    our_peg: Option<u64>,
) -> Result<(), String> {
    if cumulative_amount.is_none() {
        // A provider that charges co-signs only a receipt that pays.
        return match our_peg {
            Some(_) => Err("this provider charges: the receipt carries no payment (cumulative amount)".to_string()),
            None => Ok(()),
        };
    }
    let Some(our_peg) = our_peg else {
        return Err(
            "crypto-rail receipt (carries a cumulative voucher) but this provider has no settlement \
             peg configured — refusing to co-sign an unvalidated denomination (v5)"
                .to_string(),
        );
    };
    if settlement_asset != Some(SettlementAsset::Usdc) {
        return Err("crypto-rail receipt settlement asset is missing or unsupported (v5)".to_string());
    }
    if peg_micros_per_credit != Some(our_peg) {
        return Err("crypto-rail receipt peg is missing or does not match the provider's configured peg (v5)".to_string());
    }
    Ok(())
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
            settled_cumulative: Mutex::new(HashMap::new()),
            settled_receipts: Mutex::new(HashMap::new()),
            receipt_lock: Mutex::new(()),
            recent_cosigns: Mutex::new(HashMap::new()),
            settled_claims: Mutex::new(HashMap::new()),
            settlement_chain: None,
            self_consumer: None,
            paid_generating: Arc::new(Mutex::new(HashMap::new())),
            next_reservation: std::sync::atomic::AtomicU64::new(0),
            private_members: Mutex::new(HashMap::new()),
            payment_defaults: Mutex::new(HashMap::new()),
            settlement_peg: None,
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
            attestor: None,
            attest_rl: Arc::new(RateLimiter::new(RateLimitConfig {
                rps: ATTEST_RATE_RPS,
                burst: ATTEST_RATE_BURST,
                max_inflight: 0,
                max_tracked: ATTEST_RATE_MAX_TRACKED,
            })),
            recently_served: Mutex::new(HashMap::new()),
            attest_global_rl: Arc::new(RateLimiter::new(RateLimitConfig {
                rps: ATTEST_GLOBAL_RPS,
                burst: ATTEST_GLOBAL_BURST,
                max_inflight: 0,
                max_tracked: 1,
            })),
            attest_served_rl: Arc::new(RateLimiter::new(RateLimitConfig {
                rps: ATTEST_SERVED_RPS,
                burst: ATTEST_SERVED_BURST,
                max_inflight: 0,
                max_tracked: 1,
            })),
            prompt_scoring: false,
            score_gate: InflightGate::new(),
            scored_recently: Mutex::new(HashMap::new()),
            scoring_now: Mutex::new(std::collections::HashSet::new()),
            score_models: Mutex::new(HashMap::new()),
            score_turn: Mutex::new(ScoreTurn::default()),
            close_refreshed: Mutex::new(std::collections::HashSet::new()),
            result_fetches: Mutex::new(HashMap::new()),
            score_rl: Arc::new(RateLimiter::new(RateLimitConfig {
                rps: SCORE_RATE_RPS,
                burst: SCORE_RATE_BURST,
                max_inflight: 0,
                max_tracked: ATTEST_RATE_MAX_TRACKED,
            })),
            detected: Mutex::new(HashMap::new()),
            declared_quant: None,
            streams: Mutex::new(HashMap::new()),
            seen_nonces: Mutex::new(SeenNonces::default()),
            serving_nonces: Mutex::new(std::collections::HashSet::new()),
            policy: PolicyWatcher::r#static(SharePolicy::share_all()),
            announce_lock: Mutex::new(()),
            last_announced: Mutex::new(BTreeSet::new()),
            pending_withdrawals: Mutex::new(BTreeMap::new()),
            absence_streak: Mutex::new(BTreeMap::new()),
            swarm_auth: None,
            remote_cmd_last_applied: Mutex::new(HashMap::new()),
        }
    }

    /// M4-base: attach the owner-side swarm authorizer (reads `~/.openhydra/swarms/`). Required for a
    /// `Private`-scope model to be servable — without it every private serve is refused (fail-closed).
    pub fn with_swarm_authorizer(mut self, auth: Arc<crate::swarms::SwarmAuthorizer>) -> Self {
        self.swarm_auth = Some(auth);
        self
    }

    /// Opt in to device attestation: answer `ATTEST_CHALLENGE` with claims signed by `attestor`
    /// (builder form). Only models this node announces publicly are attestable.
    pub fn with_attestor(mut self, attestor: Attestor) -> Self {
        // Its claims name this provider, so a consumer can tell them from a relayed attestation.
        let peer = self.net.libp2p_peer_id().to_string();
        self.attestor = Some(Arc::new(attestor.with_provider_peer(peer)));
        self
    }

    /// Opt in to answering `PROMPT_SCORE_CHALLENGE` (builder form). The adapter must be able to
    /// score (`OpenAiAdapter::with_prompt_scoring`); only publicly-announced models are scorable.
    pub fn with_prompt_scoring(mut self) -> Self {
        self.prompt_scoring = true;
        self
    }

    /// Declare the quantisation of models whose engine doesn't report one (builder form) — e.g. the
    /// checkpoint a vLLM / SGLang server loaded (`bf16`, `fp8`). Engines that report a quant (Ollama,
    /// llama.cpp from the GGUF header) keep their own.
    pub fn with_declared_quant(mut self, quant: Option<String>) -> Self {
        self.declared_quant = quant.filter(|q| !q.trim().is_empty());
        self
    }

    /// The identity this provider declares for `model_ref` in a model-info reply: the engine's own
    /// family / params / quant where it reports them, else parsed from the name, with the quant from
    /// `--declare-quant` or a tag in the name. `None` for a model the engine didn't report.
    fn model_info(&self, model_ref: &str) -> Option<ModelInfo> {
        let m = self.detected.lock().unwrap_or_else(|e| e.into_inner()).get(model_ref).cloned()?;
        let (name_family, name_params, _) = openhydra_protocol::model_id::parse_hf_model_name(&m.engine_ref);
        let or = |a: &str, b: String| if a.trim().is_empty() { b } else { a.to_string() };
        let quant = if m.quant.trim().is_empty() {
            self.declared_quant.clone().or_else(|| quant_hint_from_name(&m.engine_ref)).unwrap_or_default()
        } else {
            m.quant.clone()
        };
        Some(ModelInfo {
            family: or(&m.family, name_family),
            params: or(&m.params, name_params),
            quant,
            weight_digest: m.weight_digest.clone(),
        })
    }

    /// Set the provider's settlement peg (USDC micro-units per credit unit) for the F2 floor check
    /// (builder form). `0`/unset leaves it `None` → monotonic-only cumulative validation.
    /// Settle paying receipts on `chain` (builder form): co-sign one only with the consumer's valid
    /// on-chain claim for that deployment, and serve only consumers settling there.
    pub fn with_settlement_chain(mut self, chain: Option<openhydra_protocol::settlement::ChainDomain>) -> Self {
        self.settlement_chain = chain;
        self
    }

    /// Name this machine's own gateway (builder form), exempt from the payment gate.
    pub fn with_self_consumer(mut self, peer: Option<String>) -> Self {
        self.self_consumer = peer.filter(|p| !p.trim().is_empty());
        self
    }

    pub fn with_settlement_peg(mut self, micros_per_credit: u64) -> Self {
        self.settlement_peg = (micros_per_credit > 0).then_some(micros_per_credit);
        self
    }

    /// M4-base reach gate: enforce the model's `scope` for an inbound serve from `source_peer` (the
    /// libp2p-authenticated sender). Returns `None` to proceed, or `Some(framed error)` to refuse.
    /// Delegates the decision to the pure [`scope_gate`]; here we just supply this node's identity +
    /// live authorizer, frame the refusal, and bump the swarm-refusal counter.
    /// Remember serves that went unpaid (a provider that charges only), for the credit window.
    fn note_defaults(&self, peers: Vec<String>, now: u64) {
        if self.settlement_peg.is_none() || peers.is_empty() {
            return;
        }
        let mut defaults = self.payment_defaults.lock().unwrap_or_else(|e| e.into_inner());
        for peer in peers {
            if defaults.len() >= MAX_DEFAULT_PEERS && !defaults.contains_key(&peer) {
                defaults.retain(|_, at| at.iter().any(|t| now.saturating_sub(*t) <= DEFAULT_WINDOW_MS));
                if defaults.len() >= MAX_DEFAULT_PEERS {
                    let oldest = defaults
                        .iter()
                        .min_by_key(|(_, at)| at.iter().max().copied().unwrap_or(0))
                        .map(|(k, _)| k.clone());
                    if let Some(k) = oldest {
                        defaults.remove(&k);
                    }
                }
            }
            let at = defaults.entry(peer).or_default();
            at.retain(|t| now.saturating_sub(*t) <= DEFAULT_WINDOW_MS);
            at.push(now);
        }
    }

    /// Refuse a serve whose nonce was used before: admitted recently (see `seen_nonces`), or still
    /// backing a commitment, a stream or a buffered result (review round 5, R5S-1). Our own consumer
    /// is exempt (it never settles with us). An admitted serve's nonce is remembered once the payment
    /// gate lets it through ([`Self::remember_nonce`]).
    ///
    /// The check and the reservation are one step under the `seen_nonces` lock (review round 6,
    /// R6-1): two concurrent serves with one nonce can't both pass. A serve the payment gate then
    /// refuses gives its nonce back ([`Self::release_nonce`]), so refused requests can't push real
    /// nonces out of the memory.
    fn nonce_refusal(&self, req: &ServeRequest, source_peer: &str) -> Option<Vec<u8>> {
        if self.is_own_consumer(source_peer) {
            return None;
        }
        let n = &req.nonce;
        let mut seen = self.seen_nonces.lock().unwrap_or_else(|e| e.into_inner());
        let now = now_unix_ms();
        let used = seen.seen(n, now)
            || self.serving_nonces.lock().unwrap_or_else(|e| e.into_inner()).contains(n)
            || self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner()).contains_key(n)
            || self.streams.lock().unwrap_or_else(|e| e.into_inner()).contains_key(n)
            || self.results.lock().unwrap_or_else(|e| e.into_inner()).contains_key(n);
        if !used {
            seen.remember(*n, now);
        }
        used.then(|| {
            frame_response(&[ServeChunk::Error("this request's nonce was already used; send each request with a new one".into()).encode()])
        })
    }

    /// Hold `nonce` while its buffered serve generates (see `serving_nonces`).
    fn hold_nonce(&self, nonce: [u8; 16], source_peer: &str) {
        if !self.is_own_consumer(source_peer) {
            self.serving_nonces.lock().unwrap_or_else(|e| e.into_inner()).insert(nonce);
        }
    }

    /// The buffered serve under `nonce` ended: its nonce is remembered from now (its commitment, if
    /// any, holds it meanwhile too).
    fn end_hold(&self, nonce: [u8; 16], source_peer: &str) {
        if !self.is_own_consumer(source_peer) {
            let mut seen = self.seen_nonces.lock().unwrap_or_else(|e| e.into_inner());
            self.serving_nonces.lock().unwrap_or_else(|e| e.into_inner()).remove(&nonce);
            seen.remember(nonce, now_unix_ms());
        }
    }

    /// Give back the nonce of a serve the payment gate refused (see [`Self::nonce_refusal`]).
    fn release_nonce(&self, nonce: [u8; 16], source_peer: &str) {
        if !self.is_own_consumer(source_peer) {
            let mut seen = self.seen_nonces.lock().unwrap_or_else(|e| e.into_inner());
            seen.at.remove(&nonce);
            // Its place in the order goes too (it is the newest, unless another came meanwhile): refused
            // requests mustn't pile up there (R7-5).
            if seen.order.back().is_some_and(|(n, _)| *n == nonce) {
                seen.order.pop_back();
            }
        }
    }

    /// The payment gate of a provider that charges (a settlement peg is configured): the request
    /// must offer to pay at our peg (on our deployment), and its consumer must be within the credit
    /// window ([`payment_window`]). An admitted paid serve holds a [`PaymentReservation`] while it
    /// runs, grown as it generates ([`Self::grow_reservation`]). Our own (co-located) consumer never settles
    /// with us and is exempt. `Err` is the refusal.
    fn payment_gate(&self, req: &ServeRequest, source_peer: &str) -> Result<Option<PaymentReservation>, Vec<u8>> {
        let Some(peg) = self.settlement_peg else {
            return Ok(None);
        };
        if self.is_own_consumer(source_peer) {
            return Ok(None);
        }
        let refuse = |reason: String| {
            frame_response(&[ServeChunk::Error(format!("{}{reason}", crate::serve::PAYMENT_REFUSAL)).encode()])
        };
        if req.settlement_peg != Some(peg) {
            return Err(refuse(format!(
                "this provider charges (settlement peg {peg} micro-USDC per credit); the request doesn't offer to pay it"
            )));
        }
        if let Some(chain) = &self.settlement_chain {
            if req.settlement_domain.as_deref() != Some(chain.fingerprint().as_str()) {
                return Err(refuse(format!(
                    "this provider settles on {} deployment {}; the request doesn't settle there",
                    chain.chain().label(),
                    chain.fingerprint()
                )));
            }
        }
        if let Some(reason) = self.wallet_refusal(req) {
            return Err(refuse(reason));
        }
        let on_solana = matches!(self.settlement_chain, Some(openhydra_protocol::settlement::ChainDomain::Solana(_)));
        if on_solana && req.model_ref.len() > openhydra_protocol::settlement::MAX_CLAIM_MODEL_ID_LEN {
            return Err(refuse(format!(
                "this provider can't take payment for a model name longer than {} bytes",
                openhydra_protocol::settlement::MAX_CLAIM_MODEL_ID_LEN
            )));
        }
        let now = now_unix_ms();
        // The running serves are held for the whole decision (lock order: reservations, streams,
        // commitments), so one ending meanwhile is counted once, not missed (round 8 S8-4).
        let mut generating = self.paid_generating.lock().unwrap_or_else(|e| e.into_inner());
        let running: Vec<Running> = generating.get(source_peer).cloned().unwrap_or_default();
        let generated = generated_of(&self.streams.lock().unwrap_or_else(|e| e.into_inner()), source_peer);
        let (unsettled, full): (Vec<Unsettled>, bool) = {
            let mut map = self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner());
            self.note_defaults(prune_expired_commitments(&mut map, now), now);
            let full = map.len() >= MAX_COMMITMENTS && oldest_undelivered(&map).is_none();
            (unsettled_of(&map, source_peer, &running, &generated), full)
        };
        // A full commitment map can't record the serve, so it couldn't be paid for or seen.
        if full {
            return Err(refuse("this provider has too many unpaid serves outstanding; retry later".into()));
        }
        let defaults = self
            .payment_defaults
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(source_peer)
            .map_or(0, |at| at.iter().filter(|t| now.saturating_sub(**t) <= DEFAULT_WINDOW_MS).count());
        let tokens = paid_reservation(req.max_tokens);
        let running: Vec<u64> = running.iter().map(|r| r.tokens).collect();
        payment_window(&unsettled, &running, tokens, defaults, now).map_err(refuse)?;
        let id = self.next_reservation.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        generating.entry(source_peer.to_string()).or_default().push(Running { id, tokens, nonce: req.nonce });
        Ok(Some(PaymentReservation {
            generating: Arc::clone(&self.paid_generating),
            peer: source_peer.to_string(),
            id,
            initial: tokens,
        }))
    }

    /// Grow `r` to `to` tokens if its consumer's credit window has room: what it owes on its
    /// unsettled serves and its other running reservations, plus `to`, within
    /// [`MAX_UNSETTLED_TOKENS`]. A paid serve calls this as it generates past its reservation, and
    /// stops when it can't (review round 3, S2).
    fn grow_reservation(&self, r: &PaymentReservation, to: u64) -> bool {
        // Held throughout, as in `payment_gate` (round 8 S8-4).
        let mut generating = self.paid_generating.lock().unwrap_or_else(|e| e.into_inner());
        let running: Vec<Running> = generating.get(&r.peer).cloned().unwrap_or_default();
        let generated = generated_of(&self.streams.lock().unwrap_or_else(|e| e.into_inner()), &r.peer);
        let unsettled: u64 = unsettled_of(&self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner()), &r.peer, &running, &generated)
            .iter()
            .map(|u| u.tokens)
            .sum();
        let Some(mine) = generating.get_mut(&r.peer) else {
            return false;
        };
        let others: u64 = mine.iter().filter(|o| o.id != r.id).map(|o| o.tokens).sum();
        if unsettled.saturating_add(others).saturating_add(to) > MAX_UNSETTLED_TOKENS {
            return false;
        }
        if let Some(e) = mine.iter_mut().find(|o| o.id == r.id) {
            e.tokens = to;
        }
        true
    }

    /// The budget a serve runs under ([`crate::serve::handle_serve_request_within`]): unlimited for a
    /// free serve; for a paid one, its reservation, grown a step at a time while the window allows.
    fn serve_budget<'a>(&'a self, reservation: Option<&'a PaymentReservation>) -> impl FnMut(u64) -> bool + 'a {
        let mut reserved = reservation.map_or(0, |r| r.initial);
        move |sent| match reservation {
            None => true,
            Some(_) if sent < reserved => true,
            Some(r) => {
                let want = reserved + RESERVE_STEP_TOKENS;
                let ok = self.grow_reservation(r, want);
                if ok {
                    reserved = want;
                }
                ok
            }
        }
    }

    /// On Base, the request must name the wallet its claims will be signed with: the receipt's claim
    /// must come from it, and each wallet has its own running total (as on-chain).
    fn wallet_refusal(&self, req: &ServeRequest) -> Option<String> {
        if !matches!(self.settlement_chain, Some(openhydra_protocol::settlement::ChainDomain::Base(_))) {
            return None;
        }
        match req.settlement_wallet.as_deref().and_then(parse_evm_address) {
            Some(_) => None,
            None => Some("this provider settles on Base; the request names no wallet to pay from".into()),
        }
    }

    /// The reply to an identical receipt co-signed recently (see `recent_cosigns`).
    fn recent_cosign(&self, key: &[u8; 32]) -> Option<Vec<u8>> {
        let now = now_unix_ms();
        let m = self.recent_cosigns.lock().unwrap_or_else(|e| e.into_inner());
        m.get(key).filter(|(_, at)| now.saturating_sub(*at) <= COMMITMENT_TTL_MS).map(|(r, _)| r.clone())
    }

    /// Remember the reply to a receipt just co-signed, keeping at most [`MAX_COMMITMENTS`].
    fn note_cosign(&self, key: [u8; 32], reply: &[u8]) {
        let now = now_unix_ms();
        let mut m = self.recent_cosigns.lock().unwrap_or_else(|e| e.into_inner());
        m.retain(|_, (_, at)| now.saturating_sub(*at) <= COMMITMENT_TTL_MS);
        if m.len() < MAX_COMMITMENTS {
            m.insert(key, (reply.to_vec(), now));
        }
    }

    /// Remember that we served `peer` now (see [`SERVED_ATTEST_PRIORITY_MS`]).
    fn note_served(&self, peer: &str) {
        let now = now_unix_ms();
        let mut m = self.recently_served.lock().unwrap_or_else(|e| e.into_inner());
        if !m.contains_key(peer) && m.len() >= MAX_RECENTLY_SERVED {
            m.retain(|_, at| now.saturating_sub(*at) <= SERVED_ATTEST_PRIORITY_MS);
            if m.len() >= MAX_RECENTLY_SERVED {
                if let Some(oldest) = m.iter().min_by_key(|(_, at)| **at).map(|(k, _)| k.clone()) {
                    m.remove(&oldest);
                }
            }
        }
        m.insert(peer.to_string(), now);
    }

    /// Whether we served `peer` within [`SERVED_ATTEST_PRIORITY_MS`].
    fn served_recently(&self, peer: &str) -> bool {
        let now = now_unix_ms();
        self.recently_served
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(peer)
            .is_some_and(|at| now.saturating_sub(*at) <= SERVED_ATTEST_PRIORITY_MS)
    }

    /// Whether `peer` is this node, or our own co-located consumer (the desktop's gateway): it never
    /// settles with us.
    fn is_own_consumer(&self, peer: &str) -> bool {
        peer == self.net.libp2p_peer_id() || self.self_consumer.as_deref() == Some(peer)
    }

    /// `peer` got this buffered serve's completion (the reply reached it, or its fetch by nonce
    /// did): mark the commitment delivered. A buffered serve's commitment is recorded before its
    /// reply goes out, so there is always one to mark; a nonce without one (a refusal, a failed
    /// serve) is left alone.
    fn mark_delivered(&self, nonce: [u8; 16], peer: &str) {
        let mut map = self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = map.get_mut(&nonce).filter(|c| c.served_peer == peer) {
            c.delivered = true;
            drop(map);
            self.note_served(peer);
        }
    }

    /// A poll returned stream frames no earlier one did: the consumer now has them, and owes for
    /// them ([`crate::serve::StreamTally`]). Records the stream's commitment, or raises it, as
    /// delivered (whatever it read, it had). Called with the streams lock held (see
    /// [`Self::handle_fetch_chunks`]), so a receipt settling the stream meanwhile can't leave a
    /// commitment behind it.
    fn commit_fetched(&self, nonce: [u8; 16], peer: &str, fetched: Fetched) {
        if fetched.tokens == 0 || self.is_own_consumer(peer) {
            return;
        }
        // It read output from us: served, for attestation priority (output it never read isn't).
        self.note_served(peer);
        let now = now_unix_ms();
        let mut map = self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = map.get_mut(&nonce).filter(|c| c.served_peer == peer) {
            c.tokens = fetched.tokens;
            c.recorded_ms = now;
            return;
        }
        self.note_defaults(prune_expired_commitments(&mut map, now), now);
        insert_commitment(
            &mut map,
            nonce,
            ServeCommitment {
                model_id: fetched.model_id,
                tokens: fetched.tokens,
                recorded_ms: now,
                served_peer: peer.to_string(),
                delivered: true,
                wallet: fetched.wallet,
            },
        );
    }

    /// Whether the stream under `nonce` should keep generating: its consumer still polls it (within
    /// [`STREAM_ABANDON_MS`]), and it is still open (not settled, evicted, or expired).
    fn stream_wanted(&self, nonce: &[u8; 16]) -> bool {
        let now = now_unix_ms();
        self.streams
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(nonce)
            .is_some_and(|b| {
                let limit = if b.polled { STREAM_ABANDON_MS } else { NEVER_POLLED_ABANDON_MS };
                !b.closed && now.saturating_sub(b.last_poll_ms) <= limit
            })
    }

    /// Handle a [`STREAM_CLOSE`]: `payload` = `nonce(16) ‖ offset(4, big-endian)`.
    fn handle_stream_close(&self, source_peer: &str, payload: &[u8]) -> Vec<u8> {
        if payload.len() != 20 {
            return FetchChunksResponse::NotFound.encode();
        }
        let nonce: [u8; 16] = payload[0..16].try_into().unwrap();
        let offset = u32::from_be_bytes(payload[16..20].try_into().unwrap());
        if self.fetch_rl.try_acquire(source_peer, now_unix_ms()).is_err() {
            return FetchChunksResponse::Overloaded.encode(); // the consumer retries its close
        }
        let now = now_unix_ms();
        let mut map = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, b| !stream_expired(b, now));
        let reply = close_stream_in(&mut map, &nonce, source_peer, offset, now);
        // Its consumer is here to settle it: the commitment is kept as long as from a poll, so
        // closes retried through an outage still find it (round 8 S8-3).
        // Once per stream (round 9 B9-5): closes sent again can't keep it from ever expiring.
        if matches!(reply, FetchChunksResponse::Chunks { .. }) {
            let first = {
                let mut done = self.close_refreshed.lock().unwrap_or_else(|e| e.into_inner());
                if done.len() >= 2 * MAX_COMMITMENTS {
                    done.clear();
                }
                done.insert(nonce)
            };
            if let Some(c) = self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&nonce) {
                if c.served_peer == source_peer && first {
                    c.recorded_ms = now;
                }
            }
        }
        reply.encode()
    }

    fn scope_refusal(&self, req: &ServeRequest, source_peer: &str) -> Option<Vec<u8>> {
        // Resolve the effective scope. A `model_ref` that exactly matches a model our engine serves
        // (detected, or announced) uses that model's own scope: a private model is never announced
        // globally, so the announced set alone would gate it at the strictest scope (round 8 N8-1).
        // A ref that does NOT — yet `is_shared` let it through (e.g. `All`
        // mode says yes to anything, or a single-model engine serves its one model for any name) —
        // could be an alias the engine resolves to any shared model, so it is gated at the STRICTEST
        // scope this node shares (M4 review HIGH: closes an is_shared/scope_of divergence that would
        // otherwise serve a private model under an unrecognised alias with no credential). An
        // all-Global node's strictest is Global, so a purely-public node sees no change.
        let known = self.detected.lock().map(|d| d.contains_key(&req.model_ref)).unwrap_or(false)
            || self.last_announced.lock().map(|s| s.contains(&req.model_ref)).unwrap_or(false);
        let scope = match known.then(|| self.policy.scope_of(&req.model_ref)) {
            // Global without the operator's consent to publish isn't public yet: gated as an unknown
            // name would be (round 9 C9-6).
            Some(crate::share_policy::Scope::Global) if !self.policy.announce_globally(&req.model_ref) => self.policy.strictest_scope(),
            Some(scope) => scope,
            None => self.policy.strictest_scope(),
        };
        match scope_gate(
            scope,
            &req.model_ref,
            source_peer,
            self.net.libp2p_peer_id(),
            req.credential.as_ref(),
            self.swarm_auth.as_deref(),
            now_unix_ms(),
        ) {
            Ok(()) => {
                // Only a model our engine serves, under its own name: an alias let through at the
                // strictest scope could be any string, and isn't a model a challenge names.
                if known
                    && matches!(scope, crate::share_policy::Scope::Private | crate::share_policy::Scope::Unknown)
                    && req.model_ref.len() <= crate::attest::MAX_MODEL_ID_LEN
                {
                    self.note_private_member(source_peer, &req.model_ref, req.credential.clone());
                }
                None
            }
            Err((reason, is_swarm_refusal)) => {
                if is_swarm_refusal {
                    if let Some(stats) = &self.stats {
                        stats.swarm_refusals.fetch_add(1, Ordering::Relaxed);
                    }
                }
                Some(frame_response(&[ServeChunk::Error(reason).encode()]))
            }
        }
    }

    /// Remember that `peer` passed the swarm gate for private `model` with `credential`, for
    /// attestation.
    fn note_private_member(
        &self,
        peer: &str,
        model: &str,
        credential: Option<openhydra_network::membership::MembershipCredential>,
    ) {
        let mut m = self.private_members.lock().unwrap_or_else(|e| e.into_inner());
        note_private_member_in(&mut m, peer, model, now_unix_ms(), credential);
    }

    /// Whether `peer` may attest private `model`: it passed the model's swarm gate recently
    /// ([`private_member_in`]) and would still pass it now — the model still shared and private, and
    /// its credential not revoked or expired since.
    fn private_attestable(&self, peer: &str, model: &str) -> bool {
        use crate::share_policy::Scope;
        let now = now_unix_ms();
        let credential = {
            let m = self.private_members.lock().unwrap_or_else(|e| e.into_inner());
            match private_member_in(&m, peer, model, now) {
                Some(c) => c.clone(),
                None => return false,
            }
        };
        let scope = self.policy.scope_of(model);
        self.model_shared(model)
            && matches!(scope, Scope::Private | Scope::Unknown)
            && scope_gate(
                scope,
                model,
                peer,
                self.net.libp2p_peer_id(),
                credential.as_ref(),
                self.swarm_auth.as_deref(),
                now,
            )
            .is_ok()
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
    fn record_commitment(
        &self,
        nonce: [u8; 16],
        model_id: String,
        tokens: u64,
        served_peer: &str,
        delivered: bool,
        wallet: Option<[u8; 20]>,
    ) {
        if delivered {
            self.note_served(served_peer);
        }
        // Our own consumer never settles with us: its serves would only fill the map.
        if self.is_own_consumer(served_peer) {
            return;
        }
        let now = now_unix_ms();
        let mut map = self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner());
        self.note_defaults(prune_expired_commitments(&mut map, now), now);
        insert_commitment(
            &mut map,
            nonce,
            ServeCommitment { model_id, tokens, recorded_ms: now, served_peer: served_peer.to_string(), delivered, wallet },
        );
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
        let resp = {
            let mut map = self.results.lock().unwrap_or_else(|e| e.into_inner());
            map.retain(|_, r| now.saturating_sub(r.recorded_ms) <= RESULT_TTL_MS);
            fetch_from_buffer(&map, &nonce, source_peer)
        };
        let encoded = resp.encode();
        if FetchResponse::encoded_is_ready(&encoded) {
            let mut fetches = self.result_fetches.lock().unwrap_or_else(|e| e.into_inner());
            if fetches.len() >= 2 * MAX_RESULT_ENTRIES {
                fetches.clear();
            }
            let n = fetches.entry(nonce).or_insert(0);
            *n = n.saturating_add(1);
            if *n > MAX_RESULT_FETCHES {
                return FetchResponse::NotFound.encode();
            }
        }
        encoded
    }

    /// P1 streaming: admit a new stream under `nonce` (empty, `done=false`). Called at the start
    /// of an accepted `SERVE_STREAM` generation job, so the buffer exists before the ack is sent
    /// and never lingers without a producer (a shed submit never reaches this).
    fn stream_begin(&self, nonce: [u8; 16], reply_to: &str, billing: StreamBilling) {
        let now = now_unix_ms();
        let mut map = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        stream_begin_billed_in(&mut map, nonce, reply_to, now, billing);
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
        // A shed poll doesn't count as reading (review round 4 M4): a consumer that floods polls so
        // its real ones are shed can't keep a stream generating while it fetches nothing.
        if self.fetch_rl.try_acquire(source_peer, now_unix_ms()).is_err() {
            return FetchChunksResponse::Chunks { framed: Vec::new(), next_offset: offset, done: false }
                .encode();
        }
        let now = now_unix_ms();
        let mut map = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        map.retain(|_, b| !stream_expired(b, now));
        let (resp, fetched) = fetch_chunks_billed(&mut map, &nonce, source_peer, offset, now);
        // What it was just sent is owed (streams lock still held: see `commit_fetched`).
        if let Some(fetched) = fetched {
            self.commit_fetched(nonce, source_peer, fetched);
        }
        drop(map);
        resp.encode()
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
        // M4-base reach gate: a `Private` model needs a valid swarm credential bound to this sender;
        // a `Device` model is loopback-only. Runs after `is_shared` (which-models) — orthogonal.
        if let Some(refusal) = self.scope_refusal(&req, &source_peer) {
            respond_done(refusal, 1);
            return;
        }
        if let Some(refusal) = self.nonce_refusal(&req, &source_peer) {
            respond_done(refusal, 1);
            return;
        }
        // A provider that charges: the consumer must offer to pay, and have paid for what it had.
        let reservation = match self.payment_gate(&req, &source_peer) {
            Ok(r) => r,
            Err(refusal) => {
                self.release_nonce(req.nonce, &source_peer);
                respond_done(refusal, 1);
                return;
            }
        };
        // Capture the small fields needed after `req` is moved into the serve (which consumes
        // `messages`), for the ack, buffer, commitment, and stats.
        let nonce = req.nonce;
        let model_ref = req.model_ref.clone();
        let reply_to = req.reply_to.clone();
        let billing = StreamBilling {
            model_id: model_ref.clone(),
            max_tokens: req.max_tokens,
            wallet: req.settlement_wallet.as_deref().and_then(parse_evm_address),
        };
        // Create the buffer, then ack immediately (before generating), so the consumer starts
        // polling right away and never races an absent buffer (begin happens before the ack).
        self.stream_begin(nonce, &reply_to, billing);
        let _ = self.net.respond(
            request_id,
            FetchChunksResponse::Chunks { framed: Vec::new(), next_offset: 0, done: false }.encode(),
        );
        // Generate, appending each frame (deltas, then the terminal Done/Error which flips `done`).
        // F5: `generate_guarded` catches a panic in the adapter/engine and appends a synthetic
        // terminal Error, so a bug can't leave this buffer `done=false` forever (which would stall
        // the consumer ~120s before it re-serves).
        // It stops once the consumer stops reading, or settles for what it read (the stream is then
        // closed): nobody would read the rest.
        let mut budget = self.serve_budget(reservation.as_ref());
        let summary = generate_guarded(req, &self.adapter, &mut |c| self.stream_append(nonce, c), &mut |sent| {
            self.stream_wanted(&nonce) && budget(sent)
        });
        // B-S1: the serve's commitment (what its settlement receipt is checked against) is what the
        // consumer's polls returned, recorded as they return it (`commit_fetched`): a stream it read
        // only part of is owed for that part, and one it read nothing of is owed nothing.
        if !self.stream_wanted(&nonce) {
            let owed = self.committed_tokens(&nonce);
            tracing::info!(peer = %source_peer, generated = summary.tokens, owed = ?owed,
                "stream stopped: its consumer stopped reading or settled; owed what it read");
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
    /// `floor_tokens` is the commitment the payment floor was checked against
    /// ([`Self::validate_settlement_cumulative`]): if a poll raised it since, the receipt is refused
    /// (retryable), so a consumer can't pay the lower count while it fetches more (review round 4
    /// S4-7).
    fn check_and_consume_commitment(
        &self,
        p: &openhydra_protocol::receipts::ReceiptPayload,
        source_peer: &str,
        floor_tokens: Option<u64>,
    ) -> Result<(), String> {
        let consumer_peer = self.net.peer_id_from_ed25519_pubkey(p.consumer.as_bytes())?;
        // The streams lock first, as a poll takes them (see `commit_fetched`): a stream settled here is
        // closed in the same step, so no later poll can open a new commitment for it, and it stops
        // generating (its consumer has what it is paying for).
        let mut streams = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        let mut map = self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner());
        self.note_defaults(prune_expired_commitments(&mut map, now_unix_ms()), now_unix_ms());
        if map.get(&p.nonce).map(|c| c.tokens) != floor_tokens {
            return Err("the serve's count changed while this receipt was checked; retry".into());
        }
        validate_and_consume_commitment(
            &mut map,
            &p.nonce,
            &p.model_id,
            p.tokens,
            p.ts_unix_ms,
            now_unix_ms(),
            &consumer_peer,
            source_peer,
        )?;
        if streams.get(&p.nonce).is_some_and(|b| b.reply_to == source_peer) {
            streams.remove(&p.nonce);
        }
        Ok(())
    }

    /// The `(consumer, model)` key for the provider-side settled-cumulative map (review F2), scoped to
    /// the deployment paying receipts settle on: each deployment's escrow starts from nothing, so a
    /// running total built under one must not carry over to another (the first claim there would
    /// re-authorize the whole history).
    /// On Base, also by the wallet that signs the pair's claims (`wallet`): on-chain, what was paid
    /// so far is kept per wallet, so a new wallet's running total starts from nothing there, and must
    /// here too, or its first claim would pay the old wallet's history again.
    fn settled_key(&self, p: &openhydra_protocol::receipts::ReceiptPayload, wallet: Option<[u8; 20]>) -> String {
        let key = openhydra_protocol::verify::reputation_key(&hex::encode(p.consumer.as_bytes()), &p.model_id);
        match (&self.settlement_chain, wallet) {
            (Some(chain), Some(w)) => format!("{key}@{}@0x{}", chain.fingerprint(), hex::encode(w)),
            (Some(chain), None) => format!("{key}@{}", chain.fingerprint()),
            (None, _) => key,
        }
    }

    /// The wallet the serve under `nonce` was requested with, while its commitment is open.
    fn committed_wallet(&self, nonce: &[u8; 16]) -> Option<[u8; 20]> {
        self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner()).get(nonce).and_then(|c| c.wallet)
    }

    /// The receipt must come from its consumer: the peer its key derives, authenticated as the sender.
    fn check_sender(&self, p: &openhydra_protocol::receipts::ReceiptPayload, source_peer: &str) -> Result<(), String> {
        let consumer_peer = self.net.peer_id_from_ed25519_pubkey(p.consumer.as_bytes())?;
        if consumer_peer != source_peer {
            return Err("receipt is not from the consumer that signed it".into());
        }
        Ok(())
    }

    /// Review F2: validate a receipt's `cumulative_amount` against the last one we co-signed for this
    /// `(consumer, model)` **before** counter-signing — monotonic, and (when a peg is configured)
    /// above the token floor, so a consumer can't under-sign or replay a lower amount and have us
    /// accept it. Read-only (the accepted value is recorded only after a successful co-sign). A
    /// `None` cumulative (legacy / crypto-rail-off receipt) always passes.
    /// The completion tokens we recorded serving under this receipt's nonce (our own count), if
    /// the commitment is still open. Read without consuming it.
    fn committed_tokens(&self, nonce: &[u8; 16]) -> Option<u64> {
        self.serve_commitments.lock().unwrap_or_else(|e| e.into_inner()).get(nonce).map(|c| c.tokens)
    }

    /// On success, the commitment the floor was taken from (`None`: there was none).
    fn validate_settlement_cumulative(
        &self,
        p: &openhydra_protocol::receipts::ReceiptPayload,
        wallet: Option<[u8; 20]>,
    ) -> Result<Option<u64>, BindRejection> {
        // v5 denomination binding: reject a crypto-rail receipt whose declared asset/peg we never
        // agreed to, before `check_cumulative` and before the serve-commitment nonce is consumed.
        check_settlement_denomination(
            p.cumulative_amount,
            p.settlement_asset,
            p.peg_micros_per_credit,
            self.settlement_peg,
        )?;
        let key = self.settled_key(p, wallet);
        let last = self
            .settled_cumulative
            .lock()
            .ok()
            .and_then(|m| m.get(&key).copied())
            .unwrap_or(0);
        // The floor is what we served, by our own count: the receipt's `tokens` and its
        // completion/prompt split are the consumer's to fill in, so a floor built from them could
        // be talked down to zero. Weighted like the consumer's own step (blended rate, no split).
        // No open commitment → no floor here; the commitment check then rejects the receipt.
        let committed = self.committed_tokens(&p.nonce);
        let units = settlement_floor_units(committed, p.tokens);
        openhydra_protocol::payout::check_cumulative(last, p.cumulative_amount, units, self.settlement_peg)
            .map_err(|e| {
                // A consumer behind our high-water (a co-sign reply it never got, a voucher it lost)
                // would be refused on every receipt from here on: send it the receipt that set the
                // high-water, which it signed, so it can sign on top of it.
                // On a settlement chain its claim goes with it: the claim is what shows the consumer
                // the receipt was signed for this deployment.
                let latest = self.settled_receipts.lock().unwrap_or_else(|e| e.into_inner()).get(&key).cloned();
                match (e, latest) {
                    (_, Some(r)) if r.payload.cumulative_amount == Some(last) => {
                        let claim = self
                            .settled_claims
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .get(&key)
                            .filter(|k| k.cumulative == last)
                            .map(|k| crate::receipt::ClaimSig { expires_at: k.expires_at, sig: k.sig.clone() });
                        BindRejection::Behind(Box::new(crate::receipt::LatestReceipt { receipt: r, claim }))
                    }
                    (openhydra_protocol::payout::CumulativeError::NonMonotonic, _) => {
                        "cumulative amount is non-monotonic".into()
                    }
                    (openhydra_protocol::payout::CumulativeError::BelowFloor, _) => {
                        "cumulative amount below the floor for tokens served".into()
                    }
                }
            })
            .map(|()| committed)
    }

    /// Record the co-signed cumulative for this `(consumer, model)` after a successful co-sign
    /// (review F2). Monotonic `max`, so a concurrent or out-of-order co-sign can't lower it;
    /// best-effort persisted so a restart can't accept a lower/replayed cumulative.
    /// On a settlement chain, a paying receipt must carry the consumer's valid on-chain claim for it.
    /// Returns the claim to keep (`None` off-chain). On Base it must be signed by the wallet the serve
    /// was requested with, and its running total is that wallet's own ([`Self::settled_key`]).
    fn check_settlement_claim(
        &self,
        p: &openhydra_protocol::receipts::ReceiptPayload,
        claim: Option<&crate::receipt::ClaimSig>,
    ) -> Result<Option<KeptClaim>, String> {
        let (Some(chain), Some(cumulative)) = (&self.settlement_chain, p.cumulative_amount) else {
            return Ok(None);
        };
        let c = claim.ok_or("this provider settles on-chain: the receipt carries no claim")?;
        let settlement = crate::receipt::claim_for(p, c.expires_at).ok_or("no cumulative")?;
        let signer = openhydra_protocol::settlement::verify_claim(chain, &settlement, &c.sig, now_unix_ms() / 1000)
            .map_err(|e| format!("the receipt's on-chain claim is not valid here: {e:?}"))?;
        let signer = match signer {
            openhydra_protocol::settlement::ClaimSigner::Evm(addr) => Some(addr),
            openhydra_protocol::settlement::ClaimSigner::Solana => None,
        };
        if let Some(signer) = signer {
            match self.committed_wallet(&p.nonce) {
                Some(w) if w == signer => {}
                Some(w) => {
                    return Err(format!(
                        "the claim is signed by wallet 0x{}, not 0x{} the serve was requested with",
                        hex::encode(signer),
                        hex::encode(w)
                    ))
                }
                None => return Err("the serve named no wallet for this claim".into()),
            }
        }
        Ok(Some(KeptClaim { cumulative, expires_at: c.expires_at, signer, sig: c.sig.clone() }))
    }

    fn record_settlement_cumulative(
        &self,
        receipt: &openhydra_protocol::receipts::CoSignedReceipt,
        claim: Option<&KeptClaim>,
    ) {
        let p = &receipt.payload;
        let Some(cumulative) = p.cumulative_amount else {
            return;
        };
        let key = self.settled_key(p, claim.and_then(|c| c.signer));
        let raised = {
            let mut map = match self.settled_cumulative.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            let e = map.entry(key.clone()).or_insert(0);
            let raised = cumulative >= *e;
            *e = (*e).max(cumulative);
            raised
        };
        // Monotonic `max`: only a receipt at the new high-water replaces the kept one.
        if !raised {
            return;
        }
        self.settled_receipts.lock().unwrap_or_else(|e| e.into_inner()).insert(key.clone(), receipt.clone());
        // A verified claim replaces the kept one; a receipt without one leaves it (still settleable).
        if let Some(c) = claim {
            tracing::debug!(consumer = %hex::encode(p.consumer.as_bytes()), provider = %hex::encode(p.provider.as_bytes()),
                model = %p.model_id, cumulative, expires_at = c.expires_at, sig = %hex::encode(&c.sig),
                "kept the on-chain claim");
            self.settled_claims.lock().unwrap_or_else(|e| e.into_inner()).insert(key.clone(), c.clone());
        }
        if let Some(store) = &self.store {
            let blob = claim.map(KeptClaim::to_bytes);
            if let Err(e) = store.put_provider_settled_receipt(&key, cumulative, receipt, blob.as_deref()) {
                eprintln!("openhydra-agent: settled-cumulative persist failed: {e}");
            }
        }
    }

    /// Attach a ledger so accepted co-signed receipts are persisted (M2.3), and rehydrate
    /// the give/take credit map from it so throttling survives a restart.
    pub fn with_store(self, store: Store) -> Self {
        if let Ok(mut map) = self.credit.lock() {
            if let Err(e) = store.load_credit_into_memory(&mut map) {
                eprintln!("openhydra-agent: could not rehydrate credit from store: {e}");
            }
        }
        // Review F2: rehydrate the last co-signed cumulative per (consumer, model) so a restart can't
        // accept a lower/replayed cumulative than one already co-signed.
        if let Ok(mut map) = self.settled_cumulative.lock() {
            match store.load_provider_settled_into_memory(&mut map) {
                Ok(0) => {}
                Ok(quarantined) => eprintln!(
                    "openhydra-agent: quarantined {quarantined} corrupt settled-cumulative row(s) on rehydrate \
                     (fail-closed at max high-water until the row is fixed); kept the rest"
                ),
                Err(e) => eprintln!("openhydra-agent: could not rehydrate settled cumulatives from store: {e}"),
            }
        }
        if let Ok(mut map) = self.settled_receipts.lock() {
            match store.load_provider_settled_receipts_into_memory(&mut map) {
                Ok(0) => {}
                Ok(skipped) => eprintln!("openhydra-agent: skipped {skipped} undecodable settled receipt(s) on rehydrate"),
                Err(e) => eprintln!("openhydra-agent: could not rehydrate settled receipts from store: {e}"),
            }
        }
        if let Ok(mut map) = self.settled_claims.lock() {
            let mut raw = HashMap::new();
            match store.load_provider_settled_claims_into_memory(&mut raw) {
                Ok(()) => {
                    let total = raw.len();
                    map.extend(raw.into_iter().filter_map(|(k, b)| KeptClaim::from_bytes(&b).map(|c| (k, c))));
                    if map.len() < total {
                        eprintln!("openhydra-agent: skipped {} undecodable settled claim(s) on rehydrate", total - map.len());
                    }
                }
                Err(e) => eprintln!("openhydra-agent: could not rehydrate settled claims from store: {e}"),
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
        // Credit-tiering: the cost-weighted "credits earned" is weighted by the model the provider
        // ACTUALLY served — its own validated engine **handle** (`payload.model_id`, bound to the
        // serve commitment) — NOT the consumer-supplied `canonical_model_id` (which no commitment
        // check validates: a malicious consumer could downgrade it to underpay us — review H1/H3).
        // The handle-implied tier is conservative (assumes cheapest quant) and is the honest ceiling
        // until model-authenticity fingerprinting lands. Uses the signed completion split if present.
        let earned = openhydra_protocol::weight::credits_for_receipt(
            openhydra_protocol::weight::cost_tier_for_handle(&accepted.payload.model_id).weight(),
            accepted.payload.tokens,
            accepted.payload.completion_tokens,
        );
        if let Some(stats) = &self.stats {
            stats.add_credits_earned(earned);
        }
        // Give/take throttle = bandwidth priority in RAW tokens (review H2: the throttle constants
        // are token-scaled; cost-weighting them truncated earns and over-throttled consumers).
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
    /// Detect + announce the shared models and publish the status view. Serialized against a
    /// concurrent watcher/poll announce (blocking) — used by the startup announce and the
    /// filesystem-watcher thread (blocking on the poll thread's serve loop is fine).
    pub fn announce_models(&self) -> Result<usize, AdapterError> {
        let _serialize = self.announce_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.announce_models_inner()
    }

    /// The poll loop's announce: yields to an in-flight announce (the watcher's) via `try_lock`
    /// instead of blocking the serve loop. `None` = skipped because an announce is already running
    /// (it's publishing the fresh policy, so this tick has nothing to add).
    fn announce_models_polled(&self) -> Option<Result<usize, AdapterError>> {
        match self.announce_lock.try_lock() {
            Ok(_g) => Some(self.announce_models_inner()),
            Err(std::sync::TryLockError::WouldBlock) => None,
            Err(std::sync::TryLockError::Poisoned(e)) => {
                let _g = e.into_inner();
                Some(self.announce_models_inner())
            }
        }
    }

    fn announce_models_inner(&self) -> Result<usize, AdapterError> {
        let mut models = self.adapter.detect_models()?;
        // Snapshot the policy ONCE so the loop filter and the published view see one consistent
        // policy (also the source for `shared_models` below).
        let policy = self.policy.snapshot();
        // The weights digest log-prob references are matched to, for engines whose adapter reveals
        // where the weights are but not their digest (LM Studio): hashed in the background (cached)
        // and only for the models actually shared — never for everything an engine has on disk.
        if self.prompt_scoring {
            for m in models.iter_mut().filter(|m| m.weight_digest.is_empty() && policy.announce_globally(&m.engine_ref)) {
                if let Some(path) = self.adapter.weights_path(&m.engine_ref) {
                    m.weight_digest = crate::weights::weights_digest(&path);
                }
            }
        }
        *self.detected.lock().unwrap_or_else(|e| e.into_inner()) =
            models.iter().map(|m| (m.engine_ref.clone(), m.clone())).collect();
        // M6: the FULL intended global-announce set this pass (policy ∩ detected ∩ globally-shared),
        // computed up-front so it's independent of where the announce loop below might break on a
        // network error — a still-intended model we simply didn't *reach* must not look "withdrawn".
        // This is the diff basis for the tombstones published at the end. M1: this is the
        // *global-announce* set (`announce_globally`), not merely `is_shared` — a `private`/`device`
        // model is still served when reached (the serve gate stays `is_shared`) but is never
        // published to global discovery, so it must not appear here or get an advert/tombstone.
        let desired: BTreeSet<String> = models
            .iter()
            .filter(|m| policy.announce_globally(&m.engine_ref))
            .map(|m| m.engine_ref.clone())
            .collect();
        let mut announced: Vec<String> = Vec::new();
        let mut outcome: Result<(), AdapterError> = Ok(());
        for model in &models {
            // Global-announce gate (M1): publish an advert only for a model whose scope is `global`
            // and whose Global publish is consented. There is NO grandfather — a legacy file keeps
            // announcing only because `SharePolicy::load` materialises a consent record on migration.
            // `private`/`device` models are skipped here but still served if reached — the serve
            // path enforces `is_shared`, the security boundary; this is discovery hygiene.
            if !policy.announce_globally(&model.engine_ref) {
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
        // M6: publish a per-model tombstone for every model no longer intended (the operator
        // un-shared it, or it vanished from the engine for good), so consumers evict just that
        // `(peer, model)` entry immediately instead of on the ~300s TTL. F2: re-publish it over the
        // next few passes ([`plan_withdrawals`]) so a briefly-unmeshed consumer still gets it,
        // cancelling on re-share. Finding 2: the diff basis is the grace-adjusted `effective` set
        // ([`apply_absence_grace`]), so a transient detection blip is NOT tombstoned. Best-effort and
        // outcome-independent: a withdraw publish error is logged, never fatal, and the model would
        // still expire on the old TTL. `effective` is stored as the new baseline regardless of the
        // announce outcome so the next pass diffs against true intent.
        {
            let mut last = self.last_announced.lock().unwrap_or_else(|e| e.into_inner());
            let mut pending = self.pending_withdrawals.lock().unwrap_or_else(|e| e.into_inner());
            let mut absence = self.absence_streak.lock().unwrap_or_else(|e| e.into_inner());
            // Finding 2: damp a transient detection blip. A model dropped from `desired` because it
            // VANISHED from detection (a possible `auto`-mode partial-`Ok`) is held in `effective` for
            // a short grace so it isn't tombstoned; a model dropped because the POLICY changed (still
            // detected, no longer announceable) is withdrawn at once. Diff/baseline against `effective`.
            let detected: BTreeSet<String> = models.iter().map(|m| m.engine_ref.clone()).collect();
            let effective = apply_absence_grace(
                &last,
                &desired,
                &detected,
                &mut absence,
                WITHDRAW_ABSENCE_GRACE_PASSES,
            );
            for model in plan_withdrawals(&last, &effective, &mut pending, WITHDRAW_REPUBLISH_PASSES) {
                if let Err(e) = self.net.withdraw_model(model.clone()) {
                    eprintln!("openhydra-agent: MODEL_WITHDRAWN for {model} failed: {e}");
                }
            }
            *last = effective;
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
    /// R6/B2: block on a filesystem watcher and reload+re-announce the share policy the instant the
    /// file changes (a desktop toggle), rather than waiting out the poll tick. The poll loop's
    /// periodic `reload_if_changed` remains the portable fallback, so a watcher init/watch failure —
    /// or a missed event on a flaky filesystem — is non-fatal. No-op for a static policy.
    fn watch_policy_file(self: Arc<Self>)
    where
        A: Send + Sync + 'static,
    {
        use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
        let Some(path) = self.policy.watched_path() else { return };
        let Some(dir) = path.parent().map(|p| p.to_path_buf()) else { return };
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = match RecommendedWatcher::new(tx, Config::default()) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("openhydra-agent: share-policy watcher init failed ({e}) — using the poll fallback");
                return;
            }
        };
        // Watch the PARENT dir (non-recursive): the desktop writes atomically (temp → rename), so the
        // file's inode changes and watching it directly would miss the rename — the dir sees it.
        if let Err(e) = watcher.watch(&dir, RecursiveMode::NonRecursive) {
            eprintln!("openhydra-agent: share-policy watch({}) failed ({e}) — using the poll fallback", dir.display());
            return;
        }
        eprintln!("openhydra-agent: watching {} for share-policy changes (event-driven reload)", path.display());
        for res in rx {
            match res {
                // React only to events touching OUR file (not siblings like the ledger). Match by
                // FILE NAME, not full path: we watch a single dir, and FSEvents (macOS) may report a
                // canonicalized/firmlinked path (/System/Volumes/Data/…, /private/var vs /var) that
                // wouldn't byte-equal the configured path — a file-name match is immune to that and
                // still excludes the `.json.tmp` write. `announce_models` is serialized (announce_lock)
                // and `reload_if_changed` is atomic (reload_lock) + mtime-deduped, so a coincident
                // poll-loop reload/announce can't double-announce.
                Ok(event) if event.paths.iter().any(|p| p.file_name() == path.file_name()) => {
                    if self.policy.reload_if_changed() {
                        eprintln!("openhydra-agent: share policy changed (watcher) — re-announcing");
                        if let Err(e) = self.announce_models() {
                            eprintln!("openhydra-agent: re-announce after policy change failed: {e}");
                        }
                    }
                }
                Ok(_) => {}
                Err(e) => eprintln!("openhydra-agent: share-policy watch error: {e}"),
            }
        }
    }

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
        let control_pool = WorkerPool::new(CONTROL_WORKERS);
        let receipt_pool = WorkerPool::new(CONTROL_WORKERS);
        // R6/B2: event-driven share-policy reload. Spawn a filesystem watcher so a desktop toggle
        // applies in ~ms; the poll loop below keeps its periodic `reload_if_changed` as the portable
        // fallback (missed events / flaky-FS platforms). No-op for a static `--share-models` policy.
        {
            let watcher = Arc::clone(&self);
            std::thread::spawn(move || watcher.watch_policy_file());
        }
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
                    let queued = std::time::Instant::now();
                    let accepted = pool.submit(move || {
                        // Its consumer gave up waiting for the open while it queued: generating now
                        // would be for nobody.
                        if queued.elapsed() > STREAM_OPEN_STALE {
                            let _ = provider.net.respond(request_id, FetchChunksResponse::Overloaded.encode());
                            return;
                        }
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
                let is_attest = data.first() == Some(&ATTEST_CHALLENGE);
                let is_score = matches!(data.first(), Some(&PROMPT_SCORE_CHALLENGE) | Some(&MODEL_INFO_REQUEST));
                let is_stream_poll = matches!(data.first(), Some(&FETCH_CHUNKS) | Some(&STREAM_CLOSE));
                let is_fetch = data.first() == Some(&FETCH_RESULT);
                // Polls, closes, fetches and receipts are quick and must never wait behind
                // generations (each stream holds a serve worker while it generates): they have their
                // own small pool (round 8 G8-1).
                let target = if data.first() == Some(&RECEIPT_REQUEST) {
                    // Receipts sign and write the store: their own workers, so they never hold up
                    // polls (round 9 B9-2).
                    &receipt_pool
                } else if is_control_message(data.first().copied()) {
                    &control_pool
                } else {
                    &pool
                };
                let accepted = target.submit(move || {
                    // F-C5: decode the serve request ONCE here and thread it through
                    // throttle / AUP / dispatch, instead of each re-decoding the (possibly
                    // large) prompt. `None` for non-serve or unparseable frames.
                    let parsed: Option<ServeRequest> = if data.first() == Some(&SERVE_REQUEST) {
                        ServeRequest::decode(&data[1..]).ok()
                    } else {
                        None
                    };
                    let parsed_nonce = parsed.as_ref().map(|r| r.nonce);
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
                    // A served completion (or its fetch by nonce) is delivered once its reply reaches
                    // the consumer: only then can an unpaid one be held against it.
                    let delivers = match data.first() {
                        Some(&SERVE_REQUEST) => parsed_nonce,
                        Some(&FETCH_RESULT) if FetchResponse::encoded_is_ready(&response) => {
                            data.get(1..17).and_then(|n| n.try_into().ok())
                        }
                        _ => None,
                    };
                    // Best-effort reply; if the swarm is gone the next poll will error too.
                    let _ = match delivers {
                        Some(nonce) => {
                            let (p, peer) = (Arc::clone(&provider), source_peer.clone());
                            provider.net.respond_then(
                                request_id,
                                response,
                                Box::new(move |sent| {
                                    if sent {
                                        p.mark_delivered(nonce, &peer);
                                    }
                                }),
                            )
                        }
                        None => provider.net.respond(request_id, response),
                    };
                });
                if !accepted {
                    // A3: the bounded serve queue is full — shed with a retryable error so
                    // the consumer fails fast instead of waiting out its request timeout,
                    // and provider memory stays bounded under a burst.
                    // See docs/CODEBASE_HARDENING_PLAN.md (A3).
                    // An attestation challenge gets an attestation-framed refusal, so the consumer
                    // retries later instead of mistaking the serve-error frame for a legacy provider.
                    let busy = if is_attest {
                        attest_refusal("provider overloaded, retry")
                    } else if is_score {
                        score_refusal("provider overloaded, retry")
                    } else if is_stream_poll {
                        // A poll or close shed in its own reply shape, so the consumer backs off and
                        // retries rather than taking it for an older provider (review round 4 M5).
                        FetchChunksResponse::Overloaded.encode()
                    } else if is_fetch {
                        // A fetch too: "not ready yet, retry", so a finished result isn't abandoned.
                        FetchResponse::Generating.encode()
                    } else {
                        frame_response(&[ServeChunk::Error("provider overloaded, retry".into()).encode()])
                    };
                    let _ = self.net.respond(shed_id, busy);
                }
            }
            // FALLBACK hot-reload (the `watch_policy_file` thread is the primary, event-driven path):
            // if the watcher missed an event or couldn't start, this catches a policy change within one
            // poll slice (<1s). Cheap — one `stat` per poll, parse+swap only on an actual change; the
            // reload_lock + mtime dedup prevent a double *reload*; announce_lock (via the polled
            // variant below) prevents the watcher's and this loop's announce passes from interleaving.
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
                // Non-blocking: if the watcher thread is mid-announce, skip this tick (it's already
                // publishing the fresh policy) rather than stall the serve loop on its blocking `detect`.
                match self.announce_models_polled() {
                    Some(Ok(n)) => eprintln!("openhydra-agent: re-announced {n} model(s)"),
                    Some(Err(e)) => eprintln!("openhydra-agent: re-announce failed: {e}"),
                    None => {} // an announce is already in flight (watcher) — nothing to add
                }
                last_announce = std::time::Instant::now();
            }
        }
    }

    /// M5: handle a `REMOTE_SCOPE_SET` command — a control-capable swarm member, from another
    /// device, flips one of this rig's models between Private/Global (etc.). `body` is the
    /// json-encoded [`RemoteScopeSet`] (the method byte already stripped); returns the json-encoded
    /// [`RemoteScopeAck`]. Every check is fail-closed and the reply never leaks internal state.
    ///
    /// Order matters: replay window → owner-authoritative credential authz ([`CAP_CONTROL`] + bound
    /// to `source_peer`) → the member's own signature over the exact command → scope parse → apply.
    /// The credential check is the real trust gate (it verifies against a swarm THIS node owns and
    /// its live revocation set); the command signature adds non-repudiation + integrity independent
    /// of the transport. Applying rides the normal share-policy hot-reload (a later re-announce
    /// publishes a newly-Global model — announced *by this rig*, per the M5 design).
    fn handle_remote_scope_set(&self, source_peer: &str, body: &[u8]) -> Vec<u8> {
        use crate::share_policy::Scope;
        use openhydra_network::remote_control::{RemoteScopeAck, RemoteScopeSet};
        let refuse = |msg: String| {
            tracing::warn!(peer = %source_peer, %msg, "remote-scope-set refused (M5)");
            RemoteScopeAck::Refused(msg).encode()
        };
        // Only an owner rig (one with a swarm authorizer) can be remote-controlled.
        let Some(auth) = self.swarm_auth.as_deref() else {
            return refuse("this provider is not a swarm owner (no rig control)".into());
        };
        let cmd = match RemoteScopeSet::decode(body) {
            Ok(c) => c,
            Err(e) => return refuse(format!("undecodable command: {e}")),
        };
        let now = now_unix_ms();
        // 1. Replay/skew window — a captured command can't be re-applied later.
        if !cmd.within_replay_window(now) {
            return refuse("command timestamp is outside the accepted window".into());
        }
        // 2. Owner-authoritative authz: the credential must be for a swarm we own, verify (sig +
        //    not-revoked + unexpired), grant CAP_CONTROL, and bind to the connecting peer.
        let member_key = match auth.authorize_control(Some(&cmd.credential), source_peer, now) {
            Ok(a) => a.member_public_key,
            Err(e) => return refuse(format!("{e}")),
        };
        // 3. The member signed THIS command (non-repudiation + integrity beyond the transport).
        if let Err(e) = cmd.verify_command_sig() {
            return refuse(format!("bad command signature: {e}"));
        }
        // 4. Parse the requested scope (fail-closed on an unknown string).
        let scope = match cmd.scope.as_str() {
            "global" => Scope::Global,
            "private" => Scope::Private,
            "device" => Scope::Device,
            other => return refuse(format!("unknown scope '{other}'")),
        };
        // Read the FRESHEST policy for the gates below — a control credential grants rig-wide
        // authority over the owner's shared models (not per-swarm/per-model), so both the
        // shared-model check and the publish gate must see any change the owner just made on disk,
        // not a stale hot-reload snapshot (review LOW: closes the ~poll-interval TOCTOU where a
        // Global command could slip through just after the owner set `allow_remote_publish=false`).
        self.policy.reload_if_changed();
        // 5. Only re-scope a model this rig ALREADY shares — remote control adjusts the reach of a
        //    shared model, it never shares a new one (that would let a control device expose a model
        //    the owner never chose to share, and would leave a latent Global+consent record on it).
        if !self.model_shared(&cmd.model_id) {
            return refuse(format!("model '{}' is not shared by this rig", cmd.model_id));
        }
        // 5b. Public exposure gate (M5 review, HIGH/MED): making a model MORE private is always fine,
        //     but flipping it to Global (public/marketplace) via a remote command is allowed only
        //     when the rig owner has left `allow_remote_publish` on. The owner can switch it off to
        //     forbid any remote publish outright; the CAP_CONTROL grant itself is still the primary
        //     authorisation.
        if scope == Scope::Global && !self.policy.snapshot().allow_remote_publish {
            return refuse(
                "remote publish to Global is disabled on this rig (allow_remote_publish=false)".into(),
            );
        }
        // 5c. Monotonic replay guard: a command must be strictly newer than the last we applied for
        //     this member. Check-and-advance together so two concurrent commands can't both pass; a
        //     failed apply below still consumes the timestamp (the sender just re-issues a fresh one).
        {
            let mut last = self.remote_cmd_last_applied.lock().unwrap_or_else(|e| e.into_inner());
            if last.get(&member_key).is_some_and(|&prev| cmd.issued_at_ms <= prev) {
                return refuse("command is not newer than the last applied for this member".into());
            }
            last.insert(member_key.clone(), cmd.issued_at_ms);
        }
        // 6. Apply to the live, file-backed policy (a static policy can't be remote-set).
        let Some(path) = self.policy.watched_path() else {
            return refuse("provider share policy is not file-backed".into());
        };
        match SharePolicy::set_model_scope(&path, &cmd.model_id, scope, now) {
            Ok(_) => {
                tracing::info!(model = %cmd.model_id, scope = %cmd.scope, peer = %source_peer,
                    "remote-scope-set applied (M5)");
                RemoteScopeAck::Applied { model_id: cmd.model_id, scope: cmd.scope }.encode()
            }
            Err(e) => refuse(format!("could not apply scope: {e}")),
        }
    }

    /// Route one inbound request by its method byte: serve completions, settle receipts.
    /// `source_peer` is the libp2p-authenticated sender, used to rate-limit per peer.
    /// `parsed` is the already-decoded serve request (F-C5), `None` for non-serve frames.
    fn dispatch(&self, source_peer: &str, data: &[u8], parsed: Option<ServeRequest>) -> Vec<u8> {
        match data.first() {
            // M5: an owner-authorised remote-control device flips a model's share scope on this rig.
            Some(&REMOTE_SCOPE_SET) => self.handle_remote_scope_set(source_peer, &data[1..]),
            // Device attestation challenge (Phase 0). Rate-limited per authenticated peer before any
            // signing; publicly-announced models are attestable, and a private one by a swarm member
            // that recently passed its gate. Anyone else gets the same refusal as for an unshared
            // model, so a challenge can't probe for private models.
            Some(&ATTEST_CHALLENGE) => {
                if self.attest_rl.try_acquire(source_peer, now_unix_ms()).is_err() {
                    return attest_refusal(crate::attest::ATTEST_RATE_LIMITED);
                }
                // The provider-wide cap is spent only on a challenge about to be signed, so ones
                // refused anyway (no attestor, a model not attestable, malformed) can't drain it.
                crate::attest::handle_attest_inbound_limited(
                    &data[1..],
                    self.attestor.as_deref(),
                    &|model| self.policy.announce_globally(model) || self.private_attestable(source_peer, model),
                    now_unix_ms(),
                    // A consumer we served recently has a bucket of its own once the provider-wide one
                    // is spent (review round 3, B2): free identities flooding challenges can't make us
                    // refuse the consumers we serve.
                    &|| {
                        self.attest_global_rl.try_acquire(ATTEST_GLOBAL_KEY, now_unix_ms()).is_ok()
                            || (self.served_recently(source_peer)
                                && self.attest_served_rl.try_acquire(ATTEST_GLOBAL_KEY, now_unix_ms()).is_ok())
                    },
                )
            }
            // Prompt-scoring challenge (Track A2-Logprob): opt-in, per-peer rate-limited, and capped
            // provider-wide; same public-models-only rule as attestation.
            Some(&PROMPT_SCORE_CHALLENGE) => {
                if self.score_rl.try_acquire(source_peer, now_unix_ms()).is_err() {
                    return score_refusal("prompt scoring rate limited, retry");
                }
                let now = now_unix_ms();
                let in_round = self
                    .scored_recently
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(source_peer)
                    .is_some_and(|at| now.saturating_sub(*at) <= IN_ROUND_MS);
                // Challenges come from consumers we served (a gateway checks a provider right after
                // it served it) or one mid-round, one at a time each (review round 4 H3): strangers
                // can't keep the scoring slots, or a side-scorer, busy to make real checks fail.
                if !in_round && !self.served_recently(source_peer) && !self.is_own_consumer(source_peer) {
                    return score_refusal("prompt scoring is for consumers this provider served; retry after a request");
                }
                let Some(_one) = OneAtATime::enter(&self.scoring_now, source_peer) else {
                    return score_refusal("a challenge from you is already being scored, retry");
                };
                // A model counts toward the peer's recent ones as soon as its challenge is let through
                // (one that starts a side-scorer counts though it is told to retry, R6-6).
                if let Some(model) = crate::prompt_score::challenge_model(&data[1..]) {
                    let mut sm = self.score_models.lock().unwrap_or_else(|e| e.into_inner());
                    if !note_score_model(&mut sm, source_peer, model, now) {
                        return score_refusal("switched models too recently, retry");
                    }
                }
                // Verifiers take turns: one that has had its texts yields to one waiting.
                if self.score_turn.lock().unwrap_or_else(|e| e.into_inner()).must_yield(source_peer, now) {
                    return score_refusal("another verifier's turn, retry");
                }
                let reply = handle_prompt_score_inbound(
                    &data[1..],
                    self.prompt_scoring.then_some(&self.adapter as &dyn EngineAdapter),
                    &self.score_gate,
                    &|model| self.policy.announce_globally(model),
                    in_round,
                );
                {
                    let refused = crate::prompt_score::is_score_refusal(&reply);
                    let busy = refused && String::from_utf8_lossy(&reply).contains("retry");
                    self.score_turn.lock().unwrap_or_else(|e| e.into_inner()).note(source_peer, !refused, busy, now);
                }
                if !crate::prompt_score::is_score_refusal(&reply) {
                    let mut m = self.scored_recently.lock().unwrap_or_else(|e| e.into_inner());
                    m.retain(|_, at| now.saturating_sub(*at) <= IN_ROUND_MS);
                    if m.len() < MAX_RECENTLY_SERVED {
                        m.insert(source_peer.to_string(), now);
                    }
                }
                reply
            }
            // Which exact model a handle is (so the verifier can pick the reference for its quant).
            Some(&MODEL_INFO_REQUEST) => {
                if self.score_rl.try_acquire(source_peer, now_unix_ms()).is_err() {
                    return score_refusal("prompt scoring rate limited, retry");
                }
                handle_model_info_inbound(
                    &data[1..],
                    self.prompt_scoring,
                    &|model| self.policy.announce_globally(model),
                    &|model| self.model_info(model),
                )
            }
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
                // B-S1: only co-sign a receipt that settles a real serve we recorded. Review F2: also
                // reject a non-monotonic / under-signed crypto-rail cumulative BEFORE consuming the
                // commitment (so a bad cumulative doesn't burn the single-use nonce).
                // The claim that came with the receipt, kept once it is co-signed.
                let accepted_claim = std::cell::RefCell::new(None);
                let bind = |p: &openhydra_protocol::receipts::ReceiptPayload, claim: Option<&crate::receipt::ClaimSig>| {
                    // From its own consumer first: nothing (not even a BEHIND reply) for a replay.
                    self.check_sender(p, source_peer).map_err(BindRejection::from)?;
                    // The claim first: on Base its signer picks the running total to check against.
                    let kept = self.check_settlement_claim(p, claim).map_err(BindRejection::from)?;
                    let floor = self.validate_settlement_cumulative(p, kept.as_ref().and_then(|k| k.signer))?;
                    self.check_and_consume_commitment(p, source_peer, floor).map_err(BindRejection::from)?;
                    *accepted_claim.borrow_mut() = kept;
                    Ok(())
                };
                // Validate, co-sign and record as one step (see `receipt_lock`).
                let _receipt_guard = self.receipt_lock.lock().unwrap_or_else(|e| e.into_inner());
                let replay_key = recent_cosign_key(source_peer, data);
                if let Some(reply) = self.recent_cosign(&replay_key) {
                    return reply;
                }
                let (response, accepted) =
                    handle_receipt_inbound(data, &sign, &provider_pub, &bind);
                if accepted.is_some() {
                    self.note_cosign(replay_key, &response);
                }
                // Persist the accepted co-signed receipt (best-effort; never fails the reply).
                ledger_receipt(self.store.as_ref(), accepted.as_ref());
                // M2.3: accrue the consumer's give/take (it consumed `tokens` from us).
                if let Some(receipt) = &accepted {
                    self.record_consumption(receipt);
                    // Review F2: advance our co-signed cumulative high-water mark for this pair.
                    self.record_settlement_cumulative(receipt, accepted_claim.borrow().as_ref());
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
                        // M4-base reach gate (buffered twin of the streaming path): a `Private` model
                        // needs a valid swarm credential bound to `source_peer`; `Device` is loopback-only.
                        if let Some(refusal) = self.scope_refusal(&req, source_peer) {
                            return refusal;
                        }
                        if let Some(refusal) = self.nonce_refusal(&req, source_peer) {
                            return refusal;
                        }
                        let reservation = match self.payment_gate(&req, source_peer) {
                            Ok(r) => r,
                            Err(refusal) => {
                                self.release_nonce(req.nonce, source_peer);
                                return refusal;
                            }
                        };
                        // Capture the small fields we still need after `req` is moved into
                        // the serve (which consumes `messages`), for commitment + stats +
                        // reconnect-and-fetch buffering.
                        let nonce = req.nonce;
                        let model_ref = req.model_ref.clone();
                        let reply_to = req.reply_to.clone();
                        let max_tokens = req.max_tokens;
                        let wallet = req.settlement_wallet.as_deref().and_then(parse_evm_address);
                        self.hold_nonce(nonce, &source_peer);
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
                            &mut self.serve_budget(reservation.as_ref()),
                        );
                        // B-S1: record what we served under the consumer's committed nonce so
                        // its settlement receipt can be validated (right tokens/model, once). Not
                        // delivered yet: it is once the reply (or a fetch of it by nonce) is sent
                        // to the consumer (`Provider::mark_delivered`). Recorded before the result is
                        // fetchable, so a fetch always finds it.
                        // A serve that failed part-way (an engine error, its time limit) still
                        // delivers the text it produced: that is owed, counted as the consumer
                        // counts it (round 8 S8-2).
                        let owed = if summary.ok { summary.tokens } else { failed_serve_billable(&chunks, max_tokens) };
                        if owed > 0 {
                            self.record_commitment(nonce, model_ref.clone(), owed, &source_peer, false, wallet);
                        }
                        // Reconnect-and-fetch: buffer the produced frames under the nonce so a
                        // consumer whose connection dropped can fetch them on a fresh circuit.
                        self.store_result(nonce, &reply_to, &chunks);
                        self.end_hold(nonce, &source_peer);
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
            Some(&STREAM_CLOSE) => self.handle_stream_close(source_peer, &data[1..]),
            // Unknown method byte → a framed Error from the serve handler.
            _ => handle_serve_inbound(data, &self.adapter),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_settlement_floor_is_what_we_served_not_what_the_consumer_declares() {
        use openhydra_protocol::weight::receipt_weighted_units;
        // A consumer declaring 0 tokens (or any split) still owes the floor for the 100 served.
        assert_eq!(settlement_floor_units(Some(100), 0), receipt_weighted_units(100, None));
        assert_eq!(settlement_floor_units(Some(100), 100), receipt_weighted_units(100, None));
        assert!(settlement_floor_units(Some(100), 0) > 0);
        let peg = 1_000;
        assert_eq!(
            openhydra_protocol::payout::check_cumulative(500, Some(500), settlement_floor_units(Some(100), 0), Some(peg)),
            Err(openhydra_protocol::payout::CumulativeError::BelowFloor),
            "a zero step for a real serve is refused"
        );
    }

    use super::*;
    use crate::adapter::{InferenceRequest, ServeOutcome};
    use crate::serve::{parse_response, ServeChunk, ServeRequest};

    // ── M6/F2: withdrawal re-publish planning ────────────────────────────
    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn m6f2_new_withdrawal_publishes_and_starts_countdown() {
        // Last pass announced {a,b}; this pass only {a} is intended → b is withdrawn now and gets a
        // 3-pass countdown (2 remaining after this publish).
        let mut pending = BTreeMap::new();
        let pub1 = plan_withdrawals(&set(&["a", "b"]), &set(&["a"]), &mut pending, 3);
        assert_eq!(pub1, vec!["b".to_string()]);
        assert_eq!(pending.get("b"), Some(&2));
    }

    #[test]
    fn m6f2_republishes_exactly_n_passes_then_stops() {
        let mut pending = BTreeMap::new();
        let last = set(&["a", "b"]);
        let desired = set(&["a"]); // b stays un-shared every pass
        // Pass 1 (the drop) + 2 re-publishes = 3 total, then it stops.
        let p1 = plan_withdrawals(&last, &desired, &mut pending, 3);
        let p2 = plan_withdrawals(&desired, &desired, &mut pending, 3); // last==desired now (no new drop)
        let p3 = plan_withdrawals(&desired, &desired, &mut pending, 3);
        let p4 = plan_withdrawals(&desired, &desired, &mut pending, 3);
        assert_eq!(p1, vec!["b".to_string()]);
        assert_eq!(p2, vec!["b".to_string()]);
        assert_eq!(p3, vec!["b".to_string()]);
        assert!(p4.is_empty(), "must stop after {WITHDRAW_REPUBLISH_PASSES} passes");
        assert!(pending.is_empty());
    }

    #[test]
    fn m6f2_reshare_cancels_pending_withdrawal() {
        // b is un-shared (countdown running), then re-shared before it expires → it must NOT be
        // withdrawn again, and the countdown is cleared.
        let mut pending = BTreeMap::new();
        let p1 = plan_withdrawals(&set(&["a", "b"]), &set(&["a"]), &mut pending, 3);
        assert_eq!(p1, vec!["b".to_string()]);
        // Next pass b is intended again (desired contains b); last was {a} (b not announced last pass).
        let p2 = plan_withdrawals(&set(&["a"]), &set(&["a", "b"]), &mut pending, 3);
        assert!(p2.is_empty(), "a re-shared model is never withdrawn");
        assert!(pending.is_empty());
    }

    #[test]
    fn m6f2_no_withdrawal_when_nothing_dropped() {
        let mut pending = BTreeMap::new();
        // Steady state and a brand-new share both withdraw nothing.
        assert!(plan_withdrawals(&set(&["a"]), &set(&["a"]), &mut pending, 3).is_empty());
        assert!(plan_withdrawals(&set(&[]), &set(&["a", "b"]), &mut pending, 3).is_empty());
        assert!(pending.is_empty());
    }

    #[test]
    fn m6f2_republish_passes_1_publishes_once() {
        // Guard the saturating_sub edge: passes=1 → published exactly once, no underflow.
        let mut pending = BTreeMap::new();
        let p1 = plan_withdrawals(&set(&["a", "b"]), &set(&["a"]), &mut pending, 1);
        assert_eq!(p1, vec!["b".to_string()]);
        assert!(pending.is_empty());
        assert!(plan_withdrawals(&set(&["a"]), &set(&["a"]), &mut pending, 1).is_empty());
    }

    // ── Finding 2: withdrawal hysteresis (a transient detection blip must not tombstone) ──

    /// Run one announce pass's withdrawal pipeline exactly as `announce_models_inner`: grace-adjust
    /// `desired` (vs the raw `detected` set) into `effective`, run `plan_withdrawals`, advance the
    /// baseline. Returns the models tombstoned this pass.
    fn withdraw_pass(
        last: &mut BTreeSet<String>,
        absence: &mut BTreeMap<String, u8>,
        pending: &mut BTreeMap<String, u8>,
        desired: &[&str],
        detected: &[&str],
        grace: u8,
    ) -> Vec<String> {
        let effective = apply_absence_grace(last, &set(desired), &set(detected), absence, grace);
        let withdrawn = plan_withdrawals(last, &effective, pending, WITHDRAW_REPUBLISH_PASSES);
        *last = effective;
        withdrawn
    }

    #[test]
    fn f2_transient_single_pass_vanish_does_not_tombstone() {
        // Pass 1: {a,b} announced & detected. Pass 2: b's engine blips — b vanishes from detection
        // (auto-mode partial Ok) and is not re-detected. With grace=2 it must NOT be tombstoned.
        // Pass 3: b is back → still never withdrawn, and no lingering countdown/streak.
        let (mut last, mut abs, mut pend) = (BTreeSet::new(), BTreeMap::new(), BTreeMap::new());
        assert!(withdraw_pass(&mut last, &mut abs, &mut pend, &["a", "b"], &["a", "b"], 2).is_empty());
        let w2 = withdraw_pass(&mut last, &mut abs, &mut pend, &["a"], &["a"], 2); // b vanished (blip)
        assert!(w2.is_empty(), "a one-pass detection blip must not tombstone");
        assert!(last.contains("b"), "b held in the baseline through the grace window");
        let w3 = withdraw_pass(&mut last, &mut abs, &mut pend, &["a", "b"], &["a", "b"], 2); // b back
        assert!(w3.is_empty(), "b returned within grace → never withdrawn");
        assert!(pend.is_empty() && abs.is_empty(), "no lingering countdown/streak");
    }

    #[test]
    fn f2_policy_withdrawal_of_a_still_detected_model_evicts_immediately() {
        // b is still DETECTED but no longer `desired` (operator un-shared it / flipped to Private).
        // Intentional withdrawal → tombstoned at once, no grace, no streak tracked.
        let (mut last, mut abs, mut pend) = (set(&["a", "b"]), BTreeMap::new(), BTreeMap::new());
        let w = withdraw_pass(&mut last, &mut abs, &mut pend, &["a"], &["a", "b"], 2);
        assert_eq!(w, vec!["b".to_string()], "policy drop of a still-detected model evicts now");
        assert!(!last.contains("b"));
        assert!(abs.is_empty(), "no streak for an intentional withdrawal");
    }

    #[test]
    fn f2_genuine_vanish_tombstones_after_the_grace_window() {
        // b's engine goes away for good: not withdrawn on the first missing pass, withdrawn on the
        // second consecutive miss (grace=2).
        let (mut last, mut abs, mut pend) = (set(&["a", "b"]), BTreeMap::new(), BTreeMap::new());
        let w1 = withdraw_pass(&mut last, &mut abs, &mut pend, &["a"], &["a"], 2); // miss 1
        assert!(w1.is_empty(), "grace holds b on the first missing pass");
        let w2 = withdraw_pass(&mut last, &mut abs, &mut pend, &["a"], &["a"], 2); // miss 2 → withdraw
        assert_eq!(w2, vec!["b".to_string()], "withdrawn after the grace window");
        assert!(!last.contains("b"));
    }

    #[test]
    fn f2_grace_of_one_is_immediate_withdrawal_the_pre_hysteresis_behaviour() {
        // grace<=1 disables hysteresis: a vanished model is withdrawn on the first missing pass.
        let (mut last, mut abs, mut pend) = (set(&["a", "b"]), BTreeMap::new(), BTreeMap::new());
        let w = withdraw_pass(&mut last, &mut abs, &mut pend, &["a"], &["a"], 1);
        assert_eq!(w, vec!["b".to_string()]);
    }

    fn detected(canonical: &str) -> DetectedModel {
        DetectedModel {
            engine_ref: "qwen2.5:7b".into(),
            canonical_id: canonical.into(),
            family: "qwen2".into(),
            params: "7.6b".into(),
            quant: "q4_k_m".into(),
            size_bytes: 42,
            weight_digest: String::new(),
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

    // ── M4-base: the pure reach-gate decision ──
    mod scope_gate_tests {
        use super::super::scope_gate;
        use crate::share_policy::Scope;
        use crate::swarms::{approve_member, build_enrollment_request, create_swarm, SwarmAuthorizer};
        use openhydra_network::identity::Identity;
        use openhydra_network::membership::credential_member_peer_id;

        const OWN: &str = "12D3KooWOwnNode";
        const HOUR: u64 = 3600;

        #[test]
        fn global_is_open_to_anyone_no_credential_needed() {
            assert!(scope_gate(Scope::Global, "m", "12D3KooWStranger", OWN, None, None, 0).is_ok());
        }

        #[test]
        fn device_is_loopback_only() {
            // Our own node (self-serve) passes; any other peer is refused as device-only.
            assert!(scope_gate(Scope::Device, "m", OWN, OWN, None, None, 0).is_ok());
            let err = scope_gate(Scope::Device, "m", "12D3KooWOther", OWN, None, None, 0).unwrap_err();
            assert!(err.0.contains("device-only") && !err.1);
        }

        #[test]
        fn private_self_serve_needs_no_credential() {
            assert!(scope_gate(Scope::Private, "m", OWN, OWN, None, None, 0).is_ok());
        }

        #[test]
        fn private_without_an_authorizer_is_refused_fail_closed() {
            let err = scope_gate(Scope::Private, "m", "12D3KooWMember", OWN, None, None, 0).unwrap_err();
            assert!(err.0.contains("no swarm configured") && !err.1, "device-only-style, not a swarm-counter refusal");
        }

        #[test]
        fn private_admits_a_valid_member_and_refuses_a_stranger() {
            let owner_dir = tempfile::tempdir().unwrap();
            let member = Identity::load_or_create(&tempfile::tempdir().unwrap().path().join("id")).unwrap();
            let swarm = create_swarm(owner_dir.path(), "rig", 1_000).unwrap();
            let req = build_enrollment_request(&member, &swarm.swarm_public_key, "m", 2_000).unwrap();
            let approved = approve_member(owner_dir.path(), &swarm.swarm_public_key, &req.magnet, "m", 24 * HOUR, 3_000).unwrap();
            let member_peer = credential_member_peer_id(&approved.credential).unwrap().to_string();
            let auth = SwarmAuthorizer::new(owner_dir.path().to_path_buf());

            // Valid credential bound to the connecting member → proceed.
            assert!(scope_gate(Scope::Private, "m", &member_peer, OWN, Some(&approved.credential), Some(&auth), 4_000).is_ok());
            // A stranger with no credential → refused, and it counts as a swarm refusal.
            let err = scope_gate(Scope::Private, "m", "12D3KooWStranger", OWN, None, Some(&auth), 4_000).unwrap_err();
            assert!(err.1, "a credential-gate refusal bumps the swarm counter");
            // `Unknown` scope is gated exactly like `Private` (fail-closed): the same stranger is refused.
            assert!(scope_gate(Scope::Unknown, "m", "12D3KooWStranger", OWN, None, Some(&auth), 4_000).is_err());
        }
    }

    #[test]
    fn a_paid_serve_grows_its_reservation_until_the_window_is_full() {
        let dir = tempfile::tempdir().unwrap();
        let net = openhydra_network::handle::NetworkHandle::start(openhydra_network::node::NodeConfig {
            identity_path: dir.path().join("id.key"),
            listen_addrs: vec!["/ip4/127.0.0.1/tcp/0".into()],
            bootstrap_peers: vec![],
            enable_peer_relay: false,
            enable_connection_reversal: false,
            pcp_gateway: None,
        })
        .unwrap();
        let provider = Provider::new(StubAdapter, net);
        // The consumer owes 56,000 tokens on unsettled serves.
        provider.serve_commitments.lock().unwrap().insert([1; 16], ServeCommitment {
            model_id: "m".into(), tokens: 56_000, recorded_ms: now_unix_ms(), served_peer: "c".into(), delivered: true, wallet: None,
        });
        // The serve itself is a stream whose consumer has read 3,000 tokens of it so far: its
        // reservation already counts them, so its commitment isn't counted again.
        provider.serve_commitments.lock().unwrap().insert([9; 16], ServeCommitment {
            model_id: "m".into(), tokens: 3_000, recorded_ms: now_unix_ms(), served_peer: "c".into(), delivered: true, wallet: None,
        });
        let running = |tokens| Running { id: 7, tokens, nonce: [9; 16] };
        provider.paid_generating.lock().unwrap().insert("c".into(), vec![running(4_096)]);
        let r = PaymentReservation { generating: Arc::clone(&provider.paid_generating), peer: "c".into(), id: 7, initial: 4_096 };
        let mut budget = provider.serve_budget(Some(&r));
        // A stream asks after every token: it grows a step at a time while 56,000 + the
        // reservation fits in 65,536, so it runs to 9,216 and stops there.
        let stopped_at = (1..u64::MAX).find(|sent| !budget(*sent)).unwrap();
        assert_eq!(stopped_at, 9_216);
        assert_eq!(provider.paid_generating.lock().unwrap()["c"], vec![running(9_216)]);
        // A free serve has no budget to run out of.
        assert!(provider.serve_budget(None)(u64::MAX));
        drop(budget);
        drop(r);
        assert!(provider.paid_generating.lock().unwrap().get("c").is_none(), "released when the serve ends");
    }

    /// Review round 4 M4: a finished stream counts against the window for all it generated, read or
    /// not (its reservation ended with it); a running one only through its reservation.
    #[test]
    fn a_finished_streams_unread_output_counts_against_the_window() {
        let mut map = HashMap::new();
        map.insert([1; 16], ServeCommitment {
            model_id: "m".into(), tokens: 100, recorded_ms: 5, served_peer: "c".into(), delivered: true, wallet: None,
        });
        let generated: HashMap<[u8; 16], u64> = [([1; 16], 900), ([2; 16], 700), ([3; 16], 50)].into();
        let running = [Running { id: 1, tokens: 64, nonce: [3; 16] }];
        let mut owed: Vec<u64> = unsettled_of(&map, "c", &running, &generated).iter().map(|u| u.tokens).collect();
        owed.sort();
        assert_eq!(owed, vec![700, 900], "read 100 of 900: owes for 900; never read: 700; running: by its reservation");
        assert!(unsettled_of(&map, "other", &[], &HashMap::new()).is_empty());
    }

    #[test]
    fn each_base_wallet_has_its_own_running_total_and_signs_only_its_own_serves() {
        use openhydra_protocol::settlement::{evm_address, sign_evm_claim, ChainDomain, Eip712Domain};
        let dir = tempfile::tempdir().unwrap();
        let net = openhydra_network::handle::NetworkHandle::start(openhydra_network::node::NodeConfig {
            identity_path: dir.path().join("id.key"),
            listen_addrs: vec!["/ip4/127.0.0.1/tcp/0".into()],
            bootstrap_peers: vec![],
            enable_peer_relay: false,
            enable_connection_reversal: false,
            pcp_gateway: None,
        })
        .unwrap();
        let domain = Eip712Domain { chain_id: 8453, verifying_contract: [7; 20] };
        let provider = Provider::new(StubAdapter, net).with_settlement_chain(Some(ChainDomain::Base(domain.clone())));
        let w1 = k256::ecdsa::SigningKey::from_slice(&[1; 32]).unwrap();
        let w2 = k256::ecdsa::SigningKey::from_slice(&[2; 32]).unwrap();
        let (a1, a2) = (evm_address(w1.verifying_key()), evm_address(w2.verifying_key()));
        let consumer = ed25519_dalek::SigningKey::from_bytes(&[5; 32]);
        let payload = openhydra_protocol::receipts::payload_from_bytes(
            &[9; 32], consumer.verifying_key().as_bytes(), "m", 10, &[3; 16], 1, None, "",
            Some(500), Some(openhydra_protocol::payout::SettlementAsset::Usdc), Some(1),
        )
        .unwrap();
        let claim_by = |w: &k256::ecdsa::SigningKey| {
            let expires_at = now_unix_ms() / 1000 + openhydra_protocol::settlement::CLAIM_LIFETIME_SECS;
            let c = crate::receipt::claim_for(&payload, expires_at).unwrap();
            crate::receipt::ClaimSig { expires_at, sig: sign_evm_claim(w, &domain, &c).to_vec() }
        };
        // The serve was requested with wallet 1.
        provider.serve_commitments.lock().unwrap().insert([3; 16], ServeCommitment {
            model_id: "m".into(), tokens: 10, recorded_ms: now_unix_ms(), served_peer: "c".into(), delivered: true, wallet: Some(a1),
        });
        let kept = provider.check_settlement_claim(&payload, Some(&claim_by(&w1))).unwrap().unwrap();
        assert_eq!(kept.signer, Some(a1));
        // A claim from another wallet is refused: it isn't the one the serve named.
        assert!(provider.check_settlement_claim(&payload, Some(&claim_by(&w2))).unwrap_err().contains("not 0x"));
        // Each wallet's running total is its own (review round 3, S1): as on-chain.
        assert_ne!(provider.settled_key(&payload, Some(a1)), provider.settled_key(&payload, Some(a2)));
        assert!(provider.settled_key(&payload, Some(a1)).ends_with(&format!("@0x{}", hex::encode(a1))));
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
            think: None,
            nonce: [0u8; 16],
            credential: None, deterministic: None, settlement_peg: None, settlement_domain: None, settlement_wallet: None,
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
    fn v5_denomination_gate_rejects_only_a_real_mismatch() {
        use openhydra_protocol::payout::SettlementAsset;
        let usdc = Some(SettlementAsset::Usdc);
        let our = Some(1_200u64);

        // Matching asset + peg → accepted.
        assert!(check_settlement_denomination(Some(9_600), usdc, Some(1_200), our).is_ok());
        // A provider that charges refuses a receipt that pays nothing.
        assert!(check_settlement_denomination(None, None, None, our).is_err());
        // A non-crypto receipt on a peg-less provider is fine (nothing to settle).
        assert!(check_settlement_denomination(None, None, None, None).is_ok());

        // Review A: a crypto voucher (cumulative present) on a provider with NO peg → fail-CLOSED,
        // rather than silently co-signing an unvalidated denomination on misconfiguration.
        assert!(check_settlement_denomination(Some(9_600), usdc, Some(9_999), None).is_err());

        // A peg the provider never agreed to → rejected (the core v5 payoff).
        assert!(check_settlement_denomination(Some(9_600), usdc, Some(2_400), our).is_err());
        // A crypto receipt missing the peg or asset entirely → rejected.
        assert!(check_settlement_denomination(Some(9_600), usdc, None, our).is_err());
        assert!(check_settlement_denomination(Some(9_600), None, Some(1_200), our).is_err());
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
            completion_tokens: None,
            canonical_model_id: String::new(),
            cumulative_amount: None,
            settlement_asset: None,
            peg_micros_per_credit: None,
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
        map.insert(n, ServeCommitment { model_id: model.into(), tokens, recorded_ms: now, served_peer: "c".into(), delivered: true, wallet: None });
    }

    #[test]
    fn commitment_accepts_match_once_then_rejects_replay() {
        let now = 1_000_000u64;
        let mut map = HashMap::new();
        commit(&mut map, [7u8; 16], "qwen2.5:7b", 100, now);
        assert!(validate_and_consume_commitment(&mut map, &[7u8; 16], "qwen2.5:7b", 100, now, now, "c", "c").is_ok());
        // Single-use: the commitment is consumed, so a replay of the same receipt is rejected.
        assert!(validate_and_consume_commitment(&mut map, &[7u8; 16], "qwen2.5:7b", 100, now, now, "c", "c")
            .unwrap_err()
            .contains("no serve commitment"));
    }

    #[test]
    fn commitment_allows_underclaim_rejects_overclaim() {
        let now = 1_000_000u64;
        let mut map = HashMap::new();
        commit(&mut map, [8u8; 16], "m", 100, now);
        // Claiming fewer tokens than served is harmless (only reduces the provider's credit).
        assert!(validate_and_consume_commitment(&mut map, &[8u8; 16], "m", 40, now, now, "c", "c").is_ok());
        // The B-S1 attack — claiming more than served — is rejected.
        commit(&mut map, [9u8; 16], "m", 100, now);
        assert!(validate_and_consume_commitment(&mut map, &[9u8; 16], "m", u64::MAX, now, now, "c", "c")
            .unwrap_err()
            .contains("exceed"));
    }

    #[test]
    fn commitment_rejects_unknown_nonce_and_model_mismatch() {
        let now = 1_000_000u64;
        let mut map = HashMap::new();
        assert!(validate_and_consume_commitment(&mut map, &[1u8; 16], "m", 1, now, now, "c", "c")
            .unwrap_err()
            .contains("no serve commitment"));
        // Right nonce, wrong model → reject (no cross-model misattribution).
        commit(&mut map, [2u8; 16], "cheap-model", 100, now);
        assert!(validate_and_consume_commitment(&mut map, &[2u8; 16], "expensive-model", 100, now, now, "c", "c")
            .unwrap_err()
            .contains("model"));
    }

    #[test]
    fn a_swarm_member_may_attest_the_private_model_it_was_served() {
        let now = 10 * PRIVATE_ATTEST_WINDOW_MS;
        let mut m = HashMap::new();
        assert!(private_member_in(&m, "member", "rig-model", now).is_none(), "not before passing the gate");
        note_private_member_in(&mut m, "member", "rig-model", now, 7u8);
        assert_eq!(private_member_in(&m, "member", "rig-model", now + PRIVATE_ATTEST_WINDOW_MS), Some(&7));
        assert!(private_member_in(&m, "member", "rig-model", now + PRIVATE_ATTEST_WINDOW_MS + 1).is_none(), "expires");
        assert!(private_member_in(&m, "member", "other-model", now).is_none(), "only the model it passed for");
        assert!(private_member_in(&m, "stranger", "rig-model", now).is_none(), "only that peer");
        // Bounded: a full map forgets expired pairs, then the oldest…
        for i in 0..MAX_PRIVATE_ATTEST_PAIRS as u64 {
            note_private_member_in(&mut m, &format!("p{i}"), "m", now + i, 0);
        }
        assert_eq!(m.len(), MAX_PRIVATE_ATTEST_PAIRS);
        // …but refreshing a pair already there evicts nobody (review A3).
        let before: std::collections::HashSet<_> = m.keys().cloned().collect();
        note_private_member_in(&mut m, "p5", "m", now + 10 * MAX_PRIVATE_ATTEST_PAIRS as u64, 0);
        assert_eq!(m.keys().cloned().collect::<std::collections::HashSet<_>>(), before);
    }

    /// Round 8 L8-4: a peer switches models at most once per window, however it alternates.
    #[test]
    fn a_peer_switches_models_at_most_once_a_window() {
        let mut m = HashMap::new();
        assert!(note_score_model(&mut m, "p", "x".into(), 0));
        assert!(note_score_model(&mut m, "p", "x".into(), 10), "the same model, again and again");
        assert!(!note_score_model(&mut m, "p", "y".into(), 20), "a switch too soon");
        assert!(note_score_model(&mut m, "p", "y".into(), SCORE_SWITCH_MIN_MS), "a switch after the window");
        assert!(!note_score_model(&mut m, "p", "x".into(), SCORE_SWITCH_MIN_MS + 1), "and back: too soon");
        assert!(note_score_model(&mut m, "q", "x".into(), 20), "per peer");
    }

    /// Round 8 O8-4 / round 9 B9-7: a provider charging on Solana won't take payment for a handle
    /// too long to settle there; off Solana the length doesn't matter.
    #[test]
    fn a_handle_too_long_to_settle_is_not_charged_for() {
        use openhydra_protocol::settlement::{ChainDomain, SolanaDomain};
        let long = "m".repeat(openhydra_protocol::settlement::MAX_CLAIM_MODEL_ID_LEN + 1);
        let req = |model: String, domain: Option<String>| ServeRequest {
            reply_to: "c".into(), model_ref: model, messages: vec![],
            max_tokens: Some(16), temperature: None, tools: Vec::new(), think: None, nonce: [5u8; 16],
            credential: None, deterministic: None, settlement_peg: Some(10), settlement_domain: domain, settlement_wallet: None,
        };
        let (_dir, provider) = test_provider(StubAdapter);
        let chain = ChainDomain::Solana(SolanaDomain { program_id: [1; 32], cluster: [2; 32] });
        let fp = chain.fingerprint();
        let provider = provider.with_settlement_peg(10).with_settlement_chain(Some(chain));
        let refused = provider.payment_gate(&req(long.clone(), Some(fp.clone())), "c").err().expect("refused");
        assert!(matches!(&parse_response(&refused).unwrap()[0], ServeChunk::Error(e) if e.contains("longer than")));
        assert!(provider.payment_gate(&req("qwen2.5:7b".into(), Some(fp)), "c").is_ok());
        let (_dir2, off_chain) = test_provider(StubAdapter);
        assert!(off_chain.with_settlement_peg(10).payment_gate(&req(long, None), "c").is_ok(), "off Solana: any length");
    }

    /// Round 9 P9-2: a verifier that has had its texts yields to one waiting.
    #[test]
    fn verifiers_take_turns_on_the_scorer() {
        let mut t = ScoreTurn::default();
        for i in 0..SCORE_TURN_TEXTS {
            assert!(!t.must_yield("q", i as u64));
            t.note("q", true, false, i as u64);
        }
        assert!(!t.must_yield("q", 10), "nobody waiting: q goes on");
        t.note("p", false, true, 10);
        assert!(t.must_yield("q", 11), "p waits: q yields");
        assert!(!t.must_yield("p", 11));
        t.note("p", true, false, 12);
        assert!(!t.must_yield("q", 13), "p got its turn; nobody waiting now");
        t.note("q", false, true, 14);
        assert!(!t.must_yield("p", 15), "p hasn't had its texts yet");
        t.note("x", false, true, 16 + SCORE_TURN_WAIT_MS);
        assert_eq!(t.waiting.as_ref().map(|(w, _)| w.as_str()), Some("x"), "a stale claim gives way");
    }

    /// Round 8 G8-1: polls, closes, fetches and receipts go to the control pool; serves don't.
    #[test]
    fn quick_messages_have_their_own_workers() {
        for m in [FETCH_CHUNKS, STREAM_CLOSE, FETCH_RESULT, RECEIPT_REQUEST] {
            assert!(is_control_message(Some(m)));
        }
        for m in [SERVE_REQUEST, SERVE_STREAM, ATTEST_CHALLENGE, PROMPT_SCORE_CHALLENGE] {
            assert!(!is_control_message(Some(m)));
        }
        assert!(!is_control_message(None));
    }

    /// Round 8 N8-1: a private model is gated at its own scope (not the strictest one this node
    /// shares), and a request that passes its gate is remembered for attestation.
    #[test]
    fn a_private_model_is_gated_at_its_own_scope() {
        use crate::share_policy::Scope;
        let (_dir, mut provider) = test_provider(StubAdapter);
        let mut policy = SharePolicy::share_all();
        policy.scopes.insert("rig".into(), Scope::Private);
        policy.scopes.insert("dev".into(), Scope::Device);
        provider.policy = PolicyWatcher::r#static(policy);
        let detected = |r: &str| DetectedModel {
            engine_ref: r.into(), canonical_id: String::new(), family: String::new(), params: String::new(),
            quant: String::new(), size_bytes: 0, weight_digest: String::new(),
        };
        provider.detected.lock().unwrap().extend([("rig".to_string(), detected("rig")), ("dev".to_string(), detected("dev"))]);
        let req = |model: &str| ServeRequest {
            reply_to: "c".into(), model_ref: model.into(), messages: vec![],
            max_tokens: None, temperature: None, tools: Vec::new(), think: None, nonce: [0u8; 16],
            credential: None, deterministic: None, settlement_peg: None, settlement_domain: None, settlement_wallet: None,
        };
        let refusal = |model: &str, from: &str| {
            provider.scope_refusal(&req(model), from).map(|r| match &parse_response(&r).unwrap()[0] {
                ServeChunk::Error(e) => e.clone(),
                other => panic!("{other:?}"),
            })
        };
        // A stranger is refused "rig" as private (needs a credential), not as device-only.
        let why = refusal("rig", "stranger").expect("refused");
        assert!(why.contains("private") && !why.contains("device-only"), "{why}");
        // An unknown alias is still gated at the strictest scope.
        assert!(refusal("alias", "stranger").expect("refused").contains("device-only"));
        // Our own request passes, and is remembered as a member of the private model.
        let own = provider.net.libp2p_peer_id().to_string();
        assert!(refusal("rig", &own).is_none());
        assert!(provider.private_members.lock().unwrap().keys().any(|(p, m)| *p == own && m == "rig"));
    }

    #[test]
    fn the_credit_window_bounds_what_a_consumer_takes_unpaid() {
        let now = 10 * DEFAULT_WINDOW_MS;
        let got = |at: u64, tokens: u64| Unsettled { at, tokens, delivered: true };
        assert!(payment_window(&[], &[], 100, 0, now).is_ok());
        // A serve just finished, its receipt on the way: fine.
        assert!(payment_window(&[got(now - 5_000, 500)], &[], 100, 0, now).is_ok());
        // One unpaid past the grace period: no more serves.
        assert!(payment_window(&[got(now - SETTLE_GRACE_MS - 1, 500)], &[], 100, 0, now).is_err());
        // Many parallel serves, none paid yet: capped by tokens.
        let burst: Vec<Unsettled> = (0..40).map(|_| got(now - 1_000, 2_000)).collect();
        assert!(payment_window(&burst, &[], 100, 0, now).is_err());
        // Review S2: a stream read but abandoned before its end still counts toward the caps (only
        // not as unpaid past the grace period: the consumer may never have got it).
        let read: Vec<Unsettled> =
            (0..MAX_UNSETTLED_SERVES as u64).map(|_| Unsettled { at: now - 1_000, tokens: 10, delivered: false }).collect();
        assert!(payment_window(&read, &[], 100, 0, now).is_err());
        let late = Unsettled { at: now - SETTLE_GRACE_MS - 1, tokens: 500, delivered: false };
        assert!(payment_window(&[late], &[], 100, 0, now).is_ok());
        assert!(payment_window(&[Unsettled { tokens: 65_000, ..late }], &[], 1_000, 0, now).is_err());
        // Repeated defaults.
        assert!(payment_window(&[], &[], 100, MAX_DEFAULTS - 1, now).is_ok());
        assert!(payment_window(&[], &[], 100, MAX_DEFAULTS, now).is_err());
        // Paid serves still running count: too many at once, or reserving too many tokens.
        assert!(payment_window(&[], &[100; MAX_UNPAID_CONCURRENT - 1], 100, 0, now).is_ok());
        assert!(payment_window(&[], &[100; MAX_UNPAID_CONCURRENT], 100, 0, now).is_err());
        assert!(payment_window(&[got(now, 60_000)], &[8_000], 100, 0, now).is_err());
        // The request's own reservation counts too, so one huge request can't leave more unpaid than
        // the window, and a request that fits beside nothing is told so.
        assert!(payment_window(&[], &[], MAX_UNSETTLED_TOKENS, 0, now).is_ok());
        assert!(payment_window(&[], &[32_768], 32_768, 0, now).is_ok());
        let err = payment_window(&[], &[32_768, 32_768], 32_768, 0, now).unwrap_err();
        assert!(err.contains("lower max_tokens"), "{err}");
        // A paid serve starts with a modest reservation and grows it as it generates (see
        // `a_paid_serve_grows_its_reservation_until_the_window_is_full`).
        assert_eq!(paid_reservation(None), DEFAULT_RESERVE_TOKENS);
        assert_eq!(paid_reservation(Some(u32::MAX)), DEFAULT_RESERVE_TOKENS);
        assert_eq!(paid_reservation(Some(512)), 512);
        // Review S3: one peer can't crowd the commitment map; past its cap its oldest goes.
        let mut map = HashMap::new();
        let commit = |peer: &str, at: u64| ServeCommitment { model_id: "m".into(), tokens: 1, recorded_ms: at, served_peer: peer.into(), delivered: false, wallet: None };
        for i in 0..(MAX_COMMITMENTS_PER_PEER as u64 + 10) {
            let mut nonce = [0u8; 16];
            nonce[..8].copy_from_slice(&i.to_le_bytes());
            insert_commitment(&mut map, nonce, commit("greedy", i));
        }
        insert_commitment(&mut map, [0xff; 16], commit("other", 0));
        assert_eq!(map.values().filter(|c| c.served_peer == "greedy").count(), MAX_COMMITMENTS_PER_PEER);
        assert!(map.values().all(|c| c.served_peer != "greedy" || c.recorded_ms >= 10), "the oldest went");
        assert!(map.contains_key(&[0xff; 16]));
        assert_eq!(parse_evm_address(&format!("0x{}", "ab".repeat(20))), Some([0xab; 20]));
        assert_eq!(parse_evm_address(&"ab".repeat(20)), Some([0xab; 20]));
        assert_eq!(parse_evm_address("0xabcd"), None);
        // A kept claim survives its encoding.
        let kc = KeptClaim { cumulative: 9, expires_at: 10, signer: Some([3; 20]), sig: vec![1; 65] };
        assert_eq!(KeptClaim::from_bytes(&kc.to_bytes()), Some(kc));
        let ks = KeptClaim { cumulative: 9, expires_at: 10, signer: None, sig: vec![1; 64] };
        assert_eq!(KeptClaim::from_bytes(&ks.to_bytes()), Some(ks));
        // A row from before the tag (an expiry, then a signature) doesn't decode as a claim.
        let mut old = 10u64.to_le_bytes().to_vec();
        old.extend_from_slice(&[0; 64]);
        assert_eq!(KeptClaim::from_bytes(&old), None);
        // An expired commitment is a receipt that never came: pruned, and its peer reported.
        let mut map = HashMap::new();
        map.insert([1u8; 16], ServeCommitment { model_id: "m".into(), tokens: 9, recorded_ms: 0, served_peer: "c".into(), delivered: true, wallet: None });
        map.insert([2u8; 16], ServeCommitment { model_id: "m".into(), tokens: 9, recorded_ms: now, served_peer: "d".into(), delivered: true, wallet: None });
        // A serve its consumer never got (a stream it gave up on) expires without counting against it.
        map.insert([3u8; 16], ServeCommitment { model_id: "m".into(), tokens: 9, recorded_ms: 0, served_peer: "e".into(), delivered: false, wallet: None });
        assert_eq!(prune_expired_commitments(&mut map, now), vec!["c".to_string()]);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn only_the_served_consumer_can_settle_its_serve() {
        let now = 1_000_000u64;
        let mut map = HashMap::new();
        commit(&mut map, [5u8; 16], "m", 100, now);
        // Another consumer key (say a throwaway one, with no history at this provider)…
        assert!(validate_and_consume_commitment(&mut map, &[5u8; 16], "m", 100, now, now, "other", "other")
            .unwrap_err()
            .contains("not from the consumer"));
        // …or the right key sent by another peer: refused, and the commitment is still there.
        assert!(validate_and_consume_commitment(&mut map, &[5u8; 16], "m", 100, now, now, "c", "relay")
            .unwrap_err()
            .contains("not from the consumer"));
        assert!(validate_and_consume_commitment(&mut map, &[5u8; 16], "m", 100, now, now, "c", "c").is_ok());
    }

    #[test]
    fn commitment_enforces_freshness_and_prunes_stale() {
        let now = 10 * COMMITMENT_TTL_MS;
        let mut map = HashMap::new();
        commit(&mut map, [3u8; 16], "m", 100, now);
        // Receipt timestamp older than the window, or far in the future → reject (B-S7).
        assert!(validate_and_consume_commitment(&mut map, &[3u8; 16], "m", 100, 0, now, "c", "c")
            .unwrap_err()
            .contains("freshness"));
        assert!(validate_and_consume_commitment(&mut map, &[3u8; 16], "m", 100, now + 10 * CLOCK_SKEW_MS, now, "c", "c")
            .unwrap_err()
            .contains("freshness"));
        // A commitment older than the TTL is pruned on the next validate, bounding the map.
        let mut map2 = HashMap::new();
        commit(&mut map2, [4u8; 16], "m", 100, 0); // recorded at ts 0, now is 10*TTL later
        assert!(validate_and_consume_commitment(&mut map2, &[4u8; 16], "m", 100, now, now, "c", "c")
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
            think: None,
            nonce,
            credential: None, deterministic: None, settlement_peg: None, settlement_domain: None, settlement_wallet: None,
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
            max_tokens: None, temperature: None, tools: Vec::new(), think: None, nonce: [0u8; 16],
            credential: None, deterministic: None, settlement_peg: None, settlement_domain: None, settlement_wallet: None,
        };
        let mut frames: Vec<Vec<u8>> = Vec::new();
        // Must NOT propagate the panic; returns a failed summary.
        let summary = generate_guarded(req, &PanicAdapter, &mut |c| frames.push(c.to_vec()), &mut |_| true);
        assert!(!summary.ok);
        assert_eq!(summary.tokens, 0, "no tokens → no commitment recorded");
        let last = frames.last().expect("a terminal frame was appended");
        assert!(matches!(ServeChunk::decode(last).unwrap(), ServeChunk::Error(_)));
        assert!(ServeChunk::frame_is_terminal(last), "the appended frame ends the stream");
    }

    /// Review round 4 H3: one challenge per peer at a time.
    #[test]
    fn a_peers_challenges_are_scored_one_at_a_time() {
        let set = Mutex::new(std::collections::HashSet::new());
        let a = OneAtATime::enter(&set, "a").unwrap();
        assert!(OneAtATime::enter(&set, "a").is_none());
        let _b = OneAtATime::enter(&set, "b").unwrap();
        drop(a);
        assert!(OneAtATime::enter(&set, "a").is_some());
    }

    /// Review round 4 C2 / round 5 R5S-1: a serve nonce is remembered once admitted, and forgotten
    /// past its memory or its oldest place.
    #[test]
    fn a_serve_nonce_is_remembered_once_admitted() {
        let mut seen = SeenNonces::default();
        assert!(!seen.seen(&[1; 16], 1_000));
        seen.remember([1; 16], 1_000);
        assert!(seen.seen(&[1; 16], 1_000 + NONCE_MEMORY_MS), "reuse refused");
        assert!(!seen.seen(&[1; 16], 1_001 + NONCE_MEMORY_MS), "forgotten after the memory");
        // Bounded: past the cap the oldest goes, in order.
        let mut full = SeenNonces::default();
        for i in 0..MAX_SEEN_NONCES as u32 {
            let mut n = [0u8; 16];
            n[..4].copy_from_slice(&i.to_be_bytes());
            full.remember(n, 5_000);
        }
        full.remember([9; 16], 5_001);
        assert_eq!(full.at.len(), MAX_SEEN_NONCES);
        assert!(!full.seen(&[0u8; 16], 5_001), "the oldest was forgotten");
        assert!(full.seen(&[9; 16], 5_001));
        // A commitment is never replaced by another serve's insert.
        let mut map = HashMap::new();
        let c = |tokens| ServeCommitment {
            model_id: "m".into(), tokens, recorded_ms: 1, served_peer: "c".into(), delivered: true, wallet: None,
        };
        insert_commitment(&mut map, [1; 16], c(8_000));
        insert_commitment(&mut map, [1; 16], c(1));
        assert_eq!(map[&[1; 16]].tokens, 8_000);
    }

    #[test]
    fn stream_begin_ttl_prunes_expired_buffers() {
        let mut map = HashMap::new();
        stream_begin_in(&mut map, [1u8; 16], "c", 0); // recorded at t=0, never finished
        // A begin past the TTL (an unfinished stream's: past the abandon window too) prunes the
        // stale buffer as a side effect.
        stream_begin_in(&mut map, [2u8; 16], "c", STREAM_TTL_MS.max(STREAM_ABANDON_MS + 30_000) + 1);
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

    // ── Review round 3, D4: pay for what was received ─────────────────────

    fn test_provider<A: EngineAdapter>(adapter: A) -> (tempfile::TempDir, Provider<A>) {
        let dir = tempfile::tempdir().unwrap();
        let net = openhydra_network::handle::NetworkHandle::start(openhydra_network::node::NodeConfig {
            identity_path: dir.path().join("id.key"),
            listen_addrs: vec!["/ip4/127.0.0.1/tcp/0".into()],
            bootstrap_peers: vec![],
            enable_peer_relay: false,
            enable_connection_reversal: false,
            pcp_gateway: None,
        })
        .unwrap();
        (dir, Provider::new(adapter, net))
    }

    /// Round 8 S8-2: a buffered serve's nonce is held while it generates, however long, then
    /// remembered from its end.
    #[test]
    fn a_buffered_serves_nonce_is_held_while_it_runs() {
        let (_dir, provider) = test_provider(StubAdapter);
        let req = ServeRequest {
            reply_to: "c".into(), model_ref: "m".into(), messages: vec![],
            max_tokens: None, temperature: None, tools: Vec::new(), think: None, nonce: [3u8; 16],
            credential: None, deterministic: None, settlement_peg: None, settlement_domain: None, settlement_wallet: None,
        };
        assert!(provider.nonce_refusal(&req, "c").is_none(), "admitted");
        provider.hold_nonce(req.nonce, "c");
        // The serve outlasted the nonce memory.
        provider.seen_nonces.lock().unwrap().at.clear();
        assert!(provider.nonce_refusal(&req, "c").is_some(), "still held while generating");
        provider.end_hold(req.nonce, "c");
        assert!(provider.serving_nonces.lock().unwrap().is_empty());
        assert!(provider.nonce_refusal(&req, "c").is_some(), "remembered from the serve's end");
    }

    fn billing(max_tokens: Option<u32>) -> StreamBilling {
        StreamBilling { model_id: "m".into(), max_tokens, wallet: None }
    }

    /// The consumer's side of a poll: the billable count of the frames it got, as it counts them
    /// (each text fragment through `on_delta`, the `Done` count when it arrives).
    fn consumer_count(tally: &mut crate::serve::StreamTally, resp: &FetchChunksResponse) {
        if let FetchChunksResponse::Chunks { framed, .. } = resp {
            for chunk in parse_response(framed).unwrap() {
                match chunk {
                    ServeChunk::Delta(t) => tally.push_delta(&t),
                    other => tally.push_encoded(&other.encode()),
                }
            }
        }
    }

    #[test]
    fn a_stream_is_owed_what_its_polls_returned_and_both_sides_count_it_alike() {
        let n = [5u8; 16];
        let mut map = HashMap::new();
        stream_begin_billed_in(&mut map, n, "c", 0, billing(None));
        // Nothing generated yet: a poll returns nothing, and nothing is owed.
        assert_eq!(fetch_chunks_billed(&mut map, &n, "c", 0, 1).1, None);
        let words = ["Hello", " world", ",", " this", " is", " a", " long", " answer", " that", " stops"];
        for w in &words[..9] {
            stream_append_in(&mut map, &n, &delta(w), 2);
        }
        let mut consumer = crate::serve::StreamTally::default();
        let (resp, owed) = fetch_chunks_billed(&mut map, &n, "c", 0, 3);
        consumer_count(&mut consumer, &resp);
        let owed = owed.expect("frames returned → owed");
        // Nine fragments, nine tokens (under the text bound), and the consumer counts the same.
        assert_eq!(owed.tokens, 9);
        assert_eq!(consumer.billable(None), owed.tokens);
        // A re-poll of the same offset (its reply was lost) returns the same frames: nothing more owed.
        assert_eq!(fetch_chunks_billed(&mut map, &n, "c", 0, 4).1, None);
        // The consumer stops reading here: the 10th fragment and the Done are never polled, so a
        // stream read to 90% is owed 90%.
        stream_append_in(&mut map, &n, &delta(words[9]), 5);
        stream_append_in(&mut map, &n, &done(10), 5);
        assert_eq!(owed.model_id, "m");

        // A stream read to its end is owed the Done count, on both sides.
        let mut map = HashMap::new();
        stream_begin_billed_in(&mut map, n, "c", 0, billing(Some(8)));
        let mut consumer = crate::serve::StreamTally::default();
        let mut offset = 0;
        let mut last = None;
        for (i, w) in words.iter().enumerate() {
            stream_append_in(&mut map, &n, &delta(w), i as u64);
            if i % 3 == 2 {
                let (resp, owed) = fetch_chunks_billed(&mut map, &n, "c", offset, i as u64);
                consumer_count(&mut consumer, &resp);
                if let FetchChunksResponse::Chunks { next_offset, .. } = resp {
                    offset = next_offset;
                }
                let owed = owed.unwrap();
                // Mid-stream, capped by the request's max_tokens like any count.
                assert_eq!(owed.tokens, consumer.billable(Some(8)));
                assert!(owed.tokens <= 8);
                last = Some(owed.tokens);
            }
        }
        assert_eq!(last, Some(8));
        stream_append_in(&mut map, &n, &done(7), 20);
        let (resp, owed) = fetch_chunks_billed(&mut map, &n, "c", offset, 21);
        consumer_count(&mut consumer, &resp);
        assert_eq!(owed.unwrap().tokens, 7, "the Done frame carries the whole completion's count");
        assert_eq!(consumer.billable(Some(8)), 7);
        // Nobody else's poll is owed anything.
        assert_eq!(fetch_chunks_billed(&mut map, &n, "x", 0, 22).1, None);
    }

    #[test]
    fn a_partially_read_stream_is_committed_as_delivered_and_an_unread_one_never_is() {
        let (_dir, provider) = test_provider(StubAdapter);
        let n = [6u8; 16];
        provider.stream_begin(n, "c", billing(None));
        for w in ["a", "b", "c"] {
            provider.stream_append(n, &delta(w));
        }
        // An unread stream has no commitment: nothing to pay, nothing to default on.
        assert!(provider.serve_commitments.lock().unwrap().is_empty());
        let mut poll = n.to_vec();
        poll.extend_from_slice(&0u32.to_be_bytes());
        provider.handle_fetch_chunks("c", &poll);
        {
            let map = provider.serve_commitments.lock().unwrap();
            let c = &map[&n];
            assert_eq!((c.tokens, c.delivered, c.served_peer.as_str()), (3, true, "c"));
        }
        // More read later raises it.
        provider.stream_append(n, &delta("d"));
        let mut poll = n.to_vec();
        poll.extend_from_slice(&3u32.to_be_bytes());
        provider.handle_fetch_chunks("c", &poll);
        assert_eq!(provider.serve_commitments.lock().unwrap()[&n].tokens, 4);
        // Left unpaid, it is a default like an unpaid whole serve.
        let later = now_unix_ms() + COMMITMENT_TTL_MS + 1;
        assert_eq!(prune_expired_commitments(&mut provider.serve_commitments.lock().unwrap(), later), vec!["c".to_string()]);
        // A stream nobody polled leaves nothing behind.
        let m = [7u8; 16];
        provider.stream_begin(m, "c", billing(None));
        provider.stream_append(m, &delta("unread"));
        provider.stream_append(m, &done(1));
        assert!(provider.serve_commitments.lock().unwrap().get(&m).is_none());
    }

    #[test]
    fn a_stream_stops_generating_once_its_consumer_stops_polling() {
        /// Streams up to 1,000 fragments, stopping when asked (as every adapter checks `stop`).
        struct Counting(std::sync::atomic::AtomicU64);
        impl EngineAdapter for Counting {
            fn engine_name(&self) -> &'static str {
                "counting"
            }
            fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
                Ok(vec![])
            }
            fn serve_stream(&self, req: &InferenceRequest, on_delta: &mut dyn FnMut(&str)) -> Result<ServeOutcome, AdapterError> {
                for _ in 0..1_000 {
                    if req.stop.is_set() {
                        break;
                    }
                    self.0.fetch_add(1, Ordering::Relaxed);
                    on_delta("x ");
                }
                Ok(ServeOutcome { tokens: 1_000, done: true, engine: Default::default(), tool_calls: Vec::new() })
            }
        }
        let (_dir, provider) = test_provider(Counting(Default::default()));
        let req = |nonce| ServeRequest {
            reply_to: "c".into(),
            model_ref: "m".into(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
            tools: Vec::new(),
            think: None,
            nonce,
            credential: None, deterministic: None, settlement_peg: None, settlement_domain: None, settlement_wallet: None,
        };
        let run = |nonce: [u8; 16]| {
            generate_guarded(req(nonce), &provider.adapter, &mut |c| provider.stream_append(nonce, c), &mut |_| {
                provider.stream_wanted(&nonce)
            })
        };
        // Polled recently: it runs to the end.
        let live = [1u8; 16];
        provider.stream_begin(live, "c", billing(None));
        assert_eq!(run(live).tokens, 1_000);
        assert_eq!(provider.adapter.0.swap(0, Ordering::Relaxed), 1_000);
        // Last polled over STREAM_ABANDON_MS ago: it stops after the fragment that noticed.
        let gone = [2u8; 16];
        provider.stream_begin(gone, "c", billing(None));
        {
            let mut streams = provider.streams.lock().unwrap();
            let b = streams.get_mut(&gone).unwrap();
            b.polled = true;
            b.last_poll_ms = now_unix_ms() - STREAM_ABANDON_MS - 1;
        }
        assert_eq!(run(gone).tokens, 1);
        assert_eq!(provider.adapter.0.swap(0, Ordering::Relaxed), 1);
        // Never polled at all, past its own (longer) limit: it stops too.
        let never = [4u8; 16];
        provider.stream_begin(never, "c", billing(None));
        provider.streams.lock().unwrap().get_mut(&never).unwrap().last_poll_ms = now_unix_ms() - NEVER_POLLED_ABANDON_MS - 1;
        assert_eq!(run(never).tokens, 1);
        assert_eq!(provider.adapter.0.swap(0, Ordering::Relaxed), 1);
        // Closed (settled, or evicted): it stops too.
        let closed = [3u8; 16];
        assert_eq!(run(closed).tokens, 1);
    }

    #[test]
    fn a_buffered_reply_counts_as_delivered_only_once_it_reached_its_consumer() {
        let (_dir, provider) = test_provider(StubAdapter);
        let n = [8u8; 16];
        provider.record_commitment(n, "m".into(), 5, "c", false, None);
        // Its reply never went out (or failed): not delivered, so expiring unpaid isn't held against
        // the consumer (`prune_expired_commitments` counts only delivered ones).
        assert!(!provider.serve_commitments.lock().unwrap()[&n].delivered);
        // Another peer's fetch of it reaching that peer marks nothing.
        provider.mark_delivered(n, "x");
        assert!(!provider.serve_commitments.lock().unwrap()[&n].delivered);
        // Its consumer got it: now it is.
        provider.mark_delivered(n, "c");
        assert!(provider.serve_commitments.lock().unwrap()[&n].delivered);
        // A nonce with no commitment (a refusal) is left alone.
        provider.mark_delivered([9u8; 16], "c");
        assert_eq!(provider.serve_commitments.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_full_commitment_map_makes_room_from_replies_that_never_arrived() {
        let commit = |peer: &str, at: u64, delivered: bool| ServeCommitment {
            model_id: "m".into(), tokens: 1, recorded_ms: at, served_peer: peer.into(), delivered, wallet: None,
        };
        let mut map = HashMap::new();
        for i in 0..MAX_COMMITMENTS {
            let mut n = [0u8; 16];
            n[..8].copy_from_slice(&(i as u64).to_le_bytes());
            // Spread over peers so the per-peer cap doesn't bite; two undelivered, the rest delivered.
            insert_commitment(&mut map, n, commit(&format!("p{}", i % 100), 10 + i as u64, i != 50 && i != 70));
        }
        assert_eq!(map.len(), MAX_COMMITMENTS);
        insert_commitment(&mut map, [0xAA; 16], commit("new", 1_000_000, true));
        assert!(map.contains_key(&[0xAA; 16]), "room was made");
        let mut n50 = [0u8; 16];
        n50[..8].copy_from_slice(&50u64.to_le_bytes());
        assert!(!map.contains_key(&n50), "the oldest undelivered one went");
        insert_commitment(&mut map, [0xBB; 16], commit("new2", 1_000_001, true));
        insert_commitment(&mut map, [0xCC; 16], commit("new3", 1_000_002, true));
        assert!(map.contains_key(&[0xBB; 16]));
        assert!(!map.contains_key(&[0xCC; 16]), "with only delivered ones left, nothing more is added");
        assert_eq!(map.len(), MAX_COMMITMENTS);
    }

    // ── Review round 3, S8: close-and-fetch ──

    #[test]
    fn a_close_returns_what_was_counted_and_nothing_newer() {
        let n = [11u8; 16];
        let mut map = HashMap::new();
        stream_begin_billed_in(&mut map, n, "c", 0, billing(None));
        for w in ["a", "b", "c"] {
            stream_append_in(&mut map, &n, &delta(w), 1);
        }
        // The consumer polled from 0; the provider sent 3 frames (counted) but the reply was lost.
        let owed = fetch_chunks_billed(&mut map, &n, "c", 0, 2).1.unwrap().tokens;
        // More is generated meanwhile; nobody has polled for it.
        stream_append_in(&mut map, &n, &delta("d"), 3);
        // The consumer closes from where it had got to (0): it gets exactly the counted frames.
        let mut consumer = crate::serve::StreamTally::default();
        let resp = close_stream_in(&mut map, &n, "c", 0, 4);
        consumer_count(&mut consumer, &resp);
        assert!(matches!(resp, FetchChunksResponse::Chunks { next_offset: 3, done: true, .. }), "{resp:?}");
        assert_eq!(consumer.billable(None), owed, "both sides now count the same");
        // Closed: a later poll, or close, returns nothing new and owes nothing more.
        stream_append_in(&mut map, &n, &delta("e"), 5);
        assert_eq!(fetch_chunks_billed(&mut map, &n, "c", 3, 6).1, None);
        assert!(matches!(fetch_chunks_billed(&mut map, &n, "c", 3, 6).0, FetchChunksResponse::Chunks { ref framed, .. } if parse_response(framed).unwrap().is_empty()));
        // Someone else can't close it, and an unknown stream is NotFound.
        assert_eq!(close_stream_in(&mut map, &n, "x", 0, 7), FetchChunksResponse::Forbidden);
        assert_eq!(close_stream_in(&mut map, &[0u8; 16], "c", 0, 7), FetchChunksResponse::NotFound);
    }

    #[test]
    fn a_read_stream_is_kept_for_its_payment_window_and_closing_stops_generation() {
        let (_dir, provider) = test_provider(StubAdapter);
        let n = [12u8; 16];
        provider.stream_begin(n, "c", billing(None));
        provider.stream_append(n, &delta("a"));
        assert!(provider.stream_wanted(&n));
        let mut close = n.to_vec();
        close.extend_from_slice(&0u32.to_be_bytes());
        // Its commitment, last stamped long ago, is refreshed by its consumer's close (round 8 S8-3).
        provider.serve_commitments.lock().unwrap().insert(
            n,
            ServeCommitment { model_id: "m".into(), tokens: 1, recorded_ms: 1, served_peer: "c".into(), delivered: true, wallet: None },
        );
        let resp = FetchChunksResponse::decode(&provider.handle_stream_close("c", &close)).unwrap();
        assert!(matches!(resp, FetchChunksResponse::Chunks { done: true, .. }));
        assert!(!provider.stream_wanted(&n), "closing stops generation");
        assert!(provider.serve_commitments.lock().unwrap()[&n].recorded_ms > 1, "refreshed by the close");
        // Retention: an unread finished stream goes after the stream TTL, an unread one still
        // generating outlives the abandon window (R5S-6); a read one lasts the commitment's.
        let map = provider.streams.lock().unwrap();
        let mut b = StreamBuffer { recorded_ms: 0, fetched_hi: 0, done: true, ..clone_buffer(&map[&n]) };
        assert!(stream_expired(&b, STREAM_TTL_MS + 1));
        b.done = false;
        assert!(!stream_expired(&b, STREAM_ABANDON_MS));
        assert!(stream_expired(&b, STREAM_ABANDON_MS + 30_001));
        b.done = true;
        b.fetched_hi = 1;
        assert!(!stream_expired(&b, STREAM_TTL_MS + 1));
        assert!(stream_expired(&b, COMMITMENT_TTL_MS.max(STREAM_TTL_MS) + 1));
    }

    fn clone_buffer(b: &StreamBuffer) -> StreamBuffer {
        StreamBuffer {
            reply_to: b.reply_to.clone(),
            base_offset: b.base_offset,
            chunks: b.chunks.clone(),
            done: b.done,
            bytes: b.bytes,
            recorded_ms: b.recorded_ms,
            last_poll_ms: b.last_poll_ms,
            tally: b.tally,
            fetched_hi: b.fetched_hi,
            billing: b.billing.clone(),
            closed: b.closed,
            polled: b.polled,
        }
    }
}
