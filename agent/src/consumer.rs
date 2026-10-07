// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Consumer-side serve client — the core the HTTP/SSE gateway calls.
//!
//! [`request_completion`] sends a [`ServeRequest`] to a chosen provider over an injected
//! transport (live: a `proxy_forward` to the provider's libp2p id), parses the buffered
//! framed response, and pushes each text delta to a callback (→ SSE). Transport is
//! injected so the request→serve→response→parse loop is unit-tested in-process against
//! the provider handler — no swarm, no engine.
//!
//! Provider *selection* (discover → filter → rank → pick) and the HTTP front door land on
//! top of this; this is just "talk to one provider".

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use openhydra_network::handle::NetworkHandle;
use openhydra_network::types::DiscoveredPeer;
use openhydra_protocol::credit::CreditAccount;
use openhydra_protocol::model_id::is_compatible;
use openhydra_protocol::receipts::NonceTracker;
use openhydra_protocol::router::{rank_peers, PeerScoreInput};
use openhydra_protocol::store::Store;
use openhydra_protocol::verify::{
    reputation_key, resolve_audit, sample_rate_for_reputation, split_reputation_key,
    AuditReport, RedundantVerdict,
    ReputationTracker,
    VerificationOutcome, NEUTRAL_REPUTATION,
};
use openhydra_protocol::award::{AuditClass, AuditStats};
use openhydra_protocol::trust::{
    AssurancePolicy, AssuranceTier, AuthenticityEvidence, AuthenticityTier,
    MethodFlags, PolicyMode, PrivacyTier, RedundantExecAuthenticity, SettlementTier, Verifier,
};
use openhydra_protocol::payout::{gated_accrual_asset, FeeSplit, GatedVoucher, SettlementAsset};
use openhydra_protocol::weight::{
    cost_tier_for_handle, credits_for_receipt, receipt_weighted_units, CostTier,
};

use serde_json::Value;

use crate::adapter::{AdapterError, ChatMessage, ToolCall};
use crate::attest::{attest_provider, AttestRecord};
use crate::prompt_score::{
    decode_logprob_state, encode_logprob_state, identity_in, priced_tier, probe_scoring, quant_hint_from_name, reference_tag, request_model_info, run_round,
    select_reference, LoadedReference, LogprobStatus, RoundOutcome, ScoreChallengeError, StatusEffect,
    LOGPROB_BACKOFF_MS, LOGPROB_RETRY_MS,
};
use openhydra_protocol::logprob::ChallengeRound;
use crate::provider::{FETCH_CHUNKS, FETCH_RESULT, SERVE_REQUEST, SERVE_STREAM};
use crate::receipt::{request_receipt, ReceiptFailure};
use crate::serve::{
    parse_response, FetchChunksResponse, FetchResponse, ServeChunk,
    ServeRequest, ServeSummary,
};

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The provider the consumer chose to serve a request.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedProvider {
    /// libp2p id — the `proxy_forward` dial target.
    pub libp2p_peer_id: String,
    /// OpenHydra id (64-bit) — what logs and status views show.
    pub peer_id: String,
    /// The 128-bit id its standing is kept under ([`provider_state_id`]): the peer half of every
    /// per-`(peer, model)` key — reputation, audits, authenticity, log-prob checks, attestation,
    /// payment.
    pub state_id: String,
    /// Provider's ed25519 public key (hex) — for the co-signed receipt at EOS.
    pub public_key: String,
    /// The model id the provider serves.
    pub model_id: String,
    /// The provider's canonical model id (`family/params/quant/hash`), or `""` if it advertised
    /// none (legacy). The model half of the per-`(peer, model)` reputation key — canonical when
    /// present, else the raw [`model_id`](Self::model_id) (see [`rep_model_key`]).
    pub canonical_model_id: String,
    /// M4: the swarm (hex group public key) this candidate is gated on, when it came from a private
    /// card — the consumer presents the matching credential on this route. `""` for a public route.
    pub swarm_public_key: String,
}

// Per-provider attempt budget for the (buffered) serve round-trip — see [`attempt_timeout`].
// A cold model load + prefill on a big agentic system prompt + generating `max_tokens` on a slow
// remote model over a relay can take minutes; the old fixed 45s cap abandoned long agentic serves
// mid-generation (live-diagnosed 2026-08-14: `proxy_forward timed out after 45s`). We scale the
// budget by the request's `max_tokens` so a small request still fails over fast from a dead
// provider while a big one gets room to finish. This only WIDENS the buffered window — the durable
// fix is end-to-end streaming (a stall timeout instead of a total cap); see STREAMING_SERVE_PLAN.
const ATTEMPT_TIMEOUT_BASE_SECS: u64 = 45; // discovery jitter + a cold model load
const ATTEMPT_TIMEOUT_MS_PER_TOKEN: u64 = 40; // generation + prefill headroom on a slow remote model
const ATTEMPT_TIMEOUT_MIN_SECS: u64 = 45;
const ATTEMPT_TIMEOUT_MAX_SECS: u64 = 240; // bound so a dead provider still frees its slot for failover
/// The id a provider's standing is kept under: [`openhydra_network::identity::openhydra_state_id`] of
/// its (verified) public key, a 128-bit id whose first 16 hex characters are its 64-bit peer id; a
/// legacy provider without a key falls back to its peer id.
pub fn provider_state_id(peer_id: &str, public_key_hex: &str) -> String {
    match hex::decode(public_key_hex) {
        Ok(pk) if pk.len() == 32 => openhydra_network::identity::openhydra_state_id(&pk),
        _ => peer_id.to_string(),
    }
}

/// The crypto-rail voucher key of a provider: its `(peer, model)` reputation key plus its full
/// public key. Payment state follows the key the on-chain claim is made with, so two providers whose
/// 64-bit OpenHydra ids collide never share (and each claim) one running total.
pub(crate) fn voucher_key(peer_id: &str, model_key: &str, provider_pub: &[u8]) -> String {
    format!("{}#{}", reputation_key(peer_id, model_key), hex::encode(provider_pub))
}

/// Assumed output length when a request omits `max_tokens`, for budgeting only.
const ATTEMPT_TIMEOUT_DEFAULT_TOKENS: u64 = 1024;

/// The per-attempt serve budget for a request of `max_tokens`. Honors an
/// `OPENHYDRA_ATTEMPT_TIMEOUT_SECS` ops override (fixed seconds); otherwise
/// `base + max_tokens · per_token`, clamped to `[MIN, MAX]`.
fn attempt_timeout(max_tokens: Option<u32>) -> std::time::Duration {
    if let Some(secs) = std::env::var("OPENHYDRA_ATTEMPT_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        return std::time::Duration::from_secs(secs.clamp(ATTEMPT_TIMEOUT_MIN_SECS, ATTEMPT_TIMEOUT_MAX_SECS));
    }
    default_attempt_timeout(max_tokens)
}

/// [`attempt_timeout`] without the ops override: what a provider can expect a consumer to wait.
pub(crate) fn default_attempt_timeout(max_tokens: Option<u32>) -> std::time::Duration {
    let tokens = max_tokens.map(u64::from).unwrap_or(ATTEMPT_TIMEOUT_DEFAULT_TOKENS);
    let secs = ATTEMPT_TIMEOUT_BASE_SECS + tokens.saturating_mul(ATTEMPT_TIMEOUT_MS_PER_TOKEN) / 1000;
    std::time::Duration::from_secs(secs.clamp(ATTEMPT_TIMEOUT_MIN_SECS, ATTEMPT_TIMEOUT_MAX_SECS))
}

/// Reconnect-and-fetch: per fetch-attempt timeout. A fetch is a cheap provider-side buffer
/// lookup, so this is short — the cost is re-dialing over a (re-establishing) relay circuit.
const FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);
/// Reconnect-and-fetch: backoff between fetch attempts, tuned to the ~10s relay re-establish
/// window (a fetch that fails at the transport layer means the circuit isn't back yet).
const FETCH_BACKOFF: std::time::Duration = std::time::Duration::from_secs(8);
/// Reconnect-and-fetch: extra window (beyond the serve budget) to keep re-polling for the buffered
/// result across relay reconnects before giving up. The overall recovery deadline is
/// `attempt_timeout(max_tokens) + this`, so a long generation the consumer reconnected into can
/// still finish and be fetched.
const FETCH_RECONNECT_GRACE: std::time::Duration = std::time::Duration::from_secs(90);

/// Bound on the detached receipt-settlement round-trip (G5). Settlement runs off the response
/// path, so this only caps how long a provider that black-holes the RECEIPT_REQUEST can park
/// the detached settlement thread — kept well below the serve budget ([`attempt_timeout`]) since credit is
/// advisory and a dropped settlement never affects the delivered completion.
const SETTLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// Settlements in flight at once, across all providers. Each is a detached thread; past this a new
/// one is dropped (logged) rather than spawned, so a burst of serves can't grow threads without bound.
const MAX_SETTLES_IN_FLIGHT: usize = 512;
/// How long a provider that refused us over payment is tried last (see `payment_refusals`).
const PAYMENT_REFUSAL_TTL_MS: u64 = 10 * 60 * 1000;
/// Providers whose recent settlements a consumer remembers (for [`ConsumerNode::paid_recently`]).
const MAX_SETTLE_RECORDS: usize = 4096;
/// How long a provider that shed our request, or couldn't be reached, is tried last.
const UNAVAILABLE_RANK_LAST_MS: u64 = 60 * 1000;
/// A second shed-free failure to reach a provider within this long costs it reputation.
const UNAVAILABLE_REPEAT_MS: u64 = 10 * 60 * 1000;
/// How recent a completed serve shows a provider's engine working (for a follow-up's refusal).
const SERVED_OK_RECENT_MS: u64 = 10 * 60 * 1000;
/// Only one audit shed per provider and model is excused within this long.
const AUDIT_SHED_WINDOW_MS: u64 = 24 * 60 * 60 * 1000;
/// How close to a follow-up's last unheard try our own request of the model must have been served,
/// for running out of tries to count as avoiding it.
const AUDIT_SHED_EVIDENCE_MS: u64 = 60 * 1000;
/// Times a follow-up audit is tried again after its subject couldn't be heard (engine down, network).
const FOLLOW_UP_RETRIES: u32 = 3;
/// Payment refusals remembered at once (the oldest are forgotten first).
const MAX_PAYMENT_REFUSALS: usize = 1024;
/// Settlements running or queued for one `(peer, model)`. They run one at a time (its lock), so with
/// a provider that black-holes receipts each waits out [`SETTLE_TIMEOUT`] in turn; more than a few
/// queued behind it would only wait to fail.
const MAX_SETTLES_PER_PAIR: usize = 32;
/// Retries of a settlement whose round-trip failed (or the provider asked us to retry), and the wait
/// before each: a serve that goes unpaid counts against us at a provider that charges.
const SETTLE_RETRY_WAITS: [Duration; 2] = [Duration::from_secs(2), Duration::from_secs(8)];

/// The waits before each try of a stream close (S8): spread over most of the provider's commitment
/// lifetime (5 min), so a close outlives a short outage or a busy provider (review round 4 M5).
const CLOSE_RETRY_WAITS: [Duration; 6] = [
    Duration::ZERO,
    Duration::from_secs(2),
    Duration::from_secs(8),
    Duration::from_secs(20),
    Duration::from_secs(40),
    Duration::from_secs(60),
];
/// The most a stream close's reply may add to what we read, in fragments and in billed tokens alike,
/// when the request sets no `max_tokens` (review round 5, R5S-2: a lower cap would underpay an
/// honest provider whose long poll reply was lost).
const CLOSE_TAIL_MAX_TOKENS: u64 = 65_536;

/// What a stream close's reply may add to `tally`'s bill: [`CLOSE_TAIL_MAX_TOKENS`], or the request's
/// `max_tokens`.
fn close_tail_cap(max_tokens: Option<u32>) -> u64 {
    max_tokens.map_or(CLOSE_TAIL_MAX_TOKENS, u64::from)
}

/// Add a stream close's reply (`framed`) to `tally`: at most [`close_tail_cap`] more text fragments
/// (what we missed is at most a lost poll reply or two), and its other frames.
fn absorb_close_reply(tally: &mut crate::serve::StreamTally, framed: &[u8], max_tokens: Option<u32>) {
    let before = tally.fragments();
    let cap = close_tail_cap(max_tokens);
    for chunk in parse_response(framed).unwrap_or_default() {
        if matches!(chunk, crate::serve::ServeChunk::Delta(_)) && tally.fragments() - before >= cap {
            continue;
        }
        tally.push_encoded(&chunk.encode());
    }
}
/// The most delivered text (in [`crate::serve::DeliveredText`] units, about bytes) one attempt
/// reads before failing (round 8 G8-3): a flat bound on what a provider can make the gateway hold.
/// Not per token: an image or audio result arrives as one large data-URL fragment, whatever
/// `max_tokens` says (round 9 B9-1); billing is bounded by `max_tokens` regardless.
const MAX_DELIVERED_TEXT_UNITS: u64 = 128 * 1024 * 1024;

fn delivered_text_cap(_max_tokens: Option<u32>) -> u64 {
    MAX_DELIVERED_TEXT_UNITS
}

/// Stream closes running at once (each a detached thread retrying for minutes) (R5S-3).
const MAX_CLOSES_IN_FLIGHT: usize = 256;

/// Whether a failed settlement is worth sending again: the round-trip failed, or the provider asked
/// for a retry (rate limited). A provider's considered refusal isn't.
fn settle_retryable(e: &AdapterError) -> bool {
    let msg = e.to_string();
    !msg.contains("provider rejected receipt") || msg.contains("retry")
}

/// Whether a reported count above the billable one (`billable`) counts against the provider: only
/// from a provider that says it follows the shared billing rule ([`crate::serve::BILLING_VERSION`]),
/// which never reports more. An older provider reported the engine's own count, which can honestly
/// be higher (reasoning it didn't stream); it is just signed for less.
/// A provider on an older rule (below 3) is held to the most generous rule there ever was (the
/// engine's own count, never more than the request's `max_tokens`): above that, it can't be honest
/// (review B6). A newer rule than ours isn't judged ([`newer_billing_rule`]).
fn over_reported_by_billing(summary: &crate::serve::ServeSummary, billable: u64, max_tokens: Option<u32>) -> bool {
    let billing = summary.metrics.billing;
    if crate::serve::same_completed_billing(billing) {
        summary.tokens > billable
    } else if billing < 3 {
        max_tokens.is_some_and(|m| summary.tokens > u64::from(m))
    } else {
        false
    }
}

/// Whether a provider says it bills by a rule newer than this build knows: not judged, but noted
/// (logged, and counted in status), since a "newer rule" is also what a provider would claim to
/// avoid being judged (review B6).
fn newer_billing_rule(summary: &crate::serve::ServeSummary) -> bool {
    summary.metrics.billing > crate::serve::BILLING_VERSION
}

/// The network an address sits on, for telling providers' operators apart: its IPv4 /24 or IPv6 /48.
fn network_of(ip: &str) -> Option<String> {
    match ip.trim_matches(['[', ']']).parse::<std::net::IpAddr>().ok()? {
        std::net::IpAddr::V4(v4) => {
            let o = v4.octets();
            Some(format!("{}.{}.{}.0/24", o[0], o[1], o[2]))
        }
        std::net::IpAddr::V6(v6) => {
            let s = v6.segments();
            Some(format!("{:x}:{:x}:{:x}::/48", s[0], s[1], s[2]))
        }
    }
}

/// What a consumer makes of the co-signed receipt a provider sends back when the consumer's
/// cumulative is behind its high-water.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BehindReceipt {
    /// Not a valid receipt between this consumer and provider for this model, with a cumulative.
    NotOurs,
    /// Ours, but no further than what this serve's receipt built on: nothing to catch up to (a
    /// provider refusing this serve's amount isn't ahead of us; it disputes it).
    Stale,
    /// Ours and further along, but not shown to be signed for the deployment we settle on: on a
    /// settlement chain only the receipt's claim, signed for this deployment, shows that. A receipt
    /// from another deployment (or from before one) would re-authorize that history here.
    NotThisDeployment,
    /// This very serve's receipt (same nonce), co-signed before at this cumulative: it settled, and
    /// only the reply was lost.
    ThisServe(u64),
    /// An earlier serve's receipt at this cumulative, past our voucher: adopt it and sign again.
    Adopt(u64),
}

/// What a "behind" receipt is judged against: this serve's receipt and the settlement it uses.
struct BehindContext<'a> {
    consumer_pub: &'a [u8],
    provider_pub: &'a [u8],
    model_id: &'a str,
    nonce: &'a [u8; 16],
    /// What this serve's receipt built on: the higher of the pair's voucher and the most we had
    /// signed for it before this serve.
    base: u64,
    /// The peg receipts are signed at now.
    peg: Option<u64>,
    /// Whether the receipt was signed for the deployment we settle on (always, off-chain).
    signed_here: &'a dyn Fn(&crate::receipt::LatestReceipt) -> bool,
}

/// Judge a provider's "behind" receipt. Only a receipt this consumer signed (and the provider
/// co-signed) for this pair and model, at the peg we sign at now, counts: the provider can claim
/// anything we signed, so its cumulative is safe to adopt when it was signed for this deployment and
/// is past what this serve built on; nothing else is.
fn judge_behind_receipt(latest: &crate::receipt::LatestReceipt, cx: &BehindContext) -> BehindReceipt {
    let p = &latest.receipt.payload;
    let ours = openhydra_protocol::receipts::verify_receipt(&latest.receipt).is_ok()
        && p.consumer.as_bytes() == cx.consumer_pub
        && p.provider.as_bytes() == cx.provider_pub
        && p.model_id == cx.model_id
        // A paid receipt in the asset we pay in. Its amount is in that asset's micro-units whatever
        // peg it was stamped at, so a peg change doesn't strand the pair behind it (review S7).
        && p.peg_micros_per_credit.is_some() == cx.peg.is_some()
        && p.settlement_asset == cx.peg.map(|_| SettlementAsset::Usdc);
    match p.cumulative_amount {
        Some(c) if ours && &p.nonce == cx.nonce => BehindReceipt::ThisServe(c),
        Some(c) if ours && c <= cx.base => BehindReceipt::Stale,
        Some(_) if ours && !(cx.signed_here)(latest) => BehindReceipt::NotThisDeployment,
        Some(c) if ours => BehindReceipt::Adopt(c),
        _ => BehindReceipt::NotOurs,
    }
}

/// Whether `latest` was signed for `chain`'s deployment, by this consumer: its claim's signature
/// checks there (Solana: by the receipt's consumer key; Base: by our wallet). Off-chain, always.
fn signed_for_deployment(chain: Option<&crate::settlement_config::SettlementChain>, latest: &crate::receipt::LatestReceipt) -> bool {
    use openhydra_protocol::settlement::{claim_signer, evm_address, ClaimSigner};
    let Some(chain) = chain else { return true };
    let Some(sig) = &latest.claim else { return false };
    let Some(claim) = crate::receipt::claim_for(&latest.receipt.payload, sig.expires_at) else { return false };
    match claim_signer(&chain.domain, &claim, &sig.sig) {
        Ok(ClaimSigner::Solana) => true,
        Ok(ClaimSigner::Evm(addr)) => chain.evm_wallet.as_ref().is_some_and(|w| evm_address(w.verifying_key()) == addr),
        Err(_) => false,
    }
}

/// Whether a new settlement may start, given how many are in flight and how many already hold or
/// wait on its pair's lock.
fn settle_admitted(in_flight: usize, pair_queue: usize) -> bool {
    in_flight < MAX_SETTLES_IN_FLIGHT && pair_queue < MAX_SETTLES_PER_PAIR
}

/// Decrements the in-flight settlement count when a settlement thread ends (however it ends).
struct SettleSlot(Arc<std::sync::atomic::AtomicUsize>);

impl Drop for SettleSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// Bound on one device-attestation challenge round-trip (detached, off the response path).
const ATTEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);

/// Consumer-side device attestation of providers (ATTESTED_ENGINE_TIER_PLAN_v1 Phase 0, opt-in).
/// The result is surfaced as a device-trust badge; it never changes routing or the authenticity tier.
struct ProviderAttestation {
    /// SHA-256 hashes of trusted OpenHydra release binaries (see `AttestPolicy::trusted_binaries`).
    trusted_binaries: Vec<[u8; 32]>,
    /// Latest challenge record per provider, keyed by OpenHydra peer id.
    records: Mutex<HashMap<String, AttestRecord>>,
    /// Providers with a challenge in flight, so concurrent serves don't all challenge at once.
    inflight: Mutex<std::collections::HashSet<String>>,
}

/// Track A2-Logprob: prompt-scoring challenges to providers whose model this gateway holds a
/// reference set for (opt-in, `serve --logprob-reference`). A passing round sets the log-prob
/// method flag (→ A2 once the redundant-exec gates are met); a failing one pins the floor.
struct ProviderLogprob {
    refs: Vec<Arc<LoadedReference>>,
    /// A reference registry to ask for round packs when no local reference covers a model.
    registry: Option<crate::registry_client::RegistryClient>,
    /// Identities of registry references seen this run, by reference tag (for pricing).
    registry_identities: Mutex<HashMap<[u8; 16], openhydra_protocol::logprob::ModelIdentity>>,
    /// Weights files of the file-tied registry references seen this run, by reference tag.
    registry_weights: Mutex<HashMap<[u8; 16], String>>,
    /// Per `(peer, model)`: texts issued so far + the round in progress. Persisted (with the status)
    /// in the store's log-prob table and saved before each text is sent, so a restart never re-sends
    /// a text a provider has seen.
    rounds: Mutex<HashMap<String, ChallengeRound>>,
    status: Mutex<HashMap<String, LogprobStatus>>,
    inflight: Mutex<std::collections::HashSet<String>>,
    /// The scorer binary that counts delivered tokens with a reference's tokenizer (P7), when some
    /// reference carries one (`vocab`) and the binary is there.
    count_scorer: Option<std::path::PathBuf>,
    /// Counts running now (bounded by [`crate::token_count::MAX_CONCURRENT_COUNTS`]).
    counts_running: Arc<std::sync::atomic::AtomicUsize>,
    /// Digests of every local reference text this gateway has sent to any provider: each goes to one
    /// provider only, so one that answered it under a throwaway identity can't be sent it again under
    /// a fresh one (review round 3, LP5). Rebuilt on boot from the persisted rounds.
    issued_here: Mutex<std::collections::HashSet<[u8; 32]>>,
    /// Per `(peer, model)`: the registry reference a run was using when it stopped with a text
    /// still pending (busy, a lost connection), so the next run sends that text again instead of
    /// fetching a new pack.
    /// With the pack's id when it is a registry first stage, so its continuation can still be
    /// fetched (review round 4 L4-4). At most [`MAX_RESUMES`].
    resume: Mutex<HashMap<String, (Arc<LoadedReference>, Option<[u8; 16]>, u64)>>,
    /// Per `(peer, model)`: runs in a row that ended without a verdict, which spaces the next ones
    /// out (see [`logprob_retry_ms`]).
    no_verdict_runs: Mutex<HashMap<String, u32>>,
    /// When first-contact rounds were started within the last hour, per provider network ("*": all;
    /// see [`FIRST_CONTACT_PACKS_PER_HOUR`]).
    first_contact_packs: Mutex<HashMap<String, std::collections::VecDeque<u64>>>,
}

impl ProviderLogprob {
    /// Take one of this hour's first-contact rounds for a provider on `network` (its observed network,
    /// or "" when reached through a relay only), if the gateway's ([`FIRST_CONTACT_PACKS_PER_HOUR`])
    /// and that network's ([`FIRST_CONTACT_PER_NETWORK_PER_HOUR`]) allowances both have one left:
    /// throwaway identities on one network can't use up the gateway's (review round 4 L4-6).
    /// A relay-only provider with standing here (its reputation with us at R_MIN, which only serving
    /// us well earns) isn't held to the shared relay allowance, only to the gateway's: throwaway
    /// relay-only identities can't keep pushing back the first checks of the providers that actually
    /// serve us (round 8 L8-5).
    fn spend_first_contact_pack(&self, network: &str, standing: bool, now_ms: u64) -> bool {
        let mut m = self.first_contact_packs.lock().unwrap_or_else(|e| e.into_inner());
        for q in m.values_mut() {
            while q.front().is_some_and(|at| now_ms.saturating_sub(*at) > 60 * 60 * 1000) {
                q.pop_front();
            }
        }
        m.retain(|_, q| !q.is_empty());
        let net = if network.is_empty() { "relay" } else { network };
        let per_net = if network.is_empty() { FIRST_CONTACT_RELAYED_PER_HOUR } else { FIRST_CONTACT_PER_NETWORK_PER_HOUR };
        let net_limited = !(network.is_empty() && standing);
        if m.get("*").map_or(0, |q| q.len()) >= FIRST_CONTACT_PACKS_PER_HOUR
            || (net_limited && m.get(net).map_or(0, |q| q.len()) >= per_net)
        {
            return false;
        }
        m.entry("*".into()).or_default().push_back(now_ms);
        if net_limited {
            m.entry(net.into()).or_default().push_back(now_ms);
        }
        true
    }

    /// Give back a first-contact round taken for `network` (with `standing`, as it was spent) that
    /// never started (no pack came).
    fn refund_first_contact_pack(&self, network: &str, standing: bool) {
        let mut m = self.first_contact_packs.lock().unwrap_or_else(|e| e.into_inner());
        let net = if network.is_empty() { "relay" } else { network };
        let keys: &[&str] = if network.is_empty() && standing { &["*"] } else { &["*", net] };
        for &k in keys {
            if let Some(q) = m.get_mut(k) {
                q.pop_back();
            }
        }
    }

    /// The weights file a reference tag was made from, when it names a file-tied reference this
    /// gateway knows (a local file, or a registry pack seen this run).
    fn tied_weights(&self, tag: [u8; 16]) -> Option<String> {
        self.refs
            .iter()
            .find(|r| reference_tag(&r.set) == tag)
            .and_then(|r| r.set.weights.clone())
            .or_else(|| self.registry_weights.lock().unwrap_or_else(|e| e.into_inner()).get(&tag).cloned())
    }
}

/// Registry packs a gateway spends per hour on providers it has no verdict for yet: throwaway
/// identities that answer the probe and then decline can burn one pack each, and this keeps them from
/// using up the gateway's registry allowance for the providers it already checks (review round 3, LP2).
const FIRST_CONTACT_PACKS_PER_HOUR: usize = 20;
/// …and per provider network, and for all providers reached only through a relay together.
const FIRST_CONTACT_PER_NETWORK_PER_HOUR: usize = 2;
const FIRST_CONTACT_RELAYED_PER_HOUR: usize = 4;
/// Registry runs kept for resuming, at most (each holds its pack).
const MAX_RESUMES: usize = 256;



/// When to try a provider again after `runs` runs in a row ended without a verdict: the usual retry,
/// doubling with each, at most 12 hours.
fn logprob_retry_ms(runs: u32) -> u64 {
    (LOGPROB_RETRY_MS << runs.saturating_sub(1).min(16)).min(12 * 60 * 60 * 1000)
}

/// Per-challenge round-trip budget. One pass is ~1 s on a GPU, but a side-scorer on a laptop CPU
/// takes 25-90 s for a 9B model's text; kept under the transport's own 120 s request limit.
const LOGPROB_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(110);
/// Pause after a busy refusal before the next challenge.
const LOGPROB_BUSY_PAUSE: std::time::Duration = std::time::Duration::from_secs(2);
/// Minimum gap between challenges to one provider. Keeps a round under the provider's per-peer rate
/// limit (2/s) even on a fast engine — a refused challenge still burns its text — and spreads the
/// round out instead of firing it as one burst.
const LOGPROB_PACE: std::time::Duration = std::time::Duration::from_secs(1);

/// Redundant-execution audit sampling bounds (M2.2(b)), fed to
/// [`sample_rate_for_reputation`]. A fully-trusted provider is audited rarely
/// ([`AUDIT_MIN_RATE`]); a fresh / low-reputation one often ([`AUDIT_MAX_RATE`]), since the
/// audit re-runs the work on a second provider and so costs real compute. Tunable.
const AUDIT_MIN_RATE: f64 = 0.02;
/// How long a failed redundant-execution audit holds a provider's authenticity tier at the floor
/// for that model (review T4).
const AUDIT_FAIL_FLOOR_MS: u64 = 24 * 60 * 60 * 1000;
const AUDIT_MAX_RATE: f64 = 0.5;

/// How much of two answers an audit compares: their first this many characters (about 64
/// tokens). Honest greedy runs of one model agree on a long opening; a provider running something
/// else diverges within it. Later divergence between honest engines (accumulation order, kernels)
/// isn't held against anyone.
const AUDIT_COMPARE_CHARS: usize = 256;

/// Serves kept for replay as audits (review round 3, D1), at most, and for how long. In memory
/// only: a prompt kept for an audit is never written to disk, and goes once replayed or stale.
const REPLAY_KEEP: usize = 32;
const REPLAY_TTL_MS: u64 = 60 * 60 * 1000;
/// A replayable serve's prompt is at most this many bytes (larger ones are left alone).
const REPLAY_MAX_PROMPT_BYTES: usize = 64 * 1024;
/// …and its answer is kept up to this many characters, and must reach [`MIN_REPLAY_ANSWER_CHARS`]
/// to be worth comparing.
const REPLAY_KEEP_CHARS: usize = AUDIT_COMPARE_CHARS;
const MIN_REPLAY_ANSWER_CHARS: usize = 64;

/// The longest answer asked of a probe-question writer (about 8 short questions).
const WRITER_MAX_TOKENS: u32 = 400;
/// Finished probe prompts remembered as sent (forgotten all at once past this; the space of
/// finished prompts is far larger, so a repeat after that is still unlikely).
const MAX_PROBES_REMEMBERED: usize = 8192;
/// The most of a writer's answer read (well past [`WRITER_MAX_TOKENS`] of text): a writer that
/// streams on can't grow our memory.
const WRITER_MAX_TEXT_BYTES: usize = 16 * 1024;
/// How long a model's written probe questions are used before fresh ones are written.
const PROBE_POOL_TTL_MS: u64 = 24 * 60 * 60 * 1000;

/// How long a provider an audit outvoted carries a strike (review round 4 C3): outvoted again on
/// another prompt within it, it is failed; agreeing on one first clears it.
const STRIKE_WINDOW_MS: u64 = 24 * 60 * 60 * 1000;
/// The follow-up audit of a provider with a strike runs after a random wait in this range (ms), on
/// a different prompt, so nothing about it says which request it is.
const FOLLOW_UP_DELAY_MS: std::ops::Range<u64> = 60_000..600_000;
/// Strikes remembered at once (the oldest are forgotten first).
const MAX_STRIKES: usize = 4096;
/// Audits running at once, across all providers (each holds a thread and pays for its serves); past
/// it a sampled audit is skipped (review round 4 M6).
const MAX_AUDITS_IN_FLIGHT: usize = 4;
/// Follow-up audits waiting or running at once, and how many times one tries for a slot (a minute
/// apart) before giving up (R5T-2).
const MAX_FOLLOW_UPS: usize = 64;
const FOLLOW_UP_TRIES: usize = 10;
/// The longest an audit waits for one provider's answer (a stream dripping output can't hold an
/// audit's slot indefinitely, R5T-5); a user's client may stop reading too, so it stands out no more.
const AUDIT_SERVE_MAX: Duration = Duration::from_secs(300);
/// Qualifying audits (H2) per model per hour at most, and one running at a time: a crowd of providers
/// below the floor can't take every audit slot or run up the gateway's costs (R5T-5).
const QUALIFY_PER_MODEL_PER_HOUR: usize = 6;
/// The most providers one audit sends its request to, the audited one included (what the replay
/// disclosure promises).
const MAX_AUDIT_PROVIDERS: usize = 4;

/// Under a hard authenticity floor, the share of routed requests that also start a qualifying audit
/// of a provider the floor left out, and how often one provider is audited that way at most: a
/// provider below the floor gets no traffic to earn standing from, so audits are its way up
/// (review round 4 H2).
const QUALIFY_SAMPLE: f64 = 0.2;
const QUALIFY_INTERVAL_MS: u64 = 30 * 60 * 1000;
/// Providers whose last qualifying audit is remembered at once.
const MAX_QUALIFY_RECORDS: usize = 4096;

/// A model's questions written for probes (P1), and who wrote them: a writer that comes to serve
/// the model is no longer used for it (it would recognise its questions, review round 4 H6).
#[derive(Debug, Clone)]
struct ProbePool {
    writer: String,
    filled_at: u64,
    questions: std::collections::VecDeque<String>,
}

/// An audit running ([`ConsumerNode::audit_slot`]); its slot frees when dropped.
struct AuditSlot(Arc<std::sync::atomic::AtomicUsize>);

impl Drop for AuditSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Whether an agreeing subject's open strike is cleared (C3, R5T-2, R6-2, R7-3, round 8 T8-4): by a
/// replay of its own traffic (`replayed`), or by a follow-up (`strict`) on written questions, unless
/// those came from the outside writer whose questions gave the strike (`struck_by`); our own
/// provider's questions (`writer_is_own`) always count.
fn strike_cleared_by(replayed: bool, strict: bool, probe_writer: Option<&str>, writer_is_own: bool, struck_by: Option<&str>) -> bool {
    replayed || (strict && probe_writer.is_some_and(|w| writer_is_own || Some(w) != struck_by))
}

/// How long a provider that served us with standing stays a candidate probe writer.
const WRITER_CANDIDATE_TTL_MS: u64 = 7 * 24 * 60 * 60 * 1000;
/// Candidate probe writers remembered (the oldest go first).
const MAX_WRITER_CANDIDATES: usize = 256;
/// Candidate writers tried per probe pool written.
const MAX_WRITERS_TRIED: usize = 3;
/// After a search for an outside writer found none, how long until the next one (templates meanwhile).
const NO_WRITER_RETRY_MS: u64 = 10 * 60 * 1000;

/// Whether a provider whose reputation with us is `earned` may write probe questions: only serving
/// us well raises a reputation to [`openhydra_protocol::award::R_MIN`], so a provider we don't know
/// can't (round 8 T8-1).
fn has_writer_standing(earned: Option<f64>) -> bool {
    earned.is_some_and(|r| r >= openhydra_protocol::award::R_MIN)
}

/// An open strike (C3): when it was given, and the writer of the probe that gave it, when that was a
/// provider other than ours (whose questions then can neither confirm nor clear it, round 8 T8-1/T8-4).
#[derive(Debug, Clone, PartialEq)]
struct Strike {
    at: u64,
    writer: Option<String>,
    /// Follow-ups that couldn't hear its subject so far.
    retries: u32,
}

/// `(peer, model)` pairs whose reputation, or audit history, this consumer keeps (round 8 T8-8).
const MAX_TRACKED_PAIRS: usize = 8192;
/// A pair untouched this long has decayed to nothing: the first to go when the cap is reached.
const STALE_PAIR_MS: u64 = 30 * 24 * 60 * 60 * 1000;

/// Make room in `map` for a new pair once it holds [`MAX_TRACKED_PAIRS`]: every pair untouched for
/// [`STALE_PAIR_MS`] (by `last`), else the least recently touched one not `kept`. Returns the keys
/// removed (to forget them on disk too).
fn evict_for_new_pair<V>(map: &mut HashMap<String, V>, last: impl Fn(&V) -> u64, kept: impl Fn(&V) -> bool, now: u64) -> Vec<String> {
    if map.len() < MAX_TRACKED_PAIRS {
        return Vec::new();
    }
    let mut gone: Vec<String> =
        map.iter().filter(|(_, v)| now.saturating_sub(last(v)) > STALE_PAIR_MS && !kept(v)).map(|(k, _)| k.clone()).collect();
    if gone.is_empty() {
        gone.extend(map.iter().filter(|(_, v)| !kept(v)).min_by_key(|(_, v)| last(v)).map(|(k, _)| k.clone()));
    }
    for k in &gone {
        map.remove(k);
    }
    gone
}

/// A model's recent requests a probe takes its shape from (at most, by count).
const TRAFFIC_WINDOW: u32 = 64;
/// The share of a model's recent requests that must be greedy (temperature 0) before this consumer
/// sends greedy probes for it: a probe at temperature 0 from a consumer that never asks for it would
/// stand out (and comparing answers needs greedy decoding).
const MIN_GREEDY_SHARE: f64 = 0.1;

// Crypto-rail settlement peg (USDC micro-units per credit unit): **opt-in, no numeric default**
// (review F3). A node stamps a crypto-rail cumulative voucher on its receipts ONLY when a peg is
// explicitly configured (`OPENHYDRA_USDC_MICROS_PER_CREDIT` → [`ConsumerNode::with_settlement_peg`]);
// unconfigured, it stamps no crypto amount (a legacy receipt) rather than mis-denominated ~1-µUSDC
// vouchers. The real $/credit peg is a CREDIT_TIERING business parameter (ideally a signed receipt
// term) — set it via config before any real settlement.

/// Pick the best provider from `peers` for a request whose canonical id is
/// `request_canonical` (use `""` to match any provider of the discovered model_id).
///
/// Filters by canonical-id compatibility (a provider that advertised none is kept —
/// backward-compatible), ranks the rest with the shared router scoring
/// ([`rank_peers`], reputation-aware), and returns the top. `None` if nothing compatible.
pub fn select_provider(
    peers: &[DiscoveredPeer],
    request_canonical: &str,
    tier: u8,
) -> Option<SelectedProvider> {
    rank_providers(peers, request_canonical, tier).into_iter().next()
}

/// All compatible providers for the request, **in preference order** (best first).
///
/// Same filter + [`rank_peers`] scoring as [`select_provider`], but returns the whole
/// ranked list so the caller can fail over to the next provider when one is dead or
/// unreachable (discovery can surface stale-but-still-advertised providers).
///
/// Every provider at neutral reputation (a record's own `reputation_score` isn't signed, so it is
/// never read). For **earned** reputation from the consumer's local [`ReputationTracker`], use
/// [`rank_providers_with_reputation`].
pub fn rank_providers(
    peers: &[DiscoveredPeer],
    request_canonical: &str,
    tier: u8,
) -> Vec<SelectedProvider> {
    rank_providers_with_reputation(peers, request_canonical, tier, &|_, _| None)
}

/// The model half of a per-`(peer, model)` reputation key for a provider (credit-tiering
/// Phase 4a): the raw engine **handle** (`DiscoveredPeer.model_id`). We key on the handle, not the
/// self-declared canonical, because (a) the handle is the commitment-bound identity the audit
/// actually exercises, and (b) it matches `KnownProvider.model_id` so the desktop can join
/// reputation to a provider row. Using one derivation everywhere (serve + audit + settle) keeps a
/// served outcome and a later lookup in the same reputation bucket. The `canonical` argument is
/// retained for call-site symmetry but no longer selected (an untrusted label must not partition
/// trust differently from what was served).
fn rep_model_key<'a>(_canonical_model_id: &'a str, model_id: &'a str) -> &'a str {
    model_id
}

/// As [`rank_providers`], but `earned_reputation(peer_id, model_key)` may override a provider's
/// reputation with the consumer's locally-earned score for **that model** (M2.2(a) + Phase 4a
/// per-`(peer, model)` keying). `Some(score)` wins; `None` falls back to the self-reported DHT
/// value (neutral when absent). This is the closed loop that downranks a provider that has
/// misbehaved with *this* consumer *on this model* — trust on another model does not carry over.
pub fn rank_providers_with_reputation(
    peers: &[DiscoveredPeer],
    request_canonical: &str,
    tier: u8,
    earned_reputation: &dyn Fn(&str, &str) -> Option<f64>,
) -> Vec<SelectedProvider> {
    // No buyer assurance policy → the open policy (admits everything), so the authenticity
    // resolver is never consulted and ranking is unchanged.
    rank_providers_with_assurance(
        peers,
        request_canonical,
        tier,
        earned_reputation,
        &|_, _| AuthenticityTier::SelfReported,
        &AssurancePolicy::OPEN,
    )
}

/// As [`rank_providers_with_reputation`], but a buyer's [`AssurancePolicy`] gates the candidate set
/// (UNIFIED_TRUST_STACK Increment 3). For each canonical-compatible provider, the caller's
/// `achieved_authenticity(peer_id, model_key)` resolves the tier this consumer has established for
/// the `(peer, model)` from its own redundant-exec audit history (the tier is never the provider's
/// word). The gate then depends on the policy's [`PolicyMode`]:
///
/// * [`HardGate`](PolicyMode::HardGate) — a provider whose achieved assurance does not clear the
///   policy's floors is **dropped** before scoring (AntSeed's `--require-verifier`). If none clear
///   it, routing is empty and the serve fails over to nothing (fail-closed) — the honest outcome
///   when a buyer demands a tier no reachable provider has proven.
/// * [`VerifyButRoute`](PolicyMode::VerifyButRoute) — every compatible provider is kept and ranked
///   as before; the achieved tier is surfaced (see `economy_snapshot`), not gated on.
///
/// Privacy and Settlement are resolved at their disclosed baseline (`P0`/`S0`) here — their
/// verifiers are not built yet — so a policy that requires a privacy/settlement floor above the
/// baseline admits nobody under `HardGate` (fail-closed, honest) until those axes are verifiable.
pub fn rank_providers_with_assurance(
    peers: &[DiscoveredPeer],
    request_canonical: &str,
    tier: u8,
    earned_reputation: &dyn Fn(&str, &str) -> Option<f64>,
    achieved_authenticity: &dyn Fn(&str, &str) -> AuthenticityTier,
    policy: &AssurancePolicy,
) -> Vec<SelectedProvider> {
    let compatible: Vec<&DiscoveredPeer> = peers
        .iter()
        .filter(|p| {
            let canonical_ok = request_canonical.is_empty()
                || p.canonical_model_id.is_empty()
                || is_compatible(request_canonical, &p.canonical_model_id);
            if !canonical_ok {
                return false;
            }
            // Buyer assurance gate. Only HardGate drops candidates; VerifyButRoute keeps them
            // (the achieved tier is surfaced, not used to route). Privacy/Settlement are at their
            // disclosed baseline until their verifiers exist.
            match policy.mode {
                PolicyMode::VerifyButRoute => true,
                PolicyMode::HardGate => {
                    let achieved = AssuranceTier {
                        authenticity: achieved_authenticity(
                            &provider_state_id(&p.peer_id, &p.public_key),
                            rep_model_key(&p.canonical_model_id, &p.model_id),
                        ),
                        privacy: PrivacyTier::None,
                        settlement: SettlementTier::CreditsOnly,
                    };
                    policy.admits(&achieved)
                }
            }
        })
        .collect();
    if compatible.is_empty() {
        return Vec::new();
    }
    let inputs: Vec<PeerScoreInput> = compatible
        .iter()
        .map(|p| PeerScoreInput {
            // Ranked (and matched back) by the 128-bit id: two providers sharing a 64-bit id stay apart.
            peer_id: provider_state_id(&p.peer_id, &p.public_key),
            latency_ms: 1.0, // RTT survey deferred (matches the M1.3 router default)
            load_pct: p.load_pct,
            // Earned reputation (local tracker, per (peer, model)) wins; else neutral. Never the
            // record's own `reputation_score`: it isn't signed, so any provider could claim the top
            // and take the audit slots (review T2). A provider that failed *us* on this model ranks
            // below an untried one.
            reputation: earned_reputation(&provider_state_id(&p.peer_id, &p.public_key), rep_model_key(&p.canonical_model_id, &p.model_id))
                .unwrap_or(50.0),
            bandwidth_mbps: 0.0,
            s2s_rtt_ms: 0.0,
            throughput_tok_s: p.throughput_tok_s,
            queue_depth: p.queue_depth,
        })
        .collect();
    let mut ranked: Vec<SelectedProvider> = rank_peers(&inputs, tier)
        .into_iter()
        .filter_map(|ranked| {
            compatible
                .iter()
                .find(|p| provider_state_id(&p.peer_id, &p.public_key) == ranked.peer_id)
                .map(|p| SelectedProvider {
                    libp2p_peer_id: p.libp2p_peer_id.clone(),
                    peer_id: p.peer_id.clone(),
                    state_id: ranked.peer_id.clone(),
                    public_key: p.public_key.clone(),
                    model_id: p.model_id.clone(),
                    canonical_model_id: p.canonical_model_id.clone(),
                    swarm_public_key: p.swarm_public_key.clone(),
                })
        })
        .collect();

    // R-DHT-8: prefer providers we already have a live connection to. The
    // capability/reputation score (above) decides order *within* the connected
    // and disconnected groups; this stable partition just floats live providers
    // to the front so a stale-but-still-advertised record is only tried after
    // every connected provider — failover as the exception, not the rule.
    let connected: std::collections::HashSet<String> = peers
        .iter()
        .filter(|p| p.connected)
        .map(|p| provider_state_id(&p.peer_id, &p.public_key))
        .collect();
    ranked.sort_by_key(|sp| !connected.contains(&sp.state_id));
    ranked
}

/// Send `request` over `transport`, stream each delta to `on_delta`, and return the
/// [`ServeSummary`] (token count for the receipt) on a clean `Done`.
///
/// `transport(framed_request) -> response_bytes` is the network round-trip — live, a
/// `proxy_forward` to the chosen provider. Errors if the transport fails, the provider
/// returns an `Error` frame, or the stream ends without a `Done`.
pub fn request_completion(
    transport: &mut dyn FnMut(&[u8]) -> Result<Vec<u8>, AdapterError>,
    request: &ServeRequest,
    on_delta: &mut dyn FnMut(&str),
) -> Result<ServeSummary, AdapterError> {
    let mut framed = Vec::with_capacity(1 + request.messages.len() * 32);
    framed.push(SERVE_REQUEST);
    framed.extend_from_slice(&request.encode());

    let response = transport(&framed)?;

    // Tool calls arrive in their own terminal frame just before `Done`; hold them so the
    // `Done` arm can attach them to the summary the gateway turns into the assistant turn.
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    for (i, chunk) in parse_response(&response)?.into_iter().enumerate() {
        match chunk {
            // A refusal is its reply's only frame (B3).
            ServeChunk::Error(msg) => return Err(provider_error(msg, i == 0)),
            ServeChunk::Delta(text) => on_delta(&text),
            ServeChunk::ToolCalls(tc) => tool_calls = tc,
            ServeChunk::Done { tokens, metrics } => {
                return Ok(ServeSummary {
                    tokens,
                    ok: true,
                    metrics,
                    discover_ns: 0,
                    proxy_roundtrip_ns: 0,
                    tool_calls,
                    authenticity: None,
                })
            }
        }
    }
    Err(AdapterError::Parse(
        "serve response ended without a Done/Error frame".into(),
    ))
}

// ── P1c: streaming serve client ─────────────────────────────────────────────────────────
//
// The buffered `request_completion` above is one round-trip: the whole completion returns in a
// single blob under a total-time cap, so a token doesn't reach the client until the last token is
// generated. The streaming client opens a stream (`SERVE_STREAM 0x13`), then polls
// (`FETCH_CHUNKS 0x14`) with an adaptive interval, delivering each delta the instant it lands —
// turning the total-time cap into a *stall* cap (reset on every new token) and making drop-resume
// (re-poll the same offset) the steady state. A legacy provider that doesn't know `0x13` answers
// with a blob that fails to decode as a `FetchChunksResponse`, which the caller detects and falls
// back to `request_completion`. See `STREAMING_SERVE_P1_PLAN.md`.

/// Poll cadence: tight while tokens flow (`POLL_MIN`), backed off on idle (`POLL_MAX`). Bounds
/// relay poll traffic without adding much token latency on an active stream.
const POLL_MIN: Duration = Duration::from_millis(50);
const POLL_MAX: Duration = Duration::from_millis(750);
/// Give up if no new chunk arrives for this much *accumulated idle time* — the "provider wedged"
/// signal. Reset on every progress, so total generation is unbounded as long as tokens flow.
const STREAM_STALL_TIMEOUT: Duration = Duration::from_secs(120);
/// Bounded consecutive transport-drop retries at the same offset before giving up (drop-resume).
const STREAM_DROP_RETRIES: u32 = 5;
/// Backoff between drop-resume re-polls, tuned to the ~10s relay re-establish window.
const STREAM_DROP_BACKOFF: Duration = Duration::from_secs(2);
/// Per-round-trip timeout for a streaming call (the open ack or one poll). Each is a quick
/// provider-side buffer op (the provider acks the open *before* generating), so this is short —
/// a dead provider fails over fast, while a live stream's long generation is governed by the
/// per-token stall clock, not this. (The buffered fallback still uses the `max_tokens`-scaled
/// [`attempt_timeout`].)
const STREAM_CALL_TIMEOUT: Duration = Duration::from_secs(30);

// F2: absolute total-time ceiling on a streaming attempt — a defense-in-depth backstop against a
// malicious/degenerate provider that dribbles one token just under the stall deadline forever,
// pinning the gateway's blocking thread indefinitely (thousands → tokio blocking-pool exhaustion).
// The 120s idle-stall (above) is the primary bound; this only ever bites the pathological
// slow-but-never-idle dribble. Sized to be **provably larger than any legitimate serve** so it can
// never abort real work: scaled by `max_tokens` (a large output on a slow remote model is legit and
// can run for a long time). Uses a real wall clock (not the accumulated-sleep stall clock, which
// under-counts slow round-trips); test-safe because the deterministic tests finish in microseconds.
const STREAM_TOTAL_BASE: Duration = Duration::from_secs(300);
/// Worst-case seconds-per-token allowance (~0.5 tok/s — far slower than any real provider).
const STREAM_TOTAL_PER_TOKEN_SECS: u64 = 2;
const STREAM_TOTAL_FLOOR: Duration = Duration::from_secs(30 * 60);
const STREAM_TOTAL_CEILING: Duration = Duration::from_secs(6 * 60 * 60);
/// Assumed output length when a request omits `max_tokens` (budgeting only).
const STREAM_TOTAL_DEFAULT_TOKENS: u64 = 1024;

/// The absolute total-time ceiling for a streaming attempt of `max_tokens`. Honors an
/// `OPENHYDRA_STREAM_MAX_TOTAL_SECS` ops override (floored — see [`stream_total_cap_override`]);
/// otherwise `BASE + max_tokens · per_token`, clamped to `[FLOOR, CEILING]`. Deliberately
/// generous — it is a DoS backstop, not a latency SLA.
fn stream_total_cap(max_tokens: Option<u32>) -> Duration {
    if let Some(secs) = std::env::var("OPENHYDRA_STREAM_MAX_TOTAL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        return stream_total_cap_override(secs);
    }
    stream_total_cap_scaled(max_tokens)
}

/// Turn the raw `OPENHYDRA_STREAM_MAX_TOTAL_SECS` seconds into a usable cap. The override exists so
/// an operator can bound the DoS window *tighter* than the 30-min scaled floor, but a value below
/// the idle-stall ([`STREAM_STALL_TIMEOUT`]) is nonsensical — the total cap would fire before even
/// one stall window elapses, aborting every legitimate stream (a fat-fingered `=0` kills all
/// serving outright). So floor it at the stall timeout and warn when the raw value had to be
/// raised, keeping the knob useful without the silent all-streams-down footgun. Pure (no env read)
/// so the flooring is unit-tested directly.
fn stream_total_cap_override(secs: u64) -> Duration {
    let floor = STREAM_STALL_TIMEOUT;
    let requested = Duration::from_secs(secs);
    if requested < floor {
        tracing::warn!(
            requested_secs = secs,
            floored_secs = floor.as_secs(),
            "OPENHYDRA_STREAM_MAX_TOTAL_SECS below the idle-stall floor — raising it (a cap under \
             the stall window would abort every stream)"
        );
        floor
    } else {
        requested
    }
}

/// The pure `max_tokens`-scaled cap (no env), split out so the scaling/clamp math is unit-tested
/// without touching the process-global override env var (which would race the parallel test runner).
fn stream_total_cap_scaled(max_tokens: Option<u32>) -> Duration {
    let tokens = max_tokens.map(u64::from).unwrap_or(STREAM_TOTAL_DEFAULT_TOKENS);
    let scaled = STREAM_TOTAL_BASE + Duration::from_secs(tokens.saturating_mul(STREAM_TOTAL_PER_TOKEN_SECS));
    scaled.clamp(STREAM_TOTAL_FLOOR, STREAM_TOTAL_CEILING)
}

/// The assurance policy one request is routed under (D2): the node's (`base`), with its
/// authenticity floor raised to `requested` when the request asks for more, and then enforced
/// (hard gate) whatever the node's mode: a client that names a floor means it. A request never
/// lowers the node's floor or relaxes its mode.
pub fn request_policy(base: AssurancePolicy, requested: Option<AuthenticityTier>) -> AssurancePolicy {
    match requested {
        Some(t) if t > AuthenticityTier::SelfReported => AssurancePolicy {
            min_authenticity: base.min_authenticity.max(t),
            mode: PolicyMode::HardGate,
            ..base
        },
        _ => base,
    }
}

/// Why `model` has no provider to route to: none found at all, or (a buyer's floor that left
/// nobody, D2) none verified at the policy's tier, said so with how many were below it.
fn no_route_error(model: &str, policy: &AssurancePolicy, peers: &[DiscoveredPeer], tier: u8) -> AdapterError {
    if policy.mode == PolicyMode::HardGate && policy.min_authenticity > AuthenticityTier::SelfReported {
        let below =
            rank_providers_with_assurance(peers, "", tier, &|_, _| None, &|_, _| AuthenticityTier::SelfReported, &AssurancePolicy::OPEN)
                .len();
        if below > 0 {
            return AdapterError::Route(format!(
                "no provider of '{model}' is verified at {} or above ({below} found below it)",
                policy.min_authenticity.label()
            ));
        }
    }
    AdapterError::Route(format!("no provider for model '{model}'"))
}

/// One request's trip to a provider ([`ConsumerNode::exchange`]).
struct Exchange {
    result: Result<ServeSummary, AdapterError>,
    /// The first-ack (stream) or round-trip (buffered) time, ns.
    proxy_roundtrip_ns: u64,
    /// Whether the provider opened the stream (acknowledged it, not refused it): only then is there a
    /// stream to close (review round 5, R5S-3).
    opened: bool,
    /// How far the stream was read (its next poll offset), for closing it.
    stream_offset: u32,
    /// Tool calls a stream delivered before it broke (counted when it is settled, R5S-5).
    tool_calls: Vec<ToolCall>,
    /// The nonce the serve ended under: the request's, or a fresh one if it had to be served again
    /// (R5S-4). Its receipt carries this one.
    nonce: [u8; 16],
}

/// How far a stream got: what a close needs if it ends early.
#[derive(Debug, Default)]
struct StreamProgress {
    /// The offset read up to (the next poll's).
    offset: u32,
    /// The provider acknowledged the open (a stream exists to close).
    opened: bool,
    /// Tool calls received so far.
    tool_calls: Vec<ToolCall>,
}

/// Append `text` to `kept`, up to `max_chars` characters in all.
fn keep_prefix(kept: &mut String, text: &str, max_chars: usize) {
    let room = max_chars.saturating_sub(kept.chars().count());
    kept.extend(text.chars().take(room));
}

/// What an audit asks a provider for: a request as a user would send it, always greedy.
#[derive(Debug, Clone, PartialEq)]
struct AuditRequest {
    messages: Vec<ChatMessage>,
    max_tokens: Option<u32>,
    tools: Vec<Value>,
    think: Option<bool>,
}

/// A serve this consumer could replay as an audit (D1): a user's greedy request to a public model,
/// and the start of the answer its provider gave.
#[derive(Debug, Clone)]
struct Replayable {
    provider: SelectedProvider,
    /// The model the user asked for (what discovery finds other providers of).
    model: String,
    request: AuditRequest,
    /// The start of the answer, up to [`REPLAY_KEEP_CHARS`].
    answer: String,
    at: u64,
}

/// Keep `entry` for replay: within [`REPLAY_KEEP`] (the oldest go first) and [`REPLAY_TTL_MS`].
fn keep_replayable(kept: &mut Vec<Replayable>, entry: Replayable, now: u64) {
    kept.retain(|r| now.saturating_sub(r.at) <= REPLAY_TTL_MS);
    if kept.len() >= REPLAY_KEEP {
        if let Some(i) = kept.iter().enumerate().min_by_key(|(_, r)| r.at).map(|(i, _)| i) {
            kept.swap_remove(i);
        }
    }
    kept.push(entry);
}

/// Take one kept serve of `provider_state_id` for `model`, at random among its fresh ones (it is
/// replayed once).
fn take_replayable(kept: &mut Vec<Replayable>, provider_state_id: &str, model: &str, now: u64) -> Option<Replayable> {
    use rand::seq::IteratorRandom;
    kept.retain(|r| now.saturating_sub(r.at) <= REPLAY_TTL_MS);
    let i = kept
        .iter()
        .enumerate()
        .filter(|(_, r)| r.provider.state_id == provider_state_id && r.model == model)
        .map(|(i, _)| i)
        .choose(&mut rand::thread_rng())?;
    Some(kept.swap_remove(i))
}

/// The shape of a model's recent requests from this consumer, which a probe copies (D1): how many
/// were greedy, used tools, or set thinking, and the last `max_tokens` asked for.
#[derive(Debug, Clone, Default, PartialEq)]
struct TrafficShape {
    requests: u32,
    greedy: u32,
    think_on: u32,
    think_off: u32,
    /// Requests that opened with a system prompt.
    with_system: u32,
    max_tokens: Option<u32>,
    /// The `max_tokens` of the latest greedy requests (at most [`GREEDY_LENGTHS_KEPT`]): a probe asks
    /// for one of them, as a request like it would (round 8 T8-2).
    greedy_max_tokens: std::collections::VecDeque<Option<u32>>,
    /// The latest request's system prompt, kept only where audits may replay requests anyway (a
    /// probe then carries the same one a real request does); else probes use a generic one.
    system: Option<String>,
}

/// How many recent greedy requests' `max_tokens` a [`TrafficShape`] keeps.
const GREEDY_LENGTHS_KEPT: usize = 16;
/// Requests a model's traffic needs before a follow-up or qualifying audit takes its shape.
const MIN_SHAPED_REQUESTS: u32 = 3;

impl TrafficShape {
    fn note(&mut self, temperature: Option<f64>, think: Option<bool>, max_tokens: Option<u32>, system: Option<&str>, keep_system: bool) {
        // A sliding window: past it, older requests weigh half.
        if self.requests >= TRAFFIC_WINDOW {
            self.requests /= 2;
            self.greedy /= 2;
            self.think_on /= 2;
            self.think_off /= 2;
            self.with_system /= 2;
        }
        self.requests += 1;
        self.greedy += u32::from(temperature == Some(0.0));
        if temperature == Some(0.0) {
            if self.greedy_max_tokens.len() >= GREEDY_LENGTHS_KEPT {
                self.greedy_max_tokens.pop_front();
            }
            self.greedy_max_tokens.push_back(max_tokens);
        }
        match think {
            Some(true) => self.think_on += 1,
            Some(false) => self.think_off += 1,
            None => {}
        }
        self.max_tokens = max_tokens;
        if let Some(sys) = system {
            self.with_system += 1;
            if keep_system && sys.len() <= REPLAY_MAX_PROMPT_BYTES {
                self.system = Some(sys.to_string());
            }
        }
    }

    /// A shape for a model whose traffic gives none to copy (a qualifying or follow-up audit): a
    /// greedy request of a common length, with or without a generic system prompt, varied so these
    /// probes don't share one fingerprint (R5T-5).
    fn plain<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        use rand::seq::SliceRandom;
        // Always bounded: an unbounded one could run past an audit's time limit on a slow rig (R6-3).
        let max_tokens = *[Some(256), Some(512), Some(768), Some(1024)].choose(rng).expect("non-empty");
        TrafficShape { requests: 2, greedy: 2, with_system: if rng.gen_bool(0.5) { 2 } else { 0 }, max_tokens, ..Default::default() }
    }

    /// A probe of `prompt` shaped like these requests, or `None` when too few of them are greedy to
    /// send a greedy probe.
    fn probe(&self, prompt: String) -> Option<AuditRequest> {
        if self.greedy == 0 || f64::from(self.greedy) < MIN_GREEDY_SHARE * f64::from(self.requests) {
            return None;
        }
        let unset = self.requests - self.think_on - self.think_off;
        let think = if self.think_on > self.think_off.max(unset) {
            Some(true)
        } else if self.think_off > self.think_on.max(unset) {
            Some(false)
        } else {
            None
        };
        // Opened with a system prompt as often as real requests are (review round 4 M8): the real
        // one where it was kept, else a generic one.
        let mut messages = Vec::with_capacity(2);
        if self.with_system * 2 > self.requests {
            let system = self
                .system
                .clone()
                .unwrap_or_else(|| crate::audit_prompts::generic_system_prompt(&mut rand::thread_rng()).to_string());
            messages.push(ChatMessage { role: "system".into(), content: system, ..Default::default() });
        }
        messages.push(ChatMessage { role: "user".into(), content: prompt, ..Default::default() });
        let max_tokens = {
            use rand::seq::SliceRandom;
            let kept: Vec<Option<u32>> = self.greedy_max_tokens.iter().copied().collect();
            kept.choose(&mut rand::thread_rng()).copied().unwrap_or(self.max_tokens)
        };
        Some(AuditRequest { messages, max_tokens, tools: Vec::new(), think })
    }
}

/// Hold back the failures that are first strikes (review round 4 C3). A provider an audit outvoted
/// (`Failed`) with no strike within [`STRIKE_WINDOW_MS`] (`strikes`, by reputation key) isn't failed
/// yet: it gets a strike, and is returned among those to audit again, on another prompt, later. One
/// already struck is failed, and its strike is spent. A provider that agreed clears its strike: one
/// divergent answer can be a fluke (seen live, among honest providers on one engine), but a provider
/// serving another model diverges every time. The same request is never sent twice, so a provider
/// can't tell the follow-up from any request (re-asking the same one told it).
fn strike_first_failures(
    mut report: AuditReport,
    key_of: &dyn Fn(&str) -> String,
    clears: &dyn Fn(&str, Option<&str>) -> bool,
    writer: Option<&str>,
    strikes: &mut HashMap<String, Strike>,
    now: u64,
) -> (AuditReport, Vec<String>) {
    let mut follow_up = Vec::new();
    report.outcomes.retain(|(peer, outcome)| {
        let key = key_of(peer);
        match outcome {
            VerificationOutcome::Failed => {
                let open = strikes.get(&key).filter(|s| now.saturating_sub(s.at) <= STRIKE_WINDOW_MS).cloned();
                match open {
                    // Outvoted again on the same outside writer's questions: that writer alone can't
                    // fail a provider (it could have written questions honest engines disagree on);
                    // audited again on something else.
                    Some(s) if s.writer.is_some() && s.writer.as_deref() == writer => {
                        follow_up.push(peer.clone());
                        false
                    }
                    Some(_) => {
                        strikes.remove(&key);
                        true
                    }
                    None => {
                        if strikes.len() >= MAX_STRIKES {
                            strikes.retain(|_, s| now.saturating_sub(s.at) <= STRIKE_WINDOW_MS);
                            if strikes.len() >= MAX_STRIKES {
                                if let Some(oldest) = strikes.iter().min_by_key(|(_, s)| s.at).map(|(k, _)| k.clone()) {
                                    strikes.remove(&oldest);
                                }
                            }
                        }
                        strikes.insert(key, Strike { at: now, writer: writer.map(str::to_string), retries: 0 });
                        follow_up.push(peer.clone());
                        false
                    }
                }
            }
            // Only agreeing on real traffic, as the one audited, clears a strike (R5T-2): a liar that
            // lies on some requests would otherwise clear it on the next probe it answers honestly.
            VerificationOutcome::Honored => {
                if strikes.get(&key).is_some_and(|s| clears(peer, s.writer.as_deref())) {
                    strikes.remove(&key);
                }
                true
            }
            VerificationOutcome::Rejected => true,
        }
    });
    (report, follow_up)
}

/// Resolve an audit in which some providers are of unknown network (reached only through a relay):
/// two of those could be one operator, so they are judged only against the providers on known
/// networks, at least two of which must have answered and a strict majority of them agreed (R5T-3,
/// R5T-4). Known providers outside that majority fail; unknown ones fail unless they agree with it.
/// `None` when the known providers can't decide (fewer than two answered, or no majority).
fn resolve_against_known(
    results: &[(String, Result<String, String>)],
    nets_of: &HashMap<String, Vec<String>>,
    subject: &str,
) -> Option<AuditReport> {
    let known = |p: &str| nets_of.get(p).is_some_and(|n| !n.is_empty());
    let known_results: Vec<(String, Result<String, String>)> = results.iter().filter(|(p, _)| known(p)).cloned().collect();
    let known_ok = known_results.iter().filter(|(_, r)| r.is_ok()).count();
    // One known answer, not the subject's (R7-2: a relay-only partner could be its own second node):
    // everyone may pass by agreeing with it, but no one fails on it alone (R6-4).
    if known_ok == 1 && known_results.iter().any(|(p, r)| p != subject && r.is_ok()) {
        let answers: Vec<&String> = results.iter().filter_map(|(_, r)| r.as_ref().ok()).collect();
        let all_agree = answers.len() >= 2 && answers.iter().all(|a| openhydra_protocol::verify::agrees(answers[0], a));
        if !all_agree {
            return None;
        }
        let mut outcomes = Vec::new();
        let mut non_responders = Vec::new();
        for (peer, r) in results {
            match r {
                Ok(_) => outcomes.push((peer.clone(), VerificationOutcome::Honored)),
                Err(_) => {
                    outcomes.push((peer.clone(), VerificationOutcome::Rejected));
                    non_responders.push(peer.clone());
                }
            }
        }
        return Some(AuditReport { verdict: RedundantVerdict::Agree, outcomes, non_responders });
    }
    if known_ok < 2 {
        return None;
    }
    let base = resolve_audit(&known_results);
    if base.verdict == RedundantVerdict::Inconclusive {
        return None;
    }
    let reference = base
        .outcomes
        .iter()
        .find(|(_, o)| *o == VerificationOutcome::Honored)
        .and_then(|(p, _)| known_results.iter().find(|(q, _)| q == p))
        .and_then(|(_, r)| r.as_ref().ok())?
        .clone();
    let mut outcomes = base.outcomes.clone();
    let mut non_responders = base.non_responders.clone();
    for (peer, result) in results.iter().filter(|(p, _)| !known(p)) {
        match result {
            Ok(answer) => {
                let o = if openhydra_protocol::verify::agrees(&reference, answer) {
                    VerificationOutcome::Honored
                } else {
                    VerificationOutcome::Failed
                };
                outcomes.push((peer.clone(), o));
            }
            Err(_) => {
                outcomes.push((peer.clone(), VerificationOutcome::Rejected));
                non_responders.push(peer.clone());
            }
        }
    }
    let failed: Vec<usize> = results
        .iter()
        .filter(|(_, r)| r.is_ok())
        .enumerate()
        .filter(|(_, (p, _))| outcomes.iter().any(|(q, o)| q == p && *o == VerificationOutcome::Failed))
        .map(|(i, _)| i)
        .collect();
    let verdict = if failed.is_empty() { RedundantVerdict::Agree } else { RedundantVerdict::Outliers(failed) };
    Some(AuditReport { verdict, outcomes, non_responders })
}

/// Whether a provider on `nets` (its observed networks; empty = unknown, reached only through a
/// relay) may join an audit as a partner beside those already in it (`chosen`, their networks;
/// `unknown_partners`, how many partners so far are of unknown network): no network in common with
/// any of them, and at most one partner of unknown network. The subject's own unknown network
/// doesn't use that one place (review round 4 H5: a relay-only subject would otherwise leave room
/// for no partner it isn't already connected to); those of unknown network are judged against those
/// on known ones ([`resolve_against_known`]).
fn independent_partner(chosen: &[Vec<String>], unknown_partners: usize, nets: &[String]) -> bool {
    if nets.is_empty() {
        return unknown_partners == 0;
    }
    chosen.iter().all(|c| !c.iter().any(|n| nets.contains(n)))
}


/// The part of an answer an audit compares ([`AUDIT_COMPARE_CHARS`]).
fn compared(answer: &str) -> String {
    answer.chars().take(AUDIT_COMPARE_CHARS).collect()
}

/// Why a streaming attempt didn't complete.
enum StreamFail {
    /// The provider answered `SERVE_STREAM` with a non-streaming (legacy) reply — the caller
    /// retries the request over the buffered `request_completion` path and remembers this peer.
    Legacy,
    /// A real failure: a provider `Error` frame, a shed (`Overloaded`), a stall, or a dropped
    /// transport past the retry budget. Carries the error for the caller's failover logic.
    Error(AdapterError),
    /// The caller cancelled it (its client left).
    Cancelled,
}

/// Dispatch one incremental slice of serve frames: text deltas → `on_delta`, tool calls held for
/// the summary, `Done` → return the summary, `Error` → fail. `None` means "slice consumed, keep
/// polling". Shared by the open-ack and the poll loop.
fn dispatch_stream_slice(
    framed: &[u8],
    tool_calls: &mut Vec<ToolCall>,
    opening: bool,
    on_delta: &mut dyn FnMut(&str),
) -> Result<Option<ServeSummary>, StreamFail> {
    let chunks = parse_response(framed)
        .map_err(|e| StreamFail::Error(AdapterError::Parse(e.to_string())))?;
    for (i, chunk) in chunks.into_iter().enumerate() {
        match chunk {
            ServeChunk::Delta(text) => on_delta(&text),
            ServeChunk::ToolCalls(tc) => *tool_calls = tc,
            ServeChunk::Done { tokens, metrics } => {
                return Ok(Some(ServeSummary {
                    tokens,
                    ok: true,
                    metrics,
                    discover_ns: 0,
                    proxy_roundtrip_ns: 0,
                    tool_calls: std::mem::take(tool_calls),
                    authenticity: None,
                }))
            }
            // A refusal is the first frame of the stream's opening reply (B3).
            ServeChunk::Error(msg) => return Err(StreamFail::Error(provider_error(msg, opening && i == 0))),
        }
    }
    Ok(None)
}

/// A provider's error frame as an error. A payment refusal stands as one only where a provider
/// refuses (`at_admission`: before anything else it sends); anywhere else (after empty polls, after
/// tool calls, mid-stream) it is an ordinary failure, so a provider can't end a serve it took on
/// without it counting against it (review B3).
fn provider_error(msg: String, at_admission: bool) -> AdapterError {
    match msg.strip_prefix(crate::serve::PAYMENT_REFUSAL) {
        Some(rest) if !at_admission => AdapterError::Provider(format!("refused payment after the serve began: {rest}")),
        // Its serve pool was full (the provider's own shed reply, before anything was served).
        _ if at_admission && msg == SHED_REPLY => AdapterError::Overloaded(msg),
        _ => AdapterError::Provider(msg),
    }
}

/// The text of a provider's shed reply to a request its full serve pool couldn't take.
const SHED_REPLY: &str = "provider overloaded, retry";

/// Open a stream to a provider and drive the poll loop, delivering each delta to `on_delta`.
///
/// `transport(framed) -> response` is a **single, raw** `proxy_forward` round-trip (no `0x11`
/// reconnect-fetch — streaming does its own drop-resume). `sleep(dur)` is injected so the poll
/// cadence and the stall/backoff clock are deterministic in tests (accumulated idle time is summed
/// from the sleeps, not read from a wall clock). Returns the [`ServeSummary`] on a clean `Done`, or
/// [`StreamFail::Legacy`] when the provider doesn't speak streaming (→ caller falls back).
fn request_completion_streaming(
    transport: &mut dyn FnMut(&[u8]) -> Result<Vec<u8>, AdapterError>,
    sleep: &mut dyn FnMut(Duration),
    request: &ServeRequest,
    on_delta: &mut dyn FnMut(&str),
) -> Result<ServeSummary, StreamFail> {
    request_completion_streaming_with(transport, sleep, request, &|| false, &mut StreamProgress::default(), on_delta)
}

/// [`request_completion_streaming`] that stops between polls once `cancel()` says so (its client
/// left: [`StreamFail::Cancelled`]) and keeps `offset_out` at the offset it has read up to, which a
/// close of the stream starts from (review S8).
fn request_completion_streaming_with(
    transport: &mut dyn FnMut(&[u8]) -> Result<Vec<u8>, AdapterError>,
    sleep: &mut dyn FnMut(Duration),
    request: &ServeRequest,
    cancel: &dyn Fn() -> bool,
    progress: &mut StreamProgress,
    on_delta: &mut dyn FnMut(&str),
) -> Result<ServeSummary, StreamFail> {
    // F2: absolute total-time backstop (real wall clock), generous + `max_tokens`-scaled so it
    // never aborts a legitimate long serve — only the pathological never-idle dribble.
    let started = std::time::Instant::now();
    let total_cap = stream_total_cap(request.max_tokens);

    // ── Open / capability probe ──
    let mut open = Vec::with_capacity(1 + request.messages.len() * 32);
    open.push(SERVE_STREAM);
    open.extend_from_slice(&request.encode());
    let ack = transport(&open).map_err(StreamFail::Error)?;
    // A reply that doesn't decode as a FetchChunksResponse is a legacy provider's blob → fall back.
    let ack = FetchChunksResponse::decode(&ack).map_err(|_| StreamFail::Legacy)?;

    let tool_calls = &mut progress.tool_calls;
    let offset_out = &mut progress.offset;
    let mut offset: u32 = 0;
    match ack {
        FetchChunksResponse::Overloaded => {
            return Err(StreamFail::Error(AdapterError::Overloaded(
                "stream shed at open".into(),
            )))
        }
        FetchChunksResponse::Forbidden => {
            return Err(StreamFail::Error(AdapterError::Network("stream open forbidden".into())))
        }
        FetchChunksResponse::NotFound => {
            return Err(StreamFail::Error(AdapterError::Network("stream missing at open".into())))
        }
        // The open ack may already carry frames — e.g. a refusal delivered as a complete
        // (done=true) stream, or an eager first slice.
        FetchChunksResponse::Chunks { framed, next_offset, done } => {
            // Received up to here, however the slice's frames are then handled (an error frame ends
            // it partway): a close starts after it, so nothing counted is counted twice.
            *offset_out = next_offset.max(offset);
            progress.opened = !done;
            if let Some(summary) = dispatch_stream_slice(&framed, tool_calls, true, on_delta)? {
                return Ok(summary);
            }
            offset = next_offset.max(offset);
        }
    }

    // ── Poll loop ──
    let mut interval = POLL_MIN;
    let mut idle = Duration::ZERO;
    loop {
        // Its client left: stop reading (the caller closes the stream and pays for what was read).
        if cancel() {
            return Err(StreamFail::Cancelled);
        }
        // F2: absolute backstop — bail if the whole attempt has outrun its (generous) total cap.
        if started.elapsed() > total_cap {
            return Err(StreamFail::Error(AdapterError::Network(
                "stream exceeded max total time".into(),
            )));
        }
        let mut poll = Vec::with_capacity(1 + 16 + 4);
        poll.push(FETCH_CHUNKS);
        poll.extend_from_slice(&request.nonce);
        poll.extend_from_slice(&offset.to_be_bytes());

        // Drop-resume: a transport error re-polls the SAME offset (the provider still holds the
        // un-fetched tail), bounded by the retry budget + the stall clock.
        let resp = {
            let mut attempt = 0u32;
            loop {
                match transport(&poll) {
                    Ok(bytes) => break bytes,
                    Err(e) => {
                        attempt += 1;
                        if attempt > STREAM_DROP_RETRIES {
                            return Err(StreamFail::Error(AdapterError::Network(format!(
                                "stream poll dropped past retry budget: {}", e.message()
                            ))));
                        }
                        sleep(STREAM_DROP_BACKOFF);
                        idle += STREAM_DROP_BACKOFF;
                        if idle >= STREAM_STALL_TIMEOUT {
                            return Err(StreamFail::Error(AdapterError::Network(
                                "stream stalled (no progress before deadline)".into(),
                            )));
                        }
                    }
                }
            }
        };

        match FetchChunksResponse::decode(&resp) {
            Ok(FetchChunksResponse::Chunks { framed, next_offset, done }) => {
                *offset_out = next_offset.max(offset); // as at the open, before the frames are handled
                let progressed = next_offset > offset;
                if let Some(summary) = dispatch_stream_slice(&framed, tool_calls, false, on_delta)? {
                    return Ok(summary);
                }
                let _ = done; // a true `done` always carries a terminal frame → returned above
                offset = next_offset.max(offset);
                if progressed {
                    interval = POLL_MIN;
                    idle = Duration::ZERO;
                } else {
                    interval = (interval * 2).min(POLL_MAX);
                    idle += interval;
                    if idle >= STREAM_STALL_TIMEOUT {
                        return Err(StreamFail::Error(AdapterError::Network(
                            "stream stalled (no progress before deadline)".into(),
                        )));
                    }
                }
                sleep(interval);
            }
            // Buffer gone (provider restarted / TTL expired) — the caller re-serves from scratch.
            Ok(FetchChunksResponse::NotFound) => {
                return Err(StreamFail::Error(AdapterError::Network(
                    "stream result not found (provider restarted)".into(),
                )))
            }
            Ok(FetchChunksResponse::Forbidden) => {
                return Err(StreamFail::Error(AdapterError::Network("stream poll forbidden".into())))
            }
            // A poll shed by a busy provider (review round 4 M5): back off and poll the same offset
            // again, within the stall clock, as for a poll that brought nothing.
            Ok(FetchChunksResponse::Overloaded) => {
                interval = (interval * 2).min(POLL_MAX);
                idle += interval;
                if idle >= STREAM_STALL_TIMEOUT {
                    return Err(StreamFail::Error(AdapterError::Network(
                        "stream stalled (provider overloaded)".into(),
                    )));
                }
                sleep(interval);
            }
            Err(e) => {
                return Err(StreamFail::Error(AdapterError::Parse(format!(
                    "bad stream poll reply: {e}"
                ))))
            }
        }
    }
}

/// A consumer node: discovers providers and serves completions over the swarm — the
/// **synchronous core** the HTTP/SSE gateway wraps (`complete` blocks; the gateway calls
/// it from a `spawn_blocking` task and streams the deltas).
pub struct ConsumerNode {
    net: NetworkHandle,
    tier: u8,
    /// Per-provider earned reputation (M2.2(a)), keyed by OpenHydra peer_id — the same key
    /// the receipts use. Mutated on `&self` (the gateway calls `complete` from a blocking
    /// task), so it lives behind a `Mutex`.
    reputation: Mutex<HashMap<String, ReputationTracker>>,
    /// Credit-tiering Phase 4: per-`(peer, model)` audit history (`n` passing audits + pass-rate
    /// `a`), keyed by the same composite `reputation_key` as `reputation`. Drives the
    /// `max_awardable_tier` gate on how much give-side credit a provider earns for a big model it
    /// hasn't proven it serves. Fed by `audit_model`, persisted like `reputation`.
    audit_stats: Mutex<HashMap<String, AuditStats>>,
    /// Crypto-rail off-chain **gated voucher** per composite `(peer, model)` key (UNIFIED_TRUST_STACK
    /// crypto sequence step 3): the monotonic cumulative authenticity-gated amount (provider net +
    /// 4% protocol fee) this consumer has authorized to a provider — the running total the crypto
    /// rail would settle against. Metered at the *achieved*-tier rate ([`gated_accrual`]), so it only
    /// ever accrues at the proven rate (no authenticity escrow on crypto). **No real value moves** —
    /// off-chain accounting; the on-chain claim is a later, legally-gated slice. Persisted like
    /// `audit_stats`.
    gated_vouchers: Mutex<HashMap<String, GatedVoucher>>,
    /// The highest cumulative signed per voucher key, co-signed or not (persisted): what the next
    /// receipt builds on, since anything signed can be claimed.
    signed_highwater: Mutex<HashMap<String, u64>>,
    /// Optional durable backing for `reputation` (M2.2(a)), `credit` (M2.3), and `audit_stats`
    /// (Phase 4). When present, the maps are rehydrated from it at startup and flushed back on each
    /// update, so trust, give/take, and audit history survive a restart. `None` = ephemeral.
    store: Option<Store>,
    /// M2.3 give-side credit, keyed by **provider libp2p PeerId** — the dial target we
    /// settled a receipt with. When a provider co-signs our receipt it *served* us, so we
    /// record its contribution (`record_served`) here, the symmetric half of the provider's
    /// take-side `record_consumption`. Shares the store's `PEER_CREDIT` table with the
    /// provider role: a node that both serves and provides against the **same `--db`** builds
    /// one give/take ledger per counterparty. (Within a single process only this role
    /// mutates it; cross-process unification merges on rehydrate — full live unification
    /// lands with the combined-role / enforcement work.)
    credit: Mutex<HashMap<String, CreditAccount>>,
    /// Shared transfer counters (#7/#5): per-model **consumed** aggregates + recent `used`
    /// ledger rows. `None` when no status endpoint is running. The symmetric half of the
    /// provider's `TransferStats` served side; the desktop merges both processes' views.
    stats: Option<Arc<crate::status::TransferStats>>,
    /// #7: this node's own co-located provider libp2p PeerId, when it also serves (the desktop
    /// runs provider + gateway as two identities on one machine). A serve routed to *self* is
    /// not a real market transaction, so we skip receipt settlement, reputation, and give/take
    /// credit for it — no self-earning or self-deduction. `None` = consumer-only node.
    self_provider: Option<String>,
    /// P1c streaming: providers observed to answer `SERVE_STREAM` with a legacy (non-streaming)
    /// reply, keyed by libp2p id. We skip re-probing these and go straight to the buffered path.
    /// Best-effort/advisory — a wrong entry only forfeits the streaming benefit, never correctness
    /// (a re-probe would just re-learn it). Never persisted; a fresh process re-learns on first use.
    buffered_only: Mutex<std::collections::HashSet<String>>,
    /// M2: imported `.openhydra` cards — verified providers pinned locally and dialable by peer id
    /// without live discovery. Merged into the candidate set in `serve_once`. `None` = no store.
    cards: Option<Arc<crate::cards::CardStore>>,
    /// M4-base: our held swarm membership credentials, attached to a serve request so a provider will
    /// serve its private (swarm-scoped) model. `None` = we hold none / private routing off.
    creds: Option<Arc<crate::swarms::CredentialStore>>,
    /// UNIFIED_TRUST_STACK Increment 3: the buyer's required assurance for routing. Default
    /// [`AssurancePolicy::OPEN`] (admits every provider, verify-but-route) — set a stricter policy
    /// to hard-gate routing on an achieved authenticity tier. `Copy`, so it's read cheaply per serve.
    assurance: AssurancePolicy,
    /// Crypto-rail reference-price peg (USDC micro-units per credit unit) applied when stamping the
    /// gated voucher on a receipt. **`None` = the crypto rail is off** for this node — no cumulative
    /// is stamped (review F3, opt-in). Set via [`with_settlement_peg`](Self::with_settlement_peg).
    settlement_peg: Option<u64>,
    /// The on-chain deployment paying receipts settle on: each carries its claim, signed for it.
    settlement_chain: Option<crate::settlement_config::SettlementChain>,
    /// Per-`(peer, model)` serialization locks for settlement (review F1). The cumulative voucher must
    /// advance monotonically, but `settle_receipt` reads the base, co-signs over a network round-trip,
    /// then commits — so concurrent settles to the same pair would race and emit non-monotonic /
    /// divergent co-signed cumulatives. Each settle holds its pair's lock across the whole
    /// read→co-sign→commit window. Grows one entry per distinct `(peer, model)` seen.
    settle_locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// Settlement threads running now (bounded by [`MAX_SETTLES_IN_FLIGHT`]).
    settles_in_flight: Arc<std::sync::atomic::AtomicUsize>,
    /// `(provider libp2p id, model)` → when it last refused us over payment: tried last (and not
    /// audited) for [`PAYMENT_REFUSAL_TTL_MS`], so a consumer that doesn't pay its price doesn't ask
    /// it first on every request.
    payment_refusals: Mutex<HashMap<(String, String), u64>>,
    /// Per `(provider, model)`: when it last shed our request or couldn't be reached before serving
    /// anything (round 8 T8-3). Tried last for [`UNAVAILABLE_RANK_LAST_MS`]; a second within
    /// [`UNAVAILABLE_REPEAT_MS`] costs reputation.
    unavailable: Mutex<HashMap<(String, String), u64>>,
    /// Per `(provider libp2p id, model)`: when it last completed a serve for us.
    served_ok: Mutex<HashMap<(String, String), u64>>,
    /// Providers (libp2p id, model) that completed a real serve for us with standing (see
    /// [`has_writer_standing`]) within [`WRITER_CANDIDATE_TTL_MS`]: the only ones asked to write probe
    /// questions besides our own (round 9 A9-4/A9-5). At most [`MAX_WRITER_CANDIDATES`].
    writer_candidates: Mutex<HashMap<(String, String), u64>>,
    /// Per model: until when no outside writer is looked for again, after a search found none.
    no_writer_until: Mutex<HashMap<String, u64>>,
    /// Per `(provider, model)`: when an audit shed of it was last excused (round 10 X10-2).
    audit_sheds: Mutex<HashMap<(String, String), u64>>,
    /// Device attestation of providers (Phase 0). `None` = off (the default).
    provider_attest: Option<Arc<ProviderAttestation>>,
    /// Prompt-scoring (log-prob) checks of providers (Track A2-Logprob). `None` = off (the default).
    provider_logprob: Option<Arc<ProviderLogprob>>,
    /// Whether audits may replay users' greedy requests to public models to another provider (D1,
    /// see [`with_audit_replay`](Self::with_audit_replay)). Off by default.
    audit_replay: bool,
    /// Recent serves kept for replay as audits (in memory only).
    replays: Mutex<Vec<Replayable>>,
    /// Per model asked for: the shape of this consumer's recent requests, which a probe copies.
    traffic: Mutex<HashMap<String, TrafficShape>>,
    /// Per provider (libp2p id): when a paid settlement with it last succeeded and last failed, so an
    /// audit can tell a provider that just served us at our price refusing to be paid from one whose
    /// price we don't meet (review B1).
    settle_record: Mutex<HashMap<String, (u64, u64)>>,
    /// Per model probed: questions written for this gateway's probes (P1), not yet used; finished
    /// locally before use.
    probe_pool: Mutex<HashMap<String, ProbePool>>,
    /// Hashes of the finished probe prompts already sent: none is sent twice.
    probe_sent: Mutex<std::collections::HashSet<u64>>,
    /// Providers an audit outvoted once (reputation key → when), awaiting a second audit (C3).
    strikes: Mutex<HashMap<String, Strike>>,
    /// Audits running now ([`MAX_AUDITS_IN_FLIGHT`]).
    audits_in_flight: Arc<std::sync::atomic::AtomicUsize>,
    /// Stream closes running now ([`MAX_CLOSES_IN_FLIGHT`]).
    closes_in_flight: Arc<std::sync::atomic::AtomicUsize>,
    /// When each provider below a hard floor was last audited to qualify (reputation key → when, H2).
    qualified_at: Mutex<HashMap<String, u64>>,
    /// Per model: when its qualifying audits started, within the last hour (R5T-5).
    qualified_per_model: Mutex<HashMap<String, std::collections::VecDeque<u64>>>,
    /// Qualifying audits running now (one at a time).
    qualifying: Arc<std::sync::atomic::AtomicUsize>,
    /// Follow-up audits pending now ([`MAX_FOLLOW_UPS`]).
    follow_ups: Arc<std::sync::atomic::AtomicUsize>,
    /// Per provider (libp2p id): when it last served us a request we then paid for (or tried to),
    /// for [`ConsumerNode::paid_recently`] (review round 4 T4-10).
    served_paid: Mutex<HashMap<String, u64>>,
}

/// Outcome of one serve pass ([`ConsumerNode::serve_once`]), used by [`ConsumerNode::complete`]
/// to decide whether to bypass the C11 discovery cache and retry.
enum AttemptOutcome {
    /// Served successfully — return this summary.
    Served(ServeSummary),
    /// No usable provider this pass (empty discovery, or every candidate failed with **nothing
    /// streamed** yet). Safe to invalidate the discovery cache and retry once — a stale cache
    /// hit (provider moved/died, replacement not yet cached) shouldn't strand the request.
    Retryable(AdapterError),
    /// Must propagate as-is — e.g. output was already streamed to the client (a retry would
    /// duplicate it), or a discovery-transport error a cache bypass can't fix.
    Fatal(AdapterError),
}

/// Evict per-`(peer, model)` settlement locks that no settle is using, keeping the lock map bounded
/// on a long-lived node (review F1 follow-up). An entry is **idle** iff only the map references its
/// `Arc` (`strong_count == 1`); any in-flight or waiting settle captured an extra clone (in
/// `settle_receipt`, before it takes the guard), so a busy pair's lock keeps `strong_count >= 2` and
/// survives the sweep — the serialization guarantee is never broken by pruning.
fn prune_idle_settle_locks(locks: &mut HashMap<String, Arc<Mutex<()>>>) {
    locks.retain(|_, arc| Arc::strong_count(arc) > 1);
}

impl ConsumerNode {
    /// An in-memory consumer (reputation and credit are not persisted across restarts).
    pub fn new(net: NetworkHandle) -> Self {
        Self {
            net,
            tier: 2,
            reputation: Mutex::new(HashMap::new()),
            audit_stats: Mutex::new(HashMap::new()),
            gated_vouchers: Mutex::new(HashMap::new()),
            signed_highwater: Mutex::new(HashMap::new()),
            store: None,
            credit: Mutex::new(HashMap::new()),
            stats: None,
            self_provider: None,
            buffered_only: Mutex::new(std::collections::HashSet::new()),
            cards: None,
            creds: None,
            assurance: AssurancePolicy::OPEN,
            settlement_peg: None,
            settlement_chain: None,
            settle_locks: Mutex::new(HashMap::new()),
            settles_in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            payment_refusals: Mutex::new(HashMap::new()),
            unavailable: Mutex::new(HashMap::new()),
            served_ok: Mutex::new(HashMap::new()),
            writer_candidates: Mutex::new(HashMap::new()),
            no_writer_until: Mutex::new(HashMap::new()),
            audit_sheds: Mutex::new(HashMap::new()),
            provider_attest: None,
            provider_logprob: None,
            audit_replay: false,
            replays: Mutex::new(Vec::new()),
            traffic: Mutex::new(HashMap::new()),
            settle_record: Mutex::new(HashMap::new()),
            probe_pool: Mutex::new(HashMap::new()),
            probe_sent: Mutex::new(std::collections::HashSet::new()),
            strikes: Mutex::new(HashMap::new()),
            audits_in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            closes_in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            qualified_at: Mutex::new(HashMap::new()),
            qualified_per_model: Mutex::new(HashMap::new()),
            qualifying: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            follow_ups: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            served_paid: Mutex::new(HashMap::new()),
        }
    }

    /// Set the buyer's required [`AssurancePolicy`] for routing (builder form; keeps the other
    /// constructor signatures unchanged). The default is [`AssurancePolicy::OPEN`] — pass a
    /// stricter policy to hard-gate routing on an achieved authenticity tier (Increment 3).
    pub fn with_assurance_policy(mut self, policy: AssurancePolicy) -> Self {
        self.assurance = policy;
        self
    }

    /// Opt in to device attestation of providers (builder form): after a served completion, challenge
    /// the provider in the background (at most every few minutes) and surface the verified
    /// device-trust level in the status view. `trusted_binaries` are the release binary hashes to
    /// accept. Purely informational in Phase 0: routing and authenticity tiers are unchanged.
    pub fn with_provider_attestation(mut self, trusted_binaries: Vec<[u8; 32]>) -> Self {
        self.provider_attest = Some(Arc::new(ProviderAttestation {
            trusted_binaries,
            records: Mutex::new(HashMap::new()),
            inflight: Mutex::new(std::collections::HashSet::new()),
        }));
        self
    }

    /// Opt in to prompt-scoring (log-prob) checks of providers (builder form): after a served
    /// completion from a provider whose model handle one of `refs` covers, run a challenge round in
    /// the background (at most every few hours per `(peer, model)`). The verdict feeds the audit
    /// history, reputation, and the authenticity tier — a pass can lift a provider to A2.
    pub fn with_logprob_checks(
        mut self,
        refs: Vec<LoadedReference>,
        registry: Option<crate::registry_client::RegistryClient>,
    ) -> Self {
        if refs.is_empty() && registry.is_none() {
            return self;
        }
        // Rehydrate: which texts each provider has already been sent, the round in progress, and the
        // latest verdict. A corrupt row is skipped with a warning (its provider then looks fresh to
        // us — the one case a text could be re-sent).
        let (mut rounds, mut status, mut registry_identities) = (HashMap::new(), HashMap::new(), HashMap::new());
        if let Some(store) = &self.store {
            match store.load_logprob_states() {
                Ok(rows) => {
                    for (key, blob) in rows {
                        match decode_logprob_state(&blob) {
                            Some((st, round, identity)) => {
                                if let Some(id) = identity {
                                    registry_identities.insert(st.reference, id);
                                }
                                status.insert(key.clone(), st);
                                rounds.insert(key, round);
                            }
                            None => tracing::warn!(key, "skipping a corrupt log-prob state row"),
                        }
                    }
                }
                Err(e) => tracing::warn!(error = %e, "could not load log-prob state"),
            }
        }
        // P7: count delivered tokens with any reference's tokenizer, given the scorer binary.
        let count_scorer = if refs.iter().any(|r| r.vocab.is_some()) {
            let scorer = crate::token_count::locate_count_scorer();
            match &scorer {
                Some(p) => tracing::info!(scorer = %p.display(), "checking billed token counts against reference tokenizers"),
                None => tracing::warn!("a reference carries a tokenizer, but no openhydra-scorer to count with"),
            }
            scorer
        } else {
            None
        };
        let issued_here = rounds.values().flat_map(|r: &ChallengeRound| r.issued_digests().copied()).collect();
        self.provider_logprob = Some(Arc::new(ProviderLogprob {
            refs: refs.into_iter().map(Arc::new).collect(),
            registry,
            registry_identities: Mutex::new(registry_identities),
            registry_weights: Mutex::new(HashMap::new()),
            rounds: Mutex::new(rounds),
            status: Mutex::new(status),
            inflight: Mutex::new(std::collections::HashSet::new()),
            count_scorer,
            counts_running: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            issued_here: Mutex::new(issued_here),
            resume: Mutex::new(HashMap::new()),
            no_verdict_runs: Mutex::new(HashMap::new()),
            first_contact_packs: Mutex::new(HashMap::new()),
        }));
        self
    }

    /// The tokenizer (vocabulary-only GGUF) and scorer binary to count a provider's delivered tokens
    /// with (P7): a reference naming the provider's model handle, else the one its latest log-prob
    /// verdict was against, as long as it carries a vocabulary.
    fn token_counter_for(&self, provider: &SelectedProvider, model_key: &str) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let lp = self.provider_logprob.as_ref()?;
        let scorer = lp.count_scorer.clone()?;
        let with_vocab = || lp.refs.iter().filter(|r| r.vocab.is_some());
        let by_handle = with_vocab().find(|r| r.handles.iter().any(|h| h == &provider.model_id));
        let by_verdict = || {
            let key = reputation_key(&provider.state_id, model_key);
            let tag = lp.status.lock().unwrap_or_else(|e| e.into_inner()).get(&key).map(|st| st.reference)?;
            with_vocab().find(|r| crate::prompt_score::reference_tag(&r.set) == tag)
        };
        let vocab = by_handle.or_else(by_verdict)?.vocab.clone()?;
        Some((scorer, vocab))
    }

    /// Count a served completion's tokens with the model's tokenizer and hold a bill above them
    /// against the provider (P7, detached, off the response path).
    #[allow(clippy::too_many_arguments)]
    fn check_billed_tokens(
        self: &Arc<Self>,
        counter: (std::path::PathBuf, std::path::PathBuf),
        text: String,
        // The whole answer's text measure when `text` is only its start (too long to keep whole).
        total_units: Option<u64>,
        tool_calls: Vec<crate::adapter::ToolCall>,
        offered: std::collections::HashSet<String>,
        billed: u64,
        state_id: String,
        model_key: String,
    ) {
        // Bounded: a busy gateway skips the check rather than piling up scorer processes.
        let Some(slot) = self
            .provider_logprob
            .as_ref()
            .and_then(|lp| crate::token_count::CountSlot::try_take(&lp.counts_running))
        else {
            return;
        };
        let node = Arc::clone(self);
        std::thread::spawn(move || {
            let _slot = slot;
            let (scorer, vocab) = counter;
            // Text that doesn't fit beside the tool calls is cut, and the count estimated from it.
            let mut text = text;
            let mut total_units = total_units;
            let whole_units = {
                let mut t = crate::serve::DeliveredText::default();
                t.push(&text);
                t.units()
            };
            if crate::token_count::fit_for_counting(&mut text, &tool_calls) {
                total_units.get_or_insert(whole_units);
            }
            // Tool calls too large to keep: not checked.
            let Some((countable, calls)) = crate::token_count::countable_text(&text, &tool_calls, &offered) else {
                return;
            };
            // Counted in pieces; an answer too long to keep whole is estimated from its start (B5).
            let counted = crate::token_count::count_tokens_chunked(&scorer, &vocab, &countable).map(|n| match total_units {
                Some(total) => {
                    let mut part = crate::serve::DeliveredText::default();
                    part.push(&text);
                    crate::token_count::estimate_from_part(n, part.units(), total)
                }
                None => n,
            });
            match counted {
                Ok(counted) if crate::token_count::over_billed(billed, counted, calls) => {
                    tracing::warn!(provider = %state_id, model = %model_key, billed, counted,
                        "provider billed for more tokens than it delivered");
                    node.record_outcome(&state_id, &model_key, VerificationOutcome::Rejected, now_unix_ms());
                }
                Ok(counted) => tracing::debug!(provider = %state_id, billed, counted, "billed token count checks out"),
                // Visible: a vocabulary that won't load turns the check off for its model.
                Err(e) => tracing::warn!(error = %e, vocab = %vocab.display(), "couldn't count delivered tokens"),
            }
        });
    }

    /// Persist one `(peer, model)`'s log-prob state (best-effort: a failed write is logged, never
    /// fails the check).
    fn persist_logprob_state(&self, key: &str, status: LogprobStatus, round: &ChallengeRound) {
        if let Some(store) = &self.store {
            // A registry pack's identity, kept with the row (a local reference names its own).
            let identity = self.provider_logprob.as_ref().and_then(|lp| {
                lp.registry_identities.lock().unwrap_or_else(|e| e.into_inner()).get(&status.reference).cloned()
            });
            if let Err(e) = store.put_logprob_state(key, &encode_logprob_state(status, round, identity.as_ref())) {
                tracing::warn!(error = %e, key, "log-prob state persist failed");
            }
        }
    }

    /// The latest log-prob status for `(peer_id, model_key)`, if checks are on and it was tried.
    pub fn logprob_status(&self, peer_id: &str, model_key: &str) -> Option<LogprobStatus> {
        let lp = self.provider_logprob.as_ref()?;
        let key = reputation_key(peer_id, model_key);
        lp.status.lock().unwrap_or_else(|e| e.into_inner()).get(&key).copied()
    }

    /// The latest device-attestation record for a provider (OpenHydra peer id), if any.
    pub fn attest_record(&self, peer_id: &str) -> Option<AttestRecord> {
        let attest = self.provider_attest.as_ref()?;
        attest.records.lock().unwrap_or_else(|e| e.into_inner()).get(peer_id).cloned()
    }

    /// Enable the crypto rail with a settlement peg (USDC micro-units per credit unit) used when
    /// stamping the gated voucher on receipts (builder form). The live gateway wires it from
    /// `OPENHYDRA_USDC_MICROS_PER_CREDIT`. A `0` (or unset) peg leaves the crypto rail **off** — no
    /// cumulative is stamped — so a node can't silently emit mis-denominated vouchers (review F3).
    /// Settle paying receipts on `chain` (builder form): each carries its on-chain claim, signed for
    /// that deployment by our identity (Solana) or wallet (Base).
    /// Let audits replay users' requests (D1): a greedy (temperature 0) request to a public model is
    /// sent again, as is, to another provider of the model, and the answers are compared. The
    /// audited provider can't tell it was checked, and the other sees an ordinary request. It means
    /// a user's prompt also reaches a second provider: never a private swarm's (those aren't kept),
    /// never written to disk. Off, audits send probes shaped like this consumer's traffic instead.
    pub fn with_audit_replay(mut self, on: bool) -> Self {
        self.audit_replay = on;
        self
    }

    pub fn with_settlement_chain(mut self, chain: Option<crate::settlement_config::SettlementChain>) -> Self {
        self.settlement_chain = chain;
        self
    }

    pub fn with_settlement_peg(mut self, micros_per_credit: u64) -> Self {
        self.settlement_peg = (micros_per_credit > 0).then_some(micros_per_credit);
        self
    }

    /// The per-`(peer, model)` settlement lock for `vkey`, creating it on first use — held across a
    /// settle's read→co-sign→commit so concurrent settles to the same pair serialize (review F1).
    ///
    /// Bounds the lock map: a long-lived node churning through many providers would otherwise grow one
    /// permanent entry per distinct `(peer, model)` ever settled. Past a threshold we sweep entries no
    /// settle is using (`prune_idle_settle_locks`) before handing back this pair's lock. The sweep is
    /// gated on size so the common path stays O(1).
    fn settle_lock_for(&self, vkey: &str) -> Arc<Mutex<()>> {
        const PRUNE_THRESHOLD: usize = 256;
        let mut locks = self.settle_locks.lock().unwrap_or_else(|e| e.into_inner());
        if locks.len() >= PRUNE_THRESHOLD {
            prune_idle_settle_locks(&mut locks);
        }
        locks.entry(vkey.to_string()).or_insert_with(|| Arc::new(Mutex::new(()))).clone()
    }

    /// Attach the shared transfer counters so completions record per-model consumed tokens
    /// and `used` ledger rows (builder form; keeps `new`/`with_store` signatures unchanged).
    pub fn with_stats(mut self, stats: Arc<crate::status::TransferStats>) -> Self {
        self.stats = Some(stats);
        self
    }

    /// #7: mark this node's own provider peer id, so a self-serve settles no receipt and moves
    /// no credit (builder form). `None`/empty leaves it consumer-only.
    pub fn with_self_provider(mut self, peer_id: Option<String>) -> Self {
        self.self_provider = peer_id.filter(|s| !s.is_empty());
        self
    }

    /// M2: attach the imported-cards store so `serve_once` can route to card-pinned providers by peer
    /// id without live discovery (builder form). `None` leaves card routing off.
    pub fn with_cards(mut self, cards: Option<Arc<crate::cards::CardStore>>) -> Self {
        self.cards = cards;
        self
    }

    /// M4-base: attach our swarm membership credential store, so `serve_once` presents a credential to
    /// reach a provider's private model (builder form). `None` leaves private routing off.
    pub fn with_credentials(mut self, creds: Option<Arc<crate::swarms::CredentialStore>>) -> Self {
        self.creds = creds;
        self
    }

    /// #7: is `provider_libp2p` this node's own co-located provider?
    fn is_self_serve(&self, provider_libp2p: &str) -> bool {
        self.self_provider.as_deref() == Some(provider_libp2p)
    }

    /// A consumer whose earned reputation (M2.2(a)) and give-side credit (M2.3) are
    /// **persisted** to `store`: both maps are rehydrated now and flushed back durably.
    pub fn with_store(net: NetworkHandle, store: Store) -> Self {
        let mut reputation = HashMap::new();
        // The consumer doesn't replay-guard nonces (that's the provider's job); this
        // throwaway tracker just satisfies the shared rehydration API.
        let mut nonces = NonceTracker::new();
        if let Err(e) = store.load_state_into_memory(&mut nonces, &mut reputation) {
            tracing::warn!(error = %e, "could not rehydrate reputation from store; starting fresh");
        }
        let mut credit = HashMap::new();
        if let Err(e) = store.load_credit_into_memory(&mut credit) {
            tracing::warn!(error = %e, "could not rehydrate credit from store; starting fresh");
        }
        let mut audit_stats = HashMap::new();
        if let Err(e) = store.load_audit_stats_into_memory(&mut audit_stats) {
            tracing::warn!(error = %e, "could not rehydrate audit stats from store; starting fresh");
        }
        let mut gated_vouchers = HashMap::new();
        match store.load_gated_vouchers_into_memory(&mut gated_vouchers) {
            Ok(0) => {}
            Ok(skipped) => tracing::warn!(skipped, "skipped corrupt gated-voucher rows on rehydrate; kept the rest"),
            Err(e) => tracing::warn!(error = %e, "could not rehydrate gated vouchers from store; starting fresh"),
        }
        let mut signed_highwater = HashMap::new();
        if let Err(e) = store.load_consumer_signed_into_memory(&mut signed_highwater) {
            tracing::warn!(error = %e, "could not rehydrate signed cumulatives from store");
        }
        Self {
            net,
            tier: 2,
            reputation: Mutex::new(reputation),
            audit_stats: Mutex::new(audit_stats),
            gated_vouchers: Mutex::new(gated_vouchers),
            signed_highwater: Mutex::new(signed_highwater),
            store: Some(store),
            credit: Mutex::new(credit),
            stats: None,
            self_provider: None,
            buffered_only: Mutex::new(std::collections::HashSet::new()),
            cards: None,
            creds: None,
            assurance: AssurancePolicy::OPEN,
            settlement_peg: None,
            settlement_chain: None,
            settle_locks: Mutex::new(HashMap::new()),
            settles_in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            payment_refusals: Mutex::new(HashMap::new()),
            unavailable: Mutex::new(HashMap::new()),
            served_ok: Mutex::new(HashMap::new()),
            writer_candidates: Mutex::new(HashMap::new()),
            no_writer_until: Mutex::new(HashMap::new()),
            audit_sheds: Mutex::new(HashMap::new()),
            provider_attest: None,
            provider_logprob: None,
            audit_replay: false,
            replays: Mutex::new(Vec::new()),
            traffic: Mutex::new(HashMap::new()),
            settle_record: Mutex::new(HashMap::new()),
            probe_pool: Mutex::new(HashMap::new()),
            probe_sent: Mutex::new(std::collections::HashSet::new()),
            strikes: Mutex::new(HashMap::new()),
            audits_in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            closes_in_flight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            qualified_at: Mutex::new(HashMap::new()),
            qualified_per_model: Mutex::new(HashMap::new()),
            qualifying: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            follow_ups: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            served_paid: Mutex::new(HashMap::new()),
        }
    }

    /// P1c: has `libp2p_id` been observed to speak only the buffered (non-streaming) serve path?
    fn is_buffered_only(&self, libp2p_id: &str) -> bool {
        self.buffered_only.lock().map(|s| s.contains(libp2p_id)).unwrap_or(false)
    }

    /// P1c: remember that `libp2p_id` answered `SERVE_STREAM` with a legacy reply, so the next
    /// serve to it skips the probe and goes straight to buffered.
    fn mark_buffered_only(&self, libp2p_id: &str) {
        if let Ok(mut s) = self.buffered_only.lock() {
            s.insert(libp2p_id.to_string());
        }
    }

    /// The consumer's locally-earned reputation for `peer_id` **on `model_key`**, decayed to
    /// `now_ms`, or `None` if this consumer has no history with that `(peer, model)` (→ neutral in
    /// ranking). Per-`(peer, model)` since Phase 4a: trust the peer earned on another model does
    /// not answer here (anti-Sybil). `model_key` is [`rep_model_key`]'s output.
    fn earned_reputation(&self, peer_id: &str, model_key: &str, now_ms: u64) -> Option<f64> {
        let key = reputation_key(peer_id, model_key);
        self.reputation.lock().ok()?.get(&key).map(|t| t.score_at(now_ms))
    }

    /// Whether this consumer failed `(peer_id, model_key)` in an audit within [`AUDIT_FAIL_FLOOR_MS`].
    fn failed_audit_recently(&self, peer_id: &str, model_key: &str, now_ms: u64) -> bool {
        self.audit_stats
            .lock()
            .ok()
            .and_then(|m| m.get(&reputation_key(peer_id, model_key)).map(|s| s.failed_within(now_ms, AUDIT_FAIL_FLOOR_MS)))
            .unwrap_or(false)
    }

    /// The achieved [`AuthenticityTier`] this consumer has established for `(peer_id, model_key)`
    /// from its own redundant-exec audit history (UNIFIED_TRUST_STACK Increment 3 — the routing
    /// gate's resolver). Uses the **decayed** earned reputation (the live trust the router already
    /// ranks on), the recency-weighted audit pass-rate + passing count, and no hard PoI method
    /// (A0/A1 only, per 2a). An untried `(peer, model)` resolves to `A0` (`NEUTRAL_REPUTATION` is
    /// below `R_MIN`), so a buyer requiring `A1+` never routes to a provider it has not yet proven.
    fn authenticity_of(&self, peer_id: &str, model_key: &str, now_ms: u64) -> AuthenticityTier {
        let reputation = self.earned_reputation(peer_id, model_key, now_ms).unwrap_or(NEUTRAL_REPUTATION);
        let key = reputation_key(peer_id, model_key);
        let (rate, passes, audit_failed) = self
            .audit_stats
            .lock()
            .ok()
            .and_then(|m| {
                m.get(&key).map(|s| (s.pass_rate(now_ms), s.recent_passes(now_ms), s.failed_within(now_ms, AUDIT_FAIL_FLOOR_MS)))
            })
            .unwrap_or((1.0, 0, false));
        // Route through the curated redundant-exec verifier (Increment 3b) — the live caller of the
        // Verifier seam. Same core resolution as `economy_snapshot`'s display path. A recent
        // log-prob verdict (Track A2-Logprob) supplies the log-prob method flag and the latest
        // outcome; without one there is no hard method and no live outcome.
        let (logprob, latest_outcome) = self
            .provider_logprob
            .as_ref()
            .and_then(|lp| lp.status.lock().unwrap_or_else(|e| e.into_inner()).get(&key).copied())
            .map_or((false, None), |st| {
                let (flag, outcome) = st.evidence(now_ms);
                (flag && self.check_matches_name(model_key, &st), outcome)
            });
        // A redundant-execution audit it failed recently is a live failure too (review T4), whatever
        // a log-prob check said since: it may serve the real model only when it is being checked.
        let latest_outcome = if audit_failed { Some(VerificationOutcome::Failed) } else { latest_outcome };
        let evidence = AuthenticityEvidence {
            reputation,
            audit_pass_rate: rate,
            passing_audits: passes,
            methods: MethodFlags { logprob_commitment: logprob, ..MethodFlags::default() },
            latest_outcome,
        };
        RedundantExecAuthenticity.verify(&evidence).authenticity
    }

    /// Snapshot the consumer's give-to-get economy for the status endpoint: earned
    /// reputation per provider (OpenHydra-keyed) and give-side credit balance per provider
    /// (libp2p-keyed). Pure reads of both maps; safe to call from a background publisher thread.
    ///
    /// Reputation and the authenticity tier are the ones routing and pricing use (review T6): the
    /// score decayed to now, and the tier from [`authenticity_of`](Self::authenticity_of). With a
    /// week's half-life the decayed score moves slowly (about a tenth of a point per half hour
    /// near the top). The standing score as of the last outcome is reported beside it.
    pub fn economy_snapshot(&self) -> (Vec<crate::status::RepEntry>, Vec<crate::status::CreditEntry>) {
        let now = now_unix_ms();
        // Crypto-rail voucher: cumulative net owed per `(peer, model)`. Snapshotted under its own
        // lock, released before reputation is taken (same composite key, never held nested).
        let gated: std::collections::HashMap<String, u64> = self
            .gated_vouchers
            .lock()
            .map(|m| {
                // Vouchers are per provider key ("<peer:model>#<pubkey>", "@<deployment>" on a
                // chain); the view sums them per (peer, model), the key the rest of the snapshot
                // uses, for the deployment we settle on now only.
                let suffix = self.settlement_chain.as_ref().map(|_| self.voucher_scope());
                let mut by_pair = std::collections::HashMap::new();
                for (k, v) in m.iter() {
                    let current = match &suffix {
                        Some(s) => k.ends_with(s.as_str()),
                        None => !k.contains('@'),
                    };
                    if !current {
                        continue;
                    }
                    let pair = k.split_once('#').map_or(k.as_str(), |(pair, _)| pair);
                    *by_pair.entry(pair.to_string()).or_insert(0u64) += v.cumulative_net;
                }
                by_pair
            })
            .unwrap_or_default();
        // Device attestation (Phase 0): the verified device-trust level + key rotations per provider.
        // Snapshotted under its own lock, released before reputation is taken.
        let attest_now = now_unix_ms();
        let device: std::collections::HashMap<String, (Option<&'static str>, u32)> = self
            .provider_attest
            .as_ref()
            .map(|a| {
                a.records
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .iter()
                    .map(|(peer, r)| (peer.clone(), (r.device_trust(attest_now).map(|t| t.label()), r.key_rotations)))
                    .collect()
            })
            .unwrap_or_default();
        // Log-prob verdicts (Track A2-Logprob), snapshotted under their own lock like the rest.
        let logprob: std::collections::HashMap<String, LogprobStatus> = self
            .provider_logprob
            .as_ref()
            .map(|lp| lp.status.lock().unwrap_or_else(|e| e.into_inner()).clone())
            .unwrap_or_default();
        // (key, standing score, decayed score), taken under the reputation lock and released before
        // the tier is resolved (which takes that lock again).
        let scores: Vec<(String, f64, f64)> = self
            .reputation
            .lock()
            .map(|m| m.iter().map(|(k, t)| (k.clone(), t.raw_score(), t.score_at(now))).collect())
            .unwrap_or_default();
        let round = |x: f64| (x * 10.0).round() / 10.0;
        let reputation = scores
            .into_iter()
            .map(|(key, standing, score)| {
                // Phase 4a: the map key is the composite `(peer, model)` — split it back for display.
                // A key without the separator (should not occur post-migration) degrades gracefully
                // to peer-only with no model.
                let (peer, model) = match split_reputation_key(&key) {
                    Some((p, m)) => (p.to_string(), Some(m.to_string())),
                    None => (key.clone(), None),
                };
                // The tier routing and pricing see for this `(peer, model)`.
                let tier = model.as_deref().map(|m| self.authenticity_of(&peer, m, now));
                let logprob_check = logprob.get(&key).and_then(|st| st.label());
                let (device_trust, rotations) = device.get(&peer).copied().unwrap_or((None, 0));
                crate::status::RepEntry {
                    // Shown as the 64-bit peer id (the 128-bit state id's prefix) — the id the rest
                    // of the app knows the provider by.
                    openhydra_peer_id: peer.get(..16).unwrap_or(&peer).to_string(),
                    model_id: model,
                    score: round(score),
                    standing_score: Some(round(standing)),
                    authenticity: tier.map(|t| t.label().to_string()),
                    gated_net: gated.get(&key).copied().filter(|&n| n > 0),
                    device_trust: device_trust.map(str::to_string),
                    device_key_rotations: (rotations > 0).then_some(rotations),
                    logprob_check,
                }
            })
            .collect();
        let credit = self
            .credit
            .lock()
            .map(|m| {
                m.iter()
                    .map(|(id, a)| crate::status::CreditEntry {
                        libp2p_peer_id: id.clone(),
                        balance: (a.balance(now) * 10.0).round() / 10.0,
                        rate_cap: None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        (reputation, credit)
    }

    /// Record a verification outcome for `peer_id` (M2.2(a)): a served + clean completion
    /// is `Honored`; a failed/refused serve attempt is `Rejected`. Updates the in-memory
    /// tracker, then best-effort persists the snapshot (reputation is advisory — a failed
    /// write must never break the request path). Feeds the next ranking so a provider that
    /// misbehaves with this consumer is downranked out of routing.
    fn record_outcome(&self, peer_id: &str, model_key: &str, outcome: VerificationOutcome, now_ms: u64) {
        // Per-(peer, model) since Phase 4a: an outcome moves trust for THIS model only. A provider
        // that farms trust on a cheap model earns nothing toward a bigger one it later claims.
        let key = reputation_key(peer_id, model_key);
        let mut evicted = Vec::new();
        let snapshot = {
            let mut map = match self.reputation.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            if !map.contains_key(&key) {
                evicted = evict_for_new_pair(&mut map, |t| t.last_update_ms(), |_| false, now_ms);
            }
            let tracker = map
                .entry(key.clone())
                .or_insert_with(|| ReputationTracker::new(now_ms));
            tracker.record(outcome, now_ms);
            tracker.to_bytes()
        };
        // Persist outside the lock (don't hold the Mutex across the redb write).
        if let Some(store) = &self.store {
            for gone in &evicted {
                // Unless recorded again meanwhile (round 9 A9-8).
                if !self.reputation.lock().is_ok_and(|m| m.contains_key(gone)) {
                    let _ = store.remove_reputation(gone);
                }
            }
            if let Err(e) = store.put_reputation(&key, &snapshot) {
                tracing::debug!(error = %e, peer_id, model = model_key, "reputation persist failed");
            }
        }
    }

    /// The cost-tier a provider may currently **earn** at for `model_key` (credit-tiering Phase 4):
    /// the model's `claimed` cost-tier, gated by this `(peer, model)`'s audit history + reputation
    /// via `max_awardable_tier`. No audit history yet → the conservative floor (nano), so a provider
    /// earns the big-model rate only once it has *proven* (temp=0 audits) it serves the big model.
    /// Fold one redundant-exec audit outcome for `(peer, model_key)` into the per-`(peer, model)`
    /// audit history (`n`/`a`, [`AuditStats`]) and persist it. Only substantive verdicts reach here —
    /// a `Pass` (agreed) or a `Fail` (clear temp=0 divergence); non-responders / inconclusive audits
    /// are not evidence and never call this.
    ///
    /// **Note (post-review):** this collects the audit history + reputation-adjacent signal, but the
    /// `max_awardable_tier` *gate* (and the [`GradedFault`] escalation ladder) it feeds is **not**
    /// wired to any spendable credit today — the give/take throttle is raw-token (bandwidth) and
    /// displayed credit is weighted by the validated handle. The gate belongs with the future
    /// **payout** layer; the pure primitives (`max_awardable_tier`, `GradedFault`, escrow) live in
    /// [`openhydra_protocol::award`] awaiting it. Feeding [`AuditStats`] now keeps the durable audit
    /// history warm for that layer; the ephemeral fault ladder is deferred with it (a log-only
    /// escalation over unpersisted state would reset every restart and mislead — review state-F4).
    fn record_audit(&self, peer_id: &str, model_key: &str, class: AuditClass, real_traffic: bool, now_ms: u64) {
        let key = reputation_key(peer_id, model_key);
        let mut evicted = Vec::new();
        let snapshot = {
            let mut m = match self.audit_stats.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            if !m.contains_key(&key) {
                // A recent failure is never what makes room: it floors that provider.
                evicted = evict_for_new_pair(&mut m, |s| s.last_update_ms(), |s| s.failed_within(now_ms, AUDIT_FAIL_FLOOR_MS), now_ms);
            }
            let s = m.entry(key.clone()).or_insert_with(|| AuditStats::new(now_ms));
            match class {
                AuditClass::Pass if real_traffic => s.record_traffic_pass(now_ms),
                class => s.record(class, now_ms),
            }
            s.to_bytes()
        };
        if let Some(store) = &self.store {
            for gone in &evicted {
                if !self.audit_stats.lock().is_ok_and(|m| m.contains_key(gone)) {
                    let _ = store.remove_audit_stats(gone);
                }
            }
            if let Err(e) = store.put_audit_stats(&key, &snapshot) {
                tracing::debug!(error = %e, peer_id, model = model_key, "audit stats persist failed");
            }
        }
    }

    /// Accrue this serve's authenticity-gated amount into the per-`(peer, model)` crypto-rail
    /// **voucher** (UNIFIED_TRUST_STACK crypto sequence step 3). The rate is capped at the *achieved*
    /// authenticity tier ([`gated_accrual`] over the handle-implied claimed tier — the same H1/H3
    /// ceiling), so the voucher only ever accrues at the **proven** rate; the crypto rail needs no
    /// authenticity escrow. `peer_id` is the provider's OpenHydra id (keys the composite voucher +
    /// the `authenticity_of` lookup); `model_key` is its validated handle. **Off-chain accounting
    /// only — no value moves**; best-effort persisted, a failed write never affects the completion.
    /// This serve's authenticity-gated split (crypto-rail), denominated in **settlement-asset
    /// micro-units** (USDC): the *awarded*-tier amount for `tokens` (net to provider + 4% fee),
    /// capped at the achieved tier and converted to the asset via [`USDC_MICROS_PER_CREDIT`] at sign
    /// time (`SOLANA_SETTLEMENT_PROGRAM_SPEC` §2 — the on-chain program settles this number directly,
    /// no oracle). Pure over the current audit/reputation state — no voucher mutation. Used both to
    /// stamp the receipt's cumulative amount and (on a successful co-sign) to commit the voucher, so
    /// the two always agree.
    fn gated_split_for(&self, peer_id: &str, model_key: &str, tokens: u64, now_ms: u64, peg: u64) -> FeeSplit {
        let units = receipt_weighted_units(tokens, None);
        let claimed = self.claimed_tier(peer_id, model_key, now_ms);
        let achieved = self.authenticity_of(peer_id, model_key, now_ms);
        gated_accrual_asset(claimed, achieved, units, peg)
    }

    /// The cost tier a serve from `(peer_id, model_key)` is priced at before the authenticity gate.
    /// Normally the handle's conservative estimate (its size at 4-bit — the provider's name alone is
    /// never trusted for more). Once the provider has a qualifying log-prob pass streak against a
    /// reference with a known identity, the price follows **that verified identity** — the exact size
    /// and quantisation it proved it serves — so an honest BF16 provider earns the BF16 rate and a
    /// renamed model earns the rate of what it really is. The authenticity gate still applies on top.
    fn claimed_tier(&self, peer_id: &str, model_key: &str, now_ms: u64) -> CostTier {
        let Some(lp) = &self.provider_logprob else { return cost_tier_for_handle(model_key) };
        let key = reputation_key(peer_id, model_key);
        let status = lp.status.lock().unwrap_or_else(|e| e.into_inner()).get(&key).copied();
        // A bigger verified model is paid for only once audits on real traffic show it is served (D3).
        let traffic_proven = self.audit_stats.lock().unwrap_or_else(|e| e.into_inner()).get(&key).is_some_and(|s| {
            s.recent_traffic_passes(now_ms, crate::prompt_score::PRICE_UPLIFT_WINDOW_MS) >= crate::prompt_score::PRICE_UPLIFT_AUDITS
        });
        priced_tier(model_key, status.as_ref(), &|tag| self.reference_identity(tag), traffic_proven, now_ms)
    }

    /// The identity of the log-prob reference a [`reference_tag`](crate::prompt_score::reference_tag)
    /// names: a local reference file, or registry packs seen for that label.
    fn reference_identity(&self, tag: [u8; 16]) -> Option<openhydra_protocol::logprob::ModelIdentity> {
        let lp = self.provider_logprob.as_ref()?;
        identity_in(&lp.refs, tag).or_else(|| lp.registry_identities.lock().unwrap_or_else(|e| e.into_inner()).get(&tag).cloned())
    }

    /// Whether a passed log-prob check may show as A2 for `handle` (review T3): the verified model is
    /// the size its name says (when the name gives a size). Otherwise the check only informs pricing:
    /// a provider can't put an A2 badge on a name by passing checks for a different model.
    fn check_matches_name(&self, handle: &str, st: &LogprobStatus) -> bool {
        crate::prompt_score::identity_matches_name(handle, self.reference_identity(st.reference).as_ref())
    }

    /// The current cumulative gross (net + fee) of the `(peer, model)` voucher at composite `key`, or
    /// `0` if none yet — the base the next receipt's cumulative amount builds on.
    fn gated_cumulative_gross(&self, key: &str) -> u64 {
        self.gated_vouchers
            .lock()
            .ok()
            .and_then(|m| m.get(key).map(|v| v.cumulative_gross()))
            .unwrap_or(0)
    }


    /// The networks ([`network_of`]) each connected peer sits on, by libp2p id, from the addresses of
    /// our direct connections to it (relayed-only peers have none).
    fn observed_networks(&self) -> HashMap<String, Vec<String>> {
        self.net
            .status()
            .map(|st| {
                st.peers
                    .into_iter()
                    .map(|p| {
                        let mut nets: Vec<String> = p.direct_ips.iter().filter_map(|ip| network_of(ip)).collect();
                        nets.sort();
                        nets.dedup();
                        (p.peer_id, nets)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remember that `provider` refused us `model` over payment (see `payment_refusals`).
    fn note_payment_refusal(&self, provider: &str, model: &str, now: u64) {
        let mut m = self.payment_refusals.lock().unwrap_or_else(|e| e.into_inner());
        m.retain(|_, at| now.saturating_sub(*at) <= PAYMENT_REFUSAL_TTL_MS);
        if m.len() >= MAX_PAYMENT_REFUSALS {
            if let Some(oldest) = m.iter().min_by_key(|(_, at)| **at).map(|(k, _)| k.clone()) {
                m.remove(&oldest);
            }
        }
        m.insert((provider.to_string(), model.to_string()), now);
    }

    /// Note that `provider` shed our request for `model`, or couldn't be reached, before serving
    /// anything. Whether it did so before within [`UNAVAILABLE_REPEAT_MS`].
    fn note_unavailable(&self, provider: &str, model: &str, now: u64) -> bool {
        let mut m = self.unavailable.lock().unwrap_or_else(|e| e.into_inner());
        m.retain(|_, at| now.saturating_sub(*at) <= UNAVAILABLE_REPEAT_MS);
        if m.len() >= MAX_PAYMENT_REFUSALS {
            if let Some(oldest) = m.iter().min_by_key(|(_, at)| **at).map(|(k, _)| k.clone()) {
                m.remove(&oldest);
            }
        }
        m.insert((provider.to_string(), model.to_string()), now).is_some_and(|at| now.saturating_sub(at) <= UNAVAILABLE_REPEAT_MS)
    }

    /// Whether `provider` shed or missed a request of ours for `model` within
    /// [`UNAVAILABLE_RANK_LAST_MS`] (it is then tried last).
    fn briefly_unavailable(&self, provider: &str, model: &str, now: u64) -> bool {
        self.unavailable
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(provider.to_string(), model.to_string()))
            .is_some_and(|at| now.saturating_sub(*at) <= UNAVAILABLE_RANK_LAST_MS)
    }

    /// Note that `provider` (libp2p id) completed a serve of `model` for us.
    fn note_served_ok(&self, provider: &str, model: &str, now: u64) {
        let key = (provider.to_string(), model.to_string());
        let mut m = self.served_ok.lock().unwrap_or_else(|e| e.into_inner());
        if !m.contains_key(&key) && m.len() >= MAX_SETTLE_RECORDS {
            m.retain(|_, at| now.saturating_sub(*at) <= SERVED_OK_RECENT_MS);
            if m.len() >= MAX_SETTLE_RECORDS {
                return;
            }
        }
        m.insert(key, now);
    }

    /// Note that `provider` shed an audit of `model`: whether it shed one within
    /// [`AUDIT_SHED_WINDOW_MS`] before (round 10 X10-2: only the first in a day is excused, however
    /// rare audits are).
    fn note_audit_shed(&self, provider: &str, model: &str, now: u64) -> bool {
        let mut m = self.audit_sheds.lock().unwrap_or_else(|e| e.into_inner());
        m.retain(|_, at| now.saturating_sub(*at) <= AUDIT_SHED_WINDOW_MS);
        let key = (provider.to_string(), model.to_string());
        match m.get(&key) {
            Some(_) => true, // the window runs from the excused one
            None => {
                if m.len() < MAX_PAYMENT_REFUSALS {
                    m.insert(key, now);
                }
                false
            }
        }
    }

    /// Whether `provider` completed a serve of `model` for us within `window` of `now`.
    fn served_ok_within(&self, provider: &str, model: &str, now: u64, window: u64) -> bool {
        self.served_ok
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(provider.to_string(), model.to_string()))
            .is_some_and(|at| now.abs_diff(*at) <= window)
    }

    /// Whether `provider` completed a serve of `model` for us within [`SERVED_OK_RECENT_MS`] of `now`
    /// (that model's engine works: round 9 A9-3, per model, as another may be down or unshared).
    fn served_ok_recently(&self, provider: &str, model: &str, now: u64) -> bool {
        self.served_ok
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(provider.to_string(), model.to_string()))
            .is_some_and(|at| now.abs_diff(*at) <= SERVED_OK_RECENT_MS)
    }

    /// Note a provider that just completed a real serve of `model` for us, if it has standing: a
    /// candidate probe writer (round 9 A9-4/A9-5).
    fn note_writer_candidate(&self, provider: &SelectedProvider, now: u64) {
        if !has_writer_standing(self.earned_reputation(&provider.state_id, rep_model_key(&provider.canonical_model_id, &provider.model_id), now)) {
            return;
        }
        let key = (provider.libp2p_peer_id.clone(), provider.model_id.clone());
        let mut m = self.writer_candidates.lock().unwrap_or_else(|e| e.into_inner());
        m.retain(|_, at| now.saturating_sub(*at) <= WRITER_CANDIDATE_TTL_MS);
        if !m.contains_key(&key) && m.len() >= MAX_WRITER_CANDIDATES {
            if let Some(oldest) = m.iter().min_by_key(|(_, at)| **at).map(|(k, _)| k.clone()) {
                m.remove(&oldest);
            }
        }
        m.insert(key, now);
    }

    /// Whether `provider` refused us `model` over payment within [`PAYMENT_REFUSAL_TTL_MS`].
    fn refused_payment(&self, provider: &str, model: &str, now: u64) -> bool {
        self.payment_refusals
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(provider.to_string(), model.to_string()))
            .is_some_and(|at| now.saturating_sub(*at) <= PAYMENT_REFUSAL_TTL_MS)
    }

    /// What a voucher key carries after the provider's: the deployment we settle on and, on Base, our
    /// wallet (`""` off-chain). See `settle_receipt`.
    ///
    /// By our own key too (review round 4 M12): on-chain, and at the provider, what was paid so far is
    /// kept per consumer identity, so a new identity's running total starts from nothing; one carried
    /// over from an old identity would make the new one's first claim pay that history again.
    fn voucher_scope(&self) -> String {
        match &self.settlement_chain {
            Some(c) => {
                let me = self.net.public_key_bytes().map(hex::encode).unwrap_or_default();
                match self.settlement_wallet() {
                    Some(w) => format!("@{}@{w}@{me}", c.domain.fingerprint()),
                    None => format!("@{}@{me}", c.domain.fingerprint()),
                }
            }
            None => String::new(),
        }
    }

    /// On Base, the wallet our claims are signed with, as a serve request names it.
    fn settlement_wallet(&self) -> Option<String> {
        let wallet = self.settlement_chain.as_ref()?.evm_wallet.as_ref()?;
        Some(format!("0x{}", hex::encode(openhydra_protocol::settlement::evm_address(wallet.verifying_key()))))
    }

    /// What the next receipt for this voucher key builds on: the higher of its voucher and the highest
    /// cumulative we have signed for it.
    fn settle_base(&self, key: &str) -> u64 {
        let signed = self.signed_highwater.lock().unwrap_or_else(|e| e.into_inner()).get(key).copied().unwrap_or(0);
        self.gated_cumulative_gross(key).max(signed)
    }

    /// Remember (and persist) that we are signing `cumulative` for this voucher key, before sending it.
    fn note_signed(&self, key: &str, cumulative: u64) {
        let raised = {
            let mut m = self.signed_highwater.lock().unwrap_or_else(|e| e.into_inner());
            let e = m.entry(key.to_string()).or_insert(0);
            let raised = cumulative > *e;
            *e = (*e).max(cumulative);
            raised
        };
        if raised {
            if let Some(store) = &self.store {
                if let Err(e) = store.put_consumer_signed(key, cumulative) {
                    tracing::warn!(error = %e, "signed-cumulative persist failed");
                }
            }
        }
    }

    /// Raise this `(peer, model)`'s voucher to a co-signed cumulative it fell behind (see
    /// [`judge_behind_receipt`]), and persist it.
    fn raise_gated_voucher(&self, key: &str, gross: u64) {
        let snapshot = {
            let mut map = match self.gated_vouchers.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            let voucher = map.entry(key.to_string()).or_default();
            voucher.raise_to_gross(gross);
            voucher.to_bytes()
        };
        if let Some(store) = &self.store {
            if let Err(e) = store.put_gated_voucher(key, &snapshot) {
                tracing::debug!(error = %e, "gated voucher persist failed");
            }
        }
    }

    /// The distinct model ids this node currently knows about (PEX-learned / discovered).
    /// Backs the gateway's `GET /v1/models`; empty until discovery has populated the cache.
    pub fn known_models(&self) -> Result<Vec<String>, String> {
        self.net.known_models()
    }

    /// Discover a provider for `model`, pick the best, and stream the completion's text
    /// deltas to `on_delta`. Returns the [`ServeSummary`] (token count → receipt) on
    /// success. `model` is the engine handle / DHT key (e.g. `"qwen2.5:7b"`).
    #[tracing::instrument(
        name = "complete",
        skip_all,
        fields(
            model = %model,
            provider = tracing::field::Empty,
            tokens = tracing::field::Empty,
        )
    )]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn complete(
        self: &Arc<Self>,
        model: &str,
        messages: Vec<ChatMessage>,
        max_tokens: Option<u32>,
        temperature: Option<f64>,
        tools: Vec<Value>,
        think: Option<bool>,
        min_authenticity: Option<AuthenticityTier>,
        cancel: &dyn Fn() -> bool,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeSummary, AdapterError> {
        let now = now_unix_ms();
        // The buyer's floor for this request: this node's, raised (never lowered) by the request's.
        let policy = request_policy(self.assurance, min_authenticity);
        // The `complete` span (from #[instrument]); we fill in `provider`/`tokens` on success.
        let req_span = tracing::Span::current();
        // First pass may discover from the C11 cache. On a "no usable provider" outcome — empty
        // discovery, or every candidate failed with nothing streamed — the cached set may be
        // stale (a provider moved/died and its replacement isn't cached yet), so invalidate this
        // model's discovery cache and retry once with a cold get_providers lookup. The retry is
        // only reached when nothing was streamed, so it can never duplicate delivered output.
        match self.serve_once(model, &messages, max_tokens, temperature, &tools, think, &policy, cancel, now, &req_span, on_delta) {
            AttemptOutcome::Served(summary) => return Ok(summary),
            AttemptOutcome::Fatal(e) => return Err(e),
            AttemptOutcome::Retryable(e) if cancel() => return Err(e),
            AttemptOutcome::Retryable(_) => {}
        }
        tracing::info!(model, "C11 bypass: all candidates failed — invalidating discover cache and retrying with a fresh lookup");
        let _ = self.net.invalidate_discover(model);
        match self.serve_once(model, &messages, max_tokens, temperature, &tools, think, &policy, cancel, now, &req_span, on_delta) {
            AttemptOutcome::Served(summary) => Ok(summary),
            AttemptOutcome::Retryable(e) | AttemptOutcome::Fatal(e) => Err(e),
        }
    }

    /// One serve pass: discover (which may early-reply from the C11 cache) → rank → try
    /// providers in preference order with failover. Returns [`AttemptOutcome`] so
    /// [`complete`](Self::complete) can bypass the discovery cache and retry once on a
    /// `Retryable` result. `Retryable` is only returned when nothing has been streamed.
    #[allow(clippy::too_many_arguments)]
    fn serve_once(
        self: &Arc<Self>,
        model: &str,
        messages: &[ChatMessage],
        max_tokens: Option<u32>,
        temperature: Option<f64>,
        tools: &[Value],
        think: Option<bool>,
        policy: &AssurancePolicy,
        cancel: &dyn Fn() -> bool,
        now: u64,
        req_span: &tracing::Span,
        on_delta: &mut dyn FnMut(&str),
    ) -> AttemptOutcome {
        let t_discover = std::time::Instant::now();
        let mut peers = {
            let _span = tracing::info_span!("discover", model = %model).entered();
            match self.net.discover(model) {
                Ok(p) => p,
                // A discovery-transport error won't be fixed by a cache bypass → fatal.
                Err(e) => return AttemptOutcome::Fatal(AdapterError::Network(format!("discover: {e}"))),
            }
        };
        // C7: passive discovery (DHT get_providers + gossip PEX) can come up empty across NATs —
        // the D-sized gossip mesh doesn't forward a provider's advert to this specific consumer
        // and a NAT'd provider's DHT put_record times out. Fall back to asking a connected
        // bootstrap's provider registry, which has retained the verified adverts it saw. This is
        // a best-effort supplement: only when passive discovery found nothing, and a failed query
        // (no bootstrap, transport error) just leaves `peers` empty for the normal Retryable path.
        if peers.is_empty() {
            let _span = tracing::info_span!("registry_query", model = %model).entered();
            match self.net.query_registry(model) {
                Ok(reg) if !reg.is_empty() => {
                    tracing::info!(model, n = reg.len(), "C7: bootstrap registry supplied providers passive discovery missed");
                    peers = reg;
                }
                Ok(_) => {}
                Err(e) => tracing::debug!(model, error = %e, "C7: registry query failed (best-effort)"),
            }
        }
        // M2: imported `.openhydra` cards — verified providers pinned locally, dialable by peer id
        // with no live discovery. Merge them in (deduped by libp2p id) so an imported model routes
        // even when passive discovery + the bootstrap registry are both silent. The store re-verifies
        // + expiry-checks on read, so only currently-valid cards are added.
        if let Some(cards) = &self.cards {
            let pinned = cards.providers_for(model, now);
            if !pinned.is_empty() {
                let known: std::collections::HashSet<&str> =
                    peers.iter().map(|p| p.libp2p_peer_id.as_str()).collect();
                let fresh: Vec<_> = pinned
                    .into_iter()
                    .filter(|p| !known.contains(p.libp2p_peer_id.as_str()))
                    .collect();
                if !fresh.is_empty() {
                    tracing::info!(model, n = fresh.len(), "M2: added card-pinned providers");
                    peers.extend(fresh);
                }
            }
        }
        // "" canonical → any provider of this model_id (template-hash filtering is later).
        // M2.2(a): earned local reputation overrides the (neutral) self-reported score, so
        // a provider that has failed this consumer ranks below an untried one. Increment 3: the
        // buyer's assurance policy hard-gates (or, by default, verify-but-routes) on the achieved
        // authenticity tier this consumer has established from its own audits.
        // Our own provider is never held to the floor (review round 4 H2): it is never audited (it is
        // us), so it could never meet one, and the floor is about others' engines. (A private card's
        // provider is: the card is the provider's own word that it belongs to the swarm, R5T-8.)
        let own: std::collections::HashSet<String> = peers
            .iter()
            .filter(|p| self.is_self_serve(&p.libp2p_peer_id))
            .map(|p| provider_state_id(&p.peer_id, &p.public_key))
            .collect();
        let mut candidates = rank_providers_with_assurance(
            &peers,
            "",
            self.tier,
            &|pid, mkey| self.earned_reputation(pid, mkey, now),
            &|pid, mkey| if own.contains(pid) { AuthenticityTier::HardwareAttested } else { self.authenticity_of(pid, mkey, now) },
            policy,
        );
        if policy.mode == PolicyMode::HardGate {
            self.maybe_qualify(model, &peers, &candidates, now);
        }
        // Providers that just refused us over payment go last (in their order).
        candidates.sort_by_key(|c| {
            (self.refused_payment(&c.libp2p_peer_id, &c.model_id, now), self.briefly_unavailable(&c.libp2p_peer_id, &c.model_id, now))
        });
        let discover_ns = t_discover.elapsed().as_nanos() as u64;
        tracing::debug!(elapsed = ?t_discover.elapsed(), candidates = candidates.len(), "discover");
        if candidates.is_empty() {
            // Providers were found, and the floor turned them all away: looking again wouldn't
            // change that (review round 4 M11).
            if !peers.is_empty() && policy.mode == PolicyMode::HardGate {
                return AttemptOutcome::Fatal(no_route_error(model, policy, &peers, self.tier));
            }
            return AttemptOutcome::Retryable(no_route_error(model, policy, &peers, self.tier));
        }
        let total = candidates.len();

        // Try providers in preference order. A dead/stale-but-advertised provider can't
        // hang the request (bounded proxy_forward) and we fail over to the next — but only
        // while nothing has been streamed yet, since partial output can't be un-sent.
        let mut delivered = false;
        // The text the provider actually delivered (every attempt), measured as the provider
        // measures what it sends: bounds the token count we sign for (see `billable_tokens`).
        let delivered_text = std::cell::Cell::new(crate::serve::DeliveredText::default());
        let mut last_err: Option<AdapterError> = None;
        for (i, provider) in candidates.into_iter().enumerate() {
            let _attempt_span = tracing::info_span!(
                "serve_attempt",
                attempt = i + 1,
                provider = %provider.libp2p_peer_id,
            )
            .entered();
            // B-S1: the request commits a fresh receipt nonce up front, so the provider records
            // what it serves under it; the settlement receipt then reuses this exact nonce.
            let request = self.serve_request_for(&provider, messages, max_tokens, temperature, tools, think);
            let provider_libp2p = provider.libp2p_peer_id.clone();
            let t_serve = std::time::Instant::now();
            // P7: on a sample of serves, keep the delivered text to count its tokens afterwards.
            let model_key_now = rep_model_key(&provider.canonical_model_id, &provider.model_id);
            let counter = (!self.is_self_serve(&provider_libp2p)
                && rand::random::<f64>() < crate::token_count::COUNT_SAMPLE_RATE)
                .then(|| self.token_counter_for(&provider, model_key_now))
                .flatten();
            let captured = std::cell::RefCell::new(String::new());
            // Set once the text outgrew `MAX_COUNTED_BYTES`: the rest isn't kept, and the count is
            // estimated from what was (B5).
            let overflowed = std::cell::Cell::new(false);
            // What this attempt received, counted as the provider counts what its polls returned: a
            // stream that ends early is settled for it (see `StreamTally`).
            let tally = std::cell::Cell::new(crate::serve::StreamTally::default());
            // The start of the answer, kept when this serve could be replayed as an audit (D1).
            let replayable = self.replay_eligible(&provider, temperature, messages);
            let answer = std::cell::RefCell::new(String::new());
            // Set once the provider sent far more text than the request allows: the rest isn't read
            // (round 8 G8-3).
            let oversized = std::cell::Cell::new(false);
            let text_cap = delivered_text_cap(max_tokens);
            let exchange = {
                let mut guarded = |d: &str| {
                    if oversized.get() {
                        return;
                    }
                    delivered = true;
                    let mut t = delivered_text.get();
                    t.push(d);
                    delivered_text.set(t);
                    if t.units() > text_cap {
                        oversized.set(true);
                        return;
                    }
                    let mut n = tally.get();
                    n.push_delta(d);
                    tally.set(n);
                    if replayable {
                        keep_prefix(&mut answer.borrow_mut(), d, REPLAY_KEEP_CHARS);
                    }
                    if counter.is_some() && !overflowed.get() {
                        let mut c = captured.borrow_mut();
                        if c.len() + d.len() <= crate::token_count::MAX_COUNTED_BYTES {
                            c.push_str(d);
                        } else {
                            // Too long to keep whole: keep what we have, estimate the rest from it.
                            overflowed.set(true);
                        }
                    }
                    on_delta(d);
                };
                let cut = || cancel() || oversized.get();
                self.exchange(&provider_libp2p, &request, &cut, &mut guarded)
            };
            let opened = exchange.opened;
            let stream_offset = exchange.stream_offset;
            // A serve that had to be sent again went under a fresh nonce: its receipt carries that.
            let nonce = exchange.nonce;
            let mut tally = tally.get();
            tally.push_tool_calls(&exchange.tool_calls);
            let result = if oversized.get() {
                Err(AdapterError::Provider("the provider sent far more text than the request allows".into()))
            } else {
                exchange.result
            };
            let proxy_roundtrip_ns = exchange.proxy_roundtrip_ns;
            match result {
                Ok(mut summary) => {
                    summary.discover_ns = discover_ns;
                    summary.proxy_roundtrip_ns = proxy_roundtrip_ns;
                    // The tier the routing decision saw for this provider, for the client to see.
                    if !self.is_self_serve(&provider_libp2p) {
                        summary.authenticity = Some(self.authenticity_of(&provider.state_id, model_key_now, now));
                    }
                    tracing::debug!(elapsed = ?t_serve.elapsed(), attempt = i + 1, "serve ok");
                    // #7: a serve routed to our OWN co-located provider is not a market
                    // transaction — skip receipt settlement, credit, reputation, and the audit
                    // (auditing yourself is pointless). Still counts as usage below.
                    let self_serve = self.is_self_serve(&provider_libp2p);
                    // Settle the co-signed receipt at EOS (best-effort — tokens already
                    // delivered; a failed/slow settlement must not fail the completion).
                    // A provider following the shared billing rule that reports more than it is
                    // computes to is claiming tokens it didn't serve (P7).
                    let mut over_reported = false;
                    if summary.ok && summary.tokens > 0 && !self_serve {
                        // Sign for what we can vouch for, not whatever the provider reports: the
                        // count it computes from the same text (an honest one reports exactly this).
                        // The usage we report and record is that count too.
                        let delivered = delivered_text.get().units()
                            + crate::serve::tool_call_units(&summary.tool_calls, &crate::serve::offered_tool_names(tools));
                        let tokens = crate::serve::billable_tokens(summary.tokens, max_tokens, delivered);
                        if newer_billing_rule(&summary) {
                            tracing::info!(provider = %provider.peer_id, billing = summary.metrics.billing,
                                "provider bills by a rule newer than this build knows; its count isn't judged");
                            if let Some(stats) = &self.stats {
                                stats.unknown_billing_rules.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                        }
                        if tokens < summary.tokens {
                            over_reported = over_reported_by_billing(&summary, tokens, max_tokens);
                            tracing::warn!(provider = %provider.peer_id, reported = summary.tokens, signable = tokens,
                                billing = summary.metrics.billing, counted_against = over_reported,
                                "provider reported more tokens than it could have served; signing for fewer");
                            summary.tokens = tokens;
                        }
                        self.settle_receipt(&provider, tokens, nonce);
                        self.note_served_paid(&provider.libp2p_peer_id, now);
                        if let Some(counter) = counter {
                            self.check_billed_tokens(
                                counter,
                                captured.take(),
                                overflowed.get().then(|| delivered_text.get().units()),
                                summary.tool_calls.clone(),
                                crate::serve::offered_tool_names(tools),
                                tokens,
                                provider.state_id.clone(),
                                model_key_now.to_string(),
                            );
                        }
                    }
                    if summary.ok && summary.tokens > 0 {
                        self.record_usage(model, &provider.peer_id, summary.tokens, self_serve, now);
                    }
                    if summary.ok && !self_serve {
                        self.note_served_ok(&provider.libp2p_peer_id, &provider.model_id, now);
                        self.note_writer_candidate(&provider, now);
                    }
                    // M2.2(a): a clean served completion earns the provider reputation for THIS
                    // model (Phase 4a) — but not for a self-serve (no earning off yourself).
                    let model_key = rep_model_key(&provider.canonical_model_id, &provider.model_id);
                    if !self_serve {
                        let outcome =
                            if over_reported { VerificationOutcome::Rejected } else { VerificationOutcome::Honored };
                        self.record_outcome(&provider.state_id, model_key, outcome, now);
                    }
                    req_span.record("provider", provider.peer_id.as_str());
                    req_span.record("tokens", summary.tokens);
                    // M2.2(b): sampled background redundant-exec audit of this provider (off
                    // the response path — the caller already has its completion).
                    if !self_serve && provider.swarm_public_key.is_empty() {
                        // What its requests look like, for the probes that audit this model (D1).
                        self.traffic
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .entry(model.to_string())
                            .or_default()
                            .note(
                                temperature,
                                think,
                                max_tokens,
                                messages.first().filter(|m| m.role == "system").map(|m| m.content.as_str()),
                                // Kept only from a request that could have been replayed (R5T-7).
                                replayable,
                            );
                    }
                    let answer = answer.into_inner();
                    if replayable && summary.tool_calls.is_empty() && answer.chars().count() >= MIN_REPLAY_ANSWER_CHARS {
                        let entry = Replayable {
                            provider: provider.clone(),
                            model: model.to_string(),
                            request: AuditRequest { messages: messages.to_vec(), max_tokens, tools: tools.to_vec(), think },
                            answer,
                            at: now,
                        };
                        keep_replayable(&mut self.replays.lock().unwrap_or_else(|e| e.into_inner()), entry, now);
                    }
                    if !self_serve {
                        self.maybe_audit(model, model_key, &provider);
                        // Phase 0 device attestation (opt-in, detached, rate-limited per provider).
                        self.maybe_attest(&provider);
                        // Track A2-Logprob prompt-scoring round (opt-in, detached, every few hours).
                        self.maybe_logprob_check(&provider, model_key);
                    }
                    return AttemptOutcome::Served(summary);
                }
                Err(e) => {
                    // Its client left (review round 4 H1): the provider did nothing wrong.
                    let cancelled = (matches!(e, AdapterError::Cancelled(_)) || cancel()) && !oversized.get();
                    if !cancelled {
                        tracing::warn!(
                            provider = %provider.libp2p_peer_id, attempt = i + 1, total,
                            error = %e, "provider attempt failed"
                        );
                    }
                    // M2.2(a): a failed/refused serve attempt costs the provider
                    // reputation, so a dead/erroring one is downranked on the next discover.
                    // #7: never ding our own provider's reputation on a self-serve.
                    // A provider that charges refusing us over payment, before serving anything, did
                    // nothing wrong: no penalty, but it goes last for a while. A "refusal" after
                    // output was delivered is a failed serve like any other.
                    let refused = !delivered && crate::serve::is_payment_refusal(&e);
                    if refused {
                        self.note_payment_refusal(&provider.libp2p_peer_id, &provider.model_id, now);
                    }
                    // A shed (its pool was full) or a provider we couldn't reach, before anything was
                    // served, goes last for a minute; only a second miss within ten minutes costs it
                    // reputation, and a shed never does: load others cause isn't the provider's fault
                    // (round 8 T8-3).
                    let shed = !delivered && matches!(e, AdapterError::Overloaded(_));
                    let unreachable = !delivered && matches!(e, AdapterError::Network(_));
                    let missed_again = (shed || unreachable) && self.note_unavailable(&provider.libp2p_peer_id, &provider.model_id, now);
                    let excused = shed || (unreachable && !missed_again);
                    if !self.is_self_serve(&provider.libp2p_peer_id) && !refused && !cancelled && !excused {
                        let model_key = rep_model_key(&provider.canonical_model_id, &provider.model_id);
                        self.record_outcome(&provider.state_id, model_key, VerificationOutcome::Rejected, now);
                    }
                    // A stream that broke after output arrived is still owed for what we read: the
                    // provider counts what its polls returned the same way, and holds an unpaid
                    // part against us like an unpaid whole.
                    // Closed first (S8): the provider sends back anything it counted that we missed,
                    // and stops generating; then it is paid for exactly that, in the background.
                    // Closed whether or not anything reached us (review round 4 H7): a poll whose
                    // reply was lost is counted by the provider all the same, and the close brings
                    // those frames back so they're paid for rather than held against us. Only a
                    // stream the provider opened (R5S-3): a refused or unanswered open has none.
                    if opened && !self.is_self_serve(&provider.libp2p_peer_id) {
                        let tokens = tally.billable(max_tokens);
                        tracing::info!(provider = %provider.peer_id, tokens, "stream ended early; closing it and settling for what was received");
                        self.close_and_settle(&provider, nonce, stream_offset, tally, max_tokens);
                        if tokens > 0 {
                            self.record_usage(model, &provider.peer_id, tokens, false, now);
                        }
                    } else if opened {
                        // Our own provider: nothing to settle, but it stops generating for nobody
                        // (round 8 G8-2).
                        self.close_own_stream(&provider, nonce, stream_offset);
                    } else if !self.is_self_serve(&provider.libp2p_peer_id) {
                        // A buffered serve that failed after its text reached us: owed for that text,
                        // which the provider counts the same way (round 8 S8-2).
                        let tokens = tally.billable(max_tokens);
                        if tokens > 0 {
                            tracing::info!(provider = %provider.peer_id, tokens, "serve failed part-way; settling for what was received");
                            self.settle_receipt(&provider, tokens, nonce);
                            self.record_usage(model, &provider.peer_id, tokens, false, now);
                        }
                    }
                    // Its client left: nobody to serve, so no failover.
                    if delivered || cancel() {
                        // Already streamed part of a completion to the client — failing over
                        // (or a cache-bypass retry) would duplicate output. Surface it as fatal.
                        return AttemptOutcome::Fatal(e);
                    }
                    last_err = Some(e);
                }
            }
        }
        // Every candidate failed, but nothing was streamed → safe for the caller to bypass the
        // discovery cache and retry once.
        AttemptOutcome::Retryable(
            last_err.unwrap_or_else(|| AdapterError::Route(format!("all providers failed for '{model}'"))),
        )
    }

    /// The request a serve of `messages` at `provider` sends, with a fresh receipt nonce. Every serve
    /// is built here, a user's and an audit's alike, so an audit can't differ from a user's request
    /// in any field (review round 3, D1).
    fn serve_request_for(
        &self,
        provider: &SelectedProvider,
        messages: &[ChatMessage],
        max_tokens: Option<u32>,
        temperature: Option<f64>,
        tools: &[Value],
        think: Option<bool>,
    ) -> ServeRequest {
        ServeRequest {
            reply_to: self.net.libp2p_peer_id().to_string(),
            model_ref: provider.model_id.clone(),
            messages: messages.to_vec(),
            max_tokens,
            temperature,
            tools: tools.to_vec(),
            think,
            nonce: rand::random::<[u8; 16]>(),
            // M4-base: present our swarm membership credential ONLY on a private route — a
            // candidate that came from a private card names the swarm it's gated on
            // (`provider.swarm_public_key`), and we attach the matching credential. This both
            // reaches the private model AND avoids leaking swarm membership to public/global
            // providers (we never attach a credential to a public serve). `None` otherwise.
            credential: if provider.swarm_public_key.is_empty() {
                None
            } else {
                self.creds
                    .as_ref()
                    .and_then(|c| c.credential_for_swarm(&provider.swarm_public_key, now_unix_ms()))
            },
            // Every serve keeps the caller's own sampling; deterministic mode is never sent, so a
            // request carrying it would stand out.
            deterministic: None,
            // On the crypto rail, offer to pay at our peg (a provider that charges needs it).
            settlement_peg: self.settlement_peg,
            settlement_domain: self.settlement_chain.as_ref().map(|c| c.domain.fingerprint()),
            settlement_wallet: self.settlement_wallet(),
        }
    }

    /// Send `request` to `provider_libp2p` the way every serve travels: as a stream (polled), or as
    /// one buffered reply to a provider known not to stream, with reconnect-and-fetch if that reply's
    /// connection drops. Each text fragment goes to `on_delta` as it arrives.
    fn exchange(
        &self,
        provider_libp2p: &str,
        request: &ServeRequest,
        cancel: &dyn Fn() -> bool,
        on_delta: &mut dyn FnMut(&str),
    ) -> Exchange {
        // Budget the serve by the request's `max_tokens` (see `attempt_timeout`) so a big
        // agentic completion on a slow remote model isn't abandoned mid-generation.
        let serve_to = attempt_timeout(request.max_tokens);
        // Shared across the (sequential) buffered + streaming transports via a Cell so both
        // can record it without a mutable-borrow conflict. Holds the first-ack / round-trip RTT.
        let proxy_rt = std::cell::Cell::new(0u64);
        let mut progress = StreamProgress::default();
        // The nonce the serve ends under (a fresh one if it had to be served again, R5S-4).
        let nonce = std::cell::Cell::new(request.nonce);
        // Buffered (0x10) round-trip with 0x11 reconnect-fetch drop recovery — the proven
        // path, used directly when a provider is known buffered-only and as the fallback
        // when a streaming probe reveals a legacy provider.
        let mut buffered = |framed: &[u8]| -> Result<Vec<u8>, AdapterError> {
            let t = std::time::Instant::now();
            let r = self.net.proxy_forward_timeout(provider_libp2p.to_string(), framed.to_vec(), serve_to);
            proxy_rt.set(t.elapsed().as_nanos() as u64);
            match r {
                Ok(bytes) => Ok(bytes),
                // The serve connection dropped (e.g. relay/NAT eviction during a long
                // generation). The response is single-shot, so nothing was delivered
                // yet — recover the completed work by fetching it by nonce on a fresh
                // circuit rather than failing. See RECONNECT_AND_FETCH_PLAN.md.
                Err(drop_err) => self
                    .reconnect_and_fetch(provider_libp2p, request, serve_to, &nonce)
                    .map_err(|fe| AdapterError::Network(format!("proxy_forward: {drop_err}; reconnect-fetch: {fe}"))),
            }
        };
        let result = if self.is_buffered_only(provider_libp2p) {
            request_completion(&mut buffered, request, on_delta)
        } else {
            // P1c: try streaming first. A raw round-trip (no 0x11 — streaming re-polls the
            // same offset itself for drop-resume); the first-ack RTT is recorded as
            // proxy_roundtrip_ns. Real thread::sleep paces the poll loop live.
            let stream_out = {
                let mut stream_transport = |framed: &[u8]| -> Result<Vec<u8>, AdapterError> {
                    let t = std::time::Instant::now();
                    let r = self.net.proxy_forward_timeout(provider_libp2p.to_string(), framed.to_vec(), STREAM_CALL_TIMEOUT);
                    if proxy_rt.get() == 0 {
                        proxy_rt.set(t.elapsed().as_nanos() as u64); // first-ack RTT
                    }
                    r.map_err(AdapterError::Network)
                };
                let mut sleeper = |d: Duration| std::thread::sleep(d);
                request_completion_streaming_with(&mut stream_transport, &mut sleeper, request, cancel, &mut progress, on_delta)
            };
            match stream_out {
                Ok(s) => Ok(s),
                Err(StreamFail::Error(e)) => Err(e),
                Err(StreamFail::Cancelled) => Err(AdapterError::Cancelled("the client stopped reading".into())),
                // A legacy provider — remember it and fall back to buffered this same
                // attempt (the probe ran no generation, so no work is wasted).
                Err(StreamFail::Legacy) => {
                    self.mark_buffered_only(provider_libp2p);
                    tracing::debug!(provider = %provider_libp2p, "provider is buffered-only; falling back to SERVE_REQUEST");
                    request_completion(&mut buffered, request, on_delta)
                }
            }
        };
        Exchange {
            result,
            proxy_roundtrip_ns: proxy_rt.get(),
            opened: progress.opened,
            stream_offset: progress.offset,
            tool_calls: progress.tool_calls,
            nonce: nonce.get(),
        }
    }

    /// Settle a stream that ended early (S8), in the background: first close it (`STREAM_CLOSE`), so
    /// the provider stops generating and sends back whatever it counted past `offset` that we never
    /// received (a lost reply), then pay for exactly what we then hold (`tally` plus those frames),
    /// which is what the provider counted. Retried while the connection recovers, within the
    /// commitment's lifetime; a provider that doesn't know the close (older build), or no longer
    /// holds the stream, is paid for `tally` as before.
    fn close_and_settle(
        self: &Arc<Self>,
        provider: &SelectedProvider,
        nonce: [u8; 16],
        offset: u32,
        mut tally: crate::serve::StreamTally,
        max_tokens: Option<u32>,
    ) {
        use std::sync::atomic::Ordering;
        if self.closes_in_flight.fetch_add(1, Ordering::AcqRel) >= MAX_CLOSES_IN_FLIGHT {
            self.closes_in_flight.fetch_sub(1, Ordering::AcqRel);
            let tokens = tally.billable(max_tokens);
            if tokens > 0 {
                self.settle_receipt(provider, tokens, nonce);
            }
            return;
        }
        let slot = SettleSlot(Arc::clone(&self.closes_in_flight));
        let node = Arc::clone(self);
        let provider = provider.clone();
        let ceiling = tally.billable(max_tokens).saturating_add(close_tail_cap(max_tokens));
        std::thread::spawn(move || {
            let _slot = slot;
            let mut close = vec![crate::provider::STREAM_CLOSE];
            close.extend_from_slice(&nonce);
            close.extend_from_slice(&offset.to_be_bytes());
            for wait in CLOSE_RETRY_WAITS {
                std::thread::sleep(wait);
                let reply = node.net.proxy_forward_timeout(provider.libp2p_peer_id.clone(), close.clone(), SETTLE_TIMEOUT);
                match reply.map(|r| FetchChunksResponse::decode(&r)) {
                    Ok(Ok(FetchChunksResponse::Chunks { framed, .. })) => {
                        absorb_close_reply(&mut tally, &framed, max_tokens);
                        break;
                    }
                    // Throttled: try again shortly.
                    Ok(Ok(FetchChunksResponse::Overloaded)) => continue,
                    // Gone (settled, expired), not ours, or a provider without the close: pay for
                    // what we have.
                    Ok(_) => break,
                    // Unreachable: retry, in case the connection comes back.
                    Err(e) => tracing::debug!(provider = %provider.peer_id, error = %e, "couldn't close the stream yet"),
                }
            }
            // The close adds at most its cap to the bill, however large the frames it returned (a
            // `Done` bounds the count by the text's size, and fragments can be any size).
            let tokens = tally.billable(max_tokens).min(ceiling);
            if tokens > 0 {
                node.settle_receipt(&provider, tokens, nonce);
            }
        });
    }

    /// Close a stream on our own provider that ended early (its client left): it stops generating.
    /// Nothing is settled; tried twice, in the background.
    fn close_own_stream(self: &Arc<Self>, provider: &SelectedProvider, nonce: [u8; 16], offset: u32) {
        use std::sync::atomic::Ordering;
        if self.closes_in_flight.fetch_add(1, Ordering::AcqRel) >= MAX_CLOSES_IN_FLIGHT {
            self.closes_in_flight.fetch_sub(1, Ordering::AcqRel);
            return;
        }
        let slot = SettleSlot(Arc::clone(&self.closes_in_flight));
        let node = Arc::clone(self);
        let peer = provider.libp2p_peer_id.clone();
        std::thread::spawn(move || {
            let _slot = slot;
            let mut close = vec![crate::provider::STREAM_CLOSE];
            close.extend_from_slice(&nonce);
            close.extend_from_slice(&offset.to_be_bytes());
            for wait in &CLOSE_RETRY_WAITS[..2] {
                std::thread::sleep(*wait);
                if node.net.proxy_forward_timeout(peer.clone(), close.clone(), SETTLE_TIMEOUT).is_ok() {
                    break;
                }
            }
        });
    }

    /// #7/#5: per-model consumed tracking (pure usage, kept for a self-serve too), and a `used`
    /// give/take ledger row for a real counterparty (`provider_peer`), in memory and in the store (the
    /// consumer's side isn't a receipt, so it lives only here) so the totals survive a restart.
    fn record_usage(&self, model: &str, provider_peer: &str, tokens: u64, self_serve: bool, now: u64) {
        if let Some(stats) = &self.stats {
            stats.record_consume(model, tokens);
            if !self_serve {
                stats.record_ledger(now, "used", model, provider_peer, tokens);
            }
        }
        if !self_serve {
            if let Some(store) = &self.store {
                let _ = store.append_ledger_row(&openhydra_protocol::store::LedgerEntry {
                    ts_ms: now,
                    kind: "used".to_string(),
                    model: model.to_string(),
                    counterparty: provider_peer.to_string(),
                    tokens,
                });
            }
        }
    }

    /// Reconnect-and-fetch (RECONNECT_AND_FETCH_PLAN.md): the serve connection to `provider`
    /// dropped mid-request (e.g. the relay/NAT mapping was evicted during a long generation).
    /// Poll the provider for the buffered result **by nonce** on fresh circuits — each
    /// `proxy_forward` re-dials — backing off while the relay re-establishes or the provider
    /// is still generating, until the recovery deadline (`serve_timeout + FETCH_RECONNECT_GRACE`).
    /// `NotFound` (provider restarted / buffer
    /// TTL expired) → re-serve the original request once; deterministic engines cache-hit.
    /// Returns the framed serve response for [`request_completion`] to parse, exactly as a
    /// first-try round-trip would.
    fn reconnect_and_fetch(
        &self,
        provider: &str,
        request: &ServeRequest,
        serve_timeout: std::time::Duration,
        nonce_out: &std::cell::Cell<[u8; 16]>,
    ) -> Result<Vec<u8>, String> {
        let mut fetch_req = Vec::with_capacity(1 + request.nonce.len());
        fetch_req.push(FETCH_RESULT);
        fetch_req.extend_from_slice(&request.nonce);
        // Keep re-polling for the buffered result long enough for a generation the consumer
        // reconnected into to finish (`serve_timeout`) plus the relay reconnect window.
        let deadline = std::time::Instant::now() + serve_timeout + FETCH_RECONNECT_GRACE;
        loop {
            if std::time::Instant::now() >= deadline {
                return Err("result not recovered before deadline".into());
            }
            match self.net.proxy_forward_timeout(
                provider.to_string(),
                fetch_req.clone(),
                FETCH_TIMEOUT,
            ) {
                Ok(resp) => match FetchResponse::decode(&resp) {
                    Ok(FetchResponse::Ready(framed)) => {
                        tracing::debug!(provider = %provider, "reconnect-fetch recovered result");
                        return Ok(framed);
                    }
                    // Still generating — wait out the compute, then re-fetch.
                    Ok(FetchResponse::Generating) => std::thread::sleep(FETCH_BACKOFF),
                    // Provider no longer holds it → re-serve once (idempotent for deterministic
                    // engines; the ComfyUI /history cache makes the re-run instant), under a fresh
                    // nonce: a provider serves each nonce once (review round 5, R5S-4).
                    Ok(FetchResponse::NotFound) => {
                        tracing::debug!(provider = %provider, "reconnect-fetch NotFound → re-serve");
                        let mut again = request.clone();
                        again.nonce = rand::random();
                        nonce_out.set(again.nonce);
                        let mut framed = Vec::with_capacity(1 + again.messages.len() * 32);
                        framed.push(SERVE_REQUEST);
                        framed.extend_from_slice(&again.encode());
                        return self
                            .net
                            .proxy_forward_timeout(provider.to_string(), framed, serve_timeout)
                            .map_err(|e| format!("re-serve: {e}"));
                    }
                    Ok(FetchResponse::Forbidden) => return Err("fetch forbidden".into()),
                    Err(e) => return Err(format!("bad fetch response: {e}")),
                },
                // Transport still down (relay re-establishing) → back off and retry the fetch.
                Err(_) => std::thread::sleep(FETCH_BACKOFF),
            }
        }
    }

    /// Fire the co-signed receipt for a completed request. Skips a provider that
    /// advertised no usable public key; swallows all errors (trust settlement is
    /// auxiliary to delivering the completion).
    fn settle_receipt(self: &Arc<Self>, provider: &SelectedProvider, tokens: u64, nonce: [u8; 16]) {
        // Cheap up-front checks on the caller thread so an unkeyed provider costs no thread.
        let provider_pub = match hex::decode(&provider.public_key) {
            Ok(b) if b.len() == 32 => b,
            _ => return, // legacy / unkeyed provider — nothing to settle against
        };
        let consumer_pub = match self.net.public_key_bytes() {
            Ok(b) => b,
            Err(_) => return,
        };
        // G5: settlement is a *follow-up* round-trip to the provider, and `complete` emits the
        // terminal `Done`/`[DONE]` only after it returns. Run inline, it let a provider that
        // served correctly but then black-holes the RECEIPT_REQUEST pin the consumer's response
        // thread and withhold `[DONE]` for the full protocol timeout — a flood → unbounded
        // parked threads. Move it fully off the response path (detached, like `maybe_audit`) and
        // bound the transport with `SETTLE_TIMEOUT` so a dead provider can't park the thread
        // indefinitely. Credit is advisory, so a dropped settlement never affects the completion.
        let node = Arc::clone(self);
        let provider_libp2p = provider.libp2p_peer_id.clone();
        let model_id = provider.model_id.clone();
        // `canonical` is still carried in the receipt (records what the provider claimed, for future
        // fingerprint verification); `model_key` is the validated handle used for reputation + the
        // cost-weighted display credit (the identity the audit actually exercises — review H1/H3).
        let canonical = provider.canonical_model_id.clone();
        let model_key = rep_model_key(&provider.canonical_model_id, &provider.model_id).to_string();
        // The provider's 128-bit state id — keys the per-(peer, model) gated voucher + authenticity lookup.
        let peer_id = provider.state_id.clone();
        // The voucher is keyed by the provider's full public key — what the on-chain claim (and its
        // already-settled amount) is keyed by — not just its 64-bit OpenHydra id: two keys sharing an
        // id (a ~2^32 search) would otherwise share one running total, and each claim all of it
        // on-chain.
        // Scoped to the deployment paying receipts settle on: each deployment's escrow starts from
        // nothing, so a running total built under one must not carry over (the first claim on another
        // would re-authorize the whole history).
        // On Base, by our wallet too: on-chain, what was paid so far is kept per wallet, so a new
        // wallet's running total starts from nothing (review round 3, S1).
        let vkey = format!("{}{}", voucher_key(&peer_id, &model_key, &provider_pub), self.voucher_scope());
        // Review F1: serialize settlement for this `(peer, model)` across the whole
        // read→co-sign→commit window (which spans a network round-trip), so concurrent settles to the
        // same pair can't read the same base and emit non-monotonic / divergent co-signed cumulatives.
        // Different pairs still settle in parallel. Taken here so its holders (the map's own
        // reference aside) say how many settlements this pair already has running or queued.
        let key_lock = node.settle_lock_for(&vkey);
        let in_flight = node.settles_in_flight.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let slot = SettleSlot(Arc::clone(&node.settles_in_flight));
        if !settle_admitted(in_flight, Arc::strong_count(&key_lock).saturating_sub(2)) {
            tracing::warn!(provider = %provider.peer_id, in_flight, tokens,
                "too many settlements in flight; dropping this one (the serve stays unpaid)");
            return;
        }
        std::thread::spawn(move || {
            let _slot = slot;
            let sign = |msg: &[u8]| node.net.sign(msg).unwrap_or_default();
            let mut transport = |framed: &[u8]| -> Result<Vec<u8>, AdapterError> {
                node.net
                    .proxy_forward_timeout(provider_libp2p.clone(), framed.to_vec(), SETTLE_TIMEOUT)
                    .map_err(AdapterError::Http)
            };
            let _settle_guard = key_lock.lock().unwrap_or_else(|e| e.into_inner());
            // Stamped once this settlement's turn comes, not when it was queued: a receipt that
            // waited behind others for its pair must still be fresh when the provider checks it.
            let now = now_unix_ms();

            // Crypto-rail (off-chain), only when a settlement peg is configured (review F3): compute
            // this serve's authenticity-gated split and the resulting cumulative voucher amount
            // (previous cumulative + this serve's gross), stamped BEFORE the consumer signs so both
            // parties co-sign the same claim. The voucher is committed only after a successful
            // co-sign, keeping it equal to the latest co-signed cumulative. No value moves.
            // Built on the highest cumulative we ever signed for the pair (co-signed or not: anything
            // signed can be claimed), so a lost reply or a lost voucher never shortchanges the provider.
            let compute_voucher = |base: u64| {
                node.settlement_peg.map(|peg| {
                    let split = node.gated_split_for(&peer_id, &model_key, tokens, now, peg);
                    (split, base.saturating_add(split.net_to_provider + split.protocol_fee))
                })
            };
            // On a settlement chain: the receipt's on-chain claim, valid for CLAIM_LIFETIME_SECS.
            let chain = node.settlement_chain.clone();
            let sign_claim = |claim: &openhydra_protocol::settlement::SettlementClaim| -> Vec<u8> {
                use openhydra_protocol::settlement::ChainDomain;
                match chain.as_ref().map(|c| (&c.domain, c.evm_wallet.as_deref())) {
                    Some((d @ ChainDomain::Solana(_), _)) => node.net.sign(&d.claim_message(claim)).unwrap_or_default(),
                    Some((ChainDomain::Base(d), Some(wallet))) => {
                        openhydra_protocol::settlement::sign_evm_claim(wallet, d, claim).to_vec()
                    }
                    _ => Vec::new(),
                }
            };
            let expires_at = now / 1000 + openhydra_protocol::settlement::CLAIM_LIFETIME_SECS;
            let claim_signer = chain.as_ref().map(|_| {
                (expires_at, &sign_claim as &dyn Fn(&openhydra_protocol::settlement::SettlementClaim) -> Vec<u8>)
            });
            // Sent again (same receipt) when the round-trip fails or the provider asks us to retry.
            let mut settle = |voucher: Option<(FeeSplit, u64)>| {
                if let Some((_, cumulative)) = voucher {
                    node.note_signed(&vkey, cumulative);
                }
                let attempt = |transport: &mut dyn FnMut(&[u8]) -> Result<Vec<u8>, AdapterError>| request_receipt(
                    &sign,
                    transport,
                    &provider_pub,
                    &consumer_pub,
                    &model_id,
                    tokens,
                    nonce,
                    now,
                    &canonical,
                    voucher.map(|(_, cumulative)| cumulative),
                    // v5 denomination binding: when the crypto rail is on, stamp the asset (USDC
                    // today) + the exact peg used, so the provider co-signs — and can reject — the
                    // claim's units.
                    voucher.map(|_| SettlementAsset::Usdc),
                    node.settlement_peg,
                    claim_signer,
                );
                let mut outcome = attempt(&mut transport);
                for wait in SETTLE_RETRY_WAITS {
                    match &outcome {
                        Err(ReceiptFailure::Failed(e)) if settle_retryable(e) => {
                            std::thread::sleep(wait);
                            outcome = attempt(&mut transport);
                        }
                        _ => break,
                    }
                }
                outcome
            };
            let base = node.settle_base(&vkey);
            let mut voucher = compute_voucher(base);
            let mut outcome = settle(voucher);
            // Behind the provider's high-water: catch up to the receipt it sent back (ours, by its
            // signature) and sign again on top of it — once.
            let mut settled_earlier = false;
            if let Err(ReceiptFailure::Behind(latest)) = &outcome {
                let signed_here = |l: &crate::receipt::LatestReceipt| signed_for_deployment(chain.as_ref(), l);
                let cx = BehindContext {
                    consumer_pub: &consumer_pub,
                    provider_pub: &provider_pub,
                    model_id: &model_id,
                    nonce: &nonce,
                    base,
                    peg: node.settlement_peg,
                    signed_here: &signed_here,
                };
                match judge_behind_receipt(latest, &cx) {
                    BehindReceipt::ThisServe(c) => {
                        node.raise_gated_voucher(&vkey, c);
                        settled_earlier = true;
                    }
                    BehindReceipt::Adopt(c) => {
                        tracing::info!(provider = %peer_id, from = base, to = c,
                            "settlement was behind the provider's last co-signed receipt; caught up");
                        node.raise_gated_voucher(&vkey, c);
                        // On top of the provider's receipt, not of the one we just signed: that one
                        // already holds this serve.
                        voucher = compute_voucher(c);
                        outcome = settle(voucher);
                    }
                    other => tracing::warn!(provider = %peer_id, ?other,
                        "provider sent back a receipt this consumer can't adopt"),
                }
            }
            // An older provider (v0.3.20) reads only the original framing, which can't carry the
            // canonical id: a plain receipt it refused for its length is sent once more without it
            // (round 8 X8-3).
            let legacy_framing = |o: &Result<openhydra_protocol::receipts::CoSignedReceipt, ReceiptFailure>| {
                matches!(o, Err(ReceiptFailure::Failed(e)) if e.message().contains("length mismatch"))
            };
            if voucher.is_none() && claim_signer.is_none() && !canonical.is_empty() && legacy_framing(&outcome) {
                outcome = request_receipt(
                    &sign, &mut transport, &provider_pub, &consumer_pub, &model_id, tokens, nonce, now, "", None, None, None, None,
                );
            }
            // On a *successfully co-signed* receipt the provider has served us `tokens` — credit
            // its contribution (M2.3 give-side), the mirror of the provider's take-side accrual.
            node.note_settled(&provider_libp2p, settled_earlier || outcome.is_ok());
            match outcome {
                _ if settled_earlier => {
                    node.record_contribution(&provider_libp2p, &peer_id, &model_key, tokens);
                }
                Ok(receipt) => {
                    tracing::debug!(provider = %peer_id, tokens, cumulative = ?receipt.payload.cumulative_amount,
                        "receipt co-signed");
                    node.record_contribution(&provider_libp2p, &peer_id, &model_key, tokens);
                    if let Some((_, cumulative)) = voucher {
                        node.raise_gated_voucher(&vkey, cumulative);
                    }
                }
                Err(ReceiptFailure::Failed(e)) => {
                    tracing::warn!(provider = %peer_id, tokens, error = %e, "receipt not co-signed");
                }
                Err(ReceiptFailure::Behind(_)) if !settled_earlier => {
                    tracing::warn!(provider = %peer_id, tokens, "receipt still behind the provider after catching up");
                }
                Err(ReceiptFailure::Behind(_)) => {}
            }
        });
    }

    /// Record that a provider **served** us `tokens` (give-side, M2.3): the provider earns give/take
    /// throttle credit in our ledger, keyed by its libp2p id, in **raw tokens** — the throttle is
    /// bandwidth priority and its constants are token-scaled (review H2). Separately, the
    /// cost-weighted "credits spent" display counter is incremented, weighted by the validated
    /// handle. The per-counterparty cap still bounds collusion. Best-effort persisted; a failed
    /// write never affects the completion.
    ///
    /// `model_key` is the provider's validated engine **handle** (= [`rep_model_key`]) — it keys the
    /// per-`(peer, model)` reputation and sizes the cost-weighted display credit (the handle-implied
    /// tier is the ceiling until model-authenticity fingerprinting lands; a self-declared canonical
    /// can't inflate it — review H1/H3). The give-side receipt carries no prompt/completion split
    /// yet (Phase 2b) → the blended in/out rate.
    fn record_contribution(&self, provider_libp2p_id: &str, peer_id: &str, model_key: &str, tokens: u64) {
        let now = now_unix_ms();
        let me = self.net.libp2p_peer_id().to_string();

        // Give/take throttle = BANDWIDTH priority, in RAW tokens (review H2: decoupled from cost-tier
        // weighting — `STARTER_GRANT`/`PER_COUNTERPARTY_CAP`/`DEFICIT_SPAN` are token-scaled, and
        // feeding 150× weighted amounts through them truncated earns and over-throttled consumers).
        let snapshot = {
            let mut map = match self.credit.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            let acct = map
                .entry(provider_libp2p_id.to_string())
                .or_insert_with(|| CreditAccount::new(now));
            acct.record_served(&me, tokens, now);
            acct.to_bytes()
        };
        // Display: cost-weighted credit SPENT, weighted by the **validated handle** (`model_key`) —
        // the ceiling that stops a mislabelled canonical inflating the number.
        if let Some(stats) = &self.stats {
            let spent = credits_for_receipt(self.claimed_tier(peer_id, model_key, now).weight(), tokens, None);
            stats.add_credits_spent(spent);
        }
        if let Some(store) = &self.store {
            if let Err(e) = store.put_credit(provider_libp2p_id, &snapshot) {
                tracing::debug!(error = %e, provider_libp2p_id, "credit persist failed");
            }
        }
    }

    /// The give/take **balance** we hold for a provider (by libp2p id), decayed to now —
    /// `STARTER_GRANT` when we've no record. Positive ⇒ a net contributor to us. Exposed for
    /// observability and tests; the provider role consults its own map's `rate_cap` to
    /// throttle (enforcement is the concurrency-gated step).
    pub fn provider_balance(&self, provider_libp2p_id: &str) -> f64 {
        let now = now_unix_ms();
        self.credit
            .lock()
            .ok()
            .and_then(|m| m.get(provider_libp2p_id).map(|a| a.balance(now)))
            .unwrap_or(openhydra_protocol::credit::STARTER_GRANT)
    }

    /// The redundant-execution audit sampling rate for `peer_id` (M2.2(b)): low for a
    /// trusted provider, high for a fresh / low-reputation one. A background sampler draws
    /// against this to decide whether to audit a provider after it served.
    /// Pure read of the earned-reputation map — exposed so the trigger policy is testable
    /// and tunable separately from the dispatch.
    pub fn audit_rate_for(&self, peer_id: &str, model_key: &str, now_ms: u64) -> f64 {
        // A provider with a strike open is audited at the highest rate until it is resolved (R5T-2).
        let struck = self
            .strikes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&reputation_key(peer_id, model_key))
            .is_some_and(|s| now_ms.saturating_sub(s.at) <= STRIKE_WINDOW_MS);
        if struck {
            return AUDIT_MAX_RATE;
        }
        let rep = self.earned_reputation(peer_id, model_key, now_ms).unwrap_or(NEUTRAL_REPUTATION);
        sample_rate_for_reputation(rep, AUDIT_MIN_RATE, AUDIT_MAX_RATE)
    }

    /// Device attestation (Phase 0): if enabled and this provider is due, challenge it in a detached
    /// thread (bounded by [`ATTEST_TIMEOUT`]) and record the outcome. Never touches the completion.
    fn maybe_attest(self: &Arc<Self>, provider: &SelectedProvider) {
        let Some(attest) = self.provider_attest.clone() else { return };
        let now = now_unix_ms();
        let prev = attest.records.lock().unwrap_or_else(|e| e.into_inner()).get(&provider.state_id).cloned();
        if prev.as_ref().is_some_and(|r| !r.due(now)) {
            return;
        }
        if !attest.inflight.lock().unwrap_or_else(|e| e.into_inner()).insert(provider.state_id.clone()) {
            return; // a challenge to this provider is already running
        }
        let node = Arc::clone(self);
        let peer_id = provider.state_id.clone();
        let libp2p = provider.libp2p_peer_id.clone();
        let model = provider.model_id.clone();
        std::thread::spawn(move || {
            let mut transport = |framed: &[u8]| -> Result<Vec<u8>, String> {
                node.net.proxy_forward_timeout(libp2p.clone(), framed.to_vec(), ATTEST_TIMEOUT)
            };
            let record = attest_provider(
                &mut transport,
                &model,
                rand::random(),
                now_unix_ms(),
                &attest.trusted_binaries,
                prev.as_ref(),
                // The peer we challenge: its claim must name it, not another provider it relays.
                &libp2p,
            );
            if record.key_rotations > prev.as_ref().map_or(0, |p| p.key_rotations) {
                tracing::warn!(provider = %peer_id, rotations = record.key_rotations,
                    "provider attestation key changed (expected on restart for unsigned builds)");
            }
            tracing::info!(provider = %peer_id, model, outcome = ?record.outcome, "device attestation");
            let mut records = attest.records.lock().unwrap_or_else(|e| e.into_inner());
            crate::attest::insert_record_bounded(&mut records, peer_id.clone(), record, now_unix_ms());
            drop(records);
            attest.inflight.lock().unwrap_or_else(|e| e.into_inner()).remove(&peer_id);
        });
    }

    /// Track A2-Logprob: if checks are on, a reference set covers this provider's model handle, and
    /// the `(peer, model)` is due, run a challenge round in a detached thread and fold the verdict
    /// into the audit history + reputation. Never touches the completion.
    fn maybe_logprob_check(self: &Arc<Self>, provider: &SelectedProvider, model_key: &str) {
        let Some(lp) = self.provider_logprob.clone() else { return };
        let key = reputation_key(&provider.state_id, model_key);
        let now = now_unix_ms();
        // Not due yet: skip, except for an occasional spot check of a provider that passed (LP7).
        let skip = lp.status.lock().unwrap_or_else(|e| e.into_inner()).get(&key).is_some_and(|s| {
            now < s.next_due_ms
                && !(s.verdict.is_some_and(|(v, _)| v.passed())
                    && rand::random::<f64>() < crate::prompt_score::LOGPROB_SPOT_CHECK_RATE)
        });
        if skip {
            return;
        }
        if !lp.inflight.lock().unwrap_or_else(|e| e.into_inner()).insert(key.clone()) {
            return; // a round for this (peer, model) is already running
        }
        let node = Arc::clone(self);
        let peer_id = provider.state_id.clone();
        let libp2p = provider.libp2p_peer_id.clone();
        let model = provider.model_id.clone();
        let model_key = model_key.to_string();
        std::thread::spawn(move || {
            // Which exact model is this? The provider declares its identity (engine-reported family /
            // size / precise quant); the reference for that identity is used, else one listing the
            // model's name. A legacy provider can't say — its name is all there is.
            let mut info_transport = |framed: &[u8]| -> Result<Vec<u8>, String> {
                node.net.proxy_forward_timeout(libp2p.clone(), framed.to_vec(), LOGPROB_TIMEOUT)
            };
            // …and which exact weights file: references made from a file apply only to that file.
            // Whether the provider answered (a busy shed, a rate limit or a lost connection isn't an
            // answer, round 8 L8-3).
            let mut info_answered = true;
            let (declared, weights) = match request_model_info(&mut info_transport, &model) {
                Ok(info) => (info.identity(), info.weights()),
                Err(ScoreChallengeError::Unsupported) => (None, None),
                Err(ScoreChallengeError::Refused(r)) => {
                    info_answered = !r.contains("retry");
                    (None, None)
                }
                Err(e) => {
                    tracing::debug!(provider = %peer_id, model, error = ?e, "model-info request failed");
                    info_answered = false;
                    (None, None)
                }
            };
            // Not answered (shed, rate limited, lost): without its identity the wrong reference could
            // be picked; the run waits for the next one (round 9 P9-3).
            if !info_answered {
                let round = lp.rounds.lock().unwrap_or_else(|e| e.into_inner()).remove(&key).unwrap_or_default();
                node.settle_logprob_round(&lp, &key, &peer_id, &model, &model_key, RoundOutcome::Busy, round, 0, None);
                return;
            }
            if let (Some(hint), Some(id)) = (quant_hint_from_name(&model), declared.as_ref()) {
                if hint != id.quant {
                    tracing::warn!(provider = %peer_id, model, name_says = hint, declared = id.quant,
                        "model name and declared quantisation disagree; checking the declared one");
                }
            }
            // Verified on one weights file, now declaring another or none: that pass doesn't carry over.
            let current = lp.status.lock().unwrap_or_else(|e| e.into_inner()).get(&key).map(|st| st.reference);
            let verified_on = current.filter(|t| *t != [0; 16]).and_then(|t| lp.tied_weights(t));
            if let Some(verified_on) = verified_on.filter(|w| weights.as_deref() != Some(w.as_str())) {
                let mut round = lp.rounds.lock().unwrap_or_else(|e| e.into_inner()).remove(&key).unwrap_or_default();
                if weights.is_none() {
                    // None declared: an agent hashes the file in the background after a restart, so
                    // retry soon (and keep the streak); one that keeps not declaring it is dodging.
                    tracing::info!(provider = %peer_id, model, verified_on, "provider no longer declares the weights file it was verified on");
                    let outcome = RoundOutcome::Undeclared;
                    node.settle_logprob_round(&lp, &key, &peer_id, &model, &model_key, outcome, round, 0, None);
                    return;
                }
                tracing::info!(provider = %peer_id, model, verified_on, now = weights.as_deref().unwrap_or_default(),
                    "provider declares a different weights file than it was verified on; its pass is void");
                // A text it was stalling on, dropped by its switch, counts as declined (round 9 P9-5).
                let dropped = round.pending_id().is_some();
                let voided = {
                    let mut status = lp.status.lock().unwrap_or_else(|e| e.into_inner());
                    let st = status.entry(key.clone()).or_default();
                    st.switch_reference([0; 16], &mut round);
                    *st
                };
                if dropped {
                    round.carry_skipped(round.skipped() + 1);
                }
                node.persist_logprob_state(&key, voided, &round);
                lp.rounds.lock().unwrap_or_else(|e| e.into_inner()).insert(key.clone(), round);
            }
            // Can it score right now? A throwaway prompt first, before any reference text is spent
            // (a registry pack's texts are used up network-wide the moment it's issued) — only when
            // there is something to check against.
            if lp.registry.is_some() || select_reference(&lp.refs, &model, declared.as_ref(), weights.as_deref()).is_some() {
                let pause = &mut || std::thread::sleep(LOGPROB_BUSY_PAUSE);
                if let Some(outcome) = probe_scoring(&mut info_transport, &model, pause) {
                    let round = lp.rounds.lock().unwrap_or_else(|e| e.into_inner()).remove(&key).unwrap_or_default();
                    node.settle_logprob_round(&lp, &key, &peer_id, &model, &model_key, outcome, round, 0, None);
                    return;
                }
            }
            // A local reference file first; else a fresh round pack from the registry (never-issued
            // texts, signature checked against the pinned key). A pack is one round's worth, so its
            // round starts clean — the registry, not the per-provider record, prevents re-use.
            let mut from_registry = false;
            let mut first_pack = None;
            let resumed = lp.resume.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
            let first_contact = lp.status.lock().unwrap_or_else(|e| e.into_inner()).get(&key).is_none_or(|st| st.verdict.is_none());
            // A provider with no verdict yet takes one of this hour's first-contact rounds, local
            // texts included (review round 4 L4-5: a local set is shared gateway-wide too).
            let network = node.observed_networks().get(&libp2p).and_then(|n| n.first().cloned()).unwrap_or_default();
            let standing = has_writer_standing(node.earned_reputation(&peer_id, &model_key, now_unix_ms()));
            let budget_ok = || !first_contact || lp.spend_first_contact_pack(&network, standing, now_unix_ms());
            let reference = match resumed {
                // A registry run stopped with a text pending: carry on with the same pack.
                Some((r, pack, _)) => {
                    from_registry = true;
                    first_pack = pack;
                    Some(r)
                }
                None => match select_reference(&lp.refs, &model, declared.as_ref(), weights.as_deref()).cloned() {
                Some(_) if !budget_ok() => {
                    tracing::debug!(provider = %peer_id, model, "first-contact budget spent this hour; checking later");
                    None
                }
                Some(r) => Some(r),
                None => match &lp.registry {
                    Some(_) if !budget_ok() => {
                        tracing::debug!(provider = %peer_id, model, "first-contact pack budget spent this hour; checking later");
                        None
                    }
                    Some(reg) => match reg.fetch_pack(declared.as_ref(), weights.as_deref(), &model) {
                        Ok(pack) => {
                            from_registry = true;
                            first_pack = Some(pack.pack_id);
                            let set = pack.to_reference_set();
                            if let Some(id) = &set.identity {
                                lp.registry_identities
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner())
                                    .insert(reference_tag(&set), id.clone());
                            }
                            if let Some(w) = &set.weights {
                                lp.registry_weights.lock().unwrap_or_else(|e| e.into_inner()).insert(reference_tag(&set), w.clone());
                            }
                            Some(Arc::new(LoadedReference { handles: pack.handles, set, vocab: None }))
                        }
                        // No pack came: the first-contact round wasn't spent.
                        Err(crate::registry_client::RegistryError::NoReference) => {
                            if first_contact {
                                lp.refund_first_contact_pack(&network, standing);
                            }
                            None
                        }
                        Err(e) => {
                            if first_contact {
                                lp.refund_first_contact_pack(&network, standing);
                            }
                            tracing::warn!(provider = %peer_id, model, error = ?e, "no round pack from the registry");
                            None
                        }
                    },
                    None => None,
                },
                },
            };
            let Some(reference) = reference else {
                let mut status = lp.status.lock().unwrap_or_else(|e| e.into_inner());
                status.entry(key.clone()).or_default().next_due_ms = now_unix_ms() + LOGPROB_BACKOFF_MS;
                drop(status);
                tracing::debug!(provider = %peer_id, model, declared = ?declared.as_ref().map(|i| i.key()),
                    "no log-prob reference for this model");
                lp.inflight.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
                return;
            };
            let mut round = lp.rounds.lock().unwrap_or_else(|e| e.into_inner()).remove(&key).unwrap_or_default();
            // A fresh pack is one round's worth, so its round starts clean; a resumed one carries on.
            if from_registry && round.pending_text(&reference.set).is_none() {
                // Texts it declined in the round so far still count (R5L-6).
                let skipped = round.skipped();
                round = ChallengeRound::new();
                round.carry_skipped(skipped);
            }
            let switched = {
                let mut status = lp.status.lock().unwrap_or_else(|e| e.into_inner());
                let st = status.entry(key.clone()).or_default();
                let tag = reference_tag(&reference.set);
                // A row saved before tags named the identity is on this same reference (L4-8), unless
                // another loaded reference shares that old tag with a different identity: then which
                // one it was earned on can't be told (R5L-5).
                let legacy = crate::prompt_score::legacy_reference_tag(&reference.set);
                let ambiguous = lp.refs.iter().any(|r| {
                    crate::prompt_score::legacy_reference_tag(&r.set) == legacy && r.set.identity != reference.set.identity
                });
                if st.reference == legacy && !ambiguous {
                    st.reference = tag;
                }
                st.switch_reference(tag, &mut round)
            };
            if switched {
                tracing::info!(provider = %peer_id, model, reference = reference.set.model_key,
                    "log-prob checks now use this reference");
            }
            let mut last_send: Option<std::time::Instant> = None;
            let mut transport = |framed: &[u8]| -> Result<Vec<u8>, String> {
                if let Some(wait) = last_send.and_then(|t| LOGPROB_PACE.checked_sub(t.elapsed())) {
                    std::thread::sleep(wait);
                }
                last_send = Some(std::time::Instant::now());
                node.net.proxy_forward_timeout(libp2p.clone(), framed.to_vec(), LOGPROB_TIMEOUT)
            };
            let current = |lp: &ProviderLogprob| {
                lp.status.lock().unwrap_or_else(|e| e.into_inner()).get(&key).copied().unwrap_or_default()
            };
            // A registry pack may be a first stage: if the round is still undecided when its texts run
            // out, ask once for the rest of the round (same batch) and carry on.
            let issued_before = round.issued_count();
            let resumable_pack = first_pack;
            let mut reference = reference;
            let outcome = loop {
                let outcome = run_round(
                    &mut round,
                    &reference.set,
                    &model,
                    &mut transport,
                    &mut |n| rand::random::<u64>() as usize % n,
                    // A local reference's texts go to one provider each, gateway-wide; a registry
                    // pack's are already issued once network-wide.
                    &|t| from_registry || !lp.issued_here.lock().unwrap_or_else(|e| e.into_inner()).contains(&t.digest()),
                    &mut |t| {
                        lp.issued_here.lock().unwrap_or_else(|e| e.into_inner()).insert(t.digest());
                    },
                    &mut || std::thread::sleep(LOGPROB_BUSY_PAUSE),
                    &mut |r| node.persist_logprob_state(&key, current(&lp), r),
                );
                let (RoundOutcome::Exhausted, Some(first), Some(reg)) = (&outcome, first_pack.take(), &lp.registry) else {
                    break outcome;
                };
                if round.progress() == 0 {
                    break outcome;
                }
                match reg.continue_pack(first) {
                    Ok(more) => {
                        let mut set = reference.set.clone();
                        set.texts.extend(more.texts);
                        reference = Arc::new(LoadedReference { handles: reference.handles.clone(), set, vocab: reference.vocab.clone() });
                    }
                    Err(e) => break RoundOutcome::Incomplete(format!("no continuation pack: {e:?}")),
                }
            };
            let texts_used = round.issued_count() - issued_before;
            // A first-contact round that issued nothing (busy, gone) didn't use its place (R5L-4).
            if first_contact && texts_used == 0 && first_pack.is_none() && !from_registry {
                lp.refund_first_contact_pack(&network, standing);
            }
            // Stopped with a text still pending on a registry pack: the next run resumes that pack,
            // and can still fetch its continuation.
            if from_registry && matches!(outcome, RoundOutcome::Busy | RoundOutcome::Incomplete(_)) && round.pending_text(&reference.set).is_some() {
                let mut resume = lp.resume.lock().unwrap_or_else(|e| e.into_inner());
                if resume.len() >= MAX_RESUMES {
                    if let Some(k) = resume.iter().min_by_key(|(_, (_, _, at))| *at).map(|(k, _)| k.clone()) {
                        resume.remove(&k);
                    }
                }
                resume.insert(key.clone(), (Arc::clone(&reference), resumable_pack, now_unix_ms()));
            }
            node.settle_logprob_round(&lp, &key, &peer_id, &model, &model_key, outcome, round, texts_used, Some(&reference.set.model_key));
        });
    }

    /// Apply how a log-prob run ended: status (verdict, streak, evasions, next check), audit /
    /// outcome effects, logs, persistence; then release the (peer, model) for the next run.
    #[allow(clippy::too_many_arguments)]
    fn settle_logprob_round(
        &self,
        lp: &ProviderLogprob,
        key: &str,
        peer_id: &str,
        model: &str,
        model_key: &str,
        outcome: RoundOutcome,
        round: ChallengeRound,
        texts_used: usize,
        reference_key: Option<&str>,
    ) {
        let now = now_unix_ms();
        let mut status = lp.status.lock().unwrap_or_else(|e| e.into_inner());
        let st = status.entry(key.to_string()).or_default();
        // A stopped run only counts against a provider that has scored texts for us before (in an
        // earlier round, or earlier in this one) — never a first contact or our own text shortage.
        let answered = st.verdict.is_some() || round.progress() > 0;
        // Runs without a verdict space the next ones out (review round 3, LP2).
        let retry = {
            let mut runs = lp.no_verdict_runs.lock().unwrap_or_else(|e| e.into_inner());
            if matches!(outcome, RoundOutcome::Verdict(_)) {
                runs.remove(key);
                LOGPROB_RETRY_MS
            } else {
                if !runs.contains_key(key) && runs.len() >= MAX_RESUMES * 16 {
                    runs.clear();
                }
                let n = runs.entry(key.to_string()).or_insert(0);
                *n = n.saturating_add(1);
                logprob_retry_ms(*n)
            }
        };
        // An Evasion leaves the next round its full allowance (R6-5).
        let mut round = round;
        if matches!(outcome, RoundOutcome::Evasion(_)) {
            round.carry_skipped(0);
        }
        let effect = match &outcome {
            RoundOutcome::Verdict(v) => st.on_verdict_drawn(*v, now, 1.0 - rand::random::<f64>()),
            // A run cut short by a timeout or a dropped connection isn't held against a provider:
            // that is what contention others cause looks like, and a provider stalling on purpose
            // answers "busy" at no cost anyway (round 8 L8-1).
            RoundOutcome::Incomplete(_) => st.on_no_verdict(false, retry, now),
            // Busy isn't held against a provider (it can't be told from contention others cause): the
            // text it stalled on waits for it. A provider that stays busy gains nothing: a fail stands
            // until a later round passes, and a pass lapses (review round 6).
            RoundOutcome::Busy => st.on_no_verdict(false, retry, now),
            RoundOutcome::Evasion(_) => st.on_evasion(retry, now),
            RoundOutcome::Unsupported => st.on_no_verdict(answered, LOGPROB_BACKOFF_MS.max(retry), now),
            // Out of texts because the provider declined some (L4-4): a stopped run like any other.
            RoundOutcome::Exhausted if round.skipped() > 0 => st.on_no_verdict(answered, LOGPROB_BACKOFF_MS, now),
            RoundOutcome::Exhausted => st.on_no_verdict(false, LOGPROB_BACKOFF_MS, now),
            // Retried soon (an agent hashes its file in the background after a restart); one that
            // keeps not declaring it is dodging.
            RoundOutcome::Undeclared => st.on_no_verdict(answered, retry, now),
        };
        let streak = st.pass_streak;
        let final_status = *st;
        drop(status);
        // A pass earns no audit credit or reputation by itself (it only sets the method flag once
        // the streak qualifies): reputation must come from served traffic, audits from audits.
        match effect {
            StatusEffect::None => {}
            StatusEffect::Fail => {
                self.record_audit(peer_id, model_key, AuditClass::Fail, false, now);
                self.record_outcome(peer_id, model_key, VerificationOutcome::Failed, now);
            }
            StatusEffect::Evasion => self.record_audit(peer_id, model_key, AuditClass::Fail, false, now),
        }
        match outcome {
            RoundOutcome::Verdict(v) if v.passed() => {
                tracing::info!(provider = %peer_id, model, aggregate = v.aggregate(), streak, texts = texts_used, "log-prob check passed")
            }
            RoundOutcome::Verdict(v) => tracing::warn!(provider = %peer_id, model, aggregate = v.aggregate(),
                texts = texts_used, "log-prob check FAILED (different model or quantization)"),
            RoundOutcome::Incomplete(reason) => {
                tracing::debug!(provider = %peer_id, model, reason, "log-prob round interrupted")
            }
            RoundOutcome::Unsupported => {
                tracing::debug!(provider = %peer_id, model, "provider does not answer prompt-scoring challenges")
            }
            RoundOutcome::Exhausted => tracing::warn!(provider = %peer_id, model, reference = reference_key.unwrap_or("?"),
                "log-prob reference set exhausted for this provider; add fresh texts"),
            RoundOutcome::Busy => tracing::debug!(provider = %peer_id, model, "provider busy; the pending text waits for the next run"),
            RoundOutcome::Undeclared => tracing::info!(provider = %peer_id, model, "provider no longer declares the weights file it passed on"),
            RoundOutcome::Evasion(reason) => tracing::warn!(provider = %peer_id, model, reason,
                "provider declined texts mid-round; counted as dodging the check"),
        }
        if effect == StatusEffect::Evasion {
            tracing::warn!(provider = %peer_id, model, "provider keeps dodging log-prob checks; counted as a failed audit");
        }
        self.persist_logprob_state(key, final_status, &round);
        lp.rounds.lock().unwrap_or_else(|e| e.into_inner()).insert(key.to_string(), round);
        lp.inflight.lock().unwrap_or_else(|e| e.into_inner()).remove(key);
    }
    /// Whether a serve at `provider` could be replayed as an audit (D1): replay is on, the user asked
    /// for greedy decoding (comparing answers needs it), the model is public (a private swarm's
    /// prompts never leave it), it isn't our own provider, and the prompt isn't huge.
    fn replay_eligible(&self, provider: &SelectedProvider, temperature: Option<f64>, messages: &[ChatMessage]) -> bool {
        self.audit_replay
            && temperature == Some(0.0)
            && provider.swarm_public_key.is_empty()
            && !self.is_self_serve(&provider.libp2p_peer_id)
            && messages.iter().map(|m| m.content.len()).sum::<usize>() <= REPLAY_MAX_PROMPT_BYTES
    }

    /// Sampled at the provider's audit rate after a serve: audit it in the background (D1), by
    /// replaying one of its recent serves to another provider when this consumer allows replay and
    /// has one, or else with a probe shaped like this consumer's requests for the model.
    fn maybe_audit(self: &Arc<Self>, model: &str, model_key: &str, provider: &SelectedProvider) {
        let rate = self.audit_rate_for(&provider.state_id, model_key, now_unix_ms());
        if rand::random::<f64>() >= rate {
            return; // not sampled this time
        }
        self.spawn_audit(model, provider, Duration::ZERO, false, "sampled");
    }

    /// Under a hard floor, now and then audit one provider of `model` the floor turned away (one of
    /// `peers` not among `admitted`), so a provider below it can earn its way up: it gets no traffic
    /// from us otherwise (review round 4 H2). Each such provider at most every
    /// [`QUALIFY_INTERVAL_MS`].
    fn maybe_qualify(self: &Arc<Self>, model: &str, peers: &[DiscoveredPeer], admitted: &[SelectedProvider], now: u64) {
        if rand::random::<f64>() >= QUALIFY_SAMPLE {
            return;
        }
        // One at a time, and a few an hour per model (R5T-5).
        {
            let mut per_model = self.qualified_per_model.lock().unwrap_or_else(|e| e.into_inner());
            for q in per_model.values_mut() {
                while q.front().is_some_and(|at| now.saturating_sub(*at) > 60 * 60 * 1000) {
                    q.pop_front();
                }
            }
            per_model.retain(|_, q| !q.is_empty());
            if per_model.get(model).map_or(0, |q| q.len()) >= QUALIFY_PER_MODEL_PER_HOUR {
                return;
            }
        }
        // Public providers only: a private card's provider isn't audited (R5T-8).
        let left_out: Vec<SelectedProvider> =
            rank_providers_with_reputation(peers, "", self.tier, &|pid, mkey| self.earned_reputation(pid, mkey, now))
                .into_iter()
                .filter(|p| {
                    !self.is_self_serve(&p.libp2p_peer_id)
                        && p.swarm_public_key.is_empty()
                        && !admitted.iter().any(|a| a.state_id == p.state_id)
                })
                .collect();
        let pick = {
            use rand::seq::SliceRandom;
            let mut last = self.qualified_at.lock().unwrap_or_else(|e| e.into_inner());
            last.retain(|_, at| now.saturating_sub(*at) <= QUALIFY_INTERVAL_MS);
            let due: Vec<&SelectedProvider> = left_out
                .iter()
                .filter(|p| !last.contains_key(&reputation_key(&p.state_id, rep_model_key(&p.canonical_model_id, &p.model_id))))
                .collect();
            due.choose(&mut rand::thread_rng()).map(|p| (*p).clone())
        };
        if let Some(provider) = pick {
            use std::sync::atomic::Ordering;
            if self.qualifying.fetch_add(1, Ordering::AcqRel) >= 1 {
                self.qualifying.fetch_sub(1, Ordering::AcqRel);
                return;
            }
            let running = SettleSlot(Arc::clone(&self.qualifying));
            let Some(slot) = self.audit_slot() else { return };
            // Its interval starts once its audit does (round 8 T8-6): a pick that found no slot isn't
            // spent.
            {
                let mut last = self.qualified_at.lock().unwrap_or_else(|e| e.into_inner());
                if last.len() < MAX_QUALIFY_RECORDS {
                    last.insert(reputation_key(&provider.state_id, rep_model_key(&provider.canonical_model_id, &provider.model_id)), now);
                }
            }
            self.qualified_per_model
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(model.to_string())
                .or_default()
                .push_back(now);
            // A provider left out over a log-prob fail gets a new round too: a pass is its way back
            // (review round 7, R7-1). Its own schedule (`next_due`) still applies.
            self.maybe_logprob_check(&provider, rep_model_key(&provider.canonical_model_id, &provider.model_id));
            let node = Arc::clone(self);
            let model = model.to_string();
            std::thread::spawn(move || {
                let (_running, _slot) = (running, slot);
                match node.audit(&model, &provider, true, false) {
                    Ok((how, report)) => tracing::info!(model, subject = %provider.state_id, how, verdict = ?report.verdict,
                        outcomes = ?report.outcomes, "qualifying audit complete"),
                    Err(e) => tracing::debug!(model, subject = %provider.state_id, error = %e, "qualifying audit skipped"),
                }
            });
        }
    }

    /// The follow-up audit of a provider with a strike (C3): after `delay`, a slot is taken (waited
    /// for, up to [`FOLLOW_UP_TRIES`] times a minute apart), so busy audits don't drop it (R5T-2). It
    /// probes with a plain shape when the traffic gives none, and the subject not answering counts
    /// against it. Follow-ups pending at once are capped ([`MAX_FOLLOW_UPS`]).
    fn spawn_follow_up(self: &Arc<Self>, model: &str, provider: &SelectedProvider, delay: Duration) {
        use std::sync::atomic::Ordering;
        if self.follow_ups.fetch_add(1, Ordering::AcqRel) >= MAX_FOLLOW_UPS {
            self.follow_ups.fetch_sub(1, Ordering::AcqRel);
            tracing::debug!(model, subject = %provider.state_id, "follow-up audit skipped: too many pending");
            return;
        }
        let pending = SettleSlot(Arc::clone(&self.follow_ups));
        let node = Arc::clone(self);
        let model = model.to_string();
        let provider = provider.clone();
        std::thread::spawn(move || {
            let _pending = pending;
            std::thread::sleep(delay);
            for _ in 0..FOLLOW_UP_TRIES {
                if let Some(slot) = node.audit_slot_for(true) {
                    let _slot = slot;
                    match node.audit(&model, &provider, true, true) {
                        Ok((how, report)) => tracing::info!(model, subject = %provider.state_id, how, verdict = ?report.verdict,
                            outcomes = ?report.outcomes, "follow-up audit complete"),
                        Err(e) => tracing::debug!(model, subject = %provider.state_id, error = %e, "follow-up audit couldn't run"),
                    }
                    return;
                }
                std::thread::sleep(Duration::from_secs(60));
            }
            tracing::debug!(model, subject = %provider.state_id, "follow-up audit found no free slot");
        });
    }

    /// A slot to run an audit in, while fewer than [`MAX_AUDITS_IN_FLIGHT`] run; the last one is kept
    /// for follow-ups (`follow_up`), which resolve strikes and must not lose it to sampled audits
    /// (round 8 T8-7).
    fn audit_slot_for(&self, follow_up: bool) -> Option<AuditSlot> {
        use std::sync::atomic::Ordering;
        let limit = if follow_up { MAX_AUDITS_IN_FLIGHT } else { MAX_AUDITS_IN_FLIGHT - 1 };
        if self.audits_in_flight.fetch_add(1, Ordering::Relaxed) >= limit {
            self.audits_in_flight.fetch_sub(1, Ordering::Relaxed);
            return None;
        }
        Some(AuditSlot(Arc::clone(&self.audits_in_flight)))
    }

    /// A slot for a sampled or qualifying audit (see [`Self::audit_slot_for`]).
    fn audit_slot(&self) -> Option<AuditSlot> {
        self.audit_slot_for(false)
    }

    /// Audit `provider` for `model` in the background, if a slot is free (`why` is for the log).
    /// `plain`: probe with a plain greedy shape when this consumer's traffic gives none to copy (a
    /// provider it has nothing else to audit by, H2).
    fn spawn_audit(self: &Arc<Self>, model: &str, provider: &SelectedProvider, delay: Duration, plain: bool, why: &'static str) {
        let Some(slot) = self.audit_slot() else {
            tracing::debug!(model, subject = %provider.state_id, why, "audit skipped: too many running");
            return;
        };
        let node = Arc::clone(self);
        let model = model.to_string();
        let provider = provider.clone();
        std::thread::spawn(move || {
            let _slot = slot;
            std::thread::sleep(delay);
            match node.audit(&model, &provider, plain, false) {
                Ok((how, report)) => tracing::info!(model, subject = %provider.state_id, why, how, verdict = ?report.verdict,
                    outcomes = ?report.outcomes, "redundant-exec audit complete"),
                Err(e) => tracing::debug!(model, subject = %provider.state_id, why, error = %e, "redundant-exec audit skipped"),
            }
        });
    }

    /// Audit `provider` for `model` (see [`maybe_audit`](Self::maybe_audit)), saying how ("replay" or
    /// "probe"). `plain` as for [`spawn_audit`](Self::spawn_audit); `strict`: a follow-up, whose
    /// subject not answering counts against it.
    fn audit(self: &Arc<Self>, model: &str, provider: &SelectedProvider, plain: bool, strict: bool) -> Result<(&'static str, AuditReport), AdapterError> {
        let now = now_unix_ms();
        let kept = take_replayable(&mut self.replays.lock().unwrap_or_else(|e| e.into_inner()), &provider.state_id, model, now);
        if let Some(entry) = kept {
            return self
                .audit_model(&entry.model, &entry.request, &entry.provider, Some(entry.answer.as_str()), strict, None)
                .map(|r| ("replay", r));
        }
        // A probe only where this consumer's traffic allows one (checked first: writing it costs);
        // a plain one where it doesn't and `plain` says to.
        let shape = self.traffic.lock().unwrap_or_else(|e| e.into_inner()).get(model).cloned();
        let shape = match shape {
            Some(t) if t.probe(String::new()).is_some() => Some(t),
            // A follow-up or qualifying audit on a model whose traffic is mostly sampled: shaped like
            // that traffic all the same (thinking, system prompt, lengths), only greedy, which
            // comparing answers needs (round 8 T8-2).
            Some(mut t) if plain && t.requests >= MIN_SHAPED_REQUESTS => {
                t.greedy = t.requests;
                Some(t)
            }
            _ if plain => Some(TrafficShape::plain(&mut rand::thread_rng())),
            _ => None,
        };
        // A question written for this audit and finished here (P1), else a template's. Only a written
        // one can clear a strike: the templates are public (R7-3). A follow-up never uses the questions
        // of the outside writer whose probe gave the strike (round 8 T8-4).
        let avoid = strict
            .then(|| {
                let key = reputation_key(&provider.state_id, rep_model_key(&provider.canonical_model_id, &provider.model_id));
                self.strikes.lock().unwrap_or_else(|e| e.into_inner()).get(&key).and_then(|s| s.writer.clone())
            })
            .flatten();
        let mut writer = None;
        let probe = shape.filter(|t| t.probe(String::new()).is_some()).and_then(|t| {
            let prompt = match self.fresh_probe_prompt(model, avoid.as_deref()) {
                Some((p, by)) => {
                    writer = Some(by);
                    p
                }
                None => crate::audit_prompts::random_prompt(&mut rand::thread_rng()),
            };
            t.probe(prompt)
        });
        match probe {
            Some(request) => self.audit_model(model, &request, provider, None, strict, writer.as_deref()).map(|r| ("probe", r)),
            None => Err(AdapterError::Route(format!(
                "no audit for '{model}': nothing to replay, and too few greedy requests to probe like them"
            ))),
        }
    }

    /// A probe prompt written for this audit (P1): taken from questions a model wrote for this
    /// gateway's probes of `model` (refilled when low or a day old), finished here so the writer never
    /// sees it, and never sent twice. A pool whose writer has come to serve `model` is dropped (it
    /// would recognise its questions). Returns the prompt and its writer. `None` when no writer could
    /// be had, or the only questions are `avoid`'s (the caller falls back to the templates).
    fn fresh_probe_prompt(self: &Arc<Self>, model: &str, avoid: Option<&str>) -> Option<(String, String)> {
        let now = now_unix_ms();
        let usable = {
            let pools = self.probe_pool.lock().unwrap_or_else(|e| e.into_inner());
            pools.get(model).is_some_and(|p| p.questions.len() >= 2 && now.saturating_sub(p.filled_at) <= PROBE_POOL_TTL_MS).then(|| {
                pools.get(model).map(|p| p.writer.clone()).unwrap_or_default()
            })
        };
        let writer_serves_it = |writer: &str| {
            !self.is_self_serve(writer)
                && self.net.discover(model).is_ok_and(|peers| peers.iter().any(|p| p.libp2p_peer_id == writer))
        };
        if usable.as_deref().is_none_or(writer_serves_it) {
            let pool = self.write_probe_questions(model).map(|(writer, questions)| ProbePool {
                writer,
                filled_at: now,
                questions: questions.into(),
            });
            let mut pools = self.probe_pool.lock().unwrap_or_else(|e| e.into_inner());
            match pool {
                Some(p) => {
                    pools.insert(model.to_string(), p);
                }
                None => {
                    pools.remove(model);
                }
            }
        }
        let mut rng = rand::thread_rng();
        loop {
            let (prompt, writer) = {
                let mut pools = self.probe_pool.lock().unwrap_or_else(|e| e.into_inner());
                let pool = pools.get_mut(model).filter(|p| Some(p.writer.as_str()) != avoid)?;
                (crate::audit_prompts::finish(&mut pool.questions, &mut rng)?, pool.writer.clone())
            };
            let digest = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                prompt.hash(&mut h);
                h.finish()
            };
            let mut sent = self.probe_sent.lock().unwrap_or_else(|e| e.into_inner());
            if sent.len() >= MAX_PROBES_REMEMBERED {
                sent.clear();
            }
            if sent.insert(digest) {
                return Some((prompt, writer));
            }
        }
    }

    /// Questions to probe `model` with, written by a model (P1), and who wrote them (libp2p id). Our
    /// own provider whenever it serves any model (the prompts never leave this machine, and it costs
    /// nothing, review round 4 H6), found from its own records; otherwise a provider of another model
    /// that serves none of the audited model and has standing here (round 8 T8-1), paid like any
    /// serve. `None` when nobody answers (the caller falls back to the templates).
    fn write_probe_questions(self: &Arc<Self>, model: &str) -> Option<(String, Vec<String>)> {
        let now = now_unix_ms();
        let ranked = |m: &str| -> Vec<SelectedProvider> {
            self.net
                .discover(m)
                .map(|peers| rank_providers_with_reputation(&peers, "", self.tier, &|pid, mkey| self.earned_reputation(pid, mkey, now)))
                .unwrap_or_default()
        };
        let audited = ranked(model);
        let audited_ids: std::collections::HashSet<String> = audited.iter().map(|p| p.libp2p_peer_id.clone()).collect();
        let mut tried = Vec::new();
        // Our own provider first, under whichever model it serves: from the records it signed.
        if let Some(own) = self.self_provider.as_deref() {
            let mine = std::iter::once(model.to_string())
                .chain(self.net.models_of(own).unwrap_or_default())
                .find_map(|m| ranked(&m).into_iter().find(|p| p.libp2p_peer_id == own));
            tried.extend(mine);
        }
        // Else a provider of another model, serving none of the audited one, that completed a real
        // serve for us with standing (round 9 A9-4/A9-5): only those, never one picked from what the
        // network advertises, and no network search when there is none.
        let mut searched_outside = false;
        if tried.is_empty() && self.no_writer_until.lock().unwrap_or_else(|e| e.into_inner()).get(model).is_none_or(|t| now >= *t) {
            searched_outside = true;
            let mut candidates: Vec<(String, String)> = self
                .writer_candidates
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .filter(|((peer, m), at)| m != model && !audited_ids.contains(peer) && now.saturating_sub(**at) <= WRITER_CANDIDATE_TTL_MS)
                .map(|(k, _)| k.clone())
                .collect();
            {
                use rand::seq::SliceRandom;
                candidates.shuffle(&mut rand::thread_rng());
            }
            for (peer, m) in candidates {
                if tried.len() >= MAX_WRITERS_TRIED {
                    break;
                }
                let found = ranked(&m).into_iter().find(|p| {
                    p.libp2p_peer_id == peer
                        && has_writer_standing(self.earned_reputation(&p.state_id, rep_model_key(&p.canonical_model_id, &p.model_id), now))
                });
                tried.extend(found);
            }
        }
        for writer in tried {
            let written = self.write_questions(&writer);
            tracing::info!(model, writer = %writer.peer_id, own = self.is_self_serve(&writer.libp2p_peer_id), n = written.len(), "probe questions written");
            if !written.is_empty() {
                return Some((writer.libp2p_peer_id, written));
            }
        }
        if searched_outside {
            let mut m = self.no_writer_until.lock().unwrap_or_else(|e| e.into_inner());
            m.retain(|_, t| now < *t);
            if m.len() < MAX_WRITER_CANDIDATES {
                m.insert(model.to_string(), now + NO_WRITER_RETRY_MS);
            }
        }
        None
    }

    /// Ask `writer` for a batch of questions ([`crate::audit_prompts::writer_request`]), sampled (not
    /// greedy: they should vary), paid like any serve unless it is our own provider. At most
    /// [`WRITER_MAX_TEXT_BYTES`] of its answer is read.
    fn write_questions(self: &Arc<Self>, writer: &SelectedProvider) -> Vec<String> {
        let ask = crate::audit_prompts::writer_request(&mut rand::thread_rng());
        let messages = vec![ChatMessage { role: "user".into(), content: ask, ..Default::default() }];
        let request = self.serve_request_for(writer, &messages, Some(WRITER_MAX_TOKENS), Some(0.9), &[], Some(false));
        let mut text = String::new();
        let mut full = false;
        let mut delivered = crate::serve::DeliveredText::default();
        let mut tally = crate::serve::StreamTally::default();
        let exchange = self.exchange(&writer.libp2p_peer_id, &request, &|| false, &mut |d| {
            delivered.push(d);
            tally.push_delta(d);
            if !full && text.len() + d.len() <= WRITER_MAX_TEXT_BYTES {
                text.push_str(d);
            } else {
                full = true; // nothing after a piece that didn't fit: the text stays contiguous
            }
        });
        let paying = !self.is_self_serve(&writer.libp2p_peer_id);
        match exchange.result {
            Ok(summary) => {
                if paying && summary.ok && summary.tokens > 0 {
                    let tokens = crate::serve::billable_tokens(summary.tokens, request.max_tokens, delivered.units());
                    self.settle_receipt(writer, tokens, exchange.nonce);
                }
                crate::audit_prompts::parse_questions(&text)
            }
            Err(e) => {
                if paying && exchange.opened {
                    tally.push_tool_calls(&exchange.tool_calls);
                    self.close_and_settle(writer, exchange.nonce, exchange.stream_offset, tally, request.max_tokens);
                } else if paying && tally.billable(request.max_tokens) > 0 {
                    self.settle_receipt(writer, tally.billable(request.max_tokens), exchange.nonce);
                }
                tracing::debug!(writer = %writer.peer_id, error = %e, "couldn't get probe questions written");
                Vec::new()
            }
        }
    }

    /// The serve request an audit sends `provider`: built as a user's is ([`Self::serve_request_for`]),
    /// greedy.
    fn audit_request_for(&self, provider: &SelectedProvider, request: &AuditRequest) -> ServeRequest {
        self.serve_request_for(provider, &request.messages, request.max_tokens, Some(0.0), &request.tools, request.think)
    }

    /// Serve `request` at `provider` for an audit, exactly as a user's serve goes (the same request
    /// fields, transport, and settlement), and return the answer's text. It is paid for like any
    /// serve: the provider did the work, and an unpaid one would stand out (and, at a provider that
    /// charges, count against us). A stream that breaks after output is settled for what was read.
    fn audit_serve(self: &Arc<Self>, provider: &SelectedProvider, request: &AuditRequest) -> Result<String, AdapterError> {
        let serve = self.audit_request_for(provider, request);
        let mut answer = String::new();
        let mut delivered = crate::serve::DeliveredText::default();
        let mut tally = crate::serve::StreamTally::default();
        // Read to the end, as a user's client does, but only the compared start is kept (M6), and for
        // at most AUDIT_SERVE_MAX (R5T-5).
        let started = std::time::Instant::now();
        let exchange = self.exchange(&provider.libp2p_peer_id, &serve, &|| started.elapsed() > AUDIT_SERVE_MAX, &mut |d| {
            delivered.push(d);
            tally.push_delta(d);
            keep_prefix(&mut answer, d, AUDIT_COMPARE_CHARS);
        });
        let paying = !self.is_self_serve(&provider.libp2p_peer_id);
        // Stopped at the time limit with the compared opening in hand: that is all an audit needs (R6-3).
        let result = match exchange.result {
            Err(AdapterError::Cancelled(_)) if answer.chars().count() >= AUDIT_COMPARE_CHARS => Err(AdapterError::Cancelled(String::new())),
            other => other,
        };
        if matches!(&result, Err(AdapterError::Cancelled(m)) if m.is_empty()) {
            if paying && exchange.opened {
                tally.push_tool_calls(&exchange.tool_calls);
                self.close_and_settle(provider, exchange.nonce, exchange.stream_offset, tally, serve.max_tokens);
            }
            return Ok(answer);
        }
        match result {
            Ok(summary) => {
                if paying && summary.ok && summary.tokens > 0 {
                    let offered = crate::serve::offered_tool_names(&serve.tools);
                    let units = delivered.units() + crate::serve::tool_call_units(&summary.tool_calls, &offered);
                    self.settle_receipt(provider, crate::serve::billable_tokens(summary.tokens, serve.max_tokens, units), exchange.nonce);
                }
                Ok(answer)
            }
            Err(e) => {
                if paying && exchange.opened {
                    tally.push_tool_calls(&exchange.tool_calls);
                    self.close_and_settle(provider, exchange.nonce, exchange.stream_offset, tally, serve.max_tokens);
                } else if paying && tally.billable(serve.max_tokens) > 0 {
                    // A buffered serve that failed part-way is owed for its text (round 9 B9-4).
                    self.settle_receipt(provider, tally.billable(serve.max_tokens), exchange.nonce);
                }
                Err(e)
            }
        }
    }

    /// Whether `provider` (libp2p id) served us at our price recently: a paid settlement with it
    /// succeeded within [`PAYMENT_REFUSAL_TTL_MS`] and none has failed since. Its payment refusal
    /// during an audit then isn't about price (review B1).
    fn paid_recently(&self, provider: &str, now: u64) -> bool {
        if self.settlement_peg.is_none() {
            return false;
        }
        let (ok, failed) = self.settle_record.lock().unwrap_or_else(|e| e.into_inner()).get(provider).copied().unwrap_or((0, 0));
        // Or it served us a paid request since its last failed settlement (review round 4 T4-10): a
        // provider can't dodge by refusing to co-sign a receipt first, or by settling slowly.
        let served = self.served_paid.lock().unwrap_or_else(|e| e.into_inner()).get(provider).copied().unwrap_or(0);
        let latest = ok.max(served);
        latest > 0 && now.saturating_sub(latest) <= PAYMENT_REFUSAL_TTL_MS && failed < latest
    }

    /// Note that `provider` (libp2p id) just served us a request we are paying for.
    fn note_served_paid(&self, provider: &str, now: u64) {
        if self.settlement_peg.is_none() {
            return;
        }
        let mut m = self.served_paid.lock().unwrap_or_else(|e| e.into_inner());
        if !m.contains_key(provider) && m.len() >= MAX_SETTLE_RECORDS {
            m.retain(|_, at| now.saturating_sub(*at) <= PAYMENT_REFUSAL_TTL_MS);
            if m.len() >= MAX_SETTLE_RECORDS {
                return;
            }
        }
        m.insert(provider.to_string(), now);
    }

    /// Note how a settlement with `provider` (libp2p id) went.
    fn note_settled(&self, provider: &str, ok: bool) {
        let now = now_unix_ms();
        let mut m = self.settle_record.lock().unwrap_or_else(|e| e.into_inner());
        if !m.contains_key(provider) && m.len() >= MAX_SETTLE_RECORDS {
            m.retain(|_, (o, f)| now.saturating_sub((*o).max(*f)) <= PAYMENT_REFUSAL_TTL_MS);
        }
        if m.contains_key(provider) || m.len() < MAX_SETTLE_RECORDS {
            let e = m.entry(provider.to_string()).or_insert((0, 0));
            if ok {
                e.0 = now;
            } else {
                e.1 = now;
            }
        }
    }

    /// Run one **redundant-execution audit** (M2.2(b)) of `model`: the same greedy `request` served
    /// by independent providers, their answers compared by their opening ([`compared`]) via
    /// [`resolve_audit`], and the outcomes recorded (an outvoted provider → `Failed`; agreeing ones →
    /// `Honored`; a non-responder → `Rejected`). A 1-vs-1 split goes to a third provider.
    ///
    /// `subject` is the provider audited, asked first. When `request` is its user's serve being
    /// replayed, `served` is the answer it gave: it never sees the audit (its serve was real), and
    /// the others see an ordinary request. Otherwise `request` is a probe, sent to it too.
    ///
    /// Off the user's latency path (see [`maybe_audit`](Self::maybe_audit)). Errors when fewer than
    /// two answers can be had.
    ///
    /// *Caveat:* providers of the same canonical `model_id` may run different quantizations / engine
    /// builds whose greedy output differs; comparing only the opening, the `Inconclusive`-on-tie
    /// rule, and long-run reputation keep that from punishing an honest provider, but heterogeneous
    /// fleets weaken the signal.
    fn audit_model(
        self: &Arc<Self>,
        model: &str,
        request: &AuditRequest,
        subject: &SelectedProvider,
        served: Option<&str>,
        strict_subject: bool,
        probe_writer: Option<&str>,
    ) -> Result<AuditReport, AdapterError> {
        let now = now_unix_ms();
        let peers = self
            .net
            .discover(model)
            .map_err(|e| AdapterError::Network(format!("discover: {e}")))?;
        let candidates = rank_providers_with_reputation(&peers, "", self.tier, &|pid, mkey| {
            self.earned_reputation(pid, mkey, now)
        });

        // Providers are picked at random, not by rank, and must be independent (review T2): no two
        // on the same network as we observe it, and at most one partner we only reach through a
        // relay (the subject's own unknown network doesn't use that place, H5). Where some are of
        // unknown network, they are judged only against at least two on known networks
        // (`resolve_against_known`). A provider whose network we can't see until we've served it
        // is served and then judged; at most one per audit may turn out not independent and be left
        // out (R5T-1), after which only providers already known to be independent are asked. A
        // provider that charges and refuses us over payment isn't a non-responder (a price, not a
        // fault): it is skipped for the next one, and so is one that just refused us. Unless it
        // served us at our price just now (B1): then refusing is dodging the audit.
        let networks = self.observed_networks();
        let nets_now = |peer: &str| self.observed_networks().get(peer).cloned().unwrap_or_default();
        let mut order: Vec<&SelectedProvider> = candidates
            .iter()
            .filter(|c| c.libp2p_peer_id != subject.libp2p_peer_id)
            .filter(|c| !self.refused_payment(&c.libp2p_peer_id, &c.model_id, now) || self.paid_recently(&c.libp2p_peer_id, now))
            // Not one this consumer failed in an audit within the floor's day: it isn't a partner to
            // judge others by, nor to send user prompts to (round 8 T8-7).
            .filter(|c| !self.failed_audit_recently(&c.state_id, rep_model_key(&c.canonical_model_id, &c.model_id), now))
            .collect();
        {
            use rand::seq::SliceRandom;
            order.shuffle(&mut rand::thread_rng());
        }
        let mut pool = order.into_iter();
        let mut chosen: Vec<Vec<String>> = Vec::new();
        let mut chosen_peers = std::collections::HashSet::new();
        let mut unknown_partners = 0usize;
        let mut discarded = 0usize;
        // Every provider sent the request counts toward the "at most four" the disclosure promises,
        // one that refuses over payment too (R6-5).
        let mut contacted = 1usize;
        let mut results: Vec<(String, Result<String, String>)> = Vec::with_capacity(4);
        // Model keys and networks of everyone audited (by state id), and who they are.
        let mut key_of: HashMap<String, String> = HashMap::new();
        let mut nets_of: HashMap<String, Vec<String>> = HashMap::new();
        let mut who: HashMap<String, SelectedProvider> = HashMap::new();
        who.insert(subject.state_id.clone(), subject.clone());
        // Whether the subject's non-answer was its own doing (an error it sent, or a payment refusal
        // after serving us at our price), not the network's or a busy pool's (R6-3).
        let mut subject_refused = false;
        // Whether the subject shed the request (its pool was full): no outcome then (T8-3).
        let mut subject_shed = false;
        let first = match served {
            Some(answer) => Ok(answer.to_string()),
            None => match self.audit_serve(subject, request) {
                Err(e) if crate::serve::is_payment_refusal(&e) && !self.paid_recently(&subject.libp2p_peer_id, now) => {
                    self.note_payment_refusal(&subject.libp2p_peer_id, &subject.model_id, now);
                    return Err(e);
                }
                Err(e) => {
                    // An error it sent counts as its own doing only while its engine evidently works:
                    // it completed a real serve for us around now (round 8 T8-5). An engine outage,
                    // or a model it stopped sharing, fails our traffic too.
                    subject_refused = matches!(e, AdapterError::Provider(_))
                        && self.served_ok_recently(&subject.libp2p_peer_id, &subject.model_id, now_unix_ms());
                    subject_shed = matches!(e, AdapterError::Overloaded(_));
                    Err(e)
                }
                out => out,
            },
        };
        let subject_nets = nets_now(&subject.libp2p_peer_id);
        chosen.push(subject_nets.clone());
        chosen_peers.insert(subject.libp2p_peer_id.clone());
        nets_of.insert(subject.state_id.clone(), subject_nets);
        key_of.insert(subject.state_id.clone(), rep_model_key(&subject.canonical_model_id, &subject.model_id).to_string());
        results.push((subject.state_id.clone(), first.map(|a| compared(&a)).map_err(|e| e.to_string())));
        // Ask the next independent provider; `known_only` takes only one on a known network.
        let mut ask = |results: &mut Vec<(String, Result<String, String>)>,
                       key_of: &mut HashMap<String, String>,
                       nets_of: &mut HashMap<String, Vec<String>>,
                       who: &mut HashMap<String, SelectedProvider>,
                       known_only: bool| {
            for provider in pool.by_ref() {
                if chosen_peers.contains(&provider.libp2p_peer_id) {
                    continue;
                }
                let before = networks.get(&provider.libp2p_peer_id).cloned().unwrap_or_default();
                if before.is_empty() {
                    // Its network shows only once served: only while none has been left out yet.
                    if discarded > 0 {
                        continue;
                    }
                } else if !independent_partner(&chosen, unknown_partners, &before) {
                    continue; // known to share a network: skipped without serving it
                }
                if contacted >= MAX_AUDIT_PROVIDERS {
                    return;
                }
                contacted += 1;
                let out = match self.audit_serve(provider, request) {
                    Err(e) if crate::serve::is_payment_refusal(&e) => {
                        if !self.paid_recently(&provider.libp2p_peer_id, now) {
                            self.note_payment_refusal(&provider.libp2p_peer_id, &provider.model_id, now);
                            continue;
                        }
                        Err(e)
                    }
                    // A partner that shed the request or couldn't be reached is skipped, not held to
                    // it (T8-3); the next one is asked, within the contact limit.
                    Err(AdapterError::Overloaded(_)) | Err(AdapterError::Network(_)) => continue,
                    out => out,
                };
                let nets = if before.is_empty() { nets_now(&provider.libp2p_peer_id) } else { before };
                if !independent_partner(&chosen, unknown_partners, &nets) || (known_only && nets.is_empty()) {
                    discarded += 1; // served, but not independent of the others: not counted
                    continue;
                }
                unknown_partners += usize::from(nets.is_empty());
                chosen.push(nets.clone());
                chosen_peers.insert(provider.libp2p_peer_id.clone());
                key_of.insert(provider.state_id.clone(), rep_model_key(&provider.canonical_model_id, &provider.model_id).to_string());
                nets_of.insert(provider.state_id.clone(), nets);
                who.insert(provider.state_id.clone(), provider.clone());
                results.push((provider.state_id.clone(), out.map(|a| compared(&a)).map_err(|e| e.to_string())));
                return;
            }
        };
        while results.len() < 2 {
            let before = results.len();
            ask(&mut results, &mut key_of, &mut nets_of, &mut who, false);
            if results.len() == before {
                break;
            }
        }
        if results.len() < 2 {
            return Err(AdapterError::Route(format!(
                "redundant-exec audit needs ≥2 providers that serve us '{model}', found {}",
                results.len()
            )));
        }
        let unknown_in = |results: &[(String, Result<String, String>)], nets_of: &HashMap<String, Vec<String>>| {
            results.iter().any(|(p, _)| nets_of.get(p).is_none_or(|n| n.is_empty()))
        };
        let known_answers = |results: &[(String, Result<String, String>)], nets_of: &HashMap<String, Vec<String>>| {
            results.iter().filter(|(p, r)| r.is_ok() && nets_of.get(p).is_some_and(|n| !n.is_empty())).count()
        };
        let report = if unknown_in(&results, &nets_of) {
            // Some are of unknown network: hear providers on known networks until two have answered
            // (at most four providers in all), and judge everyone against their majority.
            while known_answers(&results, &nets_of) < 2 && results.len() < 4 {
                let before = results.len();
                ask(&mut results, &mut key_of, &mut nets_of, &mut who, true);
                if results.len() == before {
                    break;
                }
            }
            let mut report = resolve_against_known(&results, &nets_of, &subject.state_id);
            if report.is_none() && known_answers(&results, &nets_of) == 2 && results.len() < 4 {
                ask(&mut results, &mut key_of, &mut nets_of, &mut who, true);
                report = resolve_against_known(&results, &nets_of, &subject.state_id);
            }
            report.ok_or_else(|| {
                AdapterError::Route(format!(
                    "redundant-exec audit of '{model}': providers of unknown network, and too few on known networks to judge them by"
                ))
            })?
        } else {
            let mut report = resolve_audit(&results);
            if report.verdict == RedundantVerdict::Inconclusive {
                ask(&mut results, &mut key_of, &mut nets_of, &mut who, true);
                if results.len() == 3 {
                    report = if unknown_in(&results, &nets_of) {
                        resolve_against_known(&results, &nets_of, &subject.state_id).unwrap_or(report)
                    } else {
                        resolve_audit(&results)
                    };
                }
            }
            report
        };
        // A follow-up's subject that refused it is held to that like a wrong answer (R5T-2); one the
        // network or a busy pool kept from answering isn't (R6-3).
        let mut report = report;
        if strict_subject && subject_refused {
            for (peer, outcome) in report.outcomes.iter_mut() {
                if *peer == subject.state_id && *outcome == VerificationOutcome::Rejected {
                    *outcome = VerificationOutcome::Failed;
                }
            }
            if results[0].1.is_err() && !report.outcomes.iter().any(|(p, _)| *p == subject.state_id) {
                report.outcomes.push((subject.state_id.clone(), VerificationOutcome::Failed));
            }
        }
        // A shed is the provider's own word that it was busy: excused once in ten minutes, not every
        // time (round 9 A9-1: else a provider could shed every audit it recognises at no cost).
        let drop_subject_rejected = |report: &mut AuditReport| {
            report.outcomes.retain(|(peer, outcome)| !(*peer == subject.state_id && *outcome == VerificationOutcome::Rejected));
        };
        let shed_excused = subject_shed && !self.note_audit_shed(&subject.libp2p_peer_id, &subject.model_id, now);
        if shed_excused && !strict_subject {
            drop_subject_rejected(&mut report);
        }
        // A follow-up whose subject couldn't be heard (not its own doing): no outcome for that (round 9
        // A9-2: an outage mustn't stack penalties); the strike stands, and it is asked again later, a
        // few times (T8-5). Unheard on every try while it kept serving our traffic of the model: that
        // is avoiding the follow-up, a failure (A9-1).
        if strict_subject && results[0].1.is_err() && !subject_refused {
            drop_subject_rejected(&mut report);
            let key = reputation_key(&subject.state_id, key_of.get(&subject.state_id).map_or(model, String::as_str));
            let retries = self.strikes.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&key).map(|s| {
                s.retries += 1;
                s.retries
            });
            match retries {
                Some(n) if n <= FOLLOW_UP_RETRIES => {
                    let delay = Duration::from_millis(rand::Rng::gen_range(&mut rand::thread_rng(), FOLLOW_UP_DELAY_MS));
                    self.spawn_follow_up(model, subject, delay);
                }
                // Our own request of the model got through within a minute of this last try: the
                // provider was taking requests, just not this one (round 10 X10-3: a busy provider
                // whose pool happened to be full each time isn't failed on a looser test).
                Some(_) if self.served_ok_within(&subject.libp2p_peer_id, &subject.model_id, now_unix_ms(), AUDIT_SHED_EVIDENCE_MS) => {
                    report.outcomes.push((subject.state_id.clone(), VerificationOutcome::Failed));
                }
                _ => {}
            }
        }
        // A first divergence is a strike, not a failure (C3): the struck provider is audited again
        // later, on another prompt, and only a second divergence fails it.
        let outside_writer = probe_writer.filter(|w| !self.is_self_serve(w));
        let (report, follow_up) = {
            let mut strikes = self.strikes.lock().unwrap_or_else(|e| e.into_inner());
            strike_first_failures(
                report,
                &|peer| reputation_key(peer, key_of.get(peer).map_or(model, String::as_str)),
                // A replay pass as the one audited, or a passing follow-up on written questions (a
                // different, unpredictable request), clears it: with replay off, follow-ups are the only
                // way (R6-2). Not questions by the outside writer whose probe gave the strike (T8-4).
                &|peer, struck_by| {
                    peer == subject.state_id
                        && strike_cleared_by(
                            served.is_some(),
                            strict_subject,
                            probe_writer,
                            probe_writer.is_some_and(|w| self.is_self_serve(w)),
                            struck_by,
                        )
                },
                outside_writer,
                &mut strikes,
                now,
            )
        };
        for peer in &follow_up {
            if let Some(provider) = who.get(peer) {
                let delay = Duration::from_millis(rand::Rng::gen_range(&mut rand::thread_rng(), FOLLOW_UP_DELAY_MS));
                tracing::info!(model, peer = %peer, after = ?delay, "audit: outvoted once; audited again on another prompt before it counts");
                self.spawn_follow_up(model, provider, delay);
            }
        }
        // What disagreed, for whoever looks into an audit that didn't agree: sizes and digests only,
        // never the text (a replayed answer can echo a user's prompt, review round 4 M9).
        if report.verdict != RedundantVerdict::Agree {
            for (peer, answer) in &results {
                let summary = answer.as_ref().map(|a| {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    a.hash(&mut h);
                    format!("{} chars, digest {:016x}", a.chars().count(), h.finish())
                });
                tracing::debug!(model, peer = %peer, answer = ?summary, "audit answer");
            }
        }

        // On a replayed user request, an agreeing provider's pass is evidence about the traffic it
        // serves (D3), and may raise its price to a verified bigger model's: only when it is verified
        // as one now and another provider that agreed was verified as that same model too (review
        // round 4 H4, round 5 R5T-6). Agreeing with providers of the name's usual (smaller) model
        // shows it serves that one; an unverified provider's pass is an ordinary pass.
        let identities: HashMap<&String, Option<openhydra_protocol::logprob::ModelIdentity>> = report
            .outcomes
            .iter()
            .filter(|(_, o)| *o == VerificationOutcome::Honored)
            .map(|(p, _)| (p, self.verified_identity(p, key_of.get(p).map_or(model, String::as_str), now)))
            .collect();
        let vouched = |peer: &String| -> bool {
            match identities.get(peer).cloned().flatten() {
                None => false,
                Some(id) => identities.iter().any(|(q, other)| *q != peer && other.as_ref() == Some(&id)),
            }
        };

        // Phase 4a: record each audit outcome under the SAME per-(peer, model) key the serve path
        // uses, so an audit `Failed` downranks the provider on this model specifically.
        for (peer_id, outcome) in &report.outcomes {
            let model_key = key_of.get(peer_id).map_or(model, String::as_str);
            self.record_outcome(peer_id, model_key, *outcome, now);
            // Phase 4: fold the substantive temp=0 verdicts into the per-(peer,model) audit history
            // — an agreed serve is a Pass (lifts `n`), a clear divergence a Fail. A non-responder
            // (`Rejected`, a timeout) is not model-fraud evidence, so it never touches the history.
            let class = match outcome {
                VerificationOutcome::Honored => Some(AuditClass::Pass),
                VerificationOutcome::Failed => Some(AuditClass::Fail),
                VerificationOutcome::Rejected => None,
            };
            if let Some(class) = class {
                let real_traffic = served.is_some() && (class != AuditClass::Pass || vouched(peer_id));
                self.record_audit(peer_id, model_key, class, real_traffic, now);
            }
        }
        Ok(report)
    }

    /// The model a provider has verified it serves, by its current log-prob pass streak against a
    /// reference whose identity is known; `None` without one.
    fn verified_identity(&self, peer_id: &str, model_key: &str, now_ms: u64) -> Option<openhydra_protocol::logprob::ModelIdentity> {
        let lp = self.provider_logprob.as_ref()?;
        let status = lp.status.lock().unwrap_or_else(|e| e.into_inner()).get(&reputation_key(peer_id, model_key)).copied()?;
        if !status.evidence(now_ms).0 {
            return None;
        }
        self.reference_identity(status.reference)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_reply(chunks: &[crate::serve::ServeChunk]) -> Vec<u8> {
        crate::serve::frame_response(&chunks.iter().map(|c| c.encode()).collect::<Vec<_>>())
    }

    fn read_deltas(n: usize) -> crate::serve::StreamTally {
        let mut tally = crate::serve::StreamTally::default();
        for _ in 0..n {
            tally.push_encoded(&crate::serve::ServeChunk::Delta("ab".into()).encode());
        }
        tally
    }

    #[test]
    fn a_close_reply_adds_at_most_its_cap_however_large_its_frames() {
        use crate::serve::ServeChunk;
        // Round 8 S8-1: few fragments, each huge, then an inflated Done.
        for max_tokens in [None, Some(300)] {
            let mut tally = read_deltas(10);
            let ceiling = tally.billable(max_tokens).saturating_add(close_tail_cap(max_tokens));
            let mut chunks: Vec<ServeChunk> = (0..20).map(|_| ServeChunk::Delta("x".repeat(1 << 20))).collect();
            chunks.push(ServeChunk::Done { tokens: 100_000_000, metrics: Default::default() });
            absorb_close_reply(&mut tally, &close_reply(&chunks), max_tokens);
            let billed = tally.billable(max_tokens).min(ceiling);
            assert_eq!(billed, max_tokens.map_or(10 + CLOSE_TAIL_MAX_TOKENS, |m| u64::from(m).min(10 + u64::from(m))));
        }
    }

    #[test]
    fn delivered_text_is_capped_flat_so_media_results_fit() {
        // A 512x512 PNG as a data URL is ~400 KB however small max_tokens is (round 9 B9-1).
        assert!(delivered_text_cap(Some(2048)) > 100 * 1024 * 1024);
        assert_eq!(delivered_text_cap(None), delivered_text_cap(Some(1)));
    }

    #[test]
    fn an_honest_close_reply_is_paid_in_full() {
        use crate::serve::ServeChunk;
        let mut tally = read_deltas(10);
        let ceiling = tally.billable(None).saturating_add(close_tail_cap(None));
        let mut chunks: Vec<ServeChunk> = (0..5).map(|_| ServeChunk::Delta("ab".into())).collect();
        chunks.push(ServeChunk::Done { tokens: 15, metrics: Default::default() });
        absorb_close_reply(&mut tally, &close_reply(&chunks), None);
        assert_eq!(tally.billable(None).min(ceiling), 15);
    }

    #[test]
    fn two_keys_sharing_a_64_bit_id_keep_separate_standing() {
        // Two providers whose (verified) records carry the same 64-bit id under different keys — a
        // ~2^32 collision. Each is ranked as itself, under its own 128-bit state id, and what one
        // earned is never applied to the other.
        let (ka, kb) = ([7u8; 32], [9u8; 32]);
        let (sa, sb) = (
            openhydra_network::identity::openhydra_state_id(&ka),
            openhydra_network::identity::openhydra_state_id(&kb),
        );
        let mut a = discovered("0123456789abcdef", "", 10.0);
        a.public_key = hex::encode(ka);
        a.libp2p_peer_id = "A".into();
        let mut b = discovered("0123456789abcdef", "", 10.0);
        b.public_key = hex::encode(kb);
        b.libp2p_peer_id = "B".into();
        let asked = std::cell::RefCell::new(Vec::new());
        let ranked = rank_providers_with_reputation(&[a, b], "", 0, &|pid, _| {
            asked.borrow_mut().push(pid.to_string());
            (pid == sa).then_some(99.0) // only A has earned anything
        });
        assert_eq!(ranked.len(), 2, "both kept, neither shadowing the other");
        let ids: Vec<(&str, &str)> = ranked.iter().map(|p| (p.libp2p_peer_id.as_str(), p.state_id.as_str())).collect();
        assert!(ids.contains(&("A", sa.as_str())) && ids.contains(&("B", sb.as_str())), "{ids:?}");
        assert_eq!(ranked[0].libp2p_peer_id, "A", "A's earned reputation ranks it first; B doesn't inherit it");
        assert!(asked.borrow().contains(&sb), "B is looked up under its own id");
    }

    #[test]
    fn a_state_id_extends_the_peer_id_and_legacy_peers_keep_theirs() {
        let key = [3u8; 32];
        let peer = openhydra_network::identity::openhydra_peer_id(&key);
        let state = provider_state_id(&peer, &hex::encode(key));
        assert_eq!(state.len(), 32);
        assert!(state.starts_with(&peer), "the 64-bit id is its prefix");
        assert_eq!(provider_state_id("legacypeer", ""), "legacypeer");
        assert_eq!(provider_state_id("legacypeer", "not-hex"), "legacypeer");
    }

    #[test]
    fn vouchers_of_two_keys_with_one_peer_id_are_separate() {
        // A colliding pair of provider keys under one 64-bit id: separate running totals.
        let a = voucher_key("0123456789abcdef", "qwen", &[1u8; 32]);
        let b = voucher_key("0123456789abcdef", "qwen", &[2u8; 32]);
        assert_ne!(a, b);
        assert!(a.starts_with(&reputation_key("0123456789abcdef", "qwen")));
    }

    use crate::adapter::{DetectedModel, EngineAdapter, InferenceRequest, ServeOutcome};
    use crate::provider::handle_serve_inbound;

    #[test]
    fn a_consumer_adopts_only_its_own_receipt_from_a_provider() {
        use crate::receipt::LatestReceipt;
        use ed25519_dalek::{Signer, SigningKey};
        use openhydra_protocol::receipts::{cosign_bytes, payload_from_bytes, CoSignedReceipt};
        let consumer = SigningKey::from_bytes(&[7u8; 32]);
        let provider = SigningKey::from_bytes(&[9u8; 32]);
        let (cpub, ppub) = (consumer.verifying_key().to_bytes(), provider.verifying_key().to_bytes());
        let signed_at = |signer: &SigningKey, nonce: u8, model: &str, cumulative: u64, peg: u64| {
            let payload = payload_from_bytes(&ppub, &cpub, model, 10, &[nonce; 16], 1, None, "",
                Some(cumulative), Some(SettlementAsset::Usdc), Some(peg)).unwrap();
            let consumer_sig = signer.sign(&payload.canonical_bytes());
            let provider_sig = provider.sign(&cosign_bytes(&payload, &consumer_sig));
            LatestReceipt { receipt: CoSignedReceipt { payload, consumer_sig, provider_sig }, claim: None }
        };
        let receipt = |signer: &SigningKey, nonce: u8, model: &str, cumulative: u64| signed_at(signer, nonce, model, cumulative, 1);
        let judge_where = |r: &LatestReceipt, base: u64, here: bool| {
            let signed_here = |_: &LatestReceipt| here;
            let cx = BehindContext {
                consumer_pub: &cpub,
                provider_pub: &ppub,
                model_id: "m",
                nonce: &[2u8; 16],
                base,
                peg: Some(1),
                signed_here: &signed_here,
            };
            judge_behind_receipt(r, &cx)
        };
        let judge = |r: &LatestReceipt, base: u64| judge_where(r, base, true);
        // An earlier serve's receipt past what this serve built on: adopt it.
        assert_eq!(judge(&receipt(&consumer, 1, "m", 500), 300), BehindReceipt::Adopt(500));
        // Not past it: nothing to catch up to. That includes a provider refusing this serve's amount
        // while holding a receipt we had already built on (review S6: adopting it would charge this
        // serve twice).
        assert_eq!(judge(&receipt(&consumer, 1, "m", 500), 500), BehindReceipt::Stale);
        assert_eq!(judge(&receipt(&consumer, 1, "m", 400), 500), BehindReceipt::Stale);
        // This serve's own receipt (its co-sign reply was lost): it settled.
        assert_eq!(judge(&receipt(&consumer, 2, "m", 500), 300), BehindReceipt::ThisServe(500));
        // Signed by someone else, or for another model: never adopted.
        assert_eq!(judge(&receipt(&SigningKey::from_bytes(&[8u8; 32]), 1, "m", 9_999), 0), BehindReceipt::NotOurs);
        assert_eq!(judge(&receipt(&consumer, 1, "other", 9_999), 0), BehindReceipt::NotOurs);
        // Stamped at another peg: its amount is still USDC micros, so it is ours to build on (S7).
        assert_eq!(judge(&signed_at(&consumer, 1, "m", 9_999, 2), 0), BehindReceipt::Adopt(9_999));
        // Ours, but not shown to be signed for the deployment we settle on (review S1).
        assert_eq!(judge_where(&receipt(&consumer, 1, "m", 9_999), 0, false), BehindReceipt::NotThisDeployment);
    }

    #[test]
    fn audit_partners_come_from_different_networks() {
        assert_eq!(network_of("203.0.113.7").as_deref(), Some("203.0.113.0/24"));
        assert_eq!(network_of("2001:db8:abcd:12::1").as_deref(), Some("2001:db8:abcd::/48"));
        assert_eq!(network_of("[2001:db8:abcd:12::1]").as_deref(), Some("2001:db8:abcd::/48"));
        assert_eq!(network_of("not an ip"), None);
        let a = vec!["203.0.113.0/24".to_string()];
        let b = vec!["198.51.100.0/24".to_string()];
        assert!(independent_partner(&[], 0, &a));
        assert!(independent_partner(&[a.clone()], 0, &b));
        assert!(!independent_partner(&[a.clone()], 0, &a), "same network");
        // One partner we only reach through a relay may join; a second may not.
        assert!(independent_partner(&[a.clone()], 0, &[]));
        assert!(!independent_partner(&[a.clone(), vec![]], 1, &[]));
        assert!(independent_partner(&[vec![]], 0, &b));
    }

    #[test]
    fn only_a_claim_signed_for_this_deployment_shows_a_receipt_belongs_to_it() {
        use crate::receipt::{claim_for, ClaimSig, LatestReceipt};
        use crate::settlement_config::SettlementChain;
        use ed25519_dalek::{Signer, SigningKey};
        use openhydra_protocol::receipts::{cosign_bytes, payload_from_bytes, CoSignedReceipt};
        use openhydra_protocol::settlement::{ChainDomain, SolanaDomain};
        let consumer = SigningKey::from_bytes(&[7u8; 32]);
        let provider = SigningKey::from_bytes(&[9u8; 32]);
        let (cpub, ppub) = (consumer.verifying_key().to_bytes(), provider.verifying_key().to_bytes());
        let payload = payload_from_bytes(&ppub, &cpub, "m", 10, &[1; 16], 1, None, "",
            Some(500), Some(SettlementAsset::Usdc), Some(1)).unwrap();
        let consumer_sig = consumer.sign(&payload.canonical_bytes());
        let provider_sig = provider.sign(&cosign_bytes(&payload, &consumer_sig));
        let receipt = CoSignedReceipt { payload, consumer_sig, provider_sig };
        let here = ChainDomain::Solana(SolanaDomain { program_id: [4; 32], cluster: [5; 32] });
        let elsewhere = ChainDomain::Solana(SolanaDomain { program_id: [4; 32], cluster: [6; 32] });
        let with_claim_for = |domain: &ChainDomain| {
            let claim = claim_for(&receipt.payload, 1_800_000_000).unwrap();
            let sig = consumer.sign(&domain.claim_message(&claim)).to_bytes().to_vec();
            LatestReceipt { receipt: receipt.clone(), claim: Some(ClaimSig { expires_at: 1_800_000_000, sig }) }
        };
        let chain = SettlementChain { domain: here.clone(), evm_wallet: None };
        assert!(signed_for_deployment(Some(&chain), &with_claim_for(&here)));
        assert!(!signed_for_deployment(Some(&chain), &with_claim_for(&elsewhere)), "another cluster's claim");
        let bare = LatestReceipt { receipt: receipt.clone(), claim: None };
        assert!(!signed_for_deployment(Some(&chain), &bare), "no claim: nothing shows where it was signed");
        assert!(signed_for_deployment(None, &bare), "off-chain there is no deployment to tell apart");
    }

    #[test]
    fn only_a_provider_on_the_shared_billing_rule_is_faulted_for_over_reporting() {
        let summary = |tokens: u64, billing: u8| crate::serve::ServeSummary {
            tokens,
            ok: true,
            metrics: crate::serve::ServeMetrics { billing, ..Default::default() },
            ..Default::default()
        };
        assert!(over_reported_by_billing(&summary(500, crate::serve::BILLING_VERSION), 40, None));
        assert!(!over_reported_by_billing(&summary(40, crate::serve::BILLING_VERSION), 40, None));
        // An older provider's engine count can honestly be higher: signed for less, not faulted.
        assert!(!over_reported_by_billing(&summary(500, 0), 40, None));
        assert!(!over_reported_by_billing(&summary(500, 0), 40, Some(500)));
        // …but not above what the request allowed, which no rule ever billed (B6).
        assert!(over_reported_by_billing(&summary(501, 0), 40, Some(500)));
        assert!(over_reported_by_billing(&summary(501, 2), 40, Some(500)));
        // Rule 3 bills a completed serve as 4 does.
        assert!(over_reported_by_billing(&summary(500, 3), 40, None));
        // Nor is a newer rule, which we can't judge by ours.
        assert!(!over_reported_by_billing(&summary(500, crate::serve::BILLING_VERSION + 1), 40, Some(10)));
        assert!(newer_billing_rule(&summary(500, crate::serve::BILLING_VERSION + 1)));
        assert!(!newer_billing_rule(&summary(500, crate::serve::BILLING_VERSION)));
    }

    #[test]
    fn settlements_are_bounded_overall_and_per_pair() {
        assert!(settle_admitted(0, 0));
        assert!(settle_admitted(MAX_SETTLES_IN_FLIGHT - 1, MAX_SETTLES_PER_PAIR - 1));
        assert!(!settle_admitted(MAX_SETTLES_IN_FLIGHT, 0), "the global cap");
        assert!(!settle_admitted(0, MAX_SETTLES_PER_PAIR), "a pair already queued deep enough");
        // The slot gives its place back however the settlement ends.
        let n = Arc::new(std::sync::atomic::AtomicUsize::new(1));
        drop(SettleSlot(Arc::clone(&n)));
        assert_eq!(n.load(std::sync::atomic::Ordering::Acquire), 0);
    }

    #[test]
    fn prune_idle_settle_locks_evicts_only_unheld_entries() {
        // F1 follow-up: the settlement-lock map must stay bounded. An idle pair (only the map holds
        // its Arc) is evicted; a pair with an in-flight settle (an extra Arc clone alive, as
        // `settle_receipt` captures before locking) is retained so serialization is never broken.
        let mut locks: HashMap<String, Arc<Mutex<()>>> = HashMap::new();
        locks.insert("idle-a".to_string(), Arc::new(Mutex::new(())));
        locks.insert("idle-b".to_string(), Arc::new(Mutex::new(())));
        let held = Arc::new(Mutex::new(()));
        locks.insert("busy".to_string(), held.clone()); // an in-flight settle holds this clone

        prune_idle_settle_locks(&mut locks);

        assert!(locks.contains_key("busy"), "a pair with an in-flight settle is never evicted");
        assert!(!locks.contains_key("idle-a"), "idle entries are swept");
        assert!(!locks.contains_key("idle-b"));
        assert_eq!(locks.len(), 1);
        drop(held);
    }

    /// A canned engine: emits fixed deltas (or fails).
    struct StubAdapter {
        deltas: Vec<&'static str>,
        tokens: u64,
        fail: Option<&'static str>,
    }
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
            if let Some(e) = self.fail {
                return Err(AdapterError::Http(e.into()));
            }
            for d in &self.deltas {
                on_delta(d);
            }
            Ok(ServeOutcome { tokens: self.tokens, done: true, engine: Default::default(), tool_calls: Vec::new() })
        }
    }

    fn request() -> ServeRequest {
        ServeRequest {
            reply_to: "12D3KooWConsumer".into(),
            model_ref: "qwen2.5:7b".into(),
            messages: vec![ChatMessage { role: "user".into(), content: "hi".into(), ..Default::default() }],
            max_tokens: Some(64),
            temperature: None,
            tools: Vec::new(),
            think: None,
            nonce: [0u8; 16],
            credential: None,
            deterministic: None,
            settlement_peg: None,
            settlement_domain: None,
            settlement_wallet: None,
        }
    }

    #[test]
    fn attempt_timeout_scales_with_max_tokens_and_clamps() {
        // A tiny request stays at the floor (fast failover from a dead provider).
        assert_eq!(attempt_timeout(Some(10)).as_secs(), ATTEMPT_TIMEOUT_MIN_SECS);
        // A small request is just above the floor: 45 + 64·40/1000 = 47s.
        assert_eq!(attempt_timeout(Some(64)).as_secs(), 47);
        // A mid request gets room: 45 + 4096·40/1000 = 208s.
        assert_eq!(attempt_timeout(Some(4096)).as_secs(), 208);
        // A huge request clamps to the ceiling, not unbounded.
        assert_eq!(attempt_timeout(Some(1_000_000)).as_secs(), ATTEMPT_TIMEOUT_MAX_SECS);
        // No max_tokens → the default-estimate budget, still within bounds.
        let d = attempt_timeout(None).as_secs();
        assert!((ATTEMPT_TIMEOUT_MIN_SECS..=ATTEMPT_TIMEOUT_MAX_SECS).contains(&d));
        // The env override wins and is itself clamped. (Test-local; safe — single-threaded.)
        std::env::set_var("OPENHYDRA_ATTEMPT_TIMEOUT_SECS", "9999");
        assert_eq!(attempt_timeout(Some(64)).as_secs(), ATTEMPT_TIMEOUT_MAX_SECS);
        std::env::remove_var("OPENHYDRA_ATTEMPT_TIMEOUT_SECS");
    }

    #[test]
    fn round_trips_request_through_the_provider_handler() {
        // The mock transport IS the provider: consumer encodes → provider serves → consumer
        // parses. Proves the full in-process loop without a swarm.
        let adapter = StubAdapter { deltas: vec!["Hello", " world"], tokens: 5, fail: None };
        let mut transport = |req: &[u8]| -> Result<Vec<u8>, AdapterError> {
            Ok(handle_serve_inbound(req, &adapter))
        };
        let mut out = String::new();
        let summary =
            request_completion(&mut transport, &request(), &mut |d| out.push_str(d)).unwrap();
        assert_eq!(out, "Hello world");
        assert_eq!(summary.tokens, 5);
        assert!(summary.ok);
    }

    #[test]
    fn tool_calls_flow_back_through_the_provider_loop() {
        // The full in-process loop for a tool-call turn: the provider's engine returns tool
        // calls (no text), and the consumer surfaces them on the summary with an empty stream.
        use crate::adapter::{ToolCall, ToolCallFunction};
        struct ToolStub;
        impl EngineAdapter for ToolStub {
            fn engine_name(&self) -> &'static str {
                "toolstub"
            }
            fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
                Ok(vec![])
            }
            fn serve_stream(
                &self,
                _req: &InferenceRequest,
                _on_delta: &mut dyn FnMut(&str),
            ) -> Result<ServeOutcome, AdapterError> {
                Ok(ServeOutcome {
                    tokens: 1,
                    done: true,
                    engine: Default::default(),
                    tool_calls: vec![ToolCall {
                        id: "call_1".into(),
                        kind: "function".into(),
                        function: ToolCallFunction { name: "search".into(), arguments: r#"{"q":"rust"}"#.into() },
                    }],
                })
            }
        }
        let mut transport = |req: &[u8]| -> Result<Vec<u8>, AdapterError> {
            Ok(handle_serve_inbound(req, &ToolStub))
        };
        let mut text = String::new();
        let summary =
            request_completion(&mut transport, &request(), &mut |d| text.push_str(d)).unwrap();
        assert!(text.is_empty(), "a tool-call turn streams no text");
        assert_eq!(summary.tool_calls.len(), 1);
        assert_eq!(summary.tool_calls[0].function.name, "search");
        assert_eq!(summary.tool_calls[0].function.arguments, r#"{"q":"rust"}"#);
    }

    #[test]
    fn provider_error_frame_becomes_an_error() {
        let adapter = StubAdapter { deltas: vec![], tokens: 0, fail: Some("engine down") };
        let mut transport = |req: &[u8]| Ok(handle_serve_inbound(req, &adapter));
        let err = request_completion(&mut transport, &request(), &mut |_| {}).unwrap_err();
        assert!(matches!(err, AdapterError::Provider(m) if m.contains("engine down")));
    }

    #[test]
    fn transport_failure_propagates() {
        let mut transport =
            |_: &[u8]| -> Result<Vec<u8>, AdapterError> { Err(AdapterError::Http("no route".into())) };
        let err = request_completion(&mut transport, &request(), &mut |_| {}).unwrap_err();
        assert!(matches!(err, AdapterError::Http(_)));
    }

    #[test]
    fn missing_done_frame_is_an_error() {
        // A response with only a delta (no Done) → the consumer flags an incomplete stream.
        let mut transport = |_: &[u8]| {
            Ok(crate::serve::frame_response(&[ServeChunk::Delta("x".into()).encode()]))
        };
        let err = request_completion(&mut transport, &request(), &mut |_| {}).unwrap_err();
        assert!(matches!(err, AdapterError::Parse(_)));
    }

    // ── P1c: streaming serve client ─────────────────────────────────────────

    /// Encode a `FetchChunksResponse::Chunks` reply from a set of encoded frames.
    fn chunks_reply(frames: &[Vec<u8>], next_offset: u32, done: bool) -> Vec<u8> {
        FetchChunksResponse::Chunks { framed: crate::serve::frame_response(frames), next_offset, done }
            .encode()
    }

    /// The full encoded serve frames an adapter would emit for `req` (what the streaming provider
    /// reveals incrementally, and what the buffered provider returns whole).
    fn serve_frames(adapter: &dyn EngineAdapter, req: &ServeRequest) -> Vec<Vec<u8>> {
        let mut frames = Vec::new();
        crate::serve::handle_serve_request(&req.encode(), adapter, &mut |c| frames.push(c.to_vec()));
        frames
    }

    /// A faithful in-test streaming provider: reveals `reveal_per_poll` more frames on each poll
    /// (simulating tokens arriving between polls) and serves the incremental slice from the polled
    /// offset, pruning the acknowledged prefix exactly like the real provider buffer.
    struct MockStreamProvider {
        pending: std::collections::VecDeque<Vec<u8>>,
        buf: Vec<Vec<u8>>,
        base: u32,
        accept: bool,
        reveal_per_poll: usize,
    }
    impl MockStreamProvider {
        fn new(frames: Vec<Vec<u8>>, accept: bool, reveal_per_poll: usize) -> Self {
            Self { pending: frames.into(), buf: Vec::new(), base: 0, accept, reveal_per_poll }
        }
        fn respond(&mut self, framed: &[u8]) -> Result<Vec<u8>, AdapterError> {
            match framed.first() {
                Some(&SERVE_STREAM) if self.accept => Ok(chunks_reply(&[], 0, false)),
                // A legacy provider answers the unknown method byte with an error blob.
                Some(&SERVE_STREAM) => Ok(crate::serve::frame_response(&[
                    ServeChunk::Error("unsupported method".into()).encode()
                ])),
                Some(&FETCH_CHUNKS) => {
                    let off = u32::from_be_bytes(framed[17..21].try_into().unwrap());
                    for _ in 0..self.reveal_per_poll {
                        if let Some(f) = self.pending.pop_front() {
                            self.buf.push(f);
                        }
                    }
                    let have_hi = self.base + self.buf.len() as u32;
                    let start_abs = off.clamp(self.base, have_hi);
                    let li = (start_abs - self.base) as usize;
                    let out = crate::serve::frame_response(&self.buf[li..]);
                    let done = self.buf.last().map(|f| ServeChunk::frame_is_terminal(f)).unwrap_or(false);
                    if li > 0 {
                        self.buf.drain(0..li);
                        self.base = start_abs;
                    }
                    Ok(chunks_reply_raw(out, have_hi, done))
                }
                _ => Ok(crate::serve::frame_response(&[ServeChunk::Error("unsupported".into()).encode()])),
            }
        }
    }
    fn chunks_reply_raw(framed: Vec<u8>, next_offset: u32, done: bool) -> Vec<u8> {
        FetchChunksResponse::Chunks { framed, next_offset, done }.encode()
    }

    #[test]
    fn streaming_delivers_incrementally_and_matches_the_buffered_result() {
        let adapter = StubAdapter { deltas: vec!["Hello", ", ", "world"], tokens: 7, fail: None };
        // Buffered reference: the whole completion in one blob.
        let mut buffered_text = String::new();
        let mut buf_tp = |req: &[u8]| Ok(handle_serve_inbound(req, &adapter));
        let buf_summary =
            request_completion(&mut buf_tp, &request(), &mut |d| buffered_text.push_str(d)).unwrap();

        // Streaming: the same frames revealed one-per-poll.
        let frames = serve_frames(&adapter, &request());
        let mut mock = MockStreamProvider::new(frames, true, 1);
        let mut transport = |f: &[u8]| mock.respond(f);
        let mut sleeps: Vec<Duration> = Vec::new();
        let mut sleeper = |d: Duration| sleeps.push(d);
        let mut stream_text = String::new();
        let stream_summary = request_completion_streaming(
            &mut transport,
            &mut sleeper,
            &request(),
            &mut |d| stream_text.push_str(d),
        )
        .unwrap_or_else(|_| panic!("streaming should complete"));

        assert_eq!(stream_text, buffered_text, "streamed text must equal the buffered text");
        assert_eq!(stream_text, "Hello, world");
        assert_eq!(stream_summary.tokens, buf_summary.tokens);
        // Multiple polls happened → it really streamed rather than returning one blob.
        assert!(sleeps.len() >= 2, "expected several polls, got {}", sleeps.len());
    }

    #[test]
    fn streaming_falls_back_to_buffered_on_a_legacy_provider() {
        // A provider that doesn't speak streaming answers the open with a legacy error blob; the
        // client reports Legacy so serve_once retries over the buffered path.
        let frames = serve_frames(&StubAdapter { deltas: vec!["x"], tokens: 1, fail: None }, &request());
        let mut mock = MockStreamProvider::new(frames, /*accept=*/ false, 1);
        let mut transport = |f: &[u8]| mock.respond(f);
        let mut sleeper = |_: Duration| {};
        let out = request_completion_streaming(&mut transport, &mut sleeper, &request(), &mut |_| {});
        assert!(matches!(out, Err(StreamFail::Legacy)));
    }

    #[test]
    fn streaming_overloaded_open_ack_is_a_failover_error() {
        // A pool-shed at open → Overloaded → a real error (serve_once fails over to another peer).
        let mut transport = |_: &[u8]| Ok(FetchChunksResponse::Overloaded.encode());
        let mut sleeper = |_: Duration| {};
        let out = request_completion_streaming(&mut transport, &mut sleeper, &request(), &mut |_| {});
        assert!(matches!(out, Err(StreamFail::Error(AdapterError::Overloaded(_)))));
    }

    #[test]
    fn streaming_interval_tightens_on_progress_and_backs_off_on_idle() {
        // progress → idle → progress → idle → done. The sleeps prove the interval resets to
        // POLL_MIN on each progress and doubles on each idle.
        let script = vec![
            chunks_reply(&[ServeChunk::Delta("a".into()).encode()], 1, false), // progress
            chunks_reply(&[], 1, false),                                        // idle
            chunks_reply(&[ServeChunk::Delta("b".into()).encode()], 2, false), // progress (reset)
            chunks_reply(&[], 2, false),                                        // idle
            chunks_reply(&[ServeChunk::Done { tokens: 2, metrics: Default::default() }.encode()], 3, true),
        ];
        let mut idx = 0usize;
        let mut transport = |f: &[u8]| -> Result<Vec<u8>, AdapterError> {
            if f.first() == Some(&SERVE_STREAM) {
                Ok(chunks_reply(&[], 0, false))
            } else {
                let r = script[idx].clone();
                idx += 1;
                Ok(r)
            }
        };
        let mut sleeps: Vec<Duration> = Vec::new();
        let mut sleeper = |d: Duration| sleeps.push(d);
        let mut text = String::new();
        let summary =
            request_completion_streaming(&mut transport, &mut sleeper, &request(), &mut |d| text.push_str(d))
                .unwrap_or_else(|_| panic!("should complete"));
        assert_eq!(text, "ab");
        assert_eq!(summary.tokens, 2);
        // sleeps: [after progress = MIN, after idle = 2·MIN, after progress = MIN (reset), after idle = 2·MIN]
        assert_eq!(sleeps.len(), 4);
        assert_eq!(sleeps[0], POLL_MIN, "tight after first progress");
        assert!(sleeps[1] > sleeps[0], "backs off on idle");
        assert_eq!(sleeps[2], POLL_MIN, "resets to tight after the next progress");
        assert!(sleeps[3] > sleeps[2], "backs off again on idle");
    }

    #[test]
    fn streaming_backoff_is_capped_at_poll_max() {
        // A permanently-idle stream: the interval doubles but never exceeds POLL_MAX, and the run
        // is bounded by the stall deadline (no real sleeping — the clock sums injected sleeps).
        let mut transport = |f: &[u8]| -> Result<Vec<u8>, AdapterError> {
            if f.first() == Some(&SERVE_STREAM) {
                Ok(chunks_reply(&[], 0, false))
            } else {
                Ok(chunks_reply(&[], 0, false)) // never any progress
            }
        };
        let mut sleeps: Vec<Duration> = Vec::new();
        let mut sleeper = |d: Duration| sleeps.push(d);
        let out = request_completion_streaming(&mut transport, &mut sleeper, &request(), &mut |_| {});
        assert!(matches!(out, Err(StreamFail::Error(_))), "all-idle stream must stall out");
        assert!(sleeps.iter().all(|d| *d <= POLL_MAX), "interval must never exceed POLL_MAX");
        assert!(sleeps.iter().any(|d| *d == POLL_MAX), "long idle should reach the POLL_MAX cap");
    }

    #[test]
    fn streaming_stalls_out_when_no_progress_before_the_deadline() {
        let mut transport = |f: &[u8]| -> Result<Vec<u8>, AdapterError> {
            if f.first() == Some(&SERVE_STREAM) {
                Ok(chunks_reply(&[], 0, false))
            } else {
                Ok(chunks_reply(&[], 0, false)) // never any progress
            }
        };
        let mut sleeper = |_: Duration| {}; // no real time; the stall clock sums injected sleeps
        let out = request_completion_streaming(&mut transport, &mut sleeper, &request(), &mut |_| {});
        assert!(matches!(out, Err(StreamFail::Error(AdapterError::Network(m))) if m.contains("stall")));
    }

    #[test]
    fn streaming_drop_resume_repolls_the_same_offset_and_completes() {
        // The first two polls fail at the transport (relay dropped); drop-resume re-polls the same
        // offset and the stream still completes with the full, correct output — no duplication.
        let adapter = StubAdapter { deltas: vec!["re", "sume"], tokens: 3, fail: None };
        let frames = serve_frames(&adapter, &request());
        let mut mock = MockStreamProvider::new(frames, true, 1);
        let mut fail_budget = 2u32;
        let mut transport = |f: &[u8]| -> Result<Vec<u8>, AdapterError> {
            if f.first() == Some(&FETCH_CHUNKS) && fail_budget > 0 {
                fail_budget -= 1;
                return Err(AdapterError::Http("relay dropped".into()));
            }
            mock.respond(f)
        };
        let mut sleeper = |_: Duration| {};
        let mut text = String::new();
        let summary =
            request_completion_streaming(&mut transport, &mut sleeper, &request(), &mut |d| text.push_str(d))
                .unwrap_or_else(|_| panic!("drop-resume should recover"));
        assert_eq!(text, "resume");
        assert_eq!(summary.tokens, 3);
    }

    #[test]
    fn streaming_gives_up_when_drops_exceed_the_retry_budget() {
        // The transport is permanently dead: every poll errors. Past the bounded retry budget the
        // client surfaces an error (serve_once then fails over / retries).
        let mut transport = |f: &[u8]| -> Result<Vec<u8>, AdapterError> {
            if f.first() == Some(&SERVE_STREAM) {
                Ok(chunks_reply(&[], 0, false))
            } else {
                Err(AdapterError::Http("dead".into()))
            }
        };
        let mut sleeper = |_: Duration| {};
        let out = request_completion_streaming(&mut transport, &mut sleeper, &request(), &mut |_| {});
        assert!(matches!(out, Err(StreamFail::Error(AdapterError::Network(m))) if m.contains("dropped")));
    }

    #[test]
    fn streaming_provider_error_frame_becomes_an_error() {
        let mut transport = |f: &[u8]| -> Result<Vec<u8>, AdapterError> {
            if f.first() == Some(&SERVE_STREAM) {
                Ok(chunks_reply(&[], 0, false))
            } else {
                Ok(chunks_reply(&[ServeChunk::Error("engine exploded".into()).encode()], 1, true))
            }
        };
        let mut sleeper = |_: Duration| {};
        let out = request_completion_streaming(&mut transport, &mut sleeper, &request(), &mut |_| {});
        assert!(matches!(out, Err(StreamFail::Error(AdapterError::Provider(m))) if m.contains("engine exploded")));
    }

    #[test]
    fn streaming_carries_tool_calls_through_to_the_summary() {
        use crate::adapter::{ToolCall, ToolCallFunction};
        let tc = ServeChunk::ToolCalls(vec![ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: ToolCallFunction { name: "search".into(), arguments: r#"{"q":"rust"}"#.into() },
        }])
        .encode();
        let mut transport = |f: &[u8]| -> Result<Vec<u8>, AdapterError> {
            if f.first() == Some(&SERVE_STREAM) {
                Ok(chunks_reply(&[], 0, false))
            } else {
                Ok(chunks_reply(&[tc.clone(), ServeChunk::Done { tokens: 1, metrics: Default::default() }.encode()], 2, true))
            }
        };
        let mut sleeper = |_: Duration| {};
        let mut text = String::new();
        let summary =
            request_completion_streaming(&mut transport, &mut sleeper, &request(), &mut |d| text.push_str(d))
                .unwrap_or_else(|_| panic!("should complete"));
        assert!(text.is_empty(), "a tool-call turn streams no text");
        assert_eq!(summary.tool_calls.len(), 1);
        assert_eq!(summary.tool_calls[0].function.name, "search");
    }

    #[test]
    fn stream_total_cap_scales_and_never_cuts_a_legit_serve() {
        // F2 no-regression guard: the cap is provably larger than any real serve.
        // Small requests clamp to the 30-min FLOOR (never abort quick work).
        assert_eq!(stream_total_cap_scaled(Some(64)), STREAM_TOTAL_FLOOR);
        assert_eq!(stream_total_cap_scaled(Some(700)), STREAM_TOTAL_FLOOR); // 300+1400=1700 < 1800 floor
        assert!(stream_total_cap_scaled(None) >= STREAM_TOTAL_FLOOR);
        // A big legit output scales UP (8k tokens → 300 + 16000 = 16300s ≈ 4.5h), well above realistic.
        let big = stream_total_cap_scaled(Some(8000));
        assert!(big > STREAM_TOTAL_FLOOR && big < STREAM_TOTAL_CEILING, "got {big:?}");
        assert_eq!(big, Duration::from_secs(300 + 8000 * 2));
        // A pathological max_tokens clamps to the 6-h CEILING (still bounds the dribble DoS).
        assert_eq!(stream_total_cap_scaled(Some(u32::MAX)), STREAM_TOTAL_CEILING);
        // Every possible value is at least the floor — no legit serve is ever cut below 30 min.
        for t in [0u32, 1, 100, 750, 5000, 100_000, u32::MAX] {
            assert!(stream_total_cap_scaled(Some(t)) >= STREAM_TOTAL_FLOOR);
        }
    }

    #[test]
    fn stream_total_cap_override_floors_a_nonsensical_value() {
        // Pure (env-free) flooring rule: an override at/above the idle-stall is honored verbatim;
        // one below it (incl. a fat-fingered 0) is raised to the stall floor so it can never abort
        // every stream. Tested here rather than via env so it never races the parallel runner.
        assert_eq!(stream_total_cap_override(600), Duration::from_secs(600), "honored as-is");
        assert_eq!(stream_total_cap_override(STREAM_STALL_TIMEOUT.as_secs()), STREAM_STALL_TIMEOUT);
        assert_eq!(stream_total_cap_override(0), STREAM_STALL_TIMEOUT, "0 floored, not instant-abort");
        assert_eq!(stream_total_cap_override(7), STREAM_STALL_TIMEOUT, "below-floor raised");
    }

    #[test]
    fn stream_total_cap_env_override_wins() {
        // The ONLY test that touches the process-global override env var (the scaling + flooring
        // math is tested via `stream_total_cap_scaled` / `stream_total_cap_override`, which never
        // read env), so no cross-test race. A value above the floor rides through unchanged.
        std::env::set_var("OPENHYDRA_STREAM_MAX_TOTAL_SECS", "600");
        assert_eq!(stream_total_cap(Some(64)), Duration::from_secs(600));
        std::env::remove_var("OPENHYDRA_STREAM_MAX_TOTAL_SECS");
        assert_eq!(stream_total_cap(Some(64)), STREAM_TOTAL_FLOOR, "reverts to the scaled cap");
    }

    // ── provider selection ──

    const TPL_A: &str = "qwen3.5/2b/fp16/aaaaaaaaaaaaaaaa";
    const TPL_B: &str = "qwen3.5/2b/fp16/bbbbbbbbbbbbbbbb";

    fn discovered(peer_id: &str, canonical: &str, throughput: f64) -> DiscoveredPeer {
        DiscoveredPeer {
            peer_id: peer_id.into(),
            libp2p_peer_id: format!("{peer_id}-libp2p"),
            public_key: format!("{peer_id}-pk"),
            model_id: "m".into(),
            canonical_model_id: canonical.into(),
            throughput_tok_s: throughput,
            ..Default::default()
        }
    }

    #[test]
    fn selects_highest_ranked_compatible_provider() {
        let peers = vec![discovered("slow", TPL_A, 5.0), discovered("fast", TPL_A, 45.0)];
        let sel = select_provider(&peers, "qwen3.5/2b/*/*", 2).unwrap();
        assert_eq!(sel.peer_id, "fast");
        assert_eq!(sel.libp2p_peer_id, "fast-libp2p"); // dial target
        assert_eq!(sel.public_key, "fast-pk"); // for the receipt
    }

    #[test]
    fn filters_incompatible_canonical_id() {
        let peers = vec![discovered("wrong", TPL_B, 99.0), discovered("right", TPL_A, 1.0)];
        let sel = select_provider(&peers, TPL_A, 2).unwrap();
        assert_eq!(sel.peer_id, "right"); // incompatible dropped despite higher throughput
    }

    #[test]
    fn keeps_legacy_provider_without_canonical_id() {
        let sel = select_provider(&[discovered("legacy", "", 10.0)], "qwen3.5/2b/*/*", 2).unwrap();
        assert_eq!(sel.peer_id, "legacy");
    }

    #[test]
    fn none_when_nothing_compatible_or_empty() {
        let peers = vec![discovered("x", "gemma-4/e4b/fp16/cccccccccccccccc", 10.0)];
        assert!(select_provider(&peers, TPL_A, 2).is_none());
        assert!(select_provider(&[], TPL_A, 2).is_none());
    }

    #[test]
    fn rank_providers_returns_all_compatible_best_first_for_failover() {
        // The failover list: every compatible provider, ranked — so a dead top pick can
        // hand off to the next. Incompatible ones are still dropped.
        let peers = vec![
            discovered("slow", TPL_A, 5.0),
            discovered("incompatible", TPL_B, 99.0),
            discovered("fast", TPL_A, 45.0),
        ];
        let ranked = rank_providers(&peers, TPL_A, 2);
        let ids: Vec<&str> = ranked.iter().map(|p| p.peer_id.as_str()).collect();
        assert_eq!(ids, vec!["fast", "slow"]); // best first, incompatible excluded
        assert!(rank_providers(&[], TPL_A, 2).is_empty());
    }

    #[test]
    fn rank_providers_floats_connected_above_higher_scored_disconnected() {
        // R-DHT-8: a live (connected) provider is preferred even when a
        // disconnected one scores higher on throughput — a stale-but-advertised
        // record shouldn't be dialed first and eat a failover round-trip.
        let mut fast_dead = discovered("fast_dead", TPL_A, 45.0);
        fast_dead.connected = false;
        let mut slow_live = discovered("slow_live", TPL_A, 5.0);
        slow_live.connected = true;
        let ranked = rank_providers(&[fast_dead, slow_live], TPL_A, 2);
        let ids: Vec<&str> = ranked.iter().map(|p| p.peer_id.as_str()).collect();
        // Connected first; the disconnected (higher-scored) one is still present
        // as a failover fallback, just after.
        assert_eq!(ids, vec!["slow_live", "fast_dead"]);
    }

    #[test]
    fn rank_providers_score_order_within_connected_group() {
        // Within the connected group, capability score still decides order.
        let mut fast = discovered("fast", TPL_A, 45.0);
        fast.connected = true;
        let mut slow = discovered("slow", TPL_A, 5.0);
        slow.connected = true;
        let ranked = rank_providers(&[slow, fast], TPL_A, 2);
        let ids: Vec<&str> = ranked.iter().map(|p| p.peer_id.as_str()).collect();
        assert_eq!(ids, vec!["fast", "slow"]);
    }

    // ── M2.2(a): earned-reputation-aware ranking ──

    #[test]
    fn earned_reputation_downranks_a_provider_that_failed_us() {
        // The closed loop: two equal-throughput providers, both self-reporting neutral;
        // the one with a low EARNED reputation ranks last.
        let peers = vec![discovered("bad", TPL_A, 20.0), discovered("good", TPL_A, 20.0)];
        let earned = |pid: &str, _model: &str| if pid == "bad" { Some(5.0) } else { None };
        let ranked = rank_providers_with_reputation(&peers, TPL_A, 2, &earned);
        let ids: Vec<&str> = ranked.iter().map(|p| p.peer_id.as_str()).collect();
        assert_eq!(ids, vec!["good", "bad"], "low earned reputation must rank a provider last");
    }

    #[test]
    fn ranking_threads_the_provider_model_to_the_reputation_lookup() {
        // Phase 4a: the earned-reputation closure is asked per (peer, MODEL) — the ranker must pass
        // the provider's canonical model, so a low score keyed to model X only bites on model X.
        let peers = vec![discovered("p", TPL_A, 20.0)];
        // The reputation lookup is keyed by the provider's validated HANDLE (`model_id`, "m" in the
        // `discovered` helper), not its self-declared canonical.
        let seen = std::cell::RefCell::new(Vec::new());
        let spy = |_pid: &str, model: &str| {
            seen.borrow_mut().push(model.to_string());
            None
        };
        rank_providers_with_reputation(&peers, "", 2, &spy);
        assert_eq!(seen.into_inner(), vec!["m".to_string()], "lookup keyed by the provider's handle");
    }

    #[test]
    fn only_a_qualifying_pass_streak_moves_a_provider_to_a2() {
        use crate::prompt_score::{LOGPROB_MIN_PASSES, LOGPROB_MIN_SPAN_MS, LOGPROB_VALID_MS};
        use openhydra_protocol::logprob::LogprobVerdict;
        let pass = LogprobVerdict::Pass { aggregate: 0.97 };
        let fail = LogprobVerdict::Fail { aggregate: 1.62 };
        let resolve = |st: &LogprobStatus, now: u64| {
            let (flag, latest_outcome) = st.evidence(now);
            RedundantExecAuthenticity
                .verify(&AuthenticityEvidence {
                    reputation: 80.0,
                    audit_pass_rate: 1.0,
                    passing_audits: 5,
                    methods: MethodFlags { logprob_commitment: flag, ..MethodFlags::default() },
                    latest_outcome,
                })
                .authenticity
        };
        let day = 24 * 60 * 60 * 1000;
        let mut st = LogprobStatus::default();
        // No verdict: a proven provider stays at A1.
        assert_eq!(resolve(&st, day), AuthenticityTier::ReputationRedundant);
        // One pass, or several within a day, is not enough.
        for i in 0..LOGPROB_MIN_PASSES {
            st.on_verdict(pass, day + u64::from(i) * 60_000);
        }
        assert_eq!(resolve(&st, day + 3 * 60_000), AuthenticityTier::ReputationRedundant);
        // Passes spread over the minimum span → A2, while the latest stays fresh.
        let later = day + LOGPROB_MIN_SPAN_MS;
        st.on_verdict(pass, later);
        assert_eq!(resolve(&st, later + 1), AuthenticityTier::EnergyFingerprintPossession);
        assert_eq!(resolve(&st, later + LOGPROB_VALID_MS), AuthenticityTier::ReputationRedundant, "stale");
        // A fail pins the floor and resets the streak.
        st.on_verdict(fail, later + 2);
        assert_eq!(resolve(&st, later + 3), AuthenticityTier::SelfReported);
        st.on_verdict(pass, later + 4);
        assert_eq!(resolve(&st, later + 5), AuthenticityTier::ReputationRedundant);
        // A pass alone doesn't lift an unproven provider: A2 still needs the reputation gates.
        let unproven = RedundantExecAuthenticity.verify(&AuthenticityEvidence {
            reputation: NEUTRAL_REPUTATION,
            audit_pass_rate: 1.0,
            passing_audits: 0,
            methods: MethodFlags { logprob_commitment: true, ..MethodFlags::default() },
            latest_outcome: None,
        });
        assert_eq!(unproven.authenticity, AuthenticityTier::SelfReported);
    }

    #[test]
    fn a_failed_audit_holds_the_provider_at_the_floor_for_a_day() {
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
        let node = ConsumerNode::new(net);
        let t0 = 1_000_000;
        // A proven provider: reputation earned and audits passed → A1. `q` has the same history
        // without the failure below.
        for peer in ["p", "q"] {
            for i in 0..30 {
                node.record_outcome(peer, "m", VerificationOutcome::Honored, t0 + i);
            }
            for i in 0..12 {
                node.record_audit(peer, "m", AuditClass::Pass, false, t0 + i);
            }
        }
        assert_eq!(node.authenticity_of("p", "m", t0 + 100), AuthenticityTier::ReputationRedundant);
        // One failed audit: the floor, whatever the rest of its history says…
        node.record_audit("p", "m", AuditClass::Fail, false, t0 + 200);
        node.record_audit("p", "m", AuditClass::Pass, false, t0 + 300);
        node.record_audit("q", "m", AuditClass::Pass, false, t0 + 300);
        assert_eq!(node.authenticity_of("p", "m", t0 + 400), AuthenticityTier::SelfReported);
        assert_eq!(node.authenticity_of("q", "m", t0 + 400), AuthenticityTier::ReputationRedundant);
        assert_eq!(node.authenticity_of("p", "m", t0 + 200 + AUDIT_FAIL_FLOOR_MS), AuthenticityTier::SelfReported);
        // …for a day: then it is judged on its history like anyone (the failure still weighs in its
        // pass rate, so not necessarily back to where it was).
        let after = t0 + 201 + AUDIT_FAIL_FLOOR_MS;
        let without_floor = {
            let key = reputation_key("p", "m");
            let st = node.audit_stats.lock().unwrap()[&key].clone();
            RedundantExecAuthenticity
                .verify(&AuthenticityEvidence {
                    reputation: node.earned_reputation("p", "m", after).unwrap_or(NEUTRAL_REPUTATION),
                    audit_pass_rate: st.pass_rate(after),
                    passing_audits: st.recent_passes(after),
                    methods: MethodFlags::default(),
                    latest_outcome: None,
                })
                .authenticity
        };
        assert_eq!(node.authenticity_of("p", "m", after), without_floor);
    }

    fn test_node() -> (tempfile::TempDir, ConsumerNode) {
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
        (dir, ConsumerNode::new(net))
    }

    fn empty_logprob() -> ProviderLogprob {
        ProviderLogprob {
            refs: Vec::new(),
            registry: None,
            registry_identities: Mutex::new(HashMap::new()),
            registry_weights: Mutex::new(HashMap::new()),
            rounds: Mutex::new(HashMap::new()),
            status: Mutex::new(HashMap::new()),
            inflight: Mutex::new(std::collections::HashSet::new()),
            count_scorer: None,
            counts_running: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            issued_here: Mutex::new(std::collections::HashSet::new()),
            resume: Mutex::new(HashMap::new()),
            no_verdict_runs: Mutex::new(HashMap::new()),
            first_contact_packs: Mutex::new(HashMap::new()),
        }
    }

    /// Round 8 T8-3/T8-5: a shed or a missed provider goes last for a minute and only a second miss
    /// within ten minutes counts; a recent completed serve shows its engine works.
    #[test]
    fn a_missed_provider_goes_last_and_only_a_repeat_counts() {
        let (_dir, node) = test_node();
        let t = 1_000_000;
        assert!(!node.note_unavailable("p", "m", t), "first miss");
        assert!(node.briefly_unavailable("p", "m", t + UNAVAILABLE_RANK_LAST_MS));
        assert!(!node.briefly_unavailable("p", "m", t + UNAVAILABLE_RANK_LAST_MS + 1));
        assert!(!node.briefly_unavailable("p", "other", t));
        assert!(node.note_unavailable("p", "m", t + UNAVAILABLE_REPEAT_MS), "second within ten minutes");
        assert!(!node.note_unavailable("p", "m", t + 3 * UNAVAILABLE_REPEAT_MS), "long after: a first miss again");
        assert!(!node.served_ok_recently("p", "m", t));
        node.note_served_ok("p", "m", t);
        assert!(node.served_ok_recently("p", "m", t + SERVED_OK_RECENT_MS));
        assert!(!node.served_ok_recently("p", "other", t), "per model (round 9 A9-3)");
        assert!(!node.served_ok_recently("p", "m", t + SERVED_OK_RECENT_MS + 1));
        assert!(matches!(provider_error(SHED_REPLY.into(), true), AdapterError::Overloaded(_)));
        assert!(matches!(provider_error(SHED_REPLY.into(), false), AdapterError::Provider(_)));
        assert!(matches!(provider_error("engine down".into(), true), AdapterError::Provider(_)));
    }

    /// Round 8 T8-8: the pair maps are bounded; stale pairs go first, then the least recent one,
    /// never one kept (a recent failure).
    #[test]
    fn tracked_pairs_are_bounded() {
        let now = 100 * STALE_PAIR_MS;
        let mut m: HashMap<String, u64> = (0..MAX_TRACKED_PAIRS).map(|i| (format!("p{i}"), now - i as u64)).collect();
        m.insert("p5".into(), now - 2 * STALE_PAIR_MS); // stale
        let gone = evict_for_new_pair(&mut m, |v| *v, |_| false, now);
        assert_eq!(gone, vec!["p5".to_string()]);
        m.insert("new".into(), now);
        // None stale: the least recent goes, unless kept.
        let oldest = format!("p{}", MAX_TRACKED_PAIRS - 1);
        let gone = evict_for_new_pair(&mut m, |v| *v, |v| *v == now - (MAX_TRACKED_PAIRS as u64 - 1), now);
        assert_eq!(gone.len(), 1);
        assert_ne!(gone[0], oldest, "a kept pair isn't what makes room");
        assert!(m.len() < MAX_TRACKED_PAIRS);
    }

    /// Round 8 L8-5/L8-6: providers with standing aren't held to the shared relay allowance; a
    /// reference switch keeps the round's declined texts.
    #[test]
    fn relay_allowance_and_skips_hold() {
        let lp = empty_logprob();
        let t = 1_000_000;
        for _ in 0..FIRST_CONTACT_RELAYED_PER_HOUR {
            assert!(lp.spend_first_contact_pack("", false, t));
        }
        assert!(!lp.spend_first_contact_pack("", false, t), "the relay allowance is spent");
        assert!(lp.spend_first_contact_pack("", true, t), "one with standing still gets its check");
        lp.refund_first_contact_pack("", true);
        let mut round = ChallengeRound::default();
        round.carry_skipped(1);
        let mut st = LogprobStatus::default();
        assert!(st.switch_reference([7; 16], &mut round));
        assert_eq!(round.skipped(), 1, "switching what it declares doesn't restore the allowance");
    }

    /// Round 8 L8-1: runs cut short by timeouts or dropped connections never count against a
    /// provider that has answered before; declining texts mid-round still does.
    #[test]
    fn a_run_cut_short_is_never_held_against_a_provider() {
        let (_dir, node) = test_node();
        let lp = empty_logprob();
        lp.status.lock().unwrap().entry("k".into()).or_default().pass_streak = 2;
        for _ in 0..10 {
            node.settle_logprob_round(&lp, "k", "p", "m", "m", RoundOutcome::Incomplete("transport: timed out".into()), ChallengeRound::default(), 0, None);
        }
        let st = lp.status.lock().unwrap()["k"];
        assert_eq!(st.pass_streak, 2, "streak kept");
        assert!(node.audit_stats.lock().unwrap().is_empty(), "no failed audit recorded");
        // Round 8 L8-3: a lost or refused model-info request is "busy"; only an answer without the
        // file counts.
        for _ in 0..10 {
            node.settle_logprob_round(&lp, "k", "p", "m", "m", RoundOutcome::Busy, ChallengeRound::default(), 0, None);
        }
        assert_eq!(lp.status.lock().unwrap()["k"].pass_streak, 2);
        node.settle_logprob_round(&lp, "k", "p", "m", "m", RoundOutcome::Evasion("declined".into()), ChallengeRound::default(), 0, None);
        assert_eq!(lp.status.lock().unwrap()["k"].pass_streak, 0, "an evasion still counts");
    }

    fn selected(libp2p: &str) -> SelectedProvider {
        SelectedProvider {
            peer_id: format!("oh-{libp2p}"),
            state_id: format!("state-{libp2p}"),
            libp2p_peer_id: libp2p.into(),
            model_id: "m".into(),
            canonical_model_id: String::new(),
            public_key: String::new(),
            swarm_public_key: String::new(),
        }
    }

    fn user(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage { role: "user".into(), content: text.into(), ..Default::default() }]
    }

    #[test]
    fn an_audit_asks_exactly_what_a_user_would() {
        let (_dir, node) = test_node();
        let node = node.with_settlement_peg(10);
        let a = selected("A");
        let tools = vec![serde_json::json!({"type": "function", "function": {"name": "f", "parameters": {}}})];
        let as_user = node.serve_request_for(&a, &user("hi there"), Some(300), Some(0.0), &tools, Some(true));
        let audit = node.audit_request_for(
            &a,
            &AuditRequest { messages: user("hi there"), max_tokens: Some(300), tools: tools.clone(), think: Some(true) },
        );
        assert_ne!(as_user.nonce, audit.nonce, "each serve commits its own nonce");
        let (mut u, mut v) = (serde_json::to_value(&as_user).unwrap(), serde_json::to_value(&audit).unwrap());
        u.as_object_mut().unwrap().remove("nonce");
        v.as_object_mut().unwrap().remove("nonce");
        assert_eq!(u, v, "field for field the same");
        // No audit tell on the wire: no deterministic mode, the user's own cap and thinking flag.
        assert!(v.get("deterministic").is_none_or(|d| d.is_null()));
        assert_eq!(v["max_tokens"], 300);
        assert_eq!(v["think"], true);
        assert_eq!(v["settlement_peg"], 10);
    }

    #[test]
    fn only_greedy_public_serves_of_others_are_kept_for_replay() {
        let (_dir, node) = test_node();
        let a = selected("A");
        // Replay is opt-in.
        assert!(!node.replay_eligible(&a, Some(0.0), &user("x")));
        let node = node.with_audit_replay(true).with_self_provider(Some("SELF".into()));
        assert!(node.replay_eligible(&a, Some(0.0), &user("x")));
        assert!(!node.replay_eligible(&a, Some(0.7), &user("x")), "sampled answers can't be compared");
        assert!(!node.replay_eligible(&a, None, &user("x")), "the engine's default may sample");
        let mut private = a.clone();
        private.swarm_public_key = "swarm".into();
        assert!(!node.replay_eligible(&private, Some(0.0), &user("x")), "a private swarm's prompt never leaves it");
        assert!(!node.replay_eligible(&selected("SELF"), Some(0.0), &user("x")), "not our own provider");
        let huge = "x".repeat(REPLAY_MAX_PROMPT_BYTES + 1);
        assert!(!node.replay_eligible(&a, Some(0.0), &user(&huge)));
    }

    #[test]
    fn kept_serves_are_bounded_fresh_and_replayed_once() {
        let entry = |p: &str, at: u64| Replayable {
            provider: selected(p),
            model: "m".into(),
            request: AuditRequest { messages: user("q"), max_tokens: None, tools: vec![], think: None },
            answer: "a".repeat(100),
            at,
        };
        let mut kept = Vec::new();
        for i in 0..(REPLAY_KEEP as u64 + 10) {
            keep_replayable(&mut kept, entry(if i % 2 == 0 { "A" } else { "B" }, 1_000 + i), 1_000 + i);
        }
        assert_eq!(kept.len(), REPLAY_KEEP);
        assert!(kept.iter().all(|r| r.at >= 1_010), "the oldest went first");
        let now = 1_000 + REPLAY_KEEP as u64 + 10;
        let taken = take_replayable(&mut kept, "state-A", "m", now).unwrap();
        assert_eq!(taken.provider.libp2p_peer_id, "A");
        assert_eq!(kept.len(), REPLAY_KEEP - 1, "replayed once");
        assert!(take_replayable(&mut kept, "state-A", "other", now).is_none());
        assert!(take_replayable(&mut kept, "state-C", "m", now).is_none());
        // Stale ones are dropped, not replayed.
        assert!(take_replayable(&mut kept, "state-B", "m", now + REPLAY_TTL_MS + 100).is_none());
        assert!(kept.is_empty());
    }

    #[test]
    fn a_probe_copies_the_shape_of_greedy_traffic_and_is_skipped_without_it() {
        let mut t = TrafficShape::default();
        for _ in 0..20 {
            t.note(Some(0.8), Some(false), Some(512), None, false);
        }
        assert_eq!(t.probe("p".into()), None, "no greedy traffic: a greedy probe would stand out");
        for _ in 0..5 {
            t.note(Some(0.0), Some(false), Some(700), None, false);
        }
        let probe = t.probe("Explain DNS.".into()).unwrap();
        assert_eq!(probe.messages, user("Explain DNS."));
        assert_eq!((probe.max_tokens, probe.think), (Some(700), Some(false)));
        assert!(probe.tools.is_empty());
        // The window slides: old traffic fades.
        for _ in 0..200 {
            t.note(Some(1.0), None, None, None, false);
        }
        assert!(t.requests <= TRAFFIC_WINDOW);
        assert_eq!(t.probe("p".into()), None);
        let mut thinking = TrafficShape::default();
        for _ in 0..3 {
            thinking.note(Some(0.0), Some(true), None, None, false);
        }
        thinking.note(Some(0.0), None, None, None, false);
        assert_eq!(thinking.probe("p".into()).unwrap().think, Some(true));
    }

    /// Round 8 T8-2: probes ask for lengths greedy traffic asked for, and a sampled model's shape is
    /// still copied (greedy aside) when a follow-up must be sent.
    #[test]
    fn probes_ask_for_the_lengths_real_requests_do() {
        let mut t = TrafficShape::default();
        for m in [Some(300), Some(2048), None] {
            t.note(Some(0.0), Some(true), m, Some("sys"), false);
        }
        for _ in 0..20 {
            let probe = t.probe("p".into()).unwrap();
            assert!([Some(300), Some(2048), None].contains(&probe.max_tokens));
            assert_eq!(probe.think, Some(true));
            assert_eq!(probe.messages.len(), 2, "opens with a system prompt, as the traffic does");
        }
        for _ in 0..(GREEDY_LENGTHS_KEPT + 4) {
            t.note(Some(0.0), None, Some(64), None, false);
        }
        assert_eq!(t.greedy_max_tokens.len(), GREEDY_LENGTHS_KEPT);
        assert_eq!(t.probe("p".into()).unwrap().max_tokens, Some(64));
        // Mostly sampled traffic: no ordinary probe; a follow-up's copy forces greedy only.
        let mut sampled = TrafficShape::default();
        for _ in 0..10 {
            sampled.note(Some(0.7), Some(false), Some(4096), Some("sys"), false);
        }
        assert!(sampled.probe("p".into()).is_none());
        let mut forced = sampled.clone();
        forced.greedy = forced.requests;
        let probe = forced.probe("p".into()).unwrap();
        assert_eq!((probe.max_tokens, probe.think, probe.messages.len()), (Some(4096), Some(false), 2));
    }

    #[test]
    fn replayed_answers_are_compared_by_their_opening() {
        let opening = "The quick brown fox jumps over the lazy dog. ".repeat(8);
        assert!(opening.chars().count() >= AUDIT_COMPARE_CHARS);
        // The served answer was kept up to the window; a second honest run agrees on it even when
        // their tails part ways.
        let a = compared(&format!("{opening} and then A's own ending"));
        let b = compared(&format!("{opening} while B ends differently"));
        let c = compared("Something else entirely, from a different model, from the first word onward. ");
        let report = resolve_audit(&[("A".into(), Ok(a.clone())), ("B".into(), Ok(b.clone()))]);
        assert!(report.outcomes.iter().all(|(_, o)| *o == VerificationOutcome::Honored));
        let split = resolve_audit(&[("A".into(), Ok(c.clone())), ("B".into(), Ok(b.clone()))]);
        assert_eq!(split.verdict, RedundantVerdict::Inconclusive, "1 vs 1 goes to a third provider");
        let decided = resolve_audit(&[("A".into(), Ok(c)), ("B".into(), Ok(b)), ("C".into(), Ok(a))]);
        assert!(decided.outcomes.contains(&("A".into(), VerificationOutcome::Failed)), "the served answer was outvoted");
    }

    #[test]
    fn a_payment_refusal_from_a_provider_we_just_paid_is_not_about_price() {
        let (_dir, node) = test_node();
        let now = now_unix_ms();
        // Without a peg nothing is paid, so no refusal is ever held against anyone.
        node.note_settled("P", true);
        assert!(!node.paid_recently("P", now));
        let (_dir2, node) = test_node();
        let node = node.with_settlement_peg(10);
        assert!(!node.paid_recently("P", now));
        node.note_settled("P", true);
        assert!(node.paid_recently("P", now_unix_ms()));
        assert!(!node.paid_recently("Q", now_unix_ms()));
        assert!(!node.paid_recently("P", now_unix_ms() + PAYMENT_REFUSAL_TTL_MS + 1), "only recently");
        // A settlement that failed since: we may be the ones not paying.
        node.note_settled("P", false);
        assert!(!node.paid_recently("P", now_unix_ms()));
    }

    #[test]
    fn a_request_raises_the_floor_and_has_it_enforced_but_never_lowers_it() {
        use AuthenticityTier::*;
        let base = AssurancePolicy::OPEN;
        assert_eq!(request_policy(base, None), base);
        assert_eq!(request_policy(base, Some(SelfReported)), base, "asking for A0 asks for nothing");
        let raised = request_policy(base, Some(ReputationRedundant));
        assert_eq!((raised.min_authenticity, raised.mode), (ReputationRedundant, PolicyMode::HardGate));
        // A gateway that only reports its A1 floor: a request naming A1 has it enforced.
        let verify_a1 = AssurancePolicy { min_authenticity: ReputationRedundant, ..base };
        let r = request_policy(verify_a1, Some(ReputationRedundant));
        assert_eq!((r.min_authenticity, r.mode), (ReputationRedundant, PolicyMode::HardGate));
        // A request can't lower an enforced A2 floor, nor relax it to verify.
        let gate_a2 = AssurancePolicy { min_authenticity: EnergyFingerprintPossession, mode: PolicyMode::HardGate, ..base };
        assert_eq!(request_policy(gate_a2, Some(ReputationRedundant)), gate_a2);
        assert_eq!(request_policy(gate_a2, Some(SelfReported)), gate_a2);
        assert_eq!(request_policy(gate_a2, None), gate_a2);
    }

    #[test]
    fn a_floor_that_leaves_nobody_says_which_tier_it_wanted() {
        let peers = vec![discovered("a", TPL_A, 5.0), discovered("b", TPL_A, 9.0)];
        let gate = |t| AssurancePolicy { min_authenticity: t, mode: PolicyMode::HardGate, ..AssurancePolicy::OPEN };
        let e = no_route_error("m", &gate(AuthenticityTier::ReputationRedundant), &peers, 2).to_string();
        assert!(e.contains("verified at A1-reputation-redundant or above") && e.contains("2 found below it"), "{e}");
        // Said as what it is, not as an engine failure (review E1).
        assert!(e.starts_with("no provider of 'm'"), "{e}");
        // Nobody there at all, or no floor: the plain message.
        assert!(no_route_error("m", &gate(AuthenticityTier::ReputationRedundant), &[], 2).to_string().contains("no provider for model 'm'"));
        assert!(no_route_error("m", &AssurancePolicy::OPEN, &peers, 2).to_string().contains("no provider for model 'm'"));
    }

    #[test]
    fn a_payment_refusal_counts_only_where_a_provider_refuses() {
        let refusal = |s: &str| ServeChunk::Error(format!("{}{s}", crate::serve::PAYMENT_REFUSAL)).encode();
        let is_refusal = |r: Result<Option<ServeSummary>, StreamFail>| match r {
            Err(StreamFail::Error(e)) => crate::serve::is_payment_refusal(&e),
            _ => panic!("expected an error"),
        };
        let mut tc = Vec::new();
        // First frame of the opening reply: a refusal.
        assert!(is_refusal(dispatch_stream_slice(&crate::serve::frame_response(&[refusal("settle first")]), &mut tc, true, &mut |_| {})));
        // In a later poll (after empty polls), or after a tool-calls frame: an ordinary failure.
        assert!(!is_refusal(dispatch_stream_slice(&crate::serve::frame_response(&[refusal("x")]), &mut tc, false, &mut |_| {})));
        let tools = ServeChunk::ToolCalls(vec![]).encode();
        assert!(!is_refusal(dispatch_stream_slice(&crate::serve::frame_response(&[tools, refusal("x")]), &mut tc, true, &mut |_| {})));
        // A buffered reply: only as its only frame.
        let only = |frames: Vec<Vec<u8>>| {
            let reply = crate::serve::frame_response(&frames);
            let mut t = |_: &[u8]| -> Result<Vec<u8>, AdapterError> { Ok(reply.clone()) };
            request_completion(&mut t, &request(), &mut |_| {}).unwrap_err()
        };
        assert!(crate::serve::is_payment_refusal(&only(vec![refusal("x")])));
        let after = only(vec![ServeChunk::Delta("hi".into()).encode(), refusal("x")]);
        assert!(!crate::serve::is_payment_refusal(&after));
        assert!(after.to_string().contains("refused payment after the serve began"), "{after}");
    }

    #[test]
    fn a_stream_stops_reading_when_its_client_leaves_and_knows_how_far_it_got() {
        // An open ack, then one slice of two deltas per poll, forever.
        let mut polls = 0u32;
        let mut transport = |framed: &[u8]| -> Result<Vec<u8>, AdapterError> {
            if framed[0] == SERVE_STREAM {
                return Ok(FetchChunksResponse::Chunks { framed: Vec::new(), next_offset: 0, done: false }.encode());
            }
            polls += 1;
            let frames = [ServeChunk::Delta("ab".into()).encode(), ServeChunk::Delta("cd".into()).encode()];
            Ok(FetchChunksResponse::Chunks { framed: crate::serve::frame_response(&frames), next_offset: polls * 2, done: false }.encode())
        };
        let mut sleeper = |_: Duration| {};
        let seen = std::cell::Cell::new(0);
        let mut progress = StreamProgress::default();
        let out = request_completion_streaming_with(&mut transport, &mut sleeper, &request(), &|| seen.get() >= 6, &mut progress, &mut |_| {
            seen.set(seen.get() + 1)
        });
        assert!(matches!(out, Err(StreamFail::Cancelled)));
        assert_eq!((seen.get(), progress.offset), (6, 6), "stopped after the slice that told it to, at its offset");
        assert!(progress.opened, "the provider acknowledged the open");
    }

    /// Round 9 A9-4/A9-5/A9-6: only providers that served us with standing become writer
    /// candidates; the clear rule the audit applies refuses the striking writer's questions.
    #[test]
    fn writer_candidates_need_standing_and_the_striking_writer_cant_clear() {
        let (_dir, node) = test_node();
        let now = 1_000_000;
        let p = selected("P");
        node.note_writer_candidate(&p, now);
        assert!(node.writer_candidates.lock().unwrap().is_empty(), "no standing: not a candidate");
        for _ in 0..10 {
            node.record_outcome(&p.state_id, rep_model_key(&p.canonical_model_id, &p.model_id), VerificationOutcome::Honored, now);
        }
        node.note_writer_candidate(&p, now);
        assert!(node.writer_candidates.lock().unwrap().contains_key(&("P".to_string(), "m".to_string())));
        assert!(strike_cleared_by(true, false, None, false, Some("W")), "a replay");
        assert!(strike_cleared_by(false, true, Some("V"), false, Some("W")), "another writer's follow-up");
        assert!(!strike_cleared_by(false, true, Some("W"), false, Some("W")), "not the striking writer's");
        assert!(strike_cleared_by(false, true, Some("W"), true, Some("W")), "our own provider's always");
        assert!(!strike_cleared_by(false, true, None, false, None), "not a template");
        assert!(!strike_cleared_by(false, false, Some("V"), false, None), "not a sampled probe");
    }

    /// Review round 4 C3: a first divergence is a strike (audited again later, on another prompt),
    /// a second within the window a failure; agreeing in between clears the strike.
    #[test]
    fn an_outvoted_provider_is_failed_only_on_a_second_strike() {
        let good = "The quick brown fox jumps over the lazy dog, again and again, all afternoon.".to_string();
        let odd = "Something else entirely, word after word, from the very first one onward.".to_string();
        let results = vec![
            ("A".to_string(), Ok(odd.clone())),
            ("B".to_string(), Ok(good.clone())),
            ("C".to_string(), Ok(good.clone())),
        ];
        let report = resolve_audit(&results);
        assert!(report.outcomes.contains(&("A".into(), VerificationOutcome::Failed)));
        let key = |p: &str| format!("{p}@m");
        let clears = |p: &str, _: Option<&str>| p == "A";
        let mut strikes = HashMap::new();
        // First divergence: not failed, struck, and audited again.
        let (first, follow) = strike_first_failures(report.clone(), &key, &clears, None, &mut strikes, 1_000);
        assert!(!first.outcomes.iter().any(|(p, _)| p == "A"));
        assert_eq!(first.outcomes.iter().filter(|(_, o)| *o == VerificationOutcome::Honored).count(), 2);
        assert_eq!(follow, vec!["A".to_string()]);
        assert!(strikes.contains_key("A@m"));
        // Diverging again within the window: failed, strike spent.
        let (second, follow) = strike_first_failures(report.clone(), &key, &clears, None, &mut strikes, 2_000);
        assert!(second.outcomes.contains(&("A".into(), VerificationOutcome::Failed)));
        assert!(follow.is_empty() && !strikes.contains_key("A@m"));
        // A strike past the window is a first strike again.
        strike_first_failures(report.clone(), &key, &clears, None, &mut strikes, 3_000);
        let (late, _) = strike_first_failures(report.clone(), &key, &clears, None, &mut strikes, 3_001 + STRIKE_WINDOW_MS);
        assert!(!late.outcomes.iter().any(|(p, _)| p == "A"));
        // Agreeing in between clears it.
        let honest = resolve_audit(&[("A".to_string(), Ok(good.clone())), ("B".to_string(), Ok(good.clone()))]);
        strike_first_failures(honest.clone(), &key, &|_, _| false, None, &mut strikes, 4_000);
        assert!(strikes.contains_key("A@m"), "agreeing elsewhere (not a replay pass as the one audited) doesn't clear it");
        strike_first_failures(honest, &key, &clears, None, &mut strikes, 4_000);
        assert!(!strikes.contains_key("A@m"));
    }

    /// Round 8 T8-1/T8-4: an outside writer's questions alone can neither fail a provider nor clear
    /// the strike they gave; anything else can.
    #[test]
    fn one_outside_writer_neither_fails_nor_clears_a_provider_alone() {
        let good = "The quick brown fox jumps over the lazy dog, again and again, all afternoon.".to_string();
        let odd = "Something else entirely, word after word, from the very first one onward.".to_string();
        let outvoted = resolve_audit(&[
            ("A".to_string(), Ok(odd)),
            ("B".to_string(), Ok(good.clone())),
            ("C".to_string(), Ok(good.clone())),
        ]);
        let honest = resolve_audit(&[("A".to_string(), Ok(good.clone())), ("B".to_string(), Ok(good))]);
        let key = |p: &str| format!("{p}@m");
        let mut strikes = HashMap::new();
        strike_first_failures(outvoted.clone(), &key, &|_, _| false, Some("W"), &mut strikes, 1_000);
        assert_eq!(strikes["A@m"].writer.as_deref(), Some("W"));
        // W's questions again: not a failure, audited again on something else.
        let (again, follow) = strike_first_failures(outvoted.clone(), &key, &|_, _| false, Some("W"), &mut strikes, 2_000);
        assert!(!again.outcomes.iter().any(|(p, _)| p == "A"));
        assert_eq!(follow, vec!["A".to_string()]);
        assert!(strikes.contains_key("A@m"));
        // A clear that only W's questions would give doesn't (the closure the audit passes refuses it).
        let by_w = |p: &str, struck_by: Option<&str>| p == "A" && Some("W") != struck_by;
        strike_first_failures(honest.clone(), &key, &by_w, Some("W"), &mut strikes, 3_000);
        assert!(strikes.contains_key("A@m"));
        // Another writer's questions (or a template, a replay) confirm it.
        let (other, _) = strike_first_failures(outvoted, &key, &|_, _| false, Some("V"), &mut strikes, 4_000);
        assert!(other.outcomes.contains(&("A".into(), VerificationOutcome::Failed)));
        assert!(has_writer_standing(Some(openhydra_protocol::award::R_MIN)));
        assert!(!has_writer_standing(Some(openhydra_protocol::verify::NEUTRAL_REPUTATION)));
        assert!(!has_writer_standing(None));
    }

    /// Review round 4 H5: a relay-only subject leaves room for a partner of unknown network, and two
    /// of unknown network need a third on a known one.
    #[test]
    fn partners_are_independent_and_two_unknowns_need_a_known_third() {
        let subject_unknown: Vec<Vec<String>> = vec![vec![]];
        assert!(independent_partner(&subject_unknown, 0, &[]), "the subject's unknown network doesn't use the place");
        assert!(!independent_partner(&subject_unknown, 1, &[]), "one partner of unknown network at most");
        let known = vec![vec!["1.2.3.0/24".to_string()]];
        assert!(!independent_partner(&known, 0, &["1.2.3.0/24".to_string()]), "same network");
        assert!(independent_partner(&known, 0, &["5.6.7.0/24".to_string()]));
    }

    /// Review round 5 R5T-3/R5T-4: providers of unknown network are judged only against at least two
    /// on known networks that agree; two unknown colluders can't outvote, or stand in for, them.
    #[test]
    fn providers_of_unknown_network_are_judged_by_the_known_ones() {
        let good = "The quick brown fox jumps over the lazy dog, again and again, all afternoon.".to_string();
        let odd = "Something else entirely, word after word, from the very first one onward.".to_string();
        let nets: HashMap<String, Vec<String>> = [
            ("S".to_string(), vec![]),
            ("U".to_string(), vec![]),
            ("K1".to_string(), vec!["5.6.7.0/24".to_string()]),
            ("K2".to_string(), vec!["8.8.9.0/24".to_string()]),
        ]
        .into();
        // Two colluding unknowns against two agreeing known providers: the unknowns fail.
        let r = resolve_against_known(
            &[("S".into(), Ok(odd.clone())), ("U".into(), Ok(odd.clone())), ("K1".into(), Ok(good.clone())), ("K2".into(), Ok(good.clone()))],
            &nets,
            "S",
        )
        .unwrap();
        assert!(r.outcomes.contains(&("S".into(), VerificationOutcome::Failed)));
        assert!(r.outcomes.contains(&("U".into(), VerificationOutcome::Failed)));
        assert!(r.outcomes.contains(&("K1".into(), VerificationOutcome::Honored)));
        // One known answer (R6-6/R6-4): all agreeing with it pass; any disagreement decides nothing.
        let one = resolve_against_known(&[("S".into(), Ok(good.clone())), ("U".into(), Ok(good.clone())), ("K1".into(), Ok(good.clone()))], &nets, "S").unwrap();
        assert_eq!(one.verdict, RedundantVerdict::Agree);
        assert!(resolve_against_known(&[("S".into(), Ok(odd.clone())), ("U".into(), Ok(odd.clone())), ("K1".into(), Ok(good.clone()))], &nets, "S").is_none());
        // The one known answer being the subject's own doesn't count (R7-2): its relay-only partner
        // could be its own second node.
        let k_subject: HashMap<String, Vec<String>> =
            [("S".to_string(), vec!["5.6.7.0/24".to_string()]), ("U".to_string(), vec![])].into();
        assert!(resolve_against_known(&[("S".into(), Ok(good.clone())), ("U".into(), Ok(good.clone()))], &k_subject, "S").is_none());
        // Known ones split: no verdict.
        // A known provider that errors doesn't count as an answer: one known answer, all agree.
        let erred = resolve_against_known(
            &[("S".into(), Ok(good.clone())), ("K1".into(), Ok(good.clone())), ("K2".into(), Err("timeout".into()))],
            &nets,
            "S",
        )
        .unwrap();
        assert!(erred.outcomes.contains(&("K2".into(), VerificationOutcome::Rejected)));
        assert!(erred.outcomes.contains(&("S".into(), VerificationOutcome::Honored)));
        assert!(resolve_against_known(&[("S".into(), Ok(good.clone())), ("K1".into(), Ok(good)), ("K2".into(), Ok(odd))], &nets, "S").is_none());
    }

    /// Review round 4 M8: probes open with a system prompt as often as real requests do; the real
    /// one is kept only where requests may be replayed anyway.
    #[test]
    fn a_probe_opens_with_a_system_prompt_like_the_traffic() {
        let mut t = TrafficShape::default();
        for _ in 0..4 {
            t.note(Some(0.0), None, None, Some("You are Acme's support bot."), false);
        }
        let probe = t.probe("Explain DNS.".into()).unwrap();
        assert_eq!(probe.messages.len(), 2);
        assert_eq!(probe.messages[0].role, "system");
        assert_ne!(probe.messages[0].content, "You are Acme's support bot.", "not kept where replay is off");
        t.note(Some(0.0), None, None, Some("You are Acme's support bot."), true);
        assert_eq!(t.probe("p".into()).unwrap().messages[0].content, "You are Acme's support bot.");
        let mut plain = TrafficShape::default();
        plain.note(Some(0.0), None, None, None, true);
        assert_eq!(plain.probe("p".into()).unwrap().messages.len(), 1);
    }

    #[test]
    fn rep_model_key_uses_the_validated_handle_not_the_canonical() {
        // Keyed on the handle (matches KnownProvider.model_id + the commitment-bound identity),
        // regardless of the untrusted self-declared canonical.
        assert_eq!(rep_model_key("qwen3.5/2b/fp16/aa", "qwen2.5:7b"), "qwen2.5:7b");
        assert_eq!(rep_model_key("", "qwen2.5:7b"), "qwen2.5:7b");
    }

    #[test]
    fn no_earned_history_falls_back_to_the_plain_ranking() {
        // Earned reputation only ever *overrides* — with no history (lookup → None for
        // all), the result must equal the plain reputation-agnostic ranking.
        let peers = vec![discovered("a", TPL_A, 5.0), discovered("b", TPL_A, 45.0)];
        let with_none = rank_providers_with_reputation(&peers, TPL_A, 2, &|_, _| None);
        let plain = rank_providers(&peers, TPL_A, 2);
        assert_eq!(
            with_none.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>(),
            plain.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>(),
        );
    }

    // ── Increment 3: buyer assurance policy gates routing ──

    fn require_a1(mode: PolicyMode) -> AssurancePolicy {
        AssurancePolicy {
            min_authenticity: AuthenticityTier::ReputationRedundant,
            min_privacy: PrivacyTier::None,
            min_settlement: SettlementTier::CreditsOnly,
            mode,
        }
    }

    #[test]
    fn hard_gate_drops_providers_below_the_required_authenticity_tier() {
        // "proven" resolves to A1, "fresh" to A0; a policy requiring A1 under HardGate keeps only
        // the proven provider — even though "fresh" has the higher capability score.
        let peers = vec![discovered("fresh", TPL_A, 45.0), discovered("proven", TPL_A, 5.0)];
        let tier = |pid: &str, _m: &str| {
            if pid == "proven" {
                AuthenticityTier::ReputationRedundant
            } else {
                AuthenticityTier::SelfReported
            }
        };
        let ranked = rank_providers_with_assurance(
            &peers,
            TPL_A,
            2,
            &|_, _| None,
            &tier,
            &require_a1(PolicyMode::HardGate),
        );
        let ids: Vec<&str> = ranked.iter().map(|p| p.peer_id.as_str()).collect();
        assert_eq!(ids, vec!["proven"], "HardGate drops the sub-tier provider");
    }

    #[test]
    fn verify_but_route_keeps_every_provider_regardless_of_tier() {
        // The same split, but VerifyButRoute keeps both (the achieved tier is surfaced elsewhere,
        // not used to route), so the result equals the ungated ranking.
        let peers = vec![discovered("fresh", TPL_A, 45.0), discovered("proven", TPL_A, 5.0)];
        let tier = |pid: &str, _m: &str| {
            if pid == "proven" {
                AuthenticityTier::ReputationRedundant
            } else {
                AuthenticityTier::SelfReported
            }
        };
        let gated = rank_providers_with_assurance(
            &peers,
            TPL_A,
            2,
            &|_, _| None,
            &tier,
            &require_a1(PolicyMode::VerifyButRoute),
        );
        let plain = rank_providers(&peers, TPL_A, 2);
        assert_eq!(
            gated.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>(),
            plain.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>(),
            "VerifyButRoute never drops a candidate",
        );
    }

    #[test]
    fn hard_gate_with_no_qualifying_provider_fails_closed() {
        // Every provider is A0; requiring A1 under HardGate admits nobody (fail-closed) — the honest
        // outcome when a buyer demands a tier no reachable provider has proven.
        let peers = vec![discovered("a", TPL_A, 45.0), discovered("b", TPL_A, 5.0)];
        let ranked = rank_providers_with_assurance(
            &peers,
            TPL_A,
            2,
            &|_, _| None,
            &|_, _| AuthenticityTier::SelfReported,
            &require_a1(PolicyMode::HardGate),
        );
        assert!(ranked.is_empty(), "no provider clears the floor → empty routing");
    }

    #[test]
    fn open_policy_matches_plain_reputation_ranking() {
        // The OPEN policy admits everything, so assurance-gated ranking equals the plain path even
        // when the resolver would report A0 for all — the backward-compatible default.
        let peers = vec![discovered("a", TPL_A, 5.0), discovered("b", TPL_A, 45.0)];
        let gated = rank_providers_with_assurance(
            &peers,
            TPL_A,
            2,
            &|_, _| None,
            &|_, _| AuthenticityTier::SelfReported,
            &AssurancePolicy::OPEN,
        );
        let plain = rank_providers(&peers, TPL_A, 2);
        assert_eq!(
            gated.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>(),
            plain.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>(),
        );
    }
}
