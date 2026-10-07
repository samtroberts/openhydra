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
    audit_body, reputation_key, resolve_audit, sample_rate_for_reputation, split_reputation_key,
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
    parse_response, DeterministicParams, FetchChunksResponse, FetchResponse, ServeChunk,
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
/// Payment refusals remembered at once (the oldest are forgotten first).
const MAX_PAYMENT_REFUSALS: usize = 1024;
/// Settlements running or queued for one `(peer, model)`. They run one at a time (its lock), so with
/// a provider that black-holes receipts each waits out [`SETTLE_TIMEOUT`] in turn; more than a few
/// queued behind it would only wait to fail.
const MAX_SETTLES_PER_PAIR: usize = 32;
/// Retries of a settlement whose round-trip failed (or the provider asked us to retry), and the wait
/// before each: a serve that goes unpaid counts against us at a provider that charges.
const SETTLE_RETRY_WAITS: [Duration; 2] = [Duration::from_secs(2), Duration::from_secs(8)];

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
fn over_reported_by_billing(summary: &crate::serve::ServeSummary, billable: u64) -> bool {
    // Only a rule that bills a completed serve as ours does: a newer rule may bill differently, and
    // we can't judge it by ours.
    crate::serve::same_completed_billing(summary.metrics.billing) && summary.tokens > billable
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

/// Whether a provider on `nets` (its observed networks; empty = unknown, reached only through a
/// relay) may join an audit beside those already in it (`chosen`): no network in common with any of
/// them, and at most one of unknown network, since two of those could be one operator.
fn independent_of(chosen: &[Vec<String>], nets: &[String]) -> bool {
    if nets.is_empty() {
        return chosen.iter().all(|c| !c.is_empty());
    }
    chosen.iter().all(|c| !c.iter().any(|n| nets.contains(n)))
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
        && p.peg_micros_per_credit == cx.peg
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
    resume: Mutex<HashMap<String, Arc<LoadedReference>>>,
    /// Per `(peer, model)`: runs in a row that ended without a verdict, which spaces the next ones
    /// out (see [`logprob_retry_ms`]).
    no_verdict_runs: Mutex<HashMap<String, u32>>,
    /// When registry packs were fetched for providers with no verdict yet, within the last hour
    /// (see [`FIRST_CONTACT_PACKS_PER_HOUR`]).
    first_contact_packs: Mutex<std::collections::VecDeque<u64>>,
}

impl ProviderLogprob {
    /// Take one of this hour's [`FIRST_CONTACT_PACKS_PER_HOUR`] packs, if any is left.
    fn spend_first_contact_pack(&self, now_ms: u64) -> bool {
        let mut q = self.first_contact_packs.lock().unwrap_or_else(|e| e.into_inner());
        while q.front().is_some_and(|at| now_ms.saturating_sub(*at) > 60 * 60 * 1000) {
            q.pop_front();
        }
        if q.len() >= FIRST_CONTACT_PACKS_PER_HOUR {
            return false;
        }
        q.push_back(now_ms);
        true
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
const FIRST_CONTACT_PACKS_PER_HOUR: usize = 10;

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

/// Token budget for an audit challenge completion — kept short to bound the audit's compute
/// cost while leaving enough output to discriminate an honest run from a freeloader.
///
/// KNOWN LIMITATION (thinking/reasoning models). The audit probe sets `think:false` (see
/// [`ConsumerNode::dispatch_full`]) so a reasoning model answers directly within this small
/// budget and the required nonce echo lands in `audit_body`'s near-the-start window. That holds
/// once the fleet uniformly honours the switch. During a mixed-version rollout it can misfire
/// for a provider that does NOT honour `think:false` (a pre-Fix binary, a `/api/show` probe that
/// failed, or an OpenAI-path template that ignores `enable_thinking`): it still emits a
/// `<think>…</think>` block, which either (a) spends this whole 64-token budget mid-thought so the
/// nonce is never reached → recorded as a non-responder, or (b) yields a post-thinking answer that
/// legitimately diverges (at `temperature 0`, a continuation conditioned on chain-of-thought
/// differs from one without) from a peer that answered directly → a `common_prefix_ratio` miss.
/// Deliberately NOT "fixed" by inflating this budget universally: that would 8× the compute of the
/// (majority) non-thinking audits and *widen* the benign-divergence window, weakening the signal
/// for everyone. The exposure is bounded and self-correcting — audits are sampled, a 1-vs-1 tie is
/// `Inconclusive` (no penalty; it escalates to a third), and reputation is long-run — and it
/// disappears once the fleet converges on honouring `think:false`. If it ever needs an active fix,
/// the right shape is capability-scoped (down-weight / skip greedy-agreement auditing for
/// thinking-capable models, whose variable reasoning makes output-agreement the wrong primitive),
/// not a bigger global budget.
const AUDIT_MAX_TOKENS: u32 = 64;

/// The fixed RNG seed every audit probe pins (UNIFIED_TRUST_STACK 2b). A single shared value makes
/// a temp=0 redundant-exec probe reproducible *across* honest providers running the same model +
/// quant — tightening output agreement (an honest pair should now match to the token, not merely
/// "agree closely"), which sharpens the A1 authenticity signal. Providers that predate the
/// `deterministic` field ignore the seed and still serve temp=0 via the request's `temperature`.
const AUDIT_SEED: u64 = 0;

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
    for chunk in parse_response(&response)? {
        match chunk {
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
                })
            }
            ServeChunk::Error(msg) => return Err(AdapterError::Http(msg)),
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

/// One request's trip to a provider ([`ConsumerNode::exchange`]).
struct Exchange {
    result: Result<ServeSummary, AdapterError>,
    /// The first-ack (stream) or round-trip (buffered) time, ns.
    proxy_roundtrip_ns: u64,
    /// Whether it was read as a stream (a stream that breaks after output is settled for what was read).
    streamed: bool,
}

/// Why a streaming attempt didn't complete.
enum StreamFail {
    /// The provider answered `SERVE_STREAM` with a non-streaming (legacy) reply — the caller
    /// retries the request over the buffered `request_completion` path and remembers this peer.
    Legacy,
    /// A real failure: a provider `Error` frame, a shed (`Overloaded`), a stall, or a dropped
    /// transport past the retry budget. Carries the error for the caller's failover logic.
    Error(AdapterError),
}

/// Dispatch one incremental slice of serve frames: text deltas → `on_delta`, tool calls held for
/// the summary, `Done` → return the summary, `Error` → fail. `None` means "slice consumed, keep
/// polling". Shared by the open-ack and the poll loop.
fn dispatch_stream_slice(
    framed: &[u8],
    tool_calls: &mut Vec<ToolCall>,
    on_delta: &mut dyn FnMut(&str),
) -> Result<Option<ServeSummary>, StreamFail> {
    let chunks = parse_response(framed)
        .map_err(|e| StreamFail::Error(AdapterError::Parse(e.to_string())))?;
    for chunk in chunks {
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
                }))
            }
            ServeChunk::Error(msg) => return Err(StreamFail::Error(AdapterError::Http(msg))),
        }
    }
    Ok(None)
}

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

    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut offset: u32 = 0;
    match ack {
        FetchChunksResponse::Overloaded => {
            return Err(StreamFail::Error(AdapterError::Http(
                "provider overloaded (stream shed at open)".into(),
            )))
        }
        FetchChunksResponse::Forbidden => {
            return Err(StreamFail::Error(AdapterError::Http("stream open forbidden".into())))
        }
        FetchChunksResponse::NotFound => {
            return Err(StreamFail::Error(AdapterError::Http("stream missing at open".into())))
        }
        // The open ack may already carry frames — e.g. a refusal delivered as a complete
        // (done=true) stream, or an eager first slice.
        FetchChunksResponse::Chunks { framed, next_offset, done } => {
            if let Some(summary) = dispatch_stream_slice(&framed, &mut tool_calls, on_delta)? {
                return Ok(summary);
            }
            offset = next_offset.max(offset);
            let _ = done; // done=true with no terminal frame can't happen; a real Done returns above
        }
    }

    // ── Poll loop ──
    let mut interval = POLL_MIN;
    let mut idle = Duration::ZERO;
    loop {
        // F2: absolute backstop — bail if the whole attempt has outrun its (generous) total cap.
        if started.elapsed() > total_cap {
            return Err(StreamFail::Error(AdapterError::Http(
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
                            return Err(StreamFail::Error(AdapterError::Http(format!(
                                "stream poll dropped past retry budget: {e}"
                            ))));
                        }
                        sleep(STREAM_DROP_BACKOFF);
                        idle += STREAM_DROP_BACKOFF;
                        if idle >= STREAM_STALL_TIMEOUT {
                            return Err(StreamFail::Error(AdapterError::Http(
                                "stream stalled (no progress before deadline)".into(),
                            )));
                        }
                    }
                }
            }
        };

        match FetchChunksResponse::decode(&resp) {
            Ok(FetchChunksResponse::Chunks { framed, next_offset, done }) => {
                let progressed = next_offset > offset;
                if let Some(summary) = dispatch_stream_slice(&framed, &mut tool_calls, on_delta)? {
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
                        return Err(StreamFail::Error(AdapterError::Http(
                            "stream stalled (no progress before deadline)".into(),
                        )));
                    }
                }
                sleep(interval);
            }
            // Buffer gone (provider restarted / TTL expired) — the caller re-serves from scratch.
            Ok(FetchChunksResponse::NotFound) => {
                return Err(StreamFail::Error(AdapterError::Http(
                    "stream result not found (provider restarted)".into(),
                )))
            }
            Ok(FetchChunksResponse::Forbidden) => {
                return Err(StreamFail::Error(AdapterError::Http("stream poll forbidden".into())))
            }
            Ok(FetchChunksResponse::Overloaded) => {
                return Err(StreamFail::Error(AdapterError::Http("stream overloaded mid-poll".into())))
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
    /// Device attestation of providers (Phase 0). `None` = off (the default).
    provider_attest: Option<Arc<ProviderAttestation>>,
    /// Prompt-scoring (log-prob) checks of providers (Track A2-Logprob). `None` = off (the default).
    provider_logprob: Option<Arc<ProviderLogprob>>,
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
            provider_attest: None,
            provider_logprob: None,
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
        let (mut rounds, mut status) = (HashMap::new(), HashMap::new());
        if let Some(store) = &self.store {
            match store.load_logprob_states() {
                Ok(rows) => {
                    for (key, blob) in rows {
                        match decode_logprob_state(&blob) {
                            Some((st, round)) => {
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
            registry_identities: Mutex::new(HashMap::new()),
            registry_weights: Mutex::new(HashMap::new()),
            rounds: Mutex::new(rounds),
            status: Mutex::new(status),
            inflight: Mutex::new(std::collections::HashSet::new()),
            count_scorer,
            counts_running: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            issued_here: Mutex::new(issued_here),
            resume: Mutex::new(HashMap::new()),
            no_verdict_runs: Mutex::new(HashMap::new()),
            first_contact_packs: Mutex::new(std::collections::VecDeque::new()),
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
            // Too long to count (text and tool calls together): not checked, as for a long stream.
            let Some((countable, calls)) = crate::token_count::countable_text(&text, &tool_calls, &offered) else {
                return;
            };
            match crate::token_count::count_tokens(&scorer, &vocab, &countable) {
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
            if let Err(e) = store.put_logprob_state(key, &encode_logprob_state(status, round)) {
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
            provider_attest: None,
            provider_logprob: None,
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
            .map_or((false, None), |st| st.evidence(now_ms));
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
    /// Reputation reports the **standing (raw) score** — the value as of the last verification
    /// outcome — NOT the time-decayed `score_at(now)`. The decayed value drifts toward neutral on
    /// every poll, which made the desktop's reputation flicker and appear to "reset" on a bare
    /// re-announce (which produces no outcome, so decay is all that changed). The router still
    /// reads the decayed `score_at` directly from the tracker for ranking; only the human-facing
    /// display uses the stable standing score, which persists across re-announce and restart.
    pub fn economy_snapshot(&self) -> (Vec<crate::status::RepEntry>, Vec<crate::status::CreditEntry>) {
        let now = now_unix_ms();
        // Snapshot the redundant-exec audit history first, under its own lock, released before the
        // reputation lock is taken — the two maps share the composite `(peer, model)` key, but they
        // are never held nested, so taking them in sequence avoids any lock-ordering hazard.
        let audit: std::collections::HashMap<String, (f64, u32)> = self
            .audit_stats
            .lock()
            .map(|m| {
                m.iter().map(|(k, s)| (k.clone(), (s.pass_rate(now), s.recent_passes(now)))).collect()
            })
            .unwrap_or_default();
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
        let reputation = self
            .reputation
            .lock()
            .map(|m| {
                m.iter()
                    .map(|(key, t)| {
                        // Phase 4a: the map key is the composite `(peer, model)` — split it back
                        // for display. A key without the separator (should not occur post-migration)
                        // degrades gracefully to peer-only with no model.
                        let (peer, model) = match split_reputation_key(key) {
                            Some((p, m)) => (p.to_string(), Some(m.to_string())),
                            None => (key.clone(), None),
                        };
                        // The achieved authenticity tier from the redundant-exec evidence already
                        // tracked for this `(peer, model)` (UNIFIED_TRUST_STACK Increment 2a), plus a
                        // recent log-prob verdict when checks are on (Track A2-Logprob): a pass is
                        // the hard method that allows A2, a fail pins the floor. Reputation uses the
                        // stable standing score (same as `score`), so the tier does not flicker on
                        // decay between polls.
                        let (rate, passes) = audit.get(key).copied().unwrap_or((1.0, 0));
                        let lp = logprob.get(key);
                        let (lp_flag, latest_outcome) = lp.map_or((false, None), |st| st.evidence(now));
                        let tier = RedundantExecAuthenticity
                            .verify(&AuthenticityEvidence {
                                reputation: t.raw_score(),
                                audit_pass_rate: rate,
                                passing_audits: passes,
                                methods: MethodFlags { logprob_commitment: lp_flag, ..MethodFlags::default() },
                                latest_outcome,
                            })
                            .authenticity;
                        let logprob_check = lp.and_then(|st| st.label());
                        let (device_trust, rotations) = device.get(&peer).copied().unwrap_or((None, 0));
                        crate::status::RepEntry {
                            // Shown as the 64-bit peer id (the 128-bit state id's prefix) — the id
                            // the rest of the app knows the provider by.
                            openhydra_peer_id: peer.get(..16).unwrap_or(&peer).to_string(),
                            model_id: model,
                            score: (t.raw_score() * 10.0).round() / 10.0,
                            authenticity: Some(tier.label().to_string()),
                            gated_net: gated.get(key).copied().filter(|&n| n > 0),
                            device_trust: device_trust.map(str::to_string),
                            device_key_rotations: (rotations > 0).then_some(rotations),
                            logprob_check,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
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
        let snapshot = {
            let mut map = match self.reputation.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            let tracker = map
                .entry(key.clone())
                .or_insert_with(|| ReputationTracker::new(now_ms));
            tracker.record(outcome, now_ms);
            tracker.to_bytes()
        };
        // Persist outside the lock (don't hold the Mutex across the redb write).
        if let Some(store) = &self.store {
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
    fn record_audit(&self, peer_id: &str, model_key: &str, class: AuditClass, now_ms: u64) {
        let key = reputation_key(peer_id, model_key);
        let snapshot = {
            let mut m = match self.audit_stats.lock() {
                Ok(m) => m,
                Err(_) => return,
            };
            let s = m.entry(key.clone()).or_insert_with(|| AuditStats::new(now_ms));
            s.record(class, now_ms);
            s.to_bytes()
        };
        if let Some(store) = &self.store {
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
        let identity_of = |tag: [u8; 16]| {
            identity_in(&lp.refs, tag)
                .or_else(|| lp.registry_identities.lock().unwrap_or_else(|e| e.into_inner()).get(&tag).cloned())
        };
        priced_tier(model_key, status.as_ref(), &identity_of, now_ms)
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
    fn voucher_scope(&self) -> String {
        match &self.settlement_chain {
            Some(c) => match self.settlement_wallet() {
                Some(w) => format!("@{}@{w}", c.domain.fingerprint()),
                None => format!("@{}", c.domain.fingerprint()),
            },
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
    pub fn complete(
        self: &Arc<Self>,
        model: &str,
        messages: Vec<ChatMessage>,
        max_tokens: Option<u32>,
        temperature: Option<f64>,
        tools: Vec<Value>,
        think: Option<bool>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeSummary, AdapterError> {
        let now = now_unix_ms();
        // The `complete` span (from #[instrument]); we fill in `provider`/`tokens` on success.
        let req_span = tracing::Span::current();
        // First pass may discover from the C11 cache. On a "no usable provider" outcome — empty
        // discovery, or every candidate failed with nothing streamed — the cached set may be
        // stale (a provider moved/died and its replacement isn't cached yet), so invalidate this
        // model's discovery cache and retry once with a cold get_providers lookup. The retry is
        // only reached when nothing was streamed, so it can never duplicate delivered output.
        match self.serve_once(model, &messages, max_tokens, temperature, &tools, think, now, &req_span, on_delta) {
            AttemptOutcome::Served(summary) => return Ok(summary),
            AttemptOutcome::Fatal(e) => return Err(e),
            AttemptOutcome::Retryable(_) => {}
        }
        tracing::info!(model, "C11 bypass: all candidates failed — invalidating discover cache and retrying with a fresh lookup");
        let _ = self.net.invalidate_discover(model);
        match self.serve_once(model, &messages, max_tokens, temperature, &tools, think, now, &req_span, on_delta) {
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
                Err(e) => return AttemptOutcome::Fatal(AdapterError::Http(format!("discover: {e}"))),
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
        let mut candidates = rank_providers_with_assurance(
            &peers,
            "",
            self.tier,
            &|pid, mkey| self.earned_reputation(pid, mkey, now),
            &|pid, mkey| self.authenticity_of(pid, mkey, now),
            &self.assurance,
        );
        // Providers that just refused us over payment go last (in their order).
        candidates.sort_by_key(|c| self.refused_payment(&c.libp2p_peer_id, &c.model_id, now));
        let discover_ns = t_discover.elapsed().as_nanos() as u64;
        tracing::debug!(elapsed = ?t_discover.elapsed(), candidates = candidates.len(), "discover");
        if candidates.is_empty() {
            return AttemptOutcome::Retryable(AdapterError::Http(format!(
                "no provider for model '{model}'"
            )));
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
            let nonce = request.nonce;
            let provider_libp2p = provider.libp2p_peer_id.clone();
            let t_serve = std::time::Instant::now();
            // P7: on a sample of serves, keep the delivered text to count its tokens afterwards.
            let model_key_now = rep_model_key(&provider.canonical_model_id, &provider.model_id);
            let counter = (!self.is_self_serve(&provider_libp2p)
                && rand::random::<f64>() < crate::token_count::COUNT_SAMPLE_RATE)
                .then(|| self.token_counter_for(&provider, model_key_now))
                .flatten();
            let captured = std::cell::RefCell::new(String::new());
            // Set once the text outgrew `MAX_COUNTED_BYTES`: no longer kept, and not checked.
            let overflowed = std::cell::Cell::new(false);
            // What this attempt received, counted as the provider counts what its polls returned: a
            // stream that ends early is settled for it (see `StreamTally`).
            let tally = std::cell::Cell::new(crate::serve::StreamTally::default());
            let exchange = {
                let mut guarded = |d: &str| {
                    delivered = true;
                    let mut t = delivered_text.get();
                    t.push(d);
                    delivered_text.set(t);
                    let mut n = tally.get();
                    n.push_delta(d);
                    tally.set(n);
                    if counter.is_some() && !overflowed.get() {
                        let mut c = captured.borrow_mut();
                        if c.len() + d.len() <= crate::token_count::MAX_COUNTED_BYTES {
                            c.push_str(d);
                        } else {
                            // Too long to count: stop keeping it, and don't check this serve.
                            overflowed.set(true);
                            *c = String::new();
                        }
                    }
                    on_delta(d);
                };
                self.exchange(&provider_libp2p, &request, &mut guarded)
            };
            let streamed = exchange.streamed;
            let result = exchange.result;
            let proxy_roundtrip_ns = exchange.proxy_roundtrip_ns;
            match result {
                Ok(mut summary) => {
                    summary.discover_ns = discover_ns;
                    summary.proxy_roundtrip_ns = proxy_roundtrip_ns;
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
                        if tokens < summary.tokens {
                            over_reported = over_reported_by_billing(&summary, tokens);
                            tracing::warn!(provider = %provider.peer_id, reported = summary.tokens, signable = tokens,
                                billing = summary.metrics.billing, counted_against = over_reported,
                                "provider reported more tokens than it could have served; signing for fewer");
                            summary.tokens = tokens;
                        }
                        self.settle_receipt(&provider, tokens, nonce);
                        if let Some(counter) = counter.filter(|_| !overflowed.get()) {
                            self.check_billed_tokens(
                                counter,
                                captured.take(),
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
                    if !self_serve {
                        self.maybe_audit(model, model_key, &provider.state_id);
                        // Phase 0 device attestation (opt-in, detached, rate-limited per provider).
                        self.maybe_attest(&provider);
                        // Track A2-Logprob prompt-scoring round (opt-in, detached, every few hours).
                        self.maybe_logprob_check(&provider, model_key);
                    }
                    return AttemptOutcome::Served(summary);
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %provider.libp2p_peer_id, attempt = i + 1, total,
                        error = %e, "provider attempt failed"
                    );
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
                    if !self.is_self_serve(&provider.libp2p_peer_id) && !refused {
                        let model_key = rep_model_key(&provider.canonical_model_id, &provider.model_id);
                        self.record_outcome(&provider.state_id, model_key, VerificationOutcome::Rejected, now);
                    }
                    // A stream that broke after output arrived is still owed for what we read: the
                    // provider counts what its polls returned the same way, and holds an unpaid
                    // part against us like an unpaid whole.
                    if delivered && streamed && !self.is_self_serve(&provider.libp2p_peer_id) {
                        let tokens = tally.get().billable(max_tokens);
                        if tokens > 0 {
                            tracing::info!(provider = %provider.peer_id, tokens, "stream ended early; settling for what was received");
                            self.settle_receipt(&provider, tokens, nonce);
                            self.record_usage(model, &provider.peer_id, tokens, false, now);
                        }
                    }
                    if delivered {
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
            last_err.unwrap_or_else(|| AdapterError::Http(format!("all providers failed for '{model}'"))),
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
    fn exchange(&self, provider_libp2p: &str, request: &ServeRequest, on_delta: &mut dyn FnMut(&str)) -> Exchange {
        // Budget the serve by the request's `max_tokens` (see `attempt_timeout`) so a big
        // agentic completion on a slow remote model isn't abandoned mid-generation.
        let serve_to = attempt_timeout(request.max_tokens);
        // Shared across the (sequential) buffered + streaming transports via a Cell so both
        // can record it without a mutable-borrow conflict. Holds the first-ack / round-trip RTT.
        let proxy_rt = std::cell::Cell::new(0u64);
        let mut streamed = false;
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
                    .reconnect_and_fetch(provider_libp2p, &request.nonce, framed, serve_to)
                    .map_err(|fe| AdapterError::Http(format!("proxy_forward: {drop_err}; reconnect-fetch: {fe}"))),
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
                    r.map_err(AdapterError::Http)
                };
                let mut sleeper = |d: Duration| std::thread::sleep(d);
                streamed = true;
                request_completion_streaming(&mut stream_transport, &mut sleeper, request, on_delta)
            };
            match stream_out {
                Ok(s) => Ok(s),
                Err(StreamFail::Error(e)) => Err(e),
                // A legacy provider — remember it and fall back to buffered this same
                // attempt (the probe ran no generation, so no work is wasted).
                Err(StreamFail::Legacy) => {
                    streamed = false;
                    self.mark_buffered_only(provider_libp2p);
                    tracing::debug!(provider = %provider_libp2p, "provider is buffered-only; falling back to SERVE_REQUEST");
                    request_completion(&mut buffered, request, on_delta)
                }
            }
        };
        Exchange { result, proxy_roundtrip_ns: proxy_rt.get(), streamed }
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
        nonce: &[u8; 16],
        original_serve: &[u8],
        serve_timeout: std::time::Duration,
    ) -> Result<Vec<u8>, String> {
        let mut fetch_req = Vec::with_capacity(1 + nonce.len());
        fetch_req.push(FETCH_RESULT);
        fetch_req.extend_from_slice(nonce);
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
                    // engines; the ComfyUI /history cache makes the re-run instant).
                    Ok(FetchResponse::NotFound) => {
                        tracing::debug!(provider = %provider, "reconnect-fetch NotFound → re-serve");
                        return self
                            .net
                            .proxy_forward_timeout(
                                provider.to_string(),
                                original_serve.to_vec(),
                                serve_timeout,
                            )
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
            // On a *successfully co-signed* receipt the provider has served us `tokens` — credit
            // its contribution (M2.3 give-side), the mirror of the provider's take-side accrual.
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
    /// against this to decide whether to [`audit_model`](Self::audit_model) after serving.
    /// Pure read of the earned-reputation map — exposed so the trigger policy is testable
    /// and tunable separately from the dispatch.
    pub fn audit_rate_for(&self, peer_id: &str, model_key: &str, now_ms: u64) -> f64 {
        let rep = self.earned_reputation(peer_id, model_key, now_ms).unwrap_or(NEUTRAL_REPUTATION);
        sample_rate_for_reputation(rep, AUDIT_MIN_RATE, AUDIT_MAX_RATE)
    }

    /// M2.2(b) trigger: after serving, *sometimes* audit the provider that served. Draws
    /// against [`audit_rate_for`](Self::audit_rate_for) (rare for a trusted provider, up to
    /// [`AUDIT_MAX_RATE`] for a fresh/suspect one); on a hit it spawns a **detached
    /// background thread** that runs [`audit_model`](Self::audit_model) with a fresh
    /// unpredictable [`default_challenge`], so the redundant-exec cross-check never touches
    /// the caller's latency path. The audit records its own outcomes (a confirmed outlier →
    /// `Failed`); here we only log the verdict. A `< 2 providers` error just means the model
    /// can't be cross-checked right now — logged at debug, not an error.
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
        if lp.status.lock().unwrap_or_else(|e| e.into_inner()).get(&key).is_some_and(|s| now < s.next_due_ms) {
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
            let (declared, weights) = match request_model_info(&mut info_transport, &model) {
                Ok(info) => (info.identity(), info.weights()),
                Err(ScoreChallengeError::Unsupported) | Err(ScoreChallengeError::Refused(_)) => (None, None),
                Err(e) => {
                    tracing::debug!(provider = %peer_id, model, error = ?e, "model-info request failed");
                    (None, None)
                }
            };
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
                    let outcome = RoundOutcome::Incomplete("weights file not declared".into());
                    node.settle_logprob_round(&lp, &key, &peer_id, &model, &model_key, outcome, round, 0, None);
                    return;
                }
                tracing::info!(provider = %peer_id, model, verified_on, now = weights.as_deref().unwrap_or_default(),
                    "provider declares a different weights file than it was verified on; its pass is void");
                let voided = {
                    let mut status = lp.status.lock().unwrap_or_else(|e| e.into_inner());
                    let st = status.entry(key.clone()).or_default();
                    st.switch_reference([0; 16], &mut round);
                    *st
                };
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
            let reference = match resumed {
                // A registry run stopped with a text pending: carry on with the same pack.
                Some(r) => {
                    from_registry = true;
                    Some(r)
                }
                None => match select_reference(&lp.refs, &model, declared.as_ref(), weights.as_deref()).cloned() {
                Some(r) => Some(r),
                None => match &lp.registry {
                    Some(_) if first_contact && !lp.spend_first_contact_pack(now_unix_ms()) => {
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
                        Err(crate::registry_client::RegistryError::NoReference) => None,
                        Err(e) => {
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
                round = ChallengeRound::new();
            }
            if lp.status.lock().unwrap_or_else(|e| e.into_inner()).entry(key.clone()).or_default()
                .switch_reference(reference_tag(&reference.set), &mut round)
            {
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
            // Stopped with a text still pending on a registry pack: the next run resumes that pack.
            if from_registry && matches!(outcome, RoundOutcome::Busy | RoundOutcome::Incomplete(_)) && round.pending_text(&reference.set).is_some() {
                lp.resume.lock().unwrap_or_else(|e| e.into_inner()).insert(key.clone(), Arc::clone(&reference));
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
                let n = runs.entry(key.to_string()).or_insert(0);
                *n = n.saturating_add(1);
                logprob_retry_ms(*n)
            }
        };
        let effect = match &outcome {
            RoundOutcome::Verdict(v) => st.on_verdict_jittered(*v, now, rand::random::<f64>() * 2.0 - 1.0),
            RoundOutcome::Incomplete(_) => st.on_no_verdict(answered, retry, now),
            // Busy isn't held against a provider: the text it stalled on waits for it.
            RoundOutcome::Busy => st.on_no_verdict(false, retry, now),
            RoundOutcome::Evasion(_) => st.on_evasion(retry, now),
            RoundOutcome::Unsupported => st.on_no_verdict(answered, LOGPROB_BACKOFF_MS.max(retry), now),
            RoundOutcome::Exhausted => st.on_no_verdict(false, LOGPROB_BACKOFF_MS, now),
        };
        let streak = st.pass_streak;
        let final_status = *st;
        drop(status);
        // A pass earns no audit credit or reputation by itself (it only sets the method flag once
        // the streak qualifies): reputation must come from served traffic, audits from audits.
        match effect {
            StatusEffect::None => {}
            StatusEffect::Fail => {
                self.record_audit(peer_id, model_key, AuditClass::Fail, now);
                self.record_outcome(peer_id, model_key, VerificationOutcome::Failed, now);
            }
            StatusEffect::Evasion => self.record_audit(peer_id, model_key, AuditClass::Fail, now),
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

    fn maybe_audit(self: &Arc<Self>, model: &str, model_key: &str, provider_peer_id: &str) {
        let rate = self.audit_rate_for(provider_peer_id, model_key, now_unix_ms());
        if rand::random::<f64>() >= rate {
            return; // not sampled this time
        }
        let node = Arc::clone(self);
        let model = model.to_string();
        let (challenge, nonce) = default_challenge();
        std::thread::spawn(move || match node.audit_model(&model, &challenge, &nonce) {
            Ok(report) => {
                tracing::info!(model, verdict = ?report.verdict, "redundant-exec audit complete")
            }
            Err(e) => tracing::debug!(model, error = %e, "redundant-exec audit skipped"),
        });
    }

    /// Dispatch `challenge` to one specific `provider` deterministically (`temperature = 0`)
    /// and return its **full** completion text (the audit compares whole outputs, not a
    /// streamed view). Bounded by [`attempt_timeout`] (short — the audit is a 64-token probe)
    /// so a dead provider can't stall the audit.
    fn dispatch_full(
        self: &Arc<Self>,
        provider: &SelectedProvider,
        challenge: &[ChatMessage],
    ) -> Result<String, AdapterError> {
        let request = ServeRequest {
            reply_to: self.net.libp2p_peer_id().to_string(),
            model_ref: provider.model_id.clone(),
            messages: challenge.to_vec(),
            max_tokens: Some(AUDIT_MAX_TOKENS),
            // Redundant-execution comparison only holds for greedy decoding — a sampled
            // (temperature > 0) answer would legitimately differ between honest providers.
            temperature: Some(0.0),
            // The audit is a plain-text probe — no tools.
            tools: Vec::new(),
            // Disable thinking for the probe: a reasoning model would otherwise spend the tiny
            // 64-token budget on chain-of-thought (and prepend a `<think>…</think>` block that
            // pushes the required nonce echo out of `audit_body`'s near-the-start window),
            // making an honest provider look like a non-responder. Off ⇒ the nonce + greedy
            // continuation come straight out, keeping the redundant-exec comparison deterministic.
            // (Providers that can't honour `think:false` are still covered by `audit_body`'s
            // leading-`<think>` strip.)
            think: Some(false),
            // Settled like any serve (below), so the provider is paid for it and it looks like one.
            nonce: rand::random::<[u8; 16]>(),
            // Audits probe public marketplace providers; no swarm credential.
            credential: None,
            // Pin a fixed seed (2b): a temp=0 probe with a shared seed is reproducible across
            // honest providers, so redundant-exec agreement is exact rather than approximate.
            deterministic: Some(DeterministicParams { seed: AUDIT_SEED }),
            // Audits are paid for like any serve (and so look like one).
            settlement_peg: self.settlement_peg,
            settlement_domain: self.settlement_chain.as_ref().map(|c| c.domain.fingerprint()),
            settlement_wallet: self.settlement_wallet(),
        };
        let provider_libp2p = provider.libp2p_peer_id.clone();
        let mut transport = |framed: &[u8]| -> Result<Vec<u8>, AdapterError> {
            self.net
                .proxy_forward_timeout(provider_libp2p.clone(), framed.to_vec(), attempt_timeout(Some(AUDIT_MAX_TOKENS)))
                .map_err(|e| AdapterError::Http(format!("proxy_forward: {e}")))
        };
        let mut output = String::new();
        let mut delivered = crate::serve::DeliveredText::default();
        let mut sink = |delta: &str| {
            delivered.push(delta);
            output.push_str(delta)
        };
        let summary = request_completion(&mut transport, &request, &mut sink)?;
        // Pay for the probe as for any serve: the provider did the work, and an unpaid audit would
        // stand out (and, at a provider that charges, count as a serve never paid for).
        if summary.ok && summary.tokens > 0 && !self.is_self_serve(&provider.libp2p_peer_id) {
            let units = delivered.units() + crate::serve::tool_call_units(&summary.tool_calls, &crate::serve::offered_tool_names(&request.tools));
            let tokens = crate::serve::billable_tokens(summary.tokens, request.max_tokens, units);
            self.settle_receipt(provider, tokens, request.nonce);
        }
        Ok(output)
    }

    /// Run one **redundant-execution audit** (M2.2(b)) for `model`: send the same
    /// deterministic `challenge` to up to two distinct providers of the model, compare
    /// their full outputs via [`resolve_audit`], and record the resulting reputation
    /// outcomes (a confirmed outlier → `Failed`; agreeing providers → `Honored`; a
    /// non-responder → `Rejected`). On a 1-vs-1 `Inconclusive` split, escalates to a third
    /// provider (if discovery surfaced one) for a majority before recording.
    ///
    /// This is the *audit* path — separate from [`complete`](Self::complete) and off the
    /// user's latency path (the user already has their answer; a background sampler invokes
    /// this against a sampled fraction, rate from [`audit_rate_for`](Self::audit_rate_for)).
    /// Use an unpredictable `challenge` (see [`default_challenge`]) so a provider can't
    /// pre-cache the answer. Errors only if fewer than two providers are discoverable.
    ///
    /// *Caveat:* providers of the same canonical `model_id` may run different quantizations
    /// / engine builds whose greedy output differs; [`agrees`](openhydra_protocol::verify::agrees)
    /// absorbs benign late divergence and the `Inconclusive`-on-tie rule avoids punishing on
    /// ambiguous evidence, but heterogeneous fleets weaken the signal (reputation still
    /// carries the long-run trust). Reasoning models are a specific case of this — see the
    /// thinking-model limitation documented on [`AUDIT_MAX_TOKENS`].
    pub fn audit_model(
        self: &Arc<Self>,
        model: &str,
        challenge: &[ChatMessage],
        nonce: &str,
    ) -> Result<AuditReport, AdapterError> {
        let now = now_unix_ms();
        let peers = self
            .net
            .discover(model)
            .map_err(|e| AdapterError::Http(format!("discover: {e}")))?;
        let candidates = rank_providers_with_reputation(&peers, "", self.tier, &|pid, mkey| {
            self.earned_reputation(pid, mkey, now)
        });
        if candidates.len() < 2 {
            return Err(AdapterError::Http(format!(
                "redundant-exec audit needs ≥2 providers for '{model}', found {}",
                candidates.len()
            )));
        }

        // Dispatch to two providers, then escalate to a third only if they can't decide. They are
        // picked at random, not by rank, and must be independent (review T2): no two on the same
        // network as we observe it, and at most one we only reach through a relay — otherwise two
        // identities of one operator could take both slots and confirm each other. A provider that
        // charges and refuses us over payment isn't a non-responder (a price, not a fault): it is
        // skipped for the next one, and so is one that just refused us.
        let networks = self.observed_networks();
        let mut order: Vec<&SelectedProvider> =
            candidates.iter().filter(|c| !self.refused_payment(&c.libp2p_peer_id, &c.model_id, now)).collect();
        {
            use rand::seq::SliceRandom;
            order.shuffle(&mut rand::thread_rng());
        }
        let mut pool = order.into_iter();
        let mut chosen: Vec<Vec<String>> = Vec::new();
        let mut chosen_peers = std::collections::HashSet::new();
        let mut ask = |results: &mut Vec<(String, Result<String, String>)>| {
            for provider in pool.by_ref() {
                let nets = networks.get(&provider.libp2p_peer_id).cloned().unwrap_or_default();
                if !independent_of(&chosen, &nets) || chosen_peers.contains(&provider.libp2p_peer_id) {
                    continue;
                }
                match self.dispatch_full(provider, challenge) {
                    Err(e) if crate::serve::is_payment_refusal(&e) => {
                        self.note_payment_refusal(&provider.libp2p_peer_id, &provider.model_id, now);
                    }
                    out => {
                        chosen.push(nets);
                        chosen_peers.insert(provider.libp2p_peer_id.clone());
                        results.push((provider.state_id.clone(), out.map_err(|e| e.to_string())));
                        return;
                    }
                }
            }
        };
        let mut results: Vec<(String, Result<String, String>)> = Vec::with_capacity(3);
        ask(&mut results);
        ask(&mut results);
        if results.len() < 2 {
            return Err(AdapterError::Http(format!(
                "redundant-exec audit needs ≥2 providers that serve us '{model}', found {}",
                results.len()
            )));
        }
        // B-S2: gate each response on the required nonce echo and compare only the generated
        // *body* (after the nonce) — a freeloader that parrots the copyable prompt is filtered
        // to a non-responder rather than agreeing its way to Honored.
        let gate = |raw: &[(String, Result<String, String>)]| -> Vec<(String, Result<String, String>)> {
            raw.iter()
                .map(|(pid, r)| {
                    let mapped = match r {
                        Ok(out) => match audit_body(out, nonce) {
                            Some(body) => Ok(body.to_string()),
                            None => Err("audit response did not echo the challenge nonce".to_string()),
                        },
                        Err(e) => Err(e.clone()),
                    };
                    (pid.clone(), mapped)
                })
                .collect()
        };
        let mut report = resolve_audit(&gate(&results));
        if report.verdict == RedundantVerdict::Inconclusive {
            ask(&mut results);
            if results.len() == 3 {
                report = resolve_audit(&gate(&results));
            }
        }

        // Phase 4a: record each audit outcome under the SAME per-(peer, model) key the serve path
        // uses, so an audit `Failed` downranks the provider on this model specifically. Every
        // peer_id in `report.outcomes` came from `candidates`, so resolve the model key from there
        // (falling back to the audited `model` handle for safety).
        let key_of: std::collections::HashMap<&str, &str> = candidates
            .iter()
            .map(|c| (c.state_id.as_str(), rep_model_key(&c.canonical_model_id, &c.model_id)))
            .collect();
        for (peer_id, outcome) in &report.outcomes {
            let model_key = key_of.get(peer_id.as_str()).copied().unwrap_or(model);
            self.record_outcome(peer_id, model_key, *outcome, now);
            // Phase 4: fold the substantive temp=0 verdicts into the per-(peer,model) audit history
            // — an agreed serve is a Pass (lifts `n`), a clear divergence a Fail. A non-responder
            // (`Rejected`, a timeout) is not model-fraud evidence, so it never touches the history.
            // (This history feeds the future payout gate; it does not gate any spendable credit
            // today — the throttle is raw-token and display is handle-weighted.)
            let class = match outcome {
                VerificationOutcome::Honored => Some(AuditClass::Pass),
                VerificationOutcome::Failed => Some(AuditClass::Fail),
                VerificationOutcome::Rejected => None,
            };
            if let Some(class) = class {
                self.record_audit(peer_id, model_key, class, now);
            }
        }
        Ok(report)
    }
}

/// An unpredictable, deterministic audit challenge (M2.2(b)), returned with its `nonce` so the
/// verifier can require the echo and compare only the generated body (B-S2). The nonce defeats
/// answer-caching; the **generative continuation** (not present in the prompt) is what two
/// honest greedy runs of the same model reproduce and a prompt-parroting freeloader cannot.
/// Prompt design is heuristic and tunable.
pub fn default_challenge() -> (Vec<ChatMessage>, String) {
    let nonce_val: u128 = rand::random();
    let nonce = format!("{nonce_val:032x}");
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: format!(
            "Verification probe {nonce}. Respond with no preamble or explanation. \
             Line 1: output exactly the token {nonce}. From line 2 onward: deterministically \
             continue this passage — \"A peer joins the distributed network by\""
        ),
        ..Default::default()
    }];
    (messages, nonce)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Signed by someone else, for another model, or at another peg: never adopted.
        assert_eq!(judge(&receipt(&SigningKey::from_bytes(&[8u8; 32]), 1, "m", 9_999), 0), BehindReceipt::NotOurs);
        assert_eq!(judge(&receipt(&consumer, 1, "other", 9_999), 0), BehindReceipt::NotOurs);
        assert_eq!(judge(&signed_at(&consumer, 1, "m", 9_999, 2), 0), BehindReceipt::NotOurs);
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
        assert!(independent_of(&[], &a));
        assert!(independent_of(&[a.clone()], &b));
        assert!(!independent_of(&[a.clone()], &a), "same network");
        // One provider we only reach through a relay may join; a second may not.
        assert!(independent_of(&[a.clone()], &[]));
        assert!(!independent_of(&[a.clone(), vec![]], &[]));
        assert!(independent_of(&[vec![]], &b));
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
        assert!(over_reported_by_billing(&summary(500, crate::serve::BILLING_VERSION), 40));
        assert!(!over_reported_by_billing(&summary(40, crate::serve::BILLING_VERSION), 40));
        // An older provider's engine count can honestly be higher: signed for less, not faulted.
        assert!(!over_reported_by_billing(&summary(500, 0), 40));
        // Rule 3 bills a completed serve as 4 does.
        assert!(over_reported_by_billing(&summary(500, 3), 40));
        // Nor is a newer rule, which we can't judge by ours.
        assert!(!over_reported_by_billing(&summary(500, crate::serve::BILLING_VERSION + 1), 40));
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
        assert!(matches!(err, AdapterError::Http(m) if m.contains("engine down")));
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
        assert!(matches!(out, Err(StreamFail::Error(AdapterError::Http(m))) if m.contains("overloaded")));
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
        assert!(matches!(out, Err(StreamFail::Error(AdapterError::Http(m))) if m.contains("stall")));
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
        assert!(matches!(out, Err(StreamFail::Error(AdapterError::Http(m))) if m.contains("dropped")));
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
        assert!(matches!(out, Err(StreamFail::Error(AdapterError::Http(m))) if m.contains("engine exploded")));
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
                node.record_audit(peer, "m", AuditClass::Pass, t0 + i);
            }
        }
        assert_eq!(node.authenticity_of("p", "m", t0 + 100), AuthenticityTier::ReputationRedundant);
        // One failed audit: the floor, whatever the rest of its history says…
        node.record_audit("p", "m", AuditClass::Fail, t0 + 200);
        node.record_audit("p", "m", AuditClass::Pass, t0 + 300);
        node.record_audit("q", "m", AuditClass::Pass, t0 + 300);
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

    #[test]
    fn rep_model_key_uses_the_validated_handle_not_the_canonical() {
        // Keyed on the handle (matches KnownProvider.model_id + the commitment-bound identity),
        // regardless of the untrusted self-declared canonical.
        assert_eq!(rep_model_key("qwen3.5/2b/fp16/aa", "qwen2.5:7b"), "qwen2.5:7b");
        assert_eq!(rep_model_key("", "qwen2.5:7b"), "qwen2.5:7b");
    }

    #[test]
    fn default_challenge_is_unpredictable_and_well_formed() {
        // A single user-role message; the random nonce makes successive challenges differ
        // (so a provider can't pre-cache the answer) and appear twice (echo + restate).
        let (c1, n1) = default_challenge();
        let (c2, _) = default_challenge();
        assert_eq!(c1.len(), 1);
        assert_eq!(c1[0].role, "user");
        assert_ne!(c1[0].content, c2[0].content, "challenge nonce must vary");
        assert!(c1[0].content.contains(&n1), "prompt must echo its nonce");
        assert!(c1[0].content.contains("continue this passage"));
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
