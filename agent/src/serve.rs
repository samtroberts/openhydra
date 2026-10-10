// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The provider-side serve protocol — what a consumer sends to ask a provider to
//! generate, and the streamed frames the provider sends back.
//!
//! This is **transport-agnostic**: [`handle_serve_request`] decodes a request, drives
//! the engine via [`EngineAdapter::serve_stream`], and emits encoded [`ServeChunk`]s to
//! an injected sink. The libp2p wiring (advertise to the DHT, poll inbound proxy
//! requests, and *how* the chunks travel back — pushed to the consumer's peer via
//! `proxy_forward_no_wait`, since the inbound response is one-shot) lands on top in the
//! `network` integration step. Keeping the protocol + handler here makes it unit-testable
//! against a stub adapter with no swarm and no live engine.
//!
//! Wire formats:
//! * [`ServeRequest`] — JSON (one per request; structured, low-frequency).
//! * [`ServeChunk`] — compact tagged framing (one per token on the hot path): a 1-byte
//!   tag + payload (`0x01` delta ‖ utf8 · `0x02` done ‖ u64-LE tokens · `0x03` error ‖
//!   utf8).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::adapter::{
    AdapterError, ChatMessage, EngineAdapter, EngineMetrics, InferenceRequest, ToolCall,
};

/// Per-request metrics carried on the terminal `Done` frame: the engine's own numbers
/// plus the provider's serve-side wall time (so the consumer can separate network RTT
/// from provider processing).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServeMetrics {
    /// The engine's own per-request metrics (Ollama `eval_*`/`prompt_eval_*`/`load_*`).
    pub engine: EngineMetrics,
    /// Provider wall time inside `serve_stream` (engine HTTP call + reading its stream), ns.
    pub provider_serve_ns: u64,
    /// The billing rule the provider's token count follows: [`BILLING_VERSION`] when it reports
    /// [`billable_tokens`] (so a count above what the consumer computes is a false claim), `0` from a
    /// provider that predates it (which reported the engine's own count).
    #[serde(default)]
    pub billing: u8,
    /// The provider stopped generating before the engine finished: its consumer's credit window
    /// ran out, or nobody was reading. The answer is cut short, and the client is told so
    /// (`finish_reason: "length"`) rather than shown it as complete (review round 4 H8).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stopped: bool,
}

/// The billing rule this build follows: the reported count is [`billable_tokens`]. 3: whitespace
/// runs count by length, and only real calls to offered tools earn the template allowance
/// ([`allowance_earners`]). 4: a stream that ends early is owed for what the consumer read of it
/// ([`StreamTally`]); a completed one is billed as in 3.
pub const BILLING_VERSION: u8 = 4;

/// Whether a provider reporting billing rule `billing` computes a completed serve's count as this
/// build does: 3 and 4 differ only for streams that end early.
pub fn same_completed_billing(billing: u8) -> bool {
    (3..=BILLING_VERSION).contains(&billing)
}

/// How a provider that charges starts the error refusing a serve over payment (the consumer's offer
/// or its unpaid serves). A consumer doesn't hold it against the provider: it is a price, not a fault.
pub const PAYMENT_REFUSAL: &str = "payment required: ";

/// Whether a serve error is a [`PAYMENT_REFUSAL`]: the provider's error frame, starting with it.
/// (Not an error that merely mentions it: a provider's other errors, an engine's text among them,
/// can say anything.) Only a refusal before anything was delivered is one; a provider refuses at
/// admission, never mid-stream.
pub fn is_payment_refusal(error: &crate::adapter::AdapterError) -> bool {
    matches!(error, crate::adapter::AdapterError::Provider(m) | crate::adapter::AdapterError::Http(m) if m.starts_with(PAYMENT_REFUSAL))
}

/// Tokens with no text of their own (end-of-turn and other special markers) allowed on top of the
/// delivered text in [`billable_tokens`].
pub const BILLABLE_TOKEN_SLACK: u64 = 16;
/// Per tool call: the template tokens it is wrapped in (`<tool_call>`, the JSON envelope around the
/// name and arguments), which the engine generates but neither side sees as text.
pub const TOOL_CALL_TOKEN_ALLOWANCE: u64 = 32;

/// Whitespace characters one unit of a whitespace run stands for: tokenizers merge runs of spaces
/// and newlines into tokens of up to about this length, so a long run (deep indentation, a table)
/// is still paid for, while padding is worth a sixteenth of its size.
const WHITESPACE_PER_UNIT: u64 = 16;

/// A measure of generated text that bounds the tokens it could have taken: the UTF-8 bytes of its
/// non-whitespace characters, plus one per [`WHITESPACE_PER_UNIT`] characters of each run of
/// whitespace. Every token decodes to at least one byte, and whitespace tokens merge runs, so padding
/// (spaces in tool-call arguments, blank lines in a reasoning block) adds little. Fed delta by delta;
/// a run split across two deltas is one run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeliveredText {
    units: u64,
    /// Length of the whitespace run the text currently ends in (0 = not in one).
    run: u64,
}

impl DeliveredText {
    pub fn push(&mut self, text: &str) {
        for c in text.chars() {
            if c.is_whitespace() {
                if self.run % WHITESPACE_PER_UNIT == 0 {
                    self.units += 1;
                }
                self.run += 1;
            } else {
                self.units += c.len_utf8() as u64;
                self.run = 0;
            }
        }
    }

    pub fn units(&self) -> u64 {
        self.units
    }
}

/// The most tool calls in one completion that earn [`TOOL_CALL_TOKEN_ALLOWANCE`].
pub const MAX_ALLOWED_TOOL_CALLS: usize = 16;

/// The function names a request's OpenAI `tools` offer.
pub fn offered_tool_names(tools: &[Value]) -> std::collections::HashSet<String> {
    tools
        .iter()
        .filter_map(|t| t.get("function").and_then(|f| f.get("name")).or_else(|| t.get("name")))
        .filter_map(|n| n.as_str().map(str::to_string))
        .collect()
}

/// Which of `tool_calls` earn [`TOOL_CALL_TOKEN_ALLOWANCE`]: a call to a tool the request offered
/// (`offered`, [`offered_tool_names`]) whose arguments are a JSON object, as a real call's are, once
/// per distinct call, for at most [`MAX_ALLOWED_TOOL_CALLS`] calls. A call to nothing offered, with
/// junk arguments, repeated, or past the cap is billed for its text only, so made-up calls can't pad
/// a bill. Provider and consumer (and the token-count check) all use this.
pub fn allowance_earners(tool_calls: &[ToolCall], offered: &std::collections::HashSet<String>) -> Vec<bool> {
    let mut seen = std::collections::HashSet::new();
    tool_calls
        .iter()
        .map(|c| {
            let earns = seen.len() < MAX_ALLOWED_TOOL_CALLS
                && offered.contains(&c.function.name)
                && serde_json::from_str::<Value>(&c.function.arguments).is_ok_and(|v| v.is_object())
                && seen.insert((c.function.name.as_str(), c.function.arguments.as_str()));
            earns
        })
        .collect()
}

/// [`DeliveredText`] units of `tool_calls` (name and arguments), plus [`TOOL_CALL_TOKEN_ALLOWANCE`]
/// for each call that earns it ([`allowance_earners`]).
pub fn tool_call_units(tool_calls: &[ToolCall], offered: &std::collections::HashSet<String>) -> u64 {
    tool_calls
        .iter()
        .zip(allowance_earners(tool_calls, offered))
        .map(|(c, earns)| {
            let mut t = DeliveredText::default();
            t.push(&c.function.name);
            let mut a = DeliveredText::default();
            a.push(&c.function.arguments);
            t.units() + a.units() + if earns { TOOL_CALL_TOKEN_ALLOWANCE } else { 0 }
        })
        .sum()
}

/// [`tool_call_units`] at their most, whatever tools were offered: every call earns the allowance, up
/// to [`MAX_ALLOWED_TOOL_CALLS`]. Never less than [`tool_call_units`] for any offered set, so bounding
/// an honest count by it changes nothing.
pub fn tool_call_units_at_most(tool_calls: &[ToolCall]) -> u64 {
    tool_calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut t = DeliveredText::default();
            t.push(&c.function.name);
            let mut a = DeliveredText::default();
            a.push(&c.function.arguments);
            t.units() + a.units() + if i < MAX_ALLOWED_TOOL_CALLS { TOOL_CALL_TOKEN_ALLOWANCE } else { 0 }
        })
        .sum()
}

/// The completion tokens a serve is billed for, which provider and consumer each compute from the
/// same inputs: the engine's count, but never more than the request allowed (`max_tokens`) nor more
/// than the delivered text (`delivered`, from [`DeliveredText`] and [`tool_call_units`]) could have
/// taken, plus [`BILLABLE_TOKEN_SLACK`]. The provider reports this count and commits to it; the
/// consumer co-signs for the same count from the text it received, so an honest provider's receipt
/// always matches. Tokens the engine counted but never sent (hidden reasoning, padding) aren't billed.
pub fn billable_tokens(engine_tokens: u64, max_tokens: Option<u32>, delivered: u64) -> u64 {
    let by_text = delivered.saturating_add(BILLABLE_TOKEN_SLACK);
    let by_request = max_tokens.map_or(u64::MAX, u64::from);
    engine_tokens.min(by_text).min(by_request)
}

/// The billable count of a stream as far as one side has it, frame by frame ([`BILLING_VERSION`] 4).
/// Provider and consumer feed it the same frames (the provider those a poll returned, the consumer
/// those it received), so they arrive at the same count:
/// * up to the `Done` frame, a stream that ends early is owed [`billable_tokens`] of its text
///   fragments (about one token each, as the engine streams them) for its text: what a stopped serve
///   reports too;
/// * the `Done` frame carries the count for the whole completion, tool calls included, bounded as
///   any count is by [`billable_tokens`]: by `max_tokens`, and by the text and tool calls the frames
///   carried (a provider can't make its consumer sign for more by putting a bigger number in a
///   `Done`, review round 4 C1). The tool calls are bounded at their most ([`tool_call_units_at_most`]),
///   so an honest `Done`, already bounded by the offered tools, passes unchanged on both sides.
///
/// A `ToolCalls` or `Error` frame adds nothing on its own.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamTally {
    fragments: u64,
    text: DeliveredText,
    tools: u64,
    done: Option<u64>,
}

impl StreamTally {
    /// Count one received text fragment.
    pub fn push_delta(&mut self, text: &str) {
        self.fragments += 1;
        self.text.push(text);
    }

    /// Count one encoded frame ([`ServeChunk::encode`]).
    pub fn push_encoded(&mut self, frame: &[u8]) {
        match frame.first() {
            Some(&TAG_DELTA) => self.push_delta(&String::from_utf8_lossy(&frame[1..])),
            Some(&TAG_DONE) => {
                if let Ok(ServeChunk::Done { tokens, .. }) = ServeChunk::decode(frame) {
                    self.done = Some(tokens);
                }
            }
            Some(&TAG_TOOLCALLS) => {
                if let Ok(ServeChunk::ToolCalls(calls)) = ServeChunk::decode(frame) {
                    self.tools = self.tools.saturating_add(tool_call_units_at_most(&calls));
                }
            }
            _ => {}
        }
    }

    /// Count the tool calls a stream delivered (the consumer gets them decoded, not as a frame).
    pub fn push_tool_calls(&mut self, calls: &[ToolCall]) {
        self.tools = self.tools.saturating_add(tool_call_units_at_most(calls));
    }

    /// What the frames counted so far are billed for, under a request for `max_tokens`.
    pub fn billable(&self, max_tokens: Option<u32>) -> u64 {
        match self.done {
            Some(done) => billable_tokens(done, max_tokens, self.text.units().saturating_add(self.tools)),
            None => billable_tokens(self.fragments, max_tokens, self.text.units()),
        }
    }

    /// Text fragments counted so far.
    pub fn fragments(&self) -> u64 {
        self.fragments
    }
}

/// What a serve that ended in an error is billed for (round 8 S8-2): the text fragments its reply
/// carried, counted as a stream that ends early is ([`StreamTally`]). The consumer counts the same
/// fragments as they reach it, so both sides arrive at this count.
pub fn failed_serve_billable(frames: &[Vec<u8>], max_tokens: Option<u32>) -> u64 {
    let mut tally = StreamTally::default();
    for frame in frames.iter().filter(|f| f.first() == Some(&TAG_DELTA)) {
        tally.push_encoded(frame);
    }
    tally.billable(max_tokens)
}

/// A consumer's request for a provider to serve a streaming completion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServeRequest {
    /// The consumer's libp2p peer id — where the provider pushes the response chunks.
    /// (Carried for the wiring layer; [`handle_serve_request`] itself is transport-agnostic.)
    pub reply_to: String,
    /// The model the consumer wants. The provider maps this (canonical id or engine
    /// handle) to one of its engine's models.
    pub model_ref: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f64>,
    /// OpenAI `tools` specs the consumer forwards for the provider's engine (empty ⇒ none).
    /// `#[serde(default)]` so an older consumer that omits the field still deserializes.
    #[serde(default)]
    pub tools: Vec<Value>,
    /// Deterministic thinking-mode control for reasoning models. `Some(false)` asks the engine
    /// to skip its chain-of-thought (answer goes straight to `content`); `Some(true)` forces it
    /// on; `None` leaves the engine's default. Normalised at the consumer from the OpenAI-body
    /// `think` / `chat_template_kwargs.enable_thinking` fields. `#[serde(default)]` keeps the
    /// wire backward-compatible — an older consumer omits it, an older provider ignores it.
    #[serde(default)]
    pub think: Option<bool>,
    /// The receipt nonce the consumer commits *before* the serve (B-S1). The provider
    /// records the tokens it serves under this nonce and later co-signs the settlement
    /// receipt only if the same nonce is presented with `tokens <= served`, the same model,
    /// once, and fresh. Required: a serve carrying no committed nonce cannot be settled.
    pub nonce: [u8; 16],
    /// M4-base: a swarm membership credential the consumer presents to reach a provider's
    /// **private** (swarm-scoped) model. `#[serde(default)]` keeps the wire backward-compatible — an
    /// older consumer omits it (and can't reach a private model), an older provider ignores it (it
    /// has no private models to gate). The provider verifies it against a swarm it owns and binds it
    /// to the authenticated sender before generating; a `global` model ignores it entirely.
    #[serde(default)]
    pub credential: Option<openhydra_network::membership::MembershipCredential>,
    /// UNIFIED_TRUST_STACK Increment 2b: when set, ask the provider for a **deterministic serve** —
    /// temperature pinned to 0 and the carried seed pinned — so the same `(model, quant, prompt)`
    /// reproduces the same output and a fingerprint / redundant-exec check is meaningful
    /// (PROOF_OF_INFERENCE §12 Phase 1). Overrides [`temperature`](Self::temperature) to 0.
    /// `#[serde(default)]` keeps the wire backward-compatible — an older consumer omits it, an older
    /// provider ignores it (its adapter simply won't pin the seed).
    #[serde(default)]
    pub deterministic: Option<DeterministicParams>,
    /// The settlement peg (micro-USDC per credit) the consumer will pay at, when it settles on the
    /// crypto rail. A provider that charges refuses a serve that doesn't offer its own peg, before
    /// running the engine, rather than serving and then refusing an unpaid receipt. `None` from a
    /// consumer that doesn't pay (or predates the field).
    #[serde(default)]
    pub settlement_peg: Option<u64>,
    /// The on-chain deployment the consumer settles on ([`openhydra_protocol::settlement::ChainDomain::fingerprint`]).
    /// A provider settling on a chain refuses a serve for another deployment (or none), before running
    /// the engine. `None` from a consumer off-chain (or predating the field).
    #[serde(default)]
    pub settlement_domain: Option<String>,
    /// On Base, the wallet (`0x`-prefixed hex address) whose channel the consumer's claims draw on. A
    /// provider holding a pair's live claim from another wallet refuses the serve up front, rather
    /// than serving and then refusing the receipt. `None` elsewhere (or predating the field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settlement_wallet: Option<String>,
}

/// Parameters for a deterministic (reproducible) serve. The model's quantisation is already pinned
/// by the canonical model id the consumer routed to, so only the RNG seed needs carrying;
/// temperature is forced to 0 by the provider when this is present. See [`ServeRequest::deterministic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicParams {
    /// The RNG seed the provider pins for this serve. A fixed value makes generation reproducible
    /// across providers running the same model + quant.
    pub seed: u64,
}

impl ServeRequest {
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ServeRequest serializes")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, AdapterError> {
        serde_json::from_slice(bytes).map_err(|e| AdapterError::Parse(e.to_string()))
    }
}

const TAG_DELTA: u8 = 0x01;
const TAG_DONE: u8 = 0x02;
const TAG_ERROR: u8 = 0x03;
const TAG_TOOLCALLS: u8 = 0x04;

/// One frame the provider streams back to the consumer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServeChunk {
    /// A text fragment of the completion.
    Delta(String),
    /// The tool calls the model requested this turn (OpenAI `tool_calls`). Sent once,
    /// terminal, just before `Done` — structured data that can't ride the text `Delta`
    /// path. Absent when the model produced plain text.
    ToolCalls(Vec<ToolCall>),
    /// End of stream + the completion token count (for the co-signed receipt) and the
    /// per-request [`ServeMetrics`] (engine numbers + provider serve time).
    Done { tokens: u64, metrics: ServeMetrics },
    /// The provider/engine failed; the stream ends with this instead of `Done`.
    Error(String),
}

impl ServeChunk {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            ServeChunk::Delta(s) => {
                let mut b = Vec::with_capacity(1 + s.len());
                b.push(TAG_DELTA);
                b.extend_from_slice(s.as_bytes());
                b
            }
            ServeChunk::Done { tokens, metrics } => {
                // tag · u64 tokens · JSON(metrics). One frame per request, so JSON's fine
                // and keeps the metric set extensible.
                let json = serde_json::to_vec(metrics).unwrap_or_default();
                let mut b = Vec::with_capacity(9 + json.len());
                b.push(TAG_DONE);
                b.extend_from_slice(&tokens.to_le_bytes());
                b.extend_from_slice(&json);
                b
            }
            ServeChunk::Error(s) => {
                let mut b = Vec::with_capacity(1 + s.len());
                b.push(TAG_ERROR);
                b.extend_from_slice(s.as_bytes());
                b
            }
            ServeChunk::ToolCalls(calls) => {
                // tag · JSON(Vec<ToolCall>). One frame per request (terminal), so JSON is
                // fine and keeps the tool-call shape extensible.
                let json = serde_json::to_vec(calls).unwrap_or_default();
                let mut b = Vec::with_capacity(1 + json.len());
                b.push(TAG_TOOLCALLS);
                b.extend_from_slice(&json);
                b
            }
        }
    }

    /// Whether an encoded frame is a **terminal** (`Done`/`Error`) chunk — the last frame of
    /// a stream. The provider's streaming buffer sets its `done` flag atomically with appending
    /// such a frame, so a poll never observes `done` without the terminal chunk being fetchable.
    pub fn frame_is_terminal(encoded: &[u8]) -> bool {
        matches!(encoded.first(), Some(&TAG_DONE) | Some(&TAG_ERROR))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, AdapterError> {
        match bytes.first() {
            Some(&TAG_DELTA) => Ok(ServeChunk::Delta(
                String::from_utf8_lossy(&bytes[1..]).into_owned(),
            )),
            Some(&TAG_DONE) if bytes.len() >= 9 => {
                let mut t = [0u8; 8];
                t.copy_from_slice(&bytes[1..9]);
                let metrics = serde_json::from_slice(&bytes[9..]).unwrap_or_default();
                Ok(ServeChunk::Done { tokens: u64::from_le_bytes(t), metrics })
            }
            Some(&TAG_ERROR) => Ok(ServeChunk::Error(
                String::from_utf8_lossy(&bytes[1..]).into_owned(),
            )),
            Some(&TAG_TOOLCALLS) => serde_json::from_slice(&bytes[1..])
                .map(ServeChunk::ToolCalls)
                .map_err(|e| AdapterError::Parse(format!("serve chunk: bad tool_calls frame: {e}"))),
            _ => Err(AdapterError::Parse("serve chunk: bad/short frame".into())),
        }
    }
}

/// Outcome of serving one request (returned to the wiring layer for the receipt).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServeSummary {
    /// Completion tokens generated (0 on a failed request).
    pub tokens: u64,
    /// Whether the stream completed cleanly (vs ending in an error frame).
    pub ok: bool,
    /// Engine + provider-serve metrics from the `Done` frame.
    pub metrics: ServeMetrics,
    /// Consumer-side: time spent in `discover` (ns). Set by the consumer; 0 provider-side.
    pub discover_ns: u64,
    /// Consumer-side: the `proxy_forward` round-trip (ns) = network RTT + provider serve.
    /// Set by the consumer; 0 provider-side.
    pub proxy_roundtrip_ns: u64,
    /// Tool calls the model requested (OpenAI `tool_calls`), from the `ToolCalls` frame;
    /// empty for a plain-text completion. The gateway emits these as the assistant turn's
    /// `tool_calls` with `finish_reason: "tool_calls"`.
    pub tool_calls: Vec<ToolCall>,
    /// Consumer-side: the authenticity tier this consumer had established for the provider that
    /// served (shown to the client, which can require a minimum). `None` provider-side and for a
    /// serve by our own provider.
    pub authenticity: Option<openhydra_protocol::trust::AuthenticityTier>,
    /// Consumer-side: the provider streamed this serve, so `proxy_roundtrip_ns` is the time to its
    /// first acknowledgement (sent before it generates), not the whole serve.
    pub streamed: bool,
}

/// Provider-side: decode an inbound `request`, serve it via `adapter`, and emit each
/// response frame (encoded [`ServeChunk`]) to `send_chunk`.
///
/// The stream always terminates with exactly one `Done` *or* one `Error` frame, so the
/// consumer knows the generation ended. A malformed request, or an engine failure, is
/// reported as an `Error` frame (never panics, never silently drops the stream). The
/// returned [`ServeSummary.tokens`] feeds the co-signed receipt at the wiring layer.
pub fn handle_serve_request(
    request: &[u8],
    adapter: &dyn EngineAdapter,
    send_chunk: &mut dyn FnMut(&[u8]),
) -> ServeSummary {
    let req = match ServeRequest::decode(request) {
        Ok(r) => r,
        Err(e) => {
            send_chunk(&ServeChunk::Error(format!("bad serve request: {e}")).encode());
            return ServeSummary {
                tokens: 0,
                ok: false,
                metrics: ServeMetrics::default(),
                discover_ns: 0,
                proxy_roundtrip_ns: 0,
                tool_calls: Vec::new(),
                authenticity: None,
                streamed: false,
            };
        }
    };
    handle_serve_request_parsed(req, adapter, send_chunk)
}

/// F-C5: serve an already-decoded [`ServeRequest`]. The provider decodes the request once on
/// the inbound path and threads it through throttle/AUP/dispatch, so the hot serve path no
/// longer re-decodes the (potentially large) prompt several times per request.
pub fn handle_serve_request_parsed(
    req: ServeRequest,
    adapter: &dyn EngineAdapter,
    send_chunk: &mut dyn FnMut(&[u8]),
) -> ServeSummary {
    handle_serve_request_within(req, adapter, send_chunk, &mut |_| true)
}

/// [`handle_serve_request_parsed`] that ends the serve early once `keep_going(tokens)` says no, given
/// the tokens sent so far (the stream's text fragments, about one token each): the engine stops,
/// and the serve finishes normally, reported and billed for what was sent.
pub fn handle_serve_request_within(
    req: ServeRequest,
    adapter: &dyn EngineAdapter,
    send_chunk: &mut dyn FnMut(&[u8]),
    keep_going: &mut dyn FnMut(u64) -> bool,
) -> ServeSummary {
    // A deterministic serve pins temperature to 0 and carries a fixed seed, overriding any
    // temperature the request also set (2b). Otherwise the request's own sampling is used.
    // Engines disagree on what a zero cap means (some read it as "no limit"), and nothing generated
    // under it is billable; refuse it before the engine runs.
    if req.max_tokens == Some(0) {
        send_chunk(&ServeChunk::Error("max_tokens must be at least 1".into()).encode());
        return ServeSummary { ok: false, ..Default::default() };
    }
    let max_tokens = req.max_tokens;
    let offered = offered_tool_names(&req.tools);
    let (temperature, seed) = match req.deterministic {
        Some(d) => (Some(0.0), Some(d.seed)),
        None => (req.temperature, None),
    };
    let infer = InferenceRequest {
        model_ref: req.model_ref,
        messages: req.messages,
        max_tokens: req.max_tokens,
        temperature,
        tools: req.tools,
        think: req.think,
        seed,
        stop: Default::default(),
    };
    let stop = infer.stop.clone();

    // Scope the delta sink so its &mut borrow of `send_chunk` ends before the terminal
    // Done/Error frame is sent. Time the whole serve so the consumer can separate network
    // RTT from provider-side processing.
    let serve_start = std::time::Instant::now();
    // The text exactly as sent, measured for the billable count (the consumer measures what it
    // receives the same way).
    let mut delivered = DeliveredText::default();
    let mut fragments = 0u64;
    let outcome = {
        let mut on_delta = |d: &str| {
            delivered.push(d);
            fragments += 1;
            send_chunk(&ServeChunk::Delta(d.to_string()).encode());
            if !stop.is_set() && !keep_going(fragments) {
                stop.set();
            }
        };
        adapter.serve_stream(&infer, &mut on_delta)
    };
    // Stopped early, the engine never reported its count: what was sent stands in for it.
    let outcome = outcome.map(|mut o| {
        if stop.is_set() {
            o.tokens = fragments;
            o.done = true;
        }
        o
    });
    let provider_serve_ns = serve_start.elapsed().as_nanos() as u64;

    match outcome {
        Ok(o) => {
            let metrics =
                ServeMetrics { engine: o.engine, provider_serve_ns, billing: BILLING_VERSION, stopped: stop.is_set() };
            // Tool calls (terminal, structured) go in their own frame just before `Done`, so
            // the consumer sees text deltas → tool_calls → done in order.
            if !o.tool_calls.is_empty() {
                send_chunk(&ServeChunk::ToolCalls(o.tool_calls.clone()).encode());
            }
            // Reported and committed to (and so paid) is the billable count, not the engine's:
            // the engine's own count stays in `metrics.engine`.
            let tokens =
                billable_tokens(o.tokens, max_tokens, delivered.units() + tool_call_units(&o.tool_calls, &offered));
            send_chunk(&ServeChunk::Done { tokens, metrics }.encode());
            ServeSummary {
                tokens,
                ok: true,
                metrics,
                discover_ns: 0,
                proxy_roundtrip_ns: 0,
                tool_calls: o.tool_calls,
                authenticity: None,
                streamed: false,
            }
        }
        Err(e) => {
            send_chunk(&ServeChunk::Error(e.to_string()).encode());
            ServeSummary {
                tokens: 0,
                ok: false,
                metrics: ServeMetrics { provider_serve_ns, ..Default::default() },
                discover_ns: 0,
                proxy_roundtrip_ns: 0,
                tool_calls: Vec::new(),
                authenticity: None,
                streamed: false,
            }
        }
    }
}

/// Frame a buffered sequence of encoded chunks for one-shot delivery: repeated
/// `[u32-LE len][encoded ServeChunk]`. The first transport is buffered request/response
/// (the inbound libp2p reply is one-shot); push-based incremental streaming is a later
/// refinement that pushes each [`ServeChunk`] to the consumer's `reply_to` directly.
pub fn frame_response(encoded_chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut buf = Vec::new();
    for c in encoded_chunks {
        buf.extend_from_slice(&(c.len() as u32).to_le_bytes());
        buf.extend_from_slice(c);
    }
    buf
}

/// Reconnect-and-fetch (long-serve drop tolerance): a consumer whose serve connection was
/// evicted during a long generation re-requests the buffered result **by nonce** on a fresh
/// circuit, instead of losing the completed work. The provider keys the buffer on the same
/// [`ServeRequest::nonce`] it already records for settlement, so no new correlation id is
/// needed. See `RECONNECT_AND_FETCH_PLAN.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchResponse {
    /// The buffered serve frames — byte-identical to what a fresh serve round-trip returns,
    /// so the consumer decodes them with [`parse_response`] exactly as usual.
    Ready(Vec<u8>),
    /// The serve is still running (buffer holds a Generating marker); retry after a backoff.
    Generating,
    /// No buffered result under this nonce — the provider restarted or the buffer TTL expired.
    /// The consumer re-serves from scratch (deterministic engines cache-hit).
    NotFound,
    /// The fetch's libp2p-authenticated sender is not the consumer that committed the nonce.
    /// Guards against a peer that learns/guesses a nonce fetching another consumer's result.
    Forbidden,
}

const FETCH_READY: u8 = 0x01;
const FETCH_GENERATING: u8 = 0x02;
const FETCH_NOTFOUND: u8 = 0x03;
const FETCH_FORBIDDEN: u8 = 0x04;

impl FetchResponse {
    /// Whether an encoded response is [`FetchResponse::Ready`] (carries the result), without
    /// decoding it.
    pub fn encoded_is_ready(bytes: &[u8]) -> bool {
        bytes.first() == Some(&FETCH_READY)
    }

    pub fn encode(&self) -> Vec<u8> {
        match self {
            FetchResponse::Ready(framed) => {
                let mut b = Vec::with_capacity(1 + framed.len());
                b.push(FETCH_READY);
                b.extend_from_slice(framed);
                b
            }
            FetchResponse::Generating => vec![FETCH_GENERATING],
            FetchResponse::NotFound => vec![FETCH_NOTFOUND],
            FetchResponse::Forbidden => vec![FETCH_FORBIDDEN],
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, AdapterError> {
        match bytes.first() {
            Some(&FETCH_READY) => Ok(FetchResponse::Ready(bytes[1..].to_vec())),
            Some(&FETCH_GENERATING) => Ok(FetchResponse::Generating),
            Some(&FETCH_NOTFOUND) => Ok(FetchResponse::NotFound),
            Some(&FETCH_FORBIDDEN) => Ok(FetchResponse::Forbidden),
            _ => Err(AdapterError::Parse("fetch response: bad/empty frame".into())),
        }
    }
}

/// Response to a `FETCH_CHUNKS` poll (streaming serve, P1): the newly-produced serve frames at
/// or after the requested chunk offset, the offset to poll from next, and whether the stream is
/// complete (a terminal `Done`/`Error` chunk is included). `NotFound`/`Forbidden` mirror
/// [`FetchResponse`]. Unlike the buffered `FetchResponse::Ready` (the *whole* result), this
/// carries only the incremental slice `[offset..]`, so a long generation streams in bounded polls
/// instead of one blob under a total-time cap.
#[derive(Debug, PartialEq, Eq)]
pub enum FetchChunksResponse {
    /// `framed` = [`frame_response`] of the chunks `[offset..]` — decode with [`parse_response`].
    /// `next_offset` is the chunk index to request next; `done` marks end-of-stream.
    Chunks { framed: Vec<u8>, next_offset: u32, done: bool },
    /// No buffer under this nonce (provider restarted / TTL expired) — the consumer re-serves.
    NotFound,
    /// The poll's authenticated sender is not the consumer that committed the nonce.
    Forbidden,
    /// The provider shed the `SERVE_STREAM` at submit (worker pool full) — no buffer was
    /// created. Only ever returned as the immediate ack to a `SERVE_STREAM`; the consumer
    /// fails over to another provider (or retries) rather than polling a stream that will
    /// never exist.
    Overloaded,
}

// High, unambiguous discriminator bytes so a streaming consumer can tell a real
// `FetchChunksResponse` from a legacy provider's reply by structural decode alone. A legacy
// provider that doesn't know the `SERVE_STREAM` method byte answers it with a *fixed*
// `frame_response([Error("unsupported method")])`, which begins with that frame's little-endian
// `u32` length prefix (low byte `0x13` = 19, the frame's byte length) — far below `0xF1`. So a
// reply whose first byte is `0xF1..=0xF4` is unambiguously a real streaming reply, and the
// legacy error blob decodes cleanly to a decode error (→ the consumer falls back to the buffered
// `SERVE_REQUEST`). The bytes are kept in the high range as defense-in-depth against any other
// small `frame_response` blob a consumer might mis-probe.
const FC_CHUNKS: u8 = 0xF1;
const FC_NOTFOUND: u8 = 0xF2;
const FC_FORBIDDEN: u8 = 0xF3;
const FC_OVERLOADED: u8 = 0xF4;

impl FetchChunksResponse {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            FetchChunksResponse::Chunks { framed, next_offset, done } => {
                let mut b = Vec::with_capacity(6 + framed.len());
                b.push(FC_CHUNKS);
                b.extend_from_slice(&next_offset.to_be_bytes());
                b.push(u8::from(*done));
                b.extend_from_slice(framed);
                b
            }
            FetchChunksResponse::NotFound => vec![FC_NOTFOUND],
            FetchChunksResponse::Forbidden => vec![FC_FORBIDDEN],
            FetchChunksResponse::Overloaded => vec![FC_OVERLOADED],
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, AdapterError> {
        match bytes.first() {
            Some(&FC_CHUNKS) => {
                if bytes.len() < 6 {
                    return Err(AdapterError::Parse("fetch-chunks: short header".into()));
                }
                let next_offset = u32::from_be_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
                let done = bytes[5] != 0;
                Ok(FetchChunksResponse::Chunks { framed: bytes[6..].to_vec(), next_offset, done })
            }
            Some(&FC_NOTFOUND) => Ok(FetchChunksResponse::NotFound),
            Some(&FC_FORBIDDEN) => Ok(FetchChunksResponse::Forbidden),
            Some(&FC_OVERLOADED) => Ok(FetchChunksResponse::Overloaded),
            _ => Err(AdapterError::Parse("fetch-chunks: bad/empty frame".into())),
        }
    }
}

/// Parse a [`frame_response`] buffer back into chunks (consumer side).
pub fn parse_response(buffer: &[u8]) -> Result<Vec<ServeChunk>, AdapterError> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < buffer.len() {
        if i + 4 > buffer.len() {
            return Err(AdapterError::Parse("serve response: truncated length prefix".into()));
        }
        let len = u32::from_le_bytes(buffer[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        if i + len > buffer.len() {
            return Err(AdapterError::Parse("serve response: truncated chunk body".into()));
        }
        out.push(ServeChunk::decode(&buffer[i..i + len])?);
        i += len;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::{DetectedModel, ServeOutcome, ToolCallFunction};

    fn tool_call(name: &str) -> ToolCall {
        ToolCall {
            id: "call_1".into(),
            kind: "function".into(),
            function: ToolCallFunction { name: name.into(), arguments: "{}".into() },
        }
    }

    fn req(reply_to: &str) -> ServeRequest {
        ServeRequest {
            reply_to: reply_to.into(),
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

    /// A canned adapter: emits fixed deltas (or fails) without any engine.
    struct StubAdapter {
        deltas: Vec<String>,
        tokens: u64,
        fail: Option<AdapterError>,
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
            _request: &InferenceRequest,
            on_delta: &mut dyn FnMut(&str),
        ) -> Result<ServeOutcome, AdapterError> {
            if let Some(e) = &self.fail {
                return Err(e.clone());
            }
            for d in &self.deltas {
                on_delta(d);
            }
            Ok(ServeOutcome { tokens: self.tokens, done: true, engine: Default::default(), tool_calls: Vec::new() })
        }
    }

    fn collect_chunks(request: &[u8], adapter: &dyn EngineAdapter) -> (Vec<ServeChunk>, ServeSummary) {
        let mut frames: Vec<ServeChunk> = Vec::new();
        let summary = handle_serve_request(request, adapter, &mut |bytes| {
            frames.push(ServeChunk::decode(bytes).unwrap());
        });
        (frames, summary)
    }

    #[test]
    fn a_serve_out_of_budget_stops_the_engine_and_bills_what_was_sent() {
        struct Long;
        impl EngineAdapter for Long {
            fn engine_name(&self) -> &'static str {
                "long"
            }
            fn detect_models(&self) -> Result<Vec<crate::adapter::DetectedModel>, AdapterError> {
                Ok(vec![])
            }
            fn serve_stream(
                &self,
                req: &InferenceRequest,
                on_delta: &mut dyn FnMut(&str),
            ) -> Result<crate::adapter::ServeOutcome, AdapterError> {
                let mut sent = 0;
                for _ in 0..100 {
                    if req.stop.is_set() {
                        break;
                    }
                    on_delta("tok ");
                    sent += 1;
                }
                // An engine that ran to the end would report 100.
                Ok(crate::adapter::ServeOutcome { tokens: if sent == 100 { 100 } else { 0 }, done: sent == 100, ..Default::default() })
            }
        }
        let mut req = req("p");
        req.max_tokens = Some(1000);
        let mut frames = Vec::new();
        let summary = handle_serve_request_within(req, &Long, &mut |c| frames.push(c.to_vec()), &mut |sent| sent < 7);
        assert!(summary.ok);
        let deltas = frames.iter().filter(|f| matches!(ServeChunk::decode(f), Ok(ServeChunk::Delta(_)))).count();
        assert_eq!(deltas, 7, "the engine stopped once the budget said so");
        assert_eq!(summary.tokens, 7, "billed for what was sent");
        // A consumer that read the stream up to just before Done is owed the same count as the
        // stopped serve reports (rule 4), and with the Done frame, exactly what it says.
        let mut before_done = StreamTally::default();
        let mut all = StreamTally::default();
        for f in &frames {
            if matches!(ServeChunk::decode(f), Ok(ServeChunk::Delta(_))) {
                before_done.push_encoded(f);
            }
            all.push_encoded(f);
        }
        assert_eq!(before_done.billable(Some(1000)), 7);
        assert_eq!(all.billable(Some(1000)), summary.tokens);
    }

    /// Tool-call template overhead measured per model family (review B4): the tokens of the
    /// assistant's tool-call turn, as the family's own chat template renders it (end-of-turn
    /// included), beyond the tokens of the call's name and arguments. Measured with each family's
    /// tokenizer and template (transformers `apply_chat_template`) on three calls; the worst is kept.
    /// Families not yet measured: Mistral, Gemma 3, DeepSeek, gpt-oss.
    const MEASURED_TOOL_CALL_OVERHEAD: &[(&str, u64)] = &[
        ("Qwen3 0.6B/4B (empty think block included)", 17),
        ("Llama 3.2 1B", 9),
    ];

    #[test]
    fn the_tool_call_allowance_covers_every_measured_template() {
        for (family, overhead) in MEASURED_TOOL_CALL_OVERHEAD {
            assert!(TOOL_CALL_TOKEN_ALLOWANCE >= overhead + 4, "{family}: overhead {overhead}");
        }
    }

    #[test]
    fn a_partial_stream_is_counted_by_fragment_within_the_text_and_request_bounds() {
        let mut t = StreamTally::default();
        assert_eq!(t.billable(None), 0);
        for _ in 0..40 {
            t.push_delta("ab");
        }
        assert_eq!(t.billable(None), 40, "one per fragment");
        assert_eq!(t.billable(Some(25)), 25, "never past max_tokens");
        // Fragments with no text are bounded by the text measure (+ the slack), as any count is.
        let mut empty = StreamTally::default();
        for _ in 0..500 {
            empty.push_delta("");
        }
        assert_eq!(empty.billable(None), BILLABLE_TOKEN_SLACK);
        // Counting encoded frames is the same as counting received fragments; tool calls and errors
        // add nothing before Done.
        let mut enc = StreamTally::default();
        for _ in 0..40 {
            enc.push_encoded(&ServeChunk::Delta("ab".into()).encode());
        }
        enc.push_encoded(&ServeChunk::ToolCalls(vec![]).encode());
        enc.push_encoded(&ServeChunk::Error("engine died".into()).encode());
        assert_eq!(enc, t);
        enc.push_encoded(&ServeChunk::Done { tokens: 33, metrics: Default::default() }.encode());
        assert_eq!(enc.billable(Some(40)), 33, "an honest Done count stands");
        assert_eq!(enc.billable(Some(25)), 25, "never past max_tokens");
    }

    /// Review round 4 C1: a `Done` frame can't bill more than the frames carried, so a provider's
    /// reply to a stream close can't make its consumer sign for an invented count.
    #[test]
    fn a_done_count_is_bounded_by_what_the_stream_carried() {
        let mut t = StreamTally::default();
        for _ in 0..10 {
            t.push_encoded(&ServeChunk::Delta("ab".into()).encode());
        }
        t.push_encoded(&ServeChunk::Done { tokens: 10_000_000, metrics: Default::default() }.encode());
        assert_eq!(t.billable(None), 20 + BILLABLE_TOKEN_SLACK, "bounded by the text it carried");
        assert_eq!(t.billable(Some(15)), 15);
        // Tool calls raise the bound by at most their units and the allowance.
        let call = ToolCall {
            id: "c".into(),
            kind: "function".into(),
            function: crate::adapter::ToolCallFunction { name: "f".into(), arguments: "{}".into() },
        };
        let mut with_calls = t;
        with_calls.push_encoded(&ServeChunk::ToolCalls(vec![call.clone()]).encode());
        assert_eq!(with_calls.billable(None), 20 + 3 + TOOL_CALL_TOKEN_ALLOWANCE + BILLABLE_TOKEN_SLACK);
        // The bound is never below the exact allowance for any offered set.
        let offered: std::collections::HashSet<String> = ["f".to_string()].into();
        assert!(tool_call_units_at_most(&[call.clone()]) >= tool_call_units(&[call], &offered));
    }

    #[test]
    fn only_a_provider_error_starting_with_the_refusal_is_one() {
        use crate::adapter::AdapterError;
        assert!(is_payment_refusal(&AdapterError::Http(format!("{PAYMENT_REFUSAL}settle first"))));
        // An error that merely mentions it (an engine's text, say) isn't one.
        assert!(!is_payment_refusal(&AdapterError::Http(format!("engine said: {PAYMENT_REFUSAL}x"))));
        assert!(!is_payment_refusal(&AdapterError::Parse(format!("{PAYMENT_REFUSAL}x"))));
    }

    #[test]
    fn the_billable_count_is_bounded_by_the_request_and_the_text() {
        // An honest count is billed as reported.
        assert_eq!(billable_tokens(40, Some(64), 150), 40);
        assert_eq!(billable_tokens(40, None, 150), 40);
        // A billion tokens claimed for a short reply is held to what the text could have taken…
        assert_eq!(billable_tokens(1_000_000_000, None, 30), 30 + BILLABLE_TOKEN_SLACK);
        // …and never past the request's own limit (an engine that runs one over is billed the cap).
        assert_eq!(billable_tokens(1_000_000_000, Some(8), 10_000), 8);
        assert_eq!(billable_tokens(65, Some(64), 10_000), 64);
        // A few textless tokens (end of turn) are allowed for.
        assert_eq!(billable_tokens(12, None, 10), 12);
        assert_eq!(billable_tokens(u64::MAX, None, u64::MAX), u64::MAX);
    }

    #[test]
    fn padding_with_whitespace_adds_nothing_to_the_measure() {
        let measure = |parts: &[&str]| {
            let mut t = DeliveredText::default();
            parts.iter().for_each(|p| t.push(p));
            t.units()
        };
        assert_eq!(measure(&["héllo"]), 6, "bytes of non-whitespace text");
        assert_eq!(measure(&["a b"]), 3);
        assert_eq!(measure(&["a", "               ", " b"]), 3, "one run of 16, across deltas");
        assert_eq!(measure(&["a", "                ", "\n\n\n\t", " b"]), 4, "a run of 21: two units");
        assert_eq!(measure(&["<think>", "\n\n\n\n\n\n", "</think>"]), 16);
        assert_eq!(measure(&[&"\n".repeat(4_000)]), 250, "a long run is still paid for");
        let call = |args: &str| ToolCall {
            id: "c".into(),
            kind: "function".into(),
            function: crate::adapter::ToolCallFunction { name: "ls".into(), arguments: args.into() },
        };
        let ls = offered_tool_names(&[serde_json::json!({"type": "function", "function": {"name": "ls"}})]);
        assert_eq!(tool_call_units(&[call(r#"{"p":"/"}"#)], &ls), 2 + 9 + TOOL_CALL_TOKEN_ALLOWANCE);
        assert_eq!(
            tool_call_units(&[call(r#"{"p":"/"}"#)], &ls),
            tool_call_units(&[call("{ \"p\" :                  \"/\" }")], &ls) - 5,
            "padding in the arguments counts little"
        );
        assert_eq!(
            tool_call_units(&[call(r#"{"a":1}"#), call(r#"{"a":2}"#)], &ls),
            2 * (2 + 7 + TOOL_CALL_TOKEN_ALLOWANCE)
        );
        // A call to a tool nobody offered, with arguments that aren't a JSON object, repeated, or past
        // the cap earns no allowance: made-up calls can't pad.
        let none = std::collections::HashSet::new();
        assert_eq!(tool_call_units(&[call("{}")], &none), 4);
        assert_eq!(tool_call_units(&[call("junk")], &ls), 6);
        assert_eq!(tool_call_units(&[call("[1]")], &ls), 5);
        assert_eq!(tool_call_units(&[call("{}"), call("{}")], &ls), 2 * 4 + TOOL_CALL_TOKEN_ALLOWANCE);
        let many: Vec<ToolCall> = (0..40).map(|i| call(&format!(r#"{{"i":{i}}}"#))).collect();
        let text: u64 = many.iter().map(|c| 2 + c.function.arguments.len() as u64).sum();
        assert_eq!(tool_call_units(&many, &ls), text + MAX_ALLOWED_TOOL_CALLS as u64 * TOOL_CALL_TOKEN_ALLOWANCE);
    }

    #[test]
    fn a_provider_reports_the_billable_count_the_consumer_recomputes() {
        // The engine counts 500 tokens (hidden reasoning, say) for 11 bytes of answer: the provider
        // reports and commits to the billable count, which is what the consumer computes from the
        // frames it receives — so the honest provider's receipt matches.
        let adapter = StubAdapter { deltas: vec!["Hello".into(), "   ".into(), "world".into()], tokens: 500, fail: None };
        let (frames, summary) = collect_chunks(&req("p").encode(), &adapter);
        let mut seen = DeliveredText::default();
        let mut reported = None;
        for f in &frames {
            match f {
                ServeChunk::Delta(d) => seen.push(d),
                ServeChunk::Done { tokens, .. } => reported = Some(*tokens),
                _ => {}
            }
        }
        let max_tokens = req("p").max_tokens;
        let consumer_signs = billable_tokens(reported.unwrap(), max_tokens, seen.units());
        assert_eq!(summary.tokens, 11 + BILLABLE_TOKEN_SLACK);
        assert_eq!(reported, Some(summary.tokens));
        assert_eq!(consumer_signs, summary.tokens);
    }

    #[test]
    fn a_zero_token_cap_is_refused_before_the_engine_runs() {
        let adapter = StubAdapter { deltas: vec!["never".into()], tokens: 5, fail: None };
        let mut r = req("p");
        r.max_tokens = Some(0);
        let (frames, summary) = collect_chunks(&r.encode(), &adapter);
        assert!(!summary.ok);
        assert_eq!(summary.tokens, 0);
        assert!(matches!(frames.as_slice(), [ServeChunk::Error(_)]));
    }

    #[test]
    fn serve_request_round_trips() {
        let r = req("12D3KooWConsumer");
        assert_eq!(ServeRequest::decode(&r.encode()).unwrap(), r);
    }

    #[test]
    fn serve_request_think_round_trips_and_is_back_compatible() {
        // The new `think` field round-trips…
        let mut r = req("12D3KooWConsumer");
        r.think = Some(false);
        assert_eq!(ServeRequest::decode(&r.encode()).unwrap().think, Some(false));
        // …and an OLD consumer that omits the key still decodes (defaults to None) — no wire break.
        let legacy = br#"{"reply_to":"c","model_ref":"m","messages":[],"nonce":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"#;
        let decoded = ServeRequest::decode(legacy).expect("legacy ServeRequest without `think` decodes");
        assert_eq!(decoded.think, None);
    }

    #[test]
    fn deterministic_request_pins_temp_zero_and_threads_the_seed() {
        use std::cell::Cell;
        // A capturing adapter that records the temperature + seed it was handed.
        struct CaptureAdapter {
            temperature: Cell<Option<f64>>,
            seed: Cell<Option<u64>>,
        }
        impl EngineAdapter for CaptureAdapter {
            fn engine_name(&self) -> &'static str {
                "capture"
            }
            fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
                Ok(vec![])
            }
            fn serve_stream(
                &self,
                request: &InferenceRequest,
                _on_delta: &mut dyn FnMut(&str),
            ) -> Result<ServeOutcome, AdapterError> {
                self.temperature.set(request.temperature);
                self.seed.set(request.seed);
                Ok(ServeOutcome { tokens: 1, done: true, engine: Default::default(), tool_calls: Vec::new() })
            }
        }

        // A deterministic request overrides even an explicit temperature and pins the seed.
        let mut r = req("c");
        r.temperature = Some(0.9);
        r.deterministic = Some(DeterministicParams { seed: 7 });
        let adapter = CaptureAdapter { temperature: Cell::new(None), seed: Cell::new(None) };
        let _ = handle_serve_request_parsed(r, &adapter, &mut |_| {});
        assert_eq!(adapter.temperature.get(), Some(0.0), "deterministic pins temp=0");
        assert_eq!(adapter.seed.get(), Some(7), "deterministic threads the seed");

        // A plain request keeps its own sampling and pins no seed.
        let mut r = req("c");
        r.temperature = Some(0.9);
        let adapter = CaptureAdapter { temperature: Cell::new(None), seed: Cell::new(None) };
        let _ = handle_serve_request_parsed(r, &adapter, &mut |_| {});
        assert_eq!(adapter.temperature.get(), Some(0.9));
        assert_eq!(adapter.seed.get(), None);
    }

    #[test]
    fn serve_request_deterministic_round_trips_and_is_back_compatible() {
        // The new `deterministic` field round-trips…
        let mut r = req("c");
        r.deterministic = Some(DeterministicParams { seed: 42 });
        assert_eq!(
            ServeRequest::decode(&r.encode()).unwrap().deterministic,
            Some(DeterministicParams { seed: 42 })
        );
        // …and an OLD consumer that omits the key still decodes (defaults to None) — no wire break.
        let legacy = br#"{"reply_to":"c","model_ref":"m","messages":[],"nonce":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}"#;
        let decoded = ServeRequest::decode(legacy).expect("legacy ServeRequest without `deterministic` decodes");
        assert_eq!(decoded.deterministic, None);
    }

    #[test]
    fn serve_chunk_round_trips_each_variant() {
        for c in [
            ServeChunk::Delta("héllo".into()), // multi-byte utf8
            ServeChunk::ToolCalls(vec![tool_call("get_weather"), tool_call("get_time")]),
            ServeChunk::Done {
                tokens: 4096,
                metrics: ServeMetrics {
                    engine: EngineMetrics { eval_count: 4096, eval_duration_ns: 1_234_567, ..Default::default() },
                    provider_serve_ns: 9_000,
                    billing: BILLING_VERSION,
                    stopped: true,
                },
            },
            ServeChunk::Error("boom".into()),
        ] {
            assert_eq!(ServeChunk::decode(&c.encode()).unwrap(), c);
        }
    }

    #[test]
    fn handles_request_streaming_deltas_then_done() {
        let adapter = StubAdapter {
            deltas: vec!["Hello".into(), ", ".into(), "world".into()],
            tokens: 9,
            fail: None,
        };
        let (frames, summary) = collect_chunks(&req("c").encode(), &adapter);
        // The metrics' provider_serve_ns is wall-timed, so check the deltas + Done.tokens
        // structurally rather than comparing the whole metrics struct.
        assert_eq!(frames.len(), 4);
        assert_eq!(frames[0], ServeChunk::Delta("Hello".into()));
        assert_eq!(frames[2], ServeChunk::Delta("world".into()));
        assert!(matches!(frames[3], ServeChunk::Done { tokens: 9, .. }));
        assert_eq!(summary.tokens, 9);
        assert!(summary.ok);
    }

    #[test]
    fn serve_emits_a_tool_calls_frame_before_done() {
        // An adapter that returns tool calls (no text) must produce exactly: ToolCalls → Done,
        // and the summary must carry the calls for the gateway to shape into the assistant turn.
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
                _request: &InferenceRequest,
                _on_delta: &mut dyn FnMut(&str),
            ) -> Result<ServeOutcome, AdapterError> {
                Ok(ServeOutcome {
                    tokens: 2,
                    done: true,
                    engine: Default::default(),
                    tool_calls: vec![tool_call("get_weather")],
                })
            }
        }
        let (frames, summary) = collect_chunks(&req("c").encode(), &ToolStub);
        assert_eq!(frames.len(), 2, "no text deltas → just ToolCalls then Done");
        assert!(matches!(&frames[0], ServeChunk::ToolCalls(tc) if tc.len() == 1));
        assert!(matches!(frames[1], ServeChunk::Done { tokens: 2, .. }));
        assert_eq!(summary.tool_calls.len(), 1);
        assert_eq!(summary.tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn malformed_request_yields_a_single_error_frame() {
        let adapter = StubAdapter { deltas: vec![], tokens: 0, fail: None };
        let (frames, summary) = collect_chunks(b"not json", &adapter);
        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], ServeChunk::Error(_)));
        assert!(!summary.ok);
    }

    #[test]
    fn a_failed_serve_is_billed_for_the_text_it_returned() {
        // Round 8 S8-2: counted from the text frames, as the consumer counts what reached it.
        let frames = vec![
            ServeChunk::Delta("ab".into()).encode(),
            ServeChunk::Delta("cd".into()).encode(),
            ServeChunk::Error("engine timed out".into()).encode(),
        ];
        assert_eq!(failed_serve_billable(&frames, None), 2);
        assert_eq!(failed_serve_billable(&frames, Some(1)), 1);
        let mut consumer = StreamTally::default();
        consumer.push_delta("ab");
        consumer.push_delta("cd");
        assert_eq!(consumer.billable(None), failed_serve_billable(&frames, None));
        assert_eq!(failed_serve_billable(&[ServeChunk::Error("refused".into()).encode()], None), 0);
    }

    #[test]
    fn frame_response_round_trips() {
        let chunks: Vec<Vec<u8>> = vec![
            ServeChunk::Delta("Hi".into()).encode(),
            ServeChunk::Delta(" there".into()).encode(),
            ServeChunk::Done { tokens: 2, metrics: ServeMetrics::default() }.encode(),
        ];
        let parsed = parse_response(&frame_response(&chunks)).unwrap();
        assert_eq!(
            parsed,
            vec![
                ServeChunk::Delta("Hi".into()),
                ServeChunk::Delta(" there".into()),
                ServeChunk::Done { tokens: 2, metrics: ServeMetrics::default() },
            ]
        );
        assert!(parse_response(b"\xff\xff\xff\xff").is_err()); // truncated chunk body
    }

    #[test]
    fn engine_failure_ends_the_stream_with_an_error_frame() {
        let adapter = StubAdapter {
            deltas: vec![],
            tokens: 0,
            fail: Some(AdapterError::Http("engine down".into())),
        };
        let (frames, summary) = collect_chunks(&req("c").encode(), &adapter);
        assert!(matches!(frames.last().unwrap(), ServeChunk::Error(_)));
        assert!(!summary.ok);
        assert_eq!(summary.tokens, 0);
    }

    #[test]
    fn fetch_response_roundtrips_each_variant() {
        // Ready carries the framed serve response verbatim, so the consumer parses it exactly
        // like a first-try round-trip.
        let framed = frame_response(&[
            ServeChunk::Delta("hi".into()).encode(),
            ServeChunk::Done { tokens: 3, metrics: ServeMetrics::default() }.encode(),
        ]);
        for r in [
            FetchResponse::Ready(framed.clone()),
            FetchResponse::Generating,
            FetchResponse::NotFound,
            FetchResponse::Forbidden,
        ] {
            assert_eq!(FetchResponse::decode(&r.encode()).unwrap(), r);
        }
        // The Ready payload really is the framed serve bytes.
        if let FetchResponse::Ready(bytes) = FetchResponse::decode(&FetchResponse::Ready(framed.clone()).encode()).unwrap() {
            assert_eq!(parse_response(&bytes).unwrap().len(), 2);
        } else {
            panic!("expected Ready");
        }
        // An empty frame is a decode error, not a silent misparse.
        assert!(FetchResponse::decode(&[]).is_err());
    }

    #[test]
    fn fetch_chunks_response_round_trips() {
        // Frame two deltas as the incremental slice; carry next_offset + done.
        let framed = frame_response(&[
            ServeChunk::Delta("he".into()).encode(),
            ServeChunk::Delta("llo".into()).encode(),
        ]);
        let r = FetchChunksResponse::Chunks { framed: framed.clone(), next_offset: 2, done: false };
        let back = FetchChunksResponse::decode(&r.encode()).unwrap();
        assert_eq!(back, r);
        // The carried `framed` decodes with the normal serve parser.
        if let FetchChunksResponse::Chunks { framed, next_offset, done } = back {
            assert_eq!((next_offset, done), (2, false));
            let chunks = parse_response(&framed).unwrap();
            assert_eq!(chunks.len(), 2);
        } else {
            panic!("expected Chunks");
        }
        // A terminal slice (done=true) + the sentinels round-trip.
        for r in [
            FetchChunksResponse::Chunks { framed: frame_response(&[ServeChunk::Done { tokens: 5, metrics: Default::default() }.encode()]), next_offset: 3, done: true },
            FetchChunksResponse::NotFound,
            FetchChunksResponse::Forbidden,
            FetchChunksResponse::Overloaded,
        ] {
            assert_eq!(FetchChunksResponse::decode(&r.encode()).unwrap(), r);
        }
        assert!(FetchChunksResponse::decode(&[]).is_err());
        assert!(FetchChunksResponse::decode(&[FC_CHUNKS, 0, 0]).is_err()); // short header
    }

    #[test]
    fn fetch_chunks_discriminator_cannot_collide_with_a_legacy_error_blob() {
        // A legacy provider answers the unknown SERVE_STREAM method byte with the fixed
        // `frame_response([Error("unsupported method")])`. That blob begins with the frame's
        // little-endian u32 length prefix — low byte 0x13 (= the 19-byte frame length), which is
        // far below the 0xF1..=0xF4 streaming discriminators. So the consumer can tell a real
        // streaming reply from that blob by structural decode alone.
        let legacy = frame_response(&[ServeChunk::Error("unsupported method".into()).encode()]);
        let lead = legacy.first().copied().unwrap();
        assert!(lead < 0xF1, "legacy blob leads with a length byte ({lead:#x}) below the discriminators");
        assert!(FetchChunksResponse::decode(&legacy).is_err(), "must not misdecode as a streaming reply");
        for tag in [FC_CHUNKS, FC_NOTFOUND, FC_FORBIDDEN, FC_OVERLOADED] {
            assert!(tag >= 0xF1);
        }
    }

    #[test]
    fn frame_is_terminal_flags_done_and_error_only() {
        assert!(ServeChunk::frame_is_terminal(&ServeChunk::Done { tokens: 1, metrics: Default::default() }.encode()));
        assert!(ServeChunk::frame_is_terminal(&ServeChunk::Error("x".into()).encode()));
        assert!(!ServeChunk::frame_is_terminal(&ServeChunk::Delta("hi".into()).encode()));
        assert!(!ServeChunk::frame_is_terminal(&ServeChunk::ToolCalls(vec![tool_call("f")]).encode()));
        assert!(!ServeChunk::frame_is_terminal(&[]));
    }

    /// F3 (audit): the serve decoders run on RAW, UNTRUSTED bytes — the network layer hands any
    /// inbound `request_response` body straight to the agent, which decodes it with these. They
    /// MUST return `Err`, never panic, on adversarial input. Restores the guarantee the deleted
    /// `network/tests/adversarial.rs::decoders_never_panic_on_malformed_input` gave the (removed)
    /// sharding parsers, for the surface that replaced them. Passing = no unwind escapes the loop.
    #[test]
    fn serve_decoders_never_panic_on_malformed_input() {
        // A corpus of hostile shapes: empty, every single-byte tag, truncated/oversized length
        // prefixes, valid-tag-then-truncated-body, short headers, and a few pseudo-random blobs.
        let mut corpus: Vec<Vec<u8>> = vec![
            vec![],
            vec![0x00],
            vec![0x01],
            vec![0x02],           // TAG_DONE with no u64 tokens
            vec![0x02, 1, 2, 3],  // TAG_DONE, truncated (<9)
            vec![0x04],           // TAG_TOOLCALLS with no JSON
            vec![0x04, b'{'],     // TAG_TOOLCALLS, truncated JSON
            vec![0xFF],
            vec![0xF1],           // FC_CHUNKS, no header
            vec![0xF1, 0, 0],     // FC_CHUNKS, short header (<6)
            vec![0xF1, 0, 0, 0, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF], // FC_CHUNKS ok header + truncated framed
            vec![0xF2],
            vec![0xF4, 9, 9, 9],  // FC_OVERLOADED with trailing garbage
            vec![0xFF, 0xFF, 0xFF, 0xFF],             // parse_response: claims a 4GB chunk
            vec![0x01, 0x00, 0x00, 0x00],             // parse_response: len=1, no body
            vec![0x00, 0x00, 0x00],                   // parse_response: truncated length prefix
            b"not json at all".to_vec(),
            b"{".to_vec(),
            vec![0x7b, 0xff, 0xfe, 0x00, 0x80],       // invalid utf8 tail
        ];
        // A handful of deterministic pseudo-random blobs (no rng in the test path).
        for seed in 0u16..64 {
            let n = (seed % 37) as usize;
            corpus.push((0..n).map(|i| (seed.wrapping_mul(31).wrapping_add(i as u16)) as u8).collect());
        }

        for bytes in &corpus {
            // Every decoder on every input — results ignored; the point is "no panic".
            let _ = ServeRequest::decode(bytes);
            let _ = ServeChunk::decode(bytes);
            let _ = FetchResponse::decode(bytes);
            let _ = FetchChunksResponse::decode(bytes);
            let _ = parse_response(bytes);
            let _ = ServeChunk::frame_is_terminal(bytes);
        }
    }
}
