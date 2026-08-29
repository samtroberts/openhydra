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
    let infer = InferenceRequest {
        model_ref: req.model_ref,
        messages: req.messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        tools: req.tools,
        think: req.think,
    };

    // Scope the delta sink so its &mut borrow of `send_chunk` ends before the terminal
    // Done/Error frame is sent. Time the whole serve so the consumer can separate network
    // RTT from provider-side processing.
    let serve_start = std::time::Instant::now();
    let outcome = {
        let mut on_delta = |d: &str| send_chunk(&ServeChunk::Delta(d.to_string()).encode());
        adapter.serve_stream(&infer, &mut on_delta)
    };
    let provider_serve_ns = serve_start.elapsed().as_nanos() as u64;

    match outcome {
        Ok(o) => {
            let metrics = ServeMetrics { engine: o.engine, provider_serve_ns };
            // Tool calls (terminal, structured) go in their own frame just before `Done`, so
            // the consumer sees text deltas → tool_calls → done in order.
            if !o.tool_calls.is_empty() {
                send_chunk(&ServeChunk::ToolCalls(o.tool_calls.clone()).encode());
            }
            send_chunk(&ServeChunk::Done { tokens: o.tokens, metrics }.encode());
            ServeSummary {
                tokens: o.tokens,
                ok: true,
                metrics,
                discover_ns: 0,
                proxy_roundtrip_ns: 0,
                tool_calls: o.tool_calls,
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
    fn serve_chunk_round_trips_each_variant() {
        for c in [
            ServeChunk::Delta("héllo".into()), // multi-byte utf8
            ServeChunk::ToolCalls(vec![tool_call("get_weather"), tool_call("get_time")]),
            ServeChunk::Done {
                tokens: 4096,
                metrics: ServeMetrics {
                    engine: EngineMetrics { eval_count: 4096, eval_duration_ns: 1_234_567, ..Default::default() },
                    provider_serve_ns: 9_000,
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
