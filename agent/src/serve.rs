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

use crate::adapter::{AdapterError, ChatMessage, EngineAdapter, InferenceRequest};

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

/// One frame the provider streams back to the consumer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServeChunk {
    /// A text fragment of the completion.
    Delta(String),
    /// End of stream + the completion token count (for the co-signed receipt).
    Done { tokens: u64 },
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
            ServeChunk::Done { tokens } => {
                let mut b = Vec::with_capacity(9);
                b.push(TAG_DONE);
                b.extend_from_slice(&tokens.to_le_bytes());
                b
            }
            ServeChunk::Error(s) => {
                let mut b = Vec::with_capacity(1 + s.len());
                b.push(TAG_ERROR);
                b.extend_from_slice(s.as_bytes());
                b
            }
        }
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, AdapterError> {
        match bytes.first() {
            Some(&TAG_DELTA) => Ok(ServeChunk::Delta(
                String::from_utf8_lossy(&bytes[1..]).into_owned(),
            )),
            Some(&TAG_DONE) if bytes.len() >= 9 => {
                let mut t = [0u8; 8];
                t.copy_from_slice(&bytes[1..9]);
                Ok(ServeChunk::Done { tokens: u64::from_le_bytes(t) })
            }
            Some(&TAG_ERROR) => Ok(ServeChunk::Error(
                String::from_utf8_lossy(&bytes[1..]).into_owned(),
            )),
            _ => Err(AdapterError::Parse("serve chunk: bad/short frame".into())),
        }
    }
}

/// Outcome of serving one request (returned to the wiring layer for the receipt).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServeSummary {
    /// Completion tokens generated (0 on a failed request).
    pub tokens: u64,
    /// Whether the stream completed cleanly (vs ending in an error frame).
    pub ok: bool,
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
            return ServeSummary { tokens: 0, ok: false };
        }
    };

    let infer = InferenceRequest {
        model_ref: req.model_ref,
        messages: req.messages,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
    };

    // Scope the delta sink so its &mut borrow of `send_chunk` ends before the terminal
    // Done/Error frame is sent.
    let outcome = {
        let mut on_delta = |d: &str| send_chunk(&ServeChunk::Delta(d.to_string()).encode());
        adapter.serve_stream(&infer, &mut on_delta)
    };

    match outcome {
        Ok(o) => {
            send_chunk(&ServeChunk::Done { tokens: o.tokens }.encode());
            ServeSummary { tokens: o.tokens, ok: true }
        }
        Err(e) => {
            send_chunk(&ServeChunk::Error(e.to_string()).encode());
            ServeSummary { tokens: 0, ok: false }
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
    use crate::adapter::{DetectedModel, ServeOutcome};

    fn req(reply_to: &str) -> ServeRequest {
        ServeRequest {
            reply_to: reply_to.into(),
            model_ref: "qwen2.5:7b".into(),
            messages: vec![ChatMessage { role: "user".into(), content: "hi".into() }],
            max_tokens: Some(64),
            temperature: None,
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
            Ok(ServeOutcome { tokens: self.tokens, done: true })
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
    fn serve_chunk_round_trips_each_variant() {
        for c in [
            ServeChunk::Delta("héllo".into()), // multi-byte utf8
            ServeChunk::Done { tokens: 4096 },
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
        assert_eq!(
            frames,
            vec![
                ServeChunk::Delta("Hello".into()),
                ServeChunk::Delta(", ".into()),
                ServeChunk::Delta("world".into()),
                ServeChunk::Done { tokens: 9 },
            ]
        );
        assert_eq!(summary, ServeSummary { tokens: 9, ok: true });
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
            ServeChunk::Done { tokens: 2 }.encode(),
        ];
        let parsed = parse_response(&frame_response(&chunks)).unwrap();
        assert_eq!(
            parsed,
            vec![
                ServeChunk::Delta("Hi".into()),
                ServeChunk::Delta(" there".into()),
                ServeChunk::Done { tokens: 2 },
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
}
