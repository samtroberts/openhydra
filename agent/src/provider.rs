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

use openhydra_network::handle::NetworkHandle;
use openhydra_network::types::PeerRecord;

use crate::adapter::{AdapterError, DetectedModel, EngineAdapter};
use crate::serve::{frame_response, handle_serve_request};

/// libp2p proxy method byte for an inference serve request (consumer → provider). Sits
/// alongside the existing peer method bytes (0x01 Forward … 0x07 Receipt).
pub const SERVE_REQUEST: u8 = 0x10;

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

/// An engine joined to the swarm: advertises its models and serves inbound requests.
pub struct Provider<A: EngineAdapter> {
    adapter: A,
    net: NetworkHandle,
    host: String,
    port: u16,
}

impl<A: EngineAdapter> Provider<A> {
    pub fn new(adapter: A, net: NetworkHandle) -> Self {
        Self { adapter, net, host: String::new(), port: 0 }
    }

    /// Set the advisory host/port advertised in records (routing is via libp2p regardless).
    pub fn with_address(mut self, host: impl Into<String>, port: u16) -> Self {
        self.host = host.into();
        self.port = port;
        self
    }

    /// Detect the engine's models and announce a record for each. Returns how many were
    /// announced.
    pub fn announce_models(&self) -> Result<usize, AdapterError> {
        let models = self.adapter.detect_models()?;
        for model in &models {
            let record = build_peer_record(
                model,
                self.net.openhydra_peer_id(),
                self.net.libp2p_peer_id(),
                self.net.public_key_hex(),
                &self.host,
                self.port,
            );
            self.net
                .announce(record)
                .map_err(|e| AdapterError::Http(format!("announce: {e}")))?;
        }
        Ok(models.len())
    }

    /// Blocking serve loop: poll inbound requests, serve them, reply. Runs until the
    /// process exits (call on a dedicated thread). Polls in `timeout` slices so it stays
    /// responsive; a failed reply is logged-by-return and the loop continues.
    pub fn run_inbound(&self, timeout: std::time::Duration) -> ! {
        loop {
            if let Some((request_id, data)) = self.net.poll_inbound(timeout) {
                let response = handle_serve_inbound(&data, &self.adapter);
                // Best-effort reply; if the swarm is gone the next poll will error too.
                let _ = self.net.respond(request_id, response);
            }
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
            Ok(ServeOutcome { tokens: 5, done: true })
        }
    }

    fn serve_request_bytes() -> Vec<u8> {
        let req = ServeRequest {
            reply_to: "12D3KooWConsumer".into(),
            model_ref: "qwen2.5:7b".into(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
        };
        let mut data = vec![SERVE_REQUEST];
        data.extend_from_slice(&req.encode());
        data
    }

    #[test]
    fn serves_an_inbound_request_into_a_framed_completion() {
        let response = handle_serve_inbound(&serve_request_bytes(), &StubAdapter);
        let chunks = parse_response(&response).unwrap();
        assert_eq!(
            chunks,
            vec![
                ServeChunk::Delta("Hello".into()),
                ServeChunk::Delta(" world".into()),
                ServeChunk::Done { tokens: 5 },
            ]
        );
    }

    #[test]
    fn rejects_unknown_method_byte() {
        let response = handle_serve_inbound(&[0xEE, 1, 2, 3], &StubAdapter);
        let chunks = parse_response(&response).unwrap();
        assert!(matches!(chunks.as_slice(), [ServeChunk::Error(_)]));
    }
}
