//! CP-2: Rust Proxy Dispatcher — inbound message routing and demux.
//!
//! Replaces the Python `_proxy_handler_loop` (peer/server.py lines 142–383)
//! and `_coordinator_proxy_handler_loop` (lines 386–566).
//!
//! The dispatcher's job:
//!   1. Extract the 1-byte method prefix (0x01–0x06, or legacy no-prefix)
//!   2. Detect wire format (ForwardMsg OHV2 magic vs legacy protobuf)
//!   3. For ForwardMsg: parse CBOR header (< 10μs)
//!   4. Return a `DispatchAction` describing what the event loop should do
//!
//! The event loop acts on the `DispatchAction`:
//!   - `ForwardToWorker` → send to Python worker via IPC bridge, await response
//!   - `ForwardToWorkerAsync` → ACK immediately, then IPC in background
//!   - `PushResultBlocking` → route to ring session / coordinator
//!   - `PushResultAsync` → ACK immediately, then route
//!   - `PingResponse` → respond inline (no Python round-trip)
//!   - `StatusResponse` → respond inline
//!   - `LegacyFallthrough` → push to SharedProxyQueue for Python handling
//!   - `UnsupportedMethod` → error response (coordinator mode only)
//!   - `ParseError` → error response

use crate::forward_msg;
use crate::ipc_codec::{IpcForwardHeader, IpcResponseHeader};

// ── Method prefix constants ───────────────────────────────────────────
// Must match Python's PROXY_METHOD_* in peer/server.py lines 133–139.

/// ForwardRequest → call Forward(), block for response.
pub const METHOD_FORWARD: u8 = 0x01;

/// ForwardResponse → call PushResult(), block for response.
pub const METHOD_PUSH_RESULT: u8 = 0x02;

/// ForwardRequest → ACK immediately, Forward() in background.
pub const METHOD_FIRE_FORGET: u8 = 0x03;

/// ForwardResponse → ACK immediately, PushResult() in background.
pub const METHOD_FIRE_FORGET_RESULT: u8 = 0x04;

/// PingRequest → call Ping(), block for response.
pub const METHOD_PING: u8 = 0x05;

/// PeerStatusRequest → call GetPeerStatus(), block for response.
pub const METHOD_GET_STATUS: u8 = 0x06;

// ── Types ─────────────────────────────────────────────────────────────

/// Recognised proxy method after prefix extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyMethod {
    /// 0x01: blocking forward.
    Forward,
    /// 0x02: blocking push-result.
    PushResult,
    /// 0x03: fire-and-forget forward (ACK immediately).
    ForwardFireForget,
    /// 0x04: fire-and-forget push-result (ACK immediately).
    PushResultFireForget,
    /// 0x05: blocking ping.
    Ping,
    /// 0x06: blocking peer-status query.
    GetPeerStatus,
    /// No recognised prefix → legacy Forward (backward compat).
    Legacy,
}

impl ProxyMethod {
    /// Classify a method prefix byte.
    #[inline]
    pub fn from_prefix(byte: u8) -> Option<Self> {
        match byte {
            METHOD_FORWARD => Some(Self::Forward),
            METHOD_PUSH_RESULT => Some(Self::PushResult),
            METHOD_FIRE_FORGET => Some(Self::ForwardFireForget),
            METHOD_FIRE_FORGET_RESULT => Some(Self::PushResultFireForget),
            METHOD_PING => Some(Self::Ping),
            METHOD_GET_STATUS => Some(Self::GetPeerStatus),
            _ => None,
        }
    }

    /// Whether this method requires an immediate ACK (fire-and-forget).
    #[inline]
    pub fn is_fire_and_forget(&self) -> bool {
        matches!(
            self,
            ProxyMethod::ForwardFireForget | ProxyMethod::PushResultFireForget
        )
    }

    /// Whether this method carries a Forward request (vs a PushResult response).
    #[inline]
    pub fn is_forward(&self) -> bool {
        matches!(
            self,
            ProxyMethod::Forward | ProxyMethod::ForwardFireForget | ProxyMethod::Legacy
        )
    }

    /// The single-byte ACK for fire-and-forget methods.
    pub fn ack_byte(&self) -> Option<u8> {
        match self {
            ProxyMethod::ForwardFireForget => Some(METHOD_FIRE_FORGET),
            ProxyMethod::PushResultFireForget => Some(METHOD_FIRE_FORGET_RESULT),
            _ => None,
        }
    }
}

/// Extract method prefix and payload from raw proxy data.
///
/// Returns `(ProxyMethod, payload)` where `payload` is everything
/// after the 1-byte prefix. For `Legacy` (no recognised prefix),
/// `payload` is the entire input (the first byte is part of the
/// protobuf message).
#[inline]
pub fn extract_method(data: &[u8]) -> (ProxyMethod, &[u8]) {
    if data.is_empty() {
        return (ProxyMethod::Legacy, data);
    }
    match ProxyMethod::from_prefix(data[0]) {
        Some(method) => (method, &data[1..]),
        None => (ProxyMethod::Legacy, data),
    }
}

/// Dispatcher mode — determines which methods are accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchMode {
    /// Full peer with local model shard — handles all 6 methods.
    Peer,
    /// Pure coordinator (no local model) — only PushResult (0x02, 0x04).
    Coordinator,
}

/// Parsed forward data (header + activation), ready for IPC dispatch.
#[derive(Debug, Clone)]
pub struct ParsedForward {
    pub header: IpcForwardHeader,
    pub activation: Vec<u8>,
}

/// Parsed push-result data (response header + activation).
#[derive(Debug, Clone)]
pub struct ParsedPushResult {
    pub header: IpcResponseHeader,
    pub activation: Vec<u8>,
}

/// Routing decision from the dispatcher.
///
/// All data is fully owned (no lifetimes) so the event loop can
/// act on it asynchronously without borrowing the raw input.
#[derive(Debug)]
pub enum DispatchAction {
    /// Forward to local Python worker via IPC bridge, wait for response.
    ///
    /// Event loop: IPC send → await response → encode ForwardMsg response → respond.
    ForwardToWorker(ParsedForward),

    /// Forward fire-and-forget: ACK immediately, then IPC in background.
    ///
    /// Event loop: respond with `ack` immediately → spawn IPC send (no response needed).
    ForwardToWorkerAsync {
        ack: Vec<u8>,
        forward: ParsedForward,
    },

    /// PushResult blocking: route to ring session / coordinator.
    ///
    /// Event loop: process PushResult (head-sample, ring token emit) → respond.
    PushResultBlocking(ParsedPushResult),

    /// PushResult fire-and-forget: ACK immediately, then route.
    ///
    /// Event loop: respond with `ack` immediately → process PushResult in background.
    PushResultAsync {
        ack: Vec<u8>,
        push_result: ParsedPushResult,
    },

    /// Respond to ping inline (no Python round-trip needed).
    ///
    /// The response payload is a prost-encoded `PingResponse` with
    /// the method prefix byte prepended.
    PingResponse(Vec<u8>),

    /// Respond to status request inline (no Python round-trip needed).
    ///
    /// The response payload is a prost-encoded `PeerStatusResponse`
    /// with the method prefix byte prepended.
    StatusResponse(Vec<u8>),

    /// Legacy protobuf format — fall through to SharedProxyQueue.
    ///
    /// The event loop pushes the *original* raw data (with prefix byte)
    /// to the SharedProxyQueue for Python to handle unchanged.
    LegacyFallthrough,

    /// Unsupported method for current dispatcher mode.
    ///
    /// Contains a pre-built error response (method prefix + PushAck).
    UnsupportedMethod {
        response: Vec<u8>,
        reason: String,
    },

    /// Failed to parse the message.
    ParseError(String),
}

impl DispatchAction {
    /// Whether this action requires an immediate response (ACK or error)
    /// before any async work.
    pub fn needs_immediate_response(&self) -> bool {
        matches!(
            self,
            DispatchAction::ForwardToWorkerAsync { .. }
                | DispatchAction::PushResultAsync { .. }
                | DispatchAction::UnsupportedMethod { .. }
                | DispatchAction::ParseError(_)
        )
    }
}

// ── Cached peer state for inline responses ────────────────────────────

/// Peer status snapshot — cached by the Dispatcher for inline
/// Ping/GetPeerStatus responses, avoiding a Python round-trip.
#[derive(Debug, Clone)]
pub struct PeerStatusCache {
    pub peer_id: String,
    pub model_id: String,
    pub shard_index: u32,
    pub total_shards: u32,
    pub load_pct: f64,
    pub healthy: bool,
    pub daemon_mode: String,
}

impl Default for PeerStatusCache {
    fn default() -> Self {
        Self {
            peer_id: String::new(),
            model_id: String::new(),
            shard_index: 0,
            total_shards: 1,
            load_pct: 0.0,
            healthy: true,
            daemon_mode: "peer".into(),
        }
    }
}

// ── Dispatcher ────────────────────────────────────────────────────────

/// Inbound proxy message dispatcher.
///
/// Parses method prefix, detects wire format, decodes headers, and
/// returns a `DispatchAction` describing what the event loop should do.
///
/// Thread-safety: the Dispatcher is intended to be called from the
/// single-threaded tokio event loop. It does NOT hold async state.
pub struct Dispatcher {
    mode: DispatchMode,
    status_cache: PeerStatusCache,
}

impl Dispatcher {
    /// Create a new dispatcher in the given mode.
    pub fn new(mode: DispatchMode) -> Self {
        Self {
            mode,
            status_cache: PeerStatusCache::default(),
        }
    }

    /// Update the cached peer status (called periodically from Python).
    pub fn update_status(&mut self, status: PeerStatusCache) {
        self.status_cache = status;
    }

    /// Dispatch an inbound proxy message.
    ///
    /// `data` is the raw bytes from `request_response::Message::Request`.
    /// Returns a routing decision for the event loop to execute.
    pub fn dispatch(&self, data: &[u8]) -> DispatchAction {
        let (method, payload) = extract_method(data);

        // ── Coordinator mode: reject non-PushResult methods ───────
        if self.mode == DispatchMode::Coordinator {
            return self.dispatch_coordinator(method, payload, data);
        }

        // ── Peer mode: handle all 6 methods ──────────────────────
        match method {
            ProxyMethod::Forward => self.dispatch_forward_blocking(payload),
            ProxyMethod::ForwardFireForget => self.dispatch_forward_async(payload),
            ProxyMethod::PushResult => self.dispatch_push_result_blocking(payload),
            ProxyMethod::PushResultFireForget => self.dispatch_push_result_async(payload),
            ProxyMethod::Ping => self.dispatch_ping(payload),
            ProxyMethod::GetPeerStatus => self.dispatch_get_status(payload),
            ProxyMethod::Legacy => self.dispatch_legacy(data),
        }
    }

    // ── Peer-mode dispatch arms ───────────────────────────────────────

    fn dispatch_forward_blocking(&self, payload: &[u8]) -> DispatchAction {
        if !forward_msg::is_forward_msg(payload) {
            // Legacy protobuf Forward — fall through to Python.
            return DispatchAction::LegacyFallthrough;
        }

        match forward_msg::decode(payload) {
            Ok(decoded) => DispatchAction::ForwardToWorker(ParsedForward {
                header: decoded.header,
                activation: decoded.activation.to_vec(),
            }),
            Err(e) => DispatchAction::ParseError(format!("ForwardMsg decode: {e}")),
        }
    }

    fn dispatch_forward_async(&self, payload: &[u8]) -> DispatchAction {
        if !forward_msg::is_forward_msg(payload) {
            return DispatchAction::LegacyFallthrough;
        }

        match forward_msg::decode(payload) {
            Ok(decoded) => DispatchAction::ForwardToWorkerAsync {
                ack: vec![METHOD_FIRE_FORGET],
                forward: ParsedForward {
                    header: decoded.header,
                    activation: decoded.activation.to_vec(),
                },
            },
            Err(e) => DispatchAction::ParseError(format!("ForwardMsg decode: {e}")),
        }
    }

    fn dispatch_push_result_blocking(&self, payload: &[u8]) -> DispatchAction {
        if !forward_msg::is_forward_msg(payload) {
            return DispatchAction::LegacyFallthrough;
        }

        match forward_msg::decode_response(payload) {
            Ok((header, activation)) => {
                DispatchAction::PushResultBlocking(ParsedPushResult {
                    header,
                    activation: activation.to_vec(),
                })
            }
            Err(e) => DispatchAction::ParseError(format!("PushResult decode: {e}")),
        }
    }

    fn dispatch_push_result_async(&self, payload: &[u8]) -> DispatchAction {
        if !forward_msg::is_forward_msg(payload) {
            return DispatchAction::LegacyFallthrough;
        }

        match forward_msg::decode_response(payload) {
            Ok((header, activation)) => DispatchAction::PushResultAsync {
                ack: vec![METHOD_FIRE_FORGET_RESULT],
                push_result: ParsedPushResult {
                    header,
                    activation: activation.to_vec(),
                },
            },
            Err(e) => DispatchAction::ParseError(format!("PushResult decode: {e}")),
        }
    }

    fn dispatch_ping(&self, _payload: &[u8]) -> DispatchAction {
        // Respond inline with cached peer status — no Python round-trip.
        // Build a prost PingResponse and prepend METHOD_PING prefix.
        use prost::Message;
        let ping_resp = crate::proto::PingResponse {
            peer_id: self.status_cache.peer_id.clone(),
            ok: self.status_cache.healthy,
            load_pct: self.status_cache.load_pct,
            daemon_mode: self.status_cache.daemon_mode.clone(),
            geo_nonce_signature: String::new(),
        };
        let mut buf = vec![METHOD_PING];
        buf.extend(ping_resp.encode_to_vec());
        DispatchAction::PingResponse(buf)
    }

    fn dispatch_get_status(&self, _payload: &[u8]) -> DispatchAction {
        // Respond inline with cached peer status.
        use prost::Message;
        let status_resp = crate::proto::PeerStatusResponse {
            peer_id: self.status_cache.peer_id.clone(),
            model_id: self.status_cache.model_id.clone(),
            shard_index: self.status_cache.shard_index,
            total_shards: self.status_cache.total_shards,
            load_pct: self.status_cache.load_pct,
            healthy: self.status_cache.healthy,
            daemon_mode: self.status_cache.daemon_mode.clone(),
            dp_noise_configured_variance: 0.0,
            dp_noise_payloads: 0,
            dp_noise_observed_variance_ema: 0.0,
            dp_noise_last_audit_tag: String::new(),
        };
        let mut buf = vec![METHOD_GET_STATUS];
        buf.extend(status_resp.encode_to_vec());
        DispatchAction::StatusResponse(buf)
    }

    fn dispatch_legacy(&self, _data: &[u8]) -> DispatchAction {
        // No recognised method prefix — legacy protobuf Forward.
        // Fall through to SharedProxyQueue for Python handling.
        DispatchAction::LegacyFallthrough
    }

    // ── Coordinator-mode dispatch ─────────────────────────────────────

    fn dispatch_coordinator(
        &self,
        method: ProxyMethod,
        payload: &[u8],
        _data: &[u8],
    ) -> DispatchAction {
        match method {
            ProxyMethod::PushResult => self.dispatch_push_result_blocking(payload),
            ProxyMethod::PushResultFireForget => self.dispatch_push_result_async(payload),
            _ => {
                // Coordinator doesn't serve Forward/Ping/GetStatus.
                // Build a PushAck error so the sender unblocks.
                use prost::Message;
                let err = crate::proto::PushAck {
                    request_id: String::new(),
                    ok: false,
                    error: "pure_coordinator_unsupported_proxy_method".into(),
                };
                let mut buf = vec![METHOD_PUSH_RESULT];
                buf.extend(err.encode_to_vec());
                DispatchAction::UnsupportedMethod {
                    response: buf,
                    reason: format!(
                        "coordinator mode does not handle method {:?}",
                        method
                    ),
                }
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_msg::{self, MsgType};
    use crate::ipc_codec::{ActivationDtype, IpcForwardHeader, IpcResponseHeader, IpcStatus};

    // ── Helper: build a ForwardMsg-encoded payload ────────────────────

    fn make_forward_msg(
        msg_type: MsgType,
        request_id: &str,
        activation: &[u8],
    ) -> Vec<u8> {
        let header = IpcForwardHeader {
            request_id: request_id.into(),
            stage_index: 1,
            total_stages: 4,
            push_mode: true,
            shard_layer_start: 8,
            shard_layer_end: 16,
            shard_total_layers: 32,
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, activation.len() as u32 / 4],
            ..Default::default()
        };
        forward_msg::encode(msg_type, &header, activation).unwrap()
    }

    fn make_response_msg(
        request_id: &str,
        activation: &[u8],
    ) -> Vec<u8> {
        let header = IpcResponseHeader {
            status: IpcStatus::Ok,
            request_id: request_id.into(),
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, activation.len() as u32 / 4],
            ..Default::default()
        };
        forward_msg::encode_response(&header, activation).unwrap()
    }

    fn test_activation() -> Vec<u8> {
        vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    // ── Method prefix extraction ──────────────────────────────────────

    #[test]
    fn test_extract_method_forward() {
        let data = [METHOD_FORWARD, 0x0A, 0x10];
        let (method, payload) = extract_method(&data);
        assert_eq!(method, ProxyMethod::Forward);
        assert_eq!(payload, &[0x0A, 0x10]);
    }

    #[test]
    fn test_extract_method_push_result() {
        let data = [METHOD_PUSH_RESULT, 0xFF];
        let (method, payload) = extract_method(&data);
        assert_eq!(method, ProxyMethod::PushResult);
        assert_eq!(payload, &[0xFF]);
    }

    #[test]
    fn test_extract_method_fire_forget() {
        let data = [METHOD_FIRE_FORGET, 0x01, 0x02];
        let (method, _) = extract_method(&data);
        assert_eq!(method, ProxyMethod::ForwardFireForget);
    }

    #[test]
    fn test_extract_method_fire_forget_result() {
        let data = [METHOD_FIRE_FORGET_RESULT, 0x01];
        let (method, _) = extract_method(&data);
        assert_eq!(method, ProxyMethod::PushResultFireForget);
    }

    #[test]
    fn test_extract_method_ping() {
        let data = [METHOD_PING, 0x08, 0x01];
        let (method, payload) = extract_method(&data);
        assert_eq!(method, ProxyMethod::Ping);
        assert_eq!(payload, &[0x08, 0x01]);
    }

    #[test]
    fn test_extract_method_get_status() {
        let data = [METHOD_GET_STATUS];
        let (method, payload) = extract_method(&data);
        assert_eq!(method, ProxyMethod::GetPeerStatus);
        assert!(payload.is_empty());
    }

    #[test]
    fn test_extract_method_legacy_no_prefix() {
        // Protobuf messages start with field tags (0x0A = field 1, type 2).
        let data = [0x0A, 0x10, 0x08, 0x01];
        let (method, payload) = extract_method(&data);
        assert_eq!(method, ProxyMethod::Legacy);
        // For legacy, payload == entire data (no prefix stripped).
        assert_eq!(payload, &data);
    }

    #[test]
    fn test_extract_method_empty() {
        let (method, payload) = extract_method(&[]);
        assert_eq!(method, ProxyMethod::Legacy);
        assert!(payload.is_empty());
    }

    #[test]
    fn test_extract_method_unknown_prefix() {
        // 0x07 is not a valid method prefix → treated as legacy.
        let data = [0x07, 0x0A];
        let (method, payload) = extract_method(&data);
        assert_eq!(method, ProxyMethod::Legacy);
        assert_eq!(payload, &data);
    }

    // ── ProxyMethod properties ────────────────────────────────────────

    #[test]
    fn test_method_fire_and_forget_flag() {
        assert!(!ProxyMethod::Forward.is_fire_and_forget());
        assert!(!ProxyMethod::PushResult.is_fire_and_forget());
        assert!(ProxyMethod::ForwardFireForget.is_fire_and_forget());
        assert!(ProxyMethod::PushResultFireForget.is_fire_and_forget());
        assert!(!ProxyMethod::Ping.is_fire_and_forget());
        assert!(!ProxyMethod::GetPeerStatus.is_fire_and_forget());
        assert!(!ProxyMethod::Legacy.is_fire_and_forget());
    }

    #[test]
    fn test_method_is_forward_flag() {
        assert!(ProxyMethod::Forward.is_forward());
        assert!(ProxyMethod::ForwardFireForget.is_forward());
        assert!(ProxyMethod::Legacy.is_forward());
        assert!(!ProxyMethod::PushResult.is_forward());
        assert!(!ProxyMethod::PushResultFireForget.is_forward());
        assert!(!ProxyMethod::Ping.is_forward());
        assert!(!ProxyMethod::GetPeerStatus.is_forward());
    }

    #[test]
    fn test_method_ack_byte() {
        assert_eq!(ProxyMethod::ForwardFireForget.ack_byte(), Some(METHOD_FIRE_FORGET));
        assert_eq!(ProxyMethod::PushResultFireForget.ack_byte(), Some(METHOD_FIRE_FORGET_RESULT));
        assert_eq!(ProxyMethod::Forward.ack_byte(), None);
        assert_eq!(ProxyMethod::PushResult.ack_byte(), None);
        assert_eq!(ProxyMethod::Ping.ack_byte(), None);
    }

    // ── Peer-mode dispatch: Forward (blocking) ────────────────────────

    #[test]
    fn test_dispatch_forward_blocking_forward_msg() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let activation = test_activation();
        let fwd_msg = make_forward_msg(MsgType::Forward, "req-001", &activation);

        // Prepend method prefix.
        let mut data = vec![METHOD_FORWARD];
        data.extend(&fwd_msg);

        match dispatcher.dispatch(&data) {
            DispatchAction::ForwardToWorker(parsed) => {
                assert_eq!(parsed.header.request_id, "req-001");
                assert_eq!(parsed.header.stage_index, 1);
                assert_eq!(parsed.header.total_stages, 4);
                assert!(parsed.header.push_mode);
                assert_eq!(parsed.header.shard_layer_start, 8);
                assert_eq!(parsed.activation, activation);
            }
            other => panic!("expected ForwardToWorker, got {:?}", other),
        }
    }

    #[test]
    fn test_dispatch_forward_blocking_legacy() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        // Legacy protobuf: method prefix + non-OHV2 payload.
        let data = [METHOD_FORWARD, 0x0A, 0x10, 0x08, 0x01];
        match dispatcher.dispatch(&data) {
            DispatchAction::LegacyFallthrough => {} // expected
            other => panic!("expected LegacyFallthrough, got {:?}", other),
        }
    }

    // ── Peer-mode dispatch: Forward (fire-and-forget) ─────────────────

    #[test]
    fn test_dispatch_forward_fire_forget() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let activation = test_activation();
        let fwd_msg = make_forward_msg(MsgType::Forward, "req-ff", &activation);

        let mut data = vec![METHOD_FIRE_FORGET];
        data.extend(&fwd_msg);

        match dispatcher.dispatch(&data) {
            DispatchAction::ForwardToWorkerAsync { ack, forward } => {
                assert_eq!(ack, vec![METHOD_FIRE_FORGET]);
                assert_eq!(forward.header.request_id, "req-ff");
                assert_eq!(forward.activation, activation);
            }
            other => panic!("expected ForwardToWorkerAsync, got {:?}", other),
        }
    }

    #[test]
    fn test_dispatch_forward_fire_forget_legacy() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let data = [METHOD_FIRE_FORGET, 0x0A, 0x10];
        match dispatcher.dispatch(&data) {
            DispatchAction::LegacyFallthrough => {}
            other => panic!("expected LegacyFallthrough, got {:?}", other),
        }
    }

    // ── Peer-mode dispatch: PushResult (blocking) ─────────────────────

    #[test]
    fn test_dispatch_push_result_blocking() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let activation = test_activation();
        let resp_msg = make_response_msg("push-001", &activation);

        let mut data = vec![METHOD_PUSH_RESULT];
        data.extend(&resp_msg);

        match dispatcher.dispatch(&data) {
            DispatchAction::PushResultBlocking(parsed) => {
                assert_eq!(parsed.header.request_id, "push-001");
                assert_eq!(parsed.header.status, IpcStatus::Ok);
                assert_eq!(parsed.activation, activation);
            }
            other => panic!("expected PushResultBlocking, got {:?}", other),
        }
    }

    #[test]
    fn test_dispatch_push_result_blocking_legacy() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let data = [METHOD_PUSH_RESULT, 0x0A, 0x04, 0x08, 0x01];
        match dispatcher.dispatch(&data) {
            DispatchAction::LegacyFallthrough => {}
            other => panic!("expected LegacyFallthrough, got {:?}", other),
        }
    }

    // ── Peer-mode dispatch: PushResult (fire-and-forget) ──────────────

    #[test]
    fn test_dispatch_push_result_fire_forget() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let activation = test_activation();
        let resp_msg = make_response_msg("push-ff", &activation);

        let mut data = vec![METHOD_FIRE_FORGET_RESULT];
        data.extend(&resp_msg);

        match dispatcher.dispatch(&data) {
            DispatchAction::PushResultAsync { ack, push_result } => {
                assert_eq!(ack, vec![METHOD_FIRE_FORGET_RESULT]);
                assert_eq!(push_result.header.request_id, "push-ff");
                assert_eq!(push_result.activation, activation);
            }
            other => panic!("expected PushResultAsync, got {:?}", other),
        }
    }

    // ── Peer-mode dispatch: Ping ──────────────────────────────────────

    #[test]
    fn test_dispatch_ping_inline() {
        let mut dispatcher = Dispatcher::new(DispatchMode::Peer);
        dispatcher.update_status(PeerStatusCache {
            peer_id: "peer-abc".into(),
            healthy: true,
            load_pct: 0.42,
            daemon_mode: "peer".into(),
            ..Default::default()
        });

        // Ping prefix + dummy protobuf PingRequest body.
        let data = [METHOD_PING, 0x08, 0x01]; // field 1: sent_unix_ms

        match dispatcher.dispatch(&data) {
            DispatchAction::PingResponse(resp) => {
                assert_eq!(resp[0], METHOD_PING);
                // Decode the prost PingResponse from the rest.
                use prost::Message;
                let ping_resp =
                    crate::proto::PingResponse::decode(&resp[1..]).unwrap();
                assert_eq!(ping_resp.peer_id, "peer-abc");
                assert!(ping_resp.ok);
                assert!((ping_resp.load_pct - 0.42).abs() < 1e-6);
                assert_eq!(ping_resp.daemon_mode, "peer");
            }
            other => panic!("expected PingResponse, got {:?}", other),
        }
    }

    // ── Peer-mode dispatch: GetPeerStatus ─────────────────────────────

    #[test]
    fn test_dispatch_get_status_inline() {
        let mut dispatcher = Dispatcher::new(DispatchMode::Peer);
        dispatcher.update_status(PeerStatusCache {
            peer_id: "peer-xyz".into(),
            model_id: "openhydra-qwen3.5-2b".into(),
            shard_index: 1,
            total_shards: 4,
            load_pct: 0.75,
            healthy: true,
            daemon_mode: "peer".into(),
        });

        let data = [METHOD_GET_STATUS]; // PeerStatusRequest is empty.

        match dispatcher.dispatch(&data) {
            DispatchAction::StatusResponse(resp) => {
                assert_eq!(resp[0], METHOD_GET_STATUS);
                use prost::Message;
                let status =
                    crate::proto::PeerStatusResponse::decode(&resp[1..]).unwrap();
                assert_eq!(status.peer_id, "peer-xyz");
                assert_eq!(status.model_id, "openhydra-qwen3.5-2b");
                assert_eq!(status.shard_index, 1);
                assert_eq!(status.total_shards, 4);
                assert!(status.healthy);
            }
            other => panic!("expected StatusResponse, got {:?}", other),
        }
    }

    // ── Peer-mode dispatch: Legacy (no prefix) ────────────────────────

    #[test]
    fn test_dispatch_legacy_no_prefix() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let data = [0x0A, 0x10, 0x08, 0x01]; // protobuf
        match dispatcher.dispatch(&data) {
            DispatchAction::LegacyFallthrough => {}
            other => panic!("expected LegacyFallthrough, got {:?}", other),
        }
    }

    #[test]
    fn test_dispatch_empty_data() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        match dispatcher.dispatch(&[]) {
            DispatchAction::LegacyFallthrough => {}
            other => panic!("expected LegacyFallthrough, got {:?}", other),
        }
    }

    // ── Coordinator-mode dispatch ─────────────────────────────────────

    #[test]
    fn test_coordinator_push_result_blocking() {
        let dispatcher = Dispatcher::new(DispatchMode::Coordinator);
        let activation = test_activation();
        let resp_msg = make_response_msg("coord-001", &activation);

        let mut data = vec![METHOD_PUSH_RESULT];
        data.extend(&resp_msg);

        match dispatcher.dispatch(&data) {
            DispatchAction::PushResultBlocking(parsed) => {
                assert_eq!(parsed.header.request_id, "coord-001");
            }
            other => panic!("expected PushResultBlocking, got {:?}", other),
        }
    }

    #[test]
    fn test_coordinator_push_result_fire_forget() {
        let dispatcher = Dispatcher::new(DispatchMode::Coordinator);
        let activation = test_activation();
        let resp_msg = make_response_msg("coord-ff", &activation);

        let mut data = vec![METHOD_FIRE_FORGET_RESULT];
        data.extend(&resp_msg);

        match dispatcher.dispatch(&data) {
            DispatchAction::PushResultAsync { ack, push_result } => {
                assert_eq!(ack, vec![METHOD_FIRE_FORGET_RESULT]);
                assert_eq!(push_result.header.request_id, "coord-ff");
            }
            other => panic!("expected PushResultAsync, got {:?}", other),
        }
    }

    #[test]
    fn test_coordinator_rejects_forward() {
        let dispatcher = Dispatcher::new(DispatchMode::Coordinator);
        let fwd_msg = make_forward_msg(MsgType::Forward, "bad", &test_activation());

        let mut data = vec![METHOD_FORWARD];
        data.extend(&fwd_msg);

        match dispatcher.dispatch(&data) {
            DispatchAction::UnsupportedMethod { response, reason } => {
                assert_eq!(response[0], METHOD_PUSH_RESULT);
                assert!(reason.contains("coordinator"));
                // Verify the PushAck error can be decoded.
                use prost::Message;
                let ack = crate::proto::PushAck::decode(&response[1..]).unwrap();
                assert!(!ack.ok);
                assert_eq!(ack.error, "pure_coordinator_unsupported_proxy_method");
            }
            other => panic!("expected UnsupportedMethod, got {:?}", other),
        }
    }

    #[test]
    fn test_coordinator_rejects_fire_forget_forward() {
        let dispatcher = Dispatcher::new(DispatchMode::Coordinator);
        let data = [METHOD_FIRE_FORGET, 0x0A, 0x10];
        match dispatcher.dispatch(&data) {
            DispatchAction::UnsupportedMethod { .. } => {}
            other => panic!("expected UnsupportedMethod, got {:?}", other),
        }
    }

    #[test]
    fn test_coordinator_rejects_ping() {
        let dispatcher = Dispatcher::new(DispatchMode::Coordinator);
        let data = [METHOD_PING, 0x08, 0x01];
        match dispatcher.dispatch(&data) {
            DispatchAction::UnsupportedMethod { reason, .. } => {
                assert!(reason.contains("coordinator"));
            }
            other => panic!("expected UnsupportedMethod, got {:?}", other),
        }
    }

    #[test]
    fn test_coordinator_rejects_get_status() {
        let dispatcher = Dispatcher::new(DispatchMode::Coordinator);
        let data = [METHOD_GET_STATUS];
        match dispatcher.dispatch(&data) {
            DispatchAction::UnsupportedMethod { .. } => {}
            other => panic!("expected UnsupportedMethod, got {:?}", other),
        }
    }

    #[test]
    fn test_coordinator_rejects_legacy() {
        let dispatcher = Dispatcher::new(DispatchMode::Coordinator);
        let data = [0x0A, 0x10, 0x08]; // no prefix
        match dispatcher.dispatch(&data) {
            DispatchAction::UnsupportedMethod { .. } => {}
            other => panic!("expected UnsupportedMethod, got {:?}", other),
        }
    }

    // ── Parse error paths ─────────────────────────────────────────────

    #[test]
    fn test_dispatch_forward_truncated_forward_msg() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        // OHV2 magic but truncated — only 6 bytes after prefix.
        let mut data = vec![METHOD_FORWARD];
        data.extend(&0x4F485632u32.to_le_bytes()); // magic
        data.extend(&[0x00, 0x00]); // truncated

        match dispatcher.dispatch(&data) {
            DispatchAction::ParseError(reason) => {
                assert!(reason.contains("ForwardMsg"));
            }
            other => panic!("expected ParseError, got {:?}", other),
        }
    }

    #[test]
    fn test_dispatch_push_result_truncated_forward_msg() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let mut data = vec![METHOD_PUSH_RESULT];
        data.extend(&0x4F485632u32.to_le_bytes()); // magic
        data.extend(&[0x01, 0x00, 0x00, 0x00]); // version
        // No msg_type or header — truncated.

        match dispatcher.dispatch(&data) {
            DispatchAction::ParseError(reason) => {
                assert!(reason.contains("decode") || reason.contains("truncated")
                    || reason.contains("short"));
            }
            other => panic!("expected ParseError, got {:?}", other),
        }
    }

    // ── Full round-trip: encode → prefix → dispatch → verify ──────────

    #[test]
    fn test_full_roundtrip_forward() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let activation: Vec<u8> = (0..896)
            .map(|i| (i as f32) * 0.001)
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let header = IpcForwardHeader {
            request_id: "roundtrip-001".into(),
            stage_index: 2,
            total_stages: 4,
            push_mode: true,
            next_hop_peer_id: "12D3KooWXyz".into(),
            shard_layer_start: 16,
            shard_layer_end: 24,
            shard_total_layers: 32,
            kv_session_id: "sess-rt".into(),
            kv_store_activation: true,
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 896],
            ..Default::default()
        };

        let wire = forward_msg::encode(MsgType::Forward, &header, &activation).unwrap();
        let mut data = vec![METHOD_FORWARD];
        data.extend(&wire);

        match dispatcher.dispatch(&data) {
            DispatchAction::ForwardToWorker(parsed) => {
                // Verify all header fields survived the round-trip.
                assert_eq!(parsed.header.request_id, "roundtrip-001");
                assert_eq!(parsed.header.stage_index, 2);
                assert_eq!(parsed.header.total_stages, 4);
                assert!(parsed.header.push_mode);
                assert_eq!(parsed.header.next_hop_peer_id, "12D3KooWXyz");
                assert_eq!(parsed.header.shard_layer_start, 16);
                assert_eq!(parsed.header.shard_layer_end, 24);
                assert_eq!(parsed.header.shard_total_layers, 32);
                assert_eq!(parsed.header.kv_session_id, "sess-rt");
                assert!(parsed.header.kv_store_activation);
                assert_eq!(parsed.header.activation_dtype, ActivationDtype::Fp32);
                assert_eq!(parsed.header.activation_shape, vec![1, 1, 896]);
                // Verify activation data.
                assert_eq!(parsed.activation.len(), 896 * 4);
                assert_eq!(parsed.activation, activation);
            }
            other => panic!("expected ForwardToWorker, got {:?}", other),
        }
    }

    #[test]
    fn test_full_roundtrip_push_result() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let activation = test_activation();

        let header = IpcResponseHeader {
            status: IpcStatus::Ok,
            request_id: "push-rt".into(),
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 4],
            ..Default::default()
        };

        let wire = forward_msg::encode_response(&header, &activation).unwrap();
        let mut data = vec![METHOD_PUSH_RESULT];
        data.extend(&wire);

        match dispatcher.dispatch(&data) {
            DispatchAction::PushResultBlocking(parsed) => {
                assert_eq!(parsed.header.request_id, "push-rt");
                assert_eq!(parsed.header.status, IpcStatus::Ok);
                assert_eq!(parsed.activation, activation);
            }
            other => panic!("expected PushResultBlocking, got {:?}", other),
        }
    }

    // ── DispatchAction properties ─────────────────────────────────────

    #[test]
    fn test_needs_immediate_response() {
        let activation = test_activation();

        let worker = DispatchAction::ForwardToWorker(ParsedForward {
            header: IpcForwardHeader { request_id: "x".into(), ..Default::default() },
            activation: activation.clone(),
        });
        assert!(!worker.needs_immediate_response());

        let worker_async = DispatchAction::ForwardToWorkerAsync {
            ack: vec![METHOD_FIRE_FORGET],
            forward: ParsedForward {
                header: IpcForwardHeader { request_id: "x".into(), ..Default::default() },
                activation: activation.clone(),
            },
        };
        assert!(worker_async.needs_immediate_response());

        let unsupported = DispatchAction::UnsupportedMethod {
            response: vec![],
            reason: "test".into(),
        };
        assert!(unsupported.needs_immediate_response());

        let parse_err = DispatchAction::ParseError("test".into());
        assert!(parse_err.needs_immediate_response());

        let legacy = DispatchAction::LegacyFallthrough;
        assert!(!legacy.needs_immediate_response());
    }

    // ── Dispatch consistency: all 6 prefixes in peer mode ─────────────

    #[test]
    fn test_all_six_prefixes_peer_mode() {
        let dispatcher = Dispatcher::new(DispatchMode::Peer);
        let activation = test_activation();
        let fwd_wire = make_forward_msg(MsgType::Forward, "all6", &activation);
        let resp_wire = make_response_msg("all6", &activation);

        // 0x01: Forward blocking → ForwardToWorker
        let mut d = vec![METHOD_FORWARD];
        d.extend(&fwd_wire);
        assert!(matches!(dispatcher.dispatch(&d), DispatchAction::ForwardToWorker(_)));

        // 0x02: PushResult blocking → PushResultBlocking
        let mut d = vec![METHOD_PUSH_RESULT];
        d.extend(&resp_wire);
        assert!(matches!(dispatcher.dispatch(&d), DispatchAction::PushResultBlocking(_)));

        // 0x03: Forward fire-and-forget → ForwardToWorkerAsync
        let mut d = vec![METHOD_FIRE_FORGET];
        d.extend(&fwd_wire);
        assert!(matches!(dispatcher.dispatch(&d), DispatchAction::ForwardToWorkerAsync { .. }));

        // 0x04: PushResult fire-and-forget → PushResultAsync
        let mut d = vec![METHOD_FIRE_FORGET_RESULT];
        d.extend(&resp_wire);
        assert!(matches!(dispatcher.dispatch(&d), DispatchAction::PushResultAsync { .. }));

        // 0x05: Ping → PingResponse
        let d = [METHOD_PING, 0x08, 0x01];
        assert!(matches!(dispatcher.dispatch(&d), DispatchAction::PingResponse(_)));

        // 0x06: GetPeerStatus → StatusResponse
        let d = [METHOD_GET_STATUS];
        assert!(matches!(dispatcher.dispatch(&d), DispatchAction::StatusResponse(_)));
    }
}
