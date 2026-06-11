//! ForwardMsg — binary peer-to-peer wire format (replaces protobuf for the hot path).
//!
//! Layout:
//! ```text
//! [0:4]      magic              (u32 LE: 0x4F485632 = "OHV2")
//! [4:8]      version            (u32 LE: 1)
//! [8:10]     msg_type           (u16 LE: 0=forward, 1=push_result, 2=ping)
//! [10:12]    header_len         (u16 LE)
//! [12:12+H]  header             (CBOR-encoded ForwardHeader)
//! [12+H:12+H+4] activation_len (u32 LE — self-delimiting for batch splitting)
//! [12+H+4:..] activation_bytes (raw, dtype from header)
//! ```
//!
//! The header reuses `IpcForwardHeader` from `ipc_codec.rs` (same 31+ field
//! struct) to avoid duplicating the field list. CBOR omits unpopulated
//! fields, keeping wire size comparable to protobuf (~200–400 bytes) while
//! being extensible and forward-compatible.

use crate::ipc_codec::{IpcForwardHeader, IpcResponseHeader};

/// Magic bytes: "OHV2" in little-endian u32.
pub const FORWARD_MSG_MAGIC: u32 = 0x4F485632;

/// Current wire format version.
pub const FORWARD_MSG_VERSION: u32 = 1;

/// Fixed preamble size: magic(4) + version(4) + msg_type(2) + header_len(2).
const PREAMBLE_SIZE: usize = 12;

/// Message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum MsgType {
    Forward = 0,
    PushResult = 1,
    Ping = 2,
}

impl MsgType {
    pub fn from_u16(v: u16) -> Result<Self, String> {
        match v {
            0 => Ok(Self::Forward),
            1 => Ok(Self::PushResult),
            2 => Ok(Self::Ping),
            _ => Err(format!("unknown ForwardMsg type: {v}")),
        }
    }
}

/// A decoded ForwardMsg — header + activation reference.
pub struct DecodedForwardMsg<'a> {
    pub msg_type: MsgType,
    pub header: IpcForwardHeader,
    pub activation: &'a [u8],
}

/// Encode a ForwardMsg into the binary wire format.
///
/// Uses the same `IpcForwardHeader` from CP-0 to avoid duplicating the
/// 31+ field struct. The only structural difference from the IPC format
/// is the 12-byte preamble (magic + version + msg_type + header_len).
pub fn encode(
    msg_type: MsgType,
    header: &IpcForwardHeader,
    activation: &[u8],
) -> Result<Vec<u8>, String> {
    // CBOR-encode the header.
    let mut header_bytes = Vec::with_capacity(512);
    ciborium::into_writer(header, &mut header_bytes)
        .map_err(|e| format!("ForwardMsg CBOR encode failed: {e}"))?;

    let header_len = header_bytes.len();
    if header_len > u16::MAX as usize {
        return Err(format!(
            "ForwardMsg header too large: {header_len} bytes (max {})",
            u16::MAX
        ));
    }

    let activation_len = activation.len() as u32;
    let total = PREAMBLE_SIZE + header_len + 4 + activation.len();
    let mut buf = Vec::with_capacity(total);

    // Preamble.
    buf.extend_from_slice(&FORWARD_MSG_MAGIC.to_le_bytes());
    buf.extend_from_slice(&FORWARD_MSG_VERSION.to_le_bytes());
    buf.extend_from_slice(&(msg_type as u16).to_le_bytes());
    buf.extend_from_slice(&(header_len as u16).to_le_bytes());

    // CBOR header.
    buf.extend_from_slice(&header_bytes);

    // Self-delimiting activation.
    buf.extend_from_slice(&activation_len.to_le_bytes());
    buf.extend_from_slice(activation);

    Ok(buf)
}

/// Encode a response ForwardMsg (PushResult) with an `IpcResponseHeader`.
pub fn encode_response(
    header: &IpcResponseHeader,
    activation: &[u8],
) -> Result<Vec<u8>, String> {
    let mut header_bytes = Vec::with_capacity(256);
    ciborium::into_writer(header, &mut header_bytes)
        .map_err(|e| format!("ForwardMsg response CBOR encode failed: {e}"))?;

    let header_len = header_bytes.len();
    if header_len > u16::MAX as usize {
        return Err(format!(
            "ForwardMsg response header too large: {header_len} bytes"
        ));
    }

    let activation_len = activation.len() as u32;
    let total = PREAMBLE_SIZE + header_len + 4 + activation.len();
    let mut buf = Vec::with_capacity(total);

    buf.extend_from_slice(&FORWARD_MSG_MAGIC.to_le_bytes());
    buf.extend_from_slice(&FORWARD_MSG_VERSION.to_le_bytes());
    buf.extend_from_slice(&(MsgType::PushResult as u16).to_le_bytes());
    buf.extend_from_slice(&(header_len as u16).to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&activation_len.to_le_bytes());
    buf.extend_from_slice(activation);

    Ok(buf)
}

/// Check if raw bytes start with the ForwardMsg magic.
///
/// Used by proxy.rs for format negotiation: first 4 bytes decide
/// whether to dispatch as ForwardMsg or legacy protobuf.
#[inline]
pub fn is_forward_msg(data: &[u8]) -> bool {
    data.len() >= 4
        && u32::from_le_bytes(data[0..4].try_into().unwrap_or([0; 4])) == FORWARD_MSG_MAGIC
}

/// Decode a ForwardMsg from raw wire bytes.
///
/// Zero-copy for the activation payload: returns a reference into `data`.
pub fn decode(data: &[u8]) -> Result<DecodedForwardMsg<'_>, String> {
    if data.len() < PREAMBLE_SIZE {
        return Err(format!(
            "ForwardMsg too short: {} bytes (need at least {PREAMBLE_SIZE})",
            data.len()
        ));
    }

    // Parse preamble.
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != FORWARD_MSG_MAGIC {
        return Err(format!(
            "ForwardMsg bad magic: 0x{magic:08X} (expected 0x{FORWARD_MSG_MAGIC:08X})"
        ));
    }

    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != FORWARD_MSG_VERSION {
        return Err(format!(
            "ForwardMsg unsupported version: {version} (expected {FORWARD_MSG_VERSION})"
        ));
    }

    let msg_type_raw = u16::from_le_bytes(data[8..10].try_into().unwrap());
    let msg_type = MsgType::from_u16(msg_type_raw)?;

    let header_len = u16::from_le_bytes(data[10..12].try_into().unwrap()) as usize;

    // Validate we have enough bytes for header + activation_len.
    let min_len = PREAMBLE_SIZE + header_len + 4;
    if data.len() < min_len {
        return Err(format!(
            "ForwardMsg truncated: need {min_len} bytes, have {}",
            data.len()
        ));
    }

    // Decode CBOR header.
    let header: IpcForwardHeader =
        ciborium::from_reader(&data[PREAMBLE_SIZE..PREAMBLE_SIZE + header_len])
            .map_err(|e| format!("ForwardMsg CBOR decode failed: {e}"))?;

    // Self-delimiting activation.
    let act_len_offset = PREAMBLE_SIZE + header_len;
    let activation_len =
        u32::from_le_bytes(data[act_len_offset..act_len_offset + 4].try_into().unwrap()) as usize;

    let act_start = act_len_offset + 4;
    let act_end = act_start + activation_len;
    if data.len() < act_end {
        return Err(format!(
            "ForwardMsg activation truncated: declared {activation_len} bytes, have {}",
            data.len() - act_start
        ));
    }

    Ok(DecodedForwardMsg {
        msg_type,
        header,
        activation: &data[act_start..act_end],
    })
}

/// Decode a response ForwardMsg (PushResult) into an `IpcResponseHeader`.
pub fn decode_response(data: &[u8]) -> Result<(IpcResponseHeader, &[u8]), String> {
    if data.len() < PREAMBLE_SIZE {
        return Err(format!(
            "ForwardMsg response too short: {} bytes",
            data.len()
        ));
    }

    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != FORWARD_MSG_MAGIC {
        return Err(format!("ForwardMsg response bad magic: 0x{magic:08X}"));
    }

    // Audit 2.2: mirror decode()'s version gate (decode_response runs on the
    // same untrusted network bytes and previously validated only the magic).
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != FORWARD_MSG_VERSION {
        return Err(format!(
            "ForwardMsg response unsupported version: {version} (expected {FORWARD_MSG_VERSION})"
        ));
    }

    let header_len = u16::from_le_bytes(data[10..12].try_into().unwrap()) as usize;

    let min_len = PREAMBLE_SIZE + header_len + 4;
    if data.len() < min_len {
        return Err(format!(
            "ForwardMsg response truncated: need {min_len}, have {}",
            data.len()
        ));
    }

    let header: IpcResponseHeader =
        ciborium::from_reader(&data[PREAMBLE_SIZE..PREAMBLE_SIZE + header_len])
            .map_err(|e| format!("ForwardMsg response CBOR decode: {e}"))?;

    let act_len_offset = PREAMBLE_SIZE + header_len;
    let activation_len =
        u32::from_le_bytes(data[act_len_offset..act_len_offset + 4].try_into().unwrap()) as usize;
    let act_start = act_len_offset + 4;
    let act_end = act_start + activation_len;
    if data.len() < act_end {
        return Err(format!(
            "ForwardMsg response activation truncated: {activation_len} vs {}",
            data.len() - act_start
        ));
    }

    Ok((header, &data[act_start..act_end]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_codec::{ActivationDtype, IpcStatus};

    fn test_activation() -> Vec<u8> {
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    #[test]
    fn test_forward_roundtrip() {
        let header = IpcForwardHeader {
            request_id: "fwd-001".into(),
            stage_index: 1,
            total_stages: 4,
            push_mode: true,
            shard_layer_start: 8,
            shard_layer_end: 16,
            shard_total_layers: 32,
            kv_session_id: "sess-abc".into(),
            kv_store_activation: true,
            decode_temperature: 0.7,
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 6],
            ..Default::default()
        };
        let activation = test_activation();

        let wire = encode(MsgType::Forward, &header, &activation).unwrap();

        // Verify magic bytes.
        assert!(is_forward_msg(&wire));
        assert_eq!(
            u32::from_le_bytes(wire[0..4].try_into().unwrap()),
            FORWARD_MSG_MAGIC
        );

        // Decode.
        let decoded = decode(&wire).unwrap();
        assert_eq!(decoded.msg_type, MsgType::Forward);
        assert_eq!(decoded.header.request_id, "fwd-001");
        assert_eq!(decoded.header.stage_index, 1);
        assert_eq!(decoded.header.total_stages, 4);
        assert!(decoded.header.push_mode);
        assert_eq!(decoded.header.shard_layer_start, 8);
        assert_eq!(decoded.header.shard_layer_end, 16);
        assert_eq!(decoded.header.kv_session_id, "sess-abc");
        assert!(decoded.header.kv_store_activation);
        assert!((decoded.header.decode_temperature - 0.7).abs() < 1e-6);
        assert_eq!(decoded.header.activation_shape, vec![1, 1, 6]);

        // Activation is bit-for-bit identical.
        assert_eq!(decoded.activation, &activation[..]);
    }

    #[test]
    fn test_push_result_roundtrip() {
        let header = IpcForwardHeader {
            request_id: "push-001".into(),
            ring_mode: true,
            ring_tokens_remaining: 42,
            ..Default::default()
        };
        let activation = test_activation();

        let wire = encode(MsgType::PushResult, &header, &activation).unwrap();
        let decoded = decode(&wire).unwrap();

        assert_eq!(decoded.msg_type, MsgType::PushResult);
        assert_eq!(decoded.header.request_id, "push-001");
        assert!(decoded.header.ring_mode);
        assert_eq!(decoded.header.ring_tokens_remaining, 42);
        assert_eq!(decoded.activation, &activation[..]);
    }

    #[test]
    fn test_ping_roundtrip() {
        let header = IpcForwardHeader {
            request_id: "ping-001".into(),
            ..Default::default()
        };

        let wire = encode(MsgType::Ping, &header, &[]).unwrap();
        let decoded = decode(&wire).unwrap();

        assert_eq!(decoded.msg_type, MsgType::Ping);
        assert_eq!(decoded.header.request_id, "ping-001");
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn test_response_roundtrip() {
        let header = IpcResponseHeader {
            request_id: "resp-001".into(),
            status: IpcStatus::Ok,
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 6],
            metadata_json: r#"{"elapsed_ms":5.2}"#.into(),
            ..Default::default()
        };
        let activation = test_activation();

        let wire = encode_response(&header, &activation).unwrap();
        assert!(is_forward_msg(&wire));

        let (decoded_hdr, decoded_act) = decode_response(&wire).unwrap();
        assert_eq!(decoded_hdr.request_id, "resp-001");
        assert_eq!(decoded_hdr.status, IpcStatus::Ok);
        assert_eq!(decoded_hdr.activation_shape, vec![1, 1, 6]);
        assert_eq!(decoded_hdr.metadata_json, r#"{"elapsed_ms":5.2}"#);
        assert_eq!(decoded_act, &activation[..]);
    }

    #[test]
    fn test_empty_activation_forward() {
        let header = IpcForwardHeader {
            request_id: "empty-fwd".into(),
            ..Default::default()
        };

        let wire = encode(MsgType::Forward, &header, &[]).unwrap();
        let decoded = decode(&wire).unwrap();

        assert_eq!(decoded.header.request_id, "empty-fwd");
        assert!(decoded.activation.is_empty());
    }

    #[test]
    fn test_bad_magic_rejected() {
        let mut wire = encode(
            MsgType::Forward,
            &IpcForwardHeader {
                request_id: "x".into(),
                ..Default::default()
            },
            &[],
        )
        .unwrap();

        // Corrupt magic bytes.
        wire[0] = 0xFF;
        assert!(decode(&wire).is_err());
    }

    #[test]
    fn test_bad_version_rejected() {
        let mut wire = encode(
            MsgType::Forward,
            &IpcForwardHeader {
                request_id: "x".into(),
                ..Default::default()
            },
            &[],
        )
        .unwrap();

        // Set version to 99.
        wire[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(decode(&wire).is_err());
    }

    #[test]
    fn test_truncated_rejected() {
        assert!(decode(&[0u8; 8]).is_err()); // Too short for preamble.

        // Valid preamble but header_len exceeds available data.
        let mut buf = Vec::new();
        buf.extend_from_slice(&FORWARD_MSG_MAGIC.to_le_bytes());
        buf.extend_from_slice(&FORWARD_MSG_VERSION.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // msg_type
        buf.extend_from_slice(&200u16.to_le_bytes()); // header_len = 200
        buf.extend_from_slice(&[0u8; 10]); // Only 10 bytes of "header"
        assert!(decode(&buf).is_err());
    }

    #[test]
    fn test_is_forward_msg_false_for_protobuf() {
        // Protobuf messages start with field tags, not the OHV2 magic.
        let protobuf_like = vec![0x0A, 0x10, 0x08, 0x01];
        assert!(!is_forward_msg(&protobuf_like));
        assert!(!is_forward_msg(&[]));
        assert!(!is_forward_msg(&[0x01]));
    }

    #[test]
    fn test_large_activation_roundtrip() {
        // 896-dim hidden state (typical Qwen3.5-0.8B).
        let floats: Vec<f32> = (0..896).map(|i| (i as f32) * 0.001).collect();
        let activation: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let header = IpcForwardHeader {
            request_id: "large-act".into(),
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 896],
            ..Default::default()
        };

        let wire = encode(MsgType::Forward, &header, &activation).unwrap();
        let decoded = decode(&wire).unwrap();

        // Verify bit-for-bit activation integrity.
        assert_eq!(decoded.activation.len(), activation.len());
        assert_eq!(decoded.activation, &activation[..]);

        // Verify individual floats.
        let decoded_floats: Vec<f32> = decoded
            .activation
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_floats.len(), 896);
        for (i, (orig, dec)) in floats.iter().zip(decoded_floats.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                dec.to_bits(),
                "float mismatch at index {i}: {orig} vs {dec}"
            );
        }
    }

    #[test]
    fn test_encode_benchmark_under_10us() {
        // Verify encode is fast enough (< 10μs target from plan).
        let floats: Vec<f32> = (0..896).map(|i| (i as f32) * 0.001).collect();
        let activation: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

        let header = IpcForwardHeader {
            request_id: "bench".into(),
            stage_index: 1,
            total_stages: 4,
            shard_layer_start: 8,
            shard_layer_end: 16,
            shard_total_layers: 32,
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 896],
            ..Default::default()
        };

        // Warmup.
        for _ in 0..100 {
            let _ = encode(MsgType::Forward, &header, &activation).unwrap();
        }

        // Measure.
        let n = 10_000;
        let start = std::time::Instant::now();
        for _ in 0..n {
            let _ = encode(MsgType::Forward, &header, &activation).unwrap();
        }
        let elapsed = start.elapsed();
        let per_encode_us = elapsed.as_micros() as f64 / n as f64;

        // Target: < 10μs per encode.
        assert!(
            per_encode_us < 10.0,
            "ForwardMsg encode too slow: {per_encode_us:.2}μs (target <10μs)"
        );
    }

    #[test]
    fn test_all_msg_types_roundtrip() {
        for msg_type in [MsgType::Forward, MsgType::PushResult, MsgType::Ping] {
            let header = IpcForwardHeader {
                request_id: format!("{msg_type:?}"),
                ..Default::default()
            };
            let wire = encode(msg_type, &header, &[42u8; 16]).unwrap();
            let decoded = decode(&wire).unwrap();
            assert_eq!(decoded.msg_type, msg_type);
            assert_eq!(decoded.activation, &[42u8; 16]);
        }
    }

    #[test]
    fn test_wire_size_reasonable() {
        // Minimal header with empty activation should be compact.
        let header = IpcForwardHeader {
            request_id: "x".into(),
            ..Default::default()
        };
        let wire = encode(MsgType::Forward, &header, &[]).unwrap();

        // Preamble (12) + small CBOR header + activation_len (4) + 0 activation.
        // Should be well under 100 bytes.
        assert!(
            wire.len() < 100,
            "Minimal ForwardMsg too large: {} bytes",
            wire.len()
        );

        // Full header with 896-dim activation should be under 4KB.
        let full_header = IpcForwardHeader {
            request_id: "full-test-request-id-001".into(),
            stage_index: 3,
            total_stages: 4,
            push_mode: true,
            shard_layer_start: 24,
            shard_layer_end: 32,
            shard_total_layers: 32,
            kv_session_id: "session-xyz-123".into(),
            kv_store_activation: true,
            decode_temperature: 0.7,
            decode_top_p: 0.9,
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 896],
            ring_mode: true,
            ring_tokens_remaining: 100,
            ..Default::default()
        };
        let act: Vec<u8> = vec![0u8; 896 * 4]; // 896 float32s
        let wire = encode(MsgType::Forward, &full_header, &act).unwrap();
        assert!(
            wire.len() < 4096,
            "Full ForwardMsg too large: {} bytes",
            wire.len()
        );
    }
}
