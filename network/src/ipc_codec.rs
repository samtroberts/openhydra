//! IPC codec — CBOR header + raw activation wire format for Rust ↔ Python IPC.
//!
//! Wire format (request, Rust → Python):
//!   [0:4]     header_len      (u32 LE)
//!   [4:4+H]   header          (CBOR-encoded IpcForwardHeader)
//!   [4+H:4+H+4] activation_len (u32 LE)
//!   [4+H+4:..] activation     (raw bytes, dtype from header)
//!
//! Wire format (response, Python → Rust):
//!   [0:4]     header_len      (u32 LE)
//!   [4:4+H]   header          (CBOR-encoded IpcResponseHeader)
//!   [4+H:4+H+4] activation_len (u32 LE)
//!   [4+H+4:..] activation     (raw bytes)

use serde::{Deserialize, Serialize};

/// Activation dtype tag.
///
/// Serializes as the variant name ("Fp32", "Fp16", "Int8") for both CBOR and JSON.
/// Deserializes from either variant name strings OR integer tags (0, 1, 2)
/// for ergonomic use from Python dicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u8)]
pub enum ActivationDtype {
    Fp32 = 0,
    Fp16 = 1,
    Int8 = 2,
}

impl<'de> serde::Deserialize<'de> for ActivationDtype {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct DtypeVisitor;
        impl<'de> serde::de::Visitor<'de> for DtypeVisitor {
            type Value = ActivationDtype;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("integer 0-2 or string \"Fp32\"/\"Fp16\"/\"Int8\"")
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(ActivationDtype::from(v as u8))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(ActivationDtype::from(v as u8))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                match v {
                    "Fp32" => Ok(ActivationDtype::Fp32),
                    "Fp16" => Ok(ActivationDtype::Fp16),
                    "Int8" => Ok(ActivationDtype::Int8),
                    _ => Err(E::unknown_variant(v, &["Fp32", "Fp16", "Int8"])),
                }
            }
        }
        deserializer.deserialize_any(DtypeVisitor)
    }
}

impl Default for ActivationDtype {
    fn default() -> Self {
        Self::Fp32
    }
}

impl From<u8> for ActivationDtype {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Fp32,
            1 => Self::Fp16,
            2 => Self::Int8,
            _ => Self::Fp32,
        }
    }
}

/// Response status codes.
///
/// Serializes as variant name; deserializes from either name or integer tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[repr(u8)]
pub enum IpcStatus {
    Ok = 0,
    Error = 1,
    KvCacheHit = 2,
}

impl<'de> serde::Deserialize<'de> for IpcStatus {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct StatusVisitor;
        impl<'de> serde::de::Visitor<'de> for StatusVisitor {
            type Value = IpcStatus;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("integer 0-2 or string \"Ok\"/\"Error\"/\"KvCacheHit\"")
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(IpcStatus::from(v as u8))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(IpcStatus::from(v as u8))
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                match v {
                    "Ok" => Ok(IpcStatus::Ok),
                    "Error" => Ok(IpcStatus::Error),
                    "KvCacheHit" => Ok(IpcStatus::KvCacheHit),
                    _ => Err(E::unknown_variant(v, &["Ok", "Error", "KvCacheHit"])),
                }
            }
        }
        deserializer.deserialize_any(StatusVisitor)
    }
}

impl Default for IpcStatus {
    fn default() -> Self {
        Self::Ok
    }
}

impl From<u8> for IpcStatus {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Ok,
            1 => Self::Error,
            2 => Self::KvCacheHit,
            _ => Self::Ok,
        }
    }
}

/// IPC forward request header — carries all fields that `_push_to_next_hop` propagates.
///
/// CBOR-encoded. Fields with default/empty values are omitted by serde to
/// minimize header size (~200–400 bytes depending on population).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcForwardHeader {
    // ── Core routing ───────────────────────────────────────────────────
    pub request_id: String,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub stage_index: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub total_stages: u32,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub push_mode: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub next_hop_address: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub next_hop_peer_id: String,

    // ── Layer sharding ─────────────────────────────────────────────────
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub shard_layer_start: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub shard_layer_end: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub shard_total_layers: u32,

    // ── KV cache ───────────────────────────────────────────────────────
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub kv_session_id: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub kv_store_activation: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub kv_use_cached_activation: bool,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub kv_rollback_to: u32,

    // ── Decode parameters ──────────────────────────────────────────────
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub decode_do_sample: bool,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub decode_temperature: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub decode_top_p: f32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub decode_top_k: u32,
    #[serde(default)]
    pub decode_seed: i64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub sample_on_coordinator: bool,

    // ── Activation metadata ────────────────────────────────────────────
    #[serde(default)]
    pub activation_dtype: ActivationDtype,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activation_shape: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub quantized_scales: Vec<u8>,

    // ── Pipeline / speculative ─────────────────────────────────────────
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub slot_id: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub pipeline_depth: u32,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub draft_block: bool,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub block_index: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub draft_token_ids: Vec<i64>,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub verify_batch_size: u32,

    // ── Ring autoregressive ────────────────────────────────────────────
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub ring_mode: bool,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub ring_tokens_remaining: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ring_generated_ids: Vec<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ring_eos_ids: Vec<i64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub ring_first_hop_address: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub ring_first_hop_peer_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub ring_first_hop_libp2p_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ring_full_route: Vec<u8>,

    // ── Callback routing ───────────────────────────────────────────────
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub final_callback_address: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub final_callback_request_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub final_callback_libp2p_peer_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remaining_route: Vec<u8>,

    // ── Prompt (stage 0 only) ──────────────────────────────────────────
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompt_token_ids: Vec<i64>,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub max_tokens: u32,

    // ── Compression (tensor autoencoder) ────────────────────────────────
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub compression_codec: String,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub compression_original_dim: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub compression_latent_dim: u32,

    // ── Encryption (pass-through) ──────────────────────────────────────
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub encrypted_activation: Vec<u8>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub encryption_suite: String,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub encryption_layers: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub encryption_nonces: Vec<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub encryption_ephemeral_keys: Vec<Vec<u8>>,

    // ── Onion routing ────────────────────────────────────────────────────
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub onion_route_ciphertext: Vec<u8>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub onion_route_nonces: Vec<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub onion_route_ephemeral_public_keys: Vec<Vec<u8>>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub onion_route_suite: String,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub onion_route_layers: u32,

    // ── Geo verification ─────────────────────────────────────────────────
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub geo_claimed_region: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub geo_nonce: Vec<u8>,
}

/// A single hop in a push-mode route (serialized into `remaining_route`).
///
/// Python serializes `remaining_route` as `serde_json::to_vec([PeerHopEntry, ...])`
/// before passing it to `encode_forward_msg`. The Rust side treats it as opaque
/// bytes in `IpcForwardHeader.remaining_route`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerHopEntry {
    pub address: String,
    pub peer_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub libp2p_peer_id: String,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub shard_layer_start: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub shard_layer_end: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub shard_total_layers: u32,
}

/// IPC response header — returned by the Python worker after forward().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcResponseHeader {
    pub request_id: String,
    #[serde(default)]
    pub status: IpcStatus,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub peer_id: String,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub stage_index: u32,
    #[serde(default)]
    pub activation_dtype: ActivationDtype,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activation_shape: Vec<u32>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub metadata_json: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub error_message: String,
    /// TOPLOC integrity hash of the activation (audit H1/H2).
    ///
    /// CANONICAL ENCODING: lowercase hex of the 32-byte SHA-256 digest.
    /// The Python peer MUST encode with `bytes.hex()` and decode with
    /// `bytes.fromhex()`. Do NOT `str()` the raw digest — that yields the
    /// Python repr (`"b'\\xab..'"`), corrupts the hash, and silently breaks
    /// `verify_hash` on the coordinator. Empty string means "no hash".
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub activation_hash: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_hidden_state: bool,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub slot_id: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub block_size: u32,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub block_index: u32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub onion_next_peer_id: String,

    // ── Onion route pass-through ──────────────────────────────────────
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub onion_route_ciphertext: Vec<u8>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub onion_route_nonces: Vec<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub onion_route_ephemeral_public_keys: Vec<Vec<u8>>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub onion_route_suite: String,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub onion_route_layers: u32,

    // ── Differential privacy audit ────────────────────────────────────
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dp_noise_applied: bool,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub dp_noise_configured_variance: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub dp_noise_observed_variance: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub dp_noise_observed_std: f64,
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub dp_noise_payload_index: u32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub dp_noise_audit_tag: String,
}

// ── Serde helpers ──────────────────────────────────────────────────────

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}

// ── Batch wire magic ──────────────────────────────────────────────────

/// Magic prefix for batch IPC messages — "BTCH" as u32 LE.
///
/// Batch wire format (request or response):
///   [0:4]   BATCH_MAGIC (0x48435442 LE)
///   [4:8]   batch_count (u32 LE)
///   For each item (batch_count times):
///     Standard single-item encoding:
///       [0:4]   header_len (u32 LE)
///       [4:4+H] header     (CBOR)
///       [4+H:4+H+4] act_len (u32 LE)
///       [4+H+4:..] activation (raw bytes)
///
/// Detection: first 4 bytes as u32 LE. Valid single-request `header_len`
/// values are 50–500; `BATCH_MAGIC` = 1,212,498,498 — zero collision risk.
pub const BATCH_MAGIC: u32 = 0x48435442;

/// Minimum on-wire bytes for a single batch item (audit 2.2).
///
/// Every item carries at least a 4-byte header-length prefix, a non-empty
/// CBOR header, and a 4-byte activation-length prefix, so it cannot be
/// smaller than this. Used to clamp the pre-allocation in the batch
/// decoders so an attacker-supplied `batch_count` (a raw u32, up to ~4.3 B)
/// cannot drive a multi-terabyte `Vec::with_capacity` that aborts the
/// process before any per-item validation runs.
const MIN_BATCH_ITEM_BYTES: usize = 8;

/// Check if a wire message is a batch (starts with BATCH_MAGIC).
pub fn is_batch_message(data: &[u8]) -> bool {
    data.len() >= 4
        && u32::from_le_bytes(data[0..4].try_into().unwrap()) == BATCH_MAGIC
}

/// Compute the wire length of a single item (request or response) without
/// decoding the CBOR header.  Used for offset tracking in batch decoders.
fn single_item_wire_len(data: &[u8]) -> Result<usize, String> {
    if data.len() < 8 {
        return Err(format!(
            "item too short for wire length: {} bytes",
            data.len()
        ));
    }
    let header_len =
        u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let prefix = 4 + header_len;
    if data.len() < prefix + 4 {
        return Err(format!(
            "item truncated: header_len={header_len}, have {} bytes",
            data.len()
        ));
    }
    let act_len =
        u32::from_le_bytes(data[prefix..prefix + 4].try_into().unwrap()) as usize;
    Ok(prefix + 4 + act_len)
}

// ── Encode / decode ────────────────────────────────────────────────────

/// Encode an IPC forward request into the wire format.
pub fn encode_forward_request(
    header: &IpcForwardHeader,
    activation: &[u8],
) -> Result<Vec<u8>, String> {
    let mut header_bytes = Vec::with_capacity(512);
    ciborium::into_writer(header, &mut header_bytes)
        .map_err(|e| format!("CBOR encode failed: {e}"))?;

    let header_len = header_bytes.len() as u32;
    let activation_len = activation.len() as u32;
    let total_len = 4 + header_bytes.len() + 4 + activation.len();
    let mut buf = Vec::with_capacity(total_len);

    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&activation_len.to_le_bytes());
    buf.extend_from_slice(activation);

    Ok(buf)
}

/// Decode an IPC forward request from wire bytes.
pub fn decode_forward_request(data: &[u8]) -> Result<(IpcForwardHeader, &[u8]), String> {
    if data.len() < 8 {
        return Err(format!("IPC request too short: {} bytes", data.len()));
    }

    let header_len =
        u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    if data.len() < 4 + header_len + 4 {
        return Err(format!(
            "IPC request truncated: need {} bytes for header, have {}",
            4 + header_len + 4,
            data.len()
        ));
    }

    let header: IpcForwardHeader =
        ciborium::from_reader(&data[4..4 + header_len])
            .map_err(|e| format!("CBOR decode failed: {e}"))?;

    let act_offset = 4 + header_len;
    let activation_len =
        u32::from_le_bytes(data[act_offset..act_offset + 4].try_into().unwrap()) as usize;

    let act_start = act_offset + 4;
    let act_end = act_start + activation_len;
    if data.len() < act_end {
        return Err(format!(
            "IPC activation truncated: declared {} bytes, have {}",
            activation_len,
            data.len() - act_start
        ));
    }

    Ok((header, &data[act_start..act_end]))
}

/// Encode an IPC response into the wire format.
pub fn encode_response(
    header: &IpcResponseHeader,
    activation: &[u8],
) -> Result<Vec<u8>, String> {
    let mut header_bytes = Vec::with_capacity(256);
    ciborium::into_writer(header, &mut header_bytes)
        .map_err(|e| format!("CBOR encode failed: {e}"))?;

    let header_len = header_bytes.len() as u32;
    let activation_len = activation.len() as u32;
    let total_len = 4 + header_bytes.len() + 4 + activation.len();
    let mut buf = Vec::with_capacity(total_len);

    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&activation_len.to_le_bytes());
    buf.extend_from_slice(activation);

    Ok(buf)
}

/// Decode an IPC response from wire bytes.
pub fn decode_response(data: &[u8]) -> Result<(IpcResponseHeader, &[u8]), String> {
    if data.len() < 8 {
        return Err(format!("IPC response too short: {} bytes", data.len()));
    }

    let header_len =
        u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    if data.len() < 4 + header_len + 4 {
        return Err(format!(
            "IPC response truncated: need {} bytes for header, have {}",
            4 + header_len + 4,
            data.len()
        ));
    }

    let header: IpcResponseHeader =
        ciborium::from_reader(&data[4..4 + header_len])
            .map_err(|e| format!("CBOR decode failed: {e}"))?;

    let act_offset = 4 + header_len;
    let activation_len =
        u32::from_le_bytes(data[act_offset..act_offset + 4].try_into().unwrap()) as usize;

    let act_start = act_offset + 4;
    let act_end = act_start + activation_len;
    if data.len() < act_end {
        return Err(format!(
            "IPC response activation truncated: declared {} bytes, have {}",
            activation_len,
            data.len() - act_start
        ));
    }

    Ok((header, &data[act_start..act_end]))
}

// ── Batch encode / decode ─────────────────────────────────────────────

/// Encode a batch of IPC forward requests into the batch wire format.
///
/// Each item is a (header, activation_bytes) pair.  The output is a single
/// contiguous buffer suitable for a single IPC send.
pub fn encode_batch_request(
    items: &[(&IpcForwardHeader, &[u8])],
) -> Result<Vec<u8>, String> {
    let batch_count = items.len() as u32;
    let mut buf = Vec::with_capacity(8 + items.len() * 512);

    buf.extend_from_slice(&BATCH_MAGIC.to_le_bytes());
    buf.extend_from_slice(&batch_count.to_le_bytes());

    for (header, activation) in items {
        let single = encode_forward_request(header, activation)?;
        buf.extend_from_slice(&single);
    }

    Ok(buf)
}

/// Decode a batch of IPC forward requests from wire bytes.
///
/// Returns a Vec of (header, owned_activation) pairs.
pub fn decode_batch_request(data: &[u8]) -> Result<Vec<(IpcForwardHeader, Vec<u8>)>, String> {
    if data.len() < 8 {
        return Err(format!("batch request too short: {} bytes", data.len()));
    }

    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != BATCH_MAGIC {
        return Err(format!(
            "invalid batch magic: expected {BATCH_MAGIC:#010x}, got {magic:#010x}"
        ));
    }

    let batch_count =
        u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    // Audit 2.2: clamp the pre-allocation to what the buffer could possibly
    // hold. The loop below still validates and errors on truncation; this
    // only bounds the up-front reservation.
    let cap = batch_count.min(data.len() / MIN_BATCH_ITEM_BYTES);
    let mut results = Vec::with_capacity(cap);
    let mut offset = 8;

    for i in 0..batch_count {
        if offset >= data.len() && batch_count > 0 {
            return Err(format!(
                "batch request truncated at item {i}/{batch_count}"
            ));
        }
        let item_data = &data[offset..];
        let item_len = single_item_wire_len(item_data)?;
        let (header, activation) = decode_forward_request(item_data)?;
        results.push((header, activation.to_vec()));
        offset += item_len;
    }

    Ok(results)
}

/// Encode a batch of IPC responses into the batch wire format.
pub fn encode_batch_response(
    items: &[(&IpcResponseHeader, &[u8])],
) -> Result<Vec<u8>, String> {
    let batch_count = items.len() as u32;
    let mut buf = Vec::with_capacity(8 + items.len() * 256);

    buf.extend_from_slice(&BATCH_MAGIC.to_le_bytes());
    buf.extend_from_slice(&batch_count.to_le_bytes());

    for (header, activation) in items {
        let single = encode_response(header, activation)?;
        buf.extend_from_slice(&single);
    }

    Ok(buf)
}

/// Decode a batch of IPC responses from wire bytes.
///
/// Returns a Vec of (header, owned_activation) pairs.
pub fn decode_batch_response(data: &[u8]) -> Result<Vec<(IpcResponseHeader, Vec<u8>)>, String> {
    if data.len() < 8 {
        return Err(format!("batch response too short: {} bytes", data.len()));
    }

    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != BATCH_MAGIC {
        return Err(format!(
            "invalid batch magic: expected {BATCH_MAGIC:#010x}, got {magic:#010x}"
        ));
    }

    let batch_count =
        u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    // Audit 2.2: clamp the pre-allocation (see decode_batch_request).
    let cap = batch_count.min(data.len() / MIN_BATCH_ITEM_BYTES);
    let mut results = Vec::with_capacity(cap);
    let mut offset = 8;

    for i in 0..batch_count {
        if offset >= data.len() && batch_count > 0 {
            return Err(format!(
                "batch response truncated at item {i}/{batch_count}"
            ));
        }
        let item_data = &data[offset..];
        let item_len = single_item_wire_len(item_data)?;
        let (header, activation) = decode_response(item_data)?;
        results.push((header, activation.to_vec()));
        offset += item_len;
    }

    Ok(results)
}

/// Create a minimal IpcForwardHeader with defaults for testing.
impl Default for IpcForwardHeader {
    fn default() -> Self {
        Self {
            request_id: String::new(),
            stage_index: 0,
            total_stages: 1,
            push_mode: false,
            next_hop_address: String::new(),
            next_hop_peer_id: String::new(),
            shard_layer_start: 0,
            shard_layer_end: 0,
            shard_total_layers: 0,
            kv_session_id: String::new(),
            kv_store_activation: false,
            kv_use_cached_activation: false,
            kv_rollback_to: 0,
            decode_do_sample: false,
            decode_temperature: 0.0,
            decode_top_p: 0.0,
            decode_top_k: 0,
            decode_seed: 0,
            sample_on_coordinator: false,
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: Vec::new(),
            quantized_scales: Vec::new(),
            slot_id: 0,
            pipeline_depth: 0,
            draft_block: false,
            block_index: 0,
            draft_token_ids: Vec::new(),
            verify_batch_size: 0,
            ring_mode: false,
            ring_tokens_remaining: 0,
            ring_generated_ids: Vec::new(),
            ring_eos_ids: Vec::new(),
            ring_first_hop_address: String::new(),
            ring_first_hop_peer_id: String::new(),
            ring_first_hop_libp2p_id: String::new(),
            ring_full_route: Vec::new(),
            final_callback_address: String::new(),
            final_callback_request_id: String::new(),
            final_callback_libp2p_peer_id: String::new(),
            remaining_route: Vec::new(),
            prompt: String::new(),
            prompt_token_ids: Vec::new(),
            max_tokens: 0,
            compression_codec: String::new(),
            compression_original_dim: 0,
            compression_latent_dim: 0,
            encrypted_activation: Vec::new(),
            encryption_suite: String::new(),
            encryption_layers: 0,
            encryption_nonces: Vec::new(),
            encryption_ephemeral_keys: Vec::new(),
            onion_route_ciphertext: Vec::new(),
            onion_route_nonces: Vec::new(),
            onion_route_ephemeral_public_keys: Vec::new(),
            onion_route_suite: String::new(),
            onion_route_layers: 0,
            geo_claimed_region: String::new(),
            geo_nonce: Vec::new(),
        }
    }
}

impl Default for IpcResponseHeader {
    fn default() -> Self {
        Self {
            request_id: String::new(),
            status: IpcStatus::Ok,
            peer_id: String::new(),
            stage_index: 0,
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: Vec::new(),
            metadata_json: String::new(),
            error_message: String::new(),
            activation_hash: String::new(),
            is_hidden_state: false,
            slot_id: 0,
            block_size: 0,
            block_index: 0,
            onion_next_peer_id: String::new(),
            onion_route_ciphertext: Vec::new(),
            onion_route_nonces: Vec::new(),
            onion_route_ephemeral_public_keys: Vec::new(),
            onion_route_suite: String::new(),
            onion_route_layers: 0,
            dp_noise_applied: false,
            dp_noise_configured_variance: 0.0,
            dp_noise_observed_variance: 0.0,
            dp_noise_observed_std: 0.0,
            dp_noise_payload_index: 0,
            dp_noise_audit_tag: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_header_roundtrip() {
        let header = IpcForwardHeader {
            request_id: "test-req-001".into(),
            stage_index: 2,
            total_stages: 4,
            push_mode: true,
            shard_layer_start: 8,
            shard_layer_end: 16,
            shard_total_layers: 32,
            kv_session_id: "session-abc".into(),
            kv_store_activation: true,
            decode_temperature: 0.7,
            decode_top_p: 0.9,
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 896],
            ring_mode: true,
            ring_tokens_remaining: 50,
            ..Default::default()
        };

        // Some fake activation data (4 float32s = 16 bytes).
        let activation: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let encoded = encode_forward_request(&header, &activation).unwrap();
        let (decoded_header, decoded_act) = decode_forward_request(&encoded).unwrap();

        assert_eq!(decoded_header.request_id, "test-req-001");
        assert_eq!(decoded_header.stage_index, 2);
        assert_eq!(decoded_header.total_stages, 4);
        assert!(decoded_header.push_mode);
        assert_eq!(decoded_header.shard_layer_start, 8);
        assert_eq!(decoded_header.shard_layer_end, 16);
        assert_eq!(decoded_header.shard_total_layers, 32);
        assert_eq!(decoded_header.kv_session_id, "session-abc");
        assert!(decoded_header.kv_store_activation);
        assert!((decoded_header.decode_temperature - 0.7).abs() < 1e-6);
        assert!((decoded_header.decode_top_p - 0.9).abs() < 1e-6);
        assert_eq!(decoded_header.activation_shape, vec![1, 1, 896]);
        assert!(decoded_header.ring_mode);
        assert_eq!(decoded_header.ring_tokens_remaining, 50);
        assert_eq!(decoded_act, &activation[..]);
    }

    #[test]
    fn test_response_roundtrip() {
        let header = IpcResponseHeader {
            request_id: "test-req-001".into(),
            status: IpcStatus::Ok,
            activation_dtype: ActivationDtype::Fp32,
            activation_shape: vec![1, 1, 896],
            ..Default::default()
        };

        let activation: Vec<u8> = vec![5.0f32, 6.0, 7.0, 8.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let encoded = encode_response(&header, &activation).unwrap();
        let (decoded_header, decoded_act) = decode_response(&encoded).unwrap();

        assert_eq!(decoded_header.request_id, "test-req-001");
        assert_eq!(decoded_header.status, IpcStatus::Ok);
        assert_eq!(decoded_header.activation_shape, vec![1, 1, 896]);
        assert_eq!(decoded_act, &activation[..]);
    }

    #[test]
    fn test_cbor_omits_defaults() {
        // A minimal header should produce a small CBOR blob.
        let header = IpcForwardHeader {
            request_id: "x".into(),
            ..Default::default()
        };
        let mut buf = Vec::new();
        ciborium::into_writer(&header, &mut buf).unwrap();
        // With all defaults skipped, the header should be well under 100 bytes.
        assert!(buf.len() < 100, "CBOR too large: {} bytes", buf.len());
    }

    #[test]
    fn test_empty_activation() {
        let header = IpcForwardHeader {
            request_id: "empty-act".into(),
            ..Default::default()
        };

        let encoded = encode_forward_request(&header, &[]).unwrap();
        let (decoded, act) = decode_forward_request(&encoded).unwrap();

        assert_eq!(decoded.request_id, "empty-act");
        assert!(act.is_empty());
    }

    #[test]
    fn test_error_response() {
        let header = IpcResponseHeader {
            request_id: "err-req".into(),
            status: IpcStatus::Error,
            error_message: "model not loaded".into(),
            ..Default::default()
        };

        let encoded = encode_response(&header, &[]).unwrap();
        let (decoded, _) = decode_response(&encoded).unwrap();

        assert_eq!(decoded.status, IpcStatus::Error);
        assert_eq!(decoded.error_message, "model not loaded");
    }

    #[test]
    fn test_truncated_request_rejected() {
        // Less than the minimum 8 bytes (4 header_len + 4 activation_len).
        assert!(decode_forward_request(&[0, 0, 0]).is_err());

        // Header length says 100 but only 10 bytes available.
        let mut bad = vec![0u8; 14];
        bad[0..4].copy_from_slice(&100u32.to_le_bytes());
        assert!(decode_forward_request(&bad).is_err());
    }

    #[test]
    fn test_activation_dtype_variants() {
        for (tag, expected) in [(0u8, ActivationDtype::Fp32), (1, ActivationDtype::Fp16), (2, ActivationDtype::Int8)] {
            let dt: ActivationDtype = tag.into();
            assert_eq!(dt, expected);
        }
        // Unknown tag falls back to Fp32.
        let dt: ActivationDtype = 99u8.into();
        assert_eq!(dt, ActivationDtype::Fp32);
    }

    // ── Batch wire format tests ──────────────────────────────────────

    #[test]
    fn test_batch_request_roundtrip() {
        let h1 = IpcForwardHeader {
            request_id: "req-1".into(),
            shard_layer_start: 0,
            shard_layer_end: 16,
            ..Default::default()
        };
        let h2 = IpcForwardHeader {
            request_id: "req-2".into(),
            shard_layer_start: 16,
            shard_layer_end: 32,
            ..Default::default()
        };
        let act1: Vec<u8> = vec![1.0f32, 2.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let act2: Vec<u8> = vec![3.0f32, 4.0, 5.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let encoded =
            encode_batch_request(&[(&h1, &act1), (&h2, &act2)]).unwrap();

        assert!(is_batch_message(&encoded));

        let decoded = decode_batch_request(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].0.request_id, "req-1");
        assert_eq!(decoded[0].0.shard_layer_start, 0);
        assert_eq!(decoded[0].1, act1);
        assert_eq!(decoded[1].0.request_id, "req-2");
        assert_eq!(decoded[1].0.shard_layer_start, 16);
        assert_eq!(decoded[1].1, act2);
    }

    #[test]
    fn test_batch_response_roundtrip() {
        let h1 = IpcResponseHeader {
            request_id: "resp-1".into(),
            status: IpcStatus::Ok,
            activation_shape: vec![1, 1, 4],
            ..Default::default()
        };
        let h2 = IpcResponseHeader {
            request_id: "resp-2".into(),
            status: IpcStatus::Ok,
            activation_shape: vec![1, 1, 2],
            ..Default::default()
        };
        let act1: Vec<u8> = vec![10.0f32, 20.0, 30.0, 40.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let act2: Vec<u8> = vec![50.0f32, 60.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let encoded =
            encode_batch_response(&[(&h1, &act1), (&h2, &act2)]).unwrap();
        assert!(is_batch_message(&encoded));

        let decoded = decode_batch_response(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].0.request_id, "resp-1");
        assert_eq!(decoded[0].0.activation_shape, vec![1, 1, 4]);
        assert_eq!(decoded[0].1, act1);
        assert_eq!(decoded[1].0.request_id, "resp-2");
        assert_eq!(decoded[1].1, act2);
    }

    #[test]
    fn test_batch_magic_not_confused_with_single() {
        let header = IpcForwardHeader {
            request_id: "single".into(),
            ..Default::default()
        };
        let single =
            encode_forward_request(&header, &[0u8; 16]).unwrap();
        assert!(!is_batch_message(&single));
    }

    #[test]
    fn test_single_item_batch() {
        let h = IpcForwardHeader {
            request_id: "solo".into(),
            ..Default::default()
        };
        let act: Vec<u8> = vec![1.0f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let encoded = encode_batch_request(&[(&h, &act)]).unwrap();
        let decoded = decode_batch_request(&encoded).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0.request_id, "solo");
        assert_eq!(decoded[0].1, act);
    }

    #[test]
    fn test_empty_batch() {
        let encoded = encode_batch_request(&[]).unwrap();
        assert!(is_batch_message(&encoded));
        let decoded = decode_batch_request(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_batch_count_overflow_does_not_oom() {
        // Audit 2.2: a tiny frame claiming u32::MAX items must NOT try to
        // pre-allocate billions of entries — it must error on truncation.
        let mut data = Vec::new();
        data.extend_from_slice(&BATCH_MAGIC.to_le_bytes());
        data.extend_from_slice(&u32::MAX.to_le_bytes()); // claim ~4.3B items
        // No item bytes follow.
        let req = decode_batch_request(&data);
        assert!(req.is_err(), "huge batch_count must error, not allocate");
        let resp = decode_batch_response(&data);
        assert!(resp.is_err(), "huge batch_count must error, not allocate");
    }

    #[test]
    fn test_batch_cross_format_response() {
        // Encode as batch request, verify we can't decode as batch response
        // (and vice versa) — the CBOR headers are structurally different.
        let h = IpcForwardHeader {
            request_id: "cross".into(),
            ..Default::default()
        };
        let act = vec![0u8; 8];
        let encoded = encode_batch_request(&[(&h, &act)]).unwrap();

        // Decoding as batch response should still work structurally
        // (CBOR is flexible), but the fields will differ.
        let result = decode_batch_response(&encoded);
        assert!(result.is_ok());
        // The response header will have request_id populated from the
        // CBOR map (shared field name).
        assert_eq!(result.unwrap()[0].0.request_id, "cross");
    }
}
