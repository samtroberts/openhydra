//! Shared data types for the OpenHydra P2P networking layer.
//!
//! These mirror the Python `Announcement` dataclass (40+ fields) and
//! `PeerEndpoint` dataclass, serializable to both CBOR (for Kademlia
//! records) and Python dicts (via serde_json → PyO3).

use serde::{Deserialize, Serialize};

/// A peer's full announcement record — stored in Kademlia DHT.
///
/// Mirrors `peer/dht_announce.py::Announcement` dataclass.
/// All fields optional except peer_id, model_id, host, port.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerRecord {
    // Required
    pub peer_id: String,
    pub model_id: String,
    pub host: String,
    pub port: u16,

    // Identity
    #[serde(default)]
    pub operator_id: Option<String>,
    #[serde(default)]
    pub region: Option<String>,

    // Load & daemon
    #[serde(default)]
    pub load_pct: f64,
    #[serde(default = "default_daemon_mode")]
    pub daemon_mode: String,

    // Runtime profile
    #[serde(default = "default_runtime_backend")]
    pub runtime_backend: String,
    #[serde(default)]
    pub runtime_target: String,
    #[serde(default)]
    pub runtime_model_id: String,
    #[serde(default)]
    pub quantization_mode: String,
    #[serde(default)]
    pub quantization_bits: u32,
    #[serde(default)]
    pub runtime_gpu_available: bool,
    #[serde(default)]
    pub runtime_estimated_tokens_per_sec: f64,
    #[serde(default)]
    pub runtime_estimated_memory_mb: u64,

    // protocol.md §4 — canonical model id + capability record (M1.2). All
    // #[serde(default)] so pre-M1.2 records (lacking these keys) still decode.
    // (region / requires_relay / reputation_score / runtime_backend already exist
    // above and cover the spec's region/requires_relay/reputation/backend.)
    #[serde(default)]
    pub canonical_model_id: String,
    #[serde(default)]
    pub context_length: u32,
    #[serde(default)]
    pub max_output_tokens: u32,
    /// Live measured decode throughput (tokens/s), distinct from the static
    /// `runtime_estimated_tokens_per_sec` estimate above.
    #[serde(default)]
    pub throughput_tok_s: f64,
    #[serde(default)]
    pub queue_depth: u32,
    #[serde(default)]
    pub hardware_class: String,

    // Privacy
    #[serde(default)]
    pub privacy_noise_variance: f64,

    // Reputation
    #[serde(default)]
    pub reputation_score: f64,

    // Expert specialization
    #[serde(default)]
    pub expert_tags: Vec<String>,
    #[serde(default)]
    pub expert_layer_indices: Vec<u32>,

    // Layer sharding
    #[serde(default)]
    pub layer_start: u32,
    #[serde(default)]
    pub layer_end: u32,
    #[serde(default)]
    pub total_layers: u32,

    // NAT traversal
    #[serde(default = "default_nat_type")]
    pub nat_type: String,
    #[serde(default)]
    pub requires_relay: bool,
    #[serde(default)]
    pub relay_address: String,

    /// IPv6 address for dual-stack peers (serde default empty).
    #[serde(default)]
    pub host_ipv6: String,

    // Identity keys
    #[serde(default)]
    pub peer_public_key: String,
    /// libp2p PeerId (base58 multihash) — new field for Rust networking.
    #[serde(default)]
    pub libp2p_peer_id: String,
    /// Ed25519 public key hex (from Rust P2PNode identity, Task 6.2).
    #[serde(default)]
    pub public_key: String,
    /// Ed25519 signature hex over canonical record bytes (Task 6.2).
    #[serde(default)]
    pub signature: String,

    // Timestamp
    #[serde(default)]
    pub updated_unix_ms: u64,
}

fn default_daemon_mode() -> String {
    "polite".to_string()
}
fn default_runtime_backend() -> String {
    "toy_cpu".to_string()
}
fn default_nat_type() -> String {
    "unknown".to_string()
}

/// NAT detection result — returned by `P2PNode.nat_status()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NatInfo {
    pub nat_type: String,
    /// First confirmed external address (backward compat).
    pub external_ip: String,
    /// Confirmed external IPv4 address (empty until classified).
    pub external_ipv4: String,
    /// Confirmed external IPv6 address (empty until classified).
    pub external_ipv6: String,
    pub external_port: u16,
    pub is_public: bool,
}

/// A discovered peer — returned by `P2PNode.discover()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPeer {
    pub peer_id: String,
    pub libp2p_peer_id: String,
    pub host: String,
    pub host_ipv6: String,
    pub port: u16,
    pub model_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub total_layers: u32,
    pub nat_type: String,
    pub requires_relay: bool,
    pub relay_address: String,
    pub runtime_backend: String,
    pub runtime_model_id: String,
    // protocol.md §4 capability fields (M1.2) — surfaced from PeerRecord so the
    // libp2p discover() path is not blind to canonical id / live capacity.
    pub canonical_model_id: String,
    pub context_length: u32,
    pub max_output_tokens: u32,
    pub throughput_tok_s: f64,
    pub queue_depth: u32,
    pub hardware_class: String,
    /// The resolved reachable address (direct or via relay).
    pub reachable_address: String,
}

impl PeerRecord {
    /// Serialize to CBOR bytes for Kademlia storage.
    pub fn to_cbor(&self) -> Result<Vec<u8>, ciborium::ser::Error<std::io::Error>> {
        let mut buf = Vec::new();
        ciborium::ser::into_writer(self, &mut buf)?;
        Ok(buf)
    }

    /// Deserialize from CBOR bytes.
    pub fn from_cbor(data: &[u8]) -> Result<Self, ciborium::de::Error<std::io::Error>> {
        ciborium::de::from_reader(data)
    }

    /// Kademlia key for this record: `/openhydra/model/{model_id}/{peer_id}`
    pub fn dht_key(&self) -> Vec<u8> {
        format!("/openhydra/model/{}/{}", self.model_id, self.peer_id)
            .into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbor_roundtrip() {
        let record = PeerRecord {
            peer_id: "test-peer".into(),
            model_id: "openhydra-qwen3.5-2b".into(),
            host: "192.168.1.10".into(),
            port: 50051,
            layer_start: 0,
            layer_end: 12,
            total_layers: 24,
            ..Default::default()
        };

        let bytes = record.to_cbor().unwrap();
        let decoded = PeerRecord::from_cbor(&bytes).unwrap();
        assert_eq!(decoded.peer_id, "test-peer");
        assert_eq!(decoded.layer_start, 0);
        assert_eq!(decoded.layer_end, 12);
        assert_eq!(decoded.total_layers, 24);
    }

    #[test]
    fn test_cbor_roundtrip_capability_fields() {
        // protocol.md §4 (M1.2): the new capability fields survive a CBOR round-trip.
        let record = PeerRecord {
            peer_id: "cap-peer".into(),
            model_id: "openhydra-qwen3.5-2b".into(),
            canonical_model_id: "qwen3.5/2b/fp16/5632a1b48425a5ae".into(),
            context_length: 32768,
            max_output_tokens: 4096,
            throughput_tok_s: 13.4,
            queue_depth: 2,
            hardware_class: "nvidia-t4".into(),
            ..Default::default()
        };
        let decoded = PeerRecord::from_cbor(&record.to_cbor().unwrap()).unwrap();
        assert_eq!(decoded.canonical_model_id, "qwen3.5/2b/fp16/5632a1b48425a5ae");
        assert_eq!(decoded.context_length, 32768);
        assert_eq!(decoded.max_output_tokens, 4096);
        assert_eq!(decoded.throughput_tok_s, 13.4);
        assert_eq!(decoded.queue_depth, 2);
        assert_eq!(decoded.hardware_class, "nvidia-t4");
    }

    #[test]
    fn test_cbor_backward_compat_missing_new_fields() {
        // An "old" record (pre-M1.2) lacks the §4 capability keys entirely. With
        // #[serde(default)] it must still decode, defaulting the new fields — so a
        // freshly-upgraded node can still read records announced by older peers.
        use ciborium::value::Value;
        let old = Value::Map(vec![
            (Value::Text("peer_id".into()), Value::Text("old-peer".into())),
            (Value::Text("model_id".into()), Value::Text("m".into())),
            (Value::Text("host".into()), Value::Text("10.0.0.9".into())),
            (Value::Text("port".into()), Value::Integer(50051u16.into())),
        ]);
        let mut buf = Vec::new();
        ciborium::ser::into_writer(&old, &mut buf).unwrap();

        let decoded = PeerRecord::from_cbor(&buf).unwrap();
        assert_eq!(decoded.peer_id, "old-peer");
        assert_eq!(decoded.canonical_model_id, ""); // all §4 fields defaulted
        assert_eq!(decoded.context_length, 0);
        assert_eq!(decoded.max_output_tokens, 0);
        assert_eq!(decoded.throughput_tok_s, 0.0);
        assert_eq!(decoded.queue_depth, 0);
        assert_eq!(decoded.hardware_class, "");
    }

    #[test]
    fn test_json_roundtrip() {
        let json = r#"{
            "peer_id": "mac-a",
            "model_id": "openhydra-qwen3.5-2b",
            "host": "10.0.0.1",
            "port": 50051,
            "layer_start": 0,
            "layer_end": 12,
            "total_layers": 24
        }"#;
        let record: PeerRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.peer_id, "mac-a");
        assert_eq!(record.nat_type, "unknown"); // default
        // §4 capability fields default to empty/zero when absent from the record.
        assert_eq!(record.canonical_model_id, "");
        assert_eq!(record.context_length, 0);
        assert_eq!(record.hardware_class, "");
    }
}

impl Default for PeerRecord {
    fn default() -> Self {
        Self {
            peer_id: String::new(),
            model_id: String::new(),
            host: String::new(),
            port: 0,
            operator_id: None,
            region: None,
            load_pct: 0.0,
            daemon_mode: "polite".into(),
            runtime_backend: "toy_cpu".into(),
            runtime_target: String::new(),
            runtime_model_id: String::new(),
            quantization_mode: String::new(),
            quantization_bits: 0,
            runtime_gpu_available: false,
            runtime_estimated_tokens_per_sec: 0.0,
            runtime_estimated_memory_mb: 0,
            canonical_model_id: String::new(),
            context_length: 0,
            max_output_tokens: 0,
            throughput_tok_s: 0.0,
            queue_depth: 0,
            hardware_class: String::new(),
            privacy_noise_variance: 0.0,
            reputation_score: 0.0,
            expert_tags: Vec::new(),
            expert_layer_indices: Vec::new(),
            layer_start: 0,
            layer_end: 0,
            total_layers: 0,
            nat_type: "unknown".into(),
            requires_relay: false,
            relay_address: String::new(),
            host_ipv6: String::new(),
            peer_public_key: String::new(),
            libp2p_peer_id: String::new(),
            public_key: String::new(),
            signature: String::new(),
            updated_unix_ms: 0,
        }
    }
}
