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

    // Privacy
    #[serde(default)]
    pub privacy_noise_variance: f64,

    // Reputation
    #[serde(default)]
    pub reputation_score: f64,
    #[serde(default)]
    pub staked_balance: f64,

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

    // Identity keys
    #[serde(default)]
    pub peer_public_key: String,
    /// libp2p PeerId (base58 multihash) — new field for Rust networking.
    #[serde(default)]
    pub libp2p_peer_id: String,

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
    pub external_ip: String,
    pub external_port: u16,
    pub is_public: bool,
}

/// A discovered peer — returned by `P2PNode.discover()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPeer {
    pub peer_id: String,
    pub libp2p_peer_id: String,
    pub host: String,
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
            privacy_noise_variance: 0.0,
            reputation_score: 0.0,
            staked_balance: 0.0,
            expert_tags: Vec::new(),
            expert_layer_indices: Vec::new(),
            layer_start: 0,
            layer_end: 0,
            total_layers: 0,
            nat_type: "unknown".into(),
            requires_relay: false,
            relay_address: String::new(),
            peer_public_key: String::new(),
            libp2p_peer_id: String::new(),
            updated_unix_ms: 0,
        }
    }
}
