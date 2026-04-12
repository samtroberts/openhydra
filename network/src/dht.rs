//! Kademlia DHT operations for peer record storage and discovery.

use libp2p::kad;

use crate::types::PeerRecord;

/// Kademlia key prefix for OpenHydra model records.
const MODEL_KEY_PREFIX: &str = "/openhydra/model/";

/// Build a Kademlia record key for a peer's model announcement.
pub fn peer_record_key(model_id: &str, peer_id: &str) -> kad::RecordKey {
    kad::RecordKey::new(&format!("{MODEL_KEY_PREFIX}{model_id}/{peer_id}"))
}

/// Build a Kademlia record key for querying all peers of a model.
///
/// Uses the provider record pattern: peers "provide" a model_id key.
pub fn model_provider_key(model_id: &str) -> kad::RecordKey {
    kad::RecordKey::new(&format!("{MODEL_KEY_PREFIX}{model_id}"))
}

/// Encode a PeerRecord as a Kademlia record value (CBOR).
pub fn encode_record(record: &PeerRecord) -> Result<Vec<u8>, String> {
    record.to_cbor().map_err(|e| format!("cbor encode: {e}"))
}

/// Decode a PeerRecord from a Kademlia record value (CBOR).
pub fn decode_record(data: &[u8]) -> Result<PeerRecord, String> {
    PeerRecord::from_cbor(data).map_err(|e| format!("cbor decode: {e}"))
}

/// Extract the model_id from a Kademlia record key.
///
/// Key format: `/openhydra/model/{model_id}/{peer_id}`
pub fn parse_model_id_from_key(key: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(key).ok()?;
    let stripped = s.strip_prefix(MODEL_KEY_PREFIX)?;
    let slash_pos = stripped.find('/')?;
    Some(stripped[..slash_pos].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_format() {
        let key = peer_record_key("openhydra-qwen3.5-2b", "mac-a");
        let raw = key.as_ref();
        let s = std::str::from_utf8(raw).unwrap();
        assert_eq!(s, "/openhydra/model/openhydra-qwen3.5-2b/mac-a");
    }

    #[test]
    fn test_parse_model_id() {
        let key = b"/openhydra/model/openhydra-qwen3.5-2b/mac-a";
        assert_eq!(
            parse_model_id_from_key(key),
            Some("openhydra-qwen3.5-2b".to_string())
        );
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let record = PeerRecord {
            peer_id: "test".into(),
            model_id: "test-model".into(),
            host: "1.2.3.4".into(),
            port: 50051,
            layer_start: 0,
            layer_end: 12,
            total_layers: 24,
            ..Default::default()
        };
        let encoded = encode_record(&record).unwrap();
        let decoded = decode_record(&encoded).unwrap();
        assert_eq!(decoded.peer_id, "test");
        assert_eq!(decoded.layer_end, 12);
    }
}
