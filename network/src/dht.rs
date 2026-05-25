//! Kademlia DHT operations for peer record storage and discovery.

use libp2p::kad;

use crate::types::PeerRecord;

/// Kademlia key prefix for OpenHydra model records.
const MODEL_KEY_PREFIX: &str = "/openhydra/model/";

/// Build a Kademlia record key for a peer's model announcement.
///
/// Uses the libp2p PeerId (base58) in the key, NOT the OpenHydra peer_id,
/// so that records can be looked up by provider PeerId after `get_providers`.
pub fn peer_record_key(model_id: &str, libp2p_peer_id: &str) -> kad::RecordKey {
    kad::RecordKey::new(&format!("{MODEL_KEY_PREFIX}{model_id}/{libp2p_peer_id}"))
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

/// Compute canonical bytes for signing a PeerRecord.
///
/// Matches the Python canonical format:
/// `json.dumps({"host": host, "model_id": model_id, "peer_id": peer_id, "port": port}, sort_keys=True)`
/// Keys are already alphabetical, so the JSON is deterministic.
pub fn canonical_bytes(record: &PeerRecord) -> Vec<u8> {
    format!(
        r#"{{"host": "{}", "model_id": "{}", "peer_id": "{}", "port": {}}}"#,
        record.host, record.model_id, record.peer_id, record.port,
    )
    .into_bytes()
}

/// Sign a PeerRecord with the given keypair and populate its `signature`
/// and `public_key` fields.  Returns the mutated record (Task 6.2).
pub fn sign_peer_record(
    record: &PeerRecord,
    keypair: &libp2p::identity::Keypair,
) -> Result<PeerRecord, String> {
    let canonical = canonical_bytes(record);
    let sig_bytes = keypair
        .sign(&canonical)
        .map_err(|e| format!("sign failed: {e}"))?;
    let ed25519_pk = keypair
        .public()
        .try_into_ed25519()
        .map_err(|e| format!("not ed25519: {e}"))?;
    let mut signed = record.clone();
    signed.signature = base64_urlsafe_encode(&sig_bytes);
    signed.public_key = hex::encode(ed25519_pk.to_bytes());
    signed.libp2p_peer_id =
        libp2p::PeerId::from_public_key(&keypair.public()).to_string();
    Ok(signed)
}

/// URL-safe base64 encoding (matches Python's `base64.urlsafe_b64encode`).
fn base64_urlsafe_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE.encode(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_format() {
        let key = peer_record_key("openhydra-qwen3.5-2b", "12D3KooWEL5wELabcdef");
        let raw = key.as_ref();
        let s = std::str::from_utf8(raw).unwrap();
        assert_eq!(s, "/openhydra/model/openhydra-qwen3.5-2b/12D3KooWEL5wELabcdef");
    }

    #[test]
    fn test_parse_model_id() {
        let key = b"/openhydra/model/openhydra-qwen3.5-2b/12D3KooWEL5wELabcdef";
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
