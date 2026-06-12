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

/// URL-safe base64 decoding (matches Python's `base64.urlsafe_b64decode`).
fn base64_urlsafe_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE
        .decode(s)
        .map_err(|e| format!("base64 decode failed: {e}"))
}

/// Verify a `PeerRecord`'s Ed25519 signature and bind its key to the advertised
/// libp2p peer id.  Returns `Ok(())` only when the record is authentic.
///
/// SECURITY: this is the missing read-side counterpart to `sign_peer_record`.
/// Untrusted DHT records (Kademlia GetRecord / provider-store reads) MUST pass
/// this before any field is trusted — otherwise any peer can publish records
/// with attacker-chosen `host`/`port`/`layer_*`/`libp2p_peer_id` (DHT
/// poisoning + routing-table address injection).  Never panics on malformed
/// input — all decode failures return `Err`.
pub fn verify_peer_record(record: &PeerRecord) -> Result<(), String> {
    if record.signature.is_empty() || record.public_key.is_empty() {
        return Err("missing signature or public_key".into());
    }
    // Decode the ed25519 public key (hex, matches sign_peer_record).
    let pk_bytes =
        hex::decode(&record.public_key).map_err(|e| format!("bad public_key hex: {e}"))?;
    let ed_pk = libp2p::identity::ed25519::PublicKey::try_from_bytes(&pk_bytes)
        .map_err(|e| format!("bad ed25519 public_key: {e}"))?;
    // Bind the key to the advertised libp2p_peer_id so a valid signature from
    // identity A cannot be replayed under identity B's peer id.
    if !record.libp2p_peer_id.is_empty() {
        let pubkey = libp2p::identity::PublicKey::from(ed_pk.clone());
        let derived = libp2p::PeerId::from_public_key(&pubkey).to_string();
        if record.libp2p_peer_id != derived {
            return Err(format!(
                "peer_id mismatch: record={} derived={derived}",
                record.libp2p_peer_id
            ));
        }
    }
    let sig = base64_urlsafe_decode(&record.signature)?;
    if ed_pk.verify(&canonical_bytes(record), &sig) {
        Ok(())
    } else {
        Err("signature verification failed".into())
    }
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

    #[test]
    fn test_sign_verify_roundtrip() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let record = PeerRecord {
            peer_id: "oh-peer".into(),
            model_id: "test-model".into(),
            host: "1.2.3.4".into(),
            port: 50051,
            ..Default::default()
        };
        let signed = sign_peer_record(&record, &kp).unwrap();
        // A genuine signed record verifies.
        verify_peer_record(&signed).unwrap();
    }

    #[test]
    fn test_verify_rejects_tampered_field() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut signed = sign_peer_record(
            &PeerRecord { host: "1.2.3.4".into(), port: 50051, ..Default::default() },
            &kp,
        )
        .unwrap();
        // Attacker rewrites host after signing → must fail.
        signed.host = "6.6.6.6".into();
        assert!(verify_peer_record(&signed).is_err());
    }

    #[test]
    fn test_verify_rejects_unsigned_and_mismatched_peer_id() {
        // Unsigned record (no signature/public_key) is rejected.
        let unsigned = PeerRecord { host: "1.2.3.4".into(), port: 1, ..Default::default() };
        assert!(verify_peer_record(&unsigned).is_err());

        // Valid signature from key A replayed under a different peer id → reject.
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut signed =
            sign_peer_record(&PeerRecord { port: 1, ..Default::default() }, &kp).unwrap();
        signed.libp2p_peer_id = "12D3KooWFakePeerIdThatDoesNotMatchTheKey".into();
        assert!(verify_peer_record(&signed).is_err());
    }
}
