// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Device attestation — the pure core of the attested-engine tier (ATTESTED_ENGINE_TIER_PLAN_v1,
//! Phase 0).
//!
//! A provider's device signs an [`AttestationClaim`] — what it is (chip, OS, SIP / Secure Boot, the
//! running binary's hash) and what it serves (model id + weight hash) — over a consumer-supplied
//! nonce, with a P-256 key (on Apple Silicon, a Secure Enclave key). The consumer checks it with
//! [`verify_attestation`]. Verification is pure Rust (RustCrypto `p256`), so a consumer on any OS can
//! check an Apple SE signature.
//!
//! ## What a verified attestation does and does not prove (H1 honest labeling)
//!
//! A bare P-256 signature proves only that the signer holds *some* private key and answered *this*
//! nonce. A remote verifier **cannot** tell a Secure Enclave key from a software key without an
//! Apple-signed attestation chain (Managed Device Attestation via MDM, or App Attest), and every
//! field of the claim — including `binary_hash` and `sip_enabled` — is the device's own word until
//! such a chain vouches for the key. So [`verify_attestation`] yields
//! [`DeviceTrust::SelfSigned`] (darkbloom's `self_signed` level): a live, nonce-bound, key-continuous
//! identity plus a *claimed* device profile. It does **not** lift the authenticity tier.
//!
//! The A3 lift is a separate pure rule, [`attested_methods`]: `hardware_attested` is set **only**
//! when the key is vouched for by a hardware chain ([`DeviceTrust::HardwareChain`], Phase 2), the
//! model hash was computed by the in-process engine (`active_model_hash`, Phase 1), **and** that hash
//! matches a trusted expected value (the reference anchor). Phase 0 can satisfy none of these, by
//! design.

use p256::ecdsa::signature::Verifier as _;
use p256::ecdsa::{Signature, VerifyingKey};

use crate::trust::MethodFlags;

/// Domain separator for the attestation signing preimage.
pub const ATTEST_DOMAIN: &[u8] = b"openhydra/attest/v1";
/// The only attestation format version this build produces and accepts. 2: the claim names the
/// provider it is from (`provider_peer`).
pub const ATTEST_VERSION: u8 = 2;
/// Domain separator for a [`ServeAttestation`]'s signing preimage.
pub const SERVE_ATTEST_DOMAIN: &[u8] = b"openhydra/attest-serve/v1";
/// Length of an uncompressed SEC1 P-256 public key (`0x04 || X || Y`) — the form Apple's
/// `SecKeyCopyExternalRepresentation` returns for an EC public key.
pub const P256_SEC1_UNCOMPRESSED_LEN: usize = 65;
/// Upper bound on a DER-encoded P-256 ECDSA signature (two 33-byte INTEGERs + framing).
pub const MAX_SIG_DER_LEN: usize = 72;

/// What the attesting device claims. Every field is bound into the signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationClaim {
    /// Hardware model, e.g. `"Apple M3 Pro"`.
    pub chip: String,
    /// OS build, e.g. `"25G56"`.
    pub os_build: String,
    /// System Integrity Protection enabled (claimed).
    pub sip_enabled: bool,
    /// Secure Boot at full security (claimed).
    pub secure_boot_enabled: bool,
    /// SHA-256 of the running `openhydra-agent` binary (claimed; checked against a trusted list).
    pub binary_hash: [u8; 32],
    /// Canonical model id the device says it serves.
    pub model_id: String,
    /// The libp2p peer id of the provider answering the challenge. A consumer requires it to be the
    /// peer it challenged: otherwise a provider could relay another (honest, attested) provider's
    /// answer as its own.
    pub provider_peer: String,
    /// Weight digest reported by an **external** engine (e.g. Ollama's `sha256-…` blob digest). The
    /// provider's word about a process it doesn't control — never a basis for a tier lift.
    pub model_digest: Option<[u8; 32]>,
    /// Hash of the weights the **in-process** engine actually loaded (Phase 1+). `None` while the
    /// engine is external.
    pub active_model_hash: Option<[u8; 32]>,
    /// X25519 key the consumer seals prompts to (Phase 3). `None` until sealing exists.
    pub session_pubkey: Option<[u8; 32]>,
    /// The consumer's challenge nonce.
    pub nonce: [u8; 32],
    /// When the device produced the claim (unix ms).
    pub ts_unix_ms: u64,
}

/// A signed claim: the claim, the signer's public key, and a DER ECDSA-P256-SHA256 signature over
/// [`signing_bytes`](Self::signing_bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attestation {
    pub claim: AttestationClaim,
    /// Uncompressed SEC1 P-256 public key.
    pub signing_key: [u8; P256_SEC1_UNCOMPRESSED_LEN],
    /// DER-encoded ECDSA signature (Apple's `kSecKeyAlgorithmECDSASignatureMessageX962SHA256`).
    pub signature_der: Vec<u8>,
}

/// How far the verifier trusts the attesting device's key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviceTrust {
    /// A valid, fresh, nonce-bound signature from a P-256 key, with no hardware chain behind the key.
    /// The claim's device fields are the device's own word. The only level Phase 0 produces.
    SelfSigned,
    /// The key is vouched for by an Apple-signed hardware attestation chain (Managed Device
    /// Attestation / App Attest). **No verifier produces this yet** — reserved for Phase 2 so the
    /// A3 rule in [`attested_methods`] can be written and tested now.
    HardwareChain,
}

impl DeviceTrust {
    /// Stable display label (status API / badges), mirroring darkbloom's trust levels.
    pub fn label(self) -> &'static str {
        match self {
            DeviceTrust::SelfSigned => "self_signed",
            DeviceTrust::HardwareChain => "hardware",
        }
    }
}

/// Why an attestation was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestError {
    /// Wire bytes malformed or truncated.
    Malformed,
    /// A format version this build doesn't understand.
    UnsupportedVersion,
    /// The public key isn't a valid P-256 point.
    BadKey,
    /// The signature doesn't parse or doesn't verify.
    BadSignature,
    /// The nonce isn't the one the consumer issued.
    NonceMismatch,
    /// The claim is older than the policy's `max_age_ms`.
    Stale,
    /// The claim is dated further in the future than the allowed clock skew.
    FromFuture,
    /// The device claims SIP is off.
    SipDisabled,
    /// The device claims Secure Boot is off.
    SecureBootDisabled,
    /// `binary_hash` isn't on the trusted-release list (an empty list trusts nothing).
    UntrustedBinary,
    /// The key differs from the one pinned for this provider.
    KeyChanged,
    /// The claim names another provider than the one challenged (a relayed attestation).
    ProviderMismatch,
}

/// The consumer's acceptance policy for one verification.
#[derive(Debug, Clone, Copy)]
pub struct AttestPolicy<'a> {
    /// The nonce the consumer issued for this challenge.
    pub expected_nonce: [u8; 32],
    /// The libp2p peer id of the provider the consumer challenged (the authenticated responder).
    pub expected_provider: &'a str,
    /// The consumer's clock (unix ms).
    pub now_unix_ms: u64,
    /// Oldest acceptable claim age.
    pub max_age_ms: u64,
    /// How far in the future a claim may be dated (clock skew tolerance).
    pub max_skew_ms: u64,
    /// SHA-256 hashes of trusted OpenHydra release binaries. **Empty ⇒ nothing is trusted.** Without
    /// a hardware chain this is release hygiene (it rejects honest-but-unofficial builds), not a
    /// defence: a modified binary can report any hash.
    pub trusted_binaries: &'a [[u8; 32]],
    /// The key previously seen for this provider, if pinned (trust-on-first-use continuity).
    pub pinned_key: Option<&'a [u8; P256_SEC1_UNCOMPRESSED_LEN]>,
}

impl AttestationClaim {
    /// The claim's canonical encoding: version, then fields in a fixed order — strings u32-LE
    /// length-prefixed, optionals a `0`/`1` presence byte followed by the value when present.
    /// Deterministic and unambiguous.
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(
            1 + 12 + self.chip.len() + self.os_build.len() + self.model_id.len() + 2 + 32 + 3 * 33 + 32 + 8,
        );
        b.push(ATTEST_VERSION);
        put_str(&mut b, &self.chip);
        put_str(&mut b, &self.os_build);
        b.push(self.sip_enabled as u8);
        b.push(self.secure_boot_enabled as u8);
        b.extend_from_slice(&self.binary_hash);
        put_str(&mut b, &self.model_id);
        put_str(&mut b, &self.provider_peer);
        put_opt32(&mut b, self.model_digest.as_ref());
        put_opt32(&mut b, self.active_model_hash.as_ref());
        put_opt32(&mut b, self.session_pubkey.as_ref());
        b.extend_from_slice(&self.nonce);
        b.extend_from_slice(&self.ts_unix_ms.to_le_bytes());
        b
    }

    fn decode(r: &mut Reader<'_>) -> Result<Self, AttestError> {
        if r.u8()? != ATTEST_VERSION {
            return Err(AttestError::UnsupportedVersion);
        }
        Ok(Self {
            chip: r.string()?,
            os_build: r.string()?,
            sip_enabled: r.flag()?,
            secure_boot_enabled: r.flag()?,
            binary_hash: r.array32()?,
            model_id: r.string()?,
            provider_peer: r.string()?,
            model_digest: r.opt32()?,
            active_model_hash: r.opt32()?,
            session_pubkey: r.opt32()?,
            nonce: r.array32()?,
            ts_unix_ms: r.u64()?,
        })
    }
}

impl Attestation {
    /// The bytes the device signs: `ATTEST_DOMAIN || claim.encode() || signing_key`. Binding the
    /// public key in prevents presenting one key's signature as another's.
    pub fn signing_bytes(claim: &AttestationClaim, signing_key: &[u8; P256_SEC1_UNCOMPRESSED_LEN]) -> Vec<u8> {
        let claim_bytes = claim.encode();
        let mut b = Vec::with_capacity(ATTEST_DOMAIN.len() + claim_bytes.len() + signing_key.len());
        b.extend_from_slice(ATTEST_DOMAIN);
        b.extend_from_slice(&claim_bytes);
        b.extend_from_slice(signing_key);
        b
    }

    /// Wire encoding: `claim.encode() || signing_key || u16-LE sig length || signature_der`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = self.claim.encode();
        b.extend_from_slice(&self.signing_key);
        b.extend_from_slice(&(self.signature_der.len() as u16).to_le_bytes());
        b.extend_from_slice(&self.signature_der);
        b
    }

    /// Decode [`to_bytes`](Self::to_bytes) output. Never panics on arbitrary input; rejects trailing
    /// bytes and oversized signatures.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AttestError> {
        let mut r = Reader { buf: bytes };
        let claim = AttestationClaim::decode(&mut r)?;
        let signing_key: [u8; P256_SEC1_UNCOMPRESSED_LEN] =
            r.take(P256_SEC1_UNCOMPRESSED_LEN)?.try_into().map_err(|_| AttestError::Malformed)?;
        let sig_len = u16::from_le_bytes(r.take(2)?.try_into().map_err(|_| AttestError::Malformed)?) as usize;
        if sig_len > MAX_SIG_DER_LEN {
            return Err(AttestError::Malformed);
        }
        let signature_der = r.take(sig_len)?.to_vec();
        if !r.buf.is_empty() {
            return Err(AttestError::Malformed);
        }
        Ok(Self { claim, signing_key, signature_der })
    }
}

/// Verify an attestation against the consumer's policy. The signature is checked **first**, so no
/// claim field is trusted before it is authenticated. Returns the [`DeviceTrust`] level reached —
/// always [`SelfSigned`](DeviceTrust::SelfSigned) in Phase 0 (no hardware-chain verifier exists).
pub fn verify_attestation(att: &Attestation, policy: &AttestPolicy<'_>) -> Result<DeviceTrust, AttestError> {
    let key = VerifyingKey::from_sec1_bytes(&att.signing_key).map_err(|_| AttestError::BadKey)?;
    let sig = Signature::from_der(&att.signature_der).map_err(|_| AttestError::BadSignature)?;
    key.verify(&Attestation::signing_bytes(&att.claim, &att.signing_key), &sig)
        .map_err(|_| AttestError::BadSignature)?;

    let c = &att.claim;
    if c.nonce != policy.expected_nonce {
        return Err(AttestError::NonceMismatch);
    }
    if c.provider_peer != policy.expected_provider {
        return Err(AttestError::ProviderMismatch);
    }
    if c.ts_unix_ms > policy.now_unix_ms.saturating_add(policy.max_skew_ms) {
        return Err(AttestError::FromFuture);
    }
    if c.ts_unix_ms.saturating_add(policy.max_age_ms) < policy.now_unix_ms {
        return Err(AttestError::Stale);
    }
    if !c.sip_enabled {
        return Err(AttestError::SipDisabled);
    }
    if !c.secure_boot_enabled {
        return Err(AttestError::SecureBootDisabled);
    }
    if !policy.trusted_binaries.contains(&c.binary_hash) {
        return Err(AttestError::UntrustedBinary);
    }
    if let Some(pinned) = policy.pinned_key {
        if pinned != &att.signing_key {
            return Err(AttestError::KeyChanged);
        }
    }
    Ok(DeviceTrust::SelfSigned)
}

/// A serve signed by the attested key: the device's statement that **this** serve (its receipt nonce)
/// produced output with this hash, from the engine whose state the attestation describes. A device
/// attestation alone says what the machine was at challenge time, not that a given serve came from
/// it; a serve attestation ties the two together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeAttestation {
    /// The serve's receipt nonce (what the consumer committed to before the serve).
    pub serve_nonce: [u8; 16],
    /// SHA-256 of the output the consumer received.
    pub output_hash: [u8; 32],
    /// DER ECDSA-P256-SHA256 over [`serve_signing_bytes`], by the attestation's key.
    pub signature_der: Vec<u8>,
}

/// The bytes the attested key signs for a serve: `SERVE_ATTEST_DOMAIN || serve_nonce || output_hash
/// || model_id (u32-LE length-prefixed)`.
pub fn serve_signing_bytes(serve_nonce: &[u8; 16], output_hash: &[u8; 32], model_id: &str) -> Vec<u8> {
    let mut b = Vec::with_capacity(SERVE_ATTEST_DOMAIN.len() + 16 + 32 + 4 + model_id.len());
    b.extend_from_slice(SERVE_ATTEST_DOMAIN);
    b.extend_from_slice(serve_nonce);
    b.extend_from_slice(output_hash);
    put_str(&mut b, model_id);
    b
}

/// What a consumer knows about one serve it is crediting: the nonce it committed and the hash of the
/// output it received. A [`ServeAttestation`] must match both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServeEvidence {
    pub serve_nonce: [u8; 16],
    pub output_hash: [u8; 32],
}

/// Whether `serve` is a valid signature by `att`'s key over this serve (`evidence`) of `att`'s model.
pub fn verify_serve_attestation(att: &Attestation, serve: &ServeAttestation, evidence: &ServeEvidence) -> bool {
    if serve.serve_nonce != evidence.serve_nonce || serve.output_hash != evidence.output_hash {
        return false;
    }
    let Ok(key) = VerifyingKey::from_sec1_bytes(&att.signing_key) else { return false };
    let Ok(sig) = Signature::from_der(&serve.signature_der) else { return false };
    key.verify(&serve_signing_bytes(&serve.serve_nonce, &serve.output_hash, &att.claim.model_id), &sig).is_ok()
}

/// The Proof-of-Inference methods a verified attestation establishes **for one serve** — the single
/// place the A3 lift is decided. `hardware_attested` is set only when **all four** hold:
///
/// 1. the key is vouched for by a hardware chain ([`DeviceTrust::HardwareChain`]);
/// 2. the claim carries an in-process `active_model_hash` (an external engine's `model_digest` never
///    counts — it is the provider's word about a process it doesn't control);
/// 3. that hash equals the trusted expected hash for the canonical model (the reference anchor);
///    `None` — no anchor — never lifts;
/// 4. the serve being credited is signed by the same attested key ([`verify_serve_attestation`]):
///    the device's state at challenge time says nothing about a serve it didn't sign.
///
/// Anything less returns no methods: a self-signed attestation is identity + liveness, not proof of
/// which model ran.
pub fn attested_methods(
    trust: DeviceTrust,
    att: &Attestation,
    expected_model_hash: Option<&[u8; 32]>,
    serve: Option<(&ServeAttestation, &ServeEvidence)>,
) -> MethodFlags {
    let anchored = matches!(
        (att.claim.active_model_hash.as_ref(), expected_model_hash),
        (Some(active), Some(expected)) if active == expected
    );
    let serve_bound = serve.is_some_and(|(s, e)| verify_serve_attestation(att, s, e));
    MethodFlags {
        hardware_attested: trust == DeviceTrust::HardwareChain && anchored && serve_bound,
        ..MethodFlags::default()
    }
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u32).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

fn put_opt32(b: &mut Vec<u8>, v: Option<&[u8; 32]>) {
    match v {
        Some(x) => {
            b.push(1);
            b.extend_from_slice(x);
        }
        None => b.push(0),
    }
}

/// Bounds-checked cursor; every read returns `Malformed` instead of panicking.
struct Reader<'a> {
    buf: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], AttestError> {
        if self.buf.len() < n {
            return Err(AttestError::Malformed);
        }
        let (head, tail) = self.buf.split_at(n);
        self.buf = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8, AttestError> {
        Ok(self.take(1)?[0])
    }
    fn flag(&mut self) -> Result<bool, AttestError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(AttestError::Malformed),
        }
    }
    fn u64(&mut self) -> Result<u64, AttestError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().map_err(|_| AttestError::Malformed)?))
    }
    fn array32(&mut self) -> Result<[u8; 32], AttestError> {
        self.take(32)?.try_into().map_err(|_| AttestError::Malformed)
    }
    fn opt32(&mut self) -> Result<Option<[u8; 32]>, AttestError> {
        if self.flag()? {
            Ok(Some(self.array32()?))
        } else {
            Ok(None)
        }
    }
    fn string(&mut self) -> Result<String, AttestError> {
        let len = u32::from_le_bytes(self.take(4)?.try_into().map_err(|_| AttestError::Malformed)?) as usize;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|_| AttestError::Malformed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trust::{authenticity_tier_from, AuthenticityTier};
    use p256::ecdsa::signature::Signer as _;
    use p256::ecdsa::SigningKey;

    const NOW: u64 = 1_790_000_000_000;
    const BIN: [u8; 32] = [0xB1; 32];
    const NONCE: [u8; 32] = [0x7A; 32];

    fn signer(seed: u8) -> SigningKey {
        SigningKey::from_slice(&[seed; 32]).unwrap()
    }

    fn sec1(sk: &SigningKey) -> [u8; P256_SEC1_UNCOMPRESSED_LEN] {
        sk.verifying_key().to_encoded_point(false).as_bytes().try_into().unwrap()
    }

    fn claim() -> AttestationClaim {
        AttestationClaim {
            chip: "Apple M3 Pro".into(),
            os_build: "25G56".into(),
            sip_enabled: true,
            secure_boot_enabled: true,
            binary_hash: BIN,
            model_id: "qwen2.5-0.5b".into(),
            provider_peer: PROVIDER.into(),
            model_digest: Some([0xD1; 32]),
            active_model_hash: None,
            session_pubkey: None,
            nonce: NONCE,
            ts_unix_ms: NOW - 1_000,
        }
    }

    const PROVIDER: &str = "12D3KooWProvider";

    fn sign(sk: &SigningKey, claim: AttestationClaim) -> Attestation {
        let key = sec1(sk);
        let sig: Signature = sk.sign(&Attestation::signing_bytes(&claim, &key));
        Attestation { claim, signing_key: key, signature_der: sig.to_der().as_bytes().to_vec() }
    }

    fn policy<'a>(trusted: &'a [[u8; 32]]) -> AttestPolicy<'a> {
        AttestPolicy {
            expected_nonce: NONCE,
            expected_provider: PROVIDER,
            now_unix_ms: NOW,
            max_age_ms: 60_000,
            max_skew_ms: 5_000,
            trusted_binaries: trusted,
            pinned_key: None,
        }
    }

    #[test]
    fn valid_attestation_verifies_as_self_signed_only() {
        let att = sign(&signer(1), claim());
        assert_eq!(verify_attestation(&att, &policy(&[BIN])), Ok(DeviceTrust::SelfSigned));
    }

    #[test]
    fn wire_round_trip_preserves_verification() {
        let mut c = claim();
        c.active_model_hash = Some([0xA1; 32]);
        c.session_pubkey = Some([0x5E; 32]);
        let att = sign(&signer(2), c);
        let decoded = Attestation::from_bytes(&att.to_bytes()).unwrap();
        assert_eq!(decoded, att);
        assert_eq!(verify_attestation(&decoded, &policy(&[BIN])), Ok(DeviceTrust::SelfSigned));
    }

    #[test]
    fn any_tampered_claim_field_breaks_the_signature() {
        let att = sign(&signer(3), claim());
        let tampers: Vec<fn(&mut AttestationClaim)> = vec![
            |c| c.chip.push('X'),
            |c| c.os_build.push('X'),
            |c| c.binary_hash[0] ^= 1,
            |c| c.model_id.push('X'),
            |c| c.provider_peer.push('X'),
            |c| c.model_digest = None,
            |c| c.active_model_hash = Some([0; 32]),
            |c| c.session_pubkey = Some([0; 32]),
            |c| c.ts_unix_ms += 1,
        ];
        for tamper in tampers {
            let mut bad = att.clone();
            tamper(&mut bad.claim);
            assert_eq!(verify_attestation(&bad, &policy(&[BIN])), Err(AttestError::BadSignature));
        }
    }

    #[test]
    fn a_signature_cannot_be_rebound_to_another_key() {
        // Attacker swaps in their own public key alongside the victim's signature.
        let mut att = sign(&signer(4), claim());
        att.signing_key = sec1(&signer(5));
        assert_eq!(verify_attestation(&att, &policy(&[BIN])), Err(AttestError::BadSignature));
    }

    #[test]
    fn policy_rejections() {
        let trusted = [BIN];
        let sk = signer(6);

        let mut c = claim();
        c.nonce = [0; 32];
        assert_eq!(verify_attestation(&sign(&sk, c), &policy(&trusted)), Err(AttestError::NonceMismatch));

        let mut c = claim();
        c.ts_unix_ms = NOW - 60_001;
        assert_eq!(verify_attestation(&sign(&sk, c), &policy(&trusted)), Err(AttestError::Stale));

        let mut c = claim();
        c.ts_unix_ms = NOW + 5_001;
        assert_eq!(verify_attestation(&sign(&sk, c), &policy(&trusted)), Err(AttestError::FromFuture));

        let mut c = claim();
        c.sip_enabled = false;
        assert_eq!(verify_attestation(&sign(&sk, c), &policy(&trusted)), Err(AttestError::SipDisabled));

        let mut c = claim();
        c.secure_boot_enabled = false;
        assert_eq!(verify_attestation(&sign(&sk, c), &policy(&trusted)), Err(AttestError::SecureBootDisabled));

        let att = sign(&sk, claim());
        assert_eq!(verify_attestation(&att, &policy(&[[0; 32]])), Err(AttestError::UntrustedBinary));
        // Fail closed: an empty trusted list trusts nothing.
        assert_eq!(verify_attestation(&att, &policy(&[])), Err(AttestError::UntrustedBinary));
    }

    #[test]
    fn freshness_boundaries_are_inclusive() {
        let sk = signer(7);
        let mut c = claim();
        c.ts_unix_ms = NOW - 60_000;
        assert!(verify_attestation(&sign(&sk, c), &policy(&[BIN])).is_ok());
        let mut c = claim();
        c.ts_unix_ms = NOW + 5_000;
        assert!(verify_attestation(&sign(&sk, c), &policy(&[BIN])).is_ok());
    }

    #[test]
    fn pinned_key_detects_a_key_change() {
        let trusted = [BIN];
        let att = sign(&signer(8), claim());
        let pinned = att.signing_key;
        let mut p = policy(&trusted);
        p.pinned_key = Some(&pinned);
        assert_eq!(verify_attestation(&att, &p), Ok(DeviceTrust::SelfSigned));
        let other = sign(&signer(9), claim());
        assert_eq!(verify_attestation(&other, &p), Err(AttestError::KeyChanged));
    }

    #[test]
    fn invalid_key_and_signature_bytes_are_rejected() {
        let mut att = sign(&signer(10), claim());
        att.signing_key = [0x04; P256_SEC1_UNCOMPRESSED_LEN];
        assert_eq!(verify_attestation(&att, &policy(&[BIN])), Err(AttestError::BadKey));

        let mut att = sign(&signer(10), claim());
        att.signature_der = vec![0x30, 0x00];
        assert_eq!(verify_attestation(&att, &policy(&[BIN])), Err(AttestError::BadSignature));
    }

    #[test]
    fn decoder_never_panics_and_rejects_malformed_input() {
        let bytes = sign(&signer(11), claim()).to_bytes();
        // Every truncation fails cleanly.
        for n in 0..bytes.len() {
            assert!(Attestation::from_bytes(&bytes[..n]).is_err());
        }
        // Trailing garbage is rejected.
        let mut long = bytes.clone();
        long.push(0);
        assert_eq!(Attestation::from_bytes(&long), Err(AttestError::Malformed));
        // Unknown versions, the earlier one (no provider in the claim) included.
        for version in [ATTEST_VERSION - 1, ATTEST_VERSION + 1] {
            let mut v = bytes.clone();
            v[0] = version;
            assert_eq!(Attestation::from_bytes(&v), Err(AttestError::UnsupportedVersion));
        }
        // Huge string length prefix, bad presence flag, oversized signature length.
        assert!(Attestation::from_bytes(&[ATTEST_VERSION, 0xFF, 0xFF, 0xFF, 0xFF]).is_err());
        let mut flag = bytes.clone();
        let flag_at = 1 + 4 + "Apple M3 Pro".len() + 4 + "25G56".len();
        flag[flag_at] = 7;
        assert_eq!(Attestation::from_bytes(&flag), Err(AttestError::Malformed));
        let mut sig = bytes.clone();
        let len_at = bytes.len() - sign(&signer(11), claim()).signature_der.len() - 2;
        sig[len_at..len_at + 2].copy_from_slice(&(MAX_SIG_DER_LEN as u16 + 1).to_le_bytes());
        assert_eq!(Attestation::from_bytes(&sig), Err(AttestError::Malformed));
        // Arbitrary byte flips never panic.
        for i in 0..bytes.len() {
            let mut f = bytes.clone();
            f[i] ^= 0xFF;
            let _ = Attestation::from_bytes(&f);
        }
    }

    // ── the H1 rule: what may lift the authenticity tier ──

    #[test]
    fn a_relayed_attestation_names_the_wrong_provider() {
        // A provider forwarding an honest provider's challenge answers with a claim naming the honest
        // one: valid signature, but not from the peer the consumer challenged.
        let att = sign(&signer(2), claim());
        let mut p = policy(&[BIN]);
        p.expected_provider = "12D3KooWRelayer";
        assert_eq!(verify_attestation(&att, &p), Err(AttestError::ProviderMismatch));
    }

    /// The serve signed by `sk` (output of `model`).
    fn serve_sig(sk: &SigningKey, evidence: &ServeEvidence, model: &str) -> ServeAttestation {
        let sig: Signature = sk.sign(&serve_signing_bytes(&evidence.serve_nonce, &evidence.output_hash, model));
        ServeAttestation {
            serve_nonce: evidence.serve_nonce,
            output_hash: evidence.output_hash,
            signature_der: sig.to_der().as_bytes().to_vec(),
        }
    }

    #[test]
    fn a_verified_self_signed_attestation_never_lifts_the_tier() {
        let sk = signer(5);
        let mut c = claim();
        c.active_model_hash = Some([0xA1; 32]);
        let att = sign(&sk, c);
        let expected = [0xA1; 32];
        let ev = ServeEvidence { serve_nonce: [3; 16], output_hash: [4; 32] };
        let serve = serve_sig(&sk, &ev, "qwen2.5-0.5b");
        // Even with a matching in-process hash, an anchor and a signed serve, a self-signed key earns nothing.
        let m = attested_methods(DeviceTrust::SelfSigned, &att, Some(&expected), Some((&serve, &ev)));
        assert_eq!(m, MethodFlags::default());
        assert_eq!(authenticity_tier_from(0.0, 0.0, 0, m), AuthenticityTier::SelfReported);
    }

    #[test]
    fn hardware_chain_lifts_to_a3_only_with_an_anchored_hash_and_a_signed_serve() {
        let sk = signer(6);
        let expected = [0xA1; 32];
        let ev = ServeEvidence { serve_nonce: [3; 16], output_hash: [4; 32] };
        let with_hash = |hash: Option<[u8; 32]>, digest: Option<[u8; 32]>| {
            let mut c = claim();
            c.active_model_hash = hash;
            c.model_digest = digest;
            sign(&sk, c)
        };
        let serve = serve_sig(&sk, &ev, "qwen2.5-0.5b");
        let hw = |att: &Attestation, anchor: Option<&[u8; 32]>, s: Option<(&ServeAttestation, &ServeEvidence)>| {
            attested_methods(DeviceTrust::HardwareChain, att, anchor, s).hardware_attested
        };

        // External engine only (model_digest equal to the anchor, no active_model_hash): no lift.
        assert!(!hw(&with_hash(None, Some(expected)), Some(&expected), Some((&serve, &ev))));
        // In-process hash but no anchor, or one that doesn't match (substituted weights): no lift.
        assert!(!hw(&with_hash(Some(expected), None), None, Some((&serve, &ev))));
        assert!(!hw(&with_hash(Some([0xEE; 32]), None), Some(&expected), Some((&serve, &ev))));
        // Anchored, but the serve isn't signed by the attested key, isn't this serve, or is missing.
        let att = with_hash(Some(expected), None);
        assert!(!hw(&att, Some(&expected), None), "no serve signature");
        let other_key = serve_sig(&signer(9), &ev, "qwen2.5-0.5b");
        assert!(!hw(&att, Some(&expected), Some((&other_key, &ev))), "signed by another key");
        let other_serve = ServeEvidence { serve_nonce: [8; 16], ..ev };
        assert!(!hw(&att, Some(&expected), Some((&serve, &other_serve))), "another serve's signature");
        let other_model = serve_sig(&sk, &ev, "another-model");
        assert!(!hw(&att, Some(&expected), Some((&other_model, &ev))), "signed for another model");

        // All four conditions: A3.
        let m = attested_methods(DeviceTrust::HardwareChain, &att, Some(&expected), Some((&serve, &ev)));
        assert!(m.hardware_attested);
        assert_eq!(authenticity_tier_from(0.0, 0.0, 0, m), AuthenticityTier::HardwareAttested);
    }

    #[test]
    fn high_s_signatures_are_accepted() {
        // Apple's Secure Enclave emits high-S ECDSA signatures (~1/3 of the time, measured on an M1,
        // macOS 26.6.1). A low-S rule would reject a third of genuine attestations — never add one.
        let att = sign(&signer(13), claim());
        let sig = Signature::from_der(&att.signature_der).unwrap();
        let (r, s) = sig.split_scalars();
        let high = Signature::from_scalars(r, -*s).unwrap();
        let (low, high) = if sig.normalize_s().is_none() { (sig, high) } else { (high, sig) };
        assert!(low.normalize_s().is_none() && high.normalize_s().is_some());
        for s in [low, high] {
            let mut a = att.clone();
            a.signature_der = s.to_der().as_bytes().to_vec();
            assert_eq!(verify_attestation(&a, &policy(&[BIN])), Ok(DeviceTrust::SelfSigned));
        }
    }

    #[test]
    fn signing_bytes_are_deterministic_and_domain_separated() {
        let key = sec1(&signer(12));
        let a = Attestation::signing_bytes(&claim(), &key);
        assert_eq!(a, Attestation::signing_bytes(&claim(), &key));
        assert!(a.starts_with(ATTEST_DOMAIN));
        let mut other = claim();
        other.model_digest = None;
        assert_ne!(a, Attestation::signing_bytes(&other, &key));
    }
}
