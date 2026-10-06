// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Device attestation — the provider side (ATTESTED_ENGINE_TIER_PLAN_v1, Phase 0).
//!
//! Collects what the device reports about itself ([`DeviceFacts`]), hashes the running binary, and
//! — on Apple Silicon — signs an [`AttestationClaim`] with a Secure Enclave P-256 key
//! ([`SecureEnclaveSigner`]). Verification and the rules for what a verified attestation is worth
//! are pure and live in [`openhydra_protocol::attest`].
//!
//! Phase-0 limits:
//! * The result is `self_signed` only: without an Apple attestation chain a consumer can't tell an
//!   SE key from a software key, and every device fact is the device's own word.
//! * The key is **ephemeral** (per process). A persistent SE key must live in the data-protection
//!   keychain, which requires a codesigned binary with a keychain-access-group entitlement —
//!   unsigned builds get `errSecMissingEntitlement` (-34018). Key continuity across restarts
//!   therefore arrives with signed release builds (Phase 2).
//! * Device facts are read through direct system APIs (`sysctlbyname`, `csr_get_active_config`,
//!   the IORegistry), never by spawning a helper process a hijacked `PATH` could substitute.

use openhydra_protocol::attest::{verify_attestation, AttestPolicy, Attestation, AttestationClaim, DeviceTrust};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

#[cfg(target_os = "macos")]
pub use macos::{device_facts, SecureEnclaveSigner};

/// libp2p proxy method byte for an attestation challenge (consumer → provider):
/// `0x16 ‖ nonce(32) ‖ u16-LE model_len ‖ model_id`. (0x15 is `REMOTE_SCOPE_SET`.)
pub const ATTEST_CHALLENGE: u8 = 0x16;

/// Every attestation response starts with this magic. A legacy provider answers an unknown method
/// byte with a framed serve error, which starts with a u32-LE frame length — so it can't be mistaken
/// for an attestation response (that length would be ~826 MB).
pub const ATTEST_RESP_MAGIC: &[u8; 4] = b"OHA1";
const RESP_OK: u8 = 0x00;
const RESP_REFUSED: u8 = 0x01;
/// Longest model id a challenge may carry.
const MAX_MODEL_ID_LEN: usize = 256;

/// A request/response round-trip to one provider (a `proxy_forward` in production).
pub type Transport<'a> = dyn FnMut(&[u8]) -> Result<Vec<u8>, String> + 'a;

/// Why a challenge didn't yield an attestation (consumer side).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeError {
    /// The request never got a reply.
    Transport(String),
    /// The provider doesn't speak the attestation protocol (a legacy build).
    Unsupported,
    /// The provider understood but declined (no attestor, model not attestable, signing failed).
    Refused(String),
    /// The reply was malformed.
    Malformed,
    /// The attestation names a different model than the one challenged.
    WrongModel,
}

/// What the device reports about itself. Unverified: see the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFacts {
    /// e.g. `"Apple M1"` (`machdep.cpu.brand_string`).
    pub chip: String,
    /// e.g. `"25G76"` (`kern.osversion`).
    pub os_build: String,
    /// SIP fully enabled — no `csr-active-config` allow-bits set.
    pub sip_enabled: bool,
    /// Secure Boot on (`IODeviceTree:/chosen` `secure-boot` = 1).
    pub secure_boot_enabled: bool,
}

/// Why a Secure Enclave attestation couldn't be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestSignError {
    /// SE key generation failed (the `CFError` code) — e.g. no Secure Enclave (Intel Mac, most VMs).
    KeyGen(isize),
    /// The public key couldn't be exported as a 65-byte uncompressed SEC1 point.
    PublicKey,
    /// The SE refused to sign (the `CFError` code).
    Sign(isize),
}

/// SHA-256 of the running executable — the claim's `binary_hash`.
pub fn binary_hash() -> std::io::Result<[u8; 32]> {
    let mut file = std::fs::File::open(std::env::current_exe()?)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher.finalize().into())
}

/// A signing function: turns a claim into a signed [`Attestation`].
pub type ClaimSigner = Box<dyn Fn(AttestationClaim) -> Result<Attestation, AttestSignError> + Send + Sync>;

/// Provider side: answers attestation challenges. Built once at startup (device facts and the binary
/// hash don't change while the process runs), only when the operator opts in.
pub struct Attestor {
    facts: DeviceFacts,
    binary_hash: [u8; 32],
    signer: ClaimSigner,
    /// This provider's libp2p peer id, signed into every claim (see `AttestationClaim::provider_peer`).
    provider_peer: String,
}

impl Attestor {
    pub fn new(facts: DeviceFacts, binary_hash: [u8; 32], signer: ClaimSigner) -> Self {
        Self { facts, binary_hash, signer, provider_peer: String::new() }
    }

    /// Name the provider this attestor answers for (its libp2p peer id), so a consumer can tell its
    /// claims from another provider's relayed through it.
    pub fn with_provider_peer(mut self, peer: impl Into<String>) -> Self {
        self.provider_peer = peer.into();
        self
    }

    /// An attestor backed by a fresh Secure Enclave key and this Mac's device facts.
    #[cfg(target_os = "macos")]
    pub fn secure_enclave() -> Result<Self, String> {
        let signer = SecureEnclaveSigner::new().map_err(|e| format!("Secure Enclave key: {e:?}"))?;
        let bin = binary_hash().map_err(|e| format!("hash running binary: {e}"))?;
        Ok(Self::new(device_facts(), bin, Box::new(move |claim| signer.sign(claim))))
    }

    /// Sign a claim that this device serves `model_id`, over the consumer's `nonce`.
    pub fn attest(&self, model_id: &str, nonce: [u8; 32], now_unix_ms: u64) -> Result<Attestation, AttestSignError> {
        (self.signer)(AttestationClaim {
            chip: self.facts.chip.clone(),
            os_build: self.facts.os_build.clone(),
            sip_enabled: self.facts.sip_enabled,
            secure_boot_enabled: self.facts.secure_boot_enabled,
            binary_hash: self.binary_hash,
            model_id: model_id.to_string(),
            provider_peer: self.provider_peer.clone(),
            // Phase 0: the engine is external, so there is no in-process `active_model_hash`, and the
            // external digest is not bound either — neither would lift a tier (see protocol docs).
            model_digest: None,
            active_model_hash: None,
            session_pubkey: None,
            nonce,
            ts_unix_ms: now_unix_ms,
        })
    }
}

/// Consumer side: frame a challenge body (method byte included).
pub fn encode_challenge(nonce: &[u8; 32], model_id: &str) -> Vec<u8> {
    let mut b = Vec::with_capacity(1 + 32 + 2 + model_id.len());
    b.push(ATTEST_CHALLENGE);
    b.extend_from_slice(nonce);
    b.extend_from_slice(&(model_id.len() as u16).to_le_bytes());
    b.extend_from_slice(model_id.as_bytes());
    b
}

fn decode_challenge(body: &[u8]) -> Option<([u8; 32], String)> {
    let nonce: [u8; 32] = body.get(..32)?.try_into().ok()?;
    let len = u16::from_le_bytes(body.get(32..34)?.try_into().ok()?) as usize;
    if len > MAX_MODEL_ID_LEN || body.len() != 34 + len {
        return None;
    }
    let model = String::from_utf8(body[34..].to_vec()).ok()?;
    Some((nonce, model))
}

/// A `Refused` attestation reply — for refusals decided outside [`handle_attest_inbound`] (e.g. the
/// provider's rate limit), so the consumer reads a refusal rather than a legacy "unsupported" reply.
pub fn attest_refusal(reason: &str) -> Vec<u8> {
    attest_response(RESP_REFUSED, reason.as_bytes())
}

fn attest_response(status: u8, body: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(5 + body.len());
    b.extend_from_slice(ATTEST_RESP_MAGIC);
    b.push(status);
    b.extend_from_slice(body);
    b
}

/// Provider side: answer an inbound challenge (`body` = the bytes after the method byte). Never
/// panics — every failure is a `Refused` reply. `attestable(model)` must be true only for models
/// this node announces publicly: a model that is unshared or private-scope gets the **same**
/// refusal, so a challenge can't reveal that a private model exists.
pub fn handle_attest_inbound(
    body: &[u8],
    attestor: Option<&Attestor>,
    attestable: &dyn Fn(&str) -> bool,
    now_unix_ms: u64,
) -> Vec<u8> {
    handle_attest_inbound_limited(body, attestor, attestable, now_unix_ms, &|| true)
}

/// The refusal text of a challenge declined for load ([`handle_attest_inbound_limited`]): transient,
/// so a consumer keeps what it had verified rather than dropping the badge.
pub const ATTEST_RATE_LIMITED: &str = "attestation rate limited, retry";

/// [`handle_attest_inbound`], with `may_sign` asked right before a signature is made (a provider-wide
/// rate limit): a challenge refused for any other reason doesn't use it up.
pub fn handle_attest_inbound_limited(
    body: &[u8],
    attestor: Option<&Attestor>,
    attestable: &dyn Fn(&str) -> bool,
    now_unix_ms: u64,
    may_sign: &dyn Fn() -> bool,
) -> Vec<u8> {
    let Some(attestor) = attestor else {
        return attest_response(RESP_REFUSED, b"attestation not offered by this provider");
    };
    let Some((nonce, model)) = decode_challenge(body) else {
        return attest_response(RESP_REFUSED, b"malformed challenge");
    };
    if !attestable(&model) {
        return attest_response(RESP_REFUSED, b"model not attestable here");
    }
    if !may_sign() {
        return attest_response(RESP_REFUSED, ATTEST_RATE_LIMITED.as_bytes());
    }
    match attestor.attest(&model, nonce, now_unix_ms) {
        Ok(att) => attest_response(RESP_OK, &att.to_bytes()),
        Err(_) => attest_response(RESP_REFUSED, b"attestation signing failed"),
    }
}

/// Consumer side: send one challenge for `model_id` over `transport` and decode the reply. This does
/// **not** verify the signature or policy — pass the result to
/// [`openhydra_protocol::attest::verify_attestation`]. It does check that the attestation is for the
/// model challenged.
pub fn request_attestation(
    transport: &mut Transport<'_>,
    nonce: [u8; 32],
    model_id: &str,
) -> Result<Attestation, ChallengeError> {
    if model_id.len() > MAX_MODEL_ID_LEN {
        return Err(ChallengeError::Malformed);
    }
    let reply = transport(&encode_challenge(&nonce, model_id)).map_err(ChallengeError::Transport)?;
    let Some(rest) = reply.strip_prefix(ATTEST_RESP_MAGIC.as_slice()) else {
        return Err(ChallengeError::Unsupported);
    };
    match rest.split_first() {
        Some((&RESP_OK, body)) => {
            let att = Attestation::from_bytes(body).map_err(|_| ChallengeError::Malformed)?;
            if att.claim.model_id != model_id {
                return Err(ChallengeError::WrongModel);
            }
            Ok(att)
        }
        Some((&RESP_REFUSED, reason)) => Err(ChallengeError::Refused(bounded_reason(&String::from_utf8_lossy(
            &reason[..reason.len().min(MAX_FAILURE_REASON)],
        )))),
        _ => Err(ChallengeError::Malformed),
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::{c_char, CString};
    use std::ptr::null_mut;

    use core_foundation::base::{CFType, TCFType};
    use core_foundation::data::CFData;
    use core_foundation::string::CFString;
    use core_foundation_sys::base::{kCFAllocatorDefault, CFAllocatorRef, CFTypeRef};
    use core_foundation_sys::string::CFStringRef;
    use openhydra_protocol::attest::{Attestation, AttestationClaim, P256_SEC1_UNCOMPRESSED_LEN};
    use security_framework::key::{Algorithm, GenerateKeyOptions, KeyType, SecKey, Token};

    use super::{AttestSignError, DeviceFacts};

    extern "C" {
        // libsystem_kernel: the active SIP configuration (0 = fully enabled).
        fn csr_get_active_config(config: *mut u32) -> libc::c_int;
    }

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IORegistryEntryFromPath(main_port: u32, path: *const c_char) -> u32;
        fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: CFStringRef,
            allocator: CFAllocatorRef,
            options: u32,
        ) -> CFTypeRef;
        fn IOObjectRelease(object: u32) -> libc::c_int;
    }

    /// Read this Mac's [`DeviceFacts`]. An unreadable string fact is empty; an unreadable security
    /// flag is `false` (fail closed — the consumer's policy then rejects it).
    pub fn device_facts() -> DeviceFacts {
        DeviceFacts {
            chip: sysctl_string("machdep.cpu.brand_string").unwrap_or_default(),
            os_build: sysctl_string("kern.osversion").unwrap_or_default(),
            sip_enabled: sip_enabled(),
            secure_boot_enabled: secure_boot_enabled(),
        }
    }

    fn sysctl_string(name: &str) -> Option<String> {
        let name = CString::new(name).ok()?;
        let mut len: libc::size_t = 0;
        // SAFETY: a size query (null output buffer); `len` is a valid out-pointer.
        if unsafe { libc::sysctlbyname(name.as_ptr(), null_mut(), &mut len, null_mut(), 0) } != 0 {
            return None;
        }
        let mut buf = vec![0u8; len];
        // SAFETY: `buf` has `len` writable bytes; sysctlbyname writes at most `len` and updates it.
        if unsafe { libc::sysctlbyname(name.as_ptr(), buf.as_mut_ptr().cast(), &mut len, null_mut(), 0) } != 0 {
            return None;
        }
        buf.truncate(len);
        while buf.last() == Some(&0) {
            buf.pop();
        }
        String::from_utf8(buf).ok()
    }

    fn sip_enabled() -> bool {
        let mut config = 0u32;
        // SAFETY: `config` is a valid out-pointer for one u32.
        unsafe { csr_get_active_config(&mut config) == 0 && config == 0 }
    }

    fn secure_boot_enabled() -> bool {
        let path = CString::new("IODeviceTree:/chosen").expect("static path has no NUL");
        // SAFETY: 0 is kIOMainPortDefault; `path` is a valid NUL-terminated string.
        let entry = unsafe { IORegistryEntryFromPath(0, path.as_ptr()) };
        if entry == 0 {
            return false;
        }
        let key = CFString::from_static_string("secure-boot");
        // SAFETY: `entry` is a live registry entry, released right after; the returned property
        // follows the CF create rule and is owned by the `CFType` wrapper below.
        let prop = unsafe {
            let prop = IORegistryEntryCreateCFProperty(entry, key.as_concrete_TypeRef(), kCFAllocatorDefault, 0);
            IOObjectRelease(entry);
            prop
        };
        if prop.is_null() {
            return false;
        }
        // SAFETY: non-null CFTypeRef returned under the create rule.
        let prop = unsafe { CFType::wrap_under_create_rule(prop) };
        match prop.downcast_into::<CFData>() {
            Some(data) => data.bytes() == [1, 0, 0, 0],
            None => false,
        }
    }

    /// An ephemeral Secure Enclave P-256 key (private half never leaves the SE) that signs
    /// attestation claims. One per process — see the module docs for why it isn't persistent yet.
    pub struct SecureEnclaveSigner {
        key: SecKey,
        public: [u8; P256_SEC1_UNCOMPRESSED_LEN],
    }

    impl SecureEnclaveSigner {
        /// Generate a fresh SE key. Fails on hardware without a Secure Enclave.
        pub fn new() -> Result<Self, AttestSignError> {
            let mut options = GenerateKeyOptions::default();
            options.set_key_type(KeyType::ec()).set_size_in_bits(256).set_token(Token::SecureEnclave);
            let key = SecKey::new(&options).map_err(|e| AttestSignError::KeyGen(e.code()))?;
            let public = key
                .public_key()
                .and_then(|p| p.external_representation())
                .ok_or(AttestSignError::PublicKey)?;
            let public = public.bytes().try_into().map_err(|_| AttestSignError::PublicKey)?;
            Ok(Self { key, public })
        }

        /// The uncompressed SEC1 public key a consumer verifies against.
        pub fn public_key(&self) -> &[u8; P256_SEC1_UNCOMPRESSED_LEN] {
            &self.public
        }

        /// Sign `claim` (over [`Attestation::signing_bytes`]) inside the Secure Enclave.
        pub fn sign(&self, claim: AttestationClaim) -> Result<Attestation, AttestSignError> {
            let signature_der = self
                .key
                .create_signature(
                    Algorithm::ECDSASignatureMessageX962SHA256,
                    &Attestation::signing_bytes(&claim, &self.public),
                )
                .map_err(|e| AttestSignError::Sign(e.code()))?;
            Ok(Attestation { claim, signing_key: self.public, signature_der })
        }
    }
}

/// How often a consumer re-challenges a provider after a verified or failed attempt.
pub const ATTEST_REFRESH_MS: u64 = 5 * 60 * 1000;
/// How long a consumer leaves a provider that doesn't speak the protocol (a legacy build).
pub const ATTEST_UNSUPPORTED_BACKOFF_MS: u64 = 60 * 60 * 1000;
/// Oldest acceptable claim (covers the round-trip plus modest clock drift; the fresh nonce is the
/// real replay defence).
const ATTEST_MAX_AGE_MS: u64 = 120_000;
/// How long a verified attestation counts after it was made. Serves re-attest every
/// [`ATTEST_REFRESH_MS`], so a provider still being used stays verified; one no longer reached
/// stops showing a badge it may no longer deserve.
pub const ATTEST_VALID_MS: u64 = 24 * 60 * 60 * 1000;
/// Providers whose attestation records a consumer keeps (see [`insert_record_bounded`]).
pub const MAX_ATTEST_RECORDS: usize = 4096;
/// Longest failure reason kept in a record, in bytes: a refusal reason is the provider's own text.
const MAX_FAILURE_REASON: usize = 256;

/// `reason` cut to [`MAX_FAILURE_REASON`] bytes, on a character boundary.
fn bounded_reason(reason: &str) -> String {
    let mut end = reason.len().min(MAX_FAILURE_REASON);
    while !reason.is_char_boundary(end) {
        end -= 1;
    }
    reason[..end].to_string()
}

/// Store `record` under `key`, keeping at most [`MAX_ATTEST_RECORDS`]: when full, records older than
/// [`ATTEST_VALID_MS`] go first, then the oldest.
pub fn insert_record_bounded(map: &mut HashMap<String, AttestRecord>, key: String, record: AttestRecord, now_unix_ms: u64) {
    if map.len() >= MAX_ATTEST_RECORDS && !map.contains_key(&key) {
        map.retain(|_, r| now_unix_ms.saturating_sub(r.last_attempt_ms) <= ATTEST_VALID_MS);
        while map.len() >= MAX_ATTEST_RECORDS {
            let Some(oldest) = map.iter().min_by_key(|(_, r)| r.last_attempt_ms).map(|(k, _)| k.clone()) else {
                break;
            };
            map.remove(&oldest);
        }
    }
    map.insert(key, record);
}
/// How far ahead of the consumer's clock a provider's claim may be dated.
const ATTEST_MAX_SKEW_MS: u64 = 30_000;

/// The outcome of the latest challenge to one provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestOutcome {
    /// The attestation verified at this level.
    Verified(DeviceTrust),
    /// The provider doesn't speak the protocol.
    Unsupported,
    /// Anything else: refused, transport error, malformed, wrong model, or failed verification.
    Failed(String),
}

/// Consumer-side record of the latest challenge to one provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestRecord {
    pub last_attempt_ms: u64,
    pub outcome: AttestOutcome,
    /// The key of the last *verified* attestation (kept across failed attempts).
    pub key: Option<[u8; 65]>,
    /// How many times a verified key differed from the previous verified key. In Phase 0 unsigned
    /// builds use a per-process key, so a restart rotates it; this is surfaced, not blocked.
    pub key_rotations: u32,
}

impl AttestRecord {
    /// Whether it's time to challenge this provider again.
    pub fn due(&self, now_unix_ms: u64) -> bool {
        let wait = match self.outcome {
            AttestOutcome::Unsupported => ATTEST_UNSUPPORTED_BACKOFF_MS,
            _ => ATTEST_REFRESH_MS,
        };
        now_unix_ms.saturating_sub(self.last_attempt_ms) >= wait
    }

    /// The device-trust level this record supports at `now_unix_ms` — only a verified outcome counts,
    /// and only for [`ATTEST_VALID_MS`] after it was made.
    pub fn device_trust(&self, now_unix_ms: u64) -> Option<DeviceTrust> {
        match self.outcome {
            AttestOutcome::Verified(t) if now_unix_ms.saturating_sub(self.last_attempt_ms) <= ATTEST_VALID_MS => Some(t),
            _ => None,
        }
    }
}

/// Consumer side: challenge one provider for `model_id`, verify the reply, and fold the result into
/// the provider's previous record. Pure apart from `transport`.
///
/// A verified attestation's key is compared with the previous verified key: a change increments
/// `key_rotations` (and is logged by the caller) rather than failing — see [`AttestRecord`].
pub fn attest_provider(
    transport: &mut Transport<'_>,
    model_id: &str,
    nonce: [u8; 32],
    now_unix_ms: u64,
    trusted_binaries: &[[u8; 32]],
    prev: Option<&AttestRecord>,
    provider_peer: &str,
) -> AttestRecord {
    let key = prev.and_then(|p| p.key);
    let key_rotations = prev.map_or(0, |p| p.key_rotations);
    let failed = |outcome| AttestRecord { last_attempt_ms: now_unix_ms, outcome, key, key_rotations };

    let att = match request_attestation(transport, nonce, model_id) {
        Ok(att) => att,
        Err(ChallengeError::Unsupported) => return failed(AttestOutcome::Unsupported),
        // Declined for load: keep what was verified (it still expires on its own), and try again
        // when next due rather than dropping the badge.
        Err(ChallengeError::Refused(r)) if r == ATTEST_RATE_LIMITED => {
            if let Some(p) = prev {
                return p.clone();
            }
            return failed(AttestOutcome::Failed(bounded_reason(&r)));
        }
        Err(e) => return failed(AttestOutcome::Failed(bounded_reason(&format!("{e:?}")))),
    };
    let policy = AttestPolicy {
        expected_nonce: nonce,
        expected_provider: provider_peer,
        now_unix_ms,
        max_age_ms: ATTEST_MAX_AGE_MS,
        max_skew_ms: ATTEST_MAX_SKEW_MS,
        trusted_binaries,
        pinned_key: None,
    };
    match verify_attestation(&att, &policy) {
        Ok(trust) => AttestRecord {
            last_attempt_ms: now_unix_ms,
            outcome: AttestOutcome::Verified(trust),
            key: Some(att.signing_key),
            key_rotations: key_rotations + u32::from(key.is_some_and(|k| k != att.signing_key)),
        },
        Err(e) => failed(AttestOutcome::Failed(bounded_reason(&format!("{e:?}")))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p256::ecdsa::signature::Signer as _;

    const NOW: u64 = 1_790_000_000_000;
    /// The provider the test attestors answer for.
    const PROV: &str = "12D3KooWProvider";
    const BIN: [u8; 32] = [0xB1; 32];

    /// An attestor over a software P-256 key (stands in for the Secure Enclave, so the wire protocol
    /// is tested on every platform).
    fn software_attestor() -> Attestor {
        let sk = p256::ecdsa::SigningKey::from_slice(&[7; 32]).unwrap();
        let public: [u8; 65] = sk.verifying_key().to_encoded_point(false).as_bytes().try_into().unwrap();
        let facts = DeviceFacts {
            chip: "Apple M1".into(),
            os_build: "25G76".into(),
            sip_enabled: true,
            secure_boot_enabled: true,
        };
        Attestor::new(
            facts,
            BIN,
            Box::new(move |claim| {
                let sig: p256::ecdsa::Signature = sk.sign(&Attestation::signing_bytes(&claim, &public));
                Ok(Attestation { claim, signing_key: public, signature_der: sig.to_der().as_bytes().to_vec() })
            }),
        )
        .with_provider_peer(PROV)
    }

    fn public_only(model: &str) -> bool {
        model == "qwen2.5-0.5b"
    }

    /// A transport that delivers the challenge straight to a provider handler.
    fn loopback<'a>(
        attestor: Option<&'a Attestor>,
    ) -> impl FnMut(&[u8]) -> Result<Vec<u8>, String> + 'a {
        move |framed: &[u8]| {
            assert_eq!(framed.first(), Some(&ATTEST_CHALLENGE));
            Ok(handle_attest_inbound(&framed[1..], attestor, &public_only, NOW))
        }
    }

    fn refusal(reply: &[u8]) -> String {
        let rest = reply.strip_prefix(ATTEST_RESP_MAGIC.as_slice()).unwrap();
        assert_eq!(rest[0], RESP_REFUSED);
        String::from_utf8(rest[1..].to_vec()).unwrap()
    }

    #[test]
    fn challenge_round_trip_verifies() {
        let attestor = software_attestor();
        let nonce = [0x33; 32];
        let att = request_attestation(&mut loopback(Some(&attestor)), nonce, "qwen2.5-0.5b").unwrap();
        let trusted = [BIN];
        let policy = AttestPolicy {
            expected_nonce: nonce,
            expected_provider: PROV,
            now_unix_ms: NOW,
            max_age_ms: 60_000,
            max_skew_ms: 5_000,
            trusted_binaries: &trusted,
            pinned_key: None,
        };
        assert_eq!(verify_attestation(&att, &policy), Ok(DeviceTrust::SelfSigned));
        assert_eq!(att.claim.model_id, "qwen2.5-0.5b");
        assert_eq!(att.claim.active_model_hash, None);
    }

    #[test]
    fn unshared_and_private_models_get_the_same_refusal() {
        let attestor = software_attestor();
        // `public_only` stands in for `announce_globally`: both an unshared and a private model are
        // simply "not attestable" — the refusal text is identical, so nothing leaks.
        let a = handle_attest_inbound(&encode_challenge(&[1; 32], "private-model")[1..], Some(&attestor), &public_only, NOW);
        let b = handle_attest_inbound(&encode_challenge(&[1; 32], "never-heard-of-it")[1..], Some(&attestor), &public_only, NOW);
        assert_eq!(a, b);
        assert_eq!(refusal(&a), "model not attestable here");
        assert_eq!(
            request_attestation(&mut loopback(Some(&attestor)), [1; 32], "private-model"),
            Err(ChallengeError::Refused("model not attestable here".into()))
        );
    }

    #[test]
    fn a_provider_without_an_attestor_refuses() {
        assert_eq!(
            request_attestation(&mut loopback(None), [1; 32], "qwen2.5-0.5b"),
            Err(ChallengeError::Refused("attestation not offered by this provider".into()))
        );
    }

    #[test]
    fn malformed_challenges_are_refused() {
        let attestor = software_attestor();
        let good = encode_challenge(&[1; 32], "qwen2.5-0.5b");
        let mut long_len = good.clone();
        long_len[33..35].copy_from_slice(&(MAX_MODEL_ID_LEN as u16 + 1).to_le_bytes());
        let mut trailing = good.clone();
        trailing.push(0);
        for body in [&good[1..10], &good[1..34], &long_len[1..], &trailing[1..], &[][..]] {
            assert_eq!(
                refusal(&handle_attest_inbound(body, Some(&attestor), &public_only, NOW)),
                "malformed challenge"
            );
        }
    }

    #[test]
    fn a_legacy_provider_reply_reads_as_unsupported() {
        // What a pre-attestation provider sends for an unknown method byte: a framed serve error.
        let legacy = crate::serve::frame_response(&[crate::serve::ServeChunk::Error("unsupported method".into()).encode()]);
        let mut transport = |_: &[u8]| Ok(legacy.clone());
        assert_eq!(request_attestation(&mut transport, [1; 32], "m"), Err(ChallengeError::Unsupported));
    }

    #[test]
    fn an_attestation_for_another_model_is_rejected() {
        let attestor = software_attestor();
        // A provider that answers with an attestation for a different (public) model than asked.
        let mut transport = |_: &[u8]| {
            Ok(handle_attest_inbound(&encode_challenge(&[1; 32], "qwen2.5-0.5b")[1..], Some(&attestor), &|_| true, NOW))
        };
        assert_eq!(request_attestation(&mut transport, [1; 32], "gpt-oss-120b"), Err(ChallengeError::WrongModel));
    }

    #[test]
    fn a_refusal_reason_is_cut_short_and_the_records_stay_bounded() {
        let long = format!("{}{}", "é".repeat(200), "x".repeat(10_000));
        let mut reply = ATTEST_RESP_MAGIC.to_vec();
        reply.push(RESP_REFUSED);
        reply.extend_from_slice(long.as_bytes());
        let mut t = |_: &[u8]| Ok(reply.clone());
        let Err(ChallengeError::Refused(reason)) = request_attestation(&mut t, [1; 32], "m") else { panic!() };
        assert!(reason.len() <= MAX_FAILURE_REASON);
        let rec = attest_provider(&mut t, "m", [1; 32], NOW, &[], None, PROV);
        let AttestOutcome::Failed(why) = rec.outcome else { panic!() };
        assert!(why.len() <= MAX_FAILURE_REASON);

        let record = |at: u64| AttestRecord { last_attempt_ms: at, outcome: AttestOutcome::Unsupported, key: None, key_rotations: 0 };
        let mut map = HashMap::new();
        for i in 0..MAX_ATTEST_RECORDS as u64 {
            map.insert(format!("p{i}"), record(NOW - ATTEST_VALID_MS + i));
        }
        insert_record_bounded(&mut map, "new".into(), record(NOW), NOW + 10);
        // p0..p9 are past ATTEST_VALID_MS by then: they go, the rest stay.
        assert_eq!(map.len(), MAX_ATTEST_RECORDS - 10 + 1);
        assert!(map.contains_key("new") && map.contains_key("p10"));
        assert!(!map.contains_key("p0") && !map.contains_key("p9"), "the expired ones went first");
        // Nothing expired: the oldest goes.
        let mut full: HashMap<String, AttestRecord> =
            (0..MAX_ATTEST_RECORDS as u64).map(|i| (format!("p{i}"), record(NOW + i))).collect();
        insert_record_bounded(&mut full, "new".into(), record(NOW), NOW + 10);
        assert_eq!(full.len(), MAX_ATTEST_RECORDS);
        assert!(!full.contains_key("p0"));
    }

    #[test]
    fn a_load_refusal_keeps_the_verified_record_and_spends_nothing_on_other_refusals() {
        let attestor = software_attestor();
        let signs = std::cell::Cell::new(0);
        let limit = || {
            signs.set(signs.get() + 1);
            false
        };
        // Refused before signing (model not attestable): the limiter isn't consulted.
        handle_attest_inbound_limited(&encode_challenge(&[1; 32], "private")[1..], Some(&attestor), &|_| false, NOW, &limit);
        assert_eq!(signs.get(), 0);
        let reply = handle_attest_inbound_limited(&encode_challenge(&[1; 32], "m")[1..], Some(&attestor), &|_| true, NOW, &limit);
        assert_eq!(signs.get(), 1);
        // A consumer that had verified this provider keeps that record through a load refusal.
        let prev = AttestRecord {
            last_attempt_ms: NOW - 1,
            outcome: AttestOutcome::Verified(DeviceTrust::SelfSigned),
            key: None,
            key_rotations: 0,
        };
        let mut t = |_: &[u8]| Ok(reply.clone());
        let rec = attest_provider(&mut t, "m", [1; 32], NOW, &[], Some(&prev), PROV);
        assert_eq!(rec.device_trust(NOW), Some(DeviceTrust::SelfSigned));
    }

    #[test]
    fn a_verified_attestation_expires() {
        let rec = AttestRecord {
            last_attempt_ms: NOW,
            outcome: AttestOutcome::Verified(DeviceTrust::SelfSigned),
            key: None,
            key_rotations: 0,
        };
        assert_eq!(rec.device_trust(NOW + ATTEST_VALID_MS), Some(DeviceTrust::SelfSigned));
        assert_eq!(rec.device_trust(NOW + ATTEST_VALID_MS + 1), None);
    }

    #[test]
    fn transport_errors_and_garbage_replies_never_panic() {
        let mut down = |_: &[u8]| Err("timeout".to_string());
        assert_eq!(request_attestation(&mut down, [1; 32], "m"), Err(ChallengeError::Transport("timeout".into())));

        let attestor = software_attestor();
        let good = handle_attest_inbound(&encode_challenge(&[1; 32], "qwen2.5-0.5b")[1..], Some(&attestor), &public_only, NOW);
        for i in 0..good.len() {
            let mut bad = good.clone();
            bad[i] ^= 0xFF;
            let mut t = |_: &[u8]| Ok(bad.clone());
            let _ = request_attestation(&mut t, [1; 32], "qwen2.5-0.5b");
        }
        for n in 0..good.len() {
            let cut = good[..n].to_vec();
            let mut t = |_: &[u8]| Ok(cut.clone());
            assert!(request_attestation(&mut t, [1; 32], "qwen2.5-0.5b").is_err());
        }
    }

    /// An attestor over a software key seeded with `seed` (a different seed = a rotated key).
    fn attestor_with_key(seed: u8) -> Attestor {
        let sk = p256::ecdsa::SigningKey::from_slice(&[seed; 32]).unwrap();
        let public: [u8; 65] = sk.verifying_key().to_encoded_point(false).as_bytes().try_into().unwrap();
        let facts = software_attestor().facts;
        Attestor::new(
            facts,
            BIN,
            Box::new(move |claim| {
                let sig: p256::ecdsa::Signature = sk.sign(&Attestation::signing_bytes(&claim, &public));
                Ok(Attestation { claim, signing_key: public, signature_der: sig.to_der().as_bytes().to_vec() })
            }),
        )
        .with_provider_peer(PROV)
    }

    #[test]
    fn an_attestation_relayed_from_another_provider_fails() {
        // The challenged peer forwards the challenge to an honest provider and relays its answer.
        let honest = attestor_with_key(1);
        let rec = attest_provider(&mut loopback(Some(&honest)), "qwen2.5-0.5b", [1; 32], NOW, &[BIN], None, "12D3KooWRelayer");
        assert_eq!(rec.device_trust(NOW), None);
        assert!(matches!(rec.outcome, AttestOutcome::Failed(ref why) if why.contains("ProviderMismatch")), "{:?}", rec.outcome);
    }

    #[test]
    fn attest_provider_records_verification_and_key_rotation() {
        let first = attest_provider(&mut loopback(Some(&attestor_with_key(1))), "qwen2.5-0.5b", [1; 32], NOW, &[BIN], None, PROV);
        assert_eq!(first.outcome, AttestOutcome::Verified(DeviceTrust::SelfSigned));
        assert_eq!(first.device_trust(NOW), Some(DeviceTrust::SelfSigned));
        assert_eq!(first.key_rotations, 0);

        // Same key again: no rotation.
        let again = attest_provider(&mut loopback(Some(&attestor_with_key(1))), "qwen2.5-0.5b", [2; 32], NOW, &[BIN], Some(&first), PROV);
        assert_eq!(again.key_rotations, 0);

        // Provider restarted with a fresh per-process key: still verified, rotation counted.
        let rotated = attest_provider(&mut loopback(Some(&attestor_with_key(2))), "qwen2.5-0.5b", [3; 32], NOW, &[BIN], Some(&again), PROV);
        assert_eq!(rotated.outcome, AttestOutcome::Verified(DeviceTrust::SelfSigned));
        assert_eq!(rotated.key_rotations, 1);
        assert_ne!(rotated.key, first.key);
    }

    #[test]
    fn attest_provider_failures_keep_the_last_verified_key() {
        let ok = attest_provider(&mut loopback(Some(&attestor_with_key(1))), "qwen2.5-0.5b", [1; 32], NOW, &[BIN], None, PROV);

        // Untrusted binary → Failed, but the last verified key and rotation count are kept.
        let untrusted = attest_provider(&mut loopback(Some(&attestor_with_key(1))), "qwen2.5-0.5b", [2; 32], NOW, &[[0; 32]], Some(&ok), PROV);
        assert!(matches!(untrusted.outcome, AttestOutcome::Failed(ref e) if e.contains("UntrustedBinary")));
        assert_eq!(untrusted.device_trust(NOW), None);
        assert_eq!(untrusted.key, ok.key);

        // Refusal and transport errors are Failed too.
        let refused = attest_provider(&mut loopback(None), "qwen2.5-0.5b", [3; 32], NOW, &[BIN], Some(&ok), PROV);
        assert!(matches!(refused.outcome, AttestOutcome::Failed(_)));
        let mut down = |_: &[u8]| Err("timeout".to_string());
        let dropped = attest_provider(&mut down, "qwen2.5-0.5b", [4; 32], NOW, &[BIN], Some(&ok), PROV);
        assert!(matches!(dropped.outcome, AttestOutcome::Failed(_)));
        assert_eq!(dropped.key, ok.key);
    }

    #[test]
    fn retry_timing_backs_off_legacy_providers() {
        let legacy = crate::serve::frame_response(&[crate::serve::ServeChunk::Error("unsupported method".into()).encode()]);
        let mut t = |_: &[u8]| Ok(legacy.clone());
        let rec = attest_provider(&mut t, "m", [1; 32], NOW, &[BIN], None, PROV);
        assert_eq!(rec.outcome, AttestOutcome::Unsupported);
        assert!(!rec.due(NOW + ATTEST_REFRESH_MS));
        assert!(rec.due(NOW + ATTEST_UNSUPPORTED_BACKOFF_MS));

        let ok = attest_provider(&mut loopback(Some(&attestor_with_key(1))), "qwen2.5-0.5b", [1; 32], NOW, &[BIN], None, PROV);
        assert!(!ok.due(NOW + ATTEST_REFRESH_MS - 1));
        assert!(ok.due(NOW + ATTEST_REFRESH_MS));
    }

    #[test]
    fn binary_hash_is_stable() {
        let a = binary_hash().unwrap();
        assert_eq!(a, binary_hash().unwrap());
        assert_ne!(a, [0u8; 32]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn device_facts_are_readable() {
        let facts = device_facts();
        assert!(!facts.chip.is_empty(), "{facts:?}");
        assert!(!facts.os_build.is_empty(), "{facts:?}");
    }

    /// Live Secure Enclave round-trip: SE-sign a real claim, verify it with the pure verifier.
    /// Needs Apple Silicon with SIP + Secure Boot on (most CI VMs have no SE):
    /// `cargo test -p openhydra-agent attest -- --ignored`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn secure_enclave_attestation_verifies() {
        use openhydra_protocol::attest::{
            verify_attestation, AttestError, AttestPolicy, AttestationClaim, DeviceTrust,
        };

        let facts = device_facts();
        let bin = binary_hash().unwrap();
        let signer = SecureEnclaveSigner::new().expect("Secure Enclave key");
        let now = 1_790_000_000_000;
        let nonce = [0x42; 32];
        let claim = AttestationClaim {
            chip: facts.chip.clone(),
            os_build: facts.os_build.clone(),
            sip_enabled: facts.sip_enabled,
            secure_boot_enabled: facts.secure_boot_enabled,
            binary_hash: bin,
            model_id: "qwen2.5-0.5b".into(),
            provider_peer: "12D3KooWThisMac".into(),
            model_digest: None,
            active_model_hash: None,
            session_pubkey: None,
            nonce,
            ts_unix_ms: now,
        };
        let trusted = [bin];
        let policy = AttestPolicy {
            expected_nonce: nonce,
            expected_provider: "12D3KooWThisMac",
            now_unix_ms: now,
            max_age_ms: 60_000,
            max_skew_ms: 5_000,
            trusted_binaries: &trusted,
            pinned_key: Some(signer.public_key()),
        };
        // Several signatures, so the ~1/3 high-S signatures the SE emits are exercised too.
        for _ in 0..8 {
            let att = signer.sign(claim.clone()).unwrap();
            let wire = openhydra_protocol::attest::Attestation::from_bytes(&att.to_bytes()).unwrap();
            assert_eq!(verify_attestation(&wire, &policy), Ok(DeviceTrust::SelfSigned), "{facts:?}");
        }
        let mut tampered = signer.sign(claim).unwrap();
        tampered.claim.model_id = "gpt-oss-120b".into();
        assert_eq!(verify_attestation(&tampered, &policy), Err(AttestError::BadSignature));
    }
}
