//! Swarm membership credentials (M3) — the "private key for private sharing" crypto core.
//!
//! A **swarm** is a private trust group identified by a **group public key**. Its *owner* holds the
//! group **keypair** (a fresh Ed25519 keypair, distinct from any node identity — the private half
//! never leaves the owner's machine). Membership is granted by the owner signing a
//! [`MembershipCredential`] over a member's **public** key; a member later presents that credential
//! on the serve path (M4) to prove it belongs. **No shared secret is ever transmitted** — enrollment
//! moves only public keys + a signed credential, so an intercepted request or credential grants no
//! standing access (it is scope-limited to one member key, expiring, and revocable).
//!
//! Design invariants (mirror [`crate::card`] and [`crate::dht::sign_peer_record`]):
//! * **Signed, or it's a rumor.** Both the [`EnrollmentRequest`] (member-signed, proving it holds
//!   `member_public_key`) and the [`MembershipCredential`] (group-signed) cover every field except the
//!   signature, over a **domain-separated** preimage that **binds `sig_alg`** — so a signature can
//!   never be replayed across artifact types or downgraded.
//! * **No secret, ever.** Neither artifact carries key material beyond *public* keys + a signature.
//!   Re-sharing one grants nothing. This is asserted on the serialized bytes ([`tests`]).
//! * **Expiry + revocation.** A credential expires; the owner also keeps a revocation set (by member
//!   public key). `verify_credential` refuses an expired or revoked credential — expiry is checked
//!   LAST, only once the signature (which covers `expires_at`) has verified.
//! * **Bind key ↔ identity.** A verified credential authorises exactly one `member_public_key`; the
//!   caller derives that key's libp2p peer id ([`credential_member_peer_id`]) to gate a live serve.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use openhydra_protocol::crypto_agility::SigAlg;

/// Schema version new artifacts are written at. A verifier rejects a *newer* schema it can't fully
/// validate rather than trusting fields it doesn't understand (parity with [`crate::card`]).
pub const MEMBERSHIP_SCHEMA_VERSION: u32 = 1;

/// Domain-separation header for the credential preimage (bumped with any layout change).
const CRED_DOMAIN: &str = "openhydra-swarm-credential-v1";
/// Domain-separation header for the enrollment-request preimage.
const ENROLL_DOMAIN: &str = "openhydra-swarm-enroll-request-v1";

/// Magnet-string scheme prefixes: `openhydra:enroll:<b64url(cbor)>` / `openhydra:cred:<b64url(cbor)>`.
const ENROLL_MAGNET_PREFIX: &str = "openhydra:enroll:";
const CRED_MAGNET_PREFIX: &str = "openhydra:cred:";

/// Upper bound on a user-supplied label (swarm name / member name). Bounds the signed preimage and
/// the UI; a label over this is rejected on sign so a card can't carry an unbounded blob.
const MAX_LABEL_LEN: usize = 128;

/// Upper bound on a machine id / key-hint scalar (`member_openhydra_peer_id`, the `swarm_public_key`
/// hint). Real values are short hex; this just stops an unbounded field in a self-signed artifact.
const MAX_ID_LEN: usize = 256;

fn default_schema() -> u32 {
    MEMBERSHIP_SCHEMA_VERSION
}
fn default_sig_alg() -> u8 {
    SigAlg::Ed25519.to_u8()
}

fn b64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE.encode(data)
}
fn b64_decode(s: &str) -> Result<Vec<u8>, MembershipError> {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE
        .decode(s)
        .map_err(|e| MembershipError::Crypto(format!("base64 decode: {e}")))
}

/// Everything that can be wrong with an enrollment request or a membership credential. `PartialEq`
/// so tests can assert the exact variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MembershipError {
    /// Schema is newer than this build can validate.
    UnsupportedSchema(u32),
    /// A fielded string carries a preimage delimiter (`\n`), or a label exceeds [`MAX_LABEL_LEN`] —
    /// rejected so the signing preimage stays injective and bounded.
    Malformed(String),
    /// No signature or no (relevant) public key.
    MissingSignature,
    /// Signature did not verify against the preimage / the signing key.
    BadSignature,
    /// The credential's `member_public_key` does not match the identity the caller expected.
    MemberMismatch { expected: String, found: String },
    /// The credential authorises a member on this swarm's revocation list.
    Revoked(String),
    /// `now_ms >= expires_at`.
    Expired { expires_at: u64, now_ms: u64 },
    /// Encoding/decoding or key-parse failure.
    Crypto(String),
}

impl std::fmt::Display for MembershipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MembershipError::UnsupportedSchema(v) => write!(f, "unsupported schema version {v}"),
            MembershipError::Malformed(s) => write!(f, "malformed: {s}"),
            MembershipError::MissingSignature => write!(f, "missing signature or public key"),
            MembershipError::BadSignature => write!(f, "signature verification failed"),
            MembershipError::MemberMismatch { expected, found } => {
                write!(f, "member key mismatch: expected={expected} found={found}")
            }
            MembershipError::Revoked(k) => write!(f, "credential revoked for member {k}"),
            MembershipError::Expired { expires_at, now_ms } => {
                write!(f, "credential expired (expires_at={expires_at} now={now_ms})")
            }
            MembershipError::Crypto(s) => write!(f, "crypto error: {s}"),
        }
    }
}
impl std::error::Error for MembershipError {}

/// A member's request to join a swarm: its **public** identity + a self-chosen label, **signed by
/// the member's node identity key** so the owner knows the requester actually controls
/// `member_public_key` (and the request wasn't tampered with in transit). Carries no secret.
///
/// The `swarm_public_key` is a *hint* of which swarm the member wants to join (from a card, say);
/// the owner confirms it against the swarm they're approving into. Delivered out-of-band (short
/// code / QR / file) — an interceptor learns only public data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnrollmentRequest {
    #[serde(default = "default_schema")]
    pub schema_version: u32,
    /// Which swarm the member wants to join (hex group public key), or "" if unspecified.
    #[serde(default)]
    pub swarm_public_key: String,
    /// The member's OpenHydra identity id (for display/reputation).
    pub member_openhydra_peer_id: String,
    /// The member's Ed25519 **public** key (hex) — the identity the credential will authorise.
    pub member_public_key: String,
    /// Member's self-chosen display label (e.g. "Sam's MacBook"). Bounded, no newline.
    #[serde(default)]
    pub label: String,
    pub requested_at: u64,
    #[serde(default = "default_sig_alg")]
    pub sig_alg: u8,
    /// base64url signature over [`enroll_canonical_bytes`], by the member's identity key.
    #[serde(default)]
    pub signature: String,
}

/// An owner-issued membership credential: the **group key** vouches for `member_public_key` on this
/// swarm until `expires_at`. Presented by the member on the serve path (M4). Carries no secret.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MembershipCredential {
    #[serde(default = "default_schema")]
    pub schema_version: u32,
    /// The swarm's Ed25519 **group public key** (hex) — the trust anchor that signed this.
    pub swarm_public_key: String,
    /// The authorised member's Ed25519 **public** key (hex). Its libp2p peer id is the serve-gate key.
    pub member_public_key: String,
    /// The member's OpenHydra identity id (display/reputation; not the trust anchor).
    #[serde(default)]
    pub member_openhydra_peer_id: String,
    /// Owner-attached label for the swarm (e.g. "Home rig"). Bounded, no newline.
    #[serde(default)]
    pub swarm_label: String,
    pub issued_at: u64,
    pub expires_at: u64,
    #[serde(default = "default_sig_alg")]
    pub sig_alg: u8,
    /// base64url signature over [`cred_canonical_bytes`], by the swarm's group key.
    #[serde(default)]
    pub signature: String,
}

/// A credential whose signature (against the swarm group key), schema, revocation status, and expiry
/// have all been checked. Safe to act on: `member_public_key` is authorised on `swarm_public_key`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCredential {
    pub credential: MembershipCredential,
}

/// A verified enrollment request — the member proved it holds `member_public_key`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRequest {
    pub request: EnrollmentRequest,
}

// ── wellformedness (injective, bounded preimage) ──

fn check_label(label: &str, what: &str) -> Result<(), MembershipError> {
    if label.contains('\n') {
        return Err(MembershipError::Malformed(format!("{what} contains a newline: {label:?}")));
    }
    if label.chars().count() > MAX_LABEL_LEN {
        return Err(MembershipError::Malformed(format!("{what} exceeds {MAX_LABEL_LEN} chars")));
    }
    Ok(())
}

/// A bounded, newline-free scalar (ids, key hints). Bounds the signed preimage — review #5: the free
/// `member_openhydra_peer_id` and the `swarm_public_key` hint were newline-checked but unbounded, so a
/// self-signed request could carry a multi-MB field that then persists into the owner's file and the
/// returned credential. Real keys (64 hex) sit well under the cap.
fn check_no_newline(s: &str, what: &str) -> Result<(), MembershipError> {
    if s.contains('\n') {
        return Err(MembershipError::Malformed(format!("{what} contains a newline: {s:?}")));
    }
    if s.len() > MAX_ID_LEN {
        return Err(MembershipError::Malformed(format!("{what} exceeds {MAX_ID_LEN} bytes")));
    }
    Ok(())
}

/// Verify `sig` over `msg` with `ed_pk`, branching explicitly on `alg` — review #2: even though
/// `is_implemented()` gates the caller today (only Ed25519), a future algorithm marked implemented
/// (PQC3.1) must NOT silently fall through to Ed25519 verification. Adding the `match` now means the
/// later algorithm has to supply its own verify path rather than being confused for Ed25519.
fn verify_with_alg(
    alg: SigAlg,
    ed_pk: &libp2p::identity::ed25519::PublicKey,
    msg: &[u8],
    sig: &[u8],
) -> Result<(), MembershipError> {
    let ok = match alg {
        SigAlg::Ed25519 => ed_pk.verify(msg, sig),
        other => {
            return Err(MembershipError::Crypto(format!("verify not implemented for {other:?}")))
        }
    };
    if ok {
        Ok(())
    } else {
        Err(MembershipError::BadSignature)
    }
}

// ── enrollment request ──

/// Deterministic signing preimage for an enrollment request. Domain-separated, binds `sig_alg`,
/// excludes only `signature`. Signer and verifier are both Rust and reproduce this identically.
pub fn enroll_canonical_bytes(r: &EnrollmentRequest) -> Vec<u8> {
    format!(
        "{ENROLL_DOMAIN}\nsig_alg={}\nschema_version={}\n\
         swarm_public_key={}\nmember_openhydra_peer_id={}\nmember_public_key={}\n\
         label={}\nrequested_at={}",
        r.sig_alg,
        r.schema_version,
        r.swarm_public_key,
        r.member_openhydra_peer_id,
        r.member_public_key,
        r.label,
        r.requested_at,
    )
    .into_bytes()
}

/// Sign an enrollment request with the **member's node identity keypair**, populating
/// `member_public_key`, `member_openhydra_peer_id` is caller-supplied, `sig_alg`, and `signature`.
/// The member proves possession of its identity key so the owner can trust the bound public key.
pub fn sign_enrollment_request(
    mut req: EnrollmentRequest,
    member_keypair: &libp2p::identity::Keypair,
) -> Result<EnrollmentRequest, MembershipError> {
    let ed = member_keypair
        .public()
        .try_into_ed25519()
        .map_err(|e| MembershipError::Crypto(format!("not ed25519: {e}")))?;
    req.member_public_key = hex::encode(ed.to_bytes());
    req.sig_alg = SigAlg::Ed25519.to_u8();
    check_label(&req.label, "label")?;
    check_no_newline(&req.member_openhydra_peer_id, "member_openhydra_peer_id")?;
    check_no_newline(&req.swarm_public_key, "swarm_public_key")?;
    let sig = member_keypair
        .sign(&enroll_canonical_bytes(&req))
        .map_err(|e| MembershipError::Crypto(format!("sign failed: {e}")))?;
    req.signature = b64_encode(&sig);
    Ok(req)
}

/// Verify an enrollment request: schema, wellformedness, and the member's signature over the
/// preimage (against the embedded `member_public_key`). Proves the requester holds that key.
pub fn verify_enrollment_request(
    req: &EnrollmentRequest,
) -> Result<VerifiedRequest, MembershipError> {
    if req.schema_version > MEMBERSHIP_SCHEMA_VERSION {
        return Err(MembershipError::UnsupportedSchema(req.schema_version));
    }
    check_label(&req.label, "label")?;
    check_no_newline(&req.member_openhydra_peer_id, "member_openhydra_peer_id")?;
    check_no_newline(&req.swarm_public_key, "swarm_public_key")?;
    if req.signature.is_empty() || req.member_public_key.is_empty() {
        return Err(MembershipError::MissingSignature);
    }
    let alg = SigAlg::from_u8(req.sig_alg).map_err(|e| MembershipError::Crypto(e.to_string()))?;
    if !alg.is_implemented() {
        return Err(MembershipError::Crypto(format!("unsupported signature algorithm: {alg:?}")));
    }
    let ed_pk = ed_pubkey_from_hex(&req.member_public_key)?;
    let sig = b64_decode(&req.signature)?;
    verify_with_alg(alg, &ed_pk, &enroll_canonical_bytes(req), &sig)?;
    Ok(VerifiedRequest { request: req.clone() })
}

// ── membership credential ──

/// Deterministic signing preimage for a membership credential. Domain-separated, binds `sig_alg`,
/// excludes only `signature`.
pub fn cred_canonical_bytes(c: &MembershipCredential) -> Vec<u8> {
    format!(
        "{CRED_DOMAIN}\nsig_alg={}\nschema_version={}\n\
         swarm_public_key={}\nmember_public_key={}\nmember_openhydra_peer_id={}\n\
         swarm_label={}\nissued_at={}\nexpires_at={}",
        c.sig_alg,
        c.schema_version,
        c.swarm_public_key,
        c.member_public_key,
        c.member_openhydra_peer_id,
        c.swarm_label,
        c.issued_at,
        c.expires_at,
    )
    .into_bytes()
}

/// Sign a membership credential with the **swarm's group keypair**, populating `swarm_public_key`,
/// `sig_alg`, and `signature`. The member fields (`member_public_key`, etc.) are caller-supplied
/// (from a verified [`EnrollmentRequest`]) and part of the signed preimage.
pub fn sign_credential(
    mut cred: MembershipCredential,
    group_keypair: &libp2p::identity::Keypair,
) -> Result<MembershipCredential, MembershipError> {
    let ed = group_keypair
        .public()
        .try_into_ed25519()
        .map_err(|e| MembershipError::Crypto(format!("not ed25519: {e}")))?;
    cred.swarm_public_key = hex::encode(ed.to_bytes());
    cred.sig_alg = SigAlg::Ed25519.to_u8();
    check_label(&cred.swarm_label, "swarm_label")?;
    check_no_newline(&cred.member_public_key, "member_public_key")?;
    check_no_newline(&cred.member_openhydra_peer_id, "member_openhydra_peer_id")?;
    // A credential must actually bind a member; refuse an empty member key at sign time.
    if cred.member_public_key.is_empty() {
        return Err(MembershipError::MissingSignature);
    }
    let sig = group_keypair
        .sign(&cred_canonical_bytes(&cred))
        .map_err(|e| MembershipError::Crypto(format!("sign failed: {e}")))?;
    cred.signature = b64_encode(&sig);
    Ok(cred)
}

/// Verify a membership credential end-to-end: schema, wellformedness, the group-key signature over
/// the preimage, revocation (by `member_public_key`), and — LAST, only once the signature is trusted
/// — expiry. `now_ms` and `revoked` are passed in (no hidden clock, testable). Does **not** by itself
/// bind the credential to a live connection: the caller derives [`credential_member_peer_id`] and
/// checks it against the peer it's serving (that binding is the M4 serve gate).
pub fn verify_credential(
    cred: &MembershipCredential,
    now_ms: u64,
    revoked: &BTreeSet<String>,
) -> Result<VerifiedCredential, MembershipError> {
    if cred.schema_version > MEMBERSHIP_SCHEMA_VERSION {
        return Err(MembershipError::UnsupportedSchema(cred.schema_version));
    }
    check_label(&cred.swarm_label, "swarm_label")?;
    check_no_newline(&cred.member_public_key, "member_public_key")?;
    check_no_newline(&cred.member_openhydra_peer_id, "member_openhydra_peer_id")?;
    if cred.signature.is_empty() || cred.swarm_public_key.is_empty() || cred.member_public_key.is_empty()
    {
        return Err(MembershipError::MissingSignature);
    }
    let alg = SigAlg::from_u8(cred.sig_alg).map_err(|e| MembershipError::Crypto(e.to_string()))?;
    if !alg.is_implemented() {
        return Err(MembershipError::Crypto(format!("unsupported signature algorithm: {alg:?}")));
    }
    // Verify the group-key signature.
    let group_pk = ed_pubkey_from_hex(&cred.swarm_public_key)?;
    let sig = b64_decode(&cred.signature)?;
    verify_with_alg(alg, &group_pk, &cred_canonical_bytes(cred), &sig)?;
    // `member_public_key` must be a real Ed25519 key (so a serve gate can derive its peer id). This
    // also rejects a credential whose member key can't map to any identity.
    ed_pubkey_from_hex(&cred.member_public_key)?;
    // Revocation (owner-local list of member keys). Checked before expiry: a revoked member is out
    // regardless of the clock.
    if revoked.contains(&cred.member_public_key) {
        return Err(MembershipError::Revoked(cred.member_public_key.clone()));
    }
    // Signature is good → the timestamps are now trustworthy.
    if now_ms >= cred.expires_at {
        return Err(MembershipError::Expired { expires_at: cred.expires_at, now_ms });
    }
    Ok(VerifiedCredential { credential: cred.clone() })
}

/// Verify a credential AND bind it to an expected member public key (hex) — the convenience the M4
/// serve gate wants: "does this credential authorise *this* identity on a swarm I trust?". Compares
/// keys case-insensitively (hex).
pub fn verify_credential_for_member(
    cred: &MembershipCredential,
    expected_member_public_key: &str,
    now_ms: u64,
    revoked: &BTreeSet<String>,
) -> Result<VerifiedCredential, MembershipError> {
    let v = verify_credential(cred, now_ms, revoked)?;
    if !v.credential.member_public_key.eq_ignore_ascii_case(expected_member_public_key) {
        return Err(MembershipError::MemberMismatch {
            expected: expected_member_public_key.to_string(),
            found: v.credential.member_public_key.clone(),
        });
    }
    Ok(v)
}

/// Derive the libp2p peer id of the credential's authorised member, from `member_public_key`. The M4
/// serve gate compares this to the peer id of the live connection it's about to serve.
pub fn credential_member_peer_id(
    cred: &MembershipCredential,
) -> Result<libp2p::PeerId, MembershipError> {
    let ed = ed_pubkey_from_hex(&cred.member_public_key)?;
    Ok(libp2p::PeerId::from_public_key(&libp2p::identity::PublicKey::from(ed)))
}

/// Generate a fresh Ed25519 **group keypair** for a new swarm; returns `(public_hex, secret_hex)`.
/// The secret stays on the owner's machine (persisted 0600 by the agent) and is never transmitted;
/// it is reconstructed only to [`sign_credential_with_secret_hex`]. Kept here so the agent crate
/// stays libp2p-free (parity with how card signing lives in the network crate).
pub fn generate_group_keypair_hex() -> Result<(String, String), MembershipError> {
    let kp = libp2p::identity::Keypair::generate_ed25519();
    let ed = kp
        .try_into_ed25519()
        .map_err(|e| MembershipError::Crypto(format!("not ed25519: {e}")))?;
    let public = hex::encode(ed.public().to_bytes());
    let secret = hex::encode(ed.secret().as_ref());
    Ok((public, secret))
}

/// Sign a credential using a swarm's stored **group secret** (hex). Reconstructs the group keypair,
/// scrubs the transient secret copy, and delegates to [`sign_credential`] (which sets
/// `swarm_public_key`/`sig_alg`/`signature`). The agent calls this with the secret it holds on disk.
pub fn sign_credential_with_secret_hex(
    cred: MembershipCredential,
    group_secret_hex: &str,
) -> Result<MembershipCredential, MembershipError> {
    let kp = keypair_from_secret_hex(group_secret_hex)?;
    sign_credential(cred, &kp)
}

/// Reconstruct an Ed25519 keypair from a 32-byte secret (hex). Scrubs the decoded bytes.
fn keypair_from_secret_hex(secret_hex: &str) -> Result<libp2p::identity::Keypair, MembershipError> {
    use zeroize::Zeroize;
    let mut bytes = hex::decode(secret_hex)
        .map_err(|e| MembershipError::Crypto(format!("bad secret hex: {e}")))?;
    if bytes.len() != 32 {
        let n = bytes.len();
        bytes.zeroize();
        return Err(MembershipError::Crypto(format!("secret must be 32 bytes, got {n}")));
    }
    let mut arr: [u8; 32] = match bytes.as_slice().try_into() {
        Ok(a) => a,
        Err(_) => {
            bytes.zeroize();
            return Err(MembershipError::Crypto("secret length".into()));
        }
    };
    // `try_from_bytes` zeroizes `arr` on success; scrub the Vec copy regardless.
    let secret = libp2p::identity::ed25519::SecretKey::try_from_bytes(&mut arr)
        .map_err(|e| MembershipError::Crypto(format!("bad ed25519 secret: {e}")));
    bytes.zeroize();
    let secret = secret?;
    Ok(libp2p::identity::Keypair::from(libp2p::identity::ed25519::Keypair::from(secret)))
}

/// The hex-encoded Ed25519 **public** key of a node identity keypair — so the agent can compute a
/// member's key for binding without naming libp2p directly.
pub fn keypair_public_hex(
    keypair: &libp2p::identity::Keypair,
) -> Result<String, MembershipError> {
    let ed = keypair
        .public()
        .try_into_ed25519()
        .map_err(|e| MembershipError::Crypto(format!("not ed25519: {e}")))?;
    Ok(hex::encode(ed.to_bytes()))
}

fn ed_pubkey_from_hex(
    hex_str: &str,
) -> Result<libp2p::identity::ed25519::PublicKey, MembershipError> {
    let bytes =
        hex::decode(hex_str).map_err(|e| MembershipError::Crypto(format!("bad public_key hex: {e}")))?;
    libp2p::identity::ed25519::PublicKey::try_from_bytes(&bytes)
        .map_err(|e| MembershipError::Crypto(format!("bad ed25519 public_key: {e}")))
}

/// A short, human-verifiable fingerprint of a public key (hex) for out-of-band confirmation before
/// approval — anti-mis-enrollment. `SHA256(pubkey_bytes)[..8]` rendered as four space-separated
/// upper-hex groups (e.g. `A1B2 C3D4 E5F6 0718`). Returns "invalid-key" for un-decodable input rather
/// than erroring (this is a display aid, always called on already-validated keys in practice).
pub fn key_fingerprint(public_key_hex: &str) -> String {
    let Ok(bytes) = hex::decode(public_key_hex) else {
        return "invalid-key".to_string();
    };
    let digest = Sha256::digest(&bytes);
    let short = &digest[..8];
    let hex = hex::encode_upper(short);
    hex.as_bytes()
        .chunks(4)
        .map(|c| std::str::from_utf8(c).unwrap_or(""))
        .collect::<Vec<_>>()
        .join(" ")
}

impl EnrollmentRequest {
    /// A new UNSIGNED enrollment request; [`sign_enrollment_request`] fills `member_public_key`,
    /// `sig_alg`, and `signature`.
    pub fn new_unsigned(
        member_openhydra_peer_id: impl Into<String>,
        swarm_public_key: impl Into<String>,
        label: impl Into<String>,
        requested_at: u64,
    ) -> EnrollmentRequest {
        EnrollmentRequest {
            schema_version: MEMBERSHIP_SCHEMA_VERSION,
            swarm_public_key: swarm_public_key.into(),
            member_openhydra_peer_id: member_openhydra_peer_id.into(),
            member_public_key: String::new(),
            label: label.into(),
            requested_at,
            sig_alg: default_sig_alg(),
            signature: String::new(),
        }
    }

    pub fn to_json(&self) -> Result<String, MembershipError> {
        serde_json::to_string_pretty(self).map_err(|e| MembershipError::Crypto(format!("json: {e}")))
    }
    pub fn from_json(s: &str) -> Result<EnrollmentRequest, MembershipError> {
        serde_json::from_str(s).map_err(|e| MembershipError::Crypto(format!("json: {e}")))
    }
    pub fn to_magnet(&self) -> Result<String, MembershipError> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf)
            .map_err(|e| MembershipError::Crypto(format!("cbor: {e}")))?;
        Ok(format!("{ENROLL_MAGNET_PREFIX}{}", b64_encode(&buf)))
    }
    pub fn from_magnet(s: &str) -> Result<EnrollmentRequest, MembershipError> {
        let b64 = s
            .trim()
            .strip_prefix(ENROLL_MAGNET_PREFIX)
            .ok_or_else(|| MembershipError::Crypto("not an openhydra:enroll: string".into()))?;
        let buf = b64_decode(b64)?;
        ciborium::from_reader(&buf[..]).map_err(|e| MembershipError::Crypto(format!("cbor: {e}")))
    }
}

impl MembershipCredential {
    /// A new UNSIGNED credential binding a member; [`sign_credential`] fills `swarm_public_key`,
    /// `sig_alg`, and `signature`.
    pub fn new_unsigned(
        member_public_key: impl Into<String>,
        member_openhydra_peer_id: impl Into<String>,
        swarm_label: impl Into<String>,
        issued_at: u64,
        expires_at: u64,
    ) -> MembershipCredential {
        MembershipCredential {
            schema_version: MEMBERSHIP_SCHEMA_VERSION,
            swarm_public_key: String::new(),
            member_public_key: member_public_key.into(),
            member_openhydra_peer_id: member_openhydra_peer_id.into(),
            swarm_label: swarm_label.into(),
            issued_at,
            expires_at,
            sig_alg: default_sig_alg(),
            signature: String::new(),
        }
    }

    pub fn to_json(&self) -> Result<String, MembershipError> {
        serde_json::to_string_pretty(self).map_err(|e| MembershipError::Crypto(format!("json: {e}")))
    }
    pub fn from_json(s: &str) -> Result<MembershipCredential, MembershipError> {
        serde_json::from_str(s).map_err(|e| MembershipError::Crypto(format!("json: {e}")))
    }
    pub fn to_magnet(&self) -> Result<String, MembershipError> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf)
            .map_err(|e| MembershipError::Crypto(format!("cbor: {e}")))?;
        Ok(format!("{CRED_MAGNET_PREFIX}{}", b64_encode(&buf)))
    }
    pub fn from_magnet(s: &str) -> Result<MembershipCredential, MembershipError> {
        let b64 = s
            .trim()
            .strip_prefix(CRED_MAGNET_PREFIX)
            .ok_or_else(|| MembershipError::Crypto("not an openhydra:cred: string".into()))?;
        let buf = b64_decode(b64)?;
        ciborium::from_reader(&buf[..]).map_err(|e| MembershipError::Crypto(format!("cbor: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A stable "now" between issue and expiry for the fixtures below.
    fn now_before_expiry() -> u64 {
        1_750_000_000_000
    }
    fn issued() -> u64 {
        1_700_000_000_000
    }
    fn expires() -> u64 {
        1_800_000_000_000
    }
    fn no_revocations() -> BTreeSet<String> {
        BTreeSet::new()
    }

    fn a_request(member: &libp2p::identity::Keypair, swarm_pk: &str) -> EnrollmentRequest {
        let req = EnrollmentRequest::new_unsigned(
            "oh_member",
            swarm_pk,
            "Sam's MacBook",
            issued(),
        );
        sign_enrollment_request(req, member).unwrap()
    }

    fn a_credential(
        group: &libp2p::identity::Keypair,
        member_pk_hex: &str,
    ) -> MembershipCredential {
        let cred = MembershipCredential::new_unsigned(
            member_pk_hex,
            "oh_member",
            "Home rig",
            issued(),
            expires(),
        );
        sign_credential(cred, group).unwrap()
    }

    fn pubkey_hex(kp: &libp2p::identity::Keypair) -> String {
        hex::encode(kp.public().try_into_ed25519().unwrap().to_bytes())
    }

    #[test]
    fn enrollment_request_round_trips() {
        let member = libp2p::identity::Keypair::generate_ed25519();
        let req = a_request(&member, "");
        let v = verify_enrollment_request(&req).unwrap();
        assert_eq!(v.request.member_public_key, pubkey_hex(&member));
        assert_eq!(v.request.label, "Sam's MacBook");
    }

    #[test]
    fn a_tampered_enrollment_request_is_rejected() {
        let member = libp2p::identity::Keypair::generate_ed25519();
        let req = a_request(&member, "");
        // Swap in a different member key (as if claiming someone else's identity): the signature no
        // longer matches the preimage.
        let other = libp2p::identity::Keypair::generate_ed25519();
        let mut t = req.clone();
        t.member_public_key = pubkey_hex(&other);
        assert_eq!(verify_enrollment_request(&t).unwrap_err(), MembershipError::BadSignature);
        // Tampering the label after signing also breaks it.
        let mut t2 = req.clone();
        t2.label = "attacker".into();
        assert_eq!(verify_enrollment_request(&t2).unwrap_err(), MembershipError::BadSignature);
    }

    #[test]
    fn credential_sign_then_verify_round_trips() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let cred = a_credential(&group, &pubkey_hex(&member));
        assert!(!cred.signature.is_empty());
        assert_eq!(cred.swarm_public_key, pubkey_hex(&group));
        let v = verify_credential(&cred, now_before_expiry(), &no_revocations()).unwrap();
        assert_eq!(v.credential.member_public_key, pubkey_hex(&member));
        // The bound member's peer id derives from its key and equals the member node's peer id.
        assert_eq!(
            credential_member_peer_id(&cred).unwrap(),
            member.public().to_peer_id()
        );
    }

    #[test]
    fn a_credential_from_a_wrong_group_key_is_rejected() {
        // A credential must be signed by the swarm's OWN group key. Sign with group A's key but claim
        // group B's public key → the signature can't verify against the claimed anchor.
        let group_a = libp2p::identity::Keypair::generate_ed25519();
        let group_b = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let mut cred = a_credential(&group_a, &pubkey_hex(&member));
        // Repoint the anchor to B while keeping A's signature.
        cred.swarm_public_key = pubkey_hex(&group_b);
        assert_eq!(
            verify_credential(&cred, now_before_expiry(), &no_revocations()).unwrap_err(),
            MembershipError::BadSignature
        );
    }

    #[test]
    fn expired_credential_is_rejected_after_the_signature_verifies() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let cred = a_credential(&group, &pubkey_hex(&member));
        let err = verify_credential(&cred, cred.expires_at, &no_revocations()).unwrap_err();
        assert!(matches!(err, MembershipError::Expired { .. }));
        // One ms before expiry still passes.
        assert!(verify_credential(&cred, cred.expires_at - 1, &no_revocations()).is_ok());
    }

    #[test]
    fn a_revoked_member_is_rejected_even_before_expiry() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let cred = a_credential(&group, &pubkey_hex(&member));
        let mut revoked = BTreeSet::new();
        revoked.insert(pubkey_hex(&member));
        let err = verify_credential(&cred, now_before_expiry(), &revoked).unwrap_err();
        assert!(matches!(err, MembershipError::Revoked(_)));
    }

    #[test]
    fn tampering_any_signed_credential_field_breaks_the_signature() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let cred = a_credential(&group, &pubkey_hex(&member));
        // Extend expiry after signing.
        let mut t = cred.clone();
        t.expires_at += 1;
        assert_eq!(
            verify_credential(&t, now_before_expiry(), &no_revocations()).unwrap_err(),
            MembershipError::BadSignature
        );
        // Swap the authorised member to a different key.
        let attacker = libp2p::identity::Keypair::generate_ed25519();
        let mut t2 = cred.clone();
        t2.member_public_key = pubkey_hex(&attacker);
        assert_eq!(
            verify_credential(&t2, now_before_expiry(), &no_revocations()).unwrap_err(),
            MembershipError::BadSignature
        );
    }

    #[test]
    fn a_flipped_signature_byte_is_rejected() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let mut cred = a_credential(&group, &pubkey_hex(&member));
        let mut raw = b64_decode(&cred.signature).unwrap();
        raw[0] ^= 0x01;
        cred.signature = b64_encode(&raw);
        assert_eq!(
            verify_credential(&cred, now_before_expiry(), &no_revocations()).unwrap_err(),
            MembershipError::BadSignature
        );
    }

    #[test]
    fn verify_for_member_binds_the_expected_identity() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let cred = a_credential(&group, &pubkey_hex(&member));
        // Right member: ok.
        assert!(verify_credential_for_member(
            &cred,
            &pubkey_hex(&member),
            now_before_expiry(),
            &no_revocations()
        )
        .is_ok());
        // A different member's key: mismatch (even though the credential itself is validly signed).
        let other = libp2p::identity::Keypair::generate_ed25519();
        let err = verify_credential_for_member(
            &cred,
            &pubkey_hex(&other),
            now_before_expiry(),
            &no_revocations(),
        )
        .unwrap_err();
        assert!(matches!(err, MembershipError::MemberMismatch { .. }));
    }

    #[test]
    fn a_newer_schema_is_rejected() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let mut cred = MembershipCredential::new_unsigned(
            pubkey_hex(&member),
            "oh_member",
            "Home rig",
            issued(),
            expires(),
        );
        cred.schema_version = MEMBERSHIP_SCHEMA_VERSION + 1;
        let cred = sign_credential(cred, &group).unwrap();
        assert_eq!(
            verify_credential(&cred, now_before_expiry(), &no_revocations()).unwrap_err(),
            MembershipError::UnsupportedSchema(MEMBERSHIP_SCHEMA_VERSION + 1)
        );
    }

    #[test]
    fn a_newline_or_overlong_label_is_rejected_on_sign_and_verify() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        // Newline in the label → rejected on sign (keeps the preimage injective).
        let nl = MembershipCredential::new_unsigned(
            pubkey_hex(&member),
            "oh_member",
            "home\nmalicious=1",
            issued(),
            expires(),
        );
        assert!(matches!(sign_credential(nl, &group), Err(MembershipError::Malformed(_))));
        // Overlong label → rejected on sign.
        let long = MembershipCredential::new_unsigned(
            pubkey_hex(&member),
            "oh_member",
            "x".repeat(MAX_LABEL_LEN + 1),
            issued(),
            expires(),
        );
        assert!(matches!(sign_credential(long, &group), Err(MembershipError::Malformed(_))));
        // A hostile pre-signed credential with a newline label (bypassing the sign guard) is rejected
        // on verify too.
        let mut c = MembershipCredential::new_unsigned(
            pubkey_hex(&member),
            "oh_member",
            "ok",
            issued(),
            expires(),
        );
        c = sign_credential(c, &group).unwrap();
        c.swarm_label = "x\ny".into();
        assert!(matches!(
            verify_credential(&c, now_before_expiry(), &no_revocations()),
            Err(MembershipError::Malformed(_))
        ));
    }

    #[test]
    fn json_and_magnet_round_trip() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let cred = a_credential(&group, &pubkey_hex(&member));
        let from_json = MembershipCredential::from_json(&cred.to_json().unwrap()).unwrap();
        assert_eq!(from_json, cred);
        let from_magnet = MembershipCredential::from_magnet(&cred.to_magnet().unwrap()).unwrap();
        assert_eq!(from_magnet, cred);
        assert!(verify_credential(&from_json, now_before_expiry(), &no_revocations()).is_ok());

        let req = a_request(&member, &pubkey_hex(&group));
        assert_eq!(EnrollmentRequest::from_json(&req.to_json().unwrap()).unwrap(), req);
        assert_eq!(EnrollmentRequest::from_magnet(&req.to_magnet().unwrap()).unwrap(), req);
    }

    #[test]
    fn magnet_prefixes_are_distinct_and_checked() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let cred = a_credential(&group, &pubkey_hex(&member));
        let req = a_request(&member, "");
        // A credential magnet is not an enrollment magnet and vice-versa.
        assert!(EnrollmentRequest::from_magnet(&cred.to_magnet().unwrap()).is_err());
        assert!(MembershipCredential::from_magnet(&req.to_magnet().unwrap()).is_err());
        assert!(MembershipCredential::from_magnet("openhydra:card:abc").is_err());
    }

    #[test]
    fn no_private_key_material_appears_in_any_serialized_artifact() {
        // The whole point: enrollment moves only PUBLIC keys. Assert neither the group nor the member
        // secret bytes ever appear in a request or a credential (JSON or magnet).
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let group_secret = hex::encode(
            group.clone().try_into_ed25519().unwrap().secret().as_ref(),
        );
        let member_secret = hex::encode(
            member.clone().try_into_ed25519().unwrap().secret().as_ref(),
        );
        let cred = a_credential(&group, &pubkey_hex(&member));
        let req = a_request(&member, &pubkey_hex(&group));
        for blob in [
            cred.to_json().unwrap(),
            cred.to_magnet().unwrap(),
            req.to_json().unwrap(),
            req.to_magnet().unwrap(),
        ] {
            let lower = blob.to_ascii_lowercase();
            assert!(!lower.contains(&group_secret), "group secret leaked in: {blob}");
            assert!(!lower.contains(&member_secret), "member secret leaked in: {blob}");
        }
    }

    #[test]
    fn fingerprint_is_stable_grouped_hex() {
        let member = libp2p::identity::Keypair::generate_ed25519();
        let pk = pubkey_hex(&member);
        let fp = key_fingerprint(&pk);
        // Four 4-char groups, space-joined = 19 chars; stable across calls.
        assert_eq!(fp.len(), 19);
        assert_eq!(fp, key_fingerprint(&pk));
        assert_eq!(fp.matches(' ').count(), 3);
        // Different key → different fingerprint (overwhelmingly).
        let other = libp2p::identity::Keypair::generate_ed25519();
        assert_ne!(fp, key_fingerprint(&pubkey_hex(&other)));
        assert_eq!(key_fingerprint("nothex"), "invalid-key");
    }

    #[test]
    fn an_overlong_id_field_is_rejected() {
        // Review #5: `member_openhydra_peer_id` / the `swarm_public_key` hint are bounded, so a
        // self-signed request can't carry a multi-MB field that then persists + echoes into a credential.
        let member = libp2p::identity::Keypair::generate_ed25519();
        let mut req = EnrollmentRequest::new_unsigned(
            "x".repeat(MAX_ID_LEN + 1),
            "",
            "ok",
            issued(),
        );
        assert!(matches!(sign_enrollment_request(req, &member), Err(MembershipError::Malformed(_))));
        // Overlong swarm hint likewise.
        req = EnrollmentRequest::new_unsigned("oh_m", "a".repeat(MAX_ID_LEN + 1), "ok", issued());
        assert!(matches!(sign_enrollment_request(req, &member), Err(MembershipError::Malformed(_))));
    }

    #[test]
    fn an_unimplemented_sig_alg_is_rejected_not_downgraded() {
        let group = libp2p::identity::Keypair::generate_ed25519();
        let member = libp2p::identity::Keypair::generate_ed25519();
        let mut cred = a_credential(&group, &pubkey_hex(&member));
        // Claim ML-DSA (reserved, not implemented): must be refused, never silently treated as Ed25519.
        cred.sig_alg = SigAlg::MlDsa65.to_u8();
        assert!(matches!(
            verify_credential(&cred, now_before_expiry(), &no_revocations()),
            Err(MembershipError::Crypto(_))
        ));
    }
}
