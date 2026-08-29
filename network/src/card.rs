//! `.openhydra` card — a signed, secret-free public descriptor for a provider+model (M2).
//!
//! A card is a portable "magnet for a model": the provider signs its own **public** claims
//! (identity, model, capability, posture) with its Ed25519 identity, so the peer id *owns* the
//! claims and the file is tamper-evident. Someone imports the card to add the model to their client
//! **without waiting for live discovery** — verify the signature, then dial the `libp2p_peer_id`
//! directly (the swarm resolves current addresses; the card's `addr_hints` are only hints).
//!
//! Design invariants (see `docs/PORTABLE_DISCOVERY_SWARM_AND_ADS_PLAN_v2.md`):
//! * **Signed, or it's a rumor.** The signature covers every field except itself, over a
//!   domain-separated preimage that binds the `sig_alg` — mirrors [`crate::dht::sign_peer_record`].
//! * **No secret.** A card carries no key material beyond the provider's *public* key + a signature.
//!   Re-sharing a card grants no access; swarm membership is separate keypair auth (M3).
//! * **No path/username leak.** `model_id` is the CLEAN handle (never a raw engine `-m` path); this
//!   is asserted on both sign and verify ([`card_is_privacy_safe`]) — the card is a public identity
//!   surface even in the private-group case.
//! * **Self-claims are hints.** capability / pricing / posture are provider-declared and only
//!   *tamper-evident* (signed), not *trustworthy* — trust comes from verification/reputation/TEE.

use serde::{Deserialize, Serialize};

use openhydra_protocol::crypto_agility::SigAlg;

/// The schema version new cards are written at. A verifier rejects a card from a *newer* schema it
/// can't fully validate rather than trusting fields it doesn't understand.
pub const CARD_SCHEMA_VERSION: u32 = 1;

/// Domain-separation header for the signing preimage (bumped with any preimage layout change, so a
/// signature can never be replayed across card versions or against another OpenHydra signed type).
const CARD_DOMAIN: &str = "openhydra-card-v1";

/// Magnet-string scheme prefix: `openhydra:card:<base64url(cbor)>`.
const MAGNET_PREFIX: &str = "openhydra:card:";

fn default_schema() -> u32 {
    CARD_SCHEMA_VERSION
}
fn default_sig_alg() -> u8 {
    SigAlg::Ed25519.to_u8()
}

/// How the provider offers the model. Declarative posture (no pricing engine exists yet — this is a
/// self-claim, not an enforced rate). `Unknown` degrades a future/typo'd value safely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PricingMode {
    /// Free, reciprocity-based (give-to-get) — the default.
    #[default]
    Reciprocal,
    /// Metered (credits/fiat); see [`RateCard`].
    Paid,
    /// Ad-supported (deferred tier).
    AdSupported,
    /// Unrecognised value from a newer/typo'd card — treated as the safe default at the UI.
    #[serde(other)]
    Unknown,
}
impl PricingMode {
    /// Stable tag for the signing preimage (never the Debug repr, which could change).
    fn tag(&self) -> &'static str {
        match self {
            PricingMode::Reciprocal => "reciprocal",
            PricingMode::Paid => "paid",
            PricingMode::AdSupported => "ad_supported",
            PricingMode::Unknown => "unknown",
        }
    }
}

/// Self-declared rate for a `Paid` card (per 1M tokens, split in/out). `unit` names the currency
/// (e.g. "credits"). Present only when the provider meters; free/reciprocal cards omit it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RateCard {
    pub in_per_mtok: u64,
    pub out_per_mtok: u64,
    pub unit: String,
}

/// Self-declared capability hints (needed for routing/tiering; a *hint* until verified).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    /// Parameter size label (e.g. "7.6b"), from the detected model.
    #[serde(default)]
    pub params: String,
    #[serde(default)]
    pub context_length: u32,
    #[serde(default)]
    pub max_output_tokens: u32,
    /// e.g. ["text"], ["text","image"]. Rendered comma-joined in the preimage.
    #[serde(default)]
    pub modalities: Vec<String>,
}

/// Content posture the importer sees *before* routing. Self-declared.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AupFlags {
    #[serde(default)]
    pub uncensored: bool,
    #[serde(default)]
    pub nsfw: bool,
}

/// A signed, secret-free provider+model descriptor. Serializes to JSON (a `.openhydra` file) or a
/// base64url CBOR magnet string; both deserialize to this and go through [`verify_card`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Card {
    #[serde(default = "default_schema")]
    pub schema_version: u32,

    // ── identity: trust anchor + dial target (populated by `sign_card`) ──
    /// The OpenHydra identity id (reputation/record key).
    pub openhydra_peer_id: String,
    /// The libp2p peer id — the dial target; bound to `public_key` on verify.
    #[serde(default)]
    pub libp2p_peer_id: String,
    /// Ed25519 public key (hex) that produced `signature`.
    #[serde(default)]
    pub public_key: String,

    // ── model (CLEAN handle only — never a path) ──
    /// Normalized engine handle (e.g. `llama3.2:1b`, `Qwen/Qwen2.5-7B`). NEVER a filesystem path.
    pub model_id: String,
    /// Canonical id `family/params/quant/template_hash` (or "" if unknown).
    #[serde(default)]
    pub canonical_id: String,
    /// Weight/template hash tying the card to *this* build (or "").
    #[serde(default)]
    pub weight_hash: String,

    // ── self-declared hints (tamper-evident, not trusted) ──
    #[serde(default)]
    pub capability: Capability,
    #[serde(default)]
    pub pricing_mode: PricingMode,
    #[serde(default)]
    pub rate_card: Option<RateCard>,
    #[serde(default)]
    pub aup_flags: AupFlags,
    #[serde(default)]
    pub region: Option<String>,
    /// Multiaddr hints (NON-authoritative — the truth is resolved live by `libp2p_peer_id`).
    #[serde(default)]
    pub addr_hints: Vec<String>,

    // ── freshness + tamper-evidence ──
    pub signed_at: u64,
    pub expires_at: u64,
    #[serde(default = "default_sig_alg")]
    pub sig_alg: u8,
    /// base64url Ed25519 signature over [`card_canonical_bytes`]. Empty until signed.
    #[serde(default)]
    pub signature: String,
}

/// A card whose signature, key↔peer-id binding, schema, privacy, and expiry have all been checked.
/// The wrapped `card` is safe to act on (dial `card.libp2p_peer_id`, add `card.model_id`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCard {
    pub card: Card,
}

/// Everything that can be wrong with a card. `PartialEq` so tests can assert the exact variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardError {
    /// A model/canonical id that looks like a filesystem path or leaks a home dir / username.
    PathLeak(String),
    /// A fielded string carries a preimage delimiter (`\n`, or `,` in a list entry) — rejected so
    /// the signing preimage stays injective.
    Malformed(String),
    /// Card schema is newer than this build can validate.
    UnsupportedSchema(u32),
    /// No signature or no public key.
    MissingSignature,
    /// Signature did not verify against the preimage.
    BadSignature,
    /// `libp2p_peer_id` does not derive from `public_key` (key replayed under another id).
    PeerIdMismatch { claimed: String, derived: String },
    /// `now_ms >= expires_at`.
    Expired { expires_at: u64, now_ms: u64 },
    /// Encoding/decoding or key-parse failure.
    Crypto(String),
}

impl std::fmt::Display for CardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CardError::PathLeak(s) => write!(f, "card privacy check failed: {s}"),
            CardError::Malformed(s) => write!(f, "card is malformed: {s}"),
            CardError::UnsupportedSchema(v) => write!(f, "unsupported card schema version {v}"),
            CardError::MissingSignature => write!(f, "card missing signature or public key"),
            CardError::BadSignature => write!(f, "card signature verification failed"),
            CardError::PeerIdMismatch { claimed, derived } => {
                write!(f, "card peer_id mismatch: claimed={claimed} derived={derived}")
            }
            CardError::Expired { expires_at, now_ms } => {
                write!(f, "card expired (expires_at={expires_at} now={now_ms})")
            }
            CardError::Crypto(s) => write!(f, "card crypto error: {s}"),
        }
    }
}
impl std::error::Error for CardError {}

fn b64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE.encode(data)
}
fn b64_decode(s: &str) -> Result<Vec<u8>, CardError> {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE
        .decode(s)
        .map_err(|e| CardError::Crypto(format!("base64 decode: {e}")))
}

/// Whether a string looks like an absolute filesystem path (mirrors the path-detection in the
/// adapter's `normalize_engine_ref`, kept here so the network crate can guard independently). A
/// namespaced logical id like `Qwen/Qwen2.5-7B` is NOT a path (no leading `/`, no `.gguf`), so a bare
/// `/` mid-string is allowed; a leading `/`, `~`, `\`, a `.gguf` suffix, or a Windows drive is not.
fn is_pathlike(s: &str) -> bool {
    let t = s.trim();
    if t.starts_with('/') || t.starts_with('~') || t.starts_with('\\') {
        return true;
    }
    if t.to_ascii_lowercase().ends_with(".gguf") {
        return true;
    }
    // Windows drive: `X:\` or `X:/`.
    let b = t.as_bytes();
    b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'/' || b[2] == b'\\')
}

/// Home-dir / user / weight-file fragments that must never appear (mid-string) in a clean handle.
const LEAK_MARKERS: [&str; 7] =
    ["/home/", "/users/", "\\users\\", "c:\\", ".gguf", ".safetensors", ".bin"];

/// Reject a card that would leak a filesystem path, home dir, username, or weight-file name through
/// any of its clean-handle fields. Applied on BOTH sign (never emit a leaky card) and verify (never
/// trust one). `addr_hints` are multiaddrs and legitimately contain `/`, so they are not scanned
/// here (a `/unix/` hint check belongs with hint emission, which currently emits none).
pub fn card_is_privacy_safe(card: &Card) -> Result<(), CardError> {
    // Review fix: the marker scan runs over `canonical_id` and `weight_hash` too, not just
    // `model_id` — `canonical_id` (`family/params/quant/template_hash`) is machine-derived and the
    // field most likely to inherit a weight-file path/username, and it's signed + re-shared.
    for (field, val) in [
        ("model_id", &card.model_id),
        ("canonical_id", &card.canonical_id),
        ("weight_hash", &card.weight_hash),
    ] {
        if is_pathlike(val) {
            return Err(CardError::PathLeak(format!("{field} looks like a path: {val:?}")));
        }
        let hay = val.to_ascii_lowercase();
        for marker in LEAK_MARKERS {
            if hay.contains(marker) {
                return Err(CardError::PathLeak(format!(
                    "{field} contains leak marker {marker:?}: {val:?}"
                )));
            }
        }
    }
    // A `/unix/<path>` multiaddr hint would bake a filesystem path (+ username) into the signed card.
    // `build_card` never emits addr_hints and the consumer ignores them (it dials by peer id), so
    // this is inert today — but reject it so a hand-crafted hint can't leak a path.
    for hint in &card.addr_hints {
        if hint.to_ascii_lowercase().starts_with("/unix/") {
            return Err(CardError::PathLeak(format!("addr_hint is a filesystem path: {hint:?}")));
        }
    }
    Ok(())
}

/// Reject a card whose fielded strings carry a `card_canonical_bytes` delimiter — a `\n` in any
/// string field, or a `,` in a `modalities`/`addr_hints` entry. Those would make the preimage
/// non-injective (two distinct cards sharing one signature). Applied on sign + verify. The
/// routing-trust fields (`public_key`/`libp2p_peer_id`) are machine-generated hex/base58 and can't
/// carry a delimiter, so this closes a robustness gap (parity with `dht::canonical_bytes`, which
/// renders only scalars), not a live forgery vector.
fn card_is_wellformed(card: &Card) -> Result<(), CardError> {
    let scalars = [
        card.model_id.as_str(),
        card.canonical_id.as_str(),
        card.weight_hash.as_str(),
        card.openhydra_peer_id.as_str(),
        card.capability.params.as_str(),
        card.region.as_deref().unwrap_or(""),
        card.rate_card.as_ref().map(|r| r.unit.as_str()).unwrap_or(""),
    ];
    for s in scalars {
        if s.contains('\n') {
            return Err(CardError::Malformed(format!("field contains a newline: {s:?}")));
        }
    }
    for m in &card.capability.modalities {
        if m.contains(',') || m.contains('\n') {
            return Err(CardError::Malformed(format!("modality contains a delimiter: {m:?}")));
        }
    }
    for h in &card.addr_hints {
        if h.contains(',') || h.contains('\n') {
            return Err(CardError::Malformed(format!("addr_hint contains a delimiter: {h:?}")));
        }
    }
    Ok(())
}

/// Deterministic signing preimage: the domain header, the `sig_alg` discriminant, then every signed
/// field in a fixed order and rendering. Excludes only `signature`. Signer and verifier are both Rust
/// and reproduce this identically (mirrors [`crate::dht::canonical_bytes`]).
pub fn card_canonical_bytes(c: &Card) -> Vec<u8> {
    let (rin, rout, runit) = match &c.rate_card {
        Some(r) => (r.in_per_mtok, r.out_per_mtok, r.unit.as_str()),
        None => (0, 0, ""),
    };
    format!(
        "{CARD_DOMAIN}\nsig_alg={}\nschema_version={}\n\
         openhydra_peer_id={}\nlibp2p_peer_id={}\npublic_key={}\n\
         model_id={}\ncanonical_id={}\nweight_hash={}\n\
         cap_params={}\ncap_context_length={}\ncap_max_output_tokens={}\ncap_modalities={}\n\
         pricing_mode={}\nrate_in={}\nrate_out={}\nrate_unit={}\n\
         aup_uncensored={}\naup_nsfw={}\n\
         region={}\naddr_hints={}\n\
         signed_at={}\nexpires_at={}",
        c.sig_alg,
        c.schema_version,
        c.openhydra_peer_id,
        c.libp2p_peer_id,
        c.public_key,
        c.model_id,
        c.canonical_id,
        c.weight_hash,
        c.capability.params,
        c.capability.context_length,
        c.capability.max_output_tokens,
        c.capability.modalities.join(","),
        c.pricing_mode.tag(),
        rin,
        rout,
        runit,
        c.aup_flags.uncensored,
        c.aup_flags.nsfw,
        c.region.as_deref().unwrap_or(""),
        c.addr_hints.join(","),
        c.signed_at,
        c.expires_at,
    )
    .into_bytes()
}

/// Sign a card with `keypair`, populating `public_key`, `libp2p_peer_id`, `sig_alg`, and `signature`
/// (everything else — identity id, model, hints, timestamps — is caller-supplied and part of the
/// signed preimage). Refuses to emit a card that fails the privacy check.
pub fn sign_card(
    mut card: Card,
    keypair: &libp2p::identity::Keypair,
) -> Result<Card, CardError> {
    let ed = keypair
        .public()
        .try_into_ed25519()
        .map_err(|e| CardError::Crypto(format!("not ed25519: {e}")))?;
    card.public_key = hex::encode(ed.to_bytes());
    card.libp2p_peer_id = libp2p::PeerId::from_public_key(&keypair.public()).to_string();
    card.sig_alg = SigAlg::Ed25519.to_u8();
    // Never sign a card that would leak a path/username (defence: the caller should already pass a
    // clean handle, but the signature makes the leak permanent + re-shareable, so gate here too).
    card_is_privacy_safe(&card)?;
    // Keep the signing preimage injective — no fielded string may carry a delimiter.
    card_is_wellformed(&card)?;
    let canonical = card_canonical_bytes(&card);
    let sig = keypair
        .sign(&canonical)
        .map_err(|e| CardError::Crypto(format!("sign failed: {e}")))?;
    card.signature = b64_encode(&sig);
    Ok(card)
}

/// Verify a card end-to-end: schema, privacy, signature-algorithm, the Ed25519 signature over the
/// preimage, the `public_key` ↔ `libp2p_peer_id` binding, and (only after the signature is trusted)
/// expiry. Returns a [`VerifiedCard`] whose `libp2p_peer_id`/`model_id` are safe to act on.
///
/// `now_ms` is passed in (no hidden clock) so expiry is testable. Order matters: the expiry check
/// runs LAST, because `expires_at` lives inside the signed preimage — trust it only once the
/// signature has verified.
pub fn verify_card(card: &Card, now_ms: u64) -> Result<VerifiedCard, CardError> {
    if card.schema_version > CARD_SCHEMA_VERSION {
        return Err(CardError::UnsupportedSchema(card.schema_version));
    }
    card_is_privacy_safe(card)?;
    card_is_wellformed(card)?;
    if card.signature.is_empty() || card.public_key.is_empty() {
        return Err(CardError::MissingSignature);
    }
    let alg = SigAlg::from_u8(card.sig_alg).map_err(|e| CardError::Crypto(e.to_string()))?;
    if !alg.is_implemented() {
        return Err(CardError::Crypto(format!("unsupported signature algorithm: {alg:?}")));
    }
    let pk_bytes = hex::decode(&card.public_key)
        .map_err(|e| CardError::Crypto(format!("bad public_key hex: {e}")))?;
    let ed_pk = libp2p::identity::ed25519::PublicKey::try_from_bytes(&pk_bytes)
        .map_err(|e| CardError::Crypto(format!("bad ed25519 public_key: {e}")))?;
    // Bind key ↔ peer id UNCONDITIONALLY (the field is in the preimage; an empty/forged id must not
    // slip through) — mirrors dht::verify_peer_record's D-S4.
    if card.libp2p_peer_id.is_empty() {
        return Err(CardError::Crypto("missing libp2p_peer_id".into()));
    }
    let derived = libp2p::PeerId::from_public_key(&libp2p::identity::PublicKey::from(ed_pk.clone()))
        .to_string();
    if card.libp2p_peer_id != derived {
        return Err(CardError::PeerIdMismatch {
            claimed: card.libp2p_peer_id.clone(),
            derived,
        });
    }
    let sig = b64_decode(&card.signature)?;
    if !ed_pk.verify(&card_canonical_bytes(card), &sig) {
        return Err(CardError::BadSignature);
    }
    // Signature is good → the timestamps are now trustworthy.
    if now_ms >= card.expires_at {
        return Err(CardError::Expired { expires_at: card.expires_at, now_ms });
    }
    Ok(VerifiedCard { card: card.clone() })
}

impl Card {
    /// An UNSIGNED card with the essentials filled and every hint field defaulted. The crypto fields
    /// (`public_key`, `libp2p_peer_id`, `sig_alg`, `signature`) are populated by [`sign_card`]. Set
    /// `canonical_id`, `capability`, `pricing_mode`, `region`, etc. on the returned value before
    /// signing. Keeps card construction (and its defaults) in one place.
    pub fn new_unsigned(
        openhydra_peer_id: impl Into<String>,
        model_id: impl Into<String>,
        signed_at: u64,
        expires_at: u64,
    ) -> Card {
        Card {
            schema_version: CARD_SCHEMA_VERSION,
            openhydra_peer_id: openhydra_peer_id.into(),
            libp2p_peer_id: String::new(),
            public_key: String::new(),
            model_id: model_id.into(),
            canonical_id: String::new(),
            weight_hash: String::new(),
            capability: Capability::default(),
            pricing_mode: PricingMode::default(),
            rate_card: None,
            aup_flags: AupFlags::default(),
            region: None,
            addr_hints: Vec::new(),
            signed_at,
            expires_at,
            sig_alg: default_sig_alg(),
            signature: String::new(),
        }
    }

    /// Pretty JSON for a `.openhydra` file.
    pub fn to_json(&self) -> Result<String, CardError> {
        serde_json::to_string_pretty(self).map_err(|e| CardError::Crypto(format!("json encode: {e}")))
    }
    /// Parse a `.openhydra` JSON file (does NOT verify — call [`verify_card`] after).
    pub fn from_json(s: &str) -> Result<Card, CardError> {
        serde_json::from_str(s).map_err(|e| CardError::Crypto(format!("json decode: {e}")))
    }
    /// Compact copy-paste string: `openhydra:card:<base64url(cbor)>`.
    pub fn to_magnet(&self) -> Result<String, CardError> {
        let mut buf = Vec::new();
        ciborium::into_writer(self, &mut buf)
            .map_err(|e| CardError::Crypto(format!("cbor encode: {e}")))?;
        Ok(format!("{MAGNET_PREFIX}{}", b64_encode(&buf)))
    }
    /// Parse a magnet string (does NOT verify — call [`verify_card`] after).
    pub fn from_magnet(s: &str) -> Result<Card, CardError> {
        let b64 = s
            .trim()
            .strip_prefix(MAGNET_PREFIX)
            .ok_or_else(|| CardError::Crypto("not an openhydra:card: string".into()))?;
        let buf = b64_decode(b64)?;
        ciborium::from_reader(&buf[..]).map_err(|e| CardError::Crypto(format!("cbor decode: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_card() -> Card {
        Card {
            schema_version: CARD_SCHEMA_VERSION,
            openhydra_peer_id: "oh_test".into(),
            libp2p_peer_id: String::new(), // filled by sign
            public_key: String::new(),     // filled by sign
            model_id: "qwen3:1.7b".into(),
            canonical_id: "qwen3/1.7b/int4/abc123".into(),
            weight_hash: "abc123".into(),
            capability: Capability {
                params: "1.7b".into(),
                context_length: 32768,
                max_output_tokens: 4096,
                modalities: vec!["text".into()],
            },
            pricing_mode: PricingMode::Reciprocal,
            rate_card: None,
            aup_flags: AupFlags::default(),
            region: Some("in".into()),
            addr_hints: vec!["/ip4/1.2.3.4/tcp/4111".into()],
            signed_at: 1_700_000_000_000,
            expires_at: 1_800_000_000_000,
            sig_alg: default_sig_alg(),
            signature: String::new(),
        }
    }

    fn now_before_expiry() -> u64 {
        1_750_000_000_000
    }

    #[test]
    fn sign_then_verify_round_trips() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let signed = sign_card(a_card(), &kp).unwrap();
        assert!(!signed.signature.is_empty());
        assert!(!signed.public_key.is_empty());
        assert!(!signed.libp2p_peer_id.is_empty());
        let v = verify_card(&signed, now_before_expiry()).unwrap();
        assert_eq!(v.card.model_id, "qwen3:1.7b");
        // The bound peer id is the dial target.
        assert_eq!(v.card.libp2p_peer_id, signed.libp2p_peer_id);
    }

    #[test]
    fn expired_card_is_rejected_after_the_signature_verifies() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let signed = sign_card(a_card(), &kp).unwrap();
        let err = verify_card(&signed, signed.expires_at).unwrap_err();
        assert!(matches!(err, CardError::Expired { .. }));
        // One ms before expiry still passes.
        assert!(verify_card(&signed, signed.expires_at - 1).is_ok());
    }

    #[test]
    fn tampering_any_signed_field_breaks_the_signature() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let signed = sign_card(a_card(), &kp).unwrap();
        // A self-claim (price) is tamper-evident: flipping it after signing must fail.
        let mut t = signed.clone();
        t.pricing_mode = PricingMode::Paid;
        assert_eq!(verify_card(&t, now_before_expiry()).unwrap_err(), CardError::BadSignature);
        // Model id (still a valid non-path handle) tamper.
        let mut t2 = signed.clone();
        t2.model_id = "qwen3:8b".into();
        assert_eq!(verify_card(&t2, now_before_expiry()).unwrap_err(), CardError::BadSignature);
        // Extending expiry after signing.
        let mut t3 = signed.clone();
        t3.expires_at += 1;
        assert_eq!(verify_card(&t3, now_before_expiry()).unwrap_err(), CardError::BadSignature);
    }

    #[test]
    fn a_flipped_signature_byte_is_rejected() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut signed = sign_card(a_card(), &kp).unwrap();
        // Corrupt the signature (decode, flip a byte, re-encode) — stays valid base64.
        let mut raw = b64_decode(&signed.signature).unwrap();
        raw[0] ^= 0x01;
        signed.signature = b64_encode(&raw);
        assert_eq!(verify_card(&signed, now_before_expiry()).unwrap_err(), CardError::BadSignature);
    }

    #[test]
    fn a_key_replayed_under_another_peer_id_is_rejected() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let signed = sign_card(a_card(), &kp).unwrap();
        // Swap the libp2p_peer_id to a different (real) peer's id: the key no longer derives it.
        let other = libp2p::identity::Keypair::generate_ed25519();
        let mut t = signed.clone();
        t.libp2p_peer_id =
            libp2p::PeerId::from_public_key(&other.public()).to_string();
        assert!(matches!(verify_card(&t, now_before_expiry()), Err(CardError::PeerIdMismatch { .. })));
    }

    #[test]
    fn a_newer_schema_is_rejected() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut c = a_card();
        c.schema_version = CARD_SCHEMA_VERSION + 1;
        let signed = sign_card(c, &kp).unwrap();
        assert_eq!(
            verify_card(&signed, now_before_expiry()).unwrap_err(),
            CardError::UnsupportedSchema(CARD_SCHEMA_VERSION + 1)
        );
    }

    #[test]
    fn a_path_or_pii_model_id_is_never_signed_and_never_trusted() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        for leak in [
            "/home/alice/models/qwen3-1.7b-q4_k_m.gguf",
            "~/models/foo.gguf",
            "C:\\Users\\bob\\model.gguf",
            "qwen3-1.7b-Q4_K_M.gguf",
        ] {
            let mut c = a_card();
            c.model_id = leak.into();
            // sign refuses.
            assert!(matches!(sign_card(c.clone(), &kp), Err(CardError::PathLeak(_))), "sign must reject {leak:?}");
        }
        // A clean namespaced handle with a '/' (HF org/repo) is NOT flagged.
        let mut ok = a_card();
        ok.model_id = "Qwen/Qwen2.5-7B".into();
        assert!(sign_card(ok, &kp).is_ok());
    }

    #[test]
    fn a_leaky_card_signed_elsewhere_is_rejected_on_import() {
        // Simulate a hostile/buggy signer that bypassed the sign-side scrub: hand-build a card whose
        // preimage includes the path, sign it, then import → verify must still refuse it.
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut c = a_card();
        c.model_id = "/home/alice/secret.gguf".into();
        // Sign the leaky preimage directly (bypassing sign_card's guard).
        let ed = kp.public().try_into_ed25519().unwrap();
        c.public_key = hex::encode(ed.to_bytes());
        c.libp2p_peer_id = libp2p::PeerId::from_public_key(&kp.public()).to_string();
        c.sig_alg = SigAlg::Ed25519.to_u8();
        c.signature = b64_encode(&kp.sign(&card_canonical_bytes(&c)).unwrap());
        // The signature is valid, but verify still refuses on the privacy gate.
        assert!(matches!(verify_card(&c, now_before_expiry()), Err(CardError::PathLeak(_))));
    }

    #[test]
    fn exported_card_bytes_never_contain_a_path_or_home_dir() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let signed = sign_card(a_card(), &kp).unwrap();
        for blob in [signed.to_json().unwrap(), signed.to_magnet().unwrap()] {
            let lower = blob.to_ascii_lowercase();
            assert!(!lower.contains("/home/"), "leak in: {blob}");
            assert!(!lower.contains(".gguf"), "leak in: {blob}");
            assert!(!lower.contains("\\users\\"), "leak in: {blob}");
        }
    }

    #[test]
    fn json_and_magnet_round_trip_to_the_same_card() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let signed = sign_card(a_card(), &kp).unwrap();
        let from_json = Card::from_json(&signed.to_json().unwrap()).unwrap();
        assert_eq!(from_json, signed);
        let from_magnet = Card::from_magnet(&signed.to_magnet().unwrap()).unwrap();
        assert_eq!(from_magnet, signed);
        // Both still verify.
        assert!(verify_card(&from_json, now_before_expiry()).is_ok());
        assert!(verify_card(&from_magnet, now_before_expiry()).is_ok());
    }

    #[test]
    fn from_magnet_rejects_a_non_card_string() {
        assert!(Card::from_magnet("magnet:?xt=urn:btih:...").is_err());
        assert!(Card::from_magnet("openhydra:card:!!!not-base64!!!").is_err());
    }

    #[test]
    fn a_paid_card_round_trips_its_rate() {
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut c = a_card();
        c.pricing_mode = PricingMode::Paid;
        c.rate_card = Some(RateCard { in_per_mtok: 50, out_per_mtok: 150, unit: "credits".into() });
        let signed = sign_card(c, &kp).unwrap();
        let v = verify_card(&signed, now_before_expiry()).unwrap();
        assert_eq!(v.card.rate_card.unwrap().out_per_mtok, 150);
    }

    #[test]
    fn a_leaky_canonical_id_or_weight_hash_is_never_signed_or_trusted() {
        // Review fix (MEDIUM): the leak-marker scan must cover canonical_id + weight_hash, not only
        // model_id — otherwise a path/username/weight-file name in canonical_id is signed and trusted.
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let leaky = [
            ("qwen3/2b/int4/home/alice-hash", ""),   // "/home/" + username mid canonical_id
            ("qwen-2b-q4.safetensors", ""),          // weight-file name in canonical_id
            ("qwen3/2b/int4/ok", "MODEL.GGUF"),      // uppercase weight-file name in weight_hash
        ];
        for (cid, wh) in leaky {
            let mut c = a_card();
            c.canonical_id = cid.into();
            c.weight_hash = wh.into();
            assert!(
                matches!(sign_card(c, &kp), Err(CardError::PathLeak(_))),
                "must reject canonical_id={cid:?} weight_hash={wh:?}"
            );
        }
        // A clean, legitimately-slashed canonical_id still passes.
        assert!(sign_card(a_card(), &kp).is_ok());
        // A `/unix/<path>` addr_hint (would leak a filesystem path) is rejected.
        let mut u = a_card();
        u.addr_hints = vec!["/unix/home/alice/.openhydra/sock".into()];
        assert!(matches!(sign_card(u, &kp), Err(CardError::PathLeak(_))));
    }

    #[test]
    fn a_delimiter_in_a_fielded_string_is_rejected_on_sign_and_verify() {
        // Review fix (LOW): keep card_canonical_bytes injective — no field may carry a `\n`, and no
        // modality/addr_hint may carry a `,` (which would collide with join(",")).
        let kp = libp2p::identity::Keypair::generate_ed25519();
        let mut nl = a_card();
        nl.region = Some("us\nmalicious=1".into());
        assert!(matches!(sign_card(nl, &kp), Err(CardError::Malformed(_))));
        let mut comma_mod = a_card();
        comma_mod.capability.modalities = vec!["text,image".into()];
        assert!(matches!(sign_card(comma_mod, &kp), Err(CardError::Malformed(_))));
        let mut comma_hint = a_card();
        comma_hint.addr_hints = vec!["/ip4/1.2.3.4/tcp/1,evil".into()];
        assert!(matches!(sign_card(comma_hint, &kp), Err(CardError::Malformed(_))));

        // A hostile pre-signed malformed card (bypassing sign_card's gate) is rejected on verify too.
        let mut c = a_card();
        c.region = Some("x\ny".into());
        let ed = kp.public().try_into_ed25519().unwrap();
        c.public_key = hex::encode(ed.to_bytes());
        c.libp2p_peer_id = libp2p::PeerId::from_public_key(&kp.public()).to_string();
        c.sig_alg = SigAlg::Ed25519.to_u8();
        c.signature = b64_encode(&kp.sign(&card_canonical_bytes(&c)).unwrap());
        assert!(matches!(verify_card(&c, now_before_expiry()), Err(CardError::Malformed(_))));
    }
}
