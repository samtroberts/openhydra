//! Agent-side swarm glue (M3): persist swarm state and drive the owner/member operations on top of
//! the pure credential crypto in [`openhydra_network::membership`]. The desktop calls these in-process
//! (like card import), so identity/path handling stays here.
//!
//! State lives under `~/.openhydra/swarms/<swarm_public_key>.json`, one file per swarm:
//! * **Owner** files hold the group **secret** key (0600), the approved-member list, and a revocation
//!   set. The secret NEVER leaves this file — it is not serialized into any credential, view, or log.
//! * **Member** files hold our credential + the swarm's public key (no secret).
//!
//! Enrollment is **offline copy/paste** for v1 (no wire protocol): a member exports an
//! [`EnrollmentRequest`] string, the owner approves it into a [`MembershipCredential`] string, the
//! member accepts that back. Every artifact is public; the group secret stays on the owner's machine.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use openhydra_network::identity::Identity;
use openhydra_network::membership::{
    credential_member_peer_id, generate_group_keypair_hex, key_fingerprint, keypair_public_hex,
    sign_credential_with_secret_hex, sign_enrollment_request, verify_credential,
    verify_credential_for_member, verify_enrollment_request, EnrollmentRequest, MembershipCredential,
    MembershipError, MEMBERSHIP_SCHEMA_VERSION,
};

/// Schema version for the on-disk swarm record.
const SWARM_RECORD_SCHEMA: u32 = 1;

/// Whether this node owns the swarm (holds the group key) or is a member of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmRole {
    Owner,
    Member,
}

/// An approved member, as the owner records it (owner-side only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberRecord {
    pub member_public_key: String,
    #[serde(default)]
    pub member_openhydra_peer_id: String,
    #[serde(default)]
    pub label: String,
    pub issued_at: u64,
    pub expires_at: u64,
}

/// The on-disk swarm record (`~/.openhydra/swarms/<swarm_public_key>.json`). One struct covers both
/// roles; role-specific fields are optional and documented. **The owner's `group_secret_key` is the
/// only secret here and must never be serialized anywhere but this 0600 file.**
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmRecord {
    #[serde(default = "default_record_schema")]
    pub schema_version: u32,
    /// The swarm's Ed25519 group **public** key (hex) — the file key + trust anchor.
    pub swarm_public_key: String,
    /// Human label (owner-chosen; for a member, the label carried on the credential).
    #[serde(default)]
    pub label: String,
    pub role: SwarmRole,
    /// OWNER ONLY: the group **secret** key (hex). Present iff `role == Owner`. Never exported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_secret_key: Option<String>,
    /// OWNER ONLY: approved members.
    #[serde(default)]
    pub members: Vec<MemberRecord>,
    /// OWNER ONLY: revoked member public keys (checked by `verify_credential`).
    #[serde(default)]
    pub revoked: BTreeSet<String>,
    /// MEMBER ONLY: our credential from the owner.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential: Option<MembershipCredential>,
    pub created_at: u64,
}

fn default_record_schema() -> u32 {
    SWARM_RECORD_SCHEMA
}

/// A member entry as shown in the UI (never carries a secret).
#[derive(Debug, Clone, Serialize)]
pub struct MemberView {
    pub member_public_key: String,
    pub fingerprint: String,
    pub member_openhydra_peer_id: String,
    pub label: String,
    pub issued_at: u64,
    pub expires_at: u64,
}

/// A redacted swarm view for the desktop — **guaranteed free of the group secret**. `list_swarms`
/// only ever returns these, so the private key physically cannot reach the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct SwarmView {
    pub swarm_public_key: String,
    /// Short human-verifiable fingerprint of the group public key (out-of-band confirmation).
    pub fingerprint: String,
    pub label: String,
    pub role: SwarmRole,
    /// OWNER: approved members.
    pub members: Vec<MemberView>,
    pub member_count: usize,
    pub revoked_count: usize,
    /// MEMBER: when our credential expires (ms), if we hold one.
    pub credential_expires_at: Option<u64>,
    pub created_at: u64,
}

impl SwarmRecord {
    /// Redact to a UI-safe view (drops `group_secret_key` and never reintroduces it).
    pub fn to_view(&self) -> SwarmView {
        SwarmView {
            swarm_public_key: self.swarm_public_key.clone(),
            fingerprint: key_fingerprint(&self.swarm_public_key),
            label: self.label.clone(),
            role: self.role,
            members: self
                .members
                .iter()
                .map(|m| MemberView {
                    member_public_key: m.member_public_key.clone(),
                    fingerprint: key_fingerprint(&m.member_public_key),
                    member_openhydra_peer_id: m.member_openhydra_peer_id.clone(),
                    label: m.label.clone(),
                    issued_at: m.issued_at,
                    expires_at: m.expires_at,
                })
                .collect(),
            member_count: self.members.len(),
            revoked_count: self.revoked.len(),
            credential_expires_at: self.credential.as_ref().map(|c| c.expires_at),
            created_at: self.created_at,
        }
    }
}

// ── persistence: one file per swarm, keyed by group public key ──

/// A 64-hex group public key is a safe filename; reject anything else so a crafted `swarm_public_key`
/// can't escape the swarms directory (path traversal). Also the shape check for a real Ed25519 pubkey.
fn is_valid_swarm_key(pk: &str) -> bool {
    pk.len() == 64 && pk.bytes().all(|b| b.is_ascii_hexdigit())
}

fn swarm_file_path(dir: &Path, swarm_public_key: &str) -> Result<PathBuf, String> {
    if !is_valid_swarm_key(swarm_public_key) {
        return Err(format!("invalid swarm public key: {swarm_public_key:?}"));
    }
    Ok(dir.join(format!("{swarm_public_key}.json")))
}

/// Read one swarm record by public key. `Ok(None)` if the file is absent.
pub fn read_swarm(dir: &Path, swarm_public_key: &str) -> Result<Option<SwarmRecord>, String> {
    let path = swarm_file_path(dir, swarm_public_key)?;
    match std::fs::read_to_string(&path) {
        Ok(s) => serde_json::from_str(&s).map(Some).map_err(|e| format!("parse swarm file: {e}")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("read swarm file: {e}")),
    }
}

/// Atomically write a swarm record (temp + rename). Owner files carry the group secret, so on Unix the
/// temp file is created with `0600` **before any bytes are written** (review #1: the old write-then-
/// chmod left a window where `<pk>.json.tmp` existed world-readable and a local user could race it to
/// read the group key). A stale temp from a crashed prior write is cleared first so `create_new`
/// succeeds.
fn write_swarm(dir: &Path, record: &SwarmRecord) -> Result<(), String> {
    use std::io::Write;
    let path = swarm_file_path(dir, &record.swarm_public_key)?;
    std::fs::create_dir_all(dir).map_err(|e| format!("create swarms dir: {e}"))?;
    let json =
        serde_json::to_string_pretty(record).map_err(|e| format!("encode swarm record: {e}"))?;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let _ = std::fs::remove_file(&tmp); // clear a stale temp so create_new can't collide

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600); // private from creation — the secret is never on disk at wider perms
    }
    let mut f = opts.open(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    f.write_all(json.as_bytes()).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    f.sync_all().map_err(|e| format!("sync {}: {e}", tmp.display()))?;
    drop(f);
    std::fs::rename(&tmp, &path).map_err(|e| format!("rename into {}: {e}", path.display()))
}

/// Every swarm record on disk, as UI-safe views (no secrets). Best-effort: an unreadable/corrupt file
/// is skipped, not fatal.
pub fn list_swarms(dir: &Path) -> Result<Vec<SwarmView>, String> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("read swarms dir: {e}")),
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Ok(s) = std::fs::read_to_string(&path) {
            if let Ok(rec) = serde_json::from_str::<SwarmRecord>(&s) {
                out.push(rec.to_view());
            }
        }
    }
    out.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    Ok(out)
}

// ── owner operations ──

/// Create a new swarm: generate a group keypair, persist an OWNER record (0600), return its view.
pub fn create_swarm(dir: &Path, label: &str, now_ms: u64) -> Result<SwarmView, String> {
    let (public, secret) = generate_group_keypair_hex().map_err(|e| e.to_string())?;
    let record = SwarmRecord {
        schema_version: SWARM_RECORD_SCHEMA,
        swarm_public_key: public,
        label: label.to_string(),
        role: SwarmRole::Owner,
        group_secret_key: Some(secret),
        members: Vec::new(),
        revoked: BTreeSet::new(),
        credential: None,
        created_at: now_ms,
    };
    write_swarm(dir, &record)?;
    Ok(record.to_view())
}

/// Owner approves an enrollment request into a signed credential. Verifies the request (proving the
/// member holds `member_public_key`), signs a credential with the group key valid for `ttl_secs`,
/// records the member, and persists. Returns `(credential, magnet)` — the owner sends either back to
/// the member out-of-band. Refuses if the request names a *different* swarm, or the member is revoked.
pub fn approve_member(
    dir: &Path,
    swarm_public_key: &str,
    request_str: &str,
    member_label: &str,
    ttl_secs: u64,
    now_ms: u64,
) -> Result<ApprovedCredential, String> {
    let mut record = read_swarm(dir, swarm_public_key)?
        .ok_or_else(|| format!("no swarm {swarm_public_key} on this node"))?;
    if record.role != SwarmRole::Owner {
        return Err("only the swarm owner can approve members".to_string());
    }
    let secret = record
        .group_secret_key
        .clone()
        .ok_or_else(|| "owner record is missing its group key".to_string())?;

    let req = parse_enrollment_request(request_str)?;
    let verified = verify_enrollment_request(&req).map_err(|e| format!("bad enrollment request: {e}"))?;
    let member_pk = verified.request.member_public_key.clone();
    // If the request names a swarm, it must be THIS one (defends against approving into the wrong group).
    if !verified.request.swarm_public_key.is_empty()
        && !verified.request.swarm_public_key.eq_ignore_ascii_case(swarm_public_key)
    {
        return Err(format!(
            "enrollment request targets a different swarm ({}) than {swarm_public_key}",
            verified.request.swarm_public_key
        ));
    }
    if record.revoked.contains(&member_pk) {
        return Err("this member is revoked; un-revoke before re-approving".to_string());
    }

    let expires_at = now_ms.saturating_add(ttl_secs.saturating_mul(1000));
    let cred = MembershipCredential::new_unsigned(
        member_pk.clone(),
        verified.request.member_openhydra_peer_id.clone(),
        &record.label,
        now_ms,
        expires_at,
    );
    let signed =
        sign_credential_with_secret_hex(cred, &secret).map_err(|e| format!("sign credential: {e}"))?;
    let magnet = signed.to_magnet().map_err(|e| e.to_string())?;

    // Record the member (replace any prior entry for the same key — a re-approval refreshes it).
    record.members.retain(|m| m.member_public_key != member_pk);
    record.members.push(MemberRecord {
        member_public_key: member_pk,
        member_openhydra_peer_id: verified.request.member_openhydra_peer_id.clone(),
        label: member_label.to_string(),
        issued_at: now_ms,
        expires_at,
    });
    write_swarm(dir, &record)?;
    Ok(ApprovedCredential { credential: signed, magnet })
}

/// Revoke a member: add its key to the revocation set and drop it from the member list. Idempotent.
pub fn revoke_member(dir: &Path, swarm_public_key: &str, member_public_key: &str) -> Result<(), String> {
    let mut record = read_swarm(dir, swarm_public_key)?
        .ok_or_else(|| format!("no swarm {swarm_public_key} on this node"))?;
    if record.role != SwarmRole::Owner {
        return Err("only the swarm owner can revoke members".to_string());
    }
    record.revoked.insert(member_public_key.to_string());
    record.members.retain(|m| m.member_public_key != member_public_key);
    write_swarm(dir, &record)
}

// ── member operations ──

/// Build a signed enrollment request from THIS node's identity, to send to a swarm owner out-of-band.
/// `swarm_public_key` is an optional hint (from a card) of which swarm to join.
pub fn build_enrollment_request(
    identity: &Identity,
    swarm_public_key_hint: &str,
    label: &str,
    now_ms: u64,
) -> Result<EnrollmentRequestExport, String> {
    let req = EnrollmentRequest::new_unsigned(
        identity.openhydra_peer_id.clone(),
        swarm_public_key_hint,
        label,
        now_ms,
    );
    let signed =
        sign_enrollment_request(req, &identity.keypair).map_err(|e| format!("sign request: {e}"))?;
    let magnet = signed.to_magnet().map_err(|e| e.to_string())?;
    Ok(EnrollmentRequestExport { request: signed, magnet })
}

/// Load this node's identity from `identity_path` and build a signed enrollment request — the
/// desktop entry point (keeps identity loading in the agent, like `run_export`).
pub fn enroll_request_at(
    identity_path: &Path,
    swarm_public_key_hint: &str,
    label: &str,
    now_ms: u64,
) -> Result<EnrollmentRequestExport, String> {
    let id = Identity::load_or_create(identity_path).map_err(|e| format!("load identity: {e}"))?;
    build_enrollment_request(&id, swarm_public_key_hint, label, now_ms)
}

/// Load this node's identity from `identity_path` and accept a credential into `dir` — the desktop
/// entry point for the member accept flow.
pub fn accept_credential_at(
    dir: &Path,
    identity_path: &Path,
    credential_str: &str,
    label: &str,
    now_ms: u64,
) -> Result<SwarmView, String> {
    let id = Identity::load_or_create(identity_path).map_err(|e| format!("load identity: {e}"))?;
    accept_credential(dir, &id, credential_str, label, now_ms)
}

/// Accept a credential the owner returned: verify it is (a) validly signed by its group key, (b)
/// bound to OUR identity, and (c) unexpired, then persist a MEMBER record. Binding to our own key
/// means we refuse a credential minted for someone else. Returns the stored view.
pub fn accept_credential(
    dir: &Path,
    identity: &Identity,
    credential_str: &str,
    label: &str,
    now_ms: u64,
) -> Result<SwarmView, String> {
    let cred = parse_credential(credential_str)?;
    let our_pk = keypair_public_hex(&identity.keypair).map_err(|e| e.to_string())?;
    // Bind to our identity + verify signature/expiry (no revocation set on the member side — the
    // owner enforces revocation at serve time; an accepted credential can still be centrally revoked).
    verify_credential_for_member(&cred, &our_pk, now_ms, &BTreeSet::new())
        .map_err(|e| format!("credential not valid for this node: {e}"))?;

    let record = SwarmRecord {
        schema_version: SWARM_RECORD_SCHEMA,
        swarm_public_key: cred.swarm_public_key.clone(),
        label: if label.is_empty() { cred.swarm_label.clone() } else { label.to_string() },
        role: SwarmRole::Member,
        group_secret_key: None,
        members: Vec::new(),
        revoked: BTreeSet::new(),
        credential: Some(cred),
        created_at: now_ms,
    };
    write_swarm(dir, &record)?;
    Ok(record.to_view())
}

/// Forget a swarm entirely (owner or member): delete its file. For an owner this destroys the group
/// key, so no further credentials can be issued.
pub fn forget_swarm(dir: &Path, swarm_public_key: &str) -> Result<(), String> {
    let path = swarm_file_path(dir, swarm_public_key)?;
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("remove swarm file: {e}")),
    }
}

// ── export envelopes + parse helpers ──

/// `{ credential, magnet }` returned by [`approve_member`] — the owner sends one to the member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovedCredential {
    pub credential: MembershipCredential,
    pub magnet: String,
}

/// `{ request, magnet }` returned by [`build_enrollment_request`] — the member sends one to the owner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrollmentRequestExport {
    pub request: EnrollmentRequest,
    pub magnet: String,
}

/// A previewed credential (verified signature only — expiry/binding reported to the UI, not enforced
/// here) so the member sees who it's from before accepting.
pub fn preview_credential(
    credential_str: &str,
    now_ms: u64,
) -> Result<MembershipCredential, String> {
    let cred = parse_credential(credential_str)?;
    // Full verify (signature + schema + expiry); no revocation set (member side).
    verify_credential(&cred, now_ms, &BTreeSet::new())
        .map(|v| v.credential)
        .map_err(|e| e.to_string())
}

/// A previewed enrollment request (verified member signature) so the owner sees who's asking + the
/// fingerprint to confirm out-of-band before approving.
pub fn preview_enrollment_request(request_str: &str) -> Result<EnrollmentRequest, String> {
    let req = parse_enrollment_request(request_str)?;
    verify_enrollment_request(&req).map(|v| v.request).map_err(|e| e.to_string())
}

/// Cap on a pasted enrollment request / credential (review #5: parity with the M2 card file cap). A
/// real artifact is a few hundred bytes; this bounds the work a hostile self-signed paste can force
/// on preview/approve before the per-field bounds in the crypto core apply.
const MAX_ARTIFACT_BYTES: usize = 64 * 1024;

fn parse_enrollment_request(s: &str) -> Result<EnrollmentRequest, String> {
    let t = s.trim();
    if t.len() > MAX_ARTIFACT_BYTES {
        return Err(format!("enrollment request too large (> {MAX_ARTIFACT_BYTES} bytes)"));
    }
    if t.starts_with("openhydra:enroll:") {
        EnrollmentRequest::from_magnet(t).map_err(|e| e.to_string())
    } else {
        EnrollmentRequest::from_json(t).map_err(|e| e.to_string())
    }
}

fn parse_credential(s: &str) -> Result<MembershipCredential, String> {
    let t = s.trim();
    if t.len() > MAX_ARTIFACT_BYTES {
        return Err(format!("credential too large (> {MAX_ARTIFACT_BYTES} bytes)"));
    }
    if t.starts_with("openhydra:cred:") {
        MembershipCredential::from_magnet(t).map_err(|e| e.to_string())
    } else {
        MembershipCredential::from_json(t).map_err(|e| e.to_string())
    }
}

/// The membership schema version this build writes (surfaced for diagnostics/UI).
pub fn schema_version() -> u32 {
    MEMBERSHIP_SCHEMA_VERSION
}

/// A cheap freshness signature of the swarms directory: for each `*.json`, its name, byte length, and
/// mtime, sorted. Used to invalidate the hot-reload caches. More robust than a bare directory mtime
/// (M4 review MED): it survives a filesystem where `modified()` returns `None` (per-file length still
/// changes on a rewrite) and a coarse-granularity mtime clock (a `revoke_member` grows the revoked
/// set, so the owner file's byte length changes even within the same second). Recomputed per call —
/// a handful of `stat`s on a tiny directory; the expensive JSON parse still runs only on a change.
type DirSig = Vec<(String, u64, Option<SystemTime>)>;

fn swarms_dir_signature(dir: &Path) -> DirSig {
    let mut sig = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            if let (Some(name), Ok(meta)) =
                (p.file_name().and_then(|n| n.to_str()).map(String::from), e.metadata())
            {
                sig.push((name, meta.len(), meta.modified().ok()));
            }
        }
    }
    sig.sort();
    sig
}

// ── M4-base: consumer-side credential store ──
//
// The member/consumer counterpart to [`SwarmAuthorizer`]: a hot-reloaded set of the credentials THIS
// node holds (from member swarm records), so the consumer can attach one to a serve request to reach
// a provider's private model. Selection is intentionally simple for v1 (the first still-valid
// credential) — the common case is a single swarm; per-provider credential matching for multi-swarm
// membership is a follow-up.

#[derive(Default)]
struct CredCache {
    sig: Option<DirSig>,
    creds: Vec<MembershipCredential>,
}

/// Hot-reloaded set of the membership credentials this node holds (member records). Rescans the
/// swarms directory only when its mtime changes.
pub struct CredentialStore {
    dir: PathBuf,
    cache: Mutex<CredCache>,
}

impl CredentialStore {
    pub fn new(dir: PathBuf) -> Arc<Self> {
        Arc::new(Self { dir, cache: Mutex::new(CredCache::default()) })
    }

    fn refresh(&self) {
        let sig = swarms_dir_signature(&self.dir);
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if c.sig.as_ref() == Some(&sig) {
            return;
        }
        let mut creds = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                if let Ok(s) = std::fs::read_to_string(&path) {
                    if let Ok(rec) = serde_json::from_str::<SwarmRecord>(&s) {
                        if let Some(cred) = rec.credential {
                            creds.push(cred);
                        }
                    }
                }
            }
        }
        c.creds = creds;
        c.sig = Some(sig);
    }

    /// The best membership credential to present right now — the first that hasn't expired. `None`
    /// when this node holds no live credential (the common public-only case).
    pub fn credential_for(&self, now_ms: u64) -> Option<MembershipCredential> {
        self.refresh();
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.creds.iter().find(|cr| now_ms < cr.expires_at).cloned()
    }

    /// The live credential for a SPECIFIC swarm (hex group public key) — used when a private card
    /// names the swarm its provider gates on, so a multi-swarm member presents the RIGHT credential.
    /// `None` if we hold no live credential for that swarm.
    pub fn credential_for_swarm(&self, swarm_public_key: &str, now_ms: u64) -> Option<MembershipCredential> {
        self.refresh();
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.creds
            .iter()
            .find(|cr| now_ms < cr.expires_at && cr.swarm_public_key == swarm_public_key)
            .cloned()
    }
}

// ── M4-base: owner-side serve-gate authorizer ──
//
// The provider consults this before serving a `Private`-scope model. It authorizes a credential ONLY
// against swarms THIS node OWNS — the owner holds the revocation set, so revocation is enforced where
// it's authoritative (M3 review guardrail: a non-owner verifying membership runs against an empty
// revoked set and would admit a revoked member). Trust binds to `member_public_key` → its libp2p peer
// id, checked against the live connection's authenticated peer id (never the self-asserted
// `member_openhydra_peer_id`, guardrail #4).

/// Why a serve request was refused at the swarm gate. Distinct variants so the provider can surface a
/// precise error and bump the right counter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthzError {
    /// A `Private` model was requested but the request carried no membership credential.
    NoCredential,
    /// The credential is for a swarm this provider does not own (so it can't vouch for or revoke it).
    UnknownSwarm(String),
    /// The credential failed verification against the owned swarm (bad signature, revoked, expired).
    NotAuthorized(MembershipError),
    /// The credential is valid but authorises a different identity than the connecting peer.
    WrongPeer { credential_peer: String, connection_peer: String },
}

impl std::fmt::Display for AuthzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthzError::NoCredential => {
                write!(f, "this model is private — a swarm membership credential is required")
            }
            AuthzError::UnknownSwarm(pk) => {
                write!(f, "not authorised: credential is for a swarm this provider does not own ({pk})")
            }
            AuthzError::NotAuthorized(e) => write!(f, "not authorised for this swarm: {e}"),
            AuthzError::WrongPeer { credential_peer, connection_peer } => write!(
                f,
                "credential authorises {credential_peer} but the request came from {connection_peer}"
            ),
        }
    }
}
impl std::error::Error for AuthzError {}

/// A successful gate pass — the member + swarm that authorised this serve (for logging + the
/// self-serve-like credit flag).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthorizedServe {
    pub swarm_public_key: String,
    pub member_public_key: String,
}

struct OwnedSwarm {
    revoked: BTreeSet<String>,
}

#[derive(Default)]
struct AuthCache {
    /// The directory signature last scanned (see [`swarms_dir_signature`]). Reloaded whenever it
    /// changes — robust for a security-critical set (a revoked member must stop being served
    /// promptly) where a bare directory mtime could go stale.
    sig: Option<DirSig>,
    /// Owned swarms indexed by group public key. Only OWNER records (which carry the revoked set).
    owned: BTreeMap<String, OwnedSwarm>,
}

/// Hot-reloaded index of the swarms THIS node owns, used to gate `Private` serves. Rescans the swarms
/// directory only when its mtime changes, so the steady-state serve path is a cheap `stat`.
pub struct SwarmAuthorizer {
    dir: PathBuf,
    cache: Mutex<AuthCache>,
}

impl SwarmAuthorizer {
    pub fn new(dir: PathBuf) -> Arc<Self> {
        Arc::new(Self { dir, cache: Mutex::new(AuthCache::default()) })
    }

    /// Reload the owned-swarm index iff the directory changed (or was never loaded). Best-effort: an
    /// unreadable dir/file is skipped, never errors the serve path. Poison-tolerant.
    fn refresh(&self) {
        let sig = swarms_dir_signature(&self.dir);
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if c.sig.as_ref() == Some(&sig) {
            return;
        }
        let mut owned = BTreeMap::new();
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                if let Ok(s) = std::fs::read_to_string(&path) {
                    if let Ok(rec) = serde_json::from_str::<SwarmRecord>(&s) {
                        if rec.role == SwarmRole::Owner {
                            owned.insert(
                                rec.swarm_public_key.clone(),
                                OwnedSwarm { revoked: rec.revoked.clone() },
                            );
                        }
                    }
                }
            }
        }
        c.owned = owned;
        c.sig = Some(sig);
    }

    /// Authorize a serve of a `Private` model. `credential` is what the request carried (if any),
    /// `connection_peer_id` is the libp2p-authenticated sender. Passes only when the credential is for
    /// a swarm we own, verifies (signature + not-revoked + unexpired) against that swarm's revocation
    /// set, and its bound member key derives exactly `connection_peer_id`.
    pub fn authorize(
        &self,
        credential: Option<&MembershipCredential>,
        connection_peer_id: &str,
        now_ms: u64,
    ) -> Result<AuthorizedServe, AuthzError> {
        let cred = credential.ok_or(AuthzError::NoCredential)?;
        self.refresh();
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let owned = c
            .owned
            .get(&cred.swarm_public_key)
            .ok_or_else(|| AuthzError::UnknownSwarm(cred.swarm_public_key.clone()))?;
        // Verify against THIS swarm's revocation set (owner-authoritative).
        verify_credential(cred, now_ms, &owned.revoked).map_err(AuthzError::NotAuthorized)?;
        // Bind to the live connection: the credential's member key must derive the sender's peer id.
        let member_peer = credential_member_peer_id(cred)
            .map_err(AuthzError::NotAuthorized)?
            .to_string();
        if member_peer != connection_peer_id {
            return Err(AuthzError::WrongPeer {
                credential_peer: member_peer,
                connection_peer: connection_peer_id.to_string(),
            });
        }
        Ok(AuthorizedServe {
            swarm_public_key: cred.swarm_public_key.clone(),
            member_public_key: cred.member_public_key.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn an_identity() -> Identity {
        let dir = tempfile::tempdir().unwrap();
        Identity::load_or_create(&dir.path().join("id.key")).unwrap()
    }

    fn our_pk(id: &Identity) -> String {
        keypair_public_hex(&id.keypair).unwrap()
    }

    const HOUR: u64 = 3600;

    #[test]
    fn create_swarm_persists_owner_record_and_view_hides_secret() {
        let dir = tempfile::tempdir().unwrap();
        let view = create_swarm(dir.path(), "Home rig", 1_000).unwrap();
        assert_eq!(view.role, SwarmRole::Owner);
        assert!(is_valid_swarm_key(&view.swarm_public_key));
        assert_eq!(view.fingerprint.len(), 19);
        // The on-disk record has the secret; the file is 0600.
        let rec = read_swarm(dir.path(), &view.swarm_public_key).unwrap().unwrap();
        assert!(rec.group_secret_key.is_some());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let path = swarm_file_path(dir.path(), &view.swarm_public_key).unwrap();
            let mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "owner swarm file must be private");
        }
        // list_swarms returns the redacted view — serialize it and assert no secret leaks.
        let views = list_swarms(dir.path()).unwrap();
        let json = serde_json::to_string(&views).unwrap();
        assert!(!json.contains(rec.group_secret_key.as_ref().unwrap()), "secret leaked into view");
    }

    #[test]
    fn full_enroll_approve_accept_round_trip() {
        let owner_dir = tempfile::tempdir().unwrap();
        let member_dir = tempfile::tempdir().unwrap();
        let member = an_identity();

        // Owner creates a swarm.
        let swarm = create_swarm(owner_dir.path(), "Home rig", 1_000).unwrap();
        // Member builds an enrollment request naming that swarm.
        let req = build_enrollment_request(&member, &swarm.swarm_public_key, "Sam's MacBook", 2_000).unwrap();
        // Owner approves it → credential.
        let approved = approve_member(
            owner_dir.path(),
            &swarm.swarm_public_key,
            &req.magnet,
            "Sam's MacBook",
            24 * HOUR,
            3_000,
        )
        .unwrap();
        assert_eq!(approved.credential.member_public_key, our_pk(&member));
        // Owner now lists one member.
        let owner_view = &list_swarms(owner_dir.path()).unwrap()[0];
        assert_eq!(owner_view.member_count, 1);
        assert_eq!(owner_view.members[0].label, "Sam's MacBook");

        // Member accepts the credential (bound to its own key) → member record.
        let mview = accept_credential(member_dir.path(), &member, &approved.magnet, "", 4_000).unwrap();
        assert_eq!(mview.role, SwarmRole::Member);
        assert_eq!(mview.swarm_public_key, swarm.swarm_public_key);
        assert_eq!(mview.label, "Home rig", "falls back to the swarm label on the credential");
        assert!(mview.credential_expires_at.unwrap() > 4_000);
    }

    #[test]
    fn a_credential_minted_for_someone_else_is_refused_on_accept() {
        let owner_dir = tempfile::tempdir().unwrap();
        let attacker_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let attacker = an_identity();
        let swarm = create_swarm(owner_dir.path(), "Home rig", 1_000).unwrap();
        let req = build_enrollment_request(&member, &swarm.swarm_public_key, "member", 2_000).unwrap();
        let approved =
            approve_member(owner_dir.path(), &swarm.swarm_public_key, &req.magnet, "member", HOUR, 3_000)
                .unwrap();
        // The attacker tries to accept the member's credential as their own → member-mismatch refusal.
        let err = accept_credential(attacker_dir.path(), &attacker, &approved.magnet, "", 4_000).unwrap_err();
        assert!(err.contains("not valid for this node"), "got: {err}");
    }

    #[test]
    fn approving_into_the_wrong_swarm_is_refused() {
        let owner_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let swarm_a = create_swarm(owner_dir.path(), "A", 1_000).unwrap();
        let _swarm_b = create_swarm(owner_dir.path(), "B", 1_000).unwrap();
        // Request explicitly names swarm A.
        let req = build_enrollment_request(&member, &swarm_a.swarm_public_key, "m", 2_000).unwrap();
        // But we try to approve it under a different existing swarm's key.
        let other_key = list_swarms(owner_dir.path())
            .unwrap()
            .into_iter()
            .find(|v| v.swarm_public_key != swarm_a.swarm_public_key)
            .unwrap()
            .swarm_public_key;
        let err =
            approve_member(owner_dir.path(), &other_key, &req.magnet, "m", HOUR, 3_000).unwrap_err();
        assert!(err.contains("different swarm"), "got: {err}");
    }

    #[test]
    fn revoked_member_credential_fails_owner_side_verification() {
        let owner_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let swarm = create_swarm(owner_dir.path(), "Home rig", 1_000).unwrap();
        let req = build_enrollment_request(&member, &swarm.swarm_public_key, "m", 2_000).unwrap();
        let approved =
            approve_member(owner_dir.path(), &swarm.swarm_public_key, &req.magnet, "m", HOUR, 3_000).unwrap();
        // Revoke.
        revoke_member(owner_dir.path(), &swarm.swarm_public_key, &our_pk(&member)).unwrap();
        let rec = read_swarm(owner_dir.path(), &swarm.swarm_public_key).unwrap().unwrap();
        assert!(rec.members.is_empty(), "revoked member dropped from the list");
        // The credential no longer verifies against the owner's revocation set.
        let err = verify_credential(&approved.credential, 3_500, &rec.revoked).unwrap_err();
        assert!(matches!(err, openhydra_network::membership::MembershipError::Revoked(_)));
    }

    #[test]
    fn re_approval_refreshes_rather_than_duplicates_a_member() {
        let owner_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let swarm = create_swarm(owner_dir.path(), "Home rig", 1_000).unwrap();
        let req = build_enrollment_request(&member, &swarm.swarm_public_key, "m", 2_000).unwrap();
        approve_member(owner_dir.path(), &swarm.swarm_public_key, &req.magnet, "first", HOUR, 3_000).unwrap();
        approve_member(owner_dir.path(), &swarm.swarm_public_key, &req.magnet, "second", HOUR, 4_000).unwrap();
        let view = &list_swarms(owner_dir.path()).unwrap()[0];
        assert_eq!(view.member_count, 1, "same key re-approved → one entry");
        assert_eq!(view.members[0].label, "second", "label refreshed");
    }

    #[test]
    fn a_member_cannot_approve_or_revoke() {
        let member_dir = tempfile::tempdir().unwrap();
        let owner_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let swarm = create_swarm(owner_dir.path(), "Home rig", 1_000).unwrap();
        let req = build_enrollment_request(&member, &swarm.swarm_public_key, "m", 2_000).unwrap();
        let approved =
            approve_member(owner_dir.path(), &swarm.swarm_public_key, &req.magnet, "m", HOUR, 3_000).unwrap();
        accept_credential(member_dir.path(), &member, &approved.magnet, "", 4_000).unwrap();
        // The member holds a MEMBER record for this swarm; approving/revoking on it must be refused.
        let err = revoke_member(member_dir.path(), &swarm.swarm_public_key, "deadbeef").unwrap_err();
        assert!(err.contains("owner"), "got: {err}");
    }

    #[test]
    fn forget_swarm_removes_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let swarm = create_swarm(dir.path(), "Home rig", 1_000).unwrap();
        assert_eq!(list_swarms(dir.path()).unwrap().len(), 1);
        forget_swarm(dir.path(), &swarm.swarm_public_key).unwrap();
        assert!(list_swarms(dir.path()).unwrap().is_empty());
        forget_swarm(dir.path(), &swarm.swarm_public_key).unwrap(); // idempotent
    }

    #[test]
    fn an_invalid_swarm_key_cannot_escape_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        // Path-traversal / non-hex keys are rejected before touching the filesystem.
        assert!(read_swarm(dir.path(), "../../etc/passwd").is_err());
        assert!(read_swarm(dir.path(), "not-hex-key").is_err());
        assert!(swarm_file_path(dir.path(), &"a".repeat(64)).is_ok());
    }

    #[test]
    fn preview_helpers_verify_without_persisting() {
        let owner_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let swarm = create_swarm(owner_dir.path(), "Home rig", 1_000).unwrap();
        let req = build_enrollment_request(&member, &swarm.swarm_public_key, "Sam", 2_000).unwrap();
        // Owner previews the request → sees the member fingerprint, nothing persisted.
        let pr = preview_enrollment_request(&req.magnet).unwrap();
        assert_eq!(pr.member_public_key, our_pk(&member));
        let approved =
            approve_member(owner_dir.path(), &swarm.swarm_public_key, &req.magnet, "Sam", HOUR, 3_000).unwrap();
        // Member previews the credential before accepting.
        let pc = preview_credential(&approved.magnet, 3_500).unwrap();
        assert_eq!(pc.swarm_public_key, swarm.swarm_public_key);
    }

    // ── M4-base: the owner-side serve-gate authorizer ──

    /// Owner creates a swarm + approves `member`; returns (owner_dir kept alive by caller, swarm pk,
    /// the signed credential, the member's libp2p peer-id string).
    fn approved_member(
        owner_dir: &Path,
        member: &Identity,
        ttl_secs: u64,
    ) -> (String, MembershipCredential, String) {
        let swarm = create_swarm(owner_dir, "Home rig", 1_000).unwrap();
        let req = build_enrollment_request(member, &swarm.swarm_public_key, "m", 2_000).unwrap();
        let approved =
            approve_member(owner_dir, &swarm.swarm_public_key, &req.magnet, "m", ttl_secs, 3_000).unwrap();
        let member_peer =
            credential_member_peer_id(&approved.credential).unwrap().to_string();
        (swarm.swarm_public_key, approved.credential, member_peer)
    }

    #[test]
    fn authorizer_admits_a_valid_credential_bound_to_the_connecting_peer() {
        let owner_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let (_swarm, cred, member_peer) = approved_member(owner_dir.path(), &member, 24 * HOUR);
        let auth = SwarmAuthorizer::new(owner_dir.path().to_path_buf());
        let ok = auth.authorize(Some(&cred), &member_peer, 4_000).unwrap();
        assert_eq!(ok.member_public_key, our_pk(&member));
    }

    #[test]
    fn authorizer_refuses_a_private_serve_with_no_credential() {
        let owner_dir = tempfile::tempdir().unwrap();
        let auth = SwarmAuthorizer::new(owner_dir.path().to_path_buf());
        assert_eq!(auth.authorize(None, "12D3KooWanyone", 4_000).unwrap_err(), AuthzError::NoCredential);
    }

    #[test]
    fn authorizer_refuses_a_credential_for_a_swarm_we_do_not_own() {
        // A credential issued by a DIFFERENT owner (a swarm this provider doesn't own) is refused —
        // the provider can't vouch for or revoke a group whose key it doesn't hold (guardrail #3).
        let other_owner_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let (_swarm, cred, member_peer) = approved_member(other_owner_dir.path(), &member, 24 * HOUR);
        // A provider with an EMPTY swarms dir (owns nothing).
        let our_dir = tempfile::tempdir().unwrap();
        let auth = SwarmAuthorizer::new(our_dir.path().to_path_buf());
        assert!(matches!(
            auth.authorize(Some(&cred), &member_peer, 4_000).unwrap_err(),
            AuthzError::UnknownSwarm(_)
        ));
    }

    #[test]
    fn authorizer_refuses_a_credential_bound_to_a_different_peer() {
        // A valid credential presented over a connection from a DIFFERENT peer id than it authorises
        // (a stolen/replayed credential) is refused — trust binds to member_public_key -> peer id.
        let owner_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let (_swarm, cred, _member_peer) = approved_member(owner_dir.path(), &member, 24 * HOUR);
        let auth = SwarmAuthorizer::new(owner_dir.path().to_path_buf());
        let err = auth.authorize(Some(&cred), "12D3KooWimposter", 4_000).unwrap_err();
        assert!(matches!(err, AuthzError::WrongPeer { .. }));
    }

    #[test]
    fn authorizer_refuses_an_expired_credential() {
        let owner_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        // TTL 1s → issued at 3_000, expires at 4_000.
        let (_swarm, cred, member_peer) = approved_member(owner_dir.path(), &member, 1);
        let auth = SwarmAuthorizer::new(owner_dir.path().to_path_buf());
        assert!(auth.authorize(Some(&cred), &member_peer, 3_999).is_ok(), "still valid before expiry");
        let err = auth.authorize(Some(&cred), &member_peer, 4_000).unwrap_err();
        assert!(matches!(err, AuthzError::NotAuthorized(_)));
    }

    #[test]
    fn authorizer_hot_reloads_a_revocation() {
        // A member admitted now must be refused after the owner revokes them — the authorizer picks
        // up the rewritten swarm file (dir mtime changes on the temp+rename).
        let owner_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let (swarm, cred, member_peer) = approved_member(owner_dir.path(), &member, 24 * HOUR);
        let auth = SwarmAuthorizer::new(owner_dir.path().to_path_buf());
        assert!(auth.authorize(Some(&cred), &member_peer, 4_000).is_ok(), "admitted before revoke");
        revoke_member(owner_dir.path(), &swarm, &our_pk(&member)).unwrap();
        let err = auth.authorize(Some(&cred), &member_peer, 4_000).unwrap_err();
        assert!(
            matches!(err, AuthzError::NotAuthorized(MembershipError::Revoked(_))),
            "revoked member must be refused after hot-reload, got: {err:?}"
        );
    }

    #[test]
    fn credential_store_returns_a_held_unexpired_credential() {
        // A member node's CredentialStore surfaces the credential it holds (from its member record),
        // and stops offering it once expired.
        let owner_dir = tempfile::tempdir().unwrap();
        let member_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        // TTL 1s → issued 3_000, expires 4_000.
        let (swarm, cred, _peer) = approved_member(owner_dir.path(), &member, 1);
        let member_record = SwarmRecord {
            schema_version: SWARM_RECORD_SCHEMA,
            swarm_public_key: swarm,
            label: "Home rig".into(),
            role: SwarmRole::Member,
            group_secret_key: None,
            members: Vec::new(),
            revoked: BTreeSet::new(),
            credential: Some(cred),
            created_at: 4_000,
        };
        write_swarm(member_dir.path(), &member_record).unwrap();
        let store = CredentialStore::new(member_dir.path().to_path_buf());
        assert!(store.credential_for(3_500).is_some(), "held credential offered before expiry");
        assert!(store.credential_for(4_000).is_none(), "expired credential not offered");
        // An empty dir yields nothing.
        let empty = tempfile::tempdir().unwrap();
        assert!(CredentialStore::new(empty.path().to_path_buf()).credential_for(3_500).is_none());
    }

    #[test]
    fn authorizer_ignores_member_records_only_owner_swarms_authorize() {
        // A node that is only a MEMBER of a swarm (holds a credential, not the group key) must NOT
        // authorize serves for it — it lacks the revocation set (guardrail #3). Persist a genuine
        // MEMBER record for the swarm, then confirm the authorizer still reports UnknownSwarm (it only
        // indexes owner records).
        let owner_dir = tempfile::tempdir().unwrap();
        let member_dir = tempfile::tempdir().unwrap();
        let member = an_identity();
        let (swarm, cred, member_peer) = approved_member(owner_dir.path(), &member, 24 * HOUR);
        let member_record = SwarmRecord {
            schema_version: SWARM_RECORD_SCHEMA,
            swarm_public_key: swarm.clone(),
            label: "Home rig".into(),
            role: SwarmRole::Member,
            group_secret_key: None,
            members: Vec::new(),
            revoked: BTreeSet::new(),
            credential: Some(cred.clone()),
            created_at: 4_000,
        };
        write_swarm(member_dir.path(), &member_record).unwrap();
        // The record exists and is for this swarm, but role == Member → the authorizer skips it.
        let auth = SwarmAuthorizer::new(member_dir.path().to_path_buf());
        assert!(matches!(
            auth.authorize(Some(&cred), &member_peer, 4_000).unwrap_err(),
            AuthzError::UnknownSwarm(_)
        ));
    }
}
