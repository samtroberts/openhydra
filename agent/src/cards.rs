//! Agent-side `.openhydra` card glue (M2): assemble a [`Card`] from a locally-detected model + this
//! node's identity, gate it on the share policy, and sign it. The signed [`Card`] type + crypto live
//! in [`openhydra_network::card`]; this module is the thin bridge to the agent's detection, identity,
//! and share-policy so the desktop (which shells out to `openhydra-agent card export`) gets a ready
//! artifact. Import (verify + registry injection) is separate — verify is pure in the network crate.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use openhydra_network::card::{sign_card, verify_card, Capability, Card, PricingMode};
use openhydra_network::types::DiscoveredPeer;

use crate::adapter::DetectedModel;

/// The export envelope printed by `openhydra-agent card export`: the signed card plus its compact
/// copy-paste magnet string. JSON on stdout; the desktop parses it back into this.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CardExport {
    pub card: Card,
    pub magnet: String,
}

/// Build an UNSIGNED card from a detected model + this node's OpenHydra id. Pure + unit-testable.
/// `public_id` is the name the provider announces and serves the model under
/// ([`crate::provider::PublicNames`]), so a raw engine path never reaches `model_id`
/// (belt-and-suspenders with the privacy gate in `sign_card`). Capability context/output are left 0
/// (not known at detection — hint fields).
#[allow(clippy::too_many_arguments)]
pub fn build_card(
    model: &DetectedModel,
    public_id: &str,
    openhydra_peer_id: &str,
    signed_at_ms: u64,
    ttl_secs: u64,
    pricing: PricingMode,
    region: Option<String>,
    swarm_public_key: Option<String>,
) -> Card {
    let expires_at = signed_at_ms.saturating_add(ttl_secs.saturating_mul(1000));
    let mut card = Card::new_unsigned(
        openhydra_peer_id,
        public_id,
        signed_at_ms,
        expires_at,
    );
    card.canonical_id = model.canonical_id.clone();
    // Disclose the engine's possession digest on the card's (already signed) weight_hash, when the
    // engine reports one (UNIFIED_TRUST_STACK 2c). Empty for engines that expose no digest — the
    // field stays "" and the card is byte-identical to before. **Disclosed, not verified**: no
    // expected-digest registry exists to check it against yet, so a verifier must not treat a
    // present weight_hash as a possession proof (it never raises the authenticity tier on its own).
    card.weight_hash = model.weight_digest.clone();
    card.capability = Capability {
        params: model.params.clone(),
        context_length: 0,
        max_output_tokens: 0,
        modalities: vec!["text".to_string()],
    };
    card.pricing_mode = pricing;
    card.region = region;
    // M4: a swarm binding makes this a private card (the model is served only to a member of this
    // swarm). `sign_card` promotes it to schema 2.
    if let Some(pk) = swarm_public_key.filter(|s| !s.is_empty()) {
        card.swarm_public_key = pk;
    }
    card
}

/// A model bound to some swarms is served only to them, so a card for any other swarm would point
/// its members at a refusal.
fn card_swarm_allowed(
    policy: &crate::share_policy::SharePolicy,
    engine_ref: &str,
    model: &str,
    swarm_public_key: &str,
) -> Result<(), String> {
    match policy.swarms_of(engine_ref) {
        Some(bound) if !bound.contains(&swarm_public_key.to_ascii_lowercase()) => Err(format!(
            "model {model:?} is not shared with swarm {swarm_public_key} — add that swarm to the model's \
             sharing first, or export a card for one of the swarms it is shared with"
        )),
        _ => Ok(()),
    }
}

/// Run the `card export` subcommand: load the identity, detect `model` locally, run the reach gate,
/// then build + sign the card. Two modes: a PUBLIC card (no `--swarm`) requires the model to be
/// Global + consented; a PRIVATE card (`--swarm <pk>`, M4) requires the model to be shared with reach
/// Private and the node to OWN that swarm (`--swarms-dir`), so the card names the swarm its provider
/// gates on. `now_ms` is injected (no hidden clock).
#[allow(clippy::too_many_arguments)]
pub fn run_export(
    model: &str,
    share_policy_file: Option<PathBuf>,
    ttl_secs: u64,
    pricing: PricingMode,
    region: Option<String>,
    identity_path: &Path,
    now_ms: u64,
    swarm_public_key: Option<String>,
    swarms_dir: Option<PathBuf>,
    // Models of an engine detection can't see (the built-in engine runs inside the provider's own
    // process), described by the caller; detected as if an engine had reported them.
    built_in: Vec<DetectedModel>,
) -> Result<CardExport, String> {
    // Export gate, part 1 — fail-closed without a policy file (checked FIRST so it fails fast and is
    // testable without a live engine). Review fix (MODERATE): previously `None` fell back to
    // `share_all()`, so `card export --model X` with the flag omitted would export a model the user
    // had marked private. A card carries reach, so refuse unless the policy confirms the model's reach.
    // Name the model as detection and the share policy do: a llama.cpp `.gguf` path becomes its file
    // stem, so `--model /models/x.gguf` (the engine's own id) finds the detected `x`.
    let model = crate::adapter::normalize_engine_ref(model);
    let model = model.as_str();
    let policy_file = share_policy_file.ok_or_else(|| {
        format!(
            "refusing to export {model:?} without --share-policy-file — pass the share-policy file to \
             run the reach/consent gate"
        )
    })?;
    let policy = crate::share_policy::SharePolicy::load(&policy_file)
        .map_err(|e| format!("read share policy {}: {e}", policy_file.display()))?;

    let id = openhydra_network::identity::Identity::load_or_create(identity_path)
        .map_err(|e| format!("load identity: {e}"))?;

    // Detect the model on a local engine (fast localhost probes; no swarm). Models of the built-in
    // engine, when the caller describes some, are the only ones: the provider running that engine
    // serves only them (and names them among themselves), and a same-named model on another engine
    // running here must not lend the card its weights.
    let engines = if built_in.is_empty() { crate::detect::detect_engines() } else { Vec::new() };
    let all: Vec<&DetectedModel> = engines.iter().flat_map(|e| &e.models).chain(built_in.iter()).collect();
    // The name the provider announces each model under: assigned over every detected model, as its
    // announce pass does. `--model` may be either name (a ComfyUI checkpoint's file name, or the stem
    // it is announced and carded under).
    let names = crate::provider::PublicNames::from_engine_refs(all.iter().map(|m| m.engine_ref.as_str()));
    let engine_ref = names.engine_ref(model).to_string();
    let detected = all
        .iter()
        .copied()
        .find(|m| m.engine_ref == model || m.engine_ref == engine_ref)
        .ok_or_else(|| {
            format!("model {model:?} is not detected on any local engine — start the engine that serves it, then retry")
        })?;

    // Export gate, part 2 — reach-dependent:
    let swarm_binding = match swarm_public_key.filter(|s| !s.is_empty()) {
        // M4 PRIVATE card: bind to a swarm. The model must be shared + Private (not a public model),
        // and we must OWN that swarm — the provider's serve gate only authorizes credentials for
        // swarms it owns, so a card naming a swarm we don't own would be unusable. Verified against
        // the swarms dir.
        Some(pk) => {
            if !policy.is_shared(&detected.engine_ref) {
                return Err(format!("model {model:?} is not shared, so it can't be exported"));
            }
            if policy.scope_of(&detected.engine_ref) != crate::share_policy::Scope::Private {
                return Err(format!(
                    "model {model:?} is not Private — a swarm-bound card is for a private model; set \
                     its reach to Private first (or export a public card without --swarm)"
                ));
            }
            let dir = swarms_dir.ok_or_else(|| {
                "a swarm-bound card needs --swarms-dir to confirm you own the swarm".to_string()
            })?;
            match crate::swarms::read_swarm(&dir, &pk)? {
                Some(rec) if rec.role == crate::swarms::SwarmRole::Owner => {}
                _ => {
                    return Err(format!(
                        "you don't own swarm {pk} — only a swarm's owner can issue a card for it"
                    ))
                }
            }
            card_swarm_allowed(&policy, &detected.engine_ref, model, &pk)?;
            Some(pk)
        }
        // PUBLIC card: only a globally-shared, consented model may become one.
        None => {
            if !policy.announce_globally(&detected.engine_ref) {
                return Err(format!(
                    "model {model:?} is not shared to the global network (or its global-publish consent \
                     is missing) — set the model's reach to Global, or pass --swarm to export a private card"
                ));
            }
            None
        }
    };

    let public_id = names.public(&detected.engine_ref).to_string();
    let card = build_card(detected, &public_id, &id.openhydra_peer_id, now_ms, ttl_secs, pricing, region, swarm_binding);
    let signed = sign_card(card, &id.keypair).map_err(|e| e.to_string())?;
    let magnet = signed.to_magnet().map_err(|e| e.to_string())?;
    Ok(CardExport { card: signed, magnet })
}

/// The file extension the desktop app opens cards from (a double-click on one imports it).
pub const CARD_FILE_EXTENSION: &str = "openhydra";

/// `path` ending in `.openhydra`: as given when it already does, otherwise with the extension added
/// after the whole name (`Qwen2.5-7B` → `Qwen2.5-7B.openhydra`, not `Qwen2.openhydra`).
pub fn card_file_path(path: &Path) -> PathBuf {
    let has_ext = path.extension().is_some_and(|e| e.eq_ignore_ascii_case(CARD_FILE_EXTENSION));
    if has_ext {
        path.to_path_buf()
    } else {
        let mut p = path.as_os_str().to_owned();
        p.push(".");
        p.push(CARD_FILE_EXTENSION);
        PathBuf::from(p)
    }
}

/// Save `card` as a `.openhydra` file at `path` (the extension added when it isn't there, see
/// [`card_file_path`]), and return where it was written. The file holds the card's JSON, which is
/// what the app opens and `card import --file` reads. The card is verified first (signature, expiry,
/// privacy, schema), so a file is never written for a card nobody could import.
///
/// An existing file is replaced only with `overwrite` (a save dialog asks first; the command line
/// asks for `--force`), and only when it is itself a card: `--force` with a mistyped path must not
/// destroy a key or a document. A link is never followed. The new file is written beside the old one
/// and renamed over it, so a failed write leaves the old card whole.
pub fn write_card_file(path: &Path, card: &Card, overwrite: bool, now_ms: u64) -> Result<PathBuf, String> {
    verify_card(card, now_ms).map_err(|e| format!("not saving an invalid card: {e}"))?;
    let path = card_file_path(path);
    let shown = path.display();
    let json = card.to_json().map_err(|e| e.to_string())?;
    let mut body = json.into_bytes();
    body.push(b'\n');
    match std::fs::symlink_metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // A new file: never replaces anything that appears meanwhile.
            let mut file = std::fs::OpenOptions::new().write(true).create_new(true).open(&path).map_err(|e| {
                if e.kind() == std::io::ErrorKind::AlreadyExists {
                    format!("{shown} already exists (pass --force to replace it)")
                } else {
                    format!("{shown}: {e}")
                }
            })?;
            use std::io::Write;
            if let Err(e) = file.write_all(&body) {
                drop(file);
                let _ = std::fs::remove_file(&path);
                return Err(format!("{shown}: {e}"));
            }
            Ok(path)
        }
        Err(e) => Err(format!("{shown}: {e}")),
        Ok(_) if !overwrite => Err(format!("{shown} already exists (pass --force to replace it)")),
        Ok(meta) if !meta.file_type().is_file() => {
            Err(format!("{shown} is not a regular file (a link or a folder); not replacing it"))
        }
        Ok(_) => {
            let old = std::fs::read(&path).map_err(|e| format!("{shown}: {e}"))?;
            let is_card = std::str::from_utf8(&old).ok().is_some_and(|t| Card::from_json(t.trim()).is_ok());
            if !is_card {
                return Err(format!("{shown} exists and is not a card; not replacing it"));
            }
            // Beside it (so the rename stays on one file system), under a name nothing else uses.
            let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
            let mut tmp_name = path.file_name().unwrap_or_default().to_owned();
            tmp_name.push(format!(".{}-{nanos}.tmp", std::process::id()));
            let tmp = path.with_file_name(tmp_name);
            let written = std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp).and_then(|mut f| {
                use std::io::Write;
                f.write_all(&body)?;
                f.sync_all()
            });
            if let Err(e) = written.and_then(|_| std::fs::rename(&tmp, &path)) {
                let _ = std::fs::remove_file(&tmp);
                return Err(format!("{shown}: {e}"));
            }
            Ok(path)
        }
    }
}

/// A file name for `model`'s card: the name the model is announced under (so an engine's path, e.g.
/// llama.cpp's `/Users/alice/models/x.gguf`, leaves only `x`: no folder or user name travels with
/// the file) with anything a file system dislikes replaced, and the `.openhydra` extension (e.g.
/// `Qwen/Qwen2.5-7B-Instruct` → `Qwen-Qwen2.5-7B-Instruct.openhydra`).
pub fn card_file_name(model: &str) -> String {
    let public = openhydra_protocol::model_id::public_model_id(model);
    let stem: String = public
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '-' })
        .collect();
    let stem = stem.trim_matches(|c| c == '-' || c == '.');
    format!("{}.{CARD_FILE_EXTENSION}", if stem.is_empty() { "model" } else { stem })
}

/// Parse a card from either a magnet string (`openhydra:card:...`) or JSON, then fully verify it
/// (signature, key↔peer-id binding, privacy, schema, expiry). The shared parse+verify entry point for
/// `card verify` and, later, import. `now_ms` is injected.
pub fn parse_and_verify(
    input: &str,
    now_ms: u64,
) -> Result<openhydra_network::card::VerifiedCard, String> {
    let trimmed = input.trim();
    let card = if trimmed.starts_with("openhydra:card:") {
        Card::from_magnet(trimmed).map_err(|e| e.to_string())?
    } else {
        Card::from_json(trimmed).map_err(|e| e.to_string())?
    };
    verify_card(&card, now_ms).map_err(|e| e.to_string())
}

// ── import store: verified cards routed by peer id without live discovery ──

/// Map a verified card to the consumer's discovery record. Host/port are left empty on purpose — the
/// dial resolves the current address by `libp2p_peer_id` (the card's whole reason to exist), so the
/// only fields that matter are the ids, the model, and the capability hints used for ranking.
pub fn card_to_discovered(card: &Card) -> DiscoveredPeer {
    DiscoveredPeer {
        peer_id: card.openhydra_peer_id.clone(),
        libp2p_peer_id: card.libp2p_peer_id.clone(),
        model_id: card.model_id.clone(),
        canonical_model_id: card.canonical_id.clone(),
        context_length: card.capability.context_length,
        max_output_tokens: card.capability.max_output_tokens,
        // M4: a private card carries its swarm binding through to routing, so the consumer presents
        // the matching credential (and only on this private route).
        swarm_public_key: card.swarm_public_key.clone(),
        ..Default::default()
    }
}

/// How imported cards pin a requested model. A request names a model, never a card; when an
/// imported card names that model, the card's providers serve it rather than anyone announcing the
/// same name.
#[derive(Debug, Clone)]
pub enum CardPin {
    /// No imported card names it: routed as any model is.
    None,
    /// A private card names it: only these providers (and this machine's own), never anyone found
    /// on the network, and the network isn't asked about it.
    Private {
        /// Its unexpired private cards' providers (none once every such card has expired).
        peers: Vec<DiscoveredPeer>,
        /// The swarms its private cards (expired ones too) name.
        swarms: Vec<String>,
    },
    /// Public cards name it: these providers first; another provider only as a fallback, and only
    /// one announcing one of these exact versions (canonical ids; empty when the cards state none).
    Public { peers: Vec<DiscoveredPeer>, canonicals: Vec<String> },
}

/// What a gateway does when a public card's providers can't serve (offline, overloaded, refused).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CardFallback {
    /// Fail with the reason and how many others could serve, so a person can choose.
    Ask,
    /// Use another provider announcing the same exact version, and say so in the reply.
    SameVersion,
    /// Fail; never use another provider.
    Never,
}

impl CardFallback {
    /// From `ask` / `same-version` / `never` (case and surrounding space ignored).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ask" => Some(Self::Ask),
            "same-version" | "same_version" => Some(Self::SameVersion),
            "never" => Some(Self::Never),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ask => "ask",
            Self::SameVersion => "same-version",
            Self::Never => "never",
        }
    }
}

/// The saved fallback choices, by model (`<cards file stem>.prefs.json` beside the cards file).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CardPrefs {
    #[serde(default)]
    pub fallback: std::collections::BTreeMap<String, CardFallback>,
}

/// Where the saved choices for the cards file `cards` live: `cards.json` → `cards.prefs.json`.
pub fn prefs_path(cards: &Path) -> PathBuf {
    let stem = cards.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "cards".into());
    cards.with_file_name(format!("{stem}.prefs.json"))
}

/// Read the saved choices; missing or unreadable → none saved.
pub fn read_card_prefs(path: &Path) -> CardPrefs {
    std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
}

/// Save (or, with `None`, clear) the fallback choice for `model` beside the cards file `cards`.
///
/// The CLI and the app can both write it, so a write holds a lock file beside it (two at once would
/// otherwise lose one's choice), and a file that doesn't parse is refused rather than replaced
/// (that would wipe the other models' choices). The new file is written under a fresh name, never
/// through a link, then renamed into place.
pub fn set_card_fallback(cards: &Path, model: &str, choice: Option<CardFallback>) -> Result<CardPrefs, String> {
    let path = prefs_path(cards);
    let _lock = PrefsLock::acquire(&path)?;
    let mut prefs = match std::fs::read_to_string(&path) {
        Ok(text) if text.trim().is_empty() => CardPrefs::default(),
        Ok(text) => serde_json::from_str(&text)
            .map_err(|e| format!("{} doesn't parse ({e}); fix or remove it first", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => CardPrefs::default(),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    match choice {
        Some(c) => {
            prefs.fallback.insert(model.to_string(), c);
        }
        None => {
            prefs.fallback.remove(model);
        }
    }
    let json = serde_json::to_string_pretty(&prefs).map_err(|e| e.to_string())?;
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let mut tmp_name = path.file_name().unwrap_or_default().to_owned();
    tmp_name.push(format!(".{}-{nanos}.tmp", std::process::id()));
    let tmp = path.with_file_name(tmp_name);
    let written = std::fs::OpenOptions::new().write(true).create_new(true).open(&tmp).and_then(|mut f| {
        use std::io::Write;
        f.write_all(json.as_bytes())?;
        f.sync_all()
    });
    if let Err(e) = written.and_then(|_| std::fs::rename(&tmp, &path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("{}: {e}", path.display()));
    }
    Ok(prefs)
}

/// A lock file beside the saved choices, held while they are rewritten ([`set_card_fallback`]).
struct PrefsLock(PathBuf);

impl PrefsLock {
    /// Wait (up to ~2 s) for the lock. One older than 30 s is a writer that died holding it: taken over.
    fn acquire(prefs: &Path) -> Result<Self, String> {
        let mut name = prefs.file_name().unwrap_or_default().to_owned();
        name.push(".lock");
        let lock = prefs.with_file_name(name);
        for _ in 0..100 {
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&lock) {
                Ok(_) => return Ok(PrefsLock(lock)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let stale = std::fs::metadata(&lock)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .is_some_and(|age| age > std::time::Duration::from_secs(30));
                    if stale {
                        let _ = std::fs::remove_file(&lock);
                        continue;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                Err(e) => return Err(format!("{}: {e}", lock.display())),
            }
        }
        Err(format!("{} is held by another writer; try again", lock.display()))
    }
}

impl Drop for PrefsLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Read a cards file (a JSON array of [`Card`]). Missing/empty → `[]`; a parse error is surfaced.
pub fn read_cards_file(path: &Path) -> Result<Vec<Card>, String> {
    match std::fs::read_to_string(path) {
        Ok(s) if s.trim().is_empty() => Ok(Vec::new()),
        Ok(s) => serde_json::from_str(&s).map_err(|e| format!("parse cards file: {e}")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("read cards file: {e}")),
    }
}

/// Atomically write the cards array (temp + rename, same dir).
fn write_cards_file(path: &Path, cards: &[Card]) -> Result<(), String> {
    let json = serde_json::to_string_pretty(cards).map_err(|e| format!("encode cards: {e}"))?;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("rename into {}: {e}", path.display()))
}

/// Add a verified card to the file, replacing any existing entry for the same `(libp2p_peer_id,
/// model_id)` (a re-import refreshes rather than duplicates). Atomic.
pub fn add_card(path: &Path, card: &Card) -> Result<(), String> {
    let mut cards = read_cards_file(path)?;
    cards.retain(|c| !(c.libp2p_peer_id == card.libp2p_peer_id && c.model_id == card.model_id));
    cards.push(card.clone());
    write_cards_file(path, &cards)
}

/// Remove every card matching `(libp2p_peer_id, model_id)`. Returns how many were removed.
pub fn remove_card(path: &Path, libp2p_peer_id: &str, model_id: &str) -> Result<usize, String> {
    let mut cards = read_cards_file(path)?;
    let before = cards.len();
    cards.retain(|c| !(c.libp2p_peer_id == libp2p_peer_id && c.model_id == model_id));
    let removed = before - cards.len();
    if removed > 0 {
        write_cards_file(path, &cards)?;
    }
    Ok(removed)
}

/// A hot-reloaded set of imported cards the consumer merges into discovery. Re-reads + re-verifies
/// the file only when its (mtime, size) changes, so a `providers_for` call on the serve path is a
/// cheap `stat` in steady state. Verification happens on load (signature/privacy/schema; a bad or
/// already-expired card is dropped); expiry is re-checked per query so a card that lapses while
/// cached stops being offered.
pub struct CardStore {
    path: PathBuf,
    cache: Mutex<CardCache>,
}

#[derive(Default)]
struct CardCache {
    /// The raw file content last parsed (`Some("")` for absent/empty; `None` until first load).
    /// Content-based (not `(mtime,size)`), so an equal-size content swap or a metadata() error can't
    /// leave stale cards — a small local file re-read per query is cheap; the expensive per-card
    /// verify still runs only when the content actually changed.
    raw: Option<String>,
    /// Cards that verified at load time (expiry re-checked per query).
    cards: Vec<Card>,
}

impl CardStore {
    pub fn new(path: PathBuf) -> Arc<Self> {
        Arc::new(Self { path, cache: Mutex::new(CardCache::default()) })
    }

    /// Re-read + re-verify iff the file's CONTENT changed. A missing file means no cards. A file
    /// that can't be read or parsed keeps the cards last read (and says so): an empty list would
    /// drop every private card's pin at once, and its model would then be routed to anyone
    /// announcing the name. Cards are kept when their signature is good even once expired: queries
    /// leave them out, but a private one still pins its model ([`pin_for`](Self::pin_for)).
    /// Poison-tolerant lock.
    fn refresh(&self, _now_ms: u64) {
        let raw = match std::fs::read_to_string(&self.path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => {
                tracing::warn!(path = %self.path.display(), error = %e, "can't read the imported cards; keeping the ones last read");
                return;
            }
        };
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if c.raw.as_deref() == Some(raw.as_str()) {
            return;
        }
        let parsed: Vec<Card> = if raw.trim().is_empty() {
            Vec::new()
        } else {
            match serde_json::from_str(&raw) {
                Ok(cards) => cards,
                Err(e) => {
                    tracing::warn!(path = %self.path.display(), error = %e, "the imported cards file doesn't parse; keeping the cards last read");
                    return;
                }
            }
        };
        // Signature checked as of a moment the card was valid, so an expired card is kept.
        c.cards = parsed.into_iter().filter(|card| verify_card(card, card.expires_at.saturating_sub(1)).is_ok()).collect();
        c.raw = Some(raw);
    }

    /// Verified, unexpired card providers for `model_id`, as discovery records to dial by peer id.
    pub fn providers_for(&self, model_id: &str, now_ms: u64) -> Vec<DiscoveredPeer> {
        self.refresh(now_ms);
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.cards
            .iter()
            .filter(|card| card.model_id == model_id && now_ms < card.expires_at)
            .map(card_to_discovered)
            .collect()
    }

    /// All currently-valid (verified, unexpired) cards — for the desktop "imported" list.
    pub fn valid_cards(&self, now_ms: u64) -> Vec<Card> {
        self.refresh(now_ms);
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        c.cards.iter().filter(|card| now_ms < card.expires_at).cloned().collect()
    }

    /// How the imported cards pin `model_id` (see [`CardPin`]). A private card wins over a public one
    /// for the same name: once a private card names a model, the request never leaves its swarm.
    pub fn pin_for(&self, model_id: &str, now_ms: u64) -> CardPin {
        self.refresh(now_ms);
        let c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        // A private card pins its model even once it has expired: the model then fails (no
        // provider) until the card is renewed or removed, rather than going to anyone announcing
        // the name. Its expired providers aren't used.
        let private_named = c.cards.iter().any(|card| card.model_id == model_id && !card.swarm_public_key.is_empty());
        let cards: Vec<&Card> = c.cards.iter().filter(|card| card.model_id == model_id && now_ms < card.expires_at).collect();
        if private_named {
            let peers = cards.iter().filter(|c| !c.swarm_public_key.is_empty()).map(|c| card_to_discovered(c)).collect();
            let mut swarms: Vec<String> = c
                .cards
                .iter()
                .filter(|card| card.model_id == model_id && !card.swarm_public_key.is_empty())
                .map(|card| card.swarm_public_key.to_ascii_lowercase())
                .collect();
            swarms.sort();
            swarms.dedup();
            return CardPin::Private { peers, swarms };
        }
        if cards.is_empty() {
            return CardPin::None;
        }
        let mut canonicals: Vec<String> = cards.iter().map(|c| c.canonical_id.clone()).filter(|c| !c.is_empty()).collect();
        canonicals.sort();
        canonicals.dedup();
        CardPin::Public { peers: cards.iter().map(|c| card_to_discovered(c)).collect(), canonicals }
    }

    /// The fallback choice saved for `model_id` (`card fallback`, the app's selector), if any.
    pub fn fallback_pref(&self, model_id: &str) -> Option<CardFallback> {
        read_card_prefs(&prefs_path(&self.path)).fallback.get(model_id).copied()
    }

    /// The peer ids of the providers of every unexpired imported card: the peers a gateway's cards
    /// let in (`--allow-swarm-peers`). Importing a card is a deliberate choice of that provider.
    pub fn provider_peer_ids(&self, now_ms: u64) -> std::collections::HashSet<String> {
        self.valid_cards(now_ms).into_iter().map(|c| c.libp2p_peer_id).filter(|p| !p.is_empty()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detected(engine_ref: &str, canonical: &str, params: &str) -> DetectedModel {
        DetectedModel {
            engine_ref: engine_ref.to_string(),
            canonical_id: canonical.to_string(),
            family: "qwen3".to_string(),
            params: params.to_string(),
            quant: "int4".to_string(),
            size_bytes: 0,
            weight_digest: String::new(),
        }
    }

    #[test]
    fn build_card_maps_fields_and_computes_expiry() {
        let m = detected("qwen3:1.7b", "qwen3/1.7b/int4/deadbeef", "1.7b");
        let c = build_card(&m, &m.engine_ref, "oh_abc", 1_000, 3600, PricingMode::Reciprocal, Some("in".into()), None);
        assert_eq!(c.openhydra_peer_id, "oh_abc");
        assert_eq!(c.model_id, "qwen3:1.7b");
        assert_eq!(c.canonical_id, "qwen3/1.7b/int4/deadbeef");
        assert_eq!(c.capability.params, "1.7b");
        assert_eq!(c.region.as_deref(), Some("in"));
        assert_eq!(c.expires_at, 1_000 + 3600 * 1000);
    }

    #[test]
    fn build_card_discloses_the_possession_digest_on_weight_hash() {
        // 2c: an engine that reports a possession digest has it carried onto the (signed)
        // weight_hash…
        let mut m = detected("qwen3:1.7b", "qwen3/1.7b/int4/deadbeef", "1.7b");
        m.weight_digest = "sha256:abc123".into();
        let c = build_card(&m, &m.engine_ref, "oh_abc", 0, 60, PricingMode::Reciprocal, None, None);
        assert_eq!(c.weight_hash, "sha256:abc123", "digest disclosed on the card");
        // …and an engine that reports none leaves weight_hash empty (byte-identical to before).
        let plain = detected("qwen3:1.7b", "qwen3/1.7b/int4/deadbeef", "1.7b"); // weight_digest = ""
        let pc = build_card(&plain, &plain.engine_ref, "oh_abc", 0, 60, PricingMode::Reciprocal, None, None);
        assert!(pc.weight_hash.is_empty(), "no digest → empty weight_hash");
    }

    #[test]
    fn build_card_with_a_swarm_binding_is_private() {
        let m = detected("qwen3:1.7b", "qwen3/1.7b/int4/deadbeef", "1.7b");
        let pk = "a".repeat(64);
        let c = build_card(&m, &m.engine_ref, "oh_abc", 0, 60, PricingMode::Reciprocal, None, Some(pk.clone()));
        assert_eq!(c.swarm_public_key, pk);
        assert!(c.is_private());
        // No binding → public.
        let pubc = build_card(&m, &m.engine_ref, "oh_abc", 0, 60, PricingMode::Reciprocal, None, None);
        assert!(!pubc.is_private());
    }

    #[test]
    fn build_card_uses_the_public_name_never_an_engine_path() {
        // An engine id that is a path (`-m /home/alice/...gguf`, a vLLM directory) is carded under the
        // name the provider announces, which is path-free.
        let m = detected(
            "/home/alice/models/Qwen3-1.7B-Q4_K_M.gguf",
            "qwen3/1.7b/int4/deadbeef",
            "1.7b",
        );
        let names = crate::provider::PublicNames::from_engine_refs([m.engine_ref.as_str()]);
        let c = build_card(&m, names.public(&m.engine_ref), "oh_abc", 0, 60, PricingMode::Reciprocal, None, None);
        assert_eq!(c.model_id, "Qwen3-1.7B-Q4_K_M");
        assert!(!c.model_id.contains('/') && !c.model_id.to_ascii_lowercase().contains(".gguf"));
        // The card would pass the sign-side privacy gate.
        assert!(openhydra_network::card::card_is_privacy_safe(&c).is_ok());
    }

    // ── import store ──

    fn an_identity() -> openhydra_network::identity::Identity {
        // A throwaway ed25519 identity (the temp key file is gone after this call; the keypair lives
        // in the returned value).
        let dir = tempfile::tempdir().unwrap();
        openhydra_network::identity::Identity::load_or_create(&dir.path().join("id.key")).unwrap()
    }

    fn signed_card(id: &openhydra_network::identity::Identity, model_id: &str, expires_at: u64) -> Card {
        let c = Card::new_unsigned(id.openhydra_peer_id.clone(), model_id, 1_000, expires_at);
        sign_card(c, &id.keypair).unwrap()
    }

    fn card_with(id: &openhydra_network::identity::Identity, model_id: &str, canonical: &str, swarm: &str, expires_at: u64) -> Card {
        let mut c = Card::new_unsigned(id.openhydra_peer_id.clone(), model_id, 1_000, expires_at);
        c.canonical_id = canonical.into();
        c.swarm_public_key = swarm.into();
        sign_card(c, &id.keypair).unwrap()
    }

    #[test]
    fn cards_pin_their_model_and_a_private_card_wins() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let (a, b, p) = (an_identity(), an_identity(), an_identity());
        let v = "qwen2.5/7b/q4_k_m/abcd0123abcd0123";
        add_card(&path, &card_with(&a, "qwen", v, "", 9_999_999_999_999)).unwrap();
        add_card(&path, &card_with(&b, "qwen", "", "", 9_999_999_999_999)).unwrap();
        add_card(&path, &card_with(&a, "old", v, "", 5_000)).unwrap();
        let store = CardStore::new(path.clone());
        match store.pin_for("qwen", 2_000) {
            CardPin::Public { peers, canonicals } => {
                assert_eq!(peers.len(), 2, "both cards' providers");
                assert_eq!(canonicals, vec![v.to_string()], "only stated versions");
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(store.pin_for("unknown", 2_000), CardPin::None));
        assert!(matches!(store.pin_for("old", 6_000), CardPin::None), "an expired card pins nothing");
        // A private card for the same name: only its provider, never the public cards' too.
        add_card(&path, &card_with(&p, "qwen", v, &"ab".repeat(32), 9_999_999_999_999)).unwrap();
        match store.pin_for("qwen", 2_000) {
            CardPin::Private { peers, .. } => {
                assert_eq!(peers.len(), 1);
                assert_eq!(peers[0].libp2p_peer_id, p.libp2p_peer_id.to_string());
            }
            other => panic!("a private card must win: {other:?}"),
        }
    }

    #[test]
    fn a_private_card_fails_closed_when_it_expires_or_the_file_breaks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let (p, q) = (an_identity(), an_identity());
        let swarm = "ab".repeat(32);
        add_card(&path, &card_with(&p, "secret", "", &swarm, 5_000)).unwrap();
        add_card(&path, &card_with(&q, "open", "", "", 9_999_999_999_999)).unwrap();
        let store = CardStore::new(path.clone());
        // Expired: still pinned (no providers left), never `None` (which would route it publicly).
        match store.pin_for("secret", 6_000) {
            CardPin::Private { peers, swarms } => {
                assert!(peers.is_empty());
                assert_eq!(swarms, vec![swarm.clone()]);
            }
            other => panic!("an expired private card must still pin: {other:?}"),
        }
        // A file that stops parsing (a bad hand edit) keeps the cards last read.
        std::fs::write(&path, "[{ broken").unwrap();
        assert!(matches!(store.pin_for("secret", 6_000), CardPin::Private { .. }), "still pinned");
        assert!(matches!(store.pin_for("open", 6_000), CardPin::Public { .. }));
        // Removing the file is removing the cards.
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(store.pin_for("secret", 6_000), CardPin::None));
    }

    #[test]
    fn a_gateways_cards_name_the_providers_it_may_reach() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let (a, b) = (an_identity(), an_identity());
        add_card(&path, &card_with(&a, "m1", "", "", 9_999_999_999_999)).unwrap();
        add_card(&path, &card_with(&b, "m2", "", &"cd".repeat(32), 5_000)).unwrap();
        let store = CardStore::new(path);
        let peers = store.provider_peer_ids(1_000);
        assert_eq!(peers.len(), 2);
        assert!(peers.contains(&a.libp2p_peer_id.to_string()));
        assert_eq!(store.provider_peer_ids(6_000), [a.libp2p_peer_id.to_string()].into(), "an expired card's provider no longer");
    }

    #[test]
    fn fallback_choices_are_saved_beside_the_cards_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards-gw.json");
        assert_eq!(prefs_path(&path), dir.path().join("cards-gw.prefs.json"));
        let store = CardStore::new(path.clone());
        assert_eq!(store.fallback_pref("qwen"), None);
        set_card_fallback(&path, "qwen", Some(CardFallback::Never)).unwrap();
        set_card_fallback(&path, "other", Some(CardFallback::Ask)).unwrap();
        assert_eq!(store.fallback_pref("qwen"), Some(CardFallback::Never));
        set_card_fallback(&path, "qwen", None).unwrap();
        assert_eq!(store.fallback_pref("qwen"), None);
        assert_eq!(store.fallback_pref("other"), Some(CardFallback::Ask));
        for (s, c) in [("ask", CardFallback::Ask), (" Same-Version ", CardFallback::SameVersion), ("never", CardFallback::Never)] {
            assert_eq!(CardFallback::parse(s), Some(c));
            assert_eq!(CardFallback::parse(c.as_str()), Some(c));
        }
        assert_eq!(CardFallback::parse("always"), None);
        // Two writers at once: neither loses the other's choice.
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let path = path.clone();
                std::thread::spawn(move || set_card_fallback(&path, &format!("m{i}"), Some(CardFallback::Never)).unwrap())
            })
            .collect();
        handles.into_iter().for_each(|h| { h.join().unwrap(); });
        assert_eq!(read_card_prefs(&prefs_path(&path)).fallback.len(), 9, "8 new + `other`");
        // A file that doesn't parse is refused, not replaced (the other choices would be lost).
        std::fs::write(prefs_path(&path), "{ not json").unwrap();
        assert!(set_card_fallback(&path, "qwen", Some(CardFallback::Ask)).unwrap_err().contains("doesn't parse"));
        assert_eq!(std::fs::read_to_string(prefs_path(&path)).unwrap(), "{ not json");
    }

    #[test]
    fn card_to_discovered_maps_ids_and_model() {
        let id = an_identity();
        let c = signed_card(&id, "qwen3:1.7b", 9_999_999_999_999);
        let d = card_to_discovered(&c);
        assert_eq!(d.libp2p_peer_id, c.libp2p_peer_id, "the dial target");
        assert_eq!(d.peer_id, c.openhydra_peer_id, "the reputation/receipt key");
        assert_eq!(d.model_id, "qwen3:1.7b");
    }

    #[test]
    fn add_card_dedups_by_peer_and_model_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let id = an_identity();
        let c1 = signed_card(&id, "qwen3:1.7b", 9_999_999_999_999);
        add_card(&path, &c1).unwrap();
        add_card(&path, &c1).unwrap(); // same (peer, model) → replace, not duplicate
        let back = read_cards_file(&path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0], c1);
        // a different model from the same provider is a distinct entry
        add_card(&path, &signed_card(&id, "qwen2.5:0.5b", 9_999_999_999_999)).unwrap();
        assert_eq!(read_cards_file(&path).unwrap().len(), 2);
    }

    #[test]
    fn store_providers_for_filters_by_model_and_expiry() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let id = an_identity();
        add_card(&path, &signed_card(&id, "qwen3:1.7b", 5_000)).unwrap();
        let store = CardStore::new(path);
        let p = store.providers_for("qwen3:1.7b", 4_000);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].libp2p_peer_id, id.libp2p_peer_id.to_string());
        assert!(store.providers_for("other:model", 4_000).is_empty(), "wrong model → none");
        assert!(store.providers_for("qwen3:1.7b", 5_000).is_empty(), "at expiry → none");
    }

    #[test]
    fn store_hot_reloads_on_add_and_remove() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let id = an_identity();
        let store = CardStore::new(path.clone());
        assert!(store.providers_for("qwen3:1.7b", 1_500).is_empty(), "empty at first");
        add_card(&path, &signed_card(&id, "qwen3:1.7b", 9_999_999_999_999)).unwrap();
        assert_eq!(store.providers_for("qwen3:1.7b", 1_500).len(), 1, "picks up the import");
        assert_eq!(remove_card(&path, &id.libp2p_peer_id.to_string(), "qwen3:1.7b").unwrap(), 1);
        assert!(store.providers_for("qwen3:1.7b", 1_500).is_empty(), "picks up the removal");
    }

    #[test]
    fn store_drops_an_unverifiable_card_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let id = an_identity();
        let mut tampered = signed_card(&id, "qwen3:1.7b", 9_999_999_999_999);
        tampered.model_id = "qwen3:8b".into(); // breaks the signature
        std::fs::write(&path, serde_json::to_string(&vec![tampered]).unwrap()).unwrap();
        let store = CardStore::new(path);
        assert!(
            store.providers_for("qwen3:8b", 1_500).is_empty(),
            "a card whose signature doesn't verify must never be served"
        );
    }

    #[test]
    fn export_refuses_without_a_share_policy_file() {
        // Review fix (MODERATE): fail-closed — no --share-policy-file must NOT fall back to share-all
        // (which would export a private model). Checked before detection, so no live engine needed.
        let dir = tempfile::tempdir().unwrap();
        let err = run_export("qwen3:1.7b", None, 60, PricingMode::Reciprocal, None, &dir.path().join("id.key"), 1_000, None, None, Vec::new())
            .unwrap_err();
        assert!(err.contains("--share-policy-file"), "got: {err}");
    }

    #[test]
    fn store_reloads_on_an_equal_size_content_swap() {
        // Review fix (LOW): content-based invalidation must catch a change that keeps the file's
        // (mtime, size) identical — e.g. swapping one card for another of the same byte length.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cards.json");
        let id = an_identity();
        // Two models with the SAME-length engine handle → same-length serialized files.
        let a = signed_card(&id, "modelaaaa", 9_999_999_999_999);
        let b = signed_card(&id, "modelbbbb", 9_999_999_999_999);
        std::fs::write(&path, serde_json::to_string(&vec![a]).unwrap()).unwrap();
        let store = CardStore::new(path.clone());
        assert_eq!(store.providers_for("modelaaaa", 1_500).len(), 1);
        // Overwrite with the equal-length b (same file size); content-based refresh must notice.
        std::fs::write(&path, serde_json::to_string(&vec![b]).unwrap()).unwrap();
        assert!(store.providers_for("modelaaaa", 1_500).is_empty(), "old card must be gone");
        assert_eq!(store.providers_for("modelbbbb", 1_500).len(), 1, "new card must be served");
    }

    #[test]
    fn a_private_card_must_name_a_swarm_the_model_is_shared_with() {
        let (a, b) = ("aa".repeat(32), "bb".repeat(32));
        let mut policy = crate::share_policy::SharePolicy::share_all();
        assert!(card_swarm_allowed(&policy, "m", "m", &b).is_ok(), "unbound: any owned swarm");
        policy.swarms.insert("m".into(), std::collections::BTreeSet::from([a.clone()]));
        assert!(card_swarm_allowed(&policy, "m", "m", &a.to_ascii_uppercase()).is_ok());
        let err = card_swarm_allowed(&policy, "m", "m", &b).unwrap_err();
        assert!(err.contains("not shared with swarm"), "{err}");
    }

    #[test]
    fn a_saved_card_file_opens_as_the_same_card_and_is_never_clobbered() {
        let dir = tempfile::tempdir().unwrap();
        let id = an_identity();
        let card = signed_card(&id, "qwen3:1.7b", 9_999_999_999_999);
        // No extension: `.openhydra` is added.
        let written = write_card_file(&dir.path().join("mine"), &card, false, 1_000).unwrap();
        assert_eq!(written, dir.path().join("mine.openhydra"));
        // It imports as the same card, through the same entry point the app and `card import` use.
        let text = std::fs::read_to_string(&written).unwrap();
        let back = parse_and_verify(&text, 1_000).unwrap();
        assert_eq!(back.card, card);
        // An existing file is kept unless asked.
        let err = write_card_file(&written, &card, false, 1_000).unwrap_err();
        assert!(err.contains("already exists"), "{err}");
        assert!(write_card_file(&written, &card, true, 1_000).is_ok(), "replaced when asked");
        assert_eq!(parse_and_verify(&std::fs::read_to_string(&written).unwrap(), 1_000).unwrap().card, card);
        // A dotted name keeps its whole name and still gets the extension.
        let dotted = write_card_file(&dir.path().join("Qwen2.5-7B"), &card, false, 1_000).unwrap();
        assert_eq!(dotted, dir.path().join("Qwen2.5-7B.openhydra"));
        assert_eq!(write_card_file(&dir.path().join("x.OPENHYDRA"), &card, false, 1_000).unwrap(), dir.path().join("x.OPENHYDRA"));
        // `--force` never replaces a file that isn't a card (a mistyped path to a key), nor a link.
        let key = dir.path().join("id.key.openhydra");
        std::fs::write(&key, b"not a card").unwrap();
        let err = write_card_file(&key, &card, true, 1_000).unwrap_err();
        assert!(err.contains("not a card"), "{err}");
        assert_eq!(std::fs::read(&key).unwrap(), b"not a card");
        #[cfg(unix)]
        {
            let target = dir.path().join("elsewhere");
            std::fs::write(&target, b"keep").unwrap();
            let link = dir.path().join("link.openhydra");
            std::os::unix::fs::symlink(&target, &link).unwrap();
            assert!(write_card_file(&link, &card, true, 1_000).is_err(), "a link is not followed");
            assert!(write_card_file(&link, &card, false, 1_000).is_err());
            assert_eq!(std::fs::read(&target).unwrap(), b"keep");
        }
        // No temporary file is left behind.
        assert!(std::fs::read_dir(dir.path()).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().ends_with(".tmp")));
        // A tampered or expired card is not saved.
        let mut tampered = card.clone();
        tampered.model_id = "qwen3:8b".into();
        assert!(write_card_file(&dir.path().join("bad.openhydra"), &tampered, false, 1_000).is_err());
        assert!(!dir.path().join("bad.openhydra").exists());
        let old = signed_card(&id, "qwen3:1.7b", 2_000);
        assert!(write_card_file(&dir.path().join("old.openhydra"), &old, false, 3_000).is_err());
    }

    #[test]
    fn a_card_file_name_comes_from_the_model_name() {
        assert_eq!(card_file_name("Qwen/Qwen2.5-7B-Instruct"), "Qwen-Qwen2.5-7B-Instruct.openhydra");
        assert_eq!(card_file_name("qwen3:1.7b"), "qwen3-1.7b.openhydra");
        assert_eq!(card_file_name("///"), "model.openhydra");
        // An engine's path leaves only the model's own name: no folder or user name in the file name.
        assert_eq!(card_file_name("/Users/alice/models/Qwen3.5-9B-Q4_K_M.gguf"), "Qwen3.5-9B-Q4_K_M.openhydra");
        assert_eq!(card_file_name("v1-5-pruned-emaonly-fp16.safetensors"), "v1-5-pruned-emaonly-fp16.openhydra");
    }

    /// A ComfyUI checkpoint is carded under its stem; the verifier stays as strict as ever (a card
    /// naming the file is still refused), and the stem maps back to the file for serving.
    #[test]
    fn a_checkpoint_is_carded_under_its_stem_without_relaxing_the_verifier() {
        let id = an_identity();
        let file = "v1-5-pruned-emaonly-fp16.safetensors";
        let names = crate::provider::PublicNames::from_engine_refs([file]);
        let public = names.public(file).to_string();
        assert_eq!(public, "v1-5-pruned-emaonly-fp16");
        assert_eq!(names.engine_ref(&public), file, "requests for the stem are served by the file");
        let mut m = detected(file, "", "");
        m.family = "stable-diffusion".into();
        let card = build_card(&m, &public, &id.openhydra_peer_id, 1_000, 3_600, PricingMode::Reciprocal, None, None);
        let signed = sign_card(card, &id.keypair).expect("a stem-named card signs");
        assert!(verify_card(&signed, 2_000).is_ok());
        let by_file = build_card(&m, file, &id.openhydra_peer_id, 1_000, 3_600, PricingMode::Reciprocal, None, None);
        assert!(sign_card(by_file, &id.keypair).is_err(), "a card naming the weight file is still refused");
    }
}
