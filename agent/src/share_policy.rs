// Copyright 2026 OpenHydra
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//! Which models a provider shares (announces + serves), as an **explicit, durable policy** rather
//! than a launch-time allowlist.
//!
//! The old model was `Option<HashSet<String>>` where an *empty* set meant "share everything" — a
//! sentinel that made "share nothing" unrepresentable and conflated "the operator hasn't chosen"
//! with "share all". [`SharePolicy`] replaces it with explicit intent:
//!
//! * [`ShareMode::All`] — share every model the engine currently exposes, **including models that
//!   appear later** (the desktop's "Share everything" master switch).
//! * [`ShareMode::List`] — share **exactly** the engine-handles in `models`; a model not listed is
//!   neither announced nor served, and a newly-detected model stays off until explicitly added.
//!
//! The policy is serialized to `~/.openhydra/share-policy.json` (owned by the agent, written by the
//! desktop on toggle) and hot-reloaded by the running provider — so a user can change what they
//! share at any time without restarting the node. This module is the single definition shared by
//! the agent binary and the desktop (which path-depends on the agent crate as a library).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};
use std::time::SystemTime;

use crate::adapter::normalize_engine_ref;

/// How a provider decides which detected models to share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShareMode {
    /// Share every model the engine exposes now or in the future.
    All,
    /// Share only the explicitly-listed engine handles (see [`SharePolicy::models`]). The default
    /// for an unpopulated status view (a gateway, or a provider before its first announce) — it
    /// pairs with an empty list to read honestly as "nothing shared yet".
    #[default]
    List,
}

/// How **far** a shared model reaches (M1 — orthogonal to [`ShareMode`], which decides *which*
/// models are shared). Only [`Scope::Global`] reaches the public DHT / marketplace, and only with
/// recorded consent — see [`SharePolicy::announce_globally`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    /// Loopback only — this machine's gateway. Never announced globally. (Full loopback-serve
    /// enforcement is later; M1 gates the global announce.)
    Device,
    /// The operator's trust domain — LAN/mDNS today, cross-network in M4. **Not** on the global
    /// marketplace. The privacy-first default the desktop preselects for a newly-shared model.
    Private,
    /// Announced to the global DHT / marketplace. **Always requires a recorded consent** — a
    /// per-model [`SharePolicy::global_consent`] entry, or (for a default-Global model) the
    /// policy-level [`SharePolicy::default_global_consent`]. There is no un-consented global path.
    Global,
    /// An unrecognised scope string from a forward/typo'd file. Deserialises here (via
    /// `#[serde(other)]`) instead of failing the *whole* policy closed, and is treated as **not
    /// globally announced** (safe) — so one stray character can't silently un-share everything.
    #[serde(other)]
    Unknown,
}

fn default_version() -> u32 {
    1
}

/// Rank a scope by how restrictive it is (higher = stricter): `Global` 0 < `Private`/`Unknown` 1 <
/// `Device` 2. `Unknown` ranks with `Private` (fail-closed — an unrecognised scope is at least
/// swarm-gated).
fn scope_rank(s: Scope) -> u8 {
    match s {
        Scope::Global => 0,
        Scope::Private | Scope::Unknown => 1,
        Scope::Device => 2,
    }
}

/// The stricter of two scopes (used to fold a policy down to its most-restrictive reach).
fn stricter(a: Scope, b: Scope) -> Scope {
    if scope_rank(b) > scope_rank(a) {
        b
    } else {
        a
    }
}

/// The reach of a shared model with no explicit [`SharePolicy::scopes`] entry. Serde-defaults to
/// `Global` **on purpose**: a pre-scope (v1) policy file had no scope concept and announced every
/// shared model globally, so reading one back must keep doing exactly that — upgrading the binary
/// must never silently un-share a user's models. The desktop writes `Private` here once it owns the
/// scope UI, which flips new/unset models to private-by-default from that point on.
fn default_scope() -> Scope {
    Scope::Global
}

/// The schema version new policies are written at. Bumped to 3 for consent-hardening: a pre-3
/// file is migrated on [`SharePolicy::load`] (its grandfathered-Global announce set is preserved
/// by materialising a consent record — never silently un-shared). The serde default for a
/// *missing* `version` field stays `1` ([`default_version`]) so an old/unversioned file is still
/// recognised as pre-3 and migrated.
const CURRENT_VERSION: u32 = 3;

/// A wall-clock consent timestamp (unix ms) for a config-derived global consent recorded **now** —
/// an explicit `--share-models` / `share_all()` opt-in this invocation. A real, orderable marker;
/// never `0`. (Migration-*inherited* consent uses [`MIGRATION_CONSENT_TS`] instead, so it doesn't
/// churn across restarts — see [`SharePolicy::migrate_consent`].)
fn consent_ts() -> u64 {
    SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(1)
}

/// Sentinel consent timestamp (unix ms) for a **migration-inherited** global consent. A legacy
/// pre-hardening file expressed the operator's opt-in without ever recording a time, so migration
/// stamps a stable, deterministic marker rather than `SystemTime::now()` — which would churn on
/// every restart of a file that's never persisted back (`load()` migrates in memory only; a headless
/// node re-migrates each start — Finding 3). `1` = "epoch+1ms": non-zero, orderable **before** any
/// real consent, so the audit trail reads it as "inherited, pre-v3" rather than a fabricated moment.
const MIGRATION_CONSENT_TS: u64 = 1;

/// A provider's model-sharing policy. Serializes to e.g.
/// `{"version":1,"mode":"list","models":["qwen3-coder:30b-a3b-q8_0"]}`.
///
/// `models` is meaningful only in [`ShareMode::List`]; it is ignored (and conventionally empty) in
/// [`ShareMode::All`]. A `BTreeSet` gives deterministic serialization (stable diffs + testable
/// output) and free de-duplication.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SharePolicy {
    /// Schema version, so the shape can evolve without silently misreading old files.
    #[serde(default = "default_version")]
    pub version: u32,
    pub mode: ShareMode,
    /// The explicit allowlist for [`ShareMode::List`]. Empty in [`ShareMode::All`].
    #[serde(default)]
    pub models: BTreeSet<String>,
    /// Per-model reach (M1). A shared model absent here uses [`Self::default_scope`]. Keyed by the
    /// clean engine handle (migrated like [`Self::models`]).
    #[serde(default)]
    pub scopes: BTreeMap<String, Scope>,
    /// Reach for a shared model with no explicit [`Self::scopes`] entry. Serde-defaults to `Global`
    /// so a pre-scope policy keeps announcing exactly what it did; the desktop writes `Private` here
    /// to make new/unset models private-by-default.
    #[serde(default = "default_scope")]
    pub default_scope: Scope,
    /// Per-model consent timestamps (unix ms) recorded when the operator confirms Global publish.
    /// An **explicit** [`Scope::Global`] announces globally only with a matching entry here
    /// (fail-closed). Keyed by clean handle.
    #[serde(default)]
    pub global_consent: BTreeMap<String, u64>,
    /// Policy-level consent (unix ms) to announce every **default-Global** model globally —
    /// including models added later. This is what a "share everything globally" choice (or the
    /// migration of a legacy default-Global policy) records, so a model reaching global discovery
    /// via [`Self::default_scope`] `== Global` still has a recorded consent. `None` ⇒ default-Global
    /// models are **not** announced (fail-closed). See [`Self::announce_globally`].
    #[serde(default)]
    pub default_global_consent: Option<u64>,
}

impl Default for SharePolicy {
    /// The headless default when nothing is configured: share everything (preserves the historical
    /// `--share-models`-absent behavior).
    fn default() -> Self {
        Self::share_all()
    }
}

impl SharePolicy {
    /// Share every detected model (now and future). Headless "share everything" — the operator's
    /// explicit config IS their consent, so it carries a policy-level global consent (announces).
    pub fn share_all() -> Self {
        Self {
            version: CURRENT_VERSION,
            mode: ShareMode::All,
            models: BTreeSet::new(),
            scopes: BTreeMap::new(),
            default_scope: default_scope(),
            global_consent: BTreeMap::new(),
            default_global_consent: Some(consent_ts()),
        }
    }

    /// Share **no** models (mode `list`, empty). The fail-closed state: what a corrupt policy or a
    /// poisoned lock resolves to, so a sharing control never silently *widens* to share-all.
    pub fn share_nothing() -> Self {
        Self {
            version: CURRENT_VERSION,
            mode: ShareMode::List,
            models: BTreeSet::new(),
            scopes: BTreeMap::new(),
            default_scope: default_scope(),
            global_consent: BTreeMap::new(),
            default_global_consent: None,
        }
    }

    /// Share exactly the given engine handles. Duplicates collapse; order is irrelevant.
    pub fn share_list<I, S>(models: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            version: CURRENT_VERSION,
            mode: ShareMode::List,
            models: models.into_iter().map(Into::into).collect(),
            scopes: BTreeMap::new(),
            default_scope: default_scope(),
            global_consent: BTreeMap::new(),
            default_global_consent: None,
        }
    }

    /// Migrate a legacy `--share-models` list into a policy: an **empty** list historically meant
    /// "share everything" ([`ShareMode::All`]); a non-empty list becomes an explicit
    /// [`ShareMode::List`]. Keeps existing CLI/settings behavior identical across the upgrade.
    pub fn from_legacy_list<I, S>(models: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        // Normalise here too: a legacy `settings.shared_models` may key a llama.cpp model by its
        // absolute `-m` path, which no longer matches the clean handle the adapter now advertises.
        let set: BTreeSet<String> =
            models.into_iter().map(|m| normalize_engine_ref(&m.into())).collect();
        if set.is_empty() {
            Self::share_all()
        } else {
            // An explicit `--share-models` list is the operator's opt-in → carries a policy-level
            // global consent so the listed models keep announcing (default-Global) under the
            // consent-required rule, same as before hardening.
            Self {
                version: CURRENT_VERSION,
                mode: ShareMode::List,
                models: set,
                scopes: BTreeMap::new(),
                default_scope: default_scope(),
                global_consent: BTreeMap::new(),
                default_global_consent: Some(consent_ts()),
            }
        }
    }

    /// Migrate the `models` set to clean engine handles (see [`normalize_engine_ref`]). Older policy
    /// files — and legacy `settings.shared_models` — may key a llama.cpp model by its absolute `-m`
    /// path (`/home/user/models/X.gguf`); the adapter now advertises the clean handle (`X`), so
    /// without this a previously-shared model would silently stop matching and un-share until the
    /// user re-toggled it. Idempotent for already-clean ids; a no-op in `all` mode.
    fn normalize_models(&mut self) {
        if self.mode == ShareMode::List {
            self.models = self.models.iter().map(|m| normalize_engine_ref(m)).collect();
        }
        // The scope/consent maps are keyed by the same model handle, so migrate them too — a
        // future path-keyed entry would otherwise stop matching the clean advertised handle.
        if !self.scopes.is_empty() {
            self.scopes =
                self.scopes.iter().map(|(k, v)| (normalize_engine_ref(k), *v)).collect();
        }
        if !self.global_consent.is_empty() {
            self.global_consent =
                self.global_consent.iter().map(|(k, v)| (normalize_engine_ref(k), *v)).collect();
        }
    }

    /// The reach of `engine_ref`: its explicit [`Self::scopes`] entry, or [`Self::default_scope`].
    /// (Only meaningful for a shared model; the caller pairs this with [`Self::is_shared`].)
    pub fn scope_of(&self, engine_ref: &str) -> Scope {
        self.scopes.get(engine_ref).copied().unwrap_or(self.default_scope)
    }

    /// The STRICTEST reach across the whole policy — `default_scope` and every per-model entry.
    /// Used as the fail-closed scope for a serve whose `model_ref` is not a recognised announced
    /// handle (M4 review HIGH): in `All` mode `is_shared` says yes to any ref, but such a ref could
    /// be an alias the engine resolves to *any* shared model — including the strictest — so it must
    /// be gated at least as strictly as the strictest model this node shares. Strictness order:
    /// `Device` > `Private`/`Unknown` > `Global`. An all-`Global` policy stays `Global` (no
    /// regression for a purely-public node); one private model makes an unknown alias require a
    /// credential.
    pub fn strictest_scope(&self) -> Scope {
        let mut strictest = self.default_scope;
        for s in self.scopes.values() {
            strictest = stricter(strictest, *s);
        }
        strictest
    }

    /// Whether `engine_ref` should be announced to the **global** discovery (DHT / marketplace).
    /// True only when the model is shared, its scope resolves to [`Scope::Global`], and Global
    /// publish is consented. Consent rule (fail-closed):
    /// * an **explicit** `Global` (set via the desktop's scope control) needs a matching
    ///   [`Self::global_consent`] entry — a hand-edited `scope:"global"` with no consent is NOT
    ///   announced;
    /// * a model that is Global via [`Self::default_scope`] (no explicit entry) needs the
    ///   policy-level [`Self::default_global_consent`] — with no record it is NOT announced. There
    ///   is **no un-consented global path**; a pre-hardening file keeps its behaviour because
    ///   [`Self::load`] materialises the record on migration (never silently un-shares).
    pub fn announce_globally(&self, engine_ref: &str) -> bool {
        if !self.is_shared(engine_ref) || self.scope_of(engine_ref) != Scope::Global {
            return false;
        }
        if self.scopes.contains_key(engine_ref) {
            // Explicit Global → needs a matching per-model consent.
            self.global_consent.contains_key(engine_ref)
        } else {
            // Global via `default_scope` → needs the policy-level consent. NO grandfather: a
            // default-Global model with no `default_global_consent` is NOT announced (fail-closed).
            // Legacy behaviour is preserved instead by [`Self::load`] materialising the record.
            self.default_global_consent.is_some()
        }
    }

    /// Whether `engine_ref` (a detected model's engine handle — the exact string a consumer sends
    /// as `ServeRequest.model_ref`) is shared under this policy. This is the single decision both
    /// the announce filter and the serve gate consult.
    pub fn is_shared(&self, engine_ref: &str) -> bool {
        // An empty `engine_ref` is never a real model handle → never shared, in either mode.
        if engine_ref.is_empty() {
            return false;
        }
        match self.mode {
            ShareMode::All => true,
            ShareMode::List => self.models.contains(engine_ref),
        }
    }

    /// Parse a policy from a JSON file. Errors on a missing file (`NotFound`) or malformed JSON
    /// (`InvalidData`) — the caller decides the fallback (the running provider keeps its previous
    /// in-memory policy, so a bad write never silently opens up or shuts down sharing).
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let mut policy: Self = serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        // F5 migration: fold legacy path-keyed entries to clean handles so a model shared under an
        // old build's absolute-path id stays shared against the now-clean advertised handle.
        policy.normalize_models();
        // Consent-hardening migration (v< CURRENT_VERSION): before hardening, a default-Global model
        // was announced globally with no consent record (the removed grandfather). announce_globally
        // now REQUIRES a record — so to preserve the exact announced set without silently
        // un-sharing, materialise the policy-level consent for a pre-3 policy that was actually
        // announcing default-Global models (mode All, or a non-empty list). A file that shared
        // nothing gets no consent (so a later share is a fresh, consent-gated decision). Idempotent.
        policy.migrate_consent();
        Ok(policy)
    }

    /// One-shot, idempotent migration of a pre-hardening policy so the airtight consent rule keeps
    /// the same models announced. Under the removed grandfather a model was announced iff
    /// `is_shared && scope_of == Global`, so migration materialises a consent record for **both**
    /// global paths that were announcing: the policy-level default, and every explicit
    /// `scopes:{m:"global"}` entry (whose per-model consent map didn't exist pre-hardening). Never
    /// pre-consents a model that wasn't shared, and leaves an already-hardened (`v>=CURRENT`)
    /// policy, a Private default, or a share-nothing policy untouched.
    ///
    /// **Deterministic** (Finding 3): the materialised records use the fixed [`MIGRATION_CONSENT_TS`]
    /// sentinel, not `SystemTime::now()`, so re-migrating the same on-disk file (a headless node that
    /// never persists back re-migrates on every `load()`/hot-reload) produces byte-identical output —
    /// the consent timestamp can't churn across restarts. See [`Self::load`].
    fn migrate_consent(&mut self) {
        if self.version >= CURRENT_VERSION {
            return;
        }
        let announced_something = self.mode == ShareMode::All || !self.models.is_empty();
        if self.default_scope == Scope::Global
            && self.default_global_consent.is_none()
            && announced_something
        {
            self.default_global_consent = Some(MIGRATION_CONSENT_TS);
        }
        // Explicit `scope:"global"` entries were announced under the old grandfather too, but
        // `announce_globally` routes them through the per-model consent branch (which ignores the
        // default consent above). Materialise a per-model record for each **shared** explicit-Global
        // model missing one — so migration preserves exactly the old announced set with no silent
        // un-share (adversarial review Finding 1). Guarded by `is_shared`: an explicit-Global model
        // that wasn't actually shared wasn't announced, so it must NOT be pre-consented.
        let needs_consent: Vec<String> = self
            .scopes
            .iter()
            .filter(|(m, s)| {
                **s == Scope::Global && self.is_shared(m) && !self.global_consent.contains_key(*m)
            })
            .map(|(m, _)| m.clone())
            .collect();
        for m in needs_consent {
            self.global_consent.insert(m, MIGRATION_CONSENT_TS);
        }
        self.version = CURRENT_VERSION;
    }

    /// Write the policy to `path` **atomically** (temp file + rename), so a crash or a concurrent
    /// reader can never observe a half-written file — the reader sees either the previous complete
    /// file or the new one. Rename is atomic on the same filesystem, so the temp sits beside `path`.
    pub fn write_atomic(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, json.as_bytes())?;
        std::fs::rename(&tmp, path)
    }
}

/// The provider's live share state as the status API reports it: the current policy (mode + the
/// explicit list, empty in `all`) plus the engine handles actually published by the most recent
/// announce. The desktop reads this to render each model's real state (announced / pending / off)
/// instead of guessing from detection + optimistic settings.
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ShareStatusView {
    pub share_mode: ShareMode,
    /// The explicit allowlist (meaningful in `list` mode; empty in `all`).
    pub shared_models: Vec<String>,
    /// What the most recent announce actually advertised — the provider's real broadcast set.
    pub announced_models: Vec<String>,
}

/// The last-modified time of `path`, or `None` if it can't be stat'd (missing / permissions).
fn file_mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Holds a provider's live [`SharePolicy`] and, when file-backed, hot-reloads it when the file
/// changes on disk. Self-contained (no network / engine), so the reload state machine is unit
/// testable. All accessors are `&self` (interior mutability) so it can live behind an `Arc` inside
/// the provider and be read from many worker threads while the poll thread reloads it.
pub struct PolicyWatcher {
    policy: RwLock<SharePolicy>,
    /// `Some(path)` → file-backed + hot-reloaded; `None` → a static policy that never reloads.
    path: Option<PathBuf>,
    /// Last-seen mtime of `path`, for cheap change detection (one `stat` per [`Self::reload_if_changed`]).
    last_mtime: Mutex<Option<SystemTime>>,
    /// Serializes the whole check→load→swap→record of [`Self::reload_if_changed`] so the poll thread
    /// and the filesystem-watcher thread can't both observe one change and double-reload/announce.
    reload_lock: Mutex<()>,
}

impl PolicyWatcher {
    /// A static (non-reloading) watcher around a fixed policy — for tests and non-file callers.
    pub fn r#static(policy: SharePolicy) -> Self {
        Self { policy: RwLock::new(policy), path: None, last_mtime: Mutex::new(None), reload_lock: Mutex::new(()) }
    }

    /// A static watcher from a legacy `--share-models` list (empty ⇒ share-all).
    pub fn from_legacy_list<I, S>(models: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::r#static(SharePolicy::from_legacy_list(models))
    }

    /// A file-backed, hot-reloaded watcher. Loads `path` now; a missing or malformed file falls
    /// back to share-everything with a warning (so a first-run/parse hiccup never silently shares
    /// nothing — the desktop writes a valid file before launch in the normal flow).
    pub fn from_file(path: PathBuf) -> Self {
        // Stat BEFORE load (M2): the recorded mtime must never be *newer* than the content we read.
        // If a write lands between the stat and the load, we record the older mtime and `reload_if_
        // changed` catches the new content on the next poll — vs stat-after-load, which could record
        // the new mtime against old content and never reload it.
        let mtime = file_mtime(&path);
        let policy = match SharePolicy::load(&path) {
            Ok(p) => p,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Absent on first run (before the desktop writes it) → share-all default.
                eprintln!(
                    "openhydra-agent: share-policy file {} not found — sharing all detected models until it is written",
                    path.display()
                );
                SharePolicy::share_all()
            }
            Err(e) => {
                // Corrupt (M1): fail CLOSED. A restrictive policy must never silently widen to
                // share-all on a bad read — share nothing until it's fixed. The desktop self-heals
                // the file + notifies; this is the agent-side floor for CLI / post-heal corruption.
                eprintln!(
                    "openhydra-agent: share-policy file {} unreadable ({e}) — sharing NOTHING until it is fixed (fail-closed)",
                    path.display()
                );
                SharePolicy::share_nothing()
            }
        };
        Self { policy: RwLock::new(policy), path: Some(path), last_mtime: Mutex::new(mtime), reload_lock: Mutex::new(()) }
    }

    /// Whether `engine_ref` is shared under the current (possibly hot-reloaded) policy. A poisoned
    /// lock falls back to share-nothing (fail-closed — matters for the serve gate).
    pub fn is_shared(&self, engine_ref: &str) -> bool {
        self.policy.read().map(|p| p.is_shared(engine_ref)).unwrap_or(false)
    }

    /// Whether `engine_ref` should be announced to the **global** discovery under the current
    /// (possibly hot-reloaded) policy — see [`SharePolicy::announce_globally`]. Fail-closed
    /// (do NOT announce) on a poisoned lock, consistent with [`Self::is_shared`].
    pub fn announce_globally(&self, engine_ref: &str) -> bool {
        self.policy.read().map(|p| p.announce_globally(engine_ref)).unwrap_or(false)
    }

    /// The reach of `engine_ref` under the current (possibly hot-reloaded) policy — the M4 serve gate
    /// consults this to decide whether a request needs a swarm credential. Fail-closed to
    /// [`Scope::Private`] on a poisoned lock: never report `Global` (which would drop the credential
    /// requirement) from a broken lock — the most a poison can do is force the auth gate, never open a
    /// private model. (`is_shared` runs first and already fails closed, so this is a defence-in-depth.)
    pub fn scope_of(&self, engine_ref: &str) -> Scope {
        self.policy.read().map(|p| p.scope_of(engine_ref)).unwrap_or(Scope::Private)
    }

    /// The strictest reach in the policy — see [`SharePolicy::strictest_scope`]. Fail-closed to
    /// [`Scope::Private`] on a poisoned lock (require a credential, never open).
    pub fn strictest_scope(&self) -> Scope {
        self.policy.read().map(|p| p.strictest_scope()).unwrap_or(Scope::Private)
    }

    /// A clone of the current policy (for the status API).
    pub fn snapshot(&self) -> SharePolicy {
        // Fail CLOSED on a poisoned lock (share-nothing), consistent with `is_shared` — never report
        // share-all from a broken lock (which `unwrap_or_default()` would, since the default is All).
        self.policy.read().map(|p| p.clone()).unwrap_or_else(|_| SharePolicy::share_nothing())
    }

    /// The file this watcher hot-reloads, if it's file-backed — for a filesystem watcher to observe.
    /// `None` for a static (`--share-models`) policy that never reloads.
    pub fn watched_path(&self) -> Option<PathBuf> {
        self.path.clone()
    }

    /// Re-read the file if its mtime changed, swapping the in-memory policy. Returns `true` only
    /// when the policy was actually swapped (so the caller can trigger an immediate re-announce).
    ///
    /// Fail-safe semantics: a **malformed** file (e.g. a partial write) keeps the previous policy
    /// *and* leaves the recorded mtime unchanged, so the next call retries once the file settles. A
    /// **removed** file keeps the previous policy but records the new (absent) mtime, so we don't
    /// retry every poll. A static watcher (no path) always returns `false`.
    pub fn reload_if_changed(&self) -> bool {
        let Some(path) = self.path.as_ref() else { return false };
        // Hold the reload lock for the whole check→load→swap→record so a concurrent caller (the
        // poll thread vs the filesystem-watcher thread) can't both see one change: whichever gets
        // here first reloads and records the new mtime; the other then short-circuits below.
        let _guard = self.reload_lock.lock().unwrap_or_else(|e| e.into_inner());
        let current = file_mtime(path);
        // Recover a poisoned mtime lock rather than panic (it only ever guards a trivial value, so
        // poison is practically unreachable — but this keeps the poll thread alive if it ever isn't).
        if *self.last_mtime.lock().unwrap_or_else(|e| e.into_inner()) == current {
            return false;
        }
        match SharePolicy::load(path) {
            Ok(policy) => {
                if let Ok(mut guard) = self.policy.write() {
                    *guard = policy;
                }
                *self.last_mtime.lock().unwrap_or_else(|e| e.into_inner()) = current;
                true
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                *self.last_mtime.lock().unwrap_or_else(|e| e.into_inner()) = current; // record absence; keep last good policy
                eprintln!(
                    "openhydra-agent: share-policy file {} removed — keeping the last policy",
                    path.display()
                );
                false
            }
            Err(e) => {
                // Record this mtime too (L1): a persistently-malformed file must not be re-read +
                // re-parsed + re-logged on every poll slice (500ms, and once per inbound request).
                // The previous policy is kept; a later *good* write bumps the mtime and is picked up
                // (safe because the desktop writes atomically — a reader never sees a partial file).
                *self.last_mtime.lock().unwrap_or_else(|e| e.into_inner()) = current;
                eprintln!(
                    "openhydra-agent: share-policy file {} unreadable ({e}) — keeping the last policy",
                    path.display()
                );
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_shares_anything() {
        let p = SharePolicy::share_all();
        assert!(p.is_shared("tinyllama:latest"));
        assert!(p.is_shared("qwen3.8:27b-q8_0"));
        // An empty ref is never a real model handle → never shared, even under All.
        assert!(!p.is_shared(""));
    }

    #[test]
    fn strictest_scope_catches_what_scope_of_misses_on_an_alias() {
        // M4 review HIGH: All mode + default Global + one Private model. `scope_of` on a non-exact
        // alias falls through to default (Global) — the bypass. `strictest_scope` folds the whole
        // policy to Private, so the provider gates the alias correctly.
        let mut p = SharePolicy::share_all(); // mode = All, default_scope = Global
        p.scopes.insert("llama3.2:latest".into(), Scope::Private);
        // The bug scope_of would have used for an alias:
        assert_eq!(p.scope_of("llama3.2"), Scope::Global, "alias misses the exact key → default");
        // The fail-closed scope the fix uses instead:
        assert_eq!(p.strictest_scope(), Scope::Private, "one private model tightens the whole policy");
        // A purely-global policy is unaffected (no regression for a public node).
        assert_eq!(SharePolicy::share_all().strictest_scope(), Scope::Global);
        // A Device model is stricter still.
        let mut d = SharePolicy::share_all();
        d.scopes.insert("x".into(), Scope::Device);
        assert_eq!(d.strictest_scope(), Scope::Device);
    }

    #[test]
    fn share_nothing_shares_nothing() {
        let p = SharePolicy::share_nothing();
        assert_eq!(p.mode, ShareMode::List);
        assert!(p.models.is_empty());
        assert!(!p.is_shared("tinyllama:latest"));
        assert!(!p.is_shared(""));
    }

    #[test]
    fn list_restricts_to_members() {
        let p = SharePolicy::share_list(["tinyllama:latest", "qwen3.8:27b-q8_0"]);
        assert!(p.is_shared("tinyllama:latest")); // listed → shared
        assert!(p.is_shared("qwen3.8:27b-q8_0")); // listed → shared (dotted family is opaque)
        assert!(!p.is_shared("qwen3-vl:30b")); // not listed → refused (announce + serve)
        assert!(!p.is_shared("")); // empty ref never matches a list entry
    }

    #[test]
    fn f5_legacy_path_entry_migrates_to_clean_handle() {
        // A policy listing a llama.cpp model by its old absolute path must match the clean handle
        // the adapter now advertises — without a re-toggle — and must not keep the leaky path.
        let p = SharePolicy::from_legacy_list([
            "/home/user/models/Qwen3.5-9B-UD-Q4_K_XL.gguf".to_string(),
            "tinyllama:latest".to_string(),
        ]);
        assert_eq!(p.mode, ShareMode::List);
        assert!(p.is_shared("Qwen3.5-9B-UD-Q4_K_XL"));
        assert!(!p.models.iter().any(|m| m.contains("/home/user")));
        assert!(p.is_shared("tinyllama:latest")); // clean id untouched
    }

    #[test]
    fn f5_load_migrates_path_keyed_policy_file() {
        // A file written by an older build (path-keyed) loads as clean handles.
        let path = std::env::temp_dir().join("oh-f5-load-migrate-test.json");
        std::fs::write(
            &path,
            r#"{"version":1,"mode":"list","models":["/home/user/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"]}"#,
        )
        .unwrap();
        let p = SharePolicy::load(&path).unwrap();
        assert!(p.is_shared("Qwen3.5-9B-UD-Q4_K_XL"));
        assert!(!p.is_shared("/home/user/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_list_shares_nothing() {
        // The state the old sentinel could not express: sharing is "on" but nothing is selected.
        let p = SharePolicy::share_list(Vec::<String>::new());
        assert_eq!(p.mode, ShareMode::List);
        assert!(!p.is_shared("tinyllama:latest"));
    }

    #[test]
    fn list_deduplicates() {
        let p = SharePolicy::share_list(["a", "a", "b"]);
        assert_eq!(p.models.len(), 2);
    }

    #[test]
    fn legacy_empty_migrates_to_all_non_empty_to_list() {
        assert_eq!(SharePolicy::from_legacy_list(Vec::<String>::new()).mode, ShareMode::All);
        let p = SharePolicy::from_legacy_list(["qwen3-coder:30b-a3b-q8_0"]);
        assert_eq!(p.mode, ShareMode::List);
        assert!(p.is_shared("qwen3-coder:30b-a3b-q8_0"));
    }

    #[test]
    fn json_round_trips_both_modes() {
        for p in [SharePolicy::share_all(), SharePolicy::share_list(["m1", "m2"])] {
            let json = serde_json::to_string(&p).unwrap();
            let back: SharePolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(p, back);
        }
    }

    #[test]
    fn json_shape_is_the_documented_contract() {
        let p = SharePolicy::share_list(["b", "a"]); // insertion order irrelevant
        let v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(v["version"], CURRENT_VERSION);
        assert_eq!(v["mode"], "list");
        // BTreeSet ⇒ sorted, deterministic array.
        assert_eq!(v["models"], serde_json::json!(["a", "b"]));
    }

    // ── scope + consent (M1) ──

    #[test]
    fn default_global_needs_the_policy_level_consent_no_grandfather() {
        // The core airtight assertion: a shared model that is Global via `default_scope` (no
        // explicit `scopes` entry) is announced ONLY with a policy-level `default_global_consent`.
        // Without it → withheld (the removed grandfather). `share_list` carries no consent.
        let mut p = SharePolicy::share_list(["a"]);
        assert_eq!(p.default_scope, Scope::Global);
        assert_eq!(p.scope_of("a"), Scope::Global, "default-Global (no explicit entry)");
        assert!(p.default_global_consent.is_none());
        assert!(!p.announce_globally("a"), "default-Global with NO policy consent → withheld");
        p.default_global_consent = Some(1);
        assert!(p.announce_globally("a"), "policy-level consent → announced");
    }

    #[test]
    fn no_grandfather_a_raw_v1_policy_does_not_announce_without_a_consent_record() {
        // Airtight rule: a pre-hardening (v1) policy deserialized WITHOUT migration has no consent
        // record → default-Global models are NOT announced. (Migration below is what preserves the
        // legacy set — this proves the grandfather is gone, not just relocated.)
        let p: SharePolicy =
            serde_json::from_str(r#"{"version":1,"mode":"list","models":["a","b"]}"#).unwrap();
        assert_eq!(p.default_scope, Scope::Global);
        assert!(p.default_global_consent.is_none());
        assert!(!p.announce_globally("a"), "no consent record → not announced");
    }

    #[test]
    fn migration_preserves_the_legacy_global_announce_set_with_a_consent_record() {
        // The no-silent-un-share guarantee: a pre-3 default-Global policy that WAS announcing keeps
        // announcing exactly the same models after load()-time migration, now with a record.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::write(&path, r#"{"version":1,"mode":"list","models":["a","b"]}"#).unwrap();
        let p = SharePolicy::load(&path).unwrap();
        assert_eq!(p.version, CURRENT_VERSION, "migrated to current version");
        assert!(p.default_global_consent.is_some(), "consent record materialised");
        assert!(p.announce_globally("a") && p.announce_globally("b"), "same set still announced");
        assert!(!p.announce_globally("c"), "not shared → still not announced");
    }

    #[test]
    fn migration_of_share_nothing_does_not_pre_consent_a_future_share() {
        // A pre-3 policy that shared NOTHING must NOT get a consent record — otherwise a model the
        // user shares later would auto-announce globally without a consent moment.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::write(&path, r#"{"version":1,"mode":"list","models":[]}"#).unwrap();
        let mut p = SharePolicy::load(&path).unwrap();
        assert!(p.default_global_consent.is_none(), "share-nothing → no materialised consent");
        // Sharing a model now (still default-Global) is NOT announced until consented.
        p.models.insert("a".into());
        assert!(!p.announce_globally("a"));
    }

    #[test]
    fn migration_preserves_an_explicit_global_scope_entry_from_a_pre3_file() {
        // Regression (adversarial review Finding 1): a pre-3 file that announced a model via an
        // EXPLICIT `scopes:{m:"global"}` entry (no consent map — that build predated it) was
        // announcing `m` under the old grandfather. `announce_globally` routes an explicit-scope
        // model through the per-model consent branch, which `migrate_consent` used to leave empty →
        // the model silently un-shared on upgrade. Migration must materialise the per-model consent.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");

        // (a) explicit Global under a Private default — only the per-model path can save it.
        std::fs::write(
            &path,
            r#"{"version":2,"mode":"list","models":["a","b"],"default_scope":"private","scopes":{"a":"global"}}"#,
        )
        .unwrap();
        let p = SharePolicy::load(&path).unwrap();
        assert_eq!(p.version, CURRENT_VERSION);
        assert!(p.announce_globally("a"), "explicit-Global model must stay announced after migration");
        assert!(p.global_consent.contains_key("a"), "per-model consent materialised for it");
        assert!(!p.announce_globally("b"), "a Private-default model stays off the global net");

        // (b) explicit Global with the default *also* Global (field absent → defaults Global): the
        // explicit entry must still get a per-model record, not rely on default_global_consent.
        std::fs::write(
            &path,
            r#"{"version":1,"mode":"list","models":["a"],"scopes":{"a":"global"}}"#,
        )
        .unwrap();
        let p = SharePolicy::load(&path).unwrap();
        assert!(p.announce_globally("a"), "explicit-Global model announced after migration");
        assert!(p.global_consent.contains_key("a"), "per-model consent materialised");

        // A model with an explicit Global scope but NOT shared (list mode, not in `models`) was not
        // announced before, so migration must NOT pre-consent it.
        std::fs::write(
            &path,
            r#"{"version":2,"mode":"list","models":[],"default_scope":"private","scopes":{"z":"global"}}"#,
        )
        .unwrap();
        let p = SharePolicy::load(&path).unwrap();
        assert!(!p.global_consent.contains_key("z"), "unshared explicit-Global not pre-consented");
        assert!(!p.announce_globally("z"));
    }

    #[test]
    fn migration_is_idempotent_and_leaves_a_hardened_policy_untouched() {
        // A v3 policy is not re-migrated (no spurious consent), and re-migrating is a no-op.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::write(&path, r#"{"version":3,"mode":"list","models":["a"],"default_scope":"private"}"#).unwrap();
        let p = SharePolicy::load(&path).unwrap();
        assert_eq!(p.version, 3);
        assert!(p.default_global_consent.is_none(), "hardened policy not force-consented");
        assert!(!p.announce_globally("a"), "private default → not announced");
    }

    #[test]
    fn migration_consent_timestamps_are_deterministic_across_repeated_loads() {
        // Finding 3: `load()` migrates in memory but never persists, so a headless node re-migrates a
        // still-pre-3 file on every start/hot-reload. The materialised consent records must be
        // DETERMINISTIC (a fixed sentinel), not a fresh wall-clock each time, or the audit trail
        // churns on every restart. This file triggers BOTH paths: default-Global (mode All) and an
        // explicit `scopes` entry.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::write(&path, r#"{"version":1,"mode":"all","models":[],"scopes":{"x":"global"}}"#).unwrap();
        let p1 = SharePolicy::load(&path).unwrap();
        let p2 = SharePolicy::load(&path).unwrap();
        assert_eq!(p1, p2, "re-migrating the same still-pre-3 file is byte-identical");
        // The sentinel, not a churning wall-clock — and it stayed announcing (Finding 1 preserved).
        assert_eq!(p1.default_global_consent, Some(MIGRATION_CONSENT_TS));
        assert_eq!(p1.global_consent.get("x"), Some(&MIGRATION_CONSENT_TS));
        assert!(p1.announce_globally("x"));
        // An explicit CLI/config opt-in still carries a real wall-clock (only migration is sentinelled).
        assert!(SharePolicy::share_all().default_global_consent.unwrap() > MIGRATION_CONSENT_TS);
    }

    #[test]
    fn unknown_scope_string_degrades_safely_instead_of_failing_the_whole_policy() {
        // A typo'd / future scope value must not fail the entire policy closed. It parses to
        // `Scope::Unknown` and is treated as not-globally-announced (safe).
        let p: SharePolicy = serde_json::from_str(
            r#"{"version":3,"mode":"list","models":["a","b"],"scopes":{"a":"lan","b":"global"},
                "global_consent":{"b":5}}"#,
        )
        .expect("one bad scope value must not fail the whole policy");
        assert_eq!(p.scope_of("a"), Scope::Unknown);
        assert!(!p.announce_globally("a"), "Unknown scope → never announced");
        assert!(p.is_shared("a"), "still shared (scope only affects reach)");
        assert!(p.announce_globally("b"), "the valid consented-global entry is unaffected");
    }

    #[test]
    fn private_default_keeps_unset_models_off_the_global_net() {
        // Once the desktop writes default_scope=private, a shared-but-unset model is Private → not
        // globally announced (privacy-first for new/unset models), but still shared (served).
        let mut p = SharePolicy::share_list(["a"]);
        p.default_scope = Scope::Private;
        assert!(p.is_shared("a"));
        assert_eq!(p.scope_of("a"), Scope::Private);
        assert!(!p.announce_globally("a"));
    }

    #[test]
    fn explicit_global_requires_consent_fail_closed() {
        let mut p = SharePolicy::share_list(["a"]);
        p.default_scope = Scope::Private;
        p.scopes.insert("a".into(), Scope::Global); // hand-set Global, no consent yet
        assert_eq!(p.scope_of("a"), Scope::Global);
        assert!(!p.announce_globally("a"), "explicit Global without consent must not announce");

        p.global_consent.insert("a".into(), 1_725_000_000_000);
        assert!(p.announce_globally("a"), "consent recorded → announces");
    }

    #[test]
    fn device_and_private_scopes_never_announce_globally() {
        let mut p = SharePolicy::share_list(["dev", "priv"]);
        p.default_scope = Scope::Private;
        p.scopes.insert("dev".into(), Scope::Device);
        p.scopes.insert("priv".into(), Scope::Private);
        // Even a stray consent entry can't promote a non-Global scope.
        p.global_consent.insert("dev".into(), 1);
        p.global_consent.insert("priv".into(), 1);
        assert!(!p.announce_globally("dev"));
        assert!(!p.announce_globally("priv"));
    }

    #[test]
    fn scope_and_consent_survive_json_round_trip() {
        let mut p = SharePolicy::share_list(["a", "b"]);
        p.default_scope = Scope::Private;
        p.scopes.insert("a".into(), Scope::Global);
        p.global_consent.insert("a".into(), 42);
        let back: SharePolicy = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(p, back);
        assert!(back.announce_globally("a"));
        assert!(!back.announce_globally("b"));
    }

    #[test]
    fn load_migrates_path_keyed_scope_and_consent_entries() {
        // A hand-edited/older file could key scope/consent by a path; load() must fold them to the
        // clean handle so they keep matching the advertised model.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::write(
            &path,
            r#"{"version":2,"mode":"list","models":["/home/u/models/Foo-7B-Q4_K_M.gguf"],
                "default_scope":"private",
                "scopes":{"/home/u/models/Foo-7B-Q4_K_M.gguf":"global"},
                "global_consent":{"/home/u/models/Foo-7B-Q4_K_M.gguf":7}}"#,
        )
        .unwrap();
        let p = SharePolicy::load(&path).unwrap();
        assert!(p.is_shared("Foo-7B-Q4_K_M"));
        assert_eq!(p.scope_of("Foo-7B-Q4_K_M"), Scope::Global);
        assert!(p.announce_globally("Foo-7B-Q4_K_M"), "consent migrated with the key");
        assert!(!p.scopes.keys().any(|k| k.contains("/home/u")));
    }

    #[test]
    fn version_defaults_when_absent() {
        // A file written before versioning (or hand-edited) still parses.
        let p: SharePolicy = serde_json::from_str(r#"{"mode":"all"}"#).unwrap();
        assert_eq!(p.version, 1);
        assert_eq!(p.mode, ShareMode::All);
        assert!(p.models.is_empty());
    }

    #[test]
    fn malformed_json_is_invalid_data() {
        // Not a NotFound — distinguishable so the caller can keep the previous policy on a bad edit.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::write(&path, b"{ this is not json").unwrap();
        let err = SharePolicy::load(&path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn write_atomic_round_trips_and_leaves_no_temp() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("share-policy.json"); // parent auto-created
        let p = SharePolicy::share_list(["qwen3.8:27b-q8_0"]);
        p.write_atomic(&path).unwrap();
        assert_eq!(SharePolicy::load(&path).unwrap(), p);
        assert!(!path.with_extension("json.tmp").exists(), "temp file should be renamed away");
    }

    // ── PolicyWatcher (hot-reload state machine) ──
    // A short sleep between writes guarantees a distinct mtime on the nanosecond-resolution
    // filesystems this ships on (APFS / ext4 / NTFS); the watcher keys change-detection on mtime.

    fn tick() {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    #[test]
    fn static_watcher_never_reloads() {
        let w = PolicyWatcher::from_legacy_list(["a"]);
        assert!(w.is_shared("a"));
        assert!(!w.is_shared("b"));
        assert!(!w.reload_if_changed()); // no path → never reloads
    }

    #[test]
    fn from_file_missing_falls_back_to_share_all() {
        let dir = tempfile::tempdir().unwrap();
        let w = PolicyWatcher::from_file(dir.path().join("absent.json"));
        assert!(w.is_shared("anything:at-all")); // absent (first run) → share-all
    }

    #[test]
    fn from_file_corrupt_fails_closed_to_share_nothing() {
        // M1: a *corrupt* (not merely missing) policy must NOT widen a restrictive config to
        // share-all. It fails closed to share-nothing until the file is fixed/healed.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::write(&path, b"{ this is not valid json").unwrap();
        let w = PolicyWatcher::from_file(path);
        assert!(!w.is_shared("tinyllama:latest"));
        assert!(!w.is_shared("anything:at-all"));
    }

    #[test]
    fn reload_swaps_policy_when_the_file_changes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        SharePolicy::share_list(["a"]).write_atomic(&path).unwrap();
        let w = PolicyWatcher::from_file(path.clone());
        assert!(w.is_shared("a") && !w.is_shared("b"));

        tick();
        SharePolicy::share_list(["b"]).write_atomic(&path).unwrap();
        assert!(w.reload_if_changed(), "changed file should swap");
        assert!(!w.is_shared("a") && w.is_shared("b"), "policy applied");

        assert!(!w.reload_if_changed(), "no further change → no swap");
    }

    #[test]
    fn reload_keeps_previous_on_malformed_then_recovers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        SharePolicy::share_list(["a"]).write_atomic(&path).unwrap();
        let w = PolicyWatcher::from_file(path.clone());

        tick();
        std::fs::write(&path, b"{ half-written not json").unwrap(); // a partial/garbage write
        assert!(!w.reload_if_changed(), "malformed → no swap");
        assert!(w.is_shared("a"), "previous policy retained on a bad write");
        // L1: a second call on the SAME malformed file short-circuits (mtime recorded) → no retry.
        assert!(!w.reload_if_changed(), "unchanged malformed file → not re-processed");
        assert!(w.is_shared("a"));

        // A later *good* write bumps the mtime and IS picked up (recovery after the bad write).
        tick();
        SharePolicy::share_list(["b"]).write_atomic(&path).unwrap();
        assert!(w.reload_if_changed(), "recovers after the file settles");
        assert!(w.is_shared("b") && !w.is_shared("a"));
    }

    #[test]
    fn reload_keeps_previous_when_file_removed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        SharePolicy::share_list(["a"]).write_atomic(&path).unwrap();
        let w = PolicyWatcher::from_file(path.clone());

        tick();
        std::fs::remove_file(&path).unwrap();
        assert!(!w.reload_if_changed(), "removal → no swap");
        assert!(w.is_shared("a"), "previous policy retained when the file vanishes");
        // Absence recorded → we don't thrash re-reading a missing file every poll.
        assert!(!w.reload_if_changed());
    }

    #[test]
    fn concurrent_reload_only_one_observes_the_change() {
        // R6: the poll thread and the filesystem-watcher thread can both call reload_if_changed for
        // the same change; the reload_lock + mtime dedup must let EXACTLY ONE observe it (no
        // double-announce), and the policy still lands.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        SharePolicy::share_list(["a"]).write_atomic(&path).unwrap();
        let w = std::sync::Arc::new(PolicyWatcher::from_file(path.clone()));
        tick();
        SharePolicy::share_list(["b"]).write_atomic(&path).unwrap();
        let (w1, w2) = (w.clone(), w.clone());
        let h1 = std::thread::spawn(move || w1.reload_if_changed());
        let h2 = std::thread::spawn(move || w2.reload_if_changed());
        let (r1, r2) = (h1.join().unwrap(), h2.join().unwrap());
        assert!(r1 ^ r2, "exactly one racing reload should observe the change (got {r1}, {r2})");
        assert!(w.is_shared("b") && !w.is_shared("a"), "policy applied after the race");
    }
}
