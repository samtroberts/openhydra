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

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};
use std::time::SystemTime;

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

fn default_version() -> u32 {
    1
}

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
}

impl Default for SharePolicy {
    /// The headless default when nothing is configured: share everything (preserves the historical
    /// `--share-models`-absent behavior).
    fn default() -> Self {
        Self::share_all()
    }
}

impl SharePolicy {
    /// Share every detected model (now and future).
    pub fn share_all() -> Self {
        Self { version: default_version(), mode: ShareMode::All, models: BTreeSet::new() }
    }

    /// Share **no** models (mode `list`, empty). The fail-closed state: what a corrupt policy or a
    /// poisoned lock resolves to, so a sharing control never silently *widens* to share-all.
    pub fn share_nothing() -> Self {
        Self { version: default_version(), mode: ShareMode::List, models: BTreeSet::new() }
    }

    /// Share exactly the given engine handles. Duplicates collapse; order is irrelevant.
    pub fn share_list<I, S>(models: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            version: default_version(),
            mode: ShareMode::List,
            models: models.into_iter().map(Into::into).collect(),
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
        let set: BTreeSet<String> = models.into_iter().map(Into::into).collect();
        if set.is_empty() {
            Self::share_all()
        } else {
            Self { version: default_version(), mode: ShareMode::List, models: set }
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
        serde_json::from_str(&raw).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
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
}

impl PolicyWatcher {
    /// A static (non-reloading) watcher around a fixed policy — for tests and non-file callers.
    pub fn r#static(policy: SharePolicy) -> Self {
        Self { policy: RwLock::new(policy), path: None, last_mtime: Mutex::new(None) }
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
        Self { policy: RwLock::new(policy), path: Some(path), last_mtime: Mutex::new(mtime) }
    }

    /// Whether `engine_ref` is shared under the current (possibly hot-reloaded) policy. A poisoned
    /// lock falls back to share-nothing (fail-closed — matters for the serve gate).
    pub fn is_shared(&self, engine_ref: &str) -> bool {
        self.policy.read().map(|p| p.is_shared(engine_ref)).unwrap_or(false)
    }

    /// A clone of the current policy (for the status API).
    pub fn snapshot(&self) -> SharePolicy {
        // Fail CLOSED on a poisoned lock (share-nothing), consistent with `is_shared` — never report
        // share-all from a broken lock (which `unwrap_or_default()` would, since the default is All).
        self.policy.read().map(|p| p.clone()).unwrap_or_else(|_| SharePolicy::share_nothing())
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
        assert_eq!(v["version"], 1);
        assert_eq!(v["mode"], "list");
        // BTreeSet ⇒ sorted, deterministic array.
        assert_eq!(v["models"], serde_json::json!(["a", "b"]));
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
}
