// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! OpenHydra Desktop — a Tauri v2 shell around `openhydra-agent`.
//!
//! Two roles, two supervised child processes of the CLI binary (the validated dual-role
//! pattern: separate identities + listen ports per role):
//! * **provider** — `provide --engine-kind auto`: share this machine's local engines.
//! * **gateway** — `serve`: a local OpenAI-compatible endpoint backed by the network.
//!
//! The GUI never runs the swarm in-process — the CLI stays the single source of truth,
//! a crash is isolated to the child, and stop is just a kill. Status comes from parsing
//! the agent's own log lines; the engines panel calls the agent crate's
//! [`detect_engines`](openhydra_agent::detect::detect_engines) directly (pure local HTTP
//! probes, no swarm needed).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::{HashMap, VecDeque};
use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{Emitter, Manager};
use tauri_plugin_deep_link::DeepLinkExt;
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

mod cli;
mod hostinfo;
mod connectors;
mod installer;

// ── Child-pid registry (crash/kill safety) ──
//
// `RunEvent::Exit` covers a normal quit, but NOT a SIGTERM/SIGINT to the app (observed
// live: killing the app orphaned both agents, which kept the ports and served
// invisibly). Two layers close that:
// * a **signal handler** kills the registered pids before the app dies — the registry is
//   plain atomics because only `kill(2)` is async-signal-safe, not mutexes;
// * a **pidfile sweep at next launch** reaps agents that survived even SIGKILL (where no
//   handler can run), matching on process name so a reused pid is never killed.

static PROVIDER_PID: AtomicU32 = AtomicU32::new(0);
static GATEWAY_PID: AtomicU32 = AtomicU32::new(0);

fn pidfile() -> PathBuf {
    openhydra_dir().join("desktop-agents.pid")
}

/// Record a role's child pid (0 = none) in its atomic slot + the on-disk pidfile.
fn registry_set(slot: &AtomicU32, pid: u32) {
    slot.store(pid, Ordering::SeqCst);
    let (p, g) = (PROVIDER_PID.load(Ordering::SeqCst), GATEWAY_PID.load(Ordering::SeqCst));
    if p == 0 && g == 0 {
        let _ = std::fs::remove_file(pidfile());
    } else {
        let _ = std::fs::create_dir_all(openhydra_dir());
        let _ = std::fs::write(pidfile(), format!("{p}\n{g}\n"));
    }
}

/// At launch, kill agents a previous app instance left behind (its pidfile survived a
/// SIGKILL/crash). Only pids whose command is actually `openhydra-agent` are touched —
/// never a reused pid, and never agents the operator runs by hand.
fn sweep_stale_agents() {
    let Ok(content) = std::fs::read_to_string(pidfile()) else { return };
    for pid in content.lines().filter_map(|l| l.trim().parse::<u32>().ok()) {
        if pid == 0 {
            continue;
        }
        let is_agent = Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "comm="])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).contains("openhydra-agent"))
            .unwrap_or(false);
        if is_agent {
            eprintln!("openhydra-desktop: reaping stale agent pid {pid} from a previous run");
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGKILL);
            }
        }
    }
    let _ = std::fs::remove_file(pidfile());
}

/// Kill the registered children and re-raise. Only async-signal-safe calls in here.
#[cfg(unix)]
extern "C" fn on_signal(sig: libc::c_int) {
    for slot in [&PROVIDER_PID, &GATEWAY_PID] {
        let pid = slot.load(Ordering::SeqCst);
        if pid != 0 {
            unsafe { libc::kill(pid as i32, libc::SIGKILL) };
        }
    }
    unsafe {
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

#[cfg(unix)]
fn install_signal_handlers() {
    unsafe {
        for sig in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
            libc::signal(sig, on_signal as libc::sighandler_t);
        }
    }
}

#[cfg(not(unix))]
fn install_signal_handlers() {}

/// Cap on retained log lines per role (ring buffer).
const LOG_CAP: usize = 500;
/// Listen ports for the two roles (distinct identities + ports, like the live tests).
const PROVIDER_PORT: u16 = 4111;
const GATEWAY_PORT: u16 = 4112;
/// Loopback ports for each role's `--status-bind` introspection endpoint (P0/P1).
const PROVIDER_STATUS_PORT: u16 = 9464;
const GATEWAY_STATUS_PORT: u16 = 9465;

/// A per-launch random bearer token for the status endpoints, so another local user or
/// process can't read this node's peer/DHT data. Derived once at process start from the
/// pid + a monotonic-ish counter (Math.random-free; the runtime forbids system-time RNG
/// helpers but std's `RandomState` gives us entropy for a hash seed).
fn status_token() -> &'static str {
    use std::sync::OnceLock;
    static TOKEN: OnceLock<String> = OnceLock::new();
    TOKEN.get_or_init(|| {
        use std::hash::{BuildHasher, Hasher};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u32(std::process::id());
        h.write_usize(&TOKEN as *const _ as usize);
        format!("{:016x}{:016x}", h.finish(), {
            let mut h2 = std::collections::hash_map::RandomState::new().build_hasher();
            h2.write_u64(h.finish());
            h2.finish()
        })
    })
}

/// The local gateway key (`~/.openhydra/gateway.key`, owner-only, created on first use): the
/// desktop's gateway requires it on `/v1/*`, the app's own calls present it, and Connect / `launch`
/// write it into the coding tools' configs. `None` only if the file can't be written; the gateway then
/// runs without one (open to local programs, refusing web pages) and says so.
fn gateway_api_key() -> Option<&'static str> {
    use std::sync::OnceLock;
    static KEY: OnceLock<Option<String>> = OnceLock::new();
    KEY.get_or_init(|| match openhydra_agent::gateway_key::load_or_create(&openhydra_dir().join("gateway.key")) {
        Ok(k) => Some(k),
        Err(e) => {
            eprintln!("openhydra: could not create the local gateway key ({e}); the gateway runs without one");
            None
        }
    })
    .as_deref()
}

/// An HTTP client for this app's own gateway (presents the gateway key). A streamed chat may run as
/// long as the gateway lets it (up to 6 h for a long `max_tokens`), not the client's 10-minute
/// default, which cut long replies off (ADVERSARIAL item 6).
fn gateway_client() -> Result<openhydra_agent::ReqwestClient, String> {
    match gateway_api_key() {
        Some(k) => openhydra_agent::ReqwestClient::with_bearer(k),
        None => openhydra_agent::ReqwestClient::new(),
    }
    .map(|c| c.with_stream_timeout(std::time::Duration::from_secs(6 * 60 * 60 + 10 * 60)))
    .map_err(|e| e.to_string())
}

// ── Settings (persisted at ~/.openhydra/desktop.json) ──

/// Whether Settings shows the "Act as a relay for other peers" switch (SL14: not until community
/// relays ship, plan X14). While it's hidden the saved `peer_relay` is ignored.
const PEER_RELAY_SWITCH_SHOWN: bool = false;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    /// Extra bootstrap multiaddrs (`/ip4/…/tcp/4001/p2p/<peer-id>`), one per entry, used on top of
    /// the release build's built-in ones (see [`builtin_bootstraps`]). Empty is valid on a LAN
    /// (mDNS discovers peers); joining the public network needs at least one, built in or here.
    bootstraps: Vec<String>,
    /// Local port the OpenAI-compatible gateway binds on 127.0.0.1.
    gateway_port: u16,
    /// Pass `--engine-autostart` to the provider (start LM Studio / Ollama if down).
    engine_autostart: bool,
    /// Pass `--prompt-scoring auto`: let gateways run the teacher-forced log-prob identity check on
    /// this provider's engines (the route to the A2 authenticity tier): vLLM / SGLang, llama.cpp /
    /// mistral.rs builds with the OpenHydra scoring endpoint, and Ollama / LM Studio models through
    /// the bundled side-scorer. Opt-in — the checks spend engine time (at most 2 at once).
    authenticity_checks: bool,
    /// Let the gateway's audits replay a greedy request to a public model to a second provider
    /// (`--audit-replay on`): the strongest check, but the prompt reaches that provider too. Off by
    /// default: audits then send generated test prompts instead.
    audit_replay: bool,
    /// Let the gateway download a model's tokenizer from Hugging Face the first time it serves a model
    /// the signed tokenizer index lists, to check billed token counts (W4), on the public network. On
    /// by default; off passes `--no-tokenizer-downloads` (only tokenizers already downloaded are
    /// used). The download tells Hugging Face which models this gateway uses, never what it sends.
    tokenizer_downloads: bool,
    /// The same, in a private deployment (`public_network` off): off by default there, so a private
    /// deployment contacts no outside service unless the user turns this on (review 2026-10-14, D1).
    /// See [`tokenizer_downloads_on`].
    private_tokenizer_downloads: bool,
    /// Join through OpenHydra's public network: the built-in bootstraps and the production relays.
    /// Off = a private deployment: only the user's own bootstraps (and the LAN, never OpenHydra's
    /// own, even listed), with `--public-relays off` and `--tokenizer-index-url off`, and no
    /// automatic update check. On by default.
    public_network: bool,
    /// Only use providers this gateway has verified at least to this tier: `"off"` (any provider,
    /// the default), `"A1"` or `"A2"` (`--min-authenticity <tier> --assurance gate`).
    min_authenticity: String,
    /// Optional SearXNG-compatible search endpoint (e.g. `http://127.0.0.1:8888`) powering
    /// the chat "web" toggle. Empty → the toggle is hidden. The operator brings their own
    /// search backend, same BYO philosophy as engines.
    search_url: String,
    /// #4: verbose agent logging. Off by default (quiet `RUST_LOG=warn`) so the log ring stays
    /// light; on raises the agent to info/debug for troubleshooting. The essential status lines
    /// are `eprintln!` (not `RUST_LOG`-gated), so quiet mode never breaks the status views.
    verbose_logs: bool,
    /// #9: user-editable device name, shown in the app and offered as this device's label when it
    /// joins a swarm. Empty → derive from the OS.
    device_name: String,
    /// Act as a circuit relay for peers behind strict NATs (`--peer-relay`). Off by default, like
    /// the agent; it only takes effect when this node is publicly reachable.
    peer_relay: bool,
    /// Which local engine the provider serves (`--engine-kind`): `auto` (detect every running
    /// engine) or one of [`ENGINE_KINDS`].
    engine_kind: String,
    /// The engine's base URL (`--engine`), for a specific `engine_kind`. Empty → that engine's
    /// standard port.
    engine_endpoint: String,
    /// Legacy per-model share allowlist. Superseded by `~/.openhydra/share-policy.json` (the
    /// hot-reloaded source of truth — see `save_share_policy`); kept as a one-release **mirror** so a
    /// downgrade still reads it. Migrated into the policy file on first run (empty → share-all,
    /// non-empty → that list). Not read by the running provider anymore.
    #[serde(default)]
    shared_models: Vec<String>,
    /// Persisted sharing INTENT — was the provider role running when we last toggled it? The provider
    /// is a child process killed on quit/update (`kill_all` on exit), so without this every restart
    /// silently stopped sharing. Set by start/stop_provider; read on launch to resume.
    sharing_enabled: bool,
    /// Resume sharing automatically on launch when `sharing_enabled` is set (informed opt-out — the UI
    /// can clear it). Default on.
    resume_on_launch: bool,
    /// Set once this install has written its own share-policy file. From then on a missing policy
    /// file means "share nothing" (fail closed), never a migration from the legacy list, which
    /// would share every model globally. Managed by the backend; `save_settings` preserves it.
    share_policy_initialized: bool,
    /// The user has confirmed what they share in the Share dialog (or was already sharing when they
    /// updated). Until then, starting to share opens the dialog, even if an older build left a
    /// "share everything" policy behind. Managed by `confirm_sharing`; `save_settings` preserves it.
    sharing_confirmed: bool,
    /// Persisted-settings schema version, for future migrations (see `load_settings`). Files written
    /// before versioning (field absent) parse as 1; fresh installs + re-saves write the current one.
    #[serde(default = "default_schema_version")]
    schema_version: u32,
}

/// Current settings schema version. Bump when a field's meaning/shape changes and add a migration arm.
const SCHEMA_VERSION: u32 = 2;
fn default_schema_version() -> u32 {
    1
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            bootstraps: Vec::new(),
            // #2: default off 8080 — that's llama.cpp's default, so a user running llama.cpp
            // locally would collide with the OpenHydra gateway. 16527 avoids the common engine
            // ports (llama.cpp 8080, vLLM/OpenAI 8000, LM Studio 1234, Ollama 11434, Exo 52415,
            // ComfyUI 8188). Still user-overridable in Settings.
            gateway_port: 16527,
            engine_autostart: true,
            authenticity_checks: false,
            audit_replay: false,
            tokenizer_downloads: true,
            private_tokenizer_downloads: false,
            public_network: true,
            min_authenticity: "off".into(),
            search_url: String::new(),
            verbose_logs: false,
            device_name: String::new(),
            peer_relay: false,
            engine_kind: "auto".into(),
            engine_endpoint: String::new(),
            shared_models: Vec::new(), // empty ⇒ share every detected model (the default)
            sharing_enabled: false,
            resume_on_launch: true,
            share_policy_initialized: false,
            sharing_confirmed: false,
            schema_version: SCHEMA_VERSION,
        }
    }
}

fn openhydra_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".openhydra")
}

fn settings_path() -> PathBuf {
    openhydra_dir().join("desktop.json")
}

fn load_settings() -> Settings {
    let path = settings_path();
    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(_) => return Settings::default(), // no file yet — fresh install
    };
    // Container `#[serde(default)]` fills any missing field, so adding fields across an update is
    // safe. A hard parse failure (e.g. a field's TYPE changed) must NOT silently wipe the user's
    // settings — preserve the original beside the file, then default, so nothing is lost.
    let mut s: Settings = match serde_json::from_str(&raw) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("openhydra: {} failed to parse ({e}); backing it up → .corrupt.bak, using defaults", path.display());
            let _ = std::fs::write(path.with_extension("json.corrupt.bak"), &raw);
            Settings::default()
        }
    };
    let mut migrated = false;
    // C5 migration: the gateway port was never user-editable before v0.3.8, so a persisted 8080 (the
    // pre-16527 default that collided with llama.cpp) is stale — bump it. Fresh installs use 16527.
    if s.gateway_port == 8080 {
        s.gateway_port = 16527;
        migrated = true;
    }
    if s.schema_version < SCHEMA_VERSION {
        s.schema_version = SCHEMA_VERSION;
        migrated = true;
    }
    if migrated {
        let _ = store_settings(&s);
    }
    s
}

fn store_settings(s: &Settings) -> Result<(), String> {
    let dir = openhydra_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    // Atomic (temp + rename) like the sessions/policy files: a crash mid-write can never truncate
    // desktop.json — the reader always sees the previous or the new complete file, never a partial.
    let json = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    write_atomic(&settings_path(), &json)
}

// ── #1: chat sessions persisted to disk (WebView localStorage isn't durable across restarts
// on any platform). The UI owns the JSON shape; we just read/write the blob to a file. ──
fn sessions_path() -> PathBuf {
    openhydra_dir().join("sessions.json")
}

/// Read the persisted sessions blob (raw JSON string the UI wrote), or "" if none yet.
#[tauri::command]
fn load_sessions() -> String {
    std::fs::read_to_string(sessions_path()).unwrap_or_default()
}

/// Persist the UI's sessions blob (verbatim JSON string). #1: written **atomically** (temp +
/// rename) so a crash mid-write can never truncate/corrupt the chat-history file — the reader
/// always sees either the previous complete file or the new complete one, never a partial write.
#[tauri::command]
fn save_sessions(data: String) -> Result<(), String> {
    let dir = openhydra_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_atomic(&sessions_path(), &data)
}

/// Write `data` to `path` atomically: to a sibling temp file, then rename over `path`. Rename is
/// atomic on the same filesystem, so `path` is never observed half-written.
fn write_atomic(path: &std::path::Path, data: &str) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

// ── Share policy (~/.openhydra/share-policy.json) — the single source of truth for which models
// the provider shares. The agent hot-reloads this file (see `openhydra_agent::SharePolicy`); the
// desktop writes it on toggle so a change applies to the running provider with no restart. ──

fn share_policy_path() -> PathBuf {
    openhydra_dir().join("share-policy.json")
}

/// Ensure the share-policy file exists, migrating from the legacy `settings.shared_models` on first
/// run after upgrade: an empty list historically meant "share everything" (mode `all`); a non-empty
/// list becomes an explicit `list`. Existing behavior is preserved across the upgrade. Best-effort —
/// if the write fails, the agent falls back to share-all with a warning rather than sharing nothing.
/// Ensure a **valid** share-policy file exists before the provider reads it:
/// - absent on a **fresh install** (no `desktop.json` either) → share nothing: the user picks models
///   and gives consent in the Share dialog, so no model is announced without being asked;
/// - absent after an **upgrade** (`desktop.json` exists) → migrate from the legacy list;
/// - present-but-**corrupt** → self-heal: back the bad file up to `.corrupt.bak`, regenerate a safe
///   default (share-nothing — fail-closed, never over-shares), and raise `reset_flag` so the UI can
///   tell the user their selection was reset;
/// - present-and-valid → untouched.
///
/// Mirrors the corruption handling `load_settings` already does for `desktop.json`.
fn ensure_valid_share_policy(
    settings: &Settings,
    reset_flag: &std::sync::atomic::AtomicBool,
) -> PathBuf {
    let path = share_policy_path();
    // Start from "share nothing" on a fresh install (no desktop.json yet), whenever this install has
    // had its own policy before (a missing file then is a deletion, not an old-version upgrade), and
    // for an upgrade from a user who wasn't sharing: an empty legacy list meant "everything", which
    // they never chose (ADVERSARIAL item 3). A user who was sharing keeps what they shared.
    let start_empty = policy_starts_empty(settings_path().exists(), settings);
    ensure_valid_share_policy_at(&path, &settings.shared_models, start_empty, reset_flag);
    path
}

/// Whether a missing share-policy file starts as share-nothing (rather than migrating the legacy
/// list): a fresh install, an install that has had its own policy before, or an upgrade from a user
/// who wasn't sharing.
fn policy_starts_empty(settings_file_exists: bool, settings: &Settings) -> bool {
    !settings_file_exists
        || settings.share_policy_initialized
        || (!settings.sharing_enabled && !settings.sharing_confirmed)
}

/// Path-parametrized core of [`ensure_valid_share_policy`] (testable without `~/.openhydra`): sets
/// `reset_flag` true iff a corrupt file was healed.
fn ensure_valid_share_policy_at(
    path: &std::path::Path,
    legacy: &[String],
    fresh_install: bool,
    reset_flag: &std::sync::atomic::AtomicBool,
) {
    if !path.exists() {
        if fresh_install {
            if let Err(e) = openhydra_agent::SharePolicy::share_nothing().write_atomic(path) {
                eprintln!("openhydra: could not write initial share policy to {}: {e}", path.display());
            }
        } else {
            migrate_share_policy_if_absent(path, legacy);
        }
        return;
    }
    match openhydra_agent::SharePolicy::load(path) {
        Ok(_) => {} // valid → untouched
        // Genuinely CORRUPT (JSON parse error) → self-heal + notify, but only flag the reset if the
        // heal actually landed (a failed write must not toast-storm on every poll).
        Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
            if heal_corrupt_share_policy(path).is_ok() {
                reset_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        // A *transient* IO error (permissions, a file briefly locked by AV/backup/indexer, or a
        // just-deleted file racing `exists()`): the file may be a perfectly valid policy we simply
        // can't read this instant. Do NOT overwrite it or raise the reset flag — leave it, and let
        // the agent's `from_file` / `read_share_policy` fail closed in memory. (Closes review F1/F2/F4.)
        Err(e) => eprintln!(
            "openhydra: share-policy file {} temporarily unreadable ({e}) — leaving it untouched",
            path.display()
        ),
    }
}

/// Back a corrupt policy file up to `.corrupt.bak` (best-effort, for debugging) and overwrite it with
/// a safe **share-nothing** default. A sharing control never widens on corruption.
/// Back a **corrupt** policy file up to `.corrupt.bak` and overwrite it with a safe share-nothing
/// default. Returns `Err` if the safe write fails (so the caller can avoid falsely flagging a reset).
fn heal_corrupt_share_policy(path: &std::path::Path) -> Result<(), String> {
    if let Ok(raw) = std::fs::read_to_string(path) {
        let _ = std::fs::write(path.with_extension("json.corrupt.bak"), &raw);
    }
    openhydra_agent::SharePolicy::share_nothing()
        .write_atomic(path)
        .map_err(|e| e.to_string())?;
    eprintln!(
        "openhydra: share-policy file {} was corrupt → backed up to .corrupt.bak and reset to share-nothing",
        path.display()
    );
    Ok(())
}

/// An inherited "share everything" from an older build, for a user who isn't sharing and never chose
/// it here: v0.3.20 wrote mode `all` (with consent) as soon as its Share view was opened, so starting
/// to share — or adding one model — would have published every model to everyone (ADVERSARIAL item
/// 3). Reset to share-nothing; the user then picks models and reach in the Share dialog. Returns
/// whether it reset. Run once at startup, before anything reads the policy.
fn drop_inherited_share_all(path: &std::path::Path) -> bool {
    match openhydra_agent::SharePolicy::load(path) {
        Ok(p) if p.mode == openhydra_agent::share_policy::ShareMode::All => {
            match openhydra_agent::SharePolicy::share_nothing().write_atomic(path) {
                Ok(()) => {
                    eprintln!("openhydra: reset an inherited share-everything policy (never chosen in this version, not sharing) to share-nothing");
                    true
                }
                Err(e) => {
                    eprintln!("openhydra: could not reset the inherited share policy at {}: {e}", path.display());
                    false
                }
            }
        }
        _ => false,
    }
}

/// Write a policy file at `path` migrated from a legacy `--share-models` list, but only if the file
/// doesn't already exist (so an existing policy — the source of truth — is never clobbered). Empty
/// list → `all`; non-empty → `list`. Best-effort: a write error logs and leaves the file absent,
/// which the agent handles by sharing-all with a warning.
fn migrate_share_policy_if_absent(path: &std::path::Path, legacy: &[String]) {
    if path.exists() {
        return;
    }
    let policy = openhydra_agent::SharePolicy::from_legacy_list(legacy.to_vec());
    if let Err(e) = policy.write_atomic(path) {
        eprintln!("openhydra: could not write initial share policy to {}: {e}", path.display());
    }
}

/// Keep the legacy `settings.shared_models` mirror in sync with the policy (one-release
/// belt-and-suspenders: if the user downgrades, the old desktop still reads this and passes
/// `--share-models`). `all` ⇒ empty list (legacy "share everything"); `list` ⇒ the explicit models.
///
/// KNOWN LIMITATION (M3): the legacy 2-state encoding cannot express "share nothing" — both `all`
/// and `list []` map to the empty list, which an old build reads as "share everything". In-version
/// this is harmless (the policy FILE is authoritative and self-heals); the only exposure is a
/// *downgrade* to a pre-share-policy build while in the share-nothing state, which is documented, not
/// engineered around (the old format simply can't represent it).
fn mirror_policy_into_settings(policy: &openhydra_agent::SharePolicy) -> Vec<String> {
    match policy.mode {
        openhydra_agent::ShareMode::All => Vec::new(),
        openhydra_agent::ShareMode::List => policy.models.iter().cloned().collect(),
    }
}

// ── #7/#10: lifetime served/consumed model stats + daily buckets, persisted to disk. The
// agent's per-model counters reset to zero on every process restart, so the durable
// lifetime totals + time-series live here (the desktop accumulates poll-to-poll deltas and
// owns the JSON shape); this file just reads/writes the blob. ──
fn stats_path() -> PathBuf {
    openhydra_dir().join("stats.json")
}

/// Read the persisted model-stats blob (raw JSON string the UI wrote), or "" if none yet.
#[tauri::command]
fn load_stats() -> String {
    std::fs::read_to_string(stats_path()).unwrap_or_default()
}

/// Persist the UI's model-stats blob (verbatim JSON string).
#[tauri::command]
fn save_stats(data: String) -> Result<(), String> {
    let dir = openhydra_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    write_atomic(&stats_path(), &data) // #1: atomic — never leave the durable stats file half-written
}

// ── #9: OS device name, used as the default until the user edits it. ──
fn hostname_cmd() -> Option<String> {
    Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().trim_end_matches(".local").to_string())
        .filter(|s| !s.is_empty())
}

/// The system's device name (friendly name on macOS/Windows), for the Settings default.
#[tauri::command]
fn device_hostname() -> String {
    #[cfg(windows)]
    let name = std::env::var("COMPUTERNAME").ok();
    #[cfg(target_os = "macos")]
    let name = Command::new("scutil")
        .args(["--get", "ComputerName"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());
    #[cfg(all(unix, not(target_os = "macos")))]
    let name = hostname_cmd();
    name.filter(|s: &String| !s.is_empty())
        .or_else(hostname_cmd)
        .unwrap_or_else(|| "This machine".into())
}

/// The desktop bundle's version (from this crate's Cargo.toml, kept in lockstep with
/// tauri.conf.json). This is the authoritative "what am I running" string the UI shows — unlike
/// the agent crate's `CARGO_PKG_VERSION` surfaced by `/status`, which can lag the bundle.
#[tauri::command]
fn app_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ── AppImage desktop integration ──
// An AppImage is a portable single file: double-clickable, but it does NOT add itself to the
// applications menu. When we detect we're running from one (the AppImage runtime exports the
// `APPIMAGE` env var → the .AppImage path, and `APPDIR` → its mount), we offer a first-run
// "Add to Applications" step that drops a `.desktop` launcher pointing back at the AppImage, so
// it appears in the menu like an installed app. No-op on macOS / Windows / a non-AppImage Linux
// build (where `APPIMAGE` is unset), so the UI prompt simply never shows.

/// The path to the running `.AppImage`, or `None` when not running as one.
fn appimage_path() -> Option<String> {
    std::env::var("APPIMAGE").ok().filter(|s| !s.is_empty())
}

/// `~/.local/share/applications/openhydra.desktop`.
fn desktop_entry_path() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".local/share/applications/openhydra.desktop"))
}

/// The `.desktop` launcher body pointing at this AppImage. Pure (no I/O) so it's unit-testable.
/// `Exec` is double-quoted because an AppImage path can contain spaces.
fn desktop_entry(appimage: &str, icon: &str) -> String {
    format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=OpenHydra\n\
         Comment=Share your machine's inference engines and use the OpenHydra network.\n\
         Exec=\"{appimage}\" %U\n\
         Icon={icon}\n\
         Categories=Utility;Network;\n\
         Terminal=false\n\
         StartupWMClass=OpenHydra\n"
    )
}

#[derive(Serialize)]
struct AppImageStatus {
    /// True when running from an AppImage (the "Add to Applications" prompt only applies then).
    is_appimage: bool,
    /// True once a launcher entry already exists (don't offer to add it again).
    integrated: bool,
}

/// Whether we're an AppImage and whether a launcher entry already exists — the UI uses this to
/// decide whether to show the first-run "Add to Applications" prompt.
#[tauri::command]
fn appimage_status() -> AppImageStatus {
    AppImageStatus {
        is_appimage: appimage_path().is_some(),
        integrated: desktop_entry_path().map(|p| p.exists()).unwrap_or(false),
    }
}

/// Write a `.desktop` launcher (and best-effort copy the AppImage's icon into the hicolor theme)
/// so OpenHydra shows up in the applications menu. Idempotent; errors are surfaced to the UI.
#[tauri::command]
fn integrate_appimage() -> Result<(), String> {
    let appimage = appimage_path().ok_or("not running as an AppImage")?;
    let home = std::env::var("HOME").map_err(|_| "no HOME set".to_string())?;
    let apps_dir = PathBuf::from(&home).join(".local/share/applications");
    std::fs::create_dir_all(&apps_dir).map_err(|e| e.to_string())?;

    // Prefer the AppImage's own icon (`$APPDIR/.DirIcon`, the standard AppImage icon) copied into
    // the user's icon theme so the launcher shows the real logo; fall back to the app id.
    let mut icon = "co.openhydra.desktop".to_string();
    if let Ok(appdir) = std::env::var("APPDIR") {
        let diricon = PathBuf::from(&appdir).join(".DirIcon");
        let icons_dir = PathBuf::from(&home).join(".local/share/icons/hicolor/256x256/apps");
        if diricon.exists()
            && std::fs::create_dir_all(&icons_dir).is_ok()
            && std::fs::copy(&diricon, icons_dir.join("openhydra.png")).is_ok()
        {
            icon = "openhydra".to_string();
        }
    }

    let path = apps_dir.join("openhydra.desktop");
    std::fs::write(&path, desktop_entry(&appimage, &icon)).map_err(|e| e.to_string())?;
    // Best-effort refresh of the menu database (harmless if the tool is absent).
    let _ = Command::new("update-desktop-database").arg(&apps_dir).status();
    Ok(())
}

// ── #4: bundle the (bounded, in-memory) agent logs + environment into one file to send the
// developer. The agent logs network/status events only — never prompt/response content — so
// the bundle is prompt-free by construction. ──
#[tauri::command]
fn export_logs(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut buf = format!(
        "OpenHydra desktop {} · {} {} · exported@{}\n(agent network/status logs only — no prompt content)\n\n",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        ts,
    );
    for (name, role) in [("provider", &state.provider), ("gateway", &state.gateway)] {
        buf.push_str(&format!("──── {name} ────\n"));
        if let Ok(r) = role.lock() {
            buf.push_str(&format!(
                "running={} pid={:?} peer_id={:?}\n",
                r.status.running, r.status.pid, r.status.peer_id
            ));
            for line in &r.logs {
                buf.push_str(line);
                buf.push('\n');
            }
        }
        buf.push('\n');
    }
    let dir = openhydra_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("openhydra-logs-{ts}.txt"));
    std::fs::write(&path, buf).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

// ── Role state (one per supervised child) ──

#[derive(Debug, Default, Clone, Serialize)]
struct RoleStatus {
    running: bool,
    pid: Option<u32>,
    /// libp2p peer id, from the agent's "node up — libp2p=…" line.
    peer_id: Option<String>,
    /// Provider: the "auto-detected N engine(s): …" summary line.
    engines: Option<String>,
    /// Provider: models in the latest (re-)announce.
    announced: Option<u64>,
    /// Relay reservations accepted since start (needs RUST_LOG network=info).
    relays: u64,
    /// Exit description if the child died on its own (crash visibility).
    exited: Option<String>,
    /// The `public_network` setting this role was started with: Settings shows a pending restart
    /// while the saved one differs (review 2026-10-14, M2).
    started_public_network: Option<bool>,
    /// Gateway: whether it was started with tokenizer downloads on.
    started_tokenizer_downloads: Option<bool>,
}

struct Role {
    child: Option<Child>,
    status: RoleStatus,
    logs: VecDeque<String>,
    /// This role's slot in the signal-safe pid registry.
    pid_slot: &'static AtomicU32,
}

impl Role {
    fn new(pid_slot: &'static AtomicU32) -> Self {
        Self { child: None, status: RoleStatus::default(), logs: VecDeque::new(), pid_slot }
    }

    /// Reap a finished child, folding its exit status into the visible state.
    fn reap(&mut self) {
        if let Some(child) = &mut self.child {
            if let Ok(Some(exit)) = child.try_wait() {
                self.status.running = false;
                self.status.pid = None;
                self.status.exited = Some(format!("exited: {exit}"));
                self.child = None;
                registry_set(self.pid_slot, 0);
            }
        }
    }

    fn kill(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.status.running = false;
        self.status.pid = None;
        registry_set(self.pid_slot, 0);
    }
}

struct AppState {
    provider: Arc<Mutex<Role>>,
    gateway: Arc<Mutex<Role>>,
    settings: Mutex<Settings>,
    /// Set true when this launch auto-resumed sharing — drives the one-time "Resuming your shared
    /// models…" notice (with a "Don't resume" opt-out) in the UI. Transient, not persisted.
    resumed_on_launch: std::sync::atomic::AtomicBool,
    /// Set true by the share-policy self-heal (a corrupt file was backed up + reset to a safe
    /// default). Surfaced once via `get_state` (read-and-clear) so the UI can toast the user that
    /// their sharing selection was reset. Transient, not persisted.
    share_policy_reset: std::sync::atomic::AtomicBool,
    /// What the user approved in a native dialog this session for global announcement (see
    /// `global_exposure`), so one action asks once: a policy save and the start that follows it
    /// don't both ask. In memory only: a new session asks again before page-started sharing.
    global_approval: Mutex<GlobalExposure>,
}

// ── Log ingestion: strip ANSI, ring-buffer, parse status out of known lines ──

/// Strip ANSI SGR escape sequences (`ESC [ … m`) from tracing's colored output.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            // Skip until the terminating alphabetic byte of the CSI sequence.
            if chars.peek() == Some(&'[') {
                for c2 in chars.by_ref() {
                    if c2.is_ascii_alphabetic() {
                        break;
                    }
                }
                continue;
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// Parse the number right after `prefix` in `line` (e.g. "announced 6 model(s)").
fn number_after(line: &str, prefix: &str) -> Option<u64> {
    let rest = &line[line.find(prefix)? + prefix.len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Fold one agent log line into the role's status. Pure → unit-tested.
fn absorb_line(status: &mut RoleStatus, line: &str) {
    if let Some(i) = line.find("node up — libp2p=") {
        let rest = &line[i + "node up — libp2p=".len()..];
        status.peer_id = Some(rest.split_whitespace().next().unwrap_or("").to_string());
    } else if let Some(i) = line.find("auto-detected ") {
        status.engines = Some(line[i..].trim().to_string());
    } else if line.contains("re-announced ") {
        if let Some(n) = number_after(line, "re-announced ") {
            status.announced = Some(n);
        }
    } else if line.contains("announced ") && line.contains(" model") {
        if let Some(n) = number_after(line, "announced ") {
            status.announced = Some(n);
        }
    } else if line.contains("relay reservation accepted") {
        status.relays += 1;
    }
}

/// Pump one child output stream into the role's ring buffer + status, until EOF.
fn pump(stream: impl std::io::Read + Send + 'static, role: Arc<Mutex<Role>>) {
    std::thread::spawn(move || {
        let reader = std::io::BufReader::new(stream);
        for line in reader.lines() {
            let Ok(raw) = line else { break };
            let line = strip_ansi(&raw);
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(mut role) = role.lock() {
                absorb_line(&mut role.status, &line);
                if role.logs.len() >= LOG_CAP {
                    role.logs.pop_front();
                }
                role.logs.push_back(line);
            }
        }
    });
}

// ── Agent binary resolution: bundled sidecar first, dev tree fallback ──

/// The `openhydra-agent` binary to supervise. In a bundle, Tauri places the sidecar
/// next to the app executable (e.g. `Contents/MacOS/`); in dev, walk up from the exe
/// to find the repo's `target/release` build.
fn agent_binary() -> Option<PathBuf> {
    let name = if cfg!(windows) { "openhydra-agent.exe" } else { "openhydra-agent" };
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?.to_path_buf();
    let bundled = exe_dir.join(name);
    if bundled.is_file() {
        return Some(bundled);
    }
    let mut dir = exe_dir;
    for _ in 0..6 {
        for sub in [format!("target/release/{name}"), format!("release/{name}"), name.to_string()] {
            let cand = dir.join(sub);
            if cand.is_file() {
                return Some(cand);
            }
        }
        dir = dir.parent()?.to_path_buf();
    }
    None
}

/// Bootstrap peers built into this build: release builds set `OPENHYDRA_DEFAULT_BOOTSTRAPS` (comma
/// or whitespace separated multiaddrs) at compile time, so the addresses live in CI configuration,
/// not in tracked code. Dev builds have none and rely on mDNS or the user's own list.
fn builtin_bootstraps() -> Vec<String> {
    parse_bootstrap_list(option_env!("OPENHYDRA_DEFAULT_BOOTSTRAPS").unwrap_or(""))
}

fn parse_bootstrap_list(raw: &str) -> Vec<String> {
    raw.split(|c: char| c == ',' || c.is_whitespace())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// The bootstraps a role dials: the built-in ones plus the user's extras, without duplicates.
fn effective_bootstraps(settings: &Settings) -> Vec<String> {
    // A private deployment (public network off) uses only the user's own bootstraps.
    let mut out = if settings.public_network { builtin_bootstraps() } else { Vec::new() };
    for b in settings.bootstraps.iter().map(|b| b.trim()).filter(|b| !b.is_empty()) {
        if !out.iter().any(|x| x == b) {
            out.push(b.to_string());
        }
    }
    out
}

/// Whether the gateway downloads tokenizers from Hugging Face: the public network's setting, or the
/// private deployment's (off unless turned on).
fn tokenizer_downloads_on(settings: &Settings) -> bool {
    if settings.public_network {
        settings.tokenizer_downloads
    } else {
        settings.private_tokenizer_downloads
    }
}

/// The gateway's W4 flags: no tokenizer downloads from Hugging Face unless they are on (off by
/// default in a private deployment), and in a private deployment no fetch of the signed tokenizer
/// list from dl.openhydra.co either: the gateway uses the list built into this release (review
/// 2026-10-14, M1).
fn tokenizer_args(settings: &Settings) -> Vec<String> {
    let mut out = Vec::new();
    if !tokenizer_downloads_on(settings) {
        out.push("--no-tokenizer-downloads".to_string());
    }
    if !settings.public_network {
        out.extend(["--tokenizer-index-url".to_string(), "off".to_string()]);
    }
    out
}

/// The agents' `--public-relays`: decided from the bootstraps on the public network (the built-in
/// ones are OpenHydra's), never in a private deployment.
fn public_relays_arg(settings: &Settings) -> &'static str {
    if settings.public_network { "auto" } else { "off" }
}

/// `start_provider`'s refusal when the user hasn't confirmed what they share yet.
const NOT_CONFIRMED: &str = "choose what to share first";

// ── Native confirmations ──
// Actions that widen what this machine exposes (publishing models to everyone, admitting a swarm
// member, pointing the node at other peers or endpoints, …) are confirmed in an OS dialog drawn
// outside the webview. A page can call the command, but only a person can answer the dialog, so even
// script that got into the page can't take these actions on its own.

/// The error a command returns when the user declines its native dialog.
const CANCELLED: &str = "Cancelled";

/// Text from the page or a file, made safe to quote in a dialog: one line, no control characters,
/// at most `max` characters (so it can't impersonate the dialog's own wording).
fn dialog_text(s: &str, max: usize) -> String {
    let one_line: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    let t = one_line.trim();
    if t.chars().count() > max {
        format!("{}…", t.chars().take(max).collect::<String>())
    } else {
        t.to_string()
    }
}

/// Ask the user in a native dialog; true only if they press `ok`. Blocks until answered, so it must
/// not run on the main thread: callers are `#[tauri::command(async)]` or run in spawn_blocking.
fn native_confirm(app: &tauri::AppHandle, title: &str, message: &str, ok: &str) -> bool {
    native_choice(app, title, message, ok, "Cancel")
}

/// [`native_confirm`] with its own label for the other button (when it isn't "cancel the change").
fn native_choice(app: &tauri::AppHandle, title: &str, message: &str, ok: &str, other: &str) -> bool {
    let mut d = app
        .dialog()
        .message(message)
        .title(title)
        .kind(MessageDialogKind::Warning)
        .buttons(MessageDialogButtons::OkCancelCustom(ok.to_string(), other.to_string()));
    if let Some(w) = app.get_webview_window("main") {
        d = d.parent(&w);
    }
    d.blocking_show()
}

/// What a share policy announces to everyone: `all` = every shared model (including ones added
/// later), plus the models announced globally by name.
#[derive(Debug, Default, Clone, PartialEq)]
struct GlobalExposure {
    all: bool,
    models: std::collections::BTreeSet<String>,
}

fn global_exposure(p: &openhydra_agent::SharePolicy) -> GlobalExposure {
    use openhydra_agent::share_policy::Scope;
    let all = p.mode == openhydra_agent::ShareMode::All
        && p.default_scope == Scope::Global
        && p.default_global_consent.is_some();
    let models = p
        .models
        .iter()
        .chain(p.scopes.keys())
        .chain(p.global_consent.keys())
        .filter(|m| p.announce_globally(m))
        .cloned()
        .collect();
    GlobalExposure { all, models }
}

/// What `next` exposes that `prev` doesn't (empty when nothing widens).
fn exposure_beyond(next: &GlobalExposure, prev: &GlobalExposure) -> GlobalExposure {
    GlobalExposure {
        all: next.all && !prev.all,
        models: next.models.difference(&prev.models).cloned().collect(),
    }
}

impl GlobalExposure {
    fn is_empty(&self) -> bool {
        !self.all && self.models.is_empty()
    }
    fn absorb(&mut self, more: &GlobalExposure) {
        self.all |= more.all;
        self.models.extend(more.models.iter().cloned());
    }
    /// "every model you share (including ones you add later)" and/or a short list of names.
    fn describe(&self) -> String {
        let mut names: Vec<String> = self.models.iter().take(8).map(|m| dialog_text(m, 80)).collect();
        if self.models.len() > 8 {
            names.push(format!("and {} more", self.models.len() - 8));
        }
        match (self.all, names.is_empty()) {
            (true, true) => "every model you share, including ones you add later".to_string(),
            (true, false) => format!("every model you share, including ones you add later ({})", names.join(", ")),
            (false, _) => names.join(", "),
        }
    }
}

/// The global-publish dialog: true if the user agrees to announce `what` to everyone.
fn confirm_global(app: &tauri::AppHandle, what: &GlobalExposure, starting: bool, public_network: bool) -> bool {
    let head = if starting { "Start sharing?" } else { "Share with everyone?" };
    // "Everyone" is the network this node is on: OpenHydra's public one, or a private deployment.
    let who = if public_network { "Anyone on the OpenHydra network" } else { "Anyone on your private deployment" };
    let msg = format!(
        "{who} will be able to discover and send requests to: {}.\n\n\
         Your machine serves those requests. Your prompts, files and identity are not shared. You can \
         stop sharing, or make a model private, at any time.",
        what.describe()
    );
    native_confirm(app, head, &msg, if starting { "Start sharing" } else { "Share with everyone" })
}

/// Whether the provider accepts `--prompt-scoring auto` for this engine kind: auto-detect and the
/// engines with their own scoring API do; Ollama and LM Studio need the side-scorer next to the
/// agent; Exo and ComfyUI can't be scored (the agent refuses to start with the flag).
fn scoring_supported(kind: &str, side_scorer_present: bool) -> bool {
    match kind.trim() {
        "vllm" | "llama-cpp" => true,
        "ollama" | "lm-studio" => side_scorer_present,
        "exo" | "comfyui" => false,
        _ => true, // auto
    }
}

/// The engine kinds the provider accepts for `--engine-kind` (besides `auto`).
const ENGINE_KINDS: &[&str] = &["ollama", "vllm", "lm-studio", "llama-cpp", "exo", "comfyui"];

/// Provider flags for the chosen engine: nothing for `auto` (the agent's default), else
/// `--engine-kind <kind>` and, when set, `--engine <url>`. An unknown kind falls back to `auto`.
fn engine_args(kind: &str, endpoint: &str) -> Vec<String> {
    let kind = kind.trim();
    if !ENGINE_KINDS.contains(&kind) {
        return vec!["--engine-kind".into(), "auto".into()];
    }
    let mut out = vec!["--engine-kind".into(), kind.into()];
    let endpoint = endpoint.trim();
    if !endpoint.is_empty() {
        out.extend(["--engine".into(), endpoint.into()]);
    }
    out
}

/// Spawn one agent role and wire its output pumps. Global flags before the subcommand
/// (clap requirement), pinned identity per role, distinct listen port.
fn spawn_role(
    role: &Arc<Mutex<Role>>,
    identity: &str,
    listen_port: u16,
    status_port: u16,
    settings: &Settings,
    subcommand: &[String],
) -> Result<(), String> {
    let bin = agent_binary().ok_or_else(|| {
        "openhydra-agent binary not found (bundle is missing its sidecar, or no \
         target/release build in dev)"
            .to_string()
    })?;
    let dir = openhydra_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let mut cmd = Command::new(&bin);
    cmd.arg("--identity")
        .arg(dir.join(identity))
        .arg("--listen")
        .arg(format!("/ip4/0.0.0.0/tcp/{listen_port}"))
        .arg("--listen")
        .arg(format!("/ip4/0.0.0.0/udp/{listen_port}/quic-v1"))
        // P1: read-only introspection endpoint the app polls, loopback + bearer token.
        .arg("--status-bind")
        .arg(format!("127.0.0.1:{status_port}"));
    for b in effective_bootstraps(settings) {
        cmd.arg("--bootstrap").arg(b);
    }
    cmd.arg("--public-relays").arg(public_relays_arg(settings));
    // SL14: the switch is hidden until community relays ship (no other node can find a peer relay
    // yet), so a setting saved by an earlier version, which can't be turned off now, is ignored.
    if settings.peer_relay && PEER_RELAY_SWITCH_SHOWN {
        cmd.arg("--peer-relay");
    }
    cmd.args(subcommand)
        // #4: quiet by default (essential status lines are eprintln!, not RUST_LOG-gated, so
        // this never breaks the status views); verbose opt-in raises to info/debug.
        .env(
            "RUST_LOG",
            if settings.verbose_logs { "openhydra_network=info,debug" } else { "warn" },
        )
        .env("OPENHYDRA_STATUS_TOKEN", status_token())
        .env_remove("OPENHYDRA_API_KEY")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // The gateway requires the local key (passed in its environment, never on its command line,
    // which other local users can read).
    if subcommand.first().map(String::as_str) == Some("serve") {
        if let Some(key) = gateway_api_key() {
            cmd.env("OPENHYDRA_API_KEY", key);
        }
    }
    let mut child = cmd.spawn().map_err(|e| format!("spawn {}: {e}", bin.display()))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let mut guard = role.lock().map_err(|e| e.to_string())?;
    registry_set(guard.pid_slot, child.id());
    guard.status = RoleStatus {
        running: true,
        pid: Some(child.id()),
        started_public_network: Some(settings.public_network),
        started_tokenizer_downloads: Some(tokenizer_downloads_on(settings)),
        ..Default::default()
    };
    guard.logs.clear();
    guard.child = Some(child);
    drop(guard);

    if let Some(s) = stderr {
        pump(s, Arc::clone(role));
    }
    if let Some(s) = stdout {
        pump(s, Arc::clone(role));
    }
    Ok(())
}

// ── IPC commands ──

#[derive(Serialize)]
struct RoleView {
    status: RoleStatus,
    logs: Vec<String>,
}

#[derive(Serialize)]
struct FullState {
    provider: RoleView,
    gateway: RoleView,
    settings: Settings,
    agent_found: bool,
    gateway_url: String,
    /// True on the launch that auto-resumed sharing — the UI shows a one-time "Resuming…" notice.
    resumed_on_launch: bool,
    /// One-shot: the share policy was corrupt and self-healed to a safe default this poll — the UI
    /// toasts the user once that their sharing selection was reset. Read-and-cleared here.
    share_policy_reset: bool,
    /// How many bootstrap peers this build has built in (release builds; 0 in dev).
    builtin_bootstraps: usize,
    /// How many of the user's own bootstraps are OpenHydra's public ones (a private deployment
    /// doesn't use them; Settings says so).
    openhydra_bootstraps_listed: usize,
}

fn view(role: &Arc<Mutex<Role>>) -> RoleView {
    let mut guard = role.lock().expect("role lock");
    guard.reap();
    RoleView {
        status: guard.status.clone(),
        logs: guard.logs.iter().rev().take(200).rev().cloned().collect(),
    }
}

#[tauri::command]
fn get_state(state: tauri::State<'_, AppState>) -> FullState {
    let settings = state.settings.lock().expect("settings lock").clone();
    let gateway_url = format!("http://127.0.0.1:{}/v1", settings.gateway_port);
    let openhydra_bootstraps_listed = settings.bootstraps.iter().filter(|b| openhydra_agent::is_openhydra_bootstrap(b)).count();
    FullState {
        provider: view(&state.provider),
        gateway: view(&state.gateway),
        settings,
        agent_found: agent_binary().is_some(),
        gateway_url,
        resumed_on_launch: state.resumed_on_launch.load(std::sync::atomic::Ordering::Relaxed),
        // Read-and-clear (swap) so the reset toast fires exactly once per heal, unlike the
        // session-sticky `resumed_on_launch` load above.
        share_policy_reset: state.share_policy_reset.swap(false, std::sync::atomic::Ordering::Relaxed),
        builtin_bootstraps: builtin_bootstraps().len(),
        openhydra_bootstraps_listed,
    }
}

/// Open a link from a chat answer in the default browser. Only http(s) and mailto links, passed as a
/// single argument (no shell), so an answer can't make the app run or open anything else.
// Off the main thread: an opener (xdg-open on Linux) may wait for the program it starts, which
// would freeze the window until it exits (ADVERSARIAL item 14).
#[tauri::command(async)]
fn open_external(url: String) -> Result<(), String> {
    let u = url.trim();
    let lower = u.to_ascii_lowercase();
    if !(lower.starts_with("https://") || lower.starts_with("http://") || lower.starts_with("mailto:")) || u.chars().any(char::is_control) {
        return Err("only web and email links can be opened".into());
    }
    #[cfg(target_os = "macos")]
    let r = Command::new("open").arg(u).status();
    #[cfg(target_os = "windows")]
    let r = Command::new("rundll32").args(["url.dll,FileProtocolHandler", u]).status();
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let r = Command::new("xdg-open").arg(u).status();
    r.map(|_| ()).map_err(|e| e.to_string())
}

/// The user confirmed what they share in the Share dialog: later starts no longer ask.
#[tauri::command]
fn confirm_sharing(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let mut s = state.settings.lock().map_err(|e| e.to_string())?;
    if !s.sharing_confirmed {
        s.sharing_confirmed = true;
        store_settings(&s)?;
    }
    Ok(())
}

/// One of this app's two identities: the key file (under ~/.openhydra) and its libp2p peer id.
#[derive(Serialize)]
struct IdentityView {
    key_file: String,
    peer_id: Option<String>,
}

/// The app's identities for Settings › Identity: the provider key (sharing) and the consumer key
/// (chatting, swarm membership). Shown with the real file names instead of a generic path.
#[tauri::command]
fn identity_info() -> std::collections::BTreeMap<&'static str, IdentityView> {
    let mut out = std::collections::BTreeMap::new();
    for (role, file) in [("sharing", "desktop-provider.key"), ("chat", "desktop-consumer.key")] {
        out.insert(role, IdentityView {
            key_file: format!("~/.openhydra/{file}"),
            peer_id: peer_id_of(file),
        });
    }
    out
}

/// Show a file the app wrote (e.g. an exported log) in the OS file manager. Only paths inside
/// ~/.openhydra are accepted, so the UI can't be used to open arbitrary locations.
// Off the main thread: an opener (xdg-open on Linux) may wait for the program it starts, which
// would freeze the window until it exits (ADVERSARIAL item 14).
#[tauri::command(async)]
fn reveal_path(path: String) -> Result<(), String> {
    let p = std::path::Path::new(&path).canonicalize().map_err(|e| e.to_string())?;
    let root = openhydra_dir().canonicalize().map_err(|e| e.to_string())?;
    if !p.starts_with(&root) {
        return Err("only files OpenHydra wrote can be shown".into());
    }
    #[cfg(target_os = "macos")]
    let r = Command::new("open").arg("-R").arg(&p).status();
    #[cfg(target_os = "windows")]
    let r = {
        // canonicalize gives a `\\?\C:\…` path, which Explorer doesn't accept.
        let shown = p.display().to_string();
        let shown = shown.strip_prefix(r"\\?\").unwrap_or(&shown).to_string();
        Command::new("explorer").arg(format!("/select,{shown}")).status()
    };
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let r = Command::new("xdg-open").arg(p.parent().unwrap_or(&root)).status();
    r.map(|_| ()).map_err(|e| e.to_string())
}

/// Persist whether the user is sharing, so a restart can resume it (the provider is a child process
/// killed on quit/update, so intent must live in settings, not just the process). No-op if unchanged.
fn set_sharing_enabled(state: &tauri::State<'_, AppState>, on: bool) {
    if let Ok(mut s) = state.settings.lock() {
        if s.sharing_enabled != on {
            s.sharing_enabled = on;
            let _ = store_settings(&s);
        }
    }
}

/// Start sharing from the app window. Models announced to everyone that the user hasn't approved
/// in a native dialog this session are confirmed first (the tray and resume-on-launch use
/// [`start_provider_now`] directly: those are the user's own OS-level actions).
#[tauri::command(async)]
fn start_provider(app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<(), String> {
    let confirmed = state.settings.lock().map_err(|e| e.to_string())?.sharing_confirmed;
    if !confirmed {
        return Err(NOT_CONFIRMED.into());
    }
    let policy = openhydra_agent::SharePolicy::load(&share_policy_path())
        .unwrap_or_else(|_| openhydra_agent::SharePolicy::share_nothing());
    let approved = state.global_approval.lock().map(|a| a.clone()).unwrap_or_default();
    let unapproved = exposure_beyond(&global_exposure(&policy), &approved);
    if !unapproved.is_empty() {
        let public = state.settings.lock().map(|s| s.public_network).unwrap_or(true);
        if !confirm_global(&app, &unapproved, true, public) {
            return Err(CANCELLED.into());
        }
        if let Ok(mut a) = state.global_approval.lock() {
            a.absorb(&unapproved);
        }
    }
    let _one_at_a_time = PROVIDER_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
    start_provider_now(state)
}

/// Restart sharing (to apply saved settings): the same sharing, so no new consent. A restart widens
/// nothing; a wider audience (joining the public network) is confirmed when it is saved.
#[tauri::command(async)]
fn restart_provider(state: tauri::State<'_, AppState>) -> Result<(), String> {
    // One start, stop or restart at a time: a Stop pressed mid-restart isn't undone, and two
    // restarts at once don't start two providers (second review 2026-10-19).
    let _one_at_a_time = PROVIDER_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
    if !state.provider.lock().map(|r| r.status.running).unwrap_or(false) {
        return Err("not sharing".into());
    }
    state.provider.lock().map_err(|e| e.to_string())?.kill();
    start_provider_now(state)
}

/// Held while sharing starts, stops or restarts ([`restart_provider`]).
static PROVIDER_LIFECYCLE: Mutex<()> = Mutex::new(());

fn start_provider_now(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let settings = state.settings.lock().map_err(|e| e.to_string())?.clone();
    // Sharing starts only after the user has confirmed what they share (the Share dialog), whatever
    // asks for it — the in-app button, the tray menu. Resume-on-launch counts as confirmed.
    if !settings.sharing_confirmed {
        return Err(NOT_CONFIRMED.into());
    }
    let mut sub = vec!["provide".to_string()];
    sub.extend(engine_args(&settings.engine_kind, &settings.engine_endpoint));
    if settings.engine_autostart {
        sub.push("--engine-autostart".into());
    }
    if settings.authenticity_checks {
        // The same lookup the agent does (it runs next to this binary): the GGUF scorer only where
        // this CPU can run it, or the MLX scorer.
        use openhydra_agent::side_scorer::{cpu_runs_gguf_scorer, locate_scorer, MLX_SCORER_BIN, SCORER_BIN};
        let scorer_next_to_agent = (locate_scorer(None, SCORER_BIN).is_some() && cpu_runs_gguf_scorer())
            || locate_scorer(None, MLX_SCORER_BIN).is_some();
        if scoring_supported(&settings.engine_kind, scorer_next_to_agent) {
            sub.push("--prompt-scoring".into());
            sub.push("auto".into());
        } else {
            eprintln!("openhydra: authenticity checks aren't available for engine '{}'; sharing without them", settings.engine_kind);
        }
    }
    // Persist the receipt ledger + take-side credit across restarts. Own redb file: redb is
    // single-writer per process, so the gateway role uses a separate file (gateway-ledger.redb).
    sub.push("--db".into());
    sub.push(openhydra_dir().join("provider-ledger.redb").to_string_lossy().into_owned());
    // Per-model share policy: a hot-reloaded file, so toggling a model in the UI applies to the
    // running provider without a restart (see `save_share_policy`). Migrates the legacy
    // `settings.shared_models` on first run, and self-heals a corrupt file (→ share-nothing) before
    // launch so the provider never starts against a bad policy.
    let policy_path = ensure_valid_share_policy(&settings, &state.share_policy_reset);
    sub.push("--share-policy-file".into());
    sub.push(policy_path.to_string_lossy().into_owned());
    // M4-base: gate Private-scope models on swarm membership (owner-side authorizer reads this dir).
    sub.push("--swarms-dir".into());
    sub.push(swarms_dir().to_string_lossy().into_owned());
    // Our own gateway runs under the consumer key: it never settles with our provider (a
    // self-serve), so a provider that charges must not hold that against it.
    if let Some(id) = peer_id_of("desktop-consumer.key") {
        sub.push("--self-consumer".into());
        sub.push(id);
    }
    spawn_role(
        &state.provider,
        "desktop-provider.key",
        PROVIDER_PORT,
        PROVIDER_STATUS_PORT,
        &settings,
        &sub,
    )?;
    set_sharing_enabled(&state, true); // remember intent so a restart resumes sharing
    Ok(())
}

/// Persist the share policy (called by the UI when the user toggles a model or the master switch).
/// Written atomically; the running provider **hot-reloads it within ~1s — no restart**. Also mirrors
/// into `settings.shared_models` for one-release downgrade safety.
#[tauri::command(async)]
fn save_share_policy(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    policy: openhydra_agent::SharePolicy,
) -> Result<(), String> {
    let prev = openhydra_agent::SharePolicy::load(&share_policy_path())
        .unwrap_or_else(|_| openhydra_agent::SharePolicy::share_nothing());
    // Publishing more to everyone takes effect at once while sharing runs: ask now. When sharing is
    // off, the policy is inert and `start_provider` asks before anything is announced.
    let running = state.provider.lock().map(|r| r.status.running).unwrap_or(false);
    let widened = exposure_beyond(&global_exposure(&policy), &global_exposure(&prev));
    if running && !widened.is_empty() {
        let public = state.settings.lock().map(|s| s.public_network).unwrap_or(true);
        if !confirm_global(&app, &widened, false, public) {
            return Err(CANCELLED.into());
        }
        if let Ok(mut a) = state.global_approval.lock() {
            a.absorb(&widened);
        }
    }
    if policy.allow_remote_publish && !prev.allow_remote_publish
        && !native_confirm(
            &app,
            "Allow remote publishing?",
            "Devices you've given a control credential in your swarms will be able to make your shared models public (announced to everyone).",
            "Allow",
        )
    {
        return Err(CANCELLED.into());
    }
    persist_share_policy(&state, &policy)
}

/// Reset sharing preferences to a clean **share-nothing** default (the "Reset sharing preferences"
/// control, and the safe target for self-heal parity). Non-destructive: clears the selection; the
/// user re-picks, or one-taps "Share everything". Hot-reloads — no restart.
#[tauri::command]
fn reset_share_policy(state: tauri::State<'_, AppState>) -> Result<(), String> {
    persist_share_policy(&state, &openhydra_agent::SharePolicy::share_nothing())
}

/// Write the policy file atomically and keep the legacy `settings.shared_models` mirror in sync.
/// The running provider hot-reloads the file within ~1s (no restart).
fn persist_share_policy(
    state: &tauri::State<'_, AppState>,
    policy: &openhydra_agent::SharePolicy,
) -> Result<(), String> {
    std::fs::create_dir_all(openhydra_dir()).map_err(|e| e.to_string())?;
    policy.write_atomic(&share_policy_path()).map_err(|e| e.to_string())?;
    if let Ok(mut s) = state.settings.lock() {
        let mirror = mirror_policy_into_settings(policy);
        // This install now has its own policy: a missing file later is a deletion (share nothing),
        // not an upgrade to migrate.
        if s.shared_models != mirror || !s.share_policy_initialized {
            s.shared_models = mirror;
            s.share_policy_initialized = true;
            let _ = store_settings(&s);
        }
    }
    Ok(())
}

/// The current share policy (migrated from the legacy `settings.shared_models` when the file is
/// absent). The UI renders each model's *intended* on/off state from this; the *real* announced set
/// comes from `/status/share` in the status snapshot.
#[tauri::command]
fn read_share_policy(
    state: tauri::State<'_, AppState>,
) -> Result<openhydra_agent::SharePolicy, String> {
    let settings = state.settings.lock().map_err(|e| e.to_string())?.clone();
    // Heal-if-corrupt (backs up + resets + flags) then read the now-valid file; fail-closed to
    // share-nothing if it somehow still doesn't parse.
    let path = ensure_valid_share_policy(&settings, &state.share_policy_reset);
    Ok(openhydra_agent::SharePolicy::load(&path)
        .unwrap_or_else(|_| openhydra_agent::SharePolicy::share_nothing()))
}

/// #7: the libp2p PeerId of this desktop's own provider identity. The gateway is told this so
/// a self-serve (same machine provides *and* consumes) settles no receipt and moves no credit.
/// Runs the agent's `peer-id` subcommand — stable, no swarm; the id is the same whether or not
/// the provider is currently running. Best-effort: `None` (consumer-only) if it can't resolve.
fn provider_peer_id() -> Option<String> {
    peer_id_of("desktop-provider.key")
}

/// The libp2p PeerId of the desktop identity file `key_name` (in the OpenHydra dir).
fn peer_id_of(key_name: &str) -> Option<String> {
    let bin = agent_binary()?;
    let key = openhydra_dir().join(key_name);
    let out = Command::new(&bin).arg("--identity").arg(&key).arg("peer-id").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let id = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!id.is_empty()).then_some(id)
}

/// M2/M4: export a signed `.openhydra` card for a shared model. Shells to the agent's `card export`
/// (loads the provider identity, detects the model, enforces the reach/consent gate, signs). With
/// `swarm` set (M4) it exports a PRIVATE card bound to that owned swarm; otherwise a public card.
/// Returns `{ card, magnet }` for the UI to save/copy. The identity is `desktop-provider.key` — the
/// key the provider serves with, which the private serve gate authorizes members against.
#[tauri::command]
fn export_card(
    model: String,
    ttl_secs: Option<u64>,
    region: Option<String>,
    swarm: Option<String>,
) -> Result<openhydra_agent::CardExport, String> {
    let bin = agent_binary().ok_or_else(|| {
        "openhydra-agent binary not found (bundle missing its sidecar, or no dev build)".to_string()
    })?;
    let key = openhydra_dir().join("desktop-provider.key");
    let mut cmd = Command::new(&bin);
    cmd.arg("--identity")
        .arg(&key)
        .arg("card")
        .arg("export")
        .arg("--model")
        .arg(&model)
        .arg("--share-policy-file")
        .arg(share_policy_path())
        .arg("--ttl-secs")
        .arg(ttl_secs.unwrap_or(30 * 24 * 3600).to_string());
    if let Some(r) = region.filter(|r| !r.is_empty()) {
        cmd.arg("--region").arg(r);
    }
    // M4: a private card names the swarm it's gated on + points the export at the swarms dir (to
    // confirm ownership).
    if let Some(pk) = swarm.filter(|s| !s.is_empty()) {
        cmd.arg("--swarm").arg(pk).arg("--swarms-dir").arg(swarms_dir());
    }
    let out = cmd.output().map_err(|e| format!("run card export: {e}"))?;
    if !out.status.success() {
        return Err(agent_failure(&out.stderr, "card export failed"));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("parse card export output: {e}"))
}

/// The reason a one-shot agent command failed, from its stderr. The agent ends a failure with one
/// `openhydra-agent: <reason>` line; a command that brings up a network node (`swarm remote-scope`)
/// also logs before it (coloured tracing lines), which the user should not have to read through.
/// Falls back to the last line, then to `fallback`.
fn agent_failure(stderr: &[u8], fallback: &str) -> String {
    let text = String::from_utf8_lossy(stderr);
    // Drop ANSI colour sequences (ESC [ … letter).
    let mut plain = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            while let Some(n) = chars.next() {
                if n.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            plain.push(c);
        }
    }
    let lines: Vec<&str> = plain.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    lines
        .iter()
        .rev()
        .find_map(|l| l.strip_prefix("openhydra-agent: "))
        .or(lines.last().copied())
        .map(str::to_string)
        .unwrap_or_else(|| fallback.to_string())
}

/// The imported-cards store the gateway reads (`--cards-file`).
fn cards_file_path() -> PathBuf {
    openhydra_dir().join("cards.json")
}

/// Serializes the read-modify-write of `cards.json` across this process's card commands, so two
/// concurrent imports (or an import racing a removal) can't lose an update. The desktop is the only
/// writer; the atomic temp+rename in `add_card`/`remove_card` already prevents torn reads.
static CARDS_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// M2.1: a card (magnet text, or the contents of a `.openhydra` file) delivered by an OS file/URL
/// open, awaiting frontend pickup via `take_pending_card`. Covers the cold-launch race where the app
/// is opened BY a card before the UI is ready to receive the `card-opened` event.
struct PendingCard(std::sync::Mutex<Option<String>>);

/// Cap on a `.openhydra` file read — a card is a few KB; this bounds a symlink to `/dev/zero` or a
/// giant file so an opened file can never hang / OOM the app (review fix #1).
const MAX_CARD_FILE_BYTES: u64 = 256 * 1024;

/// M2.1: turn an opened URL into card text. An `openhydra:` URL *is* the magnet string; a `file:` URL
/// (a `.openhydra` file association) is read from disk (guarded). Scheme allowlist — anything else
/// never touches disk (review fix #1: keeps the `file:` read unreachable from an attacker's
/// `openhydra:` link, which only ever hits the string arm).
fn card_text_from_url(url: &url::Url) -> Option<String> {
    match url.scheme() {
        "openhydra" => Some(url.as_str().to_string()),
        "file" => read_card_file(url),
        _ => None,
    }
}

/// Read a `.openhydra` file safely (review fix #1): resolve symlinks and require the RESOLVED file to
/// actually be a `.openhydra` (so a `.openhydra` symlink pointing at `~/.ssh/id_rsa` or `/dev/zero` is
/// refused, not read), then read it under a hard size cap. Any failure → `None` (never panics).
fn read_card_file(url: &url::Url) -> Option<String> {
    use std::io::Read;
    let path = url.to_file_path().ok()?;
    // Canonicalize BEFORE trusting the extension — resolves symlinks so the check applies to the real
    // target, not the `.openhydra`-named link.
    let resolved = std::fs::canonicalize(&path).ok()?;
    if !resolved.extension().is_some_and(|e| e.eq_ignore_ascii_case("openhydra")) {
        return None;
    }
    let file = std::fs::File::open(&resolved).ok()?;
    if file.metadata().map(|m| m.len()).unwrap_or(u64::MAX) > MAX_CARD_FILE_BYTES {
        return None;
    }
    let mut buf = String::new();
    // `take` bounds the read even for a device file that reports size 0 but streams forever.
    file.take(MAX_CARD_FILE_BYTES).read_to_string(&mut buf).ok()?;
    Some(buf)
}

/// M2.1 (Win/Linux): a second-launch file open / deep link arrives as a process argument (macOS uses
/// Apple events → `RunEvent::Opened` / the deep-link plugin, and its file isn't in argv, so this is a
/// no-op there). Route the first `openhydra:` URL or `.openhydra` file path found in `args`.
fn deliver_card_from_args<I: IntoIterator<Item = String>>(app: &tauri::AppHandle, args: I) {
    for arg in args {
        let url = if arg.starts_with("openhydra:") {
            arg.parse::<url::Url>().ok()
        } else if arg.to_ascii_lowercase().ends_with(".openhydra") {
            std::fs::canonicalize(&arg).ok().and_then(|p| url::Url::from_file_path(p).ok())
        } else {
            None
        };
        if let Some(u) = url {
            if let Some(text) = card_text_from_url(&u) {
                deliver_card(app, text);
                return; // one card per launch
            }
        }
    }
}

/// M2.1: stash opened card text for cold-launch pickup AND emit `card-opened` for the already-running
/// case, then focus the window.
fn deliver_card(app: &tauri::AppHandle, text: String) {
    if let Some(state) = app.try_state::<PendingCard>() {
        if let Ok(mut g) = state.0.lock() {
            *g = Some(text.clone());
        }
    }
    let _ = app.emit("card-opened", text);
    show_main_window(app);
}

/// Bring the main window back: closing it only hides it (sharing keeps running), so whatever
/// should show the app — the Dock icon, opening the app again, a card, the tray — shows it,
/// restores it if minimised, and focuses it. Focusing alone left a hidden window hidden.
fn show_main_window<R: tauri::Runtime>(app: &impl tauri::Manager<R>) {
    if let Some(w) = app.get_webview_window("main") {
        if w.is_minimized().unwrap_or(false) {
            let _ = w.unminimize();
        }
        let _ = w.show();
        let _ = w.set_focus();
    }
}

/// M2.1: the frontend calls this once on startup to pick up a card the app was launched with (a
/// double-clicked `.openhydra` file or an `openhydra:` link). Take-and-clear so it fires once.
#[tauri::command]
fn take_pending_card(state: tauri::State<'_, PendingCard>) -> Option<String> {
    state.0.lock().ok().and_then(|mut g| g.take())
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// M2: verify a `.openhydra` card (JSON or magnet) WITHOUT persisting it — powers the import preview
/// so the user sees the provider/model/expiry + ✔ signature before adding.
#[tauri::command]
fn preview_card(input: String) -> Result<openhydra_agent::Card, String> {
    Ok(openhydra_agent::cards::parse_and_verify(&input, now_unix_ms())?.card)
}

/// M2: import a `.openhydra` card (JSON or magnet). Verifies it in-process (signature, key↔peer-id
/// binding, privacy, schema, expiry), persists it to the cards store, and returns the verified card
/// for the preview. The running gateway hot-reloads the store, so the model becomes routable at once.
#[tauri::command]
fn import_card(input: String) -> Result<openhydra_agent::Card, String> {
    let verified = openhydra_agent::cards::parse_and_verify(&input, now_unix_ms())?;
    let _guard = CARDS_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    openhydra_agent::cards::add_card(&cards_file_path(), &verified.card)?;
    Ok(verified.card)
}

/// M2: the currently-valid (verified, unexpired) imported cards, for the "Added via card" list.
#[tauri::command]
fn list_cards() -> Result<Vec<openhydra_agent::Card>, String> {
    let store = openhydra_agent::cards::CardStore::new(cards_file_path());
    Ok(store.valid_cards(now_unix_ms()))
}

/// M2: remove an imported card by its `(libp2p_peer_id, model_id)`. Returns how many were removed.
#[tauri::command]
fn remove_imported_card(libp2p_peer_id: String, model_id: String) -> Result<usize, String> {
    let _guard = CARDS_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    openhydra_agent::cards::remove_card(&cards_file_path(), &libp2p_peer_id, &model_id)
}

// ── M3: swarm membership (private sharing) ─────────────────────────────────────────────────────
// Swarm state lives under ~/.openhydra/swarms/ (owner files carry the group secret, 0600). These
// commands call the agent glue in-process (like card import); the group secret never crosses into JS
// — list_swarms returns only redacted views. Enrollment is offline copy/paste (no wire in v1). The
// node identity used to sign enrollment requests / bind accepted credentials is `desktop-provider.key`
// (the same identity the provider/card-export use), so a swarm binds to this machine's real peer id.

/// Directory holding one JSON file per swarm.
fn swarms_dir() -> PathBuf {
    openhydra_dir().join("swarms")
}

/// The node identity swarm membership binds to. This MUST be the identity that dials OUT to consume,
/// i.e. the gateway/**consumer** key — a credential is bound to the member's peer id and checked
/// against the connection's authenticated peer id at the provider's serve gate (M4-base). Enrolling
/// with the consumer key means the credential the gateway later presents matches the peer it connects
/// as. (Swarm *ownership* — create/approve/revoke — uses the group key, independent of this.)
fn swarm_identity_path() -> PathBuf {
    openhydra_dir().join("desktop-consumer.key")
}

/// M5 dedicated control identity: the key a **control** credential binds to, distinct from the
/// consumer/gateway key. A remote-scope one-shot dials as this key, so it never shares a PeerId with
/// the running gateway (review finding B6). Generated lazily on first control enrollment (0600 via the
/// agent's identity loader). Kept separate so leaking the always-online consumer key can't republish a
/// rig, and a genuinely control-only device carries no serve access.
fn swarm_control_identity_path() -> PathBuf {
    openhydra_dir().join("desktop-control.key")
}

/// Serializes the read-modify-write of a swarm file across this process's swarm commands (approve +
/// revoke both mutate an owner record). The atomic temp+rename in the agent prevents torn reads.
static SWARMS_WRITE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// M3: every swarm on this node, as redacted views (no group secret). Owner + member.
#[tauri::command]
fn list_swarms() -> Result<Vec<openhydra_agent::SwarmView>, String> {
    openhydra_agent::swarms::list_swarms(&swarms_dir())
}

/// A swarm's label and whether we own it, for a dialog ("this swarm" if it isn't found).
fn swarm_label_of(swarm_public_key: &str) -> (String, bool) {
    openhydra_agent::swarms::list_swarms(&swarms_dir())
        .ok()
        .and_then(|v| v.into_iter().find(|s| s.swarm_public_key == swarm_public_key.trim()))
        .map(|s| (dialog_text(&s.label, 60), matches!(s.role, openhydra_agent::SwarmRole::Owner)))
        .unwrap_or_else(|| ("this swarm".to_string(), false))
}

/// M3: create a swarm (generate the group keypair, persist an owner record). Returns its view.
#[tauri::command]
fn create_swarm(label: String) -> Result<openhydra_agent::SwarmView, String> {
    let _guard = SWARMS_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    openhydra_agent::swarms::create_swarm(&swarms_dir(), label.trim(), now_unix_ms())
}

/// One or both enrollment requests a member emits (M5 dedicated control identity). *Consume* fills
/// `consume` (from the consumer key), *control* fills `control` (from the control key), *both* fills
/// both — the member sends whichever are present to the owner in one message.
#[derive(serde::Serialize)]
struct EnrollRequests {
    consume: Option<openhydra_agent::EnrollmentRequestExport>,
    control: Option<openhydra_agent::EnrollmentRequestExport>,
}

/// M3/M5 (member): build the signed enrollment request(s) to send a swarm owner out-of-band. `swarm`
/// is an optional public-key hint (e.g. from a card) of which swarm to join. `role` selects which
/// identities to request from: `consume` (consumer key, default), `control` (control key — grants no
/// serve access), or `both` (a request from each key; "both" is inherently two credentials because
/// control and consume bind to different keys).
#[tauri::command]
fn swarm_enroll_request(
    swarm: Option<String>,
    label: String,
    role: Option<String>,
) -> Result<EnrollRequests, String> {
    let hint = swarm.as_deref().unwrap_or("").trim().to_string();
    let label = label.trim().to_string();
    let now = now_unix_ms();
    let want_consume;
    let want_control;
    match role.as_deref().unwrap_or("consume") {
        "consume" | "" => {
            want_consume = true;
            want_control = false;
        }
        "control" => {
            want_consume = false;
            want_control = true;
        }
        "both" => {
            want_consume = true;
            want_control = true;
        }
        other => return Err(format!("invalid role '{other}' (expected consume|control|both)")),
    }
    let consume = if want_consume {
        Some(openhydra_agent::swarms::enroll_request_at(&swarm_identity_path(), &hint, &label, now)?)
    } else {
        None
    };
    let control = if want_control {
        Some(openhydra_agent::swarms::enroll_request_at(
            &swarm_control_identity_path(),
            &hint,
            &label,
            now,
        )?)
    } else {
        None
    };
    Ok(EnrollRequests { consume, control })
}

/// M3 (owner): preview an incoming enrollment request — verifies the member's signature and returns
/// their identity + fingerprint to confirm out-of-band, without approving.
#[tauri::command]
fn preview_enroll_request(request: String) -> Result<openhydra_agent::EnrollmentRequest, String> {
    openhydra_agent::swarms::preview_enrollment_request(&request)
}

/// The credential(s) an owner returns to a member (M5 dedicated control identity). *Consume* fills
/// `consume` (a `CAP_SERVE` credential on the member's consumer key); *control* fills `control` (a
/// `CAP_CONTROL`-only credential on the member's control key); *both* fills both.
#[derive(serde::Serialize)]
struct ApprovedCredentials {
    consume: Option<openhydra_agent::ApprovedCredential>,
    control: Option<openhydra_agent::ApprovedCredential>,
}

/// M3/M5 (owner): approve enrollment request(s) into signed credential(s) valid for `ttl_secs`
/// (default 90 days), records the member(s), returns the credential(s) to send back out-of-band.
///
/// `grant` selects the capability: `consume` (default) → a serve credential over `request`; `control`
/// → a `CAP_CONTROL`-only credential over `request`; `both` → a serve credential over `request` *and*
/// a control credential over `control_request` (the member's two requests bind different keys, so
/// "both" is two credentials). A control credential lets that device flip this owner's rig models'
/// scope via `swarm_remote_scope` — grant it only to your own trusted devices; it carries no serve
/// access.
#[tauri::command(async)]
fn swarm_approve_member(
    app: tauri::AppHandle,
    swarm_public_key: String,
    request: String,
    control_request: Option<String>,
    member_label: String,
    ttl_secs: Option<u64>,
    grant: Option<String>,
) -> Result<ApprovedCredentials, String> {
    // Admitting a member lets their key use (or, with control, re-scope) this owner's private models:
    // confirm who, with the fingerprint the user is told to check out-of-band.
    let grant_text = match grant.as_deref().unwrap_or("consume") {
        "control" => "control your rigs' model sharing (no access to the models)",
        "both" => "use your private models and control your rigs' model sharing",
        _ => "use your private models",
    };
    let (swarm_label, _) = swarm_label_of(&swarm_public_key);
    let mut keys = Vec::new();
    for req in std::iter::once(request.as_str()).chain(control_request.as_deref()) {
        if req.trim().is_empty() {
            continue;
        }
        let r = openhydra_agent::swarms::preview_enrollment_request(req)?;
        keys.push(format!("{} — key {}", dialog_text(&r.label, 60), openhydra_agent::key_fingerprint(&r.member_public_key)));
    }
    let msg = format!(
        "Approve this member into “{swarm_label}”? They will be able to {grant_text}.\n\n{}\n\n\
         Check the key fingerprint with them (in person or by voice) before approving.",
        keys.join("\n")
    );
    if !native_confirm(&app, "Approve swarm member?", &msg, "Approve") {
        return Err(CANCELLED.into());
    }
    let _guard = SWARMS_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let ttl = ttl_secs.unwrap_or(90 * 24 * 3600);
    let label = member_label.trim().to_string();
    let approve = |req: &str, caps: u32| {
        openhydra_agent::swarms::approve_member_with_caps(
            &swarms_dir(),
            &swarm_public_key,
            req,
            &label,
            ttl,
            caps,
            now_unix_ms(),
        )
    };
    match grant.as_deref().unwrap_or("consume") {
        "consume" | "" => Ok(ApprovedCredentials {
            consume: Some(approve(&request, openhydra_agent::CAP_SERVE)?),
            control: None,
        }),
        "control" => Ok(ApprovedCredentials {
            consume: None,
            control: Some(approve(&request, openhydra_agent::CAP_CONTROL)?),
        }),
        "both" => {
            let control_req = control_request
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    "grant 'both' needs the member's control request too (control_request)".to_string()
                })?;
            Ok(ApprovedCredentials {
                consume: Some(approve(&request, openhydra_agent::CAP_SERVE)?),
                control: Some(approve(control_req, openhydra_agent::CAP_CONTROL)?),
            })
        }
        other => Err(format!("invalid grant '{other}' (expected consume|control|both)")),
    }
}

/// M5 (control member): send a signed `REMOTE_SCOPE_SET` to a rig — flip one of its shared models'
/// scope (`global` | `private` | `device`) from this device. Shells to the agent's `swarm
/// remote-scope` (it needs a live one-shot node to dial the rig), dialing as the **control** key the
/// stored control credential is bound to (see `swarm_control_identity_path`). Returns the rig's ack
/// line (`applied: <model> -> <scope>`), or the rig's refusal / a transport error.
#[tauri::command(async)]
fn swarm_remote_scope(
    app: tauri::AppHandle,
    swarm_public_key: String,
    provider: String,
    model: String,
    scope: String,
) -> Result<String, String> {
    let scope = scope.trim().to_lowercase();
    if !matches!(scope.as_str(), "global" | "private" | "device") {
        return Err(format!("invalid scope '{scope}' (expected global|private|device)"));
    }
    if scope == "global"
        && !native_confirm(
            &app,
            "Publish on the rig?",
            &format!(
                "Ask the rig {} to publish “{}” to everyone? It becomes discoverable and usable on the public network, served by that rig. The rig refuses if its owner turned remote publishing off.",
                dialog_text(&provider, 60),
                dialog_text(&model, 80)
            ),
            "Publish",
        )
    {
        return Err(CANCELLED.into());
    }
    let bin = agent_binary().ok_or_else(|| {
        "openhydra-agent binary not found (bundle missing its sidecar, or no dev build)".to_string()
    })?;
    // Dial as the key the control credential is bound to — the dedicated control key on the clean path,
    // or the consumer key on the Inc-D fallback (a legacy dual-cap credential). This is the B6 fix: the
    // one-shot control node no longer reuses the gateway's consumer-key PeerId.
    let identity = openhydra_agent::swarms::control_identity_for(
        &swarms_dir(),
        swarm_public_key.trim(),
        &[swarm_control_identity_path(), swarm_identity_path()],
    )?;
    let mut cmd = Command::new(&bin);
    cmd.arg("--identity").arg(identity);
    let settings = load_settings();
    for b in effective_bootstraps(&settings) {
        cmd.arg("--bootstrap").arg(b);
    }
    cmd.arg("--public-relays").arg(public_relays_arg(&settings));
    cmd.arg("swarm")
        .arg("remote-scope")
        .arg("--swarm")
        .arg(swarm_public_key.trim())
        .arg("--provider")
        .arg(provider.trim())
        .arg("--model")
        .arg(model.trim())
        .arg("--scope")
        .arg(&scope)
        .arg("--swarms-dir")
        .arg(swarms_dir());
    let out = cmd.output().map_err(|e| format!("run remote-scope: {e}"))?;
    if !out.status.success() {
        return Err(agent_failure(&out.stderr, "remote-scope failed"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// M3 (owner): revoke a member by its public key (drops it from the list + adds to the revocation set).
#[tauri::command]
fn swarm_revoke_member(swarm_public_key: String, member_public_key: String) -> Result<(), String> {
    let _guard = SWARMS_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    openhydra_agent::swarms::revoke_member(&swarms_dir(), &swarm_public_key, &member_public_key)
}

/// M3 (member): preview a credential before accepting — verifies its signature + expiry and returns
/// the swarm it grants, without persisting.
#[tauri::command]
fn preview_swarm_credential(
    credential: String,
) -> Result<openhydra_agent::MembershipCredential, String> {
    openhydra_agent::swarms::preview_credential(&credential, now_unix_ms())
}

/// Accept one credential, binding against the identity its capability implies: a `CAP_CONTROL`-only
/// credential binds to the control key; anything granting serve binds to the consumer key. The agent's
/// accept then re-verifies the binding and routes storage into the matching slot (Inc A).
fn accept_one_credential(credential: &str, label: &str) -> Result<openhydra_agent::SwarmView, String> {
    let preview = openhydra_agent::swarms::preview_credential(credential, now_unix_ms())?;
    let is_control_only =
        preview.has_capability(openhydra_agent::CAP_CONTROL) && !preview.has_capability(openhydra_agent::CAP_SERVE);
    let identity = if is_control_only { swarm_control_identity_path() } else { swarm_identity_path() };
    openhydra_agent::swarms::accept_credential_at(&swarms_dir(), &identity, credential, label, now_unix_ms())
}

/// M3/M5 (member): accept the credential(s) the owner returned — verifies each is signed, bound to the
/// right local identity (control creds → control key, serve creds → consumer key), and unexpired, then
/// persists them. `control_credential` carries the second credential of a "both" grant. Returns the
/// final stored view (both slots populated for "both").
#[tauri::command]
fn swarm_accept_credential(
    credential: String,
    control_credential: Option<String>,
    label: Option<String>,
) -> Result<openhydra_agent::SwarmView, String> {
    let _guard = SWARMS_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let label = label.as_deref().unwrap_or("").trim();
    let mut view = accept_one_credential(credential.trim(), label)?;
    if let Some(cc) = control_credential.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        view = accept_one_credential(cc, label)?;
    }
    Ok(view)
}

/// M3: forget a swarm entirely (owner: destroys the group key; member: drops our credential).
#[tauri::command(async)]
fn forget_swarm(app: tauri::AppHandle, swarm_public_key: String) -> Result<(), String> {
    let (label, owner) = swarm_label_of(&swarm_public_key);
    let ok = if owner {
        native_confirm(
            &app,
            "Delete swarm?",
            &format!("Delete “{label}”? This destroys its group key: you can't issue any more credentials for it (existing members keep theirs until they expire)."),
            "Delete",
        )
    } else {
        native_confirm(&app, "Leave swarm?", &format!("Leave “{label}”? You'll drop your credential for it."), "Leave")
    };
    if !ok {
        return Err(CANCELLED.into());
    }
    let _guard = SWARMS_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    openhydra_agent::swarms::forget_swarm(&swarms_dir(), &swarm_public_key)
}

/// The gateway's `--min-authenticity` for the `min_authenticity` setting: `None` for off (or anything
/// unrecognised, rather than passing the gateway a value it would refuse to start with). A2 needs
/// log-prob checks, which the desktop's gateway doesn't run (it has no reference source), so no
/// provider could reach it: a saved "A2" (an unreleased build offered it) is used as A1.
fn gateway_floor(setting: &str) -> Option<&'static str> {
    match setting {
        "A1" | "A2" => Some("A1"),
        _ => None,
    }
}

/// The chat view's error for a gateway reply with a non-success `status` and JSON `body`: the
/// status, the error's type and its message, so the UI can say what to do (e.g. a floor refusal).
fn gateway_error(status: u16, body: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let kind = v["error"]["type"].as_str().unwrap_or("");
    let msg = v["error"]["message"].as_str().unwrap_or(body);
    format!("gateway {status} {kind}: {msg}")
}

#[tauri::command]
fn start_gateway(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let settings = state.settings.lock().map_err(|e| e.to_string())?.clone();
    let mut sub = vec![
        "serve".to_string(),
        "--bind".into(),
        format!("127.0.0.1:{}", settings.gateway_port),
    ];
    // #7: mark our own provider so the gateway never settles credit against itself on a
    // self-serve. Computed from the provider identity key (stable across runs).
    if let Some(id) = provider_peer_id() {
        sub.push("--self-provider".into());
        sub.push(id);
    }
    // Persist earned reputation + give-side credit across restarts (separate redb from the
    // provider role — redb allows a single writer process per file).
    sub.push("--db".into());
    sub.push(openhydra_dir().join("gateway-ledger.redb").to_string_lossy().into_owned());
    // M2: route to imported `.openhydra` cards (verified providers dialable by peer id without
    // discovery). Hot-reloaded, so a card imported while the gateway runs takes effect immediately.
    sub.push("--cards-file".into());
    sub.push(cards_file_path().to_string_lossy().into_owned());
    // M4-base: present a held swarm membership credential so a private (swarm-scoped) provider serves
    // us. The credential is bound to this same `desktop-consumer.key` (see `swarm_identity_path`).
    sub.push("--swarms-dir".into());
    sub.push(swarms_dir().to_string_lossy().into_owned());
    // D1: the desktop replays prompts for audits only when the user turned it on.
    sub.push("--audit-replay".into());
    sub.push(if settings.audit_replay { "on" } else { "off" }.into());
    sub.extend(tokenizer_args(&settings));
    // D2: the user's floor on which providers serve them, enforced.
    if let Some(tier) = gateway_floor(&settings.min_authenticity) {
        sub.extend(["--min-authenticity".into(), tier.into(), "--assurance".into(), "gate".into()]);
    }
    spawn_role(
        &state.gateway,
        "desktop-consumer.key",
        GATEWAY_PORT,
        GATEWAY_STATUS_PORT,
        &settings,
        &sub,
    )
}

#[tauri::command]
fn stop_provider(state: tauri::State<'_, AppState>) {
    let _one_at_a_time = PROVIDER_LIFECYCLE.lock().unwrap_or_else(|e| e.into_inner());
    state.provider.lock().expect("role lock").kill();
    set_sharing_enabled(&state, false); // user turned sharing off — don't resume on next launch
}

#[tauri::command]
fn stop_gateway(state: tauri::State<'_, AppState>) {
    state.gateway.lock().expect("role lock").kill();
}

#[derive(Serialize)]
struct EngineView {
    label: String,
    url: String,
    models: Vec<String>,
}

/// P1: fetch a role's `--status-bind` snapshot (bearer-gated loopback GET), returning the
/// raw JSON the agent serves. Provider-first (it carries peers + transfer counters), then
/// the gateway. Returns `null` if neither role is running / reachable — the UI treats that
/// as "start a role to see the network views".
#[tauri::command]
async fn status_snapshot(state: tauri::State<'_, AppState>) -> Result<Option<serde_json::Value>, ()> {
    let prov = state.provider.lock().map(|r| r.status.running).unwrap_or(false);
    let gw = state.gateway.lock().map(|r| r.status.running).unwrap_or(false);
    if !prov && !gw {
        return Ok(None);
    }
    // The two roles hold different halves of the economy: the provider process tracks
    // take-side credit + serve-rate caps; the gateway (consumer) tracks earned reputation of
    // the providers it used + give-side credit. Fetch provider-first for the base snapshot
    // (peers + transfer counters), then — when both run — overlay the gateway's economy so the
    // UI sees reputation and credit together.
    Ok(tauri::async_runtime::spawn_blocking(move || {
        let base_port = if prov { PROVIDER_STATUS_PORT } else { GATEWAY_STATUS_PORT };
        let mut base = fetch_status(base_port, "/status")?;
        // Attach the provider's REAL share view (policy mode + intended list + actually-announced
        // set) so the UI renders each model's true state instead of guessing from detection.
        if prov {
            if let Some(share) = fetch_status(PROVIDER_STATUS_PORT, "/status/share") {
                base["share"] = share;
            }
        }
        if prov {
            // Provider base carries the served side (peers, served counters, `served` ledger
            // rows); the gateway process holds the consumed side + `used` ledger rows. Pull the
            // gateway's full status once and overlay both its economy and its transfers.
            let gw_status = if gw { fetch_status(GATEWAY_STATUS_PORT, "/status") } else { None };
            let mut merged = false;
            if let Some(gw) = gw_status {
                merge_economy(&mut base, gw.get("economy").cloned().unwrap_or_default());
                merge_known_providers(&mut base, gw.get("network").cloned().unwrap_or_default());
                if let Some(gw_transfers) = gw.get("transfers").cloned() {
                    merge_transfers(&mut base, gw_transfers);
                    merged = true;
                }
            }
            if !merged {
                // The consumed side lives in the gateway; without its reading, report it as unknown
                // rather than the provider's zeros, which the UI would take for a counter reset.
                unknown_consumed_side(&mut base);
            }
        }
        Some(base)
    })
    .await
    .ok()
    .flatten())
}

/// Merge the gateway's economy view (earned reputation + give-side credit) into the provider
/// snapshot's `economy` block: reputation is consumer-only, so it always comes from the
/// gateway; credit entries are unioned by libp2p id (provider's take-side rate_cap wins).
/// The models and providers the app shows are the ones its gateway can route to: union the gateway's
/// view into the provider-based snapshot (the provider role may know of fewer, or none, which left
/// the Models view without providers, reputation or trust while sharing).
fn merge_known_providers(base: &mut serde_json::Value, gw_net: serde_json::Value) {
    let Some(net) = base.get_mut("network") else { return };
    if let Some(gw_models) = gw_net.get("known_models").and_then(|v| v.as_array()) {
        let mut models: Vec<serde_json::Value> = net.get("known_models").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        for m in gw_models {
            if !models.contains(m) {
                models.push(m.clone());
            }
        }
        net["known_models"] = serde_json::Value::Array(models);
    }
    if let Some(gw_provs) = gw_net.get("known_providers").and_then(|v| v.as_array()) {
        let key = |p: &serde_json::Value| {
            (
                p.get("libp2p_peer_id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                p.get("model_id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            )
        };
        let mut provs: Vec<serde_json::Value> = net.get("known_providers").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let mut seen: std::collections::HashSet<(String, String)> = provs.iter().map(key).collect();
        for p in gw_provs {
            if seen.insert(key(p)) {
                provs.push(p.clone());
            }
        }
        net["known_providers"] = serde_json::Value::Array(provs);
    }
}

fn merge_economy(base: &mut serde_json::Value, gw_econ: serde_json::Value) {
    let Some(econ) = base.get_mut("economy") else { return };
    // Reputation: take the gateway's list wholesale (the provider role never has any).
    if let Some(rep) = gw_econ.get("reputation").cloned() {
        econ["reputation"] = rep;
    }
    if let Some(avg) = gw_econ.get("avg_reputation").cloned() {
        econ["avg_reputation"] = avg;
    }
    // Credit: union by libp2p_peer_id, keeping the base (provider take-side, has rate_cap).
    if let (Some(base_credit), Some(gw_credit)) = (
        econ.get("credit").and_then(|c| c.as_array()).cloned(),
        gw_econ.get("credit").and_then(|c| c.as_array()).cloned(),
    ) {
        let mut seen: std::collections::HashSet<String> = base_credit
            .iter()
            .filter_map(|c| c.get("libp2p_peer_id").and_then(|v| v.as_str()).map(String::from))
            .collect();
        let mut merged = base_credit;
        for c in gw_credit {
            if let Some(id) = c.get("libp2p_peer_id").and_then(|v| v.as_str()) {
                if seen.insert(id.to_string()) {
                    merged.push(c);
                }
            }
        }
        econ["credit"] = serde_json::Value::Array(merged);
    }
}

/// Null out the consumed-side counters (they belong to the gateway process) when its status
/// couldn't be read this poll.
fn unknown_consumed_side(base: &mut serde_json::Value) {
    if let Some(t) = base.get_mut("transfers") {
        for key in ["requests_consumed", "tokens_consumed", "consumed_per_model", "credits_spent"] {
            t[key] = serde_json::Value::Null;
        }
    }
}

/// Merge the gateway's transfer counters (the consumed side + `used` ledger rows) into the
/// provider base snapshot's `transfers` block. The provider process only ever fills the served
/// side, so consumed fields are taken wholesale from the gateway; the two roles' recent-ledger
/// rings (provider = `served`, gateway = `used`) are concatenated and re-sorted newest-first.
fn merge_transfers(base: &mut serde_json::Value, gw: serde_json::Value) {
    let Some(t) = base.get_mut("transfers") else { return };
    // The consumed side, including the credits spent on it, lives in the gateway process.
    for key in ["requests_consumed", "tokens_consumed", "consumed_per_model", "credits_spent"] {
        if let Some(v) = gw.get(key).cloned() {
            t[key] = v;
        }
    }
    // Union the recent-ledger rings and sort by timestamp descending (newest first).
    let mut rows: Vec<serde_json::Value> = t
        .get("recent")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    if let Some(gw_rows) = gw.get("recent").and_then(|r| r.as_array()) {
        rows.extend(gw_rows.iter().cloned());
    }
    rows.sort_by(|a, b| {
        let ta = a.get("ts_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        let tb = b.get("ts_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        tb.cmp(&ta)
    });
    t["recent"] = serde_json::Value::Array(rows);
}

/// Blocking loopback GET of the status endpoint. Zero-dep: the agent serves tiny
/// `Connection: close` JSON, so a `std::net::TcpStream` request/read is enough (no reqwest
/// needed for a same-host call, and it avoids CORS in the webview entirely).
fn fetch_status(port: u16, path: &str) -> Option<serde_json::Value> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(3))).ok()?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n",
        status_token(),
    )
    .ok()?;
    let mut raw = String::new();
    stream.read_to_string(&mut raw).ok()?;
    let body = raw.split("\r\n\r\n").nth(1)?;
    serde_json::from_str(body).ok()
}

/// Probe the host's CPU / RAM / GPU(s) for the system panel (like LM Studio's). Runs stock
/// system tools off the UI thread; never fails (unknown fields fall back gracefully).
#[tauri::command]
async fn system_info() -> hostinfo::SystemInfo {
    tauri::async_runtime::spawn_blocking(hostinfo::probe).await.unwrap_or_else(|_| {
        hostinfo::SystemInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            cpu: "Unknown".into(),
            ram_bytes: 0,
            gpus: vec![],
        }
    })
}

/// Probe the local engines right now (the agent crate's own concurrent detection).
#[tauri::command]
async fn detect_engines_now() -> Vec<EngineView> {
    tauri::async_runtime::spawn_blocking(|| {
        openhydra_agent::detect::detect_engines()
            .into_iter()
            .map(|e| EngineView {
                label: e.label.to_string(),
                url: e.url.to_string(),
                models: e.models.into_iter().map(|m| m.engine_ref).collect(),
            })
            .collect()
    })
    .await
    .unwrap_or_default()
}

/// What an install *would* do — resolved without running, for the consent UI (B5).
#[derive(Serialize)]
struct InstallPlan {
    engine: String,
    /// False when there's no Tier-1 recipe for this engine/OS (UI shows "Guided install").
    supported: bool,
    /// One-line "what will run, from where" for the consent prompt.
    summary: String,
    /// True only where the exact commands are vendor-verified on this OS.
    verified: bool,
    /// Already answering on its port → Install is a no-op.
    already_installed: bool,
    /// A blocking prereq (e.g. missing Homebrew), with an actionable message.
    blocker: Option<String>,
    /// This engine also offers a headless CLI install on this OS → the UI shows an app/CLI toggle.
    cli_available: bool,
}

/// Resolve (without executing) what installing `engine` (in flavour `variant`) would do — powers
/// the consent modal. `variant` is None/"app"/"cli"; the app is the default where both exist.
#[tauri::command]
async fn install_plan(engine: String, accel: Option<String>, variant: Option<String>) -> InstallPlan {
    let fallback = InstallPlan {
        engine: engine.clone(),
        supported: false,
        summary: "install planning failed".into(),
        verified: false,
        already_installed: false,
        blocker: Some("internal error".into()),
        cli_available: false,
    };
    tauri::async_runtime::spawn_blocking(move || {
        let Some(os) = installer::Os::current() else {
            return InstallPlan {
                engine,
                supported: false,
                summary: "this OS isn't supported by the installer".into(),
                verified: false,
                already_installed: false,
                blocker: Some("unsupported OS".into()),
                cli_available: false,
            };
        };
        let already_installed = installer::already_installed(&engine);
        let accel = installer::Accel::from_str_opt(accel.as_deref());
        let variant = installer::Variant::from_str_opt(variant.as_deref());
        let cli_available = installer::has_cli_variant(&engine, os);
        match installer::recipe_for_variant(&engine, os, accel, variant) {
            Ok(r) => InstallPlan {
                supported: true,
                summary: r.summary,
                verified: r.verified,
                already_installed,
                blocker: installer::prereq_blocker(&engine, os),
                cli_available,
                engine,
            },
            // The blocker carries the "can't install here" message; leaving summary empty avoids
            // the UI rendering the same text twice (blocker + summary).
            Err(e) => InstallPlan {
                supported: false,
                summary: String::new(),
                verified: false,
                already_installed,
                blocker: Some(e),
                cli_available,
                engine,
            },
        }
    })
    .await
    .unwrap_or(fallback)
}

/// Install a Tier-1 engine, streaming progress to the webview as `install://progress` events
/// (`phase`/`log`/`done`/`error`). Idempotent (a detect-first hit returns immediately); refuses
/// on a prereq blocker. The heavy work runs off the UI thread.
#[tauri::command]
async fn install_engine(app: tauri::AppHandle, engine: String, accel: Option<String>, variant: Option<String>) -> Result<(), String> {
    let os = installer::Os::current().ok_or_else(|| "unsupported OS for the installer".to_string())?;
    // EVERYTHING here runs inside spawn_blocking. `already_installed` builds an
    // `openhydra_agent::ReqwestClient` (reqwest::blocking, which owns a tokio runtime); creating
    // or dropping that in the async command context panics ("Cannot drop a runtime in a context
    // where blocking is not allowed"). A blocking-pool thread has no runtime context, so it's safe.
    // Every terminal path emits a `done`/`error` event so the UI overlay can never hang silently.
    tauri::async_runtime::spawn_blocking(move || {
        if installer::already_installed(&engine) {
            installer::emit_done(&app, &engine, format!("{engine} is already installed"));
            return Ok(());
        }
        if let Some(blocker) = installer::prereq_blocker(&engine, os) {
            installer::emit_error(&app, &engine, blocker.clone());
            return Err(blocker);
        }
        let accel = installer::Accel::from_str_opt(accel.as_deref());
        let variant = installer::Variant::from_str_opt(variant.as_deref());
        let recipe = match installer::recipe_for_variant(&engine, os, accel, variant) {
            Ok(r) => r,
            Err(e) => {
                installer::emit_error(&app, &engine, e.clone());
                return Err(e);
            }
        };
        // Downloads and runs an installer: confirmed natively, outside the page.
        if !native_confirm(
            &app,
            "Install engine?",
            &format!("Download and install {}? It's fetched from its official source and installed on this machine.", dialog_text(&engine, 40)),
            "Install",
        ) {
            installer::emit_error(&app, &engine, "Install cancelled".to_string());
            return Err(CANCELLED.into());
        }
        let default_model = recipe.default_model;
        let done_msg = recipe.completion_message();
        let eng = engine.clone();
        let result = installer::run_recipe(&app, &recipe).and_then(|_| match default_model {
            Some(m) => installer::pull_and_warm(&app, &eng, m),
            None => {
                installer::emit_done(&app, &eng, done_msg.clone());
                Ok(())
            }
        });
        if let Err(e) = &result {
            installer::emit_error(&app, &eng, e.clone());
        }
        result
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Which engines are installed on disk (present, regardless of whether their server runs). Powers
/// the "Run" CTA for installed-but-idle engines. `reqwest`-free, but cheap fs/PATH checks still run
/// off the UI thread for consistency.
#[tauri::command]
async fn installed_engines() -> Vec<String> {
    tauri::async_runtime::spawn_blocking(|| {
        ["ollama", "lm-studio", "llama.cpp", "comfyui", "vllm", "exo"]
            .into_iter()
            .filter(|e| installer::installed_on_disk(e))
            .map(String::from)
            .collect()
    })
    .await
    .unwrap_or_default()
}

/// Start an installed engine's local server (the "Run" CTA). Best-effort; errors for engines that
/// need a model/cluster arg. `start_lm_studio_core` builds a reqwest client, so run off-thread.
#[tauri::command]
async fn run_engine(engine: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || installer::run_engine(&engine))
        .await
        .map_err(|e| e.to_string())?
}

/// One chat message from the UI (Chat/Code views).
#[derive(Deserialize, Clone)]
struct ChatMsgIn {
    role: String,
    content: String,
}

/// The most a streamed chat reply may carry into the chat view (room for an image as a data URL).
const MAX_CHAT_STREAM_BYTES: usize = 64 * 1024 * 1024;

/// Chats being streamed, by the id the UI gave them: setting the flag stops reading (the gateway
/// sees the connection close and stops the provider's generation).
static CHAT_STREAMS: std::sync::LazyLock<Mutex<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Stream a chat completion through the local gateway: each answer fragment goes to `on_event` as
/// `{ "kind": "delta", "text" }` as it arrives, then one `{ "kind": "done", "openhydra", "usage",
/// "model", "finish_reason" }`, or `{ "kind": "error", "message" }` (the gateway's own error text, as
/// [`gateway_error`] formats it). Resolves when the stream ends. `chat_stop(stream_id)` ends it early.
#[tauri::command]
async fn chat_completion_stream(
    state: tauri::State<'_, AppState>,
    stream_id: String,
    model: String,
    messages: Vec<ChatMsgIn>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    on_event: tauri::ipc::Channel<serde_json::Value>,
) -> Result<(), String> {
    let port = state.settings.lock().map_err(|e| e.to_string())?.gateway_port;
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages.iter().map(|m| serde_json::json!({ "role": m.role, "content": m.content })).collect::<Vec<_>>(),
        "max_tokens": max_tokens.unwrap_or(4096),
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    if let Some(t) = temperature.filter(|t| t.is_finite() && (0.0..=2.0).contains(t)) {
        body["temperature"] = serde_json::json!(t);
    }
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    if let Ok(mut m) = CHAT_STREAMS.lock() {
        m.insert(stream_id.clone(), stop.clone());
    }
    let id = stream_id.clone();
    let joined = tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        // Every exit sends a final "end", so the UI knows when the last event has arrived: Tauri
        // delivers large channel messages (an image as a data URL) after the command returns.
        let end = |on_event: &tauri::ipc::Channel<serde_json::Value>| {
            let _ = on_event.send(serde_json::json!({ "kind": "end" }));
        };
        let client = gateway_client()?;
        // Open the request on its own thread, so Stop works while the gateway is still choosing a
        // model and provider (before it sends anything back): the chat ends at once, and the
        // response is dropped (closing the connection) when it arrives.
        let (rtx, rrx) = std::sync::mpsc::sync_channel(1);
        {
            let url = format!("http://127.0.0.1:{port}/v1/chat/completions");
            let body = body.to_string();
            std::thread::spawn(move || {
                let _ = rtx.send(client.post_stream_any_status(&url, &body));
            });
        }
        let opened = loop {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = on_event.send(serde_json::json!({ "kind": "stopped" }));
                end(&on_event);
                return Ok(());
            }
            match rrx.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(r) => break r,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    let _ = on_event.send(serde_json::json!({ "kind": "error", "message": "gateway: the request ended unexpectedly" }));
                    end(&on_event);
                    return Ok(());
                }
            }
        };
        let lines = match opened {
            Err(e) => {
                let _ = on_event.send(serde_json::json!({ "kind": "error", "message": format!("gateway: {e}") }));
                end(&on_event);
                return Ok(());
            }
            Ok(Err((status, text))) => {
                let _ = on_event.send(serde_json::json!({ "kind": "error", "message": gateway_error(status, &text) }));
                end(&on_event);
                return Ok(());
            }
            Ok(Ok(lines)) => lines,
        };
        // Read lines on their own thread so Stop takes effect at once, even while the stream is
        // quiet (a model loading sends only a keep-alive every 15 s). The reader ends at the next
        // line once we stop listening, which drops the connection and stops the generation.
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<String, String>>(256);
        std::thread::spawn(move || {
            for line in lines {
                if tx.send(line.map_err(|e| e.to_string())).is_err() {
                    break;
                }
            }
        });
        let mut done = false;
        let mut received = 0usize;
        loop {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = on_event.send(serde_json::json!({ "kind": "stopped" }));
                break;
            }
            let line = match rx.recv_timeout(std::time::Duration::from_millis(200)) {
                Ok(Ok(l)) => l,
                Ok(Err(e)) => {
                    let _ = on_event.send(serde_json::json!({ "kind": "error", "message": format!("gateway: {e}") }));
                    break;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            };
            let Some(data) = line.strip_prefix("data:").map(str::trim) else { continue };
            // More than the chat view can hold: stop reading (the generation stops with it) and say
            // why, rather than freeze the window (ADVERSARIAL item 11).
            received = received.saturating_add(data.len());
            if received > MAX_CHAT_STREAM_BYTES {
                let _ = on_event.send(serde_json::json!({ "kind": "error", "message": "the reply is too long to show here (over 64 MB); stopped" }));
                break;
            }
            if data == "[DONE]" {
                done = true;
                break;
            }
            let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) else { continue };
            // A window that went away (reload) stops listening: stop reading too.
            if stream_events(&chunk).into_iter().any(|ev| on_event.send(ev).is_err()) {
                break;
            }
        }
        if !done {
            let _ = on_event.send(serde_json::json!({ "kind": "incomplete" }));
        }
        end(&on_event);
        Ok(())
    })
    .await;
    if let Ok(mut m) = CHAT_STREAMS.lock() {
        m.remove(&id);
    }
    joined.map_err(|e| e.to_string())?
}

/// The UI events for one SSE chunk of `/v1/chat/completions`: the text delta, and, on the final
/// chunk, the summary (telemetry, usage, model) or the gateway's error.
fn stream_events(chunk: &serde_json::Value) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    if let Some(err) = chunk.get("error") {
        let kind = err["type"].as_str().unwrap_or("");
        let msg = err["message"].as_str().unwrap_or("the provider stopped with an error");
        out.push(serde_json::json!({ "kind": "error", "message": format!("gateway 502 {kind}: {msg}") }));
        return out;
    }
    let choice = &chunk["choices"][0];
    if let Some(t) = choice["delta"]["content"].as_str().filter(|t| !t.is_empty()) {
        out.push(serde_json::json!({ "kind": "delta", "text": t }));
    }
    if let Some(t) = choice["delta"]["reasoning_content"].as_str().filter(|t| !t.is_empty()) {
        out.push(serde_json::json!({ "kind": "reasoning", "text": t }));
    }
    let finish = choice["finish_reason"].as_str();
    if finish.is_some() || chunk.get("openhydra").is_some() {
        out.push(serde_json::json!({
            "kind": "done",
            "finish_reason": finish,
            "openhydra": chunk.get("openhydra").cloned().unwrap_or_default(),
            "usage": chunk.get("usage").cloned().unwrap_or_default(),
            "model": chunk.get("model").cloned().unwrap_or_default(),
        }));
    }
    out
}

/// Stop every chat being streamed. Run when the page (re)loads: a reloaded page doesn't know the
/// stream ids, and the stream would otherwise read to the end (spending credits) into a channel
/// nothing listens to (ADVERSARIAL item 4).
fn stop_all_chat_streams() -> usize {
    let streams = CHAT_STREAMS.lock().map(|m| m.values().cloned().collect::<Vec<_>>()).unwrap_or_default();
    for flag in &streams {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    streams.len()
}

/// Stop a chat started with `chat_completion_stream`.
#[tauri::command]
fn chat_stop(stream_id: String) {
    if let Some(flag) = CHAT_STREAMS.lock().ok().and_then(|m| m.get(&stream_id).cloned()) {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Run a (non-streaming) chat completion through the LOCAL gateway — so the Chat/Code
/// views exercise the same network path as any OpenAI client pointed at this node:
/// gateway → discover → provider → engine. Returns the gateway's full JSON (including
/// the `openhydra` metrics block the UI shows under each reply). Rust-side HTTP so the
/// webview needs no CORS story.
#[tauri::command]
async fn chat_completion(
    state: tauri::State<'_, AppState>,
    model: String,
    messages: Vec<ChatMsgIn>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
) -> Result<serde_json::Value, String> {
    let port = state.settings.lock().map_err(|e| e.to_string())?.gateway_port;
    // Default when the UI doesn't supply one: a generous chat budget, NOT the engine's own
    // default. LM Studio (and others) fall back to ~512 tokens when `max_tokens` is absent —
    // far too small for a reasoning model, which spends that entirely on hidden chain-of-thought
    // and never reaches the answer ("no visible text"). 4096 lets it finish thinking AND reply.
    const DEFAULT_CHAT_MAX_TOKENS: u32 = 4096;
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages
            .iter()
            .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
            .collect::<Vec<_>>(),
        "max_tokens": max_tokens.unwrap_or(DEFAULT_CHAT_MAX_TOKENS),
    });
    // Sent when the UI passes one (the user set it): temperature 0 makes answers repeatable, and lets
    // the gateway's audits replay the request (it needs an explicit 0; an absent one may sample).
    if let Some(t) = temperature.filter(|t| t.is_finite() && (0.0..=2.0).contains(t)) {
        body["temperature"] = serde_json::json!(t);
    }
    tauri::async_runtime::spawn_blocking(move || {
        let client = gateway_client()?;
        let (status, resp) = client
            .post_json_any_status(&format!("http://127.0.0.1:{port}/v1/chat/completions"), &body.to_string())
            .map_err(|e| format!("gateway: {e}"))?;
        if !(200..300).contains(&status) {
            return Err(gateway_error(status, &resp));
        }
        serde_json::from_str::<serde_json::Value>(&resp).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

// ── Connectors: detect installed coding tools + one-click wire them to the gateway ──

/// Detection status for every known connector (installed? where?). Read-only.
#[tauri::command]
fn connector_status() -> Vec<connectors::ConnectorStatus> {
    connectors::statuses()
}

/// The gateway origin the connectors point tools at (`http://127.0.0.1:<port>`).
fn gateway_origin(state: &tauri::State<'_, AppState>) -> Result<String, String> {
    let port = state.settings.lock().map_err(|e| e.to_string())?.gateway_port;
    Ok(format!("http://127.0.0.1:{port}"))
}

/// Preview (no write) what Connect would do for `key` — path, create/update, the full new content,
/// and any caveat. The UI shows this for confirmation before Apply.
#[tauri::command]
fn connector_preview(state: tauri::State<'_, AppState>, key: String) -> Result<connectors::ConnectPreview, String> {
    let origin = gateway_origin(&state)?;
    connectors::preview(&key, &origin)
}

/// Write the OpenHydra block into `key`'s config (backs up any existing file first). Called only
/// after the user confirms the preview. `models` declares specific network model ids in the tool's
/// own picker (opencode/pi/continue only); omit/empty ⇒ just `openhydra/auto`.
#[tauri::command(async)]
fn connector_apply(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    key: String,
    models: Option<Vec<String>>,
) -> Result<connectors::ConnectReport, String> {
    let origin = gateway_origin(&state)?;
    // Writes into another program's config file: say which one, and that it's backed up first.
    let preview = connectors::preview(&key, &origin)?;
    let msg = format!(
        "Connect {} to OpenHydra? This {} {} so it sends its requests to your local OpenHydra gateway. An existing file is backed up first; Disconnect restores it.",
        dialog_text(&preview.kind, 40),
        if preview.action == "create" { "creates" } else { "updates" },
        dialog_text(&preview.path, 160),
    );
    if !native_confirm(&app, "Connect tool?", &msg, "Connect") {
        return Err(CANCELLED.into());
    }
    connectors::apply_with_models(&key, &origin, &models.unwrap_or_default())
}

/// Un-wire `key`: restore the pristine pre-OpenHydra config from its backup, or delete a file we
/// created. The inverse of `connector_apply` (the Disconnect button).
#[tauri::command]
fn connector_disconnect(key: String) -> Result<connectors::DisconnectReport, String> {
    connectors::disconnect(&key)
}

// ── `openhydra` CLI-on-PATH (Layer 1) ──────────────────────────────────────────
// The bundled agent sidecar is the full CLI but isn't on PATH for app installs, so the Connectors
// "Terminal" snippets + `openhydra launch` fail with command-not-found. These expose an in-app
// "Install command-line tool" action (the VS Code model). See docs/CLI_ON_PATH_PLAN_v1.md.

/// Is `openhydra` runnable from a terminal, where from, and where would we install it? Read-only.
#[tauri::command]
fn cli_status() -> cli::CliStatus {
    cli::status()
}

/// Link/copy the bundled sidecar onto PATH as `openhydra`. On macOS this shows one admin prompt
/// (osascript) — run off the UI thread so the prompt doesn't freeze the window.
#[tauri::command]
async fn install_cli() -> Result<cli::InstallReport, String> {
    tauri::async_runtime::spawn_blocking(cli::install)
        .await
        .map_err(|e| format!("install task failed: {e}"))?
}

/// Remove the managed `openhydra` command.
#[tauri::command]
async fn uninstall_cli() -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(cli::uninstall)
        .await
        .map_err(|e| format!("uninstall task failed: {e}"))?
}

/// Open a tool's GUI (the App/Editor "Connect & Open" action): the OpenCode/Hermes desktop app or the
/// Continue/Claude editor, via the tool's `gui_target`. Best-effort — an Err leaves the (already
/// written) config in place and the UI toasts a manual-open fallback.
#[tauri::command]
fn open_gui(key: String) -> Result<(), String> {
    use openhydra_agent::connect::spec;
    use std::process::Command;
    let spec = spec(&key).ok_or_else(|| format!("unknown connector '{key}'"))?;
    let app = spec.gui_target.ok_or_else(|| format!("{key} has no GUI to open"))?;
    // Run + WAIT for the launcher to exit so we report real success. `open`/`code` return promptly
    // after handing off to LaunchServices, but `open -a` exits non-zero when the app is missing — a
    // bare spawn would mask that as success and the UI would falsely toast "opening…".
    let run = |mut cmd: Command, what: String| -> Result<(), String> {
        match cmd.status() {
            Ok(s) if s.success() => Ok(()),
            Ok(s) => Err(format!("{what} failed ({s})")),
            Err(e) => Err(format!("{what}: {e}")),
        }
    };
    // Editor tools: prefer the `code` CLI (via the GUI-PATH resolver) so an open workspace is reused;
    // fall back to macOS `open -a <app>`. App tools: `open -a <app>`.
    if matches!(spec.kind, openhydra_agent::connect::Kind::ContinueYaml | openhydra_agent::connect::Kind::ClaudeSettings) {
        if let Some(code) = crate::installer::resolve_program("code") {
            return run(Command::new(code), "launch VS Code".into());
        }
    }
    #[cfg(target_os = "macos")]
    {
        let mut c = Command::new("open");
        c.args(["-a", app]);
        run(c, format!("open {app}"))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = app;
        Err(format!("opening {key}'s GUI is only wired for macOS so far; open {app} manually"))
    }
}

/// "Test" button: a bounded 1-token `openhydra/auto` completion through the local gateway; returns
/// the model that served it, or an actionable error (gateway down / no provider).
#[tauri::command]
async fn connector_test(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let port = state.settings.lock().map_err(|e| e.to_string())?.gateway_port;
    let body = serde_json::json!({
        "model": "openhydra/auto",
        "max_tokens": 1,
        "messages": [{ "role": "user", "content": "ping" }],
    });
    tauri::async_runtime::spawn_blocking(move || {
        let client = gateway_client()?;
        // Read the body whatever the status: a refusal (no provider, none at the authenticity floor
        // set in Settings) says why, and isn't the gateway being unreachable (review round 4 G4-9).
        let (status, resp) = client
            .post_json_any_status(&format!("http://127.0.0.1:{port}/v1/chat/completions"), &body.to_string())
            .map_err(|e| format!("gateway unreachable on :{port} — is OpenHydra sharing/serving? ({e})"))?;
        let v: serde_json::Value = serde_json::from_str(&resp).map_err(|_| format!("gateway answered {status}: {resp}"))?;
        if let Some(err) = v.get("error") {
            return Err(err.get("message").and_then(|m| m.as_str()).unwrap_or("gateway error").to_string());
        }
        if !(200..300).contains(&status) {
            return Err(format!("gateway answered {status}"));
        }
        Ok(v.get("model").and_then(|m| m.as_str()).unwrap_or("ok").to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// One web-search hit handed to the UI.
#[derive(Serialize)]
struct SearchHit {
    title: String,
    url: String,
    snippet: String,
}

/// Query the operator's SearXNG-compatible endpoint (`/search?format=json`) for the chat
/// "web" toggle. Backend-side HTTP (no webview CORS); top 5 hits.
#[tauri::command]
async fn web_search(
    state: tauri::State<'_, AppState>,
    query: String,
) -> Result<Vec<SearchHit>, String> {
    let base = state.settings.lock().map_err(|e| e.to_string())?.search_url.trim().to_string();
    if base.is_empty() {
        return Err("no search endpoint configured (Settings → Search URL)".into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        use openhydra_agent::adapter::HttpClient;
        let client = openhydra_agent::ReqwestClient::new().map_err(|e| e.to_string())?;
        let mut encoded = String::new();
        for b in query.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    encoded.push(b as char)
                }
                _ => encoded.push_str(&format!("%{b:02X}")),
            }
        }
        let url = format!("{}/search?q={encoded}&format=json", base.trim_end_matches('/'));
        let body = client.get(&url).map_err(|e| format!("search: {e}"))?;
        let json: serde_json::Value =
            serde_json::from_str(&body).map_err(|e| format!("search parse: {e}"))?;
        Ok(json["results"]
            .as_array()
            .map(|rs| {
                rs.iter()
                    .take(5)
                    .map(|r| SearchHit {
                        title: r["title"].as_str().unwrap_or("").to_string(),
                        url: r["url"].as_str().unwrap_or("").to_string(),
                        snippet: r["content"].as_str().unwrap_or("").to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Is the local gateway answering? (backend-side to avoid webview CORS)
#[tauri::command]
async fn gateway_health(state: tauri::State<'_, AppState>) -> Result<bool, ()> {
    let port = state.settings.lock().expect("settings lock").gateway_port;
    Ok(tauri::async_runtime::spawn_blocking(move || {
        use openhydra_agent::adapter::HttpClient;
        gateway_client()
            .map(|c| c.get(&format!("http://127.0.0.1:{port}/v1/models")).is_ok())
            .unwrap_or(false)
    })
    .await
    .unwrap_or(false))
}

/// Reject settings the app could act on harmfully: they choose who this node talks to (bootstraps),
/// where the provider sends the network's prompts (engine endpoint) and what the chat fetches
/// (search URL). Errors are shown to the user as "Save failed: …".
/// The changes in `next` that send this node's traffic somewhere new or widen what it does for
/// others, each as a sentence for the confirmation dialog. Empty when nothing widens.
fn settings_widening(prev: &Settings, next: &Settings) -> Vec<String> {
    let mut out = Vec::new();
    let added: Vec<String> = next
        .bootstraps
        .iter()
        .map(|b| b.trim())
        .filter(|b| !b.is_empty() && !prev.bootstraps.iter().any(|p| p.trim() == *b))
        .map(|b| dialog_text(b, 120))
        .collect();
    if !added.is_empty() {
        out.push(format!("Connect to new bootstrap peers: {}", added.join(", ")));
    }
    let endpoint = next.engine_endpoint.trim();
    if !endpoint.is_empty() && endpoint != prev.engine_endpoint.trim() {
        out.push(format!("Send the requests your machine serves to this engine: {}", dialog_text(endpoint, 120)));
    }
    let search = next.search_url.trim();
    if !search.is_empty() && search != prev.search_url.trim() {
        out.push(format!("Use this web search service: {}", dialog_text(search, 120)));
    }
    if next.peer_relay && !prev.peer_relay {
        out.push("Relay other peers' traffic through your machine.".to_string());
    }
    if next.public_network && !prev.public_network {
        out.push("Join OpenHydra's public network (its bootstrap nodes and relays).".to_string());
    }
    if tokenizer_downloads_on(next) && !tokenizer_downloads_on(prev) {
        out.push("Download model tokenizers from Hugging Face the first time each model is used (huggingface.co then learns which models you use).".to_string());
    }
    if next.audit_replay && !prev.audit_replay {
        out.push("Send some of your own requests to a second provider to check answers.".to_string());
    }
    out
}

fn validate_settings(s: &Settings) -> Result<(), String> {
    if s.bootstraps.len() > 32 {
        return Err("at most 32 bootstrap addresses".into());
    }
    for b in &s.bootstraps {
        let b = b.trim();
        // Parsed exactly as the node parses it at startup: a bad one would stop sharing and the
        // gateway from starting at all.
        let ok = b.starts_with('/') && b.len() <= 256 && openhydra_agent::check_bootstrap_addr(b).is_ok();
        if !ok {
            return Err(format!(
                "bootstrap address isn't a multiaddr like /ip4/1.2.3.4/tcp/4001/p2p/12D3Koo…: {}",
                b.chars().take(80).collect::<String>().escape_debug()
            ));
        }
    }
    if s.gateway_port < 1024 {
        return Err("gateway port must be 1024–65535".into());
    }
    if !matches!(s.min_authenticity.as_str(), "off" | "A1" | "A2") {
        return Err("minimum authenticity must be off, A1 or A2".into());
    }
    if s.engine_kind != "auto" && !ENGINE_KINDS.contains(&s.engine_kind.as_str()) {
        return Err(format!("unknown engine: {}", s.engine_kind.escape_debug()));
    }
    for (label, url) in [("engine endpoint", &s.engine_endpoint), ("search URL", &s.search_url)] {
        let url = url.trim();
        if url.is_empty() {
            continue;
        }
        match tauri::Url::parse(url) {
            Ok(u) if matches!(u.scheme(), "http" | "https") && u.host_str().is_some() && u.username().is_empty() && u.password().is_none() => {}
            _ => return Err(format!("{label} must be an http(s) URL without a user name or password")),
        }
    }
    if s.device_name.chars().count() > 64 || s.device_name.chars().any(char::is_control) {
        return Err("device name: at most 64 characters, no control characters".into());
    }
    Ok(())
}

#[tauri::command(async)]
fn save_settings(app: tauri::AppHandle, state: tauri::State<'_, AppState>, mut settings: Settings) -> Result<(), String> {
    validate_settings(&settings)?;
    let prev = state.settings.lock().map_err(|e| e.to_string())?.clone();
    let mut widening = settings_widening(&prev, &settings);
    // Joining the public network makes what is shared with everyone public too: say what.
    if settings.public_network && !prev.public_network {
        let policy = openhydra_agent::SharePolicy::load(&share_policy_path()).unwrap_or_else(|_| openhydra_agent::SharePolicy::share_nothing());
        let shared = global_exposure(&policy);
        if !shared.is_empty() {
            widening.push(format!("Make what you share with everyone visible on the public network: {}.", shared.describe()));
        }
    }
    if !widening.is_empty()
        && !native_confirm(
            &app,
            "Change network settings?",
            &format!(
                "{}\n\nOnly continue if you made this change.",
                widening.iter().map(|w| format!("• {w}")).collect::<Vec<_>>().join("\n")
            ),
            "Save",
        )
    {
        return Err(CANCELLED.into());
    }
    // Going private with OpenHydra's own bootstraps in the list: they won't be used (the agent drops
    // them); offer to take them out of the list too (D3).
    if prev.public_network && !settings.public_network {
        let public: Vec<String> = settings.bootstraps.iter().filter(|b| openhydra_agent::is_openhydra_bootstrap(b)).cloned().collect();
        if !public.is_empty()
            && native_choice(
                &app,
                "Remove OpenHydra's bootstrap peers?",
                &format!(
                    "Your list includes OpenHydra's public bootstrap peers. A private deployment doesn't use them either way.\n\n{}",
                    public.iter().map(|b| dialog_text(b, 120)).collect::<Vec<_>>().join("\n")
                ),
                "Remove",
                "Keep",
            )
        {
            settings.bootstraps.retain(|b| !openhydra_agent::is_openhydra_bootstrap(b));
        }
    }
    let mut cur = state.settings.lock().map_err(|e| e.to_string())?;
    // Preserve fields the settings FORM doesn't own — taking the payload verbatim would reset them.
    // `sharing_enabled` is managed by start/stop_provider (a save must not clear a live sharing intent
    // → it would break resume-on-launch); `schema_version` is managed by load_settings' migration.
    // `resume_on_launch` IS surfaced by the settings toggle now, so its payload value is honoured.
    settings.sharing_enabled = cur.sharing_enabled;
    settings.schema_version = cur.schema_version;
    settings.share_policy_initialized = cur.share_policy_initialized;
    settings.sharing_confirmed = cur.sharing_confirmed;
    // `shared_models` is the policy MIRROR now, owned by the Share view (save_share_policy /
    // reset_share_policy), not the settings form. Preserve it so a stale echoed payload from the
    // form can't clobber the freshly-synced mirror (L4). The form no longer sends it either.
    settings.shared_models = cur.shared_models.clone();
    store_settings(&settings)?;
    *cur = settings;
    Ok(())
}

// ── Menubar/tray menu ──
// Mirrors the in-app tray preview: Open · Sharing (checkable) · Model · ▲ served · ▼ used · Quit.
// macOS auto-shows a tray menu on left-click, so there's no just-in-time hook to refresh it first —
// instead a lightweight background thread rebuilds the menu every few seconds (and the Sharing
// handler rebuilds it immediately) so it's already current when the user clicks.

/// What the tray menu displays. `sharing` = the provider role is running; the counters + model come
/// from the same `--status-bind` snapshot the UI polls (a loopback GET, no extra deps).
struct TrayStats {
    sharing: bool,
    model: String,
    served: u64,
    used: u64,
}

fn tray_stats(state: &AppState) -> TrayStats {
    let sharing = state.provider.lock().map(|r| r.status.running).unwrap_or(false);
    let gw = state.gateway.lock().map(|r| r.status.running).unwrap_or(false);
    let (mut model, mut served, mut used) = (String::new(), 0u64, 0u64);
    if sharing || gw {
        let base_port = if sharing { PROVIDER_STATUS_PORT } else { GATEWAY_STATUS_PORT };
        if let Some(mut snap) = fetch_status(base_port, "/status") {
            if sharing && gw {
                if let Some(g) = fetch_status(GATEWAY_STATUS_PORT, "/status") {
                    merge_transfers(&mut snap, g.get("transfers").cloned().unwrap_or_default());
                }
            }
            let t = snap.get("transfers");
            served = t.and_then(|t| t.get("tokens_served")).and_then(|v| v.as_u64()).unwrap_or(0);
            used = t.and_then(|t| t.get("tokens_consumed")).and_then(|v| v.as_u64()).unwrap_or(0);
            // Prefer a model THIS node serves; else the first model seen on the network.
            if sharing {
                if let Some(k) = t
                    .and_then(|t| t.get("per_model"))
                    .and_then(|v| v.as_object())
                    .and_then(|o| o.keys().next())
                {
                    model = k.clone();
                }
            }
            if model.is_empty() {
                if let Some(first) = snap
                    .get("network")
                    .and_then(|n| n.get("known_models"))
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.iter().find_map(|v| v.as_str()))
                {
                    model = first.to_string();
                }
            }
        }
    }
    TrayStats { sharing, model, served, used }
}

/// Compact human count for the menu (1280 → "1k", 2_400_000 → "2.4M").
fn fmt_count(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{}k", ((n as f64) / 1_000.0).round() as u64)
    } else {
        format!("{:.1}M", (n as f64) / 1_000_000.0)
    }
}

/// Build the tray menu for a given snapshot. Actionable: Open, Sharing (checkable), Quit. The
/// Model/served/used rows are disabled — glanceable info, matching the in-app preview.
fn build_tray_menu<R: tauri::Runtime, M: tauri::Manager<R>>(
    m: &M,
    s: &TrayStats,
) -> tauri::Result<tauri::menu::Menu<R>> {
    use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
    let open = MenuItem::with_id(m, "open", "Open OpenHydra", true, None::<&str>)?;
    let sharing = CheckMenuItem::with_id(m, "sharing", "Sharing", true, s.sharing, None::<&str>)?;
    let model = MenuItem::with_id(
        m,
        "model",
        format!("Model · {}", if s.model.is_empty() { "—" } else { &s.model }),
        false,
        None::<&str>,
    )?;
    let served =
        MenuItem::with_id(m, "served", format!("▲ {} served", fmt_count(s.served)), false, None::<&str>)?;
    let used =
        MenuItem::with_id(m, "used", format!("▼ {} used", fmt_count(s.used)), false, None::<&str>)?;
    let quit = MenuItem::with_id(m, "quit", "Quit OpenHydra", true, None::<&str>)?;
    Menu::with_items(
        m,
        &[
            &open,
            &PredefinedMenuItem::separator(m)?,
            &sharing,
            &model,
            &PredefinedMenuItem::separator(m)?,
            &served,
            &used,
            &PredefinedMenuItem::separator(m)?,
            &quit,
        ],
    )
}

/// Rebuild + install a fresh tray menu (call on the main thread — menu events already run there).
fn refresh_tray_menu<R: tauri::Runtime>(app: &tauri::AppHandle<R>) {
    let s = tray_stats(&app.state::<AppState>());
    if let Some(tray) = app.tray_by_id("main") {
        if let Ok(menu) = build_tray_menu(app, &s) {
            let _ = tray.set_menu(Some(menu));
        }
    }
}

// ── App wiring: tray, hide-on-close, kill children on exit ──

fn kill_all(state: &AppState) {
    state.provider.lock().expect("role lock").kill();
    state.gateway.lock().expect("role lock").kill();
}

/// The pages the app window may show: its own bundled UI (`tauri://localhost` on macOS and Linux,
/// `http(s)://tauri.localhost` on Windows) and, under `tauri dev`, the Vite dev server. A link or a
/// script can't load any other page into the window.
fn navigation_allowed(url: &tauri::Url) -> bool {
    match url.scheme() {
        "tauri" => url.host_str() == Some("localhost"),
        "http" | "https" => {
            url.host_str() == Some("tauri.localhost")
                || (cfg!(dev) && url.host_str() == Some("localhost") && url.port() == Some(1420))
        }
        "about" => url.as_str() == "about:blank",
        _ => false,
    }
}

fn main() {
    // Reap agents a crashed/killed previous instance left behind, then arm the signal
    // path so THIS instance can't leave any (see the pid-registry block above).
    install_signal_handlers();
    // Settle the share policy before anything can write desktop.json: whether this is a fresh install
    // (→ share nothing until the user chooses) is decided by desktop.json being absent, so it must be
    // checked first thing, not when the Share view is first opened. Then record that this install
    // has its own policy, and that a user who was already sharing has confirmed it.
    let mut initial_settings = load_settings();
    let share_policy_reset = std::sync::atomic::AtomicBool::new(false);
    let policy_path = ensure_valid_share_policy(&initial_settings, &share_policy_reset);
    if !initial_settings.sharing_enabled && !initial_settings.sharing_confirmed {
        drop_inherited_share_all(&policy_path);
    }
    let mut changed = false;
    if policy_path.exists() && !initial_settings.share_policy_initialized {
        initial_settings.share_policy_initialized = true;
        changed = true;
    }
    if initial_settings.sharing_enabled && !initial_settings.sharing_confirmed {
        initial_settings.sharing_confirmed = true;
        changed = true;
    }
    if changed {
        let _ = store_settings(&initial_settings);
    }
    tauri::Builder::default()
        // M2.1: single-instance MUST be first — it forwards a second launch's deep link / opened file
        // to the running instance (Win/Linux spawn a new process; the `deep-link` feature routes the
        // URL into `on_open_url`) instead of opening a duplicate window. On macOS the OS delivers to
        // the running app directly; here we just focus it.
        .plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            show_main_window(app);
            // Win/Linux: a second launch by a `.openhydra` file (or an `openhydra:` link the plugin
            // didn't already forward to on_open_url) carries it in argv. macOS: no-op.
            deliver_card_from_args(app, argv);
        }))
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            provider: Arc::new(Mutex::new(Role::new(&PROVIDER_PID))),
            gateway: Arc::new(Mutex::new(Role::new(&GATEWAY_PID))),
            settings: Mutex::new(initial_settings),
            resumed_on_launch: std::sync::atomic::AtomicBool::new(false),
            share_policy_reset,
            global_approval: Mutex::new(GlobalExposure::default()),
        })
        .manage(PendingCard(std::sync::Mutex::new(None)))
        .invoke_handler(tauri::generate_handler![
            get_state,
            identity_info,
            reveal_path,
            confirm_sharing,
            open_external,
            start_provider,
            restart_provider,
            start_gateway,
            stop_provider,
            stop_gateway,
            save_share_policy,
            read_share_policy,
            export_card,
            preview_card,
            import_card,
            list_cards,
            remove_imported_card,
            take_pending_card,
            list_swarms,
            create_swarm,
            swarm_enroll_request,
            preview_enroll_request,
            swarm_approve_member,
            swarm_remote_scope,
            swarm_revoke_member,
            preview_swarm_credential,
            swarm_accept_credential,
            forget_swarm,
            reset_share_policy,
            detect_engines_now,
            system_info,
            install_plan,
            install_engine,
            installed_engines,
            run_engine,
            gateway_health,
            chat_completion,
            chat_completion_stream,
            chat_stop,
            web_search,
            status_snapshot,
            save_settings,
            load_sessions,
            save_sessions,
            load_stats,
            save_stats,
            device_hostname,
            app_version,
            appimage_status,
            integrate_appimage,
            export_logs,
            connector_status,
            connector_preview,
            connector_apply,
            connector_disconnect,
            open_gui,
            connector_test,
            cli_status,
            install_cli,
            uninstall_cli,
        ])
        .setup(|app| {
            // Kill agents a crashed previous instance left behind. Here, not at the top of main():
            // a second launch exits inside the single-instance plugin before `setup` runs, so it
            // can't kill the running app's agents (their pids are in the same pidfile).
            sweep_stale_agents();
            // The local gateway key, before anything connects a tool or starts the gateway; then
            // re-wire tools connected by an older build (or with an older key), which the gateway
            // would now refuse. Off the main thread: it reads and writes a few config files.
            if gateway_api_key().is_some() {
                let port = app.state::<AppState>().settings.lock().map(|s| s.gateway_port).unwrap_or(16527);
                std::thread::spawn(move || {
                    for (tool, outcome) in openhydra_agent::connect::refresh_keys(&format!("http://127.0.0.1:{port}")) {
                        match outcome {
                            Ok(r) => eprintln!("openhydra: {tool} now presents the gateway key ({})", r.path),
                            Err(e) => eprintln!("openhydra: could not update {tool}'s config with the gateway key: {e}"),
                        }
                    }
                });
            }
            // The main window is created here rather than from the config so it can refuse to
            // navigate away from the app's own pages and refuse to open new windows.
            if let Some(cfg) = app.config().app.windows.first() {
                tauri::WebviewWindowBuilder::from_config(app.handle(), cfg)?
                    .on_navigation(|url| {
                        let ok = navigation_allowed(url);
                        if !ok {
                            eprintln!("openhydra: blocked navigation to {}", url.scheme());
                        }
                        ok
                    })
                    .on_new_window(|url, _| {
                        eprintln!("openhydra: blocked a new window ({})", url.scheme());
                        tauri::webview::NewWindowResponse::Deny
                    })
                    .on_page_load(|_, payload| {
                        if payload.event() == tauri::webview::PageLoadEvent::Started {
                            let n = stop_all_chat_streams();
                            if n > 0 {
                                eprintln!("openhydra: page reloaded — stopped {n} chat stream(s) it can no longer show");
                            }
                        }
                    })
                    .build()?;
            }
            // M2.1: card deep links. A cold launch by an `openhydra:` link surfaces via
            // `get_current`; subsequent links (app already running) via `on_open_url`. A `.openhydra`
            // FILE open arrives as `RunEvent::Opened` (below), so the two channels don't overlap.
            if let Ok(Some(urls)) = app.deep_link().get_current() {
                for u in &urls {
                    if let Some(text) = card_text_from_url(u) {
                        deliver_card(app.handle(), text);
                    }
                }
            }
            {
                let handle = app.handle().clone();
                app.deep_link().on_open_url(move |event| {
                    for u in event.urls() {
                        if let Some(text) = card_text_from_url(&u) {
                            deliver_card(&handle, text);
                        }
                    }
                });
            }
            // Win/Linux cold launch by a `.openhydra` file / `openhydra:` link arrives in the process
            // args (macOS uses Apple events, so this is a no-op there).
            deliver_card_from_args(app.handle(), std::env::args().skip(1));
            // Resume sharing if the user was sharing when they last quit (informed opt-out via
            // `resume_on_launch`). The provider is a child process killed on exit, so without this
            // every restart/update silently stopped sharing. The agent re-announces only models the
            // engine actually serves (re-probing each tick), so a not-yet-ready engine resolves itself.
            {
                let (enabled, resume) = app
                    .state::<AppState>()
                    .settings
                    .lock()
                    .map(|s| (s.sharing_enabled, s.resume_on_launch))
                    .unwrap_or((false, true));
                if enabled && resume {
                    match start_provider_now(app.state()) {
                        Ok(()) => {
                            app.state::<AppState>()
                                .resumed_on_launch
                                .store(true, std::sync::atomic::Ordering::Relaxed);
                            eprintln!("openhydra: resumed sharing on launch (was sharing at last quit)");
                        }
                        Err(e) => eprintln!("openhydra: resume sharing on launch failed: {e}"),
                    }
                }
            }
            use tauri::tray::TrayIconBuilder;
            let menu = build_tray_menu(app, &tray_stats(&app.state::<AppState>()))?;
            let mut tray = TrayIconBuilder::with_id("main")
                .menu(&menu)
                .show_menu_on_left_click(true)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "open" => show_main_window(app),
                    // Sharing = start/stop the provider role, same as the in-app toggle. Rebuild the
                    // menu right away so the checkmark reflects the new state.
                    "sharing" => {
                        let running = app
                            .state::<AppState>()
                            .provider
                            .lock()
                            .map(|r| r.status.running)
                            .unwrap_or(false);
                        // Nothing chosen to share (never confirmed, or the policy was reset): ask in
                        // the app instead of starting a provider that shares nothing.
                        let nothing_chosen = openhydra_agent::SharePolicy::load(&share_policy_path())
                            .map(|p| p.mode == openhydra_agent::share_policy::ShareMode::List && p.models.is_empty())
                            .unwrap_or(false);
                        if running {
                            stop_provider(app.state());
                        } else if let Err(e) = if nothing_chosen { Err(NOT_CONFIRMED.to_string()) } else { start_provider_now(app.state()) } {
                            if e == NOT_CONFIRMED {
                                // Ask in the app: show the window on the Share dialog.
                                show_main_window(app);
                                let _ = app.emit("open-share-dialog", ());
                            } else {
                                eprintln!("tray: start provider failed: {e}");
                            }
                        }
                        refresh_tray_menu(app);
                    }
                    "quit" => {
                        kill_all(&app.state::<AppState>());
                        app.exit(0);
                    }
                    _ => {}
                });
            // Menubar/tray icon: a background-less mark, not the cream-tiled app icon. On macOS it's
            // a *template* image (alpha-only silhouette) so it auto-inverts for light/dark menubars —
            // the brand mark is light, so a colored copy would vanish on a light menubar. Elsewhere
            // fall back to the window icon.
            #[cfg(target_os = "macos")]
            {
                if let Ok(icon) = tauri::image::Image::from_bytes(include_bytes!(
                    "../icons/trayTemplate.png"
                )) {
                    tray = tray.icon(icon).icon_as_template(true);
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                if let Some(icon) = app.default_window_icon() {
                    tray = tray.icon(icon.clone());
                }
            }
            tray.build(app)?;

            // Keep the menu current (sharing check, model, served/used) so it's fresh on click.
            // The status fetch is blocking, so it runs on this worker thread; the actual menu swap
            // is marshalled back to the main thread (required for menu/tray mutation on macOS).
            let handle = app.handle().clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let s = tray_stats(&handle.state::<AppState>());
                let h = handle.clone();
                let _ = handle.run_on_main_thread(move || {
                    if let Some(tray) = h.tray_by_id("main") {
                        if let Ok(menu) = build_tray_menu(&h, &s) {
                            let _ = tray.set_menu(Some(menu));
                        }
                    }
                });
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the window hides to tray; the agents keep serving.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .build(tauri::generate_context!())
        .expect("error building OpenHydra app")
        .run(|app, event| match event {
            tauri::RunEvent::Exit => kill_all(&app.state::<AppState>()),
            // macOS: clicking the Dock icon (or reopening the app) while its window is hidden or
            // minimised. Without this the click did nothing until the tray's Open.
            #[cfg(target_os = "macos")]
            tauri::RunEvent::Reopen { has_visible_windows, .. } => {
                if !has_visible_windows {
                    show_main_window(app);
                }
            }
            // M2.1: a `.openhydra` file was opened (fileAssociations → `file:` URL). `openhydra:`
            // scheme URLs are handled by the deep-link plugin above, so only take `file:` here.
            // `RunEvent::Opened` exists only on macOS/iOS (Win/Linux deliver file opens via argv —
            // a follow-up); the scheme path is cross-platform via deep-link + single-instance.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            tauri::RunEvent::Opened { urls } => {
                for u in &urls {
                    if u.scheme() == "file" {
                        if let Some(text) = card_text_from_url(u) {
                            deliver_card(app, text);
                        }
                    }
                }
            }
            _ => {}
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every screen was checked at 900×600 (2026-10-19; below 900 wide Share and Engines overflow),
    /// and at 900×500 (2026-10-22): a 1366×768 screen at 125% leaves ~534 px for the content, so the
    /// height stays at or under that.
    #[test]
    fn the_window_cant_shrink_below_the_size_every_screen_fits() {
        let conf: serde_json::Value = serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let w = &conf["app"]["windows"][0];
        let n = |k: &str| w[k].as_u64().unwrap_or_else(|| panic!("{k} missing"));
        assert!(n("minWidth") >= 900 && (500..=530).contains(&n("minHeight")), "minimum {}×{}", n("minWidth"), n("minHeight"));
        assert!(n("width") >= n("minWidth") && n("height") >= n("minHeight"), "the default fits the minimum");
    }

    #[test]
    fn a_failed_agent_command_shows_only_its_reason() {
        // `swarm remote-scope` logs (coloured) while its node runs, then prints the reason last.
        let stderr = b"openhydra-agent: node up \xe2\x80\x94 libp2p=12D3KooWX; dialing rig 12D3KooWY to set m -> global\n\
            \x1b[2m2026-10-11T20:54:07Z\x1b[0m \x1b[31mERROR\x1b[0m libp2p_mdns: error sending packet\n\
            \x1b[33m WARN\x1b[0m listener_closed\n\
            openhydra-agent: rig refused: remote publish to Global is disabled on this rig (allow_remote_publish=false)\n";
        assert_eq!(
            agent_failure(stderr, "x"),
            "rig refused: remote publish to Global is disabled on this rig (allow_remote_publish=false)"
        );
        // A plain failure line (card export) reads the same.
        assert_eq!(agent_failure(b"openhydra-agent: model \"m\" is not shared\n", "x"), "model \"m\" is not shared");
        // No prefixed line: the last line, colour removed; nothing at all: the fallback.
        assert_eq!(agent_failure(b"\x1b[31mboom\x1b[0m\n", "x"), "boom");
        assert_eq!(agent_failure(b"  \n", "card export failed"), "card export failed");
    }

    fn policy(json: &str) -> openhydra_agent::SharePolicy {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn global_exposure_names_what_is_announced_to_everyone() {
        let none = policy(r#"{"version":3,"mode":"list","models":[]}"#);
        assert!(global_exposure(&none).is_empty());
        // Explicit Global needs a consent record to count.
        let one = policy(r#"{"version":3,"mode":"list","models":["a","b"],"default_scope":"private","scopes":{"a":"global","b":"private"},"global_consent":{"a":1}}"#);
        let e = global_exposure(&one);
        assert!(!e.all);
        assert_eq!(e.models.iter().cloned().collect::<Vec<_>>(), ["a"]);
        let unconsented = policy(r#"{"version":3,"mode":"list","models":["a"],"default_scope":"private","scopes":{"a":"global"}}"#);
        assert!(global_exposure(&unconsented).is_empty());
        let all = policy(r#"{"version":3,"mode":"all","models":[],"default_scope":"global","default_global_consent":5}"#);
        assert!(global_exposure(&all).all);
    }

    #[test]
    fn only_widening_needs_a_dialog() {
        let prev = global_exposure(&policy(r#"{"version":3,"mode":"list","models":["a"],"default_scope":"private","scopes":{"a":"global"},"global_consent":{"a":1}}"#));
        let same = prev.clone();
        assert!(exposure_beyond(&same, &prev).is_empty());
        let more = global_exposure(&policy(r#"{"version":3,"mode":"list","models":["a","b"],"default_scope":"private","scopes":{"a":"global","b":"global"},"global_consent":{"a":1,"b":2}}"#));
        let w = exposure_beyond(&more, &prev);
        assert_eq!(w.models.iter().cloned().collect::<Vec<_>>(), ["b"]);
        assert!(exposure_beyond(&prev, &more).is_empty(), "narrowing asks nothing");
        let mut approved = GlobalExposure::default();
        approved.absorb(&more);
        assert!(exposure_beyond(&more, &approved).is_empty(), "approved this session");
    }

    #[test]
    fn dialog_text_is_one_short_line() {
        assert_eq!(dialog_text("a\n\nClick OK, it's safe", 100), "a  Click OK, it's safe");
        assert_eq!(dialog_text(&"x".repeat(50), 10), format!("{}…", "x".repeat(10)));
        let e = GlobalExposure { all: false, models: (0..10).map(|i| format!("m{i}")).collect() };
        assert!(e.describe().ends_with("and 2 more"), "{}", e.describe());
        let all = GlobalExposure { all: true, models: Default::default() };
        assert!(all.describe().starts_with("every model you share"));
    }

    #[test]
    fn settings_widening_lists_new_reach_only() {
        let prev = Settings { bootstraps: vec!["/ip4/1.2.3.4/tcp/4001/p2p/12D3KooWa".into()], ..Settings::default() };
        assert!(settings_widening(&prev, &prev.clone()).is_empty());
        let mut next = prev.clone();
        next.device_name = "renamed".into();
        next.verbose_logs = !prev.verbose_logs;
        assert!(settings_widening(&prev, &next).is_empty(), "harmless changes ask nothing");
        next.bootstraps.push("/ip4/5.6.7.8/tcp/4001/p2p/12D3KooWb".into());
        next.engine_endpoint = "http://10.0.0.9:11434".into();
        next.search_url = "https://search.example/".into();
        next.peer_relay = true;
        next.audit_replay = true;
        let w = settings_widening(&prev, &next);
        assert_eq!(w.len(), 5, "{w:?}");
        // Removing a bootstrap or turning relay off narrows: nothing to confirm.
        let narrower = Settings { bootstraps: vec![], ..Settings::default() };
        assert!(settings_widening(&prev, &narrower).is_empty());
    }

    #[test]
    fn settings_validation_accepts_real_values_and_refuses_hostile_ones() {
        let ok = Settings {
            bootstraps: vec!["/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWJiDwh7X73f87VGyh9dHnyAMsXow93hKiw4K7m1g3uEWr".into(), "/dns4/bootstrap-us.openhydra.co/tcp/4001/p2p/12D3KooWNDk3cfXtqmCKn8vt8fz88xPQe78tqBBJ6rkfU3rn1pis".into()],
            engine_endpoint: "http://127.0.0.1:11434".into(),
            search_url: "https://search.example/".into(),
            device_name: "Sam's MacBook Air".into(),
            ..Settings::default()
        };
        validate_settings(&ok).unwrap();
        validate_settings(&Settings::default()).unwrap();
        let bad = |f: &dyn Fn(&mut Settings)| { let mut s = ok.clone(); f(&mut s); validate_settings(&s).is_err() };
        assert!(bad(&|s| s.bootstraps = vec!["not a multiaddr".into()]));
        assert!(bad(&|s| s.bootstraps = vec!["/ip4/1.2.3.4/tcp/4001".into()]), "needs a peer id");
        assert!(bad(&|s| s.bootstraps = vec!["/ip4/1.2.3.4/tcp/4001/p2p/x\";rm".into()]));
        assert!(bad(&|s| s.bootstraps = vec!["/ip4/127.0.0.1/tcp/4999/p2p/12D3KooWTestBootstrapPeer".into()]), "peer id must parse");
        assert!(bad(&|s| s.gateway_port = 80));
        assert!(bad(&|s| s.min_authenticity = "A9".into()));
        assert!(bad(&|s| s.engine_kind = "evil".into()));
        assert!(bad(&|s| s.engine_endpoint = "file:///etc/passwd".into()));
        assert!(bad(&|s| s.engine_endpoint = "javascript:alert(1)".into()));
        assert!(bad(&|s| s.search_url = "http://user:pw@host/".into()));
        assert!(bad(&|s| s.device_name = "x\ny".into()));
        assert!(bad(&|s| s.device_name = "x".repeat(65)));
    }

    #[test]
    fn the_window_only_navigates_to_the_apps_own_pages() {
        let ok = |u: &str| navigation_allowed(&u.parse().unwrap());
        assert!(ok("tauri://localhost/index.html"));
        assert!(ok("http://tauri.localhost/"));
        assert!(ok("https://tauri.localhost/"));
        assert!(ok("about:blank"));
        for bad in ["https://evil.example/", "http://localhost:16527/", "tauri://evil/", "file:///etc/passwd", "javascript:alert(1)", "data:text/html,<b>x</b>"] {
            assert!(!ok(bad), "{bad}");
        }
    }

    #[test]
    fn strips_ansi_sgr_sequences() {
        let colored = "\u{1b}[2m2026-07-02\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m announced";
        assert_eq!(strip_ansi(colored), "2026-07-02  INFO announced");
    }

    // ── M2.1 review fix #1: safe `.openhydra` file read ──

    fn file_url(p: &std::path::Path) -> url::Url {
        url::Url::from_file_path(p).unwrap()
    }

    #[test]
    fn read_card_file_reads_a_valid_openhydra_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("card.openhydra");
        std::fs::write(&p, r#"{"model_id":"x"}"#).unwrap();
        assert_eq!(read_card_file(&file_url(&p)).as_deref(), Some(r#"{"model_id":"x"}"#));
    }

    #[test]
    fn read_card_file_refuses_a_non_openhydra_extension() {
        // The user could point a `file:` open at any file; only a real `.openhydra` is read.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("secret.txt");
        std::fs::write(&p, "sensitive").unwrap();
        assert_eq!(read_card_file(&file_url(&p)), None);
    }

    #[test]
    fn read_card_file_caps_the_read_size() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("huge.openhydra");
        std::fs::write(&p, vec![b'a'; (MAX_CARD_FILE_BYTES + 10) as usize]).unwrap();
        assert_eq!(read_card_file(&file_url(&p)), None, "oversize file refused");
    }

    #[cfg(unix)]
    #[test]
    fn read_card_file_refuses_a_symlink_to_a_non_card_target() {
        // A `.openhydra`-named symlink pointing at a sensitive file must be refused (canonicalize
        // resolves the link, then the extension check fails on the real target).
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("id_rsa"); // no .openhydra extension
        std::fs::write(&target, "PRIVATE KEY").unwrap();
        let link = dir.path().join("evil.openhydra");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert_eq!(read_card_file(&file_url(&link)), None);
    }

    #[test]
    fn card_text_from_url_never_reads_disk_for_a_non_file_scheme() {
        // An `openhydra:` URL is the magnet string verbatim — it must never touch the filesystem.
        let u: url::Url = "openhydra:card:sW5z".parse().unwrap();
        assert_eq!(card_text_from_url(&u).as_deref(), Some("openhydra:card:sW5z"));
    }

    #[test]
    fn desktop_entry_quotes_exec_and_sets_fields() {
        // An AppImage path can contain spaces, so Exec must be quoted or the launcher breaks.
        let d = desktop_entry("/home/me/My Apps/OpenHydra.AppImage", "openhydra");
        assert!(d.contains("Exec=\"/home/me/My Apps/OpenHydra.AppImage\" %U"));
        assert!(d.contains("Name=OpenHydra"));
        assert!(d.contains("Icon=openhydra"));
        assert!(d.contains("Type=Application"));
        assert!(d.contains("Categories=Utility;Network;"));
        assert!(d.starts_with("[Desktop Entry]"));
    }

    #[test]
    fn absorbs_the_agent_status_lines() {
        let mut s = RoleStatus::default();
        absorb_line(&mut s, "openhydra-agent: node up — libp2p=12D3KooWabc openhydra=7a13");
        absorb_line(
            &mut s,
            "openhydra-agent: auto-detected 3 engine(s): ollama(2 models) @ http://127.0.0.1:11434",
        );
        absorb_line(&mut s, "openhydra-agent: announced 6 model(s) from auto");
        absorb_line(&mut s, "INFO openhydra_network::event_loop: relay reservation accepted");
        absorb_line(&mut s, "openhydra-agent: re-announced 5 model(s)");
        assert_eq!(s.peer_id.as_deref(), Some("12D3KooWabc"));
        assert!(s.engines.as_deref().unwrap().starts_with("auto-detected 3 engine(s)"));
        assert_eq!(s.announced, Some(5)); // re-announce supersedes
        assert_eq!(s.relays, 1);
    }

    #[test]
    fn settings_default_and_roundtrip() {
        let d = Settings::default();
        assert_eq!(d.gateway_port, 16527);
        assert!(d.engine_autostart);
        assert!(!d.authenticity_checks, "authenticity checks are opt-in");
        assert!(!d.audit_replay, "replaying prompts for audits is opt-in on the desktop");
        assert!(d.tokenizer_downloads, "token-count checks download tokenizers by default");
        assert!(d.public_network, "the public network by default");
        assert_eq!(d.min_authenticity, "off");
        assert_eq!(gateway_floor("off"), None);
        assert_eq!(gateway_floor("A1"), Some("A1"));
        assert_eq!(gateway_floor("A2"), Some("A1"), "A2 can't be reached by this gateway: used as A1");
        let e = gateway_error(503, r#"{"error":{"message":"no provider of 'm' is verified at A1-reputation-redundant or above (1 found below it)","type":"no_provider","code":null}}"#);
        assert!(e.starts_with("gateway 503 no_provider: no provider of 'm' is verified at A1"), "{e}");
        assert_eq!(gateway_error(502, "not json"), "gateway 502 : not json");
        assert_eq!(gateway_floor("A9"), None, "an unknown value never reaches the gateway");
        let json = serde_json::to_string(&d).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back.gateway_port, 16527);
        // Partial JSON (older config) still parses via serde(default).
        let partial: Settings = serde_json::from_str(r#"{"gateway_port": 9999}"#).unwrap();
        assert_eq!(partial.gateway_port, 9999);
        assert!(partial.bootstraps.is_empty());
    }

    /// The actual "an update stopped my sharing" guard: a settings.json written by an OLD version
    /// (no sharing/resume/version fields) must parse WITHOUT losing the fields it does have, and the
    /// new fields must default sanely — never a wipe.
    #[test]
    fn old_settings_json_migrates_without_losing_state() {
        let old = r#"{"bootstraps":["/dns4/x/tcp/4001"],"gateway_port":16527,"engine_autostart":true,
            "search_url":"","verbose_logs":false,"device_name":"Asus",
            "shared_models":["qwen3-coder:30b-a3b-q8_0"]}"#;
        let s: Settings = serde_json::from_str(old).unwrap();
        // present fields survive the schema bump
        assert_eq!(s.shared_models, vec!["qwen3-coder:30b-a3b-q8_0"]);
        assert_eq!(s.device_name, "Asus");
        assert!(s.engine_autostart);
        assert!(!s.authenticity_checks, "an older config without the field loads as off");
        assert!(!s.audit_replay, "an older config without the field doesn't replay prompts");
        assert!(s.tokenizer_downloads, "an older config without the field checks token counts, as the agent does");
        assert!(s.public_network, "an older config without the field stays on the public network");
        assert_eq!(s.min_authenticity, "off", "an older config without the field uses any provider");
        // new fields default sanely (unknown at old-schema time)
        assert!(!s.sharing_enabled, "unknown sharing intent defaults off, not a spurious on");
        assert!(s.resume_on_launch, "resume defaults on");
        assert_eq!(s.schema_version, 1, "a pre-versioning file reads as v1 → load_settings migrates it up");
    }

    /// Sharing intent round-trips: once set, it persists so a restart can resume it.
    #[test]
    fn sharing_intent_round_trips() {
        let mut s = Settings::default();
        assert!(!s.sharing_enabled);
        assert_eq!(s.schema_version, SCHEMA_VERSION); // a fresh install writes the current version
        s.sharing_enabled = true;
        s.shared_models = vec!["llama3.1:8b".into()];
        let back: Settings = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert!(back.sharing_enabled, "sharing intent survives a save/load");
        assert_eq!(back.shared_models, vec!["llama3.1:8b"]);
    }

    // ── M3: share-policy migration + mirror ──
    use openhydra_agent::{ShareMode, SharePolicy};

    #[test]
    fn migrate_empty_legacy_list_writes_share_all() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        migrate_share_policy_if_absent(&path, &[]);
        assert_eq!(SharePolicy::load(&path).unwrap().mode, ShareMode::All);
    }

    #[test]
    fn migrate_non_empty_legacy_list_writes_explicit_list() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        migrate_share_policy_if_absent(&path, &["qwen3.8:27b-q8_0".into(), "tinyllama:latest".into()]);
        let p = SharePolicy::load(&path).unwrap();
        assert_eq!(p.mode, ShareMode::List);
        assert!(p.is_shared("qwen3.8:27b-q8_0") && p.is_shared("tinyllama:latest"));
    }

    #[test]
    fn migration_never_clobbers_an_existing_policy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        // A user's real policy already on disk...
        SharePolicy::share_list(["only-this"]).write_atomic(&path).unwrap();
        // ...must survive a migration attempt driven by a stale legacy list.
        migrate_share_policy_if_absent(&path, &["something-else".into()]);
        let p = SharePolicy::load(&path).unwrap();
        assert!(p.is_shared("only-this") && !p.is_shared("something-else"));
    }

    #[test]
    fn merged_transfers_take_the_consumed_side_and_credits_spent_from_the_gateway() {
        let mut base = serde_json::json!({ "transfers": { "tokens_served": 5, "credits_earned": 9, "credits_spent": 0, "recent": [] } });
        merge_transfers(&mut base, serde_json::json!({ "tokens_consumed": 7, "credits_spent": 42, "recent": [] }));
        assert_eq!(base["transfers"]["credits_spent"], 42);
        assert_eq!(base["transfers"]["credits_earned"], 9);
        assert_eq!(base["transfers"]["tokens_consumed"], 7);
    }

    #[test]
    fn known_providers_union_the_gateways_view() {
        let mut base = serde_json::json!({ "network": { "known_models": ["a"], "known_providers": [ { "libp2p_peer_id": "P1", "model_id": "a" } ] } });
        merge_known_providers(&mut base, serde_json::json!({
            "known_models": ["a", "b"],
            "known_providers": [ { "libp2p_peer_id": "P1", "model_id": "a" }, { "libp2p_peer_id": "P2", "model_id": "b" } ],
        }));
        assert_eq!(base["network"]["known_models"], serde_json::json!(["a", "b"]));
        assert_eq!(base["network"]["known_providers"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn stream_chunks_become_ui_events() {
        let delta = serde_json::json!({ "choices": [{ "delta": { "content": "Hel" }, "finish_reason": null }] });
        assert_eq!(stream_events(&delta), vec![serde_json::json!({ "kind": "delta", "text": "Hel" })]);
        let role = serde_json::json!({ "choices": [{ "delta": { "role": "assistant" }, "finish_reason": null }] });
        assert!(stream_events(&role).is_empty());
        let done = serde_json::json!({ "model": "m", "choices": [{ "delta": {}, "finish_reason": "stop" }],
            "openhydra": { "provider_authenticity": "A1-reputation-redundant" }, "usage": { "completion_tokens": 3 } });
        let ev = stream_events(&done);
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0]["kind"], "done");
        assert_eq!(ev[0]["usage"]["completion_tokens"], 3);
        let err = serde_json::json!({ "choices": [{ "delta": {}, "finish_reason": "error" }], "error": { "message": "provider overloaded", "type": "upstream_overloaded" } });
        assert_eq!(stream_events(&err)[0]["message"], "gateway 502 upstream_overloaded: provider overloaded");
    }

    #[test]
    fn authenticity_checks_only_for_engines_that_can_be_scored() {
        assert!(scoring_supported("auto", false));
        assert!(scoring_supported("vllm", false));
        assert!(!scoring_supported("ollama", false) && scoring_supported("ollama", true));
        assert!(!scoring_supported("exo", true) && !scoring_supported("comfyui", true));
    }

    #[test]
    fn a_missing_gateway_reading_is_unknown_not_zero() {
        let mut base = serde_json::json!({ "transfers": { "credits_spent": 0, "tokens_consumed": 0, "credits_earned": 5 } });
        unknown_consumed_side(&mut base);
        assert!(base["transfers"]["credits_spent"].is_null());
        assert!(base["transfers"]["tokens_consumed"].is_null());
        assert_eq!(base["transfers"]["credits_earned"], 5);
    }

    #[test]
    fn mirror_reflects_policy_mode() {
        // `all` mirrors to the legacy "empty = share everything" sentinel...
        assert!(mirror_policy_into_settings(&SharePolicy::share_all()).is_empty());
        // ...and `list` mirrors to the explicit models (so a downgrade still shares the same set).
        let mut m = mirror_policy_into_settings(&SharePolicy::share_list(["b", "a"]));
        m.sort();
        assert_eq!(m, vec!["a", "b"]);
    }

    // ── R2: self-heal + reset ──
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn heal_backs_up_corrupt_and_resets_to_share_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::write(&path, b"{ corrupt not json").unwrap();
        heal_corrupt_share_policy(&path).unwrap();
        // the bad bytes are preserved for debugging...
        assert_eq!(std::fs::read_to_string(path.with_extension("json.corrupt.bak")).unwrap(), "{ corrupt not json");
        // ...and the live file is now a valid, fail-closed share-nothing policy.
        let p = SharePolicy::load(&path).unwrap();
        assert_eq!(p.mode, ShareMode::List);
        assert!(!p.is_shared("tinyllama:latest"));
    }

    #[test]
    fn a_page_load_stops_every_running_chat_stream() {
        let a = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let b = Arc::new(std::sync::atomic::AtomicBool::new(false));
        CHAT_STREAMS.lock().unwrap().insert("test-a".into(), a.clone());
        CHAT_STREAMS.lock().unwrap().insert("test-b".into(), b.clone());
        assert!(stop_all_chat_streams() >= 2);
        assert!(a.load(std::sync::atomic::Ordering::Relaxed) && b.load(std::sync::atomic::Ordering::Relaxed));
        CHAT_STREAMS.lock().unwrap().retain(|k, _| !k.starts_with("test-"));
    }

    #[test]
    fn an_inherited_share_all_is_dropped_but_a_chosen_list_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        // What v0.3.20 wrote when its Share view was opened: everything, globally, with consent.
        SharePolicy::from_legacy_list(Vec::<String>::new()).write_atomic(&path).unwrap();
        assert!(SharePolicy::load(&path).unwrap().announce_globally("anything"));
        assert!(drop_inherited_share_all(&path));
        let p = SharePolicy::load(&path).unwrap();
        assert!(!p.is_shared("anything") && !p.announce_globally("anything"));
        assert!(!drop_inherited_share_all(&path), "nothing left to drop");
        // A list the user picked stays.
        SharePolicy::share_list(["qwen2.5:7b"]).write_atomic(&path).unwrap();
        assert!(!drop_inherited_share_all(&path));
        assert!(SharePolicy::load(&path).unwrap().is_shared("qwen2.5:7b"));
    }

    #[test]
    fn ensure_valid_heals_corrupt_and_raises_the_reset_flag() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::write(&path, b"garbage").unwrap();
        let flag = AtomicBool::new(false);
        ensure_valid_share_policy_at(&path, &[], false, &flag);
        assert!(flag.load(Ordering::Relaxed), "reset flag raised on heal");
        assert!(!SharePolicy::load(&path).unwrap().is_shared("x")); // healed to share-nothing
    }

    #[test]
    fn ensure_valid_leaves_a_good_file_untouched_and_flag_clear() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        SharePolicy::share_list(["keep-me"]).write_atomic(&path).unwrap();
        let flag = AtomicBool::new(false);
        ensure_valid_share_policy_at(&path, &["ignored".into()], false, &flag);
        assert!(!flag.load(Ordering::Relaxed), "no reset on a valid file");
        assert!(SharePolicy::load(&path).unwrap().is_shared("keep-me")); // untouched
    }

    #[test]
    fn ensure_valid_leaves_a_transiently_unreadable_file_untouched() {
        // A path that exists but yields a NON-InvalidData IO error on read (here: a directory) must
        // NOT be healed/overwritten or flagged — it may be a valid file we just can't read now. (F1/F2)
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::create_dir(&path).unwrap(); // exists() == true, but load() errors non-InvalidData
        let flag = AtomicBool::new(false);
        ensure_valid_share_policy_at(&path, &[], false, &flag);
        assert!(!flag.load(Ordering::Relaxed), "transient IO error must not raise the reset flag");
        assert!(path.is_dir(), "the path must be left untouched, not overwritten");
    }

    #[test]
    fn ensure_valid_migrates_when_absent_without_flag() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        let flag = AtomicBool::new(false);
        ensure_valid_share_policy_at(&path, &["m1".into()], false, &flag);
        assert!(!flag.load(Ordering::Relaxed), "migration is not a reset");
        assert!(SharePolicy::load(&path).unwrap().is_shared("m1"));
    }

    #[test]
    fn engine_args_pass_a_known_kind_and_its_endpoint() {
        assert_eq!(engine_args("auto", ""), vec!["--engine-kind", "auto"]);
        assert_eq!(engine_args("vllm", ""), vec!["--engine-kind", "vllm"]);
        assert_eq!(
            engine_args("llama-cpp", " http://127.0.0.1:8081 "),
            vec!["--engine-kind", "llama-cpp", "--engine", "http://127.0.0.1:8081"]
        );
        // Unknown kinds (or the UI's labels) never reach the agent: fall back to auto-detect.
        assert_eq!(engine_args("LM Studio", "http://x"), vec!["--engine-kind", "auto"]);
    }

    #[test]
    fn bootstrap_lists_split_and_merge_without_duplicates() {
        assert_eq!(parse_bootstrap_list(" /ip4/1.2.3.4/tcp/4001/p2p/A, /ip4/5.6.7.8/tcp/4001/p2p/B\n"), vec![
            "/ip4/1.2.3.4/tcp/4001/p2p/A".to_string(),
            "/ip4/5.6.7.8/tcp/4001/p2p/B".to_string(),
        ]);
        assert!(parse_bootstrap_list("").is_empty());
        let mut s = Settings::default();
        s.bootstraps = vec![" /ip4/9.9.9.9/tcp/4001/p2p/C ".into(), "".into(), "/ip4/9.9.9.9/tcp/4001/p2p/C".into()];
        let eff = effective_bootstraps(&s);
        assert_eq!(eff.iter().filter(|b| b.as_str() == "/ip4/9.9.9.9/tcp/4001/p2p/C").count(), 1);
        assert_eq!(eff.len(), builtin_bootstraps().len() + 1);
        assert_eq!(public_relays_arg(&s), "auto");
        // A private deployment: only its own bootstraps, never the public relays.
        s.public_network = false;
        assert_eq!(effective_bootstraps(&s), vec!["/ip4/9.9.9.9/tcp/4001/p2p/C".to_string()]);
        assert_eq!(public_relays_arg(&s), "off");
    }

    /// Review 2026-10-14 M1/D1: a private deployment fetches no tokenizer list and downloads no
    /// tokenizers unless the user turns its own switch on (which asks first).
    #[test]
    fn a_private_deployment_contacts_no_tokenizer_service_by_default() {
        let mut s = Settings::default();
        assert!(tokenizer_args(&s).is_empty(), "the public network: downloads on, list fetched");
        s.public_network = false;
        assert_eq!(tokenizer_args(&s), vec!["--no-tokenizer-downloads", "--tokenizer-index-url", "off"]);
        s.private_tokenizer_downloads = true;
        assert_eq!(tokenizer_args(&s), vec!["--tokenizer-index-url", "off"]);
        // An older config without the private field: off.
        let old: Settings = serde_json::from_str(r#"{"public_network": false, "tokenizer_downloads": true}"#).unwrap();
        assert!(!tokenizer_downloads_on(&old));
        // Turning downloads on asks first, in either mode (L1).
        let (mut prev, mut next) = (Settings::default(), Settings::default());
        prev.public_network = false;
        next.public_network = false;
        next.private_tokenizer_downloads = true;
        assert!(settings_widening(&prev, &next).iter().any(|w| w.contains("huggingface.co")));
        prev.public_network = true;
        prev.tokenizer_downloads = false;
        next = prev.clone();
        next.tokenizer_downloads = true;
        assert!(settings_widening(&prev, &next).iter().any(|w| w.contains("huggingface.co")));
        // Going private never asks about downloads.
        next = prev.clone();
        next.public_network = false;
        assert!(!settings_widening(&prev, &next).iter().any(|w| w.contains("huggingface.co")));
    }

    #[test]
    fn fresh_install_shares_nothing_until_the_user_chooses() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        let flag = AtomicBool::new(false);
        ensure_valid_share_policy_at(&path, &[], true, &flag);
        let p = SharePolicy::load(&path).unwrap();
        assert_eq!(p.mode, ShareMode::List);
        assert!(!p.is_shared("tinyllama:latest"), "nothing shared on a fresh install");
        assert!(p.default_global_consent.is_none(), "no consent recorded without asking");
        assert!(!flag.load(Ordering::Relaxed));
    }

    #[test]
    fn when_a_missing_policy_starts_empty() {
        let mut s = Settings::default();
        assert!(policy_starts_empty(false, &s), "fresh install");
        s.share_policy_initialized = true;
        assert!(policy_starts_empty(true, &s), "this install had its own policy: a deletion");
        s.share_policy_initialized = false;
        assert!(policy_starts_empty(true, &s), "upgrade from a user who wasn't sharing");
        s.sharing_enabled = true;
        assert!(!policy_starts_empty(true, &s), "upgrade from a sharer keeps what they shared");
    }

    #[test]
    fn a_missing_policy_after_this_install_had_one_shares_nothing() {
        // start_empty = true is what ensure_valid_share_policy passes once share_policy_initialized is set.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        let flag = AtomicBool::new(false);
        ensure_valid_share_policy_at(&path, &["was-private".into()], true, &flag);
        let p = SharePolicy::load(&path).unwrap();
        assert!(!p.is_shared("was-private") && p.default_global_consent.is_none());
    }

    #[test]
    fn upgrade_with_empty_legacy_list_still_shares_everything() {
        // An existing user whose old settings meant "share everything" keeps sharing after the update.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        let flag = AtomicBool::new(false);
        ensure_valid_share_policy_at(&path, &[], false, &flag);
        assert_eq!(SharePolicy::load(&path).unwrap().mode, ShareMode::All);
    }
}
