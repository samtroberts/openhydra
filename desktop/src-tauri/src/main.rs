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

use std::collections::VecDeque;
use std::io::BufRead;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::Manager;

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

// ── Settings (persisted at ~/.openhydra/desktop.json) ──

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct Settings {
    /// Bootstrap multiaddrs (`/ip4/…/tcp/4001/p2p/<peer-id>`), one per entry. Empty is
    /// valid on a LAN (mDNS discovers peers), but joining the public network needs at
    /// least one. Deliberately not baked into the app: bootstrap IPs/peer ids stay out
    /// of tracked code and are the operator's to configure.
    bootstraps: Vec<String>,
    /// Local port the OpenAI-compatible gateway binds on 127.0.0.1.
    gateway_port: u16,
    /// Pass `--engine-autostart` to the provider (start LM Studio / Ollama if down).
    engine_autostart: bool,
    /// Optional SearXNG-compatible search endpoint (e.g. `http://127.0.0.1:8888`) powering
    /// the chat "web" toggle. Empty → the toggle is hidden. The operator brings their own
    /// search backend, same BYO philosophy as engines.
    search_url: String,
    /// #4: verbose agent logging. Off by default (quiet `RUST_LOG=warn`) so the log ring stays
    /// light; on raises the agent to info/debug for troubleshooting. The essential status lines
    /// are `eprintln!` (not `RUST_LOG`-gated), so quiet mode never breaks the status views.
    verbose_logs: bool,
    /// #9: user-editable device name shown to peers / in the UI. Empty → derive from the OS.
    device_name: String,
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
            search_url: String::new(),
            verbose_logs: false,
            device_name: String::new(),
            shared_models: Vec::new(), // empty ⇒ share every detected model (the default)
            sharing_enabled: false,
            resume_on_launch: true,
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
/// - absent → migrate from the legacy list (first run after upgrade);
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
    ensure_valid_share_policy_at(&path, &settings.shared_models, reset_flag);
    path
}

/// Path-parametrized core of [`ensure_valid_share_policy`] (testable without `~/.openhydra`): sets
/// `reset_flag` true iff a corrupt file was healed.
fn ensure_valid_share_policy_at(
    path: &std::path::Path,
    legacy: &[String],
    reset_flag: &std::sync::atomic::AtomicBool,
) {
    if !path.exists() {
        migrate_share_policy_if_absent(path, legacy);
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

/// Spawn one agent role and wire its output pumps. Global flags before the subcommand
/// (clap requirement), pinned identity per role, distinct listen port.
fn spawn_role(
    role: &Arc<Mutex<Role>>,
    identity: &str,
    listen_port: u16,
    status_port: u16,
    bootstraps: &[String],
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
    for b in bootstraps {
        let b = b.trim();
        if !b.is_empty() {
            cmd.arg("--bootstrap").arg(b);
        }
    }
    cmd.args(subcommand)
        // #4: quiet by default (essential status lines are eprintln!, not RUST_LOG-gated, so
        // this never breaks the status views); verbose opt-in raises to info/debug.
        .env(
            "RUST_LOG",
            if load_settings().verbose_logs { "openhydra_network=info,debug" } else { "warn" },
        )
        .env("OPENHYDRA_STATUS_TOKEN", status_token())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("spawn {}: {e}", bin.display()))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let mut guard = role.lock().map_err(|e| e.to_string())?;
    registry_set(guard.pid_slot, child.id());
    guard.status = RoleStatus { running: true, pid: Some(child.id()), ..Default::default() };
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
    }
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

#[tauri::command]
fn start_provider(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let settings = state.settings.lock().map_err(|e| e.to_string())?.clone();
    let mut sub = vec!["provide".to_string(), "--engine-kind".into(), "auto".into()];
    if settings.engine_autostart {
        sub.push("--engine-autostart".into());
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
    spawn_role(
        &state.provider,
        "desktop-provider.key",
        PROVIDER_PORT,
        PROVIDER_STATUS_PORT,
        &settings.bootstraps,
        &sub,
    )?;
    set_sharing_enabled(&state, true); // remember intent so a restart resumes sharing
    Ok(())
}

/// Persist the share policy (called by the UI when the user toggles a model or the master switch).
/// Written atomically; the running provider **hot-reloads it within ~1s — no restart**. Also mirrors
/// into `settings.shared_models` for one-release downgrade safety.
#[tauri::command]
fn save_share_policy(
    state: tauri::State<'_, AppState>,
    policy: openhydra_agent::SharePolicy,
) -> Result<(), String> {
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
        if s.shared_models != mirror {
            s.shared_models = mirror;
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
    let bin = agent_binary()?;
    let key = openhydra_dir().join("desktop-provider.key");
    let out = Command::new(&bin).arg("--identity").arg(&key).arg("peer-id").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let id = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!id.is_empty()).then_some(id)
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
    spawn_role(
        &state.gateway,
        "desktop-consumer.key",
        GATEWAY_PORT,
        GATEWAY_STATUS_PORT,
        &settings.bootstraps,
        &sub,
    )
}

#[tauri::command]
fn stop_provider(state: tauri::State<'_, AppState>) {
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
        if prov && gw {
            // Provider base carries the served side (peers, served counters, `served` ledger
            // rows); the gateway process holds the consumed side + `used` ledger rows. Pull the
            // gateway's full status once and overlay both its economy and its transfers.
            if let Some(gw) = fetch_status(GATEWAY_STATUS_PORT, "/status") {
                merge_economy(&mut base, gw.get("economy").cloned().unwrap_or_default());
                if let Some(gw_transfers) = gw.get("transfers").cloned() {
                    merge_transfers(&mut base, gw_transfers);
                }
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

/// Merge the gateway's transfer counters (the consumed side + `used` ledger rows) into the
/// provider base snapshot's `transfers` block. The provider process only ever fills the served
/// side, so consumed fields are taken wholesale from the gateway; the two roles' recent-ledger
/// rings (provider = `served`, gateway = `used`) are concatenated and re-sorted newest-first.
fn merge_transfers(base: &mut serde_json::Value, gw: serde_json::Value) {
    let Some(t) = base.get_mut("transfers") else { return };
    for key in ["requests_consumed", "tokens_consumed", "consumed_per_model"] {
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
) -> Result<serde_json::Value, String> {
    let port = state.settings.lock().map_err(|e| e.to_string())?.gateway_port;
    // Default when the UI doesn't supply one: a generous chat budget, NOT the engine's own
    // default. LM Studio (and others) fall back to ~512 tokens when `max_tokens` is absent —
    // far too small for a reasoning model, which spends that entirely on hidden chain-of-thought
    // and never reaches the answer ("no visible text"). 4096 lets it finish thinking AND reply.
    const DEFAULT_CHAT_MAX_TOKENS: u32 = 4096;
    let body = serde_json::json!({
        "model": model,
        "messages": messages
            .iter()
            .map(|m| serde_json::json!({ "role": m.role, "content": m.content }))
            .collect::<Vec<_>>(),
        "max_tokens": max_tokens.unwrap_or(DEFAULT_CHAT_MAX_TOKENS),
    });
    tauri::async_runtime::spawn_blocking(move || {
        use openhydra_agent::adapter::HttpClient;
        let client = openhydra_agent::ReqwestClient::new().map_err(|e| e.to_string())?;
        let resp = client
            .post_json(&format!("http://127.0.0.1:{port}/v1/chat/completions"), &body.to_string())
            .map_err(|e| format!("gateway: {e}"))?;
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
#[tauri::command]
fn connector_apply(
    state: tauri::State<'_, AppState>,
    key: String,
    models: Option<Vec<String>>,
) -> Result<connectors::ConnectReport, String> {
    let origin = gateway_origin(&state)?;
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
        use openhydra_agent::adapter::HttpClient;
        let client = openhydra_agent::ReqwestClient::new().map_err(|e| e.to_string())?;
        let resp = client
            .post_json(&format!("http://127.0.0.1:{port}/v1/chat/completions"), &body.to_string())
            .map_err(|e| format!("gateway unreachable on :{port} — is OpenHydra sharing/serving? ({e})"))?;
        let v: serde_json::Value = serde_json::from_str(&resp).map_err(|e| e.to_string())?;
        if let Some(err) = v.get("error") {
            return Err(err.get("message").and_then(|m| m.as_str()).unwrap_or("gateway error").to_string());
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
        openhydra_agent::ReqwestClient::new()
            .map(|c| c.get(&format!("http://127.0.0.1:{port}/v1/models")).is_ok())
            .unwrap_or(false)
    })
    .await
    .unwrap_or(false))
}

#[tauri::command]
fn save_settings(state: tauri::State<'_, AppState>, mut settings: Settings) -> Result<(), String> {
    let mut cur = state.settings.lock().map_err(|e| e.to_string())?;
    // Preserve fields the settings FORM doesn't own — taking the payload verbatim would reset them.
    // `sharing_enabled` is managed by start/stop_provider (a save must not clear a live sharing intent
    // → it would break resume-on-launch); `schema_version` is managed by load_settings' migration.
    // `resume_on_launch` IS surfaced by the settings toggle now, so its payload value is honoured.
    settings.sharing_enabled = cur.sharing_enabled;
    settings.schema_version = cur.schema_version;
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

fn main() {
    // Reap agents a crashed/killed previous instance left behind, then arm the signal
    // path so THIS instance can't leave any (see the pid-registry block above).
    sweep_stale_agents();
    install_signal_handlers();
    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .manage(AppState {
            provider: Arc::new(Mutex::new(Role::new(&PROVIDER_PID))),
            gateway: Arc::new(Mutex::new(Role::new(&GATEWAY_PID))),
            settings: Mutex::new(load_settings()),
            resumed_on_launch: std::sync::atomic::AtomicBool::new(false),
            share_policy_reset: std::sync::atomic::AtomicBool::new(false),
        })
        .invoke_handler(tauri::generate_handler![
            get_state,
            start_provider,
            start_gateway,
            stop_provider,
            stop_gateway,
            save_share_policy,
            read_share_policy,
            reset_share_policy,
            detect_engines_now,
            system_info,
            install_plan,
            install_engine,
            installed_engines,
            run_engine,
            gateway_health,
            chat_completion,
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
                    match start_provider(app.state()) {
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
                    "open" => {
                        if let Some(w) = app.get_webview_window("main") {
                            let _ = w.show();
                            let _ = w.set_focus();
                        }
                    }
                    // Sharing = start/stop the provider role, same as the in-app toggle. Rebuild the
                    // menu right away so the checkmark reflects the new state.
                    "sharing" => {
                        let running = app
                            .state::<AppState>()
                            .provider
                            .lock()
                            .map(|r| r.status.running)
                            .unwrap_or(false);
                        if running {
                            stop_provider(app.state());
                        } else if let Err(e) = start_provider(app.state()) {
                            eprintln!("tray: start provider failed: {e}");
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
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                kill_all(&app.state::<AppState>());
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_ansi_sgr_sequences() {
        let colored = "\u{1b}[2m2026-07-02\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m announced";
        assert_eq!(strip_ansi(colored), "2026-07-02  INFO announced");
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
    fn ensure_valid_heals_corrupt_and_raises_the_reset_flag() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        std::fs::write(&path, b"garbage").unwrap();
        let flag = AtomicBool::new(false);
        ensure_valid_share_policy_at(&path, &[], &flag);
        assert!(flag.load(Ordering::Relaxed), "reset flag raised on heal");
        assert!(!SharePolicy::load(&path).unwrap().is_shared("x")); // healed to share-nothing
    }

    #[test]
    fn ensure_valid_leaves_a_good_file_untouched_and_flag_clear() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        SharePolicy::share_list(["keep-me"]).write_atomic(&path).unwrap();
        let flag = AtomicBool::new(false);
        ensure_valid_share_policy_at(&path, &["ignored".into()], &flag);
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
        ensure_valid_share_policy_at(&path, &[], &flag);
        assert!(!flag.load(Ordering::Relaxed), "transient IO error must not raise the reset flag");
        assert!(path.is_dir(), "the path must be left untouched, not overwritten");
    }

    #[test]
    fn ensure_valid_migrates_when_absent_without_flag() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("share-policy.json");
        let flag = AtomicBool::new(false);
        ensure_valid_share_policy_at(&path, &["m1".into()], &flag);
        assert!(!flag.load(Ordering::Relaxed), "migration is not a reset");
        assert!(SharePolicy::load(&path).unwrap().is_shared("m1"));
    }
}
