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

mod hostinfo;
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
    let mut s: Settings = std::fs::read_to_string(settings_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    // C5 migration: the gateway port was never user-editable before v0.3.8, so a persisted
    // 8080 (the pre-16527 default that collided with llama.cpp) is stale — bump it to the
    // current default and re-save. Fresh installs already default to 16527.
    if s.gateway_port == 8080 {
        s.gateway_port = 16527;
        let _ = store_settings(&s);
    }
    s
}

fn store_settings(s: &Settings) -> Result<(), String> {
    let dir = openhydra_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(settings_path(), serde_json::to_string_pretty(s).unwrap())
        .map_err(|e| e.to_string())
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
    spawn_role(
        &state.provider,
        "desktop-provider.key",
        PROVIDER_PORT,
        PROVIDER_STATUS_PORT,
        &settings.bootstraps,
        &sub,
    )
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
fn save_settings(state: tauri::State<'_, AppState>, settings: Settings) -> Result<(), String> {
    store_settings(&settings)?;
    *state.settings.lock().map_err(|e| e.to_string())? = settings;
    Ok(())
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
        })
        .invoke_handler(tauri::generate_handler![
            get_state,
            start_provider,
            start_gateway,
            stop_provider,
            stop_gateway,
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
            export_logs,
        ])
        .setup(|app| {
            use tauri::menu::{Menu, MenuItem};
            use tauri::tray::TrayIconBuilder;
            let open = MenuItem::with_id(app, "open", "Open OpenHydra", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit OpenHydra", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&open, &quit])?;
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
                    "quit" => {
                        kill_all(&app.state::<AppState>());
                        app.exit(0);
                    }
                    _ => {}
                });
            if let Some(icon) = app.default_window_icon() {
                tray = tray.icon(icon.clone());
            }
            tray.build(app)?;
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
}
