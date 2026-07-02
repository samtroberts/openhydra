// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Opt-in engine autostart (`--engine-autostart`).
//!
//! OpenHydra is BYO-engine: it proxies to an engine the operator already runs and never
//! runs a model itself. The one gap that trips up non-devs is engines whose OpenAI
//! server is a *separate toggle from the app being open* — LM Studio (the app can be
//! running while its `:1234` server is off) — or a daemon that simply isn't up yet
//! (Ollama). For **those two**, and only when `--engine-autostart` is set, this module
//! starts the engine's server before we announce, then waits for it to answer.
//!
//! It deliberately does **not** cover vLLM / llama.cpp / Exo: those need a model or
//! cluster argument OpenHydra can't invent, so "autostart" there would just be running
//! the operator's own launch command — the operator's job.
//!
//! This is the only part of the agent that manages a child process. It's isolated here,
//! feature-gated (`engine-autostart`), and gated again at runtime behind the opt-in flag,
//! so the lean pure-protocol build carries no process-spawning code.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// How long to wait for a just-launched engine to start answering before giving up.
const READY_TIMEOUT: Duration = Duration::from_secs(30);
/// How often to re-probe readiness while waiting.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// A recipe for launching one engine's local server.
pub struct LaunchSpec {
    /// Human-facing engine label for logs/errors (e.g. `"LM Studio"`).
    pub label: &'static str,
    /// Candidate executable paths to try, in order; the first that exists and is
    /// executable is launched.
    program_candidates: Vec<PathBuf>,
    /// Arguments passed to the launcher (e.g. `["server", "start"]`).
    args: Vec<&'static str>,
}

impl LaunchSpec {
    /// The launcher for this engine label, or `None` if OpenHydra has no safe way to start
    /// it unattended (needs a model/cluster arg we can't invent). Keyed by the adapter's
    /// [`engine_name`](crate::adapter::EngineAdapter::engine_name).
    pub fn for_engine(engine_name: &str) -> Option<LaunchSpec> {
        match engine_name {
            "ollama" => Some(Self::ollama()),
            "lm-studio" => Some(Self::lm_studio()),
            _ => None,
        }
    }

    /// `ollama serve` — the daemon is usually already up (macOS app / Linux systemd), in
    /// which case the readiness probe short-circuits and we never spawn it.
    fn ollama() -> Self {
        LaunchSpec {
            label: "Ollama",
            program_candidates: candidate_paths("ollama", &[]),
            args: vec!["serve"],
        }
    }

    /// `lms server start` — starts LM Studio's OpenAI server (which may be toggled off even
    /// while the desktop app is open). Returns once the background server is up.
    fn lm_studio() -> Self {
        LaunchSpec {
            label: "LM Studio",
            // The `lms` CLI installs under ~/.lmstudio/bin (also on PATH once bootstrapped).
            program_candidates: candidate_paths("lms", &[".lmstudio/bin", ".cache/lm-studio/bin"]),
            args: vec!["server", "start"],
        }
    }

    /// First candidate that exists and is executable.
    fn resolve_program(&self) -> Option<&Path> {
        self.program_candidates.iter().map(PathBuf::as_path).find(|p| is_executable(p))
    }

    /// Comma-joined candidate list, for the "cannot find launcher" error.
    fn candidates_display(&self) -> String {
        self.program_candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Ensure the engine described by `spec` is reachable, launching its server if it isn't.
///
/// `is_ready` probes the engine (the caller backs it with the adapter's `detect_models`, so
/// "ready" == the exact check announce uses: the server answers, model or not). Flow:
///
/// 1. Already answering → no-op (never double-start a running daemon).
/// 2. Down → locate the launcher, spawn it detached, poll `is_ready` up to [`READY_TIMEOUT`].
/// 3. Launcher missing, spawn fails, or it never comes up → a clear, actionable error.
pub fn ensure_running(
    base_url: &str,
    spec: &LaunchSpec,
    is_ready: impl Fn() -> bool,
) -> Result<(), String> {
    if is_ready() {
        eprintln!("openhydra-agent: {} already reachable at {base_url}", spec.label);
        return Ok(());
    }

    let program = spec.resolve_program().ok_or_else(|| {
        format!(
            "--engine-autostart: cannot find the {} launcher (looked for: {}) — install {0} \
             or start its server yourself",
            spec.label,
            spec.candidates_display(),
        )
    })?;

    eprintln!(
        "openhydra-agent: {} not reachable at {base_url} — launching `{} {}`",
        spec.label,
        program.display(),
        spec.args.join(" "),
    );
    spawn_detached(program, &spec.args)
        .map_err(|e| format!("--engine-autostart: failed to launch {}: {e}", spec.label))?;

    let deadline = Instant::now() + READY_TIMEOUT;
    loop {
        std::thread::sleep(POLL_INTERVAL);
        if is_ready() {
            eprintln!("openhydra-agent: {} is up", spec.label);
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "--engine-autostart: {} did not become reachable at {base_url} within {}s of \
                 launch — check its logs, or start it yourself",
                spec.label,
                READY_TIMEOUT.as_secs(),
            ));
        }
    }
}

/// Spawn `program` with `args` fully detached: no inherited stdio, and (on unix) in its own
/// session so it survives the agent's controlling terminal (SIGHUP) and isn't torn down when
/// the agent exits. A detached reaper thread waits on the child so a quick-exiting launcher
/// (`lms server start`, which returns once its background server is up) doesn't linger as a
/// zombie, and a long-lived one (`ollama serve`) is waited on harmlessly for our lifetime.
fn spawn_detached(program: &Path, args: &[&'static str]) -> std::io::Result<()> {
    let mut cmd = Command::new(program);
    cmd.args(args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: setsid() is async-signal-safe and touches no shared state; a failure is
        // non-fatal (the child still runs, just not fully detached), so we ignore it.
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }

    let child = cmd.spawn()?;
    std::thread::spawn(move || {
        let mut child = child;
        let _ = child.wait();
    });
    Ok(())
}

/// Build the ordered executable-candidate list for `bin`: every directory on `PATH`, then
/// the given `home_subdirs` (relative to `$HOME`/`%USERPROFILE%`) plus `~/.local/bin`, then
/// a few common install dirs (present even under the minimal PATH of a systemd/tmux launch).
fn candidate_paths(bin: &str, home_subdirs: &[&str]) -> Vec<PathBuf> {
    let exe = exe_name(bin);
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    if let Some(home) = home_dir() {
        for sub in home_subdirs {
            dirs.push(home.join(sub));
        }
        dirs.push(home.join(".local/bin"));
    }
    dirs.extend(COMMON_BIN_DIRS.iter().map(PathBuf::from));

    join_exe(&dirs, &exe)
}

/// Pure helper (testable without touching the environment): `dir.join(exe)` for each dir,
/// de-duplicated in first-seen order.
fn join_exe(dirs: &[PathBuf], exe: &str) -> Vec<PathBuf> {
    let mut seen = std::collections::HashSet::new();
    dirs.iter()
        .map(|d| d.join(exe))
        .filter(|p| seen.insert(p.clone()))
        .collect()
}

/// `$HOME` (unix) / `%USERPROFILE%` (windows), if set.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from)
}

/// Executable file name for `bin` on this OS (`.exe` suffix on Windows).
fn exe_name(bin: &str) -> String {
    #[cfg(windows)]
    {
        format!("{bin}.exe")
    }
    #[cfg(not(windows))]
    {
        bin.to_string()
    }
}

/// Common install dirs to check beyond `PATH` (a systemd/tmux launch often has a minimal one).
#[cfg(unix)]
const COMMON_BIN_DIRS: &[&str] = &["/usr/local/bin", "/usr/bin", "/bin", "/opt/homebrew/bin"];
#[cfg(not(unix))]
const COMMON_BIN_DIRS: &[&str] = &[];

/// Whether `p` is a regular file with an execute bit (unix) / a regular file (elsewhere).
fn is_executable(p: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(p)
            .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        p.is_file()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launcher_only_for_startable_engines() {
        assert!(LaunchSpec::for_engine("ollama").is_some());
        assert!(LaunchSpec::for_engine("lm-studio").is_some());
        // Engines needing a model/cluster arg we can't invent have no launcher.
        assert!(LaunchSpec::for_engine("vllm").is_none());
        assert!(LaunchSpec::for_engine("llama.cpp").is_none());
        assert!(LaunchSpec::for_engine("openai").is_none());
        assert!(LaunchSpec::for_engine("something-else").is_none());
    }

    #[test]
    fn launch_commands_are_the_expected_server_subcommands() {
        assert_eq!(LaunchSpec::ollama().args, vec!["serve"]);
        assert_eq!(LaunchSpec::lm_studio().args, vec!["server", "start"]);
    }

    #[test]
    fn lm_studio_candidates_include_the_home_cli_dir() {
        let dirs = vec![PathBuf::from("/home/u/.lmstudio/bin")];
        let joined = join_exe(&dirs, "lms");
        assert_eq!(joined, vec![PathBuf::from("/home/u/.lmstudio/bin/lms")]);
    }

    #[test]
    fn join_exe_dedupes_first_seen() {
        let dirs = vec![
            PathBuf::from("/usr/bin"),
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/usr/bin"), // duplicate PATH entry
        ];
        let joined = join_exe(&dirs, "ollama");
        assert_eq!(
            joined,
            vec![
                PathBuf::from("/usr/bin/ollama"),
                PathBuf::from("/usr/local/bin/ollama"),
            ]
        );
    }

    #[test]
    fn ready_engine_is_a_noop_even_with_no_launcher() {
        // An already-reachable engine must never trigger a spawn or a launcher lookup, so a
        // spec with no valid program still succeeds when is_ready() is true.
        let spec = LaunchSpec {
            label: "Test",
            program_candidates: vec![PathBuf::from("/nonexistent/nope")],
            args: vec![],
        };
        assert!(ensure_running("http://127.0.0.1:1", &spec, || true).is_ok());
    }

    #[test]
    fn missing_launcher_errors_without_spawning() {
        // Down engine + no resolvable launcher → an actionable error, no spawn, no wait.
        let spec = LaunchSpec {
            label: "Test",
            program_candidates: vec![PathBuf::from("/nonexistent/definitely/not/here")],
            args: vec!["serve"],
        };
        let err = ensure_running("http://127.0.0.1:1", &spec, || false).unwrap_err();
        assert!(err.contains("cannot find the Test launcher"), "got: {err}");
    }
}
