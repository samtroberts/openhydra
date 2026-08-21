// Expose the bundled `openhydra-agent` sidecar as an `openhydra` command on the user's PATH.
//
// The app already ships the FULL CLI as a Tauri sidecar sitting next to the desktop binary
// (macOS `Contents/MacOS/`, Linux next to the exe, Windows the install dir). But nothing puts it on
// PATH, so app-installed users can't run the Connectors "Terminal" snippets or `openhydra launch`.
// This module is the "Install command-line tool" action (the VS Code *Install 'code' command* model):
// it links/copies that sidecar into a PATH directory, per platform. See docs/CLI_ON_PATH_PLAN_v1.md.

use serde::Serialize;
use std::path::{Path, PathBuf};

/// The command name we expose (matches install.sh, the Connectors snippets, and the docs).
const CLI_NAME: &str = "openhydra";
/// The bundled sidecar's file name (Tauri strips the target-triple suffix at bundle time).
const AGENT_BIN: &str = if cfg!(windows) { "openhydra-agent.exe" } else { "openhydra-agent" };

#[derive(Serialize)]
pub struct CliStatus {
    /// `openhydra` resolves on PATH right now (via `which`/`where` + the common GUI-PATH dirs).
    pub on_path: bool,
    /// Where it currently resolves (if `on_path`) — surfaces a shadowing binary.
    pub resolved: Option<String>,
    /// The bundled sidecar we'd link/copy from (None if it can't be located — e.g. a broken build).
    pub source: Option<String>,
    /// Where `install_cli` would place the `openhydra` command on this OS.
    pub target: String,
    /// True when OUR managed link/shim exists but points at a missing binary (e.g. the app moved) —
    /// the UI can offer to re-point or remove it.
    pub managed_broken: bool,
}

#[derive(Serialize)]
pub struct InstallReport {
    /// Absolute path of the installed command.
    pub path: String,
    /// How it was installed: "symlink" | "copy" | "shim".
    pub method: String,
    /// True if the command is expected to resolve on PATH in a NEW shell right after this.
    pub on_path: bool,
    /// A user-facing note when a follow-up is needed (e.g. "open a new terminal", "we edited ~/.zprofile").
    pub note: Option<String>,
}

/// Path to the bundled agent sidecar — it sits next to the desktop binary in every Tauri bundle,
/// and next to it in `target/release` during `tauri dev` (the `npm run sidecar` step copies it there).
pub fn sidecar_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let cand = exe.parent()?.join(AGENT_BIN);
    cand.exists().then_some(cand)
}

/// Where `install_cli` places the command on this OS (the primary target; macOS may fall back to
/// ~/.local/bin if the admin prompt is declined).
fn default_target() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        PathBuf::from("/usr/local/bin").join(CLI_NAME)
    }
    #[cfg(target_os = "linux")]
    {
        user_local_bin().join(CLI_NAME)
    }
    #[cfg(target_os = "windows")]
    {
        windows_bin_dir().join(format!("{CLI_NAME}.exe"))
    }
}

#[cfg(unix)]
fn user_local_bin() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    Path::new(&home).join(".local/bin")
}

#[cfg(target_os = "windows")]
fn windows_bin_dir() -> PathBuf {
    let base = std::env::var("LOCALAPPDATA").unwrap_or_default();
    Path::new(&base).join("OpenHydra").join("bin")
}

/// Detection: is `openhydra` runnable, where from, and where would we install it?
pub fn status() -> CliStatus {
    let resolved = crate::installer::resolve_program(CLI_NAME);
    let target = default_target();
    // A dangling managed link: our target path exists as a symlink but its destination is gone.
    let managed_broken = {
        let p = &target;
        // symlink_metadata succeeds for a dangling link; metadata (follows) fails.
        std::fs::symlink_metadata(p).is_ok() && std::fs::metadata(p).is_err()
    };
    CliStatus {
        on_path: resolved.is_some(),
        resolved: resolved.map(|p| p.display().to_string()),
        source: sidecar_path().map(|p| p.display().to_string()),
        target: target.display().to_string(),
        managed_broken,
    }
}

/// Install the `openhydra` command onto PATH. Per-platform mechanics (see the plan doc).
pub fn install() -> Result<InstallReport, String> {
    let source =
        sidecar_path().ok_or_else(|| "couldn't locate the bundled openhydra-agent binary".to_string())?;

    #[cfg(target_os = "macos")]
    {
        install_macos(&source)
    }
    #[cfg(target_os = "linux")]
    {
        install_linux(&source)
    }
    #[cfg(target_os = "windows")]
    {
        install_windows(&source)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = source;
        Err("unsupported OS".into())
    }
}

/// Remove the managed `openhydra` command.
pub fn uninstall() -> Result<(), String> {
    let target = default_target();
    #[cfg(target_os = "macos")]
    {
        // The primary target lives in root-owned /usr/local/bin → needs admin to remove; also try the
        // user-local fallback location without a prompt.
        let user = user_local_bin().join(CLI_NAME);
        let _ = std::fs::remove_file(&user);
        if std::fs::symlink_metadata(&target).is_ok() {
            let script = format!(
                "do shell script \"rm -f '{}'\" with administrator privileges",
                target.display()
            );
            run_osascript(&script)?;
        }
        Ok(())
    }
    #[cfg(not(target_os = "macos"))]
    {
        std::fs::remove_file(&target).map_err(|e| format!("couldn't remove {}: {e}", target.display()))
    }
}

// ── macOS ─────────────────────────────────────────────────────────────────────
#[cfg(target_os = "macos")]
fn install_macos(source: &Path) -> Result<InstallReport, String> {
    let target = "/usr/local/bin/openhydra";
    // Primary: symlink into /usr/local/bin (on the default macOS PATH) with one admin prompt — exactly
    // VS Code's "Install 'code' command in PATH". Survives app updates (the .app path is stable).
    let script = format!(
        "do shell script \"mkdir -p /usr/local/bin && ln -sf '{}' '{}'\" with administrator privileges",
        source.display(),
        target
    );
    match run_osascript(&script) {
        Ok(()) => Ok(InstallReport {
            path: target.into(),
            method: "symlink".into(),
            on_path: true,
            note: None,
        }),
        // -128 = the user cancelled the admin prompt; anything else (not writable, etc.) → fall back
        // to a no-admin ~/.local/bin install.
        Err(e) => {
            let mut r = install_user_local(source)?;
            r.note = Some(match r.note.take() {
                Some(n) => format!("Used ~/.local/bin (admin install skipped: {e}). {n}"),
                None => format!("Used ~/.local/bin (admin install skipped: {e})."),
            });
            Ok(r)
        }
    }
}

#[cfg(target_os = "macos")]
fn run_osascript(script: &str) -> Result<(), String> {
    let out = std::process::Command::new("osascript")
        .arg("-e")
        .arg(script)
        .output()
        .map_err(|e| format!("osascript failed to launch: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    if err.contains("-128") || err.contains("User canceled") {
        return Err("cancelled".into());
    }
    Err(err.trim().to_string())
}

// ── user-local (~/.local/bin) symlink + PATH edit — the no-admin unix fallback ──
#[cfg(unix)]
fn install_user_local(source: &Path) -> Result<InstallReport, String> {
    let bindir = user_local_bin();
    std::fs::create_dir_all(&bindir).map_err(|e| format!("mkdir {}: {e}", bindir.display()))?;
    let target = bindir.join(CLI_NAME);
    let _ = std::fs::remove_file(&target); // replace an existing link/file
    std::os::unix::fs::symlink(source, &target).map_err(|e| format!("symlink: {e}"))?;
    let note = ensure_local_bin_on_path();
    Ok(InstallReport {
        path: target.display().to_string(),
        method: "symlink".into(),
        on_path: crate::installer::resolve_program(CLI_NAME).is_some(),
        note,
    })
}

// ── Linux ─────────────────────────────────────────────────────────────────────
#[cfg(target_os = "linux")]
fn install_linux(source: &Path) -> Result<InstallReport, String> {
    let bindir = user_local_bin();
    std::fs::create_dir_all(&bindir).map_err(|e| format!("mkdir {}: {e}", bindir.display()))?;
    let target = bindir.join(CLI_NAME);
    let _ = std::fs::remove_file(&target);
    // Running from an AppImage: the sidecar lives inside the ephemeral mount (a new random path each
    // run), so a symlink would dangle next launch → COPY the binary out instead.
    if std::env::var_os("APPIMAGE").is_some() {
        std::fs::copy(source, &target).map_err(|e| format!("copy: {e}"))?;
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755));
    } else {
        std::os::unix::fs::symlink(source, &target).map_err(|e| format!("symlink: {e}"))?;
    }
    let note = ensure_local_bin_on_path();
    Ok(InstallReport {
        path: target.display().to_string(),
        method: if std::env::var_os("APPIMAGE").is_some() { "copy".into() } else { "symlink".into() },
        on_path: crate::installer::resolve_program(CLI_NAME).is_some(),
        note,
    })
}

// ── Windows ───────────────────────────────────────────────────────────────────
#[cfg(target_os = "windows")]
fn install_windows(source: &Path) -> Result<InstallReport, String> {
    let bindir = windows_bin_dir();
    std::fs::create_dir_all(&bindir).map_err(|e| format!("mkdir {}: {e}", bindir.display()))?;
    let target = bindir.join(format!("{CLI_NAME}.exe"));
    std::fs::copy(source, &target).map_err(|e| format!("copy: {e}"))?;
    // Add the bin dir to the USER PATH (no admin) via PowerShell's Environment API, then broadcast so
    // new shells pick it up. Idempotent — only appends when absent.
    let dir = bindir.display().to_string();
    let ps = format!(
        "$d='{dir}'; $p=[Environment]::GetEnvironmentVariable('Path','User'); \
         if ($p -notlike \"*$d*\") {{ [Environment]::SetEnvironmentVariable('Path', ($p.TrimEnd(';') + ';' + $d), 'User') }}"
    );
    let status = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", &ps])
        .status()
        .map_err(|e| format!("powershell failed: {e}"))?;
    let on_path = status.success();
    Ok(InstallReport {
        path: target.display().to_string(),
        method: "copy".into(),
        on_path,
        note: Some("Open a new terminal for the updated PATH to take effect.".into()),
    })
}

/// Ensure `~/.local/bin` is on PATH by appending an export to the login shell's rc file (idempotent).
/// Returns a user-facing note. macOS/Linux only (that dir isn't on the stock macOS PATH).
#[cfg(unix)]
fn ensure_local_bin_on_path() -> Option<String> {
    // Already resolvable → nothing to do.
    if crate::installer::resolve_program(CLI_NAME).is_some() {
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    let shell = std::env::var("SHELL").unwrap_or_default();
    let rc = if shell.ends_with("zsh") {
        ".zprofile"
    } else if shell.ends_with("bash") {
        ".bash_profile"
    } else {
        ".profile"
    };
    let rc_path = Path::new(&home).join(rc);
    let line = "export PATH=\"$HOME/.local/bin:$PATH\"";
    let existing = std::fs::read_to_string(&rc_path).unwrap_or_default();
    if !existing.contains(".local/bin") {
        let mut block = String::new();
        if !existing.is_empty() && !existing.ends_with('\n') {
            block.push('\n');
        }
        block.push_str("\n# added by OpenHydra — put the openhydra CLI on PATH\n");
        block.push_str(line);
        block.push('\n');
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&rc_path) {
            use std::io::Write;
            let _ = f.write_all(block.as_bytes());
        }
        return Some(format!(
            "Added ~/.local/bin to your PATH in ~/{rc} — open a new terminal (or run `source ~/{rc}`) to use `openhydra`."
        ));
    }
    Some("Open a new terminal to use `openhydra`.".into())
}
