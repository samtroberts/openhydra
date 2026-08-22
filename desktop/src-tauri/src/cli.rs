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

/// Resolve the FULL path `openhydra` runs from (for display). `resolve_program` returns just the bare
/// name on its `which`-success path, so ask the shell's resolver directly to capture the real path;
/// fall back to `resolve_program`'s common-dir hits (already absolute).
fn resolved_path() -> Option<PathBuf> {
    let probe = if cfg!(windows) { "where" } else { "which" };
    if let Ok(out) = std::process::Command::new(probe).arg(CLI_NAME).output() {
        if out.status.success() {
            if let Some(line) = String::from_utf8_lossy(&out.stdout).lines().next() {
                let p = line.trim();
                if !p.is_empty() {
                    return Some(PathBuf::from(p));
                }
            }
        }
    }
    // A Finder-launched app has a minimal PATH, so `which` can miss a tool that IS installed under a
    // known dir — fall back to those (they return absolute paths).
    crate::installer::resolve_program(CLI_NAME).filter(|p| p.is_absolute())
}

/// Detection: is `openhydra` runnable, where from, and where would we install it?
pub fn status() -> CliStatus {
    let resolved = resolved_path();
    let target = default_target();
    // A dangling managed link: a symlink whose destination is gone (symlink_metadata succeeds for a
    // dangling link; metadata, which follows, fails). Check the primary target AND the ~/.local/bin
    // fallback, since an admin-declined macOS install (or any unix install) lands there.
    let mut broken_candidates = vec![target.clone()];
    #[cfg(unix)]
    broken_candidates.push(user_local_bin().join(CLI_NAME));
    let managed_broken = broken_candidates.iter().any(|p| is_dangling_symlink(p));
    CliStatus {
        on_path: resolved.is_some(),
        resolved: resolved.map(|p| p.display().to_string()),
        source: sidecar_path().map(|p| p.display().to_string()),
        target: target.display().to_string(),
        managed_broken,
    }
}

/// A symlink whose destination is gone: symlink_metadata succeeds for the link itself, metadata
/// (which follows it) fails. False for a regular file, a live symlink, or a missing path.
fn is_dangling_symlink(p: &Path) -> bool {
    std::fs::symlink_metadata(p).is_ok() && std::fs::metadata(p).is_err()
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
            // `target` is a constant today, but quote it safely regardless (defense-in-depth — this runs as root).
            let script = build_uninstall_script(&target.display().to_string());
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
/// Escape a string as an AppleScript double-quoted string LITERAL (the `\` / `"` layer).
#[cfg(target_os = "macos")]
fn as_applescript_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// An AppleScript expression yielding the SHELL-safe form of `s`: escaped for the AppleScript literal,
/// then `quoted form of` for the shell — so a path never reaches the shell parser un-quoted.
#[cfg(target_os = "macos")]
fn as_quoted_shell_arg(s: &str) -> String {
    format!("quoted form of {}", as_applescript_str(s))
}

/// Build the (root) install AppleScript. Pure — separated from `run_osascript` so the quoting is
/// unit-testable without side effects. The command is assembled in a variable so `with administrator
/// privileges` binds to the whole concatenation.
#[cfg(target_os = "macos")]
fn build_install_script(source: &str, target: &str) -> String {
    format!(
        "set cmd to \"mkdir -p /usr/local/bin && ln -sf \" & {} & \" \" & {}\n\
         do shell script cmd with administrator privileges",
        as_quoted_shell_arg(source),
        as_quoted_shell_arg(target),
    )
}

/// Build the (root) uninstall AppleScript. Pure, same quoting discipline.
#[cfg(target_os = "macos")]
fn build_uninstall_script(target: &str) -> String {
    format!(
        "do shell script \"rm -f \" & {} with administrator privileges",
        as_quoted_shell_arg(target)
    )
}

#[cfg(target_os = "macos")]
fn install_macos(source: &Path) -> Result<InstallReport, String> {
    // Primary: symlink into /usr/local/bin (on the default macOS PATH) with one admin prompt — exactly
    // VS Code's "Install 'code' command in PATH". Survives app updates (the .app path is stable).
    //
    // SECURITY: this runs as ROOT, and `source` is the folder the app runs from — which can contain
    // quotes / `;` / `$` (e.g. "~/Downloads/Sam's Apps/OpenHydra.app"). build_install_script quotes it
    // (AppleScript-literal escaping + `quoted form of`); it is never raw-interpolated into the shell.
    let script = build_install_script(&source.display().to_string(), "/usr/local/bin/openhydra");
    match run_osascript(&script) {
        Ok(()) => Ok(InstallReport {
            path: "/usr/local/bin/openhydra".into(),
            method: "symlink".into(),
            on_path: true,
            note: None,
        }),
        // The user CANCELLED the privilege prompt → do nothing and report it. We must NOT silently
        // fall back to writing ~/.local/bin + editing their shell rc behind a "success" toast.
        Err(e) if e == "cancelled" => Err("cancelled".into()),
        // A genuine failure (dir not writable, etc.) → fall back to the no-admin ~/.local/bin install.
        Err(e) => {
            let mut r = install_user_local(source)?;
            r.note = Some(match r.note.take() {
                Some(n) => format!("Used ~/.local/bin (admin install failed: {e}). {n}"),
                None => format!("Used ~/.local/bin (admin install failed: {e})."),
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
/// Build the PowerShell one-liner that adds `dir` to the USER Path. Pure (unit-testable off-Windows —
/// only the invocation in install_windows is Windows-gated). Doubles single quotes (usernames may
/// contain `'`), guards a $null Path, and matches an EXACT `;`-segment (not a `-like` wildcard, which
/// mishandles `[ ] * ?` and substring collisions). Idempotent — appends only when absent.
#[allow(dead_code)] // used only by install_windows (cfg windows) + the tests
fn build_win_path_add(dir: &str) -> String {
    let dir_lit = dir.replace('\'', "''");
    format!(
        "$d = '{dir_lit}'; $p = [Environment]::GetEnvironmentVariable('Path','User'); if (-not $p) {{ $p = '' }}; \
         if (($p -split ';') -notcontains $d) {{ if ($p) {{ $p = $p.TrimEnd(';') + ';' + $d }} else {{ $p = $d }}; \
         [Environment]::SetEnvironmentVariable('Path', $p, 'User') }}"
    )
}

#[cfg(target_os = "windows")]
fn install_windows(source: &Path) -> Result<InstallReport, String> {
    let bindir = windows_bin_dir();
    std::fs::create_dir_all(&bindir).map_err(|e| format!("mkdir {}: {e}", bindir.display()))?;
    let target = bindir.join(format!("{CLI_NAME}.exe"));
    std::fs::copy(source, &target).map_err(|e| format!("copy: {e}"))?;
    // Add the bin dir to the USER PATH (no admin) via PowerShell's Environment API, then broadcast so
    // new shells pick it up. Idempotent — only appends when absent.
    let ps = build_win_path_add(&bindir.display().to_string());
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

/// The rc files an interactive shell of `shell` (a $SHELL path) ACTUALLY sources. zsh sources
/// ~/.zshrc for both login and interactive; bash needs ~/.bashrc (interactive, most Linux terminals)
/// AND ~/.bash_profile (login, e.g. macOS Terminal). A login-only file like ~/.zprofile silently
/// misses non-login shells. Pure — unit-tested.
#[cfg(unix)]
fn rc_files_for_shell(shell: &str) -> &'static [&'static str] {
    if shell.ends_with("zsh") {
        &[".zshrc"]
    } else if shell.ends_with("bash") {
        &[".bashrc", ".bash_profile"]
    } else {
        &[".profile"]
    }
}

/// Ensure `~/.local/bin` is on PATH by appending an export to the shell rc files interactive shells
/// source (idempotent, gated on our own marker). Returns a user-facing note. macOS/Linux only.
#[cfg(unix)]
fn ensure_local_bin_on_path() -> Option<String> {
    // Already resolvable → nothing to do.
    if crate::installer::resolve_program(CLI_NAME).is_some() {
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    let shell = std::env::var("SHELL").unwrap_or_default();
    let rc_files = rc_files_for_shell(&shell);
    const MARKER: &str = "# added by OpenHydra — put the openhydra CLI on PATH";
    let line = "export PATH=\"$HOME/.local/bin:$PATH\"";
    let mut wrote = Vec::new();
    for rc in rc_files {
        let rc_path = Path::new(&home).join(rc);
        let existing = std::fs::read_to_string(&rc_path).unwrap_or_default();
        // Idempotent on OUR marker (not a generic ".local/bin" substring, which false-skips on a
        // comment or an unrelated line and leaves PATH unfixed).
        if existing.contains(MARKER) {
            continue;
        }
        let mut block = String::new();
        if !existing.is_empty() && !existing.ends_with('\n') {
            block.push('\n');
        }
        block.push('\n');
        block.push_str(MARKER);
        block.push('\n');
        block.push_str(line);
        block.push('\n');
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&rc_path) {
            use std::io::Write;
            if f.write_all(block.as_bytes()).is_ok() {
                wrote.push(format!("~/{rc}"));
            }
        }
    }
    if wrote.is_empty() {
        Some("Open a new terminal to use `openhydra`.".into())
    } else {
        Some(format!(
            "Added ~/.local/bin to your PATH in {} — open a new terminal to use `openhydra`.",
            wrote.join(" and ")
        ))
    }
}

// ── tests ──────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── macOS AppleScript quoting (the root-injection regression guard) ──
    #[cfg(target_os = "macos")]
    #[test]
    fn applescript_literal_escapes_backslash_and_quote() {
        assert_eq!(as_applescript_str("plain"), "\"plain\"");
        assert_eq!(as_applescript_str("a\"b"), "\"a\\\"b\"");   // " -> \"
        assert_eq!(as_applescript_str("a\\b"), "\"a\\\\b\"");   // \ -> \\
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn install_script_quotes_and_never_raw_interpolates() {
        let malicious = "/tmp/x'; touch /tmp/pwned; :";
        let s = build_install_script(malicious, "/usr/local/bin/openhydra");
        assert!(s.contains("quoted form of"), "must use `quoted form of` for the shell layer");
        assert!(s.contains("with administrator privileges"));
        // the path must NOT sit inside a shell-active single-quoted literal (the old vulnerable shape)
        assert!(!s.contains("ln -sf '/tmp/x'"), "path must not be raw-interpolated into the shell");
    }

    // The real proof: build the SAME quoting the installer uses and let osascript evaluate it (with
    // `echo` in place of the root `ln`). A crafted path must be treated as DATA, not commands.
    #[cfg(target_os = "macos")]
    #[test]
    fn osascript_quoting_blocks_command_injection() {
        // Unambiguous proof via a SIDE EFFECT: the payload tries to `touch` a marker file. If the path
        // were injected the file would be created; safe quoting echoes it as literal data and it isn't.
        let marker = std::env::temp_dir().join(format!("oh-inj-{}", std::process::id()));
        let _ = std::fs::remove_file(&marker);
        let payload = format!("/tmp/x'; touch {}; echo '", marker.display());
        let script = format!("set cmd to \"echo \" & {}\ndo shell script cmd", as_quoted_shell_arg(&payload));
        let out = std::process::Command::new("osascript").arg("-e").arg(&script).output().unwrap();
        assert!(out.status.success(), "osascript failed: {}", String::from_utf8_lossy(&out.stderr));
        // And the whole payload should come back echoed as one literal line.
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim_end(), payload, "path must be echoed literally");
        let injected = marker.exists();
        let _ = std::fs::remove_file(&marker);
        assert!(!injected, "the `touch` payload must NOT have executed — that would be root injection");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn uninstall_script_is_quoted() {
        let s = build_uninstall_script("/usr/local/bin/openhydra");
        assert!(s.contains("quoted form of") && s.contains("rm -f") && s.contains("with administrator privileges"));
    }

    // ── Windows PowerShell PATH builder (pure — runs on any host) ──
    #[test]
    fn win_path_add_escapes_quotes_and_matches_exact_segment() {
        let ps = build_win_path_add(r"C:\Users\O'Brien\AppData\Local\OpenHydra\bin");
        assert!(ps.contains("O''Brien"), "single quote must be doubled: {ps}");
        assert!(ps.contains("-split ';'") && ps.contains("-notcontains"), "exact-segment match, not -like");
        assert!(ps.contains("if (-not $p)"), "must guard a $null user Path");
        assert!(!ps.contains("-like"), "must not use a -like wildcard test");
    }

    // ── rc-file selection (#4) ──
    #[cfg(unix)]
    #[test]
    fn rc_files_target_interactive_shells() {
        assert_eq!(rc_files_for_shell("/bin/zsh").to_vec(), vec![".zshrc"]);
        assert_eq!(rc_files_for_shell("/usr/bin/bash").to_vec(), vec![".bashrc", ".bash_profile"]);
        assert_eq!(rc_files_for_shell("/usr/bin/fish").to_vec(), vec![".profile"]);
        assert_eq!(rc_files_for_shell("").to_vec(), vec![".profile"]);
    }

    // ── dangling-symlink detection (#8) ──
    #[cfg(unix)]
    #[test]
    fn dangling_symlink_detection() {
        use std::os::unix::fs::symlink;
        let dir = std::env::temp_dir().join(format!("oh-cli-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let real = dir.join("real");
        std::fs::write(&real, b"x").unwrap();
        let live = dir.join("live");
        symlink(&real, &live).unwrap();
        let dangling = dir.join("dangling");
        symlink(dir.join("gone"), &dangling).unwrap();

        assert!(!is_dangling_symlink(&live), "a live symlink is not dangling");
        assert!(is_dangling_symlink(&dangling), "a symlink to a missing target is dangling");
        assert!(!is_dangling_symlink(&real), "a regular file is not a dangling symlink");
        assert!(!is_dangling_symlink(&dir.join("nope")), "a missing path is not a dangling symlink");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
