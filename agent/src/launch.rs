// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! `openhydra launch <tool>` — run a coding tool wired to the local OpenHydra gateway,
//! ollama-run style. It sets the tool's endpoint env vars so the tool's requests hit
//! OpenHydra's OpenAI-/Anthropic-compatible gateway, then execs the tool. The model defaults
//! to `openhydra/auto` (the gateway resolves it to a live model).
//!
//! Adding a tool is one `TOOLS` entry + (if its env differs) one arm in [`tool_env`].

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use clap::Args;

#[derive(Debug, Args)]
pub struct LaunchArgs {
    /// The coding tool to launch (e.g. `claude`, `opencode`). Omit only with `--list`.
    pub tool: Option<String>,
    /// List supported tools (and whether each is installed) and exit.
    #[arg(long)]
    pub list: bool,
    /// Model to request through OpenHydra — the gateway resolves `openhydra/auto` to a live model.
    #[arg(long, default_value = "openhydra/auto")]
    pub model: String,
    /// Gateway origin.
    #[arg(long, default_value = "http://127.0.0.1:16527")]
    pub gateway: String,
    /// API key presented to the gateway. Loopback gateways are open by default; a placeholder
    /// still satisfies tools that refuse to start without the var set.
    #[arg(long, default_value = "openhydra-local")]
    pub api_key: String,
    /// Print the env it would set (shell-eval'able) and exit — don't launch.
    #[arg(long)]
    pub print_env: bool,
    /// Arguments after `--` are forwarded verbatim to the tool.
    #[arg(last = true)]
    pub args: Vec<String>,
}

#[derive(Clone, Copy)]
enum Api {
    OpenAi,
    Anthropic,
}

struct ToolSpec {
    key: &'static str,
    bins: &'static [&'static str],
    api: Api,
    /// Whether the tool takes the model via env (so `--model` actually pins it). When false we
    /// can only set the endpoint; the user picks the model in the tool's own config.
    sets_model: bool,
    install_hint: &'static str,
}

/// The supported tools. Kept deliberately small + accurate — an entry asserts a real env
/// contract, so only add a tool once its wiring is verified.
const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        // Claude Code validates the model id client-side and rejects non-`claude-*` names, so we
        // do NOT pin ANTHROPIC_MODEL — it uses its own default id, which the gateway's
        // `/v1/messages` bridges to a live OpenHydra model. Pin one with OPENHYDRA_AUTO_MODEL.
        key: "claude",
        bins: &["claude"],
        api: Api::Anthropic,
        sets_model: false,
        install_hint: "npm install -g @anthropic-ai/claude-code",
    },
    ToolSpec {
        key: "opencode",
        bins: &["opencode"],
        api: Api::OpenAi,
        sets_model: false,
        install_hint: "see https://opencode.ai",
    },
];

/// The env a tool needs to route through OpenHydra. Anthropic tools point at the origin (they
/// append `/v1/messages`); OpenAI tools point at `<origin>/v1`.
fn tool_env(spec: &ToolSpec, origin: &str, model: &str, key: &str) -> Vec<(String, String)> {
    match spec.api {
        Api::Anthropic => {
            let mut e = vec![
                ("ANTHROPIC_BASE_URL".to_string(), origin.to_string()),
                ("ANTHROPIC_API_KEY".to_string(), key.to_string()),
            ];
            if spec.sets_model {
                e.push(("ANTHROPIC_MODEL".to_string(), model.to_string()));
            }
            e
        }
        Api::OpenAi => vec![
            ("OPENAI_BASE_URL".to_string(), format!("{origin}/v1")),
            ("OPENAI_API_KEY".to_string(), key.to_string()),
        ],
    }
}

fn spec_for(tool: &str) -> Option<&'static ToolSpec> {
    let t = tool.to_ascii_lowercase();
    TOOLS.iter().find(|s| s.key == t)
}

/// Resolve a tool binary to a runnable path — PATH first (a shell-launched CLI has it), then
/// common install dirs + node/pipx/bun user bins (covers a minimal env, same rationale as the
/// desktop engine resolver).
fn resolve_tool_bin(bins: &[&str]) -> Option<PathBuf> {
    let probe = if cfg!(windows) { "where" } else { "which" };
    for b in bins {
        // Capture the *full resolved path* (with extension on Windows, e.g. `…\claude.cmd`) — a bare
        // name won't run on Windows, and the full path is what `exec_tool` needs to detect a shim.
        if let Ok(out) = Command::new(probe).arg(b).stderr(Stdio::null()).output() {
            if out.status.success() {
                if let Some(line) = String::from_utf8_lossy(&out.stdout).lines().next() {
                    let p = line.trim();
                    if !p.is_empty() {
                        return Some(PathBuf::from(p));
                    }
                }
            }
        }
        #[cfg(not(windows))]
        {
            for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"] {
                let c = Path::new(dir).join(b);
                if c.exists() {
                    return Some(c);
                }
            }
        }
        if let Ok(home) = std::env::var("HOME") {
            for rel in [".npm-global/bin", ".local/bin", ".bun/bin"] {
                let c = Path::new(&home).join(rel).join(b);
                if c.exists() {
                    return Some(c);
                }
            }
        }
    }
    None
}

/// Best-effort TCP reachability of the gateway origin. Unparseable/unresolvable → treat as
/// reachable (don't nag); this only warns on a clear "nothing is listening".
fn gateway_reachable(origin: &str) -> bool {
    use std::net::ToSocketAddrs;
    let hostport = origin
        .split("://")
        .nth(1)
        .unwrap_or(origin)
        .split('/')
        .next()
        .unwrap_or("");
    if !hostport.contains(':') {
        return true;
    }
    // Try EVERY resolved address (a dual-stack host like `localhost` often yields `::1` first while
    // the gateway listens only on IPv4 — probing just the first would falsely warn). A refused
    // connection returns immediately, so this stays fast; `.any` short-circuits on the first hit.
    match hostport.to_socket_addrs() {
        Ok(addrs) => addrs.into_iter().any(|a| {
            std::net::TcpStream::connect_timeout(&a, std::time::Duration::from_millis(500)).is_ok()
        }),
        Err(_) => true,
    }
}

fn shell_quote(v: &str) -> String {
    format!("'{}'", v.replace('\'', "'\\''"))
}

fn print_tools() {
    println!("Supported tools (openhydra launch <tool>):");
    for s in TOOLS {
        let found = if resolve_tool_bin(s.bins).is_some() { "installed" } else { "not found" };
        let api = match s.api {
            Api::OpenAi => "OpenAI",
            Api::Anthropic => "Anthropic",
        };
        println!("  {:10} {:9} {}", s.key, api, found);
    }
}

/// Entry point for the `launch` subcommand. On success it execs the tool (never returns on unix).
pub fn run(args: LaunchArgs) -> Result<(), String> {
    if args.list {
        print_tools();
        return Ok(());
    }
    let tool = args.tool.as_deref().ok_or_else(|| {
        "usage: openhydra launch <tool> [-- <tool args>]   (see `openhydra launch --list`)"
            .to_string()
    })?;
    let spec = spec_for(tool).ok_or_else(|| {
        let keys: Vec<_> = TOOLS.iter().map(|s| s.key).collect();
        format!("unknown tool '{tool}'. Supported: {}", keys.join(", "))
    })?;
    let origin = args.gateway.trim_end_matches('/').to_string();
    let env = tool_env(spec, &origin, &args.model, &args.api_key);

    // `--print-env` is a dry run — show the wiring without needing the tool installed or a gateway up.
    if args.print_env {
        for (k, v) in &env {
            println!("export {k}={}", shell_quote(v));
        }
        println!("# then run: {}", spec.bins[0]);
        return Ok(());
    }

    let bin = resolve_tool_bin(spec.bins).ok_or_else(|| {
        format!("'{tool}' is not installed / not on PATH. Install it: {}", spec.install_hint)
    })?;

    if !gateway_reachable(&origin) {
        eprintln!(
            "openhydra launch: warning — no OpenHydra gateway at {origin}. Start the OpenHydra app, \
             or run `openhydra serve --bind 127.0.0.1:16527`. Launching {tool} anyway."
        );
    }
    if !spec.sets_model {
        match spec.api {
            Api::OpenAi => eprintln!(
                "openhydra launch: set {tool}'s model to '{}' in its config to route through OpenHydra.",
                args.model
            ),
            Api::Anthropic => eprintln!(
                "openhydra launch: {tool} sends its own model id; OpenHydra routes it to a live model \
                 (set OPENHYDRA_AUTO_MODEL on the gateway to pin one)."
            ),
        }
    }
    let model_note = if spec.sets_model {
        format!("model {}", args.model)
    } else {
        "model: OpenHydra-routed".to_string()
    };
    eprintln!("openhydra launch: starting {tool} → {origin} ({model_note})");

    exec_tool(&bin, &args.args, &env)
}

#[cfg(unix)]
fn exec_tool(bin: &Path, args: &[String], env: &[(String, String)]) -> Result<(), String> {
    use std::os::unix::process::CommandExt;
    let mut cmd = Command::new(bin);
    cmd.args(args);
    for (k, v) in env {
        cmd.env(k, v);
    }
    // exec replaces this process; it only returns on failure.
    Err(format!("failed to launch: {}", cmd.exec()))
}

#[cfg(windows)]
fn exec_tool(bin: &Path, args: &[String], env: &[(String, String)]) -> Result<(), String> {
    // npm/yarn/bun global installs create `.cmd`/`.bat` shims that CreateProcess can't execute
    // directly — those must go through `cmd /C`. A real `.exe` is launched directly.
    let is_shim = bin
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let e = e.to_ascii_lowercase();
            e == "cmd" || e == "bat"
        })
        .unwrap_or(false);
    let mut cmd = if is_shim {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(bin).args(args);
        c
    } else {
        let mut c = Command::new(bin);
        c.args(args);
        c
    };
    for (k, v) in env {
        cmd.env(k, v);
    }
    match cmd.status() {
        Ok(s) => std::process::exit(s.code().unwrap_or(0)),
        Err(e) => Err(format!("failed to launch: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_lookup_is_case_insensitive() {
        assert!(spec_for("claude").is_some());
        assert!(spec_for("OpenCode").is_some());
        assert!(spec_for("nope").is_none());
    }

    #[test]
    fn anthropic_env_points_at_origin_without_pinning_model() {
        // Claude Code rejects non-`claude-*` model ids client-side, so we set only the endpoint +
        // key and let the gateway bridge whatever id Claude Code sends. No ANTHROPIC_MODEL.
        let s = spec_for("claude").unwrap();
        let e = tool_env(s, "http://127.0.0.1:16527", "openhydra/auto", "k");
        assert!(e.contains(&("ANTHROPIC_BASE_URL".into(), "http://127.0.0.1:16527".into())));
        assert!(e.contains(&("ANTHROPIC_API_KEY".into(), "k".into())));
        assert!(!e.iter().any(|(k, _)| k == "ANTHROPIC_MODEL"));
    }

    #[test]
    fn openai_env_appends_v1_and_omits_model() {
        let s = spec_for("opencode").unwrap();
        let e = tool_env(s, "http://127.0.0.1:16527", "openhydra/auto", "k");
        assert!(e.contains(&("OPENAI_BASE_URL".into(), "http://127.0.0.1:16527/v1".into())));
        assert!(e.contains(&("OPENAI_API_KEY".into(), "k".into())));
        assert!(!e.iter().any(|(k, _)| k.contains("MODEL")));
    }

    #[test]
    fn shell_quote_escapes_single_quotes() {
        assert_eq!(shell_quote("a'b"), "'a'\\''b'");
    }

    #[test]
    fn gateway_reachable_skips_probe_when_no_port() {
        assert!(gateway_reachable("not-a-url"));
    }
}
