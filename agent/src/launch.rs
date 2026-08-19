// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! `openhydra launch <tool>` — run a coding tool wired to the local OpenHydra gateway,
//! ollama-run style. **`launch` = `connect` + run:** it ALWAYS persists the OpenHydra block into the
//! tool's config file via the shared [`openhydra_agent::connect`] writers (idempotent, backed up) —
//! so the tool stays wired across future sessions, not just this one — then execs the tool.
//!   • Env tools (Claude Code, OpenCode) additionally get an endpoint env var set on the spawned
//!     process: a redundant, immediate guarantee independent of when the tool re-reads its config.
//!   • Config tools (Hermes, Pi) are steered by the written file (plus selecting the OpenHydra
//!     provider on the CLI where needed, e.g. Pi's `--provider openhydra`).
//! The model defaults to `openhydra/auto` (the gateway resolves it to a live model).
//!
//! Adding a tool is one `TOOLS` entry (+ an env arm in [`tool_env`] only if it's an env tool).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use clap::Args;

use openhydra_agent::connect;

#[derive(Debug, Args)]
pub struct LaunchArgs {
    /// The coding tool to launch (claude, opencode, hermes, pi). Omit only with `--list`.
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
    /// Print the wiring it would apply (env / config target) and exit — don't launch or write.
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

/// How `launch` wires a tool to the gateway before exec.
#[derive(Clone, Copy)]
enum Wiring {
    /// The tool reads an endpoint env var (OPENAI_/ANTHROPIC_BASE_URL): set it, then exec. Ephemeral.
    Env(Api),
    /// The tool is configured by a file, not env: persist the OpenHydra block via `connect`
    /// (idempotent, backs up), then exec. Used for config-only tools (Hermes, Pi).
    Config,
}

struct ToolSpec {
    key: &'static str,
    bins: &'static [&'static str],
    wiring: Wiring,
    /// Whether the tool takes the model via env (so `--model` actually pins it). Env tools only.
    sets_model: bool,
    /// Args prepended before the user's forwarded `--` args; a `{model}` token is replaced with the
    /// `--model` value. Lets a config tool select the OpenHydra provider on the CLI (Pi).
    default_args: &'static [&'static str],
    install_hint: &'static str,
}

/// The supported tools. Kept deliberately small + accurate — an entry asserts a real wiring
/// contract, so only add a tool once its wiring is verified.
const TOOLS: &[ToolSpec] = &[
    ToolSpec {
        // Claude Code validates the model id client-side and rejects non-`claude-*` names, so we
        // do NOT pin ANTHROPIC_MODEL — it uses its own default id, which the gateway's
        // `/v1/messages` bridges to a live OpenHydra model. Pin one with OPENHYDRA_AUTO_MODEL.
        key: "claude",
        bins: &["claude"],
        wiring: Wiring::Env(Api::Anthropic),
        sets_model: false,
        default_args: &[],
        install_hint: "npm install -g @anthropic-ai/claude-code",
    },
    ToolSpec {
        key: "opencode",
        bins: &["opencode"],
        wiring: Wiring::Env(Api::OpenAi),
        sets_model: false,
        default_args: &[],
        install_hint: "see https://opencode.ai",
    },
    ToolSpec {
        // Hermes reads ~/.hermes/config.yaml (no endpoint env var); `connect` writes the OpenHydra
        // `model:` block, which becomes Hermes' active model — so a bare `hermes` routes through us.
        key: "hermes",
        bins: &["hermes", "hermes-agent"],
        wiring: Wiring::Config,
        sets_model: false,
        default_args: &[],
        install_hint: "see https://github.com/NousResearch/hermes",
    },
    ToolSpec {
        // Pi reads ~/.pi/agent/models.json; `connect` writes the `openhydra` provider, then we select
        // it on the CLI (`--provider openhydra --model <model>`).
        key: "pi",
        bins: &["pi"],
        wiring: Wiring::Config,
        sets_model: false,
        default_args: &["--provider", "openhydra", "--model", "{model}"],
        install_hint: "curl -fsSL https://pi.dev/install.sh | sh",
    },
];

/// The env a tool needs to route through OpenHydra. Anthropic tools point at the origin (they
/// append `/v1/messages`); OpenAI tools point at `<origin>/v1`. Config tools take no env.
fn tool_env(spec: &ToolSpec, origin: &str, model: &str, key: &str) -> Vec<(String, String)> {
    match spec.wiring {
        Wiring::Env(Api::Anthropic) => {
            let mut e = vec![
                ("ANTHROPIC_BASE_URL".to_string(), origin.to_string()),
                ("ANTHROPIC_API_KEY".to_string(), key.to_string()),
            ];
            if spec.sets_model {
                e.push(("ANTHROPIC_MODEL".to_string(), model.to_string()));
            }
            e
        }
        Wiring::Env(Api::OpenAi) => vec![
            ("OPENAI_BASE_URL".to_string(), format!("{origin}/v1")),
            ("OPENAI_API_KEY".to_string(), key.to_string()),
        ],
        Wiring::Config => vec![],
    }
}

/// Prepend a tool's `default_args` (with `{model}` substituted) to the user's forwarded args.
fn build_argv(spec: &ToolSpec, model: &str, user_args: &[String]) -> Vec<String> {
    // If the user forwards their own `--model`, don't ALSO inject ours — two `--model` flags confuse
    // the tool. Drop the injected `--model {model}` pair (keep the rest, e.g. Pi's `--provider`).
    let user_sets_model = user_args.iter().any(|a| a == "--model");
    let mut argv: Vec<String> = Vec::new();
    let mut skip_next = false;
    for a in spec.default_args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if user_sets_model && *a == "--model" {
            skip_next = true; // also skip the following `{model}`
            continue;
        }
        argv.push(if *a == "{model}" { model.to_string() } else { a.to_string() });
    }
    argv.extend(user_args.iter().cloned());
    argv
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
                let text = String::from_utf8_lossy(&out.stdout);
                let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
                // On Windows `where` may list an extensionless bash shim before the runnable
                // `.cmd`/`.exe`; prefer a directly-runnable extension. Elsewhere the one `which`
                // line is the answer.
                let picked = if cfg!(windows) {
                    lines
                        .iter()
                        .find(|l| {
                            let low = l.to_ascii_lowercase();
                            low.ends_with(".exe") || low.ends_with(".cmd") || low.ends_with(".bat")
                        })
                        .or_else(|| lines.first())
                        .copied()
                } else {
                    lines.first().copied()
                };
                if let Some(p) = picked {
                    return Some(PathBuf::from(p));
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
            // `.opencode/bin` = OpenCode's official installer; `.local/bin` = npm-user / curl
            // installers (pi, hermes); `.bun/bin`, `.npm-global/bin` = other JS toolchains.
            for rel in [".opencode/bin", ".npm-global/bin", ".local/bin", ".bun/bin"] {
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

fn wiring_label(w: Wiring) -> &'static str {
    match w {
        Wiring::Env(Api::OpenAi) => "OpenAI (env)",
        Wiring::Env(Api::Anthropic) => "Anthropic (env)",
        Wiring::Config => "config file",
    }
}

fn print_tools() {
    println!("Supported tools (openhydra launch <tool>):");
    for s in TOOLS {
        let found = if resolve_tool_bin(s.bins).is_some() { "installed" } else { "not found" };
        println!("  {:10} {:16} {}", s.key, wiring_label(s.wiring), found);
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

    // `--print-env` is a dry run — show the wiring without needing the tool installed, a gateway up,
    // or (for config tools) touching any file.
    if args.print_env {
        match spec.wiring {
            Wiring::Env(_) => {
                for (k, v) in &env {
                    println!("export {k}={}", shell_quote(v));
                }
                let path = connect::spec(spec.key)
                    .and_then(|s| connect::config_path(s.kind))
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "<config file>".into());
                println!("# launch also persists {tool}'s config so it stays wired: {path}");
                println!("# then run: {}", spec.bins[0]);
            }
            Wiring::Config => {
                let path = connect::spec(spec.key)
                    .and_then(|s| connect::config_path(s.kind))
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "<config file>".into());
                println!("# {tool} is wired via its config file (no env): {path}");
                println!("# `openhydra launch {tool}` writes the OpenHydra block there, then runs it.");
            }
        }
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

    // Wire the tool: launch = connect + run. ALWAYS persist the OpenHydra block into the tool's config
    // (idempotent, backed up) via the SAME writers as `openhydra connect` / the desktop button — so the
    // tool stays wired across future sessions, not just this one. Env tools ADDITIONALLY get endpoint
    // env vars on the spawned process (see `env`): a redundant, immediate guarantee independent of when
    // the tool re-reads its config.
    let rep = connect::apply(spec.key, &origin)?;
    let backup = rep
        .backup
        .as_ref()
        .map(|b| format!(" (original backed up → {b})"))
        .unwrap_or_default();
    eprintln!("openhydra launch: wired {tool} config → {}{}", rep.path, backup);
    if matches!(spec.wiring, Wiring::Env(Api::Anthropic)) && !spec.sets_model {
        eprintln!(
            "openhydra launch: {tool} sends its own model id; OpenHydra routes it to a live model \
             (set OPENHYDRA_AUTO_MODEL on the gateway to pin one)."
        );
    }

    let model_note = match spec.wiring {
        Wiring::Config => "model: openhydra/auto (via config)".to_string(),
        _ if spec.sets_model => format!("model {}", args.model),
        _ => "model: OpenHydra-routed".to_string(),
    };
    eprintln!("openhydra launch: starting {tool} → {origin} ({model_note})");

    let argv = build_argv(spec, &args.model, &args.args);
    exec_tool(&bin, &argv, &env)
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
    // Rust's std runs `.cmd`/`.bat` shims correctly (with batch-safe arg escaping) since 1.77.2, so
    // invoke the resolved path directly. A manual `cmd /C` would mis-escape spaced npm paths and let
    // cmd interpret metacharacters in the forwarded args — worse, not better.
    let mut cmd = Command::new(bin);
    cmd.args(args);
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
        assert!(spec_for("Hermes").is_some());
        assert!(spec_for("pi").is_some());
        assert!(spec_for("nope").is_none());
    }

    #[test]
    fn every_launch_tool_has_a_connect_writer_so_launch_persists() {
        // launch = connect + run: EVERY launchable tool (env tools included) must be wireable by the
        // shared connect module, or `connect::apply` in run() would fail. This is the precondition
        // that makes the "always persist config" redefinition sound.
        for s in TOOLS {
            assert!(
                connect::spec(s.key).is_some(),
                "{} has no connect writer — launch can't persist its config",
                s.key
            );
        }
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
    fn config_tools_take_no_env() {
        // Hermes + Pi are wired by their config file, not env — `launch` writes that via `connect`.
        for key in ["hermes", "pi"] {
            let s = spec_for(key).unwrap();
            assert!(matches!(s.wiring, Wiring::Config), "{key} is a config tool");
            assert!(tool_env(s, "http://h", "m", "k").is_empty(), "{key} sets no env");
        }
    }

    #[test]
    fn pi_selects_the_openhydra_provider_on_the_cli_with_the_model_substituted() {
        let s = spec_for("pi").unwrap();
        let argv = build_argv(s, "openhydra/auto", &["-p".into(), "hi".into()]);
        // default_args (with {model} → openhydra/auto) come first, then the user's forwarded args.
        assert_eq!(
            argv,
            vec!["--provider", "openhydra", "--model", "openhydra/auto", "-p", "hi"]
        );
    }

    #[test]
    fn a_user_provided_model_flag_suppresses_the_injected_one() {
        let s = spec_for("pi").unwrap();
        // User forwards their own --model: keep --provider, drop our injected --model {model}.
        let argv = build_argv(s, "openhydra/auto", &["--model".into(), "qwen3-coder".into(), "-p".into(), "hi".into()]);
        assert_eq!(argv, vec!["--provider", "openhydra", "--model", "qwen3-coder", "-p", "hi"]);
        assert_eq!(argv.iter().filter(|a| *a == "--model").count(), 1, "exactly one --model");
    }

    #[test]
    fn hermes_needs_no_extra_args() {
        let s = spec_for("hermes").unwrap();
        assert_eq!(build_argv(s, "openhydra/auto", &["-z".into()]), vec!["-z"]);
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
