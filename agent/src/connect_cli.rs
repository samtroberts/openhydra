//! `openhydra connect <tool>` — the CLI twin of the desktop "Connect" button. Writes the OpenHydra
//! block into a coding tool's own config file, using the SAME shared writers as the desktop app
//! ([`openhydra_agent::connect`]). No swarm; just edits a config file and exits.

use clap::Args;

use openhydra_agent::connect::{self, TOOLS};

#[derive(Args, Debug)]
pub struct ConnectArgs {
    /// The coding tool to wire (claude, opencode, continue, hermes, pi). Omit only with `--list`.
    pub tool: Option<String>,

    /// List the tools `connect` can wire, then exit.
    #[arg(long)]
    pub list: bool,

    /// Gateway origin the tool should call (Claude Code appends `/v1/messages`; the OpenAI-compatible
    /// tools get `/v1`).
    #[arg(long, default_value = "http://127.0.0.1:16527")]
    pub gateway: String,

    /// Show the target path + the exact file that would be written, WITHOUT touching anything.
    #[arg(long)]
    pub dry_run: bool,

    /// Declare a specific network model in the tool's own model picker (repeatable). Without this the
    /// config declares only `openhydra/auto`; add one so you can pin it in the tool and see its name.
    /// List live models with `/models` in the tool. E.g. `--model qwen3-coder:30b-a3b-q8_0`.
    /// (Only applies to picker tools: opencode, pi, continue.)
    #[arg(long = "model", value_name = "ID")]
    pub models: Vec<String>,
}

#[derive(Args, Debug)]
pub struct DisconnectArgs {
    /// The coding tool to un-wire (claude, opencode, continue, hermes, pi).
    pub tool: String,
}

/// `openhydra disconnect <tool>` — restore the pristine pre-OpenHydra config (or delete a file we
/// created). The CLI twin of the desktop "Disconnect" button; uses the same [`connect::disconnect`].
pub fn run_disconnect(args: DisconnectArgs) -> Result<(), String> {
    let rep = connect::disconnect(&args.tool)?;
    let what = match rep.action.as_str() {
        "restored" => "original config restored",
        "stripped" => "removed our block (kept your other config)",
        "removed" => "removed the config we created",
        "not-connected" => "was not connected — nothing to undo",
        other => other,
    };
    eprintln!("openhydra disconnect: {} → {} ({})", args.tool, rep.path, what);
    Ok(())
}

pub fn run(args: ConnectArgs) -> Result<(), String> {
    if args.list {
        println!("Connectable tools (openhydra connect <tool>):");
        for t in TOOLS {
            let path = connect::config_path(t.kind)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<no home dir>".into());
            println!("  {:10} {:12} {}", t.key, t.label, path);
        }
        return Ok(());
    }

    let tool = args.tool.as_deref().ok_or(
        "usage: openhydra connect <tool> [--gateway URL] [--dry-run]   (see `openhydra connect --list`)",
    )?;
    // A trailing slash would double up (`…//v1`); normalise the origin once.
    let origin = args.gateway.trim_end_matches('/');

    if args.dry_run {
        let pv = connect::preview_with_models(tool, origin, &args.models)?;
        println!("# {} {} → {}", pv.action, tool, pv.path);
        if let Some(w) = &pv.warning {
            eprintln!("# warning: {w}");
        }
        print!("{}", pv.preview);
        if !pv.preview.ends_with('\n') {
            println!();
        }
        return Ok(());
    }

    let rep = connect::apply_with_models(tool, origin, &args.models)?;
    let backup = rep
        .backup
        .as_ref()
        .map(|b| format!(" (original backed up → {b})"))
        .unwrap_or_default();
    eprintln!("openhydra connect: {} {} → {}{}", rep.action, tool, rep.path, backup);
    eprintln!("  {tool} now routes to OpenHydra at {origin}. Model: openhydra/auto.");
    if !args.models.is_empty() {
        eprintln!("  declared in {tool}'s model picker: {}", args.models.join(", "));
    }
    Ok(())
}
