// OpenHydra Desktop — Connectors: detect installed coding tools and one-click wire them to the
// local gateway.
//
// The config-writing logic (the per-tool merge writers, preview/apply, backup) lives in the shared
// `openhydra_agent::connect` module so the desktop "Connect" button and the `openhydra connect` CLI
// subcommand use ONE implementation. This file adds only what's desktop-specific: binary/config-dir
// DETECTION (which needs the GUI-PATH `resolve_program` fix) and the Tauri command surface.

use serde::Serialize;

use openhydra_agent::connect::{config_path, declared_models, is_connected, kind_str, Kind, ToolSpec, TOOLS};

use crate::installer::resolve_program;

// Re-export the shared preview/apply + result types so the Tauri commands keep their paths.
pub use openhydra_agent::connect::{
    apply_with_models, disconnect, preview, ConnectPreview, ConnectReport, DisconnectReport,
};

// ── Detection (desktop-only: needs the GUI-PATH resolver) ─────────────────────

#[derive(Serialize)]
pub struct ConnectorStatus {
    key: String,
    label: String,
    /// "opencode" | "claude" | "continue" | "hermes" | "pi" — the config kind, for the UI.
    kind: String,
    installed: bool,
    /// The resolved binary path, or the config dir that evidences an install.
    detail: Option<String>,
    /// Surfaces this tool runs on ("terminal"/"editor"/"app"); first is the default-selected. Drives
    /// the header switcher + which run action ("run" copies `launch`; "open" opens the GUI).
    surfaces: Vec<String>,
    /// Whether declaring specific network models in this tool's picker is meaningful (shows the
    /// model selector). Only opencode/pi/continue.
    declares_models: bool,
    /// Whether the tool has a GUI to open (App/Editor surface). Enables the "Connect & Open" action.
    has_gui: bool,
    /// The tool's natural CLI verb, shown in its snippet + primary button: "launch" or "connect".
    natural_verb: String,
    /// Whether the tool's config currently contains the OpenHydra block (drives the Connected /
    /// Disconnect state).
    connected: bool,
    /// Specific network models already declared in the tool's picker (excludes `auto`). Pre-populates
    /// the selector so a re-Connect doesn't drop them.
    declared_models: Vec<String>,
}

fn detect(spec: &ToolSpec) -> (bool, Option<String>) {
    // Binary tools: resolve on PATH / common install dirs (the GUI-PATH fix — a Finder-launched app
    // gets a minimal PATH, so we also check ~/.local/bin, homebrew, etc.).
    if !spec.bins.is_empty() {
        if let Some(p) = spec.bins.iter().find_map(|b| resolve_program(b)) {
            return (true, Some(p.display().to_string()));
        }
    }
    // GUI apps / editor extensions with no CLI on PATH are evidenced by their config DIR, which the
    // tool itself creates: the OpenCode desktop app (`ai.opencode.desktop`) → ~/.config/opencode,
    // the Continue VS Code/JetBrains extension → ~/.continue.
    if matches!(spec.kind, Kind::OpencodeJson | Kind::ContinueYaml) {
        if let Some(dir) = config_path(spec.kind).and_then(|c| c.parent().map(|p| p.to_path_buf())) {
            if dir.exists() {
                return (true, Some(dir.display().to_string()));
            }
        }
    }
    (false, None)
}

/// Detection status for every known connector (installed? where? wired?). Read-only.
pub fn statuses() -> Vec<ConnectorStatus> {
    TOOLS
        .iter()
        .map(|t| {
            let (installed, detail) = detect(t);
            ConnectorStatus {
                key: t.key.into(),
                label: t.label.into(),
                kind: kind_str(t.kind).into(),
                installed,
                detail,
                surfaces: t.surfaces.iter().map(|s| s.as_str().to_string()).collect(),
                declares_models: t.declares_models,
                has_gui: t.has_gui(),
                natural_verb: t.natural_verb().into(),
                connected: is_connected(t.key),
                declared_models: if t.declares_models { declared_models(t.key) } else { vec![] },
            }
        })
        .collect()
}
