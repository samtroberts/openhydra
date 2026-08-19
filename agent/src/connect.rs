//! Connectors: wire a local coding tool to the OpenHydra gateway by merging an OpenHydra block into
//! the tool's own config file. Shared by the `openhydra connect` CLI subcommand and the desktop
//! "Connect" button — one implementation, so a fix (e.g. Continue's required `name`/`version`)
//! lands in both.
//!
//! Design (grounded in live validation, 2026-08-14 / 2026-08-18):
//!   • OpenCode  → ~/.config/opencode/opencode.json — a custom `@ai-sdk/openai-compatible` provider.
//!   • Claude Code → ~/.claude/settings.json — an `env` block (ANTHROPIC_BASE_URL/KEY).
//!   • Continue  → ~/.continue/config.yaml — an entry in `models:` (+ required top-level name/version).
//!   • Hermes    → ~/.hermes/config.yaml — the active `model:` block (provider=custom, id field `name`).
//!   • Pi        → ~/.pi/agent/models.json — a `providers.openhydra` entry.
//!
//! Every writer is a PURE merge (unit-tested below): parse the existing file, insert/replace ONLY the
//! OpenHydra-owned key (idempotent, never clobbers unrelated config), re-serialize. [`apply`] backs the
//! file up first and reports exactly what changed.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{json, Value};

/// A dummy non-empty key: the loopback gateway is open, but OpenAI/Anthropic clients require a
/// non-empty key field.
const LOCAL_KEY: &str = "oh-local";
/// The meta-model every connector defaults to (the gateway resolves it to a live model). Used where
/// a FULLY-QUALIFIED reference is expected (Hermes' `name`, Continue's `model`, OpenCode's top-level
/// `provider/model`).
const AUTO_MODEL: &str = "openhydra/auto";
/// The bare meta-model id used INSIDE a provider block (OpenCode's `models` key, Pi's `id`). Tools
/// that split `--model provider/model` on `/` would parse `openhydra/auto` as provider=`openhydra`,
/// model=`auto` and then fail to find a declared model called `openhydra/auto` — silently falling
/// back to their default (OpenCode → kimi, Pi → "custom model id" warning). The gateway accepts a
/// bare `auto` (see `is_auto_model`), so the id declared under the `openhydra` provider must be `auto`.
const AUTO_ID: &str = "auto";

/// How a connector is wired (which config file + shape).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    OpencodeJson,
    ClaudeSettings,
    ContinueYaml,
    HermesYaml,
    PiModelsJson,
}

/// Where a tool can run, so the UI can offer a per-surface switcher (Terminal ↔ App/Editor) and pick
/// the right "run" action. A tool's surfaces share ONE config file — switching surface changes only
/// the snippet + run button, never what Connect writes. See `docs/CONNECT_AND_RUN_PLAN.md`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Surface {
    /// A CLI/TUI run in a terminal — "run" = copy `openhydra launch <tool>`.
    Terminal,
    /// A code-editor extension (VS Code / JetBrains) — "run" = open the editor.
    Editor,
    /// A standalone desktop GUI app — "run" = open the app.
    App,
}

impl Surface {
    /// Lowercase tag used on the wire / in the UI dataset.
    pub fn as_str(self) -> &'static str {
        match self {
            Surface::Terminal => "terminal",
            Surface::Editor => "editor",
            Surface::App => "app",
        }
    }
}

pub struct ToolSpec {
    pub key: &'static str,
    pub label: &'static str,
    /// Candidate binary names for PATH detection; empty ⇒ detect by config dir instead.
    pub bins: &'static [&'static str],
    pub kind: Kind,
    /// Surfaces this tool runs on; the first is the default-selected one in the UI switcher.
    pub surfaces: &'static [Surface],
    /// Whether declaring specific network models in this tool's own picker is meaningful (drives the
    /// GUI model selector). Only opencode/pi/continue — claude/hermes are single-endpoint.
    pub declares_models: bool,
    /// macOS `.app` name for `open -a` when the active surface is App/Editor. `None` ⇒ no GUI to open
    /// (Terminal-only tools). PROVISIONAL — confirm against the installed app in the live test.
    pub gui_target: Option<&'static str>,
}

pub const TOOLS: &[ToolSpec] = &[
    // OpenCode — `opencode` CLI/TUI *and* the `ai.opencode.desktop` app; both read ~/.config/opencode.
    ToolSpec { key: "opencode", label: "OpenCode", bins: &["opencode"], kind: Kind::OpencodeJson,
               surfaces: &[Surface::Terminal, Surface::App], declares_models: true, gui_target: Some("OpenCode") },
    // Claude Code — `claude` CLI *and* the VS Code/JetBrains extension; both read ~/.claude/settings.json.
    ToolSpec { key: "claude", label: "Claude Code", bins: &["claude"], kind: Kind::ClaudeSettings,
               surfaces: &[Surface::Terminal, Surface::Editor], declares_models: false, gui_target: Some("Visual Studio Code") },
    // Continue is a VS Code / JetBrains extension (no CLI) — detected by its config dir.
    ToolSpec { key: "continue", label: "Continue", bins: &[], kind: Kind::ContinueYaml,
               surfaces: &[Surface::Editor], declares_models: true, gui_target: Some("Visual Studio Code") },
    // Hermes Agent (Nous Research) — YAML config, OpenAI-compatible custom provider; CLI *and* app.
    ToolSpec { key: "hermes", label: "Hermes", bins: &["hermes", "hermes-agent"], kind: Kind::HermesYaml,
               surfaces: &[Surface::Terminal, Surface::App], declares_models: false, gui_target: Some("Hermes") },
    // Pi (earendil-works) — static models.json, OpenAI-compatible provider. `pi` risks a PATH
    // collision, but the config dir (~/.pi/agent) is the real evidence; bins gives a fast path.
    ToolSpec { key: "pi", label: "Pi", bins: &["pi"], kind: Kind::PiModelsJson,
               surfaces: &[Surface::Terminal], declares_models: true, gui_target: None },
];

pub fn spec(key: &str) -> Option<&'static ToolSpec> {
    TOOLS.iter().find(|t| t.key == key)
}

impl ToolSpec {
    /// True if the tool has a Terminal surface (its "run" can copy `openhydra launch <key>`).
    pub fn has_terminal(&self) -> bool {
        self.surfaces.contains(&Surface::Terminal)
    }
    /// True if the tool has a GUI surface to open (App or Editor) and a target to open it with.
    pub fn has_gui(&self) -> bool {
        self.gui_target.is_some()
    }
    /// The verb natural to this tool, shown on its card + in its Terminal snippet: `launch` for
    /// tools we can run (terminal CLIs), `connect` for run-less tools (editor extensions).
    pub fn natural_verb(&self) -> &'static str {
        if self.has_terminal() {
            "launch"
        } else {
            "connect"
        }
    }
}

/// `$HOME` (unix) / `%USERPROFILE%` (windows).
fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// The config file a tool's OpenHydra block lives in.
pub fn config_path(kind: Kind) -> Option<PathBuf> {
    let h = home()?;
    Some(match kind {
        // OpenCode honours $XDG_CONFIG_HOME; fall back to ~/.config.
        Kind::OpencodeJson => xdg_config().unwrap_or_else(|| h.join(".config")).join("opencode/opencode.json"),
        Kind::ClaudeSettings => h.join(".claude/settings.json"),
        Kind::ContinueYaml => h.join(".continue/config.yaml"),
        Kind::HermesYaml => h.join(".hermes/config.yaml"),
        Kind::PiModelsJson => h.join(".pi/agent/models.json"),
    })
}

fn xdg_config() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).filter(|p| !p.as_os_str().is_empty())
}

pub fn kind_str(k: Kind) -> &'static str {
    match k {
        Kind::OpencodeJson => "opencode",
        Kind::ClaudeSettings => "claude",
        Kind::ContinueYaml => "continue",
        Kind::HermesYaml => "hermes",
        Kind::PiModelsJson => "pi",
    }
}

// ── Pure config merges (unit-tested) ─────────────────────────────────────────

/// The base URL a tool should call. OpenCode/Continue/Hermes/Pi speak OpenAI (`/v1`); Claude Code
/// speaks the Anthropic Messages API and appends `/v1/messages` itself, so it takes the bare origin.
fn base_for(kind: Kind, origin: &str) -> String {
    match kind {
        Kind::OpencodeJson | Kind::ContinueYaml | Kind::HermesYaml | Kind::PiModelsJson => format!("{origin}/v1"),
        Kind::ClaudeSettings => origin.to_string(),
    }
}

/// Merge the `openhydra` provider into OpenCode's JSON config. Idempotent; only touches
/// `provider.openhydra`, `$schema`, and (when absent) the default `model`.
pub fn merge_opencode_json(existing: &str, api_base: &str, extra_models: &[String]) -> Result<(String, &'static str), String> {
    let mut root: Value = parse_json_or_empty(existing, "opencode.json")?;
    let obj = root.as_object_mut().ok_or("opencode.json is not a JSON object")?;
    obj.entry("$schema").or_insert_with(|| json!("https://opencode.ai/config.json"));
    let had = obj.get("provider").and_then(|p| p.get("openhydra")).is_some();
    // Bare `auto` id (NOT `openhydra/auto`) — OpenCode splits `provider/model` on `/`. Plus each
    // user-chosen model, so it shows in OpenCode's model picker as `openhydra/<id>`.
    let mut models_map = serde_json::Map::new();
    models_map.insert(AUTO_ID.into(), json!({ "name": "OpenHydra Auto" }));
    for m in extra_models {
        models_map.insert(m.clone(), json!({ "name": m }));
    }
    let provider = obj.entry("provider").or_insert_with(|| json!({}));
    let provider = provider.as_object_mut().ok_or("`provider` in opencode.json is not an object")?;
    provider.insert(
        "openhydra".into(),
        json!({
            "npm": "@ai-sdk/openai-compatible",
            "name": "OpenHydra",
            "options": { "baseURL": api_base, "apiKey": LOCAL_KEY },
            "models": Value::Object(models_map),
        }),
    );
    // Activate OpenHydra as the default model only if the user hasn't chosen one — never override a
    // REAL selection. But DO migrate our own earlier buggy default (`openhydra/openhydra/auto`, which
    // OpenCode mis-parses to model `openhydra/auto` and then can't find → falls back to its default)
    // to the correct `openhydra/auto` (= provider `openhydra`, model `auto`, the bare id declared above).
    let our_ref = format!("openhydra/{AUTO_ID}");
    let stale_default = format!("openhydra/openhydra/{AUTO_ID}");
    let cur = obj.get("model").and_then(|v| v.as_str());
    if cur.is_none() || cur == Some(stale_default.as_str()) {
        obj.insert("model".into(), json!(our_ref));
    }
    Ok((to_json_pretty(&root)?, if had { "updated" } else { "added" }))
}

/// Merge the OpenHydra endpoint into Claude Code's `settings.json` `env` block. Idempotent; only
/// touches `env.ANTHROPIC_BASE_URL` / `env.ANTHROPIC_API_KEY`.
pub fn merge_claude_settings(existing: &str, origin: &str) -> Result<(String, &'static str), String> {
    let mut root: Value = parse_json_or_empty(existing, "settings.json")?;
    let obj = root.as_object_mut().ok_or("settings.json is not a JSON object")?;
    let had = obj.get("env").and_then(|e| e.get("ANTHROPIC_BASE_URL")).is_some();
    let env = obj.entry("env").or_insert_with(|| json!({}));
    let env = env.as_object_mut().ok_or("`env` in settings.json is not an object")?;
    env.insert("ANTHROPIC_BASE_URL".into(), json!(origin));
    env.insert("ANTHROPIC_API_KEY".into(), json!(LOCAL_KEY));
    Ok((to_json_pretty(&root)?, if had { "updated" } else { "added" }))
}

/// Merge the OpenHydra entry into Continue's YAML `models:` list, keyed by `name: OpenHydra`
/// (idempotent — a re-run replaces our entry, never duplicates it, and leaves other models intact).
/// NOTE: serde_yaml round-trips structure but drops comments/formatting — the caller backs the file
/// up first and the UI warns.
pub fn merge_continue_yaml(existing: &str, api_base: &str, extra_models: &[String]) -> Result<(String, &'static str), String> {
    use serde_yaml::Value as Y;
    let mut root: Y = if existing.trim().is_empty() {
        Y::Mapping(serde_yaml::Mapping::new())
    } else {
        serde_yaml::from_str(existing).map_err(|e| format!("parse config.yaml: {e}"))?
    };
    let map = root.as_mapping_mut().ok_or("config.yaml is not a mapping")?;
    // Continue's config.yaml schema REQUIRES top-level `name` + `version`; the CLI/extension rejects
    // a file without them ("name: Required, version: Required"). On a fresh create we must supply
    // them — but never clobber a user's existing assistant `name`/`version`.
    let name_key = Y::String("name".into());
    if !map.contains_key(&name_key) {
        map.insert(name_key, Y::String("OpenHydra".into()));
    }
    let version_key = Y::String("version".into());
    if !map.contains_key(&version_key) {
        map.insert(version_key, Y::String("0.0.1".into()));
    }
    let models_key = Y::String("models".into());
    if !map.contains_key(&models_key) {
        map.insert(models_key.clone(), Y::Sequence(Vec::new()));
    }
    let seq = map
        .get_mut(&models_key)
        .and_then(|v| v.as_sequence_mut())
        .ok_or("`models` in config.yaml is not a list")?;
    let before = seq.len();
    // Remove all OUR prior entries (name starts with "OpenHydra"), keeping the user's other models,
    // then re-add: the auto entry + one per user-chosen model (each with a distinct `name:` Continue
    // shows in its picker).
    // Remove ONLY our own entries so a re-run replaces (not duplicates) them: the auto entry (old
    // exact `OpenHydra`, new `OpenHydra Auto`) and per-model entries (`OpenHydra: <id>`). Crucially do
    // NOT match on the prefix alone — a user model merely named e.g. `OpenHydra Local` must survive.
    seq.retain(|item| !is_our_continue_entry(item));
    let had = seq.len() != before;
    seq.push(continue_model_entry("OpenHydra Auto", AUTO_MODEL, api_base));
    for m in extra_models {
        seq.push(continue_model_entry(&format!("OpenHydra: {m}"), m, api_base));
    }
    let out = serde_yaml::to_string(&root).map_err(|e| e.to_string())?;
    Ok((out, if had { "updated" } else { "added" }))
}

fn continue_model_entry(name: &str, model: &str, api_base: &str) -> serde_yaml::Value {
    use serde_yaml::Value as Y;
    let s = |x: &str| Y::String(x.to_string());
    let mut m = serde_yaml::Mapping::new();
    m.insert(s("name"), s(name));
    m.insert(s("provider"), s("openai"));
    m.insert(s("model"), s(model));
    m.insert(s("apiBase"), s(api_base));
    m.insert(s("apiKey"), s(LOCAL_KEY));
    m.insert(s("roles"), Y::Sequence(vec![s("chat"), s("edit"), s("apply")]));
    Y::Mapping(m)
}

/// Merge the OpenHydra endpoint into Hermes' `~/.hermes/config.yaml` active `model:` block
/// (provider=custom + base_url/api_key/name). Sets OpenHydra as Hermes' active model. Idempotent;
/// preserves any other `model:` sub-keys and all other top-level keys. Hermes appends
/// `/chat/completions` to `base_url`, so `api_base` ends in `/v1`. Schema verified against a live
/// `~/.hermes/config.yaml` (the model id field is `name`, not `model`). NOTE: serde_yaml drops
/// comments — the caller backs the file up first and the UI warns.
pub fn merge_hermes_yaml(existing: &str, api_base: &str) -> Result<(String, &'static str), String> {
    use serde_yaml::Value as Y;
    let mut root: Y = if existing.trim().is_empty() {
        Y::Mapping(serde_yaml::Mapping::new())
    } else {
        serde_yaml::from_str(existing).map_err(|e| format!("parse config.yaml: {e}"))?
    };
    let map = root.as_mapping_mut().ok_or("config.yaml is not a mapping")?;
    let s = |x: &str| Y::String(x.to_string());
    let model_key = s("model");
    if !map.contains_key(&model_key) {
        map.insert(model_key.clone(), Y::Mapping(serde_yaml::Mapping::new()));
    }
    let model = map
        .get_mut(&model_key)
        .and_then(|v| v.as_mapping_mut())
        .ok_or("`model` in config.yaml is not a mapping")?;
    let had = model.contains_key(&s("base_url"));
    model.insert(s("provider"), s("custom"));
    model.insert(s("base_url"), s(api_base));
    model.insert(s("api_key"), s(LOCAL_KEY));
    model.insert(s("name"), s(AUTO_MODEL));
    let out = serde_yaml::to_string(&root).map_err(|e| e.to_string())?;
    Ok((out, if had { "updated" } else { "added" }))
}

/// Merge the OpenHydra provider into Pi's `~/.pi/agent/models.json`. Idempotent; only touches
/// `providers.openhydra`. Pi hot-reloads this file on `/model` (no restart). Model-entry shape
/// (`reasoning`/`input`/`cost`/`contextWindow`/`maxTokens`) verified against a live `models.json`.
pub fn merge_pi_models_json(existing: &str, api_base: &str, extra_models: &[String]) -> Result<(String, &'static str), String> {
    let mut root: Value = parse_json_or_empty(existing, "models.json")?;
    let obj = root.as_object_mut().ok_or("models.json is not a JSON object")?;
    let had = obj.get("providers").and_then(|p| p.get("openhydra")).is_some();
    let providers = obj.entry("providers").or_insert_with(|| json!({}));
    let providers = providers.as_object_mut().ok_or("`providers` in models.json is not an object")?;
    // Always declare bare `auto` (Pi splits `--model provider/model` on `/`, so `openhydra/auto`
    // would resolve to model `auto`; the gateway accepts `auto`). Then declare each user-chosen model
    // so it appears in Pi's own `/model` picker and the footer shows its name.
    let mut models = vec![pi_model_entry(AUTO_ID, "OpenHydra Auto")];
    for m in extra_models {
        models.push(pi_model_entry(m, m));
    }
    providers.insert(
        "openhydra".into(),
        json!({ "baseUrl": api_base, "apiKey": LOCAL_KEY, "api": "openai-completions", "models": models }),
    );
    Ok((to_json_pretty(&root)?, if had { "updated" } else { "added" }))
}

/// One Pi `models.json` entry in the rich shape Pi expects. `name` is what Pi's picker/footer shows.
fn pi_model_entry(id: &str, name: &str) -> Value {
    json!({
        "id": id,
        "name": name,
        "reasoning": false,
        "input": ["text"],
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
        "contextWindow": 128000,
        "maxTokens": 8192,
    })
}

fn parse_json_or_empty(existing: &str, what: &str) -> Result<Value, String> {
    if existing.trim().is_empty() {
        Ok(json!({}))
    } else {
        serde_json::from_str(existing).map_err(|e| format!("parse {what}: {e}"))
    }
}

fn to_json_pretty(v: &Value) -> Result<String, String> {
    serde_json::to_string_pretty(v).map_err(|e| e.to_string())
}

/// Merge dispatch: produce the new file content for a tool, given its existing content and the
/// gateway origin (`http://127.0.0.1:<port>`).
fn merge_for(kind: Kind, existing: &str, origin: &str, models: &[String]) -> Result<(String, &'static str), String> {
    let base = base_for(kind, origin);
    // De-dupe (order-preserving) and drop the auto meta-model — it's always declared, so a repeated
    // `--model X` or an explicit `--model auto` must not create duplicate entries.
    let mut seen = std::collections::HashSet::new();
    let models: Vec<String> = models
        .iter()
        .map(|m| m.trim())
        .filter(|m| !m.is_empty() && !m.eq_ignore_ascii_case(AUTO_ID) && !m.eq_ignore_ascii_case(AUTO_MODEL))
        .filter(|m| seen.insert(m.to_string()))
        .map(str::to_string)
        .collect();
    match kind {
        Kind::OpencodeJson => merge_opencode_json(existing, &base, &models),
        Kind::ClaudeSettings => merge_claude_settings(existing, &base),
        Kind::ContinueYaml => merge_continue_yaml(existing, &base, &models),
        // Hermes = one active `model:` block (no picker) and Claude = claude-* bridged — neither takes
        // a declared model list; `models` is only for the multi-model tools (opencode/pi/continue).
        Kind::HermesYaml => merge_hermes_yaml(existing, &base),
        Kind::PiModelsJson => merge_pi_models_json(existing, &base, &models),
    }
}

// ── Preview / apply ──────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ConnectPreview {
    pub key: String,
    pub kind: String,
    pub path: String,
    /// "create" (no file yet) | "update" (merge into existing).
    pub action: String,
    /// The new full file content that Apply would write (for a diff/preview).
    pub preview: String,
    /// A caveat to surface before writing (e.g. YAML comment loss), if any.
    pub warning: Option<String>,
}

#[derive(Serialize)]
pub struct ConnectReport {
    pub key: String,
    pub path: String,
    /// Where the prior file was backed up (None when the file was freshly created).
    pub backup: Option<String>,
    /// "added" | "updated" — whether an OpenHydra block already existed.
    pub action: String,
}

fn read_existing(path: &Path) -> Result<String, String> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(s),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(format!("read {}: {e}", path.display())),
    }
}

/// Compute (but don't write) what Connect would do for `key`, given the gateway `origin`.
pub fn preview(key: &str, origin: &str) -> Result<ConnectPreview, String> {
    preview_with_models(key, origin, &[])
}

/// As [`preview`], additionally declaring `models` (specific network model ids) in the tool's config
/// so they appear in its own model picker. Only affects the multi-model tools (opencode/pi/continue).
pub fn preview_with_models(key: &str, origin: &str, models: &[String]) -> Result<ConnectPreview, String> {
    let spec = spec(key).ok_or_else(|| format!("unknown connector '{key}'"))?;
    let path = config_path(spec.kind).ok_or("cannot resolve home directory")?;
    let existing = read_existing(&path)?;
    let create = existing.trim().is_empty();
    let (preview, _action) = merge_for(spec.kind, &existing, origin, models)?;
    // serde_yaml drops comments/spacing on re-serialize, so any YAML-backed tool loses them on an
    // in-place merge (Continue and Hermes). JSON tools (opencode/pi) and Claude are unaffected.
    let warning = (matches!(spec.kind, Kind::ContinueYaml | Kind::HermesYaml) && !create).then(|| {
        let f = if spec.kind == Kind::HermesYaml { "Hermes'" } else { "Continue's" };
        format!("{f} config.yaml will be reformatted (comments/spacing are not preserved). The original is backed up.")
    });
    Ok(ConnectPreview {
        key: key.into(),
        kind: kind_str(spec.kind).into(),
        path: path.display().to_string(),
        action: if create { "create".into() } else { "update".into() },
        preview,
        warning,
    })
}

/// The single, stable path holding the PRISTINE pre-OpenHydra file: `<file>.openhydra.bak`. One fixed
/// location (no numeric suffixes) so `apply` preserves the original untouched across re-connects and
/// `disconnect` can find + restore it.
fn backup_path(path: &Path) -> PathBuf {
    path.with_extension(format!(
        "{}.openhydra.bak",
        path.extension().and_then(|e| e.to_str()).unwrap_or("")
    ))
}

/// Write the OpenHydra block into `key`'s config, backing up any existing file first.
pub fn apply(key: &str, origin: &str) -> Result<ConnectReport, String> {
    apply_with_models(key, origin, &[])
}

/// As [`apply`], additionally declaring `models` (specific network model ids) in the tool's config so
/// they show in its own model picker (and its footer shows the name). Only affects opencode/pi/continue.
pub fn apply_with_models(key: &str, origin: &str, models: &[String]) -> Result<ConnectReport, String> {
    let spec = spec(key).ok_or_else(|| format!("unknown connector '{key}'"))?;
    let path = config_path(spec.kind).ok_or("cannot resolve home directory")?;
    let existing = read_existing(&path)?;
    let (new_content, action) = merge_for(spec.kind, &existing, origin, models)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    // Back up the PRISTINE pre-OpenHydra file exactly once — only if a non-empty file exists that does
    // NOT already contain our block (so the backup can never capture an already-wired state, e.g. from a
    // second apply on a file we created), AND we haven't backed it up before. `disconnect` restores it.
    let bak = backup_path(&path);
    let backup = if !existing.trim().is_empty() && !bak.exists() && !has_openhydra_block(spec.kind, &existing) {
        std::fs::write(&bak, &existing).map_err(|e| format!("backup {}: {e}", bak.display()))?;
        Some(bak.display().to_string())
    } else if bak.exists() {
        Some(bak.display().to_string())
    } else {
        None
    };
    std::fs::write(&path, new_content).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(ConnectReport { key: key.into(), path: path.display().to_string(), backup, action: action.into() })
}

/// Whether `content` already contains OpenHydra's block for `kind` — mirrors the exact markers each
/// merge uses for its "added vs updated" decision. Drives the desktop card's connected state and the
/// disconnect no-op guard.
fn has_openhydra_block(kind: Kind, content: &str) -> bool {
    if content.trim().is_empty() {
        return false;
    }
    match kind {
        Kind::OpencodeJson => serde_json::from_str::<Value>(content)
            .ok()
            .and_then(|v| v.get("provider").and_then(|p| p.get("openhydra")).cloned())
            .is_some(),
        Kind::PiModelsJson => serde_json::from_str::<Value>(content)
            .ok()
            .and_then(|v| v.get("providers").and_then(|p| p.get("openhydra")).cloned())
            .is_some(),
        // Claude shares no provider block — only an `env`. Key off OUR sentinel api key (`oh-local`),
        // NOT the mere presence of ANTHROPIC_BASE_URL: a user pointing Claude Code at their own proxy
        // has ANTHROPIC_BASE_URL set but is NOT connected to us — treating that as "connected" would let
        // Disconnect delete their settings.json.
        Kind::ClaudeSettings => serde_json::from_str::<Value>(content)
            .ok()
            .and_then(|v| v.get("env").and_then(|e| e.get("ANTHROPIC_API_KEY")).and_then(|k| k.as_str().map(str::to_string)))
            .map(|k| k == LOCAL_KEY)
            .unwrap_or(false),
        Kind::HermesYaml => serde_yaml::from_str::<serde_yaml::Value>(content)
            .ok()
            .and_then(|v| v.get("model").and_then(|m| m.get("name")).and_then(|n| n.as_str().map(str::to_string)))
            .map(|name| name == AUTO_MODEL)
            .unwrap_or(false),
        // Match ONLY our own entries — the SAME precise predicate `merge_continue_yaml` uses to retain
        // (never a bare `starts_with("OpenHydra")`, or a user's model named "OpenHydra Local" reads as us).
        Kind::ContinueYaml => serde_yaml::from_str::<serde_yaml::Value>(content)
            .ok()
            .and_then(|v| v.get("models").and_then(|m| m.as_sequence().cloned()))
            .map(|seq| seq.iter().any(is_our_continue_entry))
            .unwrap_or(false),
    }
}

/// Whether a Continue `models:` entry is one WE wrote (auto or a per-model `OpenHydra: <id>`).
fn is_our_continue_entry(item: &serde_yaml::Value) -> bool {
    item.get("name")
        .and_then(|n| n.as_str())
        .map(|n| n == "OpenHydra" || n == "OpenHydra Auto" || n.starts_with("OpenHydra: "))
        .unwrap_or(false)
}

/// Whether `key`'s config currently contains the OpenHydra block (i.e. the tool is wired to us).
pub fn is_connected(key: &str) -> bool {
    let Some(spec) = spec(key) else { return false };
    let Some(path) = config_path(spec.kind) else { return false };
    read_existing(&path).map(|c| has_openhydra_block(spec.kind, &c)).unwrap_or(false)
}

/// The specific network models already declared in `key`'s config picker (excludes the always-on
/// `auto`). Lets the desktop selector pre-populate, so a re-Connect doesn't silently drop them.
pub fn declared_models(key: &str) -> Vec<String> {
    let Some(spec) = spec(key) else { return vec![] };
    let Some(path) = config_path(spec.kind) else { return vec![] };
    let Ok(content) = read_existing(&path) else { return vec![] };
    if content.trim().is_empty() {
        return vec![];
    }
    match spec.kind {
        Kind::OpencodeJson => serde_json::from_str::<Value>(&content)
            .ok()
            .and_then(|v| v.get("provider").and_then(|p| p.get("openhydra")).and_then(|o| o.get("models")).and_then(|m| m.as_object().cloned()))
            .map(|m| m.keys().filter(|k| *k != AUTO_ID).cloned().collect())
            .unwrap_or_default(),
        Kind::PiModelsJson => serde_json::from_str::<Value>(&content)
            .ok()
            .and_then(|v| v.get("providers").and_then(|p| p.get("openhydra")).and_then(|o| o.get("models")).and_then(|m| m.as_array().cloned()))
            .map(|arr| arr.iter().filter_map(|e| e.get("id").and_then(|i| i.as_str())).filter(|id| *id != AUTO_ID).map(String::from).collect())
            .unwrap_or_default(),
        Kind::ContinueYaml => serde_yaml::from_str::<serde_yaml::Value>(&content)
            .ok()
            .and_then(|v| v.get("models").and_then(|m| m.as_sequence().cloned()))
            .map(|seq| seq.iter().filter_map(|e| e.get("name").and_then(|n| n.as_str()).and_then(|n| n.strip_prefix("OpenHydra: ")).map(String::from)).collect())
            .unwrap_or_default(),
        Kind::ClaudeSettings | Kind::HermesYaml => vec![],
    }
}

#[derive(Serialize)]
pub struct DisconnectReport {
    pub key: String,
    pub path: String,
    /// "restored" (pristine backup put back) | "stripped" (our block removed, user's own content kept)
    /// | "removed" (a file that was only ours is deleted) | "not-connected" (nothing to undo).
    pub action: String,
}

/// Remove ONLY OpenHydra's block from `content` for `kind`, leaving the user's own config intact.
/// The inverse of the merge writers; used by [`disconnect`] when there's no pristine backup to restore.
fn strip_openhydra_block(kind: Kind, content: &str) -> Result<String, String> {
    use serde_yaml::Value as Y;
    match kind {
        Kind::OpencodeJson => {
            let mut root: Value = parse_json_or_empty(content, "opencode.json")?;
            if let Some(obj) = root.as_object_mut() {
                if let Some(prov) = obj.get_mut("provider").and_then(|p| p.as_object_mut()) {
                    prov.remove("openhydra");
                    if prov.is_empty() {
                        obj.remove("provider");
                    }
                }
                if obj.get("model").and_then(|m| m.as_str()) == Some(&format!("openhydra/{AUTO_ID}")) {
                    obj.remove("model");
                }
            }
            to_json_pretty(&root)
        }
        Kind::PiModelsJson => {
            let mut root: Value = parse_json_or_empty(content, "models.json")?;
            if let Some(obj) = root.as_object_mut() {
                if let Some(prov) = obj.get_mut("providers").and_then(|p| p.as_object_mut()) {
                    prov.remove("openhydra");
                    if prov.is_empty() {
                        obj.remove("providers");
                    }
                }
            }
            to_json_pretty(&root)
        }
        Kind::ClaudeSettings => {
            let mut root: Value = parse_json_or_empty(content, "settings.json")?;
            if let Some(obj) = root.as_object_mut() {
                // Only strip if it's OUR wiring (sentinel api key) — never touch a user's own env.
                let ours = obj
                    .get("env")
                    .and_then(|e| e.get("ANTHROPIC_API_KEY"))
                    .and_then(|k| k.as_str())
                    == Some(LOCAL_KEY);
                if ours {
                    if let Some(env) = obj.get_mut("env").and_then(|e| e.as_object_mut()) {
                        env.remove("ANTHROPIC_BASE_URL");
                        env.remove("ANTHROPIC_API_KEY");
                        if env.is_empty() {
                            obj.remove("env");
                        }
                    }
                }
            }
            to_json_pretty(&root)
        }
        Kind::HermesYaml => {
            let mut root: Y = if content.trim().is_empty() {
                Y::Mapping(serde_yaml::Mapping::new())
            } else {
                serde_yaml::from_str(content).map_err(|e| format!("parse config.yaml: {e}"))?
            };
            if let Some(map) = root.as_mapping_mut() {
                let mk = Y::String("model".into());
                let ours = map
                    .get(&mk)
                    .and_then(|m| m.get("name"))
                    .and_then(|n| n.as_str())
                    == Some(AUTO_MODEL);
                if ours {
                    if let Some(model) = map.get_mut(&mk).and_then(|v| v.as_mapping_mut()) {
                        for k in ["provider", "base_url", "api_key", "name"] {
                            model.remove(&Y::String(k.into()));
                        }
                        if model.is_empty() {
                            map.remove(&mk);
                        }
                    }
                }
            }
            serde_yaml::to_string(&root).map_err(|e| e.to_string())
        }
        Kind::ContinueYaml => {
            let mut root: Y = if content.trim().is_empty() {
                Y::Mapping(serde_yaml::Mapping::new())
            } else {
                serde_yaml::from_str(content).map_err(|e| format!("parse config.yaml: {e}"))?
            };
            if let Some(map) = root.as_mapping_mut() {
                if let Some(seq) = map.get_mut(&Y::String("models".into())).and_then(|v| v.as_sequence_mut()) {
                    seq.retain(|item| !is_our_continue_entry(item));
                }
            }
            serde_yaml::to_string(&root).map_err(|e| e.to_string())
        }
    }
}

/// After stripping, is the file just OpenHydra's own scaffolding (so it can be removed rather than
/// left as an empty shell)? Errs on the safe side — an unparseable/ambiguous remainder is NOT "ours".
fn is_ours_only(kind: Kind, stripped: &str) -> bool {
    if stripped.trim().is_empty() {
        return true;
    }
    match kind {
        // We add `$schema` on a fresh create; anything else is the user's.
        Kind::OpencodeJson => serde_json::from_str::<Value>(stripped)
            .ok()
            .and_then(|v| v.as_object().cloned())
            .map(|o| o.keys().all(|k| k == "$schema"))
            .unwrap_or(false),
        Kind::PiModelsJson | Kind::ClaudeSettings => serde_json::from_str::<Value>(stripped)
            .ok()
            .and_then(|v| v.as_object().cloned())
            .map(|o| o.is_empty())
            .unwrap_or(false),
        Kind::HermesYaml => serde_yaml::from_str::<serde_yaml::Value>(stripped)
            .ok()
            .and_then(|v| v.as_mapping().cloned())
            .map(|m| m.is_empty())
            .unwrap_or(false),
        // We add `name: OpenHydra` + `version: 0.0.1` + an empty `models:` on a fresh create.
        Kind::ContinueYaml => serde_yaml::from_str::<serde_yaml::Value>(stripped)
            .ok()
            .and_then(|v| v.as_mapping().cloned())
            .map(|m| {
                m.iter().all(|(k, val)| {
                    let ks = k.as_str().unwrap_or("");
                    (ks == "name" && val.as_str() == Some("OpenHydra"))
                        || (ks == "version" && val.as_str() == Some("0.0.1"))
                        || (ks == "models" && val.as_sequence().map(|s| s.is_empty()).unwrap_or(false))
                })
            })
            .unwrap_or(false),
    }
}

/// Undo [`apply`] for `key`. Prefers restoring the pristine pre-OpenHydra backup (exact original
/// bytes/formatting); otherwise strips ONLY our block, preserving any of the user's own content, and
/// deletes the file only when nothing but our scaffolding is left. Never deletes a file with user data.
pub fn disconnect(key: &str) -> Result<DisconnectReport, String> {
    let spec = spec(key).ok_or_else(|| format!("unknown connector '{key}'"))?;
    let path = config_path(spec.kind).ok_or("cannot resolve home directory")?;
    let report = |action: &str| DisconnectReport {
        key: key.into(),
        path: path.display().to_string(),
        action: action.into(),
    };

    let existing = read_existing(&path)?;
    if !has_openhydra_block(spec.kind, &existing) {
        return Ok(report("not-connected"));
    }

    // Prefer the pristine backup (exact bytes). Only trust it if it doesn't itself contain our block.
    let bak = backup_path(&path);
    if bak.exists() {
        if let Ok(original) = std::fs::read_to_string(&bak) {
            if !has_openhydra_block(spec.kind, &original) {
                std::fs::write(&path, &original).map_err(|e| format!("restore {}: {e}", path.display()))?;
                let _ = std::fs::remove_file(&bak);
                return Ok(report("restored"));
            }
        }
        let _ = std::fs::remove_file(&bak); // non-pristine/unreadable — discard and strip instead
    }

    // No pristine backup: surgically remove only our block, keeping any user content.
    let stripped = strip_openhydra_block(spec.kind, &existing)?;
    if is_ours_only(spec.kind, &stripped) {
        std::fs::remove_file(&path).map_err(|e| format!("remove {}: {e}", path.display()))?;
        Ok(report("removed"))
    } else {
        std::fs::write(&path, &stripped).map_err(|e| format!("write {}: {e}", path.display()))?;
        Ok(report("stripped"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Surface classification the UI switcher depends on: OpenCode/Claude/Hermes are dual-surface,
    /// Pi/Continue single; model selector only on opencode/pi/continue; GUI-openable everywhere but Pi.
    #[test]
    fn surface_classification_matches_the_plan() {
        let by = |k| spec(k).unwrap();
        // dual-surface (2 surfaces, Terminal first)
        for k in ["opencode", "claude", "hermes"] {
            assert_eq!(by(k).surfaces.len(), 2, "{k} is dual-surface");
            assert_eq!(by(k).surfaces[0], Surface::Terminal, "{k} defaults to Terminal");
            assert!(by(k).has_terminal() && by(k).has_gui(), "{k} runs in terminal AND has a GUI");
        }
        // single-surface
        assert_eq!(by("pi").surfaces, &[Surface::Terminal]);
        assert!(by("pi").has_terminal() && !by("pi").has_gui(), "pi is terminal-only, no GUI");
        assert_eq!(by("continue").surfaces, &[Surface::Editor]);
        assert!(!by("continue").has_terminal() && by("continue").has_gui(), "continue is editor-only");
        // model selector eligibility
        for k in ["opencode", "pi", "continue"] {
            assert!(by(k).declares_models, "{k} can declare specific models");
        }
        for k in ["claude", "hermes"] {
            assert!(!by(k).declares_models, "{k} is single-endpoint — no model selector");
        }
        // Natural verb: launch for terminal tools, connect for the editor-only one.
        for k in ["opencode", "claude", "hermes", "pi"] {
            assert_eq!(by(k).natural_verb(), "launch", "{k} is a terminal tool");
        }
        assert_eq!(by("continue").natural_verb(), "connect", "continue has no exec → connect");
    }

    /// Serializes every test that mutates the process-global `$HOME` (they'd otherwise race under
    /// parallel execution). Poison-tolerant: a panic in one test must not wedge the rest.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Sandbox `$HOME`, isolated per test-run, so the real-fs connect/disconnect tests never touch
    /// actual dotfiles. Returns the sandbox dir + the held env lock (keep both alive for the test).
    fn sandbox_home(tag: &str) -> (std::path::PathBuf, std::sync::MutexGuard<'static, ()>) {
        let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("oh-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("HOME", &dir);
        std::env::set_var("XDG_CONFIG_HOME", dir.join(".config"));
        (dir, guard)
    }

    #[test]
    fn disconnect_restores_a_pristine_original_even_after_multiple_applies() {
        let (_sb, _lock) = sandbox_home("disc-restore");
        let origin = "http://127.0.0.1:16527";
        // A user's pre-existing opencode.json with their own settings.
        let path = config_path(Kind::OpencodeJson).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let original = "{\n  \"theme\": \"dark\",\n  \"model\": \"gpt-4o\"\n}\n";
        std::fs::write(&path, original).unwrap();

        assert!(!is_connected("opencode"), "not connected before apply");
        apply("opencode", origin).unwrap();
        // A second apply (e.g. re-connect with models) must NOT clobber the pristine backup.
        apply_with_models("opencode", origin, &["qwen3-coder:30b".to_string()]).unwrap();
        assert!(is_connected("opencode"), "connected after apply");

        let rep = disconnect("opencode").unwrap();
        assert_eq!(rep.action, "restored");
        assert!(!is_connected("opencode"), "disconnected");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original, "byte-identical original restored");
        assert!(!backup_path(&path).exists(), "backup cleaned up");
    }

    #[test]
    fn disconnect_removes_a_file_we_created_from_nothing() {
        let (_sb, _lock) = sandbox_home("disc-remove");
        let origin = "http://127.0.0.1:16527";
        let path = config_path(Kind::PiModelsJson).unwrap();
        assert!(!path.exists(), "no pi config to start");

        apply("pi", origin).unwrap();
        assert!(path.exists() && is_connected("pi"));

        let rep = disconnect("pi").unwrap();
        assert_eq!(rep.action, "removed");
        assert!(!path.exists(), "the file we created is gone");
        assert!(!is_connected("pi"));
    }

    #[test]
    fn disconnect_on_an_unconnected_tool_is_a_no_op() {
        let (_sb, _lock) = sandbox_home("disc-noop");
        let path = config_path(Kind::HermesYaml).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "model:\n  provider: someone-else\n").unwrap();
        let rep = disconnect("hermes").unwrap();
        assert_eq!(rep.action, "not-connected");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "model:\n  provider: someone-else\n", "untouched");
    }

    #[test]
    fn is_connected_tracks_apply_across_all_tools() {
        let (_sb, _lock) = sandbox_home("is-conn");
        let origin = "http://127.0.0.1:16527";
        for k in ["opencode", "claude", "continue", "hermes", "pi"] {
            assert!(!is_connected(k), "{k} starts unconnected");
            apply(k, origin).unwrap();
            assert!(is_connected(k), "{k} connected after apply");
            disconnect(k).unwrap();
            assert!(!is_connected(k), "{k} unconnected after disconnect");
        }
    }

    /// Finding 1 (data loss): a user pointing Claude Code at their OWN proxy (ANTHROPIC_BASE_URL set,
    /// but NOT our sentinel key) must NOT read as connected — and Disconnect must be a no-op that leaves
    /// their settings.json fully intact, never deleted.
    #[test]
    fn a_users_own_anthropic_base_url_is_not_us_and_disconnect_never_deletes_it() {
        let (_sb, _lock) = sandbox_home("claude-own-proxy");
        let path = config_path(Kind::ClaudeSettings).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let user = r#"{"env":{"ANTHROPIC_BASE_URL":"https://my-proxy.example","ANTHROPIC_API_KEY":"sk-user"},"theme":"dark"}"#;
        std::fs::write(&path, user).unwrap();
        assert!(!is_connected("claude"), "a foreign base URL is NOT an OpenHydra connection");
        let rep = disconnect("claude").unwrap();
        assert_eq!(rep.action, "not-connected");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), user, "the user's settings.json is untouched");
    }

    /// Finding 2: if we created the file and the user then added their OWN content, Disconnect strips
    /// only our block and KEEPS their content (never deletes the whole file).
    #[test]
    fn disconnect_strips_our_block_but_keeps_user_content_they_added() {
        let (_sb, _lock) = sandbox_home("strip-keep");
        let origin = "http://127.0.0.1:16527";
        let path = config_path(Kind::OpencodeJson).unwrap();
        apply("opencode", origin).unwrap(); // no file yet → we create it (no backup)
        // user augments the file we created with their own provider + a setting
        let mut v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        v.as_object_mut().unwrap().insert("theme".into(), json!("gruvbox"));
        v["provider"].as_object_mut().unwrap().insert("mine".into(), json!({ "npm": "x" }));
        std::fs::write(&path, serde_json::to_string_pretty(&v).unwrap()).unwrap();

        let rep = disconnect("opencode").unwrap();
        assert_eq!(rep.action, "stripped");
        assert!(path.exists(), "file kept — it has user content");
        let after: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(after["provider"].get("openhydra").is_none(), "our provider stripped");
        assert!(after["provider"].get("mine").is_some(), "user's provider kept");
        assert_eq!(after["theme"], "gruvbox", "user's other keys kept");
        assert!(!is_connected("opencode"));
    }

    /// Finding 3: a Continue model merely NAMED like ours ("OpenHydra Local") is the user's own, not a
    /// connection — detection must use the precise predicate, not a bare prefix.
    #[test]
    fn a_user_continue_model_named_like_ours_is_not_a_connection() {
        let (_sb, _lock) = sandbox_home("cont-nameclash");
        let path = config_path(Kind::ContinueYaml).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let user = "name: me\nversion: 1.0.0\nmodels:\n- name: OpenHydra Local\n  provider: openai\n  model: x\n";
        std::fs::write(&path, user).unwrap();
        assert!(!is_connected("continue"), "'OpenHydra Local' is a user model, not us");
        assert_eq!(disconnect("continue").unwrap().action, "not-connected");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), user, "untouched");
    }

    /// Real file-write validation (`openhydra connect <tool>` / the desktop button → [`apply`]):
    /// drives the ACTUAL filesystem write/backup path against a sandbox `HOME`, without touching real
    /// dotfiles.
    #[test]
    fn apply_writes_backs_up_and_is_idempotent_on_the_real_fs() {
        // Redirect every tool's config root into an isolated sandbox (home() reads $HOME /
        // $XDG_CONFIG_HOME); the helper holds ENV_LOCK so this never races other HOME-mutating tests.
        let (sandbox, _lock) = sandbox_home("connect-test");
        let origin = "http://127.0.0.1:16527";

        // 1) Fresh create for every wired tool: file written to the real fs, no backup, block present.
        for key in ["opencode", "claude", "continue", "pi", "hermes"] {
            let rep = apply(key, origin).unwrap();
            assert_eq!(rep.action, "added", "{key}: first apply adds the block");
            assert!(rep.backup.is_none(), "{key}: a fresh create has nothing to back up");
            let written = std::fs::read_to_string(&rep.path).unwrap();
            assert!(written.contains("16527"), "{key}: written config points at the gateway");
        }

        // 2) Seed a REAL pre-existing user file → apply → the original is backed up verbatim and the
        //    user's own keys survive the merge (the non-clobber guarantee, on the real fs).
        let claude_path = sandbox.join(".claude/settings.json");
        let user_json = r#"{"model":"opus","permissions":{"allow":["Bash"]}}"#;
        std::fs::write(&claude_path, user_json).unwrap();
        let rep = apply("claude", origin).unwrap();
        let bak1 = rep.backup.expect("a non-empty existing file must be backed up");
        assert_eq!(std::fs::read_to_string(&bak1).unwrap(), user_json, "backup holds the ORIGINAL bytes");
        let merged: Value = serde_json::from_str(&std::fs::read_to_string(&claude_path).unwrap()).unwrap();
        assert_eq!(merged["model"], "opus", "user's model preserved");
        assert_eq!(merged["permissions"]["allow"][0], "Bash", "user's permissions preserved");
        assert_eq!(merged["env"]["ANTHROPIC_BASE_URL"], origin, "OpenHydra env block added");

        // 3) Idempotency + PRISTINE backup: a second apply → "updated", reports the SAME stable backup
        //    path (not a new numbered one), and that backup still holds the ORIGINAL user bytes — so a
        //    re-connect can never overwrite the pristine original that `disconnect` restores.
        let rep2 = apply("claude", origin).unwrap();
        assert_eq!(rep2.action, "updated", "second apply updates the existing block in place");
        let bak2 = rep2.backup.expect("the pristine backup is still reported");
        assert_eq!(bak2, bak1, "same stable pristine-backup path, not a fresh one");
        assert_eq!(std::fs::read_to_string(&bak2).unwrap(), user_json, "pristine backup still holds the ORIGINAL bytes");

        // 4) preview() reports an update without writing (the opencode file exists from step 1).
        assert_eq!(preview("opencode", origin).unwrap().action, "update");

        // 5) Hermes YAML comment-loss path (the exact defect the adversarial review caught): a user
        //    with a COMMENTED ~/.hermes/config.yaml must (a) get a warning from preview, (b) have the
        //    original (comment and all) backed up verbatim, (c) keep their other top-level keys, and
        //    (d) end up with the OpenHydra model block — even though serde_yaml drops the comment.
        let hermes_path = sandbox.join(".hermes/config.yaml");
        let commented = "# my hermes config — keep me\nmodel:\n  provider: openai\n  name: gpt-4\nkeep_this: 42\n";
        std::fs::write(&hermes_path, commented).unwrap();
        let pv = preview("hermes", origin).unwrap();
        assert_eq!(pv.action, "update", "hermes: existing file is an update");
        assert!(
            pv.warning.as_deref().is_some_and(|w| w.contains("reformatted")),
            "hermes: preview MUST warn about YAML comment loss (regression guard for the fixed bug)"
        );
        let rep = apply("hermes", origin).unwrap();
        let hbak = rep.backup.expect("hermes: commented config must be backed up");
        assert_eq!(std::fs::read_to_string(&hbak).unwrap(), commented, "hermes backup holds the ORIGINAL incl. comment");
        let hout = std::fs::read_to_string(&hermes_path).unwrap();
        assert!(!hout.contains("# my hermes config"), "hermes: serde_yaml drops the comment (why the warning exists)");
        assert!(hout.contains("keep_this"), "hermes: user's other top-level keys survive the merge");
        assert!(hout.contains("openhydra/auto") && hout.contains("16527"), "hermes: OpenHydra model block written");

        std::env::remove_var("HOME");
        std::env::remove_var("XDG_CONFIG_HOME");
        let _ = std::fs::remove_dir_all(&sandbox);
    }

    #[test]
    fn opencode_merge_adds_provider_and_default_model() {
        let (out, action) = merge_opencode_json("", "http://127.0.0.1:16527/v1", &[]).unwrap();
        assert_eq!(action, "added");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["provider"]["openhydra"]["npm"], "@ai-sdk/openai-compatible");
        assert_eq!(v["provider"]["openhydra"]["options"]["baseURL"], "http://127.0.0.1:16527/v1");
        // The id under the provider is BARE `auto` (OpenCode splits provider/model on `/`), and the
        // top-level ref is `openhydra/auto` — NOT the old double-scoped `openhydra/openhydra/auto`.
        assert!(v["provider"]["openhydra"]["models"].get("auto").is_some(), "provider model id is bare `auto`");
        assert!(v["provider"]["openhydra"]["models"].get("openhydra/auto").is_none(), "must NOT double-scope the id");
        assert_eq!(v["model"], "openhydra/auto");
    }

    #[test]
    fn opencode_merge_preserves_other_config_and_user_model() {
        let existing = r#"{ "theme": "dark", "model": "anthropic/claude", "provider": { "other": { "x": 1 } } }"#;
        let (out, action) = merge_opencode_json(existing, "http://h/v1", &[]).unwrap();
        assert_eq!(action, "added"); // no prior openhydra provider
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["theme"], "dark"); // untouched
        assert_eq!(v["provider"]["other"]["x"], 1); // untouched
        assert_eq!(v["provider"]["openhydra"]["name"], "OpenHydra"); // added
        assert_eq!(v["model"], "anthropic/claude"); // user's model NOT overridden
        // Idempotent: a second run updates in place.
        let (out2, action2) = merge_opencode_json(&out, "http://h/v1", &[]).unwrap();
        assert_eq!(action2, "updated");
        let v2: Value = serde_json::from_str(&out2).unwrap();
        assert_eq!(v2["provider"].as_object().unwrap().len(), 2); // other + openhydra, no dupes
    }

    #[test]
    fn opencode_migrates_our_stale_double_scoped_default_but_not_a_user_choice() {
        // A config left by an OLD OpenHydra connect (the buggy `openhydra/openhydra/auto`) must be
        // migrated to `openhydra/auto` — OpenCode couldn't resolve the old one → fell back to kimi.
        let stale = r#"{"model":"openhydra/openhydra/auto","provider":{"openhydra":{"models":{"openhydra/auto":{}}}}}"#;
        let (out, _) = merge_opencode_json(stale, "http://h/v1", &[]).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["model"], "openhydra/auto", "stale double-scoped default is migrated");
        assert!(v["provider"]["openhydra"]["models"].get("auto").is_some());
        // A real user choice is preserved untouched.
        let user = r#"{"model":"anthropic/claude-sonnet","provider":{}}"#;
        let (out2, _) = merge_opencode_json(user, "http://h/v1", &[]).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out2).unwrap()["model"], "anthropic/claude-sonnet");
    }

    #[test]
    fn declaring_specific_models_puts_them_in_the_picker() {
        let models = vec!["qwen3-coder:30b-a3b-q8_0".to_string()];
        // Pi: the id appears in the models array (so Pi's own /model picker + footer show it).
        let (pi, _) = merge_pi_models_json("", "http://h/v1", &models).unwrap();
        let pv: Value = serde_json::from_str(&pi).unwrap();
        let ids: Vec<_> = pv["providers"]["openhydra"]["models"].as_array().unwrap().iter()
            .map(|m| m["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["auto", "qwen3-coder:30b-a3b-q8_0"], "auto + the declared model");
        // OpenCode: the id becomes a key under the provider's `models` map.
        let (oc, _) = merge_opencode_json("", "http://h/v1", &models).unwrap();
        let ov: Value = serde_json::from_str(&oc).unwrap();
        assert!(ov["provider"]["openhydra"]["models"].get("qwen3-coder:30b-a3b-q8_0").is_some());
        assert!(ov["provider"]["openhydra"]["models"].get("auto").is_some());
        // Continue: a distinct-named entry per model (so its picker lists both).
        let (cn, _) = merge_continue_yaml("", "http://h/v1", &models).unwrap();
        let cv: serde_yaml::Value = serde_yaml::from_str(&cn).unwrap();
        let names: Vec<_> = cv["models"].as_sequence().unwrap().iter()
            .map(|m| m["name"].as_str().unwrap().to_string()).collect();
        assert!(names.contains(&"OpenHydra Auto".to_string()));
        assert!(names.contains(&"OpenHydra: qwen3-coder:30b-a3b-q8_0".to_string()));
    }

    #[test]
    fn continue_does_not_clobber_a_user_model_named_like_ours() {
        // A user's own model named "OpenHydra Local" must survive; only OUR entries (old exact
        // "OpenHydra", new "OpenHydra Auto", "OpenHydra: …") get replaced on re-run.
        let existing = "name: a\nversion: 1\nmodels:\n- name: OpenHydra Local\n  provider: openai\n  model: x\n- name: OpenHydra\n  provider: openai\n  model: old\n";
        let (out, _) = merge_continue_yaml(existing, "http://h/v1", &[]).unwrap();
        let v: serde_yaml::Value = serde_yaml::from_str(&out).unwrap();
        let names: Vec<_> = v["models"].as_sequence().unwrap().iter()
            .map(|m| m["name"].as_str().unwrap().to_string()).collect();
        assert!(names.contains(&"OpenHydra Local".to_string()), "user's OpenHydra-Local model preserved");
        assert!(names.contains(&"OpenHydra Auto".to_string()), "our auto entry (re)added");
        assert_eq!(names.iter().filter(|n| n.as_str() == "OpenHydra").count(), 0, "old exact-OpenHydra entry migrated");
    }

    #[test]
    fn claude_merge_sets_env_block() {
        let (out, action) = merge_claude_settings(r#"{"model":"sonnet"}"#, "http://127.0.0.1:16527").unwrap();
        assert_eq!(action, "added");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["model"], "sonnet"); // untouched
        assert_eq!(v["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:16527");
        assert_eq!(v["env"]["ANTHROPIC_API_KEY"], "oh-local");
        // Idempotent update.
        let (_out2, action2) = merge_claude_settings(&out, "http://127.0.0.1:16527").unwrap();
        assert_eq!(action2, "updated");
    }

    #[test]
    fn continue_merge_appends_and_is_idempotent() {
        let existing = "name: my-assistant\nmodels:\n  - name: Local\n    provider: ollama\n    model: llama3\n";
        let (out, action) = merge_continue_yaml(existing, "http://127.0.0.1:16527/v1", &[]).unwrap();
        assert_eq!(action, "added");
        let v: serde_yaml::Value = serde_yaml::from_str(&out).unwrap();
        let models = v["models"].as_sequence().unwrap();
        assert_eq!(models.len(), 2); // Local + OpenHydra
        assert_eq!(v["name"].as_str(), Some("my-assistant")); // top-level untouched (not clobbered)
        let oh = models.iter().find(|m| m["name"].as_str() == Some("OpenHydra Auto")).unwrap();
        assert_eq!(oh["apiBase"].as_str(), Some("http://127.0.0.1:16527/v1"));
        assert_eq!(oh["model"].as_str(), Some("openhydra/auto"));
        // Re-run replaces, doesn't duplicate.
        let (out2, action2) = merge_continue_yaml(&out, "http://127.0.0.1:16527/v1", &[]).unwrap();
        assert_eq!(action2, "updated");
        let v2: serde_yaml::Value = serde_yaml::from_str(&out2).unwrap();
        assert_eq!(v2["models"].as_sequence().unwrap().len(), 2); // still 2, no dupe
    }

    #[test]
    fn continue_merge_from_empty_creates_models_list() {
        let (out, action) = merge_continue_yaml("", "http://h/v1", &[]).unwrap();
        assert_eq!(action, "added");
        let v: serde_yaml::Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(v["models"].as_sequence().unwrap().len(), 1);
        // Continue's schema requires top-level name+version; a fresh create MUST include them or the
        // Continue CLI/extension rejects the file ("name: Required, version: Required"). Regression guard.
        assert_eq!(v["name"].as_str(), Some("OpenHydra"), "fresh Continue config needs a top-level name");
        assert!(v["version"].as_str().is_some(), "fresh Continue config needs a top-level version");
    }

    #[test]
    fn base_url_shape_per_tool() {
        assert_eq!(base_for(Kind::OpencodeJson, "http://127.0.0.1:16527"), "http://127.0.0.1:16527/v1");
        assert_eq!(base_for(Kind::ContinueYaml, "http://127.0.0.1:16527"), "http://127.0.0.1:16527/v1");
        // Claude Code appends /v1/messages itself → bare origin.
        assert_eq!(base_for(Kind::ClaudeSettings, "http://127.0.0.1:16527"), "http://127.0.0.1:16527");
        // Hermes + Pi are OpenAI-compatible → /v1.
        assert_eq!(base_for(Kind::HermesYaml, "http://127.0.0.1:16527"), "http://127.0.0.1:16527/v1");
        assert_eq!(base_for(Kind::PiModelsJson, "http://127.0.0.1:16527"), "http://127.0.0.1:16527/v1");
    }

    #[test]
    fn hermes_merge_sets_custom_model_block() {
        let (out, action) = merge_hermes_yaml("", "http://127.0.0.1:16527/v1").unwrap();
        assert_eq!(action, "added");
        let v: serde_yaml::Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(v["model"]["provider"].as_str(), Some("custom"));
        assert_eq!(v["model"]["base_url"].as_str(), Some("http://127.0.0.1:16527/v1"));
        assert_eq!(v["model"]["name"].as_str(), Some("openhydra/auto")); // Hermes' model-id field is `name`
        assert_eq!(v["model"]["api_key"].as_str(), Some("oh-local"));
    }

    #[test]
    fn hermes_merge_preserves_other_keys_and_is_idempotent() {
        // A prior model block (different provider) + an unrelated top-level key.
        let existing = "terminal:\n  backend: local\nmodel:\n  provider: openai\n  name: gpt-4\n  reasoning: high\n";
        let (out, action) = merge_hermes_yaml(existing, "http://h/v1").unwrap();
        assert_eq!(action, "added"); // no base_url in the prior model block
        let v: serde_yaml::Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(v["terminal"]["backend"].as_str(), Some("local")); // top-level untouched
        assert_eq!(v["model"]["reasoning"].as_str(), Some("high")); // unknown model sub-key preserved
        assert_eq!(v["model"]["provider"].as_str(), Some("custom")); // overwritten to custom
        assert_eq!(v["model"]["name"].as_str(), Some("openhydra/auto")); // pointed at OpenHydra
        // Idempotent: base_url now present → second run reports "updated", no dupes.
        let (_out2, action2) = merge_hermes_yaml(&out, "http://h/v1").unwrap();
        assert_eq!(action2, "updated");
    }

    #[test]
    fn pi_merge_adds_provider_and_preserves_others() {
        let existing = r#"{"providers":{"other":{"baseUrl":"http://x/v1","api":"openai-completions","models":[]}}}"#;
        let (out, action) = merge_pi_models_json(existing, "http://127.0.0.1:16527/v1", &[]).unwrap();
        assert_eq!(action, "added");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["providers"]["other"]["baseUrl"], "http://x/v1"); // other provider untouched
        assert_eq!(v["providers"]["openhydra"]["api"], "openai-completions");
        assert_eq!(v["providers"]["openhydra"]["baseUrl"], "http://127.0.0.1:16527/v1");
        // Bare `auto` id — Pi splits `--model provider/model` on `/`.
        assert_eq!(v["providers"]["openhydra"]["models"][0]["id"], "auto");
        assert_eq!(v["providers"]["openhydra"]["models"][0]["cost"]["input"], 0); // rich shape Pi expects
        // Idempotent: replaces our provider, keeps the other.
        let (out2, action2) = merge_pi_models_json(&out, "http://127.0.0.1:16527/v1", &[]).unwrap();
        assert_eq!(action2, "updated");
        let v2: Value = serde_json::from_str(&out2).unwrap();
        assert_eq!(v2["providers"].as_object().unwrap().len(), 2); // other + openhydra, no dupe
    }
}
