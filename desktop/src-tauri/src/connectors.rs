// OpenHydra Desktop — Connectors: detect installed coding tools and one-click wire them to the
// local gateway by merging an OpenHydra block into the tool's own config file.
//
// Design (grounded in live validation on 2026-08-14):
//   • OpenCode  → ~/.config/opencode/opencode.json — a custom `@ai-sdk/openai-compatible` provider
//     `openhydra` pointing at the gateway (VERIFIED: this exact block routed `opencode run` to a
//     live network model).
//   • Claude Code → ~/.claude/settings.json — an `env` block setting ANTHROPIC_BASE_URL/KEY
//     (VERIFIED: those env values drove Claude Code end-to-end via native /v1/messages).
//   • Continue  → ~/.continue/config.yaml — an OpenHydra entry in the `models:` list (standard
//     Continue OpenAI-compatible shape).
//
// Every writer is a PURE merge function (unit-tested here): parse the existing file, insert/replace
// ONLY the OpenHydra-owned key (idempotent, never clobbers unrelated config), re-serialize. The
// Tauri `connector_apply` command backs the file up first and reports exactly what changed, so the
// UI can show a confirm/diff before touching a file we didn't create.

use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{json, Value};

use crate::installer::resolve_program;

/// A dummy non-empty key: the loopback gateway is open, but OpenAI/Anthropic clients require a
/// non-empty key field.
const LOCAL_KEY: &str = "oh-local";
/// The meta-model every connector defaults to (the gateway resolves it to a live model).
const AUTO_MODEL: &str = "openhydra/auto";

/// How a connector is wired. All current tools are file-config; the enum leaves room for a future
/// launch-only tool.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    OpencodeJson,
    ClaudeSettings,
    ContinueYaml,
}

struct ToolSpec {
    key: &'static str,
    label: &'static str,
    /// Candidate binary names for PATH detection; empty ⇒ detect by config dir instead.
    bins: &'static [&'static str],
    kind: Kind,
}

const TOOLS: &[ToolSpec] = &[
    ToolSpec { key: "opencode", label: "OpenCode", bins: &["opencode"], kind: Kind::OpencodeJson },
    ToolSpec { key: "claude", label: "Claude Code", bins: &["claude"], kind: Kind::ClaudeSettings },
    // Continue is a VS Code / JetBrains extension (no CLI) — detected by its config dir.
    ToolSpec { key: "continue", label: "Continue", bins: &[], kind: Kind::ContinueYaml },
];

fn spec(key: &str) -> Option<&'static ToolSpec> {
    TOOLS.iter().find(|t| t.key == key)
}

/// `$HOME` (unix) / `%USERPROFILE%` (windows).
fn home() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// The config file a tool's OpenHydra block lives in.
fn config_path(kind: Kind) -> Option<PathBuf> {
    let h = home()?;
    Some(match kind {
        // OpenCode honours $XDG_CONFIG_HOME; fall back to ~/.config.
        Kind::OpencodeJson => xdg_config().unwrap_or_else(|| h.join(".config")).join("opencode/opencode.json"),
        Kind::ClaudeSettings => h.join(".claude/settings.json"),
        Kind::ContinueYaml => h.join(".continue/config.yaml"),
    })
}

fn xdg_config() -> Option<PathBuf> {
    std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).filter(|p| !p.as_os_str().is_empty())
}

// ── Detection ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ConnectorStatus {
    key: String,
    label: String,
    /// "opencode" | "claude" | "continue" config kind, for the UI.
    kind: String,
    installed: bool,
    /// The resolved binary path, or the config dir that evidences an install.
    detail: Option<String>,
}

fn kind_str(k: Kind) -> &'static str {
    match k {
        Kind::OpencodeJson => "opencode",
        Kind::ClaudeSettings => "claude",
        Kind::ContinueYaml => "continue",
    }
}

fn detect(spec: &ToolSpec) -> (bool, Option<String>) {
    // Binary tools: resolve on PATH / common install dirs (the GUI-PATH fix reused).
    if !spec.bins.is_empty() {
        if let Some(p) = spec.bins.iter().find_map(|b| resolve_program(b)) {
            return (true, Some(p.display().to_string()));
        }
    }
    // Continue (extension, no CLI): evidenced by its config dir.
    if spec.kind == Kind::ContinueYaml {
        if let Some(dir) = home().map(|h| h.join(".continue")) {
            if dir.exists() {
                return (true, Some(dir.display().to_string()));
            }
        }
    }
    (false, None)
}

/// Detection status for every known connector (installed? where?). Read-only.
pub fn statuses() -> Vec<ConnectorStatus> {
    TOOLS
        .iter()
        .map(|t| {
            let (installed, detail) = detect(t);
            ConnectorStatus { key: t.key.into(), label: t.label.into(), kind: kind_str(t.kind).into(), installed, detail }
        })
        .collect()
}

// ── Pure config merges (unit-tested) ─────────────────────────────────────────

/// The base URL a tool should call. OpenCode/Continue speak OpenAI (`/v1`); Claude Code speaks the
/// Anthropic Messages API and appends `/v1/messages` itself, so it takes the bare origin.
fn base_for(kind: Kind, origin: &str) -> String {
    match kind {
        Kind::OpencodeJson | Kind::ContinueYaml => format!("{origin}/v1"),
        Kind::ClaudeSettings => origin.to_string(),
    }
}

/// Merge the `openhydra` provider into OpenCode's JSON config. Idempotent; only touches
/// `provider.openhydra`, `$schema`, and (when absent) the default `model`.
pub fn merge_opencode_json(existing: &str, api_base: &str) -> Result<(String, &'static str), String> {
    let mut root: Value = parse_json_or_empty(existing, "opencode.json")?;
    let obj = root.as_object_mut().ok_or("opencode.json is not a JSON object")?;
    obj.entry("$schema").or_insert_with(|| json!("https://opencode.ai/config.json"));
    let had = obj.get("provider").and_then(|p| p.get("openhydra")).is_some();
    let provider = obj.entry("provider").or_insert_with(|| json!({}));
    let provider = provider.as_object_mut().ok_or("`provider` in opencode.json is not an object")?;
    provider.insert(
        "openhydra".into(),
        json!({
            "npm": "@ai-sdk/openai-compatible",
            "name": "OpenHydra",
            "options": { "baseURL": api_base, "apiKey": LOCAL_KEY },
            "models": { AUTO_MODEL: { "name": "OpenHydra Auto" } },
        }),
    );
    // Activate OpenHydra as the default model only if the user hasn't chosen one — never override an
    // existing selection (they can pick "OpenHydra Auto" in OpenCode's model picker).
    let our_ref = json!(format!("openhydra/{AUTO_MODEL}"));
    if obj.get("model").is_none() {
        obj.insert("model".into(), our_ref);
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
pub fn merge_continue_yaml(existing: &str, api_base: &str) -> Result<(String, &'static str), String> {
    use serde_yaml::Value as Y;
    let mut root: Y = if existing.trim().is_empty() {
        Y::Mapping(serde_yaml::Mapping::new())
    } else {
        serde_yaml::from_str(existing).map_err(|e| format!("parse config.yaml: {e}"))?
    };
    let map = root.as_mapping_mut().ok_or("config.yaml is not a mapping")?;
    let models_key = Y::String("models".into());
    if !map.contains_key(&models_key) {
        map.insert(models_key.clone(), Y::Sequence(Vec::new()));
    }
    let seq = map
        .get_mut(&models_key)
        .and_then(|v| v.as_sequence_mut())
        .ok_or("`models` in config.yaml is not a list")?;
    let before = seq.len();
    seq.retain(|item| item.get("name").and_then(|n| n.as_str()) != Some("OpenHydra"));
    let had = seq.len() != before;
    seq.push(continue_model_entry(api_base));
    let out = serde_yaml::to_string(&root).map_err(|e| e.to_string())?;
    Ok((out, if had { "updated" } else { "added" }))
}

fn continue_model_entry(api_base: &str) -> serde_yaml::Value {
    use serde_yaml::Value as Y;
    let s = |x: &str| Y::String(x.to_string());
    let mut m = serde_yaml::Mapping::new();
    m.insert(s("name"), s("OpenHydra"));
    m.insert(s("provider"), s("openai"));
    m.insert(s("model"), s(AUTO_MODEL));
    m.insert(s("apiBase"), s(api_base));
    m.insert(s("apiKey"), s(LOCAL_KEY));
    m.insert(s("roles"), Y::Sequence(vec![s("chat"), s("edit"), s("apply")]));
    Y::Mapping(m)
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
fn merge_for(kind: Kind, existing: &str, origin: &str) -> Result<(String, &'static str), String> {
    let base = base_for(kind, origin);
    match kind {
        Kind::OpencodeJson => merge_opencode_json(existing, &base),
        Kind::ClaudeSettings => merge_claude_settings(existing, &base),
        Kind::ContinueYaml => merge_continue_yaml(existing, &base),
    }
}

// ── Preview / apply (Tauri commands) ─────────────────────────────────────────

#[derive(Serialize)]
pub struct ConnectPreview {
    key: String,
    kind: String,
    path: String,
    /// "create" (no file yet) | "update" (merge into existing).
    action: String,
    /// The new full file content that Apply would write (for a diff/preview).
    preview: String,
    /// A caveat to surface before writing (e.g. YAML comment loss), if any.
    warning: Option<String>,
}

#[derive(Serialize)]
pub struct ConnectReport {
    key: String,
    path: String,
    /// Where the prior file was backed up (None when the file was freshly created).
    backup: Option<String>,
    /// "added" | "updated" — whether an OpenHydra block already existed.
    action: String,
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
    let spec = spec(key).ok_or_else(|| format!("unknown connector '{key}'"))?;
    let path = config_path(spec.kind).ok_or("cannot resolve home directory")?;
    let existing = read_existing(&path)?;
    let create = existing.trim().is_empty();
    let (preview, _action) = merge_for(spec.kind, &existing, origin)?;
    let warning = (spec.kind == Kind::ContinueYaml && !create)
        .then(|| "Continue's config.yaml will be reformatted (comments/spacing are not preserved). The original is backed up.".to_string());
    Ok(ConnectPreview {
        key: key.into(),
        kind: kind_str(spec.kind).into(),
        path: path.display().to_string(),
        action: if create { "create".into() } else { "update".into() },
        preview,
        warning,
    })
}

/// A backup path that does not overwrite an existing backup: `<file>.openhydra.bak`, then `.1`, `.2`…
fn backup_path(path: &Path) -> PathBuf {
    let base = path.with_extension(format!(
        "{}.openhydra.bak",
        path.extension().and_then(|e| e.to_str()).unwrap_or("")
    ));
    if !base.exists() {
        return base;
    }
    for n in 1.. {
        let cand = base.with_extension(format!("bak.{n}"));
        if !cand.exists() {
            return cand;
        }
    }
    unreachable!()
}

/// Write the OpenHydra block into `key`'s config, backing up any existing file first.
pub fn apply(key: &str, origin: &str) -> Result<ConnectReport, String> {
    let spec = spec(key).ok_or_else(|| format!("unknown connector '{key}'"))?;
    let path = config_path(spec.kind).ok_or("cannot resolve home directory")?;
    let existing = read_existing(&path)?;
    let (new_content, action) = merge_for(spec.kind, &existing, origin)?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    // Back up a non-empty existing file before overwriting.
    let backup = if !existing.trim().is_empty() {
        let bak = backup_path(&path);
        std::fs::write(&bak, &existing).map_err(|e| format!("backup {}: {e}", bak.display()))?;
        Some(bak.display().to_string())
    } else {
        None
    };
    std::fs::write(&path, new_content).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(ConnectReport { key: key.into(), path: path.display().to_string(), backup, action: action.into() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-app file-write validation (the "Connect" button → `connector_apply` → [`apply`]):
    /// drives the ACTUAL filesystem write/backup path against a sandbox `HOME`, so it exercises
    /// what the mock-bridge UI test couldn't (the real Rust write) without touching real dotfiles.
    /// The only thing a literal in-window click adds on top is the JS→`invoke` dispatch.
    #[test]
    fn apply_writes_backs_up_and_is_idempotent_on_the_real_fs() {
        let sandbox = std::env::temp_dir().join(format!("oh-connect-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&sandbox);
        std::fs::create_dir_all(&sandbox).unwrap();
        // Redirect every tool's config root into the sandbox (home() reads $HOME / $XDG_CONFIG_HOME).
        std::env::set_var("HOME", &sandbox);
        std::env::set_var("XDG_CONFIG_HOME", sandbox.join(".config"));
        let origin = "http://127.0.0.1:16527";

        // 1) Fresh create for all three tools: file written to the real fs, no backup, block present.
        for key in ["opencode", "claude", "continue"] {
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

        // 3) Idempotency + backup non-clobber: apply again → "updated", a SECOND, distinct backup.
        let rep2 = apply("claude", origin).unwrap();
        assert_eq!(rep2.action, "updated", "second apply updates the existing block in place");
        let bak2 = rep2.backup.expect("second apply backs up again");
        assert_ne!(bak2, bak1, "the second backup does not clobber the first");

        // 4) preview() reports an update without writing (the opencode file exists from step 1).
        assert_eq!(preview("opencode", origin).unwrap().action, "update");

        std::env::remove_var("HOME");
        std::env::remove_var("XDG_CONFIG_HOME");
        let _ = std::fs::remove_dir_all(&sandbox);
    }

    #[test]
    fn opencode_merge_adds_provider_and_default_model() {
        let (out, action) = merge_opencode_json("", "http://127.0.0.1:16527/v1").unwrap();
        assert_eq!(action, "added");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["provider"]["openhydra"]["npm"], "@ai-sdk/openai-compatible");
        assert_eq!(v["provider"]["openhydra"]["options"]["baseURL"], "http://127.0.0.1:16527/v1");
        assert_eq!(v["model"], "openhydra/openhydra/auto");
    }

    #[test]
    fn opencode_merge_preserves_other_config_and_user_model() {
        let existing = r#"{ "theme": "dark", "model": "anthropic/claude", "provider": { "other": { "x": 1 } } }"#;
        let (out, action) = merge_opencode_json(existing, "http://h/v1").unwrap();
        assert_eq!(action, "added"); // no prior openhydra provider
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["theme"], "dark"); // untouched
        assert_eq!(v["provider"]["other"]["x"], 1); // untouched
        assert_eq!(v["provider"]["openhydra"]["name"], "OpenHydra"); // added
        assert_eq!(v["model"], "anthropic/claude"); // user's model NOT overridden
        // Idempotent: a second run updates in place.
        let (out2, action2) = merge_opencode_json(&out, "http://h/v1").unwrap();
        assert_eq!(action2, "updated");
        let v2: Value = serde_json::from_str(&out2).unwrap();
        assert_eq!(v2["provider"].as_object().unwrap().len(), 2); // other + openhydra, no dupes
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
        let (out, action) = merge_continue_yaml(existing, "http://127.0.0.1:16527/v1").unwrap();
        assert_eq!(action, "added");
        let v: serde_yaml::Value = serde_yaml::from_str(&out).unwrap();
        let models = v["models"].as_sequence().unwrap();
        assert_eq!(models.len(), 2); // Local + OpenHydra
        assert_eq!(v["name"].as_str(), Some("my-assistant")); // top-level untouched
        let oh = models.iter().find(|m| m["name"].as_str() == Some("OpenHydra")).unwrap();
        assert_eq!(oh["apiBase"].as_str(), Some("http://127.0.0.1:16527/v1"));
        assert_eq!(oh["model"].as_str(), Some("openhydra/auto"));
        // Re-run replaces, doesn't duplicate.
        let (out2, action2) = merge_continue_yaml(&out, "http://127.0.0.1:16527/v1").unwrap();
        assert_eq!(action2, "updated");
        let v2: serde_yaml::Value = serde_yaml::from_str(&out2).unwrap();
        assert_eq!(v2["models"].as_sequence().unwrap().len(), 2); // still 2, no dupe
    }

    #[test]
    fn continue_merge_from_empty_creates_models_list() {
        let (out, action) = merge_continue_yaml("", "http://h/v1").unwrap();
        assert_eq!(action, "added");
        let v: serde_yaml::Value = serde_yaml::from_str(&out).unwrap();
        assert_eq!(v["models"].as_sequence().unwrap().len(), 1);
    }

    #[test]
    fn base_url_shape_per_tool() {
        assert_eq!(base_for(Kind::OpencodeJson, "http://127.0.0.1:16527"), "http://127.0.0.1:16527/v1");
        assert_eq!(base_for(Kind::ContinueYaml, "http://127.0.0.1:16527"), "http://127.0.0.1:16527/v1");
        // Claude Code appends /v1/messages itself → bare origin.
        assert_eq!(base_for(Kind::ClaudeSettings, "http://127.0.0.1:16527"), "http://127.0.0.1:16527");
    }
}
