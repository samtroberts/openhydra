// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Gateway-side `/`-commands for connected coding tools.
//!
//! OpenHydra can't inject UI into OpenCode / Continue / Claude Code, but every message those
//! tools send flows through the gateway — so the gateway parses a small set of slash-commands
//! out of the latest user turn and answers them itself (as an ordinary assistant message)
//! instead of routing to inference. The effect: `/models` and `/model <id>` work **uniformly
//! across every connected tool**, which none of them offer natively, at the cost of no tokens
//! and no receipt.
//!
//! Two pure pieces (unit-tested without axum): [`parse`] turns a user turn into a
//! [`SlashCommand`] (strict — a message that merely mentions a slash is left as prose), and
//! [`render`] turns a command + the live model set + the session's current pin into the reply
//! text and any new sticky selection. [`SessionModels`] holds the per-session pin.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Default idle lifetime of a `/model` pin (1 h). A coding-tool session holds a fixed API key,
/// so this survives an active session and is refreshed on each request; a long-idle key is
/// forgotten. Never persisted.
pub const DEFAULT_SESSION_TTL: Duration = Duration::from_secs(3600);

/// A recognized gateway command parsed from a user turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    /// `/models [filter]` — list the currently-served models (optionally substring-filtered).
    Models(Option<String>),
    /// `/model` — show the session's current selection.
    ModelShow,
    /// `/model <id>` — pin this session to a specific model (or `/model auto` to reset).
    ModelSet(String),
    /// `/help` — list the commands.
    Help,
}

/// Is `m` the auto meta-model (`auto` / `openhydra/auto`, case/space-insensitive)? Kept local so
/// this module needs nothing from `gateway`.
fn is_auto(m: &str) -> bool {
    let m = m.trim();
    m.eq_ignore_ascii_case("auto") || m.eq_ignore_ascii_case("openhydra/auto")
}

/// Parse the text of a user turn into a [`SlashCommand`], or `None` for ordinary prose.
///
/// Deliberately strict so it never hijacks a real prompt: the trimmed text must be a single
/// line that *starts* with a known command token. A message that merely mentions a slash, spans
/// multiple lines, or carries stray words after a keyword that doesn't take them is prose.
pub fn parse(text: &str) -> Option<SlashCommand> {
    let t = text.trim();
    // Single-line only: a multi-line message is a prompt, never a command.
    if !t.starts_with('/') || t.contains(['\n', '\r']) {
        return None;
    }
    let mut it = t.splitn(2, char::is_whitespace);
    let head = it.next().unwrap_or("");
    let rest = it.next().map(str::trim).filter(|s| !s.is_empty());
    match head {
        "/help" if rest.is_none() => Some(SlashCommand::Help),
        "/models" => Some(SlashCommand::Models(rest.map(str::to_string))),
        "/model" => match rest {
            None => Some(SlashCommand::ModelShow),
            // A model id has no whitespace; `/model a b` is prose, not a set.
            Some(id) if !id.contains(char::is_whitespace) => Some(SlashCommand::ModelSet(id.to_string())),
            Some(_) => None,
        },
        _ => None,
    }
}

/// The outcome of handling a command: the assistant reply text, and (for `/model <id>`) the new
/// sticky selection to store for the session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandResult {
    pub reply: String,
    /// `Some(model)` ⇒ store as the session's pin; `None` ⇒ leave the selection unchanged.
    pub set_model: Option<String>,
}

const HELP: &str = "OpenHydra gateway commands:\n\
• `/models [filter]` — list the models currently served on the network\n\
• `/model <id>` — route this session to a specific model\n\
• `/model auto` — let the gateway pick a served model automatically (default)\n\
• `/model` — show the current selection\n\
• `/help` — show this message\n\
\n\
Handled by OpenHydra (not your model); they cost no inference.";

/// Render a command against the live model set (`known`) and the session's current pin
/// (`current`). Pure — the caller applies `set_model` and formats `reply` into the wire shape.
pub fn render(cmd: &SlashCommand, known: &[String], current: Option<&str>) -> CommandResult {
    match cmd {
        SlashCommand::Help => CommandResult { reply: HELP.to_string(), set_model: None },

        SlashCommand::Models(filter) => {
            let needle = filter.as_deref().map(str::to_ascii_lowercase);
            let mut lines: Vec<String> = Vec::new();
            for m in known {
                if let Some(n) = &needle {
                    if !m.to_ascii_lowercase().contains(n.as_str()) {
                        continue;
                    }
                }
                let marker = if current == Some(m.as_str()) { "  ● (current)" } else { "" };
                lines.push(format!("• {m}{marker}"));
            }
            let reply = if known.is_empty() {
                "No models are being served on the network yet — try again once a provider announces."
                    .to_string()
            } else if lines.is_empty() {
                format!(
                    "No served model matches '{}'. Use `/models` to see all {} live model(s).",
                    filter.as_deref().unwrap_or(""),
                    known.len()
                )
            } else {
                let header = match current {
                    Some(c) if !is_auto(c) => format!("Models on the network — `{c}` pinned:"),
                    _ => "Models on the network — auto-selecting:".to_string(),
                };
                format!(
                    "{header}\n{}\n\nPin one with `/model <id>`, or `/model auto` for automatic.",
                    lines.join("\n")
                )
            };
            CommandResult { reply, set_model: None }
        }

        SlashCommand::ModelShow => {
            let reply = match current {
                Some(c) if !is_auto(c) => format!(
                    "Current model: `{c}`.\nChange with `/model <id>`, list with `/models`, reset with `/model auto`."
                ),
                _ => "No model pinned — requests use `openhydra/auto` (automatic selection).\n\
                      Pin one with `/model <id>`; list with `/models`."
                    .to_string(),
            };
            CommandResult { reply, set_model: None }
        }

        SlashCommand::ModelSet(id) => {
            if is_auto(id) {
                CommandResult {
                    reply: "Now using `openhydra/auto` — the gateway picks a served model automatically."
                        .to_string(),
                    set_model: Some("openhydra/auto".to_string()),
                }
            } else if known.iter().any(|m| m == id) {
                CommandResult { reply: format!("Now using `{id}`."), set_model: Some(id.clone()) }
            } else {
                // Discovery is laggy (a just-online model can take ~2 min to appear), so pin it
                // anyway and let resolution fall back to auto if no provider turns up — but say so.
                CommandResult {
                    reply: format!(
                        "Pinned `{id}`, but it isn't in the current served list — requests will try it \
                         and fall back to `openhydra/auto` if no provider is found. `/models` shows what's live."
                    ),
                    set_model: Some(id.clone()),
                }
            }
        }
    }
}

/// Per-session sticky `/model` selections, keyed by the caller's API-key identity (see
/// `gateway::session_key`). Entries expire after `ttl` of inactivity and are never persisted;
/// a read refreshes the idle timer so an active session stays pinned.
pub struct SessionModels {
    inner: Mutex<HashMap<String, Entry>>,
    ttl: Duration,
}

struct Entry {
    model: String,
    touched: Instant,
}

impl SessionModels {
    pub fn new(ttl: Duration) -> Self {
        Self { inner: Mutex::new(HashMap::new()), ttl }
    }

    /// The pinned model for `key`, if set and not expired. A hit refreshes the idle timer.
    pub fn get(&self, key: &str) -> Option<String> {
        let mut m = self.inner.lock().unwrap();
        match m.get_mut(key) {
            Some(e) if e.touched.elapsed() <= self.ttl => {
                e.touched = Instant::now();
                Some(e.model.clone())
            }
            Some(_) => {
                m.remove(key);
                None
            }
            None => None,
        }
    }

    /// Pin `model` for `key`, opportunistically evicting expired entries to bound memory.
    pub fn set(&self, key: &str, model: String) {
        let mut m = self.inner.lock().unwrap();
        m.retain(|_, e| e.touched.elapsed() <= self.ttl);
        m.insert(key.to_string(), Entry { model, touched: Instant::now() });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_recognizes_commands() {
        assert_eq!(parse("/help"), Some(SlashCommand::Help));
        assert_eq!(parse("  /help  "), Some(SlashCommand::Help));
        assert_eq!(parse("/models"), Some(SlashCommand::Models(None)));
        assert_eq!(parse("/models qwen"), Some(SlashCommand::Models(Some("qwen".into()))));
        assert_eq!(parse("/model"), Some(SlashCommand::ModelShow));
        assert_eq!(parse("/model qwen2.5:7b"), Some(SlashCommand::ModelSet("qwen2.5:7b".into())));
        assert_eq!(parse("/model auto"), Some(SlashCommand::ModelSet("auto".into())));
    }

    #[test]
    fn parse_rejects_prose_and_near_misses() {
        assert_eq!(parse("hello world"), None);
        assert_eq!(parse("please run /models for me"), None); // doesn't start with /
        assert_eq!(parse("/model a b"), None); // id can't contain whitespace
        assert_eq!(parse("/help me"), None); // /help takes no args
        assert_eq!(parse("/unknown"), None);
        assert_eq!(parse("/model gpt\nand more"), None); // multi-line ⇒ prose
        assert_eq!(parse("/"), None);
        assert_eq!(parse(""), None);
    }

    #[test]
    fn render_models_marks_current_and_filters() {
        let known = vec!["llama3.1:8b".to_string(), "qwen2.5:7b".to_string()];
        let r = render(&SlashCommand::Models(None), &known, Some("qwen2.5:7b"));
        assert!(r.reply.contains("qwen2.5:7b"));
        assert!(r.reply.contains("● (current)"));
        assert!(r.reply.contains("llama3.1:8b"));
        assert_eq!(r.set_model, None);

        let f = render(&SlashCommand::Models(Some("llama".into())), &known, None);
        assert!(f.reply.contains("llama3.1:8b"));
        assert!(!f.reply.contains("qwen2.5:7b"));

        let empty = render(&SlashCommand::Models(None), &[], None);
        assert!(empty.reply.contains("No models"));
    }

    #[test]
    fn render_model_set_validates_against_known() {
        let known = vec!["llama3.1:8b".to_string()];
        let served = render(&SlashCommand::ModelSet("llama3.1:8b".into()), &known, None);
        assert_eq!(served.set_model.as_deref(), Some("llama3.1:8b"));
        assert!(served.reply.contains("Now using"));

        let unserved = render(&SlashCommand::ModelSet("gpt-9".into()), &known, None);
        assert_eq!(unserved.set_model.as_deref(), Some("gpt-9")); // pinned anyway (lag-tolerant)
        assert!(unserved.reply.contains("isn't in the current served list"));

        let auto = render(&SlashCommand::ModelSet("auto".into()), &known, None);
        assert_eq!(auto.set_model.as_deref(), Some("openhydra/auto"));
    }

    #[test]
    fn render_model_show_reflects_pin() {
        let none = render(&SlashCommand::ModelShow, &[], None);
        assert!(none.reply.contains("No model pinned"));
        let pinned = render(&SlashCommand::ModelShow, &[], Some("qwen2.5:7b"));
        assert!(pinned.reply.contains("qwen2.5:7b"));
        // A stored auto pin reads as "no pin".
        let auto = render(&SlashCommand::ModelShow, &[], Some("openhydra/auto"));
        assert!(auto.reply.contains("No model pinned"));
    }

    #[test]
    fn session_store_roundtrips_and_expires() {
        let s = SessionModels::new(Duration::from_secs(3600));
        assert_eq!(s.get("k"), None);
        s.set("k", "qwen2.5:7b".into());
        assert_eq!(s.get("k").as_deref(), Some("qwen2.5:7b"));
        // Distinct keys are isolated.
        assert_eq!(s.get("other"), None);

        // ttl = 0 ⇒ any elapsed time expires the entry on the next read.
        let z = SessionModels::new(Duration::ZERO);
        z.set("k", "m".into());
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(z.get("k"), None);
    }
}
