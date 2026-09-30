// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Teacher-forced prompt scoring against the engines that expose it (Track A2-Logprob).
//!
//! Each engine names the same measurement differently; every dialect here returns the log-prob of
//! each actual next token (`token_ids[i + 1]` given `token_ids[..=i]`), so one fewer value than ids,
//! and checks the engine scored exactly the ids it was sent:
//!
//! * **vLLM** — `POST /v1/completions` with the ids as `prompt`, `max_tokens: 1`,
//!   `prompt_logprobs: 1`; each position's entry is keyed by token id.
//! * **SGLang** — native `POST /generate` with `input_ids`, `return_logprob`,
//!   `logprob_start_len: 0`; `meta_info.input_token_logprobs` is `[logprob, id, text]` per position.
//! * **The OpenHydra scoring API** — `POST /v1/prompt_logprobs` with `prompt_token_ids`, answered by
//!   the OpenHydra patches for mistral.rs (on `2370966`) and llama.cpp's `llama-server` (on
//!   `a25c986`, started with `--prompt-scoring`); `scores.logprobs` is already in this shape.
//!
//! Ollama, LM Studio and stock llama-server expose generation log-probs only, not prompt log-probs,
//! so they can't answer.

use std::str::FromStr;
use std::sync::Mutex;

use serde_json::{json, Value};

use crate::adapter::{AdapterError, HttpClient};

/// Which prompt-scoring API the engine behind an OpenAI-compatible adapter speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringDialect {
    Vllm,
    Sglang,
    /// `POST /v1/prompt_logprobs` — the OpenHydra patches for mistral.rs and llama.cpp.
    OpenHydra,
}

impl FromStr for ScoringDialect {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "vllm" => Ok(Self::Vllm),
            "sglang" => Ok(Self::Sglang),
            "openhydra" | "mistralrs" | "mistral.rs" | "mistral-rs" | "llamacpp" | "llama.cpp" | "llama-cpp" => {
                Ok(Self::OpenHydra)
            }
            other => Err(format!(
                "unknown prompt-scoring dialect '{other}' (expected vllm, sglang, or openhydra — the \
                 mistral.rs / llama.cpp scoring patches, also accepted as mistralrs / llamacpp)"
            )),
        }
    }
}

/// What `--prompt-scoring` asked for: a named API, or `auto` — identify the engine by a route only
/// it has ([`fingerprint`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringChoice {
    Auto,
    Dialect(ScoringDialect),
}

impl FromStr for ScoringChoice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        s.parse().map(Self::Dialect).map_err(|e: String| e.replace("(expected ", "(expected auto, "))
    }
}

/// Identify which prompt-scoring API the OpenAI-compatible engine at `root` speaks, from a route
/// only that engine has. Checked against the installed sources and live servers (2026-09-30):
///
/// * **The OpenHydra scoring API** (patched mistral.rs, or patched llama-server started with
///   `--prompt-scoring`): `GET /v1/prompt_logprobs` → 405 (the route exists, POST only); unpatched
///   builds and other engines → 404.
/// * **SGLang** (0.5.20): `GET /server_info` → a JSON object with `launch_command`.
/// * **vLLM** (0.30): `GET /version` → `{"version": …}`. SGLang has no `/version`; it's checked first
///   anyway.
///
/// `None` when nothing matches (Ollama, LM Studio, stock llama-server: none expose prompt log-probs).
pub fn fingerprint<H: HttpClient>(http: &H, root: &str) -> Option<ScoringDialect> {
    if http.get_status(&format!("{root}/v1/prompt_logprobs")).ok() == Some(405) {
        return Some(ScoringDialect::OpenHydra);
    }
    let json = |path: &str| -> Option<Value> { serde_json::from_str(&http.get(&format!("{root}{path}")).ok()?).ok() };
    if json("/server_info").is_some_and(|v| v.get("launch_command").is_some()) {
        return Some(ScoringDialect::Sglang);
    }
    if json("/version").is_some_and(|v| v.get("version").is_some_and(Value::is_string)) {
        return Some(ScoringDialect::Vllm);
    }
    None
}

/// How an adapter knows which prompt-scoring API its engine speaks (`provide --prompt-scoring`).
#[derive(Debug, Default)]
pub enum ScoringConfig {
    /// Scoring off: challenges are refused.
    #[default]
    Off,
    /// Declared by the operator.
    Fixed(ScoringDialect),
    /// Fingerprinted on first use and cached; the cache is cleared when scoring fails, so a different
    /// engine taking over the port is re-identified on the next challenge.
    Auto(Mutex<Option<ScoringDialect>>),
}

impl ScoringConfig {
    /// Identify the engine on first use ([`fingerprint`]).
    pub fn auto() -> Self {
        Self::Auto(Mutex::new(None))
    }

    /// Score `token_ids` on the engine at `root` (labelled `engine` in errors) as configured.
    pub fn score<H: HttpClient>(
        &self,
        http: &H,
        root: &str,
        engine: &str,
        model_ref: &str,
        token_ids: &[u32],
    ) -> Result<Vec<f32>, AdapterError> {
        match self {
            Self::Off => Err(AdapterError::Unsupported(format!("prompt scoring not enabled for {engine}"))),
            Self::Fixed(dialect) => score_prompt(http, root, *dialect, model_ref, token_ids),
            Self::Auto(cache) => {
                let cached = *cache.lock().unwrap_or_else(|e| e.into_inner());
                let Some(dialect) = cached.or_else(|| fingerprint(http, root)) else {
                    return Err(AdapterError::Unsupported(format!(
                        "no prompt-scoring API recognised at {root} ({engine})"
                    )));
                };
                let result = score_prompt(http, root, dialect, model_ref, token_ids);
                *cache.lock().unwrap_or_else(|e| e.into_inner()) = result.is_ok().then_some(dialect);
                result
            }
        }
    }
}

/// Score `token_ids` on the engine at `root` (server root, no `/v1`).
pub fn score_prompt<H: HttpClient>(
    http: &H,
    root: &str,
    dialect: ScoringDialect,
    model_ref: &str,
    token_ids: &[u32],
) -> Result<Vec<f32>, AdapterError> {
    if token_ids.len() < 2 {
        return Err(AdapterError::Parse("need at least two tokens to score".into()));
    }
    let (path, body) = request(dialect, model_ref, token_ids);
    let reply = http.post_json(&format!("{root}{path}"), &body.to_string())?;
    let json: Value = serde_json::from_str(&reply).map_err(|e| AdapterError::Parse(e.to_string()))?;
    parse(dialect, &json, token_ids)
}

fn request(dialect: ScoringDialect, model_ref: &str, ids: &[u32]) -> (&'static str, Value) {
    match dialect {
        ScoringDialect::Vllm => (
            "/v1/completions",
            json!({"model": model_ref, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}),
        ),
        ScoringDialect::Sglang => (
            "/generate",
            json!({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                   "return_logprob": true, "logprob_start_len": 0, "top_logprobs_num": 1}),
        ),
        ScoringDialect::OpenHydra => ("/v1/prompt_logprobs", json!({"model": model_ref, "prompt_token_ids": ids})),
    }
}

fn parse(dialect: ScoringDialect, json: &Value, ids: &[u32]) -> Result<Vec<f32>, AdapterError> {
    let bad = |m: &str| AdapterError::Parse(format!("prompt scores: {m}"));
    let num = |v: &Value| v.as_f64().map(|x| x as f32).ok_or_else(|| bad("non-numeric log-prob"));
    match dialect {
        ScoringDialect::Vllm => {
            let pl = json
                .pointer("/choices/0/prompt_logprobs")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("missing choices[0].prompt_logprobs"))?;
            if pl.len() != ids.len() {
                return Err(bad("position count differs from the ids sent"));
            }
            pl.iter()
                .zip(ids)
                .skip(1)
                .map(|(entry, id)| {
                    entry
                        .get(id.to_string())
                        .and_then(|e| e.get("logprob"))
                        .ok_or_else(|| bad("a position lacks the actual token"))
                        .and_then(num)
                })
                .collect()
        }
        ScoringDialect::Sglang => {
            let itl = json
                .pointer("/meta_info/input_token_logprobs")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("missing meta_info.input_token_logprobs"))?;
            if itl.len() != ids.len() {
                return Err(bad("position count differs from the ids sent"));
            }
            itl.iter()
                .zip(ids)
                .skip(1)
                .map(|(entry, &id)| {
                    if entry.get(1).and_then(Value::as_u64) != Some(u64::from(id)) {
                        return Err(bad("token alignment"));
                    }
                    entry.get(0).ok_or_else(|| bad("missing log-prob")).and_then(num)
                })
                .collect()
        }
        ScoringDialect::OpenHydra => {
            let scores = json.get("scores").ok_or_else(|| bad("missing scores"))?;
            let echoed: Option<Vec<u64>> =
                scores.get("token_ids").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect());
            if echoed.as_deref() != Some(&ids.iter().map(|&i| u64::from(i)).collect::<Vec<_>>()[..]) {
                return Err(bad("token alignment"));
            }
            let lps = scores.get("logprobs").and_then(Value::as_array).ok_or_else(|| bad("missing logprobs"))?;
            if lps.len() + 1 != ids.len() {
                return Err(bad("position count differs from the ids sent"));
            }
            lps.iter().map(num).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Canned {
        reply: String,
        seen: RefCell<Vec<(String, Value)>>,
    }

    /// Answers GETs from a route table: `(path, status, body)`; anything else → 404.
    struct Routes(Vec<(&'static str, u16, &'static str)>);

    impl HttpClient for Routes {
        fn get(&self, url: &str) -> Result<String, AdapterError> {
            match self.0.iter().find(|(p, _, _)| url.ends_with(p)) {
                Some((_, 200, body)) => Ok(body.to_string()),
                Some((_, code, _)) => Err(AdapterError::Http(format!("status {code}"))),
                None => Err(AdapterError::Http("status 404".into())),
            }
        }
        fn get_status(&self, url: &str) -> Result<u16, AdapterError> {
            Ok(self.0.iter().find(|(p, _, _)| url.ends_with(p)).map_or(404, |r| r.1))
        }
        fn post_json(&self, _: &str, _: &str) -> Result<String, AdapterError> {
            unreachable!()
        }
        fn post_stream(
            &self,
            _: &str,
            _: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            unreachable!()
        }
    }

    #[test]
    fn engines_are_fingerprinted_by_a_route_only_they_have() {
        let fp = |r: Vec<(&'static str, u16, &'static str)>| fingerprint(&Routes(r), "http://h:1");
        // Patched mistral.rs (as observed live: 405 on GET, and no /version or /server_info).
        assert_eq!(fp(vec![("/v1/prompt_logprobs", 405, ""), ("/health", 200, "OK")]), Some(ScoringDialect::OpenHydra));
        assert_eq!(
            fp(vec![("/server_info", 200, r#"{"launch_command":"python -m sglang.launch_server","version":"0.5.20"}"#)]),
            Some(ScoringDialect::Sglang)
        );
        assert_eq!(fp(vec![("/version", 200, r#"{"version":"0.30.0"}"#)]), Some(ScoringDialect::Vllm));
        // Unpatched mistral.rs, LM Studio, llama-server …: nothing matches.
        assert_eq!(fp(vec![("/health", 200, "OK"), ("/v1/models", 200, "{}")]), None);
        // Look-alikes don't match: a /version that isn't vLLM's shape, a /server_info without the field.
        assert_eq!(fp(vec![("/version", 200, r#"{"v":1}"#), ("/server_info", 200, r#"{"x":1}"#)]), None);
    }

    /// A patched llama-server: 405 on GET of the scoring route, scores on POST.
    struct PatchedLlamaServer;

    impl HttpClient for PatchedLlamaServer {
        fn get(&self, _url: &str) -> Result<String, AdapterError> {
            Err(AdapterError::Http("404".into()))
        }
        fn get_status(&self, url: &str) -> Result<u16, AdapterError> {
            Ok(if url.ends_with("/v1/prompt_logprobs") { 405 } else { 404 })
        }
        fn post_json(&self, url: &str, body: &str) -> Result<String, AdapterError> {
            assert!(url.ends_with("/v1/prompt_logprobs"));
            let ids: Value = serde_json::from_str::<Value>(body).unwrap()["prompt_token_ids"].clone();
            Ok(json!({"model": "Qwen3.5-9B-Q4_K_M.gguf", "scores": {"token_ids": ids, "logprobs": [-0.25, -2.0]}}).to_string())
        }
        fn post_stream(
            &self,
            _: &str,
            _: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            unreachable!()
        }
    }

    #[test]
    fn auto_config_finds_and_uses_the_openhydra_endpoint() {
        let cfg = ScoringConfig::auto();
        assert_eq!(cfg.score(&PatchedLlamaServer, "http://h:8080", "llama.cpp", "m", &[1, 2, 3]), Ok(vec![-0.25, -2.0]));
        assert!(matches!(&cfg, ScoringConfig::Auto(c) if *c.lock().unwrap() == Some(ScoringDialect::OpenHydra)));
        assert!(matches!(
            ScoringConfig::Off.score(&PatchedLlamaServer, "http://h:8080", "llama.cpp", "m", &[1, 2, 3]),
            Err(AdapterError::Unsupported(_))
        ));
    }

    #[test]
    fn scoring_choice_parses_auto_or_a_dialect() {
        assert_eq!("auto".parse(), Ok(ScoringChoice::Auto));
        assert_eq!("sglang".parse(), Ok(ScoringChoice::Dialect(ScoringDialect::Sglang)));
        assert!("ollama".parse::<ScoringChoice>().unwrap_err().contains("expected auto, vllm"));
    }

    impl HttpClient for Canned {
        fn get(&self, _url: &str) -> Result<String, AdapterError> {
            unreachable!()
        }
        fn post_json(&self, url: &str, body: &str) -> Result<String, AdapterError> {
            self.seen.borrow_mut().push((url.to_string(), serde_json::from_str(body).unwrap()));
            Ok(self.reply.clone())
        }
        fn post_stream(
            &self,
            _url: &str,
            _body: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            unreachable!()
        }
    }

    fn run(dialect: ScoringDialect, reply: Value, ids: &[u32]) -> (Result<Vec<f32>, AdapterError>, String, Value) {
        let http = Canned { reply: reply.to_string(), seen: RefCell::new(vec![]) };
        let out = score_prompt(&http, "http://h:1", dialect, "m", ids);
        let (url, body) = http.seen.into_inner().pop().unwrap();
        (out, url, body)
    }

    #[test]
    fn vllm_reads_the_actual_token_at_each_position() {
        let reply = json!({"choices": [{"prompt_logprobs": [
            null,
            {"7": {"logprob": -0.5, "rank": 2}, "9": {"logprob": -0.1, "rank": 1}},
            {"8": {"logprob": -1.25, "rank": 1}}
        ]}]});
        let (out, url, body) = run(ScoringDialect::Vllm, reply, &[3, 7, 8]);
        assert_eq!(out, Ok(vec![-0.5, -1.25]));
        assert_eq!(url, "http://h:1/v1/completions");
        assert_eq!(body["prompt"], json!([3, 7, 8]));
        assert_eq!(body["prompt_logprobs"], 1);
        assert_eq!(body["model"], "m");
    }

    #[test]
    fn vllm_rejects_a_missing_actual_token_or_a_length_mismatch() {
        let reply = json!({"choices": [{"prompt_logprobs": [null, {"9": {"logprob": -0.1}}]}]});
        assert!(matches!(run(ScoringDialect::Vllm, reply, &[3, 7]).0, Err(AdapterError::Parse(_))));
        let reply = json!({"choices": [{"prompt_logprobs": [null]}]});
        assert!(matches!(run(ScoringDialect::Vllm, reply, &[3, 7]).0, Err(AdapterError::Parse(_))));
    }

    #[test]
    fn sglang_checks_alignment_and_skips_the_first_position() {
        let reply = json!({"meta_info": {"input_token_logprobs": [[null, 3, null], [-0.5, 7, null], [-1.25, 8, null]]}});
        let (out, url, body) = run(ScoringDialect::Sglang, reply, &[3, 7, 8]);
        assert_eq!(out, Ok(vec![-0.5, -1.25]));
        assert_eq!(url, "http://h:1/generate");
        assert_eq!(body["input_ids"], json!([3, 7, 8]));
        assert_eq!(body["logprob_start_len"], 0);
        let misaligned = json!({"meta_info": {"input_token_logprobs": [[null, 3, null], [-0.5, 6, null]]}});
        assert!(matches!(run(ScoringDialect::Sglang, misaligned, &[3, 7]).0, Err(AdapterError::Parse(_))));
    }

    #[test]
    fn openhydra_api_requires_the_echoed_ids_to_match() {
        let reply = json!({"model": "m", "scores": {"token_ids": [3, 7, 8], "logprobs": [-0.5, -1.25]}});
        let (out, url, body) = run(ScoringDialect::OpenHydra, reply, &[3, 7, 8]);
        assert_eq!(out, Ok(vec![-0.5, -1.25]));
        assert_eq!(url, "http://h:1/v1/prompt_logprobs");
        assert_eq!(body["prompt_token_ids"], json!([3, 7, 8]));
        let other = json!({"scores": {"token_ids": [3, 7, 9], "logprobs": [-0.5, -1.25]}});
        assert!(matches!(run(ScoringDialect::OpenHydra, other, &[3, 7, 8]).0, Err(AdapterError::Parse(_))));
    }

    #[test]
    fn non_numeric_log_probs_are_a_parse_error() {
        let reply = json!({"scores": {"token_ids": [3, 7], "logprobs": ["x"]}});
        assert!(matches!(run(ScoringDialect::OpenHydra, reply, &[3, 7]).0, Err(AdapterError::Parse(_))));
    }

    #[test]
    fn dialect_names_parse() {
        assert_eq!("vLLM".parse(), Ok(ScoringDialect::Vllm));
        assert_eq!("sglang".parse(), Ok(ScoringDialect::Sglang));
        assert_eq!("mistral.rs".parse(), Ok(ScoringDialect::OpenHydra));
        assert_eq!("llamacpp".parse(), Ok(ScoringDialect::OpenHydra));
        assert_eq!("openhydra".parse(), Ok(ScoringDialect::OpenHydra));
        assert!("ollama".parse::<ScoringDialect>().is_err());
    }
}
