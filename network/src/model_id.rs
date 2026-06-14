// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Canonical model identity & equivalence (protocol.md §4).
//!
//! OpenHydra's equivalent of a BitTorrent infohash. The same "model" is not the
//! same everywhere: `Llama-3-8B` quantised Q4_K_M produces different outputs than
//! fp16, and two engines can disagree on the chat template even at the same
//! weights+quant. Routing a request to an *incompatible* provider silently
//! degrades quality, so the protocol routes on a **canonical id**:
//!
//! ```text
//! model_id = {family}/{params}/{quantization}/{chat_template_hash}
//! //  e.g.  qwen3.5/2b/fp16/9f2c8a1b4d6e0f37
//! ```
//!
//! * `family`        — model family, lower-cased (`qwen3.5`, `gemma-4`).
//! * `params`        — parameter scale, incl. MoE active params (`35b-a3b`).
//! * `quant`         — the quantisation **actually loaded by the provider at
//!   runtime**, *not* the catalog's recommended default. Equivalence is about what
//!   is served, not what is suggested.
//! * `template_hash` — a stable hash of the tokenizer's chat template, so two
//!   providers that agree on weights+quant but differ in chat template are treated
//!   as *incompatible* (the safe default).
//!
//! This module is additive: existing user-facing ids (`openhydra-qwen3.5-2b`)
//! remain valid aliases; the canonical id is an *additional* field providers
//! advertise and the router matches on.
//!
//! The `family`/`params` split is derived from a model's HuggingFace id via a
//! documented heuristic ([`parse_hf_model_name`]) that is tested against every
//! entry shipped in `models.catalog.json`. Curated catalogs may carry explicit
//! `family`/`params` fields that override the heuristic (the governance path).

use sha2::{Digest, Sha256};
use std::fmt;

/// Length (hex chars) of the truncated chat-template hash. 16 hex = 64 bits —
/// ample collision resistance for template equivalence while keeping the id short
/// and human-eyeballable. Versioned: changing this is a wire-format change.
pub const TEMPLATE_HASH_LEN: usize = 16;

/// Wildcard component in a *request* id: a consumer may name a partial id
/// (`qwen3.5/2b/*/*`) and let the router pick a compatible quant/template.
pub const WILDCARD: &str = "*";

/// A parsed canonical model id: `{family}/{params}/{quant}/{template_hash}`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalModelId {
    pub family: String,
    pub params: String,
    pub quant: String,
    pub template_hash: String,
}

/// Errors from constructing or parsing a [`CanonicalModelId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelIdError {
    /// Not a `family/params/quant/template_hash` string (wrong arity / empty part).
    Malformed(String),
    /// A required component was empty after normalisation.
    EmptyComponent(&'static str),
    /// A canonical id was requested without a usable (non-empty) chat template.
    MissingTemplate,
}

impl fmt::Display for ModelIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelIdError::Malformed(s) => write!(
                f,
                "not a canonical model id (need family/params/quant/template_hash): {s:?}"
            ),
            ModelIdError::EmptyComponent(c) => write!(f, "canonical model id has empty {c}"),
            ModelIdError::MissingTemplate => {
                write!(f, "canonical model id requires a non-empty chat template")
            }
        }
    }
}

impl std::error::Error for ModelIdError {}

impl CanonicalModelId {
    /// Parse a `family/params/quant/template_hash` string.
    ///
    /// Components may be the wildcard `*` (used in request ids). Returns
    /// [`ModelIdError::Malformed`] if the id does not have exactly four non-empty
    /// components.
    pub fn parse(value: &str) -> Result<Self, ModelIdError> {
        let parts: Vec<&str> = value.trim().split('/').collect();
        if parts.len() != 4 || parts.iter().any(|p| p.is_empty()) {
            return Err(ModelIdError::Malformed(value.to_string()));
        }
        Ok(CanonicalModelId {
            family: parts[0].to_string(),
            params: parts[1].to_string(),
            quant: parts[2].to_string(),
            template_hash: parts[3].to_string(),
        })
    }

    /// True if no component is a wildcard (a fully-specified provider id).
    pub fn is_concrete(&self) -> bool {
        [&self.family, &self.params, &self.quant, &self.template_hash]
            .iter()
            .all(|c| c.as_str() != WILDCARD)
    }
}

impl fmt::Display for CanonicalModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{}/{}/{}",
            self.family, self.params, self.quant, self.template_hash
        )
    }
}

/// Normalise a chat template before hashing.
///
/// Chat templates are Jinja strings where interior whitespace can be semantically
/// significant, so we deliberately do **not** collapse it. We only normalise line
/// endings and strip leading/trailing whitespace — differences that arise from
/// editors/transport, not template meaning. Anything beyond that hashes as a
/// *different* template, the safe (incompatible-by-default) direction.
pub fn normalize_chat_template(template: &str) -> String {
    template
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .trim()
        .to_string()
}

/// Stable short hash of a tokenizer chat template.
///
/// Returns [`TEMPLATE_HASH_LEN`] hex chars of SHA-256 over the normalised template.
pub fn chat_template_hash(template: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(normalize_chat_template(template).as_bytes());
    let digest = hasher.finalize();
    let mut hex = hex::encode(digest);
    hex.truncate(TEMPLATE_HASH_LEN);
    hex
}

/// Canonicalise a quantisation label (`Q4_K_M` → `int4`, `float16` → `fp16`, …).
///
/// Unknown labels are lower-cased and trimmed but otherwise preserved, so a novel
/// quant still produces a stable (if un-folded) component rather than being dropped.
pub fn normalize_quant(quant: &str) -> String {
    let q = quant.trim().to_lowercase();
    match q.as_str() {
        "fp32" | "float32" | "f32" => return "fp32".to_string(),
        "fp16" | "float16" | "f16" | "half" => return "fp16".to_string(),
        "bf16" | "bfloat16" => return "bf16".to_string(),
        "fp8" | "float8" => return "fp8".to_string(),
        "int8" | "q8" | "8bit" => return "int8".to_string(),
        "int4" | "q4" | "4bit" => return "int4".to_string(),
        _ => {}
    }
    // Fold common GGUF-style spellings: "q4_k_m" → "int4", "q8_0" → "int8".
    if let Some(rest) = q.strip_prefix('q') {
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        match digits.as_str() {
            "4" => return "int4".to_string(),
            "8" => return "int8".to_string(),
            _ => {}
        }
    }
    q
}

/// A params token: an optional single leading letter (gemma's `E2B`/`E4B`
/// "effective params"), a number (possibly decimal), and a `B`/`M` unit.
/// Matches `35B`, `0.5B`, `360M`, `15M`, `E2B`; rejects a bare `4` (a family
/// version segment like gemma-`4`) and `FP8` (no digits after the letter).
fn is_params_token(seg: &str) -> bool {
    let b = seg.as_bytes();
    if b.is_empty() {
        return false;
    }
    let mut i = 0;
    if b[i].is_ascii_alphabetic() {
        i += 1;
    }
    let digit_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == digit_start {
        return false; // need at least one digit
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return false; // dot with no following digits
        }
    }
    // exactly one trailing B/M unit and nothing after it
    if i < b.len() && matches!(b[i], b'B' | b'b' | b'M' | b'm') {
        i += 1;
        return i == b.len();
    }
    false
}

/// A trailing MoE active-params token that belongs *with* the params, e.g. `A3B`
/// in `Qwen3.5-35B-A3B` (35B total, 3B active).
fn is_moe_active(seg: &str) -> bool {
    let b = seg.as_bytes();
    if b.len() < 3 {
        return false;
    }
    if !matches!(b[0], b'A' | b'a') {
        return false;
    }
    let mut i = 1;
    let digit_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == digit_start {
        return false;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return false;
        }
    }
    i + 1 == b.len() && matches!(b[i], b'B' | b'b')
}

/// Heuristically split an HF model id into `(family, params, variants)`.
///
/// Rules (grounded in and tested against `models.catalog.json`):
///
/// * Take the name after the org prefix (`Qwen/Qwen3.5-2B` → `Qwen3.5-2B`).
/// * Split on `-`. The first segment matching [`is_params_token`] is the params;
///   everything before it is the family (joined by `-`, lower-cased).
/// * A segment immediately after params matching [`is_moe_active`] (`A3B`) is
///   folded into params as `35b-a3b`.
/// * Remaining trailing segments are returned as `variants` (`instruct`, `it`,
///   `fp8` …) — informational.
///
/// If no params token is found, `params` is `"unknown"` and the whole name becomes
/// the family — a signal that the entry needs an explicit catalog `params` field.
pub fn parse_hf_model_name(hf_model_id: &str) -> (String, String, Vec<String>) {
    let raw = hf_model_id.trim();
    let name = raw.rsplit('/').next().unwrap_or(raw);
    if name.is_empty() {
        return (String::new(), "unknown".to_string(), Vec::new());
    }

    let segments: Vec<&str> = name.split('-').collect();
    let params_idx = segments.iter().position(|s| is_params_token(s));

    let Some(idx) = params_idx else {
        return (name.to_lowercase(), "unknown".to_string(), Vec::new());
    };

    let mut family = segments[..idx].join("-").to_lowercase();
    let mut params = segments[idx].to_lowercase();
    let mut tail = idx + 1;
    if tail < segments.len() && is_moe_active(segments[tail]) {
        params.push('-');
        params.push_str(&segments[tail].to_lowercase());
        tail += 1;
    }
    let variants: Vec<String> = segments[tail..].iter().map(|s| s.to_lowercase()).collect();

    if family.is_empty() {
        // Params token was the first segment (no family prefix); fall back to the
        // bare name so we never emit an empty family component.
        family = name.to_lowercase();
    }
    (family, params, variants)
}

/// Assemble a [`CanonicalModelId`] from components plus a precomputed template hash.
///
/// `family`/`params` are lower-cased; `quant` is normalised. Returns an error if
/// any component is empty.
pub fn canonical_model_id(
    family: &str,
    params: &str,
    quant: &str,
    template_hash: &str,
) -> Result<CanonicalModelId, ModelIdError> {
    let family = family.trim().to_lowercase();
    let params = params.trim().to_lowercase();
    let quant = normalize_quant(quant);
    let template_hash = template_hash.trim().to_string();
    if family.is_empty() {
        return Err(ModelIdError::EmptyComponent("family"));
    }
    if params.is_empty() {
        return Err(ModelIdError::EmptyComponent("params"));
    }
    if quant.is_empty() {
        return Err(ModelIdError::EmptyComponent("quant"));
    }
    if template_hash.is_empty() {
        return Err(ModelIdError::MissingTemplate);
    }
    Ok(CanonicalModelId {
        family,
        params,
        quant,
        template_hash,
    })
}

/// Compute the canonical id for a model from its HF id, runtime quant, and the
/// engine's live chat template.
///
/// `family`/`params` are parsed from `hf_model_id`; `quant` and the template are
/// runtime-supplied (they reflect what the provider actually loaded). Errors if the
/// template is empty after normalisation.
pub fn canonical_id_from_hf(
    hf_model_id: &str,
    quant: &str,
    chat_template: &str,
) -> Result<CanonicalModelId, ModelIdError> {
    if normalize_chat_template(chat_template).is_empty() {
        return Err(ModelIdError::MissingTemplate);
    }
    let (family, params, _variants) = parse_hf_model_name(hf_model_id);
    let template_hash = chat_template_hash(chat_template);
    canonical_model_id(&family, &params, quant, &template_hash)
}

/// Is a `provider`'s concrete canonical id compatible with a `request`?
///
/// The `request` may use the wildcard `*` in any component (`qwen3.5/2b/*/*`) to
/// let the router pick a compatible quant/template (protocol.md §4). The `provider`
/// id must be concrete (no wildcards). Compatibility holds when, for every
/// component, the request is `*` or exactly equals the provider's. Malformed ids
/// return `false` rather than erroring — the router treats an unparseable id as
/// "no match" and moves on.
pub fn is_compatible(request: &str, provider: &str) -> bool {
    let (Ok(req), Ok(prov)) = (
        CanonicalModelId::parse(request),
        CanonicalModelId::parse(provider),
    ) else {
        return false;
    };
    if !prov.is_concrete() {
        return false;
    }
    let pairs = [
        (req.family.as_str(), prov.family.as_str()),
        (req.params.as_str(), prov.params.as_str()),
        (req.quant.as_str(), prov.quant.as_str()),
        (req.template_hash.as_str(), prov.template_hash.as_str()),
    ];
    pairs.iter().all(|(r, p)| *r == WILDCARD || r == p)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden vectors: expected `(family, params)` for every `hf_model_id` shipped
    /// in `models.catalog.json`. This is the contract the heuristic parser must
    /// satisfy; a new catalog entry whose name doesn't parse fails here loudly and
    /// gets an explicit `family`/`params` field (the governance fallback).
    const HF_PARSE: &[(&str, &str, &str)] = &[
        ("Qwen/Qwen3.5-35B-A3B-FP8", "qwen3.5", "35b-a3b"),
        ("Qwen/Qwen3.5-27B-FP8", "qwen3.5", "27b"),
        ("Qwen/Qwen3.5-2B", "qwen3.5", "2b"),
        ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5", "7b"),
        ("Qwen/Qwen3-8B", "qwen3", "8b"),
        ("Qwen/Qwen2.5-0.5B", "qwen2.5", "0.5b"),
        ("google/gemma-4-E2B", "gemma-4", "e2b"),
        ("google/gemma-4-E4B", "gemma-4", "e4b"),
        ("google/gemma-4-E4B-it", "gemma-4", "e4b"),
        ("Qwen/Qwen3.5-0.8B", "qwen3.5", "0.8b"),
        ("Qwen/Qwen3.5-4B", "qwen3.5", "4b"),
        ("Qwen/Qwen3.5-9B", "qwen3.5", "9b"),
        ("Qwen/Qwen3.5-27B", "qwen3.5", "27b"),
        ("HuggingFaceTB/SmolLM2-360M", "smollm2", "360m"),
        ("google/gemma-3-270m", "gemma-3", "270m"),
        ("nickypro/tinyllama-15M", "tinyllama", "15m"),
    ];

    #[test]
    fn parse_hf_model_name_golden_table() {
        for (hf, family, params) in HF_PARSE {
            let (f, p, _) = parse_hf_model_name(hf);
            assert_eq!((f.as_str(), p.as_str()), (*family, *params), "hf={hf}");
        }
    }

    #[test]
    fn parse_hf_model_name_variants() {
        assert_eq!(
            parse_hf_model_name("Qwen/Qwen2.5-7B-Instruct").2,
            vec!["instruct"]
        );
        assert_eq!(parse_hf_model_name("google/gemma-4-E4B-it").2, vec!["it"]);
        assert_eq!(
            parse_hf_model_name("Qwen/Qwen3.5-35B-A3B-FP8").2,
            vec!["fp8"]
        );
    }

    #[test]
    fn parse_hf_model_name_no_params_token() {
        let (family, params, _) = parse_hf_model_name("acme/MysteryModel");
        assert_eq!(family, "mysterymodel");
        assert_eq!(params, "unknown");
    }

    #[test]
    fn chat_template_hash_deterministic_and_sized() {
        let tpl = "{{ bos }}{% for m in messages %}{{ m.content }}{% endfor %}";
        let h1 = chat_template_hash(tpl);
        assert_eq!(h1, chat_template_hash(tpl));
        assert_eq!(h1.len(), TEMPLATE_HASH_LEN);
        assert!(h1
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn chat_template_hash_differs_on_meaningful_change() {
        assert_ne!(
            chat_template_hash("{{ messages[0].content }}"),
            chat_template_hash("{{ messages[0].content }} EXTRA")
        );
    }

    #[test]
    fn chat_template_hash_ignores_line_endings_and_edge_whitespace() {
        assert_eq!(
            chat_template_hash("line one\nline two"),
            chat_template_hash("  line one\r\nline two  ")
        );
    }

    #[test]
    fn normalize_chat_template_preserves_interior_whitespace() {
        assert_eq!(normalize_chat_template("a   b"), "a   b");
    }

    #[test]
    fn normalize_quant_table() {
        for (raw, expected) in [
            ("fp16", "fp16"),
            ("FP16", "fp16"),
            ("float16", "fp16"),
            ("bf16", "bf16"),
            ("bfloat16", "bf16"),
            ("fp8", "fp8"),
            ("int4", "int4"),
            ("q4", "int4"),
            ("Q4_K_M", "int4"),
            ("q8_0", "int8"),
            ("8bit", "int8"),
            ("some-novel-quant", "some-novel-quant"),
        ] {
            assert_eq!(normalize_quant(raw), expected, "raw={raw}");
        }
    }

    #[test]
    fn canonical_id_display_and_parse_roundtrip() {
        let cid = canonical_model_id("Qwen3.5", "2B", "fp16", "0000000000000000").unwrap();
        assert_eq!(cid.to_string(), "qwen3.5/2b/fp16/0000000000000000");
        assert_eq!(CanonicalModelId::parse(&cid.to_string()).unwrap(), cid);
    }

    #[test]
    fn canonical_id_rejects_empty_components_and_template() {
        assert!(matches!(
            canonical_model_id("", "2b", "fp16", "abc"),
            Err(ModelIdError::EmptyComponent("family"))
        ));
        assert!(matches!(
            canonical_model_id("qwen3.5", "2b", "fp16", ""),
            Err(ModelIdError::MissingTemplate)
        ));
    }

    #[test]
    fn canonical_id_from_hf_requires_template() {
        assert!(matches!(
            canonical_id_from_hf("Qwen/Qwen3.5-2B", "fp16", "   "),
            Err(ModelIdError::MissingTemplate)
        ));
    }

    #[test]
    fn parse_rejects_malformed() {
        for bad in ["", "a/b/c", "a/b/c/d/e", "qwen3.5//fp16/abc"] {
            assert!(CanonicalModelId::parse(bad).is_err(), "bad={bad:?}");
        }
    }

    #[test]
    fn canonical_id_from_hf_uses_runtime_quant() {
        // Entry 1's HF name says FP8; the canonical quant must reflect what the
        // provider actually loaded at runtime (here, fp8), parsed family/params.
        let cid = canonical_id_from_hf("Qwen/Qwen3.5-35B-A3B-FP8", "fp8", "tpl-v1").unwrap();
        assert_eq!(cid.family, "qwen3.5");
        assert_eq!(cid.params, "35b-a3b");
        assert_eq!(cid.quant, "fp8");
        assert_eq!(cid.template_hash, chat_template_hash("tpl-v1"));
    }

    // --- the equivalence exit-test (protocol.md §4 headline) ---

    #[test]
    fn equivalence_same_weights_and_template_match() {
        let tpl = "the-shared-template";
        let a = canonical_id_from_hf("Qwen/Qwen3.5-2B", "fp16", tpl).unwrap();
        let b = canonical_id_from_hf("Qwen/Qwen3.5-2B", "fp16", tpl).unwrap();
        assert_eq!(a, b);
        assert!(is_compatible(&a.to_string(), &b.to_string()));
    }

    #[test]
    fn equivalence_different_template_is_incompatible() {
        let a = canonical_id_from_hf("Qwen/Qwen3.5-2B", "fp16", "template-A").unwrap();
        let b = canonical_id_from_hf("Qwen/Qwen3.5-2B", "fp16", "template-B").unwrap();
        assert_ne!(a.template_hash, b.template_hash);
        assert!(!is_compatible(&a.to_string(), &b.to_string()));
    }

    #[test]
    fn equivalence_different_quant_is_incompatible() {
        let a = canonical_id_from_hf("Qwen/Qwen3.5-2B", "fp16", "t").unwrap();
        let b = canonical_id_from_hf("Qwen/Qwen3.5-2B", "int4", "t").unwrap();
        assert!(!is_compatible(&a.to_string(), &b.to_string()));
    }

    #[test]
    fn request_wildcards_match_any_provider_component() {
        let prov = "qwen3.5/2b/fp16/abcabcabcabcabca";
        assert!(is_compatible("qwen3.5/2b/*/*", prov));
        assert!(is_compatible("qwen3.5/*/*/*", prov));
        assert!(is_compatible("*/*/*/*", prov));
        assert!(!is_compatible("qwen3/2b/*/*", prov)); // family must match
        assert!(!is_compatible("qwen3.5/9b/*/*", prov)); // params must match
    }

    #[test]
    fn provider_id_must_be_concrete() {
        assert!(!is_compatible(
            "qwen3.5/2b/fp16/abcabcabcabcabca",
            "qwen3.5/2b/*/*"
        ));
    }

    #[test]
    fn is_compatible_malformed_returns_false() {
        assert!(!is_compatible(
            "garbage",
            "qwen3.5/2b/fp16/abcabcabcabcabca"
        ));
        assert!(!is_compatible(
            "qwen3.5/2b/fp16/abcabcabcabcabca",
            "garbage"
        ));
    }
}
