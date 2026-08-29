// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! llama.cpp (`llama-server`) engine adapter (protocol plan M3.2).
//!
//! llama.cpp's server speaks the OpenAI API too, so *serving* reuses
//! [`crate::adapters::openai::serve_chat_completions`] over `/v1/chat/completions`. The
//! reason this is a bespoke adapter rather than just pointing the OpenAI one at
//! `llama-server` is **detection**: `GET /props` exposes the loaded model's chat template
//! and GGUF path, so — unlike the bare OpenAI `/v1/models` — we can compute a real
//! protocol canonical id (`family/params/quant/template_hash`) from the filename + the
//! live template.
//!
//! Parsing is pure; HTTP is injected via [`HttpClient`](crate::adapter::HttpClient).

use serde::Deserialize;

use openhydra_protocol::model_id::{
    canonical_model_id, chat_template_hash, normalize_chat_template, parse_hf_model_name,
};

use crate::adapter::{
    normalize_engine_ref, AdapterError, DetectedModel, EngineAdapter, HttpClient, InferenceRequest,
    ServeOutcome,
};
use crate::adapters::openai::serve_chat_completions;

/// Default `llama-server` endpoint.
pub const DEFAULT_LLAMACPP_URL: &str = "http://127.0.0.1:8080";

// ── /props (only the fields we use; unknown fields ignored) ──

#[derive(Debug, Default, Deserialize)]
struct Props {
    /// The loaded model's chat template. Absent on very old builds → uncanonicalised.
    #[serde(default)]
    chat_template: String,
    /// On-disk path of the loaded GGUF, e.g. `/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf`.
    #[serde(default)]
    model_path: String,
    /// The quantization file type as `llama-server` reports it, e.g. `"Q4_K - Medium"`,
    /// `"Q8_0"`, `"BF16"`. A reliable quant source even when the filename carries no tag (see
    /// [`quant_from_ftype`]).
    #[serde(default)]
    model_ftype: String,
}

// ── /v1/models (the id(s) a consumer addresses when serving) ──

#[derive(Debug, Default, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Debug, Default, Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: String,
}

/// Does `s` look like a GGUF quant tag (`Q4_K_M`, `Q8_0`, `IQ4_XS`, `F16`, `BF16`)?
fn is_quant_token(s: &str) -> bool {
    let u = s.to_ascii_uppercase();
    let b = u.as_bytes();
    (u.starts_with('Q') && b.len() >= 2 && b[1].is_ascii_digit())
        || (u.starts_with("IQ") && b.len() >= 3 && b[2].is_ascii_digit())
        || matches!(u.as_str(), "F16" | "F32" | "BF16" | "FP16" | "FP32")
}

/// Split a GGUF path into an HF-style name + quant tag for canonicalisation. The quant is
/// the trailing `.`- or `-`-delimited token when it looks like one (`…-Instruct-Q4_K_M`
/// or `…-Instruct.Q8_0`); dotted version/family segments (`Qwen2.5`, `v0.2`) are left in
/// the name. Returns an empty quant for a name with no recognisable tag — the caller then
/// advertises the model uncanonicalised.
fn parse_gguf_path(model_path: &str) -> (String, String) {
    let base = model_path.rsplit(['/', '\\']).next().unwrap_or(model_path);
    let stem = base
        .strip_suffix(".gguf")
        .or_else(|| base.strip_suffix(".GGUF"))
        .unwrap_or(base);
    // The quant is the last dot- or dash-delimited segment, if it matches the pattern.
    if let Some((head, last)) = stem.rsplit_once('.') {
        if is_quant_token(last) {
            return (head.to_string(), last.to_string());
        }
    }
    if let Some((head, last)) = stem.rsplit_once('-') {
        if is_quant_token(last) {
            return (head.to_string(), last.to_string());
        }
    }
    (stem.to_string(), String::new())
}

/// Map `llama-server`'s `/props` `model_ftype` string to a GGUF quant tag, e.g.
/// `"Q4_K - Medium"` → `Q4_K_M`, `"Q5_K - Small"` → `Q5_K_S`, `"Q8_0"` → `Q8_0`,
/// `"IQ4_XS - 4.25 bpw"` → `IQ4_XS`, `"BF16"` → `BF16`. Returns `None` for an empty/unknown
/// ftype so the caller falls back to the filename tag. More reliable than the filename because
/// it comes from the loaded model itself.
fn quant_from_ftype(ftype: &str) -> Option<String> {
    let t = ftype.trim().trim_start_matches("mostly ").trim();
    if t.is_empty() {
        return None;
    }
    // A trailing " - <descriptor>" carries the K-quant size class (Small/Medium/Large); a code
    // with no descriptor is already a full tag.
    let (code, desc) = match t.split_once(" - ") {
        Some((c, d)) => (c.trim(), Some(d.trim().to_ascii_lowercase())),
        None => (t, None),
    };
    if code.is_empty() {
        return None;
    }
    // Quant tags are conventionally upper-case (Q4_K_M, BF16, IQ4_XS); normalise the code so a
    // lower-case ftype can't leak a mixed-case tag into the display/handle.
    let code = code.to_ascii_uppercase();
    if code.ends_with("_K") {
        let suffix = match desc.as_deref() {
            Some(d) if d.starts_with("small") => "_S",
            Some(d) if d.starts_with("medium") => "_M",
            Some(d) if d.starts_with("large") => "_L",
            _ => "", // "Q6_K" (no size class) stays as-is
        };
        return Some(format!("{code}{suffix}"));
    }
    Some(code)
}

/// Does `s` look like an opaque, content-addressed handle rather than a human model name —
/// e.g. an Ollama blob (`sha256-…`) or a bare hex digest? Such a handle carries no model info,
/// so [`detect_models`](LlamaCppAdapter::detect_models) prefers a name synthesised from the GGUF
/// header instead (when the file is co-located and readable).
fn is_opaque_ref(s: &str) -> bool {
    s.starts_with("sha256-")
        || s.starts_with("sha256:")
        || (s.len() >= 32 && s.chars().all(|c| c.is_ascii_hexdigit()))
}

/// A model's self-described identity, read from the GGUF file header's metadata KV store
/// (`general.*`). Independent of the filename — so even a hash-named blob names itself.
#[derive(Debug, Default, Clone)]
struct GgufMeta {
    architecture: String,
    name: String,
    basename: String,
    size_label: String,
}

/// An HF-style model name synthesised from GGUF metadata — `Qwen2.5-0.5B` from `basename` plus
/// `size_label` — for feeding to `parse_hf_model_name`, so the canonical family/params come out
/// normalised the same way the filename path does. Prefers `basename` (clean, no size baked in);
/// falls back to `name` (spaces → dashes). `None` when neither is present.
fn gguf_hf_name(meta: &GgufMeta) -> Option<String> {
    let basename = meta.basename.trim();
    if !basename.is_empty() {
        let size = meta.size_label.trim();
        return Some(if size.is_empty() {
            basename.to_string()
        } else {
            format!("{basename}-{size}")
        });
    }
    let name = meta.name.trim();
    (!name.is_empty()).then(|| name.replace(' ', "-"))
}

/// Synthesise a clean, addressable handle from GGUF metadata — `Qwen3-1.7B-Q4_K_M` from
/// `basename=Qwen3`, `size_label=1.7B`, `quant=Q4_K_M`. Used only to replace an opaque engine
/// id. Prefers `basename` (no size baked in) over `name`; omits a size already present in the
/// base, and the quant when unknown.
fn synth_ref(meta: &GgufMeta, quant: &str) -> Option<String> {
    let base = [meta.basename.as_str(), meta.name.as_str()]
        .into_iter()
        .find(|s| !s.trim().is_empty())?;
    let mut parts = vec![base.trim().replace(' ', "-")];
    if !meta.size_label.is_empty() && !base.contains(&meta.size_label) {
        parts.push(meta.size_label.clone());
    }
    if !quant.is_empty() {
        parts.push(quant.to_string());
    }
    let cleaned = normalize_engine_ref(&parts.join("-"));
    (!cleaned.is_empty()).then_some(cleaned)
}

// ── GGUF header reader (co-located enrichment) ──
//
// Reads the GGUF metadata KV store — a length-prefixed key/type/value stream right after a small
// fixed header — pulling out the `general.*` identity keys and seeking past everything else
// (including the large tokenizer arrays). Every read is fallible and bounded; any short read,
// bad magic, unknown value type, or oversized field yields `None` so a non-co-located or
// unreadable engine simply falls back to the filename. See the GGUF spec (ggml-org/ggml).

fn rd_u32<R: std::io::Read>(r: &mut R) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

fn rd_u64<R: std::io::Read>(r: &mut R) -> Option<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

/// Read a GGUF string (`u64` length + UTF-8 bytes). Rejects an implausibly long metadata string
/// (>1 MiB) so a corrupt/hostile header can't drive a huge allocation.
fn rd_gstr<R: std::io::Read>(r: &mut R) -> Option<String> {
    let n = rd_u64(r)? as usize;
    if n > 1 << 20 {
        return None;
    }
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).ok()?;
    String::from_utf8(buf).ok()
}

/// Seek past a GGUF value of type `vtype` without materialising it (used for every KV we don't
/// want). Handles scalars, strings, and arrays of scalars/strings; bails (`None`) on a nested
/// array or unknown type rather than guessing.
fn skip_gguf_value<R: std::io::Read + std::io::Seek>(r: &mut R, vtype: u32) -> Option<()> {
    use std::io::SeekFrom;
    let scalar = |t: u32| -> Option<i64> {
        Some(match t {
            0 | 1 | 7 => 1,       // u8 / i8 / bool
            2 | 3 => 2,           // u16 / i16
            4..=6 => 4,           // u32 / i32 / f32
            10..=12 => 8,         // u64 / i64 / f64
            _ => return None,
        })
    };
    match vtype {
        8 => {
            let n = rd_u64(r)? as i64;
            r.seek(SeekFrom::Current(n)).ok()?;
        }
        9 => {
            let elem = rd_u32(r)?;
            let count = rd_u64(r)? as i64;
            if elem == 8 {
                for _ in 0..count {
                    let n = rd_u64(r)? as i64;
                    r.seek(SeekFrom::Current(n)).ok()?;
                }
            } else {
                let sz = scalar(elem)?;
                r.seek(SeekFrom::Current(sz.checked_mul(count)?)).ok()?;
            }
        }
        t => {
            r.seek(SeekFrom::Current(scalar(t)?)).ok()?;
        }
    }
    Some(())
}

/// Best-effort read of the `general.*` identity from a local GGUF file. `None` if the path isn't
/// readable (a remote engine), isn't a GGUF, or the header can't be parsed — the caller then
/// uses the filename. Only the small header prefix is touched; large arrays are seeked over.
fn read_gguf_metadata(path: &str) -> Option<GgufMeta> {
    let mut r = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    let mut magic = [0u8; 4];
    std::io::Read::read_exact(&mut r, &mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let version = rd_u32(&mut r)?;
    if !(2..=3).contains(&version) {
        return None; // v1 used u32 counts; only v2/v3 are handled
    }
    let _tensor_count = rd_u64(&mut r)?;
    let kv_count = rd_u64(&mut r)?;
    if kv_count > 100_000 {
        return None; // sanity bound on a corrupt header
    }
    let mut meta = GgufMeta::default();
    let mut got = 0;
    for _ in 0..kv_count {
        let key = rd_gstr(&mut r)?;
        let vtype = rd_u32(&mut r)?;
        let slot = match key.as_str() {
            "general.architecture" => Some(&mut meta.architecture),
            "general.name" => Some(&mut meta.name),
            "general.basename" => Some(&mut meta.basename),
            "general.size_label" => Some(&mut meta.size_label),
            _ => None,
        };
        match slot {
            Some(dst) if vtype == 8 => {
                *dst = rd_gstr(&mut r)?;
                got += 1;
                if got >= 4 {
                    break; // have every key we want; stop before the big tensor/tokenizer KVs
                }
            }
            _ => skip_gguf_value(&mut r, vtype)?,
        }
    }
    if meta.architecture.is_empty() && meta.name.is_empty() && meta.basename.is_empty() {
        return None; // nothing useful
    }
    Some(meta)
}

/// Adapter for a local `llama-server`, generic over the injected HTTP transport.
pub struct LlamaCppAdapter<H: HttpClient> {
    base_url: String,
    http: H,
}

impl<H: HttpClient> LlamaCppAdapter<H> {
    /// New adapter against `base_url`, e.g. [`DEFAULT_LLAMACPP_URL`]. A trailing `/v1` is
    /// stripped so callers may pass either the root or the OpenAI base URL.
    pub fn new(base_url: impl Into<String>, http: H) -> Self {
        let root = base_url
            .into()
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .trim_end_matches('/')
            .to_string();
        Self { base_url: root, http }
    }
}

impl<H: HttpClient> EngineAdapter for LlamaCppAdapter<H> {
    fn engine_name(&self) -> &'static str {
        "llama.cpp"
    }

    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        // /props → the loaded model's template + GGUF path (the canonicalisation source).
        let props_json = self.http.get(&format!("{}/props", self.base_url))?;
        let props: Props =
            serde_json::from_str(&props_json).map_err(|e| AdapterError::Parse(e.to_string()))?;

        // Model identity — prefer the model's own GGUF header (co-located, filename-independent)
        // over parsing the path, and the engine-reported `model_ftype` over the filename tag for
        // the quant. The filename remains the fallback for a remote/unreadable engine.
        let gguf = read_gguf_metadata(&props.model_path);
        let (hf_name, fn_quant) = parse_gguf_path(&props.model_path);
        let (fn_family, fn_params, _variants) = parse_hf_model_name(&hf_name);

        // Family/params: trust the filename when it parsed a size (`params != "unknown"`) — this
        // preserves every existing canonical id exactly. Only when the filename is unparseable (a
        // hash-named blob, or a name with no size token) fall back to the GGUF header, synthesised
        // into an HF-style name and run through the *same* `parse_hf_model_name`, so the family
        // ("Qwen2.5") and the rounded params come out normalised identically to the filename path.
        // `general.architecture` is deliberately NOT used as the family: it's the coarse
        // architecture (`qwen2` for Qwen2.5, `llama` for the Llama-3.x/distill cluster), which
        // would both change existing ids and collide genuinely distinct models.
        let (family, params) = if fn_params != "unknown" {
            (fn_family, fn_params)
        } else if let Some((f, p, _)) = gguf.as_ref().and_then(gguf_hf_name).map(|n| parse_hf_model_name(&n)) {
            (f, p)
        } else {
            (fn_family, fn_params)
        };
        let quant = quant_from_ftype(&props.model_ftype).unwrap_or(fn_quant);

        // Canonicalise from the resolved components. `parse_hf_model_name` yields
        // `params == "unknown"` when it finds no size token; refuse a misleading `…/unknown/…`
        // id. A missing template or quant also leaves the model advertised uncanonicalised.
        let has_template = !normalize_chat_template(&props.chat_template).is_empty();
        let canonical_id = if !has_template
            || family.is_empty()
            || params.is_empty()
            || params == "unknown"
            || quant.is_empty()
        {
            String::new()
        } else {
            let th = chat_template_hash(&props.chat_template);
            match canonical_model_id(&family, &params, &quant, &th) {
                Ok(c) => format!("{}/{}/{}/{}", c.family, c.params, c.quant, c.template_hash),
                Err(_) => String::new(),
            }
        };

        // /v1/models → the id(s) a consumer addresses when serving. `llama-server` usually
        // serves one model; map each id, sharing the /props template (single-model case).
        let models_json = self.http.get(&format!("{}/v1/models", self.base_url))?;
        let models: ModelsResponse =
            serde_json::from_str(&models_json).map_err(|e| AdapterError::Parse(e.to_string()))?;

        // A clean handle synthesised from GGUF metadata, used only to replace an *opaque* engine id
        // (e.g. an Ollama blob's `sha256-…`); `None` for a remote/unreadable engine. The `/props`
        // GGUF describes a single model, so only substitute it when the server lists exactly one
        // model — otherwise two distinct opaque ids would both collapse to this one name and one
        // would be dropped as a collision.
        let gguf_ref = gguf.as_ref().and_then(|m| synth_ref(m, &quant));
        let single_model = models.data.len() == 1;
        let deopaque = |clean: String| -> String {
            if is_opaque_ref(&clean) {
                gguf_ref.clone().unwrap_or(clean)
            } else {
                clean
            }
        };

        // Advertise a clean, path-free handle — never the raw `-m` path (privacy + readability),
        // and never an opaque blob hash when the GGUF header offers a real name (single-model only).
        // Skip an id that cleans to empty (a pathological `"/"` or bare `".gguf"`), and drop a
        // basename collision (two GGUFs sharing a name in different dirs) with a log rather than
        // silently — a single-model server never hits either, they guard the rare multi-model case.
        let mut refs: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for id in models.data.into_iter().map(|m| m.id) {
            if id.trim().is_empty() {
                continue;
            }
            let clean = normalize_engine_ref(&id);
            if clean.is_empty() {
                continue;
            }
            // Only de-opaque a single-model server, so distinct opaque ids stay distinct.
            let engine_ref = if single_model { deopaque(clean) } else { clean };
            if !seen.insert(engine_ref.clone()) {
                eprintln!(
                    "openhydra-agent: llama.cpp model '{id}' collides with already-detected '{engine_ref}' — skipping"
                );
                continue;
            }
            refs.push(engine_ref);
        }
        // Fall back to the GGUF basename if /v1/models reported nothing addressable. This is
        // inherently the single-model case, so de-opaque unconditionally here.
        if refs.is_empty() && !props.model_path.is_empty() {
            let clean = normalize_engine_ref(&props.model_path);
            if !clean.is_empty() {
                refs.push(deopaque(clean));
            }
        }

        Ok(refs
            .into_iter()
            .map(|engine_ref| DetectedModel {
                engine_ref,
                canonical_id: canonical_id.clone(),
                family: family.clone(),
                params: params.clone(),
                quant: quant.clone(),
                size_bytes: 0,
            })
            .collect())
    }

    fn serve_stream(
        &self,
        request: &InferenceRequest,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeOutcome, AdapterError> {
        // llama-server speaks the OpenAI chat route — reuse the shared SSE serve.
        serve_chat_completions(
            &self.http,
            &format!("{}/v1/chat/completions", self.base_url),
            request,
            on_delta,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ChatMessage;

    const QWEN_TEMPLATE: &str = "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n<|im_start|>assistant\n";

    fn props_fixture() -> String {
        serde_json::json!({
            "model_path": "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "chat_template": QWEN_TEMPLATE,
            "total_slots": 1,
        })
        .to_string()
    }

    const MODELS_FIXTURE: &str =
        r#"{"object":"list","data":[{"id":"Qwen2.5-7B-Instruct-Q4_K_M","object":"model"}]}"#;

    #[derive(Default)]
    struct MockHttp {
        props: String,
        models: String,
        stream_lines: Vec<String>,
    }

    impl HttpClient for MockHttp {
        fn get(&self, url: &str) -> Result<String, AdapterError> {
            if url.ends_with("/props") {
                Ok(self.props.clone())
            } else if url.ends_with("/v1/models") {
                Ok(self.models.clone())
            } else {
                Err(AdapterError::Http(format!("unexpected GET {url}")))
            }
        }
        fn post_json(&self, url: &str, _body: &str) -> Result<String, AdapterError> {
            Err(AdapterError::Http(format!("unexpected POST {url}")))
        }
        fn post_stream(
            &self,
            url: &str,
            _body: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            assert!(url.ends_with("/v1/chat/completions"), "stream url: {url}");
            let lines: Vec<Result<String, AdapterError>> =
                self.stream_lines.iter().cloned().map(Ok).collect();
            Ok(Box::new(lines.into_iter()))
        }
    }

    #[test]
    fn parse_gguf_path_extracts_name_and_quant() {
        assert_eq!(
            parse_gguf_path("/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
            ("Qwen2.5-7B-Instruct".into(), "Q4_K_M".into())
        );
        // Dot-delimited quant; dotted family/version segments stay in the name.
        assert_eq!(
            parse_gguf_path("Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"),
            ("Meta-Llama-3.1-8B-Instruct".into(), "Q8_0".into())
        );
        // No recognisable quant tag → empty quant.
        assert_eq!(parse_gguf_path("model.gguf"), ("model".into(), String::new()));
    }

    #[test]
    fn clean_engine_ref_strips_paths_and_extension() {
        // The kastru case: an absolute path id → basename without .gguf (no home dir / username).
        assert_eq!(
            normalize_engine_ref("/home/kastru/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"),
            "Qwen3.5-9B-UD-Q4_K_XL"
        );
        // Windows-style separator + uppercase extension.
        assert_eq!(normalize_engine_ref(r"C:\models\Llama-3.1-8B.GGUF"), "Llama-3.1-8B");
        // Bare filename with extension (no directory).
        assert_eq!(normalize_engine_ref("mistral-7b-Q4_K_M.gguf"), "mistral-7b-Q4_K_M");
        // Already-clean alias → untouched (no path chars, no .gguf).
        assert_eq!(normalize_engine_ref("Qwen2.5-7B-Instruct-Q4_K_M"), "Qwen2.5-7B-Instruct-Q4_K_M");
        // Ollama-style tag → untouched (the colon is not a path separator).
        assert_eq!(normalize_engine_ref("llama3.2:1b"), "llama3.2:1b");
        // HF-style namespaced id has a slash but is NOT a path → untouched (don't drop the org).
        assert_eq!(normalize_engine_ref("Qwen/Qwen2.5-7B-Instruct"), "Qwen/Qwen2.5-7B-Instruct");
        // Home-relative path → basename.
        assert_eq!(normalize_engine_ref("~/models/phi-3-mini.gguf"), "phi-3-mini");
        // Mixed-case extension is stripped case-insensitively (parity with the JS displayModelName).
        assert_eq!(normalize_engine_ref("/models/Phi-3.GGuf"), "Phi-3");
        assert_eq!(normalize_engine_ref("model.GgUf"), "model");
        // Pathological ids that clean to empty (detect_models then skips these).
        assert_eq!(normalize_engine_ref("/"), "");
        assert_eq!(normalize_engine_ref(".gguf"), "");
    }

    #[test]
    fn detect_never_advertises_a_filesystem_path() {
        // A llama-server that reports its model id AS the launch path must not leak that path onto
        // the network — the advertised engine_ref is the clean basename.
        let http = MockHttp {
            props: serde_json::json!({
                "model_path": "/home/kastru/models/Qwen3.5-9B-UD-Q4_K_XL.gguf",
                "chat_template": QWEN_TEMPLATE,
                "total_slots": 1,
            })
            .to_string(),
            models: r#"{"object":"list","data":[{"id":"/home/kastru/models/Qwen3.5-9B-UD-Q4_K_XL.gguf"}]}"#.into(),
            ..Default::default()
        };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].engine_ref, "Qwen3.5-9B-UD-Q4_K_XL");
        assert!(!models[0].engine_ref.contains('/'), "must not advertise a path");
        assert!(!models[0].engine_ref.contains("kastru"), "must not leak the username");
    }

    #[test]
    fn detect_canonicalises_from_props_template_and_gguf_name() {
        let http = MockHttp {
            props: props_fixture(),
            models: MODELS_FIXTURE.into(),
            ..Default::default()
        };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].engine_ref, "Qwen2.5-7B-Instruct-Q4_K_M");
        // Real canonical id: family/params from the GGUF name, quant normalised, template
        // hashed from /props.
        assert!(
            models[0].canonical_id.starts_with("qwen2.5/7b/int4/"),
            "canonical_id = {}",
            models[0].canonical_id
        );
        assert_eq!(models[0].family, "qwen2.5");
        assert_eq!(models[0].params, "7b");
        assert_eq!(models[0].quant, "Q4_K_M");
        assert_eq!(adapter.engine_name(), "llama.cpp");
    }

    #[test]
    fn detect_uncanonicalised_without_a_template() {
        // No chat_template in /props → still detected and addressable, just no canonical id.
        let props = serde_json::json!({
            "model_path": "/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        })
        .to_string();
        let http = MockHttp { props, models: MODELS_FIXTURE.into(), ..Default::default() };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models.len(), 1);
        assert!(models[0].canonical_id.is_empty());
        assert_eq!(models[0].engine_ref, "Qwen2.5-7B-Instruct-Q4_K_M");
    }

    /// A GGUF string (`u64` length + UTF-8) for the fixture builder.
    fn gstr(s: &str) -> Vec<u8> {
        let b = s.as_bytes();
        let mut v = (b.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(b);
        v
    }

    /// A minimal GGUF v3 header: a string-array KV and a scalar KV (both must be *skipped*) then
    /// the four `general.*` identity strings. Exercises the reader's skip paths.
    fn build_gguf(arch: &str, name: &str, basename: &str, size_label: &str) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&6u64.to_le_bytes()); // kv_count: array + scalar + 4 general
        // array-of-strings (like tokenizer.ggml.tokens) → must be seeked past
        buf.extend_from_slice(&gstr("tokenizer.ggml.tokens"));
        buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        buf.extend_from_slice(&8u32.to_le_bytes()); // of STRING
        buf.extend_from_slice(&2u64.to_le_bytes()); // count 2
        buf.extend_from_slice(&gstr("<a>"));
        buf.extend_from_slice(&gstr("<bb>"));
        // scalar u32 → must be seeked past
        buf.extend_from_slice(&gstr("some.count"));
        buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        buf.extend_from_slice(&123u32.to_le_bytes());
        for (k, v) in [
            ("general.architecture", arch),
            ("general.name", name),
            ("general.basename", basename),
            ("general.size_label", size_label),
        ] {
            buf.extend_from_slice(&gstr(k));
            buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
            buf.extend_from_slice(&gstr(v));
        }
        buf
    }

    fn write_gguf(arch: &str, name: &str, basename: &str, size: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&build_gguf(arch, name, basename, size)).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn quant_from_ftype_maps_llama_server_strings() {
        assert_eq!(quant_from_ftype("Q4_K - Medium").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_from_ftype("Q5_K - Small").as_deref(), Some("Q5_K_S"));
        assert_eq!(quant_from_ftype("Q6_K").as_deref(), Some("Q6_K")); // no size class
        assert_eq!(quant_from_ftype("Q8_0").as_deref(), Some("Q8_0"));
        assert_eq!(quant_from_ftype("IQ4_XS - 4.25 bpw").as_deref(), Some("IQ4_XS"));
        assert_eq!(quant_from_ftype("BF16").as_deref(), Some("BF16"));
        assert_eq!(quant_from_ftype("mostly Q4_K - Medium").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_from_ftype(""), None);
        assert_eq!(quant_from_ftype("   "), None);
    }

    #[test]
    fn read_gguf_metadata_extracts_general_keys_and_skips_arrays() {
        let f = write_gguf("qwen3", "Qwen3 1.7B", "Qwen3", "1.7B");
        let meta = read_gguf_metadata(f.path().to_str().unwrap()).expect("parses");
        assert_eq!(meta.architecture, "qwen3");
        assert_eq!(meta.name, "Qwen3 1.7B");
        assert_eq!(meta.basename, "Qwen3");
        assert_eq!(meta.size_label, "1.7B");
        // A missing / non-GGUF file → None so the caller falls back to the filename.
        assert!(read_gguf_metadata("/no/such/file.gguf").is_none());
    }

    #[test]
    fn detect_names_a_hash_blob_from_gguf_header_and_ftype() {
        // The exact case that surfaced the ugly id: llama-server launched on an Ollama
        // content-addressed blob → model_path + /v1/models id are the opaque `sha256-…`, and the
        // quant lives only in `model_ftype`. The GGUF header + ftype recover a real identity.
        let f = write_gguf("qwen3", "Qwen3 1.7B", "Qwen3", "1.7B");
        let props = serde_json::json!({
            "model_path": f.path().to_str().unwrap(),
            "chat_template": QWEN_TEMPLATE,
            "model_ftype": "Q4_K - Medium",
            "total_slots": 1,
        })
        .to_string();
        let http = MockHttp {
            props,
            models: r#"{"object":"list","data":[{"id":"sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"}]}"#.into(),
            ..Default::default()
        };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models.len(), 1);
        // Engine handle synthesised from GGUF metadata instead of the opaque hash.
        assert_eq!(models[0].engine_ref, "Qwen3-1.7B-Q4_K_M");
        assert!(!models[0].engine_ref.starts_with("sha256"), "opaque blob hash must be replaced");
        // Canonical id from GGUF architecture/size + ftype quant + the live template.
        assert!(
            models[0].canonical_id.starts_with("qwen3/1.7b/int4/"),
            "canonical_id = {}",
            models[0].canonical_id
        );
        assert_eq!(models[0].family, "qwen3");
        assert_eq!(models[0].params, "1.7b");
        assert_eq!(models[0].quant, "Q4_K_M");
    }

    #[test]
    fn detect_uses_gguf_basename_not_architecture_for_family() {
        // GGUF `general.architecture` is the coarse arch — "qwen2" for a Qwen2.5 model. The
        // canonical family MUST come from `general.basename` ("Qwen2.5"), else distinct models
        // collapse (Qwen2 vs Qwen2.5 → both "qwen2") and existing ids silently change. Regression
        // guard for the gap the other tests miss (they use a non-existent path or arch==family).
        let f = write_gguf("qwen2", "Qwen2.5 0.5B Instruct", "Qwen2.5", "0.5B");
        let props = serde_json::json!({
            "model_path": f.path().to_str().unwrap(), // temp name → filename unparseable → GGUF fallback
            "chat_template": QWEN_TEMPLATE,
            "model_ftype": "Q4_K - Medium",
        })
        .to_string();
        let http = MockHttp {
            props,
            models: r#"{"object":"list","data":[{"id":"sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515"}]}"#.into(),
            ..Default::default()
        };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models[0].family, "qwen2.5", "family from basename, NOT architecture 'qwen2'");
        assert_eq!(models[0].params, "0.5b");
        assert!(
            models[0].canonical_id.starts_with("qwen2.5/0.5b/int4/"),
            "canonical_id = {}",
            models[0].canonical_id
        );
        assert_eq!(models[0].engine_ref, "Qwen2.5-0.5B-Q4_K_M");
    }

    #[test]
    fn detect_keeps_distinct_opaque_ids_for_a_multi_model_server() {
        // Two opaque ids on one server must not both collapse to the single GGUF-synthesised name
        // (the /props GGUF describes one model). Both models must survive.
        let f = write_gguf("qwen3", "Qwen3 1.7B", "Qwen3", "1.7B");
        let props = serde_json::json!({
            "model_path": f.path().to_str().unwrap(),
            "chat_template": QWEN_TEMPLATE,
            "model_ftype": "Q4_K - Medium",
        })
        .to_string();
        let http = MockHttp {
            props,
            models: r#"{"object":"list","data":[{"id":"sha256-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},{"id":"sha256-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]}"#.into(),
            ..Default::default()
        };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models.len(), 2, "both distinct opaque models survive (no collapse)");
    }

    #[test]
    fn detect_respects_a_clean_engine_id_over_gguf_synthesis() {
        // A non-opaque id (an operator `--alias`) is authoritative for the handle; GGUF only
        // enriches the canonical id, it does not override a real name the operator chose.
        let f = write_gguf("qwen3", "Qwen3 1.7B", "Qwen3", "1.7B");
        let props = serde_json::json!({
            "model_path": f.path().to_str().unwrap(),
            "chat_template": QWEN_TEMPLATE,
            "model_ftype": "Q4_K - Medium",
        })
        .to_string();
        let http = MockHttp {
            props,
            models: r#"{"object":"list","data":[{"id":"qwen3:1.7b"}]}"#.into(),
            ..Default::default()
        };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let models = adapter.detect_models().unwrap();
        assert_eq!(models[0].engine_ref, "qwen3:1.7b", "operator alias is kept");
        assert!(models[0].canonical_id.starts_with("qwen3/1.7b/int4/"));
    }

    #[test]
    fn serve_delegates_to_the_openai_chat_route() {
        let http = MockHttp {
            stream_lines: vec![
                r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#.into(),
                r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}"#.into(),
                "data: [DONE]".into(),
            ],
            ..Default::default()
        };
        let adapter = LlamaCppAdapter::new(DEFAULT_LLAMACPP_URL, http);
        let mut out = String::new();
        let outcome = adapter
            .serve_stream(
                &InferenceRequest {
                    model_ref: "Qwen2.5-7B-Instruct-Q4_K_M".into(),
                    messages: vec![ChatMessage { role: "user".into(), content: "hi".into(), ..Default::default() }],
                    max_tokens: None,
                    temperature: None,
                    tools: Vec::new(),
                    think: None,
                },
                &mut |d| out.push_str(d),
            )
            .unwrap();
        assert_eq!(out, "hi");
        assert_eq!(outcome.tokens, 1);
        assert!(outcome.done);
    }
}
