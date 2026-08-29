// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! ComfyUI engine adapter — the network's first **image** engine.
//!
//! ComfyUI (default `:8188`) executes node-graph workflows. This adapter maps OpenHydra's
//! chat-shaped serve protocol onto a minimal txt2img graph:
//!
//! * **Detection**: `GET /object_info/CheckpointLoaderSimple` lists the installed
//!   checkpoints — each becomes a [`DetectedModel`] (`engine_ref` = checkpoint filename),
//!   so Stable-Diffusion checkpoints are announced/routed exactly like LLM models.
//! * **Serving**: the last user message is the positive prompt. The adapter queues a
//!   standard 7-node workflow (`POST /prompt`), polls `GET /history/{id}` until the
//!   graph finishes, fetches the PNG (`GET /view`, binary), and emits it as ONE delta:
//!   a markdown image with a base64 data-URL — so the gateway stays OpenAI-shaped, any
//!   chat client renders/receives it as message content, and the buffered one-shot
//!   transport (100 MB frames) carries it fine.
//! * **Receipts**: image gen has no tokens; [`ServeOutcome::tokens`] = **sampler steps**
//!   (the honest compute proxy — proportional to GPU work like tokens are for LLMs).
//!
//! Knobs (v1, fixed): 512×512, 20 steps, cfg 7, euler/normal, empty negative prompt, and
//! a seed derived from the prompt so identical requests are reproducible (useful for the
//! M2.2(b) redundant-execution audit, which compares outputs across providers).
//!
//! Parsing is pure; HTTP is injected via [`HttpClient`](crate::adapter::HttpClient).

use base64::Engine as _;
use std::path::Path;

use crate::adapter::{
    AdapterError, DetectedModel, EngineAdapter, EngineMetrics, HttpClient, InferenceRequest,
    ServeOutcome,
};

/// Default ComfyUI endpoint.
pub const DEFAULT_COMFYUI_URL: &str = "http://127.0.0.1:8188";

/// Fixed v1 sampling knobs.
const STEPS: u64 = 20;
const CFG: f64 = 7.0;
const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
/// Generation poll cadence + cap (SD on CPU can be minutes; a wedged graph must not
/// hang the serve worker forever).
const POLL_MS: u64 = 500;
const POLL_MAX_SECS: u64 = 300;

/// A provider-supplied ComfyUI workflow (API-format JSON) with `%prompt%` / `%seed%` /
/// `%negative%` markers. The filename (sans `.json`) is the advertised model id. This is
/// what makes the adapter model-agnostic: ComfyUI already speaks every model via a graph,
/// so the operator brings the graph (BYO-workflow) and we just inject the prompt.
#[derive(Clone, Debug)]
pub struct WorkflowTemplate {
    /// Advertised model id (the template filename without `.json`).
    pub model_id: String,
    /// Raw template text, markers intact.
    pub json: String,
    /// Steps parsed from the template (for step-as-tokens billing); [`STEPS`] if absent.
    pub steps: u64,
}

/// Adapter for a local ComfyUI, generic over the injected HTTP transport.
pub struct ComfyUiAdapter<H: HttpClient> {
    base_url: String,
    http: H,
    /// Empty ⇒ built-in SD txt2img graph + checkpoint detection (back-compat). Non-empty ⇒
    /// BYO-workflow mode: advertise these templates, inject the prompt, submit as-is.
    templates: Vec<WorkflowTemplate>,
}

impl<H: HttpClient> ComfyUiAdapter<H> {
    /// New adapter against `base_url`, e.g. [`DEFAULT_COMFYUI_URL`]. Built-in SD graph mode.
    pub fn new(base_url: impl Into<String>, http: H) -> Self {
        Self { base_url: base_url.into().trim_end_matches('/').to_string(), http, templates: Vec::new() }
    }

    /// BYO-workflow mode: serve provider-supplied workflow templates instead of the built-in
    /// SD graph. Any ComfyUI-supported model (Flux, SDXL, video, upscale chains) works with
    /// zero code change — just drop a template file.
    pub fn with_templates(
        base_url: impl Into<String>,
        http: H,
        templates: Vec<WorkflowTemplate>,
    ) -> Self {
        Self { base_url: base_url.into().trim_end_matches('/').to_string(), http, templates }
    }
}

/// Substitute the request tokens into a workflow template and parse the result. Pure.
/// `"%seed%"` (a *quoted* placeholder) is de-quoted to the numeric seed; `%prompt%` /
/// `%negative%` are JSON-escaped and spliced into their string values. Keying on the marker
/// (not node type) is what makes this work for SD, Flux, or any other graph.
fn inject_template(
    tpl: &str,
    prompt: &str,
    negative: &str,
    seed: u64,
) -> Result<serde_json::Value, AdapterError> {
    // JSON-escape without the surrounding quotes (the marker already sits inside a "…").
    let esc = |s: &str| -> String {
        let j = serde_json::Value::String(s.to_string()).to_string();
        j[1..j.len() - 1].to_string()
    };
    let injected = tpl
        .replace("\"%seed%\"", &seed.to_string())
        .replace("%prompt%", &esc(prompt))
        .replace("%negative%", &esc(negative));
    serde_json::from_str(&injected)
        .map_err(|e| AdapterError::Parse(format!("workflow invalid after injection: {e}")))
}

/// Best-effort step count from a raw template (first node input named `steps`), for
/// step-as-tokens billing. Falls back to [`STEPS`]. Pure.
fn parse_template_steps(tpl: &str) -> u64 {
    serde_json::from_str::<serde_json::Value>(tpl)
        .ok()
        .and_then(|v| {
            v.as_object().and_then(|o| {
                o.values().find_map(|n| {
                    n.get("inputs").and_then(|i| i.get("steps")).and_then(|s| s.as_u64())
                })
            })
        })
        .unwrap_or(STEPS)
}

/// Load provider-supplied workflow templates from a directory (`*.json`). File I/O is kept
/// out of the adapter (which stays HTTP-injected / unit-testable); the caller hands the
/// result to [`ComfyUiAdapter::with_templates`]. Each file must contain a `%prompt%` marker.
pub fn load_workflow_templates(dir: &Path) -> Result<Vec<WorkflowTemplate>, AdapterError> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| AdapterError::Http(format!("workflow dir {}: {e}", dir.display())))?;
    for entry in entries {
        let path = entry.map_err(|e| AdapterError::Http(e.to_string()))?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let json = std::fs::read_to_string(&path)
            .map_err(|e| AdapterError::Http(format!("read {}: {e}", path.display())))?;
        if !json.contains("%prompt%") {
            return Err(AdapterError::Parse(format!(
                "workflow {} has no %prompt% marker",
                path.display()
            )));
        }
        let model_id =
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("workflow").to_string();
        let steps = parse_template_steps(&json);
        out.push(WorkflowTemplate { model_id, json, steps });
    }
    out.sort_by(|a, b| a.model_id.cmp(&b.model_id));
    Ok(out)
}

/// The minimal txt2img workflow graph, in ComfyUI's API format (`/prompt` body's
/// `prompt` field): checkpoint → CLIP encode (pos/neg) → empty latent → KSampler →
/// VAE decode → save. Pure → unit-tested.
fn build_workflow(ckpt: &str, prompt: &str, seed: u64) -> serde_json::Value {
    serde_json::json!({
        "4": { "class_type": "CheckpointLoaderSimple",
               "inputs": { "ckpt_name": ckpt } },
        "5": { "class_type": "EmptyLatentImage",
               "inputs": { "width": WIDTH, "height": HEIGHT, "batch_size": 1 } },
        "6": { "class_type": "CLIPTextEncode",
               "inputs": { "text": prompt, "clip": ["4", 1] } },
        "7": { "class_type": "CLIPTextEncode",
               "inputs": { "text": "", "clip": ["4", 1] } },
        "3": { "class_type": "KSampler",
               "inputs": { "seed": seed, "steps": STEPS, "cfg": CFG,
                           "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                           "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                           "latent_image": ["5", 0] } },
        "8": { "class_type": "VAEDecode",
               "inputs": { "samples": ["3", 0], "vae": ["4", 2] } },
        "9": { "class_type": "SaveImage",
               "inputs": { "filename_prefix": "openhydra", "images": ["8", 0] } },
    })
}

/// Deterministic seed from the prompt (identical request → identical image), so the
/// M2.2(b) dual-dispatch audit can meaningfully compare providers.
fn seed_from_prompt(prompt: &str) -> u64 {
    // FNV-1a — tiny, dependency-free, stable across platforms.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in prompt.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Extract the checkpoint list from `/object_info/CheckpointLoaderSimple`. Pure.
fn parse_checkpoints(json: &str) -> Result<Vec<String>, AdapterError> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| AdapterError::Parse(e.to_string()))?;
    // Shape: {"CheckpointLoaderSimple":{"input":{"required":{"ckpt_name":[[names...],...]}}}}
    let names = v["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
        .as_array()
        .ok_or_else(|| AdapterError::Parse("no ckpt_name list in object_info".into()))?;
    Ok(names.iter().filter_map(|n| n.as_str().map(String::from)).collect())
}

/// Output keys ComfyUI uses across modalities: images (SaveImage), gifs/videos
/// (VHS_VideoCombine / SaveVideo), audio (SaveAudio). Checked in this order per node.
const MEDIA_KEYS: [&str; 4] = ["images", "gifs", "videos", "audio"];

/// Extract the first output media file's `(filename, subfolder, type)` from a
/// `/history/{id}` response — image, video, OR audio — or `None` while the graph is still
/// running. Media-agnostic: keys on the output arrays, not the modality. Pure.
fn parse_history_media(json: &str, prompt_id: &str) -> Option<(String, String, String)> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let outputs = &v[prompt_id]["outputs"];
    for (_node, out) in outputs.as_object()? {
        for key in MEDIA_KEYS {
            if let Some(arr) = out[key].as_array() {
                if let Some(m) = arr.first() {
                    return Some((
                        m["filename"].as_str()?.to_string(),
                        m["subfolder"].as_str().unwrap_or("").to_string(),
                        m["type"].as_str().unwrap_or("output").to_string(),
                    ));
                }
            }
        }
    }
    None
}

/// MIME type from a ComfyUI output filename's extension, for the returned data-URL. Pure.
fn mime_for(name: &str) -> &'static str {
    match name.rsplit('.').next().unwrap_or("").to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mkv" => "video/x-matroska",
        "flac" => "audio/flac",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "ogg" => "audio/ogg",
        _ => "application/octet-stream",
    }
}

/// Minimal query-string escaper for the /view params (filenames are engine-generated,
/// but escape anyway).
fn qs(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

impl<H: HttpClient> EngineAdapter for ComfyUiAdapter<H> {
    fn engine_name(&self) -> &'static str {
        "comfyui"
    }

    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        // BYO-workflow: advertise the provider's templates by filename (no dependency on
        // CheckpointLoaderSimple, so Flux — which isn't a single checkpoint — is detectable).
        if !self.templates.is_empty() {
            return Ok(self
                .templates
                .iter()
                .map(|t| DetectedModel {
                    engine_ref: t.model_id.clone(),
                    // Distinct canonical id per template so multiple workflows on one
                    // provider don't collide to the same DHT record (empty → collision).
                    canonical_id: t.model_id.clone(),
                    family: "comfyui-workflow".into(),
                    params: String::new(),
                    quant: String::new(),
                    size_bytes: 0,
                })
                .collect());
        }
        let json =
            self.http.get(&format!("{}/object_info/CheckpointLoaderSimple", self.base_url))?;
        Ok(parse_checkpoints(&json)?
            .into_iter()
            .map(|ckpt| DetectedModel {
                engine_ref: ckpt,
                // Diffusion checkpoints have no chat template / LLM params — advertised
                // uncanonicalised, addressed by filename (like the OpenAI-family models).
                canonical_id: String::new(),
                family: "stable-diffusion".into(),
                params: String::new(),
                quant: String::new(),
                size_bytes: 0,
            })
            .collect())
    }

    fn serve_stream(
        &self,
        request: &InferenceRequest,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ServeOutcome, AdapterError> {
        // The prompt = the last user message (system/history don't map to txt2img).
        let prompt = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or_default();
        if prompt.trim().is_empty() {
            return Err(AdapterError::Http("empty image prompt".into()));
        }

        let started = std::time::Instant::now();
        let seed = seed_from_prompt(prompt);
        // Pick the provider's workflow template for this model; else the built-in SD graph.
        let (workflow, bill_steps) =
            match self.templates.iter().find(|t| t.model_id == request.model_ref) {
                Some(t) => (inject_template(&t.json, prompt, "", seed)?, t.steps),
                None if self.templates.is_empty() => {
                    (build_workflow(&request.model_ref, prompt, seed), STEPS)
                }
                None => {
                    return Err(AdapterError::Http(format!(
                        "no workflow template for model '{}'",
                        request.model_ref
                    )))
                }
            };
        let body = serde_json::json!({ "prompt": workflow, "client_id": "openhydra" });
        let resp = self.http.post_json(&format!("{}/prompt", self.base_url), &body.to_string())?;
        let resp: serde_json::Value =
            serde_json::from_str(&resp).map_err(|e| AdapterError::Parse(e.to_string()))?;
        let prompt_id = resp["prompt_id"]
            .as_str()
            .ok_or_else(|| {
                AdapterError::Parse(format!("no prompt_id in /prompt response: {resp}"))
            })?
            .to_string();

        // Poll history until the graph reports outputs (bounded). A transient /history
        // blip must NOT abort an otherwise-healthy multi-minute render: retry until the
        // deadline. The deadline is checked at the TOP of each iteration so a slow GET
        // can't bypass it, and the GET is now bounded by the client's idle read timeout
        // (A1). A *persistent* failure (ComfyUI died) gives up after MAX_POLL_ERRORS
        // consecutive errors so we don't wait the full POLL_MAX_SECS on a dead engine.
        // See docs/CODEBASE_HARDENING_PLAN.md (A4).
        const MAX_POLL_ERRORS: u32 = 10;
        let deadline = started + std::time::Duration::from_secs(POLL_MAX_SECS);
        let mut consecutive_errs: u32 = 0;
        let media = loop {
            if std::time::Instant::now() >= deadline {
                return Err(AdapterError::Http(format!(
                    "comfyui generation timed out after {POLL_MAX_SECS}s (prompt {prompt_id})"
                )));
            }
            match self.http.get(&format!("{}/history/{prompt_id}", self.base_url)) {
                Ok(hist) => {
                    consecutive_errs = 0;
                    if let Some(m) = parse_history_media(&hist, &prompt_id) {
                        break m;
                    }
                }
                Err(e) => {
                    consecutive_errs += 1;
                    if consecutive_errs >= MAX_POLL_ERRORS {
                        return Err(AdapterError::Http(format!(
                            "comfyui /history unreachable ({consecutive_errs} consecutive errors, last: {e}); prompt {prompt_id}"
                        )));
                    }
                    tracing::debug!(error = %e, consecutive_errs, prompt_id, "comfyui /history poll blip; retrying");
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(POLL_MS));
        };

        // Fetch the media (any type) and emit it as one markdown data-URL delta with the
        // right MIME: `![…]` for images (renders inline), a `[…]` link for audio/video.
        let (filename, subfolder, kind) = media;
        let bytes = self.http.get_bytes(&format!(
            "{}/view?filename={}&subfolder={}&type={}",
            self.base_url,
            qs(&filename),
            qs(&subfolder),
            qs(&kind),
        ))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let mime = mime_for(&filename);
        let delta = if mime.starts_with("image/") {
            format!("![{}](data:{mime};base64,{b64})", qs(&filename))
        } else {
            format!("[{}](data:{mime};base64,{b64})", qs(&filename))
        };
        on_delta(&delta);

        let elapsed_ns = started.elapsed().as_nanos() as u64;
        Ok(ServeOutcome {
            // No tokens in image gen: bill sampler steps (compute-proportional, like
            // tokens are for LLMs). Deterministic per request → receipt-stable. In
            // BYO-workflow mode `bill_steps` comes from the template; else the fixed STEPS.
            tokens: bill_steps,
            done: true,
            engine: EngineMetrics {
                eval_count: bill_steps,
                eval_duration_ns: elapsed_ns,
                total_duration_ns: elapsed_ns,
                ..EngineMetrics::default()
            },
            tool_calls: Vec::new(), // image gen has no tool calls
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::ChatMessage;
    use std::cell::RefCell;

    const OBJECT_INFO: &str = r#"{"CheckpointLoaderSimple":{"input":{"required":{
        "ckpt_name":[["sd_xl_base_1.0.safetensors","dreamshaper_8.safetensors"],{}]}}}}"#;

    fn history_done(prompt_id: &str) -> String {
        format!(
            r#"{{"{prompt_id}":{{"outputs":{{"9":{{"images":[
                {{"filename":"openhydra_00001_.png","subfolder":"","type":"output"}}]}}}},
                "status":{{"completed":true}}}}}}"#
        )
    }

    /// Mock transport: scripted GET responses (history returns "running" N times first),
    /// records the /prompt body, hands back fixed PNG bytes.
    struct MockHttp {
        history_pending: RefCell<u32>,
        posted: RefCell<Vec<String>>,
    }

    impl HttpClient for MockHttp {
        fn get(&self, url: &str) -> Result<String, AdapterError> {
            if url.contains("/object_info/CheckpointLoaderSimple") {
                Ok(OBJECT_INFO.into())
            } else if url.contains("/history/") {
                let mut pending = self.history_pending.borrow_mut();
                if *pending > 0 {
                    *pending -= 1;
                    Ok("{}".into()) // still running
                } else {
                    Ok(history_done("test-prompt-id"))
                }
            } else {
                Err(AdapterError::Http(format!("unexpected GET {url}")))
            }
        }
        fn get_bytes(&self, url: &str) -> Result<Vec<u8>, AdapterError> {
            assert!(url.contains("/view?filename=openhydra_00001_.png"), "view url: {url}");
            Ok(vec![0x89, b'P', b'N', b'G']) // enough to prove the bytes round-trip
        }
        fn post_json(&self, url: &str, body: &str) -> Result<String, AdapterError> {
            assert!(url.ends_with("/prompt"), "post url: {url}");
            self.posted.borrow_mut().push(body.to_string());
            Ok(r#"{"prompt_id":"test-prompt-id","number":1}"#.into())
        }
        fn post_stream(
            &self,
            url: &str,
            _body: &str,
        ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            Err(AdapterError::Http(format!("unexpected stream {url}")))
        }
    }

    fn request(prompt: &str) -> InferenceRequest {
        InferenceRequest {
            model_ref: "dreamshaper_8.safetensors".into(),
            messages: vec![ChatMessage { role: "user".into(), content: prompt.into(), ..Default::default() }],
            max_tokens: None,
            temperature: None,
            tools: Vec::new(),
            think: None,
        }
    }

    #[test]
    fn detects_checkpoints_as_models() {
        let adapter = ComfyUiAdapter::new(
            DEFAULT_COMFYUI_URL,
            MockHttp { history_pending: RefCell::new(0), posted: RefCell::new(vec![]) },
        );
        let models = adapter.detect_models().unwrap();
        let ids: Vec<_> = models.iter().map(|m| m.engine_ref.as_str()).collect();
        assert_eq!(ids, vec!["sd_xl_base_1.0.safetensors", "dreamshaper_8.safetensors"]);
        assert!(models.iter().all(|m| m.family == "stable-diffusion"));
        assert_eq!(adapter.engine_name(), "comfyui");
    }

    #[test]
    fn serves_an_image_as_a_data_url_delta_and_bills_steps() {
        let http = MockHttp { history_pending: RefCell::new(2), posted: RefCell::new(vec![]) };
        let adapter = ComfyUiAdapter::new(DEFAULT_COMFYUI_URL, http);
        let mut out = String::new();
        let outcome =
            adapter.serve_stream(&request("a llama in a server room"), &mut |d| out.push_str(d)).unwrap();
        // One markdown image delta with the PNG bytes base64'd.
        assert!(out.starts_with("![openhydra_00001_.png](data:image/png;base64,"), "got: {}", &out[..60]);
        assert!(out.contains(&base64::engine::general_purpose::STANDARD.encode([0x89, b'P', b'N', b'G'])));
        // Steps billed as tokens; engine timing measured.
        assert_eq!(outcome.tokens, STEPS);
        assert!(outcome.done);
        assert!(outcome.engine.eval_duration_ns > 0);
        // The queued workflow carried the checkpoint + prompt + deterministic seed.
        let posted = adapter.http.posted.borrow();
        let wf: serde_json::Value = serde_json::from_str(&posted[0]).unwrap();
        assert_eq!(wf["prompt"]["4"]["inputs"]["ckpt_name"], "dreamshaper_8.safetensors");
        assert_eq!(wf["prompt"]["6"]["inputs"]["text"], "a llama in a server room");
        assert_eq!(
            wf["prompt"]["3"]["inputs"]["seed"].as_u64().unwrap(),
            seed_from_prompt("a llama in a server room"),
        );
    }

    #[test]
    fn empty_prompt_is_refused() {
        let adapter = ComfyUiAdapter::new(
            DEFAULT_COMFYUI_URL,
            MockHttp { history_pending: RefCell::new(0), posted: RefCell::new(vec![]) },
        );
        assert!(adapter.serve_stream(&request("   "), &mut |_| {}).is_err());
    }

    #[test]
    fn seed_is_deterministic_per_prompt() {
        assert_eq!(seed_from_prompt("x"), seed_from_prompt("x"));
        assert_ne!(seed_from_prompt("x"), seed_from_prompt("y"));
    }

    // ---- BYO-workflow ----

    #[test]
    fn inject_substitutes_prompt_and_dequotes_seed_and_escapes() {
        let tpl = r#"{"6":{"class_type":"CLIPTextEncode","inputs":{"text":"%prompt%"}},
                      "3":{"class_type":"KSampler","inputs":{"seed":"%seed%","steps":25}}}"#;
        let v = inject_template(tpl, "a \"quoted\" llama", "", 42).unwrap();
        assert_eq!(v["6"]["inputs"]["text"], "a \"quoted\" llama"); // escaping round-trips
        assert_eq!(v["3"]["inputs"]["seed"].as_u64().unwrap(), 42); // "%seed%" → int, not string
    }

    #[test]
    fn parse_steps_reads_first_steps_or_defaults() {
        assert_eq!(parse_template_steps(r#"{"3":{"inputs":{"steps":37}}}"#), 37);
        assert_eq!(parse_template_steps(r#"{"3":{"inputs":{"cfg":7}}}"#), STEPS);
    }

    #[test]
    fn byo_mode_detects_templates_not_checkpoints() {
        let tpls = vec![WorkflowTemplate {
            model_id: "flux2-klein".into(),
            json: "{\"6\":{\"inputs\":{\"text\":\"%prompt%\"}}}".into(),
            steps: 28,
        }];
        let a = ComfyUiAdapter::with_templates(
            DEFAULT_COMFYUI_URL,
            MockHttp { history_pending: RefCell::new(0), posted: RefCell::new(vec![]) },
            tpls,
        );
        let models = a.detect_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].engine_ref, "flux2-klein");
        assert_eq!(models[0].family, "comfyui-workflow");
    }

    // ---- media-agnostic output ----

    #[test]
    fn parse_media_finds_audio_and_video_not_just_images() {
        let audio = r#"{"p":{"outputs":{"9":{"audio":[{"filename":"song.flac","subfolder":"","type":"output"}]}}}}"#;
        assert_eq!(parse_history_media(audio, "p").unwrap().0, "song.flac");
        let video = r#"{"p":{"outputs":{"9":{"gifs":[{"filename":"clip.mp4","subfolder":"vid","type":"output"}]}}}}"#;
        let (f, sf, _) = parse_history_media(video, "p").unwrap();
        assert_eq!((f.as_str(), sf.as_str()), ("clip.mp4", "vid"));
    }

    #[test]
    fn mime_by_extension() {
        assert_eq!(mime_for("x.png"), "image/png");
        assert_eq!(mime_for("x.flac"), "audio/flac");
        assert_eq!(mime_for("x.mp4"), "video/mp4");
        assert_eq!(mime_for("x.webm"), "video/webm");
        assert_eq!(mime_for("noext"), "application/octet-stream");
    }
}
