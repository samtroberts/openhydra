// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The built-in Mac engine against a real MLX model folder. Needs a `--features mlx-swift` build and a
//! small local model; ignored by default:
//!
//!   OHMLX_TEST_MODEL=<MLX model folder> [OHMLX_TEST_TOOL_MODEL=<tool-calling MLX model, e.g. Qwen3.5>] \
//!     cargo test -p openhydra-agent --features mlx-swift --test mlx_swift_engine -- --ignored --test-threads 1
#![cfg(all(feature = "mlx-swift", target_os = "macos", target_arch = "aarch64"))]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use openhydra_agent::adapter::{AdapterError, ChatMessage, EngineAdapter, InferenceRequest};
use openhydra_agent::adapters::mlx_swift::MlxSwiftAdapter;

/// Stage the engine's kernels beside this test executable.
fn kernels() {
    let exe = std::env::current_exe().unwrap();
    std::fs::copy(openhydra_mlx_engine::BUILT_METALLIB_PATH, exe.with_file_name("mlx.metallib")).unwrap();
}

/// The test model, with the kernels staged.
fn model() -> PathBuf {
    let dir = std::env::var_os("OHMLX_TEST_MODEL").expect("set OHMLX_TEST_MODEL to an MLX model folder");
    kernels();
    PathBuf::from(dir)
}

fn request(model: &str, prompt: &str, max_tokens: u32) -> InferenceRequest {
    InferenceRequest {
        model_ref: model.into(),
        messages: vec![ChatMessage::new("user", prompt)],
        max_tokens: Some(max_tokens),
        temperature: Some(0.0),
        tools: vec![],
        think: Some(false),
        seed: None,
        stop: Default::default(),
    }
}

/// A copy of `src` (as symlinks) in a fresh folder under the system temp dir, plus `extra` files.
fn linked_copy(src: &Path, name: &str, extra: &[(&str, &[u8])]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ohmlx-it-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        std::os::unix::fs::symlink(entry.path().canonicalize().unwrap(), dir.join(entry.file_name())).unwrap();
    }
    for (file, bytes) in extra {
        std::fs::write(dir.join(file), bytes).unwrap();
    }
    dir
}

#[test]
#[ignore = "needs OHMLX_TEST_MODEL and a --features mlx-swift build"]
fn an_early_stop_ends_generation_and_leaves_the_engine_idle() {
    let adapter = MlxSwiftAdapter::load(&model(), None, None).expect("load");
    let name = adapter.model().engine_ref.clone();
    let req = request(&name, "Count from 1 to 200, one number per line.", 400);
    let stop = req.stop.clone();
    let mut pieces = 0;
    let outcome = adapter
        .serve_stream(&req, &mut |_| {
            pieces += 1;
            if pieces == 3 {
                stop.set();
            }
        })
        .expect("serve");
    assert!(!outcome.done, "a stopped serve is not done");
    assert!(!outcome.hit_token_limit, "a stopped serve did not reach the limit");
    assert!(outcome.tokens >= 3 && outcome.tokens < 100, "tokens {}", outcome.tokens);
    // The generation has ended: scoring gets the engine at once instead of waiting it out.
    let started = Instant::now();
    adapter.score_prompt(&name, &[1, 2, 3]).expect("score right after a stop");
    assert!(started.elapsed() < Duration::from_secs(2), "{:?}", started.elapsed());
    let mut text = String::new();
    let next = adapter.serve_stream(&request(&name, "Say OK.", 8), &mut |d| text.push_str(d)).expect("second serve");
    assert!(next.done && !text.is_empty(), "{text:?}");
}

#[test]
#[ignore = "needs OHMLX_TEST_MODEL and a --features mlx-swift build"]
fn hitting_max_tokens_is_done_not_cancelled() {
    let adapter = MlxSwiftAdapter::load(&model(), None, None).expect("load");
    let name = adapter.model().engine_ref.clone();
    let outcome = adapter.serve_stream(&request(&name, "Write a long story about the sea.", 4), &mut |_| {}).expect("serve");
    assert_eq!(outcome.tokens, 4);
    assert!(outcome.done, "a generation that ran to max_tokens is done");
    assert!(outcome.hit_token_limit, "and it says it was cut at the limit");
    let whole = adapter.serve_stream(&request(&name, "Say OK.", 64), &mut |_| {}).expect("serve");
    assert!(whole.done && !whole.hit_token_limit, "{whole:?}");
}

#[test]
#[ignore = "needs OHMLX_TEST_MODEL and a --features mlx-swift build"]
fn a_stop_during_prefill_returns_without_finishing_it() {
    let adapter = MlxSwiftAdapter::load(&model(), None, None).expect("load");
    let name = adapter.model().engine_ref.clone();
    let long = "The quick brown fox jumps over the lazy dog. ".repeat(1800); // about 18k tokens
    let req = request(&name, &long, 16);
    let stop = req.stop.clone();
    let timer = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        stop.set();
    });
    let started = Instant::now();
    let outcome = adapter.serve_stream(&req, &mut |_| {}).expect("serve");
    timer.join().unwrap();
    assert!(!outcome.done && outcome.tokens == 0, "{outcome:?}");
    assert!(started.elapsed() < Duration::from_secs(3), "prefill ran on: {:?}", started.elapsed());
}

#[test]
#[ignore = "needs OHMLX_TEST_MODEL and a --features mlx-swift build"]
fn scoring_answers_busy_during_a_long_generation() {
    let adapter = MlxSwiftAdapter::load(&model(), None, None).expect("load");
    let name = adapter.model().engine_ref.clone();
    std::thread::scope(|scope| {
        let generating = scope.spawn(|| {
            let req = request(&name, "Count from 1 to 5000, one number per line.", 4000);
            adapter.serve_stream(&req, &mut |_| {})
        });
        std::thread::sleep(Duration::from_secs(1));
        let started = Instant::now();
        let err = adapter.score_prompt(&name, &[1, 2, 3]).expect_err("busy");
        assert!(matches!(err, AdapterError::Busy(_)), "{err:?}");
        assert!(started.elapsed() >= Duration::from_secs(19), "{:?}", started.elapsed());
        generating.join().unwrap().expect("the generation itself completes");
    });
}

#[test]
#[ignore = "needs OHMLX_TEST_MODEL and a --features mlx-swift build"]
fn oversized_requests_are_refused_before_the_template_runs() {
    let adapter = MlxSwiftAdapter::load(&model(), None, None).expect("load");
    let name = adapter.model().engine_ref.clone();
    let huge = "x".repeat((adapter.context_length * 8 + 1) as usize);
    let started = Instant::now();
    let err = adapter.serve_stream(&request(&name, &huge, 8), &mut |_| {}).expect_err("refused");
    assert!(err.to_string().contains("bytes"), "{err}");
    assert!(started.elapsed() < Duration::from_millis(500), "{:?}", started.elapsed());
}

#[test]
#[ignore = "needs OHMLX_TEST_MODEL and a --features mlx-swift build"]
fn the_engine_refuses_ids_outside_the_vocabulary_by_itself() {
    let dir = model();
    let adapter = MlxSwiftAdapter::load(&dir, None, None).expect("load");
    let name = adapter.model().engine_ref.clone();
    let scores = adapter.score_prompt(&name, &[1, 2, 3]).expect("score");
    assert_eq!(scores.len(), 2);
    assert!(scores.iter().all(|s| s.is_finite() && *s <= 0.0));
    drop(adapter);
    // Straight to the engine, past the adapter's own check: the Swift bound must hold on its own.
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| !n.starts_with('.'))
        .collect();
    files.sort();
    let engine = openhydra_mlx_engine::Engine::load(&dir, &files).expect("engine load");
    let manifest: serde_json::Value = serde_json::from_str(engine.manifest()).unwrap();
    let vocab = manifest["vocab_size"].as_u64().unwrap() as u32;
    engine.score(&[1, vocab - 1]).expect("the last id in the table");
    let err = engine.score(&[1, vocab]).expect_err("past the table");
    assert!(err.to_string().contains("outside this model's vocabulary"), "{err}");
}

#[test]
#[ignore = "needs OHMLX_TEST_MODEL and a --features mlx-swift build"]
fn the_digest_is_the_weights_rule_and_stray_weights_are_ignored() {
    let src = model();
    let original = MlxSwiftAdapter::load(&src, None, None).expect("load");
    assert_eq!(original.model().weight_digest, openhydra_agent::weights::digest_now(&src).unwrap());
    let digest = original.model().weight_digest.clone();
    drop(original);

    // A temp-dir folder loads, and a stray hidden weights file next to the model is neither loaded nor
    // in the digest: it's reported as ignored.
    let tampered = linked_copy(&src, "hidden", &[(".override.safetensors", b"not weights")]);
    let adapter = MlxSwiftAdapter::load(&tampered, Some("copy"), None).expect("loads without the stray file");
    assert_eq!(adapter.model().weight_digest, digest);
    assert_eq!(adapter.ignored_weights, [".override.safetensors"]);
    let mut text = String::new();
    adapter.serve_stream(&request("copy", "Say OK.", 8), &mut |d| text.push_str(d)).expect("serves");
    assert!(!text.is_empty());
    let _ = std::fs::remove_dir_all(&tampered);
}

/// An assistant turn's earlier tool call must reach the template with its arguments: a model can name
/// the city it looked up only if `get_weather(city="Paris")` was rendered (the tool result alone says
/// "sunny"). Templates that iterate the arguments (Qwen3.5) dropped them when they arrived as a string.
#[test]
#[ignore = "needs OHMLX_TEST_TOOL_MODEL (a tool-calling MLX model, e.g. Qwen3.5) and a --features mlx-swift build"]
fn earlier_tool_call_arguments_reach_the_template() {
    let dir = std::env::var_os("OHMLX_TEST_TOOL_MODEL").expect("set OHMLX_TEST_TOOL_MODEL");
    kernels();
    let adapter = MlxSwiftAdapter::load(Path::new(&dir), None, None).expect("load");
    let messages: Vec<ChatMessage> = serde_json::from_value(serde_json::json!([
        {"role": "user", "content": "What's the weather where I'm going?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
        ]},
        {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
        {"role": "user", "content": "Which city did you look up the weather for? Answer with the city name only."}
    ]))
    .unwrap();
    let tools = vec![serde_json::json!({"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "default": null}}, "required": ["city"]}
    }})];
    let mut req = request(&adapter.model().engine_ref.clone(), "", 32);
    req.messages = messages;
    req.tools = tools;
    let mut text = String::new();
    adapter.serve_stream(&req, &mut |d| text.push_str(d)).expect("serve");
    assert!(text.contains("Paris"), "{text:?}");
    // A malformed tools list is an error, not a silent serve without tools.
    let mut bad = request(&adapter.model().engine_ref.clone(), "hi", 8);
    bad.tools = vec![serde_json::json!("not an object")];
    let err = adapter.serve_stream(&bad, &mut |_| {}).expect_err("malformed tools");
    assert!(err.to_string().contains("tools"), "{err}");
}

/// `card export` describes a built-in model without loading it; the card must name and hash exactly
/// what the loaded engine serves.
#[test]
#[ignore = "needs OHMLX_TEST_MODEL and a --features mlx-swift build"]
fn describing_a_model_matches_what_loading_it_serves() {
    kernels();
    let adapter = MlxSwiftAdapter::load(&model(), None, None).expect("load");
    let described = MlxSwiftAdapter::describe(&model(), None).expect("describe");
    assert_eq!(&described, adapter.model());
    let named = MlxSwiftAdapter::describe(&model(), Some("my-model")).expect("describe");
    assert_eq!(named.engine_ref, "my-model");
}
