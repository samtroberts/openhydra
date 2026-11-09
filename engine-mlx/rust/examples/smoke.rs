// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Load an MLX model folder, print its manifest, stream one reply and score the prompt's own tokens.
//!
//!   engine-mlx/stage.sh engine-mlx/rust/target/debug/examples
//!   cargo run --example smoke --manifest-path engine-mlx/rust/Cargo.toml -- <model-dir> [prompt]

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: smoke <model-dir> [prompt]");
    let prompt = args.next().unwrap_or_else(|| "Name three primary colours.".into());
    println!("versions: {}", openhydra_mlx_engine::version());
    println!("pinned metallib: {}", openhydra_mlx_engine::PINNED_METALLIB_SHA256);

    let t0 = std::time::Instant::now();
    // Every plain top-level file (the agent loads only the model's own files; this is a smoke test).
    let mut files: Vec<String> = std::fs::read_dir(&dir)
        .expect("model dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| !n.starts_with('.'))
        .collect();
    files.sort();
    let engine = openhydra_mlx_engine::Engine::load(std::path::Path::new(&dir), &files).expect("load");
    println!("loaded in {:?}\nmanifest: {}", t0.elapsed(), engine.manifest());

    let messages = format!(r#"[{{"role":"user","content":{}}}]"#, json_str(&prompt));
    let mut text = String::new();
    let result = engine
        .generate(&messages, r#"{"max_tokens":64,"temperature":0,"enable_thinking":false}"#, &mut |piece| {
            text.push_str(piece);
            true
        })
        .expect("generate");
    println!("reply: {text:?}\nresult: {result}");

    let ids = engine.tokenize(&prompt).expect("tokenize");
    let scores = engine.score(&ids).expect("score");
    println!("ids: {ids:?}\nscores: {scores:?}");
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
