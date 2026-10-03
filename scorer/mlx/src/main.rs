// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! `openhydra-mlx-scorer`: teacher-forced prompt scoring for MLX models, the side-scorer the
//! OpenHydra agent runs for LM Studio's MLX engine (which can't score a prompt itself). Same HTTP
//! interface and flags as `openhydra-scorer` (the patched llama-server), so the agent drives both
//! the same way:
//!
//! * `POST /v1/prompt_logprobs {"prompt_token_ids": [...]}` → `{"model", "scores": {"token_ids",
//!   "logprobs", "ranks", "top1_ids", "top1_logprobs"}}` (one entry per token after the first);
//! * `GET /v1/prompt_logprobs` → 405 when scoring works, 501 when it was disabled because a
//!   startup check (scoring the longest accepted prompt) failed, e.g. out of memory;
//! * `GET /v1/models` (readiness).
//!
//! Flags: `-m <model dir>`, `--host`, `--port`, `-ngl 0` (CPU instead of the GPU),
//! `--exit-on-stdin-eof` (exit when the supervising agent goes away). llama.cpp-only flags the agent
//! may pass (`--prompt-scoring`, `-c N`, `-np N`, `--load-mode X`) are accepted and ignored.

mod qwen3;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use serde_json::{json, Value};

/// Longest prompt accepted (= the agent's MAX_CHALLENGE_TOKENS and the llama.cpp scorer's limit).
const MAX_TOKENS: usize = 512;

struct Args {
    model: PathBuf,
    host: String,
    port: u16,
    cpu: bool,
    exit_on_stdin_eof: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let (mut model, mut host, mut port, mut cpu, mut eof) = (None, "127.0.0.1".to_string(), 8080u16, false, false);
    while let Some(a) = it.next() {
        let mut val = || it.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "-m" | "--model" => model = Some(PathBuf::from(val()?)),
            "--host" => host = val()?,
            "--port" => port = val()?.parse().map_err(|_| "bad --port")?,
            "-ngl" | "--n-gpu-layers" => cpu = val()? == "0",
            "--exit-on-stdin-eof" => eof = true,
            "--prompt-scoring" => {}
            "-c" | "--ctx-size" | "-np" | "--parallel" | "--load-mode" | "-t" | "--threads" => {
                val()?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args { model: model.ok_or("-m <model dir> is required")?, host, port, cpu, exit_on_stdin_eof: eof })
}

/// Per-position scores of `ids` under the model, in the endpoint's shape.
fn score(model: &mut qwen3::Model, ids: &[u32]) -> Result<Value, String> {
    let lp = model.log_probs(ids).map_err(|e| e.to_string())?;
    lp.eval().map_err(|e| e.to_string())?;
    let v = lp.shape()[1] as usize;
    let flat: &[f32] = lp.as_slice();
    let (mut lps, mut ranks, mut top_ids, mut top_lps) = (vec![], vec![], vec![], vec![]);
    for (i, &next) in ids[1..].iter().enumerate() {
        let row = &flat[i * v..(i + 1) * v];
        let actual = row[next as usize];
        let (arg, top) = row.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (j, &x)| if x > b.1 { (j, x) } else { b });
        lps.push(actual as f64);
        ranks.push(1 + row.iter().filter(|&&x| x > actual).count());
        top_ids.push(arg);
        top_lps.push(top as f64);
    }
    Ok(json!({"token_ids": ids, "logprobs": lps, "ranks": ranks, "top1_ids": top_ids, "top1_logprobs": top_lps}))
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(stream: &TcpStream) -> std::io::Result<Request> {
    let mut r = BufReader::new(stream);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0; len.min(16 << 20)];
    r.read_exact(&mut body)?;
    Ok(Request { method, path, body })
}

fn respond(mut stream: &TcpStream, status: u16, body: &Value) {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        501 => "Not Implemented",
        _ => "Service Unavailable",
    };
    let b = body.to_string();
    let _ = write!(stream, "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
}

fn error(code: u16, message: &str) -> Value {
    json!({"error": {"code": code, "message": message}})
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("openhydra-mlx-scorer: {e}");
            std::process::exit(2);
        }
    };
    if args.exit_on_stdin_eof {
        // The supervisor holds our stdin open for as long as it wants us.
        std::thread::spawn(|| {
            let mut buf = [0u8; 256];
            while std::io::stdin().read(&mut buf).map(|n| n > 0).unwrap_or(false) {}
            std::process::exit(0);
        });
    }
    if args.cpu {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    }
    let t = std::time::Instant::now();
    let mut model = match qwen3::Model::load(&args.model) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("openhydra-mlx-scorer: {}: {e}", args.model.display());
            std::process::exit(1);
        }
    };
    let vocab = model.vocab_size();
    eprintln!("openhydra-mlx-scorer: loaded {} in {:?} ({})", args.model.display(), t.elapsed(), if args.cpu { "CPU" } else { "GPU" });

    // Score the longest accepted prompt once now, so a scorer that can't (out of memory) says so at
    // startup instead of failing challenges later.
    let probe: Vec<u32> = (0..MAX_TOKENS).map(|i| ((1000 + 7919 * i) % vocab) as u32).collect();
    let enabled = match score(&mut model, &probe) {
        Ok(_) => {
            eprintln!("openhydra-mlx-scorer: prompt scoring ready (up to {MAX_TOKENS} tokens)");
            true
        }
        Err(e) => {
            eprintln!("openhydra-mlx-scorer: prompt scoring DISABLED: a {MAX_TOKENS}-token prompt failed ({e})");
            false
        }
    };
    let name = args.model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let listener = match TcpListener::bind((args.host.as_str(), args.port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("openhydra-mlx-scorer: bind {}:{}: {e}", args.host, args.port);
            std::process::exit(1);
        }
    };
    for stream in listener.incoming().flatten() {
        let Ok(req) = read_request(&stream) else { continue };
        let path = req.path.split('?').next().unwrap_or("");
        match (req.method.as_str(), path) {
            ("GET", "/v1/models") => respond(&stream, 200, &json!({"object": "list", "data": [{"id": name, "object": "model"}]})),
            ("GET", "/health") => respond(&stream, 200, &json!({"status": "ok"})),
            ("GET", "/v1/prompt_logprobs") if enabled => respond(&stream, 405, &error(405, "use POST")),
            (_, "/v1/prompt_logprobs") if !enabled => {
                respond(&stream, 501, &error(501, "prompt scoring is disabled: it did not fit in memory at startup"))
            }
            ("POST", "/v1/prompt_logprobs") => {
                let ids: Option<Vec<u32>> = serde_json::from_slice::<Value>(&req.body)
                    .ok()
                    .and_then(|v| serde_json::from_value(v.get("prompt_token_ids")?.clone()).ok());
                match ids {
                    Some(ids) if (2..=MAX_TOKENS).contains(&ids.len()) && ids.iter().all(|&i| (i as usize) < vocab) => {
                        match score(&mut model, &ids) {
                            Ok(scores) => respond(&stream, 200, &json!({"model": name, "scores": scores})),
                            Err(e) => respond(&stream, 503, &error(503, &format!("prompt scoring failed: {e}"))),
                        }
                    }
                    _ => respond(&stream, 400, &error(400, &format!("prompt_token_ids must hold 2 to {MAX_TOKENS} valid token ids"))),
                }
            }
            _ => respond(&stream, 404, &error(404, "not found")),
        }
    }
}
