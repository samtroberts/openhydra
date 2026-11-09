// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Safe Rust over the C ABI of OpenHydra's built-in Mac engine (`engine-mlx/include/ohmlx.h`): load an
//! MLX model folder in-process, stream a chat completion (cancellable from another thread), score a
//! prompt, tokenize, read the load manifest.
//!
//! Every call blocks (model load, generation) and must run on a plain thread, not an async runtime's
//! worker. The Swift side serialises calls on one engine, so `Engine` is `Send + Sync`.

use std::ffi::{c_char, c_void, CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

/// SHA-256 (hex) of the `default.metallib` this binary was built with: MLX's ahead-of-time kernel
/// library. The agent compares the `mlx.metallib` beside the executable against it; a mismatch means
/// foreign AOT kernels. (Most kernels are JIT-compiled at run time from sources inside the binary, which
/// this pin doesn't cover.)
pub const PINNED_METALLIB_SHA256: &str = env!("OHMLX_METALLIB_SHA256");

/// Where the build's `default.metallib` is (in the engine's build tree), for staging it beside
/// development and test executables. Releases stage it with `engine-mlx/stage.sh`.
pub const BUILT_METALLIB_PATH: &str = env!("OHMLX_METALLIB_PATH");

#[repr(C)]
struct RawEngine {
    _private: [u8; 0],
}

type OnText = extern "C" fn(user: *mut c_void, text: *const c_char, len: usize) -> i32;

const STATUS_OK: i32 = 0;
const STATUS_BUSY: i32 = 2;

extern "C" {
    fn ohmlx_version() -> *mut c_char;
    fn ohmlx_load(model_dir: *const c_char, files_json: *const c_char, err: *mut *mut c_char) -> *mut RawEngine;
    fn ohmlx_manifest(engine: *mut RawEngine) -> *mut c_char;
    fn ohmlx_generate(
        engine: *mut RawEngine,
        messages_json: *const c_char,
        params_json: *const c_char,
        on_text: OnText,
        user: *mut c_void,
        result: *mut *mut c_char,
        err: *mut *mut c_char,
    ) -> i32;
    fn ohmlx_cancel(engine: *mut RawEngine);
    fn ohmlx_score(engine: *mut RawEngine, ids: *const u32, n: usize, out: *mut f32, err: *mut *mut c_char) -> i32;
    fn ohmlx_tokenize(engine: *mut RawEngine, text: *const c_char, err: *mut *mut c_char) -> *mut c_char;
    fn ohmlx_free(engine: *mut RawEngine);
    fn ohmlx_free_string(s: *mut c_char);
}

/// Why an engine call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// Another call held the engine too long; try again shortly.
    Busy(String),
    /// Anything else, with the engine's message.
    Failed(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Busy(m) | EngineError::Failed(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for EngineError {}

/// Takes ownership of an engine-allocated string.
unsafe fn take(s: *mut c_char) -> String {
    if s.is_null() {
        return String::new();
    }
    let owned = CStr::from_ptr(s).to_string_lossy().into_owned();
    ohmlx_free_string(s);
    owned
}

fn c(bytes: &[u8]) -> Result<CString, EngineError> {
    CString::new(bytes).map_err(|_| EngineError::Failed("string contains a NUL byte".into()))
}

/// The engine and library versions compiled in, as JSON.
pub fn version() -> String {
    unsafe { take(ohmlx_version()) }
}

/// A loaded MLX model.
pub struct Engine {
    raw: *mut RawEngine,
    manifest: String,
}

// SAFETY: the Swift engine guards every call with its own lock (cancel uses a separate one and is
// documented as callable from another thread), and the handle is only freed in Drop.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

/// The caller's text sink as the trampoline sees it, plus whether it panicked.
struct Sink<'a> {
    on_text: &'a mut (dyn FnMut(&str) -> bool + Send),
    panicked: bool,
}

impl Engine {
    /// Load the model from exactly `files` (plain names in `model_dir`: weights, config, tokenizer and
    /// template files), hashing them before and after loading. No other file in the folder is read.
    pub fn load(model_dir: &Path, files: &[String]) -> Result<Engine, EngineError> {
        let dir = c(model_dir.as_os_str().as_bytes())?;
        let list = c(json_strings(files).as_bytes())?;
        let mut err = std::ptr::null_mut();
        let raw = unsafe { ohmlx_load(dir.as_ptr(), list.as_ptr(), &mut err) };
        if raw.is_null() {
            return Err(EngineError::Failed(unsafe { take(err) }));
        }
        let manifest = unsafe { take(ohmlx_manifest(raw)) };
        Ok(Engine { raw, manifest })
    }

    /// The load manifest (JSON): the loaded files and their digests, `metallib_sha256`, `model_type`,
    /// `vocab_size`, `context_length`, `gpu_working_set` and the library versions.
    pub fn manifest(&self) -> &str {
        &self.manifest
    }

    /// Stream a chat completion. `on_text` returns `false` to stop early; [`Engine::cancel`] stops it from
    /// another thread. Returns the engine's result JSON (`prompt_tokens`, `completion_tokens`,
    /// `stop_reason`, `tool_calls`, timings). A panic in `on_text` stops generation and is an error.
    pub fn generate(
        &self,
        messages_json: &str,
        params_json: &str,
        on_text: &mut (dyn FnMut(&str) -> bool + Send),
    ) -> Result<String, EngineError> {
        let messages = c(messages_json.as_bytes())?;
        let params = c(params_json.as_bytes())?;
        let mut sink = Sink { on_text, panicked: false };
        let user = &mut sink as *mut Sink<'_> as *mut c_void;
        let mut result = std::ptr::null_mut();
        let mut err = std::ptr::null_mut();
        let status = unsafe {
            ohmlx_generate(self.raw, messages.as_ptr(), params.as_ptr(), on_text_trampoline, user, &mut result, &mut err)
        };
        if status != STATUS_OK {
            return Err(EngineError::Failed(unsafe { take(err) }));
        }
        let result = unsafe { take(result) };
        if sink.panicked {
            return Err(EngineError::Failed("the text sink panicked; generation was stopped".into()));
        }
        Ok(result)
    }

    /// Stop the generation running on this engine, if any. Safe to call from any thread at any time.
    pub fn cancel(&self) {
        unsafe { ohmlx_cancel(self.raw) }
    }

    /// Teacher-forced log-probabilities: `out[i]` = log p(ids[i + 1] | ids[..=i]).
    pub fn score(&self, ids: &[u32]) -> Result<Vec<f32>, EngineError> {
        if ids.len() < 2 {
            return Ok(Vec::new());
        }
        let mut out = vec![0f32; ids.len() - 1];
        let mut err = std::ptr::null_mut();
        let status = unsafe { ohmlx_score(self.raw, ids.as_ptr(), ids.len(), out.as_mut_ptr(), &mut err) };
        match status {
            STATUS_OK => Ok(out),
            STATUS_BUSY => Err(EngineError::Busy(unsafe { take(err) })),
            _ => Err(EngineError::Failed(unsafe { take(err) })),
        }
    }

    /// The model tokenizer's ids for `text`.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        let text = c(text.as_bytes())?;
        let mut err = std::ptr::null_mut();
        let json = unsafe { ohmlx_tokenize(self.raw, text.as_ptr(), &mut err) };
        if json.is_null() {
            return Err(EngineError::Failed(unsafe { take(err) }));
        }
        let json = unsafe { take(json) };
        parse_ids(&json).ok_or_else(|| EngineError::Failed(format!("unexpected tokenizer output {json:?}")))
    }
}

/// `names` as a JSON array of strings.
fn json_strings(names: &[String]) -> String {
    let mut out = String::from("[");
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        for ch in name.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
    }
    out.push(']');
    out
}

/// A JSON array of non-negative integers (the engine's tokenizer output).
fn parse_ids(json: &str) -> Option<Vec<u32>> {
    let inner = json.trim().strip_prefix('[')?.strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(Vec::new());
    }
    inner.split(',').map(|n| n.trim().parse().ok()).collect()
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ohmlx_free(self.raw) }
    }
}

/// Forwards each text piece to the caller's closure. The engine calls it from its own thread while
/// `generate` blocks the caller's, so the closure (which is `Send`) is never used from two threads at
/// once. A panic is caught here, recorded, and turned into "stop".
extern "C" fn on_text_trampoline(user: *mut c_void, text: *const c_char, len: usize) -> i32 {
    if user.is_null() || text.is_null() || len == 0 {
        return 0;
    }
    let sink = unsafe { &mut *(user as *mut Sink<'_>) };
    let bytes = unsafe { std::slice::from_raw_parts(text as *const u8, len) };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (sink.on_text)(&String::from_utf8_lossy(bytes))));
    match outcome {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(_) => {
            sink.panicked = true;
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_lists_are_json() {
        assert_eq!(json_strings(&["a.json".into(), "b\"c".into()]), r#"["a.json","b\"c"]"#);
        assert_eq!(json_strings(&[]), "[]");
    }

    #[test]
    fn tokenizer_output_parses() {
        assert_eq!(parse_ids("[1, 2,3]"), Some(vec![1, 2, 3]));
        assert_eq!(parse_ids("[]"), Some(vec![]));
        assert_eq!(parse_ids("[1,-2]"), None);
        assert_eq!(parse_ids("{}"), None);
    }
}
