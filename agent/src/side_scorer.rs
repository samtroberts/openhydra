// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The side-scorer: answer prompt-scoring challenges for engines that can't score a prompt
//! themselves (Ollama, LM Studio) by running OpenHydra's patched llama-server (`openhydra-scorer`,
//! built with the prompt-scoring endpoint) on the model's own weights file.
//!
//! * It is started on the first challenge for a file and stopped after [`IDLE_STOP`] without one, so
//!   it holds no memory between checks. A challenge that arrives while it is still loading is told
//!   to retry ([`AdapterError::Busy`]); gateways wait out a warm-up before counting anything.
//! * Two scorers: `openhydra-scorer` (patched llama-server) for a GGUF file, `openhydra-mlx-scorer`
//!   for an MLX model directory (LM Studio's MLX engine), with the same interface.
//! * GPU when it fits: it starts with the engine's defaults (GPU offload, sized to free memory), and
//!   the scorer's own startup check disables scoring when the scoring buffers don't fit next to the
//!   model — then a GGUF scorer is restarted without the memory-mapped load (Apple Silicon maps the
//!   whole file into GPU memory) and finally on the CPU alone. MLX runs on the GPU only.
//! * What a pass proves is unchanged: the provider holds the claimed weights file and computes with
//!   it. It doesn't prove the engine serves that file (challenges are recognisable either way).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use crate::adapter::{AdapterError, DetectedModel, EngineAdapter, HttpClient, InferenceRequest, ServeOutcome};
use crate::adapters::scoring::{score_prompt, ScoringDialect};

/// File name of the bundled GGUF scorer (OpenHydra's patched llama-server), next to the agent.
pub const SCORER_BIN: &str = "openhydra-scorer";
/// File name of the bundled MLX scorer (`scorer/mlx`, macOS), next to the agent.
pub const MLX_SCORER_BIN: &str = "openhydra-mlx-scorer";
/// Stop the scorer after this long without a challenge.
pub const IDLE_STOP: Duration = Duration::from_secs(15 * 60);
/// How long a challenge waits for a starting scorer before it is told to retry.
const CHALLENGE_WAIT: Duration = Duration::from_secs(20);
/// After every way of starting a file's scorer failed, don't try that file again for this long.
const FAILED_BACKOFF: Duration = Duration::from_secs(30 * 60);

/// A scorer binary: `explicit` if given, else `name` next to the running agent.
pub fn locate_scorer(explicit: Option<&Path>, name: &str) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return p.is_file().then(|| p.to_path_buf());
    }
    let exe = std::env::current_exe().ok()?;
    let name = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    let p = exe.parent()?.join(name);
    p.is_file().then_some(p)
}

/// The ways a scorer is started, in order; each is kept only if its startup check leaves scoring
/// on. A GGUF file: GPU defaults, then (Apple Silicon) without the memory-mapped load, then CPU
/// only. An MLX model directory: the GPU only — MLX's CPU path takes minutes for one prompt.
fn attempts(weights: &Path) -> Vec<Vec<String>> {
    if weights.is_dir() {
        return vec![vec![]];
    }
    let mut v = vec![vec![]];
    if cfg!(target_os = "macos") {
        v.push(vec!["--load-mode".into(), "none".into()]);
    }
    v.push(vec!["-ngl".into(), "0".into()]);
    v
}

/// A running scorer process.
pub trait ScorerInstance: Send {
    /// Its base URL, e.g. `http://127.0.0.1:41234`.
    fn base_url(&self) -> &str;
    /// Whether the process is still running.
    fn alive(&mut self) -> bool;
}

/// Starts scorer processes (a real child process in production; a stub in tests).
pub trait ScorerLauncher: Send + Sync {
    /// Start a scorer on `weights` with `extra` arguments and return once it answers HTTP.
    fn launch(&self, weights: &Path, extra: &[String]) -> Result<Box<dyn ScorerInstance>, String>;
}

enum State {
    Idle,
    Starting(PathBuf),
    Ready { weights: PathBuf, instance: Box<dyn ScorerInstance>, last_used: Instant },
    Failed { weights: PathBuf, at: Instant },
}

struct Inner<H> {
    launcher: Box<dyn ScorerLauncher>,
    http: H,
    state: Mutex<State>,
    changed: Condvar,
}

/// Manages one scorer at a time (one weights file); see the module docs.
pub struct SideScorer<H: HttpClient + Send + Sync + 'static> {
    inner: Arc<Inner<H>>,
}

impl<H: HttpClient + Send + Sync + 'static> SideScorer<H> {
    /// A side-scorer starting processes with `launcher` and talking to them with `http`. Starts a
    /// background thread that stops an idle scorer.
    pub fn new(launcher: Box<dyn ScorerLauncher>, http: H) -> Self {
        let inner = Arc::new(Inner { launcher, http, state: Mutex::new(State::Idle), changed: Condvar::new() });
        let weak = Arc::downgrade(&inner);
        std::thread::spawn(move || idle_reaper(weak));
        Self { inner }
    }

    /// Score `token_ids` with a scorer on `weights` (starting or switching it as needed).
    /// [`AdapterError::Busy`] while it is still starting.
    pub fn score(&self, weights: &Path, token_ids: &[u32]) -> Result<Vec<f32>, AdapterError> {
        let base = self.ready(weights)?;
        let out = score_prompt(&self.inner.http, &base, ScoringDialect::OpenHydra, "default", token_ids);
        if out.is_err() {
            // A dead process is replaced on the next challenge; a live one just failed this text.
            let mut st = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
            if let State::Ready { instance, .. } = &mut *st {
                if !instance.alive() {
                    *st = State::Idle;
                }
            }
        }
        out
    }

    /// The base URL of a ready scorer on `weights`, starting one if needed.
    fn ready(&self, weights: &Path) -> Result<String, AdapterError> {
        let inner = &self.inner;
        let mut st = inner.state.lock().unwrap_or_else(|e| e.into_inner());
        let deadline = Instant::now() + CHALLENGE_WAIT;
        loop {
            // A scorer process that died is replaced.
            if let State::Ready { instance, .. } = &mut *st {
                if !instance.alive() {
                    *st = State::Idle;
                }
            }
            match &mut *st {
                State::Ready { weights: w, instance, last_used } if w == weights => {
                    *last_used = Instant::now();
                    return Ok(instance.base_url().to_string());
                }
                State::Failed { weights: w, at } if w == weights && at.elapsed() < FAILED_BACKOFF => {
                    return Err(AdapterError::Unsupported("prompt scoring doesn't fit in memory next to this model".into()));
                }
                State::Starting(w) if w == weights => {
                    let now = Instant::now();
                    if now >= deadline {
                        return Err(AdapterError::Busy("prompt scorer is still starting".into()));
                    }
                    st = inner.changed.wait_timeout(st, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
                }
                State::Starting(_) => {
                    // Another file's scorer is starting; this one waits its turn.
                    return Err(AdapterError::Busy("prompt scorer is busy with another model".into()));
                }
                _ => {
                    // Idle, a dead or other file's scorer, or an expired failure: (re)start.
                    *st = State::Starting(weights.to_path_buf());
                    let (inner2, path) = (Arc::clone(inner), weights.to_path_buf());
                    std::thread::spawn(move || start(&inner2, &path));
                }
            }
        }
    }
}

/// Bring up a scorer on `weights`, trying each start-up mode until one keeps scoring enabled.
fn start<H: HttpClient>(inner: &Inner<H>, weights: &Path) {
    let started = Instant::now();
    let mut outcome = None;
    for extra in attempts(weights) {
        match inner.launcher.launch(weights, &extra) {
            // The scorer's own startup check answers GET with 405 when scoring is on, 501 when it
            // had to disable it (the scoring buffers don't fit next to the model).
            Ok(instance) if inner.http.get_status(&format!("{}/v1/prompt_logprobs", instance.base_url())).ok() == Some(405) => {
                tracing::info!(weights = %weights.display(), mode = ?extra, secs = started.elapsed().as_secs(), "prompt scorer ready");
                outcome = Some(instance);
                break;
            }
            Ok(_) => tracing::info!(weights = %weights.display(), mode = ?extra, "prompt scoring didn't fit this way; trying the next"),
            Err(e) => tracing::warn!(weights = %weights.display(), mode = ?extra, error = %e, "prompt scorer failed to start"),
        }
    }
    let mut st = inner.state.lock().unwrap_or_else(|e| e.into_inner());
    *st = match outcome {
        Some(instance) => State::Ready { weights: weights.to_path_buf(), instance, last_used: Instant::now() },
        None => {
            tracing::warn!(weights = %weights.display(), "prompt scoring unavailable for this model: it doesn't fit in memory");
            State::Failed { weights: weights.to_path_buf(), at: Instant::now() }
        }
    };
    inner.changed.notify_all();
}

/// Stop a scorer that has been idle for [`IDLE_STOP`]; exits when the side-scorer is dropped.
fn idle_reaper<H>(inner: Weak<Inner<H>>) {
    loop {
        std::thread::sleep(Duration::from_secs(30));
        let Some(inner) = inner.upgrade() else { return };
        let mut st = inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(&*st, State::Ready { last_used, .. } if last_used.elapsed() >= IDLE_STOP) {
            tracing::info!("stopping the idle prompt scorer");
            *st = State::Idle; // dropping the instance stops the process
        }
    }
}

// ── the real scorer process ──

/// Starts a scorer as a child process on a free local port: `openhydra-scorer` (a patched
/// llama-server) for a GGUF file, `openhydra-mlx-scorer` for an MLX model directory. Both take the
/// same flags, including `--exit-on-stdin-eof`.
pub struct ProcessLauncher {
    gguf: Option<PathBuf>,
    mlx: Option<PathBuf>,
    /// How long a start may take (loading a multi-GB model on the CPU is slow).
    ready_timeout: Duration,
}

impl ProcessLauncher {
    pub fn new(gguf: Option<PathBuf>, mlx: Option<PathBuf>) -> Self {
        Self { gguf, mlx, ready_timeout: Duration::from_secs(300) }
    }
}

struct ProcessInstance {
    child: std::process::Child,
    /// The scorer's stdin, held open: it runs with `--exit-on-stdin-eof`, so it exits when this
    /// closes — when the instance is dropped, and also when the agent dies by a signal or a crash,
    /// which skips every destructor (no orphaned scorer holding model memory).
    _lifeline: Option<std::process::ChildStdin>,
    base: String,
}

impl ScorerInstance for ProcessInstance {
    fn base_url(&self) -> &str {
        &self.base
    }
    fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for ProcessInstance {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl ScorerLauncher for ProcessLauncher {
    fn launch(&self, weights: &Path, extra: &[String]) -> Result<Box<dyn ScorerInstance>, String> {
        let binary = if weights.is_dir() { &self.mlx } else { &self.gguf };
        let binary = binary.as_ref().ok_or_else(|| {
            format!("no {} available for {}", if weights.is_dir() { MLX_SCORER_BIN } else { SCORER_BIN }, weights.display())
        })?;
        let port = std::net::TcpListener::bind("127.0.0.1:0").and_then(|l| l.local_addr()).map_err(|e| e.to_string())?.port();
        let log = std::fs::File::create(std::env::temp_dir().join("openhydra-scorer.log")).map_err(|e| e.to_string())?;
        let mut child = std::process::Command::new(binary)
            .args(["--host", "127.0.0.1", "--port", &port.to_string(), "--prompt-scoring", "--exit-on-stdin-eof"])
            .args(["-c", "4096", "-np", "1", "-m"])
            .arg(weights)
            .args(extra)
            .stdin(std::process::Stdio::piped())
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log)
            .spawn()
            .map_err(|e| format!("{}: {e}", binary.display()))?;
        let lifeline = child.stdin.take();
        let mut instance = ProcessInstance { child, _lifeline: lifeline, base: format!("http://127.0.0.1:{port}") };
        let http = crate::http::ReqwestClient::new().map_err(|e| e.to_string())?;
        let deadline = Instant::now() + self.ready_timeout;
        while Instant::now() < deadline {
            if !instance.alive() {
                return Err("the scorer exited while loading (see openhydra-scorer.log in the temp directory)".into());
            }
            if http.get(&format!("{}/v1/models", instance.base)).is_ok() {
                return Ok(Box::new(instance));
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        Err("the scorer didn't come up in time".into())
    }
}

/// The scorer's HTTP client for production: [`crate::http::ReqwestClient`].
pub type LiveSideScorer = SideScorer<crate::http::ReqwestClient>;

/// A side-scorer running these scorer binaries (see [`locate_scorer`]); at least one is needed.
pub fn live_side_scorer(gguf: Option<PathBuf>, mlx: Option<PathBuf>) -> Result<Arc<LiveSideScorer>, String> {
    if gguf.is_none() && mlx.is_none() {
        return Err("no scorer binary".into());
    }
    Ok(Arc::new(SideScorer::new(Box::new(ProcessLauncher::new(gguf, mlx)), crate::http::ReqwestClient::new().map_err(|e| e.to_string())?)))
}

// ── the adapter wrapper ──

/// Wraps an engine adapter so that a challenge its engine can't score (`Unsupported`) is scored
/// by the side-scorer on the model's weights file ([`EngineAdapter::weights_path`]). Everything
/// else passes through; with no side-scorer it is a plain pass-through.
pub struct SideScored<A> {
    inner: A,
    scorer: Option<Arc<LiveSideScorer>>,
}

impl<A> SideScored<A> {
    pub fn new(inner: A, scorer: Option<Arc<LiveSideScorer>>) -> Self {
        Self { inner, scorer }
    }
}

impl<A: EngineAdapter> EngineAdapter for SideScored<A> {
    fn engine_name(&self) -> &'static str {
        self.inner.engine_name()
    }
    fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
        self.inner.detect_models()
    }
    fn serve_stream(&self, request: &InferenceRequest, on_delta: &mut dyn FnMut(&str)) -> Result<ServeOutcome, AdapterError> {
        self.inner.serve_stream(request, on_delta)
    }
    fn score_prompt(&self, model_ref: &str, token_ids: &[u32]) -> Result<Vec<f32>, AdapterError> {
        match self.inner.score_prompt(model_ref, token_ids) {
            Err(AdapterError::Unsupported(why)) => match (&self.scorer, self.inner.weights_path(model_ref)) {
                (Some(scorer), Some(path)) => scorer.score(&path, token_ids),
                _ => Err(AdapterError::Unsupported(why)),
            },
            other => other,
        }
    }
    fn weights_path(&self, model_ref: &str) -> Option<PathBuf> {
        self.inner.weights_path(model_ref)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A launcher whose instances answer the scoring check with `statuses[attempt]`, and an HTTP
    /// stub that scores every position −0.5.
    struct Stub {
        launches: Arc<AtomicUsize>,
        /// Status the n-th launched instance answers GET /v1/prompt_logprobs with.
        statuses: Vec<u16>,
        delay: Duration,
    }
    struct StubInstance(String);
    impl ScorerInstance for StubInstance {
        fn base_url(&self) -> &str {
            &self.0
        }
        fn alive(&mut self) -> bool {
            true
        }
    }
    impl ScorerLauncher for Stub {
        fn launch(&self, _: &Path, _: &[String]) -> Result<Box<dyn ScorerInstance>, String> {
            std::thread::sleep(self.delay);
            let n = self.launches.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(StubInstance(format!("http://stub/{}", self.statuses.get(n).copied().unwrap_or(501)))))
        }
    }
    struct StubHttp;
    impl HttpClient for StubHttp {
        fn get(&self, _: &str) -> Result<String, AdapterError> {
            Ok(String::new())
        }
        fn get_status(&self, url: &str) -> Result<u16, AdapterError> {
            // The instance's base URL carries the status it answers.
            Ok(url.trim_end_matches("/v1/prompt_logprobs").rsplit('/').next().unwrap().parse().unwrap())
        }
        fn post_json(&self, _: &str, body: &str) -> Result<String, AdapterError> {
            let v: serde_json::Value = serde_json::from_str(body).unwrap();
            let n = v["prompt_token_ids"].as_array().unwrap().len();
            Ok(serde_json::json!({"scores": {"token_ids": v["prompt_token_ids"], "logprobs": vec![-0.5; n - 1]}}).to_string())
        }
        fn post_stream(&self, _: &str, _: &str) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            unreachable!()
        }
    }
    fn scorer(statuses: Vec<u16>, delay: Duration) -> (SideScorer<StubHttp>, Arc<AtomicUsize>) {
        let launches = Arc::new(AtomicUsize::new(0));
        (SideScorer::new(Box::new(Stub { launches: launches.clone(), statuses, delay }), StubHttp), launches)
    }
    /// Score, waiting out warm-up refusals as a gateway would.
    fn score_patiently(s: &SideScorer<StubHttp>, path: &str) -> Result<Vec<f32>, AdapterError> {
        for _ in 0..50 {
            match s.score(Path::new(path), &[1, 2, 3]) {
                Err(AdapterError::Busy(_)) => std::thread::sleep(Duration::from_millis(20)),
                other => return other,
            }
        }
        panic!("never became ready")
    }

    #[test]
    fn the_first_start_mode_whose_scoring_fits_is_kept() {
        // GPU defaults: scoring disabled (501); the next mode fits (405).
        let (s, launches) = scorer(vec![501, 405], Duration::ZERO);
        assert_eq!(score_patiently(&s, "/m.gguf").unwrap(), vec![-0.5, -0.5]);
        assert_eq!(launches.load(Ordering::SeqCst), 2);
        // Warm now: no new launch.
        assert!(s.score(Path::new("/m.gguf"), &[1, 2]).is_ok());
        assert_eq!(launches.load(Ordering::SeqCst), 2);
        // Another file replaces it.
        let _ = score_patiently(&s, "/other.gguf");
        assert!(launches.load(Ordering::SeqCst) > 2);
    }

    #[test]
    fn a_slow_start_is_busy_not_a_failure_and_a_model_that_never_fits_is_unsupported() {
        let (s, _) = scorer(vec![405], Duration::from_millis(30));
        // CHALLENGE_WAIT (20 s) is longer than this start, so the first call simply waits.
        assert!(s.score(Path::new("/m.gguf"), &[1, 2]).is_ok());
        let (s, launches) = scorer(vec![501, 501, 501], Duration::ZERO);
        assert!(matches!(score_patiently(&s, "/big.gguf"), Err(AdapterError::Unsupported(_))));
        let tried = launches.load(Ordering::SeqCst);
        assert_eq!(tried, attempts(Path::new("/big.gguf")).len());
        // Not retried right away.
        assert!(matches!(s.score(Path::new("/big.gguf"), &[1, 2]), Err(AdapterError::Unsupported(_))));
        assert_eq!(launches.load(Ordering::SeqCst), tried);
    }

    #[test]
    fn the_wrapper_scores_only_what_the_engine_cannot() {
        struct Engine(Option<PathBuf>, bool);
        impl EngineAdapter for Engine {
            fn engine_name(&self) -> &'static str {
                "stub"
            }
            fn detect_models(&self) -> Result<Vec<DetectedModel>, AdapterError> {
                Ok(vec![])
            }
            fn serve_stream(&self, _: &InferenceRequest, _: &mut dyn FnMut(&str)) -> Result<ServeOutcome, AdapterError> {
                unreachable!()
            }
            fn score_prompt(&self, _: &str, ids: &[u32]) -> Result<Vec<f32>, AdapterError> {
                if self.1 {
                    Ok(vec![-1.0; ids.len() - 1])
                } else {
                    Err(AdapterError::Unsupported("no".into()))
                }
            }
            fn weights_path(&self, _: &str) -> Option<PathBuf> {
                self.0.clone()
            }
        }
        // No side-scorer configured: plain pass-through.
        let w = SideScored::new(Engine(Some("/m.gguf".into()), false), None);
        assert!(matches!(w.score_prompt("m", &[1, 2]), Err(AdapterError::Unsupported(_))));
        // An engine that scores itself is never side-scored.
        let w = SideScored::new(Engine(None, true), None);
        assert_eq!(w.score_prompt("m", &[1, 2]).unwrap(), vec![-1.0]);
    }
}
