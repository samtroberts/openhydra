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
//!   (patched mistral.rs, `scorer/mistralrs`) for an MLX model directory (LM Studio's MLX engine),
//!   with the same scoring endpoint.
//! * GPU when it fits: it starts with the engine's defaults (GPU offload, sized to free memory), and
//!   the scorer's own startup check disables scoring when the scoring buffers don't fit next to the
//!   model — then a GGUF scorer is restarted without the memory-mapped load (Apple Silicon maps the
//!   whole file into GPU memory) and finally on the CPU alone. MLX runs on the GPU only.
//! * What a pass proves is unchanged: the provider holds the claimed weights file and computes with
//!   it. It doesn't prove the engine serves that file (challenges are recognisable either way).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use crate::adapter::{AdapterError, DetectedModel, EngineAdapter, HttpClient, InferenceRequest, ServeOutcome};
use crate::adapters::scoring::{score_prompt, ScoringDialect};

/// File name of the bundled GGUF scorer (OpenHydra's patched llama-server), next to the agent.
pub const SCORER_BIN: &str = "openhydra-scorer";
/// File name of the bundled MLX scorer (patched mistral.rs, `scorer/mistralrs`, macOS), next to the agent.
pub const MLX_SCORER_BIN: &str = "openhydra-mlx-scorer";
/// Stop the scorer after this long without a challenge.
pub const IDLE_STOP: Duration = Duration::from_secs(15 * 60);
/// How long a challenge waits for a starting scorer before it is told to retry.
const CHALLENGE_WAIT: Duration = Duration::from_secs(20);
/// What a launcher reports when a scorer process started but never answered.
const START_TIMED_OUT: &str = "the scorer didn't come up in time";

/// After every way of starting a file's scorer failed, don't try that file again for this long.
const FAILED_BACKOFF: Duration = Duration::from_secs(30 * 60);
/// A scorer used this recently is not replaced by another file's: a gateway's round sends its texts
/// back to back, and switching files under it would end the round early — held against the
/// provider. The other file's challenge is told to retry (its gateway waits several minutes).
const SWITCH_HOLD: Duration = if cfg!(test) { Duration::from_millis(200) } else { Duration::from_secs(30) };

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
    /// `attempt` counts the start-up modes tried for this start (0 = the first).
    fn launch(&self, weights: &Path, extra: &[String], attempt: usize) -> Result<Box<dyn ScorerInstance>, String>;
    /// Whether there is a scorer for this kind of weights at all (no MLX scorer off macOS, …).
    fn can_score(&self, _weights: &Path) -> bool {
        true
    }
}

enum State {
    Idle,
    Starting(PathBuf),
    /// `active`: challenges being scored right now; the process is never replaced under one.
    /// `stamp`: the weights' size and modification time at start (a file replaced in place gets a
    /// new scorer). `errors`: scorings in a row the live process failed on its side.
    Ready { weights: PathBuf, stamp: Option<Stamp>, instance: Box<dyn ScorerInstance>, last_used: Instant, active: usize, errors: u32 },
}

/// Size and latest modification time of a weights file, or of an MLX directory's files.
type Stamp = (u64, std::time::SystemTime);

fn stamp(weights: &Path) -> Option<Stamp> {
    let files: Vec<PathBuf> = if weights.is_dir() {
        std::fs::read_dir(weights).ok()?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file()).collect()
    } else {
        vec![weights.to_path_buf()]
    };
    files.iter().try_fold((0u64, std::time::UNIX_EPOCH), |(size, newest), f| {
        let m = std::fs::metadata(f).ok()?;
        Some((size + m.len(), newest.max(m.modified().ok()?)))
    })
}

/// Scorings in a row a live scorer may fail on its own side (not the challenger's) before it is
/// stopped and its file backed off like a failed start — e.g. it ran out of memory after starting.
const MAX_SCORER_ERRORS: u32 = 3;

struct Inner<H> {
    launcher: Box<dyn ScorerLauncher>,
    http: H,
    state: Mutex<State>,
    changed: Condvar,
    /// Files whose scorer couldn't be started (or kept failing), when, and why; not retried for
    /// [`FAILED_BACKOFF`]. Per file, so a challenge for another model doesn't erase it.
    failed: Mutex<HashMap<PathBuf, (Instant, String)>>,
}

/// Manages one scorer at a time (one weights file); see the module docs.
pub struct SideScorer<H: HttpClient + Send + Sync + 'static> {
    inner: Arc<Inner<H>>,
}

impl<H: HttpClient + Send + Sync + 'static> SideScorer<H> {
    /// A side-scorer starting processes with `launcher` and talking to them with `http`. Starts a
    /// background thread that stops an idle scorer.
    pub fn new(launcher: Box<dyn ScorerLauncher>, http: H) -> Self {
        let inner = Arc::new(Inner {
            launcher,
            http,
            state: Mutex::new(State::Idle),
            changed: Condvar::new(),
            failed: Mutex::new(HashMap::new()),
        });
        let weak = Arc::downgrade(&inner);
        std::thread::spawn(move || idle_reaper(weak));
        Self { inner }
    }

    /// Score `token_ids` with a scorer on `weights` (starting or switching it as needed).
    /// [`AdapterError::Busy`] while it is still starting.
    pub fn score(&self, weights: &Path, token_ids: &[u32]) -> Result<Vec<f32>, AdapterError> {
        if weights.is_dir() {
            // mistral.rs doesn't bound-check token ids, and these come from the network: one past
            // the embedding table would be an out-of-bounds read on the GPU.
            let vocab = mlx_vocab_size(weights).ok_or_else(|| AdapterError::Unsupported("model config has no vocab_size".into()))?;
            if token_ids.iter().any(|&id| id >= vocab) {
                return Err(AdapterError::Unsupported("token id outside the model's vocabulary".into()));
            }
        }
        let base = self.ready(weights)?;
        let out = score_prompt(&self.inner.http, &base, ScoringDialect::OpenHydra, "default", token_ids);
        let mut st = self.inner.state.lock().unwrap_or_else(|e| e.into_inner());
        if let State::Ready { weights: w, instance, active, last_used, errors, .. } = &mut *st {
            if w == weights {
                *active = active.saturating_sub(1);
                *last_used = Instant::now();
                match &out {
                    Ok(_) => *errors = 0,
                    // A request the scorer rejected (4xx) is the challenger's doing, not the scorer's.
                    Err(AdapterError::Http(m)) if m.contains("client error") => {}
                    Err(_) => *errors += 1,
                }
            }
            if out.is_err() && !instance.alive() {
                *st = State::Idle; // a dead process is replaced on the next challenge
            } else if matches!(&*st, State::Ready { active: 0, errors, .. } if *errors >= MAX_SCORER_ERRORS) {
                tracing::warn!(weights = %weights.display(), "the prompt scorer keeps failing; stopping it");
                self.inner.failed.lock().unwrap_or_else(|e| e.into_inner()).insert(
                    weights.to_path_buf(),
                    (Instant::now(), "the prompt scorer kept failing (out of memory?)".into()),
                );
                *st = State::Idle;
            }
        }
        out
    }

    /// The base URL of a ready scorer on `weights`, starting one if needed.
    fn ready(&self, weights: &Path) -> Result<String, AdapterError> {
        let inner = &self.inner;
        if !inner.launcher.can_score(weights) {
            return Err(AdapterError::Unsupported("no prompt scorer for this kind of model here".into()));
        }
        if let Some((at, why)) = inner.failed.lock().unwrap_or_else(|e| e.into_inner()).get(weights) {
            if at.elapsed() < FAILED_BACKOFF {
                return Err(AdapterError::Unsupported(why.clone()));
            }
        }
        let current = stamp(weights);
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
                State::Ready { weights: w, stamp, instance, last_used, active, .. } if w == weights && *stamp == current => {
                    *last_used = Instant::now();
                    *active += 1;
                    return Ok(instance.base_url().to_string());
                }
                State::Ready { active, last_used, .. } if *active > 0 || last_used.elapsed() < SWITCH_HOLD => {
                    // Another file's scorer (or this file's, replaced on disk) in use: wait for it.
                    return Err(AdapterError::Busy("prompt scorer is busy with another model".into()));
                }
                State::Starting(w) if w == weights => {
                    let now = Instant::now();
                    if now >= deadline {
                        return Err(AdapterError::Busy("prompt scorer is still starting".into()));
                    }
                    st = inner.changed.wait_timeout(st, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
                    if let Some((_, why)) = inner.failed.lock().unwrap_or_else(|e| e.into_inner()).get(weights) {
                        if matches!(&*st, State::Idle) {
                            return Err(AdapterError::Unsupported(why.clone())); // this start just failed
                        }
                    }
                }
                State::Starting(_) => {
                    // Another file's scorer is starting; this one waits its turn.
                    return Err(AdapterError::Busy("prompt scorer is busy with another model".into()));
                }
                _ => {
                    // Idle, or an idle scorer on another file (or this file before it changed): start.
                    *st = State::Starting(weights.to_path_buf());
                    let (inner2, path) = (Arc::clone(inner), weights.to_path_buf());
                    std::thread::spawn(move || start(&inner2, &path, current));
                }
            }
        }
    }
}

/// Bring up a scorer on `weights`, trying each start-up mode until one keeps scoring enabled.
fn start<H: HttpClient>(inner: &Inner<H>, weights: &Path, stamp: Option<Stamp>) {
    let started = Instant::now();
    let (mut outcome, mut didnt_fit, mut last_error) = (None, false, String::new());
    for (attempt, extra) in attempts(weights).into_iter().enumerate() {
        match inner.launcher.launch(weights, &extra, attempt) {
            // The llama.cpp scorer's own startup check answers GET with 405 when scoring is on, 501
            // when it had to disable it (the scoring buffers don't fit next to the model). The MLX
            // scorer has no such check, so it is done here: the longest accepted prompt must score.
            Ok(instance)
                if inner.http.get_status(&format!("{}/v1/prompt_logprobs", instance.base_url())).ok() == Some(405)
                    && (!weights.is_dir()
                        || score_prompt(&inner.http, instance.base_url(), ScoringDialect::OpenHydra, "default", &probe_prompt()).is_ok()) =>
            {
                tracing::info!(weights = %weights.display(), mode = ?extra, secs = started.elapsed().as_secs(), "prompt scorer ready");
                outcome = Some(instance);
                break;
            }
            Ok(_) => {
                didnt_fit = true;
                tracing::info!(weights = %weights.display(), mode = ?extra, "prompt scoring didn't fit this way; trying the next");
            }
            Err(e) => {
                tracing::warn!(weights = %weights.display(), mode = ?extra, error = %e, "prompt scorer failed to start");
                // A hang isn't a memory fit the next mode could fix; trying it too would keep every
                // other model's challenges waiting ("busy") for several start timeouts in a row.
                let hung = e == START_TIMED_OUT;
                last_error = e;
                if hung {
                    break;
                }
            }
        }
    }
    let mut st = inner.state.lock().unwrap_or_else(|e| e.into_inner());
    *st = match outcome {
        Some(instance) => {
            inner.failed.lock().unwrap_or_else(|e| e.into_inner()).remove(weights);
            State::Ready { weights: weights.to_path_buf(), stamp, instance, last_used: Instant::now(), active: 0, errors: 0 }
        }
        None => {
            let why = if didnt_fit {
                "prompt scoring doesn't fit in memory next to this model".to_string()
            } else {
                format!("the prompt scorer failed to start: {last_error}")
            };
            tracing::warn!(weights = %weights.display(), reason = %why, "prompt scoring unavailable for this model");
            inner.failed.lock().unwrap_or_else(|e| e.into_inner()).insert(weights.to_path_buf(), (Instant::now(), why));
            State::Idle
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
        if matches!(&*st, State::Ready { last_used, active: 0, .. } if last_used.elapsed() >= IDLE_STOP) {
            tracing::info!("stopping the idle prompt scorer");
            *st = State::Idle; // dropping the instance stops the process
        }
    }
}

// ── the real scorer process ──

/// Starts a scorer as a child process on a free local port: `openhydra-scorer` (a patched
/// llama-server) for a GGUF file, `openhydra-mlx-scorer` (patched mistral.rs) for an MLX model
/// directory ([`scorer_args`]); both run with `--exit-on-stdin-eof`.
pub struct ProcessLauncher {
    gguf: Option<PathBuf>,
    mlx: Option<PathBuf>,
    /// How long a start may take (loading a multi-GB model on the CPU is slow).
    ready_timeout: Duration,
}

impl ProcessLauncher {
    pub fn new(gguf: Option<PathBuf>, mlx: Option<PathBuf>) -> Self {
        // A 9B model on a CPU-only Linux box took 120 s to start with the file cached and over
        // 300 s from cold network storage (the startup check runs a full 512-token pass). A
        // challenge never waits on this: it is told to retry while the scorer starts.
        Self { gguf, mlx, ready_timeout: Duration::from_secs(900) }
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
    fn launch(&self, weights: &Path, extra: &[String], attempt: usize) -> Result<Box<dyn ScorerInstance>, String> {
        let binary = self.binary_for(weights).ok_or_else(|| {
            format!("no {} available for {}", if weights.is_dir() { MLX_SCORER_BIN } else { SCORER_BIN }, weights.display())
        })?;
        let log_path = scorer_log_path();
        // Each start's log holds all of its modes (the first truncates); one log per agent process.
        let mut log = open_log(&log_path, attempt == 0).map_err(|e| format!("{}: {e}", log_path.display()))?;
        // The port is picked by binding and releasing it, so another process can take it before the
        // scorer binds: then the scorer exits saying so, and another port is tried.
        for _ in 0..3 {
            let port = std::net::TcpListener::bind("127.0.0.1:0").and_then(|l| l.local_addr()).map_err(|e| e.to_string())?.port();
            use std::io::Write;
            let _ = writeln!(log, "== openhydra-agent: starting {} {:?} on port {port}", binary.display(), extra);
            let mut child = std::process::Command::new(binary)
                .args(scorer_args(weights, port, extra))
                .stdin(std::process::Stdio::piped())
                .stdout(log.try_clone().map_err(|e| e.to_string())?)
                .stderr(log.try_clone().map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| format!("{}: {e}", binary.display()))?;
            let lifeline = child.stdin.take();
            let mut instance = ProcessInstance { child, _lifeline: lifeline, base: format!("http://127.0.0.1:{port}") };
            let http = crate::http::ReqwestClient::new().map_err(|e| e.to_string())?;
            let deadline = Instant::now() + self.ready_timeout;
            let mut exited = false;
            while Instant::now() < deadline {
                if !instance.alive() {
                    exited = true;
                    break;
                }
                if http.get(&format!("{}/v1/models", instance.base)).is_ok() {
                    return Ok(Box::new(instance));
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            if !exited {
                return Err(START_TIMED_OUT.into());
            }
            let tail = std::fs::read_to_string(&log_path).unwrap_or_default();
            let tail = tail.rsplit("== openhydra-agent: starting").next().unwrap_or_default();
            if !(tail.contains("in use") || tail.contains("couldn't bind") || tail.contains("bind HTTP")) {
                return Err(format!("the scorer exited while loading (see {})", log_path.display()));
            }
        }
        Err("no free local port for the scorer".into())
    }

    fn can_score(&self, weights: &Path) -> bool {
        self.binary_for(weights).is_some()
    }
}

impl ProcessLauncher {
    fn binary_for(&self, weights: &Path) -> Option<&PathBuf> {
        if weights.is_dir() {
            self.mlx.as_ref()
        } else {
            self.gguf.as_ref()
        }
    }
}

/// The scorer's log: in the temp directory, named for this agent process (two agents on one machine
/// don't share one).
pub fn scorer_log_path() -> PathBuf {
    std::env::temp_dir().join(format!("openhydra-scorer-{}.log", std::process::id()))
}

/// Open the log for appending (`truncate`: start it afresh), never following a symlink planted at
/// that path (the temp directory is shared with other users on Linux).
fn open_log(path: &Path, truncate: bool) -> std::io::Result<std::fs::File> {
    let mut opts = std::fs::OpenOptions::new();
    opts.create(true).write(true);
    if truncate {
        opts.truncate(true);
    } else {
        opts.append(true);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.custom_flags(libc::O_NOFOLLOW).mode(0o600);
    }
    let mut f = opts.open(path)?;
    if !truncate {
        use std::io::Seek;
        f.seek(std::io::SeekFrom::End(0))?;
    }
    Ok(f)
}

/// Whether this CPU can run the GGUF scorer. Its x86 build targets AVX2-class CPUs (ggml's portable
/// default: AVX, AVX2, FMA, F16C, BMI2 — Intel Haswell / AMD Excavator and later); on an older one
/// it would die of an illegal instruction on every start. Other architectures: yes.
pub fn cpu_runs_gguf_scorer() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("bmi2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

/// The command line for a scorer on `weights`, listening on `port` (localhost only), with `extra`
/// start-mode arguments. A GGUF file: the patched llama-server. An MLX model directory: patched
/// mistral.rs (`serve`), with its web UI off.
fn scorer_args(weights: &Path, port: u16, extra: &[String]) -> Vec<std::ffi::OsString> {
    let port = port.to_string();
    let mut args: Vec<std::ffi::OsString> = if weights.is_dir() {
        ["serve", "-p", port.as_str(), "--host", "127.0.0.1", "--no-ui", "--exit-on-stdin-eof", "-m"].map(Into::into).to_vec()
    } else {
        // -c only sizes the serving slot, which a scorer never uses (scoring makes its own context)
        ["--host", "127.0.0.1", "--port", port.as_str(), "--prompt-scoring", "--exit-on-stdin-eof", "-c", "512", "-np", "1", "-m"]
            .map(Into::into)
            .to_vec()
    };
    args.push(weights.as_os_str().to_owned());
    args.extend(extra.iter().map(Into::into));
    args
}

/// A synthetic prompt of the longest accepted length, for the MLX scorer's startup check (low token
/// ids, valid in any vocabulary).
fn probe_prompt() -> Vec<u32> {
    (0..crate::prompt_score::MAX_CHALLENGE_TOKENS as u32).map(|i| 1000 + (i * 37) % 20000).collect()
}

/// The vocabulary size in an MLX model directory's `config.json` (top level, or under `text_config`
/// for multimodal checkpoints).
fn mlx_vocab_size(dir: &Path) -> Option<u32> {
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    let v = cfg.get("vocab_size").or_else(|| cfg.get("text_config").and_then(|t| t.get("vocab_size")))?;
    u32::try_from(v.as_u64()?).ok()
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
        /// Status the n-th launched instance answers GET /v1/prompt_logprobs with (0: it fails to start).
        statuses: Vec<u16>,
        delay: Duration,
        /// Whether a scorer exists for a path containing "nokind" (`can_score`).
        nokind: bool,
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
        fn launch(&self, _: &Path, _: &[String], _: usize) -> Result<Box<dyn ScorerInstance>, String> {
            std::thread::sleep(self.delay);
            let n = self.launches.fetch_add(1, Ordering::SeqCst);
            match self.statuses.get(n).copied().unwrap_or(501) {
                0 => Err("boom".into()),
                1 => Err(START_TIMED_OUT.into()),
                status => Ok(Box::new(StubInstance(format!("http://stub/{status}")))),
            }
        }
        fn can_score(&self, weights: &Path) -> bool {
            self.nokind || !weights.to_string_lossy().contains("nokind")
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
            // Token 998: the scorer rejects the request; 999: it fails on its side.
            match v["prompt_token_ids"][0].as_u64() {
                Some(998) => return Err(AdapterError::Http("HTTP status client error (400 Bad Request)".into())),
                Some(999) => return Err(AdapterError::Http("HTTP status server error (503 Service Unavailable)".into())),
                _ => {}
            }
            let n = v["prompt_token_ids"].as_array().unwrap().len();
            Ok(serde_json::json!({"scores": {"token_ids": v["prompt_token_ids"], "logprobs": vec![-0.5; n - 1]}}).to_string())
        }
        fn post_stream(&self, _: &str, _: &str) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
            unreachable!()
        }
    }
    fn scorer(statuses: Vec<u16>, delay: Duration) -> (SideScorer<StubHttp>, Arc<AtomicUsize>) {
        let launches = Arc::new(AtomicUsize::new(0));
        (SideScorer::new(Box::new(Stub { launches: launches.clone(), statuses, delay, nokind: false }), StubHttp), launches)
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
        // Another file waits while this one is in use (a round in progress), then replaces it.
        assert!(matches!(s.score(Path::new("/other.gguf"), &[1, 2]), Err(AdapterError::Busy(m)) if m.contains("another model")));
        assert_eq!(launches.load(Ordering::SeqCst), 2);
        std::thread::sleep(SWITCH_HOLD);
        let _ = score_patiently(&s, "/other.gguf");
        assert!(launches.load(Ordering::SeqCst) > 2);
    }

    #[test]
    fn a_failed_file_is_remembered_with_its_reason_while_other_files_are_scored() {
        let n = attempts(Path::new("/big.gguf")).len();
        let mut statuses = vec![501; n];
        statuses.push(405); // the next file's scorer fits
        let (s, launches) = scorer(statuses, Duration::ZERO);
        assert!(matches!(score_patiently(&s, "/big.gguf"), Err(AdapterError::Unsupported(m)) if m.contains("doesn't fit")));
        assert!(score_patiently(&s, "/small.gguf").is_ok());
        // Scoring another file didn't erase the failure: still refused, nothing relaunched.
        assert!(matches!(s.score(Path::new("/big.gguf"), &[1, 2]), Err(AdapterError::Unsupported(_))));
        assert_eq!(launches.load(Ordering::SeqCst), n + 1);
        // A scorer that can't start at all says so (not "doesn't fit").
        let (s, _) = scorer(vec![0; n], Duration::ZERO);
        assert!(matches!(score_patiently(&s, "/m.gguf"), Err(AdapterError::Unsupported(m)) if m.contains("failed to start: boom")));
    }

    #[test]
    fn a_scorer_that_hangs_on_start_is_not_retried_in_other_modes() {
        // The first mode hangs (timed out); the next modes are not tried, so the start fails once.
        let (s, launches) = scorer(vec![1, 405, 405], Duration::ZERO);
        assert!(matches!(score_patiently(&s, "/m.gguf"), Err(AdapterError::Unsupported(m)) if m.contains("in time")));
        assert_eq!(launches.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_kind_of_model_without_a_scorer_leaves_the_running_one_alone() {
        let (s, launches) = scorer(vec![405], Duration::ZERO);
        assert!(score_patiently(&s, "/m.gguf").is_ok());
        std::thread::sleep(SWITCH_HOLD); // idle enough to be replaced, if it were
        assert!(matches!(s.score(Path::new("/nokind"), &[1, 2]), Err(AdapterError::Unsupported(_))));
        assert!(s.score(Path::new("/m.gguf"), &[1, 2]).is_ok(), "still the same scorer");
        assert_eq!(launches.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_scorer_failing_on_its_own_side_is_stopped_but_bad_requests_dont_count() {
        let (s, launches) = scorer(vec![405, 405], Duration::ZERO);
        assert!(score_patiently(&s, "/m.gguf").is_ok());
        for _ in 0..MAX_SCORER_ERRORS + 2 {
            assert!(s.score(Path::new("/m.gguf"), &[998, 1]).is_err()); // a challenger's bad request
        }
        assert!(s.score(Path::new("/m.gguf"), &[1, 2]).is_ok(), "bad requests don't stop the scorer");
        for _ in 0..MAX_SCORER_ERRORS {
            assert!(s.score(Path::new("/m.gguf"), &[999, 1]).is_err());
        }
        assert!(matches!(s.score(Path::new("/m.gguf"), &[1, 2]), Err(AdapterError::Unsupported(m)) if m.contains("kept failing")));
        assert_eq!(launches.load(Ordering::SeqCst), 1, "backed off, not restarted");
    }

    #[test]
    fn weights_replaced_in_place_get_a_new_scorer() {
        let dir = std::env::temp_dir().join(format!("oh-stamp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("m.gguf");
        std::fs::write(&file, b"v1").unwrap();
        let (s, launches) = scorer(vec![405, 405], Duration::ZERO);
        assert!(score_patiently(&s, file.to_str().unwrap()).is_ok());
        std::fs::write(&file, b"version 2").unwrap();
        std::thread::sleep(SWITCH_HOLD);
        assert!(score_patiently(&s, file.to_str().unwrap()).is_ok());
        assert_eq!(launches.load(Ordering::SeqCst), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_log_keeps_every_mode_of_a_start_and_never_follows_a_planted_link() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("oh-log-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("scorer.log");
        writeln!(open_log(&log, true).unwrap(), "first mode").unwrap();
        writeln!(open_log(&log, false).unwrap(), "second mode").unwrap();
        assert_eq!(std::fs::read_to_string(&log).unwrap(), "first mode\nsecond mode\n");
        writeln!(open_log(&log, true).unwrap(), "next start").unwrap();
        assert_eq!(std::fs::read_to_string(&log).unwrap(), "next start\n");
        #[cfg(unix)]
        {
            let target = dir.join("victim");
            std::fs::write(&target, b"keep").unwrap();
            let link = dir.join("planted.log");
            std::os::unix::fs::symlink(&target, &link).unwrap();
            assert!(open_log(&link, true).is_err());
            assert_eq!(std::fs::read(&target).unwrap(), b"keep");
        }
        assert!(scorer_log_path().to_string_lossy().contains(&std::process::id().to_string()));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_scorer_is_never_replaced_while_it_scores() {
        let (s, launches) = scorer(vec![405], Duration::ZERO);
        let base = s.ready(Path::new("/a.gguf")).unwrap(); // a challenge for A is being scored
        assert!(base.starts_with("http://stub/"));
        std::thread::sleep(SWITCH_HOLD);
        assert!(matches!(s.score(Path::new("/b.gguf"), &[1, 2]), Err(AdapterError::Busy(_))));
        assert_eq!(launches.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn mlx_token_ids_outside_the_vocabulary_never_reach_the_scorer() {
        let dir = std::env::temp_dir().join(format!("oh-mlx-vocab-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Multimodal checkpoints keep it under text_config.
        std::fs::write(dir.join("config.json"), r#"{"text_config": {"vocab_size": 100}}"#).unwrap();
        let (s, launches) = scorer(vec![405], Duration::ZERO);
        assert!(matches!(s.score(&dir, &[1, 100]), Err(AdapterError::Unsupported(m)) if m.contains("vocabulary")));
        assert_eq!(launches.load(Ordering::SeqCst), 0);
        std::fs::write(dir.join("config.json"), "{}").unwrap();
        assert!(matches!(s.score(&dir, &[1, 2]), Err(AdapterError::Unsupported(_))));
        assert_eq!(mlx_vocab_size(Path::new("/nonexistent")), None);
        std::fs::write(dir.join("config.json"), r#"{"vocab_size": 151936}"#).unwrap();
        assert_eq!(mlx_vocab_size(&dir), Some(151936));
        std::fs::remove_dir_all(&dir).ok();
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
    fn each_kind_of_weights_gets_its_scorers_command_line_on_localhost() {
        let s = |v: Vec<std::ffi::OsString>| v.into_iter().map(|a| a.into_string().unwrap()).collect::<Vec<_>>().join(" ");
        let dir = std::env::temp_dir().join(format!("oh-mlx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mlx = s(scorer_args(&dir, 4242, &[]));
        assert_eq!(mlx, format!("serve -p 4242 --host 127.0.0.1 --no-ui --exit-on-stdin-eof -m {}", dir.display()));
        let gguf = s(scorer_args(Path::new("/m/x.gguf"), 4242, &["-ngl".into(), "0".into()]));
        assert_eq!(gguf, "--host 127.0.0.1 --port 4242 --prompt-scoring --exit-on-stdin-eof -c 512 -np 1 -m /m/x.gguf -ngl 0");
        std::fs::remove_dir_all(&dir).ok();
        let probe = probe_prompt();
        assert_eq!(probe.len(), crate::prompt_score::MAX_CHALLENGE_TOKENS);
        assert!(probe.iter().all(|&t| (1000..21000).contains(&t)));
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
