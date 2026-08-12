// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! `openhydra-agent` — the runnable pure-protocol host.
//!
//! Two roles, one binary (the single static daemon the plan targets):
//!
//! * **provide** — join the swarm, detect the operator's local engine (Ollama) models,
//!   advertise their canonical ids, and serve inbound inference by proxying to the engine.
//!   Never runs a model itself.
//! * **serve** — the consumer front door: an OpenAI-compatible HTTP/SSE gateway that
//!   discovers a provider for each request, streams the completion over libp2p, and
//!   settles a co-signed receipt at EOS.
//!
//! Both roles drive the swarm through [`NetworkHandle`] synchronously (no tokio in the
//! libp2p path), so `main` is a plain function — the gateway builds its own runtime
//! internally. A single node can't yet be both provider and gateway at once
//! (`NetworkHandle` is owned by one role); run two processes for the M3.1 two-node test.

use std::path::PathBuf;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};

use openhydra_agent::{
    live_comfyui, live_comfyui_with_workflows, live_exo, live_llamacpp, live_ollama, live_openai,
    serve_http, AupPolicy,
    ByokConfig, EconomyStats, EmbeddingConfig, EngineAdapter, Provider, RateLimitConfig,
    StatusServer, TransferStats, DEFAULT_ANTHROPIC_URL, DEFAULT_COMFYUI_URL, DEFAULT_EXO_URL,
    DEFAULT_GEMINI_URL, DEFAULT_LLAMACPP_URL, DEFAULT_LM_STUDIO_URL, DEFAULT_OLLAMA_URL,
    DEFAULT_OPENAI_EMBEDDINGS_URL, DEFAULT_VLLM_URL,
};
use openhydra_network::handle::NetworkHandle;
use openhydra_network::node::NodeConfig;
use openhydra_protocol::store::Store;

/// OpenHydra agent — a gateway in front of whatever inference engine you already run.
#[derive(Parser)]
#[command(name = "openhydra-agent", version, about)]
struct Cli {
    #[command(flatten)]
    node: NodeArgs,

    #[command(subcommand)]
    role: Role,
}

/// Swarm-node flags shared by both roles. Omitted flags fall back to [`NodeConfig`]'s
/// defaults (per-user identity file, dual-stack TCP+QUIC on :4001, no bootstrap peers).
#[derive(Args)]
struct NodeArgs {
    /// Path to the ed25519 identity key (created if absent).
    #[arg(long)]
    identity: Option<PathBuf>,

    /// libp2p listen multiaddr (repeatable). Defaults to dual-stack TCP+QUIC on :4001.
    #[arg(long = "listen")]
    listen: Vec<String>,

    /// Bootstrap peer multiaddr to dial on start (repeatable). On a LAN, mDNS discovers
    /// peers without this; across networks, point at a known peer/relay.
    #[arg(long = "bootstrap")]
    bootstrap: Vec<String>,

    /// Opt into acting as a temporary circuit-relay for other peers (off by default).
    #[arg(long)]
    peer_relay: bool,

    /// Experimental: Tier-2 connection reversal — when this node holds only a relayed
    /// connection to a peer that advertises globally-routable direct addresses, dial
    /// those directly to escape the relay (off by default). The one NAT escape that
    /// works on symmetric CGNAT. See docs/PEER_CONNECTIVITY.md.
    #[arg(long)]
    connection_reversal: bool,

    /// Experimental (#43-W2): CPE gateway IP for the PCP (RFC 6887) inbound-v6
    /// firewall-pinhole maintainer. Off unless set. When your router speaks PCP,
    /// this opens the listen ports inbound on your global IPv6 so AutoNAT can
    /// confirm reachability and promote the node to a relay/server — the v6
    /// sibling of the v4 UPnP/NAT-PMP mapping. See docs/IPV6_REACHABILITY.md.
    #[arg(long = "pcp-gateway")]
    pcp_gateway: Option<std::net::IpAddr>,

    /// Serve a read-only introspection endpoint (peers / DHT / swarm / transfer
    /// counters as JSON) on this address, e.g. `127.0.0.1:9464` or `127.0.0.1:0`
    /// (ephemeral). Off unless set; bind loopback unless you mean to expose it. If
    /// `OPENHYDRA_STATUS_TOKEN` is set, requests must send it as a bearer token.
    #[arg(long = "status-bind")]
    status_bind: Option<String>,
}

impl NodeArgs {
    /// Build a [`NodeConfig`], overriding defaults only where a flag was given.
    fn into_config(self) -> NodeConfig {
        let mut config = NodeConfig {
            enable_peer_relay: self.peer_relay,
            enable_connection_reversal: self.connection_reversal,
            bootstrap_peers: self.bootstrap,
            pcp_gateway: self.pcp_gateway,
            ..NodeConfig::default()
        };
        if let Some(identity) = self.identity {
            config.identity_path = identity;
        }
        if !self.listen.is_empty() {
            config.listen_addrs = self.listen;
        }
        config
    }
}

#[derive(Subcommand)]
enum Role {
    /// Advertise a local engine's models and serve inbound inference requests.
    Provide(ProvideArgs),
    /// Run the consumer HTTP/SSE gateway (the front door for OpenAI-compatible clients).
    Serve(ServeArgs),
    /// Print the libp2p PeerId for the `--identity` key (creating it if absent) and exit —
    /// no swarm. Used to wire `serve --self-provider <id>` when one machine both provides and
    /// consumes (#7 self-serve credit skip).
    PeerId,
}

/// Which local engine an agent proxies to. Selects the adapter; the `--engine` URL
/// defaults to the kind's standard port when omitted.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum EngineKind {
    /// Auto-detect (the default): probe the standard ports, serve **every** engine found
    /// (the union of their models), and — with `--engine-autostart` — start one if none is
    /// up. Zero-config for the common case where you just run one engine.
    Auto,
    /// Ollama native API (`/api/*`) — rich metadata, full canonical ids.
    Ollama,
    /// vLLM (OpenAI-compatible `/v1/*`).
    Vllm,
    /// LM Studio (OpenAI-compatible `/v1/*`).
    LmStudio,
    /// llama.cpp (`llama-server`) — OpenAI serve route + `/props` canonical-id detection.
    LlamaCpp,
    /// Exo MLX cluster — OpenAI serve route + `/state` detection (announces only
    /// placed-and-ready models, not Exo's whole downloadable catalog).
    Exo,
    /// ComfyUI (image generation) — announces Stable-Diffusion checkpoints as models;
    /// serves txt2img, returning the image as a data-URL in the completion. Steps = tokens.
    Comfyui,
    /// Any other OpenAI-compatible server (LocalAI, …).
    Openai,
}

impl EngineKind {
    /// The engine's conventional local base URL, used when `--engine` is omitted.
    fn default_url(self) -> &'static str {
        match self {
            // Auto never resolves a single URL (it scans standard ports and branches to
            // `provide_auto` before this is called); map it to Ollama's for exhaustiveness.
            EngineKind::Auto | EngineKind::Ollama => DEFAULT_OLLAMA_URL,
            EngineKind::Vllm | EngineKind::Openai => DEFAULT_VLLM_URL,
            EngineKind::LmStudio => DEFAULT_LM_STUDIO_URL,
            EngineKind::LlamaCpp => DEFAULT_LLAMACPP_URL,
            EngineKind::Exo => DEFAULT_EXO_URL,
            EngineKind::Comfyui => DEFAULT_COMFYUI_URL,
        }
    }
}

/// Acceptable-use policy limits (the AUP floor). Every limit defaults to off (permissive);
/// set any to opt in. Shared by `provide` (applied to inbound serve requests from the open
/// network — the security-critical point) and `serve` (the gateway front door).
#[derive(Args, Clone)]
struct AupArgs {
    /// Reject a request with more than this many messages (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    aup_max_messages: usize,

    /// Reject a request whose total prompt exceeds this many characters (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    aup_max_prompt_chars: usize,

    /// Reject a request whose explicit `max_tokens` exceeds this (0 = unlimited).
    #[arg(long, default_value_t = 0)]
    aup_max_completion_tokens: u32,

    /// Refuse any request containing this (case-insensitive) substring. Repeatable.
    #[arg(long = "aup-deny")]
    aup_deny: Vec<String>,
}

impl AupArgs {
    fn into_policy(self) -> AupPolicy {
        AupPolicy {
            max_messages: self.aup_max_messages,
            max_prompt_chars: self.aup_max_prompt_chars,
            max_completion_tokens: self.aup_max_completion_tokens,
            denied_substrings: self.aup_deny,
        }
    }
}

/// Ingress DoS rate-limit (the gateway's own front door — `serve` only). Every lever defaults
/// to off; concurrency is the primary one (each completion ties up a generation for seconds),
/// rps/burst the secondary anti-flood guard. Keyed by API key → socket IP (never a spoofable
/// header unless `--trusted-proxy`).
#[derive(Args, Clone)]
struct RateLimitArgs {
    /// Max concurrent in-flight requests per identity (0 = off — the primary lever).
    #[arg(long, default_value_t = 0)]
    rate_limit_max_inflight: u32,

    /// Sustained requests/sec per identity (0 = off).
    #[arg(long, default_value_t = 0)]
    rate_limit_rps: u32,

    /// Token-bucket burst capacity (0 → defaults to the rps value).
    #[arg(long, default_value_t = 0)]
    rate_limit_burst: u32,

    /// Hard cap on distinct tracked identities (memory bound; idle ones are evicted).
    #[arg(long, default_value_t = 10_000)]
    rate_limit_max_tracked: usize,

    /// Trust `X-Forwarded-For` for per-IP keying — only set this behind a reverse proxy you
    /// control, since the header is otherwise client-spoofable.
    #[arg(long)]
    trusted_proxy: bool,
}

impl RateLimitArgs {
    fn into_config(&self) -> RateLimitConfig {
        RateLimitConfig {
            rps: self.rate_limit_rps as f64,
            burst: self.rate_limit_burst as f64,
            max_inflight: self.rate_limit_max_inflight,
            max_tracked: self.rate_limit_max_tracked,
        }
    }
}

/// BYOK passthrough (#34, `serve` only): map specific model names to a hosted frontier
/// backend the gateway calls directly with the operator's key. Keys fall back to the
/// providers' standard env vars; a caller may override per-request via `X-Provider-Api-Key`.
#[derive(Args, Clone)]
struct ByokArgs {
    /// Route this model name to Anthropic (Claude). Repeatable.
    #[arg(long = "byok-anthropic-model")]
    byok_anthropic_model: Vec<String>,

    /// Route this model name to Google Gemini. Repeatable.
    #[arg(long = "byok-gemini-model")]
    byok_gemini_model: Vec<String>,

    /// Operator Anthropic key (falls back to `ANTHROPIC_API_KEY`).
    #[arg(long)]
    anthropic_key: Option<String>,

    /// Operator Gemini key (falls back to `GEMINI_API_KEY`).
    #[arg(long)]
    gemini_key: Option<String>,

    /// Override the Anthropic base URL (default `https://api.anthropic.com`; for proxies/tests).
    #[arg(long)]
    anthropic_url: Option<String>,

    /// Override the Gemini base URL (default Google's; for proxies/tests).
    #[arg(long)]
    gemini_url: Option<String>,

    /// Route this model name to the OpenAI-compatible embeddings backend (`/v1/embeddings`).
    /// Repeatable.
    #[arg(long = "byok-embedding-model")]
    byok_embedding_model: Vec<String>,

    /// Embeddings backend base URL (default OpenAI; point at a Gemini-OAI-compat / Voyage /
    /// local endpoint).
    #[arg(long)]
    embedding_url: Option<String>,

    /// Operator embeddings key (falls back to `OPENAI_API_KEY`).
    #[arg(long)]
    embedding_key: Option<String>,
}

impl ByokArgs {
    fn into_config(self) -> ByokConfig {
        let anthropic_key = self.anthropic_key.or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());
        let gemini_key = self.gemini_key.or_else(|| std::env::var("GEMINI_API_KEY").ok());
        ByokConfig::new(
            self.byok_anthropic_model,
            self.anthropic_url.unwrap_or_else(|| DEFAULT_ANTHROPIC_URL.to_string()),
            anthropic_key,
            self.byok_gemini_model,
            self.gemini_url.unwrap_or_else(|| DEFAULT_GEMINI_URL.to_string()),
            gemini_key,
        )
    }

    fn embedding_config(&self) -> EmbeddingConfig {
        let key = self.embedding_key.clone().or_else(|| std::env::var("OPENAI_API_KEY").ok());
        EmbeddingConfig::new(
            self.byok_embedding_model.clone(),
            self.embedding_url.clone().unwrap_or_else(|| DEFAULT_OPENAI_EMBEDDINGS_URL.to_string()),
            key,
        )
    }
}

#[derive(Args)]
struct ProvideArgs {
    /// Which local engine to proxy to. Defaults to `auto` (detect + serve whatever's running).
    #[arg(long = "engine-kind", value_enum, default_value_t = EngineKind::Auto)]
    engine_kind: EngineKind,

    /// Base URL of the local engine. Defaults to the engine kind's standard port
    /// (Ollama 11434, vLLM/OpenAI 8000, LM Studio 1234, llama.cpp 8080).
    #[arg(long)]
    engine: Option<String>,

    /// Directory of ComfyUI API-format workflow templates (BYO-workflow; `--engine-kind
    /// comfyui` only). Each `*.json` with a `%prompt%` marker is advertised as a model by
    /// filename, and the prompt/seed are injected at serve time — so any ComfyUI-supported
    /// model (Flux, SDXL, video, upscale chains) works with no code change. Omit for the
    /// built-in SD txt2img graph.
    #[arg(long = "comfyui-workflow-dir")]
    comfyui_workflow_dir: Option<PathBuf>,

    /// If the engine's server isn't already up, start it before announcing. Off by default.
    /// Only applies to engines OpenHydra can launch unattended — `ollama` (`ollama serve`)
    /// and `lm-studio` (`lms server start`, whose OpenAI server is a separate toggle from
    /// the app). vLLM / llama.cpp / Exo need a model or cluster argument, so those you still
    /// start yourself. Needs the `engine-autostart` build feature (on by default).
    #[arg(long)]
    engine_autostart: bool,

    /// Restrict which models to share, by engine handle (repeat, or comma-separate). Omit to
    /// share every model the engine exposes (the default). A model not in this list is neither
    /// announced nor served — a request for it is refused. E.g. `--share-models tinyllama:latest
    /// --share-models qwen3-vl:4b`.
    #[arg(long = "share-models", value_delimiter = ',')]
    share_models: Vec<String>,

    /// Advisory host advertised in records (routing is by libp2p peer id regardless).
    #[arg(long, default_value = "")]
    host: String,

    /// Advisory port advertised in records.
    #[arg(long, default_value_t = 0)]
    port: u16,

    /// Path to the receipt ledger (redb). Omit for an ephemeral in-memory ledger that
    /// does not survive a restart.
    #[arg(long)]
    db: Option<PathBuf>,

    /// Seconds between DHT re-announcements. Must stay below the relays' provider-record
    /// TTL (300s) or the node periodically falls out of discovery.
    #[arg(long, default_value_t = 120)]
    reannounce_secs: u64,

    /// Maximum concurrent serves. The poll loop hands each request to a worker pool of this
    /// size instead of serving inline, so one long generation can't head-of-line-block the
    /// rest. The external engine has its own concurrency limit; this just caps how many we
    /// dispatch at once. Default 8.
    #[arg(long, default_value_t = 8)]
    max_concurrency: usize,

    #[command(flatten)]
    aup: AupArgs,
}

#[derive(Args)]
struct ServeArgs {
    /// Address the HTTP/SSE gateway binds (loopback by default — no firewall prompt). #2:
    /// default off 8080 (llama.cpp's default) to avoid colliding with a locally-run engine.
    #[arg(long, default_value = "127.0.0.1:16527")]
    bind: String,

    /// Require this API key on `/v1/*` (`Authorization: Bearer <key>`). Omit — or set the
    /// `OPENHYDRA_API_KEY` env var — to leave the gateway open (fine on loopback).
    #[arg(long)]
    api_key: Option<String>,

    /// Persist earned provider reputation to this redb file (durable across restarts).
    /// Omit for an ephemeral, in-memory reputation.
    #[arg(long)]
    db: Option<PathBuf>,

    /// #7: this gateway's own co-located provider libp2p PeerId. When a request is served by
    /// this peer (the same machine provides *and* consumes), it's a self-serve — no receipt is
    /// settled and no give/take credit moves. Get it from `openhydra-agent peer-id --identity
    /// <provider-key>`. Omit on a consumer-only node.
    #[arg(long = "self-provider")]
    self_provider: Option<String>,

    #[command(flatten)]
    aup: AupArgs,

    #[command(flatten)]
    rate_limit: RateLimitArgs,

    #[command(flatten)]
    byok: ByokArgs,
}

fn main() {
    // PQC0.2: keep secret key material off disk (core dumps / swap) before anything
    // loads or generates the identity key. Best-effort; never fails startup.
    openhydra_agent::harden_process();
    if let Err(e) = run() {
        eprintln!("openhydra-agent: {e}");
        std::process::exit(1);
    }
}

/// If `OH_PROFILE_SECS` is set (and built with `--features profiling`), sample the whole
/// process for that many seconds and write an SVG flamegraph to `/tmp/oh_flame.svg`. The
/// guard samples all threads via SIGPROF (no kernel perf), so it works in containers.
#[cfg(feature = "profiling")]
fn start_profiler_if_requested() {
    let secs = match std::env::var("OH_PROFILE_SECS").ok().and_then(|s| s.parse::<u64>().ok()) {
        Some(s) => s,
        None => return,
    };
    let guard = match pprof::ProfilerGuardBuilder::default()
        .frequency(997)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
    {
        Ok(g) => g,
        Err(e) => {
            eprintln!("openhydra-agent: profiler failed to start: {e}");
            return;
        }
    };
    eprintln!("openhydra-agent: profiling {secs}s → /tmp/oh_flame.svg");
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(secs));
        match guard.report().build() {
            Ok(report) => match std::fs::File::create("/tmp/oh_flame.svg") {
                Ok(f) => {
                    let _ = report.flamegraph(f);
                    eprintln!("openhydra-agent: wrote /tmp/oh_flame.svg");
                }
                Err(e) => eprintln!("openhydra-agent: flamegraph file error: {e}"),
            },
            Err(e) => eprintln!("openhydra-agent: profiler report failed: {e}"),
        }
    });
}

#[cfg(not(feature = "profiling"))]
fn start_profiler_if_requested() {}

fn run() -> Result<(), String> {
    // Surface the network crate's `tracing` events under `RUST_LOG` (default: warnings) and,
    // with `--features otel` + `OTEL_EXPORTER_OTLP_ENDPOINT`, export request spans over OTLP.
    openhydra_agent::telemetry::init();
    start_profiler_if_requested();
    let cli = Cli::parse();
    // #7: `peer-id` resolves the identity's libp2p PeerId without starting a swarm, so the
    // desktop can compute its own provider id and pass it to `serve --self-provider`.
    if let Role::PeerId = cli.role {
        let config = cli.node.into_config();
        let id = openhydra_network::identity::Identity::load_or_create(&config.identity_path)
            .map_err(|e| format!("load identity: {e}"))?;
        println!("{}", id.libp2p_peer_id);
        return Ok(());
    }
    let status_bind = cli.node.status_bind.clone();
    let config = cli.node.into_config();
    // Start the swarm first; both roles need the live node.
    let net = NetworkHandle::start(config)?;
    eprintln!(
        "openhydra-agent: node up — libp2p={} openhydra={}",
        net.libp2p_peer_id(),
        net.openhydra_peer_id(),
    );

    // P0 introspection: shared transfer counters + the optional read-only status endpoint.
    // The provider role writes the counters; the gateway's transfers stay zero for now
    // (its request metrics live on the Prometheus surface, #33).
    let stats = std::sync::Arc::new(TransferStats::default());
    // Give-to-get economy (M2.2 reputation + M2.3 credit): the running role publishes a
    // fresh snapshot into this on a short interval; the status server reads it under /status.
    let economy = std::sync::Arc::new(EconomyStats::default());
    let started_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    if let Some(bind) = status_bind {
        let role = match &cli.role {
            Role::Provide(_) => "provider",
            Role::Serve(_) => "gateway",
            Role::PeerId => unreachable!("peer-id returns before the status server"),
        };
        let local = StatusServer {
            role,
            agent_version: env!("CARGO_PKG_VERSION"),
            libp2p_peer_id: net.libp2p_peer_id().to_string(),
            openhydra_peer_id: net.openhydra_peer_id().to_string(),
            net: net.status_client(),
            stats: std::sync::Arc::clone(&stats),
            economy: std::sync::Arc::clone(&economy),
            started_at_ms,
            token: std::env::var("OPENHYDRA_STATUS_TOKEN").ok(),
        }
        .spawn(&bind)?;
        eprintln!("openhydra-agent: status endpoint at http://{local}/status");
    }

    match cli.role {
        Role::Provide(args) => provide(net, stats, economy, args),
        Role::Serve(args) => serve(net, stats, economy, args),
        Role::PeerId => unreachable!("peer-id handled before the swarm starts"),
    }
}

/// Publish `snapshot()` into `economy` every 2s on a background thread. `snapshot` returns
/// the role's (reputation, credit) lists; `role` labels the perspective. Keeps the status
/// endpoint's economy view fresh without touching the blocking serve/gateway loops.
fn spawn_economy_publisher<F>(role: &'static str, economy: std::sync::Arc<EconomyStats>, snapshot: F)
where
    F: Fn() -> (Vec<openhydra_agent::RepEntry>, Vec<openhydra_agent::CreditEntry>) + Send + 'static,
{
    std::thread::spawn(move || loop {
        let (reputation, credit) = snapshot();
        economy.publish(openhydra_agent::EconomyView::new(role, reputation, credit));
        std::thread::sleep(Duration::from_secs(2));
    });
}

/// Provider role: pick the engine adapter from `--engine-kind`, then detect + announce +
/// serve. The post-adapter logic is generic over [`EngineAdapter`] (see [`run_provider`]),
/// so each kind monomorphises without boxing.
fn provide(
    net: NetworkHandle,
    stats: std::sync::Arc<TransferStats>,
    economy: std::sync::Arc<EconomyStats>,
    args: ProvideArgs,
) -> Result<(), String> {
    // Auto mode scans the standard ports and serves the union of whatever's running — the
    // zero-config default. It handles its own autostart, so it branches out before we resolve
    // a single engine URL.
    if args.engine_kind == EngineKind::Auto {
        return provide_auto(net, stats, economy, args);
    }

    let url = args
        .engine
        .clone()
        .unwrap_or_else(|| args.engine_kind.default_url().to_string());
    // Bind the adapter first so its construction borrow of `url` ends before `url` is
    // moved into `run_single_engine`.
    match args.engine_kind {
        EngineKind::Auto => unreachable!("handled above"),
        EngineKind::Ollama => {
            let a = live_ollama(&url).map_err(|e| format!("engine {url}: {e}"))?;
            run_single_engine(a, url, args, net, stats, economy)
        }
        EngineKind::Vllm => {
            let a = live_openai(&url, "vllm").map_err(|e| format!("engine {url}: {e}"))?;
            run_single_engine(a, url, args, net, stats, economy)
        }
        EngineKind::LmStudio => {
            let a = live_openai(&url, "lm-studio").map_err(|e| format!("engine {url}: {e}"))?;
            run_single_engine(a, url, args, net, stats, economy)
        }
        EngineKind::LlamaCpp => {
            let a = live_llamacpp(&url).map_err(|e| format!("engine {url}: {e}"))?;
            run_single_engine(a, url, args, net, stats, economy)
        }
        EngineKind::Exo => {
            let a = live_exo(&url).map_err(|e| format!("engine {url}: {e}"))?;
            run_single_engine(a, url, args, net, stats, economy)
        }
        EngineKind::Comfyui => {
            // BYO-workflow when a template dir is given; else the built-in SD graph.
            let a = match args.comfyui_workflow_dir.clone() {
                Some(dir) => live_comfyui_with_workflows(&url, &dir)
                    .map_err(|e| format!("engine {url}: {e}"))?,
                None => live_comfyui(&url).map_err(|e| format!("engine {url}: {e}"))?,
            };
            run_single_engine(a, url, args, net, stats, economy)
        }
        EngineKind::Openai => {
            let a = live_openai(&url, "openai").map_err(|e| format!("engine {url}: {e}"))?;
            run_single_engine(a, url, args, net, stats, economy)
        }
    }
}

/// Provider over a single explicitly-chosen engine: optionally autostart it, then run. The
/// autostart hook lives here (not in [`run_provider`]) so auto mode — which does its own
/// autostart across engines — doesn't double-handle it.
fn run_single_engine<A: EngineAdapter + Send + Sync + 'static>(
    adapter: A,
    url: String,
    args: ProvideArgs,
    net: NetworkHandle,
    stats: std::sync::Arc<TransferStats>,
    economy: std::sync::Arc<EconomyStats>,
) -> Result<(), String> {
    ensure_engine_if_requested(&args, &url, &adapter)?;
    run_provider(adapter, url, args, net, stats, economy)
}

/// Auto-detect provider: probe the standard ports and serve the union of every engine found.
/// If none is up and `--engine-autostart` was given, try to start one, then re-detect. Fails
/// fast with an actionable message if nothing can be found or started.
fn provide_auto(
    net: NetworkHandle,
    stats: std::sync::Arc<TransferStats>,
    economy: std::sync::Arc<EconomyStats>,
    args: ProvideArgs,
) -> Result<(), String> {
    if args.engine.is_some() {
        eprintln!(
            "openhydra-agent: --engine is ignored in auto mode (auto scans the standard ports); \
             pass --engine-kind to target one engine at a custom URL",
        );
    }

    let mut engines = openhydra_agent::detect::detect_engines();
    if engines.is_empty() && args.engine_autostart {
        eprintln!("openhydra-agent: no engine detected — --engine-autostart: attempting to start one");
        if autostart_when_none_detected() {
            engines = openhydra_agent::detect::detect_engines();
        }
    }
    if engines.is_empty() {
        return Err(no_engine_message(args.engine_autostart));
    }

    let summary = engines
        .iter()
        .map(|e| {
            let n = e.models.len();
            format!("{}({n} model{}) @ {}", e.label, if n == 1 { "" } else { "s" }, e.url)
        })
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!("openhydra-agent: auto-detected {} engine(s): {summary}", engines.len());

    // The MultiAdapter re-probes on every re-announce tick, so engines/models started later
    // are absorbed with no restart. Drop the decision-scan's adapters; it rebuilds its own.
    run_provider(openhydra_agent::detect::MultiAdapter::new(), "auto".to_string(), args, net, stats, economy)
}

/// The fail-fast message when auto mode finds no engine (§5 of the autostart plan).
fn no_engine_message(autostart_tried: bool) -> String {
    let base = "no local engine detected on the standard ports (ollama :11434, llama.cpp :8080, \
        LM Studio :1234, vLLM :8000, Exo :52415). Start your engine's server, or pass \
        --engine-kind/--engine to target one";
    if autostart_tried {
        format!("{base}. (--engine-autostart could not start one — is Ollama or LM Studio installed?)")
    } else {
        format!("{base}. Tip: pass --engine-autostart to have OpenHydra start Ollama / LM Studio for you")
    }
}

/// Auto-mode autostart (real impl): with nothing detected, try to start a known headless-ish
/// engine — Ollama first (pure daemon), then LM Studio. Returns whether one came up.
#[cfg(feature = "engine-autostart")]
fn autostart_when_none_detected() -> bool {
    use openhydra_agent::autostart::{ensure_running, LaunchSpec};
    if let (Ok(a), Some(spec)) = (live_ollama(DEFAULT_OLLAMA_URL), LaunchSpec::for_engine("ollama")) {
        if ensure_running(DEFAULT_OLLAMA_URL, &spec, || a.detect_models().is_ok()).is_ok() {
            return true;
        }
    }
    if let (Ok(a), Some(spec)) =
        (live_openai(DEFAULT_LM_STUDIO_URL, "lm-studio"), LaunchSpec::for_engine("lm-studio"))
    {
        if ensure_running(DEFAULT_LM_STUDIO_URL, &spec, || a.detect_models().is_ok()).is_ok() {
            return true;
        }
    }
    false
}

/// Auto-mode autostart (feature disabled): nothing to start.
#[cfg(not(feature = "engine-autostart"))]
fn autostart_when_none_detected() -> bool {
    false
}

/// Autostart hook (real impl): when `--engine-autostart` is set, ensure the engine's server
/// is up — launching LM Studio / Ollama if we have a launcher for it — before we announce.
/// Readiness reuses the adapter's own detection, so "up" means exactly what announce checks.
#[cfg(feature = "engine-autostart")]
fn ensure_engine_if_requested<A: EngineAdapter>(
    args: &ProvideArgs,
    engine_url: &str,
    adapter: &A,
) -> Result<(), String> {
    use openhydra_agent::autostart::LaunchSpec;
    if !args.engine_autostart {
        return Ok(());
    }
    match LaunchSpec::for_engine(adapter.engine_name()) {
        Some(spec) => openhydra_agent::autostart::ensure_running(engine_url, &spec, || {
            adapter.detect_models().is_ok()
        }),
        None => {
            // vLLM / llama.cpp / Exo need a model or cluster arg we can't invent; leave them
            // to the operator. If it's down, the announce step below reports it.
            eprintln!(
                "openhydra-agent: --engine-autostart has no launcher for {} — start it yourself",
                adapter.engine_name(),
            );
            Ok(())
        }
    }
}

/// Autostart hook (feature disabled): the flag parses but does nothing except warn, so a lean
/// `--no-default-features` build never links a process-spawning path.
#[cfg(not(feature = "engine-autostart"))]
fn ensure_engine_if_requested<A: EngineAdapter>(
    args: &ProvideArgs,
    _engine_url: &str,
    _adapter: &A,
) -> Result<(), String> {
    if args.engine_autostart {
        eprintln!(
            "openhydra-agent: WARNING — --engine-autostart ignored (built without the \
             engine-autostart feature)",
        );
    }
    Ok(())
}

/// Run a provider over a chosen engine adapter: open the ledger, announce the engine's
/// models, then serve inbound requests forever. Generic so every [`EngineKind`] reuses it.
fn run_provider<A: EngineAdapter + Send + Sync + 'static>(
    adapter: A,
    engine_url: String,
    args: ProvideArgs,
    net: NetworkHandle,
    stats: std::sync::Arc<TransferStats>,
    economy: std::sync::Arc<EconomyStats>,
) -> Result<(), String> {
    // Open the receipt ledger: file-backed if --db was given (durable across restarts),
    // else an ephemeral in-memory ledger.
    let store = match &args.db {
        Some(path) => {
            let s = Store::open(path).map_err(|e| format!("open ledger {}: {e}", path.display()))?;
            eprintln!("openhydra-agent: receipt ledger at {}", path.display());
            s
        }
        None => {
            eprintln!("openhydra-agent: receipt ledger in-memory (ephemeral; pass --db to persist)");
            Store::open_in_memory().map_err(|e| format!("open in-memory ledger: {e}"))?
        }
    };

    // Rehydrate the durable Ledger (recent rows + lifetime served/used totals) so the desktop
    // Ledger view and its counters survive a restart instead of resetting to zero. 250 = the
    // status ring cap. Read from the store before it's moved into the provider below.
    if let (Ok(rows), Ok((served, used, n))) = (store.recent_ledger_rows(250), store.ledger_totals()) {
        stats.rehydrate_ledger(&rows, served, used, n);
    }
    let aup = args.aup.clone().into_policy();
    let provider = Provider::new(adapter, net)
        .with_address(args.host, args.port)
        .with_store(store)
        .with_aup(aup)
        .with_shared_models(args.share_models.clone())
        .with_stats(stats);
    if !args.share_models.is_empty() {
        eprintln!(
            "openhydra-agent: sharing only {} selected model(s): {}",
            args.share_models.len(),
            args.share_models.join(", ")
        );
    }

    let announced = provider
        .announce_models()
        .map_err(|e| format!("announce models from {engine_url}: {e}"))?;
    if announced == 0 {
        // Not fatal — the operator may load models later — but the node serves nothing
        // until something is advertised, so make the silence visible. No restart needed:
        // the serve loop re-detects and re-announces on every interval below.
        eprintln!(
            "openhydra-agent: WARNING — engine {engine_url} reported 0 models; serving nothing \
             until a model is loaded — the node re-detects automatically every {}s (no restart needed)",
            args.reannounce_secs,
        );
    } else {
        eprintln!("openhydra-agent: announced {announced} model(s) from {engine_url}");
    }

    eprintln!(
        "openhydra-agent: serving inbound requests, re-announcing every {}s (Ctrl-C to stop)",
        args.reannounce_secs,
    );
    // Blocks forever (returns `!`); poll in short slices to stay responsive and
    // re-announce within the relays' provider-record TTL. Serves concurrently via a bounded
    // worker pool so one slow generation can't block the others.
    let provider = std::sync::Arc::new(provider);
    // Publish the take-side credit view (per-consumer balance + serve-rate cap) to the status
    // endpoint every 2s.
    let pub_provider = std::sync::Arc::clone(&provider);
    spawn_economy_publisher("provider", economy, move || pub_provider.economy_snapshot());
    provider.run_inbound(
        Duration::from_millis(500),
        Duration::from_secs(args.reannounce_secs),
        args.max_concurrency,
    )
}

/// Consumer role: run the HTTP/SSE gateway until the process exits.
fn serve(
    net: NetworkHandle,
    stats: std::sync::Arc<TransferStats>,
    economy: std::sync::Arc<EconomyStats>,
    args: ServeArgs,
) -> Result<(), String> {
    // CLI flag wins; otherwise fall back to the env var (avoids depending on clap's `env`).
    let api_key = args.api_key.or_else(|| std::env::var("OPENHYDRA_API_KEY").ok());
    // M2.2(a): persist earned provider reputation across restarts if --db was given.
    let store = match &args.db {
        Some(path) => {
            let s = Store::open(path)
                .map_err(|e| format!("open reputation db {}: {e}", path.display()))?;
            eprintln!("openhydra-agent: reputation persisted at {}", path.display());
            Some(s)
        }
        None => None,
    };
    eprintln!(
        "openhydra-agent: gateway listening on http://{} (auth: {})",
        args.bind,
        if api_key.is_some() { "API key required on /v1/*" } else { "open" },
    );
    // H (Wave 2): warn loudly when the gateway is exposed without auth. We don't block
    // (loopback stays open by design), but a non-loopback bind with no API key is an OPEN
    // inference gateway — and with BYOK models mapped it's an open, operator-funded proxy to
    // paid frontier APIs. Mirrors the status endpoint's non-loopback warning.
    let exposed = args
        .bind
        .parse::<std::net::SocketAddr>()
        .map(|s| !s.ip().is_loopback())
        .unwrap_or(true); // unparseable → assume exposed
    if exposed && api_key.is_none() {
        eprintln!(
            "openhydra-agent: WARNING — gateway bound to a non-loopback address ({}) with NO API \
             key: /v1/* is OPEN to anyone who can reach it. Set --api-key (or OPENHYDRA_API_KEY).",
            args.bind
        );
        let byok_mapped = !args.byok.byok_anthropic_model.is_empty()
            || !args.byok.byok_gemini_model.is_empty()
            || !args.byok.byok_embedding_model.is_empty();
        if byok_mapped {
            eprintln!(
                "openhydra-agent: WARNING — BYOK models are mapped on this OPEN gateway: an \
                 unauthenticated caller who sends no X-Provider-Api-Key will spend YOUR operator \
                 key on paid APIs. Require auth before exposing BYOK."
            );
        }
    }
    let aup = args.aup.clone().into_policy();
    let rate_limit = args.rate_limit.into_config();
    let trusted_proxy = args.rate_limit.trusted_proxy;
    let embeddings = args.byok.embedding_config();
    let byok = args.byok.clone().into_config();
    serve_http(net, economy, stats, &args.bind, api_key, store, aup, rate_limit, trusted_proxy, byok, embeddings, args.self_provider.clone())
        .map_err(|e| format!("gateway on {}: {e}", args.bind))
}
