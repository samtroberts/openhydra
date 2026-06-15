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
    live_ollama, live_openai, serve_http, EngineAdapter, Provider, DEFAULT_LM_STUDIO_URL,
    DEFAULT_OLLAMA_URL, DEFAULT_VLLM_URL,
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
}

impl NodeArgs {
    /// Build a [`NodeConfig`], overriding defaults only where a flag was given.
    fn into_config(self) -> NodeConfig {
        let mut config = NodeConfig {
            enable_peer_relay: self.peer_relay,
            bootstrap_peers: self.bootstrap,
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
}

/// Which local engine an agent proxies to. Selects the adapter; the `--engine` URL
/// defaults to the kind's standard port when omitted.
#[derive(Copy, Clone, Debug, clap::ValueEnum)]
enum EngineKind {
    /// Ollama native API (`/api/*`) — rich metadata, full canonical ids.
    Ollama,
    /// vLLM (OpenAI-compatible `/v1/*`).
    Vllm,
    /// LM Studio (OpenAI-compatible `/v1/*`).
    LmStudio,
    /// Any other OpenAI-compatible server (Exo, llama.cpp `--api`, LocalAI, …).
    Openai,
}

impl EngineKind {
    /// The engine's conventional local base URL, used when `--engine` is omitted.
    fn default_url(self) -> &'static str {
        match self {
            EngineKind::Ollama => DEFAULT_OLLAMA_URL,
            EngineKind::Vllm | EngineKind::Openai => DEFAULT_VLLM_URL,
            EngineKind::LmStudio => DEFAULT_LM_STUDIO_URL,
        }
    }
}

#[derive(Args)]
struct ProvideArgs {
    /// Which local engine to proxy to.
    #[arg(long = "engine-kind", value_enum, default_value_t = EngineKind::Ollama)]
    engine_kind: EngineKind,

    /// Base URL of the local engine. Defaults to the engine kind's standard port
    /// (Ollama 11434, vLLM/OpenAI 8000, LM Studio 1234).
    #[arg(long)]
    engine: Option<String>,

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
}

#[derive(Args)]
struct ServeArgs {
    /// Address the HTTP/SSE gateway binds (loopback by default — no firewall prompt).
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
}

fn main() {
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

/// Surface the network crate's `tracing` events under `RUST_LOG` (default: warnings).
/// Without a subscriber, libp2p/Kademlia/relay diagnostics are invisible.
fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = fmt().with_env_filter(filter).with_writer(std::io::stderr).try_init();
}

fn run() -> Result<(), String> {
    init_tracing();
    start_profiler_if_requested();
    let cli = Cli::parse();
    let config = cli.node.into_config();
    // Start the swarm first; both roles need the live node.
    let net = NetworkHandle::start(config)?;
    eprintln!(
        "openhydra-agent: node up — libp2p={} openhydra={}",
        net.libp2p_peer_id(),
        net.openhydra_peer_id(),
    );

    match cli.role {
        Role::Provide(args) => provide(net, args),
        Role::Serve(args) => serve(net, args),
    }
}

/// Provider role: pick the engine adapter from `--engine-kind`, then detect + announce +
/// serve. The post-adapter logic is generic over [`EngineAdapter`] (see [`run_provider`]),
/// so each kind monomorphises without boxing.
fn provide(net: NetworkHandle, args: ProvideArgs) -> Result<(), String> {
    let url = args
        .engine
        .clone()
        .unwrap_or_else(|| args.engine_kind.default_url().to_string());
    // Bind the adapter first so its construction borrow of `url` ends before `url` is
    // moved into `run_provider`.
    match args.engine_kind {
        EngineKind::Ollama => {
            let a = live_ollama(&url).map_err(|e| format!("engine {url}: {e}"))?;
            run_provider(a, url, args, net)
        }
        EngineKind::Vllm => {
            let a = live_openai(&url, "vllm").map_err(|e| format!("engine {url}: {e}"))?;
            run_provider(a, url, args, net)
        }
        EngineKind::LmStudio => {
            let a = live_openai(&url, "lm-studio").map_err(|e| format!("engine {url}: {e}"))?;
            run_provider(a, url, args, net)
        }
        EngineKind::Openai => {
            let a = live_openai(&url, "openai").map_err(|e| format!("engine {url}: {e}"))?;
            run_provider(a, url, args, net)
        }
    }
}

/// Run a provider over a chosen engine adapter: open the ledger, announce the engine's
/// models, then serve inbound requests forever. Generic so every [`EngineKind`] reuses it.
fn run_provider<A: EngineAdapter>(
    adapter: A,
    engine_url: String,
    args: ProvideArgs,
    net: NetworkHandle,
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

    let provider = Provider::new(adapter, net).with_address(args.host, args.port).with_store(store);

    let announced = provider
        .announce_models()
        .map_err(|e| format!("announce models from {engine_url}: {e}"))?;
    if announced == 0 {
        // Not fatal — the operator may pull models later — but the node serves nothing
        // until something is advertised, so make the silence visible.
        eprintln!(
            "openhydra-agent: WARNING — engine {engine_url} reported 0 models; serving nothing \
             until models are pulled and the node re-announces (restart to re-detect)",
        );
    } else {
        eprintln!("openhydra-agent: announced {announced} model(s) from {engine_url}");
    }

    eprintln!(
        "openhydra-agent: serving inbound requests, re-announcing every {}s (Ctrl-C to stop)",
        args.reannounce_secs,
    );
    // Blocks forever (returns `!`); poll in short slices to stay responsive and
    // re-announce within the relays' provider-record TTL.
    provider.run_inbound(Duration::from_millis(500), Duration::from_secs(args.reannounce_secs))
}

/// Consumer role: run the HTTP/SSE gateway until the process exits.
fn serve(net: NetworkHandle, args: ServeArgs) -> Result<(), String> {
    eprintln!("openhydra-agent: gateway listening on http://{}", args.bind);
    serve_http(net, &args.bind).map_err(|e| format!("gateway on {}: {e}", args.bind))
}
