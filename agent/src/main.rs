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

use openhydra_agent::{live_ollama, serve_http, Provider};
use openhydra_agent::ollama::DEFAULT_OLLAMA_URL;
use openhydra_network::handle::NetworkHandle;
use openhydra_network::node::NodeConfig;

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

#[derive(Args)]
struct ProvideArgs {
    /// Base URL of the local engine to proxy to (Ollama's HTTP API).
    #[arg(long, default_value = DEFAULT_OLLAMA_URL)]
    engine: String,

    /// Advisory host advertised in records (routing is by libp2p peer id regardless).
    #[arg(long, default_value = "")]
    host: String,

    /// Advisory port advertised in records.
    #[arg(long, default_value_t = 0)]
    port: u16,
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

fn run() -> Result<(), String> {
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

/// Provider role: detect + announce the engine's models, then serve inbound forever.
fn provide(net: NetworkHandle, args: ProvideArgs) -> Result<(), String> {
    let adapter = live_ollama(&args.engine).map_err(|e| format!("engine {}: {e}", args.engine))?;
    let provider = Provider::new(adapter, net).with_address(args.host, args.port);

    let announced = provider
        .announce_models()
        .map_err(|e| format!("announce models from {}: {e}", args.engine))?;
    if announced == 0 {
        // Not fatal — the operator may pull models later — but the node serves nothing
        // until something is advertised, so make the silence visible.
        eprintln!(
            "openhydra-agent: WARNING — engine {} reported 0 models; serving nothing until \
             models are pulled and the node re-announces (restart to re-detect)",
            args.engine,
        );
    } else {
        eprintln!("openhydra-agent: announced {announced} model(s) from {}", args.engine);
    }

    eprintln!("openhydra-agent: serving inbound requests (Ctrl-C to stop)");
    // Blocks forever (returns `!`); poll in short slices to stay responsive.
    provider.run_inbound(Duration::from_millis(500))
}

/// Consumer role: run the HTTP/SSE gateway until the process exits.
fn serve(net: NetworkHandle, args: ServeArgs) -> Result<(), String> {
    eprintln!("openhydra-agent: gateway listening on http://{}", args.bind);
    serve_http(net, &args.bind).map_err(|e| format!("gateway on {}: {e}", args.bind))
}
