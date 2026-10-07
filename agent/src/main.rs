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

use openhydra_agent::side_scorer::SideScored;
use std::path::PathBuf;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};

use openhydra_agent::adapters::scoring::ScoringChoice;
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

mod launch;
mod connect_cli;
mod refbuild;
mod refbuild_texts;
mod registry_cli;

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
    /// Launch a coding tool (Claude Code, OpenCode, …) wired to the local OpenHydra gateway:
    /// sets its endpoint env, defaults the model to `openhydra/auto`, then execs it. No swarm.
    /// See `openhydra launch --list`.
    Launch(launch::LaunchArgs),
    /// Persistently wire a coding tool to the local gateway by writing an OpenHydra block into its
    /// own config file (Claude Code, OpenCode, Continue, Hermes, Pi) — the CLI twin of the desktop
    /// "Connect" button. Backs up any existing config first. No swarm. See `openhydra connect --list`.
    Connect(connect_cli::ConnectArgs),
    /// Un-wire a coding tool: restore the pristine pre-OpenHydra config from its backup (or delete a
    /// file we created). The inverse of `connect`, and the CLI twin of the desktop "Disconnect" button.
    Disconnect(connect_cli::DisconnectArgs),
    /// Export a signed `.openhydra` card — a secret-free "magnet for a model" others import to dial
    /// you directly, no live discovery. Only for a globally-shared model. No swarm. See `card export`.
    Card(CardArgs),
    /// Manage swarm membership (M3/M4): create a swarm, enroll members, accept a credential. The
    /// scriptable twin of the desktop Swarms panel — enrollment is offline copy/paste. See `swarm`.
    Swarm(SwarmArgs),
    /// Build a reference set for the log-prob identity check (for gateway operators): tokenize texts,
    /// score them on the reference, honest and substitute engine configurations, then assemble a
    /// reference with a calibrated cut-off for `serve --logprob-reference`. No swarm.
    Refbuild(refbuild::RefbuildArgs),
    /// Run a reference registry (for network operators): holds reference texts built with
    /// `refbuild` and hands gateways signed round packs, issuing every text once network-wide.
    /// See `registry init`. No swarm.
    Registry(registry_cli::RegistryArgs),
}

/// `swarm <action>` — the membership CLI (M3/M4). State lives under `--swarms-dir`.
#[derive(Args)]
struct SwarmArgs {
    #[command(subcommand)]
    action: SwarmAction,
}

#[derive(Subcommand)]
enum SwarmAction {
    /// Create a swarm (generate the group keypair, persist an owner record). Prints its JSON view.
    Create(SwarmCreateArgs),
    /// Build a signed enrollment request from THIS node's `--identity`, to send to a swarm owner.
    /// Prints the `openhydra:enroll:...` magnet.
    Request(SwarmRequestArgs),
    /// (Owner) Approve an enrollment request into a signed credential. Prints the `openhydra:cred:...`
    /// magnet to send back to the member.
    Approve(SwarmApproveArgs),
    /// (Member) Accept a credential the owner returned (verifies + binds to `--identity`). Prints the
    /// stored JSON view.
    Accept(SwarmAcceptArgs),
    /// List this node's swarms as redacted JSON views (no secrets).
    List(SwarmDirArg),
    /// (Owner) Revoke a member by its public key.
    Revoke(SwarmRevokeArgs),
    /// (Control member, M5) Send a signed `REMOTE_SCOPE_SET` to a rig: flip one of its models'
    /// share scope (Private/Global/Device) from this device. Needs a stored control credential.
    RemoteScope(SwarmRemoteScopeArgs),
}

/// The swarms directory, shared by every action. Defaults to `~/.openhydra/swarms`.
#[derive(Args)]
struct SwarmDirArg {
    #[arg(long = "swarms-dir")]
    swarms_dir: Option<std::path::PathBuf>,
}

#[derive(Args)]
struct SwarmCreateArgs {
    /// Human label (e.g. "Home rig").
    #[arg(long)]
    label: String,
    #[command(flatten)]
    dir: SwarmDirArg,
}

#[derive(Args)]
struct SwarmRequestArgs {
    /// A label for this device (the owner sees it).
    #[arg(long)]
    label: String,
    /// Optional group-public-key hint of which swarm to join.
    #[arg(long)]
    swarm: Option<String>,
    #[command(flatten)]
    dir: SwarmDirArg,
}

#[derive(Args)]
struct SwarmApproveArgs {
    /// The swarm (group public key) to approve into — must be one you own.
    #[arg(long)]
    swarm: String,
    /// The enrollment request (magnet or JSON) the member sent.
    #[arg(long)]
    request: String,
    /// A label for this member.
    #[arg(long, default_value = "")]
    label: String,
    /// Credential validity in seconds (default 90 days).
    #[arg(long, default_value_t = 90 * 24 * 3600)]
    ttl_secs: u64,
    /// What this credential grants (M5 dedicated control identity):
    /// `serve` (default) → a consume/serve credential; `control` → a **remote-control**-only
    /// credential (`CAP_CONTROL`) that can send `swarm remote-scope` commands to flip this owner's rig
    /// models' scope. A control credential is minted against the member's *control* key and carries no
    /// serve access. Granting "both" is two approvals — run this once per request (the member sends a
    /// consume request from their consumer key and a control request from their control key).
    #[arg(long, default_value = "serve")]
    grant: String,
    #[command(flatten)]
    dir: SwarmDirArg,
}

#[derive(Args)]
struct SwarmRemoteScopeArgs {
    /// The swarm (group public key) whose stored control credential to present. This node must hold a
    /// control credential for it (from `swarm accept` of a `--grant control` credential), and the
    /// global `--identity` must be the **control key** that credential is bound to.
    #[arg(long)]
    swarm: String,
    /// The target rig's libp2p peer id (the provider to command).
    #[arg(long)]
    provider: String,
    /// The model (engine handle) to re-scope on the rig.
    #[arg(long)]
    model: String,
    /// The new scope: `global` | `private` | `device`.
    #[arg(long)]
    scope: String,
    /// Round-trip timeout in seconds.
    #[arg(long, default_value_t = 30)]
    timeout_secs: u64,
    #[command(flatten)]
    dir: SwarmDirArg,
}

#[derive(Args)]
struct SwarmAcceptArgs {
    /// The credential (magnet or JSON) the owner returned.
    #[arg(long)]
    credential: String,
    /// Optional local label override (defaults to the swarm's label on the credential).
    #[arg(long, default_value = "")]
    label: String,
    #[command(flatten)]
    dir: SwarmDirArg,
}

#[derive(Args)]
struct SwarmRevokeArgs {
    /// The swarm (group public key) you own.
    #[arg(long)]
    swarm: String,
    /// The member's public key to revoke.
    #[arg(long)]
    member: String,
    #[command(flatten)]
    dir: SwarmDirArg,
}

/// `card <action>` — the `.openhydra` card CLI (M2).
#[derive(Args)]
struct CardArgs {
    #[command(subcommand)]
    action: CardAction,
}

#[derive(Subcommand)]
enum CardAction {
    /// Sign and print a card as JSON `{ "card": {...}, "magnet": "openhydra:card:..." }` for MODEL.
    Export(CardExportArgs),
    /// Verify a card (from `--magnet`, `--file`, or stdin) and print its model + provider peer id,
    /// or the reason it's invalid/expired. Exit non-zero on failure.
    Verify(CardVerifyArgs),
}

#[derive(Args)]
struct CardVerifyArgs {
    /// A magnet string `openhydra:card:...`.
    #[arg(long, conflicts_with = "file")]
    magnet: Option<String>,
    /// A `.openhydra` JSON file.
    #[arg(long)]
    file: Option<std::path::PathBuf>,
}

#[derive(Args)]
struct CardExportArgs {
    /// The engine handle of the model to export (e.g. `qwen3:1.7b`). Must be detected locally.
    #[arg(long)]
    model: String,
    /// Share-policy file consulted for the global-share/consent export gate.
    #[arg(long = "share-policy-file")]
    share_policy_file: Option<std::path::PathBuf>,
    /// Card validity in seconds from now (default 30 days).
    #[arg(long, default_value_t = 30 * 24 * 3600)]
    ttl_secs: u64,
    /// Pricing posture the card declares (a self-claim).
    #[arg(long, value_enum, default_value_t = CardPricingArg::Reciprocal)]
    pricing: CardPricingArg,
    /// Optional region hint (e.g. "in", "us").
    #[arg(long)]
    region: Option<String>,
    /// M4: export a PRIVATE card bound to this swarm (hex group public key). The model must be shared
    /// with reach Private and you must own the swarm (`--swarms-dir`). Omit for a public card.
    #[arg(long = "swarm")]
    swarm: Option<String>,
    /// M4: swarms directory (`~/.openhydra/swarms/`), to confirm you own the `--swarm` you're binding.
    #[arg(long = "swarms-dir")]
    swarms_dir: Option<std::path::PathBuf>,
}

#[derive(Copy, Clone, clap::ValueEnum)]
enum CardPricingArg {
    Reciprocal,
    Paid,
    AdSupported,
}
impl From<CardPricingArg> for openhydra_agent::PricingMode {
    fn from(p: CardPricingArg) -> Self {
        match p {
            CardPricingArg::Reciprocal => openhydra_agent::PricingMode::Reciprocal,
            CardPricingArg::Paid => openhydra_agent::PricingMode::Paid,
            CardPricingArg::AdSupported => openhydra_agent::PricingMode::AdSupported,
        }
    }
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

    /// Path to a JSON share-policy file (see `openhydra_agent::SharePolicy`), **hot-reloaded** while
    /// running: the desktop rewrites it when the user toggles a model, and the node applies the
    /// change within one poll slice (no restart). Takes precedence over `--share-models`. Omit for
    /// the static `--share-models` behavior.
    #[arg(long = "share-policy-file")]
    share_policy_file: Option<std::path::PathBuf>,

    /// This machine's own gateway (consumer) libp2p PeerId, when it runs one under its own identity
    /// (the desktop app does): its serves are self-serves it never settles, so a provider that charges
    /// exempts it from the payment gate. Get it from `openhydra-agent peer-id --identity <gateway-key>`.
    #[arg(long = "self-consumer")]
    self_consumer: Option<String>,

    /// M4-base: directory of owned/joined swarm records (`~/.openhydra/swarms/`). When set, a
    /// `Private`-scope model is served only to a swarm member presenting a valid credential for a
    /// swarm this node owns. Omit ⇒ no swarm auth (a private model is then refused for everyone).
    #[arg(long = "swarms-dir")]
    swarms_dir: Option<std::path::PathBuf>,

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

    /// Answer device-attestation challenges with a Secure Enclave key (Apple Silicon only; off by
    /// default). Consumers see a `self_signed` device badge — a live, nonce-bound key and this
    /// Mac's *claimed* profile (chip, OS, SIP, Secure Boot, binary hash). It does not raise the
    /// authenticity tier. Only publicly announced models are attested.
    #[arg(long)]
    attest: bool,

    /// Answer prompt-scoring challenges (the teacher-forced log-prob identity check behind the A2
    /// authenticity tier). `auto` identifies each engine's scoring API itself (vLLM, SGLang, or the
    /// OpenHydra scoring endpoint of patched mistral.rs / llama-server `--prompt-scoring`) — the only
    /// choice with `--engine-kind auto`; `vllm` / `sglang` / `openhydra` name it explicitly (needs
    /// `--engine-kind vllm`, `openai` or `llama-cpp`). Off by default; at most 2 challenges are scored
    /// at once. Only publicly announced models are scorable. Ollama can't score a prompt itself: its
    /// models are scored by the side-scorer (see `--side-scorer`) on their weights file; without
    /// one, and for LM Studio and stock llama-server, challenges are refused.
    #[arg(long = "prompt-scoring")]
    prompt_scoring: Option<openhydra_agent::adapters::scoring::ScoringChoice>,

    /// The quantisation to declare, for authenticity checks, for models whose engine doesn't report
    /// one — e.g. `bf16` or `fp8` for the checkpoint a vLLM / SGLang server loaded. Gateways test the
    /// model against the reference for exactly this quant, so declare what you actually serve.
    /// Ollama and llama.cpp report their own quant and ignore this.
    #[arg(long = "declare-quant")]
    declare_quant: Option<String>,

    /// The side-scorer for `--prompt-scoring` on engines that can't score a prompt themselves
    /// (Ollama): OpenHydra's patched llama-server, started on the model's own weights file when a
    /// challenge arrives and stopped when idle. Defaults to `openhydra-scorer` next to this binary.
    #[arg(long = "side-scorer")]
    side_scorer: Option<std::path::PathBuf>,

    /// The side-scorer for LM Studio's MLX models: `openhydra-mlx-scorer` (macOS). Defaults to the
    /// one next to this binary.
    #[arg(long = "mlx-scorer")]
    mlx_scorer: Option<std::path::PathBuf>,

    /// Don't use a side-scorer (models of engines that can't score are then refused, as before).
    #[arg(long = "no-side-scorer", conflicts_with_all = ["side_scorer", "mlx_scorer"])]
    no_side_scorer: bool,

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

    /// M2: path to the imported-cards store — verified `.openhydra` cards routed by peer id without
    /// live discovery. Hot-reloaded on change. Omit to disable card routing.
    #[arg(long = "cards-file")]
    cards_file: Option<std::path::PathBuf>,

    /// M4-base: directory of joined swarm records (`~/.openhydra/swarms/`). When set, a held swarm
    /// membership credential is presented on serve requests, so a private (swarm-scoped) provider will
    /// serve us. Hot-reloaded. Omit to disable private routing.
    #[arg(long = "swarms-dir")]
    swarms_dir: Option<std::path::PathBuf>,

    /// How audits check providers. `on` (the default): a greedy (temperature 0) request
    /// to a public model may be sent again, as is, to a second provider of it, and the two answers
    /// compared, so the audited provider can't tell it was checked. Your prompt then also reaches
    /// that second provider (never a private swarm's prompt; nothing is written to disk). `off`:
    /// audits send generated test prompts shaped like your traffic instead.
    #[arg(long = "audit-replay", default_value = "on", value_parser = ["on", "off"])]
    audit_replay: String,

    /// Challenge providers for a device attestation after serving (in the background, at most every
    /// few minutes each) and show the verified level as a device badge in the status view. Off by
    /// default; informational only — it changes neither routing nor the authenticity tier.
    #[arg(long = "attest-providers")]
    attest_providers: bool,

    /// SHA-256 (hex) of a provider binary to trust when verifying attestations (repeatable). This
    /// gateway's own binary hash is always trusted, so providers on the same release build verify
    /// with no extra config. Without a hardware attestation chain this is release hygiene, not a
    /// security boundary.
    #[arg(long = "trusted-binary-hash")]
    trusted_binary_hash: Vec<String>,

    /// A private log-prob reference set (JSON) for the teacher-forced identity check (repeatable, one
    /// per model). After serving a model a set covers, the gateway challenges the provider in the
    /// background (every few hours) and a passing round lifts it toward the A2 authenticity tier; a
    /// failing one drops it to A0. Keep these files private: a provider that has seen a text can
    /// replay an honest answer. Providers must run `provide --prompt-scoring`.
    #[arg(long = "logprob-reference")]
    logprob_reference: Vec<std::path::PathBuf>,

    /// A reference registry (`openhydra-agent registry serve`) to get round packs from when no local
    /// reference covers a provider's model. HTTPS, or http:// to localhost only.
    #[arg(long = "logprob-registry", requires_all = ["logprob_registry_key_file", "logprob_registry_pubkey"])]
    logprob_registry: Option<String>,

    /// File holding this gateway's registry API key (from `registry keys add`).
    #[arg(long = "logprob-registry-key-file")]
    logprob_registry_key_file: Option<std::path::PathBuf>,

    /// The registry's public key (hex, from `registry pubkey`); packs signed by anything else are
    /// rejected.
    #[arg(long = "logprob-registry-pubkey")]
    logprob_registry_pubkey: Option<String>,

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
    // Roles that need no swarm are handled (and returned) before starting the network:
    //  • `peer-id` (#7) resolves the identity's libp2p PeerId so the desktop can wire
    //    `serve --self-provider <id>` for the self-serve credit skip.
    //  • `launch` just execs a coding tool wired to the local gateway.
    //  • `connect` writes an OpenHydra block into a coding tool's config file and exits.
    let role = match cli.role {
        Role::PeerId => {
            let config = cli.node.into_config();
            let id = openhydra_network::identity::Identity::load_or_create(&config.identity_path)
                .map_err(|e| format!("load identity: {e}"))?;
            println!("{}", id.libp2p_peer_id);
            return Ok(());
        }
        Role::Launch(args) => return launch::run(args),
        Role::Connect(args) => return connect_cli::run(args),
        Role::Disconnect(args) => return connect_cli::run_disconnect(args),
        Role::Refbuild(args) => return refbuild::run(args),
        Role::Registry(args) => return registry_cli::run(args),
        Role::Card(args) => {
            // `card export` / `card verify`. No swarm.
            let config = cli.node.into_config();
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            match args.action {
                CardAction::Export(a) => {
                    let out = openhydra_agent::cards::run_export(
                        &a.model,
                        a.share_policy_file,
                        a.ttl_secs,
                        a.pricing.into(),
                        a.region,
                        &config.identity_path,
                        now_ms,
                        a.swarm,
                        a.swarms_dir,
                    )?;
                    println!(
                        "{}",
                        serde_json::to_string(&out).map_err(|e| format!("serialize card: {e}"))?
                    );
                }
                CardAction::Verify(a) => {
                    let input = match (a.magnet, a.file) {
                        (Some(m), _) => m,
                        (None, Some(f)) => std::fs::read_to_string(&f)
                            .map_err(|e| format!("read {}: {e}", f.display()))?,
                        (None, None) => {
                            use std::io::Read;
                            let mut s = String::new();
                            std::io::stdin()
                                .read_to_string(&mut s)
                                .map_err(|e| format!("read stdin: {e}"))?;
                            s
                        }
                    };
                    let v = openhydra_agent::cards::parse_and_verify(&input, now_ms)?;
                    println!(
                        "OK: model {:?} served by {} (openhydra id {}), expires {}",
                        v.card.model_id, v.card.libp2p_peer_id, v.card.openhydra_peer_id, v.card.expires_at
                    );
                }
            }
            return Ok(());
        }
        Role::Swarm(args) => {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            // Resolve `--swarms-dir`, defaulting to `~/.openhydra/swarms`.
            let resolve_dir = |d: Option<std::path::PathBuf>| -> std::path::PathBuf {
                d.unwrap_or_else(|| {
                    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
                    std::path::PathBuf::from(home).join(".openhydra").join("swarms")
                })
            };
            // Keep the whole config: most swarm actions are offline (identity only), but `remote-scope`
            // needs the bootstrap peers to start a live node and dial the rig.
            let config = cli.node.into_config();
            let identity_path = config.identity_path.clone();
            match args.action {
                SwarmAction::Create(a) => {
                    let v = openhydra_agent::swarms::create_swarm(
                        &resolve_dir(a.dir.swarms_dir),
                        a.label.trim(),
                        now_ms,
                    )?;
                    println!("{}", serde_json::to_string(&v).map_err(|e| format!("serialize: {e}"))?);
                }
                SwarmAction::Request(a) => {
                    let out = openhydra_agent::swarms::enroll_request_at(
                        &identity_path,
                        a.swarm.as_deref().unwrap_or("").trim(),
                        a.label.trim(),
                        now_ms,
                    )?;
                    // Print just the magnet on stdout (easy to capture over ssh); the request JSON is
                    // in the magnet.
                    println!("{}", out.magnet);
                }
                SwarmAction::Approve(a) => {
                    use openhydra_network::membership::{CAP_CONTROL, CAP_SERVE};
                    // Map the grant to a single capability — control credentials are CONTROL-only (bound
                    // to the member's control key), serve credentials are SERVE-only. "Both" is two
                    // approvals, one per key (see --grant help), so it is not a single grant here.
                    let (caps, kind) = match a.grant.trim().to_lowercase().as_str() {
                        "serve" | "" => (CAP_SERVE, "serve"),
                        "control" => (CAP_CONTROL, "control"),
                        "both" => {
                            return Err("--grant both is two approvals: run `swarm approve --grant \
                                        serve` on the member's consume request and `--grant control` \
                                        on their control request (each binds a different key)"
                                .to_string())
                        }
                        other => {
                            return Err(format!("invalid --grant '{other}' (expected serve|control)"))
                        }
                    };
                    let out = openhydra_agent::swarms::approve_member_with_caps(
                        &resolve_dir(a.dir.swarms_dir),
                        a.swarm.trim(),
                        a.request.trim(),
                        a.label.trim(),
                        a.ttl_secs,
                        caps,
                        now_ms,
                    )?;
                    eprintln!("openhydra-agent: issued a {kind} credential");
                    println!("{}", out.magnet);
                }
                SwarmAction::Accept(a) => {
                    let v = openhydra_agent::swarms::accept_credential_at(
                        &resolve_dir(a.dir.swarms_dir),
                        &identity_path,
                        a.credential.trim(),
                        a.label.trim(),
                        now_ms,
                    )?;
                    println!("{}", serde_json::to_string(&v).map_err(|e| format!("serialize: {e}"))?);
                }
                SwarmAction::List(a) => {
                    let v = openhydra_agent::swarms::list_swarms(&resolve_dir(a.swarms_dir))?;
                    println!("{}", serde_json::to_string(&v).map_err(|e| format!("serialize: {e}"))?);
                }
                SwarmAction::Revoke(a) => {
                    openhydra_agent::swarms::revoke_member(
                        &resolve_dir(a.dir.swarms_dir),
                        a.swarm.trim(),
                        a.member.trim(),
                    )?;
                    println!("revoked {}", a.member.trim());
                }
                SwarmAction::RemoteScope(a) => {
                    use openhydra_network::identity::Identity;
                    use openhydra_network::membership::CAP_CONTROL;
                    use openhydra_network::remote_control::{RemoteScopeAck, RemoteScopeSet};
                    // Validate the scope before dialing anything.
                    let scope = a.scope.trim().to_lowercase();
                    if !matches!(scope.as_str(), "global" | "private" | "device") {
                        return Err(format!(
                            "invalid --scope '{}' (expected global|private|device)",
                            a.scope
                        ));
                    }
                    let dir = resolve_dir(a.dir.swarms_dir);
                    // Load the CONTROL credential (dedicated control identity, M5). `--identity` must be
                    // the control key this credential is bound to — the rig binds it to the dialing peer.
                    let cred = openhydra_agent::swarms::control_credential_for(&dir, a.swarm.trim())?;
                    if !cred.has_capability(CAP_CONTROL) {
                        return Err("this swarm credential does not grant CAP_CONTROL — ask the \
                                    owner to approve your control device with `swarm approve --grant \
                                    control`"
                            .to_string());
                    }
                    let identity = Identity::load_or_create(&identity_path)
                        .map_err(|e| format!("load identity: {e}"))?;
                    let cmd = RemoteScopeSet::signed(
                        a.model.trim(),
                        scope.clone(),
                        now_ms,
                        cred,
                        &identity.keypair,
                    )
                    .map_err(|e| format!("sign command: {e}"))?;
                    // Bring up a live node and dial the rig (via relay if needed).
                    let net = NetworkHandle::start(config)?;
                    eprintln!(
                        "openhydra-agent: node up — libp2p={}; dialing rig {} to set {} -> {}",
                        net.libp2p_peer_id(),
                        a.provider.trim(),
                        a.model.trim(),
                        scope,
                    );
                    // Give the node a moment to connect to the bootstrap relay so a relayed
                    // provider is reachable before the one-shot command.
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    let mut data = vec![openhydra_agent::provider::REMOTE_SCOPE_SET];
                    data.extend_from_slice(&cmd.encode());
                    let resp = net.proxy_forward_timeout(
                        a.provider.trim().to_string(),
                        data,
                        std::time::Duration::from_secs(a.timeout_secs),
                    )?;
                    match RemoteScopeAck::decode(&resp) {
                        Ok(RemoteScopeAck::Applied { model_id, scope }) => {
                            println!("applied: {model_id} -> {scope}");
                        }
                        Ok(RemoteScopeAck::Refused(reason)) => {
                            return Err(format!("rig refused: {reason}"));
                        }
                        Err(e) => return Err(format!("unexpected reply from rig: {e}")),
                    }
                }
            }
            return Ok(());
        }
        other => other, // Provide | Serve — need the swarm
    };
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
        let role_str = match &role {
            Role::Provide(_) => "provider",
            Role::Serve(_) => "gateway",
            _ => unreachable!("no-swarm roles return before the status server"),
        };
        let local = StatusServer {
            role: role_str,
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

    match role {
        Role::Provide(args) => provide(net, stats, economy, args),
        Role::Serve(args) => serve(net, stats, economy, args),
        _ => unreachable!("no-swarm roles handled before the swarm starts"),
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
        if matches!(args.prompt_scoring, Some(ScoringChoice::Dialect(_))) {
            return Err("with --engine-kind auto, use --prompt-scoring auto (each engine is identified separately)".into());
        }
        return provide_auto(net, stats, economy, args);
    }
    let scorer = side_scorer(&args)?;
    if args.prompt_scoring.is_some()
        && !matches!(args.engine_kind, EngineKind::Vllm | EngineKind::Openai | EngineKind::LlamaCpp)
        && !(scorer.is_some() && matches!(args.engine_kind, EngineKind::Ollama | EngineKind::LmStudio))
    {
        return Err("--prompt-scoring needs --engine-kind vllm, openai or llama-cpp, or ollama / lm-studio with a \
                    side-scorer (openhydra-scorer / openhydra-mlx-scorer next to this binary, or --side-scorer / --mlx-scorer)"
            .into());
    }
    if matches!(args.prompt_scoring, Some(ScoringChoice::Dialect(_))) && matches!(args.engine_kind, EngineKind::Ollama | EngineKind::LmStudio) {
        return Err("Ollama and LM Studio are scored by the side-scorer: use --prompt-scoring auto".into());
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
            run_single_engine(SideScored::new(a, scorer.clone()), url, args, net, stats, economy)
        }
        EngineKind::Vllm => {
            let mut a = live_openai(&url, "vllm").map_err(|e| format!("engine {url}: {e}"))?;
            a = match args.prompt_scoring {
                Some(ScoringChoice::Dialect(dialect)) => a.with_prompt_scoring(dialect),
                Some(ScoringChoice::Auto) => a.with_auto_prompt_scoring(),
                None => a,
            };
            run_single_engine(SideScored::new(a, scorer.clone()), url, args, net, stats, economy)
        }
        EngineKind::LmStudio => {
            let a = live_openai(&url, "lm-studio").map_err(|e| format!("engine {url}: {e}"))?;
            run_single_engine(SideScored::new(a, scorer.clone()), url, args, net, stats, economy)
        }
        EngineKind::LlamaCpp => {
            let mut a = live_llamacpp(&url).map_err(|e| format!("engine {url}: {e}"))?;
            a = match args.prompt_scoring {
                Some(ScoringChoice::Dialect(dialect)) => a.with_prompt_scoring(dialect),
                Some(ScoringChoice::Auto) => a.with_auto_prompt_scoring(),
                None => a,
            };
            run_single_engine(SideScored::new(a, scorer.clone()), url, args, net, stats, economy)
        }
        EngineKind::Exo => {
            let a = live_exo(&url).map_err(|e| format!("engine {url}: {e}"))?;
            run_single_engine(SideScored::new(a, scorer.clone()), url, args, net, stats, economy)
        }
        EngineKind::Comfyui => {
            // BYO-workflow when a template dir is given; else the built-in SD graph.
            let a = match args.comfyui_workflow_dir.clone() {
                Some(dir) => live_comfyui_with_workflows(&url, &dir)
                    .map_err(|e| format!("engine {url}: {e}"))?,
                None => live_comfyui(&url).map_err(|e| format!("engine {url}: {e}"))?,
            };
            run_single_engine(SideScored::new(a, scorer.clone()), url, args, net, stats, economy)
        }
        EngineKind::Openai => {
            let mut a = live_openai(&url, "openai").map_err(|e| format!("engine {url}: {e}"))?;
            a = match args.prompt_scoring {
                Some(ScoringChoice::Dialect(dialect)) => a.with_prompt_scoring(dialect),
                Some(ScoringChoice::Auto) => a.with_auto_prompt_scoring(),
                None => a,
            };
            run_single_engine(SideScored::new(a, scorer.clone()), url, args, net, stats, economy)
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
    let multi = openhydra_agent::detect::MultiAdapter::new();
    let multi = if args.prompt_scoring.is_some() { multi.with_auto_prompt_scoring() } else { multi };
    let scorer = side_scorer(&args)?;
    run_provider(SideScored::new(multi, scorer), "auto".to_string(), args, net, stats, economy)
}

/// The side-scorer `--prompt-scoring` uses for engines that can't score (see `--side-scorer`):
/// `None` when scoring is off, `--no-side-scorer` is given, or no scorer binary is found.
fn side_scorer(args: &ProvideArgs) -> Result<Option<std::sync::Arc<openhydra_agent::side_scorer::LiveSideScorer>>, String> {
    if args.prompt_scoring.is_none() || args.no_side_scorer {
        return Ok(None);
    }
    use openhydra_agent::side_scorer::{cpu_runs_gguf_scorer, live_side_scorer, locate_scorer, MLX_SCORER_BIN, SCORER_BIN};
    let mut gguf = locate_scorer(args.side_scorer.as_deref(), SCORER_BIN);
    let mlx = locate_scorer(args.mlx_scorer.as_deref(), MLX_SCORER_BIN);
    for (given, found, flag) in [(&args.side_scorer, &gguf, "--side-scorer"), (&args.mlx_scorer, &mlx, "--mlx-scorer")] {
        if let (Some(p), None) = (given, found) {
            return Err(format!("{flag}: no such file: {}", p.display()));
        }
    }
    if gguf.is_some() && !cpu_runs_gguf_scorer() {
        eprintln!("openhydra-agent: this CPU lacks AVX2/FMA/F16C, which the GGUF side-scorer needs");
        gguf = None;
    }
    for (found, what) in [(&gguf, "GGUF models (Ollama, LM Studio)"), (&mlx, "MLX models (LM Studio)")] {
        match found {
            Some(bin) => eprintln!("openhydra-agent: side-scorer for {what}: {}", bin.display()),
            None => eprintln!("openhydra-agent: no side-scorer for {what}: their authenticity checks are refused"),
        }
    }
    if gguf.is_none() && mlx.is_none() {
        return Ok(None);
    }
    live_side_scorer(gguf, mlx).map(Some)
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
/// The Secure Enclave attestor for `provide --attest`.
#[cfg(target_os = "macos")]
fn secure_enclave_attestor() -> Result<openhydra_agent::attest::Attestor, String> {
    let attestor = openhydra_agent::attest::Attestor::secure_enclave()
        .map_err(|e| format!("--attest: {e} (needs an Apple Silicon Mac with a Secure Enclave)"))?;
    eprintln!("openhydra-agent: device attestation on (Secure Enclave, self_signed; per-process key)");
    Ok(attestor)
}

#[cfg(not(target_os = "macos"))]
fn secure_enclave_attestor() -> Result<openhydra_agent::attest::Attestor, String> {
    Err("--attest needs an Apple Silicon Mac (Secure Enclave); not available on this platform".into())
}

/// The settlement chain from the environment, checked against the peg (a chain settles only paying
/// receipts, so it needs one), and announced.
fn settlement_chain_for(
    peg: u64,
    consumer: bool,
) -> Result<Option<openhydra_agent::settlement_config::SettlementChain>, String> {
    let chain = openhydra_agent::settlement_config::from_env(consumer)?;
    if let Some(c) = &chain {
        if peg == 0 {
            return Err("OPENHYDRA_SETTLEMENT_CHAIN needs OPENHYDRA_USDC_MICROS_PER_CREDIT: only paying receipts settle on-chain".into());
        }
        eprintln!(
            "openhydra-agent: paying receipts settle on {} (deployment {})",
            c.domain.chain().label(),
            c.domain.fingerprint()
        );
    }
    Ok(chain)
}

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
    // Review F2: the provider validates a receipt's crypto-rail cumulative against a token floor when
    // it knows the settlement peg — the same opt-in env var the gateway uses. Unset/zero → the
    // provider validates monotonicity only.
    let settlement_peg = std::env::var("OPENHYDRA_USDC_MICROS_PER_CREDIT")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    // The on-chain deployment paying receipts settle on (opt-in; needs the peg: only paying receipts
    // carry claims).
    let settlement_chain = settlement_chain_for(settlement_peg, false)?;
    let mut provider = Provider::new(adapter, net)
        .with_address(args.host, args.port)
        .with_store(store)
        .with_aup(aup)
        .with_stats(stats)
        .with_settlement_peg(settlement_peg)
        .with_settlement_chain(settlement_chain.map(|c| c.domain))
        .with_self_consumer(args.self_consumer.clone());
    // A hot-reloadable policy file (the desktop path) takes precedence over the static
    // `--share-models` list. Only one governs sharing, so we never mix the two.
    if let Some(path) = args.share_policy_file.clone() {
        eprintln!(
            "openhydra-agent: share policy from {} (hot-reloaded on change)",
            path.display()
        );
        provider = provider.with_share_policy_file(path);
    } else {
        provider = provider.with_shared_models(args.share_models.clone());
        if !args.share_models.is_empty() {
            eprintln!(
                "openhydra-agent: sharing only {} selected model(s): {}",
                args.share_models.len(),
                args.share_models.join(", ")
            );
        }
    }
    // Phase 0 device attestation (opt-in). The operator asked for it explicitly, so a missing Secure
    // Enclave is a startup error rather than a silent downgrade.
    if args.attest {
        provider = provider.with_attestor(secure_enclave_attestor()?);
    }
    // Track A2-Logprob: answer prompt-scoring challenges (the adapter was given the dialect above).
    if let Some(choice) = args.prompt_scoring {
        let api = match choice {
            ScoringChoice::Auto => "identified per engine".to_string(),
            ScoringChoice::Dialect(d) => format!("{d:?} API"),
        };
        eprintln!("openhydra-agent: answering prompt-scoring challenges ({api}, ≤2 at once)");
        provider = provider.with_prompt_scoring().with_declared_quant(args.declare_quant.clone());
    }
    // M4-base: attach the owner-side swarm authorizer so `Private`-scope models can be served to
    // enrolled members. Without it, a private model is refused for everyone (fail-closed).
    if let Some(dir) = args.swarms_dir.clone() {
        eprintln!("openhydra-agent: swarm auth from {} (private models gated on membership)", dir.display());
        provider = provider
            .with_swarm_authorizer(openhydra_agent::swarms::SwarmAuthorizer::new(dir));
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
    // M2: imported-card routing (verified providers dialable by peer id without discovery).
    let cards = args.cards_file.clone().map(|p| {
        eprintln!("openhydra-agent: card imports at {}", p.display());
        openhydra_agent::cards::CardStore::new(p)
    });
    // M4-base: present a held swarm credential so a private provider serves us.
    let creds = args.swarms_dir.clone().map(|p| {
        eprintln!("openhydra-agent: swarm credentials from {}", p.display());
        openhydra_agent::swarms::CredentialStore::new(p)
    });
    // Phase 0 device attestation of providers (opt-in).
    let provider_attest = if args.attest_providers {
        let mut trusted = vec![openhydra_agent::attest::binary_hash()
            .map_err(|e| format!("hash this binary for --attest-providers: {e}"))?];
        for h in &args.trusted_binary_hash {
            let bytes = hex::decode(h.trim()).map_err(|e| format!("--trusted-binary-hash {h}: {e}"))?;
            trusted.push(bytes.try_into().map_err(|_| format!("--trusted-binary-hash {h}: expected 32 bytes"))?);
        }
        eprintln!(
            "openhydra-agent: attesting providers (device badge only); trusting {} binary hash(es), incl. this build {}",
            trusted.len(),
            hex::encode(trusted[0]),
        );
        Some(trusted)
    } else {
        None
    };
    // The on-chain deployment paying receipts settle on: a bad configuration is a startup error (the
    // gateway reads it again from the same environment).
    let peg = std::env::var("OPENHYDRA_USDC_MICROS_PER_CREDIT").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    settlement_chain_for(peg, true)?;
    // Track A2-Logprob reference sets (opt-in). A bad file is a startup error, not a silent skip.
    let mut logprob_refs = Vec::new();
    for path in &args.logprob_reference {
        let r = openhydra_agent::prompt_score::load_reference(path)?;
        let matches = match (&r.set.identity, r.handles.is_empty()) {
            (Some(id), true) => format!("model {}", id.key()),
            (Some(id), false) => format!("model {}, names {}", id.key(), r.handles.join(", ")),
            (None, _) => format!("names {}", r.handles.join(", ")),
        };
        eprintln!(
            "openhydra-agent: log-prob checks for {} ({} texts, cut-off {:.2}x, {matches})",
            r.set.model_key,
            r.set.texts.len(),
            r.set.cutoff,
        );
        logprob_refs.push(r);
    }
    let logprob_registry = match &args.logprob_registry {
        Some(url) => {
            let key_file = args.logprob_registry_key_file.as_ref().expect("clap requires it");
            let key = std::fs::read_to_string(key_file).map_err(|e| format!("{}: {e}", key_file.display()))?;
            let pubkey = args.logprob_registry_pubkey.as_deref().expect("clap requires it");
            let client = openhydra_agent::registry_client::RegistryClient::new(url, &key, pubkey)?;
            eprintln!("openhydra-agent: log-prob round packs from registry {url}");
            Some(client)
        }
        None => None,
    };
    serve_http(net, economy, stats, &args.bind, api_key, store, aup, rate_limit, trusted_proxy, byok, embeddings, args.self_provider.clone(), cards, creds, provider_attest, logprob_refs, logprob_registry, args.audit_replay == "on")
        .map_err(|e| format!("gateway on {}: {e}", args.bind))
}
