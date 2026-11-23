//! oh-sim: many real OpenHydra network nodes in one process, for bootstrap load tests
//! (docs: "OpenHydra Bootstrap Load Simulation Plan", 30 Oct 2026). Each simulated peer is a full
//! `NetworkHandle` node — the same transports, Kademlia, gossipsub, relay client and proxy protocol
//! a real agent runs — given only the test bootstrap's addresses, relaying only through it.
//!
//! usage: oh-sim --bootstrap ADDR [--bootstrap ADDR ...]
//!               --role idle|provider|relayed-provider|active|flooder|junk
//!               --count N [--first I] --dir DIR [--rate PER_SEC] [--duration SECS]
//!               [--providers A-B] [--req-every SECS] [--req-bytes B] [--reply-bytes B]
//!               [--sample-every K] [--models K] [--flood-rate PER_SEC] [--nat] [--vpc-is-public]
//!               [--kad-replication off|SECS] [--announce-gossip-only] [--kad-refresh off|SECS] [--kad-slow SECS]
//! `--kad-refresh`: Kademlia's periodic routing-table refresh (default 300 s; off = only when the
//! table runs low). `--kad-slow SECS`: Kademlia as a slow index (re-announce only on change or
//! after SECS; records live 1.2 x SECS). `--role lookup`: a just-joined node times 10 lookups of
//! models from `--providers A-B` through the registry and through discovery (OH_SIM_LOOKUP lines).
//! `--announce-gossip-only`: providers announce over gossip only (no Kademlia record or provider
//! record), to measure what Kademlia announcements cost.
//! `--kad-replication`: Kademlia's record-replication interval for every node (default 60 s), to
//! measure what replication costs.
//! `--vpc-is-public`: count 172.16.0.0/12 (the AWS simulation's VPC addresses) as public, so a
//! node AutoNAT finds reachable there is promoted to DHT server as on the real internet (without it
//! no simulated agent ever was: 20 Nov). 10/8 (the NAT homes) and 192.168/16 stay private.
//! `--models K`: a provider announces K models (sim-I, sim-I.1 … sim-I.K-1; record stuffing at K
//! in the hundreds). `--nat`: listen on loopback only, like a peer behind NAT (no dial-backs).
//! `--v6-prefix P`: peer I listens (TCP) only on address P+I, e.g. `2406:da1b:c0:6601:1::` (needs
//! the prefix routed to this host). Its TCP dials still leave from the host's default address:
//! libp2p-tcp reuses a listen port by binding the wildcard address, not the listen address.
//! `flooder`: announces a new model `--flood-rate` times a second. `junk`: publishes invalid
//! gossip (random bytes, oversized, unknown type, forged and stale announces) at `--flood-rate`.
//! `--announce-every SECS` (default 120): how often a provider re-announces (a long one stands for
//! announcing only on change, #11). `--listen-ip IP`: listen only on IP, one process per address on
//! a host with several (per-address limits at the bootstrap). Only QUIC dials leave from IP (they
//! use the listening socket); TCP dials leave from the host's default address (libp2p-tcp binds the
//! wildcard to reuse a port), so give the bootstraps as `/udp/…/quic-v1` addresses to spread the
//! peers over the host's addresses (19 Nov: over TCP, a host's peers all counted as one address).
//! Providers announce records the size of a real agent's (as `dht::pex_cost_measure`).
//! Prints one JSON line of aggregate counters and percentiles per second to stdout, with the
//! process's gossip counters (bytes received, unique and duplicate messages, signatures checked:
//! one process's total; divide by `started` for a node's share, or run an observer with --count 1),
//! and the sampled peers' gossip mesh: how many have no mesh peer, and their mesh and topic peers
//! summed. Logs go to stderr only when RUST_LOG is set (e.g. RUST_LOG=warn).

use openhydra_network::handle::NetworkHandle;
use openhydra_network::node::NodeConfig;
use openhydra_network::types::PeerRecord;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Stats {
    started: AtomicU64,
    start_errors: AtomicU64,
    announces: AtomicU64,
    served: AtomicU64,
    flood_sent: AtomicU64,
    req_ok: AtomicU64,
    req_err: AtomicU64,
    req_no_provider: AtomicU64,
    req_ms: Mutex<Vec<u64>>,
    reserve_ms: Mutex<Vec<u64>>,
    sampled: AtomicU64,
    sampled_reserved: AtomicU64,
    sampled_known_models: AtomicU64,
    sampled_bootstrap_connected: AtomicU64,
    sampled_other_models: AtomicU64,
    /// Sampled peers with no gossip mesh peer (they hear announces only through IHAVE, or not at all).
    sampled_no_mesh: AtomicU64,
    /// Sampled peers' gossip mesh and topic peers, summed.
    sampled_mesh_peers: AtomicU64,
    sampled_topic_peers: AtomicU64,
    /// Sampled peers' connections, by kind, and their Kademlia routing-table entries, summed.
    sampled_conns_quic: AtomicU64,
    sampled_conns_tcp: AtomicU64,
    sampled_conns_relay: AtomicU64,
    sampled_kad_peers: AtomicU64,
    errors: Mutex<std::collections::BTreeMap<String, u64>>,
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn args_multi(args: &[String], name: &str) -> Vec<String> {
    args.windows(2).filter(|w| w[0] == name).map(|w| w[1].clone()).collect()
}

fn pct(v: &mut Vec<u64>, p: f64) -> Option<u64> {
    if v.is_empty() {
        return None;
    }
    v.sort_unstable();
    let i = ((v.len() as f64 - 1.0) * p).round() as usize;
    v.get(i).copied()
}

fn note_err(stats: &Stats, e: &str) {
    // Bucket by the error's first words so the map stays small.
    let key: String = e.split_whitespace().take(5).collect::<Vec<_>>().join(" ");
    if let Ok(mut m) = stats.errors.lock() {
        if m.len() < 64 || m.contains_key(&key) {
            *m.entry(key).or_default() += 1;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bootstraps = args_multi(&args, "--bootstrap");
    let role = arg(&args, "--role").unwrap_or_else(|| "idle".into());
    let count: u64 = arg(&args, "--count").and_then(|v| v.parse().ok()).unwrap_or(10);
    let first: u64 = arg(&args, "--first").and_then(|v| v.parse().ok()).unwrap_or(0);
    let dir = std::path::PathBuf::from(arg(&args, "--dir").unwrap_or_else(|| "/tmp/oh-sim".into()));
    let rate: f64 = arg(&args, "--rate").and_then(|v| v.parse().ok()).unwrap_or(20.0);
    let duration: u64 = arg(&args, "--duration").and_then(|v| v.parse().ok()).unwrap_or(3600);
    let req_every: f64 = arg(&args, "--req-every").and_then(|v| v.parse().ok()).unwrap_or(10.0);
    let req_bytes: usize = arg(&args, "--req-bytes").and_then(|v| v.parse().ok()).unwrap_or(600);
    let reply_bytes: usize = arg(&args, "--reply-bytes").and_then(|v| v.parse().ok()).unwrap_or(4000);
    let sample_every: u64 = arg(&args, "--sample-every").and_then(|v| v.parse().ok()).unwrap_or(20);
    let (prov_lo, prov_hi): (u64, u64) = arg(&args, "--providers")
        .and_then(|v| {
            let (a, b) = v.split_once('-')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((0, 0));
    let models: u64 = arg(&args, "--models").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let flood_rate: f64 = arg(&args, "--flood-rate").and_then(|v| v.parse().ok()).unwrap_or(10.0);
    let nat = args.iter().any(|a| a == "--nat");
    let v6_prefix: Option<std::net::Ipv6Addr> = arg(&args, "--v6-prefix").map(|v| v.parse().expect("--v6-prefix"));
    let announce_every: f64 = arg(&args, "--announce-every").and_then(|v| v.parse().ok()).unwrap_or(120.0);
    let listen_ip: Option<std::net::Ipv4Addr> = arg(&args, "--listen-ip").map(|v| v.parse().expect("--listen-ip"));
    let opts = Opts { req_every, req_bytes, reply_bytes, prov_lo, prov_hi, models, flood_rate, nat, v6_prefix, announce_every, listen_ip };
    if bootstraps.is_empty() {
        eprintln!("oh-sim: --bootstrap required");
        std::process::exit(2);
    }
    std::fs::create_dir_all(&dir).expect("dir");
    openhydra_network::node::disable_mdns();
    if let Some(v) = arg(&args, "--kad-refresh") {
        openhydra_network::swarm::sim_set_kad_refresh(if v == "off" { None } else { Some(v.parse().expect("--kad-refresh off|SECS")) });
    }
    if let Some(v) = arg(&args, "--kad-slow") {
        openhydra_network::swarm::sim_set_kad_slow(v.parse().expect("--kad-slow SECS"));
    }
    if args.iter().any(|a| a == "--announce-gossip-only") {
        openhydra_network::event_loop::sim_announce_gossip_only();
    }
    if let Some(v) = arg(&args, "--kad-replication") {
        openhydra_network::swarm::sim_set_kad_replication(if v == "off" { None } else { Some(v.parse().expect("--kad-replication off|SECS")) });
    }
    if args.iter().any(|a| a == "--vpc-is-public") {
        openhydra_network::event_loop::sim_treat_172_16_as_public();
    }
    // Logs (stderr) only when asked: RUST_LOG=warn shows e.g. gossipsub giving up on a connection.
    if std::env::var_os("RUST_LOG").is_some() {
        let _ = tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).with_writer(std::io::stderr).try_init();
    }

    let stats = Arc::new(Stats::default());
    let t0 = Instant::now();

    // Reporter: one JSON line a second.
    {
        let stats = stats.clone();
        let role = role.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_secs(1));
            let mut req = std::mem::take(&mut *stats.req_ms.lock().unwrap());
            let mut res = std::mem::take(&mut *stats.reserve_ms.lock().unwrap());
            let errors = std::mem::take(&mut *stats.errors.lock().unwrap());
            let line = serde_json::json!({
                "t": t0.elapsed().as_secs(),
                "unix_ms": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0),
                "role": role,
                "started": stats.started.load(Relaxed),
                "start_errors": stats.start_errors.load(Relaxed),
                "announces": stats.announces.load(Relaxed),
                "served": stats.served.load(Relaxed),
                "flood_sent": stats.flood_sent.load(Relaxed),
                "req_ok": stats.req_ok.load(Relaxed),
                "req_err": stats.req_err.load(Relaxed),
                "req_no_provider": stats.req_no_provider.load(Relaxed),
                "req_n": req.len(),
                "req_p50": pct(&mut req, 0.50), "req_p95": pct(&mut req, 0.95), "req_p99": pct(&mut req, 0.99),
                "reserve_n": res.len(),
                "reserve_p50": pct(&mut res, 0.50), "reserve_p95": pct(&mut res, 0.95), "reserve_p99": pct(&mut res, 0.99),
                "sampled": stats.sampled.load(Relaxed),
                "sampled_reserved": stats.sampled_reserved.load(Relaxed),
                "sampled_known_models": stats.sampled_known_models.load(Relaxed),
                "sampled_bootstrap_connected": stats.sampled_bootstrap_connected.load(Relaxed),
                "sampled_other_models": stats.sampled_other_models.load(Relaxed),
                "sampled_no_mesh": stats.sampled_no_mesh.load(Relaxed),
                "sampled_mesh_peers": stats.sampled_mesh_peers.load(Relaxed),
                "sampled_topic_peers": stats.sampled_topic_peers.load(Relaxed),
                "sampled_conns_quic": stats.sampled_conns_quic.load(Relaxed),
                "sampled_conns_tcp": stats.sampled_conns_tcp.load(Relaxed),
                "sampled_conns_relay": stats.sampled_conns_relay.load(Relaxed),
                "sampled_kad_peers": stats.sampled_kad_peers.load(Relaxed),
                "rss_bytes": std::fs::read_to_string("/proc/self/statm").ok().and_then(|s| s.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok())).map(|p| p * 4096),
                "gossip_queued_bytes": libp2p::gossipsub::openhydra_counters::QUEUED_BYTES.load(Relaxed),
                "gossip_handlers_alive": libp2p::gossipsub::openhydra_counters::HANDLERS_ALIVE.load(Relaxed),
                "gossip_bytes_in": libp2p::gossipsub::openhydra_counters::RPC_BYTES_IN.load(Relaxed),
                "gossip_unique": libp2p::gossipsub::openhydra_counters::UNIQUE_MESSAGES.load(Relaxed),
                "gossip_duplicates": libp2p::gossipsub::openhydra_counters::DUPLICATES.load(Relaxed),
                "gossip_checked": libp2p::gossipsub::openhydra_counters::COPIES_CHECKED.load(Relaxed),
                "gossip_handlers_disabled": libp2p::gossipsub::openhydra_counters::HANDLERS_DISABLED.load(Relaxed),
                "errors": errors,
            });
            println!("{line}");
        });
    }

    let bootstrap_ids: Vec<String> = bootstraps
        .iter()
        .filter_map(|a| a.rsplit("/p2p/").next().map(String::from))
        .collect();
    let mut handles = Vec::new();
    for i in first..first + count {
        let stats = stats.clone();
        let bootstraps = bootstraps.clone();
        let bootstrap_ids = bootstrap_ids.clone();
        let role = role.clone();
        let dir = dir.clone();
        let sampled = (i % sample_every.max(1)) == 0;
        handles.push(
            std::thread::Builder::new()
                .stack_size(256 * 1024)
                .spawn(move || run_peer(i, &role, &dir, bootstraps, bootstrap_ids, sampled, stats, opts))
                .expect("thread"),
        );
        std::thread::sleep(Duration::from_secs_f64(1.0 / rate.max(0.1)));
    }
    let end = t0 + Duration::from_secs(duration);
    while Instant::now() < end {
        std::thread::sleep(Duration::from_secs(1));
    }
    std::process::exit(0);
}

#[derive(Clone, Copy)]
struct Opts {
    req_every: f64,
    req_bytes: usize,
    reply_bytes: usize,
    prov_lo: u64,
    prov_hi: u64,
    models: u64,
    flood_rate: f64,
    nat: bool,
    v6_prefix: Option<std::net::Ipv6Addr>,
    announce_every: f64,
    listen_ip: Option<std::net::Ipv4Addr>,
}

fn run_peer(
    i: u64,
    role: &str,
    dir: &std::path::Path,
    bootstraps: Vec<String>,
    bootstrap_ids: Vec<String>,
    sampled: bool,
    stats: Arc<Stats>,
    opts: Opts,
) {
    // The lookup role checks its connection against these; the sampling thread below takes its own.
    let lookup_bootstraps = bootstrap_ids.clone();
    let Opts { req_every, req_bytes, reply_bytes, prov_lo, prov_hi, models, flood_rate, nat, v6_prefix, announce_every, listen_ip } = opts;
    let listen = if let Some(p) = v6_prefix {
        let a = std::net::Ipv6Addr::from(u128::from(p) + i as u128);
        vec![format!("/ip6/{a}/tcp/0")]
    } else if role == "relayed-provider" || nat {
        // Loopback only: no address another peer could dial directly, so every request to it
        // comes through the test bootstrap's relay (and hole punching has nothing to work with).
        vec!["/ip4/127.0.0.1/tcp/0".to_string()]
    } else if let Some(ip) = listen_ip {
        vec![format!("/ip4/{ip}/tcp/0"), format!("/ip4/{ip}/udp/0/quic-v1")]
    } else {
        vec!["/ip4/0.0.0.0/tcp/0".to_string(), "/ip4/0.0.0.0/udp/0/quic-v1".to_string()]
    };
    let config = NodeConfig {
        identity_path: dir.join(format!("{role}-{i}.key")),
        listen_addrs: listen,
        bootstrap_peers: bootstraps,
        enable_peer_relay: false,
        enable_connection_reversal: false,
        pcp_gateway: None,
        public_relays: openhydra_network::relay::PublicRelays::Off,
    };
    let started_at = Instant::now();
    let net = match NetworkHandle::start(config) {
        Ok(n) => n,
        Err(e) => {
            stats.start_errors.fetch_add(1, Relaxed);
            note_err(&stats, &format!("start: {e}"));
            return;
        }
    };
    stats.started.fetch_add(1, Relaxed);
    let net = Arc::new(net);

    // Sampled peers watch their own status: first reservation time, discovery completeness.
    if sampled {
        stats.sampled.fetch_add(1, Relaxed);
        let net = net.clone();
        let stats = stats.clone();
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(move || {
                let (mut reserved, mut connected, mut known, mut other) = (false, false, 0u64, 0u64);
                let (mut no_mesh, mut mesh, mut topic) = (false, 0u64, 0u64);
                let (mut cq, mut ct, mut cr, mut kp) = (0u64, 0u64, 0u64, 0u64);
                // A sampled peer that has heard of no model for a minute after starting is deaf:
                // say so once on stderr, with who it is connected to, and again if it recovers.
                let mut deaf = false;
                loop {
                    std::thread::sleep(Duration::from_secs(5));
                    let Ok(s) = net.status() else { continue };
                    let r = !s.relay_reservations.is_empty();
                    if r && !reserved {
                        stats.reserve_ms.lock().unwrap().push(started_at.elapsed().as_millis() as u64);
                    }
                    if r != reserved {
                        if r { stats.sampled_reserved.fetch_add(1, Relaxed); } else { stats.sampled_reserved.fetch_sub(1, Relaxed); }
                        reserved = r;
                    }
                    let c = s.peers.iter().any(|p| bootstrap_ids.contains(&p.peer_id));
                    if c != connected {
                        if c { stats.sampled_bootstrap_connected.fetch_add(1, Relaxed); } else { stats.sampled_bootstrap_connected.fetch_sub(1, Relaxed); }
                        connected = c;
                    }
                    let k = s.known_models.iter().filter(|m| m.starts_with("sim-")).count() as u64;
                    let o = s.known_models.iter().filter(|m| !m.starts_with("sim-")).count() as u64;
                    stats.sampled_other_models.fetch_add(o, Relaxed);
                    stats.sampled_other_models.fetch_sub(other, Relaxed);
                    other = o;
                    let (m, t) = (s.gossip_mesh_peers.len() as u64, s.gossip_topic_peers.len() as u64);
                    if (m == 0) != no_mesh {
                        if m == 0 { stats.sampled_no_mesh.fetch_add(1, Relaxed); } else { stats.sampled_no_mesh.fetch_sub(1, Relaxed); }
                        no_mesh = m == 0;
                    }
                    stats.sampled_mesh_peers.fetch_add(m, Relaxed);
                    stats.sampled_mesh_peers.fetch_sub(mesh, Relaxed);
                    mesh = m;
                    stats.sampled_topic_peers.fetch_add(t, Relaxed);
                    stats.sampled_topic_peers.fetch_sub(topic, Relaxed);
                    topic = t;
                    let now_deaf = k == 0 && started_at.elapsed() > Duration::from_secs(60);
                    if now_deaf != deaf {
                        deaf = now_deaf;
                        let line = serde_json::json!({
                            "deaf": deaf,
                            "peer": net.libp2p_peer_id().to_string(),
                            "secs": started_at.elapsed().as_secs(),
                            "topic_peers": s.gossip_topic_peers,
                            "mesh_peers": s.gossip_mesh_peers,
                            "connected": s.peers.iter().map(|p| p.peer_id.clone()).collect::<Vec<_>>(),
                        });
                        eprintln!("OH_SIM_DEAF {line}");
                    }
                    let q: u64 = s.peers.iter().map(|p| (p.quic_direct_v4 + p.quic_direct_v6) as u64).sum();
                    let t: u64 = s.peers.iter().map(|p| p.tcp_direct as u64).sum();
                    let r: u64 = s.peers.iter().map(|p| p.tcp_relay as u64).sum();
                    let kad = s.kad_routing_peers as u64;
                    for (gauge, old, new) in [(&stats.sampled_conns_quic, &mut cq, q), (&stats.sampled_conns_tcp, &mut ct, t), (&stats.sampled_conns_relay, &mut cr, r), (&stats.sampled_kad_peers, &mut kp, kad)] {
                        gauge.fetch_add(new, Relaxed);
                        gauge.fetch_sub(*old, Relaxed);
                        *old = new;
                    }
                    stats.sampled_known_models.fetch_add(k, Relaxed);
                    stats.sampled_known_models.fetch_sub(known, Relaxed);
                    known = k;
                }
            })
            .ok();
    }

    match role {
        "provider" | "relayed-provider" => {
            // The size of a real agent's record (the fields `dht::pex_cost_measure` fills).
            let record = |model_id: String| PeerRecord {
                peer_id: net.openhydra_peer_id().to_string(),
                canonical_model_id: model_id.clone(),
                runtime_model_id: model_id.clone(),
                model_id,
                host: "203.0.113.25".into(),
                port: 18001,
                region: Some("ap-south".into()),
                daemon_mode: "provider".into(),
                runtime_backend: "ollama".into(),
                runtime_target: "cuda".into(),
                quantization_mode: "q4_k_m".into(),
                quantization_bits: 4,
                runtime_gpu_available: true,
                runtime_estimated_tokens_per_sec: 42.0,
                runtime_estimated_memory_mb: 6144,
                context_length: 40960,
                max_output_tokens: 8192,
                hardware_class: "consumer-gpu".into(),
                nat_type: "port_restricted".into(),
                requires_relay: true,
                relay_address: "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit".into(),
                libp2p_peer_id: net.libp2p_peer_id().to_string(),
                public_key: net.public_key_hex().to_string(),
                ..Default::default()
            };
            let records: Vec<PeerRecord> =
                (0..models).map(|k| record(if k == 0 { format!("sim-{i}") } else { format!("sim-{i}.{k}") })).collect();
            std::thread::sleep(Duration::from_secs(3));
            // Started at a random point of the cycle, as real providers are (they don't start together).
            let mut next_announce = Instant::now() + Duration::from_secs_f64(rand::random::<f64>() * announce_every.min(120.0));
            let mut first = true;
            let reply = vec![b'x'; reply_bytes];
            loop {
                if first || Instant::now() >= next_announce {
                    for r in &records {
                        match net.announce(r.clone()) {
                            Ok(()) => { stats.announces.fetch_add(1, Relaxed); }
                            Err(e) => note_err(&stats, &format!("announce: {e}")),
                        }
                    }
                    if !first {
                        next_announce = Instant::now() + Duration::from_secs_f64(announce_every);
                    }
                    first = false;
                }
                if let Some((request_id, source, _data)) = net.poll_inbound(Duration::from_secs(1)) {
                    net.serve_accepted(&source);
                    match net.respond(request_id, reply.clone()) {
                        Ok(()) => { stats.served.fetch_add(1, Relaxed); }
                        Err(e) => note_err(&stats, &format!("respond: {e}")),
                    }
                }
            }
        }
        "active" => {
            std::thread::sleep(Duration::from_secs(20));
            let span = prov_hi.saturating_sub(prov_lo).max(1);
            let mut n: u64 = i.wrapping_mul(2654435761);
            let req = vec![b'q'; req_bytes];
            loop {
                n = n.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let model = format!("sim-{}", prov_lo + (n >> 33) % span);
                let t = Instant::now();
                match net.discover(model.clone()) {
                    Ok(peers) if !peers.is_empty() => {
                        let peer = peers[(n as usize >> 8) % peers.len()].libp2p_peer_id.clone();
                        match net.proxy_forward_timeout(peer, req.clone(), Duration::from_secs(30)) {
                            Ok(_) => {
                                stats.req_ok.fetch_add(1, Relaxed);
                                stats.req_ms.lock().unwrap().push(t.elapsed().as_millis() as u64);
                            }
                            Err(e) => {
                                stats.req_err.fetch_add(1, Relaxed);
                                note_err(&stats, &e);
                            }
                        }
                    }
                    Ok(_) => { stats.req_no_provider.fetch_add(1, Relaxed); }
                    Err(e) => {
                        stats.req_err.fetch_add(1, Relaxed);
                        note_err(&stats, &format!("discover: {e}"));
                    }
                }
                let jitter = 0.5 + ((n >> 20) % 1000) as f64 / 1000.0;
                std::thread::sleep(Duration::from_secs_f64(req_every * jitter));
            }
        }
        "lookup" => {
            // A node that has just joined looks up 10 models it hasn't heard of (from --providers
            // A-B): the bootstraps' registry and the normal discovery, each timed. One OH_SIM_LOOKUP
            // line (stderr) per model.
            // Wait for a bootstrap connection first (the registry answers empty at once without one),
            // then a moment for its protocols to be negotiated.
            let joined = Instant::now();
            while joined.elapsed() < Duration::from_secs(60)
                && !net.status().is_ok_and(|s| s.peers.iter().any(|p| lookup_bootstraps.contains(&p.peer_id)))
            {
                std::thread::sleep(Duration::from_millis(200));
            }
            let connected_after_ms = joined.elapsed().as_millis() as u64;
            std::thread::sleep(Duration::from_secs(2));
            let span = prov_hi.saturating_sub(prov_lo).max(1);
            let mut n: u64 = i.wrapping_mul(2654435761).wrapping_add(17);
            for _ in 0..10 {
                n = n.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let model = format!("sim-{}", prov_lo + (n >> 33) % span);
                // Discovery first: the registry's answer goes into the node's cache, which discovery
                // would then answer from.
                let t = Instant::now();
                let disc = net.discover(model.clone());
                let disc_ms = t.elapsed().as_millis() as u64;
                let t = Instant::now();
                let reg = net.query_registry(model.clone());
                let reg_ms = t.elapsed().as_millis() as u64;
                let line = serde_json::json!({
                    "connected_after_ms": connected_after_ms,
                    "model": model,
                    "registry_ms": reg_ms,
                    "registry_found": reg.as_ref().map(|v| v.len()).unwrap_or(0),
                    "registry_err": reg.err(),
                    "discover_ms": disc_ms,
                    "discover_found": disc.as_ref().map(|v| v.len()).unwrap_or(0),
                    "discover_err": disc.err(),
                });
                eprintln!("OH_SIM_LOOKUP {line}");
            }
            loop {
                std::thread::sleep(Duration::from_secs(3600));
            }
        }
        "flooder" => {
            // A new model every 1/flood_rate s, each a signed, valid announce from this one author.
            std::thread::sleep(Duration::from_secs(10));
            let mut k: u64 = 0;
            loop {
                let r = PeerRecord {
                    peer_id: net.openhydra_peer_id().to_string(),
                    model_id: format!("flood-{i}-{k}"),
                    host: "0.0.0.0".into(),
                    port: 18001,
                    libp2p_peer_id: net.libp2p_peer_id().to_string(),
                    public_key: net.public_key_hex().to_string(),
                    ..Default::default()
                };
                match net.announce(r) {
                    Ok(()) => { stats.flood_sent.fetch_add(1, Relaxed); }
                    Err(e) => note_err(&stats, &format!("flood: {e}")),
                }
                k += 1;
                std::thread::sleep(Duration::from_secs_f64(1.0 / flood_rate.max(0.01)));
            }
        }
        "junk" => {
            // Invalid gossip in rotation: random bytes, oversized, an unknown type, an announce
            // forged for another peer, and an announce a day old.
            std::thread::sleep(Duration::from_secs(10));
            let mut n: u64 = i.wrapping_mul(0x9E3779B97F4A7C15) | 1;
            let other = "12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb";
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
            loop {
                n ^= n << 13; n ^= n >> 7; n ^= n << 17;
                let payload: Vec<u8> = match n % 5 {
                    0 => (0..512).map(|j| (n >> (j % 56)) as u8).collect(),
                    1 => serde_json::to_vec(&serde_json::json!({ "type": "PEER_DEPARTED", "pad": "x".repeat(20_000) })).unwrap(),
                    2 => serde_json::to_vec(&serde_json::json!({ "type": "SIM_UNKNOWN", "n": n })).unwrap(),
                    3 => serde_json::to_vec(&serde_json::json!({ "type": "PROVIDER_ANNOUNCE", "record": {
                        "peer_id": other, "model_id": format!("junk-{n}"), "libp2p_peer_id": other, "signature": "00" } })).unwrap(),
                    _ => serde_json::to_vec(&serde_json::json!({ "type": "MODEL_WITHDRAWN", "libp2p_peer_id": other,
                        "model_id": "sim-1", "timestamp": now / 1000 - 86_400 })).unwrap(),
                };
                match net.publish_raw(payload) {
                    Ok(()) => { stats.flood_sent.fetch_add(1, Relaxed); }
                    Err(e) => note_err(&stats, &format!("junk: {e}")),
                }
                std::thread::sleep(Duration::from_secs_f64(1.0 / flood_rate.max(0.01)));
            }
        }
        _ => loop {
            std::thread::sleep(Duration::from_secs(3600));
        },
    }
}
