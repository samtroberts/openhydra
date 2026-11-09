//! oh-sim: many real OpenHydra network nodes in one process, for bootstrap load tests
//! (docs: "OpenHydra Bootstrap Load Simulation Plan", 30 Oct 2026). Each simulated peer is a full
//! `NetworkHandle` node — the same transports, Kademlia, gossipsub, relay client and proxy protocol
//! a real agent runs — given only the test bootstrap's addresses, relaying only through it.
//!
//! usage: oh-sim --bootstrap ADDR [--bootstrap ADDR ...]
//!               --role idle|provider|relayed-provider|active|flooder|junk
//!               --count N [--first I] --dir DIR [--rate PER_SEC] [--duration SECS]
//!               [--providers A-B] [--req-every SECS] [--req-bytes B] [--reply-bytes B]
//!               [--sample-every K] [--models K] [--flood-rate PER_SEC] [--nat]
//! `--models K`: a provider announces K models (sim-I, sim-I.1 … sim-I.K-1; record stuffing at K
//! in the hundreds). `--nat`: listen on loopback only, like a peer behind NAT (no dial-backs).
//! `--v6-prefix P`: peer I listens (and so dials, with port reuse) only from address P+I, e.g.
//! `2406:da1b:c0:6601:1::` — many sources in one /64 (needs the prefix routed to this host).
//! `flooder`: announces a new model `--flood-rate` times a second. `junk`: publishes invalid
//! gossip (random bytes, oversized, unknown type, forged and stale announces) at `--flood-rate`.
//! Prints one JSON line of aggregate counters and percentiles per second to stdout.

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
    let opts = Opts { req_every, req_bytes, reply_bytes, prov_lo, prov_hi, models, flood_rate, nat, v6_prefix };
    if bootstraps.is_empty() {
        eprintln!("oh-sim: --bootstrap required");
        std::process::exit(2);
    }
    std::fs::create_dir_all(&dir).expect("dir");
    openhydra_network::node::disable_mdns();

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
    let Opts { req_every, req_bytes, reply_bytes, prov_lo, prov_hi, models, flood_rate, nat, v6_prefix } = opts;
    let listen = if let Some(p) = v6_prefix {
        let a = std::net::Ipv6Addr::from(u128::from(p) + i as u128);
        vec![format!("/ip6/{a}/tcp/0")]
    } else if role == "relayed-provider" || nat {
        // Loopback only: no address another peer could dial directly, so every request to it
        // comes through the test bootstrap's relay (and hole punching has nothing to work with).
        vec!["/ip4/127.0.0.1/tcp/0".to_string()]
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
                    stats.sampled_known_models.fetch_add(k, Relaxed);
                    stats.sampled_known_models.fetch_sub(known, Relaxed);
                    known = k;
                }
            })
            .ok();
    }

    match role {
        "provider" | "relayed-provider" => {
            let record = |model_id: String| PeerRecord {
                peer_id: net.openhydra_peer_id().to_string(),
                model_id,
                host: "0.0.0.0".into(),
                port: 18001,
                libp2p_peer_id: net.libp2p_peer_id().to_string(),
                public_key: net.public_key_hex().to_string(),
                ..Default::default()
            };
            let records: Vec<PeerRecord> =
                (0..models).map(|k| record(if k == 0 { format!("sim-{i}") } else { format!("sim-{i}.{k}") })).collect();
            std::thread::sleep(Duration::from_secs(3));
            let mut next_announce = Instant::now();
            let reply = vec![b'x'; reply_bytes];
            loop {
                if Instant::now() >= next_announce {
                    for r in &records {
                        match net.announce(r.clone()) {
                            Ok(()) => { stats.announces.fetch_add(1, Relaxed); }
                            Err(e) => note_err(&stats, &format!("announce: {e}")),
                        }
                    }
                    next_announce = Instant::now() + Duration::from_secs(120);
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
