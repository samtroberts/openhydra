// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! `registry` — run a reference registry for the log-prob identity check.
//!
//! The registry holds batches of reference texts (from `refbuild assemble`) and hands verifying
//! gateways signed round packs, issuing every text once network-wide (see
//! [`openhydra_protocol::refpool`]). State lives in one directory: `pool.redb` (the texts — secret
//! until issued) and `registry.key` (the Ed25519 signing key), both created owner-only.
//!
//! `registry serve` speaks plain HTTP; put it behind a TLS terminator for anything but localhost —
//! gateways refuse a non-HTTPS registry that isn't on loopback.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::{Args, Subcommand};
use ed25519_dalek::SigningKey;
use openhydra_agent::prompt_score::load_reference;
use openhydra_agent::registry_client::pack_reply;
use openhydra_protocol::logprob::ModelIdentity;
use openhydra_protocol::refpool::{sign_pack, PoolError, RefPool};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Args)]
pub struct RegistryArgs {
    #[command(subcommand)]
    action: RegistryAction,
}

#[derive(Args)]
struct DirArg {
    /// Registry state directory (pool.redb + registry.key).
    #[arg(long)]
    dir: PathBuf,
}

#[derive(Subcommand)]
enum RegistryAction {
    /// Create the state directory: an empty pool and a new signing key. Prints the public key.
    Init(DirArg),
    /// Print the registry's public key (gateways pin it with `--logprob-registry-pubkey`).
    Pubkey(DirArg),
    /// Add a reference file from `refbuild assemble` to the pool as a new batch.
    Import(ImportArgs),
    /// Show every batch and how many texts are left.
    Status(DirArg),
    /// Manage gateway API keys.
    Keys(KeysArgs),
    /// Serve round packs over HTTP.
    Serve(ServeArgs),
}

#[derive(Args)]
struct ImportArgs {
    #[command(flatten)]
    dir: DirArg,
    /// Reference file from `refbuild assemble`.
    #[arg(long)]
    reference: PathBuf,
}

#[derive(Args)]
struct KeysArgs {
    #[command(subcommand)]
    action: KeysAction,
}

#[derive(Subcommand)]
enum KeysAction {
    /// Create a key for a gateway; prints it once (only its hash is stored).
    Add(KeyNameArgs),
    /// List keys (names and state, never the keys).
    List(DirArg),
    /// Revoke a gateway's key.
    Revoke(KeyNameArgs),
}

#[derive(Args)]
struct KeyNameArgs {
    #[command(flatten)]
    dir: DirArg,
    #[arg(long)]
    name: String,
}

#[derive(Args)]
struct ServeArgs {
    #[command(flatten)]
    dir: DirArg,
    /// Address to listen on.
    #[arg(long, default_value = "127.0.0.1:7870")]
    bind: String,
    /// Extra texts per pack beyond one round, so a refused challenge doesn't sink the round.
    #[arg(long, default_value_t = 4)]
    spare: u32,
    /// Packs one gateway key may take per hour.
    #[arg(long, default_value_t = 60)]
    packs_per_hour: u32,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn pool_path(dir: &Path) -> PathBuf {
    dir.join("pool.redb")
}

fn key_path(dir: &Path) -> PathBuf {
    dir.join("registry.key")
}

fn open_pool(dir: &Path) -> Result<RefPool, String> {
    let p = pool_path(dir);
    if !p.exists() {
        return Err(format!("{} not found — run `registry init --dir {}` first", p.display(), dir.display()));
    }
    RefPool::open(&p).map_err(|e| format!("{}: {e:?}", p.display()))
}

fn load_key(dir: &Path) -> Result<SigningKey, String> {
    let p = key_path(dir);
    let raw = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let bytes: [u8; 32] = raw.as_slice().try_into().map_err(|_| format!("{}: expected 32 bytes", p.display()))?;
    Ok(SigningKey::from_bytes(&bytes))
}

/// Write `bytes` to a new file readable only by the owner.
fn write_private(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn sha256_hex(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

pub fn run(args: RegistryArgs) -> Result<(), String> {
    match args.action {
        RegistryAction::Init(d) => {
            std::fs::create_dir_all(&d.dir).map_err(|e| format!("{}: {e}", d.dir.display()))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&d.dir, std::fs::Permissions::from_mode(0o700));
            }
            if key_path(&d.dir).exists() || pool_path(&d.dir).exists() {
                return Err(format!("{} already holds a registry", d.dir.display()));
            }
            let key = SigningKey::from_bytes(&rand::random::<[u8; 32]>());
            write_private(&key_path(&d.dir), &key.to_bytes())?;
            RefPool::open(pool_path(&d.dir)).map_err(|e| format!("{e:?}"))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(pool_path(&d.dir), std::fs::Permissions::from_mode(0o600));
            }
            println!("{}", hex::encode(key.verifying_key().to_bytes()));
            eprintln!("registry: initialised {} (public key above — gateways pin it)", d.dir.display());
            Ok(())
        }
        RegistryAction::Pubkey(d) => {
            println!("{}", hex::encode(load_key(&d.dir)?.verifying_key().to_bytes()));
            Ok(())
        }
        RegistryAction::Import(a) => {
            let pool = open_pool(&a.dir.dir)?;
            let r = load_reference(&a.reference)?;
            let info = pool
                .import(&r.set, &r.handles, rand::random::<[u8; 16]>(), now_ms())
                .map_err(|e| format!("import: {e:?}"))?;
            eprintln!(
                "registry: imported batch {} — {} texts of {} ({}), cut-off x{:.2}, {} per round",
                hex::encode(info.batch_id),
                info.total,
                info.label,
                info.identity.as_ref().map(|i| i.key()).unwrap_or_else(|| info.handles.join(", ")),
                info.cutoff,
                info.min_texts
            );
            Ok(())
        }
        RegistryAction::Status(d) => {
            let pool = open_pool(&d.dir)?;
            for b in pool.status().map_err(|e| format!("{e:?}"))? {
                println!(
                    "{}  {:24} {:22} cut-off x{:.2}  {}/{} texts left (≈{} rounds)",
                    hex::encode(b.info.batch_id),
                    b.info.label,
                    b.info.identity.as_ref().map(|i| i.key()).unwrap_or_else(|| b.info.handles.join(",")),
                    b.info.cutoff,
                    b.remaining,
                    b.info.total,
                    b.remaining / b.info.min_texts.max(1)
                );
            }
            Ok(())
        }
        RegistryAction::Keys(k) => match k.action {
            KeysAction::Add(a) => {
                let pool = open_pool(&a.dir.dir)?;
                let key = format!("ohreg_{}", hex::encode(rand::random::<[u8; 32]>()));
                pool.add_key(&a.name, &sha256_hex(&key), now_ms()).map_err(|e| format!("{e:?}"))?;
                println!("{key}");
                eprintln!("registry: key for {} created — it is shown only once", a.name);
                Ok(())
            }
            KeysAction::List(d) => {
                for k in open_pool(&d.dir)?.keys().map_err(|e| format!("{e:?}"))? {
                    println!("{:24} {}", k.name, if k.revoked { "revoked" } else { "active" });
                }
                Ok(())
            }
            KeysAction::Revoke(a) => {
                open_pool(&a.dir.dir)?.revoke_key(&a.name).map_err(|e| format!("{e:?}"))?;
                eprintln!("registry: revoked {}", a.name);
                Ok(())
            }
        },
        RegistryAction::Serve(a) => serve(a),
    }
}

// ── HTTP service ──

struct Registry {
    pool: RefPool,
    key: SigningKey,
    spare: u32,
    packs_per_hour: u32,
    /// Per-key token bucket: (tokens, last refill).
    buckets: Mutex<HashMap<String, (f64, Instant)>>,
}

impl Registry {
    /// The key's name if the bearer key is valid and not revoked.
    fn authorize(&self, headers: &HeaderMap) -> Result<String, StatusCode> {
        let bearer = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .ok_or(StatusCode::UNAUTHORIZED)?;
        match self.pool.key(&sha256_hex(bearer.trim())) {
            Ok(Some(rec)) => Ok(rec.name),
            Ok(None) => Err(StatusCode::UNAUTHORIZED),
            Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
        }
    }

    /// Take one pack token for `name`; false when over the hourly rate.
    fn take_token(&self, name: &str) -> bool {
        let cap = self.packs_per_hour.max(1) as f64;
        let mut b = self.buckets.lock().unwrap_or_else(|e| e.into_inner());
        let (tokens, last) = b.entry(name.to_string()).or_insert((cap, Instant::now()));
        *tokens = (*tokens + last.elapsed().as_secs_f64() * cap / 3600.0).min(cap);
        *last = Instant::now();
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

fn router(reg: Arc<Registry>) -> Router {
    Router::new()
        .route("/v1/health", get(health))
        .route("/v1/pool", get(pool_view))
        .route("/v1/packs", post(packs))
        .with_state(reg)
}

async fn health(State(reg): State<Arc<Registry>>) -> Json<Value> {
    Json(json!({"ok": true, "key": hex::encode(reg.key.verifying_key().to_bytes())}))
}

async fn pool_view(State(reg): State<Arc<Registry>>, headers: HeaderMap) -> Result<Json<Value>, StatusCode> {
    reg.authorize(&headers)?;
    let batches = reg.pool.status().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(json!(batches
        .iter()
        .map(|b| json!({
            "label": b.info.label,
            "identity": b.info.identity.as_ref().map(|i| i.key()),
            "handles": b.info.handles,
            "min_texts": b.info.min_texts,
            "remaining": b.remaining,
            "rounds_left": b.remaining / b.info.min_texts.max(1),
        }))
        .collect::<Vec<_>>())))
}

async fn packs(
    State(reg): State<Arc<Registry>>,
    headers: HeaderMap,
    Json(req): Json<Value>,
) -> Result<Json<Value>, StatusCode> {
    let name = reg.authorize(&headers)?;
    // The rest of an undecided round: `{"continue": "<first pack id hex>"}`.
    if let Some(first) = req.get("continue").and_then(Value::as_str) {
        let first: [u8; 16] = hex::decode(first).ok().and_then(|b| b.try_into().ok()).ok_or(StatusCode::BAD_REQUEST)?;
        return match reg.pool.continue_pack(first, reg.spare, &name, rand::random::<[u8; 16]>(), now_ms()) {
            Ok(pack) => {
                let (bytes, sig) = sign_pack(&pack, &reg.key);
                Ok(Json(pack_reply(&bytes, &sig, &reg.key.verifying_key().to_bytes())))
            }
            Err(PoolError::Exhausted) => Ok(Json(json!({"status": "exhausted"}))),
            Err(PoolError::Key(_)) => Err(StatusCode::FORBIDDEN),
            Err(e) => {
                tracing::error!(error = ?e, "continuing a pack failed");
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        };
    }
    let identity = req.get("identity").filter(|v| !v.is_null()).map(|v| {
        let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("");
        ModelIdentity::new(s("family"), s("params"), s("quant"))
    });
    let identity = match identity {
        Some(None) => return Err(StatusCode::BAD_REQUEST),
        Some(Some(id)) => Some(id),
        None => None,
    };
    let handle = req.get("handle").and_then(Value::as_str).filter(|h| !h.is_empty() && h.len() <= 256);
    if identity.is_none() && handle.is_none() {
        return Err(StatusCode::BAD_REQUEST);
    }
    if !reg.take_token(&name) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }
    match reg.pool.issue_pack(identity.as_ref(), handle, reg.spare, &name, rand::random::<[u8; 16]>(), now_ms()) {
        Ok(pack) => {
            let (bytes, sig) = sign_pack(&pack, &reg.key);
            tracing::info!(gateway = name, label = pack.label, texts = pack.texts.len(), "issued a round pack");
            Ok(Json(pack_reply(&bytes, &sig, &reg.key.verifying_key().to_bytes())))
        }
        Err(PoolError::NoReference) => Ok(Json(json!({"status": "no_reference"}))),
        Err(PoolError::Exhausted) => {
            tracing::warn!(gateway = name, identity = ?identity.map(|i| i.key()), handle, "pool exhausted for this model");
            Ok(Json(json!({"status": "exhausted"})))
        }
        Err(e) => {
            tracing::error!(error = ?e, "issuing a pack failed");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

fn serve(a: ServeArgs) -> Result<(), String> {
    let reg = Arc::new(Registry {
        pool: open_pool(&a.dir.dir)?,
        key: load_key(&a.dir.dir)?,
        spare: a.spare,
        packs_per_hour: a.packs_per_hour,
        buckets: Mutex::new(HashMap::new()),
    });
    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&a.bind).await.map_err(|e| format!("{}: {e}", a.bind))?;
        eprintln!(
            "registry: serving round packs on http://{} (key {})",
            a.bind,
            hex::encode(reg.key.verifying_key().to_bytes())
        );
        axum::serve(listener, router(reg)).await.map_err(|e| e.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use openhydra_agent::registry_client::RegistryClient;
    use openhydra_protocol::logprob::{ReferenceSet, ReferenceText};

    fn registry(texts: usize, packs_per_hour: u32) -> (Arc<Registry>, String) {
        let pool = RefPool::open_in_memory().unwrap();
        let set = ReferenceSet {
            model_key: "qwen3.5-9b@q4_k_m".into(),
            identity: ModelIdentity::new("qwen3.5", "9b", "q4_k_m"),
            texts: (0..texts)
                .map(|i| {
                    let ids: Vec<u32> = (0..20).map(|k| (i * 50 + k) as u32).collect();
                    ReferenceText { id: format!("t{i}"), logprobs: vec![-0.5; 19], token_ids: ids, band: 0.02 }
                })
                .collect(),
            cutoff: 2.24,
            min_texts: 4,
            early_stop: vec![],
        };
        pool.import(&set, &["Qwen3.5-9B-Q4_K_M".into()], [1; 16], 0).unwrap();
        let key = "ohreg_test".to_string();
        pool.add_key("gw1", &sha256_hex(&key), 0).unwrap();
        let reg = Registry {
            pool,
            key: SigningKey::from_bytes(&[5; 32]),
            spare: 1,
            packs_per_hour,
            buckets: Mutex::new(HashMap::new()),
        };
        (Arc::new(reg), key)
    }

    /// Serve `reg` on an ephemeral loopback port for the duration of `f`.
    fn with_server<T>(reg: Arc<Registry>, f: impl FnOnce(String) -> T + Send + 'static) -> T
    where
        T: Send + 'static,
    {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        rt.spawn(async move { axum::serve(listener, router(reg)).await.unwrap() });
        let out = std::thread::spawn(move || f(url)).join().unwrap();
        rt.shutdown_background();
        out
    }

    #[test]
    fn a_gateway_gets_verified_packs_until_the_pool_runs_out() {
        let (reg, key) = registry(10, 60);
        let pubkey = hex::encode(reg.key.verifying_key().to_bytes());
        with_server(reg, move |url| {
            let client = RegistryClient::new(&url, &key, &pubkey).unwrap();
            let id = ModelIdentity::new("Qwen3.5", "9B", "Q4_K_M");
            let a = client.fetch_pack(id.as_ref(), "whatever").unwrap();
            let b = client.fetch_pack(None, "Qwen3.5-9B-Q4_K_M").unwrap();
            assert_eq!((a.texts.len(), b.texts.len()), (5, 5));
            assert_eq!(a.issued_to, "gw1");
            assert!(a.texts.iter().all(|t| b.texts.iter().all(|u| u.digest() != t.digest())));
            assert_eq!(client.fetch_pack(id.as_ref(), "x"), Err(openhydra_agent::registry_client::RegistryError::Exhausted));
            let other = ModelIdentity::new("qwen3.5", "9b", "q4_0");
            assert_eq!(client.fetch_pack(other.as_ref(), "y"), Err(openhydra_agent::registry_client::RegistryError::NoReference));
        });
    }

    #[test]
    fn an_undecided_first_stage_is_continued_once_over_http() {
        let (reg, key) = registry(4, 60);
        let mut set = ReferenceSet {
            model_key: "g".into(),
            identity: ModelIdentity::new("gemma4", "e4b", "q4_k_m"),
            texts: (0..30)
                .map(|i| {
                    let ids: Vec<u32> = (0..20).map(|k| (10_000 + i * 50 + k) as u32).collect();
                    ReferenceText { id: format!("g{i}"), logprobs: vec![-0.5; 19], token_ids: ids, band: 0.02 }
                })
                .collect(),
            cutoff: 2.0,
            min_texts: 12,
            early_stop: vec![],
        };
        set.early_stop = vec![openhydra_protocol::logprob::Checkpoint { after: 4, pass_below: Some(1.5), fail_above: 2.5 }];
        reg.pool.import(&set, &[], [7; 16], 0).unwrap();
        let pubkey = hex::encode(reg.key.verifying_key().to_bytes());
        with_server(reg, move |url| {
            use openhydra_agent::registry_client::RegistryError;
            let client = RegistryClient::new(&url, &key, &pubkey).unwrap();
            let first = client.fetch_pack(set.identity.as_ref(), "x").unwrap();
            assert_eq!(first.texts.len(), 5, "checkpoint 4 + 1 spare");
            let rest = client.continue_pack(first.pack_id).unwrap();
            assert_eq!(rest.texts.len(), 8, "to min_texts 12 + 1 spare");
            assert!(matches!(client.continue_pack(first.pack_id), Err(RegistryError::Request(e)) if e.contains("403")));
        });
    }

    #[test]
    fn bad_keys_wrong_pins_and_rate_limits_are_refused() {
        let (reg, key) = registry(40, 2);
        let pubkey = hex::encode(reg.key.verifying_key().to_bytes());
        with_server(reg, move |url| {
            use openhydra_agent::registry_client::RegistryError;
            let id = ModelIdentity::new("qwen3.5", "9b", "q4_k_m");
            let bad = RegistryClient::new(&url, "ohreg_wrong", &pubkey).unwrap();
            assert!(matches!(bad.fetch_pack(id.as_ref(), "x"), Err(RegistryError::Request(e)) if e.contains("401")));
            let wrong_pin = RegistryClient::new(&url, &key, &"11".repeat(32)).unwrap();
            assert!(matches!(wrong_pin.fetch_pack(id.as_ref(), "x"), Err(RegistryError::Invalid(_))));
            // The wrong-pin fetch still consumed a pack token: one left, then 429.
            let good = RegistryClient::new(&url, &key, &pubkey).unwrap();
            assert!(good.fetch_pack(id.as_ref(), "x").is_ok());
            assert!(matches!(good.fetch_pack(id.as_ref(), "x"), Err(RegistryError::Request(e)) if e.contains("429")));
        });
    }
}
