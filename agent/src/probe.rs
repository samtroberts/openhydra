// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! `openhydra-agent probe` (soft launch, SL10): one look at the public network from outside, for
//! monitoring.
//!
//! A probe is an ordinary node that joins the public network, waits for it to settle, and reports:
//! - each OpenHydra bootstrap: connected or not, and whether its relay granted a reservation (a
//!   full or broken relay refuses one);
//! - the signed seed list: fetched, and which seed models have a seed provider online;
//! - optionally, one short request to a seed model, sent only to seed providers (as
//!   `openhydra/auto` would for a newcomer), and how long it took;
//! - totals the network showed this node: peers, models, providers (aggregates, no peer ids). From
//!   gossip by default, which reaches a fresh node partially and late; `count_models` also asks the
//!   bootstrap registry about every model name the node knows of (the signed tokenizer index, the
//!   seed list, gossip), a fuller lower bound for the daily count.
//!
//! It prints the report as JSON and exits non-zero when something needs a person, so a timer can
//! alert on the exit code. Nothing it learns about individual peers leaves the report.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use openhydra_network::types::StatusSnapshot;
use openhydra_network::handle::NetworkHandle;
use serde::Serialize;

use crate::adapter::ChatMessage;
use crate::consumer::ConsumerNode;
use crate::seed_list::SeedStore;

/// The production bootstraps by peer id, with the names alerts use.
pub const BOOTSTRAPS: &[(&str, &str)] = &[
    ("12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb", "us-dallas"),
    ("12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT", "eu-london"),
    ("12D3KooWPgqZBgLZ1f94AQ7sbeyEz5UJ4jiT4d3zuQp2t61VLPZo", "ap-singapore"),
    ("12D3KooWHNQ9nMedT3ZaMjj4xX7Mf4dxmTZ3cAX3tnJfyERV9Bua", "de-netcup"),
];

/// What to check, beyond the bootstraps.
pub struct ProbeOptions {
    /// How long to let the node connect and reserve before judging (it stops waiting early once
    /// every bootstrap is connected and reserved).
    pub settle: Duration,
    /// How long to listen before counting the totals: providers announce every two minutes, so a
    /// count taken as soon as the bootstraps answer misses most of them. Zero for alert runs.
    pub observe: Duration,
    /// How many seed models must have a seed provider online; fewer is a problem. 0 while no seed
    /// is meant to be running.
    pub expect_seeds: usize,
    /// Send one short request to a seed model that has a provider online.
    pub test_request: bool,
    /// Count models and providers through the bootstrap registry too (one lookup per known name:
    /// ~170, for a daily count rather than every alert run).
    pub count_models: bool,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct BootstrapRow {
    pub name: String,
    pub connected: bool,
    pub relay_reserved: bool,
}

#[derive(Debug, Serialize)]
pub struct SeedRow {
    pub model: String,
    /// Seed providers of this model the bootstrap registry knows now.
    pub seed_providers: usize,
}

#[derive(Debug, Serialize)]
pub struct TestResult {
    pub model: String,
    pub ok: bool,
    pub ms: u64,
    pub tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Totals {
    pub peers: usize,
    pub models: usize,
    pub providers: usize,
    /// Where the model and provider counts come from: `gossip`, or `gossip+registry`.
    pub source: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub version: &'static str,
    pub at_ms: u64,
    pub settled_ms: u64,
    pub bootstraps: Vec<BootstrapRow>,
    pub seed_list: Option<u64>,
    pub seeds: Vec<SeedRow>,
    pub test: Option<TestResult>,
    pub totals: Totals,
    /// What needs a person, one line each; empty when all is well.
    pub problems: Vec<String>,
}

/// The bootstraps' rows from a status snapshot.
pub fn bootstrap_rows(s: &StatusSnapshot) -> Vec<BootstrapRow> {
    BOOTSTRAPS
        .iter()
        .map(|(id, name)| {
            let circuit = format!("/p2p/{id}/p2p-circuit");
            BootstrapRow {
                name: name.to_string(),
                connected: s.peers.iter().any(|p| p.peer_id == *id),
                relay_reserved: s.relay_reservations.iter().any(|r| r.contains(&circuit)),
            }
        })
        .collect()
}

/// Whether every bootstrap is connected and has granted a reservation (the probe can stop waiting).
fn settled(rows: &[BootstrapRow]) -> bool {
    rows.iter().all(|b| b.connected && b.relay_reserved)
}

/// The problems a report shows, one line each.
pub fn problems(
    rows: &[BootstrapRow],
    seed_list: Option<u64>,
    seed_fetch: &Result<(), String>,
    seeds: &[SeedRow],
    test: Option<&TestResult>,
    expect_seeds: usize,
) -> Vec<String> {
    let mut out = Vec::new();
    for b in rows {
        if !b.connected {
            out.push(format!("bootstrap {} unreachable", b.name));
        } else if !b.relay_reserved {
            out.push(format!("bootstrap {} refused a relay reservation (full or broken relay)", b.name));
        }
    }
    if let Err(e) = seed_fetch {
        out.push(format!("seed list not fetched: {e}"));
    }
    if seed_list.unwrap_or(0) == 0 {
        out.push("no valid seed list".into());
    }
    let online = seeds.iter().filter(|s| s.seed_providers > 0).count();
    if online < expect_seeds {
        out.push(format!("{online} of {expect_seeds} expected seed models online"));
    }
    if let Some(t) = test.filter(|t| !t.ok) {
        out.push(format!("test request to {} failed: {}", t.model, t.error.as_deref().unwrap_or("no answer")));
    }
    out
}

/// Run the probe on a started node. Blocking.
pub fn run(net: NetworkHandle, store: Arc<SeedStore>, client: &reqwest::blocking::Client, opts: &ProbeOptions) -> Result<Report, String> {
    let at_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let seed_fetch = store.refresh(client).map(|_| ());
    let start = Instant::now();
    let mut status = net.status()?;
    while !settled(&bootstrap_rows(&status)) && start.elapsed() < opts.settle {
        std::thread::sleep(Duration::from_millis(500));
        status = net.status()?;
    }
    let settled_ms = start.elapsed().as_millis() as u64;
    let rows = bootstrap_rows(&status);
    let list = store.current();
    let seed_list = (list.sequence > 0).then_some(list.sequence);
    let keys = list.public_keys();
    let is_seed = |key: &str| keys.contains(&key.to_ascii_lowercase());
    // Which seed models have a seed provider online, and who (only kept for the test request).
    let mut seeds = Vec::new();
    let mut seed_peers: BTreeMap<String, HashSet<String>> = BTreeMap::new();
    for model in list.models() {
        let found = net.query_registry(model.as_str()).unwrap_or_default();
        let ids: HashSet<String> = found.into_iter().filter(|p| is_seed(&p.public_key)).map(|p| p.libp2p_peer_id).collect();
        seeds.push(SeedRow { model: model.clone(), seed_providers: ids.len() });
        if !ids.is_empty() {
            seed_peers.insert(model, ids);
        }
    }
    if let Some(left) = opts.observe.checked_sub(start.elapsed()) {
        std::thread::sleep(left);
    }
    let status = net.status()?;
    let mut models: HashSet<String> = status.known_models.iter().cloned().collect();
    let mut providers: HashSet<String> = status.known_providers.iter().map(|p| p.libp2p_peer_id.clone()).collect();
    if opts.count_models {
        let mut names: std::collections::BTreeSet<String> = models.iter().cloned().collect();
        names.extend(list.models());
        names.extend(crate::tokenizer_index::bundled().models.into_iter().flat_map(|e| e.names));
        for name in names {
            let found = net.query_registry(name.as_str()).unwrap_or_default();
            if !found.is_empty() {
                models.insert(name);
                providers.extend(found.into_iter().map(|p| p.libp2p_peer_id));
            }
        }
    }
    let totals = Totals {
        peers: status.peers.len(),
        models: models.len(),
        providers: providers.len(),
        source: if opts.count_models { "gossip+registry" } else { "gossip" },
    };
    let test = match seed_peers.into_iter().next() {
        Some((model, only)) if opts.test_request => Some(test_request(net, &model, &only)),
        _ => None,
    };
    let problems = problems(&rows, seed_list, &seed_fetch, &seeds, test.as_ref(), opts.expect_seeds);
    Ok(Report { version: env!("CARGO_PKG_VERSION"), at_ms, settled_ms, bootstraps: rows, seed_list, seeds, test, totals, problems })
}

/// One short request to `model`, only to the seed providers `only` (as `openhydra/auto` does).
fn test_request(net: NetworkHandle, model: &str, only: &HashSet<String>) -> TestResult {
    let node = Arc::new(ConsumerNode::new(net));
    let messages = vec![ChatMessage::new("user", "Reply with one word: ok")];
    let start = Instant::now();
    let result = node.complete(model, messages, Some(8), Some(0.0), Vec::new(), Some(false), None, Some(only), None, &|| false, &mut |_| {});
    let ms = start.elapsed().as_millis() as u64;
    match result {
        Ok(s) => TestResult { model: model.to_string(), ok: s.ok, ms, tokens: s.tokens, error: None },
        Err(e) => TestResult { model: model.to_string(), ok: false, ms, tokens: 0, error: Some(e.to_string()) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openhydra_network::types::PeerStatus;

    fn snapshot(peers: &[(&str, &[&str])], reserved: &[&str]) -> StatusSnapshot {
        StatusSnapshot {
            peers: peers
                .iter()
                .map(|(id, ips)| PeerStatus { peer_id: id.to_string(), direct_ips: ips.iter().map(|s| s.to_string()).collect(), ..Default::default() })
                .collect(),
            relay_reservations: reserved.iter().map(|id| format!("/ip4/1.2.3.4/tcp/4001/p2p/{id}/p2p-circuit")).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn a_bootstrap_down_or_refusing_a_reservation_is_a_problem() {
        let (us, eu, ap, de) = (BOOTSTRAPS[0].0, BOOTSTRAPS[1].0, BOOTSTRAPS[2].0, BOOTSTRAPS[3].0);
        let s = snapshot(&[(us, &["45.79.190.172", "2600:3c03::1"]), (eu, &["172.105.69.49"]), (ap, &[])], &[us, eu]);
        let rows = bootstrap_rows(&s);
        assert_eq!(rows[0], BootstrapRow { name: "us-dallas".into(), connected: true, relay_reserved: true });
        assert!(rows[2].connected && !rows[2].relay_reserved, "connected, but no reservation");
        assert!(!rows[3].connected);
        assert!(!settled(&rows));
        let p = problems(&rows, Some(1), &Ok(()), &[], None, 0);
        assert_eq!(p, vec!["bootstrap ap-singapore refused a relay reservation (full or broken relay)", "bootstrap de-netcup unreachable"]);
        let all = snapshot(&[(us, &["a"]), (eu, &["b"]), (ap, &["c"]), (de, &["d"])], &[us, eu, ap, de]);
        assert!(settled(&bootstrap_rows(&all)) && problems(&bootstrap_rows(&all), Some(1), &Ok(()), &[], None, 0).is_empty());
    }

    #[test]
    fn missing_seeds_and_a_failed_test_are_problems_only_when_expected() {
        let rows = bootstrap_rows(&snapshot(&BOOTSTRAPS.iter().map(|(id, _)| (*id, &["x"][..])).collect::<Vec<_>>(), &BOOTSTRAPS.iter().map(|(id, _)| *id).collect::<Vec<_>>()));
        let none = [SeedRow { model: "qwen3:8b".into(), seed_providers: 0 }];
        // Seeds scaled to zero on purpose: nothing to alert on.
        assert!(problems(&rows, Some(1), &Ok(()), &none, None, 0).is_empty());
        assert_eq!(problems(&rows, Some(1), &Ok(()), &none, None, 1), vec!["0 of 1 expected seed models online"]);
        let up = [SeedRow { model: "qwen3:8b".into(), seed_providers: 1 }];
        let failed = TestResult { model: "qwen3:8b".into(), ok: false, ms: 30_000, tokens: 0, error: Some("timed out".into()) };
        assert_eq!(problems(&rows, Some(1), &Ok(()), &up, Some(&failed), 1), vec!["test request to qwen3:8b failed: timed out"]);
        assert_eq!(problems(&rows, None, &Err("HTTP 404".into()), &up, None, 0), vec!["seed list not fetched: HTTP 404", "no valid seed list"]);
    }
}
