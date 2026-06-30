// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Gateway-side observability (#33) — a dependency-free Prometheus exposition.
//!
//! Rather than pull in the `prometheus` crate (and its protobuf chain) for a handful of
//! series, this is a tiny hand-rolled registry: atomic counters and fixed-bucket
//! histograms rendered to the Prometheus text exposition format (v0.0.4) at `/metrics`.
//! Lock-free (all `AtomicU64`), so recording on the request path is cheap and safe from the
//! gateway's worker threads.
//!
//! Scope is the **gateway**: request counts, completion success/error, tokens served, and
//! the pipeline latencies already surfaced in [`ServeSummary`](crate::serve::ServeSummary)
//! (end-to-end wall, discovery, proxy round-trip). OpenTelemetry *tracing* (spans across the
//! HTTP→swarm→engine path) is a heavier follow-up that can hang off these same record
//! points; it is intentionally out of this first cut.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Upper bounds (seconds) for the latency histograms. `+Inf` is implicit (an extra trailing
/// bucket). Spans sub-100ms cache hits up to multi-second cold generations.
const LATENCY_BUCKETS: &[f64] = &[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0];

/// A fixed-bucket cumulative histogram backed by atomics. `counts[i]` holds the number of
/// observations that fell in `(bounds[i-1], bounds[i]]`; the final slot is the `+Inf`
/// overflow. Rendered cumulatively (Prometheus `_bucket` semantics).
struct Histogram {
    /// Per-bucket observation counts; length is `bounds.len() + 1` (the last is `+Inf`).
    counts: Vec<AtomicU64>,
    /// Sum of observations, in microseconds (kept integer so it can be a plain atomic).
    sum_micros: AtomicU64,
    /// Total observation count.
    count: AtomicU64,
    bounds: &'static [f64],
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            counts: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_micros: AtomicU64::new(0),
            count: AtomicU64::new(0),
            bounds,
        }
    }

    /// Record one observation of `seconds` (clamped at 0 — a clock skew never goes negative).
    fn observe(&self, seconds: f64) {
        let seconds = seconds.max(0.0);
        let idx = self
            .bounds
            .iter()
            .position(|&b| seconds <= b)
            .unwrap_or(self.bounds.len());
        self.counts[idx].fetch_add(1, Ordering::Relaxed);
        self.sum_micros.fetch_add((seconds * 1e6) as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    fn render(&self, name: &str, help: &str, out: &mut String) {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
        let mut cumulative = 0u64;
        for (i, &bound) in self.bounds.iter().enumerate() {
            cumulative += self.counts[i].load(Ordering::Relaxed);
            out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}\n"));
        }
        cumulative += self.counts[self.bounds.len()].load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}\n"));
        let sum = self.sum_micros.load(Ordering::Relaxed) as f64 / 1e6;
        out.push_str(&format!("{name}_sum {sum}\n"));
        out.push_str(&format!("{name}_count {}\n", self.count.load(Ordering::Relaxed)));
    }
}

/// The gateway's metric registry. One instance lives in the app state behind an `Arc`; all
/// fields are atomic so recording never blocks the request path.
pub struct Metrics {
    requests_chat: AtomicU64,
    requests_models: AtomicU64,
    completions_ok: AtomicU64,
    completions_error: AtomicU64,
    completion_tokens: AtomicU64,
    request_duration: Histogram,
    discover_duration: Histogram,
    proxy_roundtrip: Histogram,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            requests_chat: AtomicU64::new(0),
            requests_models: AtomicU64::new(0),
            completions_ok: AtomicU64::new(0),
            completions_error: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            request_duration: Histogram::new(LATENCY_BUCKETS),
            discover_duration: Histogram::new(LATENCY_BUCKETS),
            proxy_roundtrip: Histogram::new(LATENCY_BUCKETS),
        }
    }

    /// A `POST /v1/chat/completions` arrived (counted on entry, before routing succeeds).
    pub fn incr_chat(&self) {
        self.requests_chat.fetch_add(1, Ordering::Relaxed);
    }

    /// A `GET /v1/models` arrived.
    pub fn incr_models(&self) {
        self.requests_models.fetch_add(1, Ordering::Relaxed);
    }

    /// A completion finished successfully: record its tokens and the three pipeline latencies
    /// (`wall` end-to-end, plus discovery and proxy round-trip from the [`ServeSummary`]).
    ///
    /// [`ServeSummary`]: crate::serve::ServeSummary
    pub fn record_completion(&self, tokens: u64, wall: Duration, discover_ns: u64, proxy_ns: u64) {
        self.completions_ok.fetch_add(1, Ordering::Relaxed);
        self.completion_tokens.fetch_add(tokens, Ordering::Relaxed);
        self.request_duration.observe(wall.as_secs_f64());
        self.discover_duration.observe(discover_ns as f64 / 1e9);
        self.proxy_roundtrip.observe(proxy_ns as f64 / 1e9);
    }

    /// A completion failed (route/transport/engine error).
    pub fn record_error(&self) {
        self.completions_error.fetch_add(1, Ordering::Relaxed);
    }

    /// Render the whole registry to the Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(2048);
        let counter = |out: &mut String, name: &str, help: &str, v: &AtomicU64| {
            out.push_str(&format!(
                "# HELP {name} {help}\n# TYPE {name} counter\n{name} {}\n",
                v.load(Ordering::Relaxed)
            ));
        };
        counter(&mut out, "openhydra_chat_requests_total", "Chat-completion requests received.", &self.requests_chat);
        counter(&mut out, "openhydra_models_requests_total", "Model-list requests received.", &self.requests_models);
        counter(&mut out, "openhydra_completions_total", "Completions served successfully.", &self.completions_ok);
        counter(&mut out, "openhydra_completion_errors_total", "Completions that failed.", &self.completions_error);
        counter(&mut out, "openhydra_completion_tokens_total", "Completion tokens served.", &self.completion_tokens);
        self.request_duration.render("openhydra_request_duration_seconds", "End-to-end completion wall time.", &mut out);
        self.discover_duration.render("openhydra_discover_seconds", "Provider-discovery latency.", &mut out);
        self.proxy_roundtrip.render("openhydra_proxy_roundtrip_seconds", "Provider proxy round-trip latency.", &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_render_and_increment() {
        let m = Metrics::new();
        m.incr_chat();
        m.incr_chat();
        m.record_error();
        let out = m.render_prometheus();
        assert!(out.contains("openhydra_chat_requests_total 2"), "{out}");
        assert!(out.contains("openhydra_completion_errors_total 1"), "{out}");
        assert!(out.contains("openhydra_completions_total 0"), "{out}");
    }

    #[test]
    fn completion_records_tokens_and_latencies() {
        let m = Metrics::new();
        m.record_completion(128, Duration::from_millis(300), 1_000_000, 600_000_000);
        let out = m.render_prometheus();
        assert!(out.contains("openhydra_completions_total 1"), "{out}");
        assert!(out.contains("openhydra_completion_tokens_total 128"), "{out}");
        assert!(out.contains("openhydra_request_duration_seconds_count 1"), "{out}");
    }

    #[test]
    fn histogram_buckets_are_cumulative_and_correctly_placed() {
        let m = Metrics::new();
        // 0.3s falls in the (0.25, 0.5] bucket: le="0.5" counts it, le="0.25" does not.
        m.record_completion(1, Duration::from_millis(300), 0, 0);
        let out = m.render_prometheus();
        assert!(out.contains("openhydra_request_duration_seconds_bucket{le=\"0.25\"} 0"), "{out}");
        assert!(out.contains("openhydra_request_duration_seconds_bucket{le=\"0.5\"} 1"), "{out}");
        assert!(out.contains("openhydra_request_duration_seconds_bucket{le=\"+Inf\"} 1"), "{out}");
    }

    #[test]
    fn observation_above_all_bounds_lands_in_inf_only() {
        let m = Metrics::new();
        m.record_completion(1, Duration::from_secs(60), 0, 0); // > 30s top bound
        let out = m.render_prometheus();
        assert!(out.contains("openhydra_request_duration_seconds_bucket{le=\"30\"} 0"), "{out}");
        assert!(out.contains("openhydra_request_duration_seconds_bucket{le=\"+Inf\"} 1"), "{out}");
    }
}
