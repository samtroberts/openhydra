// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Tracing initialization, with an optional OpenTelemetry/OTLP span-export layer (#33).
//!
//! The default build installs a single stderr `fmt` layer filtered by `RUST_LOG` (default
//! `warn`) — identical to the old inline init, just composed over a `Registry` so another
//! layer can slot in. The request path is instrumented with `tracing` spans
//! (`complete` → `discover` / `serve_attempt`), so a trace already exists; this module only
//! decides where it goes.
//!
//! Built with `--features otel`, and pointed at a collector via the standard
//! `OTEL_EXPORTER_OTLP_ENDPOINT`, it additionally installs an OpenTelemetry layer that
//! exports those spans over OTLP/HTTP to a Jaeger / Tempo / OpenTelemetry-Collector backend.
//! The OTel dependency chain is heavy, so it stays behind the feature flag and the default
//! build pulls none of it.

/// Initialize the global tracing subscriber. Idempotent (`try_init` — a second call is a
/// no-op), so tests and repeated entry points are safe.
pub fn init() {
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer().with_writer(std::io::stderr));

    #[cfg(feature = "otel")]
    {
        if let Some(otel_layer) = otel::build_layer() {
            let _ = registry.with(otel_layer).try_init();
            return;
        }
    }

    let _ = registry.try_init();
}

#[cfg(feature = "otel")]
mod otel {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry::KeyValue;
    use opentelemetry_otlp::WithExportConfig;
    use opentelemetry_sdk::{trace::TracerProvider as SdkTracerProvider, Resource};
    use tracing::Subscriber;
    use tracing_subscriber::registry::LookupSpan;
    use tracing_subscriber::Layer;

    /// Build the OpenTelemetry tracing layer, or `None` if no OTLP endpoint is configured
    /// (so the feature can be compiled in but stay dormant until pointed at a collector).
    ///
    /// Endpoint from `OTEL_EXPORTER_OTLP_ENDPOINT`; service name from `OTEL_SERVICE_NAME`
    /// (default `openhydra-agent`). Uses the **simple** (synchronous) exporter so init needs
    /// no background async runtime — fine for an agent's request volume; swap to a batch
    /// exporter under heavy load.
    pub fn build_layer<S>() -> Option<impl Layer<S>>
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok()?;
        let service_name =
            std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "openhydra-agent".to_string());

        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .build()
            .map_err(|e| eprintln!("openhydra-agent: OTLP exporter init failed: {e}"))
            .ok()?;

        let provider = SdkTracerProvider::builder()
            .with_simple_exporter(exporter)
            .with_resource(Resource::new(vec![KeyValue::new("service.name", service_name)]))
            .build();

        let tracer = provider.tracer("openhydra-agent");
        // Keep the provider alive for the process so spans flush on export.
        opentelemetry::global::set_tracer_provider(provider);
        Some(tracing_opentelemetry::layer().with_tracer(tracer))
    }
}
