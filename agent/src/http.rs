// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The live HTTP transport: a [`reqwest`] blocking client implementing
//! [`HttpClient`](crate::adapter::HttpClient).
//!
//! Blocking (not async) so it drops straight into the sync adapter trait; the async
//! gateway (R1) will call it via `spawn_blocking`. Streaming is real — `post_stream`
//! hands back a `BufReader::lines()` iterator over the response body, so Ollama's
//! newline-delimited chunks are pulled as they arrive rather than buffered.

use std::io::BufRead;
use std::time::Duration;

use reqwest::blocking::Client;

use crate::adapter::{AdapterError, HttpClient};

fn http_err(e: reqwest::Error) -> AdapterError {
    AdapterError::Http(e.to_string())
}

// Per-request total timeouts (A1). reqwest's *blocking* client exposes no idle/read
// timeout, so each request bounds its **total** duration instead — a wedged engine can no
// longer pin a serve worker forever. Values are generous so legitimate work never trips
// them. See docs/CODEBASE_HARDENING_PLAN.md (A1).

/// Quick GETs: detection probes, ComfyUI history poll, image fetch.
const GET_TIMEOUT: Duration = Duration::from_secs(30);
/// Non-streaming POSTs — some engines return a whole completion in the body.
const POST_JSON_TIMEOUT: Duration = Duration::from_secs(120);
/// Streaming completions: a **total** cap (not idle). Local generations finish well inside
/// this; a stalled stream is reclaimed here instead of hanging a worker forever.
const STREAM_TIMEOUT: Duration = Duration::from_secs(600);

/// Production HTTP transport for engine adapters.
pub struct ReqwestClient {
    client: Client,
}

impl ReqwestClient {
    /// Build a client with a short connect timeout (the engine is local). Per-request total
    /// timeouts (the module `*_TIMEOUT` consts) bound each call so a wedged engine can't pin
    /// a serve worker forever.
    pub fn new() -> Result<Self, AdapterError> {
        Self::with_connect_timeout(Duration::from_secs(5))
    }

    /// Like [`new`](Self::new) but with an explicit connect timeout — used by engine
    /// auto-detection, which probes several ports and wants to fail fast on a dead one.
    /// Per-request timeouts still apply, so an adapter built this way serves long streams
    /// fine while a stalled request stays bounded.
    pub fn with_connect_timeout(connect: Duration) -> Result<Self, AdapterError> {
        let client = Client::builder().connect_timeout(connect).build().map_err(http_err)?;
        Ok(Self { client })
    }
}

impl HttpClient for ReqwestClient {
    fn get(&self, url: &str) -> Result<String, AdapterError> {
        self.client
            .get(url)
            .timeout(GET_TIMEOUT)
            .send()
            .map_err(http_err)?
            .error_for_status()
            .map_err(http_err)?
            .text()
            .map_err(http_err)
    }

    fn get_bytes(&self, url: &str) -> Result<Vec<u8>, AdapterError> {
        Ok(self
            .client
            .get(url)
            .timeout(GET_TIMEOUT)
            .send()
            .map_err(http_err)?
            .error_for_status()
            .map_err(http_err)?
            .bytes()
            .map_err(http_err)?
            .to_vec())
    }

    fn post_json(&self, url: &str, body: &str) -> Result<String, AdapterError> {
        self.client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.to_string())
            .timeout(POST_JSON_TIMEOUT)
            .send()
            .map_err(http_err)?
            .error_for_status()
            .map_err(http_err)?
            .text()
            .map_err(http_err)
    }

    fn post_stream(
        &self,
        url: &str,
        body: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
        self.post_stream_with_headers(url, body, &[])
    }

    fn get_with_headers(&self, url: &str, headers: &[(&str, &str)]) -> Result<String, AdapterError> {
        let mut req = self.client.get(url).timeout(GET_TIMEOUT);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        req.send()
            .map_err(http_err)?
            .error_for_status()
            .map_err(http_err)?
            .text()
            .map_err(http_err)
    }

    fn post_json_with_headers(
        &self,
        url: &str,
        body: &str,
        headers: &[(&str, &str)],
    ) -> Result<String, AdapterError> {
        let mut req = self
            .client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.to_string())
            .timeout(POST_JSON_TIMEOUT);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        req.send()
            .map_err(http_err)?
            .error_for_status()
            .map_err(http_err)?
            .text()
            .map_err(http_err)
    }

    fn post_stream_with_headers(
        &self,
        url: &str,
        body: &str,
        headers: &[(&str, &str)],
    ) -> Result<Box<dyn Iterator<Item = Result<String, AdapterError>>>, AdapterError> {
        let mut req = self
            .client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.to_string())
            .timeout(STREAM_TIMEOUT);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let resp = req.send().map_err(http_err)?.error_for_status().map_err(http_err)?;
        // The Lines iterator owns the BufReader which owns the Response → 'static, so it
        // can be boxed and pulled lazily as chunks arrive.
        let reader = std::io::BufReader::new(resp);
        Ok(Box::new(
            reader
                .lines()
                .map(|line| line.map_err(|e| AdapterError::Http(e.to_string()))),
        ))
    }
}
