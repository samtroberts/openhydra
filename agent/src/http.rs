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
    // reqwest's text for a timeout doesn't say so: callers tell timeouts by "timed out" (round 10).
    if e.is_timeout() {
        AdapterError::Http(format!("timed out: {e}"))
    } else {
        AdapterError::Http(e.to_string())
    }
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
    /// The total cap on one streamed request ([`STREAM_TIMEOUT`] unless set).
    stream_timeout: Duration,
}

impl ReqwestClient {
    /// Build a client with a short connect timeout (the engine is local). Per-request total
    /// timeouts (the module `*_TIMEOUT` consts) bound each call so a wedged engine can't pin
    /// a serve worker forever.
    pub fn new() -> Result<Self, AdapterError> {
        Self::with_connect_timeout(Duration::from_secs(5))
    }

    /// Like [`new`](Self::new), presenting `key` as `Authorization: Bearer <key>` on every request:
    /// for calls to a local gateway that requires its key.
    pub fn with_bearer(key: &str) -> Result<Self, AdapterError> {
        let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| AdapterError::Http("gateway key is not a valid header value".into()))?;
        value.set_sensitive(true);
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(reqwest::header::AUTHORIZATION, value);
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .default_headers(headers)
            .build()
            .map_err(http_err)?;
        Ok(Self { client, stream_timeout: STREAM_TIMEOUT })
    }

    /// The total cap on a streamed request through this client. A local gateway bounds its own
    /// streams (30 min to 6 h, by `max_tokens`), so its clients wait as long as it does.
    pub fn with_stream_timeout(mut self, total: Duration) -> Self {
        self.stream_timeout = total;
        self
    }

    /// Like [`new`](Self::new) but with an explicit connect timeout — used by engine
    /// auto-detection, which probes several ports and wants to fail fast on a dead one.
    /// Per-request timeouts still apply, so an adapter built this way serves long streams
    /// fine while a stalled request stays bounded.
    pub fn with_connect_timeout(connect: Duration) -> Result<Self, AdapterError> {
        let client = Client::builder().connect_timeout(connect).build().map_err(http_err)?;
        Ok(Self { client, stream_timeout: STREAM_TIMEOUT })
    }

    /// `POST {url}` with a JSON body, returning the status and body whatever the status: for a
    /// caller that reads an error response's body (a gateway's refusal says why).
    pub fn post_json_any_status(&self, url: &str, body: &str) -> Result<(u16, String), AdapterError> {
        let resp = self
            .client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.to_string())
            .timeout(POST_JSON_TIMEOUT)
            .send()
            .map_err(http_err)?;
        let status = resp.status().as_u16();
        Ok((status, resp.text().map_err(http_err)?))
    }

    /// Streaming `POST {url}` (an SSE response read line by line, like `post_stream`), except a
    /// non-2xx reply is returned as `Err((status, body))` with its body instead of a bare status
    /// error, so the caller can show why the server refused.
    #[allow(clippy::type_complexity)]
    pub fn post_stream_any_status(
        &self,
        url: &str,
        body: &str,
    ) -> Result<Result<Box<dyn Iterator<Item = Result<String, AdapterError>> + Send>, (u16, String)>, AdapterError> {
        let resp = self
            .client
            .post(url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(body.to_string())
            .timeout(self.stream_timeout)
            .send()
            .map_err(http_err)?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            return Ok(Err((status, resp.text().unwrap_or_default())));
        }
        let reader = std::io::BufReader::new(resp);
        Ok(Ok(Box::new(reader.lines().map(|line| line.map_err(|e| AdapterError::Http(e.to_string()))))))
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

    fn get_status(&self, url: &str) -> Result<u16, AdapterError> {
        Ok(self.client.get(url).timeout(GET_TIMEOUT).send().map_err(http_err)?.status().as_u16())
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
