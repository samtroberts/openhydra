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
    // Without the engine's URL: an error's text reaches the consumer that asked, and where this
    // provider's engine listens is its own business (review 11 Nov: replies named
    // `http://127.0.0.1:8080/...`). The operator's own log keeps it.
    if let Some(url) = e.url() {
        tracing::debug!(%url, error = %e, "engine request failed");
    }
    let e = e.without_url();
    // reqwest's text for a timeout doesn't say so: callers tell timeouts by "timed out" (round 10).
    if e.is_timeout() {
        AdapterError::Http(format!("timed out: {e}"))
    } else {
        AdapterError::Http(e.to_string())
    }
}

/// The reply itself when it succeeded; else an error that says what the engine said. Replaces
/// `error_for_status`, which dropped the engine's explanation (live 11 Nov: vLLM's "auto tool
/// choice requires --enable-auto-tool-choice" became a bare "400 Bad Request").
fn ok_or_status(resp: reqwest::blocking::Response) -> Result<reqwest::blocking::Response, AdapterError> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body: String = resp.text().unwrap_or_default().chars().take(8192).collect();
    Err(status_error(status.as_u16(), status.canonical_reason().unwrap_or(""), &body))
}

/// `HTTP status client error (400 Bad Request): <what the engine said>` (the prefix as reqwest
/// words it, which callers match on).
pub(crate) fn status_error(code: u16, reason: &str, body: &str) -> AdapterError {
    let kind = if (400..500).contains(&code) { "client" } else { "server" };
    let said = engine_message(body);
    let head = format!("HTTP status {kind} error ({code} {reason})");
    AdapterError::Http(if said.is_empty() { head } else { format!("{head}: {said}") })
}

/// An engine's error body as one short line: its JSON `error` (text, or an object's `message`),
/// `message` or `detail`, else the text itself; at most 300 characters, URLs taken out.
pub(crate) fn engine_message(body: &str) -> String {
    let from_json = serde_json::from_str::<serde_json::Value>(body).ok().and_then(|v| {
        let field = |x: &serde_json::Value| x.as_str().map(str::to_string);
        v.get("error")
            .and_then(|e| field(e).or_else(|| e.get("message").and_then(field)))
            .or_else(|| v.get("message").and_then(field))
            .or_else(|| v.get("detail").and_then(|d| field(d).or_else(|| Some(d.to_string()))))
    });
    let text = from_json.unwrap_or_else(|| body.to_string());
    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| !(w.contains("http://") || w.contains("https://")))
        .collect();
    let mut line = words.join(" ");
    if line.chars().count() > 300 {
        line = line.chars().take(299).collect::<String>() + "…";
    }
    line
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
            .map_err(http_err)
            .and_then(ok_or_status)?
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
            .map_err(http_err)
            .and_then(ok_or_status)?
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
            .map_err(http_err)
            .and_then(ok_or_status)?
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
            .map_err(http_err)
            .and_then(ok_or_status)?
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
            .map_err(http_err)
            .and_then(ok_or_status)?
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
        let resp = req.send().map_err(http_err).and_then(ok_or_status)?;
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

#[cfg(test)]
mod engine_error_tests {
    use super::*;

    /// Review 11 Nov: the engine's explanation is kept, its address never.
    #[test]
    fn an_engine_refusal_says_why_and_never_where() {
        let vllm = r#"{"object":"error","message":"\"auto\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set","type":"BadRequestError","code":400}"#;
        let AdapterError::Http(m) = status_error(400, "Bad Request", vllm) else { panic!() };
        assert!(m.starts_with("HTTP status client error (400 Bad Request): "), "{m}");
        assert!(m.contains("--enable-auto-tool-choice"), "{m}");
        let AdapterError::Http(m) = status_error(503, "Service Unavailable", r#"{"error":{"message":"loading model, see http://127.0.0.1:8080/health"}}"#) else { panic!() };
        assert!(m.starts_with("HTTP status server error (503") && !m.contains("127.0.0.1") && m.contains("loading model"), "{m}");
        assert_eq!(engine_message(r#"{"detail":"Not Found"}"#), "Not Found");
        assert_eq!(engine_message("plain   text\n reply"), "plain text reply");
        assert_eq!(engine_message(&"x ".repeat(400)).chars().count(), 300);
        let AdapterError::Http(m) = status_error(400, "Bad Request", "") else { panic!() };
        assert_eq!(m, "HTTP status client error (400 Bad Request)");
    }
}
