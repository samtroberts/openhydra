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

/// Production HTTP transport for engine adapters.
pub struct ReqwestClient {
    client: Client,
}

impl ReqwestClient {
    /// Build a client with a short connect timeout (the engine is local) and **no**
    /// read timeout — a completion stream may legitimately run for minutes.
    pub fn new() -> Result<Self, AdapterError> {
        Self::with_connect_timeout(Duration::from_secs(5))
    }

    /// Like [`new`](Self::new) but with an explicit connect timeout — used by engine
    /// auto-detection, which probes several ports and wants to fail fast on a dead one
    /// (still no read timeout, so an adapter built this way serves long streams fine).
    pub fn with_connect_timeout(connect: Duration) -> Result<Self, AdapterError> {
        let client = Client::builder().connect_timeout(connect).build().map_err(http_err)?;
        Ok(Self { client })
    }
}

impl HttpClient for ReqwestClient {
    fn get(&self, url: &str) -> Result<String, AdapterError> {
        self.client
            .get(url)
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
        let mut req = self.client.get(url);
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
            .body(body.to_string());
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
            .body(body.to_string());
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
