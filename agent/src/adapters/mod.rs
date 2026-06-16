// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Engine adapters (BYO-engine).
//!
//! Each submodule wraps one local inference engine behind
//! [`crate::adapter::EngineAdapter`]: detect the models it serves (→ canonical ids for
//! swarm advertisement) and proxy an inbound streaming completion to it. Pure parsing is
//! kept separate from I/O — HTTP is injected via [`crate::adapter::HttpClient`] so every
//! adapter is unit-tested against fixtures with no live engine.
//!
//! * [`ollama`] — Ollama's native `/api/*` API (rich metadata: family/params/quant +
//!   chat template → full canonical ids).
//! * [`openai`] — any OpenAI-compatible server (vLLM, LM Studio, Exo, LocalAI). One
//!   adapter for the whole family; thinner metadata, so models are advertised by their
//!   engine id (uncanonicalised).
//! * [`llama_cpp`] — `llama-server`. Serves over the OpenAI route (reuses [`openai`]'s
//!   stream) but detects bespoke-ly via `/props`, whose chat template + GGUF path yield a
//!   full canonical id.

pub mod llama_cpp;
pub mod ollama;
pub mod openai;
