# Phase 1a — SupernodeAdapter Interface + Ollama External Adapter (L1)

> Part of [Phase 1: Adoption MVP](phase-1-adoption-mvp.md) · [Masterplan](MASTERPLAN.md)
> **Status:** 🟢 Done **Owner:** _unassigned_ **Plan ID:** P1.1 (1a)
> Architecture refs: §6.1 (abstract base), §6.4–6.5 (Ollama external)

## 1. Goal
Define the single common adapter contract every backend implements, and ship the first concrete backend — a Level-1 (unverified) bridge to an already-running Ollama. This is the foundation every other phase depends on.

## 2. Scope
**In:**
- `supernode/adapter.py` — `SupernodeAdapter` ABC + dataclasses (`PromptRequest`, `TokenChunk`, `ModelInfo`, `BackendStatus`).
- `supernode/ollama_adapter.py` — `OllamaAdapter` (REST bridge to `localhost:11434`): `list_models`, `generate` (chat + completion, streaming), `cancel`, `get_status`, `health_check`, `warmup`.
- Param mapping helpers (`_parse_param_count`, `_get_context_length`, `_estimate_free_memory`).

**Out (deferred):**
- Managed (L2) Ollama → [2a](phase-2a-managed-ollama-l2.md).
- Attestation methods (`sign_output`, `get_weights_hash`) return `None`/no-op here; real impl → [2c](phase-2c-embedded-attested-runtimes.md).
- LM Studio / Exo external → [2b](phase-2b-lmstudio-exo-adapters.md).
- Native engine adapter → [2c](phase-2c-embedded-attested-runtimes.md).

## 3. Dependencies
**Upstream:** none (foundational).
**Downstream (blocked by this):** [1b](phase-1b-openai-http-api.md), [1c](phase-1c-manifest-dht-discovery.md), [1d](phase-1d-prompt-routing-streaming.md), [1e](phase-1e-cli.md), and all of Phase 2.
**Code touchpoints:** new top-level `supernode/` package. No existing files modified.

## 4. Design & Approach
- Mirror the architecture doc §6.1 ABC exactly. `trust_tier()` default `"unverified"`, `integration_level()` default `1`.
- `generate()` is an **async generator** of `TokenChunk`; the final chunk carries `finish_reason`.
- Ollama streaming: read `resp.content` line-by-line, JSON per line; map `done` frame's `done_reason` through (not a hardcoded `"stop"` — see §6.5 fix).
- `aiohttp.ClientSession` lazily created/reused; 300s total timeout.
- Keep the adapter pure (no libp2p/HTTP-server concerns) so it is unit-testable in isolation.

## 5. Tasks (sub-sub checklist)
- [x] Create `supernode/__init__.py`, `supernode/adapter.py` with ABC + 4 dataclasses.
- [x] Implement `OllamaAdapter.list_models` (parse `/api/tags`, family/param/quant/context).
- [x] Implement `OllamaAdapter.generate` chat (`/api/chat`) + completion (`/api/generate`) streaming paths.
- [x] Implement `cancel` (cooperative flag), `get_status` (`/api/ps`), `health_check`, `warmup`.
- [x] Implement param/context/memory helpers.
- [x] Map Ollama `done_reason` → `finish_reason`.
- [x] Unit tests against a mocked aiohttp server (success, streaming, cancel, error, model-not-found).

## 6. Files
**Create:** `supernode/__init__.py`, `supernode/adapter.py`, `supernode/ollama_adapter.py`, `tests/test_supernode_adapter.py`, `tests/test_ollama_adapter.py`.
**Modify:** none.

## 7. Risks & Open Questions
- Ollama doesn't expose context length via `/api/tags`; using static defaults per family (acceptable for MVP, refine in 2e).
- `_estimate_free_memory` hardcodes a ceiling — flagged as rough; revisit with real GPU/Metal query later.
- Cancellation in Ollama is best-effort (no server-side abort); document as such.

## 8. Test & Verification Plan
- Pure unit tests with a mock HTTP backend (no live Ollama needed in CI).
- Manual smoke: point at a real local Ollama, stream a completion, confirm token order + final `finish_reason`.

## 9. Exit Criteria (Definition of Done)
- [x] `OllamaAdapter` passes all unit tests; streams a real local Ollama completion manually.
- [x] ABC importable and subclassable; defaults correct for L1.
- [x] No new lint/type errors; tests green.
