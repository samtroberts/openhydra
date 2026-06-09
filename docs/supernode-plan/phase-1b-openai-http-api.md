# Phase 1b — OpenAI-Compatible HTTP API

> Part of [Phase 1: Adoption MVP](phase-1-adoption-mvp.md) · [Masterplan](MASTERPLAN.md)
> **Status:** 🟢 Done **Owner:** _unassigned_ **Plan ID:** P1.2 (1b)
> Architecture refs: §8.3 (HTTP API), §5.2 (PromptRequest), §5.3 (PromptChunk)

## 1. Goal
Expose the **product surface**: an OpenAI-compatible HTTP server on the coordinator that accepts `/v1/chat/completions` and `/v1/completions`, resolves a model, routes (locally in MVP, remotely via 1d), and streams SSE tokens back. This is the headline deliverable of Phase 1.

## 2. Scope
**In:**
- `POST /v1/chat/completions` (messages) + `POST /v1/completions` (prompt), streaming **and** non-streaming.
- `GET /v1/models` (aggregate of reachable supernodes' models).
- `GET /v1/supernodes` (OpenHydra-specific status list).
- SSE framing identical to OpenAI (`data: {...}\n\n`, terminal `data: [DONE]`).
- Translation: HTTP request → internal `PromptRequest`; internal `TokenChunk`/`PromptChunk` → SSE deltas.

**Out (deferred):**
- `min_trust_level` parameter surface → [2e](phase-2e-autodetect-trust-normalization-sticky.md) (MVP routes L1 only).
- Web dashboard → [2f](phase-2f-web-dashboard-cli-surfaces.md).
- Sticky sessions → [2e](phase-2e-autodetect-trust-normalization-sticky.md).

## 3. Dependencies
**Upstream:** [1a](phase-1a-adapter-and-ollama-bridge.md) (adapter to call locally). Integrates with [1d](phase-1d-prompt-routing-streaming.md) for remote dispatch (can stub a local-only path first).
**Downstream:** [1e](phase-1e-cli.md) (`openhydra chat` calls this API); Phase 2 UI + min_trust extend it.
**Code touchpoints:** existing coordinator HTTP server (extend, don't replace).

## 4. Design & Approach
- **Async-first.** Even though MVP can serve a local adapter directly, write the handler to `await` a router interface so wiring 1d later is a swap, not a rewrite.
- Handler flow (§8.3): receive `model` → normalize (basic, full registry in 2e) → discover candidates (1c; in single-node MVP this is "self") → select (1d) → dispatch → relay stream as SSE.
- Map `PromptChunk(status=…)` to keep the SSE connection warm while a model loads (emit OpenAI-compatible empty deltas or a custom heartbeat comment).
- Non-streaming = accumulate the async generator then return one JSON body.

## 5. Tasks (sub-sub checklist)
- [x] Add routes to the coordinator's HTTP server (framework already in use).
- [x] Request models: parse OpenAI chat + completion bodies into `PromptRequest`.
- [x] Streaming SSE writer with correct framing + `[DONE]`.
- [x] Non-streaming aggregation path.
- [x] `GET /v1/models` and `GET /v1/supernodes` (read from 1c discovery cache).
- [x] Error mapping → OpenAI-style error JSON (model not found, no provider, timeout).
- [x] Router interface seam (local-only impl now, remote via 1d later).
- [x] Tests: streaming, non-streaming, error bodies, SSE framing byte-for-byte.

## 6. Files
**Create:** `coordinator/openai_api.py` (or extend existing HTTP module), `tests/test_openai_api.py`.
**Modify:** coordinator HTTP server entry to mount the new routes.

## 7. Risks & Open Questions
- SSE keep-alive vs. client timeouts during model cold-load — decide heartbeat strategy.
- Token usage accounting (prompt/completion counts) for `usage` field — Ollama returns eval counts; map them.
- Exact OpenAI error schema fidelity (some clients are strict).

## 8. Test & Verification Plan
- Golden-file test of SSE byte stream for a fixed mocked token sequence.
- Contract test with the official `openai` Python SDK pointed at `localhost`.
- Manual: `curl -N` streaming + a real chat UI (e.g. any OpenAI-compatible frontend).

## 9. Exit Criteria (Definition of Done)
- [ ] `openai` SDK and `curl -N` both stream a completion through the endpoint.
- [ ] Non-streaming returns a valid OpenAI JSON body with `usage`.
- [ ] `/v1/models` lists at least the local Ollama models.
- [ ] Router seam in place so 1d can be wired without changing handlers.
