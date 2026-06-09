# Phase 1d — Prompt Routing Protocol + Token Streaming

> Part of [Phase 1: Adoption MVP](phase-1-adoption-mvp.md) · [Masterplan](MASTERPLAN.md)
> **Status:** 🟡 In progress **Owner:** _unassigned_ **Plan ID:** P1.4 (1d)
> Architecture refs: §5 (wire protocol), §4.1 (routing flow), §4.3.1 (selection), §4.4 (failover), §8.1–8.2 (dispatch/integration)

## 1. Goal
Carry prompts and stream tokens **peer-to-peer over libp2p** using new method prefixes on the existing request-response/stream channels, select a supernode without `argmax` herd behavior, and fail over cleanly. This closes the loop: 1b's HTTP request reaches 1c's discovered supernode and tokens come back.

## 2. Scope
**In:**
- Method prefixes `0x10 PROMPT_REQUEST`, `0x11 PROMPT_CANCEL`, `0x12 MANIFEST_REQUEST`, `0x13 MANIFEST_RESPONSE`, `0x14 LOAD_PROBE` (basic) in the Rust dispatcher.
- `PromptRequest` / `PromptChunk` CBOR frames (§5.2–5.3); token streaming over `libp2p_stream::Behaviour` (already in `behaviour.rs`).
- Router: candidate scoring (basic — latency + a direct LoadProbe), **randomized near-equal selection (§4.3.1)**.
- **Fail-fast** mid-stream failover (§4.4 option 1): resend before first token; after first token, terminate with `finish_reason="error"` (no silent restart).
- Cancellation (`0x11`) → cooperative stop on the supernode.

**Out (deferred):**
- Full LoadCache + gossipsub load broadcasts → [3a](phase-3a-realtime-load.md).
- Full scoring pipeline (measured_tps EMA, queue backpressure, trust weighting) → [3b](phase-3b-full-scoring.md).
- Greedy re-prefill failover (§4.4 option 2) → [3d](phase-3d-failover.md).
- Multi-hop re-dispatch (`hops`) — single-hop only in MVP.

## 3. Dependencies
**Upstream:** [1a](phase-1a-adapter-and-ollama-bridge.md) (adapter to run inference), [1c](phase-1c-manifest-dht-discovery.md) (candidate list), [1b](phase-1b-openai-http-api.md) (consumer of the stream).
**Downstream:** [1e](phase-1e-cli.md); all of Phase 3 builds on this.
**Code touchpoints:** `network/src/dispatcher.rs`, `network/src/event_loop.rs`, `network/src/node.rs` (expose `send_prompt_request()`/`poll_prompt_stream()` to Python), `peer/server.py` (`_proxy_handler_loop`).

## 4. Design & Approach
- **Thread↔asyncio bridge is the critical risk.** `_proxy_handler_loop` is thread-based (`threading.Event`); the adapter layer is async. Run a dedicated asyncio loop in the prompt handler and drive adapters via `asyncio.run_coroutine_threadsafe`, or a per-handler event loop — decide and document. Do NOT block the proxy thread on inference.
- Dispatcher: extend the existing `0x01–0x06` match with `0x10–0x14` (mirror the established pattern).
- Streaming uses persistent bidirectional `libp2p_stream`; backpressure-aware writes.
- Selection (§4.3.1): score candidates, take those within 10% of best, **weighted-random** among them (or power-of-two-choices). Never deterministic argmax.
- Failover: pre-sort candidates so retry needs no new DHT query; on pre-first-token failure resend to next; decrement failed node's reputation (reputation store stubbed until 3c).

## 5. Tasks (sub-sub checklist)
- [ ] Rust: add `0x10–0x14` to `dispatcher.rs`; new `SwarmCommand` variants in `event_loop.rs`.
- [ ] Rust: `send_prompt_request()` + `poll_prompt_stream()` PyO3 bindings in `node.rs`.
- [x] CBOR `PromptRequest`/`PromptChunk` encode/decode (Python — `supernode/prompt_protocol.py`; Rust deferred to maturin rebuild).
- [ ] Python prompt handler in `peer/server.py` with async bridge → adapter `generate()`.
- [ ] Cancellation path (`0x11`).
- [x] Router: basic score + `select_supernode()` randomized near-equal (§4.3.1) — `supernode/selector.py`.
- [x] Fail-fast failover + reputation-decrement hook (stub store) — wired into `SupernodeRouter._generate_with_failover()`.
- [x] Tests: selection spread (statistical), pre-first-token failover, mid-stream error, CBOR roundtrip — 44 tests across `test_prompt_protocol.py`, `test_selector.py`, `test_supernode_router.py`.

## 6. Files
**Create:** `supernode/router.py`, `supernode/prompt_protocol.py`, `tests/test_prompt_routing.py`, Rust tests in `network/`.
**Modify:** `network/src/dispatcher.rs`, `network/src/event_loop.rs`, `network/src/node.rs`, `peer/server.py`.

## 7. Risks & Open Questions
- **Thread↔asyncio bridge** (above) — prototype this first; everything else depends on it.
- libp2p-stream backpressure / partial-frame handling under slow consumers.
- Maturin rebuild + redeploy of the Rust wheel to both GPUs for testing.

## 8. Test & Verification Plan
- Two-node loop: HTTP on A → routed to Ollama on B → streamed back.
- Selection-spread test: 100 routes across 3 equal nodes land ~evenly (herd check).
- Failover test: kill B before first token → A retries C; kill mid-stream → clean error, no dup tokens.

## 9. Exit Criteria (Definition of Done)
- [ ] End-to-end two-node streamed completion via libp2p.
- [ ] Selection is randomized among near-equal candidates (verified statistically).
- [ ] Pre-first-token failover works; mid-stream failure ends cleanly (no silent restart).
- [ ] Cancellation stops generation on the supernode.
- [ ] Rust wheel rebuilt; existing 22 Rust tests still pass.
