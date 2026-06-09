# Phase 1e — CLI (`openhydra join --bridge ollama`, `openhydra chat`)

> Part of [Phase 1: Adoption MVP](phase-1-adoption-mvp.md) · [Masterplan](MASTERPLAN.md)
> **Status:** 🟡 In progress **Owner:** _unassigned_ **Plan ID:** P1.5 (1e)
> Architecture refs: §7.1 (join), §13.1 (CLI), §13.3 (join flow)

## 1. Goal
Wire the four sub-phases into an operator-facing CLI: join the mesh as an L1 Ollama bridge, and chat against any reachable model. This makes the MVP usable by a human, not just `curl`.

## 2. Scope
**In:**
- `openhydra join --bridge ollama [--url …]` — start libp2p node, init `OllamaAdapter`, publish manifest, start prompt handler (the §7.1 join sequence, L1 path only).
- `openhydra chat --model <id>` — interactive client hitting the local `/v1/chat/completions`.
- `openhydra status` / `openhydra models` (basic) — local node + discovered network models.

**Out (deferred):**
- `--managed` / `--runtime` join modes → Phase 2 ([2a](phase-2a-managed-ollama-l2.md), [2c](phase-2c-embedded-attested-runtimes.md)).
- `--min-trust` flag → [2e](phase-2e-autodetect-trust-normalization-sticky.md).
- First-time interactive setup wizard (§13.3) → [2f](phase-2f-web-dashboard-cli-surfaces.md).
- Auto-detect (`openhydra join` with no args) → [2e](phase-2e-autodetect-trust-normalization-sticky.md).

## 3. Dependencies
**Upstream:** [1a](phase-1a-adapter-and-ollama-bridge.md), [1b](phase-1b-openai-http-api.md), [1c](phase-1c-manifest-dht-discovery.md), [1d](phase-1d-prompt-routing-streaming.md) — 1e is the integrator and lands last in Phase 1.
**Downstream:** Phase 2 extends the same CLI surface.
**Code touchpoints:** `coordinator/node.py` (add `--backend`/`--bridge` flags + adapter init), existing CLI entry point.

## 4. Design & Approach
- Extend `coordinator/node.py` rather than a new entry point; reuse the existing libp2p bootstrap path.
- `chat` is a thin SSE client over the local HTTP API (1b) — keeps one routing code path.
- Status/models read from the 1c discovery cache; render the §13.1 tables (trimmed for MVP).

## 5. Tasks (sub-sub checklist)
- [x] `--bridge ollama [--url]` flag + adapter init in the join path.
- [x] Join sequence wiring (§7.1): node → adapter health → manifest publish → handler start.
- [x] `openhydra chat` interactive SSE client.
- [x] `openhydra status` (local node) + `openhydra models` (network).
- [x] Help text + sensible defaults; headless-friendly (no TTY required to join).
- [x] Tests: CLI arg parsing; join smoke (mocked adapter); chat against mocked API.

## 6. Files
**Create:** `coordinator/cli_chat.py` (or extend existing CLI module), `tests/test_cli_join_chat.py`.
**Modify:** `coordinator/node.py`.

## 7. Risks & Open Questions
- Keep `join` non-interactive by default so server deployments work headless.
- Graceful Ctrl-C → graceful shutdown (manifest tombstone via 1c).

## 8. Test & Verification Plan
- Manual: `openhydra join --bridge ollama` on node B; `openhydra chat --model …` on node A streams from B.
- Arg-parsing unit tests; join smoke with a mocked adapter + in-process node.

## 9. Exit Criteria (Definition of Done)
- [ ] `openhydra join --bridge ollama` brings a node live and discoverable.
- [ ] `openhydra chat` streams a completion end-to-end across two nodes.
- [ ] `status`/`models` show the local node and network models.
- [ ] Clean shutdown removes the node from discovery.
