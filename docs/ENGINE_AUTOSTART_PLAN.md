# Engine Auto-Detection & Autostart (`--engine-kind auto`) — implementation plan

**Status:** PLAN (not yet implemented). Design for a zero-config provider mode that
probes for whatever local engine(s) the operator is already running, announces the
union of their models, and — opt-in — can start the one class of engine that ships
a headless CLI (LM Studio).

**Why:** today `provide` takes `--engine-kind` (default `Ollama`) and builds exactly
one adapter for one fixed URL ([`main.rs:429`](../agent/src/main.rs)); there is **no**
detection of which engine is running. If the named engine isn't up, `live_*` errors
and the node announces nothing. This plan makes `openhydra provide` work with no
flags — a direct win for the "install must work for a non-dev" goal.

---

## 1. Scope

**In scope — the 5 `EngineKind` variants** (mapped to detect endpoint + default port):

| Engine | `EngineKind` | Adapter | Default port | Detect fingerprint |
|---|---|---|---|---|
| ollama | `Ollama` | `ollama` | 11434 | `GET /api/tags` (unique) |
| llama.cpp | `LlamaCpp` | `llama_cpp` | 8080 | `GET /props` (unique) |
| LM Studio | `LmStudio` | `openai` | 1234 | `GET /v1/models` |
| vLLM | `Vllm` | `openai` | 8000 | `GET /v1/models` |
| Exo (+ generic) | `Openai` | `openai` | 52415 | `GET /v1/models` |

**Deferred (note, not built):** other OpenAI-compatible servers that also map to the
existing `openai` adapter — Jan (~1337), GPT4All (~4891), LocalAI (~8080, collides
with llama.cpp), oobabooga/text-gen-webui (~5000), KoboldCpp (~5001), LiteLLM proxy
(~4000), cortex/Nitro. These need **only a probe-table entry**, no new adapter. Ports
approximate; verify before adding. Left out of v1 to keep the probe small and the
port-collision surface low.

## 2. Where it lives — NOT a new crate

A module in the **`agent`** crate: `agent/src/detect.rs`. The probe is intrinsically
coupled to the adapters (each engine's URL + fingerprint + constructor live beside its
adapter), so a separate crate would have to depend back on `agent` for `EngineKind` +
the adapters, or duplicate them — no benefit. Reuse the existing `live_ollama` /
`live_openai` / `live_llamacpp` constructors.

## 3. Detection ("which engines are up?")

A per-kind descriptor table:

```rust
struct EngineProbe {
    kind: EngineKind,
    url: &'static str,          // default base URL (host root, no /v1)
    fingerprint: &'static str,  // path to GET, e.g. "/api/tags", "/props", "/v1/models"
    label: &'static str,        // "ollama", "llama.cpp", "lm-studio", "vllm", "openai"
}
```

- Probe order matters for disambiguation: **ollama (`/api/tags`)** and
  **llama.cpp (`/props`)** first — their fingerprints are unique. Then the `/v1/models`
  trio by **port** (LM Studio 1234, vLLM 8000, Exo 52415).
- Each probe = one cheap `GET` with a **short timeout (~500 ms)** (`http.rs` already
  wraps the client). Run them **concurrently** (`spawn_blocking` fan-out; join).
- A probe "hits" if the fingerprint returns 2xx **and** parses as the expected shape
  (guards against an unrelated service squatting the port).
- **Port-collision fallback:** if only `/v1/models` answers on a non-standard port,
  classify as generic `openai` (`label = "openai"`). Never guess vLLM vs LM Studio
  beyond the default-port mapping.

Output: `Vec<DetectedEngine { kind, url, label, models: Vec<DetectedModel> }>` (reuse
`adapter.detect_models()` so the probe already yields the announce set).

## 4. Multi-engine provide (union announce)

Today `run_provider` takes one adapter. Two options; **B** is the target:

- **A (minimal):** `auto` picks the *first* live engine. Simple, but a box running
  ollama + LM Studio only advertises one. Ship as an intermediate step if needed.
- **B (target):** hold a **set** of adapters; announce the **union** of their models.
  The router already handles many providers/models, so nothing downstream changes.
  Refactor: `run_provider(adapters: Vec<Box<dyn EngineAdapter>>, …)`, and at serve
  time route each inbound request to the adapter that owns the requested model id
  (build a `model_id → adapter` map at announce time; rebuild on the periodic
  re-detect). **Do NOT** run N providers announcing the same cluster (that's the
  redundant-advertisement failure mode).

**Interaction with existing re-detection:** `run_inbound` already re-detects every
`reannounce_every` via `announce_models()` → `detect_models()`
([`provider.rs:276`](../agent/src/provider.rs), [`:350`](../agent/src/provider.rs)).
Extend that loop to **re-probe the engine set** on the same tick, so an engine started
*after* the agent (or a newly-loaded model) is picked up within one interval — no
restart. (Also fixes the stale "restart to re-detect" warning at
[`main.rs:489`](../agent/src/main.rs).)

## 5. Fail-fast when nothing is found

If the probe finds zero live engines at `provide` startup, **exit non-zero with a
clear message** rather than silently announcing nothing:

```
openhydra-agent: no local engine detected on the standard ports
  (ollama :11434, LM Studio :1234, vLLM :8000, llama.cpp :8080, Exo :52415).
  Start your engine's server, or pass --engine-kind/--engine explicitly.
```

## 6. Opt-in autostart (`--engine-autostart`) — ✅ IMPLEMENTED (2026-07-02)

**Status:** shipped standalone (ahead of P1/P2 auto-detection) in `agent/src/autostart.rs`,
feature-gated `engine-autostart` (default on), flag off by default. Works with an explicit
`--engine-kind`: probes the engine via the adapter's own `detect_models`, and if it's down
launches **LM Studio** (`lms server start`) or **Ollama** (`ollama serve`) — the two startable
kinds — then polls readiness (30 s cap). Detached via `setsid` + a reaper thread. Live-proven
on the Mac: LM Studio server stopped → autostart relaunched it → `:1234` back to 200; already-up
engines no-op. 6 unit tests. vLLM/llama.cpp/Exo intentionally excluded (need a model/cluster arg).


Separate flag, **off by default** — this is the only part that does process
management, so it's isolated and opt-in. It targets the one gotcha class: **apps that
run but whose OpenAI server is a separate toggle** (LM Studio; later Jan/GPT4All).

- If `--engine-autostart` is set and the probe finds nothing, attempt to start a
  *known* engine whose CLI is detectable and needs no model arg:
  - **LM Studio:** locate `lms` (`~/.lmstudio/bin/lms`), run `lms server start`
    (optionally `lms load <model>` or rely on JIT load), then re-probe `:1234`.
  - **ollama:** `ollama serve` if the binary exists and `:11434` is down (usually a
    daemon already).
- **Not** vLLM / llama.cpp / Exo — those need a model/cluster arg OpenHydra can't
  invent; for them "autostart" would just be "run the operator's launch command,"
  which is the operator's job.
- Implementation: `std::process::Command` (the agent has none today — keep it walled
  in `detect.rs` behind the flag), spawn detached, poll the endpoint with a bounded
  retry. Cross-platform care: `lms`/`ollama` path discovery per-OS.

## 7. CLI / UX

- Add `EngineKind::Auto`; make it the **default** for `--engine-kind` (replaces the
  current `Ollama` default). Explicit `--engine-kind X` / `--engine <url>` still force
  a single engine (unchanged behaviour, escape hatch).
- `--engine-autostart` (bool, default false).
- Log what was found: `detected engines: ollama(:11434, 3 models), lm-studio(:1234, 1)`.

## 8. Testing

- **Unit (pure):** fingerprint classification (given a canned `/api/tags` vs `/props`
  vs `/v1/models` body + port → correct `EngineKind`/label); port-collision →
  `openai`; zero-hit → fail-fast error. Reuse the adapters' existing mock-HTTP tests.
- **Integration:** probe against a stub HTTP server exposing each fingerprint; assert
  union-announce covers both when two are up.
- **Autostart:** gated/mocked (don't shell out in CI) — unit-test the command-builder
  and path-discovery; manual live test for `lms server start`.
- Live matrix (mirrors this session): ollama, llama.cpp, LM Studio, vLLM, Exo each
  detected + served with `--engine-kind auto`, and two-at-once union announce.

## 9. Phasing

1. **P1 — probe + `auto` + fail-fast** — ✅ **DONE (2026-07-02)**, `agent/src/detect.rs`.
2. **P2 — multi-engine union announce** (option B) + re-probe on the re-detect tick — ✅
   **DONE (2026-07-02)** via `MultiAdapter` (see status box below).
3. **P3 — `--engine-autostart`** for LM Studio + ollama — ✅ **DONE** (§6).
4. **P4 — deferred engines** (Jan/GPT4All/LocalAI/…): probe-table entries + fingerprint
   refinement for shared ports — not started.

### P1 + P2 status (✅ IMPLEMENTED 2026-07-02)

`agent/src/detect.rs`: `detect_engines()` probes the 5 standard ports **concurrently** (one
thread each, 1 s connect timeout) using each engine's own adapter, so a hit reuses the exact
fingerprint (`/api/tags`, `/props`, `/v1/models`) and yields the model list. Ports: ollama
11434, llama.cpp 8080, LM Studio 1234, vLLM 8000, **Exo 52415**. `MultiAdapter` *is* an
`EngineAdapter` (so it drops into the existing generic `run_provider` with **zero** provider.rs
changes): `detect_models` re-probes and returns the **union** (de-duped by model id, first
engine wins); `serve_stream` routes each request to the owning engine's adapter via an
`Arc`-shared route table (rebuilt each re-announce tick → engines/models started *after*
startup are absorbed, no restart). `--engine-kind auto` is now the **default**; explicit
`--engine-kind X` still forces one engine. Autostart is wired into auto: nothing detected +
`--engine-autostart` → try Ollama then LM Studio, re-detect; still nothing → fail-fast (§5).
5 unit tests (union/dedup/routing/unknown-model/empty).

**Exo over-announce fix (2026-07-02):** Exo's `/v1/models` returns its whole *downloadable
catalog* (54 models on the Mac), but it can only serve **placed instances whose runners are
all `RunnerReady`**. So Exo got its own adapter (`agent/src/adapters/exo.rs`, `EngineKind::Exo`,
port 52415) that detects via `GET /state` — never the catalog — and fails *closed* (announces
nothing if `/state` is missing or readiness can't be confirmed). It still serves over the
OpenAI route (`serve_chat_completions`); the generic `openai` kind stays for LocalAI/other. 5
unit tests. **Live-proven on the Mac:** bare `auto` found ollama(2)+lm-studio(3)+**exo(1, was
54)** → union **6 models** (was 59); a completion for `llama3.2:1b` routed through
`MultiAdapter` to ollama and returned "PONG.".

## 10. Risks / notes

- **Probe finds running *servers*, not installed *apps*** — LM Studio with the server
  toggled off is invisible (nothing listening). That residual gap is exactly what P3
  autostart closes for the toggle-class apps.
- Keep the probe timeout short and concurrent so `provide` startup isn't delayed by a
  down port.
- Feature-gate the autostart/process-management code so the lean pure-protocol build
  can exclude it if desired.
