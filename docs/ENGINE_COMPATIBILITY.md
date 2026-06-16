# Engine compatibility — running your engine's models on OpenHydra

OpenHydra never runs a model. A **provider** proxies inference to whatever engine
it already runs locally, over that engine's HTTP API, via an `EngineAdapter`
(`agent/src/adapters/`). Pick the engine with `--engine-kind`:

| Engine | `--engine-kind` | Default port | Canonical ids? |
|---|---|---|---|
| Ollama | `ollama` | 11434 | yes (rich metadata) |
| vLLM | `vllm` | 8000 | no (advertised by engine id) |
| LM Studio | `lm-studio` | 1234 | no |
| llama.cpp (`llama-server`) | `llama-cpp` | 8080 | yes (via `/props`) |
| Exo / LocalAI / any OpenAI-shaped server | `openai` | 8000 | no |

```sh
openhydra-agent --bootstrap <relay-multiaddr> provide --engine-kind vllm --engine http://127.0.0.1:8000
```

## Unsloth (fine-tuned models)

Unsloth is a **fine-tuning** library, not an inference server — so there is no
"Unsloth adapter". Serve an Unsloth model through the engine Unsloth's own docs
target, and OpenHydra's existing adapters cover it:

```sh
# vLLM (Unsloth's recommended serving path) → the openai adapter
vllm serve unsloth/Llama-3.2-1B-Instruct --port 8000
openhydra-agent --bootstrap <relay> provide --engine-kind vllm --engine http://127.0.0.1:8000

# or an exported GGUF via llama-server → the llama-cpp adapter
llama-server -m my-finetune-Q4_K_M.gguf --port 8080
openhydra-agent --bootstrap <relay> provide --engine-kind llama-cpp
```

A fine-tuned model is unique, so it is correctly advertised **uncanonicalised**
by its engine id (e.g. `unsloth/Llama-3.2-1B-Instruct`) rather than collapsed
onto the base model's canonical id — consumers request it by that id.

**Validated live (2026-06-16):** `vllm serve unsloth/Llama-3.2-1B-Instruct` on a
T4, provider via `--engine-kind vllm`, consumer gateway → discover (gossip PEX) →
serve → 27-token completion streamed back + co-signed receipt ledgered. (On a
Turing GPU set `VLLM_ATTENTION_BACKEND=TRITON_ATTN` — FlashInfer hangs at warmup.)
