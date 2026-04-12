# OpenHydra Universal AI OS — Architecture Plan

**Date**: 2026-03-31
**Vision**: Transform OpenHydra from a P2P inference engine into a universal AI operating system where any node can run agents, serve models, and participate in a decentralised AI economy.

---

## The Big Picture

```
Today:                              Target:
┌─────────────┐                     ┌──────────────────────────────────────┐
│  User       │                     │  User / Agent / External Service     │
│  ↓ prompt   │                     │  ↓ task (inference, code, research)  │
│  Coordinator│                     │  Agent Runtime (local or remote)     │
│  ↓ pipeline │                     │  ├── Tool Registry                  │
│  Peer₁→Peer₂│                     │  ├── Memory Store                   │
│  ↓ tokens   │                     │  ├── Multi-turn Planner             │
│  Response   │                     │  └── OpenHydra Network ←────────────┤
└─────────────┘                     │      ├── Inference pipelines        │
                                    │      ├── Agent-to-agent messaging   │
                                    │      ├── Model marketplace          │
                                    │      └── Federated training jobs    │
                                    └──────────────────────────────────────┘
```

OpenHydra today: stateless request→response inference.
OpenHydra target: a **runtime for AI agents** that use the network as their brain, hands, and memory.

---

## Phase 1: Agent Runtime (4–6 weeks)

### 1.1 Agent Loop Core (`agent/runtime.py`)

A minimal agent loop that wraps the existing `infer()` pipeline:

```python
class AgentRuntime:
    def __init__(self, agent_id, engine, tool_registry, memory):
        ...

    async def run(self, task: str) -> AgentResult:
        """ReAct-style loop: think → act → observe → repeat."""
        messages = [{"role": "user", "content": task}]
        for step in range(self.max_steps):
            response = await self.engine.infer_chat(messages, tools=self.tools)
            if response.tool_calls:
                results = await self.tool_registry.execute(response.tool_calls)
                messages.extend(results)
            elif response.stop_reason == "end_turn":
                return AgentResult(output=response.text, steps=step+1)
        return AgentResult(output=response.text, truncated=True)
```

**Key design decisions:**
- **Not a framework** — a thin loop over existing `infer_chat()`. No new inference path.
- **Tool calls via OpenAI function-calling format** — already parsed in `api_server.py`'s chat endpoint.
- **Async-first** — agents yield control between inference calls, enabling concurrency.
- **Stateless between runs** — memory is explicit, not hidden in the runtime.

### 1.2 Tool Registry (`agent/tools.py`)

Built-in tools available to any agent:

| Tool | Description | Implementation |
|------|-------------|----------------|
| `shell` | Execute shell commands | `subprocess.run()` with sandbox (seccomp/landlock on Linux, sandbox-exec on macOS) |
| `read_file` / `write_file` | Filesystem access | Scoped to agent workspace directory |
| `web_search` | Search the web | Proxy to SearXNG or Brave Search API |
| `web_fetch` | Fetch a URL | `requests.get()` with size limits |
| `python` | Execute Python code | Isolated subprocess with resource limits |
| `openhydra_infer` | Call another model on the network | Routes through coordinator — agent can use GPT-4-class models for planning and small models for execution |
| `agent_call` | Invoke another agent | See Phase 2 (agent-to-agent) |

**Custom tools**: Agents register tools as JSON schema + callable. Stored in agent workspace.

### 1.3 Agent Memory (`agent/memory.py`)

Three tiers, mirroring how humans remember:

| Tier | What | Storage | TTL |
|------|------|---------|-----|
| **Working** | Current task context, tool outputs | In-memory list | Session |
| **Episodic** | Completed task summaries, key decisions | SQLite per agent | 30 days default |
| **Semantic** | User preferences, project facts, learned patterns | SQLite per agent + vector embeddings | Permanent |

Vector embeddings computed locally via the same MLX/PyTorch runtime (small embedding model like `nomic-embed-text`). No external vector DB dependency.

### 1.4 CLI & API Surface

```bash
# Run an agent locally
openhydra-agent --task "Write a Python script that fetches HN front page"

# Run an agent that uses the network for inference
openhydra-agent --task "Research quantum computing advances in 2026" \
    --model meta-llama/Llama-3.1-70B \
    --tools shell,web_search,web_fetch

# Expose agent as an API
openhydra-agent serve --port 8080  # OpenAI-compatible /v1/chat/completions with tool_use
```

New API endpoints on coordinator:
- `POST /v1/agents/run` — submit a task, get back agent result
- `GET /v1/agents/{id}/status` — poll agent progress
- `WS /v1/agents/{id}/stream` — WebSocket for live agent output

### 1.5 Desktop Integration

Add "Agent" tab to the Tauri desktop app:
- Task input box (like a chat, but with tool output panels)
- Agent workspace file browser (shows files the agent created/modified)
- Tool permission toggles (allow/deny shell, web, filesystem per agent)
- Live step-by-step trace (think → tool_call → observation → think...)

---

## Phase 2: Agent-to-Agent Communication (2–3 weeks)

### 2.1 Agent Mesh Protocol

Agents communicate over the existing DHT + gRPC infrastructure:

```
Agent A (local) ──gRPC──→ Coordinator ──DHT lookup──→ Agent B (remote peer)
                                                        ↓
                                                    Agent B executes task
                                                        ↓
                                              Result streams back via gRPC
```

**New proto RPC:**
```proto
service AgentService {
    rpc Invoke(AgentRequest) returns (stream AgentResponse);
    rpc Discover(AgentDiscoveryRequest) returns (AgentDiscoveryResponse);
}
```

**Agent discovery via DHT:**
- Peers announce agent capabilities: `agent_capabilities: ["code", "research", "data_analysis"]`
- Coordinator routes `agent_call` tool invocations to capable peers
- Reputation system extends to agent quality (task completion rate, user ratings)

### 2.2 Agent Marketplace

Agents are published as **agent manifests** (JSON):

```json
{
    "name": "code-reviewer",
    "version": "1.0.0",
    "description": "Reviews pull requests and suggests improvements",
    "author": "sam",
    "tools_required": ["read_file", "shell"],
    "model_preference": "meta-llama/Llama-3.1-70B",
    "system_prompt_hash": "sha256:abc123..."
}
```

- Stored in DHT under `agent_{name}_{version}` keys
- Peers can host agents and earn HYDRA tokens per invocation
- System prompt is hashed (not stored) — the hosting peer holds the actual prompt
- Verification: mystery-shopper agents test quality periodically

---

## Phase 3: HuggingFace Auto-Population (2 weeks)

### 3.1 Model Registry Service (`coordinator/model_registry.py`)

Replace the static `bench_peers.json` model catalog with a dynamic registry:

```python
class ModelRegistry:
    async def discover_models(self) -> list[ModelEntry]:
        """Scan HF Hub for compatible models, ranked by popularity + compatibility."""
        # 1. Query HF API: trending models, filter by architecture
        # 2. Check compatibility: tokenizer type, model size vs swarm capacity
        # 3. Cross-reference with what peers already have cached (via DHT)
        # 4. Return ranked list with download size, quantization options, min VRAM
```

**Supported architectures** (auto-detected from HF `config.json`):

| Architecture | PyTorch | MLX | Status |
|-------------|---------|-----|--------|
| LlamaForCausalLM | ✅ | ✅ | Working |
| Qwen2ForCausalLM | ✅ | ✅ | Working |
| MistralForCausalLM | ✅ | ✅ | Working |
| PhiForCausalLM | ✅ | ✅ | Needs testing |
| GemmaForCausalLM | ✅ | ✅ | Needs testing |
| GPTNeoXForCausalLM | ✅ | ❌ | PyTorch only |
| MixtralForCausalLM | ✅ | ✅ | MoE routing exists |
| StableLMForCausalLM | ✅ | ✅ | Needs testing |

### 3.2 Auto-Download & Cache Warming

When a user requests a model not yet in the swarm:

1. Coordinator checks HF compatibility (architecture, size)
2. Selects peers with enough VRAM/disk (from DHT announcements)
3. Sends download directive (like rebalance directive but for model acquisition)
4. Peers download via P2P cache (existing `p2p_model_cache.py`) or HF CDN fallback
5. Once enough peers have the model, it appears in `/v1/models`

**Smart pre-caching**: Top 20 trending HF models are pre-cached across the swarm. Peers with idle bandwidth/disk volunteer as seeders.

### 3.3 GGUF Support

Add GGUF format alongside HF safetensors:

- **Why**: Massive ecosystem (llama.cpp, Ollama, LM Studio). Thousands of pre-quantized models.
- **How**: `llama-cpp-python` as optional dependency. New runtime: `GGUFRuntime` in `peer/gguf_runtime.py`.
- **Layer sharding**: GGUF models support layer extraction natively via `llama_model_loader`.

```toml
# pyproject.toml
gguf = ["llama-cpp-python>=0.3"]
```

---

## Phase 4: Smartphone & Low-Compute Support (3–4 weeks)

### 4.1 Compute Tiers

| Tier | Device | Role | Earning Rate |
|------|--------|------|-------------|
| **Tier 0** | Server (A100, H100) | Full model host, pipeline anchor | 10× base |
| **Tier 1** | Desktop (M1+ Mac, RTX 3060+) | Layer shard host, agent runtime | 3× base |
| **Tier 2** | Laptop (8GB+ RAM) | Small model host, embedding, verification | 1× base |
| **Tier 3** | Phone (iOS/Android) | KV cache relay, embedding, verification only | 0.3× base |

### 4.2 Mobile Peer (`peer/mobile/`)

Phones can't run 7B models, but they CAN:

1. **Run tiny models** (< 1B params): Qwen3.5-0.8B via Core ML / NNAPI
2. **Serve as KV cache relays**: Store KV cache entries for nearby desktop peers (reduces TTFT)
3. **Participate in verification**: Mystery shopper for small-model tiers
4. **Contribute bandwidth**: Relay encrypted activations between peers (onion routing node)
5. **Run embedding models**: `nomic-embed-text` (137M params) runs fine on phones

**Implementation**:
- iOS: Swift + Core ML + gRPC-Swift — ship as TestFlight beta
- Android: Kotlin + NNAPI + gRPC-Kotlin
- Shared: Proto definitions are already cross-platform (peer.proto)
- Thin client: Phone connects to coordinator API, no local coordinator needed

### 4.3 Pricing & Incentive Structure

```
Inference pricing (HYDRA tokens per 1K tokens):
┌─────────────────────────┬──────────┬──────────┐
│ Model Size              │ Standard │ Priority │
├─────────────────────────┼──────────┼──────────┤
│ < 1B (Qwen 0.8B)       │ 0.1      │ 0.5      │
│ 1-8B (Llama 3.2 1B-8B) │ 1.0      │ 5.0      │
│ 8-70B (Llama 3.1 70B)  │ 10.0     │ 50.0     │
│ 70B+ (Llama 3.1 405B)  │ 100.0    │ 500.0    │
├─────────────────────────┼──────────┼──────────┤
│ Agent invocation (any)  │ 5.0      │ 25.0     │
│ Training batch (Phase 6)│ 50.0     │ N/A      │
└─────────────────────────┴──────────┴──────────┘

Earning rates (HYDRA tokens per hour of uptime):
- Tier 0 (GPU server):   10.0/hr + inference fees
- Tier 1 (Desktop GPU):   3.0/hr + inference fees
- Tier 2 (Laptop CPU):    1.0/hr + verification fees
- Tier 3 (Phone):         0.3/hr + relay/embed fees

Daily decay: 5% on barter credits (unchanged)
HYDRA supply cap: 69,000,000 (unchanged)
Staking minimum: 100 HYDRA
Slash penalty: 10% of stake for verified fraud
```

**Free tier**: New users get 1000 barter credits (= 1M tokens on small models). Earn more by contributing compute.

---

## Phase 5: Currently Supported LLM Architectures & Expansion

### What Works Today

| Feature | PyTorch Runtime | MLX Runtime | Notes |
|---------|----------------|-------------|-------|
| Causal LM inference | ✅ | ✅ | Core path |
| Layer sharding | ✅ | ✅ (via reshard) | Activated end-to-end |
| Tensor parallelism | ❌ | ✅ (PipelineParallelMLX) | MLX only for now |
| NF4 quantization | ✅ (bitsandbytes) | ✅ (native) | 4× VRAM reduction |
| Int8 quantization | ✅ (bitsandbytes) | ✅ (native) | 2× VRAM reduction |
| KV cache compaction | ✅ | ❌ | PyTorch only (HAK/OMP) |
| Request coalescing | ✅ | ✅ | True tensor batching |
| Streaming decode | ✅ | ✅ | Token-by-token SSE |
| MoE routing | ✅ (expert_tags) | ❌ | Mixtral support |

### Expansion Targets

| Capability | Priority | Effort | Value |
|-----------|----------|--------|-------|
| **GGUF runtime** | P0 | 2 weeks | Opens llama.cpp ecosystem (thousands of models) |
| **Vision models** (LLaVA, Qwen-VL) | P1 | 2 weeks | Image → text; requires multi-modal forward pass |
| **Embedding models** (nomic, BGE) | P1 | 1 week | Enables RAG, semantic search, agent memory |
| **Speech-to-text** (Whisper) | P2 | 2 weeks | Voice input for mobile/desktop agents |
| **Text-to-speech** (Bark, XTTS) | P2 | 2 weeks | Voice output for assistants |
| **Image generation** (SDXL, Flux) | P3 | 3 weeks | Diffusion pipeline is architecturally different |
| **Code models** (Qwen-Coder, DeepSeek-Coder) | P0 | 0 weeks | Already works — just add to catalog |

---

## Phase 6: Federated Training (6–8 weeks)

### 6.1 Architecture

Federated training alongside inference on the same peer network:

```
Coordinator                          Peers
┌─────────────┐                     ┌──────────┐
│ Training Job│──gradient request──→│ Peer A   │
│ Scheduler   │                     │ (GPU)    │
│             │←──local gradients───│          │
│             │                     └──────────┘
│ Aggregator  │──gradient request──→┌──────────┐
│ (FedAvg /   │                     │ Peer B   │
│  FedProx)   │←──local gradients───│ (GPU)    │
│             │                     └──────────┘
│ Model       │                     ┌──────────┐
│ Checkpoint  │──updated weights───→│ Peer C   │
│ Publisher   │                     │ (serves) │
└─────────────┘                     └──────────┘
```

### 6.2 Key Components

| Component | Location | Description |
|-----------|----------|-------------|
| `TrainingJobScheduler` | `coordinator/training_scheduler.py` | Accepts training jobs, splits data, assigns peers |
| `GradientAggregator` | `coordinator/gradient_aggregator.py` | FedAvg / FedProx / scaffold aggregation |
| `LocalTrainer` | `peer/local_trainer.py` | Runs local training steps, computes gradients |
| `DPGradientClipper` | `peer/dp_gradients.py` | Clips and noises gradients for differential privacy |
| `CheckpointPublisher` | `coordinator/checkpoint_publisher.py` | Publishes aggregated weights to DHT for peers to download |

### 6.3 Privacy Guarantees

- **Differential privacy**: Gradient clipping + Gaussian noise (ε-DP per round)
- **Secure aggregation**: Peer gradients encrypted; coordinator only sees sum (via additive secret sharing)
- **No raw data leaves the device**: Only gradients (with DP noise) are transmitted

### 6.4 Training ↔ Inference Coexistence

- Peers declare `available_for_training: bool` in DHT announcements
- Training uses **off-peak compute**: when inference load < 30%, peer accepts training batches
- Inference always preempts training (drain training batch, serve inference, resume)
- Separate VRAM budget: training limited to leftover VRAM after inference KV cache reservation

---

## Implementation Priority & Timeline

```
Phase 1: Agent Runtime              ████████████████████░░░░  Weeks 1-5
Phase 3: HuggingFace Auto-Pop      ░░░░████████░░░░░░░░░░░░  Weeks 3-4
Phase 2: Agent-to-Agent            ░░░░░░░░████████░░░░░░░░  Weeks 5-7
Phase 5: Model Expansion (GGUF)    ░░░░░░████████░░░░░░░░░░  Weeks 4-5
Phase 4: Mobile Support            ░░░░░░░░░░░░████████████  Weeks 7-10
Phase 6: Federated Training        ░░░░░░░░░░░░░░░░████████  Weeks 9-14
```

**Critical path**: Phase 1 (agent runtime) unlocks everything else. Without it, OpenHydra is just another inference API. With it, OpenHydra becomes a platform.

---

## What Makes This Different

| Feature | OpenHydra AI OS | LangChain/CrewAI | Claude Code / Cursor |
|---------|----------------|-------------------|---------------------|
| **Decentralized inference** | ✅ Runs on volunteer hardware | ❌ Needs API key | ❌ Needs API key |
| **No single point of failure** | ✅ DHT + peer mesh | ❌ Central server | ❌ Central server |
| **Earn while idle** | ✅ HYDRA tokens | ❌ | ❌ |
| **Privacy** | ✅ Onion routing + DP | ❌ Data sent to API | ❌ Data sent to API |
| **Agent marketplace** | ✅ Peer-hosted agents | ❌ | ❌ |
| **Federated training** | ✅ Train on user data without sharing | ❌ | ❌ |
| **Mobile support** | ✅ Phone as compute node | ❌ | ❌ |
| **Open source** | ✅ Apache 2.0 | ✅ | ❌ |

---

## Files to Create/Modify

### New Files
| File | Phase | Lines (est.) |
|------|-------|-------------|
| `agent/__init__.py` | 1 | 5 |
| `agent/runtime.py` | 1 | ~400 |
| `agent/tools.py` | 1 | ~300 |
| `agent/memory.py` | 1 | ~250 |
| `agent/sandbox.py` | 1 | ~150 |
| `coordinator/model_registry.py` | 3 | ~300 |
| `coordinator/training_scheduler.py` | 6 | ~500 |
| `coordinator/gradient_aggregator.py` | 6 | ~400 |
| `peer/gguf_runtime.py` | 5 | ~350 |
| `peer/local_trainer.py` | 6 | ~300 |
| `peer/dp_gradients.py` | 6 | ~200 |

### Modified Files
| File | Phase | Changes |
|------|-------|---------|
| `coordinator/api_server.py` | 1 | Add `/v1/agents/*` endpoints |
| `coordinator/engine.py` | 1 | Add `run_agent()` delegation |
| `peer/server.py` | 2 | Add `AgentService` gRPC handlers |
| `peer/peer.proto` | 2 | Add `AgentService` RPCs |
| `peer/dht_announce.py` | 2,4 | Add agent capabilities, compute tier fields |
| `pyproject.toml` | 1,5 | Add `agent` and `gguf` optional deps |
| `desktop/src/` | 1 | Add Agent tab to Tauri app |

---

## First Concrete Step

**Phase 1.1**: Create `agent/runtime.py` with the ReAct loop, wired to existing `engine.infer_chat()`. Add `shell` and `read_file`/`write_file` tools with basic sandboxing. Add `POST /v1/agents/run` endpoint. Ship it — an agent that can write code using the decentralized network is the demo that makes people understand the vision.
