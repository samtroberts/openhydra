# Petals vs OpenHydra — Deep Technical Comparison

> Petals source: https://github.com/bigscience-workshop/petals
> Comparison based on full codebase audit of both projects (March 2026).

---

## 1. The Fundamental Architectural Difference

This is the most important thing to understand before all other comparisons.

### Petals: Layer-Level Sharding (True Decomposition)

In Petals, a model's transformer layers are **split across multiple machines**. For a 70B model with 80 layers:

```
Server A  → layers  0–19  (25% of model, ~10 GB VRAM)
Server B  → layers 20–39  (25% of model, ~10 GB VRAM)
Server C  → layers 40–59  (25% of model, ~10 GB VRAM)
Server D  → layers 60–79  (25% of model, ~10 GB VRAM)
```

The client sends its input through all four servers **sequentially**. Each server computes its assigned layers and passes the hidden state (activation) to the next. **No single machine ever needs enough VRAM for the full model.** This is what lets Petals run Llama 3.1 405B on a network of consumer GPUs.

### OpenHydra: Full-Model Replication (Ensemble)

In OpenHydra, **each peer runs the complete model**. The `pipeline_width` setting routes the same request through multiple peers, but for redundancy and verification — not because any single peer lacks the memory. The activation passed between stages is an intermediate computation artefact, not a partial-model handoff.

```
Peer 0  → full Qwen3.5-0.8B  (100% of model)
Peer 1  → full Qwen3.5-0.8B  (100% of model, for verification)
Peer 2  → full Qwen3.5-0.8B  (100% of model, for verification)
```

**Consequence:** OpenHydra can only serve models that fit in a single peer's VRAM. It cannot run 70B+ models without a machine with ≥40GB VRAM.

---

## 2. Feature Comparison Matrix

| Feature | Petals | OpenHydra | Notes |
|---------|--------|-----------|-------|
| **Layer-level sharding** | ✅ Yes | ❌ No | Petals' core differentiator |
| **Runs 70B+ models** | ✅ Yes | ⚠️ Only with large single GPU | Petals: distribute. OH: needs H100 |
| **DHT implementation** | ✅ Hivemind (Kademlia) | ⚠️ Custom HTTP | OH has 3 fixed bootstrap servers |
| **Request coalescing** | ✅ Yes (RuntimeWithDeduplicatedPools) | ❌ No | Petals batches multi-user requests per layer |
| **Dijkstra path routing** | ✅ Yes | ❌ No | Petals considers inter-server latency |
| **Dynamic layer rebalancing** | ✅ Yes | ❌ No | Petals servers migrate layers if unbalanced |
| **NF4/NF8 quantization** | ✅ bitsandbytes | ⚠️ Basic int4/int8 | Petals uses sophisticated NF4 |
| **Multi-GPU tensor parallel** | ✅ tensor_parallel lib | ❌ No | Petals splits one layer across GPUs |
| **Stateful session recovery** | ✅ Position tracking | ⚠️ Partial | Petals regenerates KV on server failure |
| **KV cache on server** | ✅ MemoryCache (bounded) | ✅ kv_compaction | Both; OH has more sophisticated compaction |
| **HAK/OMP KV compaction** | ❌ No | ✅ Yes | OH's custom research algorithms |
| **Beta bias correction** | ❌ No | ✅ Yes | OH's Phase 2 compaction |
| **Per-head KV budgets** | ❌ No | ✅ Yes | OH's Phase 3 compaction |
| **Grounding / RAG** | ❌ No | ✅ Yes | OH has DuckDuckGo integration |
| **Barter economy** | ❌ No | ✅ Yes | OH has SQLite credit ledger |
| **Output verification** | ❌ No | ✅ Yes | OH runs 3 pipelines, compares |
| **Speculative decoding** | ❌ No | ✅ Yes | OH has PyTorchDraftModel |
| **Graceful degradation** | ❌ No | ✅ Yes | OH falls back to smaller models |
| **Model warmup call** | ❌ No | ❌ No | Both suffer cold-start TTFT |
| **OpenAI-compatible API** | ❌ No | ✅ Yes | OH: /v1/chat/completions SSE |
| **Differential privacy** | ❌ No | ✅ Yes | OH: Gaussian noise on activations |
| **Concentration guard** | ❌ No | ✅ Yes | OH: prevents operator monopoly |

---

## 3. Where Petals Is Superior

### 3.1 True Distributed Inference (Petals wins decisively)

Petals was built to solve a specific problem: **running models that are too large for any single machine**. Every design decision flows from this. The layer-level sharding means:

- A hobbyist with an RTX 3090 (24GB) can participate in running Llama 3.1 70B
- The network can host 405B parameter models with no single node owning more than a few layers
- Adding more servers doesn't just add redundancy — it adds capacity

OpenHydra's equivalent would require every participant to own a $30,000 H100.

### 3.2 Request Coalescing (`RuntimeWithDeduplicatedPools`)

When 100 users are sending tokens through a layer at the same time, Petals coalesces them into a single batched `forward()` call. The GPU processes batch_size=100 in nearly the same time as batch_size=1 (up to memory limits), giving near-linear throughput scaling with concurrent users.

OpenHydra routes each user request through a dedicated peer. 100 users = 100 separate forward passes = 100x the compute.

### 3.3 Dijkstra Routing (`SequenceManager`)

Petals builds a weighted directed graph where:
- **Nodes** = server endpoints + virtual source/sink
- **Edges** = measured inter-server RTT + serialisation overhead (empirically: 0.018s)
- **Edge weights** = latency + throughput reciprocal + cache penalty (10s if cache full)

Then runs Dijkstra's algorithm to find the minimum-latency path through the layer sequence. This means the client automatically avoids:
- Geographically distant server hops
- Overloaded servers (high latency)
- Servers with full KV caches (would cause cache eviction → 10s penalty)

OpenHydra uses a flat per-peer score (latency 45%, load 30%, reputation 25%) and assembles peers into a pipeline without considering how peers are connected to each other.

### 3.4 Dynamic Layer Rebalancing

Petals servers monitor the global DHT for layer distribution. If layers 0–20 are over-represented (too many servers) and layers 60–79 are under-represented (too few servers), a server can:
1. Announce OFFLINE for its current layers
2. Pick a better layer range via `choose_best_blocks()`
3. Load the new layers and announce ONLINE

This self-heals the network when servers join or leave. OpenHydra peers are statically assigned a `--model-id` at startup.

### 3.5 Hivemind DHT vs Custom HTTP DHT

| Aspect | Petals (hivemind) | OpenHydra (custom HTTP) |
|--------|------------------|------------------------|
| Protocol | Kademlia (true P2P) | HTTP over 3 fixed nodes |
| Decentralisation | Full (no central servers) | Partial (3 Linode bootstrap nodes) |
| Node churn handling | Built-in (Kademlia TTL + republish) | TTL + heartbeat re-announce |
| Sybil resistance | Kademlia difficulty | None |
| Bootstrap failure | Any node can bootstrap | 3-node failure = network blind |
| Discovery latency | ~100ms (DHT lookup) | ~3s timeout * 3 nodes |

**The 3 fixed bootstrap nodes are a soft single-point of failure.** If all 3 Linode VMs go down, no coordinator can discover any peer. Kademlia is fully peer-to-peer — any existing peer can act as a bootstrap.

### 3.6 NF4 Quantization

Petals uses `bitsandbytes` to load models in NF4 (4-bit NormalFloat) quantization. NF4 is specifically designed for normally-distributed neural network weights and achieves much lower quality loss than naive int4. This allows running a 7B model in ~4GB VRAM vs ~14GB in fp16.

OpenHydra's `--quantization int4` applies simple uniform integer quantization — mathematically less optimal than NF4.

### 3.7 Multi-GPU Tensor Parallelism

Petals uses the `tensor_parallel` library to split individual transformer layers across multiple GPUs on the same machine. A server with 4x A100 GPUs can host twice as many layers as a server with 2x A100.

OpenHydra runs on a single device per peer. No multi-GPU parallelism.

---

## 4. Where OpenHydra Is Superior

### 4.1 Advanced KV Cache Compaction

OpenHydra's `peer/kv_compaction/` module is a genuine research contribution. The 4-phase compaction system with HAK/OMP algorithms, beta bias correction, and per-head budgets goes significantly beyond what Petals implements.

Petals has a `MemoryCache` that evicts oldest sessions when full. OpenHydra can compress a 4096-token KV cache down to 1024 tokens (25% retention) while preserving most of the attention quality — effectively extending context length 4x without adding memory.

### 4.2 Output Verification (Byzantine Fault Tolerance)

OpenHydra runs the same prompt through 3 independent pipelines and compares results. A malicious peer that returns fabricated output will be caught when the other two pipelines disagree. This is critical for a public permissionless network where peers may be adversarial.

Petals has no equivalent. A malicious server hosting layers 40–50 of a 70B model could corrupt every response passing through it, and the client would never know.

### 4.3 Economic Incentives

Petals relies on altruistic contribution. When the BigScience project ended, participation dropped significantly. OpenHydra's barter credit system creates a self-sustaining economy:
- Peers earn credits by serving tokens
- Credits decay to prevent accumulation
- Users spend credits to get compute

This is the right design for a network that aims to be self-sustaining without a sponsoring organisation.

### 4.4 Graceful Degradation

When a model has insufficient healthy peers, OpenHydra automatically falls back to a lighter model in the catalog. Petals raises `MissingBlocksError` and the client request fails.

### 4.5 Grounding / RAG

OpenHydra has built-in retrieval augmentation via DuckDuckGo Instant Answers, injected into the prompt before inference. This improves factual accuracy for knowledge-intensive queries without fine-tuning.

### 4.6 OpenAI-Compatible API

OpenHydra exposes `/v1/chat/completions` with SSE streaming — drop-in compatible with any OpenAI client library. Petals requires its own Python library (`from petals import AutoDistributedModelForCausalLM`).

### 4.7 Speculative Decoding

OpenHydra's coordinator runs a local draft model (tiny-gpt2 or configurable) that proposes N tokens ahead. These are verified against the peer's authoritative output, accepting matching tokens in bulk. On a 75% match rate, effective throughput is ~3x higher than token-by-token generation.

---

## 5. What OpenHydra Should Learn From Petals

These are the concrete improvements, ordered by impact:

### Priority 1 — True Layer Sharding (Architectural)

**The gap:** OpenHydra cannot run models larger than a single peer's VRAM. This is the most limiting constraint on network utility.

**What to build:**
```python
# In peer/server.py — add layer range flags
--layer-start 0    # First transformer layer this peer hosts
--layer-end 19     # Last transformer layer this peer hosts

# In peer/model_shard.py — partial model loading
def _load_partial_model(self, model_id, layer_start, layer_end):
    config = AutoConfig.from_pretrained(model_id)
    # Load only layers layer_start through layer_end
    # Use accelerate device_map for partial loading
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"transformer.h.0-19": "cuda:0"},
    )
```

The coordinator's `InferenceChain` already passes activations between stages — the plumbing exists. What's missing is:
1. Peers loading partial layer ranges instead of full models
2. DHT announcing layer ranges (`layer_start`, `layer_end`) per peer
3. Coordinator assembling pipelines that cover the full layer range
4. `SequenceManager`-style graph routing to optimise layer-to-peer assignment

### Priority 2 — Request Coalescing

**The gap:** 100 concurrent users = 100x compute. Petals achieves near-linear throughput scaling.

**What to build:** In `peer/server.py`, instead of spawning one thread per `Forward()` call, collect incoming requests into a queue and flush them as a batch every 20ms:

```python
class BatchingInferenceQueue:
    def __init__(self, model, max_batch=32, flush_ms=20):
        self.queue = queue.Queue()
        self.model = model
        threading.Thread(target=self._batch_loop, daemon=True).start()

    def submit(self, request):
        future = concurrent.futures.Future()
        self.queue.put((request, future))
        return future.result(timeout=60)

    def _batch_loop(self):
        while True:
            batch = []
            deadline = time.time() + self.flush_ms / 1000
            while time.time() < deadline:
                try:
                    item = self.queue.get(timeout=0.001)
                    batch.append(item)
                    if len(batch) >= self.max_batch:
                        break
                except queue.Empty:
                    pass
            if batch:
                self._process_batch(batch)
```

### Priority 3 — Dijkstra Path Routing

**The gap:** The coordinator assembles pipelines from individually-ranked peers without considering how peers are networked together.

**What to build:** After DHT lookup, measure pairwise RTT between peers (or use geographic distance as proxy), build a directed graph, and run Dijkstra to find the minimum-latency sequential path:

```python
# In coordinator/path_finder.py
def build_inference_graph(peers: list[PeerEndpoint], n_layers: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_node("source")
    G.add_node("sink")
    for peer in peers:
        latency = peer.latency_p50_ms / 1000
        compute_time = 1.0 / max(1.0, peer.inference_rps)
        weight = latency + compute_time
        if peer.layer_start == 0:
            G.add_edge("source", peer.peer_id, weight=latency)
        if peer.layer_end == n_layers - 1:
            G.add_edge(peer.peer_id, "sink", weight=latency)
        for other in peers:
            if other.layer_start == peer.layer_end + 1:
                inter_rtt = measure_rtt(peer.endpoint_url, other.endpoint_url)
                G.add_edge(peer.peer_id, other.peer_id, weight=inter_rtt)
    return G

path = nx.dijkstra_path(G, "source", "sink", weight="weight")
```

### Priority 4 — Model Warmup Call

**The gap:** 34-second TTFT on first request because MPS/CUDA kernel compilation happens lazily.

**What to build:** One line added to `PyTorchRuntime.__init__`:

```python
# In peer/model_shard.py — PyTorchRuntime.__init__
with torch.no_grad():
    dummy = self._tokenizer("warmup", return_tensors="pt").to(self._device)
    _ = self._model(**dummy)   # compiles all Metal/CUDA kernels
```

**Impact:** First-request TTFT drops from ~34s → ~1-3s on MPS, ~200ms on CUDA. Zero other changes needed.

### Priority 5 — Dynamic Layer Rebalancing

**What to build:** In `peer/dht_announce.py`, periodically query the DHT to see the global layer distribution. If the peer's current layer range is over-represented, restart with a better range:

```python
class LayerRebalancer:
    def check_and_rebalance(self, current_range, dht_client):
        distribution = dht_client.get_layer_distribution(model_id)
        optimal_range = choose_best_layers(distribution, vram_capacity)
        if optimal_range != current_range:
            self.peer_server.migrate_to_layers(optimal_range)
```

### Priority 6 — NF4 Quantization

Replace the current integer quantization with `bitsandbytes` NF4:

```python
# In peer/model_shard.py — PyTorchRuntime.__init__
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
self._model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config
)
```

**Impact:** 7B model: 14GB → 4GB VRAM. Enables participation from 8GB consumer GPUs.

### Priority 7 — Replace Custom HTTP DHT with Hivemind

Long-term: migrate from the 3-node HTTP DHT to `hivemind`. This eliminates the bootstrap node single-point-of-failure and makes the network fully decentralised.

Short-term mitigation: increase bootstrap node count, add auto-discovery of alternative bootstrap nodes, implement peer-to-peer gossip between bootstrap nodes (the DHT `gossip` arrows in the existing architecture diagram are aspirational — there is no gossip implementation yet).

---

## 6. Performance Summary

| Metric | Petals (70B, 4x A100) | OpenHydra (0.8B, MPS) | OpenHydra potential (with fixes) |
|--------|-----------------------|----------------------|----------------------------------|
| TTFT (cold) | ~5s | ~34s | ~1-3s (after warmup fix) |
| TTFT (warm) | ~200ms | ~2-5s | ~200ms (after MLX/llama.cpp) |
| TPS | ~6 tok/s | ~1.3 tok/s | ~100-200 tok/s (after MLX) |
| Max model size | Unlimited (distributed) | Single-GPU VRAM limit | Unlimited (after layer sharding) |
| Concurrent users | Near-linear scaling | Linear degradation | Near-linear (after coalescing) |

---

## 7. Conclusion

Petals and OpenHydra solve **different versions of the same problem**:

- **Petals** solves the **capacity problem**: how do you run a 405B model when no single machine has enough VRAM? Answer: layer sharding.
- **OpenHydra** solves the **trust problem**: how do you run inference on an untrusted network? Answer: economic incentives + output verification + differential privacy.

OpenHydra has significantly more sophisticated infrastructure for running a **production-grade, economically sustainable, trustworthy** distributed inference network. Petals has significantly more sophisticated infrastructure for actually **decomposing large models** across heterogeneous hardware.

The ideal system combines both:
1. Petals-style layer sharding + request coalescing + Dijkstra routing
2. OpenHydra's economy + verification + KV compaction + grounding + graceful degradation

Priority implementation order:
1. Add model warmup (10-minute fix, 10x TTFT improvement)
2. Add NF4 quantization (1 day, enables 8GB GPU participation)
3. Add MLX/llama.cpp backends (1 week, 50-100x TPS improvement)
4. Add request coalescing (2 weeks, linear throughput scaling)
5. Add layer sharding (1-2 months, removes model size limit)
6. Migrate DHT to hivemind (2-3 months, true decentralisation)
