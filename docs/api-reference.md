# API Reference

The OpenHydra coordinator exposes an **OpenAI-compatible REST API** on port `8080` (default). All endpoints accept and return JSON.

Base URL: `http://<coordinator-host>:8080`

---

## Chat completions

### `POST /v1/chat/completions`

Send a chat completion request. The coordinator routes it to the best available peer.

**Request**

```json
{
  "model": "llama3-8b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is OpenHydra?"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 512
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | ✓ | Model ID to route to (must match a registered peer) |
| `messages` | array | ✓ | Conversation turns (`system`, `user`, `assistant`) |
| `stream` | boolean | | Stream tokens via SSE (default: `false`) |
| `temperature` | float | | Sampling temperature 0–2 (default: `1.0`) |
| `max_tokens` | integer | | Maximum tokens to generate |

**Response (non-streaming)**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "llama3-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "OpenHydra is a decentralised P2P inference network..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 87,
    "total_tokens": 129
  }
}
```

**Response (streaming, `stream: true`)**

Returns `text/event-stream` SSE:

```
data: {"choices":[{"delta":{"content":"Open"},"index":0}]}
data: {"choices":[{"delta":{"content":"Hydra"},"index":0}]}
data: [DONE]
```

---

## Models

### `GET /v1/models`

List all models currently available across registered peers.

**Response**

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3-8b",
      "object": "model",
      "created": 1710000000,
      "owned_by": "openhydra",
      "peer_count": 3,
      "avg_latency_ms": 142
    },
    {
      "id": "mistral-7b",
      "object": "model",
      "created": 1710000000,
      "owned_by": "openhydra",
      "peer_count": 1,
      "avg_latency_ms": 210
    }
  ]
}
```

---

## Peers

### `GET /v1/peers`

List all registered peers and their health status.

**Response**

```json
{
  "peers": [
    {
      "id": "peer-us-east-01",
      "model_id": "llama3-8b",
      "dht_url": "http://bootstrap-us.openhydra.co:8468",
      "region": "us-east",
      "latency_ms": 98,
      "status": "healthy",
      "registered_at": 1710000000
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `status` | `healthy` \| `degraded` \| `unreachable` |
| `latency_ms` | Rolling average gRPC round-trip latency |
| `region` | Inferred from DHT URL or declared by peer at registration |

---

## Account / Earnings

### `GET /v1/account/balance`

Retrieve the current node's barter credit balance and HYDRA token holdings.

**Response**

```json
{
  "peer_id": "my-laptop",
  "hydra_balance": 12.3456,
  "barter_credits": 47.20,
  "total_requests_served": 1203,
  "payout_threshold": 100.0
}
```

---

## Health

### `GET /health`

Simple liveness probe.

```json
{"status": "ok", "version": "0.1.0"}
```

---

## Error responses

All errors follow a consistent shape:

```json
{
  "error": {
    "message": "No healthy peers available for model 'llama3-70b'",
    "type": "no_peers_available",
    "code": 503
  }
}
```

| HTTP status | `type` | Meaning |
|-------------|--------|---------|
| 400 | `invalid_request` | Malformed JSON or missing required fields |
| 404 | `model_not_found` | No peer registered for the requested model |
| 429 | `rate_limit_exceeded` | Too many requests from this client IP |
| 503 | `no_peers_available` | All peers for the model are unreachable |
| 500 | `internal_error` | Unexpected coordinator error |
