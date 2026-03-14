# Python SDK

The OpenHydra Python SDK is a zero-dependency client that works with any Python 3.11+ environment. It speaks the OpenAI-compatible REST API, so it also works against the OpenAI API by changing the `base_url`.

---

## Installation

```bash
# From PyPI (when published)
pip install openhydra-sdk

# From source
pip install -e "sdk/python"
```

---

## Quick start

```python
from openhydra_sdk import OpenHydraClient

client = OpenHydraClient(base_url="http://localhost:8080")

# List available models
models = client.models()
print(models)

# Send a chat completion
response = client.chat_completions({
    "model": "llama3-8b",
    "messages": [
        {"role": "user", "content": "Explain KV cache compaction in one sentence."}
    ]
})
print(response["choices"][0]["message"]["content"])
```

---

## API reference

### `OpenHydraClient(base_url)`

Create a new client instance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_url` | `str` | Base URL of the OpenHydra coordinator, e.g. `http://localhost:8080` |

---

### `client.models() → dict`

Fetch the list of available models from the coordinator.

```python
result = client.models()
# {
#   "object": "list",
#   "data": [
#     {"id": "llama3-8b", "peer_count": 3, "avg_latency_ms": 120},
#     ...
#   ]
# }
```

---

### `client.chat_completions(payload) → dict`

Send a chat completion request and return the full response.

```python
response = client.chat_completions({
    "model": "llama3-8b",
    "messages": [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "What is 2 + 2?"}
    ],
    "temperature": 0.2,
    "max_tokens": 64
})
text = response["choices"][0]["message"]["content"]
```

---

## Streaming example

The SDK uses `urllib.request` and does not have built-in SSE support. For streaming, use `httpx` or `requests` with a chunked iterator:

```python
import httpx, json

with httpx.stream(
    "POST",
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Count to 10"}],
        "stream": True,
    },
    timeout=60,
) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
```

---

## Error handling

The SDK raises `urllib.error.HTTPError` on non-2xx responses. The response body contains a JSON error object:

```python
import json
from urllib.error import HTTPError
from openhydra_sdk import OpenHydraClient

client = OpenHydraClient("http://localhost:8080")

try:
    response = client.chat_completions({
        "model": "nonexistent-model",
        "messages": [{"role": "user", "content": "Hi"}]
    })
except HTTPError as e:
    error = json.loads(e.read().decode())
    print(error["error"]["message"])
    # "No healthy peers available for model 'nonexistent-model'"
```

---

## Source

The SDK source lives in `sdk/python/openhydra_sdk/client.py`. It has no third-party dependencies — only the Python standard library.
