# TypeScript SDK

The OpenHydra TypeScript SDK works in browsers, Node.js, and Deno. It uses the native `fetch` API — no extra dependencies required.

---

## Installation

```bash
# npm
npm install openhydra-sdk

# From source
cd sdk/typescript
npm install
npm run build
```

---

## Quick start

```typescript
import { OpenHydraClient } from "openhydra-sdk";

const client = new OpenHydraClient("http://localhost:8080");

// List available models
const models = await client.models();
console.log(models);

// Send a chat completion
const response = await client.chatCompletions({
  model: "llama3-8b",
  messages: [
    { role: "user", content: "What is OpenHydra?" }
  ],
});
console.log(response);
```

---

## API reference

### `new OpenHydraClient(baseUrl)`

Create a new client instance.

| Parameter | Type | Description |
|-----------|------|-------------|
| `baseUrl` | `string` | Base URL of the OpenHydra coordinator |

---

### `client.models() → Promise<unknown>`

Fetch the list of available models.

```typescript
const result = await client.models();
// {
//   object: "list",
//   data: [
//     { id: "llama3-8b", peer_count: 3, avg_latency_ms: 120 },
//   ]
// }
```

---

### `client.chatCompletions(payload) → Promise<unknown>`

Send a chat completion request.

```typescript
import type { ChatCompletionRequest } from "openhydra-sdk";

const payload: ChatCompletionRequest = {
  model: "llama3-8b",
  messages: [
    { role: "system", content: "You are concise." },
    { role: "user",   content: "Explain DHT in one sentence." },
  ],
};

const response = await client.chatCompletions(payload);
```

**`ChatCompletionRequest` type**

```typescript
type ChatCompletionRequest = {
  model: string;
  messages: Array<{ role: string; content: string }>;
};
```

---

## Streaming example

For streaming responses, call the API directly using `fetch` with `stream: true`:

```typescript
const response = await fetch("http://localhost:8080/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "llama3-8b",
    messages: [{ role: "user", content: "Count to 5" }],
    stream: true,
  }),
});

const reader = response.body!.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (line.startsWith("data: ") && line !== "data: [DONE]") {
      const chunk = JSON.parse(line.slice(6));
      const delta = chunk.choices?.[0]?.delta?.content ?? "";
      process.stdout.write(delta);   // Node.js; use console.log in browser
    }
  }
}
```

---

## Error handling

The SDK throws on non-2xx HTTP responses. Parse the response body for details:

```typescript
try {
  const response = await client.chatCompletions({
    model: "unknown-model",
    messages: [{ role: "user", content: "Hi" }],
  });
} catch (err) {
  if (err instanceof Error) {
    console.error(err.message);
  }
}
```

---

## Browser usage

The SDK works in any modern browser that supports `fetch`. Point it at a CORS-enabled coordinator or route through the same origin:

```html
<script type="module">
  import { OpenHydraClient } from "/js/openhydra-sdk.min.js";
  const client = new OpenHydraClient("https://openhydra.co");
  const models = await client.models();
  console.log(models);
</script>
```

---

## Source

The SDK source lives in `sdk/typescript/src/index.ts`. Build output is ES modules targeting ES2020.
