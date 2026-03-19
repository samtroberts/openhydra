const BASE_URL = "http://127.0.0.1:8080";

export async function fetchHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

export async function fetchBalance(): Promise<{ hydra: number; credits: number }> {
  try {
    const res = await fetch(`${BASE_URL}/v1/account/balance`, { signal: AbortSignal.timeout(3000) });
    if (!res.ok) return { hydra: 0, credits: 0 };
    const data = await res.json();
    return { hydra: data.hydra_balance ?? 0, credits: data.barter_credits ?? 0 };
  } catch {
    return { hydra: 0, credits: 0 };
  }
}

export async function* streamChat(
  model: string,
  messages: { role: string; content: string }[],
  signal?: AbortSignal
): AsyncGenerator<string> {
  const res = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, messages, stream: true, max_tokens: 1024 }),
    signal,
  });

  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const reader = res.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data === "[DONE]") return;
      try {
        const parsed = JSON.parse(data);
        const content = parsed.choices?.[0]?.delta?.content;
        if (content) yield content;
      } catch { /* skip malformed SSE */ }
    }
  }
}
