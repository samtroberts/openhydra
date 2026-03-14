export type ChatCompletionRequest = {
  model: string;
  messages: Array<{ role: string; content: string }>;
};

export class OpenHydraClient {
  constructor(private readonly baseUrl: string) {}

  async models(): Promise<unknown> {
    const res = await fetch(`${this.baseUrl}/v1/models`);
    return res.json();
  }

  async chatCompletions(payload: ChatCompletionRequest): Promise<unknown> {
    const res = await fetch(`${this.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  }
}
