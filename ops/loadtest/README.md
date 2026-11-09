# Load tests

`chat_load.py` load-tests an OpenHydra gateway (or any OpenAI-compatible endpoint) and checks a
time-to-first-token target. It streams every request, so the time to the first token is measured
where a client sees it. It groups requests into conversations whose later turns resend their history,
as coding agents do; that is what the gateway's cache affinity keys on. Standard library only.

```bash
python3 ops/loadtest/chat_load.py --url http://127.0.0.1:16527 --model qwen2.5:1.5b --requests 200 --concurrency 16 --turns 4 --ttft-p95-target 3
```

It prints a JSON summary:

- requests, successes and errors by message;
- p50, p95, p99 and maximum time to first token;
- total time;
- finish reasons.

It exits 1 when the p95 misses `--ttft-p95-target`, or when any request failed (unless
`--allow-errors`).

The gateway's own view is at `/metrics` (send the gateway's API key when it has one). Besides the existing request histograms, it now has
`openhydra_time_to_first_token_seconds` and `openhydra_decode_tokens_per_second`, each by model.
