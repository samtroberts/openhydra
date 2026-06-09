from __future__ import annotations

import json

import aiohttp

from .adapter import (
    SupernodeAdapter,
    PromptRequest,
    TokenChunk,
    ModelInfo,
    BackendStatus,
    BackendError,
)


class OllamaAdapter(SupernodeAdapter):

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self._active_requests: dict[str, bool] = {}

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def list_models(self) -> list[ModelInfo]:
        await self._ensure_session()
        async with self._session.get(f"{self._base_url}/api/tags") as resp:
            if resp.status != 200:
                raise BackendError(f"Ollama /api/tags returned {resp.status}")
            data = await resp.json()

        models = []
        for m in data.get("models", []):
            name = m["name"]
            details = m.get("details", {})
            family = details.get("family", name.split(":")[0])
            param_str = details.get("parameter_size", "0B")
            param_count = _parse_param_count(param_str)
            quant = details.get("quantization_level", "unknown")

            models.append(ModelInfo(
                model_id=name,
                model_family=family,
                parameter_count=param_count,
                quantization=quant,
                context_length=_get_context_length(name, details),
                supports_streaming=True,
                supports_system_prompt=True,
            ))
        return models

    async def generate(self, request: PromptRequest):
        await self._ensure_session()
        self._active_requests[request.request_id] = False

        if request.messages:
            url = f"{self._base_url}/api/chat"
            messages = list(request.messages)
            if request.system_prompt:
                messages = [{"role": "system", "content": request.system_prompt}] + messages
            body = {
                "model": request.model_id,
                "messages": messages,
                "stream": request.stream,
                "options": self._build_options(request),
            }
        else:
            url = f"{self._base_url}/api/generate"
            body = {
                "model": request.model_id,
                "prompt": request.prompt or "",
                "stream": request.stream,
                "options": self._build_options(request),
            }
            if request.system_prompt:
                body["system"] = request.system_prompt

        if request.response_format == "json":
            body["format"] = "json"

        try:
            async with self._session.post(url, json=body) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise BackendError(f"Ollama error {resp.status}: {error_text}")

                async for line in resp.content:
                    if self._active_requests.get(request.request_id):
                        yield TokenChunk(token="", finish_reason="cancelled")
                        return

                    line = line.strip()
                    if not line:
                        continue

                    chunk = json.loads(line)

                    if request.messages:
                        token_text = chunk.get("message", {}).get("content", "")
                    else:
                        token_text = chunk.get("response", "")

                    done = chunk.get("done", False)

                    if done:
                        finish = chunk.get("done_reason", "stop")
                        yield TokenChunk(token=token_text, finish_reason=finish)
                    elif token_text:
                        yield TokenChunk(token=token_text)
        finally:
            self._active_requests.pop(request.request_id, None)

    async def cancel(self, request_id: str) -> None:
        self._active_requests[request_id] = True

    async def get_status(self) -> BackendStatus:
        await self._ensure_session()
        async with self._session.get(f"{self._base_url}/api/ps") as resp:
            if resp.status != 200:
                raise BackendError(f"Ollama /api/ps returned {resp.status}")
            data = await resp.json()

        running = data.get("models", [])
        models_loaded = [m["name"] for m in running]
        total_mem = sum(m.get("size_vram", 0) for m in running)
        active = len(self._active_requests)

        return BackendStatus(
            current_load=min(active / max(self._max_concurrent(), 1), 1.0),
            active_requests=active,
            max_concurrent=self._max_concurrent(),
            gpu_memory_free_mb=_estimate_free_memory(total_mem),
            models_loaded=models_loaded,
        )

    async def health_check(self) -> bool:
        try:
            await self._ensure_session()
            async with self._session.get(
                self._base_url,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def warmup(self, model_id: str) -> bool:
        await self._ensure_session()
        body = {"model": model_id, "prompt": "", "keep_alive": "10m"}
        try:
            async with self._session.post(
                f"{self._base_url}/api/generate", json=body
            ) as resp:
                async for _ in resp.content:
                    pass
                return resp.status == 200
        except Exception:
            return False

    def backend_type(self) -> str:
        return "ollama"

    def _build_options(self, request: PromptRequest) -> dict:
        opts: dict = {}
        if request.temperature is not None:
            opts["temperature"] = request.temperature
        if request.top_p is not None:
            opts["top_p"] = request.top_p
        if request.top_k is not None:
            opts["top_k"] = request.top_k
        if request.max_tokens is not None:
            opts["num_predict"] = request.max_tokens
        if request.stop:
            opts["stop"] = request.stop
        return opts

    def _max_concurrent(self) -> int:
        return 4


def _parse_param_count(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("B"):
        try:
            return int(float(s[:-1]) * 1000)
        except ValueError:
            return 0
    return 0


def _get_context_length(name: str, details: dict) -> int:
    defaults = {
        "llama3.1": 131072,
        "llama3": 8192,
        "qwen": 32768,
        "mistral": 32768,
        "gemma": 8192,
        "phi": 4096,
        "deepseek": 65536,
    }
    lower_name = name.lower()
    for prefix, ctx in defaults.items():
        if prefix in lower_name:
            return ctx
    return 4096


def _estimate_free_memory(used_bytes: int) -> int:
    return max(0, 16000 - used_bytes // (1024 * 1024))
