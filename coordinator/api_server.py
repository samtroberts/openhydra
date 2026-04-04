# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import gc
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import signal
import threading
import time
import uuid
from typing import Any
from urllib.parse import parse_qs, urlparse

from coordinator.bandwidth_roles import estimate_prompt_tokens
from coordinator.engine import CoordinatorEngine, EngineConfig
from openhydra_defaults import PRODUCTION_BOOTSTRAP_URLS
from openhydra_logging import configure_logging
from openhydra_secrets import is_insecure_secret_value, load_secret_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request validation limits
# ---------------------------------------------------------------------------
_MAX_TOKENS_LIMIT: int = 8192       # elastic ceiling; engine enforces 2048 floor based on redundancy
_MAX_PIPELINE_WIDTH: int = 16       # hard ceiling on pipeline_width
_MAX_PROMPT_CHARS: int = 65_536     # fast char pre-filter (raised so token limit is authoritative)
_MAX_PROMPT_TOKENS: int = 8_192     # hard ceiling on input tokens (word-count estimate)
_MAX_CLIENT_ID_LEN: int = 128       # sanity cap on client_id / channel_id


# ---------------------------------------------------------------------------
# Per-IP sliding-window rate limiter
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Thread-safe sliding-window rate limiter keyed by client IP."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check(self, client_ip: str) -> tuple[bool, int, int]:
        """Return (allowed, remaining, reset_unix_timestamp).

        reset_unix_timestamp is the Unix epoch second at which the oldest
        request in the current window expires, i.e. when the window resets
        and ``remaining`` becomes ``_max`` again.
        """
        now = time.monotonic()
        now_wall = int(time.time())
        cutoff = now - self._window
        with self._lock:
            timestamps = self._buckets.get(client_ip, [])
            timestamps = [ts for ts in timestamps if ts > cutoff]
            count = len(timestamps)
            if count >= self._max:
                self._buckets[client_ip] = timestamps
                oldest = timestamps[0] if timestamps else now
                reset_unix = now_wall + max(1, int(oldest + self._window - now))
                return False, 0, reset_unix
            timestamps.append(now)
            self._buckets[client_ip] = timestamps
            remaining = self._max - len(timestamps)
            oldest = timestamps[0]
            reset_unix = now_wall + max(1, int(oldest + self._window - now))
            return True, remaining, reset_unix

    def is_allowed(self, client_ip: str) -> bool:
        """Backward-compatible wrapper; prefer check() to obtain header data."""
        allowed, _, _ = self.check(client_ip)
        return allowed


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _parse_expert_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [item.strip().lower() for item in raw.split(",")]
    else:
        try:
            values = [str(item).strip().lower() for item in list(raw)]
        except TypeError:
            values = []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _parse_expert_layers(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [item.strip() for item in raw.split(",")]
    else:
        try:
            values = [str(item).strip() for item in list(raw)]
        except TypeError:
            values = []
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        if not value:
            continue
        try:
            idx = int(value)
        except ValueError:
            continue
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return sorted(out)


def _parse_decode_options(body: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "do_sample" in body:
        out["decode_do_sample"] = bool(body.get("do_sample"))
    if "temperature" in body and body.get("temperature") is not None:
        try:
            out["decode_temperature"] = float(body.get("temperature"))
        except (TypeError, ValueError):
            pass
    if "top_p" in body and body.get("top_p") is not None:
        try:
            out["decode_top_p"] = float(body.get("top_p"))
        except (TypeError, ValueError):
            pass
    if "top_k" in body and body.get("top_k") is not None:
        try:
            out["decode_top_k"] = int(body.get("top_k"))
        except (TypeError, ValueError):
            pass
    if "seed" in body and body.get("seed") is not None:
        try:
            out["decode_seed"] = int(body.get("seed"))
        except (TypeError, ValueError):
            pass
    return out


def _parse_ollama_options(opts: dict[str, Any]) -> dict[str, Any]:
    """Map an Ollama ``options`` dict to engine decode kwargs + optional ``max_tokens``.

    Handles ``num_predict``, ``temperature``, ``top_p``, ``top_k``, ``seed``.
    All other Ollama options are silently ignored (forward-compatibility).
    """
    out: dict[str, Any] = {}
    if "num_predict" in opts:
        try:
            out["max_tokens"] = int(opts["num_predict"])
        except (TypeError, ValueError):
            pass
    if "temperature" in opts and opts.get("temperature") is not None:
        try:
            out["decode_temperature"] = float(opts["temperature"])
        except (TypeError, ValueError):
            pass
    if "top_p" in opts and opts.get("top_p") is not None:
        try:
            out["decode_top_p"] = float(opts["top_p"])
        except (TypeError, ValueError):
            pass
    if "top_k" in opts and opts.get("top_k") is not None:
        try:
            out["decode_top_k"] = int(opts["top_k"])
        except (TypeError, ValueError):
            pass
    if "seed" in opts and opts.get("seed") is not None:
        try:
            out["decode_seed"] = int(opts["seed"])
        except (TypeError, ValueError):
            pass
    return out


def _validate_infer_params(body: dict[str, Any]) -> str | None:
    """Validate inference request parameters. Returns an error string or None."""
    max_tokens = body.get("max_tokens")
    if max_tokens is not None:
        try:
            v = int(max_tokens)
        except (TypeError, ValueError):
            return "max_tokens must be an integer"
        if v < 1:
            return "max_tokens must be >= 1"
        if v > _MAX_TOKENS_LIMIT:
            return f"max_tokens must be <= {_MAX_TOKENS_LIMIT}"

    pipeline_width = body.get("pipeline_width")
    if pipeline_width is not None:
        try:
            v = int(pipeline_width)
        except (TypeError, ValueError):
            return "pipeline_width must be an integer"
        if v < 1:
            return "pipeline_width must be >= 1"
        if v > _MAX_PIPELINE_WIDTH:
            return f"pipeline_width must be <= {_MAX_PIPELINE_WIDTH}"

    prompt = body.get("prompt")
    if prompt is not None:
        if len(str(prompt)) > _MAX_PROMPT_CHARS:
            return f"prompt exceeds maximum length of {_MAX_PROMPT_CHARS} characters"
        tok_est = estimate_prompt_tokens(str(prompt))
        if tok_est > _MAX_PROMPT_TOKENS:
            return (
                f"prompt exceeds maximum of {_MAX_PROMPT_TOKENS} input tokens "
                f"(estimated {tok_est})"
            )

    messages = body.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            return "messages must be an array"
        total = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
        if total > _MAX_PROMPT_CHARS:
            return f"total message content exceeds {_MAX_PROMPT_CHARS} characters"
        tok_est = sum(
            estimate_prompt_tokens(str(m.get("content", "")))
            for m in messages
            if isinstance(m, dict)
        )
        if tok_est > _MAX_PROMPT_TOKENS:
            return (
                f"total message content exceeds {_MAX_PROMPT_TOKENS} input tokens "
                f"(estimated {tok_est})"
            )

    client_id = body.get("client_id")
    if client_id is not None and len(str(client_id)) > _MAX_CLIENT_ID_LEN:
        return f"client_id exceeds maximum length of {_MAX_CLIENT_ID_LEN}"

    return None


def _resolve_runtime_profile_settings(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict[str, Any]:
    profile = str(getattr(args, "deployment_profile", "dev") or "dev").strip().lower()
    if profile not in {"dev", "prod"}:
        parser.error("unsupported deployment profile")
    secrets_file = getattr(args, "secrets_file", None)
    try:
        secret_store = load_secret_store(secrets_file)
    except RuntimeError as exc:
        parser.error(str(exc))

    mock_mode_arg = getattr(args, "hydra_ledger_bridge_mock_mode", None)
    if mock_mode_arg is None:
        hydra_ledger_bridge_mock_mode = (profile != "prod")
    else:
        hydra_ledger_bridge_mock_mode = bool(mock_mode_arg)

    advanced_encryption_seed = str(getattr(args, "advanced_encryption_seed", "") or "").strip()
    if profile == "prod":
        if hydra_ledger_bridge_mock_mode:
            parser.error("prod profile forbids mock ledger mode; use --no-hydra-ledger-bridge-mock-mode")

        if not bool(getattr(args, "tls_enable", False)):
            parser.error("prod profile requires --tls-enable")
        if not getattr(args, "tls_root_cert_path", None):
            parser.error("prod profile requires --tls-root-cert-path")
        if not getattr(args, "tls_client_cert_path", None):
            parser.error("prod profile requires --tls-client-cert-path")
        if not getattr(args, "tls_client_key_path", None):
            parser.error("prod profile requires --tls-client-key-path")
        if not getattr(args, "tls_server_name_override", None):
            parser.error("prod profile requires --tls-server-name-override")

        if is_insecure_secret_value(advanced_encryption_seed):
            advanced_encryption_seed = str(
                secret_store.get("OPENHYDRA_ADVANCED_ENCRYPTION_SEED", advanced_encryption_seed) or ""
            ).strip()
        if is_insecure_secret_value(advanced_encryption_seed):
            parser.error(
                "prod profile requires a strong advanced encryption seed via "
                "--advanced-encryption-seed or OPENHYDRA_ADVANCED_ENCRYPTION_SEED"
            )

    return {
        "deployment_profile": profile,
        "hydra_ledger_bridge_mock_mode": hydra_ledger_bridge_mock_mode,
        "advanced_encryption_seed": advanced_encryption_seed or str(getattr(args, "advanced_encryption_seed")),
    }


def _safe_free_memory() -> None:
    """Free Python objects and MLX Metal buffer pool.

    Called during mode transitions to prevent OOM on constrained
    hardware (8 GB M1).  Safe to call even if MLX is not installed.
    """
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass


class OpenHydraHandler(BaseHTTPRequestHandler):
    engine: CoordinatorEngine | None = None
    local_engine: Any = None  # LocalInferenceEngine when in local mode
    _api_key: str | None = None           # None → auth disabled
    _rate_limiter: _RateLimiter | None = None
    _mode_switching: bool = False          # 503 drain gate during mode transition
    _mode_switch_lock = threading.Lock()   # Serialize concurrent mode switches
    _metrics_lock = threading.Lock()
    _http_requests_total: int = 0
    _http_request_errors_total: int = 0
    _http_request_latency_seconds_sum: float = 0.0
    _http_request_latency_seconds_count: int = 0

    def log_message(self, fmt: str, *args: Any) -> None:  # type: ignore[override]
        """Route BaseHTTPRequestHandler access logs through the stdlib logger."""
        logger.info("http %s - %s", self.address_string(), fmt % args)

    def _check_auth(self) -> bool:
        """Return True if the request is authorised (or auth is disabled)."""
        key = self.__class__._api_key
        if not key:
            return True
        auth = str(self.headers.get("Authorization", "")).strip()
        if auth.startswith("Bearer "):
            return auth[7:].strip() == key
        x_key = str(self.headers.get("X-API-Key", "")).strip()
        return x_key == key

    def _check_rate_limit(self) -> bool:
        """Backward-compatible wrapper; prefer _rate_limit_check() for header data."""
        return self._rate_limit_check()[0]

    def _rate_limit_check(self) -> tuple[bool, dict[str, str]]:
        """Return (allowed, rate_limit_headers).

        Headers follow the de-facto standard understood by Cloudflare and most
        API clients:
          X-RateLimit-Limit     — max requests per window
          X-RateLimit-Remaining — requests left in the current window
          X-RateLimit-Reset     — Unix epoch when the oldest window entry expires
          Retry-After           — seconds until retry (only on 429 responses)
        """
        rl = self.__class__._rate_limiter
        if rl is None:
            return True, {}
        client_ip = str(self.client_address[0])
        allowed, remaining, reset_unix = rl.check(client_ip)
        headers: dict[str, str] = {
            "X-RateLimit-Limit": str(rl._max),
            "X-RateLimit-Remaining": str(max(0, remaining)),
            "X-RateLimit-Reset": str(reset_unix),
        }
        if not allowed:
            headers["Retry-After"] = str(max(1, reset_unix - int(time.time())))
        return allowed, headers

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK, headers: dict[str, str] | None = None) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)
        self._last_response_status = int(status)

    def _send_text(
        self,
        payload: str,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)
        self._last_response_status = int(status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _require_engine(self) -> CoordinatorEngine:
        if self.engine is None:
            raise RuntimeError("engine is not initialized")
        return self.engine

    # ------------------------------------------------------------------
    # Hybrid Local/Swarm routing (Pillar 2)
    # ------------------------------------------------------------------

    def _active_engine(self) -> Any:
        """Return the currently active engine (local takes priority).

        In Local Mode, returns the LocalInferenceEngine.
        In Swarm Mode, returns the CoordinatorEngine.
        Raises RuntimeError if neither is available.
        """
        le = self.__class__.local_engine
        if le is not None:
            return le
        if self.__class__.engine is not None:
            return self.__class__.engine
        raise RuntimeError("no engine available (neither local nor swarm)")

    def _is_local_mode(self) -> bool:
        """Return True if the local inference engine is active."""
        return self.__class__.local_engine is not None

    def _handle_chat_completions(self, body: dict[str, Any] | None = None) -> None:
        """Handle POST /v1/chat/completions for both local and swarm engines.

        Args:
            body: Pre-parsed JSON body from do_POST.  If None, reads from stream.
        """
        # 503 drain gate: reject during mode transition
        if self.__class__._mode_switching:
            self._send_json(
                {"error": "mode switching in progress — retry shortly"},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                headers={"Retry-After": "5"},
            )
            return
        if body is None:
            body = self._read_json()
        stream = bool(body.get("stream", False))
        messages = list(body.get("messages") or [])
        max_tokens = int(body.get("max_tokens", 128))
        temperature = float(body.get("temperature", 1.0))
        top_p = float(body.get("top_p", 1.0))
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]

        if self._is_local_mode():
            le = self.__class__.local_engine
            if stream:
                chunks = le.chat_stream(
                    messages, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, stop=stop,
                )
                self._send_local_sse(chunks)
            else:
                result = le.chat(
                    messages, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, stop=stop,
                )
                self._send_json(result)
        else:
            # Swarm mode — delegate to existing CoordinatorEngine path
            engine = self._require_engine()
            request_id = str(uuid.uuid4())
            if stream:
                payload = self._chat_stream_payload(body, request_id)
                model_meta = payload.get("model", {})
                served_model = str(model_meta.get("served", body.get("model", "")))
                self._send_sse(
                    request_id=payload["request_id"],
                    model_id=served_model,
                    chunks=payload["stream"],
                )
            else:
                payload = self._chat_payload(body, request_id)
                model_meta = payload.get("model", {})
                served_model = str(model_meta.get("served", body.get("model", "")))
                self._send_json({
                    "id": payload["request_id"],
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": served_model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": payload["response"]},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": len(payload["response"].split()),
                        "total_tokens": len(payload["response"].split()),
                    },
                    "openhydra": payload,
                })

    def _send_local_sse(self, chunks) -> None:
        """Stream LocalInferenceEngine chunks as SSE events."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_chunk = None
        for chunk in chunks:
            last_chunk = chunk
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self._last_response_status = int(HTTPStatus.OK)

    def _handle_list_models(self) -> None:
        """Handle GET /v1/models for both local and swarm engines."""
        if self._is_local_mode():
            models = self.__class__.local_engine.list_models()
            # Wrap in OpenAI format if not already
            if isinstance(models, list):
                self._send_json({"object": "list", "data": models})
            else:
                self._send_json(models)
        else:
            engine = self._require_engine()
            payload = engine.list_models()
            self._send_json(payload)

    def _handle_tags_ollama(self) -> None:
        """Handle GET /api/tags for both engines in Ollama format."""
        if self._is_local_mode():
            models = self.__class__.local_engine.list_models()
            if isinstance(models, list):
                model_list = models
            else:
                model_list = models.get("data", [])
        else:
            engine = self._require_engine()
            payload = engine.list_models()
            model_list = payload.get("data", [])

        ollama_models = [
            {
                "name": str(m.get("id", "")),
                "model": str(m.get("id", "")),
                "modified_at": "2025-01-01T00:00:00Z",
                "size": 0,
                "digest": "",
                "details": {
                    "format": "",
                    "family": "",
                    "families": None,
                    "parameter_size": "",
                    "quantization_level": str(m.get("meta", {}).get("quantization", "")),
                },
            }
            for m in model_list
        ]
        self._send_json({"models": ollama_models})

    def _handle_show_ollama(self) -> None:
        """Handle POST /api/show — return model configuration details."""
        body = self._read_json()
        model_name = str(body.get("name", ""))

        if self._is_local_mode():
            models = self.__class__.local_engine.list_models()
            model_list = models if isinstance(models, list) else models.get("data", [])
        else:
            engine = self._require_engine()
            model_list = engine.list_models().get("data", [])

        model_info = {}
        for m in model_list:
            if m.get("id") == model_name:
                model_info = m
                break

        self._send_json({
            "modelfile": f"FROM {model_name}",
            "modelinfo": {
                "general.architecture": model_info.get("meta", {}).get("backend", "transformer"),
                "general.name": model_name,
                "general.quantization": model_info.get("meta", {}).get("quantization", "fp32"),
            },
            "parameters": "",
            "template": "",
        })

    def _handle_ps_ollama(self) -> None:
        """Handle GET /api/ps — return currently loaded models."""
        if self._is_local_mode():
            models = self.__class__.local_engine.list_models()
            model_list = models if isinstance(models, list) else models.get("data", [])
        else:
            engine = self._require_engine()
            model_list = engine.list_models().get("data", [])

        running = [
            {
                "name": str(m.get("id", "")),
                "model": str(m.get("id", "")),
                "size": 0,
                "digest": "",
                "expires_at": "2099-01-01T00:00:00Z",
            }
            for m in model_list
        ]
        self._send_json({"models": running})

    def _handle_generate_ollama(self, stream: bool = False) -> None:
        """Handle POST /api/generate for both engines in Ollama format."""
        body = self._read_json()
        prompt = str(body.get("prompt", ""))
        model = str(body.get("model", ""))
        opts = _parse_ollama_options(dict(body.get("options") or {}))
        max_tokens = int(opts.pop("max_tokens", 24))

        if self._is_local_mode():
            le = self.__class__.local_engine
            result = le.infer(prompt, max_tokens=max_tokens, **opts)
            content = result["choices"][0]["message"]["content"]
            created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            self._send_json({
                "model": le.model_id,
                "created_at": created_at,
                "response": content,
                "done": True,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": result["usage"]["prompt_tokens"],
                "eval_count": result["usage"]["completion_tokens"],
                "eval_duration": 0,
            })
        else:
            engine = self._require_engine()
            request_id = str(uuid.uuid4())
            rid_headers: dict[str, str] = {}
            self._generate_ollama(
                body=body, request_id=request_id,
                rid_headers=rid_headers, stream=stream,
            )

    def _handle_chat_ollama(self, stream: bool = False) -> None:
        """Handle POST /api/chat for both engines in Ollama format."""
        body = self._read_json()
        messages = list(body.get("messages") or [])
        model = str(body.get("model", ""))
        opts = _parse_ollama_options(dict(body.get("options") or {}))
        max_tokens = int(opts.pop("max_tokens", 24))

        if self._is_local_mode():
            le = self.__class__.local_engine
            result = le.chat(messages, max_tokens=max_tokens, **opts)
            content = result["choices"][0]["message"]["content"]
            created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            self._send_json({
                "model": le.model_id,
                "created_at": created_at,
                "message": {"role": "assistant", "content": content},
                "done": True,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": result["usage"]["prompt_tokens"],
                "eval_count": result["usage"]["completion_tokens"],
                "eval_duration": 0,
            })
        else:
            engine = self._require_engine()
            request_id = str(uuid.uuid4())
            rid_headers: dict[str, str] = {}
            self._chat_ollama(
                body=body, request_id=request_id,
                rid_headers=rid_headers, stream=stream,
            )

    # ------------------------------------------------------------------
    # Mode switch (Pillar 3: The Toggle)
    # ------------------------------------------------------------------

    def _handle_mode_switch(self, body: dict[str, Any] | None = None) -> None:
        """Handle POST /v1/internal/mode — switch between local and swarm.

        Only accepts requests from 127.0.0.1 (localhost).
        Sets the 503 drain gate during transition.

        Args:
            body: Pre-parsed JSON body (from do_POST router).  If None,
                reads from the request stream (for direct calls in tests).
        """
        # Security: only accept from localhost
        client_ip = str(self.client_address[0])
        if client_ip not in ("127.0.0.1", "::1"):
            self._send_json(
                {"error": "mode switch only allowed from localhost"},
                status=HTTPStatus.FORBIDDEN,
            )
            return

        if body is None:
            body = self._read_json()
        mode = str(body.get("mode", "")).strip().lower()

        if mode not in ("local", "swarm"):
            self._send_json(
                {"error": f"invalid mode '{mode}'; must be 'local' or 'swarm'"},
                status=HTTPStatus(400),
            )
            return

        model_id = str(body.get("model_id", "")).strip() or None
        cls = self.__class__

        # Noop if already in the requested mode
        if mode == "local" and cls.local_engine is not None:
            self._send_json({"status": "ok", "mode": "local",
                             "model_id": cls.local_engine.model_id})
            return
        if mode == "swarm" and cls.local_engine is None:
            swarm_model = ""
            if cls.engine and hasattr(cls.engine, "config"):
                swarm_model = str(getattr(cls.engine.config, "default_model", ""))
            self._send_json({"status": "ok", "mode": "swarm",
                             "model_id": swarm_model})
            return

        # Serialize concurrent switches
        with cls._mode_switch_lock:
            cls._mode_switching = True
            try:
                if mode == "local":
                    self._switch_to_local(model_id=model_id)
                else:
                    self._switch_to_swarm(model_id=model_id)
            finally:
                cls._mode_switching = False

        active_model_id = ""
        if cls.local_engine is not None:
            active_model_id = cls.local_engine.model_id
        elif cls.engine and hasattr(cls.engine, "config"):
            active_model_id = str(getattr(cls.engine.config, "default_model", ""))
        self._send_json({"status": "ok", "mode": mode, "model_id": active_model_id})

    def _switch_to_local(self, model_id: str | None = None) -> None:
        """Transition from swarm to local mode.

        Memory-safe sequence:
        1. Determine target model ID (explicit or hardware default)
        2. Free memory (gc.collect + Metal cache clear)
        3. Create new ModelShard + LocalInferenceEngine

        Args:
            model_id: Explicit model to load.  Falls back to the
                coordinator's default_model if not provided.
        """
        from coordinator.local_engine import LocalInferenceEngine
        from peer.model_shard import ModelShard, ToyShardConfig

        cls = self.__class__

        # 1. Resolve target model
        if not model_id:
            model_id = "openhydra-toy-345m"
            if cls.engine and hasattr(cls.engine, "config"):
                model_id = str(getattr(cls.engine.config, "default_model", model_id))

        # 2. Free memory before allocating new model
        _safe_free_memory()

        # 3. Detect best runtime backend for this platform
        import platform as _plat
        _backend = "toy_auto"
        _quant = "fp32"
        _runtime_model = model_id
        _MLX_4BIT_MAP = {
            "Qwen/Qwen3.5-0.8B": "mlx-community/Qwen3.5-0.8B-4bit",
            "openhydra-qwen3.5-0.8b": "mlx-community/Qwen3.5-0.8B-4bit",
        }
        try:
            if _plat.system() == "Darwin" and model_id in _MLX_4BIT_MAP:
                import mlx.core  # noqa: F401 — probe availability
                _backend = "mlx"
                _runtime_model = _MLX_4BIT_MAP[model_id]
                _quant = "fp32"  # Already quantized, don't re-quantize
                logger.info("local_engine: MLX backend, model=%s", _runtime_model)
        except ImportError:
            pass

        # 4. Create new shard + engine
        shard = ModelShard(ToyShardConfig(
            model_id=model_id,
            runtime_backend=_backend,
            runtime_model_id=_runtime_model,
            quantization_mode=_quant,
        ))
        cls.local_engine = LocalInferenceEngine(
            model_id=model_id,
            shard=shard,
        )

    def _switch_to_swarm(self, model_id: str | None = None) -> None:
        """Transition from local to swarm mode.

        Memory-safe sequence:
        1. Call unload() on local engine (releases shard + runtime)
        2. Set local_engine to None (drop Python reference)
        3. gc.collect() + Metal cache clear

        Args:
            model_id: Reserved for future use (target swarm model).
        """
        cls = self.__class__
        if cls.local_engine is not None:
            unload = getattr(cls.local_engine, "unload", None)
            if callable(unload):
                unload()
            cls.local_engine = None

        # Free memory after unloading
        _safe_free_memory()

    def _handle_mode_status(self) -> None:
        """Handle GET /v1/internal/mode — return current mode."""
        mode = "local" if self.__class__.local_engine is not None else "swarm"
        self._send_json({
            "mode": mode,
            "switching": bool(self.__class__._mode_switching),
        })

    @classmethod
    def _record_http_request_metrics(
        cls,
        *,
        status_code: int,
        latency_seconds: float,
    ) -> None:
        with cls._metrics_lock:
            cls._http_requests_total += 1
            cls._http_request_latency_seconds_sum += max(0.0, float(latency_seconds))
            cls._http_request_latency_seconds_count += 1
            if int(status_code) >= 400:
                cls._http_request_errors_total += 1

    @classmethod
    def _http_metrics_snapshot(cls) -> dict[str, float | int]:
        with cls._metrics_lock:
            count = int(cls._http_request_latency_seconds_count)
            total = float(cls._http_request_latency_seconds_sum)
            avg = (total / float(count)) if count else 0.0
            return {
                "http_requests_total": int(cls._http_requests_total),
                "http_request_errors_total": int(cls._http_request_errors_total),
                "http_request_latency_seconds_sum": total,
                "http_request_latency_seconds_count": count,
                "http_request_latency_seconds_avg": avg,
            }

    def _render_metrics(self) -> str:
        api = self._http_metrics_snapshot()
        engine_metrics: dict[str, float | int] = {}
        if self.engine is not None:
            try:
                engine_metrics = dict(self.engine.metrics_snapshot())
            except Exception:
                engine_metrics = {}

        lines = [
            "# HELP openhydra_http_requests_total Total HTTP API requests.",
            "# TYPE openhydra_http_requests_total counter",
            f"openhydra_http_requests_total {int(api.get('http_requests_total', 0))}",
            "# HELP openhydra_http_request_errors_total Total HTTP API requests returning >=400.",
            "# TYPE openhydra_http_request_errors_total counter",
            f"openhydra_http_request_errors_total {int(api.get('http_request_errors_total', 0))}",
            "# HELP openhydra_http_request_latency_seconds_sum Sum of HTTP request latencies in seconds.",
            "# TYPE openhydra_http_request_latency_seconds_sum counter",
            f"openhydra_http_request_latency_seconds_sum {float(api.get('http_request_latency_seconds_sum', 0.0))}",
            "# HELP openhydra_http_request_latency_seconds_count Count of observed HTTP request latencies.",
            "# TYPE openhydra_http_request_latency_seconds_count counter",
            f"openhydra_http_request_latency_seconds_count {int(api.get('http_request_latency_seconds_count', 0))}",
            "# HELP openhydra_http_request_latency_seconds_avg Average HTTP request latency in seconds.",
            "# TYPE openhydra_http_request_latency_seconds_avg gauge",
            f"openhydra_http_request_latency_seconds_avg {float(api.get('http_request_latency_seconds_avg', 0.0))}",
            "# HELP openhydra_dht_lookup_attempts_total Total DHT lookup attempts.",
            "# TYPE openhydra_dht_lookup_attempts_total counter",
            f"openhydra_dht_lookup_attempts_total {int(engine_metrics.get('dht_lookup_attempts', 0))}",
            "# HELP openhydra_dht_lookup_successes_total Total successful DHT lookups.",
            "# TYPE openhydra_dht_lookup_successes_total counter",
            f"openhydra_dht_lookup_successes_total {int(engine_metrics.get('dht_lookup_successes', 0))}",
            "# HELP openhydra_dht_lookup_failures_total Total failed DHT lookups.",
            "# TYPE openhydra_dht_lookup_failures_total counter",
            f"openhydra_dht_lookup_failures_total {int(engine_metrics.get('dht_lookup_failures', 0))}",
            "# HELP openhydra_dht_lookup_success_rate DHT lookup success ratio.",
            "# TYPE openhydra_dht_lookup_success_rate gauge",
            f"openhydra_dht_lookup_success_rate {float(engine_metrics.get('dht_lookup_success_rate', 0.0))}",
            "# HELP openhydra_hydra_bridge_total_burned_total Total HYDRA burned for compute settlement.",
            "# TYPE openhydra_hydra_bridge_total_burned_total counter",
            f"openhydra_hydra_bridge_total_burned_total {float(engine_metrics.get('hydra_bridge_total_burned', 0.0))}",
            "# HELP openhydra_hydra_bridge_total_minted_total Total HYDRA minted for provider rewards.",
            "# TYPE openhydra_hydra_bridge_total_minted_total counter",
            f"openhydra_hydra_bridge_total_minted_total {float(engine_metrics.get('hydra_bridge_total_minted', 0.0))}",
            "# HELP openhydra_hydra_bridge_total_supply Current HYDRA bridge supply.",
            "# TYPE openhydra_hydra_bridge_total_supply gauge",
            f"openhydra_hydra_bridge_total_supply {float(engine_metrics.get('hydra_bridge_total_supply', 0.0))}",
            "# HELP openhydra_hydra_bridge_supply_cap HYDRA bridge supply cap.",
            "# TYPE openhydra_hydra_bridge_supply_cap gauge",
            f"openhydra_hydra_bridge_supply_cap {float(engine_metrics.get('hydra_bridge_supply_cap', 0.0))}",
            # Phase D: KV compaction + inference SLO counters
            "# HELP openhydra_kv_store_ops_total KV cache write ops issued to peers.",
            "# TYPE openhydra_kv_store_ops_total counter",
            f"openhydra_kv_store_ops_total {int(engine_metrics.get('kv_store_ops_total', 0))}",
            "# HELP openhydra_kv_retrieve_ops_total KV cache read ops issued to peers.",
            "# TYPE openhydra_kv_retrieve_ops_total counter",
            f"openhydra_kv_retrieve_ops_total {int(engine_metrics.get('kv_retrieve_ops_total', 0))}",
            "# HELP openhydra_inference_requests_total Total inference requests dispatched.",
            "# TYPE openhydra_inference_requests_total counter",
            f"openhydra_inference_requests_total {int(engine_metrics.get('inference_requests_total', 0))}",
            # Pass 6: KV compaction SLO metrics
            "# HELP openhydra_compact_tokens_saved_total Lifetime tokens pruned by KV compaction.",
            "# TYPE openhydra_compact_tokens_saved_total counter",
            f"openhydra_compact_tokens_saved_total {int(engine_metrics.get('compact_tokens_saved_total', 0))}",
            "# HELP openhydra_compact_latency_ms_total Lifetime compaction overhead in milliseconds.",
            "# TYPE openhydra_compact_latency_ms_total counter",
            f"openhydra_compact_latency_ms_total {round(float(engine_metrics.get('compact_latency_total_ms', 0.0)), 1)}",
        ]
        return "\n".join(lines) + "\n"

    def _send_metrics(self) -> None:
        self._send_text(
            self._render_metrics(),
            content_type="text/plain; version=0.0.4; charset=utf-8",
            status=HTTPStatus.OK,
        )

    def do_GET(self) -> None:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        parsed = urlparse(self.path)
        self._last_response_status = int(HTTPStatus.INTERNAL_SERVER_ERROR)
        rid_headers: dict[str, str] = {"X-Request-ID": request_id}
        logger.info("request_start req_id=%s method=GET path=%s client=%s", request_id, parsed.path, self.client_address[0])
        try:
            # /metrics, /health, and /readyz are always public (no auth required).
            if parsed.path in {"/metrics", "/health"}:
                self._send_metrics()
                return

            if parsed.path == "/readyz":
                # Ready as soon as the engine has been wired in.
                if self.engine is not None:
                    self._send_json({"status": "ok"}, headers=rid_headers)
                else:
                    self._send_json(
                        {"status": "not_ready", "reason": "engine_not_initialized"},
                        status=HTTPStatus.SERVICE_UNAVAILABLE,
                        headers=rid_headers,
                    )
                return

            if not self._check_auth():
                logger.warning("auth_failed req_id=%s GET %s from %s", request_id, parsed.path, self.client_address[0])
                self._send_json({"error": "unauthorized"}, status=HTTPStatus.UNAUTHORIZED, headers=rid_headers)
                return

            _rl_allowed, _rl_headers = self._rate_limit_check()
            rid_headers.update(_rl_headers)
            if not _rl_allowed:
                logger.warning("rate_limited req_id=%s GET %s from %s", request_id, parsed.path, self.client_address[0])
                self._send_json({"error": "rate_limit_exceeded"}, status=HTTPStatus.TOO_MANY_REQUESTS, headers=rid_headers)
                return

            engine = self._require_engine()
            if parsed.path == "/v1/models":
                self._send_json(engine.list_models(), headers=rid_headers)
                return

            if parsed.path == "/v1/network/status":
                self._send_json(engine.network_status(), headers=rid_headers)
                return

            if parsed.path == "/v1/account/balance":
                qs = parse_qs(parsed.query)
                client_id = qs.get("client_id", ["anonymous"])[0]
                self._send_json(engine.account_balance(client_id=client_id), headers=rid_headers)
                return

            if parsed.path == "/v1/hydra/status":
                self._send_json(engine.hydra_status(), headers=rid_headers)
                return

            if parsed.path == "/v1/hydra/account":
                qs = parse_qs(parsed.query)
                client_id = qs.get("client_id", ["anonymous"])[0]
                self._send_json(engine.hydra_account(client_id=client_id), headers=rid_headers)
                return

            if parsed.path == "/v1/hydra/governance/params":
                self._send_json(engine.hydra_governance_params(), headers=rid_headers)
                return

            # Ollama-compatible endpoint: list models in Ollama /api/tags format.
            if parsed.path == "/api/tags":
                self._tags_ollama(rid_headers=rid_headers)
                return

            # Internal mode status endpoint.
            if parsed.path == "/v1/internal/mode":
                self._handle_mode_status()
                return

            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND, headers=rid_headers)
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_GATEWAY, headers=rid_headers)
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            status = int(getattr(self, "_last_response_status", int(HTTPStatus.INTERNAL_SERVER_ERROR)))
            logger.info("request_done req_id=%s method=GET path=%s status=%d latency_ms=%.1f", request_id, parsed.path, status, latency_ms)
            self._record_http_request_metrics(
                status_code=status,
                latency_seconds=latency_ms / 1000.0,
            )

    def _chat_payload(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        engine = self._require_engine()
        messages = body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]  # compat
        decode_kwargs = _parse_decode_options(body)

        return engine.infer_chat(
            messages=messages,
            max_tokens=int(body.get("max_tokens", 24)),
            pipeline_width=(int(body["pipeline_width"]) if body.get("pipeline_width") is not None else None),
            grounding=bool(body.get("grounding", True)),
            priority=bool(body.get("priority", False)),
            client_id=str(body.get("client_id", "anonymous")),
            model_id=str(body.get("model", engine.config.default_model)),
            allow_degradation=bool(body.get("allow_degradation", True)),
            session_id=(str(body.get("session_id")) if body.get("session_id") is not None else None),
            expert_tags=_parse_expert_tags(body.get("expert_tags")),
            expert_layer_indices=_parse_expert_layers(body.get("expert_layers", body.get("expert_layer_indices"))),
            request_id=request_id,
            **decode_kwargs,
        )

    def _completion_payload(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        engine = self._require_engine()
        decode_kwargs = _parse_decode_options(body)
        return engine.infer(
            prompt=str(body.get("prompt", "")),
            max_tokens=int(body.get("max_tokens", 24)),
            pipeline_width=(int(body["pipeline_width"]) if body.get("pipeline_width") is not None else None),
            grounding=bool(body.get("grounding", True)),
            priority=bool(body.get("priority", False)),
            client_id=str(body.get("client_id", "anonymous")),
            model_id=str(body.get("model", engine.config.default_model)),
            allow_degradation=bool(body.get("allow_degradation", True)),
            session_id=(str(body.get("session_id")) if body.get("session_id") is not None else None),
            expert_tags=_parse_expert_tags(body.get("expert_tags")),
            expert_layer_indices=_parse_expert_layers(body.get("expert_layers", body.get("expert_layer_indices"))),
            request_id=request_id,
            **decode_kwargs,
        )

    def _chat_stream_payload(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        engine = self._require_engine()
        messages = body.get("messages") or [{"role": "user", "content": body.get("prompt", "")}]  # compat
        decode_kwargs = _parse_decode_options(body)
        return engine.infer_chat_stream(
            messages=messages,
            max_tokens=int(body.get("max_tokens", 24)),
            pipeline_width=(int(body["pipeline_width"]) if body.get("pipeline_width") is not None else None),
            grounding=bool(body.get("grounding", True)),
            priority=bool(body.get("priority", False)),
            client_id=str(body.get("client_id", "anonymous")),
            model_id=str(body.get("model", engine.config.default_model)),
            allow_degradation=bool(body.get("allow_degradation", True)),
            session_id=(str(body.get("session_id")) if body.get("session_id") is not None else None),
            expert_tags=_parse_expert_tags(body.get("expert_tags")),
            expert_layer_indices=_parse_expert_layers(body.get("expert_layers", body.get("expert_layer_indices"))),
            request_id=request_id,
            **decode_kwargs,
        )

    def _completion_stream_payload(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        engine = self._require_engine()
        decode_kwargs = _parse_decode_options(body)
        return engine.infer_stream(
            prompt=str(body.get("prompt", "")),
            max_tokens=int(body.get("max_tokens", 24)),
            pipeline_width=(int(body["pipeline_width"]) if body.get("pipeline_width") is not None else None),
            grounding=bool(body.get("grounding", True)),
            priority=bool(body.get("priority", False)),
            client_id=str(body.get("client_id", "anonymous")),
            model_id=str(body.get("model", engine.config.default_model)),
            allow_degradation=bool(body.get("allow_degradation", True)),
            session_id=(str(body.get("session_id")) if body.get("session_id") is not None else None),
            expert_tags=_parse_expert_tags(body.get("expert_tags")),
            expert_layer_indices=_parse_expert_layers(body.get("expert_layers", body.get("expert_layer_indices"))),
            request_id=request_id,
            **decode_kwargs,
        )

    def _send_sse(self, request_id: str, model_id: str, chunks, headers: dict[str, str] | None = None) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()

        for text_chunk in chunks:
            if not text_chunk:
                continue
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": str(text_chunk)}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

        done_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(done_chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self._last_response_status = int(HTTPStatus.OK)

    # ------------------------------------------------------------------
    # Ollama payload helpers (called from do_GET / do_POST + testable directly)
    # ------------------------------------------------------------------

    def _tags_ollama(self, *, rid_headers: dict[str, str]) -> None:
        """Serve ``GET /api/tags`` in Ollama format."""
        engine = self._require_engine()
        models_payload = engine.list_models()
        ollama_models = [
            {
                "name": str(m.get("id", "")),
                "model": str(m.get("id", "")),
                "modified_at": "2025-01-01T00:00:00Z",
                "size": 0,
                "digest": "",
                "details": {
                    "format": "",
                    "family": "",
                    "families": None,
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }
            for m in models_payload.get("data", [])
        ]
        self._send_json({"models": ollama_models}, headers=rid_headers)

    def _generate_ollama(
        self,
        *,
        body: dict[str, Any],
        request_id: str,
        rid_headers: dict[str, str],
        stream: bool,
    ) -> None:
        """Serve ``POST /api/generate`` in Ollama format."""
        engine = self._require_engine()
        opts = _parse_ollama_options(dict(body.get("options") or {}))
        max_tokens = int(opts.pop("max_tokens", 24))
        prompt = str(body.get("prompt", ""))
        model = str(body.get("model", engine.config.default_model))
        client_id = str(body.get("client_id", "anonymous"))

        if stream:
            payload = engine.infer_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                model_id=model,
                client_id=client_id,
                request_id=request_id,
                **opts,
            )
            served_model = str(payload.get("model", {}).get("served", model))
            self._send_ollama_stream(
                model_id=served_model,
                chunks=payload["stream"],
                is_chat=False,
                headers=rid_headers,
            )
        else:
            payload = engine.infer(
                prompt=prompt,
                max_tokens=max_tokens,
                model_id=model,
                client_id=client_id,
                request_id=request_id,
                **opts,
            )
            served_model = str(payload.get("model", {}).get("served", model))
            created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            self._send_json(
                {
                    "model": served_model,
                    "created_at": created_at,
                    "response": payload["response"],
                    "done": True,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                    "eval_count": len(payload["response"].split()),
                    "eval_duration": 0,
                },
                headers=rid_headers,
            )

    def _chat_ollama(
        self,
        *,
        body: dict[str, Any],
        request_id: str,
        rid_headers: dict[str, str],
        stream: bool,
    ) -> None:
        """Serve ``POST /api/chat`` in Ollama format."""
        engine = self._require_engine()
        opts = _parse_ollama_options(dict(body.get("options") or {}))
        max_tokens = int(opts.pop("max_tokens", 24))
        messages = list(body.get("messages") or [])
        model = str(body.get("model", engine.config.default_model))
        client_id = str(body.get("client_id", "anonymous"))

        if stream:
            payload = engine.infer_chat_stream(
                messages=messages,
                max_tokens=max_tokens,
                model_id=model,
                client_id=client_id,
                request_id=request_id,
                **opts,
            )
            served_model = str(payload.get("model", {}).get("served", model))
            self._send_ollama_stream(
                model_id=served_model,
                chunks=payload["stream"],
                is_chat=True,
                headers=rid_headers,
            )
        else:
            payload = engine.infer_chat(
                messages=messages,
                max_tokens=max_tokens,
                model_id=model,
                client_id=client_id,
                request_id=request_id,
                **opts,
            )
            served_model = str(payload.get("model", {}).get("served", model))
            created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            self._send_json(
                {
                    "model": served_model,
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": payload["response"]},
                    "done": True,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_count": 0,
                    "eval_count": len(payload["response"].split()),
                    "eval_duration": 0,
                },
                headers=rid_headers,
            )

    def _send_ollama_stream(
        self,
        model_id: str,
        chunks: Any,
        is_chat: bool,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Stream Ollama-format NDJSON (newline-delimited JSON, not SSE).

        Each intermediate line: ``{"model": ..., "response": token, "done": false}``
        Final line:             ``{"model": ..., "response": "", "done": true, ...}``
        For chat (``is_chat=True``) the payload key is ``message`` instead of ``response``.
        """
        created_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()

        for text_chunk in chunks:
            if not text_chunk:
                continue
            if is_chat:
                obj: dict[str, Any] = {
                    "model": model_id,
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": str(text_chunk)},
                    "done": False,
                }
            else:
                obj = {
                    "model": model_id,
                    "created_at": created_at,
                    "response": str(text_chunk),
                    "done": False,
                }
            self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
            self.wfile.flush()

        # Final "done" line.
        if is_chat:
            done_obj: dict[str, Any] = {
                "model": model_id,
                "created_at": created_at,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "eval_duration": 0,
            }
        else:
            done_obj = {
                "model": model_id,
                "created_at": created_at,
                "response": "",
                "done": True,
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "eval_count": 0,
                "eval_duration": 0,
            }
        self.wfile.write((json.dumps(done_obj) + "\n").encode("utf-8"))
        self.wfile.flush()
        self._last_response_status = int(HTTPStatus.OK)

    def do_POST(self) -> None:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        parsed = urlparse(self.path)
        self._last_response_status = int(HTTPStatus.INTERNAL_SERVER_ERROR)
        rid_headers: dict[str, str] = {"X-Request-ID": request_id}
        logger.info("request_start req_id=%s method=POST path=%s client=%s", request_id, parsed.path, self.client_address[0])
        try:
            if not self._check_auth():
                logger.warning("auth_failed req_id=%s POST %s from %s", request_id, parsed.path, self.client_address[0])
                self._send_json({"error": "unauthorized"}, status=HTTPStatus.UNAUTHORIZED, headers=rid_headers)
                return

            _rl_allowed, _rl_headers = self._rate_limit_check()
            rid_headers.update(_rl_headers)
            if not _rl_allowed:
                logger.warning("rate_limited req_id=%s POST %s from %s", request_id, parsed.path, self.client_address[0])
                self._send_json({"error": "rate_limit_exceeded"}, status=HTTPStatus.TOO_MANY_REQUESTS, headers=rid_headers)
                return

            body = self._read_json()
            engine = self._require_engine()
            requested_model = str(body.get("model", engine.config.default_model))
            stream = bool(body.get("stream", False))

            # Validate inference params for inference endpoints
            if parsed.path in {"/v1/chat/completions", "/v1/completions"}:
                err = _validate_infer_params(body)
                if err:
                    logger.warning("invalid_params req_id=%s POST %s: %s", request_id, parsed.path, err)
                    self._send_json({"error": err}, status=HTTPStatus.UNPROCESSABLE_ENTITY, headers=rid_headers)
                    return

            if parsed.path == "/v1/chat/completions":
                # Hybrid routing: local engine bypasses the full coordinator stack
                if self._is_local_mode():
                    self._handle_chat_completions(body=body)
                    return

                payload = self._chat_stream_payload(body, request_id) if stream else self._chat_payload(body, request_id)
                model_meta = payload.get("model", {})
                served_model = str(model_meta.get("served", requested_model))
                headers = {
                    **rid_headers,
                    "X-OpenHydra-Requested-Model": str(model_meta.get("requested", requested_model)),
                    "X-OpenHydra-Served-Model": served_model,
                    "X-OpenHydra-Degradation-Reason": str(model_meta.get("reason", "none")),
                    "X-OpenHydra-Degradation-Detail": str(model_meta.get("detail", "")),
                }
                if stream:
                    self._send_sse(
                        request_id=payload["request_id"],
                        model_id=served_model,
                        chunks=payload["stream"],
                        headers=headers,
                    )
                    return
                self._send_json(
                    {
                        "id": payload["request_id"],
                        "object": "chat.completion",
                        "model": served_model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": payload["response"]},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": len(payload["response"].split()),
                            "total_tokens": len(payload["response"].split()),
                        },
                        "openhydra": payload,
                    },
                    headers=headers,
                )
                return

            if parsed.path == "/v1/completions":
                payload = self._completion_stream_payload(body, request_id) if stream else self._completion_payload(body, request_id)
                model_meta = payload.get("model", {})
                served_model = str(model_meta.get("served", requested_model))
                headers = {
                    **rid_headers,
                    "X-OpenHydra-Requested-Model": str(model_meta.get("requested", requested_model)),
                    "X-OpenHydra-Served-Model": served_model,
                    "X-OpenHydra-Degradation-Reason": str(model_meta.get("reason", "none")),
                    "X-OpenHydra-Degradation-Detail": str(model_meta.get("detail", "")),
                }
                if stream:
                    self._send_sse(
                        request_id=payload["request_id"],
                        model_id=served_model,
                        chunks=payload["stream"],
                        headers=headers,
                    )
                    return
                self._send_json(
                    {
                        "id": payload["request_id"],
                        "object": "text_completion",
                        "model": served_model,
                        "choices": [
                            {
                                "index": 0,
                                "text": payload["response"],
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": len(payload["response"].split()),
                            "total_tokens": len(payload["response"].split()),
                        },
                        "openhydra": payload,
                    },
                    headers=headers,
                )
                return

            if parsed.path == "/v1/hydra/transfer":
                self._send_json(
                    engine.hydra_transfer(
                        from_client_id=str(body.get("from_client_id", "anonymous")),
                        to_client_id=str(body.get("to_client_id", "")),
                        amount=float(body.get("amount", 0.0)),
                    ),
                    headers=rid_headers,
                )
                return

            if parsed.path == "/v1/hydra/stake":
                self._send_json(
                    engine.hydra_stake(
                        client_id=str(body.get("client_id", "anonymous")),
                        amount=float(body.get("amount", 0.0)),
                    ),
                    headers=rid_headers,
                )
                return

            if parsed.path == "/v1/hydra/unstake":
                self._send_json(
                    engine.hydra_unstake(
                        client_id=str(body.get("client_id", "anonymous")),
                        amount=float(body.get("amount", 0.0)),
                    ),
                    headers=rid_headers,
                )
                return

            if parsed.path == "/v1/hydra/channels/open":
                channel_id = str(body.get("channel_id", "")).strip() or str(uuid.uuid4())
                self._send_json(
                    engine.hydra_open_channel(
                        channel_id=channel_id,
                        payer=str(body.get("payer", "anonymous")),
                        payee=str(body.get("payee", "")),
                        deposit=float(body.get("deposit", 0.0)),
                        ttl_seconds=(
                            int(body["ttl_seconds"])
                            if body.get("ttl_seconds") is not None
                            else None
                        ),
                    ),
                    headers=rid_headers,
                )
                return

            if parsed.path == "/v1/hydra/channels/charge":
                provider_peer_id = (
                    str(body.get("provider_peer_id"))
                    if body.get("provider_peer_id") is not None
                    else None
                )
                if provider_peer_id is None:
                    payload = engine.hydra_charge_channel(
                        channel_id=str(body.get("channel_id", "")),
                        amount=float(body.get("amount", 0.0)),
                    )
                else:
                    payload = engine.hydra_charge_channel(
                        channel_id=str(body.get("channel_id", "")),
                        amount=float(body.get("amount", 0.0)),
                        provider_peer_id=provider_peer_id,
                    )
                self._send_json(payload, headers=rid_headers)
                return

            if parsed.path == "/v1/hydra/channels/reconcile":
                self._send_json(
                    engine.hydra_reconcile_channel(
                        channel_id=str(body.get("channel_id", "")),
                        total_spent=float(body.get("total_spent", 0.0)),
                        nonce=int(body.get("nonce", 0)),
                    ),
                    headers=rid_headers,
                )
                return

            if parsed.path == "/v1/hydra/channels/close":
                self._send_json(
                    engine.hydra_close_channel(
                        channel_id=str(body.get("channel_id", "")),
                    ),
                    headers=rid_headers,
                )
                return

            if parsed.path == "/v1/hydra/governance/vote":
                self._send_json(
                    engine.hydra_governance_vote(
                        pubkey=str(body.get("pubkey", "anonymous")),
                        proposal_id=str(body.get("proposal_id", "")),
                        vote=str(body.get("vote", "")),
                    ),
                    headers=rid_headers,
                )
                return

            # ------------------------------------------------------------------
            # Ollama-compatible endpoints: /api/generate and /api/chat
            # ------------------------------------------------------------------

            if parsed.path == "/api/generate":
                self._generate_ollama(
                    body=body,
                    request_id=request_id,
                    rid_headers=rid_headers,
                    stream=stream,
                )
                return

            if parsed.path == "/api/chat":
                self._chat_ollama(
                    body=body,
                    request_id=request_id,
                    rid_headers=rid_headers,
                    stream=stream,
                )
                return

            if parsed.path == "/v1/internal/mode":
                self._handle_mode_switch(body=body)
                return

            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND, headers=rid_headers)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid_json"}, status=HTTPStatus.BAD_REQUEST, headers=rid_headers)
        except RuntimeError as exc:
            message = str(exc)
            err_payload = {"error": message}
            if message == "insufficient_priority_credits":
                err_status = HTTPStatus.PAYMENT_REQUIRED
            elif message.startswith("no_viable_model:"):
                err_status = HTTPStatus.SERVICE_UNAVAILABLE
                err_payload = {
                    "error": "no_viable_model",
                    "reason": message.split(":", 1)[1] or "unknown",
                }
            elif message.startswith("unknown_model:"):
                err_status = HTTPStatus.BAD_REQUEST
            elif message.startswith("insufficient_peers:") or message.startswith("no_viable_fallback:"):
                err_status = HTTPStatus.SERVICE_UNAVAILABLE
            elif message.startswith("hydra_") or message.startswith("hydra_bridge_"):
                err_status = HTTPStatus.BAD_REQUEST
            else:
                err_status = HTTPStatus.BAD_GATEWAY
            logger.warning("runtime_error req_id=%s: %s", request_id, message)
            self._send_json(err_payload, status=err_status, headers=rid_headers)
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            status = int(getattr(self, "_last_response_status", int(HTTPStatus.INTERNAL_SERVER_ERROR)))
            logger.info("request_done req_id=%s method=POST path=%s status=%d latency_ms=%.1f", request_id, parsed.path, status, latency_ms)
            self._record_http_request_metrics(
                status_code=status,
                latency_seconds=latency_ms / 1000.0,
            )


def serve(
    host: str,
    port: int,
    config: EngineConfig,
    *,
    api_key: str | None = None,
    rate_limiter: _RateLimiter | None = None,
) -> None:
    OpenHydraHandler.engine = CoordinatorEngine(config)
    OpenHydraHandler._api_key = api_key or None
    OpenHydraHandler._rate_limiter = rate_limiter
    if api_key:
        logger.info("API key authentication enabled")
    else:
        logger.warning("API key authentication is DISABLED — set --api-key for production")
    if rate_limiter:
        logger.info("Rate limiting enabled")

    # Warn when the HYDRA ledger bridge is running in mock mode (default).
    # All burn/mint/settlement calls are in-memory only — no real L1 settlement occurs.
    engine = OpenHydraHandler.engine
    if engine is not None and getattr(getattr(engine, "ledger_bridge", None), "mock_mode", False):
        logger.warning(
            "HYDRA_BRIDGE_MOCK_MODE=true — token burn/mint/settlement is simulated in memory only. "
            "No real on-chain transactions will occur. "
            "Set --no-hydra-ledger-bridge-mock-mode and wire a real L1 resolver for production."
        )

    server = ThreadingHTTPServer((host, port), OpenHydraHandler)

    def _on_sigterm(signum: int, _frame: object) -> None:
        # Called on the main thread; server.shutdown() blocks until
        # serve_forever() returns, so dispatch it to a daemon thread.
        logger.info("shutdown_requested signal=%s", signal.Signals(signum).name)
        threading.Thread(target=server.shutdown, daemon=True, name="sigterm-shutdown").start()

    signal.signal(signal.SIGTERM, _on_sigterm)

    logger.info("OpenHydra API listening on http://%s:%d", host, port)
    print(f"OpenHydra API listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutdown_requested signal=SIGINT")
    finally:
        server.server_close()
        OpenHydraHandler.engine.close()
        logger.info("shutdown_complete")


def main() -> None:
    def _parse_dht_urls(raw_values: list[str] | None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in list(raw_values or []):
            for token in str(raw).split(","):
                value = token.strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                out.append(value)
        return out

    parser = argparse.ArgumentParser(description="OpenHydra OpenAI-compatible API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--deployment-profile", choices=["dev", "prod"], default="dev")
    parser.add_argument("--secrets-file", default=None, help="Path to KEY=VALUE secrets file (0600 permissions required)")
    parser.add_argument("--peers", default=None, help="Path to peer config")
    parser.add_argument(
        "--dht-url",
        action="append",
        default=None,
        help=(
            "DHT bootstrap URL(s) — repeat the flag or use comma-separated values. "
            "Defaults to the three production OpenHydra bootstrap nodes. "
            "Passing even one --dht-url replaces the entire default list."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token / API key required on all requests (recommended for production). "
             "Also readable from OPENHYDRA_API_KEY env var.",
    )
    parser.add_argument("--rate-limit-requests", type=int, default=120,
                        help="Max requests per IP per rate-limit window (default: 120)")
    parser.add_argument("--rate-limit-window-seconds", type=float, default=60.0,
                        help="Sliding window length in seconds for rate limiting (default: 60)")
    parser.add_argument("--dht-lookup-timeout", type=float, default=3.0,
                        help="Per-bootstrap-node HTTP timeout for DHT /lookup calls (seconds). "
                             "Default 3.0 s accommodates geo-distributed nodes; lower only in "
                             "single-region LAN deployments.")
    parser.add_argument("--dht-lookup-cache-ttl", type=float, default=120.0,
                        help="How long (seconds) to cache a successful DHT peer list before "
                             "re-querying bootstrap nodes (default: 120).")
    parser.add_argument("--dht-lookup-limit", type=int, default=0)
    parser.add_argument("--dht-lookup-sloppy-factor", type=int, default=3)
    parser.add_argument("--dht-lookup-dsht-replicas", type=int, default=2)
    parser.add_argument("--dht-preferred-region", default=None)
    parser.add_argument("--pipeline-width", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=500)
    parser.add_argument("--max-latency-ms", type=float, default=5000.0)
    parser.add_argument("--audit-rate", type=float, default=0.10)
    parser.add_argument("--redundant-exec-rate", type=float, default=0.25)
    parser.add_argument("--auditor-rate", type=float, default=0.0)
    parser.add_argument("--verification-alert-min-events", type=int, default=10)
    parser.add_argument("--verification-alert-min-success-rate", type=float, default=0.80)
    parser.add_argument("--verification-qos-min-events", type=int, default=10)
    parser.add_argument("--verification-qos-min-success-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tier", type=int, default=2)
    parser.add_argument("--max-failovers-per-stage", type=int, default=1)
    parser.add_argument("--ledger-path", default=".openhydra/credits.db")
    parser.add_argument("--barter-decay-per-day", type=float, default=0.05)
    parser.add_argument("--hydra-token-ledger-path", default=".openhydra/hydra_tokens.db")
    parser.add_argument("--hydra-reward-per-1k-tokens", type=float, default=1.0)
    parser.add_argument("--hydra-slash-per-failed-verification", type=float, default=0.0)
    parser.add_argument("--hydra-channel-default-ttl-seconds", type=int, default=900)
    parser.add_argument("--hydra-channel-max-open-per-payer", type=int, default=8)
    parser.add_argument("--hydra-channel-min-deposit", type=float, default=0.01)
    parser.add_argument("--hydra-supply-cap", type=float, default=69_000_000.0)
    parser.add_argument("--hydra-ledger-bridge-mock-mode", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hydra-stake-priority-boost", type=float, default=12.0)
    parser.add_argument("--hydra-no-stake-penalty-events", type=int, default=8)
    parser.add_argument("--hydra-governance-daily-mint-rate", type=float, default=250_000.0)
    parser.add_argument("--hydra-governance-min-slash-penalty", type=float, default=0.1)
    parser.add_argument("--health-store-path", default=".openhydra/health.json")
    parser.add_argument("--required-replicas", type=int, default=3)
    parser.add_argument("--allow-dynamic-model-ids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-catalog-path", default=None)
    parser.add_argument("--allow-degradation-default", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--operator-cap-fraction", type=float, default=(1.0 / 3.0))
    parser.add_argument("--enforce-pipeline-diversity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diversity-window", type=int, default=3)
    parser.add_argument("--diversity-max-per-window", type=int, default=1)
    parser.add_argument("--prefill-token-threshold", type=int, default=256)
    parser.add_argument("--prefill-min-bandwidth-mbps", type=float, default=500.0)
    parser.add_argument("--decode-max-bandwidth-mbps", type=float, default=50.0)
    parser.add_argument("--grounding-cache-path", default=".openhydra/grounding_cache.json")
    parser.add_argument("--grounding-cache-ttl-seconds", type=int, default=900)
    parser.add_argument("--grounding-timeout-s", type=float, default=3.0)
    parser.add_argument("--grounding-use-network", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grounding-fallback-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speculative-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--speculative-draft-tokens", type=int, default=4)
    parser.add_argument("--speculative-seed", type=int, default=13)
    parser.add_argument("--speculative-adaptive-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--speculative-min-draft-tokens", type=int, default=2)
    parser.add_argument("--speculative-max-draft-tokens", type=int, default=8)
    parser.add_argument("--speculative-acceptance-low-watermark", type=float, default=0.55)
    parser.add_argument("--speculative-acceptance-high-watermark", type=float, default=0.80)
    parser.add_argument("--pipeline-parallel-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pipeline-parallel-workers", type=int, default=1)
    parser.add_argument("--tensor-autoencoder-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tensor-autoencoder-latent-dim", type=int, default=1024)
    parser.add_argument("--advanced-encryption-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--advanced-encryption-seed", default="openhydra-tier3-dev-seed")
    parser.add_argument(
        "--advanced-encryption-level",
        choices=["standard", "enhanced", "maximum"],
        default="standard",
    )
    parser.add_argument("--kv-peer-cache-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--moe-geo-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--moe-geo-min-tag-matches", type=int, default=1)
    parser.add_argument("--moe-geo-min-layer-matches", type=int, default=1)
    parser.add_argument("--moe-geo-prompt-hints-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pytorch-generation-model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--pytorch-speculative-draft-model-id", default="sshleifer/tiny-gpt2")
    parser.add_argument("--tls-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tls-root-cert-path", default=None)
    parser.add_argument("--tls-client-cert-path", default=None)
    parser.add_argument("--tls-client-key-path", default=None)
    parser.add_argument("--tls-server-name-override", default=None)
    parser.add_argument(
        "--database-url",
        default=None,
        help=(
            "PostgreSQL connection URL (e.g. postgresql://user:pass@host:5432/db). "
            "When set, Postgres is used for ledger storage instead of SQLite. "
            "Also readable from the DATABASE_URL environment variable."
        ),
    )
    args = parser.parse_args()
    profile_settings = _resolve_runtime_profile_settings(parser, args)
    configure_logging(json_logs=str(profile_settings.get("deployment_profile", "dev")) == "prod")

    dht_urls = _parse_dht_urls(args.dht_url) or list(PRODUCTION_BOOTSTRAP_URLS)

    import os as _os

    # Resolve API key: CLI flag > env var > None (auth disabled)
    api_key: str | None = (
        str(args.api_key).strip() if args.api_key else
        (str(_os.environ.get("OPENHYDRA_API_KEY", "")).strip() or None)
    )

    # Resolve database URL: CLI flag > env var > None (SQLite default)
    database_url: str | None = (
        str(args.database_url).strip() if args.database_url else
        (_os.environ.get("DATABASE_URL", "").strip() or None)
    )

    rate_limiter = _RateLimiter(
        max_requests=max(1, int(args.rate_limit_requests)),
        window_seconds=max(1.0, float(args.rate_limit_window_seconds)),
    )

    config = EngineConfig(
        deployment_profile=str(profile_settings["deployment_profile"]),
        peers_config_path=args.peers,
        dht_urls=dht_urls,
        dht_url=(dht_urls[0] if dht_urls else None),
        dht_lookup_timeout_s=max(0.1, float(args.dht_lookup_timeout)),
        dht_lookup_cache_ttl_s=max(0.0, float(args.dht_lookup_cache_ttl)),
        dht_lookup_limit=max(0, int(args.dht_lookup_limit)),
        dht_lookup_sloppy_factor=max(0, int(args.dht_lookup_sloppy_factor)),
        dht_lookup_dsht_replicas=max(0, int(args.dht_lookup_dsht_replicas)),
        dht_preferred_region=(str(args.dht_preferred_region) if args.dht_preferred_region else None),
        tls_enabled=args.tls_enable,
        tls_root_cert_path=args.tls_root_cert_path,
        tls_client_cert_path=args.tls_client_cert_path,
        tls_client_key_path=args.tls_client_key_path,
        tls_server_name_override=args.tls_server_name_override,
        timeout_ms=args.timeout_ms,
        max_latency_ms=args.max_latency_ms,
        pipeline_width=args.pipeline_width,
        tier=args.tier,
        audit_rate=args.audit_rate,
        redundant_exec_rate=args.redundant_exec_rate,
        auditor_rate=max(0.0, min(1.0, args.auditor_rate)),
        verification_alert_min_events=max(1, args.verification_alert_min_events),
        verification_alert_min_success_rate=max(0.0, min(1.0, args.verification_alert_min_success_rate)),
        verification_qos_min_events=max(1, args.verification_qos_min_events),
        verification_qos_min_success_rate=max(0.0, min(1.0, args.verification_qos_min_success_rate)),
        seed=args.seed,
        max_failovers_per_stage=max(0, args.max_failovers_per_stage),
        ledger_path=args.ledger_path,
        barter_decay_per_day=max(0.0, args.barter_decay_per_day),
        hydra_token_ledger_path=args.hydra_token_ledger_path,
        hydra_reward_per_1k_tokens=max(0.0, args.hydra_reward_per_1k_tokens),
        hydra_slash_per_failed_verification=max(0.0, args.hydra_slash_per_failed_verification),
        hydra_channel_default_ttl_seconds=max(1, args.hydra_channel_default_ttl_seconds),
        hydra_channel_max_open_per_payer=max(1, args.hydra_channel_max_open_per_payer),
        hydra_channel_min_deposit=max(0.0, args.hydra_channel_min_deposit),
        hydra_supply_cap=max(0.0, float(args.hydra_supply_cap)),
        hydra_ledger_bridge_mock_mode=bool(profile_settings["hydra_ledger_bridge_mock_mode"]),
        hydra_stake_priority_boost=max(0.0, float(args.hydra_stake_priority_boost)),
        hydra_no_stake_penalty_events=max(1, int(args.hydra_no_stake_penalty_events)),
        hydra_governance_daily_mint_rate=max(0.0, float(args.hydra_governance_daily_mint_rate)),
        hydra_governance_min_slash_penalty=max(0.0, float(args.hydra_governance_min_slash_penalty)),
        health_store_path=args.health_store_path,
        database_url=database_url,
        required_replicas=max(1, args.required_replicas),
        allow_dynamic_model_ids=bool(args.allow_dynamic_model_ids),
        model_catalog_path=args.model_catalog_path,
        allow_degradation_default=args.allow_degradation_default,
        operator_cap_fraction=max(0.01, min(1.0, args.operator_cap_fraction)),
        enforce_pipeline_diversity=args.enforce_pipeline_diversity,
        diversity_window=max(2, args.diversity_window),
        diversity_max_per_window=max(1, args.diversity_max_per_window),
        prefill_token_threshold=max(1, args.prefill_token_threshold),
        prefill_min_bandwidth_mbps=max(1.0, args.prefill_min_bandwidth_mbps),
        decode_max_bandwidth_mbps=max(0.0, args.decode_max_bandwidth_mbps),
        grounding_cache_path=args.grounding_cache_path,
        grounding_cache_ttl_seconds=max(1, args.grounding_cache_ttl_seconds),
        grounding_timeout_s=max(0.1, args.grounding_timeout_s),
        grounding_use_network=args.grounding_use_network,
        grounding_fallback_enabled=args.grounding_fallback_enabled,
        speculative_enabled=args.speculative_enabled,
        speculative_draft_tokens=max(1, args.speculative_draft_tokens),
        speculative_seed=int(args.speculative_seed),
        speculative_adaptive_enabled=args.speculative_adaptive_enabled,
        speculative_min_draft_tokens=max(1, args.speculative_min_draft_tokens),
        speculative_max_draft_tokens=max(1, args.speculative_max_draft_tokens),
        speculative_acceptance_low_watermark=max(0.0, min(1.0, args.speculative_acceptance_low_watermark)),
        speculative_acceptance_high_watermark=max(0.0, min(1.0, args.speculative_acceptance_high_watermark)),
        pipeline_parallel_enabled=args.pipeline_parallel_enabled,
        pipeline_parallel_workers=max(1, args.pipeline_parallel_workers),
        tensor_autoencoder_enabled=args.tensor_autoencoder_enabled,
        tensor_autoencoder_latent_dim=max(1, args.tensor_autoencoder_latent_dim),
        advanced_encryption_enabled=args.advanced_encryption_enabled,
        advanced_encryption_seed=str(profile_settings["advanced_encryption_seed"]),
        advanced_encryption_level=str(args.advanced_encryption_level),
        kv_peer_cache_enabled=args.kv_peer_cache_enabled,
        moe_geo_enabled=args.moe_geo_enabled,
        moe_geo_min_tag_matches=max(1, args.moe_geo_min_tag_matches),
        moe_geo_min_layer_matches=max(1, args.moe_geo_min_layer_matches),
        moe_geo_prompt_hints_enabled=args.moe_geo_prompt_hints_enabled,
        pytorch_generation_model_id=str(args.pytorch_generation_model_id),
        pytorch_speculative_draft_model_id=str(args.pytorch_speculative_draft_model_id),
    )
    serve(args.host, args.port, config, api_key=api_key, rate_limiter=rate_limiter)


if __name__ == "__main__":
    main()
