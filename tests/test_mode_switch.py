# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0

"""TDD tests for Pillar 3: The Toggle (State Transitions).

Tests the POST /v1/internal/mode endpoint, the 503 drain gate during
transitions, and the engine swap mechanics.

Run:  pytest tests/test_mode_switch.py -v
"""

from __future__ import annotations

import io
import json
import threading
import time
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from coordinator.api_server import OpenHydraHandler


# ─── Helpers ────────────────────────────────────────────────────────────────

def _build_handler(engine=None, local_engine=None):
    """Create an OpenHydraHandler with mocked I/O."""
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    sent_headers: list[tuple[str, str | int]] = []
    handler.wfile = io.BytesIO()
    handler.send_response = lambda status: sent_headers.append(("_status", int(status)))
    handler.send_header = lambda key, value: sent_headers.append((key, value))
    handler.end_headers = lambda: None
    handler.client_address = ("127.0.0.1", 0)
    handler.__class__.engine = engine
    handler.__class__.local_engine = local_engine
    handler.__class__._api_key = None
    handler.__class__._rate_limiter = None
    # Reset mode controller state
    handler.__class__._mode_switching = False
    return handler, sent_headers


def _get_status(headers):
    """Extract HTTP status code from sent_headers."""
    for key, val in headers:
        if key == "_status":
            return int(val)
    return None


def _parse_json(handler) -> dict:
    """Parse JSON from handler's wfile."""
    return json.loads(handler.wfile.getvalue().decode("utf-8"))


class _MockLocalEngine:
    def __init__(self):
        self.model_id = "openhydra-toy-345m"
        self.shard = MagicMock()

    def chat(self, messages, **kw):
        return {
            "id": "local-1", "object": "chat.completion",
            "model": self.model_id, "created": 1000,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Local"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }

    def list_models(self):
        return [{"id": self.model_id, "object": "model"}]

    def unload(self):
        self.shard = None


class _MockSwarmEngine:
    def __init__(self):
        self.config = SimpleNamespace(default_model="openhydra-toy-345m")

    def infer_chat(self, **kw):
        return {
            "request_id": "swarm-1", "response": "Swarm",
            "model": {"served": "openhydra-toy-345m", "requested": "openhydra-toy-345m"},
        }

    def list_models(self):
        return {"data": [{"id": "openhydra-toy-345m"}]}


# ═══════════════════════════════════════════════════════════════════════════════
# Group A — POST /v1/internal/mode endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestModeEndpoint:

    def test_switch_to_local_sets_local_engine(self):
        """POST /v1/internal/mode {mode: local} should create and set local_engine."""
        swarm = _MockSwarmEngine()
        handler, headers = _build_handler(engine=swarm)

        body = json.dumps({"mode": "local"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_mode_switch()

        resp = _parse_json(handler)
        assert resp["status"] == "ok"
        assert resp["mode"] == "local"
        assert handler.__class__.local_engine is not None

    def test_switch_to_swarm_clears_local_engine(self):
        """POST /v1/internal/mode {mode: swarm} should clear local_engine."""
        local = _MockLocalEngine()
        swarm = _MockSwarmEngine()
        handler, headers = _build_handler(engine=swarm, local_engine=local)

        body = json.dumps({"mode": "swarm"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_mode_switch()

        resp = _parse_json(handler)
        assert resp["status"] == "ok"
        assert resp["mode"] == "swarm"
        assert handler.__class__.local_engine is None

    def test_switch_to_invalid_mode_returns_400(self):
        """POST /v1/internal/mode {mode: potato} should return 400."""
        handler, headers = _build_handler(engine=_MockSwarmEngine())

        body = json.dumps({"mode": "potato"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_mode_switch()

        status = _get_status(headers)
        assert status == 400

    def test_switch_rejects_non_localhost(self):
        """Mode switch must only be accepted from 127.0.0.1."""
        handler, headers = _build_handler(engine=_MockSwarmEngine())
        handler.client_address = ("192.168.1.50", 12345)

        body = json.dumps({"mode": "local"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_mode_switch()

        status = _get_status(headers)
        assert status == 403

    def test_double_switch_to_same_mode_is_noop(self):
        """Switching to current mode returns ok without side effects."""
        local = _MockLocalEngine()
        handler, headers = _build_handler(engine=_MockSwarmEngine(), local_engine=local)

        body = json.dumps({"mode": "local"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_mode_switch()

        resp = _parse_json(handler)
        assert resp["status"] == "ok"
        assert resp["mode"] == "local"
        # local_engine should still be the same object
        assert handler.__class__.local_engine is local


# ═══════════════════════════════════════════════════════════════════════════════
# Group B — 503 Drain Gate
# ═══════════════════════════════════════════════════════════════════════════════

class TestDrainGate:

    def test_503_during_mode_switch(self):
        """While _mode_switching is True, inference requests get 503."""
        handler, headers = _build_handler(engine=_MockSwarmEngine())
        handler.__class__._mode_switching = True

        body = json.dumps({
            "messages": [{"role": "user", "content": "Hi"}],
        }).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"

        handler._handle_chat_completions()

        status = _get_status(headers)
        assert status == 503
        resp = _parse_json(handler)
        assert "switching" in resp.get("error", "").lower() or "mode" in resp.get("error", "").lower()

    def test_503_includes_retry_after_header(self):
        """503 response during mode switch must include Retry-After header."""
        handler, headers = _build_handler(engine=_MockSwarmEngine())
        handler.__class__._mode_switching = True

        body = json.dumps({"messages": [{"role": "user", "content": "Hi"}]}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"

        handler._handle_chat_completions()

        retry_after = None
        for key, val in headers:
            if key == "Retry-After":
                retry_after = val
        assert retry_after is not None
        assert int(retry_after) > 0

    def test_requests_resume_after_switch(self):
        """After _mode_switching goes False, requests should succeed again."""
        local = _MockLocalEngine()
        handler, headers = _build_handler(local_engine=local)
        handler.__class__._mode_switching = False

        body = json.dumps({"messages": [{"role": "user", "content": "Hi"}]}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler.path = "/v1/chat/completions"

        handler._handle_chat_completions()

        status = _get_status(headers)
        assert status == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Group C — Thread-safe concurrent switch protection
# ═══════════════════════════════════════════════════════════════════════════════

class TestConcurrencySafety:

    def test_concurrent_switch_serialized(self):
        """Two simultaneous mode switches should serialize — not crash."""
        swarm = _MockSwarmEngine()
        results = []
        errors = []

        def _switch(mode):
            try:
                h, hdrs = _build_handler(engine=swarm)
                body = json.dumps({"mode": mode}).encode()
                h.rfile = io.BytesIO(body)
                h.headers = {"Content-Length": str(len(body))}
                h.client_address = ("127.0.0.1", 0)
                h._handle_mode_switch()
                resp = json.loads(h.wfile.getvalue().decode())
                results.append(resp)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=_switch, args=("local",))
        t2 = threading.Thread(target=_switch, args=("swarm",))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert len(errors) == 0, f"Concurrent switch errors: {errors}"
        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# Group D — Mode status query
# ═══════════════════════════════════════════════════════════════════════════════

class TestModeStatus:

    def test_get_current_mode_local(self):
        """GET /v1/internal/mode should return current mode."""
        handler, _ = _build_handler(local_engine=_MockLocalEngine())

        handler._handle_mode_status()

        resp = _parse_json(handler)
        assert resp["mode"] == "local"

    def test_get_current_mode_swarm(self):
        """GET /v1/internal/mode should return 'swarm' when no local engine."""
        handler, _ = _build_handler(engine=_MockSwarmEngine())

        handler._handle_mode_status()

        resp = _parse_json(handler)
        assert resp["mode"] == "swarm"
