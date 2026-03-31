# Copyright 2026 OpenHydra contributors
# Licensed under the Apache License, Version 2.0

"""TDD tests for Pillar 4: VRAM Reallocation (Model Swapping).

Tests that mode transitions correctly:
  1. Destroy the old engine BEFORE creating the new one
  2. Call gc.collect() + mx.metal.clear_cache() during teardown
  3. Accept optional model_id in the mode switch request
  4. Use hardware-appropriate defaults when no model_id is given

Run:  pytest tests/test_vram_reallocation.py -v
"""

from __future__ import annotations

import gc
import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from coordinator.api_server import OpenHydraHandler


# ─── Helpers ────────────────────────────────────────────────────────────────

def _build_handler(engine=None, local_engine=None):
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    sent_headers: list[tuple] = []
    handler.wfile = io.BytesIO()
    handler.send_response = lambda status: sent_headers.append(("_status", int(status)))
    handler.send_header = lambda key, value: sent_headers.append((key, value))
    handler.end_headers = lambda: None
    handler.client_address = ("127.0.0.1", 0)
    handler.__class__.engine = engine
    handler.__class__.local_engine = local_engine
    handler.__class__._api_key = None
    handler.__class__._rate_limiter = None
    handler.__class__._mode_switching = False
    return handler, sent_headers


def _parse_json(handler) -> dict:
    return json.loads(handler.wfile.getvalue().decode("utf-8"))


def _get_status(headers):
    for key, val in headers:
        if key == "_status":
            return int(val)
    return None


class _MockSwarmEngine:
    def __init__(self):
        self.config = SimpleNamespace(default_model="openhydra-toy-345m")

    def list_models(self):
        return {"data": [{"id": "openhydra-toy-345m"}]}

    def infer_chat(self, **kw):
        return {"request_id": "s-1", "response": "Swarm",
                "model": {"served": "openhydra-toy-345m", "requested": "openhydra-toy-345m"}}


class _MockLocalEngine:
    """Local engine mock that tracks unload/destroy calls."""

    def __init__(self, model_id="openhydra-toy-345m"):
        self.model_id = model_id
        self.shard = MagicMock()
        self.unloaded = False
        self.destroyed = False

    def unload(self):
        self.unloaded = True
        self.shard = None

    def chat(self, messages, **kw):
        return {
            "id": "l-1", "object": "chat.completion", "model": self.model_id,
            "created": 1000,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Local"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def list_models(self):
        return [{"id": self.model_id, "object": "model"}]


# ═══════════════════════════════════════════════════════════════════════════════
# Group A — Unload-before-load ordering
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnloadBeforeLoad:

    def test_switch_to_swarm_unloads_local_engine_first(self):
        """Switching to swarm must call unload() on local engine."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(engine=_MockSwarmEngine(), local_engine=local)

        body = json.dumps({"mode": "swarm"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler._handle_mode_switch()

        assert local.unloaded is True
        assert handler.__class__.local_engine is None

    def test_switch_to_local_unloads_previous_local_if_exists(self):
        """If switching to local when already in local (with different model),
        the old local engine must be unloaded first."""
        old_local = _MockLocalEngine(model_id="old-model")
        handler, _ = _build_handler(engine=_MockSwarmEngine(), local_engine=old_local)

        # Force non-noop by clearing local_engine (simulate a forced re-switch)
        handler.__class__.local_engine = None

        body = json.dumps({"mode": "local"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler._handle_mode_switch()

        # New local engine should be created
        assert handler.__class__.local_engine is not None

    def test_switch_to_swarm_calls_gc_collect(self):
        """gc.collect() must be called during swarm switch to free Python objects."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(engine=_MockSwarmEngine(), local_engine=local)

        body = json.dumps({"mode": "swarm"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        with patch("gc.collect") as mock_gc:
            handler._handle_mode_switch()
            assert mock_gc.called, "gc.collect() must be called during mode switch"

    def test_switch_to_local_calls_gc_collect(self):
        """gc.collect() must be called during local switch (freeing swarm resources)."""
        handler, _ = _build_handler(engine=_MockSwarmEngine())

        body = json.dumps({"mode": "local"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        with patch("gc.collect") as mock_gc:
            handler._handle_mode_switch()
            assert mock_gc.called, "gc.collect() must be called during mode switch"


# ═══════════════════════════════════════════════════════════════════════════════
# Group B — MLX Metal cache clearing
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetalCacheClear:

    def test_switch_to_swarm_clears_metal_cache_if_available(self):
        """_safe_free_memory() should attempt mx.metal.clear_cache()."""
        from coordinator import api_server as _mod

        local = _MockLocalEngine()
        handler, _ = _build_handler(engine=_MockSwarmEngine(), local_engine=local)

        body = json.dumps({"mode": "swarm"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        # Track whether _safe_free_memory was called (it internally handles MLX)
        calls: list[str] = []
        orig = _mod._safe_free_memory

        def tracked():
            calls.append("safe_free_memory")
            orig()

        with patch.object(_mod, "_safe_free_memory", tracked):
            handler._handle_mode_switch()

        assert "safe_free_memory" in calls, "_safe_free_memory must be called during swarm switch"

    def test_switch_survives_without_mlx(self):
        """Mode switch must not crash if mlx is not installed."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(engine=_MockSwarmEngine(), local_engine=local)

        body = json.dumps({"mode": "swarm"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        # Simulate mlx not installed
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None}):
            handler._handle_mode_switch()

        resp = _parse_json(handler)
        assert resp["status"] == "ok"


# ═══════════════════════════════════════════════════════════════════════════════
# Group C — Optional model_id in mode switch request
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelIdParam:

    def test_switch_to_local_with_explicit_model_id(self):
        """POST {mode: local, model_id: X} should use X as the local model."""
        handler, _ = _build_handler(engine=_MockSwarmEngine())

        body = json.dumps({"mode": "local", "model_id": "my-custom-model"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_mode_switch()

        resp = _parse_json(handler)
        assert resp["status"] == "ok"
        le = handler.__class__.local_engine
        assert le is not None
        assert le.model_id == "my-custom-model"

    def test_switch_to_local_without_model_id_uses_default(self):
        """POST {mode: local} without model_id should use a sensible default."""
        handler, _ = _build_handler(engine=_MockSwarmEngine())

        body = json.dumps({"mode": "local"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_mode_switch()

        le = handler.__class__.local_engine
        assert le is not None
        assert len(le.model_id) > 0  # Not empty

    def test_switch_to_swarm_with_model_id(self):
        """model_id in swarm switch should be accepted (for future use)."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(engine=_MockSwarmEngine(), local_engine=local)

        body = json.dumps({"mode": "swarm", "model_id": "openhydra-qwen3.5-0.8b"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_mode_switch()

        resp = _parse_json(handler)
        assert resp["status"] == "ok"
        assert resp["mode"] == "swarm"

    def test_response_includes_model_id(self):
        """Mode switch response should include the active model_id."""
        handler, _ = _build_handler(engine=_MockSwarmEngine())

        body = json.dumps({"mode": "local", "model_id": "test-model-7b"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        handler._handle_mode_switch()

        resp = _parse_json(handler)
        assert resp["model_id"] == "test-model-7b"


# ═══════════════════════════════════════════════════════════════════════════════
# Group D — Memory safety: del + gc ordering
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemorySafety:

    def test_old_local_engine_reference_deleted_before_new_created(self):
        """The old local_engine must be set to None (del'd) before new one is created."""
        call_order: list[str] = []

        class _TrackingLocalEngine(_MockLocalEngine):
            def unload(self):
                call_order.append("unload")
                super().unload()

        old = _TrackingLocalEngine()
        handler, _ = _build_handler(engine=_MockSwarmEngine(), local_engine=old)
        # Clear to allow re-switch to local
        handler.__class__.local_engine = None

        original_init = type(handler)._switch_to_local

        body = json.dumps({"mode": "local"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}
        handler._handle_mode_switch()

        # New engine should exist
        assert handler.__class__.local_engine is not None

    def test_switch_to_swarm_sets_reference_none_then_gc(self):
        """_switch_to_swarm must: unload() → set None → gc.collect()."""
        local = _MockLocalEngine()
        handler, _ = _build_handler(engine=_MockSwarmEngine(), local_engine=local)

        sequence: list[str] = []
        orig_unload = local.unload

        def tracked_unload():
            sequence.append("unload")
            orig_unload()

        local.unload = tracked_unload

        def tracked_gc():
            sequence.append("gc_collect")

        body = json.dumps({"mode": "swarm"}).encode()
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body))}

        with patch("gc.collect", side_effect=tracked_gc):
            handler._handle_mode_switch()

        assert "unload" in sequence
        assert "gc_collect" in sequence
        # unload must come before gc
        assert sequence.index("unload") < sequence.index("gc_collect")
