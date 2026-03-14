import io
import json
from types import SimpleNamespace

from coordinator.api_server import (
    OpenHydraHandler,
    _RateLimiter,
    _validate_infer_params,
)


def _build_handler():
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    headers = []
    handler.wfile = io.BytesIO()
    handler.send_response = lambda status: headers.append(("status", int(status)))
    handler.send_header = lambda key, value: headers.append((key, value))
    handler.end_headers = lambda: None
    handler.client_address = ("127.0.0.1", 0)
    return handler, headers


def test_send_sse_emits_provided_chunks_without_word_splitting():
    handler, headers = _build_handler()

    handler._send_sse(
        request_id="req-1",
        model_id="openhydra-toy-345m",
        chunks=iter(["Hydra", " swarm", "."]),
    )

    assert ("Content-Type", "text/event-stream") in headers

    body = handler.wfile.getvalue().decode("utf-8")
    lines = [line for line in body.splitlines() if line.startswith("data: ")]

    assert lines[-1] == "data: [DONE]"

    json_chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert json_chunks[0]["choices"][0]["delta"]["content"] == "Hydra"
    assert json_chunks[1]["choices"][0]["delta"]["content"] == " swarm"
    assert json_chunks[2]["choices"][0]["delta"]["content"] == "."
    assert json_chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_payload_forwards_session_id():
    class _DummyEngine:
        def __init__(self):
            self.config = SimpleNamespace(default_model="openhydra-toy-345m")
            self.last_kwargs = None

        def infer_chat(self, **kwargs):
            self.last_kwargs = kwargs
            return {"ok": True}

    engine = _DummyEngine()
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    handler.engine = engine

    handler._chat_payload(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "session_id": "session-42",
            "grounding": False,
        },
        request_id="test-req-id",
    )

    assert engine.last_kwargs is not None
    assert engine.last_kwargs["session_id"] == "session-42"


def test_completion_payload_forwards_expert_tags():
    class _DummyEngine:
        def __init__(self):
            self.config = SimpleNamespace(default_model="openhydra-toy-345m")
            self.last_kwargs = None

        def infer(self, **kwargs):
            self.last_kwargs = kwargs
            return {"ok": True}

    engine = _DummyEngine()
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    handler.engine = engine

    handler._completion_payload(
        {
            "prompt": "hello",
            "grounding": False,
            "expert_tags": ["Vision", "code", "vision"],
            "expert_layers": [4, "2", -1, "x", 4],
        },
        request_id="test-req-id",
    )

    assert engine.last_kwargs is not None
    assert engine.last_kwargs["expert_tags"] == ["vision", "code"]
    assert engine.last_kwargs["expert_layer_indices"] == [2, 4]


def test_completion_payload_forwards_decode_controls():
    class _DummyEngine:
        def __init__(self):
            self.config = SimpleNamespace(default_model="openhydra-toy-345m")
            self.last_kwargs = None

        def infer(self, **kwargs):
            self.last_kwargs = kwargs
            return {"ok": True}

    engine = _DummyEngine()
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    handler.engine = engine

    handler._completion_payload(
        {
            "prompt": "hello",
            "grounding": False,
            "do_sample": True,
            "temperature": 0.72,
            "top_p": 0.91,
            "top_k": 24,
            "seed": 7,
        },
        request_id="test-req-id",
    )

    assert engine.last_kwargs is not None
    assert engine.last_kwargs["decode_do_sample"] is True
    assert engine.last_kwargs["decode_temperature"] == 0.72
    assert engine.last_kwargs["decode_top_p"] == 0.91
    assert engine.last_kwargs["decode_top_k"] == 24
    assert engine.last_kwargs["decode_seed"] == 7


def test_hydra_account_route_forwards_client_id():
    class _DummyEngine:
        def __init__(self):
            self.config = SimpleNamespace(default_model="openhydra-toy-345m")
            self.last_client_id = None

        def hydra_account(self, client_id):
            self.last_client_id = client_id
            return {"hydra": {"peer_id": client_id}}

    engine = _DummyEngine()
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    handler.engine = engine
    handler.path = "/v1/hydra/account?client_id=alice"
    handler.client_address = ("127.0.0.1", 0)
    captured = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update({"payload": payload, "status": status})

    handler.do_GET()

    assert engine.last_client_id == "alice"
    assert captured["payload"]["hydra"]["peer_id"] == "alice"


def test_hydra_transfer_route_forwards_payload():
    class _DummyEngine:
        def __init__(self):
            self.config = SimpleNamespace(default_model="openhydra-toy-345m")
            self.called = None

        def hydra_transfer(self, from_client_id, to_client_id, amount):
            self.called = (from_client_id, to_client_id, amount)
            return {"ok": True}

    engine = _DummyEngine()
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    handler.engine = engine
    handler.path = "/v1/hydra/transfer"
    handler.client_address = ("127.0.0.1", 0)
    handler._read_json = lambda: {"from_client_id": "alice", "to_client_id": "bob", "amount": 1.25}
    captured = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update({"payload": payload, "status": status})

    handler.do_POST()

    assert engine.called == ("alice", "bob", 1.25)
    assert captured["payload"] == {"ok": True}


def test_hydra_channel_open_forwards_ttl_seconds():
    class _DummyEngine:
        def __init__(self):
            self.config = SimpleNamespace(default_model="openhydra-toy-345m")
            self.called = None

        def hydra_open_channel(self, channel_id, payer, payee, deposit, ttl_seconds=None):
            self.called = (channel_id, payer, payee, deposit, ttl_seconds)
            return {"hydra_channel": {"channel_id": channel_id}}

    engine = _DummyEngine()
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    handler.engine = engine
    handler.path = "/v1/hydra/channels/open"
    handler.client_address = ("127.0.0.1", 0)
    handler._read_json = lambda: {
        "channel_id": "ch-ttl",
        "payer": "alice",
        "payee": "bob",
        "deposit": 2.0,
        "ttl_seconds": 120,
    }
    captured = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update({"payload": payload, "status": status})

    handler.do_POST()

    assert engine.called == ("ch-ttl", "alice", "bob", 2.0, 120)
    assert captured["payload"]["hydra_channel"]["channel_id"] == "ch-ttl"


def test_hydra_governance_params_route_returns_payload():
    class _DummyEngine:
        def __init__(self):
            self.config = SimpleNamespace(default_model="openhydra-toy-345m")
            self.called = False

        def hydra_governance_params(self):
            self.called = True
            return {"hydra_governance": {"params": {"supply_cap": 69_000_000.0}}}

    engine = _DummyEngine()
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    handler.engine = engine
    handler.path = "/v1/hydra/governance/params"
    handler.client_address = ("127.0.0.1", 0)
    captured = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update({"payload": payload, "status": status})

    handler.do_GET()

    assert engine.called is True
    assert captured["payload"]["hydra_governance"]["params"]["supply_cap"] == 69_000_000.0


def test_hydra_governance_vote_route_forwards_payload():
    class _DummyEngine:
        def __init__(self):
            self.config = SimpleNamespace(default_model="openhydra-toy-345m")
            self.called = None

        def hydra_governance_vote(self, pubkey, proposal_id, vote):
            self.called = (pubkey, proposal_id, vote)
            return {"hydra_governance_vote": {"accepted": True}}

    engine = _DummyEngine()
    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    handler.engine = engine
    handler.path = "/v1/hydra/governance/vote"
    handler.client_address = ("127.0.0.1", 0)
    handler._read_json = lambda: {"pubkey": "alice", "proposal_id": "cap-upd-1", "vote": "yes"}
    captured = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update({"payload": payload, "status": status})

    handler.do_POST()

    assert engine.called == ("alice", "cap-upd-1", "yes")
    assert captured["payload"]["hydra_governance_vote"]["accepted"] is True


def test_metrics_endpoint_returns_prometheus_telemetry():
    class _DummyEngine:
        def metrics_snapshot(self):
            return {
                "dht_lookup_attempts": 8,
                "dht_lookup_successes": 6,
                "dht_lookup_failures": 2,
                "dht_lookup_success_rate": 0.75,
                "hydra_bridge_total_minted": 21.0,
                "hydra_bridge_total_burned": 13.5,
                "hydra_bridge_total_supply": 100.0,
                "hydra_bridge_supply_cap": 69_000_000.0,
            }

    with OpenHydraHandler._metrics_lock:
        OpenHydraHandler._http_requests_total = 0
        OpenHydraHandler._http_request_errors_total = 0
        OpenHydraHandler._http_request_latency_seconds_sum = 0.0
        OpenHydraHandler._http_request_latency_seconds_count = 0

    handler, headers = _build_handler()
    handler.engine = _DummyEngine()
    handler.path = "/metrics"

    handler.do_GET()

    assert ("status", 200) in headers
    assert any(key == "Content-Type" and "text/plain" in str(value) for key, value in headers)
    payload = handler.wfile.getvalue().decode("utf-8")
    assert "openhydra_http_requests_total" in payload
    assert "openhydra_http_request_latency_seconds_avg" in payload
    assert "openhydra_dht_lookup_success_rate 0.75" in payload
    assert "openhydra_hydra_bridge_total_burned_total 13.5" in payload
    assert "openhydra_hydra_bridge_total_minted_total 21.0" in payload


def test_no_viable_model_error_maps_to_503():
    class _DummyEngine:
        def __init__(self):
            self.config = SimpleNamespace(default_model="openhydra-toy-345m")

        def infer(self, **kwargs):
            raise RuntimeError("no_viable_model:insufficient_peers")

    handler = OpenHydraHandler.__new__(OpenHydraHandler)
    handler.engine = _DummyEngine()
    handler.path = "/v1/completions"
    handler.client_address = ("127.0.0.1", 0)
    handler._read_json = lambda: {"prompt": "hello"}

    captured = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update({"payload": payload, "status": status})

    handler.do_POST()

    assert captured["status"] == 503
    assert captured["payload"] == {
        "error": "no_viable_model",
        "reason": "insufficient_peers",
    }


def test_readyz_returns_ok_when_engine_is_set():
    handler, headers = _build_handler()
    handler.engine = object()  # any non-None value signals "ready"
    handler.path = "/readyz"

    captured = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update(
        {"payload": payload, "status": int(status) if not isinstance(status, int) else status}
    )

    handler.do_GET()

    assert captured["payload"]["status"] == "ok"


def test_readyz_returns_503_when_engine_not_initialized():
    handler, headers = _build_handler()
    handler.__class__.engine = None  # simulate pre-init state
    handler.path = "/readyz"

    captured = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update(
        {"payload": payload, "status": int(status) if not isinstance(status, int) else status}
    )

    handler.do_GET()

    assert captured["status"] == 503
    assert captured["payload"]["status"] == "not_ready"
    assert "reason" in captured["payload"]


# ---------------------------------------------------------------------------
# Helpers shared by the new tests below
# ---------------------------------------------------------------------------

class _FakeHeaders:
    """Minimal stand-in for http.server.BaseHTTPRequestHandler.headers."""

    def __init__(self, mapping: dict[str, str] | None = None):
        self._data = {k.lower(): v for k, v in (mapping or {}).items()}

    def get(self, key: str, default: str = "") -> str:
        return self._data.get(str(key).lower(), default)


def _build_handler_with_engine(engine=None):
    """Build a handler wired with an optional engine and no-op send_json."""
    handler, headers = _build_handler()
    if engine is not None:
        handler.engine = engine
    handler.headers = _FakeHeaders()
    captured: dict = {}
    handler._send_json = lambda payload, status=200, headers=None: captured.update(
        {"payload": payload, "status": int(status) if not isinstance(status, int) else status}
    )
    return handler, captured


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------

def test_auth_rejects_request_when_api_key_configured_but_missing():
    """do_GET returns 401 when an API key is required but the client provides none."""
    prev_key = OpenHydraHandler._api_key
    try:
        OpenHydraHandler._api_key = "test-secret"
        handler, captured = _build_handler_with_engine()
        handler.path = "/v1/models"
        handler.do_GET()
    finally:
        OpenHydraHandler._api_key = prev_key

    assert captured["status"] == 401
    assert captured["payload"] == {"error": "unauthorized"}


def test_auth_accepts_correct_bearer_token():
    """do_GET succeeds when the correct Bearer token is supplied."""
    class _MinimalEngine:
        config = SimpleNamespace(default_model="m")
        def list_models(self): return {"models": []}

    prev_key = OpenHydraHandler._api_key
    prev_rl = OpenHydraHandler._rate_limiter
    try:
        OpenHydraHandler._api_key = "good-key"
        OpenHydraHandler._rate_limiter = None
        handler, captured = _build_handler_with_engine(_MinimalEngine())
        handler.headers = _FakeHeaders({"Authorization": "Bearer good-key"})
        handler.path = "/v1/models"
        handler.do_GET()
    finally:
        OpenHydraHandler._api_key = prev_key
        OpenHydraHandler._rate_limiter = prev_rl

    assert captured.get("status", 200) == 200


def test_auth_accepts_x_api_key_header():
    """do_GET succeeds when the correct X-API-Key header is supplied."""
    class _MinimalEngine:
        config = SimpleNamespace(default_model="m")
        def list_models(self): return {"models": []}

    prev_key = OpenHydraHandler._api_key
    prev_rl = OpenHydraHandler._rate_limiter
    try:
        OpenHydraHandler._api_key = "secret-42"
        OpenHydraHandler._rate_limiter = None
        handler, captured = _build_handler_with_engine(_MinimalEngine())
        handler.headers = _FakeHeaders({"X-API-Key": "secret-42"})
        handler.path = "/v1/models"
        handler.do_GET()
    finally:
        OpenHydraHandler._api_key = prev_key
        OpenHydraHandler._rate_limiter = prev_rl

    assert captured.get("status", 200) == 200


# ---------------------------------------------------------------------------
# Rate-limiting tests
# ---------------------------------------------------------------------------

def test_rate_limit_returns_429_when_window_exhausted():
    """do_GET returns 429 when the sliding-window rate limiter says no."""
    class _NeverAllowed:
        _max = 0  # needed for X-RateLimit-Limit header
        def check(self, _ip: str) -> tuple[bool, int, int]:
            import time
            return False, 0, int(time.time()) + 60
        def is_allowed(self, _ip: str) -> bool:
            return False

    prev_key = OpenHydraHandler._api_key
    prev_rl = OpenHydraHandler._rate_limiter
    try:
        OpenHydraHandler._api_key = None          # auth off
        OpenHydraHandler._rate_limiter = _NeverAllowed()
        handler, captured = _build_handler_with_engine()
        handler.path = "/v1/models"
        handler.do_GET()
    finally:
        OpenHydraHandler._api_key = prev_key
        OpenHydraHandler._rate_limiter = prev_rl

    assert captured["status"] == 429
    assert captured["payload"] == {"error": "rate_limit_exceeded"}


def test_rate_limiter_sliding_window_allows_then_blocks():
    """_RateLimiter permits requests up to the cap and blocks the next one."""
    rl = _RateLimiter(max_requests=3, window_seconds=60.0)
    assert rl.is_allowed("1.2.3.4") is True
    assert rl.is_allowed("1.2.3.4") is True
    assert rl.is_allowed("1.2.3.4") is True
    assert rl.is_allowed("1.2.3.4") is False   # 4th request should be blocked
    assert rl.is_allowed("9.9.9.9") is True    # different IP unaffected


# ---------------------------------------------------------------------------
# Inference endpoint tests
# ---------------------------------------------------------------------------

def test_post_completions_returns_engine_response():
    """POST /v1/completions proxies through to engine.infer()."""
    class _InferEngine:
        config = SimpleNamespace(default_model="openhydra-test")
        def infer(self, **kwargs):
            return {
                "response": "hello",
                "request_id": "r1",
                "model": {"served": "openhydra-test", "requested": "openhydra-test", "reason": "none", "detail": ""},
            }

    prev_key = OpenHydraHandler._api_key
    prev_rl = OpenHydraHandler._rate_limiter
    try:
        OpenHydraHandler._api_key = None
        OpenHydraHandler._rate_limiter = None
        handler, captured = _build_handler_with_engine(_InferEngine())
        handler.path = "/v1/completions"
        handler._read_json = lambda: {"prompt": "hello", "max_tokens": 10}
        handler.do_POST()
    finally:
        OpenHydraHandler._api_key = prev_key
        OpenHydraHandler._rate_limiter = prev_rl

    assert captured.get("status", 200) == 200
    # do_POST wraps in OpenAI text_completion format; raw text is in choices[0].text
    assert captured["payload"]["object"] == "text_completion"
    assert captured["payload"]["choices"][0]["text"] == "hello"


def test_post_chat_completions_returns_engine_response():
    """POST /v1/chat/completions proxies through to engine.infer_chat()."""
    class _ChatEngine:
        config = SimpleNamespace(default_model="openhydra-test")
        def infer_chat(self, **kwargs):
            return {
                "response": "hi there",
                "request_id": "r2",
                "model": {"served": "openhydra-test", "requested": "openhydra-test", "reason": "none", "detail": ""},
            }

    prev_key = OpenHydraHandler._api_key
    prev_rl = OpenHydraHandler._rate_limiter
    try:
        OpenHydraHandler._api_key = None
        OpenHydraHandler._rate_limiter = None
        handler, captured = _build_handler_with_engine(_ChatEngine())
        handler.path = "/v1/chat/completions"
        handler._read_json = lambda: {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }
        handler.do_POST()
    finally:
        OpenHydraHandler._api_key = prev_key
        OpenHydraHandler._rate_limiter = prev_rl

    assert captured.get("status", 200) == 200
    # do_POST wraps in OpenAI chat.completion format; text is in choices[0].message.content
    assert captured["payload"]["object"] == "chat.completion"
    assert captured["payload"]["choices"][0]["message"]["content"] == "hi there"


def test_post_invalid_json_body_returns_422_or_400():
    """POST with a malformed JSON body should not crash the handler."""
    class _NullEngine:
        config = SimpleNamespace(default_model="openhydra-test")
        def infer(self, **kwargs):
            return {"response": "ok", "request_id": "x", "model": {}}

    prev_key = OpenHydraHandler._api_key
    prev_rl = OpenHydraHandler._rate_limiter
    try:
        OpenHydraHandler._api_key = None
        OpenHydraHandler._rate_limiter = None
        handler, captured = _build_handler_with_engine(_NullEngine())
        handler.path = "/v1/completions"
        # Simulate a JSON decode error
        import json as _json
        def _bad_read():
            raise _json.JSONDecodeError("Expecting value", "", 0)
        handler._read_json = _bad_read
        handler.do_POST()
    finally:
        OpenHydraHandler._api_key = prev_key
        OpenHydraHandler._rate_limiter = prev_rl

    # Should respond with an error status, not crash
    assert captured["status"] in {400, 422, 502}


# ---------------------------------------------------------------------------
# GET endpoint tests
# ---------------------------------------------------------------------------

def test_get_models_returns_payload():
    """GET /v1/models returns the engine's model list."""
    class _ModelsEngine:
        config = SimpleNamespace(default_model="openhydra-test")
        def list_models(self):
            return {"models": [{"id": "openhydra-test"}]}

    prev_key = OpenHydraHandler._api_key
    prev_rl = OpenHydraHandler._rate_limiter
    try:
        OpenHydraHandler._api_key = None
        OpenHydraHandler._rate_limiter = None
        handler, captured = _build_handler_with_engine(_ModelsEngine())
        handler.path = "/v1/models"
        handler.do_GET()
    finally:
        OpenHydraHandler._api_key = prev_key
        OpenHydraHandler._rate_limiter = prev_rl

    assert captured.get("status", 200) == 200
    assert "models" in captured["payload"]


def test_get_network_status_returns_payload():
    """GET /v1/network/status returns the engine's network status."""
    class _NetEngine:
        config = SimpleNamespace(default_model="openhydra-test")
        def network_status(self):
            return {"peers": 7, "models": ["openhydra-test"]}

    prev_key = OpenHydraHandler._api_key
    prev_rl = OpenHydraHandler._rate_limiter
    try:
        OpenHydraHandler._api_key = None
        OpenHydraHandler._rate_limiter = None
        handler, captured = _build_handler_with_engine(_NetEngine())
        handler.path = "/v1/network/status"
        handler.do_GET()
    finally:
        OpenHydraHandler._api_key = prev_key
        OpenHydraHandler._rate_limiter = prev_rl

    assert captured.get("status", 200) == 200
    assert captured["payload"]["peers"] == 7


def test_get_unknown_path_returns_404():
    """GET to an unrecognised path returns 404."""
    class _AnyEngine:
        config = SimpleNamespace(default_model="openhydra-test")

    prev_key = OpenHydraHandler._api_key
    prev_rl = OpenHydraHandler._rate_limiter
    try:
        OpenHydraHandler._api_key = None
        OpenHydraHandler._rate_limiter = None
        handler, captured = _build_handler_with_engine(_AnyEngine())
        handler.path = "/v1/this/does/not/exist"
        handler.do_GET()
    finally:
        OpenHydraHandler._api_key = prev_key
        OpenHydraHandler._rate_limiter = prev_rl

    assert captured["status"] == 404
    assert captured["payload"] == {"error": "not_found"}


# ---------------------------------------------------------------------------
# _validate_infer_params tests
# ---------------------------------------------------------------------------

def test_validate_infer_params_accepts_valid_body():
    assert _validate_infer_params({"prompt": "hi", "max_tokens": 32}) is None


def test_validate_infer_params_rejects_max_tokens_too_large():
    # One over the 8192 elastic ceiling must be rejected at the API level
    err = _validate_infer_params({"max_tokens": 8193})
    assert err is not None
    assert "max_tokens" in err


def test_validate_infer_params_accepts_max_tokens_at_limit():
    # Exactly at the elastic ceiling is valid (engine enforces 2048 floor via redundancy)
    assert _validate_infer_params({"max_tokens": 8192}) is None


def test_validate_infer_params_rejects_max_tokens_zero():
    err = _validate_infer_params({"max_tokens": 0})
    assert err is not None
    assert ">= 1" in err


def test_validate_infer_params_rejects_prompt_too_long():
    # Exceeds the 65 536-char fast pre-filter
    err = _validate_infer_params({"prompt": "x" * 70_000})
    assert err is not None
    assert "prompt" in err


def test_validate_infer_params_rejects_prompt_too_many_tokens():
    # 9 000 words >> 8 192-token ceiling (word-count estimate)
    long_prompt = " ".join(["word"] * 9_000)
    err = _validate_infer_params({"prompt": long_prompt})
    assert err is not None
    assert "8192" in err or "input tokens" in err


def test_validate_infer_params_rejects_messages_too_long():
    # Exceeds the 65 536-char fast pre-filter
    long_msgs = [{"role": "user", "content": "x" * 70_000}]
    err = _validate_infer_params({"messages": long_msgs})
    assert err is not None
    assert "message" in err


def test_validate_infer_params_rejects_messages_too_many_tokens():
    # 9 000-word message >> 8 192-token ceiling
    long_content = " ".join(["word"] * 9_000)
    err = _validate_infer_params({"messages": [{"role": "user", "content": long_content}]})
    assert err is not None
    assert "8192" in err or "input tokens" in err


def test_validate_infer_params_rejects_bad_pipeline_width():
    err = _validate_infer_params({"pipeline_width": 999})
    assert err is not None
    assert "pipeline_width" in err
