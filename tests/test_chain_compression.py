from coordinator.chain import InferenceChain, _StageResult
from coordinator.path_finder import PeerEndpoint
from peer import peer_pb2


def _pipeline() -> list[PeerEndpoint]:
    return [
        PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=5001),
        PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=5002),
    ]


def test_chain_autoencoder_compresses_transfer_hop(monkeypatch):
    calls: list[tuple[int, list[float]]] = []

    def fake_request(self, peer, request_id, prompt, activation, stage_index, total_stages, max_tokens, **kwargs):
        calls.append((stage_index, list(activation)))
        if stage_index == 0:
            return _StageResult(activation=[1.0, 2.0, 3.0, 4.0], latency_ms=1.0, latent_dim=0)
        return _StageResult(activation=list(activation), latency_ms=1.0, latent_dim=2)

    monkeypatch.setattr(InferenceChain, "_request_stage", fake_request)

    chain = InferenceChain(
        _pipeline(),
        timeout_ms=1000,
        tensor_autoencoder_enabled=True,
        tensor_autoencoder_latent_dim=2,
    )

    result = chain.run("hello", max_tokens=4)

    assert calls[0] == (0, [])
    assert calls[1] == (1, [1.0, 2.0, 3.0, 4.0])
    assert result.compression is not None
    assert result.compression["enabled"] is True
    assert result.compression["hops_compressed"] == 1
    assert result.compression["total_input_elements"] == 4
    assert result.compression["total_latent_elements"] == 2
    assert result.compression["avg_compression_ratio"] == 0.5


def test_chain_without_autoencoder_skips_compression(monkeypatch):
    calls: list[tuple[int, list[float]]] = []

    def fake_request(self, peer, request_id, prompt, activation, stage_index, total_stages, max_tokens, **kwargs):
        calls.append((stage_index, list(activation)))
        if stage_index == 0:
            return _StageResult(activation=[1.0, 2.0, 3.0, 4.0], latency_ms=1.0, latent_dim=0)
        return _StageResult(activation=list(activation), latency_ms=1.0, latent_dim=0)

    monkeypatch.setattr(InferenceChain, "_request_stage", fake_request)

    chain = InferenceChain(_pipeline(), timeout_ms=1000, tensor_autoencoder_enabled=False)
    result = chain.run("hello", max_tokens=4)

    assert calls[1] == (1, [1.0, 2.0, 3.0, 4.0])
    assert result.compression is not None
    assert result.compression["enabled"] is False
    assert result.compression["hops_compressed"] == 0


class _DummyChannel:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


def test_request_stage_includes_compression_metadata(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_channel(_address, _transport_config):
        return _DummyChannel()

    class _Stub:
        def __init__(self, _channel):
            pass

        def Forward(self, req, timeout):
            captured["request"] = req
            return peer_pb2.ForwardResponse(
                request_id=req.request_id,
                peer_id="peer-b",
                activation=[2.0, 2.0, 2.0, 2.0],
                stage_index=req.stage_index,
                error="",
                compression_latent_dim=2,
            )

    monkeypatch.setattr("coordinator.chain.create_channel", fake_create_channel)
    monkeypatch.setattr("coordinator.chain.peer_pb2_grpc.PeerStub", _Stub)

    chain = InferenceChain(
        _pipeline(),
        timeout_ms=1000,
        tensor_autoencoder_enabled=True,
        tensor_autoencoder_latent_dim=2,
    )
    chain._request_stage(
        peer=_pipeline()[1],
        request_id="r1",
        prompt="ignored",
        activation=[1.0, 2.0, 3.0, 4.0],
        stage_index=1,
        total_stages=2,
        max_tokens=4,
    )

    req = captured["request"]
    # Activation is now binary-packed (activation_packed field) instead of
    # repeated float. Unpack and verify.
    import struct
    _packed = bytes(req.activation_packed)  # type: ignore[union-attr]
    if _packed:
        _n = len(_packed) // 4
        _unpacked = list(struct.unpack(f'<{_n}f', _packed))
    else:
        _unpacked = list(req.activation)  # type: ignore[union-attr]
    assert len(_unpacked) == 2
    assert abs(_unpacked[0] - 1.5) < 1e-5
    assert abs(_unpacked[1] - 3.5) < 1e-5
    assert req.compression_codec == "tensor_autoencoder_mean_pool"  # type: ignore[union-attr]
    assert req.compression_original_dim == 4  # type: ignore[union-attr]
    assert req.compression_latent_dim == 2  # type: ignore[union-attr]


def test_request_stage_includes_kv_cache_hints(monkeypatch):
    captured: dict[str, object] = {}

    def fake_create_channel(_address, _transport_config):
        return _DummyChannel()

    class _Stub:
        def __init__(self, _channel):
            pass

        def Forward(self, req, timeout):
            captured["request"] = req
            return peer_pb2.ForwardResponse(
                request_id=req.request_id,
                peer_id="peer-a",
                activation=[3.0, 4.0],
                stage_index=req.stage_index,
                error="",
                kv_cache_hit=True,
            )

    monkeypatch.setattr("coordinator.chain.create_channel", fake_create_channel)
    monkeypatch.setattr("coordinator.chain.peer_pb2_grpc.PeerStub", _Stub)

    chain = InferenceChain(_pipeline(), timeout_ms=1000, tensor_autoencoder_enabled=False)
    result = chain._request_stage(
        peer=_pipeline()[0],
        request_id="kv-r1",
        prompt="prefill",
        activation=[],
        stage_index=0,
        total_stages=2,
        max_tokens=4,
        kv_session_id="session-1",
        kv_store_activation=True,
        kv_use_cached_activation=True,
    )

    req = captured["request"]
    assert result.activation == [3.0, 4.0]
    assert req.kv_session_id == "session-1"  # type: ignore[union-attr]
    assert req.kv_store_activation is True  # type: ignore[union-attr]
    assert req.kv_use_cached_activation is True  # type: ignore[union-attr]
    assert chain._last_stage_kv_cache_hit is True


def test_chain_compression_telemetry_uses_stage_latent_dim_without_reencoding(monkeypatch):
    class _CountingAutoencoder:
        def __init__(self):
            self.calls = 0

        def encode(self, vector):
            self.calls += 1
            return [1.5, 3.5]

    def fake_request(self, peer, request_id, prompt, activation, stage_index, total_stages, max_tokens, **kwargs):
        if stage_index == 0:
            return _StageResult(activation=[1.0, 2.0, 3.0, 4.0], latency_ms=1.0, latent_dim=0)
        return _StageResult(activation=[5.0, 6.0, 7.0], latency_ms=1.0, latent_dim=7)

    monkeypatch.setattr(InferenceChain, "_request_stage", fake_request)

    chain = InferenceChain(
        _pipeline(),
        timeout_ms=1000,
        tensor_autoencoder_enabled=True,
        tensor_autoencoder_latent_dim=2,
    )
    counting_autoencoder = _CountingAutoencoder()
    chain._autoencoder = counting_autoencoder  # type: ignore[assignment]

    result = chain.run("hello", max_tokens=4)
    assert counting_autoencoder.calls == 0
    assert result.compression is not None
    assert result.compression["total_latent_elements"] == 7
