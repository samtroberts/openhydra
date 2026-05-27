from coordinator.chain import InferenceChain, _StageResult
from coordinator.path_finder import PeerEndpoint
from peer import peer_pb2


def _pipeline() -> list[PeerEndpoint]:
    return [
        PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=5001, libp2p_peer_id="12D3KooW_a"),
        PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=5002, libp2p_peer_id="12D3KooW_b"),
    ]


class _MockP2PNode:
    """Mock P2P node that deserializes ForwardRequest, calls handler, returns serialized response.

    Supports both legacy protobuf and OHV2 wire formats.
    """

    def __init__(self, handler):
        self._handler = handler
        self.libp2p_peer_id = "12D3KooW_coord"
        from openhydra_network import P2PNode as _RealNode
        self._real = _RealNode

    def proxy_forward(self, target_peer_id, data):
        raw = bytes(data)
        prefix = raw[0:1]
        payload = raw[1:]
        if self._real.is_ohv2_msg(payload):
            from peer.ohv2_adapter import OHV2Request
            hdr, act, _ = self._real.decode_forward_msg(payload)
            req = OHV2Request(hdr, act)
            resp = self._handler(req)
            resp_hdr = {
                "request_id": str(getattr(resp, "request_id", "")),
                "status": 0,
                "peer_id": str(getattr(resp, "peer_id", "")),
                "stage_index": int(getattr(resp, "stage_index", 0)),
            }
            _err = str(getattr(resp, "error", "") or "")
            if _err:
                resp_hdr["status"] = 1
                resp_hdr["error_message"] = _err
            if bool(getattr(resp, "kv_cache_hit", False)):
                resp_hdr["status"] = 2
            _onp = str(getattr(resp, "onion_next_peer_id", "") or "")
            if _onp:
                resp_hdr["onion_next_peer_id"] = _onp
            _meta = str(getattr(resp, "metadata_json", "") or "")
            if _meta:
                resp_hdr["metadata_json"] = _meta
            import struct as _s
            _act_list = list(getattr(resp, "activation", []))
            _packed_out = bytes(getattr(resp, "activation_packed", b"") or b"")
            if not _packed_out and _act_list:
                _packed_out = _s.pack(f'<{len(_act_list)}f', *_act_list)
            return prefix + bytes(self._real.encode_response_msg(resp_hdr, _packed_out))
        else:
            req = peer_pb2.ForwardRequest()
            req.ParseFromString(payload)
            resp = self._handler(req)
            return prefix + resp.SerializeToString()

    def proxy_forward_no_wait(self, target_peer_id, data):
        pass

    def is_peer_connected(self, peer_id):
        return True

    @staticmethod
    def is_ohv2_msg(data):
        from openhydra_network import P2PNode as _R
        return _R.is_ohv2_msg(data)

    def encode_forward_msg(self, header_dict, activation, msg_type=0):
        return self._real.encode_forward_msg(header_dict, activation, msg_type)

    def encode_response_msg(self, header_dict, activation):
        return self._real.encode_response_msg(header_dict, activation)

    def decode_forward_msg(self, data):
        return self._real.decode_forward_msg(data)

    def decode_response_msg(self, data):
        return self._real.decode_response_msg(data)


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


def test_request_stage_includes_compression_metadata(monkeypatch):
    captured: dict[str, object] = {}

    def _handler(req):
        captured["request"] = req
        return peer_pb2.ForwardResponse(
            request_id=req.request_id,
            peer_id="peer-b",
            activation=[2.0, 2.0, 2.0, 2.0],
            stage_index=req.stage_index,
            error="",
            compression_latent_dim=2,
        )

    chain = InferenceChain(
        _pipeline(),
        timeout_ms=1000,
        tensor_autoencoder_enabled=True,
        tensor_autoencoder_latent_dim=2,
    )
    chain._p2p_node = _MockP2PNode(_handler)
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
    import struct
    _packed = bytes(req.activation_packed)
    if _packed:
        _n = len(_packed) // 4
        _unpacked = list(struct.unpack(f'<{_n}f', _packed))
    else:
        _unpacked = list(req.activation)
    assert len(_unpacked) == 2
    assert abs(_unpacked[0] - 1.5) < 1e-5
    assert abs(_unpacked[1] - 3.5) < 1e-5
    assert req.compression_codec == "tensor_autoencoder_mean_pool"
    assert req.compression_original_dim == 4
    assert req.compression_latent_dim == 2


def test_request_stage_includes_kv_cache_hints(monkeypatch):
    captured: dict[str, object] = {}

    def _handler(req):
        captured["request"] = req
        return peer_pb2.ForwardResponse(
            request_id=req.request_id,
            peer_id="peer-a",
            activation=[3.0, 4.0],
            stage_index=req.stage_index,
            error="",
            kv_cache_hit=True,
        )

    chain = InferenceChain(_pipeline(), timeout_ms=1000, tensor_autoencoder_enabled=False)
    chain._p2p_node = _MockP2PNode(_handler)
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
    assert req.kv_session_id == "session-1"
    assert req.kv_store_activation is True
    assert req.kv_use_cached_activation is True
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
