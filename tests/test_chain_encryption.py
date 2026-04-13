import pytest

from coordinator.chain import InferenceChain
from coordinator.path_finder import PeerEndpoint
from peer.crypto import ActivationEnvelope, build_onion_route_envelope, build_privacy_audit_tag
from peer import peer_pb2


class _DummyChannel:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


def _peer() -> PeerEndpoint:
    return PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=5001)


def test_request_stage_sends_encrypted_activation(monkeypatch):
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
                activation=[1.0],
                stage_index=req.stage_index,
                error="",
            )

    monkeypatch.setattr("coordinator.chain.create_channel", fake_create_channel)
    monkeypatch.setattr("coordinator.chain.peer_pb2_grpc.PeerStub", _Stub)

    chain = InferenceChain(
        [_peer()],
        timeout_ms=1000,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="enc-seed",
        advanced_encryption_level="enhanced",
    )

    result = chain._request_stage(
        peer=_peer(),
        request_id="r1",
        prompt="hello",
        activation=[0.1, 0.2, 0.3],
        stage_index=1,
        total_stages=2,
        max_tokens=4,
    )
    req = captured["request"]

    assert result.activation == [1.0]
    assert list(req.activation) == []  # type: ignore[union-attr]
    assert bytes(req.encrypted_activation) != b""  # type: ignore[union-attr]
    assert len(req.encryption_nonces) == 2  # type: ignore[union-attr]
    assert len(req.encryption_ephemeral_public_keys) == 2  # type: ignore[union-attr]
    assert req.encryption_layers == 2  # type: ignore[union-attr]


def test_request_stage_prefers_pubkey_encryption_when_available(monkeypatch):
    captured: dict[str, object] = {"used_pubkey": False}

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
                activation=[1.0],
                stage_index=req.stage_index,
                error="",
            )

    def fake_build_activation_envelope_with_pubkey(*args, **kwargs):
        captured["used_pubkey"] = True
        return ActivationEnvelope(
            ciphertext=b"cipher",
            nonces=(b"nonce",),
            ephemeral_public_keys=(b"key",),
            suite="suite",
            layers=1,
        )

    def fail_seed_path(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("seed path should not be used when peer public key is available")

    monkeypatch.setattr("coordinator.chain.create_channel", fake_create_channel)
    monkeypatch.setattr("coordinator.chain.peer_pb2_grpc.PeerStub", _Stub)
    monkeypatch.setattr("coordinator.chain.build_activation_envelope_with_pubkey", fake_build_activation_envelope_with_pubkey)
    monkeypatch.setattr("coordinator.chain.build_activation_envelope", fail_seed_path)

    peer = PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=5001, public_key_hex="aa" * 32)
    chain = InferenceChain(
        [peer],
        timeout_ms=1000,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="enc-seed",
        advanced_encryption_level="standard",
    )
    chain._request_stage(
        peer=peer,
        request_id="r-pub",
        prompt="hello",
        activation=[0.1],
        stage_index=1,
        total_stages=2,
        max_tokens=4,
    )

    req = captured["request"]
    assert captured["used_pubkey"] is True
    assert bytes(req.encrypted_activation) == b"cipher"  # type: ignore[union-attr]


def test_request_stage_sends_plain_activation_when_disabled(monkeypatch):
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
                activation=[2.0],
                stage_index=req.stage_index,
                error="",
            )

    monkeypatch.setattr("coordinator.chain.create_channel", fake_create_channel)
    monkeypatch.setattr("coordinator.chain.peer_pb2_grpc.PeerStub", _Stub)

    chain = InferenceChain(
        [_peer()],
        timeout_ms=1000,
        advanced_encryption_enabled=False,
    )

    result = chain._request_stage(
        peer=_peer(),
        request_id="r2",
        prompt="hello",
        activation=[0.4, 0.5],
        stage_index=1,
        total_stages=2,
        max_tokens=4,
    )
    req = captured["request"]

    assert result.activation == [2.0]
    # Activation is binary-packed when not encrypted/quantized.
    import struct
    _packed = bytes(req.activation_packed)  # type: ignore[union-attr]
    if _packed:
        _n = len(_packed) // 4
        _vals = list(struct.unpack(f'<{_n}f', _packed))
    else:
        _vals = list(req.activation)  # type: ignore[union-attr]
    assert _vals == pytest.approx([0.4, 0.5], abs=1e-6)
    assert bytes(req.encrypted_activation) == b""  # type: ignore[union-attr]


def test_request_stage_sends_onion_route_and_tracks_remaining_layers(monkeypatch):
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
                activation=[1.0],
                stage_index=req.stage_index,
                error="",
                onion_route_ciphertext=b"next-layer",
                onion_route_nonces=[b"n1"],
                onion_route_ephemeral_public_keys=[b"k1"],
                onion_route_suite="suite-route",
                onion_route_layers=2,
                onion_next_peer_id="peer-b",
            )

    monkeypatch.setattr("coordinator.chain.create_channel", fake_create_channel)
    monkeypatch.setattr("coordinator.chain.peer_pb2_grpc.PeerStub", _Stub)

    chain = InferenceChain(
        [_peer()],
        timeout_ms=1000,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="enc-seed",
        advanced_encryption_level="enhanced",
    )
    onion = build_onion_route_envelope(
        ["peer-a", "peer-b", "peer-c"],
        request_id="r3",
        shared_secret_seed="enc-seed",
    )

    result = chain._request_stage(
        peer=_peer(),
        request_id="r3",
        prompt="hello",
        activation=[0.9],
        stage_index=0,
        total_stages=3,
        max_tokens=4,
        onion_route_state={
            "ciphertext": onion.ciphertext,
            "nonces": list(onion.nonces),
            "ephemeral_public_keys": list(onion.ephemeral_public_keys),
            "suite": onion.suite,
            "layers": onion.layers,
        },
    )
    req = captured["request"]

    assert result.activation == [1.0]
    assert bytes(req.onion_route_ciphertext) != b""  # type: ignore[union-attr]
    assert req.onion_route_layers == 3  # type: ignore[union-attr]
    assert chain._last_onion_route_state is not None
    assert int(chain._last_onion_route_state["layers"]) == 2
    assert chain._last_onion_next_peer_id == "peer-b"


def test_maximum_privacy_mode_verifies_dp_audit_tags(monkeypatch):
    def fake_create_channel(_address, _transport_config):
        return _DummyChannel()

    class _Stub:
        def __init__(self, _channel):
            pass

        def Forward(self, req, timeout):
            if int(req.stage_index) == 0:
                configured = 1e-6
                observed = 1.02e-6
                observed_std = observed ** 0.5
                payload_index = 11
                tag = build_privacy_audit_tag(
                    peer_id="peer-a",
                    request_id=req.request_id,
                    stage_index=0,
                    payload_index=payload_index,
                    configured_variance=configured,
                    observed_variance=observed,
                    observed_std=observed_std,
                    shared_secret_seed="enc-seed",
                )
                return peer_pb2.ForwardResponse(
                    request_id=req.request_id,
                    peer_id="peer-a",
                    activation=[0.25, 0.33],
                    stage_index=req.stage_index,
                    error="",
                    onion_next_peer_id="peer-b",
                    dp_noise_applied=True,
                    dp_noise_configured_variance=configured,
                    dp_noise_observed_variance=observed,
                    dp_noise_observed_std=observed_std,
                    dp_noise_payload_index=payload_index,
                    dp_noise_audit_tag=tag,
                )
            return peer_pb2.ForwardResponse(
                request_id=req.request_id,
                peer_id="peer-b",
                activation=[0.8],
                stage_index=req.stage_index,
                error="",
                onion_next_peer_id="",
            )

    monkeypatch.setattr("coordinator.chain.create_channel", fake_create_channel)
    monkeypatch.setattr("coordinator.chain.peer_pb2_grpc.PeerStub", _Stub)

    pipeline = [
        PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=5001, privacy_noise_variance=1e-6),
        PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=5002),
    ]
    chain = InferenceChain(
        pipeline,
        timeout_ms=1000,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="enc-seed",
        advanced_encryption_level="maximum",
    )

    result = chain.run("hello", max_tokens=1, request_id="rid-privacy-ok")
    assert result.encryption["privacy_audit_required"] is True
    assert result.encryption["privacy_audit_verified"] is True


def test_maximum_privacy_mode_rejects_missing_dp_audit(monkeypatch):
    def fake_create_channel(_address, _transport_config):
        return _DummyChannel()

    class _Stub:
        def __init__(self, _channel):
            pass

        def Forward(self, req, timeout):
            if int(req.stage_index) == 0:
                return peer_pb2.ForwardResponse(
                    request_id=req.request_id,
                    peer_id="peer-a",
                    activation=[0.25, 0.33],
                    stage_index=req.stage_index,
                    error="",
                    onion_next_peer_id="peer-b",
                    dp_noise_applied=False,
                    dp_noise_configured_variance=0.0,
                    dp_noise_observed_variance=0.0,
                    dp_noise_observed_std=0.0,
                    dp_noise_payload_index=0,
                    dp_noise_audit_tag="",
                )
            return peer_pb2.ForwardResponse(
                request_id=req.request_id,
                peer_id="peer-b",
                activation=[0.8],
                stage_index=req.stage_index,
                error="",
                onion_next_peer_id="",
            )

    monkeypatch.setattr("coordinator.chain.create_channel", fake_create_channel)
    monkeypatch.setattr("coordinator.chain.peer_pb2_grpc.PeerStub", _Stub)

    pipeline = [
        PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=5001, privacy_noise_variance=1e-6),
        PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=5002),
    ]
    chain = InferenceChain(
        pipeline,
        timeout_ms=1000,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="enc-seed",
        advanced_encryption_level="maximum",
    )

    with pytest.raises(RuntimeError, match="privacy_audit_failed"):
        chain.run("hello", max_tokens=1, request_id="rid-privacy-fail")
