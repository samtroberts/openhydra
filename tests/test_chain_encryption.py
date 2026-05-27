import pytest

from coordinator.chain import InferenceChain
from coordinator.path_finder import PeerEndpoint
from peer.crypto import ActivationEnvelope, build_onion_route_envelope, build_privacy_audit_tag
from peer import peer_pb2


class _MockP2PNode:
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
            import struct as _s
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
            # Carry through all response fields used by chain.py.
            for _str_f in ("dp_noise_audit_tag", "onion_route_suite"):
                _sv = str(getattr(resp, _str_f, "") or "")
                if _sv:
                    resp_hdr[_str_f] = _sv
            for _bool_f in ("dp_noise_applied",):
                if bool(getattr(resp, _bool_f, False)):
                    resp_hdr[_bool_f] = True
            for _int_f in ("dp_noise_payload_index", "onion_route_layers"):
                _iv = int(getattr(resp, _int_f, 0) or 0)
                if _iv:
                    resp_hdr[_int_f] = _iv
            for _float_f in ("dp_noise_configured_variance", "dp_noise_observed_variance", "dp_noise_observed_std"):
                _fv = float(getattr(resp, _float_f, 0.0) or 0.0)
                if _fv:
                    resp_hdr[_float_f] = _fv
            # Bytes fields (onion route pass-through).
            _orc = bytes(getattr(resp, "onion_route_ciphertext", b"") or b"")
            if _orc:
                resp_hdr["onion_route_ciphertext"] = list(_orc)
            _orn = list(getattr(resp, "onion_route_nonces", []) or [])
            if _orn:
                resp_hdr["onion_route_nonces"] = [list(x) for x in _orn]
            _orek = list(getattr(resp, "onion_route_ephemeral_public_keys", []) or [])
            if _orek:
                resp_hdr["onion_route_ephemeral_public_keys"] = [list(x) for x in _orek]
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


def _peer() -> PeerEndpoint:
    return PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=5001, libp2p_peer_id="12D3KooW_a")


def test_request_stage_sends_encrypted_activation():
    captured: dict[str, object] = {}

    def handler(req):
        captured["request"] = req
        return peer_pb2.ForwardResponse(
            request_id=req.request_id,
            peer_id="peer-a",
            activation=[1.0],
            stage_index=req.stage_index,
            error="",
        )

    chain = InferenceChain(
        [_peer()],
        timeout_ms=1000,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="enc-seed",
        advanced_encryption_level="enhanced",
    )
    chain._p2p_node = _MockP2PNode(handler)

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

    def handler(req):
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

    monkeypatch.setattr("coordinator.chain.build_activation_envelope_with_pubkey", fake_build_activation_envelope_with_pubkey)
    monkeypatch.setattr("coordinator.chain.build_activation_envelope", fail_seed_path)

    peer = PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=5001, public_key_hex="aa" * 32, libp2p_peer_id="12D3KooW_a")
    chain = InferenceChain(
        [peer],
        timeout_ms=1000,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="enc-seed",
        advanced_encryption_level="standard",
    )
    chain._p2p_node = _MockP2PNode(handler)
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


def test_request_stage_sends_plain_activation_when_disabled():
    captured: dict[str, object] = {}

    def handler(req):
        captured["request"] = req
        return peer_pb2.ForwardResponse(
            request_id=req.request_id,
            peer_id="peer-a",
            activation=[2.0],
            stage_index=req.stage_index,
            error="",
        )

    chain = InferenceChain(
        [_peer()],
        timeout_ms=1000,
        advanced_encryption_enabled=False,
    )
    chain._p2p_node = _MockP2PNode(handler)

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


def test_request_stage_sends_onion_route_and_tracks_remaining_layers():
    captured: dict[str, object] = {}

    def handler(req):
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

    chain = InferenceChain(
        [_peer()],
        timeout_ms=1000,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="enc-seed",
        advanced_encryption_level="enhanced",
    )
    chain._p2p_node = _MockP2PNode(handler)
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


def test_maximum_privacy_mode_verifies_dp_audit_tags():
    def handler(req):
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

    pipeline = [
        PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=5001, privacy_noise_variance=1e-6, libp2p_peer_id="12D3KooW_a"),
        PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=5002, libp2p_peer_id="12D3KooW_b"),
    ]
    chain = InferenceChain(
        pipeline,
        timeout_ms=1000,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="enc-seed",
        advanced_encryption_level="maximum",
    )
    chain._p2p_node = _MockP2PNode(handler)

    result = chain.run("hello", max_tokens=1, request_id="rid-privacy-ok")
    assert result.encryption["privacy_audit_required"] is True
    assert result.encryption["privacy_audit_verified"] is True


def test_maximum_privacy_mode_rejects_missing_dp_audit():
    def handler(req):
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

    pipeline = [
        PeerEndpoint(peer_id="peer-a", host="127.0.0.1", port=5001, privacy_noise_variance=1e-6, libp2p_peer_id="12D3KooW_a"),
        PeerEndpoint(peer_id="peer-b", host="127.0.0.1", port=5002, libp2p_peer_id="12D3KooW_b"),
    ]
    chain = InferenceChain(
        pipeline,
        timeout_ms=1000,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="enc-seed",
        advanced_encryption_level="maximum",
    )
    chain._p2p_node = _MockP2PNode(handler)

    with pytest.raises(RuntimeError, match="privacy_audit_failed"):
        chain.run("hello", max_tokens=1, request_id="rid-privacy-fail")
