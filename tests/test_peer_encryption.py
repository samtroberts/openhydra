import pytest

from peer.crypto import (
    build_activation_envelope,
    build_activation_envelope_with_pubkey,
    build_onion_route_envelope,
    build_onion_route_envelope_with_pubkeys,
    cryptography_available,
    generate_identity,
    private_key_from_identity,
)
from peer import peer_pb2
from peer.server import PeerService


pytestmark = pytest.mark.skipif(not cryptography_available(), reason="cryptography dependency unavailable")


def _service(*, encryption_enabled: bool) -> PeerService:
    return PeerService(
        peer_id="peer-a",
        model_id="openhydra-toy-345m",
        shard_index=0,
        total_shards=3,
        daemon_mode="polite",
        broken=False,
        advanced_encryption_enabled=encryption_enabled,
        advanced_encryption_seed="seed-1",
    )


def test_peer_service_decrypts_encrypted_forward_request():
    service = _service(encryption_enabled=True)
    env = build_activation_envelope(
        [0.1, 0.2, 0.3],
        peer_id="peer-a",
        request_id="req-enc",
        stage_index=1,
        shared_secret_seed="seed-1",
        level="enhanced",
    )
    req = peer_pb2.ForwardRequest(
        request_id="req-enc",
        prompt="",
        activation=[],
        stage_index=1,
        total_stages=2,
        max_tokens=4,
        encrypted_activation=env.ciphertext,
        encryption_nonces=list(env.nonces),
        encryption_ephemeral_public_keys=list(env.ephemeral_public_keys),
        encryption_suite=env.suite,
        encryption_layers=env.layers,
    )

    resp = service.Forward(req, None)
    assert resp.error == ""
    assert list(resp.activation)


def test_peer_service_rejects_plain_stage_hop_when_encryption_required():
    service = _service(encryption_enabled=True)
    req = peer_pb2.ForwardRequest(
        request_id="req-plain",
        prompt="",
        activation=[1.0, 2.0],
        stage_index=1,
        total_stages=2,
        max_tokens=4,
    )

    resp = service.Forward(req, None)
    assert "encrypted_activation_required" in resp.error


def test_peer_service_decodes_compressed_activation_before_forward():
    service = _service(encryption_enabled=False)
    req = peer_pb2.ForwardRequest(
        request_id="req-comp",
        prompt="",
        activation=[1.5, 3.5],
        stage_index=1,
        total_stages=2,
        max_tokens=4,
        compression_codec="tensor_autoencoder_mean_pool",
        compression_original_dim=4,
        compression_latent_dim=2,
    )

    resp = service.Forward(req, None)
    assert resp.error == ""
    assert list(resp.activation)


def test_peer_service_peels_onion_route_layer():
    service = _service(encryption_enabled=True)
    route = build_onion_route_envelope(
        ["peer-a", "peer-b", "peer-c"],
        request_id="req-onion",
        shared_secret_seed="seed-1",
    )
    req = peer_pb2.ForwardRequest(
        request_id="req-onion",
        prompt="hello",
        activation=[],
        stage_index=0,
        total_stages=3,
        max_tokens=4,
        onion_route_ciphertext=route.ciphertext,
        onion_route_nonces=list(route.nonces),
        onion_route_ephemeral_public_keys=list(route.ephemeral_public_keys),
        onion_route_suite=route.suite,
        onion_route_layers=route.layers,
    )

    resp = service.Forward(req, None)
    assert resp.error == ""
    assert resp.onion_next_peer_id == "peer-b"
    assert bytes(resp.onion_route_ciphertext) != b""
    assert int(resp.onion_route_layers) == 2
    assert int(service.onion_layers_peeled) == 1


def test_peer_service_decrypts_encrypted_forward_request_with_private_key():
    identity = generate_identity(seed="peer-a-key")
    service = PeerService(
        peer_id="peer-a",
        model_id="openhydra-toy-345m",
        shard_index=0,
        total_shards=3,
        daemon_mode="polite",
        broken=False,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="seed-1",
        peer_public_key=identity.public_key,
        peer_private_key=private_key_from_identity(identity),
    )
    env = build_activation_envelope_with_pubkey(
        [0.5, 0.6],
        raw_public_key_bytes=bytes.fromhex(identity.public_key),
        peer_id="peer-a",
        request_id="req-enc-pub",
        stage_index=1,
        level="standard",
    )
    req = peer_pb2.ForwardRequest(
        request_id="req-enc-pub",
        prompt="",
        activation=[],
        stage_index=1,
        total_stages=2,
        max_tokens=4,
        encrypted_activation=env.ciphertext,
        encryption_nonces=list(env.nonces),
        encryption_ephemeral_public_keys=list(env.ephemeral_public_keys),
        encryption_suite=env.suite,
        encryption_layers=env.layers,
    )
    resp = service.Forward(req, None)
    assert resp.error == ""
    assert list(resp.activation)


def test_peer_service_peels_onion_route_layer_with_private_key():
    first = generate_identity(seed="peer-a-key")
    second = generate_identity(seed="peer-b-key")
    third = generate_identity(seed="peer-c-key")
    route = build_onion_route_envelope_with_pubkeys(
        ["peer-a", "peer-b", "peer-c"],
        peer_public_keys={
            "peer-a": bytes.fromhex(first.public_key),
            "peer-b": bytes.fromhex(second.public_key),
            "peer-c": bytes.fromhex(third.public_key),
        },
        request_id="req-onion-pub",
    )
    service = PeerService(
        peer_id="peer-a",
        model_id="openhydra-toy-345m",
        shard_index=0,
        total_shards=3,
        daemon_mode="polite",
        broken=False,
        advanced_encryption_enabled=True,
        advanced_encryption_seed="seed-1",
        peer_public_key=first.public_key,
        peer_private_key=private_key_from_identity(first),
    )
    req = peer_pb2.ForwardRequest(
        request_id="req-onion-pub",
        prompt="hello",
        activation=[],
        stage_index=0,
        total_stages=3,
        max_tokens=4,
        onion_route_ciphertext=route.ciphertext,
        onion_route_nonces=list(route.nonces),
        onion_route_ephemeral_public_keys=list(route.ephemeral_public_keys),
        onion_route_suite=route.suite,
        onion_route_layers=route.layers,
    )
    resp = service.Forward(req, None)
    assert resp.error == ""
    assert resp.onion_next_peer_id == "peer-b"
