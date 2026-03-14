import pytest
import os
import stat

from peer.crypto import (
    build_activation_envelope,
    build_activation_envelope_with_pubkey,
    build_onion_route_envelope,
    build_onion_route_envelope_with_pubkeys,
    cryptography_available,
    decrypt_activation_envelope,
    decrypt_activation_envelope_with_privkey,
    generate_identity,
    load_identity_keyfile,
    load_or_create_identity_keyfile,
    peel_onion_route_layer,
    peel_onion_route_layer_with_privkey,
    private_key_from_identity,
    required_layers_for_level,
    save_identity_keyfile,
)


pytestmark = pytest.mark.skipif(not cryptography_available(), reason="cryptography dependency unavailable")


def test_activation_envelope_roundtrip_standard():
    activation = [0.25, -1.5, 3.0, 8.125]
    env = build_activation_envelope(
        activation,
        peer_id="peer-a",
        request_id="req-1",
        stage_index=1,
        shared_secret_seed="seed-1",
        level="standard",
    )

    restored = decrypt_activation_envelope(
        ciphertext=env.ciphertext,
        nonces=env.nonces,
        ephemeral_public_keys=env.ephemeral_public_keys,
        peer_id="peer-a",
        request_id="req-1",
        stage_index=1,
        shared_secret_seed="seed-1",
    )

    assert env.layers == 1
    assert restored == pytest.approx(activation, abs=1e-6)


def test_activation_envelope_roundtrip_enhanced_layers():
    activation = [1.0, 2.0, 3.0]
    env = build_activation_envelope(
        activation,
        peer_id="peer-z",
        request_id="req-2",
        stage_index=3,
        shared_secret_seed="seed-2",
        level="enhanced",
    )

    restored = decrypt_activation_envelope(
        ciphertext=env.ciphertext,
        nonces=env.nonces,
        ephemeral_public_keys=env.ephemeral_public_keys,
        peer_id="peer-z",
        request_id="req-2",
        stage_index=3,
        shared_secret_seed="seed-2",
    )

    assert env.layers == 2
    assert len(env.nonces) == 2
    assert len(env.ephemeral_public_keys) == 2
    assert restored == pytest.approx(activation, abs=1e-6)


def test_required_layers_for_level():
    assert required_layers_for_level("standard") == 1
    assert required_layers_for_level("enhanced") == 2
    assert required_layers_for_level("maximum") == 3


def test_onion_route_envelope_peels_one_layer_per_peer():
    env = build_onion_route_envelope(
        ["peer-a", "peer-b", "peer-c"],
        request_id="req-route-1",
        shared_secret_seed="seed-route",
    )
    assert env.layers == 3

    first = peel_onion_route_layer(
        ciphertext=env.ciphertext,
        nonces=env.nonces,
        ephemeral_public_keys=env.ephemeral_public_keys,
        peer_id="peer-a",
        request_id="req-route-1",
        stage_index=0,
        shared_secret_seed="seed-route",
    )
    assert first.next_peer_id == "peer-b"
    assert first.remaining_ciphertext
    assert first.remaining_layers == 2

    second = peel_onion_route_layer(
        ciphertext=first.remaining_ciphertext,
        nonces=first.remaining_nonces,
        ephemeral_public_keys=first.remaining_ephemeral_public_keys,
        peer_id="peer-b",
        request_id="req-route-1",
        stage_index=1,
        shared_secret_seed="seed-route",
    )
    assert second.next_peer_id == "peer-c"
    assert second.remaining_ciphertext
    assert second.remaining_layers == 1

    third = peel_onion_route_layer(
        ciphertext=second.remaining_ciphertext,
        nonces=second.remaining_nonces,
        ephemeral_public_keys=second.remaining_ephemeral_public_keys,
        peer_id="peer-c",
        request_id="req-route-1",
        stage_index=2,
        shared_secret_seed="seed-route",
    )
    assert third.next_peer_id == ""
    assert third.remaining_ciphertext == b""
    assert third.remaining_layers == 0


def test_generate_identity_real_keys():
    identity = generate_identity()
    assert len(identity.peer_id) == 16
    assert len(identity.public_key) == 64
    assert identity.public_key == identity.public_key.lower()

    from cryptography.hazmat.primitives.asymmetric import x25519

    raw_pub = bytes.fromhex(identity.public_key)
    x25519.X25519PublicKey.from_public_bytes(raw_pub)


def test_generate_identity_deterministic_with_seed():
    a = generate_identity(seed="test-seed-abc")
    b = generate_identity(seed="test-seed-abc")
    assert a == b


def test_generate_identity_different_without_seed():
    a = generate_identity()
    b = generate_identity()
    assert a.public_key != b.public_key


def test_save_and_load_identity_keyfile(tmp_path):
    identity = generate_identity()
    path = tmp_path / "peer.key"
    save_identity_keyfile(identity, path)
    st = path.stat()
    if hasattr(stat, "S_IMODE"):
        assert (st.st_mode & 0o777) == 0o600 or os.name == "nt"
    loaded = load_identity_keyfile(path)
    assert loaded.peer_id == identity.peer_id
    assert loaded.public_key == identity.public_key
    assert loaded.private_key == identity.private_key


def test_load_or_create_identity_keyfile(tmp_path):
    path = tmp_path / "subdir" / "peer.key"
    a = load_or_create_identity_keyfile(path)
    assert path.exists()
    b = load_or_create_identity_keyfile(path)
    assert a == b


def test_build_activation_envelope_with_pubkey_round_trip():
    identity = generate_identity()
    raw_pub = bytes.fromhex(identity.public_key)
    priv_key = private_key_from_identity(identity)
    activation = [1.0, 2.0, 3.0]
    envelope = build_activation_envelope_with_pubkey(
        activation,
        raw_public_key_bytes=raw_pub,
        peer_id=identity.peer_id,
        request_id="req-test",
        stage_index=0,
        level="standard",
    )
    recovered = decrypt_activation_envelope_with_privkey(
        ciphertext=envelope.ciphertext,
        nonces=envelope.nonces,
        ephemeral_public_keys=envelope.ephemeral_public_keys,
        private_key=priv_key,
        peer_id=identity.peer_id,
        request_id="req-test",
        stage_index=0,
    )
    assert recovered == pytest.approx(activation)


def test_build_onion_route_with_pubkeys_round_trip():
    peers = [generate_identity() for _ in range(3)]
    route_peer_ids = [p.peer_id for p in peers]
    peer_public_keys = {p.peer_id: bytes.fromhex(p.public_key) for p in peers}
    envelope = build_onion_route_envelope_with_pubkeys(
        route_peer_ids,
        peer_public_keys=peer_public_keys,
        request_id="req-onion",
    )
    layer = peel_onion_route_layer_with_privkey(
        ciphertext=envelope.ciphertext,
        nonces=envelope.nonces,
        ephemeral_public_keys=envelope.ephemeral_public_keys,
        private_key=private_key_from_identity(peers[0]),
        peer_id=peers[0].peer_id,
        request_id="req-onion",
        stage_index=0,
    )
    assert layer.next_peer_id == peers[1].peer_id
