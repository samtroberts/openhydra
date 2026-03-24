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

import base64
import hmac
import hashlib
import json
import os
from pathlib import Path
import secrets
import struct
from dataclasses import dataclass
from typing import Any
import warnings

try:  # pragma: no cover - availability is exercised via runtime guards.
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import x25519
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
except ImportError:  # pragma: no cover
    hashes = None  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]
    x25519 = None  # type: ignore[assignment]
    AESGCM = None  # type: ignore[assignment]
    HKDF = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Identity:
    peer_id: str
    public_key: str
    private_key: str


def generate_identity(seed: str | None = None) -> Identity:
    """Generate a real X25519 Identity.

    When *seed* is supplied the private key is deterministic (HKDF from seed)
    — useful for reproducible unit tests. Without a seed a fresh random key
    is generated.

    Falls back to the legacy sha256 stub if *cryptography* is unavailable and
    emits a DeprecationWarning.
    """
    if not cryptography_available():
        warnings.warn(
            "cryptography not available; using insecure sha256 identity stub",
            DeprecationWarning,
            stacklevel=2,
        )
        material = seed or secrets.token_hex(32)
        private_key = hashlib.sha256(f"priv:{material}".encode()).hexdigest()
        public_key = hashlib.sha256(f"pub:{private_key}".encode()).hexdigest()
        peer_id = public_key[:16]
        return Identity(peer_id=peer_id, public_key=public_key, private_key=private_key)

    _require_crypto()
    if seed is None:
        private_obj = x25519.X25519PrivateKey.generate()
    else:
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"openhydra-identity",
            info=b"x25519-private-key",
        )
        private_material = hkdf.derive(str(seed).encode("utf-8"))
        clamped = bytearray(private_material)
        clamped[0] &= 248
        clamped[31] &= 127
        clamped[31] |= 64
        private_obj = x25519.X25519PrivateKey.from_private_bytes(bytes(clamped))

    raw_private = private_obj.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    raw_public = private_obj.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    peer_id = hashlib.sha256(raw_public).hexdigest()[:16]
    return Identity(
        peer_id=peer_id,
        public_key=raw_public.hex(),
        private_key=raw_private.hex(),
    )


def save_identity_keyfile(identity: Identity, path: str | Path) -> None:
    """Write the private key to *path* in PKCS8 PEM format, mode 0600."""
    _require_crypto()
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    private_obj = x25519.X25519PrivateKey.from_private_bytes(bytes.fromhex(identity.private_key))
    pem = private_obj.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path_obj.write_bytes(pem)
    os.chmod(path_obj, 0o600)


def load_identity_keyfile(path: str | Path) -> Identity:
    """Load an Identity from a PKCS8 PEM private-key file."""
    _require_crypto()
    path_obj = Path(path)
    pem = path_obj.read_bytes()
    private_obj = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(private_obj, x25519.X25519PrivateKey):
        raise ValueError(f"keyfile {path_obj} does not contain an X25519 private key")
    raw_private = private_obj.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    raw_public = private_obj.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    peer_id = hashlib.sha256(raw_public).hexdigest()[:16]
    return Identity(
        peer_id=peer_id,
        public_key=raw_public.hex(),
        private_key=raw_private.hex(),
    )


def load_or_create_identity_keyfile(path: str | Path) -> Identity:
    """Load existing identity from *path*, or generate a fresh one and save it."""
    _require_crypto()
    path_obj = Path(path)
    if path_obj.exists():
        return load_identity_keyfile(path_obj)
    identity = generate_identity()
    save_identity_keyfile(identity, path_obj)
    return identity


def private_key_from_identity(identity: Identity) -> Any:
    """Return an x25519.X25519PrivateKey from an Identity's private_key hex field."""
    _require_crypto()
    return x25519.X25519PrivateKey.from_private_bytes(bytes.fromhex(identity.private_key))


@dataclass(frozen=True)
class ActivationEnvelope:
    ciphertext: bytes
    nonces: tuple[bytes, ...]
    ephemeral_public_keys: tuple[bytes, ...]
    suite: str
    layers: int


@dataclass(frozen=True)
class OnionRouteEnvelope:
    ciphertext: bytes
    nonces: tuple[bytes, ...]
    ephemeral_public_keys: tuple[bytes, ...]
    suite: str
    layers: int


@dataclass(frozen=True)
class OnionRouteLayer:
    next_peer_id: str
    remaining_ciphertext: bytes
    remaining_nonces: tuple[bytes, ...]
    remaining_ephemeral_public_keys: tuple[bytes, ...]
    remaining_suite: str
    remaining_layers: int


_LEVEL_TO_LAYERS = {
    "standard": 1,
    "enhanced": 2,
    "maximum": 3,
}
_LEVEL_TO_PADDING_BYTES = {
    "standard": 0,
    "enhanced": 32,
    "maximum": 96,
}
_SUITE_PREFIX = "x25519_hkdf_sha256_aes256_gcm"


def cryptography_available() -> bool:
    return all(item is not None for item in (hashes, serialization, x25519, AESGCM, HKDF))


def _geo_challenge_secret(peer_id: str, shared_secret_seed: str) -> bytes:
    return hashlib.sha256(f"geo_challenge:{shared_secret_seed}:{peer_id}".encode("utf-8")).digest()


def sign_geo_challenge(
    *,
    peer_id: str,
    nonce: str,
    claimed_region: str | None,
    shared_secret_seed: str,
) -> str:
    region = str(claimed_region or "").strip().lower()
    message = f"{peer_id}:{nonce}:{region}".encode("utf-8")
    secret = _geo_challenge_secret(peer_id=peer_id, shared_secret_seed=shared_secret_seed)
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def verify_geo_challenge(
    *,
    peer_id: str,
    nonce: str,
    claimed_region: str | None,
    signature: str,
    shared_secret_seed: str,
) -> bool:
    expected = sign_geo_challenge(
        peer_id=peer_id,
        nonce=nonce,
        claimed_region=claimed_region,
        shared_secret_seed=shared_secret_seed,
    )
    return hmac.compare_digest(str(signature or "").strip().lower(), expected.lower())


def _privacy_audit_secret(peer_id: str, shared_secret_seed: str) -> bytes:
    return hashlib.sha256(f"privacy_audit:{shared_secret_seed}:{peer_id}".encode("utf-8")).digest()


def _privacy_audit_message(
    *,
    peer_id: str,
    request_id: str,
    stage_index: int,
    payload_index: int,
    configured_variance: float,
    observed_variance: float,
    observed_std: float,
) -> bytes:
    configured = format(max(0.0, float(configured_variance)), ".12g")
    observed_var = format(max(0.0, float(observed_variance)), ".12g")
    observed_sigma = format(max(0.0, float(observed_std)), ".12g")
    return (
        f"{peer_id}:{request_id}:{int(stage_index)}:{int(payload_index)}:"
        f"{configured}:{observed_var}:{observed_sigma}"
    ).encode("utf-8")


def build_privacy_audit_tag(
    *,
    peer_id: str,
    request_id: str,
    stage_index: int,
    payload_index: int,
    configured_variance: float,
    observed_variance: float,
    observed_std: float,
    shared_secret_seed: str,
) -> str:
    secret = _privacy_audit_secret(peer_id=peer_id, shared_secret_seed=shared_secret_seed)
    message = _privacy_audit_message(
        peer_id=peer_id,
        request_id=request_id,
        stage_index=stage_index,
        payload_index=payload_index,
        configured_variance=configured_variance,
        observed_variance=observed_variance,
        observed_std=observed_std,
    )
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def verify_privacy_audit_tag(
    *,
    peer_id: str,
    request_id: str,
    stage_index: int,
    payload_index: int,
    configured_variance: float,
    observed_variance: float,
    observed_std: float,
    audit_tag: str,
    shared_secret_seed: str,
) -> bool:
    expected = build_privacy_audit_tag(
        peer_id=peer_id,
        request_id=request_id,
        stage_index=stage_index,
        payload_index=payload_index,
        configured_variance=configured_variance,
        observed_variance=observed_variance,
        observed_std=observed_std,
        shared_secret_seed=shared_secret_seed,
    )
    return hmac.compare_digest(str(audit_tag or "").strip().lower(), expected.lower())


def _require_crypto() -> None:
    if not cryptography_available():
        raise RuntimeError("cryptography_not_available: install 'cryptography>=42'")


def _normalize_level(level: str | None) -> str:
    normalized = str(level or "standard").strip().lower()
    if normalized not in _LEVEL_TO_LAYERS:
        raise ValueError(f"unsupported_encryption_level: {level}")
    return normalized


def required_layers_for_level(level: str | None) -> int:
    return _LEVEL_TO_LAYERS[_normalize_level(level)]


def _padding_bytes_for_level(level: str | None) -> int:
    return _LEVEL_TO_PADDING_BYTES[_normalize_level(level)]


def _peer_private_key(peer_id: str, shared_secret_seed: str):
    _require_crypto()
    material = hashlib.sha256(f"x25519:{shared_secret_seed}:{peer_id}".encode("utf-8")).digest()
    key_bytes = bytearray(material)
    key_bytes[0] &= 248
    key_bytes[31] &= 127
    key_bytes[31] |= 64
    return x25519.X25519PrivateKey.from_private_bytes(bytes(key_bytes))


def peer_static_public_key(peer_id: str, shared_secret_seed: str) -> bytes:
    _require_crypto()
    private_key = _peer_private_key(peer_id=peer_id, shared_secret_seed=shared_secret_seed)
    public_key = private_key.public_key()
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _derive_aes_key(
    shared_secret: bytes,
    request_id: str,
    stage_index: int,
    layer_index: int,
    *,
    purpose: str = "activation",
) -> bytes:
    _require_crypto()
    salt = hashlib.sha256(f"{request_id}:{stage_index}".encode("utf-8")).digest()
    info = f"openhydra/{purpose}/{layer_index}".encode("utf-8")
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=info,
    )
    return hkdf.derive(shared_secret)


def _aad(
    request_id: str,
    stage_index: int,
    peer_id: str,
    layer_index: int,
    layers: int,
    *,
    purpose: str = "activation",
) -> bytes:
    return f"{request_id}:{stage_index}:{peer_id}:{layer_index}:{layers}:{purpose}".encode("utf-8")


def _encode_activation(activation: list[float], padding_bytes: int) -> bytes:
    payload = struct.pack(f"<{len(activation)}f", *activation) if activation else b""
    header = struct.pack("<BI", 1, len(payload))
    padding = os.urandom(max(0, padding_bytes)) if padding_bytes else b""
    return header + payload + padding


def _decode_activation(payload: bytes) -> list[float]:
    if len(payload) < 5:
        raise ValueError("invalid_encrypted_activation_payload: too short")
    version, size = struct.unpack("<BI", payload[:5])
    if version != 1:
        raise ValueError(f"unsupported_encrypted_activation_version: {version}")
    end = 5 + size
    if len(payload) < end:
        raise ValueError("invalid_encrypted_activation_payload: truncated")
    body = payload[5:end]
    if len(body) % 4 != 0:
        raise ValueError("invalid_encrypted_activation_payload: misaligned_float_bytes")
    if not body:
        return []
    count = len(body) // 4
    return list(struct.unpack(f"<{count}f", body))


def _b64_encode(value: bytes) -> str:
    if not value:
        return ""
    return base64.b64encode(value).decode("ascii")


def _b64_decode(value: str | None) -> bytes:
    text = str(value or "").strip()
    if not text:
        return b""
    try:
        return base64.b64decode(text.encode("ascii"))
    except Exception as exc:
        raise ValueError("invalid_onion_route_payload:base64") from exc


def _encode_route_layer_payload(
    *,
    next_peer_id: str,
    remaining_ciphertext: bytes,
    remaining_nonces: list[bytes] | tuple[bytes, ...],
    remaining_ephemeral_public_keys: list[bytes] | tuple[bytes, ...],
    remaining_suite: str,
    remaining_layers: int,
) -> bytes:
    payload = {
        "v": 1,
        "next_peer_id": str(next_peer_id or ""),
        "remaining_ciphertext": _b64_encode(bytes(remaining_ciphertext)),
        "remaining_nonces": [_b64_encode(bytes(item)) for item in list(remaining_nonces)],
        "remaining_ephemeral_public_keys": [
            _b64_encode(bytes(item)) for item in list(remaining_ephemeral_public_keys)
        ],
        "remaining_suite": str(remaining_suite or ""),
        "remaining_layers": int(max(0, remaining_layers)),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _decode_route_layer_payload(payload: bytes) -> OnionRouteLayer:
    try:
        raw = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ValueError("invalid_onion_route_payload:json") from exc

    if not isinstance(raw, dict):
        raise ValueError("invalid_onion_route_payload:type")
    if int(raw.get("v", 0)) != 1:
        raise ValueError("invalid_onion_route_payload:version")

    nonces_raw = raw.get("remaining_nonces", [])
    keys_raw = raw.get("remaining_ephemeral_public_keys", [])
    if not isinstance(nonces_raw, list) or not isinstance(keys_raw, list):
        raise ValueError("invalid_onion_route_payload:layer_metadata")

    remaining_nonces = tuple(_b64_decode(item) for item in nonces_raw)
    remaining_ephemeral_keys = tuple(_b64_decode(item) for item in keys_raw)
    if len(remaining_nonces) != len(remaining_ephemeral_keys):
        raise ValueError("invalid_onion_route_payload:mismatched_layer_metadata")

    return OnionRouteLayer(
        next_peer_id=str(raw.get("next_peer_id", "") or ""),
        remaining_ciphertext=_b64_decode(raw.get("remaining_ciphertext")),
        remaining_nonces=remaining_nonces,
        remaining_ephemeral_public_keys=remaining_ephemeral_keys,
        remaining_suite=str(raw.get("remaining_suite", "") or ""),
        remaining_layers=max(0, int(raw.get("remaining_layers", 0) or 0)),
    )


def _encrypt_payload_for_peer_public_key(
    payload: bytes,
    *,
    peer_pub: Any,
    peer_id: str,
    request_id: str,
    stage_index: int,
    layer_index: int,
    layers: int,
    purpose: str,
) -> tuple[bytes, bytes, bytes]:
    _require_crypto()
    eph_private = x25519.X25519PrivateKey.generate()
    eph_public = eph_private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    shared_secret = eph_private.exchange(peer_pub)
    key = _derive_aes_key(
        shared_secret,
        request_id=request_id,
        stage_index=stage_index,
        layer_index=layer_index,
        purpose=purpose,
    )
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(
        nonce,
        payload,
        _aad(
            request_id=request_id,
            stage_index=stage_index,
            peer_id=peer_id,
            layer_index=layer_index,
            layers=layers,
            purpose=purpose,
        ),
    )
    return ciphertext, nonce, eph_public


def _encrypt_payload_for_peer(
    payload: bytes,
    *,
    peer_id: str,
    request_id: str,
    stage_index: int,
    shared_secret_seed: str,
    layer_index: int,
    layers: int,
    purpose: str,
) -> tuple[bytes, bytes, bytes]:
    _require_crypto()
    peer_pub = x25519.X25519PublicKey.from_public_bytes(
        peer_static_public_key(peer_id=peer_id, shared_secret_seed=shared_secret_seed)
    )
    return _encrypt_payload_for_peer_public_key(
        payload,
        peer_pub=peer_pub,
        peer_id=peer_id,
        request_id=request_id,
        stage_index=stage_index,
        layer_index=layer_index,
        layers=layers,
        purpose=purpose,
    )


def _build_activation_envelope_inner(
    activation: list[float],
    peer_pub: Any,
    *,
    peer_id: str,
    request_id: str,
    stage_index: int,
    level: str = "standard",
) -> ActivationEnvelope:
    _require_crypto()
    normalized_level = _normalize_level(level)
    layers = required_layers_for_level(normalized_level)
    padding_bytes = _padding_bytes_for_level(normalized_level)

    ciphertext = _encode_activation(activation, padding_bytes=padding_bytes)
    nonces: list[bytes] = []
    ephemeral_public_keys: list[bytes] = []
    for layer_index in range(1, layers + 1):
        ciphertext, nonce, eph_public = _encrypt_payload_for_peer_public_key(
            ciphertext,
            peer_pub=peer_pub,
            peer_id=peer_id,
            request_id=request_id,
            stage_index=stage_index,
            layer_index=layer_index,
            layers=layers,
            purpose="activation",
        )
        nonces.append(nonce)
        ephemeral_public_keys.append(eph_public)

    return ActivationEnvelope(
        ciphertext=ciphertext,
        nonces=tuple(nonces),
        ephemeral_public_keys=tuple(ephemeral_public_keys),
        suite=f"{_SUITE_PREFIX};level={normalized_level};layers={layers}",
        layers=layers,
    )


def build_activation_envelope(
    activation: list[float],
    *,
    peer_id: str,
    request_id: str,
    stage_index: int,
    shared_secret_seed: str,
    level: str = "standard",
) -> ActivationEnvelope:
    _require_crypto()
    peer_pub = x25519.X25519PublicKey.from_public_bytes(
        peer_static_public_key(peer_id=peer_id, shared_secret_seed=shared_secret_seed)
    )
    return _build_activation_envelope_inner(
        activation,
        peer_pub,
        peer_id=peer_id,
        request_id=request_id,
        stage_index=stage_index,
        level=level,
    )

def build_activation_envelope_with_pubkey(
    activation: list[float],
    *,
    raw_public_key_bytes: bytes,
    peer_id: str,
    request_id: str,
    stage_index: int,
    level: str = "standard",
) -> ActivationEnvelope:
    _require_crypto()
    peer_pub = x25519.X25519PublicKey.from_public_bytes(bytes(raw_public_key_bytes))
    return _build_activation_envelope_inner(
        activation,
        peer_pub,
        peer_id=peer_id,
        request_id=request_id,
        stage_index=stage_index,
        level=level,
    )


def _decrypt_activation_envelope_inner(
    *,
    ciphertext: bytes,
    nonces: list[bytes] | tuple[bytes, ...],
    ephemeral_public_keys: list[bytes] | tuple[bytes, ...],
    private_key: Any,
    peer_id: str,
    request_id: str,
    stage_index: int,
) -> list[float]:
    _require_crypto()
    nonces_list = list(nonces)
    ephemeral_keys = list(ephemeral_public_keys)
    if not nonces_list:
        raise ValueError("invalid_encrypted_activation: missing_nonces")
    if len(nonces_list) != len(ephemeral_keys):
        raise ValueError("invalid_encrypted_activation: mismatched_layer_metadata")

    layers = len(nonces_list)
    plaintext = bytes(ciphertext)

    for layer_index in range(layers, 0, -1):
        nonce = nonces_list[layer_index - 1]
        peer_ephemeral_pub = x25519.X25519PublicKey.from_public_bytes(ephemeral_keys[layer_index - 1])
        shared_secret = private_key.exchange(peer_ephemeral_pub)
        key = _derive_aes_key(
            shared_secret,
            request_id=request_id,
            stage_index=stage_index,
            layer_index=layer_index,
            purpose="activation",
        )
        plaintext = AESGCM(key).decrypt(
            nonce,
            plaintext,
            _aad(
                request_id=request_id,
                stage_index=stage_index,
                peer_id=peer_id,
                layer_index=layer_index,
                layers=layers,
                purpose="activation",
            ),
        )

    return _decode_activation(plaintext)


def decrypt_activation_envelope_with_privkey(
    *,
    ciphertext: bytes,
    nonces: list[bytes] | tuple[bytes, ...],
    ephemeral_public_keys: list[bytes] | tuple[bytes, ...],
    private_key: Any,
    peer_id: str,
    request_id: str,
    stage_index: int,
) -> list[float]:
    return _decrypt_activation_envelope_inner(
        ciphertext=ciphertext,
        nonces=nonces,
        ephemeral_public_keys=ephemeral_public_keys,
        private_key=private_key,
        peer_id=peer_id,
        request_id=request_id,
        stage_index=stage_index,
    )


def decrypt_activation_envelope(
    *,
    ciphertext: bytes,
    nonces: list[bytes] | tuple[bytes, ...],
    ephemeral_public_keys: list[bytes] | tuple[bytes, ...],
    peer_id: str,
    request_id: str,
    stage_index: int,
    shared_secret_seed: str,
) -> list[float]:
    private_key = _peer_private_key(peer_id=peer_id, shared_secret_seed=shared_secret_seed)
    return _decrypt_activation_envelope_inner(
        ciphertext=ciphertext,
        nonces=nonces,
        ephemeral_public_keys=ephemeral_public_keys,
        private_key=private_key,
        peer_id=peer_id,
        request_id=request_id,
        stage_index=stage_index,
    )


def build_onion_route_envelope(
    route_peer_ids: list[str] | tuple[str, ...],
    *,
    request_id: str,
    shared_secret_seed: str,
) -> OnionRouteEnvelope:
    _require_crypto()
    route = [str(item).strip() for item in list(route_peer_ids) if str(item).strip()]
    if not route:
        raise ValueError("invalid_onion_route:empty")

    remaining_ciphertext = b""
    remaining_nonces: tuple[bytes, ...] = ()
    remaining_ephemeral_keys: tuple[bytes, ...] = ()
    remaining_suite = ""
    remaining_layers = 0

    total_hops = len(route)
    suite = f"{_SUITE_PREFIX};purpose=route_onion;hops={total_hops}"
    for stage_index in range(total_hops - 1, -1, -1):
        peer_id = route[stage_index]
        next_peer_id = route[stage_index + 1] if stage_index + 1 < total_hops else ""
        layer_payload = _encode_route_layer_payload(
            next_peer_id=next_peer_id,
            remaining_ciphertext=remaining_ciphertext,
            remaining_nonces=remaining_nonces,
            remaining_ephemeral_public_keys=remaining_ephemeral_keys,
            remaining_suite=remaining_suite,
            remaining_layers=remaining_layers,
        )
        ciphertext, nonce, eph_public = _encrypt_payload_for_peer(
            layer_payload,
            peer_id=peer_id,
            request_id=request_id,
            stage_index=stage_index,
            shared_secret_seed=shared_secret_seed,
            layer_index=1,
            layers=1,
            purpose="route",
        )
        remaining_ciphertext = ciphertext
        remaining_nonces = (nonce,)
        remaining_ephemeral_keys = (eph_public,)
        remaining_suite = suite
        remaining_layers = total_hops - stage_index

    return OnionRouteEnvelope(
        ciphertext=remaining_ciphertext,
        nonces=remaining_nonces,
        ephemeral_public_keys=remaining_ephemeral_keys,
        suite=suite,
        layers=total_hops,
    )


def build_onion_route_envelope_with_pubkeys(
    route_peer_ids: list[str] | tuple[str, ...],
    *,
    peer_public_keys: dict[str, bytes],
    request_id: str,
) -> OnionRouteEnvelope:
    _require_crypto()
    route = [str(item).strip() for item in list(route_peer_ids) if str(item).strip()]
    if not route:
        raise ValueError("invalid_onion_route:empty")

    remaining_ciphertext = b""
    remaining_nonces: tuple[bytes, ...] = ()
    remaining_ephemeral_keys: tuple[bytes, ...] = ()
    remaining_suite = ""
    remaining_layers = 0

    total_hops = len(route)
    suite = f"{_SUITE_PREFIX};purpose=route_onion;hops={total_hops}"
    for stage_index in range(total_hops - 1, -1, -1):
        peer_id = route[stage_index]
        if peer_id not in peer_public_keys:
            raise KeyError(peer_id)
        peer_pub = x25519.X25519PublicKey.from_public_bytes(bytes(peer_public_keys[peer_id]))
        next_peer_id = route[stage_index + 1] if stage_index + 1 < total_hops else ""
        layer_payload = _encode_route_layer_payload(
            next_peer_id=next_peer_id,
            remaining_ciphertext=remaining_ciphertext,
            remaining_nonces=remaining_nonces,
            remaining_ephemeral_public_keys=remaining_ephemeral_keys,
            remaining_suite=remaining_suite,
            remaining_layers=remaining_layers,
        )
        ciphertext, nonce, eph_public = _encrypt_payload_for_peer_public_key(
            layer_payload,
            peer_pub=peer_pub,
            peer_id=peer_id,
            request_id=request_id,
            stage_index=stage_index,
            layer_index=1,
            layers=1,
            purpose="route",
        )
        remaining_ciphertext = ciphertext
        remaining_nonces = (nonce,)
        remaining_ephemeral_keys = (eph_public,)
        remaining_suite = suite
        remaining_layers = total_hops - stage_index

    return OnionRouteEnvelope(
        ciphertext=remaining_ciphertext,
        nonces=remaining_nonces,
        ephemeral_public_keys=remaining_ephemeral_keys,
        suite=suite,
        layers=total_hops,
    )


def _peel_onion_route_layer_inner(
    *,
    ciphertext: bytes,
    nonces: list[bytes] | tuple[bytes, ...],
    ephemeral_public_keys: list[bytes] | tuple[bytes, ...],
    private_key: Any,
    peer_id: str,
    request_id: str,
    stage_index: int,
) -> OnionRouteLayer:
    _require_crypto()
    nonces_list = list(nonces)
    ephemeral_keys = list(ephemeral_public_keys)
    if not nonces_list:
        raise ValueError("invalid_onion_route:missing_nonces")
    if len(nonces_list) != len(ephemeral_keys):
        raise ValueError("invalid_onion_route:mismatched_layer_metadata")

    plaintext = bytes(ciphertext)
    layers = len(nonces_list)
    for layer_index in range(layers, 0, -1):
        nonce = nonces_list[layer_index - 1]
        peer_ephemeral_pub = x25519.X25519PublicKey.from_public_bytes(ephemeral_keys[layer_index - 1])
        shared_secret = private_key.exchange(peer_ephemeral_pub)
        key = _derive_aes_key(
            shared_secret,
            request_id=request_id,
            stage_index=stage_index,
            layer_index=layer_index,
            purpose="route",
        )
        plaintext = AESGCM(key).decrypt(
            nonce,
            plaintext,
            _aad(
                request_id=request_id,
                stage_index=stage_index,
                peer_id=peer_id,
                layer_index=layer_index,
                layers=layers,
                purpose="route",
            ),
        )

    return _decode_route_layer_payload(plaintext)


def peel_onion_route_layer_with_privkey(
    *,
    ciphertext: bytes,
    nonces: list[bytes] | tuple[bytes, ...],
    ephemeral_public_keys: list[bytes] | tuple[bytes, ...],
    private_key: Any,
    peer_id: str,
    request_id: str,
    stage_index: int,
) -> OnionRouteLayer:
    return _peel_onion_route_layer_inner(
        ciphertext=ciphertext,
        nonces=nonces,
        ephemeral_public_keys=ephemeral_public_keys,
        private_key=private_key,
        peer_id=peer_id,
        request_id=request_id,
        stage_index=stage_index,
    )


def peel_onion_route_layer(
    *,
    ciphertext: bytes,
    nonces: list[bytes] | tuple[bytes, ...],
    ephemeral_public_keys: list[bytes] | tuple[bytes, ...],
    peer_id: str,
    request_id: str,
    stage_index: int,
    shared_secret_seed: str,
) -> OnionRouteLayer:
    private_key = _peer_private_key(peer_id=peer_id, shared_secret_seed=shared_secret_seed)
    return _peel_onion_route_layer_inner(
        ciphertext=ciphertext,
        nonces=nonces,
        ephemeral_public_keys=ephemeral_public_keys,
        private_key=private_key,
        peer_id=peer_id,
        request_id=request_id,
        stage_index=stage_index,
    )
