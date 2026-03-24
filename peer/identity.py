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
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    PrivateFormat,
    NoEncryption,
)


def generate_keypair() -> tuple[Ed25519PrivateKey, bytes]:
    """Generate a new Ed25519 keypair. Returns (private_key, public_key_bytes)."""
    private_key = Ed25519PrivateKey.generate()
    public_key_bytes = private_key.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    return private_key, public_key_bytes


def peer_id_from_public_key(public_key_bytes: bytes) -> str:
    """Derive a 16-char hex peer_id from the raw Ed25519 public key bytes."""
    return hashlib.sha256(public_key_bytes).hexdigest()[:16]


def load_or_create_identity(identity_path: str) -> dict:
    """Load JSON identity from path or create a new keypair and persist it.

    File format: {"public_key": "<hex>", "private_key": "<hex>", "peer_id": "<str>"}
    File permissions are set to 0600 on creation.

    Returns a dict with keys:
        "private_key"    - Ed25519PrivateKey object
        "public_key_hex" - hex string of the 32-byte raw public key
        "peer_id"        - 16-char hex derived from the public key
    """
    path = Path(identity_path)
    if path.exists():
        raw = json.loads(path.read_text())
        private_key_bytes = bytes.fromhex(raw["private_key"])
        private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        public_key_hex = str(raw["public_key"])
        peer_id = str(raw["peer_id"])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        private_key, public_key_bytes = generate_keypair()
        public_key_hex = public_key_bytes.hex()
        peer_id = peer_id_from_public_key(public_key_bytes)
        raw_private_bytes = private_key.private_bytes(
            encoding=Encoding.Raw,
            format=PrivateFormat.Raw,
            encryption_algorithm=NoEncryption(),
        )
        data = {
            "public_key": public_key_hex,
            "private_key": raw_private_bytes.hex(),
            "peer_id": peer_id,
        }
        path.write_text(json.dumps(data))
        os.chmod(path, 0o600)

    return {
        "private_key": private_key,
        "public_key_hex": public_key_hex,
        "peer_id": peer_id,
    }


def sign_announce(
    private_key: Ed25519PrivateKey,
    peer_id: str,
    host: str,
    port: int,
    model_id: str,
) -> str:
    """Sign an announce payload and return the base64url-encoded signature."""
    message = json.dumps(
        {"host": host, "model_id": model_id, "peer_id": peer_id, "port": port},
        sort_keys=True,
    ).encode()
    signature = private_key.sign(message)
    return base64.urlsafe_b64encode(signature).decode()


def verify_announce(
    public_key_hex: str,
    peer_id: str,
    host: str,
    port: int,
    model_id: str,
    signature_b64: str,
) -> bool:
    """Verify an announce signature. Returns False (never raises) on any error."""
    try:
        public_key_bytes = bytes.fromhex(public_key_hex)
        pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        message = json.dumps(
            {"host": host, "model_id": model_id, "peer_id": peer_id, "port": port},
            sort_keys=True,
        ).encode()
        signature = base64.urlsafe_b64decode(signature_b64.encode())
        pub.verify(signature, message)
        return True
    except Exception:
        return False
