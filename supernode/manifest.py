from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import cbor2
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)
from cryptography.exceptions import InvalidSignature


MANIFEST_TTL_MS = 5 * 60 * 1000
MANIFEST_REFRESH_S = 5 * 60


@dataclass
class ModelCapability:
    model_id: str
    model_family: str
    parameter_count: int
    quantization: str
    context_length: int
    supports_streaming: bool = True
    supports_system_prompt: bool = True
    warm: bool = False
    estimated_tps: float = 0.0
    weights_hash: str = ""
    weights_size: int = 0


@dataclass
class HardwareInfo:
    accelerator: str = "cpu"
    gpu_name: str = ""
    gpu_memory_mb: int = 0
    gpu_memory_free_mb: int = 0
    cpu_cores: int = 0
    ram_mb: int = 0


@dataclass
class SupernodeManifest:
    peer_id: str
    libp2p_peer_id: str
    backend_type: str
    version: str = "0.1.0"

    integration_level: int = 1
    trust_tier: str = "unverified"
    binary_hash: str = ""

    models: list[ModelCapability] = field(default_factory=list)
    max_concurrent: int = 4
    max_context_length: int = 4096

    hardware: HardwareInfo = field(default_factory=HardwareInfo)

    listen_addrs: list[str] = field(default_factory=list)
    nat_status: str = "unknown"
    region: str = ""

    timestamp: int = 0
    signature: bytes = b""
    public_key: bytes = b""

    def to_signable_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("signature", None)
        d.pop("public_key", None)
        return d

    def to_cbor(self) -> bytes:
        return cbor2.dumps(asdict(self), canonical=True)

    def to_signable_cbor(self) -> bytes:
        return cbor2.dumps(self.to_signable_dict(), canonical=True)

    @classmethod
    def from_cbor(cls, data: bytes) -> SupernodeManifest:
        d = cbor2.loads(data)
        models = [ModelCapability(**m) for m in d.pop("models", [])]
        hw = HardwareInfo(**d.pop("hardware", {}))
        return cls(models=models, hardware=hw, **d)

    def sign(self, private_key: Ed25519PrivateKey) -> None:
        self.timestamp = int(time.time() * 1000)
        self.public_key = private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )
        payload = self.to_signable_cbor()
        self.signature = private_key.sign(payload)

    def verify_signature(self) -> bool:
        if not self.signature or not self.public_key:
            return False
        try:
            pub = Ed25519PublicKey.from_public_bytes(self.public_key)
            pub.verify(self.signature, self.to_signable_cbor())
            return True
        except (InvalidSignature, ValueError):
            return False

    def is_fresh(self, now_ms: int | None = None) -> bool:
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        return (now_ms - self.timestamp) < MANIFEST_TTL_MS

    def model_ids(self) -> list[str]:
        return [m.model_id for m in self.models]


# ---------------------------------------------------------------------------
# DHT key helpers
# ---------------------------------------------------------------------------

SUPERNODE_KEY_PREFIX = "/openhydra/supernode/"
MODEL_PROVIDER_PREFIX = "/openhydra/model/"


def supernode_record_key(libp2p_peer_id: str) -> str:
    return f"{SUPERNODE_KEY_PREFIX}{libp2p_peer_id}"


def model_provider_key(model_id: str) -> str:
    return f"{MODEL_PROVIDER_PREFIX}{normalize_model_id(model_id)}/provider"


# ---------------------------------------------------------------------------
# Model ID normalization
# ---------------------------------------------------------------------------

_NORMALIZE_RE = re.compile(r"[/_\s]+")


def normalize_model_id(model_id: str) -> str:
    s = model_id.lower().strip()
    s = _NORMALIZE_RE.sub("-", s)
    s = s.replace(":", "-")
    for prefix in ("meta-llama-", "meta-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s
