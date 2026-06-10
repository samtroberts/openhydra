from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import AsyncIterator

import cbor2
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.exceptions import InvalidSignature

from .adapter import (
    SupernodeAdapter,
    PromptRequest,
    TokenChunk,
    ModelInfo,
    BackendStatus,
)
from .attestation_utils import sha256_file, canonical_request_hash, multi_file_manifest_hash

logger = logging.getLogger(__name__)

WEIGHT_EXTENSIONS = frozenset({
    ".safetensors", ".bin", ".gguf", ".ggml",
    ".pt", ".pth", ".npz", ".npy",
})

CONFIG_GLOBS = ("config.json", "tokenizer*", "*.tiktoken", "special_tokens_map.json")


class AttestedRuntime(SupernodeAdapter):
    """Base class for Level-3 attested runtimes.

    Subclasses implement generate/list_models/etc. This base provides:
    - Weight-manifest hashing (_register_weights)
    - Ed25519 output signing (sign_output / verify_output_signature)
    - trust_tier/integration_level overrides
    """

    def __init__(
        self,
        peer_id: str,
        private_key: Ed25519PrivateKey,
        model_id: str,
    ):
        self._peer_id = peer_id
        self._private_key = private_key
        self._public_key = private_key.public_key()
        self._public_key_bytes = self._public_key.public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )
        self._model_id = model_id
        self._weights_hash: str | None = None

    def trust_tier(self) -> str:
        return "attested"

    def integration_level(self) -> int:
        return 3

    def get_weights_hash(self, model_id: str) -> str | None:
        return self._weights_hash

    def _register_weights(self, model_dir: str | Path) -> str:
        """Scan model_dir for weight/config files, compute manifest hash."""
        model_dir = Path(model_dir)
        entries: list[tuple[str, int, str]] = []

        for path in sorted(model_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in WEIGHT_EXTENSIONS:
                entries.append((path.name, path.stat().st_size, sha256_file(path)))
            else:
                for glob in CONFIG_GLOBS:
                    if path.match(glob):
                        entries.append((path.name, path.stat().st_size, sha256_file(path)))
                        break

        if not entries:
            raise ValueError(f"No weight/config files found in {model_dir}")

        self._weights_hash = multi_file_manifest_hash(entries)
        logger.info(
            "weights_registered model=%s files=%d hash=%s",
            self._model_id, len(entries), self._weights_hash[:16],
        )
        return self._weights_hash

    def sign_output(
        self,
        request: PromptRequest,
        model_id: str,
        output_token_ids: list[int],
        timestamp_ms: int,
    ) -> bytes:
        """Ed25519-sign a canonical CBOR payload binding request+output+identity+time."""
        request_hash = canonical_request_hash(
            model_id=model_id,
            prompt=request.prompt,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            system_prompt=request.system_prompt,
        )

        output_hash = hashlib.sha256(
            cbor2.dumps(output_token_ids, canonical=True)
        ).hexdigest()

        payload = {
            "v": 1,
            "peer_id": self._peer_id,
            "request_id": request.request_id,
            "request_hash": request_hash,
            "model_id": model_id,
            "weights_hash": self._weights_hash or "",
            "output_token_hash": output_hash,
            "completion_tokens": len(output_token_ids),
            "timestamp_ms": timestamp_ms,
        }
        payload_cbor = cbor2.dumps(payload, canonical=True)
        return self._private_key.sign(payload_cbor)

    @staticmethod
    def verify_output_signature(
        payload_cbor: bytes,
        signature: bytes,
        public_key_bytes: bytes,
    ) -> bool:
        """Verify an output signature from any attested runtime."""
        try:
            pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            pub.verify(signature, payload_cbor)
            return True
        except (InvalidSignature, ValueError):
            return False
