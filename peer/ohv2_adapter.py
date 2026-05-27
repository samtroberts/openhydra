"""OHV2 wire format adapter classes for protobuf-compatible attribute access.

These classes wrap decoded CBOR dicts from P2PNode.decode_forward_msg() and
provide the same attribute interface as peer_pb2.ForwardRequest / ForwardResponse.
This lets Forward() (~600 lines) work unchanged while the serialization boundary
switches from protobuf to OHV2 binary format.

Usage (receiver side):
    header_dict, act_bytes, msg_type = P2PNode.decode_forward_msg(payload)
    request = OHV2Request(header_dict, act_bytes)
    response = service.Forward(request, context=None)

Usage (sender side):
    header = OHV2Request.to_header_dict(request_id=..., stage_index=..., ...)
    wire = P2PNode.encode_forward_msg(header, activation_packed, msg_type=0)
"""

from __future__ import annotations

import json as _json
from typing import Any


class _DictHop:
    """Attribute-access wrapper for a PeerHop dict decoded from JSON bytes.

    Provides ``.address``, ``.peer_id``, ``.libp2p_peer_id``,
    ``.shard_layer_start``, ``.shard_layer_end``, ``.shard_total_layers``
    exactly like a ``peer_pb2.PeerHop`` protobuf message.
    """
    __slots__ = ("_d",)

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, name: str):
        try:
            return self._d[name]
        except KeyError:
            return 0 if name.startswith("shard_") else ""

    def __repr__(self) -> str:
        return f"_DictHop({self._d!r})"


class OHV2Request:
    """Drop-in replacement for peer_pb2.ForwardRequest using OHV2 decoded dict.

    Provides protobuf-compatible attribute access (request.field_name) and
    getattr(request, "field", default) patterns used throughout Forward().

    The underlying dict (_d) can also be mutated directly for the push-path
    dict-copy optimization (avoiding 40-line protobuf field-by-field copy).
    """

    __slots__ = ("_d", "_activation_bytes")

    def __init__(self, header_dict: dict, activation_bytes: bytes = b""):
        self._d = header_dict
        self._activation_bytes = activation_bytes

    # ── Core routing ─────────────────────────────────────────────────
    @property
    def request_id(self) -> str:
        return self._d.get("request_id", "")

    @property
    def stage_index(self) -> int:
        return int(self._d.get("stage_index", 0))

    @property
    def total_stages(self) -> int:
        return int(self._d.get("total_stages", 0))

    @property
    def push_mode(self) -> bool:
        return bool(self._d.get("push_mode", False))

    @property
    def next_hop_address(self) -> str:
        return self._d.get("next_hop_address", "")

    @property
    def next_hop_peer_id(self) -> str:
        return self._d.get("next_hop_peer_id", "")

    # ── Activation ───────────────────────────────────────────────────
    @property
    def activation_packed(self) -> bytes:
        return self._activation_bytes

    @property
    def activation(self) -> list:
        """OHV2 always uses packed activation bytes; the float list is empty."""
        return []

    @property
    def quantized_activation(self) -> bytes:
        # INT8 quantized data is carried in the activation payload when
        # activation_dtype == Int8. The header's quantized_scales field
        # carries the scale factors.
        return b""

    @property
    def quantized_scales(self) -> list:
        return self._d.get("quantized_scales", [])

    @property
    def activation_quantization(self) -> str:
        # OHV2 uses activation_dtype enum, not a string field.
        # Map Int8 dtype to "int8" string for backward compat.
        dtype = self._d.get("activation_dtype", "Fp32")
        if dtype == "Int8" or dtype == 2:
            return "int8"
        return ""

    # ── Layer sharding ───────────────────────────────────────────────
    @property
    def shard_layer_start(self) -> int:
        return int(self._d.get("shard_layer_start", 0))

    @property
    def shard_layer_end(self) -> int:
        return int(self._d.get("shard_layer_end", 0))

    @property
    def shard_total_layers(self) -> int:
        return int(self._d.get("shard_total_layers", 0))

    # ── KV cache ─────────────────────────────────────────────────────
    @property
    def kv_session_id(self) -> str:
        return self._d.get("kv_session_id", "")

    @property
    def kv_store_activation(self) -> bool:
        return bool(self._d.get("kv_store_activation", False))

    @property
    def kv_use_cached_activation(self) -> bool:
        return bool(self._d.get("kv_use_cached_activation", False))

    @property
    def kv_rollback_to(self) -> int:
        return int(self._d.get("kv_rollback_to", 0))

    # ── Decode parameters ────────────────────────────────────────────
    @property
    def decode_do_sample(self) -> bool:
        return bool(self._d.get("decode_do_sample", False))

    @property
    def decode_temperature(self) -> float:
        return float(self._d.get("decode_temperature", 0.0))

    @property
    def decode_top_p(self) -> float:
        return float(self._d.get("decode_top_p", 0.0))

    @property
    def decode_top_k(self) -> int:
        return int(self._d.get("decode_top_k", 0))

    @property
    def decode_seed(self) -> int:
        return int(self._d.get("decode_seed", 0))

    @property
    def sample_on_coordinator(self) -> bool:
        return bool(self._d.get("sample_on_coordinator", False))

    # ── Pipeline / speculative ───────────────────────────────────────
    @property
    def slot_id(self) -> int:
        return int(self._d.get("slot_id", 0))

    @property
    def pipeline_depth(self) -> int:
        return int(self._d.get("pipeline_depth", 0))

    @property
    def draft_block(self) -> bool:
        return bool(self._d.get("draft_block", False))

    @property
    def block_index(self) -> int:
        return int(self._d.get("block_index", 0))

    @property
    def draft_token_ids(self) -> list:
        return list(self._d.get("draft_token_ids", []))

    @property
    def verify_batch_size(self) -> int:
        return int(self._d.get("verify_batch_size", 0))

    # ── Ring autoregressive ──────────────────────────────────────────
    @property
    def ring_mode(self) -> bool:
        return bool(self._d.get("ring_mode", False))

    @property
    def ring_tokens_remaining(self) -> int:
        return int(self._d.get("ring_tokens_remaining", 0))

    @property
    def ring_generated_ids(self) -> list:
        return list(self._d.get("ring_generated_ids", []))

    @property
    def ring_eos_ids(self) -> list:
        return list(self._d.get("ring_eos_ids", []))

    @property
    def ring_first_hop_address(self) -> str:
        return self._d.get("ring_first_hop_address", "")

    @property
    def ring_first_hop_peer_id(self) -> str:
        return self._d.get("ring_first_hop_peer_id", "")

    @property
    def ring_first_hop_libp2p_id(self) -> str:
        return self._d.get("ring_first_hop_libp2p_id", "")

    @property
    def ring_full_route(self) -> list:
        """Decode ring_full_route from opaque bytes to PeerHop-like dicts.

        Same encoding as remaining_route: JSON-serialized list of
        ``{address, peer_id, libp2p_peer_id, shard_layer_start, ...}``
        packed into ``Vec<u8>``.
        """
        raw = self._d.get("ring_full_route", [])
        if isinstance(raw, bytes):
            try:
                return [_DictHop(h) for h in _json.loads(raw)]
            except Exception:
                return []
        if isinstance(raw, list) and raw and isinstance(raw[0], int):
            try:
                return [_DictHop(h) for h in _json.loads(bytes(raw))]
            except Exception:
                return []
        return list(raw)

    # ── Callback routing ─────────────────────────────────────────────
    @property
    def final_callback_address(self) -> str:
        return self._d.get("final_callback_address", "")

    @property
    def final_callback_request_id(self) -> str:
        return self._d.get("final_callback_request_id", "")

    @property
    def final_callback_libp2p_peer_id(self) -> str:
        return self._d.get("final_callback_libp2p_peer_id", "")

    @property
    def remaining_route(self) -> list:
        """Remaining route as list of PeerHopEntry-like objects.

        In OHV2, remaining_route is serialized as opaque bytes (JSON of
        [{address, peer_id, libp2p_peer_id, shard_layer_start, ...}]).
        Decode on access and wrap in _DictHop for attribute access.
        """
        raw = self._d.get("remaining_route", [])
        if isinstance(raw, bytes):
            try:
                return [_DictHop(h) for h in _json.loads(raw)]
            except Exception:
                return []
        if isinstance(raw, list) and raw and isinstance(raw[0], int):
            # Raw byte array from CBOR — decode as JSON bytes
            try:
                return [_DictHop(h) for h in _json.loads(bytes(raw))]
            except Exception:
                return []
        return list(raw)

    # ── Prompt ───────────────────────────────────────────────────────
    @property
    def prompt(self) -> str:
        # OHV2 doesn't carry the prompt text field (only prompt_token_ids).
        # Prompt is only used at stage 0 and is set separately.
        return self._d.get("prompt", "")

    @property
    def prompt_token_ids(self) -> list:
        return list(self._d.get("prompt_token_ids", []))

    @property
    def max_tokens(self) -> int:
        return int(self._d.get("max_tokens", 0))

    # ── Encryption ───────────────────────────────────────────────────
    @property
    def encrypted_activation(self) -> bytes:
        raw = self._d.get("encrypted_activation", [])
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, list):
            return bytes(raw)
        return b""

    @property
    def encryption_nonces(self) -> list:
        return list(self._d.get("encryption_nonces", []))

    @property
    def encryption_ephemeral_public_keys(self) -> list:
        return list(self._d.get("encryption_ephemeral_keys", []))

    @property
    def encryption_suite(self) -> str:
        v = self._d.get("encryption_suite", 0)
        return str(v) if v else ""

    @property
    def encryption_layers(self) -> int:
        return int(self._d.get("encryption_layers", 0))

    # ── Compression ──────────────────────────────────────────────────
    @property
    def compression_codec(self) -> str:
        return self._d.get("compression_codec", "")

    @property
    def compression_original_dim(self) -> int:
        return int(self._d.get("compression_original_dim", 0))

    @property
    def compression_latent_dim(self) -> int:
        return int(self._d.get("compression_latent_dim", 0))

    # ── Onion routing ────────────────────────────────────────────────
    @property
    def onion_route_ciphertext(self) -> bytes:
        raw = self._d.get("onion_route_ciphertext", [])
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, list):
            return bytes(raw)
        return b""

    @property
    def onion_route_nonces(self) -> list:
        return list(self._d.get("onion_route_nonces", []))

    @property
    def onion_route_ephemeral_public_keys(self) -> list:
        return list(self._d.get("onion_route_ephemeral_public_keys", []))

    @property
    def onion_route_suite(self) -> str:
        return self._d.get("onion_route_suite", "")

    @property
    def onion_route_layers(self) -> int:
        return int(self._d.get("onion_route_layers", 0))

    # ── Geo ──────────────────────────────────────────────────────────
    @property
    def geo_claimed_region(self) -> str:
        return self._d.get("geo_claimed_region", "")

    @property
    def geo_nonce(self) -> bytes:
        raw = self._d.get("geo_nonce", [])
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, list):
            return bytes(raw)
        return b""

    # ── Generic attribute access ─────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        """Fallback for getattr(request, "field", default) patterns."""
        if name.startswith("_"):
            raise AttributeError(name)
        return self._d.get(name, _FIELD_DEFAULTS.get(name, ""))


class OHV2Response:
    """Drop-in replacement for peer_pb2.ForwardResponse using OHV2 decoded dict."""

    __slots__ = ("_d", "_activation_bytes")

    def __init__(self, header_dict: dict, activation_bytes: bytes = b""):
        self._d = header_dict
        self._activation_bytes = activation_bytes

    @property
    def request_id(self) -> str:
        return self._d.get("request_id", "")

    @property
    def peer_id(self) -> str:
        return self._d.get("peer_id", "")

    @property
    def activation(self) -> list:
        return []

    @property
    def activation_packed(self) -> bytes:
        return self._activation_bytes

    @property
    def stage_index(self) -> int:
        return int(self._d.get("stage_index", 0))

    @property
    def error(self) -> str:
        return self._d.get("error_message", "") or self._d.get("error", "")

    @property
    def kv_cache_hit(self) -> bool:
        status = self._d.get("status", "Ok")
        if status == "KvCacheHit" or status == 2:
            return True
        return bool(self._d.get("kv_cache_hit", False))

    @property
    def activation_hash(self) -> str:
        return self._d.get("activation_hash", "")

    @property
    def is_hidden_state(self) -> bool:
        return bool(self._d.get("is_hidden_state", False))

    @property
    def slot_id(self) -> int:
        return int(self._d.get("slot_id", 0))

    @property
    def block_size(self) -> int:
        return int(self._d.get("block_size", 0))

    @property
    def block_index(self) -> int:
        return int(self._d.get("block_index", 0))

    @property
    def metadata_json(self) -> str:
        return self._d.get("metadata_json", "")

    @property
    def onion_next_peer_id(self) -> str:
        return self._d.get("onion_next_peer_id", "")

    @property
    def onion_route_ciphertext(self) -> bytes:
        raw = self._d.get("onion_route_ciphertext", [])
        if isinstance(raw, bytes):
            return raw
        if isinstance(raw, list):
            return bytes(raw)
        return b""

    @property
    def onion_route_nonces(self) -> list:
        return list(self._d.get("onion_route_nonces", []))

    @property
    def onion_route_ephemeral_public_keys(self) -> list:
        return list(self._d.get("onion_route_ephemeral_public_keys", []))

    @property
    def onion_route_suite(self) -> str:
        return self._d.get("onion_route_suite", "")

    @property
    def onion_route_layers(self) -> int:
        return int(self._d.get("onion_route_layers", 0))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self._d.get(name, _FIELD_DEFAULTS.get(name, ""))


# Default values for fields not explicitly covered by properties.
# Used by __getattr__ fallback for getattr(request, "field", default) patterns.
_FIELD_DEFAULTS: dict[str, Any] = {
    # Strings
    "request_id": "",
    "prompt": "",
    "kv_session_id": "",
    "next_hop_address": "",
    "next_hop_peer_id": "",
    "final_callback_address": "",
    "final_callback_request_id": "",
    "final_callback_libp2p_peer_id": "",
    "ring_first_hop_address": "",
    "ring_first_hop_peer_id": "",
    "ring_first_hop_libp2p_id": "",
    "peer_id": "",
    "error": "",
    "error_message": "",
    "metadata_json": "",
    "activation_hash": "",
    "onion_next_peer_id": "",
    "geo_claimed_region": "",
    "encryption_suite": "",
    "compression_codec": "",
    "activation_quantization": "",
    # Integers
    "stage_index": 0,
    "total_stages": 0,
    "max_tokens": 0,
    "shard_layer_start": 0,
    "shard_layer_end": 0,
    "shard_total_layers": 0,
    "kv_rollback_to": 0,
    "decode_top_k": 0,
    "decode_seed": 0,
    "slot_id": 0,
    "pipeline_depth": 0,
    "block_index": 0,
    "block_size": 0,
    "verify_batch_size": 0,
    "ring_tokens_remaining": 0,
    "compression_original_dim": 0,
    "compression_latent_dim": 0,
    "encryption_layers": 0,
    "onion_route_layers": 0,
    # Floats
    "decode_temperature": 0.0,
    "decode_top_p": 0.0,
    # Booleans
    "push_mode": False,
    "kv_store_activation": False,
    "kv_use_cached_activation": False,
    "decode_do_sample": False,
    "sample_on_coordinator": False,
    "ring_mode": False,
    "draft_block": False,
    "is_hidden_state": False,
    "kv_cache_hit": False,
    # Lists
    "activation": [],
    "prompt_token_ids": [],
    "ring_generated_ids": [],
    "ring_eos_ids": [],
    "ring_full_route": [],
    "remaining_route": [],
    "draft_token_ids": [],
    "quantized_scales": [],
    "encryption_nonces": [],
    "encryption_ephemeral_public_keys": [],
    "onion_route_nonces": [],
    "onion_route_ephemeral_public_keys": [],
    # Bytes
    "activation_packed": b"",
    "quantized_activation": b"",
    "encrypted_activation": b"",
    "onion_route_ciphertext": b"",
    "geo_nonce": b"",
}
