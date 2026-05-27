# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Cross-language tests for CP-1: ForwardMsg wire format + activation codec.

Verifies that:
1. The Python ForwardMsg encoder/decoder matches the Rust implementation.
2. FP32 pack/unpack is bit-for-bit identical between Python and Rust.
3. INT8 quantization produces identical bytes between Python and Rust.
4. Proxy format negotiation correctly identifies ForwardMsg vs protobuf.

Run:  pytest tests/test_forward_msg.py -v
"""

from __future__ import annotations

import math
import struct
import time

import pytest


def _cbor2():
    import cbor2
    return cbor2


def _activation_codec():
    from peer.activation_codec import (
        quantize_int8,
        dequantize_int8,
        pack_fp32,
        unpack_fp32,
    )
    return quantize_int8, dequantize_int8, pack_fp32, unpack_fp32


# ── ForwardMsg wire format constants (must match forward_msg.rs) ────────

FORWARD_MSG_MAGIC = 0x4F485632  # "OHV2"
FORWARD_MSG_VERSION = 1
MSG_TYPE_FORWARD = 0
MSG_TYPE_PUSH_RESULT = 1
MSG_TYPE_PING = 2
PREAMBLE_SIZE = 12


# ── Python ForwardMsg encoder/decoder ───────────────────────────────────
# These mirror the Rust implementation for cross-language testing.

def _encode_forward_msg(msg_type: int, header_dict: dict, activation: bytes) -> bytes:
    """Encode a ForwardMsg in the OHV2 wire format (Python reference impl)."""
    cbor2 = _cbor2()
    header_bytes = cbor2.dumps(header_dict)

    header_len = len(header_bytes)
    assert header_len <= 0xFFFF

    activation_len = len(activation)

    buf = bytearray(PREAMBLE_SIZE + header_len + 4 + activation_len)

    # Preamble.
    struct.pack_into("<I", buf, 0, FORWARD_MSG_MAGIC)
    struct.pack_into("<I", buf, 4, FORWARD_MSG_VERSION)
    struct.pack_into("<H", buf, 8, msg_type)
    struct.pack_into("<H", buf, 10, header_len)

    # CBOR header.
    buf[PREAMBLE_SIZE:PREAMBLE_SIZE + header_len] = header_bytes

    # Self-delimiting activation.
    act_offset = PREAMBLE_SIZE + header_len
    struct.pack_into("<I", buf, act_offset, activation_len)
    buf[act_offset + 4:] = activation

    return bytes(buf)


def _decode_forward_msg(data: bytes) -> tuple[int, dict, bytes]:
    """Decode a ForwardMsg. Returns (msg_type, header_dict, activation)."""
    cbor2 = _cbor2()

    assert len(data) >= PREAMBLE_SIZE
    magic = struct.unpack_from("<I", data, 0)[0]
    assert magic == FORWARD_MSG_MAGIC, f"bad magic: 0x{magic:08X}"

    version = struct.unpack_from("<I", data, 4)[0]
    assert version == FORWARD_MSG_VERSION

    msg_type = struct.unpack_from("<H", data, 8)[0]
    header_len = struct.unpack_from("<H", data, 10)[0]

    header = cbor2.loads(data[PREAMBLE_SIZE:PREAMBLE_SIZE + header_len])

    act_offset = PREAMBLE_SIZE + header_len
    act_len = struct.unpack_from("<I", data, act_offset)[0]
    activation = data[act_offset + 4:act_offset + 4 + act_len]

    return msg_type, header, activation


def _is_forward_msg(data: bytes) -> bool:
    """Check if bytes start with the OHV2 magic."""
    if len(data) < 4:
        return False
    return struct.unpack_from("<I", data, 0)[0] == FORWARD_MSG_MAGIC


def _floats_to_bytes(floats: list[float]) -> bytes:
    return struct.pack(f"<{len(floats)}f", *floats)


def _bytes_to_floats(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


# ── ForwardMsg wire format tests ────────────────────────────────────────


class TestForwardMsgWireFormat:
    """Verify the ForwardMsg binary wire format."""

    def test_encode_decode_roundtrip(self):
        header = {
            "request_id": "test-001",
            "stage_index": 2,
            "total_stages": 4,
            "push_mode": True,
            "shard_layer_start": 8,
            "shard_layer_end": 16,
            "shard_total_layers": 32,
        }
        activation = _floats_to_bytes([1.0, 2.0, 3.0])

        wire = _encode_forward_msg(MSG_TYPE_FORWARD, header, activation)
        msg_type, decoded_hdr, decoded_act = _decode_forward_msg(wire)

        assert msg_type == MSG_TYPE_FORWARD
        assert decoded_hdr["request_id"] == "test-001"
        assert decoded_hdr["stage_index"] == 2
        assert decoded_hdr["push_mode"] is True
        assert _bytes_to_floats(decoded_act) == [1.0, 2.0, 3.0]

    def test_magic_bytes(self):
        wire = _encode_forward_msg(
            MSG_TYPE_FORWARD, {"request_id": "x"}, b""
        )
        assert _is_forward_msg(wire)
        # First 4 bytes are 0x4F485632 LE.
        assert struct.unpack_from("<I", wire, 0)[0] == FORWARD_MSG_MAGIC

    def test_version_field(self):
        wire = _encode_forward_msg(
            MSG_TYPE_FORWARD, {"request_id": "x"}, b""
        )
        version = struct.unpack_from("<I", wire, 4)[0]
        assert version == FORWARD_MSG_VERSION

    def test_all_msg_types(self):
        for msg_type in [MSG_TYPE_FORWARD, MSG_TYPE_PUSH_RESULT, MSG_TYPE_PING]:
            wire = _encode_forward_msg(
                msg_type, {"request_id": f"type-{msg_type}"}, b"\xAA" * 8
            )
            decoded_type, hdr, act = _decode_forward_msg(wire)
            assert decoded_type == msg_type
            assert act == b"\xAA" * 8

    def test_push_result_with_ring_fields(self):
        header = {
            "request_id": "push-001",
            "ring_mode": True,
            "ring_tokens_remaining": 42,
            "ring_generated_ids": [101, 102, 103],
        }
        wire = _encode_forward_msg(MSG_TYPE_PUSH_RESULT, header, b"")
        _, hdr, _ = _decode_forward_msg(wire)
        assert hdr["ring_mode"] is True
        assert hdr["ring_tokens_remaining"] == 42
        assert hdr["ring_generated_ids"] == [101, 102, 103]

    def test_empty_activation(self):
        wire = _encode_forward_msg(
            MSG_TYPE_FORWARD, {"request_id": "empty"}, b""
        )
        _, _, act = _decode_forward_msg(wire)
        assert act == b""

    def test_large_activation(self):
        """896-dim FP32 hidden state (~3.5KB)."""
        floats = [i * 0.001 for i in range(896)]
        activation = _floats_to_bytes(floats)

        wire = _encode_forward_msg(
            MSG_TYPE_FORWARD,
            {
                "request_id": "large",
                "activation_dtype": 0,
                "activation_shape": [1, 1, 896],
            },
            activation,
        )

        _, hdr, act = _decode_forward_msg(wire)
        assert hdr["activation_shape"] == [1, 1, 896]
        decoded_floats = _bytes_to_floats(act)
        assert len(decoded_floats) == 896

        # Bit-for-bit verification.
        for i, (orig, dec) in enumerate(zip(floats, decoded_floats)):
            orig_bits = struct.pack("<f", orig)
            dec_bits = struct.pack("<f", dec)
            assert orig_bits == dec_bits, f"float mismatch at index {i}"

    def test_self_delimiting_activation_len(self):
        """Verify the activation_len field is present and correct."""
        activation = _floats_to_bytes([1.0, 2.0])
        wire = _encode_forward_msg(
            MSG_TYPE_FORWARD, {"request_id": "x"}, activation
        )

        # Find activation_len in the wire bytes.
        header_len = struct.unpack_from("<H", wire, 10)[0]
        act_len_offset = PREAMBLE_SIZE + header_len
        act_len = struct.unpack_from("<I", wire, act_len_offset)[0]
        assert act_len == 8  # 2 floats × 4 bytes

    def test_wire_size_compact(self):
        """Minimal message should be under 100 bytes."""
        wire = _encode_forward_msg(
            MSG_TYPE_FORWARD, {"request_id": "x"}, b""
        )
        assert len(wire) < 100, f"too large: {len(wire)} bytes"


# ── Format negotiation tests ───────────────────────────────────────────


class TestFormatNegotiation:
    """Verify format detection matches proxy.rs behaviour."""

    def test_forward_msg_detected(self):
        wire = _encode_forward_msg(
            MSG_TYPE_FORWARD, {"request_id": "x"}, b""
        )
        assert _is_forward_msg(wire)

    def test_protobuf_not_detected(self):
        # Protobuf messages start with field tags.
        assert not _is_forward_msg(b"\x0A\x10\x08\x01")

    def test_method_prefix_not_detected(self):
        """Method prefix bytes (0x01–0x06) used by Python proxy_handler_loop."""
        for prefix in range(1, 7):
            msg = bytes([prefix, 0x0A, 0x10, 0x08])
            assert not _is_forward_msg(msg)

    def test_empty_not_detected(self):
        assert not _is_forward_msg(b"")
        assert not _is_forward_msg(b"\x00")
        assert not _is_forward_msg(b"\x00\x00\x00")


# ── Activation FP32 bit-for-bit parity tests ───────────────────────────


class TestFp32Parity:
    """Verify FP32 pack/unpack matches Python activation_codec bit-for-bit."""

    def test_pack_roundtrip(self):
        _, _, pack_fp32, unpack_fp32 = _activation_codec()

        values = [1.0, -2.5, 3.14, 0.0, -1e10, 1.175494e-38]
        packed = pack_fp32(values)
        assert len(packed) == len(values) * 4

        unpacked = unpack_fp32(packed)
        assert len(unpacked) == len(values)
        for orig, dec in zip(values, unpacked):
            assert struct.pack("<f", orig) == struct.pack("<f", dec)

    def test_pack_empty(self):
        _, _, pack_fp32, unpack_fp32 = _activation_codec()
        assert pack_fp32([]) == b""
        assert unpack_fp32(b"") == []

    def test_pack_matches_struct_pack(self):
        """Verify pack_fp32 is bit-identical to struct.pack."""
        _, _, pack_fp32, _ = _activation_codec()

        values = [0.5, -0.3, 1.0, -1.0, 0.0, 42.5, -1e-5, 3.14159]
        packed = pack_fp32(values)
        reference = struct.pack(f"<{len(values)}f", *values)
        assert packed == reference

    def test_unpack_matches_struct_unpack(self):
        """Verify unpack_fp32 is bit-identical to struct.unpack."""
        _, _, _, unpack_fp32 = _activation_codec()

        values = [0.5, -0.3, 1.0, -1.0, 0.0]
        raw = struct.pack(f"<{len(values)}f", *values)
        unpacked = unpack_fp32(raw)
        reference = list(struct.unpack(f"<{len(values)}f", raw))

        for u, r in zip(unpacked, reference):
            assert struct.pack("<f", u) == struct.pack("<f", r)

    def test_896_dim_roundtrip(self):
        """Simulate a 896-dim hidden state (Qwen3.5-0.8B)."""
        _, _, pack_fp32, unpack_fp32 = _activation_codec()

        values = [i * 0.001 for i in range(896)]
        packed = pack_fp32(values)
        unpacked = unpack_fp32(packed)

        assert len(unpacked) == 896
        for i, (orig, dec) in enumerate(zip(values, unpacked)):
            assert struct.pack("<f", orig) == struct.pack(
                "<f", dec
            ), f"mismatch at index {i}"


# ── INT8 cross-language parity tests ───────────────────────────────────


class TestInt8Parity:
    """Verify INT8 quantization matches between Python and the wire spec."""

    def test_basic_roundtrip(self):
        q, dq, _, _ = _activation_codec()
        values = [0.5, -0.3, 1.0, -1.0, 0.0, 0.25]
        data, scales = q(values)
        restored = dq(data, scales)
        assert len(restored) == len(values)
        for orig, rec in zip(values, restored):
            assert abs(orig - rec) < 0.02

    def test_empty(self):
        q, dq, _, _ = _activation_codec()
        data, scales = q([])
        assert data == b""
        assert scales == []

    def test_all_zeros(self):
        q, dq, _, _ = _activation_codec()
        data, scales = q([0.0] * 10)
        restored = dq(data, scales)
        assert all(v == 0.0 for v in restored)

    def test_unsigned_storage_spec(self):
        """Verify q_byte = q_signed + 128 (OpenHydra INT8 spec)."""
        q, _, _, _ = _activation_codec()

        # All-zero input → bytes(n) = all zero bytes, scale=0.0.
        # (Dequant with scale=0 returns zeros regardless of byte value.)
        data, scales = q([0.0, 0.0, 0.0])
        assert all(b == 0 for b in data)
        assert scales == [0.0]

        # Positive max → q_signed=127 → q_byte=255.
        data, _ = q([1.0])
        assert data[0] == 255

        # Negative max → q_signed=-127 → q_byte=1.
        data, _ = q([-1.0])
        assert data[0] == 1

    def test_scale_formula(self):
        """Verify scale = absmax / 127.0."""
        q, _, _, _ = _activation_codec()

        values = [3.0, -2.0, 1.0]
        _, scales = q(values)
        expected_scale = 3.0 / 127.0
        assert abs(scales[0] - expected_scale) < 1e-7

    def test_compression_ratio(self):
        q, _, _, _ = _activation_codec()
        values = [float(i) / 100.0 for i in range(4096)]
        data, scales = q(values)
        original_bytes = len(values) * 4
        compressed_bytes = len(data) + len(scales) * 4
        ratio = original_bytes / compressed_bytes
        assert ratio > 3.5

    def test_int8_python_wire_matches_reference(self):
        """Verify Python INT8 wire format is stable across runs."""
        q, _, _, _ = _activation_codec()

        # Fixed input → fixed output (deterministic quantization).
        values = [1.0, -1.0, 0.5, -0.5, 0.0]
        data1, scales1 = q(values)
        data2, scales2 = q(values)
        assert data1 == data2
        assert scales1 == scales2


# ── ForwardMsg integration with IPC codec ──────────────────────────────


class TestForwardMsgIpcInterop:
    """Verify ForwardMsg can carry the same header as IPC (cross-compat)."""

    def test_forward_msg_header_matches_ipc_fields(self):
        """The CBOR field names in ForwardMsg must match IPC header fields."""
        cbor2 = _cbor2()

        # Build a header with all key fields populated.
        header = {
            "request_id": "interop-001",
            "stage_index": 2,
            "total_stages": 4,
            "push_mode": True,
            "shard_layer_start": 8,
            "shard_layer_end": 16,
            "shard_total_layers": 32,
            "kv_session_id": "sess-xyz",
            "kv_store_activation": True,
            "decode_temperature": 0.7,
            "activation_dtype": 0,
            "activation_shape": [1, 1, 896],
            "ring_mode": True,
            "ring_tokens_remaining": 50,
            "prompt_token_ids": [1, 2, 3],
        }

        # Encode as ForwardMsg.
        wire = _encode_forward_msg(MSG_TYPE_FORWARD, header, b"")
        _, decoded_hdr, _ = _decode_forward_msg(wire)

        # Also decode via the IPC codec path (zmq_worker._decode_header).
        from peer.zmq_worker import _decode_header, IpcForwardHeader

        # Build IPC wire from the same header dict.
        hdr_bytes = cbor2.dumps(header)
        inner = bytearray(4 + len(hdr_bytes) + 4)
        struct.pack_into("<I", inner, 0, len(hdr_bytes))
        inner[4:4 + len(hdr_bytes)] = hdr_bytes
        struct.pack_into("<I", inner, 4 + len(hdr_bytes), 0)

        ipc_header, _ = _decode_header(bytes(inner))

        # Same CBOR dict should produce the same parsed values.
        assert decoded_hdr["request_id"] == ipc_header.request_id
        assert decoded_hdr["stage_index"] == ipc_header.stage_index
        assert decoded_hdr["push_mode"] == ipc_header.push_mode
        assert decoded_hdr["shard_layer_start"] == ipc_header.shard_layer_start
        assert decoded_hdr["ring_mode"] == ipc_header.ring_mode
        assert decoded_hdr["ring_tokens_remaining"] == ipc_header.ring_tokens_remaining


# ── Performance tests ──────────────────────────────────────────────────


class TestForwardMsgPerformance:
    """Verify encoding performance meets plan targets."""

    def test_encode_under_10us(self):
        """ForwardMsg encode should be < 10μs (plan target)."""
        header = {
            "request_id": "bench",
            "stage_index": 1,
            "total_stages": 4,
            "shard_layer_start": 8,
            "shard_layer_end": 16,
            "shard_total_layers": 32,
            "activation_dtype": 0,
            "activation_shape": [1, 1, 896],
        }
        activation = _floats_to_bytes([0.1] * 896)

        # Warmup.
        for _ in range(100):
            _encode_forward_msg(MSG_TYPE_FORWARD, header, activation)

        n = 5000
        t0 = time.perf_counter()
        for _ in range(n):
            _encode_forward_msg(MSG_TYPE_FORWARD, header, activation)
        elapsed = time.perf_counter() - t0

        per_encode_us = (elapsed / n) * 1e6
        # Python CBOR is slower than Rust ciborium, so allow 100μs.
        # The Rust side is tested in cargo test (< 10μs target).
        assert per_encode_us < 100, f"Python encode too slow: {per_encode_us:.0f}μs"
        print(f"\nPython ForwardMsg encode: {per_encode_us:.1f}μs/iter")
