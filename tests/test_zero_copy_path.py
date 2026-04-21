# Copyright 2026 OpenHydra contributors — Apache 2.0

"""PR-1 zero-copy wire-path tests.

Covers the serialisation boundary between the coordinator, push-mode
producers, and peer receivers. Specifically:

* ``pack_fp32`` / ``unpack_fp32`` round-trip and byte-identical equivalence
  with the legacy ``struct.pack('<{n}f', *values)`` path.
* ``quantize_int8_tensor`` / ``dequantize_int8_tensor`` round-trip with the
  "OpenHydra INT8" spec (per-tensor symmetric absmax, banker's rounding,
  signed→unsigned packing).
* Wire compatibility: a tensor quantised via ``quantize_int8_tensor`` on a
  PyTorch sender must decode correctly via ``dequantize_int8`` on a
  legacy receiver (and vice versa).
* ``resolve_wire_format`` negotiation rules.
* MLX ``_hidden_to_payload`` numeric equivalence between the numpy-backed
  PR-1 path and the legacy ``.tolist()`` path (MLX-only; skipped otherwise).

Run:  pytest tests/test_zero_copy_path.py -v
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from peer.activation_codec import (
    WIRE_FORMAT_AUTO,
    WIRE_FORMAT_FP32,
    WIRE_FORMAT_INT8,
    dequantize_int8,
    dequantize_int8_tensor,
    pack_fp32,
    quantize_int8,
    quantize_int8_tensor,
    resolve_wire_format,
    unpack_fp32,
)


# -----------------------------------------------------------------------------
# pack_fp32 / unpack_fp32
# -----------------------------------------------------------------------------

class TestFp32Packing:
    def test_empty(self):
        assert pack_fp32([]) == b""
        assert unpack_fp32(b"") == []

    def test_roundtrip_small(self):
        values = [1.0, -2.5, 0.0, 3.14159, -0.001]
        packed = pack_fp32(values)
        restored = unpack_fp32(packed)
        for orig, rec in zip(values, restored):
            assert abs(orig - rec) < 1e-6

    def test_roundtrip_large(self):
        values = [float(i) / 100.0 for i in range(10000)]
        packed = pack_fp32(values)
        assert len(packed) == 10000 * 4
        restored = unpack_fp32(packed)
        assert len(restored) == 10000
        # fp32 round-trip is exact for values that fit.
        for orig, rec in zip(values[::137], restored[::137]):
            assert abs(orig - rec) < 1e-4

    def test_byte_equivalence_with_struct_pack(self):
        """pack_fp32 must produce byte-identical output to struct.pack."""
        values = [0.1, -0.2, 1.5, -2.5, 42.0]
        ours = pack_fp32(values)
        theirs = struct.pack(f"<{len(values)}f", *values)
        assert ours == theirs

    def test_numpy_array_input(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        packed = pack_fp32(arr)
        restored = unpack_fp32(packed)
        assert restored == [1.0, 2.0, 3.0]


# -----------------------------------------------------------------------------
# INT8 tensor codec
# -----------------------------------------------------------------------------

class TestInt8TensorCodec:
    def test_roundtrip_numpy(self):
        arr = np.array([0.5, -0.3, 1.0, -1.0, 0.0, 0.25], dtype=np.float32)
        packed, scale, n = quantize_int8_tensor(arr)
        assert n == 6
        assert len(packed) == 6
        restored = dequantize_int8_tensor(packed, scale, as_numpy=True)
        assert restored.dtype == np.float32
        for orig, rec in zip(arr.tolist(), restored.tolist()):
            assert abs(orig - rec) < 0.02

    def test_spec_signed_unsigned_offset(self):
        """OpenHydra INT8: byte = signed_q + 128."""
        arr = np.array([1.0, -1.0], dtype=np.float32)  # absmax=1 → scale=1/127
        packed, scale, _ = quantize_int8_tensor(arr)
        # +1.0 → signed +127 → unsigned 255
        # -1.0 → signed -127 → unsigned 1
        assert packed[0] == 255
        assert packed[1] == 1

    def test_spec_bankers_rounding(self):
        """OpenHydra INT8: round-half-to-even (matches np.round)."""
        # With absmax=1, scale=1/127 → inv_scale=127.
        # 0.5/127 * 127 = 0.5 → banker's round → 0 (even), not 1.
        # But we need to isolate the rounding step: feed 0.5 scaled by
        # a known scale. Simpler: craft an array where the scaled value
        # is exactly 0.5 away from two integers.
        arr = np.array([1.0, 0.5 / 127.0], dtype=np.float32)
        packed, scale, _ = quantize_int8_tensor(arr)
        # First element: +127 → unsigned 255.
        # Second element: 0.5 / 127 * 127 = 0.5 → banker-round → 0 → unsigned 128.
        assert packed[0] == 255
        assert packed[1] == 128

    def test_tensor_variant_matches_list_variant(self):
        """quantize_int8_tensor must produce byte-identical output to
        quantize_int8 for the same numeric input — guarantees a ring with
        mixed sender/receiver implementations stays bitwise consistent."""
        values = [0.1, -0.7, 0.42, -0.42, 0.999, -0.999, 0.0, 0.5]
        ours_bytes, ours_scale, n = quantize_int8_tensor(values)
        theirs_bytes, theirs_scale_list = quantize_int8(values)
        assert ours_bytes == theirs_bytes
        assert ours_scale == pytest.approx(theirs_scale_list[0])
        assert n == len(values)

    def test_wire_compat_legacy_receiver(self):
        """Tensor-quantised payload must decode via legacy dequantize_int8."""
        arr = np.array([0.5, -0.3, 1.0, -1.0, 0.0], dtype=np.float32)
        packed, scale, _ = quantize_int8_tensor(arr)
        legacy_restored = dequantize_int8(packed, [scale])
        for orig, rec in zip(arr.tolist(), legacy_restored):
            assert abs(orig - rec) < 0.02

    def test_torch_tensor_input(self):
        torch = pytest.importorskip("torch")
        t = torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32)
        packed, scale, n = quantize_int8_tensor(t)
        assert n == 3
        restored = dequantize_int8_tensor(packed, scale)
        assert abs(restored[0] - 0.25) < 0.02
        assert abs(restored[1] - (-0.5)) < 0.02
        assert abs(restored[2] - 0.75) < 0.02

    def test_nan_inf_handled(self):
        """Non-finite values must not poison the scale — they are zeroed."""
        arr = np.array([1.0, float("nan"), float("inf"), -1.0], dtype=np.float32)
        packed, scale, _ = quantize_int8_tensor(arr)
        # absmax should be 1.0 (NaN/inf zeroed), not inf.
        assert scale > 0.0
        assert scale < 0.01  # ~1/127
        restored = dequantize_int8_tensor(packed, scale)
        # NaN → 0, inf → 0; surviving ±1.0 should round-trip.
        assert abs(restored[0] - 1.0) < 0.02
        assert restored[1] == 0.0
        assert restored[2] == 0.0
        assert abs(restored[3] - (-1.0)) < 0.02

    def test_empty_tensor(self):
        arr = np.array([], dtype=np.float32)
        packed, scale, n = quantize_int8_tensor(arr)
        assert packed == b""
        assert scale == 0.0
        assert n == 0

    def test_all_zeros(self):
        arr = np.zeros(100, dtype=np.float32)
        packed, scale, n = quantize_int8_tensor(arr)
        assert scale == 0.0
        assert n == 100
        assert len(packed) == 100
        restored = dequantize_int8_tensor(packed, scale)
        assert np.all(restored == 0.0)

    def test_non_contiguous_tensor_accepted(self):
        """Internal ascontiguousarray normalises strided input."""
        base = np.arange(20, dtype=np.float32).reshape(4, 5)
        view = base[:, ::2]  # non-contiguous slice
        packed, scale, n = quantize_int8_tensor(view)
        assert n == view.size
        restored = dequantize_int8_tensor(packed, scale).reshape(view.shape)
        np.testing.assert_allclose(restored, view, atol=scale * 1.5)


# -----------------------------------------------------------------------------
# Wire-format negotiation
# -----------------------------------------------------------------------------

class TestResolveWireFormat:
    def test_empty_defaults_to_int8_when_supported(self):
        assert resolve_wire_format("", local_int8_supported=True) == WIRE_FORMAT_INT8

    def test_empty_falls_back_to_fp32_when_unsupported(self):
        assert resolve_wire_format("", local_int8_supported=False) == WIRE_FORMAT_FP32

    def test_auto_int8_capable(self):
        assert resolve_wire_format(WIRE_FORMAT_AUTO, local_int8_supported=True) == WIRE_FORMAT_INT8

    def test_auto_int8_incapable(self):
        assert resolve_wire_format(WIRE_FORMAT_AUTO, local_int8_supported=False) == WIRE_FORMAT_FP32

    def test_fp32_hard_request(self):
        # FP32 is honoured regardless of int8 capability.
        assert resolve_wire_format(WIRE_FORMAT_FP32, local_int8_supported=True) == WIRE_FORMAT_FP32

    def test_int8_hard_request_downgrades_on_incapable(self):
        assert resolve_wire_format(WIRE_FORMAT_INT8, local_int8_supported=False) == WIRE_FORMAT_FP32

    def test_unknown_value_treated_as_auto(self):
        # Forward-compat with future wire formats — don't crash, default to auto.
        assert resolve_wire_format("fp16", local_int8_supported=True) == WIRE_FORMAT_INT8

    def test_case_insensitive_and_whitespace(self):
        assert resolve_wire_format("  Int8  ", local_int8_supported=True) == WIRE_FORMAT_INT8
        assert resolve_wire_format("AUTO", local_int8_supported=True) == WIRE_FORMAT_INT8

    def test_none_treated_as_auto(self):
        assert resolve_wire_format(None, local_int8_supported=True) == WIRE_FORMAT_INT8


# -----------------------------------------------------------------------------
# preferred_wire_format proto field
# -----------------------------------------------------------------------------

class TestProtoPreferredWireFormat:
    def test_field_exists_on_forward_request(self):
        from peer import peer_pb2
        names = {f.name for f in peer_pb2.ForwardRequest.DESCRIPTOR.fields}
        assert "preferred_wire_format" in names

    def test_default_empty_string(self):
        from peer import peer_pb2
        req = peer_pb2.ForwardRequest()
        assert req.preferred_wire_format == ""

    def test_roundtrip_serialisation(self):
        from peer import peer_pb2
        req = peer_pb2.ForwardRequest(preferred_wire_format=WIRE_FORMAT_AUTO)
        data = req.SerializeToString()
        restored = peer_pb2.ForwardRequest()
        restored.ParseFromString(data)
        assert restored.preferred_wire_format == WIRE_FORMAT_AUTO


# -----------------------------------------------------------------------------
# MLX _hidden_to_payload numeric parity (optional — skipped without MLX)
# -----------------------------------------------------------------------------

class TestMlxHiddenToPayloadParity:
    def test_numpy_path_matches_tolist_path(self):
        mx = pytest.importorskip("mlx.core")
        from peer.mlx_runtime import MLXRuntime  # noqa: F401 — import triggers class

        # Fabricate a small hidden state [1, seq, hidden] directly via MLX.
        data = [[[0.1, -0.2, 0.3, -0.4, 0.5], [0.6, -0.7, 0.8, -0.9, 1.0]]]
        hidden = mx.array(data, dtype=mx.float32)

        # Minimal instance: _hidden_to_payload only reads numpy + hidden.shape.
        runtime = object.__new__(MLXRuntime)  # type: ignore[call-arg]
        payload = runtime._hidden_to_payload(hidden)  # type: ignore[attr-defined]
        assert payload[0] == 2.0  # seq_len
        assert payload[1] == 5.0  # hidden_size
        # Flat tail.
        expected = [0.1, -0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9, 1.0]
        for exp, got in zip(expected, payload[2:]):
            assert abs(exp - got) < 1e-5


# -----------------------------------------------------------------------------
# Server-side unpack path (peer/server.py) — smoke test
# -----------------------------------------------------------------------------

class TestServerUnpackSmoke:
    def test_packed_payload_round_trip_through_unpack_fp32(self):
        """Simulates the server receive path: activation_packed bytes →
        unpack_fp32 → list[float] ready for downstream dispatch."""
        hidden_flat = [float(i) * 0.01 for i in range(2048)]
        header = [4.0, 512.0]  # seq_len, hidden_size
        full = header + hidden_flat
        packed = pack_fp32(full)
        restored = unpack_fp32(packed)
        assert len(restored) == len(full)
        assert restored[0] == 4.0
        assert restored[1] == 512.0
        # Spot-check flat payload.
        for i in (0, 100, 500, 1500):
            assert abs(restored[2 + i] - hidden_flat[i]) < 1e-5
