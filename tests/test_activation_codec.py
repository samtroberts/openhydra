# Copyright 2026 OpenHydra contributors — Apache 2.0

"""TDD tests for INT8 activation wire compression.

Run:  pytest tests/test_activation_codec.py -v
"""

from __future__ import annotations

import math
import struct

import pytest


def _codec():
    from peer.activation_codec import quantize_int8, dequantize_int8
    return quantize_int8, dequantize_int8


class TestRoundtrip:
    def test_basic_roundtrip(self):
        q, dq = _codec()
        values = [0.5, -0.3, 1.0, -1.0, 0.0, 0.25]
        data, scales = q(values)
        restored = dq(data, scales)
        assert len(restored) == len(values)
        for orig, rec in zip(values, restored):
            assert abs(orig - rec) < 0.02, f"{orig} != {rec}"

    def test_empty_input(self):
        q, dq = _codec()
        data, scales = q([])
        assert data == b""
        assert scales == []
        restored = dq(b"", [])
        assert restored == []

    def test_all_zeros(self):
        q, dq = _codec()
        values = [0.0] * 10
        data, scales = q(values)
        restored = dq(data, scales)
        assert all(v == 0.0 for v in restored)

    def test_single_element(self):
        q, dq = _codec()
        data, scales = q([0.7])
        restored = dq(data, scales)
        assert len(restored) == 1
        assert abs(restored[0] - 0.7) < 0.02

    def test_large_range(self):
        q, dq = _codec()
        values = [-1000.0, 500.0, 0.0, 999.5, -0.001]
        data, scales = q(values)
        restored = dq(data, scales)
        assert len(restored) == len(values)
        for orig, rec in zip(values, restored):
            assert abs(orig - rec) < abs(orig) * 0.01 + 0.1

    def test_compression_ratio(self):
        q, _ = _codec()
        values = [float(i) / 100.0 for i in range(4096)]
        data, scales = q(values)
        original_bytes = len(values) * 4  # fp32
        compressed_bytes = len(data) + len(scales) * 4
        ratio = original_bytes / compressed_bytes
        assert ratio > 3.5, f"Expected >3.5x compression, got {ratio:.1f}x"

    def test_byte_packing(self):
        q, _ = _codec()
        values = [1.0, -1.0, 0.5]
        data, scales = q(values)
        assert isinstance(data, bytes)
        assert len(data) == 3  # 1 byte per value
        assert isinstance(scales, list)
        assert len(scales) == 1  # single scale factor for per-tensor

    def test_negative_values_preserved(self):
        q, dq = _codec()
        values = [-0.9, -0.5, -0.1]
        data, scales = q(values)
        restored = dq(data, scales)
        for orig, rec in zip(values, restored):
            assert rec < 0, f"Sign lost: {orig} -> {rec}"
            assert abs(orig - rec) < 0.02
