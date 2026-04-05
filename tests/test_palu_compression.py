# Copyright 2026 OpenHydra contributors — Apache 2.0

"""TDD tests for PALU low-rank activation compression (P2-A).

Run:  pytest tests/test_palu_compression.py -v
"""

from __future__ import annotations

import math
import pytest


def _codec():
    from peer.palu_codec import palu_compress, palu_decompress
    return palu_compress, palu_decompress


class TestPaluRoundtrip:
    def test_basic_roundtrip(self):
        c, d = _codec()
        values = [float(i) / 10.0 for i in range(64)]
        compressed, meta = c(values, rank=8)
        restored = d(compressed, meta)
        assert len(restored) == len(values)
        # Low-rank approximation: error should be small
        mse = sum((a - b) ** 2 for a, b in zip(values, restored)) / len(values)
        assert mse < 1.0, f"MSE too high: {mse}"

    def test_compression_ratio(self):
        c, _ = _codec()
        values = [float(i) / 100.0 for i in range(4096)]
        compressed, meta = c(values, rank=8)  # Low rank = high compression
        original_bytes = len(values) * 4  # fp32
        compressed_bytes = len(compressed) * 4
        ratio = original_bytes / compressed_bytes
        assert ratio > 3.0, f"Expected >3x compression, got {ratio:.1f}x"

    def test_empty_input(self):
        c, d = _codec()
        compressed, meta = c([], rank=4)
        assert compressed == []
        restored = d([], meta)
        assert restored == []

    def test_rank_clamped_to_input_size(self):
        c, d = _codec()
        # rank=100 but only 10 values — should clamp
        values = [1.0] * 10
        compressed, meta = c(values, rank=100)
        restored = d(compressed, meta)
        assert len(restored) == 10

    def test_single_element(self):
        c, d = _codec()
        compressed, meta = c([42.0], rank=1)
        restored = d(compressed, meta)
        assert len(restored) == 1
        assert abs(restored[0] - 42.0) < 0.01

    def test_high_rank_preserves_fidelity(self):
        c, d = _codec()
        values = [math.sin(i * 0.1) for i in range(128)]
        compressed, meta = c(values, rank=64)  # rank = half of input
        restored = d(compressed, meta)
        mse = sum((a - b) ** 2 for a, b in zip(values, restored)) / len(values)
        assert mse < 0.01, f"High-rank MSE should be very low: {mse}"
