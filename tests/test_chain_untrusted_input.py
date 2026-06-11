# Copyright 2026 OpenHydra contributors — Apache 2.0

"""Audit C-F1 / 2.3: coordinator hardening against untrusted peer responses.

Run:  pytest tests/test_chain_untrusted_input.py -v
"""

from __future__ import annotations

import struct

import pytest


def _pack(values):
    return struct.pack(f"<{len(values)}f", *values)


class TestSafeUnpackActivation:
    def test_roundtrip(self):
        from coordinator.chain import _safe_unpack_activation
        vals = [1.0, -2.5, 3.0, 0.0, 7.25]
        out = _safe_unpack_activation(_pack(vals))
        assert out == pytest.approx(vals)

    def test_ragged_length_truncates_not_crashes(self):
        # Audit C-F1: a non-multiple-of-4 length must not raise struct.error
        # (which would escape the failover catch).
        from coordinator.chain import _safe_unpack_activation
        raw = _pack([1.0, 2.0]) + b"\x01\x02\x03"  # 3 trailing bytes
        out = _safe_unpack_activation(raw)
        assert out == pytest.approx([1.0, 2.0])

    def test_oversized_rejected(self):
        # Audit C-F1: a claimed size above the cap raises RuntimeError so the
        # caller fails over instead of allocating a giant list.
        from coordinator import chain
        # Build a buffer just over the cap without actually allocating the
        # huge float list: monkeypatch the cap down for the test.
        small_cap = 4
        orig = chain._MAX_ACTIVATION_FLOATS
        chain._MAX_ACTIVATION_FLOATS = small_cap
        try:
            raw = _pack([0.0] * (small_cap + 1))
            with pytest.raises(RuntimeError):
                chain._safe_unpack_activation(raw)
        finally:
            chain._MAX_ACTIVATION_FLOATS = orig

    def test_nan_inf_sanitized(self):
        # Audit 2.3: non-finite values from a peer are mapped to 0.0.
        from coordinator.chain import _safe_unpack_activation
        raw = _pack([1.0, float("nan"), float("inf"), float("-inf"), 2.0])
        out = _safe_unpack_activation(raw)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == 0.0
        assert out[2] == 0.0
        assert out[3] == 0.0
        assert out[4] == pytest.approx(2.0)


class TestDequantizeScaleGuard:
    def test_inf_scale_zeroed(self):
        # Audit 2.3: a non-finite dequant scale must not poison the output.
        from peer.activation_codec import dequantize_int8
        data = bytes([127, 64, 0, 200])  # arbitrary int8 payload
        out = dequantize_int8(data, [float("inf")])
        assert all(v == 0.0 for v in out)

    def test_nan_scale_zeroed(self):
        from peer.activation_codec import dequantize_int8
        data = bytes([127, 64, 0, 200])
        out = dequantize_int8(data, [float("nan")])
        assert all(v == 0.0 for v in out)

    def test_normal_scale_unaffected(self):
        # A finite scale must NOT be clobbered by the guard: a non-zero int8
        # payload yields a finite, non-zero dequantized value.
        import math
        from peer.activation_codec import dequantize_int8
        data = bytes([127])
        out = dequantize_int8(data, [2.0])
        assert len(out) == 1
        assert math.isfinite(out[0])
        assert out[0] != 0.0
