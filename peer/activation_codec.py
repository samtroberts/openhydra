# Copyright 2026 OpenHydra contributors — Apache 2.0

"""INT8 symmetric activation compression for inter-peer wire transfer.

Quantizes ``list[float]`` activations to packed INT8 bytes with per-tensor
scale factors.  Achieves ~4x compression (fp32 → int8) with <1% max error
for typical hidden-state ranges.

Usage::

    from peer.activation_codec import quantize_int8, dequantize_int8

    data, scales = quantize_int8(activation)       # → bytes + [scale]
    restored = dequantize_int8(data, scales)        # → list[float]
"""

from __future__ import annotations

import struct


def quantize_int8(values: list[float]) -> tuple[bytes, list[float]]:
    """Per-tensor symmetric INT8 quantization.

    Maps the full value range to [-127, 127] using a single absmax scale
    factor.  Packs each quantized value as a signed byte.

    Returns:
        (packed_int8_bytes, [scale_factor])
    """
    if not values:
        return b"", []

    absmax = max(abs(float(v)) for v in values)
    if absmax == 0.0:
        return bytes(len(values)), [0.0]

    scale = absmax / 127.0
    inv_scale = 1.0 / scale

    packed = bytearray(len(values))
    for i, v in enumerate(values):
        q = int(round(float(v) * inv_scale))
        q = max(-127, min(127, q))
        # Store as unsigned byte: signed → unsigned via (q + 128)
        packed[i] = q + 128

    return bytes(packed), [scale]


def dequantize_int8(data: bytes, scales: list[float]) -> list[float]:
    """Reconstruct floats from packed INT8 bytes + scale factors.

    Args:
        data: Packed INT8 bytes from ``quantize_int8``.
        scales: Scale factor list (single element for per-tensor).

    Returns:
        Reconstructed ``list[float]``.
    """
    if not data:
        return []

    scale = float(scales[0]) if scales else 1.0
    if scale == 0.0:
        return [0.0] * len(data)

    result: list[float] = []
    for b in data:
        q = int(b) - 128  # unsigned → signed
        result.append(q * scale)

    return result
