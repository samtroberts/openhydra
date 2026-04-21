# Copyright 2026 OpenHydra contributors — Apache 2.0

"""INT8 symmetric activation compression for inter-peer wire transfer.

**OpenHydra INT8 spec (v1.0) — enforced identically on PyTorch and MLX sides:**

* Per-tensor symmetric quantization.
* ``scale = absmax / 127.0`` (single fp32 scalar, stored as the sole element
  of ``quantized_scales``).
* Quantized values in ``[-127, +127]``, stored as unsigned bytes via
  ``q_unsigned = q_signed + 128``.
* Rounding: round-half-to-even (banker's rounding). This matches both
  Python's built-in ``round`` and NumPy's ``np.round`` behaviour, keeping
  PyTorch (sender) and MLX (receiver) bitwise compatible.
* Empty input → ``(b"", [])``. All-zero input → ``(bytes(n), [0.0])``.

This module provides two codec surfaces, both enforcing the spec above:

* ``quantize_int8(list[float])`` / ``dequantize_int8(bytes, list[float])``
  — the legacy wire format used by the auto-regressive coordinator path.
  Internally vectorised via NumPy when available (≥ 100× faster than the
  original pure-Python loop); falls back to pure Python if NumPy is absent.

* ``quantize_int8_tensor(tensor)`` / ``dequantize_int8_tensor(bytes, scale,
  shape)`` — tensor-native variants for the zero-copy push boundary.
  Accepts any object with a ``numpy()`` method (PyTorch ``Tensor``) or a
  raw NumPy array; callers producing MLX arrays should ``mx.eval`` and
  route through the NumPy bridge so a single quantisation kernel governs
  both backends.

The wire encoding is identical for both surfaces, so an INT8 payload
produced by ``quantize_int8_tensor`` on a CUDA sender decodes correctly
via ``dequantize_int8`` on a pure-Python receiver.
"""

from __future__ import annotations

import struct
from typing import Any

try:  # NumPy is effectively always present (torch is already a hard dep),
    # but we guard the import so the codec remains usable in minimal envs.
    import numpy as _np  # type: ignore
    _HAS_NUMPY = True
except Exception:  # pragma: no cover — defensive, not exercised in CI
    _np = None  # type: ignore
    _HAS_NUMPY = False


# -----------------------------------------------------------------------------
# Internal helpers (numpy fast path + pure-python fallback)
# -----------------------------------------------------------------------------

def _quantize_np(arr: "_np.ndarray") -> tuple[bytes, float]:
    """Numpy-vectorised symmetric absmax INT8 quantisation.

    ``arr`` must be a 1-D fp32/fp64 ndarray. Returns ``(packed_bytes, scale)``
    where ``packed_bytes[i] = q_signed_i + 128`` (unsigned byte storage).
    """
    if arr.size == 0:
        return b"", 0.0

    # Cast to float32 — the wire spec is fp32-scale — and reject NaNs/infs
    # by replacing them with zero (producing a valid quantised value rather
    # than propagating garbage across the ring).
    a = _np.ascontiguousarray(arr, dtype=_np.float32)
    if not _np.all(_np.isfinite(a)):
        a = _np.where(_np.isfinite(a), a, 0.0).astype(_np.float32, copy=False)

    absmax = float(_np.abs(a).max())
    if absmax == 0.0:
        return bytes(int(a.size)), 0.0

    scale = absmax / 127.0
    inv_scale = 1.0 / scale
    # np.round uses banker's rounding (half-to-even), matching Python's
    # built-in round() for integer targets — the legacy codec's semantics.
    q = _np.round(a * inv_scale).astype(_np.int32, copy=False)
    _np.clip(q, -127, 127, out=q)
    # Signed → unsigned via +128 (same as legacy codec).
    packed = (q + 128).astype(_np.uint8, copy=False)
    return packed.tobytes(), scale


def _dequantize_np(data: bytes, scale: float) -> "_np.ndarray":
    """Numpy-vectorised inverse of ``_quantize_np``.

    Returns a 1-D ``float32`` ndarray.
    """
    if not data:
        return _np.zeros((0,), dtype=_np.float32)
    if scale == 0.0:
        return _np.zeros((len(data),), dtype=_np.float32)
    # frombuffer yields a read-only view; copy so downstream code may cast /
    # reshape / mutate freely without surprise.
    raw = _np.frombuffer(data, dtype=_np.uint8)
    signed = raw.astype(_np.int32) - 128
    return (signed.astype(_np.float32) * _np.float32(scale))


def _quantize_py(values: list[float]) -> tuple[bytes, float]:
    """Pure-Python fallback — identical semantics to ``_quantize_np``."""
    if not values:
        return b"", 0.0
    absmax = 0.0
    for v in values:
        av = abs(float(v))
        if av > absmax:
            absmax = av
    if absmax == 0.0:
        return bytes(len(values)), 0.0
    scale = absmax / 127.0
    inv_scale = 1.0 / scale
    packed = bytearray(len(values))
    for i, v in enumerate(values):
        q = int(round(float(v) * inv_scale))  # banker's rounding
        if q > 127:
            q = 127
        elif q < -127:
            q = -127
        packed[i] = q + 128
    return bytes(packed), scale


def _dequantize_py(data: bytes, scale: float) -> list[float]:
    """Pure-Python fallback — identical semantics to ``_dequantize_np``."""
    if not data:
        return []
    if scale == 0.0:
        return [0.0] * len(data)
    return [(int(b) - 128) * scale for b in data]


# -----------------------------------------------------------------------------
# Public API — list[float] surface (wire-compatible with the legacy codec)
# -----------------------------------------------------------------------------

def quantize_int8(values: list[float]) -> tuple[bytes, list[float]]:
    """Per-tensor symmetric INT8 quantisation of a Python list of floats.

    Returns ``(packed_int8_bytes, [scale_factor])`` where the scale list has
    exactly one element (or zero for the empty-input edge case).

    See the OpenHydra INT8 spec at the top of this module.
    """
    if _HAS_NUMPY:
        # Route through numpy for vectorisation. The constructor is O(n)
        # memcpy, dominant cost is absmax + rounding (both CUDA-friendly
        # when callers switch to ``quantize_int8_tensor``).
        arr = _np.asarray(values, dtype=_np.float32)  # type: ignore[union-attr]
        packed, scale = _quantize_np(arr)
    else:
        packed, scale = _quantize_py(values)

    if not values:
        return b"", []
    return packed, [float(scale)]


def dequantize_int8(data: bytes, scales: list[float]) -> list[float]:
    """Inverse of ``quantize_int8``.

    ``scales`` is the list returned by ``quantize_int8`` (single element for
    per-tensor); any extra elements are ignored.
    """
    if not data:
        return []
    scale = float(scales[0]) if scales else 1.0
    if _HAS_NUMPY:
        return _dequantize_np(data, scale).tolist()  # type: ignore[no-any-return]
    return _dequantize_py(data, scale)


# -----------------------------------------------------------------------------
# Public API — tensor surface (zero-copy push boundary)
# -----------------------------------------------------------------------------

def _to_numpy_1d(tensor: Any) -> "_np.ndarray":
    """Best-effort tensor → contiguous 1-D numpy float32 ndarray.

    Supports:
    * PyTorch ``Tensor`` (via ``.detach().cpu().numpy()``)
    * NumPy ``ndarray`` (re-cast / flattened in place if already fp32)
    * Objects exposing ``__array__`` (MLX arrays route through this)
    * Plain ``list`` / ``tuple`` / any sequence (fallback)

    Callers are responsible for ensuring the tensor represents the
    wire-format layout they want to transmit — this helper only enforces
    dtype + contiguity + flattening.
    """
    if not _HAS_NUMPY:
        raise RuntimeError(
            "quantize_int8_tensor requires numpy; install numpy or use "
            "quantize_int8(list[float]) instead"
        )

    # PyTorch tensor fast path: detach (drop autograd), move to CPU,
    # request contiguous memory, then zero-copy view.
    if hasattr(tensor, "detach") and hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
        t = tensor.detach()
        # ``.cpu()`` is a no-op when already on CPU.
        t = t.cpu().contiguous()
        arr = t.numpy()
    elif isinstance(tensor, _np.ndarray):  # type: ignore[union-attr]
        arr = tensor
    else:
        # Sequences, MLX arrays via __array__, etc.
        arr = _np.asarray(tensor)  # type: ignore[union-attr]

    # Flatten to 1-D without copying when possible.
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.dtype != _np.float32:  # type: ignore[union-attr]
        arr = arr.astype(_np.float32, copy=False)  # type: ignore[union-attr]
    if not arr.flags["C_CONTIGUOUS"]:
        arr = _np.ascontiguousarray(arr)  # type: ignore[union-attr]
    return arr


def quantize_int8_tensor(tensor: Any) -> tuple[bytes, float, int]:
    """Tensor-native symmetric INT8 quantisation.

    Accepts any PyTorch ``Tensor``, NumPy ``ndarray``, or array-like object.
    Enforces the OpenHydra INT8 spec:

    * Flattens to 1-D, casts to float32, requires C-contiguous memory.
    * Per-tensor ``scale = absmax / 127.0`` (single fp32 scalar).
    * Round-half-to-even (banker's rounding) via ``np.round``.
    * Signed → unsigned storage (``q_byte = q_signed + 128``).

    Returns ``(packed_bytes, scale, element_count)``. The element count is
    returned alongside so receivers can reshape without re-deriving from
    ``len(packed)`` (which would be lossy if the wire format ever grows a
    header).
    """
    arr = _to_numpy_1d(tensor)
    packed, scale = _quantize_np(arr)
    return packed, float(scale), int(arr.size)


# -----------------------------------------------------------------------------
# Wire-format negotiation (PR-1)
# -----------------------------------------------------------------------------

# Canonical wire-format tokens carried on ``ForwardRequest.preferred_wire_format``.
WIRE_FORMAT_AUTO = "auto"
WIRE_FORMAT_FP32 = "fp32"
WIRE_FORMAT_INT8 = "int8"
# Empty string is treated as "auto" for forward compatibility with older
# coordinators that don't set the field.
_VALID_WIRE_FORMATS = {"", WIRE_FORMAT_AUTO, WIRE_FORMAT_FP32, WIRE_FORMAT_INT8}


def resolve_wire_format(
    requested: str | None,
    *,
    local_int8_supported: bool,
) -> str:
    """Negotiate the effective wire format for a single hop.

    Rules (OpenHydra INT8 spec §5):

    * ``""`` / ``"auto"`` → ``int8`` when the local side supports it, else
      ``fp32``. **This is the default for every new connection.**
    * ``"int8"`` → hard-request. If the local side doesn't support int8 we
      downgrade to ``fp32`` and log (caller's responsibility) rather than
      break the hop.
    * ``"fp32"`` → hard-request packed float32 (no quantisation).
    * Any other string → treated as ``"auto"`` (forward compat).
    """
    req = (requested or "").strip().lower()
    if req not in _VALID_WIRE_FORMATS:
        req = WIRE_FORMAT_AUTO
    if req == WIRE_FORMAT_FP32:
        return WIRE_FORMAT_FP32
    if req == WIRE_FORMAT_INT8:
        return WIRE_FORMAT_INT8 if local_int8_supported else WIRE_FORMAT_FP32
    # "" or "auto"
    return WIRE_FORMAT_INT8 if local_int8_supported else WIRE_FORMAT_FP32


# -----------------------------------------------------------------------------
# Binary fp32 packing helpers — vectorised equivalents of ``struct.pack`` /
# ``struct.unpack`` used by the push-mode and auto-regressive wire paths.
# -----------------------------------------------------------------------------

def pack_fp32(values: Any) -> bytes:
    """Pack an fp32 sequence into little-endian bytes (single memcpy when numpy
    is available).

    Equivalent to ``struct.pack(f'<{len(values)}f', *values)`` but avoids the
    Python-level argument-unpacking overhead that dominates for 10⁵+ element
    activations. Accepts any array-like that numpy can view.
    """
    if _HAS_NUMPY:
        arr = _np.asarray(values, dtype="<f4")  # type: ignore[union-attr]
        if arr.size == 0:
            return b""
        if not arr.flags["C_CONTIGUOUS"]:
            arr = _np.ascontiguousarray(arr)  # type: ignore[union-attr]
        return arr.tobytes()
    if not values:
        return b""
    # Fallback — correct but slow for large lists.
    return struct.pack(f"<{len(values)}f", *values)


def unpack_fp32(data: bytes) -> list[float]:
    """Inverse of :func:`pack_fp32` — returns a Python ``list[float]``."""
    if not data:
        return []
    n = len(data) // 4
    if _HAS_NUMPY:
        arr = _np.frombuffer(data[: n * 4], dtype="<f4")  # type: ignore[union-attr]
        return arr.astype(_np.float32, copy=False).tolist()  # type: ignore[union-attr]
    return list(struct.unpack(f"<{n}f", data[: n * 4]))


def dequantize_int8_tensor(
    data: bytes,
    scale: float,
    *,
    as_numpy: bool = True,
) -> Any:
    """Inverse of ``quantize_int8_tensor``.

    By default returns a 1-D ``float32`` NumPy array (zero-copy handoff to
    PyTorch / MLX downstream via ``torch.from_numpy`` / ``mx.array``).
    Pass ``as_numpy=False`` to get a ``list[float]`` (pure Python, slower —
    reserved for the legacy coordinator path).
    """
    if as_numpy:
        if not _HAS_NUMPY:
            raise RuntimeError("dequantize_int8_tensor(as_numpy=True) requires numpy")
        return _dequantize_np(data, float(scale))
    if _HAS_NUMPY:
        return _dequantize_np(data, float(scale)).tolist()  # type: ignore[no-any-return]
    return _dequantize_py(data, float(scale))
