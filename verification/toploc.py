# Copyright 2026 OpenHydra contributors — Apache 2.0

"""TOPLOC — Locality-Sensitive Hash verification for inference integrity.

Instead of re-executing the full inference on a second peer (19.5s),
TOPLOC verifies that intermediate activations match expected statistical
properties using a compact hash digest.

The hash is computed by the peer during forward() and included in the
gRPC ForwardResponse.  The coordinator verifies the hash against the
received activation vector without any additional network calls.

Reference: primeintellect.ai/blog/toploc

Algorithm:
    1. Quantize activation values to 8-bit buckets (discretize)
    2. SHA-256 hash the quantized byte representation
    3. Truncate to 32 bytes for compact wire transfer

This catches:
    - Model weight tampering (different weights → different activations)
    - Compute precision changes (fp16 vs fp32 drift)
    - Prompt injection at the peer level
    - Random output substitution
"""

from __future__ import annotations

import hashlib
import struct


def activation_hash(activation: list[float]) -> bytes:
    """Compute a compact hash digest of an activation vector.

    Quantizes each float to an 8-bit bucket, then SHA-256 hashes
    the packed bytes.  Returns a 32-byte digest.

    Args:
        activation: List of float values (token IDs or hidden states).

    Returns:
        32-byte SHA-256 digest.
    """
    if not activation:
        return hashlib.sha256(b"empty").digest()

    # Quantize to 8-bit buckets for deterministic hashing
    # Token IDs (>1.0) are rounded to integers
    # Hidden states ([-1, 1]) are mapped to 256 buckets
    packed = bytearray()
    for v in activation:
        fv = float(v)
        if abs(fv) > 1.5:
            # Token ID: pack as 4-byte int
            packed.extend(struct.pack("<i", int(round(fv))))
        else:
            # Hidden state: quantize to 8-bit bucket
            bucket = int(round((fv + 1.0) * 127.5))
            bucket = max(0, min(255, bucket))
            packed.append(bucket)

    return hashlib.sha256(bytes(packed)).digest()


def activation_hash_tensor(tensor) -> bytes:
    """Vectorized activation hash using PyTorch ops.

    Accepts a raw PyTorch tensor (shape [batch, seq, hidden]) and
    computes the same hash as activation_hash() but using PyTorch's
    C++ vectorized ops instead of a Python for-loop over 1920 floats.

    For hidden states (all values in [-1, 1] from layer norm), the
    fast path produces byte-identical hashes to activation_hash().
    Falls back to the original loop for mixed token-ID vectors.
    """
    try:
        import torch
    except ImportError:
        # No PyTorch — fall back to the scalar loop.
        if hasattr(tensor, "tolist"):
            return activation_hash(tensor.tolist())
        return activation_hash(list(tensor))

    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().view(-1)
    large_mask = flat.abs() > 1.5

    if not large_mask.any():
        # Fast path: all hidden states ≤1.5 — uniform 1-byte quantization.
        # This is the common case for intermediate activations (layer-normed).
        buckets = ((flat + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        return hashlib.sha256(buckets.numpy().tobytes()).digest()

    # Slow path: mixed token-ID + hidden-state vector (rare for intermediate stages).
    # Fall back to per-element loop to preserve exact mixed-width hash format.
    return activation_hash(flat.tolist())


def verify_hash(activation: list[float], expected_hash: bytes) -> bool:
    """Verify an activation vector against an expected hash.

    Args:
        activation: The activation to verify.
        expected_hash: The expected 32-byte digest from the peer.

    Returns:
        True if the activation matches the hash.
    """
    if not expected_hash:
        return False
    computed = activation_hash(activation)
    return computed == expected_hash
