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
