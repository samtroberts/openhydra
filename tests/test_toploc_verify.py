# Copyright 2026 OpenHydra contributors — Apache 2.0

"""TDD tests for TOPLOC hash verification (P2-B).

Tests that activation hashes can verify inference integrity without
redundant re-execution.

Run:  pytest tests/test_toploc_verify.py -v
"""

from __future__ import annotations

import pytest


def _hasher():
    from verification.toploc import activation_hash, verify_hash
    return activation_hash, verify_hash


class TestActivationHash:
    def test_deterministic(self):
        h, _ = _hasher()
        a = [1.0, 2.0, 3.0, 4.0]
        assert h(a) == h(a)

    def test_different_for_different_activations(self):
        h, _ = _hasher()
        h1 = h([1.0, 2.0, 3.0])
        h2 = h([1.0, 2.0, 4.0])
        assert h1 != h2

    def test_empty_activation(self):
        h, _ = _hasher()
        result = h([])
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_returns_bytes(self):
        h, _ = _hasher()
        result = h([0.5, -0.3, 1.0])
        assert isinstance(result, bytes)

    def test_compact_size(self):
        h, _ = _hasher()
        # Hash of a 4096-dim activation should be compact (< 64 bytes)
        big = [float(i) / 1000.0 for i in range(4096)]
        result = h(big)
        assert len(result) <= 64


class TestVerifyHash:
    def test_matching_activations_verify(self):
        h, v = _hasher()
        a = [1.0, 2.0, 3.0, 4.0]
        hash_val = h(a)
        assert v(a, hash_val) is True

    def test_tampered_activation_fails(self):
        h, v = _hasher()
        a = [1.0, 2.0, 3.0, 4.0]
        hash_val = h(a)
        tampered = [1.0, 2.0, 3.0, 999.0]
        assert v(tampered, hash_val) is False

    def test_empty_hash_fails(self):
        _, v = _hasher()
        assert v([1.0, 2.0], b"") is False

    def test_perturbation_detected(self):
        h, v = _hasher()
        a = [1.0, 2.0, 3.0]
        hash_val = h(a)
        perturbed = [1.0, 2.0, 4.0]  # Changed last value
        assert v(perturbed, hash_val) is False


class TestMysterShopperWithToploc:
    def test_verify_uses_hash_when_available(self):
        """When activation_hash is present in ChainResult, mystery shopper
        should use hash verification instead of re-execution."""
        from coordinator.mystery_shopper import MysteryShopper, VerificationResult
        from coordinator.chain import ChainResult

        shopper = MysteryShopper(sample_rate=1.0)  # Always audit

        primary = ChainResult(
            request_id="r1",
            text="Hello world",
            activation=[263.0, 2217.0, 7826.0],
            traces=[],
            latency_ms=10.0,
        )

        # Compute the real hash of the activation
        from verification.toploc import activation_hash
        real_hash = activation_hash(primary.activation)

        # With TOPLOC: build hash-based verification result
        result = shopper.build_hash_verified_result(
            primary=primary,
            activation_hash=real_hash,
        )

        assert isinstance(result, VerificationResult)
        assert result.audited is True
        assert result.match is True
        assert result.mode == "toploc_hash"
        assert result.winner == "primary"
        # No secondary/tertiary execution happened
        assert result.secondary_text is None
        assert result.tertiary_text is None
