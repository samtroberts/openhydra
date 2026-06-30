// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Crypto-agility registry (PQC0.1).
//!
//! Stable 1-byte wire discriminants for signature and key-exchange algorithms, so the
//! protocol's signed and exchanged artifacts can migrate to post-quantum schemes by a
//! **version bump rather than a breaking fork**. See `docs/PQC_IMPLEMENTATION_PLAN.md`.
//!
//! Today only [`SigAlg::Ed25519`] and [`KexAlg::X25519`] are *implemented*. The
//! post-quantum discriminants are **reserved** with stable numbers so later milestones
//! (PQC1.1 hybrid KEX, PQC3.1 hybrid signatures) slot in without renumbering the wire.
//! Decoders accept a reserved discriminant as *known*, but performing crypto with an
//! unimplemented one returns [`AlgError::UnsupportedSig`] — never a silent downgrade.
//!
//! The discriminant is bound into the *signed preimage* (see
//! [`crate::receipts::ReceiptPayload::canonical_bytes`]) so it cannot be stripped or
//! downgraded by a wire-level attacker.

use std::fmt;

/// Signature algorithm, with a stable 1-byte wire discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SigAlg {
    /// Ed25519 (RFC 8032). The only implemented signature algorithm today.
    Ed25519 = 1,
    /// ML-DSA-65 (FIPS-204, formerly Dilithium). Reserved; implemented in PQC3.1.
    MlDsa65 = 2,
    /// Hybrid Ed25519 + ML-DSA-65 (both must verify). Reserved; implemented in PQC3.1.
    HybridEd25519MlDsa65 = 3,
}

impl SigAlg {
    /// The stable wire discriminant.
    pub fn to_u8(self) -> u8 {
        self as u8
    }

    /// Decode a wire discriminant. Unknown bytes are **rejected**, never silently mapped.
    pub fn from_u8(v: u8) -> Result<Self, AlgError> {
        match v {
            1 => Ok(Self::Ed25519),
            2 => Ok(Self::MlDsa65),
            3 => Ok(Self::HybridEd25519MlDsa65),
            other => Err(AlgError::UnknownSig(other)),
        }
    }

    /// Whether this build can actually sign/verify with this algorithm.
    pub fn is_implemented(self) -> bool {
        matches!(self, Self::Ed25519)
    }
}

/// Key-exchange algorithm, with a stable 1-byte wire discriminant. Used by the
/// transport handshake selection (PQC1.1); defined here so the registry is one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KexAlg {
    /// X25519 ECDH. The only implemented key exchange today.
    X25519 = 1,
    /// Hybrid X25519 + ML-KEM-768 (FIPS-203). Reserved; implemented in PQC1.1.
    X25519MlKem768 = 2,
}

impl KexAlg {
    /// The stable wire discriminant.
    pub fn to_u8(self) -> u8 {
        self as u8
    }

    /// Decode a wire discriminant. Unknown bytes are **rejected**.
    pub fn from_u8(v: u8) -> Result<Self, AlgError> {
        match v {
            1 => Ok(Self::X25519),
            2 => Ok(Self::X25519MlKem768),
            other => Err(AlgError::UnknownKex(other)),
        }
    }

    /// Whether this build can actually perform this key exchange.
    pub fn is_implemented(self) -> bool {
        matches!(self, Self::X25519)
    }
}

/// A registry error: an unknown wire discriminant, or a known-but-not-yet-implemented
/// algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgError {
    /// Unknown signature-algorithm discriminant on the wire.
    UnknownSig(u8),
    /// Unknown key-exchange-algorithm discriminant on the wire.
    UnknownKex(u8),
    /// A known signature algorithm that this build cannot perform yet.
    UnsupportedSig(SigAlg),
}

impl fmt::Display for AlgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AlgError::UnknownSig(v) => write!(f, "unknown signature algorithm discriminant: {v}"),
            AlgError::UnknownKex(v) => write!(f, "unknown key-exchange algorithm discriminant: {v}"),
            AlgError::UnsupportedSig(a) => {
                write!(f, "signature algorithm not implemented in this build: {a:?}")
            }
        }
    }
}

impl std::error::Error for AlgError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sig_alg_discriminants_are_stable() {
        // These numbers are a wire contract — changing them breaks every signed artifact.
        assert_eq!(SigAlg::Ed25519.to_u8(), 1);
        assert_eq!(SigAlg::MlDsa65.to_u8(), 2);
        assert_eq!(SigAlg::HybridEd25519MlDsa65.to_u8(), 3);
    }

    #[test]
    fn sig_alg_roundtrips() {
        for a in [SigAlg::Ed25519, SigAlg::MlDsa65, SigAlg::HybridEd25519MlDsa65] {
            assert_eq!(SigAlg::from_u8(a.to_u8()), Ok(a));
        }
    }

    #[test]
    fn unknown_sig_alg_is_rejected() {
        assert_eq!(SigAlg::from_u8(0), Err(AlgError::UnknownSig(0)));
        assert_eq!(SigAlg::from_u8(4), Err(AlgError::UnknownSig(4)));
        assert_eq!(SigAlg::from_u8(255), Err(AlgError::UnknownSig(255)));
    }

    #[test]
    fn only_classical_is_implemented_today() {
        assert!(SigAlg::Ed25519.is_implemented());
        assert!(!SigAlg::MlDsa65.is_implemented());
        assert!(!SigAlg::HybridEd25519MlDsa65.is_implemented());
        assert!(KexAlg::X25519.is_implemented());
        assert!(!KexAlg::X25519MlKem768.is_implemented());
    }

    #[test]
    fn kex_alg_roundtrips_and_rejects_unknown() {
        assert_eq!(KexAlg::from_u8(1), Ok(KexAlg::X25519));
        assert_eq!(KexAlg::from_u8(2), Ok(KexAlg::X25519MlKem768));
        assert_eq!(KexAlg::from_u8(0), Err(AlgError::UnknownKex(0)));
        assert_eq!(KexAlg::from_u8(9), Err(AlgError::UnknownKex(9)));
    }
}
