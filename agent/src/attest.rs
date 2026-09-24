// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Device attestation — the provider side (ATTESTED_ENGINE_TIER_PLAN_v1, Phase 0).
//!
//! Collects what the device reports about itself ([`DeviceFacts`]), hashes the running binary, and
//! — on Apple Silicon — signs an [`AttestationClaim`] with a Secure Enclave P-256 key
//! ([`SecureEnclaveSigner`]). Verification and the rules for what a verified attestation is worth
//! are pure and live in [`openhydra_protocol::attest`].
//!
//! Phase-0 limits:
//! * The result is `self_signed` only: without an Apple attestation chain a consumer can't tell an
//!   SE key from a software key, and every device fact is the device's own word.
//! * The key is **ephemeral** (per process). A persistent SE key must live in the data-protection
//!   keychain, which requires a codesigned binary with a keychain-access-group entitlement —
//!   unsigned builds get `errSecMissingEntitlement` (-34018). Key continuity across restarts
//!   therefore arrives with signed release builds (Phase 2).
//! * Device facts are read through direct system APIs (`sysctlbyname`, `csr_get_active_config`,
//!   the IORegistry), never by spawning a helper process a hijacked `PATH` could substitute.

use sha2::{Digest, Sha256};

#[cfg(target_os = "macos")]
pub use macos::{device_facts, SecureEnclaveSigner};

/// What the device reports about itself. Unverified: see the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFacts {
    /// e.g. `"Apple M1"` (`machdep.cpu.brand_string`).
    pub chip: String,
    /// e.g. `"25G76"` (`kern.osversion`).
    pub os_build: String,
    /// SIP fully enabled — no `csr-active-config` allow-bits set.
    pub sip_enabled: bool,
    /// Secure Boot on (`IODeviceTree:/chosen` `secure-boot` = 1).
    pub secure_boot_enabled: bool,
}

/// Why a Secure Enclave attestation couldn't be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestSignError {
    /// SE key generation failed (the `CFError` code) — e.g. no Secure Enclave (Intel Mac, most VMs).
    KeyGen(isize),
    /// The public key couldn't be exported as a 65-byte uncompressed SEC1 point.
    PublicKey,
    /// The SE refused to sign (the `CFError` code).
    Sign(isize),
}

/// SHA-256 of the running executable — the claim's `binary_hash`.
pub fn binary_hash() -> std::io::Result<[u8; 32]> {
    let mut file = std::fs::File::open(std::env::current_exe()?)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher.finalize().into())
}

#[cfg(target_os = "macos")]
mod macos {
    use std::ffi::{c_char, CString};
    use std::ptr::null_mut;

    use core_foundation::base::{CFType, TCFType};
    use core_foundation::data::CFData;
    use core_foundation::string::CFString;
    use core_foundation_sys::base::{kCFAllocatorDefault, CFAllocatorRef, CFTypeRef};
    use core_foundation_sys::string::CFStringRef;
    use openhydra_protocol::attest::{Attestation, AttestationClaim, P256_SEC1_UNCOMPRESSED_LEN};
    use security_framework::key::{Algorithm, GenerateKeyOptions, KeyType, SecKey, Token};

    use super::{AttestSignError, DeviceFacts};

    extern "C" {
        // libsystem_kernel: the active SIP configuration (0 = fully enabled).
        fn csr_get_active_config(config: *mut u32) -> libc::c_int;
    }

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IORegistryEntryFromPath(main_port: u32, path: *const c_char) -> u32;
        fn IORegistryEntryCreateCFProperty(
            entry: u32,
            key: CFStringRef,
            allocator: CFAllocatorRef,
            options: u32,
        ) -> CFTypeRef;
        fn IOObjectRelease(object: u32) -> libc::c_int;
    }

    /// Read this Mac's [`DeviceFacts`]. An unreadable string fact is empty; an unreadable security
    /// flag is `false` (fail closed — the consumer's policy then rejects it).
    pub fn device_facts() -> DeviceFacts {
        DeviceFacts {
            chip: sysctl_string("machdep.cpu.brand_string").unwrap_or_default(),
            os_build: sysctl_string("kern.osversion").unwrap_or_default(),
            sip_enabled: sip_enabled(),
            secure_boot_enabled: secure_boot_enabled(),
        }
    }

    fn sysctl_string(name: &str) -> Option<String> {
        let name = CString::new(name).ok()?;
        let mut len: libc::size_t = 0;
        // SAFETY: a size query (null output buffer); `len` is a valid out-pointer.
        if unsafe { libc::sysctlbyname(name.as_ptr(), null_mut(), &mut len, null_mut(), 0) } != 0 {
            return None;
        }
        let mut buf = vec![0u8; len];
        // SAFETY: `buf` has `len` writable bytes; sysctlbyname writes at most `len` and updates it.
        if unsafe { libc::sysctlbyname(name.as_ptr(), buf.as_mut_ptr().cast(), &mut len, null_mut(), 0) } != 0 {
            return None;
        }
        buf.truncate(len);
        while buf.last() == Some(&0) {
            buf.pop();
        }
        String::from_utf8(buf).ok()
    }

    fn sip_enabled() -> bool {
        let mut config = 0u32;
        // SAFETY: `config` is a valid out-pointer for one u32.
        unsafe { csr_get_active_config(&mut config) == 0 && config == 0 }
    }

    fn secure_boot_enabled() -> bool {
        let path = CString::new("IODeviceTree:/chosen").expect("static path has no NUL");
        // SAFETY: 0 is kIOMainPortDefault; `path` is a valid NUL-terminated string.
        let entry = unsafe { IORegistryEntryFromPath(0, path.as_ptr()) };
        if entry == 0 {
            return false;
        }
        let key = CFString::from_static_string("secure-boot");
        // SAFETY: `entry` is a live registry entry, released right after; the returned property
        // follows the CF create rule and is owned by the `CFType` wrapper below.
        let prop = unsafe {
            let prop = IORegistryEntryCreateCFProperty(entry, key.as_concrete_TypeRef(), kCFAllocatorDefault, 0);
            IOObjectRelease(entry);
            prop
        };
        if prop.is_null() {
            return false;
        }
        // SAFETY: non-null CFTypeRef returned under the create rule.
        let prop = unsafe { CFType::wrap_under_create_rule(prop) };
        match prop.downcast_into::<CFData>() {
            Some(data) => data.bytes() == [1, 0, 0, 0],
            None => false,
        }
    }

    /// An ephemeral Secure Enclave P-256 key (private half never leaves the SE) that signs
    /// attestation claims. One per process — see the module docs for why it isn't persistent yet.
    pub struct SecureEnclaveSigner {
        key: SecKey,
        public: [u8; P256_SEC1_UNCOMPRESSED_LEN],
    }

    impl SecureEnclaveSigner {
        /// Generate a fresh SE key. Fails on hardware without a Secure Enclave.
        pub fn new() -> Result<Self, AttestSignError> {
            let mut options = GenerateKeyOptions::default();
            options.set_key_type(KeyType::ec()).set_size_in_bits(256).set_token(Token::SecureEnclave);
            let key = SecKey::new(&options).map_err(|e| AttestSignError::KeyGen(e.code()))?;
            let public = key
                .public_key()
                .and_then(|p| p.external_representation())
                .ok_or(AttestSignError::PublicKey)?;
            let public = public.bytes().try_into().map_err(|_| AttestSignError::PublicKey)?;
            Ok(Self { key, public })
        }

        /// The uncompressed SEC1 public key a consumer verifies against.
        pub fn public_key(&self) -> &[u8; P256_SEC1_UNCOMPRESSED_LEN] {
            &self.public
        }

        /// Sign `claim` (over [`Attestation::signing_bytes`]) inside the Secure Enclave.
        pub fn sign(&self, claim: AttestationClaim) -> Result<Attestation, AttestSignError> {
            let signature_der = self
                .key
                .create_signature(
                    Algorithm::ECDSASignatureMessageX962SHA256,
                    &Attestation::signing_bytes(&claim, &self.public),
                )
                .map_err(|e| AttestSignError::Sign(e.code()))?;
            Ok(Attestation { claim, signing_key: self.public, signature_der })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_hash_is_stable() {
        let a = binary_hash().unwrap();
        assert_eq!(a, binary_hash().unwrap());
        assert_ne!(a, [0u8; 32]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn device_facts_are_readable() {
        let facts = device_facts();
        assert!(!facts.chip.is_empty(), "{facts:?}");
        assert!(!facts.os_build.is_empty(), "{facts:?}");
    }

    /// Live Secure Enclave round-trip: SE-sign a real claim, verify it with the pure verifier.
    /// Needs Apple Silicon with SIP + Secure Boot on (most CI VMs have no SE):
    /// `cargo test -p openhydra-agent attest -- --ignored`.
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore]
    fn secure_enclave_attestation_verifies() {
        use openhydra_protocol::attest::{
            verify_attestation, AttestError, AttestPolicy, AttestationClaim, DeviceTrust,
        };

        let facts = device_facts();
        let bin = binary_hash().unwrap();
        let signer = SecureEnclaveSigner::new().expect("Secure Enclave key");
        let now = 1_790_000_000_000;
        let nonce = [0x42; 32];
        let claim = AttestationClaim {
            chip: facts.chip.clone(),
            os_build: facts.os_build.clone(),
            sip_enabled: facts.sip_enabled,
            secure_boot_enabled: facts.secure_boot_enabled,
            binary_hash: bin,
            model_id: "qwen2.5-0.5b".into(),
            model_digest: None,
            active_model_hash: None,
            session_pubkey: None,
            nonce,
            ts_unix_ms: now,
        };
        let trusted = [bin];
        let policy = AttestPolicy {
            expected_nonce: nonce,
            now_unix_ms: now,
            max_age_ms: 60_000,
            max_skew_ms: 5_000,
            trusted_binaries: &trusted,
            pinned_key: Some(signer.public_key()),
        };
        // Several signatures, so the ~1/3 high-S signatures the SE emits are exercised too.
        for _ in 0..8 {
            let att = signer.sign(claim.clone()).unwrap();
            let wire = openhydra_protocol::attest::Attestation::from_bytes(&att.to_bytes()).unwrap();
            assert_eq!(verify_attestation(&wire, &policy), Ok(DeviceTrust::SelfSigned), "{facts:?}");
        }
        let mut tampered = signer.sign(claim).unwrap();
        tampered.claim.model_id = "gpt-oss-120b".into();
        assert_eq!(verify_attestation(&tampered, &policy), Err(AttestError::BadSignature));
    }
}
