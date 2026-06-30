// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Process-level secret hygiene (PQC0.2).
//!
//! Two cheap, portable mitigations applied once at startup, complementing the
//! per-buffer `zeroize` in the key loader (`openhydra_network::identity`):
//!
//! * **Disable core dumps** (`RLIMIT_CORE = 0`) — a crash must never write a memory
//!   image (which would contain the Ed25519 identity secret) to disk.
//! * **Best-effort `mlockall`** — lock current + future pages so secret material is
//!   never paged out to swap.
//!
//! These protect *all* in-memory secrets at once, including the transient hex/string
//! copies the key loader can't individually scrub. No-op on non-unix targets.

/// Apply process hardening. Never fails the process: `mlockall` is commonly denied
/// without privilege / a raised `RLIMIT_MEMLOCK`, in which case we log and continue
/// (the core-dump disable still applies). Call once, as early in `main` as possible.
pub fn harden_process() {
    #[cfg(unix)]
    // SAFETY: both calls are simple syscalls with no memory-safety preconditions; we
    // check their return codes and never dereference their (absent) outputs.
    unsafe {
        // A core file would contain the identity secret key — refuse to write one.
        let lim = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
        if libc::setrlimit(libc::RLIMIT_CORE, &lim) != 0 {
            tracing::warn!("PQC0.2: could not disable core dumps (RLIMIT_CORE)");
        }
        // Keep secret pages off swap. Best-effort — needs CAP_IPC_LOCK or a raised
        // RLIMIT_MEMLOCK; failure is expected on stock setups, so log at debug.
        if libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) != 0 {
            tracing::debug!(
                "PQC0.2: mlockall failed (likely RLIMIT_MEMLOCK); secrets may be swappable"
            );
        } else {
            tracing::debug!("PQC0.2: process memory locked (mlockall)");
        }
    }
}
