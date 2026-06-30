// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! OpenHydra protocol core — pure & synchronous.
//!
//! This crate holds the protocol's logic, math, cryptography, and persistence with
//! **no async runtime, no libp2p, and no PyO3**. The `network` crate depends on it and
//! wires these into the live async/FFI node. Keeping this boundary pure means the
//! protocol can be unit-tested (and later reused, e.g. in a verifier or a CLI) without
//! standing up a swarm or a Python interpreter.
//!
//! - [`crypto_agility`] — algorithm registry: versioned wire discriminants (PQC0.1)
//! - [`model_id`] — canonical model identity & equivalence (protocol.md §4)
//! - [`router`] — provider scoring / ranking + resolve→route orchestration (§5)
//! - [`receipts`] — co-signed inference receipts, nested ed25519 (§6)
//! - [`verify`] — verification policy: reputation feedback + decay (§7)
//! - [`credit`] — give/take credit accounting + rate-cap throttle (§6, M2.3)
//! - [`store`] — persistent ledger over `redb` (§6 credit ledger, M2.3)

pub mod credit;
pub mod crypto_agility;
pub mod model_id;
pub mod receipts;
pub mod router;
pub mod store;
pub mod verify;
