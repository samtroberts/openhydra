// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Which on-chain deployment the crypto rail settles on, from the environment (both roles):
//!
//! - `OPENHYDRA_SETTLEMENT_CHAIN` — `solana` or `base`; unset leaves receipts off-chain only.
//! - Solana: `OPENHYDRA_SOLANA_PROGRAM` (the program id) and `OPENHYDRA_SOLANA_CLUSTER` (the
//!   cluster's genesis hash), both base58.
//! - Base: `OPENHYDRA_BASE_CHAIN_ID` and `OPENHYDRA_BASE_CONTRACT` (`0x…`); a consumer also needs
//!   `OPENHYDRA_EVM_KEY_FILE`, the hex secret of the EVM wallet that funds its channels.
//!
//! A consumer signs each paying receipt's on-chain claim for this deployment; a provider co-signs a
//! paying receipt only with a valid claim for it.

use std::sync::Arc;

use openhydra_protocol::settlement::{ChainDomain, Eip712Domain, SolanaDomain};

/// The deployment, and (a Base consumer's) wallet that signs claims.
#[derive(Clone)]
pub struct SettlementChain {
    pub domain: ChainDomain,
    pub evm_wallet: Option<Arc<k256::ecdsa::SigningKey>>,
}

impl std::fmt::Debug for SettlementChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettlementChain")
            .field("domain", &self.domain)
            .field("evm_wallet", &self.evm_wallet.as_ref().map(|_| "<secret>"))
            .finish()
    }
}

fn bs58_32(var: &str, value: &str) -> Result<[u8; 32], String> {
    let bytes = bs58::decode(value.trim()).into_vec().map_err(|e| format!("{var}: not base58: {e}"))?;
    bytes.try_into().map_err(|_| format!("{var}: must be 32 bytes"))
}

/// Parse the settlement deployment from `get` (the environment, in production). `Ok(None)` when no
/// chain is configured. `consumer` asks for the claim-signing key a role needs (a Base wallet).
pub fn from_vars(get: &dyn Fn(&str) -> Option<String>, consumer: bool) -> Result<Option<SettlementChain>, String> {
    let need = |var: &str| get(var).filter(|v| !v.trim().is_empty()).ok_or_else(|| format!("{var} is required"));
    let Some(chain) = get("OPENHYDRA_SETTLEMENT_CHAIN").filter(|v| !v.trim().is_empty()) else {
        return Ok(None);
    };
    match chain.trim() {
        "solana" => {
            let program_id = bs58_32("OPENHYDRA_SOLANA_PROGRAM", &need("OPENHYDRA_SOLANA_PROGRAM")?)?;
            let cluster = bs58_32("OPENHYDRA_SOLANA_CLUSTER", &need("OPENHYDRA_SOLANA_CLUSTER")?)?;
            Ok(Some(SettlementChain { domain: ChainDomain::Solana(SolanaDomain { program_id, cluster }), evm_wallet: None }))
        }
        "base" => {
            let chain_id = need("OPENHYDRA_BASE_CHAIN_ID")?
                .trim()
                .parse::<u64>()
                .map_err(|e| format!("OPENHYDRA_BASE_CHAIN_ID: {e}"))?;
            let contract = need("OPENHYDRA_BASE_CONTRACT")?;
            let contract = hex::decode(contract.trim().trim_start_matches("0x"))
                .map_err(|e| format!("OPENHYDRA_BASE_CONTRACT: {e}"))?;
            let verifying_contract: [u8; 20] =
                contract.try_into().map_err(|_| "OPENHYDRA_BASE_CONTRACT: must be a 20-byte address".to_string())?;
            let evm_wallet = if consumer {
                let path = need("OPENHYDRA_EVM_KEY_FILE")?;
                let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
                let secret = hex::decode(text.trim().trim_start_matches("0x")).map_err(|e| format!("{path}: {e}"))?;
                let key = k256::ecdsa::SigningKey::from_slice(&secret).map_err(|e| format!("{path}: {e}"))?;
                Some(Arc::new(key))
            } else {
                None
            };
            Ok(Some(SettlementChain { domain: ChainDomain::Base(Eip712Domain { chain_id, verifying_contract }), evm_wallet }))
        }
        other => Err(format!("OPENHYDRA_SETTLEMENT_CHAIN: unknown chain {other:?} (solana or base)")),
    }
}

/// [`from_vars`] over the process environment.
pub fn from_env(consumer: bool) -> Result<Option<SettlementChain>, String> {
    from_vars(&|k| std::env::var(k).ok(), consumer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn no_chain_means_off_chain_receipts_only() {
        assert!(from_vars(&vars(&[]), true).unwrap().is_none());
    }

    #[test]
    fn a_solana_deployment_parses_from_base58() {
        let id = bs58::encode([7u8; 32]).into_string();
        let cluster = bs58::encode([9u8; 32]).into_string();
        let c = from_vars(
            &vars(&[("OPENHYDRA_SETTLEMENT_CHAIN", "solana"), ("OPENHYDRA_SOLANA_PROGRAM", &id), ("OPENHYDRA_SOLANA_CLUSTER", &cluster)]),
            true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(c.domain, ChainDomain::Solana(SolanaDomain { program_id: [7; 32], cluster: [9; 32] }));
        assert!(from_vars(&vars(&[("OPENHYDRA_SETTLEMENT_CHAIN", "solana"), ("OPENHYDRA_SOLANA_PROGRAM", &id)]), true)
            .unwrap_err()
            .contains("OPENHYDRA_SOLANA_CLUSTER"));
        assert!(from_vars(&vars(&[("OPENHYDRA_SETTLEMENT_CHAIN", "eth")]), true).is_err());
    }

    #[test]
    fn a_base_consumer_needs_its_wallet_and_a_provider_does_not() {
        let base = [
            ("OPENHYDRA_SETTLEMENT_CHAIN", "base"),
            ("OPENHYDRA_BASE_CHAIN_ID", "8453"),
            ("OPENHYDRA_BASE_CONTRACT", "0x0101010101010101010101010101010101010101"),
        ];
        let provider = from_vars(&vars(&base), false).unwrap().unwrap();
        assert_eq!(provider.domain, ChainDomain::Base(Eip712Domain { chain_id: 8453, verifying_contract: [1; 20] }));
        assert!(provider.evm_wallet.is_none());
        assert!(from_vars(&vars(&base), true).unwrap_err().contains("OPENHYDRA_EVM_KEY_FILE"));
        let dir = tempfile::tempdir().unwrap();
        let key = dir.path().join("evm.key");
        std::fs::write(&key, format!("0x{}\n", "11".repeat(32))).unwrap();
        let mut with_key = base.to_vec();
        let key_s = key.to_string_lossy().to_string();
        with_key.push(("OPENHYDRA_EVM_KEY_FILE", &key_s));
        assert!(from_vars(&vars(&with_key), true).unwrap().unwrap().evm_wallet.is_some());
    }
}
