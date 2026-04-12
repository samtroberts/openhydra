//! NAT detection helpers using AutoNAT results.
//!
//! AutoNAT runs inside the swarm and probes reachability via bootstrap peers.
//! This module translates AutoNAT events into OpenHydra NatInfo.

use crate::types::NatInfo;

/// Map a libp2p AutoNAT status to an OpenHydra nat_type string.
///
/// AutoNAT reports:
/// - `Public(addr)` → the external address is reachable → "open"
/// - `Private` → behind NAT, not directly reachable → "symmetric" (conservative)
/// - `Unknown` → not enough probes yet → "unknown"
pub fn nat_type_from_autonat(is_public: bool, is_unknown: bool) -> &'static str {
    if is_unknown {
        "unknown"
    } else if is_public {
        "open"
    } else {
        // AutoNAT can't distinguish full_cone from symmetric.
        // We conservatively report "symmetric" so we always set requires_relay=true.
        "symmetric"
    }
}

/// Whether TCP relay is required for this NAT type.
///
/// TCP can't hole-punch — ALL non-open NAT types require relay.
/// DCUtR may upgrade the connection later, but we must relay first.
pub fn requires_relay(nat_type: &str) -> bool {
    nat_type != "open"
}

/// Build a NatInfo from AutoNAT probe results.
pub fn build_nat_info(
    nat_type: &str,
    external_ip: String,
    external_port: u16,
) -> NatInfo {
    let is_public = nat_type == "open";
    NatInfo {
        nat_type: nat_type.to_string(),
        external_ip,
        external_port,
        is_public,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nat_type_mapping() {
        assert_eq!(nat_type_from_autonat(true, false), "open");
        assert_eq!(nat_type_from_autonat(false, false), "symmetric");
        assert_eq!(nat_type_from_autonat(false, true), "unknown");
        assert_eq!(nat_type_from_autonat(true, true), "unknown"); // unknown takes priority
    }

    #[test]
    fn test_requires_relay() {
        assert!(!requires_relay("open"));
        assert!(requires_relay("symmetric"));
        assert!(requires_relay("full_cone"));
        assert!(requires_relay("unknown"));
    }

    #[test]
    fn test_build_nat_info() {
        let info = build_nat_info("open", "1.2.3.4".into(), 4001);
        assert!(info.is_public);
        assert_eq!(info.nat_type, "open");

        let info = build_nat_info("symmetric", "10.0.0.1".into(), 0);
        assert!(!info.is_public);
    }
}
