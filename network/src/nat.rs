//! NAT detection helpers.
//!
//! AutoNAT v2 runs inside the swarm and probes per-address reachability via
//! bootstrap (v2 server) peers. The event handler (`event_loop`) translates those
//! per-address verdicts directly into the OpenHydra `NatInfo`; this module holds
//! the small shared helpers (`requires_relay`, `build_nat_info`).

use crate::types::NatInfo;

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
    // Classify by address family.
    let (external_ipv4, external_ipv6) = if external_ip.contains(':') {
        (String::new(), external_ip.clone())
    } else if !external_ip.is_empty() {
        (external_ip.clone(), String::new())
    } else {
        (String::new(), String::new())
    };
    NatInfo {
        nat_type: nat_type.to_string(),
        external_ip,
        external_ipv4,
        external_ipv6,
        external_port,
        is_public,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
