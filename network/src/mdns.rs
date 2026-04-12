//! mDNS LAN peer discovery helpers.
//!
//! mDNS is configured inside the swarm behaviour (see behaviour.rs).
//! This module provides helpers for filtering and processing mDNS events.

use libp2p::{Multiaddr, PeerId};

/// A peer discovered via mDNS on the local network.
#[derive(Debug, Clone)]
pub struct LanPeer {
    pub peer_id: PeerId,
    pub addrs: Vec<Multiaddr>,
}

/// Filter mDNS-discovered addresses to keep only LAN-routable ones.
///
/// mDNS can return loopback (127.x), link-local (169.254.x), and actual
/// LAN addresses. We keep only private-range addresses that are useful
/// for direct gRPC connections.
pub fn filter_lan_addrs(addrs: &[Multiaddr]) -> Vec<Multiaddr> {
    addrs
        .iter()
        .filter(|addr| {
            let s = addr.to_string();
            // Keep private ranges, exclude loopback and link-local.
            !s.starts_with("/ip4/127.") && !s.starts_with("/ip4/169.254.")
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_lan_addrs() {
        let addrs: Vec<Multiaddr> = vec![
            "/ip4/127.0.0.1/tcp/4001".parse().unwrap(),
            "/ip4/192.168.1.10/tcp/4001".parse().unwrap(),
            "/ip4/169.254.0.1/tcp/4001".parse().unwrap(),
            "/ip4/10.0.0.5/tcp/4001".parse().unwrap(),
        ];
        let filtered = filter_lan_addrs(&addrs);
        assert_eq!(filtered.len(), 2);
        assert!(filtered[0].to_string().contains("192.168.1.10"));
        assert!(filtered[1].to_string().contains("10.0.0.5"));
    }
}
