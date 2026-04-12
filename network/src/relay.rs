//! Circuit Relay v2 helpers.
//!
//! libp2p's Circuit Relay v2 is fundamentally different from OpenHydra's
//! broken Python relay: the NATted peer opens an *outbound* connection to
//! the relay and holds it open (a "reservation"). Other peers reach the
//! NATted peer by connecting *through* the relay on that same connection.
//!
//! This module provides helpers for building relay multiaddrs and managing
//! relay reservations.

use libp2p::{Multiaddr, PeerId};

/// Build a relay circuit multiaddr for reaching a peer through a relay.
///
/// Format: `<relay_addr>/p2p/<relay_peer_id>/p2p-circuit/p2p/<target_peer_id>`
///
/// This is the address a coordinator would use to connect to a NATted peer.
pub fn relay_circuit_addr(
    relay_addr: &Multiaddr,
    relay_peer_id: &PeerId,
    target_peer_id: &PeerId,
) -> Multiaddr {
    relay_addr.clone()
        .with(libp2p::multiaddr::Protocol::P2p(*relay_peer_id))
        .with(libp2p::multiaddr::Protocol::P2pCircuit)
        .with(libp2p::multiaddr::Protocol::P2p(*target_peer_id))
}

/// Build a relay reservation address (what the NATted peer listens on).
///
/// Format: `<relay_addr>/p2p/<relay_peer_id>/p2p-circuit`
pub fn relay_reservation_addr(
    relay_addr: &Multiaddr,
    relay_peer_id: &PeerId,
) -> Multiaddr {
    relay_addr.clone()
        .with(libp2p::multiaddr::Protocol::P2p(*relay_peer_id))
        .with(libp2p::multiaddr::Protocol::P2pCircuit)
}

/// Known bootstrap relay servers (production Linode nodes).
pub const BOOTSTRAP_RELAYS: &[&str] = &[
    // US (Dallas)
    "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb",
    // EU (London)
    "/ip4/172.105.69.49/tcp/4001/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT",
    // AP (Singapore)
    "/ip4/172.104.164.98/tcp/4001/p2p/12D3KooWPgqZBgLZ1f94AQ7sbeyEz5UJ4jiT4d3zuQp2t61VLPZo",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relay_circuit_addr_format() {
        let relay_addr: Multiaddr = "/ip4/45.79.190.172/tcp/4001".parse().unwrap();
        let relay_peer = PeerId::random();
        let target_peer = PeerId::random();

        let circuit = relay_circuit_addr(&relay_addr, &relay_peer, &target_peer);
        let s = circuit.to_string();

        assert!(s.contains("/p2p-circuit/p2p/"));
        assert!(s.starts_with("/ip4/45.79.190.172/tcp/4001/p2p/"));
    }

    #[test]
    fn test_relay_reservation_addr_format() {
        let relay_addr: Multiaddr = "/ip4/172.105.69.49/tcp/4001".parse().unwrap();
        let relay_peer = PeerId::random();

        let reservation = relay_reservation_addr(&relay_addr, &relay_peer);
        let s = reservation.to_string();

        assert!(s.ends_with("/p2p-circuit"));
        assert!(s.starts_with("/ip4/172.105.69.49/tcp/4001/p2p/"));
    }
}
