//! Composed NetworkBehaviour for OpenHydra peers.
//!
//! Combines Kademlia DHT, Circuit Relay v2, DCUtR hole punching,
//! AutoNAT, Identify, and mDNS into a single behaviour.

use libp2p::swarm::NetworkBehaviour;
use libp2p::{autonat, dcutr, identify, kad, mdns, relay};

/// The composed behaviour for an OpenHydra peer node.
///
/// Bootstrap nodes additionally enable `relay::Behaviour` as a server
/// (accepting relay reservations from NATted peers). Regular peers only
/// run the relay client.
#[derive(NetworkBehaviour)]
pub struct OpenHydraBehaviour {
    /// Kademlia DHT for peer discovery.
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    /// Circuit Relay v2 client — connects through relays when behind NAT.
    pub relay_client: relay::client::Behaviour,
    /// DCUtR — Direct Connection Upgrade through Relay (hole punching).
    pub dcutr: dcutr::Behaviour,
    /// AutoNAT — automatic NAT type detection via bootstrap probes.
    pub autonat: autonat::Behaviour,
    /// Identify — exchange peer metadata on connect.
    pub identify: identify::Behaviour,
    /// mDNS — zero-config LAN peer discovery.
    pub mdns: mdns::tokio::Behaviour,
}
