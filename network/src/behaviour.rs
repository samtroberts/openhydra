//! Composed NetworkBehaviour for OpenHydra peers.
//!
//! Combines Kademlia DHT, Circuit Relay v2, DCUtR hole punching,
//! AutoNAT, Identify, mDNS, and gRPC proxy into a single behaviour.

use libp2p::request_response;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{autonat, dcutr, identify, kad, mdns, relay};

use crate::proxy::GrpcProxyCodec;

/// The composed behaviour for an OpenHydra peer node.
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
    /// gRPC proxy — tunnels gRPC through libp2p for cross-ISP inference.
    pub grpc_proxy: request_response::Behaviour<GrpcProxyCodec>,
}
