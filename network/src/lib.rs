//! OpenHydra P2P networking layer — rust-libp2p.
//!
//! The async + transport shell around the pure `openhydra-protocol` core: Kademlia
//! DHT, Circuit Relay v2, AutoNAT, DCUtR, mDNS, and the request/response proxy.
//! Drives the swarm through a synchronous [`handle::NetworkHandle`] for the
//! `openhydra-agent` host (and the `openhydra-bootstrap` binary).

pub mod behaviour;
pub mod card;
pub mod proxy;

/// Prost-generated types from peer.proto.
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/openhydra.peer.rs"));
}
pub mod dht;
pub mod event_loop;
pub mod identity;
pub mod mdns;
/// Swarm membership credentials (M3): a group keypair whose owner signs credentials over member
/// public keys — the "private key for private sharing" crypto core. Sibling of [`card`].
pub mod membership;
pub mod nat;
pub mod node;
pub mod pcp;
pub mod registry;
pub mod registry_proto;
pub mod relay;
pub mod routing_cache;
pub mod swarm;
pub mod transport;
pub mod types;

// Protocol core (canonical id §4, routing math §5, receipts §6, verify policy §7) now
// lives in the pure, synchronous `openhydra-protocol` crate (M2.3 workspace split).
// Re-export its modules at this crate's root so existing `crate::{model_id,router,
// receipts,verify}::…` paths in node.rs keep resolving unchanged — the network
// crate is the async shell around this pure core.
pub use openhydra_protocol::{model_id, receipts, router, store, verify};

/// Synchronous Rust API over the swarm — used by the `agent` host and the
/// `openhydra-bootstrap` binary. Wraps `start_node` + the command channel + the
/// inbound proxy queue into a tidy synchronous handle.
pub mod handle;
