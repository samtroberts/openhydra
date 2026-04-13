//! OpenHydra P2P networking layer — rust-libp2p + PyO3.
//!
//! This crate provides the `openhydra_network` Python module via PyO3.
//! Phase A: standalone Rust crate with Kademlia DHT, identity, and types.
//! Phase B: full PyO3 bindings (P2PNode class).

pub mod behaviour;
pub mod proxy;
pub mod dht;
pub mod event_loop;
pub mod identity;
pub mod mdns;
pub mod nat;
pub mod node;
pub mod relay;
pub mod swarm;
pub mod transport;
pub mod types;

/// Python module entry point.
#[cfg(feature = "pyo3")]
mod python {
    use pyo3::prelude::*;

    #[pymodule]
    fn openhydra_network(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", "0.1.0")?;
        m.add_class::<crate::node::PyP2PNode>()?;
        Ok(())
    }
}
