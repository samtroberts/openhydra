//! OpenHydra P2P networking layer — rust-libp2p + PyO3.
//!
//! This crate provides the `openhydra_network` Python module via PyO3.
//! Phase A: standalone Rust crate with Kademlia DHT, identity, and types.
//! Phase B: full PyO3 bindings (P2PNode class).

pub mod activation;
pub mod behaviour;
pub mod dlpack;
pub mod proxy;

/// Prost-generated types from peer.proto.
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/openhydra.peer.rs"));
}
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
        // Initialize tracing subscriber so Rust info!/warn!/debug! macros
        // produce visible output (controlled by RUST_LOG env var).
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_target(false)
            .try_init();
        m.add("__version__", "0.1.0")?;
        m.add_class::<crate::node::PyP2PNode>()?;
        m.add_class::<crate::dlpack::PyRustTensor>()?;
        m.add_function(wrap_pyfunction!(decode_activation, m)?)?;
        Ok(())
    }

    /// Decode activation_packed bytes into a zero-copy RustTensor.
    ///
    /// Usage:
    ///     tensor = openhydra_network.decode_activation(packed_bytes)
    ///     hidden = mx.from_dlpack(tensor)  # zero-copy
    #[pyfunction]
    fn decode_activation(packed: Vec<u8>) -> PyResult<crate::dlpack::PyRustTensor> {
        crate::dlpack::rust_tensor_from_packed(packed)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }
}
