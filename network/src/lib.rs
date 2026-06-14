//! OpenHydra P2P networking layer — rust-libp2p + PyO3.
//!
//! This crate provides the `openhydra_network` Python module via PyO3.
//! Phase A: standalone Rust crate with Kademlia DHT, identity, and types.
//! Phase B: full PyO3 bindings (P2PNode class).

pub mod activation;
pub mod batcher;
pub mod behaviour;
pub mod dispatcher;
pub mod dlpack;
pub mod forward_msg;
pub mod ipc;
pub mod ipc_codec;
pub mod proxy;
pub mod ring;
pub mod sampler_bridge;

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
pub mod tensor_stream;
pub mod transport;
pub mod types;

/// Canonical model identity & equivalence (protocol.md §4) — M1.1. Pure Rust
/// (sha2/hex only); will be extracted to the `protocol` crate in the iterative
/// workspace refactor.
pub mod model_id;

/// Router scoring & ranking (protocol.md §5) — M1.3. Ports the peer-ranking logic
/// from `coordinator/peer_selector.py`; resolve/route stages land on top.
pub mod router;

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
        m.add_function(wrap_pyfunction!(encode_activation, m)?)?;
        // Canonical model identity (protocol.md §4) — M1.1.
        m.add_function(wrap_pyfunction!(canonical_id_from_hf, m)?)?;
        m.add_function(wrap_pyfunction!(is_compatible, m)?)?;
        m.add_function(wrap_pyfunction!(chat_template_hash, m)?)?;
        m.add_function(wrap_pyfunction!(parse_hf_model_name, m)?)?;
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

    /// Encode a tensor into packed activation bytes via DLPack (zero-copy).
    ///
    /// Accepts any tensor implementing `__dlpack__()` (PyTorch, MLX via torch bridge).
    /// Extracts the raw float32 pointer, writes an 8-byte header (seq_len, hidden_size),
    /// and performs a single memcpy of the float buffer. Returns Python `bytes`.
    ///
    /// The tensor must be:
    ///   - CPU device
    ///   - float32 dtype
    ///   - contiguous (call `.contiguous()` first if needed)
    ///   - 2D [seq_len, hidden_size] or 3D [1, seq_len, hidden_size]
    ///
    /// Usage:
    ///     packed = openhydra_network.encode_activation(hidden_state)
    ///     # packed is bytes, ready for protobuf activation_packed field
    #[pyfunction]
    fn encode_activation(py: Python<'_>, tensor: PyObject) -> PyResult<PyObject> {
        // Call tensor.__dlpack__() to get the PyCapsule.
        let capsule = tensor.call_method0(py, "__dlpack__")?;
        let capsule_ptr = capsule.as_ptr();

        // Import the tensor via DLPack and encode to packed bytes.
        let packed = unsafe { crate::dlpack::import_dlpack_and_encode(capsule_ptr) }
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        // Return as Python bytes.
        Ok(pyo3::types::PyBytes::new(py, &packed).into())
    }

    // --- Canonical model identity (protocol.md §4) — M1.1 ---

    /// Compute the canonical model id from an HF id, runtime quant, and the
    /// engine's live chat template. Returns `family/params/quant/template_hash`.
    /// Raises `ValueError` on an empty template / component.
    #[pyfunction]
    fn canonical_id_from_hf(hf_model_id: &str, quant: &str, chat_template: &str) -> PyResult<String> {
        crate::model_id::canonical_id_from_hf(hf_model_id, quant, chat_template)
            .map(|c| c.to_string())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// True if a provider's concrete canonical id is compatible with a (possibly
    /// wildcarded) request id. Malformed ids return False.
    #[pyfunction]
    fn is_compatible(request: &str, provider: &str) -> bool {
        crate::model_id::is_compatible(request, provider)
    }

    /// Stable 16-hex-char hash of a tokenizer chat template.
    #[pyfunction]
    fn chat_template_hash(template: &str) -> String {
        crate::model_id::chat_template_hash(template)
    }

    /// Heuristically split an HF model id into `(family, params, variants)`.
    #[pyfunction]
    fn parse_hf_model_name(hf_model_id: &str) -> (String, String, Vec<String>) {
        crate::model_id::parse_hf_model_name(hf_model_id)
    }
}
