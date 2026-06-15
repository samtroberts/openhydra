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
pub mod routing_cache;
pub mod swarm;
pub mod tensor_stream;
pub mod transport;
pub mod types;

// Protocol core (canonical id §4, routing math §5, receipts §6, verify policy §7) now
// lives in the pure, synchronous `openhydra-protocol` crate (M2.3 workspace split).
// Re-export its modules at this crate's root so existing `crate::{model_id,router,
// receipts,verify}::…` paths in node.rs / the PyO3 glue keep resolving unchanged — the
// network crate is the async + FFI shell around this pure core.
pub use openhydra_protocol::{model_id, receipts, router, store, verify};

/// Non-pyo3 Rust API over the swarm — used by the pure-protocol `agent` crate (which
/// builds with `default-features = false`, no Python). Wraps `start_node` + the command
/// channel + the inbound proxy queue into a tidy synchronous handle.
pub mod handle;

/// Python module entry point.
#[cfg(feature = "pyo3")]
mod python {
    use ed25519_dalek::{Signature, SigningKey, VerifyingKey};
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
        // Co-signed receipts (protocol.md §6) — M2.1.
        m.add_function(wrap_pyfunction!(ed25519_public_key, m)?)?;
        m.add_function(wrap_pyfunction!(receipt_sign_consumer, m)?)?;
        m.add_function(wrap_pyfunction!(receipt_cosign_provider, m)?)?;
        m.add_function(wrap_pyfunction!(receipt_verify, m)?)?;
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

    // --- Co-signed receipts (protocol.md §6) — M2.1 FFI ---
    //
    // Keys and signatures cross the boundary as raw bytes (length-validated here):
    // 32-byte ed25519 signing-key seeds / public keys, 64-byte signatures, 16-byte
    // nonces. Raw bytes are the natural, allocation-light ed25519 representation.

    fn vk_from(bytes: &[u8]) -> PyResult<VerifyingKey> {
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("expected a 32-byte public key"))?;
        VerifyingKey::from_bytes(&arr)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid public key: {e}")))
    }

    fn sk_from(bytes: &[u8]) -> PyResult<SigningKey> {
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("expected a 32-byte signing key"))?;
        Ok(SigningKey::from_bytes(&arr))
    }

    fn sig_from(bytes: &[u8]) -> PyResult<Signature> {
        let arr: [u8; 64] = bytes
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("expected a 64-byte signature"))?;
        Ok(Signature::from_bytes(&arr))
    }

    fn payload_from(
        provider_pub: &[u8],
        consumer_pub: &[u8],
        model_id: &str,
        tokens: u64,
        nonce: &[u8],
        ts_unix_ms: u64,
    ) -> PyResult<crate::receipts::ReceiptPayload> {
        let nonce: [u8; 16] = nonce
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("expected a 16-byte nonce"))?;
        Ok(crate::receipts::ReceiptPayload {
            provider: vk_from(provider_pub)?,
            consumer: vk_from(consumer_pub)?,
            model_id: model_id.to_string(),
            tokens,
            nonce,
            ts_unix_ms,
        })
    }

    /// Derive the 32-byte ed25519 public key from a 32-byte signing-key seed.
    #[pyfunction]
    fn ed25519_public_key(py: Python<'_>, signing_key: Vec<u8>) -> PyResult<PyObject> {
        let pk = sk_from(&signing_key)?.verifying_key();
        Ok(pyo3::types::PyBytes::new(py, pk.as_bytes()).into())
    }

    /// Consumer-side signature over the receipt payload. Returns 64 bytes.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn receipt_sign_consumer(
        py: Python<'_>,
        consumer_signing_key: Vec<u8>,
        provider_pub: Vec<u8>,
        consumer_pub: Vec<u8>,
        model_id: &str,
        tokens: u64,
        nonce: Vec<u8>,
        ts_unix_ms: u64,
    ) -> PyResult<PyObject> {
        let ck = sk_from(&consumer_signing_key)?;
        let payload = payload_from(&provider_pub, &consumer_pub, model_id, tokens, &nonce, ts_unix_ms)?;
        let sig = crate::receipts::consumer_sign(&payload, &ck);
        Ok(pyo3::types::PyBytes::new(py, &sig.to_bytes()).into())
    }

    /// Provider-side co-signature over (payload ‖ consumer_sig). Returns 64 bytes.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn receipt_cosign_provider(
        py: Python<'_>,
        provider_signing_key: Vec<u8>,
        provider_pub: Vec<u8>,
        consumer_pub: Vec<u8>,
        model_id: &str,
        tokens: u64,
        nonce: Vec<u8>,
        ts_unix_ms: u64,
        consumer_sig: Vec<u8>,
    ) -> PyResult<PyObject> {
        let pk = sk_from(&provider_signing_key)?;
        let payload = payload_from(&provider_pub, &consumer_pub, model_id, tokens, &nonce, ts_unix_ms)?;
        let csig = sig_from(&consumer_sig)?;
        let sig = crate::receipts::provider_cosign(&payload, &csig, &pk);
        Ok(pyo3::types::PyBytes::new(py, &sig.to_bytes()).into())
    }

    /// Verify a full co-signed receipt. Returns None on success; raises ValueError
    /// ("receipt rejected: bad_consumer_sig" / "…bad_provider_sig") on rejection.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn receipt_verify(
        provider_pub: Vec<u8>,
        consumer_pub: Vec<u8>,
        model_id: &str,
        tokens: u64,
        nonce: Vec<u8>,
        ts_unix_ms: u64,
        consumer_sig: Vec<u8>,
        provider_sig: Vec<u8>,
    ) -> PyResult<()> {
        let payload = payload_from(&provider_pub, &consumer_pub, model_id, tokens, &nonce, ts_unix_ms)?;
        let receipt = crate::receipts::CoSignedReceipt {
            payload,
            consumer_sig: sig_from(&consumer_sig)?,
            provider_sig: sig_from(&provider_sig)?,
        };
        crate::receipts::verify_receipt(&receipt).map_err(|e| {
            let which = match e {
                crate::receipts::ReceiptError::BadConsumerSig => "bad_consumer_sig",
                crate::receipts::ReceiptError::BadProviderSig => "bad_provider_sig",
                crate::receipts::ReceiptError::ReplayedNonce => "replayed_nonce",
            };
            pyo3::exceptions::PyValueError::new_err(format!("receipt rejected: {which}"))
        })
    }
}
