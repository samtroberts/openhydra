//! DLPack zero-copy tensor wrapper for Rust → Python (MLX/PyTorch).
//!
//! Exposes a `RustTensor` Python class with `__dlpack__()` / `__dlpack_device__()`
//! protocol. The tensor wraps a heap-allocated `Vec<u8>` buffer and hands
//! a raw pointer to the consumer framework. Memory is freed ONLY when the
//! consumer calls the C-ABI deleter — NOT via Python GC or PyO3 ref counting.

use std::ffi::c_void;
use std::os::raw::c_int;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyRuntimeError;

// ── DLPack C ABI structs (from dlpack.h spec) ───────────────────────

/// DLPack device type.
#[repr(C)]
struct DLDevice {
    device_type: c_int,  // kDLCPU = 1
    device_id: c_int,
}

/// DLPack data type.
#[repr(C)]
struct DLDataType {
    code: u8,   // kDLFloat = 2
    bits: u8,   // 32
    lanes: u16, // 1
}

/// DLPack tensor descriptor.
#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: c_int,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

/// DLPack managed tensor — includes deleter for memory lifecycle.
#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

// ── Heap-allocated context that outlives the Python object ───────────

/// Owns the raw bytes and shape array until the DLPack consumer is done.
struct DlpackContext {
    /// The activation bytes. Payload starts at `offset`.
    _owned_bytes: Vec<u8>,
    /// Shape: [1, seq_len, hidden_size].
    shape: [i64; 3],
}

/// C-ABI deleter — called by MLX/PyTorch when the DLPack tensor is dropped.
/// Reclaims the heap-allocated context and the DLManagedTensor struct.
unsafe extern "C" fn dlpack_deleter(tensor: *mut DLManagedTensor) {
    if tensor.is_null() {
        return;
    }
    let ctx = (*tensor).manager_ctx as *mut DlpackContext;
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
    drop(Box::from_raw(tensor));
}

// ── Python-visible tensor class ─────────────────────────────────────

/// A zero-copy tensor backed by Rust-owned bytes.
///
/// Supports the DLPack protocol: `mx.from_dlpack(tensor)` or
/// `torch.from_dlpack(tensor)` wraps the raw pointer without copying.
///
/// The underlying memory is freed ONLY when the DLPack consumer calls
/// the deleter — not when this Python object is garbage collected.
#[cfg(feature = "pyo3")]
#[pyclass(name = "RustTensor")]
pub struct PyRustTensor {
    /// Raw activation bytes. `None` after `__dlpack__()` consumes it.
    raw_bytes: Option<Vec<u8>>,
    /// Byte offset where float32 payload starts.
    payload_offset: usize,
    /// Shape: [1, seq_len, hidden_size].
    shape: [i64; 3],
}

#[cfg(feature = "pyo3")]
impl PyRustTensor {
    /// Create from an ActivationBuffer.
    pub fn from_activation(
        raw_bytes: Vec<u8>,
        payload_offset: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            raw_bytes: Some(raw_bytes),
            payload_offset,
            shape: [1, seq_len as i64, hidden_size as i64],
        }
    }
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyRustTensor {
    /// DLPack protocol: return a PyCapsule wrapping a DLManagedTensor.
    ///
    /// This method can only be called ONCE — it moves the owned bytes
    /// into the heap-allocated DlpackContext.
    #[pyo3(signature = (stream=None))]
    fn __dlpack__(&mut self, py: Python<'_>, stream: Option<i64>) -> PyResult<PyObject> {
        let bytes = self
            .raw_bytes
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("RustTensor already consumed by __dlpack__"))?;

        // Pointer to the float32 payload (after header).
        let data_ptr = unsafe { bytes.as_ptr().add(self.payload_offset) as *mut c_void };

        // Heap-allocate the context. This is intentionally leaked from Rust's
        // perspective — ownership transfers to the DLPack consumer framework.
        let ctx = Box::into_raw(Box::new(DlpackContext {
            _owned_bytes: bytes,
            shape: self.shape,
        }));

        // Build DLManagedTensor on the heap (also leaked until deleter runs).
        let managed = Box::into_raw(Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: data_ptr,
                device: DLDevice {
                    device_type: 1, // kDLCPU
                    device_id: 0,
                },
                ndim: 3,
                dtype: DLDataType {
                    code: 2,  // kDLFloat
                    bits: 32,
                    lanes: 1,
                },
                shape: unsafe { &mut (*ctx).shape as *mut i64 },
                strides: std::ptr::null_mut(), // contiguous C-order
                byte_offset: 0,
            },
            manager_ctx: ctx as *mut c_void,
            deleter: Some(dlpack_deleter),
        }));

        // Wrap in a PyCapsule named "dltensor" — the DLPack protocol name.
        // IMPORTANT: the name must be a static C string that outlives the capsule.
        // Using a local CString would dangle after the function returns.
        static CAPSULE_NAME: &[u8] = b"dltensor\0";
        let capsule = unsafe {
            pyo3::ffi::PyCapsule_New(
                managed as *mut c_void,
                CAPSULE_NAME.as_ptr() as *const std::ffi::c_char,
                None,
            )
        };
        if capsule.is_null() {
            return Err(PyRuntimeError::new_err("PyCapsule_New failed"));
        }
        Ok(unsafe { PyObject::from_owned_ptr(py, capsule) })
    }

    /// DLPack device protocol.
    fn __dlpack_device__(&self) -> (i32, i32) {
        (1, 0) // (kDLCPU, device_id=0)
    }

    /// Shape as a Python tuple.
    #[getter]
    fn shape(&self) -> (i64, i64, i64) {
        (self.shape[0], self.shape[1], self.shape[2])
    }

    /// Number of float32 elements.
    #[getter]
    fn size(&self) -> i64 {
        self.shape[0] * self.shape[1] * self.shape[2]
    }
}

// ── DLPack import: Python tensor → packed bytes ─────────────────────

/// Import a tensor via DLPack and encode it to packed activation bytes.
///
/// Extracts the `DLManagedTensor*` from a PyCapsule, validates the tensor
/// is CPU/float32/contiguous, calls `encode_to_packed` for a single memcpy,
/// and renames the capsule to "used_dltensor" per the DLPack protocol.
///
/// # Safety
/// The capsule must be a valid PyCapsule containing a DLManagedTensor*.
#[cfg(feature = "pyo3")]
pub unsafe fn import_dlpack_and_encode(capsule_ptr: *mut pyo3::ffi::PyObject) -> Result<Vec<u8>, String> {
    // Extract DLManagedTensor from the PyCapsule.
    static DLTENSOR_NAME: &[u8] = b"dltensor\0";
    let managed_ptr = pyo3::ffi::PyCapsule_GetPointer(
        capsule_ptr,
        DLTENSOR_NAME.as_ptr() as *const std::ffi::c_char,
    ) as *mut DLManagedTensor;
    if managed_ptr.is_null() {
        return Err("DLPack capsule is empty or has wrong name".into());
    }

    let tensor = &(*managed_ptr).dl_tensor;

    // Validate device: must be CPU (kDLCPU = 1).
    if tensor.device.device_type != 1 {
        return Err(format!(
            "encode_activation requires CPU tensor, got device_type={}",
            tensor.device.device_type
        ));
    }

    // Validate dtype: must be float32 (code=2, bits=32, lanes=1).
    if tensor.dtype.code != 2 || tensor.dtype.bits != 32 || tensor.dtype.lanes != 1 {
        return Err(format!(
            "encode_activation requires float32, got code={} bits={} lanes={}",
            tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes
        ));
    }

    // Validate ndim: expect 2D [seq_len, hidden_size] or 3D [1, seq_len, hidden_size].
    let (seq_len, hidden_size) = match tensor.ndim {
        2 => {
            let shape = std::slice::from_raw_parts(tensor.shape, 2);
            (shape[0] as usize, shape[1] as usize)
        }
        3 => {
            let shape = std::slice::from_raw_parts(tensor.shape, 3);
            (shape[1] as usize, shape[2] as usize)
        }
        _ => {
            return Err(format!(
                "encode_activation expects 2D or 3D tensor, got ndim={}",
                tensor.ndim
            ));
        }
    };

    // Validate contiguous (strides must be NULL or C-contiguous).
    if !tensor.strides.is_null() {
        let strides = std::slice::from_raw_parts(tensor.strides, tensor.ndim as usize);
        // Check C-contiguous: last stride = 1, second-to-last = hidden_size, etc.
        let expected_last = 1i64;
        if *strides.last().unwrap_or(&0) != expected_last {
            return Err("encode_activation requires contiguous tensor (call .contiguous() first)".into());
        }
    }

    // Get the data pointer (applying byte_offset if non-zero).
    let data_ptr = (tensor.data as *const u8).add(tensor.byte_offset as usize) as *const f32;

    // Encode to packed bytes (single memcpy).
    let packed = crate::activation::encode_to_packed(data_ptr, seq_len, hidden_size);

    // Rename capsule to "used_dltensor" per DLPack protocol.
    // This prevents double-consumption.
    static USED_NAME: &[u8] = b"used_dltensor\0";
    pyo3::ffi::PyCapsule_SetName(
        capsule_ptr,
        USED_NAME.as_ptr() as *const std::ffi::c_char,
    );

    // Call the deleter if present to free the producer's memory.
    if let Some(deleter) = (*managed_ptr).deleter {
        deleter(managed_ptr);
    }

    Ok(packed)
}

// ── Public constructor (Rust-only, not exposed to Python directly) ───

/// Create a RustTensor from raw activation_packed bytes.
/// Used by the `decode_activation()` PyO3 function.
#[cfg(feature = "pyo3")]
pub fn rust_tensor_from_packed(packed: Vec<u8>) -> Result<PyRustTensor, String> {
    let buf = crate::activation::ActivationBuffer::from_packed(packed)?;
    let (raw_bytes, offset, seq_len, hidden_size) = buf.into_parts();
    Ok(PyRustTensor::from_activation(raw_bytes, offset, seq_len, hidden_size))
}
