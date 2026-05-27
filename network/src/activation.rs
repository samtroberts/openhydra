//! Zero-copy activation buffer extraction from protobuf.
//!
//! Extracts the float32 activation payload from a ForwardRequest's
//! `activation_packed` bytes field without per-element iteration.
//! Uses `bytemuck::cast_slice` for O(1) byte-to-float reinterpretation.

/// Activation buffer with zero-copy float32 view over owned bytes.
pub struct ActivationBuffer {
    /// Owned bytes — the raw activation_packed payload.
    raw_bytes: Vec<u8>,
    /// Byte offset where the float32 payload starts (after 8-byte header).
    payload_offset: usize,
    /// Sequence length from the header.
    pub seq_len: usize,
    /// Hidden dimension from the header.
    pub hidden_size: usize,
}

impl ActivationBuffer {
    /// Extract from raw `activation_packed` bytes.
    ///
    /// Format: `[seq_len_f32_le, hidden_size_f32_le, payload_f32_le...]`
    /// The first 8 bytes are the header (2 × little-endian float32).
    /// The remaining bytes are the float32 payload.
    ///
    /// Zero-copy: takes ownership of the `Vec<u8>`, no data is copied.
    pub fn from_packed(packed: Vec<u8>) -> Result<Self, String> {
        if packed.len() < 8 {
            return Err(format!(
                "activation_packed too short for header: {} bytes",
                packed.len()
            ));
        }
        // Parse header: 2 × little-endian float32.
        let seq_len = f32::from_le_bytes(
            packed[0..4]
                .try_into()
                .map_err(|_| "header parse failed")?,
        ) as usize;
        let hidden_size = f32::from_le_bytes(
            packed[4..8]
                .try_into()
                .map_err(|_| "header parse failed")?,
        ) as usize;

        let payload_bytes = packed.len() - 8;
        let expected_bytes = seq_len * hidden_size * 4;
        if payload_bytes != expected_bytes {
            return Err(format!(
                "activation size mismatch: got {} payload bytes, expected {} (seq={} hidden={})",
                payload_bytes, expected_bytes, seq_len, hidden_size
            ));
        }

        // Alignment check: activation_packed from struct.pack is contiguous
        // and 1-byte aligned. bytemuck::cast_slice requires the source to be
        // aligned to f32 (4 bytes). Vec<u8> from prost is heap-allocated and
        // typically 8-byte aligned, but we verify at runtime.
        let payload_slice = &packed[8..];
        if (payload_slice.as_ptr() as usize) % std::mem::align_of::<f32>() != 0 {
            return Err("activation payload not f32-aligned".into());
        }

        Ok(Self {
            raw_bytes: packed,
            payload_offset: 8,
            seq_len,
            hidden_size,
        })
    }

    /// Zero-copy view of the payload as `&[f32]`.
    /// O(1) — no iteration, no allocation.
    pub fn as_floats(&self) -> &[f32] {
        bytemuck::cast_slice(&self.raw_bytes[self.payload_offset..])
    }

    /// Consume self and return the owned bytes + offset.
    /// Used for DLPack handoff where the consumer takes ownership.
    pub fn into_parts(self) -> (Vec<u8>, usize, usize, usize) {
        (self.raw_bytes, self.payload_offset, self.seq_len, self.hidden_size)
    }
}

/// Encode a float32 buffer into the packed activation format.
///
/// Inverse of `ActivationBuffer::from_packed`. Writes an 8-byte header
/// (seq_len, hidden_size as little-endian float32) followed by a memcpy
/// of the raw float data.
///
/// # Safety
/// `data_ptr` must point to `seq_len * hidden_size` contiguous float32s.
pub unsafe fn encode_to_packed(
    data_ptr: *const f32,
    seq_len: usize,
    hidden_size: usize,
) -> Vec<u8> {
    let n_floats = seq_len * hidden_size;
    let payload_bytes = n_floats * 4;
    let mut packed = Vec::with_capacity(8 + payload_bytes);

    // Header: seq_len and hidden_size as little-endian float32.
    packed.extend_from_slice(&(seq_len as f32).to_le_bytes());
    packed.extend_from_slice(&(hidden_size as f32).to_le_bytes());

    // Payload: single memcpy of the float buffer.
    let src = std::slice::from_raw_parts(data_ptr as *const u8, payload_bytes);
    packed.extend_from_slice(src);

    packed
}

// ── CP-1: Standalone activation codec (replaces Python activation_codec.py) ──

/// Pack a slice of f32 values into little-endian bytes.
///
/// Bit-for-bit identical to Python's `struct.pack(f'<{n}f', *values)` and
/// `peer/activation_codec.py::pack_fp32()`.
pub fn pack_fp32(values: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.len() * 4);
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Unpack little-endian bytes into f32 values.
///
/// Bit-for-bit identical to Python's `struct.unpack(f'<{n}f', data)` and
/// `peer/activation_codec.py::unpack_fp32()`.
pub fn unpack_fp32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Per-tensor symmetric INT8 quantization (OpenHydra INT8 spec v1.0).
///
/// Matches `peer/activation_codec.py::quantize_int8()` bit-for-bit:
/// - `scale = absmax / 127.0`
/// - Round-half-to-even (banker's rounding)
/// - Signed → unsigned via `q_byte = q_signed + 128`
/// - Empty input → `(b"", 0.0)`
/// - All-zero input → `(bytes(n), 0.0)`
///
/// Returns `(packed_bytes, scale)`.
pub fn quantize_int8(values: &[f32]) -> (Vec<u8>, f32) {
    if values.is_empty() {
        return (Vec::new(), 0.0);
    }

    // Find absmax, treating NaN/inf as 0.
    let absmax = values
        .iter()
        .map(|v| {
            let a = v.abs();
            if a.is_finite() { a } else { 0.0 }
        })
        .fold(0.0f32, f32::max);

    if absmax == 0.0 {
        // Match Python: `bytes(n)` → all-zero bytes, scale=0.0.
        // Dequant with scale=0 returns all zeros regardless of byte value.
        return (vec![0u8; values.len()], 0.0);
    }

    let scale = absmax / 127.0;
    let inv_scale = 1.0 / scale;

    let packed: Vec<u8> = values
        .iter()
        .map(|v| {
            let v = if v.is_finite() { *v } else { 0.0 };
            let q = bankers_round(v * inv_scale);
            let q = q.max(-127.0).min(127.0) as i32;
            (q + 128) as u8
        })
        .collect();

    (packed, scale)
}

/// Per-tensor symmetric INT8 dequantization.
///
/// Matches `peer/activation_codec.py::dequantize_int8()` bit-for-bit.
pub fn dequantize_int8(data: &[u8], scale: f32) -> Vec<f32> {
    if data.is_empty() {
        return Vec::new();
    }
    if scale == 0.0 {
        return vec![0.0; data.len()];
    }
    data.iter()
        .map(|&b| ((b as i32) - 128) as f32 * scale)
        .collect()
}

/// Banker's rounding (round-half-to-even).
///
/// Matches Python's `round()` for integer targets and numpy's `np.round()`.
/// Critical for bit-for-bit compatibility with the Python INT8 codec.
#[inline]
fn bankers_round(v: f32) -> f32 {
    // The standard Rust f32::round() uses round-half-away-from-zero,
    // which differs from Python's round-half-to-even. We must match
    // Python's behaviour for wire compatibility.
    let rounded = v.round();
    let diff = (v - rounded).abs();
    // Check if exactly on the 0.5 boundary.
    if (diff - 0.0).abs() < f32::EPSILON {
        // Not on boundary — standard rounding is correct.
        rounded
    } else if (v.fract().abs() - 0.5).abs() < f32::EPSILON {
        // Exactly on 0.5 boundary — round to even.
        let floor = v.floor();
        let ceil = v.ceil();
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            ceil
        }
    } else {
        rounded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Legacy ActivationBuffer tests ──────────────────────────────────

    #[test]
    fn test_from_packed_roundtrip() {
        let seq_len: f32 = 2.0;
        let hidden_size: f32 = 3.0;
        let payload: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut packed = Vec::with_capacity(8 + 24);
        packed.extend_from_slice(&seq_len.to_le_bytes());
        packed.extend_from_slice(&hidden_size.to_le_bytes());
        for v in &payload {
            packed.extend_from_slice(&v.to_le_bytes());
        }

        let buf = ActivationBuffer::from_packed(packed).unwrap();
        assert_eq!(buf.seq_len, 2);
        assert_eq!(buf.hidden_size, 3);
        assert_eq!(buf.as_floats(), &payload);
    }

    #[test]
    fn test_from_packed_size_mismatch() {
        let mut packed = Vec::new();
        packed.extend_from_slice(&2.0f32.to_le_bytes()); // seq=2
        packed.extend_from_slice(&3.0f32.to_le_bytes()); // hidden=3
        packed.extend_from_slice(&[0u8; 20]); // 20 bytes != 2*3*4=24

        assert!(ActivationBuffer::from_packed(packed).is_err());
    }

    #[test]
    fn test_from_packed_too_short() {
        assert!(ActivationBuffer::from_packed(vec![0, 1, 2, 3]).is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let payload: [f32; 6] = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let packed = unsafe { super::encode_to_packed(payload.as_ptr(), 2, 3) };

        let buf = ActivationBuffer::from_packed(packed).unwrap();
        assert_eq!(buf.seq_len, 2);
        assert_eq!(buf.hidden_size, 3);
        assert_eq!(buf.as_floats(), &payload);
    }

    // ── CP-1: pack/unpack FP32 tests ───────────────────────────────────

    #[test]
    fn test_pack_fp32_roundtrip() {
        let values = vec![1.0f32, -2.5, 3.14, 0.0, -1e10, f32::MIN_POSITIVE];
        let packed = pack_fp32(&values);
        assert_eq!(packed.len(), values.len() * 4);
        let unpacked = unpack_fp32(&packed);
        assert_eq!(unpacked.len(), values.len());
        for (orig, dec) in values.iter().zip(unpacked.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits(), "{orig} != {dec}");
        }
    }

    #[test]
    fn test_pack_fp32_empty() {
        assert!(pack_fp32(&[]).is_empty());
        assert!(unpack_fp32(&[]).is_empty());
    }

    #[test]
    fn test_pack_fp32_single() {
        let packed = pack_fp32(&[42.0]);
        assert_eq!(packed.len(), 4);
        let unpacked = unpack_fp32(&packed);
        assert_eq!(unpacked, vec![42.0]);
    }

    #[test]
    fn test_pack_fp32_large_activation() {
        // Simulate a 896-dim hidden state.
        let values: Vec<f32> = (0..896).map(|i| (i as f32) * 0.001).collect();
        let packed = pack_fp32(&values);
        let unpacked = unpack_fp32(&packed);
        assert_eq!(values.len(), unpacked.len());
        for (i, (o, d)) in values.iter().zip(unpacked.iter()).enumerate() {
            assert_eq!(
                o.to_bits(),
                d.to_bits(),
                "bit mismatch at index {i}: {o} vs {d}"
            );
        }
    }

    // ── CP-1: INT8 quantization tests ──────────────────────────────────

    #[test]
    fn test_int8_basic_roundtrip() {
        let values = vec![0.5f32, -0.3, 1.0, -1.0, 0.0, 0.25];
        let (packed, scale) = quantize_int8(&values);
        let restored = dequantize_int8(&packed, scale);

        assert_eq!(restored.len(), values.len());
        for (orig, rec) in values.iter().zip(restored.iter()) {
            assert!(
                (orig - rec).abs() < 0.02,
                "INT8 roundtrip: {orig} != {rec}"
            );
        }
    }

    #[test]
    fn test_int8_empty() {
        let (packed, scale) = quantize_int8(&[]);
        assert!(packed.is_empty());
        assert_eq!(scale, 0.0);
        let restored = dequantize_int8(&packed, scale);
        assert!(restored.is_empty());
    }

    #[test]
    fn test_int8_all_zeros() {
        let values = vec![0.0f32; 10];
        let (packed, scale) = quantize_int8(&values);
        assert_eq!(scale, 0.0);
        let restored = dequantize_int8(&packed, scale);
        assert!(restored.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_int8_single() {
        let (packed, scale) = quantize_int8(&[0.7]);
        let restored = dequantize_int8(&packed, scale);
        assert_eq!(restored.len(), 1);
        assert!((restored[0] - 0.7).abs() < 0.02);
    }

    #[test]
    fn test_int8_negative_preserved() {
        let values = vec![-0.9f32, -0.5, -0.1];
        let (packed, scale) = quantize_int8(&values);
        let restored = dequantize_int8(&packed, scale);
        for (orig, rec) in values.iter().zip(restored.iter()) {
            assert!(*rec < 0.0, "Sign lost: {orig} -> {rec}");
            assert!((orig - rec).abs() < 0.02);
        }
    }

    #[test]
    fn test_int8_compression_ratio() {
        let values: Vec<f32> = (0..4096).map(|i| i as f32 / 100.0).collect();
        let (packed, _scale) = quantize_int8(&values);
        let original_bytes = values.len() * 4; // FP32
        let compressed_bytes = packed.len() + 4; // + scale
        let ratio = original_bytes as f64 / compressed_bytes as f64;
        assert!(
            ratio > 3.5,
            "Expected >3.5x compression, got {ratio:.1}x"
        );
    }

    #[test]
    fn test_int8_byte_packing() {
        let values = vec![1.0f32, -1.0, 0.5];
        let (packed, scale) = quantize_int8(&values);
        assert_eq!(packed.len(), 3); // 1 byte per value
        assert!(scale > 0.0);
    }

    #[test]
    fn test_int8_large_range() {
        let values = vec![-1000.0f32, 500.0, 0.0, 999.5, -0.001];
        let (packed, scale) = quantize_int8(&values);
        let restored = dequantize_int8(&packed, scale);
        assert_eq!(restored.len(), values.len());
        for (orig, rec) in values.iter().zip(restored.iter()) {
            assert!(
                (orig - rec).abs() < orig.abs() * 0.01 + 0.1,
                "INT8 large range: {orig} != {rec}"
            );
        }
    }

    #[test]
    fn test_int8_unsigned_storage_spec() {
        // Verify the unsigned byte storage format: q_byte = q_signed + 128.
        // All-zero input → bytes(n) = all zero bytes, scale=0.0 (Python compat).
        let (packed, scale) = quantize_int8(&[0.0, 0.0, 0.0]);
        assert!(packed.iter().all(|&b| b == 0));
        assert_eq!(scale, 0.0);

        // Positive max → q_signed=127 → q_byte=255.
        let (packed, _) = quantize_int8(&[1.0]);
        assert_eq!(packed[0], 255);

        // Negative max → q_signed=-127 → q_byte=1.
        let (packed, _) = quantize_int8(&[-1.0]);
        assert_eq!(packed[0], 1);
    }

    // ── Banker's rounding tests ────────────────────────────────────────

    #[test]
    fn test_bankers_round() {
        // Standard cases (not on boundary).
        assert_eq!(bankers_round(2.3), 2.0);
        assert_eq!(bankers_round(2.7), 3.0);
        assert_eq!(bankers_round(-2.3), -2.0);
        assert_eq!(bankers_round(-2.7), -3.0);

        // Boundary cases: round to even.
        assert_eq!(bankers_round(0.5), 0.0); // 0 is even
        assert_eq!(bankers_round(1.5), 2.0); // 2 is even
        assert_eq!(bankers_round(2.5), 2.0); // 2 is even
        assert_eq!(bankers_round(3.5), 4.0); // 4 is even
        assert_eq!(bankers_round(-0.5), 0.0);
        assert_eq!(bankers_round(-1.5), -2.0);
    }
}
