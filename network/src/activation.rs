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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_packed_roundtrip() {
        // Build a packed activation: header [2.0, 3.0] + 6 payload floats
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
}
