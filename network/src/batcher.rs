//! CP-4: Continuous Batching — heterogeneous-safe request accumulator.
//!
//! Groups incoming ForwardMsg payloads by compatible tensor characteristics
//! so they can be batched into a single GPU kernel call.
//!
//! The Heterogeneous Constraint: not all activations can be batched together.
//! Two activations are batch-compatible only when they share the same:
//!   - `layer_start` (same model shard)
//!   - `activation_dtype` (FP32/FP16/INT8 — GPU kernel is dtype-specific)
//!   - `is_prefill` flag (prefill and decode use different attention masks)
//!   - `draft_block` flag (speculative draft blocks need separate handling)
//!
//! Flush triggers (whichever fires first):
//!   - **Size-bound**: batch reaches `max_batch_size` items (default: 4)
//!   - **Time-bound**: oldest item in the batch has waited `max_wait` (default: 5ms)
//!
//! Multi-Peer Batching Realities (from the migration plan):
//!   In a 2-peer ring, both sessions arrive at each peer every token —
//!   perfect alignment. In a 4-peer ring, sessions rarely co-locate,
//!   yielding ~45% batch efficiency. The time-bound helps: long cross-ISP
//!   RTT (~180ms/hop) gives the batcher time to accumulate requests.

use std::collections::HashMap;

use tracing::{debug, info};

use crate::ipc_codec::ActivationDtype;

// ── Batch key ────────────────────────────────────────────────────────

/// Grouping key for batch-compatible activations.
///
/// Two ForwardMsg payloads can share a GPU kernel call only when all
/// four fields match. The key is cheap to hash and compare.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BatchKey {
    /// First layer index of the shard (must match — different shards
    /// process different transformer blocks).
    pub layer_start: u32,
    /// Activation data type (FP32/FP16/INT8 — kernel is dtype-specific).
    pub activation_dtype: DtypeTag,
    /// Whether this is a prefill pass (vs autoregressive decode).
    /// Prefill uses full causal masks; decode uses single-token masks.
    pub is_prefill: bool,
    /// Whether this is a speculative draft block.
    /// Draft blocks run the draft model, not the target model.
    pub draft_block: bool,
}

/// Dtype tag for the batch key.
///
/// Separate from `ActivationDtype` to derive Hash/Eq (needed for HashMap key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtypeTag {
    Fp32,
    Fp16,
    Int8,
}

impl From<ActivationDtype> for DtypeTag {
    fn from(dt: ActivationDtype) -> Self {
        match dt {
            ActivationDtype::Fp32 => Self::Fp32,
            ActivationDtype::Fp16 => Self::Fp16,
            ActivationDtype::Int8 => Self::Int8,
        }
    }
}

impl From<DtypeTag> for ActivationDtype {
    fn from(tag: DtypeTag) -> Self {
        match tag {
            DtypeTag::Fp32 => Self::Fp32,
            DtypeTag::Fp16 => Self::Fp16,
            DtypeTag::Int8 => Self::Int8,
        }
    }
}

// ── Batch item ───────────────────────────────────────────────────────

/// A single item waiting in a batch slot.
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Original request_id for response routing.
    pub request_id: String,
    /// Session ID (for KV cache correlation).
    pub session_id: String,
    /// Raw activation bytes (FP32/FP16/INT8).
    pub activation: Vec<u8>,
    /// Activation shape: [batch, seq_len, hidden_dim].
    pub activation_shape: Vec<u32>,
    /// Timestamp when this item entered the batcher.
    pub enqueued_at: std::time::Instant,
}

// ── Flushed batch ────────────────────────────────────────────────────

/// A batch ready for dispatch to the IPC bridge / GPU worker.
#[derive(Debug, Clone)]
pub struct FlushedBatch {
    /// The batch key shared by all items.
    pub key: BatchKey,
    /// Items in the batch (1..=max_batch_size).
    pub items: Vec<BatchItem>,
    /// Reason the batch was flushed.
    pub reason: FlushReason,
}

/// Why a batch was flushed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushReason {
    /// Batch reached `max_batch_size`.
    SizeBound,
    /// Oldest item exceeded `max_wait`.
    TimeBound,
    /// Explicit flush requested (e.g., session ending).
    Manual,
}

// ── Batcher configuration ────────────────────────────────────────────

/// Configuration for the batch accumulator.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum items per batch before forced flush.
    pub max_batch_size: usize,
    /// Maximum time an item can wait before forced flush.
    pub max_wait: std::time::Duration,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 4,
            // C2: Reduced from 5ms to 1ms — caps per-hop queue delay at ~1ms.
            max_wait: std::time::Duration::from_millis(1),
        }
    }
}

// ── Batcher ──────────────────────────────────────────────────────────

/// Heterogeneous-safe batch accumulator.
///
/// Maintains separate pending queues per `BatchKey`. When a new item
/// arrives, it checks for compatible pending items. If a batch reaches
/// `max_batch_size` or the oldest item has waited `max_wait`, the batch
/// is flushed.
///
/// Thread-safety: intended for single-threaded use from the tokio event
/// loop (same pattern as `RingManager` and `Dispatcher`).
pub struct Batcher {
    config: BatcherConfig,
    /// Pending items grouped by batch key.
    pending: HashMap<BatchKey, Vec<BatchItem>>,
}

impl Batcher {
    /// Create a new batcher with the given configuration.
    pub fn new(config: BatcherConfig) -> Self {
        Self {
            config,
            pending: HashMap::new(),
        }
    }

    /// Create a batcher with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BatcherConfig::default())
    }

    /// Add an item to the batcher.
    ///
    /// Returns `Some(FlushedBatch)` if the addition caused a size-bound
    /// flush. The caller should also periodically call `flush_expired()`
    /// to handle time-bound flushes.
    pub fn add(&mut self, key: BatchKey, item: BatchItem) -> Option<FlushedBatch> {
        let items = self.pending.entry(key.clone()).or_default();
        items.push(item);

        if items.len() >= self.config.max_batch_size {
            // Size-bound flush.
            let batch_items = self.pending.remove(&key).unwrap();
            info!(
                ?key,
                count = batch_items.len(),
                "batch flushed (size bound)"
            );
            return Some(FlushedBatch {
                key,
                items: batch_items,
                reason: FlushReason::SizeBound,
            });
        }

        None
    }

    /// Flush all batches whose oldest item has exceeded `max_wait`.
    ///
    /// Returns all expired batches. Should be called periodically
    /// (e.g., on a tokio::time::interval matching `max_wait`).
    pub fn flush_expired(&mut self) -> Vec<FlushedBatch> {
        let now = std::time::Instant::now();
        let max_wait = self.config.max_wait;
        let mut flushed = Vec::new();

        // Collect keys to flush (can't mutate while iterating).
        let expired_keys: Vec<BatchKey> = self
            .pending
            .iter()
            .filter(|(_, items)| {
                items
                    .first()
                    .map_or(false, |item| now.duration_since(item.enqueued_at) >= max_wait)
            })
            .map(|(key, _)| key.clone())
            .collect();

        for key in expired_keys {
            if let Some(items) = self.pending.remove(&key) {
                if !items.is_empty() {
                    debug!(
                        ?key,
                        count = items.len(),
                        "batch flushed (time bound)"
                    );
                    flushed.push(FlushedBatch {
                        key,
                        items,
                        reason: FlushReason::TimeBound,
                    });
                }
            }
        }

        flushed
    }

    /// Manually flush all pending batches (e.g., on shutdown).
    pub fn flush_all(&mut self) -> Vec<FlushedBatch> {
        let mut flushed = Vec::new();
        for (key, items) in self.pending.drain() {
            if !items.is_empty() {
                flushed.push(FlushedBatch {
                    key,
                    items,
                    reason: FlushReason::Manual,
                });
            }
        }
        flushed
    }

    /// Number of distinct batch keys with pending items.
    pub fn pending_keys(&self) -> usize {
        self.pending.len()
    }

    /// Total number of items across all pending batches.
    pub fn pending_items(&self) -> usize {
        self.pending.values().map(|v| v.len()).sum()
    }

    /// Check if any batches have items pending.
    pub fn has_pending(&self) -> bool {
        self.pending.values().any(|v| !v.is_empty())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fp32_key(layer_start: u32) -> BatchKey {
        BatchKey {
            layer_start,
            activation_dtype: DtypeTag::Fp32,
            is_prefill: false,
            draft_block: false,
        }
    }

    fn make_item(request_id: &str) -> BatchItem {
        BatchItem {
            request_id: request_id.into(),
            session_id: "sess-001".into(),
            activation: vec![0u8; 64],
            activation_shape: vec![1, 1, 16],
            enqueued_at: std::time::Instant::now(),
        }
    }

    // ── BatchKey grouping ────────────────────────────────────────────

    #[test]
    fn test_same_key_groups_together() {
        let mut batcher = Batcher::new(BatcherConfig {
            max_batch_size: 4,
            max_wait: std::time::Duration::from_millis(5),
        });

        let key = fp32_key(0);

        // Add 3 items with the same key — no flush yet.
        assert!(batcher.add(key.clone(), make_item("req-1")).is_none());
        assert!(batcher.add(key.clone(), make_item("req-2")).is_none());
        assert!(batcher.add(key.clone(), make_item("req-3")).is_none());

        assert_eq!(batcher.pending_items(), 3);
        assert_eq!(batcher.pending_keys(), 1);

        // 4th item triggers size-bound flush.
        let batch = batcher.add(key.clone(), make_item("req-4"));
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.items.len(), 4);
        assert_eq!(batch.reason, FlushReason::SizeBound);
        assert_eq!(batch.key, key);

        // Queue is now empty.
        assert_eq!(batcher.pending_items(), 0);
    }

    #[test]
    fn test_different_dtypes_separate_batches() {
        let mut batcher = Batcher::with_defaults();

        let fp32_key = BatchKey {
            layer_start: 0,
            activation_dtype: DtypeTag::Fp32,
            is_prefill: false,
            draft_block: false,
        };
        let int8_key = BatchKey {
            layer_start: 0,
            activation_dtype: DtypeTag::Int8,
            is_prefill: false,
            draft_block: false,
        };

        batcher.add(fp32_key.clone(), make_item("fp32-1"));
        batcher.add(int8_key.clone(), make_item("int8-1"));
        batcher.add(fp32_key.clone(), make_item("fp32-2"));
        batcher.add(int8_key.clone(), make_item("int8-2"));

        assert_eq!(batcher.pending_keys(), 2);
        assert_eq!(batcher.pending_items(), 4);

        // Flush all — should produce 2 separate batches.
        let batches = batcher.flush_all();
        assert_eq!(batches.len(), 2);

        let fp32_batch = batches.iter().find(|b| b.key == fp32_key).unwrap();
        let int8_batch = batches.iter().find(|b| b.key == int8_key).unwrap();
        assert_eq!(fp32_batch.items.len(), 2);
        assert_eq!(int8_batch.items.len(), 2);
    }

    #[test]
    fn test_different_layer_ranges_separate_batches() {
        let mut batcher = Batcher::with_defaults();

        let shard_a = fp32_key(0);   // layers [0, 16)
        let shard_b = fp32_key(16);  // layers [16, 32)

        batcher.add(shard_a.clone(), make_item("a-1"));
        batcher.add(shard_b.clone(), make_item("b-1"));
        batcher.add(shard_a.clone(), make_item("a-2"));

        assert_eq!(batcher.pending_keys(), 2);

        let batches = batcher.flush_all();
        assert_eq!(batches.len(), 2);

        let a_batch = batches.iter().find(|b| b.key == shard_a).unwrap();
        let b_batch = batches.iter().find(|b| b.key == shard_b).unwrap();
        assert_eq!(a_batch.items.len(), 2);
        assert_eq!(b_batch.items.len(), 1);
    }

    #[test]
    fn test_prefill_vs_decode_separate() {
        let mut batcher = Batcher::with_defaults();

        let prefill_key = BatchKey {
            layer_start: 0,
            activation_dtype: DtypeTag::Fp32,
            is_prefill: true,
            draft_block: false,
        };
        let decode_key = BatchKey {
            layer_start: 0,
            activation_dtype: DtypeTag::Fp32,
            is_prefill: false,
            draft_block: false,
        };

        batcher.add(prefill_key.clone(), make_item("prefill-1"));
        batcher.add(decode_key.clone(), make_item("decode-1"));

        assert_eq!(batcher.pending_keys(), 2);

        let batches = batcher.flush_all();
        assert_eq!(batches.len(), 2);

        let pf = batches.iter().find(|b| b.key == prefill_key).unwrap();
        let dc = batches.iter().find(|b| b.key == decode_key).unwrap();
        assert_eq!(pf.items.len(), 1);
        assert_eq!(dc.items.len(), 1);
    }

    #[test]
    fn test_draft_block_separate() {
        let mut batcher = Batcher::with_defaults();

        let normal_key = BatchKey {
            layer_start: 0,
            activation_dtype: DtypeTag::Fp32,
            is_prefill: false,
            draft_block: false,
        };
        let draft_key = BatchKey {
            layer_start: 0,
            activation_dtype: DtypeTag::Fp32,
            is_prefill: false,
            draft_block: true,
        };

        batcher.add(normal_key.clone(), make_item("normal-1"));
        batcher.add(draft_key.clone(), make_item("draft-1"));

        assert_eq!(batcher.pending_keys(), 2);

        let batches = batcher.flush_all();
        assert_eq!(batches.len(), 2);
    }

    // ── Time-bound flush ─────────────────────────────────────────────

    #[test]
    fn test_time_bound_flush() {
        let mut batcher = Batcher::new(BatcherConfig {
            max_batch_size: 8,
            max_wait: std::time::Duration::from_millis(1),
        });

        let key = fp32_key(0);

        // Add items with timestamps in the past.
        let mut item = make_item("old-1");
        item.enqueued_at = std::time::Instant::now() - std::time::Duration::from_millis(10);
        batcher.add(key.clone(), item);

        let mut item2 = make_item("old-2");
        item2.enqueued_at = std::time::Instant::now() - std::time::Duration::from_millis(5);
        batcher.add(key.clone(), item2);

        // flush_expired should catch them.
        let flushed = batcher.flush_expired();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].items.len(), 2);
        assert_eq!(flushed[0].reason, FlushReason::TimeBound);

        assert_eq!(batcher.pending_items(), 0);
    }

    #[test]
    fn test_time_bound_no_premature_flush() {
        let mut batcher = Batcher::new(BatcherConfig {
            max_batch_size: 8,
            max_wait: std::time::Duration::from_secs(60),
        });

        let key = fp32_key(0);
        batcher.add(key.clone(), make_item("fresh-1"));

        // flush_expired should NOT fire — item is fresh.
        let flushed = batcher.flush_expired();
        assert!(flushed.is_empty());
        assert_eq!(batcher.pending_items(), 1);
    }

    // ── Size-bound flush ─────────────────────────────────────────────

    #[test]
    fn test_size_bound_exact() {
        let mut batcher = Batcher::new(BatcherConfig {
            max_batch_size: 2,
            max_wait: std::time::Duration::from_secs(60),
        });

        let key = fp32_key(0);
        assert!(batcher.add(key.clone(), make_item("r1")).is_none());
        let batch = batcher.add(key.clone(), make_item("r2"));
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().items.len(), 2);
    }

    #[test]
    fn test_size_bound_only_flushes_full_key() {
        let mut batcher = Batcher::new(BatcherConfig {
            max_batch_size: 2,
            max_wait: std::time::Duration::from_secs(60),
        });

        let key_a = fp32_key(0);
        let key_b = fp32_key(16);

        batcher.add(key_a.clone(), make_item("a-1"));
        batcher.add(key_b.clone(), make_item("b-1"));

        // key_a reaches 2 — flush it, key_b stays.
        let batch = batcher.add(key_a.clone(), make_item("a-2"));
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().key, key_a);

        assert_eq!(batcher.pending_keys(), 1);
        assert_eq!(batcher.pending_items(), 1); // only key_b remains
    }

    // ── Flush all ────────────────────────────────────────────────────

    #[test]
    fn test_flush_all_empties_everything() {
        let mut batcher = Batcher::with_defaults();

        batcher.add(fp32_key(0), make_item("r1"));
        batcher.add(fp32_key(16), make_item("r2"));
        batcher.add(fp32_key(0), make_item("r3"));

        assert_eq!(batcher.pending_items(), 3);

        let batches = batcher.flush_all();
        assert_eq!(batches.len(), 2);
        assert_eq!(batcher.pending_items(), 0);
        assert!(!batcher.has_pending());

        for batch in &batches {
            assert_eq!(batch.reason, FlushReason::Manual);
        }
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_empty_batcher() {
        let mut batcher = Batcher::with_defaults();

        assert_eq!(batcher.pending_keys(), 0);
        assert_eq!(batcher.pending_items(), 0);
        assert!(!batcher.has_pending());

        let flushed = batcher.flush_expired();
        assert!(flushed.is_empty());

        let flushed = batcher.flush_all();
        assert!(flushed.is_empty());
    }

    #[test]
    fn test_batch_key_all_four_dimensions() {
        // All four dimensions must differ for separate batches.
        let keys = vec![
            BatchKey { layer_start: 0, activation_dtype: DtypeTag::Fp32, is_prefill: false, draft_block: false },
            BatchKey { layer_start: 8, activation_dtype: DtypeTag::Fp32, is_prefill: false, draft_block: false },
            BatchKey { layer_start: 0, activation_dtype: DtypeTag::Fp16, is_prefill: false, draft_block: false },
            BatchKey { layer_start: 0, activation_dtype: DtypeTag::Fp32, is_prefill: true,  draft_block: false },
            BatchKey { layer_start: 0, activation_dtype: DtypeTag::Fp32, is_prefill: false, draft_block: true  },
        ];

        let mut batcher = Batcher::with_defaults();
        for (i, key) in keys.iter().enumerate() {
            batcher.add(key.clone(), make_item(&format!("r{i}")));
        }

        assert_eq!(batcher.pending_keys(), 5, "each dimension variant must produce a separate batch key");
        assert_eq!(batcher.pending_items(), 5);
    }

    #[test]
    fn test_dtype_tag_from_activation_dtype() {
        assert_eq!(DtypeTag::from(ActivationDtype::Fp32), DtypeTag::Fp32);
        assert_eq!(DtypeTag::from(ActivationDtype::Fp16), DtypeTag::Fp16);
        assert_eq!(DtypeTag::from(ActivationDtype::Int8), DtypeTag::Int8);

        assert_eq!(ActivationDtype::from(DtypeTag::Fp32), ActivationDtype::Fp32);
        assert_eq!(ActivationDtype::from(DtypeTag::Fp16), ActivationDtype::Fp16);
        assert_eq!(ActivationDtype::from(DtypeTag::Int8), ActivationDtype::Int8);
    }

    // ── Realistic multi-session scenario ─────────────────────────────

    #[test]
    fn test_4_peer_ring_batching_scenario() {
        // Simulates 4-peer ring: sessions arrive at different times
        // with the same shard/dtype but different request_ids.
        let mut batcher = Batcher::new(BatcherConfig {
            max_batch_size: 4,
            max_wait: std::time::Duration::from_millis(5),
        });

        let key = BatchKey {
            layer_start: 0,
            activation_dtype: DtypeTag::Fp32,
            is_prefill: false,
            draft_block: false,
        };

        // Session A and B arrive at stage 0 — compatible, should batch.
        batcher.add(key.clone(), BatchItem {
            request_id: "sess-A-t1".into(),
            session_id: "sess-A".into(),
            activation: vec![0u8; 128],
            activation_shape: vec![1, 1, 32],
            enqueued_at: std::time::Instant::now(),
        });
        batcher.add(key.clone(), BatchItem {
            request_id: "sess-B-t1".into(),
            session_id: "sess-B".into(),
            activation: vec![1u8; 128],
            activation_shape: vec![1, 1, 32],
            enqueued_at: std::time::Instant::now(),
        });

        assert_eq!(batcher.pending_items(), 2);

        // Flush manually (time bound hasn't fired yet).
        let batches = batcher.flush_all();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].items.len(), 2);

        // Verify both sessions are in the same batch.
        let ids: Vec<&str> = batches[0].items.iter()
            .map(|i| i.session_id.as_str())
            .collect();
        assert!(ids.contains(&"sess-A"));
        assert!(ids.contains(&"sess-B"));
    }
}
