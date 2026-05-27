//! CP-2: Ring session tracking — minimal stub for dispatcher integration.
//!
//! Full implementation comes in CP-3 (Rust Ring Loop). This stub provides
//! the `RingSessionMap` type that the dispatcher references for PushResult
//! routing, plus the `RingSession` struct that tracks per-session state.
//!
//! In CP-2, PushResult messages are still routed to Python (via IPC bridge
//! or SharedProxyQueue fallback). The session map here is a bookkeeping
//! layer that the dispatcher can optionally consult for fast-path decisions
//! (e.g., detecting EOS without a Python round-trip).

use std::collections::HashMap;

/// Unique session identifier (matches `kv_session_id` in IpcForwardHeader).
pub type SessionId = String;

/// Per-ring-session state.
#[derive(Debug, Clone)]
pub struct RingSession {
    /// Session ID (= `kv_session_id`).
    pub session_id: SessionId,
    /// Pipeline slot index (for pipeline_depth > 1).
    pub slot_id: u32,
    /// Total tokens remaining to generate.
    pub tokens_remaining: u32,
    /// Generated token IDs accumulated so far.
    pub generated_ids: Vec<u32>,
    /// Request ID of the originating ForwardRequest.
    pub request_id: String,
    /// Whether EOS has been detected.
    pub eos_detected: bool,
}

impl RingSession {
    /// Create a new ring session.
    pub fn new(session_id: SessionId, request_id: String, max_tokens: u32, slot_id: u32) -> Self {
        Self {
            session_id,
            slot_id,
            tokens_remaining: max_tokens,
            generated_ids: Vec::with_capacity(max_tokens as usize),
            request_id,
            eos_detected: false,
        }
    }

    /// Record a generated token, decrement remaining count.
    ///
    /// Returns `true` if the session is now complete (EOS or tokens exhausted).
    pub fn record_token(&mut self, token_id: u32, is_eos: bool) -> bool {
        self.generated_ids.push(token_id);
        if self.tokens_remaining > 0 {
            self.tokens_remaining -= 1;
        }
        if is_eos {
            self.eos_detected = true;
        }
        self.is_complete()
    }

    /// Whether this session is complete.
    pub fn is_complete(&self) -> bool {
        self.eos_detected || self.tokens_remaining == 0
    }

    /// Number of tokens generated so far.
    pub fn tokens_generated(&self) -> usize {
        self.generated_ids.len()
    }
}

/// Map of active ring sessions, keyed by session_id.
///
/// Thread-safety: accessed only from the tokio event loop (single-threaded).
/// If needed from multiple threads (CP-3), wrap in `Arc<Mutex<_>>`.
#[derive(Debug, Default)]
pub struct RingSessionMap {
    sessions: HashMap<SessionId, RingSession>,
}

impl RingSessionMap {
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Register a new ring session.
    pub fn insert(&mut self, session: RingSession) {
        self.sessions.insert(session.session_id.clone(), session);
    }

    /// Look up a session by ID.
    pub fn get(&self, session_id: &str) -> Option<&RingSession> {
        self.sessions.get(session_id)
    }

    /// Mutable look-up.
    pub fn get_mut(&mut self, session_id: &str) -> Option<&mut RingSession> {
        self.sessions.get_mut(session_id)
    }

    /// Remove a completed session.
    pub fn remove(&mut self, session_id: &str) -> Option<RingSession> {
        self.sessions.remove(session_id)
    }

    /// Number of active sessions.
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }

    /// Iterate over all active sessions.
    pub fn iter(&self) -> impl Iterator<Item = (&SessionId, &RingSession)> {
        self.sessions.iter()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_session_basic() {
        let mut session = RingSession::new(
            "sess-001".into(),
            "req-001".into(),
            5, // max_tokens
            0, // slot_id
        );

        assert!(!session.is_complete());
        assert_eq!(session.tokens_remaining, 5);
        assert_eq!(session.tokens_generated(), 0);

        // Generate 4 tokens — not complete yet.
        for i in 0..4 {
            let done = session.record_token(100 + i, false);
            assert!(!done);
        }
        assert_eq!(session.tokens_remaining, 1);
        assert_eq!(session.tokens_generated(), 4);

        // 5th token — complete.
        let done = session.record_token(104, false);
        assert!(done);
        assert!(session.is_complete());
        assert_eq!(session.tokens_remaining, 0);
    }

    #[test]
    fn test_ring_session_eos() {
        let mut session = RingSession::new(
            "sess-eos".into(),
            "req-eos".into(),
            100,
            0,
        );

        // EOS on second token — completes early.
        session.record_token(42, false);
        let done = session.record_token(2, true); // EOS token
        assert!(done);
        assert!(session.is_complete());
        assert!(session.eos_detected);
        assert_eq!(session.tokens_remaining, 98);
        assert_eq!(session.generated_ids, vec![42, 2]);
    }

    #[test]
    fn test_ring_session_map() {
        let mut map = RingSessionMap::new();
        assert!(map.is_empty());

        let s1 = RingSession::new("s1".into(), "r1".into(), 10, 0);
        let s2 = RingSession::new("s2".into(), "r2".into(), 20, 1);
        map.insert(s1);
        map.insert(s2);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get("s1").unwrap().tokens_remaining, 10);
        assert_eq!(map.get("s2").unwrap().slot_id, 1);
        assert!(map.get("s3").is_none());

        // Mutate.
        map.get_mut("s1").unwrap().record_token(99, false);
        assert_eq!(map.get("s1").unwrap().tokens_remaining, 9);

        // Remove.
        let removed = map.remove("s1").unwrap();
        assert_eq!(removed.tokens_generated(), 1);
        assert_eq!(map.len(), 1);
    }
}
