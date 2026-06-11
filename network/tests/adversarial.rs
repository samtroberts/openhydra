//! Adversarial harness — exercises the network plane the way a malicious,
//! untrusted peer on an open network would, and asserts the Phase 1/2
//! hardening holds: no panics on malformed bytes, no forged ring callbacks,
//! no unbounded growth, and graceful rejection of oversized/non-finite input.
//!
//! Run:  cargo test --no-default-features --test adversarial

use openhydra_network::activation::ActivationBuffer;
use openhydra_network::dispatcher;
use openhydra_network::event_loop::{SharedProxyQueue, PROXY_QUEUE_MAX};
use openhydra_network::forward_msg;
use openhydra_network::ipc_codec::{
    self, BATCH_MAGIC, IpcResponseHeader, IpcStatus, ActivationDtype,
};
use openhydra_network::ring::{RingConfig, RingHop, RingManager, RingAction};

fn route(peers: &[&str]) -> Vec<RingHop> {
    peers
        .iter()
        .enumerate()
        .map(|(i, p)| RingHop {
            peer_id: (*p).to_string(),
            layer_start: (i * 16) as u32,
            layer_end: ((i + 1) * 16) as u32,
            total_layers: (peers.len() * 16) as u32,
        })
        .collect()
}

fn config(session_id: &str, peers: &[&str]) -> RingConfig {
    RingConfig {
        session_id: session_id.to_string(),
        request_id: "req-0".to_string(),
        max_tokens: 1_000_000, // large so the session never auto-completes
        slot_id: 0,
        route: route(peers),
        eos_ids: vec![],
        hop_timeout_ms: 500,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        seed: None,
    }
}

fn ok_header(request_id: &str) -> IpcResponseHeader {
    IpcResponseHeader {
        status: IpcStatus::Ok,
        request_id: request_id.to_string(),
        activation_dtype: ActivationDtype::Fp32,
        activation_shape: vec![1, 1, 4],
        ..Default::default()
    }
}

// ── F1: ring callback authentication ────────────────────────────────────────

#[test]
fn ring_rejects_forged_pushresult_but_accepts_genuine() {
    let mut mgr = RingManager::new();
    let _h = mgr.start_session(config("ring-1", &["peer-A", "peer-B"]));
    mgr.register_request("r0".into(), "ring-1".into()); // expected hop = peer-B

    // A flood of forged callbacks from the wrong peer must all be rejected
    // AND must leave the pending mapping intact for the genuine peer.
    for forger in ["peer-A", "evil", "peer-B-typo", ""] {
        let a = mgr.route_push_result("r0", forger, &ok_header("r0"), vec![0u8; 16]);
        assert!(
            matches!(a, RingAction::NotRingRequest),
            "forged callback from {forger:?} must be rejected"
        );
        assert!(mgr.is_ring_request("r0"), "mapping must survive forgery");
    }

    // The genuine final hop is accepted exactly once.
    let a = mgr.route_push_result("r0", "peer-B", &ok_header("r0"), vec![0u8; 16]);
    assert!(matches!(a, RingAction::NeedSample { .. }));
    assert!(!mgr.is_ring_request("r0"), "mapping consumed on success");
}

// ── F1 / leak: long run does not accumulate pending requests ─────────────────

#[test]
fn long_ring_run_does_not_leak_pending_requests() {
    let mut mgr = RingManager::new();
    let _h = mgr.start_session(config("ring-long", &["peer-A", "peer-B"]));

    for i in 0..20_000usize {
        let rid = format!("tok-{i}");
        mgr.register_request(rid.clone(), "ring-long".into());
        let a = mgr.route_push_result(&rid, "peer-B", &ok_header(&rid), vec![0u8; 16]);
        assert!(matches!(a, RingAction::NeedSample { .. }));
        // After each consumed callback nothing should linger.
        assert_eq!(mgr.pending_request_count(), 0, "pending leaked at token {i}");
    }
}

// ── Decoder panic-resistance: never panic on attacker bytes ──────────────────

fn malformed_corpus() -> Vec<Vec<u8>> {
    let mut corpus: Vec<Vec<u8>> = vec![
        vec![],
        vec![0x00],
        vec![0xFF; 3],
        vec![0xFF; 4],
        vec![0xFF; 7],
        vec![0xFF; 8],
        vec![0x00; 64],
        vec![0xFF; 1024],
    ];
    // Batch magic + absurd counts, with and without trailing bytes.
    for count in [0u32, 1, 2, u32::MAX, u32::MAX / 2] {
        let mut b = BATCH_MAGIC.to_le_bytes().to_vec();
        b.extend_from_slice(&count.to_le_bytes());
        corpus.push(b.clone());
        b.extend_from_slice(&[0xAA; 5]); // ragged trailing
        corpus.push(b);
    }
    // Plausible-looking length prefixes that overrun the buffer.
    for len in [u32::MAX, 0x7FFF_FFFF, 100 * 1024 * 1024 + 1] {
        let mut b = len.to_le_bytes().to_vec();
        b.extend_from_slice(b"short");
        corpus.push(b);
    }
    // Big-endian framed length (tensor-stream style) that overruns.
    for len in [u32::MAX, 1 << 30] {
        let mut b = len.to_be_bytes().to_vec();
        b.extend_from_slice(b"x");
        corpus.push(b);
    }
    // Every truncation of a 32-byte all-0xCD blob.
    for n in 0..=32 {
        corpus.push(vec![0xCD; n]);
    }
    corpus
}

#[test]
fn decoders_never_panic_on_malformed_input() {
    for (i, buf) in malformed_corpus().iter().enumerate() {
        // Every decoder must return Ok or Err — never panic / never OOM.
        // (A panic here aborts the test with the offending index visible.)
        let _ = forward_msg::decode(buf);
        let _ = forward_msg::decode_response(buf);
        let _ = ipc_codec::decode_forward_request(buf);
        let _ = ipc_codec::decode_response(buf);
        let _ = ipc_codec::decode_batch_request(buf);
        let _ = ipc_codec::decode_batch_response(buf);
        let _ = ActivationBuffer::from_packed(buf.clone());
        let _ = dispatcher::extract_method(buf);
        eprintln!("corpus[{i}] survived ({} bytes)", buf.len());
    }
}

// ── 2.2: batch_count overflow is bounded, not an allocation bomb ─────────────

#[test]
fn batch_count_overflow_is_rejected_fast() {
    let mut data = BATCH_MAGIC.to_le_bytes().to_vec();
    data.extend_from_slice(&u32::MAX.to_le_bytes()); // claim ~4.3B items
    assert!(ipc_codec::decode_batch_request(&data).is_err());
    assert!(ipc_codec::decode_batch_response(&data).is_err());
}

// ── M1: activation header overflow / non-finite dims rejected ────────────────

#[test]
fn activation_overflow_and_nonfinite_rejected() {
    // Overflow: 2^40 * 2^40 * 4 wraps usize in release builds.
    let big = (1u64 << 40) as f32;
    let mut packed = big.to_le_bytes().to_vec();
    packed.extend_from_slice(&big.to_le_bytes());
    packed.extend_from_slice(&[0u8; 8]);
    assert!(ActivationBuffer::from_packed(packed).is_err());

    // NaN / negative dims.
    let mut packed = f32::NAN.to_le_bytes().to_vec();
    packed.extend_from_slice(&(-5.0f32).to_le_bytes());
    packed.extend_from_slice(&[0u8; 8]);
    assert!(ActivationBuffer::from_packed(packed).is_err());
}

// ── 2.4: inbound queue stays bounded under a flood ───────────────────────────

#[test]
fn shared_proxy_queue_bounded_under_flood() {
    let q = SharedProxyQueue::new();
    for i in 0..(PROXY_QUEUE_MAX * 3) {
        q.push((format!("flood-{i}"), vec![0u8; 16]));
    }
    // Drain and count — must never exceed the cap.
    let mut drained = 0usize;
    while q.pop(std::time::Duration::from_millis(0)).is_some() {
        drained += 1;
        if drained > PROXY_QUEUE_MAX {
            panic!("queue exceeded PROXY_QUEUE_MAX under flood");
        }
    }
    assert_eq!(drained, PROXY_QUEUE_MAX);
}
