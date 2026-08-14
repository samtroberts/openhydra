//! Adversarial harness — exercises the network plane the way a malicious,
//! untrusted peer on an open network would, and asserts the hardening holds.
//!
//! The pre-pivot sharding decoders (forward_msg / ipc_codec / dispatcher / ring) that this
//! file used to fuzz have been removed; the live untrusted-input surface is now the agent's
//! serve decode (`ServeRequest::decode` / `parse_response` / `FetchChunksResponse::decode`),
//! which is fuzzed in the `openhydra-agent` crate. What remains network-side is the inbound
//! proxy queue, which must stay bounded under a flood.
//!
//! Run:  cargo test --test adversarial

use openhydra_network::event_loop::{SharedProxyQueue, PROXY_QUEUE_MAX};

// ── inbound queue stays bounded under a flood ────────────────────────────────

#[test]
fn shared_proxy_queue_bounded_under_flood() {
    let q = SharedProxyQueue::new();
    for i in 0..(PROXY_QUEUE_MAX * 3) {
        q.push((format!("flood-{i}"), "12D3KooWflood".to_string(), vec![0u8; 16]));
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
