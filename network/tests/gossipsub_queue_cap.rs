//! The patched libp2p-gossipsub (network/patched-deps/libp2p-gossipsub) holds at most its cap per
//! connection for a peer that never reads (2026-10-29: unbounded upstream, a bootstrap went past
//! 1 GB in seconds). Its own tests can't run from the vendored copy, so this drives the patch here.
//!
//! Run:  cargo test -p openhydra-network --test gossipsub_queue_cap

#[test]
fn a_peer_that_never_reads_costs_at_most_the_queue_cap() {
    // 10,000 forwards of 60 KB: upstream would hold ~600 MB for this one connection.
    let (bytes, queued, dropped, cap) = libp2p::gossipsub::openhydra_queue_selftest(10_000, 60 * 1024);
    assert!(bytes <= cap, "{bytes} bytes queued, cap {cap}");
    assert!(queued <= 4 && dropped as usize >= 10_000 - 4, "queued {queued}, dropped {dropped}");
    // Small messages fit until the cap, and none are dropped below it.
    let (_, queued, dropped, _) = libp2p::gossipsub::openhydra_queue_selftest(100, 100);
    assert_eq!((queued, dropped), (100, 0));
}

#[test]
fn one_peer_can_send_only_so_many_messages_a_second() {
    use libp2p::gossipsub::{openhydra_rate_allow, OPENHYDRA_MAX_MESSAGES_PER_SEC, OPENHYDRA_MESSAGE_BURST};
    let mut buckets = std::collections::HashMap::new();
    let p = libp2p::PeerId::random();
    let t0 = web_time::Instant::now();
    let sent = (0..30_000).filter(|_| openhydra_rate_allow(&mut buckets, p, t0)).count();
    assert_eq!(sent as f64, OPENHYDRA_MESSAGE_BURST, "a burst, then nothing at once");
    let later = t0 + std::time::Duration::from_secs(1);
    let more = (0..30_000).filter(|_| openhydra_rate_allow(&mut buckets, p, later)).count();
    assert_eq!(more as f64, OPENHYDRA_MAX_MESSAGES_PER_SEC, "then the rate");
    // Another peer has its own.
    assert!(openhydra_rate_allow(&mut buckets, libp2p::PeerId::random(), later));
}

#[test]
fn one_stream_has_only_so_many_messages_checked_at_once() {
    use libp2p::gossipsub::{openhydra_codec_selftest, OPENHYDRA_CHECKED_BURST};
    // 100 frames of 100 messages, all at once: the burst gets through, the rest are dropped
    // before their check.
    let got = openhydra_codec_selftest(100, 100);
    assert!(got as f64 >= OPENHYDRA_CHECKED_BURST && (got as f64) < OPENHYDRA_CHECKED_BURST + 100.0, "{got}");
    // An honest stream (a few messages) is untouched.
    assert_eq!(openhydra_codec_selftest(5, 3), 15);
}
