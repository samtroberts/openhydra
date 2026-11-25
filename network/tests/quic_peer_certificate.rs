//! The patched libp2p-quic (network/patched-deps/libp2p-quic) after a handshake: the peer's
//! certificate is parsed again to get its peer id, against the time now. Upstream `expect`ed every
//! step, so a certificate that expired during the handshake, or any other parse failure, panicked
//! the node (RUSTSEC-2026-0525). Each failure is now an error that fails only that connection.
//!
//! Run:  cargo test -p openhydra-network --test quic_peer_certificate

use libp2p::quic::openhydra_peer_id_from_certificate_ders as peer_id_of;

#[test]
fn an_unusable_peer_certificate_is_an_error_not_a_panic() {
    // No certificate at all.
    assert!(peer_id_of(&[]).is_err());
    // A certificate that doesn't parse (stands for one that no longer does: expired since the
    // handshake, which the same `parse` call refuses).
    assert!(peer_id_of(&[b"not a certificate".as_slice()]).is_err());
    assert!(peer_id_of(&[&[0x30, 0x82, 0x01, 0x00][..]]).is_err());
}

#[test]
fn a_valid_libp2p_certificate_still_gives_its_peer_id() {
    let key = libp2p::identity::Keypair::generate_ed25519();
    let (cert, _) = libp2p_tls::certificate::generate(&key).expect("a libp2p TLS certificate");
    let got = peer_id_of(&[cert.as_ref()]).expect("its peer id");
    assert_eq!(got, key.public().to_peer_id());
}
