//! Transport stack builder — TCP + QUIC + Noise + Yamux.

use libp2p::{identity, noise, tcp, yamux, Transport};

/// Build the TCP transport: TCP + Noise + Yamux.
pub fn build_tcp_transport(
    keypair: &identity::Keypair,
) -> std::io::Result<libp2p::core::transport::Boxed<(libp2p::PeerId, libp2p::core::muxing::StreamMuxerBox)>>
{
    let tcp_transport = tcp::tokio::Transport::new(tcp::Config::default().nodelay(true))
        .upgrade(libp2p::core::upgrade::Version::V1Lazy)
        .authenticate(noise::Config::new(keypair).expect("noise config"))
        .multiplex(yamux::Config::default())
        .boxed();

    Ok(tcp_transport)
}

/// Build the QUIC transport (UDP-based, built-in TLS 1.3 + multiplexing).
///
/// QUIC provides 0-RTT connections and is better for NAT hole punching (UDP).
/// No Noise/Yamux wrapping needed — QUIC handles encryption and muxing natively.
pub fn build_quic_transport(
    keypair: &identity::Keypair,
) -> std::io::Result<libp2p::core::transport::Boxed<(libp2p::PeerId, libp2p::core::muxing::StreamMuxerBox)>>
{
    let quic_transport = libp2p::quic::tokio::Transport::new(
        libp2p::quic::Config::new(keypair),
    )
    .map(|(peer_id, muxer), _| (peer_id, libp2p::core::muxing::StreamMuxerBox::new(muxer)))
    .boxed();

    Ok(quic_transport)
}

/// Build the legacy TCP-only transport (backward compat alias).
pub fn build_transport(
    keypair: &identity::Keypair,
) -> std::io::Result<libp2p::core::transport::Boxed<(libp2p::PeerId, libp2p::core::muxing::StreamMuxerBox)>>
{
    build_tcp_transport(keypair)
}
