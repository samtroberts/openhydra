//! Transport stack builder — TCP + QUIC + Noise + Yamux.

use libp2p::{identity, noise, tcp, yamux, Transport};

/// Build the default transport: TCP + Noise + Yamux.
///
/// QUIC support is added separately via `libp2p::quic` when available.
/// For Phase A we start with TCP only — QUIC is added in Phase B.
pub fn build_transport(
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
