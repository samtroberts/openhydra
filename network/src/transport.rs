//! Transport stack builder — TCP + QUIC + Noise + Yamux.

use std::sync::Arc;
use std::time::Duration;

use libp2p::{identity, noise, tcp, yamux, Transport};

/// The most an outgoing connection may take to set up: the dial, and for a relayed dial the circuit,
/// then the security and muxer handshakes. Generous, since a circuit over CGNAT is slow to build; the
/// same as the 30 s a queued forward waits.
pub const OUTBOUND_SETUP_TIMEOUT: Duration = Duration::from_secs(30);
/// The most an incoming connection may take to set up (security and muxer handshakes). Without a
/// limit, a socket that opens and then sends nothing holds a pending-connection slot for as long as it
/// stays open, and a few hundred of them would fill the pending limit and turn every new connection
/// away (adversarial review 2026-10-14, H2). libp2p's own `SwarmBuilder` sets 10 s for both.
pub const INBOUND_SETUP_TIMEOUT: Duration = Duration::from_secs(15);

type BoxedTransport = libp2p::core::transport::Boxed<(libp2p::PeerId, libp2p::core::muxing::StreamMuxerBox)>;

/// `transport` with time limits on setting up a connection, outgoing and incoming. Every swarm's
/// transport goes through this (it doesn't use libp2p's `SwarmBuilder`, which adds them itself).
pub fn with_setup_timeouts(transport: BoxedTransport, outbound: Duration, inbound: Duration) -> BoxedTransport {
    use libp2p::core::transport::timeout::TransportTimeout;
    TransportTimeout::with_outgoing_timeout(TransportTimeout::with_ingoing_timeout(transport, inbound), outbound).boxed()
}

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
    let quic_config = libp2p::quic::Config::new(keypair)
        .initial_mtu(1452)
        .congestion_controller_factory(Arc::new(
            quinn::congestion::BbrConfig::default(),
        ));
    let quic_transport = libp2p::quic::tokio::Transport::new(quic_config)
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

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use libp2p::multiaddr::Protocol;
    use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
    use libp2p::swarm::{dummy, NetworkBehaviour, Swarm, SwarmEvent};
    use libp2p::{Multiaddr, PeerId};

    #[derive(NetworkBehaviour)]
    #[behaviour(prelude = "libp2p::swarm::derive_prelude")]
    struct Limited {
        limits: libp2p::connection_limits::Behaviour,
        keep: dummy::Behaviour,
    }

    /// A small pending-incoming limit, so a handful of silent sockets fill it.
    const PENDING: u32 = 4;

    /// A TCP node with the production transport, a pending-incoming limit of [`PENDING`], and setup
    /// timeouts of `timeout` (none: as before H2).
    fn node(timeout: Option<Duration>) -> Swarm<Limited> {
        let key = identity::Keypair::generate_ed25519();
        let mut transport = build_tcp_transport(&key).unwrap();
        if let Some(t) = timeout {
            transport = with_setup_timeouts(transport, t, t);
        }
        let limits = libp2p::connection_limits::ConnectionLimits::default().with_max_pending_incoming(Some(PENDING));
        let behaviour = Limited { limits: libp2p::connection_limits::Behaviour::new(limits), keep: dummy::Behaviour };
        let cfg = libp2p::swarm::Config::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(60));
        Swarm::new(transport, behaviour, PeerId::from(key.public()), cfg)
    }

    async fn listen(a: &mut Swarm<Limited>) -> Multiaddr {
        a.listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap()).unwrap();
        loop {
            if let SwarmEvent::NewListenAddr { address, .. } = a.select_next_some().await {
                return address;
            }
        }
    }

    /// Run `a` for `d`.
    async fn drive(a: &mut Swarm<Limited>, d: Duration) {
        let t = tokio::time::sleep(d);
        tokio::pin!(t);
        loop {
            tokio::select! {
                _ = a.select_next_some() => {}
                _ = &mut t => return,
            }
        }
    }

    /// Whether a fresh node's dial to `a` at `addr` connects.
    async fn connects(a: &mut Swarm<Limited>, addr: &Multiaddr) -> bool {
        let mut b = node(None);
        let opts = DialOpts::peer_id(*a.local_peer_id()).addresses(vec![addr.clone()]).condition(PeerCondition::Always).build();
        b.dial(opts).unwrap();
        let deadline = tokio::time::sleep(Duration::from_secs(5));
        tokio::pin!(deadline);
        loop {
            tokio::select! {
                ev = b.select_next_some() => match ev {
                    SwarmEvent::ConnectionEstablished { .. } => return true,
                    SwarmEvent::OutgoingConnectionError { .. } => return false,
                    _ => {}
                },
                _ = a.select_next_some() => {}
                _ = &mut deadline => return false,
            }
        }
    }

    /// [`PENDING`] sockets to `addr` that connect and then send nothing (kept open by the caller).
    async fn silent_sockets(addr: &Multiaddr) -> Vec<tokio::net::TcpStream> {
        let port = addr.iter().find_map(|p| if let Protocol::Tcp(p) = p { Some(p) } else { None }).unwrap();
        let mut out = Vec::new();
        for _ in 0..PENDING {
            out.push(tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap());
        }
        out
    }

    #[tokio::test]
    async fn silent_sockets_free_their_slots_once_setup_times_out() {
        let mut a = node(Some(Duration::from_secs(1)));
        let addr = listen(&mut a).await;
        let _held = silent_sockets(&addr).await;
        drive(&mut a, Duration::from_millis(200)).await;
        assert!(!connects(&mut a, &addr).await, "the silent sockets should fill the pending limit");
        drive(&mut a, Duration::from_millis(1500)).await;
        assert!(connects(&mut a, &addr).await, "a real peer connects once the silent sockets time out");
    }

    #[tokio::test]
    async fn without_a_setup_timeout_silent_sockets_hold_their_slots() {
        // The behaviour before H2, kept as the control: the sockets never let go.
        let mut a = node(None);
        let addr = listen(&mut a).await;
        let _held = silent_sockets(&addr).await;
        drive(&mut a, Duration::from_millis(1700)).await;
        assert!(!connects(&mut a, &addr).await);
    }
}
