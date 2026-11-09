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

pub type BoxedTransport = libp2p::core::transport::Boxed<(libp2p::PeerId, libp2p::core::muxing::StreamMuxerBox)>;

/// The most a TCP connect may take (the SYN answered), before any handshake. Its own limit, inside
/// [`OUTBOUND_SETUP_TIMEOUT`], so "nobody answered" is told apart from "answered, then stalled in the
/// handshake" (a peer at its connection limit): see [`dial_means_unreachable`] (review 2026-10-29).
/// Without it the OS's own connect timeout (over a minute) applied, inside the 30 s overall limit.
/// 8 s lets a lost SYN be resent three times (at about 1, 3 and 7 s): with 5 s, two losses in a row
/// on a lossy mobile link read as "nobody there" (review 2026-10-29).
pub const TCP_CONNECT_TIMEOUT: Duration = Duration::from_secs(8);

/// `transport`, refusing to dial anything but a public address (a global IP, or a DNS name that
/// isn't local). For OpenHydra's own bootstraps: their AutoNAT server dials whatever address a
/// client asks it to test, loopback and the cloud's metadata address included (review 2026-10-29),
/// and nothing they should ever dial is private. Listening is untouched.
pub fn public_dials_only(transport: BoxedTransport) -> BoxedTransport {
    PublicDialsOnly(transport).boxed()
}

struct PublicDialsOnly<T>(T);

impl<T: libp2p::core::Transport + Unpin> libp2p::core::Transport for PublicDialsOnly<T> {
    type Output = T::Output;
    type Error = T::Error;
    type ListenerUpgrade = T::ListenerUpgrade;
    type Dial = T::Dial;

    fn listen_on(
        &mut self,
        id: libp2p::core::transport::ListenerId,
        addr: libp2p::Multiaddr,
    ) -> Result<(), libp2p::core::transport::TransportError<Self::Error>> {
        self.0.listen_on(id, addr)
    }

    fn remove_listener(&mut self, id: libp2p::core::transport::ListenerId) -> bool {
        self.0.remove_listener(id)
    }

    fn dial(
        &mut self,
        addr: libp2p::Multiaddr,
        opts: libp2p::core::transport::DialOpts,
    ) -> Result<Self::Dial, libp2p::core::transport::TransportError<Self::Error>> {
        if !crate::swarm::routable_on_public_network(&addr) {
            return Err(libp2p::core::transport::TransportError::MultiaddrNotSupported(addr));
        }
        self.0.dial(addr, opts)
    }

    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<libp2p::core::transport::TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        std::pin::Pin::new(&mut self.get_mut().0).poll(cx)
    }
}

/// `transport` with time limits on setting up a connection, outgoing and incoming. Every swarm's
/// transport goes through this (it doesn't use libp2p's `SwarmBuilder`, which adds them itself).
pub fn with_setup_timeouts(transport: BoxedTransport, outbound: Duration, inbound: Duration) -> BoxedTransport {
    use libp2p::core::transport::timeout::TransportTimeout;
    TransportTimeout::with_outgoing_timeout(
        TransportTimeout::with_ingoing_timeout(transport, inbound).map_err(unwrap_timeout_error).boxed(),
        outbound,
    )
    .map_err(unwrap_timeout_error)
    .boxed()
}

/// A connection's setup (handshakes included) took longer than its limit: the peer may well have
/// answered, so it's not "no answer" ([`addr_failure`]).
#[derive(Debug)]
pub struct SetupTimedOut;

impl std::fmt::Display for SetupTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("connection setup timed out")
    }
}

impl std::error::Error for SetupTimedOut {}

/// The error inside a timeout wrapper, as is (libp2p's wrapper prints nothing for it, so an error
/// such as "address already in use" came out as an empty message), or [`SetupTimedOut`].
fn unwrap_timeout_error(e: libp2p::core::transport::timeout::TransportTimeoutError<std::io::Error>) -> std::io::Error {
    use libp2p::core::transport::timeout::TransportTimeoutError;
    match e {
        TransportTimeoutError::Timeout => std::io::Error::new(std::io::ErrorKind::Other, SetupTimedOut),
        TransportTimeoutError::TimerError(e) | TransportTimeoutError::Other(e) => e,
    }
}

/// Build the TCP transport: TCP + Noise + Yamux.
pub fn build_tcp_transport(
    keypair: &identity::Keypair,
) -> std::io::Result<libp2p::core::transport::Boxed<(libp2p::PeerId, libp2p::core::muxing::StreamMuxerBox)>>
{
    use libp2p::core::transport::timeout::{TransportTimeout, TransportTimeoutError};
    let tcp = TransportTimeout::with_outgoing_timeout(tcp::tokio::Transport::new(tcp::Config::default().nodelay(true)), TCP_CONNECT_TIMEOUT)
        // The connect timing out is a plain `TimedOut`, like the OS's own: see `dial_means_unreachable`.
        .map_err(|e| match e {
            TransportTimeoutError::Timeout => std::io::Error::new(std::io::ErrorKind::TimedOut, "TCP connect timed out"),
            TransportTimeoutError::TimerError(e) | TransportTimeoutError::Other(e) => e,
        });
    let tcp_transport = tcp
        .upgrade(libp2p::core::upgrade::Version::V1Lazy)
        .authenticate(noise::Config::new(keypair).expect("noise config"))
        .multiplex(yamux_config())
        // Keep the TCP connect's own error as is (its kind says whether anyone answered), and mark a
        // failed security or muxer handshake as such: libp2p's `Either` error wrapper hides what it
        // wraps from `source()`, so a classifier couldn't tell them apart (review 2026-10-29).
        .map_err(|e| e.either(|e| e.either(|tcp| tcp, handshake_failed), handshake_failed))
        .map(|(peer, muxer), _| (peer, libp2p::core::muxing::StreamMuxerBox::new(muxer)))
        .boxed();

    Ok(tcp_transport)
}

/// A security or muxer handshake failed after the connection was made: whoever is at that address
/// answered ([`addr_failure`]).
#[derive(Debug)]
pub struct HandshakeFailed(String);

impl std::fmt::Display for HandshakeFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "handshake failed: {}", self.0)
    }
}

impl std::error::Error for HandshakeFailed {}

fn handshake_failed(e: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::Other, HandshakeFailed(e.to_string()))
}

/// Whether a failed dial failed only because the relays it went through refused for their limits
/// ("resource limit exceeded": a relay's circuit budget for this source is used up, or it's full).
/// Retrying soon just spends more of that budget, so the caller shouldn't (review 2026-10-29: fast
/// retries through four relays used up a consumer's circuit budget at each). A relay saying "no
/// reservation" isn't counted here: that's a provider that hasn't reserved yet, worth a retry.
pub fn dial_refused_by_relay_limits(error: &libp2p::swarm::DialError) -> bool {
    use libp2p::relay::outbound::hop::ConnectError;
    let libp2p::swarm::DialError::Transport(errors) = error else { return false };
    !errors.is_empty()
        && errors.iter().all(|(_, e)| {
            let libp2p::TransportError::Other(io) = e else { return false };
            let mut limit = false;
            walk_errors(io, &mut |e| {
                if matches!(e.downcast_ref::<ConnectError>(), Some(ConnectError::ResourceLimitExceeded)) {
                    limit = true;
                }
            });
            limit
        })
}

/// The addresses of a failed dial that failed only because the local address was in use: with TCP
/// port reuse every dial leaves from our listen port, and right after a restart the previous
/// process's connection to the same peer and address is still in TIME_WAIT, so the OS refuses the
/// same address pair (`EADDRINUSE`, ~30 s on macOS) — a restarted node couldn't reach its bootstraps
/// over TCP for that long (mesh verification 2026-10-30). Such an address works from a new port.
pub fn addrs_refused_as_in_use(error: &libp2p::swarm::DialError) -> Vec<libp2p::Multiaddr> {
    let libp2p::swarm::DialError::Transport(errors) = error else { return Vec::new() };
    errors
        .iter()
        .filter(|(_, e)| {
            let libp2p::TransportError::Other(io) = e else { return false };
            let mut in_use = io.kind() == std::io::ErrorKind::AddrInUse;
            walk_errors(io, &mut |e| {
                if e.downcast_ref::<std::io::Error>().is_some_and(|e| e.kind() == std::io::ErrorKind::AddrInUse) {
                    in_use = true;
                }
            });
            in_use
        })
        .map(|(a, _)| a.clone())
        .collect()
}

/// The yamux settings: libp2p's defaults, unless a load test sets
/// `OPENHYDRA_TEST_YAMUX_WINDOW_KB` (receive window and buffer per stream) or
/// `OPENHYDRA_TEST_YAMUX_STREAMS` (streams per connection) — bootstrap load simulation, 30 Oct.
#[allow(deprecated)]
fn yamux_config() -> yamux::Config {
    let mut cfg = yamux::Config::default();
    if let Some(kb) = std::env::var("OPENHYDRA_TEST_YAMUX_WINDOW_KB").ok().and_then(|v| v.parse::<u32>().ok()) {
        // yamux 0.12 refuses a window under its 256 KiB default (it panics): only the buffer can shrink.
        cfg.set_receive_window_size(kb.max(256) * 1024);
        cfg.set_max_buffer_size(kb as usize * 1024);
    }
    if let Some(n) = std::env::var("OPENHYDRA_TEST_YAMUX_STREAMS").ok().and_then(|v| v.parse::<usize>().ok()) {
        cfg.set_max_num_streams(n);
    }
    cfg
}

/// Two transports in one, trying `a` first: [`libp2p::core::transport::OrTransport`] with its
/// output and error unwrapped, so a dial error is the transport's own (see [`addr_failure`]).
pub fn either_transport(a: BoxedTransport, b: BoxedTransport) -> BoxedTransport {
    libp2p::core::transport::OrTransport::new(a, b)
        .map(|either, _| match either {
            futures::future::Either::Left(o) | futures::future::Either::Right(o) => o,
        })
        .map_err(|e| e.into_inner())
        .boxed()
}

/// How one address of a failed dial failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrFailure {
    /// Nothing answered there: refused, unreachable, or the TCP connect timed out.
    NoAnswer,
    /// A QUIC handshake timed out. QUIC has no separate connect step (libp2p-quic times the whole
    /// handshake as one), so this doesn't say whether the peer sent anything.
    QuicTimeout,
    /// This transport can't dial that address: says nothing about the peer.
    NotSupported,
    /// Anything else, including a handshake that started and failed: the peer (or something at its
    /// address) answered.
    Other,
}

/// `err` with everything it wraps, as one line ("a: b: c"), each message once. libp2p's transport
/// errors print nothing for the error they wrap (they leave it to `source()`), so printing just the
/// outer one gave an empty message (a port already in use came out as "build_swarm: ").
pub fn error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut parts: Vec<String> = Vec::new();
    walk_errors(err, &mut |e| {
        let msg = e.to_string();
        let msg = msg.trim();
        if !msg.is_empty() && !parts.iter().any(|p| p.contains(msg)) {
            parts.push(msg.to_string());
        }
    });
    if parts.is_empty() {
        format!("{err:?}")
    } else {
        parts.join(": ")
    }
}

/// Visit `err` and everything it wraps, including errors inside an `io::Error` (whose own
/// `source()` skips the error it wraps).
pub(crate) fn walk_errors<'a>(err: &'a (dyn std::error::Error + 'static), f: &mut dyn FnMut(&'a (dyn std::error::Error + 'static))) {
    f(err);
    if let Some(io) = err.downcast_ref::<std::io::Error>() {
        if let Some(inner) = io.get_ref() {
            walk_errors(inner, f);
            return;
        }
    }
    if let Some(next) = err.source() {
        walk_errors(next, f);
    }
}

/// Classify one address's dial error ([`AddrFailure`]).
pub fn addr_failure(err: &libp2p::TransportError<std::io::Error>) -> AddrFailure {
    use std::io::ErrorKind;
    let io = match err {
        libp2p::TransportError::MultiaddrNotSupported(_) => return AddrFailure::NotSupported,
        libp2p::TransportError::Other(io) => io,
    };
    let mut kind = AddrFailure::Other;
    let mut upgrade = false;
    walk_errors(io, &mut |e| {
        if let Some(q) = e.downcast_ref::<libp2p::quic::Error>() {
            if matches!(q, libp2p::quic::Error::HandshakeTimedOut) && kind == AddrFailure::Other {
                kind = AddrFailure::QuicTimeout;
            }
        }
        // A security or muxer handshake failed: the peer answered, whatever IO error is inside it.
        if e.downcast_ref::<HandshakeFailed>().is_some() {
            upgrade = true;
        }
        // The OS's refusals, and a TCP connect that timed out (ours, `TCP_CONNECT_TIMEOUT`, is a
        // `TimedOut` too). The overall setup timeout is `SetupTimedOut`, of kind `Other`, so it
        // never lands here.
        if let Some(io) = e.downcast_ref::<std::io::Error>() {
            let no_answer = matches!(
                io.kind(),
                ErrorKind::ConnectionRefused
                    | ErrorKind::HostUnreachable
                    | ErrorKind::NetworkUnreachable
                    | ErrorKind::AddrNotAvailable
                    | ErrorKind::TimedOut
            );
            if no_answer && kind == AddrFailure::Other {
                kind = AddrFailure::NoAnswer;
            }
        }
    });
    if upgrade {
        AddrFailure::Other
    } else {
        kind
    }
}

/// Whether a failed dial says the peer can't be reached at what we know of it: no address known,
/// or nothing answered at any address it was dialled on. QUIC handshake timeouts count only next to
/// a TCP address that got no answer in the same dial (QUIC alone can't tell a dead peer from a busy
/// one, so a QUIC-only peer is never dropped on timeouts: that errs on keeping a healthy peer). Not
/// `WrongPeerId`: someone else answering at an address says nothing about the peer itself.
pub fn dial_means_unreachable(error: &libp2p::swarm::DialError) -> bool {
    use libp2p::swarm::DialError;
    match error {
        DialError::NoAddresses => true,
        DialError::Transport(errors) => {
            let kinds: Vec<AddrFailure> = errors.iter().map(|(_, e)| addr_failure(e)).collect();
            !kinds.contains(&AddrFailure::Other) && kinds.contains(&AddrFailure::NoAnswer)
        }
        _ => false,
    }
}

/// Build the QUIC transport (UDP-based, built-in TLS 1.3 + multiplexing).
///
/// QUIC provides 0-RTT connections and is better for NAT hole punching (UDP).
/// No Noise/Yamux wrapping needed — QUIC handles encryption and muxing natively.
pub fn build_quic_transport(
    keypair: &identity::Keypair,
) -> std::io::Result<libp2p::core::transport::Boxed<(libp2p::PeerId, libp2p::core::muxing::StreamMuxerBox)>>
{
    let mut quic_config = libp2p::quic::Config::new(keypair)
        .initial_mtu(1452)
        .congestion_controller_factory(Arc::new(
            quinn::congestion::BbrConfig::default(),
        ));
    // Load simulation (plan E3): OPENHYDRA_TEST_QUIC_WINDOW_KB caps the per-stream and per-connection receive
    // windows (default 10 MB / 15 MB), OPENHYDRA_TEST_QUIC_STREAMS the concurrent streams (default 256).
    if let Some(kb) = std::env::var("OPENHYDRA_TEST_QUIC_WINDOW_KB").ok().and_then(|v| v.parse::<u32>().ok()) {
        quic_config.max_stream_data = kb.max(64) * 1024;
        quic_config.max_connection_data = kb.max(64) * 1024 * 3 / 2;
    }
    if let Some(n) = std::env::var("OPENHYDRA_TEST_QUIC_STREAMS").ok().and_then(|v| v.parse::<u32>().ok()) {
        quic_config.max_concurrent_stream_limit = n.max(8);
    }
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
    /// Mesh verification 2026-10-30: after a restart a TCP dial from our (reused) listen port failed
    /// with "Address already in use" for ~30 s; only those addresses are picked out for a new-port dial.
    #[test]
    fn address_in_use_dial_failures_are_picked_out() {
        use libp2p::{swarm::DialError, Multiaddr, TransportError};
        let a: Multiaddr = "/ip4/203.0.113.1/tcp/4001".parse().unwrap();
        let b: Multiaddr = "/ip6/2001:db8::1/tcp/4001".parse().unwrap();
        let err = DialError::Transport(vec![
            (a.clone(), TransportError::Other(std::io::Error::from(std::io::ErrorKind::AddrInUse))),
            (b.clone(), TransportError::Other(std::io::Error::from(std::io::ErrorKind::ConnectionRefused))),
        ]);
        assert_eq!(addrs_refused_as_in_use(&err), vec![a]);
        let wrapped = DialError::Transport(vec![(b.clone(), TransportError::Other(std::io::Error::new(
            std::io::ErrorKind::Other, std::io::Error::from(std::io::ErrorKind::AddrInUse))))]);
        assert_eq!(addrs_refused_as_in_use(&wrapped), vec![b]);
    }

    use super::*;

    #[tokio::test]
    async fn a_public_only_transport_refuses_to_dial_local_and_private_addresses() {
        use libp2p::core::transport::{DialOpts, PortUse, TransportError};
        use libp2p::core::Endpoint;
        let key = identity::Keypair::generate_ed25519();
        let mut t = public_dials_only(build_tcp_transport(&key).unwrap());
        let opts = || DialOpts { role: Endpoint::Dialer, port_use: PortUse::New };
        for refused in ["/ip4/127.0.0.1/tcp/22", "/ip4/169.254.169.254/tcp/80", "/ip4/10.0.0.1/tcp/4001", "/ip6/::1/tcp/4001", "/ip6/fd00::1/tcp/1", "/dns4/localhost/tcp/1"] {
            let r = t.dial(refused.parse().unwrap(), opts());
            assert!(matches!(r, Err(TransportError::MultiaddrNotSupported(_))), "{refused} should be refused");
        }
        // A public address is handed on (the dial itself isn't awaited).
        assert!(!matches!(t.dial("/ip4/8.8.8.8/tcp/4001".parse().unwrap(), opts()), Err(TransportError::MultiaddrNotSupported(_))));
    }
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

#[cfg(test)]
mod dial_failure_tests {
    use super::*;
    use libp2p::core::transport::{DialOpts, PortUse};
    use libp2p::core::Endpoint;
    use libp2p::Multiaddr;

    /// The transport a bootstrap or agent dials with (TCP and QUIC, with the setup timeouts).
    fn transport() -> BoxedTransport {
        let key = identity::Keypair::generate_ed25519();
        let tcp = build_tcp_transport(&key).unwrap();
        let quic = build_quic_transport(&key).unwrap();
        with_setup_timeouts(either_transport(quic, tcp), OUTBOUND_SETUP_TIMEOUT, INBOUND_SETUP_TIMEOUT)
    }

    /// Dial `addr` and classify how it failed (from the real error, not a constructed one).
    async fn failure(addr: &str) -> AddrFailure {
        let mut t = transport();
        let addr: Multiaddr = addr.parse().unwrap();
        let opts = DialOpts { role: Endpoint::Dialer, port_use: PortUse::New };
        let dial = match t.dial(addr, opts) {
            Ok(d) => d,
            Err(e) => return addr_failure(&e),
        };
        // Drive the transport (QUIC needs its endpoint polled) while the dial runs.
        let mut dial = Box::pin(dial);
        let result = futures::future::poll_fn(|cx| {
            use std::future::Future as _;
            let _ = std::pin::Pin::new(&mut t).poll(cx);
            dial.as_mut().poll(cx)
        });
        match tokio::time::timeout(std::time::Duration::from_secs(40), result).await {
            Ok(Err(e)) => addr_failure(&libp2p::TransportError::Other(e)),
            Ok(Ok(_)) => panic!("dial to {} unexpectedly succeeded", "the test address"),
            Err(_) => panic!("dial didn't finish"),
        }
    }

    fn closed_port() -> u16 {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let p = l.local_addr().unwrap().port();
        drop(l);
        p
    }

    #[tokio::test]
    async fn a_refused_tcp_connect_is_no_answer() {
        assert_eq!(failure(&format!("/ip4/127.0.0.1/tcp/{}", closed_port())).await, AddrFailure::NoAnswer);
    }

    #[tokio::test]
    async fn a_tcp_peer_that_answers_then_hangs_up_is_not_no_answer() {
        // A peer at its connection limit: it accepts the TCP connection, then closes it during the
        // handshake. It answered, so it's alive.
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for conn in l.incoming().take(1) {
                drop(conn);
            }
        });
        assert_eq!(failure(&format!("/ip4/127.0.0.1/tcp/{port}")).await, AddrFailure::Other);
    }

    #[tokio::test]
    async fn a_tcp_connect_nobody_answers_times_out_as_no_answer() {
        // TEST-NET-1: routed nowhere, so the SYN goes unanswered (or the OS says unreachable).
        let started = std::time::Instant::now();
        assert_eq!(failure("/ip4/192.0.2.1/tcp/4001").await, AddrFailure::NoAnswer);
        assert!(started.elapsed() < OUTBOUND_SETUP_TIMEOUT, "the TCP connect timeout, not the 30 s one");
    }

    #[tokio::test]
    async fn a_quic_dial_nobody_answers_is_a_quic_timeout() {
        assert_eq!(failure(&format!("/ip4/127.0.0.1/udp/{}/quic-v1", closed_port())).await, AddrFailure::QuicTimeout);
    }

    #[tokio::test]
    async fn a_handshake_that_stalls_past_the_setup_limit_is_not_no_answer() {
        // A peer that takes the TCP connection and then says nothing: it answered, so a setup
        // timeout must not count as "nobody there".
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let held: Vec<_> = l.incoming().take(1).collect();
            std::thread::sleep(std::time::Duration::from_secs(5));
            drop(held);
        });
        let key = identity::Keypair::generate_ed25519();
        let short = std::time::Duration::from_secs(1);
        let mut t = with_setup_timeouts(build_tcp_transport(&key).unwrap(), short, short);
        let addr: Multiaddr = format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap();
        let dial = t.dial(addr, DialOpts { role: Endpoint::Dialer, port_use: PortUse::New }).unwrap();
        let e = match dial.await {
            Err(e) => e,
            Ok(_) => panic!("the handshake can't succeed"),
        };
        assert!(e.to_string().contains("connection setup timed out"), "{e}");
        assert_eq!(addr_failure(&libp2p::TransportError::Other(e)), AddrFailure::Other);
    }

    #[test]
    fn a_listen_error_says_what_went_wrong() {
        // libp2p's timeout wrapper printed nothing for the error inside it, so a port in use came
        // out as an empty message.
        let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = taken.local_addr().unwrap().port();
        let key = identity::Keypair::generate_ed25519();
        let mut t = with_setup_timeouts(build_tcp_transport(&key).unwrap(), OUTBOUND_SETUP_TIMEOUT, INBOUND_SETUP_TIMEOUT);
        let addr: Multiaddr = format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap();
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let _guard = rt.enter();
        let err = t.listen_on(libp2p::core::transport::ListenerId::next(), addr).err().map(|e| error_chain(&e));
        match err {
            Some(msg) => assert!(msg.to_lowercase().contains("in use"), "message: {msg:?}"),
            // Some platforms report the bind failure asynchronously, as a listener event.
            None => {}
        }
    }

    #[test]
    fn only_a_dial_every_relay_refused_for_limits_counts_as_relay_limited() {
        use libp2p::relay::outbound::hop::ConnectError;
        use libp2p::swarm::DialError;
        let limited = || libp2p::TransportError::Other(std::io::Error::new(std::io::ErrorKind::Other, ConnectError::ResourceLimitExceeded));
        let no_reservation = || libp2p::TransportError::Other(std::io::Error::new(std::io::ErrorKind::Other, ConnectError::NoReservation));
        let a: Multiaddr = format!("/ip4/45.79.190.172/tcp/4001/p2p/{}/p2p-circuit", libp2p::PeerId::random()).parse().unwrap();
        let b: Multiaddr = format!("/ip4/85.209.48.209/tcp/4001/p2p/{}/p2p-circuit", libp2p::PeerId::random()).parse().unwrap();
        assert!(dial_refused_by_relay_limits(&DialError::Transport(vec![(a.clone(), limited()), (b.clone(), limited())])));
        assert!(
            !dial_refused_by_relay_limits(&DialError::Transport(vec![(a, limited()), (b, no_reservation())])),
            "a relay with no reservation for it: the provider is still joining"
        );
        assert!(!dial_refused_by_relay_limits(&DialError::NoAddresses));
        assert!(!dial_refused_by_relay_limits(&DialError::Transport(Vec::new())));
    }

    #[test]
    fn quic_timeouts_count_only_next_to_a_tcp_no_answer() {
        use libp2p::swarm::DialError;
        let err = |k: std::io::ErrorKind| libp2p::TransportError::Other(std::io::Error::from(k));
        let quic = || {
            libp2p::TransportError::Other(std::io::Error::new(std::io::ErrorKind::Other, libp2p::quic::Error::HandshakeTimedOut))
        };
        let a: Multiaddr = "/ip4/1.2.3.4/udp/4001/quic-v1".parse().unwrap();
        let b: Multiaddr = "/ip4/1.2.3.4/tcp/4001".parse().unwrap();
        assert!(!dial_means_unreachable(&DialError::Transport(vec![(a.clone(), quic())])), "QUIC alone can't tell");
        assert!(dial_means_unreachable(&DialError::Transport(vec![
            (a.clone(), quic()),
            (b.clone(), err(std::io::ErrorKind::ConnectionRefused)),
        ])));
        assert!(!dial_means_unreachable(&DialError::Transport(vec![
            (a, quic()),
            (b, err(std::io::ErrorKind::ConnectionReset)),
        ])), "the TCP side answered");
    }
}
