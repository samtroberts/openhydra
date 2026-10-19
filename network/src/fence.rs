//! A private deployment's fence (second review 2026-10-19, after M1): with OpenHydra's public network
//! off (`--public-relays off`), the node never connects to OpenHydra's own bootstraps and relays, by
//! peer id or by address, however it hears of them. Dropping them from the configured bootstraps isn't
//! enough: a peer that is also on the public network can name them (its relayed listen address in
//! identify, the peers a DHT lookup returns, the routing cache), and a stale entry can carry another
//! peer id at the same address. Two parts:
//! * [`FenceTransport`] refuses to dial an address that names one of them (its IP, a `*.openhydra.co`
//!   name, or its peer id, a relay hop included), before any packet leaves;
//! * [`Behaviour`] refuses a connection to or from their peer ids, whatever the address.

use std::collections::HashSet;
use std::net::IpAddr;
use std::pin::Pin;
use std::task::{Context, Poll};

use libp2p::core::transport::{DialOpts, ListenerId, PortUse, TransportError, TransportEvent};
use libp2p::core::{Endpoint, Transport};
use libp2p::multiaddr::Protocol;
use libp2p::swarm::{
    dummy, ConnectionDenied, ConnectionId, FromSwarm, NetworkBehaviour, THandler, THandlerInEvent, THandlerOutEvent,
    ToSwarm,
};
use libp2p::{Multiaddr, PeerId};

use crate::transport::BoxedTransport;

/// What the fence keeps out.
#[derive(Debug, Clone, Default)]
pub struct Fenced {
    ips: HashSet<IpAddr>,
    peers: HashSet<PeerId>,
}

impl Fenced {
    /// OpenHydra's own bootstraps and relays.
    pub fn openhydra() -> Self {
        Self {
            ips: crate::relay::BOOTSTRAP_RELAY_IPS.iter().filter_map(|ip| ip.parse().ok()).collect(),
            peers: crate::relay::bootstrap_relay_peer_ids().into_iter().collect(),
        }
    }

    /// Whether `peer` is one of them.
    pub fn peer(&self, peer: &PeerId) -> bool {
        self.peers.contains(peer)
    }

    /// Whether `addr` names one of them anywhere: an IP, an `openhydra.co` name, a peer id (a relay
    /// hop's too).
    pub fn addr(&self, addr: &Multiaddr) -> bool {
        addr.iter().any(|p| match p {
            Protocol::Ip4(ip) => self.ips.contains(&IpAddr::V4(ip)),
            Protocol::Ip6(ip) => self.ips.contains(&IpAddr::V6(ip)),
            Protocol::Dns(n) | Protocol::Dns4(n) | Protocol::Dns6(n) | Protocol::Dnsaddr(n) => {
                let n = n.trim_end_matches('.').to_ascii_lowercase();
                n == "openhydra.co" || n.ends_with(".openhydra.co")
            }
            Protocol::P2p(pid) => self.peers.contains(&pid),
            _ => false,
        })
    }
}

fn refused(addr: &Multiaddr) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!("{addr} is OpenHydra's public network, which this private deployment doesn't use"),
    )
}

/// Wraps a node's transport: dials to a fenced address fail at once.
pub struct FenceTransport {
    inner: BoxedTransport,
    fenced: Fenced,
}

impl FenceTransport {
    pub fn new(inner: BoxedTransport, fenced: Fenced) -> Self {
        Self { inner, fenced }
    }
}

impl Transport for FenceTransport {
    type Output = <BoxedTransport as Transport>::Output;
    type Error = <BoxedTransport as Transport>::Error;
    type ListenerUpgrade = <BoxedTransport as Transport>::ListenerUpgrade;
    type Dial = <BoxedTransport as Transport>::Dial;

    fn listen_on(&mut self, id: ListenerId, addr: Multiaddr) -> Result<(), TransportError<Self::Error>> {
        // Listening through one of OpenHydra's relays is a dial to it.
        if self.fenced.addr(&addr) {
            return Err(TransportError::Other(refused(&addr)));
        }
        self.inner.listen_on(id, addr)
    }

    fn remove_listener(&mut self, id: ListenerId) -> bool {
        self.inner.remove_listener(id)
    }

    fn dial(&mut self, addr: Multiaddr, opts: DialOpts) -> Result<Self::Dial, TransportError<Self::Error>> {
        if self.fenced.addr(&addr) {
            return Err(TransportError::Other(refused(&addr)));
        }
        self.inner.dial(addr, opts)
    }

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        Pin::new(&mut self.inner).poll(cx)
    }
}

/// Refused connection to or from one of OpenHydra's own peers.
#[derive(Debug)]
pub struct FencedPeer;

impl std::fmt::Display for FencedPeer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OpenHydra's public network, which this private deployment doesn't use")
    }
}

impl std::error::Error for FencedPeer {}

/// The behaviour: refuses connections with fenced peers (off: refuses nothing).
pub struct Behaviour {
    fenced: Option<Fenced>,
}

impl Behaviour {
    pub fn new(fenced: Option<Fenced>) -> Self {
        Self { fenced }
    }

    fn check_peer(&self, peer: Option<PeerId>) -> Result<(), ConnectionDenied> {
        match (&self.fenced, peer) {
            (Some(f), Some(p)) if f.peer(&p) => Err(ConnectionDenied::new(FencedPeer)),
            _ => Ok(()),
        }
    }
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler = dummy::ConnectionHandler;
    type ToSwarm = std::convert::Infallible;

    fn handle_pending_inbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        _local: &Multiaddr,
        remote: &Multiaddr,
    ) -> Result<(), ConnectionDenied> {
        match &self.fenced {
            Some(f) if f.addr(remote) => Err(ConnectionDenied::new(FencedPeer)),
            _ => Ok(()),
        }
    }

    fn handle_pending_outbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        maybe_peer: Option<PeerId>,
        _addresses: &[Multiaddr],
        _effective_role: Endpoint,
    ) -> Result<Vec<Multiaddr>, ConnectionDenied> {
        self.check_peer(maybe_peer)?;
        Ok(Vec::new())
    }

    fn handle_established_inbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        peer: PeerId,
        _local: &Multiaddr,
        _remote: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        self.check_peer(Some(peer))?;
        Ok(dummy::ConnectionHandler)
    }

    fn handle_established_outbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        peer: PeerId,
        _addr: &Multiaddr,
        _role_override: Endpoint,
        _port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        self.check_peer(Some(peer))?;
        Ok(dummy::ConnectionHandler)
    }

    fn on_swarm_event(&mut self, _event: FromSwarm) {}

    fn on_connection_handler_event(&mut self, _peer: PeerId, _id: ConnectionId, event: THandlerOutEvent<Self>) {
        match event {}
    }

    fn poll(&mut self, _cx: &mut Context<'_>) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bootstrap_us() -> (Multiaddr, PeerId) {
        let a: Multiaddr = crate::relay::BOOTSTRAP_RELAYS[0].parse().unwrap();
        let pid = a.iter().find_map(|p| if let Protocol::P2p(p) = p { Some(p) } else { None }).unwrap();
        (a, pid)
    }

    #[test]
    fn openhydras_own_addresses_are_fenced_however_they_are_named() {
        let f = Fenced::openhydra();
        let (us, pid) = bootstrap_us();
        assert!(f.addr(&us));
        assert!(f.peer(&pid));
        let m = |s: &str| s.parse::<Multiaddr>().unwrap();
        // Its address with another peer id (a stale or rotated entry), and by name.
        assert!(f.addr(&m("/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEzegXr4qcj37EWF2aQo9vp121MGrCaCwYcJF2oTkW3WT")));
        assert!(f.addr(&m("/ip4/45.79.190.172/udp/4001/quic-v1")));
        assert!(f.addr(&m("/dns4/bootstrap-us.openhydra.co/tcp/4001")));
        assert!(f.addr(&m("/dnsaddr/openhydra.co")));
        // A peer that is also on the public network, reached through OpenHydra's relay.
        let circuit = format!("{}/p2p-circuit/p2p/12D3KooWHNQ9nMedT3ZaMjj4xX7Mf4dxmTZ3cAX3tnJfyERV9Bua", us);
        assert!(f.addr(&m(&circuit)));
        // Everything else is untouched.
        assert!(!f.addr(&m("/ip4/192.168.1.20/tcp/4001")));
        assert!(!f.addr(&m("/dns4/relay.example.com/tcp/4001")));
        assert!(!f.addr(&m("/dns4/notopenhydra.co/tcp/4001")));
    }

    #[test]
    fn the_behaviour_refuses_openhydras_peers_only_when_on() {
        let (_, pid) = bootstrap_us();
        let other = PeerId::random();
        let mut on = Behaviour::new(Some(Fenced::openhydra()));
        assert!(on.handle_pending_outbound_connection(ConnectionId::new_unchecked(1), Some(pid), &[], Endpoint::Dialer).is_err());
        assert!(on.handle_pending_outbound_connection(ConnectionId::new_unchecked(2), Some(other), &[], Endpoint::Dialer).is_ok());
        let local: Multiaddr = "/ip4/0.0.0.0/tcp/4001".parse().unwrap();
        let from_bootstrap: Multiaddr = "/ip4/45.79.190.172/tcp/40000".parse().unwrap();
        assert!(on.handle_pending_inbound_connection(ConnectionId::new_unchecked(3), &local, &from_bootstrap).is_err());
        let mut off = Behaviour::new(None);
        assert!(off.handle_pending_outbound_connection(ConnectionId::new_unchecked(4), Some(pid), &[], Endpoint::Dialer).is_ok());
    }

    #[tokio::test]
    async fn the_transport_refuses_to_dial_them_before_sending_anything() {
        let key = libp2p::identity::Keypair::generate_ed25519();
        let mut t = FenceTransport::new(crate::transport::build_tcp_transport(&key).unwrap(), Fenced::openhydra());
        let opts = DialOpts { role: Endpoint::Dialer, port_use: PortUse::New };
        let (us, _) = bootstrap_us();
        match t.dial(us, opts) {
            Err(TransportError::Other(e)) => assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied),
            other => panic!("dialled OpenHydra's bootstrap: {:?}", other.is_ok()),
        }
        // Anything else goes to the inner transport as before (here: a local port nobody listens on).
        assert!(t.dial("/ip4/127.0.0.1/tcp/9".parse().unwrap(), opts).is_ok());
    }
}
