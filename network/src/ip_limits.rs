//! Limits on the incoming connections one network address can make a node hold (review 2026-10-14,
//! M8). `connection_limits` bounds the totals and each peer id, but peer ids are free: one host can
//! fill a node's whole incoming allowance under fresh identities and turn every other peer away. This
//! counts by the remote address instead: an IPv4 address, or an IPv6 /64 (what one host or one home
//! network gets), and an IPv6 /48 too, which may hold [`SITE_SHARE`] times as many (one site's many
//! /64s; second review 2026-10-19: a /48 alone is 65,536 /64s, so one attacker could otherwise spread
//! over enough of them to fill the node). Relayed connections aren't counted here (the relay's own
//! limits bound them), nor, by default, loopback ones (local agents: the desktop's own gateway and
//! provider, test rigs). A QUIC connection still being set up isn't counted per address either: its
//! sender's address isn't proven until the handshake ends (a forged source could otherwise use up
//! another host's share), and the QUIC listener makes senders prove it under a burst.

use std::collections::HashMap;
use std::net::{Ipv4Addr, Ipv6Addr};

use libp2p::core::transport::PortUse;
use libp2p::core::Endpoint;
use libp2p::multiaddr::Protocol;
use libp2p::swarm::{
    dummy, ConnectionClosed, ConnectionDenied, ConnectionId, FromSwarm, ListenFailure, NetworkBehaviour, THandler,
    THandlerInEvent, THandlerOutEvent, ToSwarm,
};
use libp2p::{Multiaddr, PeerId};

/// How many incoming connections one address may have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IpLimitsConfig {
    /// Still being set up (handshakes).
    pub max_pending_per_ip: u32,
    /// Established.
    pub max_established_per_ip: u32,
    /// Loopback connections aren't counted.
    pub exempt_loopback: bool,
}

impl IpLimitsConfig {
    /// A peer node.
    pub const PEER: Self = Self { max_pending_per_ip: 8, max_established_per_ip: 32, exempt_loopback: true };
    /// A bootstrap: carrier-grade NAT puts many users behind one IPv4 address.
    pub const BOOTSTRAP: Self = Self { max_pending_per_ip: 32, max_established_per_ip: 256, exempt_loopback: true };
}

/// How many times one address's limit an IPv6 /48 may hold.
pub const SITE_SHARE: u32 = 4;

/// A network a remote address is counted under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Net {
    V4(Ipv4Addr),
    /// The /64 prefix.
    V6([u8; 8]),
    /// The /48 prefix.
    V6Site([u8; 6]),
}

impl Net {
    /// What this network may hold, for a per-address `limit`.
    fn limit(self, limit: u32) -> u32 {
        match self {
            Net::V6Site(_) => limit.saturating_mul(SITE_SHARE),
            _ => limit,
        }
    }
}

/// The networks `remote` is counted under: none for a relayed connection, or a loopback one when
/// loopback is exempt.
fn nets_of(remote: &Multiaddr, exempt_loopback: bool) -> Vec<Net> {
    if remote.iter().any(|p| matches!(p, Protocol::P2pCircuit)) {
        return Vec::new();
    }
    let v4 = |ip: Ipv4Addr| if exempt_loopback && ip.is_loopback() { Vec::new() } else { vec![Net::V4(ip)] };
    match remote.iter().next() {
        Some(Protocol::Ip4(ip)) => v4(ip),
        Some(Protocol::Ip6(ip)) => match ip.to_ipv4_mapped() {
            Some(mapped) => v4(mapped),
            None if exempt_loopback && ip == Ipv6Addr::LOCALHOST => Vec::new(),
            None => {
                let o = ip.octets();
                let mut prefix = [0u8; 8];
                prefix.copy_from_slice(&o[..8]);
                let mut site = [0u8; 6];
                site.copy_from_slice(&o[..6]);
                vec![Net::V6(prefix), Net::V6Site(site)]
            }
        },
        _ => Vec::new(),
    }
}

/// Whether `remote` is a QUIC address.
fn is_quic(remote: &Multiaddr) -> bool {
    remote.iter().any(|p| matches!(p, Protocol::QuicV1 | Protocol::Quic))
}

/// An address that has as many connections as it may.
#[derive(Debug)]
pub struct TooManyFromThisAddress {
    pending: bool,
    limit: u32,
}

impl std::fmt::Display for TooManyFromThisAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = if self.pending { "connections being set up" } else { "connections" };
        write!(f, "this address already has {} incoming {what}", self.limit)
    }
}

impl std::error::Error for TooManyFromThisAddress {}

/// The behaviour: put it first in a node's behaviour, before `connection_limits`.
pub struct Behaviour {
    config: IpLimitsConfig,
    pending: HashMap<ConnectionId, Vec<Net>>,
    established: HashMap<ConnectionId, Vec<Net>>,
    pending_per_net: HashMap<Net, u32>,
    established_per_net: HashMap<Net, u32>,
}

fn bump(counts: &mut HashMap<Net, u32>, net: Net) {
    *counts.entry(net).or_default() += 1;
}

fn drop_one(counts: &mut HashMap<Net, u32>, net: Net) {
    if let Some(n) = counts.get_mut(&net) {
        *n = n.saturating_sub(1);
        if *n == 0 {
            counts.remove(&net);
        }
    }
}

impl Behaviour {
    pub fn new(config: IpLimitsConfig) -> Self {
        Self {
            config,
            pending: HashMap::new(),
            established: HashMap::new(),
            pending_per_net: HashMap::new(),
            established_per_net: HashMap::new(),
        }
    }

    fn forget_pending(&mut self, connection_id: ConnectionId) {
        for net in self.pending.remove(&connection_id).unwrap_or_default() {
            drop_one(&mut self.pending_per_net, net);
        }
    }
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler = dummy::ConnectionHandler;
    type ToSwarm = std::convert::Infallible;

    fn handle_pending_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        _local: &Multiaddr,
        remote: &Multiaddr,
    ) -> Result<(), ConnectionDenied> {
        // A QUIC sender's address is proven only once its handshake ends: counted then.
        if is_quic(remote) {
            return Ok(());
        }
        let nets = nets_of(remote, self.config.exempt_loopback);
        let per_ip = self.config.max_pending_per_ip;
        if let Some(full) = nets.iter().find(|n| self.pending_per_net.get(n).copied().unwrap_or(0) >= n.limit(per_ip)) {
            return Err(ConnectionDenied::new(TooManyFromThisAddress { pending: true, limit: full.limit(per_ip) }));
        }
        if nets.is_empty() {
            return Ok(());
        }
        for &net in &nets {
            bump(&mut self.pending_per_net, net);
        }
        self.pending.insert(connection_id, nets);
        Ok(())
    }

    fn handle_established_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        _peer: PeerId,
        _local: &Multiaddr,
        remote: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        self.forget_pending(connection_id);
        let per_ip = self.config.max_established_per_ip;
        let nets = nets_of(remote, self.config.exempt_loopback);
        if let Some(full) = nets.iter().find(|n| self.established_per_net.get(n).copied().unwrap_or(0) >= n.limit(per_ip)) {
            return Err(ConnectionDenied::new(TooManyFromThisAddress { pending: false, limit: full.limit(per_ip) }));
        }
        Ok(dummy::ConnectionHandler)
    }

    fn handle_established_outbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        _peer: PeerId,
        _addr: &Multiaddr,
        _role_override: Endpoint,
        _port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(dummy::ConnectionHandler)
    }

    fn on_swarm_event(&mut self, event: FromSwarm) {
        match event {
            // Counted once every behaviour accepted it.
            FromSwarm::ConnectionEstablished(e) => {
                if let libp2p::core::ConnectedPoint::Listener { send_back_addr, .. } = e.endpoint {
                    let nets = nets_of(send_back_addr, self.config.exempt_loopback);
                    if !nets.is_empty() {
                        for &net in &nets {
                            bump(&mut self.established_per_net, net);
                        }
                        self.established.insert(e.connection_id, nets);
                    }
                }
            }
            FromSwarm::ConnectionClosed(ConnectionClosed { connection_id, .. }) => {
                for net in self.established.remove(&connection_id).unwrap_or_default() {
                    drop_one(&mut self.established_per_net, net);
                }
            }
            FromSwarm::ListenFailure(ListenFailure { connection_id, .. }) => self.forget_pending(connection_id),
            _ => {}
        }
    }

    fn on_connection_handler_event(&mut self, _peer: PeerId, _id: ConnectionId, event: THandlerOutEvent<Self>) {
        match event {}
    }

    fn poll(&mut self, _cx: &mut std::task::Context<'_>) -> std::task::Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        std::task::Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
    use libp2p::swarm::{NetworkBehaviour, Swarm, SwarmEvent};
    use std::time::Duration;

    #[test]
    fn addresses_are_counted_by_host_and_v6_slash_64_and_slash_48() {
        let m = |s: &str| s.parse::<Multiaddr>().unwrap();
        let a = nets_of(&m("/ip4/203.0.113.7/tcp/4001"), true);
        assert_eq!(a, vec![Net::V4("203.0.113.7".parse().unwrap())]);
        assert_eq!(nets_of(&m("/ip6/2001:db8:1:2:aaaa::1/udp/4001/quic-v1"), true), nets_of(&m("/ip6/2001:db8:1:2:bbbb::9/tcp/1"), true));
        let (x, y) = (nets_of(&m("/ip6/2001:db8:1:2::1/tcp/1"), true), nets_of(&m("/ip6/2001:db8:1:3::1/tcp/1"), true));
        assert_ne!(x[0], y[0], "another /64");
        assert_eq!(x[1], y[1], "the same /48");
        assert_ne!(nets_of(&m("/ip6/2001:db8:2::1/tcp/1"), true)[1], x[1], "another /48");
        assert_eq!(nets_of(&m("/ip6/::ffff:203.0.113.7/tcp/1"), true), a, "IPv4-mapped counts as the IPv4 address");
        assert!(nets_of(&m("/ip4/127.0.0.1/tcp/1"), true).is_empty(), "loopback exempt");
        assert!(!nets_of(&m("/ip4/127.0.0.1/tcp/1"), false).is_empty());
        assert!(nets_of(&m("/ip6/::1/tcp/1"), true).is_empty());
        let circuit = m("/ip4/203.0.113.7/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit");
        assert!(nets_of(&circuit, true).is_empty(), "relayed: the relay's limits apply");
    }

    /// Second review 2026-10-19: one site's /64s together get [`SITE_SHARE`] addresses' worth, and a
    /// QUIC connection being set up (its sender unproven) isn't counted per address.
    #[test]
    fn many_slash_64s_of_one_site_share_its_limit_and_unproven_quic_is_not_counted() {
        let config = IpLimitsConfig { max_pending_per_ip: 2, max_established_per_ip: 2, exempt_loopback: true };
        let mut b = Behaviour::new(config);
        let local: Multiaddr = "/ip6/::/tcp/4001".parse().unwrap();
        let mut held = 0;
        for i in 0..64u32 {
            let remote: Multiaddr = format!("/ip6/2001:db8:7:{i:x}::1/tcp/5000").parse().unwrap();
            if b.handle_pending_inbound_connection(ConnectionId::new_unchecked(i as usize), &local, &remote).is_ok() {
                held += 1;
            }
        }
        assert_eq!(held, 2 * SITE_SHARE, "64 /64s of one /48 hold the site's share, not 64 addresses' worth");
        let other: Multiaddr = "/ip6/2001:db8:8::1/tcp/5000".parse().unwrap();
        assert!(b.handle_pending_inbound_connection(ConnectionId::new_unchecked(100), &local, &other).is_ok(), "another site is unaffected");
        // Forged-source QUIC: never counted while pending, however many.
        let victim: Multiaddr = "/ip4/198.51.100.9/udp/4001/quic-v1".parse().unwrap();
        for i in 0..20 {
            assert!(b.handle_pending_inbound_connection(ConnectionId::new_unchecked(200 + i), &local, &victim).is_ok());
        }
        assert!(!b.pending_per_net.contains_key(&Net::V4("198.51.100.9".parse().unwrap())));
        let tcp: Multiaddr = "/ip4/198.51.100.9/tcp/4001".parse().unwrap();
        assert!(b.handle_pending_inbound_connection(ConnectionId::new_unchecked(300), &local, &tcp).is_ok(), "the victim can still connect");
    }

    #[derive(NetworkBehaviour)]
    #[behaviour(prelude = "libp2p::swarm::derive_prelude")]
    struct Limited {
        ip: Behaviour,
        keep: libp2p::swarm::dummy::Behaviour,
    }

    fn node(config: IpLimitsConfig) -> Swarm<Limited> {
        let key = libp2p::identity::Keypair::generate_ed25519();
        let transport = crate::transport::build_tcp_transport(&key).unwrap();
        let cfg = libp2p::swarm::Config::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(60));
        Swarm::new(transport, Limited { ip: Behaviour::new(config), keep: libp2p::swarm::dummy::Behaviour }, PeerId::from(key.public()), cfg)
    }

    #[tokio::test]
    async fn one_address_cannot_make_us_hold_more_than_its_share() {
        // Loopback counted here, so the test's connections (all from 127.0.0.1) are one address.
        let config = IpLimitsConfig { max_pending_per_ip: 64, max_established_per_ip: 6, exempt_loopback: false };
        let mut a = node(config);
        a.listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap()).unwrap();
        let addr = loop {
            if let SwarmEvent::NewListenAddr { address, .. } = a.select_next_some().await {
                break address;
            }
        };
        // Ten dialers, fresh identities each (as an attacker would use).
        let mut dialers: Vec<Swarm<Limited>> = (0..10).map(|_| node(IpLimitsConfig::PEER)).collect();
        for d in dialers.iter_mut() {
            d.dial(DialOpts::peer_id(*a.local_peer_id()).addresses(vec![addr.clone()]).condition(PeerCondition::Always).build()).unwrap();
        }
        let deadline = tokio::time::sleep(Duration::from_secs(5));
        tokio::pin!(deadline);
        loop {
            let mut any = futures::future::select_all(dialers.iter_mut().map(|d| Box::pin(d.select_next_some())));
            tokio::select! {
                _ = a.select_next_some() => {}
                _ = &mut any => {}
                _ = &mut deadline => break,
            }
        }
        assert_eq!(a.network_info().connection_counters().num_established(), 6, "ten identities, one address: six held");
        // Another address is unaffected: the limit is per address, not in all.
        assert!(a.behaviour().ip.established_per_net.len() == 1);
    }

    /// Second review 2026-10-19: more new QUIC senders at once than the listener takes unproven are
    /// asked to prove their address (a Retry), and still connect.
    #[tokio::test]
    async fn a_burst_of_quic_dials_all_connect_through_the_address_check() {
        let quic_node = || {
            let key = libp2p::identity::Keypair::generate_ed25519();
            let transport = crate::transport::build_quic_transport(&key).unwrap();
            let cfg = libp2p::swarm::Config::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(60));
            Swarm::new(transport, Limited { ip: Behaviour::new(IpLimitsConfig::PEER), keep: libp2p::swarm::dummy::Behaviour }, PeerId::from(key.public()), cfg)
        };
        let mut a = quic_node();
        a.listen_on("/ip4/127.0.0.1/udp/0/quic-v1".parse().unwrap()).unwrap();
        let addr = loop {
            if let SwarmEvent::NewListenAddr { address, .. } = a.select_next_some().await {
                break address;
            }
        };
        let mut dialers: Vec<Swarm<Limited>> = (0..40).map(|_| quic_node()).collect();
        for d in dialers.iter_mut() {
            d.dial(DialOpts::peer_id(*a.local_peer_id()).addresses(vec![addr.clone()]).build()).unwrap();
        }
        let deadline = tokio::time::sleep(Duration::from_secs(15));
        tokio::pin!(deadline);
        while a.network_info().connection_counters().num_established() < 40 {
            let mut any = futures::future::select_all(dialers.iter_mut().map(|d| Box::pin(d.select_next_some())));
            tokio::select! {
                _ = a.select_next_some() => {}
                _ = &mut any => {}
                _ = &mut deadline => break,
            }
        }
        assert_eq!(a.network_info().connection_counters().num_established(), 40);
    }

    #[tokio::test]
    async fn silent_sockets_from_one_address_leave_room_for_others() {
        let config = IpLimitsConfig { max_pending_per_ip: 3, max_established_per_ip: 32, exempt_loopback: false };
        let mut a = node(config);
        a.listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap()).unwrap();
        let addr = loop {
            if let SwarmEvent::NewListenAddr { address, .. } = a.select_next_some().await {
                break address;
            }
        };
        let port = addr.iter().find_map(|p| if let Protocol::Tcp(p) = p { Some(p) } else { None }).unwrap();
        let mut held = Vec::new();
        for _ in 0..10 {
            held.push(tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap());
        }
        let settle = tokio::time::sleep(Duration::from_millis(300));
        tokio::pin!(settle);
        loop {
            tokio::select! {
                _ = a.select_next_some() => {}
                _ = &mut settle => break,
            }
        }
        assert_eq!(a.behaviour().ip.pending.len(), 3, "only three of ten silent sockets hold a slot");
        // And another address still gets in.
        let other = libp2p::swarm::ConnectionId::new_unchecked(usize::MAX);
        let local: Multiaddr = "/ip4/127.0.0.1/tcp/1".parse().unwrap();
        let elsewhere: Multiaddr = "/ip4/127.0.0.2/tcp/1".parse().unwrap();
        assert!(a.behaviour_mut().ip.handle_pending_inbound_connection(other, &local, &elsewhere).is_ok());
    }
}
