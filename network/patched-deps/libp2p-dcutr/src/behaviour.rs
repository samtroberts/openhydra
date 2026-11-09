// Copyright 2021 Protocol Labs.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

//! [`NetworkBehaviour`] to act as a direct connection upgrade through relay node.

use crate::{handler, protocol};
use either::Either;
use libp2p_core::connection::ConnectedPoint;
use libp2p_core::multiaddr::Protocol;
use libp2p_core::transport::PortUse;
use libp2p_core::{Endpoint, Multiaddr};
use libp2p_identity::PeerId;
use libp2p_swarm::behaviour::{ConnectionClosed, DialFailure, ExpiredListenAddr, FromSwarm, NewListenAddr};
use libp2p_swarm::dial_opts::{self, DialOpts};
use libp2p_swarm::{
    dummy, ConnectionDenied, ConnectionHandler, ConnectionId, NewExternalAddrCandidate, THandler,
    THandlerOutEvent,
};
use libp2p_swarm::{NetworkBehaviour, NotifyHandler, THandlerInEvent, ToSwarm};
use lru::LruCache;
use std::collections::{HashMap, HashSet, VecDeque};
use std::num::NonZeroUsize;
use std::task::{Context, Poll};
use thiserror::Error;
use void::Void;

pub(crate) const MAX_NUMBER_OF_UPGRADE_ATTEMPTS: u8 = 3;

/// The events produced by the [`Behaviour`].
#[derive(Debug)]
pub struct Event {
    pub remote_peer_id: PeerId,
    pub result: Result<ConnectionId, Error>,
}

#[derive(Debug, Error)]
#[error("Failed to hole-punch connection: {inner}")]
pub struct Error {
    inner: InnerError,
}

#[derive(Debug, Error)]
enum InnerError {
    #[error("Giving up after {0} dial attempts")]
    AttemptsExceeded(u8),
    #[error("Inbound stream error: {0}")]
    InboundError(protocol::inbound::Error),
    #[error("Outbound stream error: {0}")]
    OutboundError(protocol::outbound::Error),
}

pub struct Behaviour {
    /// Queue of actions to return when polled.
    queued_events: VecDeque<ToSwarm<Event, Either<handler::relayed::Command, Void>>>,

    /// All direct (non-relayed) connections.
    direct_connections: HashMap<PeerId, HashSet<ConnectionId>>,

    address_candidates: Candidates,

    direct_to_relayed_connections: HashMap<ConnectionId, ConnectionId>,

    /// Indexed by the [`ConnectionId`] of the relayed connection and
    /// the [`PeerId`] we are trying to establish a direct connection to.
    outgoing_direct_connection_attempts: HashMap<(ConnectionId, PeerId), u8>,
}

impl Behaviour {
    pub fn new(local_peer_id: PeerId) -> Self {
        Behaviour {
            queued_events: Default::default(),
            direct_connections: Default::default(),
            address_candidates: Candidates::new(local_peer_id),
            direct_to_relayed_connections: Default::default(),
            outgoing_direct_connection_attempts: Default::default(),
        }
    }

    fn observed_addresses(&self) -> Vec<Multiaddr> {
        self.address_candidates.iter().cloned().collect()
    }

    fn on_dial_failure(
        &mut self,
        DialFailure {
            peer_id,
            connection_id: failed_direct_connection,
            ..
        }: DialFailure,
    ) {
        let Some(peer_id) = peer_id else {
            return;
        };

        let Some(relayed_connection_id) = self
            .direct_to_relayed_connections
            .get(&failed_direct_connection)
        else {
            return;
        };

        let Some(attempt) = self
            .outgoing_direct_connection_attempts
            .get(&(*relayed_connection_id, peer_id))
        else {
            return;
        };

        if *attempt < MAX_NUMBER_OF_UPGRADE_ATTEMPTS {
            self.queued_events.push_back(ToSwarm::NotifyHandler {
                handler: NotifyHandler::One(*relayed_connection_id),
                peer_id,
                event: Either::Left(handler::relayed::Command::Connect),
            })
        } else {
            self.queued_events.extend([ToSwarm::GenerateEvent(Event {
                remote_peer_id: peer_id,
                result: Err(Error {
                    inner: InnerError::AttemptsExceeded(MAX_NUMBER_OF_UPGRADE_ATTEMPTS),
                }),
            })]);
        }
    }

    fn on_connection_closed(
        &mut self,
        ConnectionClosed {
            peer_id,
            connection_id,
            endpoint: connected_point,
            ..
        }: ConnectionClosed,
    ) {
        if !connected_point.is_relayed() {
            let connections = self
                .direct_connections
                .get_mut(&peer_id)
                .expect("Peer of direct connection to be tracked.");
            connections
                .remove(&connection_id)
                .then_some(())
                .expect("Direct connection to be tracked.");
            if connections.is_empty() {
                self.direct_connections.remove(&peer_id);
            }
        }
    }
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler = Either<handler::relayed::Handler, dummy::ConnectionHandler>;
    type ToSwarm = Event;

    fn handle_established_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        local_addr: &Multiaddr,
        remote_addr: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        if is_relayed(local_addr) {
            let connected_point = ConnectedPoint::Listener {
                local_addr: local_addr.clone(),
                send_back_addr: remote_addr.clone(),
            };
            let mut handler =
                handler::relayed::Handler::new(connected_point, self.observed_addresses());
            handler.on_behaviour_event(handler::relayed::Command::Connect);

            return Ok(Either::Left(handler)); // TODO: We could make two `handler::relayed::Handler` here, one inbound one outbound.
        }
        self.direct_connections
            .entry(peer)
            .or_default()
            .insert(connection_id);

        assert!(
            !self
                .direct_to_relayed_connections
                .contains_key(&connection_id),
            "state mismatch"
        );

        Ok(Either::Right(dummy::ConnectionHandler))
    }

    fn handle_established_outbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        addr: &Multiaddr,
        role_override: Endpoint,
        port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        if is_relayed(addr) {
            return Ok(Either::Left(handler::relayed::Handler::new(
                ConnectedPoint::Dialer {
                    address: addr.clone(),
                    role_override,
                    port_use,
                },
                self.observed_addresses(),
            ))); // TODO: We could make two `handler::relayed::Handler` here, one inbound one outbound.
        }

        self.direct_connections
            .entry(peer)
            .or_default()
            .insert(connection_id);

        // Whether this is a connection requested by this behaviour.
        if let Some(&relayed_connection_id) = self.direct_to_relayed_connections.get(&connection_id)
        {
            if role_override == Endpoint::Listener {
                assert!(
                    self.outgoing_direct_connection_attempts
                        .remove(&(relayed_connection_id, peer))
                        .is_some(),
                    "state mismatch"
                );
            }

            self.queued_events.extend([ToSwarm::GenerateEvent(Event {
                remote_peer_id: peer,
                result: Ok(connection_id),
            })]);
        }
        Ok(Either::Right(dummy::ConnectionHandler))
    }

    fn on_connection_handler_event(
        &mut self,
        event_source: PeerId,
        connection_id: ConnectionId,
        handler_event: THandlerOutEvent<Self>,
    ) {
        let relayed_connection_id = match handler_event.as_ref() {
            Either::Left(_) => connection_id,
            Either::Right(_) => match self.direct_to_relayed_connections.get(&connection_id) {
                None => {
                    // If the connection ID is unknown to us, it means we didn't create it so ignore any event coming from it.
                    return;
                }
                Some(relayed_connection_id) => *relayed_connection_id,
            },
        };

        match handler_event {
            Either::Left(handler::relayed::Event::InboundConnectNegotiated { remote_addrs }) => {
                tracing::debug!(target=%event_source, addresses=?remote_addrs, "Attempting to hole-punch as dialer");

                let opts = DialOpts::peer_id(event_source)
                    .addresses(remote_addrs)
                    .condition(dial_opts::PeerCondition::Always)
                    .build();

                let maybe_direct_connection_id = opts.connection_id();

                self.direct_to_relayed_connections
                    .insert(maybe_direct_connection_id, relayed_connection_id);
                self.queued_events.push_back(ToSwarm::Dial { opts });
            }
            Either::Left(handler::relayed::Event::InboundConnectFailed { error }) => {
                self.queued_events.push_back(ToSwarm::GenerateEvent(Event {
                    remote_peer_id: event_source,
                    result: Err(Error {
                        inner: InnerError::InboundError(error),
                    }),
                }));
            }
            Either::Left(handler::relayed::Event::OutboundConnectFailed { error }) => {
                self.queued_events.push_back(ToSwarm::GenerateEvent(Event {
                    remote_peer_id: event_source,
                    result: Err(Error {
                        inner: InnerError::OutboundError(error),
                    }),
                }));

                // Maybe treat these as transient and retry?
            }
            Either::Left(handler::relayed::Event::OutboundConnectNegotiated { remote_addrs }) => {
                tracing::debug!(target=%event_source, addresses=?remote_addrs, "Attempting to hole-punch as listener");

                let opts = DialOpts::peer_id(event_source)
                    .condition(dial_opts::PeerCondition::Always)
                    .addresses(remote_addrs)
                    .override_role()
                    .build();

                let maybe_direct_connection_id = opts.connection_id();

                self.direct_to_relayed_connections
                    .insert(maybe_direct_connection_id, relayed_connection_id);
                *self
                    .outgoing_direct_connection_attempts
                    .entry((relayed_connection_id, event_source))
                    .or_default() += 1;
                self.queued_events.push_back(ToSwarm::Dial { opts });
            }
            Either::Right(never) => void::unreachable(never),
        };
    }

    #[tracing::instrument(level = "trace", name = "NetworkBehaviour::poll", skip(self))]
    fn poll(&mut self, _: &mut Context<'_>) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        if let Some(event) = self.queued_events.pop_front() {
            return Poll::Ready(event);
        }

        Poll::Pending
    }

    fn on_swarm_event(&mut self, event: FromSwarm) {
        match event {
            FromSwarm::ConnectionClosed(connection_closed) => {
                self.on_connection_closed(connection_closed)
            }
            FromSwarm::DialFailure(dial_failure) => self.on_dial_failure(dial_failure),
            FromSwarm::NewExternalAddrCandidate(NewExternalAddrCandidate { addr }) => {
                self.address_candidates.add(addr.clone());
            }
            // OpenHydra patch: a listen address of ours went away (we left the network it was on):
            // forget the candidates observed on that network, so we stop offering peers addresses
            // nothing listens on any more. Our listen addresses are counted to tell.
            FromSwarm::NewListenAddr(NewListenAddr { addr, .. }) => {
                self.address_candidates.listening_on(addr);
            }
            FromSwarm::ExpiredListenAddr(ExpiredListenAddr { addr, .. }) => {
                self.address_candidates.forget_network_of(addr);
            }
            _ => {}
        }
    }
}

/// Stores our address candidates.
///
/// We use an [`LruCache`] to favor addresses that are reported more often.
/// When attempting a hole-punch, we will try more frequent addresses first.
/// Most of these addresses will come from observations by other nodes (via e.g. the identify protocol).
/// More common observations mean a more likely stable port-mapping and thus a higher chance of a successful hole-punch.
struct Candidates {
    inner: LruCache<Multiaddr, ()>,
    me: PeerId,
    /// OpenHydra patch: our listen addresses per IP (see `forget_network_of`).
    ours: HashMap<std::net::IpAddr, usize>,
}

impl Candidates {
    fn new(me: PeerId) -> Self {
        Self {
            inner: LruCache::new(NonZeroUsize::new(20).expect("20 > 0")),
            me,
            ours: HashMap::new(),
        }
    }

    fn add(&mut self, mut address: Multiaddr) {
        if is_relayed(&address) {
            return;
        }

        if address.iter().last() != Some(Protocol::P2p(self.me)) {
            address.push(Protocol::P2p(self.me));
        }

        self.inner.push(address, ());
    }

    fn iter(&self) -> impl Iterator<Item = &Multiaddr> {
        self.inner.iter().map(|(a, _)| a)
    }

    /// OpenHydra patch: we listen on `addr` (one of our interface addresses).
    fn listening_on(&mut self, addr: &Multiaddr) {
        if let Some(ip) = first_ip(addr) {
            if !is_relayed(addr) {
                *self.ours.entry(ip).or_default() += 1;
            }
        }
    }

    /// OpenHydra patch: our listen address `expired` went away. Loopback and relayed addresses
    /// aside, the candidates on its IP go (without NAT, peers saw us at that very IP). Once the last
    /// listen address on an IPv4 interface address goes and no other real IPv4 address of ours is
    /// left, every IPv4 candidate goes too: behind NAT peers saw us at the NAT's public address,
    /// which can't be told apart from another interface's, and the interface's IPv4 address going
    /// away is how leaving a network shows. Virtual interfaces (container bridges, VPNs, VMs; see
    /// `is_virtual_v4`) neither trigger that nor count as left. Connections on the new network
    /// refill the cache.
    fn forget_network_of(&mut self, expired: &Multiaddr) {
        let Some(gone) = first_ip(expired) else {
            return;
        };
        if gone.is_loopback() || is_relayed(expired) {
            return;
        }
        let still = match self.ours.get_mut(&gone) {
            Some(n) if *n > 1 => {
                *n -= 1;
                true
            }
            _ => {
                self.ours.remove(&gone);
                false
            }
        };
        let real_v4 = |ip: &std::net::IpAddr| matches!(ip, std::net::IpAddr::V4(v4) if !v4.is_loopback() && !is_virtual_v4(*v4));
        let wipe_v4 = !still && real_v4(&gone) && !self.ours.keys().any(real_v4);
        let stale: Vec<Multiaddr> = self
            .inner
            .iter()
            .filter(|(a, _)| match first_ip(a) {
                Some(ip) => (ip == gone && !still) || (wipe_v4 && ip.is_ipv4()),
                None => false,
            })
            .map(|(a, _)| a.clone())
            .collect();
        for a in stale {
            self.inner.pop(&a);
        }
    }
}

/// OpenHydra patch: IPv4 ranges virtual interfaces use — container bridges (172.16.0.0/12),
/// Tailscale and carrier NAT (100.64.0.0/10), libvirt (192.168.122.0/24), macOS virtualization
/// (192.168.64.0/24) — and link-local.
fn is_virtual_v4(ip: std::net::Ipv4Addr) -> bool {
    let o = ip.octets();
    ip.is_link_local()
        || (o[0] == 172 && (16..32).contains(&o[1]))
        || (o[0] == 100 && (64..128).contains(&o[1]))
        || (o[0] == 192 && o[1] == 168 && (o[2] == 122 || o[2] == 64))
}

/// OpenHydra patch: the IP an address starts with, if it does.
fn first_ip(addr: &Multiaddr) -> Option<std::net::IpAddr> {
    match addr.iter().next()? {
        Protocol::Ip4(ip) => Some(ip.into()),
        Protocol::Ip6(ip) => Some(ip.into()),
        _ => None,
    }
}

fn is_relayed(addr: &Multiaddr) -> bool {
    addr.iter().any(|p| p == Protocol::P2pCircuit)
}

#[cfg(test)]
mod openhydra_patch_tests {
    use super::*;

    fn addrs(c: &Candidates) -> Vec<String> {
        let mut v: Vec<String> = c.iter().map(|a| a.to_string()).collect();
        v.sort();
        v
    }

    #[test]
    fn leaving_a_network_forgets_the_candidates_seen_on_it() {
        let me: PeerId = "12D3KooWAbLYyX4tHc1TZSBj5hT8nXMYZ1oZMJ5htJGkfBqn2mmr".parse().unwrap();
        let ma = |s: &str| -> Multiaddr { s.parse().unwrap() };
        let mut c = Candidates::new(me);
        // Our listen addresses: Wi-Fi v4 (TCP + QUIC), two v6, a Docker bridge, loopback.
        for a in [
            "/ip4/192.168.0.3/tcp/4987",
            "/ip4/192.168.0.3/udp/4987/quic-v1",
            "/ip6/2406:7400:56:9e97::4/udp/4987/quic-v1",
            "/ip6/2406:7400:56:9e97::5/tcp/4987",
            "/ip4/172.17.0.1/tcp/4987",
            "/ip4/127.0.0.1/tcp/4987",
        ] {
            c.listening_on(&ma(a));
        }
        for a in [
            "/ip4/49.207.231.148/tcp/18889",
            "/ip4/49.207.231.148/udp/4987/quic-v1",
            "/ip6/2406:7400:56:9e97::4/udp/4987/quic-v1",
            "/ip6/2406:7400:56:9e97::5/tcp/4987",
        ] {
            c.add(ma(a));
        }
        // Loopback, circuit and container-bridge addresses going away change nothing.
        c.forget_network_of(&ma("/ip4/127.0.0.1/tcp/4987"));
        c.forget_network_of(&ma("/ip6/2406:7400:56:9e97::4/udp/4001/quic-v1/p2p-circuit"));
        c.forget_network_of(&ma("/ip4/172.17.0.1/tcp/4987"));
        assert_eq!(c.iter().count(), 4);
        // An IPv6 address going away takes the candidates on that IP only.
        c.forget_network_of(&ma("/ip6/2406:7400:56:9e97::4/udp/4987/quic-v1"));
        assert_eq!(
            addrs(&c),
            vec![
                format!("/ip4/49.207.231.148/tcp/18889/p2p/{me}"),
                format!("/ip4/49.207.231.148/udp/4987/quic-v1/p2p/{me}"),
                format!("/ip6/2406:7400:56:9e97::5/tcp/4987/p2p/{me}"),
            ]
        );
        // The Wi-Fi's TCP listen address goes but its QUIC one is still up: nothing yet.
        c.forget_network_of(&ma("/ip4/192.168.0.3/tcp/4987"));
        assert_eq!(c.iter().count(), 3);
        // Its last one goes, and only virtual IPv4 is left: every IPv4 candidate goes.
        c.forget_network_of(&ma("/ip4/192.168.0.3/udp/4987/quic-v1"));
        assert_eq!(addrs(&c), vec![format!("/ip6/2406:7400:56:9e97::5/tcp/4987/p2p/{me}")]);
        // New observations come back in as before.
        c.add(ma("/ip4/106.192.224.140/tcp/4995"));
        assert_eq!(c.iter().count(), 2);
    }

    #[test]
    fn another_real_ipv4_interface_still_up_keeps_the_ipv4_candidates() {
        let me: PeerId = "12D3KooWAbLYyX4tHc1TZSBj5hT8nXMYZ1oZMJ5htJGkfBqn2mmr".parse().unwrap();
        let ma = |s: &str| -> Multiaddr { s.parse().unwrap() };
        let mut c = Candidates::new(me);
        c.listening_on(&ma("/ip4/192.168.0.3/tcp/4987"));
        c.listening_on(&ma("/ip4/10.0.0.7/tcp/4987"));
        c.add(ma("/ip4/49.207.231.148/tcp/18889"));
        // Wi-Fi down, Ethernet up: kept.
        c.forget_network_of(&ma("/ip4/192.168.0.3/tcp/4987"));
        assert_eq!(c.iter().count(), 1);
        // Ethernet down too: gone.
        c.forget_network_of(&ma("/ip4/10.0.0.7/tcp/4987"));
        assert_eq!(c.iter().count(), 0);
    }

    #[test]
    fn the_behaviour_forgets_them_when_the_swarm_reports_the_address_expired() {
        let me: PeerId = "12D3KooWAbLYyX4tHc1TZSBj5hT8nXMYZ1oZMJ5htJGkfBqn2mmr".parse().unwrap();
        let mut b = Behaviour::new(me);
        let seen: Multiaddr = "/ip4/49.207.231.148/tcp/18889".parse().unwrap();
        b.on_swarm_event(FromSwarm::NewExternalAddrCandidate(NewExternalAddrCandidate { addr: &seen }));
        assert_eq!(b.observed_addresses().len(), 1);
        let gone: Multiaddr = "/ip4/192.168.0.3/tcp/4987".parse().unwrap();
        b.on_swarm_event(FromSwarm::NewListenAddr(NewListenAddr {
            listener_id: libp2p_core::transport::ListenerId::next(),
            addr: &gone,
        }));
        b.on_swarm_event(FromSwarm::ExpiredListenAddr(ExpiredListenAddr {
            listener_id: libp2p_core::transport::ListenerId::next(),
            addr: &gone,
        }));
        assert!(b.observed_addresses().is_empty());
    }
}
