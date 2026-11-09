//! Addresses we keep for our own dials to a peer (a reconnect after a drop, a dial-back by peer id),
//! and never hand out: they aren't in the routing table, so they never appear in DHT answers, and a
//! private LAN address is fine here.
//!
//! The swarm has no store of its own: `Swarm::add_peer_address` only broadcasts
//! `FromSwarm::NewExternalAddrOfPeer`, and no other behaviour of ours keeps it (identify's cache is
//! off), so addresses passed to it went nowhere (review 2026-10-29). This behaviour is that store.
//! It is filled only by [`Behaviour::insert`], called by the event loop with addresses it has
//! checked. It ignores `NewExternalAddrOfPeer`: Kademlia emits that for addresses from its table, so
//! taking it would let other peers' claims in by a side door.

use std::collections::HashMap;
use std::task::{Context, Poll};

use libp2p::core::transport::PortUse;
use libp2p::core::Endpoint;
use libp2p::swarm::{
    dummy, ConnectionDenied, ConnectionId, FromSwarm, NetworkBehaviour, THandler, THandlerInEvent, THandlerOutEvent,
    ToSwarm,
};
use libp2p::{Multiaddr, PeerId};

/// Addresses kept per peer, at most; past it, the least recently added goes.
pub const MAX_ADDRS_PER_PEER: usize = 8;
/// Peers kept, at most; past it, the least recently used goes.
pub const MAX_PEERS: usize = 4096;

#[derive(Debug, Default)]
struct Entry {
    addrs: Vec<Multiaddr>,
    used: u64,
}

/// The behaviour (see the module docs).
#[derive(Debug, Default)]
pub struct Behaviour {
    peers: HashMap<PeerId, Entry>,
    clock: u64,
}

impl Behaviour {
    pub fn new() -> Self {
        Self::default()
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Keep `addr` for dials to `peer`. Returns whether it was new.
    pub fn insert(&mut self, peer: PeerId, addr: Multiaddr) -> bool {
        let addr = without_peer_id(addr, &peer);
        let now = self.tick();
        if !self.peers.contains_key(&peer) && self.peers.len() >= MAX_PEERS {
            if let Some(oldest) = self.peers.iter().min_by_key(|(_, e)| e.used).map(|(p, _)| *p) {
                self.peers.remove(&oldest);
            }
        }
        let entry = self.peers.entry(peer).or_default();
        entry.used = now;
        if entry.addrs.contains(&addr) {
            return false;
        }
        if entry.addrs.len() >= MAX_ADDRS_PER_PEER {
            entry.addrs.remove(0);
        }
        entry.addrs.push(addr);
        true
    }

    /// Forget one address of `peer` (the peer goes when it has none left).
    pub fn remove(&mut self, peer: &PeerId, addr: &Multiaddr) {
        let addr = without_peer_id(addr.clone(), peer);
        if let Some(entry) = self.peers.get_mut(peer) {
            entry.addrs.retain(|a| *a != addr);
            if entry.addrs.is_empty() {
                self.peers.remove(peer);
            }
        }
    }

    /// Forget every address of `peer`.
    pub fn remove_peer(&mut self, peer: &PeerId) {
        self.peers.remove(peer);
    }

    /// The addresses kept for `peer`, oldest first.
    pub fn addresses(&self, peer: &PeerId) -> &[Multiaddr] {
        self.peers.get(peer).map(|e| e.addrs.as_slice()).unwrap_or(&[])
    }

    /// How many peers have addresses kept.
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }
}

/// `addr` without a trailing `/p2p/<peer>` (identify gives addresses without it, mDNS and Kademlia
/// with it), so both forms are the same entry.
fn without_peer_id(mut addr: Multiaddr, peer: &PeerId) -> Multiaddr {
    if let Some(libp2p::multiaddr::Protocol::P2p(p)) = addr.iter().last() {
        if p == *peer {
            addr.pop();
        }
    }
    addr
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler = dummy::ConnectionHandler;
    type ToSwarm = std::convert::Infallible;

    fn handle_pending_outbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        maybe_peer: Option<PeerId>,
        _addresses: &[Multiaddr],
        _effective_role: Endpoint,
    ) -> Result<Vec<Multiaddr>, ConnectionDenied> {
        let Some(peer) = maybe_peer else { return Ok(Vec::new()) };
        let now = self.tick();
        Ok(match self.peers.get_mut(&peer) {
            Some(entry) => {
                entry.used = now;
                entry.addrs.clone()
            }
            None => Vec::new(),
        })
    }

    fn handle_established_inbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        _peer: PeerId,
        _local: &Multiaddr,
        _remote: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
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

    // Deliberately ignores `NewExternalAddrOfPeer` (see the module docs).
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
    use futures::StreamExt;
    use libp2p::core::transport::{MemoryTransport, Transport};
    use libp2p::core::upgrade::Version;
    use libp2p::swarm::dial_opts::DialOpts;
    use libp2p::swarm::{NewExternalAddrOfPeer, Swarm, SwarmEvent};
    use libp2p::{identity, noise, yamux};
    use std::time::Duration;

    fn addr(n: u16) -> Multiaddr {
        format!("/ip4/192.168.1.{}/tcp/4001", n % 250 + 1).parse().unwrap()
    }

    #[test]
    fn keeps_addresses_and_supplies_them_to_a_dial_by_peer_id() {
        let mut book = Behaviour::new();
        let p = PeerId::random();
        assert!(book.insert(p, addr(1)));
        assert!(!book.insert(p, addr(1)), "a repeat isn't new");
        assert!(book.insert(p, addr(2)));
        let got = book
            .handle_pending_outbound_connection(ConnectionId::new_unchecked(1), Some(p), &[], Endpoint::Dialer)
            .unwrap();
        assert_eq!(got, vec![addr(1), addr(2)]);
        let none = book
            .handle_pending_outbound_connection(ConnectionId::new_unchecked(2), Some(PeerId::random()), &[], Endpoint::Dialer)
            .unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn per_peer_and_total_caps_hold() {
        let mut book = Behaviour::new();
        let p = PeerId::random();
        for n in 0..(MAX_ADDRS_PER_PEER as u16 + 3) {
            book.insert(p, addr(n));
        }
        assert_eq!(book.addresses(&p).len(), MAX_ADDRS_PER_PEER);
        assert_eq!(book.addresses(&p)[0], addr(3), "the oldest went first");

        let mut book = Behaviour::new();
        let first = PeerId::random();
        book.insert(first, addr(1));
        let second = PeerId::random();
        book.insert(second, addr(1));
        // `first` is used by a dial, so `second` is now the least recently used.
        book.handle_pending_outbound_connection(ConnectionId::new_unchecked(1), Some(first), &[], Endpoint::Dialer)
            .unwrap();
        for _ in 0..(MAX_PEERS - 2) {
            book.insert(PeerId::random(), addr(1));
        }
        assert_eq!(book.len(), MAX_PEERS);
        book.insert(PeerId::random(), addr(1));
        assert_eq!(book.len(), MAX_PEERS);
        assert!(!book.addresses(&first).is_empty(), "the recently used peer stays");
        assert!(book.addresses(&second).is_empty(), "the least recently used peer went");
    }

    #[test]
    fn an_address_with_or_without_the_peer_id_is_one_entry() {
        let mut book = Behaviour::new();
        let p = PeerId::random();
        book.insert(p, addr(1));
        assert!(!book.insert(p, addr(1).with_p2p(p).unwrap()), "same address");
        book.remove(&p, &addr(1).with_p2p(p).unwrap());
        assert!(book.is_empty());
    }

    #[test]
    fn remove_forgets_one_address_then_the_peer() {
        let mut book = Behaviour::new();
        let p = PeerId::random();
        book.insert(p, addr(1));
        book.insert(p, addr(2));
        book.remove(&p, &addr(1));
        assert_eq!(book.addresses(&p), &[addr(2)]);
        book.remove(&p, &addr(2));
        assert!(book.is_empty());
        book.insert(p, addr(3));
        book.remove_peer(&p);
        assert!(book.is_empty());
    }

    #[test]
    fn other_behaviours_peer_address_events_are_ignored() {
        let mut book = Behaviour::new();
        let p = PeerId::random();
        let a = addr(9);
        book.on_swarm_event(FromSwarm::NewExternalAddrOfPeer(NewExternalAddrOfPeer { peer_id: p, addr: &a }));
        assert!(book.addresses(&p).is_empty());
    }

    fn node() -> Swarm<Behaviour> {
        let key = identity::Keypair::generate_ed25519();
        let transport = MemoryTransport::default()
            .upgrade(Version::V1)
            .authenticate(noise::Config::new(&key).unwrap())
            .multiplex(yamux::Config::default())
            .boxed();
        let cfg = libp2p::swarm::Config::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(60));
        Swarm::new(transport, Behaviour::new(), PeerId::from(key.public()), cfg)
    }

    #[tokio::test]
    async fn a_dial_by_peer_id_alone_reaches_a_peer_through_its_kept_address() {
        let mut a = node();
        let mut b = node();
        let b_id = *b.local_peer_id();
        b.listen_on("/memory/0".parse().unwrap()).unwrap();
        let b_addr = loop {
            if let SwarmEvent::NewListenAddr { address, .. } = b.select_next_some().await {
                break address;
            }
        };
        a.behaviour_mut().insert(b_id, b_addr);
        // No address in the dial itself: only the book can supply one.
        a.dial(DialOpts::peer_id(b_id).build()).unwrap();
        let connected = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                tokio::select! {
                    e = a.select_next_some() => if let SwarmEvent::ConnectionEstablished { peer_id, .. } = e {
                        if peer_id == b_id { break true; }
                    },
                    _ = b.select_next_some() => {}
                }
            }
        })
        .await
        .unwrap_or(false);
        assert!(connected, "the dial by peer id didn't use the kept address");
    }
}
