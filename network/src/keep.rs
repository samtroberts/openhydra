// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Which idle connections stay open (#16).
//!
//! A connection closes once nothing has used it for [`IDLE_TIMEOUT`], unless it is to a peer this
//! node has really used: sent a request to or served one for, or exchanged a push stream with, in
//! the last [`APP_KEEP`] ([`AppPeers::note`]), or one it always keeps (its bootstraps and relays).
//! Gossip keeps its own mesh peers' connections, and a relay reservation its relay's. What closes
//! are the connections only the DHT used: about 27 per provider, each refreshed by DHT traffic
//! faster than the old 300 s timeout closed it, costly on a home router or a laptop.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use libp2p::core::transport::PortUse;
use libp2p::core::upgrade::DeniedUpgrade;
use libp2p::core::Endpoint;
use libp2p::swarm::handler::ConnectionEvent;
use libp2p::swarm::{
    ConnectionDenied, ConnectionHandler, ConnectionHandlerEvent, ConnectionId, FromSwarm, NetworkBehaviour,
    SubstreamProtocol, THandler, THandlerInEvent, THandlerOutEvent, ToSwarm,
};
use libp2p::{Multiaddr, PeerId};

/// How long an unused connection stays open (it was 300 s for every connection).
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(60);
/// How long after its last real use a peer's connections are kept open, idle or not: a gateway's
/// next request to a provider it used recently then needs no new connection.
pub const APP_KEEP: Duration = Duration::from_secs(10 * 60);
/// Peers remembered as used at most; past it, those not used within [`APP_KEEP`] are forgotten.
const MAX_REMEMBERED: usize = 4096;

#[derive(Default)]
struct Inner {
    used: HashMap<PeerId, Instant>,
    always: HashSet<PeerId>,
}

/// The peers whose connections are kept open while idle, shared between the swarm's connections,
/// its event loop and the push streams.
#[derive(Clone, Default)]
pub struct AppPeers(Arc<Mutex<Inner>>);

impl AppPeers {
    /// `peer` was really used just now (a request, a reply, a push stream).
    pub fn note(&self, peer: PeerId) {
        self.note_at(peer, Instant::now());
    }

    fn note_at(&self, peer: PeerId, now: Instant) {
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if inner.used.len() >= MAX_REMEMBERED && !inner.used.contains_key(&peer) {
            inner.used.retain(|_, at| now.saturating_duration_since(*at) < APP_KEEP);
        }
        if inner.used.len() < MAX_REMEMBERED || inner.used.contains_key(&peer) {
            inner.used.insert(peer, now);
        }
    }

    /// Always keep these peers' connections (bootstraps, relays).
    pub fn keep_always(&self, peers: impl IntoIterator<Item = PeerId>) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).always.extend(peers);
    }

    /// Whether `peer`'s connections are kept open while idle: kept always, or used within
    /// [`APP_KEEP`].
    pub fn wanted(&self, peer: &PeerId) -> bool {
        self.wanted_at(peer, Instant::now())
    }

    fn wanted_at(&self, peer: &PeerId, now: Instant) -> bool {
        let inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        inner.always.contains(peer) || inner.used.get(peer).is_some_and(|at| now.saturating_duration_since(*at) < APP_KEEP)
    }
}

/// Keeps a wanted peer's connections open ([`AppPeers::wanted`]); speaks no protocol.
pub struct Behaviour {
    peers: AppPeers,
}

impl Behaviour {
    pub fn new(peers: AppPeers) -> Self {
        Self { peers }
    }

    /// The peers it keeps (to note uses in).
    pub fn peers(&self) -> AppPeers {
        self.peers.clone()
    }
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler = Handler;
    type ToSwarm = std::convert::Infallible;

    fn handle_established_inbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        peer: PeerId,
        _local: &Multiaddr,
        _remote: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(Handler { peer, peers: self.peers.clone() })
    }

    fn handle_established_outbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        peer: PeerId,
        _addr: &Multiaddr,
        _role_override: Endpoint,
        _port_use: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(Handler { peer, peers: self.peers.clone() })
    }

    fn on_swarm_event(&mut self, _event: FromSwarm) {}

    fn on_connection_handler_event(&mut self, _peer: PeerId, _id: ConnectionId, event: THandlerOutEvent<Self>) {
        match event {}
    }

    fn poll(&mut self, _cx: &mut Context<'_>) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        Poll::Pending
    }
}

/// One connection's part of [`Behaviour`]: no streams, only whether to keep the connection.
pub struct Handler {
    peer: PeerId,
    peers: AppPeers,
}

impl ConnectionHandler for Handler {
    type FromBehaviour = std::convert::Infallible;
    type ToBehaviour = std::convert::Infallible;
    type InboundProtocol = DeniedUpgrade;
    type OutboundProtocol = DeniedUpgrade;
    type InboundOpenInfo = ();
    type OutboundOpenInfo = ();

    fn listen_protocol(&self) -> SubstreamProtocol<Self::InboundProtocol, Self::InboundOpenInfo> {
        SubstreamProtocol::new(DeniedUpgrade, ())
    }

    fn connection_keep_alive(&self) -> bool {
        self.peers.wanted(&self.peer)
    }

    fn poll(
        &mut self,
        _cx: &mut Context<'_>,
    ) -> Poll<ConnectionHandlerEvent<Self::OutboundProtocol, Self::OutboundOpenInfo, Self::ToBehaviour>> {
        Poll::Pending
    }

    fn on_behaviour_event(&mut self, event: Self::FromBehaviour) {
        match event {}
    }

    fn on_connection_event(
        &mut self,
        _event: ConnectionEvent<Self::InboundProtocol, Self::OutboundProtocol, Self::InboundOpenInfo, Self::OutboundOpenInfo>,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_peer_is_kept_for_a_while_after_its_last_use_and_always_kept_ones_for_good() {
        let peers = AppPeers::default();
        let (used, other, boot) = (PeerId::random(), PeerId::random(), PeerId::random());
        let t0 = Instant::now();
        assert!(!peers.wanted_at(&used, t0));
        peers.note_at(used, t0);
        assert!(peers.wanted_at(&used, t0 + APP_KEEP - Duration::from_secs(1)));
        assert!(!peers.wanted_at(&used, t0 + APP_KEEP), "unused for APP_KEEP");
        assert!(!peers.wanted_at(&other, t0));
        peers.keep_always([boot]);
        assert!(peers.wanted_at(&boot, t0 + APP_KEEP * 100));
    }

    #[test]
    fn remembered_peers_stay_bounded() {
        let peers = AppPeers::default();
        let t0 = Instant::now();
        for _ in 0..MAX_REMEMBERED {
            peers.note_at(PeerId::random(), t0);
        }
        // Full of recent ones: a new peer isn't remembered, and nothing grows.
        let late = PeerId::random();
        peers.note_at(late, t0 + Duration::from_secs(1));
        assert!(!peers.wanted_at(&late, t0 + Duration::from_secs(1)));
        // Once those are stale, a new one takes their place.
        peers.note_at(late, t0 + APP_KEEP);
        assert!(peers.wanted_at(&late, t0 + APP_KEEP));
        assert_eq!(peers.0.lock().unwrap().used.len(), 1);
    }
}
