use std::{
    collections::{hash_map::Entry, HashMap},
    io,
    sync::{Arc, Mutex, MutexGuard},
};

use futures::channel::mpsc;
use libp2p_identity::PeerId;
use libp2p_swarm::{ConnectionId, Stream, StreamProtocol};
use rand::seq::IteratorRandom as _;

use crate::{handler::NewStream, AlreadyRegistered, IncomingStreams};

/// OpenHydra patch: how many negotiated inbound streams of one protocol may wait for the acceptor.
/// Upstream's channel holds one, so a burst of streams that arrives between two polls of
/// [`IncomingStreams`] was dropped (each a peer's open that silently went nowhere).
const INBOUND_BUFFER: usize = 64;

/// OpenHydra patch: what is known of one connection.
struct Conn {
    peer: PeerId,
    /// It runs through a relay circuit (on either side).
    relayed: bool,
    /// A stream has been negotiated on it (either way): it is known to carry streams.
    proven: bool,
}

pub(crate) struct Shared {
    /// Tracks the supported inbound protocols created via [`Control::accept`](crate::Control::accept).
    ///
    /// For each [`StreamProtocol`], we hold the [`mpsc::Sender`] corresponding to the [`mpsc::Receiver`] in [`IncomingStreams`].
    supported_inbound_protocols: HashMap<StreamProtocol, mpsc::Sender<(PeerId, Stream)>>,

    connections: HashMap<ConnectionId, Conn>,
    senders: HashMap<ConnectionId, mpsc::Sender<NewStream>>,

    /// Tracks channel pairs for a peer whilst we are dialing them.
    pending_channels: HashMap<PeerId, (mpsc::Sender<NewStream>, mpsc::Receiver<NewStream>)>,

    /// Sender for peers we want to dial.
    ///
    /// We manage this through a channel to avoid locks as part of [`NetworkBehaviour::poll`](libp2p_swarm::NetworkBehaviour::poll).
    dial_sender: mpsc::Sender<PeerId>,
}

impl Shared {
    pub(crate) fn lock(shared: &Arc<Mutex<Shared>>) -> MutexGuard<'_, Shared> {
        shared.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Shared {
    pub(crate) fn new(dial_sender: mpsc::Sender<PeerId>) -> Self {
        Self {
            dial_sender,
            connections: Default::default(),
            senders: Default::default(),
            pending_channels: Default::default(),
            supported_inbound_protocols: Default::default(),
        }
    }

    pub(crate) fn accept(
        &mut self,
        protocol: StreamProtocol,
    ) -> Result<IncomingStreams, AlreadyRegistered> {
        // Backported from upstream 0.4.0-alpha (PR 5999): a protocol whose `IncomingStreams` was
        // dropped can be accepted again.
        self.supported_inbound_protocols
            .retain(|_, sender| !sender.is_closed());

        if self.supported_inbound_protocols.contains_key(&protocol) {
            return Err(AlreadyRegistered);
        }

        let (sender, receiver) = mpsc::channel(INBOUND_BUFFER);
        self.supported_inbound_protocols
            .insert(protocol.clone(), sender);

        Ok(IncomingStreams::new(receiver))
    }

    /// Lists the protocols for which we have an active [`IncomingStreams`] instance.
    pub(crate) fn supported_inbound_protocols(&mut self) -> Vec<StreamProtocol> {
        self.supported_inbound_protocols
            .retain(|_, sender| !sender.is_closed());

        self.supported_inbound_protocols.keys().cloned().collect()
    }

    pub(crate) fn on_inbound_stream(
        &mut self,
        remote: PeerId,
        connection: ConnectionId,
        stream: Stream,
        protocol: StreamProtocol,
    ) {
        self.mark_proven(connection);
        match self.supported_inbound_protocols.entry(protocol.clone()) {
            Entry::Occupied(mut entry) => match entry.get_mut().try_send((remote, stream)) {
                Ok(()) => {}
                Err(e) if e.is_full() => {
                    tracing::debug!(%protocol, "Channel is full, dropping inbound stream");
                }
                Err(e) if e.is_disconnected() => {
                    tracing::debug!(%protocol, "Channel is gone, dropping inbound stream");
                    entry.remove();
                }
                _ => unreachable!(),
            },
            Entry::Vacant(_) => {
                tracing::debug!(%protocol, "channel is gone, dropping inbound stream");
            }
        }
    }

    /// OpenHydra patch: a stream was negotiated on `connection`.
    pub(crate) fn mark_proven(&mut self, connection: ConnectionId) {
        if let Some(c) = self.connections.get_mut(&connection) {
            c.proven = true;
        }
    }

    pub(crate) fn on_connection_established(&mut self, conn: ConnectionId, peer: PeerId, relayed: bool) {
        self.connections.insert(conn, Conn { peer, relayed, proven: false });
    }

    pub(crate) fn on_connection_closed(&mut self, conn: ConnectionId) {
        self.connections.remove(&conn);
        // OpenHydra patch: upstream kept a closed connection's sender for the life of the
        // behaviour (one entry per connection ever made).
        self.senders.remove(&conn);
    }

    pub(crate) fn on_dial_failure(&mut self, peer: PeerId, reason: String) {
        let Some((_, mut receiver)) = self.pending_channels.remove(&peer) else {
            return;
        };

        while let Ok(new_stream) = receiver.try_recv() {
            let _ = new_stream
                .sender
                .send(Err(crate::OpenStreamError::Io(io::Error::new(
                    io::ErrorKind::NotConnected,
                    reason.clone(),
                ))));
        }
    }

    /// The connection to `peer` a new stream goes on (OpenHydra patch): one of its direct
    /// connections that has carried a stream, when it has one; otherwise any of its connections,
    /// at random as upstream (a direct connection that has carried nothing yet may be silently
    /// dead, as right after a hole punch, while a relay circuit beside it works).
    fn pick_connection(&self, peer: &PeerId) -> Option<ConnectionId> {
        let mut rng = rand::thread_rng();
        let theirs = || self.connections.iter().filter(|(_, c)| c.peer == *peer);
        theirs()
            .filter(|(_, c)| !c.relayed && c.proven)
            .map(|(id, _)| *id)
            .choose(&mut rng)
            .or_else(|| theirs().map(|(id, _)| *id).choose(&mut rng))
    }

    /// OpenHydra patch: the sender for a stream to `peer` on a connection it already has, or
    /// `None` when it has none (nothing is dialled).
    pub(crate) fn connected_sender(&mut self, peer: PeerId) -> Option<mpsc::Sender<NewStream>> {
        let conn = self.pick_connection(&peer)?;
        self.senders.get(&conn).cloned()
    }

    pub(crate) fn sender(&mut self, peer: PeerId) -> mpsc::Sender<NewStream> {
        match self.connected_sender(peer) {
            Some(sender) => {
                tracing::debug!("Returning sender to existing connection");

                sender
            }
            None => {
                tracing::debug!(%peer, "Not connected to peer, initiating dial");

                let (sender, _) = self
                    .pending_channels
                    .entry(peer)
                    .or_insert_with(|| mpsc::channel(0));

                let _ = self.dial_sender.try_send(peer);

                sender.clone()
            }
        }
    }

    pub(crate) fn receiver(
        &mut self,
        peer: PeerId,
        connection: ConnectionId,
    ) -> mpsc::Receiver<NewStream> {
        if let Some((sender, receiver)) = self.pending_channels.remove(&peer) {
            tracing::debug!(%peer, %connection, "Returning existing pending receiver");

            self.senders.insert(connection, sender);
            return receiver;
        }

        tracing::debug!(%peer, %connection, "Creating new channel pair");

        let (sender, receiver) = mpsc::channel(0);
        self.senders.insert(connection, sender);

        receiver
    }
}

#[cfg(test)]
mod openhydra_tests {
    use super::*;

    fn conn(n: usize) -> ConnectionId {
        ConnectionId::new_unchecked(n)
    }

    /// A peer id (an identity multihash: no key needed).
    fn peer(n: u8) -> PeerId {
        PeerId::from_bytes(&[0, 4, n, 1, 2, 3]).unwrap()
    }

    fn shared() -> Shared {
        let (dial, _) = mpsc::channel(0);
        Shared::new(dial)
    }

    #[test]
    fn a_proven_direct_connection_is_preferred_over_a_relay_circuit() {
        let peer = peer(1);
        let mut s = shared();
        s.on_connection_established(conn(1), peer, true);
        s.on_connection_established(conn(2), peer, false);
        s.mark_proven(conn(1));
        // The direct one hasn't carried a stream yet: either may be picked.
        let picked: std::collections::HashSet<_> = (0..200).filter_map(|_| s.pick_connection(&peer)).collect();
        assert_eq!(picked.len(), 2);
        // Once it has, it takes every stream.
        s.mark_proven(conn(2));
        assert!((0..200).all(|_| s.pick_connection(&peer) == Some(conn(2))));
    }

    #[test]
    fn a_peer_with_no_connection_gets_no_sender_and_no_dial() {
        let peer = peer(2);
        let mut s = shared();
        assert!(s.connected_sender(peer).is_none());
        assert!(s.pending_channels.is_empty());
        s.on_connection_established(conn(1), peer, false);
        let _r = s.receiver(peer, conn(1));
        assert!(s.connected_sender(peer).is_some());
    }

    #[test]
    fn a_closed_connection_leaves_nothing_behind() {
        let peer = peer(3);
        let mut s = shared();
        s.on_connection_established(conn(1), peer, false);
        let _r = s.receiver(peer, conn(1));
        s.on_connection_closed(conn(1));
        assert!(s.connections.is_empty());
        assert!(s.senders.is_empty());
        assert!(s.connected_sender(peer).is_none());
    }

    #[test]
    fn a_dropped_acceptor_can_be_replaced() {
        let mut s = shared();
        let p = StreamProtocol::new("/x/1");
        let first = s.accept(p.clone()).unwrap();
        assert!(s.accept(p.clone()).is_err());
        drop(first);
        assert!(s.accept(p).is_ok());
    }
}
