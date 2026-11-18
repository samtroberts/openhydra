use core::fmt;
use std::{
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use futures::{channel::mpsc, StreamExt};
use libp2p_core::{multiaddr::Protocol, transport::PortUse, ConnectedPoint, Endpoint, Multiaddr};
use libp2p_identity::PeerId;
use libp2p_swarm::{
    self as swarm, dial_opts::DialOpts, ConnectionDenied, ConnectionId, FromSwarm,
    NetworkBehaviour, THandler, THandlerInEvent, THandlerOutEvent, ToSwarm,
};
use swarm::{
    behaviour::ConnectionEstablished, dial_opts::PeerCondition, ConnectionClosed, DialError,
    DialFailure,
};

use crate::{handler::Handler, shared::Shared, Control};

/// A generic behaviour for stream-oriented protocols.
pub struct Behaviour {
    shared: Arc<Mutex<Shared>>,
    dial_receiver: mpsc::Receiver<PeerId>,
}

impl Default for Behaviour {
    fn default() -> Self {
        Self::new()
    }
}

impl Behaviour {
    pub fn new() -> Self {
        let (dial_sender, dial_receiver) = mpsc::channel(0);

        Self {
            shared: Arc::new(Mutex::new(Shared::new(dial_sender))),
            dial_receiver,
        }
    }

    /// Obtain a new [`Control`].
    pub fn new_control(&self) -> Control {
        Control::new(self.shared.clone())
    }
}

/// The protocol is already registered.
#[derive(Debug)]
pub struct AlreadyRegistered;

impl fmt::Display for AlreadyRegistered {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "The protocol is already registered")
    }
}

impl std::error::Error for AlreadyRegistered {}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler = Handler;
    type ToSwarm = ();

    fn handle_established_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        _: &Multiaddr,
        _: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(Handler::new(
            peer,
            connection_id,
            self.shared.clone(),
            Shared::lock(&self.shared).receiver(peer, connection_id),
        ))
    }

    fn handle_established_outbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        _: &Multiaddr,
        _: Endpoint,
        _: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        Ok(Handler::new(
            peer,
            connection_id,
            self.shared.clone(),
            Shared::lock(&self.shared).receiver(peer, connection_id),
        ))
    }

    fn on_swarm_event(&mut self, event: FromSwarm) {
        match event {
            FromSwarm::ConnectionEstablished(ConnectionEstablished {
                peer_id,
                connection_id,
                endpoint,
                ..
            }) => Shared::lock(&self.shared).on_connection_established(
                connection_id,
                peer_id,
                is_relayed(endpoint),
            ),
            FromSwarm::ConnectionClosed(ConnectionClosed { connection_id, .. }) => {
                Shared::lock(&self.shared).on_connection_closed(connection_id)
            }
            // OpenHydra patch: every failed dial fails the opens waiting on it (upstream left those
            // of an aborted dial, say, waiting forever), except one not made because the peer is
            // connected or being dialled already: that connection serves them.
            FromSwarm::DialFailure(DialFailure {
                peer_id: Some(peer_id),
                error,
                ..
            }) if !matches!(error, DialError::DialPeerConditionFalse(_)) => {
                let reason = error.to_string(); // We can only forward the string repr but it is better than nothing.

                Shared::lock(&self.shared).on_dial_failure(peer_id, reason)
            }
            _ => {}
        }
    }

    fn on_connection_handler_event(
        &mut self,
        _peer_id: PeerId,
        _connection_id: ConnectionId,
        event: THandlerOutEvent<Self>,
    ) {
        void::unreachable(event);
    }

    fn poll(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        if let Poll::Ready(Some(peer)) = self.dial_receiver.poll_next_unpin(cx) {
            return Poll::Ready(ToSwarm::Dial {
                opts: DialOpts::peer_id(peer)
                    .condition(PeerCondition::DisconnectedAndNotDialing)
                    .build(),
            });
        }

        Poll::Pending
    }
}

/// OpenHydra patch: whether a connection runs through a relay circuit, on either side (a connection
/// a peer opened to us through a relay has the circuit as its local address).
fn is_relayed(endpoint: &ConnectedPoint) -> bool {
    let circuit = |a: &Multiaddr| a.iter().any(|p| matches!(p, Protocol::P2pCircuit));
    match endpoint {
        ConnectedPoint::Dialer { address, .. } => circuit(address),
        ConnectedPoint::Listener { local_addr, send_back_addr } => circuit(local_addr) || circuit(send_back_addr),
    }
}
