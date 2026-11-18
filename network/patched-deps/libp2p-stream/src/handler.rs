use std::{
    collections::HashMap,
    io,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use futures::{
    channel::{mpsc, oneshot},
    StreamExt as _,
};
use libp2p_identity::PeerId;
use libp2p_swarm::{
    self as swarm,
    handler::{ConnectionEvent, DialUpgradeError, FullyNegotiatedInbound, FullyNegotiatedOutbound},
    ConnectionHandler, ConnectionId, Stream, StreamProtocol,
};

use crate::{shared::Shared, upgrade::Upgrade, OpenStreamError};

/// OpenHydra patch: outbound streams one connection may be negotiating at once. Upstream negotiated
/// one at a time, so each open waited for every open before it to finish negotiating (a round trip
/// or two each): a hundred opens at once took a hundred round trips for the last.
const MAX_PENDING_UPGRADES: usize = 128;

type Pending = (StreamProtocol, oneshot::Sender<Result<Stream, OpenStreamError>>);

pub struct Handler {
    remote: PeerId,
    connection: ConnectionId,
    shared: Arc<Mutex<Shared>>,

    receiver: mpsc::Receiver<NewStream>,
    /// OpenHydra patch: the outbound streams being negotiated, by the id each was opened with.
    pending_upgrades: HashMap<u64, Pending>,
    next_upgrade: u64,
}

impl Handler {
    pub(crate) fn new(
        remote: PeerId,
        connection: ConnectionId,
        shared: Arc<Mutex<Shared>>,
        receiver: mpsc::Receiver<NewStream>,
    ) -> Self {
        Self {
            shared,
            receiver,
            pending_upgrades: HashMap::new(),
            next_upgrade: 0,
            remote,
            connection,
        }
    }
}

impl ConnectionHandler for Handler {
    type FromBehaviour = void::Void;
    type ToBehaviour = void::Void;
    type InboundProtocol = Upgrade;
    type OutboundProtocol = Upgrade;
    type InboundOpenInfo = ();
    type OutboundOpenInfo = u64;

    fn listen_protocol(
        &self,
    ) -> swarm::SubstreamProtocol<Self::InboundProtocol, Self::InboundOpenInfo> {
        swarm::SubstreamProtocol::new(
            Upgrade {
                supported_protocols: Shared::lock(&self.shared).supported_inbound_protocols(),
            },
            (),
        )
    }

    fn poll(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<
        swarm::ConnectionHandlerEvent<
            Self::OutboundProtocol,
            Self::OutboundOpenInfo,
            Self::ToBehaviour,
        >,
    > {
        if self.pending_upgrades.len() >= MAX_PENDING_UPGRADES {
            return Poll::Pending;
        }

        match self.receiver.poll_next_unpin(cx) {
            Poll::Ready(Some(new_stream)) => {
                let id = self.next_upgrade;
                self.next_upgrade = self.next_upgrade.wrapping_add(1);
                self.pending_upgrades
                    .insert(id, (new_stream.protocol.clone(), new_stream.sender));
                return Poll::Ready(swarm::ConnectionHandlerEvent::OutboundSubstreamRequest {
                    protocol: swarm::SubstreamProtocol::new(
                        Upgrade {
                            supported_protocols: vec![new_stream.protocol],
                        },
                        id,
                    ),
                });
            }
            Poll::Ready(None) => {} // Sender is gone, no more work to do.
            Poll::Pending => {}
        }

        Poll::Pending
    }

    fn on_behaviour_event(&mut self, event: Self::FromBehaviour) {
        void::unreachable(event)
    }

    fn on_connection_event(
        &mut self,
        event: ConnectionEvent<
            Self::InboundProtocol,
            Self::OutboundProtocol,
            Self::InboundOpenInfo,
            Self::OutboundOpenInfo,
        >,
    ) {
        match event {
            ConnectionEvent::FullyNegotiatedInbound(FullyNegotiatedInbound {
                protocol: (stream, protocol),
                info: (),
            }) => {
                Shared::lock(&self.shared).on_inbound_stream(self.remote, self.connection, stream, protocol);
            }
            ConnectionEvent::FullyNegotiatedOutbound(FullyNegotiatedOutbound {
                protocol: (stream, actual_protocol),
                info,
            }) => {
                let Some((expected_protocol, sender)) = self.pending_upgrades.remove(&info) else {
                    debug_assert!(
                        false,
                        "Negotiated an outbound stream without a back channel"
                    );
                    return;
                };
                debug_assert_eq!(expected_protocol, actual_protocol);

                Shared::lock(&self.shared).mark_proven(self.connection);
                let _ = sender.send(Ok(stream));
            }
            ConnectionEvent::DialUpgradeError(DialUpgradeError { error, info }) => {
                let Some((p, sender)) = self.pending_upgrades.remove(&info) else {
                    debug_assert!(
                        false,
                        "Received a `DialUpgradeError` without a back channel"
                    );
                    return;
                };

                let error = match error {
                    swarm::StreamUpgradeError::Timeout => {
                        OpenStreamError::Io(io::Error::from(io::ErrorKind::TimedOut))
                    }
                    swarm::StreamUpgradeError::Apply(v) => void::unreachable(v),
                    swarm::StreamUpgradeError::NegotiationFailed => {
                        OpenStreamError::UnsupportedProtocol(p)
                    }
                    swarm::StreamUpgradeError::Io(io) => OpenStreamError::Io(io),
                };

                let _ = sender.send(Err(error));
            }
            _ => {}
        }
    }
}

/// Message from a [`Control`](crate::Control) to a [`ConnectionHandler`] to negotiate a new outbound stream.
#[derive(Debug)]
pub(crate) struct NewStream {
    pub(crate) protocol: StreamProtocol,
    pub(crate) sender: oneshot::Sender<Result<Stream, OpenStreamError>>,
}
