// Copyright 2020 Sigma Prime Pty Ltd.
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

use crate::protocol::{GossipsubCodec, ProtocolConfig};
use crate::rpc_proto::proto;
use crate::types::{PeerKind, RawMessage, Rpc, RpcOut};
use crate::ValidationError;
use asynchronous_codec::Framed;
use futures::future::Either;
use futures::prelude::*;
use futures::StreamExt;
use libp2p_core::upgrade::DeniedUpgrade;
use libp2p_identity::PeerId;
use libp2p_swarm::handler::{
    ConnectionEvent, ConnectionHandler, ConnectionHandlerEvent, DialUpgradeError,
    FullyNegotiatedInbound, FullyNegotiatedOutbound, ListenUpgradeError, StreamUpgradeError,
    SubstreamProtocol,
};
use libp2p_swarm::{ConnectionId, Stream};
use quick_protobuf::MessageWrite;
use smallvec::SmallVec;
use std::{
    pin::Pin,
    task::{Context, Poll},
};
use web_time::Instant;

/// The event emitted by the Handler. This informs the behaviour of various events created
/// by the handler.
#[derive(Debug)]
pub enum HandlerEvent {
    /// A GossipsubRPC message has been received. This also contains a list of invalid messages (if
    /// any) that were received.
    Message {
        /// The GossipsubRPC message excluding any invalid messages.
        rpc: Rpc,
        /// Any invalid messages that were received in the RPC, along with the associated
        /// validation error.
        invalid_messages: Vec<(RawMessage, ValidationError)>,
    },
    /// An inbound or outbound substream has been established with the peer and this informs over
    /// which protocol. This message only occurs once per connection.
    PeerKind(PeerKind),
}

/// A message sent from the behaviour to the handler.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum HandlerIn {
    /// A gossipsub message to send.
    Message(RpcOut),
    /// The peer has joined the mesh.
    JoinedMesh,
    /// The peer has left the mesh.
    LeftMesh,
}

/// The maximum number of inbound or outbound substreams attempts we allow.
///
/// Gossipsub is supposed to have a single long-lived inbound and outbound substream. On failure we
/// attempt to recreate these. This imposes an upper bound of new substreams before we consider the
/// connection faulty and disable the handler. This also prevents against potential substream
/// creation loops.
const MAX_SUBSTREAM_ATTEMPTS: usize = 5;

#[allow(clippy::large_enum_variant)]
pub enum Handler {
    Enabled(EnabledHandler),
    Disabled(DisabledHandler),
}

/// Protocol Handler that manages a single long-lived substream with a peer.
pub struct EnabledHandler {
    /// Upgrade configuration for the gossipsub protocol.
    listen_protocol: ProtocolConfig,

    /// The single long-lived outbound substream.
    outbound_substream: Option<OutboundSubstreamState>,

    /// The single long-lived inbound substream.
    inbound_substream: Option<InboundSubstreamState>,

    /// Queue of values that we want to send to the remote.
    send_queue: SmallVec<[proto::RPC; 16]>,

    /// OpenHydra patch: the encoded size of what `send_queue` holds (see [`MAX_QUEUED_BYTES`]).
    queued_bytes: usize,

    /// OpenHydra patch: messages dropped because the queue was full.
    dropped: u64,

    /// Flag indicating that an outbound substream is being established to prevent duplicate
    /// requests.
    outbound_substream_establishing: bool,

    /// The number of outbound substreams we have requested.
    outbound_substream_attempts: usize,

    /// The number of inbound substreams that have been created by the peer.
    inbound_substream_attempts: usize,

    /// The type of peer this handler is associated to.
    peer_kind: Option<PeerKind>,

    /// Keeps track on whether we have sent the peer kind to the behaviour.
    //
    // NOTE: Use this flag rather than checking the substream count each poll.
    peer_kind_sent: bool,

    last_io_activity: Instant,

    /// Keeps track of whether this connection is for a peer in the mesh. This is used to make
    /// decisions about the keep alive state for this connection.
    in_mesh: bool,

    /// OpenHydra patch: the peer and connection this handler serves, for its logs (upstream's
    /// handler doesn't know them, so "handler disabled" couldn't be traced to a connection).
    peer: Option<PeerId>,
    connection: Option<ConnectionId>,
}

pub enum DisabledHandler {
    /// If the peer doesn't support the gossipsub protocol we do not immediately disconnect.
    /// Rather, we disable the handler and prevent any incoming or outgoing substreams from being
    /// established.
    ProtocolUnsupported {
        /// Keeps track on whether we have sent the peer kind to the behaviour.
        peer_kind_sent: bool,
    },
    /// The maximum number of inbound or outbound substream attempts have happened and thereby the
    /// handler has been disabled.
    MaxSubstreamAttempts,
}

/// State of the inbound substream, opened either by us or by the remote.
enum InboundSubstreamState {
    /// Waiting for a message from the remote. The idle state for an inbound substream.
    WaitingInput(Framed<Stream, GossipsubCodec>),
    /// The substream is being closed.
    Closing(Framed<Stream, GossipsubCodec>),
    /// An error occurred during processing.
    Poisoned,
}

/// State of the outbound substream, opened either by us or by the remote.
enum OutboundSubstreamState {
    /// Waiting for the user to send a message. The idle state for an outbound substream.
    WaitingOutput(Framed<Stream, GossipsubCodec>),
    /// Waiting to send a message to the remote.
    PendingSend(Framed<Stream, GossipsubCodec>, proto::RPC),
    /// Waiting to flush the substream so that the data arrives to the remote.
    PendingFlush(Framed<Stream, GossipsubCodec>),
    /// An error occurred during processing.
    Poisoned,
}

impl Handler {
    /// Builds a new [`Handler`].
    pub fn new(protocol_config: ProtocolConfig) -> Self {
        crate::openhydra_counters::HANDLERS_ALIVE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Handler::Enabled(EnabledHandler {
            listen_protocol: protocol_config,
            inbound_substream: None,
            outbound_substream: None,
            outbound_substream_establishing: false,
            outbound_substream_attempts: 0,
            inbound_substream_attempts: 0,
            send_queue: SmallVec::new(),
            queued_bytes: 0,
            dropped: 0,
            peer_kind: None,
            peer_kind_sent: false,
            last_io_activity: Instant::now(),
            in_mesh: false,
            peer: None,
            connection: None,
        })
    }

    /// OpenHydra patch: name the peer and connection this handler serves (see `EnabledHandler::peer`).
    pub(crate) fn for_connection(mut self, peer: PeerId, connection: ConnectionId) -> Self {
        if let Handler::Enabled(h) = &mut self {
            h.peer = Some(peer);
            h.connection = Some(connection);
        }
        self
    }
}

/// OpenHydra patch: the most a connection's send queue holds, in encoded bytes. Upstream's queue had
/// no bound, so a peer that stopped reading while others published made us hold every message for
/// it (2026-10-29: one such subscriber took a 1 GB bootstrap past 1 GB in seconds). A message that
/// doesn't fit is dropped, as gossip may (the peer misses it, as it would a lost forward).
pub(crate) const MAX_QUEUED_BYTES: usize = 256 * 1024;

/// OpenHydra patch: the send-queue cap in use: [`MAX_QUEUED_BYTES`].
fn max_queued_bytes() -> usize {
    MAX_QUEUED_BYTES
}

/// OpenHydra patch: the most messages a connection's send queue holds, whatever their size.
pub(crate) const MAX_QUEUED_MESSAGES: usize = 4096;

impl EnabledHandler {
    /// OpenHydra patch: queue `rpc` for the remote unless that would exceed [`MAX_QUEUED_BYTES`].
    /// Control and subscription messages (no publish) get twice the room, within
    /// [`MAX_QUEUED_MESSAGES`], so a queue full of forwards doesn't lose a GRAFT or a SUBSCRIBE.
    fn enqueue(&mut self, rpc: proto::RPC) {
        let size = rpc.get_size();
        let cap = max_queued_bytes();
        let room = if rpc.publish.is_empty() { 2 * cap } else { cap };
        if self.queued_bytes.saturating_add(size) > room || self.send_queue.len() >= MAX_QUEUED_MESSAGES {
            self.dropped += 1;
            if self.dropped.is_power_of_two() {
                tracing::debug!(peer = ?self.peer, connection = ?self.connection, dropped = self.dropped, queued = self.queued_bytes, "Send queue full, dropping message");
            }
            return;
        }
        self.queued_bytes += size;
        crate::openhydra_counters::QUEUED_BYTES.fetch_add(size as u64, std::sync::atomic::Ordering::Relaxed);
        self.send_queue.push(rpc);
    }

    fn on_fully_negotiated_inbound(
        &mut self,
        (substream, peer_kind): (Framed<Stream, GossipsubCodec>, PeerKind),
    ) {
        // update the known kind of peer
        if self.peer_kind.is_none() {
            self.peer_kind = Some(peer_kind);
        }

        // new inbound substream. Replace the current one, if it exists.
        tracing::debug!(
            peer = ?self.peer, connection = ?self.connection,
            attempt = self.inbound_substream_attempts, replaces = self.inbound_substream.is_some(),
            "New inbound gossip substream"
        );
        self.inbound_substream = Some(InboundSubstreamState::WaitingInput(substream));
    }

    fn on_fully_negotiated_outbound(
        &mut self,
        FullyNegotiatedOutbound { protocol, .. }: FullyNegotiatedOutbound<
            <Handler as ConnectionHandler>::OutboundProtocol,
            <Handler as ConnectionHandler>::OutboundOpenInfo,
        >,
    ) {
        let (substream, peer_kind) = protocol;

        // update the known kind of peer
        if self.peer_kind.is_none() {
            self.peer_kind = Some(peer_kind);
        }

        assert!(
            self.outbound_substream.is_none(),
            "Established an outbound substream with one already available"
        );
        tracing::debug!(
            peer = ?self.peer, connection = ?self.connection,
            attempt = self.outbound_substream_attempts, "New outbound gossip substream"
        );
        self.outbound_substream = Some(OutboundSubstreamState::WaitingOutput(substream));
    }

    fn poll(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<
        ConnectionHandlerEvent<
            <Handler as ConnectionHandler>::OutboundProtocol,
            <Handler as ConnectionHandler>::OutboundOpenInfo,
            <Handler as ConnectionHandler>::ToBehaviour,
        >,
    > {
        if !self.peer_kind_sent {
            if let Some(peer_kind) = self.peer_kind.as_ref() {
                self.peer_kind_sent = true;
                return Poll::Ready(ConnectionHandlerEvent::NotifyBehaviour(
                    HandlerEvent::PeerKind(peer_kind.clone()),
                ));
            }
        }

        // determine if we need to create the outbound stream
        if !self.send_queue.is_empty()
            && self.outbound_substream.is_none()
            && !self.outbound_substream_establishing
        {
            self.outbound_substream_establishing = true;
            return Poll::Ready(ConnectionHandlerEvent::OutboundSubstreamRequest {
                protocol: SubstreamProtocol::new(self.listen_protocol.clone(), ()),
            });
        }

        // process outbound stream
        loop {
            match std::mem::replace(
                &mut self.outbound_substream,
                Some(OutboundSubstreamState::Poisoned),
            ) {
                // outbound idle state
                Some(OutboundSubstreamState::WaitingOutput(substream)) => {
                    if let Some(message) = self.send_queue.pop() {
                        let size = message.get_size().min(self.queued_bytes);
                        crate::openhydra_counters::QUEUED_BYTES.fetch_sub(size as u64, std::sync::atomic::Ordering::Relaxed);
                        self.queued_bytes -= size;
                        self.send_queue.shrink_to_fit();
                        self.outbound_substream =
                            Some(OutboundSubstreamState::PendingSend(substream, message));
                        continue;
                    }

                    self.outbound_substream =
                        Some(OutboundSubstreamState::WaitingOutput(substream));
                    break;
                }
                Some(OutboundSubstreamState::PendingSend(mut substream, message)) => {
                    match Sink::poll_ready(Pin::new(&mut substream), cx) {
                        Poll::Ready(Ok(())) => {
                            match Sink::start_send(Pin::new(&mut substream), message) {
                                Ok(()) => {
                                    self.outbound_substream =
                                        Some(OutboundSubstreamState::PendingFlush(substream))
                                }
                                Err(e) => {
                                    tracing::debug!(peer = ?self.peer, connection = ?self.connection, "Failed to send message on outbound stream: {e}");
                                    self.outbound_substream = None;
                                    break;
                                }
                            }
                        }
                        Poll::Ready(Err(e)) => {
                            tracing::debug!(peer = ?self.peer, connection = ?self.connection, "Failed to send message on outbound stream: {e}");
                            self.outbound_substream = None;
                            break;
                        }
                        Poll::Pending => {
                            self.outbound_substream =
                                Some(OutboundSubstreamState::PendingSend(substream, message));
                            break;
                        }
                    }
                }
                Some(OutboundSubstreamState::PendingFlush(mut substream)) => {
                    match Sink::poll_flush(Pin::new(&mut substream), cx) {
                        Poll::Ready(Ok(())) => {
                            self.last_io_activity = Instant::now();
                            self.outbound_substream =
                                Some(OutboundSubstreamState::WaitingOutput(substream))
                        }
                        Poll::Ready(Err(e)) => {
                            tracing::debug!(peer = ?self.peer, connection = ?self.connection, "Failed to flush outbound stream: {e}");
                            self.outbound_substream = None;
                            break;
                        }
                        Poll::Pending => {
                            self.outbound_substream =
                                Some(OutboundSubstreamState::PendingFlush(substream));
                            break;
                        }
                    }
                }
                None => {
                    self.outbound_substream = None;
                    break;
                }
                Some(OutboundSubstreamState::Poisoned) => {
                    unreachable!("Error occurred during outbound stream processing")
                }
            }
        }

        loop {
            match std::mem::replace(
                &mut self.inbound_substream,
                Some(InboundSubstreamState::Poisoned),
            ) {
                // inbound idle state
                Some(InboundSubstreamState::WaitingInput(mut substream)) => {
                    match substream.poll_next_unpin(cx) {
                        Poll::Ready(Some(Ok(message))) => {
                            self.last_io_activity = Instant::now();
                            self.inbound_substream =
                                Some(InboundSubstreamState::WaitingInput(substream));
                            return Poll::Ready(ConnectionHandlerEvent::NotifyBehaviour(message));
                        }
                        Poll::Ready(Some(Err(error))) => {
                            tracing::debug!(peer = ?self.peer, connection = ?self.connection, "Failed to read from inbound stream: {error}");
                            // Close this side of the stream. If the
                            // peer is still around, they will re-establish their
                            // outbound stream i.e. our inbound stream.
                            self.inbound_substream =
                                Some(InboundSubstreamState::Closing(substream));
                        }
                        // peer closed the stream
                        Poll::Ready(None) => {
                            tracing::debug!(peer = ?self.peer, connection = ?self.connection, "Inbound stream closed by remote");
                            self.inbound_substream =
                                Some(InboundSubstreamState::Closing(substream));
                        }
                        Poll::Pending => {
                            self.inbound_substream =
                                Some(InboundSubstreamState::WaitingInput(substream));
                            break;
                        }
                    }
                }
                Some(InboundSubstreamState::Closing(mut substream)) => {
                    match Sink::poll_close(Pin::new(&mut substream), cx) {
                        Poll::Ready(res) => {
                            if let Err(e) = res {
                                // Don't close the connection but just drop the inbound substream.
                                // In case the remote has more to send, they will open up a new
                                // substream.
                                tracing::debug!(peer = ?self.peer, connection = ?self.connection, "Inbound substream error while closing: {e}");
                            }
                            self.inbound_substream = None;
                            break;
                        }
                        Poll::Pending => {
                            self.inbound_substream =
                                Some(InboundSubstreamState::Closing(substream));
                            break;
                        }
                    }
                }
                None => {
                    self.inbound_substream = None;
                    break;
                }
                Some(InboundSubstreamState::Poisoned) => {
                    unreachable!("Error occurred during inbound stream processing")
                }
            }
        }

        Poll::Pending
    }
}

impl ConnectionHandler for Handler {
    type FromBehaviour = HandlerIn;
    type ToBehaviour = HandlerEvent;
    type InboundOpenInfo = ();
    type InboundProtocol = either::Either<ProtocolConfig, DeniedUpgrade>;
    type OutboundOpenInfo = ();
    type OutboundProtocol = ProtocolConfig;

    fn listen_protocol(&self) -> SubstreamProtocol<Self::InboundProtocol, Self::InboundOpenInfo> {
        match self {
            Handler::Enabled(handler) => {
                SubstreamProtocol::new(either::Either::Left(handler.listen_protocol.clone()), ())
            }
            Handler::Disabled(_) => {
                SubstreamProtocol::new(either::Either::Right(DeniedUpgrade), ())
            }
        }
    }

    fn on_behaviour_event(&mut self, message: HandlerIn) {
        match self {
            Handler::Enabled(handler) => match message {
                HandlerIn::Message(m) => handler.enqueue(m.into_protobuf()),
                HandlerIn::JoinedMesh => {
                    handler.in_mesh = true;
                }
                HandlerIn::LeftMesh => {
                    handler.in_mesh = false;
                }
            },
            Handler::Disabled(_) => {
                tracing::debug!(?message, "Handler is disabled. Dropping message");
            }
        }
    }

    fn connection_keep_alive(&self) -> bool {
        matches!(self, Handler::Enabled(h) if h.in_mesh)
    }

    #[tracing::instrument(level = "trace", name = "ConnectionHandler::poll", skip(self, cx))]
    fn poll(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<
        ConnectionHandlerEvent<Self::OutboundProtocol, Self::OutboundOpenInfo, Self::ToBehaviour>,
    > {
        match self {
            Handler::Enabled(handler) => handler.poll(cx),
            Handler::Disabled(DisabledHandler::ProtocolUnsupported { peer_kind_sent }) => {
                if !*peer_kind_sent {
                    *peer_kind_sent = true;
                    return Poll::Ready(ConnectionHandlerEvent::NotifyBehaviour(
                        HandlerEvent::PeerKind(PeerKind::NotSupported),
                    ));
                }

                Poll::Pending
            }
            Handler::Disabled(DisabledHandler::MaxSubstreamAttempts) => Poll::Pending,
        }
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
        match self {
            Handler::Enabled(handler) => {
                if event.is_inbound() {
                    handler.inbound_substream_attempts += 1;

                    if handler.inbound_substream_attempts == MAX_SUBSTREAM_ATTEMPTS {
                        tracing::warn!(
                            peer = ?handler.peer, connection = ?handler.connection,
                            "The maximum number of inbound substreams attempts has been exceeded"
                        );
                        crate::openhydra_counters::HANDLERS_DISABLED
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        *self = Handler::Disabled(DisabledHandler::MaxSubstreamAttempts);
                        return;
                    }
                }

                if event.is_outbound() {
                    handler.outbound_substream_establishing = false;

                    handler.outbound_substream_attempts += 1;

                    if handler.outbound_substream_attempts == MAX_SUBSTREAM_ATTEMPTS {
                        tracing::warn!(
                            peer = ?handler.peer, connection = ?handler.connection,
                            "The maximum number of outbound substream attempts has been exceeded"
                        );
                        crate::openhydra_counters::HANDLERS_DISABLED
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        *self = Handler::Disabled(DisabledHandler::MaxSubstreamAttempts);
                        return;
                    }
                }

                match event {
                    ConnectionEvent::FullyNegotiatedInbound(FullyNegotiatedInbound {
                        protocol,
                        ..
                    }) => match protocol {
                        Either::Left(protocol) => handler.on_fully_negotiated_inbound(protocol),
                        Either::Right(v) => void::unreachable(v),
                    },
                    ConnectionEvent::FullyNegotiatedOutbound(fully_negotiated_outbound) => {
                        handler.on_fully_negotiated_outbound(fully_negotiated_outbound)
                    }
                    ConnectionEvent::DialUpgradeError(DialUpgradeError {
                        error: StreamUpgradeError::Timeout,
                        ..
                    }) => {
                        tracing::debug!(peer = ?handler.peer, connection = ?handler.connection, "Dial upgrade error: Protocol negotiation timeout");
                    }
                    ConnectionEvent::DialUpgradeError(DialUpgradeError {
                        error: StreamUpgradeError::Apply(e),
                        ..
                    }) => void::unreachable(e),
                    ConnectionEvent::DialUpgradeError(DialUpgradeError {
                        error: StreamUpgradeError::NegotiationFailed,
                        ..
                    }) => {
                        // The protocol is not supported
                        tracing::debug!(
                            "The remote peer does not support gossipsub on this connection"
                        );
                        *self = Handler::Disabled(DisabledHandler::ProtocolUnsupported {
                            peer_kind_sent: false,
                        });
                    }
                    ConnectionEvent::DialUpgradeError(DialUpgradeError {
                        error: StreamUpgradeError::Io(e),
                        ..
                    }) => {
                        tracing::debug!(peer = ?handler.peer, connection = ?handler.connection, "Protocol negotiation failed: {e}")
                    }
                    ConnectionEvent::ListenUpgradeError(ListenUpgradeError { error, .. }) => {
                        tracing::debug!(peer = ?handler.peer, connection = ?handler.connection, ?error, "Inbound gossip substream failed to negotiate");
                    }
                    _ => {}
                }
            }
            Handler::Disabled(_) => {}
        }
    }
}

/// OpenHydra patch: exercises the send-queue cap from outside the crate (upstream's own tests need an
/// unpublished dev-dependency, so they can't run from the vendored copy): queues `n` forwards of
/// `len` bytes on a fresh handler whose peer never reads. Returns (queued bytes, queued messages,
/// dropped messages, the cap).
#[doc(hidden)]
pub fn openhydra_queue_selftest(n: usize, len: usize) -> (usize, usize, u64, usize) {
    let mut h = Handler::new(ProtocolConfig::default());
    for _ in 0..n {
        h.on_behaviour_event(HandlerIn::Message(RpcOut::Forward(RawMessage {
            source: None,
            data: vec![7; len],
            sequence_number: Some(1),
            topic: crate::TopicHash::from_raw("t"),
            signature: None,
            key: None,
            validated: true,
        })));
    }
    match &h {
        Handler::Enabled(e) => {
            debug_assert_eq!(e.queued_bytes, e.send_queue.iter().map(|m| m.get_size()).sum::<usize>());
            (e.queued_bytes, e.send_queue.len(), e.dropped, MAX_QUEUED_BYTES)
        }
        Handler::Disabled(_) => (0, 0, 0, MAX_QUEUED_BYTES),
    }
}

/// OpenHydra patch: keep the process-wide gauges (`openhydra_counters`) right when a connection's
/// handler goes, with whatever its queue still held.
impl Drop for Handler {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        // Only an enabled handler was counted (`Handler::new`); disabling one drops it here once.
        if let Handler::Enabled(h) = self {
            crate::openhydra_counters::HANDLERS_ALIVE.fetch_sub(1, Relaxed);
            crate::openhydra_counters::QUEUED_BYTES.fetch_sub(h.queued_bytes as u64, Relaxed);
        }
    }
}
