// Copyright 2020 Parity Technologies (UK) Ltd.
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

mod connecting;
mod stream;

pub use connecting::Connecting;
#[doc(hidden)]
pub use connecting::openhydra_peer_id_from_certificate_ders;
pub use stream::Stream;

use crate::{ConnectionError, Error};

use futures::{future::BoxFuture, FutureExt};
use libp2p_core::muxing::{StreamMuxer, StreamMuxerEvent};
use std::{
    net::SocketAddr,
    pin::Pin,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

/// OpenHydra patch (25 Nov 2026): the QUIC application error code a [`Connection`] is closed with
/// when its owner closes it (`StreamMuxer::poll_close`); 0, upstream's, until set. A handle, so the
/// code can be set after the connection has gone into a boxed muxer. A connection dropped without
/// being closed (the process quitting) still goes out with 0, quinn's implicit close.
#[derive(Clone, Debug, Default)]
pub struct CloseCode(Arc<AtomicU32>);

impl CloseCode {
    /// The code to close with from now on.
    pub fn set(&self, code: u32) {
        self.0.store(code, Ordering::Relaxed);
    }

    /// The code the connection will close with.
    pub fn get(&self) -> u32 {
        self.0.load(Ordering::Relaxed)
    }

    /// Whether the connection is still there: a handle held only by whoever asked for it (the
    /// connection was dropped) is not.
    pub fn in_use(&self) -> bool {
        Arc::strong_count(&self.0) > 1
    }
}

/// State for a single opened QUIC connection.
pub struct Connection {
    /// Underlying connection.
    connection: quinn::Connection,
    /// Future for accepting a new incoming bidirectional stream.
    incoming: Option<
        BoxFuture<'static, Result<(quinn::SendStream, quinn::RecvStream), quinn::ConnectionError>>,
    >,
    /// Future for opening a new outgoing bidirectional stream.
    outgoing: Option<
        BoxFuture<'static, Result<(quinn::SendStream, quinn::RecvStream), quinn::ConnectionError>>,
    >,
    /// Future to wait for the connection to be closed.
    closing: Option<BoxFuture<'static, quinn::ConnectionError>>,
    /// OpenHydra: the application error code `poll_close` closes with.
    close_code: CloseCode,
}

impl Connection {
    /// Build a [`Connection`] from raw components.
    ///
    /// This function assumes that the [`quinn::Connection`] is completely fresh and none of
    /// its methods has ever been called. Failure to comply might lead to logic errors and panics.
    fn new(connection: quinn::Connection) -> Self {
        Self {
            connection,
            incoming: None,
            outgoing: None,
            closing: None,
            close_code: CloseCode::default(),
        }
    }

    /// OpenHydra patch: a handle to the code this connection is closed with ([`CloseCode`]).
    pub fn close_code(&self) -> CloseCode {
        self.close_code.clone()
    }

    /// OpenHydra patch: the peer's address (it can change if the peer migrates).
    pub fn remote_address(&self) -> SocketAddr {
        self.connection.remote_address()
    }
}

impl StreamMuxer for Connection {
    type Substream = Stream;
    type Error = Error;

    fn poll_inbound(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Self::Substream, Self::Error>> {
        let this = self.get_mut();

        let incoming = this.incoming.get_or_insert_with(|| {
            let connection = this.connection.clone();
            async move { connection.accept_bi().await }.boxed()
        });

        let (send, recv) = futures::ready!(incoming.poll_unpin(cx)).map_err(ConnectionError)?;
        this.incoming.take();
        let stream = Stream::new(send, recv);
        Poll::Ready(Ok(stream))
    }

    fn poll_outbound(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Self::Substream, Self::Error>> {
        let this = self.get_mut();

        let outgoing = this.outgoing.get_or_insert_with(|| {
            let connection = this.connection.clone();
            async move { connection.open_bi().await }.boxed()
        });

        let (send, recv) = futures::ready!(outgoing.poll_unpin(cx)).map_err(ConnectionError)?;
        this.outgoing.take();
        let stream = Stream::new(send, recv);
        Poll::Ready(Ok(stream))
    }

    fn poll(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<StreamMuxerEvent, Self::Error>> {
        // TODO: If connection migration is enabled (currently disabled) address
        // change on the connection needs to be handled.
        Poll::Pending
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.get_mut();

        let closing = this.closing.get_or_insert_with(|| {
            this.connection.close(From::from(this.close_code.get()), &[]);
            let connection = this.connection.clone();
            async move { connection.closed().await }.boxed()
        });

        match futures::ready!(closing.poll_unpin(cx)) {
            // Expected error given that `connection.close` was called above.
            quinn::ConnectionError::LocallyClosed => {}
            error => return Poll::Ready(Err(Error::Connection(ConnectionError(error)))),
        };

        Poll::Ready(Ok(()))
    }
}
