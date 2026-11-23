// Copyright 2017-2020 Parity Technologies (UK) Ltd.
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

//! Future that drives a QUIC connection until is has performed its TLS handshake.

use crate::{Connection, ConnectionError, Error};

use futures::{
    future::{select, Either, FutureExt, Select},
    prelude::*,
};
use futures_timer::Delay;
use libp2p_identity::PeerId;
use quinn::rustls::pki_types::CertificateDer;
use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

/// A QUIC connection currently being negotiated.
#[derive(Debug)]
pub struct Connecting {
    connecting: Select<quinn::Connecting, Delay>,
}

impl Connecting {
    pub(crate) fn new(connection: quinn::Connecting, timeout: Duration) -> Self {
        Connecting {
            connecting: select(connection, Delay::new(timeout)),
        }
    }
}

impl Connecting {
    /// Returns the peer id of the node we're connected to.
    ///
    /// OpenHydra patch (23 Nov 2026), backport of the RUSTSEC-2026-0525 fix (upstream 212f3774): every
    /// step failing is an error, not a panic. The certificate is parsed again here against the time
    /// now, so one that expired during the handshake made the `expect` panic: a remote crash.
    fn remote_peer_id(connection: &quinn::Connection) -> Result<PeerId, Error> {
        let identity = connection
            .peer_identity()
            .ok_or_else(|| transport_err("no peer identity after the TLS handshake"))?;
        let certificates: Box<Vec<CertificateDer>> = identity
            .downcast()
            .map_err(|_| transport_err("peer identity is not a rustls certificate chain"))?;
        let end_entity = certificates
            .first()
            .ok_or_else(|| transport_err("no peer certificate"))?;
        let p2p_cert = libp2p_tls::certificate::parse(end_entity)
            .map_err(|_| transport_err("peer certificate no longer valid"))?;
        Ok(p2p_cert.peer_id())
    }
}

impl Future for Connecting {
    type Output = Result<(PeerId, Connection), Error>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let connection = match futures::ready!(self.connecting.poll_unpin(cx)) {
            Either::Right(_) => return Poll::Ready(Err(Error::HandshakeTimedOut)),
            Either::Left((connection, _)) => connection.map_err(ConnectionError)?,
        };

        let peer_id = Self::remote_peer_id(&connection)?;
        let muxer = Connection::new(connection);
        Poll::Ready(Ok((peer_id, muxer)))
    }
}

/// OpenHydra patch: a handshake whose peer can't be identified fails as invalid data.
fn transport_err(reason: &str) -> Error {
    Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, reason.to_string()))
}
