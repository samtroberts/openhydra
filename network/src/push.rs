// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Push streams (#4): a reply's frames go to its consumer the moment they are made, on a stream of
//! their own, instead of answering polls a round trip apart.
//!
//! The consumer opens a `/openhydra/push/1.0.0` stream on a connection it already has to the
//! provider and writes its stream open on it, the same bytes it would send as a request. The
//! provider's agent sees that open as an ordinary inbound request ([`crate::handle::NetworkHandle::poll_inbound`])
//! under a push id ([`is_push_id`]): each [`respond`](crate::handle::NetworkHandle::respond) to that
//! id is written to the stream as one message, and the stream stays open until the agent ends it
//! ([`crate::handle::NetworkHandle::end_push`]) or either side goes away. Messages the consumer
//! writes after its open (acknowledgements) reach the agent as inbound requests under an id
//! [`is_push_message_id`] says is one, which nothing answers.
//!
//! Every message is a 4-byte big-endian length, then that many bytes. Opens only ride a connection
//! that exists (the request path knows how to reach a peer: relays, reversal, hole punching), so a
//! consumer with no connection, or a provider that doesn't know the protocol, falls back to polls.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, StreamExt};
use libp2p::{PeerId, StreamProtocol};
use tokio::sync::mpsc;
use tracing::debug;

use crate::event_loop::SharedProxyQueue;

/// The push stream protocol.
pub const PUSH_PROTOCOL: StreamProtocol = StreamProtocol::new("/openhydra/push/1.0.0");

/// The id prefix of a push stream's open, as the agent receives it.
const PUSH_ID_PREFIX: &str = "push:";
/// The id prefix of a message the consumer wrote on its push stream after the open.
const PUSH_MESSAGE_PREFIX: &str = "push-in:";

/// The largest open or pushed message (a request's own limit, `proxy::MAX_MSG_SIZE`).
const MAX_MESSAGE: usize = 100 * 1024 * 1024;
/// The largest message a consumer may write after its open (an acknowledgement is 21 bytes).
const MAX_FOLLOW_UP: usize = 1024;
/// How long a provider waits for the open after the stream is negotiated.
const OPEN_READ_TIMEOUT: Duration = Duration::from_secs(10);
/// Push streams one consumer may hold open at a provider at once (its stream cap, `provider.rs`
/// `MAX_STREAMS_PER_PEER`), and all consumers together. Past them a stream is dropped unread, and
/// its consumer polls instead.
pub const MAX_PUSH_PER_PEER: usize = 128;
pub const MAX_PUSH_TOTAL: usize = 1024;
/// Messages waiting to be written to one consumer (a slow reader): past it the push ends and the
/// consumer polls for the rest.
const OUTBOX: usize = 1024;
/// A consumer's messages after its open closer together than this are dropped (it acknowledges
/// about twice a second).
const MIN_FOLLOW_UP_GAP: Duration = Duration::from_millis(50);

/// Whether `request_id` is a push stream's open (its responds go on the stream).
pub fn is_push_id(request_id: &str) -> bool {
    request_id.starts_with(PUSH_ID_PREFIX)
}

/// Whether `request_id` is a message a consumer wrote on its push stream after the open (never
/// answered).
pub fn is_push_message_id(request_id: &str) -> bool {
    request_id.starts_with(PUSH_MESSAGE_PREFIX)
}

/// Why a push stream couldn't be opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushOpenError {
    /// The peer doesn't speak the protocol (an older build): poll it.
    Unsupported,
    /// The open never reached the peer (no connection, or the stream couldn't be negotiated): send
    /// it the usual way.
    NotSent(String),
    /// The open was written, and then the stream failed: the peer may have begun serving it.
    Lost(String),
}

impl std::fmt::Display for PushOpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PushOpenError::Unsupported => write!(f, "peer doesn't push"),
            PushOpenError::NotSent(e) => write!(f, "push open not sent: {e}"),
            PushOpenError::Lost(e) => write!(f, "push open lost: {e}"),
        }
    }
}

/// A consumer's end of a push stream: blocking reads of what the provider pushes.
pub struct PushStream {
    inbox: std::sync::mpsc::Receiver<Result<Vec<u8>, String>>,
    outbox: mpsc::UnboundedSender<Vec<u8>>,
    task: tokio::task::AbortHandle,
}

impl PushStream {
    /// The next message the provider pushed, waiting up to `timeout`. An error once the stream has
    /// ended or broken, or nothing came in time (the stream is then no use: drop it).
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Vec<u8>, String> {
        match self.inbox.recv_timeout(timeout) {
            Ok(r) => r,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(format!("nothing pushed in {timeout:?}")),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err("push stream ended".into()),
        }
    }

    /// Write `message` to the provider (best effort: an error once the stream has ended).
    pub fn send(&self, message: Vec<u8>) -> Result<(), String> {
        self.outbox.send(message).map_err(|_| "push stream ended".to_string())
    }
}

impl Drop for PushStream {
    fn drop(&mut self) {
        // Drops the stream, which closes it.
        self.task.abort();
    }
}

/// A provider's push stream, as the agent writes to it.
struct Outbox {
    tx: mpsc::Sender<Vec<u8>>,
}

/// A node's push streams.
pub struct PushStreams {
    control: Mutex<libp2p_stream::Control>,
    rt: tokio::runtime::Handle,
    local: PeerId,
    /// Provider side: the open push streams, by the id their open was handed to the agent under.
    open: Mutex<HashMap<String, Outbox>>,
    /// Provider side: push streams open, per consumer.
    per_peer: Mutex<HashMap<PeerId, usize>>,
    next_id: AtomicU64,
    /// Provider side: push stream opens handed to the agent so far.
    served: AtomicU64,
}

/// Counts a provider-side push stream for as long as it lives.
struct Held(Arc<PushStreams>, PeerId);

impl Drop for Held {
    fn drop(&mut self) {
        let mut per_peer = self.0.per_peer.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = per_peer.get_mut(&self.1) {
            *n -= 1;
            if *n == 0 {
                per_peer.remove(&self.1);
            }
        }
    }
}

impl PushStreams {
    /// Must be called inside the swarm's tokio runtime.
    pub fn new(control: libp2p_stream::Control, local: PeerId) -> Arc<Self> {
        Arc::new(Self {
            control: Mutex::new(control),
            rt: tokio::runtime::Handle::current(),
            local,
            open: Mutex::new(HashMap::new()),
            per_peer: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            served: AtomicU64::new(0),
        })
    }

    /// Provider side: accept push streams, handing each open to the agent through `queue`. Must be
    /// called inside the swarm's tokio runtime.
    pub fn accept(self: &Arc<Self>, queue: Arc<SharedProxyQueue>) -> Result<(), String> {
        let mut incoming = self
            .control
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .accept(PUSH_PROTOCOL)
            .map_err(|e| e.to_string())?;
        let me = Arc::clone(self);
        tokio::spawn(async move {
            while let Some((peer, stream)) = incoming.next().await {
                let Some(held) = me.admit(peer) else {
                    debug!(%peer, "push stream refused: too many open");
                    continue;
                };
                let (me, queue) = (Arc::clone(&me), Arc::clone(&queue));
                tokio::spawn(async move { me.serve(peer, stream, queue, held).await });
            }
        });
        Ok(())
    }

    /// Count a new push stream from `peer`, within the limits.
    fn admit(self: &Arc<Self>, peer: PeerId) -> Option<Held> {
        let mut per_peer = self.per_peer.lock().unwrap_or_else(|e| e.into_inner());
        let total: usize = per_peer.values().sum();
        let mine = per_peer.entry(peer).or_default();
        if *mine >= MAX_PUSH_PER_PEER || total >= MAX_PUSH_TOTAL {
            if *mine == 0 {
                per_peer.remove(&peer);
            }
            return None;
        }
        *mine += 1;
        Some(Held(Arc::clone(self), peer))
    }

    /// Provider side: one push stream, from its open to its end.
    async fn serve(self: Arc<Self>, peer: PeerId, stream: libp2p::Stream, queue: Arc<SharedProxyQueue>, held: Held) {
        let _held = held;
        let (mut rd, mut wr) = stream.split();
        let open = match tokio::time::timeout(OPEN_READ_TIMEOUT, read_message(&mut rd, MAX_MESSAGE)).await {
            Ok(Ok(Some(open))) => open,
            _ => return,
        };
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        let id = format!("{PUSH_ID_PREFIX}{n}");
        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(OUTBOX);
        self.open.lock().unwrap_or_else(|e| e.into_inner()).insert(id.clone(), Outbox { tx });
        queue.push((id.clone(), peer.to_string(), open));
        self.served.fetch_add(1, Ordering::Relaxed);
        // What the consumer writes after its open goes to the agent, paced; its end ends the push.
        let follow_ups = {
            let (queue, peer) = (Arc::clone(&queue), peer.to_string());
            async move {
                let mut last: Option<Instant> = None;
                while let Ok(Some(m)) = read_message(&mut rd, MAX_FOLLOW_UP).await {
                    if last.is_some_and(|t| t.elapsed() < MIN_FOLLOW_UP_GAP) {
                        continue;
                    }
                    last = Some(Instant::now());
                    queue.push((format!("{PUSH_MESSAGE_PREFIX}{n}"), peer.clone(), m));
                }
            }
        };
        // What the agent pushes, until it ends the push (drops its outbox) or a write fails.
        let pushes = async move {
            while let Some(m) = rx.recv().await {
                if write_message(&mut wr, &m).await.is_err() {
                    return;
                }
            }
            let _ = wr.close().await;
        };
        tokio::select! {
            _ = follow_ups => {}
            _ = pushes => {}
        }
        self.open.lock().unwrap_or_else(|e| e.into_inner()).remove(&id);
    }

    /// Provider side: write `message` to the push stream `id`. An error when it has ended (its
    /// consumer went away or it was ended), or its consumer reads too slowly (the push then ends).
    pub fn send(&self, id: &str, message: Vec<u8>) -> Result<(), String> {
        let mut open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        let Some(outbox) = open.get(id) else {
            return Err("push stream ended".into());
        };
        match outbox.tx.try_send(message) {
            Ok(()) => Ok(()),
            Err(e) => {
                open.remove(id);
                Err(match e {
                    mpsc::error::TrySendError::Full(_) => "push consumer too slow".into(),
                    mpsc::error::TrySendError::Closed(_) => "push stream ended".into(),
                })
            }
        }
    }

    /// Provider side: end the push stream `id` once what was sent on it is written.
    pub fn end(&self, id: &str) {
        self.open.lock().unwrap_or_else(|e| e.into_inner()).remove(id);
    }

    /// Provider side: push stream opens handed to the agent so far.
    pub fn served_count(&self) -> u64 {
        self.served.load(Ordering::Relaxed)
    }

    /// Provider side: push streams open now.
    pub fn open_count(&self) -> usize {
        self.open.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Consumer side: open a push stream to `peer` on a connection it already has and write `open`
    /// on it, within `timeout`.
    pub fn open(&self, peer: &str, open: Vec<u8>, timeout: Duration) -> Result<PushStream, PushOpenError> {
        let peer: PeerId = peer.parse().map_err(|_| PushOpenError::NotSent("invalid peer_id".into()))?;
        if peer == self.local {
            return Err(PushOpenError::NotSent("own node".into()));
        }
        let mut control = self.control.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let (opened_tx, opened_rx) = std::sync::mpsc::channel::<Result<(), PushOpenError>>();
        let (inbox_tx, inbox) = std::sync::mpsc::channel::<Result<Vec<u8>, String>>();
        let (outbox, mut outbox_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let task = self.rt.spawn(async move {
            let stream = match tokio::time::timeout(timeout, control.open_connected_stream(peer, PUSH_PROTOCOL)).await {
                Ok(Ok(s)) => s,
                Ok(Err(libp2p_stream::OpenStreamError::UnsupportedProtocol(_))) => {
                    let _ = opened_tx.send(Err(PushOpenError::Unsupported));
                    return;
                }
                Ok(Err(e)) => {
                    let _ = opened_tx.send(Err(PushOpenError::NotSent(e.to_string())));
                    return;
                }
                Err(_) => {
                    let _ = opened_tx.send(Err(PushOpenError::NotSent("timed out opening".into())));
                    return;
                }
            };
            let (mut rd, mut wr) = stream.split();
            // Once written, the open may have reached the provider whatever happens next.
            if let Err(e) = write_message(&mut wr, &open).await {
                let _ = opened_tx.send(Err(PushOpenError::Lost(e.to_string())));
                return;
            }
            let _ = opened_tx.send(Ok(()));
            let reads = async {
                loop {
                    match read_message(&mut rd, MAX_MESSAGE).await {
                        Ok(Some(m)) => {
                            if inbox_tx.send(Ok(m)).is_err() {
                                return;
                            }
                        }
                        Ok(None) => {
                            let _ = inbox_tx.send(Err("push stream ended".into()));
                            return;
                        }
                        Err(e) => {
                            let _ = inbox_tx.send(Err(format!("push stream broke: {e}")));
                            return;
                        }
                    }
                }
            };
            let writes = async {
                while let Some(m) = outbox_rx.recv().await {
                    if write_message(&mut wr, &m).await.is_err() {
                        break;
                    }
                }
                // Our end let go: wait for the reads (they end with the stream).
                std::future::pending::<()>().await;
            };
            tokio::select! {
                _ = reads => {}
                _ = writes => {}
            }
        });
        let task = task.abort_handle();
        match opened_rx.recv_timeout(timeout + Duration::from_secs(1)) {
            Ok(Ok(())) => Ok(PushStream { inbox, outbox, task }),
            Ok(Err(e)) => {
                task.abort();
                Err(e)
            }
            Err(_) => {
                task.abort();
                Err(PushOpenError::NotSent("timed out opening".into()))
            }
        }
    }
}

/// Read one message (`None`: the stream ended cleanly before one began). Grows the buffer as the
/// bytes come, so a length claimed but never sent costs nothing.
async fn read_message<R: AsyncRead + Unpin>(r: &mut R, max: usize) -> std::io::Result<Option<Vec<u8>>> {
    let mut len = [0u8; 4];
    let mut got = 0;
    while got < len.len() {
        let n = r.read(&mut len[got..]).await?;
        if n == 0 {
            return if got == 0 { Ok(None) } else { Err(std::io::ErrorKind::UnexpectedEof.into()) };
        }
        got += n;
    }
    let len = u32::from_be_bytes(len) as usize;
    if len > max {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("message of {len} bytes, over {max}")));
    }
    let mut buf = Vec::with_capacity(len.min(64 * 1024));
    r.take(len as u64).read_to_end(&mut buf).await?;
    if buf.len() != len {
        return Err(std::io::ErrorKind::UnexpectedEof.into());
    }
    Ok(Some(buf))
}

/// Write one message and flush it.
async fn write_message<W: AsyncWrite + Unpin>(w: &mut W, m: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(m.len()).map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let mut framed = Vec::with_capacity(4 + m.len());
    framed.extend_from_slice(&len.to_be_bytes());
    framed.extend_from_slice(m);
    w.write_all(&framed).await?;
    w.flush().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn messages_round_trip_and_a_clean_end_is_none() {
        let mut buf = Vec::new();
        write_message(&mut buf, b"hello").await.unwrap();
        write_message(&mut buf, b"").await.unwrap();
        let mut r = futures::io::Cursor::new(buf);
        assert_eq!(read_message(&mut r, 100).await.unwrap(), Some(b"hello".to_vec()));
        assert_eq!(read_message(&mut r, 100).await.unwrap(), Some(Vec::new()));
        assert_eq!(read_message(&mut r, 100).await.unwrap(), None);
    }

    #[tokio::test]
    async fn an_oversized_or_cut_message_is_an_error() {
        let mut buf = Vec::new();
        write_message(&mut buf, &[7u8; 50]).await.unwrap();
        assert!(read_message(&mut futures::io::Cursor::new(buf.clone()), 49).await.is_err());
        buf.truncate(30);
        assert!(read_message(&mut futures::io::Cursor::new(buf), 100).await.is_err());
        // A length claimed with nothing after it.
        assert!(read_message(&mut futures::io::Cursor::new(vec![0, 0, 0, 9]), 100).await.is_err());
        assert!(read_message(&mut futures::io::Cursor::new(vec![0, 0]), 100).await.is_err());
    }

    #[test]
    fn ids_are_told_apart() {
        assert!(is_push_id("push:4"));
        assert!(!is_push_id("push-in:4"));
        assert!(is_push_message_id("push-in:4"));
        assert!(!is_push_message_id("push:4"));
        assert!(!is_push_id("17") && !is_push_message_id("17"));
    }
}
