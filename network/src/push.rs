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
/// How long a provider waits for the whole open after the stream is negotiated: a request's own
/// time limit (an open carries the whole conversation, slow to send over a relay).
const OPEN_READ_TIMEOUT: Duration = Duration::from_secs(30);
/// Push streams whose open is still on its way, per consumer and in all: a stream that never sends
/// one holds no more than this (past it, new ones are refused and polled).
const MAX_OPENING_PER_PEER: usize = 16;
const MAX_OPENING_TOTAL: usize = 128;
/// Push streams one consumer may hold open at a provider at once (its stream cap, `provider.rs`
/// `MAX_STREAMS_PER_PEER`), and all consumers together. Past them a stream is refused, and its
/// consumer sends its open by request instead.
pub const MAX_PUSH_PER_PEER: usize = 128;
pub const MAX_PUSH_TOTAL: usize = 1024;
/// Refusals being written at once; past it a stream over the limits is dropped without one (its
/// consumer then takes its open as lost, see [`PushOpenError::Lost`]).
const MAX_REFUSING: usize = 256;
/// How long a refused stream is kept, reading and dropping what its consumer still writes, so the
/// refusal reaches it before the stream goes.
const REFUSAL_LINGER: Duration = Duration::from_secs(5);
/// Messages waiting to be written to one consumer (a slow reader): past it the push ends and the
/// consumer polls for the rest.
const OUTBOX: usize = 1024;
/// How long one message may take to write before the push stream is given up (its consumer reads
/// too slowly, or is gone without the connection noticing).
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
/// A consumer's messages after its open closer together than this are dropped (it acknowledges
/// about twice a second).
const MIN_FOLLOW_UP_GAP: Duration = Duration::from_millis(250);
/// Pushed messages a consumer holds unread at most: past them its end stops reading the stream, so
/// the provider is held back by the transport.
const INBOX: usize = 8;

/// Whether `request_id` is a push stream's open (its responds go on the stream).
pub fn is_push_id(request_id: &str) -> bool {
    request_id.starts_with(PUSH_ID_PREFIX)
}

/// Whether `request_id` is a message a consumer wrote on its push stream after the open (never
/// answered).
pub fn is_push_message_id(request_id: &str) -> bool {
    request_id.starts_with(PUSH_MESSAGE_PREFIX)
}

/// The push stream (its open's id, [`is_push_id`]) a message written on it came on
/// ([`is_push_message_id`]).
pub fn push_id_of_message(message_id: &str) -> Option<String> {
    message_id.strip_prefix(PUSH_MESSAGE_PREFIX).map(|n| format!("{PUSH_ID_PREFIX}{n}"))
}

/// Why a push stream couldn't be opened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushOpenError {
    /// The peer doesn't speak the protocol (an older build): poll it.
    Unsupported,
    /// The open never reached the peer's agent (no connection, the stream couldn't be negotiated,
    /// or the peer refused it unread): send it the usual way.
    NotSent(String),
    /// The open was (perhaps partly) written, and then the stream failed: the peer may have begun
    /// serving it.
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

/// Why nothing more came from a push stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushRecvError {
    /// The provider refused the stream without handing its open to its agent: nothing was served,
    /// and the open may be sent the usual way.
    Refused,
    /// It ended, broke, or nothing came in time (it is then no use: drop it).
    Gone(String),
}

impl std::fmt::Display for PushRecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PushRecvError::Refused => write!(f, "push stream refused"),
            PushRecvError::Gone(e) => write!(f, "{e}"),
        }
    }
}

/// A consumer's end of a push stream: blocking reads of what the provider pushes.
pub struct PushStream {
    inbox: std::sync::mpsc::Receiver<Result<Vec<u8>, PushRecvError>>,
    /// Room for messages read ahead of the agent ([`INBOX`]); each one taken frees one.
    room: Arc<tokio::sync::Semaphore>,
    outbox: mpsc::UnboundedSender<Vec<u8>>,
    task: tokio::task::AbortHandle,
}

impl PushStream {
    /// The next message the provider pushed, waiting up to `timeout`.
    pub fn recv_timeout(&self, timeout: Duration) -> Result<Vec<u8>, PushRecvError> {
        match self.inbox.recv_timeout(timeout) {
            Ok(Ok(m)) => {
                self.room.add_permits(1);
                Ok(m)
            }
            Ok(Err(e)) => Err(e),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Err(PushRecvError::Gone(format!("nothing pushed in {timeout:?}"))),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => Err(PushRecvError::Gone("push stream ended".into())),
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

/// Streams held, per consumer.
#[derive(Default)]
struct Counts(HashMap<PeerId, usize>);

/// One of a consumer's streams in a [`Counts`], for as long as it lives.
struct Slot(Arc<Mutex<Counts>>, PeerId);

impl Slot {
    /// Count one more stream of `peer`'s, within `per_peer` of its own and `total` in all.
    fn take(counts: &Arc<Mutex<Counts>>, peer: PeerId, per_peer: usize, total: usize) -> Option<Slot> {
        let mut c = counts.lock().unwrap_or_else(|e| e.into_inner());
        let all: usize = c.0.values().sum();
        let mine = c.0.get(&peer).copied().unwrap_or(0);
        if mine >= per_peer || all >= total {
            return None;
        }
        c.0.insert(peer, mine + 1);
        Some(Slot(Arc::clone(counts), peer))
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        let mut c = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = c.0.get_mut(&self.1) {
            *n -= 1;
            if *n == 0 {
                c.0.remove(&self.1);
            }
        }
    }
}

/// A node's push streams.
pub struct PushStreams {
    control: Mutex<libp2p_stream::Control>,
    rt: tokio::runtime::Handle,
    local: PeerId,
    /// Provider side: the open push streams, by the id their open was handed to the agent under.
    open: Mutex<HashMap<String, Outbox>>,
    /// Provider side: push streams whose open is on its way, and those handed to the agent.
    opening: Arc<Mutex<Counts>>,
    serving: Arc<Mutex<Counts>>,
    /// Provider side: refusals being written now.
    refusing: Arc<std::sync::atomic::AtomicUsize>,
    next_id: AtomicU64,
    /// Provider side: push stream opens handed to the agent so far.
    served: AtomicU64,
}

/// Refuse a push stream: one empty message (a reply is never empty), then its end. What the
/// consumer still writes is read and dropped for [`REFUSAL_LINGER`], so the refusal reaches it.
async fn refuse<R: AsyncRead + Unpin, W: AsyncWrite + Unpin>(mut rd: R, mut wr: W) {
    let _ = tokio::time::timeout(WRITE_TIMEOUT, async {
        let _ = write_message(&mut wr, &[]).await;
        let _ = wr.close().await;
    })
    .await;
    let _ = tokio::time::timeout(REFUSAL_LINGER, async {
        let mut sink = [0u8; 4096];
        while matches!(rd.read(&mut sink).await, Ok(n) if n > 0) {}
    })
    .await;
}

impl PushStreams {
    /// Must be called inside the swarm's tokio runtime.
    pub fn new(control: libp2p_stream::Control, local: PeerId) -> Arc<Self> {
        Arc::new(Self {
            control: Mutex::new(control),
            rt: tokio::runtime::Handle::current(),
            local,
            open: Mutex::new(HashMap::new()),
            opening: Arc::default(),
            serving: Arc::default(),
            refusing: Arc::default(),
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
                match Slot::take(&me.opening, peer, MAX_OPENING_PER_PEER, MAX_OPENING_TOTAL) {
                    Some(slot) => {
                        let (me, queue) = (Arc::clone(&me), Arc::clone(&queue));
                        tokio::spawn(async move { me.serve(peer, stream, queue, slot).await });
                    }
                    None => {
                        debug!(%peer, "push stream refused: too many opening");
                        me.spawn_refusal(stream);
                    }
                }
            }
        });
        Ok(())
    }

    /// Refuse `stream` in the background, within [`MAX_REFUSING`] (past it, it is just dropped).
    fn spawn_refusal(&self, stream: libp2p::Stream) {
        if self.refusing.fetch_add(1, Ordering::AcqRel) >= MAX_REFUSING {
            self.refusing.fetch_sub(1, Ordering::AcqRel);
            return;
        }
        let refusing = Arc::clone(&self.refusing);
        tokio::spawn(async move {
            let (rd, wr) = stream.split();
            refuse(rd, wr).await;
            refusing.fetch_sub(1, Ordering::AcqRel);
        });
    }

    /// Provider side: one push stream, from its open to its end.
    async fn serve(self: Arc<Self>, peer: PeerId, stream: libp2p::Stream, queue: Arc<SharedProxyQueue>, opening: Slot) {
        let (mut rd, mut wr) = stream.split();
        let open = match tokio::time::timeout(OPEN_READ_TIMEOUT, read_message(&mut rd, MAX_MESSAGE)).await {
            Ok(Ok(Some(open))) => open,
            // Too slow: refused, so its consumer sends it the usual way.
            Err(_) => return refuse(rd, wr).await,
            // Ended or broke: there is nobody to tell.
            Ok(_) => return,
        };
        drop(opening);
        let Some(_serving) = Slot::take(&self.serving, peer, MAX_PUSH_PER_PEER, MAX_PUSH_TOTAL) else {
            debug!(%peer, "push stream refused: too many open");
            return refuse(rd, wr).await;
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
        // What the agent pushes, until it ends the push (drops its outbox) or a write fails or
        // stalls.
        let pushes = async move {
            while let Some(m) = rx.recv().await {
                if !matches!(tokio::time::timeout(WRITE_TIMEOUT, write_message(&mut wr, &m)).await, Ok(Ok(()))) {
                    return;
                }
            }
            let _ = tokio::time::timeout(WRITE_TIMEOUT, wr.close()).await;
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
    /// on it, both within `timeout`.
    pub fn open(&self, peer: &str, open: Vec<u8>, timeout: Duration) -> Result<PushStream, PushOpenError> {
        let peer: PeerId = peer.parse().map_err(|_| PushOpenError::NotSent("invalid peer_id".into()))?;
        if peer == self.local {
            return Err(PushOpenError::NotSent("own node".into()));
        }
        let mut control = self.control.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let (opened_tx, opened_rx) = std::sync::mpsc::channel::<Result<(), PushOpenError>>();
        let (inbox_tx, inbox) = std::sync::mpsc::channel::<Result<Vec<u8>, PushRecvError>>();
        let (outbox, mut outbox_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let room = Arc::new(tokio::sync::Semaphore::new(INBOX));
        // Set once any of the open may have gone out: from then on it may have been served.
        let writing = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task = {
            let (room, writing) = (Arc::clone(&room), Arc::clone(&writing));
            self.rt.spawn(async move {
                let deadline = tokio::time::Instant::now() + timeout;
                let stream = match tokio::time::timeout_at(deadline, control.open_connected_stream(peer, PUSH_PROTOCOL)).await {
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
                // Read from the start, alongside the write: a refusal may come before the open has
                // all gone out.
                let reads = {
                    let opened_tx = opened_tx.clone();
                    async move {
                        let mut first = true;
                        loop {
                            match room.acquire().await {
                                Ok(p) => p.forget(),
                                Err(_) => return,
                            }
                            let r = match read_message(&mut rd, MAX_MESSAGE).await {
                                Ok(Some(m)) if first && m.is_empty() => {
                                    let _ = opened_tx.send(Err(PushOpenError::NotSent("refused".into())));
                                    let _ = inbox_tx.send(Err(PushRecvError::Refused));
                                    return;
                                }
                                Ok(Some(m)) => Ok(m),
                                Ok(None) => Err(PushRecvError::Gone("push stream ended".into())),
                                Err(e) => Err(PushRecvError::Gone(format!("push stream broke: {e}"))),
                            };
                            first = false;
                            let end = r.is_err();
                            if inbox_tx.send(r).is_err() || end {
                                return;
                            }
                        }
                    }
                };
                let writes = async move {
                    writing.store(true, Ordering::Release);
                    let lost = match tokio::time::timeout_at(deadline, write_message(&mut wr, &open)).await {
                        Ok(Ok(())) => None,
                        Ok(Err(e)) => Some(e.to_string()),
                        Err(_) => Some("timed out writing".into()),
                    };
                    if let Some(e) = lost {
                        // A refusal may be why the write failed: give it a moment to arrive first.
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        let _ = opened_tx.send(Err(PushOpenError::Lost(e)));
                        return;
                    }
                    let _ = opened_tx.send(Ok(()));
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
            })
            .abort_handle()
        };
        match opened_rx.recv_timeout(timeout + Duration::from_secs(2)) {
            Ok(Ok(())) => Ok(PushStream { inbox, room, outbox, task }),
            Ok(Err(e)) => {
                task.abort();
                Err(e)
            }
            Err(_) => {
                task.abort();
                Err(if writing.load(Ordering::Acquire) {
                    PushOpenError::Lost("timed out writing".into())
                } else {
                    PushOpenError::NotSent("timed out opening".into())
                })
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

    #[tokio::test]
    async fn a_refusal_is_one_empty_message_then_the_end() {
        let mut wrote = Vec::new();
        refuse(futures::io::Cursor::new(vec![1u8; 10]), &mut wrote).await;
        let mut r = futures::io::Cursor::new(wrote);
        assert_eq!(read_message(&mut r, 100).await.unwrap(), Some(Vec::new()));
        assert_eq!(read_message(&mut r, 100).await.unwrap(), None);
    }

    #[test]
    fn slots_hold_a_peer_and_everyone_to_their_limits_and_free_on_drop() {
        let counts: Arc<Mutex<Counts>> = Arc::default();
        let (a, b) = (PeerId::random(), PeerId::random());
        let a1 = Slot::take(&counts, a, 2, 3).unwrap();
        let _a2 = Slot::take(&counts, a, 2, 3).unwrap();
        assert!(Slot::take(&counts, a, 2, 3).is_none(), "a's own limit");
        let _b1 = Slot::take(&counts, b, 2, 3).unwrap();
        assert!(Slot::take(&counts, b, 2, 3).is_none(), "the total");
        drop(a1);
        assert!(Slot::take(&counts, b, 2, 3).is_some());
    }

    #[test]
    fn ids_are_told_apart() {
        assert!(is_push_id("push:4"));
        assert!(!is_push_id("push-in:4"));
        assert!(is_push_message_id("push-in:4"));
        assert!(!is_push_message_id("push:4"));
        assert!(!is_push_id("17") && !is_push_message_id("17"));
        assert_eq!(push_id_of_message("push-in:4").as_deref(), Some("push:4"));
        assert_eq!(push_id_of_message("push:4"), None);
    }
}
