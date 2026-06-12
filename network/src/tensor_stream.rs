//! Persistent tensor streams over libp2p-stream (Fix 1).
//!
//! Replaces the per-token `request_response::send_request()` pattern with
//! long-lived bidirectional streams. One stream is cached per peer and
//! reused across all tokens, eliminating the multistream-select negotiation
//! overhead that dominated QUIC cross-ISP latency (~360ms per token at
//! 180ms RTT).
//!
//! Wire format: 4-byte big-endian length prefix + payload (same framing
//! as `GrpcProxyCodec` in proxy.rs).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::prelude::*;
use libp2p::{PeerId, Stream, StreamProtocol};
use libp2p_stream::Control;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use crate::event_loop::SharedProxyQueue;

/// Write-half of a split tensor stream. Audit F4: inbound streams are split
/// into independent read/write halves so the reader loop and `write_response`
/// never contend on a single mutex held across a blocking read. `futures`'
/// `split()` uses a `BiLock` that is released between polls, so a response can
/// be written while the reader is parked waiting for the next frame.
type TensorWriteHalf = futures::io::WriteHalf<Stream>;

/// Map of inbound response handles: request_id → shared write-half of the
/// stream the request arrived on. All request_ids from one physical stream
/// share one `Arc<Mutex<TensorWriteHalf>>`.
pub type InboundStreamMap = HashMap<String, Arc<Mutex<TensorWriteHalf>>>;

/// The libp2p stream protocol for persistent tensor transfer.
pub const TENSOR_STREAM_PROTOCOL: StreamProtocol =
    StreamProtocol::new("/openhydra/tensor-stream/1.0.0");

/// Maximum message size (same as GrpcProxyCodec).
const MAX_MSG_SIZE: usize = 100 * 1024 * 1024; // 100 MB

/// Hard ceiling timeout for stalled writes (Fix 4).
/// OS socket errors (BrokenPipe, ConnectionReset) catch most failures
/// instantly. This timeout is only for connections that stall without
/// erroring — it is NOT a per-write budget.
///
/// WS-H: raised 250ms → 2s. The old 250ms silently dropped large activations
/// on high-RTT cross-ISP links: a 9B/27B activation (256 KB+) over a ~285ms
/// RTT path needs several round-trips in TCP slow-start (~2s worst case), so a
/// blocking `write()` legitimately exceeds 250ms. Dropping that write kills the
/// whole ring generation (120s step-0 timeout) — far worse than waiting. Small
/// LAN writes return in <1ms and never approach this ceiling, so a higher value
/// is safe; it only changes how fast a genuinely *stalled* (silent) write is
/// abandoned, and OS socket errors already catch real disconnects instantly.
const WRITE_TIMEOUT: Duration = Duration::from_millis(2000);

/// Read timeout for request-response mode. If the remote peer doesn't
/// send a response within this window, the cached stream is discarded
/// and the error is propagated. Set higher than WRITE_TIMEOUT since
/// the remote needs time to process the request.
const READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Cooldown between QUIC re-probe attempts after Degraded state (Fix 4).
const DEGRADED_REPROBE_INTERVAL: Duration = Duration::from_secs(30);

// ── Fix 4: transport preference ─────────────────────────────────────────

/// Per-peer transport preference state.
#[derive(Debug, Clone)]
enum PreferredTransport {
    /// QUIC-direct is available and working.
    QuicDirect,
    /// Degraded: QUIC failed, using fallback. After cooldown, retry QUIC.
    Degraded { since: Instant },
}

/// Manager for persistent outbound tensor streams.
///
/// Fix 1: caches one outbound `Stream` per peer, reused across tokens.
/// Fix 4: transport-aware routing with QUIC → TCP-direct → TCP-relay
/// fallback and debounced re-punch on QUIC failure.
pub struct TensorStreamManager {
    control: Mutex<Control>,
    /// Cached outbound streams, one per peer.
    outbound: Mutex<HashMap<PeerId, Stream>>,
    /// Fix 4: per-peer transport preference.
    preferences: Mutex<HashMap<PeerId, PreferredTransport>>,
    /// Fix 4: channel to send TriggerRepunch commands back to the event loop.
    repunch_tx: mpsc::UnboundedSender<PeerId>,
    /// Phase 5.4: per-peer RTT estimates for adaptive write timeout.
    /// WS-H: a std (sync) Mutex — it's locked only for a brief map get/insert
    /// and is never held across an await, so it can be updated from the SYNC
    /// `handle_swarm_event` ping handler (where `update_rtt` was previously
    /// uncallable, which is why RTT was never populated and every peer silently
    /// fell back to the 250ms default).
    peer_rtt: std::sync::Mutex<HashMap<PeerId, Duration>>,
}

impl TensorStreamManager {
    pub fn new(control: Control, repunch_tx: mpsc::UnboundedSender<PeerId>) -> Self {
        Self {
            control: Mutex::new(control),
            outbound: Mutex::new(HashMap::new()),
            preferences: Mutex::new(HashMap::new()),
            repunch_tx,
            peer_rtt: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Phase 5.4: Update the RTT estimate for a peer (called from the event
    /// loop on ping success). WS-H: now sync so the sync ping handler can call it.
    pub fn update_rtt(&self, peer: &PeerId, rtt: Duration) {
        if let Ok(mut m) = self.peer_rtt.lock() {
            m.insert(*peer, rtt);
        }
    }

    /// Phase 5.4: Compute write timeout for a peer, adaptive to observed RTT.
    /// Defaults to WRITE_TIMEOUT (2s) if no RTT estimate available.
    fn write_timeout_for(&self, peer: &PeerId) -> Duration {
        let rtt = self.peer_rtt.lock().ok().and_then(|m| m.get(peer).copied());
        match rtt {
            Some(rtt) => {
                // WS-H: ×5 (not ×3) to cover TCP slow-start of a large first
                // activation, floored at WRITE_TIMEOUT (2s) so a known-but-low
                // RTT can't drop us back below the safe cross-ISP ceiling, and
                // capped at 10s for pathological links.
                let adaptive = Duration::from_millis(
                    (rtt.as_millis() as u64 * 5).max(WRITE_TIMEOUT.as_millis() as u64),
                );
                adaptive.min(Duration::from_secs(10))
            }
            None => WRITE_TIMEOUT,
        }
    }

    /// Send tensor data to a peer (fire-and-forget, no response expected).
    ///
    /// Fix 4 routing: QUIC-direct → TCP-direct → TCP-relay.
    /// On QUIC failure: marks Degraded, triggers debounced re-punch.
    /// After 30s cooldown, optimistically retries QUIC.
    pub async fn send_tensor(
        &self,
        peer: &PeerId,
        data: &[u8],
    ) -> Result<(), TensorStreamError> {
        // Check if we should retry QUIC after degradation cooldown.
        self.maybe_reprobe_quic(peer).await;

        // Try the primary path (cached stream).
        let result = self.try_send_with_timeout(peer, data).await;
        match result {
            Ok(()) => return Ok(()),
            Err(TensorStreamError::NoCachedStream) => {
                // No cached stream — open a new one.
            }
            Err(e) => {
                // Write failed — mark degraded if QUIC, remove stale stream.
                debug!(%peer, %e, "tensor_stream: cached stream failed, reopening");
                self.handle_send_failure(peer).await;
            }
        }

        // Open a new stream and send.
        match self.open_and_cache_stream(peer).await {
            Ok(()) => self.try_send_with_timeout(peer, data).await,
            Err(e) => {
                warn!(%peer, %e, "tensor_stream: open_stream failed");
                Err(e)
            }
        }
    }

    /// Send tensor data and read a response (request-response mode).
    pub async fn send_tensor_with_reply(
        &self,
        peer: &PeerId,
        data: &[u8],
    ) -> Result<Vec<u8>, TensorStreamError> {
        self.maybe_reprobe_quic(peer).await;

        let result = self.try_send_recv_on_cached(peer, data).await;
        match result {
            Ok(resp) => return Ok(resp),
            Err(TensorStreamError::NoCachedStream) => {}
            Err(e) => {
                debug!(%peer, %e, "tensor_stream: cached stream failed (rr), reopening");
                self.handle_send_failure(peer).await;
            }
        }

        match self.open_and_cache_stream(peer).await {
            Ok(()) => self.try_send_recv_on_cached(peer, data).await,
            Err(e) => Err(e),
        }
    }

    /// Pre-open a tensor stream to a peer (Fix 4: proactive warming).
    ///
    /// Called from ConnectionEstablished when a QUIC-direct connection
    /// is established. The first token doesn't pay `open_stream()` latency.
    pub async fn warm_stream(&self, peer: &PeerId) {
        // Only warm if we don't already have a cached stream.
        if self.outbound.lock().await.contains_key(peer) {
            return;
        }
        match self.open_and_cache_stream(peer).await {
            Ok(()) => {
                info!(%peer, "tensor_stream: warmed stream proactively");
                // Mark QUIC as preferred.
                self.preferences
                    .lock()
                    .await
                    .insert(*peer, PreferredTransport::QuicDirect);
            }
            Err(e) => {
                debug!(%peer, %e, "tensor_stream: warm_stream failed (non-fatal)");
            }
        }
    }

    /// Remove cached stream for a peer (call on ConnectionClosed).
    pub async fn remove_peer(&self, peer: &PeerId) {
        if self.outbound.lock().await.remove(peer).is_some() {
            debug!(%peer, "tensor_stream: removed cached stream (peer disconnected)");
        }
        self.preferences.lock().await.remove(peer);
    }

    /// Get the current transport preference for a peer (for observability).
    pub async fn get_preference(&self, peer: &PeerId) -> String {
        match self.preferences.lock().await.get(peer) {
            Some(PreferredTransport::QuicDirect) => "quic_direct".to_string(),
            Some(PreferredTransport::Degraded { since }) => {
                format!("degraded_{}s", since.elapsed().as_secs())
            }
            None => "unknown".to_string(),
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────

    // Phase 5.3: Clone the Control handle and release the lock immediately.
    // Without this, the control mutex is held while waiting for the remote
    // peer to accept the stream — one slow peer blocks all other peers'
    // stream opens.
    async fn open_stream(&self, peer: &PeerId) -> Result<Stream, TensorStreamError> {
        let mut control = self.control.lock().await.clone();
        // Lock released — other peers can open streams concurrently.
        match control.open_stream(*peer, TENSOR_STREAM_PROTOCOL).await {
            Ok(stream) => {
                debug!(%peer, "tensor_stream: opened new stream");
                Ok(stream)
            }
            Err(e) => Err(TensorStreamError::OpenFailed(format!("{e}"))),
        }
    }

    // Phase 5.1: Check-before-open prevents stream leaks when two concurrent
    // send_tensor() calls both hit NoCachedStream and race to open.
    async fn open_and_cache_stream(&self, peer: &PeerId) -> Result<(), TensorStreamError> {
        // If another task already opened a stream while we were waiting, reuse it.
        if self.outbound.lock().await.contains_key(peer) {
            return Ok(());
        }
        let stream = self.open_stream(peer).await?;
        // Double-check: another task may have won the race.
        let mut map = self.outbound.lock().await;
        if map.contains_key(peer) {
            // Another task opened a stream while we were opening ours.
            // Drop our stream (it will be properly closed by Drop).
            debug!(%peer, "tensor_stream: race detected, discarding duplicate stream");
            return Ok(());
        }
        map.insert(*peer, stream);
        Ok(())
    }

    /// Try to send on the cached stream with an adaptive timeout ceiling.
    /// Phase 5.4: timeout adapts to observed RTT (min 250ms, default 250ms).
    async fn try_send_with_timeout(
        &self,
        peer: &PeerId,
        data: &[u8],
    ) -> Result<(), TensorStreamError> {
        let mut stream = {
            let mut map = self.outbound.lock().await;
            map.remove(peer).ok_or(TensorStreamError::NoCachedStream)?
        };
        let timeout = self.write_timeout_for(peer);
        let result = match tokio::time::timeout(timeout, write_framed(&mut stream, data)).await {
            Ok(result) => result,
            Err(_) => Err(TensorStreamError::Write(
                format!("write timed out ({}ms)", timeout.as_millis()),
            )),
        };
        if result.is_ok() {
            self.outbound.lock().await.insert(*peer, stream);
        }
        result
    }

    /// Try to send + receive on the cached stream.
    ///
    /// Takes the stream out of the cache during the operation to avoid
    /// holding the Mutex across the read (which can block for seconds
    /// while the remote processes the request). Re-inserts on success.
    async fn try_send_recv_on_cached(
        &self,
        peer: &PeerId,
        data: &[u8],
    ) -> Result<Vec<u8>, TensorStreamError> {
        let mut stream = {
            let mut map = self.outbound.lock().await;
            map.remove(peer).ok_or(TensorStreamError::NoCachedStream)?
        };

        let timeout = self.write_timeout_for(peer);
        match tokio::time::timeout(timeout, write_framed(&mut stream, data)).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(TensorStreamError::Write(
                format!("write timed out ({}ms)", timeout.as_millis()),
            )),
        }

        let result = match tokio::time::timeout(READ_TIMEOUT, read_framed(&mut stream)).await {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(TensorStreamError::Read("read timed out (30s)".into())),
        };

        if result.is_ok() {
            self.outbound.lock().await.insert(*peer, stream);
        }
        result
    }

    /// Handle a send failure: remove the dead stream and trigger re-punch
    /// if the peer was using QUIC.
    ///
    /// Phase 5.2: Hold both locks in consistent order (outbound first, then
    /// preferences) to prevent TOCTOU between releasing outbound and acquiring
    /// preferences — a concurrent sender could open a new QUIC stream in
    /// the gap, defeating the degradation intent.
    async fn handle_send_failure(&self, peer: &PeerId) {
        let mut outbound = self.outbound.lock().await;
        let mut prefs = self.preferences.lock().await;
        outbound.remove(peer);
        if let Some(PreferredTransport::QuicDirect) = prefs.get(peer) {
            info!(%peer, "tensor_stream: QUIC failed, marking degraded, triggering re-punch");
            prefs.insert(*peer, PreferredTransport::Degraded { since: Instant::now() });
            // Fire-and-forget re-punch request (debounced in event loop).
            let _ = self.repunch_tx.send(*peer);
        }
    }

    /// If peer is in Degraded state and cooldown has elapsed, clear the
    /// preference so the next send attempt tries QUIC again.
    async fn maybe_reprobe_quic(&self, peer: &PeerId) {
        let mut prefs = self.preferences.lock().await;
        if let Some(PreferredTransport::Degraded { since }) = prefs.get(peer) {
            if since.elapsed() > DEGRADED_REPROBE_INTERVAL {
                info!(%peer, "tensor_stream: degraded cooldown elapsed, will retry QUIC");
                prefs.remove(peer);
            }
        }
    }
}

/// Spawn the inbound tensor stream acceptor.
///
/// Accepts incoming streams on `/openhydra/tensor-stream/1.0.0` and for
/// each one spawns a reader loop that pushes messages to the shared
/// proxy queue (same queue that `poll_proxy_request` drains from Python).
///
/// Returns a HashMap for storing response stream handles, and a JoinHandle.
pub fn spawn_inbound_acceptor(
    mut incoming: libp2p_stream::IncomingStreams,
    proxy_queue: Arc<SharedProxyQueue>,
    inbound_streams: Arc<Mutex<InboundStreamMap>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        info!("tensor_stream_acceptor: accepting inbound streams");
        while let Some((peer_id, stream)) = incoming.next().await {
            info!(%peer_id, "tensor_stream_acceptor: inbound stream accepted");
            let pq = Arc::clone(&proxy_queue);
            let streams = Arc::clone(&inbound_streams);
            tokio::spawn(handle_inbound_stream(peer_id, stream, pq, streams));
        }
        info!("tensor_stream_acceptor: stopped");
    })
}

/// Handle one inbound tensor stream: read framed messages in a loop,
/// push each to the proxy queue.
///
/// Audit F4: the stream is split into independent read/write halves. The
/// reader loop owns the read half exclusively (no shared mutex held across
/// the blocking `read_framed`), while a single shared `Arc<Mutex<WriteHalf>>`
/// backs every response for this stream. A response can therefore be written
/// while the reader is parked waiting for the next frame.
///
/// We register a write-handle in `inbound_streams` only for messages that
/// expect a response. Fire-and-forget frames (the ring hot path) never get a
/// `respond_proxy` callback, so registering one would leak an entry per token
/// for the life of the stream. On loop exit we purge every request_id this
/// stream registered, so nothing lingers after the peer disconnects.
async fn handle_inbound_stream(
    peer_id: PeerId,
    stream: Stream,
    proxy_queue: Arc<SharedProxyQueue>,
    inbound_streams: Arc<Mutex<InboundStreamMap>>,
) {
    use futures::AsyncReadExt;
    let (mut read_half, write_half) = stream.split();
    let write_half = Arc::new(Mutex::new(write_half));
    // request_ids this stream registered a response handle for, so we can
    // purge any that were never answered when the stream closes.
    let mut registered: Vec<String> = Vec::new();

    loop {
        let data = match read_framed(&mut read_half).await {
            Ok(data) => data,
            Err(e) => {
                debug!(%peer_id, %e, "tensor_stream_inbound: read error, closing");
                break;
            }
        };

        // Generate a unique request ID for this inbound message.
        let req_id = format!("ts-{}-{}", peer_id, uuid_short());

        // Only register a write-handle when the message expects a response.
        // Fire-and-forget methods (0x03/0x04) never trigger respond_proxy.
        let method = data.first().copied().unwrap_or(0);
        let is_fire_forget = method == crate::dispatcher::METHOD_FIRE_FORGET
            || method == crate::dispatcher::METHOD_FIRE_FORGET_RESULT;
        if !is_fire_forget {
            inbound_streams
                .lock()
                .await
                .insert(req_id.clone(), Arc::clone(&write_half));
            registered.push(req_id.clone());
        }

        proxy_queue.push((req_id, data));
    }

    // Purge any unanswered response handles registered by this stream.
    if !registered.is_empty() {
        let mut map = inbound_streams.lock().await;
        for rid in &registered {
            map.remove(rid);
        }
    }
}

/// Write a length-prefixed frame to a stream (or write-half of one).
async fn write_framed<W: AsyncWrite + Unpin>(
    stream: &mut W,
    data: &[u8],
) -> Result<(), TensorStreamError> {
    use futures::AsyncWriteExt;
    let len = data.len() as u32;
    let len_bytes = len.to_be_bytes();
    stream
        .write_all(&len_bytes)
        .await
        .map_err(|e| TensorStreamError::Write(e.to_string()))?;
    stream
        .write_all(data)
        .await
        .map_err(|e| TensorStreamError::Write(e.to_string()))?;
    stream
        .flush()
        .await
        .map_err(|e| TensorStreamError::Write(e.to_string()))?;
    Ok(())
}

/// Read a length-prefixed frame from a stream (or read-half of one).
async fn read_framed<R: AsyncRead + Unpin>(stream: &mut R) -> Result<Vec<u8>, TensorStreamError> {
    use futures::AsyncReadExt;
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| TensorStreamError::Read(e.to_string()))?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_MSG_SIZE {
        return Err(TensorStreamError::Read(format!(
            "message too large: {len} > {MAX_MSG_SIZE}"
        )));
    }
    let mut data = vec![0u8; len];
    stream
        .read_exact(&mut data)
        .await
        .map_err(|e| TensorStreamError::Read(e.to_string()))?;
    Ok(data)
}

/// Generate a short random ID for inbound request tracking.
fn uuid_short() -> String {
    use rand::Rng;
    let n: u64 = rand::thread_rng().gen();
    format!("{:016x}", n)
}

/// Write a response on an inbound tensor stream (called from RespondProxy).
pub async fn write_response(
    inbound_streams: &Mutex<InboundStreamMap>,
    req_id: &str,
    data: &[u8],
) -> Result<(), TensorStreamError> {
    let write_half = {
        let mut map = inbound_streams.lock().await;
        match map.remove(req_id) {
            Some(s) => s,
            None => return Err(TensorStreamError::UnknownReqId(req_id.to_string())),
        }
    };
    let mut w = write_half.lock().await;
    write_framed(&mut *w, data).await
}

#[derive(Debug, thiserror::Error)]
pub enum TensorStreamError {
    #[error("no cached stream for peer")]
    NoCachedStream,
    #[error("open_stream failed: {0}")]
    OpenFailed(String),
    #[error("write error: {0}")]
    Write(String),
    #[error("read error: {0}")]
    Read(String),
    #[error("unknown request_id: {0}")]
    UnknownReqId(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_name() {
        assert_eq!(
            TENSOR_STREAM_PROTOCOL.as_ref(),
            "/openhydra/tensor-stream/1.0.0"
        );
    }

    #[test]
    fn test_uuid_short_length() {
        let id = uuid_short();
        assert_eq!(id.len(), 16);
    }

    #[tokio::test]
    async fn test_framed_roundtrip() {
        // Test the framing codec using in-memory pipe.
        let (mut a, mut b) = futures::io::AsyncReadExt::split(futures::io::Cursor::new(vec![]));
        // We can't easily test with a real pipe, but we can test the byte format.
        let data = b"hello tensor world";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
        buf.extend_from_slice(data);
        assert_eq!(buf.len(), 4 + data.len());
        // Verify the length prefix decodes correctly.
        let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        assert_eq!(len, data.len());
        assert_eq!(&buf[4..], data);
        let _ = (&mut a, &mut b);
    }

    #[tokio::test]
    async fn test_write_then_read_framed_generic() {
        // Audit F4: write_framed / read_framed are now generic over any
        // AsyncWrite / AsyncRead (so they work on split write/read halves).
        // Round-trip through an in-memory buffer to confirm the framing.
        let payload = b"\x03some-fire-forget-activation-bytes";
        let mut sink: Vec<u8> = Vec::new();
        write_framed(&mut sink, payload).await.expect("write");
        assert_eq!(sink.len(), 4 + payload.len());

        let mut cursor = futures::io::Cursor::new(sink);
        let got = read_framed(&mut cursor).await.expect("read");
        assert_eq!(got, payload);
    }
}
