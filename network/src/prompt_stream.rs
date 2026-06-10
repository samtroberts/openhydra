//! Stream-based prompt protocol for supernode token streaming.
//!
//! Uses `libp2p_stream::Behaviour` (bidirectional streams) instead of
//! request_response to enable true token-by-token streaming and avoid
//! blocking a Python thread pool thread for the full inference duration.
//!
//! Framing: each message is `[4 bytes big-endian length][payload]`,
//! same as the existing `PromptCodec`.
//!
//! Server side: accepts streams, reads one request, queues for Python,
//! reads response chunks from a bounded channel, writes to stream.
//!
//! Client side: opens a stream via SwarmCommand, writes request, spawns
//! a reader task that pushes chunks to a per-stream queue for Python to
//! poll.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::{AsyncReadExt, AsyncWriteExt};
use libp2p::StreamProtocol;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::event_loop::SharedProxyQueue;

/// The libp2p stream protocol for prompt streaming.
pub const PROMPT_STREAM_PROTOCOL: StreamProtocol =
    StreamProtocol::new("/openhydra/prompt-stream/1.0.0");

/// Max message size: 10 MB (same as PromptCodec).
const MAX_MESSAGE_SIZE: usize = 10 * 1024 * 1024;

/// Bounded channel capacity for server-side response chunks.
/// Applies backpressure when network is slower than inference.
const CHUNK_CHANNEL_CAPACITY: usize = 1024;

/// Server side: per-stream chunk senders (Python → Rust stream writer).
/// Bounded channel — `blocking_send()` blocks the Python thread when full,
/// pausing `adapter.generate()` and naturally throttling the GPU.
pub type PromptStreamWriters = Arc<Mutex<HashMap<String, mpsc::Sender<Vec<u8>>>>>;

/// Client side: per-stream chunk queues (Rust stream reader → Python poll).
pub type PromptStreamReaders = Arc<Mutex<HashMap<String, Arc<SharedProxyQueue>>>>;

/// Client side: per-stream cancel senders.
pub type PromptStreamCancels = Arc<Mutex<HashMap<String, tokio::sync::watch::Sender<bool>>>>;

// ── Framing helpers ─────────────────────────────────────────────────

/// Read one length-prefixed message from a futures::AsyncRead stream.
async fn read_message(
    reader: &mut (impl futures::AsyncRead + Unpin),
) -> Result<Vec<u8>, String> {
    let mut len_buf = [0u8; 4];
    reader
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| format!("read length: {e}"))?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_MESSAGE_SIZE {
        return Err(format!("message too large: {len} > {MAX_MESSAGE_SIZE}"));
    }
    let mut buf = vec![0u8; len];
    reader
        .read_exact(&mut buf)
        .await
        .map_err(|e| format!("read payload: {e}"))?;
    Ok(buf)
}

/// Write one length-prefixed message to a futures::AsyncWrite stream.
async fn write_message(
    writer: &mut (impl futures::AsyncWrite + Unpin),
    data: &[u8],
) -> Result<(), String> {
    let len = (data.len() as u32).to_be_bytes();
    writer
        .write_all(&len)
        .await
        .map_err(|e| format!("write length: {e}"))?;
    writer
        .write_all(data)
        .await
        .map_err(|e| format!("write payload: {e}"))?;
    writer
        .flush()
        .await
        .map_err(|e| format!("flush: {e}"))?;
    Ok(())
}

// ── Server side ─────────────────────────────────────────────────────

/// Spawn the prompt stream responder: accept inbound streams, read the
/// request, queue for Python, then relay response chunks from a bounded
/// channel back to the stream.
pub fn spawn_prompt_stream_responder(
    mut incoming: libp2p_stream::IncomingStreams,
    prompt_queue: Arc<SharedProxyQueue>,
    stream_writers: PromptStreamWriters,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        use futures::StreamExt;
        info!("prompt_stream_responder_started");
        let mut counter: u64 = 0;
        while let Some((peer, stream)) = incoming.next().await {
            counter += 1;
            let stream_id = format!("pstream-{counter}");
            info!(%peer, id = %stream_id, "prompt_stream: inbound stream accepted");

            let queue = Arc::clone(&prompt_queue);
            let writers = Arc::clone(&stream_writers);
            let sid = stream_id.clone();

            tokio::spawn(async move {
                if let Err(e) =
                    handle_inbound_prompt_stream(peer, stream, sid.clone(), queue, writers.clone())
                        .await
                {
                    warn!(id = %sid, error = %e, "prompt_stream: inbound handler error");
                }
                writers.lock().unwrap().remove(&sid);
                debug!(id = %sid, "prompt_stream: inbound handler finished");
            });
        }
        info!("prompt_stream_responder_stopped");
    })
}

/// Handle one inbound prompt stream (server side).
async fn handle_inbound_prompt_stream(
    peer: libp2p::PeerId,
    stream: libp2p::Stream,
    stream_id: String,
    prompt_queue: Arc<SharedProxyQueue>,
    stream_writers: PromptStreamWriters,
) -> Result<(), String> {
    let (mut reader, mut writer) = AsyncReadExt::split(stream);

    // 1. Read the request (one length-prefixed message).
    let request_data = read_message(&mut reader).await?;
    debug!(
        %peer, id = %stream_id, bytes = request_data.len(),
        "prompt_stream: request read"
    );

    // 2. Create bounded channel for Python to send response chunks.
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(CHUNK_CHANNEL_CAPACITY);
    {
        let mut writers = stream_writers.lock().unwrap();
        writers.insert(stream_id.clone(), tx);
    }

    // 3. Queue the request for Python (same queue as request-response prompts).
    prompt_queue.push((stream_id.clone(), request_data));

    // 4. Read chunks from the channel and write to the stream.
    while let Some(chunk) = rx.recv().await {
        if let Err(e) = write_message(&mut writer, &chunk).await {
            warn!(id = %stream_id, error = %e, "prompt_stream: write chunk failed");
            break;
        }
    }

    // 5. Close the write half.
    let _ = writer.close().await;
    Ok(())
}

// ── Client side ─────────────────────────────────────────────────────

/// Open a prompt stream to a remote peer (client side).
///
/// 1. Opens a libp2p stream via `stream_control`.
/// 2. Writes the request with length prefix.
/// 3. Spawns a reader task that pushes chunks to a per-stream queue.
/// 4. Returns the stream_id for polling.
pub async fn open_prompt_stream(
    mut control: libp2p_stream::Control,
    peer_id: libp2p::PeerId,
    request_data: Vec<u8>,
    stream_readers: PromptStreamReaders,
    stream_cancels: PromptStreamCancels,
    stream_id: String,
) -> Result<(), String> {
    // Open the stream.
    let stream = control
        .open_stream(peer_id, PROMPT_STREAM_PROTOCOL)
        .await
        .map_err(|e| format!("open_stream to {peer_id}: {e}"))?;

    let (mut reader, mut writer) = AsyncReadExt::split(stream);

    // Write the request.
    write_message(&mut writer, &request_data)
        .await
        .map_err(|e| format!("write request: {e}"))?;

    // Create per-stream queue for response chunks.
    let chunk_queue = Arc::new(SharedProxyQueue::new());
    let (cancel_tx, mut cancel_rx) = tokio::sync::watch::channel(false);

    {
        let mut readers = stream_readers.lock().unwrap();
        readers.insert(stream_id.clone(), Arc::clone(&chunk_queue));
    }
    {
        let mut cancels = stream_cancels.lock().unwrap();
        cancels.insert(stream_id.clone(), cancel_tx);
    }

    // Spawn reader task: reads length-prefixed chunks, pushes to queue.
    let sid = stream_id.clone();
    let readers_ref = Arc::clone(&stream_readers);
    let cancels_ref = Arc::clone(&stream_cancels);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = cancel_rx.changed() => {
                    debug!(id = %sid, "prompt_stream: client reader cancelled");
                    break;
                }
                result = read_message(&mut reader) => {
                    match result {
                        Ok(chunk) => {
                            chunk_queue.push((sid.clone(), chunk));
                        }
                        Err(e) => {
                            debug!(id = %sid, error = %e, "prompt_stream: client reader finished");
                            break;
                        }
                    }
                }
            }
        }
        // Push empty sentinel to signal end-of-stream.
        chunk_queue.push((sid.clone(), Vec::new()));
        // Don't remove from readers/cancels here — Python's poll_prompt_chunk
        // may not have drained the queue yet. Cleanup happens when Python
        // calls close_prompt_stream.
        cancels_ref.lock().unwrap().remove(&sid);
    });

    info!(%peer_id, id = %stream_id, "prompt_stream: client stream opened");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_name() {
        assert_eq!(
            PROMPT_STREAM_PROTOCOL.as_ref(),
            "/openhydra/prompt-stream/1.0.0"
        );
    }

    #[tokio::test]
    async fn test_framing_roundtrip() {
        let data = b"hello world";
        let mut buf = Vec::new();
        write_message(&mut buf, data).await.unwrap();

        let mut cursor = futures::io::Cursor::new(buf);
        let read_back = read_message(&mut cursor).await.unwrap();
        assert_eq!(read_back, data);
    }

    #[tokio::test]
    async fn test_framing_empty() {
        let data = b"";
        let mut buf = Vec::new();
        write_message(&mut buf, data).await.unwrap();

        let mut cursor = futures::io::Cursor::new(buf);
        let read_back = read_message(&mut cursor).await.unwrap();
        assert_eq!(read_back, b"");
    }

    #[tokio::test]
    async fn test_framing_max_size_rejected() {
        // Craft a length header claiming 11 MB.
        let len = (11 * 1024 * 1024u32).to_be_bytes();
        let mut buf = len.to_vec();
        buf.extend_from_slice(&[0u8; 16]); // some payload
        let mut cursor = futures::io::Cursor::new(buf);
        let result = read_message(&mut cursor).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too large"));
    }
}
