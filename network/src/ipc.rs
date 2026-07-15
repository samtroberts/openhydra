//! ZMQ IPC bridge — Rust ROUTER ↔ Python DEALER over Unix domain sockets.
//!
//! The bridge sends forward requests to the Python worker daemon and
//! receives activation responses.  This is the foundational CP-0
//! infrastructure that all subsequent Rust control plane phases build on.
//!
//! Socket pattern: ROUTER (Rust) ↔ DEALER (Python worker).
//! The ROUTER socket binds; the DEALER connects.  ROUTER can address
//! multiple DEALER workers by identity frame (future: CP-4 multi-session).
//!
//! Because we use the `zeromq` pure-Rust crate (async, tokio-native),
//! the IPC bridge runs on the existing tokio runtime alongside the
//! libp2p swarm.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, error, info};

use crate::ipc_codec::{
    self, IpcForwardHeader, IpcResponseHeader,
};

/// Default IPC socket path template.  `{peer_id}` is replaced at runtime.
const DEFAULT_SOCKET_TEMPLATE: &str = "/tmp/openhydra-worker-{peer_id}.sock";

/// Maximum time to wait for a Python worker response before timing out.
const DEFAULT_RECV_TIMEOUT: Duration = Duration::from_secs(120);

/// Response from the Python worker.
#[derive(Debug)]
pub struct IpcResponse {
    pub header: IpcResponseHeader,
    pub activation: Vec<u8>,
}

/// Commands for the IPC event loop.
enum IpcCommand {
    /// Send a forward request to the Python worker.
    Forward {
        header: IpcForwardHeader,
        activation: Vec<u8>,
        reply: oneshot::Sender<Result<IpcResponse, String>>,
    },
    /// Send a batch of forward requests to the Python worker (CP-4).
    ///
    /// The worker processes all items and returns all responses in one
    /// round-trip using the batch wire format (BATCH_MAGIC prefix).
    ForwardBatch {
        items: Vec<(IpcForwardHeader, Vec<u8>)>,
        reply: oneshot::Sender<Result<Vec<IpcResponse>, String>>,
    },
    /// Graceful shutdown.
    Shutdown,
}

/// Handle to the IPC bridge.  Clone-safe — all clones share the same
/// underlying command channel to the IPC event loop.
#[derive(Clone)]
pub struct IpcBridge {
    cmd_tx: mpsc::Sender<IpcCommand>,
    socket_path: PathBuf,
}

impl IpcBridge {
    /// Create and start a new IPC bridge.
    ///
    /// Spawns a tokio task that binds a ZMQ ROUTER socket at the given
    /// path and processes send/recv in a loop.
    pub async fn start(
        peer_id: &str,
        socket_path: Option<&str>,
        runtime: tokio::runtime::Handle,
    ) -> Result<Self, String> {
        let path = match socket_path {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from(
                DEFAULT_SOCKET_TEMPLATE.replace("{peer_id}", peer_id),
            ),
        };

        // Clean up stale socket file from previous run.
        if path.exists() {
            std::fs::remove_file(&path).ok();
        }

        let (cmd_tx, cmd_rx) = mpsc::channel::<IpcCommand>(256);

        let path_clone = path.clone();
        runtime.spawn(ipc_event_loop(path_clone, cmd_rx));

        info!(path = %path.display(), "IPC bridge started");

        Ok(Self {
            cmd_tx,
            socket_path: path,
        })
    }

    /// Start the IPC bridge synchronously (for use from non-async contexts).
    pub fn start_sync(
        peer_id: &str,
        socket_path: Option<&str>,
        runtime: &tokio::runtime::Handle,
    ) -> Result<Self, String> {
        let path = match socket_path {
            Some(p) => PathBuf::from(p),
            None => PathBuf::from(
                DEFAULT_SOCKET_TEMPLATE.replace("{peer_id}", peer_id),
            ),
        };

        // Clean up stale socket file from previous run.
        if path.exists() {
            std::fs::remove_file(&path).ok();
        }

        let (cmd_tx, cmd_rx) = mpsc::channel::<IpcCommand>(256);

        let path_clone = path.clone();
        runtime.spawn(ipc_event_loop(path_clone, cmd_rx));

        info!(path = %path.display(), "IPC bridge started (sync)");

        Ok(Self {
            cmd_tx,
            socket_path: path,
        })
    }

    /// Send a forward request and wait for the response.
    pub async fn forward(
        &self,
        header: IpcForwardHeader,
        activation: Vec<u8>,
    ) -> Result<IpcResponse, String> {
        let (reply_tx, reply_rx) = oneshot::channel();

        self.cmd_tx
            .send(IpcCommand::Forward {
                header,
                activation,
                reply: reply_tx,
            })
            .await
            .map_err(|_| "IPC bridge shut down".to_string())?;

        match tokio::time::timeout(DEFAULT_RECV_TIMEOUT, reply_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("IPC reply channel dropped".into()),
            Err(_) => Err("IPC forward timed out (120s)".into()),
        }
    }

    /// Send a batch of forward requests and wait for all responses.
    ///
    /// For single-item batches, delegates to the proven single-item path.
    /// For multi-item batches, uses the batch wire format (BATCH_MAGIC).
    pub async fn forward_batch(
        &self,
        items: Vec<(IpcForwardHeader, Vec<u8>)>,
    ) -> Result<Vec<IpcResponse>, String> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        if items.len() == 1 {
            // Single item: use the proven single-request path.
            let (header, activation) = items.into_iter().next().unwrap();
            let resp = self.forward(header, activation).await?;
            return Ok(vec![resp]);
        }

        let (reply_tx, reply_rx) = oneshot::channel();

        self.cmd_tx
            .send(IpcCommand::ForwardBatch {
                items,
                reply: reply_tx,
            })
            .await
            .map_err(|_| "IPC bridge shut down".to_string())?;

        match tokio::time::timeout(DEFAULT_RECV_TIMEOUT, reply_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("IPC batch reply channel dropped".into()),
            Err(_) => Err("IPC batch forward timed out (120s)".into()),
        }
    }

    /// Blocking version of `forward()` — for use from synchronous Python threads.
    pub fn forward_blocking(
        &self,
        header: IpcForwardHeader,
        activation: Vec<u8>,
        runtime: &tokio::runtime::Handle,
    ) -> Result<IpcResponse, String> {
        runtime.block_on(self.forward(header, activation))
    }

    /// Shut down the IPC bridge.
    pub async fn shutdown(&self) {
        let _ = self.cmd_tx.send(IpcCommand::Shutdown).await;
    }

    /// Get the socket path.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

impl Drop for IpcBridge {
    fn drop(&mut self) {
        // Best-effort cleanup of the socket file.
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path).ok();
        }
    }
}

/// The IPC event loop task.  Owns the ZMQ ROUTER socket and dispatches
/// commands from `IpcBridge`.
///
/// Currently uses raw Unix domain sockets with the IPC codec wire format
/// (length-prefixed CBOR + activation) instead of ZMQ, to avoid adding a
/// C library dependency.  The protocol is compatible: a future migration
/// to ZMQ ROUTER/DEALER only changes the framing layer, not the codec.
///
/// Protocol:
///   1. Bind a Unix stream listener at `socket_path`.
///   2. Accept one connection from the Python DEALER worker.
///   3. Loop: encode forward request → send → recv response → dispatch.
#[cfg(unix)]
async fn ipc_event_loop(
    socket_path: PathBuf,
    mut cmd_rx: mpsc::Receiver<IpcCommand>,
) {
    // Bind the Unix socket listener.
    let listener = match tokio::net::UnixListener::bind(&socket_path) {
        Ok(l) => l,
        Err(e) => {
            error!(path = %socket_path.display(), err = %e, "failed to bind IPC socket");
            return;
        }
    };

    info!(path = %socket_path.display(), "IPC socket bound, waiting for worker");

    // Wait for the Python worker to connect (with timeout).
    let stream = tokio::select! {
        accept_result = listener.accept() => {
            match accept_result {
                Ok((stream, _addr)) => {
                    info!("IPC worker connected");
                    stream
                }
                Err(e) => {
                    error!(err = %e, "IPC accept failed");
                    return;
                }
            }
        }
        // Also drain commands while waiting — if Shutdown arrives before
        // a worker connects, we exit cleanly.
        cmd = cmd_rx.recv() => {
            match cmd {
                Some(IpcCommand::Shutdown) | None => {
                    info!("IPC bridge shutdown before worker connected");
                    std::fs::remove_file(&socket_path).ok();
                    return;
                }
                Some(IpcCommand::Forward { reply, .. }) => {
                    let _ = reply.send(Err("no worker connected".into()));
                    return;
                }
                Some(IpcCommand::ForwardBatch { reply, .. }) => {
                    let _ = reply.send(Err("no worker connected".into()));
                    return;
                }
            }
        }
    };

    // Split into read/write halves.
    let (reader, writer) = stream.into_split();
    let writer = Arc::new(Mutex::new(writer));
    let reader = Arc::new(Mutex::new(reader));

    // Main command loop.
    loop {
        let cmd = match cmd_rx.recv().await {
            Some(cmd) => cmd,
            None => break, // All senders dropped.
        };

        match cmd {
            IpcCommand::Forward {
                header,
                activation,
                reply,
            } => {
                let w = writer.clone();
                let r = reader.clone();

                // Encode the request.
                let wire = match ipc_codec::encode_forward_request(&header, &activation) {
                    Ok(w) => w,
                    Err(e) => {
                        let _ = reply.send(Err(format!("encode failed: {e}")));
                        continue;
                    }
                };

                // Send request.
                {
                    use tokio::io::AsyncWriteExt;
                    let mut w = w.lock().await;
                    // Length-prefix the entire message so the Python side
                    // knows how many bytes to read.
                    let msg_len = wire.len() as u32;
                    if let Err(e) = w.write_all(&msg_len.to_le_bytes()).await {
                        let _ = reply.send(Err(format!("IPC write len failed: {e}")));
                        continue;
                    }
                    if let Err(e) = w.write_all(&wire).await {
                        let _ = reply.send(Err(format!("IPC write failed: {e}")));
                        continue;
                    }
                    if let Err(e) = w.flush().await {
                        let _ = reply.send(Err(format!("IPC flush failed: {e}")));
                        continue;
                    }
                }

                // Receive response.
                let response = {
                    use tokio::io::AsyncReadExt;
                    let mut r = r.lock().await;

                    // Read message length.
                    let mut len_buf = [0u8; 4];
                    if let Err(e) = r.read_exact(&mut len_buf).await {
                        let _ = reply.send(Err(format!("IPC read len failed: {e}")));
                        continue;
                    }
                    let msg_len = u32::from_le_bytes(len_buf) as usize;

                    if msg_len > 100 * 1024 * 1024 {
                        let _ = reply.send(Err(format!(
                            "IPC response too large: {msg_len} bytes"
                        )));
                        continue;
                    }

                    // Read message body.
                    let mut body = vec![0u8; msg_len];
                    if let Err(e) = r.read_exact(&mut body).await {
                        let _ = reply.send(Err(format!("IPC read body failed: {e}")));
                        continue;
                    }

                    // Decode response.
                    match ipc_codec::decode_response(&body) {
                        Ok((hdr, act)) => IpcResponse {
                            header: hdr,
                            activation: act.to_vec(),
                        },
                        Err(e) => {
                            let _ = reply.send(Err(format!("IPC decode failed: {e}")));
                            continue;
                        }
                    }
                };

                debug!(
                    request_id = %response.header.request_id,
                    status = ?response.header.status,
                    act_len = response.activation.len(),
                    "IPC response received"
                );

                let _ = reply.send(Ok(response));
            }
            IpcCommand::ForwardBatch { items, reply } => {
                let w = writer.clone();
                let r = reader.clone();

                // Encode the batch request.
                let refs: Vec<(&ipc_codec::IpcForwardHeader, &[u8])> = items
                    .iter()
                    .map(|(h, a)| (h, a.as_slice()))
                    .collect();
                let wire = match ipc_codec::encode_batch_request(&refs) {
                    Ok(w) => w,
                    Err(e) => {
                        let _ = reply.send(Err(format!("batch encode: {e}")));
                        continue;
                    }
                };

                let batch_count = items.len();

                // Send batch request.
                {
                    use tokio::io::AsyncWriteExt;
                    let mut w = w.lock().await;
                    let msg_len = wire.len() as u32;
                    if let Err(e) = w.write_all(&msg_len.to_le_bytes()).await {
                        let _ = reply.send(Err(format!("IPC batch write len: {e}")));
                        continue;
                    }
                    if let Err(e) = w.write_all(&wire).await {
                        let _ = reply.send(Err(format!("IPC batch write: {e}")));
                        continue;
                    }
                    if let Err(e) = w.flush().await {
                        let _ = reply.send(Err(format!("IPC batch flush: {e}")));
                        continue;
                    }
                }

                // Receive batch response.
                let responses = {
                    use tokio::io::AsyncReadExt;
                    let mut r = r.lock().await;

                    let mut len_buf = [0u8; 4];
                    if let Err(e) = r.read_exact(&mut len_buf).await {
                        let _ = reply.send(Err(format!("IPC batch read len: {e}")));
                        continue;
                    }
                    let msg_len = u32::from_le_bytes(len_buf) as usize;

                    if msg_len > 100 * 1024 * 1024 {
                        let _ = reply.send(Err(format!(
                            "IPC batch response too large: {msg_len} bytes"
                        )));
                        continue;
                    }

                    let mut body = vec![0u8; msg_len];
                    if let Err(e) = r.read_exact(&mut body).await {
                        let _ = reply.send(Err(format!("IPC batch read body: {e}")));
                        continue;
                    }

                    match ipc_codec::decode_batch_response(&body) {
                        Ok(items) => items
                            .into_iter()
                            .map(|(hdr, act)| IpcResponse {
                                header: hdr,
                                activation: act,
                            })
                            .collect::<Vec<_>>(),
                        Err(e) => {
                            let _ = reply.send(Err(format!("IPC batch decode: {e}")));
                            continue;
                        }
                    }
                };

                debug!(
                    batch_count,
                    received = responses.len(),
                    "IPC batch response received"
                );

                let _ = reply.send(Ok(responses));
            }
            IpcCommand::Shutdown => {
                info!("IPC bridge shutting down");
                break;
            }
        }
    }

    // Cleanup.
    std::fs::remove_file(&socket_path).ok();
    info!("IPC event loop exited");
}

/// Windows fallback: the legacy Python IPC bridge relies on Unix domain
/// sockets, which are unavailable on Windows. This path is inactive in the
/// pure-Rust build (there is no Python worker); drain commands with an error
/// so callers never hang waiting for a reply.
#[cfg(not(unix))]
async fn ipc_event_loop(
    _socket_path: PathBuf,
    mut cmd_rx: mpsc::Receiver<IpcCommand>,
) {
    tracing::warn!("IPC bridge unavailable on this platform (requires Unix domain sockets)");
    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            IpcCommand::Shutdown => break,
            IpcCommand::Forward { reply, .. } => {
                let _ = reply.send(Err("IPC bridge not supported on this platform".into()));
            }
            IpcCommand::ForwardBatch { reply, .. } => {
                let _ = reply.send(Err("IPC bridge not supported on this platform".into()));
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::ipc_codec::{IpcForwardHeader, IpcResponseHeader, IpcStatus, ActivationDtype};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Integration test: start bridge, connect a mock Python worker,
    /// send a forward request, receive a response.
    #[tokio::test]
    async fn test_ipc_bridge_roundtrip() {
        let socket_path = format!(
            "/tmp/openhydra-test-ipc-{}.sock",
            std::process::id()
        );

        // Clean up from previous test runs.
        std::fs::remove_file(&socket_path).ok();

        let bridge = IpcBridge::start(
            "test-peer",
            Some(&socket_path),
            tokio::runtime::Handle::current(),
        )
        .await
        .unwrap();

        // Spawn a mock Python worker that connects and echoes.
        let sp = socket_path.clone();
        let worker_handle = tokio::spawn(async move {
            // Give the bridge a moment to bind.
            tokio::time::sleep(Duration::from_millis(50)).await;

            let stream = tokio::net::UnixStream::connect(&sp).await.unwrap();
            let (mut reader, mut writer) = stream.into_split();

            // Read one request.
            let mut len_buf = [0u8; 4];
            reader.read_exact(&mut len_buf).await.unwrap();
            let msg_len = u32::from_le_bytes(len_buf) as usize;

            let mut body = vec![0u8; msg_len];
            reader.read_exact(&mut body).await.unwrap();

            // Decode the request to get the request_id.
            let (req_header, req_act) =
                ipc_codec::decode_forward_request(&body).unwrap();
            assert_eq!(req_header.request_id, "roundtrip-test");
            assert_eq!(req_act.len(), 16); // 4 floats × 4 bytes

            // Build a response — double the activation values.
            let resp_header = IpcResponseHeader {
                request_id: req_header.request_id.clone(),
                status: IpcStatus::Ok,
                activation_dtype: ActivationDtype::Fp32,
                activation_shape: vec![1, 1, 4],
                ..Default::default()
            };

            // Create response activation (doubled values).
            let in_floats: Vec<f32> = req_act
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) * 2.0)
                .collect();
            let resp_act: Vec<u8> = in_floats
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();

            let resp_wire =
                ipc_codec::encode_response(&resp_header, &resp_act).unwrap();

            // Send the response with length prefix.
            let resp_len = resp_wire.len() as u32;
            writer.write_all(&resp_len.to_le_bytes()).await.unwrap();
            writer.write_all(&resp_wire).await.unwrap();
            writer.flush().await.unwrap();
        });

        // Give the worker time to connect.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Send a forward request.
        let activation: Vec<u8> = vec![1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let header = IpcForwardHeader {
            request_id: "roundtrip-test".into(),
            stage_index: 0,
            total_stages: 2,
            shard_layer_start: 0,
            shard_layer_end: 16,
            shard_total_layers: 32,
            ..Default::default()
        };

        let response = bridge.forward(header, activation).await.unwrap();

        assert_eq!(response.header.request_id, "roundtrip-test");
        assert_eq!(response.header.status, IpcStatus::Ok);
        assert_eq!(response.header.activation_shape, vec![1, 1, 4]);

        // Verify the activation values were doubled.
        let out_floats: Vec<f32> = response
            .activation
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(out_floats, vec![2.0, 4.0, 6.0, 8.0]);

        // Cleanup.
        bridge.shutdown().await;
        worker_handle.await.unwrap();
    }
}
