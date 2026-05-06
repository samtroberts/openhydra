//! TCP-to-libp2p tunnel — Phase 2 "superhighway" for cross-ISP gRPC.
//!
//! When DCUtR hole-punches a direct connection to a remote peer, the Python
//! layer calls `open_tunnel(peer_id)` which:
//!
//! 1. Binds a local TCP listener on `127.0.0.1:0` (ephemeral port).
//! 2. Returns the local address (e.g. `"127.0.0.1:52431"`) to Python.
//! 3. For each inbound TCP connection (from the local gRPC client), opens
//!    a raw libp2p substream to the remote peer using the
//!    `/openhydra/tunnel/1.0.0` protocol, then bidirectionally copies
//!    bytes between the TCP socket and the libp2p stream.
//!
//! On the responder side, inbound libp2p tunnel streams are connected to
//! the local gRPC server at `127.0.0.1:50051` using the same bidirectional
//! copy.
//!
//! The result: gRPC traffic bypasses the request_response proxy layer and
//! flows directly over the hole-punched QUIC/TCP connection, cutting
//! per-hop latency from ~200-600ms (relay) to ~5-30ms (direct).

use libp2p::PeerId;
use libp2p::StreamProtocol;
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

/// The libp2p stream protocol name for tunnel substreams.
pub const TUNNEL_PROTOCOL: StreamProtocol = StreamProtocol::new("/openhydra/tunnel/1.0.0");

/// Default local gRPC port that the tunnel responder connects to.
pub const DEFAULT_GRPC_PORT: u16 = 50051;

/// State for one active tunnel to a remote peer.
pub struct TunnelState {
    /// Local address string, e.g. "127.0.0.1:52431".
    pub local_addr: String,
    /// Send `true` to cancel the tunnel listener task.
    pub cancel_tx: tokio::sync::watch::Sender<bool>,
    /// Handle to the listener task (for join on cleanup).
    pub _handle: JoinHandle<()>,
}

/// Run the tunnel initiator loop: accept TCP connections on `listener`,
/// for each one open a libp2p stream to `peer` and bidirectionally copy.
///
/// Runs until `cancel_rx` receives `true` or the listener errors out.
pub async fn run_tunnel_initiator(
    listener: TcpListener,
    control: libp2p_stream::Control,
    peer: PeerId,
    mut cancel_rx: tokio::sync::watch::Receiver<bool>,
) {
    let addr = listener
        .local_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "?".into());
    info!(%peer, %addr, "tunnel_initiator_started");

    loop {
        tokio::select! {
            biased;
            _ = cancel_rx.changed() => {
                info!(%peer, "tunnel_initiator_cancelled");
                break;
            }
            result = listener.accept() => {
                match result {
                    Ok((tcp_stream, tcp_addr)) => {
                        debug!(%peer, %tcp_addr, "tunnel_initiator: tcp connection accepted");
                        let mut ctrl = control.clone();
                        tokio::spawn(async move {
                            match ctrl.open_stream(peer, TUNNEL_PROTOCOL).await {
                                Ok(libp2p_stream) => {
                                    debug!(%peer, "tunnel_initiator: libp2p stream opened, copying");
                                    bidirectional_copy(tcp_stream, libp2p_stream).await;
                                    debug!(%peer, "tunnel_initiator: stream closed");
                                }
                                Err(e) => {
                                    warn!(%peer, error = %e, "tunnel_initiator: open_stream failed");
                                }
                            }
                        });
                    }
                    Err(e) => {
                        warn!(%peer, error = %e, "tunnel_initiator: accept error");
                        break;
                    }
                }
            }
        }
    }
}

/// Spawn the tunnel responder: accept ALL inbound tunnel streams from any
/// peer and connect each to the local gRPC server.
///
/// This runs for the lifetime of the node. When a stream arrives, a new
/// TCP connection to `127.0.0.1:{local_grpc_port}` is opened and bytes
/// are copied bidirectionally.
pub fn spawn_tunnel_responder(
    mut incoming: libp2p_stream::IncomingStreams,
    local_grpc_port: u16,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        use futures::StreamExt;
        info!(port = local_grpc_port, "tunnel_responder_started");
        while let Some((peer, stream)) = incoming.next().await {
            info!(%peer, "tunnel_responder: inbound stream accepted");
            let port = local_grpc_port;
            tokio::spawn(async move {
                let target = format!("127.0.0.1:{port}");
                match TcpStream::connect(&target).await {
                    Ok(tcp_stream) => {
                        debug!(%peer, %target, "tunnel_responder: connected to gRPC, copying");
                        bidirectional_copy(tcp_stream, stream).await;
                        debug!(%peer, "tunnel_responder: stream closed");
                    }
                    Err(e) => {
                        warn!(%peer, %target, error = %e, "tunnel_responder: connect failed");
                    }
                }
            });
        }
        info!("tunnel_responder_stopped");
    })
}

/// Bidirectional byte copy between a tokio TcpStream and a libp2p Stream.
///
/// libp2p's `Stream` implements `futures::AsyncRead + futures::AsyncWrite`
/// while tokio uses `tokio::io::AsyncRead + tokio::io::AsyncWrite`.
/// We bridge the two with manual buffer loops rather than adding a
/// `tokio-util` compat dependency.
async fn bidirectional_copy(tcp_stream: TcpStream, libp2p_stream: libp2p::Stream) {
    use futures::AsyncReadExt as FutAsyncReadExt;
    use futures::AsyncWriteExt as FutAsyncWriteExt;
    use tokio::io::AsyncReadExt as TokioAsyncReadExt;
    use tokio::io::AsyncWriteExt as TokioAsyncWriteExt;

    let (mut tcp_read, mut tcp_write) = tcp_stream.into_split();
    let (mut lp2p_read, mut lp2p_write) = FutAsyncReadExt::split(libp2p_stream);

    // TCP → libp2p
    let tcp_to_lp2p = tokio::spawn(async move {
        let mut buf = vec![0u8; 16384];
        loop {
            let n = match TokioAsyncReadExt::read(&mut tcp_read, &mut buf).await {
                Ok(0) => break,
                Ok(n) => n,
                Err(_) => break,
            };
            if FutAsyncWriteExt::write_all(&mut lp2p_write, &buf[..n])
                .await
                .is_err()
            {
                break;
            }
            // Flush after each write to minimize latency for gRPC frames.
            if FutAsyncWriteExt::flush(&mut lp2p_write).await.is_err() {
                break;
            }
        }
        let _ = FutAsyncWriteExt::close(&mut lp2p_write).await;
    });

    // libp2p → TCP
    let lp2p_to_tcp = tokio::spawn(async move {
        let mut buf = vec![0u8; 16384];
        loop {
            let n = match FutAsyncReadExt::read(&mut lp2p_read, &mut buf).await {
                Ok(0) => break,
                Ok(n) => n,
                Err(_) => break,
            };
            if TokioAsyncWriteExt::write_all(&mut tcp_write, &buf[..n])
                .await
                .is_err()
            {
                break;
            }
            if TokioAsyncWriteExt::flush(&mut tcp_write).await.is_err() {
                break;
            }
        }
    });

    // Wait for both directions to finish. When one side closes, the other
    // will see EOF and exit too.
    let _ = tokio::join!(tcp_to_lp2p, lp2p_to_tcp);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tunnel_protocol_name() {
        assert_eq!(TUNNEL_PROTOCOL.as_ref(), "/openhydra/tunnel/1.0.0");
    }
}
