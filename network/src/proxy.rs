//! gRPC-over-libp2p proxy — tunnels gRPC through Circuit Relay v2.

use std::io;

use async_trait::async_trait;
use futures::prelude::*;
use libp2p::request_response::{self, Codec, ProtocolSupport};
use libp2p::StreamProtocol;
use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

pub const PROXY_PROTOCOL: StreamProtocol = StreamProtocol::new("/openhydra/tensor/1.0.0");
const MAX_MSG_SIZE: usize = 100 * 1024 * 1024;
/// E-S6: cap on the buffer we reserve up-front. The 4-byte length header is
/// attacker-controlled and unauthenticated, so we must NOT `vec![0u8; declared]`
/// — a peer could declare 100 MB and send nothing, forcing that allocation per
/// stream for free. Instead we reserve at most this much and grow the buffer only
/// as real bytes arrive (see `read_framed`), so a lying header costs ~nothing and
/// a large buffer only ever backs bytes the peer actually paid to send.
const INITIAL_ALLOC_CAP: usize = 64 * 1024;

/// Read a `u32`-length-prefixed frame without pre-allocating the declared length.
/// Preserves exact framing (`read_exact` errors on EOF/short read) while bounding
/// the up-front allocation to `INITIAL_ALLOC_CAP` (E-S6).
async fn read_framed<T>(io: &mut T) -> io::Result<Vec<u8>>
where
    T: AsyncRead + Unpin + Send,
{
    let mut len_buf = [0u8; 4];
    io.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_MSG_SIZE {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "message exceeds MAX_MSG_SIZE"));
    }
    let mut buf = Vec::with_capacity(len.min(INITIAL_ALLOC_CAP));
    let mut remaining = len;
    let mut chunk = [0u8; 64 * 1024];
    while remaining > 0 {
        let want = remaining.min(chunk.len());
        io.read_exact(&mut chunk[..want]).await?;
        buf.extend_from_slice(&chunk[..want]);
        remaining -= want;
    }
    Ok(buf)
}

#[derive(Debug, Clone)]
pub struct ProxyRequest(pub Vec<u8>);
#[derive(Debug, Clone)]
pub struct ProxyResponse(pub Vec<u8>);

#[derive(Debug, Clone, Default)]
pub struct GrpcProxyCodec;

#[async_trait]
impl Codec for GrpcProxyCodec {
    type Protocol = StreamProtocol;
    type Request = ProxyRequest;
    type Response = ProxyResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        Ok(ProxyRequest(read_framed(io).await?))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        Ok(ProxyResponse(read_framed(io).await?))
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let len_buf = (req.0.len() as u32).to_be_bytes();
        io.write_all(&len_buf).await?;
        io.write_all(&req.0).await?;
        io.flush().await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        resp: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let len_buf = (resp.0.len() as u32).to_be_bytes();
        io.write_all(&len_buf).await?;
        io.write_all(&resp.0).await?;
        io.flush().await
    }
}

pub fn proxy_behaviour() -> request_response::Behaviour<GrpcProxyCodec> {
    let mut config = request_response::Config::default();
    // Increase timeout for model inference — CPU inference can take 10-30s.
    config.set_request_timeout(std::time::Duration::from_secs(120));
    request_response::Behaviour::new(
        [(PROXY_PROTOCOL, ProtocolSupport::Full)],
        config,
    )
}

/// Handle inbound proxy request by forwarding to local gRPC server.
pub async fn handle_proxy_request(request: ProxyRequest, local_grpc_port: u16) -> ProxyResponse {
    let target = format!("127.0.0.1:{}", local_grpc_port);
    match forward_to_local(&request.0, &target).await {
        Ok(resp) => ProxyResponse(resp),
        Err(e) => {
            warn!(error=%e, "proxy forward failed");
            ProxyResponse(Vec::new())
        }
    }
}

/// A wedged local engine must not hang the inbound-serve task forever. Bound the whole
/// connect+write+read below the request_response protocol timeout (120s) so the local
/// socket is reclaimed with a clean error before the remote peer's request expires.
/// See docs/CODEBASE_HARDENING_PLAN.md (A2).
const LOCAL_FORWARD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(115);

async fn forward_to_local(
    data: &[u8],
    target: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    tokio::time::timeout(LOCAL_FORWARD_TIMEOUT, async {
        let mut stream = TcpStream::connect(target).await?;
        stream.write_all(data).await?;
        stream.shutdown().await?;
        let mut resp = Vec::new();
        stream.read_to_end(&mut resp).await?;
        Ok::<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>(resp)
    })
    .await
    .map_err(|_| -> Box<dyn std::error::Error + Send + Sync> {
        format!("local forward timed out after {}s", LOCAL_FORWARD_TIMEOUT.as_secs()).into()
    })?
}

pub async fn start_proxy_listener() -> io::Result<(TcpListener, String)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = format!("127.0.0.1:{}", listener.local_addr()?.port());
    info!(proxy=%addr, "gRPC proxy listener started");
    Ok((listener, addr))
}

// ── CP-1: Wire format negotiation ──────────────────────────────────────

/// Detected wire format of an inbound proxy message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireFormat {
    /// New ForwardMsg binary format (magic 0x4F485632 "OHV2").
    ForwardMsg,
    /// Legacy protobuf (anything that doesn't start with OHV2 magic).
    LegacyProtobuf,
}

/// Detect the wire format of an inbound proxy message by inspecting
/// the first 4 bytes.
///
/// This is the sole format-negotiation point: all inbound proxy
/// messages pass through this check before dispatch.
#[inline]
pub fn detect_wire_format(data: &[u8]) -> WireFormat {
    if crate::forward_msg::is_forward_msg(data) {
        WireFormat::ForwardMsg
    } else {
        WireFormat::LegacyProtobuf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_forward_msg() {
        // Build a minimal ForwardMsg.
        let header = crate::ipc_codec::IpcForwardHeader {
            request_id: "x".into(),
            ..Default::default()
        };
        let wire = crate::forward_msg::encode(
            crate::forward_msg::MsgType::Forward,
            &header,
            &[],
        )
        .unwrap();
        assert_eq!(detect_wire_format(&wire), WireFormat::ForwardMsg);
    }

    #[test]
    fn test_detect_legacy_protobuf() {
        // Protobuf messages start with field tags, not OHV2.
        let protobuf = vec![0x0A, 0x10, 0x08, 0x01, 0x12, 0x04];
        assert_eq!(detect_wire_format(&protobuf), WireFormat::LegacyProtobuf);
    }

    #[test]
    fn test_detect_empty_is_legacy() {
        assert_eq!(detect_wire_format(&[]), WireFormat::LegacyProtobuf);
    }

    #[test]
    fn test_detect_method_prefix_is_legacy() {
        // Method prefix bytes (0x01–0x06) used by Python proxy handler.
        for prefix in 1u8..=6u8 {
            let msg = vec![prefix, 0x0A, 0x10, 0x08];
            assert_eq!(
                detect_wire_format(&msg),
                WireFormat::LegacyProtobuf,
                "prefix 0x{:02X} should be legacy",
                prefix
            );
        }
    }

    // ── E-S6: framed read bounds the pre-auth allocation ───────────────────

    fn framed(payload: &[u8]) -> Vec<u8> {
        let mut v = (payload.len() as u32).to_be_bytes().to_vec();
        v.extend_from_slice(payload);
        v
    }

    #[tokio::test]
    async fn read_framed_roundtrips_payload() {
        let payload = vec![7u8; 128 * 1024]; // larger than INITIAL_ALLOC_CAP → exercises growth
        let wire = framed(&payload);
        let mut io: &[u8] = &wire;
        let got = read_framed(&mut io).await.unwrap();
        assert_eq!(got, payload);
    }

    #[tokio::test]
    async fn read_framed_rejects_oversize_header() {
        // Declared length above MAX_MSG_SIZE is rejected before any body read.
        let mut wire = ((MAX_MSG_SIZE as u64 + 1) as u32).to_be_bytes().to_vec();
        wire.extend_from_slice(&[0u8; 8]);
        let mut io: &[u8] = &wire;
        let err = read_framed(&mut io).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn read_framed_lying_header_errors_without_preallocating() {
        // Header claims a huge body but the peer sends only a few bytes then EOF.
        // The chunked reader never allocates the declared length — it errors on the
        // short read instead of OOMing (E-S6: the old vec![0u8; len] would have
        // reserved ~100 MB up-front here).
        let mut wire = (100u32 * 1024 * 1024).to_be_bytes().to_vec();
        wire.extend_from_slice(&[1, 2, 3, 4]); // far fewer bytes than declared
        let mut io: &[u8] = &wire;
        let err = read_framed(&mut io).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }
}
