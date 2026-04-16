//! gRPC-over-libp2p proxy — tunnels gRPC through Circuit Relay v2.

use std::io;

use async_trait::async_trait;
use futures::prelude::*;
use libp2p::request_response::{self, Codec, ProtocolSupport};
use libp2p::StreamProtocol;
use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

pub const PROXY_PROTOCOL: StreamProtocol = StreamProtocol::new("/openhydra/grpc-proxy/1.0.0");
const MAX_MSG_SIZE: usize = 100 * 1024 * 1024;

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
        let mut len_buf = [0u8; 4];
        io.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;
        if len > MAX_MSG_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "too large"));
        }
        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;
        Ok(ProxyRequest(buf))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut len_buf = [0u8; 4];
        io.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;
        if len > MAX_MSG_SIZE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "too large"));
        }
        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;
        Ok(ProxyResponse(buf))
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

async fn forward_to_local(
    data: &[u8],
    target: &str,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = TcpStream::connect(target).await?;
    stream.write_all(data).await?;
    stream.shutdown().await?;
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).await?;
    Ok(resp)
}

pub async fn start_proxy_listener() -> io::Result<(TcpListener, String)> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = format!("127.0.0.1:{}", listener.local_addr()?.port());
    info!(proxy=%addr, "gRPC proxy listener started");
    Ok((listener, addr))
}
