//! Supernode prompt routing protocol — `/openhydra/prompt/1.0.0`.
//!
//! Carries CBOR-encoded prompt requests and token stream chunks between
//! supernodes. Separate from the tensor proxy protocol so prompt routing
//! traffic doesn't mix with activation-forwarding traffic.
//!
//! Wire format: same length-prefixed codec as gRPC proxy.
//! The CBOR payload's first field encodes the method prefix (0x10–0x14)
//! but dispatch happens in Python — Rust treats it as opaque bytes.

use std::io;

use async_trait::async_trait;
use futures::prelude::*;
use libp2p::request_response::{self, Codec, ProtocolSupport};
use libp2p::StreamProtocol;

pub const PROMPT_PROTOCOL: StreamProtocol = StreamProtocol::new("/openhydra/prompt/1.0.0");
const MAX_MSG_SIZE: usize = 10 * 1024 * 1024; // 10 MB — prompts are smaller than activations

#[derive(Debug, Clone)]
pub struct PromptRequest(pub Vec<u8>);
#[derive(Debug, Clone)]
pub struct PromptResponse(pub Vec<u8>);

#[derive(Debug, Clone, Default)]
pub struct PromptCodec;

#[async_trait]
impl Codec for PromptCodec {
    type Protocol = StreamProtocol;
    type Request = PromptRequest;
    type Response = PromptResponse;

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
            return Err(io::Error::new(io::ErrorKind::InvalidData, "prompt too large"));
        }
        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;
        Ok(PromptRequest(buf))
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
            return Err(io::Error::new(io::ErrorKind::InvalidData, "prompt response too large"));
        }
        let mut buf = vec![0u8; len];
        io.read_exact(&mut buf).await?;
        Ok(PromptResponse(buf))
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

pub fn prompt_behaviour() -> request_response::Behaviour<PromptCodec> {
    let mut config = request_response::Config::default();
    config.set_request_timeout(std::time::Duration::from_secs(300));
    request_response::Behaviour::new(
        [(PROMPT_PROTOCOL, ProtocolSupport::Full)],
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_string() {
        assert_eq!(PROMPT_PROTOCOL.to_string(), "/openhydra/prompt/1.0.0");
    }

    #[test]
    fn test_max_msg_size() {
        assert_eq!(MAX_MSG_SIZE, 10 * 1024 * 1024);
    }
}
