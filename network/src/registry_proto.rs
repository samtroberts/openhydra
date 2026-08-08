//! C7: the `/openhydra/registry/1.0.0` query protocol.
//!
//! A tiny request/response protocol a consumer uses to ask a bootstrap "who serves model X?".
//! The bootstrap answers from its [`crate::registry::ProviderRegistry`] — the verified
//! `PROVIDER_ANNOUNCE` records it has retained. This makes discovery reliable across NATs, where
//! the D-sized gossipsub mesh does not forward a provider's advert to the specific consumer and
//! the NAT'd provider's DHT `put_record` times out.
//!
//! Trust: the records returned are the providers' own **self-signed** [`PeerRecord`]s. The
//! bootstrap already ran `dht::pex_record_is_authentic` on ingest, but it is not a trust anchor —
//! the consumer MUST re-verify each record (`dht::verify_peer_record`) before dialing it. This
//! protocol only conveys signed records; it never vouches for them.
//!
//! Framing mirrors [`crate::proxy`]: a `u32` big-endian length prefix + a JSON body. Payloads
//! are tiny (a model id; a handful of signed records), so the size cap is far below the proxy's.

use std::io;

use async_trait::async_trait;
use futures::prelude::*;
use libp2p::request_response::{self, Codec, ProtocolSupport};
use libp2p::StreamProtocol;

use crate::types::PeerRecord;

/// The wire protocol id. A bootstrap answers it; a consumer asks.
pub const REGISTRY_PROTOCOL: StreamProtocol = StreamProtocol::new("/openhydra/registry/1.0.0");

/// Registry payloads are tiny — a model id and a few signed records. Cap well below the proxy's
/// 100 MB: a registry peer never moves bulk data, so any larger declared length is junk and is
/// rejected before we read a body.
const MAX_MSG_SIZE: usize = 1024 * 1024;

/// A consumer's "who serves this model?" query.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegistryQuery {
    pub model_id: String,
}

/// The bootstrap's answer: the fresh, verified-on-ingest provider records it holds for the model.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RegistryReply {
    pub records: Vec<PeerRecord>,
}

/// Read a `u32`-length-prefixed frame, bounded to [`MAX_MSG_SIZE`]. The length is
/// attacker-controlled, so it is checked before the (already ≤ 1 MB) body allocation.
async fn read_framed<T>(io: &mut T) -> io::Result<Vec<u8>>
where
    T: AsyncRead + Unpin + Send,
{
    let mut len_buf = [0u8; 4];
    io.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_MSG_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "registry message exceeds MAX_MSG_SIZE",
        ));
    }
    let mut buf = vec![0u8; len];
    io.read_exact(&mut buf).await?;
    Ok(buf)
}

async fn write_framed<T>(io: &mut T, data: &[u8]) -> io::Result<()>
where
    T: AsyncWrite + Unpin + Send,
{
    io.write_all(&(data.len() as u32).to_be_bytes()).await?;
    io.write_all(data).await?;
    io.flush().await
}

#[derive(Debug, Clone, Default)]
pub struct RegistryCodec;

#[async_trait]
impl Codec for RegistryCodec {
    type Protocol = StreamProtocol;
    type Request = RegistryQuery;
    type Response = RegistryReply;

    async fn read_request<T>(&mut self, _p: &Self::Protocol, io: &mut T) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let bytes = read_framed(io).await?;
        serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn read_response<T>(
        &mut self,
        _p: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let bytes = read_framed(io).await?;
        serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    async fn write_request<T>(
        &mut self,
        _p: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let bytes = serde_json::to_vec(&req)?;
        write_framed(io, &bytes).await
    }

    async fn write_response<T>(
        &mut self,
        _p: &Self::Protocol,
        io: &mut T,
        resp: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let bytes = serde_json::to_vec(&resp)?;
        write_framed(io, &bytes).await
    }
}

/// Build the registry request/response behaviour.
///
/// `support` sets the role: consumers use [`ProtocolSupport::Outbound`] (ask only), bootstraps
/// use [`ProtocolSupport::Inbound`] (answer only). The 10 s timeout is generous for a
/// single-round-trip lookup against a connected bootstrap.
pub fn registry_behaviour(support: ProtocolSupport) -> request_response::Behaviour<RegistryCodec> {
    let config = request_response::Config::default()
        .with_request_timeout(std::time::Duration::from_secs(10));
    request_response::Behaviour::new([(REGISTRY_PROTOCOL, support)], config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn query_roundtrips_through_the_codec() {
        let mut codec = RegistryCodec;
        let proto = REGISTRY_PROTOCOL;
        let mut buf = Vec::new();
        codec
            .write_request(&proto, &mut buf, RegistryQuery { model_id: "qwen2.5-coder:1.5b".into() })
            .await
            .unwrap();
        let mut io: &[u8] = &buf;
        let got = codec.read_request(&proto, &mut io).await.unwrap();
        assert_eq!(got.model_id, "qwen2.5-coder:1.5b");
    }

    #[tokio::test]
    async fn reply_roundtrips_records_through_the_codec() {
        let mut codec = RegistryCodec;
        let proto = REGISTRY_PROTOCOL;
        let reply = RegistryReply {
            records: vec![PeerRecord {
                peer_id: "p1".into(),
                model_id: "m1".into(),
                libp2p_peer_id: "12D3KooWabc".into(),
                ..Default::default()
            }],
        };
        let mut buf = Vec::new();
        codec.write_response(&proto, &mut buf, reply).await.unwrap();
        let mut io: &[u8] = &buf;
        let got = codec.read_response(&proto, &mut io).await.unwrap();
        assert_eq!(got.records.len(), 1);
        assert_eq!(got.records[0].peer_id, "p1");
    }

    #[tokio::test]
    async fn read_framed_rejects_oversize_header() {
        // Declared length above the cap is refused before any body read.
        let mut wire = ((MAX_MSG_SIZE as u64 + 1) as u32).to_be_bytes().to_vec();
        wire.extend_from_slice(&[0u8; 8]);
        let mut io: &[u8] = &wire;
        let err = read_framed(&mut io).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
