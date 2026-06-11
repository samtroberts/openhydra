//! CP-3: HeadSampler bridge — IPC to Python's `StandaloneHead` for token sampling.
//!
//! The coordinator's ring manager needs to sample tokens from hidden states.
//! The model's `final_norm + lm_head + sample()` pipeline lives in Python
//! (`coordinator/head_sampler.py`). This bridge connects them via a second
//! IPC socket (separate from the per-peer worker bridge).
//!
//! Protocol (request, Rust → Python):
//! ```text
//! [0:4]    header_len       (u32 LE)
//! [4:4+H]  header           (CBOR SampleRequest)
//! [4+H:4+H+4] activation_len (u32 LE)
//! [4+H+4:..]  activation     (raw float32 bytes — hidden state)
//! ```
//!
//! Protocol (response, Python → Rust):
//! ```text
//! [0:4]    header_len       (u32 LE)
//! [4:4+H]  header           (CBOR SampleResponse)
//! [4+H:4+H+4] embedding_len (u32 LE)
//! [4+H+4:..]  embedding     (raw float32 bytes — next token embedding)
//! ```
//!
//! Round-trip: <1ms (IPC + Python sample is ~0.3ms).
//! The embedding is returned alongside the token_id so the ring manager
//! can re-inject it without a separate embedding table lookup.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

/// Request sent to the HeadSampler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleRequest {
    /// Session ID for KV cache / state correlation.
    pub session_id: String,
    /// Request ID for tracing.
    pub request_id: String,
    /// Sampling temperature (0.0 = greedy).
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub temperature: f64,
    /// Top-p (nucleus) sampling threshold.
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub top_p: f64,
    /// Top-k sampling (0 = disabled).
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub top_k: u32,
    /// Random seed (None = non-deterministic).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// Response from the HeadSampler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleResponse {
    /// The sampled token ID.
    pub token_id: u32,
    /// Whether this token is end-of-sequence.
    pub is_eos: bool,
    /// Human-readable token text.
    #[serde(default)]
    pub token_text: String,
}

fn is_zero_f64(v: &f64) -> bool {
    *v == 0.0
}

fn is_zero_u32(v: &u32) -> bool {
    *v == 0
}

// ── Wire codec (stateless helpers) ───────────────────────────────────

/// Encode a sample request to wire format.
pub fn encode_request(
    request: &SampleRequest,
    activation: &[u8],
) -> Result<Vec<u8>, String> {
    let mut header_bytes = Vec::with_capacity(256);
    ciborium::into_writer(request, &mut header_bytes)
        .map_err(|e| format!("SampleRequest CBOR encode failed: {e}"))?;

    let header_len = header_bytes.len() as u32;
    let activation_len = activation.len() as u32;
    let total = 4 + header_bytes.len() + 4 + activation.len();
    let mut buf = Vec::with_capacity(total);

    buf.extend_from_slice(&header_len.to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    buf.extend_from_slice(&activation_len.to_le_bytes());
    buf.extend_from_slice(activation);

    Ok(buf)
}

/// Decode a sample response from wire format.
pub fn decode_response(data: &[u8]) -> Result<(SampleResponse, Vec<u8>), String> {
    if data.len() < 4 {
        return Err("SampleResponse too short".into());
    }

    let header_len =
        u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;

    if data.len() < 4 + header_len + 4 {
        return Err(format!(
            "SampleResponse truncated: need {}, have {}",
            4 + header_len + 4,
            data.len()
        ));
    }

    let response: SampleResponse =
        ciborium::from_reader(&data[4..4 + header_len])
            .map_err(|e| format!("SampleResponse CBOR decode: {e}"))?;

    let emb_len_offset = 4 + header_len;
    let embedding_len = u32::from_le_bytes(
        data[emb_len_offset..emb_len_offset + 4]
            .try_into()
            .unwrap(),
    ) as usize;

    let emb_start = emb_len_offset + 4;
    let emb_end = emb_start + embedding_len;
    if data.len() < emb_end {
        return Err(format!(
            "SampleResponse embedding truncated: {} vs {}",
            embedding_len,
            data.len() - emb_start
        ));
    }

    Ok((response, data[emb_start..emb_end].to_vec()))
}

// ── Async SamplerBridge ──────────────────────────────────────────────

/// Commands for the sampler IPC event loop.
enum SamplerCommand {
    /// Send hidden-state activation → Python HeadSampler → (token, embedding).
    Sample {
        request: SampleRequest,
        activation: Vec<u8>,
        reply: oneshot::Sender<Result<(SampleResponse, Vec<u8>), String>>,
    },
    Shutdown,
}

/// Bridge to the Python HeadSampler via Unix domain socket.
///
/// Architecture mirrors `IpcBridge` (CP-0): a clone-safe handle that
/// sends commands to an internal tokio task owning the socket.
///
/// The HeadSampler Python process (`coordinator/head_sampler.py`) binds
/// a Unix socket and implements `final_norm → lm_head → sample()`.
/// Round-trip: <1ms on LAN (IPC + Python sample is ~0.3ms).
#[derive(Clone)]
pub struct SamplerBridge {
    cmd_tx: mpsc::Sender<SamplerCommand>,
    socket_path: PathBuf,
}

/// Maximum time to wait for a HeadSampler response.
const SAMPLER_RECV_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

impl SamplerBridge {
    /// Start the SamplerBridge.
    ///
    /// Spawns a tokio task that connects to the HeadSampler Unix socket.
    /// The Python HeadSampler must already be listening (bind-first).
    pub async fn start(socket_path: &str) -> Result<Self, String> {
        let path = PathBuf::from(socket_path);

        let (cmd_tx, cmd_rx) = mpsc::channel::<SamplerCommand>(64);

        let path_clone = path.clone();
        tokio::spawn(sampler_event_loop(path_clone, cmd_rx));

        info!(path = %path.display(), "SamplerBridge started");

        Ok(Self {
            cmd_tx,
            socket_path: path,
        })
    }

    /// Send a hidden-state activation to the HeadSampler and await the
    /// sampled token + next-token embedding.
    ///
    /// Returns `(SampleResponse, embedding_bytes)`.
    pub async fn sample(
        &self,
        request: SampleRequest,
        activation: Vec<u8>,
    ) -> Result<(SampleResponse, Vec<u8>), String> {
        let (reply_tx, reply_rx) = oneshot::channel();

        self.cmd_tx
            .send(SamplerCommand::Sample {
                request,
                activation,
                reply: reply_tx,
            })
            .await
            .map_err(|_| "SamplerBridge shut down".to_string())?;

        match tokio::time::timeout(SAMPLER_RECV_TIMEOUT, reply_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("SamplerBridge reply channel dropped".into()),
            Err(_) => Err("SamplerBridge sample timed out (30s)".into()),
        }
    }

    /// Shut down the bridge.
    pub async fn shutdown(&self) {
        let _ = self.cmd_tx.send(SamplerCommand::Shutdown).await;
    }

    /// Get the socket path.
    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }
}

/// Internal event loop — owns the Unix socket connection to the HeadSampler.
///
/// Connects lazily on first sample request. Reconnects if the connection
/// drops (Python HeadSampler restarts).
async fn sampler_event_loop(
    socket_path: PathBuf,
    mut cmd_rx: mpsc::Receiver<SamplerCommand>,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream: Option<tokio::net::UnixStream> = None;

    loop {
        let cmd = match cmd_rx.recv().await {
            Some(cmd) => cmd,
            None => break,
        };

        match cmd {
            SamplerCommand::Sample {
                request,
                activation,
                reply,
            } => {
                // Lazy connect / reconnect.
                if stream.is_none() {
                    match tokio::net::UnixStream::connect(&socket_path).await {
                        Ok(s) => {
                            info!(path = %socket_path.display(), "SamplerBridge connected");
                            stream = Some(s);
                        }
                        Err(e) => {
                            let _ = reply.send(Err(format!(
                                "SamplerBridge connect failed: {e}"
                            )));
                            continue;
                        }
                    }
                }

                let s = stream.as_mut().unwrap();

                // Encode and send request.
                let wire = match encode_request(&request, &activation) {
                    Ok(w) => w,
                    Err(e) => {
                        let _ = reply.send(Err(format!("encode: {e}")));
                        continue;
                    }
                };

                let msg_len = wire.len() as u32;
                let send_result = async {
                    s.write_all(&msg_len.to_le_bytes()).await?;
                    s.write_all(&wire).await?;
                    s.flush().await
                }
                .await;

                if let Err(e) = send_result {
                    warn!(%e, "SamplerBridge write failed, dropping connection");
                    stream = None;
                    let _ = reply.send(Err(format!("write failed: {e}")));
                    continue;
                }

                // Read response. Audit F9: bound the read with the same
                // timeout the caller uses. Without this, a Python sampler
                // that accepts the request but never replies (deadlock /
                // GIL-stuck) wedges this loop on read_exact forever — every
                // queued Sample command then times out caller-side while the
                // loop never advances and never reconnects, failing ALL ring
                // sessions on this coordinator until process restart. On
                // timeout we drop the connection so the next command
                // reconnects.
                let recv_fut = async {
                    let mut len_buf = [0u8; 4];
                    s.read_exact(&mut len_buf)
                        .await
                        .map_err(|e| format!("read len: {e}"))?;
                    let resp_len = u32::from_le_bytes(len_buf) as usize;

                    if resp_len > 10 * 1024 * 1024 {
                        return Err(format!("response too large: {resp_len}"));
                    }

                    let mut body = vec![0u8; resp_len];
                    s.read_exact(&mut body)
                        .await
                        .map_err(|e| format!("read body: {e}"))?;

                    decode_response(&body)
                };
                let recv_result: Result<(SampleResponse, Vec<u8>), String> =
                    match tokio::time::timeout(SAMPLER_RECV_TIMEOUT, recv_fut).await {
                        Ok(r) => r,
                        Err(_) => {
                            warn!(
                                "SamplerBridge read timed out ({}s), dropping connection",
                                SAMPLER_RECV_TIMEOUT.as_secs()
                            );
                            stream = None;
                            let _ = reply.send(Err("sampler read timed out".into()));
                            continue;
                        }
                    };

                match recv_result {
                    Ok((resp, emb)) => {
                        debug!(
                            token_id = resp.token_id,
                            is_eos = resp.is_eos,
                            emb_len = emb.len(),
                            "SamplerBridge: token sampled"
                        );
                        let _ = reply.send(Ok((resp, emb)));
                    }
                    Err(e) => {
                        warn!(%e, "SamplerBridge recv failed, dropping connection");
                        stream = None;
                        let _ = reply.send(Err(e));
                    }
                }
            }
            SamplerCommand::Shutdown => {
                info!("SamplerBridge shutting down");
                break;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_request_roundtrip() {
        let req = SampleRequest {
            session_id: "sess-001".into(),
            request_id: "req-001".into(),
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            seed: Some(42),
        };
        let activation = vec![1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();

        let wire = encode_request(&req, &activation).unwrap();

        // Verify we can decode the header.
        let header_len =
            u32::from_le_bytes(wire[0..4].try_into().unwrap()) as usize;
        let decoded: SampleRequest =
            ciborium::from_reader(&wire[4..4 + header_len]).unwrap();

        assert_eq!(decoded.session_id, "sess-001");
        assert_eq!(decoded.temperature, 0.7);
        assert_eq!(decoded.top_k, 50);
        assert_eq!(decoded.seed, Some(42));

        // Verify activation.
        let act_offset = 4 + header_len;
        let act_len = u32::from_le_bytes(
            wire[act_offset..act_offset + 4].try_into().unwrap(),
        ) as usize;
        assert_eq!(act_len, 12); // 3 floats × 4 bytes
    }

    #[test]
    fn test_sample_response_roundtrip() {
        let resp = SampleResponse {
            token_id: 42,
            is_eos: false,
            token_text: "hello".into(),
        };
        let embedding = vec![0.1f32, 0.2, 0.3, 0.4]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();

        // Encode manually (simulating Python side).
        let mut header_bytes = Vec::new();
        ciborium::into_writer(&resp, &mut header_bytes).unwrap();
        let mut wire = Vec::new();
        wire.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
        wire.extend_from_slice(&header_bytes);
        wire.extend_from_slice(&(embedding.len() as u32).to_le_bytes());
        wire.extend_from_slice(&embedding);

        let (decoded, emb) = decode_response(&wire).unwrap();
        assert_eq!(decoded.token_id, 42);
        assert!(!decoded.is_eos);
        assert_eq!(decoded.token_text, "hello");
        assert_eq!(emb.len(), 16); // 4 floats × 4 bytes
    }

    #[test]
    fn test_sample_request_omits_defaults() {
        let req = SampleRequest {
            session_id: "s".into(),
            request_id: "r".into(),
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            seed: None,
        };
        let wire = encode_request(&req, &[]).unwrap();
        // With defaults omitted, wire should be small.
        assert!(wire.len() < 50);
    }

    #[tokio::test]
    async fn test_sampler_bridge_roundtrip() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let socket_path = format!(
            "/tmp/openhydra-test-sampler-{}.sock",
            std::process::id()
        );
        std::fs::remove_file(&socket_path).ok();

        // Start a mock HeadSampler (bind first, like the real Python side).
        let sp = socket_path.clone();
        let mock_handle = tokio::spawn(async move {
            let listener = tokio::net::UnixListener::bind(&sp).unwrap();
            let (mut stream, _) = listener.accept().await.unwrap();

            // Read one request.
            let mut len_buf = [0u8; 4];
            stream.read_exact(&mut len_buf).await.unwrap();
            let msg_len = u32::from_le_bytes(len_buf) as usize;
            let mut body = vec![0u8; msg_len];
            stream.read_exact(&mut body).await.unwrap();

            // Build response: token_id=42, is_eos=false, 4-float embedding.
            let resp = SampleResponse {
                token_id: 42,
                is_eos: false,
                token_text: "hello".into(),
            };
            let embedding: Vec<u8> = vec![0.1f32, 0.2, 0.3, 0.4]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();

            let mut hdr_bytes = Vec::new();
            ciborium::into_writer(&resp, &mut hdr_bytes).unwrap();
            let mut wire = Vec::new();
            wire.extend_from_slice(&(hdr_bytes.len() as u32).to_le_bytes());
            wire.extend_from_slice(&hdr_bytes);
            wire.extend_from_slice(&(embedding.len() as u32).to_le_bytes());
            wire.extend_from_slice(&embedding);

            let resp_len = wire.len() as u32;
            stream.write_all(&resp_len.to_le_bytes()).await.unwrap();
            stream.write_all(&wire).await.unwrap();
            stream.flush().await.unwrap();
        });

        // Give the mock time to bind.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let bridge = SamplerBridge::start(&socket_path).await.unwrap();

        let req = SampleRequest {
            session_id: "sess-rt".into(),
            request_id: "req-rt".into(),
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            seed: None,
        };
        let activation: Vec<u8> = vec![1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let (resp, emb) = bridge.sample(req, activation).await.unwrap();

        assert_eq!(resp.token_id, 42);
        assert!(!resp.is_eos);
        assert_eq!(resp.token_text, "hello");
        assert_eq!(emb.len(), 16);

        bridge.shutdown().await;
        mock_handle.await.unwrap();
        std::fs::remove_file(&socket_path).ok();
    }
}
