// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! A tidy, synchronous Rust handle over the swarm — the API the pure-protocol `agent`
//! crate drives (it builds `default-features = false`, so no pyo3/Python).
//!
//! This is the same plumbing the pyo3 `P2PNode` uses (`start_node` + a `SwarmCommand`
//! channel + the inbound `SharedProxyQueue`), wrapped so a Rust caller gets
//! `announce` / `discover` / `poll_inbound` / `respond` / `push` without touching tokio
//! channels. All methods are blocking and must be called from outside a tokio runtime
//! (a plain thread), like the agent's provider loop.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::event_loop::{SharedProxyQueue, SwarmCommand};
use crate::identity::Identity;
use crate::node::{send_and_wait, start_node, NodeConfig};
use crate::types::{DiscoveredPeer, PeerRecord};

/// A running swarm node, driven synchronously.
pub struct NetworkHandle {
    cmd_tx: mpsc::Sender<SwarmCommand>,
    proxy_queue: Arc<SharedProxyQueue>,
    _thread: std::thread::JoinHandle<()>,
    libp2p_peer_id: String,
    openhydra_peer_id: String,
    public_key_hex: String,
}

impl NetworkHandle {
    /// Load (or create) the identity, start the swarm, and cache the node's ids.
    pub fn start(config: NodeConfig) -> Result<Self, String> {
        let identity =
            Identity::load_or_create(&config.identity_path).map_err(|e| format!("identity: {e}"))?;
        let libp2p_peer_id = identity.libp2p_peer_id.to_string();
        let openhydra_peer_id = identity.openhydra_peer_id.clone();
        let public_key_hex = {
            let pk = identity
                .keypair
                .public()
                .try_into_ed25519()
                .map_err(|e| format!("identity not ed25519: {e}"))?;
            hex::encode(pk.to_bytes())
        };
        let (cmd_tx, proxy_queue, thread) = start_node(&config)?;
        Ok(Self {
            cmd_tx,
            proxy_queue,
            _thread: thread,
            libp2p_peer_id,
            openhydra_peer_id,
            public_key_hex,
        })
    }

    /// libp2p PeerId (the `proxy_forward` dial target / `reply_to` for serve requests).
    pub fn libp2p_peer_id(&self) -> &str {
        &self.libp2p_peer_id
    }
    /// OpenHydra peer id (the reputation / record key).
    pub fn openhydra_peer_id(&self) -> &str {
        &self.openhydra_peer_id
    }
    /// Hex-encoded 32-byte ed25519 public key (for `PeerRecord.public_key` / receipts).
    pub fn public_key_hex(&self) -> &str {
        &self.public_key_hex
    }

    /// Publish a peer record to the Kademlia DHT.
    pub fn announce(&self, record: PeerRecord) -> Result<(), String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::Announce { record, reply })?
    }

    /// Discover providers serving `model_id` (empty on an empty DHT).
    pub fn discover(&self, model_id: impl Into<String>) -> Result<Vec<DiscoveredPeer>, String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::Discover {
            model_id: model_id.into(),
            reply,
        })?
    }

    /// Block up to `timeout` for the next inbound proxy request `(request_id, data)`.
    pub fn poll_inbound(&self, timeout: Duration) -> Option<(String, Vec<u8>)> {
        self.proxy_queue.pop(timeout)
    }

    /// Send the one-shot response for an inbound request (used as an ACK; the streamed
    /// body is delivered separately via [`push`](Self::push)).
    pub fn respond(&self, request_id: String, data: Vec<u8>) -> Result<(), String> {
        self.cmd_tx
            .blocking_send(SwarmCommand::RespondProxy { request_id, data })
            .map_err(|_| "swarm not running".to_string())
    }

    /// Fire-and-forget push of raw bytes to a peer (used to stream serve chunks back to
    /// the consumer's `reply_to`, since the inbound response is one-shot).
    pub fn push(&self, peer_id: String, data: Vec<u8>) -> Result<(), String> {
        self.cmd_tx
            .blocking_send(SwarmCommand::ProxyForwardNoWait { peer_id, data })
            .map_err(|_| "swarm not running".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loopback_config(dir: &std::path::Path) -> NodeConfig {
        NodeConfig {
            identity_path: dir.join("id.key"),
            // Ephemeral loopback ports: no fixed ports, no LAN exposure.
            listen_addrs: vec![
                "/ip4/127.0.0.1/tcp/0".into(),
                "/ip4/127.0.0.1/udp/0/quic-v1".into(),
            ],
            bootstrap_peers: vec![],
            enable_peer_relay: false,
        }
    }

    #[test]
    fn starts_exposes_identity_and_discovers_empty() {
        let dir = tempfile::tempdir().unwrap();
        let net = NetworkHandle::start(loopback_config(dir.path())).unwrap();
        assert!(!net.libp2p_peer_id().is_empty());
        assert!(!net.openhydra_peer_id().is_empty());
        assert_eq!(net.public_key_hex().len(), 64); // 32-byte ed25519 key, hex
        // Empty DHT → no providers; returns promptly (must not hang).
        let peers = net.discover("openhydra-smoke-nonexistent").unwrap();
        assert!(peers.is_empty());
    }
}
