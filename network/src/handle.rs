// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! A tidy, synchronous Rust handle over the swarm — the API the `agent` crate drives.
//!
//! Wraps `start_node` + a `SwarmCommand` channel + the inbound `SharedProxyQueue`
//! so a Rust caller gets
//! `announce` / `discover` / `poll_inbound` / `respond` / `push` without touching tokio
//! channels. All methods are blocking and must be called from outside a tokio runtime
//! (a plain thread), like the agent's provider loop.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::event_loop::{SharedProxyQueue, SwarmCommand};
use crate::identity::Identity;
use crate::node::{send_and_wait, start_node, NodeConfig};
use crate::types::{DiscoveredPeer, PeerRecord, StatusSnapshot};

/// A lightweight, cloneable handle that can ONLY read the node's status snapshot —
/// safe to hand to a background thread (the agent's `--status-bind` HTTP server)
/// without moving the full [`NetworkHandle`] out of its owning role.
#[derive(Clone)]
pub struct StatusClient {
    cmd_tx: mpsc::Sender<SwarmCommand>,
}

impl StatusClient {
    /// One read-only snapshot of the node's live network state.
    pub fn status(&self) -> Result<StatusSnapshot, String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::Status { reply })
    }
}

/// A running swarm node, driven synchronously.
pub struct NetworkHandle {
    cmd_tx: mpsc::Sender<SwarmCommand>,
    proxy_queue: Arc<SharedProxyQueue>,
    _thread: std::thread::JoinHandle<()>,
    libp2p_peer_id: String,
    openhydra_peer_id: String,
    public_key_hex: String,
    /// The node's ed25519 identity keypair — used to sign receipts. **Never exported**
    /// (the agent signs via [`sign`](NetworkHandle::sign); only signatures cross out).
    keypair: libp2p::identity::Keypair,
    /// #42: bumped by the event loop on every `rebootstrap()` (network change).
    /// The provider run-loop reads it each poll iteration and re-announces its
    /// DHT record the moment it changes, so a roam/wake refreshes the record's
    /// relay addresses under the *same* pinned PeerId without waiting out the
    /// periodic re-announce interval.
    net_generation: Arc<std::sync::atomic::AtomicU64>,
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
        let keypair = identity.keypair.clone();
        let (cmd_tx, proxy_queue, thread, net_generation) = start_node(&config)?;
        Ok(Self {
            cmd_tx,
            proxy_queue,
            _thread: thread,
            libp2p_peer_id,
            openhydra_peer_id,
            public_key_hex,
            keypair,
            net_generation,
        })
    }

    /// #42: the current network generation. The event loop bumps this each time
    /// it rebuilds connectivity after a network change (roam / wake / interface
    /// up-down). A provider polls it and re-announces its models whenever the
    /// value advances — refreshing the DHT record's relay addresses under the
    /// same PeerId. A consumer can ignore it. Cheap (a relaxed atomic load).
    pub fn network_generation(&self) -> u64 {
        self.net_generation.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Sign `data` with the node's identity key (RFC-8032 ed25519, 64 bytes). The key
    /// never leaves the handle — used for co-signing receipts.
    pub fn sign(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        self.keypair.sign(data).map_err(|e| format!("sign: {e}"))
    }

    /// The node's raw 32-byte ed25519 public key (for receipt payloads).
    pub fn public_key_bytes(&self) -> Result<Vec<u8>, String> {
        let pk = self
            .keypair
            .public()
            .try_into_ed25519()
            .map_err(|e| format!("identity not ed25519: {e}"))?;
        Ok(pk.to_bytes().to_vec())
    }

    /// Derive the libp2p PeerId string for a raw 32-byte ed25519 public key. Lets the
    /// provider key per-peer state (M2.3 credit) by the same libp2p id a consumer
    /// announces (its `reply_to`), given only the ed25519 pubkey carried in a co-signed
    /// receipt. (Stateless — takes `&self` only to sit alongside the other identity helpers.)
    pub fn peer_id_from_ed25519_pubkey(&self, pubkey: &[u8]) -> Result<String, String> {
        let ed = libp2p::identity::ed25519::PublicKey::try_from_bytes(pubkey)
            .map_err(|e| format!("bad ed25519 pubkey: {e}"))?;
        let pk = libp2p::identity::PublicKey::from(ed);
        Ok(libp2p::PeerId::from_public_key(&pk).to_string())
    }

    /// The raw 32-byte ed25519 public key a libp2p PeerId string embeds (the inverse of
    /// [`Self::peer_id_from_ed25519_pubkey`]). Ed25519 peer ids inline their key, so the
    /// authenticated sender of a request names the key its receipts are signed with.
    pub fn ed25519_pubkey_from_peer_id(&self, peer_id: &str) -> Result<[u8; 32], String> {
        ed25519_pubkey_of_peer(peer_id)
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

    /// P0 introspection: one read-only snapshot of the node's live network state.
    pub fn status(&self) -> Result<StatusSnapshot, String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::Status { reply })
    }

    /// A cloneable read-only status handle for a background thread (the agent's
    /// `--status-bind` server) — see [`StatusClient`].
    pub fn status_client(&self) -> StatusClient {
        StatusClient { cmd_tx: self.cmd_tx.clone() }
    }

    /// Publish a peer record to the Kademlia DHT.
    pub fn announce(&self, record: PeerRecord) -> Result<(), String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::Announce { record, reply })?
    }

    /// M6: publish a `MODEL_WITHDRAWN` tombstone so consumers evict this node's `model_id`
    /// entry immediately (the operator un-shared it) instead of waiting out the discover-record
    /// TTL. Best-effort — a no-mesh publish is reported as `Ok` (the model still expires on the
    /// old TTL), so a benign early call never trips the provider's re-announce error path.
    pub fn withdraw_model(&self, model_id: impl Into<String>) -> Result<(), String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::WithdrawModel {
            model_id: model_id.into(),
            reply,
        })?
    }

    /// Discover providers serving `model_id` (empty on an empty DHT).
    pub fn discover(&self, model_id: impl Into<String>) -> Result<Vec<DiscoveredPeer>, String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::Discover {
            model_id: model_id.into(),
            reply,
        })?
    }

    /// C11: drop `model_id` from the discover cache so the next [`discover`](Self::discover)
    /// does a fresh `get_providers` lookup. The consumer calls this after every candidate from
    /// a (possibly cached) discovery failed, then retries discover once. Best-effort: an error
    /// here (event loop gone) just means the retry falls back to whatever discover returns.
    pub fn invalidate_discover(&self, model_id: impl Into<String>) -> Result<(), String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::InvalidateDiscover {
            model_id: model_id.into(),
            reply,
        })
    }

    /// C7: ask a connected bootstrap's provider registry "who serves `model_id`?". A best-effort
    /// supplementary discovery source for the cross-NAT case where DHT `get_providers` and gossip
    /// PEX both return nothing. Returned records are already re-verified + made dialable; an empty
    /// `Vec` means "no connected bootstrap / registry had nothing" (not an error to escalate).
    pub fn query_registry(&self, model_id: impl Into<String>) -> Result<Vec<DiscoveredPeer>, String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::QueryRegistry {
            model_id: model_id.into(),
            reply,
        })?
    }

    /// The distinct model ids this node currently knows about (PEX-learned / discovered
    /// providers). Empty until gossip/discovery has populated the cache. Powers the
    /// gateway's `GET /v1/models`.
    pub fn known_models(&self) -> Result<Vec<String>, String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::KnownModels { reply })
    }

    /// Models whose last known provider withdrew them in the last few minutes and that nobody serves
    /// now: the gateway drops these from its sticky model list at once.
    pub fn recently_withdrawn(&self) -> Result<Vec<String>, String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::RecentlyWithdrawn { reply })
    }

    /// The model ids the provider `libp2p_peer_id` is known to serve, from its own signed records
    /// (nobody else can publish one for it).
    pub fn models_of(&self, libp2p_peer_id: impl Into<String>) -> Result<Vec<String>, String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::ModelsOf { libp2p_peer_id: libp2p_peer_id.into(), reply })
    }

    /// Every (model, provider) pair this node currently knows about, for the gateway's
    /// `openhydra/auto` choice (which models its own provider serves, how many providers serve each).
    pub fn known_providers(&self) -> Result<Vec<crate::types::KnownProvider>, String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::KnownProviders { reply })
    }

    /// Forward `data` to `peer_id` and block for the one-shot response (the consumer's
    /// serve round-trip: `SERVE_REQUEST` → buffered framed completion).
    pub fn proxy_forward(&self, peer_id: String, data: Vec<u8>) -> Result<Vec<u8>, String> {
        send_and_wait(&self.cmd_tx, |reply| SwarmCommand::ProxyForward {
            peer_id,
            data,
            reply,
        })?
    }

    /// Like [`proxy_forward`](Self::proxy_forward) but bounded by `timeout`. A dead or
    /// unreachable provider otherwise blocks until libp2p's own request-response timeout
    /// (≈15s) — or, worst case, indefinitely — with no way for the caller to give up and
    /// try another provider. Runs the blocking round-trip on a worker thread and bounds
    /// the wait; on timeout the worker is abandoned (its late reply, if any, is dropped).
    pub fn proxy_forward_timeout(
        &self,
        peer_id: String,
        data: Vec<u8>,
        timeout: Duration,
    ) -> Result<Vec<u8>, String> {
        let (tx, rx) = std::sync::mpsc::channel();
        let cmd_tx = self.cmd_tx.clone();
        std::thread::spawn(move || {
            let res = send_and_wait(&cmd_tx, |reply| SwarmCommand::ProxyForward {
                peer_id,
                data,
                reply,
            });
            let _ = tx.send(res); // receiver may be gone (we timed out) — that's fine
        });
        match rx.recv_timeout(timeout) {
            Ok(Ok(inner)) => inner, // worker ran send_and_wait → the proxy result
            Ok(Err(e)) => Err(e),   // event loop / channel failure
            Err(_) => Err(format!("proxy_forward timed out after {timeout:?}")),
        }
    }

    /// Block up to `timeout` for the next inbound proxy request
    /// `(request_id, source_peer, data)`. E-S8: `source_peer` is the
    /// libp2p-authenticated sender (or our own id for loopback), suitable as a
    /// per-peer rate-limit key.
    pub fn poll_inbound(&self, timeout: Duration) -> Option<crate::event_loop::InboundProxyItem> {
        self.proxy_queue.pop(timeout)
    }

    /// Send the one-shot response for an inbound request (used as an ACK; the streamed
    /// body is delivered separately via [`push`](Self::push)).
    pub fn respond(&self, request_id: String, data: Vec<u8>) -> Result<(), String> {
        self.cmd_tx
            .blocking_send(SwarmCommand::RespondProxy { request_id, data, on_sent: None })
            .map_err(|_| "swarm not running".to_string())
    }

    /// [`respond`](Self::respond), then call `on_sent` once with whether the response reached the
    /// peer (libp2p reported it sent) or not. It runs on the swarm's event loop: keep it short, and
    /// never call back into this handle from it.
    pub fn respond_then(
        &self,
        request_id: String,
        data: Vec<u8>,
        on_sent: crate::event_loop::OnSent,
    ) -> Result<(), String> {
        self.cmd_tx
            .blocking_send(SwarmCommand::RespondProxy { request_id, data, on_sent: Some(on_sent) })
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

/// See [`NetworkHandle::ed25519_pubkey_from_peer_id`].
pub fn ed25519_pubkey_of_peer(peer_id: &str) -> Result<[u8; 32], String> {
    let peer: libp2p::PeerId = peer_id.parse().map_err(|e| format!("bad peer id: {e}"))?;
    let mh: &libp2p::multihash::Multihash<64> = peer.as_ref();
    if mh.code() != 0x00 {
        return Err("peer id doesn't inline its key".into());
    }
    let pk = libp2p::identity::PublicKey::try_decode_protobuf(mh.digest())
        .map_err(|e| format!("bad public key in peer id: {e}"))?;
    let ed = pk.try_into_ed25519().map_err(|e| format!("peer key not ed25519: {e}"))?;
    Ok(ed.to_bytes())
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
            enable_connection_reversal: false,
            pcp_gateway: None,
            public_relays: crate::relay::PublicRelays::Off,
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

    #[test]
    fn status_snapshot_is_sane_on_a_fresh_node() {
        let dir = tempfile::tempdir().unwrap();
        let net = NetworkHandle::start(loopback_config(dir.path())).unwrap();
        // Give the swarm a beat to bind its loopback listeners.
        std::thread::sleep(Duration::from_millis(400));
        let snap = net.status().unwrap();
        assert!(!snap.listen_addrs.is_empty(), "loopback listeners should be up");
        assert!(!snap.kad_server_mode, "fresh nodes start as Kad clients (R-DHT-2)");
        assert_eq!(snap.network_generation, 0);
        assert!(snap.peers.is_empty(), "no peers on an isolated node");
        assert!(snap.known_models.is_empty());
        // The cloneable read-only client returns the same shape from another thread.
        let client = net.status_client();
        let handle = std::thread::spawn(move || client.status().unwrap());
        let snap2 = handle.join().unwrap();
        assert_eq!(snap2.listen_addrs.len(), snap.listen_addrs.len());
    }

    #[test]
    fn an_ed25519_peer_id_gives_back_its_key() {
        let key = libp2p::identity::Keypair::generate_ed25519();
        let peer = key.public().to_peer_id().to_string();
        let ed = key.public().try_into_ed25519().unwrap().to_bytes();
        assert_eq!(super::ed25519_pubkey_of_peer(&peer), Ok(ed));
        assert!(super::ed25519_pubkey_of_peer("not a peer id").is_err());
    }
}
