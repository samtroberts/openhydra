// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! The patched request-response's "a request is arriving" mark (`oh_receiving_from`, which keeps the
//! idle sweep off a connection) is gone once the request is read or its read fails, however fast
//! (review 25 Nov: a read failing on its first poll was reported before the mark, which then stayed
//! until the connection closed, so a peer could keep its connections from being closed as idle).

use async_trait::async_trait;
use futures::prelude::*;
use libp2p::core::{transport::Transport, upgrade::Version};
use libp2p::identity::Keypair;
use libp2p::request_response as rr;
use libp2p::swarm::SwarmEvent;
use libp2p::{Multiaddr, PeerId, StreamProtocol, Swarm};
use std::{io, time::Duration};

#[derive(Clone, Default)]
struct Codec {
    /// `read_request` fails before reading anything (an oversized length prefix, say).
    fail_at_once: bool,
}

#[async_trait]
impl rr::Codec for Codec {
    type Protocol = StreamProtocol;
    type Request = u8;
    type Response = u8;

    async fn read_request<T: AsyncRead + Unpin + Send>(&mut self, _: &StreamProtocol, io: &mut T) -> io::Result<u8> {
        if self.fail_at_once {
            return Err(io::Error::other("bad length prefix"));
        }
        let mut b = Vec::new();
        io.read_to_end(&mut b).await?;
        b.first().copied().ok_or_else(|| io::Error::from(io::ErrorKind::UnexpectedEof))
    }

    async fn read_response<T: AsyncRead + Unpin + Send>(&mut self, _: &StreamProtocol, io: &mut T) -> io::Result<u8> {
        let mut b = Vec::new();
        io.read_to_end(&mut b).await?;
        Ok(b.first().copied().unwrap_or(0))
    }

    async fn write_request<T: AsyncWrite + Unpin + Send>(&mut self, _: &StreamProtocol, io: &mut T, r: u8) -> io::Result<()> {
        io.write_all(&[r]).await?;
        io.close().await
    }

    async fn write_response<T: AsyncWrite + Unpin + Send>(&mut self, _: &StreamProtocol, io: &mut T, r: u8) -> io::Result<()> {
        io.write_all(&[r]).await?;
        io.close().await
    }
}

fn swarm(fail_at_once: bool) -> Swarm<rr::Behaviour<Codec>> {
    let key = Keypair::generate_ed25519();
    let peer = PeerId::from(key.public());
    let transport = libp2p::tcp::tokio::Transport::default()
        .upgrade(Version::V1)
        .authenticate(libp2p::noise::Config::new(&key).unwrap())
        .multiplex(libp2p::yamux::Config::default())
        .boxed();
    let behaviour = rr::Behaviour::with_codec(
        Codec { fail_at_once },
        [(StreamProtocol::new("/oh-test/1"), rr::ProtocolSupport::Full)],
        rr::Config::default(),
    );
    Swarm::new(
        transport,
        behaviour,
        peer,
        libp2p::swarm::Config::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(60)),
    )
}

/// Whether the server still marks the client as sending, once the request's outcome is known.
async fn still_marked(fail_at_once: bool) -> bool {
    let mut server = swarm(fail_at_once);
    let mut client = swarm(false);
    server.listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap()).unwrap();
    let addr: Multiaddr = loop {
        if let SwarmEvent::NewListenAddr { address, .. } = server.select_next_some().await {
            break address;
        }
    };
    let (server_id, client_id) = (*server.local_peer_id(), *client.local_peer_id());
    client.behaviour_mut().add_address(&server_id, addr);
    client.behaviour_mut().send_request(&server_id, 1);
    let deadline = tokio::time::sleep(Duration::from_secs(10));
    tokio::pin!(deadline);
    let mut settled: Option<tokio::time::Instant> = None;
    loop {
        tokio::select! {
            e = server.select_next_some() => match e {
                // A read that fails at once produces no behaviour event: give it time to end.
                SwarmEvent::ConnectionEstablished { .. } if fail_at_once => {
                    settled.get_or_insert_with(|| tokio::time::Instant::now() + Duration::from_millis(1500));
                }
                SwarmEvent::Behaviour(rr::Event::Message { message: rr::Message::Request { channel, .. }, .. }) => {
                    let _ = server.behaviour_mut().send_response(channel, 2);
                }
                SwarmEvent::Behaviour(rr::Event::ResponseSent { .. } | rr::Event::InboundFailure { .. }) => {
                    // Let anything still queued in the connection reach the behaviour.
                    settled.get_or_insert_with(|| tokio::time::Instant::now() + Duration::from_millis(300));
                }
                _ => {}
            },
            _ = client.select_next_some() => {}
            _ = tokio::time::sleep_until(settled.unwrap_or_else(|| tokio::time::Instant::now() + Duration::from_secs(3600))), if settled.is_some() => {
                return server.behaviour().oh_receiving_from(&client_id);
            }
            _ = &mut deadline => panic!("the request never finished (fail_at_once: {fail_at_once})"),
        }
    }
}

#[tokio::test]
async fn a_read_or_its_failure_clears_the_arriving_mark() {
    assert!(!still_marked(false).await, "a request read and answered");
    assert!(!still_marked(true).await, "a request whose read failed at once");
}
