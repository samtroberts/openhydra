//! Composed NetworkBehaviour for OpenHydra peers.
//!
//! Combines Kademlia DHT, Circuit Relay v2, DCUtR hole punching,
//! AutoNAT, Identify, mDNS, and gRPC proxy into a single behaviour.

use libp2p::request_response;
use libp2p::swarm::behaviour::toggle::Toggle;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{autonat, dcutr, gossipsub, identify, kad, mdns, ping, relay};

use crate::proxy::GrpcProxyCodec;

/// Connections one remote peer may hold with us at once. Normal use stays well under it: QUIC and
/// TCP on each address family, a circuit through each production relay, a hole-punched upgrade and an
/// AutoNAT dial-back (the dedups close extras); past it, a peer is refused more.
pub const MAX_ESTABLISHED_PER_PEER: u32 = 16;
/// A peer node: incoming connections still being set up (handshakes), and established ones.
pub const PEER_MAX_PENDING_INCOMING: u32 = 128;
pub const PEER_MAX_ESTABLISHED_INCOMING: u32 = 1024;
/// A peer node: all established connections, both directions.
pub const PEER_MAX_ESTABLISHED_TOTAL: u32 = 2048;
/// A bootstrap: every node of the network may connect to it.
pub const BOOTSTRAP_MAX_PENDING_INCOMING: u32 = 512;
pub const BOOTSTRAP_MAX_ESTABLISHED_INCOMING: u32 = 8192;

/// Bounds on how many connections other nodes can make a peer node hold (pending handshakes, and
/// established connections in all and per peer). Without them, connections under fresh identities
/// would pile up until the idle timeout.
pub fn peer_connection_limits() -> libp2p::connection_limits::ConnectionLimits {
    libp2p::connection_limits::ConnectionLimits::default()
        .with_max_pending_incoming(Some(PEER_MAX_PENDING_INCOMING))
        .with_max_established_incoming(Some(PEER_MAX_ESTABLISHED_INCOMING))
        .with_max_established(Some(PEER_MAX_ESTABLISHED_TOTAL))
        .with_max_established_per_peer(Some(MAX_ESTABLISHED_PER_PEER))
}

/// The same bounds for a bootstrap node, sized for the whole network connecting to it.
pub fn bootstrap_connection_limits() -> libp2p::connection_limits::ConnectionLimits {
    libp2p::connection_limits::ConnectionLimits::default()
        .with_max_pending_incoming(Some(BOOTSTRAP_MAX_PENDING_INCOMING))
        .with_max_established_incoming(Some(BOOTSTRAP_MAX_ESTABLISHED_INCOMING))
        .with_max_established_per_peer(Some(MAX_ESTABLISHED_PER_PEER))
}

/// The composed behaviour for an OpenHydra peer node.
#[derive(NetworkBehaviour)]
pub struct OpenHydraBehaviour {
    /// Bounds on the connections other nodes can make us hold ([`peer_connection_limits`]).
    pub connection_limits: libp2p::connection_limits::Behaviour,
    /// Kademlia DHT for peer discovery.
    pub kademlia: kad::Behaviour<kad::store::MemoryStore>,
    /// Circuit Relay v2 client — connects through relays when behind NAT.
    pub relay_client: relay::client::Behaviour,
    /// WS-F F-4: optional Circuit Relay v2 SERVER. Disabled (`Toggle::None`)
    /// unless this node opts into being a temporary peer-relay (operator flag,
    /// for publicly-reachable nodes). When enabled it accepts reservations from
    /// NATted peers and enforces the SAME caps + F-6 leech lockout as the Linode
    /// bootstraps. Noise keeps relayed payloads E2E-encrypted (transport-only).
    pub relay_server: Toggle<relay::Behaviour>,
    /// DCUtR — Direct Connection Upgrade through Relay (hole punching).
    pub dcutr: dcutr::Behaviour,
    /// AutoNAT **v2 client** (the only AutoNAT — v1 retired). Asks v2 servers to
    /// dial-back a *specific* candidate address and returns a per-address verdict
    /// (`result: Ok` ⇒ that exact address is reachable by an arbitrary peer). This
    /// fixes the two v1 weaknesses the live test exposed: v1 couldn't reliably emit
    /// a negative verdict (it timed out into `Unknown`, so demotion never fired)
    /// and it tested only one address family. Drives R-DHT-2 promotion on a
    /// positive per-address result, and the consumer-facing `nat_info` /
    /// UPnP-re-assert suppression that v1's `StatusChanged` used to own.
    pub autonat_v2_client: autonat::v2::client::Behaviour,
    /// Identify — exchange peer metadata on connect.
    pub identify: identify::Behaviour,
    /// mDNS — zero-config LAN peer discovery.
    pub mdns: mdns::tokio::Behaviour,
    /// gRPC proxy — tunnels gRPC through libp2p for cross-ISP inference.
    pub grpc_proxy: request_response::Behaviour<GrpcProxyCodec>,
    /// C7: `/openhydra/registry/1.0.0` — consumer→bootstrap "who serves model X?" query. This
    /// node is Outbound-only (it asks; bootstraps answer from their `ProviderRegistry`). Fills
    /// the cross-NAT discovery gap where the D-sized gossip mesh doesn't forward a provider's
    /// advert to this specific consumer and the NAT'd provider's DHT `put_record` times out.
    pub registry_query: request_response::Behaviour<crate::registry_proto::RegistryCodec>,
    /// Gossipsub (PR-3 / B1) — swarm-wide event bus on a single topic,
    /// ``openhydra/swarm/v1/events``. Used for:
    /// * ``PEER_DEAD`` — broadcast when a peer observes another peer's
    ///   gRPC socket drop mid-generation. Drives sub-2 s failover.
    /// * ``REQUEST_HOLE_PUNCH`` — pairs with A3's DCUtR fix: a peer asks
    ///   a specific counterpart to dial back *now*, giving both sides a
    ///   coordinated simultaneous-dial window even behind symmetric NAT.
    pub gossipsub: gossipsub::Behaviour,
    /// Ping keepalive — sends periodic pings on ALL connections (including
    /// relay circuits) to prevent NAT mapping eviction.  Without this,
    /// mobile hotspot NAT drops TCP mappings during the 1-3 s inference
    /// silence, killing the relay circuit between tokens.
    pub ping: ping::Behaviour,
    /// R-DHT-4: UPnP/NAT-PMP port mapping. On a home router that supports IGD,
    /// this maps our listen port and, on success, emits
    /// ``ToSwarm::ExternalAddrConfirmed`` for the mapped public address — which
    /// flows through the R-DHT-2 promotion path (AutoNAT-independent reachability
    /// for full/restricted-cone home NATs). Does nothing useful behind CGNAT
    /// (the gateway itself is NATed → ``NonRoutableGateway``).
    pub upnp: libp2p::upnp::tokio::Behaviour,
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use libp2p::core::transport::{MemoryTransport, Transport};
    use libp2p::core::upgrade::Version;
    use libp2p::swarm::dial_opts::{DialOpts, PeerCondition};
    use libp2p::swarm::{dummy, Swarm, SwarmEvent};
    use libp2p::{identity, noise, yamux, Multiaddr, PeerId};
    use std::time::Duration;

    #[derive(NetworkBehaviour)]
    #[behaviour(prelude = "libp2p::swarm::derive_prelude")]
    struct Limited {
        limits: libp2p::connection_limits::Behaviour,
        keep: dummy::Behaviour,
    }

    fn node() -> Swarm<Limited> {
        let key = identity::Keypair::generate_ed25519();
        let transport = MemoryTransport::default()
            .upgrade(Version::V1)
            .authenticate(noise::Config::new(&key).unwrap())
            .multiplex(yamux::Config::default())
            .boxed();
        let behaviour = Limited {
            limits: libp2p::connection_limits::Behaviour::new(peer_connection_limits()),
            keep: dummy::Behaviour,
        };
        let cfg = libp2p::swarm::Config::with_tokio_executor().with_idle_connection_timeout(Duration::from_secs(60));
        Swarm::new(transport, behaviour, PeerId::from(key.public()), cfg)
    }

    #[tokio::test]
    async fn one_peer_cannot_make_us_hold_more_than_the_per_peer_limit() {
        let mut a = node();
        let mut b = node();
        let addr: Multiaddr = format!("/memory/{}", rand::random::<u64>()).parse().unwrap();
        a.listen_on(addr.clone()).unwrap();
        let a_id = *a.local_peer_id();
        // Wait for the listener before dialing.
        loop {
            if let SwarmEvent::NewListenAddr { .. } = a.select_next_some().await {
                break;
            }
        }
        let attempts = MAX_ESTABLISHED_PER_PEER + 8;
        for _ in 0..attempts {
            b.dial(DialOpts::peer_id(a_id).addresses(vec![addr.clone()]).condition(PeerCondition::Always).build()).unwrap();
        }
        // Drive both until every dial has resolved one way or the other.
        let (mut ok, mut refused) = (0u32, 0u32);
        let deadline = tokio::time::sleep(Duration::from_secs(10));
        tokio::pin!(deadline);
        while ok + refused < attempts {
            tokio::select! {
                ev = b.select_next_some() => match ev {
                    SwarmEvent::ConnectionEstablished { .. } => ok += 1,
                    SwarmEvent::OutgoingConnectionError { .. } => refused += 1,
                    _ => {}
                },
                _ = a.select_next_some() => {}
                _ = &mut deadline => break,
            }
        }
        // Let A finish closing the ones its limit refused.
        let settle = tokio::time::sleep(Duration::from_millis(500));
        tokio::pin!(settle);
        loop {
            tokio::select! {
                _ = a.select_next_some() => {}
                _ = b.select_next_some() => {}
                _ = &mut settle => break,
            }
        }
        let held = a.network_info().connection_counters().num_established();
        assert_eq!(held, MAX_ESTABLISHED_PER_PEER, "dialed {attempts}, b saw {ok} established / {refused} refused");
    }

    #[test]
    fn limits_leave_room_for_normal_use() {
        // QUIC + TCP on both families, four relay circuits, DCUtR, an AutoNAT dial-back.
        assert!(MAX_ESTABLISHED_PER_PEER >= 2 + 2 + 4 + 1 + 1);
        assert!(PEER_MAX_ESTABLISHED_INCOMING <= PEER_MAX_ESTABLISHED_TOTAL);
        assert!(BOOTSTRAP_MAX_ESTABLISHED_INCOMING >= PEER_MAX_ESTABLISHED_INCOMING);
    }
}
