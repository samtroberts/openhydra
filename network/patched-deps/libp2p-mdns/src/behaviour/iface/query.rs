// Copyright 2018 Parity Technologies (UK) Ltd.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

use super::dns;
use crate::{META_QUERY_SERVICE_FQDN, SERVICE_NAME_FQDN};
use hickory_proto::{
    op::Message,
    rr::{Name, RData},
};
use libp2p_core::multiaddr::{Multiaddr, Protocol};
use libp2p_swarm::_address_translation;

use libp2p_identity::PeerId;
use std::time::Instant;
use std::{fmt, net::SocketAddr, str, time::Duration};

/// A valid mDNS packet received by the service.
#[derive(Debug)]
pub(crate) enum MdnsPacket {
    /// A query made by a remote.
    Query(MdnsQuery),
    /// A response sent by a remote in response to one of our queries.
    Response(MdnsResponse),
    /// A request for service discovery.
    ServiceDiscovery(MdnsServiceDiscovery),
}

impl MdnsPacket {
    pub(crate) fn new_from_bytes(
        buf: &[u8],
        from: SocketAddr,
    ) -> Result<Option<MdnsPacket>, hickory_proto::error::ProtoError> {
        if !header_counts_fit(buf) {
            return Ok(None);
        }
        let packet = Message::from_vec(buf)?;

        if packet.query().is_none() {
            return Ok(Some(MdnsPacket::Response(MdnsResponse::new(&packet, from))));
        }

        if packet
            .queries()
            .iter()
            .any(|q| q.name().to_utf8() == SERVICE_NAME_FQDN)
        {
            return Ok(Some(MdnsPacket::Query(MdnsQuery {
                from,
                query_id: packet.header().id(),
            })));
        }

        if packet
            .queries()
            .iter()
            .any(|q| q.name().to_utf8() == META_QUERY_SERVICE_FQDN)
        {
            // TODO: what if multiple questions, one with SERVICE_NAME and one with META_QUERY_SERVICE?
            return Ok(Some(MdnsPacket::ServiceDiscovery(MdnsServiceDiscovery {
                from,
                query_id: packet.header().id(),
            })));
        }

        Ok(None)
    }
}

/// A received mDNS query.
pub(crate) struct MdnsQuery {
    /// Sender of the address.
    from: SocketAddr,
    /// Id of the received DNS query. We need to pass this ID back in the results.
    query_id: u16,
}

impl MdnsQuery {
    /// Source address of the packet.
    pub(crate) fn remote_addr(&self) -> &SocketAddr {
        &self.from
    }

    /// Query id of the packet.
    pub(crate) fn query_id(&self) -> u16 {
        self.query_id
    }
}

impl fmt::Debug for MdnsQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MdnsQuery")
            .field("from", self.remote_addr())
            .field("query_id", &self.query_id)
            .finish()
    }
}

/// A received mDNS service discovery query.
pub(crate) struct MdnsServiceDiscovery {
    /// Sender of the address.
    from: SocketAddr,
    /// Id of the received DNS query. We need to pass this ID back in the results.
    query_id: u16,
}

impl MdnsServiceDiscovery {
    /// Source address of the packet.
    pub(crate) fn remote_addr(&self) -> &SocketAddr {
        &self.from
    }

    /// Query id of the packet.
    pub(crate) fn query_id(&self) -> u16 {
        self.query_id
    }
}

impl fmt::Debug for MdnsServiceDiscovery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MdnsServiceDiscovery")
            .field("from", self.remote_addr())
            .field("query_id", &self.query_id)
            .finish()
    }
}

/// A received mDNS response.
pub(crate) struct MdnsResponse {
    peers: Vec<MdnsPeer>,
    from: SocketAddr,
}

impl MdnsResponse {
    /// Creates a new `MdnsResponse` based on the provided `Packet`.
    pub(crate) fn new(packet: &Message, from: SocketAddr) -> MdnsResponse {
        let peers = packet
            .answers()
            .iter()
            .filter_map(|record| {
                if record.name().to_string() != SERVICE_NAME_FQDN {
                    return None;
                }

                let RData::PTR(record_value) = record.data()? else {
                    return None;
                };

                MdnsPeer::new(packet, record_value, record.ttl())
            })
            .collect();

        MdnsResponse { peers, from }
    }

    pub(crate) fn extract_discovered(
        &self,
        now: Instant,
        local_peer_id: PeerId,
    ) -> impl Iterator<Item = (PeerId, Multiaddr, Instant)> + '_ {
        self.discovered_peers()
            .filter(move |peer| peer.id() != &local_peer_id)
            .flat_map(move |peer| {
                let observed = self.observed_address();
                let new_expiration = now + peer.ttl();

                peer.addresses().iter().filter_map(move |address| {
                    // OpenHydra patch: a relay-circuit address can't be reached directly on the
                    // LAN; with its first IP rewritten to the sender's (below), it would be a dial
                    // to the sender that expects the relay's peer id, and always fail. Patched
                    // senders no longer advertise them; this drops those from older ones.
                    if address.iter().any(|p| matches!(p, Protocol::P2pCircuit)) {
                        return None;
                    }
                    let new_addr = _address_translation(address, &observed)?;
                    let new_addr = new_addr.with_p2p(*peer.id()).ok()?;

                    Some((*peer.id(), new_addr, new_expiration))
                })
            })
    }

    /// Source address of the packet.
    pub(crate) fn remote_addr(&self) -> &SocketAddr {
        &self.from
    }

    fn observed_address(&self) -> Multiaddr {
        // We replace the IP address with the address we observe the
        // remote as and the address they listen on.
        let obs_ip = Protocol::from(self.remote_addr().ip());
        let obs_port = Protocol::Udp(self.remote_addr().port());

        Multiaddr::empty().with(obs_ip).with(obs_port)
    }

    /// Returns the list of peers that have been reported in this packet.
    ///
    /// > **Note**: Keep in mind that this will also contain the responses we sent ourselves.
    fn discovered_peers(&self) -> impl Iterator<Item = &MdnsPeer> {
        self.peers.iter()
    }
}

impl fmt::Debug for MdnsResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MdnsResponse")
            .field("from", self.remote_addr())
            .finish()
    }
}

/// A peer discovered by the service.
pub(crate) struct MdnsPeer {
    addrs: Vec<Multiaddr>,
    /// Id of the peer.
    peer_id: PeerId,
    /// TTL of the record in seconds.
    ttl: u32,
}

impl MdnsPeer {
    /// Creates a new `MdnsPeer` based on the provided `Packet`.
    pub(crate) fn new(packet: &Message, record_value: &Name, ttl: u32) -> Option<MdnsPeer> {
        let mut my_peer_id: Option<PeerId> = None;
        let addrs = packet
            .additionals()
            .iter()
            .filter_map(|add_record| {
                if add_record.name() != record_value {
                    return None;
                }

                if let Some(RData::TXT(ref txt)) = add_record.data() {
                    Some(txt)
                } else {
                    None
                }
            })
            .flat_map(|txt| txt.iter())
            .filter_map(|txt| {
                // TODO: wrong, txt can be multiple character strings
                let addr = dns::decode_character_string(txt).ok()?;

                if !addr.starts_with(b"dnsaddr=") {
                    return None;
                }

                let mut addr = str::from_utf8(&addr[8..]).ok()?.parse::<Multiaddr>().ok()?;

                match addr.pop() {
                    Some(Protocol::P2p(peer_id)) => {
                        if let Some(pid) = &my_peer_id {
                            if peer_id != *pid {
                                return None;
                            }
                        } else {
                            my_peer_id.replace(peer_id);
                        }
                    }
                    _ => return None,
                };
                Some(addr)
            })
            .collect();

        my_peer_id.map(|peer_id| MdnsPeer {
            addrs,
            peer_id,
            ttl,
        })
    }

    /// Returns the id of the peer.
    #[inline]
    pub(crate) fn id(&self) -> &PeerId {
        &self.peer_id
    }

    /// Returns the requested time-to-live for the record.
    #[inline]
    pub(crate) fn ttl(&self) -> Duration {
        Duration::from_secs(u64::from(self.ttl))
    }

    /// Returns the list of addresses the peer says it is listening on.
    ///
    /// Filters out invalid addresses.
    pub(crate) fn addresses(&self) -> &Vec<Multiaddr> {
        &self.addrs
    }
}

impl fmt::Debug for MdnsPeer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MdnsPeer")
            .field("peer_id", &self.peer_id)
            .finish()
    }
}

/// OpenHydra patch (23 Nov 2026, RUSTSEC-2026-0491): whether the four counts in a DNS header fit in
/// `buf`. hickory-proto 0.24 reserves room for as many records as the header claims before reading
/// any, so a 12-byte packet claiming 65,535 records made it reserve ~18 MB; any device on the LAN can
/// send mDNS packets. On the wire a question takes at least 5 bytes (a 1-byte root name, type,
/// class) and a record at least 11 (name, type, class, TTL, data length), so a header claiming more
/// than its packet can hold is dropped unread. A packet too short for a header is left to the
/// decoder, which refuses it. The fix upstream is hickory-proto 0.26.2 (a libp2p upgrade).
fn header_counts_fit(buf: &[u8]) -> bool {
    if buf.len() < 12 {
        return true;
    }
    let count = |at: usize| u16::from_be_bytes([buf[at], buf[at + 1]]) as usize;
    let questions = count(4);
    let records = count(6) + count(8) + count(10);
    12 + questions * 5 + records * 11 <= buf.len()
}

#[cfg(test)]
mod tests {
    use super::super::dns::build_query_response;
    use super::*;

    #[test]
    fn a_header_claiming_more_records_than_the_packet_holds_is_dropped_unread() {
        let from: SocketAddr = "192.0.2.7:5353".parse().unwrap();
        // A response header (QR set) claiming 65,535 answers, and nothing after it.
        let mut claim = vec![0u8; 12];
        claim[2] = 0x84;
        claim[6] = 0xff;
        claim[7] = 0xff;
        assert!(!header_counts_fit(&claim));
        assert!(matches!(MdnsPacket::new_from_bytes(&claim, from), Ok(None)));
        // Real packets pass: our own response and query.
        let peer_id = PeerId::random();
        let addr: Multiaddr = "/ip4/1.2.3.4/tcp/5000".parse().unwrap();
        for packet in build_query_response(0x1234, peer_id, vec![&addr], Duration::from_secs(60)) {
            assert!(header_counts_fit(&packet));
        }
        assert!(header_counts_fit(&super::super::dns::build_query()));
        // Too short for a header: left to the decoder, which refuses it.
        assert!(header_counts_fit(&[0u8; 5]));
        assert!(MdnsPacket::new_from_bytes(&[0u8; 5], from).is_err());
    }

    #[test]
    fn test_create_mdns_peer() {
        let ttl = 300;
        let peer_id = PeerId::random();

        let mut addr1: Multiaddr = "/ip4/1.2.3.4/tcp/5000".parse().expect("bad multiaddress");
        let mut addr2: Multiaddr = "/ip6/::1/udp/10000".parse().expect("bad multiaddress");
        addr1.push(Protocol::P2p(peer_id));
        addr2.push(Protocol::P2p(peer_id));

        let packets = build_query_response(
            0xf8f8,
            peer_id,
            vec![&addr1, &addr2],
            Duration::from_secs(60),
        );

        for bytes in packets {
            let packet = Message::from_vec(&bytes).expect("unable to parse packet");
            let record_value = packet
                .answers()
                .iter()
                .filter_map(|record| {
                    if record.name().to_utf8() != SERVICE_NAME_FQDN {
                        return None;
                    }
                    let Some(RData::PTR(record_value)) = record.data() else {
                        return None;
                    };
                    Some(record_value)
                })
                .next()
                .expect("empty record value");

            let peer = MdnsPeer::new(&packet, record_value, ttl).expect("fail to create peer");
            assert_eq!(peer.peer_id, peer_id);
        }
    }

    // OpenHydra patch: a response's relay-circuit addresses aren't taken up (an older sender
    // still advertises them), its direct ones are, with the sender's observed IP.
    #[test]
    fn relay_circuit_addresses_are_not_discovered() {
        let peer_id = PeerId::random();
        let relay = PeerId::random();
        let direct: Multiaddr = "/ip4/192.168.0.6/tcp/4001".parse().unwrap();
        let circuit: Multiaddr = format!("/ip4/45.79.190.172/tcp/4001/p2p/{relay}/p2p-circuit")
            .parse()
            .unwrap();
        let packets = build_query_response(0x1234, peer_id, vec![&direct, &circuit], Duration::from_secs(60));
        let from: SocketAddr = "192.168.0.6:5353".parse().unwrap();
        let mut found = Vec::new();
        for bytes in packets {
            let Ok(Some(MdnsPacket::Response(response))) = MdnsPacket::new_from_bytes(&bytes, from) else {
                panic!("not a response");
            };
            found.extend(response.extract_discovered(Instant::now(), PeerId::random()).map(|(p, a, _)| (p, a)));
        }
        let expected: Multiaddr = format!("/ip4/192.168.0.6/tcp/4001/p2p/{peer_id}").parse().unwrap();
        assert_eq!(found, vec![(peer_id, expected)]);
    }
}
