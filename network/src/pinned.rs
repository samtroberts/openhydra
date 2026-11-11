//! Where the gateway's own provider listens on this machine.
//!
//! The desktop app runs a provider and a gateway side by side, and tells the gateway the provider's
//! peer id (`serve --self-provider`). Without its address, the gateway had to find it like any other
//! peer, through mDNS, the DHT or a relay circuit, and with LAN discovery off, offline, on a network
//! that blocks multicast or on a keyed network none of them may give one: every request to the
//! machine's own models then failed with "peer unreachable".
//!
//! A pinned address is a loopback address the gateway dials its own provider on, directly. It is
//! kept in the event loop only: never written to Kademlia, the address book, identify, adverts or any
//! DHT answer, so it never leaves this node. Loopback only, because a self provider gets self-serve
//! treatment (no receipts, first place in `openhydra/auto`, the Device and Private models): "self"
//! must stay this machine.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use libp2p::multiaddr::Protocol;
use libp2p::{Multiaddr, PeerId};

/// The pinned addresses of one node, shared by its event loop (which dials them) and its Kademlia
/// address filter (which keeps them out of the routing table: [`kad_allows`]).
pub type PinnedPeers = Arc<RwLock<HashMap<PeerId, Vec<Multiaddr>>>>;

/// Whether Kademlia may keep `addr` for `peer`: not a loopback address of a pinned peer. Kademlia
/// adds the address of every connection it confirms, a pinned dial's included, and on a private
/// network no other filter would stop it, so this node would hand `127.0.0.1` out in DHT answers,
/// where it names the querier's own machine.
pub fn kad_allows(pinned: &PinnedPeers, peer: &PeerId, addr: &Multiaddr) -> bool {
    let loopback = match addr.iter().next() {
        Some(Protocol::Ip4(ip)) => ip.is_loopback(),
        Some(Protocol::Ip6(ip)) => ip.is_loopback(),
        _ => false,
    };
    !loopback || !pinned.read().unwrap_or_else(|e| e.into_inner()).contains_key(peer)
}

/// Pinned addresses per peer, at most.
pub const MAX_PINNED_ADDRS: usize = 4;

/// Check `addr` as an address of the own provider `provider` on this machine, and return it without
/// a trailing `/p2p/<id>`. Accepted: `/ip4/127.x.x.x` or `/ip6/::1`, then `/tcp/<port>` or
/// `/udp/<port>/quic-v1`, then optionally `/p2p/<provider>`. Anything else is refused with the reason.
pub fn self_provider_addr(addr: &Multiaddr, provider: &PeerId) -> Result<Multiaddr, String> {
    let mut parts = addr.iter();
    let mut out = Multiaddr::empty();
    match parts.next() {
        Some(p @ Protocol::Ip4(ip)) if ip.is_loopback() => out.push(p),
        Some(p @ Protocol::Ip6(ip)) if ip.is_loopback() => out.push(p),
        Some(Protocol::Ip4(_) | Protocol::Ip6(_)) => {
            return Err(format!("{addr}: not a loopback address (the own provider must be on this machine)"))
        }
        Some(Protocol::Dns(_) | Protocol::Dns4(_) | Protocol::Dns6(_) | Protocol::Dnsaddr(_)) => {
            return Err(format!("{addr}: a host name; give the loopback IP (127.0.0.1 or ::1)"))
        }
        _ => return Err(format!("{addr}: expected /ip4/127.0.0.1/... or /ip6/::1/...")),
    }
    match parts.next() {
        Some(p @ Protocol::Tcp(port)) if port != 0 => out.push(p),
        Some(p @ Protocol::Udp(port)) if port != 0 => {
            out.push(p);
            match parts.next() {
                Some(q @ Protocol::QuicV1) => out.push(q),
                _ => return Err(format!("{addr}: a UDP address must be /udp/<port>/quic-v1")),
            }
        }
        _ => return Err(format!("{addr}: expected /tcp/<port> or /udp/<port>/quic-v1 with a port")),
    }
    match parts.next() {
        None => {}
        Some(Protocol::P2p(id)) if id == *provider => {}
        Some(Protocol::P2p(id)) => {
            return Err(format!("{addr}: names peer {id}, but the own provider is {provider}"))
        }
        Some(_) => return Err(format!("{addr}: only /p2p/<own provider> may follow the port (no relay circuits)")),
    }
    if parts.next().is_some() {
        return Err(format!("{addr}: nothing may follow /p2p/<id>"));
    }
    Ok(out)
}

/// The pinned addresses to dial: the TCP ones when there are any, else the QUIC ones (only when QUIC
/// runs: on a keyed network without it they could only fail), at most [`MAX_PINNED_ADDRS`], no
/// repeats. TCP first because a stopped provider refuses a TCP dial at once, while a QUIC dial to a
/// closed port waits out its handshake timeout (5 s): every request to a stopped own provider took
/// that long to fail. On loopback QUIC gains nothing; it stays for a provider listening on QUIC alone.
pub fn dialable(addrs: &[Multiaddr], quic: bool) -> Vec<Multiaddr> {
    let is_quic = |a: &Multiaddr| a.iter().any(|p| matches!(p, Protocol::QuicV1));
    let tcp: Vec<&Multiaddr> = addrs.iter().filter(|a| !is_quic(a)).collect();
    let pick: Vec<&Multiaddr> = if !tcp.is_empty() {
        tcp
    } else if quic {
        addrs.iter().collect()
    } else {
        Vec::new()
    };
    let mut out: Vec<Multiaddr> = Vec::new();
    for a in pick {
        if !out.contains(a) && out.len() < MAX_PINNED_ADDRS {
            out.push(a.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ma(s: &str) -> Multiaddr {
        s.parse().unwrap()
    }

    #[test]
    fn loopback_tcp_and_quic_are_accepted_and_lose_the_peer_id() {
        let p = PeerId::random();
        assert_eq!(self_provider_addr(&ma("/ip4/127.0.0.1/tcp/4111"), &p).unwrap(), ma("/ip4/127.0.0.1/tcp/4111"));
        assert_eq!(
            self_provider_addr(&ma("/ip4/127.0.0.1/udp/4111/quic-v1"), &p).unwrap(),
            ma("/ip4/127.0.0.1/udp/4111/quic-v1")
        );
        assert_eq!(self_provider_addr(&ma("/ip6/::1/tcp/4111"), &p).unwrap(), ma("/ip6/::1/tcp/4111"));
        assert_eq!(self_provider_addr(&ma("/ip4/127.0.0.2/tcp/9"), &p).unwrap(), ma("/ip4/127.0.0.2/tcp/9"));
        assert_eq!(
            self_provider_addr(&ma(&format!("/ip4/127.0.0.1/tcp/4111/p2p/{p}")), &p).unwrap(),
            ma("/ip4/127.0.0.1/tcp/4111")
        );
    }

    #[test]
    fn anything_that_could_reach_another_machine_is_refused() {
        let p = PeerId::random();
        let other = PeerId::random();
        for bad in [
            "/ip4/192.168.1.5/tcp/4111".to_string(),
            "/ip4/10.0.0.1/tcp/4111".to_string(),
            "/ip4/45.79.190.172/tcp/4001".to_string(),
            "/ip4/0.0.0.0/tcp/4111".to_string(),
            "/ip6/fe80::1/tcp/4111".to_string(),
            "/ip6/::/tcp/4111".to_string(),
            "/dns4/localhost/tcp/4111".to_string(),
            "/ip4/127.0.0.1/tcp/0".to_string(),
            "/ip4/127.0.0.1/udp/4111".to_string(),
            "/ip4/127.0.0.1/udp/4111/quic".to_string(),
            "/ip4/127.0.0.1/tcp/4111/ws".to_string(),
            format!("/ip4/127.0.0.1/tcp/4111/p2p/{other}"),
            format!("/ip4/127.0.0.1/tcp/4001/p2p/{other}/p2p-circuit/p2p/{p}"),
            format!("/ip4/127.0.0.1/tcp/4111/p2p/{p}/p2p-circuit"),
            format!("/p2p/{p}"),
        ] {
            assert!(self_provider_addr(&ma(&bad), &p).is_err(), "{bad} must be refused");
        }
    }

    #[test]
    fn kademlia_never_keeps_a_pinned_peers_loopback_address() {
        let pinned = PinnedPeers::default();
        let (p, q) = (PeerId::random(), PeerId::random());
        let lo = ma("/ip4/127.0.0.1/tcp/4111");
        let lo6 = ma("/ip6/::1/tcp/4111");
        let lan = ma("/ip4/192.168.1.5/tcp/4111");
        assert!(kad_allows(&pinned, &p, &lo), "nothing pinned: as before");
        pinned.write().unwrap().insert(p, vec![lo.clone()]);
        assert!(!kad_allows(&pinned, &p, &lo));
        assert!(!kad_allows(&pinned, &p, &lo6));
        assert!(kad_allows(&pinned, &p, &lan), "its other addresses are judged as before");
        assert!(kad_allows(&pinned, &q, &lo), "other peers (a loopback test network) as before");
    }

    #[test]
    fn tcp_is_dialled_when_there_is_tcp_and_quic_only_when_it_runs() {
        let tcp = ma("/ip4/127.0.0.1/tcp/4111");
        let quic = ma("/ip4/127.0.0.1/udp/4111/quic-v1");
        let both = [quic.clone(), tcp.clone(), tcp.clone()];
        assert_eq!(dialable(&both, true), vec![tcp.clone()], "TCP fails at once when the provider is stopped");
        assert_eq!(dialable(&both, false), vec![tcp]);
        assert_eq!(dialable(&[quic.clone()], true), vec![quic.clone()], "a QUIC-only provider");
        assert!(dialable(&[quic], false).is_empty(), "no QUIC transport: nothing to dial it with");
        let many: Vec<Multiaddr> = (1..=9).map(|i| ma(&format!("/ip4/127.0.0.1/tcp/{i}"))).collect();
        assert_eq!(dialable(&many, true).len(), MAX_PINNED_ADDRS);
    }
}
