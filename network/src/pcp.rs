// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! PCP — Port Control Protocol (RFC 6887) client for opening an **inbound IPv6
//! firewall pinhole** (#43-W2).
//!
//! ## Why
//! A globally-routable public IPv6 address is *not* inbound-reachable: consumer
//! CPEs run a default-deny stateful IPv6 firewall that drops unsolicited inbound,
//! so AutoNAT v2's dial-back fails and the node stays relay-bound even with a
//! public v6 (see `docs/IPV6_REACHABILITY.md`). To promote such a node to a
//! relay/Kad server it must accept inbound from *arbitrary* peers with no prior
//! outbound flow — which pairwise hole-punching (DCUtR) cannot do. The node has
//! to actually open the CPE firewall. PCP is the IPv6-era successor to
//! NAT-PMP/UPnP that can request an inbound v6 firewall pinhole (RFC 6887 §11
//! MAP opcode); it is the v6 sibling of R-DHT-4's v4 UPnP/NAT-PMP mapping.
//!
//! ## Scope
//! This module is the pure wire codec (fully unit-tested) plus a best-effort
//! async UDP client. It is **opt-in** (a node only speaks PCP when an operator
//! supplies the CPE gateway address) and **not live-validated** — no
//! PCP-capable CPE was reachable during development; the codec is verified
//! against the RFC byte layout, the network round-trip is not.
//!
//! ## Wire layout (RFC 6887 §7 common header + §11 MAP opcode)
//! ```text
//! Request  = 24-byte common header + 36-byte MAP body  = 60 bytes
//! Response = 24-byte common header + 36-byte MAP body  = 60 bytes
//! ```
//! All multi-byte integers are network byte order (big-endian). Addresses are
//! 128-bit; an IPv4 address is carried IPv4-mapped (`::ffff:a.b.c.d`).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::time::Duration;

/// PCP protocol version (RFC 6887 = 2; version 1 was NAT-PMP-only).
pub const PCP_VERSION: u8 = 2;
/// PCP server (CPE) UDP port.
pub const PCP_SERVER_PORT: u16 = 5351;
/// MAP opcode (RFC 6887 §11) — creates an explicit inbound mapping / firewall
/// pinhole for a listening port.
pub const OPCODE_MAP: u8 = 1;
/// Response `R` bit set in the opcode byte (request=0, response=1).
const RESPONSE_BIT: u8 = 0x80;

/// IANA protocol numbers used in a MAP request.
pub mod proto {
    pub const TCP: u8 = 6;
    pub const UDP: u8 = 17;
}

/// A 12-byte mapping nonce. The client picks it once and reuses it across
/// renewals of the *same* mapping (RFC 6887 §11.1): the PCP server keys the
/// mapping on `(nonce, protocol, internal_port)`, so a fresh nonce on renewal
/// would create a second mapping instead of extending the first.
pub type Nonce = [u8; 12];

/// PCP result codes (RFC 6887 §7.4). Only the ones we act on are named; the rest
/// fall through to `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultCode {
    Success,
    UnsuppVersion,
    NotAuthorized,
    MalformedRequest,
    UnsuppOpcode,
    NetworkFailure,
    NoResources,
    UnsuppProtocol,
    CannotProvideExternal,
    AddressMismatch,
    Other(u8),
}

impl ResultCode {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Success,
            1 => Self::UnsuppVersion,
            2 => Self::NotAuthorized,
            3 => Self::MalformedRequest,
            4 => Self::UnsuppOpcode,
            7 => Self::NetworkFailure,
            8 => Self::NoResources,
            9 => Self::UnsuppProtocol,
            11 => Self::CannotProvideExternal,
            12 => Self::AddressMismatch,
            other => Self::Other(other),
        }
    }
    pub fn is_success(self) -> bool {
        matches!(self, Self::Success)
    }
}

/// Errors from encoding/decoding or the network round-trip.
#[derive(Debug)]
pub enum PcpError {
    /// Response was shorter than the 60-byte MAP response.
    ShortResponse(usize),
    /// Version byte was not 2.
    BadVersion(u8),
    /// The `R` bit was not set (not a response) or the opcode wasn't MAP.
    NotMapResponse(u8),
    /// The response nonce didn't match the request (a foreign / spoofed reply).
    NonceMismatch,
    /// The PCP server returned a non-success result code.
    Result(ResultCode),
    /// Underlying socket / I/O failure.
    Io(std::io::Error),
    /// No reply within the timeout.
    Timeout,
}

impl std::fmt::Display for PcpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShortResponse(n) => write!(f, "pcp: short response ({n} bytes)"),
            Self::BadVersion(v) => write!(f, "pcp: bad version {v}"),
            Self::NotMapResponse(b) => write!(f, "pcp: not a MAP response (opcode byte {b:#04x})"),
            Self::NonceMismatch => write!(f, "pcp: nonce mismatch"),
            Self::Result(rc) => write!(f, "pcp: server result {rc:?}"),
            Self::Io(e) => write!(f, "pcp: io: {e}"),
            Self::Timeout => write!(f, "pcp: timeout"),
        }
    }
}

impl std::error::Error for PcpError {}

/// Normalise any IP to the 16-byte PCP address encoding (IPv4 → IPv4-mapped v6).
fn ip_to_16(ip: IpAddr) -> [u8; 16] {
    match ip {
        IpAddr::V6(v6) => v6.octets(),
        IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
    }
}

/// A MAP request: open (or renew) an inbound mapping / firewall pinhole for
/// `internal_port` on this host.
#[derive(Debug, Clone)]
pub struct MapRequest {
    pub nonce: Nonce,
    /// IANA protocol number (`proto::TCP` / `proto::UDP`).
    pub protocol: u8,
    /// The port we listen on internally (the port to open inbound).
    pub internal_port: u16,
    /// Suggested external port — normally equal to `internal_port`; the server
    /// may assign a different one and reports it in the response.
    pub suggested_external_port: u16,
    /// This host's address as the PCP client (RFC 6887 §8.1 requires the source
    /// address; a mismatch yields `ADDRESS_MISMATCH`).
    pub client_addr: IpAddr,
    /// Suggested external address (usually the same global v6 as `client_addr`);
    /// all-zeros lets the server choose.
    pub suggested_external_addr: IpAddr,
    /// Requested lifetime in seconds. 0 deletes the mapping.
    pub lifetime_secs: u32,
}

impl MapRequest {
    /// Encode to the 60-byte on-wire PCP MAP request.
    pub fn encode(&self) -> [u8; 60] {
        let mut buf = [0u8; 60];
        // ── Common request header (24 bytes) ──
        buf[0] = PCP_VERSION;
        buf[1] = OPCODE_MAP; // R bit = 0 (request)
        // buf[2..4] reserved = 0
        buf[4..8].copy_from_slice(&self.lifetime_secs.to_be_bytes());
        buf[8..24].copy_from_slice(&ip_to_16(self.client_addr));
        // ── MAP opcode body (36 bytes) ──
        buf[24..36].copy_from_slice(&self.nonce);
        buf[36] = self.protocol;
        // buf[37..40] reserved = 0
        buf[40..42].copy_from_slice(&self.internal_port.to_be_bytes());
        buf[42..44].copy_from_slice(&self.suggested_external_port.to_be_bytes());
        buf[44..60].copy_from_slice(&ip_to_16(self.suggested_external_addr));
        buf
    }
}

/// A decoded MAP response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapResponse {
    pub result: ResultCode,
    /// Granted lifetime in seconds (renew before this elapses).
    pub lifetime_secs: u32,
    /// Server epoch time (RFC 6887 §8.5 — a jump signals the server lost state).
    pub epoch: u32,
    pub nonce: Nonce,
    pub protocol: u8,
    pub internal_port: u16,
    /// The external port the server actually assigned.
    pub assigned_external_port: u16,
    /// The external address the server actually assigned.
    pub assigned_external_addr: Ipv6Addr,
}

impl MapResponse {
    /// Parse a MAP response, validating version, the response bit, the opcode,
    /// and (against `expect_nonce`) the mapping nonce. Does **not** reject on a
    /// non-success result code — the caller decides how to treat each code — but
    /// the raw code is surfaced in `result`.
    pub fn decode(buf: &[u8], expect_nonce: &Nonce) -> Result<Self, PcpError> {
        if buf.len() < 60 {
            return Err(PcpError::ShortResponse(buf.len()));
        }
        if buf[0] != PCP_VERSION {
            return Err(PcpError::BadVersion(buf[0]));
        }
        // Opcode byte must be R=1 and opcode=MAP.
        if buf[1] != (RESPONSE_BIT | OPCODE_MAP) {
            return Err(PcpError::NotMapResponse(buf[1]));
        }
        let result = ResultCode::from_u8(buf[3]);
        let lifetime_secs = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let epoch = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);
        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&buf[24..36]);
        if &nonce != expect_nonce {
            return Err(PcpError::NonceMismatch);
        }
        let protocol = buf[36];
        let internal_port = u16::from_be_bytes([buf[40], buf[41]]);
        let assigned_external_port = u16::from_be_bytes([buf[42], buf[43]]);
        let mut addr = [0u8; 16];
        addr.copy_from_slice(&buf[44..60]);
        Ok(Self {
            result,
            lifetime_secs,
            epoch,
            nonce,
            protocol,
            internal_port,
            assigned_external_port,
            assigned_external_addr: Ipv6Addr::from(addr),
        })
    }

    /// The assigned external address as an `IpAddr`, un-mapping an IPv4-mapped v6
    /// back to a plain v4 (so a v4 CPE that answers PCP still yields a v4 addr).
    pub fn external_ip(&self) -> IpAddr {
        match self.assigned_external_addr.to_ipv4_mapped() {
            Some(v4) => IpAddr::V4(v4),
            None => IpAddr::V6(self.assigned_external_addr),
        }
    }
}

/// Generate a fresh 12-byte mapping nonce.
pub fn new_nonce() -> Nonce {
    use rand::RngCore;
    let mut n = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut n);
    n
}

/// Best-effort: request (or renew) a MAP mapping from the PCP server at
/// `gateway`, opening `internal_port` inbound for `protocol`.
///
/// Binds an ephemeral UDP socket, sends the 60-byte request, and waits up to
/// `timeout` for one reply, which it validates (version / response bit / opcode
/// / nonce) and returns. **Not live-validated** — see the module docs. RFC 6887
/// prescribes retransmission with exponential backoff; this single-shot version
/// leaves retries to the caller's renewal cadence.
pub async fn request_mapping(
    gateway: IpAddr,
    req: &MapRequest,
    timeout: Duration,
) -> Result<MapResponse, PcpError> {
    // Bind on the family of the gateway so the OS routes to the CPE.
    let bind: std::net::SocketAddr = match gateway {
        IpAddr::V4(_) => (Ipv4Addr::UNSPECIFIED, 0).into(),
        IpAddr::V6(_) => (Ipv6Addr::UNSPECIFIED, 0).into(),
    };
    let sock = tokio::net::UdpSocket::bind(bind).await.map_err(PcpError::Io)?;
    sock.connect(std::net::SocketAddr::new(gateway, PCP_SERVER_PORT))
        .await
        .map_err(PcpError::Io)?;
    let wire = req.encode();
    sock.send(&wire).await.map_err(PcpError::Io)?;

    let mut buf = [0u8; 1100]; // max PCP message is 1100 bytes (RFC 6887 §7)
    let n = match tokio::time::timeout(timeout, sock.recv(&mut buf)).await {
        Ok(Ok(n)) => n,
        Ok(Err(e)) => return Err(PcpError::Io(e)),
        Err(_) => return Err(PcpError::Timeout),
    };
    let resp = MapResponse::decode(&buf[..n], &req.nonce)?;
    if resp.result.is_success() {
        Ok(resp)
    } else {
        Err(PcpError::Result(resp.result))
    }
}

// ── Opt-in maintainer wiring ───────────────────────────────────────────────

/// Resolved PCP wiring handed to the event loop: the CPE gateway plus the
/// internal `(protocol, port)` pairs to pinhole.
#[derive(Debug, Clone)]
pub struct PcpBind {
    pub gateway: IpAddr,
    /// `(IANA protocol, internal port)` pairs (e.g. `(proto::UDP, 4001)`).
    pub ports: Vec<(u8, u16)>,
}

/// Is `ip` a global-unicast IPv6 (2000::/3)? Excludes loopback, ULA (fc00::/7)
/// and link-local (fe80::/10), which all fall outside 0x20..=0x3f. Stable-Rust
/// alternative to the unstable `Ipv6Addr::is_global`.
fn is_global_v6(ip: &Ipv6Addr) -> bool {
    (0x20..=0x3f).contains(&ip.octets()[0])
}

/// Best-effort discovery of this host's global IPv6 source address, by
/// connecting a UDP socket to a public v6 and reading the chosen local address.
/// Returns `None` on a v6-less host or one with only ULA/link-local v6.
pub fn local_global_v6() -> Option<Ipv6Addr> {
    use std::net::UdpSocket;
    let sock = UdpSocket::bind("[::]:0").ok()?;
    // 2606:4700:4700::1111 = Cloudflare DNS; the connect only picks a source
    // address, it sends nothing.
    sock.connect("[2606:4700:4700::1111]:53").ok()?;
    match sock.local_addr().ok()?.ip() {
        IpAddr::V6(v6) if is_global_v6(&v6) => Some(v6),
        _ => None,
    }
}

/// Derive the `(protocol, port)` pairs to pinhole from the node's listen addrs:
/// each concrete TCP port and each QUIC (UDP) port, deduped. Wildcard port 0 is
/// skipped (nothing stable to advertise). Pure + unit-tested.
pub fn ports_from_listen_addrs(addrs: &[libp2p::Multiaddr]) -> Vec<(u8, u16)> {
    use libp2p::multiaddr::Protocol;
    let mut out: Vec<(u8, u16)> = Vec::new();
    let mut push = |proto: u8, port: u16| {
        if port != 0 && !out.contains(&(proto, port)) {
            out.push((proto, port));
        }
    };
    for a in addrs {
        let (mut tcp, mut udp, mut is_quic) = (None, None, false);
        for p in a.iter() {
            match p {
                Protocol::Tcp(port) => tcp = Some(port),
                Protocol::Udp(port) => udp = Some(port),
                Protocol::QuicV1 | Protocol::Quic => is_quic = true,
                _ => {}
            }
        }
        if let Some(port) = tcp {
            push(proto::TCP, port);
        }
        if is_quic {
            if let Some(port) = udp {
                push(proto::UDP, port);
            }
        }
    }
    out
}

/// Build the external listen multiaddr a confirmed MAP response opens:
/// `/ip6/<ext>/udp/<port>/quic-v1` for UDP, `/ip6/<ext>/tcp/<port>` for TCP.
/// Returns `None` for a non-v6 external (a v4 CPE answering PCP — not our goal)
/// or an unexpected protocol.
fn external_multiaddr(resp: &MapResponse) -> Option<libp2p::Multiaddr> {
    use libp2p::multiaddr::Protocol;
    let ip = match resp.external_ip() {
        IpAddr::V6(v6) => v6,
        IpAddr::V4(_) => return None,
    };
    let port = resp.assigned_external_port;
    let mut ma = libp2p::Multiaddr::empty();
    ma.push(Protocol::Ip6(ip));
    match resp.protocol {
        proto::UDP => {
            ma.push(Protocol::Udp(port));
            ma.push(Protocol::QuicV1);
        }
        proto::TCP => ma.push(Protocol::Tcp(port)),
        _ => return None,
    }
    Some(ma)
}

/// Requested pinhole lifetime (seconds). The CPE may grant less; we renew at
/// half the *granted* lifetime.
const REQUESTED_LIFETIME_SECS: u32 = 7200;

/// Opt-in PCP maintainer loop: periodically (re)assert an inbound v6 firewall
/// pinhole for each `(protocol, port)` at `gateway`, and report each confirmed
/// external multiaddr over `candidate_tx` so the event loop can add it as an
/// external address (→ AutoNAT probe → promotion, mirroring the R-DHT-4 UPnP
/// path). Runs until `candidate_tx` closes (event loop gone).
///
/// **Not live-validated** — no PCP-capable CPE was reachable during development.
pub async fn run_maintainer(
    gateway: IpAddr,
    ports: Vec<(u8, u16)>,
    candidate_tx: tokio::sync::mpsc::UnboundedSender<libp2p::Multiaddr>,
) {
    const IO_TIMEOUT: Duration = Duration::from_secs(3);
    // One nonce per mapping, reused across renewals (RFC 6887 §11.1).
    let nonces: Vec<Nonce> = ports.iter().map(|_| new_nonce()).collect();
    loop {
        let client_v6 = match local_global_v6() {
            Some(v6) => v6,
            None => {
                // No global v6 right now (v6-less network / mid-roam) — retry.
                tokio::time::sleep(Duration::from_secs(60)).await;
                continue;
            }
        };
        let mut min_lifetime = REQUESTED_LIFETIME_SECS;
        for (i, (protocol, port)) in ports.iter().enumerate() {
            let req = MapRequest {
                nonce: nonces[i],
                protocol: *protocol,
                internal_port: *port,
                suggested_external_port: *port,
                client_addr: IpAddr::V6(client_v6),
                suggested_external_addr: IpAddr::V6(client_v6),
                lifetime_secs: REQUESTED_LIFETIME_SECS,
            };
            match request_mapping(gateway, &req, IO_TIMEOUT).await {
                Ok(resp) => {
                    min_lifetime = min_lifetime.min(resp.lifetime_secs.max(60));
                    if let Some(ma) = external_multiaddr(&resp) {
                        tracing::info!(
                            %gateway, port = *port, external = %ma,
                            lifetime = resp.lifetime_secs,
                            "pcp: inbound v6 pinhole confirmed"
                        );
                        if candidate_tx.send(ma).is_err() {
                            return; // event loop gone
                        }
                    }
                }
                Err(e) => tracing::warn!(
                    %gateway, port = *port, error = %e,
                    "pcp: pinhole request failed"
                ),
            }
        }
        // Renew at half the granted lifetime (RFC 6887 §11.2.1 guidance).
        let sleep = (min_lifetime / 2).max(30) as u64;
        tokio::time::sleep(Duration::from_secs(sleep)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_nonce() -> Nonce {
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }

    #[test]
    fn encode_map_request_wire_layout() {
        let req = MapRequest {
            nonce: sample_nonce(),
            protocol: proto::TCP,
            internal_port: 4001,
            suggested_external_port: 4001,
            client_addr: IpAddr::V6("2406:7400:56:7e7::e4c6".parse().unwrap()),
            suggested_external_addr: IpAddr::V6("2406:7400:56:7e7::e4c6".parse().unwrap()),
            lifetime_secs: 7200,
        };
        let w = req.encode();
        assert_eq!(w.len(), 60);
        // Common header.
        assert_eq!(w[0], PCP_VERSION);
        assert_eq!(w[1], OPCODE_MAP); // request → R bit clear
        assert_eq!(&w[2..4], &[0, 0]); // reserved
        assert_eq!(&w[4..8], &7200u32.to_be_bytes()); // lifetime
        let client = "2406:7400:56:7e7::e4c6".parse::<Ipv6Addr>().unwrap();
        assert_eq!(&w[8..24], &client.octets());
        // MAP body.
        assert_eq!(&w[24..36], &sample_nonce());
        assert_eq!(w[36], proto::TCP);
        assert_eq!(&w[37..40], &[0, 0, 0]); // reserved
        assert_eq!(&w[40..42], &4001u16.to_be_bytes()); // internal port
        assert_eq!(&w[42..44], &4001u16.to_be_bytes()); // suggested external port
        assert_eq!(&w[44..60], &client.octets());
    }

    #[test]
    fn encode_v4_maps_to_v6() {
        let req = MapRequest {
            nonce: sample_nonce(),
            protocol: proto::UDP,
            internal_port: 4001,
            suggested_external_port: 0,
            client_addr: IpAddr::V4("192.0.2.9".parse().unwrap()),
            suggested_external_addr: IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            lifetime_secs: 0,
        };
        let w = req.encode();
        // IPv4 192.0.2.9 mapped → ::ffff:192.0.2.9 → last 6 bytes ff ff C0 00 02 09.
        assert_eq!(&w[8..24], &"192.0.2.9".parse::<Ipv4Addr>().unwrap().to_ipv6_mapped().octets());
        assert_eq!(&w[18..24], &[0xff, 0xff, 192, 0, 2, 9]);
        assert_eq!(w[36], proto::UDP);
        assert_eq!(&w[4..8], &0u32.to_be_bytes()); // lifetime 0 = delete
    }

    /// Build a well-formed 60-byte MAP response for decode tests.
    fn build_response(result: u8, nonce: &Nonce, ext_port: u16, ext_ip: Ipv6Addr, lifetime: u32) -> Vec<u8> {
        let mut b = vec![0u8; 60];
        b[0] = PCP_VERSION;
        b[1] = RESPONSE_BIT | OPCODE_MAP;
        b[3] = result;
        b[4..8].copy_from_slice(&lifetime.to_be_bytes());
        b[8..12].copy_from_slice(&42u32.to_be_bytes()); // epoch
        b[24..36].copy_from_slice(nonce);
        b[36] = proto::TCP;
        b[40..42].copy_from_slice(&4001u16.to_be_bytes()); // internal port
        b[42..44].copy_from_slice(&ext_port.to_be_bytes());
        b[44..60].copy_from_slice(&ext_ip.octets());
        b
    }

    #[test]
    fn decode_success_response() {
        let nonce = sample_nonce();
        let ext: Ipv6Addr = "2406:7400:56:7e7::e4c6".parse().unwrap();
        let b = build_response(0, &nonce, 4001, ext, 7200);
        let r = MapResponse::decode(&b, &nonce).unwrap();
        assert_eq!(r.result, ResultCode::Success);
        assert!(r.result.is_success());
        assert_eq!(r.lifetime_secs, 7200);
        assert_eq!(r.epoch, 42);
        assert_eq!(r.assigned_external_port, 4001);
        assert_eq!(r.assigned_external_addr, ext);
        assert_eq!(r.external_ip(), IpAddr::V6(ext));
    }

    #[test]
    fn decode_v4_mapped_external_unmaps() {
        let nonce = sample_nonce();
        let mapped = "192.0.2.9".parse::<Ipv4Addr>().unwrap().to_ipv6_mapped();
        let b = build_response(0, &nonce, 4001, mapped, 100);
        let r = MapResponse::decode(&b, &nonce).unwrap();
        assert_eq!(r.external_ip(), IpAddr::V4("192.0.2.9".parse().unwrap()));
    }

    #[test]
    fn decode_rejects_nonce_mismatch() {
        let nonce = sample_nonce();
        let other = [9u8; 12];
        let b = build_response(0, &other, 4001, Ipv6Addr::LOCALHOST, 100);
        assert!(matches!(MapResponse::decode(&b, &nonce), Err(PcpError::NonceMismatch)));
    }

    #[test]
    fn decode_rejects_request_bit() {
        // A packet with R bit clear (a request, not a response) must be rejected.
        let nonce = sample_nonce();
        let mut b = build_response(0, &nonce, 4001, Ipv6Addr::LOCALHOST, 100);
        b[1] = OPCODE_MAP; // clear the response bit
        assert!(matches!(MapResponse::decode(&b, &nonce), Err(PcpError::NotMapResponse(_))));
    }

    #[test]
    fn decode_rejects_short_and_bad_version() {
        let nonce = sample_nonce();
        assert!(matches!(MapResponse::decode(&[0u8; 10], &nonce), Err(PcpError::ShortResponse(10))));
        let mut b = build_response(0, &nonce, 4001, Ipv6Addr::LOCALHOST, 100);
        b[0] = 1; // NAT-PMP version, not PCP
        assert!(matches!(MapResponse::decode(&b, &nonce), Err(PcpError::BadVersion(1))));
    }

    #[test]
    fn decode_surfaces_error_result_code() {
        let nonce = sample_nonce();
        // 2 = NOT_AUTHORIZED (e.g. PCP disabled / firewall control not permitted).
        let b = build_response(2, &nonce, 0, Ipv6Addr::UNSPECIFIED, 0);
        let r = MapResponse::decode(&b, &nonce).unwrap();
        assert_eq!(r.result, ResultCode::NotAuthorized);
        assert!(!r.result.is_success());
    }

    #[test]
    fn nonce_is_random_and_sized() {
        let a = new_nonce();
        let b = new_nonce();
        assert_eq!(a.len(), 12);
        // Astronomically unlikely to collide; guards against a stubbed RNG.
        assert_ne!(a, b);
    }

    #[test]
    fn global_v6_classification() {
        assert!(is_global_v6(&"2406:7400:56:7e7::e4c6".parse().unwrap())); // 2000::/3
        assert!(is_global_v6(&"2a03:4000:41:ed1::1".parse().unwrap()));
        assert!(!is_global_v6(&"fd00::1".parse().unwrap())); // ULA
        assert!(!is_global_v6(&"fe80::1".parse().unwrap())); // link-local
        assert!(!is_global_v6(&Ipv6Addr::LOCALHOST));
        assert!(!is_global_v6(&Ipv6Addr::UNSPECIFIED));
    }

    #[test]
    fn ports_from_listen_addrs_extracts_and_dedups() {
        let addrs: Vec<libp2p::Multiaddr> = [
            "/ip4/0.0.0.0/tcp/4001",
            "/ip6/::/tcp/4001",              // dup TCP 4001 — deduped
            "/ip4/0.0.0.0/udp/4001/quic-v1", // QUIC UDP 4001
            "/ip6/::/udp/4001/quic-v1",      // dup UDP 4001 — deduped
            "/ip4/0.0.0.0/udp/0/quic-v1",    // wildcard port 0 — skipped
        ]
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();
        let ports = ports_from_listen_addrs(&addrs);
        assert_eq!(ports, vec![(proto::TCP, 4001), (proto::UDP, 4001)]);
    }

    #[test]
    fn external_multiaddr_builds_quic_and_tcp() {
        let nonce = sample_nonce();
        let ext: Ipv6Addr = "2406:7400:56:7e7::e4c6".parse().unwrap();

        let mut udp = build_response(0, &nonce, 4001, ext, 7200);
        udp[36] = proto::UDP;
        let r = MapResponse::decode(&udp, &nonce).unwrap();
        assert_eq!(
            external_multiaddr(&r).unwrap().to_string(),
            "/ip6/2406:7400:56:7e7::e4c6/udp/4001/quic-v1"
        );

        let tcp = build_response(0, &nonce, 4001, ext, 7200); // build_response sets TCP
        let r = MapResponse::decode(&tcp, &nonce).unwrap();
        assert_eq!(
            external_multiaddr(&r).unwrap().to_string(),
            "/ip6/2406:7400:56:7e7::e4c6/tcp/4001"
        );
    }

    #[test]
    fn external_multiaddr_rejects_v4() {
        let nonce = sample_nonce();
        let mapped = "192.0.2.9".parse::<Ipv4Addr>().unwrap().to_ipv6_mapped();
        let b = build_response(0, &nonce, 4001, mapped, 100);
        let r = MapResponse::decode(&b, &nonce).unwrap();
        // A v4 CPE answering PCP is not our v6-pinhole goal.
        assert!(external_multiaddr(&r).is_none());
    }
}
