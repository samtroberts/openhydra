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

mod dns;
mod query;

use self::dns::{build_query, build_query_response, build_service_discovery_response};
use self::query::MdnsPacket;
use crate::behaviour::{socket::AsyncSocket, timer::Builder};
use crate::Config;
use futures::channel::mpsc;
use futures::task::AtomicWaker;
use futures::{SinkExt, StreamExt};
use libp2p_core::{multiaddr::Protocol, Multiaddr};
use libp2p_identity::PeerId;
use libp2p_swarm::ListenAddresses;
use socket2::{Domain, Socket, Type};
use std::future::Future;
use std::sync::{Arc, RwLock};
use std::{
    collections::VecDeque,
    io,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket},
    pin::Pin,
    task::{Context, Poll},
    time::{Duration, Instant},
};

/// Initial interval for starting probe
const INITIAL_TIMEOUT_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Debug, Clone)]
enum ProbeState {
    Probing(Duration),
    Finished(Duration),
}

impl Default for ProbeState {
    fn default() -> Self {
        ProbeState::Probing(INITIAL_TIMEOUT_INTERVAL)
    }
}

impl ProbeState {
    fn interval(&self) -> &Duration {
        match self {
            ProbeState::Probing(query_interval) => query_interval,
            ProbeState::Finished(query_interval) => query_interval,
        }
    }
}

/// The largest mDNS packet RFC 6762 allows (section 17).
const RECV_BUFFER_SIZE: usize = 9000;

/// An mDNS instance for a networking interface. To discover all peers when having multiple
/// interfaces an [`InterfaceState`] is required for each interface.
#[derive(Debug)]
pub(crate) struct InterfaceState<U, T> {
    /// Address this instance is bound to.
    addr: IpAddr,
    /// Receive socket.
    recv_socket: U,
    /// Send socket.
    send_socket: U,

    listen_addresses: Arc<RwLock<ListenAddresses>>,

    query_response_sender: mpsc::Sender<(PeerId, Multiaddr, Instant)>,

    /// Buffer used for receiving data from the main socket.
    /// RFC6762 discourages packets larger than the interface MTU, but allows sizes of up to 9000
    /// bytes. (OpenHydra patch: upstream reads 4096 bytes, but its own responses go up to 9000, so
    /// a peer with many addresses was cut short and never discovered. Reading the full 9000 lets
    /// us discover peers still running that version.) For more information see
    /// [rfc6762](https://tools.ietf.org/html/rfc6762#page-46).
    recv_buffer: [u8; RECV_BUFFER_SIZE],
    /// Buffers pending to send on the main socket.
    send_buffer: VecDeque<Vec<u8>>,
    /// Discovery interval.
    query_interval: Duration,
    /// Discovery timer.
    timeout: T,
    /// Multicast address.
    multicast_addr: IpAddr,
    /// Discovered addresses.
    discovered: VecDeque<(PeerId, Multiaddr, Instant)>,
    /// TTL
    ttl: Duration,
    probe_state: ProbeState,
    local_peer_id: PeerId,
    /// OpenHydra patch: announces this interface's addresses when they change. The behaviour
    /// wakes `addrs_changed` whenever the listen addresses change; `announce_again` repeats an
    /// announcement once, [`ANNOUNCE_REPEAT`] later.
    announcer: Announcer,
    /// OpenHydra patch: when announcements go out ([`AnnounceSchedule`]).
    schedule: AnnounceSchedule,
    addrs_changed: Arc<AtomicWaker>,
    announce_again: Option<T>,
}

impl<U, T> InterfaceState<U, T>
where
    U: AsyncSocket,
    T: Builder + futures::Stream,
{
    /// Builds a new [`InterfaceState`].
    pub(crate) fn new(
        addr: IpAddr,
        config: Config,
        local_peer_id: PeerId,
        listen_addresses: Arc<RwLock<ListenAddresses>>,
        query_response_sender: mpsc::Sender<(PeerId, Multiaddr, Instant)>,
        addrs_changed: Arc<AtomicWaker>,
    ) -> io::Result<Self> {
        tracing::info!(address=%addr, "creating instance on iface address");
        let recv_socket = match addr {
            IpAddr::V4(addr) => {
                let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(socket2::Protocol::UDP))?;
                socket.set_reuse_address(true)?;
                #[cfg(unix)]
                socket.set_reuse_port(true)?;
                socket.bind(&SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 5353).into())?;
                socket.set_multicast_loop_v4(true)?;
                socket.set_multicast_ttl_v4(255)?;
                socket.join_multicast_v4(&crate::IPV4_MDNS_MULTICAST_ADDRESS, &addr)?;
                U::from_std(UdpSocket::from(socket))?
            }
            IpAddr::V6(_) => {
                let socket = Socket::new(Domain::IPV6, Type::DGRAM, Some(socket2::Protocol::UDP))?;
                socket.set_reuse_address(true)?;
                #[cfg(unix)]
                socket.set_reuse_port(true)?;
                socket.bind(&SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 5353).into())?;
                socket.set_multicast_loop_v6(true)?;
                // TODO: find interface matching addr.
                socket.join_multicast_v6(&crate::IPV6_MDNS_MULTICAST_ADDRESS, 0)?;
                U::from_std(UdpSocket::from(socket))?
            }
        };
        let bind_addr = match addr {
            IpAddr::V4(_) => SocketAddr::new(addr, 0),
            IpAddr::V6(_addr) => {
                // TODO: if-watch should return the scope_id of an address
                // as a workaround we bind to unspecified, which means that
                // this probably won't work when using multiple interfaces.
                // SocketAddr::V6(SocketAddrV6::new(addr, 0, 0, scope_id))
                SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0)
            }
        };
        let send_socket = U::from_std(UdpSocket::bind(bind_addr)?)?;

        // randomize timer to prevent all converging and firing at the same time.
        let query_interval = {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let jitter = rng.gen_range(0..100);
            config.query_interval + Duration::from_millis(jitter)
        };
        let multicast_addr = match addr {
            IpAddr::V4(_) => IpAddr::V4(crate::IPV4_MDNS_MULTICAST_ADDRESS),
            IpAddr::V6(_) => IpAddr::V6(crate::IPV6_MDNS_MULTICAST_ADDRESS),
        };
        Ok(Self {
            addr,
            recv_socket,
            send_socket,
            listen_addresses,
            query_response_sender,
            recv_buffer: [0; RECV_BUFFER_SIZE],
            send_buffer: Default::default(),
            discovered: Default::default(),
            query_interval,
            timeout: T::interval_at(Instant::now(), INITIAL_TIMEOUT_INTERVAL),
            multicast_addr,
            ttl: config.ttl,
            probe_state: Default::default(),
            local_peer_id,
            announcer: Announcer::default(),
            schedule: AnnounceSchedule::default(),
            addrs_changed,
            announce_again: None,
        })
    }

    pub(crate) fn reset_timer(&mut self) {
        tracing::trace!(address=%self.addr, probe_state=?self.probe_state, "reset timer");
        let interval = *self.probe_state.interval();
        self.timeout = T::interval(interval);
    }

    fn mdns_socket(&self) -> SocketAddr {
        SocketAddr::new(self.multicast_addr, 5353)
    }

    /// Queues an unprompted response carrying `addrs`.
    fn announce(&mut self, addrs: &[Multiaddr]) {
        tracing::trace!(address=%self.addr, count=addrs.len(), "announcing addresses");
        self.send_buffer.extend(build_query_response(
            0,
            self.local_peer_id,
            addrs.iter().collect(),
            self.ttl,
        ));
    }

    /// The listening addresses this interface's responses carry: only those that belong to this
    /// interface. This prevents advertising loopback or other interface addresses to peers that
    /// can't reach them. (OpenHydra patch: backported from libp2p-mdns 0.49.0, PR 6500. It also
    /// keeps relay-circuit addresses out, which are what made responses too big.) Sorted, so
    /// [`Announcer`] can tell when they change.
    fn advertised_addresses(&self) -> Vec<Multiaddr> {
        let read = self
            .listen_addresses
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let mut addrs: Vec<Multiaddr> = read
            .iter()
            .filter(|multiaddr| addr_matches_interface(multiaddr, self.addr))
            .cloned()
            .collect();
        addrs.sort_by(|a, b| a.as_ref().cmp(b.as_ref()));
        addrs
    }
}

/// How long after an announcement it is repeated. RFC 6762 section 8.3 asks for at least two
/// announcements, one second apart, so one lost packet doesn't hide the addresses.
const ANNOUNCE_REPEAT: Duration = Duration::from_secs(1);

/// OpenHydra patch: when an interface sends its announcements. Each change is announced twice, a
/// second apart (RFC 6762 section 8.3), and never more than once a second (section 6.2): a change
/// within a second of the last announcement waits for the one-second mark and goes out then, with
/// the newest set, so a burst of changes (listeners coming up one by one at start-up, an address
/// flapping) costs a few packets, not one per change.
#[derive(Debug, Default)]
struct AnnounceSchedule {
    last_sent: Option<Instant>,
    /// Announcements still to send for the latest change.
    pending: u8,
    /// When the next one is due, if any.
    next: Option<Instant>,
}

impl AnnounceSchedule {
    /// The addresses changed at `now`: whether to announce right away (otherwise [`Self::next`]
    /// says when).
    fn changed(&mut self, now: Instant) -> bool {
        match self.last_sent {
            Some(last) if now < last + ANNOUNCE_REPEAT => {
                self.pending = 2;
                self.next = Some(self.next.map_or(last + ANNOUNCE_REPEAT, |n| n.min(last + ANNOUNCE_REPEAT)));
                false
            }
            _ => {
                self.sent(now, 1);
                true
            }
        }
    }

    /// The scheduled time came: whether to announce now.
    fn due(&mut self, now: Instant) -> bool {
        if self.pending == 0 {
            self.next = None;
            return false;
        }
        let left = self.pending - 1;
        self.sent(now, left);
        true
    }

    fn sent(&mut self, now: Instant, left: u8) {
        self.last_sent = Some(now);
        self.pending = left;
        self.next = (left > 0).then(|| now + ANNOUNCE_REPEAT);
    }

    /// When the next announcement is due, if any.
    fn next(&self) -> Option<Instant> {
        self.next
    }
}

/// OpenHydra patch: tells an interface when its advertised addresses have changed to a set worth
/// announcing (a non-empty one it hasn't announced yet).
#[derive(Debug, Default)]
struct Announcer {
    announced: Vec<Multiaddr>,
}

impl Announcer {
    /// The addresses to announce now, if `current` (sorted) differs from what was last seen.
    fn check(&mut self, current: Vec<Multiaddr>) -> Option<Vec<Multiaddr>> {
        if current == self.announced {
            return None;
        }
        self.announced = current;
        (!self.announced.is_empty()).then(|| self.announced.clone())
    }

    /// The addresses last seen, for repeating an announcement.
    fn announced(&self) -> &[Multiaddr] {
        &self.announced
    }
}

impl<U, T> Future for InterfaceState<U, T>
where
    U: AsyncSocket,
    T: Builder + futures::Stream,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();

        // OpenHydra patch: peers learn our addresses only from our answers to their queries, and
        // once an interface has heard any answer they query only every few minutes. So when this
        // interface's addresses change (a listener often comes up just after this instance
        // started), announce them unprompted, now and once more a second later, as RFC 6762
        // section 8.3 describes. The behaviour wakes us on every change; registering before
        // reading the addresses means no change is missed. Receivers treat any packet without a
        // question as a response.
        this.addrs_changed.register(cx.waker());
        if let Some(addrs) = this.announcer.check(this.advertised_addresses()) {
            let now = Instant::now();
            if this.schedule.changed(now) {
                this.announce(&addrs);
            }
            this.announce_again = this.schedule.next().map(T::at);
        }

        loop {
            // 1st priority: Low latency: Create packet ASAP after timeout.
            if this.timeout.poll_next_unpin(cx).is_ready() {
                tracing::trace!(address=%this.addr, "sending query on iface");
                this.send_buffer.push_back(build_query());
                tracing::trace!(address=%this.addr, probe_state=?this.probe_state, "tick");

                // Stop to probe when the initial interval reach the query interval
                if let ProbeState::Probing(interval) = this.probe_state {
                    let interval = interval * 2;
                    this.probe_state = if interval >= this.query_interval {
                        ProbeState::Finished(this.query_interval)
                    } else {
                        ProbeState::Probing(interval)
                    };
                }

                this.reset_timer();
            }

            // OpenHydra patch: a repeated or held-back announcement (see above the loop), always of
            // the newest set.
            if let Some(timer) = this.announce_again.as_mut() {
                if timer.poll_next_unpin(cx).is_ready() {
                    let send = this.schedule.due(Instant::now());
                    this.announce_again = this.schedule.next().map(T::at);
                    let addrs = this.announcer.announced().to_vec();
                    if send && !addrs.is_empty() {
                        this.announce(&addrs);
                    }
                    continue;
                }
            }

            // 2nd priority: Keep local buffers small: Send packets to remote.
            if let Some(packet) = this.send_buffer.pop_front() {
                match this.send_socket.poll_write(cx, &packet, this.mdns_socket()) {
                    Poll::Ready(Ok(_)) => {
                        tracing::trace!(address=%this.addr, "sent packet on iface address");
                        continue;
                    }
                    Poll::Ready(Err(err)) => {
                        tracing::error!(address=%this.addr, "error sending packet on iface address {}", err);
                        continue;
                    }
                    Poll::Pending => {
                        this.send_buffer.push_front(packet);
                    }
                }
            }

            // 3rd priority: Keep local buffers small: Return discovered addresses.
            if this.query_response_sender.poll_ready_unpin(cx).is_ready() {
                if let Some(discovered) = this.discovered.pop_front() {
                    match this.query_response_sender.try_send(discovered) {
                        Ok(()) => {}
                        Err(e) if e.is_disconnected() => {
                            return Poll::Ready(());
                        }
                        Err(e) => {
                            this.discovered.push_front(e.into_inner());
                        }
                    }

                    continue;
                }
            }

            // 4th priority: Remote work: Answer incoming requests.
            match this
                .recv_socket
                .poll_read(cx, &mut this.recv_buffer)
                .map_ok(|(len, from)| MdnsPacket::new_from_bytes(&this.recv_buffer[..len], from))
            {
                Poll::Ready(Ok(Ok(Some(MdnsPacket::Query(query))))) => {
                    tracing::trace!(
                        address=%this.addr,
                        remote_address=%query.remote_addr(),
                        "received query from remote address on address"
                    );

                    // Only send addresses that belong to this interface (see
                    // `advertised_addresses`).
                    let relevant_addrs = this.advertised_addresses();
                    this.send_buffer.extend(build_query_response(
                        query.query_id(),
                        this.local_peer_id,
                        relevant_addrs.iter().collect(),
                        this.ttl,
                    ));
                    continue;
                }
                Poll::Ready(Ok(Ok(Some(MdnsPacket::Response(response))))) => {
                    tracing::trace!(
                        address=%this.addr,
                        remote_address=%response.remote_addr(),
                        "received response from remote address on address"
                    );

                    this.discovered
                        .extend(response.extract_discovered(Instant::now(), this.local_peer_id));

                    // Stop probing when we have a valid response
                    if !this.discovered.is_empty() {
                        this.probe_state = ProbeState::Finished(this.query_interval);
                        this.reset_timer();
                    }
                    continue;
                }
                Poll::Ready(Ok(Ok(Some(MdnsPacket::ServiceDiscovery(disc))))) => {
                    tracing::trace!(
                        address=%this.addr,
                        remote_address=%disc.remote_addr(),
                        "received service discovery from remote address on address"
                    );

                    this.send_buffer
                        .push_back(build_service_discovery_response(disc.query_id(), this.ttl));
                    continue;
                }
                Poll::Ready(Err(err)) if err.kind() == std::io::ErrorKind::WouldBlock => {
                    // No more bytes available on the socket to read
                    continue;
                }
                Poll::Ready(Err(err)) => {
                    tracing::error!("failed reading datagram: {}", err);
                    return Poll::Ready(());
                }
                Poll::Ready(Ok(Err(err))) => {
                    tracing::debug!("Parsing mdns packet failed: {:?}", err);
                    continue;
                }
                Poll::Ready(Ok(Ok(None))) => continue,
                Poll::Pending => {}
            }

            return Poll::Pending;
        }
    }
}

/// Returns `true` if the first protocol component of `addr` is an IP address equal to
/// `iface_ip`.
///
/// Used when answering mDNS queries to advertise only the listening addresses that are
/// reachable on the interface the query arrived on, instead of every listening address.
fn addr_matches_interface(addr: &Multiaddr, iface_ip: IpAddr) -> bool {
    match addr.iter().next() {
        Some(Protocol::Ip4(v4)) => IpAddr::V4(v4) == iface_ip,
        Some(Protocol::Ip6(v6)) => IpAddr::V6(v6) == iface_ip,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // OpenHydra patch: what an interface's responses carry.
    #[test]
    fn only_addresses_on_the_interface_are_advertised() {
        let iface: IpAddr = "192.168.0.6".parse().unwrap();
        let on: [&str; 2] = ["/ip4/192.168.0.6/tcp/4001", "/ip4/192.168.0.6/udp/4001/quic-v1"];
        let off: [&str; 7] = [
            "/ip4/127.0.0.1/tcp/4001",
            "/ip4/172.17.0.1/tcp/4001",
            "/ip4/192.168.0.4/tcp/4001",
            "/ip6/fe80::1/udp/4001/quic-v1",
            "/ip4/45.79.190.172/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit",
            "/dns4/bootstrap-us.openhydra.co/tcp/4001",
            "/ip4/192.168.0.60/tcp/4001",
        ];
        for a in on {
            assert!(addr_matches_interface(&a.parse().unwrap(), iface), "{a}");
        }
        for a in off {
            assert!(!addr_matches_interface(&a.parse().unwrap(), iface), "{a}");
        }
        let v6: IpAddr = "fe80::1".parse().unwrap();
        assert!(addr_matches_interface(&"/ip6/fe80::1/udp/4001/quic-v1".parse().unwrap(), v6));
    }

    // OpenHydra patch: which address changes are announced.
    #[test]
    fn changes_are_announced_twice_and_never_more_than_once_a_second() {
        let t0 = Instant::now();
        let ms = |n: u64| t0 + Duration::from_millis(n);
        let mut s = AnnounceSchedule::default();
        let mut sent = Vec::new();
        // Three changes within 200 ms: the first goes out at once, the others are held back.
        for at in [0, 100, 200] {
            if s.changed(ms(at)) {
                sent.push(at);
            }
        }
        // Then the timer, run as the interface would.
        while let Some(next) = s.next() {
            let at = next.duration_since(t0).as_millis() as u64;
            if s.due(next) {
                sent.push(at);
            }
        }
        assert_eq!(sent, vec![0, 1000, 2000], "one now, the newest set at 1 s and again at 2 s");
        // A lone change long after: at once, and once more a second later.
        let mut sent = Vec::new();
        if s.changed(ms(10_000)) {
            sent.push(10_000);
        }
        while let Some(next) = s.next() {
            if s.due(next) {
                sent.push(next.duration_since(t0).as_millis() as u64);
            }
        }
        assert_eq!(sent, vec![10_000, 11_000]);
    }

    #[test]
    fn a_changed_set_of_addresses_is_announced_once_per_change() {
        let a: Multiaddr = "/ip4/192.168.0.6/tcp/4001".parse().unwrap();
        let b: Multiaddr = "/ip4/192.168.0.6/udp/4001/quic-v1".parse().unwrap();
        let mut ann = Announcer::default();
        // No addresses yet (the listener isn't up): nothing to announce.
        assert_eq!(ann.check(vec![]), None);
        // The listener comes up: announced, then quiet however often it is checked.
        assert_eq!(ann.check(vec![a.clone()]), Some(vec![a.clone()]));
        assert_eq!(ann.check(vec![a.clone()]), None);
        assert_eq!(ann.announced(), &[a.clone()][..]);
        // Another address appears: the new set.
        let both = vec![a.clone(), b.clone()];
        assert_eq!(ann.check(both.clone()), Some(both.clone()));
        assert_eq!(ann.check(both), None);
        // One goes away: the smaller set (peers expire the other by its TTL).
        assert_eq!(ann.check(vec![b.clone()]), Some(vec![b.clone()]));
        // All gone: nothing to announce, and nothing to repeat.
        assert_eq!(ann.check(vec![]), None);
        assert!(ann.announced().is_empty());
    }
}
