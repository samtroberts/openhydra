// Copyright 2021 Protocol Labs.
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


// OpenHydra patch (21 Nov 2026): direct addresses first, relay addresses only as a fallback, and
// QUIC before TCP — the order go-libp2p's "smart dialing" uses (relays 500 ms, TCP 250 ms after
// QUIC). Upstream dials every address at once. A circuit request then reached the relay for nearly
// every dial to a peer that also had a relay address, even when the direct address connected within
// milliseconds: in a 1,500-node simulation each bootstrap got ~275,000 circuit requests in six
// minutes and denied 97% of them. Held-back attempts are released early once everything before them
// has failed, and are dropped unstarted once one attempt connects (the relay client sends its
// request, and TCP calls `connect`, only when their future is first polled).

use crate::{transport::TransportError, Multiaddr};
use futures::{
    future::{BoxFuture, Future, FutureExt},
    stream::{FuturesUnordered, StreamExt},
};
use futures_timer::Delay;
use libp2p_core::{multiaddr::Protocol, muxing::StreamMuxerBox};
use libp2p_identity::PeerId;
use std::{
    collections::VecDeque,
    num::NonZeroU8,
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};

type Dial = BoxFuture<
    'static,
    (
        Multiaddr,
        Result<(PeerId, StreamMuxerBox), TransportError<std::io::Error>>,
    ),
>;

/// How long TCP attempts wait for QUIC ones to the same peer (go-libp2p's `PublicTCPDelay`).
pub(crate) const TCP_DELAY: Duration = Duration::from_millis(250);
/// How long relay attempts wait for direct ones to the same peer (go-libp2p's `RelayDelay`).
pub(crate) const RELAY_DELAY: Duration = Duration::from_millis(500);

/// The order attempts start in: QUIC, then other direct transports (TCP), then relay circuits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Group {
    Quic = 0,
    OtherDirect = 1,
    Relay = 2,
}

pub(crate) fn group_of(addr: &Multiaddr) -> Group {
    if addr.iter().any(|p| matches!(p, Protocol::P2pCircuit)) {
        Group::Relay
    } else if addr.iter().any(|p| matches!(p, Protocol::QuicV1 | Protocol::Quic)) {
        Group::Quic
    } else {
        Group::OtherDirect
    }
}

pub(crate) struct ConcurrentDial {
    /// Attempts in flight, each with its group.
    dials: FuturesUnordered<BoxFuture<'static, (Group, <Dial as Future>::Output)>>,
    /// Attempts not started yet, per group, in the order given.
    queued: [VecDeque<Dial>; 3],
    /// Attempts in flight per group.
    in_flight: [usize; 3],
    /// Whether a group may start (it is released at its delay, or as soon as every group before it
    /// is finished).
    released: [bool; 3],
    /// Each held-back group's timer.
    timers: [Option<Delay>; 3],
    concurrency: usize,
    errors: Vec<(Multiaddr, TransportError<std::io::Error>)>,
}

impl Unpin for ConcurrentDial {}

impl ConcurrentDial {
    pub(crate) fn new(pending_dials: Vec<(Multiaddr, Dial)>, concurrency_factor: NonZeroU8) -> Self {
        Self::with_delays(pending_dials, concurrency_factor, TCP_DELAY, RELAY_DELAY)
    }

    pub(crate) fn with_delays(
        pending_dials: Vec<(Multiaddr, Dial)>,
        concurrency_factor: NonZeroU8,
        tcp_delay: Duration,
        relay_delay: Duration,
    ) -> Self {
        let mut queued: [VecDeque<Dial>; 3] = Default::default();
        for (addr, dial) in pending_dials {
            queued[group_of(&addr) as usize].push_back(dial);
        }
        // A group waits only for groups that have something to try: with no QUIC address, TCP
        // starts at once; with no direct address at all, relays do.
        let released = [
            true,
            queued[0].is_empty(),
            queued[0].is_empty() && queued[1].is_empty(),
        ];
        let timers = [
            None,
            (!released[1]).then(|| Delay::new(tcp_delay)),
            (!released[2]).then(|| Delay::new(relay_delay)),
        ];
        let mut this = Self {
            dials: FuturesUnordered::new(),
            queued,
            in_flight: [0; 3],
            released,
            timers,
            concurrency: concurrency_factor.get() as usize,
            errors: Default::default(),
        };
        this.start_released();
        this
    }

    /// A group is finished once it may start, has nothing queued and nothing in flight.
    fn finished(&self, g: usize) -> bool {
        self.released[g] && self.queued[g].is_empty() && self.in_flight[g] == 0
    }

    /// Release every group whose predecessors are all finished, then start released attempts, in
    /// group order, up to the concurrency limit.
    fn start_released(&mut self) {
        for g in 1..3 {
            if !self.released[g] && (0..g).all(|e| self.finished(e)) {
                self.released[g] = true;
                self.timers[g] = None;
            }
        }
        while self.dials.len() < self.concurrency {
            let Some(g) = (0..3).find(|&g| self.released[g] && !self.queued[g].is_empty()) else {
                break;
            };
            let dial = self.queued[g].pop_front().expect("not empty");
            self.in_flight[g] += 1;
            let group = [Group::Quic, Group::OtherDirect, Group::Relay][g];
            self.dials.push(dial.map(move |out| (group, out)).boxed());
        }
    }
}

impl Future for ConcurrentDial {
    type Output = Result<
        // Either one dial succeeded, returning the negotiated [`PeerId`], the address, the
        // muxer and the addresses and errors of the dials that failed before.
        (
            Multiaddr,
            (PeerId, StreamMuxerBox),
            Vec<(Multiaddr, TransportError<std::io::Error>)>,
        ),
        // Or all dials failed, thus returning the address and error for each dial.
        Vec<(Multiaddr, TransportError<std::io::Error>)>,
    >;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        loop {
            // Timers: a held-back group whose delay has passed may start.
            for g in 1..3 {
                if let Some(t) = self.timers[g].as_mut() {
                    if t.poll_unpin(cx).is_ready() {
                        self.released[g] = true;
                        self.timers[g] = None;
                    }
                }
            }
            self.start_released();
            match self.dials.poll_next_unpin(cx) {
                Poll::Ready(Some((_, (addr, Ok(output))))) => {
                    let errors = std::mem::take(&mut self.errors);
                    return Poll::Ready(Ok((addr, output, errors)));
                }
                Poll::Ready(Some((group, (addr, Err(e))))) => {
                    self.in_flight[group as usize] -= 1;
                    self.errors.push((addr, e));
                    // Loop: the failure may release the next group.
                }
                Poll::Ready(None) => {
                    if self.queued.iter().all(|q| q.is_empty()) {
                        return Poll::Ready(Err(std::mem::take(&mut self.errors)));
                    }
                    // Nothing in flight but something held back (only reachable if every earlier
                    // attempt was refused before it started): release everything.
                    self.released = [true; 3];
                    self.timers = [None, None, None];
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::pending;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::time::Instant;

    fn addr(s: &str) -> Multiaddr {
        s.parse().unwrap()
    }
    const QUIC: &str = "/ip4/1.2.3.4/udp/4001/quic-v1";
    const TCP: &str = "/ip4/1.2.3.4/tcp/4001";
    const RELAY: &str = "/ip4/5.6.7.8/tcp/4001/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit";

    /// An attempt that counts when it is first polled (= when the transport would start), then
    /// fails after `fail_after`, succeeds after `ok_after`, or never finishes.
    fn attempt(a: &str, started: Arc<AtomicUsize>, outcome: Outcome) -> (Multiaddr, Dial) {
        let ma = addr(a);
        let out_addr = ma.clone();
        let dial = async move {
            started.fetch_add(1, Ordering::SeqCst);
            match outcome {
                Outcome::FailAfter(d) => {
                    Delay::new(d).await;
                    (out_addr, Err(TransportError::Other(std::io::Error::other("refused"))))
                }
                Outcome::OkAfter(d) => {
                    Delay::new(d).await;
                    let muxer = StreamMuxerBox::new(DummyMuxer);
                    (out_addr, Ok(("12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb".parse::<PeerId>().unwrap(), muxer)))
                }
                Outcome::Never => pending().await,
            }
        }
        .boxed();
        (ma, dial)
    }
    #[derive(Clone, Copy)]
    enum Outcome {
        FailAfter(Duration),
        OkAfter(Duration),
        Never,
    }

    struct DummyMuxer;
    impl libp2p_core::muxing::StreamMuxer for DummyMuxer {
        type Substream = futures::io::Cursor<Vec<u8>>;
        type Error = std::io::Error;
        fn poll_inbound(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<Self::Substream, Self::Error>> {
            Poll::Pending
        }
        fn poll_outbound(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<Self::Substream, Self::Error>> {
            Poll::Pending
        }
        fn poll_close(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<libp2p_core::muxing::StreamMuxerEvent, Self::Error>> {
            Poll::Pending
        }
    }

    fn n(v: u8) -> NonZeroU8 {
        NonZeroU8::new(v).unwrap()
    }
    const T: Duration = Duration::from_millis(100);
    const R: Duration = Duration::from_millis(200);

    #[test]
    fn groups() {
        assert_eq!(group_of(&addr(QUIC)), Group::Quic);
        assert_eq!(group_of(&addr(TCP)), Group::OtherDirect);
        assert_eq!(group_of(&addr(RELAY)), Group::Relay);
        assert_eq!(group_of(&addr("/ip4/5.6.7.8/udp/4001/quic-v1/p2p/12D3KooWEL5wEL3foSWUk1E1rXHLbveqTahoHKhAsEYhDsLUkyWb/p2p-circuit")), Group::Relay);
    }

    #[tokio::test]
    async fn a_fast_direct_connection_means_no_relay_or_tcp_attempt_starts() {
        let (q, t, r) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let dial = ConcurrentDial::with_delays(
            vec![
                attempt(RELAY, r.clone(), Outcome::OkAfter(Duration::ZERO)),
                attempt(TCP, t.clone(), Outcome::OkAfter(Duration::ZERO)),
                attempt(QUIC, q.clone(), Outcome::OkAfter(Duration::from_millis(20))),
            ],
            n(8),
            T,
            R,
        );
        let (won, _, errors) = dial.await.ok().expect("connects");
        assert_eq!(won, addr(QUIC));
        assert!(errors.is_empty());
        assert_eq!((q.load(Ordering::SeqCst), t.load(Ordering::SeqCst), r.load(Ordering::SeqCst)), (1, 0, 0));
    }

    #[tokio::test]
    async fn a_hanging_direct_address_lets_tcp_start_at_its_delay_and_relays_at_theirs() {
        let (q, t, r) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let started = Instant::now();
        let dial = ConcurrentDial::with_delays(
            vec![
                attempt(QUIC, q.clone(), Outcome::Never),
                attempt(TCP, t.clone(), Outcome::Never),
                attempt(RELAY, r.clone(), Outcome::OkAfter(Duration::ZERO)),
            ],
            n(8),
            T,
            R,
        );
        let (won, _, _) = dial.await.ok().expect("connects through the relay");
        assert_eq!(won, addr(RELAY));
        assert!(started.elapsed() >= R, "the relay waited for its delay");
        assert_eq!((q.load(Ordering::SeqCst), t.load(Ordering::SeqCst), r.load(Ordering::SeqCst)), (1, 1, 1));
    }

    #[tokio::test]
    async fn when_every_direct_attempt_fails_the_relay_starts_at_once() {
        let (q, r) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let started = Instant::now();
        let dial = ConcurrentDial::with_delays(
            vec![
                attempt(QUIC, q.clone(), Outcome::FailAfter(Duration::from_millis(10))),
                attempt(RELAY, r.clone(), Outcome::OkAfter(Duration::ZERO)),
            ],
            n(8),
            Duration::from_secs(5),
            Duration::from_secs(5),
        );
        let (won, _, errors) = dial.await.ok().expect("connects through the relay");
        assert_eq!(won, addr(RELAY));
        assert_eq!(errors.len(), 1);
        assert!(started.elapsed() < Duration::from_secs(1), "released by the failure, not the timer");
    }

    #[tokio::test]
    async fn with_no_direct_address_the_relay_starts_at_once() {
        let r = Arc::new(AtomicUsize::new(0));
        let started = Instant::now();
        let dial = ConcurrentDial::with_delays(
            vec![attempt(RELAY, r.clone(), Outcome::OkAfter(Duration::ZERO))],
            n(8),
            Duration::from_secs(5),
            Duration::from_secs(5),
        );
        assert!(dial.await.is_ok());
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[tokio::test]
    async fn direct_addresses_get_the_slots_even_when_relays_are_listed_first() {
        // Concurrency 1, relays listed first (as OpenHydra's own relay dials list them): the direct
        // address must still be tried, and win.
        let (q, r) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let mut list: Vec<_> = (0..4).map(|_| attempt(RELAY, r.clone(), Outcome::Never)).collect();
        list.push(attempt(QUIC, q.clone(), Outcome::OkAfter(Duration::ZERO)));
        let dial = ConcurrentDial::with_delays(list, n(1), T, R);
        let (won, _, _) = dial.await.ok().expect("connects directly");
        assert_eq!(won, addr(QUIC));
        assert_eq!(r.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn all_failing_returns_every_error() {
        let c = Arc::new(AtomicUsize::new(0));
        let dial = ConcurrentDial::with_delays(
            vec![
                attempt(QUIC, c.clone(), Outcome::FailAfter(Duration::ZERO)),
                attempt(TCP, c.clone(), Outcome::FailAfter(Duration::ZERO)),
                attempt(RELAY, c.clone(), Outcome::FailAfter(Duration::ZERO)),
            ],
            n(8),
            T,
            R,
        );
        let errors = dial.await.err().expect("all fail");
        assert_eq!(errors.len(), 3);
        assert_eq!(c.load(Ordering::SeqCst), 3);
    }
}
