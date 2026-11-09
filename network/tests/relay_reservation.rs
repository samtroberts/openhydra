//! A node holds one reservation per relay, and gets it back once — not a storm of them — when its
//! relay connection drops (the roaming churn of 2026-10-29: a lost reservation was retried once per
//! circuit it reported, and libp2p kept closing the duplicates).
//!
//! Runs the real relay server (`openhydra-bootstrap`) on loopback with two external addresses, so
//! each reservation reports two circuits, as a production relay reports one per transport. Counts
//! what the relay saw: every reservation request it granted, new or a renewal on the same connection.
//!
//! Run:  cargo test -p openhydra-network --test relay_reservation

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use openhydra_network::handle::NetworkHandle;
use openhydra_network::node::NodeConfig;

/// A loopback port free for both TCP and UDP (the relay listens on both).
fn free_port() -> u16 {
    loop {
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        if std::net::UdpSocket::bind(("127.0.0.1", port)).is_ok() {
            return port;
        }
    }
}

/// The relay server, killed when dropped.
struct Relay {
    child: Child,
    log: PathBuf,
}

impl Drop for Relay {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_relay(dir: &Path, key: &Path, port: u16, run: u32) -> Relay {
    let log = dir.join(format!("relay-{run}.log"));
    let out = std::fs::File::create(&log).unwrap();
    let tcp = format!("/ip4/127.0.0.1/tcp/{port}");
    let quic = format!("/ip4/127.0.0.1/udp/{port}/quic-v1");
    let child = Command::new(env!("CARGO_BIN_EXE_openhydra-bootstrap"))
        .args(["--identity", key.to_str().unwrap(), "--listen", &tcp, "--listen", &quic])
        .args(["--external", &tcp, "--external", &quic])
        .env("RUST_LOG", "openhydra_bootstrap=debug,warn")
        .env("NO_COLOR", "1")
        .stdout(Stdio::from(out.try_clone().unwrap()))
        .stderr(Stdio::from(out))
        .spawn()
        .expect("start openhydra-bootstrap");
    Relay { child, log }
}

/// Lines of the relay's log naming `peer` and containing `what`.
fn count(relay: &Relay, what: &str, peer: &str) -> usize {
    std::fs::read_to_string(&relay.log)
        .unwrap_or_default()
        .lines()
        .filter(|l| l.contains(what) && l.contains(peer))
        .count()
}

/// Reservations the relay granted `peer`: new ones and renewals on the same connection (a second
/// request on one connection is a renewal there, so the churn shows up as renewals).
fn granted(relay: &Relay, peer: &str) -> usize {
    count(relay, "relay reservation accepted", peer) + count(relay, "relay reservation renewed", peer)
}

fn wait_for(limit: Duration, mut done: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + limit;
    while Instant::now() < end {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    done()
}

#[test]
fn a_dropped_relay_connection_is_reserved_on_again_exactly_once() {
    openhydra_network::node::disable_mdns();
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("relay.key");
    let relay_id = openhydra_network::identity::Identity::load_or_create(&key).unwrap().libp2p_peer_id;
    let port = free_port();
    let relay = start_relay(dir.path(), &key, port, 1);

    let net = NetworkHandle::start(NodeConfig {
        identity_path: dir.path().join("node.key"),
        listen_addrs: vec!["/ip4/127.0.0.1/tcp/0".into()],
        bootstrap_peers: vec![format!("/ip4/127.0.0.1/tcp/{port}/p2p/{relay_id}")],
        enable_peer_relay: false,
        enable_connection_reversal: false,
        pcp_gateway: None,
        // A private deployment: its only relay is its bootstrap, the one above.
        public_relays: openhydra_network::relay::PublicRelays::Off,
    })
    .unwrap();
    let me = net.libp2p_peer_id().to_string();

    assert!(wait_for(Duration::from_secs(60), || granted(&relay, &me) >= 1), "no reservation at all");
    std::thread::sleep(Duration::from_secs(5));
    assert_eq!(granted(&relay, &me), 1, "one reservation, asked for once");
    assert!(
        wait_for(Duration::from_secs(10), || net.status().map(|s| !s.relay_reservations.is_empty()).unwrap_or(false)),
        "the node knows it holds it"
    );

    // The relay goes away and comes back at the same address: the node's connection and its
    // reservation listener close (reporting both circuits), and it reserves again.
    drop(relay);
    let relay = start_relay(dir.path(), &key, port, 2);
    assert!(wait_for(Duration::from_secs(60), || granted(&relay, &me) >= 1), "never reserved again");
    std::thread::sleep(Duration::from_secs(10));
    let again = granted(&relay, &me);
    assert_eq!(again, 1, "reserved again once, not once per circuit (relay log: {})", relay.log.display());
    assert_eq!(count(&relay, "relay reservation request denied", &me), 0);
    assert!(
        wait_for(Duration::from_secs(10), || net.status().map(|s| !s.relay_reservations.is_empty()).unwrap_or(false)),
        "and holds it"
    );
}
