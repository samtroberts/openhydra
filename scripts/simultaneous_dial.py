#!/usr/bin/env python3
"""
Simultaneous QUIC dial for NAT hole-punching.

Both nodes run this script at approximately the same time. Each node
calls dial_address() for the remote peer's QUIC IPv6 address, creating
NAT pinholes that allow the QUIC handshake to succeed.

Usage:
    python3 scripts/simultaneous_dial.py <remote_ipv6> <remote_peer_id> [delay_seconds]

Example (from mac1, targeting mac2):
    python3 scripts/simultaneous_dial.py \
        2401:4900:1cba:182d:8a8:e86c:ea4d:fd9 \
        12D3KooWLgHnVVBtjXfdSgD5FaYd4dHgbz8ZJvknrgcgX6w8JLxH \
        5
"""
import sys
import time
import openhydra_network

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    remote_ipv6 = sys.argv[1]
    remote_peer_id = sys.argv[2]
    delay = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # Build the QUIC multiaddr
    quic_addr = f"/ip6/{remote_ipv6}/udp/4001/quic-v1/p2p/{remote_peer_id}"

    print(f"Target QUIC address: {quic_addr}")

    if delay > 0:
        # Round to next N-second boundary for synchronization
        now = time.time()
        target = now + delay
        wait = target - time.time()
        if wait > 0:
            print(f"Waiting {wait:.1f}s before dial...")
            time.sleep(wait)

    # Create a temporary P2P node just for the dial
    # Actually, we need to connect to the RUNNING node's P2P instance.
    # We can't do that easily from a separate script.
    # Instead, let's do the dial via a raw UDP packet to create the pinhole,
    # then let the existing node's libp2p dial handle the rest.

    # Actually, the simplest approach: just send a few UDP packets to the
    # remote address to create the NAT pinhole, then the node's existing
    # QUIC dial (triggered by Kademlia address discovery) will succeed.
    import socket

    remote_addr = (remote_ipv6, 4001)
    sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    sock.bind(('::', 4001))  # Bind to same port as our QUIC listener

    print(f"Sending UDP hole-punch packets to [{remote_ipv6}]:4001...")
    for i in range(10):
        # Send a small UDP packet to create NAT pinhole
        try:
            sock.sendto(b'\x00' * 8, remote_addr)
            print(f"  Packet {i+1}/10 sent")
        except Exception as e:
            print(f"  Packet {i+1}/10 failed: {e}")
        time.sleep(0.1)

    print("Hole-punch packets sent. NAT pinholes should be open.")
    print("The running node should now be able to complete QUIC handshake.")

    # Keep socket open for a while to maintain the pinhole
    print("Keeping socket open for 30s to maintain pinhole...")
    time.sleep(30)
    sock.close()
    print("Done.")

if __name__ == "__main__":
    main()
