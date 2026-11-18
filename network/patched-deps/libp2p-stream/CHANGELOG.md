## 0.2.0-alpha (OpenHydra patch)

OpenHydra streams a reply's tokens to its consumer over one of these streams (`/openhydra/push/`),
opened per request; these changes make that safe and quick under load. See the tests in
`src/shared.rs`.

- `Control::open_connected_stream`: opens a stream only on a connection the peer already has,
  never dialling (`OpenStreamError::NotConnected` at once otherwise). OpenHydra's own request path
  knows how to reach a peer (relays, reversal, hole punching); this one only rides the connection it
  made.
- A new stream goes on one of the peer's direct connections that has already carried a stream, when
  it has one; otherwise on any of its connections, at random as upstream. Upstream picked at random
  every time, so a stream could ride a relay circuit beside a working direct connection, as OpenHydra
  saw with requests (11 Nov 2026, see the libp2p-request-response patch). A direct connection that
  has carried nothing yet isn't preferred: right after a hole punch it may be silently dead.
- One connection negotiates up to 128 outbound streams at once. Upstream negotiated them one at a
  time, so each open waited for every open before it to finish negotiating.
- Up to 64 negotiated inbound streams of a protocol wait for the acceptor. Upstream's channel held
  one, so a burst of opens arriving between two polls of `IncomingStreams` was dropped.
- A closed connection's channel is dropped with it (upstream kept one per connection ever made).
- Every failed dial fails the opens waiting on it, except one not made because the peer is connected
  or being dialled already (upstream left the opens of, say, an aborted dial waiting forever).
- Backported from upstream 0.4.0-alpha (PR 5999): a protocol whose `IncomingStreams` was dropped
  can be accepted again.

## 0.2.0-alpha

<!-- Update to libp2p-swarm v0.45.0 -->

## 0.1.0-alpha.1
- Implement Error for `OpenStreamError`.
  See [PR 5169](https://github.com/libp2p/rust-libp2p/pull/5169).

## 0.1.0-alpha

Initial release.
