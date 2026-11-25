## OpenHydra patch (2026-11-25)

- When our hole-punch dial fails but a direct connection to the peer already exists (the other
  side's dial, with longer limits, got through), the punch counts as a success on that connection:
  no retry (it only duplicated dials) and no failed punch beside a working direct connection.
- The retrying side's hole-punch dial gives each address 5 round trips of the CONNECT exchange,
  3-8 s (`DialOpts::with_attempt_timeout`, our patched libp2p-swarm; feature `openhydra-swarm`, on
  by default).

## 0.12.0 + OpenHydra patch (2026-10-29)

Vendored from crates.io `libp2p-dcutr` 0.12.0 (MIT, unchanged apart from this). One addition:

- When one of our listen addresses expires (`FromSwarm::ExpiredListenAddr`), the address candidates
  observed on the network it was on are forgotten: those on its IP once no listen address of ours
  is left on it, and every IPv4 candidate once the last real IPv4 interface address goes (behind
  NAT the candidates carry the NAT's public address, which can't be told apart per interface).
  Virtual interfaces (container bridges, Tailscale/CGNAT range, libvirt, macOS virtualization,
  link-local) neither trigger that nor count as still up; loopback and relayed addresses are
  ignored. Our listen addresses are counted from `FromSwarm::NewListenAddr`. Upstream forgets a
  candidate only when 20 newer ones push it out of the cache.

  Packaging: `Cargo.toml.orig` is upstream's manifest as published; the vendored `Cargo.toml` only
  adds an empty `[workspace]`.

  Why: after a node changed networks it kept offering peers the old network's addresses in hole
  punches (OpenHydra test C, 2026-10-29), where nothing listens any more; the punches to them time
  out and can use up the attempts.

## 0.12.0

<!-- Update to libp2p-swarm v0.45.0 -->

## 0.11.1
- Use `web-time` instead of `instant`.
  See [PR 5347](https://github.com/libp2p/rust-libp2p/pull/5347).

## 0.11.0

- Add `ConnectionId` to `Event::DirectConnectionUpgradeSucceeded` and `Event::DirectConnectionUpgradeFailed`.
  See [PR 4558](https://github.com/libp2p/rust-libp2p/pull/4558).
- Exchange address _candidates_ instead of external addresses in `CONNECT`.
  If hole-punching wasn't working properly for you until now, this might be the reason why.
  See [PR 4624](https://github.com/libp2p/rust-libp2p/pull/4624).
- Simplify public API.
  We now only emit a single event: whether the hole-punch was successful or not.
  See [PR 4749](https://github.com/libp2p/rust-libp2p/pull/4749).

## 0.10.0

- Raise MSRV to 1.65.
  See [PR 3715].
- Remove deprecated items. See [PR 3700].

- Keep connection alive while we are using it. See [PR 3960].

[PR 3715]: https://github.com/libp2p/rust-libp2p/pull/3715
[PR 3700]: https://github.com/libp2p/rust-libp2p/pull/3700
[PR 3960]: https://github.com/libp2p/rust-libp2p/pull/3960

## 0.9.1

- Migrate from `prost` to `quick-protobuf`. This removes `protoc` dependency. See [PR 3312].

[PR 3312]: https://github.com/libp2p/rust-libp2p/pull/3312

## 0.9.0

- Update to `libp2p-core` `v0.39.0`.

- Update to `libp2p-swarm` `v0.42.0`.

- Declare `InboundUpgradeError` and `OutboundUpgradeError` as type aliases instead of renames.
  This is a workaround for a missing feature in `cargo semver-checks`. See [PR 3213].

- Require the node's local `PeerId` to be passed into the constructor of `libp2p_dcutr::Behaviour`. See [PR 3153].

- Rename types in public API to follow naming conventions defined in [issue 2217]. See [PR 3214].

[PR 3213]: https://github.com/libp2p/rust-libp2p/pull/3213
[PR 3153]: https://github.com/libp2p/rust-libp2p/pull/3153
[issue 2217]: https://github.com/libp2p/rust-libp2p/issues/2217
[PR 3214]: https://github.com/libp2p/rust-libp2p/pull/3214

## 0.8.1

- Skip unparsable multiaddr in `InboundUpgrade::upgrade_inbound` and
  `OutboundUpgrade::upgrade_outbound`. See [PR 3300].

[PR 3300]: https://github.com/libp2p/rust-libp2p/pull/3300

## 0.8.0

- Update to `prost-codec` `v0.3.0`.

- Update to `libp2p-core` `v0.38.0`.

- Update to `libp2p-swarm` `v0.41.0`.

- Replace `Behaviour`'s `NetworkBehaviour` implemention `inject_*` methods with the new `on_*` methods.
  See [PR 3011].

- Replace `direct::Handler` and `relayed::Handler`'s `ConnectionHandler` implemention `inject_*`
  methods with the new `on_*` methods. See [PR 3085].

- Update `rust-version` to reflect the actual MSRV: 1.62.0. See [PR 3090].

[PR 3085]: https://github.com/libp2p/rust-libp2p/pull/3085
[PR 3011]: https://github.com/libp2p/rust-libp2p/pull/3011
[PR 3090]: https://github.com/libp2p/rust-libp2p/pull/3090

## 0.7.0

- Update to `libp2p-core` `v0.37.0`.

- Update to `libp2p-swarm` `v0.40.0`.

- Fix WASM compilation. See [PR 2991].

[PR 2991]: https://github.com/libp2p/rust-libp2p/pull/2991/

## 0.6.0

- Update to `libp2p-swarm` `v0.39.0`.

- Update to `libp2p-core` `v0.36.0`.

## 0.5.1

- Make default features of `libp2p-core` optional. See [PR 2836].

[PR 2836]: https://github.com/libp2p/rust-libp2p/pull/2836/

## 0.5.0

- Update prost requirement from 0.10 to 0.11 which no longer installs the protoc Protobuf compiler.
  Thus you will need protoc installed locally. See [PR 2788].

- Update to `libp2p-swarm` `v0.38.0`.

- Expose `PROTOCOL_NAME`. See [PR 2734].

- Update to `libp2p-core` `v0.35.0`.

[PR 2734]: https://github.com/libp2p/rust-libp2p/pull/2734/
[PR 2788]: https://github.com/libp2p/rust-libp2p/pull/2788

## 0.4.0

- Update to `libp2p-core` `v0.34.0`.

- Update to `libp2p-swarm` `v0.37.0`.

## 0.3.1

- Upgrade at most one inbound connect request.

## 0.3.0

- Update to `libp2p-core` `v0.33.0`.

- Update to `libp2p-swarm` `v0.36.0`.

## 0.2.0

- Expose `InboundUpgradeError` and `OutboundUpgradeError`. See [PR, 2586].

- Update to `libp2p-swarm` `v0.35.0`.

[PR 2586]: https://github.com/libp2p/rust-libp2p/pull/2586

## 0.1.0 [2022-02-22]

- Initial release.
