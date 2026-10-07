# OpenHydra on-chain settlement — DRAFT

> ⚠️ **STATUS: DRAFT / ARCHITECTURE — NOT AUDITED, NOT DEPLOYABLE.**
> These programs are the on-chain half of OpenHydra's crypto payment rail. Moving real USDC is
> **hard-gated** on a non-custodial legal opinion, a security audit, MiCA/FIU-IND geofencing, and a
> payer entity. This code exists to be built, localnet-tested, and audited — **not to ship**. Nothing
> here is authorization to deploy anything that moves value. Not legal or financial advice.

The chain settles OpenHydra's authenticity-gated **voucher** (the co-signed receipt's
`cumulative_amount`, already rate-capped to the *proven* authenticity tier off-chain) in USDC,
**non-custodially**, taking the 4% protocol fee on-chain. It does **not** re-verify authenticity — the
gate is baked into the voucher before it is signed.

| | Solana | Base (EVM) |
|---|---|---|
| Dir | `solana/` | `base/` |
| Framework | Anchor 1.2.0 | Foundry |
| On-chain sig verify | native **Ed25519** (SigVerify precompile + instruction introspection) | **secp256k1** `ecrecover` over EIP-712 |
| Spec | `docs/SOLANA_SETTLEMENT_PROGRAM_SPEC_v1.md` (local) | `docs/BASE_SETTLEMENT_CONTRACT_SPEC_v1.md` (local) |

Both mirror the off-chain primitives in `protocol/src/{settlement,payout,receipts}.rs`
**byte-for-byte** — the EIP-712 digest and the Solana claim preimage are pinned by cross-language
golden tests (`protocol/src/settlement.rs::{eip712_digest_cross_language_golden,
solana_preimage_cross_language_golden}`) that the on-chain tests reproduce.

## Escrow model (both chains)

- **Ring-fenced per provider.** A consumer funds a channel (Base) / vault (Solana) for **one** provider;
  settles pay only that provider's payout (Base: the channel's payout address, which only the current
  payout address can move; Solana: the provider's Ed25519-signed payout binding). A consumer can't pay
  itself out of escrow committed to a provider — a throwaway provider of its own reaches only what the
  consumer put in *its* channel.
- **Paid-so-far is per (consumer, provider, model)** — on Base per (wallet, provider, consumer id,
  model) — never per payout address, so moving a payout address continues from what was already
  settled instead of paying the running total again.
- **Exit = request, wait, then one withdraw within 3 days.** A consumer calls `requestExit` /
  `request_exit`, then may withdraw once the refund delay has passed — a fixed time for the provider
  to settle what it is owed, which settles can't extend. Time paused after the request extends the
  delay (a pause stops settles), by at most 30 days. The exit then stays open for 3 days
  (`EXIT_WINDOW`) and one withdraw: after that, or after a withdraw, a new request is needed, so an
  exit can't be kept armed and used months later ahead of a settle. Withdrawals aren't paused, so a pause
  can never trap escrow; a plain pause leaves an open window where it was, but a re-pause within a
  day of the last one ending continues that pause, which counts the gap too and can move an open
  window later (within the same 30-day cap). A deposit cancels a
  pending exit.
- **Off-chain duty (Base):** a provider relies on a channel only after checking that
  `channels[consumer][provider].payout` is its own payout address (the consumer picks it at open).
- **Off-chain duty (both chains): watch exits.** Once a consumer requests an exit (`ExitRequested` /
  `request_exit`), it may withdraw everything left after the refund delay. A provider serving that
  consumer must settle every claim it holds within the delay, or stop serving it while the exit is
  pending; a claim settled after the withdraw finds the escrow empty. Nothing stops a consumer
  requesting exits back to back (with a short refund delay its exit is armed most of the time), so
  the duty is standing, not occasional. The agent doesn't watch exits yet: that comes with the escrow
  check before serving (P2(a)); until then it is the provider operator's to do. A cooldown between
  exit requests was considered and not added: settling within the delay is what protects the
  provider, whatever the cadence.
- **Fee dust:** the 4% fee is floored on the running total of each (consumer, provider, model), so
  splitting payments across many keys or models can leave the protocol at most one micro-unit less
  per pair. Each extra pair costs its own escrow account and transactions, far more than the dust
  saved, so this is left as is.

## The claim (format v6)

What the consumer signs: provider, consumer, model, cumulative amount and an **expiry** (`expires_at`,
unix seconds; the agent signs 30 days ahead and providers want at least 15 left, more than any refund
delay). It names **no payout**: each chain pays the provider's payout (Solana: its binding; Base: the
channel's, which the consumer sets at open — so on Base a provider must check on-chain that the channel
pays its own address and holds what it is owed before relying on its claims). Each signature is bound to one deployment: Base by the EIP-712 domain (chain id +
contract, version `"2"`); Solana by the program id and a **cluster tag** (the cluster's genesis hash)
kept in the program's config, which only the program's **upgrade authority** can initialize. The Solana
payout binding carries the same domain. Refund delays are between 1 and 14 days on both chains, and pausing
extends an exit by at most 30 days (no indefinite lock by pausing). A claim lives 30 days of *unpaused*
time: every pause since it was signed extends it, up to 30 days in all, so a pause (or several) doesn't
void the claims that expire during it, and a pause after a claim ran out doesn't revive it. The
contracts remember the most recent 64 pauses for this. A pause begun within a day of the last one's end
continues it (the time between counts as paused, for claims and exits alike), so distinct pauses are a
day apart at least, the log covers every pause a live claim can span, and unpausing and re-pausing can't
push one out of it. Two consequences of counting that gap: a claim that ran out during it gets the gap
back (it can settle again, within the 30-day cap), and an exit requested during it is extended by the
gap before its request too (within its own 30-day cap). A settle past the escrow pays what it
holds and the rest after a top-up. On Base the paid-so-far is keyed by the consumer's Ed25519 id as well
as its wallet. Solana payout bindings can't be stamped more than 5 minutes ahead. Solana's `Settled` event names the
model by its hash, as Base's does (SHA-256 on Solana, keccak-256 on Base: not comparable across chains). Solana's `Config` grows only at its end (a deployed program
upgrading reallocates it). **Fees are pulled:** settles accrue the 4% and the fee recipient / authority
withdraws it to any address, so a blacklisted or frozen fee account can't stop settlement.

## Toolchains

- **Solana:** Agave 4.2.x (`solana`, `cargo-build-sbf`) via the Anza stable installer; Anchor 1.2.0 via `avm`.
- **Base:** Foundry (`forge`, `anvil`).

## Build & test

### Solana (`solana/`)
```bash
cd solana
anchor build                      # → target/deploy/openhydra_settlement.so (BPF)
cargo test -p openhydra-settlement # 7 host unit tests + 18 litesvm integration tests
```
`cargo build-sbf` defaults to platform-tools v1.54; one already in `~/.cache/solana` can be used
instead with `cargo build-sbf --tools-version v1.57` (as tested), saving a ~400 MB download.
The litesvm integration tests need `litesvm 0.16` with the `precompiles` feature (already in the
program's `Cargo.toml`) so the Ed25519 SigVerify precompile is registered and actually verified
in-SVM. (The Anchor scaffold's default `litesvm 0.10` rejects the Agave-4.2 SBF — see the dev-dep
comment.)

### Base (`base/`)
Solidity deps are vendored via `git` (not committed), each at a pinned commit. Fetch them once,
then test:
```bash
cd base
fetch() { git init -q "$1" && git -C "$1" fetch -q --depth 1 "$2" "$3" && git -C "$1" checkout -q FETCH_HEAD; }
fetch lib/forge-std https://github.com/foundry-rs/forge-std 0d006dafa09d0b3722575acd29338b79c23e7c2f
fetch lib/openzeppelin-contracts https://github.com/OpenZeppelin/openzeppelin-contracts 69c8def5f222ff96f2b5beff05dfba996368aa79  # v5.1.0
forge test                        # 39 tests
```

## Audit checklist

Each spec's §8 carries the adversarial-review findings the audited version must resolve. Resolved since
the 2026-10-06 review: payout-address re-payment on Base, consumer self-pay on both chains (per-provider
escrow), settles extending the refund window, `withdraw` vs. `paused`, and with claim format v6: claim
expiry, the Solana `initialize` front-run (upgrade authority only, capped refund delay), a Solana
claim/binding domain naming the program and cluster, and pulled fees. Still before any deployment: real
SPL transfers on Solana (the draft tracks balances), and an external audit of both programs.
