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
- **Paid-so-far is per (consumer, provider, model)**, never per payout address, so moving a payout
  address continues from what was already settled instead of paying the running total again.
- **Exit = request, then wait.** A consumer calls `requestExit` / `request_exit`, then may withdraw
  after the refund delay — a fixed window for the provider to settle what it is owed, which settles
  can't extend. No withdrawals while paused, and the window counts from no earlier than the last
  unpause. A deposit cancels a pending exit.
- **Off-chain duty (Base):** a provider relies on a channel only after checking that
  `channels[consumer][provider].payout` is its own payout address (the consumer picks it at open).

## Toolchains

- **Solana:** Agave 4.2.x (`solana`, `cargo-build-sbf`) via the Anza stable installer; Anchor 1.2.0 via `avm`.
- **Base:** Foundry (`forge`, `anvil`).

## Build & test

### Solana (`solana/`)
```bash
cd solana
anchor build                      # → target/deploy/openhydra_settlement.so (BPF)
cargo test -p openhydra-settlement # 6 host unit tests + 7 litesvm integration tests
```
The litesvm integration tests need `litesvm 0.16` with the `precompiles` feature (already in the
program's `Cargo.toml`) so the Ed25519 SigVerify precompile is registered and actually verified
in-SVM. (The Anchor scaffold's default `litesvm 0.10` rejects the Agave-4.2 SBF — see the dev-dep
comment.)

### Base (`base/`)
Solidity deps are vendored via `git clone` (not committed). Fetch them once, then test:
```bash
cd base
git clone --depth 1 https://github.com/foundry-rs/forge-std lib/forge-std
git clone --depth 1 --branch v5.1.0 https://github.com/OpenZeppelin/openzeppelin-contracts lib/openzeppelin-contracts
forge test                        # 21 tests
```

## Audit checklist

Each spec's §8 carries the adversarial-review findings (2026-09-18) the audited version must resolve —
notably claim freshness (a signed `deadline`), the Solana `initialize` front-run, a Solana claim/binding
domain that names the program and cluster, and fee payouts a USDC blacklist can't block. See the specs.
Resolved since (2026-10-06 review): payout-address re-payment on Base, consumer self-pay on both chains
(per-provider escrow), settles extending the refund window, and `withdraw` vs. `paused`.
