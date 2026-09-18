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

## Toolchains

- **Solana:** Agave 4.2.x (`solana`, `cargo-build-sbf`) via the Anza stable installer; Anchor 1.2.0 via `avm`.
- **Base:** Foundry (`forge`, `anvil`).

## Build & test

### Solana (`solana/`)
```bash
cd solana
anchor build                      # → target/deploy/openhydra_settlement.so (BPF)
cargo test -p openhydra-settlement # 6 host unit tests + 3 litesvm integration tests
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
forge test                        # 14 tests
```

## Audit checklist

Each spec's §8 carries the adversarial-review findings (2026-09-18) the audited version must resolve —
notably claim freshness (a signed `deadline`), the Solana `initialize` front-run, and `withdraw`
vs. `paused`. See the specs.
