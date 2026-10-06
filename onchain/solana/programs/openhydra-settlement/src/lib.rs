// Copyright 2026 OpenHydra contributors
// Licensed under the Apache License, Version 2.0 (the "License").
//
// ⚠️ STATUS: DRAFT / ARCHITECTURE — NOT AUDITED, NOT DEPLOYABLE. Moving real USDC is hard-gated on a
// non-custodial legal opinion, a program security audit, MiCA/FIU-IND geofencing, and a payer entity
// (SOLANA_SETTLEMENT_PROGRAM_SPEC_v1). This program exists to be built + localnet-tested and audited,
// not to ship.
//
// The on-chain half of OpenHydra's crypto rail: it settles the authenticity-gated *voucher* in USDC,
// non-custodially, taking the 4% protocol fee on-chain. Escrow is ring-fenced per provider: a consumer
// funds a vault for ONE provider, and settles from it pay only that provider's bound payout — so a
// consumer can't pay itself out of escrow committed to a provider (a throwaway provider of its own
// reaches only its own vault). Exits are "request, then wait `refund_delay_secs`", a delay settles
// can't extend and a pause extends by at most 30 days, then one withdraw within `EXIT_WINDOW_SECS`. It verifies the co-signed receipt's **Ed25519
// signature natively** (via the Ed25519 SigVerify program + instruction introspection), enforces
// **monotonic** per-channel settlement, and releases the delta split `0.96 provider / 0.04 platform`.
// It does NOT re-verify authenticity — that is already baked into the voucher's rate-capped
// `cumulative_amount` off-chain. The chain settles exactly what both parties co-signed.
//
// CUSTODY NOTE (draft scope): this draft models the escrow balance as a `u64` in the `Vault` account
// and marks every place a **production** build performs an SPL-token CPI (`// PRODUCTION SPL:`). The
// non-custodial *design* is a vault SPL token account owned by the vault PDA (spec §3); the accounting,
// monotonicity, fee split, and the Ed25519 introspection — the parts that must agree with the built
// off-chain `protocol/src/settlement.rs` primitives — are implemented and tested here.

use anchor_lang::prelude::*;
// Solana 3.x crate-split: these live in dedicated crates, not the anchor `solana_program` re-export.
use solana_instructions_sysvar::{load_current_index_checked, load_instruction_at_checked};
use solana_sdk_ids::ed25519_program;
use solana_sha256_hasher::hash as sha256;

declare_id!("J168LWSjxuSFFMMWCosn7dYrFcfAmCTGPRQyfqMKVe4H");

/// Domain separator for the settlement claim preimage — byte-for-byte `settlement::SETTLEMENT_CLAIM_DOMAIN`
/// (v2: claim format v6, bound to a deployment, with an expiry).
pub const SETTLEMENT_CLAIM_DOMAIN: &[u8] = b"openhydra/solana-settle/v2";
/// Domain separator for the payout binding preimage — byte-for-byte `settlement::PAYOUT_BINDING_DOMAIN`.
pub const PAYOUT_BINDING_DOMAIN: &[u8] = b"openhydra/solana-payout/v2";
/// The 4% protocol fee in basis points — matches `payout::PROTOCOL_FEE_BPS`.
pub const FEE_BPS: u16 = 400;
/// The longest refund delay a deployment may set: shorter than a claim's lifetime
/// (`settlement::CLAIM_LIFETIME_SECS`, 30 days), so a provider's last claim outlives an exit window.
pub const MAX_REFUND_DELAY_SECS: i64 = 14 * 24 * 60 * 60;
/// The shortest refund delay: with none, a consumer could request an exit and withdraw in one
/// transaction, ahead of the provider's settle.
pub const MIN_REFUND_DELAY_SECS: i64 = 24 * 60 * 60;
/// How much pausing can delay one exit, in total. A pause stops settles, so time paused after an exit
/// request extends its delay; capped, so the admin's pause can't lock escrow indefinitely. Also the
/// most a pause extends a claim ([`claim_grace`]).
pub const MAX_PAUSE_EXTENSION_SECS: i64 = 30 * 24 * 60 * 60;
/// How long an exit stays open once its delay has passed. One request opens one window, and a
/// withdraw closes it: an exit requested once can't be kept armed and used, months later, in one
/// transaction ahead of the provider's settle.
pub const EXIT_WINDOW_SECS: i64 = 3 * 24 * 60 * 60;
/// How far ahead of the cluster clock a payout binding's timestamp may be: a binding stamped far in
/// the future (say `u64::MAX`, by whoever got the identity key) would be impossible to rotate away from.
pub const BINDING_CLOCK_SKEW_SECS: i64 = 5 * 60;

// ── Pure, testable helpers (no chain state) ───────────────────────────────

/// The settlement claim preimage the Ed25519 signature is verified against — a byte-for-byte
/// reconstruction of `settlement::SettlementClaim::solana_preimage` (claim format v6):
/// `domain ‖ program_id(32) ‖ cluster(32) ‖ provider(32) ‖ consumer(32) ‖ cumulative(u64 LE) ‖
/// expires_at(u64 LE) ‖ model_len(u32 LE) ‖ model`. The program id and cluster tag bind a claim to
/// this deployment, so one signed for devnet or another deployment doesn't settle here.
pub fn settle_claim_preimage(
    program_id: &[u8; 32],
    cluster: &[u8; 32],
    provider: &[u8; 32],
    consumer: &[u8; 32],
    cumulative_amount: u64,
    expires_at: u64,
    model_id: &str,
) -> Vec<u8> {
    let model = model_id.as_bytes();
    let mut b = Vec::with_capacity(SETTLEMENT_CLAIM_DOMAIN.len() + 32 * 4 + 8 + 8 + 4 + model.len());
    b.extend_from_slice(SETTLEMENT_CLAIM_DOMAIN);
    b.extend_from_slice(program_id);
    b.extend_from_slice(cluster);
    b.extend_from_slice(provider);
    b.extend_from_slice(consumer);
    b.extend_from_slice(&cumulative_amount.to_le_bytes());
    b.extend_from_slice(&expires_at.to_le_bytes());
    b.extend_from_slice(&(model.len() as u32).to_le_bytes());
    b.extend_from_slice(model);
    b
}

/// The payout-binding preimage — byte-for-byte `settlement::binding_message`:
/// `domain ‖ program_id(32) ‖ cluster(32) ‖ payout_address(32) ‖ ts(u64 LE)`.
pub fn binding_preimage(program_id: &[u8; 32], cluster: &[u8; 32], payout_address: &[u8; 32], ts_unix_ms: u64) -> Vec<u8> {
    let mut b = Vec::with_capacity(PAYOUT_BINDING_DOMAIN.len() + 32 * 3 + 8);
    b.extend_from_slice(PAYOUT_BINDING_DOMAIN);
    b.extend_from_slice(program_id);
    b.extend_from_slice(cluster);
    b.extend_from_slice(payout_address);
    b.extend_from_slice(&ts_unix_ms.to_le_bytes());
    b
}

/// The fixed 0.96/0.04 split of a `delta`, flooring the fee (dust to the provider) — matches
/// `payout::protocol_fee`. `u128` intermediate so a large delta can't overflow.
pub fn split(delta: u64, fee_bps: u16) -> (u64, u64) {
    split_after(0, delta, fee_bps)
}

/// [`split`] of a `delta` paid on top of `prior` already settled: the fee is taken on the running
/// total, so splitting it into many settles never rounds the fee away.
pub fn split_after(prior: u64, delta: u64, fee_bps: u16) -> (u64, u64) {
    let fee_on = |x: u128| x * (fee_bps as u128) / 10_000u128;
    let fee = (fee_on(prior as u128 + delta as u128) - fee_on(prior as u128)) as u64;
    (delta - fee, fee)
}

/// A parsed single-signature Ed25519 SigVerify instruction: the pubkey that signed and the message it
/// signed. The native Ed25519 program has already verified `sig` against `(pubkey, message)`
/// cryptographically; the settle instruction only binds those to the *expected* values.
pub struct Ed25519Verified {
    pub pubkey: [u8; 32],
    pub message: Vec<u8>,
}

/// Parse a **self-contained** Ed25519 SigVerify instruction's data (one signature, offsets pointing
/// within this same instruction — what the standard builder emits, `instruction_index == u16::MAX`).
/// Layout: `count:u8 padding:u8 offsets[14] ...data...`. Returns the signed `(pubkey, message)`.
///
/// Introspection hardening (spec §5): reject any offsets that reference *another* instruction
/// (`instruction_index != u16::MAX`) — otherwise an attacker could point the "verified" message at
/// arbitrary bytes in a different instruction. All bounds are checked.
pub fn parse_ed25519_ix(data: &[u8]) -> Option<Ed25519Verified> {
    const CURRENT: u16 = u16::MAX; // the "this instruction" sentinel
    if data.len() < 16 {
        return None;
    }
    if data[0] != 1 {
        return None; // exactly one signature expected
    }
    let u16_at = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]);
    let sig_ix_index = u16_at(4);
    let pk_offset = u16_at(6) as usize;
    let pk_ix_index = u16_at(8);
    let msg_offset = u16_at(10) as usize;
    let msg_size = u16_at(12) as usize;
    let msg_ix_index = u16_at(14);
    // All three references must be to THIS instruction (self-contained), else reject.
    if sig_ix_index != CURRENT || pk_ix_index != CURRENT || msg_ix_index != CURRENT {
        return None;
    }
    let pk_end = pk_offset.checked_add(32)?;
    let msg_end = msg_offset.checked_add(msg_size)?;
    if pk_end > data.len() || msg_end > data.len() {
        return None;
    }
    let mut pubkey = [0u8; 32];
    pubkey.copy_from_slice(&data[pk_offset..pk_end]);
    Some(Ed25519Verified { pubkey, message: data[msg_offset..msg_end].to_vec() })
}

/// Model-id → 32-byte channel seed component (sha256, Solana's native hash).
fn model_hash(model_id: &str) -> [u8; 32] {
    sha256(model_id.as_bytes()).to_bytes()
}

/// Load the Ed25519 SigVerify instruction that MUST immediately precede this settle instruction, and
/// return the `(pubkey, message)` it verified. Fails closed if it is missing, is a different program,
/// or references another instruction. Binding it to `current_index − 1` (same tx) stops a signature
/// being replayed from another transaction's introspection.
fn verified_ed25519_before_current(sysvar: &AccountInfo) -> Result<Ed25519Verified> {
    let current = load_current_index_checked(sysvar)?;
    if current == 0 {
        return err!(SettlementError::MissingEd25519Instruction);
    }
    let ix = load_instruction_at_checked((current - 1) as usize, sysvar)?;
    if ix.program_id != ed25519_program::ID {
        return err!(SettlementError::MissingEd25519Instruction);
    }
    parse_ed25519_ix(&ix.data).ok_or(error!(SettlementError::MalformedEd25519Instruction))
}

// ── Program ───────────────────────────────────────────────────────────────

#[program]
pub mod openhydra_settlement {
    use super::*;

    /// The upgrade authority, once. Pins the 4% fee at init (governance-only to change, never a silent
    /// admin power) and the cluster tag claims are bound to.
    pub fn initialize(
        ctx: Context<Initialize>,
        fee_account: Pubkey,
        usdc_mint: Pubkey,
        refund_delay_secs: i64,
        cluster: [u8; 32],
    ) -> Result<()> {
        // Only the program's upgrade authority initializes (the accounts constraints check it), so
        // nobody can front-run a deployment with their own fee account or an endless refund delay.
        require!(
            (MIN_REFUND_DELAY_SECS..=MAX_REFUND_DELAY_SECS).contains(&refund_delay_secs),
            SettlementError::BadArgument
        );
        // A real cluster tag (the genesis hash), not an unset one shared by every deployment.
        require!(cluster != [0u8; 32], SettlementError::BadArgument);
        let config = &mut ctx.accounts.config;
        config.admin = ctx.accounts.admin.key();
        config.fee_account = fee_account;
        config.usdc_mint = usdc_mint;
        config.fee_bps = FEE_BPS; // the agreed 4%, fixed at init
        config.refund_delay_secs = refund_delay_secs;
        config.paused = false;
        config.paused_at = 0;
        config.last_pause_end = 0;
        config.paused_total = 0;
        config.cluster = cluster;
        config.accrued_fees = 0;
        Ok(())
    }

    /// Provider registers/rotates the account that receives its payouts. The provider's Ed25519
    /// identity must have signed `binding_preimage(payout_address, ts)` (verified via introspection),
    /// and `ts` must strictly exceed the stored one — so a kept old binding can't redirect funds back
    /// to a rotated-away address (spec §8, review F4).
    pub fn bind_payout(
        ctx: Context<BindPayout>,
        provider: [u8; 32],
        payout_address: [u8; 32],
        ts_unix_ms: u64,
    ) -> Result<()> {
        let verified = verified_ed25519_before_current(&ctx.accounts.instructions_sysvar)?;
        require!(verified.pubkey == provider, SettlementError::SignerMismatch);
        require!(
            verified.message
                == binding_preimage(&crate::ID.to_bytes(), &ctx.accounts.config.cluster, &payout_address, ts_unix_ms),
            SettlementError::MessageMismatch
        );
        let now = Clock::get()?.unix_timestamp;
        let latest_ms = now.saturating_add(BINDING_CLOCK_SKEW_SECS).max(0) as u64 * 1000;
        require!(ts_unix_ms <= latest_ms, SettlementError::BindingFromTheFuture);
        let binding = &mut ctx.accounts.payout_binding;
        // Monotonic rotation: reject a binding that is not strictly newer (blocks the rotation replay).
        require!(
            !binding.initialized || ts_unix_ms > binding.ts_unix_ms,
            SettlementError::StaleBinding
        );
        binding.provider = provider;
        binding.payout_address = payout_address;
        binding.ts_unix_ms = ts_unix_ms;
        binding.initialized = true;
        Ok(())
    }

    /// Consumer funds its escrow for ONE provider (the x402 `upto` ceiling for that provider). A
    /// deposit cancels a pending exit. DRAFT: credits the tracked balance.
    pub fn deposit(ctx: Context<Deposit>, provider: [u8; 32], amount: u64) -> Result<()> {
        require!(!ctx.accounts.config.paused, SettlementError::Paused);
        // PRODUCTION SPL: CPI `spl_token::transfer` (or `receiveWithAuthorization`-equivalent) from the
        // consumer's USDC ATA into the vault's SPL token account owned by the vault PDA. Draft tracks
        // the balance directly.
        let vault = &mut ctx.accounts.vault;
        vault.consumer = ctx.accounts.consumer.key();
        vault.provider = provider;
        vault.balance = vault.balance.checked_add(amount).ok_or(SettlementError::Overflow)?;
        vault.exit_requested_at = 0;
        Ok(())
    }

    /// Settle a co-signed voucher. Anyone (provider or facilitator) may submit. Requires an Ed25519
    /// SigVerify instruction immediately before this one that proves the **consumer** signed the claim
    /// preimage. Pays the monotonic delta, split 0.96/0.04, from the consumer's vault for THIS provider
    /// (never another provider's), and never moves an exit's deadline.
    pub fn settle(ctx: Context<Settle>, claim: SettleClaimArgs) -> Result<()> {
        require!(!ctx.accounts.config.paused, SettlementError::Paused);

        // 1. Ed25519 introspection: the consumer authorized THIS claim (native verify already checked
        //    the signature; we bind pubkey + message to the expected values).
        let verified = verified_ed25519_before_current(&ctx.accounts.instructions_sysvar)?;
        require!(verified.pubkey == claim.consumer, SettlementError::SignerMismatch);
        let expected = settle_claim_preimage(
            &crate::ID.to_bytes(),
            &ctx.accounts.config.cluster,
            &claim.provider,
            &claim.consumer,
            claim.cumulative_amount,
            claim.expires_at,
            &claim.model_id,
        );
        require!(verified.message == expected, SettlementError::MessageMismatch);
        let now = Clock::get()?.unix_timestamp;
        require!(now >= 0, SettlementError::ClaimExpired);
        let late = (now as u64).saturating_sub(claim.expires_at);
        require!(late <= claim_grace(&ctx.accounts.config, claim.expires_at), SettlementError::ClaimExpired);

        // 2. The provider is paid at its currently-bound address, whatever it is: the claim names no
        //    payout, so nobody (the consumer included) can point it elsewhere.
        let binding = &ctx.accounts.payout_binding;
        require!(binding.initialized, SettlementError::UnboundProvider);
        require!(binding.provider == claim.provider, SettlementError::SignerMismatch);
        let payout = binding.payout_address;

        // 3. Monotonic delta.
        let channel = &mut ctx.accounts.channel;
        require!(
            claim.cumulative_amount > channel.settled_amount,
            SettlementError::StaleOrReplayedClaim
        );
        let delta = claim.cumulative_amount - channel.settled_amount;

        // 4. Pay what the escrow covers: a consumer that ran past its escrow still pays all it has, and
        //    the rest settles from the same claim after a top-up (the high-water mark rises by what was
        //    paid, not by the claim).
        let vault = &mut ctx.accounts.vault;
        let paid = delta.min(vault.balance);
        require!(paid > 0, SettlementError::InsufficientEscrow);
        let (provider_net, fee) = split_after(channel.settled_amount, paid, ctx.accounts.config.fee_bps);

        // 5. Effects before interactions: advance the high-water mark + debit escrow.
        channel.settled_amount += paid;
        channel.consumer = claim.consumer;
        channel.provider = claim.provider;
        vault.balance -= paid;
        // The fee stays with the program until the fee authority withdraws it (`withdraw_fees`): paid
        // out on every settle, a fee account the token issuer froze would stop all settles.
        let config = &mut ctx.accounts.config;
        config.accrued_fees = config.accrued_fees.checked_add(fee).ok_or(SettlementError::Overflow)?;

        // PRODUCTION SPL: CPI `spl_token::transfer` from the vault token account:
        //   provider_net → the provider's bound USDC ATA (`payout`),
        //   fee          → the program's fee vault (a PDA token account).
        // Draft records the amounts in the event; no token moves.
        emit!(Settled {
            consumer: claim.consumer,
            provider: claim.provider,
            payout,
            delta: paid,
            outstanding: delta - paid,
            provider_net,
            fee,
        });
        Ok(())
    }

    /// Consumer announces it will reclaim a provider vault's unspent escrow: it can `withdraw` after
    /// `refund_delay_secs`, a fixed window for the provider to settle what it is owed, which settles
    /// can't extend (anyone could otherwise keep the escrow locked by settling small steps).
    pub fn request_exit(ctx: Context<RequestExit>, _provider: [u8; 32]) -> Result<()> {
        let now = Clock::get()?.unix_timestamp;
        let vault = &mut ctx.accounts.vault;
        vault.exit_requested_at = now;
        vault.exit_pause_mark = paused_total_at(&ctx.accounts.config, now);
        Ok(())
    }

    /// Consumer reclaims unspent escrow from a provider vault → the non-custodial exit. Opens
    /// `refund_delay_secs` after `request_exit`, plus however long the program has been paused since
    /// (a pause stops settles, so it must not use up the provider's window), at most
    /// `MAX_PAUSE_EXTENSION_SECS` — past that, even while paused, so a pause can't lock escrow forever.
    /// Stays open `EXIT_WINDOW_SECS`, for one withdraw: after it, or after a withdraw, a new exit is
    /// needed.
    pub fn withdraw(ctx: Context<Withdraw>, _provider: [u8; 32], amount: u64) -> Result<()> {
        let config = &ctx.accounts.config;
        let vault = &mut ctx.accounts.vault;
        require!(vault.exit_requested_at != 0, SettlementError::RefundLocked);
        let now = Clock::get()?.unix_timestamp;
        let paused_since_request = paused_total_at(config, now).saturating_sub(vault.exit_pause_mark).max(0);
        let open_at = vault
            .exit_requested_at
            .checked_add(config.refund_delay_secs)
            .and_then(|t| t.checked_add(paused_since_request.min(MAX_PAUSE_EXTENSION_SECS)))
            .ok_or(SettlementError::Overflow)?;
        require!(now >= open_at, SettlementError::RefundLocked);
        require!(now <= open_at.saturating_add(EXIT_WINDOW_SECS), SettlementError::ExitLapsed);
        require!(amount <= vault.balance, SettlementError::WithdrawTooLarge);
        // PRODUCTION SPL: CPI `spl_token::transfer` vault → consumer's USDC ATA.
        vault.exit_requested_at = 0;
        vault.balance -= amount;
        Ok(())
    }

    /// The fee authority (`config.fee_account`) takes accrued protocol fees, to any account it names:
    /// fees accrue in the program rather than being paid out on each settle.
    pub fn withdraw_fees(ctx: Context<WithdrawFees>, amount: u64) -> Result<()> {
        let config = &mut ctx.accounts.config;
        require!(amount <= config.accrued_fees, SettlementError::WithdrawTooLarge);
        // PRODUCTION SPL: CPI `spl_token::transfer` fee vault → the destination the authority names.
        config.accrued_fees -= amount;
        Ok(())
    }

    /// Admin pause (halts settle and deposit); no fund authority. Time paused extends pending exits'
    /// windows (see `withdraw`), at most `MAX_PAUSE_EXTENSION_SECS` per exit.
    pub fn set_paused(ctx: Context<SetPaused>, paused: bool) -> Result<()> {
        let config = &mut ctx.accounts.config;
        let now = Clock::get()?.unix_timestamp;
        if !config.paused && paused {
            config.paused_at = now;
        } else if config.paused && !paused {
            config.paused_total = config.paused_total.saturating_add(now.saturating_sub(config.paused_at).max(0));
            config.last_pause_end = now;
        }
        config.paused = paused;
        Ok(())
    }
}

/// How long past its expiry a claim still settles. A pause stops settles, so a claim that was still
/// live when the last pause began gets back the time that pause took, at most
/// `MAX_PAUSE_EXTENSION_SECS` (the same cap as an exit's, so a claim still outlives the exit window it
/// was accepted for). Only the last pause counts: a provider settles its claims once one ends.
pub fn claim_grace(config: &Config, expires_at: u64) -> u64 {
    if config.paused_at <= 0 || config.paused || expires_at < config.paused_at as u64 {
        return 0;
    }
    config.last_pause_end.saturating_sub(config.paused_at).clamp(0, MAX_PAUSE_EXTENSION_SECS) as u64
}

/// How long the program has been paused in total, up to `now` (the current pause included).
fn paused_total_at(config: &Config, now: i64) -> i64 {
    let current = if config.paused { now.saturating_sub(config.paused_at).max(0) } else { 0 };
    config.paused_total.saturating_add(current)
}

/// The claim the consumer's Ed25519 identity signs (== `settlement::SettlementClaim`, format v6).
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct SettleClaimArgs {
    pub provider: [u8; 32],
    pub consumer: [u8; 32],
    pub cumulative_amount: u64,
    /// Unix seconds after which the claim no longer settles.
    pub expires_at: u64,
    pub model_id: String,
}

// ── State ──

#[account]
#[derive(InitSpace)]
pub struct Config {
    pub admin: Pubkey,
    pub fee_account: Pubkey,
    pub usdc_mint: Pubkey,
    pub fee_bps: u16,
    pub refund_delay_secs: i64,
    pub paused: bool,
    /// When the current pause began, or the last one (0 = never paused).
    pub paused_at: i64,
    /// When the last pause ended (meaningful once a pause has ended).
    pub last_pause_end: i64,
    /// Seconds paused in total, over pauses that have ended: with `paused_at`, how much a pause delays
    /// an exit.
    pub paused_total: i64,
    /// The cluster this deployment runs on (its genesis hash), bound into every claim and binding
    /// with the program id. Set at initialize by the upgrade authority.
    pub cluster: [u8; 32],
    /// Protocol fees settled and not yet withdrawn by the fee authority.
    pub accrued_fees: u64,
}

/// A consumer's escrow for one provider.
#[account]
#[derive(InitSpace)]
pub struct Vault {
    pub consumer: Pubkey,
    pub provider: [u8; 32],
    pub balance: u64,
    /// When the consumer asked to exit (0 = no exit pending).
    pub exit_requested_at: i64,
    /// The program's total paused time when the exit was requested: pausing after that extends it.
    pub exit_pause_mark: i64,
}

#[account]
#[derive(InitSpace)]
pub struct Channel {
    pub consumer: [u8; 32],
    pub provider: [u8; 32],
    pub settled_amount: u64,
}

#[account]
#[derive(InitSpace)]
pub struct PayoutBinding {
    pub provider: [u8; 32],
    pub payout_address: [u8; 32],
    pub ts_unix_ms: u64,
    pub initialized: bool,
}

#[event]
pub struct Settled {
    pub consumer: [u8; 32],
    pub provider: [u8; 32],
    pub payout: [u8; 32],
    /// What this settle paid.
    pub delta: u64,
    /// What the claim was owed beyond the escrow (settles from the same claim after a top-up).
    pub outstanding: u64,
    pub provider_net: u64,
    pub fee: u64,
}

// ── Accounts contexts ──

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(mut)]
    pub admin: Signer<'info>,
    #[account(init, payer = admin, space = 8 + Config::INIT_SPACE, seeds = [b"config"], bump)]
    pub config: Account<'info, Config>,
    /// This program, to find its ProgramData account.
    #[account(constraint = program.programdata_address()? == Some(program_data.key()) @ SettlementError::NotUpgradeAuthority)]
    pub program: Program<'info, crate::program::OpenhydraSettlement>,
    /// Only the program's upgrade authority may initialize it.
    #[account(constraint = program_data.upgrade_authority_address == Some(admin.key()) @ SettlementError::NotUpgradeAuthority)]
    pub program_data: Account<'info, ProgramData>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct WithdrawFees<'info> {
    pub fee_authority: Signer<'info>,
    #[account(
        mut,
        seeds = [b"config"],
        bump,
        constraint = config.fee_account == fee_authority.key() @ SettlementError::NotFeeAuthority
    )]
    pub config: Account<'info, Config>,
}

#[derive(Accounts)]
#[instruction(provider: [u8; 32])]
pub struct BindPayout<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(seeds = [b"config"], bump)]
    pub config: Account<'info, Config>,
    #[account(
        init_if_needed,
        payer = payer,
        space = 8 + PayoutBinding::INIT_SPACE,
        seeds = [b"payout", provider.as_ref()],
        bump
    )]
    pub payout_binding: Account<'info, PayoutBinding>,
    /// CHECK: the Instructions sysvar, address-constrained; read via introspection helpers.
    #[account(address = solana_instructions_sysvar::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(provider: [u8; 32])]
pub struct Deposit<'info> {
    #[account(mut)]
    pub consumer: Signer<'info>,
    #[account(seeds = [b"config"], bump)]
    pub config: Account<'info, Config>,
    #[account(
        init_if_needed,
        payer = consumer,
        space = 8 + Vault::INIT_SPACE,
        seeds = [b"vault", consumer.key().as_ref(), provider.as_ref()],
        bump
    )]
    pub vault: Account<'info, Vault>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(claim: SettleClaimArgs)]
pub struct Settle<'info> {
    #[account(mut)]
    pub payer: Signer<'info>,
    #[account(mut, seeds = [b"config"], bump)]
    pub config: Account<'info, Config>,
    #[account(
        mut,
        seeds = [b"vault", claim.consumer.as_ref(), claim.provider.as_ref()],
        bump
    )]
    pub vault: Account<'info, Vault>,
    #[account(
        init_if_needed,
        payer = payer,
        space = 8 + Channel::INIT_SPACE,
        seeds = [b"channel", claim.consumer.as_ref(), claim.provider.as_ref(), &model_hash(&claim.model_id)],
        bump
    )]
    pub channel: Account<'info, Channel>,
    #[account(seeds = [b"payout", claim.provider.as_ref()], bump)]
    pub payout_binding: Account<'info, PayoutBinding>,
    /// CHECK: the Instructions sysvar, address-constrained; read via introspection helpers.
    #[account(address = solana_instructions_sysvar::ID)]
    pub instructions_sysvar: AccountInfo<'info>,
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(provider: [u8; 32])]
pub struct RequestExit<'info> {
    pub consumer: Signer<'info>,
    #[account(seeds = [b"config"], bump)]
    pub config: Account<'info, Config>,
    #[account(
        mut,
        seeds = [b"vault", consumer.key().as_ref(), provider.as_ref()],
        bump,
        constraint = vault.consumer == consumer.key() @ SettlementError::SignerMismatch
    )]
    pub vault: Account<'info, Vault>,
}

#[derive(Accounts)]
#[instruction(provider: [u8; 32])]
pub struct Withdraw<'info> {
    #[account(mut)]
    pub consumer: Signer<'info>,
    #[account(seeds = [b"config"], bump)]
    pub config: Account<'info, Config>,
    #[account(
        mut,
        seeds = [b"vault", consumer.key().as_ref(), provider.as_ref()],
        bump,
        constraint = vault.consumer == consumer.key() @ SettlementError::SignerMismatch
    )]
    pub vault: Account<'info, Vault>,
}

#[derive(Accounts)]
pub struct SetPaused<'info> {
    pub admin: Signer<'info>,
    #[account(
        mut,
        seeds = [b"config"],
        bump,
        constraint = config.admin == admin.key() @ SettlementError::NotAdmin
    )]
    pub config: Account<'info, Config>,
}

#[error_code]
pub enum SettlementError {
    #[msg("bad argument")]
    BadArgument,
    #[msg("arithmetic overflow")]
    Overflow,
    #[msg("the required Ed25519 SigVerify instruction is missing")]
    MissingEd25519Instruction,
    #[msg("the Ed25519 instruction data is malformed")]
    MalformedEd25519Instruction,
    #[msg("the verified signer is not the expected identity")]
    SignerMismatch,
    #[msg("the verified message does not match the expected preimage")]
    MessageMismatch,
    #[msg("the provider has no payout binding")]
    UnboundProvider,
    #[msg("a newer payout binding already exists")]
    StaleBinding,
    #[msg("the cumulative amount is stale or replayed (delta <= 0)")]
    StaleOrReplayedClaim,
    #[msg("insufficient escrow for the delta")]
    InsufficientEscrow,
    #[msg("withdraw exceeds the escrow balance")]
    WithdrawTooLarge,
    #[msg("escrow is refund-locked (no exit requested, or its window has not elapsed)")]
    RefundLocked,
    #[msg("the program is paused")]
    Paused,
    #[msg("only the admin may call this")]
    NotAdmin,
    #[msg("the claim has expired")]
    ClaimExpired,
    #[msg("only the program's upgrade authority may initialize it")]
    NotUpgradeAuthority,
    #[msg("only the fee authority may withdraw fees")]
    NotFeeAuthority,
    #[msg("the payout binding is stamped in the future")]
    BindingFromTheFuture,
    #[msg("the exit window closed; request a new exit")]
    ExitLapsed,
}

// ── Pure-logic unit tests (host `cargo test`, no validator) ────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// CROSS-LANGUAGE GOLDEN: the program reconstructs the SAME claim preimage as the off-chain
    /// `settlement::SettlementClaim::solana_preimage` (pinned in `settlement.rs::
    /// solana_preimage_cross_language_golden`). Same fixed inputs → the identical bytes, so the message
    /// this program's Ed25519 introspection matches is exactly the one the off-chain signer produced.
    #[test]
    fn claim_preimage_matches_off_chain_golden() {
        let pre =
            settle_claim_preimage(&[0x44; 32], &[0x55; 32], &[0x11; 32], &[0x22; 32], 100_000, 1_800_000_000, "qwen2.5:7b");
        let hex = pre.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(
            hex,
            "6f70656e68796472612f736f6c616e612d736574746c652f7632\
             4444444444444444444444444444444444444444444444444444444444444444\
             5555555555555555555555555555555555555555555555555555555555555555\
             1111111111111111111111111111111111111111111111111111111111111111\
             2222222222222222222222222222222222222222222222222222222222222222\
             a086010000000000\
             00d2496b00000000\
             0a000000\
             7177656e322e353a3762"
        );
    }

    #[test]
    fn split_is_four_percent_floored() {
        assert_eq!(split(100_000, FEE_BPS), (96_000, 4_000));
        // 101*400/10000 = 4.04 → floors to 4, dust to the provider (matches payout::protocol_fee).
        assert_eq!(split(101, FEE_BPS), (97, 4));
    }

    #[test]
    fn split_floors_the_fee_and_conserves_the_total() {
        for delta in [1u64, 25, 100, 101, 250, 9_600, 1_000_000, u64::MAX / 2] {
            let (net, fee) = split(delta, FEE_BPS);
            assert_eq!(net + fee, delta, "conservation: net + fee == delta");
            // Fee is floored: fee == floor(delta*400/10000).
            assert_eq!(fee as u128, (delta as u128) * 400 / 10_000);
        }
    }

    #[test]
    fn the_fee_on_split_settles_matches_one_settle() {
        // 48 in one settle: a fee of 1. In two of 24, each alone would floor to 0.
        assert_eq!(split_after(0, 48, FEE_BPS), (47, 1));
        let (_, first) = split_after(0, 24, FEE_BPS);
        let (_, second) = split_after(24, 24, FEE_BPS);
        assert_eq!(first + second, 1);
        // Any split of a total pays the same fee.
        let mut prior = 0u64;
        let mut fees = 0u64;
        for step in [7u64, 1, 300, 49, 643] {
            let (net, fee) = split_after(prior, step, FEE_BPS);
            assert_eq!(net + fee, step);
            fees += fee;
            prior += step;
        }
        assert_eq!(fees, split(prior, FEE_BPS).1);
    }

    #[test]
    fn parse_ed25519_ix_extracts_pubkey_and_message() {
        // Build a self-contained Ed25519 instruction data buffer the way the standard builder does:
        // count=1, padding, offsets (all instruction_index = u16::MAX), then pubkey ‖ sig ‖ message.
        let pubkey = [0x22u8; 32];
        let sig = [0x55u8; 64];
        let message = b"openhydra/solana-settle/v1 ... claim".to_vec();
        let mut data = vec![1u8, 0u8]; // count, padding
        let header_len = 2 + 14;
        let pk_off = header_len as u16;
        let sig_off = pk_off + 32;
        let msg_off = sig_off + 64;
        let cur = u16::MAX;
        // offsets: sig_offset, sig_ix_index, pk_offset, pk_ix_index, msg_offset, msg_size, msg_ix_index
        for v in [sig_off, cur, pk_off, cur, msg_off, message.len() as u16, cur] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data.extend_from_slice(&pubkey);
        data.extend_from_slice(&sig);
        data.extend_from_slice(&message);

        let parsed = parse_ed25519_ix(&data).expect("valid self-contained ed25519 ix");
        assert_eq!(parsed.pubkey, pubkey);
        assert_eq!(parsed.message, message);
    }

    #[test]
    fn parse_ed25519_ix_rejects_cross_instruction_and_malformed() {
        // An offsets struct that references ANOTHER instruction (index != u16::MAX) is rejected — the
        // classic Solana introspection attack (point the "verified" message at attacker bytes).
        let mut data = vec![1u8, 0u8];
        for v in [16u16, 0u16 /* cross-ix! */, 48, u16::MAX, 112, 4, u16::MAX] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data.extend_from_slice(&[0u8; 200]);
        assert!(parse_ed25519_ix(&data).is_none(), "cross-instruction reference rejected");
        // Too short / wrong count.
        assert!(parse_ed25519_ix(&[]).is_none());
        assert!(parse_ed25519_ix(&[2u8, 0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).is_none());
    }

    #[test]
    fn binding_preimage_matches_the_domain_layout() {
        let pre = binding_preimage(&[0x44; 32], &[0x55; 32], &[0x33; 32], 1_000);
        assert_eq!(&pre[..PAYOUT_BINDING_DOMAIN.len()], PAYOUT_BINDING_DOMAIN);
        assert_eq!(pre.len(), PAYOUT_BINDING_DOMAIN.len() + 32 * 3 + 8);
        assert_eq!(&pre[pre.len() - 8..], &1_000u64.to_le_bytes());
    }
}
