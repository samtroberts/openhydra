// Localnet integration tests for the OpenHydra Solana settlement program — DRAFT.
//
// Uses LiteSVM (in-process SVM, no validator). Exercises the full flow end-to-end, including the
// **Ed25519 SigVerify instruction + introspection** that is the program's core authorization
// mechanism: the consumer signs the claim preimage, an Ed25519 precompile instruction carries the
// (pubkey, message, signature), and `settle` introspects it to bind the verified signer + message to
// the expected claim. The Ed25519 precompile is loaded (litesvm `precompiles` feature), so its
// signatures are actually cryptographically verified in-SVM; these tests sign REAL claim/binding
// preimages and assert the accounting + monotonicity + fee split the program layers on top.
//
// TOOLCHAIN NOTE: the `avm`-latest Anchor 1.2.0 scaffold pins `litesvm = "0.10"`, whose Solana-3.x SBF
// loader REJECTS the Agave-4.2 `cargo-build-sbf` output (`add_program` → `InvalidAccountData`; the
// stock `anchor init` scaffold's own test fails identically). Fixed by moving the dev-deps to
// **litesvm 0.16** (loader on `solana-program-runtime 4.2`, matching the installed toolchain), which
// keeps `solana-pubkey`/`solana-instruction` at 3.x — the SAME major anchor-lang 1.2.0 pins — so the
// `Pubkey`/`Instruction` types still unify and no test rewrite was needed. The `precompiles` feature
// then registers the Ed25519 program so the introspected SigVerify instruction executes. Details in
// `programs/openhydra-settlement/Cargo.toml`.

use {
    anchor_lang::{
        prelude::Pubkey,
        solana_program::instruction::Instruction,
        system_program, AccountDeserialize, InstructionData, ToAccountMetas,
    },
    litesvm::LiteSVM,
    openhydra_settlement::{binding_preimage, settle_claim_preimage},
    solana_keypair::Keypair,
    solana_message::{Message, VersionedMessage},
    solana_signer::Signer,
    solana_transaction::versioned::VersionedTransaction,
};

const PROGRAM_SO: &[u8] =
    include_bytes!(concat!(env!("CARGO_TARGET_TMPDIR"), "/../deploy/openhydra_settlement.so"));

fn model_hash(model_id: &str) -> [u8; 32] {
    solana_sha256_hasher::hash(model_id.as_bytes()).to_bytes()
}

/// Build a self-contained Ed25519 SigVerify instruction: one signature, offsets pointing within this
/// instruction (`instruction_index == u16::MAX`), layout `[1][0][offsets:14][pubkey][sig][message]`.
fn ed25519_ix(pubkey: &[u8; 32], sig: &[u8; 64], message: &[u8]) -> Instruction {
    let cur = u16::MAX;
    let pk_off: u16 = 16;
    let sig_off: u16 = 48;
    let msg_off: u16 = 112;
    let mut data = vec![1u8, 0u8];
    for v in [sig_off, cur, pk_off, cur, msg_off, message.len() as u16, cur] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data.extend_from_slice(pubkey);
    data.extend_from_slice(sig);
    data.extend_from_slice(message);
    Instruction { program_id: solana_sdk_ids::ed25519_program::ID, accounts: vec![], data }
}

fn sign64(kp: &Keypair, message: &[u8]) -> [u8; 64] {
    let sig = kp.sign_message(message);
    let mut out = [0u8; 64];
    out.copy_from_slice(sig.as_ref());
    out
}

struct Ctx {
    svm: LiteSVM,
    program_id: Pubkey,
    payer: Keypair,
    admin: Keypair,
    fee_authority: Keypair,
    config: Pubkey,
    instr_sysvar: Pubkey,
}

/// The cluster tag this test deployment is initialized with (a real one is the genesis hash).
const CLUSTER: [u8; 32] = [0x55; 32];
/// Claims in these tests are valid for 30 days from the test clock.
const T0: i64 = 1_760_000_000;
const EXPIRES: u64 = T0 as u64 + 30 * 86_400;

fn program_data_address(program_id: &Pubkey) -> Pubkey {
    Pubkey::find_program_address(&[program_id.as_ref()], &solana_sdk_ids::bpf_loader_upgradeable::ID).0
}

fn setup() -> Ctx {
    let program_id = openhydra_settlement::ID;
    let mut svm = LiteSVM::new();
    svm.add_program(program_id, PROGRAM_SO).unwrap();
    let payer = Keypair::new();
    let admin = Keypair::new();
    svm.airdrop(&payer.pubkey(), 10_000_000_000).unwrap();
    svm.airdrop(&admin.pubkey(), 10_000_000_000).unwrap();
    let config = Pubkey::find_program_address(&[b"config"], &program_id).0;
    let instr_sysvar = solana_sdk_ids::sysvar::instructions::ID;
    // A real cluster's clock, not litesvm's genesis 0 (the program uses 0 as "no exit requested").
    let mut clock: solana_clock::Clock = svm.get_sysvar();
    clock.unix_timestamp = T0;
    svm.set_sysvar(&clock);
    // litesvm deploys upgradeable with no upgrade authority; make `admin` the authority (the
    // ProgramData state: tag u32 = 3 ‖ slot u64 ‖ Some(authority) = 1 ‖ pubkey).
    let pd = program_data_address(&program_id);
    let mut acct = svm.get_account(&pd).unwrap();
    acct.data[12] = 1;
    acct.data[13..45].copy_from_slice(admin.pubkey().as_ref());
    svm.set_account(pd, acct).unwrap();
    let fee_authority = Keypair::new();
    svm.airdrop(&fee_authority.pubkey(), 1_000_000_000).unwrap();
    Ctx { svm, program_id, payer, admin, fee_authority, config, instr_sysvar }
}

fn send(ctx: &mut Ctx, ixs: &[Instruction], signers: &[&Keypair]) -> Result<(), String> {
    // A fresh blockhash per transaction, so a repeated instruction (e.g. a retried withdraw) is a new
    // transaction, never rejected as an already-processed duplicate.
    ctx.svm.expire_blockhash();
    let blockhash = ctx.svm.latest_blockhash();
    let msg = Message::new_with_blockhash(ixs, Some(&signers[0].pubkey()), &blockhash);
    let tx = VersionedTransaction::try_new(VersionedMessage::Legacy(msg), signers)
        .map_err(|e| format!("sign: {e}"))?;
    ctx.svm.send_transaction(tx).map(|_| ()).map_err(|e| format!("{:?}", e.err))
}

fn try_initialize(ctx: &mut Ctx, by: &Keypair, refund_delay_secs: i64) -> Result<(), String> {
    let ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::Initialize {
            admin: by.pubkey(),
            config: ctx.config,
            program: ctx.program_id,
            program_data: program_data_address(&ctx.program_id),
            system_program: system_program::ID,
        }
        .to_account_metas(None),
        data: openhydra_settlement::instruction::Initialize {
            fee_account: ctx.fee_authority.pubkey(),
            usdc_mint: Pubkey::new_unique(),
            refund_delay_secs,
            cluster: CLUSTER,
        }
        .data(),
    };
    let by = by.insecure_clone();
    send(ctx, &[ix], &[&by])
}

fn initialize(ctx: &mut Ctx) {
    let admin = ctx.admin.insecure_clone();
    try_initialize(ctx, &admin, 86_400).expect("initialize");
}

fn bind_payout(ctx: &mut Ctx, provider: &Keypair, payout: [u8; 32], ts: u64) -> Result<(), String> {
    let provider_id = provider.pubkey().to_bytes();
    let binding_pda = Pubkey::find_program_address(&[b"payout", &provider_id], &ctx.program_id).0;
    let msg = binding_preimage(&ctx.program_id.to_bytes(), &CLUSTER, &payout, ts);
    let sig = sign64(provider, &msg);
    let ed_ix = ed25519_ix(&provider_id, &sig, &msg);
    let bind_ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::BindPayout {
            payer: ctx.payer.pubkey(),
            config: ctx.config,
            payout_binding: binding_pda,
            instructions_sysvar: ctx.instr_sysvar,
            system_program: system_program::ID,
        }
        .to_account_metas(None),
        data: openhydra_settlement::instruction::BindPayout { provider: provider_id, payout_address: payout, ts_unix_ms: ts }.data(),
    };
    let payer = ctx.payer.insecure_clone();
    send(ctx, &[ed_ix, bind_ix], &[&payer])
}

fn vault_pda(ctx: &Ctx, consumer: &Keypair, provider_id: [u8; 32]) -> Pubkey {
    Pubkey::find_program_address(&[b"vault", consumer.pubkey().as_ref(), &provider_id], &ctx.program_id).0
}

fn try_deposit(ctx: &mut Ctx, consumer: &Keypair, provider_id: [u8; 32], amount: u64) -> Result<(), String> {
    let vault = vault_pda(ctx, consumer, provider_id);
    let ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::Deposit {
            consumer: consumer.pubkey(),
            config: ctx.config,
            vault,
            system_program: system_program::ID,
        }
        .to_account_metas(None),
        data: openhydra_settlement::instruction::Deposit { provider: provider_id, amount }.data(),
    };
    let consumer = consumer.insecure_clone();
    send(ctx, &[ix], &[&consumer])
}

fn deposit(ctx: &mut Ctx, consumer: &Keypair, provider_id: [u8; 32], amount: u64) {
    try_deposit(ctx, consumer, provider_id, amount).expect("deposit");
}

fn request_exit(ctx: &mut Ctx, consumer: &Keypair, provider_id: [u8; 32]) -> Result<(), String> {
    let ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::RequestExit {
            consumer: consumer.pubkey(),
            vault: vault_pda(ctx, consumer, provider_id),
        }
        .to_account_metas(None),
        data: openhydra_settlement::instruction::RequestExit { _provider: provider_id }.data(),
    };
    let consumer = consumer.insecure_clone();
    send(ctx, &[ix], &[&consumer])
}

fn withdraw(ctx: &mut Ctx, consumer: &Keypair, provider_id: [u8; 32], amount: u64) -> Result<(), String> {
    let ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::Withdraw {
            consumer: consumer.pubkey(),
            config: ctx.config,
            vault: vault_pda(ctx, consumer, provider_id),
        }
        .to_account_metas(None),
        data: openhydra_settlement::instruction::Withdraw { _provider: provider_id, amount }.data(),
    };
    let consumer = consumer.insecure_clone();
    send(ctx, &[ix], &[&consumer])
}

fn set_paused(ctx: &mut Ctx, paused: bool) -> Result<(), String> {
    let ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::SetPaused { admin: ctx.admin.pubkey(), config: ctx.config }
            .to_account_metas(None),
        data: openhydra_settlement::instruction::SetPaused { paused }.data(),
    };
    let admin = ctx.admin.insecure_clone();
    send(ctx, &[ix], &[&admin])
}

/// Move the cluster clock forward.
fn advance(ctx: &mut Ctx, secs: i64) {
    let mut clock: solana_clock::Clock = ctx.svm.get_sysvar();
    clock.unix_timestamp += secs;
    ctx.svm.set_sysvar(&clock);
}

/// A settle instruction pair: the consumer's Ed25519 signature over `signed` (the claim it signed),
/// then `settle` with `claim`.
fn settle_ixs(ctx: &Ctx, consumer: &Keypair, signed: &[u8], claim: openhydra_settlement::SettleClaimArgs) -> [Instruction; 2] {
    let consumer_id = consumer.pubkey().to_bytes();
    let channel = Pubkey::find_program_address(
        &[b"channel", &consumer_id, &claim.provider, &model_hash(&claim.model_id)],
        &ctx.program_id,
    )
    .0;
    let binding_pda = Pubkey::find_program_address(&[b"payout", &claim.provider], &ctx.program_id).0;
    let sig = sign64(consumer, signed);
    let ed_ix = ed25519_ix(&consumer_id, &sig, signed);
    let settle_ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::Settle {
            payer: ctx.payer.pubkey(),
            config: ctx.config,
            vault: vault_pda(ctx, consumer, claim.provider),
            channel,
            payout_binding: binding_pda,
            instructions_sysvar: ctx.instr_sysvar,
            system_program: system_program::ID,
        }
        .to_account_metas(None),
        data: openhydra_settlement::instruction::Settle { claim }.data(),
    };
    [ed_ix, settle_ix]
}

fn claim(consumer: &Keypair, provider_id: [u8; 32], cumulative: u64, expires_at: u64, model_id: &str) -> openhydra_settlement::SettleClaimArgs {
    openhydra_settlement::SettleClaimArgs {
        provider: provider_id,
        consumer: consumer.pubkey().to_bytes(),
        cumulative_amount: cumulative,
        expires_at,
        model_id: model_id.to_string(),
    }
}

/// Settle `cumulative` for `(consumer, provider, model)`, the consumer signing the claim for this
/// deployment with `expires_at`.
fn settle_until(
    ctx: &mut Ctx,
    consumer: &Keypair,
    provider_id: [u8; 32],
    cumulative: u64,
    model_id: &str,
    expires_at: u64,
) -> Result<(), String> {
    let c = claim(consumer, provider_id, cumulative, expires_at, model_id);
    let msg = settle_claim_preimage(
        &ctx.program_id.to_bytes(), &CLUSTER, &c.provider, &c.consumer, cumulative, expires_at, model_id,
    );
    let ixs = settle_ixs(ctx, consumer, &msg, c);
    let payer = ctx.payer.insecure_clone();
    send(ctx, &ixs, &[&payer])
}

fn settle(ctx: &mut Ctx, consumer: &Keypair, provider_id: [u8; 32], cumulative: u64, model_id: &str) -> Result<(), String> {
    settle_until(ctx, consumer, provider_id, cumulative, model_id, EXPIRES)
}

fn vault_state(ctx: &Ctx, consumer: &Keypair, provider_id: [u8; 32]) -> openhydra_settlement::Vault {
    let vault = vault_pda(ctx, consumer, provider_id);
    let acct = ctx.svm.get_account(&vault).unwrap();
    openhydra_settlement::Vault::try_deserialize(&mut &acct.data[..]).unwrap()
}

fn channel_state(ctx: &Ctx, consumer: &Keypair, provider_id: [u8; 32], model_id: &str) -> openhydra_settlement::Channel {
    let channel = Pubkey::find_program_address(
        &[b"channel", &consumer.pubkey().to_bytes(), &provider_id, &model_hash(model_id)],
        &ctx.program_id,
    )
    .0;
    let acct = ctx.svm.get_account(&channel).unwrap();
    openhydra_settlement::Channel::try_deserialize(&mut &acct.data[..]).unwrap()
}

#[test]
fn full_flow_initialize_bind_deposit_settle_monotonic() {
    let mut ctx = setup();
    initialize(&mut ctx);

    let consumer = Keypair::new();
    let provider = Keypair::new();
    ctx.svm.airdrop(&consumer.pubkey(), 5_000_000_000).unwrap();
    let provider_id = provider.pubkey().to_bytes();
    let payout = provider_id; // self-directed payout

    bind_payout(&mut ctx, &provider, payout, 1_000).expect("bind_payout");
    deposit(&mut ctx, &consumer, provider_id, 500_000);
    assert_eq!(vault_state(&ctx, &consumer, provider_id).balance, 500_000);

    // Settle a cumulative of 100_000 → high-water advances, escrow debited by the delta.
    settle(&mut ctx, &consumer, provider_id, 100_000, "qwen2.5:7b").expect("settle #1");
    assert_eq!(channel_state(&ctx, &consumer, provider_id, "qwen2.5:7b").settled_amount, 100_000);
    assert_eq!(vault_state(&ctx, &consumer, provider_id).balance, 400_000);

    // A later cumulative of 250_000 pays only the 150_000 delta (monotonic, batched).
    settle(&mut ctx, &consumer, provider_id, 250_000, "qwen2.5:7b").expect("settle #2");
    assert_eq!(channel_state(&ctx, &consumer, provider_id, "qwen2.5:7b").settled_amount, 250_000);
    assert_eq!(vault_state(&ctx, &consumer, provider_id).balance, 250_000);
}

#[test]
fn replayed_and_stale_cumulatives_are_rejected() {
    let mut ctx = setup();
    initialize(&mut ctx);
    let consumer = Keypair::new();
    let provider = Keypair::new();
    ctx.svm.airdrop(&consumer.pubkey(), 5_000_000_000).unwrap();
    let provider_id = provider.pubkey().to_bytes();
    bind_payout(&mut ctx, &provider, provider_id, 1_000).unwrap();
    deposit(&mut ctx, &consumer, provider_id, 500_000);

    settle(&mut ctx, &consumer, provider_id, 100_000, "qwen2.5:7b").unwrap();
    // Replaying the same cumulative → delta 0 → the program rejects it.
    assert!(settle(&mut ctx, &consumer, provider_id, 100_000, "qwen2.5:7b").is_err());
    // A lower cumulative than the high-water → also rejected (never claws back).
    assert!(settle(&mut ctx, &consumer, provider_id, 50_000, "qwen2.5:7b").is_err());
    // The escrow was debited exactly once.
    assert_eq!(vault_state(&ctx, &consumer, provider_id).balance, 400_000);
}

#[test]
fn settle_rejects_a_tampered_claim_message() {
    // The Ed25519 introspection binds the verified message to the reconstructed preimage. If the
    // signed message and the claim args disagree, `settle` must reject (MessageMismatch). Here the
    // consumer signs a preimage for cumulative 100_000 but the settle args claim 999_999.
    let mut ctx = setup();
    initialize(&mut ctx);
    let consumer = Keypair::new();
    let provider = Keypair::new();
    ctx.svm.airdrop(&consumer.pubkey(), 5_000_000_000).unwrap();
    let provider_id = provider.pubkey().to_bytes();
    let consumer_id = consumer.pubkey().to_bytes();
    bind_payout(&mut ctx, &provider, provider_id, 1_000).unwrap();
    deposit(&mut ctx, &consumer, provider_id, 5_000_000);

    // Sign the HONEST claim (100_000)...
    let honest = settle_claim_preimage(
        &ctx.program_id.to_bytes(), &CLUSTER, &provider_id, &consumer_id, 100_000, EXPIRES, "qwen2.5:7b",
    );
    // ...but submit a settle claiming 999_999.
    let [ed_ix, settle_ix] = settle_ixs(&ctx, &consumer, &honest, claim(&consumer, provider_id, 999_999, EXPIRES, "qwen2.5:7b"));
    let payer = ctx.payer.insecure_clone();
    assert!(send(&mut ctx, &[ed_ix, settle_ix], &[&payer]).is_err(), "message/claim mismatch rejected");
}

/// A provider with a payout binding, and a funded consumer with a vault for it.
fn funded(ctx: &mut Ctx, amount: u64) -> (Keypair, Keypair, [u8; 32]) {
    let consumer = Keypair::new();
    let provider = Keypair::new();
    ctx.svm.airdrop(&consumer.pubkey(), 5_000_000_000).unwrap();
    let provider_id = provider.pubkey().to_bytes();
    bind_payout(ctx, &provider, provider_id, 1_000).unwrap();
    deposit(ctx, &consumer, provider_id, amount);
    (consumer, provider, provider_id)
}

#[test]
fn a_consumer_cannot_pay_itself_out_of_a_providers_vault() {
    let mut ctx = setup();
    initialize(&mut ctx);
    let (consumer, _provider, provider_id) = funded(&mut ctx, 500_000);
    let own = consumer.pubkey().to_bytes();

    // A throwaway provider the consumer controls, bound to the consumer's own address: it reaches
    // only its own vault (here 1_000), never the honest provider's.
    let throwaway = Keypair::new();
    let throwaway_id = throwaway.pubkey().to_bytes();
    bind_payout(&mut ctx, &throwaway, own, 1_000).unwrap();
    deposit(&mut ctx, &consumer, throwaway_id, 1_000);
    assert!(settle(&mut ctx, &consumer, throwaway_id, 500_000, "qwen2.5:7b").is_err(), "beyond its own vault");
    settle(&mut ctx, &consumer, throwaway_id, 1_000, "qwen2.5:7b").expect("its own 1_000 only");

    assert_eq!(vault_state(&ctx, &consumer, provider_id).balance, 500_000, "the honest vault is untouched");
}

#[test]
fn an_exit_needs_a_request_and_a_window_settles_cannot_extend() {
    let mut ctx = setup();
    initialize(&mut ctx);
    let (consumer, _provider, provider_id) = funded(&mut ctx, 500_000);

    assert!(withdraw(&mut ctx, &consumer, provider_id, 1).is_err(), "no exit requested");
    request_exit(&mut ctx, &consumer, provider_id).unwrap();
    advance(&mut ctx, 86_400 - 10);
    assert!(withdraw(&mut ctx, &consumer, provider_id, 1).is_err(), "window not over");
    // The provider settles what it is owed inside the window; that doesn't move the deadline.
    settle(&mut ctx, &consumer, provider_id, 100_000, "qwen2.5:7b").unwrap();
    advance(&mut ctx, 10);
    withdraw(&mut ctx, &consumer, provider_id, 400_000).expect("exit on time");
    assert_eq!(vault_state(&ctx, &consumer, provider_id).balance, 0);
}

#[test]
fn a_deposit_cancels_a_pending_exit() {
    let mut ctx = setup();
    initialize(&mut ctx);
    let (consumer, _provider, provider_id) = funded(&mut ctx, 500_000);
    request_exit(&mut ctx, &consumer, provider_id).unwrap();
    deposit(&mut ctx, &consumer, provider_id, 1);
    advance(&mut ctx, 86_400 + 1);
    assert!(withdraw(&mut ctx, &consumer, provider_id, 1).is_err());
}

#[test]
fn a_pause_does_not_run_out_the_providers_window() {
    let mut ctx = setup();
    initialize(&mut ctx);
    let (consumer, _provider, provider_id) = funded(&mut ctx, 500_000);
    request_exit(&mut ctx, &consumer, provider_id).unwrap();
    set_paused(&mut ctx, true).unwrap();
    advance(&mut ctx, 86_400 + 1);
    assert!(withdraw(&mut ctx, &consumer, provider_id, 1).is_err(), "no withdrawals while paused");
    set_paused(&mut ctx, false).unwrap();
    assert!(withdraw(&mut ctx, &consumer, provider_id, 1).is_err(), "a full window again after the unpause");
    advance(&mut ctx, 86_400);
    withdraw(&mut ctx, &consumer, provider_id, 1).expect("then it opens");
}

#[test]
fn only_the_upgrade_authority_initializes_and_the_refund_delay_is_capped() {
    let mut ctx = setup();
    let stranger = Keypair::new();
    ctx.svm.airdrop(&stranger.pubkey(), 1_000_000_000).unwrap();
    assert!(try_initialize(&mut ctx, &stranger, 86_400).is_err(), "a front-runner can't initialize");
    let admin = ctx.admin.insecure_clone();
    assert!(
        try_initialize(&mut ctx, &admin, openhydra_settlement::MAX_REFUND_DELAY_SECS + 1).is_err(),
        "a refund delay past the cap"
    );
    try_initialize(&mut ctx, &admin, openhydra_settlement::MAX_REFUND_DELAY_SECS).expect("the authority, within the cap");
}

#[test]
fn an_expired_claim_or_one_for_another_deployment_does_not_settle() {
    let mut ctx = setup();
    initialize(&mut ctx);
    let (consumer, _provider, provider_id) = funded(&mut ctx, 500_000);
    // Expired.
    assert!(settle_until(&mut ctx, &consumer, provider_id, 100_000, "m", T0 as u64 - 1).is_err());
    // Signed for another cluster (say devnet) with this program id.
    let c = claim(&consumer, provider_id, 100_000, EXPIRES, "m");
    let devnet = settle_claim_preimage(&ctx.program_id.to_bytes(), &[0x66; 32], &c.provider, &c.consumer, 100_000, EXPIRES, "m");
    let ixs = settle_ixs(&ctx, &consumer, &devnet, c);
    let payer = ctx.payer.insecure_clone();
    assert!(send(&mut ctx, &ixs, &[&payer]).is_err(), "another cluster's claim");
    // Still valid right up to its expiry.
    advance(&mut ctx, 1_000);
    settle_until(&mut ctx, &consumer, provider_id, 100_000, "m", T0 as u64 + 1_000).expect("not yet expired");
}

fn config_state(ctx: &Ctx) -> openhydra_settlement::Config {
    let acct = ctx.svm.get_account(&ctx.config).unwrap();
    openhydra_settlement::Config::try_deserialize(&mut &acct.data[..]).unwrap()
}

fn withdraw_fees(ctx: &mut Ctx, by: &Keypair, amount: u64) -> Result<(), String> {
    let ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::WithdrawFees { fee_authority: by.pubkey(), config: ctx.config }
            .to_account_metas(None),
        data: openhydra_settlement::instruction::WithdrawFees { amount }.data(),
    };
    let by = by.insecure_clone();
    send(ctx, &[ix], &[&by])
}

#[test]
fn fees_accrue_for_the_fee_authority_to_withdraw() {
    let mut ctx = setup();
    initialize(&mut ctx);
    let (consumer, _provider, provider_id) = funded(&mut ctx, 500_000);
    settle(&mut ctx, &consumer, provider_id, 100_000, "m").unwrap();
    assert_eq!(config_state(&ctx).accrued_fees, 4_000, "4% of the delta");
    let admin = ctx.admin.insecure_clone();
    assert!(withdraw_fees(&mut ctx, &admin, 1).is_err(), "only the fee authority");
    let fa = ctx.fee_authority.insecure_clone();
    assert!(withdraw_fees(&mut ctx, &fa, 4_001).is_err(), "no more than accrued");
    withdraw_fees(&mut ctx, &fa, 4_000).expect("the fee authority takes its fees");
    assert_eq!(config_state(&ctx).accrued_fees, 0);
}
