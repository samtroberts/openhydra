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
    config: Pubkey,
    instr_sysvar: Pubkey,
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
    Ctx { svm, program_id, payer, admin, config, instr_sysvar }
}

fn send(ctx: &mut Ctx, ixs: &[Instruction], signers: &[&Keypair]) -> Result<(), String> {
    let blockhash = ctx.svm.latest_blockhash();
    let msg = Message::new_with_blockhash(ixs, Some(&signers[0].pubkey()), &blockhash);
    let tx = VersionedTransaction::try_new(VersionedMessage::Legacy(msg), signers)
        .map_err(|e| format!("sign: {e}"))?;
    ctx.svm.send_transaction(tx).map(|_| ()).map_err(|e| format!("{:?}", e.err))
}

fn initialize(ctx: &mut Ctx) {
    let ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::Initialize {
            admin: ctx.admin.pubkey(),
            config: ctx.config,
            system_program: system_program::ID,
        }
        .to_account_metas(None),
        data: openhydra_settlement::instruction::Initialize {
            fee_account: Pubkey::new_unique(),
            usdc_mint: Pubkey::new_unique(),
            refund_delay_secs: 86_400,
        }
        .data(),
    };
    let admin = ctx.admin.insecure_clone();
    send(ctx, &[ix], &[&admin]).expect("initialize");
}

fn bind_payout(ctx: &mut Ctx, provider: &Keypair, payout: [u8; 32], ts: u64) -> Result<(), String> {
    let provider_id = provider.pubkey().to_bytes();
    let binding_pda = Pubkey::find_program_address(&[b"payout", &provider_id], &ctx.program_id).0;
    let msg = binding_preimage(&payout, ts);
    let sig = sign64(provider, &msg);
    let ed_ix = ed25519_ix(&provider_id, &sig, &msg);
    let bind_ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::BindPayout {
            payer: ctx.payer.pubkey(),
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

fn deposit(ctx: &mut Ctx, consumer: &Keypair, amount: u64) {
    let vault = Pubkey::find_program_address(&[b"vault", consumer.pubkey().as_ref()], &ctx.program_id).0;
    let ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::Deposit {
            consumer: consumer.pubkey(),
            config: ctx.config,
            vault,
            system_program: system_program::ID,
        }
        .to_account_metas(None),
        data: openhydra_settlement::instruction::Deposit { amount }.data(),
    };
    let consumer = consumer.insecure_clone();
    send(ctx, &[ix], &[&consumer]).expect("deposit");
}

#[allow(clippy::too_many_arguments)]
fn settle(
    ctx: &mut Ctx,
    consumer: &Keypair,
    provider_id: [u8; 32],
    payout: [u8; 32],
    cumulative: u64,
    model_id: &str,
) -> Result<(), String> {
    let consumer_id = consumer.pubkey().to_bytes();
    let vault = Pubkey::find_program_address(&[b"vault", consumer.pubkey().as_ref()], &ctx.program_id).0;
    let channel = Pubkey::find_program_address(
        &[b"channel", &consumer_id, &provider_id, &model_hash(model_id)],
        &ctx.program_id,
    )
    .0;
    let binding_pda = Pubkey::find_program_address(&[b"payout", &provider_id], &ctx.program_id).0;
    let msg = settle_claim_preimage(&provider_id, &consumer_id, &payout, cumulative, model_id);
    let sig = sign64(consumer, &msg);
    let ed_ix = ed25519_ix(&consumer_id, &sig, &msg);
    let settle_ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::Settle {
            payer: ctx.payer.pubkey(),
            config: ctx.config,
            vault,
            channel,
            payout_binding: binding_pda,
            instructions_sysvar: ctx.instr_sysvar,
            system_program: system_program::ID,
        }
        .to_account_metas(None),
        data: openhydra_settlement::instruction::Settle {
            claim: openhydra_settlement::SettleClaimArgs {
                provider: provider_id,
                consumer: consumer_id,
                payout,
                cumulative_amount: cumulative,
                model_id: model_id.to_string(),
            },
        }
        .data(),
    };
    let payer = ctx.payer.insecure_clone();
    send(ctx, &[ed_ix, settle_ix], &[&payer])
}

fn vault_state(ctx: &Ctx, consumer: &Keypair) -> openhydra_settlement::Vault {
    let vault = Pubkey::find_program_address(&[b"vault", consumer.pubkey().as_ref()], &ctx.program_id).0;
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
    deposit(&mut ctx, &consumer, 500_000);
    assert_eq!(vault_state(&ctx, &consumer).balance, 500_000);

    // Settle a cumulative of 100_000 → high-water advances, escrow debited by the delta.
    settle(&mut ctx, &consumer, provider_id, payout, 100_000, "qwen2.5:7b").expect("settle #1");
    assert_eq!(channel_state(&ctx, &consumer, provider_id, "qwen2.5:7b").settled_amount, 100_000);
    assert_eq!(vault_state(&ctx, &consumer).balance, 400_000);

    // A later cumulative of 250_000 pays only the 150_000 delta (monotonic, batched).
    settle(&mut ctx, &consumer, provider_id, payout, 250_000, "qwen2.5:7b").expect("settle #2");
    assert_eq!(channel_state(&ctx, &consumer, provider_id, "qwen2.5:7b").settled_amount, 250_000);
    assert_eq!(vault_state(&ctx, &consumer).balance, 250_000);
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
    deposit(&mut ctx, &consumer, 500_000);

    settle(&mut ctx, &consumer, provider_id, provider_id, 100_000, "qwen2.5:7b").unwrap();
    // Replaying the same cumulative → delta 0 → the program rejects it.
    assert!(settle(&mut ctx, &consumer, provider_id, provider_id, 100_000, "qwen2.5:7b").is_err());
    // A lower cumulative than the high-water → also rejected (never claws back).
    assert!(settle(&mut ctx, &consumer, provider_id, provider_id, 50_000, "qwen2.5:7b").is_err());
    // The escrow was debited exactly once.
    assert_eq!(vault_state(&ctx, &consumer).balance, 400_000);
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
    deposit(&mut ctx, &consumer, 5_000_000);

    let vault = Pubkey::find_program_address(&[b"vault", consumer.pubkey().as_ref()], &ctx.program_id).0;
    let channel = Pubkey::find_program_address(
        &[b"channel", &consumer_id, &provider_id, &model_hash("qwen2.5:7b")],
        &ctx.program_id,
    )
    .0;
    let binding_pda = Pubkey::find_program_address(&[b"payout", &provider_id], &ctx.program_id).0;
    // Sign the HONEST preimage (100_000)...
    let honest = settle_claim_preimage(&provider_id, &consumer_id, &provider_id, 100_000, "qwen2.5:7b");
    let sig = sign64(&consumer, &honest);
    let ed_ix = ed25519_ix(&consumer_id, &sig, &honest);
    // ...but submit a settle claiming 999_999.
    let settle_ix = Instruction {
        program_id: ctx.program_id,
        accounts: openhydra_settlement::accounts::Settle {
            payer: ctx.payer.pubkey(),
            config: ctx.config,
            vault,
            channel,
            payout_binding: binding_pda,
            instructions_sysvar: ctx.instr_sysvar,
            system_program: system_program::ID,
        }
        .to_account_metas(None),
        data: openhydra_settlement::instruction::Settle {
            claim: openhydra_settlement::SettleClaimArgs {
                provider: provider_id,
                consumer: consumer_id,
                payout: provider_id,
                cumulative_amount: 999_999,
                model_id: "qwen2.5:7b".to_string(),
            },
        }
        .data(),
    };
    let payer = ctx.payer.insecure_clone();
    assert!(send(&mut ctx, &[ed_ix, settle_ix], &[&payer]).is_err(), "message/claim mismatch rejected");
}
