// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {ERC20} from "openzeppelin-contracts/token/ERC20/ERC20.sol";
import {OpenHydraSettlement} from "../src/OpenHydraSettlement.sol";

/// 6-decimal mock USDC for localnet tests.
contract MockUSDC is ERC20 {
    constructor() ERC20("USD Coin", "USDC") {}
    function decimals() public pure override returns (uint8) {
        return 6;
    }
    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }
}

contract OpenHydraSettlementTest is Test {
    OpenHydraSettlement settlement;
    MockUSDC usdc;

    address feeRecipient = address(0xFEE);
    address admin = address(0xAD11);
    uint64 constant REFUND_DELAY = 1 days;

    // Consumer EVM wallet — its private key signs the EIP-712 claim; its address owns the escrow.
    uint256 consumerPk = 0xA11CE;
    address consumer;
    // Provider payout EVM address (from an EvmPayoutBinding off-chain).
    address providerPayout = address(0xB0B);

    // Ed25519 channel ids (opaque bytes32 on-chain — the off-chain receipt link).
    bytes32 constant PROVIDER_ID = bytes32(uint256(0x1111));
    bytes32 constant CONSUMER_ID = bytes32(uint256(0x2222));
    string constant MODEL = "qwen2.5:7b";

    function setUp() public {
        consumer = vm.addr(consumerPk);
        usdc = new MockUSDC();
        settlement = new OpenHydraSettlement(usdc, feeRecipient, 400, REFUND_DELAY, admin);

        usdc.mint(consumer, 1_000_000); // 1 USDC
        vm.prank(consumer);
        usdc.approve(address(settlement), type(uint256).max);
        vm.prank(consumer);
        settlement.deposit(500_000); // fund escrow with 0.5 USDC
    }

    function _claim(uint256 cumulative) internal view returns (OpenHydraSettlement.SettlementClaim memory) {
        return OpenHydraSettlement.SettlementClaim({
            provider: PROVIDER_ID,
            consumer: CONSUMER_ID,
            payoutAddress: providerPayout,
            cumulativeAmount: cumulative,
            modelId: MODEL
        });
    }

    function _sign(uint256 pk, OpenHydraSettlement.SettlementClaim memory claim) internal view returns (bytes memory) {
        bytes32 digest = settlement.digestOf(claim);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(pk, digest);
        return abi.encodePacked(r, s, v);
    }

    function test_deposit_credits_escrow() public view {
        assertEq(settlement.escrow(consumer), 500_000);
    }

    function test_fee_is_fixed_at_four_percent() public view {
        assertEq(settlement.feeBps(), 400);
    }

    function test_constructor_rejects_wrong_fee() public {
        vm.expectRevert(OpenHydraSettlement.BadFeeBps.selector);
        new OpenHydraSettlement(usdc, feeRecipient, 500, REFUND_DELAY, admin);
    }

    function test_settle_splits_96_4_and_advances_high_water() public {
        OpenHydraSettlement.SettlementClaim memory claim = _claim(100_000); // 0.1 USDC cumulative
        bytes memory sig = _sign(consumerPk, claim);

        settlement.settle(claim, sig);

        // delta = 100_000; fee = 100_000*400/10000 = 4_000; providerNet = 96_000.
        assertEq(usdc.balanceOf(providerPayout), 96_000, "provider gets 0.96");
        assertEq(usdc.balanceOf(feeRecipient), 4_000, "platform gets 0.04");
        assertEq(settlement.escrow(consumer), 400_000, "escrow debited by delta");
        bytes32 modelHash = keccak256(bytes(MODEL));
        assertEq(settlement.settled(consumer, providerPayout, modelHash), 100_000, "high-water advanced");
    }

    function test_settle_pays_only_the_delta_on_a_higher_cumulative() public {
        // First settle at 100_000, then a later voucher at 250_000 pays only the 150_000 delta.
        OpenHydraSettlement.SettlementClaim memory c1 = _claim(100_000);
        settlement.settle(c1, _sign(consumerPk, c1));
        OpenHydraSettlement.SettlementClaim memory c2 = _claim(250_000);
        settlement.settle(c2, _sign(consumerPk, c2));

        // Cumulative fee: floor(250000*0.04)=10_000; provider 240_000. Escrow 500k-250k=250k.
        assertEq(usdc.balanceOf(providerPayout), 240_000);
        assertEq(usdc.balanceOf(feeRecipient), 10_000);
        assertEq(settlement.escrow(consumer), 250_000);
    }

    function test_replayed_or_stale_cumulative_reverts() public {
        OpenHydraSettlement.SettlementClaim memory claim = _claim(100_000);
        bytes memory sig = _sign(consumerPk, claim);
        settlement.settle(claim, sig);
        // Re-submitting the identical claim → delta 0 → revert (monotonic no-op).
        vm.expectRevert(OpenHydraSettlement.StaleOrReplayedClaim.selector);
        settlement.settle(claim, sig);
        // A lower cumulative than the high-water also reverts (never claws back). Sign BEFORE
        // expectRevert — `_sign` makes an external `digestOf` call that would otherwise consume it.
        OpenHydraSettlement.SettlementClaim memory lower = _claim(50_000);
        bytes memory lowerSig = _sign(consumerPk, lower);
        vm.expectRevert(OpenHydraSettlement.StaleOrReplayedClaim.selector);
        settlement.settle(lower, lowerSig);
    }

    function test_insufficient_escrow_reverts() public {
        // Escrow is 500_000; a cumulative delta above it must revert (the x402 ceiling is reached).
        OpenHydraSettlement.SettlementClaim memory claim = _claim(600_000);
        bytes memory sig = _sign(consumerPk, claim);
        vm.expectRevert(OpenHydraSettlement.InsufficientEscrow.selector);
        settlement.settle(claim, sig);
    }

    function test_a_signature_from_a_non_escrow_wallet_draws_its_own_escrow_not_the_victims() public {
        // A different wallet signs → recovers to THAT wallet, which has no escrow → InsufficientEscrow.
        // The victim consumer's escrow is untouched: only the escrow owner's own signature can spend it.
        uint256 attackerPk = 0xBADA55;
        OpenHydraSettlement.SettlementClaim memory claim = _claim(100_000);
        bytes memory sig = _sign(attackerPk, claim);
        vm.expectRevert(OpenHydraSettlement.InsufficientEscrow.selector);
        settlement.settle(claim, sig);
        assertEq(settlement.escrow(consumer), 500_000, "victim escrow untouched");
    }

    function test_withdraw_is_refund_delay_locked_then_allowed() public {
        // Immediately after deposit, the inactivity window has not elapsed → locked.
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.RefundLocked.selector);
        settlement.withdraw(100_000);

        // After refundDelay of inactivity, the consumer can reclaim unspent escrow.
        vm.warp(block.timestamp + REFUND_DELAY + 1);
        uint256 before = usdc.balanceOf(consumer);
        vm.prank(consumer);
        settlement.withdraw(200_000);
        assertEq(usdc.balanceOf(consumer), before + 200_000);
        assertEq(settlement.escrow(consumer), 300_000);
    }

    function test_settle_resets_the_refund_clock() public {
        // Warp near the window, then a settle resets lastActivity → withdraw is locked again.
        vm.warp(block.timestamp + REFUND_DELAY - 10);
        OpenHydraSettlement.SettlementClaim memory claim = _claim(100_000);
        settlement.settle(claim, _sign(consumerPk, claim));
        vm.warp(block.timestamp + 20); // 10s past the ORIGINAL window, but settle moved it
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.RefundLocked.selector);
        settlement.withdraw(1);
    }

    function test_pause_blocks_settle_and_deposit() public {
        OpenHydraSettlement.SettlementClaim memory claim = _claim(100_000);
        bytes memory sig = _sign(consumerPk, claim); // sign before pausing / expectRevert
        vm.prank(admin);
        settlement.setPaused(true);
        vm.expectRevert(OpenHydraSettlement.IsPaused.selector);
        settlement.settle(claim, sig);
    }

    function test_only_admin_can_pause() public {
        vm.expectRevert(OpenHydraSettlement.NotAdmin.selector);
        settlement.setPaused(true);
    }

    function test_domain_separator_binds_chain_and_contract() public view {
        // Sanity: the digest changes with the claim (field sensitivity), matching eip712_digest.
        bytes32 d1 = settlement.digestOf(_claim(100_000));
        bytes32 d2 = settlement.digestOf(_claim(100_001));
        assertTrue(d1 != d2, "cumulative is bound into the digest");
    }

    /// CROSS-LANGUAGE PARITY: reconstruct the EIP-712 digest for the SAME fixed claim + domain the Rust
    /// golden (`settlement.rs::eip712_digest_cross_language_golden`) pins, and assert the identical
    /// 32 bytes. This proves the Solidity `_hashTypedDataV4` and the off-chain `eip712_digest` agree
    /// byte-for-byte, so a signature made/verified off-chain (`recover_evm_signer`) settles here.
    /// (Uses fixed domain params, not block.chainid/address(this), so the vector is deployment-agnostic;
    /// `digestOf` uses the exact same formula sourced from the live chain/contract.)
    function test_eip712_digest_matches_rust_golden() public pure {
        bytes32 provider = bytes32(uint256(0x1111111111111111111111111111111111111111111111111111111111111111));
        bytes32 consumer = bytes32(uint256(0x2222222222222222222222222222222222222222222222222222222222222222));
        address payout = address(bytes20(hex"ABABABABABABABABABABABABABABABABABABABAB"));
        uint256 cumulative = 100_000;
        string memory modelId = "qwen2.5:7b";
        uint256 chainId = 8453;
        address verifyingContract = address(bytes20(hex"CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"));

        bytes32 domainSep = keccak256(
            abi.encode(
                keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                keccak256("OpenHydra Settlement"),
                keccak256("1"),
                chainId,
                verifyingContract
            )
        );
        bytes32 hashStruct = keccak256(
            abi.encode(
                keccak256("SettlementClaim(bytes32 provider,bytes32 consumer,address payoutAddress,uint256 cumulativeAmount,string modelId)"),
                provider,
                consumer,
                payout,
                cumulative,
                keccak256(bytes(modelId))
            )
        );
        bytes32 digest = keccak256(abi.encodePacked(hex"1901", domainSep, hashStruct));
        assertEq(
            digest,
            bytes32(0xa3cd9b3cb6dbd590c004a95113e457831526cff559e6707131e7f39c71e80f36),
            "EIP-712 digest must match the Rust settlement.rs golden byte-for-byte"
        );
    }
}
