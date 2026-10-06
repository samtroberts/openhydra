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
        settlement.deposit(PROVIDER_ID, providerPayout, 500_000); // fund the channel with 0.5 USDC
    }

    function _claimTo(bytes32 provider, address payout, uint256 cumulative)
        internal
        pure
        returns (OpenHydraSettlement.SettlementClaim memory)
    {
        return OpenHydraSettlement.SettlementClaim({
            provider: provider,
            consumer: CONSUMER_ID,
            payoutAddress: payout,
            cumulativeAmount: cumulative,
            modelId: MODEL
        });
    }

    function _claim(uint256 cumulative) internal view returns (OpenHydraSettlement.SettlementClaim memory) {
        return _claimTo(PROVIDER_ID, providerPayout, cumulative);
    }

    function _sign(uint256 pk, OpenHydraSettlement.SettlementClaim memory claim) internal view returns (bytes memory) {
        bytes32 digest = settlement.digestOf(claim);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(pk, digest);
        return abi.encodePacked(r, s, v);
    }

    function _balance(bytes32 provider) internal view returns (uint256 bal) {
        (, bal,) = settlement.channels(consumer, provider);
    }

    function _payout(bytes32 provider) internal view returns (address payout) {
        (payout,,) = settlement.channels(consumer, provider);
    }

    function test_deposit_opens_a_channel_to_the_provider() public view {
        assertEq(_balance(PROVIDER_ID), 500_000);
        assertEq(_payout(PROVIDER_ID), providerPayout);
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
        settlement.settle(claim, _sign(consumerPk, claim));

        // delta = 100_000; fee = 100_000*400/10000 = 4_000; providerNet = 96_000.
        assertEq(usdc.balanceOf(providerPayout), 96_000, "provider gets 0.96");
        assertEq(usdc.balanceOf(feeRecipient), 4_000, "platform gets 0.04");
        assertEq(_balance(PROVIDER_ID), 400_000, "channel debited by delta");
        assertEq(settlement.settled(consumer, PROVIDER_ID, keccak256(bytes(MODEL))), 100_000, "high-water advanced");
    }

    function test_settle_pays_only_the_delta_on_a_higher_cumulative() public {
        OpenHydraSettlement.SettlementClaim memory c1 = _claim(100_000);
        settlement.settle(c1, _sign(consumerPk, c1));
        OpenHydraSettlement.SettlementClaim memory c2 = _claim(250_000);
        settlement.settle(c2, _sign(consumerPk, c2));

        // Cumulative fee: floor(250000*0.04)=10_000; provider 240_000. Channel 500k-250k=250k.
        assertEq(usdc.balanceOf(providerPayout), 240_000);
        assertEq(usdc.balanceOf(feeRecipient), 10_000);
        assertEq(_balance(PROVIDER_ID), 250_000);
    }

    function test_replayed_or_stale_cumulative_reverts() public {
        OpenHydraSettlement.SettlementClaim memory claim = _claim(100_000);
        bytes memory sig = _sign(consumerPk, claim);
        settlement.settle(claim, sig);
        vm.expectRevert(OpenHydraSettlement.StaleOrReplayedClaim.selector);
        settlement.settle(claim, sig);
        // Sign BEFORE expectRevert — `_sign` makes an external `digestOf` call that would consume it.
        OpenHydraSettlement.SettlementClaim memory lower = _claim(50_000);
        bytes memory lowerSig = _sign(consumerPk, lower);
        vm.expectRevert(OpenHydraSettlement.StaleOrReplayedClaim.selector);
        settlement.settle(lower, lowerSig);
    }

    function test_insufficient_escrow_reverts() public {
        OpenHydraSettlement.SettlementClaim memory claim = _claim(600_000);
        bytes memory sig = _sign(consumerPk, claim);
        vm.expectRevert(OpenHydraSettlement.InsufficientEscrow.selector);
        settlement.settle(claim, sig);
    }

    function test_a_signature_from_another_wallet_cannot_spend_the_consumers_channel() public {
        // Recovers to the attacker's wallet, which has no channel → NoChannel; the victim's is untouched.
        OpenHydraSettlement.SettlementClaim memory claim = _claim(100_000);
        bytes memory sig = _sign(0xBADA55, claim);
        vm.expectRevert(OpenHydraSettlement.NoChannel.selector);
        settlement.settle(claim, sig);
        assertEq(_balance(PROVIDER_ID), 500_000, "victim channel untouched");
    }

    // ── review fix: a new payout address doesn't get the running total paid again ──

    function test_moving_the_payout_address_does_not_pay_the_cumulative_again() public {
        OpenHydraSettlement.SettlementClaim memory c1 = _claim(100_000);
        settlement.settle(c1, _sign(consumerPk, c1));

        address rotated = address(0xC0C0);
        vm.prank(providerPayout);
        settlement.setPayout(consumer, PROVIDER_ID, rotated);

        // The old cumulative to the new address is stale: it was already paid.
        OpenHydraSettlement.SettlementClaim memory replay = _claimTo(PROVIDER_ID, rotated, 100_000);
        bytes memory replaySig = _sign(consumerPk, replay);
        vm.expectRevert(OpenHydraSettlement.StaleOrReplayedClaim.selector);
        settlement.settle(replay, replaySig);

        // The next voucher pays only its delta, to the new address.
        OpenHydraSettlement.SettlementClaim memory c2 = _claimTo(PROVIDER_ID, rotated, 150_000);
        settlement.settle(c2, _sign(consumerPk, c2));
        assertEq(usdc.balanceOf(rotated), 48_000, "50_000 delta, 0.96 of it");
        assertEq(usdc.balanceOf(providerPayout), 96_000, "the old address keeps only its first payment");
        assertEq(_balance(PROVIDER_ID), 350_000);
        // And a claim still naming the old address is refused.
        OpenHydraSettlement.SettlementClaim memory old = _claim(200_000);
        bytes memory oldSig = _sign(consumerPk, old);
        vm.expectRevert(OpenHydraSettlement.PayoutMismatch.selector);
        settlement.settle(old, oldSig);
    }

    // ── review fix: a consumer can't pay itself out of escrow committed to a provider ──

    function test_a_consumer_cannot_pay_itself_from_a_providers_channel() public {
        OpenHydraSettlement.SettlementClaim memory selfPay = _claimTo(PROVIDER_ID, consumer, 500_000);
        bytes memory sig = _sign(consumerPk, selfPay);
        vm.expectRevert(OpenHydraSettlement.PayoutMismatch.selector);
        settlement.settle(selfPay, sig);
        assertEq(_balance(PROVIDER_ID), 500_000);
        assertEq(usdc.balanceOf(consumer), 500_000, "no escrow came back");
    }

    function test_a_throwaway_provider_only_reaches_its_own_channel() public {
        // The consumer opens a channel to a provider id it controls, paying itself: only what it put
        // in THAT channel is reachable, the honest provider's channel is not.
        bytes32 throwaway = bytes32(uint256(0xDEAD));
        vm.prank(consumer);
        settlement.deposit(throwaway, consumer, 1_000);
        OpenHydraSettlement.SettlementClaim memory drain = _claimTo(throwaway, consumer, 500_000);
        bytes memory sig = _sign(consumerPk, drain);
        vm.expectRevert(OpenHydraSettlement.InsufficientEscrow.selector);
        settlement.settle(drain, sig);
        assertEq(_balance(PROVIDER_ID), 500_000, "the honest provider's channel is untouched");
    }

    function test_only_the_payout_address_can_move_a_channel() public {
        // The consumer can't redirect it, by deposit or by setPayout.
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.PayoutMismatch.selector);
        settlement.deposit(PROVIDER_ID, consumer, 1);
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.NotPayout.selector);
        settlement.setPayout(consumer, PROVIDER_ID, consumer);
        vm.prank(providerPayout);
        vm.expectRevert(OpenHydraSettlement.ZeroPayout.selector);
        settlement.setPayout(consumer, PROVIDER_ID, address(0));
    }

    function test_settling_without_a_channel_reverts() public {
        OpenHydraSettlement.SettlementClaim memory claim = _claimTo(bytes32(uint256(0x9999)), providerPayout, 1);
        bytes memory sig = _sign(consumerPk, claim);
        vm.expectRevert(OpenHydraSettlement.NoChannel.selector);
        settlement.settle(claim, sig);
    }

    // ── exit: request, then a fixed delay settles can't extend ──

    function test_withdraw_needs_an_exit_request_then_the_delay() public {
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.RefundLocked.selector);
        settlement.withdraw(PROVIDER_ID, 1); // no exit requested

        vm.prank(consumer);
        settlement.requestExit(PROVIDER_ID);
        vm.warp(block.timestamp + REFUND_DELAY - 1);
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.RefundLocked.selector);
        settlement.withdraw(PROVIDER_ID, 1);

        vm.warp(block.timestamp + 1);
        uint256 before = usdc.balanceOf(consumer);
        vm.prank(consumer);
        settlement.withdraw(PROVIDER_ID, 200_000);
        assertEq(usdc.balanceOf(consumer), before + 200_000);
        assertEq(_balance(PROVIDER_ID), 300_000);
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.WithdrawTooLarge.selector);
        settlement.withdraw(PROVIDER_ID, 300_001);
    }

    function test_settles_during_the_exit_window_do_not_extend_it() public {
        vm.prank(consumer);
        settlement.requestExit(PROVIDER_ID);
        uint256 t0 = block.timestamp;
        // The provider settles what it is owed inside the window…
        vm.warp(t0 + REFUND_DELAY - 10);
        OpenHydraSettlement.SettlementClaim memory claim = _claim(100_000);
        settlement.settle(claim, _sign(consumerPk, claim));
        // …and the consumer still exits on time with the rest.
        vm.warp(t0 + REFUND_DELAY);
        vm.prank(consumer);
        settlement.withdraw(PROVIDER_ID, 400_000);
        assertEq(_balance(PROVIDER_ID), 0);
    }

    function test_a_deposit_cancels_a_pending_exit() public {
        vm.prank(consumer);
        settlement.requestExit(PROVIDER_ID);
        vm.prank(consumer);
        settlement.deposit(PROVIDER_ID, providerPayout, 1);
        vm.warp(block.timestamp + REFUND_DELAY + 1);
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.RefundLocked.selector);
        settlement.withdraw(PROVIDER_ID, 1);
    }

    function test_a_pause_does_not_run_out_the_providers_window() public {
        vm.prank(consumer);
        settlement.requestExit(PROVIDER_ID);
        vm.prank(admin);
        settlement.setPaused(true);
        // No settles and no withdrawals while paused.
        vm.warp(block.timestamp + REFUND_DELAY + 1);
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.IsPaused.selector);
        settlement.withdraw(PROVIDER_ID, 1);
        // After the unpause, the provider gets a full window again.
        vm.prank(admin);
        settlement.setPaused(false);
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.RefundLocked.selector);
        settlement.withdraw(PROVIDER_ID, 1);
        vm.warp(block.timestamp + REFUND_DELAY);
        vm.prank(consumer);
        settlement.withdraw(PROVIDER_ID, 1);
    }

    function test_pause_blocks_settle_and_deposit() public {
        OpenHydraSettlement.SettlementClaim memory claim = _claim(100_000);
        bytes memory sig = _sign(consumerPk, claim); // sign before pausing / expectRevert
        vm.prank(admin);
        settlement.setPaused(true);
        vm.expectRevert(OpenHydraSettlement.IsPaused.selector);
        settlement.settle(claim, sig);
        vm.prank(consumer);
        vm.expectRevert(OpenHydraSettlement.IsPaused.selector);
        settlement.deposit(PROVIDER_ID, providerPayout, 1);
    }

    function test_only_admin_can_pause() public {
        vm.expectRevert(OpenHydraSettlement.NotAdmin.selector);
        settlement.setPaused(true);
    }

    function test_domain_separator_binds_chain_and_contract() public view {
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
