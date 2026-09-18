// SPDX-License-Identifier: Apache-2.0
pragma solidity ^0.8.24;

import {IERC20} from "openzeppelin-contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "openzeppelin-contracts/token/ERC20/utils/SafeERC20.sol";
import {ECDSA} from "openzeppelin-contracts/utils/cryptography/ECDSA.sol";
import {ReentrancyGuard} from "openzeppelin-contracts/utils/ReentrancyGuard.sol";

/// @title OpenHydra Base Settlement — DRAFT (BASE_SETTLEMENT_CONTRACT_SPEC_v1)
/// @notice The Base/EVM half of OpenHydra's crypto rail: a non-custodial escrow that settles the
///         authenticity-gated *voucher* in USDC, taking the 4% protocol fee on-chain. The chain
///         settles exactly what the consumer's EVM wallet signed (an EIP-712 `SettlementClaim`)
///         against its own escrow — it does NOT re-verify authenticity (that is baked into the
///         voucher's rate-capped `cumulativeAmount` off-chain).
///
/// @dev  ⚠️ STATUS: DRAFT / ARCHITECTURE — NOT AUDITED, NOT DEPLOYABLE TO MAINNET. Moving real USDC is
///       hard-gated on a non-custodial legal opinion, a security audit, MiCA/FIU-IND geofencing, and a
///       payer entity (see the spec). This exists to be built + localnet-tested and audited, not to
///       ship. The EIP-712 digest here is byte-for-byte `settlement::SettlementClaim::eip712_digest`
///       and `recover_evm_signer` in the Rust off-chain code, so a signature produced/verified
///       off-chain settles here.
///
/// Non-custodial invariant: no path lets any key remove escrow except `settle` (the fixed 0.96/0.04
/// split to the claim's bound payout + the fee recipient) and `withdraw` (back to the depositing
/// consumer). There is no owner-drains-escrow function. The fee is fixed at init.
contract OpenHydraSettlement is ReentrancyGuard {
    using SafeERC20 for IERC20;

    // ── EIP-712 constants (PINNED — must match settlement.rs byte-for-byte) ──
    // settlement.rs builds the domain separator manually from these exact strings.
    bytes32 private constant DOMAIN_TYPEHASH =
        keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)");
    bytes32 private constant DOMAIN_NAME_HASH = keccak256("OpenHydra Settlement");
    bytes32 private constant DOMAIN_VERSION_HASH = keccak256("1");
    bytes32 private constant CLAIM_TYPEHASH = keccak256(
        "SettlementClaim(bytes32 provider,bytes32 consumer,address payoutAddress,uint256 cumulativeAmount,string modelId)"
    );

    /// The 4% protocol fee in basis points — matches `payout::PROTOCOL_FEE_BPS`.
    uint16 public constant EXPECTED_FEE_BPS = 400;

    // ── Config (set once at init; the fee can never be changed to divert funds) ──
    IERC20 public immutable usdc;
    address public immutable feeRecipient;
    uint16 public immutable feeBps;
    uint64 public immutable refundDelay; // seconds of consumer inactivity before an unconditional withdraw
    address public admin; // pause only — CANNOT touch funds
    bool public paused;

    // ── State ──
    /// Consumer EVM address → deposited USDC balance (the x402 `upto` ceiling).
    mapping(address => uint256) public escrow;
    /// consumerEvm → providerPayout(EVM) → modelHash → cumulative USDC already settled (monotonic high-water).
    mapping(address => mapping(address => mapping(bytes32 => uint256))) public settled;
    /// Consumer EVM address → last deposit/settle timestamp (drives the refund inactivity window).
    mapping(address => uint64) public lastActivity;

    /// The claim the consumer's EVM wallet signs (EIP-712). `provider`/`consumer` are the Ed25519
    /// channel ids (the link to the off-chain receipt); `payoutAddress` is the provider's bound EVM
    /// address (the payee); the authorizing signer recovered from `sig` must own the escrow drawn from.
    struct SettlementClaim {
        bytes32 provider;
        bytes32 consumer;
        address payoutAddress;
        uint256 cumulativeAmount;
        string modelId;
    }

    event Deposited(address indexed consumer, uint256 amount);
    event Settled(
        address indexed consumer,
        address indexed payout,
        bytes32 modelHash,
        uint256 delta,
        uint256 providerNet,
        uint256 fee
    );
    event Withdrawn(address indexed consumer, uint256 amount);
    event Paused(bool paused);

    error NotAdmin();
    error IsPaused();
    error BadFeeBps();
    error ZeroSigner();
    error SignerNotEscrowOwner();
    error StaleOrReplayedClaim(); // delta == 0 (or would underflow — a lower cumulative)
    error InsufficientEscrow();
    error WithdrawTooLarge();
    error RefundLocked();

    constructor(
        IERC20 _usdc,
        address _feeRecipient,
        uint16 _feeBps,
        uint64 _refundDelay,
        address _admin
    ) {
        if (_feeBps != EXPECTED_FEE_BPS) revert BadFeeBps(); // the agreed 4%, fixed at init
        usdc = _usdc;
        feeRecipient = _feeRecipient;
        feeBps = _feeBps;
        refundDelay = _refundDelay;
        admin = _admin;
    }

    // ── EIP-712 (mirrors settlement::SettlementClaim::eip712_digest exactly) ──

    /// The domain separator, computed the same way settlement.rs does (chainId + this contract bind the
    /// signature to this deployment, blocking cross-chain / sibling-contract replay).
    function domainSeparator() public view returns (bytes32) {
        return keccak256(
            abi.encode(
                DOMAIN_TYPEHASH,
                DOMAIN_NAME_HASH,
                DOMAIN_VERSION_HASH,
                block.chainid,
                address(this)
            )
        );
    }

    /// `keccak256(0x1901 ‖ domainSeparator ‖ hashStruct(claim))` — byte-for-byte `eip712_digest`.
    function digestOf(SettlementClaim calldata claim) public view returns (bytes32) {
        bytes32 hashStruct = keccak256(
            abi.encode(
                CLAIM_TYPEHASH,
                claim.provider,
                claim.consumer,
                claim.payoutAddress,
                claim.cumulativeAmount,
                keccak256(bytes(claim.modelId))
            )
        );
        return keccak256(abi.encodePacked(hex"1901", domainSeparator(), hashStruct));
    }

    // ── Instructions ──

    /// Consumer funds its escrow (the x402 `upto` ceiling). DRAFT: a plain `transferFrom` pull
    /// (requires a prior `approve`). Production uses USDC EIP-3009 `receiveWithAuthorization` / Permit2
    /// for a gasless single-tx fund (spec §4.2) — an additive change, same escrow effect.
    function deposit(uint256 amount) external nonReentrant {
        if (paused) revert IsPaused();
        usdc.safeTransferFrom(msg.sender, address(this), amount);
        escrow[msg.sender] += amount;
        lastActivity[msg.sender] = uint64(block.timestamp);
        emit Deposited(msg.sender, amount);
    }

    /// Settle a co-signed voucher: anyone (provider or facilitator) may submit. The consumer's EVM
    /// signature over the EIP-712 claim authorizes drawing the delta from *its* escrow; the fixed
    /// 0.96/0.04 split goes to the bound payout + the fee recipient. Checks-effects-interactions.
    function settle(SettlementClaim calldata claim, bytes calldata sig) external nonReentrant {
        if (paused) revert IsPaused();

        // Recover the authorizing EVM signer. OZ ECDSA.recover enforces EIP-2 low-`s` and rejects the
        // zero address / malformed sigs — the on-chain mirror of `recover_evm_signer` (which also
        // rejects high-`s`, review F5), so off-chain and on-chain agree on validity.
        address signer = ECDSA.recover(digestOf(claim), sig);
        if (signer == address(0)) revert ZeroSigner();

        // The signer must be the escrow being spent. (`claim.consumer` is the Ed25519 id for the
        // off-chain link; the on-chain authority is the recovered EVM wallet that owns the escrow.)
        address consumerEvm = signer;

        bytes32 modelHash = keccak256(bytes(claim.modelId));
        uint256 prior = settled[consumerEvm][claim.payoutAddress][modelHash];
        // Monotonic: a replayed or stale (<= prior) cumulative is a no-op → revert. Also blocks
        // underflow (a lower cumulative can never claw back).
        if (claim.cumulativeAmount <= prior) revert StaleOrReplayedClaim();
        uint256 delta = claim.cumulativeAmount - prior;

        if (escrow[consumerEvm] < delta) revert InsufficientEscrow();

        // Fixed split, floor the fee (dust to the provider) — matches payout::protocol_fee.
        uint256 fee = (delta * feeBps) / 10_000;
        uint256 providerNet = delta - fee;

        // Effects before interactions (CEI): advance the high-water mark + debit escrow first.
        settled[consumerEvm][claim.payoutAddress][modelHash] = claim.cumulativeAmount;
        escrow[consumerEvm] -= delta;
        lastActivity[consumerEvm] = uint64(block.timestamp); // resets the refund clock (activity)

        // Interactions.
        usdc.safeTransfer(claim.payoutAddress, providerNet);
        if (fee > 0) usdc.safeTransfer(feeRecipient, fee);

        emit Settled(consumerEvm, claim.payoutAddress, modelHash, delta, providerNet, fee);
    }

    /// Consumer reclaims unspent escrow → the non-custodial exit; deposits are never trapped.
    ///
    /// DRAFT guard: an unconditional withdraw is allowed only after `refundDelay` of inactivity, so a
    /// provider holding a signed-but-unsettled cumulative has a window to `settle` first (settling
    /// resets the clock). This is spec §7 option (a), the timelock — the strongest guarantee an EVM
    /// contract can enforce without visibility into un-settled off-chain claims. Option (b),
    /// `escrow − outstanding`, protects only *settled* channels here because an un-settled signed claim
    /// is invisible on-chain; the timelock covers the gap. Choose the final model at audit.
    function withdraw(uint256 amount) external nonReentrant {
        if (amount > escrow[msg.sender]) revert WithdrawTooLarge();
        if (block.timestamp < uint256(lastActivity[msg.sender]) + refundDelay) revert RefundLocked();
        escrow[msg.sender] -= amount;
        usdc.safeTransfer(msg.sender, amount);
        emit Withdrawn(msg.sender, amount);
    }

    // ── Admin (pause only — no fund authority) ──

    function setPaused(bool _paused) external {
        if (msg.sender != admin) revert NotAdmin();
        paused = _paused;
        emit Paused(_paused);
    }
}
