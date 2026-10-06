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
/// split: the channel's payout, and accrued fees the fee recipient withdraws) and `withdraw` (back to the depositing
/// consumer, in the `EXIT_WINDOW` after an exit request and `refundDelay`). There is no
/// owner-drains-escrow function. The fee is fixed at init.
///
/// Escrow is ring-fenced per provider (a payment channel): a consumer funds `channels[consumer][provider]`
/// and that balance can only ever be paid to the channel's payout address. So a consumer can't pay
/// itself out of escrow it committed to a provider (by signing a claim to its own wallet, or to a
/// throwaway provider): a throwaway provider's channel holds only what the consumer put in it. What a
/// provider has been paid is tracked per (consumer, provider, model) — never per payout address — so
/// moving a payout address doesn't pay the running total again.
///
/// A provider relies on a channel only after checking, off-chain, that `channels[consumer][provider]
/// .payout` is its own payout address (the consumer picks it when it opens the channel; afterwards only
/// the current payout address can change it).
contract OpenHydraSettlement is ReentrancyGuard {
    using SafeERC20 for IERC20;

    // ── EIP-712 constants (PINNED — must match settlement.rs byte-for-byte) ──
    // settlement.rs builds the domain separator manually from these exact strings.
    bytes32 private constant DOMAIN_TYPEHASH =
        keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)");
    bytes32 private constant DOMAIN_NAME_HASH = keccak256("OpenHydra Settlement");
    /// "2": claim format v6 (an expiry; no payout address — a channel pays only its own payout).
    bytes32 private constant DOMAIN_VERSION_HASH = keccak256("2");
    bytes32 private constant CLAIM_TYPEHASH = keccak256(
        "SettlementClaim(bytes32 provider,bytes32 consumer,uint256 cumulativeAmount,uint256 expiresAt,string modelId)"
    );

    /// The 4% protocol fee in basis points — matches `payout::PROTOCOL_FEE_BPS`.
    uint16 public constant EXPECTED_FEE_BPS = 400;
    /// The longest refund delay a deployment may set: shorter than a claim's lifetime
    /// (`settlement::CLAIM_LIFETIME_SECS`, 30 days), so a provider's last claim outlives an exit window.
    uint64 public constant MAX_REFUND_DELAY = 14 days;
    /// The shortest refund delay: with none, a consumer could request an exit and withdraw in one
    /// transaction, ahead of the provider's settle.
    uint64 public constant MIN_REFUND_DELAY = 1 days;
    /// How much pausing can delay one exit, in total: time paused after an exit request extends its
    /// window (a pause stops settles), capped so the admin's pause can't lock escrow indefinitely.
    /// Also the most a pause extends a claim (see `claimGrace`).
    uint64 public constant MAX_PAUSE_EXTENSION = 30 days;
    /// How long an exit stays open once its delay has passed. One request opens one window, and a
    /// withdraw closes it: an exit requested once can't be kept armed and used, months later, in one
    /// transaction ahead of the provider's settle.
    uint64 public constant EXIT_WINDOW = 3 days;

    // ── Config (set once at init; the fee can never be changed to divert funds) ──
    IERC20 public immutable usdc;
    /// Takes the accrued protocol fees (`withdrawFees`), to any address it names: fees are pulled,
    /// not pushed on every settle, so a recipient the token issuer blacklists can't stop settlement.
    address public immutable feeRecipient;
    /// Protocol fees settled and not yet withdrawn.
    uint256 public accruedFees;
    uint16 public immutable feeBps;
    uint64 public immutable refundDelay; // seconds of consumer inactivity before an unconditional withdraw
    address public admin; // pause only — CANNOT touch funds
    bool public paused;
    /// When the current pause began, or the last one (0 = never paused).
    uint64 public pausedAt;
    /// When the last pause ended (meaningful once a pause has ended).
    uint64 public lastPauseEnd;
    /// Seconds paused in total over pauses that have ended: with `pausedAt`, how much a pause delays an
    /// exit.
    uint64 public pausedTotal;

    // ── State ──
    /// A consumer's escrow for one provider (the x402 `upto` ceiling for that provider).
    struct Channel {
        address payout; // the provider's payout address; set at open, changed only by itself
        uint256 balance;
        uint64 exitRequestedAt; // 0 = no exit pending
        uint64 exitPauseMark; // total paused time when the exit was requested
    }
    /// consumerEvm → provider (Ed25519 id) → channel.
    mapping(address => mapping(bytes32 => Channel)) public channels;
    /// consumerEvm → provider (Ed25519 id) → consumer (Ed25519 id) → modelHash → cumulative USDC already
    /// settled (monotonic high-water). Keyed by the provider, not its payout address: a new payout
    /// address continues from what was already paid. And by the consumer's Ed25519 id as well as its
    /// wallet, as the provider tracks it: two identities sharing a wallet (or one rotated) don't share
    /// a running total.
    mapping(address => mapping(bytes32 => mapping(bytes32 => mapping(bytes32 => uint256)))) public settled;

    /// The claim the consumer's EVM wallet signs (EIP-712, claim format v6). `provider`/`consumer` are
    /// the Ed25519 channel ids (the link to the off-chain receipt); the authorizing signer recovered
    /// from `sig` must own the escrow drawn from, which pays only the channel's payout. Settles until
    /// `expiresAt` (unix seconds).
    struct SettlementClaim {
        bytes32 provider;
        bytes32 consumer;
        uint256 cumulativeAmount;
        uint256 expiresAt;
        string modelId;
    }

    event Deposited(address indexed consumer, bytes32 indexed provider, address payout, uint256 amount);
    event Settled(
        address indexed consumer,
        bytes32 indexed provider,
        address payout,
        bytes32 modelHash,
        uint256 delta,
        uint256 providerNet,
        uint256 fee
    );
    event PayoutChanged(address indexed consumer, bytes32 indexed provider, address payout);
    event ExitRequested(address indexed consumer, bytes32 indexed provider, uint64 withdrawableAt);
    event Withdrawn(address indexed consumer, bytes32 indexed provider, uint256 amount);
    event Paused(bool paused);
    event FeesWithdrawn(address indexed to, uint256 amount);

    error NotAdmin();
    error IsPaused();
    error BadFeeBps();
    error ZeroSigner();
    error SignerNotEscrowOwner();
    error StaleOrReplayedClaim(); // delta == 0 (or would underflow — a lower cumulative)
    error InsufficientEscrow();
    error WithdrawTooLarge();
    error RefundLocked();
    error ZeroPayout();
    error NoChannel();
    error PayoutMismatch(); // a claim or deposit naming another payout than the channel's
    error NotPayout(); // only a channel's payout address may move it
    error ClaimExpired();
    error ExitLapsed(); // the exit window closed: request a new exit
    error BadRefundDelay();
    error NotFeeRecipient();

    constructor(
        IERC20 _usdc,
        address _feeRecipient,
        uint16 _feeBps,
        uint64 _refundDelay,
        address _admin
    ) {
        if (_feeBps != EXPECTED_FEE_BPS) revert BadFeeBps(); // the agreed 4%, fixed at init
        if (_refundDelay < MIN_REFUND_DELAY || _refundDelay > MAX_REFUND_DELAY) revert BadRefundDelay();
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
                claim.cumulativeAmount,
                claim.expiresAt,
                keccak256(bytes(claim.modelId))
            )
        );
        return keccak256(abi.encodePacked(hex"1901", domainSeparator(), hashStruct));
    }

    // ── Instructions ──

    /// Consumer funds its channel to `provider` (the x402 `upto` ceiling for that provider), opening it
    /// with `payout` as the provider's payout address. Later deposits must name the same payout (a
    /// consumer can't redirect a channel). A deposit also cancels a pending exit. DRAFT: a plain
    /// `transferFrom` pull (requires a prior `approve`); production uses USDC EIP-3009
    /// `receiveWithAuthorization` / Permit2 for a gasless single-tx fund (spec §4.2).
    function deposit(bytes32 provider, address payout, uint256 amount) external nonReentrant {
        if (paused) revert IsPaused();
        if (payout == address(0)) revert ZeroPayout();
        Channel storage c = channels[msg.sender][provider];
        if (c.payout == address(0)) {
            c.payout = payout;
        } else if (c.payout != payout) {
            revert PayoutMismatch();
        }
        usdc.safeTransferFrom(msg.sender, address(this), amount);
        c.balance += amount;
        c.exitRequestedAt = 0;
        emit Deposited(msg.sender, provider, c.payout, amount);
    }

    /// Settle a co-signed voucher: anyone (provider or facilitator) may submit. The consumer's EVM
    /// signature over the EIP-712 claim authorizes drawing the delta from *its* channel to
    /// `claim.provider`, which pays only that channel's payout; 0.96 of it goes there and 0.04 accrues
    /// as the protocol fee. Checks-effects-interactions. A settle never moves an exit's deadline.
    function settle(SettlementClaim calldata claim, bytes calldata sig) external nonReentrant {
        if (paused) revert IsPaused();
        if (block.timestamp > claim.expiresAt && block.timestamp - claim.expiresAt > claimGrace(claim.expiresAt)) {
            revert ClaimExpired();
        }

        // Recover the authorizing EVM signer. OZ ECDSA.recover enforces EIP-2 low-`s` and rejects the
        // zero address / malformed sigs — the on-chain mirror of `recover_evm_signer` (which also
        // rejects high-`s`, review F5), so off-chain and on-chain agree on validity.
        address signer = ECDSA.recover(digestOf(claim), sig);
        if (signer == address(0)) revert ZeroSigner();

        // The signer must be the escrow being spent. (`claim.consumer` is the Ed25519 id for the
        // off-chain link; the on-chain authority is the recovered EVM wallet that owns the escrow.)
        address consumerEvm = signer;
        Channel storage c = channels[consumerEvm][claim.provider];
        if (c.payout == address(0)) revert NoChannel();

        bytes32 modelHash = keccak256(bytes(claim.modelId));
        uint256 prior = settled[consumerEvm][claim.provider][claim.consumer][modelHash];
        // Monotonic: a replayed or stale (<= prior) cumulative is a no-op → revert. Also blocks
        // underflow (a lower cumulative can never claw back).
        if (claim.cumulativeAmount <= prior) revert StaleOrReplayedClaim();
        // Pay what the channel covers: a consumer that ran past its escrow still pays all it has, and
        // the rest settles from the same claim after a top-up (the high-water mark rises by what was
        // paid, not by the claim).
        uint256 delta = claim.cumulativeAmount - prior;
        if (delta > c.balance) delta = c.balance;
        if (delta == 0) revert InsufficientEscrow();

        // Fixed split, floor the fee (dust to the provider) — matches payout::protocol_fee.
        uint256 fee = (delta * feeBps) / 10_000;
        uint256 providerNet = delta - fee;

        // Effects before interactions (CEI): advance the high-water mark + debit the channel first.
        settled[consumerEvm][claim.provider][claim.consumer][modelHash] = prior + delta;
        c.balance -= delta;
        accruedFees += fee;
        address payout = c.payout; // the money goes only where the channel says

        // Interactions.
        usdc.safeTransfer(payout, providerNet);

        emit Settled(consumerEvm, claim.provider, payout, modelHash, delta, providerNet, fee);
    }

    /// The channel's payout address moves it (e.g. the provider rotating a wallet). What was already
    /// settled stays settled: the high-water mark is per provider, not per payout address.
    function setPayout(address consumer, bytes32 provider, address newPayout) external {
        Channel storage c = channels[consumer][provider];
        if (c.payout == address(0)) revert NoChannel();
        if (msg.sender != c.payout) revert NotPayout();
        if (newPayout == address(0)) revert ZeroPayout();
        c.payout = newPayout;
        emit PayoutChanged(consumer, provider, newPayout);
    }

    /// Consumer announces it will reclaim a channel's unspent escrow. It can `withdraw` after
    /// `refundDelay` — a fixed window for the provider to settle what it is owed, which settles can't
    /// extend (any party could otherwise keep the escrow locked by settling small steps).
    function requestExit(bytes32 provider) external {
        Channel storage c = channels[msg.sender][provider];
        if (c.payout == address(0)) revert NoChannel();
        c.exitRequestedAt = uint64(block.timestamp);
        c.exitPauseMark = _pausedTotalNow();
        emit ExitRequested(msg.sender, provider, uint64(block.timestamp) + refundDelay);
    }

    /// Consumer reclaims unspent escrow from a channel → the non-custodial exit; deposits are never
    /// trapped. Opens `refundDelay` after `requestExit`, plus however long the contract has been
    /// paused since (a pause stops settles, so it must not use up the provider's window), at most
    /// `MAX_PAUSE_EXTENSION` — past that, even while paused, so a pause can't lock escrow forever.
    /// Stays open `EXIT_WINDOW`, for one withdraw: after it, or after a withdraw, a new exit is needed.
    /// This is spec §7 option (a), the timelock, made per channel.
    function withdraw(bytes32 provider, uint256 amount) external nonReentrant {
        Channel storage c = channels[msg.sender][provider];
        if (c.exitRequestedAt == 0) revert RefundLocked();
        uint256 pausedSince = _pausedTotalNow() - c.exitPauseMark;
        if (pausedSince > MAX_PAUSE_EXTENSION) pausedSince = MAX_PAUSE_EXTENSION;
        uint256 openAt = uint256(c.exitRequestedAt) + refundDelay + pausedSince;
        if (block.timestamp < openAt) revert RefundLocked();
        if (block.timestamp > openAt + EXIT_WINDOW) revert ExitLapsed();
        if (amount > c.balance) revert WithdrawTooLarge();
        c.exitRequestedAt = 0;
        c.balance -= amount;
        usdc.safeTransfer(msg.sender, amount);
        emit Withdrawn(msg.sender, provider, amount);
    }

    /// The fee recipient takes accrued protocol fees, to any address it names.
    function withdrawFees(address to, uint256 amount) external nonReentrant {
        if (msg.sender != feeRecipient) revert NotFeeRecipient();
        if (amount > accruedFees) revert WithdrawTooLarge();
        accruedFees -= amount;
        usdc.safeTransfer(to, amount);
        emit FeesWithdrawn(to, amount);
    }

    // ── Admin (pause only — no fund authority) ──

    /// Pause settle and deposit. Time paused extends pending exits (see `withdraw`), at most
    /// `MAX_PAUSE_EXTENSION` per exit.
    function setPaused(bool _paused) external {
        if (msg.sender != admin) revert NotAdmin();
        if (!paused && _paused) {
            pausedAt = uint64(block.timestamp);
        } else if (paused && !_paused) {
            pausedTotal += uint64(block.timestamp) - pausedAt;
            lastPauseEnd = uint64(block.timestamp);
        }
        paused = _paused;
        emit Paused(_paused);
    }

    /// How long past its expiry a claim still settles. A pause stops settles, so a claim that was
    /// still live when the last pause began gets back the time that pause took, at most
    /// `MAX_PAUSE_EXTENSION` (the same cap as an exit's, so a claim still outlives the exit window it
    /// was accepted for). Only the last pause counts: a provider settles its claims once one ends.
    function claimGrace(uint256 expiresAt) public view returns (uint256) {
        if (pausedAt == 0 || paused || expiresAt < pausedAt) return 0;
        uint256 took = uint256(lastPauseEnd) - pausedAt;
        return took > MAX_PAUSE_EXTENSION ? MAX_PAUSE_EXTENSION : took;
    }

    /// Seconds paused in total up to now, the current pause included.
    function _pausedTotalNow() internal view returns (uint64) {
        return paused ? pausedTotal + (uint64(block.timestamp) - pausedAt) : pausedTotal;
    }
}
