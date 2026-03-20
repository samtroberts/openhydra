"""Economy service — extracted from CoordinatorEngine (v0.1.1).

Handles all HYDRA token operations, state channel lifecycle, barter credit
queries, and bridge settlement.  Orthogonal to the inference pipeline.
"""

from __future__ import annotations

from typing import Any


class EconomyService:
    """Manages HYDRA token economy, state channels, and barter credit queries.

    This service owns the ``_channel_provider_spend`` bookkeeping dict and
    delegates to the underlying ledger, hydra economy, and bridge objects.
    """

    def __init__(
        self,
        ledger: Any,
        hydra: Any,
        ledger_bridge: Any,
    ) -> None:
        self.ledger = ledger
        self.hydra = hydra
        self.ledger_bridge = ledger_bridge
        self._channel_provider_spend: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def account_balance(self, client_id: str) -> dict[str, Any]:
        """Return the combined barter-credit and HYDRA account balance for a client.

        Args:
            client_id: Unique identifier of the client whose balance is queried.

        Returns:
            Dict with ``client_id``, ``priority_credits``, ``hydra``, and
            ``hydra_bridge`` snapshots.
        """
        hydra_account = self.hydra.account_snapshot(client_id)
        return {
            "client_id": client_id,
            "priority_credits": round(self.ledger.balance(client_id), 6),
            "hydra": hydra_account,
            "hydra_bridge": self.ledger_bridge.account_snapshot(client_id),
        }

    def hydra_status(self) -> dict[str, Any]:
        """Return a summary of the HYDRA economy and bridge status.

        Returns:
            Dict containing ``hydra`` summary, ``hydra_bridge`` summary,
            ``mock_mode`` flag, and an optional warning string.
        """
        bridge_summary = self.ledger_bridge.summary()
        is_mock = bool(bridge_summary.get("mock_mode", True))
        return {
            "hydra": self.hydra.summary(),
            "hydra_bridge": bridge_summary,
            "mock_mode": is_mock,
            "mock_mode_warning": (
                "HYDRA bridge is running in mock mode — all token settlement is "
                "in-memory only. No real on-chain transactions occur."
                if is_mock else None
            ),
        }

    def hydra_account(self, client_id: str) -> dict[str, Any]:
        """Return the HYDRA token account snapshot for a client.

        Args:
            client_id: Unique identifier of the client.

        Returns:
            Dict with a single ``hydra`` key holding the account snapshot.
        """
        return {"hydra": self.hydra.account_snapshot(client_id)}

    def hydra_governance_params(self) -> dict[str, Any]:
        """Return the current HYDRA governance parameters from the bridge."""
        return {"hydra_governance": {"params": self.ledger_bridge.governance_params()}}

    def hydra_governance_vote(self, pubkey: str, proposal_id: str, vote: str) -> dict[str, Any]:
        """Submit a governance vote on behalf of a public key.

        Args:
            pubkey: The voter's public key.
            proposal_id: Identifier of the proposal being voted on.
            vote: Vote value (e.g. ``"yes"``, ``"no"``, ``"abstain"``).

        Returns:
            Dict wrapping the bridge vote receipt.
        """
        return {
            "hydra_governance_vote": self.ledger_bridge.submit_vote(
                pubkey=pubkey,
                proposal_id=proposal_id,
                vote=vote,
            )
        }

    # ------------------------------------------------------------------
    # Token operations
    # ------------------------------------------------------------------

    def hydra_transfer(self, from_client_id: str, to_client_id: str, amount: float) -> dict[str, Any]:
        """Transfer HYDRA tokens between two accounts.

        Args:
            from_client_id: Sender's client identifier.
            to_client_id: Recipient's client identifier.
            amount: Number of tokens to transfer.

        Returns:
            Dict with sender/receiver IDs, amount, and resulting balances.
        """
        result = self.hydra.transfer(from_peer_id=from_client_id, to_peer_id=to_client_id, amount=float(amount))
        return {
            "from_client_id": from_client_id,
            "to_client_id": to_client_id,
            "amount": float(amount),
            "result": {
                "from_balance": round(float(result["from_balance"]), 6),
                "to_balance": round(float(result["to_balance"]), 6),
            },
        }

    def hydra_stake(self, client_id: str, amount: float) -> dict[str, Any]:
        """Stake HYDRA tokens for a client, increasing their routing priority.

        Args:
            client_id: The peer's client identifier.
            amount: Number of tokens to stake.

        Returns:
            Dict with client ID, amount, and resulting balance/stake.
        """
        result = self.hydra.stake(peer_id=client_id, amount=float(amount))
        return {
            "client_id": client_id,
            "amount": float(amount),
            "result": {
                "balance": round(float(result["balance"]), 6),
                "stake": round(float(result["stake"]), 6),
            },
        }

    def hydra_unstake(self, client_id: str, amount: float) -> dict[str, Any]:
        """Unstake HYDRA tokens, returning them to the client's liquid balance.

        Args:
            client_id: The peer's client identifier.
            amount: Number of tokens to unstake.

        Returns:
            Dict with client ID, amount, and resulting balance/stake.
        """
        result = self.hydra.unstake(peer_id=client_id, amount=float(amount))
        return {
            "client_id": client_id,
            "amount": float(amount),
            "result": {
                "balance": round(float(result["balance"]), 6),
                "stake": round(float(result["stake"]), 6),
            },
        }

    # ------------------------------------------------------------------
    # State channels
    # ------------------------------------------------------------------

    def hydra_open_channel(
        self,
        channel_id: str,
        payer: str,
        payee: str,
        deposit: float,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Open a new HYDRA state channel between a payer and payee.

        Clears any previous provider-spend bookkeeping for the channel ID
        and delegates to the underlying economy.

        Args:
            channel_id: Unique identifier for the new channel.
            payer: Public key or peer ID of the payer.
            payee: Public key or peer ID of the payee.
            deposit: Initial deposit amount in HYDRA tokens.
            ttl_seconds: Optional time-to-live in seconds before auto-expiry.

        Returns:
            Dict wrapping the ``hydra_channel`` open payload.
        """
        self._channel_provider_spend.pop(str(channel_id).strip(), None)
        return {
            "hydra_channel": self.hydra.open_state_channel(
                channel_id=channel_id,
                payer=payer,
                payee=payee,
                deposit=float(deposit),
                ttl_seconds=(int(ttl_seconds) if ttl_seconds is not None else None),
            )
        }

    def hydra_charge_channel(self, channel_id: str, amount: float, provider_peer_id: str | None = None) -> dict[str, Any]:
        """Charge (debit) a state channel and record provider spend.

        Args:
            channel_id: The channel to charge.
            amount: Amount to charge in HYDRA tokens.
            provider_peer_id: Optional provider who earned the charge.

        Returns:
            Dict wrapping the ``hydra_channel`` charge payload.
        """
        payload = self.hydra.charge_state_channel(channel_id=channel_id, amount=float(amount))
        provider = str(provider_peer_id or payload.get("payee") or "").strip()
        if provider:
            self._record_channel_provider_spend(channel_id, provider, float(amount))
        return {"hydra_channel": payload}

    def hydra_reconcile_channel(self, channel_id: str, total_spent: float, nonce: int) -> dict[str, Any]:
        """Reconcile a state channel's off-chain spend with the on-chain state.

        Args:
            channel_id: The channel to reconcile.
            total_spent: Cumulative amount spent so far.
            nonce: Monotonically increasing nonce for replay protection.

        Returns:
            Dict wrapping the ``hydra_channel`` reconciliation payload.
        """
        payload = self.hydra.reconcile_state_channel(
            channel_id=channel_id,
            total_spent=float(total_spent),
            nonce=int(nonce),
        )
        payee = str(payload.get("payee", "")).strip()
        if payee:
            self._set_channel_payee_spend(channel_id, payee, float(payload.get("spent", 0.0)))
        return {"hydra_channel": payload}

    def hydra_close_channel(self, channel_id: str) -> dict[str, Any]:
        """Close a state channel and settle via the bridge (burn + mint).

        Burns the payer's spent tokens and mints provider rewards
        proportional to each provider's recorded spend.

        Args:
            channel_id: The channel to close.

        Returns:
            Dict with ``hydra_channel_close`` payload and
            ``hydra_bridge_settlement`` receipt.
        """
        close_payload = self.hydra.close_state_channel(channel_id=channel_id)
        settlement = self._bridge_settle_channel_close(close_payload)
        return {
            "hydra_channel_close": close_payload,
            "hydra_bridge_settlement": settlement,
        }

    # ------------------------------------------------------------------
    # Internal bookkeeping
    # ------------------------------------------------------------------

    def _record_channel_provider_spend(self, channel_id: str, provider_peer_id: str, amount: float) -> None:
        """Accumulate spend for a provider within a channel for settlement."""
        key = str(channel_id).strip()
        provider = str(provider_peer_id).strip()
        delta = max(0.0, float(amount))
        if not key or not provider or delta <= 0.0:
            return
        bucket = self._channel_provider_spend.setdefault(key, {})
        bucket[provider] = round(float(bucket.get(provider, 0.0)) + delta, 6)

    def _set_channel_payee_spend(self, channel_id: str, payee_peer_id: str, total_spent: float) -> None:
        """Override the payee's cumulative spend record during reconciliation."""
        key = str(channel_id).strip()
        payee = str(payee_peer_id).strip()
        target = max(0.0, float(total_spent))
        if not key or not payee:
            return
        bucket = self._channel_provider_spend.setdefault(key, {})
        allocated = sum(float(value) for value in bucket.values())
        if allocated <= 0.0:
            bucket[payee] = round(target, 6)
            return
        if target >= allocated:
            bucket[payee] = round(float(bucket.get(payee, 0.0)) + (target - allocated), 6)
            return
        bucket.clear()
        bucket[payee] = round(target, 6)

    def _bridge_settle_channel_close(self, close_payload: dict[str, Any]) -> dict[str, Any]:
        """Execute burn-and-mint bridge settlement when a channel closes.

        Burns the payer's total spent amount, then mints proportional rewards
        to each provider that served the channel.  Records any errors without
        raising so that partial settlement is preserved.

        Args:
            close_payload: The channel close payload from the economy layer.

        Returns:
            Settlement dict with ``burn_receipt``, ``mint_receipts``, and
            ``errors`` lists.
        """
        channel_id = str(close_payload.get("channel_id", "")).strip()
        payer = str(close_payload.get("payer", "")).strip()
        payee = str(close_payload.get("payee", "")).strip()
        payee_amount = max(0.0, float(close_payload.get("payee_amount", 0.0)))

        spent_by_provider = {
            str(peer_id): max(0.0, float(amount))
            for peer_id, amount in dict(self._channel_provider_spend.pop(channel_id, {})).items()
            if str(peer_id).strip() and float(amount) > 0.0
        }
        if not spent_by_provider and payee and payee_amount > 0.0:
            spent_by_provider = {payee: payee_amount}

        settlement: dict[str, Any] = {
            "enabled": True,
            "channel_id": channel_id,
            "payer_pubkey": payer,
            "burn_receipt": None,
            "mint_receipts": [],
            "errors": [],
        }
        if not payer or payee_amount <= 0.0:
            settlement["enabled"] = False
            return settlement

        burn_amount = payee_amount
        try:
            settlement["burn_receipt"] = self.ledger_bridge.burn_for_compute(
                payer_pubkey=payer,
                amount=burn_amount,
            )
        except RuntimeError as exc:
            settlement["errors"].append(f"burn_for_compute_failed:{exc}")
            return settlement

        minted_total = 0.0
        for provider_id, amount in spent_by_provider.items():
            if minted_total >= burn_amount:
                break
            mint_amount = min(max(0.0, float(amount)), burn_amount - minted_total)
            if mint_amount <= 0.0:
                continue
            try:
                receipt = self.ledger_bridge.mint_provider_rewards(
                    payee_pubkey=provider_id,
                    amount=mint_amount,
                )
                settlement["mint_receipts"].append(receipt)
                minted_total += mint_amount
            except RuntimeError as exc:
                settlement["errors"].append(f"mint_provider_rewards_failed:{provider_id}:{exc}")
                break

        return settlement
