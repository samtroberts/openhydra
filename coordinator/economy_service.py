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
        hydra_account = self.hydra.account_snapshot(client_id)
        return {
            "client_id": client_id,
            "priority_credits": round(self.ledger.balance(client_id), 6),
            "hydra": hydra_account,
            "hydra_bridge": self.ledger_bridge.account_snapshot(client_id),
        }

    def hydra_status(self) -> dict[str, Any]:
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
        return {"hydra": self.hydra.account_snapshot(client_id)}

    def hydra_governance_params(self) -> dict[str, Any]:
        return {"hydra_governance": {"params": self.ledger_bridge.governance_params()}}

    def hydra_governance_vote(self, pubkey: str, proposal_id: str, vote: str) -> dict[str, Any]:
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
        payload = self.hydra.charge_state_channel(channel_id=channel_id, amount=float(amount))
        provider = str(provider_peer_id or payload.get("payee") or "").strip()
        if provider:
            self._record_channel_provider_spend(channel_id, provider, float(amount))
        return {"hydra_channel": payload}

    def hydra_reconcile_channel(self, channel_id: str, total_spent: float, nonce: int) -> dict[str, Any]:
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
        key = str(channel_id).strip()
        provider = str(provider_peer_id).strip()
        delta = max(0.0, float(amount))
        if not key or not provider or delta <= 0.0:
            return
        bucket = self._channel_provider_spend.setdefault(key, {})
        bucket[provider] = round(float(bucket.get(provider, 0.0)) + delta, 6)

    def _set_channel_payee_spend(self, channel_id: str, payee_peer_id: str, total_spent: float) -> None:
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
