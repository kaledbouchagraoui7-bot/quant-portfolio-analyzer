"""
Bot 6 — Execution Bot (`src/multi_bot_engine/execution_bot.py`).

Simulated Paper-Only Order Router (`LIVE_BROKER_ENABLED = False`, `PAPER_TRADING_ONLY = True`).
Executes ONLY orders bearing a valid HMAC `ApprovalToken` issued by `risk_management_bot.py`.
Any unapproved or forged order is rejected immediately with `PermissionError`.
Not Investment Advice (AMF / MiFID II Compliant).
"""

from __future__ import annotations

import hmac
from typing import Any

from src.multi_bot_engine.risk_management_bot import _RISK_GATE_SECRET, ApprovedOrder
from src.multi_bot_engine.shared_infrastructure import (
    LIVE_BROKER_ENABLED,
    PAPER_TRADING_ONLY,
    ZERO_REAL_MONEY_RISK,
    PaperPortfolioLedger,
    sign_approval_token,
)


def verify_risk_approval(order: Any) -> None:
    """Reject any order that did not pass through `risk_management_bot`."""
    if not isinstance(order, ApprovedOrder):
        raise PermissionError("ExecutionBot blocked unapproved order: must be an ApprovedOrder from RiskManagementBot.")
    expected = sign_approval_token(_RISK_GATE_SECRET, order.symbol, order.target_weight, order.notional_eur)
    if not hmac.compare_digest(order.approval_token, expected):
        raise PermissionError(f"ExecutionBot rejected forged/invalid risk approval token for {order.symbol}.")


def execute_paper_orders(
    approved_orders: list[ApprovedOrder],
    latest_prices: dict[str, float],
    ledger: PaperPortfolioLedger,
    slippage_bps: float = 5.0,
    commission_bps: float = 2.0,
) -> dict[str, Any]:
    """Route verified paper orders with realistic spread/slippage & commission fees."""
    assert PAPER_TRADING_ONLY and ZERO_REAL_MONEY_RISK and (LIVE_BROKER_ENABLED is False)

    executed: list[dict[str, Any]] = []
    total_fees_eur = 0.0
    fee_rate = (slippage_bps + commission_bps) / 10000.0

    for order in approved_orders:
        verify_risk_approval(order)
        price = float(latest_prices.get(order.symbol, 0.0))
        if price <= 0.0:
            continue

        cur_qty = ledger.positions.get(order.symbol, 0.0)
        target_qty = order.notional_eur / price
        delta_qty = target_qty - cur_qty
        trade_notional = abs(delta_qty) * price
        fee_eur = round(trade_notional * fee_rate, 4)

        if abs(delta_qty) * price < 1.0:
            continue

        ledger.cash_eur = round(ledger.cash_eur - (delta_qty * price) - fee_eur, 4)
        ledger.positions[order.symbol] = round(target_qty, 6)
        total_fees_eur += fee_eur
        executed.append(
            {
                "symbol": order.symbol,
                "delta_qty": round(delta_qty, 6),
                "fill_price": price,
                "fee_eur": fee_eur,
                "target_weight": order.target_weight,
            }
        )

    nav_after = ledger.total_nav(latest_prices)
    ledger.record_audit(
        "execution_bot",
        "paper_orders_routed",
        {"executed_count": len(executed), "total_fees_eur": round(total_fees_eur, 4), "nav_eur": nav_after},
    )
    return {
        "live_broker_enabled": LIVE_BROKER_ENABLED,
        "paper_trading_only": PAPER_TRADING_ONLY,
        "executed_orders": executed,
        "total_fees_eur": round(total_fees_eur, 4),
        "nav_after_eur": nav_after,
    }
