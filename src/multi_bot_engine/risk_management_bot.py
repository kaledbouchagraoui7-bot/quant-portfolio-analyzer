"""
Bot 5 — Risk Management Bot (`src/multi_bot_engine/risk_management_bot.py`).

Hard Veto Gatekeeper — Cannot Be Bypassed.
Inspects every proposed order before execution:
  - Enforces <= 1% max risk per position (`cap_per_trade_risk`)
  - Enforces asset-class concentration caps (max 10% total Crypto, max 25% single stock)
  - Applies Fractional Kelly (`0.25x`) sizing & portfolio drawdown vetoes
  - Issues HMAC-signed `ApprovalToken` required by `execution_bot.py`.
Not Investment Advice (AMF / MiFID II Compliant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.multi_bot_engine.shared_infrastructure import (
    MAX_LEVERAGE,
    PAPER_TRADING_ONLY,
    ZERO_REAL_MONEY_RISK,
    PaperPortfolioLedger,
    sign_approval_token,
)
from src.risk_execution_engine import (
    MAX_RISK_PER_TRADE,
    cap_per_trade_risk,
    check_drawdown_circuit_breaker,
    fractional_kelly_fraction,
)

_RISK_GATE_SECRET = "ALCOA_PLUS_PAPER_RISK_GATE_V1"


@dataclass(frozen=True)
class ApprovedOrder:
    symbol: str
    asset_class: str
    target_weight: float
    notional_eur: float
    approval_token: str
    sub_strategy: str


def validate_and_approve_orders(
    proposals: list[dict[str, Any]],
    latest_prices: dict[str, float],
    ledger: PaperPortfolioLedger,
    max_crypto_total_weight: float = 0.10,
    max_single_stock_weight: float = 0.25,
) -> dict[str, Any]:
    """
    Evaluate all strategy proposals against hard risk limits and issue signed `ApprovedOrder`s.
    Returns both `approved_orders` and `vetoed_orders`.
    """
    assert PAPER_TRADING_ONLY and ZERO_REAL_MONEY_RISK
    nav = ledger.total_nav(latest_prices)
    dd = ledger.current_drawdown(latest_prices)
    cb = check_drawdown_circuit_breaker(dd)

    approved: list[ApprovedOrder] = []
    vetoed: list[dict[str, Any]] = []

    if ledger.kill_switch_triggered or cb.halted:
        for prop in proposals:
            vetoed.append({**prop, "veto_reason": f"circuit_breaker_halted:{cb.reason}"})
        ledger.record_audit("risk_management_bot", "all_orders_vetoed", {"reason": cb.reason, "drawdown": round(dd, 4)})
        return {"approved_orders": approved, "vetoed_orders": vetoed, "nav_eur": nav}

    crypto_used = 0.0
    total_used = 0.0
    kelly_cap = fractional_kelly_fraction(win_rate=0.56, payoff_ratio=1.6, fraction=0.25, max_cap=0.25)

    for prop in proposals:
        sym = prop["symbol"]
        ac = prop["asset_class"]
        raw_w = float(prop.get("proposed_weight", 0.0))
        requested_lev = float(prop.get("leverage", 1.0))

        # Hard veto on any leverage > MAX_LEVERAGE or negative/excessive weight
        if requested_lev > MAX_LEVERAGE or raw_w <= 0.0:
            vetoed.append({**prop, "veto_reason": "leverage_or_invalid_weight_veto"})
            continue

        # Enforce <= 1% equity risk per trade via stop-loss distance proxy
        stop_dist = max(0.03, min(0.12, float(prop.get("annual_vol", 0.16)) * 0.25))
        risk_capped_w = cap_per_trade_risk(stop_distance_pct=stop_dist, max_risk_pct=MAX_RISK_PER_TRADE)
        allowed_w = min(raw_w, risk_capped_w, kelly_cap) * cb.position_scale

        if ac == "crypto":
            rem_crypto = max(0.0, max_crypto_total_weight - crypto_used)
            if rem_crypto <= 1e-6:
                vetoed.append({**prop, "veto_reason": "crypto_concentration_cap_veto"})
                continue
            allowed_w = min(allowed_w, rem_crypto)
            crypto_used += allowed_w
        elif ac == "stock":
            allowed_w = min(allowed_w, max_single_stock_weight)

        if total_used + allowed_w > MAX_LEVERAGE:
            allowed_w = max(0.0, MAX_LEVERAGE - total_used)

        allowed_w = round(allowed_w, 6)
        if allowed_w <= 1e-4:
            vetoed.append({**prop, "veto_reason": "zero_post_risk_allocation"})
            continue

        total_used += allowed_w
        notional_eur = round(nav * allowed_w, 4)
        token = sign_approval_token(_RISK_GATE_SECRET, sym, allowed_w, notional_eur)
        approved.append(
            ApprovedOrder(
                symbol=sym,
                asset_class=ac,
                target_weight=allowed_w,
                notional_eur=notional_eur,
                approval_token=token,
                sub_strategy=prop.get("sub_strategy", "unknown"),
            )
        )

    ledger.record_audit(
        "risk_management_bot",
        "orders_validated",
        {"approved_count": len(approved), "vetoed_count": len(vetoed), "total_target_weight": round(total_used, 4)},
    )
    return {"approved_orders": approved, "vetoed_orders": vetoed, "nav_eur": nav}
