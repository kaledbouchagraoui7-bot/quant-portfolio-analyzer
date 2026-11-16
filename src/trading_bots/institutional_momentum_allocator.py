"""
Institutional Cross-Asset Dual-Momentum Allocator (`src/trading_bots/institutional_momentum_allocator.py`).

Rotates across liquid equities, sovereign bonds, gold, and cash equivalents using
relative + absolute momentum (Antonacci / institutional trend filter) with zero leverage.
"""

from __future__ import annotations

from typing import Any

from src.risk_execution_engine import MAX_LEVERAGE, PAPER_TRADING_ONLY, ZERO_REAL_MONEY_RISK


def compute_dual_momentum_weights(
    asset_prices: dict[str, list[float]],
    lookback: int = 20,
    defensive_asset: str = "sovereign_bonds",
    cash_asset: str = "cash_bills",
    risk_free_period_return: float = 0.002,
) -> dict[str, Any]:
    """
    Compute cross-asset dual-momentum allocation weights at each bar (`sum(weights) == 1.0`, `MAX_LEVERAGE = 1.0`).
    """
    assert PAPER_TRADING_ONLY and ZERO_REAL_MONEY_RISK
    assets = sorted(asset_prices.keys())
    if not assets:
        return {"weights_history": [], "paper_trading_only": True}

    n = min(len(v) for v in asset_prices.values())
    history: list[dict[str, float]] = []

    for i in range(n):
        if i < lookback:
            equal_w = round(1.0 / len(assets), 6)
            history.append({a: equal_w for a in assets})
            continue

        mom_scores: dict[str, float] = {}
        for a in assets:
            p_now = asset_prices[a][i]
            p_prev = max(asset_prices[a][i - lookback], 1e-8)
            mom_scores[a] = (p_now / p_prev) - 1.0

        risk_assets = [a for a in assets if a not in (defensive_asset, cash_asset)]
        ranked_risk = sorted(risk_assets, key=lambda a: mom_scores[a], reverse=True)
        top_positive = [a for a in ranked_risk[:2] if mom_scores[a] > risk_free_period_return]

        row = {a: 0.0 for a in assets}
        if len(top_positive) == 2:
            row[top_positive[0]] = 0.55
            row[top_positive[1]] = 0.45
        elif len(top_positive) == 1:
            row[top_positive[0]] = 0.60
            def_target = defensive_asset if defensive_asset in row and mom_scores.get(defensive_asset, 0.0) > 0 else (cash_asset if cash_asset in row else top_positive[0])
            row[def_target] = round(row.get(def_target, 0.0) + 0.40, 6)
        else:
            def_target = defensive_asset if defensive_asset in row and mom_scores.get(defensive_asset, 0.0) > 0 else (cash_asset if cash_asset in row else assets[0])
            row[def_target] = 1.0

        total_w = sum(row.values())
        if total_w > MAX_LEVERAGE:
            for a in row:
                row[a] = round(row[a] / total_w, 6)
        history.append(row)

    return {
        "bot": "institutional_momentum_allocator",
        "paper_trading_only": PAPER_TRADING_ONLY,
        "max_leverage": MAX_LEVERAGE,
        "weights_history": history,
    }
