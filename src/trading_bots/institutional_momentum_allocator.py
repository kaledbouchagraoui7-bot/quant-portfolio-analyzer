"""
Institutional Cross-Asset Dual-Momentum Allocator (`src/trading_bots/institutional_momentum_allocator.py`).

Synthesizes harvested open-source patterns (`PyPortfolioOpt`, `Riskfolio-Lib`, `vectorbt`, `qlib`, `empyrical`):
  1. Dual-Momentum Absolute Cash Shield: requires BOTH positive trailing excess return over T-bills
     AND price >= long-term trend filter (`200`-bar SMA, adaptive on shorter histories).
  2. Inverse-Volatility Weighting + Regime Vol-Cap: weights qualifying assets inversely to realized
     volatility and scales total risky exposure down into `cash_bills` during volatility spikes.
"""

from __future__ import annotations

import math
from typing import Any

from src.risk_execution_engine import (
    MAX_LEVERAGE,
    PAPER_TRADING_ONLY,
    ZERO_REAL_MONEY_RISK,
)


def _annualized_vol(series: list[float], end_idx: int, window: int = 20) -> float:
    start = max(1, end_idx - window + 1)
    rets: list[float] = []
    for k in range(start, end_idx + 1):
        prev = max(series[k - 1], 1e-8)
        rets.append((series[k] / prev) - 1.0)
    if len(rets) < 2:
        return 0.08
    mean_r = sum(rets) / len(rets)
    var = sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)
    return max(math.sqrt(max(var, 0.0)) * math.sqrt(252.0), 1e-4)


def _ema_series(series: list[float], span: int) -> list[float]:
    if not series:
        return []
    alpha = 2.0 / (max(span, 1) + 1.0)
    out = [float(series[0])]
    for p in series[1:]:
        out.append(alpha * float(p) + (1.0 - alpha) * out[-1])
    return out


def compute_dual_momentum_weights(
    asset_prices: dict[str, list[float]],
    lookback: int = 60,
    trend_window: int = 120,
    vol_window: int = 25,
    target_annual_vol: float = 0.025,
    rebalance_every: int = 21,
    defensive_asset: str = "sovereign_bonds",
    cash_asset: str = "cash_bills",
    risk_free_period_return: float = 0.004,
) -> dict[str, Any]:
    """
    Compute cross-asset Risk-Parity + Dual-Momentum allocation weights with:
      1. Inverse-Volatility / Variance Risk-Parity base weighting across all assets.
      2. Dual-Momentum Absolute Cash Shield: assets failing BOTH trailing return > risk-free
         AND long-term trend filter (`p_now >= SMA`) are shielded into `cash_bills`.
      3. Regime Vol-Cap: scales down total risky exposure into `cash_bills` when portfolio
         volatility exceeds `target_annual_vol` (`sum(weights) <= 1.0`, `MAX_LEVERAGE = 1.0`).
    """
    assert PAPER_TRADING_ONLY and ZERO_REAL_MONEY_RISK
    assets = sorted(asset_prices.keys())
    if not assets:
        return {"weights_history": [], "paper_trading_only": True}

    n = min(len(v) for v in asset_prices.values())
    history: list[dict[str, float]] = []

    for i in range(n):
        if i < lookback:
            if cash_asset in assets:
                row = {a: (1.0 if a == cash_asset else 0.0) for a in assets}
            else:
                equal_w = round(1.0 / len(assets), 6)
                row = {a: equal_w for a in assets}
            history.append(row)
            continue

        # Periodic rebalance cadence eliminates daily noise whipsaw and turnover drag
        if history and (i % max(rebalance_every, 1) != 0):
            history.append(dict(history[-1]))
            continue

        mom_scores: dict[str, float] = {}
        trend_ok: dict[str, bool] = {}
        vols: dict[str, float] = {}

        for a in assets:
            series = asset_prices[a]
            p_now = series[i]
            p_prev = max(series[i - lookback], 1e-8)
            mom_scores[a] = (p_now / p_prev) - 1.0

            sma_slice = series[max(0, i - trend_window + 1) : i + 1]
            sma_val = sum(sma_slice) / len(sma_slice)
            trend_ok[a] = p_now >= sma_val
            vols[a] = _annualized_vol(series, i, window=vol_window)

        # Base Inverse-Volatility Risk-Parity weights across non-cash assets
        non_cash = [a for a in assets if a != cash_asset]
        if not non_cash:
            history.append({a: round(1.0 / len(assets), 6) for a in assets})
            continue

        inv_vols = {a: 1.0 / max(vols[a], 1e-4) for a in non_cash}
        sum_inv = sum(inv_vols.values()) or 1.0
        base_weights = {a: inv_vols[a] / sum_inv for a in non_cash}

        # Dual-Momentum Absolute Cash Shield:
        # Assets failing BOTH momentum > risk_free AND trend filter are shielded 100% into cash_bills;
        # Assets failing one filter are throttled by 35% into cash_bills.
        shielded_weights: dict[str, float] = {}
        for a in non_cash:
            pos_mom = mom_scores[a] > risk_free_period_return
            if pos_mom and trend_ok[a]:
                shielded_weights[a] = base_weights[a]
            elif pos_mom or trend_ok[a]:
                shielded_weights[a] = base_weights[a] * 0.65
            else:
                shielded_weights[a] = 0.0

        # Regime Volatility Cap: scale total risky weight down if portfolio vol exceeds target_annual_vol
        approx_port_vol = math.sqrt(sum((shielded_weights[a] * vols[a]) ** 2 for a in non_cash))
        vol_scale = min(1.0, target_annual_vol / max(approx_port_vol, 1e-6))

        row = {a: 0.0 for a in assets}
        allocated = 0.0
        for a in non_cash:
            w = round(shielded_weights[a] * vol_scale, 6)
            row[a] = w
            allocated += w

        shield_target = cash_asset if cash_asset in row else (defensive_asset if defensive_asset in row else non_cash[0])
        row[shield_target] = round(row.get(shield_target, 0.0) + max(0.0, 1.0 - allocated), 6)

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
