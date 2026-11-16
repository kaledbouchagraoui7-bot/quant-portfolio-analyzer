"""
Statistical Mean-Reversion & Z-Score Bot (`src/trading_bots/mean_reversion_stat_bot.py`).

Uses rolling Z-score / Bollinger bands with strict volatility shock filters so entries
halt automatically during macro shock regimes (`PAPER_TRADING_ONLY = True`).
"""

from __future__ import annotations

import math
from typing import Any

from src.risk_execution_engine import (
    MAX_LEVERAGE,
    PAPER_TRADING_ONLY,
    ZERO_REAL_MONEY_RISK,
    cap_per_trade_risk,
)
from src.trading_bots.regime_trend_bot import _rolling_volatility


def generate_mean_reversion_signals(
    prices: list[float],
    lookback: int = 20,
    entry_z: float = -1.35,
    exit_z: float = 0.0,
    shock_vol_ceiling: float = 0.26,
    stop_loss_pct: float = 0.035,
) -> dict[str, Any]:
    """
    Enter long paper positions when price Z-score drops below `entry_z` in non-shock regimes
    and exit on mean reversion (`z >= exit_z`) or stop-loss breach.
    """
    assert PAPER_TRADING_ONLY and ZERO_REAL_MONEY_RISK
    n = len(prices)
    if n < lookback + 2:
        return {"weights": [0.0] * n, "zscores": [0.0] * n, "paper_trading_only": True}

    vols = _rolling_volatility(prices, lookback)
    weights = [0.0] * n
    zscores = [0.0] * n
    entry_price = 0.0
    active_w = 0.0

    for i in range(lookback, n):
        window = prices[i - lookback : i + 1]
        mean_p = sum(window) / len(window)
        var_p = sum((p - mean_p) ** 2 for p in window) / max(len(window) - 1, 1)
        std_p = math.sqrt(max(var_p, 1e-10))
        z = (prices[i] - mean_p) / std_p
        zscores[i] = round(z, 4)

        # Hard volatility filter: halt during macro shock regimes
        if vols[i] >= shock_vol_ceiling:
            active_w = 0.0
            weights[i] = 0.0
            continue

        if active_w > 0:
            stopped_out = prices[i] < entry_price * (1.0 - stop_loss_pct)
            if z >= exit_z or stopped_out:
                active_w = 0.0
        else:
            if z <= entry_z:
                active_w = min(cap_per_trade_risk(stop_loss_pct, max_risk_pct=0.01), MAX_LEVERAGE)
                entry_price = prices[i]

        weights[i] = active_w

    return {
        "bot": "mean_reversion_stat_bot",
        "paper_trading_only": PAPER_TRADING_ONLY,
        "max_leverage": MAX_LEVERAGE,
        "weights": weights,
        "zscores": zscores,
    }
