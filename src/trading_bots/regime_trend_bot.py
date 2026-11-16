"""
Multi-Timeframe Market-Regime & Trend Breakout Bot (`src/trading_bots/regime_trend_bot.py`).

Combines EMA trend structure, Donchian channel breakouts, ATR volatility regime detection,
and volatility-adjusted trailing stops under strict paper-trading guardrails.
"""

from __future__ import annotations

import math
from typing import Any

from src.risk_execution_engine import (
    MAX_LEVERAGE,
    PAPER_TRADING_ONLY,
    ZERO_REAL_MONEY_RISK,
    atr_volatility_target_weight,
    cap_per_trade_risk,
)


def _ema(series: list[float], span: int) -> list[float]:
    if not series:
        return []
    alpha = 2.0 / (max(span, 1) + 1.0)
    out = [float(series[0])]
    for price in series[1:]:
        out.append(alpha * float(price) + (1.0 - alpha) * out[-1])
    return out


def _rolling_volatility(prices: list[float], window: int = 20) -> list[float]:
    if len(prices) < 2:
        return [0.0 for _ in prices]
    rets = [0.0]
    for i in range(1, len(prices)):
        prev = max(prices[i - 1], 1e-8)
        rets.append((prices[i] / prev) - 1.0)
    vols: list[float] = []
    for i in range(len(rets)):
        sub = rets[max(0, i - window + 1) : i + 1]
        if len(sub) < 2:
            vols.append(0.10)
            continue
        mean_r = sum(sub) / len(sub)
        var = sum((r - mean_r) ** 2 for r in sub) / (len(sub) - 1)
        vols.append(math.sqrt(max(var, 0.0)) * math.sqrt(252.0))
    return vols


def detect_market_regime(
    prices: list[float],
    fast_span: int = 12,
    slow_span: int = 36,
    vol_window: int = 20,
    high_vol_threshold: float = 0.28,
) -> list[str]:
    """Classify each bar into 'trending', 'mean_reverting', or 'high_volatility'."""
    if not prices:
        return []
    fast = _ema(prices, fast_span)
    slow = _ema(prices, slow_span)
    vols = _rolling_volatility(prices, vol_window)
    regimes: list[str] = []
    for i in range(len(prices)):
        if vols[i] >= high_vol_threshold:
            regimes.append("high_volatility")
            continue
        spread = abs(fast[i] - slow[i]) / max(slow[i], 1e-8)
        regimes.append("trending" if spread >= 0.012 else "mean_reverting")
    return regimes


def generate_regime_trend_signals(
    prices: list[float],
    fast_span: int = 12,
    slow_span: int = 36,
    donchian_window: int = 20,
    trailing_stop_pct: float = 0.04,
) -> dict[str, Any]:
    """
    Generate causal (t-1 signal applied to t return) position weights in [0.0, MAX_LEVERAGE].
    """
    assert PAPER_TRADING_ONLY and ZERO_REAL_MONEY_RISK
    if len(prices) < 3:
        return {"weights": [0.0 for _ in prices], "regimes": [], "paper_trading_only": True}

    fast = _ema(prices, fast_span)
    slow = _ema(prices, slow_span)
    vols = _rolling_volatility(prices, 20)
    regimes = detect_market_regime(prices, fast_span, slow_span)

    weights = [0.0] * len(prices)
    peak_since_entry = prices[0]
    in_position = False

    for i in range(1, len(prices)):
        lookback = prices[max(0, i - donchian_window) : i]
        channel_high = max(lookback) if lookback else prices[i]
        trend_up = fast[i] > slow[i] and prices[i] >= 0.995 * channel_high

        if in_position:
            peak_since_entry = max(peak_since_entry, prices[i])
            hit_trailing_stop = prices[i] < peak_since_entry * (1.0 - trailing_stop_pct)
            if hit_trailing_stop or fast[i] < slow[i] or regimes[i] == "high_volatility":
                in_position = False
                weights[i] = 0.0
                continue
        elif trend_up and regimes[i] != "high_volatility":
            in_position = True
            peak_since_entry = prices[i]

        if in_position:
            vol_w = atr_volatility_target_weight(max(vols[i], 0.06), target_annual_vol=0.12)
            risk_w = cap_per_trade_risk(trailing_stop_pct, max_risk_pct=0.01)
            weights[i] = min(vol_w, risk_w, MAX_LEVERAGE)

    return {
        "bot": "regime_trend_bot",
        "paper_trading_only": PAPER_TRADING_ONLY,
        "max_leverage": MAX_LEVERAGE,
        "weights": weights,
        "regimes": regimes,
    }
