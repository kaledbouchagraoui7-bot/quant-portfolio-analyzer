"""
Bot 2 — Market Research Bot (`src/multi_bot_engine/market_research_bot.py`).

Collects and normalizes multi-asset price series (Stocks, ETFs, Crypto spot),
volatility regimes, and fundamental metadata into structured research snapshots.
Not Investment Advice (AMF / MiFID II Compliant).
"""

from __future__ import annotations

import math
from typing import Any

from src.multi_bot_engine.shared_infrastructure import (
    DEFAULT_UNIVERSE,
    PAPER_TRADING_ONLY,
    PaperPortfolioLedger,
)


def collect_market_snapshot(
    price_series: dict[str, list[float]],
    ledger: PaperPortfolioLedger,
    vol_window: int = 20,
) -> dict[str, Any]:
    """Build normalized multi-asset price, return, and volatility snapshot."""
    assert PAPER_TRADING_ONLY
    snapshot: dict[str, dict[str, Any]] = {}
    latest_prices: dict[str, float] = {}

    for sym, prices in sorted(price_series.items()):
        if len(prices) < 2:
            continue
        latest_prices[sym] = float(prices[-1])
        rets = [(prices[i] / max(prices[i - 1], 1e-8)) - 1.0 for i in range(1, len(prices))]
        sub = rets[-min(vol_window, len(rets)) :]
        mean_r = sum(sub) / len(sub)
        var = sum((r - mean_r) ** 2 for r in sub) / max(len(sub) - 1, 1)
        ann_vol = math.sqrt(max(var, 0.0)) * math.sqrt(252.0)

        lookback = min(20, len(prices) - 1)
        mom_20 = (prices[-1] / max(prices[-1 - lookback], 1e-8)) - 1.0
        sma_20 = sum(prices[-min(20, len(prices)) :]) / min(20, len(prices))
        meta = DEFAULT_UNIVERSE.get(sym, {"asset_class": "stock", "sector": "general", "roic": 0.10, "dividend_yield": 0.0})

        snapshot[sym] = {
            "symbol": sym,
            "latest_price": round(float(prices[-1]), 4),
            "annual_vol": round(ann_vol, 4),
            "mom_20": round(mom_20, 4),
            "above_sma_20": prices[-1] >= sma_20,
            "zscore_20": round((prices[-1] - sma_20) / max(math.sqrt(max(var, 1e-8)) * sma_20, 1e-6), 4),
            **meta,
        }

    avg_vol = sum(v["annual_vol"] for v in snapshot.values()) / max(len(snapshot), 1)
    macro_regime = "high_volatility" if avg_vol > 0.25 else ("expansion" if avg_vol < 0.16 else "normal")

    ledger.record_audit(
        "market_research_bot",
        "snapshot_collected",
        {"assets_count": len(snapshot), "macro_regime": macro_regime, "avg_annual_vol": round(avg_vol, 4)},
    )
    return {
        "macro_regime": macro_regime,
        "avg_annual_vol": round(avg_vol, 4),
        "latest_prices": latest_prices,
        "assets": snapshot,
    }
