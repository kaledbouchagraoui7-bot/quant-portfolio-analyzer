"""
Bot 3 — Analysis Bot (`src/multi_bot_engine/analysis_bot.py`).

Evaluates fundamental quality (ROIC, dividend stability), ETF cost efficiency,
sector trend strength, and crypto liquidity/volatility filters.
Not Investment Advice (AMF / MiFID II Compliant).
"""

from __future__ import annotations

from typing import Any

from src.multi_bot_engine.shared_infrastructure import (
    PAPER_TRADING_ONLY,
    PaperPortfolioLedger,
)


def evaluate_assets(snapshot: dict[str, Any], ledger: PaperPortfolioLedger) -> dict[str, dict[str, Any]]:
    """Score every asset in [0, 100] on quality, cost efficiency, trend, and volatility."""
    assert PAPER_TRADING_ONLY
    scores: dict[str, dict[str, Any]] = {}

    for sym, info in snapshot.get("assets", {}).items():
        asset_class = info.get("asset_class", "stock")
        roic = float(info.get("roic", 0.0))
        div_y = float(info.get("dividend_yield", 0.0))
        exp_ratio = float(info.get("expense_ratio", 0.0))
        ann_vol = float(info.get("annual_vol", 0.20))
        mom = float(info.get("mom_20", 0.0))

        quality_score = min(35.0, roic * 100.0) + min(15.0, div_y * 400.0)
        if asset_class == "etf":
            quality_score += max(0.0, 20.0 - exp_ratio * 2000.0)
        trend_score = (20.0 if info.get("above_sma_20") else 5.0) + max(-10.0, min(15.0, mom * 200.0))
        vol_penalty = max(0.0, (ann_vol - 0.18) * 60.0)

        composite = round(max(0.0, min(100.0, quality_score + trend_score - vol_penalty + 25.0)), 2)
        scores[sym] = {
            "symbol": sym,
            "asset_class": asset_class,
            "composite_score": composite,
            "annual_vol": ann_vol,
            "mom_20": mom,
            "zscore_20": info.get("zscore_20", 0.0),
            "above_sma_20": info.get("above_sma_20", False),
        }

    ledger.record_audit("analysis_bot", "assets_scored", {"scored_count": len(scores)})
    return scores
