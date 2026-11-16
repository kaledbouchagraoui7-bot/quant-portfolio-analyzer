"""
Bot 4 — Modular Multi-Strategy Signal Generator (`src/multi_bot_engine/strategy_bot.py`).

Generates independent paper-trading allocation proposals across 5 sub-strategies
(Quality/Value Stocks, Core ETF, Dividend Income, Dual-Momentum Swing, and
Statistical Mean-Reversion including Crypto Spot). Has ZERO direct access to `execution_bot`.
Not Investment Advice (AMF / MiFID II Compliant).
"""

from __future__ import annotations

from typing import Any

from src.multi_bot_engine.shared_infrastructure import PAPER_TRADING_ONLY, PaperPortfolioLedger


def generate_strategy_proposals(
    scores: dict[str, dict[str, Any]],
    snapshot: dict[str, Any],
    ledger: PaperPortfolioLedger,
    estimated_roundtrip_cost: float = 0.0014,
) -> list[dict[str, Any]]:
    """Produce candidate target-weight proposals; every proposal must pass net-of-fee edge."""
    assert PAPER_TRADING_ONLY
    proposals: list[dict[str, Any]] = []

    for sym, sc in sorted(scores.items()):
        ac = sc["asset_class"]
        comp = sc["composite_score"]
        vol = max(sc["annual_vol"], 0.04)
        mom = sc["mom_20"]
        z = sc["zscore_20"]

        if ac == "etf":
            sub_strategy = "core_etf_strategic"
            raw_w = 0.18 if comp >= 55 else 0.10
            expected_edge = 0.008
        elif ac == "stock":
            if comp >= 68 and sc["above_sma_20"]:
                sub_strategy = "quality_value_equity"
                raw_w = 0.14
                expected_edge = 0.012
            elif mom > 0.01:
                sub_strategy = "dual_momentum_swing"
                raw_w = 0.10
                expected_edge = 0.009
            else:
                sub_strategy = "dividend_defensive_hold"
                raw_w = 0.06
                expected_edge = 0.005
        else:
            # Crypto spot mean-reversion / volatility-capped sleeve
            sub_strategy = "crypto_spot_stat_reversion"
            if z <= -0.8 and vol < 0.45:
                raw_w = 0.05
                expected_edge = 0.015
            elif sc["above_sma_20"] and vol < 0.40:
                raw_w = 0.04
                expected_edge = 0.010
            else:
                raw_w = 0.02
                expected_edge = 0.004

        if expected_edge <= estimated_roundtrip_cost:
            continue

        proposals.append(
            {
                "symbol": sym,
                "asset_class": ac,
                "sub_strategy": sub_strategy,
                "proposed_weight": round(raw_w, 6),
                "annual_vol": vol,
                "expected_edge": expected_edge,
                "composite_score": comp,
            }
        )

    ledger.record_audit(
        "strategy_bot",
        "proposals_generated",
        {"proposals_count": len(proposals), "macro_regime": snapshot.get("macro_regime")},
    )
    return proposals
