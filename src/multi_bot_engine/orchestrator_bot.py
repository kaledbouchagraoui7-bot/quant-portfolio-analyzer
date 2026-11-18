"""
Bot 1 — Orchestrator Bot (`src/multi_bot_engine/orchestrator_bot.py`).

Manager & Scheduler coordinating Bots 2–7 in strict sequence across Stocks, ETFs, and Crypto:
  Bot 2 (`market_research_bot`) -> Bot 3 (`analysis_bot`) -> Bot 4 (`strategy_bot`)
  -> Bot 5 (`risk_management_bot`) -> Bot 6 (`execution_bot`) -> Bot 7 (`monitoring_bot`).

Open-Source Quantitative Research & Risk-Analytics Software — Not Investment Advice
(AMF / MiFID II Compliant).
"""

from __future__ import annotations

from typing import Any

from src.multi_bot_engine.analysis_bot import evaluate_assets
from src.multi_bot_engine.execution_bot import execute_paper_orders
from src.multi_bot_engine.market_research_bot import collect_market_snapshot
from src.multi_bot_engine.monitoring_bot import monitor_portfolio_health
from src.multi_bot_engine.risk_management_bot import validate_and_approve_orders
from src.multi_bot_engine.shared_infrastructure import (
    AGE_18_MONETIZATION_GATE,
    LEGAL_NOTICE,
    LIVE_BROKER_ENABLED,
    MAX_LEVERAGE,
    PAPER_TRADING_ONLY,
    ZERO_REAL_MONEY_RISK,
    PaperPortfolioLedger,
    build_synthetic_multi_asset_prices,
)
from src.multi_bot_engine.strategy_bot import generate_strategy_proposals


def run_seven_bot_cycle(
    price_series: dict[str, list[float]] | None = None,
    ledger: PaperPortfolioLedger | None = None,
    initial_capital_eur: float = 10000.0,
) -> dict[str, Any]:
    """Execute one complete decoupled 7-bot paper-trading & risk-governance cycle."""
    assert PAPER_TRADING_ONLY and ZERO_REAL_MONEY_RISK and (LIVE_BROKER_ENABLED is False)

    active_ledger = ledger or PaperPortfolioLedger(initial_capital_eur=initial_capital_eur, cash_eur=initial_capital_eur, peak_nav_eur=initial_capital_eur)
    prices = price_series or build_synthetic_multi_asset_prices(bars=120)

    # Bot 2: Market Research
    snapshot = collect_market_snapshot(prices, active_ledger)
    # Bot 3: Fundamental / ETF / Trend Analysis
    scores = evaluate_assets(snapshot, active_ledger)
    # Bot 4: Multi-Strategy Signal Proposals
    proposals = generate_strategy_proposals(scores, snapshot, active_ledger)
    # Bot 5: Hard Veto Risk Management Gatekeeper
    risk_out = validate_and_approve_orders(proposals, snapshot["latest_prices"], active_ledger)
    # Bot 6: Simulated Paper-Only Order Execution
    exec_out = execute_paper_orders(risk_out["approved_orders"], snapshot["latest_prices"], active_ledger)
    active_ledger.record_nav(snapshot["latest_prices"])
    # Bot 7: Telemetry & Kill-Switch Guardian
    health = monitor_portfolio_health(active_ledger, snapshot["latest_prices"])

    active_ledger.record_audit(
        "orchestrator_bot",
        "cycle_completed",
        {"nav_eur": health["nav_eur"], "status": health["status"]},
    )

    return {
        "legal_notice": LEGAL_NOTICE,
        "age_18_monetization_gate": AGE_18_MONETIZATION_GATE,
        "paper_trading_only": PAPER_TRADING_ONLY,
        "zero_real_money_risk": ZERO_REAL_MONEY_RISK,
        "live_broker_enabled": LIVE_BROKER_ENABLED,
        "max_leverage": MAX_LEVERAGE,
        "macro_regime": snapshot["macro_regime"],
        "assets_analyzed": len(scores),
        "proposals_count": len(proposals),
        "approved_count": len(risk_out["approved_orders"]),
        "vetoed_count": len(risk_out["vetoed_orders"]),
        "executed_count": len(exec_out["executed_orders"]),
        "total_fees_eur": exec_out["total_fees_eur"],
        "health": health,
        "positions": dict(active_ledger.positions),
        "audit_trail_count": len(active_ledger.audit_trail),
    }
