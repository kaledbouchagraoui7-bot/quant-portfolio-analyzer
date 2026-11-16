"""Unit tests for the 7-Bot Institutional Architecture (`src/multi_bot_engine/`)."""

from __future__ import annotations

import pytest

from src.multi_bot_engine.execution_bot import execute_paper_orders
from src.multi_bot_engine.monitoring_bot import monitor_portfolio_health
from src.multi_bot_engine.orchestrator_bot import run_seven_bot_cycle
from src.multi_bot_engine.risk_management_bot import ApprovedOrder, validate_and_approve_orders
from src.multi_bot_engine.shared_infrastructure import (
    AGE_18_MONETIZATION_GATE,
    LIVE_BROKER_ENABLED,
    MAX_LEVERAGE,
    PAPER_TRADING_ONLY,
    ZERO_REAL_MONEY_RISK,
    PaperPortfolioLedger,
    build_synthetic_multi_asset_prices,
)


def test_seven_bot_full_cycle_and_alcoa_audit_trail() -> None:
    res = run_seven_bot_cycle(initial_capital_eur=10000.0)
    assert res["age_18_monetization_gate"] is True
    assert res["paper_trading_only"] is True
    assert res["zero_real_money_risk"] is True
    assert res["live_broker_enabled"] is False
    assert res["max_leverage"] == 1.0
    assert res["assets_analyzed"] == 7
    assert res["approved_count"] >= 5
    assert res["executed_count"] >= 5
    assert res["audit_trail_count"] >= 7
    assert res["health"]["status"] == "HEALTHY"
    # Multi-asset coverage: ETFs, Stocks, Crypto spot
    assert "VWCE_ETF" in res["positions"]
    assert "NOVO_STOCK" in res["positions"]
    assert "BTC_SPOT" in res["positions"]


def test_risk_management_cannot_be_bypassed_and_vetoes_leverage() -> None:
    ledger = PaperPortfolioLedger(initial_capital_eur=10000.0, cash_eur=10000.0, peak_nav_eur=10000.0)
    prices = {"VWCE_ETF": 100.0, "BTC_SPOT": 100.0}

    # 1. Direct call to execution_bot without ApprovedOrder -> PermissionError
    with pytest.raises(PermissionError):
        execute_paper_orders([{"symbol": "VWCE_ETF", "target_weight": 0.5}], prices, ledger)  # type: ignore[list-item]

    # 2. Forged ApprovalToken -> PermissionError
    forged = ApprovedOrder(
        symbol="VWCE_ETF",
        asset_class="etf",
        target_weight=0.20,
        notional_eur=2000.0,
        approval_token="forged_token_1234",
        sub_strategy="bypass_attempt",
    )
    with pytest.raises(PermissionError):
        execute_paper_orders([forged], prices, ledger)

    # 3. Leveraged (> 1.0) or excessive crypto (> 10%) proposals are vetoed/capped
    proposals = [
        {"symbol": "VWCE_ETF", "asset_class": "etf", "proposed_weight": 0.50, "leverage": 2.0, "annual_vol": 0.12},
        {"symbol": "BTC_SPOT", "asset_class": "crypto", "proposed_weight": 0.20, "leverage": 1.0, "annual_vol": 0.30},
        {"symbol": "ETH_SPOT", "asset_class": "crypto", "proposed_weight": 0.20, "leverage": 1.0, "annual_vol": 0.30},
    ]
    risk_out = validate_and_approve_orders(proposals, prices, ledger, max_crypto_total_weight=0.10)
    assert any(v["symbol"] == "VWCE_ETF" and "leverage" in v["veto_reason"] for v in risk_out["vetoed_orders"])
    total_crypto_w = sum(o.target_weight for o in risk_out["approved_orders"] if o.asset_class == "crypto")
    assert total_crypto_w <= 0.10 + 1e-6


def test_monitoring_bot_triggers_kill_switch_on_drawdown_or_anomaly() -> None:
    ledger = PaperPortfolioLedger(initial_capital_eur=10000.0, cash_eur=8500.0, peak_nav_eur=10000.0)
    health = monitor_portfolio_health(ledger, {"VWCE_ETF": 100.0})
    assert health["status"] == "KILL_SWITCH_TRIGGERED"
    assert health["kill_switch_triggered"] is True
    assert ledger.kill_switch_triggered is True

    # Verify Risk Management vetoes all orders once kill switch is active
    risk_out = validate_and_approve_orders(
        [{"symbol": "VWCE_ETF", "asset_class": "etf", "proposed_weight": 0.10, "annual_vol": 0.10}],
        {"VWCE_ETF": 100.0},
        ledger,
    )
    assert len(risk_out["approved_orders"]) == 0
    assert len(risk_out["vetoed_orders"]) == 1
