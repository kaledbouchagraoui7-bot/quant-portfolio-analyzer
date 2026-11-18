"""Unit tests for the 7-Bot Institutional Architecture (`src/multi_bot_engine/`)."""

from __future__ import annotations

import secrets
import time
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
    get_risk_gate_secret,
    sign_approval_token,
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


def test_hmac_anti_replay_expiration_and_tamper_resistance(tmp_path, monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    monkeypatch.setenv("QPA_NONCE_DB_PATH", str(tmp_path / "nonces.sqlite3"))
    monkeypatch.setenv("QPA_RISK_GATE_SECRET", "TEST_PERSISTENT_32BYTE_KEY_ABCDEF123456")

    ledger = PaperPortfolioLedger(initial_capital_eur=10000.0, cash_eur=10000.0, peak_nav_eur=10000.0)
    prices = {"VWCE_ETF": 100.0, "ROG_STOCK": 100.0}
    risk_out = validate_and_approve_orders(
        [{"symbol": "VWCE_ETF", "asset_class": "etf", "proposed_weight": 0.15, "leverage": 1.0, "annual_vol": 0.12}],
        prices,
        ledger,
    )
    order = risk_out["approved_orders"][0]
    assert order.nonce and order.expires_at_epoch > time.time() and order.quantity > 0.0

    # Altered notional, altered quantity, altered asset all fail HMAC verification
    for forged in (
        replace(order, notional_eur=order.notional_eur + 500.0),
        replace(order, quantity=order.quantity + 10.0),
        replace(order, symbol="ROG_STOCK"),
    ):
        with pytest.raises(PermissionError):
            execute_paper_orders([forged], prices, ledger)

    # Expired token rejected even with valid HMAC signature
    exp_ts = round(time.time() - 10.0, 3)
    exp_nonce = secrets.token_hex(12)
    exp_tok = sign_approval_token(
        get_risk_gate_secret(), "VWCE_ETF", "etf", 0.10, 1000.0, nonce=exp_nonce, expires_at_epoch=exp_ts
    )
    expired = ApprovedOrder(
        symbol="VWCE_ETF",
        asset_class="etf",
        target_weight=0.10,
        notional_eur=1000.0,
        approval_token=exp_tok,
        sub_strategy="test_exp",
        nonce=exp_nonce,
        expires_at_epoch=exp_ts,
    )
    with pytest.raises(PermissionError):
        execute_paper_orders([expired], prices, ledger)

    # Valid order executes once; same-process replay & cross-restart SQLite replay raise PermissionError
    res_first = execute_paper_orders([order], prices, ledger)
    assert len(res_first["executed_orders"]) == 1
    with pytest.raises(PermissionError):
        execute_paper_orders([order], prices, ledger)

    # Simulated process restart against same SQLite nonce DB file
    from src.multi_bot_engine.execution_bot import verify_risk_approval

    with pytest.raises(PermissionError):
        verify_risk_approval(order, consume=True, nonce_db_path=tmp_path / "nonces.sqlite3")

    # Concurrent multi-thread redemption: only 1 of 8 racing threads can consume a fresh approval
    fresh_order = validate_and_approve_orders(
        [{"symbol": "VWCE_ETF", "asset_class": "etf", "proposed_weight": 0.12, "leverage": 1.0, "annual_vol": 0.12}],
        prices,
        PaperPortfolioLedger(initial_capital_eur=10000.0, cash_eur=10000.0, peak_nav_eur=10000.0),
    )["approved_orders"][0]

    def _try_consume(_idx: int) -> bool:
        try:
            verify_risk_approval(fresh_order, consume=True, nonce_db_path=tmp_path / "nonces.sqlite3")
            return True
        except PermissionError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(_try_consume, range(8)))
    assert sum(1 for ok in outcomes if ok) == 1

    # Missing signing key & keystore unavailable fail closed
    fresh2 = validate_and_approve_orders(
        [{"symbol": "VWCE_ETF", "asset_class": "etf", "proposed_weight": 0.12, "leverage": 1.0, "annual_vol": 0.12}],
        prices,
        PaperPortfolioLedger(initial_capital_eur=10000.0, cash_eur=10000.0, peak_nav_eur=10000.0),
    )["approved_orders"][0]
    monkeypatch.setenv("QPA_RISK_GATE_SECRET", "__MISSING__")
    with pytest.raises(PermissionError):
        verify_risk_approval(fresh2, consume=True, nonce_db_path=tmp_path / "nonces.sqlite3")

    monkeypatch.delenv("QPA_RISK_GATE_SECRET", raising=False)
    monkeypatch.setenv("QPA_FORCE_KEYSTORE_UNAVAILABLE", "1")
    with pytest.raises(RuntimeError):
        get_risk_gate_secret(allow_ephemeral_test_fallback=False)
    monkeypatch.delenv("QPA_FORCE_KEYSTORE_UNAVAILABLE", raising=False)

    # Corrupted SQLite nonce DB fails closed
    monkeypatch.setenv("QPA_RISK_GATE_SECRET", "TEST_PERSISTENT_32BYTE_KEY_ABCDEF123456")
    corrupt_db = tmp_path / "corrupt_nonces.sqlite3"
    corrupt_db.write_bytes(b"THIS_IS_NOT_SQLITE_HEADER_DATA")
    fresh3 = validate_and_approve_orders(
        [{"symbol": "VWCE_ETF", "asset_class": "etf", "proposed_weight": 0.12, "leverage": 1.0, "annual_vol": 0.12}],
        prices,
        PaperPortfolioLedger(initial_capital_eur=10000.0, cash_eur=10000.0, peak_nav_eur=10000.0),
    )["approved_orders"][0]
    with pytest.raises(PermissionError):
        verify_risk_approval(fresh3, consume=True, nonce_db_path=corrupt_db)


def test_monitoring_bot_measures_sharpe_or_reports_unknown() -> None:
    short_ledger = PaperPortfolioLedger(initial_capital_eur=10000.0, cash_eur=10000.0, peak_nav_eur=10000.0)
    short_ledger.nav_history.extend([10000.0, 10010.0])
    h_short = monitor_portfolio_health(short_ledger, {"VWCE_ETF": 100.0})
    assert h_short["live_paper_sharpe"] == "UNKNOWN"
    assert h_short["sharpe_status"] == "UNKNOWN_INSUFFICIENT_HISTORY"
    assert h_short["sufficiently_supported"] is False

    prelim_ledger = PaperPortfolioLedger(initial_capital_eur=10000.0, cash_eur=10000.0, peak_nav_eur=10000.0)
    prelim_ledger.nav_history.extend([10000.0, 10025.0, 10040.0, 10075.0, 10095.0, 10130.0])
    h_prelim = monitor_portfolio_health(prelim_ledger, {"VWCE_ETF": 100.0})
    assert h_prelim["sharpe_status"] == "PRELIMINARY_LOW_SAMPLE_WARNING"
    assert h_prelim["sufficiently_supported"] is False

    long_ledger = PaperPortfolioLedger(initial_capital_eur=10000.0, cash_eur=10000.0, peak_nav_eur=10000.0)
    long_ledger.nav_history.extend([10000.0 + (i * 15.0) + ((i % 3) * 4.0) for i in range(32)])
    h_long = monitor_portfolio_health(long_ledger, {"VWCE_ETF": 100.0}, backtest_sharpe=1.5)
    assert h_long["sharpe_status"] == "MEASURED"
    assert h_long["sufficiently_supported"] is True
    assert h_long["sample_size_n"] == 32
    assert isinstance(h_long["live_paper_sharpe"], float)
    assert isinstance(h_long["sharpe_drift"], float)
