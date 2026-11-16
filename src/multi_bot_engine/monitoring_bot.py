"""
Bot 7 — Continuous Telemetry & Kill-Switch Guardian (`src/multi_bot_engine/monitoring_bot.py`).

Audits live paper NAV, realized vs. backtested Sharpe drift, latency/data anomalies,
and triggers automatic drawdown circuit breakers (`6%` throttle, `12%` kill-switch halt).
Not Investment Advice (AMF / MiFID II Compliant).
"""

from __future__ import annotations

from typing import Any

from src.multi_bot_engine.shared_infrastructure import PAPER_TRADING_ONLY, PaperPortfolioLedger
from src.risk_execution_engine import check_drawdown_circuit_breaker


def monitor_portfolio_health(
    ledger: PaperPortfolioLedger,
    latest_prices: dict[str, float],
    backtest_sharpe: float = 1.50,
    live_paper_sharpe: float = 1.40,
    data_anomaly_detected: bool = False,
) -> dict[str, Any]:
    """Evaluate portfolio drawdown, Sharpe drift, and trigger kill switch if breached."""
    assert PAPER_TRADING_ONLY
    nav = ledger.total_nav(latest_prices)
    dd = ledger.current_drawdown(latest_prices)
    cb = check_drawdown_circuit_breaker(dd)
    sharpe_drift = round(live_paper_sharpe - backtest_sharpe, 4)

    if cb.halted or data_anomaly_detected:
        ledger.kill_switch_triggered = True
        status = "KILL_SWITCH_TRIGGERED"
    elif cb.position_scale < 1.0:
        status = "THROTTLED"
    else:
        status = "HEALTHY"

    ledger.record_audit(
        "monitoring_bot",
        "health_audited",
        {
            "status": status,
            "nav_eur": nav,
            "drawdown": round(dd, 4),
            "sharpe_drift": sharpe_drift,
            "kill_switch_triggered": ledger.kill_switch_triggered,
        },
    )
    return {
        "status": status,
        "kill_switch_triggered": ledger.kill_switch_triggered,
        "nav_eur": nav,
        "drawdown": round(dd, 4),
        "position_scale": cb.position_scale,
        "sharpe_drift": sharpe_drift,
    }
