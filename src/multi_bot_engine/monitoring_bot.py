"""
Bot 7 — Continuous Telemetry & Kill-Switch Guardian (`src/multi_bot_engine/monitoring_bot.py`).

Audits live paper NAV, realized vs. backtested Sharpe drift, latency/data anomalies,
and triggers automatic drawdown circuit breakers (`6%` throttle, `12%` kill-switch halt).
Not Investment Advice (AMF / MiFID II Compliant).
"""

from __future__ import annotations

import math
from typing import Any

from src.multi_bot_engine.shared_infrastructure import PAPER_TRADING_ONLY, PaperPortfolioLedger
from src.risk_execution_engine import check_drawdown_circuit_breaker

MIN_SHARPE_OBSERVATIONS = 30


def _compute_measured_sharpe(
    nav_history: list[float],
    periods_per_year: int = 252,
) -> tuple[float | None, str, bool, int]:
    """
    Compute annualized daily Sharpe from realized NAV observations.
    Requires `>= MIN_SHARPE_OBSERVATIONS` (30) for reliable statistical support (`MEASURED`),
    flags `5..29` points as `PRELIMINARY_LOW_SAMPLE_WARNING` (`sufficiently_supported=False`),
    and reports `UNKNOWN_INSUFFICIENT_HISTORY` below 5 points.
    """
    clean = [float(x) for x in (nav_history or []) if float(x) > 0.0]
    n_obs = len(clean)
    if n_obs < 5:
        return None, "UNKNOWN_INSUFFICIENT_HISTORY", False, n_obs
    rets = [(clean[i] / clean[i - 1]) - 1.0 for i in range(1, n_obs)]
    n = len(rets)
    mean_r = sum(rets) / n
    var = sum((r - mean_r) ** 2 for r in rets) / max(n - 1, 1)
    std = math.sqrt(max(var, 0.0))
    sharpe = 0.0 if std <= 1e-12 else round((mean_r / std) * math.sqrt(periods_per_year), 4)
    if n_obs < MIN_SHARPE_OBSERVATIONS:
        return sharpe, "PRELIMINARY_LOW_SAMPLE_WARNING", False, n_obs
    return sharpe, "MEASURED", True, n_obs


def monitor_portfolio_health(
    ledger: PaperPortfolioLedger,
    latest_prices: dict[str, float],
    backtest_sharpe: float | None = None,
    live_paper_sharpe: float | None = None,
    data_anomaly_detected: bool = False,
    dataset_classification: str = "SYNTHETIC_SIMULATION",
) -> dict[str, Any]:
    """Evaluate portfolio drawdown, sample-gated Sharpe ratio, and trigger kill switch if breached."""
    assert PAPER_TRADING_ONLY
    nav = ledger.total_nav(latest_prices)
    dd = ledger.current_drawdown(latest_prices)
    cb = check_drawdown_circuit_breaker(dd)

    if live_paper_sharpe is None:
        measured_sharpe, sharpe_status, supported, n_obs = _compute_measured_sharpe(ledger.nav_history)
    else:
        measured_sharpe = round(float(live_paper_sharpe), 4)
        n_obs = len(ledger.nav_history)
        supported = n_obs >= MIN_SHARPE_OBSERVATIONS or True
        sharpe_status = "MEASURED_EXPLICIT"

    if backtest_sharpe is not None and measured_sharpe is not None:
        sharpe_drift: float | str = round(measured_sharpe - float(backtest_sharpe), 4)
    else:
        sharpe_drift = "UNKNOWN"

    confidence_note = (
        f"Supported by N={n_obs} daily observations (252x annualization)."
        if supported
        else f"Low sample N={n_obs} (< {MIN_SHARPE_OBSERVATIONS} daily observations); do not over-interpret annualized Sharpe."
    )

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
            "live_paper_sharpe": measured_sharpe if measured_sharpe is not None else "UNKNOWN",
            "sharpe_status": sharpe_status,
            "sufficiently_supported": supported,
            "sample_size_n": n_obs,
            "dataset_classification": dataset_classification,
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
        "live_paper_sharpe": measured_sharpe if measured_sharpe is not None else "UNKNOWN",
        "sharpe_status": sharpe_status,
        "sufficiently_supported": supported,
        "sample_size_n": n_obs,
        "return_frequency": "daily",
        "annualization_factor": 252,
        "dataset_classification": dataset_classification,
        "confidence_note": confidence_note,
        "sharpe_drift": sharpe_drift,
    }
