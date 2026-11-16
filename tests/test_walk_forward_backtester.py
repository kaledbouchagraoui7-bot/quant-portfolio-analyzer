"""Unit tests for walk-forward backtester and institutional wealth allocation models."""

from __future__ import annotations

import math

from src.trading_bots.mean_reversion_stat_bot import generate_mean_reversion_signals
from src.trading_bots.regime_trend_bot import generate_regime_trend_signals
from src.walk_forward_backtester import (
    compute_audited_metrics,
    monte_carlo_drawdown_ci,
    run_walk_forward_backtest,
)
from src.wealth_allocation_models import (
    compare_allocation_models,
    equal_weight_allocation,
    fractional_kelly_portfolio_allocation,
    hierarchical_risk_parity_allocation,
    mean_variance_sharpe_allocation,
    risk_parity_all_weather_allocation,
)


def test_walk_forward_backtester_and_metrics() -> None:
    prices = [100.0 * (1.0025**i) * (1.0 + 0.008 * math.sin(i / 4.0)) for i in range(180)]
    wf_trend = run_walk_forward_backtest(
        prices=prices,
        signal_fn=generate_regime_trend_signals,
        train_size=60,
        test_size=30,
    )
    assert wf_trend["paper_trading_only"] is True
    assert wf_trend["zero_real_money_risk"] is True
    assert wf_trend["walk_forward_folds"] >= 3
    assert wf_trend["sample_size_n"] == 120
    assert "sharpe_ratio" in wf_trend
    assert "sortino_ratio" in wf_trend
    assert "calmar_ratio" in wf_trend
    assert "mc_p95_max_dd" in wf_trend

    wf_mr = run_walk_forward_backtest(
        prices=prices,
        signal_fn=generate_mean_reversion_signals,
        train_size=60,
        test_size=30,
    )
    assert wf_mr["sample_size_n"] == 120

    mc = monte_carlo_drawdown_ci([0.01, -0.005, 0.008, -0.004, 0.012], iterations=50)
    assert 0.0 <= mc["mc_median_max_dd"] <= mc["mc_p95_max_dd"]


def test_wealth_allocation_models_comparison() -> None:
    return_matrix = {
        "equities": [0.0012 + 0.012 * math.sin(i) for i in range(120)],
        "bonds": [0.0004 + 0.004 * math.cos(i) for i in range(120)],
        "gold": [0.0006 + 0.007 * math.sin(i / 2.0) for i in range(120)],
        "cash": [0.00015 for _ in range(120)],
    }
    for fn in (
        equal_weight_allocation,
        risk_parity_all_weather_allocation,
        mean_variance_sharpe_allocation,
        hierarchical_risk_parity_allocation,
        fractional_kelly_portfolio_allocation,
    ):
        w = fn(return_matrix)
        assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-4)
        assert all(v >= 0.0 for v in w.values())

    comp = compare_allocation_models(return_matrix)
    assert set(comp.keys()) == {
        "equal_weight",
        "mean_variance_markowitz",
        "risk_parity_all_weather",
        "hierarchical_risk_parity_hrp",
        "fractional_kelly_thorp",
    }
    for model_data in comp.values():
        assert model_data["metrics"]["sample_size_n"] == 120
        assert "cagr" in model_data["metrics"]
