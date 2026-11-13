#!/usr/bin/env python3
"""Comprehensive pytest suite (>=10 tests) for quant_portfolio_analyzer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from quant_portfolio_analyzer import (
    DISCLAIMER,
    analyze_price_series,
    compare_lump_sum_vs_dca,
    compute_expected_shortfall,
    compute_inverse_volatility_weights,
    compute_kelly_fraction,
    compute_max_drawdown,
    compute_var,
    format_markdown_report,
    load_multi_asset_prices,
    load_price_series,
)


def test_max_drawdown_detects_exact_peak_to_trough() -> None:
    labels = ["D1", "D2", "D3", "D4", "D5"]
    prices = [100.0, 120.0, 90.0, 110.0, 130.0]
    dd, peak, trough = compute_max_drawdown(prices, labels)
    assert dd == pytest.approx(-0.25, rel=1e-6)
    assert peak == "D2"
    assert trough == "D3"


def test_analyze_price_series_computes_cagr_sharpe_sortino_and_var() -> None:
    prices = [100.0, 102.0, 98.0, 104.0, 101.0, 106.0, 103.5, 108.0, 109.5, 107.0, 110.0, 111.0, 112.0]
    m = analyze_price_series(prices, periods_per_year=12, risk_free_rate=0.02)
    assert m.observations == 13
    assert m.total_return_pct == pytest.approx(12.0, abs=0.05)
    assert m.cagr_pct == pytest.approx(12.0, abs=0.05)
    assert m.var_95_hist_pct >= 0.0
    assert m.var_99_hist_pct >= m.var_95_hist_pct
    assert m.cvar_95_hist_pct >= m.var_95_hist_pct
    assert m.var_99_param_pct >= m.var_95_param_pct
    assert DISCLAIMER in m.disclaimer


def test_historical_and_parametric_var_and_cvar_accuracy() -> None:
    # Known returns: worst drop -10%, median ~0%
    returns = [-0.10, -0.04, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.02]
    v95 = compute_var(returns, confidence=0.95, method="historical")
    v99 = compute_var(returns, confidence=0.99, method="historical")
    v95_norm = compute_var(returns, confidence=0.95, method="normal")
    cvar95 = compute_expected_shortfall(returns, confidence=0.95)
    assert v95 > 0.04
    assert v99 >= v95
    assert v95_norm > 0.0
    assert cvar95 >= v95


def test_var_rejects_bad_confidence_and_method() -> None:
    with pytest.raises(ValueError):
        compute_var([-0.02, 0.01], confidence=1.5)
    with pytest.raises(ValueError):
        compute_var([-0.02, 0.01], confidence=0.95, method="unknown")
    with pytest.raises(ValueError):
        compute_expected_shortfall([], confidence=0.95)


def test_lump_sum_vs_dca_dip_scenario_favors_dca_drawdown() -> None:
    # Mid-period crash before recovery -> DCA buys cheaper & suffers smaller drawdown
    prices = [100.0, 90.0, 75.0, 60.0, 80.0, 100.0, 115.0]
    sim = compare_lump_sum_vs_dca(prices, budget=7000.0, installments=7)
    assert sim["budget"] == 7000.0
    assert sim["installments_used"] == 7
    assert sim["dca_final"] > sim["lump_sum_final"]
    assert abs(sim["dca_max_drawdown_pct"]) < abs(sim["lump_sum_max_drawdown_pct"])
    assert sim["winner_by_final_value"] == "dca"
    assert DISCLAIMER in sim["disclaimer"]


def test_lump_sum_vs_dca_bull_trend_favors_lump_sum() -> None:
    prices = [100.0, 105.0, 110.0, 116.0, 122.0, 128.0]
    sim = compare_lump_sum_vs_dca(prices, budget=6000.0, installments=6)
    assert sim["lump_sum_final"] > sim["dca_final"]
    assert sim["winner_by_final_value"] == "lump_sum"


def test_inverse_volatility_weights_lower_risk_gets_higher_weight() -> None:
    assets = {
        "BOND_LOW_VOL": [100.0, 100.5, 101.0, 101.4, 101.9, 102.3],
        "TECH_HIGH_VOL": [100.0, 112.0, 94.0, 116.0, 98.0, 120.0],
    }
    res = compute_inverse_volatility_weights(assets, periods_per_year=12)
    weights = res["weights"]
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)
    assert weights["BOND_LOW_VOL"] > weights["TECH_HIGH_VOL"]
    assert res["basket_metrics"]["observations"] == 6
    assert DISCLAIMER in res["disclaimer"]


def test_multi_asset_csv_loading_and_validation(tmp_path: Path) -> None:
    csv_path = tmp_path / "basket.csv"
    csv_path.write_text(
        "date,SPY,TLT\n2026-01,100,100\n2026-02,103,101\n2026-03,99,102\n2026-04,106,102.5\n",
        encoding="utf-8",
    )
    labels, assets = load_multi_asset_prices(csv_path)
    assert len(labels) == 4
    assert set(assets.keys()) == {"SPY", "TLT"}
    res = compute_inverse_volatility_weights(assets, periods_per_year=12)
    assert res["weights"]["TLT"] > res["weights"]["SPY"]


def test_kelly_criterion_respects_floor_and_safety_cap() -> None:
    full_k, half_k = compute_kelly_fraction(win_rate=0.60, payoff_ratio=2.0, max_cap=0.25)
    assert full_k == pytest.approx(0.25)
    assert half_k == pytest.approx(0.20)
    zero_full, zero_half = compute_kelly_fraction(win_rate=0.20, payoff_ratio=1.0)
    assert zero_full == 0.0
    assert zero_half == 0.0


def test_cli_json_and_multi_asset_markdown_outputs(tmp_path: Path) -> None:
    csv_file = tmp_path / "prices.csv"
    csv_file.write_text(
        "date,SPY,QQQ,TLT\n2026-01,100,100,100\n2026-02,104,107,101\n2026-03,101,98,102\n2026-04,108,112,102.2\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "quant_portfolio_analyzer.py",
            "--input",
            str(csv_file),
            "--periods-per-year",
            "12",
            "--multi-asset",
            "--format",
            "json",
        ],
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert "metrics" in payload
    assert "dca_vs_lump_sum" in payload
    assert "multi_asset_risk_parity" in payload
    assert sum(payload["multi_asset_risk_parity"]["weights"].values()) == pytest.approx(1.0, abs=1e-5)
    assert "not certified financial advice" in payload["metrics"]["disclaimer"].lower()


def test_markdown_report_and_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        analyze_price_series([100.0])
    with pytest.raises(ValueError):
        analyze_price_series([100.0, -5.0])
    m = analyze_price_series([100.0, 110.0, 105.0, 115.0], periods_per_year=12)
    md = format_markdown_report(m, "demo.csv")
    assert "Historical VaR (95% / 99%)" in md
    assert "Expected Shortfall (95%)" in md
    assert DISCLAIMER in md
