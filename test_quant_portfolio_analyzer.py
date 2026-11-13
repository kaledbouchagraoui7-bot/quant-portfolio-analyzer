#!/usr/bin/env python3
"""Automated pytest verification for quant_portfolio_analyzer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from quant_portfolio_analyzer import (
    DISCLAIMER,
    analyze_price_series,
    compute_kelly_fraction,
    compute_max_drawdown,
    format_markdown_report,
    load_price_series,
)


def test_max_drawdown_detects_exact_peak_to_trough() -> None:
    labels = ["D1", "D2", "D3", "D4", "D5"]
    prices = [100.0, 120.0, 90.0, 110.0, 130.0]
    dd, peak, trough = compute_max_drawdown(prices, labels)
    assert dd == pytest.approx(-0.25, rel=1e-6)
    assert peak == "D2"
    assert trough == "D3"


def test_analyze_price_series_computes_cagr_sharpe_sortino() -> None:
    # Monthly data over 12 transitions = 1.0 year: 100 -> 112.0 (+12% CAGR)
    prices = [100.0, 102.0, 101.0, 104.0, 103.5, 106.0, 105.0, 108.0, 109.5, 108.5, 110.0, 111.0, 112.0]
    m = analyze_price_series(prices, periods_per_year=12, risk_free_rate=0.02)
    assert m.observations == 13
    assert m.total_return_pct == pytest.approx(12.0, abs=0.05)
    assert m.cagr_pct == pytest.approx(12.0, abs=0.05)
    assert m.sharpe_ratio > 1.5
    assert m.sortino_ratio > m.sharpe_ratio
    assert DISCLAIMER in m.disclaimer


def test_kelly_criterion_respects_floor_and_safety_cap() -> None:
    full_k, half_k = compute_kelly_fraction(win_rate=0.60, payoff_ratio=2.0, max_cap=0.25)
    assert full_k == pytest.approx(0.25)
    assert half_k == pytest.approx(0.20)
    zero_full, zero_half = compute_kelly_fraction(win_rate=0.20, payoff_ratio=1.0)
    assert zero_full == 0.0
    assert zero_half == 0.0


def test_load_csv_and_cli_json_output(tmp_path: Path) -> None:
    csv_file = tmp_path / "prices.csv"
    csv_file.write_text("date,close\n2026-01-01,100\n2026-02-01,105\n2026-03-01,102\n2026-04-01,110\n", encoding="utf-8")
    labels, prices = load_price_series(csv_file)
    assert labels == ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]
    assert prices == [100.0, 105.0, 102.0, 110.0]

    proc = subprocess.run(
        [sys.executable, "quant_portfolio_analyzer.py", "--input", str(csv_file), "--periods-per-year", "12", "--format", "json"],
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["total_return_pct"] == pytest.approx(10.0)
    assert "not certified financial advice" in payload["disclaimer"].lower()


def test_markdown_report_and_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        analyze_price_series([100.0])
    with pytest.raises(ValueError):
        analyze_price_series([100.0, -5.0])
    m = analyze_price_series([100.0, 110.0, 105.0, 115.0], periods_per_year=12)
    md = format_markdown_report(m, "demo.csv")
    assert "Sharpe Ratio" in md
    assert "Sortino Ratio" in md
    assert DISCLAIMER in md
