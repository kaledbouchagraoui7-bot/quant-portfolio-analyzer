"""Unit tests for Market Data Pipeline & Independent Quantitative Validator."""

from __future__ import annotations

import math

import pytest

from src.independent_validator import (
    ExperimentTrialRegistry,
    StrategyLifecycleState,
    audit_momentum_allocator_claims,
    evaluate_tiered_kill_switch,
    transition_strategy_lifecycle,
)
from src.market_data_pipeline import (
    fetch_free_historical_daily_ohlcv,
    prevent_category_mixing,
    register_dataset,
    validate_ohlcv_bars,
    validate_price_series,
)
from src.wealth_allocation_models import hierarchical_risk_parity_allocation


def test_market_data_validation_and_checksum_registry() -> None:
    good = {"VWCE_ETF": [100.0, 100.5, 101.2, 100.9, 101.6], "BTC_SPOT": [200.0, 202.0, 201.0, 204.0, 205.0]}
    reg = register_dataset("sample_good", good)
    assert len(reg["checksum_sha256"]) == 64
    assert reg["validation"]["valid"] is True

    bad = {"VWCE_ETF": [100.0, float("nan"), 101.0, 102.0, 103.0]}
    assert validate_price_series(bad)["valid"] is False
    with pytest.raises(ValueError):
        register_dataset("sample_bad", bad)


def test_independent_audit_execution_tiers_and_sensitivity() -> None:
    n = 252
    equities = [100.0]
    bonds = [100.0]
    gold = [100.0]
    cash = [100.0]
    for i in range(1, n):
        r_eq = 0.0018 + 0.008 * math.sin(i / 6.0) if i < 95 else (-0.0012 + 0.011 * math.cos(i / 3.0) if 145 <= i < 180 else 0.0016 + 0.006 * math.sin(i / 5.0))
        r_bd = 0.0004 + 0.003 * math.cos(i / 7.0)
        r_au = 0.0008 + 0.005 * math.sin(i / 6.0)
        r_cb = 0.00015 + 0.0008 * math.sin(i / 3.0)
        equities.append(round(equities[-1] * (1.0 + r_eq), 6))
        bonds.append(round(bonds[-1] * (1.0 + r_bd), 6))
        gold.append(round(gold[-1] * (1.0 + r_au), 6))
        cash.append(round(cash[-1] * (1.0 + r_cb), 6))

    pm = {"equities_index": equities, "sovereign_bonds": bonds, "gold_bullion": gold, "cash_bills": cash}
    rm = {k: [(s[i] / s[i - 1]) - 1.0 for i in range(1, len(s))] for k, s in pm.items()}

    audit = audit_momentum_allocator_claims(pm, rm, start_bar=60, n_trials=8)
    assert audit["raw_metrics"]["sample_size_n"] == 191
    assert "sharpe_ratio" in audit["excess_over_cash_metrics"]
    assert 0.0 <= audit["deflated_sharpe_prob"] <= 1.0
    assert len(audit["parameter_sensitivity_grid"]) == 9
    tiers = audit["execution_scenarios"]
    assert tiers["optimistic"]["cagr"] >= tiers["base"]["cagr"] >= tiers["adverse"]["cagr"]


def test_lifecycle_transitions_and_tiered_kill_switch() -> None:
    st = transition_strategy_lifecycle(StrategyLifecycleState.PROPOSED, StrategyLifecycleState.RESEARCHED)
    assert st == StrategyLifecycleState.RESEARCHED
    with pytest.raises(ValueError):
        transition_strategy_lifecycle(StrategyLifecycleState.PROPOSED, StrategyLifecycleState.ACTIVE)

    k_warn = evaluate_tiered_kill_switch(0.045)
    assert k_warn.tier == "WARNING" and k_warn.position_scale == 0.85

    k_halt = evaluate_tiered_kill_switch(0.13)
    assert k_halt.tier == "EMERGENCY_HALT" and k_halt.halted is True and k_halt.cancel_pending_orders is True

    # Cannot auto-resume after halt without explicit human reset token
    k_sticky = evaluate_tiered_kill_switch(0.01, was_halted_previously=True, human_reset_token="")
    assert k_sticky.halted is True
    k_reset = evaluate_tiered_kill_switch(0.01, was_halted_previously=True, human_reset_token="HUMAN_CONFIRMED_RESET")
    assert k_reset.halted is False and k_reset.tier == "NORMAL"


def test_dataset_categories_ohlcv_validation_and_hrp_stability(tmp_path) -> None:
    ds_syn = register_dataset(
        "syn_series",
        {"VWCE_ETF": [100.0, 100.2, 100.4, 100.3, 100.8]},
        category="SYNTHETIC_SIMULATION",
        environment="1_synthetic_deterministic",
    )
    ds_real = register_dataset(
        "real_series",
        {"VWCE_ETF": [100.0, 100.1, 100.3, 100.5, 100.7]},
        category="HISTORICAL_REAL_MARKET_DATA",
        environment="2_historical_backtest",
    )
    with pytest.raises(ValueError):
        prevent_category_mixing([ds_syn, ds_real])

    bad_ohlcv = validate_ohlcv_bars(
        ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-02", "2026-01-05"],
        [100.0, 101.0, 102.0, 101.5, 103.0],
    )
    assert bad_ohlcv["valid"] is False
    assert any("duplicate_timestamp" in e for e in bad_ohlcv["errors"])

    # Offline CSV fallback tags USER_SUPPLIED_DATA instead of inventing real-market claim
    csv_p = tmp_path / "sample.csv"
    csv_p.write_text(
        "date,close\n2026-01-01,100\n2026-01-02,101\n2026-01-03,102\n2026-01-04,101.5\n2026-01-05,103\n",
        encoding="utf-8",
    )
    hist = fetch_free_historical_daily_ohlcv("SAMPLE", stooq_symbol="invalid.offline.symbol", timeout=1, sample_csv_path=csv_p)
    assert hist["validation"]["valid"] is True
    assert hist["survivorship_bias_free"] is False
    assert len(hist["checksum_sha256"]) == 64

    # Experiment trial registry + singular covariance HRP numerical stability
    reg = ExperimentTrialRegistry(tmp_path / "trials.json")
    cnt = reg.record_trials([{"signature": "s1:a"}, {"signature": "s2:b"}, {"signature": "s1:a"}])
    assert cnt == 2

    singular = {
        "ASSET_A": [0.01 * math.sin(i) for i in range(25)],
        "ASSET_A_DUP": [0.01 * math.sin(i) for i in range(25)],
        "FLAT": [0.0 for _ in range(25)],
    }
    w = hierarchical_risk_parity_allocation(singular)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-4)
    assert all(math.isfinite(val) and val >= 0.0 for val in w.values())
