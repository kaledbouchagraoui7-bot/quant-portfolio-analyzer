"""Unit tests for Market Data Pipeline & Independent Quantitative Validator."""

from __future__ import annotations

import math

import pytest

from src.independent_validator import (
    StrategyLifecycleState,
    audit_momentum_allocator_claims,
    evaluate_tiered_kill_switch,
    transition_strategy_lifecycle,
)
from src.market_data_pipeline import register_dataset, validate_price_series


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
