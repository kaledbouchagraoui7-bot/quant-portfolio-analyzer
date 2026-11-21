"""Unit tests for trading bots and institutional risk execution engine (`tests/test_trading_bots.py`)."""

from __future__ import annotations

import math

from src.risk_execution_engine import (
    MAX_LEVERAGE,
    MAX_RISK_PER_TRADE,
    PAPER_TRADING_ONLY,
    ZERO_REAL_MONEY_RISK,
    apply_slippage_and_commission,
    atr_volatility_target_weight,
    cap_per_trade_risk,
    check_drawdown_circuit_breaker,
    fractional_kelly_fraction,
)
from src.trading_bots.institutional_momentum_allocator import (
    compute_dual_momentum_weights,
)
from src.trading_bots.mean_reversion_stat_bot import generate_mean_reversion_signals
from src.trading_bots.regime_trend_bot import (
    detect_market_regime,
    generate_regime_trend_signals,
)


def test_paper_trading_and_leverage_constants() -> None:
    assert PAPER_TRADING_ONLY is True
    assert ZERO_REAL_MONEY_RISK is True
    assert MAX_LEVERAGE == 1.0
    assert MAX_RISK_PER_TRADE == 0.01


def test_risk_execution_sizing_and_circuit_breakers() -> None:
    # Fractional Kelly clamps to [0, min(max_cap, MAX_LEVERAGE)]
    assert fractional_kelly_fraction(win_rate=0.55, payoff_ratio=1.8, fraction=0.25) > 0.0
    assert fractional_kelly_fraction(win_rate=0.30, payoff_ratio=1.0, fraction=0.25) == 0.0

    # ATR volatility targeting
    assert atr_volatility_target_weight(0.20, target_annual_vol=0.10) == 0.5
    assert atr_volatility_target_weight(0.05, target_annual_vol=0.10) == 1.0

    # Per-trade risk cap (<= 1% equity at risk)
    w_stop = cap_per_trade_risk(stop_distance_pct=0.04, max_risk_pct=0.01)
    assert math.isclose(w_stop, 0.25, rel_tol=1e-5)

    # Circuit breaker states
    cb_ok = check_drawdown_circuit_breaker(current_drawdown=0.02, daily_return=0.005)
    assert not cb_ok.halted and cb_ok.position_scale == 1.0

    cb_half = check_drawdown_circuit_breaker(current_drawdown=0.07, max_dd_limit=0.12)
    assert not cb_half.halted and cb_half.position_scale == 0.5

    cb_halt = check_drawdown_circuit_breaker(current_drawdown=0.13, max_dd_limit=0.12)
    assert cb_halt.halted and cb_halt.position_scale == 0.0

    net = apply_slippage_and_commission(gross_return=0.01, turnover=0.5, slippage_bps=5.0, commission_bps=2.0)
    assert net < 0.01


def test_regime_trend_bot_signals_and_regimes() -> None:
    prices = [100.0 * (1.003**i) for i in range(90)]
    regimes = detect_market_regime(prices)
    assert len(regimes) == len(prices)
    assert "trending" in regimes

    out = generate_regime_trend_signals(prices)
    assert out["paper_trading_only"] is True
    assert len(out["weights"]) == len(prices)
    assert all(0.0 <= w <= MAX_LEVERAGE for w in out["weights"])
    assert any(w > 0.0 for w in out["weights"])


def test_mean_reversion_stat_bot_and_shock_filter() -> None:
    prices = [100.0 + (3.0 * math.sin(i / 2.5)) for i in range(80)]
    prices[45] = 91.5  # Oversold dip in normal volatility
    out = generate_mean_reversion_signals(prices, lookback=15, entry_z=-1.2, shock_vol_ceiling=0.45)
    assert out["paper_trading_only"] is True
    assert all(0.0 <= w <= MAX_LEVERAGE for w in out["weights"])
    assert any(w > 0.0 for w in out["weights"])

    # Shock ceiling of 0.0001 forces all positions to 0
    halted = generate_mean_reversion_signals(prices, lookback=15, shock_vol_ceiling=0.0001)
    assert all(w == 0.0 for w in halted["weights"])


def test_institutional_momentum_allocator_weights() -> None:
    asset_prices = {
        "equities_index": [100.0 * (1.004**i) for i in range(60)],
        "gold_bullion": [100.0 * (1.002**i) for i in range(60)],
        "sovereign_bonds": [100.0 * (1.0008**i) for i in range(60)],
        "cash_bills": [100.0 * (1.0002**i) for i in range(60)],
    }
    res = compute_dual_momentum_weights(asset_prices, lookback=12)
    assert res["paper_trading_only"] is True
    history = res["weights_history"]
    assert len(history) == 60
    for row in history:
        assert sum(row.values()) <= MAX_LEVERAGE + 1e-5
    assert history[-1]["equities_index"] > 0.0
    assert history[-1]["gold_bullion"] > 0.0

    # Absolute Cash Shield activates when risk assets fall below trend/risk-free rate
    bear_prices = {
        "equities_index": [100.0 * (0.996**i) for i in range(60)],
        "gold_bullion": [100.0 * (0.998**i) for i in range(60)],
        "sovereign_bonds": [100.0 * (0.999**i) for i in range(60)],
        "cash_bills": [100.0 * (1.0002**i) for i in range(60)],
    }
    bear_res = compute_dual_momentum_weights(bear_prices, lookback=12)
    assert bear_res["weights_history"][-1]["cash_bills"] == 1.0
