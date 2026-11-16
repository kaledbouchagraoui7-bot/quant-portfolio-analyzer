"""
Institutional Risk, Position-Sizing & Execution Engine (`src/risk_execution_engine.py`).

Enforces strict paper-trading & zero-debt guardrails:
  - PAPER_TRADING_ONLY = True
  - ZERO_REAL_MONEY_RISK = True
  - MAX_LEVERAGE = 1.0
  - MAX_RISK_PER_TRADE = 0.01 (<= 1% simulated equity)
"""

from __future__ import annotations

from dataclasses import dataclass

PAPER_TRADING_ONLY: bool = True
ZERO_REAL_MONEY_RISK: bool = True
MAX_LEVERAGE: float = 1.0
MAX_RISK_PER_TRADE: float = 0.01


@dataclass(frozen=True)
class CircuitBreakerState:
    halted: bool
    reason: str
    position_scale: float


def fractional_kelly_fraction(
    win_rate: float,
    payoff_ratio: float,
    fraction: float = 0.25,
    max_cap: float = 0.25,
) -> float:
    """
    Compute fractional Kelly position weight: f* = fraction * (p - (1 - p) / b),
    clamped to [0.0, min(max_cap, MAX_LEVERAGE)].
    """
    if payoff_ratio <= 0 or win_rate <= 0 or win_rate >= 1 or fraction <= 0:
        return 0.0
    raw_kelly = win_rate - ((1.0 - win_rate) / payoff_ratio)
    if raw_kelly <= 0:
        return 0.0
    sized = raw_kelly * fraction
    return round(min(sized, max_cap, MAX_LEVERAGE), 6)


def atr_volatility_target_weight(
    asset_annual_vol: float,
    target_annual_vol: float = 0.10,
    max_leverage: float = MAX_LEVERAGE,
) -> float:
    """Scale position weight inversely with realized ATR/return volatility, capped at MAX_LEVERAGE."""
    if asset_annual_vol <= 1e-8 or target_annual_vol <= 0:
        return 0.0
    cap = min(float(max_leverage), MAX_LEVERAGE)
    return round(min(target_annual_vol / asset_annual_vol, cap), 6)


def cap_per_trade_risk(
    stop_distance_pct: float,
    max_risk_pct: float = MAX_RISK_PER_TRADE,
    max_leverage: float = MAX_LEVERAGE,
) -> float:
    """
    Given a stop-loss distance (as a decimal fraction of price, e.g. 0.025 for 2.5%),
    compute the maximum position weight so a stop hit loses at most `max_risk_pct` (<= 1%) of equity.
    """
    if stop_distance_pct <= 1e-6:
        return 0.0
    allowed_risk = min(float(max_risk_pct), MAX_RISK_PER_TRADE)
    weight = allowed_risk / stop_distance_pct
    return round(min(weight, float(max_leverage), MAX_LEVERAGE), 6)


def check_drawdown_circuit_breaker(
    current_drawdown: float,
    daily_return: float = 0.0,
    max_dd_limit: float = 0.12,
    daily_loss_limit: float = 0.03,
) -> CircuitBreakerState:
    """
    Halt or de-risk paper execution when portfolio drawdown exceeds `max_dd_limit`
    or single-day loss exceeds `daily_loss_limit`.
    """
    dd = abs(float(current_drawdown))
    if dd >= max_dd_limit:
        return CircuitBreakerState(halted=True, reason="max_drawdown_breached", position_scale=0.0)
    if daily_return <= -abs(daily_loss_limit):
        return CircuitBreakerState(halted=True, reason="daily_loss_limit_breached", position_scale=0.0)
    if dd >= 0.5 * max_dd_limit:
        return CircuitBreakerState(halted=False, reason="half_drawdown_throttle", position_scale=0.5)
    return CircuitBreakerState(halted=False, reason="normal", position_scale=1.0)


def apply_slippage_and_commission(
    gross_return: float,
    turnover: float,
    slippage_bps: float = 5.0,
    commission_bps: float = 2.0,
) -> float:
    """Deduct realistic execution friction (slippage + commission in bps per unit turnover)."""
    total_cost_rate = ((slippage_bps + commission_bps) / 10000.0) * abs(turnover)
    return round(gross_return - total_cost_rate, 8)
