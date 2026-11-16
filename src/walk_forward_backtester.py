"""
Out-of-Sample Walk-Forward Backtester & Monte Carlo Simulator (`src/walk_forward_backtester.py`).

Computes audited performance metrics (CAGR, Sharpe, Sortino, Calmar, Max Drawdown,
Profit Factor, Expectancy, Sample Size N) with realistic slippage + commission friction.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable

from src.risk_execution_engine import (
    MAX_LEVERAGE,
    PAPER_TRADING_ONLY,
    ZERO_REAL_MONEY_RISK,
    apply_slippage_and_commission,
    check_drawdown_circuit_breaker,
)


def compute_audited_metrics(net_returns: list[float], periods_per_year: int = 252) -> dict[str, Any]:
    """Compute CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Profit Factor, Expectancy, and N."""
    n = len(net_returns)
    if n == 0:
        return {
            "sample_size_n": 0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
        }

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in net_returns:
        equity *= max(1.0 + r, 1e-8)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    years = max(n / float(periods_per_year), 1.0 / periods_per_year)
    cagr = (equity ** (1.0 / years)) - 1.0

    mean_r = sum(net_returns) / n
    var_r = sum((r - mean_r) ** 2 for r in net_returns) / max(n - 1, 1)
    std_r = math.sqrt(max(var_r, 1e-12))
    sharpe = (mean_r / std_r) * math.sqrt(periods_per_year) if std_r > 1e-10 else 0.0

    downside = [min(0.0, r) ** 2 for r in net_returns]
    down_dev = math.sqrt(sum(downside) / max(n, 1))
    sortino = (mean_r / down_dev) * math.sqrt(periods_per_year) if down_dev > 1e-10 else sharpe

    calmar = (cagr / max_dd) if max_dd > 1e-6 else (cagr * 10.0 if cagr > 0 else 0.0)

    gains = sum(r for r in net_returns if r > 0)
    losses = abs(sum(r for r in net_returns if r < 0))
    profit_factor = (gains / losses) if losses > 1e-9 else (2.5 if gains > 0 else 0.0)

    active_trades = [r for r in net_returns if abs(r) > 1e-8]
    expectancy = (sum(active_trades) / len(active_trades)) if active_trades else 0.0

    return {
        "sample_size_n": n,
        "active_bars": len(active_trades),
        "ending_equity_multiple": round(equity, 4),
        "cagr": round(cagr, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown": round(max_dd, 4),
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy, 6),
    }


def monte_carlo_drawdown_ci(
    net_returns: list[float],
    iterations: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Run Monte Carlo trade-shuffling simulation to estimate median and 95th-percentile max drawdown."""
    if not net_returns:
        return {"mc_median_max_dd": 0.0, "mc_p95_max_dd": 0.0}
    rng = random.Random(seed)
    dds: list[float] = []
    for _ in range(max(iterations, 10)):
        shuffled = list(net_returns)
        rng.shuffle(shuffled)
        eq = 1.0
        pk = 1.0
        mdd = 0.0
        for r in shuffled:
            eq *= max(1.0 + r, 1e-8)
            pk = max(pk, eq)
            mdd = max(mdd, (pk - eq) / pk)
        dds.append(mdd)
    dds.sort()
    mid = dds[len(dds) // 2]
    p95 = dds[min(int(0.95 * len(dds)), len(dds) - 1)]
    return {"mc_median_max_dd": round(mid, 4), "mc_p95_max_dd": round(p95, 4)}


def run_walk_forward_backtest(
    prices: list[float],
    signal_fn: Callable[[list[float]], dict[str, Any]],
    train_size: int = 60,
    test_size: int = 30,
    slippage_bps: float = 5.0,
    commission_bps: float = 2.0,
) -> dict[str, Any]:
    """
    Execute anchored/rolling out-of-sample walk-forward backtest with causal execution
    (weight[t-1] applied to return[t]), drawdown circuit breaker, and slippage/commission.
    """
    assert PAPER_TRADING_ONLY and ZERO_REAL_MONEY_RISK
    n = len(prices)
    if n < train_size + test_size:
        train_size = max(10, n // 2)
        test_size = max(5, n - train_size)

    oos_returns: list[float] = []
    folds = 0
    prev_w = 0.0
    equity = 1.0
    peak = 1.0

    start = train_size
    while start < n:
        end = min(start + test_size, n)
        slice_prices = prices[:end]
        sig_out = signal_fn(slice_prices)
        weights = sig_out.get("weights", [0.0] * len(slice_prices))

        for t in range(start, end):
            cb = check_drawdown_circuit_breaker((peak - equity) / peak)
            target_w = min(max(float(weights[t - 1]), 0.0), MAX_LEVERAGE) * cb.position_scale
            turnover = abs(target_w - prev_w)
            raw_ret = ((slice_prices[t] / max(slice_prices[t - 1], 1e-8)) - 1.0) * target_w
            net_ret = apply_slippage_and_commission(raw_ret, turnover, slippage_bps, commission_bps)
            oos_returns.append(net_ret)
            equity *= max(1.0 + net_ret, 1e-8)
            peak = max(peak, equity)
            prev_w = target_w

        folds += 1
        start = end

    metrics = compute_audited_metrics(oos_returns)
    mc = monte_carlo_drawdown_ci(oos_returns)
    return {
        "paper_trading_only": PAPER_TRADING_ONLY,
        "zero_real_money_risk": ZERO_REAL_MONEY_RISK,
        "max_leverage": MAX_LEVERAGE,
        "walk_forward_folds": folds,
        **metrics,
        **mc,
    }
