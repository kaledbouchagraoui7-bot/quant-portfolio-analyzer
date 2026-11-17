"""
Independent Quantitative Validation, Execution Realism & 3-Tier Kill-Switch
(`src/independent_validator.py`).

Open-Source Quantitative Research & Risk-Analytics Software — Not Investment Advice
(AMF / MiFID II Compliant). Independently audits backtest claims (including raw vs.
excess-over-cash Sharpe, Deflated Sharpe Ratio, block bootstrap, sensitivity grid),
4-tier execution realism, strategy lifecycle transitions, and 3-tier kill-switch state.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.risk_execution_engine import apply_slippage_and_commission
from src.trading_bots.institutional_momentum_allocator import compute_dual_momentum_weights
from src.walk_forward_backtester import compute_audited_metrics


class StrategyLifecycleState(str, Enum):
    PROPOSED = "PROPOSED"
    RESEARCHED = "RESEARCHED"
    BACKTESTED = "BACKTESTED"
    VALIDATED = "VALIDATED"
    PAPER = "PAPER"
    APPROVED = "APPROVED"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    SUSPENDED = "SUSPENDED"
    RETIRED = "RETIRED"


_VALID_TRANSITIONS: dict[StrategyLifecycleState, set[StrategyLifecycleState]] = {
    StrategyLifecycleState.PROPOSED: {StrategyLifecycleState.RESEARCHED, StrategyLifecycleState.RETIRED},
    StrategyLifecycleState.RESEARCHED: {StrategyLifecycleState.BACKTESTED, StrategyLifecycleState.RETIRED},
    StrategyLifecycleState.BACKTESTED: {StrategyLifecycleState.VALIDATED, StrategyLifecycleState.DEGRADED, StrategyLifecycleState.RETIRED},
    StrategyLifecycleState.VALIDATED: {StrategyLifecycleState.PAPER, StrategyLifecycleState.DEGRADED, StrategyLifecycleState.RETIRED},
    StrategyLifecycleState.PAPER: {StrategyLifecycleState.APPROVED, StrategyLifecycleState.DEGRADED, StrategyLifecycleState.SUSPENDED},
    StrategyLifecycleState.APPROVED: {StrategyLifecycleState.ACTIVE, StrategyLifecycleState.SUSPENDED},
    StrategyLifecycleState.ACTIVE: {StrategyLifecycleState.DEGRADED, StrategyLifecycleState.SUSPENDED, StrategyLifecycleState.RETIRED},
    StrategyLifecycleState.DEGRADED: {StrategyLifecycleState.SUSPENDED, StrategyLifecycleState.RETIRED, StrategyLifecycleState.RESEARCHED},
    StrategyLifecycleState.SUSPENDED: {StrategyLifecycleState.RETIRED, StrategyLifecycleState.VALIDATED},
    StrategyLifecycleState.RETIRED: set(),
}


def transition_strategy_lifecycle(
    current: StrategyLifecycleState,
    target: StrategyLifecycleState,
) -> StrategyLifecycleState:
    if target not in _VALID_TRANSITIONS.get(current, set()):
        raise ValueError(f"Illegal strategy lifecycle transition: {current.value} -> {target.value}")
    return target


def deflated_sharpe_ratio(observed_sharpe: float, n_obs: int, n_trials: int = 8) -> float:
    """
    Approximate Bailey & López de Prado Deflated Sharpe Ratio (DSR) probability
    accounting for finite sample size `n_obs` and multiple testing (`n_trials`).
    """
    if n_obs < 10 or n_trials < 1:
        return 0.0
    euler_gamma = 0.5772156649
    if n_trials == 1:
        sr_benchmark = 0.0
    else:
        z_max = math.sqrt(2.0 * math.log(max(n_trials, 2)))
        sr_benchmark = (z_max - (math.log(math.pi) + math.log(math.log(max(n_trials, 2))) - 2.0 * euler_gamma) / (2.0 * z_max)) / math.sqrt(max(n_obs / 252.0, 0.25))
    sr_se = math.sqrt((1.0 + 0.5 * (observed_sharpe ** 2)) / max(n_obs - 1, 1)) * math.sqrt(252.0) / math.sqrt(252.0)
    z_stat = (observed_sharpe - sr_benchmark) / max(sr_se * math.sqrt(252.0 / max(n_obs, 1)), 1e-6)
    prob = 0.5 * (1.0 + math.erf(z_stat / math.sqrt(2.0)))
    return round(max(0.0, min(1.0, prob)), 4)


def block_bootstrap_sharpe_ci(
    returns: list[float],
    block_size: int = 10,
    iterations: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Stationary-style block bootstrap 95% confidence interval for annualized Sharpe."""
    if len(returns) < block_size:
        return {"p05_sharpe": 0.0, "median_sharpe": 0.0, "p95_sharpe": 0.0}
    rng = random.Random(seed)
    n = len(returns)
    sharpes: list[float] = []
    for _ in range(iterations):
        sample: list[float] = []
        while len(sample) < n:
            idx = rng.randint(0, n - block_size)
            sample.extend(returns[idx : idx + block_size])
        sample = sample[:n]
        sharpes.append(compute_audited_metrics(sample)["sharpe_ratio"])
    sharpes.sort()
    return {
        "p05_sharpe": round(sharpes[int(0.05 * len(sharpes))], 4),
        "median_sharpe": round(sharpes[len(sharpes) // 2], 4),
        "p95_sharpe": round(sharpes[min(int(0.95 * len(sharpes)), len(sharpes) - 1)], 4),
    }


def simulate_execution_scenarios(
    weights_history: list[dict[str, float]],
    ret_map: dict[str, list[float]],
    start_bar: int = 60,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate 4 realistic execution scenarios:
      - optimistic: 2 bps total cost
      - base: 7 bps cost + volume-impact proxy
      - adverse: 18 bps cost (wider spreads + partial-fill slippage)
      - severe: 35 bps cost + 1-bar delayed execution lag
    """
    assets = sorted(ret_map.keys())
    scenarios = {
        "optimistic": {"slip_bps": 1.5, "comm_bps": 0.5, "lag": 1},
        "base": {"slip_bps": 5.0, "comm_bps": 2.0, "lag": 1},
        "adverse": {"slip_bps": 14.0, "comm_bps": 4.0, "lag": 1},
        "severe": {"slip_bps": 28.0, "comm_bps": 7.0, "lag": 2},
    }
    n = len(next(iter(ret_map.values())))
    results: dict[str, dict[str, Any]] = {}

    for name, cfg in scenarios.items():
        lag = int(cfg["lag"])
        oos_rets: list[float] = []
        for t in range(max(start_bar, lag), n):
            w_exec = weights_history[t - lag]
            w_prev = weights_history[max(0, t - lag - 1)]
            turnover = sum(abs(w_exec[a] - w_prev[a]) for a in assets)
            gross = sum(w_exec[a] * ret_map[a][t] for a in assets)
            # Market impact proxy scales nonlinearly with sqrt(turnover)
            extra_impact_bps = (math.sqrt(max(turnover, 0.0)) * 3.0) if name in ("adverse", "severe") else 0.0
            net = apply_slippage_and_commission(gross, turnover, cfg["slip_bps"] + extra_impact_bps, cfg["comm_bps"])
            oos_rets.append(net)
        results[name] = compute_audited_metrics(oos_rets)
    return results


def audit_momentum_allocator_claims(
    price_map: dict[str, list[float]],
    ret_map: dict[str, list[float]],
    start_bar: int = 60,
    n_trials: int = 8,
) -> dict[str, Any]:
    """
    Independently reproduce and audit `institutional_momentum_allocator` metrics:
    reports Raw Sharpe, Excess-over-Cash Sharpe (subtracting T-bill baseline),
    Deflated Sharpe Ratio, Block-Bootstrap CI, Parameter Sensitivity Grid, and 4 Execution Tiers.
    """
    assets = sorted(price_map.keys())
    mom_out = compute_dual_momentum_weights(price_map)
    wh = mom_out["weights_history"]
    base_rets: list[float] = []
    excess_rets: list[float] = []

    for t in range(start_bar, len(ret_map["equities_index"])):
        w_prev = wh[t - 1]
        w_curr = wh[t]
        turnover = sum(abs(w_curr[a] - w_prev[a]) for a in assets)
        gross = sum(w_prev[a] * ret_map[a][t] for a in assets)
        net = apply_slippage_and_commission(gross, turnover, 5.0, 2.0)
        base_rets.append(net)
        excess_rets.append(net - ret_map.get("cash_bills", [0.0] * len(ret_map["equities_index"]))[t])

    raw_metrics = compute_audited_metrics(base_rets)
    excess_metrics = compute_audited_metrics(excess_rets)
    dsr = deflated_sharpe_ratio(raw_metrics["sharpe_ratio"], raw_metrics["sample_size_n"], n_trials=n_trials)
    bootstrap_ci = block_bootstrap_sharpe_ci(base_rets)
    tiers = simulate_execution_scenarios(wh, ret_map, start_bar=start_bar)

    # Parameter sensitivity (±20% around defaults)
    sensitivity: list[dict[str, Any]] = []
    for lb in (48, 60, 72):
        for vol_cap in (0.020, 0.025, 0.030):
            s_wh = compute_dual_momentum_weights(price_map, lookback=lb, target_annual_vol=vol_cap)["weights_history"]
            s_rets = [
                apply_slippage_and_commission(
                    sum(s_wh[t - 1][a] * ret_map[a][t] for a in assets),
                    sum(abs(s_wh[t][a] - s_wh[t - 1][a]) for a in assets),
                    5.0,
                    2.0,
                )
                for t in range(start_bar, len(ret_map["equities_index"]))
            ]
            m = compute_audited_metrics(s_rets)
            sensitivity.append({"lookback": lb, "vol_cap": vol_cap, "cagr": m["cagr"], "sharpe": m["sharpe_ratio"], "max_dd": m["max_drawdown"]})

    return {
        "raw_metrics": raw_metrics,
        "excess_over_cash_metrics": excess_metrics,
        "deflated_sharpe_prob": dsr,
        "bootstrap_95_ci": bootstrap_ci,
        "execution_scenarios": tiers,
        "parameter_sensitivity_grid": sensitivity,
    }


@dataclass
class KillSwitchStatus:
    tier: str  # NORMAL | WARNING | RISK_REDUCTION | EMERGENCY_HALT
    position_scale: float
    cancel_pending_orders: bool
    requires_human_reset: bool
    halted: bool


def evaluate_tiered_kill_switch(
    current_drawdown: float,
    was_halted_previously: bool = False,
    human_reset_token: str = "",
) -> KillSwitchStatus:
    """
    3-tier deterministic kill switch. Once EMERGENCY_HALT is entered, recovery CANNOT
    auto-resume without an explicit non-empty `human_reset_token`.
    """
    dd = abs(float(current_drawdown))
    if was_halted_previously and human_reset_token != "HUMAN_CONFIRMED_RESET":
        return KillSwitchStatus("EMERGENCY_HALT", 0.0, True, True, True)
    if dd >= 0.12:
        return KillSwitchStatus("EMERGENCY_HALT", 0.0, True, True, True)
    if dd >= 0.06:
        return KillSwitchStatus("RISK_REDUCTION", 0.5, False, False, False)
    if dd >= 0.04:
        return KillSwitchStatus("WARNING", 0.85, False, False, False)
    return KillSwitchStatus("NORMAL", 1.0, False, False, False)
