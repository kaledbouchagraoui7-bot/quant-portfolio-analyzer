"""
Institutional Capital Allocation & Wealth-Compounding Models (`src/wealth_allocation_models.py`).

Implements:
  1. Equal-Weight (1/N)
  2. Mean-Variance (Markowitz inverse-covariance / Sharpe tangency proxy)
  3. Risk-Parity (Bridgewater All-Weather inverse-volatility risk contribution)
  4. Hierarchical Risk Parity (HRP cluster variance bisection)
  5. Fractional-Kelly Multi-Asset Compounding
"""

from __future__ import annotations

import math
from typing import Any


def _asset_stats(return_matrix: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for asset, rets in return_matrix.items():
        n = len(rets)
        if n == 0:
            means[asset] = 0.0
            stds[asset] = 0.10
            continue
        m = sum(rets) / n
        var = sum((r - m) ** 2 for r in rets) / max(n - 1, 1)
        means[asset] = m
        stds[asset] = max(math.sqrt(max(var, 1e-10)), 1e-4)
    return means, stds


def equal_weight_allocation(return_matrix: dict[str, list[float]]) -> dict[str, float]:
    assets = sorted(return_matrix.keys())
    if not assets:
        return {}
    w = 1.0 / len(assets)
    return {a: round(w, 6) for a in assets}


def risk_parity_all_weather_allocation(return_matrix: dict[str, list[float]]) -> dict[str, float]:
    """Bridgewater-style inverse-volatility risk-parity weights (`sum(w) == 1.0`)."""
    _, stds = _asset_stats(return_matrix)
    inv_vols = {a: 1.0 / stds[a] for a in sorted(return_matrix.keys())}
    total = sum(inv_vols.values()) or 1.0
    return {a: round(v / total, 6) for a, v in inv_vols.items()}


def mean_variance_sharpe_allocation(
    return_matrix: dict[str, list[float]],
    risk_free_daily: float = 0.0001,
) -> dict[str, float]:
    """Long-only mean-variance (positive excess return / variance) weights."""
    means, stds = _asset_stats(return_matrix)
    scores = {}
    for a in sorted(return_matrix.keys()):
        excess = max(means[a] - risk_free_daily, 1e-5)
        scores[a] = excess / (stds[a] ** 2)
    total = sum(scores.values()) or 1.0
    return {a: round(v / total, 6) for a, v in scores.items()}


def hierarchical_risk_parity_allocation(return_matrix: dict[str, list[float]]) -> dict[str, float]:
    """
    Hierarchical Risk Parity (López de Prado recursive bisection by volatility-ordered clusters).
    """
    _, stds = _asset_stats(return_matrix)
    ordered = [a for a, _ in sorted(stds.items(), key=lambda kv: kv[1])]
    weights = {a: 1.0 for a in ordered}

    def _cluster_var(cluster: list[str]) -> float:
        inv_v = [1.0 / (stds[a] ** 2) for a in cluster]
        s = sum(inv_v) or 1.0
        w_c = [v / s for v in inv_v]
        return sum((w_c[i] ** 2) * (stds[cluster[i]] ** 2) for i in range(len(cluster)))

    clusters = [ordered]
    while any(len(c) > 1 for c in clusters):
        next_clusters: list[list[str]] = []
        for c in clusters:
            if len(c) <= 1:
                next_clusters.append(c)
                continue
            mid = len(c) // 2
            left, right = c[:mid], c[mid:]
            v_left = _cluster_var(left)
            v_right = _cluster_var(right)
            alpha = 1.0 - (v_left / max(v_left + v_right, 1e-12))
            for a in left:
                weights[a] *= alpha
            for a in right:
                weights[a] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters

    total = sum(weights.values()) or 1.0
    return {a: round(weights[a] / total, 6) for a in sorted(return_matrix.keys())}


def fractional_kelly_portfolio_allocation(
    return_matrix: dict[str, list[float]],
    fraction: float = 0.25,
) -> dict[str, float]:
    """Ed Thorp / Renaissance-style fractional-Kelly allocation normalized to <= 1.0 leverage."""
    means, stds = _asset_stats(return_matrix)
    raw = {}
    for a in sorted(return_matrix.keys()):
        k = max(0.0, (means[a] / (stds[a] ** 2)) * fraction)
        raw[a] = k
    total = sum(raw.values())
    if total <= 1e-9:
        return equal_weight_allocation(return_matrix)
    return {a: round(v / total, 6) for a, v in raw.items()}


def compare_allocation_models(return_matrix: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    """Evaluate all 5 institutional allocation models on a common multi-asset return matrix."""
    from src.walk_forward_backtester import compute_audited_metrics

    models = {
        "equal_weight": equal_weight_allocation(return_matrix),
        "mean_variance_markowitz": mean_variance_sharpe_allocation(return_matrix),
        "risk_parity_all_weather": risk_parity_all_weather_allocation(return_matrix),
        "hierarchical_risk_parity_hrp": hierarchical_risk_parity_allocation(return_matrix),
        "fractional_kelly_thorp": fractional_kelly_portfolio_allocation(return_matrix),
    }
    assets = sorted(return_matrix.keys())
    n = min(len(v) for v in return_matrix.values()) if assets else 0
    results: dict[str, dict[str, Any]] = {}
    for name, w in models.items():
        port_rets = [
            sum(w[a] * return_matrix[a][t] for a in assets)
            for t in range(n)
        ]
        results[name] = {
            "weights": w,
            "metrics": compute_audited_metrics(port_rets),
        }
    return results
