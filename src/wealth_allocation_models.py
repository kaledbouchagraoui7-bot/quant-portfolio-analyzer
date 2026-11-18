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


def compute_covariance_matrix(
    return_matrix: dict[str, list[float]],
    ridge: float = 1e-8,
) -> dict[str, dict[str, float]]:
    """Compute full sample covariance matrix Σ_ij across aligned return series."""
    assets = sorted(return_matrix.keys())
    means: dict[str, float] = {}
    for a in assets:
        rets = return_matrix[a]
        means[a] = (sum(rets) / len(rets)) if rets else 0.0

    cov: dict[str, dict[str, float]] = {a: {} for a in assets}
    for i, a in enumerate(assets):
        for j in range(i, len(assets)):
            b = assets[j]
            ra = return_matrix[a]
            rb = return_matrix[b]
            t_len = min(len(ra), len(rb))
            if t_len <= 1:
                c_val = 0.01 if a == b else 0.0
            else:
                ma = means[a]
                mb = means[b]
                c_val = sum((ra[t] - ma) * (rb[t] - mb) for t in range(t_len)) / max(t_len - 1, 1)
            if a == b:
                c_val = max(c_val, 1e-8) + ridge
            cov[a][b] = c_val
            cov[b][a] = c_val
    return cov


def _correl_ordered_assets(assets: list[str], cov: dict[str, dict[str, float]]) -> list[str]:
    """Order assets via nearest-neighbor correlation distance d_ij = sqrt(0.5 * (1 - rho_ij))."""
    if len(assets) <= 2:
        return list(assets)

    def _corr_dist(a: str, b: str) -> float:
        denom = math.sqrt(max(cov[a][a], 1e-12) * max(cov[b][b], 1e-12))
        rho = max(-1.0, min(1.0, cov[a][b] / denom)) if denom > 0 else 0.0
        return math.sqrt(max(0.0, 0.5 * (1.0 - rho)))

    # Start from the asset with lowest total variance so ordering is deterministic
    remaining = sorted(assets, key=lambda x: cov[x][x])
    ordered = [remaining.pop(0)]
    while remaining:
        cur = ordered[-1]
        nxt = min(remaining, key=lambda x: (_corr_dist(cur, x), cov[x][x], x))
        remaining.remove(nxt)
        ordered.append(nxt)
    return ordered


def hierarchical_risk_parity_allocation(return_matrix: dict[str, list[float]]) -> dict[str, float]:
    """
    Hierarchical Risk Parity (López de Prado recursive bisection using the full
    sample covariance matrix Σ_ij, correlation-distance seriation, and w^T Σ w cluster variance).
    """
    assets = sorted(return_matrix.keys())
    if not assets:
        return {}
    cov = compute_covariance_matrix(return_matrix)
    ordered = _correl_ordered_assets(assets, cov)
    weights = {a: 1.0 for a in ordered}

    def _cluster_var(cluster: list[str]) -> float:
        inv_v = [1.0 / max(cov[a][a], 1e-12) for a in cluster]
        s = sum(inv_v) or 1.0
        w_c = [v / s for v in inv_v]
        quad = 0.0
        for i, ai in enumerate(cluster):
            for j, aj in enumerate(cluster):
                quad += w_c[i] * w_c[j] * cov[ai][aj]
        return max(quad, 1e-12)

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
