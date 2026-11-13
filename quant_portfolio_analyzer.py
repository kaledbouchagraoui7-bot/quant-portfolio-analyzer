#!/usr/bin/env python3
"""
Quantitative Portfolio Risk, VaR/CVaR, DCA Simulator & Risk-Parity Analyzer
(`quant_portfolio_analyzer.py`)

Adapted locally inside `obsidian-second-brain` using clean open-source quantitative finance
algorithms inspired by `shashankvemuri/Finance` (MIT) and `skfolio/skfolio` (BSD-3-Clause).
Zero live brokerage credentials, zero trading execution — pure mathematical analytics.

Informational Quantitative & Mathematical Analytics Only: Outputs are deterministic
educational/research math calculations on public or user-supplied CSV/JSON series —
not certified financial advice, investment recommendations, or live brokerage execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence


DISCLAIMER = (
    "Informational Quantitative & Mathematical Analytics Only: Outputs are deterministic "
    "educational/research math calculations on public or user-supplied CSV/JSON series — "
    "not certified financial advice, investment recommendations, or live brokerage execution."
)


@dataclass(frozen=True)
class PortfolioMetrics:
    observations: int
    start_price: float
    end_price: float
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    peak_label: str
    trough_label: str
    var_95_hist_pct: float
    var_99_hist_pct: float
    var_95_param_pct: float
    var_99_param_pct: float
    cvar_95_hist_pct: float
    full_kelly_pct: float
    half_kelly_pct: float
    disclaimer: str = DISCLAIMER


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile matching standard empirical pandas/numpy percentile."""
    if not sorted_vals:
        raise ValueError("Cannot compute percentile of empty sequence.")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def load_price_series(file_path: Path) -> tuple[list[str], list[float]]:
    """Load `(labels, prices)` from a single-asset CSV (date,close|price|value) or JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Price series file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            labels = [str(row.get("date") or row.get("label") or idx) for idx, row in enumerate(raw)]
            prices = [float(row.get("close") if "close" in row else row.get("price", row.get("value"))) for row in raw]
            return labels, prices
        if isinstance(raw, list) and raw:
            return [str(i) for i in range(len(raw))], [float(x) for x in raw]
        raise ValueError("JSON must be a non-empty list of numbers or {date, close} objects.")

    labels: list[str] = []
    prices: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")
        lower_map = {f.lower().strip(): f for f in reader.fieldnames}
        price_col = next((lower_map[c] for c in ("close", "adj_close", "price", "value", "nav") if c in lower_map), None)
        label_col = next((lower_map[c] for c in ("date", "timestamp", "period", "day") if c in lower_map), None)
        if not price_col:
            numeric_cols = [f for f in reader.fieldnames if f.lower().strip() not in ("date", "timestamp", "period", "day")]
            if not numeric_cols:
                raise ValueError(f"CSV must contain a numeric price column; found {reader.fieldnames}")
            price_col = numeric_cols[0]
        for idx, row in enumerate(reader):
            raw_val = (row.get(price_col) or "").strip()
            if not raw_val:
                continue
            val = float(raw_val)
            if val <= 0:
                raise ValueError(f"Prices must be strictly positive; got {val} at row {idx + 1}")
            prices.append(val)
            labels.append((row.get(label_col) or str(idx)).strip() if label_col else str(idx))
    if len(prices) < 2:
        raise ValueError("Need at least 2 positive price observations.")
    return labels, prices


def load_multi_asset_prices(file_path: Path) -> tuple[list[str], dict[str, list[float]]]:
    """Load `(labels, {asset: [prices...]})` from a multi-column CSV or JSON dict/list."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Multi-asset file not found: {path}")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            labels = [str(x) for x in raw.get("dates", [])]
            assets = {k: [float(v) for v in vals] for k, vals in raw.items() if k != "dates"}
            if not assets:
                raise ValueError("JSON dict must contain at least one asset price series.")
            first_len = len(next(iter(assets.values())))
            if not labels:
                labels = [str(i) for i in range(first_len)]
            return labels, assets
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            keys = [k for k in raw[0].keys() if k.lower() not in ("date", "label", "timestamp")]
            labels = [str(r.get("date") or r.get("label") or i) for i, r in enumerate(raw)]
            assets = {k: [float(r[k]) for r in raw] for k in keys}
            return labels, assets
        raise ValueError("Unsupported JSON format for multi-asset loader.")

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")
        label_col = next((f for f in reader.fieldnames if f.lower().strip() in ("date", "timestamp", "period", "day")), None)
        asset_cols = [f for f in reader.fieldnames if f != label_col]
        if not asset_cols:
            raise ValueError("CSV must contain at least one numeric asset column.")
        labels = []
        assets: dict[str, list[float]] = {col: [] for col in asset_cols}
        for idx, row in enumerate(reader):
            labels.append((row.get(label_col) or str(idx)).strip() if label_col else str(idx))
            for col in asset_cols:
                val = float((row.get(col) or "").strip())
                if val <= 0:
                    raise ValueError(f"All prices must be positive for {col}; got {val}")
                assets[col].append(val)
    return labels, assets


def compute_var(
    returns: Sequence[float],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """One-period Value-at-Risk returned as a non-negative loss fraction."""
    if not returns:
        raise ValueError("returns must be non-empty.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1.")
    vals = sorted(float(r) for r in returns)
    if method == "historical":
        q = _percentile(vals, 1.0 - confidence)
    elif method == "normal":
        if len(vals) < 2:
            raise ValueError("Parametric VaR requires at least two return observations.")
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = math.sqrt(var)
        q = mean + NormalDist().inv_cdf(1.0 - confidence) * std
    else:
        raise ValueError("method must be 'historical' or 'normal'")
    return float(max(0.0, -q))


def compute_expected_shortfall(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Historical Conditional Value-at-Risk (CVaR / Expected Shortfall) as a non-negative loss fraction."""
    if not returns:
        raise ValueError("returns must be non-empty.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be strictly between 0 and 1.")
    vals = sorted(float(r) for r in returns)
    cutoff = _percentile(vals, 1.0 - confidence)
    tail = [v for v in vals if v <= cutoff]
    if not tail:
        tail = [vals[0]]
    return float(max(0.0, -(sum(tail) / len(tail))))


def compare_lump_sum_vs_dca(
    prices: Sequence[float],
    budget: float = 12000.0,
    installments: int | None = None,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Compare Lump-Sum vs. spaced Dollar-Cost Averaging (DCA) over observed closes.
    Idle DCA cash earns 0%; installments split `budget` evenly across spaced bars.
    """
    if len(prices) < 2:
        raise ValueError("Need at least 2 price observations for Lump-Sum vs DCA simulation.")
    if budget <= 0:
        raise ValueError("budget must be strictly positive.")
    clean = [float(p) for p in prices]
    if any(p <= 0 for p in clean):
        raise ValueError("All prices must be strictly positive.")

    n = len(clean)
    n_inst = min(n, installments if installments is not None else min(12, n))
    if n_inst < 1:
        raise ValueError("installments must be >= 1.")

    lbls = list(labels) if labels is not None else [str(i) for i in range(n)]
    if n_inst == 1:
        invest_indices = {0}
    else:
        invest_indices = {round(i * (n - 1) / (n_inst - 1)) for i in range(n_inst)}

    installment_amt = budget / float(len(invest_indices))
    cash = budget
    shares = 0.0
    lump_series: list[float] = []
    dca_series: list[float] = []

    for idx, price in enumerate(clean):
        lump_series.append(round(budget * (price / clean[0]), 2))
        if idx in invest_indices:
            spend = min(installment_amt, cash)
            cash -= spend
            shares += spend / price
        dca_series.append(round(cash + shares * price, 2))

    ls_dd, _, _ = compute_max_drawdown(lump_series, lbls)
    dca_dd, _, _ = compute_max_drawdown(dca_series, lbls)

    return {
        "budget": round(budget, 2),
        "installments_used": len(invest_indices),
        "lump_sum_final": lump_series[-1],
        "lump_sum_return_pct": round(((lump_series[-1] / budget) - 1.0) * 100.0, 2),
        "lump_sum_max_drawdown_pct": round(ls_dd * 100.0, 2),
        "dca_final": dca_series[-1],
        "dca_return_pct": round(((dca_series[-1] / budget) - 1.0) * 100.0, 2),
        "dca_max_drawdown_pct": round(dca_dd * 100.0, 2),
        "winner_by_final_value": "lump_sum" if lump_series[-1] >= dca_series[-1] else "dca",
        "lower_drawdown_strategy": "dca" if abs(dca_dd) <= abs(ls_dd) else "lump_sum",
        "lump_sum_curve": lump_series,
        "dca_curve": dca_series,
        "disclaimer": DISCLAIMER,
    }


def compute_inverse_volatility_weights(
    asset_prices: dict[str, Sequence[float]],
    periods_per_year: int = 252,
    risk_free_rate: float = 0.03,
) -> dict[str, Any]:
    """
    Compute Inverse-Volatility / Risk-Parity weights for a basket of assets:
        w_i = (1 / sigma_i) / sum_j (1 / sigma_j)
    and report portfolio-level metrics for the weighted basket.
    """
    if not asset_prices:
        raise ValueError("asset_prices must not be empty.")
    lengths = {len(v) for v in asset_prices.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("All assets must share the same length >= 2.")

    per_asset: dict[str, dict[str, float]] = {}
    inv_vols: dict[str, float] = {}
    asset_returns: dict[str, list[float]] = {}

    for name, series_vals in asset_prices.items():
        clean = [float(x) for x in series_vals]
        if any(x <= 0 for x in clean):
            raise ValueError(f"All prices for asset {name!r} must be strictly positive.")
        rets = [(clean[i] / clean[i - 1]) - 1.0 for i in range(1, len(clean))]
        asset_returns[name] = rets
        metrics = analyze_price_series(clean, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate)
        vol = max(metrics.annualized_volatility_pct, 1e-6)
        inv_vols[name] = 1.0 / vol
        per_asset[name] = {
            "total_return_pct": metrics.total_return_pct,
            "cagr_pct": metrics.cagr_pct,
            "annualized_volatility_pct": metrics.annualized_volatility_pct,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown_pct": metrics.max_drawdown_pct,
            "var_95_hist_pct": metrics.var_95_hist_pct,
        }

    total_inv = sum(inv_vols.values())
    weights = {name: round(val / total_inv, 6) for name, val in inv_vols.items()}
    # Ensure exact 1.0 sum on the final asset
    names_list = list(weights.keys())
    if len(names_list) > 1:
        residual = round(1.0 - sum(weights[n] for n in names_list[:-1]), 6)
        weights[names_list[-1]] = residual
    else:
        weights[names_list[0]] = 1.0

    # Reconstruct synthetic weighted portfolio index starting at 100.0
    n_rets = len(next(iter(asset_returns.values())))
    basket_prices = [100.0]
    for t in range(n_rets):
        port_r = sum(weights[name] * asset_returns[name][t] for name in names_list)
        basket_prices.append(basket_prices[-1] * (1.0 + port_r))

    basket_metrics = analyze_price_series(
        basket_prices,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    return {
        "weights": weights,
        "per_asset_metrics": per_asset,
        "basket_metrics": asdict(basket_metrics),
        "disclaimer": DISCLAIMER,
    }


def compute_kelly_fraction(
    win_rate: float,
    payoff_ratio: float,
    max_cap: float = 0.25,
) -> tuple[float, float]:
    """
    Compute educational Kelly Criterion position fraction:
        f* = p - (1 - p) / b
    Returns `(capped_full_kelly, capped_half_kelly)` clamped to `[0.0, max_cap]`.
    """
    if not (0.0 <= win_rate <= 1.0):
        raise ValueError(f"win_rate must be between 0.0 and 1.0; got {win_rate}")
    if payoff_ratio <= 0.0:
        raise ValueError(f"payoff_ratio must be strictly positive; got {payoff_ratio}")
    raw_kelly = win_rate - ((1.0 - win_rate) / payoff_ratio)
    full_kelly = max(0.0, min(max_cap, raw_kelly))
    half_kelly = max(0.0, min(max_cap, raw_kelly * 0.5))
    return round(full_kelly, 6), round(half_kelly, 6)


def compute_max_drawdown(
    prices: Sequence[float],
    labels: Sequence[str] | None = None,
) -> tuple[float, str, str]:
    """Return `(max_drawdown_decimal, peak_label, trough_label)` where drawdown is <= 0.0."""
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to compute drawdown.")
    lbls = list(labels) if labels is not None else [str(i) for i in range(len(prices))]
    peak_val = float(prices[0])
    peak_lbl = lbls[0]
    best_peak_lbl = peak_lbl
    best_trough_lbl = lbls[0]
    max_dd = 0.0

    for price, lbl in zip(prices, lbls):
        p = float(price)
        if p <= 0:
            raise ValueError(f"All prices must be positive; got {p}")
        if p > peak_val:
            peak_val = p
            peak_lbl = lbl
        dd = (p - peak_val) / peak_val
        if dd < max_dd:
            max_dd = dd
            best_peak_lbl = peak_lbl
            best_trough_lbl = lbl

    return max_dd, best_peak_lbl, best_trough_lbl


def analyze_price_series(
    prices: Sequence[float],
    labels: Sequence[str] | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.03,
    win_rate: float | None = None,
    payoff_ratio: float | None = None,
) -> PortfolioMetrics:
    """Compute CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar, VaR/CVaR, and Kelly sizing cleanly."""
    if len(prices) < 2:
        raise ValueError("At least 2 positive price observations are required.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    clean_prices = [float(p) for p in prices]
    if any(p <= 0 for p in clean_prices):
        raise ValueError("All prices must be strictly positive.")
    lbls = list(labels) if labels is not None else [str(i) for i in range(len(clean_prices))]

    returns = [(clean_prices[i] / clean_prices[i - 1]) - 1.0 for i in range(1, len(clean_prices))]
    n_rets = len(returns)

    start_p = clean_prices[0]
    end_p = clean_prices[-1]
    total_ret = (end_p / start_p) - 1.0
    years = n_rets / float(periods_per_year)
    cagr = ((end_p / start_p) ** (1.0 / years) - 1.0) if years > 0 else total_ret

    mean_ret = sum(returns) / n_rets
    rf_per_period = risk_free_rate / float(periods_per_year)
    excess_returns = [r - rf_per_period for r in returns]

    if n_rets > 1:
        var = sum((r - mean_ret) ** 2 for r in returns) / (n_rets - 1)
        std_period = math.sqrt(var)
    else:
        std_period = 0.0
    ann_vol = std_period * math.sqrt(periods_per_year)

    excess_mean = sum(excess_returns) / n_rets
    sharpe = (excess_mean / std_period) * math.sqrt(periods_per_year) if std_period > 1e-12 else 0.0

    downside_sq = [min(0.0, er) ** 2 for er in excess_returns]
    downside_dev = math.sqrt(sum(downside_sq) / n_rets)
    sortino = (excess_mean / downside_dev) * math.sqrt(periods_per_year) if downside_dev > 1e-12 else 0.0

    max_dd, peak_lbl, trough_lbl = compute_max_drawdown(clean_prices, lbls)
    calmar = (cagr / abs(max_dd)) if abs(max_dd) > 1e-12 else 0.0

    var_95_hist = compute_var(returns, confidence=0.95, method="historical")
    var_99_hist = compute_var(returns, confidence=0.99, method="historical")
    if n_rets >= 2:
        var_95_param = compute_var(returns, confidence=0.95, method="normal")
        var_99_param = compute_var(returns, confidence=0.99, method="normal")
    else:
        var_95_param = var_95_hist
        var_99_param = var_99_hist
    cvar_95_hist = compute_expected_shortfall(returns, confidence=0.95)

    if win_rate is None or payoff_ratio is None:
        pos = [r for r in returns if r > 0]
        neg = [abs(r) for r in returns if r < 0]
        emp_win = (len(pos) / n_rets) if n_rets else 0.5
        avg_win = (sum(pos) / len(pos)) if pos else 0.01
        avg_loss = (sum(neg) / len(neg)) if neg else 0.01
        emp_payoff = max(0.1, avg_win / max(avg_loss, 1e-9))
        full_k, half_k = compute_kelly_fraction(emp_win, emp_payoff)
    else:
        full_k, half_k = compute_kelly_fraction(win_rate, payoff_ratio)

    return PortfolioMetrics(
        observations=len(clean_prices),
        start_price=round(start_p, 4),
        end_price=round(end_p, 4),
        total_return_pct=round(total_ret * 100.0, 2),
        cagr_pct=round(cagr * 100.0, 2),
        annualized_volatility_pct=round(ann_vol * 100.0, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        max_drawdown_pct=round(max_dd * 100.0, 2),
        calmar_ratio=round(calmar, 3),
        peak_label=peak_lbl,
        trough_label=trough_lbl,
        var_95_hist_pct=round(var_95_hist * 100.0, 2),
        var_99_hist_pct=round(var_99_hist * 100.0, 2),
        var_95_param_pct=round(var_95_param * 100.0, 2),
        var_99_param_pct=round(var_99_param * 100.0, 2),
        cvar_95_hist_pct=round(cvar_95_hist * 100.0, 2),
        full_kelly_pct=round(full_k * 100.0, 2),
        half_kelly_pct=round(half_k * 100.0, 2),
    )


def format_markdown_report(
    metrics: PortfolioMetrics,
    source_name: str = "sample_series",
    dca_summary: dict[str, Any] | None = None,
    multi_asset_summary: dict[str, Any] | None = None,
) -> str:
    lines = [
        f"# Quantitative Portfolio Risk, VaR & DCA Report (`{source_name}`)",
        "",
        f"> **Notice:** {metrics.disclaimer}",
        "",
        "## Core Performance & Tail Risk Metrics",
        "| Metric | Value | Meaning (Plain Language) |",
        "| :--- | ---: | :--- |",
        f"| Observations | `{metrics.observations}` | Number of price checkpoints analyzed |",
        f"| Total Return | `{metrics.total_return_pct:+.2f}%` | Overall gain/loss from `{metrics.start_price}` to `{metrics.end_price}` |",
        f"| CAGR | `{metrics.cagr_pct:+.2f}%` | Smoothed annual compounding rate |",
        f"| Annualized Volatility | `{metrics.annualized_volatility_pct:.2f}%` | Typical yearly price swing size |",
        f"| Sharpe Ratio | `{metrics.sharpe_ratio:.3f}` | Return earned per unit of total volatility |",
        f"| Sortino Ratio | `{metrics.sortino_ratio:.3f}` | Return earned per unit of downside drops |",
        f"| Max Drawdown | `{metrics.max_drawdown_pct:.2f}%` | Worst drop (`{metrics.peak_label}` -> `{metrics.trough_label}`) |",
        f"| Calmar Ratio | `{metrics.calmar_ratio:.3f}` | Annual return divided by worst drop |",
        f"| Historical VaR (95% / 99%) | `-{metrics.var_95_hist_pct:.2f}%` / `-{metrics.var_99_hist_pct:.2f}%` | Empirical 1-period tail loss threshold |",
        f"| Parametric VaR (95% / 99%) | `-{metrics.var_95_param_pct:.2f}%` / `-{metrics.var_99_param_pct:.2f}%` | Normal-distribution 1-period tail loss estimate |",
        f"| Historical CVaR / Expected Shortfall (95%) | `-{metrics.cvar_95_hist_pct:.2f}%` | Average loss inside worst 5% tail periods |",
        f"| Educational Kelly (Half / Capped Full) | `{metrics.half_kelly_pct:.2f}%` / `{metrics.full_kelly_pct:.2f}%` | Theoretical position sizing cap (25% hard cap) |",
    ]

    if dca_summary:
        lines.extend([
            "",
            "## Lump-Sum vs. Dollar-Cost Averaging (DCA) Simulation",
            f"- **Initial Cash Budget:** `{dca_summary['budget']}` | **DCA Installments:** `{dca_summary['installments_used']}`",
            f"- **Lump-Sum Outcome:** Final `{dca_summary['lump_sum_final']}` (`{dca_summary['lump_sum_return_pct']:+.2f}%`, Max Drawdown `{dca_summary['lump_sum_max_drawdown_pct']:.2f}%`)",
            f"- **Monthly/Spaced DCA Outcome:** Final `{dca_summary['dca_final']}` (`{dca_summary['dca_return_pct']:+.2f}%`, Max Drawdown `{dca_summary['dca_max_drawdown_pct']:.2f}%`)",
            f"- **Higher Final Value:** `{dca_summary['winner_by_final_value']}` | **Lower Drawdown:** `{dca_summary['lower_drawdown_strategy']}`",
        ])

    if multi_asset_summary:
        lines.extend([
            "",
            "## Multi-Asset Inverse-Volatility (Risk-Parity) Allocation",
            "| Asset | Inv-Vol Weight | Total Return | Volatility | Sharpe | Max Drawdown | Hist VaR (95%) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for asset, weight in multi_asset_summary["weights"].items():
            am = multi_asset_summary["per_asset_metrics"][asset]
            lines.append(
                f"| `{asset}` | `{weight * 100:.2f}%` | `{am['total_return_pct']:+.2f}%` | "
                f"`{am['annualized_volatility_pct']:.2f}%` | `{am['sharpe_ratio']:.3f}` | "
                f"`{am['max_drawdown_pct']:.2f}%` | `-{am['var_95_hist_pct']:.2f}%` |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantitative Portfolio Risk, VaR, DCA & Risk-Parity CLI (Informational Only)")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV or JSON price series (single or multi-asset).")
    parser.add_argument("--periods-per-year", type=int, default=252, help="Periods per year (252 trading days, 52 weeks, 12 months).")
    parser.add_argument("--risk-free", type=float, default=0.03, help="Annual risk-free rate decimal (default: 0.03).")
    parser.add_argument("--budget", type=float, default=12000.0, help="Initial budget for Lump-Sum vs DCA simulation (default: 12000).")
    parser.add_argument("--installments", type=int, default=12, help="Number of DCA installments (clamped to bar count).")
    parser.add_argument("--multi-asset", action="store_true", help="Treat CSV/JSON input as multi-asset columns and compute Inverse-Volatility weights.")
    parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    args = parser.parse_args()

    src = Path(args.input)
    labels, prices = load_price_series(src)
    metrics = analyze_price_series(
        prices=prices,
        labels=labels,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free,
    )
    dca = compare_lump_sum_vs_dca(prices, budget=args.budget, installments=min(args.installments, len(prices)), labels=labels)
    multi = None
    if args.multi_asset:
        _, assets_map = load_multi_asset_prices(src)
        multi = compute_inverse_volatility_weights(assets_map, periods_per_year=args.periods_per_year, risk_free_rate=args.risk_free)

    if args.format == "json":
        payload: dict[str, Any] = {"metrics": asdict(metrics), "dca_vs_lump_sum": dca}
        if multi is not None:
            payload["multi_asset_risk_parity"] = multi
        print(json.dumps(payload, indent=2))
    else:
        print(format_markdown_report(metrics, source_name=src.name, dca_summary=dca, multi_asset_summary=multi))


if __name__ == "__main__":
    main()
