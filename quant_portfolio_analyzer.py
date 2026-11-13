#!/usr/bin/env python3
"""
Quantitative Portfolio Risk, Compounding & Backtest Metrics Analyzer (`quant_portfolio_analyzer.py`)

Adapted locally inside `obsidian-second-brain` using clean open-source quantitative finance
algorithms inspired by `bshaw2019/Crypto_Trader` (MIT).
Zero live brokerage credentials, zero trading execution — pure mathematical analytics.

Informational Quantitative & Mathematical Analytics Only: Outputs are deterministic educational/research math calculations on public or user-supplied CSV/JSON series — not certified financial advice, investment recommendations, or live brokerage execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


DISCLAIMER = 'Informational Quantitative & Mathematical Analytics Only: Outputs are deterministic educational/research math calculations on public or user-supplied CSV/JSON series — not certified financial advice, investment recommendations, or live brokerage execution.'


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
    full_kelly_pct: float
    half_kelly_pct: float
    disclaimer: str = DISCLAIMER


def load_price_series(file_path: Path) -> tuple[list[str], list[float]]:
    """Load `(labels, prices)` from a CSV (date,close|price|value) or JSON file."""
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
            raise ValueError(f"CSV must contain one of close/adj_close/price/value/nav columns; found {reader.fieldnames}")
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


def compute_kelly_fraction(
    win_rate: float,
    payoff_ratio: float,
    max_cap: float = 0.25,
) -> tuple[float, float]:
    """
    Compute educational Kelly Criterion position fraction:
        f* = (p * b - (1 - p)) / b = p - (1 - p) / b
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
    """Return `(max_drawdown_decimal, peak_label, trough_label)` where drawdown is <= 0.0 (e.g. -0.20)."""
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
    """Compute CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar, and Kelly sizing cleanly."""
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
        full_kelly_pct=round(full_k * 100.0, 2),
        half_kelly_pct=round(half_k * 100.0, 2),
    )


def format_markdown_report(metrics: PortfolioMetrics, source_name: str = "sample_series") -> str:
    return "\n".join([
        f"# Quantitative Portfolio Risk & Compounding Report (`{source_name}`)",
        "",
        f"> **Notice:** {metrics.disclaimer}",
        "",
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
        f"| Educational Kelly (Half / Capped Full) | `{metrics.half_kelly_pct:.2f}%` / `{metrics.full_kelly_pct:.2f}%` | Theoretical position sizing cap (25% hard cap) |",
    ]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantitative Portfolio Risk & Backtest Metrics CLI (Informational Only)")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV (date,close) or JSON price series.")
    parser.add_argument("--periods-per-year", type=int, default=252, help="Periods per year (252 trading days, 52 weeks, 12 months).")
    parser.add_argument("--risk-free", type=float, default=0.03, help="Annual risk-free rate decimal (default: 0.03).")
    parser.add_argument("--format", choices=("md", "json"), default="md", help="Output format.")
    args = parser.parse_args()

    labels, prices = load_price_series(Path(args.input))
    metrics = analyze_price_series(
        prices=prices,
        labels=labels,
        periods_per_year=args.periods_per_year,
        risk_free_rate=args.risk_free,
    )
    if args.format == "json":
        print(json.dumps(asdict(metrics), indent=2))
    else:
        print(format_markdown_report(metrics, source_name=Path(args.input).name))


if __name__ == "__main__":
    main()
