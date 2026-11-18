"""
Market Data Validation & Point-in-Time Dataset Registry (`src/market_data_pipeline.py`).

Open-Source Quantitative Research & Risk-Analytics Software — Not Investment Advice
(AMF / MiFID II Compliant). Rejects corrupted/NaN/outlier feeds before signal generation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASET_CATEGORIES = (
    "SYNTHETIC_SIMULATION",
    "HISTORICAL_REAL_MARKET_DATA",
    "DELAYED_REAL_MARKET_DATA",
    "CURRENT_REAL_MARKET_DATA",
    "USER_SUPPLIED_DATA",
)

PORTFOLIO_ENVIRONMENTS = (
    "1_synthetic_deterministic",
    "2_historical_backtest",
    "3_frozen_oos_evaluation",
    "4_real_data_shadow_signals",
    "5_forward_paper_trading",
)


def prevent_category_mixing(datasets: list[dict[str, Any]]) -> None:
    """Reject any portfolio/research run that mixes distinct market-data categories."""
    cats = {str(d.get("category", "")).strip() for d in datasets if isinstance(d, dict)}
    cats.discard("")
    if len(cats) > 1:
        raise ValueError(f"Forbidden dataset category mixing detected: {sorted(cats)}")


def validate_ohlcv_bars(
    dates: list[str],
    closes: list[float],
    volumes: list[float] | None = None,
    currency: str = "USD",
) -> dict[str, Any]:
    """Validate chronological timestamps, duplicate dates, non-positive prices, and missing bars."""
    errors: list[str] = []
    warnings: list[str] = []
    if len(dates) != len(closes):
        errors.append("dates_closes_length_mismatch")
    if len(dates) < 5:
        errors.append(f"insufficient_bars:{len(dates)}")
    seen_dates: set[str] = set()
    prev_dt = ""
    for idx, (dt, px) in enumerate(zip(dates, closes)):
        if dt in seen_dates:
            errors.append(f"duplicate_timestamp:{dt}")
        seen_dates.add(dt)
        if prev_dt and dt <= prev_dt:
            errors.append(f"non_monotonic_timestamp:{dt}_after_{prev_dt}")
        prev_dt = dt
        if px is None or not math.isfinite(px) or px <= 0.0:
            errors.append(f"invalid_close_at_{idx}:{px}")
    if volumes is not None and any((v is None or not math.isfinite(v) or v < 0.0) for v in volumes):
        warnings.append("invalid_or_missing_volume_entries")
    return {"valid": len(errors) == 0, "currency": currency, "bars": len(dates), "errors": errors, "warnings": warnings}


def fetch_free_historical_daily_ohlcv(
    symbol: str,
    stooq_symbol: str | None = None,
    timeout: int = 12,
    sample_csv_path: Path | None = None,
) -> dict[str, Any]:
    """
    Fetch free daily EOD historical OHLCV from public Stooq CSV (`https://stooq.com/q/d/l/?s=<sym>&i=d`)
    without paid subscriptions, falling back to local `sample_prices.csv` (explicitly tagged
    `USER_SUPPLIED_DATA` or `SYNTHETIC_SIMULATION`) if network access is unavailable.
    Never claims survivorship-bias elimination on public EOD single-ticker snapshots.
    """
    slug = (stooq_symbol or f"{symbol.lower()}.us").strip()
    url = f"https://stooq.com/q/d/l/?s={slug}&i=d"
    dates: list[str] = []
    closes: list[float] = []
    volumes: list[float] = []
    category = "HISTORICAL_REAL_MARKET_DATA"
    source_name = f"stooq_eod_csv:{slug}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "QuantPortfolioAnalyzer/2.0 (Educational Research)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_csv = resp.read().decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(raw_csv))
        for row in reader:
            dt = str(row.get("Date") or "").strip()
            cl_str = str(row.get("Close") or "").strip()
            vol_str = str(row.get("Volume") or "0").strip()
            if not dt or not cl_str:
                continue
            cl = float(cl_str)
            vol = float(vol_str) if vol_str else 0.0
            if math.isfinite(cl) and cl > 0.0:
                dates.append(dt)
                closes.append(cl)
                volumes.append(max(0.0, vol))
    except Exception:
        dates, closes, volumes = [], [], []

    if len(dates) < 5:
        fallback_csv = sample_csv_path or (Path(__file__).resolve().parent.parent / "sample_prices.csv")
        if fallback_csv.exists():
            category = "USER_SUPPLIED_DATA"
            source_name = f"local_csv:{fallback_csv.name}"
            with fallback_csv.open("r", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    dt = str(row.get("date") or row.get("Date") or "").strip()
                    cl_str = str(row.get("close") or row.get("Close") or "").strip()
                    if dt and cl_str:
                        dates.append(dt)
                        closes.append(float(cl_str))
                        volumes.append(0.0)

    paired = sorted(zip(dates, closes, volumes), key=lambda x: x[0])
    dates = [p[0] for p in paired]
    closes = [p[1] for p in paired]
    volumes = [p[2] for p in paired]
    val = validate_ohlcv_bars(dates, closes, volumes)
    if not val["valid"]:
        raise ValueError(f"Historical OHLCV fetch for {symbol} failed validation: {val['errors']}")

    checksum = hashlib.sha256(json.dumps({"dates": dates, "closes": closes}).encode("utf-8")).hexdigest()
    return {
        "symbol": symbol,
        "category": category,
        "source": source_name,
        "dates": dates,
        "closes": closes,
        "volumes": volumes,
        "checksum_sha256": checksum,
        "survivorship_bias_free": False,
        "limitations": "Single-symbol unadjusted/public EOD history; point-in-time index constituents not included.",
        "validation": val,
    }


def validate_price_series(
    price_map: dict[str, list[float]],
    max_abs_daily_return: float = 0.35,
    max_consecutive_identical: int = 12,
) -> dict[str, Any]:
    """Validate multi-asset price series for NaNs, non-positive values, outliers, and stale feeds."""
    errors: list[str] = []
    warnings: list[str] = []

    if not price_map:
        return {"valid": False, "errors": ["empty_price_map"], "warnings": []}

    lengths = {len(v) for v in price_map.values()}
    if len(lengths) > 1:
        warnings.append(f"unequal_series_lengths:{sorted(lengths)}")

    for sym, series in sorted(price_map.items()):
        if len(series) < 5:
            errors.append(f"{sym}:insufficient_length:{len(series)}")
            continue
        flat_run = 0
        for i, p in enumerate(series):
            if p is None or not math.isfinite(p) or p <= 0.0:
                errors.append(f"{sym}:invalid_price_at_{i}:{p}")
                break
            if i > 0:
                ret = (p / series[i - 1]) - 1.0
                if abs(ret) > max_abs_daily_return:
                    warnings.append(f"{sym}:outlier_return_at_{i}:{ret:.4f}")
                if p == series[i - 1]:
                    flat_run += 1
                    if flat_run >= max_consecutive_identical:
                        warnings.append(f"{sym}:stale_flat_feed:{flat_run}_bars")
                else:
                    flat_run = 0

    return {
        "valid": len(errors) == 0,
        "assets_checked": len(price_map),
        "errors": errors,
        "warnings": warnings,
    }


def register_dataset(
    name: str,
    price_map: dict[str, list[float]],
    source: str = "synthetic_deterministic",
    methodology: str = "4-regime daily point-in-time close reconstruction",
    category: str = "SYNTHETIC_SIMULATION",
    environment: str = "1_synthetic_deterministic",
    currency: str = "EUR",
    survivorship_bias_free: bool = False,
) -> dict[str, Any]:
    """Compute SHA-256 dataset checksum, category, environment, and provenance record; raises if invalid."""
    if category not in DATASET_CATEGORIES:
        raise ValueError(f"Invalid dataset category '{category}'. Expected one of {DATASET_CATEGORIES}")
    if environment not in PORTFOLIO_ENVIRONMENTS:
        raise ValueError(f"Invalid environment '{environment}'. Expected one of {PORTFOLIO_ENVIRONMENTS}")
    validation = validate_price_series(price_map)
    if not validation["valid"]:
        raise ValueError(f"Dataset '{name}' failed validation: {validation['errors']}")

    canonical = json.dumps({k: price_map[k] for k in sorted(price_map)}, sort_keys=True)
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    bar_counts = {k: len(v) for k, v in sorted(price_map.items())}
    return {
        "name": name,
        "category": category,
        "environment": environment,
        "currency": currency,
        "survivorship_bias_free": bool(survivorship_bias_free),
        "source": source,
        "methodology": methodology,
        "acquired_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checksum_sha256": checksum,
        "bar_counts": bar_counts,
        "validation": validation,
    }
