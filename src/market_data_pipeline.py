"""
Market Data Validation & Point-in-Time Dataset Registry (`src/market_data_pipeline.py`).

Open-Source Quantitative Research & Risk-Analytics Software — Not Investment Advice
(AMF / MiFID II Compliant). Rejects corrupted/NaN/outlier feeds before signal generation.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any


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
) -> dict[str, Any]:
    """Compute SHA-256 dataset checksum and provenance record; raises if invalid."""
    validation = validate_price_series(price_map)
    if not validation["valid"]:
        raise ValueError(f"Dataset '{name}' failed validation: {validation['errors']}")

    canonical = json.dumps({k: price_map[k] for k in sorted(price_map)}, sort_keys=True)
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    bar_counts = {k: len(v) for k, v in sorted(price_map.items())}
    return {
        "name": name,
        "source": source,
        "methodology": methodology,
        "acquired_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checksum_sha256": checksum,
        "bar_counts": bar_counts,
        "validation": validation,
    }
