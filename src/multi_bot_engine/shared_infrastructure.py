"""
Shared Infrastructure for the 7-Bot Institutional Architecture (`src/multi_bot_engine/shared_infrastructure.py`).

Open-Source Quantitative Research & Risk-Analytics Software — Not Investment Advice
(AMF / MiFID II Compliant).

Enforces:
  - AGE_18_MONETIZATION_GATE = True
  - PAPER_TRADING_ONLY = True
  - ZERO_REAL_MONEY_RISK = True
  - LIVE_BROKER_ENABLED = False
  - MAX_LEVERAGE = 1.0
"""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

AGE_18_MONETIZATION_GATE: bool = True
PAPER_TRADING_ONLY: bool = True
ZERO_REAL_MONEY_RISK: bool = True
LIVE_BROKER_ENABLED: bool = False
MAX_LEVERAGE: float = 1.0
LEGAL_NOTICE: str = (
    "Open-Source Quantitative Research & Risk-Analytics Software — "
    "Not Investment Advice (AMF / MiFID II Compliant)"
)

DEFAULT_UNIVERSE: dict[str, dict[str, Any]] = {
    "VWCE_ETF": {"asset_class": "etf", "sector": "global_equity", "expense_ratio": 0.0022, "roic": 0.14, "dividend_yield": 0.019},
    "AGGH_ETF": {"asset_class": "etf", "sector": "sovereign_bond", "expense_ratio": 0.0010, "roic": 0.04, "dividend_yield": 0.032},
    "NOVO_STOCK": {"asset_class": "stock", "sector": "healthcare_biotech", "expense_ratio": 0.0, "roic": 0.28, "dividend_yield": 0.014},
    "ROG_STOCK": {"asset_class": "stock", "sector": "healthcare_pharma", "expense_ratio": 0.0, "roic": 0.22, "dividend_yield": 0.034},
    "MSFT_STOCK": {"asset_class": "stock", "sector": "technology", "expense_ratio": 0.0, "roic": 0.26, "dividend_yield": 0.008},
    "BTC_SPOT": {"asset_class": "crypto", "sector": "digital_asset", "expense_ratio": 0.0, "roic": 0.0, "dividend_yield": 0.0},
    "ETH_SPOT": {"asset_class": "crypto", "sector": "digital_asset", "expense_ratio": 0.0, "roic": 0.0, "dividend_yield": 0.0},
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class AuditEvent:
    timestamp_utc: str
    bot_name: str
    action: str
    details: dict[str, Any]
    event_hash: str = ""

    def __post_init__(self) -> None:
        if not self.event_hash:
            payload = f"{self.timestamp_utc}|{self.bot_name}|{self.action}|{sorted(self.details.items())}"
            self.event_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class PaperPortfolioLedger:
    """Simulated multi-asset paper portfolio (default €10,000, configurable <€1k to €100k+)."""

    initial_capital_eur: float = 10000.0
    cash_eur: float = 10000.0
    peak_nav_eur: float = 10000.0
    positions: dict[str, float] = field(default_factory=dict)
    audit_trail: list[AuditEvent] = field(default_factory=list)
    kill_switch_triggered: bool = False

    def record_audit(self, bot_name: str, action: str, details: dict[str, Any]) -> AuditEvent:
        ev = AuditEvent(timestamp_utc=now_utc_iso(), bot_name=bot_name, action=action, details=details)
        self.audit_trail.append(ev)
        return ev

    def total_nav(self, latest_prices: dict[str, float]) -> float:
        pos_val = sum(qty * latest_prices.get(sym, 0.0) for sym, qty in self.positions.items())
        nav = round(self.cash_eur + pos_val, 4)
        self.peak_nav_eur = max(self.peak_nav_eur, nav)
        return nav

    def current_drawdown(self, latest_prices: dict[str, float]) -> float:
        nav = self.total_nav(latest_prices)
        if self.peak_nav_eur <= 0:
            return 0.0
        return max(0.0, (self.peak_nav_eur - nav) / self.peak_nav_eur)


def build_synthetic_multi_asset_prices(bars: int = 120, seed_shift: int = 0) -> dict[str, list[float]]:
    """Generate deterministic multi-asset Stocks + ETFs + Crypto spot price series."""
    series: dict[str, list[float]] = {}
    params = {
        "VWCE_ETF": (100.0, 0.0006, 0.007, 6.0),
        "AGGH_ETF": (100.0, 0.0002, 0.0025, 8.0),
        "NOVO_STOCK": (100.0, 0.0009, 0.011, 5.0),
        "ROG_STOCK": (100.0, 0.0005, 0.008, 7.0),
        "MSFT_STOCK": (100.0, 0.0008, 0.010, 5.5),
        "BTC_SPOT": (100.0, 0.0011, 0.022, 4.0),
        "ETH_SPOT": (100.0, 0.0009, 0.026, 3.5),
    }
    for sym, (p0, drift, amp, freq) in params.items():
        arr = [p0]
        for i in range(1, bars):
            r = drift + amp * math.sin((i + seed_shift) / freq)
            arr.append(round(arr[-1] * max(1.0 + r, 0.5), 4))
        series[sym] = arr
    return series


def sign_approval_token(secret_key: str, symbol: str, target_weight: float, notional_eur: float) -> str:
    """Create an HMAC-SHA256 approval token binding Risk Management approval to Execution."""
    msg = f"{symbol}|{target_weight:.6f}|{notional_eur:.4f}".encode("utf-8")
    return hmac.new(secret_key.encode("utf-8"), msg, hashlib.sha256).hexdigest()
