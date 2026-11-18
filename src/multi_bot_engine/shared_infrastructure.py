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
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    import keyring  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    keyring = None

_RUNTIME_HMAC_SECRET: str = ""
_KEYRING_SERVICE = "quant_portfolio_analyzer"
_KEYRING_USERNAME = "risk_gate_hmac_v2"

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
    nav_history: list[float] = field(default_factory=list)
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

    def record_nav(self, latest_prices: dict[str, float]) -> float:
        nav = self.total_nav(latest_prices)
        self.nav_history.append(nav)
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


def get_risk_gate_secret(allow_ephemeral_test_fallback: bool = False) -> str:
    """
    Resolve the persistent HMAC risk-gate secret from:
      1) Fail-closed check if `QPA_FORCE_KEYSTORE_UNAVAILABLE == "1"` (testing / fail-closed gate),
      2) `QPA_RISK_GATE_SECRET` environment variable (testing / CI override),
      3) OS Credential Store (`keyring`), generating & persisting a 32-byte CSPRNG key with readback verification.
    Refuses ephemeral keys in operational mode; only permits ephemeral keys when `allow_ephemeral_test_fallback=True`.
    """
    global _RUNTIME_HMAC_SECRET

    if os.environ.get("QPA_FORCE_KEYSTORE_UNAVAILABLE", "").strip() == "1":
        if not allow_ephemeral_test_fallback:
            raise RuntimeError("Fail-closed: persistent OS-backed signing key unavailable.")

    env_secret = os.environ.get("QPA_RISK_GATE_SECRET", "").strip()
    if env_secret == "__MISSING__":
        raise RuntimeError("Fail-closed: signing key explicitly missing.")
    if env_secret:
        if len(env_secret) < 16:
            raise RuntimeError("Fail-closed: QPA_RISK_GATE_SECRET must be at least 16 characters.")
        return env_secret

    if keyring is not None and os.environ.get("QPA_FORCE_KEYSTORE_UNAVAILABLE", "").strip() != "1":
        try:
            stored = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
            if stored and len(stored.strip()) >= 16:
                _RUNTIME_HMAC_SECRET = stored.strip()
                return _RUNTIME_HMAC_SECRET
            generated = secrets.token_hex(32)
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, generated)
            readback = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
            if not readback or readback.strip() != generated:
                raise RuntimeError("Keyring persistence readback verification failed.")
            _RUNTIME_HMAC_SECRET = generated
            return _RUNTIME_HMAC_SECRET
        except Exception as exc:
            if not allow_ephemeral_test_fallback:
                raise RuntimeError(f"Fail-closed: persistent OS keyring error: {exc}") from exc

    if allow_ephemeral_test_fallback:
        _RUNTIME_HMAC_SECRET = secrets.token_hex(32)
        return _RUNTIME_HMAC_SECRET
    raise RuntimeError("Fail-closed: no persistent OS-backed signing key backend available.")


def sign_approval_token(
    secret_key: str,
    symbol: str,
    asset_class: str,
    target_weight: float,
    notional_eur: float,
    nonce: str = "",
    expires_at_epoch: float = 0.0,
    direction: str = "BUY",
    quantity: float = 0.0,
    max_slippage_bps: float = 15.0,
    issuing_component: str = "risk_management_bot",
    authorized_executor: str = "execution_bot",
    strategy_version: str = "v2.1",
    risk_policy_version: str = "ALCOA-Paper-v2.1",
    portfolio_version: str = "v1",
    issued_at_epoch: float = 0.0,
) -> str:
    """Create an HMAC-SHA256 approval token cryptographically binding all 13 order & policy fields."""
    if not secret_key or len(secret_key) < 16:
        raise RuntimeError("Fail-closed: cannot sign approval token with empty/weak secret.")
    msg = (
        f"{symbol}|{asset_class}|{direction}|{quantity:.6f}|{target_weight:.8f}|{notional_eur:.2f}|"
        f"{max_slippage_bps:.2f}|{issuing_component}|{authorized_executor}|{strategy_version}|"
        f"{risk_policy_version}|{portfolio_version}|{issued_at_epoch:.3f}|{expires_at_epoch:.3f}|{nonce}"
    ).encode("utf-8")
    return hmac.new(secret_key.encode("utf-8"), msg, hashlib.sha256).hexdigest()
