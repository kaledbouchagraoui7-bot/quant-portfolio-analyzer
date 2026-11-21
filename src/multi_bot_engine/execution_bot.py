"""
Bot 6 — Execution Bot (`src/multi_bot_engine/execution_bot.py`).

Simulated Paper-Only Order Router (`LIVE_BROKER_ENABLED = False`, `PAPER_TRADING_ONLY = True`).
Executes ONLY orders bearing a valid HMAC `ApprovalToken` issued by `risk_management_bot.py`.
Any unapproved or forged order is rejected immediately with `PermissionError`.
Not Investment Advice (AMF / MiFID II Compliant).
"""

from __future__ import annotations

import contextlib
import hmac
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from src.multi_bot_engine.risk_management_bot import ApprovedOrder
from src.multi_bot_engine.shared_infrastructure import (
    LIVE_BROKER_ENABLED,
    PAPER_TRADING_ONLY,
    ZERO_REAL_MONEY_RISK,
    PaperPortfolioLedger,
    get_risk_gate_secret,
    sign_approval_token,
)

_DEFAULT_NONCE_DB = Path(__file__).resolve().parent.parent.parent / ".runtime" / "consumed_nonces.sqlite3"
_REGISTRY_LOCK = threading.Lock()


class PersistentNonceRegistry:
    """
    Transactional SQLite WAL anti-replay registry surviving process restarts and serializing
    concurrent multi-thread/multi-process redemptions with `BEGIN IMMEDIATE` + `UNIQUE` nonce PK.
    Fails closed (`PermissionError`) if corrupted, locked beyond timeout, or replayed.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        env_db = os.environ.get("QPA_NONCE_DB_PATH", "").strip()
        self.db_path = (db_path or (Path(env_db) if env_db else _DEFAULT_NONCE_DB)).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.db_path, timeout=15.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            return conn
        except sqlite3.DatabaseError as exc:
            raise PermissionError(f"Fail-closed: nonce persistence corrupted/unreadable: {exc}") from exc

    def _ensure_schema(self) -> None:
        with _REGISTRY_LOCK:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS consumed_nonces (
                            nonce TEXT PRIMARY KEY,
                            symbol TEXT NOT NULL,
                            consumed_at_epoch REAL NOT NULL,
                            expires_at_epoch REAL NOT NULL
                        )
                        """
                    )
            except sqlite3.DatabaseError as exc:
                raise PermissionError(f"Fail-closed: nonce DB schema check failed: {exc}") from exc

    def consume_once(self, nonce: str, symbol: str, expires_at_epoch: float) -> None:
        now = time.time()
        with _REGISTRY_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM consumed_nonces WHERE expires_at_epoch < ?", (now - 3600.0,))
                conn.execute(
                    "INSERT INTO consumed_nonces (nonce, symbol, consumed_at_epoch, expires_at_epoch) VALUES (?, ?, ?, ?)",
                    (nonce, symbol, now, float(expires_at_epoch)),
                )
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                raise PermissionError(f"ExecutionBot rejected replayed/concurrent approval nonce for {symbol}.") from exc
            except sqlite3.DatabaseError as exc:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                raise PermissionError(f"Fail-closed: nonce DB transaction failed: {exc}") from exc
            finally:
                conn.close()


def verify_risk_approval(order: Any, consume: bool = True, nonce_db_path: Path | None = None) -> bool:
    """
    Reject any order that did not pass through `risk_management_bot`, has wrong component binding,
    forged/tampered parameters, expired TTL, unavailable signing key, or replayed/concurrent nonce.
    """
    if not isinstance(order, ApprovedOrder):
        raise PermissionError("ExecutionBot blocked unapproved order: must be an ApprovedOrder from RiskManagementBot.")
    if not order.approval_token or not order.nonce or order.expires_at_epoch <= 0.0:
        raise PermissionError(f"ExecutionBot rejected order missing nonce/expiration metadata for {order.symbol}.")
    if order.issuing_component != "risk_management_bot" or order.authorized_executor != "execution_bot":
        raise PermissionError(f"ExecutionBot rejected invalid component binding for {order.symbol}.")
    if time.time() > order.expires_at_epoch:
        raise PermissionError(f"ExecutionBot rejected expired risk approval token for {order.symbol}.")
    try:
        secret = get_risk_gate_secret()
    except Exception as exc:
        raise PermissionError(f"Fail-closed: signing key unavailable for {order.symbol}: {exc}") from exc

    expected = sign_approval_token(
        secret,
        order.symbol,
        order.asset_class,
        order.target_weight,
        order.notional_eur,
        nonce=order.nonce,
        expires_at_epoch=order.expires_at_epoch,
        direction=order.direction,
        quantity=order.quantity,
        max_slippage_bps=order.max_slippage_bps,
        issuing_component=order.issuing_component,
        authorized_executor=order.authorized_executor,
        strategy_version=order.strategy_version,
        risk_policy_version=order.risk_policy_version,
        portfolio_version=order.portfolio_version,
        issued_at_epoch=order.issued_at_epoch,
    )
    if not hmac.compare_digest(order.approval_token, expected):
        raise PermissionError(f"ExecutionBot rejected forged/invalid risk approval token for {order.symbol}.")
    if consume:
        PersistentNonceRegistry(nonce_db_path).consume_once(order.nonce, order.symbol, order.expires_at_epoch)
    return True


def execute_paper_orders(
    approved_orders: list[ApprovedOrder],
    latest_prices: dict[str, float],
    ledger: PaperPortfolioLedger,
    slippage_bps: float = 5.0,
    commission_bps: float = 2.0,
) -> dict[str, Any]:
    """Route verified paper orders with realistic spread/slippage & commission fees."""
    assert PAPER_TRADING_ONLY and ZERO_REAL_MONEY_RISK and (LIVE_BROKER_ENABLED is False)

    executed: list[dict[str, Any]] = []
    total_fees_eur = 0.0
    fee_rate = (slippage_bps + commission_bps) / 10000.0

    for order in approved_orders:
        verify_risk_approval(order, consume=True)
        price = float(latest_prices.get(order.symbol, 0.0))
        if price <= 0.0:
            continue

        cur_qty = ledger.positions.get(order.symbol, 0.0)
        target_qty = order.notional_eur / price
        delta_qty = target_qty - cur_qty
        trade_notional = abs(delta_qty) * price
        fee_eur = round(trade_notional * fee_rate, 4)

        if abs(delta_qty) * price < 1.0:
            continue

        ledger.cash_eur = round(ledger.cash_eur - (delta_qty * price) - fee_eur, 4)
        ledger.positions[order.symbol] = round(target_qty, 6)
        total_fees_eur += fee_eur
        executed.append(
            {
                "symbol": order.symbol,
                "delta_qty": round(delta_qty, 6),
                "fill_price": price,
                "fee_eur": fee_eur,
                "target_weight": order.target_weight,
            }
        )

    nav_after = ledger.total_nav(latest_prices)
    ledger.record_audit(
        "execution_bot",
        "paper_orders_routed",
        {"executed_count": len(executed), "total_fees_eur": round(total_fees_eur, 4), "nav_eur": nav_after},
    )
    return {
        "live_broker_enabled": LIVE_BROKER_ENABLED,
        "paper_trading_only": PAPER_TRADING_ONLY,
        "executed_orders": executed,
        "total_fees_eur": round(total_fees_eur, 4),
        "nav_after_eur": nav_after,
    }
