#!/usr/bin/env python3
"""
SEC EDGAR XBRL Fundamental Health & Resilience Scorer (`sec_fundamentals.py`)

Adapted from public SEC EDGAR XBRL company-facts patterns (inspired by
`SEC-API-io/sec-api-python` endpoint/backoff handling and `jankrepl/deepdow`
finite/NaN data integrity checks).

Computes deterministic fundamental health ratios from public SEC EDGAR XBRL
JSON (`https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json`) or local
offline JSON files:
  - Revenue CAGR (%) across multi-year annual 10-K/FY points
  - Operating Margin (%)
  - Free Cash Flow Proxy (Operating Cash Flow - CapEx)
  - Debt-to-Cash Ratio
  - Combined Fundamental + Drawdown Resilience Score (0-100)

Informational & Educational Analytics Only: Deterministic math on public SEC
filings — not certified financial advice, investment recommendations, or live trading.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


DISCLAIMER = (
    "Informational Quantitative & Fundamental Analytics Only: Outputs are deterministic "
    "educational/research calculations on public SEC EDGAR XBRL filings or user-supplied JSON — "
    "not certified financial advice or investment recommendations."
)

SEC_USER_AGENT = "KaledBouchagraouiResearchCLI/1.0 (kaledbouchagraoui7@gmail.com)"

# Built-in fallback CIK map so offline/rate-limited ticker queries still resolve common mega-caps
COMMON_TICKER_CIKS: dict[str, tuple[str, str]] = {
    "AAPL": ("0000320193", "Apple Inc."),
    "MSFT": ("0000789019", "Microsoft Corp."),
    "NVDA": ("0001045810", "NVIDIA Corp."),
    "AMZN": ("0001018724", "Amazon.com Inc."),
    "GOOGL": ("0001652044", "Alphabet Inc."),
    "META": ("0001326801", "Meta Platforms Inc."),
    "BRK.B": ("0001067983", "Berkshire Hathaway Inc."),
}

REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
OP_INC_TAGS = ("OperatingIncomeLoss",)
OCF_TAGS = ("NetCashProvidedByUsedInOperatingActivities",)
CAPEX_TAGS = (
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "CapitalExpendituresIncurredButNotYetPaid",
)
DEBT_TAGS = (
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "DebtAndCapitalLeaseObligations",
)
CASH_TAGS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
)


@dataclass(frozen=True)
class FundamentalMetrics:
    entity_name: str
    cik: str
    fiscal_years: list[int]
    latest_revenue: float
    revenue_cagr_pct: float
    operating_margin_pct: float
    free_cash_flow_proxy: float
    debt_to_cash_ratio: float | None
    fundamental_score: float
    drawdown_resilience_score: float
    combined_resilience_score: float
    disclaimer: str = DISCLAIMER


def _is_finite_number(val: Any) -> bool:
    """`deepdow`-style numerical hygiene guard: reject None, bool, NaN, and +-inf."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val))


def _sec_get_json(url: str, user_agent: str = SEC_USER_AGENT, retries: int = 3, timeout: int = 20) -> dict[str, Any]:
    """Fetch public SEC JSON with compliant User-Agent and exponential backoff on 429/503."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            status = getattr(exc, "code", None)
            if status in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(0.4 * (2 ** attempt))
                continue
            raise
    raise RuntimeError(f"SEC EDGAR request failed for {url}: {last_exc}")


def resolve_ticker_to_cik(ticker: str, user_agent: str = SEC_USER_AGENT) -> tuple[str, str]:
    """Resolve a US stock ticker or 10-digit CIK string into `(cik_10_digits, entity_name)`."""
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Ticker or CIK cannot be empty.")
    if cleaned.isdigit() or (cleaned.startswith("CIK") and cleaned[3:].isdigit()):
        digits = cleaned[3:] if cleaned.startswith("CIK") else cleaned
        return digits.zfill(10), f"CIK {digits.zfill(10)}"
    if cleaned in COMMON_TICKER_CIKS:
        return COMMON_TICKER_CIKS[cleaned]

    data = _sec_get_json("https://www.sec.gov/files/company_tickers.json", user_agent=user_agent)
    for row in data.values():
        if str(row.get("ticker", "")).upper() == cleaned:
            cik_str = str(row.get("cik_str", "")).zfill(10)
            title = str(row.get("title", cleaned))
            return cik_str, title
    raise ValueError(f"Could not resolve ticker '{ticker}' via SEC company_tickers.json.")


def _extract_annual_series(
    facts_us_gaap: dict[str, Any],
    candidate_tags: Sequence[str],
    instant_ok: bool = False,
) -> dict[int, float]:
    """
    Extract annual (`10-K` / `FY`) finite numeric series keyed by fiscal year.
    Handles duplicate restatements by keeping the latest `filed` entry per fiscal year.
    """
    for tag in candidate_tags:
        tag_obj = facts_us_gaap.get(tag)
        if not isinstance(tag_obj, dict):
            continue
        units = tag_obj.get("units", {})
        usd_rows = units.get("USD") or []
        if not usd_rows:
            continue

        best_by_fy: dict[int, tuple[str, float]] = {}
        for item in usd_rows:
            val = item.get("val")
            if not _is_finite_number(val):
                continue
            form = str(item.get("form", ""))
            fp = str(item.get("fp", ""))
            if form not in ("10-K", "10-K/A", "20-F") and fp != "FY":
                continue
            # Prefer full-year durations for flow statements unless instant balance-sheet item
            frame = str(item.get("frame", ""))
            if not instant_ok and frame and "Q" in frame:
                continue
            fy = item.get("fy")
            if not isinstance(fy, int):
                end_str = str(item.get("end", ""))
                if len(end_str) >= 4 and end_str[:4].isdigit():
                    fy = int(end_str[:4])
                else:
                    continue
            filed = str(item.get("filed", item.get("end", "")))
            prev = best_by_fy.get(fy)
            if prev is None or filed >= prev[0]:
                best_by_fy[fy] = (filed, float(val))

        if best_by_fy:
            return {fy: pair[1] for fy, pair in sorted(best_by_fy.items())}
    return {}


def compute_sec_fundamentals(
    company_facts: dict[str, Any],
    max_drawdown_pct: float = 0.0,
) -> FundamentalMetrics:
    """
    Compute deterministic fundamental health metrics + Combined Fundamental/Drawdown
    Resilience Score (0-100) from an SEC EDGAR `companyfacts` JSON dictionary.
    """
    if not isinstance(company_facts, dict):
        raise ValueError("company_facts must be a JSON object.")

    entity_name = str(company_facts.get("entityName") or "Unknown Entity")
    cik = str(company_facts.get("cik") or "0000000000").zfill(10)
    facts_root = company_facts.get("facts", {})
    us_gaap = facts_root.get("us-gaap", {}) if isinstance(facts_root, dict) else {}
    if not us_gaap:
        raise ValueError("SEC companyfacts JSON contains no 'facts.us-gaap' section.")

    rev_series = _extract_annual_series(us_gaap, REVENUE_TAGS)
    if not rev_series:
        raise ValueError("No finite annual revenue observations found in us-gaap facts.")

    years = sorted(rev_series.keys())
    latest_fy = years[-1]
    latest_rev = rev_series[latest_fy]
    if len(years) >= 2 and rev_series[years[0]] > 0 and latest_rev > 0:
        span = max(1, years[-1] - years[0])
        rev_cagr_pct = ((latest_rev / rev_series[years[0]]) ** (1.0 / span) - 1.0) * 100.0
    else:
        rev_cagr_pct = 0.0

    op_series = _extract_annual_series(us_gaap, OP_INC_TAGS)
    latest_op = op_series.get(latest_fy, op_series[max(op_series)] if op_series else 0.0)
    op_margin_pct = (latest_op / latest_rev * 100.0) if latest_rev > 0 else 0.0

    ocf_series = _extract_annual_series(us_gaap, OCF_TAGS)
    capex_series = _extract_annual_series(us_gaap, CAPEX_TAGS)
    latest_ocf = ocf_series.get(latest_fy, ocf_series[max(ocf_series)] if ocf_series else 0.0)
    latest_capex = abs(capex_series.get(latest_fy, capex_series[max(capex_series)] if capex_series else 0.0))
    fcf_proxy = latest_ocf - latest_capex

    debt_series = _extract_annual_series(us_gaap, DEBT_TAGS, instant_ok=True)
    cash_series = _extract_annual_series(us_gaap, CASH_TAGS, instant_ok=True)
    latest_debt = debt_series.get(latest_fy, debt_series[max(debt_series)] if debt_series else 0.0)
    latest_cash = cash_series.get(latest_fy, cash_series[max(cash_series)] if cash_series else 0.0)
    debt_to_cash = round(latest_debt / latest_cash, 3) if latest_cash > 0 else None

    # Transparent deterministic scoring (0-100):
    # - Growth (0-25): Revenue CAGR capped between -10%..+25%
    growth_pts = max(0.0, min(25.0, ((rev_cagr_pct + 10.0) / 35.0) * 25.0))
    # - Profitability (0-25): Operating margin capped 0%..30%
    margin_pts = max(0.0, min(25.0, (op_margin_pct / 30.0) * 25.0))
    # - Free cash flow positive & healthy FCF margin vs revenue (0-25)
    fcf_margin = (fcf_proxy / latest_rev * 100.0) if latest_rev > 0 else 0.0
    fcf_pts = max(0.0, min(25.0, (fcf_margin / 20.0) * 25.0)) if fcf_proxy > 0 else 0.0
    # - Balance sheet leverage (0-25): lower Debt-to-Cash earns higher points (0.0 -> 25, >=4.0 -> 0)
    if debt_to_cash is None:
        leverage_pts = 0.0 if latest_debt > 0 else 15.0
    else:
        leverage_pts = max(0.0, min(25.0, ((4.0 - min(4.0, debt_to_cash)) / 4.0) * 25.0))

    fundamental_score = round(growth_pts + margin_pts + fcf_pts + leverage_pts, 2)
    dd_clamped = max(0.0, min(100.0, float(max_drawdown_pct)))
    drawdown_resilience = round(max(0.0, 100.0 - dd_clamped * 1.5), 2)
    combined_score = round(0.65 * fundamental_score + 0.35 * drawdown_resilience, 2)

    return FundamentalMetrics(
        entity_name=entity_name,
        cik=cik,
        fiscal_years=years,
        latest_revenue=round(latest_rev, 2),
        revenue_cagr_pct=round(rev_cagr_pct, 2),
        operating_margin_pct=round(op_margin_pct, 2),
        free_cash_flow_proxy=round(fcf_proxy, 2),
        debt_to_cash_ratio=debt_to_cash,
        fundamental_score=fundamental_score,
        drawdown_resilience_score=drawdown_resilience,
        combined_resilience_score=combined_score,
    )


def load_or_fetch_company_facts(target: str, user_agent: str = SEC_USER_AGENT) -> dict[str, Any]:
    """Load local SEC companyfacts JSON file if `target` is a path, else query SEC EDGAR."""
    candidate = Path(target)
    if candidate.exists() and candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))

    cik, _ = resolve_ticker_to_cik(target, user_agent=user_agent)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return _sec_get_json(url, user_agent=user_agent)


def format_fundamentals_markdown(fm: FundamentalMetrics, max_drawdown_pct: float = 0.0) -> str:
    d2c = f"{fm.debt_to_cash_ratio:.3f}x" if fm.debt_to_cash_ratio is not None else "N/A (cash <= 0)"
    lines = [
        f"## SEC EDGAR XBRL Fundamental Health (`{fm.entity_name}`, CIK `{fm.cik}`)",
        "",
        f"> **Notice:** {fm.disclaimer}",
        "",
        "| Metric | Value | Meaning (Plain Language) |",
        "| :--- | ---: | :--- |",
        f"| Fiscal Years Covered | `{fm.fiscal_years[0]}-{fm.fiscal_years[-1]}` | Multi-year annual 10-K/FY span (`{len(fm.fiscal_years)}` yrs) |",
        f"| Latest Annual Revenue | `${fm.latest_revenue:,.0f}` | Most recent full-year top-line sales |",
        f"| Revenue CAGR | `{fm.revenue_cagr_pct:+.2f}%` | Smoothed yearly sales growth rate |",
        f"| Operating Margin | `{fm.operating_margin_pct:.2f}%` | Cents of operating profit kept per dollar of sales |",
        f"| Free Cash Flow Proxy | `${fm.free_cash_flow_proxy:,.0f}` | Operating cash flow minus capital expenditures |",
        f"| Debt-to-Cash Ratio | `{d2c}` | Long-term debt compared to cash on hand |",
        f"| Fundamental Health Score | `{fm.fundamental_score:.2f} / 100` | Growth + Margin + FCF + Low-Debt composite |",
        f"| Drawdown Resilience (`{max_drawdown_pct:.2f}%` DD) | `{fm.drawdown_resilience_score:.2f} / 100` | Price stability score against historical worst drop |",
        f"| **Combined Resilience Score** | **`{fm.combined_resilience_score:.2f} / 100`** | **65% Fundamentals + 35% Drawdown Resilience** |",
    ]
    return "\n".join(lines) + "\n"
