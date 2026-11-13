# `quant-portfolio-analyzer`

> **Informational Quantitative & Mathematical Analytics Only: Outputs are deterministic educational/research math calculations on public or user-supplied CSV/JSON series — not certified financial advice, investment recommendations, or live brokerage execution.**

Deterministic CLI and zero-install browser calculator (`demo.html`) for portfolio risk, compounding, and educational position-sizing math on CSV/JSON price series. Adapted cleanly from open-source Track A quantitative finance patterns (reference attribution: `mbk-dev/okama`, license `MIT`).

## Capabilities
- Computes Total Return, CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown (peak & trough), Calmar Ratio, and capped Fractional Kelly position sizing.
- Zero brokerage connections, zero API keys, and full offline `pytest` verification.

## Quick Start
```bash
python quant_portfolio_analyzer.py --input sample_prices.csv --periods-per-year 252 --format md
pytest -q
```
