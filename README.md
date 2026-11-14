# `quant-portfolio-analyzer`

> **Informational Quantitative & Mathematical Analytics Only:** Outputs are deterministic educational/research math calculations on public or user-supplied CSV/JSON series — not certified financial advice, investment recommendations, or live brokerage execution.

Deterministic Python CLI and zero-install browser calculator (`demo.html` / [GitHub Pages](https://kaledbouchagraoui7-bot.github.io/quant-portfolio-analyzer/)) for portfolio risk, tail loss, compounding, and risk-parity allocation analytics, adapted cleanly from open-source Track A patterns (`shashankvemuri/Finance` [MIT], `skfolio/skfolio` [BSD-3-Clause]).

## Capabilities
- **Performance & Ratios:** Total Return, CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Maximum Drawdown (peak/trough), Calmar Ratio, and capped Fractional Kelly Sizing (25% hard cap).
- **Tail Risk:** Historical & Parametric Value-at-Risk (`VaR_95`, `VaR_99`) plus Historical Expected Shortfall (`CVaR_95`).
- **Compounding Simulator:** Lump-Sum vs. spaced Dollar-Cost Averaging (DCA) portfolio comparison.
- **Multi-Asset Allocation:** Inverse-Volatility / Risk-Parity basket weighting across multiple CSV/JSON price columns.

## Quick Start
```bash
python quant_portfolio_analyzer.py --input sample_prices.csv --periods-per-year 12 --format md
pytest -v
```
