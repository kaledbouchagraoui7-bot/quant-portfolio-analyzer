"""
7-Bot Institutional Quantitative Research & Risk-Governance Architecture.

Open-Source Quantitative Research & Risk-Analytics Software — Not Investment Advice
(AMF / MiFID II Compliant). Strictly Paper-Trading Only (`PAPER_TRADING_ONLY = True`,
`ZERO_REAL_MONEY_RISK = True`, `MAX_LEVERAGE = 1.0`, `AGE_18_MONETIZATION_GATE = True`).
"""

from src.multi_bot_engine.orchestrator_bot import run_seven_bot_cycle

__all__ = ["run_seven_bot_cycle"]
