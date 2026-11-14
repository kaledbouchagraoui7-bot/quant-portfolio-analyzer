import pytest
import os
from evolution_extensions import PortfolioSecurityScanner, ReportGenerator

def test_risk_score_calculation():
    data = {"AAPL": 150.0, "GOOG": 2800.0}
    score = PortfolioSecurityScanner.calculate_risk_score(data)
    assert score == 1475.0

def test_risk_score_empty():
    assert PortfolioSecurityScanner.calculate_risk_score({}) == 0.0

def test_report_generation_success(tmp_path):
    d = tmp_path / "report.json"
    success = ReportGenerator.generate_summary_json({"volatility": 0.05}, str(d))
    assert success is True
    assert d.exists()

def test_report_generation_failure():
    # Invalid path to trigger IOError
    success = ReportGenerator.generate_summary_json({}, "/root/forbidden.json")
    assert success is False