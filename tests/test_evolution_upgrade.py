import pytest
import os
from quant_portfolio_analyzer.evolution_extensions import PortfolioEvolutionScanner

def test_grade_security_logic():
    assert PortfolioEvolutionScanner.grade_security(0.9)["rating"] == "CRITICAL"
    assert PortfolioEvolutionScanner.grade_security(0.6)["rating"] == "WARNING"
    assert PortfolioEvolutionScanner.grade_security(0.2)["rating"] == "STABLE"

def test_batch_validate_targets():
    targets = [{"ticker": "AAPL", "weight": 0.5}, {"ticker": 123, "weight": 0.1}]
    results = PortfolioEvolutionScanner.batch_validate_targets(targets)
    assert results == [True, False]

def test_export_sarif_report_success(tmp_path):
    path = tmp_path / "report.sarif"
    success = PortfolioEvolutionScanner.export_sarif_report({"volatility": 0.05}, str(path))
    assert success is True
    assert os.path.exists(path)

def test_export_sarif_report_failure():
    # Attempt to write to an invalid path to trigger exception
    success = PortfolioEvolutionScanner.export_sarif_report({}, "/root/forbidden.sarif")
    assert success is False