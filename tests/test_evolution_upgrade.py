import pytest
import os
import json
from quant_portfolio_analyzer.evolution_extensions import PortfolioEvolutionScanner

def test_security_grading():
    assert PortfolioEvolutionScanner.grade_security(0.9)["rating"] == "CRITICAL"
    assert PortfolioEvolutionScanner.grade_security(0.2)["rating"] == "STABLE"

def test_batch_validation():
    targets = [{"ticker": "AAPL", "weight": 0.5}, {"ticker": 123, "weight": 0.1}]
    results = PortfolioEvolutionScanner.batch_validate_targets(targets)
    assert results == [True, False]

def test_sarif_export(tmp_path):
    path = tmp_path / "report.sarif"
    success = PortfolioEvolutionScanner.export_sarif_report({"volatility": 0.1}, str(path))
    assert success is True
    assert os.path.exists(path)

def test_sarif_content(tmp_path):
    path = tmp_path / "content.sarif"
    PortfolioEvolutionScanner.export_sarif_report({"test": "data"}, str(path))
    with open(path, "r") as f:
        data = json.load(f)
        assert data["runs"][0]["properties"] == {"test": "data"}