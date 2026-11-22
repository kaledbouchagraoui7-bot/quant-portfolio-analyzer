import pytest
import os
import json
from evolution_extensions import PortfolioEvolutionScanner

def test_grade_security_logic():
    assert PortfolioEvolutionScanner.grade_security(0.9)["rating"] == "CRITICAL"
    assert PortfolioEvolutionScanner.grade_security(0.6)["rating"] == "WARNING"
    assert PortfolioEvolutionScanner.grade_security(0.2)["rating"] == "STABLE"

def test_batch_validate_targets_valid():
    targets = [{"ticker": "AAPL", "weight": 0.5}, {"ticker": "MSFT", "weight": 0.5}]
    assert all(PortfolioEvolutionScanner.batch_validate_targets(targets))

def test_batch_validate_targets_invalid():
    targets = [{"ticker": 123, "weight": 0.5}, {"ticker": "GOOG"}]
    results = PortfolioEvolutionScanner.batch_validate_targets(targets)
    assert results[0] is False
    assert results[1] is False

def test_export_sarif_report(tmp_path):
    path = tmp_path / "report.sarif"
    metrics = {"sharpe": 1.5}
    success = PortfolioEvolutionScanner.export_sarif_report(metrics, str(path))
    assert success is True
    with open(path, "r") as f:
        data = json.load(f)
        assert data["runs"][0]["properties"] == metrics