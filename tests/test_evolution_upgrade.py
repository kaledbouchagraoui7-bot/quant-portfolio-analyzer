import pytest
import os
from evolution_extensions import PortfolioEvolutionScanner

def test_grade_security_critical():
    result = PortfolioEvolutionScanner.grade_security(0.9)
    assert result["rating"] == "CRITICAL"

def test_grade_security_stable():
    result = PortfolioEvolutionScanner.grade_security(0.2)
    assert result["rating"] == "STABLE"

def test_batch_validate_targets():
    targets = [{"ticker": "AAPL", "weight": 0.5}, {"ticker": 123, "weight": 0.1}]
    results = PortfolioEvolutionScanner.batch_validate_targets(targets)
    assert results == [True, False]

def test_export_sarif_report(tmp_path):
    path = tmp_path / "report.sarif"
    success = PortfolioEvolutionScanner.export_sarif_report({"risk": 0.1}, str(path))
    assert success is True
    assert os.path.exists(path)