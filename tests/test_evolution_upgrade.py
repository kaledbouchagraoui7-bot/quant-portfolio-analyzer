import pytest
import os
from evolution_extensions import PortfolioEvolutionScanner, MockDataLayer

def test_security_grading():
    assert PortfolioEvolutionScanner.grade_security(0.9)["rating"] == "CRITICAL"
    assert PortfolioEvolutionScanner.grade_security(0.2)["rating"] == "STABLE"

def test_batch_validation():
    targets = [{"ticker": "AAPL", "weight": 0.5}, {"ticker": 123, "weight": 0.1}]
    results = PortfolioEvolutionScanner.batch_validate_targets(targets)
    assert results == [True, False]

def test_mock_data_layer():
    data = MockDataLayer.fetch_mock_market_data("MSFT")
    assert data["ticker"] == "MSFT"
    assert data["price"] == 100.0

def test_sarif_export(tmp_path):
    path = tmp_path / "report.sarif"
    success = PortfolioEvolutionScanner.export_sarif_report({"test": 1}, str(path))
    assert success is True
    assert os.path.exists(path)