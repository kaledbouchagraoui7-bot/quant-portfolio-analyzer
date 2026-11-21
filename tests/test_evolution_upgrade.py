import os
from evolution_extensions import PortfolioEvolutionScanner

def test_grade_security_logic():
    assert PortfolioEvolutionScanner.grade_security(0.9)["rating"] == "CRITICAL"
    assert PortfolioEvolutionScanner.grade_security(0.2)["rating"] == "STABLE"

def test_batch_validation_success():
    targets = [{"ticker": "AAPL", "weight": 0.5}, {"ticker": "MSFT", "weight": 0.5}]
    assert all(PortfolioEvolutionScanner.batch_validate_targets(targets))

def test_batch_validation_failure():
    targets = [{"ticker": 123, "weight": "invalid"}]
    assert not any(PortfolioEvolutionScanner.batch_validate_targets(targets))

def test_sarif_export(tmp_path):
    path = tmp_path / "report.sarif"
    success = PortfolioEvolutionScanner.export_sarif_report({"risk": 0.1}, str(path))
    assert success is True
    assert os.path.exists(path)