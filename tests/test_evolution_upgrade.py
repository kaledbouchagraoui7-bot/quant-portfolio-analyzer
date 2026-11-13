import pytest
from evolution_extensions import PortfolioReporter, DataValidator
import os

def test_validator_valid_data():
    assert DataValidator.validate_batch([10.0, 11.0, 12.0]) is True

def test_validator_invalid_data():
    assert DataValidator.validate_batch([-1.0, 10.0]) is False
    assert DataValidator.validate_batch([]) is False

def test_reporter_export(tmp_path):
    d = tmp_path / "report.json"
    data = {"sharpe": 1.5}
    assert PortfolioReporter.export_to_json(data, str(d)) is True
    assert d.exists()

def test_reporter_invalid_path():
    assert PortfolioReporter.export_to_json({}, "/nonexistent/path/file.json") is False