import pytest
from evolution_extensions import PortfolioReporter, DataValidator, validate_integration
import os

def test_sarif_export():
    data = {"sharpe": 1.5}
    path = "test.sarif"
    assert PortfolioReporter.export_to_sarif(data, path) is True
    assert os.path.exists(path)
    os.remove(path)

def test_batch_validation():
    assert DataValidator.validate_batch([10.0, 11.0]) is True
    assert DataValidator.validate_batch([10.0]) is False
    assert DataValidator.validate_batch([-1.0, 2.0]) is False

def test_validation_decorator():
    @validate_integration
    def mock_func(): return True
    assert mock_func() is True

def test_validation_decorator_failure():
    @validate_integration
    def fail_func(): return None
    with pytest.raises(ValueError):
        fail_func()