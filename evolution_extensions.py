import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Callable
from functools import wraps

class PortfolioReporter:
    @staticmethod
    def export_to_sarif(data: Dict[str, Any], filepath: str) -> bool:
        try:
            sarif = {
                "version": "2.1.0",
                "runs": [{
                    "tool": {"driver": {"name": "quant_portfolio_analyzer", "version": "1.0.0"}},
                    "results": [{"message": {"text": json.dumps(data)}, "level": "note"}]
                }]
            }
            with open(filepath, "w") as f:
                json.dump(sarif, f, indent=4)
            return True
        except Exception:
            return False

class DataValidator:
    @staticmethod
    def validate_batch(prices: List[float]) -> bool:
        return bool(prices and len(prices) >= 2 and all(isinstance(p, (int, float)) and p >= 0 for p in prices))

def validate_integration(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if result is None:
            raise ValueError("Integration error: Function returned None")
        return result
    return wrapper