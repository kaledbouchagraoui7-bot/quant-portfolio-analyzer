import json
from datetime import datetime
from typing import Any, Dict, List

class PortfolioReporter:
    @staticmethod
    def export_to_json(data: Dict[str, Any], filepath: str) -> bool:
        try:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": data,
                "status": "success"
            }
            with open(filepath, "w") as f:
                json.dump(report, f, indent=4)
            return True
        except Exception:
            return False

class DataValidator:
    @staticmethod
    def validate_batch(prices: List[float]) -> bool:
        if not prices or len(prices) < 2:
            return False
        return all(isinstance(p, (int, float)) and p >= 0 for p in prices)