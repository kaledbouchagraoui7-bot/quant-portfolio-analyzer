import json
from typing import Dict, Any, List

class PortfolioSecurityScanner:
    @staticmethod
    def calculate_risk_score(portfolio_data: Dict[str, float]) -> float:
        if not portfolio_data:
            return 0.0
        values = list(portfolio_data.values())
        return sum(values) / len(values)

class ReportGenerator:
    @staticmethod
    def generate_summary_json(metrics: Dict[str, Any], output_path: str) -> bool:
        try:
            report = {
                "timestamp": "2023-10-27T10:00:00Z",
                "metrics": metrics,
                "status": "validated"
            }
            with open(output_path, "w") as f:
                json.dump(report, f)
            return True
        except (IOError, TypeError):
            return False