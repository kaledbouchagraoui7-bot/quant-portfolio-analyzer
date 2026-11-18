import json
from typing import Dict, Any, List, TypedDict
from datetime import datetime

class SecurityGrade(TypedDict):
    score: float
    rating: str

class PortfolioEvolutionScanner:
    @staticmethod
    def grade_security(risk_score: float) -> SecurityGrade:
        if risk_score > 0.8: rating = "CRITICAL"
        elif risk_score > 0.5: rating = "WARNING"
        else: rating = "STABLE"
        return {"score": risk_score, "rating": rating}

    @staticmethod
    def batch_validate_targets(targets: List[Dict[str, Any]]) -> List[bool]:
        return [isinstance(t.get("ticker"), str) and isinstance(t.get("weight"), (int, float)) for t in targets]

    @staticmethod
    def export_sarif_report(metrics: Dict[str, Any], output_path: str) -> bool:
        try:
            report = {
                "version": "2.1.0",
                "runs": [{
                    "tool": {"driver": {"name": "QuantAnalyzer", "version": "1.0.0"}},
                    "results": [{"message": {"text": "Portfolio validation complete"}, "properties": metrics}]
                }],
                "timestamp": datetime.now().isoformat()
            }
            with open(output_path, "w") as f:
                json.dump(report, f)
            return True
        except Exception:
            return False