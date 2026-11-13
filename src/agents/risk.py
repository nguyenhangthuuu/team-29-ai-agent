# src/agents/risk.py
from pydantic import BaseModel
from typing import List, Dict

class RiskEvaluation(BaseModel):
    is_approved: bool
    violations: List[str]
    max_weight_allowed: float
    leveraged_exposure_pct: float
    risk_summary: str

class RiskManager:
    def __init__(self, max_single_weight=0.20, max_leveraged_total=0.15, min_cash_buffer=0.05):
        self.max_single_weight = max_single_weight
        self.max_leveraged_total = max_leveraged_total
        self.min_cash_buffer = min_cash_buffer

    def evaluate_portfolio(self, proposed_weights: Dict[str, float], etf_meta_dict: dict) -> RiskEvaluation:
        violations = []
        total_leveraged_weight = 0.0
        invested_weight = sum(proposed_weights.values())

        # Check Cash Buffer
        cash_weight = 1.0 - invested_weight
        if cash_weight < (self.min_cash_buffer - 1e-4):
            violations.append(f"Cash buffer {cash_weight*100:.1f}% below minimum {self.min_cash_buffer*100:.1f}%")

        for ticker, weight in proposed_weights.items():
            meta = etf_meta_dict.get(ticker)
            
            # Check Single Asset Cap
            if weight > self.max_single_weight:
                violations.append(f"{ticker} weight {weight*100:.1f}% exceeds max cap {self.max_single_weight*100:.1f}%")

            # Check Leveraged / Inverse Exposure
            if meta and meta.is_leveraged_or_short:
                total_leveraged_weight += weight
                if weight > 0.08:  # Strict sub-cap for 2x/3x assets
                    violations.append(f"Leveraged ETF {ticker} weight {weight*100:.1f}% exceeds single-leveraged limit of 8%")

        if total_leveraged_weight > self.max_leveraged_total:
            violations.append(f"Total leveraged exposure {total_leveraged_weight*100:.1f}% exceeds limit of {self.max_leveraged_total*100:.1f}%")

        approved = len(violations) == 0
        summary = "Portfolio compliant with risk limits." if approved else f"Rejected with {len(violations)} violations."

        return RiskEvaluation(
            is_approved=approved,
            violations=violations,
            max_weight_allowed=self.max_single_weight,
            leveraged_exposure_pct=round(total_leveraged_weight, 4),
            risk_summary=summary
        )