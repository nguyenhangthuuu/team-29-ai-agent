# src/agents/risk.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from src.pipeline.etf_universe import ETF_UNIVERSE, ETFMeta

class RiskEvaluation(BaseModel):
    is_approved: bool
    violations: List[str]
    warnings: List[str]
    total_invested_pct: float
    cash_reserve_pct: float
    leveraged_exposure_pct: float
    theme_concentrations: Dict[str, float]
    adjusted_weights: Dict[str, float]
    risk_summary: str

class RiskManager:
    """
    Enforces portfolio safety constraints, regulatory-like guardrails, and mitigates
    volatility decay risks on leveraged/inverse ETFs.
    """
    def __init__(
        self,
        max_single_weight: float = 0.20,       # Max 20% in any single ETF
        max_leveraged_total: float = 0.15,     # Max 15% combined across all leveraged/inverse ETFs
        max_single_leveraged: float = 0.08,    # Max 8% in a single 2x/3x or inverse ETF
        min_cash_buffer: float = 0.05,         # At least 5% retained in cash
        max_theme_weight: float = 0.35         # Max 35% in any single thematic bucket
    ):
        self.max_single_weight = max_single_weight
        self.max_leveraged_total = max_leveraged_total
        self.max_single_leveraged = max_single_leveraged
        self.min_cash_buffer = min_cash_buffer
        self.max_theme_weight = max_theme_weight
        self.meta_map: Dict[str, ETFMeta] = {etf.ticker: etf for etf in ETF_UNIVERSE}

    def evaluate_portfolio(self, proposed_weights: Dict[str, float]) -> RiskEvaluation:
        """
        Validates proposed portfolio weights against hard risk limits.
        If violations exist, it generates an actionable critique for the Portfolio Manager agent
        and calculates compliant auto-adjusted weights.
        """
        violations = []
        warnings = []
        
        # Clean weights: ignore zero or negative allocations
        clean_weights = {k: max(0.0, float(v)) for k, v in proposed_weights.items() if v > 0.0001}
        total_invested = sum(clean_weights.values())
        cash_pct = max(0.0, 1.0 - total_invested)

        # 1. Total Capital & Cash Reserve Check
        if total_invested > 1.0001:
            violations.append(f"Over-allocation: Total invested weight is {total_invested*100:.1f}%, exceeding 100% capital.")
        elif cash_pct < (self.min_cash_buffer - 0.001):
            violations.append(f"Insufficient Cash Reserve: Cash is {cash_pct*100:.1f}%, below required {self.min_cash_buffer*100:.1f}% minimum.")

        # 2. Single ETF Concentration & Leveraged Exposure Check
        total_leveraged_pct = 0.0
        theme_allocations: Dict[str, float] = {}

        for ticker, weight in clean_weights.items():
            meta = self.meta_map.get(ticker)
            if not meta:
                warnings.append(f"Unrecognized ticker '{ticker}' not found in official 50 ETF list.")
                continue

            # Theme tracking
            theme = meta.theme
            theme_allocations[theme] = theme_allocations.get(theme, 0.0) + weight

            # Single asset cap
            if weight > self.max_single_weight:
                violations.append(
                    f"Concentration Breach: {ticker} ({meta.full_name}) weight {weight*100:.1f}% exceeds single limit of {self.max_single_weight*100:.1f}%."
                )

            # Leveraged / Short specific rules
            if meta.is_leveraged_or_short:
                total_leveraged_pct += weight
                if weight > self.max_single_leveraged:
                    violations.append(
                        f"Leveraged Cap Breach: {ticker} (Leverage: {meta.leverage}x) weight {weight*100:.1f}% exceeds max single leveraged cap of {self.max_single_leveraged*100:.1f}%."
                    )
                warnings.append(
                    f"Volatility Decay Risk: {ticker} ({meta.leverage}x) is subject to daily compounding erosion. Recommended holding horizon < 5 trading days."
                )

        # Total Leveraged Cap Check
        if total_leveraged_pct > self.max_leveraged_total:
            violations.append(
                f"Total Leveraged Exposure Breach: Combined leveraged/inverse allocation {total_leveraged_pct*100:.1f}% exceeds portfolio cap of {self.max_leveraged_total*100:.1f}%."
            )

        # Thematic Sector Limits
        for theme, alloc in theme_allocations.items():
            if alloc > self.max_theme_weight:
                violations.append(
                    f"Thematic Over-concentration: Theme '{theme}' has {alloc*100:.1f}% allocation, exceeding thematic cap of {self.max_theme_weight*100:.1f}%."
                )

        # 3. Compute Auto-Adjusted Compliant Weights
        adjusted = self._auto_adjust_weights(clean_weights)

        is_approved = len(violations) == 0
        if is_approved:
            summary = "PASSED: Portfolio strictly satisfies all QDC risk constraints and capital rules."
        else:
            summary = f"REJECTED: Portfolio triggered {len(violations)} risk violations. Action required before human submission."

        return RiskEvaluation(
            is_approved=is_approved,
            violations=violations,
            warnings=warnings,
            total_invested_pct=round(total_invested, 4),
            cash_reserve_pct=round(cash_pct, 4),
            leveraged_exposure_pct=round(total_leveraged_pct, 4),
            theme_concentrations={k: round(v, 4) for k, v in theme_allocations.items()},
            adjusted_weights={k: round(v, 4) for k, v in adjusted.items()},
            risk_summary=summary
        )

    def _auto_adjust_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Proportionally clips and normalizes weights to guarantee compliance."""
        adjusted = {}
        for ticker, weight in weights.items():
            meta = self.meta_map.get(ticker)
            cap = self.max_single_leveraged if (meta and meta.is_leveraged_or_short) else self.max_single_weight
            adjusted[ticker] = min(weight, cap)

        # Cap total leveraged
        lev_tickers = [t for t in adjusted if self.meta_map.get(t) and self.meta_map[t].is_leveraged_or_short]
        tot_lev = sum(adjusted[t] for t in lev_tickers)
        if tot_lev > self.max_leveraged_total and tot_lev > 0:
            scale = self.max_leveraged_total / tot_lev
            for t in lev_tickers:
                adjusted[t] *= scale

        # Ensure cash buffer
        tot_invested = sum(adjusted.values())
        max_allowed_invested = 1.0 - self.min_cash_buffer
        if tot_invested > max_allowed_invested and tot_invested > 0:
            scale = max_allowed_invested / tot_invested
            for t in adjusted:
                adjusted[t] *= scale

        return adjusted
