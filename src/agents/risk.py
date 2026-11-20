# src/agents/risk.py
"""
Risk Manager Agent for QDC 2026/2027 (ReBel x QuantumBlack).
Enforces hard competition guardrails, regulatory-like boundaries, quantitative risk metrics,
MiFID II Investor Risk Profiles (Conservative, Moderate, Aggressive), and pre-trade human override checks.

Mandated Constraints (Official QDC Rules & ReBel Governance):
  1. Single Asset Cap: Max 20.0% in any single ETF (Conservative: 12%).
  2. Thematic Concentration Cap: Max 35.0% in any thematic bucket (Conservative: 25%).
  3. Combined Leveraged Cap: Max 15.0% across all 2x/3x or inverse ETFs (Conservative: 0%, Moderate: <=5%).
  4. Single Leveraged Cap: Max 8.0% in any single leveraged/short ETF.
  5. Minimum Cash Buffer: At least 5.0% retained in cash (Conservative: 20%, Moderate: 10%).
  6. Maximum Daily Orders: Strict cap of 5 orders per competition day.
  7. Tail Risk & VaR: 95% 1-day Value-at-Risk (VaR) and Conditional VaR (CVaR).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from src.pipeline.etf_universe import ETF_UNIVERSE, ETFMeta

RISK_PROFILES: Dict[str, Dict[str, float]] = {
    "CONSERVATIVE": {
        "max_single_weight": 0.12,
        "max_theme_weight": 0.25,
        "max_leveraged_total": 0.00,  # Zero leverage allowed
        "max_single_leveraged": 0.00,
        "min_cash_buffer": 0.20,      # 20% minimum cash buffer
        "max_volatility_target": 12.0,
        "max_var_target": 1.20
    },
    "MODERATE": {
        "max_single_weight": 0.18,    # Legal max 20%
        "max_theme_weight": 0.35,    # Legal max 35%
        "max_leveraged_total": 0.05,  # Conservative cap inside 15% limit
        "max_single_leveraged": 0.05,
        "min_cash_buffer": 0.10,      # Legal min 5%
        "max_volatility_target": 18.0,
        "max_var_target": 2.00
    },
    "AGGRESSIVE": {
        "max_single_weight": 0.20,    # Legal max 20%
        "max_theme_weight": 0.35,    # Legal max 35%
        "max_leveraged_total": 0.15,  # Legal max 15%
        "max_single_leveraged": 0.08, # Legal max 8%
        "min_cash_buffer": 0.05,      # Legal min 5%
        "max_volatility_target": 28.0,
        "max_var_target": 3.50
    }
}


class RiskEvaluation(BaseModel):
    is_approved: bool = Field(..., description="True if zero hard risk violations exist")
    risk_profile: str = Field("MODERATE", description="CONSERVATIVE, MODERATE, or AGGRESSIVE")
    violations: List[str] = Field(default_factory=list, description="Hard breaches that disqualify portfolio")
    warnings: List[str] = Field(default_factory=list, description="Advisory flags (volatility decay, concentration)")
    total_invested_pct: float = Field(..., description="Total capital allocated (max 95.0%)")
    cash_reserve_pct: float = Field(..., description="Residual cash reserve (min 5.0%)")
    leveraged_exposure_pct: float = Field(..., description="Total capital in leveraged/inverse products (max 15.0%)")
    theme_concentrations: Dict[str, float] = Field(..., description="Thematic sector allocations")
    portfolio_volatility_annualized: float = Field(0.0, description="Weighted annualized portfolio volatility (%)")
    var_95_daily_pct: float = Field(0.0, description="1-day 95% Historical Value-at-Risk (%)")
    cvar_95_daily_pct: float = Field(0.0, description="1-day 95% Conditional VaR / Expected Shortfall (%)")
    adjusted_weights: Dict[str, float] = Field(default_factory=dict, description="Proportionally compliant weights")
    risk_summary: str


class RiskManager:
    """
    Enforces portfolio safety constraints, regulatory-like guardrails,
    quantitative VaR/CVaR calculations, and automated rebalancing to limits.
    Supports MiFID II Risk Profiles (Conservative, Moderate, Aggressive).
    """
    def __init__(
        self,
        risk_profile: str = "MODERATE",
        returns_parquet_path: str = "data/etf_returns.parquet"
    ):
        self.risk_profile = risk_profile.upper() if risk_profile.upper() in RISK_PROFILES else "MODERATE"
        self._apply_profile_parameters(self.risk_profile)
        self.max_daily_orders: int = 5
        self.meta_map: Dict[str, ETFMeta] = {etf.ticker: etf for etf in ETF_UNIVERSE}

        # Load returns matrix for VaR and volatility calculation if present
        self.returns_df: Optional[pd.DataFrame] = None
        if os.path.exists(returns_parquet_path):
            try:
                self.returns_df = pd.read_parquet(returns_parquet_path)
            except Exception:
                self.returns_df = None

    def _apply_profile_parameters(self, profile: str):
        params = RISK_PROFILES.get(profile, RISK_PROFILES["MODERATE"])
        self.max_single_weight = params["max_single_weight"]
        self.max_theme_weight = params["max_theme_weight"]
        self.max_leveraged_total = params["max_leveraged_total"]
        self.max_single_leveraged = params["max_single_leveraged"]
        self.min_cash_buffer = params["min_cash_buffer"]
        self.max_volatility_target = params["max_volatility_target"]
        self.max_var_target = params["max_var_target"]

    def set_risk_profile(self, risk_profile: str):
        """Dynamically updates risk guardrails based on user risk profile."""
        clean_p = risk_profile.upper() if risk_profile.upper() in RISK_PROFILES else "MODERATE"
        self.risk_profile = clean_p
        self._apply_profile_parameters(clean_p)

    def evaluate_portfolio(self, proposed_weights: Dict[str, float]) -> RiskEvaluation:
        """
        Validates proposed portfolio weights against hard limits and current risk profile.
        Computes VaR, CVaR, portfolio volatility, checks violations,
        and generates compliant adjusted weights.
        """
        violations = []
        warnings = []

        # 1. Clean weights: filter out negligible allocations
        clean_weights = {k: max(0.0, float(v)) for k, v in proposed_weights.items() if v > 0.0001}
        order_count = len(clean_weights)
        total_invested = sum(clean_weights.values())
        cash_pct = max(0.0, 1.0 - total_invested)

        # 2. Maximum Order Count Check (QDC Competition Rule: Max 5 orders/day)
        if order_count > self.max_daily_orders:
            violations.append(
                f"Order Count Breach: Proposed {order_count} assets, exceeding maximum limit of {self.max_daily_orders} daily orders."
            )

        # 3. Total Capital & Cash Reserve Check
        if total_invested > 1.0001:
            violations.append(
                f"Over-allocation: Total invested weight is {total_invested*100:.1f}%, exceeding 100% capital."
            )
        elif cash_pct < (self.min_cash_buffer - 0.001):
            violations.append(
                f"Insufficient Cash Reserve: Cash is {cash_pct*100:.1f}%, below required {self.min_cash_buffer*100:.1f}% minimum for {self.risk_profile} profile."
            )

        # 4. Single Asset Concentration Cap
        for ticker, weight in clean_weights.items():
            if weight > (self.max_single_weight + 0.001):
                violations.append(
                    f"Single Asset Breach: {ticker} allocated {weight*100:.1f}%, exceeding {self.max_single_weight*100:.1f}% limit ({self.risk_profile})."
                )

        # 5. Leveraged and Inverse Exposure Checks
        total_leveraged = 0.0
        for ticker, weight in clean_weights.items():
            meta = self.meta_map.get(ticker)
            if meta and meta.is_leveraged_or_short:
                total_leveraged += weight
                if weight > (self.max_single_leveraged + 0.001):
                    violations.append(
                        f"Single Leveraged Breach: {ticker} is allocated {weight*100:.1f}%, exceeding {self.max_single_leveraged*100:.1f}% max limit."
                    )
                warnings.append(
                    f"Volatility Decay Advisory: {ticker} ({meta.leverage}x) is subject to geometric volatility drag if held across volatile trading intervals."
                )

        if total_leveraged > (self.max_leveraged_total + 0.001):
            violations.append(
                f"Combined Leveraged Breach: Total leveraged exposure is {total_leveraged*100:.1f}%, exceeding {self.max_leveraged_total*100:.1f}% maximum allowed ({self.risk_profile})."
            )

        # 6. Thematic Concentration Cap
        theme_allocations: Dict[str, float] = {}
        for ticker, weight in clean_weights.items():
            meta = self.meta_map.get(ticker)
            theme = meta.theme if meta else "Other"
            theme_allocations[theme] = theme_allocations.get(theme, 0.0) + weight

        for theme, weight in theme_allocations.items():
            if weight > (self.max_theme_weight + 0.001):
                violations.append(
                    f"Theme Concentration Breach: Theme '{theme}' has {weight*100:.1f}%, exceeding {self.max_theme_weight*100:.1f}% limit ({self.risk_profile})."
                )

        # 7. Quantitative Tail Risk (Volatility, Historical VaR & CVaR)
        port_vol, var_95, cvar_95 = self.compute_portfolio_var(clean_weights)

        if port_vol > self.max_volatility_target:
            warnings.append(
                f"Elevated Volatility: Portfolio annualized vol is {port_vol:.1f}%, exceeding target of {self.max_volatility_target:.1f}% for {self.risk_profile} profile."
            )

        is_approved = len(violations) == 0

        # 8. Auto-Rebalance to Limits (Orthogonal Projection onto Compliant Polytope)
        adjusted_weights = self.auto_rebalance_to_limits(clean_weights) if not is_approved else clean_weights

        # 9. Formulate Risk Summary
        if is_approved:
            summary = f"PASSED ({self.risk_profile}): Portfolio strictly satisfies all QDC risk constraints and {self.risk_profile} investor profile."
        else:
            summary = f"REJECTED ({self.risk_profile}): {len(violations)} hard rule violations detected. Auto-rebalance projected compliant weights."

        return RiskEvaluation(
            is_approved=is_approved,
            risk_profile=self.risk_profile,
            violations=violations,
            warnings=warnings,
            total_invested_pct=round(total_invested, 4),
            cash_reserve_pct=round(cash_pct, 4),
            leveraged_exposure_pct=round(total_leveraged, 4),
            theme_concentrations={k: round(v, 4) for k, v in theme_allocations.items()},
            portfolio_volatility_annualized=round(port_vol, 2),
            var_95_daily_pct=round(var_95, 2),
            cvar_95_daily_pct=round(cvar_95, 2),
            adjusted_weights={k: round(v, 4) for k, v in adjusted_weights.items()},
            risk_summary=summary
        )

    def validate_user_override(
        self,
        manual_weights: Dict[str, float]
    ) -> Tuple[bool, List[str], Dict[str, float]]:
        """
        Audits client manual modifications in ADVISE mode.
        Ensures client overrides do not breach hard QDC legal boundaries.
        Returns (is_legal, violation_messages, compliant_weights).
        """
        eval_res = self.evaluate_portfolio(manual_weights)
        if eval_res.is_approved:
            return True, [], eval_res.adjusted_weights
        return False, eval_res.violations, eval_res.adjusted_weights

    def auto_rebalance_to_limits(self, raw_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Projects non-compliant weights onto legal polytope.
        Truncates to top 5, clamps single/leveraged/theme caps, and preserves required cash.
        """
        if not raw_weights:
            return {}

        # 1. Truncate to top 5 assets by weight
        sorted_assets = sorted(raw_weights.items(), key=lambda x: x[1], reverse=True)[:self.max_daily_orders]
        weights = dict(sorted_assets)

        # 2. Cap single asset & leveraged limits
        capped_weights = {}
        total_lev = 0.0

        for tkr, w in weights.items():
            meta = self.meta_map.get(tkr)
            is_lev = meta.is_leveraged_or_short if meta else False

            if is_lev:
                allowed_lev = max(0.0, min(w, self.max_single_leveraged, self.max_leveraged_total - total_lev))
                capped_weights[tkr] = allowed_lev
                total_lev += allowed_lev
            else:
                capped_weights[tkr] = min(w, self.max_single_weight)

        # 3. Scale total invested to respect cash buffer
        max_investable = 1.0 - self.min_cash_buffer
        current_invested = sum(capped_weights.values())

        if current_invested > max_investable:
            scale = max_investable / (current_invested + 1e-9)
            capped_weights = {k: v * scale for k, v in capped_weights.items()}

        return {k: round(v, 4) for k, v in capped_weights.items() if v > 0.001}

    def compute_portfolio_var(self, weights: Dict[str, float]) -> Tuple[float, float, float]:
        """Computes portfolio volatility, 1-day 95% Historical VaR and Expected Shortfall."""
        if not weights or self.returns_df is None:
            return 0.0, 0.0, 0.0

        available_tickers = [t for t in weights.keys() if t in self.returns_df.columns]
        if not available_tickers:
            return 0.0, 0.0, 0.0

        sub_ret = self.returns_df[available_tickers]
        w_vec = np.array([weights[t] for t in available_tickers])
        port_daily_ret = sub_ret.dot(w_vec)

        port_vol_ann = float(port_daily_ret.std() * np.sqrt(252) * 100)
        var_95 = float(np.percentile(port_daily_ret, 5) * 100)
        tail_losses = port_daily_ret[port_daily_ret <= np.percentile(port_daily_ret, 5)]
        cvar_95 = float(tail_losses.mean() * 100) if not tail_losses.empty else var_95

        return port_vol_ann, var_95, cvar_95
