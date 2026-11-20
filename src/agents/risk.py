# src/agents/risk.py
"""
Risk Manager Agent for QDC 2026/2027 (ReBel x QuantumBlack).
Enforces hard competition guardrails, regulatory-like boundaries, quantitative risk metrics,
and mitigates volatility decay on leveraged/inverse ETFs (Cheng & Madhavan, 2009).

Mandated Constraints (Official QDC Rules & ReBel Governance):
  1. Single Asset Cap: Max 20.0% in any single ETF.
  2. Thematic Concentration Cap: Max 35.0% in any thematic bucket.
  3. Combined Leveraged Cap: Max 15.0% across all 2x/3x or inverse ETFs.
  4. Single Leveraged Cap: Max 8.0% in any single leveraged/short ETF.
  5. Minimum Cash Buffer: At least 5.0% retained in cash (expands up to 20% in RISK_OFF).
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

class RiskEvaluation(BaseModel):
    is_approved: bool = Field(..., description="True if zero hard risk violations exist")
    violations: List[str] = Field(default_factory=list, description="Hard breaches that disqualify portfolio")
    warnings: List[str] = Field(default_factory=list, description="Advisory flags (volatility decay, concentration)")
    total_invested_pct: float = Field(..., description="Total capital allocated (max 95.0%)")
    cash_reserve_pct: float = Field(..., description="Residual cash reserve (min 5.0%)")
    leveraged_exposure_pct: float = Field(..., description="Total capital in leveraged/inverse products (max 15.0%)")
    theme_concentrations: Dict[str, float] = Field(..., description="Thematic sector allocations (max 35.0% per theme)")
    portfolio_volatility_annualized: float = Field(0.0, description="Weighted annualized portfolio volatility (%)")
    var_95_daily_pct: float = Field(0.0, description="1-day 95% Historical Value-at-Risk (%)")
    cvar_95_daily_pct: float = Field(0.0, description="1-day 95% Conditional VaR / Expected Shortfall (%)")
    adjusted_weights: Dict[str, float] = Field(default_factory=dict, description="Proportionally compliant weights")
    risk_summary: str

class RiskManager:
    """
    Enforces portfolio safety constraints, regulatory-like guardrails,
    quantitative VaR/CVaR calculations, and automated rebalancing to limits.
    """
    def __init__(
        self,
        max_single_weight: float = 0.20,       # Max 20% in any single ETF
        max_leveraged_total: float = 0.15,     # Max 15% combined across leveraged/inverse ETFs
        max_single_leveraged: float = 0.08,    # Max 8% in a single 2x/3x or inverse ETF
        min_cash_buffer: float = 0.05,         # At least 5% retained in cash
        max_theme_weight: float = 0.35,        # Max 35% in any single thematic bucket
        max_daily_orders: int = 5,             # Max 5 orders per trading day
        returns_parquet_path: str = "data/etf_returns.parquet"
    ):
        self.max_single_weight = max_single_weight
        self.max_leveraged_total = max_leveraged_total
        self.max_single_leveraged = max_single_leveraged
        self.min_cash_buffer = min_cash_buffer
        self.max_theme_weight = max_theme_weight
        self.max_daily_orders = max_daily_orders
        self.meta_map: Dict[str, ETFMeta] = {etf.ticker: etf for etf in ETF_UNIVERSE}

        # Load returns matrix for VaR and volatility calculation if present
        self.returns_df: Optional[pd.DataFrame] = None
        if os.path.exists(returns_parquet_path):
            try:
                self.returns_df = pd.read_parquet(returns_parquet_path)
            except Exception:
                self.returns_df = None

    def evaluate_portfolio(self, proposed_weights: Dict[str, float]) -> RiskEvaluation:
        """
        Validates proposed portfolio weights against hard limits.
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
                f"Insufficient Cash Reserve: Cash is {cash_pct*100:.1f}%, below required {self.min_cash_buffer*100:.1f}% minimum."
            )

        # 4. Single ETF Concentration & Leveraged Exposure Check
        total_leveraged_pct = 0.0
        theme_allocations: Dict[str, float] = {}

        for ticker, weight in clean_weights.items():
            meta = self.meta_map.get(ticker)
            if not meta:
                warnings.append(f"Unrecognized ticker '{ticker}' not found in official 50 ETF list.")
                continue

            theme = meta.theme
            theme_allocations[theme] = theme_allocations.get(theme, 0.0) + weight

            # Single asset cap (20%)
            if weight > (self.max_single_weight + 0.0001):
                violations.append(
                    f"Concentration Breach: {ticker} ({meta.full_name}) weight {weight*100:.1f}% exceeds single limit of {self.max_single_weight*100:.1f}%."
                )

            # Leveraged / Short specific limits (8% single, 15% combined)
            if meta.is_leveraged_or_short:
                total_leveraged_pct += weight
                if weight > (self.max_single_leveraged + 0.0001):
                    violations.append(
                        f"Leveraged Cap Breach: {ticker} (Leverage: {meta.leverage}x) weight {weight*100:.1f}% exceeds max single leveraged cap of {self.max_single_leveraged*100:.1f}%."
                    )
                warnings.append(
                    f"Volatility Decay Risk (Cheng & Madhavan, 2009): {ticker} ({meta.leverage}x) faces daily compounding erosion. Holding horizon should not exceed 5 trading days."
                )

        # Total Leveraged Cap Check (15%)
        if total_leveraged_pct > (self.max_leveraged_total + 0.0001):
            violations.append(
                f"Total Leveraged Exposure Breach: Combined leveraged/inverse allocation {total_leveraged_pct*100:.1f}% exceeds portfolio cap of {self.max_leveraged_total*100:.1f}%."
            )

        # Thematic Sector Limits (35%)
        for theme, alloc in theme_allocations.items():
            if alloc > (self.max_theme_weight + 0.0001):
                violations.append(
                    f"Thematic Over-concentration: Theme '{theme}' has {alloc*100:.1f}% allocation, exceeding thematic cap of {self.max_theme_weight*100:.1f}%."
                )

        # 5. Compute Quantitative Risk Metrics (Volatility, VaR-95%, CVaR-95%)
        port_vol_ann, var_95, cvar_95 = self._compute_portfolio_risk_metrics(clean_weights)

        # 6. Auto-Adjust Non-Compliant Weights to Bounds
        adjusted = self.auto_rebalance_to_limits(clean_weights)

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
            portfolio_volatility_annualized=round(port_vol_ann, 2),
            var_95_daily_pct=round(var_95, 2),
            cvar_95_daily_pct=round(cvar_95, 2),
            adjusted_weights={k: round(v, 4) for k, v in adjusted.items()},
            risk_summary=summary
        )

    def auto_rebalance_to_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Proportionally clips and normalizes weights to guarantee compliance across:
          - Single asset caps (20% non-leveraged, 8% leveraged)
          - Thematic concentration caps (35% per theme)
          - Combined leveraged cap (15%)
          - Cash reserve guarantee (min 5%)
          - Order count cap (top 5 assets)
        """
        if not weights:
            return {}

        # 1. Retain only top 5 assets by weight if more than 5 proposed
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:self.max_daily_orders]
        adjusted = dict(sorted_items)

        # 2. Clip single asset caps
        for ticker, weight in adjusted.items():
            meta = self.meta_map.get(ticker)
            cap = self.max_single_leveraged if (meta and meta.is_leveraged_or_short) else self.max_single_weight
            adjusted[ticker] = min(weight, cap)

        # 3. Cap thematic concentration to max_theme_weight (35%)
        theme_groups: Dict[str, List[str]] = {}
        for ticker in adjusted:
            meta = self.meta_map.get(ticker)
            theme = meta.theme if meta else "Other"
            theme_groups.setdefault(theme, []).append(ticker)

        for theme, tickers in theme_groups.items():
            tot_theme = sum(adjusted[t] for t in tickers)
            if tot_theme > self.max_theme_weight and tot_theme > 0:
                scale = self.max_theme_weight / tot_theme
                for t in tickers:
                    adjusted[t] *= scale

        # 4. Cap total leveraged exposure (15%)
        lev_tickers = [t for t in adjusted if self.meta_map.get(t) and self.meta_map[t].is_leveraged_or_short]
        tot_lev = sum(adjusted[t] for t in lev_tickers)
        if tot_lev > self.max_leveraged_total and tot_lev > 0:
            scale = self.max_leveraged_total / tot_lev
            for t in lev_tickers:
                adjusted[t] *= scale

        # 5. Guarantee cash buffer (min 5%)
        tot_invested = sum(adjusted.values())
        max_allowed_invested = 1.0 - self.min_cash_buffer
        if tot_invested > max_allowed_invested and tot_invested > 0:
            scale = max_allowed_invested / tot_invested
            for t in adjusted:
                adjusted[t] *= scale

        return {k: round(v, 4) for k, v in adjusted.items() if v > 0.001}

    def _compute_portfolio_risk_metrics(self, weights: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Calculates 1-day 95% Historical VaR, 1-day 95% CVaR, and annualized portfolio volatility.
        """
        if self.returns_df is None or not weights:
            return 0.0, 0.0, 0.0

        available_tickers = [t for t in weights if t in self.returns_df.columns]
        if not available_tickers:
            return 0.0, 0.0, 0.0

        sub_ret = self.returns_df[available_tickers].dropna()
        if sub_ret.empty:
            return 0.0, 0.0, 0.0

        w_vec = np.array([weights[t] for t in available_tickers])
        port_daily_ret = sub_ret.dot(w_vec)

        # Annualized volatility
        port_vol_ann = float(port_daily_ret.std() * np.sqrt(252) * 100)

        # 1-day 95% Historical VaR and CVaR
        var_95 = float(np.percentile(port_daily_ret, 5) * 100)
        tail_losses = port_daily_ret[port_daily_ret <= np.percentile(port_daily_ret, 5)]
        cvar_95 = float(tail_losses.mean() * 100) if not tail_losses.empty else var_95

        return port_vol_ann, var_95, cvar_95

if __name__ == "__main__":
    rm = RiskManager()
    sample_weights = {
        "EUROPE 600": 0.18,
        "BIOTECH": 0.18,
        "WORLD MIN VOL": 0.18,
        "DIVIDEND LEADERS": 0.18,
        "DEFENSE": 0.18
    }
    eval_res = rm.evaluate_portfolio(sample_weights)
    print("=" * 65)
    print("RISK MANAGER AUDIT EVALUATION")
    print("=" * 65)
    print(f"Status Approved   : {eval_res.is_approved}")
    print(f"Total Invested    : {eval_res.total_invested_pct*100:.1f}% | Cash Reserve: {eval_res.cash_reserve_pct*100:.1f}%")
    print(f"Leveraged Exposure: {eval_res.leveraged_exposure_pct*100:.1f}%")
    print(f"Annualized Vol    : {eval_res.portfolio_volatility_annualized:.2f}%")
    print(f"1-Day 95% VaR     : {eval_res.var_95_daily_pct:.2f}% | 1-Day 95% CVaR: {eval_res.cvar_95_daily_pct:.2f}%")
    if eval_res.violations:
        print("\nViolations:")
        for v in eval_res.violations:
            print(f"  ❌ {v}")
    print(f"\nAuto-Adjusted Weights for Compliance:")
    for t, w in eval_res.adjusted_weights.items():
        print(f"  • {t:<20}: {w*100:.1f}%")
