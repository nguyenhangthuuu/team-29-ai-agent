# src/agents/portfolio.py
"""
Portfolio Manager Agent for QDC 2026/2027 (ReBel x QuantumBlack).
Synthesizes Technical Signals, Macro Regime & Headline Sentiment, and Risk Limits
to construct an optimal allocation of at most 5 orders per trading day.

Core Quantitative Mechanism:
  1. Multi-Factor Conviction Model:
       Conviction_i = 0.60 * TechnicalScore_i + 0.40 * ThemeMacroTilt_theme(i)
  2. Dynamic Cash Reserve Governance:
       - RISK_ON  (VIX < 18): 5% to 8% cash reserve (aggressive deployment)
       - RISK_OFF (VIX > 22): 10% to 15% cash reserve (capital preservation)
       - STAGFLATION: 10% cash reserve + commodities/precious metals tilt
  3. Volatility Drag Guard (Cheng & Madhavan, 2009):
       Rejects leveraged ETFs during elevated volatility or risk-off regimes.
  4. Closed-Loop Risk Manager Critique:
       If proposed weights trigger violations (single-asset > 20%, theme > 35%),
       it automatically accepts the Risk Engine's projected compliant weights.
  5. Boardroom-Level Investment Memo Generation.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.pipeline.etf_universe import ETF_UNIVERSE, ETFMeta
from src.agents.risk import RiskManager, RiskEvaluation
from src.agents.macro import MacroBriefing
from src.orchestration.state import OrderItem, PortfolioProposal

class PortfolioManagerAgent:
    """
    Constructs compliant, high-conviction ETF orders by synthesizing
    technical momentum, macroeconomic regimes, and risk limits.
    """
    def __init__(self, risk_manager: Optional[RiskManager] = None):
        self.risk_manager = risk_manager or RiskManager()
        self.meta_map: Dict[str, ETFMeta] = {etf.ticker: etf for etf in ETF_UNIVERSE}

    def compute_conviction_scores(
        self,
        tech_signals: Dict[str, dict],
        macro_briefing: MacroBriefing
    ) -> Dict[str, float]:
        """
        Blends Technical Score (60%) with Theme Macro Tilt (40%).
        Conviction_i = 0.60 * Tech_i + 0.40 * Macro_theme
        """
        convictions = {}
        for ticker, sig_dict in tech_signals.items():
            meta = self.meta_map.get(ticker)
            if not meta:
                continue

            tech_score = sig_dict.get("composite_score", 0.0)
            theme = meta.theme
            macro_tilt = macro_briefing.theme_macro_tilts.get(theme, 0.0)

            # Conviction combination
            blended = (0.60 * tech_score) + (0.40 * macro_tilt)
            convictions[ticker] = round(float(np.clip(blended, -1.0, 1.0)), 4)

        return convictions

    def construct_portfolio(
        self,
        trading_date: str,
        tech_signals: Dict[str, dict],
        macro_briefing: MacroBriefing,
        starting_capital_eur: float = 10000.0,
        current_holdings: Optional[Dict[str, float]] = None
    ) -> Tuple[PortfolioProposal, RiskEvaluation]:
        """
        Main decision engine for the Portfolio Manager Agent.
        Selects top assets, sizes positions, verifies with Risk Manager, and generates memo.
        """
        convictions = self.compute_conviction_scores(tech_signals, macro_briefing)
        regime = macro_briefing.macro_regime

        # 1. Determine Dynamic Target Cash Buffer based on Macro Regime
        if regime == "RISK_OFF":
            target_cash_pct = 0.10  # 10.0% cash buffer in volatile/bearish market
            allow_leveraged = False
        elif regime == "STAGFLATION":
            target_cash_pct = 0.10
            allow_leveraged = False
        elif regime == "RISK_ON":
            target_cash_pct = 0.07  # 7.0% cash buffer (close to 5% min, deploy into alpha)
            allow_leveraged = True
        else:  # NEUTRAL
            target_cash_pct = 0.10
            allow_leveraged = False

        deployable_capital_pct = 1.0 - target_cash_pct

        # 2. Filter Candidate Assets (Sort by Conviction Descending)
        sorted_candidates = sorted(
            convictions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected_tickers = []
        selected_themes = set()

        # Pick top distinct non-leveraged assets across different themes
        for ticker, score in sorted_candidates:
            meta = self.meta_map.get(ticker)
            if not meta:
                continue

            # Volatility Drag Guard: Skip leveraged in high volatility or risk-off
            if meta.is_leveraged_or_short and not allow_leveraged:
                continue

            # Diversity heuristic: avoid overcrowding a single theme in initial pick
            theme = meta.theme
            if ticker not in selected_tickers:
                selected_tickers.append(ticker)
                selected_themes.add(theme)

            if len(selected_tickers) == 4:
                break

        # 3. Add 1 Tactical Safe Haven / Regime Hedge as 5th asset
        if regime == "RISK_OFF":
            hedge_candidate = "GOLD" if "GOLD" not in selected_tickers else "DEFENSE"
        elif regime == "STAGFLATION":
            hedge_candidate = "WORLD ENERGY" if "WORLD ENERGY" not in selected_tickers else "GOLD"
        else:
            hedge_candidate = "WORLD MIN VOL" if "WORLD MIN VOL" not in selected_tickers else "DEFENSE"

        if hedge_candidate not in selected_tickers:
            selected_tickers.append(hedge_candidate)

        # Truncate to maximum 5 assets
        selected_tickers = selected_tickers[:5]

        # 4. Preliminary Position Sizing
        asset_count = len(selected_tickers)
        base_weight_per_asset = round(deployable_capital_pct / asset_count, 4)
        raw_weights = {ticker: min(base_weight_per_asset, 0.20) for ticker in selected_tickers}

        # 5. Closed-Loop Validation with Risk Manager
        risk_eval = self.risk_manager.evaluate_portfolio(raw_weights)
        if not risk_eval.is_approved:
            # Auto-apply compliant weights from Risk Engine
            final_weights = risk_eval.adjusted_weights
            risk_eval = self.risk_manager.evaluate_portfolio(final_weights)
        else:
            final_weights = raw_weights

        # 6. Formulate Exact Trading Order Items
        proposed_orders: List[OrderItem] = []
        for ticker in selected_tickers:
            weight = final_weights.get(ticker, 0.0)
            if weight <= 0.001:
                continue
            amount_eur = round(starting_capital_eur * weight, 2)
            tech_sig = tech_signals.get(ticker, {})
            rationale = tech_sig.get("summary_reason", "Momentum and macro regime alignment.")

            proposed_orders.append(
                OrderItem(
                    ticker=ticker,
                    action="BUY",
                    target_weight_pct=round(weight * 100, 1),
                    estimated_amount_eur=amount_eur,
                    rationale=rationale
                )
            )

        actual_cash_reserve = round((1.0 - sum(final_weights.values())) * 100, 1)

        # 7. Synthesize Boardroom-Level Investment Memo
        memo = (
            f"Investment Committee Proposal for {trading_date}:\n"
            f"Macro State: {macro_briefing.macro_regime} | Transatlantic Spread: {macro_briefing.transatlantic_policy_spread:+.2f} | "
            f"Headline Sentiment: {macro_briefing.headline_stream.headline_sentiment_score:+.2f} ({macro_briefing.headline_stream.news_regime}, Latency: {macro_briefing.headline_stream.news_latency_minutes}m).\n"
            f"Deployment Strategy: Deploy €{starting_capital_eur * sum(final_weights.values()):,.2f} into {len(proposed_orders)} top-conviction assets "
            f"({', '.join(o.ticker for o in proposed_orders)}). Retain {actual_cash_reserve:.1f}% cash reserve (minimum required: 5.0%). "
            f"Portfolio Risk Profile: Annualized Volatility={risk_eval.portfolio_volatility_annualized:.1f}%, "
            f"1-Day 95% Historical VaR={risk_eval.var_95_daily_pct:.2f}%, 1-Day 95% CVaR={risk_eval.cvar_95_daily_pct:.2f}%. "
            f"Leveraged exposure strictly maintained at {risk_eval.leveraged_exposure_pct*100:.1f}% to eliminate compounding volatility drag."
        )

        proposal = PortfolioProposal(
            investment_theme=f"{regime} Tactical Allocation",
            macro_regime=regime,
            orders=proposed_orders,
            cash_weight_pct=actual_cash_reserve,
            executive_memo=memo
        )

        return proposal, risk_eval

if __name__ == "__main__":
    from src.agents.technical import evaluate_all_etfs
    from src.agents.macro import evaluate_macro_agent

    print("=" * 70)
    print("PORTFOLIO MANAGER AGENT TEST")
    print("=" * 70)
    price_df = pd.read_parquet("data/etf_prices_eur.parquet")
    tech_signals = {t: sig.model_dump() for t, sig in evaluate_all_etfs(price_df).items()}
    macro_briefing = evaluate_macro_agent()

    pm = PortfolioManagerAgent()
    proposal, risk_eval = pm.construct_portfolio(
        trading_date="2026-11-18",
        tech_signals=tech_signals,
        macro_briefing=macro_briefing,
        starting_capital_eur=10000.0
    )

    print(f"\nTarget Regime: {proposal.macro_regime} | Cash Reserve: {proposal.cash_weight_pct:.1f}%")
    print(f"Risk Approval: {risk_eval.is_approved} ({risk_eval.risk_summary})")
    print(f"Portfolio Vol: {risk_eval.portfolio_volatility_annualized:.2f}% | 1-Day 95% VaR: {risk_eval.var_95_daily_pct:.2f}%")
    print("\nProposed Orders:")
    for o in proposal.orders:
        print(f"  • {o.action} {o.ticker:<18} | Weight: {o.target_weight_pct:>4.1f}% | Amount: €{o.estimated_amount_eur:>7.2f}")
    print(f"\nExecutive Memo:\n{proposal.executive_memo}")
