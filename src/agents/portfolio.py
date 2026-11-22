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

    def parse_user_arguments(self, user_arguments: Optional[str]) -> Dict[str, Any]:
        """
        Parses natural language client arguments/challenges in ADVISE mode.
        Extracts ticker exclusions, theme preferences, leverage overrides, and cash shifts.
        """
        if not user_arguments:
            return {
                "excluded_tickers": set(),
                "boosted_tickers": set(),
                "extra_cash_buffer": 0.0,
                "avoid_leveraged": False,
                "notes": []
            }

        args_lower = user_arguments.lower()
        excluded_tickers = set()
        boosted_tickers = set()
        extra_cash_buffer = 0.0
        avoid_leveraged = False
        notes = []

        # 1. Crypto exclusions
        if any(w in args_lower for w in ["crypto", "bitcoin", "digital asset"]):
            crypto_tickers = [e.ticker for e in ETF_UNIVERSE if "crypto" in e.theme.lower() or "digital asset" in e.theme.lower() or "bitcoin" in e.ticker.lower()]
            excluded_tickers.update(crypto_tickers)
            notes.append("Client Mandate: Excluded all Cryptocurrency / Digital Asset ETFs.")

        # 2. Tech exclusions
        if any(w in args_lower for w in ["tech", "semiconductor"]):
            tech_tickers = [e.ticker for e in ETF_UNIVERSE if "tech" in e.theme.lower() or "semiconductor" in e.theme.lower()]
            excluded_tickers.update(tech_tickers)
            notes.append("Client Mandate: Excluded Technology and Semiconductor ETFs.")

        # 3. Leverage exclusions
        if any(w in args_lower for w in ["leverage", "leveraged", "inverse"]):
            avoid_leveraged = True
            notes.append("Client Mandate: Excluded all Leveraged and Inverse ETFs.")

        # 4. Specific ticker exclusions
        for etf in ETF_UNIVERSE:
            t_lower = etf.ticker.lower()
            if f"no {t_lower}" in args_lower or f"exclude {t_lower}" in args_lower or f"drop {t_lower}" in args_lower or f"sell {t_lower}" in args_lower:
                excluded_tickers.add(etf.ticker)
                notes.append(f"Client Mandate: Excluded specific asset {etf.ticker}.")

        # 5. Asset / Theme boosts
        if any(w in args_lower for w in ["gold", "precious metals"]):
            boosted_tickers.add("GOLD")
            notes.append("Client Preference: Tactically boosted Physical Gold allocation.")
        if any(w in args_lower for w in ["defense", "aerospace"]):
            boosted_tickers.add("DEFENSE")
            notes.append("Client Preference: Tactically boosted Aerospace & Defense allocation.")
        if any(w in args_lower for w in ["energy", "oil"]):
            boosted_tickers.add("WORLD ENERGY")
            notes.append("Client Preference: Tactically boosted World Energy allocation.")
        if any(w in args_lower for w in ["cash", "money market", "conservative"]):
            extra_cash_buffer = 0.08
            notes.append("Client Preference: Increased cash reserve buffer by +8.0% for capital preservation.")

        return {
            "excluded_tickers": excluded_tickers,
            "boosted_tickers": boosted_tickers,
            "extra_cash_buffer": extra_cash_buffer,
            "avoid_leveraged": avoid_leveraged,
            "notes": notes
        }

    def compute_conviction_scores(
        self,
        tech_signals: Dict[str, dict],
        macro_briefing: MacroBriefing,
        boosted_tickers: Optional[set] = None
    ) -> Dict[str, float]:
        """
        Blends Technical Score (60%) with Theme Macro Tilt (40%).
        Applies discretionary argument boost (+0.30) if asset was favored by client.
        """
        boosted_tickers = boosted_tickers or set()
        convictions = {}

        # Canonical mapping resolving ETF_UNIVERSE themes (20 themes) to Macro Tilts (7 clusters)
        THEME_NORM_MAP = {
            "equity / tech": "Technology",
            "frontier tech": "Technology",
            "us equity": "Technology",
            "global equity (benchmark)": "European Equities",
            "european equity": "European Equities",
            "home market": "European Equities",
            "financials": "European Equities",
            "emerging markets": "Emerging Markets",
            "energy & commodities": "Energy & Commodities",
            "energy transition": "Energy & Commodities",
            "precious metals": "Precious Metals",
            "industrial metals": "Energy & Commodities",
            "defensive sector": "Defensive Sectors",
            "health care": "Defensive Sectors",
            "fixed income / bonds": "Defensive Sectors",
            "money market (cash)": "Defensive Sectors",
            "size & style factors": "Defensive Sectors",
            "income strategies": "Technology",
            "digital assets": "Digital Assets",
            "leveraged & inverse": "Technology"
        }

        macro_tilts_normalized = {k.lower(): v for k, v in macro_briefing.theme_macro_tilts.items()}

        for ticker, sig_dict in tech_signals.items():
            meta = self.meta_map.get(ticker)
            if not meta:
                continue

            tech_score = sig_dict.get("composite_score", 0.0) if isinstance(sig_dict, dict) else getattr(sig_dict, "composite_score", 0.0)
            theme = meta.theme
            theme_clean = theme.strip().lower()

            # Resolve macro tilt using case-insensitive direct match or canonical cluster mapping
            if theme_clean in macro_tilts_normalized:
                macro_tilt = macro_tilts_normalized[theme_clean]
            else:
                target_cluster = THEME_NORM_MAP.get(theme_clean)
                if target_cluster and target_cluster.lower() in macro_tilts_normalized:
                    macro_tilt = macro_tilts_normalized[target_cluster.lower()]
                else:
                    macro_tilt = macro_briefing.theme_macro_tilts.get(theme, 0.0)

            # Conviction combination
            blended = (0.60 * tech_score) + (0.40 * macro_tilt)
            if ticker in boosted_tickers:
                blended += 0.30

            convictions[ticker] = round(float(np.clip(blended, -1.0, 1.0)), 4)

        return convictions

    def construct_portfolio(
        self,
        trading_date: str,
        tech_signals: Dict[str, dict],
        macro_briefing: MacroBriefing,
        starting_capital_eur: float = 10000.0,
        risk_profile: str = "MODERATE",
        user_arguments: Optional[str] = None,
        current_holdings: Optional[Dict[str, Any]] = None
    ) -> Tuple[PortfolioProposal, RiskEvaluation]:
        """
        Main decision engine for the Portfolio Manager Agent.
        Adapts dynamically to MiFID II Risk Profiles (Conservative, Moderate, Aggressive),
        incorporates client arguments/feedback, verifies with Risk Manager, and generates boardroom memo.
        """
        clean_profile = risk_profile.upper() if risk_profile.upper() in ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"] else "MODERATE"
        self.risk_manager.set_risk_profile(clean_profile)

        # 1. Parse client argument challenges
        arg_insights = self.parse_user_arguments(user_arguments)
        excluded_tickers = arg_insights["excluded_tickers"]
        boosted_tickers = arg_insights["boosted_tickers"]
        extra_cash = arg_insights["extra_cash_buffer"]
        avoid_leveraged_arg = arg_insights["avoid_leveraged"]

        convictions = self.compute_conviction_scores(tech_signals, macro_briefing, boosted_tickers)
        regime = macro_briefing.macro_regime

        # 2. Determine Dynamic Target Cash Buffer based on Risk Profile & Macro Regime
        if clean_profile == "CONSERVATIVE":
            base_cash = 0.20  # Minimum 20% cash buffer for conservative profile
            if regime in ["RISK_OFF", "STAGFLATION"]:
                base_cash = 0.25
            allow_leveraged = False
            max_single_cap = 0.12
        elif clean_profile == "AGGRESSIVE":
            base_cash = 0.05  # Legal minimum 5% cash buffer
            if regime == "RISK_OFF":
                base_cash = 0.08
            allow_leveraged = not avoid_leveraged_arg and (regime != "RISK_OFF")
            max_single_cap = 0.20
        else:  # MODERATE
            base_cash = 0.10  # 10% cash buffer
            if regime == "RISK_ON":
                base_cash = 0.08
            elif regime == "RISK_OFF":
                base_cash = 0.12
            allow_leveraged = not avoid_leveraged_arg and (regime == "RISK_ON")
            max_single_cap = 0.18

        target_cash_pct = min(0.50, base_cash + extra_cash)
        deployable_capital_pct = 1.0 - target_cash_pct

        # 3. Filter Candidate Assets (Sort by Conviction Descending)
        sorted_candidates = sorted(
            convictions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        selected_tickers = []
        selected_themes = set()

        # Conservative profile: explicitly prefer defensive themes
        defensive_themes = {"Defensive / Low Vol", "Cash / Money Market", "Bonds / Sovereign", "Healthcare", "Consumer Staples"}

        for ticker, score in sorted_candidates:
            if ticker in excluded_tickers:
                continue

            meta = self.meta_map.get(ticker)
            if not meta:
                continue

            # Volatility Drag Guard: Skip leveraged if not permitted
            if meta.is_leveraged_or_short and not allow_leveraged:
                continue

            # In Conservative profile, filter out highly speculative/crypto assets
            if clean_profile == "CONSERVATIVE":
                theme_low = meta.theme.lower()
                if "crypto" in theme_low or "digital asset" in theme_low or "bitcoin" in meta.ticker.lower() or meta.is_leveraged_or_short:
                    continue

            theme = meta.theme
            if ticker not in selected_tickers:
                selected_tickers.append(ticker)
                selected_themes.add(theme)

            if len(selected_tickers) == 4:
                break

        # 4. Tactical Safe Haven / Dynamic Defensive Basket as 5th asset
        # Rather than rigidly forcing a single asset (e.g. GOLD which can plunge during rate hikes/dollar surges),
        # evaluate an institutional defensive candidate basket and dynamically select the highest-conviction asset
        # that shows non-negative momentum/conviction (>= 0.0).
        # If all defensive assets are trending downward, preserve capital by allocating to OVERNIGHT CASH.
        defensive_pool = [
            "EUR GOVT BOND", "GLOBAL AGG BOND", "US TREASURY 20Y",
            "WORLD MIN VOL", "STAPLES", "HEALTH CARE", "GOLD", "OVERNIGHT CASH"
        ]

        if regime in ["RISK_OFF", "STAGFLATION"] or clean_profile == "CONSERVATIVE":
            available_hedges = [
                t for t in defensive_pool
                if t not in selected_tickers and t not in excluded_tickers and t in self.meta_map
            ]
            viable_hedges = [t for t in available_hedges if convictions.get(t, -1.0) >= 0.0]
            if viable_hedges:
                viable_hedges.sort(key=lambda t: convictions.get(t, 0.0), reverse=True)
                selected_tickers.append(viable_hedges[0])
            else:
                if "OVERNIGHT CASH" not in selected_tickers and "OVERNIGHT CASH" not in excluded_tickers:
                    selected_tickers.append("OVERNIGHT CASH")
        else:
            # In RISK_ON / NEUTRAL regime:
            # If fewer than 5 assets selected, pick the next highest conviction candidate
            for ticker, score in sorted_candidates:
                if len(selected_tickers) >= 5:
                    break
                if ticker not in selected_tickers and ticker not in excluded_tickers:
                    meta = self.meta_map.get(ticker)
                    if meta:
                        if meta.is_leveraged_or_short and not allow_leveraged:
                            continue
                        selected_tickers.append(ticker)
            
            # If still fewer than 5 assets, fall back to best defensive candidate or cash
            if len(selected_tickers) < 5:
                for t in defensive_pool:
                    if t not in selected_tickers and t not in excluded_tickers:
                        selected_tickers.append(t)
                        break

        # Truncate to maximum 5 assets
        selected_tickers = selected_tickers[:5]

        # 5. Conviction-Weighted Dynamic Sizing (Alpha-Proportional Allocation)
        # Dynamically sizes positions based on model conviction, allocating up to max_single_cap
        # to top ideas while preserving risk-seeking upside and avoiding arbitrary volatility penalties.
        if selected_tickers:
            min_c = min(convictions.get(t, 0.05) for t in selected_tickers)
            shift = max(0.0, -min_c + 0.05)
            pos_scores = {t: max(0.05, convictions.get(t, 0.0) + shift) for t in selected_tickers}
            total_score = sum(pos_scores.values())

            raw_weights = {}
            for t in selected_tickers:
                ideal_w = deployable_capital_pct * (pos_scores[t] / total_score)
                raw_weights[t] = round(min(max_single_cap, max(0.05, ideal_w)), 4)

            # Rescale if sum exceeds deployable capital
            total_allocated = sum(raw_weights.values())
            if total_allocated > deployable_capital_pct:
                scale = deployable_capital_pct / total_allocated
                raw_weights = {t: round(raw_weights[t] * scale, 4) for t in selected_tickers}
        else:
            raw_weights = {}

        # 6. Closed-Loop Validation with Risk Manager
        risk_eval = self.risk_manager.evaluate_portfolio(raw_weights)
        if not risk_eval.is_approved:
            final_weights = risk_eval.adjusted_weights
            risk_eval = self.risk_manager.evaluate_portfolio(final_weights)
        else:
            final_weights = raw_weights

        # 7. Formulate Exact Trading Order Items
        proposed_orders: List[OrderItem] = []
        for ticker in selected_tickers:
            weight = final_weights.get(ticker, 0.0)
            if weight <= 0.001:
                continue
            amount_eur = round(starting_capital_eur * weight, 2)
            tech_sig = tech_signals.get(ticker, {})
            rationale = (
                tech_sig.get("summary_reason", "Momentum and macro regime alignment.")
                if isinstance(tech_sig, dict)
                else getattr(tech_sig, "summary_reason", "Momentum and macro regime alignment.")
            )

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

        # 8. Synthesize Boardroom-Level Investment Memo with Client Alignment
        client_notes_str = ""
        if arg_insights["notes"]:
            client_notes_str = "\nClient Argument Integration: " + " | ".join(arg_insights["notes"])

        memo = (
            f"Investment Committee Proposal for {trading_date} [Risk Profile: {clean_profile}]:\n"
            f"Macro State: {macro_briefing.macro_regime} | Transatlantic Spread: {macro_briefing.transatlantic_policy_spread:+.2f} | "
            f"Headline Sentiment: {macro_briefing.headline_stream.headline_sentiment_score:+.2f} ({macro_briefing.headline_stream.news_regime}, Latency: {macro_briefing.headline_stream.news_latency_minutes}m).\n"
            f"Deployment Strategy: Deploy €{starting_capital_eur * sum(final_weights.values()):,.2f} into {len(proposed_orders)} top-conviction assets "
            f"({', '.join(o.ticker for o in proposed_orders)}). Retain {actual_cash_reserve:.1f}% cash reserve (minimum required: {self.risk_manager.min_cash_buffer*100:.1f}%). "
            f"Portfolio Risk Profile: Annualized Volatility={risk_eval.portfolio_volatility_annualized:.1f}%, "
            f"1-Day 95% Historical VaR={risk_eval.var_95_daily_pct:.2f}%, 1-Day 95% CVaR={risk_eval.cvar_95_daily_pct:.2f}%. "
            f"Leveraged exposure strictly maintained at {risk_eval.leveraged_exposure_pct*100:.1f}%."
            f"{client_notes_str}"
        )

        # Collect top runner-up candidates (ranks 6-10) for Challenge [C] and Modify [M] modes
        alternative_candidates = []
        for ticker, score in sorted_candidates:
            if ticker in selected_tickers or ticker in excluded_tickers:
                continue
            meta = self.meta_map.get(ticker)
            if not meta:
                continue
            if meta.is_leveraged_or_short and not allow_leveraged:
                continue
            if clean_profile == "CONSERVATIVE" and (meta.is_leveraged_or_short or "crypto" in meta.theme.lower()):
                continue

            sig = tech_signals.get(ticker, {})
            if isinstance(sig, dict):
                rf_prob = sig.get("rf_probability", sig.get("ml_win_probability", 0.50))
            else:
                rf_prob = getattr(sig, "rf_probability", getattr(sig, "ml_win_probability", 0.50))
            theme_tilt = macro_briefing.theme_macro_tilts.get(meta.theme, 0.0)
            top_score = convictions.get(selected_tickers[0], score)
            conviction_diff = round(abs(top_score - score) * 100, 1)

            evidence = (
                f"Runner-up alternative (Rank {len(alternative_candidates) + 6}): "
                f"RF probability={rf_prob*100:.1f}%, Macro tilt={theme_tilt:+.2f}, Conviction delta=-{conviction_diff}%. "
                f"Viable sub-optimal candidate if client challenges top-5 concentration."
            )

            alternative_candidates.append({
                "ticker": ticker,
                "full_name": meta.full_name,
                "theme": meta.theme,
                "conviction_score": round(score, 3),
                "rf_win_prob": round(rf_prob, 3),
                "theme_macro_tilt": round(theme_tilt, 2),
                "momentum_label": sig.get("trend", sig.get("primary_trend", "NEUTRAL")) if isinstance(sig, dict) else getattr(sig, "trend", "NEUTRAL"),
                "trade_off_evidence": evidence
            })
            if len(alternative_candidates) >= 5:
                break

        proposal = PortfolioProposal(
            investment_theme=f"{regime} Tactical Allocation ({clean_profile})",
            macro_regime=regime,
            orders=proposed_orders,
            cash_weight_pct=actual_cash_reserve,
            executive_memo=memo,
            alternative_candidates=alternative_candidates
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
