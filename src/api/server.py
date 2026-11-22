# src/api/server.py
"""
Belfius ReBel x QuantumBlack Agentic Trader — REST API & Web Server
Institutional Asset Management Cockpit for 50 UCITS ETF Portfolio.
Strictly zero-lookahead: Day T executes at Day T-1 Close prices.
Supports Mode Mandate vs Mode Advise, Simulation Time-Stepping, and ETF Universe Statistics.
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Automatically load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline.etf_universe import ETF_UNIVERSE, ETFMeta
from src.pipeline.portfolio_ledger import PortfolioLedger, LedgerSnapshot
from src.pipeline.headline_audit import get_headline_audit_trail
from src.pipeline.simulation_engine import SimulationEngine
from src.pipeline.etf_stats import calculate_etf_universe_stats, METRIC_TOOLTIPS
from src.agents.advisory_copilot import generate_etf_advise_thesis
from src.agents.technical import evaluate_all_etfs, evaluate_etf_technical, retrain_if_new_data
from src.agents.macro import evaluate_macro_agent, fetch_market_macro_indicators
from src.agents.risk import RiskManager, RISK_PROFILES, RiskEvaluation
from src.agents.portfolio import PortfolioManagerAgent
from src.orchestration.state import OrderItem, PortfolioProposal

app = FastAPI(title="Belfius ReBel Agentic Trader API", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Singletons
PRICE_PATH = "data/etf_prices_eur.parquet"
_prices_df: Optional[pd.DataFrame] = None

# Institutional Simulation Engine Singleton
sim_engine = SimulationEngine(price_path=PRICE_PATH)
_ledger = sim_engine.ledger

_last_proposal: Optional[PortfolioProposal] = None
_last_risk_eval: Optional[RiskEvaluation] = None
_last_signals: Optional[Dict[str, Any]] = None
_last_macro: Optional[Any] = None


def get_market_data() -> pd.DataFrame:
    global _prices_df
    if _prices_df is None or _prices_df.empty:
        if os.path.exists(PRICE_PATH):
            df = pd.read_parquet(PRICE_PATH)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            _prices_df = df.sort_index()
        else:
            _prices_df = pd.DataFrame()
    return _prices_df


# Request Models
class RebalanceRequest(BaseModel):
    trading_date: Optional[str] = None
    risk_profile: str = "MODERATE"
    user_arguments: Optional[str] = None
    mode: str = "MANDATE"  # "MANDATE" or "ADVISE"
    custom_orders: Optional[List[Dict[str, Any]]] = None


class InitSimulationRequest(BaseModel):
    start_date: str = "2026-10-02"
    mode: str = "SIMULATION"
    risk_profile: str = "MODERATE"


class JumpDateRequest(BaseModel):
    target_date: str


class ModeChangeRequest(BaseModel):
    mode: Optional[str] = None
    interaction_mode: Optional[str] = None


class SwapRequest(BaseModel):
    remove_ticker: str
    add_ticker: str
    current_orders: List[Dict[str, Any]]
    trading_date: Optional[str] = None
    risk_profile: str = "MODERATE"


class ExecuteRequest(BaseModel):
    trading_date: str
    orders: List[Dict[str, Any]]
    risk_profile: str = "MODERATE"
    auto_step: bool = False


def build_natural_macro_narrative(
    regime: str,
    fed_hd: float,
    ecb_hd: float,
    spread: float,
    vix: float,
    sentiment: float,
    total_headlines: int,
    cash_reserve: float
) -> str:
    """Synthesizes macroeconomic indices into clear, natural, boardroom-level financial narrative."""
    if regime == "RISK_ON":
        regime_desc = "Global financial markets are exhibiting constructive risk appetite, supported by healthy equity momentum and manageable financial conditions."
    elif regime == "RISK_OFF":
        regime_desc = "Global financial markets are operating under defensive pressure, marked by heightened risk aversion and cross-asset volatility."
    elif regime == "STAGFLATION":
        regime_desc = "The macroeconomic landscape reflects stagflationary friction, with cooling economic output colliding with stubborn underlying price pressures."
    else:
        regime_desc = "Global markets remain in a watchful consolidation phase as institutional investors balance growth resilience against monetary policy cross-currents."

    if fed_hd > 0.20:
        fed_desc = "The US Federal Reserve maintains a restrictive monetary stance to keep downward pressure on inflation"
    elif fed_hd < -0.20:
        fed_desc = "The US Federal Reserve is pivoting toward an accommodative policy stance with planned rate reductions"
    else:
        fed_desc = "The US Federal Reserve remains in a patient, data-dependent holding pattern"

    if ecb_hd > 0.20:
        ecb_desc = "the European Central Bank continues to prioritize disinflation with elevated policy rates"
    elif ecb_hd < -0.20:
        ecb_desc = "the European Central Bank has initiated monetary easing in response to softening Eurozone industrial output and subdued credit demand"
    else:
        ecb_desc = "the European Central Bank maintains a neutral stance, evaluating incoming inflation prints meeting-by-meeting"

    if spread > 0.15:
        spread_desc = "This monetary divergence provides structural support to the US Dollar while placing relative pressure on European sovereign yields."
    elif spread < -0.15:
        spread_desc = "This divergence makes European equities increasingly attractive on a relative borrowing cost basis."
    else:
        spread_desc = "Monetary policy trajectories across the Atlantic remain largely aligned with steady currency exchange dynamics."

    if vix > 22.0:
        vol_desc = f"Equity market implied volatility is elevated (VIX at {vix:.1f}), triggering automated risk shields and strict avoidance of daily leveraged decay."
    elif vix < 16.0:
        vol_desc = f"Market volatility is well-contained (VIX at {vix:.1f}), creating an orderly environment for strategic thematic capital deployment."
    else:
        vol_desc = f"Volatility is trading within a normal historical corridor (VIX at {vix:.1f}), recommending balanced risk factor allocation."

    if sentiment > 0.10:
        news_desc = f"Analysis across {total_headlines} real-time global news items confirms positive sentiment, anchored by resilient corporate earnings and technological investment."
    elif sentiment < -0.10:
        news_desc = f"Our semantic review across {total_headlines} audited financial headlines indicates prevailing caution regarding tariff discussions, geopolitical developments, and margin compression."
    else:
        news_desc = f"Sentiment across {total_headlines} audited financial headlines remains neutral, balancing solid technological fundamentals with macro uncertainties."

    conclusion = f"Strategic Posture: ReBel maintains a prudent cash allocation of {cash_reserve:.1f}% to preserve downside liquidity while concentrating deployable capital into high-conviction momentum themes."

    return f"{regime_desc} {fed_desc}, while {ecb_desc}. {spread_desc} {vol_desc} {news_desc}\n\n{conclusion}"


# =====================================================================
# SIMULATION ENGINE CONTROLLER ROUTES
# =====================================================================

@app.get("/api/simulation/state")
def get_simulation_state():
    """Returns full current session telemetry, trading dates, and portfolio status."""
    telemetry = sim_engine.get_telemetry()
    all_dates = sim_engine.get_available_trading_dates()
    dates_rev = list(reversed(all_dates))
    telemetry["available_dates"] = dates_rev
    telemetry["latest_date"] = all_dates[-1] if all_dates else "2026-11-20"
    telemetry["risk_profiles"] = RISK_PROFILES
    telemetry["universe_count"] = len(ETF_UNIVERSE)
    return telemetry


@app.post("/api/simulation/init")
def init_simulation(req: InitSimulationRequest):
    """
    Initializes/resets simulation session at start_date with €10,000 cash.
    All market telemetry is anchored to start_date - 1 Close.
    """
    telemetry = sim_engine.init_session(
        start_date=req.start_date,
        mode=req.mode,
        risk_profile=req.risk_profile
    )
    all_dates = sim_engine.get_available_trading_dates()
    telemetry["available_dates"] = list(reversed(all_dates))
    telemetry["latest_date"] = all_dates[-1] if all_dates else "2026-11-20"
    telemetry["risk_profiles"] = RISK_PROFILES
    telemetry["universe_count"] = len(ETF_UNIVERSE)
    return telemetry


@app.post("/api/simulation/step")
def step_next_trading_day():
    """Advances simulation session by one trading day: Day T -> Day T+1."""
    telemetry = sim_engine.step_next_trading_day()
    all_dates = sim_engine.get_available_trading_dates()
    telemetry["available_dates"] = list(reversed(all_dates))
    telemetry["latest_date"] = all_dates[-1] if all_dates else "2026-11-20"
    telemetry["risk_profiles"] = RISK_PROFILES
    telemetry["universe_count"] = len(ETF_UNIVERSE)
    return telemetry


@app.post("/api/simulation/jump")
def jump_to_date(req: JumpDateRequest):
    """Fast-forwards to target_date in the future."""
    try:
        telemetry = sim_engine.jump_to_date(req.target_date)
        all_dates = sim_engine.get_available_trading_dates()
        telemetry["available_dates"] = list(reversed(all_dates))
        telemetry["latest_date"] = all_dates[-1] if all_dates else "2026-11-20"
        telemetry["risk_profiles"] = RISK_PROFILES
        telemetry["universe_count"] = len(ETF_UNIVERSE)
        return telemetry
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/simulation/mode")
def set_simulation_mode(req: ModeChangeRequest):
    """Updates simulation mode (SIMULATION vs PRACTICE_LIVE) or interaction mode (MANDATE vs ADVISE)."""
    if req.mode:
        sim_engine.mode = req.mode.upper()
    if req.interaction_mode:
        sim_engine.set_interaction_mode(req.interaction_mode)
    sim_engine._save_state()
    return sim_engine.get_telemetry()


@app.get("/api/etf/universe-stats")
def get_etf_universe_statistics(as_of_date: Optional[str] = None):
    """
    Computes institutional statistics for all 50 ETFs as of as_of_date Close (default: T-1 Close).
    Official update timestamp: 0:00 CET.
    """
    df = get_market_data()
    target_as_of = as_of_date or sim_engine.as_of_market_date
    stats_data = calculate_etf_universe_stats(df, as_of_date=target_as_of)
    return stats_data


@app.get("/api/etf/advise-thesis")
def get_etf_advise_thesis_endpoint(
    ticker: str = Query(..., description="ETF Ticker"),
    trading_date: Optional[str] = None,
    action: Optional[str] = Query("BUY", description="Order action: BUY or SELL")
):
    """
    Generates a concise 2-3 sentence AI Mini-Thesis in natural language for Mode Advise.
    Technical ML has 85% weight, Macro News carries 15% weight.
    """
    df = get_market_data()
    t_req = trading_date or sim_engine.current_trading_date
    t_date, as_of_date = sim_engine.resolve_dates(t_req)

    meta_map = {e.ticker: e for e in ETF_UNIVERSE}
    if ticker not in meta_map:
        raise HTTPException(status_code=404, detail=f"ETF ticker '{ticker}' not found in universe.")

    date_slice = df.loc[:as_of_date]
    if date_slice.empty or ticker not in date_slice.columns:
        raise HTTPException(status_code=400, detail=f"Price series unavailable for '{ticker}' as of {as_of_date}.")

    ticker_series = date_slice[ticker].dropna()
    bmk_series = date_slice["WORLD"].dropna() if "WORLD" in date_slice.columns else ticker_series
    close_price = float(ticker_series.iloc[-1]) if not ticker_series.empty else 100.0

    tech_sig = evaluate_etf_technical(ticker_series, ticker, bmk_series)
    macro_briefing = evaluate_macro_agent(trading_date=t_date, reference_date=as_of_date)

    thesis = generate_etf_advise_thesis(
        ticker=ticker,
        tech_sig=tech_sig,
        macro_briefing=macro_briefing,
        close_price=close_price,
        as_of_date=as_of_date,
        action=action or "BUY"
    )
    return thesis


# =====================================================================
# CORE PORTFOLIO & REBALANCING ROUTES
# =====================================================================

@app.get("/api/state")
def get_state():
    """Backward-compatible telemetry route."""
    return get_simulation_state()


@app.post("/api/rebalance")
def run_rebalance(req: RebalanceRequest):
    """
    Runs multi-agent portfolio rebalancing analysis.
    Supports Mode Mandate (autonomous top 5) and Mode Advise (custom 5 user slots).
    Strict Anti-Lookahead: all market data, tech signals, and stats are anchored to Day T-1 Close.
    """
    global _last_proposal, _last_risk_eval, _last_signals, _last_macro
    try:
        df = get_market_data()
        if df.empty:
            raise HTTPException(status_code=500, detail="Market price dataset unavailable at data/etf_prices_eur.parquet.")

        # Resolve T and T-1 Close Date
        t_req = req.trading_date or sim_engine.current_trading_date
        t_date, as_of_date = sim_engine.resolve_dates(t_req)
        sim_engine.current_trading_date = t_date
        sim_engine.as_of_market_date = as_of_date
        sim_engine.risk_profile = req.risk_profile
        sim_engine.set_interaction_mode(req.mode)

        # 1. Mark to Market strictly on Day T-1 Close prices
        date_slice = df.loc[:as_of_date]
        if not date_slice.empty:
            current_prices = {
                col: float(date_slice[col].dropna().iloc[-1])
                for col in date_slice.columns
                if not date_slice[col].dropna().empty
            }
        else:
            current_prices = {col: float(df[col].dropna().iloc[-1]) for col in df.columns}

        current_nav = _ledger.mark_to_market(current_prices) if current_prices else 10000.0
        meta_map = {e.ticker: e for e in ETF_UNIVERSE}

        # 2. Technical Agent (evaluated on historical data up to as_of_date)
        tech_signals = evaluate_all_etfs(date_slice)
        _last_signals = tech_signals

        # 3. Macro Agent (Strict Anti-Lookahead: reference_date = as_of_date)
        macro_briefing = evaluate_macro_agent(trading_date=t_date, reference_date=as_of_date)
        _last_macro = macro_briefing

        # 4. Portfolio & Risk Engine
        risk_engine = RiskManager(risk_profile=req.risk_profile)
        pm_agent = PortfolioManagerAgent(risk_manager=risk_engine)

        # Mode Branching: MANDATE vs ADVISE
        if req.mode.upper() == "ADVISE" and req.custom_orders:
            # Custom Orders provided by investor in Advise Mode
            valid_orders_in = [o for o in req.custom_orders if o.get("ticker") in meta_map]
            total_custom_pct = sum(float(o.get("target_weight_pct", 0.0)) for o in valid_orders_in)

            # Cap individual weight at max profile allowance (e.g. 18% for Moderate, 20% for Aggressive)
            max_single_cap = 0.20 if req.risk_profile.upper() == "AGGRESSIVE" else (0.12 if req.risk_profile.upper() == "CONSERVATIVE" else 0.18)

            custom_weights_dict = {}
            for o in valid_orders_in:
                tkr = o["ticker"]
                raw_w = min(max_single_cap, max(0.02, float(o.get("target_weight_pct", 10.0)) / 100.0))
                custom_weights_dict[tkr] = round(raw_w, 4)

            # Normalize if sum exceeds deployable capital
            min_cash = 0.08 if req.risk_profile.upper() == "AGGRESSIVE" else (0.20 if req.risk_profile.upper() == "CONSERVATIVE" else 0.12)
            max_investable = 1.0 - min_cash
            if sum(custom_weights_dict.values()) > max_investable:
                scale = max_investable / sum(custom_weights_dict.values())
                custom_weights_dict = {t: round(w * scale, 4) for t, w in custom_weights_dict.items()}

            risk_eval = risk_engine.evaluate_portfolio(custom_weights_dict)
            final_weights = risk_eval.adjusted_weights if not risk_eval.is_approved else custom_weights_dict

            order_items = []
            for tkr, w in final_weights.items():
                amt = round(current_nav * w, 2)
                sig = tech_signals.get(tkr)
                p_close = current_prices.get(tkr, 100.0)
                thesis_data = generate_etf_advise_thesis(tkr, sig, macro_briefing, p_close, as_of_date)
                order_items.append(OrderItem(
                    ticker=tkr,
                    action="BUY",
                    target_weight_pct=round(w * 100, 1),
                    estimated_amount_eur=amt,
                    rationale=thesis_data["executive_mini_thesis"]
                ))

            actual_cash = round((1.0 - sum(final_weights.values())) * 100, 1)
            proposal = PortfolioProposal(
                investment_theme=f"Investor Discretionary Portfolio ({req.risk_profile})",
                macro_regime=macro_briefing.macro_regime,
                orders=order_items,
                cash_weight_pct=actual_cash,
                executive_memo=(
                    f"Investor Advisory Mandate for {t_date} [Risk Profile: {req.risk_profile}]:\n"
                    f"Macro State: {macro_briefing.macro_regime} | Execution anchored to {as_of_date} Close.\n"
                    f"Discretionary allocation across {len(order_items)} user-selected assets with {actual_cash:.1f}% cash reserve."
                ),
                alternative_candidates=[]
            )
        else:
            # Autonomous Mandate Mode: Top 5 Multi-Agent Selection with Conviction Sizing
            proposal, risk_eval = pm_agent.construct_portfolio(
                trading_date=t_date,
                tech_signals={t: s.model_dump() for t, s in tech_signals.items()},
                macro_briefing=macro_briefing,
                starting_capital_eur=current_nav,
                risk_profile=req.risk_profile,
                user_arguments=req.user_arguments,
                current_holdings=_ledger.positions
            )

        _last_proposal = proposal
        _last_risk_eval = risk_eval

        market_ind = fetch_market_macro_indicators()
        vix = float(market_ind.get("VIX", 16.5))
        gold = float(market_ind.get("GOLD", 2600.0))
        brent = float(market_ind.get("BRENT", 75.0))
        gold_oil = round(gold / max(1.0, brent), 1)

        recent_30 = date_slice.tail(30)

        orders_out = []
        for o in proposal.orders:
            m = meta_map.get(o.ticker)
            price = float(date_slice[o.ticker].dropna().iloc[-1]) if not date_slice.empty and o.ticker in date_slice.columns else 100.0
            sig = tech_signals.get(o.ticker)
            sig_dict = sig.model_dump() if hasattr(sig, 'model_dump') else (sig if isinstance(sig, dict) else {})
            conviction = sig_dict.get("composite_score", 0.08)

            # 30-day historical daily closing prices
            price_hist_30d = []
            chg_30d = 0.0
            if o.ticker in recent_30.columns and not recent_30[o.ticker].dropna().empty:
                t_series = recent_30[o.ticker].dropna()
                price_hist_30d = [
                    {"date": str(d.date()), "price": round(float(p), 4)}
                    for d, p in t_series.items()
                ]
                if len(t_series) >= 2:
                    chg_30d = round(((t_series.iloc[-1] - t_series.iloc[0]) / t_series.iloc[0]) * 100, 2)

            orders_out.append({
                "ticker": o.ticker,
                "full_name": m.full_name if m else o.ticker,
                "theme": m.theme if m else "Broad Market",
                "isin": m.isin if m else "N/A",
                "action": o.action,
                "target_weight_pct": round(o.target_weight_pct, 2),
                "target_amount_eur": round(o.estimated_amount_eur, 2),
                "limit_price_eur": round(price, 4),
                "conviction_score": round(conviction, 3),
                "momentum_label": sig_dict.get("trend", sig_dict.get("primary_trend", "BULLISH")),
                "thesis_memo": o.rationale,
                "price_history_30d": price_hist_30d,
                "change_30d_pct": chg_30d
            })

        alternatives_out = proposal.alternative_candidates or []

        # Benchmark WORLD Performance Comparison
        bench_ret_series = []
        bench_nav_scaled = []
        chart_dates = []

        if "WORLD" in date_slice.columns:
            w_series = date_slice["WORLD"].dropna().tail(30)
            if not w_series.empty:
                w0 = w_series.iloc[0]
                bench_ret_series = [round(float((p - w0) / w0 * 100), 2) for p in w_series]
                bench_nav_scaled = [round(10000.0 * (1.0 + r / 100.0), 2) for r in bench_ret_series]
                chart_dates = [str(d.date()) for d in w_series.index]

        # Historical simulated NAV equity curve
        if len(_ledger.snapshots) > 1:
            snap_map = {s.trade_date: s.total_nav for s in _ledger.snapshots}
            port_navs = []
            running_nav = 10000.0
            for d in chart_dates:
                if d in snap_map:
                    running_nav = snap_map[d]
                port_navs.append(round(running_nav, 2))
        else:
            base_val = round(current_nav, 2)
            port_navs = [base_val] * len(chart_dates) if chart_dates else [base_val]

        port_ret_series = [round(((n - 10000.0) / 10000.0) * 100, 2) for n in port_navs]

        portfolio_chart = {
            "dates": chart_dates or [t_date],
            "portfolio_nav": port_navs,
            "portfolio_returns_pct": port_ret_series,
            "benchmark_nav_scaled": bench_nav_scaled,
            "benchmark_returns_pct": bench_ret_series,
            "current_return_pct": round(((current_nav - 10000.0) / 10000.0) * 100, 2),
            "benchmark_return_pct": bench_ret_series[-1] if bench_ret_series else 0.0,
            "alpha_pct": round(((current_nav - 10000.0) / 10000.0) * 100 - (bench_ret_series[-1] if bench_ret_series else 0.0), 2)
        }

        # 5. Audited 5-Day Headlines with Real Sources and Clickable URLs
        audit_df = get_headline_audit_trail(lookback_days=5, as_of_date=as_of_date, limit=None)
        headlines_out = []
        if not audit_df.empty:
            for _, row in audit_df.iterrows():
                raw_themes = row.get("detected_themes", "")
                if isinstance(raw_themes, list):
                    theme_str = ", ".join(raw_themes)
                else:
                    theme_str = str(raw_themes) if pd.notna(raw_themes) else "Global Macro"
                if not theme_str.strip():
                    theme_str = "Global Macro"

                pub_raw = str(row.get("published_at", ""))
                pub_fmt = pub_raw[:16].replace("T", " ") + " CET" if len(pub_raw) >= 16 else pub_raw

                headlines_out.append({
                    "title": str(row.get("title", "")),
                    "date": pub_fmt,
                    "sentiment": str(row.get("sentiment_label", "NEUTRAL")),
                    "score": round(float(row.get("sentiment_score", 0.0)), 2),
                    "source": str(row.get("source", "Reuters")),
                    "article_url": str(row.get("article_url", "https://www.reuters.com/markets")),
                    "category_theme": theme_str,
                    "primary_driver": str(row.get("primary_driver", "MACRO_EVENT")),
                    "economic_rationale": str(row.get("economic_rationale", "Macro transmission assessment."))
                })

        natural_summary = build_natural_macro_narrative(
            regime=macro_briefing.macro_regime,
            fed_hd=macro_briefing.fed_hawkish_dovish_score,
            ecb_hd=macro_briefing.ecb_hawkish_dovish_score,
            spread=macro_briefing.transatlantic_policy_spread,
            vix=vix,
            sentiment=macro_briefing.market_sentiment_score,
            total_headlines=len(headlines_out),
            cash_reserve=proposal.cash_weight_pct
        )

        min_pub = audit_df['published_at'].min()[:10] if not audit_df.empty and 'published_at' in audit_df.columns else as_of_date
        max_pub = audit_df['published_at'].max()[:10] if not audit_df.empty and 'published_at' in audit_df.columns else as_of_date

        return {
            "nav": round(current_nav, 2),
            "trading_date": t_date,
            "as_of_date": as_of_date,
            "mode": sim_engine.mode,
            "interaction_mode": sim_engine.interaction_mode,
            "investment_theme": proposal.investment_theme,
            "macro_regime": proposal.macro_regime,
            "cash_weight_pct": round(proposal.cash_weight_pct, 2),
            "orders": orders_out,
            "alternative_candidates": alternatives_out,
            "portfolio_chart": portfolio_chart,
            "risk_evaluation": {
                "passed": risk_eval.is_approved,
                "status": "PASSED" if risk_eval.is_approved else "WARNING",
                "var_95_daily_pct": round(risk_eval.var_95_daily_pct, 2),
                "cvar_95_daily_pct": round(risk_eval.cvar_95_daily_pct, 2),
                "leveraged_weight_pct": round(risk_eval.leveraged_exposure_pct * 100, 2),
                "cash_buffer_pct": round(risk_eval.cash_reserve_pct, 2),
                "theme_weights": {k: round(v * 100, 2) for k, v in risk_eval.theme_concentrations.items()},
                "critique_memos": risk_eval.violations + risk_eval.warnings,
                "recommendations": [risk_eval.risk_summary]
            },
            "macro_briefing": {
                "regime": macro_briefing.macro_regime,
                "sentiment_headline": macro_briefing.market_sentiment_score,
                "fed_hawkish_dovish_score": round(macro_briefing.fed_hawkish_dovish_score, 2),
                "ecb_hawkish_dovish_score": round(macro_briefing.ecb_hawkish_dovish_score, 2),
                "transatlantic_policy_spread": round(macro_briefing.transatlantic_policy_spread, 2),
                "vix_level": vix,
                "gold_oil_ratio": gold_oil,
                "theme_macro_tilts": {k: round(v, 2) for k, v in macro_briefing.theme_macro_tilts.items()},
                "tail_risk_alerts": macro_briefing.tail_risk_alerts,
                "executive_summary": natural_summary,
                "central_bank_sectors": {
                    "Interest Rates & Inflation": "Fed & ECB maintain restrictive bias; core services disinflation remains gradual.",
                    "Corporate Growth & Credit": "Eurozone corporate credit conditions remain stable with moderate bank lending spreads.",
                    "Energy & Green Transition": "Comfortable European natural gas inventories provide structural mitigation against winter shocks.",
                    "Trade, FX & Global Spreads": "Transatlantic policy spread anchors EUR/USD within competitive industrial corridor."
                }
            },
            "headlines": headlines_out,
            "headline_window": {
                "as_of_date": as_of_date,
                "date_range": f"{min_pub} to {max_pub}",
                "total_headlines": len(headlines_out),
                "sync_frequency": "HOURLY (0:00 - 23:00 CET)" if sim_engine.mode == "PRACTICE_LIVE" else "DAILY (0:00 CET)"
            },
            "notice_banner": (
                f"Phiên giao dịch ngày {t_date} | "
                f"Giá thực thi và số liệu phân tích được cố định theo giá đóng cửa ngày {as_of_date} (Close T-1)."
            )
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Rebalance execution failed: {str(e)}")


@app.post("/api/swap")
def swap_candidate(req: SwapRequest):
    """
    [C] Challenge Function: Swaps a runner-up candidate (Rank 6-10) in place of an existing order.
    Re-runs pre-trade risk engine audit instantly.
    """
    df = get_market_data()
    meta_map = {e.ticker: e for e in ETF_UNIVERSE}
    add_meta = meta_map.get(req.add_ticker)
    if not add_meta:
        raise HTTPException(status_code=400, detail=f"Ticker {req.add_ticker} not in universe")

    t_req = req.trading_date or sim_engine.current_trading_date
    t_date, as_of_date = sim_engine.resolve_dates(t_req)

    new_orders = []
    target_weight = 15.0
    for o in req.current_orders:
        if o["ticker"] == req.remove_ticker:
            target_weight = o["target_weight_pct"]
            continue
        new_orders.append(o)

    date_slice = df.loc[:as_of_date]
    if not date_slice.empty and add_meta.ticker in date_slice.columns:
        cand_price = float(date_slice[add_meta.ticker].dropna().iloc[-1])
    else:
        cand_price = 100.0

    current_prices = {col: float(date_slice[col].dropna().iloc[-1]) for col in date_slice.columns if not date_slice[col].dropna().empty} if not date_slice.empty else {}
    current_nav = _ledger.mark_to_market(current_prices) if current_prices else 10000.0
    cand_amount = round(current_nav * (target_weight / 100.0), 2)

    new_orders.append({
        "ticker": add_meta.ticker,
        "full_name": add_meta.full_name,
        "theme": add_meta.theme,
        "isin": add_meta.isin,
        "action": "BUY",
        "target_weight_pct": round(target_weight, 2),
        "target_amount_eur": cand_amount,
        "limit_price_eur": round(cand_price, 4),
        "conviction_score": 0.065,
        "momentum_label": "CHALLENGED",
        "thesis_memo": f"Discretionary substitution: Replaced {req.remove_ticker} with Rank candidate {add_meta.ticker} ({add_meta.full_name}) based on investor challenge."
    })

    # Re-evaluate Risk
    risk_engine = RiskManager(risk_profile=req.risk_profile)
    weights_dict = {o["ticker"]: o["target_weight_pct"] / 100.0 for o in new_orders}
    risk_eval = risk_engine.evaluate_portfolio(weights_dict)

    return {
        "orders": new_orders,
        "risk_evaluation": {
            "passed": risk_eval.is_approved,
            "status": "PASSED" if risk_eval.is_approved else "WARNING",
            "var_95_daily_pct": round(risk_eval.var_95_daily_pct, 2),
            "cash_buffer_pct": round(risk_eval.cash_reserve_pct, 2),
            "theme_weights": {k: round(v * 100, 2) for k, v in risk_eval.theme_concentrations.items()},
            "critique_memos": risk_eval.violations + risk_eval.warnings
        }
    }


@app.post("/api/execute")
def execute_orders(req: ExecuteRequest):
    """
    Executes trades at Day T-1 Close price, updates PortfolioLedger, and writes immutable audit trail.
    If auto_step=True, advances to next trading day automatically.
    """
    df = get_market_data()
    t_date, as_of_date = sim_engine.resolve_dates(req.trading_date)

    date_slice = df.loc[:as_of_date]
    if not date_slice.empty:
        current_prices = {
            col: float(date_slice[col].dropna().iloc[-1])
            for col in date_slice.columns
            if not date_slice[col].dropna().empty
        }
    else:
        current_prices = {col: float(df[col].dropna().iloc[-1]) for col in df.columns}

    target_orders = [
        {
            "ticker": o["ticker"],
            "action": str(o.get("action", "BUY")).upper(),
            "amount": float(o.get("target_amount_eur", 0.0)),
            "weight": float(o.get("target_weight_pct", 0.0)) / 100.0
        }
        for o in req.orders[:5]
    ]

    snapshot = _ledger.execute_rebalancing(
        target_orders=target_orders,
        current_prices=current_prices,
        trade_date=t_date,
        var_95=0.0,
        cvar_95=0.0
    )

    message = f"Successfully executed {len(target_orders)} orders for session {t_date} at {as_of_date} Close (Day T-1)."

    # Auto step to next trading day if requested
    next_day_stepped = False
    if req.auto_step:
        sim_engine.step_next_trading_day()
        next_day_stepped = True
        message += f" Advanced to next session: {sim_engine.current_trading_date}."

    telemetry = sim_engine.get_telemetry()

    return {
        "success": True,
        "message": message,
        "stepped_next_day": next_day_stepped,
        "nav": round(snapshot.total_nav, 2),
        "cash_balance": round(snapshot.cash, 2),
        "positions": telemetry["positions"],
        "trade_log": telemetry["trade_log"],
        "current_trading_date": sim_engine.current_trading_date,
        "as_of_market_date": sim_engine.as_of_market_date,
        "cumulative_return_pct": round(snapshot.cumulative_return_pct, 2)
    }


@app.post("/api/reset")
def reset_portfolio():
    """Resets portfolio to clean initial state (€10,000 cash)."""
    telemetry = sim_engine.init_session(
        start_date=sim_engine.start_date,
        mode=sim_engine.mode,
        risk_profile=sim_engine.risk_profile
    )
    return {
        "success": True,
        "nav": 10000.0,
        "cash_balance": 10000.0,
        "positions": {},
        "trade_log": [],
        "cumulative_return_pct": 0.0,
        "message": "Đã khôi phục toàn bộ khối tài sản về thiết lập ban đầu €10,000.00."
    }


@app.get("/api/ledger/history")
def get_ledger_history():
    """Returns complete trade log and historical mark-to-market snapshots."""
    return {
        "trade_log": [
            {
                "trade_date": tr.trade_date,
                "ticker": tr.ticker,
                "action": tr.action,
                "shares": round(tr.shares, 4),
                "price": round(tr.price, 4),
                "total_amount": round(tr.total_amount, 2),
                "fee": tr.fee
            }
            for tr in reversed(_ledger.trade_history)
        ],
        "snapshots": [
            {
                "trade_date": s.trade_date,
                "cash": round(s.cash, 2),
                "invested_value": round(s.invested_value, 2),
                "total_nav": round(s.total_nav, 2),
                "period_return_pct": round(s.period_return_pct, 2),
                "cumulative_return_pct": round(s.cumulative_return_pct, 2),
                "holdings": s.holdings
            }
            for s in _ledger.snapshots
        ]
    }


# Serve the Web Advisory Cockpit
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "..", "web", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Belfius ReBel Neo-Terminal UI Loading...</h1>"
