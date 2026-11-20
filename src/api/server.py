# src/api/server.py
"""
Belfius ReBel x QuantumBlack Agentic Trader — REST API & Web Server
High-density, professional Neo-Terminal interface for institutional ETF portfolio management.
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.pipeline.etf_universe import ETF_UNIVERSE, ETFMeta
from src.pipeline.portfolio_ledger import PortfolioLedger, LedgerSnapshot
from src.pipeline.headline_audit import get_headline_audit_trail
from src.agents.technical import evaluate_all_etfs, retrain_if_new_data
from src.agents.macro import evaluate_macro_agent, fetch_market_macro_indicators
from src.agents.risk import RiskManager, RISK_PROFILES, RiskEvaluation
from src.agents.portfolio import PortfolioManagerAgent
from src.orchestration.state import OrderItem, PortfolioProposal

app = FastAPI(title="Belfius ReBel Agentic Trader API", version="2.0.0")

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
_ledger = PortfolioLedger()
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
            _prices_df = df
        else:
            _prices_df = pd.DataFrame()
    return _prices_df

class RebalanceRequest(BaseModel):
    trading_date: str
    risk_profile: str = "MODERATE"
    user_arguments: Optional[str] = None

class SwapRequest(BaseModel):
    remove_ticker: str
    add_ticker: str
    current_orders: List[Dict[str, Any]]
    trading_date: str
    risk_profile: str = "MODERATE"

class ExecuteRequest(BaseModel):
    trading_date: str
    orders: List[Dict[str, Any]]
    risk_profile: str = "MODERATE"

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
    """
    Synthesizes macroeconomic indices into clear, natural, boardroom-level financial English.
    Avoids raw index dumps and explains transmission channels intuitively.
    """
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
        news_desc = f"Our semantic review across {total_headlines} global headlines indicates prevailing caution regarding tariff discussions, geopolitical developments, and margin compression."
    else:
        news_desc = f"Sentiment across {total_headlines} audited financial headlines remains neutral, balancing solid technological fundamentals with macro uncertainties."

    conclusion = f"Strategic Posture: ReBel maintains a prudent cash allocation of {cash_reserve:.1f}% to preserve downside liquidity while concentrating deployable capital into high-conviction momentum themes."

    return f"{regime_desc} {fed_desc}, while {ecb_desc}. {spread_desc} {vol_desc} {news_desc}\n\n{conclusion}"

@app.get("/api/state")
def get_state():
    df = get_market_data()
    dates = [str(d.date()) for d in df.index] if not df.empty else ["2026-11-20"]
    dates.reverse() # Most recent first

    # Current prices for MTM
    current_prices = {col: float(df[col].dropna().iloc[-1]) for col in df.columns} if not df.empty else {}
    current_nav = _ledger.mark_to_market(current_prices) if current_prices else 10000.0

    positions_dict = {
        tkr: {
            "shares": round(pos.shares, 4),
            "avg_entry_price": round(pos.avg_entry_price, 4),
            "current_price": round(pos.current_price, 4),
            "market_value": round(pos.market_value, 2),
            "unrealized_pnl": round(pos.unrealized_pnl, 2),
            "weight_pct": round(pos.weight_pct, 2)
        }
        for tkr, pos in _ledger.positions.items()
    }

    return {
        "current_nav": round(current_nav, 2),
        "cash_balance": round(_ledger.cash, 2),
        "cumulative_return_pct": round(((current_nav - 10000.0) / 10000.0) * 100, 2),
        "positions": positions_dict,
        "available_dates": dates,
        "latest_date": dates[0] if dates else "2026-11-20",
        "risk_profiles": RISK_PROFILES,
        "universe_count": len(ETF_UNIVERSE)
    }

@app.post("/api/rebalance")
def run_rebalance(req: RebalanceRequest):
    global _last_proposal, _last_risk_eval, _last_signals, _last_macro
    df = get_market_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="Market price dataset unavailable.")

    # Calculate MTM on current date
    date_slice = df.loc[:req.trading_date]
    if not date_slice.empty:
        current_prices = {col: float(date_slice[col].dropna().iloc[-1]) for col in date_slice.columns if not date_slice[col].dropna().empty}
    else:
        current_prices = {col: float(df[col].dropna().iloc[-1]) for col in df.columns}
    
    current_nav = _ledger.mark_to_market(current_prices) if current_prices else 10000.0

    # 1. Technical Agent
    tech_signals = evaluate_all_etfs(df)
    _last_signals = tech_signals

    # 2. Macro Agent
    macro_briefing = evaluate_macro_agent(trading_date=req.trading_date)
    _last_macro = macro_briefing

    # 3. Portfolio & Risk Agents
    risk_engine = RiskManager(risk_profile=req.risk_profile)
    pm_agent = PortfolioManagerAgent(risk_manager=risk_engine)

    proposal, risk_eval = pm_agent.construct_portfolio(
        trading_date=req.trading_date,
        tech_signals={t: s.model_dump() for t, s in tech_signals.items()},
        macro_briefing=macro_briefing,
        starting_capital_eur=current_nav,
        risk_profile=req.risk_profile,
        user_arguments=req.user_arguments,
        current_holdings=_ledger.positions
    )

    _last_proposal = proposal
    _last_risk_eval = risk_eval

    meta_map = {e.ticker: e for e in ETF_UNIVERSE}

    market_ind = fetch_market_macro_indicators()
    vix = float(market_ind.get("VIX", 16.5))
    gold = float(market_ind.get("GOLD", 2600.0))
    brent = float(market_ind.get("BRENT", 75.0))
    gold_oil = round(gold / max(1.0, brent), 1)

    date_slice = df.loc[:req.trading_date]
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
            "momentum_label": sig_dict.get("primary_trend", "BULLISH"),
            "thesis_memo": o.rationale,
            "price_history_30d": price_hist_30d,
            "change_30d_pct": chg_30d
        })

    alternatives_out = proposal.alternative_candidates or []

    # Calculate 30-Day Normalized Portfolio NAV vs Benchmark Trajectory for Charting
    chart_dates = [str(d.date()) for d in recent_30.index]
    bench_col = "WORLD" if "WORLD" in recent_30.columns else ("IWDA.AS" if "IWDA.AS" in recent_30.columns else recent_30.columns[0])
    bench_series = recent_30[bench_col].dropna()
    bench_p0 = bench_series.iloc[0] if not bench_series.empty else 1.0

    portfolio_weights = {o.ticker: o.target_weight_pct / 100.0 for o in proposal.orders}
    cash_weight = max(0.0, 1.0 - sum(portfolio_weights.values()))

    portfolio_history = []
    benchmark_history = []
    base_prices = {
        tkr: (recent_30[tkr].dropna().iloc[0] if tkr in recent_30.columns and not recent_30[tkr].dropna().empty else 1.0)
        for tkr in portfolio_weights
    }

    for dt in recent_30.index:
        port_factor = cash_weight
        for tkr, w in portfolio_weights.items():
            if tkr in recent_30.columns and dt in recent_30[tkr].index:
                pt = recent_30[tkr].loc[dt]
                if not pd.isna(pt) and base_prices[tkr] > 0:
                    port_factor += w * (float(pt) / base_prices[tkr])
                else:
                    port_factor += w
            else:
                port_factor += w
        portfolio_history.append(round(10000.0 * port_factor, 2))

        if dt in bench_series.index and not pd.isna(bench_series.loc[dt]):
            b_pt = float(bench_series.loc[dt])
            benchmark_history.append(round(10000.0 * (b_pt / bench_p0), 2))
        else:
            benchmark_history.append(benchmark_history[-1] if benchmark_history else 10000.0)

    portfolio_chart = {
        "dates": chart_dates,
        "portfolio_nav": portfolio_history,
        "benchmark_nav": benchmark_history,
        "benchmark_name": "MSCI World Benchmark (IWDA)",
        "current_return_pct": round(((portfolio_history[-1] - 10000.0) / 10000.0) * 100, 2) if portfolio_history else 0.0,
        "benchmark_return_pct": round(((benchmark_history[-1] - 10000.0) / 10000.0) * 100, 2) if benchmark_history else 0.0,
        "alpha_pct": round((((portfolio_history[-1] - 10000.0) / 10000.0) - ((benchmark_history[-1] - 10000.0) / 10000.0)) * 100, 2) if portfolio_history and benchmark_history else 0.0
    }

    # Fetch ALL headlines without 50 cap
    audit_df = get_headline_audit_trail(lookback_days=5, limit=None)
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

            headlines_out.append({
                "title": str(row.get("title", "")),
                "date": str(row.get("published_at", ""))[:10],
                "sentiment": str(row.get("sentiment_label", "NEUTRAL")),
                "score": round(float(row.get("sentiment_score", 0.0)), 2),
                "source": str(row.get("source", "Financial Wire")),
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

    return {
        "nav": round(current_nav, 2),
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
            "executive_summary": natural_summary
        },
        "headlines": headlines_out
    }

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

    new_orders = []
    target_weight = 15.0
    for o in req.current_orders:
        if o["ticker"] == req.remove_ticker:
            target_weight = o["target_weight_pct"]
            continue
        new_orders.append(o)

    # Get latest price for the candidate
    date_slice = df.loc[:req.trading_date]
    if not date_slice.empty and add_meta.ticker in date_slice.columns:
        cand_price = float(date_slice[add_meta.ticker].dropna().iloc[-1])
    else:
        cand_price = 100.0

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
    Executes trades at limit price, updates PortfolioLedger, and writes immutable audit trail.
    """
    df = get_market_data()
    date_slice = df.loc[:req.trading_date]
    if not date_slice.empty:
        current_prices = {col: float(date_slice[col].dropna().iloc[-1]) for col in date_slice.columns if not date_slice[col].dropna().empty}
    else:
        current_prices = {col: float(df[col].dropna().iloc[-1]) for col in df.columns}

    # Format orders for execute_rebalancing
    target_orders = [
        {
            "ticker": o["ticker"],
            "amount": float(o.get("target_amount_eur", 0.0)),
            "weight": float(o.get("target_weight_pct", 0.0)) / 100.0
        }
        for o in req.orders
    ]

    snapshot = _ledger.execute_rebalancing(
        target_orders=target_orders,
        current_prices=current_prices,
        trade_date=req.trading_date,
        var_95=0.0,
        cvar_95=0.0
    )

    positions_dict = {
        tkr: {
            "shares": round(pos.shares, 4),
            "avg_entry_price": round(pos.avg_entry_price, 4),
            "current_price": round(pos.current_price, 4),
            "market_value": round(pos.market_value, 2),
            "unrealized_pnl": round(pos.unrealized_pnl, 2),
            "weight_pct": round(pos.weight_pct, 2)
        }
        for tkr, pos in _ledger.positions.items()
    }

    return {
        "success": True,
        "message": f"Successfully executed rebalance of {len(target_orders)} orders on {req.trading_date}.",
        "nav": round(snapshot.total_nav, 2),
        "cash_balance": round(snapshot.cash, 2),
        "positions": positions_dict,
        "cumulative_return_pct": round(snapshot.cumulative_return_pct, 2)
    }

@app.post("/api/reset")
def reset_portfolio():
    _ledger.reset()
    return {
        "success": True,
        "nav": 10000.0,
        "cash_balance": 10000.0,
        "positions": {},
        "cumulative_return_pct": 0.0
    }

# Serve the HTML Neo-Terminal
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "..", "web", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Belfius ReBel Neo-Terminal UI Loading...</h1>"
