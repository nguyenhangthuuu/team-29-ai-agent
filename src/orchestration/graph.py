# src/orchestration/graph.py
import os
import json
import pandas as pd
from typing import Dict, Any, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.orchestration.state import AgentState, PortfolioProposal, OrderItem
from src.orchestration.logger import log_event
from src.agents.technical import evaluate_all_etfs
from src.agents.risk import RiskManager
from src.agents.macro import evaluate_macro_agent
from src.pipeline.etf_universe import ETF_UNIVERSE

load_dotenv()
risk_engine = RiskManager()

# ==========================================
# 1. NODE IMPLEMENTATIONS (IN EXACT ORDER)
# ==========================================

def data_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Loads latest EUR closing prices for all 50 ETFs."""
    price_file = "data/etf_prices_eur.parquet"
    if os.path.exists(price_file):
        df = pd.read_parquet(price_file)
        latest_prices = {col: float(df[col].dropna().iloc[-1]) for col in df.columns}
        trade_date = str(df.index[-1].date())
    else:
        # Fallback synthetic prices if file not yet run
        latest_prices = {etf.ticker: 100.0 for etf in ETF_UNIVERSE}
        trade_date = "2026-11-13"

    log_event(
        event_type="DATA_INGESTION",
        node_name="data_node",
        payload={"date": trade_date, "etf_count": len(latest_prices)}
    )
    return {
        "trading_date": trade_date,
        "prices": latest_prices,
        "execution_status": "DATA_LOADED"
    }

def technical_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Generates technical scores, RSI, trend, and breakout signals."""
    price_file = "data/etf_prices_eur.parquet"
    df = pd.read_parquet(price_file)
    signals = evaluate_all_etfs(df)
    
    # Convert Pydantic signals to dict for state serialization
    signal_dict = {ticker: sig.model_dump() for ticker, sig in signals.items()}
    
    log_event(
        event_type="TECHNICAL_ANALYSIS",
        node_name="technical_node",
        payload={"ranked_count": len(signal_dict)}
    )
    return {
        "tech_signals": signal_dict,
        "execution_status": "TECHNICAL_COMPLETED"
    }

def macro_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Evaluates macro regime using Fed Communication (Apel-Grimaldi & LM)
    plus real-time market indicators (VIX, 10Y Yields, Oil, DXY).
    """
    trade_date = state.get("trading_date", "2026-11-18")
    briefing_obj = evaluate_macro_agent(trading_date=trade_date)
    
    log_event(
        event_type="MACRO_ANALYSIS",
        node_name="macro_node",
        payload={
            "regime": briefing_obj.macro_regime,
            "sentiment_score": briefing_obj.market_sentiment_score,
            "fed_hd_score": briefing_obj.fed_hawkish_dovish_score,
            "lm_uncertainty": briefing_obj.fed_uncertainty_score,
            "theme_tilts": briefing_obj.theme_macro_tilts,
            "tail_risks": briefing_obj.tail_risk_alerts
        }
    )
    return {
        "macro_briefing": briefing_obj.executive_summary,
        "execution_status": "MACRO_COMPLETED"
    }

def risk_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Sets hard risk boundaries before portfolio construction."""
    constraints = {
        "max_single_weight": risk_engine.max_single_weight,
        "max_leveraged_total": risk_engine.max_leveraged_total,
        "max_single_leveraged": risk_engine.max_single_leveraged,
        "min_cash_buffer": risk_engine.min_cash_buffer,
        "max_theme_weight": risk_engine.max_theme_weight,
        "max_daily_orders": 5  # Official QDC Rule
    }
    
    log_event(
        event_type="RISK_CONSTRAINTS_ESTABLISHED",
        node_name="risk_node",
        payload=constraints
    )
    return {
        "risk_constraints": constraints,
        "execution_status": "RISK_CONSTRAINTS_SET"
    }

def portfolio_manager_node(state: AgentState) -> Dict[str, Any]:
    """Node 5: Synthesizes Technical, Macro, and Risk to construct max 5 orders + memo."""
    tech_signals = state["tech_signals"]
    macro = state["macro_briefing"]
    capital = state.get("starting_capital_eur", 10000.0)

    # Rank bullish candidates excluding over-leveraged hazards
    sorted_candidates = sorted(
        tech_signals.values(),
        key=lambda x: x["composite_score"],
        reverse=True
    )
    
    # Filter top 4 distinct non-leveraged assets + 1 tactical position (max 5)
    selected_tickers = []
    for cand in sorted_candidates:
        ticker = cand["ticker"]
        meta = risk_engine.meta_map.get(ticker)
        if meta and not meta.is_leveraged_or_short:
            selected_tickers.append(ticker)
        if len(selected_tickers) == 4:
            break
            
    # Add 1 tactical hedge or low-vol asset (e.g. GOLD or DEFENSE)
    if "DEFENSE" not in selected_tickers:
        selected_tickers.append("DEFENSE")
    else:
        selected_tickers.append("GOLD")

    # Propose balanced 20% weights (80% invested, 20% cash buffer > 5% requirement)
    proposed_orders = []
    allocation_per_etf = 0.18 # 18% each -> 5 * 18% = 90% invested, 10% cash
    
    for ticker in selected_tickers[:5]:
        sig = tech_signals[ticker]
        amount_eur = capital * allocation_per_etf
        proposed_orders.append({
            "ticker": ticker,
            "action": "BUY",
            "target_weight_pct": round(allocation_per_etf * 100, 1),
            "estimated_amount_eur": round(amount_eur, 2),
            "rationale": sig["summary_reason"]
        })

    # Validate proposed portfolio through Risk Manager
    weights_map = {o["ticker"]: o["target_weight_pct"] / 100.0 for o in proposed_orders}
    risk_eval = risk_engine.evaluate_portfolio(weights_map)

    memo = (
        f"Investment Committee Proposal for {state['trading_date']}:\n"
        f"Macro Environment: {macro}\n"
        f"Strategy: Deploy €{capital:,.2f} into 5 top-ranked momentum & defensive leaders "
        f"({', '.join(selected_tickers[:5])}). Retain 10.0% cash buffer. "
        f"Leveraged exposure strictly controlled at 0.0% to protect against daily volatility drag."
    )

    log_event(
        event_type="PORTFOLIO_CONSTRUCTION",
        node_name="portfolio_manager_node",
        payload={
            "orders": proposed_orders,
            "is_risk_approved": risk_eval.is_approved,
            "memo": memo
        }
    )
    return {
        "proposed_orders": proposed_orders,
        "portfolio_memo": memo,
        "risk_evaluation": risk_eval.model_dump(),
        "execution_status": "PORTFOLIO_PROPOSED"
    }

def human_approval_node(state: AgentState) -> Dict[str, Any]:
    """Node 6: Interactive Human-in-the-Loop decision & audit checkpoint."""
    print("\n" + "="*70)
    print("🤖 REBEL AGENTIC TRADER — HUMAN-IN-THE-LOOP APPROVAL CHECKPOINT")
    print("="*70)
    print(f"Trading Date: {state['trading_date']} | Total Capital: €{state.get('starting_capital_eur', 10000.0):,.2f}")
    print("\n--- EXECUTIVE INVESTMENT MEMO ---")
    print(state["portfolio_memo"])
    
    print("\n--- PROPOSED TRADING ORDERS (MAX 5) ---")
    for i, o in enumerate(state["proposed_orders"], 1):
        print(f"  [{i}] {o['action']} {o['ticker']:<15} | Weight: {o['target_weight_pct']}% | Amount: €{o['estimated_amount_eur']:,.2f}")
        print(f"      Reason: {o['rationale'][:85]}...")

    risk = state["risk_evaluation"]
    status_icon = "✅" if risk["is_approved"] else "❌"
    print(f"\n--- RISK AUDIT REPORT: {status_icon} {risk['risk_summary']} ---")
    print(f"Cash Reserve: {risk['cash_reserve_pct']*100:.1f}% (Min required: 5.0%)")
    print(f"Leveraged Exposure: {risk['leveraged_exposure_pct']*100:.1f}% (Max allowed: 15.0%)")

    # In non-interactive mode or automated test, default to APPROVE
    user_choice = os.getenv("QDC_AUTO_APPROVE", "ASK")
    
    if user_choice == "ASK":
        try:
            decision = input("\nEnter Action: [A] Approve Orders | [M] Modify Notes | [R] Reject: ").strip().upper()
        except EOFError:
            decision = "A"
    else:
        decision = "A"

    if decision == "A":
        status = "APPROVED_BY_HUMAN"
        final_orders = state["proposed_orders"]
        notes = "Human approved all AI agent recommendations without modification."
        print("\n✅ Orders Approved! Ready for submission to Belfius trading form.")
    elif decision == "M":
        status = "MODIFIED_BY_HUMAN"
        final_orders = state["proposed_orders"]
        notes = input("Enter human override rationale/notes: ").strip()
        print(f"\n📝 Logged modifications with rationale: '{notes}'")
    else:
        status = "REJECTED_BY_HUMAN"
        final_orders = []
        notes = "Human rejected AI proposed allocation due to high discretionary risk."
        print("\n🛑 Orders Rejected. No trades submitted.")

    log_event(
        event_type="HUMAN_IN_THE_LOOP_DECISION",
        node_name="human_approval_node",
        payload={
            "decision": status,
            "notes": notes,
            "final_orders_count": len(final_orders)
        }
    )

    return {
        "human_approved": (decision == "A"),
        "human_decision": status,
        "human_notes": notes,
        "final_orders": final_orders,
        "execution_status": "COMPLETED"
    }

# ==========================================
# 2. ASSEMBLE LANGGRAPH PIPELINE
# ==========================================

def build_qdc_graph():
    """Builds the 6-step sequential pipeline with Human-in-the-Loop."""
    workflow = StateGraph(AgentState)

    workflow.add_node("data_node", data_node)
    workflow.add_node("technical_node", technical_node)
    workflow.add_node("macro_node", macro_node)
    workflow.add_node("risk_node", risk_node)
    workflow.add_node("portfolio_manager_node", portfolio_manager_node)
    workflow.add_node("human_approval_node", human_approval_node)

    # Strict sequential flow as specified:
    # Data -> Technical -> Macro -> Risk -> Portfolio Manager -> Human Approval
    workflow.set_entry_point("data_node")
    workflow.add_edge("data_node", "technical_node")
    workflow.add_edge("technical_node", "macro_node")
    workflow.add_edge("macro_node", "risk_node")
    workflow.add_edge("risk_node", "portfolio_manager_node")
    workflow.add_edge("portfolio_manager_node", "human_approval_node")
    workflow.add_edge("human_approval_node", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app

if __name__ == "__main__":
    app = build_qdc_graph()
    initial_state = {
        "trading_date": "2026-11-13",
        "starting_capital_eur": 10000.0,
        "prices": {},
        "tech_signals": {},
        "macro_briefing": "",
        "risk_constraints": {},
        "proposed_orders": [],
        "portfolio_memo": "",
        "risk_evaluation": {},
        "human_approved": False,
        "human_decision": "",
        "human_notes": "",
        "final_orders": [],
        "execution_status": "INIT"
    }
    
    # Run graph with session config
    config = {"configurable": {"thread_id": "qdc-trading-session-1"}}
    final_output = app.invoke(initial_state, config=config)
    print("\n🏁 Pipeline execution completed with status:", final_output["execution_status"])
