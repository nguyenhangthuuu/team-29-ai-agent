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
from src.agents.technical import evaluate_all_etfs, retrain_if_new_data
from src.agents.risk import RiskManager
from src.agents.macro import evaluate_macro_agent
from src.agents.portfolio import PortfolioManagerAgent
from src.agents.human_loop import HumanInTheLoopAgent
from src.pipeline.portfolio_ledger import PortfolioLedger
from src.pipeline.etf_universe import ETF_UNIVERSE

load_dotenv()
risk_engine = RiskManager()
ledger = PortfolioLedger()
portfolio_engine = PortfolioManagerAgent(risk_manager=risk_engine)
hitl_engine = HumanInTheLoopAgent(risk_manager=risk_engine, ledger=ledger)

# ==========================================
# 1. NODE IMPLEMENTATIONS (IN EXACT ORDER)
# ==========================================

def data_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Loads latest EUR closing prices for all 50 ETFs and updates Mark-to-Market NAV."""
    price_file = "data/etf_prices_eur.parquet"
    if os.path.exists(price_file):
        df = pd.read_parquet(price_file)
        latest_prices = {col: float(df[col].dropna().iloc[-1]) for col in df.columns}
        trade_date = str(df.index[-1].date())
    else:
        latest_prices = {etf.ticker: 100.0 for etf in ETF_UNIVERSE}
        trade_date = "2026-11-13"

    # Mark to market current portfolio positions
    current_nav = ledger.mark_to_market(latest_prices)
    active_capital = current_nav if current_nav > 0 else state.get("starting_capital_eur", 10000.0)

    log_event(
        event_type="DATA_INGESTION",
        node_name="data_node",
        payload={"date": trade_date, "etf_count": len(latest_prices), "nav": current_nav}
    )
    return {
        "trading_date": trade_date,
        "prices": latest_prices,
        "portfolio_nav": current_nav,
        "starting_capital_eur": active_capital,
        "execution_status": "DATA_LOADED"
    }

def technical_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Auto-updates Random Forest model if fresh data exists, then generates signals."""
    price_file = "data/etf_prices_eur.parquet"
    df = pd.read_parquet(price_file)

    # 1. Self-updating ML: Auto-retrain if fresh prices detected
    retrained, ml_msg, _ = retrain_if_new_data(df, force=False)
    if retrained:
        log_event(
            event_type="MODEL_RETRAINED",
            node_name="technical_node",
            payload={"message": ml_msg}
        )

    # 2. Compute signals across 50 ETFs
    signals = evaluate_all_etfs(df)
    signal_dict = {ticker: sig.model_dump() for ticker, sig in signals.items()}
    
    log_event(
        event_type="TECHNICAL_ANALYSIS",
        node_name="technical_node",
        payload={"ranked_count": len(signal_dict), "ml_status": ml_msg}
    )
    return {
        "tech_signals": signal_dict,
        "execution_status": "TECHNICAL_COMPLETED"
    }

def macro_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Evaluates macro regime using Fed & ECB Communication, Live Headline Stream (Sources A, B, C),
    and real-time market indicators (VIX, 10Y Yields, Oil, DXY, EUR/USD).
    """
    trade_date = state.get("trading_date", "2026-11-18")
    ref_date = state.get("reference_date", None)
    briefing_obj = evaluate_macro_agent(trading_date=trade_date, reference_date=ref_date)
    
    log_event(
        event_type="MACRO_ANALYSIS",
        node_name="macro_node",
        payload={
            "regime": briefing_obj.macro_regime,
            "sentiment_score": briefing_obj.market_sentiment_score,
            "fed_hd_score": briefing_obj.fed_hawkish_dovish_score,
            "fed_uncertainty": briefing_obj.fed_uncertainty_score,
            "ecb_hd_score": briefing_obj.ecb_hawkish_dovish_score,
            "ecb_uncertainty": briefing_obj.ecb_uncertainty_score,
            "transatlantic_policy_spread": briefing_obj.transatlantic_policy_spread,
            "headline_sentiment": briefing_obj.headline_stream.headline_sentiment_score,
            "headline_regime": briefing_obj.headline_stream.news_regime,
            "news_latency_minutes": briefing_obj.headline_stream.news_latency_minutes,
            "latest_scraped_timestamp": briefing_obj.headline_stream.latest_scraped_timestamp,
            "timeframe_window": briefing_obj.headline_stream.timeframe_window_used,
            "theme_tilts": briefing_obj.theme_macro_tilts,
            "tail_risks": briefing_obj.tail_risk_alerts
        }
    )
    # Optional Trigger: Export all collected headlines as-is to CSV if enabled (default: False)
    export_trigger = state.get("export_headlines_csv", False)
    if export_trigger:
        try:
            from src.pipeline.headline_audit import export_headlines_csv
            csv_path = str(export_trigger) if isinstance(export_trigger, str) else "data/headlines_collected.csv"
            saved_csv = export_headlines_csv(output_path=csv_path)
            log_event(
                event_type="HEADLINES_EXPORTED",
                node_name="macro_node",
                payload={"csv_path": saved_csv}
            )
        except Exception as e:
            print(f"Notice during headline CSV export: {e}")

    return {
        "macro_briefing": briefing_obj.executive_summary,
        "agent_limit_notification": briefing_obj.agent_limit_notification,
        "execution_status": "MACRO_COMPLETED"
    }

def risk_node(state: AgentState) -> Dict[str, Any]:
    """Node 4: Configures risk guardrails based on state risk_profile."""
    risk_profile = state.get("risk_profile", "MODERATE").upper()
    risk_engine.set_risk_profile(risk_profile)
    constraints = {
        "risk_profile": risk_profile,
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
    """Node 5: Constructs high-conviction orders + memo with profile & argument adaptation."""
    tech_signals = state["tech_signals"]
    capital = state.get("starting_capital_eur", 10000.0)
    trade_date = state.get("trading_date", "2026-11-18")
    ref_date = state.get("reference_date", None)
    risk_profile = state.get("risk_profile", "MODERATE")
    user_args = state.get("user_arguments", None)
    
    # Retrieve structured macro briefing
    macro_briefing_obj = evaluate_macro_agent(trading_date=trade_date, reference_date=ref_date)
    
    proposal, risk_eval = portfolio_engine.construct_portfolio(
        trading_date=trade_date,
        tech_signals=tech_signals,
        macro_briefing=macro_briefing_obj,
        starting_capital_eur=capital,
        risk_profile=risk_profile,
        user_arguments=user_args,
        current_holdings=ledger.positions
    )
    
    orders_dict = [o.model_dump() for o in proposal.orders]
    
    log_event(
        event_type="PORTFOLIO_CONSTRUCTION",
        node_name="portfolio_manager_node",
        payload={
            "orders": orders_dict,
            "is_risk_approved": risk_eval.is_approved,
            "risk_profile": risk_profile,
            "memo": proposal.executive_memo
        }
    )
    return {
        "proposed_orders": orders_dict,
        "portfolio_memo": proposal.executive_memo,
        "risk_evaluation": risk_eval.model_dump(),
        "execution_status": "PORTFOLIO_PROPOSED"
    }

def human_approval_node(state: AgentState) -> Dict[str, Any]:
    """Node 6: Interactive Review / Mandate Execution & Portfolio Ledger Accounting."""
    decision_obj = hitl_engine.review_and_decide(state)
    
    # Retrieve latest ledger summary
    summary = ledger.get_summary_report()

    return {
        "human_approved": decision_obj.human_approved,
        "human_decision": decision_obj.decision,
        "human_notes": decision_obj.notes,
        "final_orders": decision_obj.final_orders,
        "risk_evaluation": decision_obj.risk_evaluation,
        "portfolio_nav": summary["current_nav"],
        "period_pnl": summary["total_cumulative_pnl"],
        "ledger_summary": summary,
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
        "risk_profile": "MODERATE",
        "service_model": "MANDATE",
        "user_arguments": None,
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
        "portfolio_nav": None,
        "period_pnl": None,
        "ledger_summary": None,
        "execution_status": "INIT"
    }
    
    # Run graph with session config
    config = {"configurable": {"thread_id": "qdc-trading-session-1"}}
    final_output = app.invoke(initial_state, config=config)
    print("\nPipeline execution completed with status:", final_output["execution_status"])
