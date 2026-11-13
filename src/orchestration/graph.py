# src/orchestration/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.orchestration.state import AgentState
from src.agents.risk import RiskManager, RiskEvaluation
from src.pipeline.etf_universe import ETF_UNIVERSE

etf_meta_dict = {etf.ticker_internal: etf for etf in ETF_UNIVERSE}
risk_governor = RiskManager()

# --- Node Implementations ---

def data_node(state: AgentState):
    # Calls your Task 1 pipeline
    return {"prices": {"WORLD": 102.5, "NASDAQ 3X": 85.2, "GOLD": 195.4}}

def technical_node(state: AgentState):
    # Calls your Task 2 technical function
    signals = {
        "WORLD": {"score": 0.35, "trend": "BULLISH", "reason": "EMA20 above SMA50"},
        "NASDAQ 3X": {"score": -0.35, "trend": "BEARISH", "reason": "Overbought exhaustion"},
    }
    return {"tech_signals": signals}

def macro_node(state: AgentState):
    # LLM node: Summarizes macro context into regime classification
    return {"macro_briefing": "Fed rate cut expected; energy steady; tech under near-term scrutiny."}

def portfolio_manager_node(state: AgentState):
    # LLM node: Generates proposed allocation based on Tech + Macro + Risk feedback
    proposals = [
        {"ticker": "WORLD", "action": "BUY", "weight": 0.20, "rationale": "Solid base trend"},
        {"ticker": "NASDAQ 3X", "action": "BUY", "weight": 0.05, "rationale": "High-beta tactical allocation"},
    ]
    memo = "Allocating 20% to Global Benchmark and 5% tactical to NASDAQ 3X with 75% in cash/short bonds."
    return {"proposed_orders": proposals, "portfolio_memo": memo}

def risk_node(state: AgentState):
    # Calls Task 2 Risk Engine
    proposed = {p["ticker"]: p["weight"] for p in state["proposed_orders"]}
    eval_result = risk_governor.evaluate_portfolio(proposed, etf_meta_dict)
    return {"risk_evaluation": eval_result.model_dump()}

def risk_router(state: AgentState):
    # If risk fails, loop back to Portfolio Manager; otherwise proceed to human check
    if state["risk_evaluation"]["is_approved"]:
        return "human_approval_node"
    return "portfolio_manager_node"

def human_approval_node(state: AgentState):
    # Checkpoint node: logs the final human decision vs AI proposal
    return {
        "human_approved": True,
        "final_trade_log": state["proposed_orders"]
    }

# --- Build LangGraph Pipeline ---

workflow = StateGraph(AgentState)

workflow.add_node("data_node", data_node)
workflow.add_node("technical_node", technical_node)
workflow.add_node("macro_node", macro_node)
workflow.add_node("portfolio_manager_node", portfolio_manager_node)
workflow.add_node("risk_node", risk_node)
workflow.add_node("human_approval_node", human_approval_node)

# Flow Order
workflow.set_entry_point("data_node")
workflow.add_edge("data_node", "technical_node")
workflow.add_edge("technical_node", "macro_node")
workflow.add_edge("macro_node", "portfolio_manager_node")
workflow.add_edge("portfolio_manager_node", "risk_node")

# Conditional Edge for Risk Guardrail
workflow.add_conditional_edges(
    "risk_node",
    risk_router,
    {
        "portfolio_manager_node": "portfolio_manager_node",
        "human_approval_node": "human_approval_node"
    }
)

workflow.add_edge("human_approval_node", END)

# Set up human checkpoint interrupt before the approval step
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer, interrupt_before=["human_approval_node"])