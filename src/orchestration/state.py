# src/orchestration/state.py
from typing import TypedDict, Dict, List, Optional
from pydantic import BaseModel

class OrderProposal(BaseModel):
    ticker: str
    action: str  # BUY, SELL, HOLD
    weight: float
    rationale: str

class AgentState(TypedDict):
    trading_date: str
    prices: Dict[str, float]
    tech_signals: Dict[str, dict]
    macro_briefing: str
    risk_evaluation: Optional[dict]
    proposed_orders: List[dict]
    portfolio_memo: str
    human_approved: bool
    human_overrides: Optional[List[dict]]
    final_trade_log: List[dict]
    token_usage_log: List[dict]