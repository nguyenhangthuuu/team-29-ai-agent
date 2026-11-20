# src/orchestration/state.py
from typing import TypedDict, List, Dict, Optional, Any
from pydantic import BaseModel, Field

class OrderItem(BaseModel):
    ticker: str
    action: str = Field(..., description="BUY or SELL")
    target_weight_pct: float = Field(..., description="Target allocation in percentage (e.g. 15.0 for 15%)")
    estimated_amount_eur: float = Field(..., description="Capital allocated in EUR based on starting capital")
    rationale: str

class PortfolioProposal(BaseModel):
    investment_theme: str
    macro_regime: str
    orders: List[OrderItem] = Field(..., max_items=5, description="Max 5 orders per competition day")
    cash_weight_pct: float = Field(..., description="Residual cash percentage (min 5%)")
    executive_memo: str = Field(..., description="Boardroom-level rationale explaining the allocation")
    alternative_candidates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top runner-up alternative recommendations (ranks 6-10) with quantitative evidence for Challenge [C] and Modify [M] modes"
    )

class AgentState(TypedDict):
    """The central state dictionary passed across all LangGraph nodes."""
    trading_date: str
    reference_date: Optional[str]  # Custom lookback date for macro headlines & news stream
    starting_capital_eur: float
    risk_profile: str  # CONSERVATIVE, MODERATE (default), AGGRESSIVE
    service_model: str  # ADVISE (client decides, can argue) vs MANDATE (Belfius manages autonomously)
    user_arguments: Optional[str]  # Client feedback, arguments, or custom constraints
    prices: Dict[str, float]
    tech_signals: Dict[str, dict]
    macro_briefing: str
    agent_limit_notification: Optional[str]  # Prominent notice for Human-in-the-Loop on timeframe limit
    risk_constraints: Dict[str, Any]
    proposed_orders: List[dict]
    portfolio_memo: str
    risk_evaluation: Dict[str, Any]
    human_approved: bool
    human_decision: str
    human_notes: str
    final_orders: List[dict]
    portfolio_nav: Optional[float]
    period_pnl: Optional[float]
    ledger_summary: Optional[Dict[str, Any]]
    export_headlines_csv: Optional[Any]  # Trigger (default: False/None) to export all collected headlines as-is to CSV
    execution_status: str
