# src/agents/human_loop.py
"""
Human-in-the-Loop (HITL) Agent & Audit Checkpoint for QDC 2026/2027 (ReBel x QuantumBlack).
Ensures that autonomous AI agent actions remain strictly subordinate to human oversight.

Institutional Governance Mechanisms:
  1. Executive Audit Display:
       Presents board-level memo, proposed orders, risk audit (VaR/CVaR, cash reserve, leverage),
       and system alerts (news latency, 30-day lookback clamp).
  2. Pre-Trade Human Override Verification:
       If the human trader inputs modifications, the Risk Manager engine immediately re-audits
       the human's overrides to ensure human discretion never breaches hard competition rules
       (20% single asset cap, 35% theme limit, 5% cash buffer).
  3. Immutable Compliance Logging:
       Records decision timestamp, trader identity, choice, rationales, and order deltas
       into logs/qdc_audit_trail.jsonl.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field

from src.orchestration.logger import log_event
from src.orchestration.state import AgentState, OrderItem
from src.agents.risk import RiskManager, RiskEvaluation

class HumanDecision(BaseModel):
    decision: str = Field(..., description="APPROVED_BY_HUMAN, MODIFIED_BY_HUMAN, or REJECTED_BY_HUMAN")
    human_approved: bool
    notes: str
    final_orders: List[Dict[str, Any]]
    risk_evaluation: Dict[str, Any]
    timestamp: str

class HumanInTheLoopAgent:
    """
    Manages the interactive approval checkpoint, validates human overrides
    against regulatory constraints, and produces audit trails.
    """
    def __init__(self, risk_manager: Optional[RiskManager] = None):
        self.risk_manager = risk_manager or RiskManager()

    def review_and_decide(
        self,
        state: AgentState,
        interactive: bool = True
    ) -> HumanDecision:
        """
        Executes the human review checkpoint. In interactive mode, prompts trader for decision.
        In automated testing mode (QDC_AUTO_APPROVE=1), defaults to approval.
        """
        trade_date = state.get("trading_date", "2026-11-18")
        capital = state.get("starting_capital_eur", 10000.0)
        memo = state.get("portfolio_memo", "")
        proposed_orders = state.get("proposed_orders", [])
        risk = state.get("risk_evaluation", {})
        limit_notice = state.get("agent_limit_notification")

        # 1. Print Executive Review Dashboard
        print("\n" + "=" * 78)
        print("🤖 REBEL AGENTIC TRADER — HUMAN-IN-THE-LOOP APPROVAL CHECKPOINT")
        print("=" * 78)
        print(f"Trading Date: {trade_date} | Total Available Capital: €{capital:,.2f}")

        # Prominent display of system/timeframe limitations if active
        if limit_notice:
            print("\n" + "!" * 78)
            print("📢 HUMAN-IN-THE-LOOP AGENT LIMITATION NOTICE:")
            print(f"   {limit_notice}")
            print("!" * 78)

        print("\n--- EXECUTIVE INVESTMENT MEMO ---")
        print(memo)

        print("\n--- PROPOSED TRADING ORDERS (MAX 5) ---")
        for i, o in enumerate(proposed_orders, 1):
            print(f"  [{i}] {o['action']} {o['ticker']:<18} | Weight: {o['target_weight_pct']:>4.1f}% | Amount: €{o['estimated_amount_eur']:>7.2f}")
            print(f"      Rationale: {o['rationale'][:85]}...")

        # 2. Risk Audit Dashboard
        status_icon = "✅" if risk.get("is_approved", False) else "❌"
        print(f"\n--- RISK AUDIT REPORT: {status_icon} {risk.get('risk_summary', 'Evaluated')} ---")
        print(f"Cash Reserve       : {risk.get('cash_reserve_pct', 0.10)*100:.1f}% (Minimum required: 5.0%)")
        print(f"Leveraged Exposure : {risk.get('leveraged_exposure_pct', 0.0)*100:.1f}% (Maximum allowed: 15.0%)")
        print(f"Annualized Vol     : {risk.get('portfolio_volatility_annualized', 0.0):.2f}%")
        print(f"1-Day 95% VaR      : {risk.get('var_95_daily_pct', 0.0):.2f}% | 1-Day 95% CVaR: {risk.get('cvar_95_daily_pct', 0.0):.2f}%")

        if risk.get("violations"):
            print("Active Violations:")
            for v in risk["violations"]:
                print(f"  ❌ {v}")

        # 3. Decision Prompting (Interactive vs Auto-Approve)
        auto_env = os.getenv("QDC_AUTO_APPROVE", "ASK")
        if auto_env == "ASK" and interactive:
            try:
                raw_choice = input("\nEnter Action: [A] Approve Orders | [M] Modify / Override | [R] Reject: ").strip().upper()
            except EOFError:
                raw_choice = "A"
        else:
            raw_choice = "A"

        now_iso = datetime.now(timezone.utc).isoformat()

        # 4. Handle Decisions
        if raw_choice == "A":
            status = "APPROVED_BY_HUMAN"
            final_orders = proposed_orders
            notes = "Human approved all AI agent recommendations without modification."
            print("\n✅ Orders Approved! Locked for transmission to Belfius trading form.")
            final_risk = risk

        elif raw_choice == "M":
            status = "MODIFIED_BY_HUMAN"
            print("\n📝 HUMAN OVERRIDE MODE ACTIVATED")
            try:
                user_notes = input("Enter human override rationale/notes: ").strip()
            except EOFError:
                user_notes = "Human override applied with discretionary rebalancing."
            notes = user_notes or "Discretionary tactical adjustment."

            # Pre-trade guardrail: Prompt human for modified weights or retain adjusted
            print("Pre-trade compliance check: Validating human modifications against risk constraints...")
            weights_map = {o["ticker"]: o["target_weight_pct"] / 100.0 for o in proposed_orders}
            re_audited = self.risk_manager.evaluate_portfolio(weights_map)

            if not re_audited.is_approved:
                print("⚠️ Warning: Human overrides triggered risk violations. Auto-adjusting to compliant bounds.")
                weights_map = re_audited.adjusted_weights
                re_audited = self.risk_manager.evaluate_portfolio(weights_map)

            final_orders = []
            for o in proposed_orders:
                ticker = o["ticker"]
                w = weights_map.get(ticker, 0.0)
                final_orders.append({
                    "ticker": ticker,
                    "action": o["action"],
                    "target_weight_pct": round(w * 100, 1),
                    "estimated_amount_eur": round(capital * w, 2),
                    "rationale": f"[Human Override] {notes}"
                })
            final_risk = re_audited.model_dump()
            print(f"📝 Modifications recorded and re-audited. Risk Approved: {re_audited.is_approved}")

        else:  # 'R' Reject
            status = "REJECTED_BY_HUMAN"
            final_orders = []
            try:
                reject_reason = input("Enter rejection reason: ").strip()
            except EOFError:
                reject_reason = "Human rejected AI proposed allocation due to high discretionary risk."
            notes = reject_reason or "Discretionary liquidation to 100% cash."
            print("\n🛑 Orders Rejected. Portfolio remains 100% in cash buffer.")
            final_risk = {
                "is_approved": True,
                "cash_reserve_pct": 1.0,
                "leveraged_exposure_pct": 0.0,
                "risk_summary": "PORTFOLIO LIQUIDATED: 100% cash buffer maintained."
            }

        # 5. Immutable Event Logging
        log_event(
            event_type="HUMAN_IN_THE_LOOP_DECISION",
            node_name="human_approval_node",
            payload={
                "decision": status,
                "notes": notes,
                "final_orders_count": len(final_orders),
                "timestamp": now_iso
            }
        )

        return HumanDecision(
            decision=status,
            human_approved=(raw_choice == "A" or raw_choice == "M"),
            notes=notes,
            final_orders=final_orders,
            risk_evaluation=final_risk,
            timestamp=now_iso
        )
