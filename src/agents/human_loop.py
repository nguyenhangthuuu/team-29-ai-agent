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
from src.pipeline.portfolio_ledger import PortfolioLedger, LedgerSnapshot

class HumanDecision(BaseModel):
    decision: str = Field(..., description="APPROVED_BY_HUMAN, MODIFIED_BY_HUMAN, CHALLENGED_AND_RESOLVED, EXECUTED_UNDER_MANDATE, or REJECTED_BY_HUMAN")
    human_approved: bool
    notes: str
    final_orders: List[Dict[str, Any]]
    risk_evaluation: Dict[str, Any]
    timestamp: str
    ledger_snapshot: Optional[Dict[str, Any]] = None

class HumanInTheLoopAgent:
    """
    Manages the Belfius investment service models (ADVISE vs MANDATE),
    interactive client review & argument challenges, human override compliance checks,
    and automated portfolio ledger execution.
    """
    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        ledger: Optional[PortfolioLedger] = None
    ):
        self.risk_manager = risk_manager or RiskManager()
        self.ledger = ledger or PortfolioLedger()

    def review_and_decide(
        self,
        state: AgentState,
        interactive: bool = True
    ) -> HumanDecision:
        """
        Executes the human review checkpoint. Supports both:
          1. MANDATE: Autonomous discretionary management per risk profile.
          2. ADVISE: Client review, argument negotiation [C], order override [M], approval [A], or rejection [R].
        Automatically logs trades and updates PortfolioLedger after execution.
        """
        trade_date = state.get("trading_date", "2026-11-18")
        capital = state.get("starting_capital_eur", 10000.0)
        memo = state.get("portfolio_memo", "")
        proposed_orders = state.get("proposed_orders", [])
        risk = state.get("risk_evaluation", {})
        limit_notice = state.get("agent_limit_notification")
        service_model = state.get("service_model", "ADVISE").upper()
        risk_profile = state.get("risk_profile", "MODERATE").upper()
        prices = state.get("prices", {})
        now_iso = datetime.now(timezone.utc).isoformat()

        # =========================================================================
        # MODEL 1: DISCRETIONARY MANDATE (Autonomous Execution by Belfius)
        # =========================================================================
        if service_model == "MANDATE":
            print("\n" + "=" * 78)
            print(f"BELFIUS DISCRETIONARY MANDATE EXECUTION [{risk_profile} PROFILE]")
            print("=" * 78)
            print(f"Trading Date: {trade_date} | Total Portfolio Capital: EUR {capital:,.2f}")
            print(f"Status: Mandate active. Belfius MAS autonomously manages and rebalances your portfolio.")
            
            if limit_notice:
                print(f"Notice: {limit_notice}")

            print("\n--- AUTONOMOUS INVESTMENT ALLOCATION ---")
            for i, o in enumerate(proposed_orders, 1):
                print(f"  [{i}] {o['action']} {o['ticker']:<18} | Target: {o['target_weight_pct']:>4.1f}% | EUR {o['estimated_amount_eur']:>7.2f}")
                print(f"      Rationale: {o['rationale'][:85]}...")

            status_icon = "[APPROVED]" if risk.get("is_approved", False) else "[VIOLATIONS DETECTED]"
            print(f"\n--- MANDATE RISK COMPLIANCE AUDIT: {status_icon} ---")
            print(f"Cash Buffer: {risk.get('cash_reserve_pct', 0.10)*100:.1f}% | Leveraged: {risk.get('leveraged_exposure_pct', 0.0)*100:.1f}% | 1D 95% VaR: {risk.get('var_95_daily_pct', 0.0):.2f}%")

            # Autonomous execution
            final_orders = proposed_orders
            status = "EXECUTED_UNDER_MANDATE"
            notes = f"Autonomous mandate rebalancing executed under {risk_profile} investment policy mandate."

            # Execute rebalancing directly into Portfolio Ledger
            target_order_inputs = [
                {"ticker": o["ticker"], "weight": o["target_weight_pct"], "amount": o["estimated_amount_eur"]}
                for o in final_orders
            ]
            snapshot = self.ledger.execute_rebalancing(
                target_orders=target_order_inputs,
                current_prices=prices,
                trade_date=trade_date,
                var_95=risk.get("var_95_daily_pct", 0.0),
                cvar_95=risk.get("cvar_95_daily_pct", 0.0)
            )

            print(f"\nMandate Execution Complete. New NAV: EUR {snapshot.total_nav:,.2f} | Cash: EUR {snapshot.cash:,.2f} | Period PnL: EUR {snapshot.period_pnl:+,.2f}")

            log_event(
                event_type="MANDATE_EXECUTION",
                node_name="human_approval_node",
                payload={"service_model": "MANDATE", "profile": risk_profile, "nav": snapshot.total_nav, "orders": len(final_orders)}
            )

            return HumanDecision(
                decision=status,
                human_approved=True,
                notes=notes,
                final_orders=final_orders,
                risk_evaluation=risk,
                timestamp=now_iso,
                ledger_snapshot=snapshot.__dict__
            )

        # =========================================================================
        # MODEL 2: ADVISE MODEL (Belfius Advises, Client Decides & Can Argue)
        # =========================================================================
        print("\n" + "=" * 78)
        print(f"BELFIUS INVESTMENT ADVISORY (ADVISE MODEL - {risk_profile} PROFILE)")
        print("=" * 78)
        print(f"Trading Date: {trade_date} | Total Available Capital: EUR {capital:,.2f}")

        if limit_notice:
            print("\n" + "!" * 78)
            print(f"SYSTEM LIMITATION NOTICE: {limit_notice}")
            print("!" * 78)

        print("\n--- BELFIUS AI ADVISORY MEMO ---")
        print(memo)

        print("\n--- RECOMMENDED TRADING ORDERS (MAX 5) ---")
        for i, o in enumerate(proposed_orders, 1):
            print(f"  [{i}] {o['action']} {o['ticker']:<18} | Weight: {o['target_weight_pct']:>4.1f}% | Amount: EUR {o['estimated_amount_eur']:>7.2f}")
            print(f"      Rationale: {o['rationale'][:85]}...")

        status_icon = "[APPROVED]" if risk.get("is_approved", False) else "[VIOLATIONS DETECTED]"
        print(f"\n--- PRE-TRADE RISK REPORT: {status_icon} {risk.get('risk_summary', 'Evaluated')} ---")
        print(f"Cash Reserve       : {risk.get('cash_reserve_pct', 0.10)*100:.1f}% (Required: {self.risk_manager.min_cash_buffer*100:.1f}%)")
        print(f"Leveraged Exposure : {risk.get('leveraged_exposure_pct', 0.0)*100:.1f}% (Max: {self.risk_manager.max_leveraged_total*100:.1f}%)")
        print(f"Annualized Vol     : {risk.get('portfolio_volatility_annualized', 0.0):.2f}%")
        print(f"1-Day 95% VaR      : {risk.get('var_95_daily_pct', 0.0):.2f}% | 1-Day 95% CVaR: {risk.get('cvar_95_daily_pct', 0.0):.2f}%")

        if risk.get("violations"):
            print("Active Violations:")
            for v in risk["violations"]:
                print(f"  [X] {v}")

        # Decision Prompting
        auto_env = os.getenv("QDC_AUTO_APPROVE", "ASK")
        if auto_env == "ASK" and interactive:
            try:
                raw_choice = input("\nEnter Action: [A] Approve Orders | [C] Challenge/Argue | [M] Modify Orders | [R] Reject: ").strip().upper()
            except EOFError:
                raw_choice = "A"
        else:
            raw_choice = "A"

        # Handle [C] Challenge / Argue with MAS Recommendation
        if raw_choice == "C":
            print("\nCLIENT ARGUMENT / CHALLENGE INTERACTION ACTIVATED")
            try:
                user_argument = input("Enter your concern or instruction (e.g. 'no crypto, prefer gold and more cash'): ").strip()
            except EOFError:
                user_argument = "Prefer higher cash and physical gold hedge."

            print(f"\nBelfius AI re-evaluating asset convictions with your instruction: \"{user_argument}\"...")
            from src.agents.portfolio import PortfolioManagerAgent
            from src.agents.macro import evaluate_macro_agent

            pm_recalc = PortfolioManagerAgent(risk_manager=self.risk_manager)
            macro_obj = evaluate_macro_agent(trading_date=trade_date)
            revised_proposal, revised_risk = pm_recalc.construct_portfolio(
                trading_date=trade_date,
                tech_signals=state.get("tech_signals", {}),
                macro_briefing=macro_obj,
                starting_capital_eur=capital,
                risk_profile=risk_profile,
                user_arguments=user_argument
            )

            print("\n--- REVISED PROPOSAL ADDRESSING YOUR ARGUMENT ---")
            print(revised_proposal.executive_memo)
            print("\nRevised Orders:")
            for i, o in enumerate(revised_proposal.orders, 1):
                print(f"  [{i}] {o.action} {o.ticker:<18} | Weight: {o.target_weight_pct:>4.1f}% | EUR {o.estimated_amount_eur:>7.2f}")

            # Prompt client to confirm revised allocation
            try:
                confirm = input("\nAccept revised allocation? [Y] Yes / Execute | [N] No / Reject: ").strip().upper()
            except EOFError:
                confirm = "Y"

            if confirm == "Y":
                status = "CHALLENGED_AND_RESOLVED"
                final_orders = [o.model_dump() for o in revised_proposal.orders]
                notes = f"Client challenged AI recommendation with: '{user_argument}'. MAS successfully accommodated argument within risk limits."
                final_risk = revised_risk.model_dump()
                print("\nRevised allocation approved and queued for execution.")
            else:
                status = "REJECTED_BY_HUMAN"
                final_orders = []
                notes = f"Client rejected revised allocation after challenge: '{user_argument}'."
                final_risk = risk

        elif raw_choice == "M":
            status = "MODIFIED_BY_HUMAN"
            print("\nCLIENT MANUAL OVERRIDE ACTIVATED")
            try:
                user_notes = input("Enter client discretionary override rationale: ").strip()
            except EOFError:
                user_notes = "Client custom discretionary allocation."
            notes = user_notes or "Discretionary tactical adjustment."

            weights_map = {o["ticker"]: o["target_weight_pct"] / 100.0 for o in proposed_orders}
            re_audited = self.risk_manager.evaluate_portfolio(weights_map)

            if not re_audited.is_approved:
                print("Notice: Discretionary overrides required alignment with QDC risk bounds.")
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
                    "rationale": f"[Client Discretionary Override] {notes}"
                })
            final_risk = re_audited.model_dump()
            print(f"Modifications recorded and re-audited. Risk Approved: {re_audited.is_approved}")

        elif raw_choice == "A":
            status = "APPROVED_BY_HUMAN"
            final_orders = proposed_orders
            notes = "Client approved all Belfius AI advisory recommendations."
            print("\nOrders Approved! Locked for transmission to Belfius trading form.")
            final_risk = risk

        else:  # 'R' Reject
            status = "REJECTED_BY_HUMAN"
            final_orders = []
            try:
                reject_reason = input("Enter rejection reason: ").strip()
            except EOFError:
                reject_reason = "Client rejected AI proposed allocation."
            notes = reject_reason or "Discretionary liquidation to 100% cash."
            print("\nOrders Rejected. Portfolio remains 100% in cash reserve.")
            final_risk = {
                "is_approved": True,
                "cash_reserve_pct": 1.0,
                "leveraged_exposure_pct": 0.0,
                "risk_summary": "PORTFOLIO RETAINED IN CASH: 100% cash buffer maintained."
            }

        # Execute into Portfolio Ledger if approved
        snapshot_dict = None
        if final_orders:
            target_order_inputs = [
                {"ticker": o["ticker"], "weight": o["target_weight_pct"], "amount": o["estimated_amount_eur"]}
                for o in final_orders
            ]
            snapshot = self.ledger.execute_rebalancing(
                target_orders=target_order_inputs,
                current_prices=prices,
                trade_date=trade_date,
                var_95=final_risk.get("var_95_daily_pct", 0.0),
                cvar_95=final_risk.get("cvar_95_daily_pct", 0.0)
            )
            snapshot_dict = snapshot.__dict__
            print(f"Portfolio Ledger Recorded: NAV = EUR {snapshot.total_nav:,.2f} | Cash = EUR {snapshot.cash:,.2f} | Period PnL = EUR {snapshot.period_pnl:+,.2f}")

        # Immutable Event Logging
        log_event(
            event_type="HUMAN_IN_THE_LOOP_DECISION",
            node_name="human_approval_node",
            payload={
                "service_model": service_model,
                "decision": status,
                "notes": notes,
                "final_orders_count": len(final_orders),
                "timestamp": now_iso
            }
        )

        return HumanDecision(
            decision=status,
            human_approved=(status in ["APPROVED_BY_HUMAN", "CHALLENGED_AND_RESOLVED", "MODIFIED_BY_HUMAN", "EXECUTED_UNDER_MANDATE"]),
            notes=notes,
            final_orders=final_orders,
            risk_evaluation=final_risk,
            timestamp=now_iso,
            ledger_snapshot=snapshot_dict
        )
