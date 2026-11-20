# run_mas.py
"""
Belfius ReBel x QuantumBlack Multi-Agent System (MAS) — CLI Execution Entry Point
QTEM Data Challenge (QDC) Fall 2026 Edition | Team 29

Usage Examples:
  # 1. Run default execution (Moderate profile, Advise model, latest date)
  python run_mas.py

  # 2. Run under Discretionary Mandate (autonomous execution)
  python run_mas.py --service MANDATE --profile CONSERVATIVE

  # 3. Run with client argument challenge (e.g., exclude crypto, prefer gold)
  python run_mas.py --service ADVISE --arguments "Exclude crypto, prefer gold and hold more cash"

  # 4. Run automated test without interactive prompts
  python run_mas.py --service MANDATE --auto-approve

  # 5. Reset portfolio ledger to clean initial state (€10,000 cash)
  python run_mas.py --reset-ledger
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timezone

from src.orchestration.graph import build_qdc_graph
from src.orchestration.state import AgentState
from src.pipeline.portfolio_ledger import PortfolioLedger
from src.agents.risk import RISK_PROFILES

def parse_args():
    parser = argparse.ArgumentParser(
        description="Belfius ReBel Multi-Agent System (MAS) Execution Engine",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="MODERATE",
        choices=["CONSERVATIVE", "MODERATE", "AGGRESSIVE"],
        help="Investor risk profile under MiFID II (default: MODERATE)"
    )
    parser.add_argument(
        "--service",
        type=str,
        default="ADVISE",
        choices=["ADVISE", "MANDATE"],
        help="Investment service model: ADVISE (client decides) or MANDATE (autonomous) (default: ADVISE)"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Trading window evaluation date (YYYY-MM-DD). Defaults to latest date in price dataset."
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Available capital in EUR (defaults to current ledger NAV or €10,000)"
    )
    parser.add_argument(
        "--arguments",
        type=str,
        default=None,
        help="Client arguments / constraints (e.g. 'Exclude crypto, prefer gold and more cash')"
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Bypass interactive prompts and auto-approve AI recommendations"
    )
    parser.add_argument(
        "--reset-ledger",
        action="store_true",
        help="Reset portfolio ledger back to initial €10,000 cash balance before running"
    )
    parser.add_argument(
        "--inspect-headlines",
        action="store_true",
        help="Display the real-time headline audit trail blotter"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=5,
        help="Lookback window in days for macro headlines and CSV export (default: 5, maximum: 30)"
    )
    parser.add_argument(
        "--export-headlines-csv",
        nargs="?",
        const="data/headlines_collected.csv",
        default=False,
        help="Trigger (default: False) to generate a CSV file summarising all headlines collected by the agent as-is. Optional custom path can be specified."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Handle headline CSV export trigger if activated
    if args.export_headlines_csv:
        from src.pipeline.headline_audit import export_headlines_csv
        clamped_days = min(max(1, args.lookback_days), 30)
        out_csv = export_headlines_csv(
            output_path=str(args.export_headlines_csv),
            lookback_days=args.lookback_days
        )
        print(f"Summary CSV generated: {out_csv} (lookback: {clamped_days} days, all headlines preserved as-is).\n")

    # If user wants to inspect headlines
    if args.inspect_headlines:
        from src.pipeline.headline_audit import print_headline_audit_blotter
        print_headline_audit_blotter(limit=20, lookback_days=args.lookback_days)
        if not any([args.arguments, args.auto_approve, args.reset_ledger]) and len(sys.argv) == 2:
            return

    # 1. Handle Ledger Reset if requested
    ledger = PortfolioLedger()
    if args.reset_ledger:
        ledger.reset()
        print("Portfolio ledger successfully reset to initial EUR 10,000 cash balance (0.00% return).\n")

    # 2. Determine Evaluation Date
    trade_date = args.date
    if not trade_date:
        price_file = "data/etf_prices_eur.parquet"
        if os.path.exists(price_file):
            df = pd.read_parquet(price_file)
            trade_date = str(df.index[-1].date()) if isinstance(df.index, pd.DatetimeIndex) else str(df.index[-1])
        else:
            trade_date = "2026-11-13"

    # 3. Determine Available Capital (Mark-to-Market)
    capital = args.capital
    if capital is None:
        summary = ledger.get_summary_report()
        capital = summary["current_nav"] if summary["current_nav"] > 0 else 10000.0

    # 4. Set Environment for Human-in-the-Loop Mode
    if args.auto_approve or args.service == "MANDATE":
        os.environ["QDC_AUTO_APPROVE"] = "1"
    else:
        os.environ["QDC_AUTO_APPROVE"] = "ASK"

    print("=" * 78)
    print("BELFIUS REBEL AGENTIC TRADER - EXECUTION ENGINE")
    print("QuantumBlack x Belfius Bank | QDC Fall 2026 Edition (Team 29)")
    print("=" * 78)
    print(f"Trading Date        : {trade_date}")
    print(f"Investor Profile    : {args.profile}")
    print(f"Service Model       : {args.service}")
    print(f"Portfolio NAV       : EUR {capital:,.2f}")
    if args.arguments:
        print(f"Client Arguments    : \"{args.arguments}\"")
    print("=" * 78)

    # 5. Build and Execute LangGraph Pipeline
    app = build_qdc_graph()

    from datetime import timedelta
    clamped_lookback = min(max(1, args.lookback_days), 30)
    calculated_ref_date = (datetime.now(timezone.utc) - timedelta(days=clamped_lookback)).strftime("%Y-%m-%d")

    initial_state: AgentState = {
        "trading_date": trade_date,
        "reference_date": calculated_ref_date,
        "starting_capital_eur": capital,
        "risk_profile": args.profile,
        "service_model": args.service,
        "user_arguments": args.arguments,
        "prices": {},
        "tech_signals": {},
        "macro_briefing": "",
        "agent_limit_notification": None,
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
        "export_headlines_csv": args.export_headlines_csv,
        "execution_status": "INIT"
    }

    session_id = f"qdc-{args.profile.lower()}-{args.service.lower()}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    config = {"configurable": {"thread_id": session_id}}

    final_output = app.invoke(initial_state, config=config)

    # 6. Print Summary Report
    print("\n" + "=" * 78)
    print("FINAL EXECUTION SUMMARY")
    print("=" * 78)
    print(f"Status              : {final_output['execution_status']}")
    print(f"Decision            : {final_output.get('human_decision', 'N/A')}")
    print(f"Orders Executed     : {len(final_output.get('final_orders', []))}")
    if final_output.get("portfolio_nav"):
        print(f"Updated NAV         : EUR {final_output['portfolio_nav']:,.2f}")
    if final_output.get("period_pnl") is not None:
        print(f"Period PnL          : EUR {final_output['period_pnl']:+,.2f}")
    print("=" * 78)
    print("Persistent Storage Logs:")
    print("  • Portfolio Ledger Parquet: data/portfolio_ledger.parquet")
    print("  • Audit Trail JSONL       : logs/portfolio_ledger_audit.jsonl")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
