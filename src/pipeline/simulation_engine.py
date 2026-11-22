# src/pipeline/simulation_engine.py
"""
Simulation & Session State Engine for Belfius ReBel Wealth Management MAS.
Enforces strict anti-lookahead rules:
- Current trading session is Day T.
- Execution prices, technical indicators, macro narrative, and statistics are strictly anchored to Day T-1 Close.
- Supports Time-Travel Stepping (Day T -> T+1), Fast-Forward Jump, and Full Portfolio Reset to €10,000.
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from src.pipeline.portfolio_ledger import PortfolioLedger, LedgerSnapshot, TradeRecord, Position

SIM_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "simulation_state.json"
)

PRICE_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "etf_prices_eur.parquet"
)


class SimulationEngine:
    def __init__(
        self,
        price_path: str = PRICE_FILE_PATH,
        state_path: str = SIM_STATE_PATH,
        initial_capital: float = 10000.0
    ):
        self.price_path = price_path
        self.state_path = state_path
        self.initial_capital = initial_capital
        self._prices_df: Optional[pd.DataFrame] = None

        # Session State
        self.mode: str = "SIMULATION"          # "SIMULATION" or "PRACTICE_LIVE"
        self.interaction_mode: str = "MANDATE" # "MANDATE" or "ADVISE"
        self.start_date: str = "2026-10-02"
        self.current_trading_date: str = "2026-11-20"
        self.as_of_market_date: str = "2026-11-19" # T-1 Close Date
        self.risk_profile: str = "MODERATE"
        self.status_message: str = "Simulation engine initialized."

        self.ledger = PortfolioLedger(initial_capital=initial_capital)
        self._load_prices()
        self._load_or_init_state()

    def _load_prices(self):
        if os.path.exists(self.price_path):
            df = pd.read_parquet(self.price_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            self._prices_df = df.sort_index()
        else:
            self._prices_df = pd.DataFrame()

    def get_prices(self) -> pd.DataFrame:
        if self._prices_df is None or self._prices_df.empty:
            self._load_prices()
        return self._prices_df

    def get_available_trading_dates(self) -> List[str]:
        df = self.get_prices()
        if df.empty:
            dates = ["2026-11-20"]
        else:
            dates = [str(d.date()) for d in df.index]

        # Ensure real-time calendar dates up to current system date (e.g. 2026-11-22) are selectable
        today_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if dates and today_str > dates[-1]:
            cur_date = datetime.date.fromisoformat(dates[-1]) + datetime.timedelta(days=1)
            target_date = datetime.date.fromisoformat(today_str)
            while cur_date <= target_date:
                dates.append(str(cur_date))
                cur_date += datetime.timedelta(days=1)
        return dates

    def resolve_dates(self, trading_date_str: str) -> Tuple[str, str]:
        """
        Given a current trading day T (which can be any date up to today in real time), resolves:
        1. T (validated current trading date)
        2. T-1 (most recent preceding trading date in market dataset strictly < T)
        """
        df = self.get_prices()
        market_dates = [str(d.date()) for d in df.index] if not df.empty else ["2026-11-20"]

        all_dates = self.get_available_trading_dates()
        t_date = trading_date_str if trading_date_str in all_dates else (all_dates[-1] if all_dates else "2026-11-20")

        # Find most recent market date strictly before t_date
        preceding = [d for d in market_dates if d < t_date]
        if preceding:
            as_of_date = preceding[-1]
        else:
            as_of_date = market_dates[0] if market_dates else t_date

        return t_date, as_of_date

    def _load_or_init_state(self):
        all_dates = self.get_available_trading_dates()
        default_latest = all_dates[-1] if all_dates else "2026-11-20"

        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.mode = data.get("mode", "SIMULATION")
                    self.interaction_mode = data.get("interaction_mode", "MANDATE")
                    self.start_date = data.get("start_date", "2026-10-02")
                    t_req = data.get("current_trading_date", default_latest)
                    self.current_trading_date, self.as_of_market_date = self.resolve_dates(t_req)
                    self.risk_profile = data.get("risk_profile", "MODERATE")
                    self.status_message = data.get("status_message", "Loaded saved simulation session.")
                    return
            except Exception:
                pass

        # Default initialization
        self.current_trading_date, self.as_of_market_date = self.resolve_dates(default_latest)
        self.start_date = all_dates[max(0, len(all_dates) - 45)] if len(all_dates) >= 45 else all_dates[0]
        self._save_state()

    def _save_state(self):
        state_dict = {
            "mode": self.mode,
            "interaction_mode": self.interaction_mode,
            "start_date": self.start_date,
            "current_trading_date": self.current_trading_date,
            "as_of_market_date": self.as_of_market_date,
            "risk_profile": self.risk_profile,
            "status_message": self.status_message,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)

    def init_session(
        self,
        start_date: str,
        mode: str = "SIMULATION",
        risk_profile: str = "MODERATE"
    ) -> Dict[str, Any]:
        """
        Resets portfolio to €10,000 starting cash and initializes session on start_date.
        All market telemetry will be anchored to start_date - 1 Close.
        """
        all_dates = self.get_available_trading_dates()
        t_date, as_of_date = self.resolve_dates(start_date)

        self.mode = mode
        self.start_date = t_date
        self.current_trading_date = t_date
        self.as_of_market_date = as_of_date
        self.risk_profile = risk_profile
        self.status_message = (
            f"Simulation initialized on {t_date}. "
            f"Decisions and order execution anchored to {as_of_date} Close (Day T-1). "
            f"Initial capital of €10,000.00 ready for deployment."
        )

        # Reset ledger completely
        self.ledger.reset()
        self._save_state()
        return self.get_telemetry()

    def step_next_trading_day(self) -> Dict[str, Any]:
        """
        Advances session by one trading day: Day T -> Day T+1.
        Updates Mark-to-Market with the newly revealed day's prices.
        """
        all_dates = self.get_available_trading_dates()
        if self.current_trading_date in all_dates:
            curr_idx = all_dates.index(self.current_trading_date)
            if curr_idx < len(all_dates) - 1:
                next_date = all_dates[curr_idx + 1]
                self.current_trading_date, self.as_of_market_date = self.resolve_dates(next_date)
                self.status_message = (
                    f"Advanced to next trading session: {self.current_trading_date}. "
                    f"Market data and signals updated to {self.as_of_market_date} Close."
                )
                self.update_mtm()
                self._save_state()
            else:
                self.status_message = f"Reached latest available session ({self.current_trading_date}). Cannot advance further."
        return self.get_telemetry()

    def jump_to_date(self, target_date_str: str) -> Dict[str, Any]:
        """
        Fast-forwards to target_date_str (must be <= latest available trading day and >= current).
        """
        all_dates = self.get_available_trading_dates()
        t_date, as_of_date = self.resolve_dates(target_date_str)
        latest_date = all_dates[-1]

        if t_date > latest_date:
            raise ValueError(f"Target date {t_date} exceeds latest available market date {latest_date}.")

        self.current_trading_date = t_date
        self.as_of_market_date = as_of_date
        self.status_message = (
            f"Fast-forwarded to trading session {t_date}. "
            f"Analytics anchored to {as_of_date} Close (Day T-1)."
        )
        self.update_mtm()
        self._save_state()
        return self.get_telemetry()

    def set_interaction_mode(self, mode: str) -> str:
        clean = mode.upper()
        if clean in ["MANDATE", "ADVISE"]:
            self.interaction_mode = clean
            self._save_state()
        return self.interaction_mode

    def update_mtm(self) -> float:
        """Updates Mark-to-Market using prices available up to as_of_market_date."""
        df = self.get_prices()
        if df.empty or self.as_of_market_date not in df.index:
            return self.ledger.cash

        date_slice = df.loc[:self.as_of_market_date]
        if date_slice.empty:
            return self.ledger.cash

        current_prices = {
            col: float(date_slice[col].dropna().iloc[-1])
            for col in date_slice.columns
            if not date_slice[col].dropna().empty
        }
        return self.ledger.mark_to_market(current_prices)

    def get_market_close_prices(self, as_of_date: Optional[str] = None) -> Dict[str, float]:
        """Returns closing prices for all ETFs on as_of_date."""
        target_dt = as_of_date or self.as_of_market_date
        df = self.get_prices()
        if df.empty:
            return {}

        date_slice = df.loc[:target_dt]
        if date_slice.empty:
            return {}

        return {
            col: float(date_slice[col].dropna().iloc[-1])
            for col in date_slice.columns
            if not date_slice[col].dropna().empty
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive session telemetry and ledger status."""
        current_nav = self.update_mtm()
        cash_val = self.ledger.cash
        invested_val = sum(p.market_value for p in self.ledger.positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self.ledger.positions.values())
        cum_ret_pct = ((current_nav - self.initial_capital) / self.initial_capital) * 100

        all_dates = self.get_available_trading_dates()
        curr_idx = all_dates.index(self.current_trading_date) if self.current_trading_date in all_dates else -1
        has_next_day = curr_idx >= 0 and curr_idx < len(all_dates) - 1

        positions_list = []
        for tkr, pos in self.ledger.positions.items():
            positions_list.append({
                "ticker": tkr,
                "shares": round(pos.shares, 4),
                "avg_entry_price": round(pos.avg_entry_price, 4),
                "current_price": round(pos.current_price, 4),
                "market_value": round(pos.market_value, 2),
                "unrealized_pnl": round(pos.unrealized_pnl, 2),
                "unrealized_pnl_pct": round(((pos.current_price - pos.avg_entry_price) / max(0.0001, pos.avg_entry_price)) * 100, 2),
                "weight_pct": round(pos.weight_pct, 2)
            })

        trade_log = []
        for tr in reversed(self.ledger.trade_history):
            trade_log.append({
                "trade_date": tr.trade_date,
                "ticker": tr.ticker,
                "action": tr.action,
                "shares": round(tr.shares, 4),
                "price": round(tr.price, 4),
                "total_amount": round(tr.total_amount, 2),
                "fee": tr.fee
            })

        return {
            "mode": self.mode,
            "interaction_mode": self.interaction_mode,
            "start_date": self.start_date,
            "current_trading_date": self.current_trading_date,
            "as_of_market_date": self.as_of_market_date,
            "risk_profile": self.risk_profile,
            "has_next_day": has_next_day,
            "portfolio_nav": round(current_nav, 2),
            "cash_balance": round(cash_val, 2),
            "cash_pct": round((cash_val / max(0.01, current_nav)) * 100, 2),
            "invested_value": round(invested_val, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "cumulative_return_pct": round(cum_ret_pct, 2),
            "positions": positions_list,
            "trade_log": trade_log,
            "status_message": self.status_message,
            "notice_banner": (
                f"Trading Session {self.current_trading_date} | "
                f"Execution price and quantitative models anchored to {self.as_of_market_date} Close (Day T-1)."
            )
        }
