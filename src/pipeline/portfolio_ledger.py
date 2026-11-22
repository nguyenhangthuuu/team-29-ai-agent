# src/pipeline/portfolio_ledger.py
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import os
import json
import pandas as pd
import numpy as np

LEDGER_PARQUET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "portfolio_ledger.parquet"
)

AUDIT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
    "portfolio_ledger_audit.jsonl"
)


@dataclass
class Position:
    ticker: str
    shares: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight_pct: float


@dataclass
class TradeRecord:
    trade_date: str
    ticker: str
    action: str  # BUY or SELL
    shares: float
    price: float
    total_amount: float
    fee: float = 0.0


@dataclass
class LedgerSnapshot:
    trade_date: str
    cash: float
    invested_value: float
    total_nav: float
    period_pnl: float
    period_return_pct: float
    cumulative_return_pct: float
    var_95_pct: float
    cvar_95_pct: float
    holdings: Dict[str, Dict[str, float]]


class PortfolioLedger:
    """
    Institutional Mark-to-Market (MtM) Ledger & Position Tracker.
    Calculates exact NAV, Cash Reserves, PnL, and Sharpe inputs across QDC trading windows.
    """
    def __init__(
        self,
        initial_capital: float = 10000.0,
        ledger_path: str = LEDGER_PARQUET_PATH,
        audit_path: str = AUDIT_LOG_PATH
    ):
        self.initial_capital = initial_capital
        self.ledger_path = ledger_path
        self.audit_path = audit_path
        
        self.cash: float = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.snapshots: List[LedgerSnapshot] = []
        
        self._load_if_exists()

    def reset(self):
        """Resets ledger state and deletes persistent storage files."""
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_history = []
        self.snapshots = []
        if os.path.exists(self.ledger_path):
            try:
                os.remove(self.ledger_path)
            except Exception:
                pass
        if os.path.exists(self.audit_path):
            try:
                os.remove(self.audit_path)
            except Exception:
                pass

    def _load_if_exists(self):
        """Loads historical snapshots from parquet if file exists."""
        if os.path.exists(self.ledger_path):
            try:
                df = pd.read_parquet(self.ledger_path)
                if not df.empty:
                    last_row = df.iloc[-1]
                    self.cash = float(last_row.get("cash", self.initial_capital))
                    # Restore positions from serialized json if available
                    holdings_raw = last_row.get("holdings_json", "{}")
                    h_dict = json.loads(holdings_raw) if isinstance(holdings_raw, str) else holdings_raw
                    for tkr, data in h_dict.items():
                        self.positions[tkr] = Position(
                            ticker=tkr,
                            shares=float(data["shares"]),
                            avg_entry_price=float(data["avg_entry_price"]),
                            current_price=float(data["current_price"]),
                            market_value=float(data["market_value"]),
                            unrealized_pnl=float(data["unrealized_pnl"]),
                            weight_pct=float(data["weight_pct"])
                        )
            except Exception:
                pass

    def mark_to_market(self, current_prices: Dict[str, float]) -> float:
        """
        Updates positions with current market prices.
        Returns total portfolio NAV (Cash + Invested Value).
        """
        invested_val = 0.0
        for tkr, pos in list(self.positions.items()):
            if tkr in current_prices and current_prices[tkr] > 0:
                pos.current_price = float(current_prices[tkr])
                pos.market_value = pos.shares * pos.current_price
                pos.unrealized_pnl = (pos.current_price - pos.avg_entry_price) * pos.shares
            invested_val += pos.market_value

        total_nav = self.cash + invested_val
        for pos in self.positions.values():
            pos.weight_pct = round((pos.market_value / (total_nav + 1e-9)) * 100, 2)

        return round(total_nav, 2)

    def execute_rebalancing(
        self,
        target_orders: List[Dict[str, Any]],
        current_prices: Dict[str, float],
        trade_date: str,
        var_95: float = 0.0,
        cvar_95: float = 0.0
    ) -> LedgerSnapshot:
        """
        Executes orders strictly capped at 5 orders per session (QDC competition rule).
        Supports explicit BUY and SELL actions.
        Does NOT silently liquidate unmentioned positions.
        """
        # 1. Update MtM prior to trade
        prev_nav = self.mark_to_market(current_prices)
        
        # Enforce strict maximum 5 orders per session
        orders_to_execute = target_orders[:5]

        # First execute any explicit SELL orders so cash proceeds become available for BUY orders
        sell_orders = [o for o in orders_to_execute if str(o.get("action", "BUY")).upper() == "SELL"]
        buy_orders = [o for o in orders_to_execute if str(o.get("action", "BUY")).upper() != "SELL"]

        # 2. Execute explicit SELL orders
        for order in sell_orders:
            tkr = order["ticker"]
            p_curr = float(current_prices.get(tkr, 0.0))
            if p_curr <= 0.0:
                continue

            current_pos = self.positions.get(tkr)
            if not current_pos or current_pos.shares <= 1e-4:
                continue

            target_amount = float(order.get("amount", 0.0))
            target_weight = float(order.get("weight", 0.0))

            if target_amount > 0:
                sell_shares = min(current_pos.shares, target_amount / p_curr)
            elif target_weight > 0:
                sell_shares = min(current_pos.shares, (target_weight * prev_nav) / p_curr)
            else:
                # Full liquidation if amount and weight are 0
                sell_shares = current_pos.shares

            if sell_shares <= 1e-4:
                continue

            proceeds = sell_shares * p_curr
            self.cash += proceeds
            current_pos.shares -= sell_shares

            if current_pos.shares <= 1e-4:
                self.positions.pop(tkr, None)
            else:
                current_pos.market_value = current_pos.shares * p_curr
                current_pos.unrealized_pnl = (p_curr - current_pos.avg_entry_price) * current_pos.shares

            self.trade_history.append(TradeRecord(
                trade_date=trade_date,
                ticker=tkr,
                action="SELL",
                shares=round(sell_shares, 4),
                price=round(p_curr, 4),
                total_amount=round(proceeds, 2),
                fee=0.0
            ))

        # 3. Execute BUY orders
        current_nav = self.cash + sum(p.market_value for p in self.positions.values())
        for order in buy_orders:
            tkr = order["ticker"]
            p_curr = float(current_prices.get(tkr, 0.0))
            if p_curr <= 0.0:
                continue

            target_amount = float(order.get("amount", 0.0))
            target_weight = float(order.get("weight", 0.0))
            if target_amount <= 0.0 and target_weight > 0.0:
                target_amount = target_weight * current_nav

            if target_amount <= 0.0:
                continue

            # Limit by available cash
            actual_cost = min(target_amount, max(0.0, self.cash))
            if actual_cost <= 0.01:
                continue

            actual_shares = actual_cost / p_curr
            self.cash -= actual_cost

            current_pos = self.positions.get(tkr)
            if current_pos:
                total_cost = (current_pos.shares * current_pos.avg_entry_price) + actual_cost
                new_shares = current_pos.shares + actual_shares
                current_pos.shares = new_shares
                current_pos.avg_entry_price = total_cost / (new_shares + 1e-9)
                current_pos.current_price = p_curr
                current_pos.market_value = new_shares * p_curr
                current_pos.unrealized_pnl = (p_curr - current_pos.avg_entry_price) * new_shares
            else:
                self.positions[tkr] = Position(
                    ticker=tkr,
                    shares=actual_shares,
                    avg_entry_price=p_curr,
                    current_price=p_curr,
                    market_value=actual_cost,
                    unrealized_pnl=0.0,
                    weight_pct=0.0
                )

            self.trade_history.append(TradeRecord(
                trade_date=trade_date,
                ticker=tkr,
                action="BUY",
                shares=round(actual_shares, 4),
                price=round(p_curr, 4),
                total_amount=round(actual_cost, 2),
                fee=0.0
            ))

        # 4. Final Mark-to-Market & Period PnL
        final_nav = self.mark_to_market(current_prices)
        invested = sum(p.market_value for p in self.positions.values())
        
        period_pnl = final_nav - prev_nav if self.snapshots else 0.0
        period_ret = (period_pnl / (prev_nav + 1e-9)) * 100 if self.snapshots else 0.0
        cum_ret = ((final_nav / self.initial_capital) - 1.0) * 100

        holdings_dict = {
            t: {
                "shares": round(p.shares, 4),
                "avg_entry_price": round(p.avg_entry_price, 4),
                "current_price": round(p.current_price, 4),
                "market_value": round(p.market_value, 2),
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "weight_pct": p.weight_pct
            }
            for t, p in self.positions.items()
        }

        snapshot = LedgerSnapshot(
            trade_date=trade_date,
            cash=round(self.cash, 2),
            invested_value=round(invested, 2),
            total_nav=round(final_nav, 2),
            period_pnl=round(period_pnl, 2),
            period_return_pct=round(period_ret, 2),
            cumulative_return_pct=round(cum_ret, 2),
            var_95_pct=round(var_95, 2),
            cvar_95_pct=round(cvar_95, 2),
            holdings=holdings_dict
        )
        self.snapshots.append(snapshot)
        self._persist_snapshot(snapshot)

        return snapshot

    def _persist_snapshot(self, snapshot: LedgerSnapshot):
        """Appends snapshot to parquet ledger and jsonl audit trail."""
        row_dict = asdict(snapshot)
        row_dict["holdings_json"] = json.dumps(row_dict.pop("holdings"))

        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)

        # 1. Parquet update
        new_df = pd.DataFrame([row_dict])
        if os.path.exists(self.ledger_path):
            try:
                existing_df = pd.read_parquet(self.ledger_path)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                combined.drop_duplicates(subset=["trade_date"], keep="last", inplace=True)
                combined.to_parquet(self.ledger_path, index=False)
            except Exception:
                new_df.to_parquet(self.ledger_path, index=False)
        else:
            new_df.to_parquet(self.ledger_path, index=False)

        # 2. JSONL Audit Log
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(snapshot)) + "\n")

    def get_summary_report(self) -> Dict[str, Any]:
        """Generates comprehensive ledger summary report."""
        current_nav = self.cash + sum(p.market_value for p in self.positions.values())
        cum_ret = ((current_nav / self.initial_capital) - 1.0) * 100
        total_pnl = current_nav - self.initial_capital

        return {
            "initial_capital": self.initial_capital,
            "current_nav": round(current_nav, 2),
            "cash_balance": round(self.cash, 2),
            "cash_reserve_pct": round((self.cash / (current_nav + 1e-9)) * 100, 2),
            "invested_value": round(current_nav - self.cash, 2),
            "total_cumulative_pnl": round(total_pnl, 2),
            "total_cumulative_return_pct": round(cum_ret, 2),
            "active_positions_count": len(self.positions),
            "completed_windows": len(self.snapshots),
            "positions": {
                t: {
                    "shares": p.shares,
                    "market_value": p.market_value,
                    "weight_pct": p.weight_pct,
                    "unrealized_pnl": p.unrealized_pnl
                }
                for t, p in self.positions.items()
            }
        }
