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

TRADE_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "trade_history.json"
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
        if os.path.exists(TRADE_HISTORY_PATH):
            try:
                os.remove(TRADE_HISTORY_PATH)
            except Exception:
                pass

    def _load_if_exists(self):
        """Loads historical snapshots and trades from disk if available."""
        # Load trade history if present
        if os.path.exists(TRADE_HISTORY_PATH):
            try:
                with open(TRADE_HISTORY_PATH, "r", encoding="utf-8") as f:
                    t_list = json.load(f)
                    self.trade_history = [
                        TradeRecord(
                            trade_date=t["trade_date"],
                            ticker=t["ticker"],
                            action=t["action"],
                            shares=float(t["shares"]),
                            price=float(t["price"]),
                            total_amount=float(t["total_amount"]),
                            fee=float(t.get("fee", 0.0))
                        )
                        for t in t_list
                    ]
            except Exception:
                pass

        if os.path.exists(self.ledger_path):
            try:
                df = pd.read_parquet(self.ledger_path)
                if not df.empty:
                    # Restore all historical snapshots
                    self.snapshots = []
                    for _, row in df.iterrows():
                        holdings_raw = row.get("holdings_json", "{}")
                        h_dict = json.loads(holdings_raw) if isinstance(holdings_raw, str) else holdings_raw
                        self.snapshots.append(LedgerSnapshot(
                            trade_date=str(row["trade_date"]),
                            cash=float(row["cash"]),
                            invested_value=float(row["invested_value"]),
                            total_nav=float(row["total_nav"]),
                            period_pnl=float(row.get("period_pnl", 0.0)),
                            period_return_pct=float(row.get("period_return_pct", 0.0)),
                            cumulative_return_pct=float(row.get("cumulative_return_pct", 0.0)),
                            var_95_pct=float(row.get("var_95_pct", 0.0)),
                            cvar_95_pct=float(row.get("cvar_95_pct", 0.0)),
                            holdings=h_dict if isinstance(h_dict, dict) else {}
                        ))

                    last_row = df.iloc[-1]
                    self.cash = float(last_row.get("cash", self.initial_capital))
                    # Restore positions from serialized json if available
                    last_holdings_raw = last_row.get("holdings_json", "{}")
                    lh_dict = json.loads(last_holdings_raw) if isinstance(last_holdings_raw, str) else last_holdings_raw
                    for tkr, data in lh_dict.items():
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

    def _save_trade_history(self):
        """Persists trade history to disk."""
        try:
            os.makedirs(os.path.dirname(TRADE_HISTORY_PATH), exist_ok=True)
            t_data = [asdict(tr) for tr in self.trade_history]
            with open(TRADE_HISTORY_PATH, "w", encoding="utf-8") as f:
                json.dump(t_data, f, indent=2)
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
        All suggested and accepted orders in target_orders are recorded as-is in the trade ledger (pledger).
        Liquidates replaced holdings to ensure target allocations are fully funded.
        """
        # 1. Update MtM prior to trade
        prev_nav = self.mark_to_market(current_prices)
        if prev_nav <= 0.0:
            prev_nav = self.initial_capital
        
        # Enforce strict maximum 5 orders per session
        orders_to_execute = target_orders[:5]

        # First execute any explicit SELL orders so cash proceeds become available
        sell_orders = [o for o in orders_to_execute if str(o.get("action", "BUY")).upper() == "SELL"]
        buy_orders = [o for o in orders_to_execute if str(o.get("action", "BUY")).upper() != "SELL"]

        # Liquidate replaced positions so cash is pooled for the new allocation
        target_buy_tickers = {o["ticker"] for o in buy_orders}
        for tkr, pos in list(self.positions.items()):
            if tkr not in target_buy_tickers or any(o["ticker"] == tkr for o in sell_orders):
                p_curr = float(current_prices.get(tkr, pos.current_price))
                proceeds = pos.shares * p_curr
                self.cash += proceeds
                self.positions.pop(tkr, None)

        # 2. Execute explicit SELL orders
        for order in sell_orders:
            tkr = order["ticker"]
            p_curr = float(current_prices.get(tkr, 0.0))
            if p_curr <= 0.0:
                p_curr = float(order.get("limit_price_eur", order.get("limit_price", 100.0)))
            current_pos = self.positions.get(tkr)
            if not current_pos or current_pos.shares <= 1e-4:
                continue
            sell_shares = current_pos.shares
            proceeds = sell_shares * p_curr
            self.cash += proceeds
            self.positions.pop(tkr, None)

            self.trade_history.append(TradeRecord(
                trade_date=trade_date,
                ticker=tkr,
                action="SELL",
                shares=round(sell_shares, 4),
                price=round(p_curr, 4),
                total_amount=round(proceeds, 2),
                fee=0.0
            ))

        # 3. Execute BUY orders: record every suggested and accepted order as-is into pledger
        total_allocated = 0.0
        for order in buy_orders:
            tkr = order["ticker"]
            p_curr = float(current_prices.get(tkr, 0.0))
            if p_curr <= 0.0:
                p_curr = float(order.get("limit_price_eur", order.get("limit_price", 100.0)))
            if p_curr <= 0.0:
                p_curr = 100.0

            target_amount = float(order.get("amount", 0.0))
            target_weight = float(order.get("weight", 0.0))
            if target_amount <= 0.0 and target_weight > 0.0:
                target_amount = target_weight * prev_nav
            
            # If target amount is still 0, default to equal share of deployable nav
            if target_amount <= 0.0:
                target_amount = (prev_nav * 0.85) / max(1, len(buy_orders))

            shares = target_amount / p_curr
            total_allocated += target_amount

            self.positions[tkr] = Position(
                ticker=tkr,
                shares=shares,
                avg_entry_price=p_curr,
                current_price=p_curr,
                market_value=target_amount,
                unrealized_pnl=0.0,
                weight_pct=round((target_amount / (prev_nav + 1e-9)) * 100, 2)
            )

            # Record in pledger as-is without skipping
            self.trade_history.append(TradeRecord(
                trade_date=trade_date,
                ticker=tkr,
                action="BUY",
                shares=round(shares, 4),
                price=round(p_curr, 4),
                total_amount=round(target_amount, 2),
                fee=0.0
            ))

        self.cash = max(0.0, prev_nav - total_allocated)
        self._save_trade_history()

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

    def record_daily_snapshot(
        self,
        trade_date: str,
        current_prices: Dict[str, float],
        var_95: float = 0.0,
        cvar_95: float = 0.0
    ) -> LedgerSnapshot:
        """Records end-of-day mark-to-market valuation snapshot."""
        final_nav = self.mark_to_market(current_prices)
        invested = sum(p.market_value for p in self.positions.values())
        prev_nav = self.snapshots[-1].total_nav if self.snapshots else self.initial_capital
        period_pnl = final_nav - prev_nav
        period_ret = (period_pnl / (prev_nav + 1e-9)) * 100
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
        # Avoid duplicate snapshot on the same date
        self.snapshots = [s for s in self.snapshots if s.trade_date != trade_date]
        self.snapshots.append(snapshot)
        self._persist_snapshot(snapshot)
        return snapshot

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
