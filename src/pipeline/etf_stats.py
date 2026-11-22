# src/pipeline/etf_stats.py
"""
ETF Universe Institutional Statistics Hub.
Calculates 6 standardized financial performance and risk metrics across all 50 UCITS ETFs
strictly anchored to Day T-1 Close (as_of_date) with an official 0:00 CET timestamp.
Includes tooltip definitions for interactive newbie guidance.
"""

import os
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from src.pipeline.etf_universe import ETF_UNIVERSE, ETFMeta

METRIC_TOOLTIPS = {
    "return_pa": {
        "title": "Return % p.a. (CAGR)",
        "definition": "Tỷ suất sinh lời kép hàng năm (Compound Annual Growth Rate) tính trên chuỗi giá 252 phiên giao dịch gần nhất, phản ánh tốc độ tăng trưởng vốn chuẩn hóa mỗi năm."
    },
    "price_trend_30d": {
        "title": "30-Day Price Trend & Change",
        "definition": "Biểu đồ xu hướng giá thu nhỏ (Sparkline) và tỷ lệ % tăng/giảm trong 30 phiên giao dịch gần nhất, cho biết hướng đi ngắn hạn của dòng tiền."
    },
    "volatility_pa": {
        "title": "Volatility % p.a. (Biến động năm hóa)",
        "definition": "Độ lệch chuẩn lợi suất hàng ngày nhân căn bậc hai 252 phiên (Annualized Volatility). Đo lường biên độ rủi ro dao động giá; tỷ lệ càng cao tài sản càng biến động mạnh."
    },
    "max_drawdown": {
        "title": "Max Drawdown % (Mức sụt giảm tối đa)",
        "definition": "Phần trăm sụt giảm lớn nhất từ đỉnh cao nhất xuống đáy sâu nhất trong vòng 252 phiên gần nhất, thể hiện mức lỗ nặng nề nhất mà nhà đầu tư có thể trải qua."
    },
    "sharpe_ratio": {
        "title": "Sharpe Ratio (Rf = 2.5%)",
        "definition": "Tỷ số đo lường hiệu quả sinh lời trên một đơn vị rủi ro so với lãi suất phi rủi ro Châu Âu (2.5%). Sharpe > 1.0 là tốt, > 1.5 là xuất sắc, < 0 là kém hiệu quả."
    },
    "momentum_6m": {
        "title": "6-Month Momentum (Động lượng 6 tháng)",
        "definition": "Tỷ suất sinh lời giá trong 126 phiên giao dịch gần nhất (~6 tháng), phản ánh sức mạnh xu hướng trung hạn của tài sản trên thị trường."
    }
}


def calculate_etf_universe_stats(
    prices_df: pd.DataFrame,
    as_of_date: str
) -> Dict[str, Any]:
    """
    Computes 6 core institutional metrics for all 50 ETFs as of as_of_date Close.
    Strictly zero lookahead: only data on or before as_of_date is used.
    """
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df.index = pd.to_datetime(prices_df.index)

    date_slice = prices_df.loc[:as_of_date]
    if date_slice.empty:
        return {
            "as_of_date": as_of_date,
            "last_updated_cet": f"{as_of_date} 00:00 CET",
            "metrics": [],
            "tooltips": METRIC_TOOLTIPS
        }

    meta_map = {e.ticker: e for e in ETF_UNIVERSE}
    stats_list = []

    for ticker in prices_df.columns:
        meta = meta_map.get(ticker)
        col_series = date_slice[ticker].dropna()
        if col_series.empty or len(col_series) < 10:
            continue

        latest_price = float(col_series.iloc[-1])
        n_days = len(col_series)

        # 1. 30-Day Trend & Price Change
        recent_30 = col_series.tail(30)
        p_history = [
            {"date": str(d.date()), "price": round(float(p), 4)}
            for d, p in recent_30.items()
        ]
        if len(recent_30) >= 2:
            change_30d = round(((recent_30.iloc[-1] - recent_30.iloc[0]) / recent_30.iloc[0]) * 100, 2)
        else:
            change_30d = 0.0

        # 2. Return % p.a. (CAGR over min(252, n_days))
        eval_window = min(252, n_days)
        window_series = col_series.tail(eval_window)
        p_start = float(window_series.iloc[0])
        p_end = float(window_series.iloc[-1])
        
        if p_start > 0 and eval_window > 20:
            cagr = ((p_end / p_start) ** (252.0 / eval_window)) - 1.0
            return_pa_pct = round(cagr * 100, 2)
        else:
            return_pa_pct = 0.0

        # 3. Volatility % p.a.
        log_rets = np.log(col_series / col_series.shift(1)).dropna()
        recent_vol_series = log_rets.tail(min(252, len(log_rets)))
        if len(recent_vol_series) >= 20:
            ann_vol = float(recent_vol_series.std() * np.sqrt(252) * 100)
            volatility_pa_pct = round(ann_vol, 2)
        else:
            volatility_pa_pct = 15.0

        # 4. Max Drawdown % (over 252 sessions)
        cummax = window_series.cummax()
        drawdowns = (window_series / cummax) - 1.0
        max_dd_pct = round(float(drawdowns.min() * 100), 2)

        # 5. Sharpe Ratio (Rf = 2.5%)
        rf_rate = 0.025
        if volatility_pa_pct > 0.5:
            sharpe = ( (return_pa_pct / 100.0) - rf_rate ) / (volatility_pa_pct / 100.0)
            sharpe_ratio = round(float(sharpe), 2)
        else:
            sharpe_ratio = 0.0

        # 6. 6-Month Momentum (ROC over 126 sessions)
        mom_window = min(126, n_days)
        if mom_window >= 20:
            p_mom_start = float(col_series.iloc[-mom_window])
            momentum_6m_pct = round(((latest_price - p_mom_start) / max(0.0001, p_mom_start)) * 100, 2)
        else:
            momentum_6m_pct = change_30d

        stats_list.append({
            "ticker": ticker,
            "full_name": meta.full_name if meta else ticker,
            "isin": meta.isin if meta else "N/A",
            "theme": meta.theme if meta else "General",
            "is_leveraged": bool(meta.is_leveraged_or_short) if meta else False,
            "latest_close_price": round(latest_price, 4),
            "return_pa_pct": return_pa_pct,
            "price_history_30d": p_history,
            "change_30d_pct": change_30d,
            "volatility_pa_pct": volatility_pa_pct,
            "max_drawdown_pct": max_dd_pct,
            "sharpe_ratio": sharpe_ratio,
            "momentum_6m_pct": momentum_6m_pct
        })

    # Sort descending by 6-month momentum
    stats_list.sort(key=lambda x: x["momentum_6m_pct"], reverse=True)

    return {
        "as_of_date": as_of_date,
        "last_updated_cet": f"{as_of_date} 00:00 CET",
        "total_etfs": len(stats_list),
        "tooltips": METRIC_TOOLTIPS,
        "statistics": stats_list
    }
