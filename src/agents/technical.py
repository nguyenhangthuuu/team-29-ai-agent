# src/agents/technical.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class TechnicalSignal(BaseModel):
    ticker: str
    composite_score: float = Field(..., ge=-1.0, le=1.0, description="Normalized score: -1.0 (Strong Sell) to +1.0 (Strong Buy)")
    trend: str = Field(..., description="BULLISH, BEARISH, or NEUTRAL")
    rsi_14: float
    rsi_signal: str
    roc_21d_pct: float
    roc_63d_pct: float
    breakout_state: str
    reasons: List[str]
    summary_reason: str

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates 14-day Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def evaluate_etf_technical(prices: pd.Series, ticker: str, benchmark_prices: Optional[pd.Series] = None) -> TechnicalSignal:
    """
    Evaluates moving average trend, RSI, momentum, and Bollinger breakout for a single ETF.
    Returns a structured TechnicalSignal with score (-1 to +1) and human-readable reasoning.
    """
    clean_series = prices.dropna()
    if len(clean_series) < 65:
        return TechnicalSignal(
            ticker=ticker,
            composite_score=0.0,
            trend="NEUTRAL",
            rsi_14=50.0,
            rsi_signal="INSUFFICIENT_DATA",
            roc_21d_pct=0.0,
            roc_63d_pct=0.0,
            breakout_state="CONSOLIDATING",
            reasons=["Insufficient historical data to compute 60d indicators."],
            summary_reason="Neutral due to insufficient data."
        )

    curr_price = clean_series.iloc[-1]
    reasons = []
    score = 0.0

    # 1. Trend Analysis: EMA 20 vs SMA 50
    ema20 = clean_series.ewm(span=20, adjust=False).mean().iloc[-1]
    sma50 = clean_series.rolling(window=50).mean().iloc[-1]
    
    if ema20 > sma50 * 1.01:
        trend = "BULLISH"
        score += 0.35
        reasons.append(f"Golden Trend: 20-day EMA ({ema20:.2f}) above 50-day SMA ({sma50:.2f})")
    elif ema20 < sma50 * 0.99:
        trend = "BEARISH"
        score -= 0.35
        reasons.append(f"Death Cross: 20-day EMA ({ema20:.2f}) below 50-day SMA ({sma50:.2f})")
    else:
        trend = "NEUTRAL"
        reasons.append("Moving averages consolidating within 1% threshold")

    # 2. RSI (14-day) Mean-Reversion & Momentum Filter
    rsi_series = calc_rsi(clean_series, 14)
    rsi_val = float(rsi_series.iloc[-1])
    
    if rsi_val < 30.0:
        rsi_state = "OVERSOLD"
        score += 0.25
        reasons.append(f"RSI is oversold at {rsi_val:.1f} (potential mean-reversion bounce)")
    elif rsi_val > 70.0:
        rsi_state = "OVERBOUGHT"
        score -= 0.25
        reasons.append(f"RSI is overbought at {rsi_val:.1f} (exhaustion risk)")
    elif 50.0 <= rsi_val <= 65.0:
        rsi_state = "BULLISH_MOMENTUM"
        score += 0.15
        reasons.append(f"Healthy bullish RSI momentum at {rsi_val:.1f}")
    else:
        rsi_state = "NEUTRAL"

    # 3. Momentum: 21-day (1m) & 63-day (3m) Rate of Change
    roc_21 = float((curr_price / clean_series.iloc[-22] - 1.0) * 100)
    roc_63 = float((curr_price / clean_series.iloc[-64] - 1.0) * 100)
    
    if roc_21 > 3.0:
        score += 0.20
        reasons.append(f"Strong 1-month momentum (+{roc_21:.1f}%)")
    elif roc_21 < -3.0:
        score -= 0.20
        reasons.append(f"Negative 1-month momentum ({roc_21:.1f}%)")

    # Relative Momentum vs Benchmark (WORLD)
    if benchmark_prices is not None:
        bmk_clean = benchmark_prices.dropna()
        if len(bmk_clean) >= 22:
            bmk_roc21 = float((bmk_clean.iloc[-1] / bmk_clean.iloc[-22] - 1.0) * 100)
            rel_strength = roc_21 - bmk_roc21
            if rel_strength > 2.0:
                score += 0.10
                reasons.append(f"Outperforming benchmark by +{rel_strength:.1f}% over 21 days")
            elif rel_strength < -2.0:
                score -= 0.10
                reasons.append(f"Underperforming benchmark by {rel_strength:.1f}% over 21 days")

    # 4. Volatility Breakout: 20-day Bollinger Bands
    sma20 = clean_series.rolling(20).mean().iloc[-1]
    std20 = clean_series.rolling(20).std().iloc[-1]
    upper_band = sma20 + 2.0 * std20
    lower_band = sma20 - 2.0 * std20

    if curr_price > upper_band:
        breakout = "UPPER_BREAKOUT"
        score += 0.10
        reasons.append(f"Upper Bollinger breakout: price ({curr_price:.2f}) > upper band ({upper_band:.2f})")
    elif curr_price < lower_band:
        breakout = "LOWER_BREAKOUT"
        score -= 0.10
        reasons.append(f"Lower Bollinger breakdown: price ({curr_price:.2f}) < lower band ({lower_band:.2f})")
    else:
        breakout = "CONSOLIDATING"

    final_score = float(np.clip(score, -1.0, 1.0))
    summary_reason = " | ".join(reasons)

    return TechnicalSignal(
        ticker=ticker,
        composite_score=round(final_score, 2),
        trend=trend,
        rsi_14=round(rsi_val, 1),
        rsi_signal=rsi_state,
        roc_21d_pct=round(roc_21, 2),
        roc_63d_pct=round(roc_63, 2),
        breakout_state=breakout,
        reasons=reasons,
        summary_reason=summary_reason
    )

def evaluate_all_etfs(prices_df: pd.DataFrame, benchmark_ticker: str = "WORLD") -> Dict[str, TechnicalSignal]:
    """Evaluates all ETFs in the price dataframe and returns a dictionary of signals."""
    benchmark = prices_df[benchmark_ticker] if benchmark_ticker in prices_df.columns else None
    results = {}
    for col in prices_df.columns:
        results[col] = evaluate_etf_technical(prices_df[col], ticker=col, benchmark_prices=benchmark)
    return results
