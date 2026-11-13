# src/agents/technical.py
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

class TechnicalSignal(BaseModel):
    ticker: str
    score: float = Field(..., ge=-1.0, le=1.0, description="Normalized score: -1 (bearish) to +1 (bullish)")
    trend: str
    rsi: float
    momentum_21d: float
    breakout_state: str
    reason: str

def evaluate_technical_signals(prices: pd.Series, ticker: str) -> TechnicalSignal:
    # 1. Moving Averages (20 EMA vs 50 SMA)
    ema20 = prices.ewm(span=20, adjust=False).mean()
    sma50 = prices.rolling(window=50).mean()
    ma_trend = "BULLISH" if ema20.iloc[-1] > sma50.iloc[-1] else "BEARISH"
    
    # 2. RSI (14-day)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    # 3. Momentum (21-day Rate of Change)
    roc21 = (prices.iloc[-1] / prices.iloc[-22] - 1.0)
    
    # 4. Volatility Breakout (Bollinger Bands, 20-day, 2 std)
    sma20 = prices.rolling(20).mean().iloc[-1]
    std20 = prices.rolling(20).std().iloc[-1]
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    curr_price = prices.iloc[-1]
    
    if curr_price > upper_band:
        breakout = "UPPER_BREAKOUT"
    elif curr_price < lower_band:
        breakout = "LOWER_BREAKOUT"
    else:
        breakout = "CONSOLIDATING"

    # Composite Scoring Logic
    score = 0.0
    reasons = []
    
    if ma_trend == "BULLISH":
        score += 0.35
        reasons.append("EMA20 above SMA50")
    else:
        score -= 0.35
        reasons.append("EMA20 below SMA50")
        
    if rsi < 30:
        score += 0.35
        reasons.append(f"Oversold bounce potential (RSI: {rsi:.1f})")
    elif rsi > 70:
        score -= 0.35
        reasons.append(f"Overbought exhaustion risk (RSI: {rsi:.1f})")
        
    if roc21 > 0.03:
        score += 0.30
        reasons.append(f"Strong 21d momentum (+{roc21*100:.1f}%)")
    elif roc21 < -0.03:
        score -= 0.30
        reasons.append(f"Negative 21d momentum ({roc21*100:.1f}%)")

    score = round(max(-1.0, min(1.0, score)), 2)
    return TechnicalSignal(
        ticker=ticker,
        score=score,
        trend=ma_trend,
        rsi=round(rsi, 2),
        momentum_21d=round(roc21, 4),
        breakout_state=breakout,
        reason="; ".join(reasons)
    )