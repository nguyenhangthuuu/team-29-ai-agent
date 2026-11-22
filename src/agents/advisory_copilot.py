# src/agents/advisory_copilot.py
"""
Advisory Copilot Agent for Mode Advise.
Generates concise, human-readable natural language AI Mini-Theses for any of the 50 ETFs
combining Technical Machine Learning signals with Macro context (macro news sentiment weight ~15%).
Strictly avoids jargon dumps and presents only vital institutional figures.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.pipeline.etf_universe import ETF_UNIVERSE, ETFMeta
from src.agents.macro import MacroBriefing


ETF_META_MAP = {e.ticker: e for e in ETF_UNIVERSE}


def generate_etf_advise_thesis(
    ticker: str,
    tech_sig: Any,
    macro_briefing: MacroBriefing,
    close_price: float,
    as_of_date: str,
    action: str = "BUY"
) -> Dict[str, Any]:
    """
    Synthesizes a boardroom-grade, concise 2-3 sentence AI Mini-Thesis in English
    for any selected ETF in Advise mode, tailored to BUY or SELL actions.
    """
    meta = ETF_META_MAP.get(ticker)
    full_name = meta.full_name if meta else ticker
    theme = meta.theme if meta else "Broad Market"
    isin = meta.isin if meta else "N/A"

    # Extract technical parameters
    if isinstance(tech_sig, dict):
        rf_prob = float(tech_sig.get("rf_probability", 0.50))
        composite_score = float(tech_sig.get("composite_score", 0.0))
        trend = str(tech_sig.get("trend", "NEUTRAL"))
        rsi_14 = float(tech_sig.get("rsi_14", 50.0))
        roc_21 = float(tech_sig.get("roc_21d_pct", 0.0))
        roc_63 = float(tech_sig.get("roc_63d_pct", 0.0))
    elif hasattr(tech_sig, "rf_probability"):
        rf_prob = float(tech_sig.rf_probability)
        composite_score = float(tech_sig.composite_score)
        trend = str(tech_sig.trend)
        rsi_14 = float(tech_sig.rsi_14)
        roc_21 = float(tech_sig.roc_21d_pct)
        roc_63 = float(tech_sig.roc_63d_pct)
    else:
        rf_prob = 0.50
        composite_score = 0.0
        trend = "NEUTRAL"
        rsi_14 = 50.0
        roc_21 = 0.0
        roc_63 = 0.0

    # Macro thematic tilt
    macro_tilt = macro_briefing.theme_macro_tilts.get(theme, 0.0)
    regime = macro_briefing.macro_regime
    clean_action = str(action).upper() if action else "BUY"

    # In Advise mode: Technical signal carries 85% weight, Macro News carries 15%
    advise_conviction = round(float(np.clip(0.85 * composite_score + 0.15 * macro_tilt, -1.0, 1.0)), 3)

    # Formulate Natural Language Mini-Thesis in institutional English
    if clean_action == "SELL":
        if rsi_14 > 65:
            tech_sentence = f"Overbought technical profile (14-day RSI {rsi_14:.1f}, 1-month return {roc_21:+.1f}%) suggests favorable conditions for profit-taking."
        elif trend == "BEARISH":
            tech_sentence = f"Bearish momentum confirmed by ML model (Underperform Probability {(1-rf_prob)*100:.1f}%, 14-day RSI {rsi_14:.1f}); liquidating curbs downside drift."
        else:
            tech_sentence = f"Neutral technical setup (RSI {rsi_14:.1f}); capital release optimizes liquidity into higher-conviction opportunities."

        if macro_tilt < -0.05:
            macro_sentence = f"Thematic macro headwinds in {theme} support tactical de-risking under {regime} policy conditions."
        else:
            macro_sentence = f"Macro influence remains balanced, with exit driven primarily by portfolio rebalancing requirements."

        risk_note = "Position reduction improves risk-adjusted cash buffer."
    else:
        # BUY thesis
        if trend == "BULLISH":
            tech_sentence = f"Bullish momentum intact (Random Forest Outperform Probability {rf_prob*100:.1f}%, 14-day RSI {rsi_14:.1f})."
        elif trend == "BEARISH":
            tech_sentence = f"Technical consolidation underway (ML Outperform Probability {rf_prob*100:.1f}%, 14-day RSI at {rsi_14:.1f})."
        else:
            tech_sentence = f"Balanced technical profile with 14-day RSI at {rsi_14:.1f} and 21-day momentum at {roc_21:+.1f}%."

        if macro_tilt > 0.08:
            macro_sentence = f"Strong structural tailwinds for {theme} supported by central bank stance during {regime}."
        elif macro_tilt < -0.08:
            macro_sentence = f"Monitored sector dispersion in {theme}; position sizing accounts for prevailing {regime} volatility."
        else:
            macro_sentence = f"Macro stance is neutral, with price action governed predominantly by underlying supply and demand."

        if rsi_14 > 68:
            risk_note = "RSI approaches upper threshold; controlled sizing mitigates near-term retracement risk."
        elif rsi_14 < 32:
            risk_note = "Oversold territory (RSI < 32) offers attractive accumulation value as downward pressure stabilizes."
        else:
            risk_note = "Volatility profile remains within baseline risk parameters."

    executive_mini_thesis = f"{tech_sentence} {macro_sentence} {risk_note}"

    return {
        "ticker": ticker,
        "full_name": full_name,
        "isin": isin,
        "theme": theme,
        "action": clean_action,
        "as_of_date": as_of_date,
        "prev_close_price": round(close_price, 4),
        "advise_conviction": advise_conviction,
        "rf_probability": round(rf_prob, 3),
        "trend_label": trend,
        "rsi_14": round(rsi_14, 1),
        "roc_21d_pct": round(roc_21, 2),
        "roc_63d_pct": round(roc_63, 2),
        "macro_tilt": round(macro_tilt, 2),
        "executive_mini_thesis": executive_mini_thesis
    }
