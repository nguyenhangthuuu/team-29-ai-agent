# src/agents/advisory_copilot.py
"""
Advisory Copilot Agent for Mode Advise.
Generates concise, human-readable natural language AI Mini-Theses for any of the 50 ETFs
combining Technical Machine Learning signals with Macro context (lowering news sentiment weight to ~15%).
Strictly avoids jargon dumps and presents only vital institutional figures.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.pipeline.etf_universe import ETF_UNIVERSE, ETFMeta
from src.agents.macro import MacroBriefing
from src.agents.technical import TechnicalSignal


ETF_META_MAP = {e.ticker: e for e in ETF_UNIVERSE}


def generate_etf_advise_thesis(
    ticker: str,
    tech_sig: Any,
    macro_briefing: MacroBriefing,
    close_price: float,
    as_of_date: str
) -> Dict[str, Any]:
    """
    Synthesizes a boardroom-grade, concise 2-3 sentence AI Mini-Thesis in natural language
    for any selected ETF in Advise mode.
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

    # In Advise mode: Technical signal carries 85% weight, Macro News carries only 15%
    advise_conviction = round(float(np.clip(0.85 * composite_score + 0.15 * macro_tilt, -1.0, 1.0)), 3)

    # Formulate Natural Language Mini-Thesis
    tech_sentence = ""
    if trend == "BULLISH":
        tech_sentence = f"Xu hướng kỹ thuật tăng trưởng tích cực (Xác suất ML Outperform {rf_prob*100:.1f}%, RSI 14 phiên ở mức {rsi_14:.1f})."
    elif trend == "BEARISH":
        tech_sentence = f"Động lượng kỹ thuật đang chịu áp lực điều chỉnh (Xác suất ML {rf_prob*100:.1f}%, RSI lùi về {rsi_14:.1f})."
    else:
        tech_sentence = f"Trạng thái kỹ thuật đang trong vùng tích lũy cân bằng (RSI {rsi_14:.1f}, đà tăng 1 tháng đạt {roc_21:+.1f}%)."

    macro_sentence = ""
    if macro_tilt > 0.08:
        macro_sentence = f"Hưởng lợi từ làn sóng vĩ mô ngành {theme} và định hướng chính sách từ NHTW trong bối cảnh {regime}."
    elif macro_tilt < -0.08:
        macro_sentence = f"Cần lưu ý rủi ro phân hóa ngành khi vĩ mô {regime} đang tạo sức ép biên lợi nhuận lên nhóm {theme}."
    else:
        macro_sentence = f"Tác động vĩ mô ở mức trung tính, diễn biến giá chủ yếu dẫn dắt bởi cung cầu nội tại của nhóm tài sản."

    risk_note = ""
    if rsi_14 > 68:
        risk_note = "Cảnh báo vùng quá mua ngắn hạn (RSI > 68); cân nhắc phân bổ tỷ trọng vừa phải để kiểm soát biến động."
    elif rsi_14 < 32:
        risk_note = "Tài sản đang ở vùng quá bán sâu (RSI < 32); mở ra cơ hội tích lũy tạo đáy nếu đà bán chững lại."
    else:
        risk_note = f"Biên độ dao động ổn định; phù hợp cho phân bổ chiến lược trong phiên."

    executive_mini_thesis = f"{tech_sentence} {macro_sentence} {risk_note}"

    return {
        "ticker": ticker,
        "full_name": full_name,
        "isin": isin,
        "theme": theme,
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
