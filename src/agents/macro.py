# src/agents/macro.py
"""
Macro & News Agent for QDC 2026/2027 (ReBel x QuantumBlack).
Quantifies unstructured central bank texts (Federal Reserve & ECB) using:
  1. Central Bank Hawkish-Dovish Dictionaries (Apel & Blix Grimaldi, 2014; Bennani & Farvaque)
  2. Loughran-McDonald (LM, 2011) Financial Sentiment & Uncertainty Lexicon
  3. Real-time Market Indicators (Yields, VIX, DXY, Oil, Gold) via yfinance
Supports Google Gemini 1.5 Flash (Free Tier), OpenAI, and 100% Deterministic Fallback.
"""

import os
import re
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# =====================================================================
# 1. ACADEMIC TEXT MINING LEXICONS (APEL-GRIMALDI & LOUGHRAN-MCDONALD)
# =====================================================================

# Central Bank Hawkish Terms (Apel & Blix Grimaldi 2014; Bennani & Farvaque)
# Signs of monetary tightening, inflation persistence, overheating, rate hikes
HAWKISH_TERMS = {
    "tighten", "tightening", "restrictive", "overheating", "inflationary", 
    "accelerate", "acceleration", "hike", "hikes", "hiking", "upside", 
    "pressures", "firming", "unanchored", "robust", "vigilant", "persistence", 
    "elevated", "capacity constraint", "wage growth", "excess demand"
}

# Central Bank Dovish Terms (Easing, disinflation, slack, rate cuts)
DOVISH_TERMS = {
    "ease", "easing", "accommodative", "cut", "cuts", "cutting", "downside", 
    "disinflation", "cooling", "slowdown", "subdued", "slack", "softening", 
    "fragile", "headwinds", "weakness", "deteriorate", "recession", 
    "underutilization", "tame", "moderate"
}

# Loughran-McDonald (2011) Financial Sentiment & Uncertainty Lexicon
LM_UNCERTAINTY = {
    "uncertain", "uncertainty", "unclear", "ambiguous", "volatile", "volatility", 
    "unpredictable", "risk", "risks", "contingent", "indefinite", "hesitant"
}
LM_POSITIVE = {
    "resilient", "resilience", "strong", "favorable", "improvement", "improving", 
    "stability", "confidence", "momentum", "gain", "gains", "progress"
}
LM_NEGATIVE = {
    "deficit", "decline", "deterioration", "loss", "losses", "default", 
    "distress", "adverse", "recessionary", "vulnerability", "shock", "shocks"
}

NEGATION_WORDS = {"not", "no", "never", "hardly", "scarcely", "without", "little"}

def analyze_central_bank_text(text: str) -> Dict[str, float]:
    """
    Parses unstructured text and computes literature-grounded indices with 3-word negation handling.
    """
    words = re.findall(r"\b[a-z\-]+\b", text.lower())
    total_words = len(words)
    if total_words == 0:
        return {"hawkish_dovish_index": 0.0, "lm_sentiment": 0.0, "lm_uncertainty": 0.0}

    hawk_count = 0
    dove_count = 0
    uncert_count = 0
    pos_count = 0
    neg_count = 0

    for i, w in enumerate(words):
        # Lookback 3 tokens for negation window
        start = max(0, i - 3)
        is_negated = any(nw in words[start:i] for nw in NEGATION_WORDS)

        if w in HAWKISH_TERMS:
            if is_negated:
                dove_count += 1
            else:
                hawk_count += 1
        elif w in DOVISH_TERMS:
            if is_negated:
                hawk_count += 1
            else:
                dove_count += 1

        if w in LM_UNCERTAINTY:
            uncert_count += 1
        elif w in LM_POSITIVE:
            if is_negated:
                neg_count += 1
            else:
                pos_count += 1
        elif w in LM_NEGATIVE:
            if is_negated:
                pos_count += 1
            else:
                neg_count += 1

    # Hawkish-Dovish Index normalized between -1.0 (Dovish) and +1.0 (Hawkish)
    # Ref: Bennani & Farvaque formula
    hd_denominator = (hawk_count + dove_count) + 1e-6
    hd_index = (hawk_count - dove_count) / hd_denominator

    # Loughran-McDonald Sentiment Score (-1.0 to +1.0)
    lm_sent_denom = (pos_count + neg_count) + 1e-6
    lm_sentiment = (pos_count - neg_count) / lm_sent_denom

    # Uncertainty Ratio (% of total tokens)
    uncert_ratio = (uncert_count / total_words) * 100

    return {
        "hawkish_dovish_index": round(float(np.clip(hd_index, -1.0, 1.0)), 2),
        "lm_sentiment": round(float(np.clip(lm_sentiment, -1.0, 1.0)), 2),
        "lm_uncertainty_pct": round(float(uncert_ratio), 2),
        "hawk_words": hawk_count,
        "dove_words": dove_count,
        "total_tokens_analyzed": total_words
    }

# =====================================================================
# 2. AUTOMATED CENTRAL BANK RSS & REPORT FETCHER (AUTO-UPDATE)
# =====================================================================

CACHE_FILE = Path("data/central_bank_cache.json")

def fetch_live_fed_communication() -> Dict[str, str]:
    """
    Automatically checks the Federal Reserve RSS feed for new releases.
    If a new FOMC statement, minutes, or report is published, it downloads,
    extracts article text, and updates local cache. Falls back gracefully if offline.
    """
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Baseline fallback text in case network is down
    fallback_data = {
        "source": "Federal Reserve Semi-Annual Monetary Policy Baseline",
        "pub_date": "2026-11-18",
        "title": "Monetary Policy Report & FOMC Baseline",
        "text": (
            "The Federal Reserve remains committed to returning inflation to its 2 percent objective. "
            "Labor market conditions have cooled from their previously overheated state but remain solid. "
            "Inflation has eased substantially over the past year though it remains somewhat elevated. "
            "Economic activity has continued to expand at a solid pace. While progress is evident, "
            "the Committee considers risks to achieving its employment and inflation goals as moving into better balance. "
            "The Committee decided to maintain the target range for the federal funds rate in restrictive territory "
            "until it has gained greater confidence that inflation is moving sustainably toward 2 percent."
        )
    }

    rss_url = "https://www.federalreserve.gov/feeds/press_monetary.xml"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        req = urllib.request.Request(rss_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            xml_content = resp.read()

        root = ET.fromstring(xml_content)
        items = root.findall(".//item")
        
        if items:
            combined_texts = []
            scraped_titles = []
            latest_pub_date = items[0].find("pubDate").text if items[0].find("pubDate") is not None else ""
            latest_link = items[0].find("link").text if items[0].find("link") is not None else ""

            # Check cache: if link and date match, reuse cache
            if CACHE_FILE.exists():
                try:
                    with open(CACHE_FILE, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if cached.get("link") == latest_link and len(cached.get("text", "")) > 500:
                        return cached
                except Exception:
                    pass

            # Scan the top 6 entries for official statements, minutes, and reports
            for item in items[:6]:
                title = item.find("title").text if item.find("title") is not None else ""
                link = item.find("link").text if item.find("link") is not None else ""
                title_lower = title.lower()
                
                if any(k in title_lower for k in ["statement", "minutes", "monetary policy report", "economic"]):
                    try:
                        art_req = urllib.request.Request(link, headers=headers)
                        with urllib.request.urlopen(art_req, timeout=5) as art_resp:
                            soup = BeautifulSoup(art_resp.read(), "html.parser")
                            art_div = soup.find("div", id="article") or soup.find("div", class_="col-xs-12")
                            if art_div:
                                txt = art_div.get_text(separator=" ", strip=True)
                                combined_texts.append(f"=== {title} ===\n{txt}")
                                scraped_titles.append(title)
                    except Exception:
                        continue

            if combined_texts:
                full_scraped_text = "\n\n".join(combined_texts)
                fresh_data = {
                    "source": "Federal Reserve Official Live RSS",
                    "title": "; ".join(scraped_titles[:2]),
                    "pub_date": latest_pub_date,
                    "link": latest_link,
                    "text": full_scraped_text
                }
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(fresh_data, f, indent=2)
                print(f"📥 Automatically updated Central Bank feed: {len(combined_texts)} documents parsed [{latest_pub_date}]")
                return fresh_data

    except Exception as e:
        # Graceful fallback: load existing cache or baseline
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    return fallback_data

# =====================================================================
# 3. REAL-TIME MACRO INDICATORS (ZERO-COST INGESTION)
# =====================================================================

def fetch_market_macro_indicators() -> Dict[str, float]:
    """Fetches yield curve, VIX, Oil, Gold, and Dollar Index with zero API cost."""
    symbols = {
        "VIX": "^VIX",
        "US10Y": "^TNX",
        "DXY": "DX-Y.NYB",
        "BRENT": "BZ=F",
        "GOLD": "GC=F",
        "EURUSD": "EURUSD=X"
    }
    raw = yf.download(list(symbols.values()), period="5d")
    close = raw["Close"] if "Close" in raw else raw
    
    indicators = {}
    for name, sym in symbols.items():
        if sym in close.columns:
            s = close[sym].dropna()
            if not s.empty:
                indicators[name] = round(float(s.iloc[-1]), 2)
            else:
                indicators[name] = 0.0
    return indicators

# =====================================================================
# 4. STRUCTURED PYDANTIC OUTPUT SCHEMA
# =====================================================================

class MacroBriefing(BaseModel):
    trading_date: str
    macro_regime: str = Field(..., description="RISK_ON, RISK_OFF, STAGFLATION, or NEUTRAL")
    market_sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    fed_hawkish_dovish_score: float = Field(..., ge=-1.0, le=1.0, description="Apel-Grimaldi Index: -1.0 (Dovish) to +1.0 (Hawkish)")
    fed_uncertainty_score: float = Field(..., description="Loughran-McDonald Uncertainty %")
    theme_macro_tilts: Dict[str, float] = Field(..., description="Granular bias (-1.0 to +1.0) for each of the ETF themes")
    tail_risk_alerts: List[str]
    executive_summary: str

# =====================================================================
# 5. MULTI-ENGINE SYNTHESIZER (GEMINI FREE / OPENAI / DETERMINISTIC)
# =====================================================================

def evaluate_macro_agent(trading_date: str = "2026-11-18") -> MacroBriefing:
    """
    Main entry point for the Macro & News Agent.
    Prioritizes Gemini 1.5 Flash (Free Tier) -> OpenAI -> Deterministic Quant Fallback.
    """
    # 1. Quantify Unstructured Fed Text (Auto-updates via Live RSS)
    fed_docs = fetch_live_fed_communication()
    text_metrics = analyze_central_bank_text(fed_docs["text"])
    
    # 2. Ingest Quantitative Market Indicators
    market_indicators = fetch_market_macro_indicators()
    vix = market_indicators.get("VIX", 16.0)
    us10y = market_indicators.get("US10Y", 4.30)
    brent = market_indicators.get("BRENT", 75.0)
    dxy = market_indicators.get("DXY", 104.0)
    eurusd = market_indicators.get("EURUSD", 1.08)

    hd_score = text_metrics["hawkish_dovish_index"]
    lm_sent = text_metrics["lm_sentiment"]
    lm_uncert = text_metrics["lm_uncertainty_pct"]

    # 3. Check Available LLM Backends (Gemini Free Tier vs OpenAI vs Offline)
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    prompt_context = f"""
    You are the Chief Macro Strategist for ReBel ETF Portfolio.
    Trading Date: {trading_date}

    A. QUANTIFIED CENTRAL BANK TEXT (Apel & Blix Grimaldi / Loughran-McDonald):
    - Fed Hawkish-Dovish Index: {hd_score:+.2f} (-1.0 Dovish / Cuts to +1.0 Hawkish / Restrictive)
    - Loughran-McDonald Financial Sentiment: {lm_sent:+.2f}
    - Loughran-McDonald Uncertainty Ratio: {lm_uncert:.2f}%
    - Fed Communication Source: Semi-Annual Monetary Policy Report & FOMC Statements

    B. QUANTITATIVE MARKET INDICATORS:
    - VIX Volatility Index: {vix:.2f} (Safe <18, Elevated >22)
    - US 10-Year Treasury Yield: {us10y:.2f}%
    - Brent Crude Oil: ${brent:.2f}/bbl
    - US Dollar Index (DXY): {dxy:.2f} | EUR/USD: {eurusd:.4f}

    TASK:
    Classify the Macro Regime (RISK_ON, RISK_OFF, STAGFLATION, or NEUTRAL).
    Assign macro tilts (-1.0 Bearish to +1.0 Bullish) for our ETF themes:
      Technology, European Equities, Emerging Markets, Energy & Commodities,
      Precious Metals, Defensive Sectors, Digital Assets.
    Highlight tail risks (especially warning against daily leveraged ETFs if VIX is high).
    """

    # --- Option A: Google Gemini 1.5 Flash (100% Free Tier) ---
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=gemini_key)
            structured_llm = llm.with_structured_output(MacroBriefing)
            return structured_llm.invoke(prompt_context)
        except Exception as e:
            print(f"Gemini API invocation note: {e}. Falling back to quant rulebook.")

    # --- Option B: OpenAI (gpt-4o-mini) ---
    elif openai_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
            structured_llm = llm.with_structured_output(MacroBriefing)
            return structured_llm.invoke(prompt_context)
        except Exception as e:
            print(f"OpenAI API note: {e}. Falling back to quant rulebook.")

    # --- Option C: Deterministic Academic Quant Engine (Offline & $0.00 Cost) ---
    # Regime classification derived directly from Hawkish-Dovish Index and VIX
    if vix > 22.0 or (hd_score > 0.4 and us10y > 4.6):
        regime = "RISK_OFF"
        sentiment = -0.40
        tail_risks = [
            f"Elevated VIX ({vix:.1f}) and Hawkish Fed policy ({hd_score:+.2f}) indicate tightening conditions. Avoid daily leveraged ETFs."
        ]
    elif vix < 16.0 and hd_score <= 0.2:
        regime = "RISK_ON"
        sentiment = 0.50
        tail_risks = [
            "Low volatility and balanced monetary policy support equity momentum; monitor sudden rate repricing."
        ]
    else:
        regime = "NEUTRAL"
        sentiment = 0.15
        tail_risks = [
            "Moderated economic growth; balanced central bank communication indicates steady rotational trading."
        ]

    # Thematic tilts mapped to quant features
    theme_tilts = {
        "Technology": 0.45 if regime == "RISK_ON" else -0.15,
        "European Equities": 0.35 if eurusd >= 1.07 else 0.10,
        "Emerging Markets": -0.20 if dxy > 104.5 else 0.25,
        "Energy & Commodities": 0.30 if brent > 80.0 else -0.05,
        "Precious Metals": 0.40 if (regime == "RISK_OFF" or hd_score < -0.1) else 0.10,
        "Defensive Sectors": 0.45 if regime == "RISK_OFF" else 0.15,
        "Digital Assets": 0.35 if (regime == "RISK_ON" and vix < 18) else -0.30
    }

    summary = (
        f"Fed Communication Analysis (Apel-Grimaldi H-D Index: {hd_score:+.2f}, LM Uncertainty: {lm_uncert:.1f}%) "
        f"alongside VIX at {vix:.1f} indicates a {regime} macroeconomic backdrop. Core European and quality equities "
        f"remain attractive, while daily leveraged products face compounding volatility drag."
    )

    return MacroBriefing(
        trading_date=trading_date,
        macro_regime=regime,
        market_sentiment_score=sentiment,
        fed_hawkish_dovish_score=hd_score,
        fed_uncertainty_score=lm_uncert,
        theme_macro_tilts=theme_tilts,
        tail_risk_alerts=tail_risks,
        executive_summary=summary
    )

if __name__ == "__main__":
    briefing = evaluate_macro_agent()
    print("=" * 65)
    print("MACRO & NEWS AGENT (FED REPORT + APEL-GRIMALDI & LM QUANT)")
    print("=" * 65)
    print(f"Trading Date: {briefing.trading_date}")
    print(f"Macro Regime: {briefing.macro_regime} | Sentiment Score: {briefing.market_sentiment_score:+.2f}")
    print(f"Fed Hawkish-Dovish Index (Apel-Grimaldi): {briefing.fed_hawkish_dovish_score:+.2f}")
    print(f"Fed Uncertainty Ratio (Loughran-McDonald): {briefing.fed_uncertainty_score:.2f}%")
    print("\nTheme Macro Tilts:")
    for theme, tilt in briefing.theme_macro_tilts.items():
        print(f"  - {theme:<22}: {tilt:+.2f}")
    print("\nTail Risk Alerts:")
    for alert in briefing.tail_risk_alerts:
        print(f"  ⚠️ {alert}")
    print(f"\nExecutive Summary:\n{briefing.executive_summary}")
