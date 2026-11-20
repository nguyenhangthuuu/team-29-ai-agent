# src/pipeline/headline_llm.py
"""
Batched LLM Headline Intelligence Engine
Belfius ReBel x QuantumBlack | QDC Fall 2026 Edition (Team 29)

Replaces naive dictionary and isolated sentence-transformer approaches with
instruction-tuned LLM reasoning (Gemini 1.5/2.5 Flash / GPT-4o-mini).
Parses complex financial syntax, contrastive clauses, sector-asymmetric impacts,
and economic transmission mechanisms with full auditability and zero API cost.
"""

import os
import json
import re
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# The 7 target QDC ETF thematic clusters
TARGET_THEMES = [
    "Technology",
    "European Equities",
    "Emerging Markets",
    "Energy & Commodities",
    "Precious Metals",
    "Defensive Sectors",
    "Digital Assets"
]


class AnalyzedHeadline(BaseModel):
    headline_id: str
    market_sentiment: float = Field(
        ..., ge=-1.0, le=1.0, 
        description="Net directional impact on equities from -1.0 (Panic/Crisis) to +1.0 (Expansion/Risk-On)"
    )
    sentiment_label: str = Field(
        ..., description="Categorical label: POSITIVE, NEGATIVE, or NEUTRAL"
    )
    primary_driver: str = Field(
        ..., description="Core economic catalyst: e.g. MONETARY_POLICY, INFLATION, EARNINGS, GEOPOLITICS, TECH_CAPEX"
    )
    impacted_themes: List[str] = Field(
        default_factory=list,
        description="Target themes impacted: Technology, European Equities, Emerging Markets, Energy & Commodities, Precious Metals, Defensive Sectors, Digital Assets"
    )
    economic_rationale: str = Field(
        ..., description="Concise one-sentence rationale explaining the economic transmission mechanism"
    )


class BatchHeadlineEvaluation(BaseModel):
    overall_regime: str = Field(
        ..., description="HEADLINE_PANIC, RISK_EXPANSION, POLICY_UNCERTAINTY, or NEUTRAL"
    )
    macro_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Aggregate macroeconomic news valence"
    )
    top_negative_catalysts: List[str] = Field(default_factory=list)
    top_positive_catalysts: List[str] = Field(default_factory=list)
    theme_tactical_tilts: Dict[str, float] = Field(
        default_factory=dict,
        description="Tactical delta adjustments (-0.30 to +0.30) for each of the 7 thematic clusters"
    )
    analyzed_headlines: List[AnalyzedHeadline] = Field(default_factory=list)


def get_configured_llm():
    """Initializes LLM client (Gemini Flash or OpenAI) if API keys are present."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key and not gemini_key.startswith("your_"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Try gemini-1.5-flash or gemini-2.5-flash
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.1,
                google_api_key=gemini_key
            )
        except Exception as e:
            print(f"Gemini LLM init notice: {e}")

    if openai_key and not openai_key.startswith("your_"):
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.1,
                api_key=openai_key
            )
        except Exception as e:
            print(f"OpenAI LLM init notice: {e}")

    return None


def fallback_contextual_analyzer(headline: Dict[str, Any]) -> AnalyzedHeadline:
    """
    Context-aware deterministic heuristic fallback when no LLM key is configured.
    Correctly resolves failure verbs and contrastive clauses.
    """
    title = headline.get("title", "")
    hid = headline.get("headline_id", "h_unknown")
    t_lower = title.lower()
    words = re.findall(r"\b[a-z\-]+\b", t_lower)

    # 1. Negative drivers & failure verbs
    failure_patterns = [
        "fails to provide", "failed to provide", "fails to deliver", "lack of confidence",
        "end lower", "ended lower", "tumble", "slump", "slide", "plunge", "crisis",
        "panic", "selloff", "sell-off", "loss", "retreat", "drop", "sink", "sinks"
    ]
    positive_patterns = [
        "surge", "surges", "soar", "soars", "rally", "rallies", "record high", "gain",
        "gains", "boom", "strong growth", "expansion", "bullish", "resilient"
    ]

    has_neg_pattern = any(p in t_lower for p in failure_patterns)
    has_pos_pattern = any(p in t_lower for p in positive_patterns)

    # Detect sector tags
    themes = []
    if any(k in t_lower for k in ["tech", "ai", "nasdaq", "chips", "semiconductor", "anthropic", "nvidia"]):
        themes.append("Technology")
    if any(k in t_lower for k in ["oil", "opec", "energy", "gas", "crude", "uranium"]):
        themes.append("Energy & Commodities")
    if any(k in t_lower for k in ["gold", "silver", "safe haven", "bullion"]):
        themes.append("Precious Metals")
    if any(k in t_lower for k in ["europe", "ecb", "dax", "eurozone", "stoxx"]):
        themes.append("European Equities")
    if any(k in t_lower for k in ["china", "india", "emerging", "tariff", "asia"]):
        themes.append("Emerging Markets")
    if any(k in t_lower for k in ["fed", "inflation", "cpi", "rate cut", "rates"]):
        themes.append("Defensive Sectors")

    if not themes:
        themes = ["Technology"] if ("s&p" in t_lower or "nasdaq" in t_lower) else ["General Macro"]

    if has_neg_pattern and not has_pos_pattern:
        score = -0.80
        label = "NEGATIVE"
        driver = "MARKET_SELLOFF"
        rationale = "Negative momentum or failed catalyst created risk-off downward pressure on equities."
    elif has_pos_pattern and not has_neg_pattern:
        score = 0.80
        label = "POSITIVE"
        driver = "EXPANSION_MOMENTUM"
        rationale = "Bullish momentum and favorable earnings/growth catalyst supported equity valuations."
    elif has_neg_pattern and has_pos_pattern:
        # Check clause resolution (e.g. "end lower as debut fails...")
        if "lower" in t_lower or "fail" in t_lower:
            score = -0.75
            label = "NEGATIVE"
            driver = "DISAPPOINTMENT_REVERSAL"
            rationale = "Headline expresses disappointment or selloff taking precedence over secondary positive catalyst."
        else:
            score = 0.0
            label = "NEUTRAL"
            driver = "POLICY_UNCERTAINTY"
            rationale = "Contrasting signals in headline produced neutral market conviction."
    else:
        score = 0.0
        label = "NEUTRAL"
        driver = "NEUTRAL_MACRO"
        rationale = "Absence of strong directional sentiment or material macroeconomic shock."

    return AnalyzedHeadline(
        headline_id=hid,
        market_sentiment=score,
        sentiment_label=label,
        primary_driver=driver,
        impacted_themes=themes,
        economic_rationale=rationale
    )


def evaluate_headlines_with_llm(
    headlines: List[Dict[str, Any]],
    batch_size: int = 35
) -> Dict[str, AnalyzedHeadline]:
    """
    Batched execution of LLM headline intelligence.
    Returns mapping: headline_id -> AnalyzedHeadline.
    """
    results: Dict[str, AnalyzedHeadline] = {}
    if not headlines:
        return results

    llm = get_configured_llm()

    # If no LLM available, use contextual fallback
    if llm is None:
        for h in headlines:
            analyzed = fallback_contextual_analyzer(h)
            results[analyzed.headline_id] = analyzed
        return results

    # Chunk into batches of up to batch_size
    from langchain_core.messages import SystemMessage, HumanMessage
    structured_llm = llm.with_structured_output(BatchHeadlineEvaluation)

    system_prompt = (
        "You are the Senior Macro Quantitative Strategist for Belfius Bank. "
        "Evaluate the following financial news headlines with economic transmission reasoning.\n"
        "Guidelines:\n"
        "1. Do NOT do naive keyword counting. Understand syntax: 'fails to provide confidence' is BEARISH (-0.80).\n"
        "2. Identify the economic driver (e.g. MONETARY_POLICY, INFLATION, EARNINGS, GEOPOLITICS).\n"
        "3. Map each headline to affected themes among: Technology, European Equities, Emerging Markets, "
        "Energy & Commodities, Precious Metals, Defensive Sectors, Digital Assets.\n"
        "4. Provide a crisp one-sentence economic rationale explaining how this affects investor risk appetite."
    )

    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        payload = [
            {
                "id": h.get("headline_id", f"h_{idx}"),
                "source": h.get("source", "News"),
                "title": h.get("title", ""),
                "published_at": h.get("published_at", "")
            }
            for idx, h in enumerate(batch)
        ]

        try:
            response: BatchHeadlineEvaluation = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"HEADLINES BATCH (Count: {len(payload)}):\n{json.dumps(payload, indent=2)}")
            ])
            for item in response.analyzed_headlines:
                results[item.headline_id] = item
        except Exception as e:
            print(f"Notice: LLM batch evaluation error ({e}). Running contextual fallback for batch.")
            for h in batch:
                analyzed = fallback_contextual_analyzer(h)
                results[analyzed.headline_id] = analyzed

    # Ensure all inputs received an analysis
    for h in headlines:
        hid = h.get("headline_id")
        if hid not in results:
            results[hid] = fallback_contextual_analyzer(h)

    return results
