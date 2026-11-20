# src/pipeline/headline_audit.py
"""
Headline Audit & Traceability Engine
Belfius ReBel x QuantumBlack | QDC Fall 2026 Edition (Team 29)

Provides line-by-line provenance, quantitative filtering, and readable audit blotters
for all macroeconomic and financial headlines ingested by the Macro Agent.
"""

import os
import json
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
from pydantic import BaseModel, Field

from src.agents.macro import HEADLINES_CACHE_FILE, THEME_KEYWORDS, LM_POSITIVE, LM_NEGATIVE, NEGATION_WORDS

AUDIT_LOG_JSONL = Path("logs/news_headlines_audit.jsonl")


class HeadlineRecord(BaseModel):
    headline_id: str
    title: str
    source: str
    published_at: str
    collected_at: str
    latency_minutes: float
    sentiment_score: float
    sentiment_label: str
    primary_driver: str = "MACRO_EVENT"
    detected_themes: List[str]
    economic_rationale: str = ""


def score_single_headline(title: str) -> Dict[str, Any]:
    """
    Computes literature-grounded sentiment score, label, and detected themes
    using Loughran-McDonald (2011) with a 4-token preceding negation window
    to accurately handle failure and reversal phrases (e.g. 'fails to provide confidence', 'not falling').
    """
    title_lower = title.lower()
    words = re.findall(r"\b[a-z\-]+\b", title_lower)
    
    pos_count = 0
    neg_count = 0
    
    for i, w in enumerate(words):
        # 4-token lookback window for negation terms (not, no, fails, lack, unable)
        window = words[max(0, i - 4):i]
        is_negated = any(nw in window for nw in NEGATION_WORDS)

        if w in LM_POSITIVE:
            if is_negated:
                neg_count += 1
            else:
                pos_count += 1
        elif w in LM_NEGATIVE:
            if is_negated:
                pos_count += 1
            else:
                neg_count += 1
    
    denom = pos_count + neg_count
    if denom == 0:
        score = 0.0
        label = "NEUTRAL"
    else:
        score = round((pos_count - neg_count) / denom, 2)
        if score > 0.15:
            label = "POSITIVE"
        elif score < -0.15:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

    # Detect theme tags
    detected = []
    for theme, kws in THEME_KEYWORDS.items():
        if any(kw in title_lower for kw in kws):
            detected.append(theme)

    return {
        "score": score,
        "label": label,
        "themes": detected or ["General Macro"]
    }


def get_headline_audit_trail(
    source: Optional[str] = None,
    theme: Optional[str] = None,
    sentiment_label: Optional[str] = None,
    search_query: Optional[str] = None,
    lookback_days: int = 5,
    use_llm: bool = True,
    limit: Optional[int] = 50
) -> pd.DataFrame:
    """
    Retrieves and filters the audit trail of ingested headlines using batched LLM intelligence.
    Reads from cache or persistent JSONL audit logs.
    
    Parameters:
      source: Filter by feed substring (e.g. 'Google', 'Yahoo', 'CNBC')
      theme: Filter by thematic cluster (e.g. 'Technology', 'Energy & Commodities', 'Precious Metals')
      sentiment_label: Filter by 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
      search_query: Substring text match in title
      lookback_days: Number of days in the past to include (default: 5, maximum: 30)
      use_llm: Whether to apply LLM contextual reasoning (default: True, with failsafe fallback)
      limit: Maximum records to return (None for unlimited)
    """
    # 1. Enforce 5-day default and 30-day maximum lookback ceiling
    if lookback_days is None:
        lookback_days = 5
    clamped_lookback = min(max(1, int(lookback_days)), 30)
    if int(lookback_days) > 30:
        print(f"[GOVERNANCE NOTICE] Lookback window requested ({lookback_days} days) exceeds maximum allowable 30-day limit. Clamped to 30 days to prevent stale narrative decay.")

    now_utc = datetime.now(timezone.utc)
    from datetime import timedelta
    cutoff_dt = now_utc - timedelta(days=clamped_lookback)

    headlines_raw = []
    seen_raw_titles = set()

    # 2. Attempt loading from cache file
    if HEADLINES_CACHE_FILE.exists():
        try:
            with open(HEADLINES_CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for h in data.get("headlines", []):
                    t = h.get("title", "").strip().lower()
                    if t and t not in seen_raw_titles:
                        seen_raw_titles.add(t)
                        headlines_raw.append(h)
        except Exception:
            pass

    # Also load accumulated history from persistent JSONL log
    if AUDIT_LOG_JSONL.exists():
        try:
            with open(AUDIT_LOG_JSONL, "r", encoding="utf-8") as f_log:
                for line in f_log:
                    if not line.strip():
                        continue
                    h = json.loads(line)
                    t = h.get("title", "").strip().lower()
                    if t and t not in seen_raw_titles:
                        seen_raw_titles.add(t)
                        headlines_raw.append(h)
        except Exception:
            pass

    # 3. Filter valid items within lookback window
    valid_items = []
    now_iso = now_utc.isoformat()

    for item in headlines_raw:
        title = item.get("title", "").strip()
        if not title:
            continue

        src = item.get("source", "Unknown")
        pub_date = item.get("pub_date", now_iso)
        
        # Calculate latency in minutes and evaluate lookback cutoff
        try:
            pub_dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            if pub_dt < cutoff_dt:
                continue  # Exclude headlines older than lookback window
            lat_min = round(max(0.0, (now_utc - pub_dt).total_seconds() / 60.0), 1)
        except Exception:
            lat_min = 0.0

        hid = "h_" + hashlib.md5(f"{src}:{title}".encode("utf-8")).hexdigest()[:8]
        valid_items.append({
            "headline_id": hid,
            "title": title,
            "source": src,
            "pub_date": pub_date,
            "latency_minutes": lat_min
        })

    # 4. Batched LLM Reasoning Pipeline
    llm_evaluations = {}
    if use_llm and valid_items:
        try:
            from src.pipeline.headline_llm import evaluate_headlines_with_llm
            llm_evaluations = evaluate_headlines_with_llm(valid_items)
        except Exception as e:
            print(f"Notice: LLM evaluation pipeline error ({e}). Using academic fallback.")

    # 5. Build Final Headline Records
    records: List[HeadlineRecord] = []
    for item in valid_items:
        hid = item["headline_id"]
        title = item["title"]
        src = item["source"]
        pub_date = item["pub_date"]
        lat_min = item["latency_minutes"]

        if hid in llm_evaluations:
            evaluated = llm_evaluations[hid]
            score = evaluated.market_sentiment
            label = evaluated.sentiment_label
            driver = evaluated.primary_driver
            themes = evaluated.impacted_themes
            rationale = evaluated.economic_rationale
        else:
            quant = score_single_headline(title)
            score = quant["score"]
            label = quant["label"]
            driver = "MACRO_EVENT"
            themes = quant["themes"]
            rationale = "Evaluated via quantitative fallback model."

        records.append(HeadlineRecord(
            headline_id=hid,
            title=title,
            source=src,
            published_at=pub_date,
            collected_at=now_iso,
            latency_minutes=lat_min,
            sentiment_score=score,
            sentiment_label=label,
            primary_driver=driver,
            detected_themes=themes,
            economic_rationale=rationale
        ))

    if not records:
        return pd.DataFrame(columns=[
            "headline_id", "source", "title", "published_at", "latency_minutes",
            "sentiment_score", "sentiment_label", "primary_driver", "detected_themes", "economic_rationale",
            "ID", "Source", "Published", "Latency (min)", "Sentiment", "Label", "Driver", "Themes", "Headline Title", "Rationale"
        ])

    # Convert to DataFrame with both canonical raw fields and display aliases
    rows = []
    for r in records:
        rows.append({
            # Canonical raw schema (as-is, unmodified)
            "headline_id": r.headline_id,
            "source": r.source,
            "title": r.title,
            "published_at": r.published_at,
            "latency_minutes": r.latency_minutes,
            "sentiment_score": r.sentiment_score,
            "sentiment_label": r.sentiment_label,
            "primary_driver": r.primary_driver,
            "detected_themes": ", ".join(r.detected_themes),
            "economic_rationale": r.economic_rationale,
            # Display aliases for blotter printing
            "ID": r.headline_id,
            "Source": r.source,
            "Published": r.published_at[:16].replace("T", " "),
            "Latency (min)": r.latency_minutes,
            "Sentiment": r.sentiment_score,
            "Label": r.sentiment_label,
            "Driver": r.primary_driver,
            "Themes": ", ".join(r.detected_themes),
            "Headline Title": r.title,
            "Rationale": r.economic_rationale
        })

    df = pd.DataFrame(rows)

    # 6. Apply Filters
    if source:
        df = df[df["source"].str.contains(source, case=False, na=False)]
    if theme:
        df = df[df["detected_themes"].str.contains(theme, case=False, na=False)]
    if sentiment_label:
        df = df[df["sentiment_label"].str.upper() == sentiment_label.upper()]
    if search_query:
        df = df[df["title"].str.contains(search_query, case=False, na=False)]

    if limit is not None and limit > 0:
        return df.head(limit)
    return df


def export_headlines_csv(
    output_path: str = "data/headlines_collected.csv",
    lookback_days: int = 5,
    source: Optional[str] = None,
    theme: Optional[str] = None,
    sentiment_label: Optional[str] = None,
    search_query: Optional[str] = None
) -> str:
    """
    Exports all collected headlines to a CSV file as-is (not shortened, not modified).
    
    Parameters:
      output_path: Filepath where the CSV will be saved (default: data/headlines_collected.csv).
      lookback_days: Lookback window in days (default: 5 days, maximum: 30 days).
      source, theme, sentiment_label, search_query: Optional filters.
      
    Returns:
      The normalized file path string of the generated CSV.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Fetch all records within the requested lookback window
    df = get_headline_audit_trail(
        source=source,
        theme=theme,
        sentiment_label=sentiment_label,
        search_query=search_query,
        lookback_days=lookback_days,
        limit=None
    )

    # If cache is missing or empty, trigger ingestion to ensure file is populated
    if df.empty:
        try:
            from src.agents.macro import fetch_live_headlines
            fetch_live_headlines()
            df = get_headline_audit_trail(
                source=source,
                theme=theme,
                sentiment_label=sentiment_label,
                search_query=search_query,
                lookback_days=lookback_days,
                limit=None
            )
        except Exception as e:
            print(f"Notice during headline fetch: {e}")

    # Canonical columns export (raw, complete, untruncated with economic rationale)
    canonical_cols = [
        "headline_id",
        "source",
        "title",
        "published_at",
        "latency_minutes",
        "sentiment_score",
        "sentiment_label",
        "primary_driver",
        "detected_themes",
        "economic_rationale"
    ]
    export_df = df[[c for c in canonical_cols if c in df.columns]]
    export_df.to_csv(out_file, index=False, encoding="utf-8")
    return str(out_file.resolve() if out_file.is_absolute() else out_file)


def print_headline_audit_blotter(
    source: Optional[str] = None,
    theme: Optional[str] = None,
    sentiment_label: Optional[str] = None,
    search_query: Optional[str] = None,
    lookback_days: int = 5,
    limit: int = 25
):
    """Prints an institutional tabular blotter of collected headlines in the terminal."""
    clamped_days = min(max(1, int(lookback_days)), 30) if lookback_days else 5
    df = get_headline_audit_trail(
        source=source,
        theme=theme,
        sentiment_label=sentiment_label,
        search_query=search_query,
        lookback_days=clamped_days,
        limit=limit
    )

    print("=" * 115)
    print("BELFIUS REBEL - REAL-TIME LLM HEADLINE INTELLIGENCE & AUDIT TRAIL")
    print("=" * 115)
    print(f"Lookback Window   : {clamped_days} calendar days (default: 5d, maximum ceiling: 30d)")
    print(f"Filter Parameters : Source='{source or 'ALL'}' | Theme='{theme or 'ALL'}' | Sentiment='{sentiment_label or 'ALL'}' | Search='{search_query or 'ALL'}'")
    print(f"Total Matches     : {len(df)} headlines displayed (max blotter limit: {limit})")
    print("=" * 115)

    if df.empty:
        print("No headlines found matching the specified filter criteria.")
        print("=" * 115)
        return

    for i, row in df.iterrows():
        print(f"[{row['ID']}] {row['Published']} UTC ({row['Latency (min)']}m ago) | {row['Source']}")
        print(f"  Sentiment: {row['Sentiment']:+.2f} ({row['Label']}) | Driver: {row['Driver']} | Themes: {row['Themes']}")
        print(f"  Headline : {row['Headline Title']}")
        if row.get('Rationale'):
            print(f"  Rationale: {row['Rationale']}")
        print("-" * 115)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect and Audit Collected Macro Headlines")
    parser.add_argument("--source", type=str, default=None, help="Filter by source substring (e.g. 'Google', 'Yahoo', 'CNBC')")
    parser.add_argument("--theme", type=str, default=None, help="Filter by ETF theme (e.g. 'Technology', 'Energy', 'Precious Metals')")
    parser.add_argument("--sentiment", type=str, default=None, choices=["POSITIVE", "NEGATIVE", "NEUTRAL"], help="Filter by sentiment label")
    parser.add_argument("--search", type=str, default=None, help="Text search substring in headline")
    parser.add_argument("--lookback-days", type=int, default=5, help="Lookback window in days (default: 5, maximum: 30)")
    parser.add_argument("--limit", type=int, default=20, help="Maximum headlines to display in blotter (default: 20)")
    parser.add_argument(
        "--export-csv",
        nargs="?",
        const="data/headlines_collected.csv",
        default=False,
        help="Trigger (default: False) to generate a CSV file summarising all headlines as-is. Optional custom path can be specified."
    )

    args = parser.parse_args()

    if args.export_csv:
        csv_path = export_headlines_csv(
            output_path=str(args.export_csv),
            lookback_days=args.lookback_days,
            source=args.source,
            theme=args.theme,
            sentiment_label=args.sentiment,
            search_query=args.search
        )
        print(f"Exported headlines as-is to CSV: {csv_path} (lookback: {args.lookback_days} days)")

    print_headline_audit_blotter(
        source=args.source,
        theme=args.theme,
        sentiment_label=args.sentiment,
        search_query=args.search,
        lookback_days=args.lookback_days,
        limit=args.limit
    )
