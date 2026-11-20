# Macroeconomic Factor Engine & Methodology

## Executive Overview

The **Macro & News Agent** (`src/agents/macro.py`) provides quantitative macroeconomic intelligence to the **Portfolio Manager** and **Risk Manager** agents within the LangGraph multi-agent architecture. Rather than relying on unstructured, hallucination-prone LLM text prompts, our engine synthesizes five deterministic, academically grounded layers:
1. **Academic Central Bank Text Mining:** Lexicon-based sentiment and policy stance extraction (**Apel & Blix Grimaldi, 2014**; **Bennani & Farvaque, 2018**; **Loughran-McDonald, 2011**) equipped with bigram detection and a 3-token contextual negation window.
2. **Automated Dual Central Bank Ingestion (Fed & ECB):** Live parsing of official XML/RSS releases from both the **Federal Reserve** (`press_monetary.xml`) and the **European Central Bank** (`rss/press.html`):
   - **Federal Reserve:** FOMC policy statements, meeting minutes, and the Semi-Annual Monetary Policy Report to Congress.
   - **European Central Bank:** Governing Council interest rate decisions, Press Conference statements, Economic Bulletins, and the semi-annual Financial Stability Review (FSR).
3. **Transatlantic Policy Spread Quantification ($\Delta_{\text{Policy}}$):** Real-time measurement of monetary divergence between the Fed and the ECB ($\Delta_{\text{Policy}} = HD_{\text{Fed}} - HD_{\text{ECB}}$) to capture foreign exchange rate transmission ($EUR/USD$) and export earnings momentum for European blue chips (`EUROPE 600`, `BEL 20`).
4. **Live Quantified Headline Sentiment Stream (Sources A, B, C):** Continuous market headline sentiment quantified via a hybrid engine (single-batch Google Gemini 1.5 Flash + Loughran-McDonald fallback) with user timeframe restrictions (max 30 days, default 5 days) and news latency tracking.
5. **Continuous Real-Time Market Barometers:** Yield curves, volatility indices, energy benchmarks, and currency movements ingested via `yfinance` at zero API cost.

```
                                      DATA INGESTION LAYER
           ┌─────────────────────────────────────┼─────────────────────────────────────┐
           ▼                                     ▼                                     ▼
 Central Bank RSS Feeds               Real-Time Market Barometers             Live Headline Stream
(Fed Statements, Minutes, ECB FSR)   (VIX, US10Y, DXY, Brent, EUR/USD)      (Sources A, B, and C Only)
           │                                     │                                     │
           ▼                                     │                                     ▼
 ACADEMIC TEXT MINING                            │                          HYBRID SENTIMENT ENGINE
(Apel-Grimaldi & Loughran-McDonald)              │                     (Batched Gemini + LM Quant Fallback)
           │                                     │                                     │
           └──────────────────┬──────────────────┴──────────────────┬──────────────────┘
                              ▼                                     ▼
                  STRUCTURED MACRO REGIME              TACTICAL HEADLINE DELTAS
                  & TRANSATLANTIC SPREAD                 (Latency & Timeframe)
                              │                                     │
                              └──────────────────┬──────────────────┘
                                                 ▼
                                     FINAL BLENDED MACRO TILTS
                                        & TAIL RISK ALERTS
```

---

## 1. Automated Central Bank Ingestion Architecture

### 1.1 Federal Reserve Live Scraper (`fetch_live_fed_communication`)
* **Source URL:** `https://www.federalreserve.gov/feeds/press_monetary.xml`
* **Target Filter:** Items matching `"statement"`, `"minutes"`, `"monetary policy report"`, or `"economic"`.
* **HTML Parsing:** Follows official article links and extracts text from `<div id="article">` or `<div class="col-xs-12">`.
* **Caching & Invalidation:** Stored in `data/central_bank_cache.json`. When a new official statement or minutes release is published, the cache updates automatically.

### 1.2 European Central Bank Live Scraper (`fetch_live_ecb_communication`)
* **Source URL:** `https://www.ecb.europa.eu/rss/press.html`
* **Target Keywords Filter:**
  ```python
  TARGET_KEYWORDS = [
      "monetary policy", "financial stability", "economic bulletin", "wage",
      "inflation", "rates", "speech", "press conference", "interview",
      "governing council", "outlook", "stability review", "resilience"
  ]
  ```
* **HTML Parsing:** Follows article links and extracts body text from `<main>`, `<div class="section">`, `<article>`, or `<div id="article">`. Long executive speeches are bounded at 6,000 characters per release to maintain low computational latency.
* **Caching & Invalidation:** Stored in `data/ecb_central_bank_cache.json`.

---

## 2. Quantifying Headline Sentiment Stream (Sources A, B, and C)

### 2.1 Authorized Data Sources
To comply strictly with the zero-budget constraint and eliminate fragile scraping of paywalled media, the Headline Sentiment stream uses **exclusively** three authorized real-time feeds:

| Source Identifier | Source Name & Type | Coverage & Function | Endpoint / Mechanism |
| :--- | :--- | :--- | :--- |
| **Source A** | **Google News RSS (Macro)** | Broad macroeconomic developments, global inflation prints, central bank leaks, tariff and geopolitical news | `https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en` |
| **Source B** | **Yahoo Finance News Stream** | Asset-specific, market-moving headlines for benchmark ETF tickers (`SPY`, `QQQ`, `CSX5.L`, `GLD`, `BZ=F`, `^VIX`) | Built-in `yfinance` `Ticker(sym).news` feed |
| **Source C** | **CNBC Economy RSS Feed** | Curated macroeconomic and Federal Reserve editorial analysis | `https://www.cnbc.com/id/20910258/device/rss/rss.html` |

### 2.2 Timeframe Filtering & "Reference Date" Governance
In live quantitative portfolio management, backtesting and live trading must restrict news ingestion to avoid lookahead bias and control news decay:

1. **User-Specified Reference Date:**
   * The user passes a `reference_date` string (e.g., `"2026-11-15"` or ISO timestamp).
   * The agent filters out any headlines published prior to `reference_date`.
2. **30-Day Recent Hard Maximum Lookback & Human-in-the-Loop Alert:**
   * Market headline sentiment decays rapidly; headlines older than 30 days are noise for short-term tactical tilts.
   * If the user specifies a reference date older than 30 days (e.g. `"2026-09-01"`), the engine **automatically clamps** the lookback window to exactly the past 30 days:
     $$\text{Start Date} = \max(\text{Reference Date}, \text{Current Time} - 30 \text{ days})$$
   * The agent generates an explicit notification to the human-in-the-loop:
     `"AGENT LIMITATION NOTICE FOR HUMAN-IN-THE-LOOP: The selected reference date is further than 30 days in the past. To prevent stale sentiment decay and maintain model precision, the agent strictly restricts headline scraping to the most recent 30-day window. News prior to the 30-day cutoff is excluded."`
   * This notification is prominently attached across three locations:
     1. As an active warning banner in `MacroBriefing.agent_limit_notification`.
     2. As the first item in `MacroBriefing.tail_risk_alerts`.
     3. Broadcast prominently at the Human-in-the-Loop approval checkpoint (`human_approval_node`) in LangGraph.
3. **Default 5-Day Trading Window:**
   * If the user does not specify a reference date (`reference_date is None`), the agent automatically defaults the start date to **5 days ago** ($\text{Current Time} - 5 \text{ days}$), matching the QDC live competition trading frequency (trades occurring every 4 to 5 days: Dec 7, 10, 14, 17, 21).
4. **Latency Measurement & Audit Reporting:**
   * To keep human-in-the-loop decision-makers fully informed of information staleness, the agent calculates and reports:
     $$\text{News Latency (minutes)} = \frac{t_{\text{execution}} - t_{\text{freshest headline}}}{60 \text{ seconds}}$$
   * The timestamp of the latest scraped headline and latency are permanently recorded in the audit trail.

### 2.3 Hybrid Text Mining Engine
Headline text is quantified through a two-tier hybrid model:

* **Tier 1 (Google Gemini 1.5 Flash Single-Batch Analysis):**
  When a free `GEMINI_API_KEY` is present, the top 20–25 curated headlines are formatted into a single batch prompt. Gemini evaluates the collection in **1 single API call per run** (strictly adhering to the 15 RPM free tier limit) and outputs a structured Pydantic object:
  - `headline_sentiment_score` ($\in [-1.0, +1.0]$)
  - `news_regime`: `HEADLINE_PANIC`, `RISK_EXPANSION`, `POLICY_UNCERTAINTY`, or `NEUTRAL`
  - `theme_headline_tilts`: tactical deltas ($\in [-0.30, +0.30]$) for each ETF theme
  - `top_breaking_catalysts`: top 2–3 macro catalysts driving the score
* **Tier 2 (Deterministic Academic Quant Fallback):**
  If `GEMINI_API_KEY` is omitted or rate-limited, the system falls back to a deterministic algorithm based on Loughran-McDonald (2011) sentiment words and Apel-Grimaldi central bank dictionaries with a 3-token negation window:
  $$S_{\text{Headline}} = \frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{pos}} + N_{\text{neg}} + \epsilon} \in [-1.0, +1.0]$$

---

## 3. Macro Regime Engine

### Definition & Purpose
The **Macro Regime** classifies the global market environment into discrete states. It acts as the primary governor determining whether the portfolio should expand risk assets or retreat into capital preservation and defensive hedges.

The model evaluates four distinct regimes:
* `RISK_ON`: Characterized by low volatility, expanding economic liquidity, accommodative monetary policy, and positive headline sentiment. Favors equity momentum, technology, and emerging markets.
* `RISK_OFF`: Characterized by elevated market volatility, central bank tightening, or breaking headline panic. Mandates increased cash buffers, gold, and consumer defensives.
* `STAGFLATION`: Characterized by accelerating energy/commodity inflation alongside slowing growth and high bond yields. Favors commodities and cash; penalizes duration assets.
* `NEUTRAL`: Characterized by range-bound volatility and balanced monetary policy. Supports disciplined rotational trading.

### Mathematical Formulation & Classification Rules
The regime is classified using a deterministic hierarchical rulebook:

$$\text{Regime} = \begin{cases} 
\text{RISK\_OFF} & \text{if } VIX > 22.0 \quad \text{or} \quad (\max(HD_{\text{Fed}}, HD_{\text{ECB}}) > 0.40 \text{ and } US10Y > 4.60\%) \\
                 & \quad \text{or } \text{News Regime} = \text{"HEADLINE\_PANIC"} \\
\text{STAGFLATION} & \text{if } BRENT > \$88.0/\text{bbl} \text{ and } US10Y > 4.50\% \text{ and } VIX > 19.0 \\
\text{RISK\_ON} & \text{if } VIX < 16.0 \text{ and } \max(HD_{\text{Fed}}, HD_{\text{ECB}}) \le 0.25 \text{ and } S_{\text{Headline}} \ge 0.0 \\
\text{NEUTRAL} & \text{otherwise}
\end{cases}$$

---

## 4. Central Bank Hawkish-Dovish Index (Apel & Blix Grimaldi, 2014)

### 4.1 Theoretical Background
Central bank communication steers asset valuations months before official rate cuts or hikes occur. To capture this without subjective bias, we implement the econometric dictionary framework developed by **Apel & Blix Grimaldi (2014)** and formalized by **Bennani & Farvaque (2018)**.

### 4.2 Academic Lexicon Specification

| Category | Description | Terminology Included |
| :--- | :--- | :--- |
| **Hawkish Vocabulary** ($N_{\text{hawk}}$) | Tightening, overheating, inflation persistence, rate hikes | `tighten`, `tightening`, `restrictive`, `overheating`, `inflationary`, `accelerate`, `acceleration`, `hike`, `hikes`, `hiking`, `upside`, `pressures`, `firming`, `unanchored`, `robust`, `vigilant`, `persistence`, `elevated`, `capacity constraint`, `wage growth`, `excess demand`, `hicp`, `second-round`, `second-round effects`, `unit labor costs`, `unit labour costs`, `wage pressures`, `domestic inflation`, `wage tracker`, `above target`, `sticky`, `stickiness`, `resilient`, `resilience` |
| **Dovish Vocabulary** ($N_{\text{dove}}$) | Easing, disinflation, slack, growth headwinds, rate cuts | `ease`, `easing`, `accommodative`, `cut`, `cuts`, `cutting`, `downside`, `disinflation`, `disinflationary`, `cooling`, `slowdown`, `subdued`, `slack`, `softening`, `fragile`, `fragility`, `headwinds`, `weakness`, `deteriorate`, `recession`, `stagnation`, `underutilization`, `tame`, `moderate`, `transmission`, `transmission mechanism`, `restrictive stance easing`, `below target`, `subdued demand` |
| **Ambiguity & Uncertainty** ($N_{\text{uncert}}$) | Loughran-McDonald (2011) risk and contingency lexicon | `uncertain`, `uncertainty`, `unclear`, `ambiguous`, `volatile`, `volatility`, `unpredictable`, `risk`, `risks`, `contingent`, `indefinite`, `hesitant`, `geoeconomic`, `fragmentation`, `geopolitical`, `fiscal imbalances`, `imbalance`, `imbalances`, `tariff`, `tariffs`, `vulnerable`, `vulnerability` |
| **Positive Sentiment** ($N_{\text{pos}}$) | Loughran-McDonald financial positive sentiment | `resilient`, `resilience`, `strong`, `favorable`, `improvement`, `improving`, `stability`, `confidence`, `momentum`, `gain`, `gains`, `progress`, `boom`, `rally`, `bullish`, `expansion`, `growth`, `outperform`, `soar`, `surge` |
| **Negative Sentiment** ($N_{\text{neg}}$) | Loughran-McDonald financial negative sentiment | `deficit`, `decline`, `deterioration`, `loss`, `losses`, `default`, `distress`, `adverse`, `recessionary`, `vulnerability`, `shock`, `shocks`, `slump`, `plunge`, `crisis`, `tumble`, `drop`, `bearish`, `crash`, `fear` |

### 4.3 Mathematical Formula
$$\text{HD Index} = \frac{N_{\text{hawk}} - N_{\text{dove}}}{N_{\text{hawk}} + N_{\text{dove}} + \epsilon} \in [-1.0, +1.0]$$

---

## 5. Transatlantic Policy Spread ($\Delta_{\text{Policy}}$)

### Economic Significance for European Portfolios
As a European investor allocating capital in EUR, the relative policy stance between the Federal Reserve and the European Central Bank determines currency valuation ($EUR/USD$) and export competitiveness for European blue chips (`EUROPE 600`, `BEL 20`, `EURO STOXX 50`).

$$\Delta_{\text{Policy}} = HD_{\text{Fed}} - HD_{\text{ECB}} \in [-2.0, +2.0]$$

| Transatlantic Spread | Macro Stance | FX Impact ($EUR/USD$) | ETF Asset Allocation Impact |
| :--- | :--- | :--- | :--- |
| **$\Delta_{\text{Policy}} > +0.15$** | **Fed Tighter than ECB** | US interest rate premium widens; **EUR depreciates** vs USD. | **Strong European Exporters:** Weaker Euro enhances foreign export competitiveness for European champions (automotive, luxury, industrials, chemicals). European revenue converted back to EUR increases earnings. $\rightarrow$ **Tilt on European Equities: $+0.40$**. |
| **$\Delta_{\text{Policy}} < -0.15$** | **ECB Tighter than Fed** | ECB cutting slower or hiking faster; **EUR appreciates** vs USD. | **US Tech Advantage:** Stronger Euro compresses European exporter margins. Favor US Equities and Technology. $\rightarrow$ **Tilt on Technology: $+0.45$**, European Equities: $+0.10$ to $+0.25$. |
| **$|\Delta_{\text{Policy}}| \le 0.15$** | **Monetary Synchronization** | Central banks move in lockstep; low interest rate differential. | **Balanced Rotational Trading:** Neutral FX impact; portfolio allocation governed primarily by technical trend signals and domestic demand. |

---

## 6. Theme Macro Tilts

The **Theme Macro Tilts** blend **Structural Macro** (Central Banks, Yield Curves, Oil, DXY) with **Tactical Headline Sentiment**:

$$\text{Final Theme Tilt}_k = \text{Structural Macro Tilt}_k + \beta \cdot \text{Headline Tilt}_k$$

Where:
* $\text{Structural Macro Tilt}_k \in [-1.0, +1.0]$
* $\text{Headline Tilt}_k \in [-0.30, +0.30]$: extracted from live headlines for theme $k$
* $\beta = 0.35$: sensitivity damping weight to prevent overreacting to transient news noise

### Granular Structural Formulations by Theme:

1. **Technology (`NASDAQ`, `SEMIS`, `CYBERSEC`, `ROBOTICS`, `SPACE`, `QUANTUM`):**
   $$\text{Base}_{\text{Tech}} = \begin{cases} +0.45 & \text{if } \text{Regime} = \text{RISK\_ON} \text{ and } HD_{\text{Fed}} \le 0.30 \\ -0.15 & \text{otherwise} \end{cases}$$
2. **European Equities (`EUROPE 600`, `BEL 20`, `EURO STOXX 50`, `IBEX 35`, `DIVIDEND LEADERS`):**
   $$\text{Base}_{\text{Europe}} = \begin{cases} +0.40 & \text{if } \Delta_{\text{Policy}} > 0.15 \\ +0.35 & \text{elif } HD_{\text{ECB}} < 0.00 \\ +0.25 & \text{elif } EURUSD \ge 1.07 \\ +0.10 & \text{otherwise} \end{cases}$$
3. **Emerging Markets (`EMMKT`, `CHINA LC`, `INDIA`, `KOREA`, `BRAZIL`, `CHINA A`):**
   $$\text{Base}_{\text{EM}} = \begin{cases} -0.20 & \text{if } DXY > 104.5 \\ +0.25 & \text{if } DXY \le 104.5 \end{cases}$$
4. **Energy & Commodities (`WORLD ENERGY`, `OIL & GAS E&P`, `URANIUM`, `CLEAN ENERGY`):**
   $$\text{Base}_{\text{Energy}} = \begin{cases} +0.30 & \text{if } BRENT > \$80.00/\text{bbl} \\ -0.05 & \text{if } BRENT \le \$80.00/\text{bbl} \end{cases}$$
5. **Precious Metals & Safe Havens (`GOLD`, `GOLD MINERS`, `SILVER MINERS`):**
   $$\text{Base}_{\text{Gold}} = \begin{cases} +0.40 & \text{if } \text{Regime} = \text{RISK\_OFF} \quad \text{or} \quad HD_{\text{Fed}} < -0.10 \quad \text{or} \quad Uncert_{\text{ECB}} > 1.50\% \\ +0.10 & \text{otherwise} \end{cases}$$
6. **Defensive Sectors (`DEFENSE`, `STAPLES`, `HEALTH CARE`):**
   $$\text{Base}_{\text{Defensive}} = \begin{cases} +0.45 & \text{if } \text{Regime} = \text{RISK\_OFF} \\ +0.15 & \text{if } \text{Regime} = \text{RISK\_ON} \end{cases}$$
7. **Digital Assets (`BITCOIN`, `CRYPTO EQUITY`):**
   $$\text{Base}_{\text{Crypto}} = \begin{cases} +0.35 & \text{if } \text{Regime} = \text{RISK\_ON} \text{ and } VIX < 18.0 \\ -0.30 & \text{if } \text{Regime} = \text{RISK\_OFF} \text{ or } VIX \ge 22.0 \end{cases}$$

---

## 7. Structured Output Schemas (`HeadlineStreamAnalysis` & `MacroBriefing`)

```python
class HeadlineStreamAnalysis(BaseModel):
    headline_sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0, description="Overall news valence (-1.0 Panic/Crisis to +1.0 Euphoria/Expansion)"
    )
    news_regime: str = Field(
        ..., description="HEADLINE_PANIC, RISK_EXPANSION, POLICY_UNCERTAINTY, or NEUTRAL"
    )
    theme_headline_tilts: Dict[str, float] = Field(
        ..., description="Tactical delta adjustments (-0.30 to +0.30) for ETF themes based on breaking headlines"
    )
    top_breaking_catalysts: List[str] = Field(
        ..., description="Top 2-3 most critical macro/market news catalysts"
    )
    total_headlines_analyzed: int
    latest_scraped_timestamp: str = Field(
        ..., description="ISO timestamp of the most recent headline scraped"
    )
    news_latency_minutes: float = Field(
        ..., description="Age in minutes of the freshest headline at run time"
    )
    timeframe_window_used: str = Field(
        ..., description="Start date to End date utilized for news collection"
    )
    timeframe_notification: Optional[str] = Field(
        None, description="System alert if reference date was clamped to 30 days or defaulted to 5 days"
    )

class MacroBriefing(BaseModel):
    trading_date: str
    macro_regime: str = Field(..., description="RISK_ON, RISK_OFF, STAGFLATION, or NEUTRAL")
    market_sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    fed_hawkish_dovish_score: float = Field(..., ge=-1.0, le=1.0, description="Apel-Grimaldi Index: -1.0 (Dovish) to +1.0 (Hawkish)")
    fed_uncertainty_score: float = Field(..., description="Fed Loughran-McDonald Uncertainty %")
    ecb_hawkish_dovish_score: float = Field(..., ge=-1.0, le=1.0, description="ECB Apel-Grimaldi Index: -1.0 (Dovish) to +1.0 (Hawkish)")
    ecb_uncertainty_score: float = Field(..., description="ECB Loughran-McDonald Uncertainty %")
    transatlantic_policy_spread: float = Field(..., description="Transatlantic Spread: Fed HD Score - ECB HD Score")
    headline_stream: HeadlineStreamAnalysis = Field(..., description="Quantified real-time market headline stream")
    theme_macro_tilts: Dict[str, float] = Field(..., description="Granular bias (-1.0 to +1.0) for each of the ETF themes")
    tail_risk_alerts: List[str]
    executive_summary: str
```

---

## 8. Portfolio Conviction Synthesis

In the LangGraph pipeline, the **Portfolio Manager Node** combines the **Technical Agent Score** with the **Final Blended Macro Tilt** using a linear conviction model:

$$\text{Final Conviction Score}_i = 0.60 \cdot \text{Technical Score}_i + 0.40 \cdot \text{Theme Macro Tilt}_k$$

This unified score drives the selection of the top 5 ETF orders while respecting all risk manager boundaries (20% single asset cap, 35% theme limit, 15% combined leveraged cap, 5% cash buffer).

---

## 9. Academic References

1. **Apel, M., & Blix Grimaldi, M. (2014).** *How Informative Are Central Bank Minutes?* Review of Economics and Statistics, 96(3), 540–556.
2. **Bennani, H., & Farvaque, E. (2018).** *Reading between the lines: How central bank communication affects financial markets.* Journal of Macroeconomics, 55, 1–15.
3. **Loughran, T., & McDonald, B. (2011).** *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.* The Journal of Finance, 66(1), 35–65.
4. **Cheng, M., & Madhavan, A. (2009).** *The Dynamics of Leveraged and Inverse Exchange-Traded Funds.* Journal of Investment Management, 7(4), 43–62.
5. **Lane, P. R. (2024).** *The practice and science of monetary policy communication: the ECB experience.* European Central Bank Speech Series.
