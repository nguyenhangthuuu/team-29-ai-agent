# Real-Time Financial Headline News Methodology & Quantification Engine

## 1. Executive Summary & Objective

In quantitative asset management, macroeconomic regime transitions are frequently preceded by narrative shifts in financial media, central bank press leaks, and geopolitical developments before they manifest in closing prices. However, incorporating unstructured news into a systematic investment framework introduces three major vulnerabilities:
1. **Unbounded Hallucination & Sentiment Drift:** Relying on unconstrained generative language models produces volatile, non-reproducible sentiment scores that corrupt portfolio optimization.
2. **Temporal Decay & Lookahead Bias:** Ingestion of stale news or future headlines violates strict quantitative backtesting standards.
3. **Black-Box Opacity:** Human portfolio managers and risk committees cannot audit why an AI model favoured or penalised an asset without line-by-line provenance back to specific source articles.

To solve these challenges under the **Belfius ReBel x QuantumBlack** institutional standard, the **Macro & News Agent** (`src/agents/macro.py`) implements a deterministic, multi-source headline quantification and traceability pipeline:
- **Zero-Budget Authorized Ingestion:** Real-time stream aggregating **Source A (Google News RSS - Macro/Central Banks)**, **Source B (Yahoo Finance News - Anchor ETFs)**, and **Source C (CNBC Economy RSS)**.
- **Strict Timeframe Governance:** Clamped lookback window (default: 5 days; hard cap: 30 days) equipped with publication-to-execution latency tracking ($\Delta t = t_{\text{eval}} - t_{\text{pub}}$).
- **Two-Tier Quantification Engine:** Single-batch structured extraction via **Google Gemini 1.5 Flash** with an automated **100% deterministic offline fallback** using the **Loughran-McDonald (2011) Financial Lexicon** and 3-token negation windows.
- **Thematic ETF Sector Mapping:** Translates headline sentiment into granular tactical adjustments ($\Delta_{\text{theme}} \in [-0.30, +0.30]$) across the 50 ETF competition universe.
- **End-to-End Traceability:** Complete line-by-line auditability enabling human traders to inspect, filter, and verify every single headline collected by the system.

---

## 2. Ingestion Pipeline & Tri-Source Architecture

The headline engine restricts data acquisition strictly to three authorized, non-paywalled institutional feeds, eliminating scraping fragility and complying with zero-budget requirements ($0.00 external API cost):

```
                               TRI-SOURCE HEADLINE INGESTION
     ┌───────────────────────────────────────┼───────────────────────────────────────┐
     ▼                                       ▼                                       ▼
SOURCE A: Google News RSS             SOURCE B: Yahoo Finance Stream          SOURCE C: CNBC Economy RSS
Query: "economy inflation Fed         Anchor Tickers: SPY, QQQ, CSX5,         Editorial macroeconomic analysis
ECB interest rates tariffs"           GLD, BZ=F, ^VIX                         and Federal Reserve coverage
     │                                       │                                       │
     └───────────────────────────────────────┼───────────────────────────────────────┘
                                             ▼
                                  TIME-WINDOW FILTERING & CLAMP
                                 (Max 30 Days Lookback / Latency)
                                             ▼
                                 REGEX DEDUPLICATION BLOTTER
                                             ▼
                           PERSISTENT CACHE & AUDIT TRAIL LOGGING
                           (`data/news_headlines_cache.json`)
```

### 2.1 Feed Specifications & Ingestion Endpoints

| Feed Identifier | Provider & Scope | Focus & Asset Universe | Endpoint / Mechanism |
| :--- | :--- | :--- | :--- |
| **Source A** | **Google News RSS (Macro)** | High-level macroeconomic shifts, inflation prints, central bank leaks, tariff and trade policy | `https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en` |
| **Source B** | **Yahoo Finance Stream** | Asset-specific breaking news on benchmark ETFs (`SPY`, `QQQ`, `CSX5.L`, `GLD`, `BZ=F`, `^VIX`) | Built-in `yfinance` `Ticker.news` object |
| **Source C** | **CNBC Economy RSS** | Real-time US and European economic coverage, labor market indicators, corporate earnings tone | `https://www.cnbc.com/id/20910258/device/rss/rss.html` |

### 2.2 Deduplication & Normalization Logic
To prevent redundant headlines across syndication networks (e.g., identical AP/Reuters dispatches mirrored across multiple outlets), the engine computes a normalized alphanumeric character key for every incoming item:
```python
key = re.sub(r"[^a-zA-Z0-9 ]", "", title.lower()).strip()
```
If $\text{len}(\text{key}) \le 12$ or $\text{key} \in \text{SeenKeys}$, the item is discarded as noise or a duplicate. Filtered headlines are then sorted in strictly descending order by publication timestamp ($t_{\text{pub}}$).

### 2.3 Timeframe Window Governance & Latency Tracking
To prevent stale sentiment decay and lookahead bias:
1. **User Reference Date Enforcement:** A user or backtesting harness can supply a `reference_date`. The agent strictly excludes any headline published outside $[t_{\text{ref}}, t_{\text{eval}}]$.
2. **Hard 30-Day Maximum Lookback Clamp:** If a user requests a lookback older than 30 days, the engine clamps the window to $[t_{\text{eval}} - 30\text{ days}, t_{\text{eval}}]$ and emits a prominent `AGENT_LIMIT_NOTIFICATION` to the human-in-the-loop dashboard.
3. **News Latency Metric ($\Delta t_{\text{latency}}$):**
   $$\Delta t_{\text{latency}} = \frac{t_{\text{now}} - \max(t_{\text{pub}})}{60} \quad [\text{minutes}]$$
   This guarantees the portfolio committee knows the exact freshness of the news stream.

---

## 3. Mathematical Quantification Engine

The agent evaluates headlines through a structured two-tier hierarchy:
- **Tier 1 (Cloud / Online):** Single-batch structured extraction via Google Gemini 1.5 Flash.
- **Tier 2 (Deterministic Offline Fallback):** Lexicon-based scoring using Loughran-McDonald (2011) financial dictionaries with a 3-token negation window.

```
                           HEADLINE STREAM EVALUATION
                                       │
                         Is GEMINI_API_KEY Configured?
                                      / \
                                Yes /     \ No / Network Error
                                  /         \
                                 ▼           ▼
                      TIER 1: GEMINI FLASH   TIER 2: ACADEMIC QUANT ENGINE
                      Structured Output      Loughran-McDonald (2011) + Negation
                                 \           /
                                  \         /
                                   ▼       ▼
                       STANDARDIZED HEADLINE METRICS
                       • Overall Sentiment: S in [-1.0, +1.0]
                       • News Regime: PANIC / EXPANSION / UNCERTAINTY / NEUTRAL
                       • Tactical Thematic Tilts: Delta_theme in [-0.30, +0.30]
                       • Top 2-3 Catalysts Identified
```

### 3.1 Tokenization & Negation Window
In the deterministic engine, raw headlines are tokenized into word and bigram sequences:
$$W = \{w_1, w_2, \dots, w_n\}$$
To prevent false-positive sentiment classification (e.g., *"not improving"*, *"no growth"*, or failure expressions such as *"fails to provide adequate confidence"*), the engine inspects a 4-token preceding window for negation and inability terms:
$$\text{NEGATION} = \{\text{"not"}, \text{"no"}, \text{"never"}, \text{"hardly"}, \text{"scarcely"}, \text{"without"}, \text{"little"}, \text{"fail"}, \text{"fails"}, \text{"failed"}, \text{"failing"}, \text{"failure"}, \text{"unable"}, \text{"lack"}, \text{"lacks"}, \text{"lacking"}\}$$
If $w_i \in \text{Positive Lexicon}$ and $\exists \, w_j \in \text{NEGATION}$ for $j \in [\max(0, i-4), i)$, the token is inverted and scored as a **Negative** occurrence. Conversely, if a negative token (e.g. *"not declining"*) is preceded by negation, it is inverted to a **Positive** occurrence.

In addition, downward market movement terms are explicitly cataloged under $\text{LM\_NEGATIVE}$:
$$\text{MARKET\_DECLINE} = \{\text{"lower"}, \text{"fall"}, \text{"falls"}, \text{"fell"}, \text{"sink"}, \text{"sinks"}, \text{"sank"}, \text{"selloff"}, \text{"sell-off"}, \text{"slide"}, \text{"slides"}, \text{"slid"}, \text{"retreat"}, \text{"retreats"}\}$$

### 3.2 Sentiment Polarity Formulation
Using the **Loughran-McDonald (2011)** financial dictionary:
- $N_{\text{pos}} = \sum \mathbb{I}(w \in \text{LM\_POSITIVE})$
- $N_{\text{neg}} = \sum \mathbb{I}(w \in \text{LM\_NEGATIVE}) + \sum \mathbb{I}(w \in \{\text{"tariff"}, \text{"war"}, \text{"crisis"}, \text{"plunge"}\})$

The aggregate headline sentiment score $S_{\text{headline}}$ is calculated as:
$$S_{\text{headline}} = \text{clip}\left(\frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{pos}} + N_{\text{neg}} + 10^{-6}}, \, -1.0, \, +1.0\right)$$

### 3.3 News Regime Classification
The continuous score $S_{\text{headline}}$ is mapped onto discrete macroeconomic news regimes:

| Sentiment Score ($S_{\text{headline}}$) | News Regime Label | Macroeconomic Interpretation |
| :--- | :--- | :--- |
| **$S_{\text{headline}} < -0.25$** | `HEADLINE_PANIC` | Severe market distress, geopolitical escalation, or acute recession fears. Triggers cash preservation and gold hedges. |
| **$S_{\text{headline}} > +0.25$** | `RISK_EXPANSION` | Broad risk-on optimism, earnings momentum, and economic expansion. Enables higher equity allocation and selective alpha. |
| **$N_{\text{pos}} + N_{\text{neg}} = 0$** | `NEUTRAL` | Absence of directional bias; market driven primarily by technical momentum. |
| **$-0.25 \le S_{\text{headline}} \le +0.25$** | `POLICY_UNCERTAINTY` | Conflicting signals (e.g., resilient growth paired with sticky inflation or tariff disputes). |

---

## 4. Thematic Sector Mapping & Portfolio Integration

Headlines rarely impact all asset classes uniformly. An energy crisis damages transport and retail equities while providing a windfall for oil producers; rising interest rate rhetoric boosts bank net interest margins while compressing growth equities.

### 4.1 ETF Thematic Keyword Sensitivity Matrix

| Thematic Cluster | Associated QDC ETFs | Target Ingestion Keywords |
| :--- | :--- | :--- |
| **Technology** | `NASDAQ`, `SEMIS`, `CYBERSEC`, `ROBOTICS`, `QUANTUM` | `ai`, `tech`, `semiconductor`, `chips`, `software`, `nvidia`, `cloud`, `capex` |
| **European Equities** | `EUROPE 600`, `EURO STOXX 50`, `BEL 20`, `IBEX 35` | `europe`, `eurozone`, `ecb`, `germany`, `france`, `dax`, `stoxx`, `euro` |
| **Emerging Markets** | `EMMKT`, `CHINA LC`, `INDIA`, `BRAZIL`, `CHINA A` | `china`, `tariff`, `trade`, `emerging`, `india`, `brazil`, `yuan`, `asia` |
| **Energy & Commodities** | `WORLD ENERGY`, `OIL & GAS E&P`, `URANIUM`, `CLEAN ENERGY` | `oil`, `opec`, `energy`, `crude`, `brent`, `gas`, `uranium`, `drilling` |
| **Precious Metals** | `GOLD`, `GOLD MINERS`, `SILVER MINERS` | `gold`, `silver`, `safe haven`, `bullion`, `central bank buying` |
| **Defensive Sectors** | `DEFENSE`, `HEALTH CARE`, `STAPLES`, `WORLD MIN VOL` | `defense`, `nato`, `military`, `healthcare`, `pharma`, `staples`, `utilities` |
| **Digital Assets** | `BITCOIN`, `CRYPTO EQUITY` | `bitcoin`, `crypto`, `blockchain`, `btc`, `etf inflows`, `ethereum` |

### 4.2 Tactical Headline Tilts ($\Delta_{\text{theme}}$)
For each headline $h$, let $\text{Impact}(h) = N_{\text{pos}}(h) - N_{\text{neg}}(h)$. If headline $h$ mentions keywords belonging to theme $T$:
$$\text{RawDelta}_T = \sum_{h \in H} \text{sign}(\text{Impact}(h)) \cdot 0.08$$
The final tactical delta is bounded to prevent news overreaction:
$$\Delta_{\text{headline}, T} = \text{clip}(\text{RawDelta}_T, \, -0.30, \, +0.30)$$

### 4.3 Synthesis into the Multi-Factor Conviction Model
In `PortfolioManagerAgent` (`src/agents/portfolio.py`), the tactical headline delta is integrated directly into the multi-factor scoring formula:
$$\text{Conviction}_i = 0.60 \cdot \text{TechnicalScore}_i + 0.40 \cdot \left(\text{ThemeMacroTilt}_{\text{theme}(i)} + \Delta_{\text{headline}, \text{theme}(i)}\right)$$
where $\text{TechnicalScore}_i = 2 \cdot P_{\text{RF}}(i) - 1.0$ is derived from the Random Forest model.

---

## 5. Headline Traceability & Human Audit Architecture

To provide institutional transparency, every headline processed by the MAS is preserved with complete metadata. A dedicated audit module (`src/pipeline/headline_audit.py`) provides human traders with three ways to trace and verify headlines:
1. **Programmatic Python API:** `get_headline_audit_trail()` for data science analysis.
2. **Interactive Command-Line Blotter:** `python -m src.pipeline.headline_audit` or `python run_mas.py --inspect-headlines`.
3. **Structured Audit Persistence:** Automatically synchronized in `data/news_headlines_cache.json` and `logs/news_headlines_audit.jsonl`.

### 5.1 Headline Audit Record Schema
Every captured headline contains the following immutable fields:
```json
{
  "headline_id": "h_440d19e1",
  "title": "S&P 500, Nasdaq, Dow End Lower As Anthropic Trading Debut Fails To Provide Adequate Confidence —  ANT, META, AAPL, GME In Focus",
  "source": "Source B: Yahoo Finance (SPY)",
  "published_at": "2026-11-19T23:01:05+00:00",
  "collected_at": "2026-11-20T11:00:00+00:00",
  "latency_minutes": 718.9,
  "sentiment_score": -0.80,
  "sentiment_label": "NEGATIVE",
  "primary_driver": "MARKET_SELLOFF",
  "detected_themes": ["Technology"],
  "economic_rationale": "Negative momentum or failed catalyst created risk-off downward pressure on equities."
}
```

### 5.2 Raw Headline CSV Export Trigger & Lookback Window Governance

To support institutional data auditability and qualitative review without modifying or truncating the original text, a dedicated trigger (`--export-headlines-csv`) is provided across the system. 

#### Lookback Window Governance:
- **Default Duration:** **5 calendar days** (`lookback_days = 5`).
- **User Discretion:** The user can configure the lookback period to any desired horizon via `--lookback-days <N>`.
- **Maximum Ceiling:** Hard clamp at **30 calendar days** (`min(max(1, lookback_days), 30)`). If a user specifies a lookback greater than 30 days, the engine automatically clamps it to 30 days and logs an institutional governance notice to prevent narrative decay.

#### CLI Invocations:
```bash
# 1. Trigger export with default 5-day lookback:
python run_mas.py --export-headlines-csv

# 2. Trigger export with user-selected lookback (e.g. 14 days):
python run_mas.py --export-headlines-csv --lookback-days 14

# 3. Trigger export with maximum allowable lookback (30 days):
python run_mas.py --export-headlines-csv --lookback-days 30

# 4. Standalone headline audit tool with custom lookback and output path:
python -m src.pipeline.headline_audit --export-csv reports/news_10d.csv --lookback-days 10
```

#### CSV Schema Specification:
| Column Name | Type | Description | Content Guarantee |
| :--- | :--- | :--- | :--- |
| `headline_id` | `str` | Deterministic 8-char MD5 hash | `h_` + MD5 of `source + title` |
| `source` | `str` | Full scraping source name | Preserved exactly as ingested |
| `title` | `str` | **Full Headline Text** | **100% Unshortened & Unmodified** |
| `published_at` | `str` | ISO 8601 UTC timestamp | Exact publication time from RSS/API |
| `latency_minutes` | `float` | Minutes elapsed until evaluation | Accurate to 1 decimal place |
| `sentiment_score` | `float` | Quantified polarity $\in [-1.0, +1.0]$ | Contextual LLM valuation |
| `sentiment_label` | `str` | `POSITIVE`, `NEGATIVE`, or `NEUTRAL` | Categorical classification |
| `primary_driver` | `str` | Identified catalyst driver | e.g. `MARKET_SELLOFF`, `MONETARY_POLICY` |
| `detected_themes` | `str` | Comma-delimited theme tags | Mapped ETF thematic clusters |
| `economic_rationale` | `str` | Transmission explanation | Full economic reasoning sentence |

---

## 6. Summary of Governance Constraints

| Parameter | Value / Rule | Rationale |
| :--- | :--- | :--- |
| **Authorized Feeds** | Sources A, B, C Only | Eliminates paywall fragility and maintains zero API expense. |
| **Max Lookback Window** | 30 Calendar Days | Prevents obsolete narrative bias from skewing present conviction. |
| **Default Window** | 5 Calendar Days | Focuses tactical allocation on the freshest market catalysts. |
| **Negation Window** | 3 Tokens Lookback | Eliminates inverted polarity errors (e.g. *"not expanding"*). |
| **Max Thematic Delta** | $\pm 0.30$ | Prevents single-day news spikes from overpowering long-term technical trends. |
| **Audit Persistence** | Parquet / JSONL Snapshot | Guarantees compliance traceability for risk and audit committees. |
