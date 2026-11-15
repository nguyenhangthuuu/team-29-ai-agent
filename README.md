# ReBel Agentic AI Investment Engine
### QTEM Data Challenge (QDC) 2026–2027 | Team 29

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Belfius](https://img.shields.io/badge/Partner-Belfius%20Bank-red.svg)](https://www.belfius.be)
[![QuantumBlack](https://img.shields.io/badge/Partner-QuantumBlack%20%7C%20McKinsey-black.svg)](https://www.mckinsey.com/capabilities/quantumblack)
[![Status](https://img.shields.io/badge/Status-Task%201--3%20Complete-brightgreen.svg)]()

---

## Executive Summary

This repository contains the end-to-end quantitative codebase developed by **Team 29** for the **QTEM Data Challenge (QDC) Fall 2026 Edition**, co-organized with **Belfius Bank**.

The challenge addresses a fundamental inflection point for **ReBel**, Belfius’ digital investment platform.

Our repo tackles the **Track A (Quantitative AI Engine)** of the challenge. Hereby, we create a state-of-the-art **LangGraph multi-agent pipeline** that autonomously analyzes a universe of 50 UCITS ETFs across technical, macro, and risk dimensions to manage a €10,000 live simulation portfolio while enforcing strict human-in-the-loop governance.

---

## Agentic AI Architecture (QuantumBlack Blueprint)

Following QuantumBlack's reference architecture, we decoupled quantitative computation (deterministic Python tools) from qualitative reasoning and synthesis (LLM agents).

```
                            ┌────────────────────────┐
                            │      Data Agent        │ (Deterministic Tool)
                            │ Ingests 50 ETFs in EUR │  Prices, Returns, Vol, Drawdown
                            └───────────┬────────────┘
                                        │
                       ┌────────────────┴────────────────┐
                       ▼                                 ▼
          ┌────────────────────────┐        ┌────────────────────────┐
          │ Technical Signal Agent │        │   Macro & News Agent   │
          │ EMA20/SMA50, RSI-14,   │        │ Regime detection, Fed/ │
          │ 21d Momentum, Bands    │        │ ECB rates, Geopolitics │
          └────────────┬───────────┘        └────────────┬───────────┘
                       │                                 │
                       └────────────────┬────────────────┘
                                        ▼
                            ┌────────────────────────┐
                            │   Risk Manager Agent   │ (Hard Governance Constraints)
                            │ • Max single asset: 20%│ • Min cash buffer: 5%
                            │ • Max leveraged: 15%   │ • Leveraged single cap: 8%
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │ Portfolio Manager Node │ (High-Level Synthesis)
                            │ Proposes max 5 orders  │  Allocates €10k capital &
                            │ with structured rationale drafts investment memo
                            └───────────┬────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │   Human-in-the-Loop    │ (Audit Checkpoint)
                            │  Interactive Approval  │  [A] Approve | [M] Override | [R] Reject
                            │  & Submission Logger   │  Saved to logs/qdc_audit_trail.jsonl
                            └────────────────────────┘
```

### Agent Roles & Guardrails

| Agent | Module | Engine Type | Core Functionality |
| :--- | :--- | :--- | :--- |
| **Data Agent** | `src/pipeline/` | Pure Python / Pandas | Ingests 3–5 years of daily data for 50 ETFs, converts USD listings to EUR via `EURUSD=X`, computes rolling volatility, drawdowns, and 50x50 asset correlations. |
| **Technical Agent** | `src/agents/technical.py` | Quantitative Math | Generates composite scores ($-1.0$ to $+1.0$) based on EMA20/SMA50 crossover, Wilder RSI-14, 21d/63d rate of change vs. `WORLD` benchmark, and Bollinger Band breakouts. |
| **Macro Agent** | `src/orchestration/graph.py` | LLM / Rule-based | Analyzes macroeconomic regimes, interest rate outlooks, and commodity shocks. Supports dual-engine mode (OpenAI LLM when API key is set, deterministic fallback when offline). |
| **Risk Governor** | `src/agents/risk.py` | Hard Constraints | Enforces strict risk parameters: single ETF limit $\le 20\%$, combined leveraged limit $\le 15\%$, single leveraged limit $\le 8\%$, cash buffer $\ge 5\%$, thematic concentration $\le 35\%$. Generates auto-rebalanced compliant allocations. |
| **Portfolio Manager** | `src/orchestration/graph.py` | LLM / Synthesis | Selects up to 5 discrete orders (respecting the official QDC trade cap), sizes positions in EUR, and drafts an executive memo justifying every allocation. |
| **Human Approver** | `src/orchestration/graph.py` | CLI / Interactive HITL | Pauses execution, presents the proposed orders and risk critique to the human team, logs decisions, overrides, and final rationales into an immutable audit trail. |

---

## 50 ETF Universe

The competition universe consists of 50 UCITS ETFs across 20 distinct themes:
* **Benchmarks & Equities:** `WORLD` (MSCI World), `S&P 500`, `EUROPE 600`, `EURO STOXX 50`, `IBEX 35`, `BEL 20` (Home Market).
* **Factors & Tech:** `WORLD SMALL CAP`, `WORLD MIN VOL`, `DIVIDEND LEADERS`, `NASDAQ`, `SEMIS`, `CYBERSEC`, `BIOTECH`, `ROBOTICS`, `SPACE`, `QUANTUM`.
* **Emerging Markets:** `EMMKT`, `CHINA LC`, `INDIA`, `KOREA`, `BRAZIL`, `CHINA A`.
* **Energy & Commodities:** `WORLD ENERGY`, `OIL & GAS E&P`, `URANIUM`, `CLEAN ENERGY`, `HYDROGEN`, `COPPER MINERS`, `RARE EARTHS`, `GOLD`, `GOLD MINERS`, `SILVER MINERS`.
* **Fixed Income & Cash:** `HIGHY BOND`, `US TREASURY 20Y`, `GLOBAL AGG BOND`, `EUR GOVT BOND`, `OVERNIGHT CASH`.
* **Defensive & Digital Assets:** `DEFENSE`, `STAPLES`, `BITCOIN`, `CRYPTO EQUITY`.
* **Leveraged & Inverse Products (Daily Rebalanced):** `STOXX SHORT (-2x)`, `COPPER LVG (3x)`, `PALLADIUM LVG (2x)`, `NASDAQ 3X (3x)`, `S&P 500 SHORT (-1x)`, `EURO STOXX 2X (2x)`.

> [!WARNING]
> **Leveraged Volatility Decay:** Leveraged and inverse ETFs rebalance daily and suffer severe mathematical compounding erosion in choppy markets. The Risk Agent strictly limits holding horizons and position sizing on these assets.

---

## Repository Structure

```text
team-29-ai-agent/
├── data/                                 # Cleaned Parquet & CSV price files (Git-ignored)
│   ├── etf_prices_eur.parquet            # 3-year normalized EUR price series for 50 ETFs
│   ├── etf_returns.parquet               # Daily return matrix
│   ├── etf_volatility.parquet            # 60-day rolling annualized volatility
│   └── etf_correlation_matrix.csv        # 50x50 cross-asset correlation matrix
├── logs/                                 # Audit trails & prompt logs (Git-ignored)
│   └── qdc_audit_trail.jsonl             # Immutable JSONL log of all agent runs & human decisions
├── notebooks/                            # Exploratory data analysis & backtesting
├── src/                                  # Core Python source package
│   ├── pipeline/                         # Task 1: Data Ingestion & Features
│   │   ├── __init__.py
│   │   ├── etf_universe.py               # 50 ETF metadata, ISINs, and Yahoo symbols
│   │   └── data_loader.py                # Download, FX conversion, and quant feature engine
│   ├── agents/                           # Task 2: Deterministic Quant & Risk Agents
│   │   ├── __init__.py
│   │   ├── technical.py                  # Trend, RSI, Momentum, and Breakout scoring
│   │   └── risk.py                       # Risk limits, concentration caps, and auto-rebalancer
│   └── orchestration/                    # Task 3: LangGraph Multi-Agent Engine
│       ├── __init__.py
│       ├── state.py                      # Pydantic schemas and AgentState TypedDict
│       ├── graph.py                      # LangGraph state machine & interactive HITL runner
│       └── logger.py                     # Structured JSONL event logger
├── .env.example                          # Environment variable template
├── .gitignore                            # Protects secrets, large datasets, and logs
├── README.md                             # Project documentation
└── requirements.txt                      # Frozen Python dependencies
```

---

## Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone this repository and create a Python virtual environment:

```bash
# Navigate to the project directory
cd team-29-ai-agent

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` (optional: add API keys for live LLM inference):

```bash
# Optional: Add OpenAI API key for LLM-powered Macro & Portfolio synthesis
# If omitted, the engine uses deterministic fallback logic with zero cost.
OPENAI_API_KEY="your-api-key-here"
```

### 3. Step 1: Run the Data Pipeline
Downloads 3 years of daily data for all 50 ETFs, converts prices to EUR, and computes returns, volatility, drawdowns, and correlations:

```bash
python -m src.pipeline.data_loader
```
*Output: Saves Parquet and CSV files into `./data/`.*

### 4. Step 2: Test Technical & Risk Agents
Computes technical momentum scores and tests risk guardrails:

```bash
python -c "
import pandas as pd
from src.agents.technical import evaluate_all_etfs
from src.agents.risk import RiskManager

prices = pd.read_parquet('data/etf_prices_eur.parquet')
signals = evaluate_all_etfs(prices)
print('Top 3 Bullish:', sorted(signals.values(), key=lambda s: s.composite_score, reverse=True)[:3])
"
```

### 5. Step 3: Run the Full Multi-Agent Graph with HITL 
Executes the full pipeline and prompts for human approval in your terminal:

```bash
python -m src.orchestration.graph
```

```text
======================================================================
🤖 REBEL AGENTIC TRADER — HUMAN-IN-THE-LOOP APPROVAL CHECKPOINT
======================================================================
Trading Date: 2026-11-13 | Total Capital: €10,000.00

--- EXECUTIVE INVESTMENT MEMO ---
Investment Committee Proposal for 2026-11-13:
Macro Environment: Modest growth with disinflation. European blue chips and defensive healthcare lead...
Strategy: Deploy €10,000.00 into 5 top-ranked momentum & defensive leaders (EUROPE 600, DIVIDEND LEADERS, BIOTECH, HEALTH CARE, DEFENSE). Retain 10.0% cash buffer.

--- PROPOSED TRADING ORDERS (MAX 5) ---
  [1] BUY EUROPE 600       | Weight: 18.0% | Amount: €1,800.00
  [2] BUY DIVIDEND LEADERS | Weight: 18.0% | Amount: €1,800.00
  [3] BUY BIOTECH          | Weight: 18.0% | Amount: €1,800.00
  [4] BUY HEALTH CARE      | Weight: 18.0% | Amount: €1,800.00
  [5] BUY DEFENSE          | Weight: 18.0% | Amount: €1,800.00

--- RISK AUDIT REPORT: ✅ PASSED: Portfolio satisfies all constraints ---
Cash Reserve: 10.0% (Min: 5.0%) | Leveraged Exposure: 0.0% (Max: 15.0%)

Enter Action: [A] Approve Orders | [M] Modify Notes | [R] Reject: 
```

---

## Key Competition Milestones

| Date | Milestone | Deliverable / Focus |
| :--- | :--- | :--- |
| **Oct 12, 2026** | **QDC Kick-off** | Challenge introduced by QTEM, Belfius, and QuantumBlack. |
| **Nov 22, 2026** | **Mid-Term Report Delivery** | 8-page PDF report covering research question, data pipeline, and initial AI architecture. |
| **Dec 7–23, 2026** | **Live Trading Window** | 5 fixed submission dates (Dec 7, 10, 14, 17, 21) via official Belfius form. |
| **Jan 16, 2027** | **Final Delivery** | 25-page main report, 15-slide pitch deck, executive summary, code repository, and trade audit logs. |
| **Spring 2027** | **Professor Oral Defense** | Individual virtual technical Q&A with academic professors. |
| **Oct 2027** | **QTEM Annual Event Final** | Top 2 finalist teams present to corporate executives and grand jury. |

---

## Responsible AI & Governance

In compliance with QTEM academic standards and Belfius AM policies:
1. **No Unsupervised Trading:** AI agents generate proposals; human team members must sign off on every transaction.
2. **Deterministic Grounding:** Financial calculations (drawdowns, returns, volatilities) are computed by mathematical libraries, preventing LLM arithmetic hallucination.
3. **Cost Optimization:** Using small, efficient models for data extraction and high-reasoning models only for executive synthesis, tracking token usage per trade in compliance with commercial feasibility expectations.
4. **Audit Trail:** Every agent prompt, response, risk critique, and human override is immutably recorded in `logs/qdc_audit_trail.jsonl`.
