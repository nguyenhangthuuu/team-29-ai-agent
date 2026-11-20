# PRODUCT.md — Belfius ReBel Agentic Trader

## 1. Product Context & Objectives
- **Product Name:** Belfius ReBel Agentic Trader
- **Institutional Context:** QTEM Data Challenge (QDC) Fall 2026 Edition (Partners: **Belfius Bank** & **QuantumBlack, AI by McKinsey**).
- **Core Mission:** €10,000 multi-asset ETF portfolio management across 5 competition trading windows (Dec 7, 10, 14, 17, 21), adhering to strict MiFID II suitability, risk guardrails (cash >= 5%, single asset <= 20%, theme <= 35%, combined leverage <= 15%), and zero transaction fees.

## 2. Investment Service Models
- **ADVISE Model:** Belfius advises, client decides. Clients can challenge recommendations [C], request runner-up alternative candidates with data-backed evidence, and modify proposed allocations [M].
- **MANDATE Model:** Belfius manages the portfolio autonomously with quantitative risk oversight.

## 3. User Personas & Progressive Disclosure
- **Level 1 (Newbie / Advisory Mode):** Plain English, zero financial jargon, interactive hover/click tooltips on every indicator (Fed score, ECB score, ML probability, Sharpe, theme tilts) explaining what it means and why the order was recommended.
- **Level 2 (Factor Breakdown):** Technical factor contribution, Random Forest binary classification probabilities, and news headline causal attribution.
- **Level 3 (Institutional Quant):** Markowitz covariance matrices, 95% Historical VaR/CVaR, and Black-Litterman conviction blending.

## 4. Key Workflows
1. **[C] Challenge / Argue:** Enables client to request sub-optimal or runner-up alternative recommendations (Ranks 6 to 10) with explicit empirical trade-off evidence.
2. **[M] Discretionary Modify:** Instant dynamic rebalancing with real-time NAV, cash buffer, and constraint re-audit.
3. **Traceable News Feed:** Complete headline audit trail with latency tracking, sentiment scoring, and economic transmission rationales.
