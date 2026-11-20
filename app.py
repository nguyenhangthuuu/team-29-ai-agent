# app.py
"""
Belfius ReBel x QuantumBlack Agentic Trader — Interactive Investment Dashboard
QTEM Data Challenge (QDC) Fall 2026 Edition | Team 29
Institutional Multi-Agent System (MAS) for ETF Portfolio Allocation.

Architecture & Governance:
  - MiFID II Investor Risk Profiles: Conservative, Moderate, Aggressive
  - Investment Service Models: Advise vs Mandate
  - Progressive Disclosure: Level 1 (Executive), Level 2 (Factor Breakdown), Level 3 (Institutional Quant)
  - Structured Discretionary Order Modification Interface
  - Continuous Machine Learning Self-Updating Engine (Random Forest Walk-Forward)
  - Mark-to-Market Portfolio Ledger Accounting & Immutable Parquet/JSONL Persistence
"""

import os
import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.pipeline.etf_universe import ETF_UNIVERSE, ETFMeta
from src.pipeline.portfolio_ledger import PortfolioLedger, LedgerSnapshot
from src.agents.technical import evaluate_all_etfs, retrain_if_new_data, get_rf_model_bundle, FEATURE_COLS
from src.agents.macro import evaluate_macro_agent
from src.agents.risk import RiskManager, RISK_PROFILES
from src.agents.portfolio import PortfolioManagerAgent
from src.agents.human_loop import HumanInTheLoopAgent
from src.orchestration.state import AgentState

# ==============================================================================
# PAGE CONFIGURATION & INSTITUTIONAL THEME
# ==============================================================================
st.set_page_config(
    page_title="Belfius ReBel | Agentic Trader",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Institutional Belfius ReBel Typography & Layout
st.markdown("""
<style>
    /* Sharp, elegant, minimal typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        letter-spacing: -0.01em;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #B40036;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.90rem;
        color: #475569;
        margin-bottom: 1.2rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        max-width: 100%;
    }
    /* Tabular numerals for financial alignment */
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"], td, th {
        font-variant-numeric: tabular-nums !important;
        font-feature-settings: 'tnum' !important;
    }
    /* Strict text boundary enforcement - zero overflow */
    * {
        box-sizing: border-box;
    }
    div, p, span, h1, h2, h3, h4, th, td {
        overflow-wrap: break-word;
        word-break: break-word;
    }
    .card-metric {
        background: #F8FAFC;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        padding: 10px 8px;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        overflow: hidden;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .badge-risk-on { background-color: #DEF7EC; color: #03543F; }
    .badge-risk-off { background-color: #FDE8E8; color: #9B1C1C; }
    .badge-stagflation { background-color: #FEF08A; color: #854D0E; }
    .badge-neutral { background-color: #E2E8F0; color: #334155; }
    .badge-conservative { background-color: #E0F2FE; color: #0369A1; }
    .badge-moderate { background-color: #FEF3C7; color: #B45309; }
    .badge-aggressive { background-color: #FCE7F3; color: #B40036; }
    .badge-service { background-color: #FDF2F8; color: #B40036; border: 1px solid #FBCFE8; }
    .info-panel {
        background-color: #FAFAFA;
        border-left: 4px solid #B40036;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 14px;
        color: #1E293B;
        font-size: 0.9rem;
    }
    .callout-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 14px;
        color: #14532D;
        font-size: 0.9rem;
    }
    .audit-box {
        background-color: #FDF2F8;
        border-left: 4px solid #B40036;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 14px;
        color: #831843;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "ledger" not in st.session_state:
    st.session_state.ledger = PortfolioLedger()
if "last_proposal" not in st.session_state:
    st.session_state.last_proposal = None
if "last_risk_eval" not in st.session_state:
    st.session_state.last_risk_eval = None
if "last_signals" not in st.session_state:
    st.session_state.last_signals = None
if "last_macro" not in st.session_state:
    st.session_state.last_macro = None
if "challenge_history" not in st.session_state:
    st.session_state.challenge_history = []
if "custom_modified_orders" not in st.session_state:
    st.session_state.custom_modified_orders = None
if "execution_msg" not in st.session_state:
    st.session_state.execution_msg = None

# Load Market Dataset
@st.cache_data(ttl=600)
def load_market_data():
    price_path = "data/etf_prices_eur.parquet"
    if os.path.exists(price_path):
        df = pd.read_parquet(price_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()

prices_df = load_market_data()
available_dates = [str(d.date()) for d in prices_df.index] if not prices_df.empty else ["2026-11-13"]

# ==============================================================================
# SIDEBAR CONTROLS & GOVERNANCE CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1C0A12 0%, #300619 100%); padding: 14px; border-radius: 6px; border: 1px solid #570D2C; margin-bottom: 15px; text-align: center; color: white;">
        <div style="font-size: 1.3rem; font-weight: 800; letter-spacing: 0.5px; color: #FFFFFF;">BELFIUS <span style="color:#B40036;">ReBel</span></div>
        <div style="font-size: 0.75rem; color: #E9D5FF; margin-top: 2px;">Institutional Multi-Agent Copilot</div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("QuantumBlack x Belfius Bank | QDC Fall 2026 (Team 29)")
    st.markdown("---")

    # 1. Investor Risk Profile (MiFID II Framework)
    st.markdown("#### 1. Investor Risk Profile (MiFID II)")
    risk_choice = st.radio(
        "Select your risk mandate:",
        options=["MODERATE", "CONSERVATIVE", "AGGRESSIVE"],
        index=0,
        format_func=lambda x: {
            "CONSERVATIVE": "Conservative (Capital Preservation)",
            "MODERATE": "Moderate (Balanced Alpha - Default)",
            "AGGRESSIVE": "Aggressive (High Growth / Alpha)"
        }[x]
    )

    profile_meta = RISK_PROFILES[risk_choice]
    st.markdown(
        f"<div class='info-panel'>"
        f"<strong>Policy Mandate Limits:</strong><br>"
        f"• Minimum Cash Buffer: <strong>{profile_meta['min_cash_buffer']*100:.0f}%</strong><br>"
        f"• Single Asset Cap: <strong>{profile_meta['max_single_weight']*100:.0f}%</strong><br>"
        f"• Combined Leverage Cap: <strong>{profile_meta['max_leveraged_total']*100:.0f}%</strong>"
        f"</div>",
        unsafe_allow_html=True
    )

    # 2. Investment Service Model Selector
    st.markdown("#### 2. Investment Service Model")
    service_model = st.selectbox(
        "Select service tier:",
        options=["ADVISE", "MANDATE"],
        format_func=lambda x: "Advise (Belfius Recommends, Client Decides)" if x == "ADVISE" else "Mandate (Belfius Discretionary Management)"
    )

    # 3. Competition Trading Window Date
    st.markdown("#### 3. Competition Trading Window")
    selected_date = st.selectbox(
        "Evaluation Close Date:",
        options=reversed(available_dates),
        index=0
    )

    # 4. Self-Updating ML Configuration
    st.markdown("#### 4. Machine Learning Governance")
    auto_ml_check = st.checkbox("Auto-Retrain if Fresh Data Available", value=True)
    force_retrain = st.button("Trigger Full Walk-Forward Retraining")

    if force_retrain:
        with st.spinner("Retraining Random Forest model across 50 ETF panel..."):
            retrained, msg, _ = retrain_if_new_data(prices_df, force=True)
            st.success(msg)

    st.markdown("---")
    run_btn = st.button("Run Multi-Agent Analysis", type="primary", use_container_width=True)
    
    if st.button("Reset Portfolio to Initial €10,000 Cash", use_container_width=True):
        st.session_state.ledger.reset()
        st.session_state.last_proposal = None
        st.session_state.custom_modified_orders = None
        st.session_state.challenge_history = []
        st.session_state.execution_msg = "Portfolio reset to initial €10,000 cash balance (0.00% cumulative return)."
        st.rerun()

# ==============================================================================
# PIPELINE EXECUTION & SYNCHRONIZATION
# ==============================================================================
ledger = st.session_state.ledger
current_prices = {col: float(prices_df[col].dropna().iloc[-1]) for col in prices_df.columns} if not prices_df.empty else {}
current_nav = ledger.mark_to_market(current_prices) if current_prices else 10000.0

if run_btn or st.session_state.last_proposal is None:
    with st.spinner("Executing Multi-Agent System (Technical, Macro, Risk, Portfolio Managers)..."):
        # Step 3: Self-updating ML check
        if auto_ml_check and not prices_df.empty:
            retrained, ml_msg, _ = retrain_if_new_data(prices_df, force=False)

        # Technical Agent (Random Forest probabilities)
        tech_signals = evaluate_all_etfs(prices_df)
        st.session_state.last_signals = tech_signals

        # Macro Agent (Central bank policy spread & headline news stream)
        macro_briefing = evaluate_macro_agent(trading_date=selected_date)
        st.session_state.last_macro = macro_briefing

        # Risk & Portfolio Manager Agents
        risk_engine = RiskManager(risk_profile=risk_choice)
        pm_agent = PortfolioManagerAgent(risk_manager=risk_engine)

        proposal, risk_eval = pm_agent.construct_portfolio(
            trading_date=selected_date,
            tech_signals={t: s.model_dump() for t, s in tech_signals.items()},
            macro_briefing=macro_briefing,
            starting_capital_eur=current_nav,
            risk_profile=risk_choice,
            user_arguments=None,
            current_holdings=ledger.positions
        )

        st.session_state.last_proposal = proposal
        st.session_state.last_risk_eval = risk_eval
        st.session_state.custom_modified_orders = None

# ==============================================================================
# HEADER & TOP-LEVEL KPI METRICS
# ==============================================================================
proposal = st.session_state.last_proposal
risk_eval = st.session_state.last_risk_eval
macro_obj = st.session_state.last_macro
signals = st.session_state.last_signals
ledger_summary = ledger.get_summary_report()

st.markdown('<div class="main-header">Belfius ReBel Agentic Trader</div>', unsafe_allow_html=True)

regime_badge = {
    "RISK_ON": '<span class="badge-pill badge-risk-on">Regime: Risk-On (Growth)</span>',
    "RISK_OFF": '<span class="badge-pill badge-risk-off">Regime: Risk-Off (Defensive)</span>',
    "STAGFLATION": '<span class="badge-pill badge-stagflation">Regime: Stagflation (Commodities Tilt)</span>',
    "NEUTRAL": '<span class="badge-pill badge-neutral">Regime: Neutral</span>'
}.get(macro_obj.macro_regime if macro_obj else "NEUTRAL", '<span class="badge-pill badge-neutral">Neutral</span>')

profile_badge = {
    "CONSERVATIVE": '<span class="badge-pill badge-conservative">Profile: Conservative</span>',
    "MODERATE": '<span class="badge-pill badge-moderate">Profile: Moderate</span>',
    "AGGRESSIVE": '<span class="badge-pill badge-aggressive">Profile: Aggressive</span>'
}.get(risk_choice, '<span class="badge-pill badge-moderate">Profile: Moderate</span>')

service_badge = (
    '<span class="badge-pill badge-service">Service: Advise Model</span>'
    if service_model == "ADVISE" else
    '<span class="badge-pill badge-service">Service: Discretionary Mandate</span>'
)

st.markdown(f'<div class="sub-header">{profile_badge} &nbsp;|&nbsp; {service_badge} &nbsp;|&nbsp; {regime_badge} &nbsp;|&nbsp; Trading Date: <strong>{selected_date}</strong></div>', unsafe_allow_html=True)

if st.session_state.execution_msg:
    st.success(st.session_state.execution_msg)

# Top KPI Summary Row with Newbie-Friendly Tooltips
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5, gap="small")
kpi1.metric("PORTFOLIO NAV", f"€{ledger_summary['current_nav']:,.2f}", help="Real-time Mark-to-Market net asset value (Cash + Current ETF value). Initial capital is €10,000.00.")
kpi2.metric("CASH RESERVE", f"€{ledger_summary['cash_balance']:,.2f}", f"{ledger_summary['cash_reserve_pct']:.1f}% of total", help="Uninvested cash held for liquidity. Regulatory mandate requires minimum 5.0% cash.")
kpi3.metric("CUMULATIVE RETURN", f"{ledger_summary['total_cumulative_return_pct']:+.2f}%", help="Total percentage return since portfolio inception: (Current NAV - €10,000) / €10,000.")
kpi4.metric("1D 95% HISTORICAL VAR", f"{var_val:.2f}%", help="Value-at-Risk (95% confidence): Maximum anticipated daily portfolio loss under normal historical market conditions.")
kpi5.metric("RISK GATE", status_label, help="Automated compliance gate verifying single asset <=20%, theme <=35%, combined leverage <=15%.")

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# PROGRESSIVE DISCLOSURE TABS INTERFACE
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Recommendations & Decision Copilot",
    "Portfolio Accounting & Ledger",
    "Investor Academy & Metric Definitions",
    "Multi-Agent Architecture & Audit Trail"
])

# ------------------------------------------------------------------------------
# TAB 1: RECOMMENDATIONS & INTERACTIVE COPILOT
# ------------------------------------------------------------------------------
with tab1:
    # Level 1: Executive Summary & Newbie Guidance
    st.markdown("### Executive Investment Proposal")
    st.markdown(
        f"<div class='info-panel'>"
        f"{proposal.executive_memo if proposal else 'Initializing portfolio recommendation...'}"
        f"</div>",
        unsafe_allow_html=True
    )

    with st.expander("Newbie-Friendly Index & Metric Guide (Click to Learn More)", expanded=False):
        g_c1, g_c2 = st.columns(2)
        with g_c1:
            st.markdown("""
            **Central Bank Policy Indices:**
            - **Fed Hawkish Score (-1.0 to +1.0):** Measures US Federal Reserve policy stance. Positive = rates high to fight inflation; Negative = rate cuts to stimulate growth.
            - **ECB Hawkish Score (-1.0 to +1.0):** European Central Bank policy direction.
            - **Transatlantic Spread:** Difference between US and European policy rates. High spread drives currency flows and banking margins.
            - **Headline News Sentiment (-1.0 to +1.0):** Real-time quantified narrative stream tilting ETF sector allocations.
            """)
        with g_c2:
            st.markdown("""
            **Quantitative & Machine Learning Metrics:**
            - **RF Win Probability (0% to 100%):** Random Forest machine learning probability that the ETF outperforms cash over the 3-day holding horizon.
            - **1D 95% Historical VaR:** Maximum expected daily loss at 95% confidence under normal market conditions.
            - **Conviction Score:** Blends 60% machine learning probability with 40% macroeconomic and news narrative tilts.
            """)

    st.markdown("### Recommended Asset Allocation (Max 5 Orders / Day)")
    st.caption("Orders formulated via Random Forest binary classification probabilities, constrained by MiFID II risk profile and QDC regulations.")

    active_orders = st.session_state.custom_modified_orders if st.session_state.custom_modified_orders else (proposal.orders if proposal else [])

    if active_orders:
        for idx, o in enumerate(active_orders, 1):
            col_left, col_right = st.columns([3, 1])
            sig = signals.get(o.ticker) if signals else None
            rf_p = sig.rf_probability if sig else 0.55
            meta = next((e for e in ETF_UNIVERSE if e.ticker == o.ticker), None)

            with col_left:
                # Progressive Disclosure Container
                with st.expander(f"[{idx}] {o.action} {o.ticker} — Allocation: {o.target_weight_pct:.1f}% (€{o.estimated_amount_eur:,.2f})", expanded=True):
                    # Level 1: Plain Language Overview (Retail Persona)
                    st.markdown(f"**Security Name:** {meta.full_name if meta else o.ticker}")
                    st.markdown(f"**Asset Theme:** `{meta.theme if meta else 'Equity'}` | **Leverage Multiplier:** `{meta.leverage if meta else 1.0}x`")
                    
                    # Contextual plain-language role
                    st.markdown("**Strategic Role:**")
                    if "GOLD" in o.ticker:
                        st.write("Defensive hedge. Provides portfolio insurance against geopolitical risk, currency devaluation, and equity market turbulence.")
                    elif "EURO BANKS" in o.ticker:
                        st.write("Income & cyclical momentum. Benefits from sustained central bank policy rates, delivering strong dividend yield and net interest margins.")
                    elif "BITCOIN" in o.ticker:
                        st.write("High-beta digital asset. Selected strictly on strong short-term price momentum, restricted to growth-oriented profiles.")
                    elif "INDIA" in o.ticker:
                        st.write("Structural growth leader. Strong corporate earnings trajectory decoupled from developed market sluggishness.")
                    elif "WORLD MIN VOL" in o.ticker:
                        st.write("Low-volatility anchor. Reduces drawdowns while capturing equity risk premia during uncertain market regimes.")
                    else:
                        st.write("High-conviction tactical allocation supported by quantitative momentum and macro regime tailwinds.")

                    # Level 2: Quant & Technical Factor Breakdown (Informed Investor Persona)
                    st.markdown("---")
                    st.caption("Quantitative Factor Drivers:")
                    st.markdown(f"• **Explainable AI Rationale:** {o.rationale}")
                    c_tech1, c_tech2 = st.columns(2)
                    with c_tech1:
                        st.markdown(f"• 14-Day Wilder RSI: **{sig.rsi_14 if sig else 50.0}** ({sig.rsi_signal if sig else 'NEUTRAL'})")
                    with c_tech2:
                        st.markdown(f"• 21-Day Relative Alpha vs MSCI World: **{sig.roc_21d_pct if sig else 0.0:+.2f}%**")

            with col_right:
                # Win Probability Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(rf_p * 100, 1),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "RF Win-Prob", 'font': {'size': 13}},
                    number={'suffix': "%", 'font': {'size': 16, 'color': "#0F2042"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#0F2042"},
                        'bar': {'color': "#16A34A" if rf_p >= 0.55 else "#F59E0B"},
                        'steps': [
                            {'range': [0, 50], 'color': "#FEE2E2"},
                            {'range': [50, 60], 'color': "#FEF3C7"},
                            {'range': [60, 100], 'color': "#DCFCE7"}
                        ],
                        'threshold': {
                            'line': {'color': "#DC2626", 'width': 2},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=140, margin=dict(l=5, r=5, t=25, b=5))
                st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    # =========================================================================
    # INTERACTIVE HUMAN-IN-THE-LOOP CONTROLS (ADVISE VS MANDATE)
    # =========================================================================
    if service_model == "MANDATE":
        st.markdown(
            "<div class='audit-box'>"
            "<strong>Belfius Discretionary Mandate Active:</strong><br>"
            "Under this agreement, the Belfius Agentic System manages and rebalances your portfolio autonomously "
            "within your selected risk profile parameters without requiring order-by-order confirmation."
            "</div>",
            unsafe_allow_html=True
        )
        if st.button("Execute Autonomous Mandate Rebalancing", type="primary"):
            target_orders = [
                {"ticker": o.ticker, "weight": o.target_weight_pct, "amount": o.estimated_amount_eur}
                for o in active_orders
            ]
            snapshot = ledger.execute_rebalancing(
                target_orders=target_orders,
                current_prices=current_prices,
                trade_date=selected_date,
                var_95=risk_eval.var_95_daily_pct if risk_eval else 0.0,
                cvar_95=risk_eval.cvar_95_daily_pct if risk_eval else 0.0
            )
            st.success(f"Mandate Rebalancing Executed. New NAV: €{snapshot.total_nav:,.2f} | Available Cash: €{snapshot.cash:,.2f}")
            st.rerun()

    else:  # ADVISE MODEL
        st.markdown("### Investment Decision Copilot (Advise Model)")
        st.caption("You maintain full decision authority over your portfolio. Review recommendations, challenge the AI rationale, or adjust allocations.")

        col_action1, col_action2 = st.columns(2)

        # Action Branch 1: Natural Language Negotiation / Argumentation
        with col_action1:
            st.markdown("#### Mode A: Argue with AI Recommendation")
            st.markdown(
                "<div class='info-panel'>"
                "Submit your market perspectives or exclusions in plain text. "
                "The Multi-Agent System will parse your mandate and recalculate allocations dynamically."
                "</div>",
                unsafe_allow_html=True
            )

            user_arg = st.text_input(
                "Enter your perspective or constraint:",
                placeholder="e.g., Exclude crypto, increase gold allocation, and hold more cash..."
            )

            def reoptimize_portfolio(argument_text: str):
                pm_recalc = PortfolioManagerAgent(risk_manager=RiskManager(risk_profile=risk_choice))
                rev_proposal, rev_risk = pm_recalc.construct_portfolio(
                    trading_date=selected_date,
                    tech_signals={t: s.model_dump() for t, s in signals.items()},
                    macro_briefing=macro_obj,
                    starting_capital_eur=current_nav,
                    risk_profile=risk_choice,
                    user_arguments=argument_text,
                    current_holdings=ledger.positions
                )
                st.session_state.last_proposal = rev_proposal
                st.session_state.last_risk_eval = rev_risk
                st.session_state.custom_modified_orders = None
                st.session_state.challenge_history.append({
                    "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "argument": argument_text,
                    "orders": [o.ticker for o in rev_proposal.orders]
                })
                st.session_state.execution_msg = f"Allocation updated to reflect your mandate: '{argument_text}'"
                st.rerun()

            # Quick Preset Argument Chips (Clicking immediately re-optimizes)
            chip_col1, chip_col2 = st.columns(2)
            if chip_col1.button("Exclude Crypto", use_container_width=True):
                reoptimize_portfolio("Exclude all cryptocurrency and digital asset ETFs")
            if chip_col2.button("Boost Gold", use_container_width=True):
                reoptimize_portfolio("Prefer physical gold hedge")
            if chip_col1.button("Cash +8%", use_container_width=True):
                reoptimize_portfolio("Hold higher cash buffer for defensive preservation")
            if chip_col2.button("Exclude Tech", use_container_width=True):
                reoptimize_portfolio("Exclude technology and semiconductor ETFs")

            if st.button("Recalculate with My Perspective"):
                if user_arg.strip():
                    reoptimize_portfolio(user_arg.strip())
                else:
                    st.warning("Please type a perspective or instruction in the text box above.")

            # Challenge / Argue: Show less optimal / runner-up alternatives
            alts = getattr(proposal, "alternative_candidates", []) if proposal else []
            if alts:
                st.markdown("---")
                st.markdown("##### Sub-Optimal Runner-Up Candidates (Ranks 6–10)")
                st.caption("Argue against top picks by exploring runner-up compliant alternatives with empirical evidence:")
                for alt_i, alt in enumerate(alts, 6):
                    with st.expander(f"Rank #{alt_i}: {alt['ticker']} ({alt['theme']}) — Conviction: {alt['conviction_score']:+.2f}"):
                        st.markdown(f"**Security:** {alt['full_name']}")
                        st.markdown(f"**Evidence & Trade-off Analysis:** {alt['trade_off_evidence']}")
                        c_ev1, c_ev2 = st.columns(2)
                        c_ev1.markdown(f"• Win Probability: **{alt['rf_win_prob']*100:.1f}%**")
                        c_ev2.markdown(f"• Macro Tilt: **{alt['theme_macro_tilt']:+.2f}**")
                        if st.button(f"Challenge Top-5: Swap with {alt['ticker']}", key=f"btn_swap_{alt['ticker']}", use_container_width=True):
                            reoptimize_portfolio(f"Replace primary allocation with runner-up alternative {alt['ticker']} ({alt['theme']})")

        # Action Branch 2: Structured Discretionary Order Modification
        with col_action2:
            st.markdown("#### Mode B: Structured Discretionary Override")
            st.markdown(
                "<div class='info-panel'>"
                "Select a specific order to modify and specify the exact target weight. "
                "The Risk Manager audits your modification against regulatory limits in real time."
                "</div>",
                unsafe_allow_html=True
            )

            order_ticker_options = [o.ticker for o in active_orders]
            selected_order_ticker = st.selectbox("Select Order to Modify:", options=order_ticker_options)

            # Option to swap with runner-up alternative
            if alts:
                st.markdown("##### Discretionary Asset Substitution")
                alt_labels = [f"{a['ticker']} — {a['theme']} (Rank #{idx})" for idx, a in enumerate(alts, 6)]
                selected_alt_label = st.selectbox("Or Substitute with Runner-Up Candidate:", options=["(Keep current ticker)"] + alt_labels)
                if selected_alt_label != "(Keep current ticker)":
                    chosen_alt_ticker = selected_alt_label.split()[0]
                    chosen_alt_obj = next((a for a in alts if a["ticker"] == chosen_alt_ticker), None)
                    if chosen_alt_obj:
                        st.caption(f"**Evidence:** {chosen_alt_obj['trade_off_evidence']}")
                    if st.button(f"Substitute {selected_order_ticker} → {chosen_alt_ticker}", use_container_width=True):
                        sub_list = []
                        for o in active_orders:
                            if o.ticker == selected_order_ticker:
                                from src.orchestration.state import OrderItem
                                sub_list.append(OrderItem(
                                    ticker=chosen_alt_ticker,
                                    action="BUY",
                                    target_weight_pct=o.target_weight_pct,
                                    estimated_amount_eur=o.estimated_amount_eur,
                                    rationale=f"[Discretionary Substitution] Substituted {selected_order_ticker} for runner-up {chosen_alt_ticker}."
                                ))
                            else:
                                sub_list.append(o)
                        st.session_state.custom_modified_orders = sub_list
                        st.session_state.execution_msg = f"Substituted {selected_order_ticker} with {chosen_alt_ticker}."
                        st.rerun()

            # Find current weight
            current_order_obj = next((o for o in active_orders if o.ticker == selected_order_ticker), None)
            default_wt = current_order_obj.target_weight_pct if current_order_obj else 15.0

            new_target_weight = st.slider(
                f"Target Weight for {selected_order_ticker} (%):",
                min_value=0.0,
                max_value=20.0,
                value=round(float(default_wt), 1),
                step=0.1,
                help="Single asset concentration cap is 20.0% per official QDC regulations."
            )

            if st.button("Apply Target Weight Modification"):
                # Formulate modified order list
                modified_list = []
                for o in active_orders:
                    if o.ticker == selected_order_ticker:
                        new_amount = round((new_target_weight / 100.0) * current_nav, 2)
                        from src.orchestration.state import OrderItem
                        modified_list.append(OrderItem(
                            ticker=o.ticker,
                            action="BUY" if new_target_weight > 0.001 else "SELL",
                            target_weight_pct=new_target_weight,
                            estimated_amount_eur=new_amount,
                            rationale=f"[Client Discretionary Override] Target weight set to {new_target_weight:.1f}%."
                        ))
                    else:
                        modified_list.append(o)

                # Pre-trade compliance audit
                weights_dict = {o.ticker: o.target_weight_pct / 100.0 for o in modified_list}
                rm = RiskManager(risk_profile=risk_choice)
                is_legal, breaches, compliant_weights = rm.validate_user_override(weights_dict)

                if is_legal:
                    st.session_state.custom_modified_orders = modified_list
                    st.session_state.execution_msg = f"Target weight for {selected_order_ticker} adjusted to {new_target_weight:.1f}%. Review blotter above and confirm below."
                    st.rerun()
                else:
                    # Apply compliant projection
                    projected_list = []
                    for o in modified_list:
                        adj_w = compliant_weights.get(o.ticker, 0.0) * 100.0
                        from src.orchestration.state import OrderItem
                        projected_list.append(OrderItem(
                            ticker=o.ticker,
                            action="BUY" if adj_w > 0.001 else "SELL",
                            target_weight_pct=round(adj_w, 1),
                            estimated_amount_eur=round((adj_w / 100.0) * current_nav, 2),
                            rationale=f"[Compliant Adjustment] Auto-aligned to regulatory limits."
                        ))
                    st.session_state.custom_modified_orders = projected_list
                    st.session_state.execution_msg = f"Constraint breach ({', '.join(breaches)}) auto-aligned to regulatory bounds."
                    st.rerun()

            if st.session_state.custom_modified_orders:
                st.markdown("**Pending Modified Allocation:**")
                preview_df = pd.DataFrame([
                    {"Ticker": o.ticker, "Weight": f"{o.target_weight_pct:.1f}%", "Amount": f"€{o.estimated_amount_eur:,.2f}"}
                    for o in st.session_state.custom_modified_orders
                ])
                st.dataframe(preview_df, use_container_width=True)

        st.markdown("---")

        # Final Confirmation & Execution
        st.markdown("#### Confirm & Execute Orders")
        c_exec1, c_exec2 = st.columns([2, 1])
        with c_exec1:
            st.write("Confirming transmits orders to the portfolio accounting ledger at the evaluation close price.")
        with c_exec2:
            if st.button("Approve & Execute Orders to Ledger", type="primary", use_container_width=True):
                target_orders = [
                    {"ticker": o.ticker, "weight": o.target_weight_pct, "amount": o.estimated_amount_eur}
                    for o in active_orders
                ]
                snapshot = ledger.execute_rebalancing(
                    target_orders=target_orders,
                    current_prices=current_prices,
                    trade_date=selected_date,
                    var_95=risk_eval.var_95_daily_pct if risk_eval else 0.0,
                    cvar_95=risk_eval.cvar_95_daily_pct if risk_eval else 0.0
                )
                st.session_state.custom_modified_orders = None
                st.session_state.execution_msg = f"Orders successfully executed! Portfolio NAV: €{snapshot.total_nav:,.2f} | Cash: €{snapshot.cash:,.2f} | Period PnL: €{snapshot.period_pnl:+,.2f}. Open Tab 2 to inspect ledger charts."
                st.rerun()

# ------------------------------------------------------------------------------
# TAB 2: PORTFOLIO ACCOUNTING & LEDGER
# ------------------------------------------------------------------------------
with tab2:
    st.markdown("### Portfolio NAV Historical Performance")
    
    # Load ledger snapshots
    ledger_df = pd.DataFrame()
    if os.path.exists(ledger.ledger_path):
        try:
            ledger_df = pd.read_parquet(ledger.ledger_path)
        except Exception:
            pass

    if not ledger_df.empty:
        fig_nav = px.line(
            ledger_df,
            x="trade_date",
            y="total_nav",
            markers=True,
            title="Portfolio Net Asset Value (NAV across Competition Trading Windows)",
            labels={"trade_date": "Trading Window Date", "total_nav": "Total NAV (€)"}
        )
        fig_nav.update_traces(line_color="#0F2042", line_width=3)
        fig_nav.add_hline(y=10000.0, line_dash="dash", line_color="#64748B", annotation_text="Initial Capital: €10,000")
        st.plotly_chart(fig_nav, use_container_width=True)
    else:
        st.info("No execution snapshots recorded yet. Approve and execute orders in Tab 1 to record the initial window.")

    col_chart_left, col_chart_right = st.columns(2)
    with col_chart_left:
        st.markdown("#### Asset Allocation Distribution")
        pos_dict = ledger.positions
        if pos_dict:
            labels = list(pos_dict.keys()) + ["CASH RESERVE"]
            values = [p.market_value for p in pos_dict.values()] + [ledger.cash]
            fig_pie = px.pie(names=labels, values=values, hole=0.45, color_discrete_sequence=px.colors.qualitative.Prism)
            fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption("Portfolio is currently 100% in cash buffer.")

    with col_chart_right:
        st.markdown("#### Thematic Exposure vs Concentration Cap")
        if risk_eval and risk_eval.theme_concentrations:
            theme_df = pd.DataFrame(list(risk_eval.theme_concentrations.items()), columns=["Thematic Cluster", "Weight"])
            theme_df["Weight (%)"] = theme_df["Weight"] * 100
            fig_bar = px.bar(theme_df, x="Weight (%)", y="Thematic Cluster", orientation='h', color="Weight (%)", color_continuous_scale="Blues")
            fig_bar.add_vline(x=35.0, line_dash="dash", line_color="#DC2626", annotation_text="Regulatory Cap: 35%")
            fig_bar.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    # Active Holdings Blotter
    st.markdown("### Active Positions Blotter")
    if ledger.positions:
        holdings_rows = []
        for tkr, pos in ledger.positions.items():
            holdings_rows.append({
                "ETF Ticker": tkr,
                "Shares Held": f"{pos.shares:.4f}",
                "Average Entry Price": f"€{pos.avg_entry_price:.2f}",
                "Current Price": f"€{pos.current_price:.2f}",
                "Market Value": f"€{pos.market_value:.2f}",
                "Unrealized PnL": f"€{pos.unrealized_pnl:+.2f}",
                "Portfolio Weight": f"{pos.weight_pct:.1f}%"
            })
        st.dataframe(pd.DataFrame(holdings_rows), use_container_width=True)
    else:
        st.write("No open positions. 100% cash allocation retained.")

# ------------------------------------------------------------------------------
# TAB 3: INVESTOR ACADEMY & METRIC DEFINITIONS (PROGRESSIVE DISCLOSURE)
# ------------------------------------------------------------------------------
with tab3:
    st.markdown("### Investor Academy & Technical Reference")
    st.caption("Explanations designed with progressive disclosure for retail, informed, and institutional investors.")

    with st.expander("1. Random Forest Binary Win-Probability (P_RF)", expanded=True):
        st.markdown(
            "• **Retail Investor Concept:** The statistical confidence score (0% to 100%) computed by an ensemble of 200 Decision Trees "
            "evaluating whether an ETF will outperform the global stock market index (MSCI World) over the next 5 trading days.<br>"
            "• **Interpretation:** Values above 55% signal significant upside probability; values between 45% and 55% denote neutral momentum; values below 45% indicate heightened risk of relative underperformance.<br>"
            "• **Institutional Rigor:** Trained on a cross-sectional panel of 34,423 historical ETF observations using 17 technical and relative momentum factors with zero data leakage.",
            unsafe_allow_html=True
        )

    with st.expander("2. 14-Day Wilder Relative Strength Index (RSI)"):
        st.markdown(
            "• **Retail Investor Concept:** An oscillator measuring the velocity of price movements on a scale of 0 to 100.<br>"
            "• **Interpretation:** RSI below 35 indicates an oversold asset where selling pressure may be exhausted; RSI between 50 and 65 denotes healthy, sustainable upward trend; RSI above 68 reflects overbought conditions with short-term consolidation risk.<br>"
            "• **Institutional Rigor:** Calculated using exponential smoothing Wilder alpha (1/14), avoiding discrete window jumps.",
            unsafe_allow_html=True
        )

    with st.expander("3. 1-Day 95% Historical Value-at-Risk (VaR) & Conditional VaR (CVaR)"):
        st.markdown(
            "• **Retail Investor Concept:** Answers the core risk question: *'What is the maximum estimated loss my portfolio could suffer tomorrow under normal market conditions?'*<br>"
            "• **Example:** A 1-Day 95% VaR of -1.81% indicates that on 95 out of 100 normal trading days, your daily drawdown will not exceed €181 on a €10,000 account.<br>"
            "• **Conditional VaR (Expected Shortfall):** Quantifies the expected average loss during the worst 5% extreme tail events, ensuring regulatory capitalization adequacy.",
            unsafe_allow_html=True
        )

    with st.expander("4. Macroeconomic Regimes (Risk-On, Risk-Off, Stagflation)"):
        st.markdown(
            "• **Risk-On (Economic Growth):** Accommodative financial conditions, low volatility (VIX < 18). Capital is deployed into high-beta equities, financials, and emerging markets.<br>"
            "• **Risk-Off (Capital Preservation):** Heightened market volatility (VIX > 22), geopolitical stress. Capital is shifted to cash buffers, sovereign bonds, and physical gold.<br>"
            "• **Stagflation (Inflationary Headwinds):** Decelerating GDP coupled with persistent inflation. Capital is allocated toward energy, commodities, and uranium.",
            unsafe_allow_html=True
        )

    with st.expander("5. Leveraged ETF Geometric Volatility Drag (Cheng & Madhavan, 2009)"):
        st.markdown(
            "• **Retail Investor Warning:** Leveraged ETFs (2x/3x) are daily rebalancing instruments. When held across volatile, non-trending markets, compounding results in mathematical decay.<br>"
            "• **Governance Implementation:** Leveraged ETFs are strictly prohibited in the Conservative mandate, capped at 5% in Moderate, and restricted to 15% in Aggressive.",
            unsafe_allow_html=True
        )

    with st.expander("6. Official QDC Competition Regulations"):
        st.markdown(
            "• **Initial Capital:** €10,000.<br>"
            "• **Order Cap:** Maximum 5 orders per competition day, executed at previous closing prices with zero trading fees.<br>"
            "• **Minimum Cash Buffer:** 5.0% required at all times (Belfius ReBel sets 10% for Moderate, 20% for Conservative).<br>"
            "• **Single Asset Limit:** Maximum 20.0% allocated to any single ETF.<br>"
            "• **Thematic Concentration Limit:** Maximum 35.0% allocated to any single thematic cluster.<br>"
            "• **Leveraged Product Limit:** Maximum 15.0% combined leverage across all leveraged/inverse ETFs; maximum 8.0% in any single leveraged product.",
            unsafe_allow_html=True
        )

# ------------------------------------------------------------------------------
# TAB 4: MULTI-AGENT ARCHITECTURE & AUDIT TRAIL
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("### Multi-Agent System Architecture (LangGraph Flow)")
    
    col_arch1, col_arch2 = st.columns([2, 1])
    with col_arch1:
        st.markdown("""
        The system executes a strictly governed, sequential 6-node decision graph:
        1. **Data Ingestion Node**: Ingests closing price series for all 50 ETFs, marks positions to market, updates NAV.
        2. **Technical RF Agent**: Automatically checks for fresh price series, executes walk-forward retraining, and generates binary outperformance probabilities.
        3. **Macroeconomic Agent**: Evaluates transatlantic policy spread (Fed vs ECB) and real-time news headline sentiment stream.
        4. **Risk Manager Agent**: Enforces MiFID II risk profile constraints and projects non-compliant vectors onto legal QDC polytope.
        5. **Portfolio Manager Agent**: Synthesizes multi-factor convictions, parses client arguments via NLP, sizes positions.
        6. **Human-in-the-Loop Node**: Directs execution through either the Discretionary Mandate path or interactive Advise review.
        """)

    with col_arch2:
        st.markdown("#### Machine Learning Model Health")
        bundle = get_rf_model_bundle()
        st.success(
            f"• Model Status: Online\n"
            f"• Last Retrained: {bundle.get('last_trained_date', 'N/A')}\n"
            f"• Training Samples: {bundle.get('num_samples', 0):,}\n"
            f"• Feature Dimension: {len(FEATURE_COLS)} institutional factors"
        )

    st.markdown("### Immutable Compliance Audit Trail")
    audit_log_path = "logs/portfolio_ledger_audit.jsonl"
    if os.path.exists(audit_log_path):
        with open(audit_log_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f.readlines() if line.strip()]
        if lines:
            audit_display_df = pd.DataFrame(lines)[["trade_date", "cash", "invested_value", "total_nav", "period_pnl", "period_return_pct", "cumulative_return_pct"]]
            st.dataframe(audit_display_df.tail(10), use_container_width=True)
    else:
        st.info("No audit records logged yet.")
