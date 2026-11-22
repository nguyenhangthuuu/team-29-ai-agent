# src/agents/technical.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
import os
import joblib
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.pipeline.etf_universe import ETF_UNIVERSE

MODEL_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "rf_technical_model.joblib"
)

PRICES_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "etf_prices_eur.parquet"
)

FEATURE_COLS = [
    "dist_sma20", "dist_sma50", "ema20_sma50_ratio",
    "roc_5d", "roc_10d", "roc_21d", "roc_63d",
    "mom_accel_5_21",
    "rsi_14", "rsi_slope_3d", "macd_hist_norm",
    "realized_vol_20d", "realized_vol_63d",
    "boll_bandwidth", "boll_pct_b", "boll_pct_b_delta",
    "rel_roc21_vs_world", "rel_roc63_vs_world",
    "is_leveraged", "theme_code"
]

# ETF Metadata lookup dictionary
ETF_META_LOOKUP = {
    etf.ticker: {
        "name": getattr(etf, "full_name", etf.ticker),
        "theme": etf.theme,
        "is_leveraged": int(etf.is_leveraged_or_short),
        "leverage": etf.leverage,
        "currency": getattr(etf, "base_currency", "EUR")
    }
    for etf in ETF_UNIVERSE
}


class TechnicalSignal(BaseModel):
    ticker: str
    composite_score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Normalized RF alpha score: -1.0 (Strong Sell) to +1.0 (Strong Buy), mapped from 2*P(RF)-1"
    )
    rf_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Random Forest binary probability of 5-day outperformance vs WORLD benchmark"
    )
    predicted_class: int = Field(
        ..., description="Binary classification label: 1 (Outperformer) if prob > 0.50, else 0 (Underperformer)"
    )
    trend: str = Field(..., description="BULLISH, BEARISH, or NEUTRAL based on RF probability thresholds")
    rsi_14: float
    rsi_signal: str
    roc_21d_pct: float
    roc_63d_pct: float
    breakout_state: str
    reasons: List[str]
    summary_reason: str


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates 14-day Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def extract_features_single_series(
    prices: pd.Series,
    ticker: str,
    bmk: Optional[pd.Series] = None,
    meta: Optional[dict] = None
) -> pd.DataFrame:
    """Extracts institutional 17-factor feature matrix for a single ETF series."""
    clean = prices.dropna()
    if len(clean) < 65:
        return pd.DataFrame()

    df = pd.DataFrame(index=clean.index)
    p = clean
    meta = meta or ETF_META_LOOKUP.get(ticker, {})

    # 1. Moving Average Trends
    sma20 = p.rolling(20).mean()
    sma50 = p.rolling(50).mean()
    ema20 = p.ewm(span=20, adjust=False).mean()
    df["dist_sma20"] = (p / (sma20 + 1e-9)) - 1.0
    df["dist_sma50"] = (p / (sma50 + 1e-9)) - 1.0
    df["ema20_sma50_ratio"] = (ema20 / (sma50 + 1e-9)) - 1.0

    # 2. Multi-Horizon Momentum & Fast Reversal Dynamics
    df["roc_5d"] = p.pct_change(5)
    df["roc_10d"] = p.pct_change(10)
    df["roc_21d"] = p.pct_change(21)
    df["roc_63d"] = p.pct_change(63)
    # Fast Reversal 1: Momentum Acceleration Spread (short-term vs intermediate-term)
    df["mom_accel_5_21"] = df["roc_5d"] - df["roc_21d"]

    # 3. Oscillators & Fast Reversal Exhaustion
    df["rsi_14"] = calc_rsi(p, 14)
    # Fast Reversal 2: 3-day RSI Slope (Overbought Exhaustion / Divergence)
    df["rsi_slope_3d"] = (df["rsi_14"] - df["rsi_14"].shift(3)) / 100.0

    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist_norm"] = (macd_line - signal_line) / (p + 1e-9)

    # 4. Volatility Dynamics & Band Slippage
    log_ret = np.log(p / p.shift(1))
    df["realized_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["realized_vol_63d"] = log_ret.rolling(63).std() * np.sqrt(252)

    std20 = p.rolling(20).std()
    upper_b = sma20 + 2.0 * std20
    lower_b = sma20 - 2.0 * std20
    df["boll_bandwidth"] = (upper_b - lower_b) / (sma20 + 1e-9)
    df["boll_pct_b"] = (p - lower_b) / (upper_b - lower_b + 1e-9)
    # Fast Reversal 3: 3-day Bollinger Band %B Slippage
    df["boll_pct_b_delta"] = df["boll_pct_b"] - df["boll_pct_b"].shift(3)

    # 5. Relative Strength vs WORLD
    if bmk is not None:
        bmk_clean = bmk.reindex(p.index).ffill()
        df["rel_roc21_vs_world"] = df["roc_21d"] - bmk_clean.pct_change(21)
        df["rel_roc63_vs_world"] = df["roc_63d"] - bmk_clean.pct_change(63)
    else:
        df["rel_roc21_vs_world"] = 0.0
        df["rel_roc63_vs_world"] = 0.0

    df["is_leveraged"] = meta.get("is_leveraged", 0)
    df["theme"] = meta.get("theme", "Other")
    df["ticker"] = ticker

    return df


def train_rf_model(
    prices_path: str = PRICES_FILE_PATH,
    save_path: str = MODEL_CACHE_PATH
) -> dict:
    """Trains Random Forest Classifier across cross-sectional ETF panel and caches model bundle."""
    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"Historical ETF price dataset not found at {prices_path}")

    prices_df = pd.read_parquet(prices_path)
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df.index = pd.to_datetime(prices_df.index)

    bmk_series = prices_df["WORLD"] if "WORLD" in prices_df.columns else prices_df.iloc[:, 0]

    panels = []
    for col in prices_df.columns:
        meta = ETF_META_LOOKUP.get(col, {})
        idf = extract_features_single_series(prices_df[col], col, bmk_series, meta)
        if not idf.empty:
            # 5-day forward return target
            clean_p = prices_df[col].dropna()
            fwd_ret_asset = (clean_p.shift(-5) / clean_p) - 1.0
            fwd_ret_bmk = (bmk_series.reindex(clean_p.index).ffill().shift(-5) / bmk_series.reindex(clean_p.index).ffill()) - 1.0
            idf["target_fwd_alpha_5d"] = fwd_ret_asset - fwd_ret_bmk
            idf["target_binary_outperform"] = (idf["target_fwd_alpha_5d"] > 0.0).astype(int)
            panels.append(idf)

    clean_panel = pd.concat(panels, axis=0).dropna().copy()
    themes = sorted(clean_panel["theme"].unique())
    theme_to_code = {t: i for i, t in enumerate(themes)}
    clean_panel["theme_code"] = clean_panel["theme"].map(theme_to_code)

    X = clean_panel[FEATURE_COLS]
    y = clean_panel["target_binary_outperform"]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=25,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    last_date_str = str(prices_df.index.max().date())
    bundle = {
        "model": rf,
        "feature_cols": FEATURE_COLS,
        "theme_to_code": theme_to_code,
        "last_trained_date": last_date_str,
        "num_samples": len(X),
        "num_features": len(FEATURE_COLS),
        "trained_at_iso": datetime.now(timezone.utc).isoformat()
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(bundle, save_path)
    return bundle


def retrain_if_new_data(
    prices_df: pd.DataFrame,
    save_path: str = MODEL_CACHE_PATH,
    force: bool = False
) -> Tuple[bool, str, dict]:
    """
    Self-Updating ML Engine for QDC trading windows.
    Automatically checks if fresh market data is present in prices_df compared to
    the model's last training checkpoint. If fresh data is detected (or force=True),
    re-trains the Random Forest model and updates the cached bundle.
    
    Returns:
        (retrained: bool, status_message: str, model_bundle: dict)
    """
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df.index = pd.to_datetime(prices_df.index)

    current_latest_date = prices_df.index.max().date()

    if not force and os.path.exists(save_path):
        try:
            bundle = joblib.load(save_path)
            last_date_str = bundle.get("last_trained_date")
            if last_date_str:
                last_trained_date = pd.to_datetime(last_date_str).date()
                if current_latest_date <= last_trained_date:
                    msg = (
                        f"Random Forest model is up to date (last trained on {last_date_str}, "
                        f"{bundle.get('num_samples', 0)} samples). No retraining needed."
                    )
                    return False, msg, bundle
        except Exception:
            pass  # Corrupted or outdated bundle -> proceed to retrain

    # New trading data detected or forced -> execute retraining
    print(f"🔄 [ML Self-Update] Fresh trading data detected up to {current_latest_date}. Retraining Random Forest...")
    # Temporarily save prices_df if needed or train directly
    bundle = train_rf_model(save_path=save_path)
    msg = (
        f"Random Forest model successfully re-trained with fresh market data up to {current_latest_date} "
        f"({bundle.get('num_samples', 0)} samples across 50 ETFs)."
    )
    print(f"✅ [ML Self-Update] {msg}")
    return True, msg, bundle


def get_rf_model_bundle() -> dict:
    """Loads cached Random Forest model bundle, or retrains if not present or feature schema mismatched."""
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            bundle = joblib.load(MODEL_CACHE_PATH)
            if bundle.get("feature_cols") == FEATURE_COLS:
                return bundle
        except Exception:
            return train_rf_model()
    return train_rf_model()


def evaluate_etf_technical(
    prices: pd.Series,
    ticker: str,
    benchmark_prices: Optional[pd.Series] = None
) -> TechnicalSignal:
    """
    Evaluates ETF forward 5-day alpha potential using pure Random Forest binary classification.
    Returns structured TechnicalSignal with composite_score mapped from RF probability.
    """
    clean_series = prices.dropna()
    if len(clean_series) < 65:
        return TechnicalSignal(
            ticker=ticker,
            composite_score=0.0,
            rf_probability=0.50,
            predicted_class=0,
            trend="NEUTRAL",
            rsi_14=50.0,
            rsi_signal="INSUFFICIENT_DATA",
            roc_21d_pct=0.0,
            roc_63d_pct=0.0,
            breakout_state="CONSOLIDATING",
            reasons=["Insufficient historical data to compute technical feature matrix."],
            summary_reason="Neutral due to insufficient data."
        )

    # 1. Extract feature row for latest date
    meta = ETF_META_LOOKUP.get(ticker, {})
    feat_df = extract_features_single_series(clean_series, ticker, benchmark_prices, meta)
    if feat_df.empty:
        return TechnicalSignal(
            ticker=ticker,
            composite_score=0.0,
            rf_probability=0.50,
            predicted_class=0,
            trend="NEUTRAL",
            rsi_14=50.0,
            rsi_signal="EXTRACTION_FAILED",
            roc_21d_pct=0.0,
            roc_63d_pct=0.0,
            breakout_state="CONSOLIDATING",
            reasons=["Failed to extract technical features."],
            summary_reason="Neutral due to extraction failure."
        )

    bundle = get_rf_model_bundle()
    rf_model: RandomForestClassifier = bundle["model"]
    theme_to_code: dict = bundle["theme_to_code"]

    latest_row = feat_df.iloc[-1:].copy()
    theme_str = latest_row["theme"].iloc[0]
    latest_row["theme_code"] = theme_to_code.get(theme_str, 0)
    X_latest = latest_row[FEATURE_COLS].fillna(0.0)

    # 2. Predict Random Forest outperformance probability
    probs = rf_model.predict_proba(X_latest)[0]
    rf_prob = float(probs[1])  # Probability of forward 5-day outperformance vs WORLD
    pred_class = int(rf_prob > 0.50)

    # Map probability [0, 1] to composite score [-1, +1]
    composite_score = float(np.clip(2.0 * rf_prob - 1.0, -1.0, 1.0))

    # Fast Reversal Tactical Guard:
    # If an asset suffers severe short-term momentum deceleration (5d ROC lagging 21d ROC by > 8%),
    # apply a small exhaustion penalty to avoid the Trailing Momentum Trap.
    mom_accel = float(latest_row["mom_accel_5_21"].iloc[0] * 100) if "mom_accel_5_21" in latest_row else 0.0
    rsi_slope = float(latest_row["rsi_slope_3d"].iloc[0] * 100) if "rsi_slope_3d" in latest_row else 0.0
    pct_b_delta = float(latest_row["boll_pct_b_delta"].iloc[0]) if "boll_pct_b_delta" in latest_row else 0.0

    if mom_accel < -8.0 and composite_score > 0:
        composite_score = max(0.01, composite_score - 0.08)
    elif mom_accel > 8.0 and composite_score > 0:
        composite_score = min(1.0, composite_score + 0.05)

    # 3. Categorize Trend & Status
    if rf_prob >= 0.56:
        trend = "BULLISH"
    elif rf_prob <= 0.44:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    # Contextual sub-indicators for explainability
    rsi_val = float(latest_row["rsi_14"].iloc[0])
    if rsi_val < 35.0:
        rsi_state = "OVERSOLD"
    elif rsi_val > 68.0:
        rsi_state = "OVERBOUGHT"
    elif 50.0 <= rsi_val <= 65.0:
        rsi_state = "BULLISH_MOMENTUM"
    else:
        rsi_state = "NEUTRAL"

    roc_21 = float(latest_row["roc_21d"].iloc[0] * 100)
    roc_63 = float(latest_row["roc_63d"].iloc[0] * 100)
    dist_sma50 = float(latest_row["dist_sma50"].iloc[0] * 100)
    rel_roc21 = float(latest_row["rel_roc21_vs_world"].iloc[0] * 100)
    boll_bw = float(latest_row["boll_bandwidth"].iloc[0])

    pct_b = float(latest_row["boll_pct_b"].iloc[0])
    if pct_b > 1.0:
        breakout = "UPPER_BREAKOUT"
    elif pct_b < 0.0:
        breakout = "LOWER_BREAKOUT"
    else:
        breakout = "CONSOLIDATING"

    # 4. Generate Institutional Explainable AI (XAI) Reasons
    reasons = []
    reasons.append(f"Random Forest Alpha Probability: {rf_prob*100:.1f}% ({'Class 1 Outperformer' if pred_class==1 else 'Class 0 Underperformer'})")

    if dist_sma50 > 2.0:
        reasons.append(f"Golden Trend: price is +{dist_sma50:.1f}% above 50-day moving average")
    elif dist_sma50 < -2.0:
        reasons.append(f"Downtrend: price is {dist_sma50:.1f}% below 50-day moving average")

    if rel_roc21 > 1.5:
        reasons.append(f"Benchmark Outperformance: +{rel_roc21:.1f}% 21d relative excess alpha vs WORLD")
    elif rel_roc21 < -1.5:
        reasons.append(f"Benchmark Lag: {rel_roc21:.1f}% 21d relative underperformance vs WORLD")

    # Fast Reversal dynamics
    if mom_accel < -6.0:
        reasons.append(f"Momentum Deceleration: 5d ROC lags 21d ROC by {abs(mom_accel):.1f}% (short-term stall warning)")
    elif mom_accel > 6.0:
        reasons.append(f"Momentum Acceleration: 5d ROC outpaces 21d ROC by +{mom_accel:.1f}%")

    if rsi_slope < -8.0 and rsi_val > 55.0:
        reasons.append(f"RSI Exhaustion: 3d RSI cooled by {abs(rsi_slope):.1f} pts (buyer fatigue)")
    elif rsi_slope > 8.0 and rsi_val < 45.0:
        reasons.append(f"RSI Rebound: 3d RSI hooked upward by +{rsi_slope:.1f} pts")

    if pct_b_delta < -0.20 and pct_b > 0.65:
        reasons.append("Bollinger Retracement: price slipping back inside upper band")

    reasons.append(f"Momentum & Volatility Quality: 14d RSI at {rsi_val:.1f} ({rsi_state}), Bollinger Bandwidth at {boll_bw:.3f}")
    summary_reason = " | ".join(reasons)

    return TechnicalSignal(
        ticker=ticker,
        composite_score=round(composite_score, 3),
        rf_probability=round(rf_prob, 3),
        predicted_class=pred_class,
        trend=trend,
        rsi_14=round(rsi_val, 1),
        rsi_signal=rsi_state,
        roc_21d_pct=round(roc_21, 2),
        roc_63d_pct=round(roc_63, 2),
        breakout_state=breakout,
        reasons=reasons,
        summary_reason=summary_reason
    )


def evaluate_all_etfs(prices_df: pd.DataFrame, benchmark_ticker: str = "WORLD") -> Dict[str, TechnicalSignal]:
    """Evaluates all ETFs in the price dataframe using Random Forest binary classification."""
    benchmark = prices_df[benchmark_ticker] if benchmark_ticker in prices_df.columns else None
    results = {}
    for col in prices_df.columns:
        results[col] = evaluate_etf_technical(prices_df[col], ticker=col, benchmark_prices=benchmark)
    return results
