import os
import sys
import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier

sys.path.append('/Users/nguyenhangthu/Desktop/QTEM/GBAC/team-29-ai-agent')
from src.pipeline.etf_universe import ETF_UNIVERSE

prices_path = '/Users/nguyenhangthu/Desktop/QTEM/GBAC/team-29-ai-agent/data/etf_prices_eur.parquet'
prices_df = pd.read_parquet(prices_path)
if not isinstance(prices_df.index, pd.DatetimeIndex):
    prices_df.index = pd.to_datetime(prices_df.index)

meta_dict = {
    etf.ticker: {
        "name": getattr(etf, "full_name", etf.ticker),
        "theme": etf.theme,
        "is_leveraged": int(etf.is_leveraged_or_short),
        "leverage": etf.leverage,
        "currency": getattr(etf, "base_currency", "EUR")
    }
    for etf in ETF_UNIVERSE
}

bmk_series = prices_df["WORLD"] if "WORLD" in prices_df.columns else prices_df.iloc[:, 0]

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def extract_features_single(prices: pd.Series, ticker: str, bmk: pd.Series, meta: dict, forward_h: int = 5) -> pd.DataFrame:
    clean = prices.dropna()
    if len(clean) < 70:
        return pd.DataFrame()
    df = pd.DataFrame(index=clean.index)
    p = clean
    sma20 = p.rolling(20).mean()
    sma50 = p.rolling(50).mean()
    ema20 = p.ewm(span=20, adjust=False).mean()
    df["dist_sma20"] = (p / sma20) - 1.0
    df["dist_sma50"] = (p / sma50) - 1.0
    df["ema20_sma50_ratio"] = (ema20 / sma50) - 1.0
    df["roc_5d"] = p.pct_change(5)
    df["roc_10d"] = p.pct_change(10)
    df["roc_21d"] = p.pct_change(21)
    df["roc_63d"] = p.pct_change(63)
    df["rsi_14"] = calc_rsi(p, 14)
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist_norm"] = (macd_line - signal_line) / p
    log_ret = np.log(p / p.shift(1))
    df["realized_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["realized_vol_63d"] = log_ret.rolling(63).std() * np.sqrt(252)
    std20 = p.rolling(20).std()
    upper_b = sma20 + 2.0 * std20
    lower_b = sma20 - 2.0 * std20
    df["boll_bandwidth"] = (upper_b - lower_b) / (sma20 + 1e-9)
    df["boll_pct_b"] = (p - lower_b) / (upper_b - lower_b + 1e-9)
    bmk_clean = bmk.reindex(p.index).ffill()
    df["rel_roc21_vs_world"] = df["roc_21d"] - bmk_clean.pct_change(21)
    df["rel_roc63_vs_world"] = df["roc_63d"] - bmk_clean.pct_change(63)
    df["is_leveraged"] = meta.get("is_leveraged", 0)
    df["theme"] = meta.get("theme", "Other")
    df["ticker"] = ticker
    fwd_ret_asset = (p.shift(-forward_h) / p) - 1.0
    fwd_ret_bmk = (bmk_clean.shift(-forward_h) / bmk_clean) - 1.0
    df["target_fwd_alpha_5d"] = fwd_ret_asset - fwd_ret_bmk
    df["target_binary_outperform"] = (df["target_fwd_alpha_5d"] > 0.0).astype(int)
    return df

panels = [extract_features_single(prices_df[c], c, bmk_series, meta_dict.get(c, {})) for c in prices_df.columns]
clean_panel = pd.concat(panels, axis=0).dropna().copy()

themes = sorted(clean_panel["theme"].unique())
theme_to_code = {t: i for i, t in enumerate(themes)}
clean_panel["theme_code"] = clean_panel["theme"].map(theme_to_code)

FEATURE_COLS = [
    "dist_sma20", "dist_sma50", "ema20_sma50_ratio",
    "roc_5d", "roc_10d", "roc_21d", "roc_63d",
    "rsi_14", "macd_hist_norm",
    "realized_vol_20d", "realized_vol_63d",
    "boll_bandwidth", "boll_pct_b",
    "rel_roc21_vs_world", "rel_roc63_vs_world",
    "is_leveraged", "theme_code"
]

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
print("RF model trained successfully!")

# Save model bundle (model, feature_cols, theme_to_code)
bundle = {
    "model": rf,
    "feature_cols": FEATURE_COLS,
    "theme_to_code": theme_to_code
}
model_path = '/Users/nguyenhangthu/Desktop/QTEM/GBAC/team-29-ai-agent/data/rf_technical_model.joblib'
joblib.dump(bundle, model_path)
print(f"Model bundle saved to {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")
