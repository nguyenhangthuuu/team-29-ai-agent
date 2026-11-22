"""
Experiment 2: Testing Fast Reversal Indicators on top of existing ML features.
Does adding short-term momentum acceleration, RSI slope, and Bollinger breakdown
fix the Trailing Momentum Trap?
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.agents.technical import (
    PRICES_FILE_PATH,
    ETF_META_LOOKUP,
    FEATURE_COLS,
    extract_features_single_series
)

prices_df = pd.read_parquet(PRICES_FILE_PATH)
if not isinstance(prices_df.index, pd.DatetimeIndex):
    prices_df.index = pd.to_datetime(prices_df.index)

bmk_series = prices_df["WORLD"] if "WORLD" in prices_df.columns else prices_df.iloc[:, 0]

panels = []
for col in prices_df.columns:
    meta = ETF_META_LOOKUP.get(col, {})
    idf = extract_features_single_series(prices_df[col], col, bmk_series, meta)
    if not idf.empty:
        clean_p = prices_df[col].dropna()
        fwd_ret_asset = (clean_p.shift(-5) / clean_p) - 1.0
        fwd_ret_bmk = (bmk_series.reindex(clean_p.index).ffill().shift(-5) / bmk_series.reindex(clean_p.index).ffill()) - 1.0
        idf["target_fwd_alpha_5d"] = fwd_ret_asset - fwd_ret_bmk
        idf["target_binary_outperform"] = (idf["target_fwd_alpha_5d"] > 0.0).astype(int)

        # FAST REVERSAL INDICATORS (Derived directly from daily close price series):
        # 1. Momentum Acceleration: Spread between fast 5d ROC and medium 21d ROC
        idf["mom_accel_5_21"] = idf["roc_5d"] - idf["roc_21d"]
        
        # 2. RSI 3-day Slope: Measures loss of buyer momentum
        idf["rsi_slope_3d"] = idf["rsi_14"].diff(3) / 100.0
        
        # 3. Bollinger Band Breakdown: Distance from upper band
        idf["boll_pct_b_delta"] = idf["boll_pct_b"].diff(3)

        panels.append(idf)

clean_panel = pd.concat(panels, axis=0).dropna().sort_index().copy()
themes = sorted(clean_panel["theme"].unique())
theme_to_code = {t: i for i, t in enumerate(themes)}
clean_panel["theme_code"] = clean_panel["theme"].map(theme_to_code)

unique_dates = np.array(sorted(clean_panel.index.unique()))
n_dates = len(unique_dates)

test_size_dates = int(n_dates * 0.15)
test_dates = unique_dates[-test_size_dates:]
test_mask = clean_panel.index.isin(test_dates)

tr_dates = unique_dates[:n_dates - test_size_dates - 5]
tr_mask = clean_panel.index.isin(tr_dates)

NEW_FEATURE_COLS = FEATURE_COLS + ["mom_accel_5_21", "rsi_slope_3d", "boll_pct_b_delta"]

print("Comparing Original Features vs With Fast Reversal Indicators on Test Window (Jun - Nov 2026):")
print("-" * 80)

for feat_set_name, cols in [("Original 17 Features", FEATURE_COLS), ("+ Fast Reversal Indicators (20 Features)", NEW_FEATURE_COLS)]:
    X_train = clean_panel.loc[tr_mask, cols]
    y_train = clean_panel.loc[tr_mask, "target_binary_outperform"]
    X_test = clean_panel.loc[test_mask, cols]
    y_test = clean_panel.loc[test_mask, "target_binary_outperform"]
    test_meta = clean_panel.loc[test_mask, ["ticker", "target_fwd_alpha_5d"]].copy()

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_test)[:, 1]
    preds = (probs > 0.50).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    df_eval = test_meta.copy()
    df_eval["prob"] = probs
    top5_alphas = []
    for d in test_dates:
        if d in df_eval.index:
            day_df = df_eval.loc[d]
            if isinstance(day_df, pd.Series):
                day_df = day_df.to_frame().T
            top5 = day_df.sort_values(by="prob", ascending=False).head(5)
            top5_alphas.append(top5["target_fwd_alpha_5d"].mean())
    mean_alpha = np.mean(top5_alphas) * 100 if top5_alphas else 0.0

    print(f"[{feat_set_name}]")
    print(f"  Accuracy: {acc*100:.2f}% | AUC: {auc:.3f} | Top-5 5-day Alpha: {mean_alpha:+.3f}%")
