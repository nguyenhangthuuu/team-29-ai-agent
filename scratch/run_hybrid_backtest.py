import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# Add project root
sys.path.append('/Users/nguyenhangthu/Desktop/QTEM/GBAC/team-29-ai-agent')
from src.agents.technical import evaluate_etf_technical

# 1. Load ETF Prices
prices_path = '/Users/nguyenhangthu/Desktop/QTEM/GBAC/team-29-ai-agent/data/etf_prices_eur.parquet'
prices_df = pd.read_parquet(prices_path)
if not isinstance(prices_df.index, pd.DatetimeIndex):
    prices_df.index = pd.to_datetime(prices_df.index)

LEVERAGED_TICKERS = [
    "NASDAQ 3X", "SOX 3X", "S&P 500 3X", "STOXX SHORT", "S&P 500 SHORT",
    "COPPER LVG", "PALLADIUM LVG", "BITCOIN", "CRYPTO EQUITY"
]
NON_LEVERAGED_TICKERS = [c for c in prices_df.columns if c not in LEVERAGED_TICKERS and c != "WORLD"]

# 2. Extract technical features
panels = []
bmk_series = prices_df["WORLD"] if "WORLD" in prices_df.columns else prices_df.iloc[:, 0]

for tkr in NON_LEVERAGED_TICKERS:
    p = prices_df[tkr].dropna()
    common_idx = p.index.intersection(bmk_series.index)
    p = p.loc[common_idx]
    bmk = bmk_series.loc[common_idx]

    df = pd.DataFrame(index=common_idx)
    df["ticker"] = tkr
    df["close"] = p
    df["ret_1d"] = p.pct_change(1)
    df["ret_5d"] = p.pct_change(5)
    df["ret_21d"] = p.pct_change(21)
    df["ret_63d"] = p.pct_change(63)

    sma20 = p.rolling(20).mean()
    sma50 = p.rolling(50).mean()
    df["dist_sma20"] = (p / (sma20 + 1e-9)) - 1.0
    df["dist_sma50"] = (p / (sma50 + 1e-9)) - 1.0

    delta = p.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / (down + 1e-9)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    bmk_ret_5d = bmk.pct_change(5)
    bmk_ret_21d = bmk.pct_change(21)
    df["rel_roc5_vs_world"] = df["ret_5d"] - bmk_ret_5d
    df["rel_roc21_vs_world"] = df["ret_21d"] - bmk_ret_21d

    vol21 = df["ret_1d"].rolling(21).std() * np.sqrt(252)
    df["vol_21d"] = vol21

    boll_mid = sma20
    boll_std = p.rolling(20).std()
    df["boll_bandwidth"] = (4 * boll_std) / (boll_mid + 1e-9)

    # 5-day forward return & target
    fwd_ret_5d = p.pct_change(5).shift(-5)
    fwd_bmk_5d = bmk.pct_change(5).shift(-5)
    df["target_fwd_alpha_5d"] = fwd_ret_5d - fwd_bmk_5d
    df["target_binary_outperform"] = (df["target_fwd_alpha_5d"] > 0).astype(int)

    panels.append(df)

full_panel = pd.concat(panels, axis=0)
clean_panel = full_panel.dropna().sort_index()

feature_cols = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d",
    "dist_sma20", "dist_sma50", "rsi_14",
    "rel_roc5_vs_world", "rel_roc21_vs_world",
    "vol_21d", "boll_bandwidth"
]

unique_dates = np.array(sorted(clean_panel.index.unique()))
n_dates = len(unique_dates)
n_splits = 4
embargo_days = 5

min_train_dates = int(n_dates * 0.45)
test_window = int((n_dates - min_train_dates) / n_splits)

cv_splits = []
for fold_i in range(n_splits):
    train_end_idx = min_train_dates + (fold_i * test_window)
    test_start_idx = train_end_idx + embargo_days
    test_end_idx = min(test_start_idx + test_window, n_dates)
    cv_splits.append((unique_dates[:train_end_idx], unique_dates[test_start_idx:test_end_idx]))

# Fit Random Forest and evaluate Baseline
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, random_state=42, n_jobs=-1)

oos_predictions = []
for fold_i, (tr_dates, te_dates) in enumerate(cv_splits, 1):
    tr_mask = clean_panel.index.isin(tr_dates)
    te_mask = clean_panel.index.isin(te_dates)

    X_train, y_train = clean_panel.loc[tr_mask, feature_cols], clean_panel.loc[tr_mask, "target_binary_outperform"]
    X_test, y_test = clean_panel.loc[te_mask, feature_cols], clean_panel.loc[te_mask, "target_binary_outperform"]
    test_meta = clean_panel.loc[te_mask, ["ticker", "target_fwd_alpha_5d"]].copy()

    # Rule-Based Baseline
    baseline_preds = []
    for dt, row in test_meta.iterrows():
        tkr = row["ticker"]
        hist_slice = prices_df.loc[:dt, tkr]
        bmk_slice = prices_df.loc[:dt, "WORLD"] if "WORLD" in prices_df.columns else None
        sig = evaluate_etf_technical(hist_slice, tkr, bmk_slice)
        prob = (sig.composite_score + 1.0) / 2.0
        baseline_preds.append(prob)

    fold_df = test_meta.copy()
    fold_df["y_true"] = y_test
    fold_df["prob_baseline"] = np.array(baseline_preds)

    rf_model.fit(X_train, y_train)
    fold_df["prob_rf"] = rf_model.predict_proba(X_test)[:, 1]

    # Hybrid Model: 50% Rulebook + 50% Random Forest
    fold_df["prob_hybrid"] = 0.50 * fold_df["prob_baseline"] + 0.50 * fold_df["prob_rf"]

    oos_predictions.append(fold_df)

all_oos = pd.concat(oos_predictions, axis=0)

# Simulate 5-day rebalanced portfolio
rebal_dates = sorted(all_oos.index.unique())[::5]
models_to_test = ["Baseline (Rule-Based)", "Random Forest", "Hybrid (50% Rule + 50% RF)"]

equity_curves = {m: [10000.0] for m in models_to_test}
equity_curves["Benchmark (WORLD)"] = [10000.0]
period_returns = {m: [] for m in models_to_test}
period_returns["Benchmark (WORLD)"] = []

for i in range(len(rebal_dates) - 1):
    dt = rebal_dates[i]
    next_dt = rebal_dates[i+1]

    bmk_start = prices_df.loc[dt, "WORLD"]
    bmk_end = prices_df.loc[next_dt, "WORLD"]
    bmk_ret = (bmk_end / bmk_start) - 1.0
    equity_curves["Benchmark (WORLD)"].append(equity_curves["Benchmark (WORLD)"][-1] * (1.0 + bmk_ret))
    period_returns["Benchmark (WORLD)"].append(bmk_ret)

    day_preds = all_oos.loc[dt]
    if isinstance(day_preds, pd.Series):
        day_preds = day_preds.to_frame().T

    for m in models_to_test:
        if m == "Baseline (Rule-Based)":
            pcol = "prob_baseline"
        elif m == "Random Forest":
            pcol = "prob_rf"
        else:
            pcol = "prob_hybrid"

        top5 = day_preds.sort_values(by=pcol, ascending=False)["ticker"].unique()[:5]

        asset_rets = []
        for tkr in top5:
            p0 = prices_df.loc[dt, tkr]
            p1 = prices_df.loc[next_dt, tkr]
            asset_rets.append((p1 / p0) - 1.0)

        port_ret = (np.mean(asset_rets) * 0.90) + (0.10 * 0.0003)
        equity_curves[m].append(equity_curves[m][-1] * (1.0 + port_ret))
        period_returns[m].append(port_ret)

results = []
for col in equity_curves.keys():
    cum_ret = (equity_curves[col][-1] / equity_curves[col][0]) - 1.0
    rets = pd.Series(period_returns[col])
    ann_ret = (1.0 + cum_ret) ** (52 / len(rets)) - 1.0
    ann_vol = rets.std() * np.sqrt(52)
    sharpe = (ann_ret - 0.025) / (ann_vol + 1e-9)
    peak = pd.Series(equity_curves[col]).cummax()
    dd = (pd.Series(equity_curves[col]) / peak - 1.0).min()
    win_rate = (rets > 0).mean() * 100

    results.append({
        "Model": col,
        "Total Return (%)": round(cum_ret * 100, 2),
        "Ann Return (%)": round(ann_ret * 100, 2),
        "Ann Vol (%)": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Max DD (%)": round(dd * 100, 2),
        "Win Rate (%)": round(win_rate, 1)
    })

res_df = pd.DataFrame(results)
print("\n=== EMPIRICAL COMPARISON: HYBRID vs PURE RF vs BASELINE ===")
print(res_df.to_string(index=False))
