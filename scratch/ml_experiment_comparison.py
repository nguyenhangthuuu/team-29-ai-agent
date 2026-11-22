"""
Benchmark Script: Evaluating ML Models (RF vs XGBoost vs CatBoost vs LightGBM)
and Train/Test Ratios on 50 UCITS ETF Universe (2023-2026).
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

from src.agents.technical import (
    PRICES_FILE_PATH,
    ETF_META_LOOKUP,
    FEATURE_COLS,
    extract_features_single_series
)

def run_ml_benchmark():
    prices_df = pd.read_parquet(PRICES_FILE_PATH)
    if not isinstance(prices_df.index, pd.DatetimeIndex):
        prices_df.index = pd.to_datetime(prices_df.index)

    bmk_series = prices_df["WORLD"] if "WORLD" in prices_df.columns else prices_df.iloc[:, 0]

    # Build dataset
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
            panels.append(idf)

    clean_panel = pd.concat(panels, axis=0).dropna().sort_index().copy()
    themes = sorted(clean_panel["theme"].unique())
    theme_to_code = {t: i for i, t in enumerate(themes)}
    clean_panel["theme_code"] = clean_panel["theme"].map(theme_to_code)

    unique_dates = np.array(sorted(clean_panel.index.unique()))
    n_dates = len(unique_dates)
    print(f"Total trading days: {n_dates} (from {unique_dates[0].date()} to {unique_dates[-1].date()})")
    print(f"Total panel rows: {len(clean_panel)}")

    # Models to benchmark
    seed = 42
    model_factories = {
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=25, max_features="sqrt", random_state=seed, n_jobs=-1
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=seed, eval_metric="logloss"
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=200, depth=5, learning_rate=0.03, l2_leaf_reg=4.0, random_seed=seed, verbose=0
        ),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1
        )
    }

    # Experiment 1: Effect of Train Ratio (45% vs 70% vs 85%) on Test Performance
    # We test on the most recent 15% window (the last ~100 trading sessions in 2026, where the market shifted)
    test_size_dates = int(n_dates * 0.15)
    test_dates = unique_dates[-test_size_dates:]
    test_mask = clean_panel.index.isin(test_dates)
    X_test = clean_panel.loc[test_mask, FEATURE_COLS]
    y_test = clean_panel.loc[test_mask, "target_binary_outperform"]
    test_meta = clean_panel.loc[test_mask, ["ticker", "target_fwd_alpha_5d"]].copy()

    print("\n" + "="*80)
    print(f"EXPERIMENT 1: EFFECT OF TRAIN RATIO (Evaluation Window: {test_dates[0].date()} to {test_dates[-1].date()}, {len(test_dates)} days)")
    print("="*80)

    train_ratios = [0.45, 0.70, 0.85]
    ratio_results = []

    for ratio in train_ratios:
        num_tr_dates = int(n_dates * ratio)
        # End training at least 5 days prior to test_start (embargo)
        tr_end_idx = n_dates - test_size_dates - 5
        tr_start_idx = max(0, tr_end_idx - num_tr_dates)
        tr_dates = unique_dates[tr_start_idx:tr_end_idx]
        tr_mask = clean_panel.index.isin(tr_dates)

        X_train = clean_panel.loc[tr_mask, FEATURE_COLS]
        y_train = clean_panel.loc[tr_mask, "target_binary_outperform"]

        print(f"\n--- Train Ratio: {ratio*100:.0f}% ({len(tr_dates)} days: {tr_dates[0].date()} to {tr_dates[-1].date()}, {len(X_train)} samples) ---")

        for m_name, m_func in model_factories.items():
            model = m_func()
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs > 0.50).astype(int)

            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            loss = log_loss(y_test, probs)

            # Top-5 simulated alpha over this test period
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

            ratio_results.append({
                "Train Ratio": f"{int(ratio*100)}%",
                "Model": m_name,
                "Train Samples": len(X_train),
                "OOS Accuracy (%)": round(acc * 100, 2),
                "OOS AUC": round(auc, 3),
                "OOS Log-Loss": round(loss, 3),
                "Top-5 5d Alpha (%)": round(mean_alpha, 3)
            })

    res_df = pd.DataFrame(ratio_results)
    print("\n" + "="*80)
    print("RESULTS SUMMARY: TRAIN RATIO & MODEL BENCHMARK")
    print("="*80)
    print(res_df.to_string(index=False))

    # Feature Importance Analysis for Best Train Ratio (85%)
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE AUDIT (Train Ratio 85%)")
    print("="*80)
    tr_dates = unique_dates[:n_dates - test_size_dates - 5]
    tr_mask = clean_panel.index.isin(tr_dates)
    X_train = clean_panel.loc[tr_mask, FEATURE_COLS]
    y_train = clean_panel.loc[tr_mask, "target_binary_outperform"]

    for m_name, m_func in model_factories.items():
        model = m_func()
        model.fit(X_train, y_train)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
            top3 = feat_imp.head(3).to_dict()
            top3_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in top3.items()])
            print(f"{m_name:15s} Top 3 Features -> {top3_str}")

if __name__ == "__main__":
    run_ml_benchmark()
