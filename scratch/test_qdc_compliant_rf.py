import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.append('/Users/nguyenhangthu/Desktop/QTEM/GBAC/team-29-ai-agent')
prices_path = '/Users/nguyenhangthu/Desktop/QTEM/GBAC/team-29-ai-agent/data/etf_prices_eur.parquet'
prices_df = pd.read_parquet(prices_path)
if not isinstance(prices_df.index, pd.DatetimeIndex):
    prices_df.index = pd.to_datetime(prices_df.index)

LEVERAGED_TICKERS = [
    "NASDAQ 3X", "SOX 3X", "S&P 500 3X", "STOXX SHORT", "S&P 500 SHORT",
    "COPPER LVG", "PALLADIUM LVG", "BITCOIN", "CRYPTO EQUITY", "EURO STOXX 2X"
]
NON_LEVERAGED = [c for c in prices_df.columns if c not in LEVERAGED_TICKERS and c != "WORLD"]

from src.pipeline.etf_universe import ETF_UNIVERSE
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

bmk_series = prices_df["WORLD"]

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def extract_features(prices: pd.Series, ticker: str, bmk: pd.Series, meta: dict, forward_h: int = 5) -> pd.DataFrame:
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

panels = [extract_features(prices_df[c], c, bmk_series, meta_dict.get(c, {})) for c in prices_df.columns]
clean_panel = pd.concat(panels, axis=0).dropna().copy()
clean_panel["theme_code"] = clean_panel["theme"].astype("category").cat.codes

feature_cols = [
    "dist_sma20", "dist_sma50", "ema20_sma50_ratio",
    "roc_5d", "roc_10d", "roc_21d", "roc_63d",
    "rsi_14", "macd_hist_norm",
    "realized_vol_20d", "realized_vol_63d",
    "boll_bandwidth", "boll_pct_b",
    "rel_roc21_vs_world", "rel_roc63_vs_world",
    "is_leveraged", "theme_code"
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

rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=25, random_state=42, n_jobs=-1)

oos_records = []
for fold_i, (tr_dates, te_dates) in enumerate(cv_splits, 1):
    tr_mask = clean_panel.index.isin(tr_dates)
    te_mask = clean_panel.index.isin(te_dates)
    X_train, y_train = clean_panel.loc[tr_mask, feature_cols], clean_panel.loc[tr_mask, "target_binary_outperform"]
    X_test, y_test = clean_panel.loc[te_mask, feature_cols], clean_panel.loc[te_mask, "target_binary_outperform"]
    test_meta = clean_panel.loc[te_mask, ["ticker", "target_fwd_alpha_5d"]].copy()
    rf.fit(X_train, y_train)
    test_meta["prob_rf"] = rf.predict_proba(X_test)[:, 1]
    oos_records.append(test_meta)

all_oos = pd.concat(oos_records, axis=0)
rebal_dates = sorted(all_oos.index.unique())[::5]

# Compare 2 cases:
# Case 1: Notebook Unconstrained (includes leveraged up to 54%)
# Case 2: QDC Compliant (Non-leveraged only in Top 5, exactly matching competition rules)

eq_unconstrained = [10000.0]
eq_compliant = [10000.0]
eq_bmk = [10000.0]

for i in range(len(rebal_dates) - 1):
    dt = rebal_dates[i]
    next_dt = rebal_dates[i+1]

    bmk_ret = (prices_df.loc[next_dt, "WORLD"] / prices_df.loc[dt, "WORLD"]) - 1.0
    eq_bmk.append(eq_bmk[-1] * (1.0 + bmk_ret))

    day_preds = all_oos.loc[dt]
    if isinstance(day_preds, pd.Series):
        day_preds = day_preds.to_frame().T

    # Case 1: Unconstrained top 5 (as in notebook)
    top5_unconstrained = day_preds.sort_values(by="prob_rf", ascending=False)["ticker"].unique()[:5]
    rets_unc = [(prices_df.loc[next_dt, t] / prices_df.loc[dt, t]) - 1.0 for t in top5_unconstrained]
    p_ret_unc = (np.mean(rets_unc) * 0.90) + (0.10 * 0.0003)
    eq_unconstrained.append(eq_unconstrained[-1] * (1.0 + p_ret_unc))

    # Case 2: QDC Compliant (Strictly non-leveraged assets only)
    day_non_lev = day_preds[day_preds["ticker"].isin(NON_LEVERAGED)]
    top5_compliant = day_non_lev.sort_values(by="prob_rf", ascending=False)["ticker"].unique()[:5]
    rets_comp = [(prices_df.loc[next_dt, t] / prices_df.loc[dt, t]) - 1.0 for t in top5_compliant]
    p_ret_comp = (np.mean(rets_comp) * 0.90) + (0.10 * 0.0003)
    eq_compliant.append(eq_compliant[-1] * (1.0 + p_ret_comp))

def get_stats(eq_curve, name):
    cum = (eq_curve[-1] / eq_curve[0]) - 1.0
    s = pd.Series(eq_curve).pct_change().dropna()
    ann_ret = (1.0 + cum) ** (52 / len(s)) - 1.0
    ann_vol = s.std() * np.sqrt(52)
    sharpe = (ann_ret - 0.025) / (ann_vol + 1e-9)
    peak = pd.Series(eq_curve).cummax()
    mdd = (pd.Series(eq_curve) / peak - 1.0).min()
    return {
        "Strategy": name,
        "Total Return (%)": round(cum * 100, 2),
        "Ann Return (%)": round(ann_ret * 100, 2),
        "Ann Vol (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(mdd * 100, 2)
    }

res = [
    get_stats(eq_unconstrained, "RF (Notebook Unconstrained - with Leveraged/Crypto)"),
    get_stats(eq_compliant, "RF (QDC Compliant - Non-Leveraged Only)"),
    get_stats(eq_bmk, "Benchmark (WORLD)")
]
print("\n" + pd.DataFrame(res).to_string(index=False))
