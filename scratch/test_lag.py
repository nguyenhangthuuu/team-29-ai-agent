import os
import sys
import numpy as np
import pandas as pd

sys.path.append('/Users/nguyenhangthu/Desktop/QTEM/GBAC/team-29-ai-agent')
prices_path = '/Users/nguyenhangthu/Desktop/QTEM/GBAC/team-29-ai-agent/data/etf_prices_eur.parquet'
prices_df = pd.read_parquet(prices_path)
if not isinstance(prices_df.index, pd.DatetimeIndex):
    prices_df.index = pd.to_datetime(prices_df.index)

from scratch.diagnose_rf import all_oos, LEVERAGED_TICKERS
NON_LEVERAGED = [c for c in prices_df.columns if c not in LEVERAGED_TICKERS and c != "WORLD"]

eq_sameday = [10000.0]
eq_lag1d = [10000.0]

valid_rebal_dates = sorted(all_oos.index.unique())[::5]

for i in range(len(valid_rebal_dates) - 1):
    dt = valid_rebal_dates[i]
    next_dt = valid_rebal_dates[i+1]

    idx_dt = prices_df.index.get_loc(dt)
    idx_next = prices_df.index.get_loc(next_dt)

    day_preds = all_oos.loc[dt]
    if isinstance(day_preds, pd.Series):
        day_preds = day_preds.to_frame().T

    day_non_lev = day_preds[day_preds["ticker"].isin(NON_LEVERAGED)]
    top5 = day_non_lev.sort_values(by="prob_rf", ascending=False)["ticker"].unique()[:5]

    # Same day close
    rets_same = [(prices_df.loc[next_dt, t] / prices_df.loc[dt, t]) - 1.0 for t in top5]
    p_same = (np.mean(rets_same) * 0.90) + (0.10 * 0.0003)
    eq_sameday.append(eq_sameday[-1] * (1.0 + p_same))

    # 1-day lag execution: buy at dt+1, sell at next_dt+1
    if idx_dt + 1 < len(prices_df) and idx_next + 1 < len(prices_df):
        dt_lag = prices_df.index[idx_dt + 1]
        next_dt_lag = prices_df.index[idx_next + 1]
        rets_lag = [(prices_df.loc[next_dt_lag, t] / prices_df.loc[dt_lag, t]) - 1.0 for t in top5]
        p_lag = (np.mean(rets_lag) * 0.90) + (0.10 * 0.0003)
        eq_lag1d.append(eq_lag1d[-1] * (1.0 + p_lag))

def calc_m(curve, name):
    cum = (curve[-1] / curve[0]) - 1.0
    s = pd.Series(curve).pct_change().dropna()
    ann_ret = (1.0 + cum) ** (52 / len(s)) - 1.0
    ann_vol = s.std() * np.sqrt(52)
    sharpe = (ann_ret - 0.025) / (ann_vol + 1e-9)
    peak = pd.Series(curve).cummax()
    mdd = (pd.Series(curve) / peak - 1.0).min()
    return {
        "Scenario": name,
        "Total Return (%)": round(cum * 100, 2),
        "Ann Return (%)": round(ann_ret * 100, 2),
        "Ann Vol (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(mdd * 100, 2)
    }

print("\n=== EXECUTION LAG CHECK (LOOKAHEAD AUDIT) ===")
print(pd.DataFrame([
    calc_m(eq_sameday, "Same-Day Close (Zero-lag assumption)"),
    calc_m(eq_lag1d, "1-Day Execution Lag (Realistic trade execution)")
]).to_string(index=False))
