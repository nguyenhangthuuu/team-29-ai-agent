# src/pipeline/data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_and_normalize_data(etf_list, period="3y") -> pd.DataFrame:
    """Downloads close prices and converts USD prices into EUR."""
    symbols = [etf.yf_symbol for etf in etf_list]
    data = yf.download(symbols + ["EURUSD=X"], period=period)["Adj Close"]
    
    eur_usd = data["EURUSD=X"] # 1 EUR = X USD
    eur_prices = pd.DataFrame(index=data.index)

    for etf in etf_list:
        raw_price = data[etf.yf_symbol]
        if etf.currency == "USD":
            # Convert USD price to EUR: Price_EUR = Price_USD / (EURUSD rate)
            eur_prices[etf.ticker_internal] = raw_price / eur_usd
        else:
            eur_prices[etf.ticker_internal] = raw_price
            
    return eur_prices.dropna(how="all").ffill()

def compute_quant_metrics(price_df: pd.DataFrame, window=60):
    """Computes daily returns, rolling annualized volatility, rolling max drawdown, and correlations."""
    daily_returns = price_df.pct_change().dropna()
    
    # Annualized Volatility: rolling std * sqrt(252)
    ann_vol = daily_returns.rolling(window).std() * np.sqrt(252)
    
    # Rolling Maximum Drawdown
    rolling_peak = price_df.cummax()
    drawdowns = (price_df - rolling_peak) / rolling_peak
    max_drawdown = drawdowns.rolling(window).min()
    
    # Correlation Matrix (over recent 60 trading days)
    corr_matrix = daily_returns.tail(window).corr()
    
    return {
        "prices": price_df,
        "returns": daily_returns,
        "volatility": ann_vol,
        "drawdowns": drawdowns,
        "correlations": corr_matrix
    }