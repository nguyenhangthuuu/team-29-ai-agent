import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from src.pipeline.etf_universe import ETF_UNIVERSE

def fetch_and_normalize_prices(period="3y") -> pd.DataFrame:
    """Downloads prices from Yahoo Finance, converts USD prices to EUR, and forward-fills gaps."""
    symbols = list(set([etf.yf_symbol for etf in ETF_UNIVERSE]))
    print(f"Downloading historical data for {len(symbols)} instruments + EURUSD FX...")
    
    # Download symbols + FX exchange rate
    raw = yf.download(symbols + ["EURUSD=X"], period=period, auto_adjust=False)["Adj Close"]
    
    eur_usd = raw["EURUSD=X"]
    prices_eur = pd.DataFrame(index=raw.index)

    for etf in ETF_UNIVERSE:
        if etf.yf_symbol in raw.columns:
            s = raw[etf.yf_symbol]
            if etf.currency == "USD":
                # Convert USD into EUR (Price_EUR = Price_USD / FX_rate)
                prices_eur[etf.ticker] = s / eur_usd
            else:
                prices_eur[etf.ticker] = s
        else:
            print(f"Warning: {etf.ticker} ({etf.yf_symbol}) missing in download!")

    # Forward fill missing values and drop initial empty rows
    prices_eur = prices_eur.ffill().dropna(how="all")
    return prices_eur

def compute_quant_features(prices: pd.DataFrame, window=60):
    """Computes daily returns, rolling annualized volatility, rolling max drawdown, and correlations."""
    daily_returns = prices.pct_change().dropna()
    
    # Annualized Volatility (daily std * sqrt(252))
    annualized_vol = daily_returns.rolling(window).std() * np.sqrt(252)
    
    # Maximum Drawdown calculation
    cumulative_max = prices.cummax()
    drawdowns = (prices - cumulative_max) / cumulative_max
    
    # 60-day Correlation Matrix
    recent_corr = daily_returns.tail(window).corr()
    
    return {
        "prices": prices,
        "returns": daily_returns,
        "volatility": annualized_vol,
        "drawdowns": drawdowns,
        "correlations": recent_corr
    }

def run_pipeline():
    """Main execution entry point for Task 1."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    prices = fetch_and_normalize_prices(period="3y")
    quant = compute_quant_features(prices)
    
    # Save processed files locally
    prices.to_parquet(data_dir / "etf_prices_eur.parquet")
    quant["returns"].to_parquet(data_dir / "etf_returns.parquet")
    quant["volatility"].to_parquet(data_dir / "etf_volatility.parquet")
    quant["correlations"].to_csv(data_dir / "etf_correlation_matrix.csv")
    
    print("\n✅ Task 1 Pipeline successfully executed!")
    print(f"Data range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Total ETFs loaded: {prices.shape[1]}")
    print("Files saved in ./data/ directory.")

if __name__ == "__main__":
    run_pipeline()